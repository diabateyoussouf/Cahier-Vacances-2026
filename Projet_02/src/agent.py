import os
import sys
from pathlib import Path
from typing import Annotated, TypedDict
from dotenv import load_dotenv

from langchain_core.messages import BaseMessage, SystemMessage
from langchain_mistralai import ChatMistralAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

SRC_DIR = Path(__file__).resolve().parent
PROJET_DIR = SRC_DIR.parent
ROOT_DIR = PROJET_DIR.parent

env_path = ROOT_DIR / ".env"
if not env_path.exists():
    env_path = PROJET_DIR / ".env"
load_dotenv(dotenv_path=env_path)

if str(PROJET_DIR) not in sys.path:
    sys.path.append(str(PROJET_DIR))

from src.tools import ALL_TOOLS

# --- INSTRUCTIONS MISES À JOUR POUR GÉRER 2030 ET FUTURS TOURNOIS ---
SYSTEM_PROMPT = """Tu es un assistant expert des Coupes du Monde de football (2026, 2030, etc.).
Tu as accès à deux outils principaux :
1. `predict_world_cup_match` pour prédire l'issue d'un affrontement direct entre deux nations.
2. `search_world_cup_knowledge` pour obtenir des faits sur l'organisation des éditions (hôtes, formats, règles).

Consignes de réponse :
- Si l'utilisateur demande "Qui gagnera la Coupe du Monde 2030 ?" (ou une autre édition future), NE REFUSE PAS de répondre.
- Explique que ton modèle prédictif s'appuie sur la forme récente et les historiques d'équipes, puis utilise l'outil `predict_world_cup_match` pour simuler des confrontations clés entre les grands favoris (ex: France, Brazil, Argentina, Spain, Morocco, Germany).
- Synthétise ensuite le favori principal qui ressort de tes simulations.
- Utilise `search_world_cup_knowledge` pour rappeler le contexte ou les pays hôtes (ex: 2030 au Maroc, Espagne, Portugal + Amérique du Sud).
- Toujours convertir les noms de pays en anglais dans l'outil de prédiction ('Brazil', 'France', 'Morocco', 'Spain', 'Argentina').
- Réponds en français de façon synthétique et structurée.
"""


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def get_agent_executor(api_key: str = None, model_name: str = "mistral-small-latest"):
    key = api_key or os.getenv("MISTRAL_API_KEY")

    if not key:
        raise ValueError("Aucune clé API Mistral AI spécifiée.")

    llm = ChatMistralAI(model=model_name, api_key=key, temperature=0)
    llm_with_tools = llm.bind_tools(ALL_TOOLS)

    def call_model(state: AgentState):
        messages = state["messages"]
        if not messages or not isinstance(messages[0], SystemMessage):
            full_messages = [SystemMessage(content=SYSTEM_PROMPT)] + list(messages)
        else:
            full_messages = messages

        response = llm_with_tools.invoke(full_messages)
        return {"messages": [response]}

    tool_node = ToolNode(ALL_TOOLS)

    workflow = StateGraph(AgentState)
    workflow.add_node("agent", call_model)
    workflow.add_node("tools", tool_node)
    workflow.set_entry_point("agent")

    workflow.add_conditional_edges("agent", tools_condition)
    workflow.add_edge("tools", "agent")

    memory = MemorySaver()
    return workflow.compile(checkpointer=memory)