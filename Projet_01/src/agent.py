import os
from pathlib import Path
from typing import TypedDict, List, Annotated
from dotenv import load_dotenv

env_path = Path(__file__).resolve().parents[2] / ".env"
load_dotenv(dotenv_path=env_path)

from langchain_core.tools import tool
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_mistralai import ChatMistralAI
from langchain_core.messages import SystemMessage, BaseMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver


class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]


embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = Chroma(persist_directory="vectorstore", embedding_function=embeddings)

if len(vectorstore.get()["ids"]) == 0:
    kb_documents = [
        Document(
            page_content="Politique d'annulation et de modification : Les billets en classe Business et First sont annulables et remboursables à 100% sans frais jusqu'à 24h avant le départ. Les billets Economy sont modifiables moyennant 50€ de frais par trajet, mais ne sont pas remboursables en cas d'annulation.",
            metadata={"source": "politique_tarifaire.pdf"}
        ),
        Document(
            page_content="Franchise Bagages : Classe Economy = 1 bagage en soute de 23kg maximum inclus. Classe Business = 2 bagages de 32kg chacun. Classe First = 3 bagages de 32kg. Tout bagage supplémentaire ou hors format est facturé 70€ forfaitairement.",
            metadata={"source": "regles_bagages.pdf"}
        ),
        Document(
            page_content="Réclamations et Retards : En vertu de la réglementation, en cas de retard de vol supérieur à 3 heures à l'arrivée non causé par la météo, une indemnisation forfaitaire de 250€ (court-courrier) à 600€ (long-courrier) est accordée au passager.",
            metadata={"source": "droits_passagers.pdf"}
        )
    ]
    vectorstore.add_documents(kb_documents)

retriever = vectorstore.as_retriever(search_kwargs={"k": 2})


@tool
def search_documents(query: str) -> str:
    """Recherche des règles d'entreprise, politiques d'annulation et règles bagages dans la base RAG vectorielle."""
    docs = retriever.invoke(query)
    if not docs:
        return "Aucun document correspondant trouvé."
    return "\n\n".join([f"[Source: {doc.metadata.get('source', 'Inconnue')}]\n{doc.page_content}" for doc in docs])


@tool
def execute_sql_query(query: str) -> str:
    """Exécute une requête SQL SELECT sur la base SQLite (passengers, flights, bookings)."""
    try:
        try:
            from .database import get_connection, execute_query
        except ImportError:
            from database import get_connection, execute_query

        conn = get_connection()
        df = execute_query(conn, query)
        if df.empty:
            return "La requête s'est exécutée avec succès mais n'a retourné aucun résultat."
        return df.to_string(index=False)
    except Exception as e:
        return f"Erreur SQL : {str(e)}. Veuillez corriger la syntaxe et réessayer."


tools = [search_documents, execute_sql_query]


def build_agent_graph(api_key: str):
    """Construit le graphe d'agent avec la clé API fournie dynamiquement."""
    llm = ChatMistralAI(
        model="mistral-small-latest",
        api_key=api_key,
        temperature=0,
        max_retries=5
    )
    llm_with_tools = llm.bind_tools(tools)

    def agent_node(state: AgentState):
        system_prompt = SystemMessage(
            content="""Tu es un assistant IA de niveau exécutif pour une compagnie aérienne.
Tu as accès à deux bases de données via des outils :
1. `execute_sql_query` : Pour les données relationnelles (Passagers, Vols, Réservations).
2. `search_documents` : Pour la documentation d'entreprise (Politiques d'annulation, bagages, réclamations).

CONSIGNES POUR LE SQL :
- Table `passengers` : id, first_name, last_name, email, nationality
- Table `flights` : id, flight_number, origin, destination, departure_time, arrival_time, aircraft, capacity, price_eur
- Table `bookings` : id, passenger_id, flight_id, booking_date, seat_class, seat_number, status ('confirmed', 'cancelled', 'pending')
- Pour trouver les passagers d'un vol, fais une jointure INNER JOIN entre `passengers`, `bookings` et `flights`.
- Utilise la casse exacte pour les numéros de vol (ex: `flight_number = 'AF5501'`).
"""
        )
        messages = [system_prompt] + state["messages"]
        response = llm_with_tools.invoke(messages)
        return {"messages": [response]}

    workflow = StateGraph(AgentState)
    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", ToolNode(tools))
    workflow.set_entry_point("agent")
    workflow.add_conditional_edges("agent", tools_condition, {"tools": "tools", END: END})
    workflow.add_edge("tools", "agent")

    memory = MemorySaver()
    return workflow.compile(checkpointer=memory)