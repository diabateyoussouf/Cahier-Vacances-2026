import os
import sys
from pathlib import Path
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

# --- RESOLUTION DYNAMIQUE DES CHEMINS & ENV ---
APP_DIR = Path(__file__).resolve().parent
ROOT_DIR = APP_DIR.parent

env_path = ROOT_DIR / ".env"
if not env_path.exists():
    env_path = APP_DIR / ".env"
load_dotenv(dotenv_path=env_path)

if str(APP_DIR) not in sys.path:
    sys.path.append(str(APP_DIR))

from src.ml import load_model_artifacts, predict_match
from src.agent import get_agent_executor
from utils import get_current_form

# --- MAPPING DRAPEAUX ---
FLAGS = {
    "France": "🇫🇷", "Brazil": "🇧🇷", "Argentina": "🇦🇷", "Spain": "🇪🇸",
    "Germany": "🇩🇪", "England": "🏴󠁧󠁢󠁥󠁮󠁧󠁿", "Morocco": "🇲🇦", "Portugal": "🇵🇹",
    "Belgium": "🇧🇪", "Netherlands": "🇳🇱", "Uruguay": "🇺🇾", "Croatia": "🇭🇷",
    "United States": "🇺🇸", "Mexico": "🇲🇽", "Canada": "🇨🇦", "Japan": "🇯🇵",
    "Senegal": "🇸🇳", "Switzerland": "🇨🇭", "Norway": "🇳🇴", "Algeria": "🇩🇿"
}

def get_flag(team_name: str) -> str:
    return FLAGS.get(team_name, "⚽")

# --- CONFIGURATION PAGE STREAMLIT ---
st.set_page_config(
    page_title="World Cup AI Center — ML & Agentic RAG",
    page_icon="🏆",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CSS CUSTOM SYSTEM (DESIGN PREMIUM) ---
st.markdown("""
    <style>
    /* Gradient Background Top Bar */
    .stApp > header {
        background: transparent;
    }
    
    /* Global Cards Styling */
    .metric-card {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
        backdrop-filter: blur(10px);
    }
    
    /* Winner Card Highlight */
    .winner-card {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.15) 0%, rgba(5, 150, 105, 0.05) 100%);
        border: 1px solid rgba(16, 185, 129, 0.4);
        border-radius: 16px;
        padding: 24px;
        text-align: center;
        margin-top: 15px;
        margin-bottom: 25px;
    }
    
    .winner-title {
        font-size: 1.1rem;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        color: #10B981;
        font-weight: 700;
        margin-bottom: 8px;
    }
    
    .winner-name {
        font-size: 2.2rem;
        font-weight: 800;
        margin-bottom: 5px;
    }
    
    /* Custom Badges */
    .badge-tag {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.82rem;
        font-weight: 600;
        background: rgba(255, 255, 255, 0.08);
        border: 1px solid rgba(255, 255, 255, 0.15);
        margin-right: 6px;
    }
    
    /* VS Divider */
    .vs-circle {
        display: flex;
        align-items: center;
        justify-content: center;
        width: 50px;
        height: 50px;
        border-radius: 50%;
        background: linear-gradient(135deg, #3B82F6, #1D4ED8);
        color: white;
        font-weight: 900;
        font-size: 1.1rem;
        margin: auto;
        box-shadow: 0 0 15px rgba(59, 130, 246, 0.5);
    }
    </style>
""", unsafe_allow_html=True)

# --- CHARGEMENT DU MODÈLE ET DONNÉES EN CACHE ---
@st.cache_resource
def get_ml_data():
    return load_model_artifacts()

try:
    model, scaler, df_processed = get_ml_data()
    teams_list = sorted(
        list(
            set(df_processed["home_team"].unique()).union(
                set(df_processed["away_team"].unique())
            )
        )
    )
except Exception as e:
    st.error(f"⚠️ Erreur d'initialisation des artefacts ML : {e}")
    st.stop()

# --- HEADER PRINCIPAL ---
st.title("🏆 World Cup AI Analytics & Assistant")
st.markdown("""
<div style='margin-bottom: 25px;'>
    <span class="badge-tag">🤖 LangGraph ReAct</span>
    <span class="badge-tag">⚡ Mistral AI LLM</span>
    <span class="badge-tag">🌲 Random Forest Classifier</span>
    <span class="badge-tag">⚽ Édit. 2026 & Projections 2030</span>
</div>
""", unsafe_allow_html=True)

# --- SIDEBAR CONFIGURATION ---
st.sidebar.image("https://img.freepik.com/free-vector/soccer-stadium-vector-illustration_1284-22432.jpg", use_container_width=True)
st.sidebar.title("⚙️ Panneau de Contrôle")

api_key_input = st.sidebar.text_input(
    "Clé API Mistral AI",
    value=os.getenv("MISTRAL_API_KEY", ""),
    type="password",
    help="Clé récupérée depuis console.mistral.ai",
)

model_choice = st.sidebar.selectbox(
    "Moteur LLM Mistral",
    options=["mistral-small-latest", "mistral-medium-latest", "mistral-large-latest"],
    index=0,
)

st.sidebar.divider()
st.sidebar.markdown("### 📌 Repères Compétition")
st.sidebar.markdown("""
- **2026** : 🇪🇸 **Espagne** (Vainqueur 1-0 vs 🇦🇷 Argentine)
- **2030** : 🇲🇦 Maroc, 🇪🇸 Espagne, 🇵🇹 Portugal +(🇦🇷🇺🇾🇵🇾)
""")

# --- NAVIGATION ONGLETS ---
tab_sim, tab_chat, tab_arch = st.tabs([
    "⚔️ Simulateur Direct (ML)",
    "💬 Assistant Intelligent (Agent ReAct)",
    "📐 Architecture & Documentation"
])

# ==========================================
# TAB 1 : SIMULATEUR ML DIRECT
# ==========================================
with tab_sim:
    st.markdown("### 🔮 Simulation d'affrontement international")
    st.caption("Sélectionnez deux nations pour comparer leurs métriques récentes et calculer les probabilités de victoire sur terrain neutre.")

    # Raccourcis de sélection rapide
    st.write("**Affiches populaires :**")
    f_col1, f_col2, f_col3, f_col4 = st.columns(4)
    
    preset_a, preset_b = None, None
    if f_col1.button("🇫🇷 France vs 🇧🇷 Brésil", use_container_width=True):
        preset_a, preset_b = "France", "Brazil"
    if f_col2.button("🇲🇦 Maroc vs 🇪🇸 Espagne", use_container_width=True):
        preset_a, preset_b = "Morocco", "Spain"
    if f_col3.button("🇦🇷 Argentine vs 🇩🇪 Allemagne", use_container_width=True):
        preset_a, preset_b = "Argentina", "Germany"
    if f_col4.button("🏴󠁧󠁢󠁥󠁮󠁧󠁿 Angleterre vs 🇳🇱 Pays-Bas", use_container_width=True):
        preset_a, preset_b = "England", "Netherlands"

    st.markdown("<br>", unsafe_allow_html=True)

    col_team1, col_vs, col_team2 = st.columns([5, 2, 5])

    default_a = teams_list.index(preset_a) if preset_a in teams_list else (teams_list.index("France") if "France" in teams_list else 0)
    default_b = teams_list.index(preset_b) if preset_b in teams_list else (teams_list.index("Brazil") if "Brazil" in teams_list else min(1, len(teams_list)-1))

    with col_team1:
        st.markdown("##### Équipe A")
        team_a = st.selectbox("Sélectionnez la première équipe", options=teams_list, index=default_a, label_visibility="collapsed", key="sel_a")
        st.markdown(f"<h1 style='text-align: center; font-size: 4rem;'>{get_flag(team_a)}</h1>", unsafe_allow_html=True)
        st.markdown(f"<h3 style='text-align: center;'>{team_a}</h3>", unsafe_allow_html=True)

    with col_vs:
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.markdown("<div class='vs-circle'>VS</div>", unsafe_allow_html=True)

    with col_team2:
        st.markdown("##### Équipe B")
        team_b = st.selectbox("Sélectionnez la seconde équipe", options=teams_list, index=default_b, label_visibility="collapsed", key="sel_b")
        st.markdown(f"<h1 style='text-align: center; font-size: 4rem;'>{get_flag(team_b)}</h1>", unsafe_allow_html=True)
        st.markdown(f"<h3 style='text-align: center;'>{team_b}</h3>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("📊 Lancer la simulation prédictive", use_container_width=True, type="primary"):
        if team_a == team_b:
            st.warning("⚠️ Veuillez choisir deux nations différentes pour effectuer une simulation.")
        else:
            with st.spinner("Calcul des deltas de forme et inférence du Random Forest..."):
                res = predict_match(team_a, team_b)
                form_a = get_current_form(df_processed, team_a)
                form_b = get_current_form(df_processed, team_b)

            winner = res["winner"]
            proba = res["probability"]
            loser = team_b if winner == team_a else team_a

            # Card Vainqueur
            st.markdown(f"""
            <div class="winner-card">
                <div class="winner-title">🏆 VAINQUEUR ESTIMÉ PAR LE MODÈLE</div>
                <div class="winner-name">{get_flag(winner)} {winner}</div>
                <p style="margin: 0; opacity: 0.9;">Probabilité de victoire : <b>{proba:.1%}</b> face à {loser}</p>
            </div>
            """, unsafe_allow_html=True)

            # Jauge visuelle de probabilité
            st.markdown("##### ⚖️ Répartition des chances de victoire")
            proba_a = proba if winner == team_a else (1.0 - proba)
            st.progress(proba_a, text=f"{team_a} ({proba_a:.1%}) vs {team_b} ({1.0 - proba_a:.1%})")

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("##### 📈 Comparatif des métriques sur les 10 derniers matchs")

            m1, m2, m3 = st.columns(3)
            
            with m1:
                st.metric(
                    label="Moyenne de Points",
                    value=f"{form_a['avg_points']:.2f} pts",
                    delta=f"{form_a['avg_points'] - form_b['avg_points']:+.2f} vs {team_b}"
                )
            with m2:
                st.metric(
                    label="Buts Marqués / match",
                    value=f"{form_a['avg_goals_scored']:.2f}",
                    delta=f"{form_a['avg_goals_scored'] - form_b['avg_goals_scored']:+.2f} vs {team_b}"
                )
            with m3:
                st.metric(
                    label="Buts Encaissés / match",
                    value=f"{form_a['avg_goals_conceded']:.2f}",
                    delta=f"{form_b['avg_goals_conceded'] - form_a['avg_goals_conceded']:+.2f} vs {team_b}",
                    delta_color="inverse"
                )

            # Tableau comparatif
            comp_df = pd.DataFrame({
                "Métrique de Forme": ["Points Moyens", "Buts Marqués / match", "Buts Encaissés / match"],
                f"{get_flag(team_a)} {team_a}": [f"{form_a['avg_points']:.2f}", f"{form_a['avg_goals_scored']:.2f}", f"{form_a['avg_goals_conceded']:.2f}"],
                f"{get_flag(team_b)} {team_b}": [f"{form_b['avg_points']:.2f}", f"{form_b['avg_goals_scored']:.2f}", f"{form_b['avg_goals_conceded']:.2f}"]
            })
            st.dataframe(comp_df, use_container_width=True, hide_index=True)

# ==========================================
# TAB 2 : ASSISTANT LANGGRAPH (CHAT)
# ==========================================
with tab_chat:
    st.markdown("### 💬 Assistant Conversaionnel RAG & Agentic AI")
    st.caption("Interrogez l'agent autonome propulsé par LangGraph et Mistral AI. Il peut exécuter l'outil ML pour prédire des rencontres ou consulter la base de connaissances du tournoi.")

    api_key_to_use = api_key_input or os.getenv("MISTRAL_API_KEY")

    if not api_key_to_use:
        st.warning("🔑 Veuillez entrer votre clé API Mistral AI dans la barre latérale pour débloquer l'agent.")
    else:
        # Puces de suggestions de questions
        st.write("**Questions suggérées :**")
        s_col1, s_col2, s_col3 = st.columns(3)
        
        prompt_suggestion = None
        if s_col1.button("🏆 Qui a gagné la Coupe du Monde 2026 ?", use_container_width=True):
            prompt_suggestion = "Qui a gagné la finale de la Coupe du Monde 2026 et quel était le score ?"
        if s_col2.button("🔮 Qui est favori pour la Coupe du Monde 2030 ?", use_container_width=True):
            prompt_suggestion = "Qui est le grand favori pour remporter la Coupe du Monde 2030 selon ton modèle ?"
        if s_col3.button("⚽ Simule un match Maroc vs Espagne", use_container_width=True):
            prompt_suggestion = "Peux-tu simuler un match entre le Maroc et l'Espagne ?"

        # Gestion des messages en session
        if "messages" not in st.session_state:
            st.session_state["messages"] = [
                {
                    "role": "assistant",
                    "content": "Bonjour ! Je suis votre expert AI des Coupes du Monde. Je peux vous donner les résultats historiques officiels (2026) ou simuler des prédictions pour les prochaines éditions (2030). Que souhaitez-vous savoir ?",
                }
            ]

        if "thread_id" not in st.session_state:
            st.session_state["thread_id"] = "session_streamlit_user"

        # Affichage du chat
        for msg in st.session_state["messages"]:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])

        # Soumission de question (Input ou Suggestion)
        user_input = st.chat_input("Ex: Quel est le format de la Coupe du Monde 2030 ?")
        prompt = prompt_suggestion or user_input

        if prompt:
            st.session_state["messages"].append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.write(prompt)

            with st.chat_message("assistant"):
                with st.spinner("🧠 Réflexion de l'agent et sélection des outils..."):
                    try:
                        agent_executor = get_agent_executor(
                            api_key=api_key_to_use,
                            model_name=model_choice,
                        )
                        config = {"configurable": {"thread_id": st.session_state["thread_id"]}}

                        response = agent_executor.invoke(
                            {"messages": [("user", prompt)]},
                            config=config,
                        )

                        assistant_reply = response["messages"][-1].content
                        st.write(assistant_reply)

                        st.session_state["messages"].append(
                            {"role": "assistant", "content": assistant_reply}
                        )

                    except Exception as err:
                        st.error(f"❌ Erreur lors de l'exécution de l'agent : {err}")

# ==========================================
# TAB 3 : ARCHITECTURE TECHNIQUE
# ==========================================
with tab_arch:
    st.markdown("### 📐 Architecture & Fonctionnement Technique")

    st.markdown("""
    #### 1. Machine Learning Engine (`src/ml.py`)
    - **Dataset Historique** : Analyse des rencontres internationales (1994 - 2026).
    - **Feature Engineering Dynamic** :
      - `avg_points` : Moyenne des points pris sur les 10 derniers matchs récents.
      - `avg_goals_scored` / `avg_goals_conceded` : Attaque et défense glissantes.
      - Deltas comparatifs entre nations sur terrain neutre.
    - **Modèle** : `RandomForestClassifier` (100 estimators) avec standardisation via `StandardScaler`.

    #### 2. Agent ReAct LangGraph (`src/agent.py` & `src/tools.py`)
    - **Orchestrateur** : `StateGraph` de LangGraph avec persistance de contexte (`MemorySaver`).
    - **Moteur LLM** : Mistral AI (`ChatMistralAI`).
    - **Outils Autonomes** :
      - `predict_world_cup_match` : Exécute l'inférence Machine Learning sauvegardée (`.joblib`).
      - `search_world_cup_knowledge` : Moteur RAG/Knowledge DB sur les règles, vainqueurs (2026) et hôtes (2030).
    """)

    st.code("""
                   ┌───────────────────────────────────┐
                   │    Requête Utilisateur (Streamlit) │
                   └─────────────────┬─────────────────┘
                                     │
                                     ▼
                        ┌───────────────────────────┐
                        │   Agent LangGraph (State) │
                        └────────────┬──────────────┘
                                     │
                      (Analyse Intent & Outillage)
                                     │
             ┌───────────────────────┴───────────────────────┐
             ▼                                               ▼
┌───────────────────────────┐                   ┌───────────────────────────┐
│ Tool: predict_match       │                   │ Tool: search_knowledge    │
│  ➜ Scikit-Learn Model     │                   │  ➜ Base d'Informations    │
└────────────┬──────────────┘                   └────────────┬──────────────┘
             │                                               │
             └───────────────────────┬───────────────────────┘
                                     │
                                     ▼
                       ┌───────────────────────────┐
                       │  Synthèse Finale (Mistral)│
                       └───────────────────────────┘
    """, language="text")