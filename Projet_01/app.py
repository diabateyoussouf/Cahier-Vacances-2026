import sys
import uuid
import os
from pathlib import Path
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from src.agent import build_agent_graph

try:
    from src.database import get_connection, execute_query
except ImportError:
    from database import get_connection, execute_query

st.set_page_config(
    page_title="SkyOps AI - Enterprise Dashboard & Agentic Assistant",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .header-container {
        background: linear-gradient(135deg, #0F172A 0%, #1E293B 100%);
        padding: 1.8rem;
        border-radius: 12px;
        color: white;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    .header-title {
        font-size: 2.2rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        margin-bottom: 0.2rem;
    }
    .header-subtitle {
        color: #94A3B8;
        font-size: 1rem;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_db_connection():
    return get_connection()

db_conn = load_db_connection()

if "thread_id" not in st.session_state:
    st.session_state.thread_id = f"session_{uuid.uuid4().hex[:8]}"

if "messages" not in st.session_state:
    st.session_state.messages = []

# --- BARRE LATÉRALE : SAISIE DE LA CLÉ API ---
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/airplane-take-off.png", width=64)
    st.title("SkyOps Control Center")
    st.caption("Plateforme décisionnelle & Agent LangGraph")
    
    st.markdown("---")
    st.markdown("### 🔑 Configuration API")
    
    # 1. Vérification si une clé existe dans les secrets ou l'environnement
    default_key = st.secrets.get("MISTRAL_API_KEY", os.getenv("MISTRAL_API_KEY", ""))
    
    # 2. Input utilisateur (Masqué comme un mot de passe)
    user_api_key = st.text_input(
        "Clé API Mistral :",
        type="password",
        value=default_key,
        help="Obtenez votre clé gratuite sur https://console.mistral.ai"
    )

    if not user_api_key:
        st.warning("⚠️ Veuillez entrer une clé API Mistral pour activer l'agent.")
    else:
        st.success("✅ Clé API configurée")

    st.markdown("---")
    st.markdown("### 📊 État de la session")
    st.info(f"**Thread ID:** `{st.session_state.thread_id}`")
    
    if st.button("🔄 Réinitialiser la session", use_container_width=True):
        st.session_state.messages = []
        st.session_state.thread_id = f"session_{uuid.uuid4().hex[:8]}"
        st.rerun()

# Chargement dynamique du graphe selon la clé saisie
@st.cache_resource
def get_cached_agent(api_key: str):
    return build_agent_graph(api_key)

graph = get_cached_agent(user_api_key) if user_api_key else None

st.markdown("""
<div class="header-container">
    <div class="header-title">✈️ SkyOps AI Operations Center</div>
    <div class="header-subtitle">Analyse analytique des vols & Assistant conversationnel décisionnel multi-bases</div>
</div>
""", unsafe_allow_html=True)

tab_dash, tab_agent, tab_arch = st.tabs([
    "📊 Executive Dashboard", 
    "🤖 Assistant Agentic (LangGraph)", 
    "🏗️ Spécifications Architecture"
])

# ==========================================
# TAB 1 : EXECUTIVE DASHBOARD (ANALYTICS)
# ==========================================
with tab_dash:
    df_bookings = execute_query(db_conn, """
        SELECT b.id, b.status, b.seat_class, f.flight_number, f.destination, f.price_eur
        FROM bookings b
        JOIN flights f ON b.flight_id = f.id
    """)
    df_passengers = execute_query(db_conn, "SELECT COUNT(*) as total FROM passengers")
    df_flights = execute_query(db_conn, "SELECT COUNT(*) as total FROM flights")
    
    total_rev = df_bookings[df_bookings['status'] == 'confirmed']['price_eur'].sum()
    confirmed_count = len(df_bookings[df_bookings['status'] == 'confirmed'])
    total_bookings = len(df_bookings)
    conversion_rate = (confirmed_count / total_bookings * 100) if total_bookings > 0 else 0

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Chiffre d'Affaires Confirmé", f"{total_rev:,.2f} €", delta="+12.4%")
    m2.metric("Réservations Totales", f"{total_bookings}", delta=f"{conversion_rate:.1f}% confirmées")
    m3.metric("Passagers Enregistrés", f"{df_passengers['total'].iloc[0]}")
    m4.metric("Vols au Catalogue", f"{df_flights['total'].iloc[0]}")

    st.markdown("---")

    col_g1, col_g2 = st.columns(2)

    with col_g1:
        st.subheader("📈 Chiffre d'Affaires par Destination (€)")
        rev_by_dest = df_bookings[df_bookings['status'] == 'confirmed'].groupby('destination')['price_eur'].sum().reset_index()
        fig_dest = px.bar(
            rev_by_dest, 
            x='destination', 
            y='price_eur',
            color='price_eur',
            color_continuous_scale='Blues',
            labels={'price_eur': 'Revenu (€)', 'destination': 'Destination'},
            text_auto='.2s'
        )
        fig_dest.update_layout(template="plotly_white", showlegend=False, margin=dict(l=20, r=20, t=30, b=20))
        st.plotly_chart(fig_dest, use_container_width=True)

    with col_g2:
        st.subheader("📊 Répartition du Statut des Réservations")
        status_counts = df_bookings['status'].value_counts().reset_index()
        fig_status = px.pie(
            status_counts, 
            names='status', 
            values='count',
            color='status',
            color_discrete_map={'confirmed': '#10B981', 'cancelled': '#EF4444', 'pending': '#F59E0B'},
            hole=0.4
        )
        fig_status.update_layout(template="plotly_white", margin=dict(l=20, r=20, t=30, b=20))
        st.plotly_chart(fig_status, use_container_width=True)

    st.subheader("📋 Aperçu Récent des Réservations")
    st.dataframe(df_bookings.head(8), use_container_width=True, hide_index=True)


# ==========================================
# TAB 2 : ASSISTANT IA & CONSOLE SQL
# ==========================================
with tab_agent:
    with st.expander("💻 Exécuter une requête SQL manuelle (Console Directe)", expanded=False):
        st.markdown("Saisissez directement votre requête SQL brute pour interroger la base SQLite (`passengers`, `flights`, `bookings`).")
        default_query = "SELECT * FROM passengers WHERE id = 1;"
        manual_sql = st.text_area("Requête SQL :", value=default_query, height=80)
        
        if st.button("Exécuter la requête SQL"):
            try:
                res_df = execute_query(db_conn, manual_sql)
                if res_df.empty:
                    st.warning("La requête s'est exécutée mais n'a retourné aucun résultat.")
                else:
                    st.success(f"Résultat ({len(res_df)} ligne(s)) :")
                    st.dataframe(res_df, use_container_width=True)
            except Exception as err:
                st.error(f"Erreur d'exécution SQL : {err}")

    st.markdown("---")

    st.subheader("💬 Interrogez l'Agent IA en Langage Naturel")
    st.caption("L'agent choisit dynamiquement entre SQL et RAG Vectoriel pour vous répondre.")

    st.write("**Suggestions de requêtes complexes :**")
    prompt_cols = st.columns(3)
    p1 = prompt_cols[0].button("🔍 Qui a réservé le vol AF5501 et quel est le chiffre d'affaires à Paris CDG ?")
    p2 = prompt_cols[1].button("📂 Quelles sont les règles d'annulation des billets ?")
    p3 = prompt_cols[2].button("👤 Récupérer les informations SQL du client avec l'ID 1")

    selected_prompt = None
    if p1: selected_prompt = "Qui sont les passagers ayant réservé le vol AF5501 et quel est le revenu total généré par les vols confirmés vers Paris CDG ?"
    if p2: selected_prompt = "Quelles sont les conditions et frais d'annulation de billet ?"
    if p3: selected_prompt = "Quelles sont les informations dans la base SQL pour le passager avec l'ID 1 ?"

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_input = st.chat_input("Ex: Quel est le tarif moyen des vols vers Nice ?") or selected_prompt

    if user_input:
        if not graph:
            st.error("Veuillez renseigner une clé API Mistral valide dans la barre latérale avant de poser une question.")
        else:
            st.session_state.messages.append({"role": "user", "content": user_input})
            with st.chat_message("user"):
                st.markdown(user_input)

            with st.chat_message("assistant"):
                with st.status("🧠 Agent Agentique en cours de réflexion...", expanded=True) as status:
                    st.write("1. Analyse de la requête et identification des outils...")
                    config = {"configurable": {"thread_id": st.session_state.thread_id}}
                    
                    try:
                        results = graph.invoke(
                            {"messages": [("user", user_input)]},
                            config=config
                        )
                        st.write("2. Interrogation des bases de données et synthèse de la réponse...")
                        status.update(label="✅ Traitement terminé avec succès", state="complete", expanded=False)
                        
                        response_text = results["messages"][-1].content
                        st.markdown(response_text)
                        st.session_state.messages.append({"role": "assistant", "content": response_text})

                    except Exception as e:
                        status.update(label="❌ Erreur lors du traitement", state="error", expanded=False)
                        if "429" in str(e) or "rate_limited" in str(e).lower():
                            error_msg = "⚠️ **Limite de requêtes API atteinte (Erreur HTTP 429)**. Patientez quelques secondes ou vérifiez votre quota."
                        elif "401" in str(e) or "unauthorized" in str(e).lower():
                            error_msg = "🔑 **Clé API non valide**. Veuillez vérifier la clé saisie dans la barre latérale."
                        else:
                            error_msg = f"❌ **Erreur d'exécution :** {str(e)}"
                        st.error(error_msg)


# ==========================================
# TAB 3 : SPÉCIFICATIONS & ARCHITECTURE
# ==========================================
with tab_arch:
    st.subheader("🏗️ Architecture Technique du Projet")
    
    col_a1, col_a2 = st.columns(2)
    
    with col_a1:
        st.markdown("""
        ### Stack Technologique
        - **Orchestration** : `LangGraph` (StateGraph avec gestion de la mémoire persistance via `MemorySaver`)
        - **Modèle de Langage** : `Mistral AI` (`mistral-small-latest`)
        - **Embeddings & Vectorstore** : `HuggingFace` (`all-MiniLM-L6-v2`) + `ChromaDB`
        - **Base de Données Relationnelle** : `SQLite` (SQL)
        - **Frontend & Visualisation** : `Streamlit` + `Plotly Express`
        """)

    with col_a2:
        st.markdown("""
        ### Flux de Fonctionnement (ReAct Loop)
        ```text
        [Utilisateur] ──► [Saisie Clé API]
                                │
                         [LangGraph State]
                                │
                         (Analyse LLM)
                                │
               ┌────────────────┴────────────────┐
               ▼                                ▼
        [Tool SQLite]                   [Tool ChromaDB]
               │                                │
               └────────────────┬────────────────┘
                                │
                       (Synthèse LLM)
                                │
                                ▼
                       [Réponse Utilisateur]
        ```
        """)