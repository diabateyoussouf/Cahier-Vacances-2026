# Cahier de vacances Machine Learnia

## 🎯 Contexte & Apports Personnels

### 💡 Le Projet Initial (Machine Learnia)
Ce dépôt est issu de l'initiative du **Cahier de Vacances Machine Learnia**, conçu pour accompagner les futurs **Data Scientists** et **Engineers IA** à travers des cas pratiques hebdomadaires durant l'été. Il couvre les fondamentaux de la Data Science : analyse de données, Machine Learning classique, architectures RAG et agents autonomes basés sur les graphes.

### 🚀 Mes Améliorations & Réalisations
En partant des travaux initiaux, j'ai enrichi le projet pour transformer ces exercices en **applications web interactives et déployées en production** :
- **Intégration des données réelles récents :** Complétion du dataset avec l'ensemble des matchs officiels jusqu'à la finale de la Coupe du Monde 2026 (Espagne championne) et adaptation du modèle pour simuler la Coupe du Monde 2030.
- **Interfaces Web Streamlit Modernes :** Conception d'applications interactives avec design soigné (cartes de métriques, jauges de probabilités, tableaux comparatifs et sélecteurs de matchs).
- **Architectures Agentiques Autonomes :** Implémentation d'agents ReAct sous **LangGraph** couplés à **Mistral AI**, capables de basculer intelligemment entre recherche documentaire (RAG) et exécution de prédictions ML en temps réel.
- **Industrialisation & Déploiement :** Gestion des dépendances avec `uv` et mise en ligne des projets sur Streamlit Community Cloud.

---

## 🌐 Applications Déployées en Ligne

| Projet | Intitulé & Description | Stack Technique | Lien de l'Application |
| :--- | :--- | :--- | :--- |
| **Projet 01** | **Automatisation SQL & Agentic AI**<br>Génération, exécution et correction autonome de requêtes SQL complexes via un agent LangGraph. | Python, Streamlit, LangGraph, SQL, LLM | 🔗 [Consulter l'application](https://automatisationsql-langgraph.streamlit.app/) |
| **Projet 02** | **World Cup Predictor & Agent RAG**<br>Simulateur ML de matchs internationaux et assistant conversationnel expert des Coupes du Monde (2026 & 2030). | Python, Streamlit, Scikit-Learn, LangGraph, Mistral AI | 🔗 [Consulter l'application](https://predictionfootball-worldcup.streamlit.app/) |

---

## 🛠️ Stack Technique Globale

* **Environnement & Dépendances :** Python 3.13+, [`uv`](https://docs.astral.sh/uv/)
* **Data & Machine Learning :** Pandas, NumPy, Scikit-Learn, Joblib
* **AI Agentique & RAG :** LangGraph (`StateGraph`, `MemorySaver`), LangChain, ChromaDB
* **Moteur LLM :** Mistral AI (`ChatMistralAI`)
* **Déploiement UI :** Streamlit, Streamlit Community Cloud

---

## 📁 Structure du Repo

Chaque projet est structuré dans son propre dossier à la racine :

```text
Cahier-Vacances-2026/          <-- RACINE DU DÉPÔT
├── Projet_01/
│   ├── app.py
│   └── projet_01.ipynb
├── Projet_02/
│   ├── app.py
│   ├── models/
│   ├── src/
│   └── data/
├── pyproject.toml
├── requirements.txt           <-- Export des dépendances globales pour le déploiement
└── README.md                  <-- Documentation générale
