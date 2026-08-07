# Cahier de vacances Machine Learnia

## Contexte et Apports Personnels

### Le Projet Initial (Machine Learnia)
Ce dépôt est issu du **Cahier de Vacances Machine Learnia**, un programme conçu pour accompagner les futurs Data Scientists et Engineers à travers des cas pratiques hebdomadaires. Il couvre les fondamentaux de la discipline : analyse de données, Machine Learning classique, architectures RAG et orchestrations basées sur les graphes.

### Mes Améliorations et Réalisations
En partant des notebooks initiaux, j'ai développé et déployé des **applications web interactives prêtes pour la production** :
- **Mise à jour des jeux de données :** Intégration des résultats réels et récents de la phase finale de la Coupe du Monde 2026 (Espagne championne) et adaptation des pipelines pour autoriser la simulation de la Coupe du Monde 2030.
- **Interfaces Web Streamlit :** Conception d'applications complètes comprenant tableaux de bord, cartes de métriques, jauges de probabilités et sélecteurs dynamiques.
- **Agents Autonomes ReAct :** Implémentation de graphes d'état avec **LangGraph** et **Mistral AI**, permettant au système de basculer entre la recherche documentaire (RAG) et l'exécution d'outils de prédiction ML ou de requêtage SQL.
- **Industrialisation et Déploiement :** Structuration des dépendances avec `uv` et mise en ligne des applications sur Streamlit Community Cloud.

---

## Applications Déployées en Ligne

| Projet | Intitulé & Description | Technologies Principales | Lien de l'Application |
| :--- | :--- | :--- | :--- |
| **Projet 01** | **Automatisation SQL & LangGraph**<br>Génération, exécution et correction automatique de requêtes SQL à partir de requêtes en langage naturel. | ![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white) ![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat-square&logo=streamlit&logoColor=white) ![LangGraph](https://img.shields.io/badge/LangGraph-1C3C3C?style=flat-square&logo=langchain&logoColor=white) ![SQLite](https://img.shields.io/badge/SQLite-003B57?style=flat-square&logo=sqlite&logoColor=white) | [Accéder à l'application](https://automatisationsql-langgraph.streamlit.app/) |
| **Projet 02** | **World Cup Predictor & Agent RAG**<br>Simulateur ML de confrontations internationales et assistant documentaire sur le tournoi mondial. | ![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white) ![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat-square&logo=streamlit&logoColor=white) ![Scikit-Learn](https://img.shields.io/badge/scikit_learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white) ![LangGraph](https://img.shields.io/badge/LangGraph-1C3C3C?style=flat-square&logo=langchain&logoColor=white) | [Accéder à l'application](https://predictionfootball-worldcup.streamlit.app/) |

---

## Stack Technique Globale

| Categorie | Technologies & Outils |
| :--- | :--- |
| **Gestionnaire & Environnement** | ![uv](https://img.shields.io/badge/uv-DE5FE9?style=flat-square&logo=astral&logoColor=white) ![Python 3.13](https://img.shields.io/badge/Python-3.13+-3776AB?style=flat-square&logo=python&logoColor=white) |
| **Machine Learning & Traitement** | ![Scikit-Learn](https://img.shields.io/badge/scikit_learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white) ![Pandas](https://img.shields.io/badge/pandas-150458?style=flat-square&logo=pandas&logoColor=white) ![NumPy](https://img.shields.io/badge/numpy-013243?style=flat-square&logo=numpy&logoColor=white) |
| **Orchestration & LLM** | ![LangChain](https://img.shields.io/badge/LangChain-1C3C3C?style=flat-square&logo=langchain&logoColor=white) ![LangGraph](https://img.shields.io/badge/LangGraph-2C2C2C?style=flat-square) ![Mistral AI](https://img.shields.io/badge/Mistral_AI-FF7000?style=flat-square&logo=mistral&logoColor=white) |
| **Bases de Données & Vectorstore** | ![SQLite](https://img.shields.io/badge/SQLite-003B57?style=flat-square&logo=sqlite&logoColor=white) ![ChromaDB](https://img.shields.io/badge/ChromaDB-FF6600?style=flat-square) |
| **Interface & Hosting** | ![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat-square&logo=streamlit&logoColor=white) ![Streamlit Cloud](https://img.shields.io/badge/Streamlit_Cloud-000000?style=flat-square&logo=streamlit&logoColor=white) |

---

## Structure du Dépôt

```text
Cahier-Vacances-2026/
├── Projet_01/
│   ├── app.py
│   └── projet_01.ipynb
├── Projet_02/
│   ├── app.py
│   ├── models/
│   ├── src/
│   └── data/
├── pyproject.toml
├── requirements.txt
└── README.md
