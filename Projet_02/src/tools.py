import sys
from pathlib import Path
from langchain_core.tools import tool

# --- RESOLUTION DYNAMIQUE DES CHEMINS ---
SRC_DIR = Path(__file__).resolve().parent
PROJET_DIR = SRC_DIR.parent

if str(PROJET_DIR) not in sys.path:
    sys.path.append(str(PROJET_DIR))

from src.ml import predict_match


@tool
def predict_world_cup_match(team_a: str, team_b: str) -> str:
    """Prédit le résultat d'un match de football international entre deux équipes sur terrain neutre.
    Entrées : team_a (ex: 'France'), team_b (ex: 'Brazil').
    Retourne le vainqueur estimé et la probabilité associée basée sur la forme récente des équipes.
    """
    try:
        res = predict_match(team_a, team_b)
        return f"Résultat de la simulation : Victoire de {res['winner']} contre {team_a if res['winner'] == team_b else team_b} avec une probabilité de {res['probability']:.1%}."
    except Exception as e:
        return f"Erreur lors du calcul de la prédiction : {str(e)}. Assurez-vous que les noms d'équipes sont en anglais (ex: 'Brazil', 'Germany', 'Spain', 'Morocco')."


@tool
def search_world_cup_knowledge(query: str) -> str:
    """Recherche des informations sur l'histoire, les vainqueurs, les résultats officiels et l'organisation des Coupes du Monde (2026, 2030, etc.)."""
    knowledge_db = {
        "2026": "Coupe du Monde 2026 : L'Espagne a été sacrée championne du monde en battant l'Argentine 1-0 après prolongations en finale le 19 juillet 2026 à East Rutherford. L'Angleterre termine 3e (victoire 6-4 contre la France).",
        "vainqueur 2026": "Vainqueur 2026 : L'Espagne (victoire 1-0 contre l'Argentine en finale).",
        "finale 2026": "Finale 2026 : Espagne 1 - 0 Argentine (après prolongations), disputée le 19 juillet 2026 au MetLife Stadium (East Rutherford, USA).",
        "2030": "Coupe du Monde 2030 : Co-organisée par le Maroc, l'Espagne et le Portugal. Trois matchs d'ouverture auront lieu en Amérique du Sud (Uruguay, Argentine, Paraguay) pour célébrer le centenaire de la compétition.",
        "format": "Le format à 48 équipes regroupe 12 groupes de 4 équipes et une phase à élimination directe débutant en 1/16e de finale (104 matchs au total).",
    }

    query_lower = query.lower()
    matches = [
        val
        for key, val in knowledge_db.items()
        if key in query_lower or any(word in query_lower for word in key.split())
    ]

    if matches:
        return "\n".join(matches)

    return (
        "Informations FIFA : L'Espagne est championne du monde 2026 (1-0 vs Argentine). "
        "La prochaine édition aura lieu en 2030 au Maroc, en Espagne et au Portugal."
    )


ALL_TOOLS = [predict_world_cup_match, search_world_cup_knowledge]