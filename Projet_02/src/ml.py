import sys
from pathlib import Path
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# --- RESOLUTION DYNAMIQUE DES CHEMINS ---
SRC_DIR = Path(__file__).resolve().parent
PROJET_DIR = SRC_DIR.parent
MODELS_DIR = PROJET_DIR / "models"
DATA_PATH = PROJET_DIR / "data" / "results.csv"

# Créer le dossier models s'il n'existe pas
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Ajouter le dossier Projet_02 au sys.path
if str(PROJET_DIR) not in sys.path:
    sys.path.append(str(PROJET_DIR))

from utils import add_recent_form, get_current_form, load_results

FEATURES = [
    "diff_avg_points",
    "diff_avg_goals_scored",
    "diff_avg_goals_conceded",
    "is_neutral",
    "is_friendly",
]


def train_and_save():
    print("⏳ Chargement et préparation des données...")
    df = load_results(str(DATA_PATH))

    # Filtrage ère moderne
    df = df[(df["date"] >= "1994-01-01") & (df["date"] <= "2026-06-30")].copy()

    # Détermination du résultat
    df["outcome"] = "draw"
    df.loc[df["home_score"] > df["away_score"], "outcome"] = "home_win"
    df.loc[df["home_score"] < df["away_score"], "outcome"] = "away_win"

    # Calcul de la forme récente
    df = add_recent_form(df, window=10, min_matches=5)
    df = df.dropna(subset=["home_avg_points", "away_avg_points"]).reset_index(
        drop=True
    )

    # Préparation du dataset pour le ML (sans les nuls)
    data = df[df["home_score"] != df["away_score"]].copy()
    data["home_win"] = (data["home_score"] > data["away_score"]).astype(int)

    data["is_neutral"] = data["neutral"].astype(int)
    data["is_friendly"] = (data["tournament"] == "Friendly").astype(int)

    data["diff_avg_points"] = (
        data["home_avg_points"] - data["away_avg_points"]
    )
    data["diff_avg_goals_scored"] = (
        data["home_avg_goals_scored"] - data["away_avg_goals_scored"]
    )
    data["diff_avg_goals_conceded"] = (
        data["home_avg_goals_conceded"] - data["away_avg_goals_conceded"]
    )

    X = data[FEATURES].copy()
    y = data["home_win"].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Standardisation
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Entraînement
    print("🌲 Entraînement du modèle Random Forest...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)

    # Évaluation
    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    baseline = y_test.mean()

    print(f"✅ Modèle entraîné avec succès !")
    print(f"📊 Accuracy : {acc:.2%} (Baseline victoire domicile : {baseline:.2%})")

    # --- SAUVEGARDE DES ARTEFACTS ---
    model_path = MODELS_DIR / "model.joblib"
    scaler_path = MODELS_DIR / "scaler.joblib"
    df_path = MODELS_DIR / "processed_df.joblib"

    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    joblib.dump(df, df_path)

    print(f"💾 Modèle sauvegardé dans : {model_path}")
    print(f"💾 Scaler sauvegardé dans : {scaler_path}")
    print(f"💾 DataFrame sauvegardé dans : {df_path}")

    return model, scaler, df


def load_model_artifacts():
    """Fonction réutilisable par app.py et tools.py pour charger les modèles."""
    model_path = MODELS_DIR / "model.joblib"
    scaler_path = MODELS_DIR / "scaler.joblib"
    df_path = MODELS_DIR / "processed_df.joblib"

    if not (model_path.exists() and scaler_path.exists() and df_path.exists()):
        print("⚠️ Modèle non trouvé, lancement de l'entraînement...")
        return train_and_save()

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    df = joblib.load(df_path)
    return model, scaler, df


def predict_match(team_a: str, team_b: str):
    """Prédit le vainqueur d'un match sur terrain neutre."""
    model, scaler, df = load_model_artifacts()

    form_a = get_current_form(df, team_a)
    form_b = get_current_form(df, team_b)

    diff_avg_points = form_a["avg_points"] - form_b["avg_points"]
    diff_avg_goals_scored = (
        form_a["avg_goals_scored"] - form_b["avg_goals_scored"]
    )
    diff_avg_goals_conceded = (
        form_a["avg_goals_conceded"] - form_b["avg_goals_conceded"]
    )

    features_df = pd.DataFrame(
        [
            [
                diff_avg_points,
                diff_avg_goals_scored,
                diff_avg_goals_conceded,
                1,  # is_neutral = 1
                0,  # is_friendly = 0
            ]
        ],
        columns=FEATURES,
    )

    features_scaled = scaler.transform(features_df)
    proba_a_wins = model.predict_proba(features_scaled)[0, 1]

    if proba_a_wins >= 0.5:
        return {"winner": team_a, "probability": float(proba_a_wins)}
    else:
        return {"winner": team_b, "probability": float(1 - proba_a_wins)}


if __name__ == "__main__":
    train_and_save()

    # Test rapide de la fonction de prédiction
    res = predict_match("France", "Greece")
    print(
        f"\n🧪 Test Prédiction : France vs Grèce -> Vainqueur : {res['winner']} ({res['probability']:.0%})"
    )