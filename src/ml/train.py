"""
Entraînement du modèle ML (Random Forest + XGBoost).
"""

import os
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
import json

from .model import build_ensemble_model, FEATURE_COLUMNS, CLASSES


def generate_synthetic_csv(output_path: str, n_samples: int = 500):
    """
    Génère un CSV agronomique synthétique pour la démo.
    Dans un vrai projet, remplacer par vos données réelles.
    """
    np.random.seed(42)
    n = n_samples

    # Paramètres par classe
    class_params = {
        "Healthy":           {"temp": (20, 4),  "humidity": (55, 10), "rain": (50, 20)},
        "Rust":              {"temp": (25, 5),  "humidity": (75, 10), "rain": (80, 30)},
        "Black_Spot":        {"temp": (22, 4),  "humidity": (80, 10), "rain": (90, 25)},
        "Powdery_Mildew":    {"temp": (28, 4),  "humidity": (45, 10), "rain": (20, 15)},
        "Bacterial_Blight":  {"temp": (30, 5),  "humidity": (85, 10), "rain": (100, 30)},
    }

    rows = []
    n_per_class = n // len(CLASSES)
    for class_name, params in class_params.items():
        for _ in range(n_per_class):
            rows.append({
                "temperature":       np.random.normal(*params["temp"]),
                "humidity":          np.clip(np.random.normal(*params["humidity"]), 20, 100),
                "rainfall":          max(0, np.random.normal(*params["rain"])),
                "soil_ph":           np.random.uniform(5.5, 7.5),
                "nitrogen":          np.random.uniform(10, 80),
                "phosphorus":        np.random.uniform(5, 50),
                "potassium":         np.random.uniform(10, 60),
                "leaf_area_index":   np.random.uniform(1.0, 5.0),
                "plant_age_days":    np.random.randint(30, 365),
                "previous_infection": np.random.choice([0, 1], p=[0.7, 0.3]),
                "label":             class_name,
            })

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"✅ CSV synthétique généré : {output_path} ({len(df)} lignes)")
    return df


def train(
    csv_path: str = "data/raw/agronomic_data.csv",
    model_out: str = "models/ml_model.pkl",
    test_size: float = 0.2,
):
    """
    Entraîne l'ensemble ML sur les données agronomiques.

    Args:
        csv_path: Chemin vers le CSV
        model_out: Chemin de sauvegarde du modèle
        test_size: Fraction de données pour le test

    Returns:
        dict avec les métriques
    """
    # Chargement / génération des données
    if not os.path.exists(csv_path):
        print(f"⚠️  {csv_path} introuvable. Génération de données synthétiques...")
        df = generate_synthetic_csv(csv_path)
    else:
        df = pd.read_csv(csv_path)

    print(f"📊 Dataset: {len(df)} lignes, distribution des classes :")
    print(df["label"].value_counts().to_string())

    X = df[FEATURE_COLUMNS].values
    y = df["label"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=42
    )

    # Modèle
    model = build_ensemble_model()

    print("\n🔄 Cross-validation (5-fold)...")
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="accuracy", n_jobs=-1)
    print(f"  CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    print("\n🏋️  Entraînement final...")
    model.fit(X_train, y_train)

    # Évaluation
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    print("\n📈 Rapport de classification :")
    print(classification_report(y_test, y_pred))

    # Sauvegarde
    os.makedirs(os.path.dirname(model_out), exist_ok=True)
    joblib.dump({"model": model, "feature_columns": FEATURE_COLUMNS, "classes": CLASSES}, model_out)
    print(f"💾 Modèle ML sauvegardé : {model_out}")

    metrics = {
        "cv_mean": float(cv_scores.mean()),
        "cv_std": float(cv_scores.std()),
        "test_accuracy": float(report["accuracy"]),
        "classes": CLASSES,
    }

    with open("models/ml_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    return metrics


if __name__ == "__main__":
    train()
