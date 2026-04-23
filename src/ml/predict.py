"""
Prédiction ML — charge le modèle ensemble et prédit sur des données tabulaires.
"""

import numpy as np
import pandas as pd
import joblib
import os

from .model import FEATURE_COLUMNS, CLASSES


def load_model(model_path: str = "models/ml_model.pkl") -> dict:
    """Charge le modèle ML depuis le fichier .pkl."""
    return joblib.load(model_path)


def predict_tabular(features: dict, model_path: str = "models/ml_model.pkl") -> dict:
    """
    Prédit la maladie à partir de données agronomiques.

    Args:
        features: dict avec les clés de FEATURE_COLUMNS
        model_path: Chemin vers le modèle sauvegardé

    Returns:
        dict avec 'class', 'confidence', 'probabilities'
    """
    if not os.path.exists(model_path):
        return _dummy_prediction()

    bundle = load_model(model_path)
    model = bundle["model"]
    feature_cols = bundle.get("feature_columns", FEATURE_COLUMNS)

    # Construire le vecteur de features
    X = np.array([[features.get(col, 0.0) for col in feature_cols]])

    probs = model.predict_proba(X)[0]
    classes = model.classes_

    predicted_idx = int(np.argmax(probs))
    return {
        "class": classes[predicted_idx],
        "confidence": float(probs[predicted_idx]),
        "probabilities": {cls: float(p) for cls, p in zip(classes, probs)},
    }


def predict_from_csv_row(csv_path: str, row_idx: int = 0, model_path: str = "models/ml_model.pkl") -> dict:
    """Prédit depuis une ligne d'un fichier CSV."""
    df = pd.read_csv(csv_path)
    row = df.iloc[row_idx]
    features = {col: row[col] for col in FEATURE_COLUMNS if col in row}
    result = predict_tabular(features, model_path)
    if "label" in row:
        result["true_label"] = row["label"]
    return result


def _dummy_prediction() -> dict:
    probs = np.random.dirichlet(np.ones(len(CLASSES)))
    idx = int(np.argmax(probs))
    return {
        "class": CLASSES[idx],
        "confidence": float(probs[idx]),
        "probabilities": {cls: float(p) for cls, p in zip(CLASSES, probs)},
        "note": "Modèle non entraîné — prédiction aléatoire"
    }
