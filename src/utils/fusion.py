"""
Module de Fusion — combine les prédictions DL (CNN) et ML (Random Forest + XGBoost).

Stratégie : Weighted Average des probabilités
    - DL (image)  : poids 0.6  (la vision est souvent plus discriminante)
    - ML (données): poids 0.4
"""

import numpy as np
import argparse
import json

CLASSES = ["Healthy", "Rust", "Black_Spot", "Powdery_Mildew", "Bacterial_Blight"]

DL_WEIGHT = 0.6
ML_WEIGHT = 0.4

DISEASE_INFO = {
    "Healthy": {
        "emoji": "✅",
        "severity": "Aucune",
        "treatment": "Aucun traitement nécessaire. Continuer les bonnes pratiques agricoles.",
        "color": "#22c55e",
    },
    "Rust": {
        "emoji": "🍂",
        "severity": "Modérée",
        "treatment": "Appliquer un fongicide à base de triazole. Retirer les feuilles infectées.",
        "color": "#f97316",
    },
    "Black_Spot": {
        "emoji": "⚫",
        "severity": "Modérée",
        "treatment": "Fongicide cuivrique préventif. Améliorer la circulation d'air.",
        "color": "#374151",
    },
    "Powdery_Mildew": {
        "emoji": "🌫️",
        "severity": "Faible à modérée",
        "treatment": "Soufre en poudre ou bicarbonate de soude. Réduire l'humidité.",
        "color": "#a78bfa",
    },
    "Bacterial_Blight": {
        "emoji": "🔴",
        "severity": "Élevée",
        "treatment": "Cuivre bactéricide. Isoler les plants infectés immédiatement.",
        "color": "#ef4444",
    },
}


def fuse_predictions(dl_result: dict, ml_result: dict, dl_weight: float = DL_WEIGHT) -> dict:
    """
    Fusionne les prédictions DL et ML par moyenne pondérée.

    Args:
        dl_result: dict retourné par dl.predict_image (contient 'probabilities')
        ml_result: dict retourné par ml.predict_tabular (contient 'probabilities')
        dl_weight: Poids accordé au modèle DL (ML reçoit 1 - dl_weight)

    Returns:
        dict avec le diagnostic final enrichi
    """
    ml_weight = 1.0 - dl_weight

    # Aligner les probabilités sur l'ordre de CLASSES
    dl_probs = np.array([dl_result["probabilities"].get(c, 0.0) for c in CLASSES])
    ml_probs = np.array([ml_result["probabilities"].get(c, 0.0) for c in CLASSES])

    # Normaliser (au cas où)
    dl_probs /= dl_probs.sum() + 1e-8
    ml_probs /= ml_probs.sum() + 1e-8

    fused = dl_weight * dl_probs + ml_weight * ml_probs
    predicted_idx = int(np.argmax(fused))
    predicted_class = CLASSES[predicted_idx]
    confidence = float(fused[predicted_idx])

    info = DISEASE_INFO.get(predicted_class, {})

    return {
        "final_class": predicted_class,
        "confidence": confidence,
        "severity": info.get("severity", "Inconnue"),
        "treatment": info.get("treatment", "Consulter un agronome."),
        "emoji": info.get("emoji", "❓"),
        "color": info.get("color", "#6b7280"),
        "probabilities": {cls: float(p) for cls, p in zip(CLASSES, fused)},
        "details": {
            "dl_prediction": dl_result.get("class"),
            "dl_confidence": dl_result.get("confidence"),
            "ml_prediction": ml_result.get("class"),
            "ml_confidence": ml_result.get("confidence"),
            "weights": {"dl": dl_weight, "ml": ml_weight},
        },
    }


def run_full_pipeline(image_path: str, features: dict) -> dict:
    """
    Lance le pipeline complet DL + ML + Fusion.

    Args:
        image_path: Chemin vers l'image de la feuille
        features: Dictionnaire des données agronomiques

    Returns:
        Diagnostic final fusionné
    """
    from src.dl.predict import predict_image
    from src.ml.predict import predict_tabular

    dl_result = predict_image(image_path)
    ml_result = predict_tabular(features)
    return fuse_predictions(dl_result, ml_result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plant Disease Detector — Pipeline complet")
    parser.add_argument("--image", type=str, required=True, help="Chemin vers l'image")
    parser.add_argument("--csv",   type=str, default=None,  help="Chemin vers le CSV agronomique")
    args = parser.parse_args()

    # Charger les features depuis le CSV ou utiliser des valeurs par défaut
    if args.csv:
        import pandas as pd
        df = pd.read_csv(args.csv)
        features = df.iloc[0].to_dict()
    else:
        features = {
            "temperature": 25.0, "humidity": 70.0, "rainfall": 60.0,
            "soil_ph": 6.5, "nitrogen": 40.0, "phosphorus": 20.0,
            "potassium": 30.0, "leaf_area_index": 3.0,
            "plant_age_days": 90, "previous_infection": 0,
        }

    result = run_full_pipeline(args.image, features)
    print(json.dumps(result, indent=2, ensure_ascii=False))
