"""
Machine Learning Module — Random Forest + XGBoost sur données agronomiques.

Features agronomiques attendues (CSV) :
- temperature       : Température moyenne (°C)
- humidity          : Humidité relative (%)
- rainfall          : Précipitations (mm)
- soil_ph           : pH du sol
- nitrogen          : Teneur en azote (mg/kg)
- phosphorus        : Teneur en phosphore (mg/kg)
- potassium         : Teneur en potassium (mg/kg)
- leaf_area_index   : Indice de surface foliaire
- plant_age_days    : Âge de la plante (jours)
- previous_infection: Infection précédente (0/1)
"""

from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import xgboost as xgb

from .model import CLASSES  # noqa: avoid circular if needed


FEATURE_COLUMNS = [
    "temperature",
    "humidity",
    "rainfall",
    "soil_ph",
    "nitrogen",
    "phosphorus",
    "potassium",
    "leaf_area_index",
    "plant_age_days",
    "previous_infection",
]

CLASSES = ["Healthy", "Rust", "Black_Spot", "Powdery_Mildew", "Bacterial_Blight"]


def build_rf_model() -> Pipeline:
    """Random Forest avec normalisation."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(
            n_estimators=200,
            max_depth=12,
            min_samples_split=4,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ))
    ])


def build_xgb_model() -> Pipeline:
    """XGBoost avec normalisation."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", xgb.XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            use_label_encoder=False,
            eval_metric="mlogloss",
            random_state=42,
        ))
    ])


def build_ensemble_model() -> VotingClassifier:
    """
    Ensemble VotingClassifier (RF + XGB).
    Combine les probabilités des deux modèles.
    """
    rf  = build_rf_model()
    xgb_model = build_xgb_model()

    ensemble = VotingClassifier(
        estimators=[("rf", rf), ("xgb", xgb_model)],
        voting="soft",  # Utilise les probabilités
        weights=[1, 1.5],  # XGB légèrement favorisé
    )
    return ensemble
