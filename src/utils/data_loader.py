"""
Utilitaires de chargement et prétraitement des données.
"""

import os
import numpy as np
import pandas as pd
from PIL import Image
from torchvision import transforms

FEATURE_COLUMNS = [
    "temperature", "humidity", "rainfall", "soil_ph",
    "nitrogen", "phosphorus", "potassium",
    "leaf_area_index", "plant_age_days", "previous_infection",
]


def load_image(image_path: str, size: int = 224) -> np.ndarray:
    """Charge et redimensionne une image en tableau numpy (H, W, C)."""
    img = Image.open(image_path).convert("RGB")
    img = img.resize((size, size))
    return np.array(img)


def load_csv(csv_path: str) -> pd.DataFrame:
    """Charge le CSV et vérifie les colonnes requises."""
    df = pd.read_csv(csv_path)
    missing = [col for col in FEATURE_COLUMNS if col not in df.columns]
    if missing:
        print(f"⚠️  Colonnes manquantes : {missing}. Remplissage avec 0.")
        for col in missing:
            df[col] = 0.0
    return df


def validate_features(features: dict) -> dict:
    """Valide et complète les features manquantes."""
    defaults = {
        "temperature": 22.0, "humidity": 60.0, "rainfall": 50.0,
        "soil_ph": 6.5, "nitrogen": 35.0, "phosphorus": 18.0,
        "potassium": 25.0, "leaf_area_index": 2.5,
        "plant_age_days": 60, "previous_infection": 0,
    }
    return {col: features.get(col, defaults[col]) for col in FEATURE_COLUMNS}
