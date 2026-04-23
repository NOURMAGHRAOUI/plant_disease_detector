"""
Tests unitaires pour le pipeline Plant Disease Detector.
"""

import os
import sys
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ──────────────────────────────────────────────
# Tests du module Fusion
# ──────────────────────────────────────────────

class TestFusion:
    def _make_result(self, class_name: str, confidence: float):
        classes = ["Healthy", "Rust", "Black_Spot", "Powdery_Mildew", "Bacterial_Blight"]
        probs = {c: 0.01 for c in classes}
        probs[class_name] = confidence
        # Normalize
        total = sum(probs.values())
        probs = {k: v/total for k, v in probs.items()}
        return {"class": class_name, "confidence": confidence, "probabilities": probs}

    def test_fusion_returns_correct_keys(self):
        from src.utils.fusion import fuse_predictions
        dl = self._make_result("Healthy", 0.9)
        ml = self._make_result("Healthy", 0.85)
        result = fuse_predictions(dl, ml)
        assert "final_class" in result
        assert "confidence" in result
        assert "probabilities" in result
        assert "treatment" in result

    def test_fusion_probabilities_sum_to_one(self):
        from src.utils.fusion import fuse_predictions
        dl = self._make_result("Rust", 0.8)
        ml = self._make_result("Rust", 0.7)
        result = fuse_predictions(dl, ml)
        total = sum(result["probabilities"].values())
        assert abs(total - 1.0) < 1e-5

    def test_fusion_all_classes_present(self):
        from src.utils.fusion import fuse_predictions, CLASSES
        dl = self._make_result("Black_Spot", 0.75)
        ml = self._make_result("Black_Spot", 0.8)
        result = fuse_predictions(dl, ml)
        for cls in CLASSES:
            assert cls in result["probabilities"]

    def test_fusion_weighted(self):
        """DL poids 0.6, ML poids 0.4 → résultat favorise DL."""
        from src.utils.fusion import fuse_predictions
        dl = self._make_result("Healthy", 0.99)
        ml = self._make_result("Rust", 0.99)
        result = fuse_predictions(dl, ml, dl_weight=0.6)
        # Avec poids DL > ML, Healthy devrait gagner
        assert result["final_class"] == "Healthy"


# ──────────────────────────────────────────────
# Tests du module ML
# ──────────────────────────────────────────────

class TestMLModel:
    def test_build_model(self):
        from src.ml.model import build_ensemble_model
        model = build_ensemble_model()
        assert model is not None

    def test_dummy_prediction(self):
        from src.ml.predict import _dummy_prediction
        result = _dummy_prediction()
        assert "class" in result
        assert "probabilities" in result
        assert 0 <= result["confidence"] <= 1

    def test_generate_synthetic_csv(self, tmp_path):
        from src.ml.train import generate_synthetic_csv
        csv_path = str(tmp_path / "test.csv")
        import pandas as pd
        generate_synthetic_csv(csv_path, n_samples=50)
        assert os.path.exists(csv_path)
        df = pd.read_csv(csv_path)
        assert len(df) >= 25
        assert "label" in df.columns

    def test_train_and_predict(self, tmp_path):
        from src.ml.train import train
        from src.ml.predict import predict_tabular
        csv_path = str(tmp_path / "data.csv")
        model_path = str(tmp_path / "model.pkl")

        metrics = train(csv_path=csv_path, model_out=model_path)
        assert "test_accuracy" in metrics
        assert metrics["test_accuracy"] > 0.1  # Au moins mieux que le hasard

        features = {
            "temperature": 25.0, "humidity": 70.0, "rainfall": 60.0,
            "soil_ph": 6.5, "nitrogen": 40.0, "phosphorus": 20.0,
            "potassium": 30.0, "leaf_area_index": 3.0,
            "plant_age_days": 90, "previous_infection": 0,
        }
        result = predict_tabular(features, model_path=model_path)
        assert "class" in result
        assert result["class"] in ["Healthy", "Rust", "Black_Spot", "Powdery_Mildew", "Bacterial_Blight"]


# ──────────────────────────────────────────────
# Tests du module DL
# ──────────────────────────────────────────────

class TestDLModel:
    def test_build_model(self):
        from src.dl.model import build_model, NUM_CLASSES
        model = build_model(NUM_CLASSES, pretrained=False)
        assert model is not None

    def test_custom_net_forward(self):
        import torch
        from src.dl.model import PlantDiseaseNet, NUM_CLASSES
        model = PlantDiseaseNet(NUM_CLASSES)
        x = torch.randn(2, 3, 224, 224)
        out = model(x)
        assert out.shape == (2, NUM_CLASSES)

    def test_dummy_dl_prediction(self):
        from src.dl.predict import _dummy_prediction
        result = _dummy_prediction()
        assert "class" in result
        assert 0 <= result["confidence"] <= 1


# ──────────────────────────────────────────────
# Tests des utilitaires
# ──────────────────────────────────────────────

class TestUtils:
    def test_validate_features(self):
        from src.utils.data_loader import validate_features, FEATURE_COLUMNS
        features = {"temperature": 30.0}
        validated = validate_features(features)
        assert set(validated.keys()) == set(FEATURE_COLUMNS)
        assert validated["temperature"] == 30.0
        assert "humidity" in validated  # Valeur par défaut

    def test_disease_info_complete(self):
        from src.utils.fusion import DISEASE_INFO, CLASSES
        for cls in CLASSES:
            assert cls in DISEASE_INFO
            info = DISEASE_INFO[cls]
            assert "treatment" in info
            assert "severity" in info
