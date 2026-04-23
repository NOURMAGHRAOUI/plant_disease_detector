"""
Application Web Flask — Interface de diagnostic des maladies des plantes.
"""

import os
import sys
import json
import tempfile
from flask import Flask, render_template, request, jsonify
from PIL import Image

# Ajouter le répertoire parent au path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dl.predict import predict_image, predict_from_pil
from src.ml.predict import predict_tabular
from src.utils.fusion import fuse_predictions, DISEASE_INFO, CLASSES
from src.utils.data_loader import validate_features

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB max

# Cache du modèle DL (chargé une seule fois)
_dl_model = None


def get_dl_model():
    global _dl_model
    if _dl_model is None:
        model_path = os.path.join(os.path.dirname(__file__), "..", "models", "cnn_model.pt")
        if os.path.exists(model_path):
            from src.dl.predict import load_model
            _dl_model = load_model(model_path)
    return _dl_model


@app.route("/")
def index():
    return render_template("index.html", classes=CLASSES, disease_info=DISEASE_INFO)


@app.route("/predict", methods=["POST"])
def predict():
    """
    Endpoint principal de prédiction.
    Accepte : image (fichier) + données agronomiques (JSON form fields)
    """
    try:
        # ── Données agronomiques
        features = {
            "temperature":        float(request.form.get("temperature",     22.0)),
            "humidity":           float(request.form.get("humidity",         60.0)),
            "rainfall":           float(request.form.get("rainfall",         50.0)),
            "soil_ph":            float(request.form.get("soil_ph",           6.5)),
            "nitrogen":           float(request.form.get("nitrogen",          35.0)),
            "phosphorus":         float(request.form.get("phosphorus",        18.0)),
            "potassium":          float(request.form.get("potassium",         25.0)),
            "leaf_area_index":    float(request.form.get("leaf_area_index",    2.5)),
            "plant_age_days":     int(request.form.get("plant_age_days",       60)),
            "previous_infection": int(request.form.get("previous_infection",    0)),
        }
        features = validate_features(features)

        # ── Prédiction ML
        ml_result = predict_tabular(features)

        # ── Prédiction DL (si image fournie)
        dl_result = None
        if "image" in request.files and request.files["image"].filename:
            file = request.files["image"]
            pil_img = Image.open(file.stream).convert("RGB")
            model = get_dl_model()
            if model:
                dl_result = predict_from_pil(pil_img, model)
            else:
                # Pas de modèle entraîné : utiliser dummy
                from src.dl.predict import _dummy_prediction
                dl_result = _dummy_prediction()
        else:
            # Pas d'image : simuler une prédiction DL neutre
            dl_result = {
                "class": ml_result["class"],
                "confidence": 0.5,
                "probabilities": {c: 1/len(CLASSES) for c in CLASSES},
            }

        # ── Fusion
        final = fuse_predictions(dl_result, ml_result)
        return jsonify({"success": True, "result": final})

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/classes")
def get_classes():
    return jsonify({"classes": CLASSES, "info": DISEASE_INFO})


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "0") == "1"
    print(f"🌿 Plant Disease Detector — http://localhost:{port}")
    app.run(host="0.0.0.0", port=port, debug=debug)
