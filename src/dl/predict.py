"""
Prédiction Deep Learning — charge le modèle CNN et prédit sur une image.
"""

import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np

from .model import build_model, PlantDiseaseNet, CLASSES, NUM_CLASSES


TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])


def load_model(model_path: str = "models/cnn_model.pt") -> torch.nn.Module:
    """Charge le modèle CNN depuis le fichier .pt."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(NUM_CLASSES, pretrained=False)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model.to(device)


def predict_image(image_path: str, model: torch.nn.Module = None, model_path: str = "models/cnn_model.pt") -> dict:
    """
    Prédit la maladie d'une plante à partir d'une image.

    Args:
        image_path: Chemin vers l'image (JPG/PNG)
        model: Modèle PyTorch (optionnel, sinon chargé depuis model_path)
        model_path: Chemin vers les poids sauvegardés

    Returns:
        dict avec 'class', 'confidence', 'probabilities'
    """
    import os
    if model is None:
        if not os.path.exists(model_path):
            return _dummy_prediction()
        model = load_model(model_path)

    device = next(model.parameters()).device

    img = Image.open(image_path).convert("RGB")
    tensor = TRANSFORM(img).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs = F.softmax(logits, dim=1).squeeze().cpu().numpy()

    predicted_idx = int(np.argmax(probs))
    return {
        "class": CLASSES[predicted_idx],
        "confidence": float(probs[predicted_idx]),
        "probabilities": {cls: float(p) for cls, p in zip(CLASSES, probs)},
    }


def predict_from_pil(pil_image: Image.Image, model: torch.nn.Module) -> dict:
    """Prédiction directe depuis un objet PIL Image."""
    device = next(model.parameters()).device
    tensor = TRANSFORM(pil_image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.nn.functional.softmax(logits, dim=1).squeeze().cpu().numpy()

    predicted_idx = int(np.argmax(probs))
    return {
        "class": CLASSES[predicted_idx],
        "confidence": float(probs[predicted_idx]),
        "probabilities": {cls: float(p) for cls, p in zip(CLASSES, probs)},
    }


def _dummy_prediction() -> dict:
    """Prédiction aléatoire quand le modèle n'est pas encore entraîné."""
    probs = np.random.dirichlet(np.ones(NUM_CLASSES))
    idx = int(np.argmax(probs))
    return {
        "class": CLASSES[idx],
        "confidence": float(probs[idx]),
        "probabilities": {cls: float(p) for cls, p in zip(CLASSES, probs)},
        "note": "Modèle non entraîné — prédiction aléatoire"
    }
