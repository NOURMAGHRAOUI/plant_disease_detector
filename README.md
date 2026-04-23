# 🌿 Plant Disease Detector — ML + DL Hybrid System

Système de détection de maladies des plantes combinant **Deep Learning (CNN)** pour l'analyse d'images et **Machine Learning classique (Random Forest / XGBoost)** pour les données agronomiques tabulaires.

---

## 🏗️ Architecture

```
plant_disease_detector/
├── data/
│   ├── raw/             # Images et CSV bruts
│   └── processed/       # Données prétraitées
├── models/              # Modèles sauvegardés (.pkl, .pt)
├── notebooks/           # Jupyter notebooks d'exploration
├── src/
│   ├── dl/              # Deep Learning (CNN PyTorch)
│   │   ├── model.py
│   │   ├── train.py
│   │   └── predict.py
│   ├── ml/              # Machine Learning classique
│   │   ├── model.py
│   │   ├── train.py
│   │   └── predict.py
│   └── utils/           # Utilitaires communs
│       ├── data_loader.py
│       └── fusion.py    # Fusion ML + DL
├── app/
│   ├── app.py           # API Flask
│   └── templates/
│       └── index.html   # Interface Web
├── tests/
│   └── test_pipeline.py
├── requirements.txt
├── train_pipeline.py    # Script principal d'entraînement
└── README.md
```

---

## 🚀 Installation

```bash
# 1. Cloner / extraire le projet
cd plant_disease_detector

# 2. Créer un environnement virtuel
python -m venv venv
source venv/bin/activate        # Linux/Mac
# venv\Scripts\activate         # Windows

# 3. Installer les dépendances
pip install -r requirements.txt
```

---

## 🎯 Utilisation

### Entraînement complet
```bash
python train_pipeline.py
```

### Lancer l'application web
```bash
python app/app.py
# → http://localhost:5000
```

### Prédiction en ligne de commande
```bash
python src/utils/fusion.py --image data/raw/leaf.jpg --csv data/raw/sample.csv
```

---

## 🧠 Modèles

| Module | Technologie | Entrée | Sortie |
|--------|------------|--------|--------|
| **DL** | CNN (ResNet-18, PyTorch) | Image de feuille | Probabilités par classe |
| **ML** | Random Forest + XGBoost | Données agronomiques | Score de maladie |
| **Fusion** | Weighted Average | Proba DL + Score ML | Diagnostic final |

### Classes de maladies détectées
- ✅ Plante saine
- 🍂 Rouille (Rust)
- ⚫ Tache noire (Black Spot)
- 🟤 Mildiou (Powdery Mildew)
- 🔴 Brûlure bactérienne (Bacterial Blight)

---

## 📊 Pipeline

```
Image JPG ──► [Preprocessing] ──► [CNN ResNet-18] ──► Proba DL ──┐
                                                                    ├──► [Fusion] ──► Diagnostic
CSV données ──► [Feature Eng.] ──► [Random Forest] ──► Score ML ──┘
```

---

## 🧪 Tests

```bash
python -m pytest tests/
```

---

## 📦 Dépendances principales

- `torch` + `torchvision` — Deep Learning (CNN)
- `scikit-learn` — Random Forest, métriques
- `xgboost` — Gradient Boosting
- `flask` — API Web
- `Pillow` — Traitement d'images
- `pandas` / `numpy` — Manipulation de données
