"""
Pipeline d'entraînement complet — DL + ML.
Lance les deux entraînements séquentiellement.
"""

import os
import sys
import json
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main(epochs: int = 10, skip_dl: bool = False, skip_ml: bool = False):
    print("=" * 60)
    print("🌿 Plant Disease Detector — Pipeline d'entraînement")
    print("=" * 60)

    os.makedirs("models", exist_ok=True)
    os.makedirs("data/raw", exist_ok=True)

    # ──────────────────────────────────────────────────────────
    # 1. Entraînement ML
    # ──────────────────────────────────────────────────────────
    if not skip_ml:
        print("\n📊 [1/2] Entraînement du modèle ML (Random Forest + XGBoost)...")
        print("-" * 40)
        from src.ml.train import train as train_ml
        ml_metrics = train_ml(
            csv_path="data/raw/agronomic_data.csv",
            model_out="models/ml_model.pkl",
        )
        print(f"✅ ML terminé — Accuracy test : {ml_metrics['test_accuracy']:.4f}")
    else:
        print("\n⏭️  ML skippé")

    # ──────────────────────────────────────────────────────────
    # 2. Entraînement DL
    # ──────────────────────────────────────────────────────────
    if not skip_dl:
        print(f"\n🧠 [2/2] Entraînement du modèle DL (CNN — {epochs} epochs)...")
        print("-" * 40)
        from src.dl.train import train as train_dl
        dl_history = train_dl(
            data_dir="data/raw",
            model_out="models/cnn_model.pt",
            epochs=epochs,
            batch_size=16,
        )
        best_acc = max(dl_history["val_acc"]) if dl_history["val_acc"] else 0
        print(f"✅ DL terminé — Meilleure val accuracy : {best_acc:.4f}")
    else:
        print("\n⏭️  DL skippé")

    print("\n" + "=" * 60)
    print("🎉 Entraînement complet terminé !")
    print("   Modèles sauvegardés dans models/")
    print("   Pour lancer l'app : python app/app.py")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plant Disease Detector — Entraînement")
    parser.add_argument("--epochs",   type=int,  default=10,    help="Nombre d'époques DL")
    parser.add_argument("--skip-dl",  action="store_true",       help="Sauter l'entraînement DL")
    parser.add_argument("--skip-ml",  action="store_true",       help="Sauter l'entraînement ML")
    args = parser.parse_args()

    main(epochs=args.epochs, skip_dl=args.skip_dl, skip_ml=args.skip_ml)
