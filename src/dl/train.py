"""
Entraînement du modèle Deep Learning (CNN).
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Dataset
from torchvision import transforms
from PIL import Image
import numpy as np
from tqdm import tqdm
import json

from .model import build_model, PlantDiseaseNet, CLASSES, NUM_CLASSES


# ──────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────

class PlantDataset(Dataset):
    """
    Dataset d'images de feuilles.
    Structure attendue :
        data/raw/
            Healthy/        *.jpg
            Rust/           *.jpg
            Black_Spot/     *.jpg
            Powdery_Mildew/ *.jpg
            Bacterial_Blight/ *.jpg
    """

    TRAIN_TRANSFORM = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    VAL_TRANSFORM = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    def __init__(self, data_dir: str, split: str = "train"):
        self.samples = []
        self.transform = self.TRAIN_TRANSFORM if split == "train" else self.VAL_TRANSFORM

        for label_idx, class_name in enumerate(CLASSES):
            class_dir = os.path.join(data_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
            for fname in os.listdir(class_dir):
                if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                    self.samples.append((os.path.join(class_dir, fname), label_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        return self.transform(img), label


# ──────────────────────────────────────────────
# Entraînement
# ──────────────────────────────────────────────

def train(
    data_dir: str = "data/raw",
    model_out: str = "models/cnn_model.pt",
    epochs: int = 20,
    batch_size: int = 32,
    lr: float = 1e-3,
    use_pretrained: bool = True,
    val_split: float = 0.2,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Device: {device}")

    # Dataset
    full_dataset = PlantDataset(data_dir, split="train")
    if len(full_dataset) == 0:
        print("⚠️  Aucune image trouvée. Génération de données synthétiques pour la démo...")
        _generate_synthetic_data(data_dir)
        full_dataset = PlantDataset(data_dir, split="train")

    val_size = int(len(full_dataset) * val_split)
    train_size = len(full_dataset) - val_size
    train_ds, val_ds = random_split(full_dataset, [train_size, val_size])
    val_ds.dataset.transform = PlantDataset.VAL_TRANSFORM

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0)

    print(f"📦  Train: {train_size} images | Val: {val_size} images")

    # Modèle
    model = build_model(NUM_CLASSES, pretrained=use_pretrained).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history = {"train_loss": [], "val_loss": [], "val_acc": []}
    best_acc = 0.0

    for epoch in range(1, epochs + 1):
        # ── Train
        model.train()
        running_loss = 0.0
        for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [Train]"):
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(imgs), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        train_loss = running_loss / len(train_loader)

        # ── Validation
        model.eval()
        val_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                val_loss += criterion(outputs, labels).item()
                correct += (outputs.argmax(1) == labels).sum().item()
                total += labels.size(0)

        val_loss /= len(val_loader)
        val_acc = correct / total
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(f"  Loss train={train_loss:.4f}  val={val_loss:.4f}  acc={val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            os.makedirs(os.path.dirname(model_out), exist_ok=True)
            torch.save(model.state_dict(), model_out)
            print(f"  💾 Meilleur modèle sauvegardé (acc={best_acc:.4f})")

    with open("models/dl_history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"\n✅ Entraînement DL terminé. Meilleure accuracy: {best_acc:.4f}")
    return history


def _generate_synthetic_data(data_dir: str):
    """Génère des images colorées aléatoires pour tester le pipeline."""
    import random
    for class_name in CLASSES:
        class_path = os.path.join(data_dir, class_name)
        os.makedirs(class_path, exist_ok=True)
        for i in range(50):
            arr = np.random.randint(50, 200, (224, 224, 3), dtype=np.uint8)
            img = Image.fromarray(arr)
            img.save(os.path.join(class_path, f"synth_{i:03d}.jpg"))
    print("✅ Données synthétiques générées (50 images × 5 classes)")


if __name__ == "__main__":
    train()
