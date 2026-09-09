"""
Fine-tuning script for the pre-trained ViT (PyTorch/HuggingFace) on the
satellite land classification dataset.

Strategy: freeze the pre-trained backbone (feature extractor) and only train
the new classification head, since our dataset (4200 training images) is
small relative to the model's 85M+ parameters. This is a standard and
efficient transfer learning approach.
"""

import json
import os
import time

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image

from vit_pytorch import build_vit, get_image_processor

SPLIT_PATH = os.path.join("data", "processed", "split.json")
BATCH_SIZE = 32
NUM_EPOCHS = 5  # ViT fine-tuning typically needs far fewer epochs than training a CNN from scratch
LEARNING_RATE = 0.001
MODEL_SAVE_PATH = os.path.join("outputs", "models", "vit_pytorch.pt")


class SatelliteViTDataset(Dataset):
    """Loads images and applies the ViT's expected preprocessing (resize to 224x224, normalize)."""

    def __init__(self, records, processor):
        self.records = records
        self.processor = processor

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        record = self.records[idx]
        image = Image.open(record["path"]).convert("RGB")
        pixel_values = self.processor(images=image, return_tensors="pt")["pixel_values"].squeeze(0)
        return pixel_values, record["label"]


def load_split():
    with open(SPLIT_PATH, "r") as f:
        return json.load(f)


def freeze_backbone(model):
    """Freeze all parameters except the classifier head."""
    for name, param in model.named_parameters():
        if "classifier" not in name:
            param.requires_grad = False


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(pixel_values=images).logits
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    return running_loss / total, correct / total


def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(pixel_values=images).logits
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    return running_loss / total, correct / total


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    split_data = load_split()
    processor = get_image_processor()

    train_dataset = SatelliteViTDataset(split_data["train"], processor)
    val_dataset = SatelliteViTDataset(split_data["val"], processor)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = build_vit(num_classes=2).to(device)
    freeze_backbone(model)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters (classifier head only): {trainable_params:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=LEARNING_RATE
    )

    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    best_val_acc = 0.0

    for epoch in range(1, NUM_EPOCHS + 1):
        start = time.time()

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        elapsed = time.time() - start
        print(
            f"Epoch {epoch}/{NUM_EPOCHS} ({elapsed:.1f}s) | "
            f"Train loss: {train_loss:.4f}, acc: {train_acc:.4f} | "
            f"Val loss: {val_loss:.4f}, acc: {val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"  -> New best model saved (val_acc={val_acc:.4f})")

    print(f"\nTraining complete. Best validation accuracy: {best_val_acc:.4f}")


if __name__ == "__main__":
    main()