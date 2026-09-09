"""
Training script for the PyTorch CNN on the satellite land classification dataset.
"""

import os
import sys
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

# Allow importing sibling modules (data_handling) when running this script directly.
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "data_handling"))

from generator_loader import SatelliteDataset
from dataset_split import split_dataset

from cnn_pytorch import SimpleCNN

DATA_DIR = os.path.join("data", "raw", "images_dataSAT")
IMAGE_SIZE = (64, 64)
BATCH_SIZE = 32
NUM_EPOCHS = 10
LEARNING_RATE = 0.001

MODEL_SAVE_PATH = os.path.join("outputs", "models", "cnn_pytorch.pt")

train_transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
])

eval_transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
])


def get_dataloaders():
    """Build train/val DataLoaders, applying augmentation only to the training subset."""
    # Split indices using an untransformed dataset first, to get consistent split assignment.
    base_dataset = SatelliteDataset(DATA_DIR)
    train_subset, val_subset, _ = split_dataset(base_dataset)

    # Re-wrap with the appropriate transform per split (train gets augmentation, val doesn't).
    train_dataset = SatelliteDataset(DATA_DIR, transform=train_transform)
    val_dataset = SatelliteDataset(DATA_DIR, transform=eval_transform)

    train_subset = torch.utils.data.Subset(train_dataset, train_subset.indices)
    val_subset = torch.utils.data.Subset(val_dataset, val_subset.indices)

    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False)

    return train_loader, val_loader


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
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
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    return running_loss / total, correct / total


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, val_loader = get_dataloaders()
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    model = SimpleCNN(num_classes=2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

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