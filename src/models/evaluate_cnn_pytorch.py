"""
Evaluate the trained PyTorch CNN on the held-out test set.
Reports accuracy, precision, recall, F1-score, and a confusion matrix.
"""

import os
import sys

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "data_handling"))

from generator_loader import SatelliteDataset
from dataset_split import split_dataset

from cnn_pytorch import SimpleCNN

DATA_DIR = os.path.join("data", "raw", "images_dataSAT")
IMAGE_SIZE = (64, 64)
BATCH_SIZE = 32
MODEL_PATH = os.path.join("outputs", "models", "cnn_pytorch.pt")

eval_transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
])


def get_test_loader():
    base_dataset = SatelliteDataset(DATA_DIR)
    _, _, test_subset = split_dataset(base_dataset)

    test_dataset = SatelliteDataset(DATA_DIR, transform=eval_transform)
    test_subset = Subset(test_dataset, test_subset.indices)

    return DataLoader(test_subset, batch_size=BATCH_SIZE, shuffle=False), base_dataset.class_names


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    test_loader, class_names = get_test_loader()

    model = SimpleCNN(num_classes=2).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds)

    print(f"\nTest set size: {len(all_labels)}")
    print(f"Classes: {class_names}")
    print(f"\nAccuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}")
    print(f"\nConfusion Matrix:")
    print(f"                 Predicted 0   Predicted 1")
    print(f"Actual 0 ({class_names[0]}):   {cm[0][0]:>6}        {cm[0][1]:>6}")
    print(f"Actual 1 ({class_names[1]}):   {cm[1][0]:>6}        {cm[1][1]:>6}")


if __name__ == "__main__":
    main()