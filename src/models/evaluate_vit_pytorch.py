"""
Evaluate the fine-tuned ViT (PyTorch) on the held-out test set.
"""

import json
import os

import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from vit_pytorch import build_vit, get_image_processor
from train_vit_pytorch import SatelliteViTDataset, SPLIT_PATH

MODEL_PATH = os.path.join("outputs", "models", "vit_pytorch.pt")
BATCH_SIZE = 32


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    with open(SPLIT_PATH, "r") as f:
        split_data = json.load(f)

    processor = get_image_processor()
    test_dataset = SatelliteViTDataset(split_data["test"], processor)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = build_vit(num_classes=2).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(pixel_values=images).logits
            _, predicted = torch.max(outputs, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds)

    print(f"\nTest set size: {len(all_labels)}")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}")
    print(f"\nConfusion Matrix:")
    print(f"                 Predicted 0   Predicted 1")
    print(f"Actual 0:   {cm[0][0]:>6}        {cm[0][1]:>6}")
    print(f"Actual 1:   {cm[1][0]:>6}        {cm[1][1]:>6}")


if __name__ == "__main__":
    main()