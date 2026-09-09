"""
Evaluate the fine-tuned ViT (Keras) on the held-out test set, using the same
900-image split.json test set used for all other models in this project.
"""

import json
import os

import numpy as np
from PIL import Image
from transformers import TFViTForImageClassification
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from vit_keras import get_image_processor

SPLIT_PATH = os.path.join("data", "processed", "split.json")
MODEL_PATH = os.path.join("outputs", "models", "vit_keras")


def main() -> None:
    with open(SPLIT_PATH, "r") as f:
        split_data = json.load(f)

    processor = get_image_processor()
    test_records = split_data["test"]

    print(f"Preprocessing {len(test_records)} test images...")
    images = [Image.open(r["path"]).convert("RGB") for r in test_records]
    pixel_values = processor(images=images, return_tensors="np")["pixel_values"]
    y_true = np.array([r["label"] for r in test_records])

    model = TFViTForImageClassification.from_pretrained(MODEL_PATH)
    outputs = model(pixel_values=pixel_values).logits
    y_pred = np.argmax(outputs, axis=1)

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)

    print(f"\nTest set size: {len(y_true)}")
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