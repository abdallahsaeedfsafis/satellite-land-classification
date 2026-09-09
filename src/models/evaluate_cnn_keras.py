"""
Evaluate the trained Keras CNN on the held-out test set, loaded from the
framework-independent split.json (see export_split.py), ensuring a fair,
apples-to-apples comparison against the PyTorch model on identical test samples.
"""

import json
import os

import numpy as np
import tensorflow as tf
from PIL import Image
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

IMAGE_SIZE = (64, 64)
MODEL_PATH = os.path.join("outputs", "models", "cnn_keras.keras")
SPLIT_PATH = os.path.join("data", "processed", "split.json")


def load_test_set():
    """Load the test split from split.json and return images/labels as NumPy arrays."""
    with open(SPLIT_PATH, "r") as f:
        split_data = json.load(f)

    images, labels = [], []
    for record in split_data["test"]:
        img = Image.open(record["path"]).convert("RGB").resize(IMAGE_SIZE)
        images.append(np.array(img, dtype="float32") / 255.0)
        labels.append(record["label"])

    return np.array(images), np.array(labels), split_data["class_names"]


def main() -> None:
    X_test, y_test, class_names = load_test_set()
    print(f"Test set size: {len(y_test)}")
    print(f"Classes: {class_names}")

    model = tf.keras.models.load_model(MODEL_PATH)
    logits = model.predict(X_test, verbose=0)
    y_pred = np.argmax(logits, axis=1)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

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