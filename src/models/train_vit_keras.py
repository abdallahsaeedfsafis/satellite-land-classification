"""
Fine-tuning script for the pre-trained ViT (TensorFlow/Keras via HuggingFace) on the
satellite land classification dataset.

Mirrors the PyTorch fine-tuning strategy (train_vit_pytorch.py): freeze the
pre-trained backbone and train only the new classification head, for a fair
framework comparison.
"""

import json
import os
import time

import numpy as np
import tensorflow as tf
from PIL import Image

from vit_keras import build_vit, get_image_processor

SPLIT_PATH = os.path.join("data", "processed", "split.json")
BATCH_SIZE = 32
NUM_EPOCHS = 5
LEARNING_RATE = 0.001
MODEL_SAVE_PATH = os.path.join("outputs", "models", "vit_keras")


def load_split():
    with open(SPLIT_PATH, "r") as f:
        return json.load(f)


def preprocess_records(records, processor):
    """Load and preprocess a list of {path, label} records into (pixel_values, labels) arrays."""
    images = [Image.open(r["path"]).convert("RGB") for r in records]
    pixel_values = processor(images=images, return_tensors="np")["pixel_values"]
    labels = np.array([r["label"] for r in records])
    return pixel_values, labels


def freeze_backbone(model):
    """Freeze all layers except the classifier head."""
    for layer in model.layers:
        if layer.name != "classifier":
            layer.trainable = False
        else:
            layer.trainable = True


def main() -> None:
    print(f"GPU available: {tf.config.list_physical_devices('GPU')}")

    split_data = load_split()
    processor = get_image_processor()

    print("Preprocessing train set...")
    X_train, y_train = preprocess_records(split_data["train"], processor)
    print("Preprocessing validation set...")
    X_val, y_val = preprocess_records(split_data["val"], processor)

    model = build_vit(num_classes=2)
    freeze_backbone(model)

    trainable_params = sum(
        tf.keras.backend.count_params(w) for w in model.trainable_weights
    )
    print(f"Trainable parameters (classifier head only): {trainable_params:,}")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )

    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)

    start = time.time()
    history = model.fit(
        x={"pixel_values": X_train},
        y=y_train,
        validation_data=({"pixel_values": X_val}, y_val),
        epochs=NUM_EPOCHS,
        batch_size=BATCH_SIZE,
    )
    elapsed = time.time() - start

    model.save_pretrained(MODEL_SAVE_PATH)
    best_val_acc = max(history.history["val_accuracy"])
    print(f"\nTraining complete in {elapsed:.1f}s. Best validation accuracy: {best_val_acc:.4f}")
    print(f"Model saved to {MODEL_SAVE_PATH}")


if __name__ == "__main__":
    main()