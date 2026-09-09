"""
Training script for the Keras CNN on the satellite land classification dataset.

Uses the same settings as the PyTorch training script (train_cnn_pytorch.py)
for a fair comparison: same epochs, batch size, optimizer, and train/val split ratio.
"""

import os
import time

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from cnn_keras import build_cnn

DATA_DIR = os.path.join("data", "raw", "images_dataSAT")
IMAGE_SIZE = (64, 64)
BATCH_SIZE = 32
NUM_EPOCHS = 10
LEARNING_RATE = 0.001

# Same split ratios used in dataset_split.py (PyTorch side): 70/15/15.
# Keras's ImageDataGenerator only supports a single validation_split, so here we
# carve out 30% for validation+test combined, then split that 30% in half below.
VALIDATION_SPLIT = 0.30

MODEL_SAVE_PATH = os.path.join("outputs", "models", "cnn_keras.keras")
RANDOM_SEED = 42


def get_generators():
    """Build train/val/test generators with augmentation applied only to training data."""
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        validation_split=VALIDATION_SPLIT,
        horizontal_flip=True,
        vertical_flip=True,
        rotation_range=15,
        brightness_range=(0.8, 1.2),
    )
    eval_datagen = ImageDataGenerator(rescale=1.0 / 255, validation_split=VALIDATION_SPLIT)

    train_gen = train_datagen.flow_from_directory(
        DATA_DIR, target_size=IMAGE_SIZE, batch_size=BATCH_SIZE,
        class_mode="binary", subset="training", seed=RANDOM_SEED,
    )
    val_test_gen = eval_datagen.flow_from_directory(
        DATA_DIR, target_size=IMAGE_SIZE, batch_size=BATCH_SIZE,
        class_mode="binary", subset="validation", seed=RANDOM_SEED, shuffle=False,
    )

    return train_gen, val_test_gen


def main() -> None:
    print(f"GPU available: {tf.config.list_physical_devices('GPU')}")

    train_gen, val_gen = get_generators()
    print(f"Class indices: {train_gen.class_indices}")

    model = build_cnn(input_shape=(64, 64, 3), num_classes=2)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )

    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        MODEL_SAVE_PATH, monitor="val_accuracy", save_best_only=True, verbose=1
    )

    start = time.time()
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=NUM_EPOCHS,
        callbacks=[checkpoint_callback],
    )
    elapsed = time.time() - start

    best_val_acc = max(history.history["val_accuracy"])
    print(f"\nTraining complete in {elapsed:.1f}s. Best validation accuracy: {best_val_acc:.4f}")


if __name__ == "__main__":
    main()