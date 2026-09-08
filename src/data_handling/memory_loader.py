"""
Memory-based data loader (Keras/TensorFlow approach).

Loads the entire dataset into memory as NumPy arrays. Suitable for small
datasets that comfortably fit in RAM (e.g. 6000 small 64x64 images).
"""

import os
import time

import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array

DATA_DIR = os.path.join("data", "raw", "images_dataSAT")
IMAGE_SIZE = (64, 64)


def load_dataset_to_memory(data_dir: str, image_size: tuple = IMAGE_SIZE):
    """
    Load all images from class subfolders into memory.

    Returns:
        X: np.ndarray of shape (num_samples, height, width, 3), pixel values in [0, 1]
        y: np.ndarray of shape (num_samples,), integer class labels
        class_names: list of class folder names, index-aligned with label integers
    """
    class_names = sorted(os.listdir(data_dir))
    images, labels = [], []

    for label_idx, class_name in enumerate(class_names):
        class_path = os.path.join(data_dir, class_name)
        if not os.path.isdir(class_path):
            continue

        for filename in os.listdir(class_path):
            if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            img_path = os.path.join(class_path, filename)
            img = load_img(img_path, target_size=image_size)
            img_array = img_to_array(img) / 255.0  # normalize to [0, 1]
            images.append(img_array)
            labels.append(label_idx)

    X = np.array(images, dtype="float32")
    y = np.array(labels, dtype="int32")
    return X, y, class_names


def main() -> None:
    start = time.time()
    X, y, class_names = load_dataset_to_memory(DATA_DIR)
    elapsed = time.time() - start

    print(f"Loaded dataset into memory in {elapsed:.2f} seconds")
    print(f"Classes: {class_names}")
    print(f"X shape: {X.shape}, dtype: {X.dtype}")
    print(f"y shape: {y.shape}, dtype: {y.dtype}")
    print(f"Memory usage of X: {X.nbytes / (1024 ** 2):.2f} MB")


if __name__ == "__main__":
    main()