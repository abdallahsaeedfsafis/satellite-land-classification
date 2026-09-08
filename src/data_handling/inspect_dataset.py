"""
Inspect the extracted satellite dataset: count images per class
and report basic statistics.
"""

import os
from PIL import Image

DATA_DIR = os.path.join("data", "raw", "images_dataSAT")


def count_images_per_class(data_dir: str) -> dict:
    """Return a dict mapping class folder name -> number of images."""
    counts = {}
    for class_name in sorted(os.listdir(data_dir)):
        class_path = os.path.join(data_dir, class_name)
        if os.path.isdir(class_path):
            images = [f for f in os.listdir(class_path) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
            counts[class_name] = len(images)
    return counts


def get_sample_image_size(data_dir: str, class_name: str) -> tuple:
    """Open the first image in a class folder and return its (width, height)."""
    class_path = os.path.join(data_dir, class_name)
    images = [f for f in os.listdir(class_path) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    if not images:
        return None
    with Image.open(os.path.join(class_path, images[0])) as img:
        return img.size


def main() -> None:
    counts = count_images_per_class(DATA_DIR)
    total = sum(counts.values())

    print("Class distribution:")
    for class_name, count in counts.items():
        percentage = (count / total * 100) if total else 0
        print(f"  {class_name}: {count} images ({percentage:.1f}%)")

    print(f"\nTotal images: {total}")

    if counts:
        first_class = list(counts.keys())[0]
        size = get_sample_image_size(DATA_DIR, first_class)
        print(f"Sample image size ({first_class}): {size}")


if __name__ == "__main__":
    main()