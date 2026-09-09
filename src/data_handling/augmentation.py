"""
Data augmentation pipeline (PyTorch/torchvision approach).

Applies random transformations to training images on-the-fly, to improve
model generalization on a relatively small dataset (6000 images).
"""

import os

from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms

from generator_loader import SatelliteDataset

DATA_DIR = os.path.join("data", "raw", "images_dataSAT")
IMAGE_SIZE = (64, 64)
BATCH_SIZE = 32


# Augmentations applied only to training data.
# Each transform introduces a plausible variation a satellite image could
# realistically have (different orientation, lighting), without distorting
# it so much it stops representing real data.
train_transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),      # satellite tiles have no fixed "up" direction
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
])

# No augmentation for validation/test data — we want to evaluate on
# realistic, unmodified images.
eval_transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
])


def save_augmentation_preview(dataset: SatelliteDataset, output_path: str, num_samples: int = 5) -> None:
    """Save a side-by-side preview image showing original vs. augmented versions of a few samples."""
    to_pil = transforms.ToPILImage()
    preview_width = num_samples * IMAGE_SIZE[0]
    preview = Image.new("RGB", (preview_width, IMAGE_SIZE[1] * 2))

    for i in range(num_samples):
        img_path, _ = dataset.samples[i]
        original = Image.open(img_path).convert("RGB").resize(IMAGE_SIZE)
        augmented_tensor = train_transform(original)
        augmented = to_pil(augmented_tensor)

        preview.paste(original, (i * IMAGE_SIZE[0], 0))
        preview.paste(augmented, (i * IMAGE_SIZE[0], IMAGE_SIZE[1]))

    preview.save(output_path)
    print(f"Saved augmentation preview to: {output_path}")


def main() -> None:
    dataset = SatelliteDataset(DATA_DIR, transform=train_transform)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    images, labels = next(iter(dataloader))
    print(f"Augmented batch shape: {images.shape}, labels shape: {labels.shape}")

    os.makedirs(os.path.join("outputs", "figures"), exist_ok=True)
    save_augmentation_preview(
        SatelliteDataset(DATA_DIR),  # no transform, so we get raw PIL images for the original row
        os.path.join("outputs", "figures", "augmentation_preview.png"),
    )


if __name__ == "__main__":
    main()