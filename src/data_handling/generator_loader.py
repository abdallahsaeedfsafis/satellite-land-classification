"""
Generator-based data loader (PyTorch approach).

Uses a custom Dataset + DataLoader to load images on-the-fly (lazily),
instead of loading the entire dataset into memory upfront. More suitable
for large datasets that don't fit comfortably in RAM.
"""

import os
import time

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

DATA_DIR = os.path.join("data", "raw", "images_dataSAT")
IMAGE_SIZE = (64, 64)
BATCH_SIZE = 32


class SatelliteDataset(Dataset):
    """Lazily loads images from class subfolders, one at a time, on __getitem__."""

    def __init__(self, data_dir: str, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.class_names = sorted(os.listdir(data_dir))
        self.samples = []  # list of (image_path, label_idx)

        for label_idx, class_name in enumerate(self.class_names):
            class_path = os.path.join(data_dir, class_name)
            if not os.path.isdir(class_path):
                continue
            for filename in os.listdir(class_path):
                if filename.lower().endswith((".jpg", ".jpeg", ".png")):
                    self.samples.append((os.path.join(class_path, filename), label_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")  # loaded from disk only when requested
        if self.transform:
            image = self.transform(image)
        return image, label


def main() -> None:
    transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),  # scales pixel values to [0, 1] automatically
    ])

    start = time.time()
    dataset = SatelliteDataset(DATA_DIR, transform=transform)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    setup_time = time.time() - start

    print(f"Dataset indexed in {setup_time:.4f} seconds (no images loaded yet)")
    print(f"Classes: {dataset.class_names}")
    print(f"Total samples: {len(dataset)}")

    # Time how long it takes to iterate through one full epoch (this is when images are actually read)
    start = time.time()
    for images, labels in dataloader:
        pass
    epoch_time = time.time() - start

    print(f"One full epoch iteration took {epoch_time:.2f} seconds")
    print(f"Sample batch shape: {images.shape}, labels shape: {labels.shape}")


if __name__ == "__main__":
    main()