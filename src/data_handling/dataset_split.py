"""
Train/Validation/Test split utility.

Splits dataset indices (not files) into three subsets, so the same images
on disk are reused across splits without duplication. Uses a fixed random
seed for reproducibility.
"""

import os

from torch.utils.data import Subset, random_split
import torch

from generator_loader import SatelliteDataset

DATA_DIR = os.path.join("data", "raw", "images_dataSAT")

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

RANDOM_SEED = 42


def split_dataset(dataset: SatelliteDataset):
    """
    Split a dataset into train/val/test subsets by ratio.

    Returns:
        (train_subset, val_subset, test_subset)
    """
    total = len(dataset)
    train_size = int(total * TRAIN_RATIO)
    val_size = int(total * VAL_RATIO)
    test_size = total - train_size - val_size  # remainder, avoids rounding gaps

    generator = torch.Generator().manual_seed(RANDOM_SEED)
    train_subset, val_subset, test_subset = random_split(
        dataset, [train_size, val_size, test_size], generator=generator
    )
    return train_subset, val_subset, test_subset


def main() -> None:
    dataset = SatelliteDataset(DATA_DIR)
    train_subset, val_subset, test_subset = split_dataset(dataset)

    print(f"Total samples: {len(dataset)}")
    print(f"Train: {len(train_subset)} ({len(train_subset)/len(dataset)*100:.1f}%)")
    print(f"Validation: {len(val_subset)} ({len(val_subset)/len(dataset)*100:.1f}%)")
    print(f"Test: {len(test_subset)} ({len(test_subset)/len(dataset)*100:.1f}%)")


if __name__ == "__main__":
    main()