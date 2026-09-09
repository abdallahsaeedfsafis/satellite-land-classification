"""
Export the train/val/test split (computed once, deterministically) to a JSON file.

This decouples the split from any specific framework (PyTorch or Keras), so both
can load the exact same test set for a fair, apples-to-apples comparison,
without either framework needing to import the other's dependencies.

Run this once (in the PyTorch environment, since it reuses dataset_split.py).
"""

import json
import os

from generator_loader import SatelliteDataset
from dataset_split import split_dataset

DATA_DIR = os.path.join("data", "raw", "images_dataSAT")
OUTPUT_PATH = os.path.join("data", "processed", "split.json")


def main() -> None:
    dataset = SatelliteDataset(DATA_DIR)
    train_subset, val_subset, test_subset = split_dataset(dataset)

    def subset_to_records(subset):
        return [
            {"path": dataset.samples[idx][0], "label": dataset.samples[idx][1]}
            for idx in subset.indices
        ]

    split_data = {
        "class_names": dataset.class_names,
        "train": subset_to_records(train_subset),
        "val": subset_to_records(val_subset),
        "test": subset_to_records(test_subset),
    }

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(split_data, f, indent=2)

    print(f"Exported split to {OUTPUT_PATH}")
    print(f"Train: {len(split_data['train'])}, Val: {len(split_data['val'])}, Test: {len(split_data['test'])}")


if __name__ == "__main__":
    main()