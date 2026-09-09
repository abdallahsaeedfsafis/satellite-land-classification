# Module 1 — Data Handling

## Dataset Overview

- **Source**: IBM Cloud Object Storage (`images-dataSAT.tar`)
- **Format**: JPEG image tiles, 64x64 pixels
- **Task type**: Binary classification
- **Classes**:
  - `class_0_non_agri` — non-agricultural land
  - `class_1_agri` — agricultural land

## Download & Extraction

Handled via `src/data_handling/download_data.py`, which:
1. Downloads the `.tar` archive (skips if already present)
2. Extracts it into `data/raw/`

Run with:
```bash
python src/data_handling/download_data.py
```

## Dataset Statistics

| Class | Count | Percentage |
|---|---|---|
| `class_0_non_agri` | 3000 | 50.0% |
| `class_1_agri` | 3000 | 50.0% |
| **Total** | **6000** | 100% |

- **Balanced dataset** — no class imbalance handling needed for training.
- **Image size**: 64x64 pixels (RGB, JPEG format).

## Next Steps

- [x] Inspect class distribution (image counts per class)
- [x] Implement memory-based data loader (Keras)
- [x] Implement generator-based data loader (PyTorch)
- [x] Apply data augmentation


## Module 1 — Summary

All Module 1 objectives are complete:
- Dataset downloaded, extracted, and inspected (6000 images, balanced 50/50 across 2 classes)
- Two data loading approaches implemented and compared (see `module1_comparison.md`)
- Data augmentation pipeline implemented (PyTorch/torchvision), validated with a visual preview

**Moving to Module 2**: CNN model development (Keras and PyTorch).