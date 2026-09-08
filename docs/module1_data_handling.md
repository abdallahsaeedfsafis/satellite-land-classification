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

## Next Steps

- [ ] Inspect class distribution (image counts per class)
- [ ] Implement memory-based data loader (Keras)
- [ ] Implement generator-based data loader (PyTorch)
- [ ] Apply data augmentation