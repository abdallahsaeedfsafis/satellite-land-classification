# Satellite Land Classification

Land classification system for agricultural applications using satellite imagery, built using
Convolutional Neural Networks (CNNs) and Vision Transformers (ViT) — inspired by IBM's
"AI Capstone Project with Deep Learning" (Coursera).

## Project Scenario

Acting as an AI Engineer at a fertilizer company, the goal is to classify different terrain types
(crops, forests, water bodies, etc.) from satellite imagery, comparing CNN and Vision Transformer
approaches across Keras and PyTorch.

## Roadmap

- [x] **Module 1 — Data Handling**: memory-based vs. generator-based loading, data augmentation, custom geospatial data loader
- [ ] **Module 2 — CNN Development**: CNN models in Keras and PyTorch, evaluation and comparison
- [ ] **Module 3 — CNN / Vision Transformer Integration**: fine-tuning pre-trained ViT models, CNN vs. ViT comparison
- [ ] **Module 4 — Final Report**: comparative analysis and final write-up

## Project Structure

```
satellite-land-classification/
├── data/               # raw & processed data (gitignored)
├── notebooks/          # exploratory notebooks per module
├── src/
│   ├── data_handling/
│   ├── models/
│   └── utils/
├── docs/               # written documentation per module
└── outputs/            # trained models & figures
```

## Status

Work in progress — built incrementally, module by module.

## Environments

This project separates TensorFlow and PyTorch work into two environments to avoid
dependency conflicts between the two frameworks:

- **TensorFlow/Keras environment** — used for Keras-based scripts (e.g. `memory_loader.py`).
  Requires: `tensorflow`, `Pillow`
- **PyTorch environment** — used for PyTorch-based scripts (e.g. `generator_loader.py`,
  CNN/ViT models). Requires: `torch`, `torchvision`, `Pillow`

You can create these as two separate virtual environments (venv/conda) named however you like:

```bash
# TensorFlow environment
python -m venv venv-tf
pip install tensorflow Pillow requests

# PyTorch environment
python -m venv venv-torch
pip install torch torchvision Pillow requests
```

Activate the relevant environment depending on which script you're running.