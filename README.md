# Satellite Land Classification

Land classification system for agricultural applications using satellite imagery, built using
Convolutional Neural Networks (CNNs) and Vision Transformers (ViT) — inspired by IBM's
"AI Capstone Project with Deep Learning" (Coursera).

## Project Scenario

Acting as an AI Engineer at a fertilizer company, the goal is to classify different terrain types
(crops, forests, water bodies, etc.) from satellite imagery, comparing CNN and Vision Transformer
approaches across Keras and PyTorch.

## Roadmap

- [ ] **Module 1 — Data Handling**: memory-based vs. generator-based loading, data augmentation, custom geospatial data loader
- [ ] **Module 2 — CNN Development**: CNN models in Keras and PyTorch, evaluation and comparison
- [ ] **Module 3 — CNN / Vision Transformer Integration**: fine-tuning pre-trained ViT models, CNN vs. ViT comparison
- [ ] **Module 4 — Final Report**: comparative analysis and final write-up

## Project Structure

satellite-land-classification/
├── data/ # raw & processed data (gitignored)
├── notebooks/ # exploratory notebooks per module
├── src/
│ ├── data_handling/
│ ├── models/
│ └── utils/
├── docs/ # written documentation per module
└── outputs/ # trained models & figures


## Status

Work in progress — built incrementally, module by module.