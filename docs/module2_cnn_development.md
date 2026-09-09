# Module 2 — CNN Development

## PyTorch CNN

### Architecture
A custom `SimpleCNN` with 3 convolutional blocks (Conv2d -> ReLU -> MaxPool2d),
followed by a fully connected classifier with dropout (0.3) for regularization.
Total trainable parameters: 1,142,210.

### Training Setup
- Optimizer: Adam (lr=0.001)
- Loss: CrossEntropyLoss
- Epochs: 10
- Batch size: 32
- Data augmentation applied to training set only (flip, rotation, color jitter)
- Trained on GPU (CUDA)

### Results (Validation, during training)
Best validation accuracy: **99.67%** (epoch 10)

### Results (Held-out Test Set)

| Metric | Score |
|---|---|
| Accuracy | 99.67% |
| Precision | 99.77% |
| Recall | 99.55% |
| F1-score | 99.66% |

**Confusion Matrix** (900 test samples):

|  | Predicted: non-agri | Predicted: agri |
|---|---|---|
| **Actual: non-agri** | 459 | 1 |
| **Actual: agri** | 2 | 438 |

Only 3 misclassifications out of 900 test samples.

### Observations
- Very high performance is expected here: this is a binary classification task
  (agricultural vs. non-agricultural), the two classes are visually quite distinct
  in satellite imagery (regular crop patterns vs. irregular terrain), and the dataset
  is perfectly balanced (50/50).
- Training and validation metrics track closely together, with no signs of
  significant overfitting, despite the relatively small dataset (6000 images).

## Next Steps

- [ ] Implement equivalent CNN in Keras
- [ ] Train and evaluate the Keras CNN
- [ ] Compare PyTorch vs. Keras CNN results