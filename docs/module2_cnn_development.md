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

## Keras CNN

### Architecture
Identical architecture to the PyTorch model (same layer structure and parameter count:
1,142,210 trainable parameters), built with `tf.keras.Sequential`.

### Training Setup
- Optimizer: Adam (lr=0.001)
- Loss: SparseCategoricalCrossentropy (from logits)
- Epochs: 10
- Batch size: 32
- Data augmentation applied via `ImageDataGenerator` (flip, rotation, brightness)
- Trained on CPU (no GPU detected in this environment)

### Results (Validation, during training)
Best validation accuracy: **99.56%** (epoch 10). Training showed one noisy epoch
(epoch 6: val_accuracy briefly dropped to 92.17%) before recovering — not unusual
during training, and did not affect the final saved (best) checkpoint.

### Results (Held-out Test Set)

Evaluated on the exact same 900 test images used for the PyTorch model
(see `data/processed/split.json`), for a fair comparison.

| Metric | Score |
|---|---|
| Accuracy | 99.56% |
| Precision | 99.32% |
| Recall | 99.77% |
| F1-score | 99.55% |

**Confusion Matrix** (900 test samples):

|  | Predicted: non-agri | Predicted: agri |
|---|---|---|
| **Actual: non-agri** | 457 | 3 |
| **Actual: agri** | 1 | 439 |

## PyTorch vs. Keras — Final Comparison

Both models share the identical architecture (1,142,210 parameters) and were
evaluated on the exact same 900-image test set.

| Metric | PyTorch | Keras |
|---|---|---|
| Accuracy | 99.67% | 99.56% |
| Precision | 99.77% | 99.32% |
| Recall | 99.55% | 99.77% |
| F1-score | 99.66% | 99.55% |
| Training time (10 epochs) | ~44s (GPU) | ~86s (CPU) |
| Misclassifications (out of 900) | 3 | 4 |

### Observations

- **Performance**: Both frameworks produce near-identical, excellent results.
  The tiny differences (0.11% accuracy) fall well within normal run-to-run variance
  given random weight initialization and augmentation randomness — not a meaningful
  difference between the frameworks themselves.
- **Speed**: PyTorch trained roughly 2x faster here, but this reflects that PyTorch
  ran on GPU while Keras/TensorFlow fell back to CPU in this environment (no GPU
  detected by TensorFlow) — an environment/setup difference, not an inherent
  performance difference between the two frameworks.
- **Data splitting**: PyTorch's split was defined explicitly and exported to
  `split.json` (70/15/15 train/val/test). Keras's `ImageDataGenerator` only supports
  a single `validation_split`, so its "validation" set during training actually
  covered 30% of the data; for the final comparison above, both models were
  evaluated on the identical 900-image test set from `split.json`, keeping the
  comparison fair regardless of how each framework handled its internal training split.
- **Developer experience**: PyTorch required more manual code (custom `Dataset`,
  training loop), while Keras's `Sequential` API and `model.fit()` were more concise
  — a well-known trade-off between the two frameworks (PyTorch: more control and
  transparency; Keras: faster to write, more abstracted).

## Module 2 — Summary

Both PyTorch and Keras CNNs achieve ~99.6% accuracy on this binary land classification
task, confirming the models are robust and the task itself is well-suited to CNNs
given the visually distinct classes and balanced dataset.

**Moving to Module 3**: Vision Transformer integration and comparison against these CNN baselines.

## Next Steps

- [x] Implement equivalent CNN in Keras
- [x] Train and evaluate the Keras CNN
- [x] Compare PyTorch vs. Keras CNN results