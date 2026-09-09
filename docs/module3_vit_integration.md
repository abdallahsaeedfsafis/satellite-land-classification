# Module 3 — CNN / Vision Transformer Integration

## Approach

Both PyTorch and TensorFlow/Keras versions use the exact same pre-trained checkpoint
(`google/vit-base-patch16-224` from HuggingFace), ensuring a fair comparison: identical
starting weights (85,800,194 total parameters), identical fine-tuning strategy (backbone
frozen, only the classification head — 1,538 parameters — trained), and evaluation on the
identical 900-image test set (`data/processed/split.json`) used throughout this project.

Images (originally 64x64) were resized to 224x224 to match the pre-trained ViT's expected
input size, as is standard practice in transfer learning.

## Training Setup

- Optimizer: Adam (lr=0.001)
- Loss: CrossEntropy / SparseCategoricalCrossentropy
- Epochs: 5 (fewer than the CNNs, since fine-tuning a frozen pre-trained model typically
  converges faster than training a CNN from scratch)
- Batch size: 32

## Results — PyTorch ViT

### Validation (during training)
Best validation accuracy: **97.11%** (epoch 5)

### Test Set

| Metric | Score |
|---|---|
| Accuracy | 97.22% |
| Precision | 97.70% |
| Recall | 96.59% |
| F1-score | 97.14% |

**Confusion Matrix**: 25 misclassifications out of 900.

## Results — Keras ViT

### Validation (during training)
Best validation accuracy: **97.78%** (epoch 5)

### Test Set

| Metric | Score |
|---|---|
| Accuracy | 97.44% |
| Precision | 97.71% |
| Recall | 97.05% |
| F1-score | 97.38% |

**Confusion Matrix**: 23 misclassifications out of 900.

## Full Comparison: CNN vs. ViT (PyTorch and Keras)

| Model | Accuracy | Precision | Recall | F1-score | Misclassifications (/900) |
|---|---|---|---|---|---|
| CNN (PyTorch) | 99.67% | 99.77% | 99.55% | 99.66% | 3 |
| CNN (Keras) | 99.56% | 99.32% | 99.77% | 99.55% | 4 |
| ViT (PyTorch) | 97.22% | 97.70% | 96.59% | 97.14% | 25 |
| ViT (Keras) | 97.44% | 97.71% | 97.05% | 97.38% | 23 |

## Key Findings

1. **CNNs outperformed the fine-tuned ViT on this task**, despite ViT being a much larger
   model (85.8M parameters vs. 1.14M for the CNN). This is a meaningful and realistic
   finding, not an implementation issue — see explanation below.

2. **Why the CNN won here**:
   - The CNN was trained **from scratch specifically on this dataset** (64x64 satellite
     tiles), while the ViT was pre-trained on natural, high-resolution images (ImageNet)
     and only had its classification head (1,538 of 85.8M parameters) fine-tuned. The
     domain gap between ImageNet photos and low-resolution satellite tiles is significant.
   - Upscaling 64x64 images to 224x224 does not add real information — it stretches
     existing pixels rather than providing genuine high-resolution detail, limiting how
     much the pre-trained ViT's learned features can help.
   - With the backbone frozen, the model has very limited capacity (1,538 trainable
     parameters) to adapt to this new, quite different domain.

3. **PyTorch vs. Keras (within each architecture)**: results were nearly identical between
   frameworks for both CNN and ViT, as expected since they used the same architectures
   (CNN) or literally the same pre-trained weights (ViT). Framework choice did not
   meaningfully affect model quality — only training speed and code ergonomics differed
   (see Module 2 for the CNN-specific comparison, and the note below for ViT training time).

4. **Training time**: ViT fine-tuning was slower than CNN training, and Keras/TensorFlow
   was substantially slower than PyTorch for ViT specifically (~2546s vs. much faster in
   PyTorch), primarily because TensorFlow only ran on CPU in this environment (no native
   Windows GPU support for TensorFlow >= 2.11), while PyTorch used CUDA/GPU throughout.

## Practical Takeaway

For this specific task — small, low-resolution, domain-specific images with a simple,
visually distinct binary classification target — a lightweight CNN trained from scratch
outperformed a much larger pre-trained Vision Transformer fine-tuned via transfer learning.
This illustrates an important general lesson: **bigger pre-trained models are not
automatically better**, especially when there's a significant domain gap between the
pre-training data and the target task, and when only a small portion of the model
(the classifier head) is fine-tuned. A full or partial unfreezing of the ViT backbone,
or a ViT pre-trained on remote-sensing/satellite imagery specifically, would likely close
this gap — a natural direction for future work.