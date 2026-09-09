"""
Vision Transformer (ViT) model setup (TensorFlow/Keras via HuggingFace) for
binary satellite land classification, using transfer learning from the same
pre-trained ViT checkpoint used in the PyTorch version (vit_pytorch.py),
for a fair, apples-to-apples framework comparison.
"""

from transformers import TFViTForImageClassification, ViTImageProcessor

MODEL_NAME = "google/vit-base-patch16-224"


def build_vit(num_classes: int = 2):
    """Load the same pre-trained ViT checkpoint as the PyTorch version, with a new classification head."""
    model = TFViTForImageClassification.from_pretrained(
        MODEL_NAME,
        num_labels=num_classes,
        ignore_mismatched_sizes=True,
    )
    return model


def get_image_processor():
    return ViTImageProcessor.from_pretrained(MODEL_NAME)


def main() -> None:
    processor = get_image_processor()
    print(f"Expected image size: {processor.size}")

    model = build_vit(num_classes=2)
    total_params = model.count_params()
    print(f"\nTotal parameters: {total_params:,}")


if __name__ == "__main__":
    main()