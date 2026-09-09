"""
Vision Transformer (ViT) model setup (PyTorch/HuggingFace) for binary satellite
land classification, using transfer learning from a pre-trained ViT.
"""

from transformers import ViTForImageClassification, ViTImageProcessor

MODEL_NAME = "google/vit-base-patch16-224"


def build_vit(num_classes: int = 2):
    """
    Load a pre-trained ViT and replace its classification head for our
    binary classification task (fine-tuning approach).
    """
    model = ViTForImageClassification.from_pretrained(
        MODEL_NAME,
        num_labels=num_classes,
        ignore_mismatched_sizes=True,  # replaces the original 1000-class head with our 2-class head
    )
    return model


def get_image_processor():
    """Returns the processor that handles resizing/normalization expected by this ViT."""
    return ViTImageProcessor.from_pretrained(MODEL_NAME)


def main() -> None:
    """Sanity check: load the model and processor, inspect expected input size."""
    processor = get_image_processor()
    print(f"Expected image size: {processor.size}")
    print(f"Normalization mean: {processor.image_mean}")
    print(f"Normalization std: {processor.image_std}")

    model = build_vit(num_classes=2)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")


if __name__ == "__main__":
    main()