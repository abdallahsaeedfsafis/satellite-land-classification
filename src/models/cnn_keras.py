"""
CNN model architecture (Keras/TensorFlow) for binary satellite land classification
(agricultural vs. non-agricultural).

Mirrors the PyTorch architecture in cnn_pytorch.py for a fair framework comparison:
3 convolutional blocks (Conv -> ReLU -> MaxPool) followed by a dense classifier head.
"""

from tensorflow.keras import layers, models


def build_cnn(input_shape: tuple = (64, 64, 3), num_classes: int = 2) -> models.Model:
    """Build and return an uncompiled Keras CNN model."""
    model = models.Sequential([
        layers.Input(shape=input_shape),

        layers.Conv2D(32, kernel_size=3, padding="same", activation="relu"),
        layers.MaxPooling2D(pool_size=2),  # 64x64 -> 32x32

        layers.Conv2D(64, kernel_size=3, padding="same", activation="relu"),
        layers.MaxPooling2D(pool_size=2),  # 32x32 -> 16x16

        layers.Conv2D(128, kernel_size=3, padding="same", activation="relu"),
        layers.MaxPooling2D(pool_size=2),  # 16x16 -> 8x8

        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.3),  # matches the PyTorch model's dropout rate
        layers.Dense(num_classes),  # logits, no activation (matches CrossEntropyLoss usage in PyTorch)
    ])
    return model


def main() -> None:
    """Quick sanity check: verify the model builds and produces the expected output shape."""
    model = build_cnn(input_shape=(64, 64, 3), num_classes=2)
    model.summary()

    total_params = model.count_params()
    print(f"\nTotal trainable parameters: {total_params:,}")


if __name__ == "__main__":
    main()