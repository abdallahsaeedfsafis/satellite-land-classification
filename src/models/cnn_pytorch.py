"""
CNN model architecture (PyTorch) for binary satellite land classification
(agricultural vs. non-agricultural).
"""

import torch
import torch.nn as nn


class SimpleCNN(nn.Module):
    """
    A small CNN suitable for 64x64 RGB images and a binary classification task.

    Architecture: 3 convolutional blocks (Conv -> ReLU -> MaxPool) followed by
    a fully connected classifier head.
    """

    def __init__(self, num_classes: int = 2):
        super().__init__()

        self.conv_block1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # 64x64 -> 32x32
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # 32x32 -> 16x16
        )
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # 16x16 -> 8x8
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 128),
            nn.ReLU(),
            nn.Dropout(0.3),  # reduces overfitting on our relatively small dataset
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.classifier(x)
        return x


def main() -> None:
    """Quick sanity check: verify the model runs and produces the expected output shape."""
    model = SimpleCNN(num_classes=2)
    dummy_input = torch.randn(4, 3, 64, 64)  # batch of 4 fake RGB images
    output = model(dummy_input)

    total_params = sum(p.numel() for p in model.parameters())

    print(model)
    print(f"\nOutput shape: {output.shape}  (expected: [4, 2])")
    print(f"Total trainable parameters: {total_params:,}")


if __name__ == "__main__":
    main()