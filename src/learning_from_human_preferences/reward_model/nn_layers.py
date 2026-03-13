"""
Modern PyTorch layer wrappers replacing the original TensorFlow layers.

These wrappers preserve the same logical order used in the original code:

Conv layer:
    Conv2D → BatchNorm (optional) → LeakyReLU

Dense layer:
    Linear → Activation (optional)

LeakyReLU uses alpha=0.01 exactly like the original TensorFlow implementation.
"""

import torch
import torch.nn as nn


class ConvLayer(nn.Module):
    """
    Convolutional layer wrapper with optional BatchNorm and activation.

    Original TensorFlow order:
        conv → batchnorm → activation
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size,
        stride,
        batchnorm: bool = False,
        activation: str = "relu",
    ):
        super().__init__()

        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
        )

        self.use_batchnorm = batchnorm

        if batchnorm:
            self.batchnorm = nn.BatchNorm2d(out_channels)

        if activation == "relu":
            # Original code actually used LeakyReLU(alpha=0.01)
            self.activation = nn.LeakyReLU(negative_slope=0.01)
        elif activation is None:
            self.activation = None
        else:
            raise ValueError(f"Unknown activation type: {activation}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = self.conv(x)

        if self.use_batchnorm:
            x = self.batchnorm(x)

        if self.activation is not None:
            x = self.activation(x)

        return x


class DenseLayer(nn.Module):
    """
    Fully connected layer wrapper.

    Original TensorFlow order:
        dense → activation
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        activation: str | None = None,
    ):
        super().__init__()

        self.linear = nn.Linear(in_features, out_features)

        if activation == "relu":
            self.activation = nn.LeakyReLU(negative_slope=0.01)
        elif activation is None:
            self.activation = None
        else:
            raise ValueError(f"Unknown activation type: {activation}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = self.linear(x)

        if self.activation is not None:
            x = self.activation(x)

        return x