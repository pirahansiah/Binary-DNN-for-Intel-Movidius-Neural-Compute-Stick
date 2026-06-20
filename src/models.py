"""Pre-built binary neural network architectures for CIFAR-10 and edge deployment."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F


class XNORLinear(nn.Module):
    """XNOR-Net linear layer with binary weights and real scaling."""

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))
        nn.init.kaiming_normal_(self.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        alpha = self.weight.abs().mean(dim=1, keepdim=True)
        binary_weight = (self.weight >= 0).float() * 2 - 1
        return F.linear(x, alpha * binary_weight, self.bias)


class XNORConv2d(nn.Module):
    """XNOR-Net conv2d layer with binary weights and per-channel scaling."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(
            torch.Tensor(out_channels, in_channels, kernel_size, kernel_size)
        )
        self.bias = nn.Parameter(torch.zeros(out_channels))
        self.stride = stride
        self.padding = padding
        nn.init.kaiming_normal_(self.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        alpha = (
            self.weight.abs()
            .view(self.weight.shape[0], -1)
            .mean(dim=1, keepdim=True)
            .view(-1, 1, 1, 1)
        )
        binary_weight = (self.weight >= 0).float() * 2 - 1
        return F.conv2d(x, alpha * binary_weight, self.bias, self.stride, self.padding)


class BinaryNet(nn.Module):
    """BinaryNet with XNOR layers for simple classification."""

    def __init__(self, num_classes: int = 10, input_channels: int = 3) -> None:
        super().__init__()
        self.features = nn.Sequential(
            XNORConv2d(input_channels, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            XNORConv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            XNORConv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            XNORLinear(256 * 4 * 4, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            XNORLinear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class BinaryCNN(nn.Module):
    """Lightweight binary CNN for CIFAR-10."""

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.features = nn.Sequential(
            XNORConv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            XNORConv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            XNORConv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = XNORLinear(256, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class BinaryBasicBlock(nn.Module):
    """Basic residual block with binary convolutions."""

    def __init__(self, channels: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = XNORConv2d(channels, channels, 3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = XNORConv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        self.shortcut = nn.Sequential()
        if stride != 1:
            self.shortcut = nn.Sequential(nn.AvgPool2d(stride, stride))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        return F.relu(out)


class BinaryResNet(nn.Module):
    """Binary ResNet for CIFAR-10 (20-layer variant)."""

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.conv1 = XNORConv2d(3, 16, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(16, 3, stride=1)
        self.layer2 = self._make_layer(32, 3, stride=2)
        self.layer3 = self._make_layer(64, 3, stride=2)
        self.fc = XNORLinear(64, num_classes)

    def _make_layer(self, channels: int, num_blocks: int, stride: int) -> nn.Sequential:
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(BinaryBasicBlock(channels, s))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


def get_model(
    name: Literal["binarynet", "binarycnn", "binaryresnet"] = "binarycnn",
    num_classes: int = 10,
) -> nn.Module:
    """Get a pre-built binary model by name."""
    models = {
        "binarynet": BinaryNet,
        "binarycnn": BinaryCNN,
        "binaryresnet": BinaryResNet,
    }
    if name not in models:
        raise ValueError(f"Unknown model: {name}. Choose from {list(models.keys())}")
    return models[name](num_classes=num_classes)


def save_model(model: nn.Module, path: str | Path) -> Path:
    """Save model weights to file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)
    return path


def load_model(
    model: nn.Module,
    path: str | Path,
    device: torch.device = torch.device("cpu"),
) -> nn.Module:
    """Load model weights from file."""
    state_dict = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    return model.to(device)
