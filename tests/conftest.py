"""Pytest fixtures for binary DNN tests."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.binarize import Binarizer
from src.models import BinaryCNN, BinaryNet, BinaryResNet


@pytest.fixture
def simple_conv_layer() -> nn.Conv2d:
    """Simple Conv2d layer for testing."""
    return nn.Conv2d(3, 16, 3, padding=1)


@pytest.fixture
def simple_linear_layer() -> nn.Linear:
    """Simple Linear layer for testing."""
    return nn.Linear(256, 10)


@pytest.fixture
def binary_cnn() -> BinaryCNN:
    """BinaryCNN model for testing."""
    return BinaryCNN(num_classes=10)


@pytest.fixture
def binary_net() -> BinaryNet:
    """BinaryNet model for testing."""
    return BinaryNet(num_classes=10)


@pytest.fixture
def binary_resnet() -> BinaryResNet:
    """BinaryResNet model for testing."""
    return BinaryResNet(num_classes=10)


@pytest.fixture
def sample_tensor() -> torch.Tensor:
    """Sample input tensor (batch=4, channels=3, 32x32)."""
    return torch.randn(4, 3, 32, 32)


@pytest.fixture
def sample_weight_tensor() -> torch.Tensor:
    """Sample weight tensor for binarization."""
    return torch.randn(16, 3, 3, 3)


@pytest.fixture
def sample_classifier() -> nn.Linear:
    """Simple classifier for forward pass tests."""
    return nn.Linear(10, 2)


@pytest.fixture
def dummy_data_loader() -> DataLoader:
    """Dummy CIFAR-10-like data loader for testing."""
    images = torch.randn(32, 3, 32, 32)
    labels = torch.randint(0, 10, (32,))
    dataset = TensorDataset(images, labels)
    return DataLoader(dataset, batch_size=8, shuffle=False)


@pytest.fixture
def binarizer_xnor() -> Binarizer:
    """XNOR-Net binarizer."""
    return Binarizer(method="xnor")


@pytest.fixture
def binarizer_binary_connect() -> Binarizer:
    """BinaryConnect binarizer."""
    return Binarizer(method="binary_connect")


@pytest.fixture
def binarizer_ternary() -> Binarizer:
    """Ternary weight binarizer."""
    return Binarizer(method="ternary")
