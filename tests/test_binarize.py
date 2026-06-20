"""Tests for binarization module."""

from __future__ import annotations

import torch
import torch.nn as nn

from src.binarize import Binarizer


class TestBinarizerSignSte:
    def test_positive_values(self):
        x = torch.tensor([1.0, 2.0, 3.0])
        result = Binarizer.sign_ste(x)
        assert torch.all(result == 1.0)

    def test_negative_values(self):
        x = torch.tensor([-1.0, -2.0, -3.0])
        result = Binarizer.sign_ste(x)
        assert torch.all(result == -1.0)

    def test_zero(self):
        x = torch.tensor([0.0])
        result = Binarizer.sign_ste(x)
        assert result.item() == 1.0

    def test_mixed(self):
        x = torch.tensor([1.0, -1.0, 0.5, -0.5])
        result = Binarizer.sign_ste(x)
        expected = torch.tensor([1.0, -1.0, 1.0, -1.0])
        assert torch.all(result == expected)


class TestTernarySte:
    def test_ternary_values(self):
        x = torch.tensor([1.0, -1.0, 0.1, -0.1, 0.8, -0.8])
        result = Binarizer.ternary_ste(x, threshold=0.7)
        assert result[0].item() == 1.0
        assert result[1].item() == -1.0
        assert result[2].item() == 0.0
        assert result[3].item() == 0.0
        assert result[4].item() == 1.0
        assert result[5].item() == -1.0


class TestXnorBin:
    def test_output_shape(self):
        w = torch.randn(16, 3, 3, 3)
        binary, alpha = Binarizer.xnor_bin(w)
        assert binary.shape == w.shape
        assert alpha.shape[0] == 16

    def test_binary_values(self):
        w = torch.randn(8, 4)
        binary, alpha = Binarizer.xnor_bin(w)
        unique = torch.unique(binary)
        assert all(v in [-1.0, 1.0] for v in unique.tolist())

    def test_alpha_positive(self):
        w = torch.randn(16, 3, 3, 3)
        _, alpha = Binarizer.xnor_bin(w)
        assert torch.all(alpha > 0)


class TestBinaryConnect:
    def test_output_is_binary(self):
        w = torch.randn(16, 3, 3, 3)
        result = Binarizer.binary_connect(w)
        unique = torch.unique(result)
        assert all(v in [-1.0, 1.0] for v in unique.tolist())


class TestTernaryWeights:
    def test_output_range(self):
        w = torch.randn(16, 3)
        result = Binarizer.ternary_weights(w, threshold=0.5)
        unique = torch.unique(result)
        assert all(v in [-1.0, 0.0, 1.0] for v in unique.tolist())


class TestBinarizeWeights:
    def test_xnor_method(self, simple_conv_layer):
        b = Binarizer(method="xnor")
        result = b.binarize_weights(simple_conv_layer)
        assert "weight" in result
        assert "weight_alpha" in result

    def test_ternary_method(self, simple_conv_layer):
        b = Binarizer(method="ternary")
        result = b.binarize_weights(simple_conv_layer)
        assert "weight" in result
        unique = torch.unique(result["weight"])
        assert all(v in [-1.0, 0.0, 1.0] for v in unique.tolist())

    def test_binary_connect_method(self, simple_conv_layer):
        b = Binarizer(method="binary_connect")
        result = b.binarize_weights(simple_conv_layer)
        assert "weight" in result
        unique = torch.unique(result["weight"])
        assert all(v in [-1.0, 1.0] for v in unique.tolist())


class TestBinarizeActivations:
    def test_sign(self):
        x = torch.tensor([0.5, -0.5, 1.0, -1.0])
        result = Binarizer.binarize_activations(x, method="sign")
        expected = torch.tensor([1.0, -1.0, 1.0, -1.0])
        assert torch.all(result == expected)

    def test_approx_sign(self):
        x = torch.tensor([0.1, 2.0, -0.1, -2.0])
        result = Binarizer.binarize_activations(x, method="approx_sign")
        assert result[0].item() == 1.0
        assert result[1].item() == 1.0
        assert result[2].item() == -1.0
        assert result[3].item() == -1.0


class TestBinarizeLayer:
    def test_binarize_conv(self, simple_conv_layer, sample_tensor):
        b = Binarizer(method="xnor")
        output, info = b.binarize_layer(simple_conv_layer, sample_tensor)
        assert output.shape[0] == 4
        assert "weight" in info


class TestModelSizeRatio:
    def test_ratio_greater_than_one(self, simple_conv_layer):
        ratio = Binarizer.model_size_ratio(simple_conv_layer)
        assert ratio == 32.0

    def test_zero_params(self):
        model = nn.Linear(0, 0)
        ratio = Binarizer.model_size_ratio(model)
        assert ratio == 1.0
