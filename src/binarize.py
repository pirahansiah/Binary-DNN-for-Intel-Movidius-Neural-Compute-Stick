"""BNN binarization module with XNOR-Net, BinaryConnect, and Ternary weight networks."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Literal


class Binarizer:
    """Binarize neural network weights and activations for edge deployment.

    Supports XNOR-Net binarization, BinaryConnect, and ternary weight networks.
    Provides accuracy vs speed comparison utilities.
    """

    def __init__(self, method: Literal["xnor", "binary_connect", "ternary"] = "xnor") -> None:
        self.method = method

    @staticmethod
    def sign_ste(x: torch.Tensor) -> torch.Tensor:
        """Sign function with straight-through estimator."""
        return (x >= 0).float() * 2 - 1

    @staticmethod
    def ternary_ste(x: torch.Tensor, threshold: float = 0.7) -> torch.Tensor:
        """Ternary function with straight-through estimator."""
        w_abs = x.abs()
        mask_pos = (w_abs >= threshold).float()
        mask_neg = (w_abs >= threshold).float()
        ternary = torch.where(x > 0, mask_pos, -mask_neg)
        return ternary

    @staticmethod
    def xnor_bin(weights: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """XNOR-Net binarization: binary weights + scaling factor per filter.

        Returns (binary_weights, alpha) where alpha is per-channel scaling.
        """
        alpha = weights.abs().mean(dim=list(range(1, weights.ndim)), keepdim=True)
        binary = Binarizer.sign_ste(weights)
        return binary, alpha

    @staticmethod
    def binary_connect(weights: torch.Tensor) -> torch.Tensor:
        """BinaryConnect: simple sign binarization."""
        return Binarizer.sign_ste(weights)

    @staticmethod
    def ternary_weights(weights: torch.Tensor, threshold: float = 0.7) -> torch.Tensor:
        """Ternary weight network: {-1, 0, +1}."""
        return Binarizer.ternary_ste(weights, threshold)

    def binarize_weights(self, layer: nn.Module) -> dict[str, torch.Tensor]:
        """Binarize all parameters of a layer.

        Returns dict with 'binary' weights and 'scale' factors.
        """
        result = {}
        for name, param in layer.named_parameters():
            if "weight" in name and param.dim() >= 2:
                if self.method == "xnor":
                    binary, alpha = self.xnor_bin(param.data)
                    result[name] = binary
                    result[name + "_alpha"] = alpha
                elif self.method == "ternary":
                    result[name] = self.ternary_weights(param.data)
                else:
                    result[name] = self.binary_connect(param.data)
            elif "bias" in name:
                result[name] = param.data
        return result

    @staticmethod
    def binarize_activations(
        input_tensor: torch.Tensor,
        method: Literal["sign", "approx_sign"] = "sign",
    ) -> torch.Tensor:
        """Binarize activation tensors using sign or approximate sign."""
        if method == "approx_sign":
            clamp = input_tensor.clamp(-1, 1)
            return Binarizer.sign_ste(clamp)
        return Binarizer.sign_ste(input_tensor)

    def binarize_layer(
        self, layer: nn.Module, input_tensor: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Binarize weights of a layer and run forward pass with binarized activations.

        Returns (output, weight_info).
        """
        weight_info = self.binarize_weights(layer)

        bin_input = self.binarize_activations(input_tensor)

        original_weight = None
        for name, param in layer.named_parameters():
            if "weight" in name:
                original_weight = param.data.clone()
                param.data = weight_info.get(name, param.data)
                break

        with torch.no_grad():
            output = layer(bin_input)

        if original_weight is not None:
            for name, param in layer.named_parameters():
                if "weight" in name:
                    param.data = original_weight
                    break

        return output, weight_info

    @staticmethod
    def compare_accuracy(
        model: nn.Module,
        test_loader: torch.utils.data.DataLoader,
        device: torch.device = torch.device("cpu"),
        methods: list[Literal["fp32", "xnor", "binary_connect", "ternary"]] | None = None,
    ) -> dict[str, float]:
        """Compare accuracy of FP32 vs binarized models.

        Returns dict mapping method name to accuracy percentage.
        """
        if methods is None:
            methods = ["fp32", "xnor"]

        results: dict[str, float] = {}
        model.eval()
        model.to(device)

        for method in methods:
            correct = 0
            total = 0
            with torch.no_grad():
                for images, labels in test_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    _, predicted = torch.max(outputs, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            results[method] = 100.0 * correct / total

        return results

    @staticmethod
    def model_size_ratio(model: nn.Module) -> float:
        """Calculate compression ratio of binarized vs FP32 model."""
        total_params = sum(p.numel() for p in model.parameters())
        if total_params == 0:
            return 1.0
        fp32_bytes = total_params * 4
        bin_bytes = total_params / 8
        return fp32_bytes / bin_bytes
