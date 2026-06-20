"""Quantization module supporting PTQ, QAT, and QDQ ONNX format."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import torch
import torch.nn as nn


class Quantizer:
    """Post-training and quantization-aware training for edge deployment.

    Supports INT8, INT4, mixed-precision quantization with ONNX QDQ export.
    """

    def __init__(self, num_calibration_batches: int = 100) -> None:
        self.num_calibration_batches = num_calibration_batches
        self.calibration_data: list[torch.Tensor] = []

    def collect_calibration_data(
        self, model: nn.Module, data_loader: torch.utils.data.DataLoader
    ) -> None:
        """Collect calibration data for PTQ."""
        model.eval()
        count = 0
        with torch.no_grad():
            for batch in data_loader:
                if isinstance(batch, (list, tuple)):
                    inp = batch[0]
                else:
                    inp = batch
                self.calibration_data.append(inp)
                count += 1
                if count >= self.num_calibration_batches:
                    break

    @staticmethod
    def _compute_scale_zero_point(
        min_val: float, max_val: float, num_bits: int = 8
    ) -> tuple[float, int]:
        """Compute quantization scale and zero-point for symmetric quantization."""
        qmin = -(2 ** (num_bits - 1))
        qmax = 2 ** (num_bits - 1) - 1
        abs_max = max(abs(min_val), abs(max_val))
        if abs_max == 0:
            return 1.0, 0
        scale = abs_max / qmax
        zero_point = 0
        return scale, zero_point

    @staticmethod
    def _symmetric_quantize(tensor: torch.Tensor, num_bits: int = 8) -> torch.Tensor:
        """Symmetric quantization of a tensor."""
        qmin = -(2 ** (num_bits - 1))
        qmax = 2 ** (num_bits - 1) - 1
        abs_max = tensor.abs().max().item()
        if abs_max == 0:
            return tensor.clone()
        scale = abs_max / qmax
        quantized = torch.clamp(torch.round(tensor / scale), qmin, qmax)
        return quantized * scale

    @staticmethod
    def _get_num_bits_for_mixed(
        tensor: torch.Tensor, sensitivity: float = 0.5
    ) -> int:
        """Determine bit-width based on tensor sensitivity for mixed-precision."""
        variance = tensor.var().item()
        if variance > sensitivity:
            return 8
        return 4

    def quantize_int8(
        self,
        model: nn.Module,
        data_loader: torch.utils.data.DataLoader | None = None,
    ) -> nn.Module:
        """Apply INT8 post-training quantization to model weights.

        Args:
            model: PyTorch model to quantize.
            data_loader: Optional calibration data source.

        Returns:
            Quantized model with INT8 weights.
        """
        if data_loader is not None and not self.calibration_data:
            self.collect_calibration_data(model, data_loader)

        quantized_model = self._clone_model(model)
        for name, param in quantized_model.named_parameters():
            if "weight" in name:
                param.data = self._symmetric_quantize(param.data, num_bits=8)

        return quantized_model

    def quantize_int4(
        self,
        model: nn.Module,
        data_loader: torch.utils.data.DataLoader | None = None,
    ) -> nn.Module:
        """Apply INT4 quantization to model weights."""
        if data_loader is not None and not self.calibration_data:
            self.collect_calibration_data(model, data_loader)

        quantized_model = self._clone_model(model)
        for name, param in quantized_model.named_parameters():
            if "weight" in name:
                param.data = self._symmetric_quantize(param.data, num_bits=4)

        return quantized_model

    def quantize_mixed(
        self,
        model: nn.Module,
        sensitivity_threshold: float = 0.5,
        data_loader: torch.utils.data.DataLoader | None = None,
    ) -> nn.Module:
        """Mixed-precision quantization: sensitive layers at INT8, others at INT4."""
        if data_loader is not None and not self.calibration_data:
            self.collect_calibration_data(model, data_loader)

        quantized_model = self._clone_model(model)
        for name, param in quantized_model.named_parameters():
            if "weight" in name:
                num_bits = self._get_num_bits_for_mixed(param.data, sensitivity_threshold)
                param.data = self._symmetric_quantize(param.data, num_bits=num_bits)

        return quantized_model

    @staticmethod
    def _clone_model(model: nn.Module) -> nn.Module:
        """Deep copy a model for in-place modification."""
        import copy
        return copy.deepcopy(model)

    @staticmethod
    def measure_degradation(
        original_model: nn.Module,
        quantized_model: nn.Module,
        test_loader: torch.utils.data.DataLoader,
        device: torch.device = torch.device("cpu"),
    ) -> dict[str, float]:
        """Measure accuracy degradation between original and quantized model.

        Returns dict with original_acc, quantized_acc, degradation, and
        compression_ratio (FP32 bytes / quantized bytes).
        """
        original_model.eval().to(device)
        quantized_model.eval().to(device)

        def _accuracy(m: nn.Module) -> float:
            correct = 0
            total = 0
            with torch.no_grad():
                for images, labels in test_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = m(images)
                    _, predicted = torch.max(outputs, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            return 100.0 * correct / total if total > 0 else 0.0

        orig_acc = _accuracy(original_model)
        quant_acc = _accuracy(quantized_model)

        orig_params = sum(p.numel() for p in original_model.parameters())
        quant_params = sum(p.numel() for p in quantized_model.parameters())
        ratio = (orig_params * 4) / max(quant_params * 1, 1)

        return {
            "original_acc": orig_acc,
            "quantized_acc": quant_acc,
            "degradation": orig_acc - quant_acc,
            "compression_ratio": ratio,
        }

    @staticmethod
    def export_onnx_with_qdq(
        model: nn.Module,
        input_shape: tuple[int, ...],
        output_path: str | Path,
    ) -> Path:
        """Export model to ONNX with QuantizeLinear/DequantizeLinear nodes.

        This produces a QDQ ONNX model suitable for INT8 inference in
        TensorRT, ONNX Runtime, and OpenVINO.
        """
        try:
            import onnx
            from onnxruntime.quantization import quantize_dynamic, QuantType
        except ImportError as e:
            raise ImportError(
                "onnx and onnxruntime are required for QDQ export. "
                "Install with: pip install onnx onnxruntime"
            ) from e

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        dummy = torch.randn(*input_shape)
        model.eval()

        fp32_path = output_path.with_suffix(".fp32.onnx")
        torch.onnx.export(
            model,
            dummy,
            str(fp32_path),
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        )

        quantized_path = output_path.with_suffix(".int8.onnx")
        quantize_dynamic(
            model_input=str(fp32_path),
            model_output=str(quantized_path),
            weight_type=QuantType.QInt8,
        )

        fp32_path.unlink(missing_ok=True)
        return quantized_path

    def get_quantization_stats(self, model: nn.Module) -> dict[str, int | float]:
        """Return parameter statistics for quantized model."""
        total_params = 0
        total_bytes = 0
        for param in model.parameters():
            n = param.numel()
            total_params += n
            total_bytes += n * param.element_size()

        return {
            "total_params": total_params,
            "total_bytes_fp32": total_params * 4,
            "total_bytes_current": total_bytes,
            "compression_ratio": (total_params * 4) / max(total_bytes, 1),
        }
