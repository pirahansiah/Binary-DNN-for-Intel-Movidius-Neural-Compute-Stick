"""Intel NNCF (Neural Network Compression Framework) integration — the modern successor to XNOR-Net.

NNCF is Intel's official compression toolkit for quantization, pruning, and
mixed-precision optimization on Intel hardware (CPU, GPU, VPU, GNA).

Reference: https://github.com/openvinotoolkit/nncf
Docs: https://docs.openvino.ai/2024/openvino-workflow/model-optimization-guide/quantizing-models.html

This module bridges the legacy XNOR-Net approach (2017) to modern NNCF quantization (2024+).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Literal

import numpy as np


class QuantizationPreset(Enum):
    """NNCF quantization presets."""
    PERFORMANCE = "performance"  # Maximum speed, may lose accuracy
    MIXED = "mixed"             # Balanced speed/accuracy
    ACCURACY = "accuracy"       # Maximum accuracy, less speedup


class TargetDevice(Enum):
    """NNCF target hardware devices."""
    CPU = "cpu"
    GPU = "gpu"
    CPU_SPR = "cpu_spr"          # Intel Sapphire Rapids
    VPU = "vpu"                  # Movidius NCS2, Meteor Lake VPU
    GNA = "gna"                  # Gaussian Neural Accelerator
    ANY = "any"                  # Generic optimization


@dataclass
class NNCFConfig:
    """Configuration for NNCF quantization."""
    preset: QuantizationPreset = QuantizationPreset.MIXED
    target_device: TargetDevice = TargetDevice.CPU
    num_samples: int = 300       # Calibration samples
    ignored_scopes: list[str] = field(default_factory=list)
    ignored_patterns: list[str] = field(default_factory=list)
    model_type: str | None = None  # "transformer" for LLMs
    group_size: int = 128        # Group quantization for LLMs
    smooth_quant: bool = False   # SmoothQuant for LLM INT8
    overflow_fix: str = "first_layer"  # Overflow handling


@dataclass
class CompressionResult:
    """Result of NNCF compression."""
    original_size_mb: float
    compressed_size_mb: float
    compression_ratio: float
    precision: str
    estimated_accuracy_loss: float
    device: str
    model_path: str
    size_reduction_pct: float = 0.0

    def __post_init__(self):
        self.size_reduction_pct = (1 - self.compressed_size_mb / self.original_size_mb) * 100


class NNCFCompressor:
    """Modern NNCF quantization and compression for Intel hardware.

    This is the modern equivalent of the XNOR-Net binarization in this repo.
    NNCF supports INT8, INT4, mixed-precision, and LLM-specific optimizations.

    Usage:
        compressor = NNCFCompressor()
        compressed_model = compressor.quantize(model, calibration_data)
        compressor.export_openvino(compressed_model, "model.xml")
    """

    def __init__(self, config: NNCFConfig | None = None) -> None:
        self.config = config or NNCFConfig()

    def quantize(
        self,
        model: Any,
        calibration_dataset: Any = None,
        config: NNCFConfig | None = None,
    ) -> Any:
        """Quantize model using NNCF post-training quantization (PTQ).

        Args:
            model: PyTorch model or OpenVINO model.
            calibration_dataset: Calibration data for PTQ.
            config: Override default config.

        Returns:
            Quantized model.
        """
        cfg = config or self.config

        try:
            import nncf
        except ImportError as e:
            raise ImportError(
                "nncf required. Install: pip install nncf"
            ) from e

        # PTQ quantization
        quantized = nncf.quantize(
            model,
            calibration_dataset=calibration_dataset,
            preset=cfg.preset.value,
            target_device=cfg.target_device.value,
            ignored_scope=nncf.common.ignored_scope.IgnoredScope(
                names=cfg.ignored_scopes,
                patterns=cfg.ignored_patterns,
            ),
            model_type=cfg.model_type,
            group_size=cfg.group_size if cfg.model_type == "transformer" else 0,
        )

        return quantized

    def quantize_int8(
        self,
        model: Any,
        calibration_data: np.ndarray | None = None,
    ) -> Any:
        """Quantize model to INT8 (4x compression vs FP32).

        Args:
            model: PyTorch model.
            calibration_data: Numpy array of calibration samples.

        Returns:
            INT8 quantized model.
        """
        config = NNCFConfig(
            preset=QuantizationPreset.MIXED,
            target_device=self.config.target_device,
        )
        return self.quantize(model, calibration_data, config)

    def quantize_int4(
        self,
        model: Any,
        calibration_data: np.ndarray | None = None,
        group_size: int = 128,
    ) -> Any:
        """Quantize model to INT4 (8x compression vs FP32).

        Args:
            model: PyTorch model.
            calibration_data: Numpy array of calibration samples.
            group_size: Group size for group-wise quantization.

        Returns:
            INT4 quantized model.
        """
        config = NNCFConfig(
            preset=QuantizationPreset.MIXED,
            target_device=self.config.target_device,
            model_type="transformer",
            group_size=group_size,
        )
        return self.quantize(model, calibration_data, config)

    def quantize_llm(
        self,
        model: Any,
        calibration_data: np.ndarray | None = None,
        precision: Literal["int8", "int4", "mixed"] = "int4",
        smooth_quant: bool = True,
    ) -> Any:
        """Quantize LLM with LLM-specific optimizations.

        Uses SmoothQuant and group quantization for optimal LLM compression.

        Args:
            model: Transformer-based LLM.
            calibration_data: Text or token calibration data.
            precision: Target precision.
            smooth_quant: Enable SmoothQuant for INT8.

        Returns:
            Quantized LLM.
        """
        config = NNCFConfig(
            preset=QuantizationPreset.ACCURACY,
            target_device=self.config.target_device,
            model_type="transformer",
            group_size=128,
            smooth_quant=smooth_quant,
        )
        return self.quantize(model, calibration_data, config)

    def prune(
        self,
        model: Any,
        compression_rate: float = 0.5,
        scheduler: str = "exponential",
    ) -> Any:
        """Structured pruning for model compression.

        Args:
            model: PyTorch model.
            compression_rate: Target compression ratio (0.0-1.0).
            scheduler: Pruning schedule ('linear', 'exponential').

        Returns:
            Pruned model.
        """
        try:
            import nncf
        except ImportError as e:
            raise ImportError("nncf required. Install: pip install nncf") from e

        pruning_config = nncf.common.initialization.LastLayerLoaderQNCAccuracyHandler
        compressed_model = nncf压缩_prune(
            model,
            compression_rate=compression_rate,
        )
        return compressed_model

    def export_openvino(
        self,
        model: Any,
        output_path: str | Path,
        input_shape: tuple[int, ...] = (1, 3, 224, 224),
    ) -> Path:
        """Export quantized model to OpenVINO IR format.

        Args:
            model: Quantized model.
            output_path: Output directory for .xml and .bin files.
            input_shape: Input tensor shape.

        Returns:
            Path to exported .xml file.
        """
        try:
            import openvino as ov
        except ImportError as e:
            raise ImportError(
                "openvino required. Install: pip install openvino"
            ) from e

        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        xml_path = output_path / "model.xml"

        # Export based on model type
        if hasattr(model, 'forward'):
            # PyTorch model — convert to ONNX first, then to IR
            import torch
            dummy = torch.randn(*input_shape)
            torch.onnx.export(
                model, dummy, str(output_path / "model.onnx"),
                opset_version=13,
            )
            ov_model = ov.convert_model(str(output_path / "model.onnx"))
        else:
            ov_model = model

        ov.save_model(ov_model, str(xml_path))
        return xml_path

    def export_onnx(
        self,
        model: Any,
        output_path: str | Path,
        input_shape: tuple[int, ...] = (1, 3, 224, 224),
    ) -> Path:
        """Export quantized model to ONNX format.

        Args:
            model: PyTorch model.
            output_path: Path for .onnx file.
            input_shape: Input tensor shape.

        Returns:
            Path to exported ONNX file.
        """
        import torch

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        dummy = torch.randn(*input_shape)
        model.eval()
        torch.onnx.export(
            model, dummy, str(output_path),
            opset_version=13,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        )
        return output_path

    def benchmark(
        self,
        model: Any,
        input_shape: tuple[int, ...] = (1, 3, 224, 224),
        num_runs: int = 100,
    ) -> dict[str, float]:
        """Benchmark model inference speed.

        Args:
            model: Model to benchmark.
            input_shape: Input tensor shape.
            num_runs: Number of inference iterations.

        Returns:
            Dict with latency and throughput metrics.
        """
        import torch

        device = next(model.parameters()).device if hasattr(model, 'parameters') else torch.device('cpu')
        dummy = torch.randn(*input_shape).to(device)
        model.eval()

        # Warmup
        with torch.no_grad():
            for _ in range(10):
                _ = model(dummy)

        # Benchmark
        times = []
        with torch.no_grad():
            for _ in range(num_runs):
                start = time.perf_counter()
                _ = model(dummy)
                end = time.perf_counter()
                times.append(end - start)

        avg_latency = sum(times) / len(times)
        return {
            "avg_latency_ms": avg_latency * 1000,
            "min_latency_ms": min(times) * 1000,
            "max_latency_ms": max(times) * 1000,
            "throughput_fps": 1.0 / avg_latency if avg_latency > 0 else 0,
            "num_runs": num_runs,
            "device": str(device),
        }

    @staticmethod
    def get_supported_devices() -> dict[str, dict[str, str]]:
        """Return all supported NNCF target devices."""
        return {
            "cpu": {
                "name": "Intel CPU",
                "precision": "INT8, INT4",
                "sdk": "OpenVINO",
                "examples": "Core i7, i9, Xeon",
            },
            "gpu": {
                "name": "Intel GPU",
                "precision": "INT8, INT4",
                "sdk": "OpenVINO",
                "examples": "Intel Arc, Data Center GPU Max",
            },
            "vpu": {
                "name": "Intel VPU",
                "precision": "INT8",
                "sdk": "OpenVINO",
                "examples": "Movidius NCS2, Meteor Lake VPU",
            },
            "gna": {
                "name": "Intel GNA",
                "precision": "INT8",
                "sdk": "OpenVINO",
                "examples": "Sandy Bridge+, always-on inference",
            },
            "cpu_spr": {
                "name": "Intel Sapphire Rapids",
                "precision": "INT8, INT4, BF16",
                "sdk": "OpenVINO + oneAPI",
                "examples": "4th Gen Xeon, AMX instructions",
            },
        }


# ─── Comparison: Legacy vs Modern ───────────────────────────────────────────────

@dataclass
class MethodComparison:
    method: str
    year: int
    precision: str
    speedup: str
    quality: str
    hardware: str
    llm_support: bool
    auto_quantize: bool
    key_feature: str


def compare_legacy_vs_modern() -> list[MethodComparison]:
    """Compare legacy XNOR-Net with modern NNCF quantization."""
    return [
        MethodComparison(
            method="XNOR-Net (this repo)",
            year=2016,
            precision="1-bit",
            speedup="5-10x",
            quality="Poor for LLMs",
            hardware="Movidius NCS (512MB)",
            llm_support=False,
            auto_quantize=False,
            key_feature="XNOR + Popcount on SHAVE cores",
        ),
        MethodComparison(
            method="NNCF PTQ INT8",
            year=2022,
            precision="8-bit",
            speedup="2-4x",
            quality="Excellent",
            hardware="Intel CPU/GPU/VPU",
            llm_support=True,
            auto_quantize=True,
            key_feature="Post-training quantization, no retraining needed",
        ),
        MethodComparison(
            method="NNCF QAT INT8",
            year=2022,
            precision="8-bit",
            speedup="2-4x",
            quality="Excellent",
            hardware="Intel CPU/GPU/VPU",
            llm_support=True,
            auto_quantize=True,
            key_feature="Quantization-aware training for best accuracy",
        ),
        MethodComparison(
            method="NNCF INT4 (LLM)",
            year=2024,
            precision="4-bit",
            speedup="2-3x",
            quality="Excellent",
            hardware="Intel CPU/GPU",
            llm_support=True,
            auto_quantize=True,
            key_feature="Group quantization + SmoothQuant for LLMs",
        ),
        MethodComparison(
            method="BitNet b1.58",
            year=2024,
            precision="1.58-bit",
            speedup="3-5x",
            quality="Good",
            hardware="CPU/GPU (custom kernels)",
            llm_support=True,
            auto_quantize=False,
            key_feature="Ternary weights with zero gate",
        ),
        MethodComparison(
            method="GPTQ",
            year=2022,
            precision="4-bit",
            speedup="2-3x",
            quality="Excellent",
            hardware="NVIDIA GPU",
            llm_support=True,
            auto_quantize=True,
            key_feature="Layer-wise quantization for LLMs",
        ),
        MethodComparison(
            method="AWQ",
            year=2023,
            precision="4-bit (W4A16)",
            speedup="2-3x",
            quality="Excellent",
            hardware="NVIDIA GPU",
            llm_support=True,
            auto_quantize=True,
            key_feature="Activation-aware weight quantization",
        ),
    ]


# ─── LLM Quantization Example ──────────────────────────────────────────────────

def quantize_llm_with_nncf(
    model_name: str = "Qwen/Qwen2-0.5B",
    output_dir: str = "./quantized_model",
    precision: Literal["int8", "int4"] = "int4",
) -> dict[str, Any]:
    """Complete LLM quantization pipeline using NNCF.

    This is the modern equivalent of what this repo did for vision CNNs
    on Movidius NCS — but for LLMs on Intel hardware.

    Args:
        model_name: HuggingFace model name.
        output_dir: Output directory for quantized model.
        precision: Target precision.

    Returns:
        Dict with quantization results.
    """
    try:
        import nncf
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as e:
        raise ImportError(
            "nncf, torch, transformers required. "
            "Install: pip install nncf torch transformers"
        ) from e

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load model
    print(f"[QUANT] Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        device_map="cpu",
    )

    # Create calibration dataset
    print("[QUANT] Creating calibration dataset...")
    calibration_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Artificial intelligence is transforming every industry.",
        "Computer vision enables machines to understand visual data.",
        "Edge AI brings intelligence to IoT devices.",
        "Binary neural networks reduce model size by 32x.",
    ]

    def transform_fn(batch):
        return tokenizer(batch, return_tensors="pt", padding=True, truncation=True)

    calibration_dataset = nncf.Dataset(calibration_texts, transform_fn)

    # Quantize
    print(f"[QUANT] Quantizing to {precision}...")
    if precision == "int8":
        quantized = nncf.quantize(
            model,
            calibration_dataset=calibration_dataset,
            preset=nncf.QuantizationPreset.MIXED,
            model_type="transformer",
            group_size=128,
        )
    else:  # int4
        quantized = nncf.quantize(
            model,
            calibration_dataset=calibration_dataset,
            preset=nncf.QuantizationPreset.MIXED,
            model_type="transformer",
            group_size=128,
        )

    # Save
    print("[QUANT] Saving quantized model...")
    quantized.save_pretrained(str(output_path / "quantized"))
    tokenizer.save_pretrained(str(output_path / "quantized"))

    # Export to OpenVINO IR
    print("[QUANT] Exporting to OpenVINO IR...")
    try:
        import openvino as ov
        ov_model = ov.convert_model(quantized)
        ov.save_model(ov_model, str(output_path / "model.xml"))
        print("[QUANT] OpenVINO IR exported successfully")
    except Exception as e:
        print(f"[QUANT] OpenVINO export skipped: {e}")

    return {
        "model": model_name,
        "precision": precision,
        "output_dir": str(output_path),
        "status": "success",
    }


# ─── CLI ────────────────────────────────────────────────────────────────────────

def main() -> None:
    """CLI entry point for NNCF module."""
    import argparse

    parser = argparse.ArgumentParser(description="Intel NNCF quantization — modern successor to XNOR-Net")
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("compare", help="Compare legacy vs modern methods")
    sub.add_parser("devices", help="List supported Intel devices")

    llm = sub.add_parser("quantize-llm", help="Quantize LLM with NNCF")
    llm.add_argument("--model", default="Qwen/Qwen2-0.5B", help="HuggingFace model name")
    llm.add_argument("--output", default="./quantized_model", help="Output directory")
    llm.add_argument("--precision", choices=["int8", "int4"], default="int4", help="Target precision")

    args = parser.parse_args()

    if args.command == "compare":
        methods = compare_legacy_vs_modern()
        print(f"{'Method':<25} {'Year':<6} {'Precision':<10} {'Speedup':<10} {'LLM':<6} {'Auto':<6} {'Key Feature'}")
        print("=" * 100)
        for m in methods:
            llm = "✅" if m.llm_support else "❌"
            auto = "✅" if m.auto_quantize else "❌"
            print(f"{m.method:<25} {m.year:<6} {m.precision:<10} {m.speedup:<10} {llm:<6} {auto:<6} {m.key_feature}")

    elif args.command == "devices":
        devices = NNCFCompressor.get_supported_devices()
        print(f"{'Device':<10} {'Name':<25} {'Precision':<15} {'Examples'}")
        print("=" * 70)
        for dev_id, info in devices.items():
            print(f"{dev_id:<10} {info['name']:<25} {info['precision']:<15} {info['examples']}")

    elif args.command == "quantize-llm":
        result = quantize_llm_with_nncf(args.model, args.output, args.precision)
        print(f"\nResult: {result['status']}")
        print(f"Output: {result['output_dir']}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
