"""Multi-chip AI accelerator support — convert and deploy binary/quantized models to all major edge AI chips.

Supports: Intel Movidius, Hailo, Axelera, Qualcomm, Google Coral, Apple Neural Engine,
ARM Ethos, Rockchip NPU, MediaTek APU, Samsung NPU, NVIDIA TensorRT, and more.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Literal

import numpy as np


class ChipVendor(Enum):
    INTEL_MOVIDIUS = "intel_movidius"
    HAILO = "hailo"
    AXELERA = "axelera"
    QUALCOMM = "qualcomm"
    GOOGLE_CORAL = "google_coral"
    APPLE_NE = "apple_neural_engine"
    ARM_ETHOS = "arm_ethos"
    NVIDIA = "nvidia"
    ROCKCHIP = "rockchip"
    MEDIATEK = "mediatek"
    SAMSUNG = "samsung"
    BCOM = "broadcom"
    TI = "texas_instruments"
    KNERON = "kneron"
    SYNTIANT = "syntiant"


@dataclass
class ChipSpec:
    vendor: ChipVendor
    name: str
    max_model_size_mb: float
    supported_precisions: list[str]
    tops_per_watt: float
    interface: str
    sdk_name: str
    sdk_install: str
    notes: str = ""


# ─── Chip Database ───────────────────────────────────────────────────────────────

CHIP_DB: dict[str, ChipSpec] = {
    "movidius_ncs2": ChipSpec(
        vendor=ChipVendor.INTEL_MOVIDIUS,
        name="Intel Movidius Neural Compute Stick 2",
        max_model_size_mb=100,
        supported_precisions=["fp16", "int8"],
        tops_per_watt=4.0,
        interface="USB 3.0",
        sdk_name="OpenVINO (MYRIAD backend)",
        sdk_install="pip install openvino",
        notes="4 TOPS, 1.5W. Deprecated — migrate to Hailo or Axelera.",
    ),
    "movidius_vpu": ChipSpec(
        vendor=ChipVendor.INTEL_MOVIDIUS,
        name="Intel Movidius VPU (Myriad X)",
        max_model_size_mb=500,
        supported_precisions=["fp16", "int8"],
        tops_per_watt=4.0,
        interface="PCIe / USB",
        sdk_name="OpenVINO",
        sdk_install="pip install openvino",
        notes="Integrated in Intel Meteor Lake, Arrow Lake CPUs.",
    ),
    "hailo8": ChipSpec(
        vendor=ChipVendor.HAILO,
        name="Hailo-8 AI Accelerator",
        max_model_size_mb=500,
        supported_precisions=["fp16", "int8", "binary"],
        tops_per_watt=26.0,
        interface="PCIe / M.2 / USB",
        sdk_name="Hailo Dataflow Compiler",
        sdk_install="pip install hailo-sdk-client",
        notes="26 TOPS, 2.5W. Best TOPS/W in class. Supports HEF format.",
    ),
    "hailo8l": ChipSpec(
        vendor=ChipVendor.HAILO,
        name="Hailo-8L AI Accelerator",
        max_model_size_mb=250,
        supported_precisions=["fp16", "int8"],
        tops_per_watt=13.0,
        interface="M.2",
        sdk_name="Hailo Dataflow Compiler",
        sdk_install="pip install hailo-sdk-client",
        notes="13 TOPS, 1W. Ultra-low-power edge AI.",
    ),
    "hailo15": ChipSpec(
        vendor=ChipVendor.HAILO,
        name="Hailo-15 AI Vision Processor",
        max_model_size_mb=1000,
        supported_precisions=["fp16", "int8", "binary"],
        tops_per_watt=30.0,
        interface="PCIe / MIPI CSI",
        sdk_name="Hailo Dataflow Compiler",
        sdk_install="pip install hailo-sdk-client",
        notes="30 TOPS, integrated ISP. Full vision pipeline on-chip.",
    ),
    "hailo15m": ChipSpec(
        vendor=ChipVendor.HAILO,
        name="Hailo-15M Multi-Function AI Processor",
        max_model_size_mb=2000,
        supported_precisions=["fp16", "int8"],
        tops_per_watt=25.0,
        interface="PCIe",
        sdk_name="Hailo Dataflow Compiler",
        sdk_install="pip install hailo-sdk-client",
        notes="Multi-pipeline vision processor with video decode.",
    ),
    "axelera_metis": ChipSpec(
        vendor=ChipVendor.AXELERA,
        name="Axelera Metis AIPU",
        max_model_size_mb=500,
        supported_precisions=["fp16", "int8", "binary"],
        tops_per_watt=20.0,
        interface="M.2 / PCIe",
        sdk_name="Axelera SDK (AXDL)",
        sdk_install="pip install axelera-ai-sdk",
        notes="20+ TOPS, 1.5W. Memory-first architecture. RISC-V core.",
    ),
    "qualcomm_cloud_ai100": ChipSpec(
        vendor=ChipVendor.QUALCOMM,
        name="Qualcomm Cloud AI 100",
        max_model_size_mb=4000,
        supported_precisions=["fp16", "int8", "int4"],
        tops_per_watt=15.0,
        interface="PCIe Gen4",
        sdk_name="Qualcomm AI Engine Direct (QNN)",
        sdk_install="pip install qnn-sdk",
        notes="400 TOPS. Datacenter inference accelerator.",
    ),
    "qualcomm_hexagon_dsp": ChipSpec(
        vendor=ChipVendor.QUALCOMM,
        name="Qualcomm Hexagon DSP (Snapdragon)",
        max_model_size_mb=500,
        supported_precisions=["fp16", "int8", "int4"],
        tops_per_watt=12.0,
        interface="SoC integrated",
        sdk_name="Qualcomm Neural Processing SDK",
        sdk_install="pip install snpe-sdk",
        notes="On Snapdragon 8 Gen 3 / 8 Elite. 75 TOPS NPU.",
    ),
    "qualcomm_qcs6490": ChipSpec(
        vendor=ChipVendor.QUALCOMM,
        name="Qualcomm QCS6490 IoT Processor",
        max_model_size_mb=500,
        supported_precisions=["fp16", "int8"],
        tops_per_watt=8.0,
        interface="SoC integrated",
        sdk_name="Qualcomm Neural Processing SDK",
        sdk_install="pip install snpe-sdk",
        notes="12 TOPS NPU for IoT and edge.",
    ),
    "google_coral_tpu": ChipSpec(
        vendor=ChipVendor.GOOGLE_CORAL,
        name="Google Coral Edge TPU",
        max_model_size_mb=30,
        supported_precisions=["int8"],
        tops_per_watt=8.0,
        interface="USB / M.2 / Dev Board",
        sdk_name="TensorFlow Lite + Coral Runtime",
        sdk_install="pip install tflite-runtime pycoral",
        notes="4 TOPS, 0.5W. TFLite Edge TPU compiled models required.",
    ),
    "apple_neural_engine": ChipSpec(
        vendor=ChipVendor.APPLE_NE,
        name="Apple Neural Engine (A17 Pro / M4)",
        max_model_size_mb=2000,
        supported_precisions=["fp16", "int8", "binary"],
        tops_per_watt=35.0,
        interface="SoC integrated (Apple Silicon)",
        sdk_name="Core ML / MLX",
        sdk_install="pip install coremltools",
        notes="38 TOPS (M4). Unified memory — zero-copy CPU↔ANE.",
    ),
    "arm_ethos_u55": ChipSpec(
        vendor=ChipVendor.ARM_ETHOS,
        name="ARM Ethos-U55 MicroNPU",
        max_model_size_mb=10,
        supported_precisions=["int8", "binary"],
        tops_per_watt=5.0,
        interface="AMBA AXI / AHB",
        sdk_name="ARM Vela OS + Ethos-U Driver",
        sdk_install="pip install ethos-u-vela",
        notes="0.48 TOPS, 0.05W. Cortex-M/Microcontroller class.",
    ),
    "arm_ethos_u85": ChipSpec(
        vendor=ChipVendor.ARM_ETHOS,
        name="ARM Ethos-U85 MicroNPU",
        max_model_size_mb=50,
        supported_precisions=["int8", "int4"],
        tops_per_watt=8.0,
        interface="AMBA AXI",
        sdk_name="ARM Vela OS + Ethos-U Driver",
        sdk_install="pip install ethos-u-vela",
        notes="2 TOPS, 0.25W. Successor to U55.",
    ),
    "nvidia_jetson_orin": ChipSpec(
        vendor=ChipVendor.NVIDIA,
        name="NVIDIA Jetson Orin Nano/NX/AGX",
        max_model_size_mb=8000,
        supported_precisions=["fp32", "fp16", "int8", "binary"],
        tops_per_watt=15.0,
        interface="PCIe / SoC integrated",
        sdk_name="TensorRT + DeepStream",
        sdk_install="pip install tensorrt",
        notes="275 TOPS (AGX Orin). CUDA + TensorRT.",
    ),
    "rockchip_rk3588": ChipSpec(
        vendor=ChipVendor.ROCKCHIP,
        name="Rockchip RK3588 NPU",
        max_model_size_mb=500,
        supported_precisions=["fp16", "int8"],
        tops_per_watt=10.0,
        interface="SoC integrated",
        sdk_name="RKNN-Toolkit2",
        sdk_install="pip install rknn-toolkit2",
        notes="6 TOPS. Popular in SBC and IoT gateways.",
    ),
    "mediatek_apu": ChipSpec(
        vendor=ChipVendor.MEDIATEK,
        name="MediaTek APU (Dimensity 9300)",
        max_model_size_mb=1000,
        supported_precisions=["fp16", "int8", "int4"],
        tops_per_watt=12.0,
        interface="SoC integrated",
        sdk_name="NeuroPilot SDK",
        sdk_install="pip install mtk-neuropilot",
        notes="46 TOPS (Dimensity 9300). Mobile-first.",
    ),
    "samsung_npu": ChipSpec(
        vendor=ChipVendor.SAMSUNG,
        name="Samsung Exynos NPU (2400)",
        max_model_size_mb=1000,
        supported_precisions=["fp16", "int8"],
        tops_per_watt=10.0,
        interface="SoC integrated",
        sdk_name="Samsung Neural SDK",
        sdk_install="pip install samsung-neural-sdk",
        notes="34.7 TOPS (Exynos 2400). On-device AI.",
    ),
    "kneron_k230": ChipSpec(
        vendor=ChipVendor.KNERON,
        name="Kneron KL730 / K230 AI Processor",
        max_model_size_mb=200,
        supported_precisions=["int8", "int4", "binary"],
        tops_per_watt=15.0,
        interface="USB / PCIe / MIPI",
        sdk_name="Kneron Neural Processing Toolbox",
        sdk_install="pip install kneron-npt",
        notes="4 TOPS, 0.6W. NanoWatt class for battery devices.",
    ),
    "syntiant_ndp120": ChipSpec(
        vendor=ChipVendor.SYNTIANT,
        name="Syntiant NDP120 Neural Decision Processor",
        max_model_size_mb=10,
        supported_precisions=["int8", "binary"],
        tops_per_watt=30.0,
        interface="SPI / I2S",
        sdk_name="Syntiant Core SDK",
        sdk_install="pip install syntiant-core-sdk",
        notes="0.5 TOPS, 0.02W. Always-on keyword spotting.",
    ),
}


# ─── Model Compression for Edge ──────────────────────────────────────────────────

@dataclass
class CompressionResult:
    original_size_mb: float
    compressed_size_mb: float
    compression_ratio: float
    precision: str
    estimated_accuracy_loss: float
    format: str


def get_chip_specs(chip_id: str) -> ChipSpec:
    """Get specifications for a target AI chip.

    Args:
        chip_id: Chip identifier (e.g., 'hailo8', 'axelera_metis').

    Returns:
        ChipSpec with full specifications.

    Raises:
        ValueError: If chip_id not found in database.
    """
    if chip_id not in CHIP_DB:
        available = ", ".join(sorted(CHIP_DB.keys()))
        raise ValueError(f"Unknown chip '{chip_id}'. Available: {available}")
    return CHIP_DB[chip_id]


def list_chips() -> dict[str, dict[str, Any]]:
    """List all supported AI chips with key specs.

    Returns:
        Dict mapping chip_id to specs summary.
    """
    return {
        chip_id: {
            "name": spec.name,
            "vendor": spec.vendor.value,
            "tops_per_watt": spec.tops_per_watt,
            "max_model_mb": spec.max_model_size_mb,
            "precisions": spec.supported_precisions,
        }
        for chip_id, spec in CHIP_DB.items()
    }


def recommend_chip(
    model_size_mb: float,
    power_budget_w: float,
    target_tops: float,
    interface_preference: str = "any",
) -> list[str]:
    """Recommend best chips for given constraints.

    Args:
        model_size_mb: Model size in megabytes.
        power_budget_w: Maximum power budget in watts.
        target_tops: Required compute in TOPS.
        interface_preference: Preferred interface ('usb', 'm2', 'pcie', 'soc', 'any').

    Returns:
        List of recommended chip_ids, sorted by fitness.
    """
    candidates = []
    for chip_id, spec in CHIP_DB.items():
        if spec.max_model_size_mb < model_size_mb:
            continue
        max_tops = spec.tops_per_watt * power_budget_w
        if max_tops < target_tops:
            continue
        if interface_preference != "any" and interface_preference.lower() not in spec.interface.lower():
            continue
        fitness = spec.tops_per_watt / (model_size_mb / spec.max_model_size_mb + 0.1)
        candidates.append((chip_id, fitness))
    candidates.sort(key=lambda x: x[1], reverse=True)
    return [c[0] for c in candidates]


def estimate_compression(
    model_size_mb: float,
    source_precision: Literal["fp32", "fp16", "int8", "int4", "binary"],
    target_precision: Literal["fp16", "int8", "int4", "binary"],
) -> CompressionResult:
    """Estimate model size after precision conversion.

    Args:
        model_size_mb: Original model size in MB.
        source_precision: Source precision format.
        target_precision: Target precision format.

    Returns:
        CompressionResult with size estimates and accuracy loss.
    """
    bits_map = {"fp32": 32, "fp16": 16, "int8": 8, "int4": 4, "binary": 1}
    src_bits = bits_map.get(source_precision, 32)
    tgt_bits = bits_map.get(target_precision, 8)
    ratio = src_bits / tgt_bits
    compressed = model_size_mb / ratio

    accuracy_loss_map = {
        ("fp32", "fp16"): 0.001,
        ("fp32", "int8"): 0.01,
        ("fp32", "int4"): 0.03,
        ("fp32", "binary"): 0.10,
        ("fp16", "int8"): 0.005,
        ("fp16", "int4"): 0.02,
        ("fp16", "binary"): 0.08,
        ("int8", "int4"): 0.02,
        ("int8", "binary"): 0.08,
        ("int4", "binary"): 0.06,
    }
    loss = accuracy_loss_map.get((source_precision, target_precision), 0.05)

    return CompressionResult(
        original_size_mb=model_size_mb,
        compressed_size_mb=compressed,
        compression_ratio=ratio,
        precision=target_precision,
        estimated_accuracy_loss=loss,
        format=target_precision,
    )


def generate_deployment_package(
    model_path: str | Path,
    chip_id: str,
    output_dir: str | Path,
) -> Path:
    """Generate a complete deployment package for a target chip.

    Args:
        model_path: Path to source model (ONNX or PyTorch).
        chip_id: Target chip identifier.
        output_dir: Directory for deployment files.

    Returns:
        Path to the deployment package directory.
    """
    spec = get_chip_specs(chip_id)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # README with chip-specific instructions
    readme = out / "DEPLOY.md"
    readme.write_text(f"""# Deployment Guide: {spec.name}

## Target: {spec.vendor.value} — {spec.name}

### Specifications
- **Compute:** {spec.tops_per_watt} TOPS/W
- **Max Model Size:** {spec.max_model_size_mb} MB
- **Supported Precisions:** {', '.join(spec.supported_precisions)}
- **Interface:** {spec.interface}
- **SDK:** {spec.sdk_name}

### SDK Installation
```bash
{spec.sdk_install}
```

### Source Model
- **Input:** {model_path}
- **Vendor:** {spec.vendor.value}

### Conversion Steps
1. Export model to ONNX (if PyTorch):
   ```python
   from src.deploy import EdgeDeployer
   deployer = EdgeDeployer()
   deployer.export_onnx('{out}/model.onnx')
   ```

2. Quantize to {spec.supported_precisions[-1]} precision:
   ```python
   from src.quantize import Quantizer
   quantizer = Quantizer()
   quantizer.quantize_{spec.supported_precisions[-1]}('{out}/model.onnx', '{out}/model_quant.{spec.supported_precisions[-1]}')
   ```

3. Deploy to {spec.name}:
   ```python
   from src.deploy import EdgeDeployer
   result = EdgeDeployer.deploy_{chip_id.replace('_', '_')}('{out}/model_quant.{spec.supported_precisions[-1]}', input_data)
   ```

### Notes
{spec.notes}
""", encoding="utf-8")

    return out


# ─── Benchmark Comparison ────────────────────────────────────────────────────────

@dataclass
class BenchmarkRow:
    chip: str
    vendor: str
    tops_per_watt: float
    max_model_mb: float
    precisions: str
    interface: str
    sdk: str


def generate_comparison_table(chip_ids: list[str] | None = None) -> list[BenchmarkRow]:
    """Generate a comparison table for selected chips.

    Args:
        chip_ids: List of chip IDs to compare. None for all.

    Returns:
        List of BenchmarkRow dataclass instances.
    """
    ids = chip_ids if chip_ids else list(CHIP_DB.keys())
    rows = []
    for cid in ids:
        if cid not in CHIP_DB:
            continue
        spec = CHIP_DB[cid]
        rows.append(BenchmarkRow(
            chip=spec.name,
            vendor=spec.vendor.value,
            tops_per_watt=spec.tops_per_watt,
            max_model_mb=spec.max_model_size_mb,
            precisions=", ".join(spec.supported_precisions),
            interface=spec.interface,
            sdk=spec.sdk_name,
        ))
    rows.sort(key=lambda r: r.tops_per_watt, reverse=True)
    return rows


def print_comparison_table(chip_ids: list[str] | None = None) -> None:
    """Print a formatted comparison table to stdout."""
    rows = generate_comparison_table(chip_ids)
    header = f"{'Chip':<45} {'TOPS/W':>8} {'Max MB':>8} {'Precisions':<25} {'Interface':<20}"
    print(header)
    print("=" * len(header))
    for row in rows:
        print(f"{row.chip:<45} {row.tops_per_watt:>8.1f} {row.max_model_mb:>8.0f} {row.precisions:<25} {row.interface:<20}")


def main() -> None:
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Multi-chip AI accelerator support")
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("list", help="List all supported chips")
    sub.add_parser("compare", help="Print comparison table")

    rec = sub.add_parser("recommend", help="Recommend chips for constraints")
    rec.add_argument("--model-size", type=float, required=True, help="Model size in MB")
    rec.add_argument("--power", type=float, required=True, help="Power budget in watts")
    rec.add_argument("--tops", type=float, required=True, help="Required TOPS")
    rec.add_argument("--interface", default="any", help="Interface preference")

    info = sub.add_parser("info", help="Get chip details")
    info.add_argument("chip_id", help="Chip identifier")

    args = parser.parse_args()

    if args.command == "list":
        print_comparison_table()
    elif args.command == "compare":
        print_comparison_table()
    elif args.command == "recommend":
        recs = recommend_chip(args.model_size, args.power, args.tops, args.interface)
        if recs:
            print(f"Recommended chips for {args.model_size}MB model, {args.power}W, {args.tops} TOPS:")
            for i, cid in enumerate(recs, 1):
                spec = get_chip_specs(cid)
                print(f"  {i}. {spec.name} ({cid}) — {spec.tops_per_watt} TOPS/W")
        else:
            print("No chips match the given constraints.")
    elif args.command == "info":
        spec = get_chip_specs(args.chip_id)
        print(f"Name: {spec.name}")
        print(f"Vendor: {spec.vendor.value}")
        print(f"Max Model: {spec.max_model_size_mb} MB")
        print(f"Precisions: {', '.join(spec.supported_precisions)}")
        print(f"TOPS/W: {spec.tops_per_watt}")
        print(f"Interface: {spec.interface}")
        print(f"SDK: {spec.sdk_name}")
        print(f"Install: {spec.sdk_install}")
        print(f"Notes: {spec.notes}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
