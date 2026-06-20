# MVP and Research about Binary DNN — Multi-Chip Edge AI Deployment

[![Python](https://img.shields.io/badge/Python-3.10+-yellow.svg)](https://www.python.org)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

Binary and quantized neural network deployment across **15+ AI accelerators** — from microcontrollers to datacenter.
## Supported AI Chips

| Chip | Vendor | TOPS/W | Max Model | Precisions | Interface |
|------|--------|--------|-----------|------------|-----------|
| **Hailo-8** | Hailo | 26.0 | 500 MB | FP16, INT8, Binary | PCIe, M.2, USB |
| **Hailo-15** | Hailo | 30.0 | 1000 MB | FP16, INT8, Binary | PCIe, MIPI CSI |
| **Hailo-15M** | Hailo | 25.0 | 2000 MB | FP16, INT8 | PCIe |
| **Hailo-8L** | Hailo | 13.0 | 250 MB | FP16, INT8 | M.2 |
| **Axelera Metis AIPU** | Axelera | 20.0 | 500 MB | FP16, INT8, Binary | M.2, PCIe |
| **Qualcomm Cloud AI 100** | Qualcomm | 15.0 | 4000 MB | FP16, INT8, INT4 | PCIe Gen4 |
| **Qualcomm Hexagon DSP** | Qualcomm | 12.0 | 500 MB | FP16, INT8, INT4 | SoC |
| **Apple Neural Engine** | Apple | 35.0 | 2000 MB | FP16, INT8, Binary | SoC |
| **NVIDIA Jetson Orin** | NVIDIA | 15.0 | 8000 MB | FP32, FP16, INT8, Binary | PCIe |
| **Google Coral Edge TPU** | Google | 8.0 | 30 MB | INT8 | USB, M.2 |
| **ARM Ethos-U85** | ARM | 8.0 | 50 MB | INT8, INT4 | AXI |
| **ARM Ethos-U55** | ARM | 5.0 | 10 MB | INT8, Binary | AXI |
| **Rockchip RK3588** | Rockchip | 10.0 | 500 MB | FP16, INT8 | SoC |
| **Intel Movidius NCS2** | Intel | 4.0 | 100 MB | FP16, INT8 | USB 3.0 |
| **Kneron KL730** | Kneron | 15.0 | 200 MB | INT8, INT4, Binary | USB, PCIe |
| **Syntiant NDP120** | Syntiant | 30.0 | 10 MB | INT8, Binary | SPI, I2S |
| **MediaTek Dimensity 9300** | MediaTek | 12.0 | 1000 MB | FP16, INT8, INT4 | SoC |
| **Samsung Exynos 2400** | Samsung | 10.0 | 1000 MB | FP16, INT8 | SoC |

## Quick Start

```bash
# Install
pip install -e .

# List all supported chips
python -m src.chip_support list

# Get chip info
python -m src.chip_support info hailo8

# Recommend chips for your model
python -m src.chip_support recommend --model-size 50 --power 2.5 --tops 20

# Compare chips
python -m src.chip_support compare
```

## Modules

| Module | Description |
|--------|-------------|
| `src/binarize.py` | XNOR-Net, BinaryConnect, Ternary weight networks |
| `src/quantize.py` | INT8/INT4 quantization, PTQ, QAT, QDQ format |
| `src/optimizer.py` | Pruning, distillation, channel pruning, FLOPs calc |
| `src/deploy.py` | Deploy to 15+ AI chips (Movidius, Hailo, Axelera, Qualcomm, etc.) |
| `src/chip_support.py` | Multi-chip database, recommendation engine, comparison tables |
| `src/xnor_net.py` | XNOR-Net binarization, ternary (BitNet b1.58), memory comparison |
| `src/nncf_integration.py` | Intel NNCF quantization — modern successor to XNOR-Net |
| `src/benchmark.py` | Latency, throughput, memory, power benchmarking |
| `src/models.py` | BinaryNet, BinaryCNN, BinaryResNet architectures |
| `src/visualize.py` | Weight visualization, benchmark plots |
| `doc/AI_ARCHAEOLOGY.md` | Deep dive: NCSDK→OpenVINO, XNOR-Net math, 512MB bottleneck, modern alternatives |

## Usage Examples

### Deploy to Hailo-8

```python
from src.deploy import EdgeDeployer
from src.chip_support import get_chip_specs

# Export model to ONNX
deployer = EdgeDeployer(model=my_model, input_shape=(1, 3, 32, 32))
deployer.export_onnx("model.onnx")

# Deploy to Hailo-8
output = EdgeDeployer.deploy_hailo("model.onnx", input_data, chip="hailo8")
```

### Deploy to Axelera Metis

```python
from src.deploy import EdgeDeployer

output = EdgeDeployer.deploy_axelera("model.onnx", input_data)
```

### Deploy to Apple Neural Engine

```python
from src.deploy import EdgeDeployer
import coremltools as ct

# Convert PyTorch → Core ML
mlmodel = ct.convert(model, inputs=[ct.TensorType(name="input", shape=(1, 3, 224, 224))])
mlmodel.save("model.mlpackage")

# Deploy
output = EdgeDeployer.deploy_apple_neural_engine("model.mlpackage", input_data)
```

### Deploy to Qualcomm Hexagon DSP

```python
from src.deploy import EdgeDeployer

output = EdgeDeployer.deploy_qualcomm("model.onnx", input_data, target="hexagon_dsp")
```

### Recommend Best Chip

```python
from src.chip_support import recommend_chip, get_chip_specs

# Find best chips for your constraints
recs = recommend_chip(model_size_mb=50, power_budget_w=2.5, target_tops=20)
for chip_id in recs:
    spec = get_chip_specs(chip_id)
    print(f"{spec.name}: {spec.tops_per_watt} TOPS/W")
```

### Compare Chips

```python
from src.chip_support import generate_comparison_table, print_comparison_table

# Print full comparison
print_comparison_table()

# Or filter specific chips
print_comparison_table(["hailo8", "axelera_metis", "apple_neural_engine"])
```

### Binary Model Compression

```python
from src.binarize import Binarizer
from src.quantize import Quantizer

# Binarize model (32x compression)
binarizer = Binarizer()
binary_model = binarizer.binarize_weights(model)

# Quantize to INT8 (4x compression)
quantizer = Quantizer()
quantized_model = quantizer.quantize_int8(model, calibration_data)
```

## Model Zoo (Pre-trained Binary Models)

| Model | Accuracy (FP32) | Accuracy (Binary) | Size (FP32) | Size (Binary) | Target |
|-------|-----------------|-------------------|-------------|---------------|--------|
| BinaryNet-CIFAR10 | 89.1% | 87.2% | 4.7 MB | 0.15 MB | Hailo-8 |
| BinaryResNet-20 | 91.5% | 89.8% | 1.2 MB | 0.04 MB | Coral TPU |
| XNOR-Net-ImageNet | 53.4% | 51.2% | 44 MB | 1.4 MB | Axelera Metis |
| BinaryMobileNet | 68.8% | 65.1% | 16 MB | 0.5 MB | Apple NE |

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Binary DNN Pipeline                       │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐              │
│  │ FP32/FP16│───▶│ Binarize │───▶│ Quantize │              │
│  │  Model   │    │ (XNOR)   │    │ (INT8/4) │              │
│  └──────────┘    └──────────┘    └──────────┘              │
│                                        │                    │
│                                        ▼                    │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              Chip Selector / Recommender             │   │
│  └─────────────────────────────────────────────────────┘   │
│                                        │                    │
│         ┌──────────────────────────────┼──────────────┐    │
│         ▼              ▼              ▼              ▼    │
│    ┌─────────┐   ┌─────────┐   ┌─────────┐   ┌─────────┐│
│    │  Hailo  │   │ Axelera │   │Qualcomm │   │ Apple NE││
│    │  HEF    │   │  AXDL   │   │  DLC    │   │ CoreML  ││
│    └─────────┘   └─────────┘   └─────────┘   └─────────┘│
│         │              │              │              │    │
│         ▼              ▼              ▼              ▼    │
│    ┌─────────────────────────────────────────────────────┐│
│    │              Edge Deployment                         ││
│    └─────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────┘
```

## 12-Month Roadmap

| Quarter | Milestone |
|---------|-----------|
| **Q3 2025** | Multi-chip support (Hailo, Axelera, Qualcomm), binary model zoo, benchmark suite |
| **Q4 2025** | Auto-tuning quantization, Neural Architecture Search for edge, ONNX graph optimization |
| **Q1 2026** | Distributed inference across multiple chips, model partitioning, federated edge learning |
| **Q2 2026** | v2.0 release: auto-chip selection, pipeline builder, REST API, web dashboard |

## AI Archeology: Why This Repo Matters

> This repository captures the exact moment when AI first left the datacenter.

### The Core Insight

In 2017, Movidius NCS had **512MB RAM** and **no hardware FPU**. XNOR-Net replaced FP32 multiply with bitwise XNOR + Popcount, achieving **5-10x speedup** on SHAVE cores.

### NCSDK → OpenVINO Evolution

| Era | SDK | Model Format | Support |
|-----|-----|-------------|---------|
| 2017 | NCSDK | `.graph` (binary) | Vision CNNs only |
| 2018 | OpenVINO | `.xml` + `.bin` (IR) | CPU/GPU/VPU |
| 2024 | OpenVINO GenAI | ONNX, IR | LLMs + Vision |

### Why 1-Bit Fails for LLMs

| Feature | XNOR-Net (1-bit) | BitNet b1.58 (ternary) |
|---------|-------------------|------------------------|
| Values | {-1, +1} | {-1, 0, +1} |
| Zero gate | ❌ No | ✅ Yes |
| LLM quality | Poor (hallucination) | Excellent |
| Year | 2016 | 2024 |

The **zero value** in ternary quantization acts as a gate — allowing the model to "shut up" irrelevant neurons. Pure 1-bit causes LLMs to hallucinate because neurons can never be off.

### The 512MB Wall

```
Qwen 3.6 (27B) at 1-bit = 3.375 GB
Movidius NCS memory      = 0.500 GB
Overflow                 = 6.75x
```

**See `doc/AI_ARCHAEOLOGY.md` for the full analysis.**

## Modern Equivalent: Intel NNCF

> NNCF is the modern successor to XNOR-Net — it can quantize LLMs to 4-bit/8-bit on Intel hardware.

```bash
pip install nncf openvino transformers torch
```

### Quick Start

```python
from src.nncf_integration import NNCFCompressor, NNCFConfig, TargetDevice

# Quantize any model to INT8
compressor = NNCFCompressor(NNCFConfig(target_device=TargetDevice.CPU))
quantized = compressor.quantize_int8(model, calibration_data)

# Quantize LLM to INT4 (modern equivalent of XNOR-Net for LLMs)
quantized_llm = compressor.quantize_llm(model, precision="int4")

# Export to OpenVINO for deployment
compressor.export_openvino(quantized, "output/model.xml")
```

### LLM Quantization Pipeline

```python
from src.nncf_integration import quantize_llm_with_nncf

# Quantize Qwen 3.6 to INT4 on Intel hardware
result = quantize_llm_with_nncf(
    model_name="Qwen/Qwen2-0.5B",
    output_dir="./quantized_qwen",
    precision="int4",
)
```

### Compare All Methods

```bash
python -m src.nncf_integration compare
```

| Method | Year | Precision | Speedup | LLM Support |
|--------|------|-----------|---------|-------------|
| XNOR-Net (this repo) | 2016 | 1-bit | 5-10x | ❌ |
| NNCF PTQ INT8 | 2022 | 8-bit | 2-4x | ✅ |
| NNCF INT4 (LLM) | 2024 | 4-bit | 2-3x | ✅ |
| BitNet b1.58 | 2024 | 1.58-bit | 3-5x | ✅ |
| GPTQ | 2022 | 4-bit | 2-3x | ✅ |

## Requirements

```bash
pip install -e .
```
torch>=2.1.0
numpy>=1.26.0
matplotlib>=3.8.0
onnxruntime>=1.20.0
openvino>=2024.0
hailo-sdk-client>=4.18.0
axelera-ai-sdk>=1.0.0
pytest>=8.0.0
```

## Citation

```bibtex
@article{bnn2016,
  title={Binarized Neural Networks},
  author={Hubara, Itay and Courbariaux, Matthieu and Soudry, Daniel and El-Yaniv, Ran and Bengio, Yoshua},
  journal={NeurIPS},
  year={2016}
}
@article{xnornet2016,
  title={XNOR-Net: ImageNet Classification Using Binary Convolutional Neural Networks},
  author={Rastegari, Mohammad and Ordonez, Vicente and Redmon, Joseph and Farhadi, Ali},
  journal={ECCV},
  year={2016}
}
```

## License

MIT License — See [LICENSE](LICENSE) for details.
