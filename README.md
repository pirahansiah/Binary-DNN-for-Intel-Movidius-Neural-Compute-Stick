# Binary DNN — Multi-Chip Edge AI Deployment

[![Python](https://img.shields.io/badge/Python-3.10+-yellow.svg)](https://www.python.org)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

Binary and quantized neural network deployment across **15+ AI accelerators** — from microcontrollers to datacenter.

> **By Dr. Farshid Pirahansiah** — [www.tiziran.com](https://www.tiziran.com) | [YouTube](https://www.youtube.com/tiziran)

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
| `src/benchmark.py` | Latency, throughput, memory, power benchmarking |
| `src/models.py` | BinaryNet, BinaryCNN, BinaryResNet architectures |
| `src/visualize.py` | Weight visualization, benchmark plots |

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

## Requirements

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
