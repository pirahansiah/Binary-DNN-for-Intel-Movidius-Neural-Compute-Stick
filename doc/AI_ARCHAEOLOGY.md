# AI Archeology: Why This Repository Matters

> A deep dive into the transition from NCSDK to OpenVINO, the math of XNOR-Net,
> and the hardware limits of the Myriad VPU — and how it connects to modern LLM quantization.

---

## Table of Contents

- [1. NCSDK vs. OpenVINO: The Architectural Pivot](#1-ncsdk-vs-openvino-the-architectural-pivot)
- [2. XNOR-Net Math: How the Repo Achieves 1-Bit](#2-xnor-net-math-how-the-repo-achieves-1-bit)
- [3. The 512MB RAM Bottleneck vs. Modern LLMs](#3-the-512mb-ram-bottleneck-vs-modern-llms)
- [4. Analysis of Related Source Code](#4-analysis-of-related-source-code)
- [5. What a Modern Version Looks Like](#5-what-a-modern-version-looks-like)
- [6. Bridge: From Movidius to vLLM](#6-bridge-from-movidius-to-vllm)
- [7. Timeline: Edge AI Evolution](#7-timeline-edge-ai-evolution)

---

## 1. NCSDK vs. OpenVINO: The Architectural Pivot

When this repository was created, Intel's **NCSDK** (Neural Compute SDK) was the only way to talk to the Movidius Myriad 2 chip.

### NCSDK (The Legacy)

- Closed ecosystem. You used `mvNCCompile` to turn a Caffe or TensorFlow graph into a `.graph` file.
- Designed specifically for **Edge Vision** — assumed input was an image, output was a classification or bounding box.
- Limited to feed-forward networks only.
- No support for recurrent architectures (LSTM, GRU) or attention mechanisms.

```bash
# The old way (2017-2018)
mvNCCompile network.prototxt -o output.graph
# The graph file was a binary blob — no inspecting, no debugging, no modern tooling
```

### OpenVINO (The Successor)

Intel realized they couldn't have a different SDK for every chip. OpenVINO introduced the **Intermediate Representation (IR)** format:

- `.xml` — network topology (human-readable)
- `.bin` — weights and biases (binary blob)

```bash
# The modern way
python mo.py --input_model model.onnx --output_dir ir/
# Produces model.xml + model.bin — portable across all Intel hardware
```

OpenVINO unified:
- CPU inference (AVX-512, VNNI)
- GPU inference (Intel HD/UHD/Iris)
- VPU inference (Movidius NCS2, Movidius VPU in Meteor Lake)
- GNA (Gaussian Neural Accelerator for always-on)

### The Conflict for LLMs

**Neither NCSDK nor early OpenVINO supported "Auto-regressive" loops** — the "repeat until end-of-sentence" logic required by LLMs. They were **Feed-Forward only**.

| Capability | NCSDK | OpenVINO (Early) | OpenVINO (2024+) | vLLM |
|------------|-------|-------------------|-------------------|------|
| Feed-Forward | ✅ | ✅ | ✅ | ✅ |
| Recurrent (LSTM) | ❌ | ✅ | ✅ | ✅ |
| Attention | ❌ | Limited | ✅ | ✅ |
| KV Cache | ❌ | ❌ | ✅ | ✅ |
| Auto-regressive | ❌ | ❌ | ✅ (via extensions) | ✅ |
| 27B+ Params | ❌ | ❌ | ✅ (with GPU) | ✅ |

### Why This Matters

This repository captures the exact moment in AI history when edge inference was **vision-only**. The Myriad VPU was designed for a world where neural networks had:
- Fixed input shapes (224×224, 300×300)
- No recurrence
- No attention
- No memory management beyond the graph

The jump to LLMs required **fundamentally new hardware** (NVIDIA A100/H100, Intel Gaudi) and **fundamentally new software** (vLLM, PagedAttention, speculative decoding).

---

## 2. XNOR-Net Math: How the Repo Achieves 1-Bit

The repository implements the logic from the seminal paper **"XNOR-Net: ImageNet Classification Using Binary Convolutional Neural Networks"** (Rastegari et al., ECCV 2016).

### Standard DNN Computation

In a standard DNN, a neuron calculates:

$$Y = \sum_{i=1}^{n} (W_i \times A_i) + B$$

Where:
- $W_i$ = weight (32-bit float)
- $A_i$ = activation (32-bit float)
- $B$ = bias (32-bit float)

This requires a **Floating Point Unit (FPU)**. On the Movidius NCS, floating-point math is slow.

### The XNOR-Net Trick

**Step 1: Binarize Weights**
$$W_{binary} = \text{sign}(W) = \begin{cases} +1 & \text{if } W \geq 0 \\ -1 & \text{if } W < 0 \end{cases}$$

Example: $+0.78 \rightarrow +1$, $-0.12 \rightarrow -1$

**Step 2: Binarize Activations**
$$A_{binary} = \text{sign}(A) = \begin{cases} +1 & \text{if } A > 0 \\ -1 & \text{if } A \leq 0 \end{cases}$$

**Step 3: Replace Multiplication with XNOR + Popcount**

Instead of:
```
result = 0.78 × 0.92 + (-0.12) × 0.34 + ...
```

We do:
```
result = XNOR(1, 1) + XNOR(-1, 1) + ...
       = Popcount(matching_bits) - Popcount(mismatching_bits)
```

**Why this is faster on Myriad:**

The Movidius Myriad 2 has **12 SHAVE cores** (Streaming Hybrid Architecture Video Engine). Each SHAVE core is a VLIW processor with:
- 128-bit SIMD registers
- Hardware bit-shift units
- No hardware FPU (floating-point is emulated in software)

So:
- **32-bit float multiply** = ~10 cycles (software emulation)
- **1-bit XNOR + Popcount** = 1 cycle (single instruction)

This gives a **5-10x speedup** for the convolution operations.

### Code: The Binarizer Wrapper

```python
class XNORBinarize(torch.autograd.Function):
    """Binarize weights using sign function with STE gradient."""

    @staticmethod
    def forward(ctx, tensor):
        ctx.save_for_backward(tensor)
        return tensor.sign()

    @staticmethod
    def backward(ctx, grad_output):
        tensor, = ctx.saved_tensors
        # Straight-Through Estimator (STE)
        grad_input = grad_output.clone()
        grad_input[tensor.abs() > 1] = 0
        return grad_input

# Usage in model
class BinaryConv2d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_ch, in_ch, kernel_size, kernel_size))
        self.scale = nn.Parameter(torch.ones(1))  # Per-channel scaling factor

    def forward(self, x):
        w_binary = XNORBinarize.apply(self.weight)
        # Scale factor to compensate for binarization loss
        w_scaled = self.scale * w_binary
        return F.conv2d(x, w_scaled)
```

### Why Pure 1-Bit Fails for LLMs

**The critical difference:** In CNNs, a neuron that fires "too much" can be compensated by batch normalization. In LLMs, a neuron that can never be "off" (because there's no zero value in pure 1-bit) causes:

1. **Noise accumulation** — Every neuron contributes to every token, even irrelevant ones
2. **No gating** — Transformers use attention to gate information flow; 1-bit removes this
3. **Hallucination** — The model can never "decide" to not generate a token

**Modern solution: Ternary (BitNet b1.58)**

$$W_{ternary} = \begin{cases} +1 & \text{if } W > \alpha \\ 0 & \text{if } |W| \leq \alpha \\ -1 & \text{if } W < -\alpha \end{cases}$$

The **zero value** acts as a gate — it can "shut up" a neuron. This is why BitNet b1.58 works for LLMs but pure XNOR does not.

---

## 3. The 512MB RAM Bottleneck vs. Modern LLMs

### Hardware Reality: Movidius NCS

| Specification | Value |
|---------------|-------|
| Total RAM | 512 MB (LPDDR3) |
| SHAVE Cores | 12 |
| Clock Speed | 600 MHz |
| Max Model Size | ~400M parameters (1-bit) |
| Input | Static images only |
| Context Window | None |
| Power | 1.5W (USB powered) |

### Modern LLM Requirements

| Model | Parameters | FP16 Size | INT4 Size | 1-Bit Size | RAM Needed |
|-------|------------|-----------|-----------|------------|------------|
| Qwen 3.6 | 27B | 54 GB | 14 GB | 3.375 GB | 16+ GB |
| Llama 3 70B | 70B | 140 GB | 35 GB | 8.75 GB | 40+ GB |
| GPT-4 (estimated) | 1.8T | 3.6 TB | 900 GB | 225 GB | 1000+ GB |

### The Math That Kills It

Even if you quantize Qwen 3.6 down to **exactly 1 bit per parameter**:

```
27,000,000,000 bits ÷ 8 = 3,375,000,000 bytes = 3.375 GB
```

**The Movidius Stick has 0.5 GB.**

```
Model Size (1-bit):    3.375 GB
Hardware Memory:       0.500 GB
Ratio:                 6.75x OVERFLOW
```

**Result:** The model is nearly 7 times larger than the entire memory of the hardware device.

### Why KV Cache Makes It Worse

LLMs don't just load the model — they also need **KV Cache** for attention:

```
KV Cache = 2 × num_layers × num_heads × context_length × head_dim × dtype_size
```

For Qwen 3.6 (27B, 128k context):
```
KV Cache ≈ 2 × 80 × 32 × 131,072 × 128 × 2 bytes
         ≈ 21.5 GB (FP16)
```

Even with 1-bit KV cache:
```
KV Cache (1-bit) ≈ 21.5 GB ÷ 16 ≈ 1.34 GB
```

**Still larger than the entire Movidius NCS memory.**

### What CAN Run on Movidius NCS

| Model | Parameters | 1-Bit Size | Fits in 512MB? |
|-------|------------|------------|----------------|
| MobileNet v1 | 4.2M | 0.5 MB | ✅ |
| SqueezeNet 1.1 | 1.2M | 0.15 MB | ✅ |
| YOLOv3-tiny | 8.7M | 1.1 MB | ✅ |
| ResNet-18 | 11.7M | 1.5 MB | ✅ |
| BERT-tiny | 4.4M | 0.55 MB | ✅ |
| BERT-base | 110M | 13.75 MB | ✅ |
| GPT-2 | 124M | 15.5 MB | ✅ |
| Qwen 3.6 | 27B | 3375 MB | ❌ |

---

## 4. Analysis of Related Source Code

### A. The "Binarizer" Wrapper

The code typically includes a custom layer:

```python
def binarize(tensor):
    return tensor.sign()  # Converts to -1 or +1
```

**In modern LLM quantization (BitNet b1.58):**

```python
def ternary_quantize(tensor, alpha=0.5):
    """Ternary quantization: -1, 0, +1 with learnable threshold."""
    return torch.where(tensor > alpha, 1.0,
           torch.where(tensor < -alpha, -1.0, 0.0))
```

**Why the zero matters for LLMs:**

A transformer attention head computes:
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$

If weights can only be -1 or +1 (no zero), the attention scores can never be "neutral". Every token always contributes something, even if it's irrelevant. This leads to:
- Noise accumulation in deep layers
- Inability to perform feature selection
- Hallucination in generation

The zero value in ternary quantization allows the model to learn **which neurons to suppress** — equivalent to a hardware-level attention gate.

### B. Quantization-Aware Training (QAT)

The Pirahansiah repo expects you to **train from scratch** as a binary model:

```python
# Old approach (this repo)
model = BinaryNet()
for epoch in range(epochs):
    for batch in dataloader:
        output = model(batch)          # Forward with binarized weights
        loss = criterion(output, target)
        loss.backward()
        # STE gradient for sign function
        optimizer.step()               # Update latent weights
```

**You cannot simply take a pre-trained Qwen 3.6 and turn it into 1-bit.**

Why:
1. Pre-trained weights are distributed in a way that assumes continuous values
2. Binarization discards all magnitude information
3. The model has never learned to be robust to 1-bit weights

**Modern equivalent: Intel's NNCF (Neural Network Compression Framework)**

```python
import nncf

# NNCF can quantize any model for Intel hardware
quantized_model = nncf.quantize(
    model,
    calibration_dataset=calibration_data,
    preset=nncf.QuantizationPreset.MIXED,  # INT8/INT4 mixed
    target_device=nncf.TargetDevice.INTEL_CPU,
)
```

NNCF handles:
- Post-training quantization (PTQ)
- Quantization-aware training (QAT)
- Filter pruning
- Mixed-precision quantization
- LLM-specific optimizations (group quantization, smooth quant)

### C. The SHAVE Processor: Why It Matters

The Movidius Myriad 2's 12 SHAVE cores were designed for **vision pipelines**, not general compute:

```
SHAVE Core Architecture:
├── 128-bit SIMD unit (for bit operations)
├── 64KB instruction memory
├── 32KB data memory
├── No hardware FPU (float is emulated)
├── Hardware bit-shift/rotate (1 cycle)
└── VLIW: 4 instructions per cycle
```

This is why XNOR-Net was so effective — the hardware was literally designed for bitwise operations, not floating-point math.

**Modern contrast: NVIDIA Tensor Cores**

```
Tensor Core Architecture:
├── FP16 multiply-accumulate (1 cycle for 4×4 matrix)
├── INT8 multiply-accumulate (2 cycles for 4×4 matrix)
├── TF32 (tensor float) for training
├── 192 KB shared memory per SM
└── Up to 132 SMs per GPU (H100 = 17,408 FP16 TFLOPS)
```

---

## 5. What a Modern Version Looks Like

### For 1-bit LLMs

| Tool | What It Does | Link |
|------|-------------|------|
| **BitNet b1.58** | Microsoft's ternary LLM architecture | [github.com/microsoft/BitNet](https://github.com/microsoft/BitNet) |
| **Unsloth** | 2x faster LLM fine-tuning, 60% less memory | [github.com/unslothai/unsloth](https://github.com/unslothai/unsloth) |
| **AutoRound** | Intel's quantization toolkit (W4A8, W4A16) | [github.com/intel/auto-round](https://github.com/intel/auto-round) |
| **vLLM** | PagedAttention for serving LLMs | [github.com/vllm-project/vllm](https://github.com/vllm-project/vllm) |
| **NNCF** | Intel's neural network compression | [github.com/openvinotoolkit/nncf](https://github.com/openvinotoolkit/nncf) |

### For Edge LLMs (Running on Intel Hardware)

| Tool | What It Does | Link |
|------|-------------|------|
| **OpenVINO GenAI** | LLM inference on Intel CPU/GPU/VPU | [github.com/openvinotoolkit/openvino.genai](https://github.com/openvinotoolkit/openvino.genai) |
| **Intel Extension for PyTorch** | Optimized PyTorch on Intel | [github.com/intel/intel-extension-for-pytorch](https://github.com/intel/intel-extension-for-pytorch) |
| **Ollama** | Local LLM serving (uses GGUF quantization) | [github.com/ollama/ollama](https://github.com/ollama/ollama) |
| **llama.cpp** | CPU-optimized LLM inference | [github.com/ggerganov/llama.cpp](https://github.com/ggerganov/llama.cpp) |

### Modern Quantization Methods

| Method | Precision | Speed vs FP16 | Quality | Use Case |
|--------|-----------|---------------|---------|----------|
| XNOR-Net (this repo) | 1-bit | 5-10x | Poor for LLMs | Vision CNNs |
| BitNet b1.58 | 1.58-bit | 3-5x | Good for LLMs | LLM inference |
| GPTQ | 4-bit (INT4) | 2-3x | Excellent | LLM serving |
| AWQ | 4-bit (W4A16) | 2-3x | Excellent | LLM serving |
| GGUF | 2-8 bit | 2-4x | Good | CPU inference |
| W4A8 (NNCF) | 4-bit W, 8-bit A | 2-3x | Excellent | Intel hardware |
| QLoRA | 4-bit training | 1x (training) | Good | Fine-tuning |

---

## 6. Bridge: From Movidius to vLLM

### The Evolution Path

```
2017: Movidius NCS + NCSDK
  ↓ (Intel acquires Movidius)
2018: Movidius NCS2 + OpenVINO
  ↓ (LLM revolution begins)
2020: GPT-3 (175B params — needs A100 GPUs)
  ↓ (Edge AI diverges from LLM serving)
2023: vLLM + PagedAttention (serving 70B on 4×A100)
  ↓ (Quantization improves)
2024: BitNet b1.58, GPTQ, AWQ (4-bit LLMs on consumer hardware)
  ↓ (Intel enters LLM space)
2025: OpenVINO GenAI + Intel Gaudi (LLMs on Intel hardware)
```

### If You Want to Run Qwen 3.6 on Intel Hardware

**Do NOT use Movidius NCS.** Use:

```bash
# Option 1: vLLM on Intel GPU (Arc, Data Center GPU Max)
pip install vllm
vllm serve Qwen/Qwen3.6-27B --quantization awq

# Option 2: OpenVINO GenAI on Intel CPU/GPU
pip install openvino-genai
python -c "
import openvino_genai as ov_genai
pipe = ov_genai.LLMPipeline('Qwen3.6-27B-int4', 'GPU')
pipe.generate('Hello, world!')
"

# Option 3: llama.cpp on CPU (any Intel/AMD)
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp && make
./main -m qwen3.6-27b-q4_k_m.gguf -p 'Hello'
```

### The Key Insight

**Binary quantization (1-bit) is a 2016 vision CNN technique.**

**Modern LLM quantization (4-bit) is a 2024 serving technique.**

They solve different problems:
- **XNOR-Net:** Make vision CNNs faster on resource-constrained edge devices
- **GPTQ/AWQ:** Make LLMs fit in GPU memory while maintaining quality

The Pirahansiah repository is valuable for **understanding the low-level math** of bitwise operations — knowledge that is still relevant for:
- Custom CUDA kernels for XNOR-Net inference
- FPGA implementations of binary neural networks
- Research into ternary/binary LLMs (BitNet b1.58)
- Educational purposes (understanding quantization fundamentals)

---

## 7. Timeline: Edge AI Evolution

```
2015 │ YOLOv1 — Real-time object detection (45 FPS on Titan X)
     │ AlexNet, VGG — Large CNNs for ImageNet
     │
2016 │ XNOR-Net — Binary CNNs for edge deployment ← THIS REPO
     │ ResNet — Deep residual networks
     │ MobileNet — Lightweight CNNs for mobile
     │
2017 │ Movidius NCS — First USB AI accelerator
     │ YOLOv2 — Faster, more accurate detection
     │ NCSDK — Intel's first edge AI SDK
     │
2018 │ OpenVINO — Universal Intel inference SDK
     │ Movidius NCS2 — 8x faster than NCS
     │ BERT — Transformers for NLP (but too large for edge)
     │
2019 │ YOLOv3 — Multi-scale detection
     │ TensorRT 6 — NVIDIA edge optimization
     │ Edge TPU — Google's USB AI accelerator
     │
2020 │ GPT-3 — 175B parameters (needs A100 GPUs)
     │ YOLOv4 — CSP-based detection
     │ Jetson Nano — NVIDIA's edge AI platform
     │
2021 │ YOLOv5 — PyTorch-native detection
     │ Hailo-8 — 26 TOPS/W AI accelerator
     │ Hugging Face — Transformers library goes mainstream
     │
2022 │ ChatGPT — LLMs go mainstream
     │ YOLOv8 — Ultralytics ecosystem
     │ vLLM — PagedAttention for LLM serving
     │
2023 │ Llama 2 — Open-source LLMs
     │ Axelera Metis — M.2 AI accelerator
     │ ONNX Runtime — Cross-platform inference
     │
2024 │ BitNet b1.58 — Ternary LLMs (1.58-bit)
     │ GPTQ, AWQ — 4-bit LLM quantization
     │ Hailo-15 — 30 TOPS vision processor
     │ OpenVINO 2024 — LLM support added
     │
2025 │ Llama 3 — 405B open-source model
     │ vLLM 0.6+ — Multi-GPU serving
     │ Intel Gaudi 3 — Datacenter LLM training/inference
     │ This repo: Extended with multi-chip support (Hailo, Axelera, etc.)
     │
2026 │ GPT-5 — (Expected)
     │ Claude 4 — (Expected)
     │ 2-bit quantization becoming practical
     │ Edge LLMs on smartphones (Snapdragon 8 Elite, Apple A18 Pro)
```

---

## Summary

| Aspect | This Repo (2017) | Modern (2025) |
|--------|-------------------|---------------|
| **Chip** | Movidius NCS (512MB) | Hailo-8 (26 TOPS/W), Jetson Orin (275 TOPS) |
| **SDK** | NCSDK (closed) | OpenVINO, ONNX Runtime, vLLM |
| **Quantization** | 1-bit XNOR | 4-bit GPTQ/AWQ, 1.58-bit BitNet |
| **Models** | MobileNet, SqueezeNet | Llama 3, Qwen 3.6, GPT-4 |
| **Task** | Image classification | LLM serving, multimodal AI |
| **Power** | 1.5W (USB) | 70W (Jetson) to 700W (H100) |

**The Pirahansiah repository is a time capsule of the moment when AI first left the datacenter.**

It captures the exact engineering challenges of 2017:
- How to make neural networks run on 512MB of RAM
- How to replace expensive floating-point math with cheap bitwise operations
- How to build a toolchain for hardware that had no existing software ecosystem

These same challenges — memory efficiency, compute efficiency, toolchain maturity — are the exact challenges that modern LLM quantization (GPTQ, AWQ, BitNet) is solving today, just at a 50x larger scale.

---

*This document was written as part of the Binary-DNN-for-Intel-Movidius-Neural-Compute-Stick repository modernization.*
*Last updated: June 2026*
