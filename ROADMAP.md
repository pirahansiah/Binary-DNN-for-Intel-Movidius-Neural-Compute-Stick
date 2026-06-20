# ROADMAP.md — Binary DNN for Intel Movidius Neural Compute Stick

## 12-Month Vision

Evolve this project from a research prototype into a production-ready binary DNN deployment toolkit with full OpenVINO 2026 support, multi-hardware targeting, and automated accuracy-latency optimization pipelines.

---

### Q1 — Foundation & Modernization (Months 1-3)

**Milestone: Functional PyTorch 2.x training pipeline with OpenVINO 2026 export**

- [ ] Port training code to Python 3.14 + PyTorch 2.x with type hints
- [ ] Implement custom autograd functions for Straight-Through Estimator (STE) binarization
- [ ] Add OpenVINO 2026 Model Optimizer integration with binary layer support
- [ ] Create automated benchmarking suite (latency, memory, accuracy, power)
- [ ] Set up CI/CD pipeline (GitHub Actions) with conda `py314` environment
- [ ] Write unit tests targeting 80%+ code coverage

### Q2 — Hardware Coverage & Optimization (Months 4-6)

**Milestone: Multi-hardware binary model deployment**

- [ ] Add Intel NPU (Lunar Lake / Arrow Lake) deployment target
- [ ] Implement Apple Neural Engine (ANE) binary model export via Core ML
- [ ] Add Raspberry Pi 5 (ARM Cortex-A76) CPU inference backend with NEON intrinsics
- [ ] Create hardware auto-selection based on available accelerators
- [ ] Benchmark across Movidius NCS2, Intel NPU, ANE, and RPi5
- [ ] Optimize batch inference pipeline for multi-camera scenarios

### Q3 — Advanced Features & Training (Months 7-9)

**Milestone: State-of-the-art binary training with accuracy recovery**

- [ ] Implement BinaryBERT / binary transformer attention layers
- [ ] Add progressive binarization scheduler (FP32 → INT8 → Binary)
- [ ] Integrate knowledge distillation from FP32 teacher to binary student
- [ ] Add binary channel pruning for further memory reduction
- [ ] Implement on-device fine-tuning for domain adaptation
- [ ] Create interactive visualization dashboard for training dynamics

### Q4 — Production & Ecosystem (Months 10-12)

**Milestone: Production deployment toolkit with documentation**

- [ ] Package as `pip install binary-dnn-toolkit` with CLI interface
- [ ] Create Docker images for reproducible builds (Ubuntu 24.04, macOS)
- [ ] Add TensorRT binary model export for NVIDIA Jetson platforms
- [ ] Implement model zoo with pre-trained binary models (ResNet-18, MobileNetV2, YOLO-nano)
- [ ] Write comprehensive documentation with deployment guides per hardware
- [ ] Publish benchmarking paper / technical report

---

## Technical Debt

| ID | Item | Priority | Est. Effort |
|---|---|---|---|
| TD-1 | No automated tests exist | High | 1 week |
| TD-2 | README lacks installation/setup instructions | High | 1 day |
| TD-3 | No dependency management (requirements.txt / pyproject.toml) | High | 1 day |
| TD-4 | No versioning or release tags | Medium | 1 day |
| TD-5 | .gitignore is Delphi-specific, not Python/C++ appropriate | Low | 1 hour |
| TD-6 | No OpenVINO version pinning or compatibility matrix | Medium | 2 days |
| TD-7 | Benchmarking results not reproducible (no seed/dataset versioning) | Medium | 3 days |
| TD-8 | Missing model serialization format documentation | Low | 1 day |

## Future Features

- **Binary Convolution Custom CUDA Kernels**: Hand-optimized CUDA 13 kernels for XNOR-popcount operations on NVIDIA Jetson/RTX hardware
- **ONNX Binary Model Standard**: Contribute binary operator extensions to ONNX spec for cross-framework portability
- **WebAssembly Binary Inference**: Compile binary DNNs to WASM for browser-based edge inference
- **Federated Binary Learning**: Enable distributed training of binary models across edge devices with differential privacy
- **AutoBNN Architecture Search**: Neural architecture search specialized for binary-convertible network topologies
- **Real-Time Video Analytics Pipeline**: End-to-end binary DNN pipeline for multi-camera surveillance with sub-10ms latency
- **Edge-to-Cloud Model Sync**: Automatic model versioning and OTA deployment for fleet management of edge devices
