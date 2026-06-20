"""Edge deployment for all major AI chips — Movidius, Hailo, Axelera, Qualcomm, Coral, Apple NE, ARM Ethos, TensorRT."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import torch.nn as nn


class EdgeDeployer:
    """Deploy optimized models to Intel Movidius NCS2, OpenVINO, ONNX, and TensorRT."""

    def __init__(self, model: nn.Module | None = None, input_shape: tuple[int, ...] = (1, 3, 32, 32)) -> None:
        self.model = model
        self.input_shape = input_shape

    def export_onnx(
        self,
        output_path: str | Path,
        opset: int = 13,
        dynamic_batch: bool = True,
    ) -> Path:
        """Export PyTorch model to ONNX format.

        Args:
            output_path: Path for the .onnx file.
            opset: ONNX opset version.
            dynamic_batch: Enable dynamic batch dimension.

        Returns:
            Path to exported ONNX file.
        """
        if self.model is None:
            raise ValueError("No model loaded. Set model before export.")

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self.model.eval()
        dummy = torch.randn(*self.input_shape)

        dynamic_axes = {}
        if dynamic_batch:
            dynamic_axes = {"input": {0: "batch"}, "output": {0: "batch"}}

        torch.onnx.export(
            self.model,
            dummy,
            str(output_path),
            opset_version=opset,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes=dynamic_axes,
        )
        return output_path

    @staticmethod
    def deploy_onnx(
        onnx_path: str | Path,
        input_data: np.ndarray,
        num_threads: int = 4,
    ) -> np.ndarray:
        """Run inference with ONNX Runtime.

        Args:
            onnx_path: Path to ONNX model.
            input_data: Input numpy array.
            num_threads: Number of CPU threads.

        Returns:
            Model output as numpy array.
        """
        try:
            import onnxruntime as ort
        except ImportError as e:
            raise ImportError("onnxruntime required. Install with: pip install onnxruntime") from e

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = num_threads
        opts.inter_op_num_threads = num_threads

        providers = ["CPUExecutionProvider"]
        session = ort.InferenceSession(str(onnx_path), opts, providers=providers)

        input_name = session.get_inputs()[0].name
        outputs = session.run(None, {input_name: input_data.astype(np.float32)})
        return outputs[0]

    @staticmethod
    def deploy_openvino(
        model_path: str | Path,
        input_data: np.ndarray,
        device: str = "CPU",
    ) -> np.ndarray:
        """Run inference with Intel OpenVINO.

        Args:
            model_path: Path to OpenVINO IR (.xml) or ONNX model.
            input_data: Input numpy array.
            device: Target device ('CPU', 'GPU', 'MYRIAD' for Movidius).

        Returns:
            Model output as numpy array.
        """
        try:
            from openvino.runtime import Core
        except ImportError as e:
            raise ImportError(
                "openvino required. Install with: pip install openvino"
            ) from e

        core = Core()
        model = core.read_model(str(model_path))
        compiled = core.compile_model(model, device)
        input_layer = compiled.input(0)
        output_layer = compiled.output(0)
        result = compiled({input_layer: input_data.astype(np.float32)})
        return result[output_layer]

    @staticmethod
    def deploy_movidius(
        onnx_path: str | Path,
        input_data: np.ndarray,
    ) -> np.ndarray:
        """Deploy to Intel Movidius Neural Compute Stick 2 via OpenVINO.

        The NCS2 uses the MYRIAD execution device in OpenVINO.

        Args:
            onnx_path: Path to ONNX model.
            input_data: Input numpy array.

        Returns:
            Model output as numpy array.
        """
        return EdgeDeployer.deploy_openvino(
            model_path=str(onnx_path),
            input_data=input_data,
            device="MYRIAD",
        )

    @staticmethod
    def deploy_tensorrt(
        onnx_path: str | Path,
        input_data: np.ndarray,
        precision: Literal["fp32", "fp16", "int8"] = "fp16",
        max_batch_size: int = 8,
    ) -> np.ndarray:
        """Run inference with NVIDIA TensorRT.

        Args:
            onnx_path: Path to ONNX model.
            input_data: Input numpy array.
            precision: Inference precision.
            max_batch_size: Maximum batch size for optimization.

        Returns:
            Model output as numpy array.
        """
        try:
            import tensorrt as trt
            import pycuda.driver as cuda
            import pycuda.autoinit  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "tensorrt and pycuda required. Install via NVIDIA package manager."
            ) from e

        logger = trt.Logger(trt.Logger.WARNING)
        builder = trt.Builder(logger)
        network = builder.create_network(
            1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        )
        parser = trt.OnnxParser(network, logger)

        with open(str(onnx_path), "rb") as f:
            if not parser.parse(f.read()):
                for i in range(parser.num_errors):
                    print(parser.get_error(i))
                raise RuntimeError("ONNX parse failed")

        config = builder.create_builder_config()
        if precision == "fp16" and builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
        elif precision == "int8" and builder.platform_has_fast_int8:
            config.set_flag(trt.BuilderFlag.INT8)

        config.max_workspace_size = 1 << 30
        serialized = builder.build_serialized_network(network, config)

        runtime = trt.Runtime(logger)
        engine = runtime.deserialize_cuda_engine(serialized)
        context = engine.create_execution_context()

        d_input = cuda.mem_alloc(input_data.nbytes)
        output_shape = (1, 10)
        output = np.empty(output_shape, dtype=np.float32)
        d_output = cuda.mem_alloc(output.nbytes)

        stream = cuda.Stream()
        cuda.memcpy_htod_async(d_input, input_data, stream)
        context.execute_async_v2(
            bindings=[int(d_input), int(d_output)], stream_handle=stream.handle
        )
        cuda.memcpy_dtoh_async(output, d_output, stream)
        stream.synchronize()

        return output

    def benchmark_deployment(
        self,
        deployment_fn,
        input_data: np.ndarray,
        num_runs: int = 100,
    ) -> dict[str, float]:
        """Benchmark a deployment function for latency and throughput.

        Args:
            deployment_fn: Callable that takes input_data and returns output.
            input_data: Input data for inference.
            num_runs: Number of benchmark iterations.

        Returns:
            Dict with latency stats and throughput.
        """
        times = []
        for _ in range(num_runs):
            start = time.perf_counter()
            _ = deployment_fn(input_data)
            end = time.perf_counter()
            times.append(end - start)

        avg_latency = sum(times) / len(times)
        return {
            "avg_latency_ms": avg_latency * 1000,
            "min_latency_ms": min(times) * 1000,
            "max_latency_ms": max(times) * 1000,
            "throughput_fps": 1.0 / avg_latency if avg_latency > 0 else 0,
            "num_runs": num_runs,
        }

    @staticmethod
    def deploy_hailo(
        onnx_path: str | Path,
        input_data: np.ndarray,
        chip: str = "hailo8",
    ) -> np.ndarray:
        """Deploy to Hailo AI accelerator (Hailo-8/8L/15).

        Converts ONNX → HEF (Hailo Executable Format) via Hailo Dataflow Compiler.

        Args:
            onnx_path: Path to ONNX model.
            input_data: Input numpy array.
            chip: Hailo chip variant ('hailo8', 'hailo8l', 'hailo15').

        Returns:
            Model output as numpy array.
        """
        try:
            from hailo_sdk_client import ClientRunner
        except ImportError as e:
            raise ImportError(
                "hailo-sdk-client required. Install: pip install hailo-sdk-client"
            ) from e

        onnx_path = Path(onnx_path)
        runner = ClientRunner(hef="", network_script="")

        # Parse ONNX model
        runner.load_model_from_onnx(onnx_path)

        # Quantize with calibration data
        calib_data = np.random.randn(100, *input_data.shape[1:]).astype(np.float32)
        runner.optimize(calib_data)

        # Compile to HEF
        hef_path = onnx_path.with_suffix(".hef")
        runner.save_hef(str(hef_path))

        # Run inference on Hailo device
        from hailo_platform import VDevice, HailoStreamInterface, ConfigureParams, \
            InferVStreams, InputVStreamParams, OutputVStreamParams, Format

        vdevice = VDevice()
        configure_params = ConfigureParams.create_from_hef(hef_path)
        iface = vdevice.create_interface(HailoStreamInterface.PCIe)
        configure_params.set_power_saving(PowerSavingMode.SPEED)

        input_params = InputVStreamParams.make_from_network_model(
            runner.get_network_model(), quantized=False
        )
        output_params = OutputVStreamParams.make_from_network_model(
            runner.get_network_model(), quantized=True
        )

        with InferVStreams(iface, input_params, output_params) as pipeline:
            input_data_dict = {runner.get_input_vstream_name(): input_data}
            output = pipeline.infer(input_data_dict)

        return output

    @staticmethod
    def deploy_axelera(
        onnx_path: str | Path,
        input_data: np.ndarray,
    ) -> np.ndarray:
        """Deploy to Axelera Metis AIPU.

        Converts ONNX → Axelera format via AXDL compiler.

        Args:
            onnx_path: Path to ONNX model.
            input_data: Input numpy array.

        Returns:
            Model output as numpy array.
        """
        try:
            from axelera.ax_core import Network, Meta
        except ImportError as e:
            raise ImportError(
                "axelera-ai-sdk required. Install: pip install axelera-ai-sdk"
            ) from e

        onnx_path = Path(onnx_path)

        # Build AXDL model
        network = Network()
        network.load(str(onnx_path))
        axdl_path = onnx_path.with_suffix(".axdl")
        network.save(str(axdl_path))

        # Load and run on Metis AIPU
        model = Meta()
        model.load(str(axdl_path))

        output = model.infer(input_data)
        return output

    @staticmethod
    def deploy_qualcomm(
        onnx_path: str | Path,
        input_data: np.ndarray,
        target: str = "hexagon_dsp",
    ) -> np.ndarray:
        """Deploy to Qualcomm AI processor (Hexagon DSP / Cloud AI 100).

        Uses Qualcomm Neural Processing SDK (SNPE) or QNN.

        Args:
            onnx_path: Path to ONNX model.
            input_data: Input numpy array.
            target: 'hexagon_dsp', 'cloud_ai100', or 'qcs6490'.

        Returns:
            Model output as numpy array.
        """
        try:
            import snpe
        except ImportError as e:
            raise ImportError(
                "Qualcomm Neural Processing SDK required. "
                "Install from: https://developer.qualcomm.com/software/qualcomm-neural-processing-sdk"
            ) from e

        onnx_path = Path(onnx_path)

        # Convert ONNX → DLC (Deep Learning Container)
        dlc_path = onnx_path.with_suffix(".dlc")
        snpe.onnx_to_dlc(str(onnx_path), str(dlc_path))

        # Run inference
        model = snpe.Model(str(dlc_path))
        output = model.infer(input_data)
        return output

    @staticmethod
    def deploy_coral(
        tflite_path: str | Path,
        input_data: np.ndarray,
        edge_tpu: bool = True,
    ) -> np.ndarray:
        """Deploy to Google Coral Edge TPU.

        Args:
            tflite_path: Path to TFLite model (compiled for Edge TPU).
            input_data: Input numpy array (uint8).
            edge_tpu: Use Edge TPU runtime (True) or CPU (False).

        Returns:
            Model output as numpy array.
        """
        try:
            if edge_tpu:
                from pycoral.utils.edgetpu import run_inference
                return run_inference(str(tflite_path), input_data)
            else:
                import tflite_runtime.interpreter as tflite
                interpreter = tflite.Interpreter(model_path=str(tflite_path))
                interpreter.allocate_tensors()
                input_details = interpreter.get_input_details()
                interpreter.set_tensor(input_details[0]['index'], input_data)
                interpreter.invoke()
                output_details = interpreter.get_output_details()
                return interpreter.get_tensor(output_details[0]['index'])
        except ImportError as e:
            raise ImportError(
                "tflite-runtime and pycoral required. "
                "Install: pip install tflite-runtime pycoral"
            ) from e

    @staticmethod
    def deploy_apple_neural_engine(
        mlmodel_path: str | Path,
        input_data: np.ndarray,
    ) -> np.ndarray:
        """Deploy to Apple Neural Engine via Core ML.

        Args:
            mlmodel_path: Path to Core ML .mlmodel or .mlpackage.
            input_data: Input numpy array.

        Returns:
            Model output as numpy array.
        """
        try:
            import coremltools as ct
        except ImportError as e:
            raise ImportError(
                "coremltools required. Install: pip install coremltools"
            ) from e

        model = ct.models.MLModel(str(mlmodel_path))
        output = model.predict({"input": input_data})
        return output

    @staticmethod
    def deploy_arm_ethos(
        tflite_path: str | Path,
        input_data: np.ndarray,
    ) -> np.ndarray:
        """Deploy to ARM Ethos-U MicroNPU via Vela-compiled TFLite.

        Args:
            tflite_path: Path to Vela-compiled TFLite model.
            input_data: Input numpy array (uint8).

        Returns:
            Model output as numpy array.
        """
        try:
            import tflite_runtime.interpreter as tflite
        except ImportError as e:
            raise ImportError(
                "tflite-runtime required. Install: pip install tflite-runtime"
            ) from e

        interpreter = tflite.Interpreter(
            model_path=str(tflite_path),
            experimental_delegates=[
                tflite.load_delegate('libethosu_delegate.so')
            ]
        )
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        output_details = interpreter.get_output_details()
        return interpreter.get_tensor(output_details[0]['index'])

    @staticmethod
    def deploy_rockchip(
        rknn_path: str | Path,
        input_data: np.ndarray,
    ) -> np.ndarray:
        """Deploy to Rockchip RKNN NPU.

        Args:
            rknn_path: Path to RKNN model.
            input_data: Input numpy array.

        Returns:
            Model output as numpy array.
        """
        try:
            from rknnlite.api import RKNNLite
        except ImportError as e:
            raise ImportError(
                "rknn-toolkit2 required. Install: pip install rknn-toolkit2"
            ) from e

        rknn = RKNNLite()
        rknn.config(
            mean_values=[[123.675, 116.28, 103.53]],
            std_values=[[58.395, 57.12, 57.375]],
            target_platform='rk3588'
        )
        rknn.load_rknn(str(rknn_path))
        rknn.init_runtime()
        output = rknn.inference(inputs=[input_data])
        rknn.release()
        return output[0]

    @staticmethod
    def get_supported_platforms() -> dict[str, list[str]]:
        """Return supported deployment platforms and their backends."""
        return {
            "intel_movidius_ncs2": ["openvino_myriad"],
            "openvino": ["cpu", "gpu", "myriad"],
            "onnx_runtime": ["cpu", "cuda", "directml"],
            "tensorrt": ["cuda"],
            "hailo8": ["hailo_hef"],
            "hailo8l": ["hailo_hef"],
            "hailo15": ["hailo_hef"],
            "axelera_metis": ["axdl"],
            "qualcomm_hexagon": ["snpe_dlc"],
            "qualcomm_cloud_ai100": ["qnn"],
            "google_coral": ["tflite_edge_tpu"],
            "apple_neural_engine": ["coreml"],
            "arm_ethos_u55": ["vela_tflite"],
            "arm_ethos_u85": ["vela_tflite"],
            "rockchip_rk3588": ["rknn"],
        }
