"""Performance benchmarking suite for model comparison."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn


class BenchmarkSuite:
    """Benchmark models for latency, throughput, memory, and power.

    Generates comparison reports and tables.
    """

    def __init__(
        self,
        device: torch.device = torch.device("cpu"),
        num_runs: int = 100,
        warmup_runs: int = 10,
    ) -> None:
        self.device = device
        self.num_runs = num_runs
        self.warmup_runs = warmup_runs
        self.results: dict[str, dict[str, Any]] = {}

    def _warmup(self, model: nn.Module, dummy: torch.Tensor) -> None:
        """Run warmup iterations to stabilize cache."""
        model.eval()
        with torch.no_grad():
            for _ in range(self.warmup_runs):
                _ = model(dummy.to(self.device))

    def measure_latency(
        self, model: nn.Module, input_shape: tuple[int, ...], name: str = "model"
    ) -> dict[str, float]:
        """Measure single-sample latency in milliseconds."""
        dummy = torch.randn(*input_shape).to(self.device)
        model.eval().to(self.device)
        self._warmup(model, dummy)

        times = []
        with torch.no_grad():
            for _ in range(self.num_runs):
                start = time.perf_counter()
                _ = model(dummy)
                end = time.perf_counter()
                times.append((end - start) * 1000)

        stats = {
            "avg_ms": float(np.mean(times)),
            "std_ms": float(np.std(times)),
            "min_ms": float(np.min(times)),
            "max_ms": float(np.max(times)),
            "p50_ms": float(np.percentile(times, 50)),
            "p95_ms": float(np.percentile(times, 95)),
            "p99_ms": float(np.percentile(times, 99)),
        }
        self.results.setdefault(name, {}).update(stats)
        return stats

    def measure_throughput(
        self, model: nn.Module, input_shape: tuple[int, ...], name: str = "model"
    ) -> float:
        """Measure throughput in frames per second."""
        batch_shape = (input_shape[0],) + input_shape[1:]
        dummy = torch.randn(*batch_shape).to(self.device)
        model.eval().to(self.device)
        self._warmup(model, dummy)

        start = time.perf_counter()
        count = 0
        with torch.no_grad():
            while time.perf_counter() - start < 5.0:
                _ = model(dummy)
                count += batch_shape[0]

        elapsed = time.perf_counter() - start
        fps = count / elapsed
        self.results.setdefault(name, {})["throughput_fps"] = fps
        return fps

    def measure_memory(self, model: nn.Module, name: str = "model") -> dict[str, float]:
        """Measure model memory usage in MB."""
        param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
        buffer_bytes = sum(b.numel() * b.element_size() for b in model.buffers())

        stats = {
            "param_mb": param_bytes / (1024 * 1024),
            "buffer_mb": buffer_bytes / (1024 * 1024),
            "total_mb": (param_bytes + buffer_bytes) / (1024 * 1024),
            "total_params": sum(p.numel() for p in model.parameters()),
        }

        try:
            import psutil
            import os

            process = psutil.Process(os.getpid())
            stats["rss_mb"] = process.memory_info().rss / (1024 * 1024)
        except ImportError:
            pass

        self.results.setdefault(name, {}).update(stats)
        return stats

    def compare_models(
        self,
        models: dict[str, nn.Module],
        input_shape: tuple[int, ...] = (1, 3, 32, 32),
    ) -> dict[str, dict[str, Any]]:
        """Benchmark multiple models and return comparison dict.

        Args:
            models: Dict mapping model names to model instances.
            input_shape: Input tensor shape for benchmarking.

        Returns:
            Dict mapping model names to their benchmark results.
        """
        comparison = {}
        for name, model in models.items():
            model.to(self.device)
            print(f"Benchmarking {name}...")
            latency = self.measure_latency(model, input_shape, name)
            throughput = self.measure_throughput(model, input_shape, name)
            memory = self.measure_memory(model, name)

            comparison[name] = {
                "latency": latency,
                "throughput_fps": throughput,
                "memory": memory,
            }

        return comparison

    def generate_report(self, output_path: str | Path | None = None) -> str:
        """Generate a markdown comparison table from collected results.

        Returns:
            Markdown string with benchmark results table.
        """
        if not self.results:
            return "No benchmark results available."

        lines = [
            "# Benchmark Report",
            "",
            "| Model | Avg Latency (ms) | Throughput (FPS) | Params | Size (MB) |",
            "|-------|-----------------|-----------------|--------|-----------|",
        ]

        for name, data in self.results.items():
            avg_ms = data.get("avg_ms", "N/A")
            fps = data.get("throughput_fps", "N/A")
            params = data.get("total_params", "N/A")
            size_mb = data.get("total_mb", "N/A")

            avg_str = f"{avg_ms:.2f}" if isinstance(avg_ms, (int, float)) else avg_ms
            fps_str = f"{fps:.1f}" if isinstance(fps, (int, float)) else fps
            params_str = f"{params:,}" if isinstance(params, (int, float)) else params
            size_str = f"{size_mb:.2f}" if isinstance(size_mb, (int, float)) else size_mb

            lines.append(f"| {name} | {avg_str} | {fps_str} | {params_str} | {size_str} |")

        report = "\n".join(lines)

        if output_path is not None:
            Path(output_path).write_text(report)

        return report

    @staticmethod
    def estimate_power(
        model: nn.Module,
        input_shape: tuple[int, ...] = (1, 3, 32, 32),
        tdp_watts: float = 15.0,
    ) -> dict[str, float]:
        """Estimate power consumption based on model size and FLOPs.

        Args:
            model: PyTorch model.
            input_shape: Input tensor shape.
            tdp_watts: Thermal design power of target device.

        Returns:
            Dict with estimated power metrics.
        """
        total_params = sum(p.numel() for p in model.parameters())
        flops = total_params * 2 * input_shape[-1] * input_shape[-2]

        energy_joules = flops * 1e-12 * 0.5
        power_watts = energy_joules * 30

        return {
            "estimated_power_watts": min(power_watts, tdp_watts),
            "tdp_watts": tdp_watts,
            "energy_efficiency_gops_watt": flops / 1e9 / max(power_watts, 1e-6),
            "flops": flops,
        }

    def save_results(self, output_path: str | Path) -> None:
        """Save benchmark results to JSON."""
        import json

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        serializable = {}
        for name, data in self.results.items():
            serializable[name] = {
                k: float(v) if isinstance(v, (int, float)) else v
                for k, v in data.items()
            }

        output_path.write_text(json.dumps(serializable, indent=2))
