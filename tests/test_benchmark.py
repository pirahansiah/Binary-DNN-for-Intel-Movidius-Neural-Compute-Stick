"""Tests for benchmark module."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from src.benchmark import BenchmarkSuite
from src.models import BinaryCNN


class TestMeasureLatency:
    def test_latency_returns_stats(self, binary_cnn):
        bs = BenchmarkSuite(num_runs=5, warmup_runs=2)
        stats = bs.measure_latency(binary_cnn, (1, 3, 32, 32), name="test_model")
        assert "avg_ms" in stats
        assert "std_ms" in stats
        assert "min_ms" in stats
        assert "max_ms" in stats
        assert stats["avg_ms"] > 0

    def test_latency_stored(self, binary_cnn):
        bs = BenchmarkSuite(num_runs=5, warmup_runs=2)
        bs.measure_latency(binary_cnn, (1, 3, 32, 32), name="m")
        assert "m" in bs.results
        assert "avg_ms" in bs.results["m"]


class TestMeasureThroughput:
    def test_throughput_positive(self, binary_cnn):
        bs = BenchmarkSuite(num_runs=5, warmup_runs=2)
        fps = bs.measure_throughput(binary_cnn, (1, 3, 32, 32), name="t")
        assert fps > 0


class TestMeasureMemory:
    def test_memory_returns_metrics(self, binary_cnn):
        bs = BenchmarkSuite()
        mem = bs.measure_memory(binary_cnn, name="mem")
        assert "param_mb" in mem
        assert "total_mb" in mem
        assert "total_params" in mem
        assert mem["total_params"] > 0


class TestCompareModels:
    def test_compare_multiple(self):
        m1 = BinaryCNN(num_classes=10)
        m2 = BinaryCNN(num_classes=10)
        bs = BenchmarkSuite(num_runs=3, warmup_runs=1)
        result = bs.compare_models(
            {"model_a": m1, "model_b": m2},
            input_shape=(1, 3, 32, 32),
        )
        assert "model_a" in result
        assert "model_b" in result
        assert "latency" in result["model_a"]
        assert "throughput_fps" in result["model_a"]


class TestGenerateReport:
    def test_empty_report(self):
        bs = BenchmarkSuite()
        report = bs.generate_report()
        assert "No benchmark results" in report

    def test_report_with_results(self, binary_cnn):
        bs = BenchmarkSuite(num_runs=3, warmup_runs=1)
        bs.measure_latency(binary_cnn, (1, 3, 32, 32), name="test")
        report = bs.generate_report()
        assert "| test |" in report
        assert "Avg Latency" in report

    def test_report_to_file(self, binary_cnn, tmp_path):
        bs = BenchmarkSuite(num_runs=3, warmup_runs=1)
        bs.measure_latency(binary_cnn, (1, 3, 32, 32), name="test")
        output = tmp_path / "report.md"
        bs.generate_report(output_path=output)
        assert output.exists()


class TestEstimatePower:
    def test_power_returns_metrics(self, binary_cnn):
        result = BenchmarkSuite.estimate_power(binary_cnn, input_shape=(1, 3, 32, 32))
        assert "estimated_power_watts" in result
        assert "energy_efficiency_gops_watt" in result
        assert "flops" in result
        assert result["flops"] > 0


class TestSaveResults:
    def test_save_results(self, binary_cnn, tmp_path):
        bs = BenchmarkSuite(num_runs=3, warmup_runs=1)
        bs.measure_latency(binary_cnn, (1, 3, 32, 32), name="test")
        output = tmp_path / "results.json"
        bs.save_results(output)
        assert output.exists()
        data = json.loads(output.read_text())
        assert "test" in data
        assert "avg_ms" in data["test"]
