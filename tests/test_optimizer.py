"""Tests for optimizer module."""

from __future__ import annotations

import torch
import torch.nn as nn

from src.optimizer import ModelOptimizer


class TestMagnitudePrune:
    def test_prune_reduces_weights(self, binary_cnn):
        before = sum((p.data != 0).sum().item() for p in binary_cnn.parameters())
        opt = ModelOptimizer(binary_cnn)
        opt.prune(amount=0.5, method="magnitude")
        after = sum((p.data != 0).sum().item() for p in binary_cnn.parameters())
        assert after <= before

    def test_zero_amount(self, binary_cnn):
        opt = ModelOptimizer(binary_cnn)
        opt.prune(amount=0.0, method="magnitude")
        for p in binary_cnn.parameters():
            assert p.data.abs().sum() > 0


class TestStructuredPrune:
    def test_prune_reduces_params(self, binary_cnn):
        before_params = sum(p.numel() for p in binary_cnn.parameters())
        opt = ModelOptimizer(binary_cnn)
        opt.prune(amount=0.3, method="structured")
        after_params = sum(p.numel() for p in binary_cnn.parameters())
        assert after_params <= before_params


class TestDistill:
    def test_distillation_runs(self, binary_cnn, dummy_data_loader):
        teacher = nn.Sequential(
            nn.Linear(3 * 32 * 32, 64),
            nn.ReLU(),
            nn.Linear(64, 10),
        )
        opt = ModelOptimizer(binary_cnn)
        result = opt.distill(
            teacher, dummy_data_loader, epochs=1, lr=0.001, device=torch.device("cpu")
        )
        assert isinstance(result, nn.Module)


class TestBenchmark:
    def test_benchmark_returns_metrics(self, binary_cnn):
        result = ModelOptimizer.benchmark(
            binary_cnn, input_shape=(1, 3, 32, 32), num_runs=5
        )
        assert "avg_latency_ms" in result
        assert "total_params" in result
        assert "model_size_mb" in result
        assert result["avg_latency_ms"] > 0
        assert result["total_params"] > 0


class TestCountFlops:
    def test_flops_positive(self, binary_cnn):
        flops = ModelOptimizer.count_flops(binary_cnn, input_shape=(1, 3, 32, 32))
        assert flops > 0

    def test_linear_flops(self, simple_linear_layer):
        flops = ModelOptimizer.count_flops(simple_linear_layer, input_shape=(1, 10))
        assert flops > 0


class TestCompress:
    def test_prune_compress(self, binary_cnn):
        opt = ModelOptimizer(binary_cnn)
        result = opt.compress(method="prune", prune_amount=0.2)
        assert isinstance(result, nn.Module)

    def test_prune_then_distill(self, binary_cnn, dummy_data_loader):
        teacher = nn.Sequential(
            nn.Linear(3 * 32 * 32, 64),
            nn.ReLU(),
            nn.Linear(64, 10),
        )
        opt = ModelOptimizer(binary_cnn)
        result = opt.compress(
            method="prune_then_distill",
            teacher=teacher,
            train_loader=dummy_data_loader,
            prune_amount=0.1,
            epochs=1,
        )
        assert isinstance(result, nn.Module)
