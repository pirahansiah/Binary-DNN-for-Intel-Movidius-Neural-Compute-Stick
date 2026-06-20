"""Tests for quantization module."""

from __future__ import annotations

import torch
import torch.nn as nn

from src.quantize import Quantizer


class TestSymmetricQuantize:
    def test_int8_range(self):
        tensor = torch.tensor([1.0, -1.0, 0.5, -0.5])
        result = Quantizer._symmetric_quantize(tensor, num_bits=8)
        assert result.max().item() <= 1.0
        assert result.min().item() >= -1.0

    def test_int4_range(self):
        tensor = torch.tensor([2.0, -2.0, 1.0, -1.0])
        result = Quantizer._symmetric_quantize(tensor, num_bits=4)
        assert result.abs().max().item() <= 2.0 + 0.1

    def test_zero_tensor(self):
        tensor = torch.zeros(4, 4)
        result = Quantizer._symmetric_quantize(tensor, num_bits=8)
        assert torch.all(result == 0)


class TestScaleZeroPoint:
    def test_zero_tensor(self):
        scale, zp = Quantizer._compute_scale_zero_point(0.0, 0.0)
        assert scale == 1.0
        assert zp == 0

    def test_normal_values(self):
        scale, zp = Quantizer._compute_scale_zero_point(-1.0, 1.0, num_bits=8)
        assert scale > 0
        assert zp == 0


class TestQuantizeInt8:
    def test_output_is_model(self, binary_cnn):
        q = Quantizer()
        result = q.quantize_int8(binary_cnn)
        assert isinstance(result, nn.Module)

    def test_weights_modified(self, binary_cnn):
        q = Quantizer()
        original = binary_cnn.classifier.weight.data.clone()
        result = q.quantize_int8(binary_cnn)
        for name, param in result.named_parameters():
            if "weight" in name:
                assert not torch.equal(param.data, torch.zeros_like(param.data))
                break


class TestQuantizeInt4:
    def test_output_is_model(self, binary_cnn):
        q = Quantizer()
        result = q.quantize_int4(binary_cnn)
        assert isinstance(result, nn.Module)


class TestQuantizeMixed:
    def test_output_is_model(self, binary_cnn):
        q = Quantizer()
        result = q.quantize_mixed(binary_cnn, sensitivity_threshold=0.5)
        assert isinstance(result, nn.Module)


class TestGetNumBitsForMixed:
    def test_high_variance(self):
        tensor = torch.randn(100) * 10
        bits = Quantizer._get_num_bits_for_mixed(tensor, sensitivity=0.5)
        assert bits == 8

    def test_low_variance(self):
        tensor = torch.randn(100) * 0.01
        bits = Quantizer._get_num_bits_for_mixed(tensor, sensitivity=0.5)
        assert bits == 4


class TestMeasureDegradation:
    def test_same_model_no_degradation(self, binary_cnn, dummy_data_loader):
        q = Quantizer()
        result = Quantizer.measure_degradation(binary_cnn, binary_cnn, dummy_data_loader)
        assert result["original_acc"] == result["quantized_acc"]
        assert result["degradation"] == 0.0

    def test_compression_ratio(self, binary_cnn, dummy_data_loader):
        q = Quantizer()
        quantized = q.quantize_int4(binary_cnn)
        result = Quantizer.measure_degradation(binary_cnn, quantized, dummy_data_loader)
        assert result["compression_ratio"] > 1.0


class TestCollectCalibrationData:
    def test_collects_data(self, binary_cnn, dummy_data_loader):
        q = Quantizer(num_calibration_batches=2)
        q.collect_calibration_data(binary_cnn, dummy_data_loader)
        assert len(q.calibration_data) == 2


class TestGetQuantizationStats:
    def test_stats(self, binary_cnn):
        q = Quantizer()
        stats = q.get_quantization_stats(binary_cnn)
        assert "total_params" in stats
        assert "total_bytes_fp32" in stats
        assert "compression_ratio" in stats
        assert stats["total_params"] > 0
