"""Model optimization: pruning, knowledge distillation, and compression."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Literal


class ModelOptimizer:
    """Structured pruning, knowledge distillation, and channel pruning for edge models."""

    def __init__(self, model: nn.Module) -> None:
        self.model = model

    @staticmethod
    def _get_conv_layers(model: nn.Module) -> list[tuple[str, nn.Conv2d]]:
        """Extract all Conv2d layers with their names."""
        convs = []
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d):
                convs.append((name, module))
        return convs

    def prune(
        self,
        amount: float = 0.3,
        method: Literal["magnitude", "structured"] = "magnitude",
    ) -> nn.Module:
        """Prune model weights.

        Args:
            amount: Fraction of weights/channels to remove (0.0-1.0).
            method: 'magnitude' for unstructured, 'structured' for filter pruning.

        Returns:
            Pruned model (in-place modified).
        """
        if method == "magnitude":
            self._magnitude_prune(amount)
        else:
            self._structured_prune(amount)
        return self.model

    def _magnitude_prune(self, amount: float) -> None:
        """Unstructured magnitude-based pruning."""
        all_weights = []
        for param in self.model.parameters():
            all_weights.append(param.data.abs().flatten())

        if not all_weights:
            return
        all_weights_cat = torch.cat(all_weights)
        threshold = torch.quantile(all_weights_cat, amount)

        for param in self.model.parameters():
            mask = param.data.abs() >= threshold
            param.data *= mask.float()

    def _structured_prune(self, amount: float) -> None:
        """Structured channel pruning: remove entire filters by L1 norm."""
        for name, module in self._get_conv_layers(self.model):
            if module.out_channels <= 1:
                continue
            num_prune = int(module.out_channels * amount)
            if num_prune <= 0:
                continue
            norms = module.weight.data.view(module.out_channels, -1).abs().mean(dim=1)
            _, indices = torch.sort(norms)
            keep = indices[num_prune:]
            if len(keep) == 0:
                continue
            module.weight.data = module.weight.data[keep]
            if module.bias is not None:
                module.bias.data = module.bias.data[keep]
            module.out_channels = len(keep)

    def distill(
        self,
        teacher: nn.Module,
        train_loader: torch.utils.data.DataLoader,
        epochs: int = 5,
        lr: float = 0.001,
        temperature: float = 3.0,
        alpha: float = 0.5,
        device: torch.device = torch.device("cpu"),
    ) -> nn.Module:
        """Knowledge distillation from teacher to student (self.model).

        Args:
            teacher: Pre-trained teacher model.
            train_loader: Training data loader.
            epochs: Number of fine-tuning epochs.
            lr: Learning rate.
            temperature: Softmax temperature for distillation.
            alpha: Balance between hard and soft loss.
            device: Training device.

        Returns:
            Distilled student model.
        """
        teacher.eval().to(device)
        self.model.train().to(device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        for epoch in range(epochs):
            total_loss = 0.0
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)

                with torch.no_grad():
                    teacher_logits = teacher(images)
                student_logits = self.model(images)

                hard_loss = F.cross_entropy(student_logits, labels)
                soft_loss = F.kl_div(
                    F.log_softmax(student_logits / temperature, dim=1),
                    F.softmax(teacher_logits / temperature, dim=1),
                    reduction="batchmean",
                ) * (temperature ** 2)
                loss = alpha * soft_loss + (1 - alpha) * hard_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

        return self.model

    def compress(
        self,
        method: Literal["prune", "distill", "prune_then_distill"] = "prune",
        teacher: nn.Module | None = None,
        train_loader: torch.utils.data.DataLoader | None = None,
        prune_amount: float = 0.3,
        epochs: int = 5,
        lr: float = 0.001,
        device: torch.device = torch.device("cpu"),
    ) -> nn.Module:
        """Compress model using specified method.

        Combines pruning and/or distillation for maximum compression.
        """
        if method == "prune":
            self.prune(amount=prune_amount, method="structured")
        elif method == "distill" and teacher is not None and train_loader is not None:
            self.distill(teacher, train_loader, epochs=epochs, lr=lr, device=device)
        elif method == "prune_then_distill" and teacher is not None and train_loader is not None:
            self.prune(amount=prune_amount, method="structured")
            self.distill(teacher, train_loader, epochs=epochs, lr=lr, device=device)
        return self.model

    @staticmethod
    def benchmark(
        model: nn.Module,
        input_shape: tuple[int, ...] = (1, 3, 32, 32),
        num_runs: int = 100,
        device: torch.device = torch.device("cpu"),
    ) -> dict[str, float]:
        """Benchmark model for latency, FLOPs, and parameter count.

        Returns dict with avg_latency_ms, total_params, estimated_flops, and
        model_size_mb.
        """
        import time

        model.eval().to(device)
        dummy = torch.randn(*input_shape).to(device)

        with torch.no_grad():
            _ = model(dummy)

        times = []
        with torch.no_grad():
            for _ in range(num_runs):
                start = time.perf_counter()
                _ = model(dummy)
                end = time.perf_counter()
                times.append((end - start) * 1000)

        total_params = sum(p.numel() for p in model.parameters())
        model_bytes = sum(p.numel() * p.element_size() for p in model.parameters())

        return {
            "avg_latency_ms": sum(times) / len(times),
            "min_latency_ms": min(times),
            "max_latency_ms": max(times),
            "total_params": total_params,
            "estimated_flops": total_params * 2 * input_shape[-1] * input_shape[-2],
            "model_size_mb": model_bytes / (1024 * 1024),
        }

    @staticmethod
    def count_flops(model: nn.Module, input_shape: tuple[int, ...] = (1, 3, 32, 32)) -> int:
        """Estimate FLOPs for convolutional models using hook-based counting."""
        flops = {"count": 0}

        def hook(module: nn.Module, inp: tuple, out: torch.Tensor) -> None:
            if isinstance(module, nn.Conv2d):
                batch = out.shape[0]
                out_c, out_h, out_w = out.shape[1], out.shape[2], out.shape[3]
                k = module.kernel_size
                in_c = inp[0].shape[1]
                flops["count"] += batch * out_c * out_h * out_w * in_c * k[0] * k[1]
            elif isinstance(module, nn.Linear):
                flops["count"] += module.in_features * module.out_features

        hooks = []
        for module in model.modules():
            hooks.append(module.register_forward_hook(hook))

        dummy = torch.randn(*input_shape)
        with torch.no_grad():
            _ = model(dummy)

        for h in hooks:
            h.remove()

        return flops["count"]
