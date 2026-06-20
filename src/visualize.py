"""Visualization utilities for binary neural networks."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


def plot_weight_distribution(
    model: nn.Module,
    output_path: str | Path | None = None,
    max_layers: int = 20,
) -> None:
    """Plot weight distribution histograms for binary vs FP32 layers.

    Creates a grid showing the distribution of weights before and after binarization.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    layers = []
    for name, param in model.named_parameters():
        if "weight" in name and param.dim() >= 2:
            layers.append((name, param.data.cpu()))
            if len(layers) >= max_layers:
                break

    n = len(layers)
    if n == 0:
        return

    cols = min(4, n)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows))
    if n == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for i, (name, weights) in enumerate(layers):
        ax = axes[i]
        flat = weights.flatten().numpy()
        ax.hist(flat, bins=50, alpha=0.7, color="steelblue", edgecolor="black")
        ax.set_title(name[:30], fontsize=8)
        ax.tick_params(labelsize=6)

    for i in range(n, len(axes)):
        axes[i].set_visible(False)

    plt.suptitle("Weight Distributions", fontsize=12)
    plt.tight_layout()
    if output_path is not None:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_binary_patterns(
    model: nn.Module,
    layer_name: str | None = None,
    output_path: str | Path | None = None,
    max_filters: int = 64,
) -> None:
    """Visualize binary weight patterns as black/white images.

    Shows the learned binary patterns in convolutional filters.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    target_layer = None
    target_name = layer_name
    for name, param in model.named_parameters():
        if layer_name is None and "weight" in name and param.dim() == 4:
            target_layer = param.data.cpu()
            target_name = name
            break
        elif name == layer_name:
            target_layer = param.data.cpu()
            break

    if target_layer is None:
        return

    n = min(target_layer.shape[0], max_filters)
    k = target_layer.shape[2]

    fig, axes = plt.subplots(1, min(n, 16), figsize=(16, 2))
    if n == 1:
        axes = np.array([axes])

    for i in range(min(n, 16)):
        pattern = target_layer[i, 0].numpy()
        binary = (pattern >= 0).astype(np.uint8)
        axes[i].imshow(binary, cmap="gray", vmin=0, vmax=1)
        axes[i].axis("off")

    plt.suptitle(f"Binary Patterns: {target_name} (kernel {k}x{k})", fontsize=10)
    plt.tight_layout()
    if output_path is not None:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_accuracy_comparison(
    results: dict[str, float],
    output_path: str | Path | None = None,
) -> None:
    """Bar chart comparing accuracy across methods.

    Args:
        results: Dict mapping method name to accuracy percentage.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    names = list(results.keys())
    accs = list(results.values())
    colors = ["steelblue" if "fp32" in n.lower() else "coral" for n in names]

    fig, ax = plt.subplots(figsize=(max(6, len(names) * 1.5), 4))
    bars = ax.bar(names, accs, color=colors, edgecolor="black")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Accuracy Comparison")
    ax.set_ylim(max(0, min(accs) - 5), min(100, max(accs) + 5))

    for bar, acc in zip(bars, accs):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f"{acc:.1f}%",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.tight_layout()
    if output_path is not None:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_benchmark_results(
    comparison: dict[str, dict[str, float]],
    output_path: str | Path | None = None,
) -> None:
    """Plot benchmark comparison: latency, throughput, and model size.

    Args:
        comparison: Dict from BenchmarkSuite.compare_models().
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    names = list(comparison.keys())
    latencies = [comparison[n].get("latency", {}).get("avg_ms", 0) for n in names]
    throughputs = [comparison[n].get("throughput_fps", 0) for n in names]
    sizes = [comparison[n].get("memory", {}).get("total_mb", 0) for n in names]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].barh(names, latencies, color="steelblue", edgecolor="black")
    axes[0].set_xlabel("Latency (ms)")
    axes[0].set_title("Inference Latency")

    axes[1].barh(names, throughputs, color="coral", edgecolor="black")
    axes[1].set_xlabel("Throughput (FPS)")
    axes[1].set_title("Throughput")

    axes[2].barh(names, sizes, color="seagreen", edgecolor="black")
    axes[2].set_xlabel("Size (MB)")
    axes[2].set_title("Model Size")

    plt.suptitle("Benchmark Comparison", fontsize=12)
    plt.tight_layout()
    if output_path is not None:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_compression_comparison(
    fp32_params: int,
    int8_params: int,
    bin_params: int,
    output_path: str | Path | None = None,
) -> None:
    """Compare model sizes across precision formats.

    Args:
        fp32_params: Number of parameters in FP32 model.
        int8_params: Number of parameters in INT8 model.
        bin_params: Number of parameters in binary model.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    formats = ["FP32", "INT8", "Binary"]
    sizes_mb = [
        fp32_params * 4 / (1024 * 1024),
        int8_params * 1 / (1024 * 1024),
        bin_params / 8 / (1024 * 1024),
    ]
    colors = ["steelblue", "coral", "gold"]

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(formats, sizes_mb, color=colors, edgecolor="black")
    ax.set_ylabel("Model Size (MB)")
    ax.set_title("Model Size by Precision Format")

    for bar, size in zip(bars, sizes_mb):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{size:.2f} MB",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.tight_layout()
    if output_path is not None:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
