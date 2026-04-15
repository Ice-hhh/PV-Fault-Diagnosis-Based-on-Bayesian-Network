from __future__ import annotations

import os
from pathlib import Path

Path("outputs/.matplotlib").mkdir(parents=True, exist_ok=True)
Path("outputs/.cache").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(Path("outputs/.matplotlib")))
os.environ.setdefault("XDG_CACHE_HOME", str(Path("outputs/.cache")))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def prediction_band_from_residuals(
    y_valid: np.ndarray,
    mean_pred: np.ndarray,
) -> np.ndarray:
    residual_std = np.std(y_valid - mean_pred, axis=0, keepdims=True)
    return np.repeat(residual_std, len(y_valid), axis=0)


def plot_training_process(
    y_valid: np.ndarray,
    mean_pred: np.ndarray,
    std_pred: np.ndarray,
    loss_history: np.ndarray,
    plot_path: Path,
    title_prefix: str,
) -> None:
    y_true = y_valid.reshape(-1)
    y_mean = mean_pred.reshape(-1)
    y_std = std_pred.reshape(-1)

    max_points = min(500, len(y_true))
    x = np.linspace(0, 1, max_points)
    y_true = y_true[:max_points]
    y_mean = y_mean[:max_points]
    y_std = y_std[:max_points]

    fig, axes = plt.subplots(2, 1, figsize=(8, 9), constrained_layout=True)

    axes[0].plot(x, y_mean, color="#1f77b4", linewidth=1.2, label="Mean Posterior Predictive")
    axes[0].fill_between(
        x,
        y_mean - 1.96 * y_std,
        y_mean + 1.96 * y_std,
        color="#1f77b4",
        alpha=0.18,
        label="Epistemic uncertainty",
    )
    axes[0].scatter(x, y_true, color="#ff7f0e", s=18, alpha=0.85, label="Actual Data Points")
    axes[0].set_title(f"{title_prefix} Posterior Predictive")
    axes[0].legend(loc="upper left")
    axes[0].grid(alpha=0.15)

    loss_x = np.arange(len(loss_history))
    axes[1].plot(loss_x, loss_history, color="#1f77b4", linewidth=1.4, label="Loss Function")
    axes[1].set_title(f"{title_prefix} Loss")
    axes[1].set_xlabel("Iteration")
    axes[1].legend(loc="upper right")
    axes[1].grid(alpha=0.15)

    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path, dpi=180)
    plt.close(fig)

