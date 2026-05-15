"""
Visualization utilities for DynaSys-EEG results and intermediate representations.

Includes:
  - Phase space trajectories (2D/3D projections)
  - Descriptor distributions per class
  - Energy landscape visualization
  - Confusion matrices
  - LOSO results
  - Ablation bar charts
  - Training loss curves
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import seaborn as sns

logger = logging.getLogger(__name__)

# Color palette for AD, FTD/aMCI, HC
CLASS_COLORS = {
    "HC":  "#2ecc71",   # Green
    "AD":  "#e74c3c",   # Red
    "FTD": "#3498db",   # Blue
    "aMCI": "#f39c12",  # Orange
}
CLASS_MARKERS = {"HC": "o", "AD": "s", "FTD": "^", "aMCI": "D"}


def setup_style():
    """Apply consistent plot style."""
    plt.style.use("seaborn-v0_8-darkgrid")
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "legend.fontsize": 10,
        "figure.dpi": 100,
    })


def plot_phase_space(
    states: np.ndarray,
    label: str = "HC",
    title: str = "Phase Space Trajectory",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot 2D projection of the phase space trajectory.
    X_t = [x_t, x_{t-τ}, x_{t-2τ}] → plot (dim0, dim1)
    """
    setup_style()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    color = CLASS_COLORS.get(label, "#95a5a6")

    # 2D phase portrait (dim 0 vs dim 1)
    ax = axes[0]
    n = min(1000, len(states))
    ax.plot(states[:n, 0], states[:n, 1],
            color=color, alpha=0.5, linewidth=0.6)
    ax.scatter(states[:n, 0], states[:n, 1],
               c=np.linspace(0, 1, n), cmap="plasma",
               s=5, alpha=0.7, zorder=3)
    ax.set_xlabel("x(t)")
    ax.set_ylabel("x(t - τ)")
    ax.set_title(f"Phase Portrait — {label}")

    # Trajectory (first 500 samples)
    ax2 = axes[1]
    t = np.arange(n)
    ax2.plot(t, states[:n, 0], color=color, linewidth=1.0, alpha=0.8)
    ax2.set_xlabel("Time (samples)")
    ax2.set_ylabel("x(t)")
    ax2.set_title(f"EEG Trajectory — {label}")

    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
        logger.info(f"Saved phase space plot to {save_path}")

    return fig


def plot_descriptors_distribution(
    Z: np.ndarray,
    y: np.ndarray,
    label_map: Dict[int, str],
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Visualize the distribution of each descriptor (λ, H, D, E, T) per class.
    Shows violin plots + strip plots.
    """
    setup_style()
    desc_names = ["Lyapunov (λ)", "Entropy (H)", "Diffusion (D)",
                  "Energy (E)", "Transition (T)"]

    n_desc = Z.shape[1]
    fig, axes = plt.subplots(1, n_desc, figsize=(4 * n_desc, 5))
    if n_desc == 1:
        axes = [axes]

    for d_idx, (ax, dname) in enumerate(zip(axes, desc_names)):
        data_per_class = {}
        for class_int, class_name in label_map.items():
            mask = (y == class_int)
            vals = Z[mask, d_idx]
            vals = vals[np.isfinite(vals)]
            data_per_class[class_name] = vals

        class_names = list(data_per_class.keys())
        colors = [CLASS_COLORS.get(cn, "#95a5a6") for cn in class_names]

        for i, (cn, vals) in enumerate(data_per_class.items()):
            if len(vals) == 0:
                continue
            parts = ax.violinplot(vals, positions=[i], widths=0.6, showmedians=True)
            for pc in parts.get("bodies", []):
                pc.set_facecolor(CLASS_COLORS.get(cn, "#95a5a6"))
                pc.set_alpha(0.7)

        ax.set_xticks(range(len(class_names)))
        ax.set_xticklabels(class_names, fontsize=9)
        ax.set_title(dname, fontsize=11, fontweight="bold")
        ax.set_ylabel("Value")

    fig.suptitle("Dynamical Descriptor Distributions per Class", fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
    return fig


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    title: str = "Confusion Matrix",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot a clean confusion matrix."""
    setup_style()
    fig, ax = plt.subplots(figsize=(6, 5))

    # Normalize
    cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-8)

    sns.heatmap(
        cm_norm,
        annot=cm,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
        linewidths=0.5,
        linecolor="gray",
        vmin=0, vmax=1,
    )
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
    return fig


def plot_ablation_results(
    ablation_results: Dict,
    metric: str = "accuracy",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Bar chart comparing full model vs. ablated versions."""
    setup_style()
    fig, ax = plt.subplots(figsize=(10, 5))

    names = []
    values = []
    colors = []
    full_val = None

    for name, m in ablation_results.items():
        if "error" in m:
            continue
        val = m.get(metric, 0) * 100
        names.append(name.replace("Without ", "w/o "))
        values.append(val)

        if name == "Full (all descriptors)":
            colors.append("#2ecc71")
            full_val = val
        else:
            colors.append("#e74c3c" if val < full_val else "#3498db")

    bars = ax.barh(names, values, color=colors, alpha=0.85, edgecolor="white")

    # Add value labels
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
            f"{val:.1f}%", va="center", fontsize=10
        )

    ax.set_xlabel(f"{metric.capitalize()} (%)", fontsize=12)
    ax.set_title("Ablation Study: Component Contribution", fontsize=14, fontweight="bold")
    ax.set_xlim(0, 105)

    # Legend
    patches = [
        mpatches.Patch(color="#2ecc71", label="Full Model"),
        mpatches.Patch(color="#e74c3c", label="Performance Drop"),
        mpatches.Patch(color="#3498db", label="Performance Maintained"),
    ]
    ax.legend(handles=patches, loc="lower right", fontsize=9)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
    return fig


def plot_training_curves(
    train_losses: List[float],
    val_losses: List[float],
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot dynamics model training loss curves."""
    setup_style()
    fig, ax = plt.subplots(figsize=(8, 4))
    epochs = np.arange(1, len(train_losses) + 1)

    ax.plot(epochs, train_losses, label="Train Loss", color="#3498db", linewidth=2)
    ax.plot(epochs, val_losses, label="Val Loss", color="#e74c3c",
            linewidth=2, linestyle="--")
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("MSE Loss", fontsize=12)
    ax.set_title("Dynamics Network Training Curves", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.set_yscale("log")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
    return fig


def plot_loso_results_per_subject(
    loso_results: Dict,
    label_map: Dict[int, str],
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot per-subject LOSO prediction correctness."""
    setup_style()
    per_subject = loso_results.get("per_subject", [])
    if not per_subject:
        return plt.figure()

    fig, ax = plt.subplots(figsize=(max(10, len(per_subject) * 0.4), 4))

    subjects = [s["subject_id"] for s in per_subject]
    correct = [1 if s["true"] == s["pred"] else 0 for s in per_subject]
    true_classes = [s["true_name"] for s in per_subject]
    colors = [CLASS_COLORS.get(c, "#95a5a6") for c in true_classes]

    bars = ax.bar(subjects, correct, color=colors, alpha=0.85, edgecolor="gray")

    ax.set_ylim(-0.1, 1.3)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Wrong", "Correct"])
    ax.set_xlabel("Subject ID", fontsize=12)
    ax.set_title("LOSO Per-Subject Classification Correctness", fontsize=13, fontweight="bold")
    ax.tick_params(axis="x", rotation=60, labelsize=7)

    # Legend for true classes
    unique_classes = list(set(true_classes))
    legend_patches = [
        mpatches.Patch(color=CLASS_COLORS.get(c, "#95a5a6"), label=c)
        for c in unique_classes
    ]
    ax.legend(handles=legend_patches, loc="upper right", title="True Class")

    # Accuracy annotation
    acc = np.mean(correct)
    ax.text(0.02, 1.2, f"Overall Accuracy: {acc*100:.1f}%",
            transform=ax.transAxes, fontsize=11, fontweight="bold",
            color="#2c3e50")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
    return fig


def plot_energy_landscape(
    states: np.ndarray,
    label: str = "Unknown",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot 2D energy landscape E(x) = -log P(x) from KDE.
    Shows how states are organized in phase space.
    """
    setup_style()
    from scipy.stats import gaussian_kde

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    color = CLASS_COLORS.get(label, "#95a5a6")

    x1 = states[:, 0]
    x2 = states[:, 1] if states.shape[1] > 1 else states[:, 0] * 0.9

    # Left: density scatter
    ax = axes[0]
    ax.scatter(x1, x2, c=color, s=2, alpha=0.3)
    ax.set_xlabel("Dimension 1")
    ax.set_ylabel("Dimension 2")
    ax.set_title(f"State Space — {label}")

    # Right: energy landscape as heatmap
    ax2 = axes[1]
    try:
        grid_size = 50
        x1_grid = np.linspace(x1.min(), x1.max(), grid_size)
        x2_grid = np.linspace(x2.min(), x2.max(), grid_size)
        X1, X2 = np.meshgrid(x1_grid, x2_grid)

        kde = gaussian_kde(np.vstack([x1, x2]), bw_method=0.3)
        Z = kde(np.vstack([X1.ravel(), X2.ravel()])).reshape(grid_size, grid_size)
        Z = np.clip(Z, 1e-12, None)
        E = -np.log(Z)

        im = ax2.contourf(X1, X2, E, levels=20, cmap="RdYlGn_r")
        ax2.contour(X1, X2, E, levels=10, colors="white", alpha=0.3, linewidths=0.5)
        plt.colorbar(im, ax=ax2, label="Energy E(x) = -log P(x)")
        ax2.set_xlabel("Dimension 1")
        ax2.set_ylabel("Dimension 2")
        ax2.set_title(f"Energy Landscape — {label}")
    except Exception:
        ax2.text(0.5, 0.5, "KDE failed", ha="center", va="center",
                 transform=ax2.transAxes)

    fig.suptitle(f"DynaSys-EEG: Energy Landscape ({label})", fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
    return fig


def plot_comparison_results(
    results_dict: Dict[str, Dict[str, float]],
    title: str = "Method Comparison",
    metric: str = "accuracy",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Bar chart comparing multiple classification methods."""
    setup_style()
    fig, ax = plt.subplots(figsize=(10, 5))

    methods = list(results_dict.keys())
    values = [results_dict[m].get(metric, 0) * 100 for m in methods]

    # Color DynaSys methods differently
    colors = []
    for m in methods:
        if "DynaSys" in m or "Nonlinear" in m or "Prototype" in m or "Energy" in m:
            colors.append("#3498db")
        else:
            colors.append("#95a5a6")

    bars = ax.bar(methods, values, color=colors, alpha=0.85, edgecolor="white",
                  width=0.6)

    # Annotate
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
            f"{val:.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold"
        )

    ax.set_ylabel(f"{metric.capitalize()} (%)", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_ylim(0, 110)
    ax.tick_params(axis="x", rotation=30, labelsize=9)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
    return fig
