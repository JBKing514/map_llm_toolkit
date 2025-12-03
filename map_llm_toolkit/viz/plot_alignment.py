"""
Visualization helpers for alignment profiles A(ℓ) and ΔA(ℓ).
"""

from typing import List, Dict, Any, Sequence, Tuple
import numpy as np
import matplotlib.pyplot as plt


def plot_alignment_profiles(
    results: Sequence[Dict[str, Any]],
    title: str = (
        "Layer-wise Alignment A for Tight vs Sparse Semantics\n"
        "(and ΔA = A_tight - A_sparse)"
    ),
    save_path: str = "alignment_delta_profile.png",
) -> None:
    """
    Plot layer-wise alignment profiles for one or more models.

    Each entry in `results` should be a dict with keys:
        - "name": str, model display name
        - "L": np.ndarray of shape (num_layers,)
        - "A_tight": np.ndarray of shape (num_layers,)
        - "A_sparse": np.ndarray of shape (num_layers,)
        - "DeltaA": np.ndarray of shape (num_layers,)

    This function will create N subplots (N = len(results)),
    sharing y-axis, with A_tight / A_sparse in solid / dashed lines
    and ΔA as a thin gray dash-dot line.
    """
    if len(results) == 0:
        raise ValueError("plot_alignment_profiles: `results` is empty.")

    num_models = len(results)
    fig, axes = plt.subplots(
        1, num_models, figsize=(6 * num_models, 4), sharey=True
    )

    if num_models == 1:
        axes = [axes]

    for ax, res in zip(axes, results):
        L = np.asarray(res["L"])
        A_tight = np.asarray(res["A_tight"])
        A_sparse = np.asarray(res["A_sparse"])
        DeltaA = np.asarray(res["DeltaA"])

        name = str(res.get("name", "Model"))

        ax.plot(
            L,
            A_tight,
            marker="o",
            linestyle="-",
            linewidth=1.5,
            label="Tight semantics A",
        )
        ax.plot(
            L,
            A_sparse,
            marker="s",
            linestyle="--",
            linewidth=1.5,
            label="Sparse semantics A",
        )
        ax.plot(
            L,
            DeltaA,
            linestyle=":",
            linewidth=1.2,
            color="gray",
            label="ΔA = A_tight - A_sparse",
        )

        ax.set_title(name)
        ax.set_xlabel("Layer Index")
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("Alignment A / ΔA")
    axes[0].legend(loc="best", fontsize=9)
    fig.suptitle(title, fontsize=12)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    fig.savefig(save_path, dpi=150)
    print(f"[MAP] Alignment profile figure saved to: {save_path}")
    plt.close(fig)
