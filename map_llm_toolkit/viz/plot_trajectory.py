from typing import List, Sequence, Tuple, Optional

import matplotlib.pyplot as plt
import numpy as np


def plot_convergence_trajectories(
    trajectories_per_model: Sequence[Tuple[str, List[np.ndarray]]],
    figsize: Tuple[int, int] = (16, 7),
    dpi: int = 300,
):
    """
    Parameters
    ----------
    trajectories_per_model :
        List of (model_name, list_of_2D_trajectories).
        每个 2D 轨迹是 (T, 2) 的 numpy 数组。
    """
    num_models = len(trajectories_per_model)
    fig, axes = plt.subplots(1, num_models, figsize=figsize, dpi=dpi)

    if num_models == 1:
        axes = [axes]

    for ax, (model_name, traj_list) in zip(axes, trajectories_per_model):
        colors = plt.cm.Blues(np.linspace(0.4, 1.0, len(traj_list)))

        for i, path in enumerate(traj_list):
            ax.plot(path[:, 0], path[:, 1], marker=".", linewidth=1, color=colors[i], alpha=0.7)
            ax.scatter(path[0, 0], path[0, 1], marker="x", s=40, color=colors[i], label="Input" if i == 0 else "")
            ax.scatter(path[-1, 0], path[-1, 1], marker="o", s=30, color="black", label="Attractor" if i == 0 else "")

        ax.set_title(f"Model: {model_name}")
        ax.set_xlabel("PC 1")
        ax.set_ylabel("PC 2")
        ax.grid(True)
        ax.legend(loc="best", fontsize=8)

    fig.suptitle("Geometric Convergence of Reasoning Trajectories (MAP)", y=0.98)
    fig.tight_layout()
    return fig

def plot_safety_trajectories(
    traj_rigid_2d: np.ndarray,
    traj_adaptive_2d: np.ndarray,
    ax: Optional[plt.Axes] = None,
):
    """
    Plot 2D trajectories for rigid vs adaptive modes on the same axes.
    """
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(6, 6))

    ax.plot(traj_rigid_2d[:, 0], traj_rigid_2d[:, 1], "r-o", label="Rigid Mode")
    ax.plot(traj_adaptive_2d[:, 0], traj_adaptive_2d[:, 1], "g-o", label="Adaptive Mode")
    ax.set_title("Visualized Safety Trajectories")
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.legend()
    ax.grid(True)

    return ax.figure
