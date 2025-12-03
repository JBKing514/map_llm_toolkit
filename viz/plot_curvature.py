from typing import Optional

import matplotlib.pyplot as plt
import numpy as np


def plot_curvature_profiles(
    curv_rigid: np.ndarray,
    curv_adaptive: np.ndarray,
    ax: Optional[plt.Axes] = None,
):
    """
    Plot curvature vs token step for rigid/adaptive modes.
    """
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(6, 4))

    ax.plot(curv_rigid, "r--", label="Rigid Curvature")
    ax.plot(curv_adaptive, "g--", label="Adaptive Curvature")
    ax.set_title("Geometric Curvature Profile")
    ax.set_xlabel("Token Step")
    ax.set_ylabel("Turning Angle")
    ax.legend()
    ax.grid(True)

    return ax.figure
