"""
Public API for map_llm_toolkit.
"""

from .core.runner import MAPModelRunner
from .core.projection import project_pca
from .core.curvature import compute_curvature
from .core.protocols import SafetyProtocol
from .core.alignment import (
    compute_alignment_profile,
    compute_alignment_delta,
)

from .viz.plot_trajectory import (
    plot_convergence_trajectories,
    plot_safety_trajectories,
)
from .viz.plot_curvature import plot_curvature_profiles
from .viz.plot_alignment import plot_alignment_profiles

__all__ = [
    # Core
    "MAPModelRunner",
    "project_pca",
    "compute_curvature",
    "SafetyProtocol",
    # Alignment
    "compute_alignment_profile",
    "compute_alignment_delta",
    # Viz
    "plot_convergence_trajectories",
    "plot_safety_trajectories",
    "plot_curvature_profiles",
    "plot_alignment_profiles",
]
