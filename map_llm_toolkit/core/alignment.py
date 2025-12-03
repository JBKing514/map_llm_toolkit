"""
Alignment utilities for MAP: layer-wise semantic alignment A(ℓ)
and alignment delta ΔA between tight vs. sparse semantic prompts.
"""

from typing import Sequence, Dict, Any, List, Tuple
import numpy as np


def compute_alignment_profile(trajectories: Sequence[np.ndarray]) -> np.ndarray:
    """
    Compute layer-wise alignment profile A_ell for a set of trajectories.

    Each trajectory is expected to be an array of shape (num_layers, hidden_dim).
    Given N prompts, we stack them into X[ell] ∈ R^{N × d} per layer, then:

        1. L2-normalize each prompt vector.
        2. Compute cosine similarity matrix S = X_norm @ X_norm^T.
        3. Take the upper-triangle pairwise similarities (i < j).
        4. Map from [-1, 1] → [0, 1] via (s + 1) / 2.
        5. Average over all pairs to obtain A_ell.

    Parameters
    ----------
    trajectories:
        Sequence of arrays, one per prompt. Each array has shape (num_layers, dim).

    Returns
    -------
    A : np.ndarray
        1-D array of shape (num_layers,) giving alignment scores in [0, 1].
    """
    if len(trajectories) == 0:
        return np.zeros(0, dtype=np.float32)

    traj_arr = np.stack(trajectories, axis=0)  # (num_prompts, num_layers, dim)
    num_prompts, num_layers, _ = traj_arr.shape

    A_profile: List[float] = []

    for ell in range(num_layers):
        # X: (num_prompts, dim)
        X = traj_arr[:, ell, :]

        # Normalize each prompt vector
        norms = np.linalg.norm(X, axis=1, keepdims=True) + 1e-8
        X_norm = X / norms

        # Cosine similarity between all prompt pairs
        sim = X_norm @ X_norm.T  # (num_prompts, num_prompts)

        # Use only upper-triangle (i < j) pairs
        iu = np.triu_indices(num_prompts, k=1)
        sims = sim[iu]

        if sims.size == 0:
            A_profile.append(0.0)
            continue

        # Map from [-1, 1] to [0, 1] and average
        A_layer = float(np.mean((sims + 1.0) / 2.0))
        A_profile.append(A_layer)

    return np.asarray(A_profile, dtype=np.float32)


def compute_alignment_delta(
    tight_trajectories: Sequence[np.ndarray],
    sparse_trajectories: Sequence[np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Convenience wrapper to compute layer indices, A_tight, A_sparse and ΔA.

    Parameters
    ----------
    tight_trajectories:
        Trajectories for a tight semantic cluster (e.g., paraphrases of one concept).
    sparse_trajectories:
        Trajectories for a semantically sparse cluster (random topics).

    Returns
    -------
    L : np.ndarray
        Layer indices 0..L-1.
    A_tight : np.ndarray
        Alignment profile for tight prompts.
    A_sparse : np.ndarray
        Alignment profile for sparse prompts.
    DeltaA : np.ndarray
        Element-wise difference A_tight - A_sparse.
    """
    A_tight = compute_alignment_profile(tight_trajectories)
    A_sparse = compute_alignment_profile(sparse_trajectories)

    num_layers = min(len(A_tight), len(A_sparse))
    if num_layers == 0:
        return (
            np.zeros(0, dtype=np.int32),
            np.zeros(0, dtype=np.float32),
            np.zeros(0, dtype=np.float32),
            np.zeros(0, dtype=np.float32),
        )

    A_tight = A_tight[:num_layers]
    A_sparse = A_sparse[:num_layers]
    DeltaA = A_tight - A_sparse

    L = np.arange(num_layers, dtype=np.int32)
    return L, A_tight, A_sparse, DeltaA
