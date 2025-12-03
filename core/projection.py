from typing import List, Tuple

import numpy as np
from sklearn.decomposition import PCA


def project_pca(
    trajectories: List[np.ndarray],
    n_components: int = 2,
) -> List[np.ndarray]:
    """
    Linear projection of high-dimensional trajectories into a shared low-D space.

    Parameters
    ----------
    trajectories : list of (T_i, D) arrays
    n_components : int

    Returns
    -------
    projected : list of (T_i, n_components) arrays
    """
    if len(trajectories) == 0:
        return []

    all_points = np.vstack(trajectories)
    pca = PCA(n_components=n_components)
    all_points_2d = pca.fit_transform(all_points)

    projected: List[np.ndarray] = []
    start = 0
    for t in trajectories:
        end = start + len(t)
        projected.append(all_points_2d[start:end])
        start = end

    return projected
