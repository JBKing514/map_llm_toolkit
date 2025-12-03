import numpy as np


def compute_curvature(points: np.ndarray) -> np.ndarray:
    """
    Discrete curvature/turning-angle along a 2D trajectory.

    Parameters
    ----------
    points : (T, 2) array

    Returns
    -------
    curvatures : (T-2,) array of turning angles in radians
    """
    if len(points) < 3:
        return np.zeros(0, dtype="float32")

    curvatures = []
    for i in range(1, len(points) - 1):
        v1 = points[i] - points[i - 1]
        v2 = points[i + 1] - points[i]

        norm_prod = np.linalg.norm(v1) * np.linalg.norm(v2)
        if norm_prod == 0:
            curvatures.append(0.0)
            continue

        cos_angle = np.dot(v1, v2) / norm_prod
        angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
        curvatures.append(float(angle))

    return np.asarray(curvatures, dtype="float32")
