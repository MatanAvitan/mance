"""Intrinsic-dimension estimation.

MANCE keeps a local tangent basis of rank ``r``. When ``r`` is not supplied,
we estimate the global intrinsic dimension once with the TwoNN estimator
(Facco et al., 2017) and keep that rank fixed throughout the editing loop.
"""
from __future__ import annotations

import numpy as np
from sklearn.neighbors import NearestNeighbors


def estimate_intrinsic_dim_two_nn(X: np.ndarray) -> float:
    """TwoNN intrinsic-dimension estimate (Facco et al., 2017).

    For each point, ``mu = r2 / r1`` is the ratio of the distances to its
    second- and first-nearest neighbours; the maximum-likelihood dimension is
    ``d = N / sum(log mu)``.
    """
    X = np.asarray(X, dtype=np.float64)
    nn = NearestNeighbors(n_neighbors=3, metric="euclidean").fit(X)
    distances, _ = nn.kneighbors(X)
    r1 = distances[:, 1]
    r2 = distances[:, 2]
    mu = (r2 + 1e-12) / (r1 + 1e-12)
    return float(len(mu) / np.maximum(np.log(mu + 1e-12).sum(), 1e-12))
