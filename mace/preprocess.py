"""Optional closed-form preprocessing for the MACE family.

These are one-shot affine projections (effective rank :math:`\\le 3`) applied
to all splits *before* the iterative MACE loop. They remove dataset-level
moment structure that already has simple closed-form erasers, so the iterative
loop does not have to spend its trust-region budget on it.

* ``MACE+``  prepends :func:`fit_leace_eraser` — LEACE (Belrose et al., 2023),
  which removes the first-moment linear signal associated with class means.
* ``MACE++`` additionally prepends :func:`covariance_matching_directions` —
  *CovMatch*, a rank-2 specialisation of k-LEACE that removes the leading
  second-moment class-conditional covariance asymmetry
  :math:`\\Delta\\Sigma = \\Sigma_+ - \\Sigma_-`.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.covariance import LedoitWolf


# ----------------------------------------------------------------------- LEACE
def _center_labels(y: np.ndarray) -> np.ndarray:
    labels = np.asarray(y, dtype=np.int64).reshape(-1)
    classes, inverse = np.unique(labels, return_inverse=True)
    if classes.size <= 1:
        return np.zeros((labels.shape[0], 0), dtype=np.float64)
    if classes.size == 2:
        encoded = inverse.astype(np.float64).reshape(-1, 1)
    else:
        encoded = np.eye(classes.size, dtype=np.float64)[inverse]
    return encoded - encoded.mean(axis=0, keepdims=True)


@dataclass
class LEACEEraser:
    """Affine LEACE eraser: ``x ↦ x - (x - mean) Pᵀ`` of rank ``<= |classes|-1``."""

    proj_left: np.ndarray
    proj_right: np.ndarray
    mean_x: np.ndarray | None

    @property
    def rank(self) -> int:
        return int(self.proj_left.shape[1])

    def erase(self, X: np.ndarray) -> np.ndarray:
        X_arr = np.asarray(X, dtype=np.float64)
        if self.rank == 0:
            return X_arr.astype(np.float32)
        delta = X_arr if self.mean_x is None else X_arr - self.mean_x
        removed = (delta @ self.proj_right.T) @ self.proj_left.T
        return (X_arr - removed).astype(np.float32)


def fit_leace_eraser(
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    shrinkage: bool = True,
    svd_tol: float = 1e-2,
) -> LEACEEraser:
    """Fit LEACE (Least-squares Concept Erasure, Belrose et al., 2023).

    Removes the subspace of representation space that is linearly predictive of
    the class label, in the whitened metric, while moving each point the least.
    """
    X = np.asarray(X_train, dtype=np.float64)
    mean_x = X.mean(axis=0, keepdims=True)
    Xc = X - mean_x
    Z = _center_labels(y_train)
    d = int(X.shape[1])
    if Z.size == 0:
        return LEACEEraser(np.zeros((d, 0)), np.zeros((0, d)), mean_x)

    if shrinkage:
        cov = LedoitWolf(assume_centered=True).fit(Xc).covariance_
    else:
        cov = (Xc.T @ Xc) / max(len(Xc), 1)
    cov = 0.5 * (cov + cov.T)
    cov_xz = (Xc.T @ Z) / max(len(Xc), 1)

    eigvals, eigvecs = np.linalg.eigh(cov)
    eigvals = np.maximum(eigvals, 0.0)
    thr = float(eigvals[-1]) * d * np.finfo(eigvals.dtype).eps if eigvals.size else 0.0
    mask = eigvals > thr
    inv_sqrt = np.zeros_like(eigvals)
    np.power(eigvals, -0.5, where=mask, out=inv_sqrt)
    sqrt = np.where(mask, np.sqrt(eigvals), 0.0)
    W = (eigvecs * inv_sqrt) @ eigvecs.T
    W_inv = (eigvecs * sqrt) @ eigvecs.T

    u, s, _ = np.linalg.svd(W @ cov_xz, full_matrices=False)
    keep = s > svd_tol
    u = u[:, keep]
    if u.size == 0:
        return LEACEEraser(np.zeros((d, 0)), np.zeros((0, d)), mean_x)
    return LEACEEraser(proj_left=W_inv @ u, proj_right=u.T @ W, mean_x=mean_x)


# -------------------------------------------------------------------- CovMatch
def covariance_matching_directions(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    n_cov_dirs: int = 2,
    positive_label: int = 1,
) -> np.ndarray:
    """Orthonormal directions ``D ∈ R^{d x (1+k)}`` for *CovMatch* erasure.

    * Column 0: the mean direction ``normalize(μ₁ − μ₀)``.
    * Columns 1..k: top-``k`` eigenvectors (by ``|eigenvalue|``) of
      ``ΔΣ = Σ₁ − Σ₀``, the within-class covariance difference computed after
      projecting out the mean direction.

    Projecting these out drives a Gaussian linear classifier toward chance.
    """
    labels = np.asarray(labels, dtype=np.int64)
    mask1 = labels == int(positive_label)
    mask0 = ~mask1

    mu0 = features[mask0].mean(axis=0)
    mu1 = features[mask1].mean(axis=0)
    d_mean = mu1 - mu0
    d_mean = d_mean / max(np.linalg.norm(d_mean), 1e-12)

    proj = features @ d_mean
    features_proj = features - np.outer(proj, d_mean)
    Sigma0 = np.cov(features_proj[mask0].T, ddof=1)
    Sigma1 = np.cov(features_proj[mask1].T, ddof=1)
    delta = (Sigma1 - Sigma0).astype(np.float64)

    eigvals, eigvecs = np.linalg.eigh(delta)
    idx = np.argsort(np.abs(eigvals))[::-1][:n_cov_dirs]
    cov_dirs = eigvecs[:, idx].astype(np.float64)

    all_dirs = np.column_stack([d_mean.reshape(-1, 1), cov_dirs])
    Q, _ = np.linalg.qr(all_dirs)
    return Q[:, : all_dirs.shape[1]].astype(np.float32)


def apply_projection(X: np.ndarray, D: np.ndarray) -> np.ndarray:
    """Remove the span of orthonormal columns ``D`` from rows of ``X``."""
    X = np.asarray(X, dtype=np.float32)
    return (X - (X @ D) @ D.T).astype(np.float32)
