"""Local tangent-space estimation by per-point PCA on nearest neighbours.

For each representation we approximate the manifold tangent space
:math:`T_{x_i}(\\mathcal{M})` by the leading right singular vectors of the
matrix of its neighbours (centred at the neighbour mean). This is the local
tangent-space approximation used by MACE to constrain the erasure edit.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors


@dataclass
class LocalPCATangentEstimator:
    """Per-point local-PCA tangent estimator.

    Parameters
    ----------
    n_neighbors:
        Number of neighbours ``k`` used to estimate each local basis.
    intrinsic_dim:
        Rank ``r`` of the tangent basis kept per point (capped at ``k-1``).
    device:
        ``"cuda"`` or ``"cpu"``. On CUDA, neighbour search uses chunked
        ``torch.cdist`` and the SVDs run on GPU; on CPU we fall back to
        scikit-learn's ``NearestNeighbors``.
    """

    n_neighbors: int = 50
    intrinsic_dim: int = 32
    device: str = "cpu"

    # Side channel: singular values from the most recent ``estimate_bases``
    # call, shape (n, r), sorted descending. Consumers that need the local
    # spectrum (e.g. the spectral weighting in MACE) read it after the call.
    last_singular_values: np.ndarray | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        self._X_ref: np.ndarray | None = None
        self._X_ref_t: torch.Tensor | None = None
        self._nn: NearestNeighbors | None = None

    # ------------------------------------------------------------------ fit
    def fit(self, X_reference: np.ndarray) -> "LocalPCATangentEstimator":
        """Index the reference set that neighbours are drawn from."""
        X = np.asarray(X_reference, dtype=np.float32)
        self._X_ref = X
        self.n_neighbors = min(int(self.n_neighbors), len(X))
        if self._use_gpu:
            self._X_ref_t = torch.from_numpy(X).to(self.device)
            self._nn = None
        else:
            self._nn = NearestNeighbors(n_neighbors=self.n_neighbors, metric="euclidean").fit(X)
            self._X_ref_t = None
        return self

    @property
    def _use_gpu(self) -> bool:
        return str(self.device).startswith("cuda") and torch.cuda.is_available()

    # ----------------------------------------------------------- neighbours
    def kneighbors(self, X_query: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(distances, indices)`` of the ``k`` nearest reference points."""
        if self._X_ref is None:
            raise RuntimeError("LocalPCATangentEstimator must be fit() first")
        X_q = np.asarray(X_query, dtype=np.float32)
        k = int(self.n_neighbors)
        if self._use_gpu and self._X_ref_t is not None:
            dev = self._X_ref_t.device
            distances = np.empty((len(X_q), k), dtype=np.float32)
            indices = np.empty((len(X_q), k), dtype=np.int64)
            chunk = 1024
            for start in range(0, len(X_q), chunk):
                end = min(start + chunk, len(X_q))
                xq = torch.from_numpy(X_q[start:end]).to(dev)
                d = torch.cdist(xq, self._X_ref_t)
                top_d, top_i = d.topk(k, dim=1, largest=False)
                distances[start:end] = top_d.cpu().numpy()
                indices[start:end] = top_i.cpu().numpy()
            return distances, indices
        distances, indices = self._nn.kneighbors(X_q, n_neighbors=k)
        return distances.astype(np.float32), indices.astype(np.int64)

    def effective_rank(self) -> int:
        return max(1, min(int(self.intrinsic_dim), int(self.n_neighbors) - 1))

    # --------------------------------------------------------------- bases
    def estimate_bases(
        self,
        X_query: np.ndarray,
        precomputed_neighbors: np.ndarray | None = None,
    ) -> np.ndarray:
        """Return per-point tangent bases of shape ``(n, d, r)``.

        Column span of ``bases[i]`` estimates ``T_{x_i}(M)``. Singular values
        are cached on :attr:`last_singular_values`. The neighbour cloud is
        centred at its own mean before the SVD (standard local PCA).
        """
        if self._X_ref is None:
            raise RuntimeError("LocalPCATangentEstimator must be fit() first")
        X_q = np.asarray(X_query, dtype=np.float32)
        neighbors = (
            precomputed_neighbors
            if precomputed_neighbors is not None
            else self.kneighbors(X_q)[1]
        )
        n, d = X_q.shape
        r = min(self.effective_rank(), d)

        dev = torch.device(self.device if self._use_gpu else "cpu")
        bases = np.empty((n, d, r), dtype=np.float32)
        singular_values = np.empty((n, r), dtype=np.float32)
        chunk = 512 if self._use_gpu else 128
        for start in range(0, n, chunk):
            end = min(start + chunk, n)
            pts = self._X_ref[neighbors[start:end]]          # (cs, k, d)
            pts = pts - pts.mean(axis=1, keepdims=True)       # centre on neighbour mean
            pts_t = torch.from_numpy(pts).to(dev)
            # Exact thin SVD (the neighbour matrices are tiny: k x d). Vh is
            # (cs, k, d); the top-r right singular vectors span the tangent.
            # Deterministic, unlike randomised svd_lowrank.
            _, S, Vh = torch.linalg.svd(pts_t, full_matrices=False)
            bases[start:end] = Vh[:, :r, :].transpose(-2, -1).cpu().numpy()  # (cs, d, r)
            singular_values[start:end] = S[:, :r].cpu().numpy()
        self.last_singular_values = singular_values
        return bases


def project_to_bases(vectors: np.ndarray, bases: np.ndarray) -> np.ndarray:
    """Project rows of ``vectors`` (n, d) onto per-row ``bases`` (n, d, r)."""
    coords = np.einsum("nd,ndr->nr", vectors, bases)
    return np.einsum("nr,ndr->nd", coords, bases)
