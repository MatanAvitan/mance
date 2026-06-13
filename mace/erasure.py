"""The MACE editing loop and its variants.

This module implements Algorithm 1 of the paper. Each round it

1. fits / reuses a nonlinear concept probe on the current representations,
2. estimates a local tangent basis at each (edited) representation from its
   nearest neighbours among the natural, unedited representations X^(0) —
   the edits move the query point, the manifold reference stays natural,
3. builds the spectrally-weighted tangent erasure direction from the probe's
   input gradient, and
4. applies the largest per-row step allowed by a local-radius trust region.

The :class:`MACE` wrapper selects the variant via ``variant``:

============  =====================================================
``mace``    iterative loop only
``mace+``   LEACE preprocessing, then the loop
``mace++``  LEACE + CovMatch preprocessing, then the loop (default)
============  =====================================================
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import torch

from mace.intrinsic_dim import estimate_intrinsic_dim_two_nn
from mace.preprocess import (
    apply_projection,
    covariance_matching_directions,
    fit_leace_eraser,
)
from mace.scorer import ConceptScorer, compute_unit_gradients, train_concept_scorer
from mace.tangent import LocalPCATangentEstimator, project_to_bases

_VARIANTS = ("mace", "mace+", "mace++")


# --------------------------------------------------------------- single step
def mace_edit(
    X: np.ndarray,
    scorer: ConceptScorer,
    tangent: LocalPCATangentEstimator,
    *,
    epsilon: float,
    lambda_max: float,
    alpha: float,
    device: torch.device | None = None,
    batch_size: int = 512,
) -> np.ndarray:
    """One MACE edit of ``X`` against a fitted ``scorer`` and ``tangent``.

    ``tangent`` must already be ``fit`` on the reference representations: the
    natural, pre-edit training split X^(0), fit once and reused across rounds.
    ``X`` is the current (edited) split being queried; its rows move each round
    while the reference cloud stays X^(0). Train rows are detected as
    self-matches in the neighbour query and excluded from their own radius.

    Implements, per row,
    :math:`\\tilde x_i = x_i - \\lambda_i \\langle x_i, \\hat u_i\\rangle\\,\\hat u_i`
    with :math:`\\hat u_i = d_i / \\lVert d_i\\rVert`,
    :math:`d_i = B_i\\,\\mathrm{diag}(\\sigma_i^{\\alpha})\\,B_i^{\\top} u_i`, and
    :math:`\\lambda_i = \\min(\\lambda_{\\max},\\ \\varepsilon r_i / \\lVert v_i\\rVert)`.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X = np.asarray(X, dtype=np.float32)

    # k+1 neighbours so a train row's own zero-distance self-match can be
    # dropped before computing the local radius r_i.
    k = int(tangent.n_neighbors)
    tangent.n_neighbors = min(k + 1, len(tangent._X_ref))
    try:
        dist_kp1, nbr_kp1 = tangent.kneighbors(X)
    finally:
        tangent.n_neighbors = k
    if dist_kp1.shape[1] >= k + 1:
        is_self = dist_kp1[:, 0] < 1e-6
        distances = np.where(is_self[:, None], dist_kp1[:, 1 : k + 1], dist_kp1[:, :k])
        neighbors = np.where(is_self[:, None], nbr_kp1[:, 1 : k + 1], nbr_kp1[:, :k])
    else:
        distances, neighbors = dist_kp1[:, :k], nbr_kp1[:, :k]

    bases = tangent.estimate_bases(X, precomputed_neighbors=neighbors)   # (n, d, r)
    sigma = tangent.last_singular_values                                 # (n, r)

    # Tangent projection of the unit concept gradient (norm acts as a
    # confidence weight); fall back to the ambient gradient where the gradient
    # has essentially no support in the estimated tangent space.
    u = compute_unit_gradients(scorer, X, batch_size=batch_size, device=device)
    u_tan = project_to_bases(u, bases)
    zero = np.linalg.norm(u_tan, axis=1) < 1e-8
    u_tan[zero] = u[zero]

    # Seed step v_i and its spectrally-weighted form d_i (Eq. 7).
    v = (X * u_tan).sum(axis=1, keepdims=True) * u_tan                   # (n, d)
    coords = np.einsum("nd,ndr->nr", v, bases)                          # (n, r)
    sigma_safe = sigma + np.maximum(1e-4 * sigma[:, :1], 1e-12)
    d = np.einsum("nr,ndr->nd", coords * sigma_safe ** float(alpha), bases)
    d_norm = np.linalg.norm(d, axis=1) + 1e-12

    r_i = distances.mean(axis=1).astype(np.float64)                     # local radius
    lam = np.minimum(float(lambda_max), float(epsilon) * r_i / d_norm)
    return (X - lam[:, None] * d).astype(np.float32)


# ------------------------------------------------------------------- results
@dataclass
class MACEResult:
    """Edited representations plus the per-round evaluation trajectory."""

    train: np.ndarray
    val: np.ndarray
    test: np.ndarray
    history: list[dict] = field(default_factory=list)
    n_neighbors: int = 0
    rank: int = 0


# --------------------------------------------------------------- the eraser
@dataclass
class MACE:
    """Manifold-aware concept eraser (MACE / MACE+ / MACE++)."""

    variant: str = "mace++"
    epsilon: float = 0.1
    n_steps: int = 60
    lambda_max: float = 64.0
    alpha: float = 1.0
    n_neighbors: int | None = None          # None -> max(8, ceil(TwoNN intrinsic dim))
    n_cov_dirs: int = 2                     # CovMatch rank (MACE++ only)
    scorer_hidden: int = 512
    scorer_steps: int = 800
    scorer_patience: int = 4
    scorer_refit_every: int = 8
    seed: int = 0
    device: str | None = None
    # Per-round evaluation probe (used only for the trajectory, not the edit).
    eval_hidden: int = 128
    eval_steps: int = 300
    # Early stopping: halt once the concept probe sits at its majority floor.
    stop_at_floor: bool = True
    floor_tol: float = 0.01
    min_steps: int = 5
    verbose: bool = True

    def __post_init__(self) -> None:
        if self.variant not in _VARIANTS:
            raise ValueError(f"variant must be one of {_VARIANTS}, got {self.variant!r}")
        self._device = torch.device(
            self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        )

    # -------------------------------------------------------- preprocessing
    def _preprocess(self, X_tr, y_tr, X_va, X_te):
        Xtr, Xva, Xte = (np.asarray(a, dtype=np.float32) for a in (X_tr, X_va, X_te))
        if self.variant in ("mace+", "mace++"):
            leace = fit_leace_eraser(Xtr, y_tr)
            Xtr, Xva, Xte = leace.erase(Xtr), leace.erase(Xva), leace.erase(Xte)
            self._log(f"[preprocess] LEACE applied (rank {leace.rank})")
        if self.variant == "mace++":
            D = covariance_matching_directions(Xtr, y_tr, n_cov_dirs=self.n_cov_dirs)
            Xtr, Xva, Xte = (apply_projection(a, D) for a in (Xtr, Xva, Xte))
            self._log(f"[preprocess] CovMatch applied (rank {D.shape[1]})")
        return Xtr, Xva, Xte

    # --------------------------------------------------------------- public
    def fit_erase(
        self,
        X_train, y_train, X_val, y_val, X_test, y_test,
        *,
        control_train=None, control_val=None, control_test=None,
    ) -> MACEResult:
        """Run the variant's preprocessing + the iterative loop on all splits.

        ``y_*`` are the target-concept labels (erased). The optional
        ``control_*`` labels (a concept to *preserve*) are only used to log the
        control-probe accuracy in the trajectory.
        """
        from mace.eval import probe_accuracy  # local import avoids a cycle

        y_tr = np.asarray(y_train, dtype=np.int64)
        y_va = np.asarray(y_val, dtype=np.int64)
        y_te = np.asarray(y_test, dtype=np.int64)

        cur_tr, cur_va, cur_te = self._preprocess(X_train, y_tr, X_val, X_test)

        # Local-basis neighbourhood size: estimate intrinsic dim once.
        if self.n_neighbors is None:
            k = max(8, math.ceil(estimate_intrinsic_dim_two_nn(cur_tr)))
        else:
            k = int(self.n_neighbors)
        rank = max(1, k - 1)
        self._log(f"[setup] k={k} neighbours, tangent rank={rank}, variant={self.variant}")

        # Tangent reference is fit ONCE on the natural (pre-edit) train manifold
        # X^(0) and reused for every round; the edited splits query into this
        # fixed reference. This realises the manifold-constrained estimator of
        # Algorithm 1: kNN(x_i^(t-1); X^(0)) rather than kNN(x_i^(t-1); X^(t-1)).
        # The query point moves with the edits each round, but the neighbourhood
        # that defines the local tangent is always drawn from natural
        # representations, so the manifold estimate stays anchored to the region
        # natural inputs occupy (the MCH premise).
        tangent = LocalPCATangentEstimator(
            n_neighbors=k, intrinsic_dim=rank, device=str(self._device),
        ).fit(cur_tr)

        floor = float(np.bincount(y_tr).max() / len(y_tr))

        def evaluate(step: int, scorer_val_acc: float) -> dict:
            row = {"step": step, "scorer_val_acc": scorer_val_acc}
            row["concept_acc"] = probe_accuracy(
                cur_tr, y_tr, cur_va, y_va, cur_te, y_te,
                hidden=self.eval_hidden, steps=self.eval_steps,
                seed=self.seed + 7, device=self._device,
            )
            if control_train is not None:
                row["control_acc"] = probe_accuracy(
                    cur_tr, np.asarray(control_train), cur_va, np.asarray(control_val),
                    cur_te, np.asarray(control_test),
                    hidden=self.eval_hidden, steps=self.eval_steps,
                    seed=self.seed + 9, device=self._device,
                )
            return row

        history = [evaluate(0, float("nan"))]
        self._log_row(history[-1], floor)

        scorer = None
        scorer_val_acc = float("nan")
        for step in range(1, self.n_steps + 1):
            if scorer is None or (step - 1) % self.scorer_refit_every == 0:
                warm = (
                    {k_: v.detach().cpu() for k_, v in scorer.state_dict().items()}
                    if scorer is not None else None
                )
                res = train_concept_scorer(
                    cur_tr, y_tr, cur_va, y_va,
                    hidden=self.scorer_hidden, steps=self.scorer_steps,
                    patience=self.scorer_patience, seed=self.seed + step,
                    device=self._device, warm_start_state=warm,
                )
                scorer, scorer_val_acc = res.scorer, res.val_acc

            # Reuse the fixed X^(0) tangent reference fit above; only the query
            # points (the edited splits) change from round to round.
            edit = lambda X: mace_edit(  # noqa: E731
                X, scorer, tangent,
                epsilon=self.epsilon, lambda_max=self.lambda_max,
                alpha=self.alpha, device=self._device,
            )
            cur_tr, cur_va, cur_te = edit(cur_tr), edit(cur_va), edit(cur_te)

            history.append(evaluate(step, scorer_val_acc))
            self._log_row(history[-1], floor)

            if (
                self.stop_at_floor
                and step >= self.min_steps
                and history[-1]["concept_acc"] <= floor + self.floor_tol
            ):
                self._log(f"[stop] concept probe at majority floor ({floor:.3f}) — done.")
                break

        return MACEResult(
            train=cur_tr, val=cur_va, test=cur_te,
            history=history, n_neighbors=k, rank=rank,
        )

    # ----------------------------------------------------------- logging
    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg, flush=True)

    def _log_row(self, row: dict, floor: float) -> None:
        if not self.verbose:
            return
        ctrl = f"  control={row['control_acc']:.3f}" if "control_acc" in row else ""
        print(
            f"  step {row['step']:>3}  concept={row['concept_acc']:.3f} "
            f"(floor {floor:.3f}){ctrl}",
            flush=True,
        )
