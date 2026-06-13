"""Nonlinear concept probe (``scorer``) and gradient utilities.

The scorer is a small MLP trained to predict the target concept from a
representation. Its *input gradient* :math:`\\nabla_x f(x)` is the direction in
which the representation most increases the concept logit; MACE erases the
locally manifold-supported part of this gradient (see :mod:`mace.erasure`).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConceptScorer(nn.Module):
    """Two-hidden-layer ReLU MLP mapping :math:`\\mathbb{R}^d \\to \\mathbb{R}`.

    Outputs a single logit; trained with binary cross-entropy.
    """

    def __init__(self, d: int, hidden: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


@dataclass
class ScorerTrainingResult:
    scorer: ConceptScorer
    val_acc: float


def train_concept_scorer(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    *,
    hidden: int = 512,
    steps: int = 800,
    batch_size: int = 512,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    seed: int = 0,
    patience: int = 4,
    val_check_every: int = 200,
    device: torch.device | None = None,
    warm_start_state: dict | None = None,
) -> ScorerTrainingResult:
    """Train a binary :class:`ConceptScorer` with early stopping on val accuracy.

    ``warm_start_state`` (a previous scorer's ``state_dict``) lets the scorer
    adapt incrementally across erasure rounds rather than restarting from a
    random init each refit, which lowers per-round gradient-direction noise.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)

    d = int(X_train.shape[1])
    scorer = ConceptScorer(d=d, hidden=hidden).to(device)
    if warm_start_state is not None:
        scorer.load_state_dict({k: v.to(device) for k, v in warm_start_state.items()})

    opt = torch.optim.AdamW(scorer.parameters(), lr=lr, weight_decay=weight_decay)
    x_tr = torch.from_numpy(X_train.astype(np.float32)).to(device)
    y_tr = torch.from_numpy(y_train.astype(np.float32)).to(device)
    x_va = torch.from_numpy(X_val.astype(np.float32)).to(device)
    y_va = torch.from_numpy(y_val.astype(np.float32)).to(device)
    n = len(x_tr)

    best_acc, best_state, no_improve = -1.0, None, 0
    for step in range(int(steps)):
        idx = torch.randint(0, n, (min(batch_size, n),), device=device)
        loss = F.binary_cross_entropy_with_logits(scorer(x_tr[idx]), y_tr[idx])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if patience > 0 and (step + 1) % val_check_every == 0:
            scorer.eval()
            with torch.no_grad():
                acc = ((scorer(x_va) >= 0).float() == y_va).float().mean().item()
            scorer.train()
            if acc > best_acc:
                best_acc = acc
                best_state = {k: v.detach().clone() for k, v in scorer.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    break

    if best_state is not None:
        scorer.load_state_dict(best_state)
    scorer.eval()
    with torch.no_grad():
        val_acc = float(((scorer(x_va) >= 0).float() == y_va).float().mean().item())
    return ScorerTrainingResult(scorer=scorer, val_acc=val_acc)


def compute_unit_gradients(
    scorer: ConceptScorer,
    X: np.ndarray,
    *,
    batch_size: int = 512,
    device: torch.device | None = None,
) -> np.ndarray:
    """Return per-row unit input gradients ``u_i = ∇f(x_i) / ‖∇f(x_i)‖``."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scorer = scorer.to(device).eval()
    out: list[np.ndarray] = []
    for start in range(0, len(X), batch_size):
        end = min(start + batch_size, len(X))
        x = torch.from_numpy(X[start:end].astype(np.float32)).to(device)
        x = x.detach().requires_grad_(True)
        score = scorer(x).sum()
        (g,) = torch.autograd.grad(score, x)
        g = g / g.norm(dim=1, keepdim=True).clamp_min(1e-12)
        out.append(g.detach().cpu().numpy())
    return np.concatenate(out, axis=0)
