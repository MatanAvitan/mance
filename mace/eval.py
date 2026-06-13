"""Probing evaluation for concept erasure.

To measure how much concept information survives an edit we train a *fresh*
nonlinear MLP probe on the edited training split and report its accuracy on the
edited test split. A target concept is erased well when its probe accuracy
falls to the majority-class floor; a control concept is preserved when its
probe accuracy stays high.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def probe_accuracy(
    X_train: np.ndarray, y_train: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
    X_test: np.ndarray, y_test: np.ndarray,
    *,
    hidden: int = 128,
    steps: int = 300,
    batch_size: int = 512,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    patience: int = 6,
    val_check_every: int = 25,
    seed: int = 0,
    device: torch.device | None = None,
) -> float:
    """Train an MLP probe on ``(X_train, y_train)``; return test accuracy.

    Handles both binary and multi-class labels. Early-stops on validation
    accuracy.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    y_train = np.asarray(y_train, dtype=np.int64)
    y_val = np.asarray(y_val, dtype=np.int64)
    y_test = np.asarray(y_test, dtype=np.int64)

    d = int(X_train.shape[1])
    n_classes = int(max(y_train.max(), y_val.max(), y_test.max())) + 1
    binary = n_classes <= 2
    out_dim = 1 if binary else n_classes
    torch.manual_seed(seed)

    net = nn.Sequential(
        nn.Linear(d, hidden), nn.GELU(), nn.Linear(hidden, out_dim),
    ).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=weight_decay)

    Xt = torch.from_numpy(X_train.astype(np.float32)).to(device)
    Xv = torch.from_numpy(X_val.astype(np.float32)).to(device)
    Xe = torch.from_numpy(X_test.astype(np.float32)).to(device)
    if binary:
        yt = torch.from_numpy(y_train.astype(np.float32)).to(device)
        yv = torch.from_numpy(y_val.astype(np.float32)).to(device)
        ye = torch.from_numpy(y_test.astype(np.float32)).to(device)
        loss_fn = lambda logit, y: F.binary_cross_entropy_with_logits(logit.squeeze(-1), y)  # noqa: E731
        acc_fn = lambda logit, y: ((logit.squeeze(-1) >= 0).long() == y.long()).float().mean().item()  # noqa: E731
    else:
        yt = torch.from_numpy(y_train).to(device)
        yv = torch.from_numpy(y_val).to(device)
        ye = torch.from_numpy(y_test).to(device)
        loss_fn = F.cross_entropy
        acc_fn = lambda logit, y: (logit.argmax(-1) == y).float().mean().item()  # noqa: E731

    n = len(Xt)
    best_acc, best_state, no_improve = -1.0, None, 0
    for step in range(int(steps)):
        idx = torch.randint(0, n, (min(batch_size, n),), device=device)
        loss = loss_fn(net(Xt[idx]), yt[idx])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % val_check_every == 0:
            net.eval()
            with torch.no_grad():
                acc = acc_fn(net(Xv), yv)
            net.train()
            if acc > best_acc:
                best_acc, no_improve = acc, 0
                best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
            else:
                no_improve += 1
                if no_improve >= patience:
                    break
    if best_state is not None:
        net.load_state_dict(best_state)
    net.eval()
    with torch.no_grad():
        return float(acc_fn(net(Xe), ye))
