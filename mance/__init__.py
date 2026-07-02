"""MANCE — Manifold Aware Concept Erasure.

A clean reference implementation of the MANCE family of manifold-aware concept
erasers:

* ``mance``   — the iterative tangent-constrained editing loop;
* ``mance+``  — LEACE preprocessing, then the loop;
* ``mance++`` — LEACE + CovMatch preprocessing, then the loop (default).

Quick start::

    from mance import MANCE
    from mance.data import load_or_extract_biasbios

    reps = load_or_extract_biasbios()
    eraser = MANCE(variant="mance++", epsilon=0.1)
    result = eraser.fit_erase(
        reps.X_train, reps.gender_train, reps.X_val, reps.gender_val,
        reps.X_test, reps.gender_test,
        control_train=reps.profession_train,
        control_val=reps.profession_val,
        control_test=reps.profession_test,
    )
"""
from mance.erasure import MANCE, MANCEResult, mance_edit
from mance.eval import probe_accuracy

__all__ = ["MANCE", "MANCEResult", "mance_edit", "probe_accuracy"]
__version__ = "0.1.0"
