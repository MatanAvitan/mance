"""MACE — Manifold Aware Concept Erasure.

A clean reference implementation of the MACE family of manifold-aware concept
erasers:

* ``mace``   — the iterative tangent-constrained editing loop;
* ``mace+``  — LEACE preprocessing, then the loop;
* ``mace++`` — LEACE + CovMatch preprocessing, then the loop (default).

Quick start::

    from mace import MACE
    from mace.data import load_or_extract_biasbios

    reps = load_or_extract_biasbios()
    eraser = MACE(variant="mace++", epsilon=0.1)
    result = eraser.fit_erase(
        reps.X_train, reps.gender_train, reps.X_val, reps.gender_val,
        reps.X_test, reps.gender_test,
        control_train=reps.profession_train,
        control_val=reps.profession_val,
        control_test=reps.profession_test,
    )
"""
from mace.erasure import MACE, MACEResult, mace_edit
from mace.eval import probe_accuracy

__all__ = ["MACE", "MACEResult", "mace_edit", "probe_accuracy"]
__version__ = "0.1.0"
