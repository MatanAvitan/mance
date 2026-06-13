"""Builds notebooks/biasbios_quickstart.ipynb from source cells.

Run from the repo root:  python notebooks/_build_notebook.py
"""
import pathlib

import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []


def md(text):
    cells.append(nbf.v4.new_markdown_cell(text.strip("\n")))


def code(text):
    cells.append(nbf.v4.new_code_cell(text.strip("\n")))


md(r"""
# MACE on Bias-in-Bios — Quickstart

<img src="../figures/figure1_mace_3d.png" width="760">

**Figure 1 (from the paper).** *(A)* Natural representations concentrate near a
low-dimensional manifold $\mathcal{M}\subset\mathbb{R}^d$, coloured by concept
strength. *(B)* Following the raw scorer gradient erases the concept but leaves
$\mathcal{M}$, corrupting unlabeled control concepts (off-manifold collateral
damage). *(C)* **MACE** projects the scorer gradient onto the locally
estimated tangent space $T_{x}\mathcal{M}$ and takes a bounded on-manifold step,
so control concepts are preserved.

This notebook makes that picture concrete: we erase **gender** from
`Qwen/Qwen2.5-0.5B` biography representations while preserving **profession**
(the control concept). A concept is **erased** when a freshly-trained nonlinear
probe can no longer recover it (accuracy → majority-class floor) and
**preserved** when its probe accuracy stays high.

The three variants differ only in optional closed-form preprocessing before the
loop: `mace` (none), `mace+` (LEACE), `mace++` (LEACE + CovMatch, default).
""")

code(r"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path.cwd().parent))  # run-from-notebooks/ convenience

import numpy as np
import torch
import matplotlib.pyplot as plt

from mace import MACE, probe_accuracy
from mace.data import load_or_extract_biasbios

# --- Paper aesthetic (whitegrid + serif) and Okabe-Ito colour palette --------
plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "font.family": "serif", "font.size": 11,
    "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
    "axes.edgecolor": "#bbb", "axes.grid": True, "axes.grid.axis": "y",
    "grid.color": "#eaeaea", "grid.linewidth": 0.8,
    "xtick.color": "#444", "ytick.color": "#444", "legend.frameon": False,
})
C_CLEAN   = "#9e9e9e"   # neutral gray  — clean / un-erased reference
C_CONCEPT = "#D55E00"   # vermilion     — the target concept (gender) being erased
C_CONTROL = "#0072B2"   # deep blue     — the control concept (profession) preserved
C_FLOOR   = "#222222"   # majority-class floor (chance)

device = "cuda" if torch.cuda.is_available() else "cpu"
np.random.seed(0); torch.manual_seed(0)
print("device:", device)
""")

md(r"""
## 1. Representations

`load_or_extract_biasbios` loads cached representations from `data/` if present,
otherwise it downloads `Qwen/Qwen2.5-0.5B` + the `LabHC/bias_in_bios` dataset and
extracts the **last-token hidden state at layer 12** (one forward pass per
biography). Everything downstream operates on these frozen vectors.
""")

code(r"""
reps = load_or_extract_biasbios("../data/biasbios_qwen2.5-0.5b_l12.npz")
print(reps.summary())
""")

md(r"""
## 2. Baseline: how recoverable are gender and profession *before* erasure?
""")

code(r"""
gender_clean = probe_accuracy(
    reps.X_train, reps.gender_train, reps.X_val, reps.gender_val,
    reps.X_test, reps.gender_test, device=device,
)
prof_clean = probe_accuracy(
    reps.X_train, reps.profession_train, reps.X_val, reps.profession_val,
    reps.X_test, reps.profession_test, device=device,
)
gender_floor = float(np.bincount(reps.gender_train).max() / len(reps.gender_train))
print(f"gender probe (clean):      {gender_clean:.3f}   (majority floor {gender_floor:.3f})")
print(f"profession probe (clean):  {prof_clean:.3f}")
""")

md(r"""
## 3. Run the MACE variants

Each variant runs the iterative editing loop with the same trust-region budget
`epsilon=0.1`. We pass the profession labels as the *control* so the loop logs
profession-probe accuracy each round — profession is **never** used by the edit
itself (only gender is).

The local tangent bases are estimated from each point's nearest neighbours
among the **natural (unedited) representations** $X^{(0)}$: the edits move the
query points round by round, but the manifold reference stays fixed at the
natural geometry (Algorithm 1, $k\mathrm{NN}(x_i^{(t-1)}; X^{(0)})$).

Expect roughly 1–2 minutes per variant on a single GPU (longer on CPU); each
round prints the freshly retrained gender / profession probe accuracies.
""")

code(r"""
results = {}
for variant in ["mace", "mace+", "mace++"]:
    print(f"\n=== {variant} ===")
    eraser = MACE(variant=variant, epsilon=0.1, n_steps=12, seed=0, device=device)
    results[variant] = eraser.fit_erase(
        reps.X_train, reps.gender_train, reps.X_val, reps.gender_val,
        reps.X_test, reps.gender_test,
        control_train=reps.profession_train,
        control_val=reps.profession_val,
        control_test=reps.profession_test,
    )
""")

md(r"""
## 4. Gender collapses toward chance; profession is preserved

For the default **MACE⁺⁺**, the gender probe (vermilion) falls from its clean
value toward the majority-class floor, while the profession control probe (blue)
stays close to its clean accuracy — the surgical leakage/preservation tradeoff
the Manifold Constraint Hypothesis (MCH) predicts.
""")

code(r"""
h      = results["mace++"].history
steps  = [r["step"] for r in h]
gender = [r["concept_acc"] for r in h]
prof   = [r["control_acc"] for r in h]
x_pad  = (steps[-1] - steps[0]) * 0.30

fig, ax = plt.subplots(figsize=(7.4, 4.6))

# Reference levels: clean gender accuracy (start) and the majority-class floor.
ax.axhline(gender_clean, ls=":",  c=C_CLEAN, lw=1.3, zorder=1)
ax.axhline(gender_floor, ls="--", c=C_FLOOR, lw=1.1, zorder=1)
ax.text(steps[0], gender_clean, "gender (clean)", va="bottom", ha="left",
        fontsize=9, c=C_CLEAN)
ax.text(steps[0], gender_floor, "chance floor",  va="bottom", ha="left",
        fontsize=9, c=C_FLOOR)

# Residual gender leakage = the gap between the gender curve and the floor;
# the shaded area shrinks to nothing as MACE++ edits round by round.
ax.fill_between(steps, gender, gender_floor, color=C_CONCEPT, alpha=0.10,
                lw=0, zorder=1)

# Trajectories — white-cored markers read cleanly over the shaded area.
ax.plot(steps, prof,   "-o", c=C_CONTROL, lw=2.5, ms=5.5, mfc="white", mew=1.5, zorder=4)
ax.plot(steps, gender, "-o", c=C_CONCEPT, lw=2.5, ms=5.5, mfc="white", mew=1.5, zorder=4)

# Direct end-of-line labels instead of a boxed legend; offset vertically so
# they never collide when the two curves finish close together.
ax.annotate("profession\ncontrol — preserved", xy=(steps[-1], prof[-1]),
            xytext=(8, 13), textcoords="offset points", va="bottom", ha="left",
            fontsize=9.5, c=C_CONTROL, weight="bold")
ax.annotate("gender\ntarget — erased", xy=(steps[-1], gender[-1]),
            xytext=(8, -13), textcoords="offset points", va="top", ha="left",
            fontsize=9.5, c=C_CONCEPT, weight="bold")

ax.set_xlabel("editing round")
ax.set_ylabel("nonlinear-probe accuracy")
ax.set_title("Erasing gender from Qwen2.5-0.5B with MACE$^{++}$", pad=10)
ax.set_xlim(steps[0], steps[-1] + x_pad)
ymin = min(gender_floor, min(gender), min(prof))
ymax = max(gender_clean, max(prof))
ax.set_ylim(ymin - 0.04, ymax + 0.05)
fig.tight_layout()
fig.savefig("../figures/biasbios_trajectory.png", dpi=200, bbox_inches="tight")
plt.show()
""")

md(r"""
## 5. Final numbers
""")

code(r"""
print(f"{'variant':10s} {'gender ↓':>10s} {'profession (preserved)':>24s}")
print(f"{'clean':10s} {gender_clean:>10.3f} {prof_clean:>24.3f}")
for v in ["mace", "mace+", "mace++"]:
    last = results[v].history[-1]
    print(f"{v:10s} {last['concept_acc']:>10.3f} {last['control_acc']:>24.3f}")
print(f"\n(gender majority-class floor = {gender_floor:.3f})")
""")

md(r"""
## 6. Where this sits in the paper

Across the full NLP suite (13 language models × 3 concepts), the MACE family
attains the lowest residual concept leakage at every control-degradation budget
$\Delta Y$, and **MACE⁺⁺** is state-of-the-art for nonlinear erasure. Lower is
better; the y-axis is the mean gap between the post-erasure concept-probe
accuracy and its chance floor.

<img src="../figures/aggregated_nlp_results.png" width="640">

**Takeaway.** The closed-form preprocessing in MACE⁺⁺ (LEACE + CovMatch)
combined with the on-manifold iterative loop drives gender leakage to chance
while keeping the profession control probe close to its clean accuracy — exactly
the behaviour Figure 1 illustrates.
""")

nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"},
}
out = pathlib.Path(__file__).parent / "biasbios_quickstart.ipynb"
nbf.write(nb, str(out))
print("wrote", out)
