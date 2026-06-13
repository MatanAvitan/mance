"""BiasBios representations for the MACE demo.

We extract last-token hidden states from a frozen language model over the
*Bias in Bios* biographies (De-Arteaga et al., 2019). The target concept to
erase is **gender** (binary); the control concept to preserve is **profession**
(28-way). Representations are cached to a single ``.npz`` so the notebook runs
instantly on reruns and offline.

The default model is ``Qwen/Qwen2.5-0.5B`` at layer 12, matching the paper's
small-model BiasBios panel.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class BiasBiosReps:
    """Train/val/test representations with gender + profession labels."""

    X_train: np.ndarray
    gender_train: np.ndarray
    profession_train: np.ndarray
    X_val: np.ndarray
    gender_val: np.ndarray
    profession_val: np.ndarray
    X_test: np.ndarray
    gender_test: np.ndarray
    profession_test: np.ndarray
    model_name: str = ""
    layer: int = -1

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            X_train=self.X_train, gender_train=self.gender_train,
            profession_train=self.profession_train,
            X_val=self.X_val, gender_val=self.gender_val,
            profession_val=self.profession_val,
            X_test=self.X_test, gender_test=self.gender_test,
            profession_test=self.profession_test,
            model_name=np.array(self.model_name), layer=np.array(self.layer),
        )

    @classmethod
    def load(cls, path: str | Path) -> "BiasBiosReps":
        z = np.load(path, allow_pickle=False)
        return cls(
            X_train=z["X_train"], gender_train=z["gender_train"],
            profession_train=z["profession_train"],
            X_val=z["X_val"], gender_val=z["gender_val"],
            profession_val=z["profession_val"],
            X_test=z["X_test"], gender_test=z["gender_test"],
            profession_test=z["profession_test"],
            model_name=str(z["model_name"]), layer=int(z["layer"]),
        )

    def summary(self) -> str:
        d = self.X_train.shape[1]
        return (
            f"BiasBios reps from {self.model_name} (layer {self.layer}), d={d}\n"
            f"  train={len(self.X_train)}  val={len(self.X_val)}  test={len(self.X_test)}\n"
            f"  gender balance (train): {np.bincount(self.gender_train)}\n"
            f"  #professions: {len(np.unique(self.profession_train))}"
        )


def _stratified_split(n, labels, seed, ratios=(0.6, 0.2, 0.2)):
    from sklearn.model_selection import train_test_split

    idx = np.arange(n)
    tr, tmp = train_test_split(idx, test_size=1 - ratios[0], random_state=seed, stratify=labels)
    val_share = ratios[1] / (ratios[1] + ratios[2])
    va, te = train_test_split(tmp, test_size=1 - val_share, random_state=seed, stratify=labels[tmp])
    return tr, va, te


def _stratified_subset(labels, max_samples, seed):
    if max_samples >= len(labels):
        return np.arange(len(labels))
    rng = np.random.default_rng(seed)
    out: list[int] = []
    for cls in np.unique(labels):
        cls_idx = np.where(labels == cls)[0]
        take = max(1, round(max_samples * len(cls_idx) / len(labels)))
        out.extend(rng.choice(cls_idx, size=min(take, len(cls_idx)), replace=False).tolist())
    out = np.array(out)
    if len(out) > max_samples:
        out = rng.choice(out, size=max_samples, replace=False)
    rng.shuffle(out)
    return out


def extract_biasbios_representations(
    *,
    model_name: str = "Qwen/Qwen2.5-0.5B",
    layer: int = 12,
    max_samples: int = 12000,
    max_length: int = 256,
    batch_size: int = 64,
    dataset_name: str = "LabHC/bias_in_bios",
    seed: int = 0,
    device: str | None = None,
    dtype: str = "bfloat16",
) -> BiasBiosReps:
    """Extract last-token layer-``layer`` hidden states over Bias in Bios.

    Pools the dataset splits, takes a stratified subset of ``max_samples``,
    then makes a stratified 60/20/20 split on gender. Returns a
    :class:`BiasBiosReps`.
    """
    import torch
    from datasets import load_dataset
    from transformers import AutoModel, AutoTokenizer

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = getattr(torch, dtype) if device.startswith("cuda") else torch.float32

    ds = load_dataset(dataset_name)
    texts, gender, profession = [], [], []
    for split in ("train", "dev", "test"):
        if split not in ds:
            continue
        s = ds[split]
        texts.extend([str(t) for t in s["hard_text"]])
        gender.extend([int(g) for g in s["gender"]])
        profession.extend([int(p) for p in s["profession"]])
    texts = np.asarray(texts, dtype=object)
    gender = np.asarray(gender, dtype=np.int64)
    profession = np.asarray(profession, dtype=np.int64)

    sub = _stratified_subset(gender, max_samples, seed)
    texts, gender, profession = texts[sub], gender[sub], profession[sub]

    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModel.from_pretrained(model_name, torch_dtype=torch_dtype).to(device).eval()

    feats = np.empty((len(texts), model.config.hidden_size), dtype=np.float32)
    for start in range(0, len(texts), batch_size):
        end = min(start + batch_size, len(texts))
        batch = list(texts[start:end])
        enc = tok(
            batch, return_tensors="pt", padding=True, truncation=True, max_length=max_length,
        ).to(device)
        with torch.no_grad():
            hs = model(**enc, output_hidden_states=True).hidden_states[layer]  # (b, T, d)
        last = enc["attention_mask"].sum(dim=1) - 1                            # last real token
        rows = hs[torch.arange(hs.shape[0]), last]
        feats[start:end] = rows.float().cpu().numpy()
        print(f"  extracted {end}/{len(texts)}", end="\r", flush=True)
    print()

    tr, va, te = _stratified_split(len(feats), gender, seed)
    return BiasBiosReps(
        X_train=feats[tr], gender_train=gender[tr], profession_train=profession[tr],
        X_val=feats[va], gender_val=gender[va], profession_val=profession[va],
        X_test=feats[te], gender_test=gender[te], profession_test=profession[te],
        model_name=model_name, layer=layer,
    )


def load_or_extract_biasbios(
    cache_path: str | Path = "data/biasbios_qwen2.5-0.5b_l12.npz",
    **extract_kwargs,
) -> BiasBiosReps:
    """Load cached BiasBios reps if present, otherwise extract and cache them."""
    cache_path = Path(cache_path)
    if cache_path.exists():
        print(f"[data] loading cached representations from {cache_path}")
        return BiasBiosReps.load(cache_path)
    print(f"[data] no cache at {cache_path}; extracting (downloads model + dataset)…")
    reps = extract_biasbios_representations(**extract_kwargs)
    reps.save(cache_path)
    print(f"[data] saved representations to {cache_path}")
    return reps
