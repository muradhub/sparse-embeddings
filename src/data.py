"""src.data — dataset registry, stratified splits, loaders.

Design (fixes legacy leakage):
  - Legacy code used AG-News TEST split for both checkpoint selection AND
    final reporting, and SST-2 VAL for both. There was no untouched test set.
  - Here: HF train -> stratified train/val split (val_fraction, default 0.1).
    HF val/test -> untouched TEST used exactly once for final reporting.
    Checkpoint selection uses VAL only.

Supported datasets (all public, Colab-fetchable via HF `datasets`):
  sst2      : nyu-mll/glue sst2                (sentiment, 2-class)
  agnews    : fancyzhx/ag_news                (news/topic, 4-class)
  imdb      : stanfordnlp/imdb                (sentiment long reviews, 2-class)
  trec      : SetFit/TREC-QC, coarse labels   (question/intent, 6-class)
  scicite   : legacy key -> ccdv/arxiv-classification (scientific, 11-class)
  arxiv     : ccdv/arxiv-classification       (scientific, 11-class; preferred key)

Parquet-only rule (datasets>=4 removed script support):
  - `cmap/go_trec` does not exist on the Hub -> use SetFit/TREC-QC
    (parquet, train/test, coarse labels in `label_coarse`).
  - `allenai/scicite` is script-based -> `RuntimeError: Dataset scripts are
    no longer supported` on datasets>=4 -> use ccdv/arxiv-classification
    (parquet, train/validation/test, 11 scientific-topic classes).

Each entry declares HF id, config, text columns, label column, and how to
map HF splits -> (train_pool, test). Text columns are concatenated with
' [SEP] ' when multiple (e.g. AG News title+description handled by loader).

All functions accept in-memory lists so unit tests run without network.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


DATASETS: Dict[str, Dict] = {
    "sst2": {
        "hf_id": "nyu-mll/glue",
        "hf_config": "sst2",
        "train_hf_split": "train",
        "test_hf_split": "validation",  # official test labels withheld
        "text_cols": ["sentence"],
        "label_col": "label",
        "label_names": {0: "Negative", 1: "Positive"},
        "domain": "sentiment/movie-reviews",
    },
    "agnews": {
        "hf_id": "fancyzhx/ag_news",
        "hf_config": None,
        "train_hf_split": "train",
        "test_hf_split": "test",
        "text_cols": ["text"],
        "label_col": "label",
        "label_names": {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"},
        "domain": "news/topic",
    },
    "imdb": {
        "hf_id": "stanfordnlp/imdb",
        "hf_config": None,
        "train_hf_split": "train",
        "test_hf_split": "test",
        "text_cols": ["text"],
        "label_col": "label",
        "label_names": {0: "Negative", 1: "Positive"},
        "domain": "sentiment/long-reviews",
    },
    "trec": {
        # Verified 2026-09-12: SetFit/TREC-QC is parquet (train 5.45k / test 500).
        # Coarse mapping observed on the Hub viewer (stable across ~100 rows):
        #   0=DESC  1=ENTY  2=ABBR  3=HUM  4=NUM  5=LOC
        # NOTE: this is NOT CogComp order (ABBR 0, ENTY 1, DESC 2, ...).
        "hf_id": "SetFit/TREC-QC",
        "hf_config": "default",
        "train_hf_split": "train",
        "test_hf_split": "test",
        "text_cols": ["text"],
        "label_col": "label_coarse",
        "label_names": {0: "Description", 1: "Entity", 2: "Abbreviation",
                        3: "Human", 4: "Numeric", 5: "Location"},
        "domain": "question/intent",
    },
    "scicite": {
        # Legacy key kept so existing notebooks listing 'scicite' keep working.
        # Now points at ccdv/arxiv-classification (parquet). Prefer key 'arxiv'.
        # NOTE: content is arXiv scientific-topic classification (11 classes),
        # not citation-intent; see `domain`. Display names are auto-populated
        # from the dataset's ClassLabel at load time (see load_hf_texts_labels).
        "hf_id": "ccdv/arxiv-classification",
        "hf_config": "default",
        "train_hf_split": "train",
        "test_hf_split": "test",
        "text_cols": ["text"],
        "label_col": "label",
        "label_names": {i: f"class-{i}" for i in range(11)},  # overwritten at load
        "domain": "scientific/arxiv-topic (replaces allenai/scicite; script-based, "
                  "unsupported in datasets>=4)",
    },
    "arxiv": {
        # Preferred key for the scientific slot (same dataset as 'scicite').
        "hf_id": "ccdv/arxiv-classification",
        "hf_config": "default",
        "train_hf_split": "train",
        "test_hf_split": "test",
        "text_cols": ["text"],
        "label_col": "label",
        "label_names": {i: f"class-{i}" for i in range(11)},  # overwritten at load
        "domain": "scientific/arxiv-topic",
    },
}


def stratified_train_val_split(
    texts: List[str],
    labels: np.ndarray,
    val_fraction: float = 0.1,
    seed: int = 42,
    max_train_samples: Optional[int] = None,
) -> Tuple[List[str], np.ndarray, List[str], np.ndarray]:
    """Stratified train/val split of the HF train pool.

    Preserves class ratios. Deterministic given seed. Optionally caps the
    TRAIN portion only (val uncapped) for smoke tests.
    """
    rng = np.random.RandomState(seed)
    labels = np.asarray(labels)
    tr_idx, va_idx = [], []
    for c in np.unique(labels):
        idx = np.where(labels == c)[0]
        rng.shuffle(idx)
        n_val = max(1, int(round(len(idx) * val_fraction)))
        va_idx.extend(idx[:n_val].tolist())
        tr_idx.extend(idx[n_val:].tolist())
    rng.shuffle(tr_idx)
    rng.shuffle(va_idx)
    tr_idx = np.array(tr_idx)
    va_idx = np.array(va_idx)
    if max_train_samples is not None and len(tr_idx) > max_train_samples:
        sel = rng.choice(len(tr_idx), max_train_samples, replace=False)
        tr_idx = tr_idx[sel]
    tr_texts = [texts[i] for i in tr_idx]
    va_texts = [texts[i] for i in va_idx]
    return tr_texts, labels[tr_idx], va_texts, labels[va_idx]


def make_loader(
    embs: torch.Tensor,
    labels: torch.Tensor,
    batch_size: int = 64,
    shuffle: bool = True,
    device: str = "cpu",
) -> DataLoader:
    ds = TensorDataset(embs, labels)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=0, pin_memory=(device == "cuda"))


def l2_normalize(embs: np.ndarray) -> np.ndarray:
    """L2-normalize rows to unit length (paper claim; legacy code omits)."""
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return (embs / norms).astype(np.float32)


def combine_text_columns(example: Dict, cols: List[str]) -> str:
    parts = [str(example.get(c, "")) for c in cols]
    return " [SEP] ".join(p for p in parts if p)


def _refresh_label_names(hf_ds, spec: Dict) -> None:
    """Best-effort display names from the dataset's own ClassLabel.

    Metrics use integer ids only; names never affect scores. Updates
    `spec["label_names"]` in place so later `DATASETS[ds]['label_names']`
    lookups (heatmaps, reports) show native names, e.g. the 11 arXiv
    categories. No-op for plain int64 columns (trec keeps its verified
    hardcoded mapping) and for string columns (handled by the remap path).
    """
    try:
        feat = hf_ds[spec["train_hf_split"]].features.get(spec["label_col"])
    except Exception:
        return
    names = getattr(feat, "names", None)
    if names:
        spec["label_names"] = {i: str(n) for i, n in enumerate(names)}


def load_hf_texts_labels(dataset: str):
    """Fetch raw texts+labels from HF. Train pool and held-out test.

    Returns dict with keys: train_texts, train_labels, test_texts, test_labels.
    Requires `datasets` + network; NOT used in local smoke tests.

    Parquet-only: script-based datasets raise on datasets>=4, so the
    registry contains parquet-native ids only (see module docstring).
    """
    from datasets import load_dataset as _load

    if dataset not in DATASETS:
        raise ValueError(f"Unknown dataset {dataset!r}. Choose from {sorted(DATASETS)}")
    spec = DATASETS[dataset]
    if spec["hf_config"] is None:
        ds = _load(spec["hf_id"])
    else:
        ds = _load(spec["hf_id"], spec["hf_config"])
    _refresh_label_names(ds, spec)
    def _extract(split):
        d = ds[split]
        texts = [combine_text_columns({k: d[k][i] for k in d.column_names}, spec["text_cols"])
                 for i in range(len(d))]
        labels = np.array(d[spec["label_col"]], dtype=np.int64)
        # remap string labels if needed (also fixes display names to match)
        if labels.dtype.kind in ("U", "S", "O"):
            uniq = sorted(map(str, np.unique(labels).tolist()))
            mapping = {v: i for i, v in enumerate(uniq)}
            labels = np.array([mapping[str(v)] for v in labels], dtype=np.int64)
            spec["label_names"] = {i: name for name, i in mapping.items()}
        return texts, labels
    tr_texts, tr_labels = _extract(spec["train_hf_split"])
    te_texts, te_labels = _extract(spec["test_hf_split"])
    return {"train_texts": tr_texts, "train_labels": tr_labels,
            "test_texts": te_texts, "test_labels": te_labels, "spec": spec}
