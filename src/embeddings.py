"""src.embeddings — pooling, extraction, caching, normalization.

Wraps legacy `extracting_embeddings.py` logic in importable functions with an
explicit `normalize` flag so the paper-vs-code normalization discrepancy can
be tested as an ablation (normalized vs non-normalized inputs).
"""
from __future__ import annotations

import os
from typing import List, Optional

import numpy as np
import torch

from .data import l2_normalize


def cls_pool(last_hidden_state: torch.Tensor) -> torch.Tensor:
    return last_hidden_state[:, 0, :]


def mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    return (last_hidden_state * mask).sum(dim=1) / torch.clamp(mask.sum(dim=1), min=1e-9)


def pick_pooling(model_name: str) -> str:
    lower = model_name.lower()
    if any(k in lower for k in ("minilm", "roberta", "mpnet", "distilbert", "sentence")):
        return "mean"
    return "cls"


@torch.no_grad()
def extract_embeddings(
    texts: List[str],
    tokenizer,
    model,
    pooling: str = "cls",
    batch_size: int = 64,
    max_length: int = 128,
    device: str = "cpu",
    normalize: bool = False,
) -> np.ndarray:
    """Encode texts in mini-batches -> (N, D) float32 array."""
    all_embs = []
    model.eval()
    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        enc = tokenizer(batch, padding=True, truncation=True,
                        max_length=max_length, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        lhs = model(**enc).last_hidden_state
        emb = cls_pool(lhs) if pooling == "cls" else mean_pool(lhs, enc["attention_mask"])
        all_embs.append(emb.cpu().float().numpy())
    embs = np.concatenate(all_embs, axis=0).astype(np.float32)
    if normalize:
        embs = l2_normalize(embs)
    return embs


def save_split(embed_dir: str, dataset: str, split: str,
               embs: np.ndarray, labels: np.ndarray) -> None:
    os.makedirs(embed_dir, exist_ok=True)
    np.save(os.path.join(embed_dir, f"{dataset}_{split}_embs.npy"), embs)
    np.save(os.path.join(embed_dir, f"{dataset}_{split}_labels.npy"), labels)


def load_split(embed_dir: str, dataset: str, split: str):
    embs = torch.from_numpy(np.load(os.path.join(embed_dir, f"{dataset}_{split}_embs.npy"))).float()
    labels = torch.from_numpy(np.load(os.path.join(embed_dir, f"{dataset}_{split}_labels.npy"))).long()
    return embs, labels


def embedding_cache_paths(embed_dir: str, dataset: str, split: str):
    return (os.path.join(embed_dir, f"{dataset}_{split}_embs.npy"),
            os.path.join(embed_dir, f"{dataset}_{split}_labels.npy"))
