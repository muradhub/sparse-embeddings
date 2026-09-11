"""src.evaluation — reconstruction, sparsity, downstream metrics.

Covers §5A–5C of the brief:
  A. train/val/test discipline + multi-seed mean/std/CI
  B. reconstruction: MSE, cosine sim, explained variance, error distribution,
     normalized reconstruction error; recon quality vs sparsity
  C. sparsity: L0 per example, mean/median L0, sparsity ratio, activation
     frequency, dead features, activation distribution, feature utilization,
     recon-vs-L0

Downstream (§7–8): dense vs recon vs sparse vs PCA / random-projection /
dense-AE baselines with accuracy, macro F1, per-class F1, precision, recall.
Same LogReg protocol everywhere (documented scaling choice).
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score)
from sklearn.preprocessing import StandardScaler
from sklearn.random_projection import GaussianRandomProjection


# --- representation extraction ---------------------------------------------

@torch.no_grad()
def encode_all(model, embs: torch.Tensor, batch_size: int = 512,
               device: str = "cpu") -> np.ndarray:
    model.eval()
    out = []
    for s in range(0, len(embs), batch_size):
        out.append(model.encode(embs[s:s + batch_size].to(device)).cpu().numpy())
    return np.concatenate(out, axis=0)


@torch.no_grad()
def reconstruct_all(model, embs: torch.Tensor, batch_size: int = 512,
                    device: str = "cpu") -> np.ndarray:
    model.eval()
    out = []
    for s in range(0, len(embs), batch_size):
        b = embs[s:s + batch_size].to(device)
        z = model.encode(b)
        # decode: support all model types (they all expose .decoder)
        out.append(model.decoder(z).cpu().numpy())
    return np.concatenate(out, axis=0)


# --- reconstruction metrics -------------------------------------------------

def reconstruction_metrics(orig: np.ndarray, recon: np.ndarray) -> Dict:
    """MSE, cosine sim, explained variance, normalized MSE, error distribution."""
    orig = np.asarray(orig, dtype=np.float64)
    recon = np.asarray(recon, dtype=np.float64)
    err = recon - orig
    mse = float(np.mean(err ** 2))
    # cosine similarity per example then mean
    on = np.linalg.norm(orig, axis=1) + 1e-12
    rn = np.linalg.norm(recon, axis=1) + 1e-12
    cos = float(np.mean(np.sum(orig * recon, axis=1) / (on * rn)))
    # explained variance: 1 - Var(err)/Var(orig)
    var_orig = float(np.var(orig))
    var_err = float(np.var(err))
    explained = float(1.0 - var_err / (var_orig + 1e-12))
    # normalized MSE: MSE / mean squared norm of orig
    nmse = float(mse / (np.mean(orig ** 2) + 1e-12))
    per_ex_mse = np.mean(err ** 2, axis=1)
    return {
        "mse": mse,
        "cosine_sim": cos,
        "explained_variance": explained,
        "normalized_mse": nmse,
        "per_example_mse_mean": float(per_ex_mse.mean()),
        "per_example_mse_std": float(per_ex_mse.std()),
        "per_example_mse_p50": float(np.median(per_ex_mse)),
        "per_example_mse_p90": float(np.percentile(per_ex_mse, 90)),
        "per_example_mse_p99": float(np.percentile(per_ex_mse, 99)),
    }


# --- sparsity metrics -------------------------------------------------------

def sparsity_metrics(codes: np.ndarray, eps: float = 1e-9) -> Dict:
    """Per-example and aggregate sparsity statistics."""
    codes = np.asarray(codes, dtype=np.float64)
    active = (np.abs(codes) > eps)
    l0 = active.sum(axis=1).astype(float)  # active features per example
    H = codes.shape[1]
    sparsity_ratio = float(1.0 - active.mean())
    freq = active.mean(axis=0)  # activation frequency per feature
    dead = int((freq == 0.0).sum())
    util_1pct = float((freq > 0.01).mean())  # features active for >1% of examples
    return {
        "mean_l0": float(l0.mean()),
        "median_l0": float(np.median(l0)),
        "std_l0": float(l0.std()),
        "min_l0": float(l0.min()),
        "max_l0": float(l0.max()),
        "sparsity_ratio": sparsity_ratio,
        "mean_active_dims": float(l0.mean()),
        "hidden_dim": int(H),
        "dead_features": dead,
        "dead_fraction": float(dead / max(H, 1)),
        "feature_utilization_1pct": util_1pct,
        "mean_activation": float(np.abs(codes).mean()),
        "activation_freq_mean": float(freq.mean()),
        "activation_freq_std": float(freq.std()),
        "l0_p10": float(np.percentile(l0, 10)),
        "l0_p90": float(np.percentile(l0, 90)),
    }


def recon_vs_l0(orig: np.ndarray, recon: np.ndarray, codes: np.ndarray,
                n_bins: int = 5) -> List[Dict]:
    """Distribution of reconstruction quality against L0 (binned)."""
    l0 = (np.abs(codes) > 1e-9).sum(axis=1)
    per_ex = np.mean((recon - orig) ** 2, axis=1)
    if len(l0) < n_bins:
        return []
    qs = np.quantile(l0, np.linspace(0, 1, n_bins + 1))
    out = []
    for b in range(n_bins):
        lo, hi = qs[b], qs[b + 1]
        m = (l0 >= lo) & (l0 <= hi) if b == n_bins - 1 else (l0 >= lo) & (l0 < hi)
        if m.sum() == 0:
            continue
        out.append({"bin": b, "l0_lo": float(lo), "l0_hi": float(hi),
                    "n": int(m.sum()), "mean_mse": float(per_ex[m].mean())})
    return out


# --- downstream probes ------------------------------------------------------

def logreg_scores(X_train: np.ndarray, y_train: np.ndarray,
                  X_test: np.ndarray, y_test: np.ndarray,
                  C: float = 1.0, max_iter: int = 1000,
                  scale: bool = True, seed: int = 42) -> Dict:
    """Single LogReg probe. Returns acc, macro/micro F1, precision, recall, per-class F1."""
    Xt, Xe = np.asarray(X_train), np.asarray(X_test)
    if scale:
        sc = StandardScaler()
        Xt = sc.fit_transform(Xt)
        Xe = sc.transform(Xe)
    clf = LogisticRegression(max_iter=max_iter, C=C, solver="lbfgs",
                             multi_class="auto", random_state=seed, n_jobs=-1)
    clf.fit(Xt, np.asarray(y_train))
    pred = clf.predict(Xe)
    yt = np.asarray(y_test)
    per_class = f1_score(yt, pred, average=None, zero_division=0)
    return {
        "accuracy": float(accuracy_score(yt, pred)),
        "macro_f1": float(f1_score(yt, pred, average="macro", zero_division=0)),
        "micro_f1": float(f1_score(yt, pred, average="micro", zero_division=0)),
        "macro_precision": float(precision_score(yt, pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(yt, pred, average="macro", zero_division=0)),
        "per_class_f1": [float(v) for v in per_class],
    }


def downstream_suite(dense_tr, dense_te, recon_tr, recon_te, sparse_tr, sparse_te,
                     y_tr, y_te, C: float = 1.0, max_iter: int = 1000,
                     seed: int = 42, extra_baselines: Optional[Dict[str, Tuple]] = None) -> Dict[str, Dict]:
    """Compare dense / recon / sparse (+ optional PCA/RP/dense-AE) under one protocol."""
    res = {
        "dense": logreg_scores(dense_tr, y_tr, dense_te, y_te, C, max_iter, scale=True, seed=seed),
        "reconstructed": logreg_scores(recon_tr, y_tr, recon_te, y_te, C, max_iter, scale=True, seed=seed),
        "sparse": logreg_scores(sparse_tr, y_tr, sparse_te, y_te, C, max_iter, scale=False, seed=seed),
    }
    if extra_baselines:
        for name, (btr, bte) in extra_baselines.items():
            res[name] = logreg_scores(btr, y_tr, bte, y_te, C, max_iter, scale=True, seed=seed)
    return res


def pca_baseline(dense_tr: np.ndarray, dense_te: np.ndarray, n_components: int,
                 seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    p = PCA(n_components=n_components, random_state=seed)
    return p.fit_transform(dense_tr), p.transform(dense_te)


def random_projection_baseline(dense_tr, dense_te, n_components: int, seed: int = 42):
    rp = GaussianRandomProjection(n_components=n_components, random_state=seed)
    return rp.fit_transform(dense_tr), rp.transform(dense_te)


# --- multi-seed aggregation -------------------------------------------------

def mean_std_ci(values: List[float], ci: float = 0.95) -> Dict[str, float]:
    """Mean / std / 95% CI (normal approx) for multi-seed reporting."""
    import math
    v = np.asarray(values, dtype=float)
    m = float(v.mean())
    s = float(v.std(ddof=1)) if len(v) > 1 else 0.0
    # 95% normal approx; for n>=2 use 1.96; small-n caveat documented
    h = 1.96 * s / math.sqrt(len(v)) if len(v) > 1 else 0.0
    return {"mean": m, "std": s, "ci95_half": float(h), "n": len(v),
            "min": float(v.min()), "max": float(v.max())}
