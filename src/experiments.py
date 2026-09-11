"""src.experiments — orchestration, stability matching, hypothesis mapping.

Provides:
  - run_single_experiment(): embeddings already cached -> train SAE -> eval
    recon/sparsity/downstream on VAL (model selection) and TEST (final, once)
  - feature stability across seeds (Hungarian matching on decoder directions,
    activation correlation, top-example overlap)
  - HYPOTHESES registry mapping H1..H6 -> required evidence
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

HYPOTHESES = {
    "H1": "Sparse representations retain most downstream information while "
          "using substantially fewer active dimensions.",
    "H2": "There exists a sparsity range where reconstruction/downstream "
          "degrade slowly relative to active-feature reduction.",
    "H3": "The C-SPINE objective yields more semantically coherent features "
          "than an equivalent non-sparse autoencoder.",
    "H4": "Learned sparse features exhibit stability across random seeds.",
    "H5": "The representation remains useful across datasets/domains.",
    "H6": "The advantage is not explained entirely by ordinary dimensionality "
          "reduction (PCA / random projection).",
}

HYPOTHESIS_EVIDENCE = {
    "H1": ["downstream_suite (sparse vs dense delta)", "sparsity_metrics (mean L0)"],
    "H2": ["sparsity sweep: sparsity<->MSE + sparsity<->accuracy curves"],
    "H3": ["mechanism ablation cspine vs dense_ae + purity/coherence report"],
    "H4": ["multi-seed matching: cosine/correlation/overlap stats"],
    "H5": ["generalization portfolio + cross-domain transfer matrix"],
    "H6": ["PCA / RP / dense-AE baselines under identical probe protocol"],
}


def decoder_direction_similarity(dec_A: np.ndarray, dec_B: np.ndarray):
    """Match features across two seeds via decoder columns.

    dec: (hidden, embed) weight matrices (Linear decoder weight T).
    Returns mean matched cosine, full assignment, correlation stats.
    Cosine on L2-normalized decoder directions; Hungarian maximizes total sim.
    """
    A = dec_A / (np.linalg.norm(dec_A, axis=1, keepdims=True) + 1e-12)
    B = dec_B / (np.linalg.norm(dec_B, axis=1, keepdims=True) + 1e-12)
    sim = A @ B.T  # (Ha, Hb)
    # Hungarian on cost = -sim (assumes square; else pad by truncation to min)
    n = min(sim.shape)
    cost = -sim[:n, :n]
    ra, ca = linear_sum_assignment(cost)
    matched = sim[ra, ca]
    return {"mean_matched_cosine": float(matched.mean()),
            "median_matched_cosine": float(np.median(matched)),
            "frac_gt_0.7": float((matched > 0.7).mean()),
            "n_matched": int(n)}


def activation_correlation(codes_A: np.ndarray, codes_B: np.ndarray,
                           row_assign=None, col_assign=None) -> Dict:
    """Correlation of matched-feature activations across same inputs.

    If col_assign provided (from decoder matching), reorder B columns first.
    Reports mean diagonal correlation of matched pairs.
    """
    if col_assign is not None:
        codes_B = codes_B[:, np.asarray(col_assign)]
    n = min(codes_A.shape[1], codes_B.shape[1])
    A, B = codes_A[:, :n], codes_B[:, :n]
    cors = []
    for j in range(n):
        a, b = A[:, j], B[:, j]
        if a.std() < 1e-12 or b.std() < 1e-12:
            cors.append(0.0)
        else:
            cors.append(float(np.corrcoef(a, b)[0, 1]))
    cors = np.array(cors)
    return {"mean_activation_corr": float(cors.mean()),
            "median_activation_corr": float(np.median(cors)),
            "frac_corr_gt_0.5": float((cors > 0.5).mean())}


def top_example_overlap(codes_A: np.ndarray, codes_B: np.ndarray,
                        top_n: int = 20, n_features: int = 50) -> Dict:
    """Jaccard overlap of top-N activating example sets for first n_features."""
    n = min(n_features, codes_A.shape[1], codes_B.shape[1])
    jacs = []
    for j in range(n):
        sa = set(np.argsort(codes_A[:, j])[::-1][:top_n].tolist())
        sb = set(np.argsort(codes_B[:, j])[::-1][:top_n].tolist())
        jacs.append(len(sa & sb) / max(len(sa | sb), 1))
    jacs = np.array(jacs)
    return {"mean_top_overlap": float(jacs.mean()),
            "median_top_overlap": float(np.median(jacs))}
