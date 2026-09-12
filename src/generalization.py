"""src.generalization — cross-domain and cross-encoder studies.

Cross-domain: train SAE on domain A, freeze it, evaluate recon/sparsity/
downstream/feature-utilization on domain B (no retraining). Asks whether
sparse directions are reusable or dataset-specific.

Encoder generalizability: same C-SPINE framework applied to different
dense encoders (bert-base, MiniLM, DistilBERT). Compares sparsity/recon/
downstream under matched hidden dims (embed_dim differs per encoder).
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np

ENCODERS = {
    "bert-base-uncased": {"embed_dim": 768, "pooling": "cls"},
    "sentence-transformers/all-MiniLM-L6-v2": {"embed_dim": 384, "pooling": "mean"},
    "distilbert-base-uncased": {"embed_dim": 768, "pooling": "cls"},
}


def cross_domain_matrix(datasets: List[str]) -> List[Dict[str, str]]:
    """All ordered (train_domain, test_domain) pairs including diagonals."""
    pairs = []
    for tr in datasets:
        for te in datasets:
            pairs.append({"train_domain": tr, "test_domain": te,
                          "is_transfer": tr != te})
    return pairs


def transfer_summary(recon_metrics, spar_metrics, downstream_sparse_acc) -> Dict:
    return {"recon_mse": recon_metrics.get("mse"),
            "recon_cosine": recon_metrics.get("cosine_sim"),
            "sparsity_ratio": spar_metrics.get("sparsity_ratio"),
            "mean_l0": spar_metrics.get("mean_l0"),
            "dead_fraction": spar_metrics.get("dead_fraction"),
            "sparse_probe_accuracy": downstream_sparse_acc}
