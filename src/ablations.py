"""src.ablations — scientifically meaningful ablation configs.

Representation: pooling (cls/mean), normalized vs raw, latent dims.
Sparsity mechanism: cspine vs topk vs batchtopk vs l1 vs dense_ae.
Loss: remove ASL / PSL / noise one at a time; MSE-only baseline.
Architecture: hidden dim, MLP capacity, activation (capped vs relu vs sigmoid).

Each helper returns ExperimentConfig lists; the notebook executes a feasible
subset under Colab budget and reports ALL outcomes (no cherry-picking).
"""
from __future__ import annotations

import copy
from typing import List


def representation_ablations(base) -> List:
    out = []
    for pooling in ("cls", "mean"):
        c = copy.deepcopy(base)
        c.experiment_name = f"{base.experiment_name}_pool-{pooling}"
        c.pooling = pooling
        out.append(c)
    for norm in (False, True):
        c = copy.deepcopy(base)
        c.experiment_name = f"{base.experiment_name}_norm-{int(norm)}"
        c.normalize_inputs = norm
        out.append(c)
    for h in (256, 512, 1024, 2048):
        c = copy.deepcopy(base)
        c.experiment_name = f"{base.experiment_name}_hidden{h}"
        c.hidden_dim = h
        out.append(c)
    return out


def sparsity_mechanism_ablations(base) -> List:
    out = []
    # cspine baseline
    c0 = copy.deepcopy(base)
    c0.experiment_name = f"{base.experiment_name}_mech-cspine"
    c0.sparsity_mechanism = "cspine"
    out.append(c0)
    # topk with matched target L0 (~64/1024 ≈ 94% sparse, near paper regime)
    for k in (32, 64, 128):
        c = copy.deepcopy(base)
        c.experiment_name = f"{base.experiment_name}_mech-topk{k}"
        c.sparsity_mechanism = "topk"
        c.topk_k = k
        c.lambda_asl = 0.0
        c.lambda_psl = 0.0
        out.append(c)
    for k in (64,):
        c = copy.deepcopy(base)
        c.experiment_name = f"{base.experiment_name}_mech-batchtopk{k}"
        c.sparsity_mechanism = "batchtopk"
        c.topk_k = k
        c.lambda_asl = 0.0
        c.lambda_psl = 0.0
        out.append(c)
    # dense AE (no sparsity)
    c = copy.deepcopy(base)
    c.experiment_name = f"{base.experiment_name}_mech-denseae"
    c.sparsity_mechanism = "dense_ae"
    c.noise_std = 0.0
    c.lambda_asl = 0.0
    c.lambda_psl = 0.0
    out.append(c)
    return out


def loss_ablations(base) -> List:
    out = []
    variants = [
        ("full", dict(use_asl=True, use_psl=True, use_l1_instead=False, noise_std=None)),
        ("no-asl", dict(use_asl=False, use_psl=True, use_l1_instead=False, noise_std=None)),
        ("no-psl", dict(use_asl=True, use_psl=False, use_l1_instead=False, noise_std=None)),
        ("no-noise", dict(use_asl=True, use_psl=True, use_l1_instead=False, noise_std=0.0)),
        ("mse-only", dict(use_asl=False, use_psl=False, use_l1_instead=False, noise_std=0.0)),
        ("l1-instead", dict(use_asl=True, use_psl=True, use_l1_instead=True, noise_std=None)),
    ]
    for name, kw in variants:
        c = copy.deepcopy(base)
        c.experiment_name = f"{base.experiment_name}_loss-{name}"
        c.use_asl = kw["use_asl"]
        c.use_psl = kw["use_psl"]
        c.use_l1_instead = kw["use_l1_instead"]
        if kw["noise_std"] is not None:
            c.noise_std = kw["noise_std"]
        out.append(c)
    return out


def architectural_ablations(base) -> List:
    out = []
    for h in (256, 1024, 2048):
        c = copy.deepcopy(base)
        c.experiment_name = f"{base.experiment_name}_arch-hidden{h}"
        c.hidden_dim = h
        c.sparsity_mechanism = "cspine"
        out.append(c)
    # capacity: mlp encoder vs linear
    c = copy.deepcopy(base)
    c.experiment_name = f"{base.experiment_name}_arch-mlp"
    c.sparsity_mechanism = "mlp"
    out.append(c)
    return out
