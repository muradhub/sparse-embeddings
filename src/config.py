"""src.config — configuration objects for reproducible experiments.

Keeps backward compatibility with legacy root ``config.py`` (Config class)
while adding explicit, serialisable experiment configs required for a
formal ML research paper: seeds, splits, encoder, pooling, latent dim,
training hyperparams, sparsity settings, checkpoint, metrics.

Every experiment must be reproducible from a config JSON file stored in
``configs/``. Use :func:`save_config` / :func:`load_config`.
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ExperimentConfig:
    """Single reproducible experiment specification."""

    experiment_name: str = "cspine_base"
    seed: int = 42
    dataset: str = "sst2"  # sst2 | agnews | imdb | trec | scicite | dbpedia14
    encoder_name: str = "bert-base-uncased"
    pooling: str = "cls"  # cls | mean
    normalize_inputs: bool = False  # L2-normalize embeddings before SAE?
    # NOTE: legacy code does NOT normalize despite paper claiming it does.
    # Default False reproduces legacy behaviour; True tests paper claim.
    embed_dim: int = 768
    hidden_dim: int = 1024
    noise_std: float = 0.1
    lambda_asl: float = 1.0
    lambda_psl: float = 1.0
    sparsity_mechanism: str = "cspine"  # cspine | topk | batchtopk | dense_ae | l1
    topk_k: int = 64  # active features per example when mechanism=topk
    lr: float = 1e-3
    batch_size: int = 64
    epochs: int = 30
    max_length: int = 128
    max_train_samples: Optional[int] = None
    # splits: train/val/test handling
    val_fraction: float = 0.1  # fraction of HF train held out as val
    checkpoint_metric: str = "val_mse"  # select best ckpt on VAL only
    # downstream probe
    probe_C: float = 1.0
    probe_max_iter: int = 1000
    # paths (relative to repo root; no hard-coded absolute paths)
    cache_dir: str = "./cache"
    output_dir: str = "./final_research"
    device: str = "auto"  # auto | cpu | cuda

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ExperimentConfig":
        known = {f.name for f in cls.__dataclass_fields__.values()}
        filt = {k: v for k, v in d.items() if k in known}
        return cls(**filt)


# Preset sweeps -------------------------------------------------------------

def sparsity_sweep_configs(base: ExperimentConfig) -> List[ExperimentConfig]:
    """Controlled sparsity sweep: low / medium / high / very_high.

    For the C-SPINE soft-penalty mechanism we vary lambda_asl; for TopK
    variants the caller should instead vary topk_k (see ablations module).
    Values chosen to span ~50% .. ~97% sparsity on 1024-dim codes based on
    pilot behaviour (sparsity rises monotonically with lambda_asl).
    """
    levels = [
        ("low", 0.1),
        ("medium", 1.0),
        ("high", 3.0),
        ("very_high", 8.0),
    ]
    out = []
    for name, lam in levels:
        import copy
        c = copy.deepcopy(base)
        c.experiment_name = f"{base.experiment_name}_sparsity_{name}"
        c.lambda_asl = lam
        out.append(c)
    return out


def latent_sweep_configs(base: ExperimentConfig) -> List[ExperimentConfig]:
    """Latent-dimension / overcompleteness ablation."""
    out = []
    for h in (256, 512, 1024, 2048):
        import copy
        c = copy.deepcopy(base)
        c.experiment_name = f"{base.experiment_name}_hidden{h}"
        c.hidden_dim = h
        out.append(c)
    return out


def seed_configs(base: ExperimentConfig, seeds=(42, 43, 44)) -> List[ExperimentConfig]:
    import copy
    out = []
    for s in seeds:
        c = copy.deepcopy(base)
        c.experiment_name = f"{base.experiment_name}_seed{s}"
        c.seed = s
        out.append(c)
    return out


# IO ------------------------------------------------------------------------

def save_config(cfg: ExperimentConfig, path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cfg.to_dict(), f, indent=2, sort_keys=True)


def load_config(path: str) -> ExperimentConfig:
    with open(path, "r", encoding="utf-8") as f:
        return ExperimentConfig.from_dict(json.load(f))
