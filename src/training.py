"""src.training — training / validation loops with VAL-only checkpointing.

Fixes vs legacy:
  - explicit train/val/test separation (val used for checkpointing, test never
    touched until final eval),
  - multi-seed support via `seed_everything`,
  - ablation-aware loss (ASL/PSL/L1 toggles),
  - per-epoch CSV logging compatible with legacy plot code,
  - returns history dict for plotting.
"""
from __future__ import annotations

import csv
import os
import random
import time
from typing import Dict, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from .losses import total_loss, total_loss_ablated
from .sparse_model import build_model


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(pref: str = "auto") -> str:
    if pref == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return pref


def train_one_epoch(model, loader: DataLoader, optimizer, device: str,
                    lambda_asl: float = 1.0, lambda_psl: float = 1.0,
                    use_asl: bool = True, use_psl: bool = True,
                    use_l1_instead: bool = False) -> Dict[str, float]:
    model.train()
    totals = dict(loss=0.0, mse=0.0, asl=0.0, psl=0.0, sparsity=0.0)
    n = 0
    for embs, _ in loader:
        embs = embs.to(device)
        optimizer.zero_grad()
        z, recon = model(embs)
        loss, mse, asl, psl = total_loss_ablated(
            recon, embs, z, lambda_asl, lambda_psl, use_asl, use_psl, use_l1_instead)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            sparsity = (z == 0.0).float().mean().item()
        for k, v in (("loss", loss.item()), ("mse", mse.item()),
                     ("asl", asl.item()), ("psl", psl.item()), ("sparsity", sparsity)):
            totals[k] += v
        n += 1
    return {k: v / max(n, 1) for k, v in totals.items()}


@torch.no_grad()
def evaluate_loss(model, loader: DataLoader, device: str,
                  lambda_asl: float = 1.0, lambda_psl: float = 1.0) -> Dict[str, float]:
    model.eval()
    totals = dict(loss=0.0, mse=0.0, asl=0.0, psl=0.0, sparsity=0.0)
    n = 0
    for embs, _ in loader:
        embs = embs.to(device)
        z, recon = model(embs)
        loss, mse, asl, psl = total_loss(recon, embs, z, lambda_asl, lambda_psl)
        sparsity = (z == 0.0).float().mean().item()
        for k, v in (("loss", loss.item()), ("mse", mse.item()),
                     ("asl", asl.item()), ("psl", psl.item()), ("sparsity", sparsity)):
            totals[k] += v
        n += 1
    return {k: v / max(n, 1) for k, v in totals.items()}


def train_model(cfg, train_loader: DataLoader, val_loader: DataLoader,
                model_dir: str) -> tuple:
    """Full training loop for one ExperimentConfig.

    Saves best checkpoint by VAL MSE to {model_dir}/{dataset}_cspine_best.pt
    (or {experiment_name}.pt when experiment_name differs) plus CSV log.
    Returns (model, history, best_val_mse).
    """
    from .config import ExperimentConfig  # local import to avoid cycle
    device = resolve_device(getattr(cfg, "device", "auto"))
    seed_everything(getattr(cfg, "seed", 42))
    model = build_model(cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    os.makedirs(model_dir, exist_ok=True)
    tag = getattr(cfg, "experiment_name", getattr(cfg, "dataset", "exp"))
    # keep legacy filenames for base sst2/agnews runs, unique names otherwise
    if tag in ("sst2", "agnews") or tag.startswith("cspine"):
        ckpt_name = f"{cfg.dataset}_cspine_best.pt" if tag in ("sst2", "agnews") else f"{tag}.pt"
    else:
        ckpt_name = f"{tag}.pt"
    ckpt_path = os.path.join(model_dir, ckpt_name)
    log_path = os.path.join(model_dir, f"{tag}_training_log.csv")
    cols = ["epoch", "tr_loss", "tr_mse", "tr_asl", "tr_psl", "tr_sparsity",
            "val_loss", "val_mse", "val_asl", "val_psl", "val_sparsity"]
    use_asl = not (getattr(cfg, "sparsity_mechanism", "cspine") in ("dense_ae",))
    # loss-ablation flags may be attached to cfg by ablations module
    use_asl_flag = getattr(cfg, "use_asl", True)
    use_psl_flag = getattr(cfg, "use_psl", True)
    use_l1 = getattr(cfg, "use_l1_instead", False)
    if getattr(cfg, "sparsity_mechanism", "cspine") == "dense_ae":
        use_asl_flag, use_psl_flag, use_l1 = False, False, False

    history = []
    best_val_mse = float("inf")
    with open(log_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for epoch in range(1, int(cfg.epochs) + 1):
            t0 = time.time()
            tr = train_one_epoch(model, train_loader, optimizer, device,
                                 cfg.lambda_asl, cfg.lambda_psl,
                                 use_asl_flag, use_psl_flag, use_l1)
            va = evaluate_loss(model, val_loader, device, cfg.lambda_asl, cfg.lambda_psl)
            if va["mse"] < best_val_mse:
                best_val_mse = va["mse"]
                torch.save({"epoch": epoch,
                            "model_state_dict": model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "val_mse": best_val_mse,
                            "config": {"embed_dim": cfg.embed_dim,
                                       "hidden_dim": cfg.hidden_dim,
                                       "noise_std": cfg.noise_std,
                                       "mechanism": getattr(cfg, "sparsity_mechanism", "cspine"),
                                       "topk_k": getattr(cfg, "topk_k", None),
                                       "seed": getattr(cfg, "seed", 42)}},
                           ckpt_path)
            row = {"epoch": epoch}
            for k, v in tr.items():
                row[f"tr_{k}"] = round(float(v), 6)
            for k, v in va.items():
                row[f"val_{k}"] = round(float(v), 6)
            w.writerow(row)
            f.flush()
            history.append(row)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    return model, history, best_val_mse
