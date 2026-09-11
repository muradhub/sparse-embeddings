"""src.losses — reconstruction + sparsity objectives.

Implements the exact legacy C-SPINE losses plus documented variants used
in ablations. All functions are pure tensor ops (no side effects) so they
can be unit-tested on synthetic data.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def mse_loss(recon: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean squared error vs *clean* target (denoising objective)."""
    return F.mse_loss(recon, target)


def average_sparsity_loss(z: torch.Tensor) -> torch.Tensor:
    """ASL = mean(z) over batch x hidden. Minimise -> most units off."""
    return z.mean()


def peak_sparsity_loss(z: torch.Tensor) -> torch.Tensor:
    """PSL = mean(max_over_batch(z)). Maximised (subtracted) to avoid dead units."""
    return z.max(dim=0).values.mean()


def l1_sparsity_loss(z: torch.Tensor) -> torch.Tensor:
    """L1 alternative: mean |z| (for ablation vs ASL/PSL)."""
    return z.abs().mean()


def total_loss(
    recon: torch.Tensor,
    target: torch.Tensor,
    z: torch.Tensor,
    lambda_asl: float = 1.0,
    lambda_psl: float = 1.0,
) -> tuple:
    """Legacy objective: L = MSE + l1*ASL - l2*PSL. Returns (total, mse, asl, psl)."""
    mse = mse_loss(recon, target)
    asl = average_sparsity_loss(z)
    psl = peak_sparsity_loss(z)
    loss = mse + lambda_asl * asl - lambda_psl * psl
    return loss, mse, asl, psl


def total_loss_ablated(
    recon: torch.Tensor,
    target: torch.Tensor,
    z: torch.Tensor,
    lambda_asl: float = 1.0,
    lambda_psl: float = 1.0,
    use_asl: bool = True,
    use_psl: bool = True,
    use_l1_instead: bool = False,
) -> tuple:
    """Ablation-friendly loss: toggle ASL / PSL / L1 independently.

    - use_asl=False  -> removes average-sparsity pressure
    - use_psl=False  -> removes lifetime-sparsity reward
    - use_l1_instead -> replaces ASL+PSL with plain L1 (tests whether the
      two-term SPINE objective matters beyond generic sparse coding)
    Returns (total, mse, asl, psl) where asl/psl are reported even when off
    (for logging) but do not contribute to `total` when disabled.
    """
    mse = mse_loss(recon, target)
    asl = average_sparsity_loss(z)
    psl = peak_sparsity_loss(z)
    if use_l1_instead:
        l1 = l1_sparsity_loss(z)
        total = mse + lambda_asl * l1
        return total, mse, asl, psl
    total = mse
    if use_asl:
        total = total + lambda_asl * asl
    if use_psl:
        total = total - lambda_psl * psl
    return total, mse, asl, psl
