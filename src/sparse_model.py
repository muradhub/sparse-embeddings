"""src.sparse_model — encoder/decoder architectures.

Contains:
  - CappedReLU (legacy, [0,1]-bounded, interpretable as soft membership)
  - CSPINE (legacy denoising sparse autoencoder, exact reproduction)
  - TopKSAE / BatchTopKSAE (sparsity-mechanism ablations)
  - DenseAE (no-sparsity baseline: same capacity, plain ReLU, MSE only)
  - MLPEncoder variants (capacity ablation)

All models expose ``encode(x) -> z`` and ``forward(x) -> (z, recon)``.
Noise is injected only in train() mode, matching legacy behaviour.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class CappedReLU(nn.Module):
    """f(x) = clamp(x, 0, 1). Prevents PSL from blowing up one neuron."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.clamp(0.0, 1.0)


class CSPINE(nn.Module):
    """Legacy C-SPINE: Linear -> CappedReLU encoder, Linear decoder."""

    def __init__(self, embed_dim: int = 768, hidden_dim: int = 1024,
                 noise_std: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.noise_std = noise_std
        self.encoder = nn.Sequential(nn.Linear(embed_dim, hidden_dim), CappedReLU())
        self.decoder = nn.Linear(hidden_dim, embed_dim)

    def forward(self, x: torch.Tensor):
        if self.training and self.noise_std > 0.0:
            x_noisy = x + torch.randn_like(x) * self.noise_std
        else:
            x_noisy = x
        z = self.encoder(x_noisy)
        return z, self.decoder(z)

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        self.eval()
        return self.encoder(x)


class TopKSAE(nn.Module):
    """Hard Top-K per example: keep k largest activations, zero rest.

    Uses ReLU pre-activation then per-row top-k masking. Deterministic L0=k.
    Capped to [0,1] after masking for comparability with C-SPINE codes.
    """

    def __init__(self, embed_dim: int = 768, hidden_dim: int = 1024,
                 noise_std: float = 0.1, k: int = 64):
        super().__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.noise_std = noise_std
        self.k = int(k)
        self.enc_linear = nn.Linear(embed_dim, hidden_dim)
        self.decoder = nn.Linear(hidden_dim, embed_dim)

    def _topk_mask(self, h: torch.Tensor) -> torch.Tensor:
        k = min(self.k, h.shape[1])
        if k <= 0:
            return torch.zeros_like(h)
        thresh = torch.topk(h, k=k, dim=1).values[:, -1:].detach()
        return (h >= thresh).float() * h.clamp(0.0, 1.0)

    def forward(self, x: torch.Tensor):
        x_noisy = x + torch.randn_like(x) * self.noise_std if (self.training and self.noise_std > 0) else x
        h = torch.relu(self.enc_linear(x_noisy))
        z = self._topk_mask(h)
        return z, self.decoder(z)

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        self.eval()
        h = torch.relu(self.enc_linear(x))
        return self._topk_mask(h)


class BatchTopKSAE(nn.Module):
    """BatchTopK (Gao et al. 2024 style): keep top k*B activations in batch.

    Allows variable L0 per example while fixing total batch sparsity — a
    useful middle ground between soft C-SPINE penalties and hard per-example TopK.
    """

    def __init__(self, embed_dim: int = 768, hidden_dim: int = 1024,
                 noise_std: float = 0.1, k: int = 64):
        super().__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.noise_std = noise_std
        self.k = int(k)
        self.enc_linear = nn.Linear(embed_dim, hidden_dim)
        self.decoder = nn.Linear(hidden_dim, embed_dim)

    def _batch_topk(self, h: torch.Tensor) -> torch.Tensor:
        B = h.shape[0]
        keep = min(self.k * B, h.numel())
        if keep <= 0:
            return torch.zeros_like(h)
        flat = h.clamp(min=0.0).reshape(-1)
        thresh = torch.topk(flat, k=keep).values[-1].detach()
        return (h >= thresh).float() * h.clamp(0.0, 1.0)

    def forward(self, x: torch.Tensor):
        x_noisy = x + torch.randn_like(x) * self.noise_std if (self.training and self.noise_std > 0) else x
        h = torch.relu(self.enc_linear(x_noisy))
        z = self._batch_topk(h)
        return z, self.decoder(z)

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        self.eval()
        h = torch.relu(self.enc_linear(x))
        return self._batch_topk(h)


class DenseAE(nn.Module):
    """Standard autoencoder baseline: same dims, ReLU (unbounded), MSE only.

    Answers: is C-SPINE providing something beyond ordinary compression?
    """

    def __init__(self, embed_dim: int = 768, hidden_dim: int = 1024,
                 noise_std: float = 0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.noise_std = noise_std
        self.encoder = nn.Sequential(nn.Linear(embed_dim, hidden_dim), nn.ReLU())
        self.decoder = nn.Linear(hidden_dim, embed_dim)

    def forward(self, x: torch.Tensor):
        x_noisy = x + torch.randn_like(x) * self.noise_std if (self.training and self.noise_std > 0) else x
        z = self.encoder(x_noisy)
        return z, self.decoder(z)

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        self.eval()
        return self.encoder(x)


class MLPCSPINE(nn.Module):
    """Capacity ablation: 2-layer encoder (Linear-ReLU-Linear-CappedReLU)."""

    def __init__(self, embed_dim: int = 768, hidden_dim: int = 1024,
                 noise_std: float = 0.1, width: int = 1024):
        super().__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.noise_std = noise_std
        self.encoder = nn.Sequential(
            nn.Linear(embed_dim, width), nn.ReLU(),
            nn.Linear(width, hidden_dim), CappedReLU(),
        )
        self.decoder = nn.Linear(hidden_dim, embed_dim)

    def forward(self, x: torch.Tensor):
        x_noisy = x + torch.randn_like(x) * self.noise_std if (self.training and self.noise_std > 0) else x
        z = self.encoder(x_noisy)
        return z, self.decoder(z)

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        self.eval()
        return self.encoder(x)


def build_model(cfg) -> nn.Module:
    """Factory from ExperimentConfig (or any object with same attrs)."""
    import warnings
    mech = getattr(cfg, "sparsity_mechanism", "cspine")
    ed, hd, ns = cfg.embed_dim, cfg.hidden_dim, cfg.noise_std
    if hd < ed:
        # Seen in the wild: MiniLM (D=384) with FAST HIDDEN=256 densified
        # instead of sparsifying. Overcompleteness (H>D) is a design
        # assumption of C-SPINE, so warn loudly rather than fail silently.
        warnings.warn(
            f"Undercomplete dictionary H={hd} < D={ed}: the C-SPINE sparsity "
            f"objective assumes overcompleteness (H>D). Expect weak/no "
            f"sparsification; raise hidden_dim.", UserWarning, stacklevel=2)
    if mech == "cspine":
        return CSPINE(ed, hd, ns)
    if mech == "topk":
        return TopKSAE(ed, hd, ns, k=getattr(cfg, "topk_k", 64))
    if mech == "batchtopk":
        return BatchTopKSAE(ed, hd, ns, k=getattr(cfg, "topk_k", 64))
    if mech == "dense_ae":
        return DenseAE(ed, hd, noise_std=ns)
    if mech == "mlp":
        return MLPCSPINE(ed, hd, ns)
    if mech == "l1":
        # same arch as CSPINE but trained with L1 (see losses.total_loss_ablated)
        return CSPINE(ed, hd, ns)
    raise ValueError(f"Unknown sparsity_mechanism={mech!r}")
