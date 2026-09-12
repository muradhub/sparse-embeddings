"""
train_cspine.py  —  Step 2 of C-SPINE pipeline
====================================================
Trains the Contextual-SPINE sparse autoencoder on pre-cached
BERT/MiniLM embeddings.  BERT is never loaded here — we only read
the .npy files produced by extracting_embeddings.py.

Architecture
────────────
  Encoder : Linear(embed_dim → hidden_dim)  →  CappedReLU(0, 1)
  Decoder : Linear(hidden_dim → embed_dim)   (no activation)

Training objective  (DAE + sparsity)
──────────────────────────────────────
  noisy = clean + N(0, σ²)
  z     = Encoder(noisy)          ← sparse codes in [0, 1]
  recon = Decoder(z)

  MSE  = ||recon - clean||²       ← reconstruction vs *clean* target
  ASL  = mean(z)                  ← Average Sparsity Loss (pushes acts → 0)
  PSL  = mean(max_over_batch(z))  ← Lifetime Sparsity Loss (no dead neurons)
  Loss = MSE + λ₁·ASL - λ₂·PSL   ← note: PSL is *subtracted* (maximise peak)

The PSL sign convention follows SPINE (Subramanian et al., 2018):
  - ASL minimised → most units off per sample
  - PSL maximised → every unit fires *at least sometimes* across the batch

Outputs (saved to C.MODEL_DIR)
───────────────────────────────
  cspine_model.pt          — full model state dict  (best val MSE)
  cspine_training_log.csv  — per-epoch metrics for plotting

Usage
─────
  python train_cspine.py

Run extracting_embeddings.py first.
"""

import os
import csv
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from config import Config as C


# ─────────────────────────────────────────────────────────────────────────────
# 1.  CappedReLU  — keeps activations in [0, 1]
# ─────────────────────────────────────────────────────────────────────────────

class CappedReLU(nn.Module):
    """
    f(x) = clamp(x, 0, 1)

    Why not plain ReLU?
      Plain ReLU gives unbounded activations — PSL would just push one
      neuron to blow up.  Capping to [0,1] makes every sparse code
      directly interpretable as a soft membership weight.
    """
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.clamp(0.0, 1.0)


# ─────────────────────────────────────────────────────────────────────────────
# 2.  C-SPINE model
# ─────────────────────────────────────────────────────────────────────────────

class CSPINE(nn.Module):
    """
    Sparse autoencoder for contextual (BERT-style) embeddings.

    Parameters
    ----------
    embed_dim  : dimensionality of the frozen BERT embeddings
    hidden_dim : number of dictionary atoms (use > embed_dim for overcomplete)
    noise_std  : σ for Gaussian DAE noise injected at training time
    """

    def __init__(self,
                 embed_dim: int  = C.EMBED_DIM,
                 hidden_dim: int = C.HIDDEN_DIM,
                 noise_std: float = C.NOISE_STD):
        super().__init__()
        self.noise_std  = noise_std
        self.embed_dim  = embed_dim
        self.hidden_dim = hidden_dim

        # Encoder: one linear layer + CappedReLU
        self.encoder = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            CappedReLU(),
        )

        # Decoder: linear reconstruction (no activation)
        self.decoder = nn.Linear(hidden_dim, embed_dim)

    # ── forward ──────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> tuple:
        """
        Parameters
        ----------
        x : (B, embed_dim)  — clean BERT embeddings

        Returns
        -------
        z     : (B, hidden_dim)  — sparse codes in [0, 1]
        recon : (B, embed_dim)   — reconstructed embeddings
        """
        # Add noise only during training (DAE)
        if self.training and self.noise_std > 0.0:
            noise = torch.randn_like(x) * self.noise_std
            x_noisy = x + noise
        else:
            x_noisy = x

        z     = self.encoder(x_noisy)
        recon = self.decoder(z)
        return z, recon

    # ── convenience: encode only (used at eval/inference) ────────────────────

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Return sparse codes for clean embeddings (no noise, no grad)."""
        self.eval()
        return self.encoder(x)


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Loss components
# ─────────────────────────────────────────────────────────────────────────────

def mse_loss(recon: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean squared error between reconstruction and *clean* target."""
    return nn.functional.mse_loss(recon, target)


def average_sparsity_loss(z: torch.Tensor) -> torch.Tensor:
    """
    ASL = mean(z)  over (batch × hidden) dimensions.
    Minimising this pushes most activations toward 0.
    """
    return z.mean()


def peak_sparsity_loss(z: torch.Tensor) -> torch.Tensor:
    """
    PSL = mean( max_over_batch(z) )  — average of per-unit batch peaks.
    Maximising this (i.e. *subtracting* λ₂·PSL from total loss) ensures
    every dictionary atom fires for at least some samples,
    preventing dead / collapsed neurons.
    """
    return z.max(dim=0).values.mean()   # (hidden_dim,) → scalar


def total_loss(recon: torch.Tensor,
               target: torch.Tensor,
               z: torch.Tensor,
               lambda_asl: float = C.LAMBDA_ASL,
               lambda_psl: float = C.LAMBDA_PSL) -> tuple:
    """
    L = MSE + λ₁·ASL - λ₂·PSL

    Returns (total, mse, asl, psl) as scalar tensors for logging.
    """
    mse = mse_loss(recon, target)
    asl = average_sparsity_loss(z)
    psl = peak_sparsity_loss(z)
    loss = mse + lambda_asl * asl - lambda_psl * psl
    return loss, mse, asl, psl


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_embeddings(dataset: str, split: str) -> tuple:
    """
    Load pre-cached embeddings from disk.

    Returns
    -------
    embs   : torch.FloatTensor  (N, embed_dim)
    labels : torch.LongTensor   (N,)
    """
    emb_path   = os.path.join(C.EMBED_DIR, f'{dataset}_{split}_embs.npy')
    label_path = os.path.join(C.EMBED_DIR, f'{dataset}_{split}_labels.npy')

    if not os.path.exists(emb_path):
        raise FileNotFoundError(
            f'Embeddings not found: {emb_path}\n'
            f'Run extracting_embeddings.py first.'
        )

    embs   = torch.from_numpy(np.load(emb_path)).float()
    labels = torch.from_numpy(np.load(label_path)).long()
    return embs, labels


def make_loader(embs: torch.Tensor,
                labels: torch.Tensor,
                shuffle: bool = True) -> DataLoader:
    """Wrap tensors in a DataLoader."""
    dataset = TensorDataset(embs, labels)
    return DataLoader(
        dataset,
        batch_size=C.BATCH_SIZE,
        shuffle=shuffle,
        num_workers=0,          # npy already in RAM; extra workers add overhead
        pin_memory=(C.DEVICE == 'cuda'),
    )


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Training & validation loops
# ─────────────────────────────────────────────────────────────────────────────

def train_one_epoch(model: CSPINE,
                    loader: DataLoader,
                    optimizer: torch.optim.Optimizer,
                    device: str) -> dict:
    """
    One training epoch.

    Returns a dict of mean metrics: {loss, mse, asl, psl, sparsity}
    """
    model.train()
    totals = dict(loss=0., mse=0., asl=0., psl=0., sparsity=0.)
    n_batches = 0

    for embs, _ in loader:           # labels not used during AE training
        embs = embs.to(device)

        optimizer.zero_grad()
        z, recon = model(embs)

        loss, mse, asl, psl = total_loss(recon, embs, z)
        loss.backward()
        optimizer.step()

        # sparsity: fraction of hidden units that are exactly 0
        with torch.no_grad():
            sparsity = (z == 0.0).float().mean().item()

        totals['loss']     += loss.item()
        totals['mse']      += mse.item()
        totals['asl']      += asl.item()
        totals['psl']      += psl.item()
        totals['sparsity'] += sparsity
        n_batches += 1

    return {k: v / n_batches for k, v in totals.items()}


@torch.no_grad()
def evaluate(model: CSPINE,
             loader: DataLoader,
             device: str) -> dict:
    """
    Validation / test pass.  No noise added (model.eval() is set inside encode).
    Returns dict of mean metrics: {loss, mse, asl, psl, sparsity}
    """
    model.eval()
    totals = dict(loss=0., mse=0., asl=0., psl=0., sparsity=0.)
    n_batches = 0

    for embs, _ in loader:
        embs = embs.to(device)
        z, recon = model(embs)        # eval mode → no noise

        loss, mse, asl, psl = total_loss(recon, embs, z)
        sparsity = (z == 0.0).float().mean().item()

        totals['loss']     += loss.item()
        totals['mse']      += mse.item()
        totals['asl']      += asl.item()
        totals['psl']      += psl.item()
        totals['sparsity'] += sparsity
        n_batches += 1

    return {k: v / n_batches for k, v in totals.items()}


# ─────────────────────────────────────────────────────────────────────────────
# 6.  Main training loop
# ─────────────────────────────────────────────────────────────────────────────

def train(dataset: str = 'sst2') -> CSPINE:
    """
    Full training loop for one dataset.

    Parameters
    ----------
    dataset : 'sst2' or 'agnews'

    Returns
    -------
    Trained CSPINE model (best checkpoint by val MSE).
    """
    device = C.DEVICE

    # ── Load data ─────────────────────────────────────────────────────────────
    val_split = 'val' if dataset == 'sst2' else 'test'

    print(f'\nLoading {dataset} embeddings …')
    train_embs, train_labels = load_embeddings(dataset, 'train')
    val_embs,   val_labels   = load_embeddings(dataset, val_split)

    print(f'  Train : {train_embs.shape}   Val : {val_embs.shape}')

    train_loader = make_loader(train_embs, train_labels, shuffle=True)
    val_loader   = make_loader(val_embs,   val_labels,   shuffle=False)

    # ── Model, optimiser ──────────────────────────────────────────────────────
    model     = CSPINE().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=C.LR)

    total_params = sum(p.numel() for p in model.parameters())
    print(f'\nC-SPINE  |  embed={C.EMBED_DIM}  hidden={C.HIDDEN_DIM}  '
          f'params={total_params:,}')
    print(f'λ_ASL={C.LAMBDA_ASL}  λ_PSL={C.LAMBDA_PSL}  '
          f'noise_σ={C.NOISE_STD}  lr={C.LR}  epochs={C.EPOCHS}')

    # ── CSV logger ─────────────────────────────────────────────────────────────
    log_path = os.path.join(C.MODEL_DIR, f'{dataset}_training_log.csv')
    csv_cols  = ['epoch',
                 'tr_loss', 'tr_mse', 'tr_asl', 'tr_psl', 'tr_sparsity',
                 'val_loss','val_mse','val_asl','val_psl','val_sparsity']
    csv_file  = open(log_path, 'w', newline='')
    writer    = csv.DictWriter(csv_file, fieldnames=csv_cols)
    writer.writeheader()

    # ── Training loop ─────────────────────────────────────────────────────────
    best_val_mse   = float('inf')
    best_ckpt_path = os.path.join(C.MODEL_DIR, f'{dataset}_cspine_best.pt')

    print(f'\n{"Epoch":>6} {"Tr-Loss":>9} {"Tr-MSE":>9} '
          f'{"Val-MSE":>9} {"Val-Spar%":>10}')
    print('─' * 55)

    for epoch in range(1, C.EPOCHS + 1):
        t0 = time.time()

        tr  = train_one_epoch(model, train_loader, optimizer, device)
        val = evaluate(model, val_loader, device)

        # Save best checkpoint
        if val['mse'] < best_val_mse:
            best_val_mse = val['mse']
            torch.save({
                'epoch':      epoch,
                'model_state_dict':     model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_mse':    best_val_mse,
                'config': {
                    'embed_dim':  C.EMBED_DIM,
                    'hidden_dim': C.HIDDEN_DIM,
                    'noise_std':  C.NOISE_STD,
                },
            }, best_ckpt_path)
            flag = ' ★'
        else:
            flag = ''

        elapsed = time.time() - t0
        print(f'{epoch:>6}  {tr["loss"]:>9.5f}  {tr["mse"]:>9.5f}  '
              f'{val["mse"]:>9.5f}  {val["sparsity"]*100:>9.1f}%'
              f'  ({elapsed:.1f}s){flag}')

        # Log to CSV
        row = {'epoch': epoch}
        for k, v in tr.items():
            row[f'tr_{k}'] = round(v, 6)
        for k, v in val.items():
            row[f'val_{k}'] = round(v, 6)
        writer.writerow(row)
        csv_file.flush()

    csv_file.close()

    # ── Load best model for return ─────────────────────────────────────────────
    ckpt = torch.load(best_ckpt_path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    print(f'\nBest val MSE: {best_val_mse:.6f}  (epoch {ckpt["epoch"]})')
    print(f'Checkpoint  : {best_ckpt_path}')
    print(f'Training log: {log_path}')

    return model


# ─────────────────────────────────────────────────────────────────────────────
# 7.  Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    torch.manual_seed(42)
    np.random.seed(42)

    os.makedirs(C.MODEL_DIR, exist_ok=True)

    print('=' * 55)
    print('  C-SPINE Autoencoder Training')
    print(f'  Device : {C.DEVICE}')
    print('=' * 55)

    # Train on SST-2 first (smaller, faster — good sanity check)
    train('sst2')

    # Then AG News
    train('agnews')

    print('\n' + '=' * 55)
    print('  Training complete.')
    print('  Next step: python evaluate.py')
    print('=' * 55)


if __name__ == '__main__':
    main()
