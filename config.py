"""
config.py  —  Central configuration for C-SPINE
All scripts import from here; change things in one place only.
"""

import os
import torch


class Config:
    # ── Embedding model ───────────────────────────────────────────────────────
    # Option A (default): bert-base-uncased  → 768-dim CLS pooling
    #   Needs a GPU; ~25-35 min for both datasets.
    # Option B (fallback if no GPU): sentence-transformers/all-MiniLM-L6-v2
    #   → 384-dim mean pooling; runs fine on CPU in ~30-50 min.
    #   If you switch, also change EMBED_DIM to 384.
    BERT_MODEL = 'bert-base-uncased'
    EMBED_DIM = 768       # ← change to 384 when using MiniLM

    # ── C-SPINE architecture ──────────────────────────────────────────────────
    HIDDEN_DIM = 1024      # overcomplete dictionary (> EMBED_DIM is key)
    NOISE_STD = 0.1       # Gaussian DAE noise σ; valid range [0.1, 0.3]
                           # higher = stronger denoising pressure on encoder

    # ── Loss weights ──────────────────────────────────────────────────────────
    LAMBDA_ASL = 1.0       # Average Sparsity Loss  (pushes mean activation → 0)
    LAMBDA_PSL = 1.0       # Peak / Lifetime Sparsity Loss (prevents dead neurons)

    # ── Training ──────────────────────────────────────────────────────────────
    BATCH_SIZE = 64
    LR = 1e-3      # Adam lr; reduce to 5e-4 if loss oscillates
    EPOCHS = 30

    # ── Tokeniser ─────────────────────────────────────────────────────────────
    MAX_LENGTH = 128       # subword tokens; covers >99 % of SST-2 & AG News

    # ── Dataset cap ───────────────────────────────────────────────────────────
    # None   → use full dataset (recommended for final results)
    # 10_000 → quick smoke-test (embedding step <5 min on CPU)
    MAX_TRAIN_SAMPLES = None

    # ── Paths ─────────────────────────────────────────────────────────────────
    CACHE_DIR = './cache'
    EMBED_DIR = os.path.join(CACHE_DIR, 'embeddings')
    MODEL_DIR = os.path.join(CACHE_DIR, 'models')

    # ── Device ────────────────────────────────────────────────────────────────
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
