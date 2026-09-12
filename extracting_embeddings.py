"""
extracting_embeddings.py  —  Step 1 of C-SPINE pipeline
========================================================
Loads SST-2 and AG News, runs them through BERT (or MiniLM),
and saves (N, embed_dim) float32 arrays to disk.

Run ONCE before anything else:
    python extracting_embeddings.py

After this, the C-SPINE autoencoder never touches BERT — it only
reads .npy files, which is orders of magnitude faster per epoch.

Estimated wall-clock time (both datasets combined)
───────────────────────────────────────────────────
  bert-base-uncased  + GPU (T4/A100) :  ~25–35 min
  bert-base-uncased  + CPU            :  ~2–4 h   ← switch to MiniLM!
  all-MiniLM-L6-v2  + GPU            :  ~8–12 min
  all-MiniLM-L6-v2  + CPU            :  ~30–50 min

Datasets produced
─────────────────
  SST-2    train 67 349 / val 872   — binary sentiment (0=neg, 1=pos)
  AG News  train 120 000 / test 7 600 — 4-class topic
             0=World  1=Sports  2=Business  3=Sci/Tech
"""

import os
import sys
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from datasets import load_dataset
from tqdm import tqdm

from config import Config as C


# ─────────────────────────────────────────────────────────────────────────────
# Pooling strategies
# ─────────────────────────────────────────────────────────────────────────────

def cls_pool(last_hidden_state: torch.Tensor) -> torch.Tensor:
    """
    Return the [CLS] token embedding (position 0).
    Standard choice for bert-base-uncased; reflects global sentence meaning
    as learned during NSP pre-training.
    """
    return last_hidden_state[:, 0, :]   # (B, D)


def mean_pool(last_hidden_state: torch.Tensor,
              attention_mask: torch.Tensor) -> torch.Tensor:
    """
    Masked mean over non-padding tokens.
    Better for sentence-transformer variants (MiniLM, RoBERTa-based) that
    were fine-tuned with mean pooling rather than CLS.
    """
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    summed = torch.sum(last_hidden_state * mask, dim=1)
    count  = torch.clamp(mask.sum(dim=1), min=1e-9)
    return summed / count   # (B, D)


def pick_pooling(model_name: str) -> str:
    """Infer pooling strategy from model name."""
    lower = model_name.lower()
    if any(k in lower for k in ('minilm', 'roberta', 'mpnet', 'distilbert')):
        return 'mean'
    return 'cls'   # bert-base-uncased and most BERT variants


# ─────────────────────────────────────────────────────────────────────────────
# Core extraction
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def extract_embeddings(texts: list,
                       tokenizer,
                       model,
                       pooling: str = 'cls') -> np.ndarray:
    """
    Encode `texts` in mini-batches.

    Parameters
    ----------
    texts   : flat list of raw strings
    pooling : 'cls' or 'mean'

    Returns
    -------
    np.ndarray of shape (N, embed_dim), dtype float32
    """
    all_embs = []

    for start in tqdm(range(0, len(texts), C.BATCH_SIZE), unit='batch', leave=False):
        batch = texts[start : start + C.BATCH_SIZE]

        enc = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=C.MAX_LENGTH,
            return_tensors='pt',
        )
        enc = {k: v.to(C.DEVICE) for k, v in enc.items()}

        out = model(**enc)
        lhs = out.last_hidden_state  # (B, T, D)

        emb = (cls_pool(lhs) if pooling == 'cls'
               else mean_pool(lhs, enc['attention_mask']))

        all_embs.append(emb.cpu().float().numpy())

    return np.concatenate(all_embs, axis=0)


# ─────────────────────────────────────────────────────────────────────────────
# I/O helpers
# ─────────────────────────────────────────────────────────────────────────────

def save_split(dataset_name: str, split: str,
               embs: np.ndarray, labels: np.ndarray) -> None:
    """Persist embeddings and labels as .npy files."""
    np.save(os.path.join(C.EMBED_DIR, f'{dataset_name}_{split}_embs.npy'),   embs)
    np.save(os.path.join(C.EMBED_DIR, f'{dataset_name}_{split}_labels.npy'), labels)
    print(f'  ✓  {dataset_name}/{split}:  embs {embs.shape}  |  '
          f'labels {labels.shape}  |  dtype {embs.dtype}')


def already_cached(dataset_name: str, split: str) -> bool:
    """True if this split was already extracted and saved."""
    path = os.path.join(C.EMBED_DIR, f'{dataset_name}_{split}_embs.npy')
    return os.path.exists(path)


def maybe_cap(texts: list, labels: np.ndarray,
              split: str) -> tuple:
    """
    Optionally cap training set to MAX_TRAIN_SAMPLES.
    Eval / test splits are never capped — we want full evaluation.
    """
    if (split == 'train'
            and C.MAX_TRAIN_SAMPLES is not None
            and len(texts) > C.MAX_TRAIN_SAMPLES):
        idx    = np.random.choice(len(texts), C.MAX_TRAIN_SAMPLES, replace=False)
        texts  = [texts[i] for i in idx]
        labels = labels[idx]
        print(f'  ↳ capped training set to {C.MAX_TRAIN_SAMPLES:,} samples')
    return texts, labels


def verify_embedding_dim(embs: np.ndarray) -> None:
    """Sanity-check that extracted dimension matches config."""
    if embs.shape[1] != C.EMBED_DIM:
        raise ValueError(
            f'Embedding dim mismatch: got {embs.shape[1]}, '
            f'expected {C.EMBED_DIM} (check EMBED_DIM in config.py).'
        )


# ─────────────────────────────────────────────────────────────────────────────
# Dataset-specific loaders
# ─────────────────────────────────────────────────────────────────────────────

def process_sst2(tokenizer, model, pooling: str) -> None:
    """
    SST-2 (Stanford Sentiment Treebank, binary).
    Source: glue/sst2 on HuggingFace.
    Note: official test labels are withheld — we use validation as test.
    """
    print('\n─── SST-2 ──────────────────────────────────────────')
    ds = load_dataset('nyu-mll/glue', 'sst2')

    splits = [
        ('train', 'train',      'sentence'),
        ('val',   'validation', 'sentence'),
    ]

    for out_split, hf_split, text_col in splits:
        if already_cached('sst2', out_split):
            print(f'  [skip] sst2/{out_split} already cached.')
            continue

        texts  = ds[hf_split][text_col]
        labels = np.array(ds[hf_split]['label'], dtype=np.int64)
        texts, labels = maybe_cap(texts, labels, out_split)

        print(f'  Encoding sst2/{out_split} ({len(texts):,} samples) …')
        embs = extract_embeddings(texts, tokenizer, model, pooling)
        verify_embedding_dim(embs)
        save_split('sst2', out_split, embs, labels)


def process_agnews(tokenizer, model, pooling: str) -> None:
    """
    AG News (4-class news topic classification).
    Source: ag_news on HuggingFace.
    Labels: 0=World, 1=Sports, 2=Business, 3=Sci/Tech
    """
    print('\n─── AG News ─────────────────────────────────────────')
    ds = load_dataset('fancyzhx/ag_news')

    splits = [
        ('train', 'train', 'text'),
        ('test',  'test',  'text'),
    ]

    for out_split, hf_split, text_col in splits:
        if already_cached('agnews', out_split):
            print(f'  [skip] agnews/{out_split} already cached.')
            continue

        texts  = ds[hf_split][text_col]
        labels = np.array(ds[hf_split]['label'], dtype=np.int64)
        texts, labels = maybe_cap(texts, labels, out_split)

        print(f'  Encoding agnews/{out_split} ({len(texts):,} samples) …')
        embs = extract_embeddings(texts, tokenizer, model, pooling)
        verify_embedding_dim(embs)
        save_split('agnews', out_split, embs, labels)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    np.random.seed(42)
    torch.manual_seed(42)

    os.makedirs(C.EMBED_DIR, exist_ok=True)
    os.makedirs(C.MODEL_DIR, exist_ok=True)

    pooling = pick_pooling(C.BERT_MODEL)

    print('=' * 55)
    print(f'  Model   : {C.BERT_MODEL}')
    print(f'  Pooling : {pooling}  ({C.EMBED_DIM}-dim)')
    print(f'  Device  : {C.DEVICE}')
    if C.MAX_TRAIN_SAMPLES:
        print(f'  Cap     : {C.MAX_TRAIN_SAMPLES:,} train samples per dataset')
    print('=' * 55)

    # ── Load model once — shared for all datasets ──────────────────────────
    print('\nLoading tokenizer + model …')
    tokenizer = AutoTokenizer.from_pretrained(C.BERT_MODEL)
    model     = AutoModel.from_pretrained(C.BERT_MODEL)
    model.eval().to(C.DEVICE)
    n_params  = sum(p.numel() for p in model.parameters()) / 1e6
    print(f'Model loaded  ({n_params:.1f}M params)\n')

    # ── Extract ────────────────────────────────────────────────────────────
    process_sst2(tokenizer, model, pooling)
    process_agnews(tokenizer, model, pooling)

    # ── Summary ────────────────────────────────────────────────────────────
    print('\n' + '=' * 55)
    print('  All embeddings saved to:', C.EMBED_DIR)
    print()
    for fname in sorted(os.listdir(C.EMBED_DIR)):
        fpath = os.path.join(C.EMBED_DIR, fname)
        arr   = np.load(fpath, mmap_mode='r')
        mb    = arr.nbytes / 1024 / 1024
        print(f'    {fname:<40}  {str(arr.shape):<20}  {mb:.1f} MB')
    print()
    print('  Next step: python train_cspine.py')
    print('=' * 55)


if __name__ == '__main__':
    main()
