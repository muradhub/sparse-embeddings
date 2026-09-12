"""
inspect_dimensions.py  —  Step 4 of C-SPINE pipeline
=========================================================
Qualitative analysis: for each dataset, finds the sparse dimensions
that specialise most strongly for each class, then retrieves the
top-N sentences that maximally activate those dimensions.

NO re-running needed. Loads from:
  - cache/embeddings/*.npy       (from extracting_embeddings.py)
  - cache/models/*_cspine_best.pt  (from train_cspine.py)

What this produces
──────────────────
  Console output:
    - Per-class top-3 specialised dimensions with their top-5 sentences
    - A "dimension purity score" showing how class-specific each dim is

  Saved files (all in cache/models/):
    - sst2_qualitative_table.txt      ← paste into your paper
    - agnews_qualitative_table.txt    ← paste into your paper
    - sst2_top_dimensions.csv         ← full data for all dims
    - agnews_top_dimensions.csv       ← full data for all dims

Usage
─────
  python 04_inspect_dimensions.py

Runs in under 2 minutes on CPU.
"""

import os
import csv
import numpy as np
import torch

from config import Config as C

# Import model class — handles the leading-digit filename issue
import importlib.util, sys

def _import_from_file(module_name, file_path):
    spec   = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

train_cspine_mod   = _import_from_file('train_cspine_mod', 'train_cspine.py')
CSPINE       = train_cspine_mod.CSPINE


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

LABEL_NAMES = {
    'sst2':   {0: 'Negative', 1: 'Positive'},
    'agnews': {0: 'World', 1: 'Sports', 2: 'Business', 3: 'Sci/Tech'},
}

# How many specialised dimensions to show per class
TOP_DIMS_PER_CLASS = 3

# How many sentences to show per dimension
TOP_SENTENCES = 5

# Raw sentence text files — we load these to show actual sentences
# These are pulled fresh from HuggingFace (same split as embeddings)
# If HuggingFace is unavailable, set LOAD_TEXTS = False and we show indices only
LOAD_TEXTS = True


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Load raw sentence texts
# ─────────────────────────────────────────────────────────────────────────────

def load_texts(dataset: str, split: str) -> list:
    """
    Load raw sentence strings for a dataset split.
    Returns a list of strings aligned with the .npy embedding rows.
    """
    if not LOAD_TEXTS:
        return None

    try:
        from datasets import load_dataset
    except ImportError:
        print('  [warn] datasets library not found — showing indices only.')
        return None

    try:
        if dataset == 'sst2':
            hf_split = 'validation' if split == 'val' else 'train'
            ds = load_dataset('nyu-mll/glue', 'sst2')
            texts = ds[hf_split]['sentence']
        else:  # agnews
            hf_split = 'test' if split == 'test' else 'train'
            ds = load_dataset('fancyzhx/ag_news')
            texts = ds[hf_split]['text']
        print(f'  Loaded {len(texts):,} raw texts for {dataset}/{split}')
        return list(texts)
    except Exception as e:
        print(f'  [warn] Could not load texts: {e}')
        return None


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Load model and embeddings
# ─────────────────────────────────────────────────────────────────────────────

def load_model(dataset: str) -> CSPINE:
    ckpt_path = os.path.join(C.MODEL_DIR, f'{dataset}_cspine_best.pt')
    ckpt = torch.load(ckpt_path, map_location='cpu')
    cfg  = ckpt.get('config', {})
    model = CSPINE(
        embed_dim  = cfg.get('embed_dim',  C.EMBED_DIM),
        hidden_dim = cfg.get('hidden_dim', C.HIDDEN_DIM),
        noise_std  = 0.0,   # no noise at inspection time
    )
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    return model


def load_npy(dataset: str, split: str) -> tuple:
    embs   = np.load(os.path.join(C.EMBED_DIR, f'{dataset}_{split}_embs.npy'))
    labels = np.load(os.path.join(C.EMBED_DIR, f'{dataset}_{split}_labels.npy'))
    return embs, labels


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Extract sparse codes
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def encode_all(model: CSPINE, embs: np.ndarray, batch_size: int = 512) -> np.ndarray:
    """Encode all embeddings → sparse codes (N, hidden_dim)."""
    codes = []
    t = torch.from_numpy(embs).float()
    for start in range(0, len(t), batch_size):
        z = model.encoder(t[start:start + batch_size])
        codes.append(z.numpy())
    return np.concatenate(codes, axis=0)


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Dimension specialisation scoring
# ─────────────────────────────────────────────────────────────────────────────

def compute_class_mean_activations(codes: np.ndarray,
                                   labels: np.ndarray) -> np.ndarray:
    """
    Returns array of shape (n_classes, hidden_dim):
      class_means[c, j] = mean activation of dimension j
                          across all samples of class c.
    """
    class_ids = sorted(np.unique(labels))
    return np.stack(
        [codes[labels == c].mean(axis=0) for c in class_ids],
        axis=0,
    )   # (n_classes, hidden_dim)


def specialisation_score(class_means: np.ndarray) -> np.ndarray:
    """
    For each dimension j, how specialised is it toward one class?

    Score = (max_class_mean - second_max_class_mean) / (max_class_mean + 1e-9)

    Ranges from 0 (equally active across all classes)
    to ~1 (almost all activation concentrated in one class).

    Returns array of shape (hidden_dim,).
    """
    sorted_means = np.sort(class_means, axis=0)[::-1]   # descending per dim
    top1  = sorted_means[0]   # (hidden_dim,)
    top2  = sorted_means[1] if len(sorted_means) > 1 else np.zeros_like(top1)
    return (top1 - top2) / (top1 + 1e-9)


def top_dims_for_class(class_idx: int,
                        class_means: np.ndarray,
                        spec_scores: np.ndarray,
                        k: int = 3) -> list:
    """
    Return the top-k dimension indices that are both:
      (a) most active for class_idx (high class_means[class_idx, j])
      (b) most specialised (high spec_scores[j])

    We rank by the product: class_mean × specialisation_score.
    This avoids picking dims that are active everywhere.
    """
    combined = class_means[class_idx] * spec_scores
    return np.argsort(combined)[::-1][:k].tolist()


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Top activating sentences per dimension
# ─────────────────────────────────────────────────────────────────────────────

def top_sentences_for_dim(dim_idx: int,
                           codes: np.ndarray,
                           labels: np.ndarray,
                           texts: list,
                           class_idx: int,
                           n: int = 5) -> list:
    """
    Find the n sentences of class_idx that maximally activate dimension dim_idx.

    Returns list of dicts: {rank, activation, label, text}
    """
    # Only look within the target class
    class_mask    = labels == class_idx
    class_codes   = codes[class_mask]
    class_indices = np.where(class_mask)[0]   # original indices

    activations = class_codes[:, dim_idx]
    top_local   = np.argsort(activations)[::-1][:n]

    results = []
    for rank, local_idx in enumerate(top_local):
        orig_idx   = class_indices[local_idx]
        activation = activations[local_idx]
        text = texts[orig_idx] if texts is not None else f'[sample index {orig_idx}]'
        # Truncate very long texts for display
        text = text.strip().replace('\n', ' ')
        if len(text) > 200:
            text = text[:197] + '...'
        results.append({
            'rank':       rank + 1,
            'activation': float(activation),
            'text':       text,
        })
    return results


# ─────────────────────────────────────────────────────────────────────────────
# 6.  Formatting helpers
# ─────────────────────────────────────────────────────────────────────────────

def fmt_table_block(dataset: str,
                    class_idx: int,
                    class_name: str,
                    dim_idx: int,
                    spec_score: float,
                    mean_act: float,
                    sentences: list) -> str:
    """Format one dimension block for the qualitative table."""
    lines = []
    lines.append(f'  Class : {class_name}  |  Dimension : {dim_idx}  '
                 f'|  Mean act : {mean_act:.4f}  |  Spec score : {spec_score:.3f}')
    lines.append('  ' + '─' * 74)
    for s in sentences:
        lines.append(f'  [{s["activation"]:.3f}]  {s["text"]}')
    lines.append('')
    return '\n'.join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# 7.  CSV export — full per-dim statistics
# ─────────────────────────────────────────────────────────────────────────────

def save_dim_csv(dataset: str,
                 class_means: np.ndarray,
                 spec_scores: np.ndarray,
                 label_names: dict) -> None:
    """Save per-dimension statistics for every class to CSV."""
    path = os.path.join(C.MODEL_DIR, f'{dataset}_top_dimensions.csv')
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        header = ['dim_idx', 'spec_score'] + \
                 [f'mean_act_{label_names[c]}' for c in sorted(label_names)]
        writer.writerow(header)
        n_dims = class_means.shape[1]
        for j in range(n_dims):
            row = [j, round(spec_scores[j], 6)] + \
                  [round(class_means[c, j], 6) for c in sorted(label_names)]
            writer.writerow(row)
    print(f'  Dimension stats saved → {path}')


# ─────────────────────────────────────────────────────────────────────────────
# 8.  Per-dataset inspection
# ─────────────────────────────────────────────────────────────────────────────

def inspect_dataset(dataset: str) -> None:
    print('\n' + '═' * 78)
    print(f'  Qualitative Inspection: {dataset.upper()}')
    print('═' * 78)

    val_split  = 'val' if dataset == 'sst2' else 'test'
    label_names = LABEL_NAMES[dataset]

    # ── Load everything ───────────────────────────────────────────────────────
    print('\n  Loading model and embeddings …')
    model              = load_model(dataset)
    val_embs, val_labels = load_npy(dataset, val_split)

    print(f'  Encoding {len(val_embs):,} val/test samples …')
    codes = encode_all(model, val_embs)

    print(f'  Loading raw texts …')
    texts = load_texts(dataset, val_split)

    # ── Compute class statistics ──────────────────────────────────────────────
    class_means = compute_class_mean_activations(codes, val_labels)
    spec_scores = specialisation_score(class_means)

    overall_sparsity = (codes == 0.0).mean()
    print(f'\n  Overall sparsity : {overall_sparsity*100:.1f}%')
    print(f'  Mean active dims : {(codes != 0).sum(axis=1).mean():.1f} / {codes.shape[1]}')

    # ── Build qualitative table ───────────────────────────────────────────────
    table_lines = []
    table_lines.append(f'QUALITATIVE DIMENSION ANALYSIS — {dataset.upper()}')
    table_lines.append('=' * 78)
    table_lines.append(
        f'Val sparsity: {overall_sparsity*100:.1f}%  |  '
        f'Hidden dim: {codes.shape[1]}  |  '
        f'Showing top-{TOP_DIMS_PER_CLASS} dims × top-{TOP_SENTENCES} sentences per class'
    )
    table_lines.append('=' * 78)

    for class_idx in sorted(label_names.keys()):
        class_name = label_names[class_idx]
        n_samples  = (val_labels == class_idx).sum()

        table_lines.append(f'\n{"─"*78}')
        table_lines.append(f'  CLASS: {class_name.upper()}  ({n_samples} val samples)')
        table_lines.append(f'{"─"*78}')

        top_dims = top_dims_for_class(class_idx, class_means, spec_scores,
                                       k=TOP_DIMS_PER_CLASS)

        for dim_idx in top_dims:
            mean_act  = class_means[class_idx, dim_idx]
            spec      = spec_scores[dim_idx]
            sentences = top_sentences_for_dim(
                dim_idx, codes, val_labels, texts, class_idx, n=TOP_SENTENCES
            )
            block = fmt_table_block(
                dataset, class_idx, class_name,
                dim_idx, spec, mean_act, sentences
            )
            table_lines.append(block)

            # Also print to console
            print(f'\n  ── {class_name} / Dim {dim_idx}  '
                  f'(spec={spec:.3f}, mean_act={mean_act:.4f}) ──')
            for s in sentences:
                print(f'    [{s["activation"]:.3f}]  {s["text"]}')

    # ── Save text table ───────────────────────────────────────────────────────
    table_path = os.path.join(C.MODEL_DIR, f'{dataset}_qualitative_table.txt')
    with open(table_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(table_lines))
    print(f'\n  Qualitative table saved → {table_path}')

    # ── Save CSV ──────────────────────────────────────────────────────────────
    save_dim_csv(dataset, class_means, spec_scores, label_names)


# ─────────────────────────────────────────────────────────────────────────────
# 9.  Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    torch.manual_seed(42)
    np.random.seed(42)

    print('=' * 78)
    print('  C-SPINE Qualitative Dimension Inspection')
    print(f'  Device : CPU (model loaded from checkpoint)')
    print('=' * 78)

    for dataset in ('sst2', 'agnews'):
        try:
            inspect_dataset(dataset)
        except FileNotFoundError as e:
            print(f'\n  [SKIP] {dataset}: {e}')

    print('\n' + '=' * 78)
    print('  Done. Check cache/models/ for output files.')
    print('  Use *_qualitative_table.txt for your paper section.')
    print('=' * 78)


if __name__ == '__main__':
    main()
