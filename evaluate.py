"""
evaluate.py  —  Step 3 of C-SPINE pipeline
===============================================
Loads the best C-SPINE checkpoints and evaluates them across three axes:

  (a) Classification accuracy
        - Dense BERT → LogReg                  (upper-bound baseline)
        - Sparse C-SPINE codes → LogReg         (main result)
        - Reconstructed dense → LogReg          (isolates sparsity vs recon error)

  (b) Sparsity quality
        - Mean % of hidden units = 0 per sample
        - Per-class mean sparsity (sanity: should be uniform)

  (c) Reconstruction quality
        - MSE on held-out split (already logged during training, confirmed here)

  (d) Visualisations  (saved to C.MODEL_DIR)
        - dimension_activation_heatmap_{dataset}.png
            rows = SST-2 pos/neg (or AG News classes), cols = top-K active dims
        - training_curves_{dataset}.png
            val MSE + sparsity % across epochs

Run after train_cspine.py:
    python evaluate.py
"""

import os
import csv
import warnings
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')          # no display needed on Colab
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.exceptions import ConvergenceWarning

from config import Config as C
# Re-use model class from training script
from train_cspine import CSPINE, evaluate, make_loader, load_embeddings

warnings.filterwarnings('ignore', category=ConvergenceWarning)


# ─────────────────────────────────────────────────────────────────────────────
# 0.  Label names (for readable outputs)
# ─────────────────────────────────────────────────────────────────────────────

LABEL_NAMES = {
    'sst2':   {0: 'Negative', 1: 'Positive'},
    'agnews': {0: 'World', 1: 'Sports', 2: 'Business', 3: 'Sci/Tech'},
}


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Load checkpoint
# ─────────────────────────────────────────────────────────────────────────────

def load_model(dataset: str) -> CSPINE:
    """Load the best C-SPINE checkpoint for a dataset."""
    ckpt_path = os.path.join(C.MODEL_DIR, f'{dataset}_cspine_best.pt')
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f'Checkpoint not found: {ckpt_path}\n'
            f'Run train_cspine.py first.'
        )
    ckpt = torch.load(ckpt_path, map_location=C.DEVICE)
    cfg  = ckpt.get('config', {})

    model = CSPINE(
        embed_dim  = cfg.get('embed_dim',  C.EMBED_DIM),
        hidden_dim = cfg.get('hidden_dim', C.HIDDEN_DIM),
        noise_std  = cfg.get('noise_std',  C.NOISE_STD),
    ).to(C.DEVICE)

    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    print(f'  Loaded {dataset} checkpoint  '
          f'(epoch {ckpt["epoch"]}, best val MSE {ckpt["val_mse"]:.6f})')
    return model


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Sparse code extraction
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def get_sparse_codes(model: CSPINE,
                     embs: torch.Tensor,
                     batch_size: int = 512) -> np.ndarray:
    """
    Encode dense embeddings → sparse codes in [0, 1].
    Processes in batches to avoid OOM on large datasets.

    Returns np.ndarray (N, hidden_dim).
    """
    model.eval()
    codes = []
    for start in range(0, len(embs), batch_size):
        batch = embs[start:start + batch_size].to(C.DEVICE)
        z = model.encoder(batch)
        codes.append(z.cpu().numpy())
    return np.concatenate(codes, axis=0)


@torch.no_grad()
def get_reconstructed(model: CSPINE,
                      embs: torch.Tensor,
                      batch_size: int = 512) -> np.ndarray:
    """
    Dense → encode → decode → reconstructed dense.
    Used to isolate whether accuracy loss comes from
    sparsity or from reconstruction error.

    Returns np.ndarray (N, embed_dim).
    """
    model.eval()
    recons = []
    for start in range(0, len(embs), batch_size):
        batch = embs[start:start + batch_size].to(C.DEVICE)
        z     = model.encoder(batch)
        recon = model.decoder(z)
        recons.append(recon.cpu().numpy())
    return np.concatenate(recons, axis=0)


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Logistic Regression probe
# ─────────────────────────────────────────────────────────────────────────────

def logreg_probe(X_train: np.ndarray,
                 y_train: np.ndarray,
                 X_test:  np.ndarray,
                 y_test:  np.ndarray,
                 label: str,
                 scale: bool = True) -> float:
    """
    Fit a simple LogReg on top of the given representations.
    Scaling is applied to dense and reconstructed features;
    sparse codes are already in [0,1] so scaling is optional but harmless.

    Returns test accuracy (float).
    """
    if scale:
        scaler  = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test  = scaler.transform(X_test)

    clf = LogisticRegression(
        max_iter  = 1000,
        C         = 1.0,
        solver    = 'lbfgs',
        multi_class = 'auto',
        random_state = 42,
        n_jobs    = -1,
    )
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    acc   = accuracy_score(y_test, preds)

    print(f'\n  [{label}]')
    print(f'    Accuracy : {acc*100:.2f}%')
    print(classification_report(y_test, preds, digits=3, zero_division=0))
    return acc


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Sparsity metrics
# ─────────────────────────────────────────────────────────────────────────────

def sparsity_report(codes: np.ndarray,
                    labels: np.ndarray,
                    dataset: str) -> dict:
    """
    Print and return sparsity statistics.

    Returns dict with keys:
      overall_sparsity   — fraction of zeros over all (sample, unit) pairs
      per_class_sparsity — {class_id: sparsity}
      mean_nonzero       — average number of active units per sample
    """
    overall = (codes == 0.0).mean()
    mean_nz = (codes != 0.0).sum(axis=1).mean()

    per_class = {}
    for c in np.unique(labels):
        mask          = labels == c
        per_class[c]  = (codes[mask] == 0.0).mean()

    label_map = LABEL_NAMES.get(dataset, {})
    print(f'\n  Sparsity report ({dataset})')
    print(f'    Overall sparsity : {overall*100:.1f}%  '
          f'(target > 70%)')
    print(f'    Mean active dims : {mean_nz:.1f} / {codes.shape[1]}')
    for c, s in per_class.items():
        name = label_map.get(c, str(c))
        print(f'    Class {c} ({name:<10}) : {s*100:.1f}% zeros')

    return {
        'overall_sparsity':   float(overall),
        'per_class_sparsity': {int(k): float(v) for k, v in per_class.items()},
        'mean_nonzero':       float(mean_nz),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Reconstruction MSE
# ─────────────────────────────────────────────────────────────────────────────

def reconstruction_mse(model: CSPINE,
                        embs: torch.Tensor,
                        dataset: str,
                        split: str) -> float:
    """Compute held-out reconstruction MSE directly from raw embeddings."""
    loader = make_loader(embs,
                         torch.zeros(len(embs), dtype=torch.long),
                         shuffle=False)
    metrics = evaluate(model, loader, C.DEVICE)
    mse = metrics['mse']
    print(f'\n  Reconstruction MSE ({dataset}/{split}) : {mse:.6f}')
    return mse


# ─────────────────────────────────────────────────────────────────────────────
# 6.  Visualisations
# ─────────────────────────────────────────────────────────────────────────────

def plot_activation_heatmap(codes: np.ndarray,
                             labels: np.ndarray,
                             dataset: str,
                             top_k: int = 50) -> None:
    """
    Per-class mean activation heatmap over the top-K most active dimensions.

    Rows    = classes (e.g. Negative / Positive for SST-2)
    Columns = top_k dimensions ranked by overall mean activation

    Saved to: {MODEL_DIR}/dimension_activation_heatmap_{dataset}.png
    """
    label_map   = LABEL_NAMES.get(dataset, {})
    class_ids   = sorted(np.unique(labels))
    class_names = [label_map.get(c, str(c)) for c in class_ids]

    # Per-class mean activation (n_classes, hidden_dim)
    class_means = np.stack(
        [codes[labels == c].mean(axis=0) for c in class_ids],
        axis=0,
    )

    # Pick top_k dims by overall mean activation
    overall_mean = codes.mean(axis=0)
    top_idx      = np.argsort(overall_mean)[::-1][:top_k]
    heatmap_data = class_means[:, top_idx]   # (n_classes, top_k)

    fig, ax = plt.subplots(figsize=(min(top_k * 0.35, 18), max(len(class_ids) * 1.2, 3)))
    im = ax.imshow(heatmap_data, aspect='auto', cmap='YlOrRd', vmin=0)

    ax.set_yticks(range(len(class_names)))
    ax.set_yticklabels(class_names, fontsize=11)
    ax.set_xlabel(f'Top-{top_k} most active sparse dimensions (ranked by overall mean)', fontsize=10)
    ax.set_title(f'C-SPINE dimension activations — {dataset.upper()}', fontsize=13, pad=12)

    plt.colorbar(im, ax=ax, label='Mean activation (0–1)')
    plt.tight_layout()

    out_path = os.path.join(C.MODEL_DIR, f'dimension_activation_heatmap_{dataset}.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Heatmap saved → {out_path}')


def plot_training_curves(dataset: str) -> None:
    """
    Plot val MSE and val sparsity % across epochs from the CSV log.

    Saved to: {MODEL_DIR}/training_curves_{dataset}.png
    """
    log_path = os.path.join(C.MODEL_DIR, f'{dataset}_training_log.csv')
    if not os.path.exists(log_path):
        print(f'  [skip] Training log not found: {log_path}')
        return

    epochs, val_mse, val_spar = [], [], []
    with open(log_path) as f:
        for row in csv.DictReader(f):
            epochs.append(int(row['epoch']))
            val_mse.append(float(row['val_mse']))
            val_spar.append(float(row['val_sparsity']) * 100)

    fig, ax1 = plt.subplots(figsize=(8, 4))
    color_mse  = '#2166ac'
    color_spar = '#d6604d'

    ax1.plot(epochs, val_mse, color=color_mse,  lw=2, label='Val MSE')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Val MSE', color=color_mse)
    ax1.tick_params(axis='y', labelcolor=color_mse)

    ax2 = ax1.twinx()
    ax2.plot(epochs, val_spar, color=color_spar, lw=2, linestyle='--', label='Val Sparsity %')
    ax2.set_ylabel('Sparsity %', color=color_spar)
    ax2.tick_params(axis='y', labelcolor=color_spar)
    ax2.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.0f%%'))

    # Combined legend
    lines1, labs1 = ax1.get_legend_handles_labels()
    lines2, labs2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labs1 + labs2, loc='center right')

    fig.suptitle(f'C-SPINE training — {dataset.upper()}', fontsize=12)
    plt.tight_layout()

    out_path = os.path.join(C.MODEL_DIR, f'training_curves_{dataset}.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Training curves saved → {out_path}')


# ─────────────────────────────────────────────────────────────────────────────
# 7.  Per-dataset evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_dataset(dataset: str) -> dict:
    """Run the full evaluation suite for one dataset."""

    val_split = 'val' if dataset == 'sst2' else 'test'

    print('\n' + '═' * 60)
    print(f'  Evaluating: {dataset.upper()}')
    print('═' * 60)

    # ── Load embeddings ───────────────────────────────────────────────────────
    train_embs, train_labels = load_embeddings(dataset, 'train')
    val_embs,   val_labels   = load_embeddings(dataset, val_split)

    train_labels_np = train_labels.numpy()
    val_labels_np   = val_labels.numpy()

    # ── Load model ────────────────────────────────────────────────────────────
    model = load_model(dataset)

    # ── Extract representations ───────────────────────────────────────────────
    print('\n  Extracting sparse codes …')
    train_codes = get_sparse_codes(model, train_embs)
    val_codes   = get_sparse_codes(model, val_embs)

    print('  Extracting reconstructed dense …')
    train_recon = get_reconstructed(model, train_embs)
    val_recon   = get_reconstructed(model, val_embs)

    # ── (a) Classification ────────────────────────────────────────────────────
    print('\n' + '─' * 60)
    print('  (a) Classification accuracy — LogReg probe')
    print('─' * 60)

    acc_dense  = logreg_probe(
        train_embs.numpy(), train_labels_np,
        val_embs.numpy(),   val_labels_np,
        label='Dense BERT → LogReg  [baseline]',
    )
    acc_sparse = logreg_probe(
        train_codes, train_labels_np,
        val_codes,   val_labels_np,
        label='Sparse C-SPINE codes → LogReg  [main result]',
        scale=False,   # codes already in [0,1]
    )
    acc_recon  = logreg_probe(
        train_recon, train_labels_np,
        val_recon,   val_labels_np,
        label='Reconstructed dense → LogReg  [recon quality check]',
    )

    acc_drop_sparse = (acc_dense - acc_sparse) * 100
    acc_drop_recon  = (acc_dense - acc_recon)  * 100

    print(f'\n  ┌─ Accuracy summary ─────────────────────────────┐')
    print(f'  │  Dense   (baseline)     : {acc_dense*100:6.2f}%             │')
    print(f'  │  Reconstructed (recon)  : {acc_recon*100:6.2f}%  '
          f'(Δ {acc_drop_recon:+.2f}%)  │')
    print(f'  │  Sparse  (main result)  : {acc_sparse*100:6.2f}%  '
          f'(Δ {acc_drop_sparse:+.2f}%)  │')
    print(f'  └────────────────────────────────────────────────┘')

    # ── (b) Sparsity ──────────────────────────────────────────────────────────
    print('\n' + '─' * 60)
    print('  (b) Sparsity report (val/test split)')
    print('─' * 60)
    spar_stats = sparsity_report(val_codes, val_labels_np, dataset)

    # ── (c) Reconstruction MSE ────────────────────────────────────────────────
    print('\n' + '─' * 60)
    print('  (c) Reconstruction MSE')
    print('─' * 60)
    val_mse = reconstruction_mse(model, val_embs, dataset, val_split)

    # ── (d) Visualisations ────────────────────────────────────────────────────
    print('\n' + '─' * 60)
    print('  (d) Visualisations')
    print('─' * 60)
    plot_activation_heatmap(val_codes, val_labels_np, dataset, top_k=50)
    plot_training_curves(dataset)

    # ── Return summary ────────────────────────────────────────────────────────
    return {
        'dataset':           dataset,
        'acc_dense':         acc_dense,
        'acc_sparse':        acc_sparse,
        'acc_recon':         acc_recon,
        'acc_drop_sparse_%': acc_drop_sparse,
        'acc_drop_recon_%':  acc_drop_recon,
        'overall_sparsity':  spar_stats['overall_sparsity'],
        'mean_active_dims':  spar_stats['mean_nonzero'],
        'val_mse':           val_mse,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 8.  Summary table
# ─────────────────────────────────────────────────────────────────────────────

def print_summary_table(results: list) -> None:
    """Print a compact cross-dataset results table."""
    print('\n' + '═' * 70)
    print('  FINAL RESULTS SUMMARY')
    print('═' * 70)
    header = (f'  {"Dataset":<10} {"Dense%":>8} {"Sparse%":>9} '
              f'{"Recon%":>8} {"Spar%":>7} {"MSE":>10}')
    print(header)
    print('  ' + '─' * 66)
    for r in results:
        print(
            f'  {r["dataset"]:<10} '
            f'{r["acc_dense"]*100:>7.2f}% '
            f'{r["acc_sparse"]*100:>8.2f}%  '
            f'{r["acc_recon"]*100:>7.2f}% '
            f'{r["overall_sparsity"]*100:>6.1f}% '
            f'{r["val_mse"]:>10.6f}'
        )
    print('═' * 70)

    # Write to CSV for the paper / report
    csv_path = os.path.join(C.MODEL_DIR, 'final_results.csv')
    keys = list(results[0].keys())
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(results)
    print(f'\n  Results saved → {csv_path}')


# ─────────────────────────────────────────────────────────────────────────────
# 9.  Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    torch.manual_seed(42)
    np.random.seed(42)

    print('=' * 60)
    print('  C-SPINE Evaluation')
    print(f'  Device : {C.DEVICE}')
    print('=' * 60)

    results = []
    for dataset in ('sst2', 'agnews'):
        try:
            r = evaluate_dataset(dataset)
            results.append(r)
        except FileNotFoundError as e:
            print(f'\n  [SKIP] {dataset}: {e}')

    if results:
        print_summary_table(results)

    print('\n  Done.')


if __name__ == '__main__':
    main()
