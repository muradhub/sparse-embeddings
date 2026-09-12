"""src.visualization — informative matplotlib plots (Agg backend, Colab-safe).

Every figure is built to answer a scientific question on its own:
  - training curves ....: does MSE fall while sparsity rises? where is the
    VAL-selected checkpoint? (twin axes in DISTINCT colors + legend + best
    epoch star; the old version drew both series in the same blue)
  - activation hist ....: what does the code distribution look like? (log-scale
    zero bar that no longer crushes the bulk; % saturated at the CappedReLU
    ceiling 1.0 annotated — the PSL peak-chasing signature)
  - class heatmap .......: which dims serve which classes? (TRUE dimension ids
    on the x-axis, not ranks; contrast fitted to the data range; optional
    specialization stars)
  - sparsity tradeoff ...: how do recon quality and downstream accuracy move
    with L0? (sparsity % annotated per point, dense-baseline reference line,
    points sorted by L0 so the line reads as a frontier)
  - recon vs L0 .........: do examples that recruit fewer dims reconstruct
    worse? (uses the collected recon_vs_l0 bins that were never plotted)
  - downstream bars .....: dense / recon / sparse / PCA / randproj side
    by side (accuracy + macro-F1 grouped bars + per-class F1 heatmap)
  - transfer matrix .....: frozen SAE trained on A, evaluated on B
    (cosine + MSE panels; diagonals = in-domain reference)
  - mechanism bars ......: C-SPINE vs TopK vs BatchTopK vs DenseAE on
    val MSE, mean L0 and dead-feature fraction.
  - purity/spec hist ....: QUAL — are the showcased dims cherry-picked or
    typical? (full-population histograms of purity@K and specialization with
    the showcased dims marked)
  - per-dim class bars ..: QUAL — does a showcased dim fire selectively for
    one class? (grouped mean-activation bars per selected dim)
  - dim coactivation ....: QUAL — are showcased dims distinct directions or
    duplicated atoms? (Pearson correlation heatmap among selected dims)
  - coherence bars ......: QUAL — pure yet incoherent? (per-dim purity@50
    beside dense-space coherence of its top activators)

Only matplotlib + numpy. No global rcParams changes (import-safe).
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_MSE_COLOR = "#2166ac"    # blue
_SPAR_COLOR = "#d6604d"   # red/orange (distinct from MSE)
_ACC_COLOR = "#1b7837"    # green
_BASELINE_COLOR = "#666666"


def _finalize(fig, out_path: str) -> None:
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# --- training curves --------------------------------------------------------

def plot_training_curves(history: List[Dict], out_path: str, title: str = "") -> None:
    """Val MSE (left, blue) + val sparsity % (right, red) across epochs.

    Star + annotation mark the VAL-MSE-selected checkpoint (the only thing
    that may be carried to TEST). Same signature as before.
    """
    if not history:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.text(0.5, 0.5, "no training history", ha="center", va="center")
        ax.set_title(title or "C-SPINE training (VAL only; TEST untouched)")
        _finalize(fig, out_path)
        return
    epochs = [h["epoch"] for h in history]
    mse = [h["val_mse"] for h in history]
    spar = [h["val_sparsity"] * 100 for h in history]
    best_i = int(np.argmin(mse)) if mse else 0
    best_i = max(0, min(best_i, len(epochs) - 1))

    fig, ax1 = plt.subplots(figsize=(8, 4.2))
    l1, = ax1.plot(epochs, mse, color=_MSE_COLOR, lw=2, marker="o",
                   markersize=4, label="Val MSE (selects checkpoint)")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Val MSE", color=_MSE_COLOR)
    ax1.tick_params(axis="y", labelcolor=_MSE_COLOR)
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    l2, = ax2.plot(epochs, spar, color=_SPAR_COLOR, lw=2, linestyle="--",
                   marker="s", markersize=4, label="Val sparsity %")
    ax2.set_ylabel("Sparsity % (zeros)", color=_SPAR_COLOR)
    ax2.tick_params(axis="y", labelcolor=_SPAR_COLOR)

    # best-checkpoint star on the MSE axis
    ax1.plot(epochs[best_i], mse[best_i], color=_MSE_COLOR, marker="*",
             markersize=14, markeredgecolor="black", zorder=5)
    ax1.annotate(f"best ep {epochs[best_i]}: MSE {mse[best_i]:.4f}, "
                 f"spar {spar[best_i]:.1f}%",
                 xy=(epochs[best_i], mse[best_i]),
                 xytext=(8, 12), textcoords="offset points", fontsize=8,
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.85),
                 arrowprops=dict(arrowstyle="->", color="black", lw=0.8))

    ax1.legend([l1, l2], [l1.get_label(), l2.get_label()],
               loc="center right", fontsize=8)
    fig.suptitle(title or "C-SPINE training (VAL only; TEST untouched)",
                 fontsize=11)
    _finalize(fig, out_path)


# --- sparsity tradeoff ------------------------------------------------------

def plot_sparsity_tradeoff(rows: List[Dict], out_path: str,
                           dense_acc: Optional[float] = None,
                           title_note: str = "single seed") -> None:
    """Sparsity <-> reconstruction and sparsity <-> downstream.

    Points sorted by mean L0 so the polyline reads as a frontier. Each point
    is annotated with level name + sparsity %. Optional dense-baseline line
    answers "how far from the upper bound?". Same signature as before
    (+ optional dense_acc).
    """
    rows = sorted([r for r in rows if np.isfinite(r.get("mean_l0", np.nan))],
                  key=lambda r: r["mean_l0"])
    if not rows:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.text(0.5, 0.5, "no sweep points", ha="center", va="center")
        _finalize(fig, out_path)
        return
    x = [r["mean_l0"] for r in rows]
    y_mse = [r["mse"] for r in rows]
    y_acc = [r["sparse_acc"] for r in rows]
    labels = [r.get("label", "") for r in rows]
    sp = [r.get("sparsity_ratio", float("nan")) * 100 for r in rows]

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.2))
    for ax in (a1, a2):
        ax.margins(x=0.12)  # padding so point annotations never clip
    # With a single point there is no frontier line — show a marker instead.
    if len(rows) == 1:
        a1.scatter(x, y_mse, color=_MSE_COLOR, s=60, zorder=3)
        a2.scatter(x, y_acc, color=_ACC_COLOR, s=60, zorder=3)
    else:
        a1.plot(x, y_mse, color=_MSE_COLOR, marker="o", lw=1.5)
        a2.plot(x, y_acc, color=_ACC_COLOR, marker="o", lw=1.5)
    for xi, yi, lb, s in zip(x, y_mse, labels, sp):
        a1.annotate(f"{lb}\n{s:.0f}% zeros", (xi, yi), xytext=(6, 6),
                    textcoords="offset points", fontsize=7)
    a1.set_xlabel("Mean L0 (active dims / example)  →  sparser on the left")
    a1.set_ylabel("Recon MSE (TEST, lower = better)")
    a1.set_title("Sparsity ↔ reconstruction")
    a1.grid(True, alpha=0.3)

    a2.plot(x, y_acc, color=_ACC_COLOR, marker="o", lw=1.5)
    for xi, yi, lb in zip(x, y_acc, labels):
        a2.annotate(lb, (xi, yi), xytext=(6, 6),
                    textcoords="offset points", fontsize=7)
    if dense_acc is not None:
        a2.axhline(dense_acc, color=_BASELINE_COLOR, linestyle=":",
                   lw=1.5, label=f"Dense {dense_acc:.3f}")
        a2.legend(fontsize=7, loc="lower right")
    a2.set_xlabel("Mean L0 (active dims / example)  →  sparser on the left")
    a2.set_ylabel("Sparse probe accuracy (TEST)")
    a2.set_title("Sparsity ↔ downstream")
    a2.grid(True, alpha=0.3)

    fig.suptitle(f"Sparsity sweep ({title_note}; H dims fixed)",
                 fontsize=11)
    _finalize(fig, out_path)


# --- activation histogram ---------------------------------------------------

def plot_activation_hist(codes: np.ndarray, out_path: str, title: str = "",
                         stats: Optional[Dict] = None) -> None:
    """Code-value distribution.

    Left (log y): the zero bar no longer crushes the bulk. Right (linear):
    non-zero bulk with the CappedReLU-ceiling spike at 1.0 annotated — that
    spike is the PSL peak-chasing signature (expected), not a bug. A stats
    box reports L0 / sparsity / dead / saturated share.
    """
    vals = np.asarray(codes).ravel()
    n = vals.size
    n_zero = int((vals == 0.0).sum())
    nz = vals[vals != 0.0]
    pct_sat = float((vals == 1.0).mean()) * 100 if n else 0.0

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.2))
    a1.hist(vals, bins=50, range=(0, 1), color=_MSE_COLOR, edgecolor="white",
            linewidth=0.3)
    a1.set_yscale("log")
    a1.set_title(f"All activations (log y; {100.0 * n_zero / max(n, 1):.1f}% exact zeros)")
    a1.set_xlabel("Activation value")
    a1.set_ylabel("Count (log scale)")
    a1.grid(True, alpha=0.3, which="both")

    if len(nz):
        a2.hist(nz, bins=50, range=(0, 1), color=_SPAR_COLOR,
                edgecolor="white", linewidth=0.3)
    a2.axvline(1.0, color="black", linestyle=":", lw=1.2)
    a2.annotate(f"ceiling spike @1.0\n{pct_sat:.1f}% of all codes\n(PSL peak-chasing + cap)",
                xy=(1.0, a2.get_ylim()[1] if a2.get_ylim()[1] > 0 else 1),
                xytext=(-8, -8), textcoords="offset points", fontsize=7,
                ha="right", va="top",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.9))
    a2.set_title("Non-zero activations (linear y)")
    a2.set_xlabel("Activation value")
    a2.grid(True, alpha=0.3)

    box = []
    if stats:
        box.append(f"L0 mean {stats.get('mean_l0', float('nan')):.1f} / "
                   f"median {stats.get('median_l0', float('nan')):.0f}")
        box.append(f"sparsity {stats.get('sparsity_ratio', float('nan')) * 100:.1f}%  |  "
                   f"dead {stats.get('dead_fraction', float('nan')) * 100:.1f}%")
        box.append(f"saturated@1.0: {pct_sat:.1f}% of codes")
    else:
        box.append(f"zeros: {100.0 * n_zero / max(n, 1):.1f}%  |  "
                   f"saturated@1.0: {pct_sat:.1f}%")
    fig.suptitle((title + "  —  " if title else "") + "  |  ".join(box),
                 fontsize=10)
    _finalize(fig, out_path)


# --- class heatmap ----------------------------------------------------------

def plot_class_heatmap(class_means: np.ndarray, class_names: List[str],
                       out_path: str, top_k: int = 50, title: str = "",
                       spec_scores: Optional[np.ndarray] = None,
                       top_spec_n: int = 3) -> None:
    """Per-class mean activation for the top-K dims by overall mean.

    X labels are TRUE dimension ids (not ranks) so dims can be cross-
    referenced with the interpretability tables. Contrast is fitted to the
    data range (the old fixed vmin=0 washed everything out when the floor
    was ~0.3). If spec_scores are given, the most specialized dim of each
    class gets a white star — correlation at a glance, not proof of
    interpretability.
    """
    class_means = np.asarray(class_means, dtype=float)
    overall = class_means.mean(axis=0)
    order = np.argsort(overall)[::-1][:top_k]
    data = class_means[:, order]
    vmin, vmax = float(data.min()), float(data.max())
    if vmax <= vmin:
        vmax = vmin + 1e-9

    w = min(max(top_k * 0.35, 8), 18)
    h = max(len(class_names) * 1.1, 3.2)
    fig, ax = plt.subplots(figsize=(w, h))
    im = ax.imshow(data, aspect="auto", cmap="YlOrRd", vmin=vmin, vmax=vmax)
    ax.set_yticks(range(len(class_names)))
    ax.set_yticklabels(class_names, fontsize=10)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([str(int(d)) for d in order], rotation=90, fontsize=6)
    ax.set_xlabel(f"Top-{len(order)} dims by overall mean (TRUE dim ids; "
                  f"star = most specialized per class)", fontsize=8)
    ax.set_title(title or "Per-class mean activation", fontsize=11, pad=10)

    if spec_scores is not None:
        spec_scores = np.asarray(spec_scores, dtype=float)
        for ci in range(data.shape[0]):
            # most specialized dim of this class among the shown ones
            cands = order[np.argsort(class_means[ci, order] * spec_scores[order])[::-1][:top_spec_n]]
            for d in cands:
                xi = int(np.where(order == d)[0][0])
                ax.plot(xi, ci, marker="*", color="white",
                        markersize=9, markeredgecolor="black", markeredgewidth=0.6)

    plt.colorbar(im, ax=ax,
                 label=f"Mean activation\n(fitted range {vmin:.2f}–{vmax:.2f})")
    _finalize(fig, out_path)


# --- recon vs L0 (uses collected recon_vs_l0 bins) ---------------------------

def plot_recon_vs_l0(bins: List[Dict], out_path: str, title: str = "") -> None:
    """Do examples that recruit fewer dims reconstruct worse?

    Marker size ∝ bin count. A falling curve = graceful degradation (more
    dims help); a flat curve = L0 buys little fidelity.
    """
    bins = [b for b in bins if b.get("n", 0) > 0]
    if not bins:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.text(0.5, 0.5, "no bins", ha="center")
        _finalize(fig, out_path)
        return
    mid = [(b["l0_lo"] + b["l0_hi"]) / 2 for b in bins]
    mse = [b["mean_mse"] for b in bins]
    n = np.array([b["n"] for b in bins], dtype=float)
    sizes = 40 + 260 * (n / n.max())

    fig, ax = plt.subplots(figsize=(8, 4.2))
    sc = ax.scatter(mid, mse, s=sizes, color=_MSE_COLOR, alpha=0.75,
                    edgecolors="black", linewidths=0.6, zorder=3)
    ax.plot(mid, mse, color=_MSE_COLOR, lw=1.2, alpha=0.6)
    for m, e, c in zip(mid, mse, n):
        ax.annotate(f"n={int(c)}", (m, e), xytext=(5, 5),
                    textcoords="offset points", fontsize=7)
    ax.set_xlabel("L0 bin midpoint (active dims for these examples)")
    ax.set_ylabel("Mean recon MSE in bin (TEST)")
    ax.set_title(title or "Reconstruction quality vs. recruited dims")
    ax.grid(True, alpha=0.3)
    _finalize(fig, out_path)


# --- downstream grouped bars -------------------------------------------------

def plot_downstream_bars(downstream: Dict[str, Dict], class_names: Sequence[str],
                         out_path: str, title: str = "") -> None:
    """Headline bars (accuracy + macro-F1 per representation) + per-class F1.

    Left: does the sparse code keep the dense signal, and do PCA/RP match it
    (H6)? Right: which classes survive sparsification and which do not?
    """
    methods = [m for m in ("dense", "reconstructed", "sparse", "pca",
                           "randproj", "dense_ae")
               if m in downstream]
    pretty = {"dense": "dense", "reconstructed": "recon", "sparse": "sparse",
              "pca": "PCA-64", "randproj": "RandProj-64", "dense_ae": "dense-AE"}
    acc = [downstream[m]["accuracy"] for m in methods]
    f1 = [downstream[m]["macro_f1"] for m in methods]

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.4),
                                 gridspec_kw={"width_ratios": [1.15, 1]})
    x = np.arange(len(methods))
    b1 = a1.bar(x - 0.2, acc, 0.4, label="accuracy", color=_ACC_COLOR,
                edgecolor="black", linewidth=0.6)
    b2 = a1.bar(x + 0.2, f1, 0.4, label="macro-F1", color=_SPAR_COLOR,
                edgecolor="black", linewidth=0.6)
    a1.set_xticks(x)
    a1.set_xticklabels([pretty.get(m, m) for m in methods], rotation=15,
                       ha="right", fontsize=8)
    a1.set_ylabel("Score (TEST)")
    a1.set_ylim(0, 1.0)
    a1.set_title("Same LogReg protocol everywhere")
    a1.legend(fontsize=8)
    a1.grid(True, axis="y", alpha=0.3)
    for r1, r2 in zip(b1, b2):
        a1.text(r1.get_x() + r1.get_width() / 2, r1.get_height() + 0.01,
                f"{r1.get_height():.2f}", ha="center", fontsize=6)
        a1.text(r2.get_x() + r2.get_width() / 2, r2.get_height() + 0.01,
                f"{r2.get_height():.2f}", ha="center", fontsize=6)

    # Per-class panel: rows may have different class counts if a dataset split
    # missed a label — pad with NaN and annotate only finite cells.
    n_cls = max([len(downstream[m].get("per_class_f1", [])) for m in methods] +
                [len(class_names)])
    per = np.full((len(methods), n_cls), np.nan)
    for i, m in enumerate(methods):
        v = np.asarray(downstream[m].get("per_class_f1", []), dtype=float)
        per[i, : len(v)] = v
    im = a2.imshow(np.nan_to_num(per, nan=0.0), aspect="auto", cmap="YlGn",
                   vmin=0, vmax=1)
    a2.set_yticks(range(len(methods)))
    a2.set_yticklabels([pretty.get(m, m) for m in methods], fontsize=8)
    a2.set_xticks(range(len(class_names)))
    a2.set_xticklabels(list(class_names), rotation=25, ha="right",
                       fontsize=8)
    a2.set_title("Per-class F1 (rows = representation)")
    for i in range(per.shape[0]):
        for j in range(per.shape[1]):
            if np.isnan(per[i, j]):
                continue
            a2.text(j, i, f"{per[i, j]:.2f}", ha="center", va="center",
                    fontsize=7)
    plt.colorbar(im, ax=a2, label="F1")
    fig.suptitle(title or "Downstream preservation (locked TEST, once)",
                 fontsize=11)
    _finalize(fig, out_path)


# --- transfer matrix ----------------------------------------------------------

def plot_transfer_matrix(transfer: Dict[str, Dict], out_path: str,
                         title: str = "") -> None:
    """Frozen SAE from row-domain evaluated on column-domain.

    Left: cosine similarity (higher = reusable directions). Right: recon MSE
    (lower = better). Diagonals are in-domain references; off-diagonals test
    cross-domain reuse. Keys look like 'xfer_sst2_on_agnews'.
    """
    pairs = {}
    for k, v in transfer.items():
        if not k.startswith("xfer_") or "_on_" not in k:
            continue
        tr, te = k[len("xfer_"):].split("_on_", 1)
        pairs[(tr, te)] = v
    doms = sorted({d for p in pairs for d in p})
    if not doms:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "no transfer pairs", ha="center")
        _finalize(fig, out_path)
        return
    cos = np.full((len(doms), len(doms)), np.nan)
    mse = np.full((len(doms), len(doms)), np.nan)
    l0m = np.full((len(doms), len(doms)), np.nan)
    for i, tr in enumerate(doms):
        for j, te in enumerate(doms):
            v = pairs.get((tr, te))
            if v is None:
                continue
            cos[i, j] = v.get("cosine", np.nan)
            mse[i, j] = v.get("mse", np.nan)
            l0m[i, j] = v.get("L0", np.nan)

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.6))
    for ax, mat, cmap, lab, fmt in (
            (a1, cos, "YlGn", "Cosine(orig, recon)", ".3f"),
            (a2, mse, "YlOrRd", "Recon MSE", ".4f")):
        im = ax.imshow(mat, aspect="auto", cmap=cmap)
        ax.set_xticks(range(len(doms)))
        ax.set_xticklabels(doms, rotation=20, ha="right")
        ax.set_yticks(range(len(doms)))
        ax.set_yticklabels(doms)
        ax.set_xlabel("evaluated on (TEST)")
        ax.set_ylabel("SAE trained on")
        ax.set_title(lab)
        for i in range(len(doms)):
            for j in range(len(doms)):
                if np.isnan(mat[i, j]):
                    continue
                extra = f"\nL0 {l0m[i, j]:.0f}" if not np.isnan(l0m[i, j]) else ""
                ax.text(j, i, f"{mat[i, j]:{fmt}}{extra}", ha="center",
                        va="center", fontsize=7)
        plt.colorbar(im, ax=ax, label=lab)
    fig.suptitle(title or "Cross-domain reuse (frozen SAE, no retraining)",
                 fontsize=11)
    _finalize(fig, out_path)


# --- mechanism comparison ------------------------------------------------------

def plot_mechanism_bars(abl: Dict[str, Dict], out_path: str,
                        title: str = "") -> None:
    """C-SPINE vs TopK vs BatchTopK vs DenseAE: val MSE, mean L0, dead share.

    Answers whether the soft ASL/PSL objective buys anything over hard TopK
    (fixed L0, lower MSE here, but catastrophic dictionary waste) or over no
    sparsity at all (best MSE, but dead ReLUs + no learned sparsity).
    """
    order = [k for k in ("abl_mech_cspine", "abl_mech_topk",
                         "abl_mech_batchtopk", "abl_mech_dense_ae") if k in abl]
    if not order:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "no mechanism runs", ha="center")
        _finalize(fig, out_path)
        return
    names = [abl[k].get("mechanism", k) for k in order]
    vmse = [abl[k].get("val_mse", float("nan")) for k in order]
    l0 = [abl[k].get("mean_l0", float("nan")) for k in order]
    dead = [abl[k].get("dead_fraction", float("nan")) * 100 for k in order]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    axes[0].bar(names, vmse, color=_MSE_COLOR, edgecolor="black",
                linewidth=0.6)
    axes[0].set_title("Val MSE (lower = better)")
    axes[0].tick_params(axis="x", labelrotation=15, labelsize=8)
    axes[0].grid(True, axis="y", alpha=0.3)
    for x, y in zip(names, vmse):
        axes[0].text(x, y, f"{y:.3f}", ha="center", va="bottom", fontsize=7)

    axes[1].bar(names, l0, color=_SPAR_COLOR, edgecolor="black",
                linewidth=0.6)
    axes[1].set_title("Mean L0 on TEST")
    axes[1].tick_params(axis="x", labelrotation=15, labelsize=8)
    axes[1].grid(True, axis="y", alpha=0.3)
    for x, y in zip(names, l0):
        axes[1].text(x, y, f"{y:.0f}", ha="center", va="bottom", fontsize=7)

    axes[2].bar(names, dead, color="#762a83", edgecolor="black",
                linewidth=0.6)
    axes[2].set_title("Dead features % (never fire on TEST)")
    axes[2].tick_params(axis="x", labelrotation=15, labelsize=8)
    axes[2].grid(True, axis="y", alpha=0.3)
    for x, y in zip(names, dead):
        axes[2].text(x, y, f"{y:.0f}%", ha="center", va="bottom", fontsize=7)

    fig.suptitle(title or "Sparsity mechanism ablation (same H, same budget)",
                 fontsize=11)
    _finalize(fig, out_path)


# --- qualitative: population histograms --------------------------------------

def plot_purity_spec_hist(purity: Sequence[float], spec: Sequence[float],
                          out_path: str, title: str = "",
                          selected_dims: Optional[Sequence[int]] = None,
                          all_spec: Optional[Sequence[float]] = None) -> None:
    """Are the showcased dims cherry-picked? Full-population background.

    Left: histogram of purity@K over ALL dims with median line. Right:
    histogram of specialization over ALL dims. Showcased dims are marked as
    ticks on top so the reader sees where the hand-picked examples sit in
    the population. Guards the paper against cherry-picking concerns.
    """
    purity = np.asarray(list(purity), dtype=float)
    spec = np.asarray(list(spec), dtype=float)
    purity = purity[np.isfinite(purity)]
    spec = spec[np.isfinite(spec)]

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.2))
    if len(purity):
        a1.hist(purity, bins=20, range=(0, 1), color=_ACC_COLOR,
                edgecolor="white", linewidth=0.4)
        a1.axvline(float(np.median(purity)), color="black", linestyle="--",
                   lw=1.2, label=f"median {np.median(purity):.2f}")
        a1.legend(fontsize=7)
    else:
        a1.text(0.5, 0.5, "no purity values", ha="center")
    a1.set_xlabel("Purity@K (max class share of top activators)")
    a1.set_ylabel("Dims")
    a1.set_title("Purity distribution (all dims)")
    a1.grid(True, axis="y", alpha=0.3)

    if len(spec):
        a2.hist(spec, bins=20, range=(0, 1), color=_SPAR_COLOR,
                edgecolor="white", linewidth=0.4)
        a2.axvline(float(np.median(spec)), color="black", linestyle="--",
                   lw=1.2, label=f"median {np.median(spec):.2f}")
        a2.legend(fontsize=7)
    else:
        a2.text(0.5, 0.5, "no spec values", ha="center")
    a2.set_xlabel("Specialization (top1-top2)/top1")
    a2.set_title("Specialization distribution (all dims)")
    a2.grid(True, axis="y", alpha=0.3)
    fig.suptitle(title or "Qualitative background: full-dim population",
                 fontsize=11)
    _finalize(fig, out_path)


# --- qualitative: per-dim class selectivity -----------------------------------

def plot_per_dim_class_bars(dims: Sequence[int], class_means: np.ndarray,
                            class_names: Sequence[str], out_path: str,
                            title: str = "") -> None:
    """Grouped mean-activation bars for each showcased dim (selectivity eye-test).

    One group per dim, one bar per class. A selective dim shows one tall bar;
    a distributed dim shows flat bars. Complements the heatmap with exact
    magnitudes and error-free reading (no colormap guessing).
    """
    dims = [int(d) for d in dims]
    cm = np.asarray(class_means, dtype=float)
    n_cls = cm.shape[0]
    if not dims or n_cls == 0:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.text(0.5, 0.5, "no dims", ha="center")
        _finalize(fig, out_path)
        return
    x = np.arange(len(dims))
    w = min(0.8 / max(n_cls, 1), 0.35)
    fig, ax = plt.subplots(figsize=(max(8, len(dims) * 1.1), 4.2))
    colors = plt.cm.Set2(np.linspace(0, 1, n_cls))
    for ci in range(n_cls):
        vals = [cm[ci, d] for d in dims]
        ax.bar(x + (ci - (n_cls - 1) / 2) * w, vals, w,
               label=str(class_names[ci]) if ci < len(class_names) else str(ci),
               color=colors[ci], edgecolor="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([str(d) for d in dims], fontsize=8)
    ax.set_xlabel("Showcased sparse dimension (true id)")
    ax.set_ylabel("Mean activation on TEST")
    ax.set_title(title or "Per-dim class selectivity")
    ax.legend(fontsize=8, title="Class")
    ax.grid(True, axis="y", alpha=0.3)
    _finalize(fig, out_path)


# --- qualitative: dim redundancy ------------------------------------------------

def plot_dim_coactivation(corr: np.ndarray, dims: Sequence[int],
                          out_path: str, title: str = "") -> None:
    """Pearson correlation heatmap among showcased dims (redundancy check).

    Near-1 off-diagonal = duplicated atoms (dictionary waste); near-0 =
    distinct directions. Values annotated so "two dims, one concept" cannot
    hide behind similar top-example lists.
    """
    corr = np.asarray(corr, dtype=float)
    dims = [int(d) for d in dims]
    n = len(dims)
    if corr.shape != (n, n):
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "corr shape mismatch", ha="center")
        _finalize(fig, out_path)
        return
    fig, ax = plt.subplots(figsize=(max(5, n * 0.7 + 2), max(4, n * 0.6 + 1.5)))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(n))
    ax.set_xticklabels([str(d) for d in dims], rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(n))
    ax.set_yticklabels([str(d) for d in dims], fontsize=8)
    ax.set_xlabel("Dim")
    ax.set_ylabel("Dim")
    ax.set_title(title or "Co-activation among showcased dims")
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{corr[i, j]:.2f}", ha="center", va="center",
                    fontsize=7,
                    color="white" if abs(corr[i, j]) > 0.5 else "black")
    plt.colorbar(im, ax=ax, label="Pearson r of activations")
    _finalize(fig, out_path)


# --- qualitative: purity vs coherence --------------------------------------------

def plot_coherence_bars(blocks: List[Dict], out_path: str,
                        title: str = "") -> None:
    """Purity@50 beside dense-space coherence per showcased dim.

    Exposes the "pure yet incoherent" case: high label purity with low dense
    neighbourhood similarity means the dim tracks the label without grouping
    semantically nearby sentences (correlation without coherence).
    Blocks are dicts with keys dim / class_name / purity / coherence as
    produced by interpretability.full_feature_report.
    """
    if not blocks:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.text(0.5, 0.5, "no feature blocks", ha="center")
        _finalize(fig, out_path)
        return
    labels = [f"{b.get('class_name', '')}\ndim {b.get('dim', '?')}" for b in blocks]
    pur = [float(b.get("purity", {}).get("purity", np.nan)) for b in blocks]
    coh = [float((b.get("coherence") or {}).get("mean_pairwise_cosine", np.nan))
           for b in blocks]
    x = np.arange(len(blocks))
    fig, ax = plt.subplots(figsize=(max(9, len(blocks) * 1.2), 4.4))
    b1 = ax.bar(x - 0.2, pur, 0.4, label="purity@50", color=_ACC_COLOR,
                edgecolor="black", linewidth=0.6)
    b2 = ax.bar(x + 0.2, coh, 0.4, label="coherence (dense cos)", color=_MSE_COLOR,
                edgecolor="black", linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylim(-0.05, 1.05)
    ax.set_ylabel("Score")
    ax.set_title(title or "Purity vs coherence per showcased dim")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    for r in list(b1) + list(b2):
        h = r.get_height()
        if np.isfinite(h):
            ax.text(r.get_x() + r.get_width() / 2, h + 0.015, f"{h:.2f}",
                    ha="center", fontsize=6)
    _finalize(fig, out_path)
