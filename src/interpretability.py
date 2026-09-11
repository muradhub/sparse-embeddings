"""src.interpretability — feature-level analysis beyond heatmaps.

For each latent feature:
  1. highest/lowest activating examples (class-conditional AND global)
  2. activation distributions (global + per-class rates)
  3. class purity among top activators (single-K AND multi-K curves)
  4. semantic consistency proxy (label agreement + dense-space coherence)
  5. seed stability hooks (see experiments/stability)
  6. automated template descriptions
  7. example -> dims view + dense-vs-sparse neighbourhood preservation
  8. paper-ready exports (txt / md / tex / csv)

Careful vocabulary (§9): discriminative (helps classifier) != correlated
(co-occurs with label) != coherent (top examples look similar) !=
interpretable (human can name the concept). We report purity/correlation
metrics but do NOT equate them with interpretability.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np


def class_mean_activations(codes: np.ndarray, labels: np.ndarray) -> np.ndarray:
    classes = sorted(np.unique(labels).tolist())
    return np.stack([codes[labels == c].mean(axis=0) for c in classes], axis=0)


def specialisation_score(class_means: np.ndarray) -> np.ndarray:
    """(top1 - top2)/(top1 + eps) per dim. 0=shared, ~1=class-specific."""
    s = np.sort(class_means, axis=0)[::-1]
    top1 = s[0]
    top2 = s[1] if len(s) > 1 else np.zeros_like(top1)
    return (top1 - top2) / (top1 + 1e-9)


def top_dims_for_class(class_idx_pos: int, class_means: np.ndarray,
                       spec: np.ndarray, k: int = 3) -> List[int]:
    combined = class_means[class_idx_pos] * spec
    return np.argsort(combined)[::-1][:k].tolist()


def top_examples_for_dim(dim: int, codes: np.ndarray, labels: np.ndarray,
                         texts: Optional[List[str]], class_value: Optional[int] = None,
                         n: int = 5, which: str = "top") -> List[Dict]:
    """Top/bottom activating examples for a dim (optionally within a class)."""
    acts = codes[:, dim]
    idx = np.arange(len(acts))
    if class_value is not None:
        m = labels == class_value
        acts, idx = acts[m], idx[m]
    order = np.argsort(acts)
    sel = order[::-1][:n] if which == "top" else order[:n]
    out = []
    for r, li in enumerate(sel):
        oi = int(idx[li])
        t = texts[oi] if texts is not None else f"[sample index {oi}]"
        t = t.strip().replace("\n", " ")
        if len(t) > 220:
            t = t[:217] + "..."
        out.append({"rank": r + 1, "index": oi, "activation": float(acts[li]),
                    "label": int(labels[oi]), "text": t})
    return out


def class_purity(codes: np.ndarray, labels: np.ndarray, dim: int,
                 top_n: int = 50) -> Dict:
    """Max class fraction among top-N activators + entropy.

    High purity = discriminative/correlated; NOT proof of interpretability.
    """
    acts = codes[:, dim]
    top = np.argsort(acts)[::-1][:top_n]
    labs = labels[top]
    classes, counts = np.unique(labs, return_counts=True)
    fracs = counts / max(len(top), 1)
    ent = float(-np.sum(fracs * np.log(fracs + 1e-12)))
    return {"purity": float(fracs.max()), "majority_class": int(classes[np.argmax(counts)]),
            "entropy": ent, "top_n": int(top_n)}


def activation_distribution(codes: np.ndarray, dim: int, bins: int = 20) -> Dict:
    vals = codes[:, dim]
    hist, edges = np.histogram(vals, bins=bins, range=(0.0, 1.0))
    return {"hist": hist.tolist(), "edges": edges.tolist(),
            "mean": float(vals.mean()), "std": float(vals.std()),
            "frac_zero": float((vals == 0.0).mean()),
            "frac_gt_half": float((vals > 0.5).mean())}


def describe_feature(dim: int, codes: np.ndarray, labels: np.ndarray,
                     label_names: Dict[int, str],
                     texts: Optional[List[str]] = None, top_n: int = 5) -> str:
    """Automated template description from top activators (a starting point,
    not a human interpretability judgment)."""
    pur = class_purity(codes, labels, dim, top_n=50)
    maj = label_names.get(pur["majority_class"], str(pur["majority_class"]))
    tops = top_examples_for_dim(dim, codes, labels, texts, n=top_n, which="top")
    ex = " | ".join(f"[{t['activation']:.2f}] {t['text'][:90]}" for t in tops)
    return (f"dim {dim}: purity@{50}={pur['purity']:.2f} (majority={maj}), "
            f"mean_act={codes[:, dim].mean():.4f}. Top-{top_n}: {ex}")


def full_feature_report(codes: np.ndarray, labels: np.ndarray,
                        label_names: Dict[int, str],
                        texts: Optional[List[str]] = None,
                        top_dims_per_class: int = 3,
                        top_sentences: int = 5,
                        n_bottom: int = 3,
                        purity_ks: Sequence[int] = (10, 20, 50),
                        dense_embs: Optional[np.ndarray] = None,
                        coherence_top_n: int = 20) -> Dict:
    """Feature report with contrast + robustness + coherence evidence.

    Backward compatible: the original keys (class, class_name, dim,
    mean_act, spec_score, purity, top_examples, description) are unchanged.
    New keys per block: bottom_examples (non-firing contrast), global_top
    (unconstrained ranking — exposes class-mismatch cases), purity_at_ks
    (K-sensitivity curve), per_class_stats (mean/std/firing-rate per class),
    coherence (dense-space similarity of top activators, if dense_embs given).
    """
    cm = class_mean_activations(codes, labels)
    spec = specialisation_score(cm)
    classes = sorted(np.unique(labels).tolist())
    blocks = []
    for pos, c in enumerate(classes):
        for d in top_dims_for_class(pos, cm, spec, k=top_dims_per_class):
            d = int(d)
            coh = (coherence_score(dense_embs, codes, d, top_n=coherence_top_n)
                   if dense_embs is not None else None)
            blocks.append({
                "class": int(c),
                "class_name": label_names.get(int(c), str(c)),
                "dim": int(d),
                "mean_act": float(cm[pos, d]),
                "spec_score": float(spec[d]),
                "purity": class_purity(codes, labels, int(d)),
                "purity_at_ks": purity_at_ks(codes, labels, int(d), ks=tuple(purity_ks)),
                "per_class_stats": per_dim_class_stats(codes, labels, int(d)),
                "top_examples": top_examples_for_dim(int(d), codes, labels, texts,
                                                     class_value=int(c), n=top_sentences),
                "bottom_examples": top_examples_for_dim(int(d), codes, labels, texts,
                                                        class_value=int(c), n=n_bottom,
                                                        which="bottom"),
                "global_top": top_examples_for_dim(int(d), codes, labels, texts,
                                                   class_value=None, n=top_sentences,
                                                   which="top"),
                "coherence": coh,
                "description": describe_feature(int(d), codes, labels, label_names, texts),
            })
    return {"num_features": int(codes.shape[1]), "features": blocks}


# --- additional qualitative tests -------------------------------------------

def purity_at_ks(codes: np.ndarray, labels: np.ndarray, dim: int,
                 ks: Sequence[int] = (10, 20, 50, 100)) -> Dict[str, Dict]:
    """Purity/entropy at several K (robustness of the single-K purity number).

    A genuinely selective dim stays pure across K; a dim that is pure only at
    K=10 but mixed at K=100 fires strongly on a few exemplars yet broadly
    otherwise. Returns {str(k): class_purity(...)}.
    """
    return {str(int(k)): class_purity(codes, labels, int(dim), top_n=int(k))
            for k in ks}


def per_dim_class_stats(codes: np.ndarray, labels: np.ndarray, dim: int) -> Dict:
    """Mean/std/firing-rate per class for one dim (selectivity at a glance).

    Distinguishes "high mean via few strong firings" from "high mean via many
    weak firings" — the failure mode behind purity/spec mismatches.
    """
    vals = np.asarray(codes)[:, int(dim)]
    out: Dict[str, Dict] = {}
    for c in sorted(np.unique(labels).tolist()):
        v = vals[np.asarray(labels) == c]
        out[str(int(c))] = {
            "mean": float(v.mean()) if len(v) else 0.0,
            "std": float(v.std()) if len(v) else 0.0,
            "frac_nonzero": float((v != 0.0).mean()) if len(v) else 0.0,
            "n": int(len(v)),
        }
    return out


def coherence_score(dense_embs: np.ndarray, codes: np.ndarray, dim: int,
                    top_n: int = 20) -> Dict:
    """Dense-space semantic coherence of a dim's top activators.

    Mean pairwise cosine similarity among the *dense* embeddings of the top-N
    sparse activators. High value = the dim groups sentences the encoder
    already placed nearby (coherent); low value = the dim fires on scattered
    inputs (polysemantic or arbitrary). Uses only geometry, no labels, so it
    complements purity (which needs labels).
    """
    dense = np.asarray(dense_embs, dtype=np.float64)
    acts = np.asarray(codes)[:, int(dim)]
    top_n = int(min(top_n, len(acts)))
    top = np.argsort(acts)[::-1][:top_n]
    vecs = dense[top]
    norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12
    cos = (vecs @ vecs.T) / (norms @ norms.T)
    iu = np.triu_indices(top_n, k=1)
    mean_pair = float(cos[iu].mean()) if len(iu[0]) else 1.0
    return {"mean_pairwise_cosine": mean_pair, "top_n": int(top_n)}


def example_to_dims(codes: np.ndarray, idx: int, top_n: int = 10) -> List[Dict]:
    """Which dims fire for one example (the dim->examples dual view).

    Shows the sparse "parts list" of a single sentence: dims sorted by
    activation. Useful qualitative check that neighbours share dims.
    """
    row = np.asarray(codes)[int(idx)]
    order = np.argsort(row)[::-1][: int(top_n)]
    return [{"dim": int(d), "activation": float(row[d])} for d in order]


def nn_preservation(dense_embs: np.ndarray, codes: np.ndarray,
                    query_idx: Sequence[int], k: int = 5) -> Dict:
    """Do sparse codes preserve dense neighbourhoods? (qualitative retrieval).

    For each query sentence, compares the top-k cosine neighbours in dense
    space vs in sparse-code space (Jaccard overlap). High overlap = the
    sparsification kept the local semantic geometry; low overlap = the
    sparse map re-arranged neighbours. Returns per-query details + mean.
    """
    dense = np.asarray(dense_embs, dtype=np.float64)
    sparse = np.asarray(codes, dtype=np.float64)
    dn = dense / (np.linalg.norm(dense, axis=1, keepdims=True) + 1e-12)
    sn = sparse / (np.linalg.norm(sparse, axis=1, keepdims=True) + 1e-12)
    dense_sim = dn @ dn.T
    sparse_sim = sn @ sn.T
    per_query = []
    for q in [int(i) for i in query_idx]:
        d_rank = np.argsort(dense_sim[q])[::-1][1: k + 1].tolist()
        s_rank = np.argsort(sparse_sim[q])[::-1][1: k + 1].tolist()
        inter = len(set(d_rank) & set(s_rank))
        union = len(set(d_rank) | set(s_rank))
        per_query.append({
            "query": q,
            "dense_nn": [int(i) for i in d_rank],
            "sparse_nn": [int(i) for i in s_rank],
            "jaccard": float(inter / max(union, 1)),
        })
    mean_j = float(np.mean([p["jaccard"] for p in per_query])) if per_query else 0.0
    return {"k": int(k), "mean_jaccard": mean_j, "per_query": per_query}


def dim_pairwise_correlation(codes: np.ndarray, dims: Sequence[int]) -> Dict:
    """Pearson correlation of activations among selected dims (redundancy).

    Near-1 pairs = duplicated atoms (wasted dictionary); near-0 = distinct
    directions. Reported alongside the heatmap so "two dims for one concept"
    is visible instead of hidden.
    """
    dims = [int(d) for d in dims]
    sub = np.asarray(codes, dtype=np.float64)[:, dims]
    if sub.shape[1] < 2:
        return {"dims": dims, "corr": [[1.0]]}
    std = sub.std(axis=0, keepdims=True) + 1e-12
    corr = np.corrcoef(sub, rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0).tolist()
    return {"dims": dims, "corr": corr}


def qualitative_summary(codes: np.ndarray, labels: np.ndarray,
                        top_n: int = 50) -> Dict:
    """Corpus-level qualitative distributions (for histograms).

    Purity@top_n and specialization for EVERY dim — the population the
    hand-picked "top dims per class" are drawn from. A paper that only shows
    the best dims without this background invites cherry-picking concerns;
    saving these arrays lets the notebook plot the full distribution.
    """
    cm = class_mean_activations(codes, labels)
    spec = specialisation_score(cm)
    pur = np.array([class_purity(codes, labels, j, top_n=top_n)["purity"]
                    for j in range(codes.shape[1])], dtype=float)
    return {"purity_at_%d" % int(top_n): pur.tolist(),
            "spec_scores": np.asarray(spec, dtype=float).tolist(),
            "class_means": np.asarray(cm, dtype=float).tolist()}


# --- paper-ready exports -----------------------------------------------------

def _latex_escape(s: str) -> str:
    return (s.replace("\\", "\\textbackslash{}")
             .replace("&", "\\&").replace("%", "\\%")
             .replace("$", "\\$")
             .replace("#", "\\#").replace("_", "\\_")
             .replace("{", "\\{").replace("}", "\\}")
             .replace("~", "\\textasciitilde{}").replace("^", "\\textasciicircum{}"))


def _shorten(s: str, n: int = 110) -> str:
    s = " ".join(str(s).split())
    return s if len(s) <= n else s[: n - 1] + "…"


def save_qualitative_tables(report: Dict, out_prefix: str,
                            label_names: Optional[Dict[int, str]] = None,
                            max_text_len: int = 110) -> Dict[str, str]:
    """Write txt / markdown / LaTeX / csv versions of a feature report.

    out_prefix: path prefix without extension, e.g. "tables/qual_sst2".
    Returns {kind: path}. The .tex table is booktabs-style and can be
    \\input directly into the IEEE paper; .md/.txt are for inspection.
    """
    feats = report.get("features", [])
    paths: Dict[str, str] = {}

    # -- plain text (paste into chat / logs) --
    lines = ["QUALITATIVE DIMENSIONS (top activators + contrast + purity@K + coherence)",
             "=" * 78]
    for b in feats:
        lines.append(f"Class {b['class_name']} | dim {b['dim']} | "
                     f"mean_act {b['mean_act']:.4f} | spec {b['spec_score']:.3f} | "
                     f"purity@50 {b['purity']['purity']:.2f} "
                     f"(maj={b['purity']['majority_class']})")
        pks = ", ".join(f"@{k}:{v['purity']:.2f}"
                        for k, v in sorted(b.get("purity_at_ks", {}).items(),
                                           key=lambda kv: int(kv[0])))
        if pks:
            lines.append(f"  purity curve: {pks}")
        if b.get("coherence") is not None:
            lines.append(f"  coherence (dense cos of top-{b['coherence']['top_n']}): "
                         f"{b['coherence']['mean_pairwise_cosine']:.3f}")
        lines.append("  top (in-class):")
        for s in b.get("top_examples", []):
            lines.append(f"    [{s['activation']:.3f}] (lbl {s['label']}) "
                         f"{_shorten(s['text'], max_text_len)}")
        lines.append("  bottom (contrast, weakest in class):")
        for s in b.get("bottom_examples", []):
            lines.append(f"    [{s['activation']:.3f}] (lbl {s['label']}) "
                         f"{_shorten(s['text'], max_text_len)}")
        lines.append("  global top (unconstrained — mismatch check):")
        for s in b.get("global_top", []):
            lines.append(f"    [{s['activation']:.3f}] (lbl {s['label']}) "
                         f"{_shorten(s['text'], max_text_len)}")
        lines.append("")
    paths["txt"] = out_prefix + "_table.txt"
    with open(paths["txt"], "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    # -- markdown --
    md = ["# Qualitative dimensions", "",
          "| class | dim | mean_act | spec | purity@50 | majority | coherence | top exemplar |",
          "|---|---|---|---|---|---|---|---|"]
    for b in feats:
        top1 = _shorten(b["global_top"][0]["text"], 80).replace("|", "/") if b.get("global_top") else ""
        coh = f"{b['coherence']['mean_pairwise_cosine']:.3f}" if b.get("coherence") else "—"
        md.append(f"| {b['class_name']} | {b['dim']} | {b['mean_act']:.4f} | "
                  f"{b['spec_score']:.3f} | {b['purity']['purity']:.2f} | "
                  f"{b['purity']['majority_class']} | {coh} | {top1} |")
    paths["md"] = out_prefix + "_table.md"
    with open(paths["md"], "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")

    # -- LaTeX (booktabs) --
    tex = ["% auto-generated qualitative table — \\input into the paper",
           "\\begin{tabular}{llllll}",
           "\\toprule",
           "Class & Dim & Spec. & Purity@50 & Coherence & Top exemplar \\\\",
           "\\midrule"]
    for b in feats:
        top1 = _latex_escape(_shorten(b["global_top"][0]["text"], 90)) if b.get("global_top") else "—"
        coh = f"{b['coherence']['mean_pairwise_cosine']:.3f}" if b.get("coherence") else "---"
        cls = _latex_escape(str(b["class_name"]))
        tex.append(f"{cls} & {b['dim']} & {b['spec_score']:.3f} & "
                   f"{b['purity']['purity']:.2f} & {coh} & "
                   f"\\parbox{{0.45\\linewidth}}{{\\scriptsize {top1}}} \\\\")
    tex += ["\\bottomrule", "\\end{tabular}"]
    paths["tex"] = out_prefix + "_table.tex"
    with open(paths["tex"], "w", encoding="utf-8") as f:
        f.write("\n".join(tex) + "\n")

    # -- CSV (every shown dim × every shown sentence, machine-readable) --
    import csv as _csv
    paths["csv"] = out_prefix + "_sentences.csv"
    with open(paths["csv"], "w", newline="", encoding="utf-8") as f:
        w = _csv.writer(f)
        w.writerow(["class", "dim", "rank_kind", "rank", "activation",
                    "label", "text"])
        for b in feats:
            for kind in ("top_examples", "bottom_examples", "global_top"):
                for s in b.get(kind, []):
                    w.writerow([b["class_name"], b["dim"], kind, s["rank"],
                                round(s["activation"], 4), s["label"], s["text"]])
    return paths
