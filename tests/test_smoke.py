"""Tiny synthetic smoke tests — local only, no network, no BERT.

Covers: shapes, loss signs, TopK exact L0, train loop executes (1-2 epochs),
metrics sanity, stratified split, stability matching, config IO.
Run:  python -m pytest tests/test_smoke.py -q   (or python tests/test_smoke.py)
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from src import sparse_model as M
from src import losses as L
from src import data as D
from src import evaluation as E
from src import interpretability as I
from src import training as T
from src import experiments as X
from src.config import ExperimentConfig, save_config, load_config


def test_shapes():
    B, Din, H = 8, 16, 32
    x = torch.randn(B, Din)
    for mdl in [M.CSPINE(Din, H, 0.1), M.TopKSAE(Din, H, 0.1, k=4),
                M.BatchTopKSAE(Din, H, 0.1, k=4), M.DenseAE(Din, H),
                M.MLPCSPINE(Din, H, 0.1, width=32)]:
        mdl.train()
        z, r = mdl(x)
        assert z.shape == (B, H), f"{type(mdl).__name__} z {z.shape}"
        assert r.shape == (B, Din), f"{type(mdl).__name__} r {r.shape}"
        ze = mdl.encode(x)
        assert ze.shape == (B, H)
    print("shapes OK")


def test_losses():
    torch.manual_seed(0)
    z = torch.rand(16, 32)
    r = torch.randn(16, 8)
    t = torch.randn(16, 8)
    tot, mse, asl, psl = L.total_loss(r, t, z, 1.0, 1.0)
    assert float(asl) >= 0 and float(psl) >= 0 and float(mse) >= 0
    # ASL pushes down, PSL subtracts: total == mse+asl-psl
    assert abs(float(tot) - (float(mse) + float(asl) - float(psl))) < 1e-5
    tot2, _, _, _ = L.total_loss_ablated(r, t, z, use_asl=False, use_psl=False)
    assert abs(float(tot2) - float(mse)) < 1e-5
    print("losses OK")


def test_topk_exact_l0():
    m = M.TopKSAE(16, 32, 0.0, k=4)
    m.eval()
    x = torch.randn(6, 16)
    z = m.encode(x).numpy()
    l0 = (np.abs(z) > 1e-9).sum(axis=1)
    assert (l0 == 4).all(), l0
    print("topk L0 OK")


def test_split_and_metrics():
    texts = [f"ex {i}" for i in range(60)]
    labels = np.array([0] * 30 + [1] * 30)
    tr_t, tr_l, va_t, va_l = D.stratified_train_val_split(texts, labels, 0.2, seed=42)
    assert len(tr_t) + len(va_t) == 60 and len(set(va_l)) == 2
    rng = np.random.RandomState(0)
    dense = rng.randn(60, 16).astype(np.float32)
    recon = dense + rng.randn(60, 16).astype(np.float32) * 0.05
    codes = (rng.rand(60, 32) < 0.1).astype(np.float32) * rng.rand(60, 32).astype(np.float32)
    rm = E.reconstruction_metrics(dense, recon)
    sm = E.sparsity_metrics(codes)
    assert rm["cosine_sim"] > 0.9 and sm["sparsity_ratio"] > 0.5
    assert 0 <= sm["mean_l0"] <= 32
    print("split+metrics OK", {k: round(rm[k], 4) for k in ("mse", "cosine_sim", "explained_variance")},
          {k: round(sm[k], 3) for k in ("mean_l0", "sparsity_ratio", "dead_fraction")})


def test_train_loop_tiny():
    cfg = ExperimentConfig(experiment_name="smoke_tiny", embed_dim=16, hidden_dim=32,
                           epochs=2, batch_size=16, seed=42, device="cpu")
    rng = np.random.RandomState(1)
    embs = torch.from_numpy(rng.randn(96, 16).astype(np.float32))
    labs = torch.from_numpy(np.array([0] * 48 + [1] * 48))
    tr = D.make_loader(embs[:80], labs[:80], batch_size=16, shuffle=True)
    va = D.make_loader(embs[80:], labs[80:], batch_size=16, shuffle=False)
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        model, hist, best = T.train_model(cfg, tr, va, d)
        assert len(hist) == 2 and np.isfinite(best)
        z = E.encode_all(model, embs[:10], device="cpu")
        assert z.shape == (10, 32)
    print("train loop OK")


def test_interpret_and_stability():
    rng = np.random.RandomState(2)
    codes = np.abs(rng.randn(100, 24)).astype(float) * (rng.rand(100, 24) < 0.2)
    labels = np.array([0] * 50 + [1] * 50)
    texts = [f"sample {i} {'good' if l else 'bad'}" for i, l in enumerate(labels)]
    rep = I.full_feature_report(codes, labels, {0: "Neg", 1: "Pos"},
                                texts, top_dims_per_class=1, top_sentences=2)
    assert len(rep["features"]) == 2
    # backward-compat keys still present + new qualitative keys
    b0 = rep["features"][0]
    for k in ("purity", "top_examples", "description"):
        assert k in b0, k
    for k in ("bottom_examples", "global_top", "purity_at_ks",
              "per_class_stats"):
        assert k in b0, f"missing new key {k}"
    assert len(b0["top_examples"]) == 2 and len(b0["bottom_examples"]) <= 3
    decA = rng.randn(24, 8)
    decB = decA + rng.randn(24, 8) * 0.01
    ms = X.decoder_direction_similarity(decA, decB)
    assert ms["mean_matched_cosine"] > 0.9, ms
    print("interpret+stability OK")


def test_qualitative_extras():
    """Synthetic-only checks for the new qualitative tests (no network)."""
    import tempfile
    rng = np.random.RandomState(3)
    codes = np.abs(rng.randn(80, 12)).astype(float) * (rng.rand(80, 12) < 0.3)
    labels = np.array([0] * 40 + [1] * 40)
    dense = rng.randn(80, 8).astype(float)
    texts = [f"doc {i} label {l}" for i, l in enumerate(labels)]
    # purity@K curve: keys present, values in [0,1], entropy finite
    pks = I.purity_at_ks(codes, labels, 0, ks=(10, 20, 50))
    assert set(pks) == {"10", "20", "50"}
    for v in pks.values():
        assert 0.0 <= v["purity"] <= 1.0 and np.isfinite(v["entropy"])
    # per-class stats: both classes, firing rates in [0,1]
    pcs = I.per_dim_class_stats(codes, labels, 0)
    assert set(pcs) == {"0", "1"}
    for v in pcs.values():
        assert 0.0 <= v["frac_nonzero"] <= 1.0
    # coherence: cosine in [-1,1]
    coh = I.coherence_score(dense, codes, 0, top_n=10)
    assert -1.0 <= coh["mean_pairwise_cosine"] <= 1.0
    # example->dims: sorted desc, correct length
    e2d = I.example_to_dims(codes, 0, top_n=5)
    assert len(e2d) == 5
    assert all(e2d[i]["activation"] >= e2d[i + 1]["activation"] for i in range(4))
    # NN preservation: jaccard in [0,1], correct k
    nn = I.nn_preservation(dense, codes, [0, 1, 2], k=3)
    assert nn["k"] == 3 and 0.0 <= nn["mean_jaccard"] <= 1.0
    assert len(nn["per_query"]) == 3
    # redundancy matrix: square, unit diagonal
    corr = I.dim_pairwise_correlation(codes, [0, 1, 2])
    assert np.allclose(np.diag(np.asarray(corr["corr"])), 1.0)
    # summary population arrays: right lengths
    summ = I.qualitative_summary(codes, labels, top_n=20)
    assert len(summ["purity_at_20"]) == 12 and len(summ["spec_scores"]) == 12
    # paper-ready exports: 4 files written, tex has tabular
    rep = I.full_feature_report(codes, labels, {0: "Neg", 1: "Pos"}, texts,
                                top_dims_per_class=1, top_sentences=2,
                                dense_embs=dense)
    with tempfile.TemporaryDirectory() as d:
        import os as _os
        paths = I.save_qualitative_tables(rep, _os.path.join(d, "qual"))
        assert set(paths) == {"txt", "md", "tex", "csv"}
        tex = open(paths["tex"], encoding="utf-8").read()
        assert "begin{tabular}" in tex
        # new qualitative plots execute on synthetic data (Agg backend)
        from src import visualization as V
        V.plot_purity_spec_hist(summ["purity_at_20"], summ["spec_scores"],
                                _os.path.join(d, "qh.png"))
        V.plot_per_dim_class_bars([b["dim"] for b in rep["features"]][:4],
                                  np.asarray(summ["class_means"]), ["Neg", "Pos"],
                                  _os.path.join(d, "pb.png"))
        V.plot_dim_coactivation(np.asarray(corr["corr"]), corr["dims"],
                                _os.path.join(d, "co.png"))
        V.plot_coherence_bars(rep["features"][:4], _os.path.join(d, "cb.png"))
        for f in ("qh.png", "pb.png", "co.png", "cb.png"):
            assert _os.path.getsize(_os.path.join(d, f)) > 0, f
    print("qualitative extras OK")


def test_config_io():
    import tempfile
    cfg = ExperimentConfig(experiment_name="io_test", seed=43)
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "c.json")
        save_config(cfg, p)
        c2 = load_config(p)
        assert c2.seed == 43 and c2.experiment_name == "io_test"
    print("config IO OK")


if __name__ == "__main__":
    test_shapes()
    test_losses()
    test_topk_exact_l0()
    test_split_and_metrics()
    test_train_loop_tiny()
    test_interpret_and_stability()
    test_qualitative_extras()
    test_config_io()
    print("ALL SMOKE TESTS PASSED")
