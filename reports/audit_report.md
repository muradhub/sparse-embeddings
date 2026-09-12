# C-SPINE Audit Report — Reproduction & Paper-vs-Code Verification (Agent 1)

**Date:** 2026-09-11 · **Scope:** local directory only · **Commit:** pre-extension codebase
(`config.py`, `extracting_embeddings.py`, `train_cspine.py`, `evaluate.py`,
`inspect_dimensions.py`, `sparse_embeddings_research.pdf`)

> Method: treat paper and implementation as two independent sources of truth.
> Every claim below was checked by reading the files listed above. No full
> training was run locally (compute constraint); verification is static +
> tiny synthetic shape checks (see `tests/test_smoke.py`).

## 1. Pipeline reconstruction (17 audit points)

| # | Item | Paper claim | Implementation | Agree? |
|---|------|-------------|----------------|--------|
| 1 | Representations from BERT | frozen `bert-base-uncased`, final-layer `[CLS]`, 768-d | `extracting_embeddings.py:42-48,98-115,238-249`: `AutoModel`, `cls_pool(last_hidden_state)`, `EMBED_DIM=768` | ✅ |
| 2 | Pooling | CLS only | Default CLS; `mean_pool()` + `pick_pooling()` exist for MiniLM/RoBERTa/MPNet/DistilBERT fallback (`extracting_embeddings.py:51-69`). Paper never mentions fallback | ⚠️ minor undocumented |
| 3 | Normalized? | §III-A: "every vector is L2 normalized to unit length … consistent with SPINE" | **No normalization anywhere.** `extract_embeddings()` returns raw `.float().numpy()`; `train_cspine.py`/`evaluate.py` never normalize; only `StandardScaler` in LogReg probe | ❌ **major discrepancy** |
| 4 | Sparse encoder | Linear 768→1024 + CappedReLU | `train_cspine.py:92-96` exact match | ✅ |
| 5 | Decoder | Linear 1024→768, no activation | `train_cspine.py:99` exact match | ✅ |
| 6 | Activation | CappedReLU [0,1], min(max(0,v),1) | `CappedReLU.forward: x.clamp(0,1)` + eq.(3) in paper | ✅ |
| 7 | Sparsity objective | ASL = mean activation →0; PSL = −mean batch-max (maximize peaks); λ1=λ2=1.0 | `average_sparsity_loss=z.mean()`, `peak_sparsity_loss=z.max(0).mean()`, `total = mse + λ1·asl − λ2·psl`, defaults 1.0 (`train_cspine.py:143-175`, `config.py:26-27`) | ✅ (notation differs: code PSL positive then subtracted; paper PSL negative; mathematically identical) |
| 8 | Reconstruction objective | Eq.(5) L_recon = ‖x−x̂‖²₂ | Code uses `F.mse_loss` = mean over B·D, not sum. Proportional (differs by 1/(B·D) factor) — matters for λ scaling interpretation | ⚠️ formulation drift |
| 9 | Noise/denoising | x̃=x+ε, ε∼N(0,σ²I), σ=0.1; recon vs *clean* target | `CSPINE.forward: x+randn*noise_std` only in `training=True`; `mse(recon, clean)` (`train_cspine.py:114-123,138-140`) | ✅ |
| 10 | Training procedure | Adam lr=1e-3, batch 64, 30 epochs, per-epoch val logging | `Adam(lr=C.LR)`, `BATCH_SIZE=64`, `EPOCHS=30`, CSV log (`train_cspine.py:320-382`) | ✅ |
| 11 | Dataset splits | SST-2 train/val (HF glue/sst2); AG News train/test (120k/7.6k) | `nyu-mll/glue:sst2` train+validation; `fancyzhx/ag_news` train+test (`extracting_embeddings.py:166-221`). Sizes in docstring match HF | ✅ |
| 12 | Checkpoint selection | "30 epochs with per-epoch validation logging" (criterion unstated) | **Best val MSE** (`train_cspine.py:353-365`). BUT val split = `test` for AG News, `val` for SST-2 (`train_cspine.py:308`) — see #11 leakage below | ⚠️ underspecified + leakage |
| 13 | Downstream classifier | "logistic regression probes" | `LogisticRegression(max_iter=1000, C=1.0, lbfgs, random_state=42)` (`evaluate.py:153-160`). **Scaling undocumented:** `StandardScaler` for dense/recon, none for sparse (`evaluate.py:148-160,360-375`) | ⚠️ scaling protocol not in paper |
| 14 | Existing eval metrics | Accuracy, sparsity %, active dims, val MSE (Table II) | `evaluate.py`: accuracy + `classification_report` (printed, not saved), sparsity fraction, mean nonzero, MSE. Macro-F1 computed by sklearn but **never extracted**; only accuracy persisted to CSV | ⚠️ narrow; F1/precision/recall lost |
| 15 | Random seeds | Unstated | `torch.manual_seed(42); np.random.seed(42)` hard-coded in all three scripts; single seed, no std/CI | ❌ no multi-seed |
| 16 | Hardware | "CUDA when available; ~15 min SST-2 / 40 min AG News on single GPU" | `DEVICE='cuda' if available else 'cpu'`; embedding docstring: BERT+GPU 25-35 min, BERT+CPU 2-4 h, MiniLM+CPU 30-50 min. No GPU model named | ⚠️ vague, not reproducible |
| 17 | Reported numbers | Table II + §V-B dynamics | Dense 86.24→sparse 81.08 (Δ5.16) SST-2, 89.28% sparse, 109.8/1024 active, MSE 0.0216; AG News 90.30→87.96 (Δ2.34), 92.76%, 74.2 active, MSE 0.0295; epoch-1→30 trajectories; recon ≈ sparse (Δ<0.6pp) | ⏳ cannot confirm locally (no checkpoints/logs in dir); numbers internally consistent (active = 1024·(1−sparsity): 1024·0.1072≈109.8 ✓; 1024·0.0724≈74.1 ✓) |

## 2. Critical weaknesses / inconsistencies

1. **Normalization false claim (major).** Paper §III-A/IV-B assert L2-unit normalization; code omits it. Effect: MSE scale, noise SNR (σ=0.1 relative to raw CLS norms ~10-30, not unit norms), and cosine geometry all differ from description. Fix: `normalize_inputs` flag (default `False` = legacy reproduction; `True` = paper claim) + ablation.
2. **Test-set leakage (major).** `train(dataset)`: `val_split='test'` for AG News → best-epoch selection peeks at the reporting set; `evaluate_dataset()` reports on the same split. SST-2 uses val for both selection and reporting with no held-out test (partly excused: official test labels withheld, but then a train/val/test carve-out of HF-train is required). No "untouched test" exists. Fix: HF-train → stratified train/val (90/10); HF-val/test → locked TEST, evaluated once (§5A).
3. **Single seed, no uncertainty (major).** All headline deltas (5.16pp, 2.34pp) are single-run point estimates. No std/CI; cannot tell if AG-News "2.3pp drop" is noise. Fix: ≥3 seeds (42,43,44), mean±std±95%CI.
4. **Thin metrics (moderate).** MSE-only recon; accuracy-only downstream (imbalanced sets need macro-F1); no cosine/explained-variance/NMSE/error-distribution; no L0 median, dead-feature counts, utilization, recon-vs-L0. Heatmap-only interpretability; Figs.2-5 are literally "Enter Caption" placeholders; Fig.6 is schematic, not measured data. Fix: full metric suite (§5B-C, §9).
5. **One sparsity operating point (moderate).** Single (λ1,λ2)=(1,1) cannot answer "how sparse can we go?". Fix: 4-level sweep (low/medium/high/very-high) + TopK/BatchTopK comparison to test if behaviour is SPINE-specific or generic sparse coding (§5D, §6).
6. **No information-preservation baselines (moderate).** Without PCA/RP/dense-AE at matched dims, cannot claim sparse coding adds value beyond compression. Fix: §8 baselines under identical probe protocol.
7. **Stale/placeholder artefacts (minor).** `inspect_dimensions.py:26` usage says `python 04_inspect_dimensions.py`; `evaluate.py` CSV drops per-class F1; `config.py` MiniLM path requires manual `EMBED_DIM` edit (error-prone). Fixed in `cspine/` package.
8. **Loss-scale ambiguity (minor).** Sum-vs-mean MSE changes effective λ weighting vs SPINE; document and keep MSE-mean with a note.

## 3. What was reproduced vs assumed

- Reproduced statically: architecture, CappedReLU semantics, DAE noise placement, ASL/PSL sign convention, optimizer/epochs/batch, CLS pooling, dataset IDs, probe type.
- Could NOT reproduce locally: any Table II number (no embeddings/checkpoints/logs shipped; full BERT+SAE training exceeds local budget by design). Numbers are arithmetically self-consistent but unverified.
- Assumed (flagged, not trusted): GPU type/timings, epoch-1 intermediate values, Fig.6 patterns, MiniLM viability.

## 4. Extension requirements derived from audit

Rigorous study must: (a) add explicit train/val/TEST + VAL-only checkpointing; (b) test normalization as ablation; (c) sweep sparsity; (d) ablate pooling/norm/latent-dim/mechanism/loss/noise/capacity; (e) report MSE+cosine+explained-variance+NMSE+error-dist + L0/dead/utilization/recon-vs-L0; (f) downstream dense/recon/sparse/PCA/RP/dense-AE with acc+macro-F1+per-class-F1+precision+recall; (g) feature-level interpretability with purity + stability across seeds (Hungarian cosine matching, activation corr, top-example overlap); (h) 5-domain portfolio + cross-domain transfer + 2 extra encoders; (i) H1-H6 with pre-registered evidence mapping; (j) Colab-executable `research_full_run.ipynb` + `final_research/` ZIP with codebase+notebook+configs+results+metrics+plots+tables+checkpoints-metadata+logs+reports+README+requirements.

## 5. Hypotheses (pre-registered, tested by notebook)

H1 sparse retains most downstream info at far fewer active dims · H2 flat sparsity-fidelity plateau exists · H3 C-SPINE more coherent than dense-AE · H4 features stable across seeds · H5 useful across domains · H6 beyond PCA/RP compression. Evidence mapping in `cspine/experiments.py:HYPOTHESIS_EVIDENCE`.

## 6. Provenance

- Paper: `sparse_embeddings_research.pdf` (7 sections, 13 refs, Table I hyperparams, Table II results).
- Code: root `*.py` (5 files, ~1300 lines) — frozen as legacy; new work lives in `cspine/` without modifying legacy files.
- New configs: `configs/base_sst2.json`, `configs/base_agnews.json`, `configs/fast_smoke.json`.
- Compute: local = static checks + synthetic micro-runs only; full runs = Colab notebook.
