# Colab FAST-run findings — plot audit + results reading (Agent 1)

Source: `final_research.zip` (FAST smoke: H=256, 3 epochs, 1 seed, train 2000 /
val 500 / locked TEST 500, sst2+agnews only). All numbers below are
**preliminary smoke evidence, not paper claims**. FULL run (H=1024, 30 epochs,
3 seeds, 5 domains) must confirm or overturn each point. Fixed figures:
`reports/figures/*_FIXED.png` (regenerated from the zip's logs+metrics, no
retraining). `*_SYNTHETIC_check.png` validate the new acthist/heatmap code
paths (real codes are not shipped in the zip; they redraw on rerun).

## 1. Verdicts on the suspected irregularities

| # | Observation | Verdict | Evidence |
|---|-------------|---------|----------|
| 1 | sst2 vs agnews training curves look identical | **Fine — not a bug.** Same hypers, same init scale, 3 epochs → similar shape is expected. CSVs differ numerically (ep-1 val sparsity 0.308 sst2 vs 0.406 agnews; ep-3 MSE 0.131 vs 0.117). The old plot hid this by drawing both series in the same blue with no legend. | `logs/cspine_*_training_log.csv` |
| 2 | Activation-histogram spike at 1.0 | **Fine — expected PSL signature.** PSL maximizes per-unit batch-max; the CappedReLU ceiling piles that mass at exactly 1.0. The new acthist annotates the saturated share explicitly. | `plots/acthist_*.png` (old), `visualization.plot_activation_hist` (new) |
| 3 | MiniLM run densifies (sparsity 0.37→0.22, ASL rising) | **Real issue, explained.** FAST smoke forces HIDDEN=256 on a D=384 encoder → undercomplete (H<D), violating C-SPINE's overcompleteness assumption, plus only 3 epochs. `build_model` now emits a `UserWarning` for H<D instead of failing silently. MiniLM needs H≥512 in FULL. | `logs/enc_all-MiniLM-L6-v2_training_log.csv` |
| 4 | sst2 heatmap/interp labels lowercase (`negative`) vs hardcoded (`Negative`) | **Fine.** `_refresh_label_names` replaced them with the HF `ClassLabel` native names (`glue/sst2` uses lowercase). agnews native names are capitalized, hence the inconsistency. Display-only; ids/scores unaffected. | `cspine/data.py::_refresh_label_names`, `metrics/interp_sst2.json` |
| 5 | `very_high` sweep: min L0 = 0.0 | **Real behavior, must be reported.** Some TEST inputs encode to the all-zero vector (recon = decoder bias). Collapse signature at extreme sparsity (λ=8, 90% zeros, 27.7% dead). Not a crash — a finding. | `metrics/sparsity_sweep.json` (very_high) |

## 2. Preliminary scientific reading (single seed — hypotheses, not conclusions)

- **H1 (preserve info):** sparse≈dense (sst2 .800 vs .792; agnews .830 vs .822). With n=500 (≈±2pp noise) this is *consistency*, not a win — do not claim sparse>dense. FULL multi-seed CIs decide. Note the protocol asymmetry (StandardScaler on dense, none on sparse) confounds small deltas.
- **Unexpected: recon < sparse** on both sets (sst2 .776 < .800; agnews .774 < .830), against the paper's recon≈sparse and against theory (recon is a linear read-out of the codes). Suspects: protocol asymmetry + small-n noise. FULL retest required; if it persists, it is a real puzzle, not a success.
- **H6 warning: PCA-64 beats sparse on both sets** (sst2 .820; agnews .862). At smoke scale (H=256) plain PCA is competitive — the honest headline is "no evidence yet of beyond-compression value." FULL H=1024 may differ.
- **H2 plateau: genuine.** Sweep acc .808/.800/.812 (low/med/high) then collapse .734 (very_high) while MSE rises monotonically .051→.204. Downstream survives 6%→78% zeros, then breaks. `recon_vs_l0` agrees: MSE 0.136 at L0~63 → 0.094 plateau beyond L0~90.
- **Mechanisms: fidelity vs utilization trade.** TopK val MSE 0.090 < C-SPINE 0.131 at fixed L0=32 — but 219/256 dims (86%) never fire on TEST. C-SPINE: 0 dead. DenseAE: best MSE 0.040 but 57% dead ReLUs and its "62% sparse" is incidental, not learned. Report both sides; neither arm wins outright.
- **Transfer: partially reusable, clearly dataset-tuned.** Off-diagonal MSE 0.147/0.150 vs 0.103/0.118; cosine 0.72/0.68 vs 0.83/0.80. Asymmetric L0 shift: sst2-model fires *more* on agnews (133 vs 100); agnews-model fires *less* on sst2 (72 vs 85).
- **Interpretability: agnews strong, sst2 honestly weak.** agnews purities to 1.00 (dim 186 World), 0.98 (dim 30 Sports), spec to 0.94. sst2 purities 0.64–0.94 with an instructive mismatch: dim 159 was selected for *negative* yet its top-50 activators are 88% *positive* — textbook discriminative≠correlated. sst2 heatmap is near-uniform: sentiment looks distributed, not localized. Negative result retained.
- **Normalization audit confirmed:** raw CLS norm mean 14.44 ± 0.44 → the paper's "unit length" claim is false in the legacy setup, and σ=0.1 noise is only ~0.7% relative — weak denoising pressure. The `normalize_inputs` ablation in FULL is now well-motivated.
- **Stability (H4): untested** — single seed by FAST design; `stability.json` records the deferral explicitly.

## 3. Plot fixes delivered (`cspine/visualization.py`, same API + 4 new figures)

- Training curves: distinct blue/red axes-matched colors, markers, grid, combined legend, best-VAL-checkpoint star + annotation (old: both series same blue, no legend).
- acthist: log-y zero panel (old: 80k zero bar crushed everything), ceiling-spike annotation, stats suptitle (L0/sparsity/dead/saturated).
- Heatmap: TRUE dim ids on x-axis (old: ranks 0–29, unreferenceable), contrast fitted to data range (old: fixed 0–1 washed out when floor≈0.3), optional specialization stars.
- Tradeoff: sparsity % per point, dense-baseline line, points sorted by L0, no clipping.
- NEW: `plot_recon_vs_l0` (bins were collected, never drawn), `plot_downstream_bars` (headline + per-class F1), `plot_transfer_matrix`, `plot_mechanism_bars`.
- FIXED figures regenerated from zip data: `reports/figures/*_FIXED.png`. acthist/heatmap need live codes → redraw via the Colab replot cell (pasted in chat) or a full rerun.

## 4. What FULL must confirm

Multi-seed CIs for sparse-vs-dense and recon-vs-sparse; PCA/RP at H=1024; MiniLM with H≥512; trec/scicite→arxiv slots; loss ablations beyond `full`; H4 matching; human-rated interpretability (current purity = correlation only).
