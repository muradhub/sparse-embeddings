# C-SPINE — Rigorous Replication & Extension (Agent 1)

Reproducible, scientifically hardened version of the C-SPINE sparse-embedding
study. Legacy implementation frozen as shipped; new work lives in `src/`.

## Layout

- Legacy (frozen): `config.py`, `extracting_embeddings.py`, `train_cspine.py`, `evaluate.py`, `inspect_dimensions.py`, `sparse_embeddings_research.pdf`
- New package: `src/` (`data`, `embeddings`, `sparse_model`, `losses`, `training`, `evaluation`, `interpretability`, `ablations`, `generalization`, `visualization`, `experiments`, `config`)
- Configs: `configs/base_sst2.json`, `configs/base_agnews.json`, `configs/fast_smoke.json`
- Reports: `reports/audit_report.md` (paper-vs-code audit), `reports/final_report.md` (paper-ready report)
- Notebook: `research_full_run.ipynb` — independent Colab run (upload + `sparse_embeddings.zip`, Run All)
- Tests: `tests/test_smoke.py`

## Local (low-compute) use only

```powershell
pip install -r requirements.txt
python tests/test_smoke.py
```

Local machine: static checks, imports, shape checks, synthetic micro-runs only.
No full BERT/SAE training locally by design.

## Full experiments (Colab)

1. Zip this directory as `sparse_embeddings.zip` (must contain `src/`, configs, legacy scripts, notebook).
2. In Colab (GPU): upload `research_full_run.ipynb` + `sparse_embeddings.zip`, Run All.
3. `FAST_MODE=True` = smoke (~15 min); set `False` for the full paper run.
4. Output: `final_research.zip` with `codebase/`, `notebook/`, `configs/`, `results/`, `metrics/`, `plots/`, `tables/`, `checkpoints_or_metadata/`, `logs/`, `reports/`, `README.md`, `requirements.txt`.

## Key fixes vs. legacy

- Train/val/LOCKED-TEST with VAL-only checkpointing (legacy peeked at the reporting split).
- `normalize_inputs` flag (paper claimed L2 norm; code never did — now an ablation).
- Multi-seed mean/std/CI; full recon (MSE/cosine/explained-var/NMSE) + sparsity (L0/dead/utilization) + downstream (acc/macro-F1/per-class) suites.
- Sparsity sweep, TopK/BatchTopK/dense-AE ablations, PCA/RP baselines, stability (Hungarian matching), 5-domain + cross-domain + multi-encoder generalization, H1–H6 pre-registration.

See `reports/audit_report.md` for the full paper-vs-code verification.
