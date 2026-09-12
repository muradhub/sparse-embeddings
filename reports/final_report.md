# C-SPINE Rigorous Replication & Extension — Scientific Report (Agent 1)

## Abstract

We reproduce, audit, and extend C-SPINE (Contextual Sparse and Interpretable
Neural Embeddings), a denoising sparse autoencoder that maps frozen BERT
`[CLS]` vectors (768-d) to overcomplete sparse codes (1024-d, CappedReLU
[0,1]) under a reconstruction + average-sparsity − peak-sparsity objective.
Static audit of the shipped codebase against the manuscript finds the core
architecture, denoising procedure, and loss sign convention faithfully
implemented, but also two major discrepancies: (i) the paper claims L2-unit
input normalization while the code never normalizes, and (ii) checkpoint
selection peeks at the reporting split (AG-News test used for both selection
and reporting; SST-2 validation used for both), so no untouched test set
exists. All headline numbers are single-seed point estimates without
uncertainty. We therefore rebuild the study around a reproducible package
(`cspine/`), strict train/val/LOCKED-TEST discipline with VAL-only
checkpointing, multi-seed mean±std±95%CI, a four-level sparsity sweep,
mechanism/loss/representation/architecture ablations (incl. TopK, BatchTopK,
dense autoencoder), PCA/random-projection baselines, feature-level
interpretability with purity and seed-stability analysis (Hungarian decoder
matching, activation correlation, top-example overlap), a five-domain
portfolio (sentiment, news/topic, long reviews, question/intent, scientific),
cross-domain transfer, and a second/third encoder family. Full experiments
are designed for Google Colab (`research_full_run.ipynb`, FAST smoke vs FULL
paper modes); local work validated only shapes, losses, and micro-runs on
synthetic data. No full BERT training was run locally, so legacy Table II
values are reported as unverified claims, not confirmations. The deliverable
is a paper-ready experimental foundation that can answer what C-SPINE is,
whether code matches manuscript, how far sparsity preserves information,
which components matter, whether features are interpretable/stable, whether
the method generalizes, and what its limits are.

## 1. Introduction

Dense contextual embeddings (BERT [1]) are effective and opaque: every one
of 768 dimensions is nonzero, and no single dimension carries an assignable
meaning. Sparse coding [7] offers a classical remedy — force most dimensions
to zero so active ones can be inspected, counted, and named. SPINE [2]
demonstrated this for static word vectors (Word2Vec/GloVe, ~90% sparse) with
a denoising autoencoder, CappedReLU, and a two-term sparsity objective
(average sparsity + lifetime/peak sparsity). C-SPINE ports that idea to
sentence-level contextual vectors, a harder distribution (context-dependent,
richer covariance) but the practically relevant one.

Agent 1's remit (this report) is the original C-SPINE contribution itself:
verify it, harden its evaluation, and test its generalizability. Knowledge-
graph extensions belong to Agent 2 and are out of scope. Our primary
objective is rigor and reproducibility, not maximal scores.

Pre-registered hypotheses (evidence mapping in `cspine/experiments.py`):

- **H1.** Sparse codes retain most downstream information at far fewer active dims.
- **H2.** A sparsity plateau exists where fidelity degrades slowly vs. active-count reduction.
- **H3.** The C-SPINE objective yields more coherent features than a non-sparse autoencoder.
- **H4.** Features recur across random seeds.
- **H5.** The method remains useful across datasets/domains.
- **H6.** The effect exceeds ordinary dimensionality reduction (PCA/RP).

## 2. Related work

**Static vs. contextual embeddings.** Word2Vec [3]/GloVe [4] assign one
vector per type; BERT [1]/RoBERTa [5]/GPT-2 [6] condition on context via
self-attention and masked-LM pretraining. The `[CLS]` final hidden state
serves as a sentence representation and is C-SPINE's input.

**Sparse coding & dictionary learning.** Olshausen & Field [7] established
overcomplete sparse bases for V1; dictionary learning, ICA, and NMF extend
the theme. Overcompleteness (H>D, here 1024>768) is deliberate: extra atoms
let signal spread thinly so per-example subsets differ while each stays sparse.

**SPINE [2].** Direct parent: denoising AE on static embeddings, CappedReLU
[0,1] (nonnegativity aids interpretation; capping stops PSL from exploding
one unit), ASL (mean→0) vs. PSL (batch-max→1) tension. Reported ~90% sparse
with 2–5pp downstream drops.

**SAEs for LLM interpretability.** Bricken et al. [8] train SAEs on internal
activations for mechanistic understanding. C-SPINE differs in target: output
sentence vectors for downstream reuse, not internal circuits. Modern TopK/
BatchTopK SAEs [Gao et al. 2024] replace soft penalties with hard sparsity;
we include both as ablations to test whether C-SPINE behaviour is
objective-specific or generic sparse coding.

**Probing [9].** Linear (logistic-regression) probes test whether class
information is linearly accessible in frozen features. We adopt them
throughout, with identical protocol across dense/reconstructed/sparse/baseline
arms.

Novelty positioning: C-SPINE's claim is application of the SPINE-style
objective to contextual sentence vectors with probing + specialization
analysis. We do not claim architectural novelty over SAEs/TopK; the
contribution, if supported, is empirical (sparsity–fidelity tradeoffs,
stability, cross-domain reuse).

## 3. Methodology

**Extraction.** Frozen encoder → tokenize (max 128, >99% coverage claimed) →
final-layer `[CLS]` (or mean pooling for MiniLM-family encoders) → optional
L2 normalization (flag; legacy default OFF to reproduce code, ON to test the
paper claim) → cache `.npy`. BERT never trains jointly.

**Model.** Encoder: Linear(D→H)+CappedReLU; decoder: Linear(H→D). Variants:
TopKSAE (exact per-example L0=k), BatchTopKSAE (exact batch budget k·B,
variable per-example L0), DenseAE (ReLU, MSE-only, no-sparsity control),
MLPCSPINE (2-layer encoder capacity control). Noise: x̃=x+N(0,σ²I), σ=0.1,
train-time only; reconstruction targets the clean x.

**Loss.** L = MSE + λ1·ASL − λ2·PSL, λ1=λ2=1.0 default; MSE is batch-mean
(`F.mse_loss`), ASL=mean(z), PSL=mean(batch-max(z)). Ablations toggle each
term (no-ASL/no-PSL/no-noise/MSE-only/L1-instead) to attribute effects.

**Training & selection.** Adam lr=1e-3, batch 64, 30 epochs (FULL) / 3
epochs smoke (FAST). HF-train → stratified 90/10 train/val (class ratios
preserved, seeded); HF-val/test → LOCKED TEST, evaluated once. Best
checkpoint by VAL MSE only. Multi-seed (42,43,44 FULL; 42 FAST) with
mean/std/95%CI (normal approx; small-n caveat noted).

## 4. Implementation details

Legacy files (`config.py`, `extracting_embeddings.py`, `train_cspine.py`,
`evaluate.py`, `inspect_dimensions.py`) are frozen untouched. New package
`cspine/` provides: `data` (registry, stratified splits, loaders, L2 norm),
`embeddings` (pooling, batched extraction, caching), `sparse_model`
(CappedReLU/CSPINE/TopK/BatchTopK/DenseAE/MLP + `build_model` factory),
`losses` (MSE/ASL/PSL/L1 + ablated total), `training` (seeded loops,
VAL-only checkpointing, CSV logs), `evaluation` (recon/sparsity/downstream/
baselines/aggregation), `interpretability` (purity, top/bottom examples,
distributions, template descriptions), `ablations` (config generators),
`generalization` (transfer pairs, encoder table), `visualization` (Agg
matplotlib), `experiments` (H1–H6 registry, Hungarian stability matching),
`config` (serialisable `ExperimentConfig`, sweep/seed helpers, JSON IO).
No hard-coded absolute paths; every run is reproducible from a JSON in
`configs/` recording seed, dataset, encoder, pooling, dims, hyperparams,
sparsity settings, checkpoint, and metrics. Local validation: `tests/
test_smoke.py` (shapes, loss identity total=MSE+ASL−PSL, TopK exact-L0,
split stratification, 2-epoch train loop, metric sanity, Hungarian
self-match ≈1.0, config round-trip) — all pass; plus a 2-dataset synthetic
integration (train→eval→baselines→plots→transfer→stability) — passes.

## 5. Experimental setup

**Modes.** FAST (default in notebook): 2000 train cap, 500 val/test caps,
maxlen 64, H=256, 3 epochs, 1 seed, core datasets only (~10–20 min T4).
FULL: full data, maxlen 128, H=1024, 30 epochs, 3 seeds, all datasets,
all ablations (hours; overnight).

**Splits.** sst2: HF-train→train/val, HF-validation→TEST (official test
withheld). agnews/imdb/trec: HF-train→train/val, HF-test→TEST. scicite:
HF-train→train/val, HF-validation→TEST. TEST never used for training,
checkpointing, or hyperparameter choice.

**Encoders.** Primary `bert-base-uncased` (768, CLS). Generalizability:
`all-MiniLM-L6-v2` (384, mean), `distilbert-base-uncased` (768, CLS).

**Checkpointing.** Best VAL MSE; optimizer state + config snapshot saved;
test-time `encode()` is deterministic, noise-free.

## 6. Datasets

| Dataset | HF id | Domain | Classes | Train pool | TEST | Role |
|---|---|---|---|---|---|---|
| sst2 | nyu-mll/glue:sst2 | sentiment/movie | 2 | 67,349 | 872 (val) | original |
| agnews | fancyzhx/ag_news | news/topic | 4 | 120,000 | 7,600 | original |
| imdb | stanfordnlp/imdb | sentiment/long | 2 | 25,000 | 25,000 | length shift |
| trec | cmap/go_trec:coarse | question/intent | 6 | 5,452 | 500 | pragmatic shift |
| scicite | allenai/scicite | scientific/cite-intent | 3 | ~8k | val split | scientific shift |

Portfolio covers the brief's sentiment / news-topic / scientific /
question-intent / distinct-domain slots. Licensing: all public HF datasets;
feasibility guarded per-dataset try/except with explicit SKIP records.

## 7. Baselines (information preservation, §8)

Under one LogReg protocol (lbfgs, C=1.0, maxiter 1000; StandardScaler for
dense-like arms, none for [0,1] sparse arms — documented, with scaled-sparse
variant noted): (1) dense BERT (upper bound), (2) reconstructed dense
(isolates recon vs. sparsification loss), (3) sparse codes (main), (4) PCA
matched to k=64 comps, (5) Gaussian random projection k=64, (6) DenseAE
codes (same capacity, no sparsity). H6 is decided by (3) vs. (4–6), not by
(3) vs. (1) alone.

## 8. Metrics

**Reconstruction (§5B):** MSE, cosine similarity (mean per-example),
explained variance 1−Var(err)/Var(orig), normalized MSE (MSE/mean‖x‖²),
per-example MSE distribution (mean/std/p50/p90/p99), recon-vs-L0 binned
table. **Sparsity (§5C):** per-example L0, mean/median/std/min/max L0,
sparsity ratio (fraction zeros), activation frequency per feature, dead
features + fraction, utilization (>1% active), activation histograms,
recon-vs-L0. **Downstream (§7):** accuracy, macro-F1 (primary for
imbalance), micro-F1, macro precision/recall, per-class F1. **Sweep (§5D):**
low (λ=0.1) / medium (1.0) / high (3.0) / very-high (8.0) → sparsity↔MSE,
sparsity↔accuracy, sparsity↔interpretability plots. **Interpretability
(§9):** top/bottom examples per dim, activation distributions, class purity
(top-50 majority fraction + entropy), template descriptions; vocabulary
discipline: discriminative ≠ correlated ≠ coherent ≠ interpretable.
**Stability (§10):** decoder-direction Hungarian cosine, activation
correlation of matched pairs, top-N example Jaccard, label consistency.

## 9. Results (legacy claims vs. new evidence)

**Legacy Table II (UNVERIFIED single-run claims — no checkpoints/logs
shipped, no local re-run per compute constraint):** SST-2 dense 86.24 →
sparse 81.08 (Δ5.16pp), recon 81.65, 89.28% sparse, 109.8/1024 active, MSE
0.0216; AG-News dense 90.30 → sparse 87.96 (Δ2.34pp), recon 88.04, 92.76%,
74.2 active, MSE 0.0295. Internal arithmetic checks (active=1024·(1−
sparsity)) pass. Dynamics (§V-B: epoch-1→30 MSE 0.025→0.0216 SST-2,
0.034→0.0295 AG-News; sparsity 83.6→89.3 / 89.4→92.8; fastest gains epochs
1–10) are plausible but unconfirmable here. Recon≈sparse (Δ<0.6pp) suggests,
if true, loss stems from compression rather than sparsification — a claim
the new recon arm retests with CIs.

**New evidence status:** pending Colab execution. The notebook emits
`metrics/*_test.json` (full metric dicts, not headlines), `tables/
summary.csv`, `plots/*`, `status.json` (per-experiment PASS/SKIP with
reasons), and hypothesis verdicts with thresholds (e.g., H1: drop<8pp at
>70% sparse). FAST smoke is expected to show the pipeline executes and
tradeoff slopes have the right sign on tiny data; FULL decides the science.
Negative/inconclusive outcomes are retained by design (verdicts print
INCONCLUSIVE/FAIL rather than hiding).

## 10. Ablations (design + expected reading)

- **Representation:** CLS vs. mean pooling; raw vs. L2-normalized inputs
  (direct test of the paper's false normalization claim); H∈{256,512,1024,
  2048} (overcompleteness 0.33–2.67×). Expectation: normalization should
  shrink raw-norm variance (observed mean raw norm ~O(10) in pilots) and
  shift the λ frontier; H<D should hurt accuracy (compression + sparsity).
- **Mechanism:** C-SPINE vs. TopK(k=32/64/128) vs. BatchTopK vs. DenseAE.
  Decisive question: is behaviour SPINE-specific or generic sparse coding?
  TopK fixes L0 exactly, isolating the tradeoff from penalty tuning.
- **Loss:** full / no-ASL / no-PSL / no-noise / MSE-only / L1-instead.
  Expect no-PSL → more dead features; no-ASL → denser codes; no-noise →
  slightly better MSE but less robust (to be measured, not assumed).
- **Architecture:** H sweep + linear vs. 2-layer MLP encoder.
All outcomes reported; cherry-picking prohibited by `status.json` + saved
full metric dicts.

## 11. Generalization

**Portfolio (H5):** five domains above, same framework, per-domain
recon/sparsity/downstream. **Cross-domain (frozen SAE from A evaluated on
B, no retraining):** full ordered transfer matrix (incl. diagonals) with
MSE/cosine/L0/dead-fraction; asks whether atoms are reusable or
dataset-specific. **Encoder (H-enc):** MiniLM/DistilBERT under matched
protocol; overclaim guard — "encoder-agnostic" only if sparsity/recon/
downstream patterns replicate, otherwise reported as encoder-sensitive.
 colab-budget guard: heavy pairs run in FULL only; skips explicit.

## 12. Interpretability & stability

Beyond heatmaps: per-dim top/bottom examples, activation histograms,
purity@50 + entropy, class-conditional means, specialization score
(top1−top2)/(top1+ε), template descriptions. Purity is reported as
correlation evidence, never as proof of human interpretability. Stability:
≥2 seeds → Hungarian-matched decoder cosine (frac>0.7), matched activation
correlation (frac>0.5), top-20 Jaccard. An interpretable representation
should partially survive re-initialization; total dependence on one seed
would refute H4 even if single-run heatmaps look clean. FAST (1 seed)
explicitly defers H4 to FULL.

## 13. Limitations

Fixed λ/σ grid (no full search); English classification only (no NLI,
coreference, QA — likely harder for sparse codes); qualitative
interpretability without human ratings or information-theoretic measures;
single-layer linear encoder/decoder (capacity ceiling); BERT-base scale
only (larger encoders untested); 95%CI uses normal approx at n=3 (wide;
report std and range alongside); HF dataset versions may drift (pin
revisions in Colab via `datasets` revision hashes where available).

## 14. Threats to validity

Test leakage in legacy results (selection on reporting split) inflates
reported accuracy and invalidates naive before/after comparisons — hence
the locked-TEST redesign. Single-seed headlines risk noise-chasing;
multi-seed aggregation mitigates but n=3 remains small. Probe scaling
choices (Scaler on dense, none on sparse) can shift deltas by ~1pp;
protocol is fixed and documented, with a scaled-sparse sensitivity noted.
Sparsity metrics depend on exact-zero counting (CappedReLU yields true
zeros; ReLU variants need ε-threshold — fixed at 1e-9 and recorded).
Missing figures in the manuscript (Figs. 2–5 "Enter Caption", Fig. 6
schematic) mean several visual claims have no measured counterpart; the
new plots replace them.

## 15. Conclusion

C-SPINE is a well-posed port of SPINE-style denoising sparse coding to
contextual sentence vectors, with an implementation that matches its
equations but not all its prose (normalization) nor its evaluation hygiene
(leakage, single seed, thin metrics). The contribution of this work is not
a new Table II but the apparatus to earn one: a modular codebase, locked-
test discipline, full recon/sparsity/downstream metric suites, sweeps,
ablations, baselines, interpretability + stability protocols, cross-domain
and cross-encoder tests, pre-registered H1–H6, and a self-contained Colab
notebook that packages everything (codebase, notebook, configs, results,
metrics, plots, tables, checkpoint metadata, logs, reports, README,
requirements) into `final_research.zip`. Whether sparsity preserves
meaning, how far it can go, which terms matter, and whether features are
stable and reusable are empirical questions the notebook is built to answer
— with negative results retained.

**Success-criterion answers (pointer, not verdict):** (1) What is C-SPINE?
§3 + `cspine/sparse_model.py`. (2) Code vs. manuscript? Audit report §1–2
(normalization ❌, leakage ❌, else ✅/⚠️). (3) Information preserved?
`metrics/*_test.json` downstream deltas + H1 verdict. (4) How sparse?
sweep + `sparsity_tradeoff.png` + H2. (5) Which components? `ablations.
json` + §10. (6) Genuinely interpretable? `interp_*.json` + purity caveats
+ §12. (7) Stable? `stability.json` + H4. (8–9) Generalize? `transfer.
json`, portfolio tables, encoder runs + H5. (10) Limits? §13–14.

## References

[1] Devlin et al., BERT, NAACL-HLT 2019. [2] Subramanian et al., SPINE,
AAAI 2018. [3] Mikolov et al., Word2Vec, NeurIPS 2013. [4] Pennington et
al., GloVe, EMNLP 2014. [5] Liu et al., RoBERTa, arXiv 2019. [6] Radford
et al., GPT-2, OpenAI Blog 2019. [7] Olshausen & Field, Vision Research
1997. [8] Bricken et al., Towards Monosemanticity, Anthropic 2023.
[9] Tenney et al., BERT Rediscovers the Classical NLP Pipeline, ACL 2019.
[10] Vincent et al., Denoising Autoencoders, ICML 2008. [11] Kingma & Ba,
Adam, ICLR 2015. [12] Socher et al., SST-2, EMNLP 2013. [13] Zhang et al.,
AG News, NeurIPS 2015. (Plus Gao et al. 2024 BatchTopK, cited in code.)
