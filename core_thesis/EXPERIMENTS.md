# Experiment Guide

This document describes the progression of experiments in `core_thesis/`. Each script builds on the previous ones, following the thesis arc from classical baseline → transformer baseline → cascade → safety-constrained routing.

## Progression

### Stage 1: Classical Baselines

**`exp1.py` — HateXplain TF-IDF + Logistic Regression**

The first experiment. Establishes a CPU-only baseline on HateXplain using combined word-unigram/bigram and character n-gram TF-IDF features (35k total) fed into Logistic Regression. Produces confidence scores needed later for cascade routing. Runs in seconds on CPU.

**`exp2_olid_tfidf.py` — OLID TF-IDF + Logistic Regression**

Same TF-IDF pipeline applied to OLID (Offensive Language Identification Dataset). Confirms the baseline approach generalises across datasets. Also inspects top discriminative features.

### Stage 2: Transformer Baselines

**`exp3_hatexplain_roberta.py` — Fine-tune RoBERTa on HateXplain**

Fine-tunes `roberta-base` on HateXplain for 3 epochs with balanced class weights and a linear schedule. Saves the best validation checkpoint to `core_thesis/models/roberta_hatexplain/`. This model is the Tier 2 component for all HateXplain cascade experiments.

**`exp7b_olid_roberta_clean_subtrain.py` — Fine-tune RoBERTa on OLID (subtrain)**

Fine-tunes `roberta-base` on 80% of the OLID official train split (the remaining 20% is used as internal validation). Saves the best checkpoint to `core_thesis/models/roberta_olid_subtrain_clean/`. This is the Tier 2 component for OLID cascade experiments.

### Stage 3: Cascade Architecture

**`exp4_hatexplain_2tier_cascade.py` — Symmetric and asymmetric threshold search**

First cascade experiment. Searches over symmetric thresholds (same `τ` for both classes) and class-specific thresholds (`τ_threat`, `τ_safe`). Demonstrates the motivation for asymmetric routing: threats and safe examples have different confidence profiles.

**`exp8_hatexplain_confidence_cost_analysis.py` — Confidence-bin analysis and cost-penalty sweep**

Analyses how TF-IDF and RoBERTa accuracy vary across confidence bins. Explores a cost-penalty objective: `F1 − λ · tier2_usage` for several values of `λ`. Motivates the safety-constrained protocol by showing that pure cost optimisation can increase FNR.

### Stage 4: Safety-Constrained Routing

**`exp13_hatexplain_safety_constrained_routing.py` — Safety-constrained cascade (HateXplain)**

The main HateXplain cascade result. Selects thresholds on a `45 × 45` validation grid (τ from 0.50 to 0.95 in steps of 0.01) subject to two constraints:

- Macro F1 must not drop more than `Δ = 0.002` below the RoBERTa standalone baseline.
- FNR must not exceed the RoBERTa standalone FNR by more than `ε = 0.010`.

Among feasible threshold pairs, selects the one that minimises Tier 2 usage (cheapest that passes the safety check). Saves the grid CSV and result JSON to `outputs/`.

**`exp13_hatexplain_constrained_routing.py` — Variant with Pareto annotation**

An alternative implementation of the same constrained routing experiment. Adds Pareto optimality annotation to the validation grid and uses a slightly different feasibility relaxation protocol. Both scripts produce the HateXplain cascade results; `exp13_hatexplain_safety_constrained_routing.py` is the canonical version.

**`exp14_olid_clean_cascade_package.py` — Full cascade package (OLID)**

The main OLID cascade result. Mirrors the HateXplain cascade protocol on OLID. Produces symmetric cascade, best-F1 class-specific cascade, and safety-constrained class-specific cascade results in a single script. Also saves per-example test predictions for Pareto analysis.

**`exp14d_olid_pareto_operating_points.py` — Pareto operating-point analysis with plots**

Reads the OLID validation grid (saved by `exp14`) and per-example test predictions. Computes the Pareto frontier on the validation set (minimising FNR and Tier 2 usage). Selects and evaluates multiple named operating points. Generates the `fig_exp14d_olid_*.png` figures.

---

## Saved Outputs

All saved result files live in `core_thesis/outputs/`. Large files (grids, per-example prediction CSVs) are gitignored. The following are committed:

| File | Source experiment |
|---|---|
| `hatexplain_roberta_results.json` | `exp3` |
| `olid_roberta_subtrain_results.json` | `exp7b` |
| `exp7b_olid_roberta_clean_subtrain_results.json` | `exp7b` |
| `exp13_hatexplain_constrained_routing_result.json` | `exp13` (constrained routing) |
| `exp13_hatexplain_safety_constrained_result.json` | `exp13` (safety constrained) |
| `exp14_olid_clean_cascade_package_result.json` | `exp14` |
| `exp14d_olid_pareto_operating_points_result.json` | `exp14d` |
| `fig_exp13_*.png` | `exp13` |
| `fig_exp14d_olid_*.png` | `exp14d` |

---

## Model Checkpoints

Model checkpoints are **not committed** (they are 475 MB each). To reproduce the cascade experiments from scratch:

1. Run `exp3_hatexplain_roberta.py` to train and save `roberta_hatexplain/`.
2. Run `exp7b_olid_roberta_clean_subtrain.py` to train and save `roberta_olid_subtrain_clean/`.
3. Then run `exp13_hatexplain_safety_constrained_routing.py` and `exp14_olid_clean_cascade_package.py`.

The cascade analysis scripts (`exp13`, `exp14`, `exp14d`) read the saved result/prediction files when models are unavailable, making it possible to regenerate summaries without retraining.
