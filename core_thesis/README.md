# core_thesis/

This directory contains all experiment scripts and their saved outputs.

## Experiment Scripts

| Script | Purpose |
|---|---|
| `exp1.py` | TF-IDF + Logistic Regression baseline on HateXplain |
| `exp2_olid_tfidf.py` | TF-IDF + Logistic Regression baseline on OLID |
| `exp3_hatexplain_roberta.py` | Fine-tune `roberta-base` on HateXplain (saves checkpoint to `models/roberta_hatexplain/`) |
| `exp4_hatexplain_2tier_cascade.py` | Symmetric and class-specific cascade threshold search on HateXplain |
| `exp7b_olid_roberta_clean_subtrain.py` | Fine-tune `roberta-base` on 80% of OLID train (saves checkpoint to `models/roberta_olid_subtrain_clean/`) |
| `exp8_hatexplain_confidence_cost_analysis.py` | Confidence-bin analysis and cost-penalty sweep (λ · tier2_usage) on HateXplain |
| `exp13_hatexplain_safety_constrained_routing.py` | **Main HateXplain result** — safety-constrained class-specific threshold selection |
| `exp13_hatexplain_constrained_routing.py` | Variant of exp13 with Pareto annotation on the validation grid |
| `exp14_olid_clean_cascade_package.py` | **Main OLID result** — full cascade package (symmetric + best-F1 + safety-constrained) |
| `exp14d_olid_pareto_operating_points.py` | Pareto operating-point analysis with plots for OLID |

For a detailed description of each experiment's purpose and place in the research arc, see [`EXPERIMENTS.md`](EXPERIMENTS.md).

## Running Order

```bash
# Stage 1 — Baselines (CPU-safe, fast)
python core_thesis/exp1.py
python core_thesis/exp2_olid_tfidf.py

# Stage 2 — Transformer training (requires GPU/MPS, ~30–60 min each)
python core_thesis/exp3_hatexplain_roberta.py
python core_thesis/exp7b_olid_roberta_clean_subtrain.py

# Stage 3 — Cascade analysis (CPU-safe once models exist)
python core_thesis/exp4_hatexplain_2tier_cascade.py
python core_thesis/exp8_hatexplain_confidence_cost_analysis.py
python core_thesis/exp13_hatexplain_safety_constrained_routing.py
python core_thesis/exp14_olid_clean_cascade_package.py
python core_thesis/exp14d_olid_pareto_operating_points.py
```

## What Is and Is Not Committed

**Committed:** saved result JSON files and figures in `outputs/`

**Not committed (gitignored):** model checkpoints (`models/`), raw dataset exports, large grid CSVs, per-example prediction CSVs, virtual environments
