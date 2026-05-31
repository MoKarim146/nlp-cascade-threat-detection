# core_thesis/outputs/

Saved result artifacts from the cascade experiments. All files here are committed.
Large intermediate files (threshold grids, per-example prediction CSVs) are gitignored.

## Result JSON Files

| File | Source | Contents |
|---|---|---|
| `hatexplain_roberta_results.json` | exp3 | RoBERTa training history and test metrics on HateXplain |
| `olid_roberta_subtrain_results.json` | exp7b (early run) | RoBERTa training metrics on OLID subtrain |
| `exp7b_olid_roberta_clean_subtrain_results.json` | exp7b | Full training history, validation and test metrics for OLID RoBERTa |
| `exp13_hatexplain_constrained_routing_result.json` | exp13 (variant) | Constrained routing result with Pareto annotation on HateXplain |
| `exp13_hatexplain_safety_constrained_result.json` | exp13 (main) | Safety-constrained cascade selection result on HateXplain |
| `exp14_olid_clean_cascade_package_result.json` | exp14 | Full OLID cascade result (symmetric + best-F1 + safety-constrained) |
| `exp14d_olid_pareto_operating_points_result.json` | exp14d | Pareto operating-point analysis for OLID |

## Figures

| File | Description |
|---|---|
| `fig_exp13_f1_vs_tier2.png` | HateXplain — Macro F1 vs Tier 2 escalation % across all threshold pairs |
| `fig_exp13_fnr_vs_tier2.png` | HateXplain — FNR vs Tier 2 escalation % across all threshold pairs |
| `fig_exp13_pareto.png` | HateXplain — Pareto frontier (minimising FNR and Tier 2 usage jointly) |
| `fig_exp14d_olid_f1_vs_tier2.png` | OLID — Macro F1 vs Tier 2 escalation % with selected operating points |
| `fig_exp14d_olid_fnr_vs_tier2.png` | OLID — FNR vs Tier 2 escalation % with selected operating points |

## Regenerating Summaries

Results summaries in `results/` are regenerated from these files by:

```bash
python scripts/collect_results.py
```

This does not require model checkpoints or raw datasets.
