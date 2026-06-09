# Reproducibility Checklist

This checklist separates the safe public workflow from optional expensive reruns.

## Runs Without Datasets, Checkpoints, GPU, Or API Keys

- `python scripts/smoke_test.py`
- `python scripts/collect_results.py`
- `pytest -q`

These commands use repository code and saved result artifacts. They do not require raw datasets, model checkpoints, GPU/MPS/CUDA, `.env` files, or paid external APIs.

## Requires Saved Artifacts

- Generated summaries in `results/` and `docs/results_summary.md` depend on committed files under `core_thesis/outputs/` and curated metrics in `results/key_metrics_source.csv`.
- Missing metrics must stay blank. Do not infer, estimate, or invent values.
- When updating result documentation, cite the exact script and output file that produced the numbers.

## Optional GPU/MPS Retraining

- Full RoBERTa training or evaluation may require Apple Silicon MPS, CUDA, or another GPU.
- Dataset downloads are optional rerun steps and must follow the original dataset licenses and split protocols.
- Transformer retraining can be slow and may produce small variation from hardware, seeds, and library versions.

## Data-Split Rules

- HateXplain uses official train, validation, and test splits.
- OLID uses the official train split to create an internal validation split.
- Do not tune thresholds on any test split.
- Use train data for fitting, validation data for model/threshold selection, and test data only for final evaluation.

## Public Release Hygiene

- Do not stage raw datasets, exported datasets, checkpoints, caches, `.env` files, secrets, tokens, or provider credentials.
- Keep external LLM tiers disabled by default in the public workflow.
- Do not change project conclusions unless the change is supported by saved outputs or a documented rerun.
