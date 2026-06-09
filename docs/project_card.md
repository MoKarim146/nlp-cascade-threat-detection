# Project Card: Cost-Aware Cascaded NLP for Linguistic Threat Detection

## Intended Use

This repository is intended as an independent AI/NLP portfolio project demonstrating cost-aware text classification, local reproducibility practices, and evaluation of cascaded routing for linguistic threat or hate-speech detection research.

The public workflow is suitable for reviewing the project structure, reading saved result summaries, running smoke checks, and inspecting how TF-IDF, RoBERTa, and cascade routing are compared.

## Not Intended Use

This repository is not a production moderation system, safety-critical threat-detection service, or deployable policy enforcement tool. The saved models and thresholds should not be used to make real-world decisions without domain-specific validation, bias review, monitoring, and operating-point selection.

The public workflow is not intended to call paid external APIs or depend on private provider credentials.

## Datasets

- HateXplain, using official train, validation, and test splits.
- OLID, using the official train split with an internal validation split.

Raw datasets are not committed to the repository. Dataset use remains governed by the original dataset licenses and access terms.

## Evaluation Metrics

The project reports Macro F1, false-negative rate (FNR), per-class recall where available, Tier 1 handled percentage, and RoBERTa usage percentage. Missing metrics are intentionally left blank and must not be invented.

## Main Limitations

- Results depend on saved outputs from prior local experiments.
- Full transformer retraining requires additional hardware and dataset setup.
- Hate-speech and offensive-language labels can be noisy and may encode annotation bias.
- Thresholds selected for one dataset or domain may not transfer to another.
- Latency and throughput claims are hardware-dependent and should be rerun in the target environment.

## Safety Notes

False negatives are treated as especially important because a missed threat can be costly. The cascade therefore tracks FNR alongside Macro F1 and routing percentage.

This project should be read as an ML engineering experiment, not as evidence that automated threat detection is sufficient for high-stakes moderation or safety decisions.

## Reproducibility Status

The public checks run without datasets, checkpoints, GPU, or external API keys:

```bash
python scripts/smoke_test.py
python scripts/collect_results.py
pytest -q
```

Summary tables are regenerated from saved artifacts and curated metric sources already in the repository. Optional full RoBERTa retraining requires dataset downloads and GPU/MPS/CUDA-capable hardware.
