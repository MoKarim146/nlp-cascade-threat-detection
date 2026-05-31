# Reproducibility

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Safe Public Commands

```bash
python scripts/smoke_test.py
python scripts/collect_results.py
pytest
```

These commands do not require Gemini, DeepSeek, OpenAI, Google, or other paid external API keys.

## Hardware Notes

- TF-IDF and result summarization are CPU-friendly.
- RoBERTa evaluation and fine-tuning can use CPU, Apple Silicon MPS, or CUDA, but full training can be slow.
- Early-exit and latency results depend on local hardware and batching.

## Reproducible Without External APIs

- Result inventory generation.
- Summary table generation from saved outputs.
- Static smoke checks.
- TF-IDF-style local scripts when datasets are available.
- Cascade summaries from saved predictions/results.
- Public core scripts for HateXplain and OLID cascade analysis, when required datasets and local checkpoints are available.

## Not Reproduced By Default

- Long RoBERTa or DistilRoBERTa training.
- External provider calls.
- Gated external model downloads.
- Raw dataset download/export steps unless the user installs and accepts the original dataset requirements.

## Data Protocol Reminder

Use train data for fitting, validation for threshold/model selection, and test only for final evaluation. For OLID, create validation only from official train and never use official test for threshold selection.
