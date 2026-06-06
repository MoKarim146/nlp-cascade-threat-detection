# Project Card: Cost-Aware Cascaded NLP for Linguistic Threat Detection

---

## Problem

Transformer classifiers are strong but expensive. Running every input through a fine-tuned RoBERTa model is unnecessary: many examples are easy enough for a cheap CPU classifier to handle correctly. The question is how to route intelligently — and how to do it safely when false negatives (missed threats) are especially costly.

---

## Approach

Build a two-tier cascade:

- **Tier 1**: TF-IDF (word + character n-grams) + Logistic Regression — fast, CPU-only, trained in seconds
- **Tier 2**: fine-tuned `roberta-base` — stronger, GPU-accelerated, reserved for uncertain examples

Routing uses class-specific confidence thresholds (`τ_threat`, `τ_safe`) selected on a held-out validation set. The selection protocol is safety-constrained: the cascade must not increase the false-negative rate by more than `ε = 0.01` relative to the RoBERTa-only baseline before the cost objective (minimise Tier 2 usage) is optimised.

Experiments run on two public datasets: **HateXplain** and **OLID** (Offensive Language Identification Dataset).

---

## Core Architecture

```
Input text
  → TF-IDF + Logistic Regression (Tier 1)
    → confidence ≥ threshold? → accept Tier 1 prediction (cheap)
    → confidence < threshold? → RoBERTa (Tier 2)  →  final prediction
```

Thresholds are asymmetric: the THREAT class uses a separate threshold from the SAFE class, reflecting that false negatives on threats are costlier than false negatives on safe examples.

---

## Main Results

All numbers are from saved output files. No metrics are invented.

| Dataset | System | Macro F1 | FNR | Tier 1 handled |
|---|---|---|---|---|
| HateXplain | TF-IDF standalone | 0.7739 | 0.2224 | 100% |
| HateXplain | RoBERTa standalone | 0.7911 | 0.1594 | 0% |
| HateXplain | Safety-constrained cascade | 0.7913 | **0.1497** | **60%** |
| OLID | TF-IDF standalone | 0.7282 | 0.3583 | 100% |
| OLID | RoBERTa standalone | 0.8105 | 0.2208 | 0% |
| OLID | Safety-constrained cascade | 0.8059 | **0.2083** | **38%** |

The safety-constrained cascade routes 38–60% of examples to the cheap CPU classifier and achieves a **lower false-negative rate than the transformer-only baseline** on both datasets.

---

## Engineering Decisions

- **Class-specific thresholds** rather than a single symmetric threshold — threat and safe predictions have different confidence profiles
- **Validation-only threshold selection** — thresholds are locked before the test set is touched; no test-set leakage
- **Safety-first constraint** — the FNR constraint is enforced before the cost objective; cheapness never overrides safety
- **Cascade framing tracks cost explicitly** — routing percentage is reported alongside F1 and FNR, not just accuracy
- **API-free public packaging** — external LLM tiers are disabled; the public workflow runs on saved outputs without any API key

---

## Reproducibility

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/smoke_test.py
python scripts/collect_results.py
pytest -q
```

No external API key or model checkpoint required for the public checks. Full retraining requires GPU/MPS and dataset downloads from Hugging Face.

---

## Limitations

- Dataset labels reflect annotation decisions and may encode bias
- Thresholds selected on one domain may not transfer to another
- False negatives are especially important in safety-critical settings; they are tracked explicitly but any production deployment would need its own operating-point selection
- Full reproducibility of transformer training depends on hardware, seed sensitivity, and Hugging Face model availability

---
