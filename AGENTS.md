# Development Guide

This repository contains machine-learning and NLP experiments for cost-aware cascaded linguistic threat detection. Treat it as an independent AI/NLP portfolio project with a public, API-free reproducibility path.

## Scientific Honesty Rules

- Do not use Gemini, DeepSeek, OpenAI, Google, or any paid external API unless the user explicitly enables it outside the public workflow.
- Present this project as an independent AI/NLP portfolio project.
- Do not commit secrets, API keys, `.env` files, tokens, model provider credentials, local caches, raw datasets, or large checkpoints.
- Do not fabricate results, metrics, tables, or experiment outcomes.
- Do not tune thresholds on the test split.
- Use train data for fitting, validation data for threshold/model selection, and test data only for final evaluation.
- Preserve scientific honesty: if a result cannot be reproduced locally, document that clearly.
- Prefer small, reproducible local experiments and smoke checks before any expensive run.
- Keep Gemini, DeepSeek, OpenAI, and other external LLM tiers disabled by default.
- When updating results, cite the exact script and output file that produced them.
- Do not modify project conclusions without evidence from saved outputs or rerun experiments.

## Development Workflow

- Prefer small, reviewable documentation or reproducibility improvements.
- Run `python scripts/smoke_test.py`, `python scripts/collect_results.py`, and `pytest -q` before proposing a public-facing change.
- Keep generated summaries tied to saved artifacts; do not hand-edit metric values in result tables.
- Stage only files that belong to the current change.
- Before committing, check that no raw datasets, caches, checkpoints, `.env` files, credentials, or local-only artifacts are included.

## Public Release Scope

The public repository should run without paid APIs. The reproducible local workflow focuses on:

- Tier 1: TF-IDF plus Logistic Regression.
- Tier 2: RoBERTa or saved transformer predictions/results.
- Cascade routing and summary generation from existing result artifacts.

External LLM experiments may be documented as historical or optional only when cached result files already exist and no private credentials are exposed.

## Dataset Notes

- HateXplain uses official train, validation, and test splits.
- OLID uses the official train split to create an internal validation split. Never select thresholds from the official OLID test split.
- Raw or exported datasets should stay out of Git unless licensing and file-size constraints are explicitly reviewed.
