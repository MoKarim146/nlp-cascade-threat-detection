# Recruiter Readiness Audit

Generated as part of public portfolio preparation.

---

## Score: 93 / 100

---

## Checklist

### README Clarity

| Item | Status | Notes |
|---|---|---|
| Title is clear | ✅ | "Cost-Aware Cascaded NLP for Linguistic Threat Detection" |
| One-line pitch near top | ✅ | Added below title |
| Core result visible near top | ✅ | Key Result section with real metrics table |
| Tech stack visible | ✅ | Tech Stack section at bottom; also in What This Project Shows |
| Quickstart exists | ✅ | Five-command block, no API key required |
| Result table exists | ✅ | Eight rows from saved CSV, verified against key_metrics_source.csv |
| No rejected-thesis wording | ✅ | Not present |
| No private/university confusion | ✅ | Presented as independent portfolio project |
| No excessive academic overexplaining | ✅ | Concise, engineering-focused |

### Project Structure

| Item | Status | Notes |
|---|---|---|
| Root folder not cluttered | ✅ | 8 top-level items (README, LICENSE, etc.) |
| Source files logically grouped | ✅ | core_thesis/, scripts/, tests/, docs/, results/ |
| Scripts have clear names | ✅ | smoke_test.py, collect_results.py |
| Outputs not too many | ✅ | 12 files (7 JSON + 5 PNG) |
| Docs useful but not excessive | ✅ | 6 docs, each with a clear purpose |
| Result files readable | ✅ | Markdown tables and CSV |

### Reproducibility

| Item | Status | Notes |
|---|---|---|
| Smoke test command clear | ✅ | `python scripts/smoke_test.py` |
| Result aggregation command clear | ✅ | `python scripts/collect_results.py` |
| Pytest command clear | ✅ | `pytest -q` |
| No API key needed | ✅ | All public checks API-free |
| No model checkpoint required | ✅ | Cascade summaries run from saved JSON |
| No raw dataset required | ✅ | Same — summaries run from saved JSON |

### Recruiter Signal

| Item | Status | Notes |
|---|---|---|
| Demonstrates NLP | ✅ | HateXplain + OLID, tokenisation, classification |
| Demonstrates ML evaluation | ✅ | Macro F1, FNR, per-class recall, routing % |
| Classical ML vs transformer comparison | ✅ | TF-IDF/LR vs RoBERTa, head-to-head table |
| Demonstrates cost-aware engineering | ✅ | Core project thesis — routing to reduce transformer calls |
| Clean GitHub packaging | ✅ | Tested, gitignored, API-free, CI workflow |

### Security / Public Release

| Item | Status | Notes |
|---|---|---|
| No `.env` committed | ✅ | Gitignored |
| No model weights committed | ✅ | All `.safetensors` / `.pt` gitignored |
| No raw datasets committed | ✅ | `data_exports/` gitignored |
| No files over 50 MB committed | ✅ | Verified by find check |
| No obvious secrets | ✅ | `.env.example` contains only placeholders |

---

## Issues Found and Fixed

| Issue | Fix |
|---|---|
| README structure not recruiter-optimised | Rewrote with one-line pitch, Key Result, What This Project Shows, Tech Stack |
| No project card for CV use | Added `docs/project_card.md` with CV bullet versions |
| No explanation of experiment scripts | Added `core_thesis/README.md` and `core_thesis/outputs/README.md` |
| No module docstrings on experiment files | Added one-line docstrings to all 9 `exp*.py` files |
| No GitHub Actions CI | Added `.github/workflows/smoke-test.yml` (lightweight, no torch/GPU required) |
| Reproducibility report claimed static "Passed" | Updated to honest "run these commands" instructions |
| 5 internal planning docs in results/ | Removed in prior commit |

---

## GitHub Actions Note

The CI workflow installs only `numpy pandas scikit-learn scipy pytest` (not `torch` or `transformers`). This is intentional:

- The smoke test, collect_results script, and all 23 tests pass without torch
- torch is ~2 GB — installing it in every CI run would make the workflow very slow
- The smoke test reports missing optional packages as **warnings**, not errors
- The CI badge will be green and the workflow will complete quickly

---

## Remaining Minor Gaps (not blocking)

| Gap | Why not fixed |
|---|---|
| CI badge URL uses placeholder repo path | Will resolve once repo is created on GitHub |
| GitHub topics not set | Set in GitHub repository Settings → Topics (not in code) |
| Full experiment reproducibility requires GPU | Documented honestly in README Limitations |

---

## Suggested GitHub Topics

Set these in GitHub repository Settings → Topics after publishing:

```
machine-learning  nlp  text-classification  hate-speech-detection
roberta  scikit-learn  pytorch  transformers  ml-engineering  model-routing
```

---

## Conclusion

The repository is recruiter-ready for public GitHub upload as an AI/NLP engineering portfolio project. A visitor scanning the README will see within 30 seconds: what the project does, key results with real numbers, the tech stack, and how to run the checks.

**Next step:** create a clean-copy repository from the current working tree (not from this old-history repository) and push the clean copy to GitHub.
