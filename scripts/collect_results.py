from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "core_thesis" / "outputs"
RESULTS_DIR = PROJECT_ROOT / "results"
DOCS_DIR = PROJECT_ROOT / "docs"

KEY_NUMBERS = RESULTS_DIR / "key_metrics_source.csv"

SUMMARY_METRICS_CSV = RESULTS_DIR / "summary_metrics.csv"
SUMMARY_TABLES_MD = RESULTS_DIR / "summary_tables.md"
EXPERIMENT_INVENTORY_MD = RESULTS_DIR / "experiment_inventory.md"
REPRO_REPORT_MD = RESULTS_DIR / "reproducibility_report.md"
DOC_RESULTS_SUMMARY_MD = DOCS_DIR / "results_summary.md"

METRIC_HINTS = (
    "macro_f1",
    "f1",
    "fnr",
    "accuracy",
    "tier1",
    "tier2",
    "roberta",
    "latency",
    "coverage",
    "cost",
)

LEGACY_PRIVATE_SUMMARY_PREFIX = "final_" + "thesis"


def read_csv_dicts(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return "\n".join(lines)


def safe_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def fmt(value: Any) -> str:
    number = safe_float(value)
    if number is None:
        return "" if value is None else str(value)
    if abs(number) >= 10:
        return f"{number:.2f}"
    return f"{number:.4f}"


def flatten_json_keys(value: Any, prefix: str = "") -> set[str]:
    keys: set[str] = set()
    if isinstance(value, dict):
        for key, item in value.items():
            full_key = f"{prefix}.{key}" if prefix else str(key)
            keys.add(full_key)
            keys.update(flatten_json_keys(item, full_key))
    elif isinstance(value, list):
        for item in value[:5]:
            keys.update(flatten_json_keys(item, prefix))
    return keys


def metrics_available(path: Path) -> str:
    try:
        if path.suffix.lower() == ".csv":
            with path.open(newline="", encoding="utf-8") as f:
                reader = csv.reader(f)
                headers = next(reader, [])
            matches = [h for h in headers if any(hint in h.lower() for hint in METRIC_HINTS)]
            return ", ".join(matches[:12])
        if path.suffix.lower() == ".json":
            with path.open(encoding="utf-8") as f:
                data = json.load(f)
            keys = sorted(k for k in flatten_json_keys(data) if any(hint in k.lower() for hint in METRIC_HINTS))
            return ", ".join(keys[:12])
        if path.suffix.lower() == ".md":
            text = path.read_text(encoding="utf-8", errors="replace").lower()
            matches = [hint for hint in METRIC_HINTS if hint in text]
            return ", ".join(matches)
    except Exception as exc:
        return f"unreadable: {type(exc).__name__}"
    return ""


def infer_experiment_name(path: Path) -> str:
    match = re.match(r"(exp\d+[a-z]?)", path.name)
    if match:
        return match.group(1)
    return path.stem


def infer_dataset(name: str) -> str:
    lowered = name.lower()
    datasets = []
    if "hatexplain" in lowered:
        datasets.append("HateXplain")
    if "olid" in lowered:
        datasets.append("OLID")
    if not datasets:
        return "mixed/unspecified"
    return ", ".join(datasets)


def infer_system(name: str) -> str:
    lowered = name.lower()
    systems = []
    if "tfidf" in lowered:
        systems.append("TF-IDF")
    if "roberta" in lowered:
        systems.append("RoBERTa")
    if "cascade" in lowered or "routing" in lowered or "pareto" in lowered:
        systems.append("cascade/routing")
    return ", ".join(dict.fromkeys(systems)) or "unspecified"


def infer_script(experiment_name: str, output_name: str) -> str:
    scripts = sorted((PROJECT_ROOT / "core_thesis").glob(f"{experiment_name}_*.py"))
    if scripts:
        return str(scripts[0].relative_to(PROJECT_ROOT))
    exact = PROJECT_ROOT / "core_thesis" / f"{experiment_name}.py"
    if exact.exists():
        return str(exact.relative_to(PROJECT_ROOT))
    lowered = output_name.lower()
    return "unknown"


def reproducibility_for(path: Path) -> tuple[str, str]:
    lowered = path.name.lower()
    if any(hint in lowered for hint in ("gemini", "deepseek", "llama", "qwen", "llm")):
        return "partial", "yes"
    if any(hint in lowered for hint in ("roberta", "distil", "early_exit")):
        return "partial", "no"
    return "yes", "no"


def build_inventory() -> list[dict[str, str]]:
    files = []
    if OUTPUT_DIR.exists():
        files.extend(
            path
            for path in sorted(OUTPUT_DIR.iterdir())
            if path.is_file() and path.suffix.lower() in {".csv", ".json", ".md", ".txt"}
            and not path.name.startswith(LEGACY_PRIVATE_SUMMARY_PREFIX)
        )
    rows = []
    for path in files:
        experiment = infer_experiment_name(path)
        reproducible, requires_api = reproducibility_for(path)
        rows.append(
            {
                "experiment_name": experiment,
                "dataset": infer_dataset(path.name),
                "model_system": infer_system(path.name),
                "script_or_notebook_source": infer_script(experiment, path.name),
                "output_file": str(path.relative_to(PROJECT_ROOT)),
                "metrics_available": metrics_available(path),
                "reproducible_locally": reproducible,
                "requires_external_api": requires_api,
                "notes": "cached output from prior run; no metrics invented",
            }
        )
    return rows


def load_key_metrics() -> list[dict[str, Any]]:
    if not KEY_NUMBERS.exists():
        return []
    rows = read_csv_dicts(KEY_NUMBERS)
    normalized = []
    for row in rows:
        normalized.append(
            {
                "category": row.get("category", ""),
                "dataset": row.get("dataset", ""),
                "system": row.get("system", ""),
                "macro_f1": row.get("macro_f1", ""),
                "fnr": row.get("fnr", ""),
                "tier1_handled_percent": row.get("tier1_handled_percent", ""),
                "roberta_usage_percent": row.get("roberta_usage_percent", ""),
                "tier3_usage_percent": row.get("tier3_usage_percent", ""),
                "project_placement": row.get("project_placement", ""),
                "interpretation": row.get("interpretation", ""),
                "source_file": str(KEY_NUMBERS.relative_to(PROJECT_ROOT)),
            }
        )
    return normalized


def write_summary_tables(rows: list[dict[str, Any]]) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Result Summary Tables",
        "",
        "These tables are generated from saved repository outputs. Missing metrics are left blank; no values are invented.",
        "",
    ]
    if not rows:
        lines.extend(
            [
                "No curated key-number CSV was found.",
                "",
                f"Expected source: `{KEY_NUMBERS.relative_to(PROJECT_ROOT)}`",
            ]
        )
    else:
        for category in ["main_cascade"]:
            category_rows = [row for row in rows if row.get("category") == category]
            if not category_rows:
                continue
            title = category.replace("_", " ").title()
            lines.extend([f"## {title}", ""])
            table_rows = [
                [
                    row.get("dataset", ""),
                    row.get("system", ""),
                    fmt(row.get("macro_f1", "")),
                    fmt(row.get("fnr", "")),
                    fmt(row.get("tier1_handled_percent", "")),
                    fmt(row.get("roberta_usage_percent", "")),
                    fmt(row.get("tier3_usage_percent", "")),
                ]
                for row in category_rows
            ]
            lines.append(
                markdown_table(
                    ["Dataset", "System", "Macro F1", "FNR", "Tier 1 %", "RoBERTa %", "Tier 3 %"],
                    table_rows,
                )
            )
            lines.append("")
    lines.extend(
        [
            "## Source Files",
            "",
            f"- Curated key numbers: `{KEY_NUMBERS.relative_to(PROJECT_ROOT)}`" if KEY_NUMBERS.exists() else "- Curated key numbers: missing",
            "",
        ]
    )
    content = "\n".join(lines)
    SUMMARY_TABLES_MD.write_text(content, encoding="utf-8")
    DOC_RESULTS_SUMMARY_MD.write_text(
        content.replace("# Result Summary Tables", "# Results Summary")
        + "\n## Reproducibility Note\n\n"
        "The main public workflow regenerates these summaries from existing saved outputs. Full transformer training and external LLM calls are not required for the API-free public version.\n",
        encoding="utf-8",
    )


def write_inventory(rows: list[dict[str, str]]) -> None:
    table_rows = [
        [
            row["experiment_name"],
            row["dataset"],
            row["model_system"],
            row["script_or_notebook_source"],
            row["output_file"],
            row["metrics_available"],
            row["reproducible_locally"],
            row["requires_external_api"],
            row["notes"],
        ]
        for row in rows
    ]
    content = "\n".join(
        [
            "# Experiment Inventory",
            "",
            "Generated from files under `core_thesis/outputs/`.",
            "",
            markdown_table(
                [
                    "Experiment",
                    "Dataset",
                    "Model/system",
                    "Script/source",
                    "Output file",
                    "Metrics available",
                    "Reproducible locally?",
                    "External API?",
                    "Notes",
                ],
                table_rows,
            ),
            "",
        ]
    )
    EXPERIMENT_INVENTORY_MD.write_text(content, encoding="utf-8")


def write_repro_report(summary_rows: list[dict[str, Any]], inventory_rows: list[dict[str, str]]) -> None:
    api_rows = [row for row in inventory_rows if row["requires_external_api"] == "yes"]
    content = f"""# Reproducibility Report

Generated by `python scripts/collect_results.py`.

## Current State

- Output directory exists: `{OUTPUT_DIR.exists()}`
- Curated key metrics found: `{KEY_NUMBERS.exists()}`
- Summary rows written: `{len(summary_rows)}`
- Experiment inventory rows: `{len(inventory_rows)}`
- Cached external/API experiment files: `{len(api_rows)}`

## Verification Commands

Run these commands to verify the public workflow locally:

```bash
python scripts/smoke_test.py
python scripts/collect_results.py
pytest -q
```

These commands do not require external API keys, model checkpoints, or raw dataset downloads.

## Not Reproduced By Default

- Full RoBERTa training (requires GPU or Apple Silicon MPS, ~30–60 min per experiment).
- External API calls (disabled in the public core).
- Raw dataset downloads (download from Hugging Face when rerunning training scripts).

## Hardware Notes

- Summary generation, smoke tests, and TF-IDF experiments are CPU-safe.
- Full transformer training may require Apple Silicon MPS, CUDA, or another GPU.
- Latency and throughput numbers depend on local hardware and batch size.

## Notes

This report is generated from existing saved output artifacts. It does not rerun full model training or paid provider calls. Cascade analysis scripts read saved JSON and CSV files produced by prior training runs.
"""
    REPRO_REPORT_MD.write_text(content, encoding="utf-8")


def main() -> int:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    DOCS_DIR.mkdir(parents=True, exist_ok=True)

    summary_rows = load_key_metrics()
    summary_fields = [
        "category",
        "dataset",
        "system",
        "macro_f1",
        "fnr",
        "tier1_handled_percent",
        "roberta_usage_percent",
        "tier3_usage_percent",
        "project_placement",
        "interpretation",
        "source_file",
    ]
    write_csv(SUMMARY_METRICS_CSV, summary_rows, summary_fields)
    write_summary_tables(summary_rows)

    inventory_rows = build_inventory()
    write_inventory(inventory_rows)
    write_repro_report(summary_rows, inventory_rows)

    print(f"wrote {SUMMARY_METRICS_CSV.relative_to(PROJECT_ROOT)}")
    print(f"wrote {SUMMARY_TABLES_MD.relative_to(PROJECT_ROOT)}")
    print(f"wrote {EXPERIMENT_INVENTORY_MD.relative_to(PROJECT_ROOT)}")
    print(f"wrote {REPRO_REPORT_MD.relative_to(PROJECT_ROOT)}")
    print(f"wrote {DOC_RESULTS_SUMMARY_MD.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
