from __future__ import annotations

import importlib.util
import os
import py_compile
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

REQUIRED_FILES = [
    "README.md",
    ".gitignore",
    ".env.example",
    "scripts/collect_results.py",
    "results/key_metrics_source.csv",
]

SCRIPTS_TO_COMPILE = [
    "scripts/collect_results.py",
    "core_thesis/exp1.py",
    "core_thesis/exp2_olid_tfidf.py",
    "core_thesis/exp4_hatexplain_2tier_cascade.py",
    "core_thesis/exp13_hatexplain_safety_constrained_routing.py",
    "core_thesis/exp14_olid_clean_cascade_package.py",
]

OPTIONAL_PACKAGES = ["numpy", "pandas", "sklearn", "torch", "transformers", "datasets"]


def git_tracked(path: str) -> bool:
    result = subprocess.run(
        ["git", "ls-files", "--error-unmatch", path],
        cwd=PROJECT_ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return result.returncode == 0


def main() -> int:
    errors: list[str] = []
    warnings: list[str] = []

    for rel_path in REQUIRED_FILES:
        if not (PROJECT_ROOT / rel_path).exists():
            errors.append(f"missing required file: {rel_path}")

    if git_tracked(".env") or git_tracked("core_thesis/.env"):
        errors.append("a .env file is tracked by Git")

    local_env = PROJECT_ROOT / "core_thesis" / ".env"
    if local_env.exists():
        warnings.append("local core_thesis/.env exists but is ignored and not required")

    for key in ["GEMINI_API_KEY", "DEEPSEEK_API_KEY", "OPENAI_API_KEY", "GOOGLE_API_KEY"]:
        if os.environ.get(key):
            warnings.append(f"{key} is set in the shell; local-safe scripts do not use it")

    for package in OPTIONAL_PACKAGES:
        if importlib.util.find_spec(package) is None:
            warnings.append(f"optional package not importable: {package}")

    for rel_path in SCRIPTS_TO_COMPILE:
        path = PROJECT_ROOT / rel_path
        if not path.exists():
            errors.append(f"cannot compile missing script: {rel_path}")
            continue
        try:
            py_compile.compile(str(path), doraise=True)
        except py_compile.PyCompileError as exc:
            errors.append(f"syntax check failed for {rel_path}: {exc.msg}")

    print("Smoke test report")
    print("=================")
    for warning in warnings:
        print(f"warning: {warning}")
    if errors:
        for error in errors:
            print(f"error: {error}")
        return 1
    print("local-safe smoke checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
