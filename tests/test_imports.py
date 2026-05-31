from pathlib import Path
import py_compile


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_public_scripts_compile():
    for rel_path in [
        "scripts/smoke_test.py",
        "scripts/collect_results.py",
        "core_thesis/exp1.py",
        "core_thesis/exp2_olid_tfidf.py",
        "core_thesis/exp4_hatexplain_2tier_cascade.py",
        "core_thesis/exp13_hatexplain_safety_constrained_routing.py",
        "core_thesis/exp14_olid_clean_cascade_package.py",
    ]:
        py_compile.compile(str(PROJECT_ROOT / rel_path), doraise=True)
