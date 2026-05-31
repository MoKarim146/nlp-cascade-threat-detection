from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_result_summary_sources_exist():
    assert (PROJECT_ROOT / "results" / "key_metrics_source.csv").exists()
    assert (PROJECT_ROOT / "results" / "summary_tables.md").exists()
