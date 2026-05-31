from pathlib import Path
import subprocess


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_public_core_does_not_track_provider_runner():
    result = subprocess.run(
        ["git", "ls-files", "core_thesis/exp16_olid_provider_prediction_runner.py"],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.stdout.strip() == ""


def test_env_example_contains_only_placeholders():
    text = (PROJECT_ROOT / ".env.example").read_text()
    assert "replace_with_your_own" in text
    assert "AIza" not in text
    assert "sk-" not in text
