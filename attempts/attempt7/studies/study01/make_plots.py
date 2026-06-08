"""
Create a small comparison plot for Attempt 7 Study 01.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parents[4]


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    attempt4_rf = _read_json(
        REPO_ROOT / "attempts/attempt4/studies/study01/artifacts/reports/summary.json"
    )
    attempt4_xgb = _read_json(
        REPO_ROOT / "attempts/attempt4/studies/study02/artifacts/reports/summary.json"
    )
    attempt4_blend = _read_json(
        REPO_ROOT / "attempts/attempt4/studies/study03/artifacts/reports/test_metrics.json"
    )
    attempt7_summary = _read_json(
        REPO_ROOT / "attempts/attempt7/studies/study01/artifacts/reports/summary.json"
    )
    attempt7_test = _read_json(
        REPO_ROOT / "attempts/attempt7/studies/study01/artifacts/reports/test_metrics.json"
    )

    labels = ["RF", "XGBoost", "Ensemble"]
    expanded_values = [
        float(attempt4_rf["winner_test_mae"]),
        float(attempt4_xgb["winner_test_mae"]),
        float(attempt4_blend["models"]["tuned_weight_blend"]["overall"]["mae"]),
    ]
    aggregated_values = [
        float(attempt7_test["models"]["rf_winner_baseline"]["overall"]["mae"]),
        float(attempt7_test["models"]["xgb_winner_baseline"]["overall"]["mae"]),
        float(attempt7_summary["winner_test_mae"]),
    ]

    x = range(len(labels))
    width = 0.36
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar([value - width / 2 for value in x], expanded_values, width=width, color="#a94735", label="Attempt 4 expanded rows")
    ax.bar([value + width / 2 for value in x], aggregated_values, width=width, color="#2f6f4e", label="Attempt 7 aggregated jitter")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylabel("Test MAE (m)")
    ax.set_title("Synthetic Comparison: Expanded-Row vs Aggregated-Jitter Models")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()

    output_path = REPO_ROOT / "attempts/attempt7/studies/study01/artifacts/reports/synthetic_family_comparison.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
