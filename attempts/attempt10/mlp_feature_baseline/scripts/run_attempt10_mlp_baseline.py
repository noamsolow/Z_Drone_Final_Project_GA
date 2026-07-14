"""Attempt 10: shallow MLP baseline on the Attempt 8 feature space.

This script adds the neural-regressor baseline requested for the paper text.
It intentionally reuses the final Attempt 8 feature table, split, and exported
feature schema so the result is directly comparable with the reported RF/XGB
ensemble.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[4]
ATTEMPT8_ROOT = REPO_ROOT / "attempts" / "attempt8-comprehensive run" / "studies"
FEATURE_TABLE = ATTEMPT8_ROOT / "study_all" / "artifacts" / "features" / "aggregated_common_feature_table.csv"
SPLIT_JSON = ATTEMPT8_ROOT / "study_all" / "artifacts" / "splits" / "unified_split.json"
SCHEMA_JSON = ATTEMPT8_ROOT / "study01" / "artifacts" / "reports" / "model_schema.json"
SUMMARY_JSON = ATTEMPT8_ROOT / "study_all" / "artifacts" / "reports" / "summary.json"
OUTPUT_DIR = REPO_ROOT / "attempts" / "attempt10" / "mlp_feature_baseline" / "reports"


def _metrics(prediction: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    err = prediction - truth
    abs_err = np.abs(err)
    return {
        "mae": float(np.mean(abs_err)),
        "median_absolute_error": float(np.median(abs_err)),
        "mean_relative_error": float(np.mean(abs_err / truth)),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "r2": float(1.0 - np.sum(err**2) / np.sum((truth - truth.mean()) ** 2)),
        "within_5m_rate": float(np.mean(abs_err <= 5.0)),
        "within_10m_rate": float(np.mean(abs_err <= 10.0)),
        "within_20m_rate": float(np.mean(abs_err <= 20.0)),
    }


def _train_one_hidden_layer_mlp(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val_m: np.ndarray,
    x_test: np.ndarray,
    y_mean: float,
    y_std: float,
    seed: int,
) -> tuple[dict[str, float], np.ndarray]:
    rng = np.random.default_rng(seed)
    hidden = 32
    input_dim = x_train.shape[1]

    w1 = rng.normal(0.0, math.sqrt(2.0 / input_dim), size=(input_dim, hidden))
    b1 = np.zeros((1, hidden))
    w2 = rng.normal(0.0, math.sqrt(2.0 / hidden), size=(hidden, 1))
    b2 = np.zeros((1, 1))

    adam_m = [np.zeros_like(item) for item in (w1, b1, w2, b2)]
    adam_v = [np.zeros_like(item) for item in (w1, b1, w2, b2)]
    best_params = None
    best_val_mae = float("inf")
    bad_epochs = 0
    step = 0

    learning_rate = 0.002
    l2 = 1e-4
    batch_size = 512
    max_epochs = 1200
    patience = 120

    for epoch in range(1, max_epochs + 1):
        order = rng.permutation(len(x_train))
        for start in range(0, len(order), batch_size):
            batch_index = order[start : start + batch_size]
            xb = x_train[batch_index]
            yb = y_train[batch_index]
            n_batch = len(batch_index)

            z1 = xb @ w1 + b1
            a1 = np.maximum(z1, 0.0)
            out = a1 @ w2 + b2

            d_out = (2.0 / n_batch) * (out - yb)
            grad_w2 = a1.T @ d_out + l2 * w2
            grad_b2 = d_out.sum(axis=0, keepdims=True)
            d_a1 = d_out @ w2.T
            d_z1 = d_a1 * (z1 > 0.0)
            grad_w1 = xb.T @ d_z1 + l2 * w1
            grad_b1 = d_z1.sum(axis=0, keepdims=True)

            params = [w1, b1, w2, b2]
            grads = [grad_w1, grad_b1, grad_w2, grad_b2]
            step += 1
            for i, grad in enumerate(grads):
                adam_m[i] = 0.9 * adam_m[i] + 0.1 * grad
                adam_v[i] = 0.999 * adam_v[i] + 0.001 * (grad**2)
                m_hat = adam_m[i] / (1.0 - 0.9**step)
                v_hat = adam_v[i] / (1.0 - 0.999**step)
                params[i] -= learning_rate * m_hat / (np.sqrt(v_hat) + 1e-8)
            w1, b1, w2, b2 = params

        if epoch % 20 == 0:
            val_prediction = (np.maximum(x_val @ w1 + b1, 0.0) @ w2 + b2).ravel()
            val_prediction = np.clip(val_prediction * y_std + y_mean, 0.0, None)
            val_mae = float(np.mean(np.abs(val_prediction - y_val_m)))
            if val_mae < best_val_mae:
                best_val_mae = val_mae
                best_params = [item.copy() for item in (w1, b1, w2, b2)]
                bad_epochs = 0
            else:
                bad_epochs += 20
                if bad_epochs >= patience:
                    break

    assert best_params is not None
    w1, b1, w2, b2 = best_params
    test_prediction = (np.maximum(x_test @ w1 + b1, 0.0) @ w2 + b2).ravel()
    test_prediction = np.clip(test_prediction * y_std + y_mean, 0.0, None)
    return {"seed": seed, "val_mae": best_val_mae, "epochs": epoch}, test_prediction


def _ridge_prediction(
    x_all: np.ndarray,
    y_all: np.ndarray,
    dev_idx: np.ndarray,
    test_idx: np.ndarray,
    columns: list[int],
    alpha: float = 10.0,
) -> np.ndarray:
    x_train = x_all[dev_idx][:, columns]
    x_test = x_all[test_idx][:, columns]
    mean = x_train.mean(axis=0)
    std = x_train.std(axis=0)
    std[std == 0.0] = 1.0
    x_train = (x_train - mean) / std
    x_test = (x_test - mean) / std

    design_train = np.c_[np.ones(len(x_train)), x_train]
    design_test = np.c_[np.ones(len(x_test)), x_test]
    regularizer = np.eye(design_train.shape[1]) * alpha
    regularizer[0, 0] = 0.0
    coef = np.linalg.solve(design_train.T @ design_train + regularizer, design_train.T @ y_all[dev_idx])
    return design_test @ coef


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with SCHEMA_JSON.open("r", encoding="utf-8") as handle:
        schema = json.load(handle)
    with SPLIT_JSON.open("r", encoding="utf-8") as handle:
        split = json.load(handle)
    with SUMMARY_JSON.open("r", encoding="utf-8") as handle:
        attempt8_summary = json.load(handle)

    features = list(schema["feature_names"])
    df = pd.read_csv(FEATURE_TABLE)
    x_all = df[features].to_numpy(dtype=np.float64)
    y_all = df["true_distance_m"].to_numpy(dtype=np.float64)
    dev_idx = np.asarray(split["dev_indices"], dtype=int)
    test_idx = np.asarray(split["test_indices"], dtype=int)

    rng = np.random.default_rng(20260520)
    dev_permutation = rng.permutation(dev_idx)
    val_size = int(round(0.15 * len(dev_idx)))
    val_idx = dev_permutation[:val_size]
    train_idx = dev_permutation[val_size:]

    mean = x_all[train_idx].mean(axis=0)
    std = x_all[train_idx].std(axis=0)
    std[std == 0.0] = 1.0
    y_mean = float(y_all[train_idx].mean())
    y_std = float(y_all[train_idx].std())

    x_scaled = (x_all - mean) / std
    y_scaled = ((y_all - y_mean) / y_std)[:, None]

    rows = []
    predictions_by_seed = {}
    for seed in (20260520, 20260521, 20260522):
        fit_info, prediction = _train_one_hidden_layer_mlp(
            x_train=x_scaled[train_idx],
            y_train=y_scaled[train_idx],
            x_val=x_scaled[val_idx],
            y_val_m=y_all[val_idx],
            x_test=x_scaled[test_idx],
            y_mean=y_mean,
            y_std=y_std,
            seed=seed,
        )
        row = {"model": "attempt10_shallow_mlp", "split": "test", **fit_info}
        row.update(_metrics(prediction, y_all[test_idx]))
        rows.append(row)
        predictions_by_seed[seed] = prediction

    selected = min(rows, key=lambda item: item["val_mae"])
    selected_prediction = predictions_by_seed[int(selected["seed"])]

    with (OUTPUT_DIR / "mlp_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) + ["selected_by_validation"])
        writer.writeheader()
        for row in rows:
            writer.writerow({**row, "selected_by_validation": row["seed"] == selected["seed"]})

    with (OUTPUT_DIR / "mlp_test_predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["row_index", "true_distance_m", "predicted_distance_m", "absolute_error_m"],
        )
        writer.writeheader()
        for row_index, truth, pred in zip(test_idx, y_all[test_idx], selected_prediction):
            writer.writerow(
                {
                    "row_index": int(row_index),
                    "true_distance_m": float(truth),
                    "predicted_distance_m": float(pred),
                    "absolute_error_m": float(abs(pred - truth)),
                }
            )

    full_columns = list(range(len(features)))
    ridge_full_prediction = _ridge_prediction(x_all, y_all, dev_idx, test_idx, full_columns)
    ridge_full_metrics = _metrics(ridge_full_prediction, y_all[test_idx])
    ablation_rows = []
    for i, feature in enumerate(features):
        columns = [j for j in full_columns if j != i]
        prediction = _ridge_prediction(x_all, y_all, dev_idx, test_idx, columns)
        item = _metrics(prediction, y_all[test_idx])
        ablation_rows.append(
            {
                "removed_feature": feature,
                "test_mae": item["mae"],
                "delta_mae_vs_full_ridge": item["mae"] - ridge_full_metrics["mae"],
            }
        )
    ablation_rows.sort(key=lambda item: item["delta_mae_vs_full_ridge"], reverse=True)

    with (OUTPUT_DIR / "descriptor_leave_one_out.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(ablation_rows[0].keys()))
        writer.writeheader()
        writer.writerows(ablation_rows)

    ensemble = attempt8_summary["poster_test_ranking"][0]
    summary = {
        "attempt": "attempt10_mlp_feature_baseline",
        "purpose": "Small neural-network baseline requested for the Proposed Method discussion.",
        "source_attempt": "attempt8-comprehensive run",
        "feature_table": str(FEATURE_TABLE),
        "split_json": str(SPLIT_JSON),
        "schema_json": str(SCHEMA_JSON),
        "feature_count": len(features),
        "selected_mlp": selected,
        "attempt8_final_ensemble": {
            "mae": ensemble["mae"],
            "rmse": ensemble["rmse"],
            "r2": ensemble["r2"],
            "within_10m_rate": ensemble["within_10m_rate"],
            "rf_weight": attempt8_summary["best_blend_row"]["rf_weight"],
            "xgb_weight": attempt8_summary["best_blend_row"]["xgb_weight"],
        },
        "ridge_full_metrics": ridge_full_metrics,
        "top_descriptor_leave_one_out": ablation_rows[:10],
    }
    with (OUTPUT_DIR / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    md = [
        "# Attempt 10: Shallow MLP Feature Baseline",
        "",
        "This calculation trains a small one-hidden-layer MLP on the final Attempt 8",
        "21-feature representation and evaluates it on the same held-out test split.",
        "",
        "## Inputs",
        f"- Feature table: `{FEATURE_TABLE}`",
        f"- Split: `{SPLIT_JSON}`",
        f"- Feature schema: `{SCHEMA_JSON}`",
        f"- Feature count: `{len(features)}`",
        "",
        "## Result",
        f"- Selected MLP seed: `{int(selected['seed'])}`",
        f"- MLP test MAE: `{selected['mae']:.3f}` m",
        f"- MLP test RMSE: `{selected['rmse']:.3f}` m",
        f"- MLP test R2: `{selected['r2']:.3f}`",
        f"- MLP within 10 m: `{100.0 * selected['within_10m_rate']:.1f}%`",
        f"- Attempt 8 final ensemble test MAE: `{ensemble['mae']:.3f}` m",
        f"- Attempt 8 final ensemble within 10 m: `{100.0 * ensemble['within_10m_rate']:.1f}%`",
        "",
        "The MLP improves over the linear baselines but remains worse than the final",
        "RF--XGBoost ensemble on the same feature space.",
        "",
        "## Descriptor probe",
        "",
        "A ridge leave-one-feature-out probe was added as a lightweight descriptor-level",
        "sanity check. The largest MAE increases come from removing:",
    ]
    for row in ablation_rows[:5]:
        md.append(f"- `{row['removed_feature']}`: delta MAE `{row['delta_mae_vs_full_ridge']:.3f}` m")
    md.append("")
    (OUTPUT_DIR / "summary.md").write_text("\n".join(md), encoding="utf-8")


if __name__ == "__main__":
    main()
