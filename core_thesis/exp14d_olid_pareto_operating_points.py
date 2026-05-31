"""Experiment 14d — Pareto operating-point analysis and figures for the OLID cascade."""
from pathlib import Path
import csv
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support


SAFE = 0
THREAT = 1

CORE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = CORE_DIR / "outputs"

GRID_PATH = OUTPUT_DIR / "exp14_olid_class_specific_grid.csv"
EXP14_RESULT_PATH = OUTPUT_DIR / "exp14_olid_clean_cascade_package_result.json"
EXP14B_POINTS_PATH = OUTPUT_DIR / "exp14b_olid_quality_first_safety_points.csv"
EXP14B_RESULT_PATH = OUTPUT_DIR / "exp14b_olid_quality_first_safety_result.json"
TEST_PREDICTIONS_PATH = OUTPUT_DIR / "exp14_olid_clean_cascade_test_predictions.csv"

PARETO_PATH = OUTPUT_DIR / "exp14d_olid_pareto_frontier.csv"
OPERATING_POINTS_PATH = OUTPUT_DIR / "exp14d_olid_operating_points.csv"
RESULT_PATH = OUTPUT_DIR / "exp14d_olid_pareto_operating_points_result.json"
FIG_F1_PATH = OUTPUT_DIR / "fig_exp14d_olid_f1_vs_tier2.png"
FIG_FNR_PATH = OUTPUT_DIR / "fig_exp14d_olid_fnr_vs_tier2.png"

REQUIRED_GRID_COLUMNS = {
    "tau_threat",
    "tau_safe",
    "validation_macro_f1",
    "validation_fnr",
    "validation_tier1_handled_percent",
    "validation_tier2_escalated_percent",
}

REQUIRED_TEST_COLUMNS = {
    "true_label",
    "tfidf_pred",
    "tfidf_confidence",
    "roberta_pred",
}


def load_validation_grid():
    grid_df = pd.read_csv(GRID_PATH)
    detected_columns = grid_df.columns.tolist()
    print("Detected validation grid columns:", detected_columns)

    missing = REQUIRED_GRID_COLUMNS - set(detected_columns)
    if missing:
        raise ValueError(
            "Validation grid is missing required columns: "
            f"{sorted(missing)}. Detected columns: {detected_columns}"
        )

    return grid_df


def validate_test_predictions(test_df):
    detected_columns = test_df.columns.tolist()
    missing = REQUIRED_TEST_COLUMNS - set(detected_columns)
    if missing:
        raise ValueError(
            "Saved test prediction CSV is insufficient for arbitrary routing. "
            f"Missing columns: {sorted(missing)}. Detected columns: {detected_columns}. "
            "A safe recomputation path would need OLID train/test, the same 80/20 "
            "internal validation split, Exp14 TF-IDF settings, and the saved "
            "models/roberta_olid_subtrain_clean checkpoint."
        )


def compute_pareto_frontier(grid_df):
    # Sort by increasing Tier 2, then keep rows that improve the best FNR seen so far.
    # This is equivalent to nondomination for two minimization objectives.
    sorted_df = grid_df.sort_values(
        by=[
            "validation_tier2_escalated_percent",
            "validation_fnr",
            "validation_macro_f1",
        ],
        ascending=[True, True, False],
    )

    pareto_rows = []
    best_fnr = float("inf")
    tolerance = 1e-12

    for _, row in sorted_df.iterrows():
        current_fnr = float(row["validation_fnr"])
        if current_fnr < best_fnr - tolerance:
            pareto_rows.append(row)
            best_fnr = current_fnr

    return pd.DataFrame(pareto_rows).sort_values(
        by=["validation_tier2_escalated_percent", "validation_fnr"],
        ascending=[True, True],
    )


def false_negative_rate(y_true, preds):
    total_threats = np.sum(y_true == THREAT)
    if total_threats == 0:
        return 0.0
    false_negatives = np.sum((y_true == THREAT) & (preds == SAFE))
    return float(false_negatives / total_threats)


def prediction_counts(preds):
    return {
        "SAFE": int(np.sum(preds == SAFE)),
        "THREAT": int(np.sum(preds == THREAT)),
    }


def metrics_from_predictions(y_true, preds, tier1_fraction, tier2_fraction):
    precision, recall, per_class_f1, support = precision_recall_fscore_support(
        y_true,
        preds,
        labels=[SAFE, THREAT],
        zero_division=0,
    )

    return {
        "macro_f1": float(f1_score(y_true, preds, average="macro")),
        "fnr": false_negative_rate(y_true, preds),
        "tier1_handled_percent": float(100.0 * tier1_fraction),
        "tier2_escalated_percent": float(100.0 * tier2_fraction),
        "confusion_matrix_rows_true_cols_pred_SAFE_THREAT": confusion_matrix(
            y_true,
            preds,
            labels=[SAFE, THREAT],
        ).astype(int).tolist(),
        "per_class": {
            "SAFE": {
                "precision": float(precision[0]),
                "recall": float(recall[0]),
                "f1": float(per_class_f1[0]),
                "support": int(support[0]),
            },
            "THREAT": {
                "precision": float(precision[1]),
                "recall": float(recall[1]),
                "f1": float(per_class_f1[1]),
                "support": int(support[1]),
            },
        },
        "prediction_counts": prediction_counts(preds),
    }


def run_class_specific_cascade(tfidf_preds, tfidf_conf, roberta_preds, tau_threat, tau_safe):
    tfidf_says_threat = tfidf_preds == THREAT
    tfidf_says_safe = tfidf_preds == SAFE

    keep_threat = tfidf_says_threat & (tfidf_conf >= tau_threat)
    keep_safe = tfidf_says_safe & (tfidf_conf >= tau_safe)
    keep_tfidf = keep_threat | keep_safe
    escalate = ~keep_tfidf

    final_preds = tfidf_preds.copy()
    final_preds[escalate] = roberta_preds[escalate]

    return final_preds, float(np.mean(keep_tfidf)), float(np.mean(escalate))


def evaluate_class_specific_on_test(test_df, tau_threat, tau_safe):
    y_true = test_df["true_label"].to_numpy(dtype=int)
    tfidf_preds = test_df["tfidf_pred"].to_numpy(dtype=int)
    tfidf_conf = test_df["tfidf_confidence"].to_numpy(dtype=float)
    roberta_preds = test_df["roberta_pred"].to_numpy(dtype=int)

    preds, tier1_fraction, tier2_fraction = run_class_specific_cascade(
        tfidf_preds,
        tfidf_conf,
        roberta_preds,
        tau_threat,
        tau_safe,
    )
    return metrics_from_predictions(y_true, preds, tier1_fraction, tier2_fraction)


def validation_metrics_from_row(row):
    return {
        "macro_f1": float(row["validation_macro_f1"]),
        "fnr": float(row["validation_fnr"]),
        "tier1_handled_percent": float(row["validation_tier1_handled_percent"]),
        "tier2_escalated_percent": float(row["validation_tier2_escalated_percent"]),
    }


def best_f1_row(grid_df):
    return grid_df.sort_values(
        by=[
            "validation_macro_f1",
            "validation_fnr",
            "validation_tier1_handled_percent",
            "tau_threat",
            "tau_safe",
        ],
        ascending=[False, True, False, True, True],
    ).iloc[0].to_dict()


def find_grid_row(grid_df, tau_threat, tau_safe):
    matched = grid_df[
        (grid_df["tau_threat"].round(3) == round(float(tau_threat), 3))
        & (grid_df["tau_safe"].round(3) == round(float(tau_safe), 3))
    ]
    if matched.empty:
        raise ValueError(f"No validation grid row found for tau_threat={tau_threat}, tau_safe={tau_safe}")
    return matched.iloc[0].to_dict()


def select_cost_first_safety(grid_df, exp14_result):
    existing = exp14_result.get("safety_constrained_class_specific_cascade")
    if existing and "tau_threat" in existing and "tau_safe" in existing:
        row = find_grid_row(grid_df, existing["tau_threat"], existing["tau_safe"])
        return row, "from_exp14_safety_constrained_point"

    roberta_val = exp14_result["roberta"]["validation"]
    feasible = grid_df[
        (grid_df["validation_macro_f1"] >= roberta_val["macro_f1"] - 0.002)
        & (grid_df["validation_fnr"] <= roberta_val["fnr"] + 0.010)
    ].copy()
    if feasible.empty:
        return None, "no_feasible_point"

    row = feasible.sort_values(
        by=[
            "validation_tier2_escalated_percent",
            "validation_macro_f1",
            "validation_fnr",
            "tau_threat",
            "tau_safe",
        ],
        ascending=[True, False, True, True, True],
    ).iloc[0].to_dict()
    return row, "selected_from_validation_grid"


def select_quality_first_safety(grid_df, exp14_result):
    if EXP14B_RESULT_PATH.exists():
        with open(EXP14B_RESULT_PATH) as f:
            exp14b = json.load(f)
        recommended = exp14b.get("recommended_point")
        if recommended:
            row = find_grid_row(grid_df, recommended["tau_threat"], recommended["tau_safe"])
            return row, "from_exp14b_recommended_point"

    roberta_val = exp14_result["roberta"]["validation"]
    feasible = grid_df[
        (grid_df["validation_fnr"] <= roberta_val["fnr"] + 0.010)
        & (grid_df["validation_tier1_handled_percent"] >= 30.0)
    ].copy()
    if feasible.empty:
        return None, "no_feasible_point"

    row = feasible.sort_values(
        by=[
            "validation_macro_f1",
            "validation_fnr",
            "validation_tier1_handled_percent",
            "tau_threat",
            "tau_safe",
        ],
        ascending=[False, True, False, True, True],
    ).iloc[0].to_dict()
    return row, "selected_from_validation_grid"


def select_safety_first(grid_df, exp14_result):
    roberta_val = exp14_result["roberta"]["validation"]
    feasible = grid_df[
        (grid_df["validation_macro_f1"] >= roberta_val["macro_f1"] - 0.010)
        & (grid_df["validation_tier2_escalated_percent"] <= 50.0)
    ].copy()
    if feasible.empty:
        return None, "no_feasible_point"

    row = feasible.sort_values(
        by=[
            "validation_fnr",
            "validation_macro_f1",
            "validation_tier2_escalated_percent",
            "tau_threat",
            "tau_safe",
        ],
        ascending=[True, False, True, True, True],
    ).iloc[0].to_dict()
    return row, "selected_from_validation_grid"


def select_target_50_compute(grid_df, exp14_result):
    roberta_val = exp14_result["roberta"]["validation"]
    feasible = grid_df[
        (grid_df["validation_tier2_escalated_percent"] <= 50.0)
        & (grid_df["validation_fnr"] <= roberta_val["fnr"] + 0.020)
    ].copy()
    if feasible.empty:
        return None, "no_feasible_point"

    row = feasible.sort_values(
        by=[
            "validation_macro_f1",
            "validation_fnr",
            "validation_tier2_escalated_percent",
            "tau_threat",
            "tau_safe",
        ],
        ascending=[False, True, True, True, True],
    ).iloc[0].to_dict()
    return row, "selected_from_validation_grid"


def build_class_specific_point(system, row, test_df, selection_source, selection_status="selected"):
    tau_threat = float(row["tau_threat"])
    tau_safe = float(row["tau_safe"])
    test_metrics = evaluate_class_specific_on_test(test_df, tau_threat, tau_safe)
    val_metrics = validation_metrics_from_row(row)

    return {
        "system": system,
        "point_type": "class_specific",
        "tau": None,
        "tau_threat": tau_threat,
        "tau_safe": tau_safe,
        "selection_source": selection_source,
        "selection_status": selection_status,
        "validation": val_metrics,
        "test": test_metrics,
    }


def build_no_feasible_point(system, selection_source):
    return {
        "system": system,
        "point_type": "class_specific",
        "tau": None,
        "tau_threat": None,
        "tau_safe": None,
        "selection_source": selection_source,
        "selection_status": "no_feasible_point",
        "validation": None,
        "test": None,
    }


def build_baseline_point(system, point_type, tau, exp14_metrics, selection_source):
    return {
        "system": system,
        "point_type": point_type,
        "tau": tau,
        "tau_threat": None,
        "tau_safe": None,
        "selection_source": selection_source,
        "selection_status": "baseline",
        "validation": exp14_metrics["validation"],
        "test": exp14_metrics["test"],
    }


def select_operating_points(grid_df, test_df, exp14_result):
    points = [
        build_baseline_point("TF-IDF standalone", "baseline", None, exp14_result["tfidf"], "exp14_result"),
        build_baseline_point("RoBERTa standalone", "baseline", None, exp14_result["roberta"], "exp14_result"),
        build_baseline_point(
            "symmetric cascade from exp14",
            "symmetric",
            float(exp14_result["symmetric_cascade"]["selected_tau"]),
            exp14_result["symmetric_cascade"],
            "exp14_result",
        ),
    ]

    selected_specs = []
    selected_specs.append(("best_F1 class-specific", best_f1_row(grid_df), "selected_from_validation_grid"))

    cost_row, cost_source = select_cost_first_safety(grid_df, exp14_result)
    selected_specs.append(("cost_first_safety", cost_row, cost_source))

    quality_row, quality_source = select_quality_first_safety(grid_df, exp14_result)
    selected_specs.append(("quality_first_safety", quality_row, quality_source))

    safety_row, safety_source = select_safety_first(grid_df, exp14_result)
    selected_specs.append(("safety_first", safety_row, safety_source))

    target_50_row, target_50_source = select_target_50_compute(grid_df, exp14_result)
    selected_specs.append(("target_50_compute", target_50_row, target_50_source))

    for system, row, source in selected_specs:
        if row is None:
            points.append(build_no_feasible_point(system, source))
            continue
        points.append(build_class_specific_point(system, row, test_df, source))

    return points


def save_operating_points_csv(points):
    fieldnames = [
        "system",
        "tau",
        "tau_threat",
        "tau_safe",
        "validation_macro_f1",
        "validation_fnr",
        "validation_tier1_handled_percent",
        "validation_tier2_escalated_percent",
        "test_macro_f1",
        "test_fnr",
        "test_tier1_handled_percent",
        "test_tier2_escalated_percent",
        "selection_status",
        "selection_source",
    ]
    with open(OPERATING_POINTS_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for point in points:
            val = point["validation"] or {}
            test = point["test"] or {}
            writer.writerow({
                "system": point["system"],
                "tau": point["tau"],
                "tau_threat": point["tau_threat"],
                "tau_safe": point["tau_safe"],
                "validation_macro_f1": val.get("macro_f1"),
                "validation_fnr": val.get("fnr"),
                "validation_tier1_handled_percent": val.get("tier1_handled_percent"),
                "validation_tier2_escalated_percent": val.get("tier2_escalated_percent"),
                "test_macro_f1": test.get("macro_f1"),
                "test_fnr": test.get("fnr"),
                "test_tier1_handled_percent": test.get("tier1_handled_percent"),
                "test_tier2_escalated_percent": test.get("tier2_escalated_percent"),
                "selection_status": point["selection_status"],
                "selection_source": point["selection_source"],
            })


def format_value(value, digits=4):
    if value is None:
        return "-"
    return f"{float(value):.{digits}f}"


def print_final_table(points):
    print("\nFinal OLID Pareto operating-point table")
    print(
        "System                              tau_threat  tau_safe  "
        "Val_F1  Val_FNR  Val_Tier1%  Val_Tier2%  "
        "Test_F1  Test_FNR  Test_Tier1%  Test_Tier2%"
    )

    for point in points:
        val = point["validation"] or {}
        test = point["test"] or {}
        print(
            f"{point['system']:<35} "
            f"{format_value(point['tau_threat'], 3):>10}  "
            f"{format_value(point['tau_safe'], 3):>8}  "
            f"{format_value(val.get('macro_f1')):>6}  "
            f"{format_value(val.get('fnr')):>7}  "
            f"{format_value(val.get('tier1_handled_percent'), 2):>10}  "
            f"{format_value(val.get('tier2_escalated_percent'), 2):>10}  "
            f"{format_value(test.get('macro_f1')):>7}  "
            f"{format_value(test.get('fnr')):>8}  "
            f"{format_value(test.get('tier1_handled_percent'), 2):>11}  "
            f"{format_value(test.get('tier2_escalated_percent'), 2):>11}"
        )


def selected_plot_points(points):
    rows = []
    for point in points:
        if point["validation"] is None:
            continue
        rows.append({
            "system": point["system"],
            "tier2": point["validation"]["tier2_escalated_percent"],
            "macro_f1": point["validation"]["macro_f1"],
            "fnr": point["validation"]["fnr"],
        })
    return rows


def annotate_points(ax, rows, y_key):
    offsets = [(6, 6), (6, -10), (-54, 8), (-54, -12), (10, 14), (-68, 0), (8, -20), (-76, 18)]
    for idx, row in enumerate(rows):
        dx, dy = offsets[idx % len(offsets)]
        ax.annotate(
            row["system"],
            (row["tier2"], row[y_key]),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=8,
            arrowprops={"arrowstyle": "-", "linewidth": 0.4, "color": "0.35"},
        )


def plot_frontier(grid_df, pareto_df, points, exp14_result, y_key, y_label, ref_value, path):
    fig, ax = plt.subplots(figsize=(9.5, 5.6))

    ax.scatter(
        grid_df["validation_tier2_escalated_percent"],
        grid_df[y_key],
        s=12,
        color="0.82",
        alpha=0.55,
        label="All threshold pairs",
    )
    ax.scatter(
        pareto_df["validation_tier2_escalated_percent"],
        pareto_df[y_key],
        s=22,
        color="0.35",
        alpha=0.8,
        label="Pareto frontier",
    )

    selected_rows = selected_plot_points(points)
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(selected_rows), 1)))
    for color, row in zip(colors, selected_rows):
        ax.scatter(
            row["tier2"],
            row["macro_f1" if y_key == "validation_macro_f1" else "fnr"],
            s=64,
            color=color,
            edgecolor="black",
            linewidth=0.6,
            label=row["system"],
            zorder=4,
        )

    ax.axhline(
        ref_value,
        color="black",
        linestyle="--",
        linewidth=1.0,
        label="RoBERTa validation reference",
    )

    annotate_points(
        ax,
        selected_rows,
        "macro_f1" if y_key == "validation_macro_f1" else "fnr",
    )

    ax.set_xlabel("Validation Tier 2 escalated %")
    ax.set_ylabel(y_label)
    ax.set_title(y_label + " vs validation Tier 2 escalation")
    ax.grid(True, color="0.9", linewidth=0.7)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def interpret(points, grid_df, exp14_result):
    selected = [point for point in points if point["validation"] is not None]
    class_specific = [point for point in selected if point["point_type"] == "class_specific"]
    roberta_val = exp14_result["roberta"]["validation"]

    f1_best = max(class_specific, key=lambda point: point["validation"]["macro_f1"])
    fnr_best = min(class_specific, key=lambda point: point["validation"]["fnr"])

    roberta_level_grid = grid_df[
        (grid_df["validation_tier2_escalated_percent"] <= 50.0)
        & (grid_df["validation_macro_f1"] >= roberta_val["macro_f1"])
        & (grid_df["validation_fnr"] <= roberta_val["fnr"])
    ]

    return {
        "max_validation_f1_point": f1_best["system"],
        "max_validation_f1": f1_best["validation"]["macro_f1"],
        "best_roberta_fnr_preservation_point": fnr_best["system"],
        "best_roberta_fnr_preservation_fnr": fnr_best["validation"]["fnr"],
        "roberta_validation_macro_f1": roberta_val["macro_f1"],
        "roberta_validation_fnr": roberta_val["fnr"],
        "roberta_level_f1_and_fnr_with_tier2_lte_50_exists": bool(len(roberta_level_grid) > 0),
        "roberta_level_f1_and_fnr_with_tier2_lte_50_grid_rows": int(len(roberta_level_grid)),
    }


def print_interpretation(summary):
    print("\nInterpretation")
    print(
        f"- The validation F1-maximizing selected point is "
        f"{summary['max_validation_f1_point']} "
        f"(Val F1={summary['max_validation_f1']:.4f})."
    )
    print(
        f"- The selected point with the lowest validation FNR is "
        f"{summary['best_roberta_fnr_preservation_point']} "
        f"(Val FNR={summary['best_roberta_fnr_preservation_fnr']:.4f}; "
        f"RoBERTa Val FNR={summary['roberta_validation_fnr']:.4f})."
    )
    if summary["roberta_level_f1_and_fnr_with_tier2_lte_50_exists"]:
        print("- At least one validation-grid point reaches RoBERTa-level validation F1 and FNR with Tier2<=50%.")
    else:
        print("- No validation-grid point reaches RoBERTa-level validation F1 and FNR with Tier2<=50%; this is a limitation of the OLID cascade trade-off.")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    grid_df = load_validation_grid()
    test_df = pd.read_csv(TEST_PREDICTIONS_PATH)
    validate_test_predictions(test_df)

    with open(EXP14_RESULT_PATH) as f:
        exp14_result = json.load(f)

    pareto_df = compute_pareto_frontier(grid_df)
    pareto_df.to_csv(PARETO_PATH, index=False)

    points = select_operating_points(grid_df, test_df, exp14_result)
    save_operating_points_csv(points)

    plot_frontier(
        grid_df,
        pareto_df,
        points,
        exp14_result,
        "validation_macro_f1",
        "Validation Macro F1",
        exp14_result["roberta"]["validation"]["macro_f1"],
        FIG_F1_PATH,
    )
    plot_frontier(
        grid_df,
        pareto_df,
        points,
        exp14_result,
        "validation_fnr",
        "Validation FNR",
        exp14_result["roberta"]["validation"]["fnr"],
        FIG_FNR_PATH,
    )

    interpretation = interpret(points, grid_df, exp14_result)

    result = {
        "experiment": "exp14d_olid_pareto_operating_points",
        "dataset": exp14_result["dataset_name"],
        "selection_split": "internal_validation_from_official_train",
        "official_test_usage": "final_evaluation_only_after_validation_thresholds_locked",
        "used_saved_grid": str(GRID_PATH),
        "used_saved_test_predictions": str(TEST_PREDICTIONS_PATH),
        "retrained_models": False,
        "refit_tfidf": False,
        "detected_grid_columns": grid_df.columns.tolist(),
        "pareto_frontier_rows": int(len(pareto_df)),
        "roberta_validation": exp14_result["roberta"]["validation"],
        "operating_points": points,
        "interpretation": interpretation,
        "outputs": {
            "pareto_frontier_csv": str(PARETO_PATH),
            "operating_points_csv": str(OPERATING_POINTS_PATH),
            "result_json": str(RESULT_PATH),
            "figure_f1_vs_tier2": str(FIG_F1_PATH),
            "figure_fnr_vs_tier2": str(FIG_FNR_PATH),
        },
    }

    with open(RESULT_PATH, "w") as f:
        json.dump(result, f, indent=2)

    print_final_table(points)
    print_interpretation(interpretation)
    print(f"\nSaved Pareto frontier to: {PARETO_PATH}")
    print(f"Saved operating points to: {OPERATING_POINTS_PATH}")
    print(f"Saved result JSON to: {RESULT_PATH}")
    print(f"Saved F1 plot to: {FIG_F1_PATH}")
    print(f"Saved FNR plot to: {FIG_FNR_PATH}")


if __name__ == "__main__":
    main()
