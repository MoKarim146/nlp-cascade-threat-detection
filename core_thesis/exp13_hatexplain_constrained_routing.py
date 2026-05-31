"""Experiment 13 (variant) — Constrained routing with Pareto frontier annotation on HateXplain."""
from collections import Counter
from pathlib import Path
import csv
import json

import numpy as np
import torch
from datasets import load_dataset
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer


SAFE = 0
THREAT = 1

RANDOM_SEED = 42
MAX_LENGTH = 128
BATCH_SIZE = 16

INITIAL_F1_DELTA = 0.002
RELAXED_F1_DELTA = 0.005
FNR_DELTA = 0.010
SYMMETRIC_BASELINE_THRESHOLD = 0.70
THRESHOLDS = np.round(np.arange(0.50, 0.951, 0.01), 3)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ROBERTA_MODEL_DIR = PROJECT_ROOT / "core_thesis" / "models" / "roberta_hatexplain"
OUTPUT_DIR = PROJECT_ROOT / "core_thesis" / "outputs"
GRID_PATH = OUTPUT_DIR / "exp13_hatexplain_constrained_routing_grid.csv"
RESULT_PATH = OUTPUT_DIR / "exp13_hatexplain_constrained_routing_result.json"


def majority_vote(labels):
    return Counter(labels).most_common(1)[0][0]


def to_binary_label(label_id):
    # HateXplain: 0=hatespeech, 1=normal, 2=offensive
    # Thesis setup: SAFE=0, THREAT=1
    if label_id == 1:
        return SAFE
    if label_id in {0, 2}:
        return THREAT
    raise ValueError(f"Unexpected HateXplain label id: {label_id!r}")


def prepare_split(split):
    texts, labels = [], []

    for row in split:
        text = " ".join(row["post_tokens"]).strip()
        majority_label = majority_vote(row["annotators"]["label"])
        binary_label = to_binary_label(majority_label)

        if text:
            texts.append(text)
            labels.append(binary_label)

    return texts, np.array(labels)


def build_tfidf(train_texts, val_texts, test_texts):
    word_vec = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        max_features=20_000,
        sublinear_tf=True,
        min_df=2,
        lowercase=True,
    )

    char_vec = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        max_features=15_000,
        sublinear_tf=True,
        min_df=2,
        lowercase=True,
    )

    X_train = hstack([
        word_vec.fit_transform(train_texts),
        char_vec.fit_transform(train_texts),
    ])

    X_val = hstack([
        word_vec.transform(val_texts),
        char_vec.transform(val_texts),
    ])

    X_test = hstack([
        word_vec.transform(test_texts),
        char_vec.transform(test_texts),
    ])

    return X_train, X_val, X_test


def train_tfidf_classifier(X_train, y_train):
    clf = LogisticRegression(
        C=1.0,
        class_weight="balanced",
        max_iter=1000,
        solver="lbfgs",
        random_state=RANDOM_SEED,
    )
    clf.fit(X_train, y_train)
    return clf


def get_tfidf_outputs(clf, X):
    probs = clf.predict_proba(X)
    preds = np.argmax(probs, axis=1)
    conf = np.max(probs, axis=1)
    return preds, conf


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def get_roberta_predictions(texts):
    if not ROBERTA_MODEL_DIR.exists():
        raise FileNotFoundError(
            f"RoBERTa model not found at: {ROBERTA_MODEL_DIR}\n"
            "Run exp3_hatexplain_roberta.py first."
        )

    device = get_device()
    tokenizer = AutoTokenizer.from_pretrained(ROBERTA_MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(ROBERTA_MODEL_DIR)

    model.to(device)
    model.eval()

    all_preds = []

    for start in range(0, len(texts), BATCH_SIZE):
        batch_texts = texts[start:start + BATCH_SIZE]

        encoded = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )

        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        probs = torch.softmax(outputs.logits, dim=1)
        preds = torch.argmax(probs, dim=1)

        all_preds.extend(preds.cpu().numpy().tolist())

    return np.array(all_preds)


def false_negative_rate(y_true, y_pred):
    total_threats = np.sum(y_true == THREAT)

    if total_threats == 0:
        return 0.0

    missed_threats = np.sum((y_true == THREAT) & (y_pred == SAFE))
    return float(missed_threats / total_threats)


def run_class_specific_cascade(
    tfidf_preds,
    tfidf_conf,
    roberta_preds,
    tau_threat,
    tau_safe,
):
    tfidf_says_threat = tfidf_preds == THREAT
    tfidf_says_safe = tfidf_preds == SAFE

    keep_threat = tfidf_says_threat & (tfidf_conf >= tau_threat)
    keep_safe = tfidf_says_safe & (tfidf_conf >= tau_safe)

    keep_tfidf = keep_threat | keep_safe
    escalate_mask = ~keep_tfidf

    cascade_preds = tfidf_preds.copy()
    cascade_preds[escalate_mask] = roberta_preds[escalate_mask]

    tier1_fraction = float(np.mean(keep_tfidf))
    tier2_fraction = float(np.mean(escalate_mask))

    return cascade_preds, tier1_fraction, tier2_fraction


def compute_metrics(y_true, preds, tier1_fraction, tier2_fraction):
    return {
        "macro_f1": float(f1_score(y_true, preds, average="macro")),
        "fnr": false_negative_rate(y_true, preds),
        "tier1_handled_percent": 100.0 * tier1_fraction,
        "tier2_escalated_percent": 100.0 * tier2_fraction,
    }


def evaluate_threshold_pair(
    y_true,
    tfidf_preds,
    tfidf_conf,
    roberta_preds,
    tau_threat,
    tau_safe,
):
    preds, tier1_fraction, tier2_fraction = run_class_specific_cascade(
        tfidf_preds,
        tfidf_conf,
        roberta_preds,
        tau_threat,
        tau_safe,
    )
    return compute_metrics(y_true, preds, tier1_fraction, tier2_fraction)


def evaluate_roberta_baseline(y_true, roberta_preds):
    return {
        "macro_f1": float(f1_score(y_true, roberta_preds, average="macro")),
        "fnr": false_negative_rate(y_true, roberta_preds),
        "tier1_handled_percent": 0.0,
        "tier2_escalated_percent": 100.0,
    }


def build_validation_grid(y_val, val_tfidf_preds, val_tfidf_conf, val_roberta_preds):
    rows = []

    for tau_threat in THRESHOLDS:
        for tau_safe in THRESHOLDS:
            metrics = evaluate_threshold_pair(
                y_val,
                val_tfidf_preds,
                val_tfidf_conf,
                val_roberta_preds,
                tau_threat,
                tau_safe,
            )
            rows.append({
                "tau_threat": float(tau_threat),
                "tau_safe": float(tau_safe),
                "validation_macro_f1": metrics["macro_f1"],
                "validation_fnr": metrics["fnr"],
                "validation_tier1_handled_percent": metrics["tier1_handled_percent"],
                "validation_tier2_escalated_percent": metrics["tier2_escalated_percent"],
            })

    return rows


def mark_pareto_points(rows):
    # Minimize validation FNR and Tier 2 escalated percentage.
    sorted_indices = sorted(
        range(len(rows)),
        key=lambda idx: (
            rows[idx]["validation_fnr"],
            rows[idx]["validation_tier2_escalated_percent"],
        ),
    )

    best_tier2_so_far = float("inf")

    for row in rows:
        row["is_pareto_optimal"] = False

    for idx in sorted_indices:
        row = rows[idx]
        tier2 = row["validation_tier2_escalated_percent"]

        if tier2 < best_tier2_so_far:
            row["is_pareto_optimal"] = True
            best_tier2_so_far = tier2

    return sum(1 for row in rows if row["is_pareto_optimal"])


def mark_feasible_rows(rows, roberta_val_metrics):
    initial_f1_floor = roberta_val_metrics["macro_f1"] - INITIAL_F1_DELTA
    relaxed_f1_floor = roberta_val_metrics["macro_f1"] - RELAXED_F1_DELTA
    fnr_ceiling = roberta_val_metrics["fnr"] + FNR_DELTA

    for row in rows:
        row["feasible_initial"] = (
            row["validation_macro_f1"] >= initial_f1_floor
            and row["validation_fnr"] <= fnr_ceiling
        )
        row["feasible_relaxed"] = (
            row["validation_macro_f1"] >= relaxed_f1_floor
            and row["validation_fnr"] <= fnr_ceiling
        )


def select_cheapest_feasible(rows):
    feasible_initial = [row for row in rows if row["feasible_initial"]]

    if feasible_initial:
        feasible = feasible_initial
        relaxation_used = False
        f1_delta_used = INITIAL_F1_DELTA
    else:
        feasible = [row for row in rows if row["feasible_relaxed"]]
        relaxation_used = True
        f1_delta_used = RELAXED_F1_DELTA

    if not feasible:
        raise RuntimeError(
            "No threshold pair satisfied the safety constraints, even after "
            f"relaxing the Macro F1 delta to {RELAXED_F1_DELTA:.3f}."
        )

    selected = min(
        feasible,
        key=lambda row: (
            row["validation_tier2_escalated_percent"],
            -row["validation_macro_f1"],
            row["validation_fnr"],
            row["tau_threat"],
            row["tau_safe"],
        ),
    )

    return selected, relaxation_used, f1_delta_used


def save_grid_csv(rows):
    fieldnames = [
        "tau_threat",
        "tau_safe",
        "validation_macro_f1",
        "validation_fnr",
        "validation_tier1_handled_percent",
        "validation_tier2_escalated_percent",
        "feasible_initial",
        "feasible_relaxed",
        "is_pareto_optimal",
    ]

    with open(GRID_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_metric_line(label, metrics):
    print(
        f"{label:<26} "
        f"F1={metrics['macro_f1']:.4f}  "
        f"FNR={metrics['fnr']:.4f}  "
        f"Tier1={metrics['tier1_handled_percent']:.2f}%  "
        f"Tier2={metrics['tier2_escalated_percent']:.2f}%"
    )


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    ds = load_dataset("hatexplain")

    train_texts, y_train = prepare_split(ds["train"])
    val_texts, y_val = prepare_split(ds["validation"])
    test_texts, y_test = prepare_split(ds["test"])

    print("Split sizes:")
    print("Train:", len(train_texts))
    print("Validation:", len(val_texts))
    print("Test:", len(test_texts))

    X_train, X_val, X_test = build_tfidf(train_texts, val_texts, test_texts)
    tfidf_clf = train_tfidf_classifier(X_train, y_train)

    val_tfidf_preds, val_tfidf_conf = get_tfidf_outputs(tfidf_clf, X_val)
    test_tfidf_preds, test_tfidf_conf = get_tfidf_outputs(tfidf_clf, X_test)

    print("\nRunning RoBERTa on validation split...")
    val_roberta_preds = get_roberta_predictions(val_texts)

    print("Running RoBERTa on test split...")
    test_roberta_preds = get_roberta_predictions(test_texts)

    roberta_val_metrics = evaluate_roberta_baseline(y_val, val_roberta_preds)
    roberta_test_metrics = evaluate_roberta_baseline(y_test, test_roberta_preds)

    grid_rows = build_validation_grid(
        y_val,
        val_tfidf_preds,
        val_tfidf_conf,
        val_roberta_preds,
    )
    pareto_count = mark_pareto_points(grid_rows)
    mark_feasible_rows(grid_rows, roberta_val_metrics)

    selected_row, relaxation_used, f1_delta_used = select_cheapest_feasible(grid_rows)

    selected_val_metrics = {
        "macro_f1": selected_row["validation_macro_f1"],
        "fnr": selected_row["validation_fnr"],
        "tier1_handled_percent": selected_row["validation_tier1_handled_percent"],
        "tier2_escalated_percent": selected_row["validation_tier2_escalated_percent"],
    }

    selected_test_metrics = evaluate_threshold_pair(
        y_test,
        test_tfidf_preds,
        test_tfidf_conf,
        test_roberta_preds,
        selected_row["tau_threat"],
        selected_row["tau_safe"],
    )

    symmetric_val_metrics = evaluate_threshold_pair(
        y_val,
        val_tfidf_preds,
        val_tfidf_conf,
        val_roberta_preds,
        SYMMETRIC_BASELINE_THRESHOLD,
        SYMMETRIC_BASELINE_THRESHOLD,
    )
    symmetric_test_metrics = evaluate_threshold_pair(
        y_test,
        test_tfidf_preds,
        test_tfidf_conf,
        test_roberta_preds,
        SYMMETRIC_BASELINE_THRESHOLD,
        SYMMETRIC_BASELINE_THRESHOLD,
    )

    save_grid_csv(grid_rows)

    result = {
        "experiment": "exp13_hatexplain_constrained_routing",
        "dataset": "hatexplain",
        "splits": {
            "train_size": len(train_texts),
            "validation_size": len(val_texts),
            "test_size": len(test_texts),
        },
        "selection_protocol": {
            "threshold_grid_start": float(THRESHOLDS[0]),
            "threshold_grid_end": float(THRESHOLDS[-1]),
            "threshold_grid_step": 0.01,
            "selection_split": "validation",
            "objective": "cheapest feasible threshold pair",
            "cost_proxy": "minimum validation Tier 2 escalated percent",
            "initial_f1_delta": INITIAL_F1_DELTA,
            "relaxed_f1_delta": RELAXED_F1_DELTA,
            "fnr_delta": FNR_DELTA,
            "relaxation_used": relaxation_used,
            "f1_delta_used": f1_delta_used,
        },
        "roberta_baseline": {
            "validation": roberta_val_metrics,
            "test": roberta_test_metrics,
        },
        "selected_thresholds": {
            "tau_threat": selected_row["tau_threat"],
            "tau_safe": selected_row["tau_safe"],
        },
        "selected_result": {
            "validation": selected_val_metrics,
            "test": selected_test_metrics,
        },
        "symmetric_tau_0_70": {
            "tau_threat": SYMMETRIC_BASELINE_THRESHOLD,
            "tau_safe": SYMMETRIC_BASELINE_THRESHOLD,
            "validation": symmetric_val_metrics,
            "test": symmetric_test_metrics,
        },
        "pareto_optimal_validation_points": pareto_count,
        "grid_csv": str(GRID_PATH),
    }

    with open(RESULT_PATH, "w") as f:
        json.dump(result, f, indent=2)

    print("\nRoBERTa baselines:")
    print_metric_line("Validation RoBERTa", roberta_val_metrics)
    print_metric_line("Test RoBERTa", roberta_test_metrics)

    print("\nSelected constrained routing:")
    print(f"tau_threat={selected_row['tau_threat']:.3f}")
    print(f"tau_safe={selected_row['tau_safe']:.3f}")
    print(f"relaxation_used={relaxation_used}")
    print(f"f1_delta_used={f1_delta_used:.3f}")
    print_metric_line("Validation selected", selected_val_metrics)
    print_metric_line("Test selected", selected_test_metrics)

    print(f"\nPareto-optimal validation points: {pareto_count}")

    print(f"\nComparison against symmetric tau={SYMMETRIC_BASELINE_THRESHOLD:.2f}:")
    print_metric_line("Validation symmetric", symmetric_val_metrics)
    print_metric_line("Test symmetric", symmetric_test_metrics)

    print(f"\nSaved grid CSV to: {GRID_PATH}")
    print(f"Saved selected result JSON to: {RESULT_PATH}")


if __name__ == "__main__":
    main()
