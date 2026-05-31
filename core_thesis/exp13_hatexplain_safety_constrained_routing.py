"""Experiment 13 — Safety-constrained class-specific cascade routing on HateXplain (main result)."""
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

DELTA_F1_INITIAL = 0.002
DELTA_F1_RELAXED = 0.005
EPSILON_FNR = 0.010
SYMMETRIC_TAU = 0.70
THRESHOLDS = np.round(np.arange(0.50, 0.951, 0.01), 3)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ROBERTA_MODEL_DIR = PROJECT_ROOT / "core_thesis" / "models" / "roberta_hatexplain"
OUTPUT_DIR = PROJECT_ROOT / "core_thesis" / "outputs"
GRID_PATH = OUTPUT_DIR / "exp13_hatexplain_safety_constrained_grid.csv"
RESULT_PATH = OUTPUT_DIR / "exp13_hatexplain_safety_constrained_result.json"


def majority_vote(labels):
    return Counter(labels).most_common(1)[0][0]


def to_binary_label(label_id):
    # HateXplain: 0=hatespeech, 1=normal, 2=offensive
    # Thesis binary setup: SAFE=0, THREAT=1
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

    false_negatives = np.sum((y_true == THREAT) & (y_pred == SAFE))
    return float(false_negatives / total_threats)


def run_class_specific_routing(
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
    escalate = ~keep_tfidf

    final_preds = tfidf_preds.copy()
    final_preds[escalate] = roberta_preds[escalate]

    return final_preds, float(np.mean(keep_tfidf)), float(np.mean(escalate))


def metrics_from_predictions(y_true, preds, tier1_fraction, tier2_fraction):
    return {
        "macro_f1": float(f1_score(y_true, preds, average="macro")),
        "fnr": false_negative_rate(y_true, preds),
        "tier1_handled_percent": 100.0 * tier1_fraction,
        "tier2_escalated_percent": 100.0 * tier2_fraction,
    }


def standalone_metrics(y_true, preds, tier1_fraction, tier2_fraction):
    return metrics_from_predictions(y_true, preds, tier1_fraction, tier2_fraction)


def evaluate_threshold_pair(
    y_true,
    tfidf_preds,
    tfidf_conf,
    roberta_preds,
    tau_threat,
    tau_safe,
):
    preds, tier1_fraction, tier2_fraction = run_class_specific_routing(
        tfidf_preds,
        tfidf_conf,
        roberta_preds,
        tau_threat,
        tau_safe,
    )
    return metrics_from_predictions(y_true, preds, tier1_fraction, tier2_fraction)


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


def annotate_feasibility(rows, roberta_val_metrics):
    initial_f1_floor = roberta_val_metrics["macro_f1"] - DELTA_F1_INITIAL
    relaxed_f1_floor = roberta_val_metrics["macro_f1"] - DELTA_F1_RELAXED
    fnr_ceiling = roberta_val_metrics["fnr"] + EPSILON_FNR

    for row in rows:
        row["feasible_initial"] = (
            row["validation_macro_f1"] >= initial_f1_floor
            and row["validation_fnr"] <= fnr_ceiling
        )
        row["feasible_relaxed"] = (
            row["validation_macro_f1"] >= relaxed_f1_floor
            and row["validation_fnr"] <= fnr_ceiling
        )


def select_constrained_threshold(rows):
    feasible_initial = [row for row in rows if row["feasible_initial"]]

    if feasible_initial:
        selected = cheapest_row(feasible_initial)
        return selected, {
            "selection_status": "feasible_initial",
            "delta_f1_used": DELTA_F1_INITIAL,
            "fallback_used": False,
        }

    feasible_relaxed = [row for row in rows if row["feasible_relaxed"]]

    if feasible_relaxed:
        selected = cheapest_row(feasible_relaxed)
        return selected, {
            "selection_status": "feasible_relaxed",
            "delta_f1_used": DELTA_F1_RELAXED,
            "fallback_used": False,
        }

    selected = best_validation_f1_row(rows)
    return selected, {
        "selection_status": "fallback_best_validation_f1",
        "delta_f1_used": None,
        "fallback_used": True,
    }


def cheapest_row(rows):
    return min(
        rows,
        key=lambda row: (
            row["validation_tier2_escalated_percent"],
            -row["validation_macro_f1"],
            row["validation_fnr"],
            row["tau_threat"],
            row["tau_safe"],
        ),
    )


def best_validation_f1_row(rows):
    return max(
        rows,
        key=lambda row: (
            row["validation_macro_f1"],
            -row["validation_fnr"],
            row["validation_tier1_handled_percent"],
            -row["tau_threat"],
            -row["tau_safe"],
        ),
    )


def row_to_metrics(row):
    return {
        "macro_f1": row["validation_macro_f1"],
        "fnr": row["validation_fnr"],
        "tier1_handled_percent": row["validation_tier1_handled_percent"],
        "tier2_escalated_percent": row["validation_tier2_escalated_percent"],
    }


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
    ]

    with open(GRID_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def format_tau(value):
    if value is None:
        return "-"
    return f"{value:.3f}"


def print_final_table(rows):
    print("\nFinal comparison table")
    print(
        "System                         "
        "tau_threat  tau_safe  Val_F1  Val_FNR  Val_Tier1%  "
        "Test_F1  Test_FNR  Test_Tier1%  Test_Tier2%"
    )

    for row in rows:
        val_metrics = row["validation"]
        test_metrics = row["test"]
        print(
            f"{row['system']:<30} "
            f"{format_tau(row['tau_threat']):>10}  "
            f"{format_tau(row['tau_safe']):>8}  "
            f"{val_metrics['macro_f1']:.4f}  "
            f"{val_metrics['fnr']:.4f}  "
            f"{val_metrics['tier1_handled_percent']:10.2f}  "
            f"{test_metrics['macro_f1']:.4f}  "
            f"{test_metrics['fnr']:.4f}  "
            f"{test_metrics['tier1_handled_percent']:11.2f}  "
            f"{test_metrics['tier2_escalated_percent']:11.2f}"
        )


def build_table_row(system, tau_threat, tau_safe, validation, test):
    return {
        "system": system,
        "tau_threat": tau_threat,
        "tau_safe": tau_safe,
        "validation": validation,
        "test": test,
    }


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

    tfidf_val_metrics = standalone_metrics(y_val, val_tfidf_preds, 1.0, 0.0)
    tfidf_test_metrics = standalone_metrics(y_test, test_tfidf_preds, 1.0, 0.0)

    roberta_val_metrics = standalone_metrics(y_val, val_roberta_preds, 0.0, 1.0)
    roberta_test_metrics = standalone_metrics(y_test, test_roberta_preds, 0.0, 1.0)

    grid_rows = build_validation_grid(
        y_val,
        val_tfidf_preds,
        val_tfidf_conf,
        val_roberta_preds,
    )
    annotate_feasibility(grid_rows, roberta_val_metrics)

    constrained_row, selection_info = select_constrained_threshold(grid_rows)
    best_f1_row = best_validation_f1_row(grid_rows)

    constrained_val_metrics = row_to_metrics(constrained_row)
    constrained_test_metrics = evaluate_threshold_pair(
        y_test,
        test_tfidf_preds,
        test_tfidf_conf,
        test_roberta_preds,
        constrained_row["tau_threat"],
        constrained_row["tau_safe"],
    )

    best_f1_val_metrics = row_to_metrics(best_f1_row)
    best_f1_test_metrics = evaluate_threshold_pair(
        y_test,
        test_tfidf_preds,
        test_tfidf_conf,
        test_roberta_preds,
        best_f1_row["tau_threat"],
        best_f1_row["tau_safe"],
    )

    symmetric_val_metrics = evaluate_threshold_pair(
        y_val,
        val_tfidf_preds,
        val_tfidf_conf,
        val_roberta_preds,
        SYMMETRIC_TAU,
        SYMMETRIC_TAU,
    )
    symmetric_test_metrics = evaluate_threshold_pair(
        y_test,
        test_tfidf_preds,
        test_tfidf_conf,
        test_roberta_preds,
        SYMMETRIC_TAU,
        SYMMETRIC_TAU,
    )

    table_rows = [
        build_table_row("TF-IDF standalone", None, None, tfidf_val_metrics, tfidf_test_metrics),
        build_table_row("RoBERTa standalone", None, None, roberta_val_metrics, roberta_test_metrics),
        build_table_row(
            "symmetric tau=0.70",
            SYMMETRIC_TAU,
            SYMMETRIC_TAU,
            symmetric_val_metrics,
            symmetric_test_metrics,
        ),
        build_table_row(
            "best-F1 class-specific",
            best_f1_row["tau_threat"],
            best_f1_row["tau_safe"],
            best_f1_val_metrics,
            best_f1_test_metrics,
        ),
        build_table_row(
            "constrained class-specific",
            constrained_row["tau_threat"],
            constrained_row["tau_safe"],
            constrained_val_metrics,
            constrained_test_metrics,
        ),
    ]

    save_grid_csv(grid_rows)

    result = {
        "experiment": "exp13_hatexplain_safety_constrained_routing",
        "dataset": "hatexplain",
        "splits": {
            "train_size": len(train_texts),
            "validation_size": len(val_texts),
            "test_size": len(test_texts),
        },
        "selection_rule": {
            "selection_split": "validation",
            "delta_f1_initial": DELTA_F1_INITIAL,
            "delta_f1_relaxed": DELTA_F1_RELAXED,
            "epsilon_fnr": EPSILON_FNR,
            "roberta_validation_macro_f1": roberta_val_metrics["macro_f1"],
            "roberta_validation_fnr": roberta_val_metrics["fnr"],
            **selection_info,
        },
        "selected_thresholds": {
            "tau_threat": constrained_row["tau_threat"],
            "tau_safe": constrained_row["tau_safe"],
        },
        "best_f1_thresholds": {
            "tau_threat": best_f1_row["tau_threat"],
            "tau_safe": best_f1_row["tau_safe"],
        },
        "systems": table_rows,
        "grid_csv": str(GRID_PATH),
    }

    with open(RESULT_PATH, "w") as f:
        json.dump(result, f, indent=2)

    print_final_table(table_rows)

    print("\nSelection status:")
    print(f"status: {selection_info['selection_status']}")
    print(f"fallback_used: {selection_info['fallback_used']}")
    print(f"selected tau_threat: {constrained_row['tau_threat']:.3f}")
    print(f"selected tau_safe:   {constrained_row['tau_safe']:.3f}")

    print(f"\nSaved full validation grid to: {GRID_PATH}")
    print(f"Saved selected result JSON to: {RESULT_PATH}")


if __name__ == "__main__":
    main()
