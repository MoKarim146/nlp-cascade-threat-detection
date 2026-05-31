"""
Experiment 14 — Full cascade package for OLID: symmetric + best-F1 + safety-constrained routing.

Protocol guardrail: the official test split is loaded for final evaluation only.
Thresholds and constraints are selected exclusively on the internal validation
split created from official train.
"""
from pathlib import Path
import csv
import json
import sys

import numpy as np
import torch
from datasets import load_dataset
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from transformers import AutoModelForSequenceClassification, AutoTokenizer


SAFE = 0
THREAT = 1

DATASET_NAME = "christophsonntag/OLID"
RANDOM_SEED = 42
VALIDATION_SIZE = 0.20

MAX_LENGTH = 128
BATCH_SIZE = 16

DELTA_F1_INITIAL = 0.002
DELTA_F1_RELAXED = 0.005
EPSILON_FNR = 0.010
SYMMETRIC_THRESHOLDS = np.round(np.arange(0.50, 0.991, 0.01), 3)
CLASS_THRESHOLDS = np.round(np.arange(0.50, 0.991, 0.01), 3)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CORE_DIR = PROJECT_ROOT / "core_thesis"
MODEL_DIR = CORE_DIR / "models" / "roberta_olid_subtrain_clean"
OUTPUT_DIR = CORE_DIR / "outputs"

GRID_PATH = OUTPUT_DIR / "exp14_olid_class_specific_grid.csv"
RESULT_PATH = OUTPUT_DIR / "exp14_olid_clean_cascade_package_result.json"
PREDICTIONS_PATH = OUTPUT_DIR / "exp14_olid_clean_cascade_test_predictions.csv"

TFIDF_SETTINGS = {
    "word_analyzer": "word",
    "word_ngram_range": [1, 2],
    "word_max_features": 20_000,
    "char_analyzer": "char_wb",
    "char_ngram_range": [3, 5],
    "char_max_features": 15_000,
    "sublinear_tf": True,
    "min_df": 2,
    "lowercase": True,
}

LOGREG_SETTINGS = {
    "C": 0.5,
    "class_weight": "balanced",
    "max_iter": 1000,
    "solver": "lbfgs",
    "random_state": RANDOM_SEED,
}


def require_roberta_checkpoint():
    required_files = [
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
    ]
    missing_files = [
        filename for filename in required_files if not (MODEL_DIR / filename).exists()
    ]
    has_model_weights = (
        (MODEL_DIR / "model.safetensors").exists()
        or (MODEL_DIR / "pytorch_model.bin").exists()
    )

    if missing_files or not has_model_weights:
        raise FileNotFoundError(
            "Run exp7b_olid_roberta_clean_subtrain.py first."
        )


def map_olid_label(label):
    label = str(label).strip().upper()

    if label == "NOT":
        return SAFE
    if label == "OFF":
        return THREAT

    raise ValueError(f"Unexpected OLID label: {label!r}")


def prepare_olid_split(split):
    texts, labels = [], []

    for row in split:
        text = str(row["tweet"]).strip()
        label = map_olid_label(row["subtask_a"])

        if text:
            texts.append(text)
            labels.append(label)

    return texts, np.array(labels)


def fit_tfidf_vectorizers(subtrain_texts):
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

    X_subtrain = hstack([
        word_vec.fit_transform(subtrain_texts),
        char_vec.fit_transform(subtrain_texts),
    ])

    return word_vec, char_vec, X_subtrain


def transform_tfidf(word_vec, char_vec, texts):
    return hstack([
        word_vec.transform(texts),
        char_vec.transform(texts),
    ])


def train_tfidf_classifier(X_subtrain, y_subtrain):
    clf = LogisticRegression(
        C=LOGREG_SETTINGS["C"],
        class_weight=LOGREG_SETTINGS["class_weight"],
        max_iter=LOGREG_SETTINGS["max_iter"],
        solver=LOGREG_SETTINGS["solver"],
        random_state=LOGREG_SETTINGS["random_state"],
    )
    clf.fit(X_subtrain, y_subtrain)
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
def get_roberta_predictions(texts, tokenizer, model, device):
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


def load_roberta_model():
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    except Exception as exc:
        raise RuntimeError(
            "Run exp7b_olid_roberta_clean_subtrain.py first."
        ) from exc

    device = get_device()
    model.to(device)
    model.eval()
    return tokenizer, model, device


def false_negative_rate(y_true, y_pred):
    total_threats = np.sum(y_true == THREAT)

    if total_threats == 0:
        return 0.0

    missed_threats = np.sum((y_true == THREAT) & (y_pred == SAFE))
    return float(missed_threats / total_threats)


def prediction_counts(preds):
    return {
        "SAFE": int(np.sum(preds == SAFE)),
        "THREAT": int(np.sum(preds == THREAT)),
    }


def metrics_from_predictions(y_true, preds, tier1_fraction, tier2_fraction):
    return {
        "macro_f1": float(
            f1_score(
                y_true,
                preds,
                labels=[SAFE, THREAT],
                average="macro",
                zero_division=0,
            )
        ),
        "fnr": false_negative_rate(y_true, preds),
        "tier1_handled_percent": 100.0 * tier1_fraction,
        "tier2_escalated_percent": 100.0 * tier2_fraction,
        "prediction_counts": prediction_counts(preds),
    }


def standalone_tfidf_metrics(y_true, preds):
    return metrics_from_predictions(y_true, preds, 1.0, 0.0)


def standalone_roberta_metrics(y_true, preds):
    return metrics_from_predictions(y_true, preds, 0.0, 1.0)


def run_symmetric_cascade(tfidf_preds, tfidf_conf, roberta_preds, tau):
    keep_tfidf = tfidf_conf >= tau
    escalate = ~keep_tfidf

    final_preds = tfidf_preds.copy()
    final_preds[escalate] = roberta_preds[escalate]

    return final_preds, float(np.mean(keep_tfidf)), float(np.mean(escalate))


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
    escalate = ~keep_tfidf

    final_preds = tfidf_preds.copy()
    final_preds[escalate] = roberta_preds[escalate]

    return final_preds, float(np.mean(keep_tfidf)), float(np.mean(escalate))


def evaluate_symmetric(y_true, tfidf_preds, tfidf_conf, roberta_preds, tau):
    preds, tier1_fraction, tier2_fraction = run_symmetric_cascade(
        tfidf_preds,
        tfidf_conf,
        roberta_preds,
        tau,
    )
    return metrics_from_predictions(y_true, preds, tier1_fraction, tier2_fraction)


def evaluate_class_specific(
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
    return metrics_from_predictions(y_true, preds, tier1_fraction, tier2_fraction)


def select_symmetric_threshold(y_val, val_tfidf_preds, val_tfidf_conf, val_roberta_preds):
    best = None

    for tau in SYMMETRIC_THRESHOLDS:
        metrics = evaluate_symmetric(
            y_val,
            val_tfidf_preds,
            val_tfidf_conf,
            val_roberta_preds,
            tau,
        )
        candidate = {
            "tau": float(tau),
            "metrics": metrics,
        }

        if best is None or symmetric_is_better(candidate, best):
            best = candidate

    return best


def symmetric_is_better(candidate, current):
    candidate_metrics = candidate["metrics"]
    current_metrics = current["metrics"]

    return (
        candidate_metrics["macro_f1"] > current_metrics["macro_f1"]
        or (
            candidate_metrics["macro_f1"] == current_metrics["macro_f1"]
            and candidate_metrics["fnr"] < current_metrics["fnr"]
        )
        or (
            candidate_metrics["macro_f1"] == current_metrics["macro_f1"]
            and candidate_metrics["fnr"] == current_metrics["fnr"]
            and candidate_metrics["tier1_handled_percent"]
            > current_metrics["tier1_handled_percent"]
        )
    )


def build_class_specific_grid(
    y_val,
    val_tfidf_preds,
    val_tfidf_conf,
    val_roberta_preds,
    roberta_val_metrics,
):
    rows = []
    initial_f1_floor = roberta_val_metrics["macro_f1"] - DELTA_F1_INITIAL
    relaxed_f1_floor = roberta_val_metrics["macro_f1"] - DELTA_F1_RELAXED
    fnr_ceiling = roberta_val_metrics["fnr"] + EPSILON_FNR

    for tau_threat in CLASS_THRESHOLDS:
        for tau_safe in CLASS_THRESHOLDS:
            metrics = evaluate_class_specific(
                y_val,
                val_tfidf_preds,
                val_tfidf_conf,
                val_roberta_preds,
                tau_threat,
                tau_safe,
            )
            row = {
                "tau_threat": float(tau_threat),
                "tau_safe": float(tau_safe),
                "validation_macro_f1": metrics["macro_f1"],
                "validation_fnr": metrics["fnr"],
                "validation_tier1_handled_percent": metrics["tier1_handled_percent"],
                "validation_tier2_escalated_percent": metrics["tier2_escalated_percent"],
            }
            row["feasible_initial"] = (
                row["validation_macro_f1"] >= initial_f1_floor
                and row["validation_fnr"] <= fnr_ceiling
            )
            row["feasible_relaxed"] = (
                row["validation_macro_f1"] >= relaxed_f1_floor
                and row["validation_fnr"] <= fnr_ceiling
            )
            rows.append(row)

    return rows


def best_f1_class_specific_row(rows):
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


def select_safety_constrained_row(rows, best_f1_row):
    feasible_initial = [row for row in rows if row["feasible_initial"]]

    if feasible_initial:
        return cheapest_row(feasible_initial), "feasible_initial", DELTA_F1_INITIAL

    feasible_relaxed = [row for row in rows if row["feasible_relaxed"]]

    if feasible_relaxed:
        return cheapest_row(feasible_relaxed), "feasible_relaxed", DELTA_F1_RELAXED

    return best_f1_row, "fallback_best_f1", DELTA_F1_RELAXED


def validation_row_to_metrics(row):
    return {
        "macro_f1": row["validation_macro_f1"],
        "fnr": row["validation_fnr"],
        "tier1_handled_percent": row["validation_tier1_handled_percent"],
        "tier2_escalated_percent": row["validation_tier2_escalated_percent"],
    }


def save_grid(rows):
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


def save_test_predictions(
    test_texts,
    y_test,
    test_tfidf_preds,
    test_tfidf_conf,
    test_roberta_preds,
    symmetric_preds,
    best_f1_preds,
    constrained_preds,
):
    fieldnames = [
        "row_id",
        "tweet",
        "true_label",
        "tfidf_pred",
        "tfidf_confidence",
        "roberta_pred",
        "symmetric_cascade_pred",
        "best_f1_class_specific_pred",
        "safety_constrained_class_specific_pred",
    ]

    with open(PREDICTIONS_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for idx, text in enumerate(test_texts):
            writer.writerow({
                "row_id": idx,
                "tweet": text,
                "true_label": int(y_test[idx]),
                "tfidf_pred": int(test_tfidf_preds[idx]),
                "tfidf_confidence": float(test_tfidf_conf[idx]),
                "roberta_pred": int(test_roberta_preds[idx]),
                "symmetric_cascade_pred": int(symmetric_preds[idx]),
                "best_f1_class_specific_pred": int(best_f1_preds[idx]),
                "safety_constrained_class_specific_pred": int(constrained_preds[idx]),
            })


def build_table_row(system, tau, tau_threat, tau_safe, validation, test):
    return {
        "system": system,
        "tau": tau,
        "tau_threat": tau_threat,
        "tau_safe": tau_safe,
        "validation": validation,
        "test": test,
    }


def format_tau(value):
    if value is None:
        return "-"
    return f"{value:.3f}"


def print_final_table(rows):
    print("\nFinal OLID clean cascade table")
    print(
        "System                                  "
        "tau     tau_threat  tau_safe  Val_F1  Val_FNR  Val_Tier1%  "
        "Test_F1  Test_FNR  Test_Tier1%  Test_Tier2%"
    )

    for row in rows:
        val_metrics = row["validation"]
        test_metrics = row["test"]
        print(
            f"{row['system']:<39} "
            f"{format_tau(row['tau']):>7}  "
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


def main():
    require_roberta_checkpoint()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    ds = load_dataset(DATASET_NAME)
    if "train" not in ds or "test" not in ds:
        raise KeyError(f"Expected train/test splits, found: {list(ds.keys())}")

    official_train_texts, official_train_labels = prepare_olid_split(ds["train"])
    test_texts, y_test = prepare_olid_split(ds["test"])

    subtrain_texts, val_texts, y_subtrain, y_val = train_test_split(
        official_train_texts,
        official_train_labels,
        test_size=VALIDATION_SIZE,
        random_state=RANDOM_SEED,
        stratify=official_train_labels,
    )

    print("Official OLID train:", len(official_train_texts))
    print("Subtrain:", len(subtrain_texts))
    print("Validation:", len(val_texts))
    print("Official test:", len(test_texts))

    word_vec, char_vec, X_subtrain = fit_tfidf_vectorizers(subtrain_texts)
    X_val = transform_tfidf(word_vec, char_vec, val_texts)
    tfidf_clf = train_tfidf_classifier(X_subtrain, y_subtrain)

    val_tfidf_preds, val_tfidf_conf = get_tfidf_outputs(tfidf_clf, X_val)

    tokenizer, roberta_model, device = load_roberta_model()
    print("\nRunning RoBERTa on validation split...")
    val_roberta_preds = get_roberta_predictions(
        val_texts,
        tokenizer,
        roberta_model,
        device,
    )

    tfidf_val_metrics = standalone_tfidf_metrics(y_val, val_tfidf_preds)

    roberta_val_metrics = standalone_roberta_metrics(y_val, val_roberta_preds)

    selected_symmetric = select_symmetric_threshold(
        y_val,
        val_tfidf_preds,
        val_tfidf_conf,
        val_roberta_preds,
    )
    symmetric_tau = selected_symmetric["tau"]
    symmetric_val_metrics = selected_symmetric["metrics"]

    grid_rows = build_class_specific_grid(
        y_val,
        val_tfidf_preds,
        val_tfidf_conf,
        val_roberta_preds,
        roberta_val_metrics,
    )
    best_f1_row = best_f1_class_specific_row(grid_rows)
    constrained_row, selection_status, delta_f1_used = select_safety_constrained_row(
        grid_rows,
        best_f1_row,
    )

    best_f1_val_metrics = validation_row_to_metrics(best_f1_row)

    constrained_val_metrics = validation_row_to_metrics(constrained_row)

    print("\nValidation-only selection complete.")
    print(f"Selected symmetric tau: {symmetric_tau:.3f}")
    print(
        "Selected best-F1 class-specific thresholds: "
        f"tau_threat={best_f1_row['tau_threat']:.3f}, "
        f"tau_safe={best_f1_row['tau_safe']:.3f}"
    )
    print(
        "Selected safety-constrained thresholds: "
        f"tau_threat={constrained_row['tau_threat']:.3f}, "
        f"tau_safe={constrained_row['tau_safe']:.3f}"
    )

    print("\nRunning final evaluation on official test split...")
    X_test = transform_tfidf(word_vec, char_vec, test_texts)
    test_tfidf_preds, test_tfidf_conf = get_tfidf_outputs(tfidf_clf, X_test)
    test_roberta_preds = get_roberta_predictions(
        test_texts,
        tokenizer,
        roberta_model,
        device,
    )

    tfidf_test_metrics = standalone_tfidf_metrics(y_test, test_tfidf_preds)
    roberta_test_metrics = standalone_roberta_metrics(y_test, test_roberta_preds)

    symmetric_test_metrics = evaluate_symmetric(
        y_test,
        test_tfidf_preds,
        test_tfidf_conf,
        test_roberta_preds,
        symmetric_tau,
    )

    best_f1_test_metrics = evaluate_class_specific(
        y_test,
        test_tfidf_preds,
        test_tfidf_conf,
        test_roberta_preds,
        best_f1_row["tau_threat"],
        best_f1_row["tau_safe"],
    )

    constrained_test_metrics = evaluate_class_specific(
        y_test,
        test_tfidf_preds,
        test_tfidf_conf,
        test_roberta_preds,
        constrained_row["tau_threat"],
        constrained_row["tau_safe"],
    )

    symmetric_test_preds, _, _ = run_symmetric_cascade(
        test_tfidf_preds,
        test_tfidf_conf,
        test_roberta_preds,
        symmetric_tau,
    )
    best_f1_test_preds, _, _ = run_class_specific_cascade(
        test_tfidf_preds,
        test_tfidf_conf,
        test_roberta_preds,
        best_f1_row["tau_threat"],
        best_f1_row["tau_safe"],
    )
    constrained_test_preds, _, _ = run_class_specific_cascade(
        test_tfidf_preds,
        test_tfidf_conf,
        test_roberta_preds,
        constrained_row["tau_threat"],
        constrained_row["tau_safe"],
    )

    save_grid(grid_rows)
    save_test_predictions(
        test_texts,
        y_test,
        test_tfidf_preds,
        test_tfidf_conf,
        test_roberta_preds,
        symmetric_test_preds,
        best_f1_test_preds,
        constrained_test_preds,
    )

    table_rows = [
        build_table_row("TF-IDF standalone", None, None, None, tfidf_val_metrics, tfidf_test_metrics),
        build_table_row("RoBERTa standalone", None, None, None, roberta_val_metrics, roberta_test_metrics),
        build_table_row("symmetric cascade", symmetric_tau, None, None, symmetric_val_metrics, symmetric_test_metrics),
        build_table_row(
            "best-F1 class-specific cascade",
            None,
            best_f1_row["tau_threat"],
            best_f1_row["tau_safe"],
            best_f1_val_metrics,
            best_f1_test_metrics,
        ),
        build_table_row(
            "safety-constrained class-specific cascade",
            None,
            constrained_row["tau_threat"],
            constrained_row["tau_safe"],
            constrained_val_metrics,
            constrained_test_metrics,
        ),
    ]

    result = {
        "dataset_name": DATASET_NAME,
        "split_sizes": {
            "official_train": len(official_train_texts),
            "subtrain": len(subtrain_texts),
            "validation": len(val_texts),
            "official_test": len(test_texts),
        },
        "random_state": RANDOM_SEED,
        "tfidf_settings": TFIDF_SETTINGS,
        "logistic_regression_settings": LOGREG_SETTINGS,
        "roberta_model_path": str(MODEL_DIR),
        "tfidf": {
            "validation": tfidf_val_metrics,
            "test": tfidf_test_metrics,
        },
        "roberta": {
            "validation": roberta_val_metrics,
            "test": roberta_test_metrics,
        },
        "symmetric_cascade": {
            "selected_tau": symmetric_tau,
            "validation": symmetric_val_metrics,
            "test": symmetric_test_metrics,
        },
        "best_f1_class_specific_cascade": {
            "tau_threat": best_f1_row["tau_threat"],
            "tau_safe": best_f1_row["tau_safe"],
            "validation": best_f1_val_metrics,
            "test": best_f1_test_metrics,
        },
        "safety_constrained_class_specific_cascade": {
            "tau_threat": constrained_row["tau_threat"],
            "tau_safe": constrained_row["tau_safe"],
            "validation": constrained_val_metrics,
            "test": constrained_test_metrics,
        },
        "selection_status": selection_status,
        "delta_f1_initial": DELTA_F1_INITIAL,
        "delta_f1_relaxed": DELTA_F1_RELAXED,
        "delta_f1_used": delta_f1_used,
        "epsilon_fnr": EPSILON_FNR,
        "outputs": {
            "grid_csv": str(GRID_PATH),
            "result_json": str(RESULT_PATH),
            "test_predictions_csv": str(PREDICTIONS_PATH),
        },
    }

    with open(RESULT_PATH, "w") as f:
        json.dump(result, f, indent=2)

    print_final_table(table_rows)

    print("\nSelection status:", selection_status)
    print(f"Saved validation grid to: {GRID_PATH}")
    print(f"Saved final result JSON to: {RESULT_PATH}")
    print(f"Saved official test predictions to: {PREDICTIONS_PATH}")


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, RuntimeError) as exc:
        print(exc)
        sys.exit(1)
