"""Experiment 8 — Confidence-bin analysis and cost-penalty threshold sweep on HateXplain."""
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer


MAX_LENGTH = 128
BATCH_SIZE = 16

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ROBERTA_MODEL_DIR = PROJECT_ROOT / "core_thesis" / "models" / "roberta_hatexplain"


def majority_vote(labels):
    return Counter(labels).most_common(1)[0][0]


def to_binary_label(label_id):
    # HateXplain: 0=hatespeech, 1=normal, 2=offensive
    # Thesis: SAFE=0, THREAT=1
    if label_id == 1:
        return 0
    if label_id in {0, 2}:
        return 1
    raise ValueError(f"Unexpected label id: {label_id!r}")


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
        random_state=42,
    )
    clf.fit(X_train, y_train)
    return clf


def get_tfidf_outputs(clf, X):
    probs = clf.predict_proba(X)
    preds = np.argmax(probs, axis=1)
    conf = np.max(probs, axis=1)
    return preds, conf, probs


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def get_roberta_outputs(texts):
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
    all_conf = []

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

        conf, preds = torch.max(probs, dim=1)

        all_preds.extend(preds.cpu().numpy().tolist())
        all_conf.extend(conf.cpu().numpy().tolist())

    return np.array(all_preds), np.array(all_conf)


def false_negative_rate(y_true, y_pred):
    threat_mask = y_true == 1
    total_threats = np.sum(threat_mask)

    if total_threats == 0:
        return 0.0

    missed_threats = np.sum((y_true == 1) & (y_pred == 0))
    return missed_threats / total_threats


def print_confidence_bins(y_true, tfidf_preds, tfidf_conf, roberta_preds):
    print("\nValidation confidence-bin analysis")
    print("Bin        Count  TF-IDF_acc  RoBERTa_acc  Better")

    bins = np.arange(0.50, 1.01, 0.05)

    for low, high in zip(bins[:-1], bins[1:]):
        if high >= 1.0:
            mask = (tfidf_conf >= low) & (tfidf_conf <= high)
        else:
            mask = (tfidf_conf >= low) & (tfidf_conf < high)

        count = np.sum(mask)

        if count == 0:
            continue

        tfidf_acc = np.mean(tfidf_preds[mask] == y_true[mask])
        roberta_acc = np.mean(roberta_preds[mask] == y_true[mask])

        if tfidf_acc > roberta_acc:
            better = "TF-IDF"
        elif roberta_acc > tfidf_acc:
            better = "RoBERTa"
        else:
            better = "Tie"

        print(
            f"{low:.2f}-{high:.2f}  "
            f"{count:5d}  "
            f"{tfidf_acc:9.3f}  "
            f"{roberta_acc:11.3f}  "
            f"{better}"
        )


def print_class_specific_bins(y_true, tfidf_preds, tfidf_conf, roberta_preds):
    print("\nClass-specific confidence-bin analysis")
    print("Predicted  Bin        Count  TF-IDF_acc  RoBERTa_acc  Better")

    bins = np.arange(0.50, 1.01, 0.05)

    for predicted_class, class_name in [(0, "SAFE"), (1, "THREAT")]:
        class_mask = tfidf_preds == predicted_class

        for low, high in zip(bins[:-1], bins[1:]):
            if high >= 1.0:
                bin_mask = (tfidf_conf >= low) & (tfidf_conf <= high)
            else:
                bin_mask = (tfidf_conf >= low) & (tfidf_conf < high)

            mask = class_mask & bin_mask
            count = np.sum(mask)

            if count == 0:
                continue

            tfidf_acc = np.mean(tfidf_preds[mask] == y_true[mask])
            roberta_acc = np.mean(roberta_preds[mask] == y_true[mask])

            if tfidf_acc > roberta_acc:
                better = "TF-IDF"
            elif roberta_acc > tfidf_acc:
                better = "RoBERTa"
            else:
                better = "Tie"

            print(
                f"{class_name:9s}  "
                f"{low:.2f}-{high:.2f}  "
                f"{count:5d}  "
                f"{tfidf_acc:9.3f}  "
                f"{roberta_acc:11.3f}  "
                f"{better}"
            )


def run_class_specific_cascade(
    tfidf_preds,
    tfidf_conf,
    roberta_preds,
    threat_threshold,
    safe_threshold,
):
    cascade_preds = tfidf_preds.copy()

    tfidf_says_threat = tfidf_preds == 1
    tfidf_says_safe = tfidf_preds == 0

    keep_threat = tfidf_says_threat & (tfidf_conf >= threat_threshold)
    keep_safe = tfidf_says_safe & (tfidf_conf >= safe_threshold)

    keep_tfidf = keep_threat | keep_safe
    escalate_mask = ~keep_tfidf

    cascade_preds[escalate_mask] = roberta_preds[escalate_mask]

    tier1_fraction = np.mean(keep_tfidf)
    gpu_fraction = np.mean(escalate_mask)

    return cascade_preds, tier1_fraction, gpu_fraction


def cost_aware_threshold_search(
    y_val,
    val_tfidf_preds,
    val_tfidf_conf,
    val_roberta_preds,
    lambda_gpu,
):
    best = {
        "threat_threshold": None,
        "safe_threshold": None,
        "score": -999.0,
        "f1": None,
        "fnr": None,
        "tier1_fraction": None,
        "gpu_fraction": None,
    }

    thresholds = np.round(np.arange(0.50, 0.951, 0.01), 3)

    for threat_threshold in thresholds:
        for safe_threshold in thresholds:
            preds, tier1_fraction, gpu_fraction = run_class_specific_cascade(
                val_tfidf_preds,
                val_tfidf_conf,
                val_roberta_preds,
                threat_threshold,
                safe_threshold,
            )

            f1 = f1_score(y_val, preds, average="macro")
            fnr = false_negative_rate(y_val, preds)

            # Cost-aware objective:
            # higher F1 is better, lower RoBERTa usage is better.
            score = f1 - lambda_gpu * gpu_fraction

            if score > best["score"]:
                best = {
                    "threat_threshold": threat_threshold,
                    "safe_threshold": safe_threshold,
                    "score": score,
                    "f1": f1,
                    "fnr": fnr,
                    "tier1_fraction": tier1_fraction,
                    "gpu_fraction": gpu_fraction,
                }

    return best


def evaluate_locked_thresholds(
    y_test,
    test_tfidf_preds,
    test_tfidf_conf,
    test_roberta_preds,
    threat_threshold,
    safe_threshold,
):
    preds, tier1_fraction, gpu_fraction = run_class_specific_cascade(
        test_tfidf_preds,
        test_tfidf_conf,
        test_roberta_preds,
        threat_threshold,
        safe_threshold,
    )

    return {
        "preds": preds,
        "f1": f1_score(y_test, preds, average="macro"),
        "fnr": false_negative_rate(y_test, preds),
        "tier1_fraction": tier1_fraction,
        "gpu_fraction": gpu_fraction,
    }


def main():
    ds = load_dataset("hatexplain")

    train_texts, y_train = prepare_split(ds["train"])
    val_texts, y_val = prepare_split(ds["validation"])
    test_texts, y_test = prepare_split(ds["test"])

    X_train, X_val, X_test = build_tfidf(train_texts, val_texts, test_texts)
    tfidf_clf = train_tfidf_classifier(X_train, y_train)

    val_tfidf_preds, val_tfidf_conf, _ = get_tfidf_outputs(tfidf_clf, X_val)
    test_tfidf_preds, test_tfidf_conf, _ = get_tfidf_outputs(tfidf_clf, X_test)

    print("Running RoBERTa on validation split...")
    val_roberta_preds, _ = get_roberta_outputs(val_texts)

    print("Running RoBERTa on test split...")
    test_roberta_preds, _ = get_roberta_outputs(test_texts)

    print("\nBaselines")
    print(f"Validation TF-IDF F1:  {f1_score(y_val, val_tfidf_preds, average='macro'):.4f}")
    print(f"Validation RoBERTa F1: {f1_score(y_val, val_roberta_preds, average='macro'):.4f}")
    print(f"Test TF-IDF F1:        {f1_score(y_test, test_tfidf_preds, average='macro'):.4f}")
    print(f"Test RoBERTa F1:       {f1_score(y_test, test_roberta_preds, average='macro'):.4f}")

    print_confidence_bins(
        y_val,
        val_tfidf_preds,
        val_tfidf_conf,
        val_roberta_preds,
    )

    print_class_specific_bins(
        y_val,
        val_tfidf_preds,
        val_tfidf_conf,
        val_roberta_preds,
    )

    print("\nCost-aware class-specific threshold search")
    print("lambda  tau_threat  tau_safe  val_F1  val_FNR  val_Tier1%  test_F1  test_FNR  test_Tier1%")

    lambda_values = [0.000, 0.002, 0.005, 0.010, 0.020]

    best_for_report = None

    for lambda_gpu in lambda_values:
        best = cost_aware_threshold_search(
            y_val,
            val_tfidf_preds,
            val_tfidf_conf,
            val_roberta_preds,
            lambda_gpu=lambda_gpu,
        )

        test_result = evaluate_locked_thresholds(
            y_test,
            test_tfidf_preds,
            test_tfidf_conf,
            test_roberta_preds,
            best["threat_threshold"],
            best["safe_threshold"],
        )

        print(
            f"{lambda_gpu:6.3f}  "
            f"{best['threat_threshold']:10.3f}  "
            f"{best['safe_threshold']:8.3f}  "
            f"{best['f1']:6.4f}  "
            f"{best['fnr']:7.4f}  "
            f"{100 * best['tier1_fraction']:10.2f}  "
            f"{test_result['f1']:7.4f}  "
            f"{test_result['fnr']:8.4f}  "
            f"{100 * test_result['tier1_fraction']:11.2f}"
        )

        if lambda_gpu == 0.005:
            best_for_report = (best, test_result)

    if best_for_report is not None:
        best, test_result = best_for_report

        print("\nDetailed classification report for lambda=0.005 selected on validation:")
        print(f"tau_threat: {best['threat_threshold']:.3f}")
        print(f"tau_safe:   {best['safe_threshold']:.3f}")
        print(f"test F1:    {test_result['f1']:.4f}")
        print(f"test FNR:   {test_result['fnr']:.4f}")
        print(f"Tier 1 %:   {100 * test_result['tier1_fraction']:.2f}%")
        print(f"Tier 2 %:   {100 * test_result['gpu_fraction']:.2f}%")

        print("\nClassification report:")
        print(classification_report(y_test, test_result["preds"], target_names=["SAFE", "THREAT"]))


if __name__ == "__main__":
    main()