"""Experiment 4 — Symmetric and class-specific cascade threshold search on HateXplain."""
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


def get_tfidf_predictions(clf, X):
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
    threat_mask = y_true == 1
    missed_threats = np.sum((y_true == 1) & (y_pred == 0))
    total_threats = np.sum(threat_mask)

    if total_threats == 0:
        return 0.0

    return missed_threats / total_threats


def run_symmetric_cascade(tfidf_preds, tfidf_conf, roberta_preds, threshold):
    cascade_preds = tfidf_preds.copy()

    escalate_mask = tfidf_conf < threshold
    cascade_preds[escalate_mask] = roberta_preds[escalate_mask]

    tier1_handled = np.sum(~escalate_mask)
    tier2_escalated = np.sum(escalate_mask)

    return cascade_preds, tier1_handled, tier2_escalated


def run_asymmetric_cascade(
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

    tier1_handled = np.sum(keep_tfidf)
    tier2_escalated = np.sum(escalate_mask)

    return cascade_preds, tier1_handled, tier2_escalated


def find_best_symmetric(y_val, val_tfidf_preds, val_tfidf_conf, val_roberta_preds):
    best = {
        "threshold": None,
        "f1": -1.0,
        "fnr": None,
        "tier1_percent": None,
    }

    for threshold in np.round(np.arange(0.50, 0.951, 0.01), 3):
        preds, tier1_handled, _ = run_symmetric_cascade(
            val_tfidf_preds,
            val_tfidf_conf,
            val_roberta_preds,
            threshold,
        )

        f1 = f1_score(y_val, preds, average="macro")
        fnr = false_negative_rate(y_val, preds)
        tier1_percent = 100 * tier1_handled / len(y_val)

        if f1 > best["f1"]:
            best = {
                "threshold": threshold,
                "f1": f1,
                "fnr": fnr,
                "tier1_percent": tier1_percent,
            }

    return best


def find_best_asymmetric(y_val, val_tfidf_preds, val_tfidf_conf, val_roberta_preds):
    best = {
        "threat_threshold": None,
        "safe_threshold": None,
        "f1": -1.0,
        "fnr": None,
        "tier1_percent": None,
    }

    thresholds = np.round(np.arange(0.50, 0.951, 0.01), 3)

    for threat_threshold in thresholds:
        for safe_threshold in thresholds:
            preds, tier1_handled, _ = run_asymmetric_cascade(
                val_tfidf_preds,
                val_tfidf_conf,
                val_roberta_preds,
                threat_threshold,
                safe_threshold,
            )

            f1 = f1_score(y_val, preds, average="macro")
            fnr = false_negative_rate(y_val, preds)
            tier1_percent = 100 * tier1_handled / len(y_val)

            # Main objective: best validation Macro F1.
            # Tie-breaker: lower false negative rate.
            if (
                f1 > best["f1"]
                or (round(f1, 4) == round(best["f1"], 4) and fnr < best["fnr"])
            ):
                best = {
                    "threat_threshold": threat_threshold,
                    "safe_threshold": safe_threshold,
                    "f1": f1,
                    "fnr": fnr,
                    "tier1_percent": tier1_percent,
                }

    return best


def main():
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

    val_tfidf_preds, val_tfidf_conf = get_tfidf_predictions(tfidf_clf, X_val)
    test_tfidf_preds, test_tfidf_conf = get_tfidf_predictions(tfidf_clf, X_test)

    print("\nRunning RoBERTa on validation split...")
    val_roberta_preds = get_roberta_predictions(val_texts)

    print("Running RoBERTa on test split...")
    test_roberta_preds = get_roberta_predictions(test_texts)

    print("\nBaselines:")
    print(f"Validation TF-IDF F1:  {f1_score(y_val, val_tfidf_preds, average='macro'):.4f}")
    print(f"Validation RoBERTa F1: {f1_score(y_val, val_roberta_preds, average='macro'):.4f}")
    print(f"Test TF-IDF F1:        {f1_score(y_test, test_tfidf_preds, average='macro'):.4f}")
    print(f"Test RoBERTa F1:       {f1_score(y_test, test_roberta_preds, average='macro'):.4f}")

    best_sym = find_best_symmetric(
        y_val,
        val_tfidf_preds,
        val_tfidf_conf,
        val_roberta_preds,
    )

    best_asym = find_best_asymmetric(
        y_val,
        val_tfidf_preds,
        val_tfidf_conf,
        val_roberta_preds,
    )

    sym_test_preds, sym_tier1, sym_tier2 = run_symmetric_cascade(
        test_tfidf_preds,
        test_tfidf_conf,
        test_roberta_preds,
        best_sym["threshold"],
    )

    asym_test_preds, asym_tier1, asym_tier2 = run_asymmetric_cascade(
        test_tfidf_preds,
        test_tfidf_conf,
        test_roberta_preds,
        best_asym["threat_threshold"],
        best_asym["safe_threshold"],
    )

    sym_test_f1 = f1_score(y_test, sym_test_preds, average="macro")
    asym_test_f1 = f1_score(y_test, asym_test_preds, average="macro")

    print("\nBest symmetric threshold selected on validation:")
    print(f"threshold:       {best_sym['threshold']:.3f}")
    print(f"val F1:          {best_sym['f1']:.4f}")
    print(f"val FNR:         {best_sym['fnr']:.4f}")
    print(f"val Tier 1 %:    {best_sym['tier1_percent']:.2f}%")

    print("\nFinal symmetric test result:")
    print(f"test F1:         {sym_test_f1:.4f}")
    print(f"test FNR:        {false_negative_rate(y_test, sym_test_preds):.4f}")
    print(f"test Tier 1 %:   {100 * sym_tier1 / len(y_test):.2f}%")
    print(f"test Tier 2 %:   {100 * sym_tier2 / len(y_test):.2f}%")

    print("\nBest asymmetric thresholds selected on validation:")
    print(f"threat_threshold: {best_asym['threat_threshold']:.3f}")
    print(f"safe_threshold:   {best_asym['safe_threshold']:.3f}")
    print(f"val F1:           {best_asym['f1']:.4f}")
    print(f"val FNR:          {best_asym['fnr']:.4f}")
    print(f"val Tier 1 %:     {best_asym['tier1_percent']:.2f}%")

    print("\nFinal asymmetric test result:")
    print(f"test F1:          {asym_test_f1:.4f}")
    print(f"test FNR:         {false_negative_rate(y_test, asym_test_preds):.4f}")
    print(f"test Tier 1 %:    {100 * asym_tier1 / len(y_test):.2f}%")
    print(f"test Tier 2 %:    {100 * asym_tier2 / len(y_test):.2f}%")

    print("\nAsymmetric classification report:")
    print(classification_report(y_test, asym_test_preds, target_names=["SAFE", "THREAT"]))


if __name__ == "__main__":
    main()