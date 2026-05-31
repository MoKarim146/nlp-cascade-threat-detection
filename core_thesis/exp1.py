"""Experiment 1 — TF-IDF + Logistic Regression baseline on HateXplain (binary threat detection)."""
from collections import Counter
from typing import Any, Dict, List, Tuple
from datasets import load_dataset
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score


def majority_vote(labels: List[Any]) -> Any:
    """
    Return the most common label in a list.
    HateXplain has multiple annotator labels per post.
    """
    return Counter(labels).most_common(1)[0][0]

# - done
def to_binary_label(raw_label: Any) -> int:
    """
    Map HateXplain labels to binary:
    - normal -> 0 (SAFE)
    - hatespeech / offensive -> 1 (THREAT)

    This function is intentionally defensive because dataset mirrors
    may store labels slightly differently.
    """
    if isinstance(raw_label, str):
        label = raw_label.strip().lower()
        if label == "normal":
            return 0
        if label in {"hatespeech", "hate", "offensive"}:
            return 1

    # Integer case for HateXplain ClassLabel
    if isinstance(raw_label, int):
        if raw_label == 1:      # normal
            return 0            # SAFE
        if raw_label in {0, 2}: # hatespeech or offensive
            return 1            # THREAT

    raise ValueError(f"Unrecognized label format: {raw_label!r}")

# ----

def extract_text_and_label(row: Dict[str, Any]) -> Tuple[str, int]:
    """
    Convert one HateXplain row into:
    - text string
    - binary label
    """
    # Standard HF HateXplain usually stores tokenized text here
    if "post_tokens" in row:
        text = " ".join(row["post_tokens"]).strip()
    elif "text" in row:
        text = str(row["text"]).strip()
    else:
        raise KeyError("Could not find text field. Expected 'post_tokens' or 'text'.")

    # Standard HF HateXplain usually stores annotator labels here
    if "annotators" in row and isinstance(row["annotators"], dict) and "label" in row["annotators"]:
        raw_label = majority_vote(row["annotators"]["label"])
    elif "label" in row:
        raw_label = row["label"]
    else:
        raise KeyError("Could not find label field. Expected annotators['label'] or 'label'.")

    y = to_binary_label(raw_label)
    return text, y


def prepare_split(split_dataset) -> Tuple[List[str], List[int]]:
    texts: List[str] = []
    labels: List[int] = []

    for row in split_dataset:
        text, label = extract_text_and_label(row)
        if text:
            texts.append(text)
            labels.append(label)

    return texts, labels


def main() -> None:
    # 1) Load dataset with official train/validation/test split
    ds = load_dataset("hatexplain")

    print("Available splits:", ds.keys())
    print("Train columns:", ds["train"].column_names)
    print("Example row:", ds["train"][0])

    # 2) Convert to plain texts + binary labels
    train_texts, y_train = prepare_split(ds["train"])
    val_texts, y_val = prepare_split(ds["validation"])
    test_texts, y_test = prepare_split(ds["test"])

    print(f"\nTrain size: {len(train_texts)}")
    print(f"Validation size: {len(val_texts)}")
    print(f"Test size: {len(test_texts)}")

    # 3) Build the exact enhanced TF-IDF representation from the thesis
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

    # IMPORTANT: fit on TRAIN ONLY
    X_train_word = word_vec.fit_transform(train_texts)
    X_train_char = char_vec.fit_transform(train_texts)
    X_train = hstack([X_train_word, X_train_char])

    # Validation/test: transform only
    X_val_word = word_vec.transform(val_texts)
    X_val_char = char_vec.transform(val_texts)
    X_val = hstack([X_val_word, X_val_char])

    X_test_word = word_vec.transform(test_texts)
    X_test_char = char_vec.transform(test_texts)
    X_test = hstack([X_test_word, X_test_char])

    print("\nFeature matrix shapes:")
    print("X_train:", X_train.shape)
    print("X_val:  ", X_val.shape)
    print("X_test: ", X_test.shape)

    # 4) Train Logistic Regression
    clf = LogisticRegression(
        C=1.0,
        class_weight="balanced",
        max_iter=1000,
        solver="lbfgs",
        random_state=42,
    )
    clf.fit(X_train, y_train)

    # 5) Evaluate
    val_pred = clf.predict(X_val)
    test_pred = clf.predict(X_test)

    val_f1 = f1_score(y_val, val_pred, average="macro")
    test_f1 = f1_score(y_test, test_pred, average="macro")

    print(f"\nValidation Macro F1: {val_f1:.4f}")
    print(f"Test Macro F1:       {test_f1:.4f}")

    print("\nTest classification report:")
    print(classification_report(y_test, test_pred, target_names=["SAFE", "THREAT"]))

    # 6) Confidence scores (you will need these later for the cascade)
    test_probs = clf.predict_proba(X_test)
    test_conf = test_probs.max(axis=1)

    print("\nExample confidence values:")
    print(test_conf[:10])


if __name__ == "__main__":
    main()