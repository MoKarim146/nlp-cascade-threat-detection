"""Experiment 2 — TF-IDF + Logistic Regression baseline on OLID Subtask A (binary threat detection)."""
from datasets import load_dataset
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score


DATASET_NAME = "christophsonntag/OLID"


def map_olid_label(label):
    """
    OLID Subtask A:
    OFF -> THREAT -> 1
    NOT -> SAFE   -> 0
    """
    label = str(label).strip().upper()

    if label == "OFF":
        return 1
    if label == "NOT":
        return 0

    raise ValueError(f"Unexpected OLID label: {label!r}")


def find_text_column(df):
    """
    Try common OLID text column names.
    """
    candidates = ["tweet", "text", "sentence"]

    for col in candidates:
        if col in df.column_names:
            return col

    raise KeyError(
        f"Could not find text column. Available columns: {df.column_names}"
    )


def find_label_column(df):
    """
    Try common OLID label column names for Subtask A.
    """
    candidates = ["subtask_a", "label", "labels"]

    for col in candidates:
        if col in df.column_names:
            return col

    raise KeyError(
        f"Could not find label column. Available columns: {df.column_names}"
    )


def prepare_split(split_dataset, text_col, label_col):
    texts = []
    labels = []

    for row in split_dataset:
        text = str(row[text_col]).strip()
        label = map_olid_label(row[label_col])

        if text:
            texts.append(text)
            labels.append(label)

    return texts, labels


def main():
    # 1) Load dataset
    ds = load_dataset(DATASET_NAME)

    print("Available splits:", ds.keys())

    # Print schema for every split so you see what is actually there
    for split_name in ds.keys():
        print(f"\n--- {split_name.upper()} ---")
        print("Columns:", ds[split_name].column_names)
        print("Example row:", ds[split_name][0])

    # 2) Detect columns from train split
    train_split_name = "train"

    if train_split_name not in ds:
        raise KeyError(f"Expected a train split, found: {list(ds.keys())}")

    text_col = find_text_column(ds[train_split_name])
    label_col = find_label_column(ds[train_split_name])

    print("\nUsing text column:", text_col)
    print("Using label column:", label_col)

    # 3) Pick train/test splits
    # OLID thesis setup uses official train/test only
    if "test" not in ds:
        raise KeyError(f"Expected a test split, found: {list(ds.keys())}")

    train_texts, y_train = prepare_split(ds["train"], text_col, label_col)
    test_texts, y_test = prepare_split(ds["test"], text_col, label_col)

    print("\nTrain size:", len(train_texts))
    print("Test size:", len(test_texts))

    # 4) Build enhanced TF-IDF representation
    # word: 20k features
    word_vec = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        max_features=20_000,
        sublinear_tf=True,
        min_df=2,
        lowercase=True,
    )

    # char: 15k features
    char_vec = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        max_features=15_000,
        sublinear_tf=True,
        min_df=2,
        lowercase=True,
    )

    # IMPORTANT: fit on train only
    X_train_word = word_vec.fit_transform(train_texts)
    X_train_char = char_vec.fit_transform(train_texts)
    X_train = hstack([X_train_word, X_train_char])

    X_test_word = word_vec.transform(test_texts)
    X_test_char = char_vec.transform(test_texts)
    X_test = hstack([X_test_word, X_test_char])

    print("\nFeature matrix shapes:")
    print("X_train:", X_train.shape)
    print("X_test: ", X_test.shape)

    print("\nActual vocab sizes after fitting:")
    print("Word vocab size:", len(word_vec.get_feature_names_out()))
    print("Char vocab size:", len(char_vec.get_feature_names_out()))
    print("Total features:", X_train.shape[1])

    # 5) Train Logistic Regression
    clf = LogisticRegression(
        C=1.0,
        class_weight="balanced",
        max_iter=1000,
        solver="lbfgs",
        random_state=42,
    )
    clf.fit(X_train, y_train)

    # 6) Evaluate
    y_pred = clf.predict(X_test)
    test_f1 = f1_score(y_test, y_pred, average="macro")

    print(f"\nOLID Test Macro F1: {test_f1:.4f}")
    print("\nClassification report:")
    print(classification_report(y_test, y_pred, target_names=["SAFE", "THREAT"]))

    # 7) Confidence scores for later cascade work
    probs = clf.predict_proba(X_test)
    conf = probs.max(axis=1)

    print("\nFirst 10 confidence values:")
    print(conf[:10])

    # 8) Optional: inspect strongest features
    all_features = list(word_vec.get_feature_names_out()) + list(char_vec.get_feature_names_out())
    weights = clf.coef_[0]

    top_threat_idx = weights.argsort()[-15:][::-1]
    top_safe_idx = weights.argsort()[:15]

    print("\nTop THREAT features:")
    for i in top_threat_idx[:10]:
        print(f"{all_features[i]:30s} {weights[i]:+.4f}")

    print("\nTop SAFE features:")
    for i in top_safe_idx[:10]:
        print(f"{all_features[i]:30s} {weights[i]:+.4f}")


if __name__ == "__main__":
    main()