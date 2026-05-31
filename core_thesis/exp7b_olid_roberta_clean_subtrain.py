"""Experiment 7b — Fine-tune roberta-base on OLID (80/20 subtrain split); saves best checkpoint."""
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)


SAFE = 0
THREAT = 1

DATASET_NAME = "christophsonntag/OLID"
MODEL_NAME = "roberta-base"

RANDOM_SEED = 42
VALIDATION_SIZE = 0.20
MAX_LENGTH = 128
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
NUM_EPOCHS = 3
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.10
GRADIENT_CLIP_NORM = 1.0

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CORE_DIR = PROJECT_ROOT / "core_thesis"
MODEL_DIR = CORE_DIR / "models" / "roberta_olid_subtrain_clean"
OUTPUT_DIR = CORE_DIR / "outputs"

RESULTS_PATH = OUTPUT_DIR / "exp7b_olid_roberta_clean_subtrain_results.json"
VAL_PREDICTIONS_PATH = OUTPUT_DIR / "exp7b_olid_roberta_validation_predictions.csv"
TEST_PREDICTIONS_PATH = OUTPUT_DIR / "exp7b_olid_roberta_test_predictions.csv"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


class TextClassificationDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoded = self.tokenizer(
            self.texts[idx],
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def prediction_counts(preds):
    preds = np.array(preds)
    return {
        "SAFE": int(np.sum(preds == SAFE)),
        "THREAT": int(np.sum(preds == THREAT)),
    }


def false_negative_rate(labels, preds):
    labels = np.array(labels)
    preds = np.array(preds)

    total_threats = np.sum(labels == THREAT)
    if total_threats == 0:
        return 0.0

    missed_threats = np.sum((labels == THREAT) & (preds == SAFE))
    return float(missed_threats / total_threats)


def macro_f1(labels, preds):
    return float(
        f1_score(
            labels,
            preds,
            labels=[SAFE, THREAT],
            average="macro",
            zero_division=0,
        )
    )


def warn_if_single_class(split_name, preds):
    counts = prediction_counts(preds)

    if counts["SAFE"] == 0 or counts["THREAT"] == 0:
        print(
            f"WARNING: {split_name} predictions contain only one class: "
            f"SAFE={counts['SAFE']} THREAT={counts['THREAT']}"
        )


@torch.no_grad()
def evaluate(model, dataloader, device, loss_fn):
    model.eval()

    total_loss = 0.0
    all_preds = []
    all_labels = []

    for batch in tqdm(dataloader, desc="Evaluating", leave=False):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits

        loss = loss_fn(logits, labels)
        total_loss += loss.item()

        preds = torch.argmax(logits, dim=1)

        all_preds.extend(preds.cpu().numpy().tolist())
        all_labels.extend(labels.cpu().numpy().tolist())

    avg_loss = total_loss / len(dataloader)
    metrics = {
        "loss": float(avg_loss),
        "macro_f1": macro_f1(all_labels, all_preds),
        "fnr": false_negative_rate(all_labels, all_preds),
        "prediction_counts": prediction_counts(all_preds),
    }
    return metrics, all_preds, all_labels


def save_predictions(path, texts, labels, preds):
    fieldnames = ["row_id", "tweet", "true_label", "pred_label"]

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for idx, text in enumerate(texts):
            writer.writerow({
                "row_id": idx,
                "tweet": text,
                "true_label": int(labels[idx]),
                "pred_label": int(preds[idx]),
            })


def print_label_counts(name, labels):
    counts = prediction_counts(labels)
    print(f"{name} label counts: SAFE={counts['SAFE']} THREAT={counts['THREAT']}")


def main():
    set_seed(RANDOM_SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    device = get_device()
    print("Using device:", device)

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
    print_label_counts("Subtrain", y_subtrain)
    print_label_counts("Validation", y_val)
    print_label_counts("Official test", y_test)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    subtrain_dataset = TextClassificationDataset(
        subtrain_texts,
        y_subtrain,
        tokenizer,
        MAX_LENGTH,
    )
    val_dataset = TextClassificationDataset(
        val_texts,
        y_val,
        tokenizer,
        MAX_LENGTH,
    )

    subtrain_loader = DataLoader(
        subtrain_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2,
    )
    model.to(device)

    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.array([SAFE, THREAT]),
        y=y_subtrain,
    )
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    total_training_steps = len(subtrain_loader) * NUM_EPOCHS
    warmup_steps = int(total_training_steps * WARMUP_RATIO)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_training_steps,
    )

    best_val_f1 = -1.0
    best_epoch = None
    history = []

    for epoch in range(NUM_EPOCHS):
        model.train()
        total_train_loss = 0.0

        progress = tqdm(
            subtrain_loader,
            desc=f"Epoch {epoch + 1}/{NUM_EPOCHS}",
        )

        for batch in progress:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = loss_fn(outputs.logits, labels)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                GRADIENT_CLIP_NORM,
            )
            optimizer.step()
            scheduler.step()

            total_train_loss += loss.item()
            progress.set_postfix(loss=f"{loss.item():.4f}")

        train_loss = float(total_train_loss / len(subtrain_loader))
        val_metrics, val_preds, _ = evaluate(model, val_loader, device, loss_fn)
        val_counts = val_metrics["prediction_counts"]

        print(
            f"\nEpoch {epoch + 1}: "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} | "
            f"val_macro_f1={val_metrics['macro_f1']:.4f} | "
            f"val_pred_counts SAFE={val_counts['SAFE']} THREAT={val_counts['THREAT']}"
        )
        warn_if_single_class("validation", val_preds)

        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "validation": val_metrics,
        })

        if val_metrics["macro_f1"] > best_val_f1:
            best_val_f1 = val_metrics["macro_f1"]
            best_epoch = epoch + 1
            model.save_pretrained(MODEL_DIR)
            tokenizer.save_pretrained(MODEL_DIR)
            print(f"Saved new best validation checkpoint to: {MODEL_DIR}")

    print(f"\nBest epoch selected on validation: {best_epoch}")
    print(f"Best validation Macro F1: {best_val_f1:.4f}")

    best_model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    best_model.to(device)

    best_val_metrics, best_val_preds, best_val_labels = evaluate(
        best_model,
        val_loader,
        device,
        loss_fn,
    )
    save_predictions(VAL_PREDICTIONS_PATH, val_texts, y_val, best_val_preds)

    print("\nRunning final evaluation once on official test split...")
    test_dataset = TextClassificationDataset(
        test_texts,
        y_test,
        tokenizer,
        MAX_LENGTH,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    test_metrics, test_preds, test_labels = evaluate(
        best_model,
        test_loader,
        device,
        loss_fn,
    )
    save_predictions(TEST_PREDICTIONS_PATH, test_texts, y_test, test_preds)

    print(f"Test Macro F1: {test_metrics['macro_f1']:.4f}")
    print(f"Test FNR: {test_metrics['fnr']:.4f}")
    print(
        "Test prediction counts: "
        f"SAFE={test_metrics['prediction_counts']['SAFE']} "
        f"THREAT={test_metrics['prediction_counts']['THREAT']}"
    )
    warn_if_single_class("test", test_preds)

    print("\nClassification report:")
    print(
        classification_report(
            test_labels,
            test_preds,
            labels=[SAFE, THREAT],
            target_names=["SAFE", "THREAT"],
            zero_division=0,
        )
    )

    if test_metrics["macro_f1"] < 0.70:
        print(
            "WARNING: OLID RoBERTa training likely failed. "
            "Do not use this checkpoint for cascade."
        )

    results = {
        "dataset_name": DATASET_NAME,
        "model_name": MODEL_NAME,
        "split_sizes": {
            "official_train": len(official_train_texts),
            "subtrain": len(subtrain_texts),
            "validation": len(val_texts),
            "official_test": len(test_texts),
        },
        "random_state": RANDOM_SEED,
        "validation_size": VALIDATION_SIZE,
        "label_mapping": {
            "NOT": SAFE,
            "OFF": THREAT,
        },
        "text_column": "tweet",
        "training_settings": {
            "epochs": NUM_EPOCHS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "max_length": MAX_LENGTH,
            "weight_decay": WEIGHT_DECAY,
            "warmup_ratio": WARMUP_RATIO,
            "warmup_steps": warmup_steps,
            "gradient_clip_norm": GRADIENT_CLIP_NORM,
            "class_weight": "balanced",
            "device": str(device),
        },
        "best_checkpoint": {
            "path": str(MODEL_DIR),
            "selected_by": "validation_macro_f1",
            "best_epoch": best_epoch,
            "best_validation_macro_f1": best_val_f1,
        },
        "best_validation_metrics": best_val_metrics,
        "test_metrics": test_metrics,
        "history": history,
        "outputs": {
            "model_dir": str(MODEL_DIR),
            "results_json": str(RESULTS_PATH),
            "validation_predictions_csv": str(VAL_PREDICTIONS_PATH),
            "test_predictions_csv": str(TEST_PREDICTIONS_PATH),
        },
    }

    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved best model to: {MODEL_DIR}")
    print(f"Saved results JSON to: {RESULTS_PATH}")
    print(f"Saved validation predictions to: {VAL_PREDICTIONS_PATH}")
    print(f"Saved test predictions to: {TEST_PREDICTIONS_PATH}")


if __name__ == "__main__":
    main()
