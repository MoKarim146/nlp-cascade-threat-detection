"""Experiment 3 — Fine-tune roberta-base on HateXplain; saves best checkpoint for cascade use."""
from pathlib import Path
from collections import Counter
import json
import random

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from tqdm import tqdm

from datasets import load_dataset
from sklearn.metrics import classification_report, f1_score
from sklearn.utils.class_weight import compute_class_weight
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)


# =========================
# Config
# =========================
MODEL_NAME = "roberta-base"
MAX_LENGTH = 128
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
NUM_EPOCHS = 3
RANDOM_SEED = 42

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "core_thesis" / "models" / "roberta_hatexplain"
OUTPUT_DIR = PROJECT_ROOT / "core_thesis" / "outputs"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =========================
# Reproducibility
# =========================
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =========================
# HateXplain label handling
# =========================
# From your actual loaded schema:
# ['hatespeech', 'normal', 'offensive']
# ids: 0 = hatespeech, 1 = normal, 2 = offensive
def majority_vote(label_ids):
    counts = Counter(label_ids)
    return counts.most_common(1)[0][0]


def to_binary_label(raw_label_id: int) -> int:
    # Thesis binary setup:
    # normal -> SAFE (0)
    # hatespeech/offensive -> THREAT (1)
    if raw_label_id == 1:
        return 0
    if raw_label_id in {0, 2}:
        return 1
    raise ValueError(f"Unexpected HateXplain label id: {raw_label_id!r}")


def extract_text_and_label(row):
    text = " ".join(row["post_tokens"]).strip()
    majority_label_id = majority_vote(row["annotators"]["label"])
    label = to_binary_label(majority_label_id)
    return text, label


def prepare_split(split_dataset):
    texts = []
    labels = []

    for row in split_dataset:
        text, label = extract_text_and_label(row)
        if text:
            texts.append(text)
            labels.append(label)

    return texts, labels


# =========================
# Dataset
# =========================
class HateXplainDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        item = {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long),
        }

        return item


# =========================
# Device
# =========================
def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# =========================
# Evaluation
# =========================
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
    macro_f1 = f1_score(all_labels, all_preds, average="macro")

    return avg_loss, macro_f1, all_preds, all_labels


# =========================
# Main
# =========================
def main():
    set_seed(RANDOM_SEED)
    device = get_device()

    print("Using device:", device)

    # 1) Load official HateXplain splits
    ds = load_dataset("hatexplain")

    print("Available splits:", ds.keys())
    print("Train columns:", ds["train"].column_names)
    print("Example row:", ds["train"][0])

    # 2) Convert to thesis binary labels
    train_texts, y_train = prepare_split(ds["train"])
    val_texts, y_val = prepare_split(ds["validation"])
    test_texts, y_test = prepare_split(ds["test"])

    print("\nSplit sizes:")
    print("Train:", len(train_texts))
    print("Validation:", len(val_texts))
    print("Test:", len(test_texts))

    print("\nTrain class counts:")
    print("SAFE  :", sum(1 for y in y_train if y == 0))
    print("THREAT:", sum(1 for y in y_train if y == 1))

    # 3) Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # 4) Datasets and loaders
    train_dataset = HateXplainDataset(train_texts, y_train, tokenizer, MAX_LENGTH)
    val_dataset = HateXplainDataset(val_texts, y_val, tokenizer, MAX_LENGTH)
    test_dataset = HateXplainDataset(test_texts, y_test, tokenizer, MAX_LENGTH)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # 5) Model
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2,
    )
    model.to(device)

    # 6) Balanced class weights
    classes = np.array([0, 1])
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=np.array(y_train),
    )
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)

    print("\nClass weights:", class_weights.detach().cpu().numpy())

    loss_fn = nn.CrossEntropyLoss(weight=class_weights)

    # 7) Optimizer + scheduler
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

    total_training_steps = len(train_loader) * NUM_EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_training_steps,
    )

    # 8) Train
    best_val_f1 = -1.0
    best_epoch = -1
    history = []

    for epoch in range(NUM_EPOCHS):
        model.train()
        total_train_loss = 0.0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{NUM_EPOCHS}")

        for batch in progress_bar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits

            loss = loss_fn(logits, labels)
            loss.backward()

            optimizer.step()
            scheduler.step()

            total_train_loss += loss.item()
            progress_bar.set_postfix(train_loss=f"{loss.item():.4f}")

        avg_train_loss = total_train_loss / len(train_loader)

        val_loss, val_f1, _, _ = evaluate(model, val_loader, device, loss_fn)

        print(
            f"\nEpoch {epoch + 1}: "
            f"train_loss={avg_train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"val_macro_f1={val_f1:.4f}"
        )

        history.append({
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "val_loss": val_loss,
            "val_macro_f1": val_f1,
        })

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch + 1

            model.save_pretrained(MODEL_DIR)
            tokenizer.save_pretrained(MODEL_DIR)

            print(f"Saved new best model to: {MODEL_DIR}")

    # 9) Reload best model
    print(f"\nBest epoch: {best_epoch} with val_macro_f1={best_val_f1:.4f}")

    best_model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    best_model.to(device)

    # 10) Final test evaluation
    test_loss, test_f1, test_preds, test_labels = evaluate(
        best_model, test_loader, device, loss_fn
    )

    print(f"\nFinal Test Loss: {test_loss:.4f}")
    print(f"Final Test Macro F1: {test_f1:.4f}")
    print("\nClassification report:")
    print(classification_report(test_labels, test_preds, target_names=["SAFE", "THREAT"]))

    # 11) Save results
    results = {
        "model_name": MODEL_NAME,
        "device": str(device),
        "max_length": MAX_LENGTH,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "num_epochs": NUM_EPOCHS,
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_val_f1,
        "test_macro_f1": test_f1,
        "history": history,
    }

    results_path = OUTPUT_DIR / "hatexplain_roberta_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved results to: {results_path}")
    print(f"Saved model to:   {MODEL_DIR}")


if __name__ == "__main__":
    main()