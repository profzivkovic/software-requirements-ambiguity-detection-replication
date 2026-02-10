"""
DistilBERT best-only experiment (CPU)

- Uses DistilBERT fine-tuned on PURE train+val
- Evaluates on PURE test (intra) and UserStories (cross)
- Uses best LR found in sweep: 3e-05
- CPU-friendly settings

Outputs:
results/
  bert_cpu_best_results.csv
  bert_cpu_report_PURE_test_intra.txt
  bert_cpu_report_PURE_to_UserStories_cross.txt

figures/
  bert_cpu_confusion_PURE_test_intra.png
  bert_cpu_confusion_PURE_to_UserStories_cross.png

Run:
  python distilbert_best_only_cpu.py
"""

import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    set_seed
)

# ----------------------------
# Config
# ----------------------------

PURE_PATH = os.path.join("data", "pure_labeled.csv")
US_PATH = os.path.join("data", "userstories_labeled.csv")

OUT_RESULTS_DIR = "results"
OUT_FIG_DIR = "figures"
os.makedirs(OUT_RESULTS_DIR, exist_ok=True)
os.makedirs(OUT_FIG_DIR, exist_ok=True)

SEED = 42
set_seed(SEED)

MODEL_NAME = "distilbert-base-uncased"
BEST_LR = 3e-5

MAX_LEN = 128
EPOCHS = 3

TRAIN_BATCH = 4
EVAL_BATCH = 8
GRAD_ACCUM = 2

WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.06

TEST_SIZE = 0.20
VAL_SIZE = 0.125  # of remaining (so 10% of total)

device = torch.device("cpu")
print("Using device:", device)


# ----------------------------
# Load data
# ----------------------------

def load_dataset(path, name):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing {name} dataset at {path}")
    df = pd.read_csv(path)
    df["requirement_text"] = df["requirement_text"].astype(str)
    df["label"] = pd.to_numeric(df["label"], errors="coerce").fillna(0).astype(int)
    return df


df_pure = load_dataset(PURE_PATH, "PURE")
df_us = load_dataset(US_PATH, "UserStories")

X_all = df_pure["requirement_text"].tolist()
y_all = df_pure["label"].tolist()

# Keep the same splitting protocol as the sweep:
X_trainval, X_test, y_trainval, y_test = train_test_split(
    X_all, y_all,
    test_size=TEST_SIZE,
    random_state=SEED,
    stratify=y_all
)

X_train, X_val, y_train, y_val = train_test_split(
    X_trainval, y_trainval,
    test_size=VAL_SIZE,
    random_state=SEED,
    stratify=y_trainval
)

# Combine train+val for final training
X_train_full = X_train + X_val
y_train_full = y_train + y_val

print(f"PURE splits: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")
print(f"PURE train+val: {len(X_train_full)}")
print("UserStories size:", len(df_us))


# ----------------------------
# Tokenize + Dataset
# ----------------------------

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize(texts):
    return tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=MAX_LEN
    )

class ReqDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.enc = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.enc.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


train_full_enc = tokenize(X_train_full)
test_enc = tokenize(X_test)

us_texts = df_us["requirement_text"].tolist()
us_labels = df_us["label"].tolist()
us_enc = tokenize(us_texts)

train_full_ds = ReqDataset(train_full_enc, y_train_full)
test_ds = ReqDataset(test_enc, y_test)
us_ds = ReqDataset(us_enc, us_labels)


# ----------------------------
# Metrics + plotting
# ----------------------------

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    p, r, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary", pos_label=1, zero_division=0
    )
    return {"precision": p, "recall": r, "f1": f1}

def save_confusion(cm, title, fig_path):
    plt.figure()
    plt.imshow(cm, interpolation="nearest")
    plt.title(title)
    plt.colorbar()
    plt.xticks([0, 1], ["Clear (0)", "Ambiguous (1)"], rotation=20)
    plt.yticks([0, 1], ["Clear (0)", "Ambiguous (1)"])
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")
    plt.tight_layout()
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close()

def evaluate_and_save(trainer, dataset, y_true, tag):
    preds = trainer.predict(dataset)
    y_pred = np.argmax(preds.predictions, axis=1)

    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", pos_label=1, zero_division=0
    )

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    report = classification_report(y_true, y_pred, digits=4, zero_division=0)
    report_path = os.path.join(OUT_RESULTS_DIR, f"bert_cpu_report_{tag}.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)

    fig_path = os.path.join(OUT_FIG_DIR, f"bert_cpu_confusion_{tag}.png")
    save_confusion(cm, f"DistilBERT Confusion Matrix - {tag}", fig_path)

    return {
        "tag": tag,
        "model": MODEL_NAME,
        "learning_rate": BEST_LR,
        "precision": float(p),
        "recall": float(r),
        "f1": float(f1),
        "confusion_matrix": cm.tolist(),
        "report_file": report_path,
        "figure_file": fig_path,
    }


# ----------------------------
# Train best model
# ----------------------------

model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
model.to(device)

# NOTE: Using eval_strategy for compatibility with older transformers
training_args = TrainingArguments(
    output_dir=os.path.join("bert_out", "best_only"),
    eval_strategy="epoch",
    learning_rate=BEST_LR,
    per_device_train_batch_size=TRAIN_BATCH,
    per_device_eval_batch_size=EVAL_BATCH,
    gradient_accumulation_steps=GRAD_ACCUM,
    num_train_epochs=EPOCHS,
    weight_decay=WEIGHT_DECAY,
    warmup_ratio=WARMUP_RATIO,
    logging_steps=50,
    save_steps=500,
    save_total_limit=1,
    report_to="none",
    seed=SEED,
    fp16=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_full_ds,
    eval_dataset=test_ds,  # only for epoch-end logs
    compute_metrics=compute_metrics,
)

trainer.train()


# ----------------------------
# Final evaluations
# ----------------------------

rows = []
rows.append(evaluate_and_save(trainer, test_ds, y_test, tag="PURE_test_intra"))
rows.append(evaluate_and_save(trainer, us_ds, us_labels, tag="PURE_to_UserStories_cross"))

df_out = pd.DataFrame(rows)
out_path = os.path.join(OUT_RESULTS_DIR, "bert_cpu_best_results.csv")
df_out.to_csv(out_path, index=False)

print("\nSaved:", out_path)
print(df_out)
