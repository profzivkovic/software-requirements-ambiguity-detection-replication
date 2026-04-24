from __future__ import annotations

import os
import gc
import time
import argparse
import warnings
from typing import Dict, List

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW

from transformers import T5Tokenizer, T5ForConditionalGeneration, get_linear_schedule_with_warmup

from experiment_utils import (
    SEEDS,
    RESULTS_DIR,
    FIGURES_DIR,
    TEXT_COL,
    LABEL_COL,
    REASONS_COL,
    ensure_dirs,
    set_global_seed,
    load_all_datasets,
    compute_metrics,
    compute_confusion_values,
    classification_report_text,
    plot_confusion_matrix,
    summarize_multiseed_results,
    per_reason_performance,
)

warnings.filterwarnings("ignore")

DEFAULT_MODELS = [
    "t5-small",
    "google/flan-t5-small",
]

MODEL_SAFE_NAMES = {
    "t5-small": "t5-small",
    "google/flan-t5-small": "google_flan-t5-small",
}

TEST_SIZE = 0.2
VAL_SIZE_FROM_TRAIN = 0.125
MAX_INPUT_LENGTH = 128
MAX_TARGET_LENGTH = 8

DEFAULT_EPOCHS = 3
DEFAULT_LR = 1e-4
DEFAULT_TRAIN_BATCH = 8
DEFAULT_EVAL_BATCH = 8
DEFAULT_WEIGHT_DECAY = 0.01
DEFAULT_WARMUP_RATIO = 0.06

PARTIAL_SAVE_EVERY = 1


def label_to_text(label: int) -> str:
    return "yes" if int(label) == 1 else "no"


def text_to_label(text: str) -> int:
    t = str(text).strip().lower()
    if t.startswith("yes"):
        return 1
    if t.startswith("no"):
        return 0
    if "yes" in t:
        return 1
    return 0


def build_prompt(requirement_text: str) -> str:
    return f"is this requirement ambiguous? requirement: {requirement_text}"


class T5ClassificationDataset(Dataset):
    def __init__(
        self,
        texts: List[str],
        labels: np.ndarray,
        tokenizer: T5Tokenizer,
        max_input_length: int,
        max_target_length: int,
    ):
        self.texts = [build_prompt(str(x)) for x in texts]
        self.targets = [label_to_text(int(y)) for y in labels]
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        source = self.tokenizer(
            self.texts[idx],
            max_length=self.max_input_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        target = self.tokenizer(
            self.targets[idx],
            max_length=self.max_target_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        labels = target["input_ids"].squeeze(0)
        labels[labels == self.tokenizer.pad_token_id] = -100

        item = {
            "input_ids": source["input_ids"].squeeze(0),
            "attention_mask": source["attention_mask"].squeeze(0),
            "labels": labels,
        }
        return item


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_device_info() -> str:
    if torch.cuda.is_available():
        return f"cuda:{torch.cuda.get_device_name(0)}"
    return "cpu"


def cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def decode_predictions(
    model: T5ForConditionalGeneration,
    tokenizer: T5Tokenizer,
    dataloader: DataLoader,
    device,
    max_target_length: int,
):
    model.eval()

    decoded_preds = []
    decoded_gold = []
    total_loss = 0.0
    steps = 0

    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}

            outputs = model(**batch)
            loss = outputs.loss

            generated_ids = model.generate(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                max_length=max_target_length,
                num_beams=1,
            )

            pred_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

            gold_labels = batch["labels"].clone()
            gold_labels[gold_labels == -100] = tokenizer.pad_token_id
            gold_texts = tokenizer.batch_decode(gold_labels, skip_special_tokens=True)

            decoded_preds.extend(pred_texts)
            decoded_gold.extend(gold_texts)

            total_loss += loss.item()
            steps += 1

    avg_loss = total_loss / max(steps, 1)
    y_true = np.array([text_to_label(x) for x in decoded_gold])
    y_pred = np.array([text_to_label(x) for x in decoded_preds])
    metrics = compute_metrics(y_true, y_pred)

    return avg_loss, metrics, y_true, y_pred, decoded_preds, decoded_gold


def train_one_epoch(model, dataloader, optimizer, scheduler, device):
    model.train()
    total_loss = 0.0
    steps = 0

    for batch_idx, batch in enumerate(dataloader, start=1):
        batch = {k: v.to(device) for k, v in batch.items()}

        optimizer.zero_grad()
        outputs = model(**batch)
        loss = outputs.loss

        if torch.isnan(loss):
            print("      [warn] NaN loss detected, skipping batch")
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)

        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        steps += 1

        if batch_idx % 25 == 0 or batch_idx == len(dataloader):
            print(f"      [batch {batch_idx}/{len(dataloader)}] loss={loss.item():.4f}")

    return total_loss / max(steps, 1)


def train_model(
    model_name: str,
    train_texts: List[str],
    train_labels: np.ndarray,
    val_texts: List[str],
    val_labels: np.ndarray,
    test_texts: List[str],
    test_labels: np.ndarray,
    seed: int,
    epochs: int,
    learning_rate: float,
    train_batch_size: int,
    eval_batch_size: int,
    max_input_length: int,
    max_target_length: int,
):
    set_global_seed(seed)

    device = get_device()
    print(f"  device={get_device_info()}")

    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    model.to(device)

    train_ds = T5ClassificationDataset(
        train_texts, train_labels, tokenizer, max_input_length, max_target_length
    )
    val_ds = T5ClassificationDataset(
        val_texts, val_labels, tokenizer, max_input_length, max_target_length
    )
    test_ds = T5ClassificationDataset(
        test_texts, test_labels, tokenizer, max_input_length, max_target_length
    )

    train_loader = DataLoader(train_ds, batch_size=train_batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=eval_batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=eval_batch_size, shuffle=False)

    optimizer = AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=DEFAULT_WEIGHT_DECAY,
    )

    total_steps = len(train_loader) * epochs
    warmup_steps = int(DEFAULT_WARMUP_RATIO * total_steps)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    best_val_f1 = -1.0
    best_state_dict = None
    best_epoch = 0
    patience = 2
    bad_epochs = 0

    for epoch in range(1, epochs + 1):
        print(f"    [EPOCH {epoch}/{epochs}] training...")
        train_loss = train_one_epoch(model, train_loader, optimizer, scheduler, device)

        print(f"    [EPOCH {epoch}/{epochs}] validating...")
        val_loss, val_metrics, _, _, _, _ = decode_predictions(
            model, tokenizer, val_loader, device, max_target_length
        )

        print(
            f"    [EPOCH {epoch}/{epochs}] "
            f"train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f} "
            f"val_f1={val_metrics['f1']:.4f} "
            f"val_acc={val_metrics['accuracy']:.4f}"
        )

        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(f"    [EARLY STOP] no improvement for {patience} epochs")
                break

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    print(f"    [BEST] epoch={best_epoch} val_f1={best_val_f1:.4f}")

    print("    [TEST] evaluating best model...")
    test_loss, test_metrics, y_true, y_pred, decoded_preds, decoded_gold = decode_predictions(
        model, tokenizer, test_loader, device, max_target_length
    )

    del train_loader, val_loader, test_loader
    del train_ds, val_ds, test_ds
    del optimizer, scheduler
    cleanup()

    return (
        model,
        tokenizer,
        test_loss,
        test_metrics,
        y_true,
        y_pred,
        decoded_preds,
        decoded_gold,
        best_epoch,
        best_val_f1,
    )


def run_one_t5_experiment(
    model_name: str,
    setting: str,
    train_dataset_name: str,
    test_dataset_name: str,
    seed: int,
    train_texts: List[str],
    train_labels: np.ndarray,
    val_texts: List[str],
    val_labels: np.ndarray,
    test_texts: List[str],
    test_labels: np.ndarray,
    test_meta_df: pd.DataFrame,
    num_epochs: int,
    learning_rate: float,
    train_batch_size: int,
    eval_batch_size: int,
    max_input_length: int,
    max_target_length: int,
):
    start = time.time()

    safe_model_name = MODEL_SAFE_NAMES.get(model_name, model_name.replace("/", "_"))
    run_name = f"{safe_model_name}_{setting}_{train_dataset_name}_to_{test_dataset_name}_seed{seed}"

    print("=" * 100)
    print(f"[START] {run_name}")
    print(f"  train={len(train_texts)} val={len(val_texts)} test={len(test_texts)}")
    print(
        f"  epochs={num_epochs} lr={learning_rate} "
        f"batch_train={train_batch_size} batch_eval={eval_batch_size}"
    )
    print("=" * 100)

    (
        model,
        tokenizer,
        test_loss,
        metrics,
        y_true,
        y_pred,
        decoded_preds,
        decoded_gold,
        best_epoch,
        best_val_f1,
    ) = train_model(
        model_name=model_name,
        train_texts=train_texts,
        train_labels=train_labels,
        val_texts=val_texts,
        val_labels=val_labels,
        test_texts=test_texts,
        test_labels=test_labels,
        seed=seed,
        epochs=num_epochs,
        learning_rate=learning_rate,
        train_batch_size=train_batch_size,
        eval_batch_size=eval_batch_size,
        max_input_length=max_input_length,
        max_target_length=max_target_length,
    )

    conf = compute_confusion_values(y_true, y_pred)

    row = {
        "setting": setting,
        "variant": "original",
        "train_dataset": train_dataset_name,
        "test_dataset": test_dataset_name,
        "model": model_name,
        "seed": seed,
        **metrics,
        **conf,
        "n_train": len(train_texts),
        "n_val": len(val_texts),
        "n_test": len(test_texts),
        "epochs": num_epochs,
        "learning_rate": learning_rate,
        "train_batch_size": train_batch_size,
        "eval_batch_size": eval_batch_size,
        "max_input_length": max_input_length,
        "max_target_length": max_target_length,
        "best_epoch": best_epoch,
        "best_val_f1": best_val_f1,
        "test_loss": test_loss,
        "runtime_sec": round(time.time() - start, 2),
        "device": get_device_info(),
    }

    instance_df = pd.DataFrame({
        "setting": setting,
        "variant": "original",
        "train_dataset": train_dataset_name,
        "test_dataset": test_dataset_name,
        "model": model_name,
        "seed": seed,
        "text": test_meta_df[TEXT_COL].astype(str).values,
        "y_true": y_true,
        "y_pred": y_pred,
        "pred_text": decoded_preds,
        "gold_text": decoded_gold,
        "reasons": test_meta_df[REASONS_COL].values if REASONS_COL in test_meta_df.columns else np.nan,
    })

    report_path = os.path.join(RESULTS_DIR, f"{run_name}_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(classification_report_text(y_true, y_pred))

    fig_path = os.path.join(FIGURES_DIR, f"{run_name}_confusion.png")
    plot_confusion_matrix(
        y_true=y_true,
        y_pred=y_pred,
        title=run_name,
        out_path=fig_path,
    )

    print(
        f"[DONE] {run_name} | "
        f"f1={row['f1']:.4f} "
        f"acc={row['accuracy']:.4f} "
        f"mcc={row['mcc']:.4f} "
        f"kappa={row['cohen_kappa']:.4f} "
        f"runtime={row['runtime_sec']}s"
    )

    del model, tokenizer
    cleanup()

    return row, instance_df


def split_intra_dataset(df: pd.DataFrame, seed: int):
    texts = df[TEXT_COL].astype(str).tolist()
    labels = df[LABEL_COL].astype(int).values
    indices = np.arange(len(df))

    x_trainval, x_test, y_trainval, y_test, idx_trainval, idx_test = train_test_split(
        texts,
        labels,
        indices,
        test_size=0.2,
        stratify=labels,
        random_state=seed,
    )

    x_train, x_val, y_train, y_val, idx_train, idx_val = train_test_split(
        x_trainval,
        y_trainval,
        idx_trainval,
        test_size=0.125,
        stratify=y_trainval,
        random_state=seed,
    )

    test_df = df.iloc[idx_test].reset_index(drop=True)

    return (
        list(x_train), np.array(y_train),
        list(x_val), np.array(y_val),
        list(x_test), np.array(y_test),
        test_df,
    )


def split_cross_dataset(train_df: pd.DataFrame, test_df: pd.DataFrame, seed: int):
    train_texts_all = train_df[TEXT_COL].astype(str).tolist()
    train_labels_all = train_df[LABEL_COL].astype(int).values
    train_indices = np.arange(len(train_df))

    x_train, x_val, y_train, y_val, idx_train, idx_val = train_test_split(
        train_texts_all,
        train_labels_all,
        train_indices,
        test_size=0.125,
        stratify=train_labels_all,
        random_state=seed,
    )

    x_test = test_df[TEXT_COL].astype(str).tolist()
    y_test = test_df[LABEL_COL].astype(int).values
    test_meta_df = test_df.reset_index(drop=True)

    return (
        list(x_train), np.array(y_train),
        list(x_val), np.array(y_val),
        list(x_test), np.array(y_test),
        test_meta_df,
    )


def save_partial_outputs(
    model_name: str,
    rows: List[Dict[str, object]],
    instance_dfs: List[pd.DataFrame],
):
    safe_model_name = MODEL_SAFE_NAMES.get(model_name, model_name.replace("/", "_"))

    results_df = pd.DataFrame(rows)
    instances_df = pd.concat(instance_dfs, ignore_index=True) if instance_dfs else pd.DataFrame()

    results_df.to_csv(
        os.path.join(RESULTS_DIR, f"{safe_model_name}_all_results.csv"),
        index=False,
    )

    if not instances_df.empty:
        instances_df.to_csv(
            os.path.join(RESULTS_DIR, f"{safe_model_name}_instance_predictions.csv"),
            index=False,
        )

        reason_perf_df = per_reason_performance(
            df_instances=instances_df[instances_df["variant"] == "original"].copy(),
            group_cols=["setting", "train_dataset", "test_dataset", "model"],
        )
        reason_perf_df.to_csv(
            os.path.join(RESULTS_DIR, f"{safe_model_name}_reason_performance.csv"),
            index=False,
        )

    original_df = results_df[results_df["variant"] == "original"].copy()
    if not original_df.empty:
        summary_df = summarize_multiseed_results(
            original_df,
            group_cols=["setting", "train_dataset", "test_dataset", "model"],
        )
        summary_df.to_csv(
            os.path.join(RESULTS_DIR, f"{safe_model_name}_summary.csv"),
            index=False,
        )

    print("-" * 100)
    print(f"[PARTIAL SAVE] model={model_name}")
    print("-" * 100)


def run_model(
    model_name: str,
    epochs: int,
    learning_rate: float,
    train_batch_size: int,
    eval_batch_size: int,
    max_input_length: int,
    max_target_length: int,
):
    ensure_dirs()

    pure_df, us_df = load_all_datasets()

    rows: List[Dict[str, object]] = []
    instance_dfs: List[pd.DataFrame] = []
    exp_counter = 0
    total_experiments = len(SEEDS) * 4

    print("#" * 100)
    print(f"Running model: {model_name}")
    print(f"Total experiments planned: {total_experiments}")
    print(f"Seeds: {SEEDS}")
    print(f"Device: {get_device_info()}")
    print("#" * 100)

    for seed in SEEDS:
        print(f"\n[SEED] ===== seed={seed} =====\n")

        configs = [
            ("intra", "PURE", "PURE", split_intra_dataset(pure_df, seed)),
            ("intra", "UserStories", "UserStories", split_intra_dataset(us_df, seed)),
            ("cross", "PURE", "UserStories", split_cross_dataset(pure_df, us_df, seed)),
            ("cross", "UserStories", "PURE", split_cross_dataset(us_df, pure_df, seed)),
        ]

        for setting, train_name, test_name, split_data in configs:
            exp_counter += 1
            print(
                f"[PROGRESS] {exp_counter}/{total_experiments} | "
                f"model={model_name} | {setting} {train_name}->{test_name}"
            )

            (
                train_texts,
                train_labels,
                val_texts,
                val_labels,
                test_texts,
                test_labels,
                test_meta_df,
            ) = split_data

            row, inst = run_one_t5_experiment(
                model_name=model_name,
                setting=setting,
                train_dataset_name=train_name,
                test_dataset_name=test_name,
                seed=seed,
                train_texts=train_texts,
                train_labels=train_labels,
                val_texts=val_texts,
                val_labels=val_labels,
                test_texts=test_texts,
                test_labels=test_labels,
                test_meta_df=test_meta_df,
                num_epochs=epochs,
                learning_rate=learning_rate,
                train_batch_size=train_batch_size,
                eval_batch_size=eval_batch_size,
                max_input_length=max_input_length,
                max_target_length=max_target_length,
            )

            rows.append(row)
            instance_dfs.append(inst)

            if exp_counter % PARTIAL_SAVE_EVERY == 0:
                save_partial_outputs(model_name, rows, instance_dfs)

    save_partial_outputs(model_name, rows, instance_dfs)

    print("\n" + "#" * 100)
    print(f"[FINISHED] model={model_name}")
    print("#" * 100)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="t5-small")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--train_batch_size", type=int, default=DEFAULT_TRAIN_BATCH)
    parser.add_argument("--eval_batch_size", type=int, default=DEFAULT_EVAL_BATCH)
    parser.add_argument("--max_input_length", type=int, default=MAX_INPUT_LENGTH)
    parser.add_argument("--max_target_length", type=int, default=MAX_TARGET_LENGTH)
    return parser.parse_args()


def main():
    args = parse_args()

    if args.model not in DEFAULT_MODELS:
        print("[WARN] Model nije u recommended listi, ali pokušavam da ga pokrenem.")
        print(f"       Prosledio si: {args.model}")
        print(f"       Recommended: {DEFAULT_MODELS}")

    run_model(
        model_name=args.model,
        epochs=args.epochs,
        learning_rate=args.lr,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
        max_input_length=args.max_input_length,
        max_target_length=args.max_target_length,
    )


if __name__ == "__main__":
    main()
