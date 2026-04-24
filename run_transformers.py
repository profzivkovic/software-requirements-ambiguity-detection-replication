
from __future__ import annotations
import os, gc, time, argparse, warnings
from typing import Dict, List, Tuple
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from experiment_utils import SEEDS, RESULTS_DIR, FIGURES_DIR, TEXT_COL, LABEL_COL, REASONS_COL, ensure_dirs, set_global_seed, load_all_datasets, compute_metrics, compute_confusion_values, classification_report_text, plot_confusion_matrix, summarize_multiseed_results, per_reason_performance
warnings.filterwarnings("ignore")
DEFAULT_MODELS = ["distilbert-base-uncased", "roberta-base", "microsoft/deberta-v3-base"]
MODEL_SAFE_NAMES = {"distilbert-base-uncased": "distilbert-base-uncased", "roberta-base": "roberta-base", "microsoft/deberta-v3-base": "microsoft_deberta-v3-base"}
TEST_SIZE = 0.2
VAL_SIZE_FROM_TRAIN = 0.125
MAX_LENGTH = 128
DEFAULT_EPOCHS = 3
DEFAULT_LR = 3e-5
DEFAULT_TRAIN_BATCH = 8
DEFAULT_EVAL_BATCH = 8
DEFAULT_WEIGHT_DECAY = 0.01
DEFAULT_WARMUP_RATIO = 0.06
PARTIAL_SAVE_EVERY = 1
class TextClassificationDataset(Dataset):
    def __init__(self, texts: List[str], labels: np.ndarray, tokenizer, max_length: int):
        self.texts = [str(x) for x in texts]
        self.labels = [int(y) for y in labels]
        self.tokenizer = tokenizer
        self.max_length = max_length
    def __len__(self): return len(self.texts)
    def __getitem__(self, idx):
        encoding = self.tokenizer(self.texts[idx], truncation=True, padding="max_length", max_length=self.max_length, return_tensors="pt")
        item = {k: v.squeeze(0) for k, v in encoding.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item
def get_device(): return torch.device("cuda" if torch.cuda.is_available() else "cpu")
def get_device_info() -> str: return f"cuda:{torch.cuda.get_device_name(0)}" if torch.cuda.is_available() else "cpu"
def cleanup():
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
def evaluate_model(model, dataloader, device):
    model.eval(); all_preds=[]; all_labels=[]; total_loss=0.0; steps=0
    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            preds = torch.argmax(outputs.logits, dim=1)
            total_loss += outputs.loss.item(); steps += 1
            all_preds.extend(preds.cpu().numpy()); all_labels.extend(batch["labels"].cpu().numpy())
    y_true = np.array(all_labels); y_pred = np.array(all_preds)
    return total_loss/max(steps,1), compute_metrics(y_true, y_pred), y_true, y_pred
def train_one_epoch(model, dataloader, optimizer, scheduler, device):
    model.train(); total_loss=0.0; steps=0
    for batch_idx, batch in enumerate(dataloader, start=1):
        batch = {k: v.to(device) for k, v in batch.items()}
        optimizer.zero_grad(); outputs = model(**batch); loss = outputs.loss; loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step(); scheduler.step()
        total_loss += loss.item(); steps += 1
        if batch_idx % 50 == 0 or batch_idx == len(dataloader):
            print(f"      [batch {batch_idx}/{len(dataloader)}] loss={loss.item():.4f}")
    return total_loss/max(steps,1)
def train_model(model_name, train_texts, train_labels, val_texts, val_labels, test_texts, test_labels, seed, epochs, learning_rate, train_batch_size, eval_batch_size, max_length):
    set_global_seed(seed); device = get_device(); print(f"  device={get_device_info()}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2).to(device)
    train_loader = DataLoader(TextClassificationDataset(train_texts, train_labels, tokenizer, max_length), batch_size=train_batch_size, shuffle=True)
    val_loader = DataLoader(TextClassificationDataset(val_texts, val_labels, tokenizer, max_length), batch_size=eval_batch_size, shuffle=False)
    test_loader = DataLoader(TextClassificationDataset(test_texts, test_labels, tokenizer, max_length), batch_size=eval_batch_size, shuffle=False)
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=DEFAULT_WEIGHT_DECAY)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(DEFAULT_WARMUP_RATIO*total_steps), num_training_steps=total_steps)
    best_val_f1=-1.0; best_state_dict=None; best_epoch=0; patience=2; bad_epochs=0
    for epoch in range(1, epochs+1):
        print(f"    [EPOCH {epoch}/{epochs}] training...")
        train_loss = train_one_epoch(model, train_loader, optimizer, scheduler, device)
        print(f"    [EPOCH {epoch}/{epochs}] validating...")
        val_loss, val_metrics, _, _ = evaluate_model(model, val_loader, device)
        print(f"    [EPOCH {epoch}/{epochs}] train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_f1={val_metrics['f1']:.4f} val_acc={val_metrics['accuracy']:.4f}")
        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]; best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}; best_epoch = epoch; bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(f"    [EARLY STOP] no improvement for {patience} epochs"); break
    if best_state_dict is not None: model.load_state_dict(best_state_dict)
    print(f"    [BEST] epoch={best_epoch} val_f1={best_val_f1:.4f}")
    print("    [TEST] evaluating best model...")
    test_loss, test_metrics, y_true, y_pred = evaluate_model(model, test_loader, device)
    del train_loader, val_loader, test_loader, optimizer, scheduler; cleanup()
    return model, tokenizer, test_loss, test_metrics, y_true, y_pred, best_epoch, best_val_f1
def run_one_transformer_experiment(model_name, setting, train_dataset_name, test_dataset_name, seed, train_texts, train_labels, val_texts, val_labels, test_texts, test_labels, test_meta_df, num_epochs, learning_rate, train_batch_size, eval_batch_size, max_length):
    start=time.time(); safe_model_name = MODEL_SAFE_NAMES.get(model_name, model_name.replace("/", "_")); run_name = f"{safe_model_name}_{setting}_{train_dataset_name}_to_{test_dataset_name}_seed{seed}"
    print("="*100); print(f"[START] {run_name}"); print(f"  train={len(train_texts)} val={len(val_texts)} test={len(test_texts)}"); print(f"  epochs={num_epochs} lr={learning_rate} batch_train={train_batch_size} batch_eval={eval_batch_size}"); print("="*100)
    model, tokenizer, test_loss, metrics, y_true, y_pred, best_epoch, best_val_f1 = train_model(model_name, train_texts, train_labels, val_texts, val_labels, test_texts, test_labels, seed, num_epochs, learning_rate, train_batch_size, eval_batch_size, max_length)
    conf = compute_confusion_values(y_true, y_pred)
    row = {"setting": setting, "variant": "original", "train_dataset": train_dataset_name, "test_dataset": test_dataset_name, "model": model_name, "seed": seed, **metrics, **conf, "n_train": len(train_texts), "n_val": len(val_texts), "n_test": len(test_texts), "epochs": num_epochs, "learning_rate": learning_rate, "train_batch_size": train_batch_size, "eval_batch_size": eval_batch_size, "max_length": max_length, "best_epoch": best_epoch, "best_val_f1": best_val_f1, "test_loss": test_loss, "runtime_sec": round(time.time()-start,2), "device": get_device_info()}
    instance_df = pd.DataFrame({"setting": setting, "variant": "original", "train_dataset": train_dataset_name, "test_dataset": test_dataset_name, "model": model_name, "seed": seed, "text": test_meta_df[TEXT_COL].astype(str).values, "y_true": y_true, "y_pred": y_pred, "reasons": test_meta_df[REASONS_COL].values if REASONS_COL in test_meta_df.columns else np.nan})
    with open(os.path.join(RESULTS_DIR, f"{run_name}_report.txt"), "w", encoding="utf-8") as f: f.write(classification_report_text(y_true, y_pred))
    plot_confusion_matrix(y_true, y_pred, run_name, os.path.join(FIGURES_DIR, f"{run_name}_confusion.png"))
    print(f"[DONE] {run_name} | f1={row['f1']:.4f} acc={row['accuracy']:.4f} mcc={row['mcc']:.4f} kappa={row['cohen_kappa']:.4f} runtime={row['runtime_sec']}s")
    del model, tokenizer; cleanup(); return row, instance_df
def split_intra_dataset(df, seed):
    texts = df[TEXT_COL].astype(str).tolist(); labels = df[LABEL_COL].astype(int).values; indices = np.arange(len(df))
    x_trainval, x_test, y_trainval, y_test, idx_trainval, idx_test = train_test_split(texts, labels, indices, test_size=TEST_SIZE, stratify=labels, random_state=seed)
    x_train, x_val, y_train, y_val, _, _ = train_test_split(x_trainval, y_trainval, idx_trainval, test_size=VAL_SIZE_FROM_TRAIN, stratify=y_trainval, random_state=seed)
    return list(x_train), np.array(y_train), list(x_val), np.array(y_val), list(x_test), np.array(y_test), df.iloc[idx_test].reset_index(drop=True)
def split_cross_dataset(train_df, test_df, seed):
    train_texts_all = train_df[TEXT_COL].astype(str).tolist(); train_labels_all = train_df[LABEL_COL].astype(int).values
    x_train, x_val, y_train, y_val, _, _ = train_test_split(train_texts_all, train_labels_all, np.arange(len(train_df)), test_size=0.125, stratify=train_labels_all, random_state=seed)
    return list(x_train), np.array(y_train), list(x_val), np.array(y_val), test_df[TEXT_COL].astype(str).tolist(), test_df[LABEL_COL].astype(int).values, test_df.reset_index(drop=True)
def save_partial_outputs(model_name, rows, instance_dfs):
    safe_model_name = MODEL_SAFE_NAMES.get(model_name, model_name.replace("/", "_"))
    results_df = pd.DataFrame(rows); instances_df = pd.concat(instance_dfs, ignore_index=True) if instance_dfs else pd.DataFrame()
    results_df.to_csv(os.path.join(RESULTS_DIR, f"{safe_model_name}_all_results.csv"), index=False)
    if not instances_df.empty:
        instances_df.to_csv(os.path.join(RESULTS_DIR, f"{safe_model_name}_instance_predictions.csv"), index=False)
        per_reason_performance(instances_df[instances_df["variant"]=="original"].copy(), ["setting","train_dataset","test_dataset","model"]).to_csv(os.path.join(RESULTS_DIR, f"{safe_model_name}_reason_performance.csv"), index=False)
    original_df = results_df[results_df["variant"]=="original"].copy()
    if not original_df.empty:
        summarize_multiseed_results(original_df, ["setting","train_dataset","test_dataset","model"]).to_csv(os.path.join(RESULTS_DIR, f"{safe_model_name}_summary.csv"), index=False)
    print("-"*100); print(f"[PARTIAL SAVE] model={model_name}"); print("-"*100)
def run_model(model_name, epochs, learning_rate, train_batch_size, eval_batch_size, max_length):
    ensure_dirs(); pure_df, us_df = load_all_datasets(); rows=[]; instance_dfs=[]; exp_counter=0; total_experiments=len(SEEDS)*4
    print("#"*100); print(f"Running model: {model_name}"); print(f"Total experiments planned: {total_experiments}"); print(f"Seeds: {SEEDS}"); print(f"Device: {get_device_info()}"); print("#"*100)
    for seed in SEEDS:
        print(f"\n[SEED] ===== seed={seed} =====\n")
        for label, split in [("intra PURE", split_intra_dataset(pure_df, seed)), ("intra UserStories", split_intra_dataset(us_df, seed)), ("cross PURE -> UserStories", split_cross_dataset(pure_df, us_df, seed)), ("cross UserStories -> PURE", split_cross_dataset(us_df, pure_df, seed))]:
            exp_counter += 1; print(f"[PROGRESS] {exp_counter}/{total_experiments} | model={model_name} | {label}")
            train_texts, train_labels, val_texts, val_labels, test_texts, test_labels, test_meta_df = split
            setting = "intra" if label.startswith("intra") else "cross"
            train_dataset_name, test_dataset_name = ("PURE","PURE") if label=="intra PURE" else ("UserStories","UserStories") if label=="intra UserStories" else ("PURE","UserStories") if label=="cross PURE -> UserStories" else ("UserStories","PURE")
            row, inst = run_one_transformer_experiment(model_name, setting, train_dataset_name, test_dataset_name, seed, train_texts, train_labels, val_texts, val_labels, test_texts, test_labels, test_meta_df, epochs, learning_rate, train_batch_size, eval_batch_size, max_length)
            rows.append(row); instance_dfs.append(inst); save_partial_outputs(model_name, rows, instance_dfs)
    save_partial_outputs(model_name, rows, instance_dfs)
    print("\n" + "#"*100); print(f"[FINISHED] model={model_name}"); print("#"*100)
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, required=True)
    p.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    p.add_argument("--lr", type=float, default=DEFAULT_LR)
    p.add_argument("--train_batch_size", type=int, default=DEFAULT_TRAIN_BATCH)
    p.add_argument("--eval_batch_size", type=int, default=DEFAULT_EVAL_BATCH)
    p.add_argument("--max_length", type=int, default=MAX_LENGTH)
    return p.parse_args()
def main():
    args = parse_args()
    if args.model not in DEFAULT_MODELS:
        print("[WARN] Model nije u recommended listi, ali pokušavam da ga pokrenem.")
    run_model(args.model, args.epochs, args.lr, args.train_batch_size, args.eval_batch_size, args.max_length)
if __name__ == "__main__":
    main()
