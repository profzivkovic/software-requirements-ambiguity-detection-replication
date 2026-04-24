
from __future__ import annotations
import os, json, random
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef, cohen_kappa_score, balanced_accuracy_score, confusion_matrix, classification_report

PURE_PATH = os.path.join("data", "pure_labeled.csv")
US_PATH = os.path.join("data", "userstories_labeled.csv")
RESULTS_DIR = "results"
FIGURES_DIR = "figures"
SEEDS = [13, 21, 42, 87, 123]
TEXT_COL = "requirement_text"
LABEL_COL = "label"
REASONS_COL = "reasons"
TRIGGER_TERMS = ["should","may","could","might","appropriate","adequate","fast","quick","efficient","user-friendly","as soon as possible","if needed","etc","and/or"]

def ensure_dirs():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

def set_global_seed(seed: int):
    random.seed(seed); np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass

def safe_float(x):
    try: return float(x)
    except Exception: return float("nan")

def _validate_dataset(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    for col in [TEXT_COL, LABEL_COL]:
        if col not in df.columns:
            raise ValueError(f"{dataset_name} missing required column: {col}")
    df = df.copy()
    df[TEXT_COL] = df[TEXT_COL].astype(str).fillna("")
    df[LABEL_COL] = pd.to_numeric(df[LABEL_COL], errors="coerce").fillna(0).astype(int)
    if REASONS_COL not in df.columns:
        df[REASONS_COL] = np.nan
    return df

def load_dataset(path: str, dataset_name: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing dataset {dataset_name}: {path}")
    return _validate_dataset(pd.read_csv(path), dataset_name)

def load_all_datasets():
    return load_dataset(PURE_PATH, "PURE"), load_dataset(US_PATH, "UserStories")

def get_xy(df: pd.DataFrame):
    return df[TEXT_COL].astype(str).tolist(), df[LABEL_COL].astype(int).values

def parse_reason_string(value) -> List[str]:
    if pd.isna(value): return []
    s = str(value).strip()
    if not s: return []
    return [part.strip() for part in s.split(",") if part.strip()]

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "accuracy": safe_float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": safe_float(balanced_accuracy_score(y_true, y_pred)),
        "precision": safe_float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": safe_float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": safe_float(f1_score(y_true, y_pred, zero_division=0)),
        "mcc": safe_float(matthews_corrcoef(y_true, y_pred)),
        "cohen_kappa": safe_float(cohen_kappa_score(y_true, y_pred)),
    }

def compute_confusion_values(y_true: np.ndarray, y_pred: np.ndarray):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()
    return {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}

def classification_report_text(y_true: np.ndarray, y_pred: np.ndarray) -> str:
    return classification_report(y_true, y_pred, labels=[0,1], target_names=["clear","ambiguous"], digits=4, zero_division=0)

def save_text(path: str, content: str):
    with open(path, "w", encoding="utf-8") as f: f.write(content)

def save_json(path: str, obj):
    with open(path, "w", encoding="utf-8") as f: json.dump(obj, f, indent=2, ensure_ascii=False)

def mask_trigger_terms(texts: List[str], mask_token="[TRIGGER]") -> List[str]:
    out = []
    for text in texts:
        t = str(text)
        for trig in TRIGGER_TERMS:
            t = t.replace(trig, mask_token).replace(trig.capitalize(), mask_token).replace(trig.upper(), mask_token)
        out.append(t)
    return out

def remove_trigger_terms(texts: List[str]) -> List[str]:
    out = []
    for text in texts:
        t = str(text)
        for trig in TRIGGER_TERMS:
            t = t.replace(trig, " ").replace(trig.capitalize(), " ").replace(trig.upper(), " ")
        out.append(" ".join(t.split()))
    return out

def plot_confusion_matrix(y_true, y_pred, title, out_path):
    cm = confusion_matrix(y_true, y_pred, labels=[0,1])
    plt.figure(figsize=(5,4))
    plt.imshow(cm, interpolation="nearest")
    plt.title(title); plt.colorbar()
    plt.xticks([0,1], ["Pred 0", "Pred 1"]); plt.yticks([0,1], ["True 0", "True 1"])
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i,j]), ha="center", va="center")
    plt.ylabel("True label"); plt.xlabel("Predicted label")
    plt.tight_layout(); plt.savefig(out_path, dpi=300, bbox_inches="tight"); plt.close()

def plot_metric_bars(summary_df: pd.DataFrame, metric: str, title: str, out_path: str, model_col: str = "model", error_col: Optional[str] = None):
    df = summary_df.copy().sort_values(metric, ascending=False)
    x = np.arange(len(df))
    plt.figure(figsize=(10,5))
    if error_col and error_col in df.columns:
        plt.bar(x, df[metric].values, yerr=df[error_col].values, capsize=4)
    else:
        plt.bar(x, df[metric].values)
    plt.xticks(x, df[model_col].tolist(), rotation=30, ha="right")
    plt.ylabel(metric); plt.title(title); plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight"); plt.close()

def plot_reason_heatmap(df_reason_perf: pd.DataFrame, title: str, out_path: str, value_col: str = "f1"):
    if df_reason_perf.empty: return
    pivot = df_reason_perf.pivot(index="reason", columns="model", values=value_col)
    plt.figure(figsize=(10, max(4, len(pivot)*0.5)))
    plt.imshow(pivot.fillna(np.nan).values, aspect="auto", interpolation="nearest")
    plt.title(title); plt.colorbar()
    plt.xticks(np.arange(len(pivot.columns)), pivot.columns, rotation=30, ha="right")
    plt.yticks(np.arange(len(pivot.index)), pivot.index)
    values = pivot.values
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            v = values[i, j]
            plt.text(j, i, "nan" if pd.isna(v) else f"{v:.2f}", ha="center", va="center", fontsize=8)
    plt.tight_layout(); plt.savefig(out_path, dpi=300, bbox_inches="tight"); plt.close()

def summarize_multiseed_results(df: pd.DataFrame, group_cols: List[str], metrics: Optional[List[str]] = None):
    if metrics is None:
        metrics = ["accuracy","balanced_accuracy","precision","recall","f1","mcc","cohen_kappa"]
    agg = {m: ["mean","std"] for m in metrics}
    out = df.groupby(group_cols, dropna=False).agg(agg)
    out.columns = ["_".join(col).strip() for col in out.columns.values]
    return out.reset_index()

def per_reason_performance(df_instances: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
    rows = []
    if REASONS_COL not in df_instances.columns:
        return pd.DataFrame()
    all_reasons = set()
    parsed_reasons = df_instances[REASONS_COL].apply(parse_reason_string)
    for items in parsed_reasons:
        all_reasons.update(items)
    for reason in sorted(all_reasons):
        mask = parsed_reasons.apply(lambda items: reason in items)
        subset = df_instances.loc[mask].copy()
        if subset.empty: continue
        for keys, sub in subset.groupby(group_cols, dropna=False):
            if not isinstance(keys, tuple): keys = (keys,)
            metrics = compute_metrics(sub["y_true"].astype(int).values, sub["y_pred"].astype(int).values)
            row = {col: key for col, key in zip(group_cols, keys)}
            row["reason"] = reason; row["n"] = len(sub); row.update(metrics); rows.append(row)
    return pd.DataFrame(rows)
