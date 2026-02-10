"""
Baseline ML experiments for ambiguity detection in requirements.

Datasets:
- data/pure_labeled.csv
- data/userstories_labeled.csv

Models:
- MultinomialNB
- LinearSVC (linear SVM)
- RandomForestClassifier
- LightGBM (LGBMClassifier)
- XGBoost (XGBClassifier)

Features:
- TF-IDF (word ngrams)

Evaluation:
- Intra-dataset: stratified train/test split
- Cross-dataset: train on PURE -> test on UserStories (and reverse)

Outputs:
- results/baseline_results.csv
- results/*.txt (classification reports)
- figures/confusion_*.png

Run:
  python baseline_ml_experiments.py
"""

from __future__ import annotations

import os
import json
from dataclasses import dataclass
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier

# Optional libs (installed via pip)
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier


# ----------------------------
# Configuration
# ----------------------------

PURE_PATH = os.path.join("data", "pure_labeled.csv")
US_PATH = os.path.join("data", "userstories_labeled.csv")

OUT_RESULTS_DIR = "results"
OUT_FIG_DIR = "figures"

RANDOM_STATE = 42
TEST_SIZE = 0.2

# TF-IDF settings (strong, standard baseline)
TFIDF_CONFIG = {
    "lowercase": True,
    "ngram_range": (1, 2),       # unigrams + bigrams
    "min_df": 2,
    "max_df": 0.95,
    "sublinear_tf": True,
}


# ----------------------------
# Utilities
# ----------------------------

def ensure_dirs() -> None:
    os.makedirs(OUT_RESULTS_DIR, exist_ok=True)
    os.makedirs(OUT_FIG_DIR, exist_ok=True)


def load_dataset(path: str, name: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing dataset {name}: {path}")
    df = pd.read_csv(path)
    # Required columns
    for col in ["requirement_text", "label"]:
        if col not in df.columns:
            raise ValueError(f"{name} missing required column: {col}")
    df["requirement_text"] = df["requirement_text"].astype(str)
    df["label"] = pd.to_numeric(df["label"], errors="coerce").fillna(0).astype(int)
    return df


def plot_confusion_matrix(cm: np.ndarray, title: str, fig_path: str) -> None:
    """
    Simple confusion matrix plot without custom colors.
    """
    plt.figure()
    plt.imshow(cm, interpolation="nearest")
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ["Clear (0)", "Ambiguous (1)"], rotation=20)
    plt.yticks(tick_marks, ["Clear (0)", "Ambiguous (1)"])
    plt.ylabel("True label")
    plt.xlabel("Predicted label")

    # annotate cells
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")

    plt.tight_layout()
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close()


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", pos_label=1, zero_division=0
    )
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


# ----------------------------
# Model definitions
# ----------------------------

def build_models() -> Dict[str, object]:
    """
    Returns models configured with sensible defaults.
    You can later add hyperparameter tuning if needed.
    """
    models = {}

    models["MultinomialNB"] = MultinomialNB()

    models["LinearSVM"] = LinearSVC(random_state=RANDOM_STATE)

    models["RandomForest"] = RandomForestClassifier(
        n_estimators=400,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        class_weight="balanced",
    )

    models["LightGBM"] = LGBMClassifier(
        n_estimators=600,
        learning_rate=0.05,
        num_leaves=31,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    models["XGBoost"] = XGBClassifier(
        n_estimators=600,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    return models


def build_pipeline(model) -> Pipeline:
    """
    TF-IDF + classifier pipeline.
    """
    return Pipeline([
        ("tfidf", TfidfVectorizer(**TFIDF_CONFIG)),
        ("clf", model),
    ])


# ----------------------------
# Experiment runners
# ----------------------------

def run_intra_dataset_experiment(
    df: pd.DataFrame,
    dataset_name: str,
    models: Dict[str, object],
) -> List[Dict[str, object]]:
    """
    Train/test on same dataset with stratified split.
    """
    X = df["requirement_text"].values
    y = df["label"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    results = []

    for model_name, model in models.items():
        pipe = build_pipeline(model)
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)

        metrics = compute_metrics(y_test, y_pred)

        cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
        fig_path = os.path.join(OUT_FIG_DIR, f"confusion_intra_{dataset_name}_{model_name}.png")
        plot_confusion_matrix(cm, f"Intra {dataset_name} - {model_name}", fig_path)

        report = classification_report(y_test, y_pred, digits=4, zero_division=0)
        report_path = os.path.join(OUT_RESULTS_DIR, f"report_intra_{dataset_name}_{model_name}.txt")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report)

        results.append({
            "setting": "intra",
            "train_dataset": dataset_name,
            "test_dataset": dataset_name,
            "model": model_name,
            **metrics,
            "confusion_matrix": cm.tolist(),
            "report_file": report_path,
            "figure_file": fig_path,
        })

    return results


def run_cross_dataset_experiment(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    train_name: str,
    test_name: str,
    models: Dict[str, object],
) -> List[Dict[str, object]]:
    """
    Train on all of train_df, evaluate on all of test_df.
    """
    X_train = train_df["requirement_text"].values
    y_train = train_df["label"].values
    X_test = test_df["requirement_text"].values
    y_test = test_df["label"].values

    results = []

    for model_name, model in models.items():
        pipe = build_pipeline(model)
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)

        metrics = compute_metrics(y_test, y_pred)

        cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
        fig_path = os.path.join(OUT_FIG_DIR, f"confusion_cross_{train_name}_to_{test_name}_{model_name}.png")
        plot_confusion_matrix(cm, f"Cross {train_name}→{test_name} - {model_name}", fig_path)

        report = classification_report(y_test, y_pred, digits=4, zero_division=0)
        report_path = os.path.join(OUT_RESULTS_DIR, f"report_cross_{train_name}_to_{test_name}_{model_name}.txt")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report)

        results.append({
            "setting": "cross",
            "train_dataset": train_name,
            "test_dataset": test_name,
            "model": model_name,
            **metrics,
            "confusion_matrix": cm.tolist(),
            "report_file": report_path,
            "figure_file": fig_path,
        })

    return results


# ----------------------------
# Main
# ----------------------------

def main() -> None:
    ensure_dirs()

    df_pure = load_dataset(PURE_PATH, "PURE")
    df_us = load_dataset(US_PATH, "UserStories")

    print("PURE:", df_pure.shape, "UserStories:", df_us.shape)
    print("PURE label counts:\n", df_pure["label"].value_counts())
    print("UserStories label counts:\n", df_us["label"].value_counts())

    models = build_models()

    all_results: List[Dict[str, object]] = []

    # Intra-dataset experiments
    all_results += run_intra_dataset_experiment(df_pure, "PURE", models)
    all_results += run_intra_dataset_experiment(df_us, "UserStories", models)

    # Cross-dataset experiments (strong generalization test)
    all_results += run_cross_dataset_experiment(df_pure, df_us, "PURE", "UserStories", models)
    all_results += run_cross_dataset_experiment(df_us, df_pure, "UserStories", "PURE", models)

    # Save results table
    results_df = pd.DataFrame(all_results)
    out_csv = os.path.join(OUT_RESULTS_DIR, "baseline_results.csv")
    results_df.to_csv(out_csv, index=False)
    print(f"\nWrote results to: {out_csv}")

    # Also save a "best models" quick view by setting
    view_cols = ["setting", "train_dataset", "test_dataset", "model", "precision", "recall", "f1"]
    best = (
        results_df[view_cols]
        .sort_values(["setting", "train_dataset", "test_dataset", "f1"], ascending=[True, True, True, False])
        .groupby(["setting", "train_dataset", "test_dataset"])
        .head(3)
    )
    best_csv = os.path.join(OUT_RESULTS_DIR, "top3_by_setting.csv")
    best.to_csv(best_csv, index=False)
    print(f"Wrote top-3 summary to: {best_csv}")


if __name__ == "__main__":
    main()
