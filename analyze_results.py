from __future__ import annotations

import os
from typing import List

import pandas as pd

from scipy.stats import ttest_rel

from experiment_utils import (
    RESULTS_DIR,
    FIGURES_DIR,
    ensure_dirs,
    plot_metric_bars,
    plot_reason_heatmap,
)

CLASSICAL_RESULTS = os.path.join(RESULTS_DIR, "classical_multiseed_results.csv")
CLASSICAL_SUMMARY = os.path.join(RESULTS_DIR, "classical_multiseed_summary.csv")
CLASSICAL_REASON = os.path.join(RESULTS_DIR, "classical_reason_performance.csv")
CLASSICAL_VARIANTS = os.path.join(RESULTS_DIR, "classical_multiseed_summary_by_variant.csv")

# Include only models actually run / discussed
MODEL_SUMMARY_FILES = {
    "distilbert-base-uncased": "distilbert-base-uncased_summary.csv",
    "roberta-base": "roberta-base_summary.csv",
    "bert-base-uncased": "bert-base-uncased_summary.csv",
    "albert-base-v2": "albert-base-v2_summary.csv",
    "t5-small": "t5-small_summary.csv",
    "google/flan-t5-small": "google_flan-t5-small_summary.csv",
}

MODEL_REASON_FILES = {
    "distilbert-base-uncased": "distilbert-base-uncased_reason_performance.csv",
    "roberta-base": "roberta-base_reason_performance.csv",
    "bert-base-uncased": "bert-base-uncased_reason_performance.csv",
    "albert-base-v2": "albert-base-v2_reason_performance.csv",
    "t5-small": "t5-small_reason_performance.csv",
    "google/flan-t5-small": "google_flan-t5-small_reason_performance.csv",
}

MODEL_ORDER = [
    "MultinomialNB",
    "LinearSVM",
    "RandomForest",
    "LightGBM",
    "XGBoost",
    "distilbert-base-uncased",
    "bert-base-uncased",
    "roberta-base",
    "albert-base-v2",
    "t5-small",
    "google/flan-t5-small",
]

DISPLAY_NAME = {
    "MultinomialNB": "NB",
    "LinearSVM": "Linear SVM",
    "RandomForest": "Random Forest",
    "LightGBM": "LightGBM",
    "XGBoost": "XGBoost",
    "distilbert-base-uncased": "DistilBERT",
    "bert-base-uncased": "BERT-base",
    "roberta-base": "RoBERTa-base",
    "albert-base-v2": "ALBERT-base",
    "t5-small": "T5-small",
    "google/flan-t5-small": "FLAN-T5-small",
}


def load_transformer_summaries() -> List[pd.DataFrame]:
    dfs = []
    for model_name, filename in MODEL_SUMMARY_FILES.items():
        path = os.path.join(RESULTS_DIR, filename)
        if os.path.exists(path):
            df = pd.read_csv(path)
            dfs.append(df)
    return dfs


def load_transformer_reason_perf() -> List[pd.DataFrame]:
    dfs = []
    for model_name, filename in MODEL_REASON_FILES.items():
        path = os.path.join(RESULTS_DIR, filename)
        if os.path.exists(path):
            df = pd.read_csv(path)
            dfs.append(df)
    return dfs


def paired_tests(df: pd.DataFrame, metric: str = "f1") -> pd.DataFrame:
    rows = []

    for (setting, train_dataset, test_dataset), sub in df.groupby(["setting", "train_dataset", "test_dataset"]):
        pivot = sub.pivot_table(index="seed", columns="model", values=metric, aggfunc="mean")
        models = list(pivot.columns)

        for i in range(len(models)):
            for j in range(i + 1, len(models)):
                a = models[i]
                b = models[j]
                paired = pivot[[a, b]].dropna()
                if len(paired) < 2:
                    continue

                stat, pvalue = ttest_rel(paired[a], paired[b])
                rows.append({
                    "setting": setting,
                    "train_dataset": train_dataset,
                    "test_dataset": test_dataset,
                    "metric": metric,
                    "model_a": a,
                    "model_b": b,
                    "n_pairs": len(paired),
                    "mean_a": paired[a].mean(),
                    "mean_b": paired[b].mean(),
                    "pvalue": pvalue,
                })

    return pd.DataFrame(rows)


def build_paper_summary(classical_summary: pd.DataFrame, transformer_summaries: List[pd.DataFrame]) -> pd.DataFrame:
    parts = [classical_summary.copy()]
    for df in transformer_summaries:
        parts.append(df.copy())

    paper_df = pd.concat(parts, ignore_index=True)

    paper_df["model_order"] = paper_df["model"].map({m: i for i, m in enumerate(MODEL_ORDER)})
    paper_df["model_display"] = paper_df["model"].map(DISPLAY_NAME).fillna(paper_df["model"])

    ordered_cols = [
        "setting",
        "train_dataset",
        "test_dataset",
        "model",
        "model_display",
        "accuracy_mean",
        "accuracy_std",
        "balanced_accuracy_mean",
        "balanced_accuracy_std",
        "precision_mean",
        "precision_std",
        "recall_mean",
        "recall_std",
        "f1_mean",
        "f1_std",
        "mcc_mean",
        "mcc_std",
        "cohen_kappa_mean",
        "cohen_kappa_std",
    ]
    existing = [c for c in ordered_cols if c in paper_df.columns]
    paper_df = paper_df[existing + [c for c in ["model_order"] if c not in existing]]
    paper_df = paper_df.sort_values(
        ["setting", "train_dataset", "test_dataset", "f1_mean"],
        ascending=[True, True, True, False],
    )
    return paper_df


def build_best_model_table(paper_summary: pd.DataFrame) -> pd.DataFrame:
    best = (
        paper_summary
        .sort_values(["setting", "train_dataset", "test_dataset", "f1_mean"], ascending=[True, True, True, False])
        .groupby(["setting", "train_dataset", "test_dataset"], as_index=False)
        .head(3)
        .copy()
    )
    return best


def save_filtered_tables(paper_summary: pd.DataFrame) -> None:
    # Main transformer-only comparison table
    transformer_rows = paper_summary[paper_summary["model"].isin(MODEL_SUMMARY_FILES.keys())].copy()
    transformer_rows.to_csv(os.path.join(RESULTS_DIR, "transformer_summary_table.csv"), index=False)

    # Recommended final paper table: exclude FLAN-T5 if it collapsed
    final_models = [
        "distilbert-base-uncased",
        "bert-base-uncased",
        "roberta-base",
        "albert-base-v2",
        "t5-small",
    ]
    final_transformer_rows = paper_summary[paper_summary["model"].isin(final_models)].copy()
    final_transformer_rows.to_csv(os.path.join(RESULTS_DIR, "final_transformer_table.csv"), index=False)

    # Classical vs best transformers
    comparison_models = [
        "LinearSVM",
        "LightGBM",
        "XGBoost",
        "distilbert-base-uncased",
        "bert-base-uncased",
        "roberta-base",
        "albert-base-v2",
        "t5-small",
    ]
    compact = paper_summary[paper_summary["model"].isin(comparison_models)].copy()
    compact.to_csv(os.path.join(RESULTS_DIR, "compact_comparison_table.csv"), index=False)


def make_metric_figures(paper_summary: pd.DataFrame) -> None:
    for (setting, train_dataset, test_dataset), sub in paper_summary.groupby(["setting", "train_dataset", "test_dataset"]):
        sub = sub.copy()
        sub["model"] = sub["model_display"]

        plot_metric_bars(
            sub,
            metric="f1_mean",
            error_col="f1_std",
            title=f"{setting} | {train_dataset} -> {test_dataset} | F1",
            out_path=os.path.join(FIGURES_DIR, f"bar_{setting}_{train_dataset}_to_{test_dataset}_f1.png"),
        )

        plot_metric_bars(
            sub,
            metric="mcc_mean",
            error_col="mcc_std",
            title=f"{setting} | {train_dataset} -> {test_dataset} | MCC",
            out_path=os.path.join(FIGURES_DIR, f"bar_{setting}_{train_dataset}_to_{test_dataset}_mcc.png"),
        )

        plot_metric_bars(
            sub,
            metric="cohen_kappa_mean",
            error_col="cohen_kappa_std",
            title=f"{setting} | {train_dataset} -> {test_dataset} | Cohen Kappa",
            out_path=os.path.join(FIGURES_DIR, f"bar_{setting}_{train_dataset}_to_{test_dataset}_kappa.png"),
        )


def make_reason_heatmaps() -> None:
    # Classical
    if os.path.exists(CLASSICAL_REASON):
        classical_reason = pd.read_csv(CLASSICAL_REASON)
        classical_reason["model"] = classical_reason["model"].map(DISPLAY_NAME).fillna(classical_reason["model"])
        for (setting, train_dataset, test_dataset), sub in classical_reason.groupby(["setting", "train_dataset", "test_dataset"]):
            plot_reason_heatmap(
                sub,
                title=f"Reason-level F1 | classical | {setting} | {train_dataset}->{test_dataset}",
                out_path=os.path.join(FIGURES_DIR, f"heatmap_classical_{setting}_{train_dataset}_to_{test_dataset}.png"),
                value_col="f1",
            )

    # Transformers
    reason_dfs = load_transformer_reason_perf()
    if reason_dfs:
        all_reason = pd.concat(reason_dfs, ignore_index=True)
        all_reason["model"] = all_reason["model"].map(DISPLAY_NAME).fillna(all_reason["model"])
        for (setting, train_dataset, test_dataset), sub in all_reason.groupby(["setting", "train_dataset", "test_dataset"]):
            plot_reason_heatmap(
                sub,
                title=f"Reason-level F1 | transformers | {setting} | {train_dataset}->{test_dataset}",
                out_path=os.path.join(FIGURES_DIR, f"heatmap_transformers_{setting}_{train_dataset}_to_{test_dataset}.png"),
                value_col="f1",
            )


def save_ablation_table() -> None:
    if not os.path.exists(CLASSICAL_VARIANTS):
        return

    df = pd.read_csv(CLASSICAL_VARIANTS)
    df["model_display"] = df["model"].map(DISPLAY_NAME).fillna(df["model"])
    df.to_csv(os.path.join(RESULTS_DIR, "classical_ablation_table.csv"), index=False)

    # Compact ablation on top classical models
    keep = ["LinearSVM", "LightGBM", "XGBoost"]
    compact = df[df["model"].isin(keep)].copy()
    compact["model_display"] = compact["model"].map(DISPLAY_NAME).fillna(compact["model"])
    compact.to_csv(os.path.join(RESULTS_DIR, "classical_ablation_compact.csv"), index=False)


def main():
    ensure_dirs()

    if not os.path.exists(CLASSICAL_RESULTS):
        raise FileNotFoundError(f"Missing {CLASSICAL_RESULTS}")
    if not os.path.exists(CLASSICAL_SUMMARY):
        raise FileNotFoundError(f"Missing {CLASSICAL_SUMMARY}")

    classical_results = pd.read_csv(CLASSICAL_RESULTS)
    classical_summary = pd.read_csv(CLASSICAL_SUMMARY)

    transformer_summaries = load_transformer_summaries()

    # Classical significance tests
    significance_df = paired_tests(
        classical_results[classical_results["variant"] == "original"].copy(),
        metric="f1",
    )
    significance_path = os.path.join(RESULTS_DIR, "classical_significance_tests.csv")
    significance_df.to_csv(significance_path, index=False)

    # Unified summary
    paper_summary = build_paper_summary(classical_summary, transformer_summaries)
    paper_summary_path = os.path.join(RESULTS_DIR, "paper_summary_table.csv")
    paper_summary.to_csv(paper_summary_path, index=False)

    # Best-per-setting table
    best_models = build_best_model_table(paper_summary)
    best_models_path = os.path.join(RESULTS_DIR, "best_models_by_setting.csv")
    best_models.to_csv(best_models_path, index=False)

    # Extra filtered tables
    save_filtered_tables(paper_summary)

    # Figures
    make_metric_figures(paper_summary)
    make_reason_heatmaps()

    # Classical ablation export
    save_ablation_table()

    print("Saved:")
    print(f"  {significance_path}")
    print(f"  {paper_summary_path}")
    print(f"  {best_models_path}")
    print(f"  {os.path.join(RESULTS_DIR, 'transformer_summary_table.csv')}")
    print(f"  {os.path.join(RESULTS_DIR, 'final_transformer_table.csv')}")
    print(f"  {os.path.join(RESULTS_DIR, 'compact_comparison_table.csv')}")
    if os.path.exists(CLASSICAL_VARIANTS):
        print(f"  {os.path.join(RESULTS_DIR, 'classical_ablation_table.csv')}")
        print(f"  {os.path.join(RESULTS_DIR, 'classical_ablation_compact.csv')}")
    print("  figures/* metric bars and heatmaps")


if __name__ == "__main__":
    main()
