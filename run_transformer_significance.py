from __future__ import annotations

import itertools
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

RESULTS_DIR = Path("results")
OUT_DIR = RESULTS_DIR

MODEL_FILES = {
    "BERT": "bert-base-uncased_all_results.csv",
    "RoBERTa": "roberta-base_all_results.csv",
    "DistilBERT": "distilbert-base-uncased_all_results.csv",
    "ALBERT": "albert-base-v2_all_results.csv",
}

MODEL_ORDER = ["BERT", "RoBERTa", "DistilBERT", "ALBERT"]

SETTINGS = [
    ("intra", "PURE", "PURE", "Intra PURE"),
    ("intra", "UserStories", "UserStories", "Intra User Stories"),
    ("cross", "PURE", "UserStories", "PURE to User Stories"),
    ("cross", "UserStories", "PURE", "User Stories to PURE"),
]

ALPHA = 0.05
METRIC = "f1"


def find_file(filename: str) -> Path:
    candidates = [
        Path(filename),
        RESULTS_DIR / filename,
        Path("/content") / filename,
        Path("/content/results") / filename,
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(
        f"Could not find {filename}. Put it either in the current folder or in results/."
    )


def load_model_results() -> dict[str, pd.DataFrame]:
    dfs = {}
    for model_name, filename in MODEL_FILES.items():
        path = find_file(filename)
        df = pd.read_csv(path)

        required = {"setting", "train_dataset", "test_dataset", "seed", METRIC}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"{path} is missing columns: {missing}")

        dfs[model_name] = df
        print(f"[OK] Loaded {model_name}: {path} ({len(df)} rows)")
    return dfs


def get_metric_vector(
    df: pd.DataFrame,
    setting: str,
    train_dataset: str,
    test_dataset: str,
    metric: str = METRIC,
) -> pd.Series:
    sub = df[
        (df["setting"] == setting)
        & (df["train_dataset"] == train_dataset)
        & (df["test_dataset"] == test_dataset)
    ][["seed", metric]].copy()

    sub = sub.dropna()
    sub = sub.sort_values("seed")
    return sub.set_index("seed")[metric]


def safe_wilcoxon(x: pd.Series, y: pd.Series) -> tuple[float | None, float | None, int, str]:
    paired = pd.concat([x.rename("x"), y.rename("y")], axis=1).dropna()
    n = len(paired)

    if n < 2:
        return None, None, n, "insufficient_pairs"

    diff = paired["x"] - paired["y"]

    if np.allclose(diff.values, 0.0):
        return 0.0, 1.0, n, "identical_values"

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            stat, pvalue = wilcoxon(paired["x"], paired["y"], zero_method="wilcox")
        return float(stat), float(pvalue), n, "ok"
    except ValueError as e:
        return None, None, n, f"error: {e}"


def run_tests(dfs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []

    for setting, train_dataset, test_dataset, setting_label in SETTINGS:
        vectors = {
            model: get_metric_vector(df, setting, train_dataset, test_dataset)
            for model, df in dfs.items()
        }

        for model_a, model_b in itertools.combinations(MODEL_ORDER, 2):
            x = vectors[model_a]
            y = vectors[model_b]

            stat, pvalue, n_pairs, status = safe_wilcoxon(x, y)

            paired = pd.concat([x.rename("a"), y.rename("b")], axis=1).dropna()
            mean_a = paired["a"].mean() if not paired.empty else np.nan
            mean_b = paired["b"].mean() if not paired.empty else np.nan
            diff = mean_a - mean_b if not paired.empty else np.nan

            rows.append(
                {
                    "setting": setting,
                    "train_dataset": train_dataset,
                    "test_dataset": test_dataset,
                    "setting_label": setting_label,
                    "metric": METRIC,
                    "model_a": model_a,
                    "model_b": model_b,
                    "n_pairs": n_pairs,
                    "mean_a": mean_a,
                    "mean_b": mean_b,
                    "mean_diff_a_minus_b": diff,
                    "wilcoxon_statistic": stat,
                    "pvalue": pvalue,
                    "significant_0_05": bool(pvalue < ALPHA) if pvalue is not None else False,
                    "status": status,
                }
            )

    return pd.DataFrame(rows)


def fmt(x: float | None) -> str:
    if x is None or pd.isna(x):
        return "--"
    if x < 0.001:
        return "<0.001"
    return f"{x:.3f}"


def significance_text(pvalue: float | None) -> str:
    if pvalue is None or pd.isna(pvalue):
        return "No"
    return "Yes" if pvalue < ALPHA else "No"


def make_latex_table(df: pd.DataFrame) -> str:
    lines = []
    lines.append(r"\begin{table}[h]")
    lines.append(r"\centering")
    lines.append(r"\begin{tabular}{llccc}")
    lines.append(r"\hline")
    lines.append(r"Setting & Comparison & Mean F1 A/B & p-value & Significant \\")
    lines.append(r"\hline")

    for setting_label, sub in df.groupby("setting_label", sort=False):
        lines.append(rf"\multicolumn{{5}}{{l}}{{\textbf{{{setting_label}}}}} \\")
        for _, row in sub.iterrows():
            comparison = f"{row['model_a']} vs. {row['model_b']}"
            means = f"{row['mean_a']:.3f}/{row['mean_b']:.3f}"
            p = fmt(row["pvalue"])
            sig = significance_text(row["pvalue"])
            lines.append(f"{setting_label} & {comparison} & {means} & {p} & {sig} \\\\")
        lines.append(r"\hline")

    lines.append(r"\end{tabular}")
    lines.append(r"\caption{Wilcoxon signed-rank test results for F1-score across encoder-based transformer models.}")
    lines.append(r"\label{tab:transformer_wilcoxon_f1}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def make_latex_table_per_setting(df: pd.DataFrame) -> str:
    blocks = []
    for setting_label, sub in df.groupby("setting_label", sort=False):
        safe_label = (
            setting_label.lower()
            .replace(" ", "_")
            .replace("-", "_")
        )

        lines = []
        lines.append(r"\begin{table}[h]")
        lines.append(r"\centering")
        lines.append(r"\begin{tabular}{lccc}")
        lines.append(r"\hline")
        lines.append(r"Comparison & Mean F1 A/B & p-value & Significant \\")
        lines.append(r"\hline")

        for _, row in sub.iterrows():
            comparison = f"{row['model_a']} vs. {row['model_b']}"
            means = f"{row['mean_a']:.3f}/{row['mean_b']:.3f}"
            p = fmt(row["pvalue"])
            sig = significance_text(row["pvalue"])
            lines.append(f"{comparison} & {means} & {p} & {sig} \\\\")

        lines.append(r"\hline")
        lines.append(r"\end{tabular}")
        lines.append(rf"\caption{{Wilcoxon signed-rank test results for F1-score in the {setting_label} setting.}}")
        lines.append(rf"\label{{tab:wilcoxon_f1_{safe_label}}}")
        lines.append(r"\end{table}")
        blocks.append("\n".join(lines))

    return "\n\n".join(blocks)


def make_paper_text() -> str:
    return r"""
\subsection{Statistical Significance Analysis}

To assess whether the observed differences among encoder-based transformer models are statistically meaningful, a Wilcoxon signed-rank test was applied to F1-scores across the five random seeds. The test was conducted separately for each evaluation setting and for pairwise comparisons among BERT, RoBERTa, DistilBERT, and ALBERT. T5 was excluded from this test because it follows a different text-to-text formulation and is used as a generative baseline rather than as a directly comparable encoder-based classifier.

The statistical results are reported in Table~\ref{tab:transformer_wilcoxon_f1}. Because only five paired runs are available per comparison, the results should be interpreted cautiously. The tests are used as supporting evidence rather than as the sole basis for ranking models. Overall, the analysis helps distinguish robust performance differences from small variations caused by random initialization and training dynamics.
""".strip()


def main():
    OUT_DIR.mkdir(exist_ok=True)

    dfs = load_model_results()
    tests = run_tests(dfs)

    csv_path = OUT_DIR / "transformer_wilcoxon_f1_tests.csv"
    tests.to_csv(csv_path, index=False)

    latex_combined = make_latex_table(tests)
    latex_combined_path = OUT_DIR / "transformer_wilcoxon_f1_table.tex"
    latex_combined_path.write_text(latex_combined, encoding="utf-8")

    latex_separate = make_latex_table_per_setting(tests)
    latex_separate_path = OUT_DIR / "transformer_wilcoxon_f1_tables_by_setting.tex"
    latex_separate_path.write_text(latex_separate, encoding="utf-8")

    paper_text = make_paper_text()
    paper_text_path = OUT_DIR / "transformer_wilcoxon_f1_paper_text.tex"
    paper_text_path.write_text(paper_text, encoding="utf-8")

    print("\nSaved:")
    print(f"  {csv_path}")
    print(f"  {latex_combined_path}")
    print(f"  {latex_separate_path}")
    print(f"  {paper_text_path}")

    print("\nPreview:")
    preview_cols = [
        "setting_label",
        "model_a",
        "model_b",
        "n_pairs",
        "mean_a",
        "mean_b",
        "pvalue",
        "significant_0_05",
        "status",
    ]
    print(tests[preview_cols].to_string(index=False))


if __name__ == "__main__":
    main()
