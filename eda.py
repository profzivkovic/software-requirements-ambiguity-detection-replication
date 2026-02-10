"""
EDA script for labeled requirements datasets (PURE + User Stories)
with automatic figure saving.

What it does:
- Loads:
    data/pure_labeled.csv
    data/userstories_labeled.csv
- Prints dataset summaries
- Generates and SAVES plots (PNG, 300 dpi) to:
    figures/
  Plots:
    1) Class distribution (counts + %)
    2) Word length distributions by class
    3) Character length distributions by class
    4) Top ambiguity trigger reasons (bar plot)
- Writes an EDA summary CSV:
    eda_summary.csv

"""

from __future__ import annotations

import os
from collections import Counter
from typing import Dict, Tuple

import pandas as pd
import matplotlib.pyplot as plt


# ----------------------------
# Configuration
# ----------------------------

PURE_PATH = os.path.join("data", "pure_labeled.csv")
US_PATH = os.path.join("data", "userstories_labeled.csv")

FIG_DIR = "figures"
OUTPUT_SUMMARY_CSV = "eda_summary.csv"

TOP_N_REASONS = 12
BINS = 30
DPI = 300

# If True, also display plots interactively (useful in PyCharm).
# If False, plots will only be saved.
SHOW_PLOTS = True


# ----------------------------
# Helpers
# ----------------------------

def ensure_dirs() -> None:
    os.makedirs(FIG_DIR, exist_ok=True)


def assert_required_columns(df: pd.DataFrame, dataset_name: str) -> None:
    required = {"requirement_text", "label", "reasons"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"{dataset_name} is missing columns: {sorted(missing)}. "
            f"Found columns: {list(df.columns)}"
        )


def load_dataset(path: str, name: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Could not find {name} file at: {path}\n"
            f"Tip: Create a data/ folder and place the CSV there."
        )
    df = pd.read_csv(path)
    assert_required_columns(df, name)

    # normalize types
    df["requirement_text"] = df["requirement_text"].astype(str)
    df["label"] = pd.to_numeric(df["label"], errors="coerce").fillna(0).astype(int)
    df["reasons"] = df["reasons"].fillna("").astype(str)
    return df


def add_length_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["char_len"] = df["requirement_text"].str.len()
    df["word_len"] = df["requirement_text"].str.split().str.len()
    return df


def class_counts_and_props(df: pd.DataFrame) -> Tuple[Dict[int, int], Dict[int, float]]:
    counts = df["label"].value_counts().to_dict()
    total = len(df)
    props = {k: (v / total if total else 0.0) for k, v in counts.items()}
    return counts, props


def print_basic_stats(df: pd.DataFrame, name: str) -> None:
    counts, props = class_counts_and_props(df)
    total = len(df)
    clear = counts.get(0, 0)
    amb = counts.get(1, 0)
    print(f"\n=== {name} ===")
    print(f"Rows: {total}")
    print(f"Clear (0): {clear} ({props.get(0, 0.0):.2%})")
    print(f"Ambiguous (1): {amb} ({props.get(1, 0.0):.2%})")


def save_and_maybe_show(fig_filename: str) -> None:
    out_path = os.path.join(FIG_DIR, fig_filename)
    plt.savefig(out_path, dpi=DPI, bbox_inches="tight")
    if SHOW_PLOTS:
        plt.show()
    plt.close()


# ----------------------------
# Plotting
# ----------------------------

def plot_class_distribution(df: pd.DataFrame, title: str, fig_filename: str) -> None:
    counts, props = class_counts_and_props(df)
    values = [counts.get(0, 0), counts.get(1, 0)]
    labels = [
        f"Clear (0)\n{props.get(0, 0.0):.1%}",
        f"Ambiguous (1)\n{props.get(1, 0.0):.1%}"
    ]

    plt.figure()
    plt.bar(labels, values)
    plt.title(title)
    plt.ylabel("Count")

    for i, v in enumerate(values):
        plt.text(i, v, str(v), ha="center", va="bottom")

    plt.tight_layout()
    save_and_maybe_show(fig_filename)


def plot_hist_by_class(df: pd.DataFrame, col: str, title: str, fig_filename: str, bins: int = BINS) -> None:
    plt.figure()
    df[df["label"] == 0][col].hist(alpha=0.6, bins=bins)
    df[df["label"] == 1][col].hist(alpha=0.6, bins=bins)
    plt.title(title)
    plt.xlabel(col)
    plt.ylabel("Frequency")
    plt.legend(["Clear (0)", "Ambiguous (1)"])
    plt.tight_layout()
    save_and_maybe_show(fig_filename)


def reasons_counter(df: pd.DataFrame) -> Counter:
    counter = Counter()
    for r in df["reasons"].fillna("").astype(str):
        r = r.strip()
        if not r:
            continue
        for item in r.split(","):
            item = item.strip()
            if item:
                counter[item] += 1
    return counter


def plot_top_reasons(df: pd.DataFrame, title: str, fig_filename: str, top_n: int = TOP_N_REASONS) -> None:
    c = reasons_counter(df)
    top = c.most_common(top_n)
    if not top:
        print(f"[{title}] No reasons found to plot.")
        return

    labels = [k for k, _ in top]
    values = [v for _, v in top]

    plt.figure()
    plt.bar(labels, values)
    plt.title(title)
    plt.ylabel("Count")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    save_and_maybe_show(fig_filename)


# ----------------------------
# Summary
# ----------------------------

def build_eda_summary(df: pd.DataFrame, name: str) -> Dict[str, object]:
    counts, props = class_counts_and_props(df)
    total = len(df)
    clear = counts.get(0, 0)
    amb = counts.get(1, 0)
    return {
        "dataset": name,
        "total": int(total),
        "clear_0": int(clear),
        "ambiguous_1": int(amb),
        "pct_clear": float(props.get(0, 0.0)),
        "pct_ambiguous": float(props.get(1, 0.0)),
        "avg_words": float(df["requirement_text"].str.split().str.len().mean()),
        "avg_chars": float(df["requirement_text"].str.len().mean()),
    }


# ----------------------------
# Main
# ----------------------------

def main() -> None:
    ensure_dirs()

    # Load datasets
    df_pure = load_dataset(PURE_PATH, "PURE")
    df_us = load_dataset(US_PATH, "UserStories")

    # Print stats
    print_basic_stats(df_pure, "PURE")
    print_basic_stats(df_us, "UserStories")

    # Add length features
    df_pure_l = add_length_features(df_pure)
    df_us_l = add_length_features(df_us)

    # 1) Class distributions
    plot_class_distribution(df_pure, "PURE: Class Distribution", "pure_class_distribution.png")
    plot_class_distribution(df_us, "User Stories: Class Distribution", "userstories_class_distribution.png")

    # 2) Word length distributions by class
    plot_hist_by_class(df_pure_l, "word_len", "PURE: Word Length by Class", "pure_word_length_by_class.png")
    plot_hist_by_class(df_us_l, "word_len", "User Stories: Word Length by Class", "userstories_word_length_by_class.png")

    # 3) Character length distributions by class
    plot_hist_by_class(df_pure_l, "char_len", "PURE: Character Length by Class", "pure_char_length_by_class.png")
    plot_hist_by_class(df_us_l, "char_len", "User Stories: Character Length by Class", "userstories_char_length_by_class.png")

    # 4) Top ambiguity triggers ("reasons")
    plot_top_reasons(df_pure, "PURE: Top Ambiguity Triggers", "pure_top_reasons.png")
    plot_top_reasons(df_us, "User Stories: Top Ambiguity Triggers", "userstories_top_reasons.png")

    # Save summary CSV
    summary = pd.DataFrame([
        build_eda_summary(df_pure, "PURE"),
        build_eda_summary(df_us, "UserStories"),
    ])
    summary.to_csv(OUTPUT_SUMMARY_CSV, index=False)
    print(f"\nWrote EDA summary to: {OUTPUT_SUMMARY_CSV}")
    print(f"Saved figures to: {FIG_DIR}/")
    print(summary)


if __name__ == "__main__":
    main()
