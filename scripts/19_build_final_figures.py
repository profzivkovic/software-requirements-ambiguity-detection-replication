from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix


# =============================================================================
# Paths
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data" / "processed"
RAW_DIR = PROJECT_ROOT / "results" / "raw"
SUMMARY_DIR = PROJECT_ROOT / "results" / "summary"
TABLE_DIR = SUMMARY_DIR / "final_tables"

TRANSFORMER_RUN_DIR = (
    RAW_DIR / "transformer_runs"
)

FIGURE_DIR = (
    PROJECT_ROOT / "figures" / "final"
)

PURE_PATH = (
    DATA_DIR / "pure_clean.csv"
)

USERSTORIES_PATH = (
    DATA_DIR / "userstories_clean.csv"
)


# =============================================================================
# Experiment constants
# =============================================================================

SEEDS = [
    13,
    21,
    42,
    87,
    123,
]

BLUE = "#1f77b4"
ORANGE = "#ff7f0e"
GREEN = "#2ca02c"
RED = "#d62728"
PURPLE = "#9467bd"


CLASSICAL_DISPLAY_ORDER = [
    "Naive Bayes",
    "Linear SVM",
    "Random Forest",
    "LightGBM",
    "XGBoost",
]

TRANSFORMER_DISPLAY_ORDER = [
    "DistilBERT",
    "BERT",
    "RoBERTa",
    "ALBERT",
    "T5",
]


CLASSICAL_HEATMAP_ORDER = [
    "LightGBM",
    "Linear SVM",
    "Naive Bayes",
    "Random Forest",
    "XGBoost",
]

TRANSFORMER_HEATMAP_ORDER = [
    "ALBERT",
    "BERT",
    "DistilBERT",
    "RoBERTa",
    "T5",
]


DATASET_DISPLAY = {
    "PURE": "PURE",
    "UserStories": "User Stories",
}


VARIANT_ORDER = [
    "original",
    "masked",
    "removed",
]

VARIANT_DISPLAY = {
    "original": "Original",
    "masked": "Masked cues",
    "removed": "Removed cues",
}

VARIANT_COLORS = {
    "original": BLUE,
    "masked": ORANGE,
    "removed": GREEN,
}


# =============================================================================
# Locked v3 sanity checks
# =============================================================================

EXPECTED_CLASS_COUNTS = {
    "PURE": {
        0: 3203,
        1: 3331,
    },
    "UserStories": {
        0: 743,
        1: 930,
    },
}


EXPECTED_REASON_SUPPORT = {
    "PURE": {
        "weak_modal": 1963,
        "vague_term": 1297,
        "missing_actor": 418,
        "multiple_actions": 303,
        "undefined_quantity": 171,
        "undefined_term": 130,
        "vague_condition": 22,
        "incomplete": 4,
    },
    "UserStories": {
        "weak_modal": 785,
        "vague_term": 244,
        "undefined_quantity": 50,
        "undefined_term": 31,
        "multiple_actions": 26,
        "vague_condition": 2,
    },
}


# =============================================================================
# Matplotlib setup
# =============================================================================

plt.style.use("default")

plt.rcParams.update(
    {
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.titlesize": 14,
        "axes.grid": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    }
)


manifest_rows: list[dict[str, str]] = []


# =============================================================================
# General helpers
# =============================================================================

def require_file(
    path: Path,
) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing required file: {path}"
        )


def read_csv(
    path: Path,
) -> pd.DataFrame:
    require_file(path)
    return pd.read_csv(path)


def assert_no_flan(
    df: pd.DataFrame,
    source_name: str,
) -> None:
    for column in df.columns:
        if df[column].dtype != object:
            continue

        contains_flan = (
            df[column]
            .astype(str)
            .str.lower()
            .str.contains(
                "flan",
                regex=False,
            )
            .any()
        )

        if contains_flan:
            raise AssertionError(
                f"FLAN reference found in "
                f"{source_name}, column={column}"
            )


def clean_output_directory() -> None:
    FIGURE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Preserve Figure 01 because it is the conceptual
    # screening-scenario diagram and is not regenerated
    # by this script.
    for pattern in [
        "figure_*.png",
        "figure_*.pdf",
    ]:
        for path in FIGURE_DIR.glob(
            pattern
        ):
            if path.name.startswith(
                "figure_01_"
            ):
                continue

            path.unlink()

    manifest_path = (
        FIGURE_DIR
        / "figure_manifest.csv"
    )

    if manifest_path.exists():
        manifest_path.unlink()


def save_figure(
    fig: plt.Figure,
    number: str,
    slug: str,
    scope: str,
    description: str,
) -> None:
    FIGURE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    base_name = (
        f"figure_{number}_{slug}"
    )

    png_path = (
        FIGURE_DIR
        / f"{base_name}.png"
    )

    pdf_path = (
        FIGURE_DIR
        / f"{base_name}.pdf"
    )

    fig.savefig(
        png_path,
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )

    fig.savefig(
        pdf_path,
        bbox_inches="tight",
        facecolor="white",
    )

    plt.close(fig)

    print(
        f"Saved {png_path}"
    )

    print(
        f"Saved {pdf_path}"
    )

    manifest_rows.append(
        {
            "figure": number,
            "slug": slug,
            "scope": scope,
            "png": png_path.name,
            "pdf": pdf_path.name,
            "description": description,
        }
    )


def remove_top_right_spines(
    ax: plt.Axes,
) -> None:
    ax.spines[
        "top"
    ].set_visible(False)

    ax.spines[
        "right"
    ].set_visible(False)


def format_reason(
    reason: str,
) -> str:
    return (
        str(reason)
        .replace(
            "_",
            " ",
        )
    )


def add_bar_value_labels(
    ax: plt.Axes,
    bars,
) -> None:
    for bar in bars:
        value = (
            bar.get_height()
        )

        ax.annotate(
            f"{value:.0f}",
            xy=(
                bar.get_x()
                + bar.get_width() / 2,
                value,
            ),
            xytext=(
                0,
                4,
            ),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
        )


# =============================================================================
# Figure 02: class distribution
# =============================================================================

def validate_class_counts(
    dataset_name: str,
    df: pd.DataFrame,
) -> None:
    for label in [
        0,
        1,
    ]:
        observed = int(
            (
                df["label"]
                == label
            ).sum()
        )

        expected = (
            EXPECTED_CLASS_COUNTS[
                dataset_name
            ][label]
        )

        if observed != expected:
            raise AssertionError(
                f"{dataset_name}: "
                f"label={label}, "
                f"expected {expected}, "
                f"observed {observed}"
            )


def build_class_distribution() -> None:
    pure = read_csv(
        PURE_PATH
    )

    userstories = read_csv(
        USERSTORIES_PATH
    )

    validate_class_counts(
        "PURE",
        pure,
    )

    validate_class_counts(
        "UserStories",
        userstories,
    )

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(
            10.5,
            4.8,
        ),
    )

    for (
        ax,
        title,
        df,
    ) in [
        (
            axes[0],
            "PURE",
            pure,
        ),
        (
            axes[1],
            "User Stories",
            userstories,
        ),
    ]:
        clear_count = int(
            (
                df["label"]
                == 0
            ).sum()
        )

        positive_count = int(
            (
                df["label"]
                == 1
            ).sum()
        )

        total = len(df)

        values = [
            clear_count,
            positive_count,
        ]

        labels = [
            (
                "Clear (0)\n"
                f"{clear_count / total:.1%}"
            ),
            (
                "Indicator-positive (1)\n"
                f"{positive_count / total:.1%}"
            ),
        ]

        bars = ax.bar(
            np.arange(2),
            values,
            color=BLUE,
            width=0.72,
        )

        add_bar_value_labels(
            ax,
            bars,
        )

        ax.set_xticks(
            np.arange(2),
            labels,
        )

        ax.set_ylabel(
            "Count"
        )

        ax.set_title(
            title
        )

        ax.set_ylim(
            0,
            max(values)
            * 1.13,
        )

        remove_top_right_spines(
            ax
        )

    fig.suptitle(
        "Class distribution after cleaning and deduplication"
    )

    fig.tight_layout(
        rect=[
            0,
            0,
            1,
            0.94,
        ]
    )

    save_figure(
        fig=fig,
        number="02",
        slug="class_distribution",
        scope="main",
        description=(
            "Class distribution in the cleaned "
            "and deduplicated v3 corpora."
        ),
    )


# =============================================================================
# Figure 03: reason distribution
# =============================================================================

def parse_reasons(
    value,
) -> list[str]:
    if pd.isna(value):
        return []

    text = str(
        value
    ).strip()

    if not text:
        return []

    return [
        part.strip()
        for part
        in text.split(",")
        if part.strip()
    ]


def reason_counts_from_clean_data(
    df: pd.DataFrame,
) -> pd.DataFrame:
    positive = df[
        df["label"] == 1
    ].copy()

    positive[
        "reason"
    ] = (
        positive[
            "reasons"
        ]
        .apply(
            parse_reasons
        )
    )

    exploded = (
        positive
        .explode(
            "reason"
        )
        .dropna(
            subset=[
                "reason"
            ]
        )
        .copy()
    )

    exploded = exploded[
        exploded[
            "reason"
        ]
        .astype(str)
        .str.strip()
        != ""
    ]

    counts = (
        exploded[
            "reason"
        ]
        .value_counts()
        .rename_axis(
            "reason"
        )
        .reset_index(
            name="support"
        )
    )

    return counts


def validate_reason_counts(
    dataset_name: str,
    counts: pd.DataFrame,
) -> None:
    observed = {
        str(row.reason):
        int(row.support)
        for row
        in counts.itertuples(
            index=False
        )
    }

    expected = (
        EXPECTED_REASON_SUPPORT[
            dataset_name
        ]
    )

    if observed != expected:
        raise AssertionError(
            f"{dataset_name} reason support "
            f"does not match locked v3 values.\n"
            f"Expected: {expected}\n"
            f"Observed: {observed}"
        )


def build_reason_distribution() -> None:
    pure = read_csv(
        PURE_PATH
    )

    userstories = read_csv(
        USERSTORIES_PATH
    )

    pure_counts = (
        reason_counts_from_clean_data(
            pure
        )
    )

    userstories_counts = (
        reason_counts_from_clean_data(
            userstories
        )
    )

    validate_reason_counts(
        "PURE",
        pure_counts,
    )

    validate_reason_counts(
        "UserStories",
        userstories_counts,
    )

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(
            12.0,
            5.3,
        ),
    )

    for (
        ax,
        title,
        sub,
    ) in [
        (
            axes[0],
            "PURE",
            pure_counts,
        ),
        (
            axes[1],
            "User Stories",
            userstories_counts,
        ),
    ]:
        sub = (
            sub
            .sort_values(
                "support",
                ascending=True,
            )
            .copy()
        )

        y = np.arange(
            len(sub)
        )

        bars = ax.barh(
            y,
            sub[
                "support"
            ],
            color=BLUE,
        )

        ax.set_yticks(
            y,
            [
                format_reason(
                    reason
                )
                for reason
                in sub[
                    "reason"
                ]
            ],
        )

        ax.set_xlabel(
            "Annotated instances"
        )

        ax.set_title(
            title
        )

        remove_top_right_spines(
            ax
        )

        max_support = float(
            sub[
                "support"
            ].max()
        )

        for (
            bar,
            value,
        ) in zip(
            bars,
            sub[
                "support"
            ],
        ):
            ax.text(
                float(value)
                + max_support
                * 0.012,
                (
                    bar.get_y()
                    + bar.get_height()
                    / 2
                ),
                str(
                    int(value)
                ),
                va="center",
                fontsize=9,
            )

        ax.set_xlim(
            0,
            max_support
            * 1.15,
        )

    fig.suptitle(
        "Distribution of annotated indicator reasons"
    )

    fig.tight_layout(
        rect=[
            0,
            0,
            1,
            0.94,
        ]
    )

    save_figure(
        fig=fig,
        number="03",
        slug="reason_distribution",
        scope="main",
        description=(
            "Reason support calculated directly "
            "from cleaned and deduplicated v3 data."
        ),
    )


# =============================================================================
# Figures 04–05: length distributions
# =============================================================================

def draw_length_histogram(
    ax: plt.Axes,
    df: pd.DataFrame,
    value_column: str,
    title: str,
    xlabel: str,
) -> None:
    clear_values = (
        df.loc[
            df["label"] == 0,
            value_column,
        ]
        .astype(float)
        .to_numpy()
    )

    positive_values = (
        df.loc[
            df["label"] == 1,
            value_column,
        ]
        .astype(float)
        .to_numpy()
    )

    all_values = np.concatenate(
        [
            clear_values,
            positive_values,
        ]
    )

    if len(
        all_values
    ) == 0:
        raise AssertionError(
            f"No values for {title}"
        )

    # Only the displayed range is limited.
    # No observations are removed from datasets
    # or experiments.
    display_limit = float(
        np.percentile(
            all_values,
            99.5,
        )
    )

    display_limit = max(
        display_limit,
        1.0,
    )

    clear_visible = (
        clear_values[
            clear_values
            <= display_limit
        ]
    )

    positive_visible = (
        positive_values[
            positive_values
            <= display_limit
        ]
    )

    bins = np.linspace(
        0,
        display_limit,
        31,
    )

    ax.hist(
        clear_visible,
        bins=bins,
        alpha=0.55,
        color=BLUE,
        label="Clear (0)",
    )

    ax.hist(
        positive_visible,
        bins=bins,
        alpha=0.55,
        color=ORANGE,
        label=(
            "Indicator-positive (1)"
        ),
    )

    ax.set_xlim(
        0,
        display_limit,
    )

    ax.set_title(
        title
    )

    ax.set_xlabel(
        xlabel
    )

    ax.set_ylabel(
        "Frequency"
    )

    ax.grid(
        True,
        linewidth=0.6,
        alpha=0.30,
    )

    ax.legend(
        frameon=True
    )


def build_word_length_distribution() -> None:
    pure = read_csv(
        PURE_PATH
    )

    userstories = read_csv(
        USERSTORIES_PATH
    )

    for df in [
        pure,
        userstories,
    ]:
        df[
            "word_length"
        ] = (
            df[
                "requirement_text"
            ]
            .astype(str)
            .str.split()
            .str.len()
        )

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(
            12.0,
            5.1,
        ),
    )

    draw_length_histogram(
        axes[0],
        pure,
        "word_length",
        "PURE",
        "Word count",
    )

    draw_length_histogram(
        axes[1],
        userstories,
        "word_length",
        "User Stories",
        "Word count",
    )

    fig.suptitle(
        "Word-length distribution by class"
    )

    fig.tight_layout(
        rect=[
            0,
            0,
            1,
            0.94,
        ]
    )

    save_figure(
        fig=fig,
        number="04",
        slug="word_length_by_class",
        scope="main",
        description=(
            "Word-length distributions. "
            "All observations remain in the "
            "analysis; displayed axes are limited "
            "to the within-dataset pooled "
            "99.5th percentile."
        ),
    )


def build_character_length_distribution() -> None:
    pure = read_csv(
        PURE_PATH
    )

    userstories = read_csv(
        USERSTORIES_PATH
    )

    for df in [
        pure,
        userstories,
    ]:
        df[
            "character_length"
        ] = (
            df[
                "requirement_text"
            ]
            .astype(str)
            .str.len()
        )

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(
            12.0,
            5.1,
        ),
    )

    draw_length_histogram(
        axes[0],
        pure,
        "character_length",
        "PURE",
        "Character count",
    )

    draw_length_histogram(
        axes[1],
        userstories,
        "character_length",
        "User Stories",
        "Character count",
    )

    fig.suptitle(
        "Character-length distribution by class"
    )

    fig.tight_layout(
        rect=[
            0,
            0,
            1,
            0.94,
        ]
    )

    save_figure(
        fig=fig,
        number="05",
        slug=(
            "character_length_by_class"
        ),
        scope="main",
        description=(
            "Character-length distributions. "
            "All observations remain in the "
            "analysis; displayed axes are limited "
            "to the within-dataset pooled "
            "99.5th percentile."
        ),
    )


# =============================================================================
# Figures 06–13: confusion matrices
# =============================================================================

CONFUSION_CONFIGS = [
    {
        "number": "06",
        "slug": (
            "confusion_intra_"
            "classical_pure"
        ),
        "family": "classical",
        "setting": "intra",
        "source": "PURE",
        "target": "PURE",
        "summary_file": (
            "table_03_classical_"
            "intra_pure_numeric.csv"
        ),
    },
    {
        "number": "07",
        "slug": (
            "confusion_intra_"
            "classical_userstories"
        ),
        "family": "classical",
        "setting": "intra",
        "source": "UserStories",
        "target": "UserStories",
        "summary_file": (
            "table_04_classical_"
            "intra_userstories_numeric.csv"
        ),
    },
    {
        "number": "08",
        "slug": (
            "confusion_intra_"
            "transformer_pure"
        ),
        "family": "transformer",
        "setting": "intra",
        "source": "PURE",
        "target": "PURE",
        "summary_file": (
            "table_05_transformer_"
            "intra_pure_numeric.csv"
        ),
    },
    {
        "number": "09",
        "slug": (
            "confusion_intra_"
            "transformer_userstories"
        ),
        "family": "transformer",
        "setting": "intra",
        "source": "UserStories",
        "target": "UserStories",
        "summary_file": (
            "table_06_transformer_"
            "intra_userstories_numeric.csv"
        ),
    },
    {
        "number": "10",
        "slug": (
            "confusion_cross_"
            "classical_pure_to_userstories"
        ),
        "family": "classical",
        "setting": "cross",
        "source": "PURE",
        "target": "UserStories",
        "summary_file": (
            "table_07_classical_cross_"
            "pure_to_userstories_numeric.csv"
        ),
    },
    {
        "number": "11",
        "slug": (
            "confusion_cross_"
            "classical_userstories_to_pure"
        ),
        "family": "classical",
        "setting": "cross",
        "source": "UserStories",
        "target": "PURE",
        "summary_file": (
            "table_08_classical_cross_"
            "userstories_to_pure_numeric.csv"
        ),
    },
    {
        "number": "12",
        "slug": (
            "confusion_cross_"
            "transformer_pure_to_userstories"
        ),
        "family": "transformer",
        "setting": "cross",
        "source": "PURE",
        "target": "UserStories",
        "summary_file": (
            "table_09_transformer_cross_"
            "pure_to_userstories_numeric.csv"
        ),
    },
    {
        "number": "13",
        "slug": (
            "confusion_cross_"
            "transformer_userstories_to_pure"
        ),
        "family": "transformer",
        "setting": "cross",
        "source": "UserStories",
        "target": "PURE",
        "summary_file": (
            "table_10_transformer_cross_"
            "userstories_to_pure_numeric.csv"
        ),
    },
]


def select_highest_mcc_model(
    summary_file: str,
) -> tuple[str, str]:
    df = read_csv(
        TABLE_DIR
        / summary_file
    )

    assert_no_flan(
        df,
        summary_file,
    )

    if len(df) != 5:
        raise AssertionError(
            f"{summary_file}: "
            f"expected five models, "
            f"got {len(df)}"
        )

    ordered = (
        df
        .sort_values(
            [
                "mcc_mean",
                "f1_mean",
                "model_display",
            ],
            ascending=[
                False,
                False,
                True,
            ],
        )
        .reset_index(
            drop=True
        )
    )

    row = ordered.iloc[0]

    return (
        str(
            row["model"]
        ),
        str(
            row[
                "model_display"
            ]
        ),
    )


def classical_mean_confusion(
    setting: str,
    source: str,
    target: str,
    model: str,
) -> np.ndarray:
    df = read_csv(
        RAW_DIR
        / "classical_predictions.csv"
    )

    if setting == "intra":
        protocol = (
            "intra_source_disjoint"
        )

    elif setting == "cross":
        protocol = (
            "cross_full_source"
        )

    else:
        raise ValueError(
            f"Unexpected setting: {setting}"
        )

    sub = df[
        (
            df["protocol"]
            == protocol
        )
        & (
            df["source_dataset"]
            == source
        )
        & (
            df["target_dataset"]
            == target
        )
        & (
            df["model"]
            == model
        )
    ].copy()

    matrices = []

    for seed in SEEDS:
        seed_df = sub[
            sub["seed"]
            == seed
        ]

        if seed_df.empty:
            raise AssertionError(
                "Missing classical predictions: "
                f"{setting}, "
                f"{source}->{target}, "
                f"{model}, "
                f"seed={seed}"
            )

        matrix = confusion_matrix(
            seed_df["label"],
            seed_df["prediction"],
            labels=[
                0,
                1,
            ],
            normalize="true",
        )

        matrices.append(
            matrix
        )

    return np.mean(
        matrices,
        axis=0,
    )


def transformer_mean_confusion(
    setting: str,
    source: str,
    target: str,
    model: str,
) -> np.ndarray:
    matrices = []

    for seed in SEEDS:
        filename = (
            f"{model}__"
            f"{setting}__"
            f"{source}_to_{target}__"
            f"seed{seed}__predictions.csv"
        )

        path = (
            TRANSFORMER_RUN_DIR
            / filename
        )

        df = read_csv(
            path
        )

        assert_no_flan(
            df,
            filename,
        )

        matrix = confusion_matrix(
            df["y_true"],
            df["y_pred"],
            labels=[
                0,
                1,
            ],
            normalize="true",
        )

        matrices.append(
            matrix
        )

    return np.mean(
        matrices,
        axis=0,
    )


def plot_confusion_matrix(
    matrix: np.ndarray,
    number: str,
    slug: str,
    title: str,
    description: str,
) -> None:
    fig, ax = plt.subplots(
        figsize=(
            6.5,
            5.7,
        )
    )

    image = ax.imshow(
        matrix,
        cmap="Blues",
        vmin=0.0,
        vmax=1.0,
    )

    tick_labels = [
        "Clear\n(0)",
        "Indicator-\npositive (1)",
    ]

    ax.set_xticks(
        [
            0,
            1,
        ],
        tick_labels,
    )

    ax.set_yticks(
        [
            0,
            1,
        ],
        tick_labels,
    )

    ax.set_xlabel(
        "Predicted label"
    )

    ax.set_ylabel(
        "True label"
    )

    ax.set_title(
        title,
        pad=14,
    )

    for i in range(
        2
    ):
        for j in range(
            2
        ):
            value = float(
                matrix[
                    i,
                    j,
                ]
            )

            text_color = (
                "white"
                if value >= 0.55
                else "black"
            )

            ax.text(
                j,
                i,
                f"{value:.3f}",
                ha="center",
                va="center",
                fontsize=15,
                fontweight="bold",
                color=text_color,
            )

    colorbar = fig.colorbar(
        image,
        ax=ax,
        fraction=0.046,
        pad=0.04,
    )

    colorbar.set_label(
        "Row-normalized proportion"
    )

    fig.tight_layout()

    save_figure(
        fig=fig,
        number=number,
        slug=slug,
        scope="main",
        description=description,
    )


def build_confusion_matrices() -> None:
    for config in (
        CONFUSION_CONFIGS
    ):
        (
            model,
            model_display,
        ) = (
            select_highest_mcc_model(
                config[
                    "summary_file"
                ]
            )
        )

        if (
            config[
                "family"
            ]
            == "classical"
        ):
            matrix = (
                classical_mean_confusion(
                    setting=(
                        config[
                            "setting"
                        ]
                    ),
                    source=(
                        config[
                            "source"
                        ]
                    ),
                    target=(
                        config[
                            "target"
                        ]
                    ),
                    model=model,
                )
            )

        else:
            matrix = (
                transformer_mean_confusion(
                    setting=(
                        config[
                            "setting"
                        ]
                    ),
                    source=(
                        config[
                            "source"
                        ]
                    ),
                    target=(
                        config[
                            "target"
                        ]
                    ),
                    model=model,
                )
            )

        if (
            config[
                "setting"
            ]
            == "intra"
        ):
            title = (
                f"{model_display}: "
                f"{DATASET_DISPLAY[config['source']]} "
                "intra-corpus"
            )

        else:
            title = (
                f"{model_display}: "
                f"{DATASET_DISPLAY[config['source']]} "
                "→ "
                f"{DATASET_DISPLAY[config['target']]}"
            )

        description = (
            f"{title}. "
            "The displayed model is selected "
            "mechanically as the model with "
            "the highest mean MCC within the "
            "corresponding v3 model family "
            "and evaluation setting. "
            "The row-normalized matrix is "
            "averaged across five seeds."
        )

        plot_confusion_matrix(
            matrix=matrix,
            number=(
                config[
                    "number"
                ]
            ),
            slug=(
                config[
                    "slug"
                ]
            ),
            title=title,
            description=description,
        )


# =============================================================================
# Figures 14–17: reason-specific detection recall
# =============================================================================

def build_reason_heatmap(
    number: str,
    slug: str,
    numeric_file: str,
    family: str,
    title: str,
) -> None:
    df = read_csv(
        TABLE_DIR
        / numeric_file
    )

    assert_no_flan(
        df,
        numeric_file,
    )

    sub = df[
        df["family"]
        == family
    ].copy()

    if sub.empty:
        raise AssertionError(
            f"No rows for family="
            f"{family} in "
            f"{numeric_file}"
        )

    if family == "Classical":
        model_order = (
            CLASSICAL_HEATMAP_ORDER
        )

    elif family == "Transformer":
        model_order = (
            TRANSFORMER_HEATMAP_ORDER
        )

    else:
        raise ValueError(
            f"Unknown family: {family}"
        )

    reason_support = (
        sub[
            [
                "reason",
                "support",
            ]
        ]
        .drop_duplicates()
        .sort_values(
            "support",
            ascending=False,
        )
    )

    reason_order = (
        reason_support[
            "reason"
        ]
        .tolist()
    )

    support_map = dict(
        zip(
            reason_support[
                "reason"
            ],
            reason_support[
                "support"
            ],
        )
    )

    pivot = (
        sub
        .pivot_table(
            index="reason",
            columns="model_display",
            values=(
                "detection_recall_mean"
            ),
            aggfunc="first",
        )
        .reindex(
            index=reason_order,
            columns=model_order,
        )
    )

    matrix = pivot.to_numpy(
        dtype=float
    )

    fig_height = max(
        4.6,
        (
            0.67
            * len(
                reason_order
            )
            + 2.2
        ),
    )

    fig, ax = plt.subplots(
        figsize=(
            9.2,
            fig_height,
        )
    )

    image = ax.imshow(
        matrix,
        cmap="viridis",
        vmin=0.0,
        vmax=1.0,
        aspect="auto",
    )

    ax.set_xticks(
        np.arange(
            len(
                model_order
            )
        ),
        model_order,
        rotation=30,
        ha="right",
    )

    reason_labels = [
        (
            f"{format_reason(reason)} "
            f"(N={int(support_map[reason])})"
        )
        for reason
        in reason_order
    ]

    ax.set_yticks(
        np.arange(
            len(
                reason_order
            )
        ),
        reason_labels,
    )

    ax.set_title(
        title,
        pad=12,
    )

    for i in range(
        matrix.shape[0]
    ):
        for j in range(
            matrix.shape[1]
        ):
            value = (
                matrix[
                    i,
                    j,
                ]
            )

            if np.isnan(
                value
            ):
                continue

            text_color = (
                "black"
                if value >= 0.62
                else "white"
            )

            ax.text(
                j,
                i,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=9,
                color=text_color,
            )

    colorbar = fig.colorbar(
        image,
        ax=ax,
        fraction=0.035,
        pad=0.035,
    )

    colorbar.set_label(
        "Reason-specific detection recall"
    )

    fig.tight_layout()

    save_figure(
        fig=fig,
        number=number,
        slug=slug,
        scope="main",
        description=(
            f"{title}. "
            "Only reasons with target support "
            "N >= 20 are included. "
            "Values are mean reason-specific "
            "detection recall across five seeds, "
            "not reason-level F1."
        ),
    )


def build_reason_heatmaps() -> None:
    build_reason_heatmap(
        number="14",
        slug=(
            "reason_recall_classical_"
            "pure_to_userstories"
        ),
        numeric_file=(
            "table_15_reason_detection_"
            "pure_to_userstories_numeric.csv"
        ),
        family="Classical",
        title=(
            "Reason-specific detection recall | "
            "Classical | PURE → User Stories"
        ),
    )

    build_reason_heatmap(
        number="15",
        slug=(
            "reason_recall_transformer_"
            "pure_to_userstories"
        ),
        numeric_file=(
            "table_15_reason_detection_"
            "pure_to_userstories_numeric.csv"
        ),
        family="Transformer",
        title=(
            "Reason-specific detection recall | "
            "Transformers | PURE → User Stories"
        ),
    )

    build_reason_heatmap(
        number="16",
        slug=(
            "reason_recall_classical_"
            "userstories_to_pure"
        ),
        numeric_file=(
            "table_16_reason_detection_"
            "userstories_to_pure_numeric.csv"
        ),
        family="Classical",
        title=(
            "Reason-specific detection recall | "
            "Classical | User Stories → PURE"
        ),
    )

    build_reason_heatmap(
        number="17",
        slug=(
            "reason_recall_transformer_"
            "userstories_to_pure"
        ),
        numeric_file=(
            "table_16_reason_detection_"
            "userstories_to_pure_numeric.csv"
        ),
        family="Transformer",
        title=(
            "Reason-specific detection recall | "
            "Transformers | User Stories → PURE"
        ),
    )


# =============================================================================
# Figures 18–21: overall F1 comparisons
# =============================================================================

def load_combined_performance(
    classical_file: str,
    transformer_file: str,
) -> pd.DataFrame:
    classical = read_csv(
        TABLE_DIR
        / classical_file
    )

    transformers = read_csv(
        TABLE_DIR
        / transformer_file
    )

    assert_no_flan(
        classical,
        classical_file,
    )

    assert_no_flan(
        transformers,
        transformer_file,
    )

    classical[
        "family"
    ] = "Classical"

    transformers[
        "family"
    ] = "Transformer"

    combined = pd.concat(
        [
            classical,
            transformers,
        ],
        ignore_index=True,
        sort=False,
    )

    if len(
        combined
    ) != 10:
        raise AssertionError(
            "Expected 10 models in "
            "combined performance plot, "
            f"got {len(combined)}"
        )

    return combined


def build_f1_comparison(
    number: str,
    slug: str,
    title: str,
    classical_file: str,
    transformer_file: str,
) -> None:
    df = (
        load_combined_performance(
            classical_file,
            transformer_file,
        )
    )

    df = (
        df
        .sort_values(
            [
                "f1_mean",
                "model_display",
            ],
            ascending=[
                False,
                True,
            ],
        )
        .reset_index(
            drop=True
        )
    )

    x = np.arange(
        len(df)
    )

    fig, ax = plt.subplots(
        figsize=(
            11.5,
            5.6,
        )
    )

    ax.bar(
        x,
        df["f1_mean"],
        yerr=df["f1_std"],
        color=BLUE,
        edgecolor=BLUE,
        capsize=4,
        error_kw={
            "ecolor": "black",
            "elinewidth": 1.3,
            "capthick": 1.3,
        },
    )

    ax.set_xticks(
        x,
        df[
            "model_display"
        ],
        rotation=32,
        ha="right",
    )

    ax.set_ylabel(
        "F1"
    )

    ax.set_ylim(
        0,
        1.05,
    )

    ax.set_title(
        title
    )

    remove_top_right_spines(
        ax
    )

    fig.tight_layout()

    save_figure(
        fig=fig,
        number=number,
        slug=slug,
        scope="main",
        description=(
            f"{title}. "
            "Models are ordered by descending "
            "mean F1. Error bars show sample SD "
            "across five seeds."
        ),
    )


def build_overall_f1_figures() -> None:
    build_f1_comparison(
        number="18",
        slug="f1_intra_pure",
        title=(
            "Intra-corpus | PURE | F1"
        ),
        classical_file=(
            "table_03_classical_"
            "intra_pure_numeric.csv"
        ),
        transformer_file=(
            "table_05_transformer_"
            "intra_pure_numeric.csv"
        ),
    )

    build_f1_comparison(
        number="19",
        slug=(
            "f1_intra_userstories"
        ),
        title=(
            "Intra-corpus | "
            "User Stories | F1"
        ),
        classical_file=(
            "table_04_classical_"
            "intra_userstories_numeric.csv"
        ),
        transformer_file=(
            "table_06_transformer_"
            "intra_userstories_numeric.csv"
        ),
    )

    build_f1_comparison(
        number="20",
        slug=(
            "f1_cross_"
            "pure_to_userstories"
        ),
        title=(
            "Cross-corpus | "
            "PURE → User Stories | F1"
        ),
        classical_file=(
            "table_07_classical_cross_"
            "pure_to_userstories_numeric.csv"
        ),
        transformer_file=(
            "table_09_transformer_cross_"
            "pure_to_userstories_numeric.csv"
        ),
    )

    build_f1_comparison(
        number="21",
        slug=(
            "f1_cross_"
            "userstories_to_pure"
        ),
        title=(
            "Cross-corpus | "
            "User Stories → PURE | F1"
        ),
        classical_file=(
            "table_08_classical_cross_"
            "userstories_to_pure_numeric.csv"
        ),
        transformer_file=(
            "table_10_transformer_cross_"
            "userstories_to_pure_numeric.csv"
        ),
    )


# =============================================================================
# Figure 22: source-size control
# =============================================================================

def build_source_size_control() -> None:
    df = read_csv(
        TABLE_DIR
        / (
            "table_12_source_size_control_"
            "pure_to_userstories_numeric.csv"
        )
    )

    assert_no_flan(
        df,
        "source-size-control table",
    )

    full_label = (
        "Full PURE source"
    )

    matched_label = (
        "Size-matched PURE source"
    )

    model_order = (
        CLASSICAL_DISPLAY_ORDER
        + TRANSFORMER_DISPLAY_ORDER
    )

    full = (
        df[
            df[
                "source_training_regime"
            ]
            == full_label
        ]
        .set_index(
            "model_display"
        )
        .reindex(
            model_order
        )
    )

    matched = (
        df[
            df[
                "source_training_regime"
            ]
            == matched_label
        ]
        .set_index(
            "model_display"
        )
        .reindex(
            model_order
        )
    )

    if (
        full[
            "f1_mean"
        ].isna().any()
        or matched[
            "f1_mean"
        ].isna().any()
    ):
        raise AssertionError(
            "Missing model in "
            "source-size-control table."
        )

    x = np.arange(
        len(
            model_order
        )
    )

    width = 0.37

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(
            12.0,
            7.4,
        ),
        sharex=True,
    )

    for (
        ax,
        metric,
        metric_label,
    ) in [
        (
            axes[0],
            "f1",
            "F1",
        ),
        (
            axes[1],
            "mcc",
            "MCC",
        ),
    ]:
        ax.bar(
            x - width / 2,
            full[
                f"{metric}_mean"
            ],
            width,
            yerr=full[
                f"{metric}_std"
            ],
            color=BLUE,
            label="Full source",
            capsize=4,
            error_kw={
                "ecolor": "black",
                "elinewidth": 1.2,
                "capthick": 1.2,
            },
        )

        ax.bar(
            x + width / 2,
            matched[
                f"{metric}_mean"
            ],
            width,
            yerr=matched[
                f"{metric}_std"
            ],
            color=ORANGE,
            label="Size-matched",
            capsize=4,
            error_kw={
                "ecolor": "black",
                "elinewidth": 1.2,
                "capthick": 1.2,
            },
        )

        ax.set_ylim(
            0,
            1.05,
        )

        ax.set_ylabel(
            metric_label
        )

        remove_top_right_spines(
            ax
        )

        ax.axvline(
            4.5,
            linewidth=0.9,
            color="0.75",
        )

    axes[0].set_title(
        "PURE → User Stories: "
        "effect of source-size control"
    )

    axes[0].legend(
        frameon=False,
        ncol=2,
        loc="upper left",
    )

    axes[1].set_xticks(
        x,
        model_order,
        rotation=32,
        ha="right",
    )

    axes[1].text(
        2.0,
        -0.29,
        "Classical",
        transform=(
            axes[1]
            .get_xaxis_transform()
        ),
        ha="center",
        fontsize=10,
    )

    axes[1].text(
        7.0,
        -0.29,
        "Transformers",
        transform=(
            axes[1]
            .get_xaxis_transform()
        ),
        ha="center",
        fontsize=10,
    )

    fig.tight_layout()

    save_figure(
        fig=fig,
        number="22",
        slug="source_size_control",
        scope="main",
        description=(
            "Full-source versus size-matched "
            "PURE-to-User-Stories transfer. "
            "F1 and MCC are shown with "
            "sample-SD error bars across "
            "five seeds."
        ),
    )


# =============================================================================
# Figures 23–24: cue ablation
# =============================================================================

def build_cue_ablation(
    number: str,
    slug: str,
    numeric_file: str,
    source: str,
    target: str,
) -> None:
    df = read_csv(
        TABLE_DIR
        / numeric_file
    )

    assert_no_flan(
        df,
        numeric_file,
    )

    model_order = (
        CLASSICAL_DISPLAY_ORDER
    )

    x = np.arange(
        len(
            model_order
        )
    )

    width = 0.25

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(
            10.5,
            7.3,
        ),
        sharex=True,
    )

    for (
        ax,
        metric,
        metric_label,
    ) in [
        (
            axes[0],
            "f1",
            "F1",
        ),
        (
            axes[1],
            "mcc",
            "MCC",
        ),
    ]:
        for (
            index,
            variant,
        ) in enumerate(
            VARIANT_ORDER
        ):
            sub = (
                df[
                    df["variant"]
                    == variant
                ]
                .set_index(
                    "model_display"
                )
                .reindex(
                    model_order
                )
            )

            if sub[
                f"{metric}_mean"
            ].isna().any():
                raise AssertionError(
                    "Missing cue-ablation model "
                    f"for variant={variant}"
                )

            offset = (
                index - 1
            ) * width

            ax.bar(
                x + offset,
                sub[
                    f"{metric}_mean"
                ],
                width,
                yerr=sub[
                    f"{metric}_std"
                ],
                color=(
                    VARIANT_COLORS[
                        variant
                    ]
                ),
                label=(
                    VARIANT_DISPLAY[
                        variant
                    ]
                ),
                capsize=3,
                error_kw={
                    "ecolor": "black",
                    "elinewidth": 1.1,
                    "capthick": 1.1,
                },
            )

        ax.set_ylim(
            0,
            1.05,
        )

        ax.set_ylabel(
            metric_label
        )

        remove_top_right_spines(
            ax
        )

    axes[0].set_title(
        "Cue ablation: "
        f"{DATASET_DISPLAY[source]} "
        "→ "
        f"{DATASET_DISPLAY[target]}"
    )

    axes[0].legend(
        frameon=False,
        ncol=3,
        loc="upper left",
    )

    axes[1].set_xticks(
        x,
        model_order,
        rotation=30,
        ha="right",
    )

    fig.tight_layout()

    save_figure(
        fig=fig,
        number=number,
        slug=slug,
        scope="main",
        description=(
            "Cue-ablation analysis comparing "
            "original, masked-cue, and "
            "removed-cue conditions. "
            "Models are retrained and evaluated "
            "under each transformation; error "
            "bars show sample SD across five seeds."
        ),
    )


def build_cue_ablation_figures() -> None:
    build_cue_ablation(
        number="23",
        slug=(
            "cue_ablation_"
            "pure_to_userstories"
        ),
        numeric_file=(
            "table_13_cue_ablation_"
            "pure_to_userstories_numeric.csv"
        ),
        source="PURE",
        target="UserStories",
    )

    build_cue_ablation(
        number="24",
        slug=(
            "cue_ablation_"
            "userstories_to_pure"
        ),
        numeric_file=(
            "table_14_cue_ablation_"
            "userstories_to_pure_numeric.csv"
        ),
        source="UserStories",
        target="PURE",
    )


# =============================================================================
# Supplementary Figure S1: seed-level variability
# =============================================================================

def build_seed_variability() -> None:
    df = read_csv(
        SUMMARY_DIR
        / "transformer_integrity_check.csv"
    )

    df = df[
        df["setting"]
        == "cross_matched"
    ].copy()

    assert_no_flan(
        df,
        "transformer_integrity_check.csv",
    )

    model_specs = [
        (
            "DistilBERT",
            BLUE,
        ),
        (
            "BERT",
            ORANGE,
        ),
        (
            "RoBERTa",
            GREEN,
        ),
        (
            "ALBERT",
            RED,
        ),
        (
            "T5",
            PURPLE,
        ),
    ]

    fig, ax = plt.subplots(
        figsize=(
            8.8,
            5.4,
        )
    )

    for (
        model,
        color,
    ) in model_specs:
        sub = (
            df[
                df["model"]
                == model
            ]
            .sort_values(
                "seed"
            )
        )

        if len(
            sub
        ) != 5:
            raise AssertionError(
                f"Expected five matched "
                f"runs for {model}, "
                f"got {len(sub)}"
            )

        ax.plot(
            sub["seed"],
            sub["mcc"],
            marker="o",
            markersize=6,
            linewidth=1.8,
            color=color,
            label=model,
        )

    ax.set_xticks(
        SEEDS
    )

    ax.set_xlabel(
        "Seed"
    )

    ax.set_ylabel(
        "MCC"
    )

    ax.set_ylim(
        0,
        1.0,
    )

    ax.set_title(
        "Seed-level variability under "
        "size-matched PURE → User Stories transfer"
    )

    ax.grid(
        True,
        linewidth=0.6,
        alpha=0.30,
    )

    ax.legend(
        frameon=False,
        ncol=3,
    )

    remove_top_right_spines(
        ax
    )

    fig.tight_layout()

    save_figure(
        fig=fig,
        number="S1",
        slug=(
            "matched_transformer_"
            "seed_variability"
        ),
        scope="supplementary",
        description=(
            "Per-seed MCC for transformer "
            "models under the source-size-matched "
            "PURE-to-User-Stories control."
        ),
    )


# =============================================================================
# Manifest
# =============================================================================

def write_manifest() -> None:
    manifest = pd.DataFrame(
        manifest_rows
    )

    path = (
        FIGURE_DIR
        / "figure_manifest.csv"
    )

    manifest.to_csv(
        path,
        index=False,
        encoding="utf-8-sig",
    )

    print(
        f"Saved {path}"
    )


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    print("=" * 80)
    print("BUILDING FINAL V3 FIGURES")
    print("=" * 80)

    print()
    print(
        "Figure 01 is the conceptual "
        "screening-scenario diagram and is "
        "not regenerated by this script."
    )

    clean_output_directory()

    # Figures 02–05
    build_class_distribution()
    build_reason_distribution()
    build_word_length_distribution()
    build_character_length_distribution()

    # Figures 06–13
    build_confusion_matrices()

    # Figures 14–17
    build_reason_heatmaps()

    # Figures 18–21
    build_overall_f1_figures()

    # Figure 22
    build_source_size_control()

    # Figures 23–24
    build_cue_ablation_figures()

    # Supplementary Figure S1
    build_seed_variability()

    write_manifest()

    main_count = sum(
        row["scope"]
        == "main"
        for row
        in manifest_rows
    )

    supplementary_count = sum(
        row["scope"]
        == "supplementary"
        for row
        in manifest_rows
    )

    print()
    print("=" * 80)
    print(
        "FINAL V3 FIGURE GENERATION COMPLETED"
    )
    print("=" * 80)

    print(
        f"Generated main figures: "
        f"{main_count}"
    )

    print(
        "Existing conceptual Figure 01: 1"
    )

    print(
        "Total planned manuscript figures: "
        f"{main_count + 1}"
    )

    print(
        "Supplementary candidate figures: "
        f"{supplementary_count}"
    )

    print(
        f"Output directory: "
        f"{FIGURE_DIR}"
    )

    print()
    print(
        "Checks and conventions:"
    )

    print(
        "  - Figure numbering preserves "
        "the original manuscript through Fig. 21"
    )

    print(
        "  - Figure 03 reason counts come "
        "directly from clean deduplicated data"
    )

    print(
        "  - Figures 04–05 limit only "
        "the displayed range to within-dataset "
        "pooled p99.5"
    )

    print(
        "  - all extreme-length observations "
        "remain in datasets and experiments"
    )

    print(
        "  - confusion models are selected "
        "mechanically by highest mean MCC "
        "within model family and setting"
    )

    print(
        "  - confusion matrices are averaged "
        "across all five seeds"
    )

    print(
        "  - reason heatmaps show mean "
        "reason-specific detection recall"
    )

    print(
        "  - performance/control figures "
        "include sample-SD error bars"
    )

    print(
        "  - explanatory methodological text "
        "is reserved for manuscript captions"
    )

    print(
        "  - FLAN-T5 is excluded everywhere"
    )


if __name__ == "__main__":
    main()