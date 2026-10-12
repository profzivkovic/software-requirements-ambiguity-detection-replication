from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


RAW_DIR = PROJECT_ROOT / "results" / "raw"
SUMMARY_DIR = PROJECT_ROOT / "results" / "summary"

TABLE_DIR = SUMMARY_DIR / "final_tables"

TRANSFORMER_RUN_DIR = (
    RAW_DIR / "transformer_runs"
)

PURE_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "pure_clean.csv"
)

USERSTORIES_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "userstories_clean.csv"
)


SEEDS = [13, 21, 42, 87, 123]


MAIN_METRICS = [
    "accuracy",
    "precision",
    "recall",
    "f1",
    "mcc",
    "cohen_kappa",
]

ALL_METRICS = [
    "accuracy",
    "balanced_accuracy",
    "precision",
    "recall",
    "f1",
    "mcc",
    "cohen_kappa",
]


METRIC_LABELS = {
    "accuracy": "Accuracy",
    "balanced_accuracy": "Balanced accuracy",
    "precision": "Precision",
    "recall": "Recall",
    "f1": "F1",
    "mcc": "MCC",
    "cohen_kappa": "Cohen's κ",
}


CLASSICAL_ORDER = [
    "MultinomialNB",
    "LinearSVM",
    "RandomForest",
    "LightGBM",
    "XGBoost",
]

TRANSFORMER_ORDER = [
    "DistilBERT",
    "BERT",
    "RoBERTa",
    "ALBERT",
    "T5",
]

ALL_MODEL_ORDER = (
    CLASSICAL_ORDER
    + TRANSFORMER_ORDER
)


MODEL_DISPLAY = {
    "MultinomialNB": "Naive Bayes",
    "LinearSVM": "Linear SVM",
    "RandomForest": "Random Forest",
    "LightGBM": "LightGBM",
    "XGBoost": "XGBoost",
    "DistilBERT": "DistilBERT",
    "BERT": "BERT",
    "RoBERTa": "RoBERTa",
    "ALBERT": "ALBERT",
    "T5": "T5",
    "Majority": "Majority",
    "RuleBased": "Rule-based",
}


VARIANT_ORDER = [
    "original",
    "masked",
    "removed",
]


manifest_rows: list[dict[str, str]] = []


def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing required file: {path}"
        )


def read_csv(path: Path) -> pd.DataFrame:
    require_file(path)
    return pd.read_csv(path)


def assert_five_seeds(
    df: pd.DataFrame,
    group_cols: list[str],
    label: str,
) -> None:
    expected = set(SEEDS)

    for keys, sub in df.groupby(
        group_cols,
        dropna=False,
    ):
        observed = set(
            sub["seed"]
            .astype(int)
            .tolist()
        )

        if observed != expected:
            raise AssertionError(
                f"{label}, {keys}: "
                f"expected seeds {sorted(expected)}, "
                f"got {sorted(observed)}"
            )

        if len(sub) != 5:
            raise AssertionError(
                f"{label}, {keys}: "
                f"expected 5 rows, got {len(sub)}"
            )


def aggregate_runs(
    df: pd.DataFrame,
    group_cols: list[str],
    value_cols: list[str],
    label: str,
) -> pd.DataFrame:
    assert_five_seeds(
        df=df,
        group_cols=group_cols,
        label=label,
    )

    grouped = (
        df
        .groupby(
            group_cols,
            dropna=False,
        )[value_cols]
        .agg(
            [
                "mean",
                "std",
                "var",
            ]
        )
        .reset_index()
    )

    grouped.columns = [
        "_".join(
            [
                str(part)
                for part in col
                if str(part) != ""
            ]
        )
        if isinstance(col, tuple)
        else str(col)
        for col in grouped.columns
    ]

    return grouped


def mean_sd(
    mean_value: float,
    std_value: float,
    decimals: int = 3,
) -> str:
    return (
        f"{float(mean_value):.{decimals}f}"
        f" ± "
        f"{float(std_value):.{decimals}f}"
    )


def model_display(value: str) -> str:
    return MODEL_DISPLAY.get(
        str(value),
        str(value),
    )


def sort_models(
    df: pd.DataFrame,
    column: str = "model",
    order: list[str] | None = None,
) -> pd.DataFrame:
    if order is None:
        order = ALL_MODEL_ORDER

    out = df.copy()

    out["_model_order"] = pd.Categorical(
        out[column],
        categories=order,
        ordered=True,
    )

    out = (
        out
        .sort_values("_model_order")
        .drop(columns=["_model_order"])
        .reset_index(drop=True)
    )

    return out


def formatted_performance_table(
    df: pd.DataFrame,
    extra_columns: list[
        tuple[str, str]
    ] | None = None,
) -> pd.DataFrame:
    if extra_columns is None:
        extra_columns = []

    out = pd.DataFrame(
        index=df.index
    )

    for source_col, display_col in extra_columns:
        out[display_col] = df[source_col]

    if "model" in df.columns:
        out["Model"] = (
            df["model"]
            .astype(str)
            .map(model_display)
        )

    for metric in MAIN_METRICS:
        mean_col = f"{metric}_mean"
        std_col = f"{metric}_std"

        out[
            METRIC_LABELS[metric]
        ] = [
            mean_sd(mean, std)
            for mean, std in zip(
                df[mean_col],
                df[std_col],
            )
        ]

    return out.reset_index(
        drop=True
    )


def numeric_performance_table(
    df: pd.DataFrame,
    id_columns: list[str],
    include_train_size: bool = False,
) -> pd.DataFrame:
    out = df.copy()

    if "model" in out.columns:
        out["model_display"] = (
            out["model"]
            .astype(str)
            .map(model_display)
        )

    wanted = list(id_columns)

    if (
        "model_display" in out.columns
        and "model_display" not in wanted
    ):
        wanted.append(
            "model_display"
        )

    if include_train_size:
        for suffix in [
            "mean",
            "std",
            "var",
        ]:
            col = (
                f"train_size_{suffix}"
            )

            if col in out.columns:
                wanted.append(col)

    for metric in ALL_METRICS:
        for suffix in [
            "mean",
            "std",
            "var",
        ]:
            column = (
                f"{metric}_{suffix}"
            )

            if column in out.columns:
                wanted.append(column)

    return (
        out[wanted]
        .copy()
        .reset_index(drop=True)
    )


def write_table(
    number: str,
    slug: str,
    manuscript_df: pd.DataFrame,
    numeric_df: pd.DataFrame | None = None,
    scope: str = "main",
) -> None:
    TABLE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    manuscript_path = (
        TABLE_DIR
        / f"table_{number}_{slug}.csv"
    )

    manuscript_df.to_csv(
        manuscript_path,
        index=False,
        encoding="utf-8-sig",
    )

    print(
        f"Saved {manuscript_path}"
    )

    manifest_rows.append(
        {
            "table": number,
            "slug": slug,
            "scope": scope,
            "type": "manuscript",
            "file": manuscript_path.name,
        }
    )

    if numeric_df is not None:
        numeric_path = (
            TABLE_DIR
            / (
                f"table_{number}_{slug}"
                f"_numeric.csv"
            )
        )

        numeric_df.to_csv(
            numeric_path,
            index=False,
            encoding="utf-8-sig",
        )

        print(
            f"Saved {numeric_path}"
        )

        manifest_rows.append(
            {
                "table": number,
                "slug": slug,
                "scope": scope,
                "type": "numeric_companion",
                "file": numeric_path.name,
            }
        )


def build_dataset_table() -> None:
    rows = []

    for dataset_name, path in [
        ("PURE", PURE_PATH),
        ("User Stories", USERSTORIES_PATH),
    ]:
        df = read_csv(path)

        words = (
            df["requirement_text"]
            .astype(str)
            .str.split()
            .str.len()
        )

        label_0 = int(
            (df["label"] == 0).sum()
        )

        label_1 = int(
            (df["label"] == 1).sum()
        )

        rows.append(
            {
                "Dataset": dataset_name,
                "Requirements": len(df),
                "Source documents": int(
                    df["source_file"].nunique()
                ),
                "Clear": label_0,
                "Indicator-positive": label_1,
                "Positive rate": (
                    f"{label_1 / len(df):.3f}"
                ),
                "Average words": (
                    f"{words.mean():.1f}"
                ),
            }
        )

    table = pd.DataFrame(rows)

    write_table(
        "01",
        "dataset_characteristics",
        table,
    )


def build_configuration_table() -> None:
    rows = [
        {
            "Component": "Random seeds",
            "Configuration": (
                "13, 21, 42, 87, 123"
            ),
        },
        {
            "Component": "Classical representation",
            "Configuration": (
                "TF-IDF word unigrams+bigrams; "
                "lowercase=True; min_df=2; "
                "max_df=0.95; sublinear_tf=True"
            ),
        },
        {
            "Component": "Classical models",
            "Configuration": (
                "Multinomial Naive Bayes; "
                "Linear SVM; Random Forest "
                "(400 trees, balanced class weights); "
                "LightGBM (600 estimators, lr=0.05, "
                "31 leaves); XGBoost "
                "(600 estimators, lr=0.05, depth=6)"
            ),
        },
        {
            "Component": "Encoder transformers",
            "Configuration": (
                "DistilBERT, BERT, RoBERTa, ALBERT; "
                "max length=128; up to 3 epochs; "
                "AdamW; lr=3e-5; batch=8; "
                "weight decay=0.01; warmup=0.06; "
                "validation-F1 checkpoint selection"
            ),
        },
        {
            "Component": "T5",
            "Configuration": (
                "t5-small text-to-text yes/no; "
                "review-support prompt; "
                "max input length=128; "
                "max target length=8; "
                "up to 3 epochs; lr=1e-4; "
                "batch=8; weight decay=0.01; "
                "warmup=0.06"
            ),
        },
        {
            "Component": "Intra-corpus protocol",
            "Configuration": (
                "Source-disjoint locked splits. "
                "Transformers use train/validation/test "
                "(approximately 70/10/20); "
                "classical models combine train+validation "
                "and evaluate on the same held-out test set."
            ),
        },
        {
            "Component": "Cross-corpus classical protocol",
            "Configuration": (
                "Train on the complete source corpus; "
                "evaluate on the complete target corpus."
            ),
        },
        {
            "Component": "Cross-corpus transformer protocol",
            "Configuration": (
                "Source-only source-disjoint "
                "approximately 90/10 train/validation; "
                "complete external target corpus used "
                "only for final testing."
            ),
        },
        {
            "Component": "Size-matched control",
            "Configuration": (
                "PURE source restricted to complete "
                "source groups approximately matching "
                "User Stories corpus size; target remains "
                "the complete User Stories corpus."
            ),
        },
        {
            "Component": "Rule baseline",
            "Configuration": (
                "Frozen requirements-smell-style "
                "surface-cue regex baseline."
            ),
        },
        {
            "Component": "Reported variability",
            "Configuration": (
                "All five-seed performance tables report "
                "mean ± sample SD; numeric companion "
                "tables additionally report sample "
                "variance (SD²)."
            ),
        },
    ]

    write_table(
        "02",
        "experimental_configuration",
        pd.DataFrame(rows),
    )


def load_aggregated_results():
    classical_runs = read_csv(
        RAW_DIR
        / "classical_results.csv"
    )

    classical_matched_runs = read_csv(
        RAW_DIR
        / "classical_size_matched_results.csv"
    )

    rule_runs = read_csv(
        RAW_DIR
        / "rule_baseline_results.csv"
    )

    # The raw rule-baseline file also contains a deterministic
    # full-corpus diagnostic evaluation stored with seed=-1.
    # It is not a five-seed experiment and therefore must not
    # be included in mean/SD/variance aggregation.
    rule_runs = rule_runs[
        rule_runs["protocol"] == "intra"
        ].copy()

    cue_runs = read_csv(
        RAW_DIR
        / "cue_ablation_results.csv"
    )

    transformer_runs = read_csv(
        SUMMARY_DIR
        / "transformer_integrity_check.csv"
    )

    classical_agg = aggregate_runs(
        df=classical_runs,
        group_cols=[
            "protocol",
            "source_dataset",
            "target_dataset",
            "model",
        ],
        value_cols=[
            "train_size",
            *ALL_METRICS,
        ],
        label="classical",
    )

    classical_matched_agg = aggregate_runs(
        df=classical_matched_runs,
        group_cols=[
            "protocol",
            "source_dataset",
            "target_dataset",
            "model",
        ],
        value_cols=[
            "train_size",
            *ALL_METRICS,
        ],
        label="classical size matched",
    )

    rule_agg = aggregate_runs(
        df=rule_runs,
        group_cols=[
            "protocol",
            "dataset",
            "model",
        ],
        value_cols=ALL_METRICS,
        label="rule baseline",
    )

    cue_agg = aggregate_runs(
        df=cue_runs,
        group_cols=[
            "protocol",
            "variant",
            "source_dataset",
            "target_dataset",
            "model",
        ],
        value_cols=ALL_METRICS,
        label="cue ablation",
    )

    transformer_agg = aggregate_runs(
        df=transformer_runs,
        group_cols=[
            "model",
            "setting",
            "train_dataset",
            "test_dataset",
        ],
        value_cols=[
            "train_size",
            *ALL_METRICS,
        ],
        label="transformers",
    )

    return (
        classical_agg,
        classical_matched_agg,
        rule_agg,
        cue_agg,
        transformer_agg,
    )


def write_standard_performance_table(
    number: str,
    slug: str,
    df: pd.DataFrame,
    model_order: list[str],
) -> None:
    df = sort_models(
        df,
        order=model_order,
    )

    manuscript = (
        formatted_performance_table(
            df
        )
    )

    numeric = (
        numeric_performance_table(
            df,
            id_columns=["model"],
        )
    )

    write_table(
        number,
        slug,
        manuscript,
        numeric,
    )


def build_main_performance_tables(
    classical_agg: pd.DataFrame,
    transformer_agg: pd.DataFrame,
) -> None:
    selections = [
        (
            "03",
            "classical_intra_pure",
            classical_agg[
                (
                    classical_agg["protocol"]
                    == "intra_source_disjoint"
                )
                & (
                    classical_agg["source_dataset"]
                    == "PURE"
                )
                & (
                    classical_agg["target_dataset"]
                    == "PURE"
                )
            ],
            CLASSICAL_ORDER,
        ),
        (
            "04",
            "classical_intra_userstories",
            classical_agg[
                (
                    classical_agg["protocol"]
                    == "intra_source_disjoint"
                )
                & (
                    classical_agg["source_dataset"]
                    == "UserStories"
                )
                & (
                    classical_agg["target_dataset"]
                    == "UserStories"
                )
            ],
            CLASSICAL_ORDER,
        ),
        (
            "05",
            "transformer_intra_pure",
            transformer_agg[
                (
                    transformer_agg["setting"]
                    == "intra"
                )
                & (
                    transformer_agg["train_dataset"]
                    == "PURE"
                )
                & (
                    transformer_agg["test_dataset"]
                    == "PURE"
                )
            ],
            TRANSFORMER_ORDER,
        ),
        (
            "06",
            "transformer_intra_userstories",
            transformer_agg[
                (
                    transformer_agg["setting"]
                    == "intra"
                )
                & (
                    transformer_agg["train_dataset"]
                    == "UserStories"
                )
                & (
                    transformer_agg["test_dataset"]
                    == "UserStories"
                )
            ],
            TRANSFORMER_ORDER,
        ),
        (
            "07",
            "classical_cross_pure_to_userstories",
            classical_agg[
                (
                    classical_agg["protocol"]
                    == "cross_full_source"
                )
                & (
                    classical_agg["source_dataset"]
                    == "PURE"
                )
                & (
                    classical_agg["target_dataset"]
                    == "UserStories"
                )
            ],
            CLASSICAL_ORDER,
        ),
        (
            "08",
            "classical_cross_userstories_to_pure",
            classical_agg[
                (
                    classical_agg["protocol"]
                    == "cross_full_source"
                )
                & (
                    classical_agg["source_dataset"]
                    == "UserStories"
                )
                & (
                    classical_agg["target_dataset"]
                    == "PURE"
                )
            ],
            CLASSICAL_ORDER,
        ),
        (
            "09",
            "transformer_cross_pure_to_userstories",
            transformer_agg[
                (
                    transformer_agg["setting"]
                    == "cross"
                )
                & (
                    transformer_agg["train_dataset"]
                    == "PURE"
                )
                & (
                    transformer_agg["test_dataset"]
                    == "UserStories"
                )
            ],
            TRANSFORMER_ORDER,
        ),
        (
            "10",
            "transformer_cross_userstories_to_pure",
            transformer_agg[
                (
                    transformer_agg["setting"]
                    == "cross"
                )
                & (
                    transformer_agg["train_dataset"]
                    == "UserStories"
                )
                & (
                    transformer_agg["test_dataset"]
                    == "PURE"
                )
            ],
            TRANSFORMER_ORDER,
        ),
    ]

    for (
        number,
        slug,
        df,
        order,
    ) in selections:
        if len(df) != 5:
            raise AssertionError(
                f"{slug}: expected 5 models, "
                f"got {len(df)}"
            )

        write_standard_performance_table(
            number=number,
            slug=slug,
            df=df.copy(),
            model_order=order,
        )


def build_rule_table(
    rule_agg: pd.DataFrame,
) -> None:
    df = rule_agg.copy()

    df["_dataset_order"] = pd.Categorical(
        df["dataset"],
        categories=[
            "PURE",
            "UserStories",
        ],
        ordered=True,
    )

    df["_model_order"] = pd.Categorical(
        df["model"],
        categories=[
            "Majority",
            "RuleBased",
        ],
        ordered=True,
    )

    df = (
        df
        .sort_values(
            [
                "_dataset_order",
                "_model_order",
            ]
        )
        .drop(
            columns=[
                "_dataset_order",
                "_model_order",
            ]
        )
        .reset_index(drop=True)
    )

    manuscript = (
        formatted_performance_table(
            df,
            extra_columns=[
                (
                    "dataset",
                    "Dataset",
                ),
            ],
        )
    )

    numeric = (
        numeric_performance_table(
            df,
            id_columns=[
                "dataset",
                "model",
            ],
        )
    )

    write_table(
        "11",
        "rule_and_majority_baselines",
        manuscript,
        numeric,
    )


def build_size_control_table(
    classical_agg: pd.DataFrame,
    classical_matched_agg: pd.DataFrame,
    transformer_agg: pd.DataFrame,
) -> None:
    pieces = []

    classical_full = classical_agg[
        (
            classical_agg["protocol"]
            == "cross_full_source"
        )
        & (
            classical_agg["source_dataset"]
            == "PURE"
        )
        & (
            classical_agg["target_dataset"]
            == "UserStories"
        )
    ].copy()

    classical_full[
        "family"
    ] = "Classical"

    classical_full[
        "source_training_regime"
    ] = "Full PURE source"

    pieces.append(
        classical_full
    )

    classical_matched = (
        classical_matched_agg[
            (
                classical_matched_agg["source_dataset"]
                == "PURE"
            )
            & (
                classical_matched_agg["target_dataset"]
                == "UserStories"
            )
        ]
        .copy()
    )

    classical_matched[
        "family"
    ] = "Classical"

    classical_matched[
        "source_training_regime"
    ] = "Size-matched PURE source"

    pieces.append(
        classical_matched
    )

    transformer_full = (
        transformer_agg[
            (
                transformer_agg["setting"]
                == "cross"
            )
            & (
                transformer_agg["train_dataset"]
                == "PURE"
            )
            & (
                transformer_agg["test_dataset"]
                == "UserStories"
            )
        ]
        .copy()
    )

    transformer_full[
        "family"
    ] = "Transformer"

    transformer_full[
        "source_training_regime"
    ] = "Full PURE source"

    pieces.append(
        transformer_full
    )

    transformer_matched = (
        transformer_agg[
            (
                transformer_agg["setting"]
                == "cross_matched"
            )
            & (
                transformer_agg["train_dataset"]
                == "PURE"
            )
            & (
                transformer_agg["test_dataset"]
                == "UserStories"
            )
        ]
        .copy()
    )

    transformer_matched[
        "family"
    ] = "Transformer"

    transformer_matched[
        "source_training_regime"
    ] = "Size-matched PURE source"

    pieces.append(
        transformer_matched
    )

    df = pd.concat(
        pieces,
        ignore_index=True,
        sort=False,
    )

    df["_family_order"] = pd.Categorical(
        df["family"],
        categories=[
            "Classical",
            "Transformer",
        ],
        ordered=True,
    )

    df["_regime_order"] = pd.Categorical(
        df[
            "source_training_regime"
        ],
        categories=[
            "Full PURE source",
            "Size-matched PURE source",
        ],
        ordered=True,
    )

    df["_model_order"] = pd.Categorical(
        df["model"],
        categories=ALL_MODEL_ORDER,
        ordered=True,
    )

    df = (
        df
        .sort_values(
            [
                "_family_order",
                "_model_order",
                "_regime_order",
            ]
        )
        .drop(
            columns=[
                "_family_order",
                "_model_order",
                "_regime_order",
            ]
        )
        .reset_index(drop=True)
    )

    manuscript = pd.DataFrame()

    manuscript["Family"] = (
        df["family"]
    )

    manuscript["Model"] = (
        df["model"]
        .map(model_display)
    )

    manuscript[
        "Source training regime"
    ] = df[
        "source_training_regime"
    ]

    manuscript["Training N"] = [
        mean_sd(
            mean,
            std,
            decimals=0,
        )
        for mean, std in zip(
            df["train_size_mean"],
            df["train_size_std"],
        )
    ]

    for metric in MAIN_METRICS:
        manuscript[
            METRIC_LABELS[metric]
        ] = [
            mean_sd(mean, std)
            for mean, std in zip(
                df[
                    f"{metric}_mean"
                ],
                df[
                    f"{metric}_std"
                ],
            )
        ]

    numeric = (
        numeric_performance_table(
            df,
            id_columns=[
                "family",
                "model",
                "source_training_regime",
            ],
            include_train_size=True,
        )
    )

    write_table(
        "12",
        "source_size_control_pure_to_userstories",
        manuscript,
        numeric,
    )


def build_cue_ablation_table(
    number: str,
    slug: str,
    cue_agg: pd.DataFrame,
    source_dataset: str,
    target_dataset: str,
) -> None:
    df = (
        cue_agg[
            (
                cue_agg["protocol"]
                == "cross_full_source"
            )
            & (
                cue_agg["source_dataset"]
                == source_dataset
            )
            & (
                cue_agg["target_dataset"]
                == target_dataset
            )
        ]
        .copy()
    )

    df["_model_order"] = pd.Categorical(
        df["model"],
        categories=CLASSICAL_ORDER,
        ordered=True,
    )

    df["_variant_order"] = pd.Categorical(
        df["variant"],
        categories=VARIANT_ORDER,
        ordered=True,
    )

    df = (
        df
        .sort_values(
            [
                "_model_order",
                "_variant_order",
            ]
        )
        .drop(
            columns=[
                "_model_order",
                "_variant_order",
            ]
        )
        .reset_index(drop=True)
    )

    manuscript = (
        formatted_performance_table(
            df,
            extra_columns=[
                (
                    "variant",
                    "Variant",
                ),
            ],
        )
    )

    manuscript["Variant"] = (
        manuscript["Variant"]
        .map(
            {
                "original": "Original",
                "masked": "Masked cues",
                "removed": "Removed cues",
            }
        )
    )

    numeric = (
        numeric_performance_table(
            df,
            id_columns=[
                "model",
                "variant",
                "source_dataset",
                "target_dataset",
            ],
        )
    )

    write_table(
        number,
        slug,
        manuscript,
        numeric,
    )


def parse_reason_string(
    value,
) -> list[str]:
    if pd.isna(value):
        return []

    text = str(value).strip()

    if not text:
        return []

    return [
        part.strip()
        for part in text.split(",")
        if part.strip()
    ]


def prepare_reason_seed_rows(
    df: pd.DataFrame,
    family: str,
    model_col: str,
    seed_col: str,
    source_col: str,
    target_col: str,
    true_col: str,
    pred_col: str,
) -> pd.DataFrame:
    work = df[
        [
            model_col,
            seed_col,
            source_col,
            target_col,
            true_col,
            pred_col,
            "reasons",
        ]
    ].copy()

    work = work.rename(
        columns={
            model_col: "model",
            seed_col: "seed",
            source_col: "source_dataset",
            target_col: "target_dataset",
            true_col: "y_true",
            pred_col: "y_pred",
        }
    )

    work["family"] = family

    work["reason"] = (
        work["reasons"]
        .apply(
            parse_reason_string
        )
    )

    work = (
        work
        .explode("reason")
        .reset_index(drop=True)
    )

    work = work[
        work["reason"].notna()
        & (
            work["reason"]
            .astype(str)
            .str.strip()
            != ""
        )
    ].copy()

    if not (
        work["y_true"]
        .astype(int)
        == 1
    ).all():
        raise AssertionError(
            "Reason-level rows contain "
            "y_true != 1."
        )

    grouped = (
        work
        .groupby(
            [
                "family",
                "model",
                "source_dataset",
                "target_dataset",
                "seed",
                "reason",
            ],
            dropna=False,
        )
        .agg(
            support=(
                "y_true",
                "size",
            ),
            detection_recall=(
                "y_pred",
                "mean",
            ),
        )
        .reset_index()
    )

    return grouped


def load_reason_seed_data() -> pd.DataFrame:
    classical = read_csv(
        RAW_DIR
        / "classical_predictions.csv"
    )

    classical = classical[
        classical["protocol"]
        == "cross_full_source"
    ].copy()

    classical_reason = (
        prepare_reason_seed_rows(
            df=classical,
            family="Classical",
            model_col="model",
            seed_col="seed",
            source_col="source_dataset",
            target_col="target_dataset",
            true_col="label",
            pred_col="prediction",
        )
    )

    transformer_files = sorted(
        TRANSFORMER_RUN_DIR.glob(
            "*__cross__*__predictions.csv"
        )
    )

    if len(transformer_files) != 50:
        raise AssertionError(
            "Expected 50 full-source "
            "transformer cross prediction files, "
            f"found {len(transformer_files)}"
        )

    transformer_frames = [
        pd.read_csv(path)
        for path in transformer_files
    ]

    transformer = pd.concat(
        transformer_frames,
        ignore_index=True,
    )

    transformer_reason = (
        prepare_reason_seed_rows(
            df=transformer,
            family="Transformer",
            model_col="model",
            seed_col="seed",
            source_col="train_dataset",
            target_col="test_dataset",
            true_col="y_true",
            pred_col="y_pred",
        )
    )

    return pd.concat(
        [
            classical_reason,
            transformer_reason,
        ],
        ignore_index=True,
    )


def aggregate_reason_results(
    seed_df: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    group_cols = [
        "family",
        "model",
        "source_dataset",
        "target_dataset",
        "reason",
    ]

    for keys, sub in seed_df.groupby(
        group_cols,
        dropna=False,
    ):
        seeds = set(
            sub["seed"]
            .astype(int)
        )

        if seeds != set(SEEDS):
            raise AssertionError(
                f"Reason group {keys} "
                f"does not contain all five seeds."
            )

        supports = (
            sub["support"]
            .astype(int)
            .unique()
        )

        if len(supports) != 1:
            raise AssertionError(
                f"Reason group {keys} "
                f"has inconsistent support: "
                f"{supports.tolist()}"
            )

        recall_values = (
            sub["detection_recall"]
            .astype(float)
        )

        row = dict(
            zip(
                group_cols,
                keys,
            )
        )

        row["support"] = int(
            supports[0]
        )

        row[
            "detection_recall_mean"
        ] = float(
            recall_values.mean()
        )

        row[
            "detection_recall_std"
        ] = float(
            recall_values.std(
                ddof=1
            )
        )

        row[
            "detection_recall_variance"
        ] = float(
            recall_values.var(
                ddof=1
            )
        )

        rows.append(row)

    return pd.DataFrame(rows)


def build_reason_table(
    number: str,
    slug: str,
    reason_summary: pd.DataFrame,
    source_dataset: str,
    target_dataset: str,
) -> None:
    df = reason_summary[
        (
            reason_summary["source_dataset"]
            == source_dataset
        )
        & (
            reason_summary["target_dataset"]
            == target_dataset
        )
    ].copy()

    df["model_display"] = (
        df["model"]
        .map(model_display)
    )

    numeric_all = (
        df
        .sort_values(
            [
                "support",
                "reason",
                "family",
                "model",
            ],
            ascending=[
                False,
                True,
                True,
                True,
            ],
        )
        .reset_index(drop=True)
    )

    main = df[
        df["support"] >= 20
    ].copy()

    main["formatted"] = [
        mean_sd(mean, std)
        for mean, std in zip(
            main[
                "detection_recall_mean"
            ],
            main[
                "detection_recall_std"
            ],
        )
    ]

    pivot = (
        main
        .pivot_table(
            index=[
                "reason",
                "support",
            ],
            columns="model_display",
            values="formatted",
            aggfunc="first",
        )
        .reset_index()
    )

    pivot.columns.name = None

    pivot = (
        pivot
        .sort_values(
            "support",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    pivot = pivot.rename(
        columns={
            "reason": "Reason",
            "support": "Support N",
        }
    )

    desired_models = [
        MODEL_DISPLAY[
            model
        ]
        for model in ALL_MODEL_ORDER
    ]

    desired_columns = [
        "Reason",
        "Support N",
    ]

    desired_columns.extend(
        [
            model
            for model in desired_models
            if model in pivot.columns
        ]
    )

    pivot = pivot[
        desired_columns
    ]

    pivot["Reason"] = (
        pivot["Reason"]
        .astype(str)
        .str.replace(
            "_",
            " ",
            regex=False,
        )
    )

    numeric_main = numeric_all[
        numeric_all["support"] >= 20
    ].copy()

    write_table(
        number,
        slug,
        pivot,
        numeric_main,
    )

    supplementary_path = (
        TABLE_DIR
        / (
            f"table_S_{slug}"
            f"_all_reasons_numeric.csv"
        )
    )

    numeric_all.to_csv(
        supplementary_path,
        index=False,
        encoding="utf-8-sig",
    )

    manifest_rows.append(
        {
            "table": f"S-{number}",
            "slug": (
                f"{slug}_all_reasons"
            ),
            "scope": "supplementary",
            "type": "numeric_companion",
            "file": supplementary_path.name,
        }
    )

    print(
        f"Saved {supplementary_path}"
    )


def build_rule_reason_coverage_table() -> None:
    df = read_csv(
        SUMMARY_DIR
        / "rule_reason_coverage.csv"
    )

    df = (
        df
        .sort_values(
            [
                "dataset",
                "support",
            ],
            ascending=[
                True,
                False,
            ],
        )
        .reset_index(drop=True)
    )

    manuscript = pd.DataFrame(
        {
            "Dataset": df["dataset"],
            "Reason": (
                df["reason"]
                .astype(str)
                .str.replace(
                    "_",
                    " ",
                    regex=False,
                )
            ),
            "Support N": df["support"],
            "Detected by rule": (
                df["detected_by_rule"]
            ),
            "Missed by rule": (
                df["missed_by_rule"]
            ),
            "Detection recall": (
                df["detection_recall"]
                .map(
                    lambda value:
                    f"{value:.3f}"
                )
            ),
        }
    )

    write_table(
        "17",
        "rule_reason_coverage",
        manuscript,
        df,
    )


def build_surface_cue_coverage_table() -> None:
    df = read_csv(
        SUMMARY_DIR
        / "cue_coverage.csv"
    )

    df = (
        df
        .sort_values(
            [
                "dataset",
                "label",
            ]
        )
        .reset_index(drop=True)
    )

    manuscript = pd.DataFrame(
        {
            "Dataset": df["dataset"],
            "Class": df["label"].map(
                {
                    0: "Clear",
                    1: "Indicator-positive",
                }
            ),
            "Support N": df["support"],
            "With frozen surface cue": (
                df["with_surface_cue"]
            ),
            "Cue coverage": (
                df["cue_rate"]
                .map(
                    lambda value:
                    f"{value:.3f}"
                )
            ),
        }
    )

    write_table(
        "18",
        "surface_cue_coverage",
        manuscript,
        df,
    )


def write_manifest() -> None:
    manifest = pd.DataFrame(
        manifest_rows
    )

    manifest_path = (
        TABLE_DIR
        / "table_manifest.csv"
    )

    manifest.to_csv(
        manifest_path,
        index=False,
        encoding="utf-8-sig",
    )

    print()
    print(
        f"Saved {manifest_path}"
    )

def build_cross_variability_table(
    number: str,
    slug: str,
    classical_agg: pd.DataFrame,
    transformer_agg: pd.DataFrame,
    source_dataset: str,
    target_dataset: str,
) -> None:
    classical = classical_agg[
        (
            classical_agg["protocol"]
            == "cross_full_source"
        )
        & (
            classical_agg["source_dataset"]
            == source_dataset
        )
        & (
            classical_agg["target_dataset"]
            == target_dataset
        )
    ].copy()

    classical["family"] = "Classical"

    transformers = transformer_agg[
        (
            transformer_agg["setting"]
            == "cross"
        )
        & (
            transformer_agg["train_dataset"]
            == source_dataset
        )
        & (
            transformer_agg["test_dataset"]
            == target_dataset
        )
    ].copy()

    transformers["family"] = "Transformer"

    df = pd.concat(
        [classical, transformers],
        ignore_index=True,
        sort=False,
    )

    if len(df) != 10:
        raise AssertionError(
            f"{slug}: expected 10 models, got {len(df)}"
        )

    df["_family_order"] = pd.Categorical(
        df["family"],
        categories=[
            "Classical",
            "Transformer",
        ],
        ordered=True,
    )

    df["_model_order"] = pd.Categorical(
        df["model"],
        categories=ALL_MODEL_ORDER,
        ordered=True,
    )

    df = (
        df
        .sort_values(
            [
                "_family_order",
                "_model_order",
            ]
        )
        .drop(
            columns=[
                "_family_order",
                "_model_order",
            ]
        )
        .reset_index(drop=True)
    )

    table = pd.DataFrame(
        {
            "Family": df["family"],
            "Model": (
                df["model"]
                .map(model_display)
            ),
            "F1 mean": df["f1_mean"],
            "F1 SD": df["f1_std"],
            "F1 variance": df["f1_var"],
            "MCC mean": df["mcc_mean"],
            "MCC SD": df["mcc_std"],
            "MCC variance": df["mcc_var"],
            "Cohen's κ mean": (
                df["cohen_kappa_mean"]
            ),
            "Cohen's κ SD": (
                df["cohen_kappa_std"]
            ),
            "Cohen's κ variance": (
                df["cohen_kappa_var"]
            ),
        }
    )

    numeric_columns = [
        "F1 mean",
        "F1 SD",
        "F1 variance",
        "MCC mean",
        "MCC SD",
        "MCC variance",
        "Cohen's κ mean",
        "Cohen's κ SD",
        "Cohen's κ variance",
    ]

    table[numeric_columns] = (
        table[numeric_columns]
        .astype(float)
        .round(6)
    )

    write_table(
        number,
        slug,
        table,
    )


def build_size_control_variability_table(
    classical_agg: pd.DataFrame,
    classical_matched_agg: pd.DataFrame,
    transformer_agg: pd.DataFrame,
) -> None:
    pieces = []

    classical_full = classical_agg[
        (
            classical_agg["protocol"]
            == "cross_full_source"
        )
        & (
            classical_agg["source_dataset"]
            == "PURE"
        )
        & (
            classical_agg["target_dataset"]
            == "UserStories"
        )
    ].copy()

    classical_full["family"] = "Classical"
    classical_full["regime"] = "Full PURE source"
    pieces.append(classical_full)

    classical_matched = classical_matched_agg[
        (
            classical_matched_agg["source_dataset"]
            == "PURE"
        )
        & (
            classical_matched_agg["target_dataset"]
            == "UserStories"
        )
    ].copy()

    classical_matched["family"] = "Classical"
    classical_matched["regime"] = (
        "Size-matched PURE source"
    )
    pieces.append(classical_matched)

    transformer_full = transformer_agg[
        (
            transformer_agg["setting"]
            == "cross"
        )
        & (
            transformer_agg["train_dataset"]
            == "PURE"
        )
        & (
            transformer_agg["test_dataset"]
            == "UserStories"
        )
    ].copy()

    transformer_full["family"] = "Transformer"
    transformer_full["regime"] = "Full PURE source"
    pieces.append(transformer_full)

    transformer_matched = transformer_agg[
        (
            transformer_agg["setting"]
            == "cross_matched"
        )
        & (
            transformer_agg["train_dataset"]
            == "PURE"
        )
        & (
            transformer_agg["test_dataset"]
            == "UserStories"
        )
    ].copy()

    transformer_matched["family"] = "Transformer"
    transformer_matched["regime"] = (
        "Size-matched PURE source"
    )
    pieces.append(transformer_matched)

    df = pd.concat(
        pieces,
        ignore_index=True,
        sort=False,
    )

    if len(df) != 20:
        raise AssertionError(
            "Table 21: expected 20 rows "
            f"(10 models × 2 regimes), got {len(df)}"
        )

    df["_family_order"] = pd.Categorical(
        df["family"],
        categories=[
            "Classical",
            "Transformer",
        ],
        ordered=True,
    )

    df["_model_order"] = pd.Categorical(
        df["model"],
        categories=ALL_MODEL_ORDER,
        ordered=True,
    )

    df["_regime_order"] = pd.Categorical(
        df["regime"],
        categories=[
            "Full PURE source",
            "Size-matched PURE source",
        ],
        ordered=True,
    )

    df = (
        df
        .sort_values(
            [
                "_family_order",
                "_model_order",
                "_regime_order",
            ]
        )
        .drop(
            columns=[
                "_family_order",
                "_model_order",
                "_regime_order",
            ]
        )
        .reset_index(drop=True)
    )

    table = pd.DataFrame(
        {
            "Family": df["family"],
            "Model": (
                df["model"]
                .map(model_display)
            ),
            "Source regime": df["regime"],
            "F1 mean": df["f1_mean"],
            "F1 SD": df["f1_std"],
            "F1 variance": df["f1_var"],
            "MCC mean": df["mcc_mean"],
            "MCC SD": df["mcc_std"],
            "MCC variance": df["mcc_var"],
            "Cohen's κ mean": (
                df["cohen_kappa_mean"]
            ),
            "Cohen's κ SD": (
                df["cohen_kappa_std"]
            ),
            "Cohen's κ variance": (
                df["cohen_kappa_var"]
            ),
        }
    )

    numeric_columns = [
        "F1 mean",
        "F1 SD",
        "F1 variance",
        "MCC mean",
        "MCC SD",
        "MCC variance",
        "Cohen's κ mean",
        "Cohen's κ SD",
        "Cohen's κ variance",
    ]

    table[numeric_columns] = (
        table[numeric_columns]
        .astype(float)
        .round(6)
    )

    write_table(
        "21",
        "source_size_control_variability",
        table,
    )

def main() -> None:
    print("=" * 80)
    print("BUILDING FINAL MANUSCRIPT TABLES")
    print("=" * 80)

    TABLE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    build_dataset_table()
    build_configuration_table()

    (
        classical_agg,
        classical_matched_agg,
        rule_agg,
        cue_agg,
        transformer_agg,
    ) = load_aggregated_results()

    build_main_performance_tables(
        classical_agg=classical_agg,
        transformer_agg=transformer_agg,
    )

    build_rule_table(
        rule_agg
    )

    build_size_control_table(
        classical_agg=classical_agg,
        classical_matched_agg=(
            classical_matched_agg
        ),
        transformer_agg=transformer_agg,
    )

    build_cue_ablation_table(
        number="13",
        slug=(
            "cue_ablation_pure_to_userstories"
        ),
        cue_agg=cue_agg,
        source_dataset="PURE",
        target_dataset="UserStories",
    )

    build_cue_ablation_table(
        number="14",
        slug=(
            "cue_ablation_userstories_to_pure"
        ),
        cue_agg=cue_agg,
        source_dataset="UserStories",
        target_dataset="PURE",
    )

    reason_seed_data = (
        load_reason_seed_data()
    )

    reason_summary = (
        aggregate_reason_results(
            reason_seed_data
        )
    )

    build_reason_table(
        number="15",
        slug=(
            "reason_detection_pure_to_userstories"
        ),
        reason_summary=reason_summary,
        source_dataset="PURE",
        target_dataset="UserStories",
    )

    build_reason_table(
        number="16",
        slug=(
            "reason_detection_userstories_to_pure"
        ),
        reason_summary=reason_summary,
        source_dataset="UserStories",
        target_dataset="PURE",
    )

    build_rule_reason_coverage_table()
    build_surface_cue_coverage_table()

    build_cross_variability_table(
        number="19",
        slug="cross_variability_pure_to_userstories",
        classical_agg=classical_agg,
        transformer_agg=transformer_agg,
        source_dataset="PURE",
        target_dataset="UserStories",
    )

    build_cross_variability_table(
        number="20",
        slug="cross_variability_userstories_to_pure",
        classical_agg=classical_agg,
        transformer_agg=transformer_agg,
        source_dataset="UserStories",
        target_dataset="PURE",
    )

    build_size_control_variability_table(
        classical_agg=classical_agg,
        classical_matched_agg=classical_matched_agg,
        transformer_agg=transformer_agg,
    )

    write_manifest()

    print()
    print("=" * 80)
    print("FINAL TABLE GENERATION COMPLETED")
    print("=" * 80)

    main_tables = [
        row
        for row in manifest_rows
        if (
            row["scope"] == "main"
            and row["type"] == "manuscript"
        )
    ]

    print(
        "Main manuscript tables:",
        len(main_tables),
    )

    print(
        "Output directory:",
        TABLE_DIR,
    )

    print()
    print(
        "Performance tables use mean ± sample SD."
    )

    print(
        "Numeric companion tables contain "
        "mean, sample SD, and sample variance."
    )

    print(
        "Reason-level tables report detection "
        "recall and support, not reason-level F1."
    )


if __name__ == "__main__":
    main()