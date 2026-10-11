from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import GroupShuffleSplit


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


from src.config import (
    PURE_CLEAN_PATH,
    SIZE_MATCHED_SPLITS_DIR,
    TRANSFORMER_SIZE_MATCHED_SPLITS_DIR,
    RESULTS_SUMMARY_DIR,
    INSTANCE_ID_COL,
    SOURCE_COL,
    LABEL_COL,
    SEEDS,
    ensure_project_dirs,
)


VALIDATION_FRACTION = 0.10
N_CANDIDATES = 5000


def normalize_selected_column(
    series: pd.Series,
) -> pd.Series:
    if series.dtype == bool:
        return series

    return (
        series
        .astype(str)
        .str.strip()
        .str.lower()
        .eq("true")
    )


def choose_train_validation_split(
    subset_df: pd.DataFrame,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Create a source-disjoint ~90/10 train/validation split
    inside the already size-matched PURE subset.

    Candidate selection uses only source-corpus information:
    - validation row fraction
    - validation positive-class rate

    No UserStories target labels are used.
    """

    groups = (
        subset_df[SOURCE_COL]
        .astype(str)
        .values
    )

    splitter = GroupShuffleSplit(
        n_splits=N_CANDIDATES,
        test_size=VALIDATION_FRACTION,
        random_state=seed + 4000,
    )

    target_positive_rate = float(
        subset_df[LABEL_COL].mean()
    )

    best_score = None
    best_train_idx = None
    best_validation_idx = None

    dummy_x = np.zeros(
        len(subset_df)
    )

    for train_idx, validation_idx in splitter.split(
        dummy_x,
        groups=groups,
    ):
        validation_df = subset_df.iloc[
            validation_idx
        ]

        validation_fraction = (
            len(validation_df)
            / len(subset_df)
        )

        validation_positive_rate = float(
            validation_df[LABEL_COL].mean()
        )

        size_error = abs(
            validation_fraction
            - VALIDATION_FRACTION
        )

        class_balance_error = abs(
            validation_positive_rate
            - target_positive_rate
        )

        score = (
            size_error
            + 0.5 * class_balance_error
        )

        if (
            best_score is None
            or score < best_score
        ):
            best_score = score
            best_train_idx = train_idx
            best_validation_idx = (
                validation_idx
            )

    if best_train_idx is None:
        raise RuntimeError(
            f"Could not create split for seed {seed}."
        )

    return (
        best_train_idx,
        best_validation_idx,
    )


def validate_split(
    subset_df: pd.DataFrame,
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
) -> None:
    subset_ids = set(
        subset_df[INSTANCE_ID_COL]
    )

    train_ids = set(
        train_df[INSTANCE_ID_COL]
    )

    validation_ids = set(
        validation_df[INSTANCE_ID_COL]
    )

    if train_ids & validation_ids:
        raise AssertionError(
            "Train/validation instance overlap detected."
        )

    if (
        train_ids | validation_ids
    ) != subset_ids:
        raise AssertionError(
            "Train + validation do not exactly cover "
            "the selected matched subset."
        )

    train_sources = set(
        train_df[SOURCE_COL]
    )

    validation_sources = set(
        validation_df[SOURCE_COL]
    )

    if train_sources & validation_sources:
        raise AssertionError(
            "Train/validation source overlap detected."
        )


def main() -> None:
    ensure_project_dirs()

    TRANSFORMER_SIZE_MATCHED_SPLITS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    pure = pd.read_csv(
        PURE_CLEAN_PATH
    )

    summary_rows = []

    print("=" * 80)
    print(
        "STEP 14: CREATE TRANSFORMER "
        "SIZE-MATCHED SPLITS"
    )
    print("=" * 80)

    for seed in SEEDS:
        classical_manifest_path = (
            SIZE_MATCHED_SPLITS_DIR
            / f"PURE_matched_seed{seed}.csv"
        )

        if not classical_manifest_path.exists():
            raise FileNotFoundError(
                classical_manifest_path
            )

        classical_manifest = pd.read_csv(
            classical_manifest_path
        )

        required_columns = {
            INSTANCE_ID_COL,
            SOURCE_COL,
            LABEL_COL,
            "selected",
        }

        missing = (
            required_columns
            - set(classical_manifest.columns)
        )

        if missing:
            raise ValueError(
                f"{classical_manifest_path} "
                f"missing columns: {sorted(missing)}"
            )

        selected_mask = (
            normalize_selected_column(
                classical_manifest[
                    "selected"
                ]
            )
        )

        selected_manifest = (
            classical_manifest.loc[
                selected_mask
            ]
            .copy()
            .reset_index(drop=True)
        )

        selected_ids = set(
            selected_manifest[
                INSTANCE_ID_COL
            ]
        )

        subset_df = (
            pure[
                pure[INSTANCE_ID_COL]
                .isin(selected_ids)
            ]
            .copy()
            .reset_index(drop=True)
        )

        if (
            len(subset_df)
            != len(selected_manifest)
        ):
            raise AssertionError(
                f"Seed {seed}: selected manifest "
                "does not match PURE clean data."
            )

        if set(
            subset_df[INSTANCE_ID_COL]
        ) != selected_ids:
            raise AssertionError(
                f"Seed {seed}: instance-ID mismatch."
            )

        (
            train_idx,
            validation_idx,
        ) = choose_train_validation_split(
            subset_df=subset_df,
            seed=seed,
        )

        train_df = (
            subset_df
            .iloc[train_idx]
            .copy()
            .reset_index(drop=True)
        )

        validation_df = (
            subset_df
            .iloc[validation_idx]
            .copy()
            .reset_index(drop=True)
        )

        validate_split(
            subset_df=subset_df,
            train_df=train_df,
            validation_df=validation_df,
        )

        split_map = {}

        for instance_id in train_df[
            INSTANCE_ID_COL
        ]:
            split_map[
                instance_id
            ] = "train"

        for instance_id in validation_df[
            INSTANCE_ID_COL
        ]:
            split_map[
                instance_id
            ] = "validation"

        output_manifest = (
            selected_manifest[
                [
                    INSTANCE_ID_COL,
                    SOURCE_COL,
                    LABEL_COL,
                ]
            ]
            .copy()
        )

        output_manifest[
            "split"
        ] = output_manifest[
            INSTANCE_ID_COL
        ].map(
            split_map
        )

        output_manifest[
            "seed"
        ] = seed

        output_manifest[
            "matched_subset_size"
        ] = len(
            subset_df
        )

        output_path = (
            TRANSFORMER_SIZE_MATCHED_SPLITS_DIR
            / (
                "PURE_matched_source_"
                f"seed{seed}.csv"
            )
        )

        output_manifest.to_csv(
            output_path,
            index=False,
        )

        subset_positive_rate = float(
            subset_df[LABEL_COL].mean()
        )

        train_positive_rate = float(
            train_df[LABEL_COL].mean()
        )

        validation_positive_rate = float(
            validation_df[
                LABEL_COL
            ].mean()
        )

        validation_fraction = (
            len(validation_df)
            / len(subset_df)
        )

        summary_rows.append(
            {
                "seed":
                    seed,

                "matched_subset_rows":
                    len(subset_df),

                "matched_subset_sources":
                    subset_df[
                        SOURCE_COL
                    ].nunique(),

                "matched_subset_positive_rate":
                    subset_positive_rate,

                "train_rows":
                    len(train_df),

                "train_sources":
                    train_df[
                        SOURCE_COL
                    ].nunique(),

                "train_positive_rate":
                    train_positive_rate,

                "validation_rows":
                    len(validation_df),

                "validation_sources":
                    validation_df[
                        SOURCE_COL
                    ].nunique(),

                "validation_positive_rate":
                    validation_positive_rate,

                "validation_fraction":
                    validation_fraction,

                "source_overlap":
                    len(
                        set(
                            train_df[
                                SOURCE_COL
                            ]
                        )
                        & set(
                            validation_df[
                                SOURCE_COL
                            ]
                        )
                    ),
            }
        )

        print()
        print(
            f"Seed {seed}"
        )

        print(
            f"  matched subset: "
            f"{len(subset_df)} rows, "
            f"{subset_df[SOURCE_COL].nunique()} sources, "
            f"positive_rate="
            f"{subset_positive_rate:.4f}"
        )

        print(
            f"  train:          "
            f"{len(train_df)} rows, "
            f"{train_df[SOURCE_COL].nunique()} sources, "
            f"positive_rate="
            f"{train_positive_rate:.4f}"
        )

        print(
            f"  validation:     "
            f"{len(validation_df)} rows, "
            f"{validation_df[SOURCE_COL].nunique()} sources, "
            f"positive_rate="
            f"{validation_positive_rate:.4f}"
        )

        print(
            f"  val fraction:   "
            f"{validation_fraction:.4f}"
        )

        print(
            f"  source overlap: 0"
        )

        print(
            f"  saved: {output_path}"
        )

    summary_df = pd.DataFrame(
        summary_rows
    )

    summary_path = (
        RESULTS_SUMMARY_DIR
        / (
            "transformer_size_matched_"
            "split_summary.csv"
        )
    )

    summary_df.to_csv(
        summary_path,
        index=False,
    )

    print()
    print("=" * 80)
    print(
        "TRANSFORMER SIZE-MATCHED "
        "SPLITS VALIDATED"
    )
    print("=" * 80)

    print(
        f"\nSummary saved to:\n"
        f"  {summary_path}"
    )


if __name__ == "__main__":
    main()