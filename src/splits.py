from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd

from sklearn.model_selection import GroupShuffleSplit

from src.config import (
    INSTANCE_ID_COL,
    SOURCE_COL,
    LABEL_COL,
)


# ============================================================
# Internal split search
# ============================================================

def _split_score(
    df: pd.DataFrame,
    test_indices: np.ndarray,
    target_fraction: float,
) -> float:
    """
    Score a candidate group-disjoint split.

    Lower is better.

    We jointly prefer:
    1. a row proportion close to target_fraction
    2. a positive-class rate close to the full candidate dataset
    """
    candidate = df.iloc[test_indices]

    actual_fraction = len(candidate) / len(df)

    overall_positive_rate = df[LABEL_COL].mean()
    candidate_positive_rate = candidate[LABEL_COL].mean()

    size_error = abs(
        actual_fraction - target_fraction
    )

    balance_error = abs(
        candidate_positive_rate
        - overall_positive_rate
    )

    # Size accuracy is slightly more important than
    # class-rate accuracy, but both matter.
    return size_error + 0.5 * balance_error


def find_best_group_split(
    df: pd.DataFrame,
    test_fraction: float,
    seed: int,
    n_candidates: int = 2000,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Search many deterministic GroupShuffleSplit candidates and
    select the one that best matches the requested row fraction
    and class balance.

    Groups are source_file values, so no source can appear
    in both partitions.
    """
    if not 0 < test_fraction < 1:
        raise ValueError(
            "test_fraction must be between 0 and 1."
        )

    groups = df[SOURCE_COL].values

    splitter = GroupShuffleSplit(
        n_splits=n_candidates,
        test_size=test_fraction,
        random_state=seed,
    )

    best_train_idx = None
    best_test_idx = None
    best_score = float("inf")

    dummy_x = np.zeros(len(df))
    y = df[LABEL_COL].values

    for train_idx, test_idx in splitter.split(
        dummy_x,
        y,
        groups=groups,
    ):
        score = _split_score(
            df=df,
            test_indices=test_idx,
            target_fraction=test_fraction,
        )

        if score < best_score:
            best_score = score
            best_train_idx = train_idx
            best_test_idx = test_idx

    if best_train_idx is None or best_test_idx is None:
        raise RuntimeError(
            "Unable to construct a group-disjoint split."
        )

    return best_train_idx, best_test_idx


# ============================================================
# Intra-dataset split
# ============================================================

def create_intra_split(
    df: pd.DataFrame,
    seed: int,
) -> pd.DataFrame:
    """
    Create approximately 70/10/20 train/validation/test splits.

    All partitions are source-disjoint.

    Classical models will later use:
        train + validation -> training
        test               -> testing

    Transformer models will use:
        train      -> training
        validation -> model selection
        test       -> final evaluation
    """

    df = df.reset_index(drop=True).copy()

    # --------------------------------------------------------
    # Step 1: approximately 20% source-disjoint test set
    # --------------------------------------------------------

    trainval_idx, test_idx = find_best_group_split(
        df=df,
        test_fraction=0.20,
        seed=seed,
    )

    trainval_df = df.iloc[
        trainval_idx
    ].reset_index(drop=True)

    test_ids = set(
        df.iloc[test_idx][INSTANCE_ID_COL]
    )

    # --------------------------------------------------------
    # Step 2: validation is approximately 10% of total.
    #
    # Since trainval contains about 80% of the dataset,
    # validation must be approximately 12.5% of trainval:
    #
    # 0.10 / 0.80 = 0.125
    # --------------------------------------------------------

    train_idx_local, val_idx_local = (
        find_best_group_split(
            df=trainval_df,
            test_fraction=0.125,
            seed=seed + 1000,
        )
    )

    train_ids = set(
        trainval_df.iloc[
            train_idx_local
        ][INSTANCE_ID_COL]
    )

    val_ids = set(
        trainval_df.iloc[
            val_idx_local
        ][INSTANCE_ID_COL]
    )

    manifest = df[
        [
            INSTANCE_ID_COL,
            SOURCE_COL,
            LABEL_COL,
        ]
    ].copy()

    def assign_split(instance_id: str) -> str:
        if instance_id in train_ids:
            return "train"

        if instance_id in val_ids:
            return "validation"

        if instance_id in test_ids:
            return "test"

        raise RuntimeError(
            f"Instance not assigned: {instance_id}"
        )

    manifest["split"] = manifest[
        INSTANCE_ID_COL
    ].apply(assign_split)

    manifest["seed"] = seed

    return manifest


# ============================================================
# Cross-dataset source split
# ============================================================

def create_cross_source_split(
    source_df: pd.DataFrame,
    seed: int,
) -> pd.DataFrame:
    """
    Create approximately 90/10 train/validation split for a
    source dataset used in cross-dataset transformer evaluation.

    The complete target dataset remains untouched and is used
    only for final testing.
    """

    source_df = source_df.reset_index(
        drop=True
    ).copy()

    train_idx, val_idx = find_best_group_split(
        df=source_df,
        test_fraction=0.10,
        seed=seed + 2000,
    )

    train_ids = set(
        source_df.iloc[
            train_idx
        ][INSTANCE_ID_COL]
    )

    val_ids = set(
        source_df.iloc[
            val_idx
        ][INSTANCE_ID_COL]
    )

    manifest = source_df[
        [
            INSTANCE_ID_COL,
            SOURCE_COL,
            LABEL_COL,
        ]
    ].copy()

    manifest["split"] = manifest[
        INSTANCE_ID_COL
    ].apply(
        lambda instance_id:
        "train"
        if instance_id in train_ids
        else "validation"
    )

    manifest["seed"] = seed

    return manifest


# ============================================================
# Split validation
# ============================================================

def validate_split_manifest(
    manifest: pd.DataFrame,
    expected_splits: set[str],
) -> None:
    """
    Verify that a split manifest has no instance overlap and
    no source_file overlap between partitions.
    """

    observed_splits = set(
        manifest["split"].unique()
    )

    if observed_splits != expected_splits:
        raise AssertionError(
            f"Unexpected split names. "
            f"Expected {expected_splits}, "
            f"got {observed_splits}."
        )

    if not manifest[INSTANCE_ID_COL].is_unique:
        raise AssertionError(
            "Instance IDs are duplicated in split manifest."
        )

    # Every source must belong to exactly one partition.
    source_split_counts = (
        manifest.groupby(SOURCE_COL)["split"]
        .nunique()
    )

    leaking_sources = source_split_counts[
        source_split_counts > 1
    ]

    if not leaking_sources.empty:
        raise AssertionError(
            "Source leakage detected:\n"
            f"{leaking_sources}"
        )


def summarize_split_manifest(
    manifest: pd.DataFrame,
    dataset_name: str,
    split_type: str,
) -> pd.DataFrame:
    """
    Produce split-level statistics for audit and manuscript use.
    """

    rows = []

    total_rows = len(manifest)

    for split_name, sub in manifest.groupby(
        "split"
    ):
        rows.append(
            {
                "dataset": dataset_name,
                "split_type": split_type,
                "seed": int(
                    manifest["seed"].iloc[0]
                ),
                "split": split_name,
                "rows": len(sub),
                "row_fraction":
                    len(sub) / total_rows,
                "sources":
                    sub[SOURCE_COL].nunique(),
                "label_0":
                    int((sub[LABEL_COL] == 0).sum()),
                "label_1":
                    int((sub[LABEL_COL] == 1).sum()),
                "positive_rate":
                    float(sub[LABEL_COL].mean()),
            }
        )

    return pd.DataFrame(rows)


def create_size_matched_source_subset(
    source_df: pd.DataFrame,
    target_size: int,
    seed: int,
) -> pd.DataFrame:
    """
    Select a source-disjoint subset whose number of instances is
    as close as possible to target_size.

    Entire source_file groups are selected together.

    The subset preserves the class distribution of the source
    corpus as closely as possible.
    """

    if target_size <= 0:
        raise ValueError(
            "target_size must be positive."
        )

    if target_size >= len(source_df):
        raise ValueError(
            "target_size must be smaller than source dataset."
        )

    target_fraction = (
        target_size / len(source_df)
    )

    _, selected_idx = find_best_group_split(
        df=source_df,
        test_fraction=target_fraction,
        seed=seed + 3000,
        n_candidates=5000,
    )

    selected_ids = set(
        source_df.iloc[selected_idx][INSTANCE_ID_COL]
    )

    manifest = source_df[
        [
            INSTANCE_ID_COL,
            SOURCE_COL,
            LABEL_COL,
        ]
    ].copy()

    manifest["selected"] = manifest[
        INSTANCE_ID_COL
    ].isin(selected_ids)

    manifest["seed"] = seed
    manifest["requested_size"] = target_size
    manifest["actual_size"] = int(
        manifest["selected"].sum()
    )

    return manifest