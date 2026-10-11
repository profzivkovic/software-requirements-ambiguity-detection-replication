from __future__ import annotations

import re
import unicodedata
from typing import Dict, Tuple

import pandas as pd

from src.config import (
    SOURCE_COL,
    TEXT_COL,
    LABEL_COL,
    REASONS_COL,
    INSTANCE_ID_COL,
    REQUIRED_COLUMNS,
)


# ============================================================
# Text / reason normalization
# ============================================================

def normalize_text(text: object) -> str:
    """
    Conservative normalization used only for duplicate detection.

    We intentionally do NOT remove punctuation or words because we do
    not want to collapse requirements that are linguistically different.

    Operations:
    - convert to string
    - Unicode NFKC normalization
    - lowercase
    - collapse repeated whitespace
    - strip leading/trailing whitespace
    """
    if pd.isna(text):
        return ""

    text = str(text)
    text = unicodedata.normalize("NFKC", text)
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def normalize_reasons(value: object) -> str:
    """
    Normalize a comma-separated reason list so that ordering differences
    do not look like annotation conflicts.

    Example:
        'weak_modal, vague_term'
        'vague_term, weak_modal'

    become the same normalized value.
    """
    if pd.isna(value):
        return ""

    value = str(value).strip()

    if not value:
        return ""

    reasons = [
        item.strip()
        for item in value.split(",")
        if item.strip()
    ]

    return ",".join(sorted(set(reasons)))


# ============================================================
# Loading / validation
# ============================================================

def load_raw_dataset(
    path,
    dataset_name: str,
) -> pd.DataFrame:
    """
    Load one raw dataset without modifying the source file.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"{dataset_name}: file does not exist: {path}"
        )

    df = pd.read_csv(path)

    missing_columns = [
        col for col in REQUIRED_COLUMNS
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"{dataset_name}: missing required columns: "
            f"{missing_columns}"
        )

    return df.copy()


def prepare_audit_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add temporary normalized columns used only for auditing.
    """
    out = df.copy()

    out["_text_norm"] = out[TEXT_COL].apply(normalize_text)
    out["_reasons_norm"] = out[REASONS_COL].apply(
        normalize_reasons
    )

    return out


# ============================================================
# Dataset-level audit
# ============================================================

def audit_dataset(
    df: pd.DataFrame,
    dataset_name: str,
) -> Dict[str, object]:
    """
    Produce one summary row describing the dataset.
    """
    work = prepare_audit_columns(df)

    numeric_labels = pd.to_numeric(
        work[LABEL_COL],
        errors="coerce",
    )

    invalid_label_mask = (
        numeric_labels.isna()
        | ~numeric_labels.isin([0, 1])
    )

    label_0 = int((numeric_labels == 0).sum())
    label_1 = int((numeric_labels == 1).sum())

    # Rows beyond the first occurrence of a duplicated normalized text.
    duplicated_extra_rows = int(
        work.duplicated(
            subset=["_text_norm"],
            keep="first",
        ).sum()
    )

    duplicated_text_groups = int(
        (
            work.groupby("_text_norm")
            .size()
            .gt(1)
        ).sum()
    )

    # Same normalized text occurring in more than one source document/project.
    cross_source_groups = int(
        (
            work.groupby("_text_norm")[SOURCE_COL]
            .nunique()
            .gt(1)
        ).sum()
    )

    # Same normalized text but inconsistent labels.
    conflicting_label_groups = int(
        (
            work.groupby("_text_norm")[LABEL_COL]
            .nunique(dropna=False)
            .gt(1)
        ).sum()
    )

    # Same normalized text but inconsistent reason annotations.
    conflicting_reason_groups = int(
        (
            work.groupby("_text_norm")["_reasons_norm"]
            .nunique(dropna=False)
            .gt(1)
        ).sum()
    )

    summary = {
        "dataset": dataset_name,
        "rows": len(work),
        "unique_sources": work[SOURCE_COL].nunique(
            dropna=True
        ),
        "label_0": label_0,
        "label_1": label_1,
        "positive_rate": (
            label_1 / len(work)
            if len(work) > 0
            else float("nan")
        ),
        "missing_source": int(
            work[SOURCE_COL].isna().sum()
        ),
        "missing_text": int(
            work[TEXT_COL].isna().sum()
        ),
        "empty_normalized_text": int(
            (work["_text_norm"] == "").sum()
        ),
        "missing_label": int(
            work[LABEL_COL].isna().sum()
        ),
        "invalid_label": int(
            invalid_label_mask.sum()
        ),
        "missing_reasons": int(
            work[REASONS_COL].isna().sum()
        ),
        "duplicated_extra_rows": duplicated_extra_rows,
        "duplicated_text_groups": duplicated_text_groups,
        "cross_source_duplicate_groups": cross_source_groups,
        "conflicting_label_groups": conflicting_label_groups,
        "conflicting_reason_groups": conflicting_reason_groups,
    }

    return summary


# ============================================================
# Duplicate reports
# ============================================================

def build_duplicate_report(
    df: pd.DataFrame,
    dataset_name: str,
) -> pd.DataFrame:
    """
    Return every row whose normalized requirement text appears more
    than once.

    This report does NOT delete anything.
    """
    work = prepare_audit_columns(df)

    counts = (
        work.groupby("_text_norm")
        .size()
        .rename("duplicate_count")
    )

    duplicate_texts = counts[counts > 1].index

    report = work[
        work["_text_norm"].isin(duplicate_texts)
    ].copy()

    if report.empty:
        return pd.DataFrame(
            columns=[
                "dataset",
                "duplicate_count",
                SOURCE_COL,
                TEXT_COL,
                LABEL_COL,
                REASONS_COL,
                "_text_norm",
                "_reasons_norm",
            ]
        )

    report = report.merge(
        counts,
        left_on="_text_norm",
        right_index=True,
        how="left",
    )

    report.insert(0, "dataset", dataset_name)

    report = report[
        [
            "dataset",
            "duplicate_count",
            SOURCE_COL,
            TEXT_COL,
            LABEL_COL,
            REASONS_COL,
            "_text_norm",
            "_reasons_norm",
        ]
    ]

    return report.sort_values(
        ["duplicate_count", "_text_norm", SOURCE_COL],
        ascending=[False, True, True],
    ).reset_index(drop=True)


def build_cross_source_duplicate_report(
    df: pd.DataFrame,
    dataset_name: str,
) -> pd.DataFrame:
    """
    Find normalized requirement texts that occur in more than one
    source document/project.
    """
    work = prepare_audit_columns(df)

    source_counts = (
        work.groupby("_text_norm")[SOURCE_COL]
        .nunique()
        .rename("source_count")
    )

    cross_source_texts = source_counts[
        source_counts > 1
    ].index

    report = work[
        work["_text_norm"].isin(cross_source_texts)
    ].copy()

    if report.empty:
        return pd.DataFrame(
            columns=[
                "dataset",
                "source_count",
                SOURCE_COL,
                TEXT_COL,
                LABEL_COL,
                REASONS_COL,
                "_text_norm",
                "_reasons_norm",
            ]
        )

    report = report.merge(
        source_counts,
        left_on="_text_norm",
        right_index=True,
        how="left",
    )

    report.insert(0, "dataset", dataset_name)

    return report[
        [
            "dataset",
            "source_count",
            SOURCE_COL,
            TEXT_COL,
            LABEL_COL,
            REASONS_COL,
            "_text_norm",
            "_reasons_norm",
        ]
    ].sort_values(
        ["source_count", "_text_norm", SOURCE_COL],
        ascending=[False, True, True],
    ).reset_index(drop=True)


# ============================================================
# Annotation consistency
# ============================================================

def build_conflict_report(
    df: pd.DataFrame,
    dataset_name: str,
) -> pd.DataFrame:
    """
    Find duplicated normalized texts that have conflicting labels
    and/or conflicting reason annotations.
    """
    work = prepare_audit_columns(df)

    grouped = work.groupby("_text_norm")

    label_conflict = grouped[LABEL_COL].nunique(
        dropna=False
    ) > 1

    reason_conflict = grouped["_reasons_norm"].nunique(
        dropna=False
    ) > 1

    conflict_texts = set(
        label_conflict[label_conflict].index
    ) | set(
        reason_conflict[reason_conflict].index
    )

    if not conflict_texts:
        return pd.DataFrame(
            columns=[
                "dataset",
                "label_conflict",
                "reason_conflict",
                SOURCE_COL,
                TEXT_COL,
                LABEL_COL,
                REASONS_COL,
                "_text_norm",
                "_reasons_norm",
            ]
        )

    report = work[
        work["_text_norm"].isin(conflict_texts)
    ].copy()

    report["label_conflict"] = report[
        "_text_norm"
    ].map(label_conflict)

    report["reason_conflict"] = report[
        "_text_norm"
    ].map(reason_conflict)

    report.insert(0, "dataset", dataset_name)

    return report[
        [
            "dataset",
            "label_conflict",
            "reason_conflict",
            SOURCE_COL,
            TEXT_COL,
            LABEL_COL,
            REASONS_COL,
            "_text_norm",
            "_reasons_norm",
        ]
    ].sort_values(
        ["_text_norm", SOURCE_COL]
    ).reset_index(drop=True)


# ============================================================
# Reason support
# ============================================================

def build_reason_support(
    df: pd.DataFrame,
    dataset_name: str,
) -> pd.DataFrame:
    """
    Count the number of instances associated with each reason category.
    """
    rows = []

    for value in df[REASONS_COL]:
        if pd.isna(value):
            continue

        reasons = [
            item.strip()
            for item in str(value).split(",")
            if item.strip()
        ]

        for reason in set(reasons):
            rows.append(reason)

    if not rows:
        return pd.DataFrame(
            columns=[
                "dataset",
                "reason",
                "support",
            ]
        )

    support = (
        pd.Series(rows)
        .value_counts()
        .rename_axis("reason")
        .reset_index(name="support")
    )

    support.insert(0, "dataset", dataset_name)

    return support


# ============================================================
# Combined audit
# ============================================================

def audit_two_datasets(
    pure_df: pd.DataFrame,
    userstories_df: pd.DataFrame,
) -> Tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """
    Run all audit analyses for both datasets.
    """
    audit_summary = pd.DataFrame(
        [
            audit_dataset(pure_df, "PURE"),
            audit_dataset(
                userstories_df,
                "UserStories",
            ),
        ]
    )

    pure_duplicates = build_duplicate_report(
        pure_df,
        "PURE",
    )

    userstories_duplicates = build_duplicate_report(
        userstories_df,
        "UserStories",
    )

    pure_cross_source = (
        build_cross_source_duplicate_report(
            pure_df,
            "PURE",
        )
    )

    userstories_cross_source = (
        build_cross_source_duplicate_report(
            userstories_df,
            "UserStories",
        )
    )

    conflicts = pd.concat(
        [
            build_conflict_report(
                pure_df,
                "PURE",
            ),
            build_conflict_report(
                userstories_df,
                "UserStories",
            ),
        ],
        ignore_index=True,
    )

    reason_support = pd.concat(
        [
            build_reason_support(
                pure_df,
                "PURE",
            ),
            build_reason_support(
                userstories_df,
                "UserStories",
            ),
        ],
        ignore_index=True,
    )

    return (
        audit_summary,
        pure_duplicates,
        userstories_duplicates,
        pure_cross_source,
        userstories_cross_source,
        conflicts,
        reason_support,
    )

# ============================================================
# Label / reason consistency
# ============================================================

def check_label_reason_consistency(
    df: pd.DataFrame,
    dataset_name: str,
) -> Dict[str, int]:
    """
    Check whether operational labels and reason annotations
    are internally consistent.

    Expected structure:
    - label 0 -> no reasons
    - label 1 -> at least one reason
    """
    work = prepare_audit_columns(df)

    labels = pd.to_numeric(
        work[LABEL_COL],
        errors="coerce",
    )

    has_reason = work["_reasons_norm"] != ""

    negative_with_reason = int(
        ((labels == 0) & has_reason).sum()
    )

    positive_without_reason = int(
        ((labels == 1) & ~has_reason).sum()
    )

    return {
        "dataset": dataset_name,
        "negative_with_reason": negative_with_reason,
        "positive_without_reason": positive_without_reason,
    }


# ============================================================
# Deterministic deduplication
# ============================================================

def clean_dataset(
    df: pd.DataFrame,
    dataset_name: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Remove exact duplicates based on conservative normalized text.

    Important:
    - source CSV is never modified
    - original requirement text is preserved
    - first occurrence is retained deterministically
    - duplicates are removed regardless of whether they occur
      within the same source or across different sources

    Returns:
        clean_df
        removed_df
    """
    work = prepare_audit_columns(df)

    # Preserve original row number for full traceability.
    work["_original_row"] = range(len(work))

    duplicate_mask = work.duplicated(
        subset=["_text_norm"],
        keep="first",
    )

    clean_df = work.loc[
        ~duplicate_mask
    ].copy()

    removed_df = work.loc[
        duplicate_mask
    ].copy()

    # Find the retained canonical occurrence for every text.
    canonical = (
        work.loc[~duplicate_mask]
        [
            [
                "_text_norm",
                "_original_row",
                SOURCE_COL,
            ]
        ]
        .rename(
            columns={
                "_original_row": "kept_original_row",
                SOURCE_COL: "kept_source_file",
            }
        )
    )

    if not removed_df.empty:
        removed_df = removed_df.merge(
            canonical,
            on="_text_norm",
            how="left",
        )

    removed_df.insert(
        0,
        "dataset",
        dataset_name,
    )

    # Audit-only helper columns should not become model features.
    clean_df = clean_df.drop(
        columns=[
            "_text_norm",
            "_reasons_norm",
            "_original_row",
        ]
    )

    # Preserve the same four-column logical dataset structure.
    clean_df = clean_df[
        [
            SOURCE_COL,
            TEXT_COL,
            LABEL_COL,
            REASONS_COL,
        ]
    ].reset_index(drop=True)

    # Stable deterministic instance IDs for reproducible splits.
    dataset_prefix = dataset_name.upper().replace(" ", "")

    clean_df.insert(
        0,
        INSTANCE_ID_COL,
        [
            f"{dataset_prefix}_{i:06d}"
            for i in range(1, len(clean_df) + 1)
        ],
    )

    return clean_df, removed_df.reset_index(
        drop=True
    )


# ============================================================
# Cross-dataset overlap
# ============================================================

def build_cross_dataset_overlap_report(
    pure_df: pd.DataFrame,
    userstories_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Identify normalized requirement texts that occur in both datasets.

    Cross-dataset exact overlap would constitute potential leakage
    in transfer experiments, so it must be explicitly checked.
    """
    pure = prepare_audit_columns(pure_df)
    userstories = prepare_audit_columns(userstories_df)

    pure_texts = set(
        pure.loc[
            pure["_text_norm"] != "",
            "_text_norm",
        ]
    )

    userstories_texts = set(
        userstories.loc[
            userstories["_text_norm"] != "",
            "_text_norm",
        ]
    )

    common_texts = pure_texts & userstories_texts

    if not common_texts:
        return pd.DataFrame(
            columns=[
                "dataset",
                SOURCE_COL,
                TEXT_COL,
                LABEL_COL,
                REASONS_COL,
                "_text_norm",
            ]
        )

    pure_overlap = pure[
        pure["_text_norm"].isin(common_texts)
    ].copy()

    pure_overlap.insert(
        0,
        "dataset",
        "PURE",
    )

    userstories_overlap = userstories[
        userstories["_text_norm"].isin(common_texts)
    ].copy()

    userstories_overlap.insert(
        0,
        "dataset",
        "UserStories",
    )

    report = pd.concat(
        [
            pure_overlap,
            userstories_overlap,
        ],
        ignore_index=True,
    )

    return report[
        [
            "dataset",
            SOURCE_COL,
            TEXT_COL,
            LABEL_COL,
            REASONS_COL,
            "_text_norm",
        ]
    ].sort_values(
        ["_text_norm", "dataset"]
    ).reset_index(drop=True)