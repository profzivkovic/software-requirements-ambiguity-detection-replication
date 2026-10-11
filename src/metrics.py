from __future__ import annotations

from typing import Dict, Iterable

import numpy as np

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    cohen_kappa_score,
    confusion_matrix,
)


def compute_binary_metrics(
    y_true: Iterable[int],
    y_pred: Iterable[int],
) -> Dict[str, float]:
    """
    Compute the complete binary-classification metric set used
    throughout the revised experiments.

    Positive class:
        1 = requirement flagged for ambiguity-related review
    """

    y_true = np.asarray(list(y_true), dtype=int)
    y_pred = np.asarray(list(y_pred), dtype=int)

    if len(y_true) != len(y_pred):
        raise ValueError(
            "y_true and y_pred must have the same length."
        )

    if len(y_true) == 0:
        raise ValueError(
            "Cannot compute metrics for an empty dataset."
        )

    allowed = {0, 1}

    if not set(np.unique(y_true)).issubset(allowed):
        raise ValueError(
            "y_true contains values outside {0, 1}."
        )

    if not set(np.unique(y_pred)).issubset(allowed):
        raise ValueError(
            "y_pred contains values outside {0, 1}."
        )

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    ).ravel()

    return {
        "accuracy": float(
            accuracy_score(y_true, y_pred)
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(
                y_true,
                y_pred,
            )
        ),
        "precision": float(
            precision_score(
                y_true,
                y_pred,
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                y_true,
                y_pred,
                zero_division=0,
            )
        ),
        "f1": float(
            f1_score(
                y_true,
                y_pred,
                zero_division=0,
            )
        ),
        "mcc": float(
            matthews_corrcoef(
                y_true,
                y_pred,
            )
        ),
        "cohen_kappa": float(
            cohen_kappa_score(
                y_true,
                y_pred,
            )
        ),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "support": int(len(y_true)),
    }


def aggregate_metric_results(
    results,
    group_columns,
):
    """
    Aggregate repeated experimental runs as mean and sample SD.

    results:
        pandas DataFrame containing metric columns

    group_columns:
        columns defining one experimental configuration
    """
    import pandas as pd

    metric_columns = [
        "accuracy",
        "balanced_accuracy",
        "precision",
        "recall",
        "f1",
        "mcc",
        "cohen_kappa",
    ]

    aggregation = {}

    for metric in metric_columns:
        aggregation[metric] = [
            "mean",
            "std",
        ]

    summary = (
        results
        .groupby(
            group_columns,
            dropna=False,
        )
        .agg(aggregation)
        .reset_index()
    )

    summary.columns = [
        "_".join(
            part for part in column
            if part
        )
        if isinstance(column, tuple)
        else column
        for column in summary.columns
    ]

    return summary