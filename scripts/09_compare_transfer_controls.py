from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


import pandas as pd

from src.config import (
    CLASSICAL_RESULTS_PATH,
    CLASSICAL_SIZE_MATCHED_RESULTS_PATH,
    TRANSFER_CONTROL_COMPARISON_PATH,
)


METRICS = [
    "accuracy",
    "precision",
    "recall",
    "f1",
    "mcc",
    "cohen_kappa",
]


def summarize(
    df: pd.DataFrame,
    condition: str,
) -> pd.DataFrame:

    summary = (
        df.groupby("model")[METRICS]
        .agg(["mean", "std"])
        .reset_index()
    )

    summary.columns = [
        column[0]
        if column[1] == ""
        else f"{column[0]}_{column[1]}"
        for column in summary.columns
    ]

    rename = {
        col: f"{condition}_{col}"
        for col in summary.columns
        if col != "model"
    }

    return summary.rename(
        columns=rename
    )


def main() -> None:

    print("=" * 70)
    print("STEP 9: TRANSFER SIZE-CONTROL COMPARISON")
    print("=" * 70)

    full = pd.read_csv(
        CLASSICAL_RESULTS_PATH
    )

    matched = pd.read_csv(
        CLASSICAL_SIZE_MATCHED_RESULTS_PATH
    )

    # --------------------------------------------------------
    # Full PURE -> UserStories
    # --------------------------------------------------------

    pure_to_us_full = full[
        (full["protocol"] == "cross_full_source")
        & (full["source_dataset"] == "PURE")
        & (full["target_dataset"] == "UserStories")
    ].copy()

    # --------------------------------------------------------
    # Size-matched PURE -> UserStories
    # --------------------------------------------------------

    pure_to_us_matched = matched.copy()

    # --------------------------------------------------------
    # Full UserStories -> PURE
    # --------------------------------------------------------

    us_to_pure = full[
        (full["protocol"] == "cross_full_source")
        & (full["source_dataset"] == "UserStories")
        & (full["target_dataset"] == "PURE")
    ].copy()

    a = summarize(
        pure_to_us_full,
        "pure_us_full",
    )

    b = summarize(
        pure_to_us_matched,
        "pure_us_matched",
    )

    c = summarize(
        us_to_pure,
        "us_pure_full",
    )

    comparison = (
        a.merge(
            b,
            on="model",
            how="inner",
        )
        .merge(
            c,
            on="model",
            how="inner",
        )
    )

    # --------------------------------------------------------
    # Key deltas
    # --------------------------------------------------------

    for metric in METRICS:

        comparison[
            f"{metric}_size_effect"
        ] = (
            comparison[
                f"pure_us_matched_{metric}_mean"
            ]
            - comparison[
                f"pure_us_full_{metric}_mean"
            ]
        )

        comparison[
            f"{metric}_matched_direction_delta"
        ] = (
            comparison[
                f"pure_us_matched_{metric}_mean"
            ]
            - comparison[
                f"us_pure_full_{metric}_mean"
            ]
        )

    comparison.to_csv(
        TRANSFER_CONTROL_COMPARISON_PATH,
        index=False,
    )

    display = comparison[
        [
            "model",

            "pure_us_full_f1_mean",
            "pure_us_matched_f1_mean",
            "us_pure_full_f1_mean",
            "f1_size_effect",
            "f1_matched_direction_delta",

            "pure_us_full_mcc_mean",
            "pure_us_matched_mcc_mean",
            "us_pure_full_mcc_mean",
            "mcc_size_effect",
            "mcc_matched_direction_delta",

            "pure_us_full_cohen_kappa_mean",
            "pure_us_matched_cohen_kappa_mean",
            "us_pure_full_cohen_kappa_mean",
            "cohen_kappa_matched_direction_delta",
        ]
    ]

    print("\nTransfer-control comparison:")
    print(
        display.to_string(
            index=False
        )
    )

    print(
        f"\nSaved to:\n"
        f"  {TRANSFER_CONTROL_COMPARISON_PATH}"
    )


if __name__ == "__main__":
    main()