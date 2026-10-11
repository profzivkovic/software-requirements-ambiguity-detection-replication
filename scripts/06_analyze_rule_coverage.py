from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


import pandas as pd

from src.config import (
    RULE_BASELINE_PREDICTIONS_PATH,
    RESULTS_SUMMARY_DIR,
    REASONS_COL,
)


RULE_COVERAGE_PATH = (
    RESULTS_SUMMARY_DIR / "rule_reason_coverage.csv"
)


def parse_reasons(value) -> list[str]:
    if pd.isna(value):
        return []

    return [
        item.strip()
        for item in str(value).split(",")
        if item.strip()
    ]


def main() -> None:
    print("=" * 70)
    print("STEP 6: RULE COVERAGE BY MANUAL REASON")
    print("=" * 70)

    df = pd.read_csv(
        RULE_BASELINE_PREDICTIONS_PATH
    )

    # Use the full-corpus evaluation so every clean instance
    # is counted exactly once.
    full = df[
        df["protocol"] == "full_corpus_rule"
    ].copy()

    rows = []

    for dataset_name, dataset_df in full.groupby(
        "dataset"
    ):

        positive_df = dataset_df[
            dataset_df["label"] == 1
        ].copy()

        all_reasons = sorted(
            {
                reason
                for value in positive_df[REASONS_COL]
                for reason in parse_reasons(value)
            }
        )

        for reason in all_reasons:

            mask = positive_df[
                REASONS_COL
            ].apply(
                lambda value:
                reason in parse_reasons(value)
            )

            subset = positive_df[
                mask
            ]

            support = len(subset)

            detected = int(
                subset["prediction"].sum()
            )

            recall = (
                detected / support
                if support > 0
                else float("nan")
            )

            rows.append(
                {
                    "dataset": dataset_name,
                    "reason": reason,
                    "support": support,
                    "detected_by_rule":
                        detected,
                    "missed_by_rule":
                        support - detected,
                    "detection_recall":
                        recall,
                }
            )

    result = pd.DataFrame(rows)

    result = result.sort_values(
        [
            "dataset",
            "support",
        ],
        ascending=[
            True,
            False,
        ],
    ).reset_index(drop=True)

    result.to_csv(
        RULE_COVERAGE_PATH,
        index=False,
    )

    print("\nRule coverage by manual reason:")
    print(
        result.to_string(
            index=False
        )
    )

    print(
        f"\nSaved to:\n  {RULE_COVERAGE_PATH}"
    )


if __name__ == "__main__":
    main()