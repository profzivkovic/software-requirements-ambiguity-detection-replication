from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.config import (
    PURE_CLEAN_PATH,
    USERSTORIES_CLEAN_PATH,
    CLEAN_DATA_AUDIT_PATH,
    CROSS_DATASET_OVERLAP_PATH,
    INSTANCE_ID_COL,
)

from src.data_prep import (
    audit_dataset,
    build_cross_dataset_overlap_report,
    check_label_reason_consistency,
)


def validate_one_dataset(
    df: pd.DataFrame,
    dataset_name: str,
) -> dict:
    """
    Validate one cleaned dataset.
    """
    audit = audit_dataset(
        df,
        dataset_name,
    )

    consistency = check_label_reason_consistency(
        df,
        dataset_name,
    )

    if INSTANCE_ID_COL not in df.columns:
        raise AssertionError(
            f"{dataset_name}: missing {INSTANCE_ID_COL}"
        )

    if df[INSTANCE_ID_COL].isna().any():
        raise AssertionError(
            f"{dataset_name}: missing instance IDs"
        )

    if not df[INSTANCE_ID_COL].is_unique:
        raise AssertionError(
            f"{dataset_name}: instance IDs are not unique"
        )

    if audit["duplicated_extra_rows"] != 0:
        raise AssertionError(
            f"{dataset_name}: normalized duplicates remain"
        )

    if audit["duplicated_text_groups"] != 0:
        raise AssertionError(
            f"{dataset_name}: duplicate text groups remain"
        )

    if audit["cross_source_duplicate_groups"] != 0:
        raise AssertionError(
            f"{dataset_name}: cross-source duplicates remain"
        )

    if audit["conflicting_label_groups"] != 0:
        raise AssertionError(
            f"{dataset_name}: conflicting labels remain"
        )

    if audit["conflicting_reason_groups"] != 0:
        raise AssertionError(
            f"{dataset_name}: conflicting reasons remain"
        )

    if consistency["negative_with_reason"] != 0:
        raise AssertionError(
            f"{dataset_name}: negative instances with reasons found"
        )

    if consistency["positive_without_reason"] != 0:
        raise AssertionError(
            f"{dataset_name}: positive instances without reasons found"
        )

    return {
        **audit,
        "negative_with_reason":
            consistency["negative_with_reason"],
        "positive_without_reason":
            consistency["positive_without_reason"],
    }


def main() -> None:
    print("=" * 70)
    print("STEP 2: VALIDATE CLEAN DATA")
    print("=" * 70)

    pure = pd.read_csv(
        PURE_CLEAN_PATH
    )

    userstories = pd.read_csv(
        USERSTORIES_CLEAN_PATH
    )

    print(
        f"\nPURE clean: {len(pure):,} rows"
    )

    print(
        f"UserStories clean: "
        f"{len(userstories):,} rows"
    )

    results = pd.DataFrame(
        [
            validate_one_dataset(
                pure,
                "PURE",
            ),
            validate_one_dataset(
                userstories,
                "UserStories",
            ),
        ]
    )

    results.to_csv(
        CLEAN_DATA_AUDIT_PATH,
        index=False,
    )

    print("\nClean-data audit:")
    print(
        results.to_string(
            index=False
        )
    )

    print("\nChecking cross-dataset exact overlap...")

    overlap = build_cross_dataset_overlap_report(
        pure,
        userstories,
    )

    overlap.to_csv(
        CROSS_DATASET_OVERLAP_PATH,
        index=False,
    )

    print(
        f"Cross-dataset overlapping rows: "
        f"{len(overlap):,}"
    )

    if not overlap.empty:
        raise AssertionError(
            "Exact normalized text overlap exists "
            "between PURE and UserStories."
        )

    print("\nVALIDATION PASSED.")
    print(
        "Clean datasets contain no normalized duplicates, "
        "no annotation conflicts, and no exact "
        "cross-dataset overlap."
    )


if __name__ == "__main__":
    main()