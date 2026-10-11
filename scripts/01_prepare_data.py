from pathlib import Path
import sys
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    PURE_RAW_PATH,
    USERSTORIES_RAW_PATH,
    DATA_AUDIT_PATH,
    PURE_DUPLICATE_REPORT_PATH,
    USERSTORIES_DUPLICATE_REPORT_PATH,
    PURE_CROSS_SOURCE_DUPLICATES_PATH,
    USERSTORIES_CROSS_SOURCE_DUPLICATES_PATH,
    DATA_CONFLICTS_PATH,
    REASON_SUPPORT_PATH,
    PURE_CLEAN_PATH,
    USERSTORIES_CLEAN_PATH,
    PURE_DUPLICATES_PATH,
    USERSTORIES_DUPLICATES_PATH,
    LABEL_REASON_CHECK_PATH,
    ensure_project_dirs,
)

from src.data_prep import (
    load_raw_dataset,
    audit_two_datasets,
    clean_dataset,
    check_label_reason_consistency,
)


def main() -> None:
    ensure_project_dirs()

    print("=" * 70)
    print("STEP 1: DATA AUDIT")
    print("=" * 70)

    print("\nLoading raw datasets...")

    pure_df = load_raw_dataset(
        PURE_RAW_PATH,
        "PURE",
    )

    userstories_df = load_raw_dataset(
        USERSTORIES_RAW_PATH,
        "UserStories",
    )

    print(
        f"PURE loaded: {len(pure_df):,} rows"
    )

    print(
        f"UserStories loaded: "
        f"{len(userstories_df):,} rows"
    )

    print("\nRunning audit...")

    (
        audit_summary,
        pure_duplicates,
        userstories_duplicates,
        pure_cross_source,
        userstories_cross_source,
        conflicts,
        reason_support,
    ) = audit_two_datasets(
        pure_df,
        userstories_df,
    )

    # --------------------------------------------------------
    # Save audit reports
    # --------------------------------------------------------

    audit_summary.to_csv(
        DATA_AUDIT_PATH,
        index=False,
    )

    pure_duplicates.to_csv(
        PURE_DUPLICATE_REPORT_PATH,
        index=False,
    )

    userstories_duplicates.to_csv(
        USERSTORIES_DUPLICATE_REPORT_PATH,
        index=False,
    )

    pure_cross_source.to_csv(
        PURE_CROSS_SOURCE_DUPLICATES_PATH,
        index=False,
    )

    userstories_cross_source.to_csv(
        USERSTORIES_CROSS_SOURCE_DUPLICATES_PATH,
        index=False,
    )

    conflicts.to_csv(
        DATA_CONFLICTS_PATH,
        index=False,
    )

    reason_support.to_csv(
        REASON_SUPPORT_PATH,
        index=False,
    )

    # --------------------------------------------------------
    # Print audit summary
    # --------------------------------------------------------

    print("\nAudit summary:")
    print(audit_summary.to_string(index=False))

    print("\nReason support:")
    print(
        reason_support.to_string(index=False)
    )

    # ========================================================
    # OVDE IDE NOVI BLOK
    # ========================================================

    print("\nChecking label/reason consistency...")

    label_reason_check = pd.DataFrame(
        [
            check_label_reason_consistency(
                pure_df,
                "PURE",
            ),
            check_label_reason_consistency(
                userstories_df,
                "UserStories",
            ),
        ]
    )

    label_reason_check.to_csv(
        LABEL_REASON_CHECK_PATH,
        index=False,
    )

    print(
        label_reason_check.to_string(
            index=False
        )
    )

    print("\nCreating deduplicated datasets...")

    pure_clean, pure_removed = clean_dataset(
        pure_df,
        "PURE",
    )

    us_clean, us_removed = clean_dataset(
        userstories_df,
        "UserStories",
    )

    pure_clean.to_csv(
        PURE_CLEAN_PATH,
        index=False,
    )

    us_clean.to_csv(
        USERSTORIES_CLEAN_PATH,
        index=False,
    )

    pure_removed.to_csv(
        PURE_DUPLICATES_PATH,
        index=False,
    )

    us_removed.to_csv(
        USERSTORIES_DUPLICATES_PATH,
        index=False,
    )

    print("\nCleaning summary:")

    print(
        f"PURE: "
        f"{len(pure_df):,} -> "
        f"{len(pure_clean):,} "
        f"(removed {len(pure_removed):,})"
    )

    print(
        f"UserStories: "
        f"{len(userstories_df):,} -> "
        f"{len(us_clean):,} "
        f"(removed {len(us_removed):,})"
    )

    print("\nClean datasets written to:")
    print(f"  {PURE_CLEAN_PATH}")
    print(f"  {USERSTORIES_CLEAN_PATH}")

    print("\nRemoved-duplicate logs written to:")
    print(f"  {PURE_DUPLICATES_PATH}")
    print(f"  {USERSTORIES_DUPLICATES_PATH}")

    # ========================================================
    # STARI FINALNI PRINTOVI OSTAJU ISPOD
    # ========================================================

    print("\nReports written to:")
    print(f"  {DATA_AUDIT_PATH}")
    print(f"  {PURE_DUPLICATE_REPORT_PATH}")
    print(
        f"  {USERSTORIES_DUPLICATE_REPORT_PATH}"
    )
    print(
        f"  {PURE_CROSS_SOURCE_DUPLICATES_PATH}"
    )
    print(
        f"  {USERSTORIES_CROSS_SOURCE_DUPLICATES_PATH}"
    )
    print(f"  {DATA_CONFLICTS_PATH}")
    print(f"  {REASON_SUPPORT_PATH}")
    print(f"  {LABEL_REASON_CHECK_PATH}")

    print("\nIMPORTANT:")
    print(
        "Raw source data were not modified."
    )



if __name__ == "__main__":
    main()