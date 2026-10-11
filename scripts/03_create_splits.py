from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


import pandas as pd

from src.config import (
    PURE_CLEAN_PATH,
    USERSTORIES_CLEAN_PATH,
    INTRA_SPLITS_DIR,
    CROSS_SPLITS_DIR,
    SPLIT_SUMMARY_PATH,
    SEEDS,
    ensure_project_dirs,
)

from src.splits import (
    create_intra_split,
    create_cross_source_split,
    validate_split_manifest,
    summarize_split_manifest,
)


def main() -> None:
    ensure_project_dirs()

    print("=" * 70)
    print("STEP 3: CREATE SOURCE-DISJOINT SPLITS")
    print("=" * 70)

    pure = pd.read_csv(
        PURE_CLEAN_PATH
    )

    userstories = pd.read_csv(
        USERSTORIES_CLEAN_PATH
    )

    datasets = {
        "PURE": pure,
        "UserStories": userstories,
    }

    all_summaries = []

    # ========================================================
    # Intra-dataset 70/10/20 manifests
    # ========================================================

    print("\nCreating intra-dataset splits...")

    for dataset_name, df in datasets.items():

        for seed in SEEDS:

            manifest = create_intra_split(
                df=df,
                seed=seed,
            )

            validate_split_manifest(
                manifest,
                expected_splits={
                    "train",
                    "validation",
                    "test",
                },
            )

            out_path = (
                INTRA_SPLITS_DIR
                / f"{dataset_name}_seed{seed}.csv"
            )

            manifest.to_csv(
                out_path,
                index=False,
            )

            summary = summarize_split_manifest(
                manifest,
                dataset_name=dataset_name,
                split_type="intra",
            )

            all_summaries.append(summary)

            print(
                f"  saved {out_path.name}"
            )

    # ========================================================
    # Cross-dataset source 90/10 manifests
    # ========================================================

    print(
        "\nCreating cross-dataset source "
        "train/validation splits..."
    )

    for dataset_name, df in datasets.items():

        for seed in SEEDS:

            manifest = create_cross_source_split(
                source_df=df,
                seed=seed,
            )

            validate_split_manifest(
                manifest,
                expected_splits={
                    "train",
                    "validation",
                },
            )

            out_path = (
                CROSS_SPLITS_DIR
                / f"{dataset_name}_source_seed{seed}.csv"
            )

            manifest.to_csv(
                out_path,
                index=False,
            )

            summary = summarize_split_manifest(
                manifest,
                dataset_name=dataset_name,
                split_type="cross_source",
            )

            all_summaries.append(summary)

            print(
                f"  saved {out_path.name}"
            )

    # ========================================================
    # Save summary
    # ========================================================

    split_summary = pd.concat(
        all_summaries,
        ignore_index=True,
    )

    split_summary.to_csv(
        SPLIT_SUMMARY_PATH,
        index=False,
    )

    print("\nSplit summary:")
    print(
        split_summary.to_string(
            index=False
        )
    )

    print(
        f"\nSummary written to:\n"
        f"  {SPLIT_SUMMARY_PATH}"
    )

    print("\nVALIDATION PASSED.")
    print(
        "No source_file appears in more than "
        "one partition within any split."
    )


if __name__ == "__main__":
    main()