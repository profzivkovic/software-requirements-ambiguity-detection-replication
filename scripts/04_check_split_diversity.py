from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


import pandas as pd

from src.config import (
    INTRA_SPLITS_DIR,
    CROSS_SPLITS_DIR,
    SEEDS,
    SOURCE_COL,
)


def source_signature(
    manifest: pd.DataFrame,
    split_name: str,
) -> tuple:
    """
    Return a deterministic signature of the UNIQUE source files
    assigned to one partition.
    """
    sources = (
        manifest.loc[
            manifest["split"] == split_name,
            SOURCE_COL,
        ]
        .drop_duplicates()
        .astype(str)
        .sort_values()
        .tolist()
    )

    return tuple(sources)


def check_dataset(
    dataset_name: str,
) -> None:

    print("\n" + "=" * 70)
    print(dataset_name)
    print("=" * 70)

    # --------------------------------------------------------
    # Intra test sets
    # --------------------------------------------------------

    print("\nINTRA TEST SOURCE SETS")

    intra_signatures = {}

    for seed in SEEDS:
        path = (
            INTRA_SPLITS_DIR
            / f"{dataset_name}_seed{seed}.csv"
        )

        df = pd.read_csv(path)

        sig = source_signature(
            df,
            "test",
        )

        intra_signatures[seed] = sig

        print(
            f"seed {seed}: "
            f"{len(sig)} test sources"
        )

    unique_intra = len(
        set(intra_signatures.values())
    )

    print(
        f"Unique intra test source sets: "
        f"{unique_intra}/{len(SEEDS)}"
    )

    # --------------------------------------------------------
    # Cross validation source sets
    # --------------------------------------------------------

    print("\nCROSS-SOURCE VALIDATION SETS")

    cross_signatures = {}

    for seed in SEEDS:
        path = (
            CROSS_SPLITS_DIR
            / f"{dataset_name}_source_seed{seed}.csv"
        )

        df = pd.read_csv(path)

        sig = source_signature(
            df,
            "validation",
        )

        cross_signatures[seed] = sig

        print(
            f"\nseed {seed}: "
            f"{len(sig)} validation sources"
        )

        for source in sig:
            print(f"  - {source}")

    unique_cross = len(
        set(cross_signatures.values())
    )

    print(
        f"\nUnique cross validation source sets: "
        f"{unique_cross}/{len(SEEDS)}"
    )


def main() -> None:
    print("=" * 70)
    print("STEP 4: CHECK SPLIT DIVERSITY")
    print("=" * 70)

    check_dataset("PURE")
    check_dataset("UserStories")


if __name__ == "__main__":
    main()