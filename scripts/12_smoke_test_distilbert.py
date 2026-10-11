from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


import pandas as pd

from src.config import (
    USERSTORIES_CLEAN_PATH,
    RESULTS_RAW_DIR,
    ensure_project_dirs,
)

from src.encoder_transformers import (
    load_intra_partitions,
    print_partition_summary,
    train_transformer,
)


SMOKE_RESULT_PATH = (
    RESULTS_RAW_DIR
    / "distilbert_smoke_test.csv"
)


def main() -> None:

    ensure_project_dirs()

    print("=" * 70)
    print("STEP 12: DISTILBERT GPU SMOKE TEST")
    print("=" * 70)

    userstories = pd.read_csv(
        USERSTORIES_CLEAN_PATH
    )

    seed = 13

    partitions = load_intra_partitions(
        dataset_df=userstories,
        dataset_name="UserStories",
        seed=seed,
    )

    print_partition_summary(
        "UserStories intra",
        partitions,
    )

    print(
        "\nIMPORTANT: this is only a "
        "1-epoch infrastructure smoke test."
    )

    print(
        "Its metrics are NOT part of "
        "the reported experiments.\n"
    )

    result, _, _ = train_transformer(
        partitions=partitions,
        model_key="DistilBERT",
        seed=seed,
        epochs=1,
    )

    result["protocol"] = (
        "smoke_test"
    )

    result["source_dataset"] = (
        "UserStories"
    )

    result["target_dataset"] = (
        "UserStories"
    )

    pd.DataFrame(
        [result]
    ).to_csv(
        SMOKE_RESULT_PATH,
        index=False,
    )

    print("\n" + "=" * 70)
    print("SMOKE TEST RESULT")
    print("=" * 70)

    print(
        f"Accuracy:  "
        f"{result['accuracy']:.4f}"
    )

    print(
        f"Precision: "
        f"{result['precision']:.4f}"
    )

    print(
        f"Recall:    "
        f"{result['recall']:.4f}"
    )

    print(
        f"F1:        "
        f"{result['f1']:.4f}"
    )

    print(
        f"MCC:       "
        f"{result['mcc']:.4f}"
    )

    print(
        f"Kappa:     "
        f"{result['cohen_kappa']:.4f}"
    )

    print(
        f"\nSaved to:\n"
        f"  {SMOKE_RESULT_PATH}"
    )


if __name__ == "__main__":
    main()