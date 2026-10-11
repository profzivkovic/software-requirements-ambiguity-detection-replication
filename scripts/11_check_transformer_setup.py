from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


import pandas as pd
import torch
import transformers

from src.config import (
    PURE_CLEAN_PATH,
    USERSTORIES_CLEAN_PATH,
)

from src.encoder_transformers import (
    MODEL_CONFIGS,
    MAX_LENGTH,
    DEFAULT_EPOCHS,
    DEFAULT_LEARNING_RATE,
    DEFAULT_TRAIN_BATCH_SIZE,
    DEFAULT_EVAL_BATCH_SIZE,
    DEFAULT_WEIGHT_DECAY,
    DEFAULT_WARMUP_RATIO,
    get_device_description,
    load_intra_partitions,
    load_cross_partitions,
    print_partition_summary,
)


def main() -> None:

    print("=" * 70)
    print("STEP 11: CHECK TRANSFORMER SETUP")
    print("=" * 70)

    print("\nSOFTWARE")

    print(
        f"PyTorch:      {torch.__version__}"
    )

    print(
        f"Transformers: {transformers.__version__}"
    )

    print(
        f"Device:       {get_device_description()}"
    )

    print("\nMODEL CONFIGURATION")

    for model_key, checkpoint in (
        MODEL_CONFIGS.items()
    ):
        print(
            f"  {model_key:10s} -> "
            f"{checkpoint}"
        )

    print(
        f"\nmax_length:       {MAX_LENGTH}"
    )

    print(
        f"epochs:           {DEFAULT_EPOCHS}"
    )

    print(
        f"learning_rate:    "
        f"{DEFAULT_LEARNING_RATE}"
    )

    print(
        f"train_batch_size: "
        f"{DEFAULT_TRAIN_BATCH_SIZE}"
    )

    print(
        f"eval_batch_size:  "
        f"{DEFAULT_EVAL_BATCH_SIZE}"
    )

    print(
        f"weight_decay:     "
        f"{DEFAULT_WEIGHT_DECAY}"
    )

    print(
        f"warmup_ratio:     "
        f"{DEFAULT_WARMUP_RATIO}"
    )

    pure = pd.read_csv(
        PURE_CLEAN_PATH
    )

    userstories = pd.read_csv(
        USERSTORIES_CLEAN_PATH
    )

    seed = 13

    print("\n" + "=" * 70)
    print(
        f"PARTITION CHECK — SEED {seed}"
    )
    print("=" * 70)

    pure_intra = load_intra_partitions(
        dataset_df=pure,
        dataset_name="PURE",
        seed=seed,
    )

    print_partition_summary(
        "PURE intra",
        pure_intra,
    )

    userstories_intra = (
        load_intra_partitions(
            dataset_df=userstories,
            dataset_name="UserStories",
            seed=seed,
        )
    )

    print_partition_summary(
        "UserStories intra",
        userstories_intra,
    )

    pure_to_us = (
        load_cross_partitions(
            source_df=pure,
            source_name="PURE",
            target_df=userstories,
            seed=seed,
        )
    )

    print_partition_summary(
        "PURE -> UserStories",
        pure_to_us,
    )

    us_to_pure = (
        load_cross_partitions(
            source_df=userstories,
            source_name="UserStories",
            target_df=pure,
            seed=seed,
        )
    )

    print_partition_summary(
        "UserStories -> PURE",
        us_to_pure,
    )

    print("\n" + "=" * 70)
    print("TRANSFORMER DATA SETUP VALIDATED")
    print("=" * 70)


if __name__ == "__main__":
    main()