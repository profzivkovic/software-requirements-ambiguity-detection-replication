from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


from src.config import (
    PURE_CLEAN_PATH,
    USERSTORIES_CLEAN_PATH,
    INSTANCE_ID_COL,
    SOURCE_COL,
    TEXT_COL,
    LABEL_COL,
    REASONS_COL,
)

from src.encoder_transformers import (
    MODEL_CONFIGS,
    DEFAULT_EPOCHS,
    load_intra_partitions,
    load_cross_partitions,
    load_matched_cross_partitions,
    print_partition_summary,
    train_transformer,
)


DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "raw"
    / "transformer_runs"
)


def load_datasets() -> dict[str, pd.DataFrame]:
    pure = pd.read_csv(
        PURE_CLEAN_PATH
    )

    userstories = pd.read_csv(
        USERSTORIES_CLEAN_PATH
    )

    return {
        "PURE": pure,
        "UserStories": userstories,
    }


def get_cross_target(
    source_name: str,
) -> str:
    if source_name == "PURE":
        return "UserStories"

    if source_name == "UserStories":
        return "PURE"

    raise ValueError(
        f"Unknown source dataset: {source_name}"
    )


def build_run_name(
    model: str,
    setting: str,
    train_dataset: str,
    test_dataset: str,
    seed: int,
) -> str:
    safe_model = (
        model
        .replace("/", "_")
        .replace(" ", "_")
    )

    return (
        f"{safe_model}"
        f"__{setting}"
        f"__{train_dataset}"
        f"_to_{test_dataset}"
        f"__seed{seed}"
    )


def save_predictions(
    output_path: Path,
    test_df: pd.DataFrame,
    y_true,
    y_pred,
    model: str,
    setting: str,
    train_dataset: str,
    test_dataset: str,
    seed: int,
) -> None:
    prediction_df = pd.DataFrame(
        {
            "model": model,
            "setting": setting,
            "train_dataset": train_dataset,
            "test_dataset": test_dataset,
            "seed": seed,

            INSTANCE_ID_COL:
                test_df[INSTANCE_ID_COL].values,

            SOURCE_COL:
                test_df[SOURCE_COL].values,

            TEXT_COL:
                test_df[TEXT_COL].astype(str).values,

            "y_true": y_true,
            "y_pred": y_pred,
        }
    )

    if REASONS_COL in test_df.columns:
        prediction_df[
            REASONS_COL
        ] = test_df[
            REASONS_COL
        ].values

    prediction_df.to_csv(
        output_path,
        index=False,
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run one encoder-transformer "
            "experiment using locked "
            "source-disjoint split manifests."
        )
    )

    parser.add_argument(
        "--model",
        required=True,
        choices=list(
            MODEL_CONFIGS.keys()
        ),
    )

    parser.add_argument(
        "--setting",
        required=True,
        choices=[
            "intra",
            "cross",
            "cross_matched",
        ],
    )

    parser.add_argument(
        "--dataset",
        required=True,
        choices=[
            "PURE",
            "UserStories",
        ],
        help=(
            "For intra: dataset used for "
            "train/validation/test. "
            "For cross: source dataset."
        ),
    )

    parser.add_argument(
        "--seed",
        required=True,
        type=int,
        choices=[
            13,
            21,
            42,
            87,
            123,
        ],
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=(
            "Directory for per-run result "
            "and prediction CSV files."
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    datasets = load_datasets()

    train_dataset_name = (
        args.dataset
    )

    if args.setting == "intra":
        test_dataset_name = (
            train_dataset_name
        )

        partitions = (
            load_intra_partitions(
                dataset_df=datasets[
                    train_dataset_name
                ],
                dataset_name=
                train_dataset_name,
                seed=args.seed,
            )
        )

    elif args.setting == "cross":
        test_dataset_name = (
            get_cross_target(
                train_dataset_name
            )
        )

        partitions = (
            load_cross_partitions(
                source_df=datasets[
                    train_dataset_name
                ],
                source_name=
                train_dataset_name,
                target_df=datasets[
                    test_dataset_name
                ],
                seed=args.seed,
            )
        )

    elif args.setting == "cross_matched":
        if train_dataset_name != "PURE":
            raise ValueError(
                "cross_matched is defined only for "
                "PURE -> UserStories because PURE is "
                "the larger source corpus."
            )

        test_dataset_name = "UserStories"

        partitions = (
            load_matched_cross_partitions(
                pure_df=datasets["PURE"],
                target_df=datasets["UserStories"],
                seed=args.seed,
            )
        )

    else:
        raise ValueError(
            f"Unknown setting: {args.setting}"
        )

    run_name = build_run_name(
        model=args.model,
        setting=args.setting,
        train_dataset=
            train_dataset_name,
        test_dataset=
            test_dataset_name,
        seed=args.seed,
    )

    output_dir = (
        args.output_dir
        .expanduser()
        .resolve()
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    result_path = (
        output_dir
        / f"{run_name}__result.csv"
    )

    predictions_path = (
        output_dir
        / f"{run_name}__predictions.csv"
    )

    print("=" * 80)
    print("ENCODER TRANSFORMER EXPERIMENT")
    print("=" * 80)

    print(
        f"Run:          {run_name}"
    )

    print(
        f"Model:        {args.model}"
    )

    print(
        f"Checkpoint:   "
        f"{MODEL_CONFIGS[args.model]}"
    )

    print(
        f"Setting:      {args.setting}"
    )

    print(
        f"Train corpus: "
        f"{train_dataset_name}"
    )

    print(
        f"Test corpus:  "
        f"{test_dataset_name}"
    )

    print(
        f"Seed:         {args.seed}"
    )

    print(
        f"Epochs:       "
        f"{DEFAULT_EPOCHS}"
    )

    print(
        f"Output dir:   "
        f"{output_dir}"
    )

    print_partition_summary(
        run_name,
        partitions,
    )

    print(
        "\nStarting training..."
    )

    (
        result,
        y_true,
        y_pred,
    ) = train_transformer(
        partitions=partitions,
        model_key=args.model,
        seed=args.seed,
        epochs=DEFAULT_EPOCHS,
    )

    result.update(
        {
            "run_name":
                run_name,

            "setting":
                args.setting,

            "train_dataset":
                train_dataset_name,

            "test_dataset":
                test_dataset_name,

            "protocol":
                "final_encoder_experiment",
        }
    )

    result_df = pd.DataFrame(
        [result]
    )

    result_df.to_csv(
        result_path,
        index=False,
    )

    save_predictions(
        output_path=
            predictions_path,
        test_df=
            partitions.test,
        y_true=y_true,
        y_pred=y_pred,
        model=args.model,
        setting=args.setting,
        train_dataset=
            train_dataset_name,
        test_dataset=
            test_dataset_name,
        seed=args.seed,
    )

    print("\n" + "=" * 80)
    print("FINAL TEST RESULT")
    print("=" * 80)

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
        f"\nBest epoch: "
        f"{result['best_epoch']}"
    )

    print(
        f"Best validation F1: "
        f"{result['best_validation_f1']:.4f}"
    )

    print(
        "\nSaved:"
    )

    print(
        f"  {result_path}"
    )

    print(
        f"  {predictions_path}"
    )

    print("\nRUN COMPLETED SUCCESSFULLY")


if __name__ == "__main__":
    main()