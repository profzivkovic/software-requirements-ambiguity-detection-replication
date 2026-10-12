from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd
from sklearn.metrics import confusion_matrix


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.encoder_transformers import (
    LABEL_COL,
    TEXT_COL,
    load_cross_partitions,
    load_intra_partitions,
    load_matched_cross_partitions,
)

from src.t5_model import (
    DEFAULT_EPOCHS,
    DEFAULT_EVAL_BATCH_SIZE,
    DEFAULT_LR,
    DEFAULT_TRAIN_BATCH_SIZE,
    MAX_INPUT_LENGTH,
    MAX_TARGET_LENGTH,
    MODEL_NAME,
    run_t5_training,
)


PURE_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "pure_clean.csv"
)

USERSTORIES_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "userstories_clean.csv"
)


VALID_SEEDS = {
    13,
    21,
    42,
    87,
    123,
}

VALID_SETTINGS = {
    "intra",
    "cross",
    "cross_matched",
}

VALID_DATASETS = {
    "PURE",
    "UserStories",
}


def load_datasets() -> tuple[pd.DataFrame, pd.DataFrame]:
    if not PURE_PATH.exists():
        raise FileNotFoundError(
            f"Missing processed PURE dataset: {PURE_PATH}"
        )

    if not USERSTORIES_PATH.exists():
        raise FileNotFoundError(
            f"Missing processed UserStories dataset: "
            f"{USERSTORIES_PATH}"
        )

    pure_df = pd.read_csv(PURE_PATH)
    userstories_df = pd.read_csv(USERSTORIES_PATH)

    required_columns = {
        TEXT_COL,
        LABEL_COL,
    }

    for dataset_name, df in [
        ("PURE", pure_df),
        ("UserStories", userstories_df),
    ]:
        missing = required_columns - set(df.columns)

        if missing:
            raise ValueError(
                f"{dataset_name} is missing required columns: "
                f"{sorted(missing)}"
            )

    return pure_df, userstories_df


def resolve_partitions(
    setting: str,
    dataset: str,
    seed: int,
    pure_df: pd.DataFrame,
    userstories_df: pd.DataFrame,
):
    if setting == "intra":
        if dataset == "PURE":
            partitions = load_intra_partitions(
                dataset_df=pure_df,
                dataset_name="PURE",
                seed=seed,
            )

            train_name = "PURE"
            test_name = "PURE"

        elif dataset == "UserStories":
            partitions = load_intra_partitions(
                dataset_df=userstories_df,
                dataset_name="UserStories",
                seed=seed,
            )

            train_name = "UserStories"
            test_name = "UserStories"

        else:
            raise ValueError(
                f"Unsupported intra dataset: {dataset}"
            )

    elif setting == "cross":
        if dataset == "PURE":
            partitions = load_cross_partitions(
                source_df=pure_df,
                source_name="PURE",
                target_df=userstories_df,
                seed=seed,
            )

            train_name = "PURE"
            test_name = "UserStories"

        elif dataset == "UserStories":
            partitions = load_cross_partitions(
                source_df=userstories_df,
                source_name="UserStories",
                target_df=pure_df,
                seed=seed,
            )

            train_name = "UserStories"
            test_name = "PURE"

        else:
            raise ValueError(
                f"Unsupported cross source dataset: {dataset}"
            )

    elif setting == "cross_matched":
        if dataset != "PURE":
            raise ValueError(
                "cross_matched is defined only for "
                "PURE -> UserStories."
            )

        partitions = load_matched_cross_partitions(
            pure_df=pure_df,
            target_df=userstories_df,
            seed=seed,
        )

        train_name = "PURE"
        test_name = "UserStories"

    else:
        raise ValueError(
            f"Unsupported setting: {setting}"
        )

    return partitions, train_name, test_name


def positive_rate(df: pd.DataFrame) -> float:
    return float(
        df[LABEL_COL].astype(int).mean()
    )


def validate_partitions(
    partitions,
    setting: str,
    train_name: str,
    test_name: str,
    seed: int,
) -> None:
    train_df = partitions.train
    validation_df = partitions.validation
    test_df = partitions.test

    for partition_name, df in [
        ("train", train_df),
        ("validation", validation_df),
        ("test", test_df),
    ]:
        if df.empty:
            raise AssertionError(
                f"{partition_name} partition is empty."
            )

        if TEXT_COL not in df.columns:
            raise AssertionError(
                f"{partition_name} is missing {TEXT_COL}."
            )

        if LABEL_COL not in df.columns:
            raise AssertionError(
                f"{partition_name} is missing {LABEL_COL}."
            )

        if df[LABEL_COL].isna().any():
            raise AssertionError(
                f"{partition_name} contains missing labels."
            )

        observed_labels = set(
            df[LABEL_COL]
            .astype(int)
            .unique()
        )

        if not observed_labels.issubset({0, 1}):
            raise AssertionError(
                f"{partition_name} contains invalid labels: "
                f"{sorted(observed_labels)}"
            )

    print("=" * 78)
    print("T5 EXPERIMENT")
    print("=" * 78)

    print(f"Setting:       {setting}")
    print(f"Train corpus:  {train_name}")
    print(f"Test corpus:   {test_name}")
    print(f"Seed:          {seed}")

    print()

    print(
        f"Train:         {len(train_df):5d} "
        f"positive_rate={positive_rate(train_df):.4f}"
    )

    print(
        f"Validation:    {len(validation_df):5d} "
        f"positive_rate={positive_rate(validation_df):.4f}"
    )

    print(
        f"Test:          {len(test_df):5d} "
        f"positive_rate={positive_rate(test_df):.4f}"
    )

    print("=" * 78)


def build_output_stem(
    setting: str,
    train_name: str,
    test_name: str,
    seed: int,
) -> str:
    return (
        f"T5__{setting}__"
        f"{train_name}_to_{test_name}__"
        f"seed{seed}"
    )


def run_experiment(
    setting: str,
    dataset: str,
    seed: int,
    output_dir: Path,
) -> None:
    if seed not in VALID_SEEDS:
        raise ValueError(
            f"Seed must be one of "
            f"{sorted(VALID_SEEDS)}, got {seed}."
        )

    if setting not in VALID_SETTINGS:
        raise ValueError(
            f"Setting must be one of "
            f"{sorted(VALID_SETTINGS)}."
        )

    if dataset not in VALID_DATASETS:
        raise ValueError(
            f"Dataset must be one of "
            f"{sorted(VALID_DATASETS)}."
        )

    pure_df, userstories_df = load_datasets()

    (
        partitions,
        train_name,
        test_name,
    ) = resolve_partitions(
        setting=setting,
        dataset=dataset,
        seed=seed,
        pure_df=pure_df,
        userstories_df=userstories_df,
    )

    validate_partitions(
        partitions=partitions,
        setting=setting,
        train_name=train_name,
        test_name=test_name,
        seed=seed,
    )

    train_df = partitions.train
    validation_df = partitions.validation
    test_df = partitions.test

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_stem = build_output_stem(
        setting=setting,
        train_name=train_name,
        test_name=test_name,
        seed=seed,
    )

    result_path = (
        output_dir
        / f"{output_stem}__result.csv"
    )

    predictions_path = (
        output_dir
        / f"{output_stem}__predictions.csv"
    )

    if result_path.exists():
        raise FileExistsError(
            f"Result already exists:\n{result_path}\n"
            "The runner will not overwrite a completed run."
        )

    start_time = time.time()

    result = run_t5_training(
        train_texts=(
            train_df[TEXT_COL]
            .astype(str)
            .tolist()
        ),
        train_labels=(
            train_df[LABEL_COL]
            .astype(int)
            .tolist()
        ),
        validation_texts=(
            validation_df[TEXT_COL]
            .astype(str)
            .tolist()
        ),
        validation_labels=(
            validation_df[LABEL_COL]
            .astype(int)
            .tolist()
        ),
        test_texts=(
            test_df[TEXT_COL]
            .astype(str)
            .tolist()
        ),
        test_labels=(
            test_df[LABEL_COL]
            .astype(int)
            .tolist()
        ),
        seed=seed,
    )

    runtime_seconds = time.time() - start_time

    metrics = result.metrics

    cm = confusion_matrix(
        result.y_true,
        result.y_pred,
        labels=[0, 1],
    )

    tn, fp, fn, tp = [
        int(value)
        for value in cm.ravel()
    ]

    result_row = {
        "model": "T5",
        "pretrained_model": MODEL_NAME,
        "setting": setting,
        "train_dataset": train_name,
        "test_dataset": test_name,
        "seed": seed,
        "train_size": len(train_df),
        "validation_size": len(validation_df),
        "test_size": len(test_df),
        "train_positive_rate": positive_rate(train_df),
        "validation_positive_rate": (
            positive_rate(validation_df)
        ),
        "test_positive_rate": positive_rate(test_df),
        "best_epoch": result.best_epoch,
        "best_validation_f1": (
            result.best_validation_f1
        ),
        "test_loss": result.test_loss,
        "accuracy": metrics["accuracy"],
        "balanced_accuracy": (
            metrics["balanced_accuracy"]
        ),
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
        "mcc": metrics["mcc"],
        "cohen_kappa": metrics["cohen_kappa"],
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "support": len(result.y_true),
        "epochs": DEFAULT_EPOCHS,
        "learning_rate": DEFAULT_LR,
        "train_batch_size": (
            DEFAULT_TRAIN_BATCH_SIZE
        ),
        "eval_batch_size": (
            DEFAULT_EVAL_BATCH_SIZE
        ),
        "max_input_length": MAX_INPUT_LENGTH,
        "max_target_length": MAX_TARGET_LENGTH,
        "runtime_seconds": runtime_seconds,
    }

    prediction_df = test_df.copy()

    prediction_df.insert(
        0,
        "model",
        "T5",
    )

    prediction_df.insert(
        1,
        "pretrained_model",
        MODEL_NAME,
    )

    prediction_df.insert(
        2,
        "setting",
        setting,
    )

    prediction_df.insert(
        3,
        "train_dataset",
        train_name,
    )

    prediction_df.insert(
        4,
        "test_dataset",
        test_name,
    )

    prediction_df.insert(
        5,
        "seed",
        seed,
    )

    prediction_df["y_true"] = (
        result.y_true.astype(int)
    )

    prediction_df["y_pred"] = (
        result.y_pred.astype(int)
    )

    prediction_df["decoded_prediction"] = (
        result.decoded_predictions
    )

    prediction_df[
        "prediction_correct"
    ] = (
        prediction_df["y_true"]
        == prediction_df["y_pred"]
    )

    normalized_generated = (
        prediction_df["decoded_prediction"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    prediction_df[
        "generation_is_expected"
    ] = normalized_generated.isin(
        {
            "yes",
            "no",
        }
    )

    unexpected_generation_count = int(
        (
            ~prediction_df[
                "generation_is_expected"
            ]
        ).sum()
    )

    result_row[
        "unexpected_generation_count"
    ] = unexpected_generation_count

    result_row[
        "unexpected_generation_rate"
    ] = (
        unexpected_generation_count
        / len(prediction_df)
    )

    result_df = pd.DataFrame(
        [result_row]
    )

    # Save predictions first.
    # The result file is the completion marker.
    prediction_df.to_csv(
        predictions_path,
        index=False,
    )

    result_df.to_csv(
        result_path,
        index=False,
    )

    print()
    print("=" * 78)
    print("RUN COMPLETED SUCCESSFULLY")
    print("=" * 78)

    print(
        f"Accuracy:       "
        f"{metrics['accuracy']:.6f}"
    )

    print(
        f"Precision:      "
        f"{metrics['precision']:.6f}"
    )

    print(
        f"Recall:         "
        f"{metrics['recall']:.6f}"
    )

    print(
        f"F1:             "
        f"{metrics['f1']:.6f}"
    )

    print(
        f"MCC:            "
        f"{metrics['mcc']:.6f}"
    )

    print(
        f"Cohen kappa:    "
        f"{metrics['cohen_kappa']:.6f}"
    )

    print(
        f"Unexpected T5 generations: "
        f"{unexpected_generation_count}"
        f"/{len(prediction_df)}"
    )

    print()

    print(
        f"Saved result:\n"
        f"  {result_path}"
    )

    print(
        f"Saved predictions:\n"
        f"  {predictions_path}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run one source-disjoint T5 experiment."
        )
    )

    parser.add_argument(
        "--setting",
        required=True,
        choices=sorted(VALID_SETTINGS),
    )

    parser.add_argument(
        "--dataset",
        required=True,
        choices=sorted(VALID_DATASETS),
        help=(
            "For cross experiments this is the "
            "source dataset. For cross_matched "
            "use PURE."
        ),
    )

    parser.add_argument(
        "--seed",
        required=True,
        type=int,
        choices=sorted(VALID_SEEDS),
    )

    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    run_experiment(
        setting=args.setting,
        dataset=args.dataset,
        seed=args.seed,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()