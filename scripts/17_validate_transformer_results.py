from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.encoder_transformers import (
    INSTANCE_ID_COL,
    LABEL_COL,
    TEXT_COL,
    load_cross_partitions,
    load_intra_partitions,
    load_matched_cross_partitions,
)


RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
    / "raw"
    / "transformer_runs"
)

SUMMARY_DIR = (
    PROJECT_ROOT
    / "results"
    / "summary"
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


MODELS = [
    "DistilBERT",
    "BERT",
    "RoBERTa",
    "ALBERT",
    "T5",
]

SEEDS = [
    13,
    21,
    42,
    87,
    123,
]

EXPERIMENTS = [
    ("intra", "PURE", "PURE"),
    ("intra", "UserStories", "UserStories"),
    ("cross", "PURE", "UserStories"),
    ("cross", "UserStories", "PURE"),
    ("cross_matched", "PURE", "UserStories"),
]

METRIC_COLUMNS = [
    "accuracy",
    "balanced_accuracy",
    "precision",
    "recall",
    "f1",
    "mcc",
    "cohen_kappa",
]

TOLERANCE = 1e-9


def fail(
    errors: list[str],
    message: str,
) -> None:
    errors.append(message)


def parse_run_filename(
    path: Path,
    suffix: str,
) -> tuple[str, str, str, str, int]:
    name = path.name

    if not name.endswith(suffix):
        raise ValueError(
            f"Unexpected filename suffix: {name}"
        )

    core = name[
        : -len(suffix)
    ]

    parts = core.split("__")

    if len(parts) != 4:
        raise ValueError(
            f"Unexpected run filename structure: {name}"
        )

    model = parts[0]
    setting = parts[1]
    route = parts[2]
    seed_part = parts[3]

    if "_to_" not in route:
        raise ValueError(
            f"Missing _to_ route in: {name}"
        )

    train_dataset, test_dataset = route.split(
        "_to_",
        maxsplit=1,
    )

    if not seed_part.startswith("seed"):
        raise ValueError(
            f"Unexpected seed component: {name}"
        )

    seed = int(
        seed_part.removeprefix("seed")
    )

    return (
        model,
        setting,
        train_dataset,
        test_dataset,
        seed,
    )


def expected_run_keys() -> set[
    tuple[str, str, str, str, int]
]:
    keys = set()

    for model in MODELS:
        for (
            setting,
            train_dataset,
            test_dataset,
        ) in EXPERIMENTS:
            for seed in SEEDS:
                keys.add(
                    (
                        model,
                        setting,
                        train_dataset,
                        test_dataset,
                        seed,
                    )
                )

    return keys


def load_clean_datasets() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    pure_df = pd.read_csv(
        PURE_PATH
    )

    userstories_df = pd.read_csv(
        USERSTORIES_PATH
    )

    return pure_df, userstories_df


def get_expected_partitions(
    setting: str,
    train_dataset: str,
    test_dataset: str,
    seed: int,
    pure_df: pd.DataFrame,
    userstories_df: pd.DataFrame,
):
    if (
        setting == "intra"
        and train_dataset == "PURE"
        and test_dataset == "PURE"
    ):
        return load_intra_partitions(
            dataset_df=pure_df,
            dataset_name="PURE",
            seed=seed,
        )

    if (
        setting == "intra"
        and train_dataset == "UserStories"
        and test_dataset == "UserStories"
    ):
        return load_intra_partitions(
            dataset_df=userstories_df,
            dataset_name="UserStories",
            seed=seed,
        )

    if (
        setting == "cross"
        and train_dataset == "PURE"
        and test_dataset == "UserStories"
    ):
        return load_cross_partitions(
            source_df=pure_df,
            source_name="PURE",
            target_df=userstories_df,
            seed=seed,
        )

    if (
        setting == "cross"
        and train_dataset == "UserStories"
        and test_dataset == "PURE"
    ):
        return load_cross_partitions(
            source_df=userstories_df,
            source_name="UserStories",
            target_df=pure_df,
            seed=seed,
        )

    if (
        setting == "cross_matched"
        and train_dataset == "PURE"
        and test_dataset == "UserStories"
    ):
        return load_matched_cross_partitions(
            pure_df=pure_df,
            target_df=userstories_df,
            seed=seed,
        )

    raise ValueError(
        "Unexpected experiment configuration: "
        f"{setting}, "
        f"{train_dataset} -> {test_dataset}, "
        f"seed={seed}"
    )


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:
    return {
        "accuracy": float(
            accuracy_score(
                y_true,
                y_pred,
            )
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(
                y_true,
                y_pred,
            )
        ),
        "precision": float(
            precision_score(
                y_true,
                y_pred,
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                y_true,
                y_pred,
                zero_division=0,
            )
        ),
        "f1": float(
            f1_score(
                y_true,
                y_pred,
                zero_division=0,
            )
        ),
        "mcc": float(
            matthews_corrcoef(
                y_true,
                y_pred,
            )
        ),
        "cohen_kappa": float(
            cohen_kappa_score(
                y_true,
                y_pred,
            )
        ),
    }


def assert_close(
    errors: list[str],
    run_name: str,
    field_name: str,
    observed,
    expected,
) -> None:
    try:
        observed_float = float(observed)
        expected_float = float(expected)
    except Exception:
        fail(
            errors,
            f"{run_name}: cannot compare "
            f"{field_name}: "
            f"{observed!r} vs {expected!r}",
        )
        return

    if not np.isclose(
        observed_float,
        expected_float,
        atol=TOLERANCE,
        rtol=TOLERANCE,
    ):
        fail(
            errors,
            f"{run_name}: {field_name} mismatch: "
            f"stored={observed_float:.12f}, "
            f"recomputed={expected_float:.12f}",
        )


def validate_result_metadata(
    errors: list[str],
    result_row: pd.Series,
    prediction_df: pd.DataFrame,
    model: str,
    setting: str,
    train_dataset: str,
    test_dataset: str,
    seed: int,
    run_name: str,
) -> None:
    expected_values = {
        "model": model,
        "setting": setting,
        "train_dataset": train_dataset,
        "test_dataset": test_dataset,
        "seed": seed,
    }

    for column, expected in expected_values.items():
        if column not in result_row.index:
            fail(
                errors,
                f"{run_name}: result missing "
                f"column {column}",
            )
            continue

        observed = result_row[column]

        if column == "seed":
            if int(observed) != int(expected):
                fail(
                    errors,
                    f"{run_name}: result {column}="
                    f"{observed}, expected {expected}",
                )
        else:
            if str(observed) != str(expected):
                fail(
                    errors,
                    f"{run_name}: result {column}="
                    f"{observed!r}, expected "
                    f"{expected!r}",
                )

    for column, expected in expected_values.items():
        if column not in prediction_df.columns:
            fail(
                errors,
                f"{run_name}: predictions missing "
                f"column {column}",
            )
            continue

        values = (
            prediction_df[column]
            .drop_duplicates()
            .tolist()
        )

        if len(values) != 1:
            fail(
                errors,
                f"{run_name}: prediction column "
                f"{column} has multiple values: "
                f"{values}",
            )
            continue

        observed = values[0]

        if column == "seed":
            if int(observed) != int(expected):
                fail(
                    errors,
                    f"{run_name}: prediction seed "
                    f"{observed}, expected {expected}",
                )
        else:
            if str(observed) != str(expected):
                fail(
                    errors,
                    f"{run_name}: prediction "
                    f"{column}={observed!r}, "
                    f"expected {expected!r}",
                )


def validate_against_locked_test_set(
    errors: list[str],
    prediction_df: pd.DataFrame,
    expected_test_df: pd.DataFrame,
    run_name: str,
) -> None:
    required_prediction_columns = {
        INSTANCE_ID_COL,
        TEXT_COL,
        "y_true",
        "y_pred",
    }

    missing = (
        required_prediction_columns
        - set(prediction_df.columns)
    )

    if missing:
        fail(
            errors,
            f"{run_name}: prediction file missing "
            f"columns {sorted(missing)}",
        )
        return

    if prediction_df[INSTANCE_ID_COL].duplicated().any():
        fail(
            errors,
            f"{run_name}: duplicate instance_id "
            f"values in predictions",
        )

    if expected_test_df[INSTANCE_ID_COL].duplicated().any():
        fail(
            errors,
            f"{run_name}: duplicate instance_id "
            f"values in locked test partition",
        )

    observed_ids = (
        prediction_df[INSTANCE_ID_COL]
        .astype(str)
        .tolist()
    )

    expected_ids = (
        expected_test_df[INSTANCE_ID_COL]
        .astype(str)
        .tolist()
    )

    if len(observed_ids) != len(expected_ids):
        fail(
            errors,
            f"{run_name}: test length mismatch: "
            f"predictions={len(observed_ids)}, "
            f"locked={len(expected_ids)}",
        )

    observed_set = set(
        observed_ids
    )

    expected_set = set(
        expected_ids
    )

    if observed_set != expected_set:
        missing_ids = sorted(
            expected_set - observed_set
        )

        extra_ids = sorted(
            observed_set - expected_set
        )

        fail(
            errors,
            f"{run_name}: prediction instances do "
            f"not match locked test partition. "
            f"missing={missing_ids[:5]}, "
            f"extra={extra_ids[:5]}",
        )

        return

    expected_lookup = (
        expected_test_df
        .assign(
            _instance_id_string=(
                expected_test_df[
                    INSTANCE_ID_COL
                ].astype(str)
            )
        )
        .set_index(
            "_instance_id_string"
        )
    )

    expected_labels = np.asarray(
        [
            int(
                expected_lookup.loc[
                    instance_id,
                    LABEL_COL,
                ]
            )
            for instance_id in observed_ids
        ],
        dtype=int,
    )

    observed_labels = (
        prediction_df["y_true"]
        .astype(int)
        .to_numpy()
    )

    if not np.array_equal(
        observed_labels,
        expected_labels,
    ):
        fail(
            errors,
            f"{run_name}: y_true does not match "
            f"labels from locked test partition",
        )

    expected_texts = [
        str(
            expected_lookup.loc[
                instance_id,
                TEXT_COL,
            ]
        )
        for instance_id in observed_ids
    ]

    observed_texts = (
        prediction_df[TEXT_COL]
        .astype(str)
        .tolist()
    )

    if observed_texts != expected_texts:
        fail(
            errors,
            f"{run_name}: requirement_text does "
            f"not match locked test partition",
        )


def validate_one_run(
    result_path: Path,
    prediction_path: Path,
    expected_partitions,
    errors: list[str],
) -> dict[str, object]:
    (
        model,
        setting,
        train_dataset,
        test_dataset,
        seed,
    ) = parse_run_filename(
        result_path,
        "__result.csv",
    )

    run_name = (
        f"{model} | {setting} | "
        f"{train_dataset}->{test_dataset} | "
        f"seed={seed}"
    )

    result_df = pd.read_csv(
        result_path
    )

    prediction_df = pd.read_csv(
        prediction_path
    )

    if len(result_df) != 1:
        fail(
            errors,
            f"{run_name}: result file contains "
            f"{len(result_df)} rows instead of 1",
        )

        return {
            "model": model,
            "setting": setting,
            "train_dataset": train_dataset,
            "test_dataset": test_dataset,
            "seed": seed,
            "status": "FAILED",
        }

    result_row = result_df.iloc[0]

    validate_result_metadata(
        errors=errors,
        result_row=result_row,
        prediction_df=prediction_df,
        model=model,
        setting=setting,
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        seed=seed,
        run_name=run_name,
    )

    expected_train_size = len(
        expected_partitions.train
    )

    expected_validation_size = len(
        expected_partitions.validation
    )

    expected_test_size = len(
        expected_partitions.test
    )

    for column, expected_value in [
        (
            "train_size",
            expected_train_size,
        ),
        (
            "validation_size",
            expected_validation_size,
        ),
        (
            "test_size",
            expected_test_size,
        ),
    ]:
        if column not in result_row.index:
            fail(
                errors,
                f"{run_name}: result missing "
                f"{column}",
            )
        elif int(result_row[column]) != expected_value:
            fail(
                errors,
                f"{run_name}: {column}="
                f"{int(result_row[column])}, "
                f"locked partition="
                f"{expected_value}",
            )

    validate_against_locked_test_set(
        errors=errors,
        prediction_df=prediction_df,
        expected_test_df=expected_partitions.test,
        run_name=run_name,
    )

    if (
        "y_true" not in prediction_df.columns
        or "y_pred" not in prediction_df.columns
    ):
        return {
            "model": model,
            "setting": setting,
            "train_dataset": train_dataset,
            "test_dataset": test_dataset,
            "seed": seed,
            "status": "FAILED",
        }

    y_true = (
        prediction_df["y_true"]
        .astype(int)
        .to_numpy()
    )

    y_pred = (
        prediction_df["y_pred"]
        .astype(int)
        .to_numpy()
    )

    if not set(
        np.unique(y_true)
    ).issubset({0, 1}):
        fail(
            errors,
            f"{run_name}: invalid y_true labels",
        )

    if not set(
        np.unique(y_pred)
    ).issubset({0, 1}):
        fail(
            errors,
            f"{run_name}: invalid y_pred labels",
        )

    recomputed = compute_metrics(
        y_true,
        y_pred,
    )

    for metric in METRIC_COLUMNS:
        if metric not in result_row.index:
            fail(
                errors,
                f"{run_name}: result missing "
                f"metric {metric}",
            )
            continue

        stored_value = result_row[
            metric
        ]

        if not np.isfinite(
            float(stored_value)
        ):
            fail(
                errors,
                f"{run_name}: non-finite stored "
                f"{metric}={stored_value}",
            )
            continue

        if not np.isfinite(
            recomputed[metric]
        ):
            fail(
                errors,
                f"{run_name}: non-finite recomputed "
                f"{metric}",
            )
            continue

        assert_close(
            errors=errors,
            run_name=run_name,
            field_name=metric,
            observed=stored_value,
            expected=recomputed[metric],
        )

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    )

    tn, fp, fn, tp = [
        int(value)
        for value in cm.ravel()
    ]

    expected_confusion = {
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "support": len(y_true),
    }

    for column, expected_value in (
        expected_confusion.items()
    ):
        if column not in result_row.index:
            fail(
                errors,
                f"{run_name}: result missing "
                f"{column}",
            )
            continue

        if int(result_row[column]) != expected_value:
            fail(
                errors,
                f"{run_name}: {column}="
                f"{int(result_row[column])}, "
                f"recomputed={expected_value}",
            )

    if model == "T5":
        required_t5_prediction_columns = {
            "decoded_prediction",
            "generation_is_expected",
        }

        missing_t5 = (
            required_t5_prediction_columns
            - set(prediction_df.columns)
        )

        if missing_t5:
            fail(
                errors,
                f"{run_name}: T5 predictions "
                f"missing {sorted(missing_t5)}",
            )

        else:
            normalized = (
                prediction_df[
                    "decoded_prediction"
                ]
                .astype(str)
                .str.strip()
                .str.lower()
            )

            actual_expected = normalized.isin(
                {
                    "yes",
                    "no",
                }
            )

            stored_expected = (
                prediction_df[
                    "generation_is_expected"
                ]
                .astype(bool)
                .to_numpy()
            )

            if not np.array_equal(
                stored_expected,
                actual_expected.to_numpy(),
            ):
                fail(
                    errors,
                    f"{run_name}: "
                    f"generation_is_expected does "
                    f"not match decoded outputs",
                )

            unexpected_count = int(
                (~actual_expected).sum()
            )

            if unexpected_count != 0:
                fail(
                    errors,
                    f"{run_name}: "
                    f"{unexpected_count} unexpected "
                    f"T5 generations",
                )

            if (
                "unexpected_generation_count"
                not in result_row.index
            ):
                fail(
                    errors,
                    f"{run_name}: T5 result missing "
                    f"unexpected_generation_count",
                )

            elif int(
                result_row[
                    "unexpected_generation_count"
                ]
            ) != unexpected_count:
                fail(
                    errors,
                    f"{run_name}: stored unexpected "
                    f"generation count does not "
                    f"match predictions",
                )

    return {
        "model": model,
        "setting": setting,
        "train_dataset": train_dataset,
        "test_dataset": test_dataset,
        "seed": seed,
        "train_size": expected_train_size,
        "validation_size": (
            expected_validation_size
        ),
        "test_size": expected_test_size,
        "accuracy": recomputed["accuracy"],
        "balanced_accuracy": (
            recomputed["balanced_accuracy"]
        ),
        "precision": recomputed["precision"],
        "recall": recomputed["recall"],
        "f1": recomputed["f1"],
        "mcc": recomputed["mcc"],
        "cohen_kappa": (
            recomputed["cohen_kappa"]
        ),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "support": len(y_true),
        "status": "PASSED",
    }


def main() -> None:
    print("=" * 80)
    print("TRANSFORMER RESULT INTEGRITY CHECK")
    print("=" * 80)

    if not RESULTS_DIR.exists():
        raise FileNotFoundError(
            f"Missing results directory: "
            f"{RESULTS_DIR}"
        )

    result_files = sorted(
        RESULTS_DIR.glob(
            "*__result.csv"
        )
    )

    prediction_files = sorted(
        RESULTS_DIR.glob(
            "*__predictions.csv"
        )
    )

    print(
        f"Result files:     "
        f"{len(result_files)}"
    )

    print(
        f"Prediction files: "
        f"{len(prediction_files)}"
    )

    errors: list[str] = []

    result_map: dict[
        tuple[str, str, str, str, int],
        Path,
    ] = {}

    prediction_map: dict[
        tuple[str, str, str, str, int],
        Path,
    ] = {}

    for path in result_files:
        try:
            key = parse_run_filename(
                path,
                "__result.csv",
            )
        except Exception as exc:
            fail(
                errors,
                f"{path.name}: {exc}",
            )
            continue

        if key in result_map:
            fail(
                errors,
                f"Duplicate result run: {key}",
            )

        result_map[key] = path

    for path in prediction_files:
        try:
            key = parse_run_filename(
                path,
                "__predictions.csv",
            )
        except Exception as exc:
            fail(
                errors,
                f"{path.name}: {exc}",
            )
            continue

        if key in prediction_map:
            fail(
                errors,
                f"Duplicate prediction run: {key}",
            )

        prediction_map[key] = path

    expected_keys = expected_run_keys()

    observed_result_keys = set(
        result_map
    )

    observed_prediction_keys = set(
        prediction_map
    )

    missing_results = (
        expected_keys
        - observed_result_keys
    )

    extra_results = (
        observed_result_keys
        - expected_keys
    )

    missing_predictions = (
        expected_keys
        - observed_prediction_keys
    )

    extra_predictions = (
        observed_prediction_keys
        - expected_keys
    )

    if missing_results:
        fail(
            errors,
            "Missing result runs: "
            f"{sorted(missing_results)}",
        )

    if extra_results:
        fail(
            errors,
            "Unexpected result runs: "
            f"{sorted(extra_results)}",
        )

    if missing_predictions:
        fail(
            errors,
            "Missing prediction runs: "
            f"{sorted(missing_predictions)}",
        )

    if extra_predictions:
        fail(
            errors,
            "Unexpected prediction runs: "
            f"{sorted(extra_predictions)}",
        )

    pure_df, userstories_df = (
        load_clean_datasets()
    )

    partition_cache = {}

    for (
        setting,
        train_dataset,
        test_dataset,
    ) in EXPERIMENTS:
        for seed in SEEDS:
            cache_key = (
                setting,
                train_dataset,
                test_dataset,
                seed,
            )

            partition_cache[
                cache_key
            ] = get_expected_partitions(
                setting=setting,
                train_dataset=train_dataset,
                test_dataset=test_dataset,
                seed=seed,
                pure_df=pure_df,
                userstories_df=userstories_df,
            )

    validated_rows = []

    common_keys = sorted(
        expected_keys
        & observed_result_keys
        & observed_prediction_keys
    )

    for index, key in enumerate(
        common_keys,
        start=1,
    ):
        (
            model,
            setting,
            train_dataset,
            test_dataset,
            seed,
        ) = key

        print(
            f"[{index:03d}/{len(common_keys):03d}] "
            f"{model:11s} | "
            f"{setting:13s} | "
            f"{train_dataset:11s} -> "
            f"{test_dataset:11s} | "
            f"seed={seed}"
        )

        partition_key = (
            setting,
            train_dataset,
            test_dataset,
            seed,
        )

        row = validate_one_run(
            result_path=result_map[key],
            prediction_path=(
                prediction_map[key]
            ),
            expected_partitions=(
                partition_cache[
                    partition_key
                ]
            ),
            errors=errors,
        )

        validated_rows.append(
            row
        )

    print()
    print("=" * 80)

    if errors:
        print(
            f"INTEGRITY CHECK FAILED "
            f"WITH {len(errors)} ERROR(S)"
        )

        print("=" * 80)

        for number, error in enumerate(
            errors,
            start=1,
        ):
            print(
                f"{number:03d}. {error}"
            )

        raise SystemExit(1)

    validation_df = pd.DataFrame(
        validated_rows
    )

    if len(validation_df) != 125:
        raise AssertionError(
            f"Expected 125 validated runs, "
            f"got {len(validation_df)}"
        )

    model_counts = (
        validation_df
        .groupby("model")
        .size()
    )

    if not (
        model_counts == 25
    ).all():
        raise AssertionError(
            "Every model must have exactly "
            "25 validated runs."
        )

    config_counts = (
        validation_df
        .groupby(
            [
                "model",
                "setting",
                "train_dataset",
                "test_dataset",
            ]
        )
        .size()
    )

    if not (
        config_counts == 5
    ).all():
        raise AssertionError(
            "Every model/setting configuration "
            "must have exactly 5 seeds."
        )

    SUMMARY_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        SUMMARY_DIR
        / "transformer_integrity_check.csv"
    )

    validation_df.to_csv(
        output_path,
        index=False,
    )

    print("INTEGRITY CHECK PASSED")
    print("=" * 80)

    print(
        f"Validated runs: {len(validation_df)}"
    )

    print()
    print("Runs per model:")

    print(
        model_counts.to_string()
    )

    print()
    print(
        "All stored metrics reproduce from "
        "instance-level predictions."
    )

    print(
        "All predictions correspond to the "
        "locked v3 test partitions."
    )

    print(
        "All stored confusion counts are "
        "consistent with predictions."
    )

    print(
        "All T5 decoded predictions are "
        "canonical yes/no generations."
    )

    print()
    print(
        f"Saved validation table:\n"
        f"{output_path}"
    )


if __name__ == "__main__":
    main()