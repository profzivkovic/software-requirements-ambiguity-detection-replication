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
    RULE_BASELINE_RESULTS_PATH,
    RULE_BASELINE_SUMMARY_PATH,
    RULE_BASELINE_PREDICTIONS_PATH,
    INSTANCE_ID_COL,
    TEXT_COL,
    LABEL_COL,
    REASONS_COL,
    SOURCE_COL,
    SEEDS,
    ensure_project_dirs,
)

from src.metrics import (
    compute_binary_metrics,
    aggregate_metric_results,
)

from src.rule_baseline import (
    explain_rule_prediction,
)


def evaluate_predictions(
    y_true,
    y_pred,
    metadata: dict,
) -> dict:
    """
    Combine experimental metadata with classification metrics.
    """
    metrics = compute_binary_metrics(
        y_true,
        y_pred,
    )

    return {
        **metadata,
        **metrics,
    }


def apply_rule_baseline(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Apply the fixed rule-based baseline to a dataframe.
    """
    out = df.copy()

    explanations = out[TEXT_COL].apply(
        explain_rule_prediction
    )

    out["prediction"] = explanations.apply(
        lambda x: x["prediction"]
    )

    out["rule_categories"] = explanations.apply(
        lambda x: x["rule_categories"]
    )

    out["rule_matches"] = explanations.apply(
        lambda x: x["rule_matches"]
    )

    return out


def get_intra_test_data(
    df: pd.DataFrame,
    dataset_name: str,
    seed: int,
) -> pd.DataFrame:
    """
    Retrieve exactly the held-out test instances specified
    by the saved source-disjoint split manifest.
    """
    manifest_path = (
        INTRA_SPLITS_DIR
        / f"{dataset_name}_seed{seed}.csv"
    )

    manifest = pd.read_csv(
        manifest_path
    )

    test_ids = set(
        manifest.loc[
            manifest["split"] == "test",
            INSTANCE_ID_COL,
        ]
    )

    test_df = df[
        df[INSTANCE_ID_COL].isin(test_ids)
    ].copy()

    if len(test_df) != len(test_ids):
        raise AssertionError(
            f"{dataset_name}, seed {seed}: "
            "test manifest and clean dataset do not match."
        )

    return test_df


def get_intra_training_labels(
    df: pd.DataFrame,
    dataset_name: str,
    seed: int,
) -> pd.Series:
    """
    For the majority baseline, classical training data are
    train + validation.
    """
    manifest_path = (
        INTRA_SPLITS_DIR
        / f"{dataset_name}_seed{seed}.csv"
    )

    manifest = pd.read_csv(
        manifest_path
    )

    training_ids = set(
        manifest.loc[
            manifest["split"].isin(
                ["train", "validation"]
            ),
            INSTANCE_ID_COL,
        ]
    )

    train_df = df[
        df[INSTANCE_ID_COL].isin(training_ids)
    ]

    return train_df[LABEL_COL]


def majority_class(
    labels: pd.Series,
) -> int:
    """
    Return the majority class in training data.

    In case of an exact tie, use class 1 because the application
    is screening-oriented and missing a review-worthy requirement
    is more costly than sending one additional item for review.

    The tie rule is defined in advance and is not data-tuned.
    """
    counts = labels.value_counts()

    count_0 = int(counts.get(0, 0))
    count_1 = int(counts.get(1, 0))

    if count_1 >= count_0:
        return 1

    return 0


def main() -> None:
    ensure_project_dirs()

    print("=" * 70)
    print("STEP 5: MAJORITY + RULE-BASED BASELINES")
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

    metric_rows = []
    prediction_frames = []

    # ========================================================
    # Intra-dataset evaluation
    # ========================================================

    print("\nINTRA-DATASET EVALUATION")

    for dataset_name, df in datasets.items():

        print(
            f"\nDataset: {dataset_name}"
        )

        for seed in SEEDS:

            test_df = get_intra_test_data(
                df=df,
                dataset_name=dataset_name,
                seed=seed,
            )

            # ------------------------------------------------
            # Majority baseline
            # ------------------------------------------------

            training_labels = (
                get_intra_training_labels(
                    df=df,
                    dataset_name=dataset_name,
                    seed=seed,
                )
            )

            majority = majority_class(
                training_labels
            )

            majority_predictions = [
                majority
            ] * len(test_df)

            majority_result = (
                evaluate_predictions(
                    y_true=test_df[LABEL_COL],
                    y_pred=majority_predictions,
                    metadata={
                        "protocol": "intra",
                        "dataset": dataset_name,
                        "source_dataset":
                            dataset_name,
                        "target_dataset":
                            dataset_name,
                        "seed": seed,
                        "model":
                            "Majority",
                    },
                )
            )

            metric_rows.append(
                majority_result
            )

            # ------------------------------------------------
            # Rule-based baseline
            # ------------------------------------------------

            rule_df = apply_rule_baseline(
                test_df
            )

            rule_result = (
                evaluate_predictions(
                    y_true=rule_df[LABEL_COL],
                    y_pred=rule_df["prediction"],
                    metadata={
                        "protocol": "intra",
                        "dataset": dataset_name,
                        "source_dataset":
                            dataset_name,
                        "target_dataset":
                            dataset_name,
                        "seed": seed,
                        "model":
                            "RuleBased",
                    },
                )
            )

            metric_rows.append(
                rule_result
            )

            prediction_output = rule_df[
                [
                    INSTANCE_ID_COL,
                    SOURCE_COL,
                    TEXT_COL,
                    LABEL_COL,
                    REASONS_COL,
                    "prediction",
                    "rule_categories",
                    "rule_matches",
                ]
            ].copy()

            prediction_output.insert(
                0,
                "seed",
                seed,
            )

            prediction_output.insert(
                0,
                "dataset",
                dataset_name,
            )

            prediction_output.insert(
                0,
                "protocol",
                "intra",
            )

            prediction_frames.append(
                prediction_output
            )

            print(
                f"  seed {seed}: "
                f"Majority F1="
                f"{majority_result['f1']:.3f} | "
                f"Rule F1="
                f"{rule_result['f1']:.3f} | "
                f"Rule MCC="
                f"{rule_result['mcc']:.3f}"
            )

    # ========================================================
    # Full-corpus fixed rule baseline
    # ========================================================
    #
    # A rule-based system has no training phase.
    # Therefore, for cross-dataset comparison we evaluate the
    # same frozen system directly on each complete target corpus.
    #
    # It would be misleading to pretend that PURE -> UserStories
    # and UserStories -> PURE "train" the rule system.
    # ========================================================

    print("\nFULL-CORPUS FIXED RULE EVALUATION")

    for dataset_name, df in datasets.items():

        rule_df = apply_rule_baseline(
            df
        )

        result = evaluate_predictions(
            y_true=rule_df[LABEL_COL],
            y_pred=rule_df["prediction"],
            metadata={
                "protocol": "full_corpus_rule",
                "dataset": dataset_name,
                "source_dataset": "none",
                "target_dataset": dataset_name,
                "seed": -1,
                "model": "RuleBased",
            },
        )

        metric_rows.append(
            result
        )

        prediction_output = rule_df[
            [
                INSTANCE_ID_COL,
                SOURCE_COL,
                TEXT_COL,
                LABEL_COL,
                REASONS_COL,
                "prediction",
                "rule_categories",
                "rule_matches",
            ]
        ].copy()

        prediction_output.insert(
            0,
            "seed",
            -1,
        )

        prediction_output.insert(
            0,
            "dataset",
            dataset_name,
        )

        prediction_output.insert(
            0,
            "protocol",
            "full_corpus_rule",
        )

        prediction_frames.append(
            prediction_output
        )

        print(
            f"  {dataset_name}: "
            f"F1={result['f1']:.3f} | "
            f"MCC={result['mcc']:.3f} | "
            f"Recall={result['recall']:.3f}"
        )

    # ========================================================
    # Save raw results
    # ========================================================

    results = pd.DataFrame(
        metric_rows
    )

    results.to_csv(
        RULE_BASELINE_RESULTS_PATH,
        index=False,
    )

    predictions = pd.concat(
        prediction_frames,
        ignore_index=True,
    )

    predictions.to_csv(
        RULE_BASELINE_PREDICTIONS_PATH,
        index=False,
    )

    # ========================================================
    # Aggregate five intra-dataset runs
    # ========================================================

    intra_results = results[
        results["protocol"] == "intra"
    ].copy()

    summary = aggregate_metric_results(
        results=intra_results,
        group_columns=[
            "protocol",
            "dataset",
            "model",
        ],
    )

    summary.to_csv(
        RULE_BASELINE_SUMMARY_PATH,
        index=False,
    )

    print("\n" + "=" * 70)
    print("INTRA-DATASET SUMMARY")
    print("=" * 70)

    display_columns = [
        "dataset",
        "model",
        "accuracy_mean",
        "accuracy_std",
        "f1_mean",
        "f1_std",
        "mcc_mean",
        "mcc_std",
        "cohen_kappa_mean",
        "cohen_kappa_std",
    ]

    print(
        summary[
            display_columns
        ].to_string(
            index=False
        )
    )

    print("\nFiles written to:")
    print(
        f"  {RULE_BASELINE_RESULTS_PATH}"
    )
    print(
        f"  {RULE_BASELINE_SUMMARY_PATH}"
    )
    print(
        f"  {RULE_BASELINE_PREDICTIONS_PATH}"
    )


if __name__ == "__main__":
    main()