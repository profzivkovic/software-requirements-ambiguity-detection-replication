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
    CLASSICAL_RESULTS_PATH,
    CLASSICAL_SUMMARY_PATH,
    CLASSICAL_PREDICTIONS_PATH,
    INSTANCE_ID_COL,
    SOURCE_COL,
    TEXT_COL,
    LABEL_COL,
    REASONS_COL,
    SEEDS,
    ensure_project_dirs,
)

from src.classical import (
    build_classical_models,
    build_classical_pipeline,
)

from src.metrics import (
    compute_binary_metrics,
    aggregate_metric_results,
)


# ============================================================
# Intra-dataset data loading
# ============================================================

def get_intra_train_test(
    df: pd.DataFrame,
    dataset_name: str,
    seed: int,
):
    """
    Classical models use:
        train + validation -> training
        test               -> final testing

    The test partition is exactly the same source-disjoint
    test partition that will later be used by transformers.
    """

    manifest_path = (
        INTRA_SPLITS_DIR
        / f"{dataset_name}_seed{seed}.csv"
    )

    manifest = pd.read_csv(
        manifest_path
    )

    manifest_subset = manifest[
        [
            INSTANCE_ID_COL,
            "split",
        ]
    ].copy()

    merged = df.merge(
        manifest_subset,
        on=INSTANCE_ID_COL,
        how="inner",
        validate="one_to_one",
    )

    if len(merged) != len(df):
        raise AssertionError(
            f"{dataset_name}, seed {seed}: "
            "split manifest does not match clean dataset."
        )

    train_df = merged[
        merged["split"].isin(
            [
                "train",
                "validation",
            ]
        )
    ].copy()

    test_df = merged[
        merged["split"] == "test"
    ].copy()

    train_sources = set(
        train_df[SOURCE_COL]
    )

    test_sources = set(
        test_df[SOURCE_COL]
    )

    overlap = (
        train_sources
        & test_sources
    )

    if overlap:
        raise AssertionError(
            f"{dataset_name}, seed {seed}: "
            f"source leakage detected: {overlap}"
        )

    return train_df, test_df


# ============================================================
# One model run
# ============================================================

def run_model(
    model_name: str,
    model: object,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    protocol: str,
    source_dataset: str,
    target_dataset: str,
    seed: int,
):
    """
    Train one classical model and return metrics plus
    instance-level predictions.
    """

    pipeline = build_classical_pipeline(
        model
    )

    pipeline.fit(
        train_df[TEXT_COL].astype(str),
        train_df[LABEL_COL],
    )

    predictions = pipeline.predict(
        test_df[TEXT_COL].astype(str)
    )

    metrics = compute_binary_metrics(
        y_true=test_df[LABEL_COL],
        y_pred=predictions,
    )

    metric_row = {
        "protocol": protocol,
        "source_dataset": source_dataset,
        "target_dataset": target_dataset,
        "model": model_name,
        "seed": seed,
        "train_size": len(train_df),
        "test_size": len(test_df),
        **metrics,
    }

    prediction_df = test_df[
        [
            INSTANCE_ID_COL,
            SOURCE_COL,
            LABEL_COL,
            REASONS_COL,
        ]
    ].copy()

    prediction_df["prediction"] = predictions

    prediction_df.insert(
        0,
        "seed",
        seed,
    )

    prediction_df.insert(
        0,
        "model",
        model_name,
    )

    prediction_df.insert(
        0,
        "target_dataset",
        target_dataset,
    )

    prediction_df.insert(
        0,
        "source_dataset",
        source_dataset,
    )

    prediction_df.insert(
        0,
        "protocol",
        protocol,
    )

    return metric_row, prediction_df


# ============================================================
# Main
# ============================================================

def main() -> None:
    ensure_project_dirs()

    print("=" * 70)
    print("STEP 7: CLASSICAL MACHINE LEARNING")
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
    # A. Intra-dataset evaluation
    # ========================================================

    print(
        "\nA. SOURCE-DISJOINT INTRA-DATASET EVALUATION"
    )

    for dataset_name, df in datasets.items():

        print(
            f"\nDataset: {dataset_name}"
        )

        for seed in SEEDS:

            train_df, test_df = (
                get_intra_train_test(
                    df=df,
                    dataset_name=dataset_name,
                    seed=seed,
                )
            )

            models = build_classical_models(
                seed=seed
            )

            print(
                f"\n  seed {seed}: "
                f"train={len(train_df):,}, "
                f"test={len(test_df):,}"
            )

            for model_name, model in models.items():

                metric_row, prediction_df = (
                    run_model(
                        model_name=model_name,
                        model=model,
                        train_df=train_df,
                        test_df=test_df,
                        protocol="intra_source_disjoint",
                        source_dataset=dataset_name,
                        target_dataset=dataset_name,
                        seed=seed,
                    )
                )

                metric_rows.append(
                    metric_row
                )

                prediction_frames.append(
                    prediction_df
                )

                print(
                    f"    {model_name:15s} "
                    f"F1={metric_row['f1']:.3f} "
                    f"MCC={metric_row['mcc']:.3f} "
                    f"Kappa="
                    f"{metric_row['cohen_kappa']:.3f}"
                )

    # ========================================================
    # B. Cross-dataset evaluation
    # ========================================================

    print(
        "\nB. FULL-SOURCE CROSS-DATASET EVALUATION"
    )

    directions = [
        (
            "PURE",
            "UserStories",
        ),
        (
            "UserStories",
            "PURE",
        ),
    ]

    for source_name, target_name in directions:

        source_df = datasets[
            source_name
        ]

        target_df = datasets[
            target_name
        ]

        print(
            f"\n{source_name} -> {target_name}"
        )

        for seed in SEEDS:

            models = build_classical_models(
                seed=seed
            )

            print(
                f"\n  seed {seed}: "
                f"train={len(source_df):,}, "
                f"test={len(target_df):,}"
            )

            for model_name, model in models.items():

                metric_row, prediction_df = (
                    run_model(
                        model_name=model_name,
                        model=model,
                        train_df=source_df,
                        test_df=target_df,
                        protocol="cross_full_source",
                        source_dataset=source_name,
                        target_dataset=target_name,
                        seed=seed,
                    )
                )

                metric_rows.append(
                    metric_row
                )

                prediction_frames.append(
                    prediction_df
                )

                print(
                    f"    {model_name:15s} "
                    f"F1={metric_row['f1']:.3f} "
                    f"MCC={metric_row['mcc']:.3f} "
                    f"Kappa="
                    f"{metric_row['cohen_kappa']:.3f}"
                )

    # ========================================================
    # Save raw metrics
    # ========================================================

    results = pd.DataFrame(
        metric_rows
    )

    results.to_csv(
        CLASSICAL_RESULTS_PATH,
        index=False,
    )

    # ========================================================
    # Save instance predictions
    # ========================================================

    predictions = pd.concat(
        prediction_frames,
        ignore_index=True,
    )

    predictions.to_csv(
        CLASSICAL_PREDICTIONS_PATH,
        index=False,
    )

    # ========================================================
    # Aggregate mean ± SD
    # ========================================================

    summary = aggregate_metric_results(
        results=results,
        group_columns=[
            "protocol",
            "source_dataset",
            "target_dataset",
            "model",
        ],
    )

    summary.to_csv(
        CLASSICAL_SUMMARY_PATH,
        index=False,
    )

    # ========================================================
    # Display compact summary
    # ========================================================

    print("\n" + "=" * 70)
    print("CLASSICAL SUMMARY")
    print("=" * 70)

    display_columns = [
        "protocol",
        "source_dataset",
        "target_dataset",
        "model",
        "accuracy_mean",
        "accuracy_std",
        "precision_mean",
        "precision_std",
        "recall_mean",
        "recall_std",
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
        f"  {CLASSICAL_RESULTS_PATH}"
    )
    print(
        f"  {CLASSICAL_SUMMARY_PATH}"
    )
    print(
        f"  {CLASSICAL_PREDICTIONS_PATH}"
    )


if __name__ == "__main__":
    main()