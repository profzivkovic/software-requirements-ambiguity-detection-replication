from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


import pandas as pd

from src.config import (
    PURE_CLEAN_PATH,
    USERSTORIES_CLEAN_PATH,
    SIZE_MATCHED_SPLITS_DIR,
    CLASSICAL_SIZE_MATCHED_RESULTS_PATH,
    CLASSICAL_SIZE_MATCHED_SUMMARY_PATH,
    INSTANCE_ID_COL,
    SOURCE_COL,
    TEXT_COL,
    LABEL_COL,
    SEEDS,
    ensure_project_dirs,
)

from src.splits import (
    create_size_matched_source_subset,
)

from src.classical import (
    build_classical_models,
    build_classical_pipeline,
)

from src.metrics import (
    compute_binary_metrics,
    aggregate_metric_results,
)


def run_one_model(
    model_name,
    model,
    train_df,
    test_df,
    seed,
):
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

    return {
        "protocol": "cross_size_matched",
        "source_dataset": "PURE",
        "target_dataset": "UserStories",
        "model": model_name,
        "seed": seed,
        "train_size": len(train_df),
        "train_sources":
            train_df[SOURCE_COL].nunique(),
        "train_positive_rate":
            float(train_df[LABEL_COL].mean()),
        "test_size": len(test_df),
        **metrics,
    }


def main() -> None:
    ensure_project_dirs()

    print("=" * 70)
    print("STEP 8: SIZE-MATCHED CROSS-DATASET CONTROL")
    print("=" * 70)

    pure = pd.read_csv(
        PURE_CLEAN_PATH
    )

    userstories = pd.read_csv(
        USERSTORIES_CLEAN_PATH
    )

    # UserStories is the smaller source corpus.
    # Match PURE training size to its number of instances.
    target_training_size = len(
        userstories
    )

    print(
        f"\nPURE full size: {len(pure):,}"
    )

    print(
        f"UserStories full size: "
        f"{len(userstories):,}"
    )

    print(
        f"Requested PURE matched size: "
        f"{target_training_size:,}"
    )

    result_rows = []

    for seed in SEEDS:

        manifest = (
            create_size_matched_source_subset(
                source_df=pure,
                target_size=target_training_size,
                seed=seed,
            )
        )

        manifest_path = (
            SIZE_MATCHED_SPLITS_DIR
            / f"PURE_matched_seed{seed}.csv"
        )

        manifest.to_csv(
            manifest_path,
            index=False,
        )

        selected_ids = set(
            manifest.loc[
                manifest["selected"],
                INSTANCE_ID_COL,
            ]
        )

        matched_pure = pure[
            pure[INSTANCE_ID_COL].isin(
                selected_ids
            )
        ].copy()

        # Safety checks
        manifest_sources = set(
            manifest.loc[
                manifest["selected"],
                SOURCE_COL,
            ]
        )

        actual_sources = set(
            matched_pure[SOURCE_COL]
        )

        if manifest_sources != actual_sources:
            raise AssertionError(
                "Source manifest mismatch."
            )

        if len(matched_pure) != len(
            selected_ids
        ):
            raise AssertionError(
                "Instance manifest mismatch."
            )

        print(
            f"\nseed {seed}: "
            f"matched PURE={len(matched_pure):,}, "
            f"sources="
            f"{matched_pure[SOURCE_COL].nunique()}, "
            f"positive_rate="
            f"{matched_pure[LABEL_COL].mean():.3f}"
        )

        models = build_classical_models(
            seed=seed
        )

        for model_name, model in models.items():

            row = run_one_model(
                model_name=model_name,
                model=model,
                train_df=matched_pure,
                test_df=userstories,
                seed=seed,
            )

            result_rows.append(
                row
            )

            print(
                f"    {model_name:15s} "
                f"F1={row['f1']:.3f} "
                f"MCC={row['mcc']:.3f} "
                f"Kappa="
                f"{row['cohen_kappa']:.3f}"
            )

    results = pd.DataFrame(
        result_rows
    )

    results.to_csv(
        CLASSICAL_SIZE_MATCHED_RESULTS_PATH,
        index=False,
    )

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
        CLASSICAL_SIZE_MATCHED_SUMMARY_PATH,
        index=False,
    )

    print("\n" + "=" * 70)
    print("SIZE-MATCHED SUMMARY")
    print("=" * 70)

    display_columns = [
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

    print("\nSaved:")
    print(
        f"  {CLASSICAL_SIZE_MATCHED_RESULTS_PATH}"
    )
    print(
        f"  {CLASSICAL_SIZE_MATCHED_SUMMARY_PATH}"
    )


if __name__ == "__main__":
    main()