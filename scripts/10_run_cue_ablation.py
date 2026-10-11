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
    CUE_ABLATION_RESULTS_PATH,
    CUE_ABLATION_SUMMARY_PATH,
    CUE_ABLATION_PREDICTIONS_PATH,
    CUE_COVERAGE_PATH,
    INSTANCE_ID_COL,
    SOURCE_COL,
    TEXT_COL,
    LABEL_COL,
    REASONS_COL,
    SEEDS,
    ensure_project_dirs,
)

from src.ablation import (
    contains_surface_cue,
    transform_text,
)

from src.classical import (
    build_classical_models,
    build_classical_pipeline,
)

from src.metrics import (
    compute_binary_metrics,
    aggregate_metric_results,
)


VARIANTS = [
    "original",
    "masked",
    "removed",
]


# ============================================================
# Cue coverage
# ============================================================

def build_cue_coverage(
    df: pd.DataFrame,
    dataset_name: str,
) -> list[dict]:

    rows = []

    has_cue = df[TEXT_COL].apply(
        contains_surface_cue
    )

    for label in [0, 1]:

        subset_mask = (
            df[LABEL_COL] == label
        )

        n = int(
            subset_mask.sum()
        )

        n_with_cue = int(
            (has_cue & subset_mask).sum()
        )

        rows.append(
            {
                "dataset": dataset_name,
                "label": label,
                "support": n,
                "with_surface_cue":
                    n_with_cue,
                "cue_rate":
                    n_with_cue / n
                    if n > 0
                    else float("nan"),
            }
        )

    return rows


# ============================================================
# Split loading
# ============================================================

def get_intra_train_test(
    df: pd.DataFrame,
    dataset_name: str,
    seed: int,
):

    manifest = pd.read_csv(
        INTRA_SPLITS_DIR
        / f"{dataset_name}_seed{seed}.csv"
    )

    split_info = manifest[
        [
            INSTANCE_ID_COL,
            "split",
        ]
    ]

    merged = df.merge(
        split_info,
        on=INSTANCE_ID_COL,
        how="inner",
        validate="one_to_one",
    )

    if len(merged) != len(df):
        raise AssertionError(
            f"{dataset_name}, seed {seed}: "
            "manifest mismatch."
        )

    train_df = merged[
        merged["split"].isin(
            ["train", "validation"]
        )
    ].copy()

    test_df = merged[
        merged["split"] == "test"
    ].copy()

    if (
        set(train_df[SOURCE_COL])
        & set(test_df[SOURCE_COL])
    ):
        raise AssertionError(
            "Source leakage detected."
        )

    return train_df, test_df


# ============================================================
# One experimental run
# ============================================================

def run_one(
    model_name: str,
    model: object,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    variant: str,
    protocol: str,
    source_dataset: str,
    target_dataset: str,
    seed: int,
):

    train_text = train_df[TEXT_COL].apply(
        lambda text:
        transform_text(
            text,
            variant,
        )
    )

    test_text = test_df[TEXT_COL].apply(
        lambda text:
        transform_text(
            text,
            variant,
        )
    )

    pipeline = build_classical_pipeline(
        model
    )

    pipeline.fit(
        train_text,
        train_df[LABEL_COL],
    )

    predictions = pipeline.predict(
        test_text
    )

    metrics = compute_binary_metrics(
        y_true=test_df[LABEL_COL],
        y_pred=predictions,
    )

    result = {
        "protocol": protocol,
        "variant": variant,
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
            TEXT_COL,
            LABEL_COL,
            REASONS_COL,
        ]
    ].copy()

    prediction_df = prediction_df.rename(
        columns={
            TEXT_COL:
                "original_text",
        }
    )

    prediction_df[
        "transformed_text"
    ] = test_text.values

    prediction_df[
        "prediction"
    ] = predictions

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
        "variant",
        variant,
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

    return result, prediction_df


# ============================================================
# Main
# ============================================================

def main() -> None:

    ensure_project_dirs()

    print("=" * 70)
    print("STEP 10: SURFACE-CUE ABLATION")
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

    # --------------------------------------------------------
    # Cue coverage
    # --------------------------------------------------------

    coverage_rows = []

    for dataset_name, df in datasets.items():
        coverage_rows.extend(
            build_cue_coverage(
                df,
                dataset_name,
            )
        )

    coverage = pd.DataFrame(
        coverage_rows
    )

    coverage.to_csv(
        CUE_COVERAGE_PATH,
        index=False,
    )

    print("\nFIXED CUE COVERAGE")
    print(
        coverage.to_string(
            index=False
        )
    )

    metric_rows = []
    prediction_frames = []

    # ========================================================
    # A. Intra-dataset ablation
    # ========================================================

    print(
        "\nA. INTRA-DATASET ABLATION"
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

            print(
                f"\n  seed {seed}"
            )

            for variant in VARIANTS:

                models = build_classical_models(
                    seed=seed
                )

                for model_name, model in models.items():

                    row, pred = run_one(
                        model_name=model_name,
                        model=model,
                        train_df=train_df,
                        test_df=test_df,
                        variant=variant,
                        protocol="intra_source_disjoint",
                        source_dataset=dataset_name,
                        target_dataset=dataset_name,
                        seed=seed,
                    )

                    metric_rows.append(
                        row
                    )

                    prediction_frames.append(
                        pred
                    )

                    print(
                        f"    "
                        f"{variant:8s} "
                        f"{model_name:15s} "
                        f"F1={row['f1']:.3f} "
                        f"MCC={row['mcc']:.3f}"
                    )

    # ========================================================
    # B. Cross-dataset ablation
    # ========================================================

    print(
        "\nB. FULL-SOURCE CROSS-DATASET ABLATION"
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

        train_df = datasets[
            source_name
        ]

        test_df = datasets[
            target_name
        ]

        print(
            f"\n{source_name} -> "
            f"{target_name}"
        )

        for seed in SEEDS:

            print(
                f"\n  seed {seed}"
            )

            for variant in VARIANTS:

                models = build_classical_models(
                    seed=seed
                )

                for model_name, model in models.items():

                    row, pred = run_one(
                        model_name=model_name,
                        model=model,
                        train_df=train_df,
                        test_df=test_df,
                        variant=variant,
                        protocol="cross_full_source",
                        source_dataset=source_name,
                        target_dataset=target_name,
                        seed=seed,
                    )

                    metric_rows.append(
                        row
                    )

                    prediction_frames.append(
                        pred
                    )

                    print(
                        f"    "
                        f"{variant:8s} "
                        f"{model_name:15s} "
                        f"F1={row['f1']:.3f} "
                        f"MCC={row['mcc']:.3f}"
                    )

    # ========================================================
    # Save results
    # ========================================================

    results = pd.DataFrame(
        metric_rows
    )

    results.to_csv(
        CUE_ABLATION_RESULTS_PATH,
        index=False,
    )

    predictions = pd.concat(
        prediction_frames,
        ignore_index=True,
    )

    predictions.to_csv(
        CUE_ABLATION_PREDICTIONS_PATH,
        index=False,
    )

    summary = aggregate_metric_results(
        results=results,
        group_columns=[
            "protocol",
            "variant",
            "source_dataset",
            "target_dataset",
            "model",
        ],
    )

    summary.to_csv(
        CUE_ABLATION_SUMMARY_PATH,
        index=False,
    )

    print("\n" + "=" * 70)
    print("CUE ABLATION SUMMARY")
    print("=" * 70)

    display_columns = [
        "protocol",
        "variant",
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
        f"  {CUE_ABLATION_RESULTS_PATH}"
    )
    print(
        f"  {CUE_ABLATION_SUMMARY_PATH}"
    )
    print(
        f"  {CUE_ABLATION_PREDICTIONS_PATH}"
    )
    print(
        f"  {CUE_COVERAGE_PATH}"
    )


if __name__ == "__main__":
    main()