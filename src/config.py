from pathlib import Path


# ============================================================
# Project paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_RAW_DIR = RESULTS_DIR / "raw"
RESULTS_SUMMARY_DIR = RESULTS_DIR / "summary"

FIGURES_DIR = PROJECT_ROOT / "figures"
LOGS_DIR = PROJECT_ROOT / "logs"

SPLITS_DIR = DATA_DIR / "splits"

INTRA_SPLITS_DIR = SPLITS_DIR / "intra"
CROSS_SPLITS_DIR = SPLITS_DIR / "cross"




# ============================================================
# Input datasets
# ============================================================

PURE_RAW_PATH = RAW_DATA_DIR / "pure_labeled.csv"
USERSTORIES_RAW_PATH = RAW_DATA_DIR / "userstories_labeled.csv"


# ============================================================
# Processed datasets
# ============================================================

PURE_CLEAN_PATH = PROCESSED_DATA_DIR / "pure_clean.csv"
USERSTORIES_CLEAN_PATH = PROCESSED_DATA_DIR / "userstories_clean.csv"


# ============================================================
# Audit / data-preparation outputs
# ============================================================

PURE_DUPLICATES_PATH = PROCESSED_DATA_DIR / "pure_duplicates_removed.csv"
USERSTORIES_DUPLICATES_PATH = (
    PROCESSED_DATA_DIR / "userstories_duplicates_removed.csv"
)

DATA_AUDIT_PATH = RESULTS_SUMMARY_DIR / "data_audit.csv"

PURE_DUPLICATE_REPORT_PATH = (
    RESULTS_SUMMARY_DIR / "pure_duplicate_report.csv"
)

USERSTORIES_DUPLICATE_REPORT_PATH = (
    RESULTS_SUMMARY_DIR / "userstories_duplicate_report.csv"
)

PURE_CROSS_SOURCE_DUPLICATES_PATH = (
    RESULTS_SUMMARY_DIR / "pure_cross_source_duplicates.csv"
)

USERSTORIES_CROSS_SOURCE_DUPLICATES_PATH = (
    RESULTS_SUMMARY_DIR / "userstories_cross_source_duplicates.csv"
)

DATA_CONFLICTS_PATH = (
    RESULTS_SUMMARY_DIR / "data_conflicts.csv"
)

REASON_SUPPORT_PATH = (
    RESULTS_SUMMARY_DIR / "reason_support.csv"
)

LABEL_REASON_CHECK_PATH = (
    RESULTS_SUMMARY_DIR / "label_reason_consistency.csv"
)

CROSS_DATASET_OVERLAP_PATH = (
    RESULTS_SUMMARY_DIR / "cross_dataset_overlap.csv"
)

CLEAN_DATA_AUDIT_PATH = (
    RESULTS_SUMMARY_DIR / "clean_data_audit.csv"
)

SPLIT_SUMMARY_PATH = (
    RESULTS_SUMMARY_DIR / "split_summary.csv"
)

RULE_BASELINE_RESULTS_PATH = (
    RESULTS_RAW_DIR / "rule_baseline_results.csv"
)

RULE_BASELINE_SUMMARY_PATH = (
    RESULTS_SUMMARY_DIR / "rule_baseline_summary.csv"
)

RULE_BASELINE_PREDICTIONS_PATH = (
    RESULTS_RAW_DIR / "rule_baseline_predictions.csv"
)

CLASSICAL_RESULTS_PATH = (
    RESULTS_RAW_DIR / "classical_results.csv"
)

CLASSICAL_SUMMARY_PATH = (
    RESULTS_SUMMARY_DIR / "classical_summary.csv"
)

CLASSICAL_PREDICTIONS_PATH = (
    RESULTS_RAW_DIR / "classical_predictions.csv"
)

SIZE_MATCHED_SPLITS_DIR = SPLITS_DIR / "size_matched"

CLASSICAL_SIZE_MATCHED_RESULTS_PATH = (
    RESULTS_RAW_DIR / "classical_size_matched_results.csv"
)

CLASSICAL_SIZE_MATCHED_SUMMARY_PATH = (
    RESULTS_SUMMARY_DIR / "classical_size_matched_summary.csv"
)

TRANSFER_CONTROL_COMPARISON_PATH = (
    RESULTS_SUMMARY_DIR / "transfer_control_comparison.csv"
)

CUE_ABLATION_RESULTS_PATH = (
    RESULTS_RAW_DIR / "cue_ablation_results.csv"
)

CUE_ABLATION_SUMMARY_PATH = (
    RESULTS_SUMMARY_DIR / "cue_ablation_summary.csv"
)

CUE_ABLATION_PREDICTIONS_PATH = (
    RESULTS_RAW_DIR / "cue_ablation_predictions.csv"
)

CUE_COVERAGE_PATH = (
    RESULTS_SUMMARY_DIR / "cue_coverage.csv"
)

# ============================================================
# Dataset column names
# ============================================================

SOURCE_COL = "source_file"
TEXT_COL = "requirement_text"
LABEL_COL = "label"
REASONS_COL = "reasons"
INSTANCE_ID_COL = "instance_id"

REQUIRED_COLUMNS = [
    SOURCE_COL,
    TEXT_COL,
    LABEL_COL,
    REASONS_COL,
]


# ============================================================
# Experimental configuration
# ============================================================

SEEDS = [13, 21, 42, 87, 123]

TEST_SIZE = 0.20
VALIDATION_SIZE = 0.10

# source_file will be used as the grouping variable so that
# the same document/project cannot occur in both train and test.
GROUP_COL = SOURCE_COL


# ============================================================
# Analysis configuration
# ============================================================

# Categories with extremely small support should not be
# interpreted as reliable category-level results.
MIN_REASON_SUPPORT = 20


# ============================================================
# Directory setup
# ============================================================

def ensure_project_dirs() -> None:
    """
    Create all directories used by the revised experimental pipeline.

    Existing directories are left unchanged.
    """
    directories = [
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        INTRA_SPLITS_DIR,
        CROSS_SPLITS_DIR,
        SIZE_MATCHED_SPLITS_DIR,
        RESULTS_RAW_DIR,
        RESULTS_SUMMARY_DIR,
        FIGURES_DIR,
        LOGS_DIR,
    ]

    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)