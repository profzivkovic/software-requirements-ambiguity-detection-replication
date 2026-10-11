from __future__ import annotations

import gc
import random
import time
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch

from torch.utils.data import Dataset, DataLoader

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

from src.config import (
    INTRA_SPLITS_DIR,
    CROSS_SPLITS_DIR,
    INSTANCE_ID_COL,
    SOURCE_COL,
    TEXT_COL,
    LABEL_COL,
)

from src.metrics import compute_binary_metrics


# ============================================================
# Model configuration
# ============================================================

MODEL_CONFIGS: Dict[str, str] = {
    "DistilBERT": "distilbert-base-uncased",
    "BERT": "bert-base-uncased",
    "RoBERTa": "roberta-base",
    "ALBERT": "albert-base-v2",
}

MAX_LENGTH = 128

DEFAULT_EPOCHS = 3
DEFAULT_LEARNING_RATE = 3e-5

DEFAULT_TRAIN_BATCH_SIZE = 8
DEFAULT_EVAL_BATCH_SIZE = 8

DEFAULT_WEIGHT_DECAY = 0.01
DEFAULT_WARMUP_RATIO = 0.06

EARLY_STOPPING_PATIENCE = 2


# ============================================================
# Reproducibility
# ============================================================

def set_reproducible_seed(seed: int) -> None:
    """
    Set Python, NumPy, and PyTorch random seeds.

    GPU execution can still contain some platform-dependent
    nondeterminism, which should be reported transparently.
    """

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ============================================================
# Device utilities
# ============================================================

def get_device() -> torch.device:
    return torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )


def get_device_description() -> str:
    if torch.cuda.is_available():
        return f"cuda: {torch.cuda.get_device_name(0)}"

    return "cpu"


def cleanup_torch() -> None:
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ============================================================
# Hugging Face model loading
# ============================================================

def load_tokenizer(model_key: str):
    if model_key not in MODEL_CONFIGS:
        raise ValueError(
            f"Unknown model key: {model_key}. "
            f"Available models: {list(MODEL_CONFIGS.keys())}"
        )

    checkpoint = MODEL_CONFIGS[model_key]

    return AutoTokenizer.from_pretrained(
        checkpoint
    )


def load_sequence_classifier(model_key: str):
    if model_key not in MODEL_CONFIGS:
        raise ValueError(
            f"Unknown model key: {model_key}. "
            f"Available models: {list(MODEL_CONFIGS.keys())}"
        )

    checkpoint = MODEL_CONFIGS[model_key]

    return AutoModelForSequenceClassification.from_pretrained(
        checkpoint,
        num_labels=2,
    )


# ============================================================
# PyTorch dataset
# ============================================================

class RequirementDataset(Dataset):
    def __init__(
        self,
        dataframe: pd.DataFrame,
        tokenizer,
        max_length: int = MAX_LENGTH,
    ):
        self.texts = (
            dataframe[TEXT_COL]
            .astype(str)
            .tolist()
        )

        self.labels = (
            dataframe[LABEL_COL]
            .astype(int)
            .tolist()
        )

        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, index: int) -> dict:
        text = self.texts[index]
        label = self.labels[index]

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        item = {
            key: value.squeeze(0)
            for key, value in encoding.items()
        }

        item["labels"] = torch.tensor(
            label,
            dtype=torch.long,
        )

        return item


# ============================================================
# Partition container
# ============================================================

@dataclass
class TransformerPartitions:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame


# ============================================================
# Partition validation helpers
# ============================================================

def _validate_required_columns(
    df: pd.DataFrame,
    dataset_name: str,
) -> None:
    required_columns = {
        INSTANCE_ID_COL,
        SOURCE_COL,
        TEXT_COL,
        LABEL_COL,
    }

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            f"{dataset_name} is missing required columns: "
            f"{sorted(missing)}"
        )

    if df[INSTANCE_ID_COL].duplicated().any():
        raise AssertionError(
            f"{dataset_name} contains duplicate instance IDs."
        )


def _validate_manifest(
    manifest: pd.DataFrame,
    manifest_name: str,
) -> None:
    required_columns = {
        INSTANCE_ID_COL,
        "split",
    }

    missing = required_columns - set(manifest.columns)

    if missing:
        raise ValueError(
            f"{manifest_name} is missing required columns: "
            f"{sorted(missing)}"
        )

    if manifest[INSTANCE_ID_COL].duplicated().any():
        raise AssertionError(
            f"{manifest_name} contains duplicate instance IDs."
        )


def _validate_instance_disjoint(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> None:
    train_ids = set(
        train_df[INSTANCE_ID_COL]
    )

    validation_ids = set(
        validation_df[INSTANCE_ID_COL]
    )

    test_ids = set(
        test_df[INSTANCE_ID_COL]
    )

    if train_ids & validation_ids:
        raise AssertionError(
            "Train/validation instance overlap detected."
        )

    if train_ids & test_ids:
        raise AssertionError(
            "Train/test instance overlap detected."
        )

    if validation_ids & test_ids:
        raise AssertionError(
            "Validation/test instance overlap detected."
        )


def _validate_intra_source_disjoint(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> None:
    train_sources = set(
        train_df[SOURCE_COL]
    )

    validation_sources = set(
        validation_df[SOURCE_COL]
    )

    test_sources = set(
        test_df[SOURCE_COL]
    )

    if train_sources & validation_sources:
        raise AssertionError(
            "Train/validation source overlap detected."
        )

    if train_sources & test_sources:
        raise AssertionError(
            "Train/test source overlap detected."
        )

    if validation_sources & test_sources:
        raise AssertionError(
            "Validation/test source overlap detected."
        )


def _validate_cross_source_disjoint(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
) -> None:
    """
    In cross-corpus evaluation only the source corpus is partitioned
    into train and validation.

    The complete other corpus is used as the external test corpus.
    """

    train_ids = set(
        train_df[INSTANCE_ID_COL]
    )

    validation_ids = set(
        validation_df[INSTANCE_ID_COL]
    )

    if train_ids & validation_ids:
        raise AssertionError(
            "Cross-corpus source train/validation "
            "instance overlap detected."
        )

    train_sources = set(
        train_df[SOURCE_COL]
    )

    validation_sources = set(
        validation_df[SOURCE_COL]
    )

    if train_sources & validation_sources:
        raise AssertionError(
            "Cross-corpus source train/validation "
            "source overlap detected."
        )


# ============================================================
# Intra-dataset partitions
# ============================================================

def load_intra_partitions(
    dataset_df: pd.DataFrame,
    dataset_name: str,
    seed: int,
) -> TransformerPartitions:
    """
    Load the already-created source-disjoint intra-corpus split.

    The transformer code never creates a new random split.

    The saved manifest contains approximately:

        70% train
        10% validation
        20% test
    """

    _validate_required_columns(
        dataset_df,
        dataset_name,
    )

    manifest_path = (
        INTRA_SPLITS_DIR
        / f"{dataset_name}_seed{seed}.csv"
    )

    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Missing intra split manifest: {manifest_path}"
        )

    manifest = pd.read_csv(
        manifest_path
    )

    _validate_manifest(
        manifest,
        str(manifest_path),
    )

    allowed_splits = {
        "train",
        "validation",
        "test",
    }

    observed_splits = set(
        manifest["split"].astype(str)
    )

    if observed_splits != allowed_splits:
        raise AssertionError(
            f"{dataset_name}, seed {seed}: "
            f"unexpected intra split labels: "
            f"{sorted(observed_splits)}"
        )

    split_info = manifest[
        [
            INSTANCE_ID_COL,
            "split",
        ]
    ].copy()

    merged = dataset_df.merge(
        split_info,
        on=INSTANCE_ID_COL,
        how="inner",
        validate="one_to_one",
        sort=False,
    )

    if len(merged) != len(dataset_df):
        raise AssertionError(
            f"{dataset_name}, seed {seed}: "
            "clean dataset and intra manifest do not match."
        )

    if len(merged) != len(manifest):
        raise AssertionError(
            f"{dataset_name}, seed {seed}: "
            "intra manifest contains unexpected or missing instances."
        )

    train_df = (
        merged[
            merged["split"] == "train"
        ]
        .drop(columns=["split"])
        .reset_index(drop=True)
    )

    validation_df = (
        merged[
            merged["split"] == "validation"
        ]
        .drop(columns=["split"])
        .reset_index(drop=True)
    )

    test_df = (
        merged[
            merged["split"] == "test"
        ]
        .drop(columns=["split"])
        .reset_index(drop=True)
    )

    if train_df.empty:
        raise AssertionError(
            "Training partition is empty."
        )

    if validation_df.empty:
        raise AssertionError(
            "Validation partition is empty."
        )

    if test_df.empty:
        raise AssertionError(
            "Test partition is empty."
        )

    _validate_instance_disjoint(
        train_df,
        validation_df,
        test_df,
    )

    _validate_intra_source_disjoint(
        train_df,
        validation_df,
        test_df,
    )

    return TransformerPartitions(
        train=train_df,
        validation=validation_df,
        test=test_df,
    )


# ============================================================
# Cross-dataset partitions
# ============================================================

def load_cross_partitions(
    source_df: pd.DataFrame,
    source_name: str,
    target_df: pd.DataFrame,
    seed: int,
) -> TransformerPartitions:
    """
    Load the saved cross-corpus transformer partition.

    Protocol:

        source corpus:
            ~90% train
            ~10% validation

        target corpus:
            100% test

    No target-corpus instances are used for training,
    validation, early stopping, or checkpoint selection.
    """

    _validate_required_columns(
        source_df,
        source_name,
    )

    _validate_required_columns(
        target_df,
        "target dataset",
    )

    manifest_path = (
        CROSS_SPLITS_DIR
        / f"{source_name}_source_seed{seed}.csv"
    )

    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Missing cross split manifest: {manifest_path}"
        )

    manifest = pd.read_csv(
        manifest_path
    )

    _validate_manifest(
        manifest,
        str(manifest_path),
    )

    allowed_splits = {
        "train",
        "validation",
    }

    observed_splits = set(
        manifest["split"].astype(str)
    )

    if observed_splits != allowed_splits:
        raise AssertionError(
            f"{source_name}, seed {seed}: "
            f"unexpected cross split labels: "
            f"{sorted(observed_splits)}"
        )

    split_info = manifest[
        [
            INSTANCE_ID_COL,
            "split",
        ]
    ].copy()

    merged = source_df.merge(
        split_info,
        on=INSTANCE_ID_COL,
        how="inner",
        validate="one_to_one",
        sort=False,
    )

    if len(merged) != len(source_df):
        raise AssertionError(
            f"{source_name}, seed {seed}: "
            "source dataset and cross manifest do not match."
        )

    if len(merged) != len(manifest):
        raise AssertionError(
            f"{source_name}, seed {seed}: "
            "cross manifest contains unexpected or missing instances."
        )

    train_df = (
        merged[
            merged["split"] == "train"
        ]
        .drop(columns=["split"])
        .reset_index(drop=True)
    )

    validation_df = (
        merged[
            merged["split"] == "validation"
        ]
        .drop(columns=["split"])
        .reset_index(drop=True)
    )

    test_df = (
        target_df
        .copy()
        .reset_index(drop=True)
    )

    if train_df.empty:
        raise AssertionError(
            "Cross-corpus training partition is empty."
        )

    if validation_df.empty:
        raise AssertionError(
            "Cross-corpus validation partition is empty."
        )

    if test_df.empty:
        raise AssertionError(
            "Cross-corpus test corpus is empty."
        )

    _validate_cross_source_disjoint(
        train_df,
        validation_df,
    )

    return TransformerPartitions(
        train=train_df,
        validation=validation_df,
        test=test_df,
    )


# ============================================================
# Partition summaries
# ============================================================

def summarize_partition(
    df: pd.DataFrame,
) -> dict:
    return {
        "rows": len(df),
        "sources": int(
            df[SOURCE_COL].nunique()
        ),
        "positive_rate": float(
            df[LABEL_COL].mean()
        ),
        "label_0": int(
            (df[LABEL_COL] == 0).sum()
        ),
        "label_1": int(
            (df[LABEL_COL] == 1).sum()
        ),
    }


def print_partition_summary(
    name: str,
    partitions: TransformerPartitions,
) -> None:
    print(
        f"\n{name}"
    )

    for split_name, partition_df in [
        (
            "train",
            partitions.train,
        ),
        (
            "validation",
            partitions.validation,
        ),
        (
            "test",
            partitions.test,
        ),
    ]:
        summary = summarize_partition(
            partition_df
        )

        print(
            f"  {split_name:10s} "
            f"rows={summary['rows']:5d} "
            f"sources={summary['sources']:3d} "
            f"positive_rate="
            f"{summary['positive_rate']:.4f} "
            f"[0={summary['label_0']}, "
            f"1={summary['label_1']}]"
        )


# ============================================================
# Evaluation
# ============================================================

def evaluate_transformer(
    model,
    dataloader,
    device: torch.device,
) -> Tuple[
    float,
    dict,
    np.ndarray,
    np.ndarray,
]:
    """
    Evaluate a transformer on one DataLoader.

    Returns:
        average_loss
        metrics
        y_true
        y_pred
    """

    model.eval()

    all_labels = []
    all_predictions = []

    total_loss = 0.0
    n_batches = 0

    with torch.no_grad():
        for batch in dataloader:
            batch = {
                key: value.to(device)
                for key, value in batch.items()
            }

            outputs = model(
                **batch
            )

            loss = outputs.loss
            logits = outputs.logits

            predictions = torch.argmax(
                logits,
                dim=1,
            )

            total_loss += float(
                loss.item()
            )

            n_batches += 1

            all_labels.extend(
                batch["labels"]
                .detach()
                .cpu()
                .numpy()
                .tolist()
            )

            all_predictions.extend(
                predictions
                .detach()
                .cpu()
                .numpy()
                .tolist()
            )

    y_true = np.asarray(
        all_labels,
        dtype=int,
    )

    y_pred = np.asarray(
        all_predictions,
        dtype=int,
    )

    metrics = compute_binary_metrics(
        y_true,
        y_pred,
    )

    if n_batches == 0:
        average_loss = float("nan")
    else:
        average_loss = (
            total_loss / n_batches
        )

    return (
        average_loss,
        metrics,
        y_true,
        y_pred,
    )


# ============================================================
# One training epoch
# ============================================================

def train_one_epoch(
    model,
    dataloader,
    optimizer,
    scheduler,
    device: torch.device,
) -> float:
    model.train()

    total_loss = 0.0
    n_batches = 0

    for batch in dataloader:
        batch = {
            key: value.to(device)
            for key, value in batch.items()
        }

        optimizer.zero_grad(
            set_to_none=True
        )

        outputs = model(
            **batch
        )

        loss = outputs.loss

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0,
        )

        optimizer.step()
        scheduler.step()

        total_loss += float(
            loss.item()
        )

        n_batches += 1

    if n_batches == 0:
        return float("nan")

    return total_loss / n_batches


# ============================================================
# Complete transformer training
# ============================================================

def train_transformer(
    partitions: TransformerPartitions,
    model_key: str,
    seed: int,
    epochs: int = DEFAULT_EPOCHS,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    train_batch_size: int = DEFAULT_TRAIN_BATCH_SIZE,
    eval_batch_size: int = DEFAULT_EVAL_BATCH_SIZE,
    max_length: int = MAX_LENGTH,
) -> Tuple[
    dict,
    np.ndarray,
    np.ndarray,
]:
    """
    Fine-tune one encoder transformer.

    Model selection and early stopping use only validation F1.

    The test partition is evaluated once, after the best
    validation checkpoint has been selected.
    """

    if epochs < 1:
        raise ValueError(
            "epochs must be at least 1."
        )

    set_reproducible_seed(
        seed
    )

    start_time = time.time()

    device = get_device()

    print(
        f"Device: {get_device_description()}"
    )

    print(
        f"Loading {model_key}..."
    )

    tokenizer = load_tokenizer(
        model_key
    )

    model = load_sequence_classifier(
        model_key
    )

    model.to(
        device
    )

    train_dataset = RequirementDataset(
        dataframe=partitions.train,
        tokenizer=tokenizer,
        max_length=max_length,
    )

    validation_dataset = RequirementDataset(
        dataframe=partitions.validation,
        tokenizer=tokenizer,
        max_length=max_length,
    )

    test_dataset = RequirementDataset(
        dataframe=partitions.test,
        tokenizer=tokenizer,
        max_length=max_length,
    )

    train_generator = torch.Generator()

    train_generator.manual_seed(
        seed
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        generator=train_generator,
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=DEFAULT_WEIGHT_DECAY,
    )

    total_training_steps = (
        len(train_loader)
        * epochs
    )

    if total_training_steps <= 0:
        raise RuntimeError(
            "No training steps were created."
        )

    warmup_steps = int(
        DEFAULT_WARMUP_RATIO
        * total_training_steps
    )

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_training_steps,
    )

    best_validation_f1 = -1.0
    best_epoch = 0
    best_state_dict = None

    bad_epochs = 0
    epochs_completed = 0

    for epoch in range(
        1,
        epochs + 1,
    ):
        print(
            f"\nEpoch {epoch}/{epochs}"
        )

        train_loss = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
        )

        (
            validation_loss,
            validation_metrics,
            _,
            _,
        ) = evaluate_transformer(
            model=model,
            dataloader=validation_loader,
            device=device,
        )

        epochs_completed = epoch

        print(
            f"  train_loss={train_loss:.4f}"
        )

        print(
            f"  val_loss="
            f"{validation_loss:.4f}"
        )

        print(
            f"  val_accuracy="
            f"{validation_metrics['accuracy']:.4f}"
        )

        print(
            f"  val_precision="
            f"{validation_metrics['precision']:.4f}"
        )

        print(
            f"  val_recall="
            f"{validation_metrics['recall']:.4f}"
        )

        print(
            f"  val_f1="
            f"{validation_metrics['f1']:.4f}"
        )

        print(
            f"  val_mcc="
            f"{validation_metrics['mcc']:.4f}"
        )

        print(
            f"  val_kappa="
            f"{validation_metrics['cohen_kappa']:.4f}"
        )

        current_validation_f1 = (
            validation_metrics["f1"]
        )

        if (
            current_validation_f1
            > best_validation_f1
        ):
            best_validation_f1 = (
                current_validation_f1
            )

            best_epoch = epoch

            best_state_dict = {
                key: (
                    value
                    .detach()
                    .cpu()
                    .clone()
                )
                for key, value
                in model.state_dict().items()
            }

            bad_epochs = 0

            print(
                "  New best validation checkpoint."
            )

        else:
            bad_epochs += 1

            print(
                f"  No validation F1 improvement "
                f"({bad_epochs}/"
                f"{EARLY_STOPPING_PATIENCE})."
            )

            if (
                bad_epochs
                >= EARLY_STOPPING_PATIENCE
            ):
                print(
                    "  Early stopping."
                )

                break

    if best_state_dict is None:
        raise RuntimeError(
            "No validation checkpoint was selected."
        )

    model.load_state_dict(
        best_state_dict
    )

    print(
        f"\nBest epoch: {best_epoch}"
    )

    print(
        f"Best validation F1: "
        f"{best_validation_f1:.4f}"
    )

    print(
        "\nEvaluating held-out test set..."
    )

    (
        test_loss,
        test_metrics,
        y_true,
        y_pred,
    ) = evaluate_transformer(
        model=model,
        dataloader=test_loader,
        device=device,
    )

    runtime_seconds = (
        time.time() - start_time
    )

    result = {
        "model": model_key,
        "checkpoint": MODEL_CONFIGS[model_key],
        "seed": seed,

        "train_size": len(
            partitions.train
        ),

        "validation_size": len(
            partitions.validation
        ),

        "test_size": len(
            partitions.test
        ),

        "epochs_requested": epochs,
        "epochs_completed": epochs_completed,

        "best_epoch": best_epoch,

        "best_validation_f1":
            best_validation_f1,

        "learning_rate":
            learning_rate,

        "train_batch_size":
            train_batch_size,

        "eval_batch_size":
            eval_batch_size,

        "max_length":
            max_length,

        "weight_decay":
            DEFAULT_WEIGHT_DECAY,

        "warmup_ratio":
            DEFAULT_WARMUP_RATIO,

        "early_stopping_patience":
            EARLY_STOPPING_PATIENCE,

        "test_loss":
            test_loss,

        "device":
            get_device_description(),

        "runtime_seconds":
            runtime_seconds,

        **test_metrics,
    }

    del train_loader
    del validation_loader
    del test_loader

    del train_dataset
    del validation_dataset
    del test_dataset

    del optimizer
    del scheduler

    del best_state_dict
    del model
    del tokenizer

    cleanup_torch()

    return (
        result,
        y_true,
        y_pred,
    )