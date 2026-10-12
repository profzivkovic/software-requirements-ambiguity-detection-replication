from __future__ import annotations

import copy
import gc
import random
from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import (
    T5ForConditionalGeneration,
    T5Tokenizer,
    get_linear_schedule_with_warmup,
)

from .metrics import compute_binary_metrics


MODEL_NAME = "t5-small"

MAX_INPUT_LENGTH = 128
MAX_TARGET_LENGTH = 8

DEFAULT_EPOCHS = 3
DEFAULT_LR = 1e-4
DEFAULT_TRAIN_BATCH_SIZE = 8
DEFAULT_EVAL_BATCH_SIZE = 8

DEFAULT_WEIGHT_DECAY = 0.01
DEFAULT_WARMUP_RATIO = 0.06
DEFAULT_PATIENCE = 2

PROMPT_PREFIX = (
    "should this requirement be flagged for ambiguity-related review? "
    "requirement: "
)


@dataclass
class T5RunResult:
    metrics: Dict[str, float]
    y_true: np.ndarray
    y_pred: np.ndarray
    decoded_predictions: List[str]
    best_epoch: int
    best_validation_f1: float
    test_loss: float


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def cleanup() -> None:
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def build_prompt(requirement_text: str) -> str:
    return PROMPT_PREFIX + str(requirement_text).strip()


def label_to_text(label: int) -> str:
    label = int(label)

    if label == 1:
        return "yes"

    if label == 0:
        return "no"

    raise ValueError(f"Binary label expected, got: {label}")


def text_to_label(text: str) -> int:
    """
    Convert generated T5 text into the binary prediction.

    Expected outputs are 'yes' and 'no'. We normalize whitespace and case.
    Unexpected generations are mapped conservatively, while callers can
    inspect decoded_predictions to audit them.
    """
    normalized = str(text).strip().lower()

    if normalized.startswith("yes"):
        return 1

    if normalized.startswith("no"):
        return 0

    if "yes" in normalized and "no" not in normalized:
        return 1

    return 0


class T5RequirementDataset(Dataset):
    def __init__(
        self,
        texts: Sequence[str],
        labels: Sequence[int],
        tokenizer: T5Tokenizer,
        max_input_length: int = MAX_INPUT_LENGTH,
        max_target_length: int = MAX_TARGET_LENGTH,
    ) -> None:
        if len(texts) != len(labels):
            raise ValueError(
                f"texts and labels must have equal length: "
                f"{len(texts)} != {len(labels)}"
            )

        self.texts = [str(text) for text in texts]
        self.labels = [int(label) for label in labels]

        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        prompt = build_prompt(self.texts[index])
        target_text = label_to_text(self.labels[index])

        source = self.tokenizer(
            prompt,
            max_length=self.max_input_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        target = self.tokenizer(
            target_text,
            max_length=self.max_target_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        labels = target["input_ids"].squeeze(0).clone()

        # Ignore target padding tokens when computing sequence loss.
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": source["input_ids"].squeeze(0),
            "attention_mask": source["attention_mask"].squeeze(0),
            "labels": labels,
        }


def _make_train_loader(
    dataset: Dataset,
    batch_size: int,
    seed: int,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
    )


def _make_eval_loader(
    dataset: Dataset,
    batch_size: int,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
    )


def train_one_epoch(
    model: T5ForConditionalGeneration,
    dataloader: DataLoader,
    optimizer: AdamW,
    scheduler,
    device: torch.device,
) -> float:
    model.train()

    total_loss = 0.0
    completed_steps = 0

    for batch_index, batch in enumerate(dataloader, start=1):
        batch = {
            key: value.to(device)
            for key, value in batch.items()
        }

        optimizer.zero_grad(set_to_none=True)

        outputs = model(**batch)
        loss = outputs.loss

        if not torch.isfinite(loss):
            raise RuntimeError(
                f"Non-finite T5 training loss at batch {batch_index}: "
                f"{loss.item()}"
            )

        loss.backward()

        # Preserve the gradient clipping used by the legacy T5 pipeline.
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=0.5,
        )

        optimizer.step()
        scheduler.step()

        total_loss += float(loss.item())
        completed_steps += 1

        if batch_index % 25 == 0 or batch_index == len(dataloader):
            print(
                f"      batch {batch_index}/{len(dataloader)} "
                f"loss={loss.item():.4f}"
            )

    return total_loss / max(completed_steps, 1)


@torch.no_grad()
def evaluate_t5(
    model: T5ForConditionalGeneration,
    tokenizer: T5Tokenizer,
    dataloader: DataLoader,
    device: torch.device,
    max_target_length: int = MAX_TARGET_LENGTH,
) -> tuple[
    float,
    Dict[str, float],
    np.ndarray,
    np.ndarray,
    List[str],
]:
    model.eval()

    total_loss = 0.0
    completed_steps = 0

    all_true: List[int] = []
    all_pred: List[int] = []
    all_decoded_predictions: List[str] = []

    for batch in dataloader:
        batch = {
            key: value.to(device)
            for key, value in batch.items()
        }

        outputs = model(**batch)

        total_loss += float(outputs.loss.item())
        completed_steps += 1

        generated_ids = model.generate(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            max_length=max_target_length,
            num_beams=1,
            do_sample=False,
        )

        decoded_predictions = tokenizer.batch_decode(
            generated_ids,
            skip_special_tokens=True,
        )

        gold_ids = batch["labels"].clone()
        gold_ids[gold_ids == -100] = tokenizer.pad_token_id

        decoded_gold = tokenizer.batch_decode(
            gold_ids,
            skip_special_tokens=True,
        )

        y_pred = [
            text_to_label(text)
            for text in decoded_predictions
        ]

        y_true = [
            text_to_label(text)
            for text in decoded_gold
        ]

        all_pred.extend(y_pred)
        all_true.extend(y_true)
        all_decoded_predictions.extend(decoded_predictions)

    y_true_array = np.asarray(all_true, dtype=int)
    y_pred_array = np.asarray(all_pred, dtype=int)

    metrics = compute_binary_metrics(
        y_true_array,
        y_pred_array,
    )

    average_loss = total_loss / max(completed_steps, 1)

    return (
        average_loss,
        metrics,
        y_true_array,
        y_pred_array,
        all_decoded_predictions,
    )


def run_t5_training(
    train_texts: Sequence[str],
    train_labels: Sequence[int],
    validation_texts: Sequence[str],
    validation_labels: Sequence[int],
    test_texts: Sequence[str],
    test_labels: Sequence[int],
    seed: int,
    model_name: str = MODEL_NAME,
    epochs: int = DEFAULT_EPOCHS,
    learning_rate: float = DEFAULT_LR,
    train_batch_size: int = DEFAULT_TRAIN_BATCH_SIZE,
    eval_batch_size: int = DEFAULT_EVAL_BATCH_SIZE,
    max_input_length: int = MAX_INPUT_LENGTH,
    max_target_length: int = MAX_TARGET_LENGTH,
    weight_decay: float = DEFAULT_WEIGHT_DECAY,
    warmup_ratio: float = DEFAULT_WARMUP_RATIO,
    patience: int = DEFAULT_PATIENCE,
) -> T5RunResult:
    set_global_seed(seed)

    device = get_device()

    print(f"Model: {model_name}")
    print(f"Seed: {seed}")
    print(f"Device: {device}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    model.to(device)

    train_dataset = T5RequirementDataset(
        train_texts,
        train_labels,
        tokenizer,
        max_input_length=max_input_length,
        max_target_length=max_target_length,
    )

    validation_dataset = T5RequirementDataset(
        validation_texts,
        validation_labels,
        tokenizer,
        max_input_length=max_input_length,
        max_target_length=max_target_length,
    )

    test_dataset = T5RequirementDataset(
        test_texts,
        test_labels,
        tokenizer,
        max_input_length=max_input_length,
        max_target_length=max_target_length,
    )

    train_loader = _make_train_loader(
        train_dataset,
        batch_size=train_batch_size,
        seed=seed,
    )

    validation_loader = _make_eval_loader(
        validation_dataset,
        batch_size=eval_batch_size,
    )

    test_loader = _make_eval_loader(
        test_dataset,
        batch_size=eval_batch_size,
    )

    optimizer = AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    total_training_steps = len(train_loader) * epochs
    warmup_steps = int(
        warmup_ratio * total_training_steps
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

    for epoch in range(1, epochs + 1):
        print(f"\nEpoch {epoch}/{epochs}")
        print("    training...")

        train_loss = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
        )

        print("    validating...")

        (
            validation_loss,
            validation_metrics,
            _,
            _,
            _,
        ) = evaluate_t5(
            model=model,
            tokenizer=tokenizer,
            dataloader=validation_loader,
            device=device,
            max_target_length=max_target_length,
        )

        validation_f1 = float(
            validation_metrics["f1"]
        )

        print(
            f"    train_loss={train_loss:.4f} "
            f"val_loss={validation_loss:.4f} "
            f"val_f1={validation_f1:.4f}"
        )

        if validation_f1 > best_validation_f1:
            best_validation_f1 = validation_f1
            best_epoch = epoch

            best_state_dict = copy.deepcopy(
                {
                    key: value.detach().cpu()
                    for key, value in model.state_dict().items()
                }
            )

            bad_epochs = 0

        else:
            bad_epochs += 1

            if bad_epochs >= patience:
                print(
                    f"    early stopping: no F1 improvement "
                    f"for {patience} epochs"
                )
                break

    if best_state_dict is None:
        raise RuntimeError(
            "No valid T5 checkpoint was selected."
        )

    model.load_state_dict(best_state_dict)
    model.to(device)

    print(
        f"\nBest epoch: {best_epoch} "
        f"validation F1={best_validation_f1:.4f}"
    )

    print("Evaluating held-out test set...")

    (
        test_loss,
        test_metrics,
        y_true,
        y_pred,
        decoded_predictions,
    ) = evaluate_t5(
        model=model,
        tokenizer=tokenizer,
        dataloader=test_loader,
        device=device,
        max_target_length=max_target_length,
    )

    result = T5RunResult(
        metrics=test_metrics,
        y_true=y_true,
        y_pred=y_pred,
        decoded_predictions=decoded_predictions,
        best_epoch=best_epoch,
        best_validation_f1=best_validation_f1,
        test_loss=test_loss,
    )

    del model
    del tokenizer
    del optimizer
    del scheduler

    cleanup()

    return result