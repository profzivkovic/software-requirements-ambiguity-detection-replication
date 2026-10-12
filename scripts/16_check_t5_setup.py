from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.encoder_transformers import (
    LABEL_COL,
    TEXT_COL,
    load_intra_partitions,
)

from src.t5_model import (
    MODEL_NAME,
    T5RequirementDataset,
    build_prompt,
    label_to_text,
)

from transformers import (
    T5ForConditionalGeneration,
    T5Tokenizer,
)


def main() -> None:
    print("=" * 78)
    print("T5 LOCAL SETUP CHECK")
    print("=" * 78)

    print("Python executable:")
    print(sys.executable)

    print()
    print("PyTorch:", torch.__version__)
    print("CUDA:", torch.cuda.is_available())
    print("Model:", MODEL_NAME)
    print("TEXT_COL:", TEXT_COL)
    print("LABEL_COL:", LABEL_COL)

    userstories_path = (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "userstories_clean.csv"
    )

    df = pd.read_csv(userstories_path)

    partitions = load_intra_partitions(
        dataset_df=df,
        dataset_name="UserStories",
        seed=13,
    )

    sample_df = (
        partitions.train
        .head(2)
        .copy()
        .reset_index(drop=True)
    )

    print()
    print("Sample requirements:")

    for _, row in sample_df.iterrows():
        print("-" * 78)

        print("Text:")
        print(row[TEXT_COL])

        print(
            "Label:",
            int(row[LABEL_COL]),
        )

        print(
            "Target:",
            label_to_text(
                int(row[LABEL_COL])
            ),
        )

        print("Prompt:")
        print(
            build_prompt(
                row[TEXT_COL]
            )
        )

    print()
    print("=" * 78)
    print("Loading tokenizer and model...")
    print("=" * 78)

    tokenizer = T5Tokenizer.from_pretrained(
        MODEL_NAME
    )

    model = (
        T5ForConditionalGeneration
        .from_pretrained(
            MODEL_NAME
        )
    )

    dataset = T5RequirementDataset(
        texts=(
            sample_df[TEXT_COL]
            .astype(str)
            .tolist()
        ),
        labels=(
            sample_df[LABEL_COL]
            .astype(int)
            .tolist()
        ),
        tokenizer=tokenizer,
    )

    items = [
        dataset[i]
        for i in range(len(dataset))
    ]

    batch = {
        key: torch.stack(
            [
                item[key]
                for item in items
            ]
        )
        for key in items[0].keys()
    }

    print()
    print("Batch tensor shapes:")

    for key, value in batch.items():
        print(
            f"{key:15s}: "
            f"{tuple(value.shape)}"
        )

    model.eval()

    with torch.no_grad():
        outputs = model(**batch)

    print()
    print(
        "Forward loss:",
        float(outputs.loss.item()),
    )

    with torch.no_grad():
        generated_ids = model.generate(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            max_length=8,
            num_beams=1,
            do_sample=False,
        )

    decoded = tokenizer.batch_decode(
        generated_ids,
        skip_special_tokens=True,
    )

    print()
    print("Raw pretrained generations:")

    for text in decoded:
        print(repr(text))

    print()
    print("=" * 78)
    print("T5 SETUP CHECK PASSED")
    print("=" * 78)

    print()
    print(
        "Note: generated text above comes from the "
        "unfine-tuned pretrained model and is NOT "
        "an experimental result."
    )


if __name__ == "__main__":
    main()