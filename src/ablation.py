from __future__ import annotations

import re

from src.rule_baseline import RULE_PATTERNS


ALL_CUE_PATTERNS = [
    pattern
    for patterns in RULE_PATTERNS.values()
    for pattern in patterns
]


COMBINED_CUE_PATTERN = re.compile(
    "|".join(
        f"(?:{pattern})"
        for pattern in ALL_CUE_PATTERNS
    ),
    flags=re.IGNORECASE,
)


MASK_TOKEN = "CUE_TOKEN"


def contains_surface_cue(text: object) -> bool:
    """
    Return True if at least one frozen lexical cue occurs.
    """
    if text is None:
        return False

    return (
        COMBINED_CUE_PATTERN.search(str(text))
        is not None
    )


def _cleanup_spacing(text: str) -> str:
    """
    Clean only spacing artifacts introduced by cue transformation.
    """
    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    text = re.sub(
        r"\s+([,.;:!?])",
        r"\1",
        text,
    )

    return text.strip()


def mask_surface_cues(text: object) -> str:
    """
    Replace every lexical cue by one common neutral token.

    This removes lexical identity but preserves the fact that
    something occurred at that position.
    """
    if text is None:
        return ""

    transformed = COMBINED_CUE_PATTERN.sub(
        f" {MASK_TOKEN} ",
        str(text),
    )

    return _cleanup_spacing(
        transformed
    )


def remove_surface_cues(text: object) -> str:
    """
    Remove all frozen lexical cues entirely.
    """
    if text is None:
        return ""

    transformed = COMBINED_CUE_PATTERN.sub(
        " ",
        str(text),
    )

    return _cleanup_spacing(
        transformed
    )


def transform_text(
    text: object,
    variant: str,
) -> str:
    """
    Supported variants:
        original
        masked
        removed
    """
    if variant == "original":
        return str(text)

    if variant == "masked":
        return mask_surface_cues(text)

    if variant == "removed":
        return remove_surface_cues(text)

    raise ValueError(
        f"Unknown ablation variant: {variant}"
    )