from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List


# ============================================================
# Rule definition
# ============================================================

@dataclass(frozen=True)
class RuleMatch:
    category: str
    matched_text: str


# IMPORTANT:
# These rules are fixed BEFORE evaluating the baseline.
# They must not be modified later simply to improve test performance.
#
# The rules are intentionally transparent and conservative.
# They represent surface-level requirements smells rather than
# an attempt to reproduce the manual annotation process.

RULE_PATTERNS: Dict[str, List[str]] = {

    # Weak / optional modal language
    "weak_modal": [
        r"\bshould\b",
        r"\bmay\b",
        r"\bmight\b",
        r"\bcould\b",
        r"\bwould\b",
    ],

    # Subjective or difficult-to-measure quality terms
    "subjective_term": [
        r"\bappropriate\b",
        r"\badequate\b",
        r"\breasonable\b",
        r"\bsufficient\b",
        r"\bacceptable\b",
        r"\befficient\b",
        r"\beffective\b",
        r"\buser[- ]friendly\b",
        r"\bintuitive\b",
        r"\beasy\b",
        r"\bsimple\b",
        r"\bconvenient\b",
        r"\bflexible\b",
        r"\brobust\b",
        r"\bseamless\b",
        r"\bfast\b",
        r"\bquick\b",
        r"\bquickly\b",
    ],

    # Imprecise quantities or ranges
    "undefined_quantity": [
        r"\bseveral\b",
        r"\bmany\b",
        r"\bfew\b",
        r"\bsome\b",
        r"\bvarious\b",
        r"\bnumerous\b",
        r"\bapproximately\b",
        r"\bapprox\.?\b",
        r"\babout\b",
        r"\baround\b",
        r"\ba number of\b",
    ],

    # Open-ended / underspecified conditions
    "open_ended_condition": [
        r"\bas appropriate\b",
        r"\bas applicable\b",
        r"\bwhere applicable\b",
        r"\bif applicable\b",
        r"\bas required\b",
        r"\bwhen required\b",
        r"\bif required\b",
        r"\bas needed\b",
        r"\bwhen needed\b",
        r"\bif needed\b",
        r"\bwhere necessary\b",
        r"\bas necessary\b",
        r"\bif necessary\b",
        r"\bwhere possible\b",
        r"\bwhen possible\b",
        r"\bif possible\b",
        r"\bas soon as possible\b",
        r"\bunless otherwise specified\b",
    ],

    # Explicitly open-ended enumeration
    "open_ended_list": [
        r"\betc\.?\b",
        r"\band so on\b",
        r"\band\/or\b",
    ],

    # Placeholder / unresolved specification language
    "placeholder": [
        r"\bTBD\b",
        r"\bTBC\b",
        r"\bto be determined\b",
        r"\bto be confirmed\b",
        r"\bto be defined\b",
    ],
}


# Compile once when the module is imported.
COMPILED_RULES = {
    category: [
        re.compile(pattern, flags=re.IGNORECASE)
        for pattern in patterns
    ]
    for category, patterns in RULE_PATTERNS.items()
}


# ============================================================
# Rule application
# ============================================================

def analyze_text(text: object) -> List[RuleMatch]:
    """
    Apply all fixed smell rules to one requirement.

    Only the requirement text is inspected.
    Gold labels and manual reason annotations are never used.
    """
    if text is None:
        text = ""

    text = str(text)

    matches: List[RuleMatch] = []

    for category, patterns in COMPILED_RULES.items():

        for pattern in patterns:

            for match in pattern.finditer(text):

                matches.append(
                    RuleMatch(
                        category=category,
                        matched_text=match.group(0),
                    )
                )

    return matches


def predict_rule_based(text: object) -> int:
    """
    Binary prediction:
        1 -> at least one requirements-smell rule fired
        0 -> no rule fired
    """
    return int(
        len(analyze_text(text)) > 0
    )


def explain_rule_prediction(
    text: object,
) -> Dict[str, object]:
    """
    Return prediction together with triggered categories
    and matched surface expressions.
    """
    matches = analyze_text(text)

    categories = sorted(
        set(
            match.category
            for match in matches
        )
    )

    matched_texts = sorted(
        set(
            match.matched_text
            for match in matches
        ),
        key=str.lower,
    )

    return {
        "prediction": int(len(matches) > 0),
        "rule_categories": ";".join(categories),
        "rule_matches": ";".join(matched_texts),
    }