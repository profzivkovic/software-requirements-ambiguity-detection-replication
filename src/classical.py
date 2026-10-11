from __future__ import annotations

from typing import Dict

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from lightgbm import LGBMClassifier
from xgboost import XGBClassifier


# ============================================================
# TF-IDF configuration
# ============================================================

TFIDF_CONFIG = {
    "lowercase": True,
    "ngram_range": (1, 2),
    "min_df": 2,
    "max_df": 0.95,
    "sublinear_tf": True,
}


# ============================================================
# Classical models
# ============================================================

def build_classical_models(
    seed: int,
) -> Dict[str, object]:
    """
    Build the fixed classical model configurations used in
    the original study.

    Hyperparameters are intentionally kept unchanged so that
    the revised experiments isolate changes in data cleaning
    and evaluation protocol rather than model tuning.
    """

    return {
        "MultinomialNB": MultinomialNB(),

        "LinearSVM": LinearSVC(
            random_state=seed,
        ),

        "RandomForest": RandomForestClassifier(
            n_estimators=400,
            random_state=seed,
            n_jobs=-1,
            class_weight="balanced",
        ),

        "LightGBM": LGBMClassifier(
            n_estimators=600,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=seed,
            n_jobs=-1,
            verbose=-1,
        ),

        "XGBoost": XGBClassifier(
            n_estimators=600,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=seed,
            n_jobs=-1,
        ),
    }


def build_classical_pipeline(
    model: object,
) -> Pipeline:
    """
    Construct TF-IDF + classifier pipeline.

    TF-IDF is fitted exclusively on the training partition.
    """
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    **TFIDF_CONFIG
                ),
            ),
            (
                "classifier",
                model,
            ),
        ]
    )