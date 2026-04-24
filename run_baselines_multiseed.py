
from __future__ import annotations
import os, warnings
from typing import Dict, List, Tuple
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from experiment_utils import SEEDS, RESULTS_DIR, FIGURES_DIR, TEXT_COL, LABEL_COL, REASONS_COL, ensure_dirs, set_global_seed, load_all_datasets, get_xy, compute_metrics, compute_confusion_values, classification_report_text, plot_confusion_matrix, summarize_multiseed_results, per_reason_performance, mask_trigger_terms, remove_trigger_terms
warnings.filterwarnings("ignore")
try:
    from lightgbm import LGBMClassifier
    HAS_LGBM = True
except Exception:
    HAS_LGBM = False
try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except Exception:
    HAS_XGB = False
TFIDF_CONFIG = {"lowercase": True, "ngram_range": (1,2), "min_df": 2, "max_df": 0.95, "sublinear_tf": True}
TEST_SIZE = 0.2
def build_models(seed: int) -> Dict[str, object]:
    models = {
        "MultinomialNB": MultinomialNB(),
        "LinearSVM": LinearSVC(random_state=seed),
        "RandomForest": RandomForestClassifier(n_estimators=400, random_state=seed, n_jobs=-1, class_weight="balanced"),
    }
    if HAS_LGBM:
        models["LightGBM"] = LGBMClassifier(n_estimators=600, learning_rate=0.05, num_leaves=31, subsample=0.9, colsample_bytree=0.9, random_state=seed, n_jobs=-1, verbose=-1)
    if HAS_XGB:
        models["XGBoost"] = XGBClassifier(n_estimators=600, learning_rate=0.05, max_depth=6, subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0, objective="binary:logistic", eval_metric="logloss", random_state=seed, n_jobs=-1)
    return models
def build_pipeline(model: object) -> Pipeline:
    return Pipeline([("tfidf", TfidfVectorizer(**TFIDF_CONFIG)), ("clf", model)])
def run_one_setting(model_name, model, setting, train_dataset, test_dataset, seed, x_train, y_train, x_test, y_test, test_df, variant="original"):
    print(f"[RUN] setting={setting} variant={variant} train={train_dataset} test={test_dataset} model={model_name} seed={seed}")
    pipe = build_pipeline(model); pipe.fit(x_train, y_train); y_pred = pipe.predict(x_test)
    metrics = compute_metrics(y_test, y_pred); conf = compute_confusion_values(y_test, y_pred)
    row = {"setting": setting, "variant": variant, "train_dataset": train_dataset, "test_dataset": test_dataset, "model": model_name, "seed": seed, **metrics, **conf, "n_test": len(y_test)}
    instance_df = pd.DataFrame({"setting": setting, "variant": variant, "train_dataset": train_dataset, "test_dataset": test_dataset, "model": model_name, "seed": seed, "text": test_df[TEXT_COL].astype(str).values, "y_true": y_test, "y_pred": y_pred, "reasons": test_df[REASONS_COL].values if REASONS_COL in test_df.columns else np.nan})
    print(f"[DONE] setting={setting} variant={variant} train={train_dataset} test={test_dataset} model={model_name} seed={seed}")
    return row, instance_df
def save_report_and_confusion(y_true, y_pred, title_stub):
    with open(os.path.join(RESULTS_DIR, f"{title_stub}_report.txt"), "w", encoding="utf-8") as f:
        f.write(classification_report_text(y_true, y_pred))
    plot_confusion_matrix(y_true, y_pred, title_stub.replace("_", " "), os.path.join(FIGURES_DIR, f"{title_stub}_confusion.png"))
def run_intra_experiments_for_dataset(df, dataset_name):
    all_rows, all_instances = [], []
    x_all, y_all = get_xy(df)
    for seed in SEEDS:
        set_global_seed(seed)
        x_train, x_test, y_train, y_test, _, idx_test = train_test_split(x_all, y_all, np.arange(len(df)), test_size=TEST_SIZE, stratify=y_all, random_state=seed)
        test_df = df.iloc[idx_test].reset_index(drop=True)
        for model_name, model in build_models(seed).items():
            row, inst = run_one_setting(model_name, model, "intra", dataset_name, dataset_name, seed, list(x_train), np.array(y_train), list(x_test), np.array(y_test), test_df, "original")
            all_rows.append(row); all_instances.append(inst)
            if seed == 42:
                save_report_and_confusion(np.array(y_test), inst["y_pred"].astype(int).values, f"intra_{dataset_name}_{model_name}_seed{seed}")
        for variant_name, x_test_variant in [("masked_triggers", mask_trigger_terms(list(x_test))), ("removed_triggers", remove_trigger_terms(list(x_test)))]:
            for model_name, model in build_models(seed).items():
                row, inst = run_one_setting(model_name, model, "intra", dataset_name, dataset_name, seed, list(x_train), np.array(y_train), x_test_variant, np.array(y_test), test_df, variant_name)
                all_rows.append(row); all_instances.append(inst)
    return all_rows, all_instances
def run_cross_experiments(train_df, test_df, train_name, test_name):
    all_rows, all_instances = [], []
    x_train, y_train = get_xy(train_df); x_test, y_test = get_xy(test_df); test_df = test_df.reset_index(drop=True)
    for seed in SEEDS:
        set_global_seed(seed)
        for model_name, model in build_models(seed).items():
            row, inst = run_one_setting(model_name, model, "cross", train_name, test_name, seed, x_train, y_train, x_test, y_test, test_df, "original")
            all_rows.append(row); all_instances.append(inst)
            if seed == 42:
                save_report_and_confusion(y_test, inst["y_pred"].astype(int).values, f"cross_{train_name}_to_{test_name}_{model_name}_seed{seed}")
        for variant_name, x_test_variant in [("masked_triggers", mask_trigger_terms(x_test)), ("removed_triggers", remove_trigger_terms(x_test))]:
            for model_name, model in build_models(seed).items():
                row, inst = run_one_setting(model_name, model, "cross", train_name, test_name, seed, x_train, y_train, x_test_variant, y_test, test_df, variant_name)
                all_rows.append(row); all_instances.append(inst)
    return all_rows, all_instances
def main():
    ensure_dirs()
    pure_df, us_df = load_all_datasets()
    print("Loaded datasets:"); print(f"  PURE:        {pure_df.shape}"); print(f"  UserStories: {us_df.shape}"); print(); print("PURE label counts:"); print(pure_df[LABEL_COL].value_counts(dropna=False)); print(); print("UserStories label counts:"); print(us_df[LABEL_COL].value_counts(dropna=False)); print()
    all_rows, all_instances = [], []
    for rows, instances in [run_intra_experiments_for_dataset(pure_df, "PURE"), run_intra_experiments_for_dataset(us_df, "UserStories"), run_cross_experiments(pure_df, us_df, "PURE", "UserStories"), run_cross_experiments(us_df, pure_df, "UserStories", "PURE")]:
        all_rows.extend(rows); all_instances.extend(instances)
    results_df = pd.DataFrame(all_rows); instances_df = pd.concat(all_instances, ignore_index=True)
    results_df.to_csv(os.path.join(RESULTS_DIR, "classical_multiseed_results.csv"), index=False)
    instances_df.to_csv(os.path.join(RESULTS_DIR, "classical_instance_predictions.csv"), index=False)
    summarize_multiseed_results(results_df[results_df["variant"]=="original"].copy(), ["setting","train_dataset","test_dataset","model"]).to_csv(os.path.join(RESULTS_DIR, "classical_multiseed_summary.csv"), index=False)
    summarize_multiseed_results(results_df.copy(), ["setting","variant","train_dataset","test_dataset","model"]).to_csv(os.path.join(RESULTS_DIR, "classical_multiseed_summary_by_variant.csv"), index=False)
    per_reason_performance(instances_df[instances_df["variant"]=="original"].copy(), ["setting","train_dataset","test_dataset","model"]).to_csv(os.path.join(RESULTS_DIR, "classical_reason_performance.csv"), index=False)
    top_models = (pd.read_csv(os.path.join(RESULTS_DIR, "classical_multiseed_summary.csv")).sort_values(["setting","train_dataset","test_dataset","f1_mean"], ascending=[True,True,True,False]).groupby(["setting","train_dataset","test_dataset"], as_index=False).head(3))
    top_models.to_csv(os.path.join(RESULTS_DIR, "classical_top3_by_setting.csv"), index=False)
    print("Done.")
if __name__ == "__main__":
    main()
