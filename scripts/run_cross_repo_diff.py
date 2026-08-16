"""
THE experiment this project exists to run: does semantic diff content
improve cross-repo generalization over metadata alone?

Same cross-repo held-out-repo methodology as run_cross_repo.py, but now
trains two versions per held-out repo — metadata-only vs metadata+diff —
so we get a direct, honest before/after comparison instead of just a
single number to interpret in isolation.

Run: python scripts/run_cross_repo_diff.py
"""

import logging

import lightgbm as lgb
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from ci_intel.config import DATA_RAW_DIR, VALID_JOB_CONCLUSIONS
from ci_intel.features.engineering import build_features, compute_job_fail_rates
from ci_intel.features.diff_embeddings import DiffFeatureReducer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

MULTI_REPO_CSV_PATH = DATA_RAW_DIR / "multi_repo_ci_data.csv"


def clean_multi_repo(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df[df["job_conclusion"].isin(VALID_JOB_CONCLUSIONS)].copy()
    logger.info("Dropped %d non-signal rows out of %d", before - len(df), before)
    df["target_failed"] = (df["job_conclusion"] == "failure").astype(int)
    df["run_created_at"] = pd.to_datetime(df["run_created_at"])
    return df


def train_and_eval(X_train, y_train, X_test, y_test, label):
    neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
    model = lgb.LGBMClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        min_child_samples=20, reg_alpha=1.0, reg_lambda=1.0,
        scale_pos_weight=neg / pos if pos > 0 else 1.0,
        random_state=42, verbose=-1,
    )
    model.fit(X_train, y_train, categorical_feature="auto")
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    pr_auc = average_precision_score(y_test, y_pred_proba)
    roc_auc = roc_auc_score(y_test, y_pred_proba) if y_test.nunique() > 1 else float("nan")
    logger.info("  [%s] PR-AUC=%.3f  ROC-AUC=%.3f", label, pr_auc, roc_auc)
    return {"pr_auc": pr_auc, "roc_auc": roc_auc}


def run_comparison():
    df = pd.read_csv(MULTI_REPO_CSV_PATH, parse_dates=["run_created_at"])
    df = clean_multi_repo(df)

    if "diff_text" not in df.columns or df["diff_text"].isna().all():
        raise SystemExit(
            "No diff_text column found (or it's all empty). You need to RE-RUN "
            "scripts/run_collect_multi.py with the updated collector.py first — "
            "this column didn't exist in your earlier collection."
        )

    repos = df["repo"].unique()
    all_results = {}

    for held_out in repos:
        train_df = df[df["repo"] != held_out].copy()
        test_df = df[df["repo"] == held_out].copy()

        if test_df["target_failed"].sum() < 3:
            logger.warning("Skipping %s — too few failures to evaluate", held_out)
            continue

        logger.info("\n=== Held out: %s (%d test rows, %d failures) ===",
                    held_out, len(test_df), test_df["target_failed"].sum())

        job_fail_rate_map = compute_job_fail_rates(train_df)
        X_train_meta, y_train = build_features(train_df, job_fail_rate_map)
        X_test_meta, y_test = build_features(test_df, job_fail_rate_map)

        # metadata-only (what we already tested)
        meta_result = train_and_eval(X_train_meta, y_train, X_test_meta, y_test, "metadata-only")

        # metadata + diff embeddings
        reducer = DiffFeatureReducer(n_components=32)
        train_diff_features = reducer.fit_transform(train_df["diff_text"])
        test_diff_features = reducer.transform(test_df["diff_text"])

        X_train_full = pd.concat([X_train_meta.reset_index(drop=True),
                                   train_diff_features.reset_index(drop=True)], axis=1)
        X_test_full = pd.concat([X_test_meta.reset_index(drop=True),
                                  test_diff_features.reset_index(drop=True)], axis=1)

        diff_result = train_and_eval(X_train_full, y_train.reset_index(drop=True),
                                      X_test_full, y_test.reset_index(drop=True), "metadata+diff")

        all_results[held_out] = {"metadata_only": meta_result, "metadata_plus_diff": diff_result}

    logger.info("\n" + "=" * 70)
    logger.info("FINAL COMPARISON — metadata-only vs metadata+diff, per held-out repo")
    logger.info("=" * 70)
    for repo, r in all_results.items():
        m, d = r["metadata_only"], r["metadata_plus_diff"]
        roc_delta = d["roc_auc"] - m["roc_auc"]
        verdict = "IMPROVED" if roc_delta > 0.02 else ("WORSE" if roc_delta < -0.02 else "NO REAL CHANGE")
        logger.info(
            "%-20s  metadata ROC=%.3f  ->  +diff ROC=%.3f  (%+.3f)  [%s]",
            repo, m["roc_auc"], d["roc_auc"], roc_delta, verdict,
        )

    return all_results


if __name__ == "__main__":
    run_comparison()