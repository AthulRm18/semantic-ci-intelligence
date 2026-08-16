"""
Model 1: metadata-only baseline (iteration 2).

Two fixes from iteration 1's suspicious PR-AUC=1.000 result:

1. job_name is now parsed into repo-agnostic components (see
   features/engineering.py) instead of used as a raw memorizable string.

2. The split is now done at the RUN level, not the row level. Multiple
   jobs share one run_id (same commit, same diff, same timestamp) â€” sorting
   rows by timestamp and cutting at a row index could put some jobs from
   run X in train and other jobs from the SAME run X in test, which leaks
   commit/diff information across the boundary. Now we sort unique run_ids
   chronologically, split THOSE, then assign every job to whichever side
   its run landed on. No run straddles the boundary.
"""

import logging
from datetime import datetime

import joblib
import lightgbm as lgb
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    roc_auc_score,
)

from ci_intel.config import CLEANED_CSV_PATH, MODELS_DIR
from ci_intel.features.engineering import build_features, compute_job_fail_rates

logger = logging.getLogger(__name__)


def run_level_chronological_split(df: pd.DataFrame, test_frac: float = 0.2):
    run_times = df.groupby("run_id")["run_created_at"].first().sort_values()
    split_idx = int(len(run_times) * (1 - test_frac))
    train_run_ids = set(run_times.index[:split_idx])
    test_run_ids = set(run_times.index[split_idx:])

    assert train_run_ids.isdisjoint(test_run_ids), "run leakage across split â€” this should be impossible"

    train_df = df[df["run_id"].isin(train_run_ids)].copy()
    test_df = df[df["run_id"].isin(test_run_ids)].copy()

    logger.info(
        "Run-level split: %d train runs (%d rows), %d test runs (%d rows), 0 overlapping runs",
        len(train_run_ids), len(train_df), len(test_run_ids), len(test_df),
    )
    return train_df, test_df


def train_baseline(df: pd.DataFrame | None = None):
    if df is None:
        df = pd.read_csv(CLEANED_CSV_PATH, parse_dates=["run_created_at"])

    train_df, test_df = run_level_chronological_split(df)

    job_fail_rate_map = compute_job_fail_rates(train_df)
    logger.info("Base job failure rates learned from training data: %s",
                {k: round(v, 3) for k, v in job_fail_rate_map.items()})

    X_train, y_train = build_features(train_df, job_fail_rate_map)
    X_test, y_test = build_features(test_df, job_fail_rate_map)

    neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
    scale_pos_weight = neg / pos if pos > 0 else 1.0
    logger.info("Class weighting: scale_pos_weight=%.2f (%d success / %d failure in train)",
                scale_pos_weight, neg, pos)

    model = lgb.LGBMClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        verbose=-1,
    )
    model.fit(X_train, y_train, categorical_feature="auto")

    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba >= 0.5).astype(int)

    pr_auc = average_precision_score(y_test, y_pred_proba)
    roc_auc = roc_auc_score(y_test, y_pred_proba) if y_test.nunique() > 1 else float("nan")

    logger.info("Test PR-AUC: %.3f", pr_auc)
    logger.info("Test ROC-AUC: %.3f", roc_auc)
    logger.info("\n%s", classification_report(y_test, y_pred, target_names=["success", "failure"]))

    baseline_pr_auc = y_test.mean()
    logger.info(
        "Baseline PR-AUC (random guessing at this failure rate): %.3f â€” "
        "model needs to clear this to be worth anything.",
        baseline_pr_auc,
    )

    # feature importance â€” worth eyeballing to sanity-check nothing's memorizing again
    importances = pd.Series(model.feature_importances_, index=X_train.columns).sort_values(ascending=False)
    logger.info("Top 10 feature importances:\n%s", importances.head(10))

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = MODELS_DIR / f"model1_baseline_{timestamp}.joblib"
    joblib.dump({"model": model, "job_fail_rate_map": job_fail_rate_map}, model_path)
    logger.info("Saved model to %s", model_path)

    return model, {"pr_auc": pr_auc, "roc_auc": roc_auc, "baseline_pr_auc": baseline_pr_auc}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    train_baseline()
