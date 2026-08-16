"""
Ablation: is job_base_historical_fail_rate alone sufficient to get PR-AUC=1.0
on this test set? If yes, that confirms the "perfect score" is really just
"predict by job type" — a real, legitimate signal, but not proof that diff
content/metadata add anything. If a job-type-only model gets a WORSE score
than the full model, that's evidence the other features are contributing
real information.

Run: python scripts/run_ablation.py
"""

import logging

import lightgbm as lgb
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from ci_intel.config import CLEANED_CSV_PATH
from ci_intel.features.engineering import build_features, compute_job_fail_rates
from ci_intel.models.train import run_level_chronological_split

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def run_ablation():
    df = pd.read_csv(CLEANED_CSV_PATH, parse_dates=["run_created_at"])
    train_df, test_df = run_level_chronological_split(df)
    job_fail_rate_map = compute_job_fail_rates(train_df)

    X_train_full, y_train = build_features(train_df, job_fail_rate_map)
    X_test_full, y_test = build_features(test_df, job_fail_rate_map)

    logger.info("Test set: %d rows, %d failures", len(y_test), y_test.sum())

    configs = {
        "job_type_only": ["job_base_historical_fail_rate"],
        "metadata_only_no_job_rate": [c for c in X_train_full.columns if c != "job_base_historical_fail_rate"],
        "full_model": list(X_train_full.columns),
    }

    results = {}
    for name, cols in configs.items():
        X_train = X_train_full[cols]
        X_test = X_test_full[cols]

        neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
        model = lgb.LGBMClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.05,
            scale_pos_weight=neg / pos if pos > 0 else 1.0,
            random_state=42, verbose=-1,
        )
        model.fit(X_train, y_train, categorical_feature="auto")
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        pr_auc = average_precision_score(y_test, y_pred_proba)
        roc_auc = roc_auc_score(y_test, y_pred_proba) if y_test.nunique() > 1 else float("nan")
        results[name] = {"pr_auc": pr_auc, "roc_auc": roc_auc, "n_features": len(cols)}
        logger.info("%-30s PR-AUC=%.4f  ROC-AUC=%.4f  (%d features)",
                    name, pr_auc, roc_auc, len(cols))

    logger.info("\n=== INTERPRETATION ===")
    if abs(results["job_type_only"]["pr_auc"] - results["full_model"]["pr_auc"]) < 0.02:
        logger.info(
            "job_base_historical_fail_rate ALONE gets nearly the same score as the full "
            "model. This test set is currently being solved by job identity alone — "
            "the diff/metadata features aren't being meaningfully tested yet. "
            "This isn't a bug; it means you need a larger, harder dataset (more repos, "
            "more runs) before this baseline is a real benchmark."
        )
    else:
        logger.info(
            "The full model meaningfully outperforms job-type-only — other features "
            "are contributing real signal, not just job identity."
        )

    return results


if __name__ == "__main__":
    run_ablation()
