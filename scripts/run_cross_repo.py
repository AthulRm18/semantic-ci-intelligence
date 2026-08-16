"""
Cross-repo generalization test.

This is the actual research question from the original design doc: can a
model trained on some repos predict failures on a repo it's NEVER seen?
That's a fundamentally harder and more meaningful test than a chronological
split within one repo — it's also what makes a "works on any new repo"
product claim honest rather than aspirational.

Run: python scripts/run_cross_repo.py
"""

import logging

import lightgbm as lgb
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from ci_intel.config import DATA_RAW_DIR, VALID_JOB_CONCLUSIONS
from ci_intel.features.engineering import build_features, compute_job_fail_rates

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


def cross_repo_split(df: pd.DataFrame, held_out_repo: str):
    """
    Everything except held_out_repo trains the model; held_out_repo is the
    test set. The model has literally never seen this repo's job names,
    diff patterns, or failure rates during training.
    """
    train_df = df[df["repo"] != held_out_repo].copy()
    test_df = df[df["repo"] == held_out_repo].copy()
    return train_df, test_df


def run_cross_repo_eval():
    df = pd.read_csv(MULTI_REPO_CSV_PATH, parse_dates=["run_created_at"])
    df = clean_multi_repo(df)

    repos = df["repo"].unique()
    logger.info("Repos in dataset: %s", list(repos))
    logger.info("Rows per repo:\n%s", df["repo"].value_counts())
    logger.info("Failure rate per repo:\n%s", df.groupby("repo")["target_failed"].mean())

    results = {}
    for held_out in repos:
        train_df, test_df = cross_repo_split(df, held_out)

        if test_df["target_failed"].sum() < 3:
            logger.warning("Skipping %s as held-out repo — too few failures (%d) to evaluate meaningfully",
                            held_out, test_df["target_failed"].sum())
            continue

        job_fail_rate_map = compute_job_fail_rates(train_df)
        X_train, y_train = build_features(train_df, job_fail_rate_map)
        X_test, y_test = build_features(test_df, job_fail_rate_map)

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
        baseline = y_test.mean()

        results[held_out] = {
            "pr_auc": pr_auc, "roc_auc": roc_auc, "random_baseline": baseline,
            "test_failures": int(y_test.sum()), "test_rows": len(y_test),
        }
        logger.info(
            "Held out: %-20s PR-AUC=%.3f (random=%.3f)  ROC-AUC=%.3f  (%d failures / %d rows)",
            held_out, pr_auc, baseline, roc_auc, y_test.sum(), len(y_test),
        )

    logger.info("\n=== SUMMARY: does this model generalize to UNSEEN repos? ===")
    for repo, r in results.items():
        lift = r["pr_auc"] / r["random_baseline"] if r["random_baseline"] > 0 else float("nan")
        logger.info("%-20s PR-AUC %.3f vs random %.3f  (%.1fx lift)",
                    repo, r["pr_auc"], r["random_baseline"], lift)

    return results


if __name__ == "__main__":
    run_cross_repo_eval()
