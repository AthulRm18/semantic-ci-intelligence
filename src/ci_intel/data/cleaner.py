"""
Cleaning: drop rows that aren't real pass/fail signal, encode the target.

Honest note: 'skipped' jobs are dropped because they usually reflect
pipeline gating logic (e.g. a conditional job that didn't need to run),
not a judgment about code quality â€” keeping them would teach the model
a false pattern. 'cancelled' is dropped for the same reason (external
interruption, not a code-quality signal).
"""

import logging

import pandas as pd

from ci_intel.config import CLEANED_CSV_PATH, RAW_CSV_PATH, VALID_JOB_CONCLUSIONS

logger = logging.getLogger(__name__)


def clean(df: pd.DataFrame | None = None) -> pd.DataFrame:
    if df is None:
        df = pd.read_csv(RAW_CSV_PATH)

    before = len(df)
    df = df[df["job_conclusion"].isin(VALID_JOB_CONCLUSIONS)].copy()
    dropped = before - len(df)
    logger.info("Dropped %d rows (skipped/cancelled/other) out of %d", dropped, before)

    df["target_failed"] = (df["job_conclusion"] == "failure").astype(int)

    # parse timestamp into usable time features up front â€” feature engineering
    # module will build on these
    df["run_created_at"] = pd.to_datetime(df["run_created_at"])

    fail_rate = df["target_failed"].mean()
    logger.info("Class balance after cleaning: %.1f%% failure rate (%d failures / %d total)",
                fail_rate * 100, df["target_failed"].sum(), len(df))

    df.to_csv(CLEANED_CSV_PATH, index=False)
    logger.info("Saved cleaned data to %s", CLEANED_CSV_PATH)
    return df
