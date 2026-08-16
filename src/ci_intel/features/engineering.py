"""
Feature engineering for Model 1 (metadata-only baseline).

FIX (iteration 2): job_name was previously used as a raw categorical string,
e.g. "test (macos-latest, 3.14t, no-deprecation, highest, starlette-git)".
That let the model memorize specific matrix combinations rather than learn
anything transferable â€” a near-unique string with a handful of occurrences
and a skewed outcome is a lookup table, not a pattern. It's also the reason
this couldn't generalize to a new repo: a new repo's job names are
completely different strings the model has never seen.

Fix: parse job_name into repo-agnostic components (OS, language/runtime
version, dependency tier, whether it's a coverage/benchmark run, whether
it's a matrix job at all) so the model learns "matrix jobs on Windows with
lowest-pinned deps tend to be riskier" instead of memorizing one exact
string. This is what makes cross-repo transfer even plausible.
"""

import re

import pandas as pd

CATEGORICAL_COLS = [
    "job_base_name", "os", "dependency_tier", "event", "workflow_name",
]
BOOLEAN_COLS = [
    "is_matrix_job", "has_coverage", "has_deprecation_flag", "is_experimental_runtime",
]
NUMERIC_COLS = ["files_changed", "additions", "deletions", "patch_chars"]

_OS_PATTERN = re.compile(r"(ubuntu|macos|windows)", re.IGNORECASE)
_VERSION_PATTERN = re.compile(r"\b(\d+\.\d+t?)\b")


def parse_job_name(job_name: str) -> dict:
    """
    Decompose a job name (possibly a matrix-expanded string) into
    repo-agnostic components. Handles both simple names ('lint', 'deploy-docs')
    and matrix names ('test (ubuntu-latest, 3.13, highest, no-deprecation, coverage)').
    """
    job_name = job_name or ""
    is_matrix = "(" in job_name

    base_name = job_name.split("(")[0].strip() if is_matrix else job_name

    os_match = _OS_PATTERN.search(job_name)
    os_name = os_match.group(1).lower() if os_match else "none"

    version_match = _VERSION_PATTERN.search(job_name)
    is_experimental_runtime = bool(version_match and version_match.group(1).endswith("t"))

    lower = job_name.lower()
    dependency_tier = (
        "lowest" if "lowest" in lower else
        "highest" if "highest" in lower else
        "none"
    )
    has_coverage = "coverage" in lower
    has_deprecation_flag = "deprecation" in lower

    return {
        "job_base_name": base_name,
        "is_matrix_job": int(is_matrix),
        "os": os_name,
        "dependency_tier": dependency_tier,
        "has_coverage": int(has_coverage),
        "has_deprecation_flag": int(has_deprecation_flag),
        "is_experimental_runtime": int(is_experimental_runtime),
    }


def compute_job_fail_rates(train_df: pd.DataFrame) -> dict:
    """
    Historical failure rate per BASE job name (e.g. 'test', 'lint'), not per
    exact matrix string â€” this is what makes the rate meaningful (enough
    occurrences to be a real rate, not a near-lookup) and repo-transferable
    (a 'test' job base name exists across most repos, a specific 40-char
    matrix string does not).

    Computed ONLY from training data â€” see the leakage warning in train.py.
    """
    parsed = train_df["job_name"].apply(parse_job_name).apply(pd.Series)
    return parsed.assign(target_failed=train_df["target_failed"].values) \
                  .groupby("job_base_name")["target_failed"].mean().to_dict()


def build_features(df: pd.DataFrame, job_fail_rate_map: dict) -> tuple[pd.DataFrame, pd.Series]:
    df = df.copy()

    parsed = df["job_name"].apply(parse_job_name).apply(pd.Series)
    df = pd.concat([df.reset_index(drop=True), parsed.reset_index(drop=True)], axis=1)

    # time-based features
    df["hour_of_day"] = df["run_created_at"].dt.hour
    df["day_of_week"] = df["run_created_at"].dt.dayofweek
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)

    df["net_change"] = df["additions"].fillna(0) - df["deletions"].fillna(0)
    df["total_change"] = df["additions"].fillna(0) + df["deletions"].fillna(0)

    global_fallback = sum(job_fail_rate_map.values()) / len(job_fail_rate_map) if job_fail_rate_map else 0.0
    df["job_base_historical_fail_rate"] = df["job_base_name"].map(job_fail_rate_map).fillna(global_fallback)

    for col in CATEGORICAL_COLS:
        df[col] = df[col].astype("category")

    feature_cols = (
        CATEGORICAL_COLS + BOOLEAN_COLS + NUMERIC_COLS +
        ["hour_of_day", "day_of_week", "is_weekend",
         "net_change", "total_change", "job_base_historical_fail_rate"]
    )

    X = df[feature_cols]
    y = df["target_failed"]
    return X, y
