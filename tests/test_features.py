"""
Tests for feature engineering â€” iteration 2, now covering job_name parsing
(the fix for the memorization/leakage-adjacent bug) alongside the existing
leakage-prevention checks.
"""

import pandas as pd
import pytest

from ci_intel.features.engineering import build_features, compute_job_fail_rates, parse_job_name


def test_parse_simple_job_name():
    result = parse_job_name("lint")
    assert result["job_base_name"] == "lint"
    assert result["is_matrix_job"] == 0
    assert result["os"] == "none"


def test_parse_matrix_job_name():
    result = parse_job_name("test (ubuntu-latest, 3.13, highest, no-deprecation, coverage)")
    assert result["job_base_name"] == "test"
    assert result["is_matrix_job"] == 1
    assert result["os"] == "ubuntu"
    assert result["dependency_tier"] == "highest"
    assert result["has_coverage"] == 1
    assert result["has_deprecation_flag"] == 1  # "no-deprecation" still contains "deprecation"


def test_parse_experimental_runtime_flagged():
    result = parse_job_name("test (macos-latest, 3.14t, no-deprecation, highest, starlette-git)")
    assert result["is_experimental_runtime"] == 1
    assert result["os"] == "macos"


def test_two_different_matrix_strings_share_base_name():
    """
    This is the actual fix: two near-unique matrix strings that would have
    been separate, near-lookup categories before now collapse into the same
    job_base_name, so the model can learn from their combined occurrences
    instead of treating each as its own tiny, memorizable bucket.
    """
    a = parse_job_name("test (ubuntu-latest, 3.13, highest, no-deprecation, coverage)")
    b = parse_job_name("test (windows-latest, 3.14t, no-deprecation, highest, starlette-git)")
    assert a["job_base_name"] == b["job_base_name"] == "test"


@pytest.fixture
def sample_train_df():
    return pd.DataFrame({
        "job_name": [
            "test (ubuntu-latest, 3.13, highest, no-deprecation, coverage)",
            "test (macos-latest, 3.14t, no-deprecation, highest, starlette-git)",
            "lint", "lint",
        ],
        "event": ["push", "push", "pull_request", "push"],
        "workflow_name": ["ci", "ci", "ci", "ci"],
        "files_changed": [1, 2, 1, 3],
        "additions": [10, 20, 5, 15],
        "deletions": [2, 3, 1, 4],
        "patch_chars": [100, 200, 50, 150],
        "run_created_at": pd.to_datetime([
            "2026-01-01", "2026-01-02", "2026-01-03", "2026-01-04",
        ]),
        "target_failed": [1, 1, 0, 0],
    })


def test_job_base_fail_rates_pool_matrix_variants(sample_train_df):
    rates = compute_job_fail_rates(sample_train_df)
    # both "test" rows failed -> base name "test" should show 1.0, not two
    # separate near-lookup entries for each matrix string
    assert rates["test"] == 1.0
    assert rates["lint"] == 0.0


def test_build_features_returns_expected_shape(sample_train_df):
    rates = compute_job_fail_rates(sample_train_df)
    X, y = build_features(sample_train_df, rates)
    assert len(X) == len(sample_train_df)
    assert "job_base_historical_fail_rate" in X.columns
    assert "is_matrix_job" in X.columns
    assert "os" in X.columns
