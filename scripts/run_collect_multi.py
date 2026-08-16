"""
Multi-repo data collection.

Same collection logic as before, now looped across several repos and
tagging every row with its source repo — this is what makes the eventual
cross-repo train/test split possible (train on some repos, test on repos
never seen during training).

Run: python scripts/run_collect_multi.py
"""

import logging

import pandas as pd

from ci_intel.config import DATA_RAW_DIR
from ci_intel.data.collector import GitHubCIClient, runtime_seconds

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Real, active Python repos with genuine multi-job CI pipelines —
# diverse enough in size/domain to make cross-repo generalization meaningful
REPOS = [
    "tiangolo/fastapi",   # already collected — will be skipped if incremental
    "psf/black",
    "encode/httpx",
    "pallets/flask",
]

MAX_RUNS_PER_REPO = 500
MULTI_REPO_CSV_PATH = DATA_RAW_DIR / "multi_repo_ci_data.csv"


def collect_repo(client: GitHubCIClient, repo: str, max_runs: int) -> pd.DataFrame:
    runs = client.fetch_runs(repo, max_runs)
    logger.info("%s: %d runs pulled", repo, len(runs))

    rows = []
    for i, run in enumerate(runs):
        run_id = run["id"]
        head_sha = run["head_sha"]

        jobs = client.fetch_jobs(repo, run_id)
        if not jobs:
            continue

        diff_info = client.fetch_commit_diff_summary(repo, head_sha)

        for job in jobs:
            rows.append({
                "repo": repo,
                "run_id": run_id,
                "workflow_name": run.get("name"),
                "run_conclusion": run.get("conclusion"),
                "head_sha": head_sha,
                "run_created_at": run.get("created_at"),
                "event": run.get("event"),
                "job_id": job["id"],
                "job_name": job["name"],
                "job_conclusion": job["conclusion"],
                "job_runtime_sec": runtime_seconds(job.get("started_at"), job.get("completed_at")),
                **diff_info,
            })

        if (i + 1) % 25 == 0:
            logger.info("  %s: processed %d/%d runs -> %d job-rows so far",
                        repo, i + 1, len(runs), len(rows))

    return pd.DataFrame(rows)


def main():
    client = GitHubCIClient()

    all_dfs = []
    for repo in REPOS:
        try:
            df = collect_repo(client, repo, MAX_RUNS_PER_REPO)
            all_dfs.append(df)
        except Exception as e:
            logger.error("Failed to collect %s: %s — skipping, continuing with the rest", repo, e)
            continue

    if not all_dfs:
        logger.error("No repos collected successfully. Nothing saved.")
        return

    combined = pd.concat(all_dfs, ignore_index=True)
    combined.to_csv(MULTI_REPO_CSV_PATH, index=False)

    logger.info("\nDone. %d total job-rows across %d repos.", len(combined), combined["repo"].nunique())
    logger.info("Saved to %s", MULTI_REPO_CSV_PATH)
    logger.info("\nRows per repo:\n%s", combined["repo"].value_counts())
    logger.info("\nFailure rate per repo:\n%s",
                combined.groupby("repo")["job_conclusion"].apply(lambda s: (s == "failure").mean()))


if __name__ == "__main__":
    main()
