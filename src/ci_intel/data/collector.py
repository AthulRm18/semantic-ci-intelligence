"""
GitHub Actions data collection, as a reusable module.

Same logic you already validated works — moved here so it can be imported
and tested, rather than only runnable as a top-level script.
"""

import json
import logging
import time
from datetime import datetime

import pandas as pd
import requests

from ci_intel.config import (
    GITHUB_API,
    GITHUB_TOKEN,
    MAX_RUNS_PER_COLLECTION,
    PER_PAGE,
    RAW_CSV_PATH,
    TARGET_REPO,
)

logger = logging.getLogger(__name__)

FMT = "%Y-%m-%dT%H:%M:%SZ"


class GitHubCIClient:
    """Thin wrapper around the GitHub Actions API calls we need."""

    def __init__(self, token: str | None = None):
        token = token or GITHUB_TOKEN
        if not token:
            raise ValueError("No GitHub token provided. Set GITHUB_TOKEN in .env.")
        self.headers = {
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
        }
        self._commit_cache: dict[str, dict] = {}

    def _get(self, url: str, **params) -> dict | None:
        for attempt in range(3):
            resp = requests.get(url, headers=self.headers, params=params)
            remaining = resp.headers.get("X-RateLimit-Remaining")
            if resp.status_code == 200:
                if remaining is not None and int(remaining) < 20:
                    reset = int(resp.headers.get("X-RateLimit-Reset", time.time() + 60))
                    sleep_for = max(reset - time.time(), 1)
                    logger.info("Rate limit low (%s left), sleeping %.0fs", remaining, sleep_for)
                    time.sleep(sleep_for)
                return resp.json()
            elif resp.status_code == 403 and "rate limit" in resp.text.lower():
                logger.warning("Rate limited, sleeping 60s and retrying...")
                time.sleep(60)
                continue
            else:
                logger.error("%s for %s: %s", resp.status_code, url, resp.text[:150])
                return None
        return None

    def fetch_runs(self, repo: str, max_runs: int, since_run_id: int | None = None) -> list[dict]:
        """
        Fetch completed workflow runs, newest first.
        If since_run_id is given, stop once we reach a run we've already
        collected — this is what makes re-collection incremental instead
        of always re-fetching everything from scratch.
        """
        runs = []
        page = 1
        logger.info("Fetching workflow runs for %s...", repo)
        while len(runs) < max_runs:
            data = self._get(
                f"{GITHUB_API}/repos/{repo}/actions/runs",
                per_page=PER_PAGE,
                page=page,
                status="completed",
            )
            if not data or not data.get("workflow_runs"):
                break

            page_runs = data["workflow_runs"]
            if since_run_id is not None:
                new_runs = [r for r in page_runs if r["id"] > since_run_id]
                runs.extend(new_runs)
                if len(new_runs) < len(page_runs):
                    # hit runs we've already seen — stop paginating
                    break
            else:
                runs.extend(page_runs)

            logger.info("  page %d: %d runs (total so far: %d)", page, len(page_runs), len(runs))
            if len(page_runs) < PER_PAGE:
                break
            page += 1

        return runs[:max_runs]

    def fetch_jobs(self, repo: str, run_id: int) -> list[dict]:
        data = self._get(f"{GITHUB_API}/repos/{repo}/actions/runs/{run_id}/jobs")
        return data.get("jobs", []) if data else []

    def fetch_commit_diff_summary(self, repo: str, sha: str) -> dict:
        if sha in self._commit_cache:
            return self._commit_cache[sha]

        data = self._get(f"{GITHUB_API}/repos/{repo}/commits/{sha}")
        if not data or "files" not in data:
            result = {
                "files_changed": None, "additions": None, "deletions": None,
                "patch_chars": None, "changed_filenames": None, "diff_text": None,
            }
        else:
            files = data["files"]
            # concatenate actual patch text (filename + patch) for embedding later.
            # Capped at 4000 chars total to keep the CSV manageable and because
            # embedding models truncate long input anyway — the first part of a
            # diff is usually the most informative for "what kind of change is this".
            patch_parts = []
            running_len = 0
            for f in files:
                snippet = f"# {f['filename']}\n{f.get('patch', '')}\n"
                if running_len + len(snippet) > 4000:
                    break
                patch_parts.append(snippet)
                running_len += len(snippet)
            diff_text = "".join(patch_parts)

            result = {
                "files_changed": len(files),
                "additions": data.get("stats", {}).get("additions"),
                "deletions": data.get("stats", {}).get("deletions"),
                "patch_chars": sum(len(f.get("patch", "")) for f in files),
                "changed_filenames": json.dumps([f["filename"] for f in files]),
                "diff_text": diff_text,
            }
        self._commit_cache[sha] = result
        return result


def runtime_seconds(started: str | None, completed: str | None) -> float | None:
    if not started or not completed:
        return None
    return (datetime.strptime(completed, FMT) - datetime.strptime(started, FMT)).total_seconds()


def collect(repo: str = TARGET_REPO, max_runs: int = MAX_RUNS_PER_COLLECTION,
            incremental: bool = True) -> pd.DataFrame:
    """
    Collect CI data for a repo. If incremental=True and a previous CSV
    exists, only fetches runs newer than what's already saved and appends —
    otherwise does a full fresh pull.
    """
    client = GitHubCIClient()

    since_run_id = None
    existing_df = None
    if incremental and RAW_CSV_PATH.exists():
        existing_df = pd.read_csv(RAW_CSV_PATH)
        if not existing_df.empty:
            since_run_id = int(existing_df["run_id"].max())
            logger.info("Incremental mode: only fetching runs newer than %s", since_run_id)

    runs = client.fetch_runs(repo, max_runs, since_run_id=since_run_id)
    logger.info("Total new runs pulled: %d", len(runs))

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
            logger.info("  processed %d/%d runs -> %d job-rows so far", i + 1, len(runs), len(rows))

    new_df = pd.DataFrame(rows)

    if existing_df is not None and not new_df.empty:
        combined = pd.concat([existing_df, new_df], ignore_index=True)
        combined = combined.drop_duplicates(subset=["job_id"])
    elif existing_df is not None:
        combined = existing_df
    else:
        combined = new_df

    combined.to_csv(RAW_CSV_PATH, index=False)
    logger.info("Saved %d total job-rows (%d new) to %s", len(combined), len(new_df), RAW_CSV_PATH)
    return combined