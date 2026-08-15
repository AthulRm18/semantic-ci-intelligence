"""
Step 2: real data collection for tiangolo/fastapi.

Pulls workflow run history, per-job outcomes + runtimes, and the triggering
commit's diff for each run, and assembles it into one flat dataframe ready
for feature engineering (Model 1: metadata-only baseline).

One row per JOB (not per run) since job-level prediction is the actual goal -
a single run with 5 jobs becomes 5 rows, each tagged with the same diff/commit
metadata but its own job-level outcome and runtime.
"""

import os
import time
import json
from pathlib import Path
from datetime import datetime

import requests
import pandas as pd

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

API = "https://api.github.com"
HEADERS = {"Accept": "application/vnd.github+json"}

_token = os.environ.get("GITHUB_TOKEN")
if _token:
    HEADERS["Authorization"] = f"Bearer {_token}"
else:
    raise SystemExit("No GITHUB_TOKEN found. Set it in .env before running this.")

REPO = "tiangolo/fastapi"
MAX_RUNS = 500
PER_PAGE = 100
OUT_DIR = Path(__file__).parent.parent / "data" / "raw"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FMT = "%Y-%m-%dT%H:%M:%SZ"


def get(url, **params):
    for attempt in range(3):
        resp = requests.get(url, headers=HEADERS, params=params)
        remaining = resp.headers.get("X-RateLimit-Remaining")
        if resp.status_code == 200:
            if remaining is not None and int(remaining) < 20:
                reset = int(resp.headers.get("X-RateLimit-Reset", time.time() + 60))
                sleep_for = max(reset - time.time(), 1)
                print(f"  (rate limit low: {remaining} left, sleeping {sleep_for:.0f}s)")
                time.sleep(sleep_for)
            return resp.json()
        elif resp.status_code == 403 and "rate limit" in resp.text.lower():
            print("  rate limited, sleeping 60s and retrying...")
            time.sleep(60)
            continue
        else:
            print(f"  ! {resp.status_code} for {url}: {resp.text[:150]}")
            return None
    return None


def runtime_seconds(started, completed):
    if not started or not completed:
        return None
    return (datetime.strptime(completed, FMT) - datetime.strptime(started, FMT)).total_seconds()


def fetch_all_runs(repo, max_runs):
    runs = []
    page = 1
    print(f"Fetching workflow runs for {repo}...")
    while len(runs) < max_runs:
        data = get(
            f"{API}/repos/{repo}/actions/runs",
            per_page=PER_PAGE,
            page=page,
            status="completed",
        )
        if not data or not data.get("workflow_runs"):
            break
        runs.extend(data["workflow_runs"])
        print(f"  page {page}: {len(data['workflow_runs'])} runs (total so far: {len(runs)})")
        if len(data["workflow_runs"]) < PER_PAGE:
            break
        page += 1
    return runs[:max_runs]


def fetch_jobs_for_run(repo, run_id):
    data = get(f"{API}/repos/{repo}/actions/runs/{run_id}/jobs")
    if not data:
        return []
    return data.get("jobs", [])


_commit_cache = {}


def fetch_commit_diff_summary(repo, sha):
    if sha in _commit_cache:
        return _commit_cache[sha]
    data = get(f"{API}/repos/{repo}/commits/{sha}")
    if not data or "files" not in data:
        result = {"files_changed": None, "additions": None, "deletions": None,
                   "patch_chars": None, "changed_filenames": None}
    else:
        files = data["files"]
        result = {
            "files_changed": len(files),
            "additions": data.get("stats", {}).get("additions"),
            "deletions": data.get("stats", {}).get("deletions"),
            "patch_chars": sum(len(f.get("patch", "")) for f in files),
            "changed_filenames": json.dumps([f["filename"] for f in files]),
        }
    _commit_cache[sha] = result
    return result


def main():
    runs = fetch_all_runs(REPO, MAX_RUNS)
    print(f"\nTotal runs pulled: {len(runs)}")

    rows = []
    for i, run in enumerate(runs):
        run_id = run["id"]
        head_sha = run["head_sha"]

        jobs = fetch_jobs_for_run(REPO, run_id)
        if not jobs:
            continue

        diff_info = fetch_commit_diff_summary(REPO, head_sha)

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
            print(f"  processed {i + 1}/{len(runs)} runs -> {len(rows)} job-rows so far")

    df = pd.DataFrame(rows)
    out_path = OUT_DIR / "fastapi_ci_data.csv"
    df.to_csv(out_path, index=False)

    print(f"\nDone. {len(df)} job-level rows from {df['run_id'].nunique()} runs.")
    print(f"Saved to {out_path}")
    print("\nJob outcome distribution:")
    print(df["job_conclusion"].value_counts())
    print("\nJob name distribution (top 10):")
    print(df["job_name"].value_counts().head(10))


if __name__ == "__main__":
    main()
