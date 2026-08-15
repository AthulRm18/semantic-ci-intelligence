"""
Step 1 validation: can we actually pull what we need from the GitHub Actions API?

For each test repo we need, per workflow run:
  - overall run outcome
  - PER-JOB pass/fail (not just overall status)
  - per-job runtime
  - the triggering commit's diff

This script hits real endpoints against a few real repos and reports what's
actually available, so we know before writing a line of model code whether
the data plan is viable.

Unauthenticated GitHub API = 60 requests/hour. Fine for this smoke test.
For real data collection later we'll want a token (5000/hour) — see the
note at the bottom.
"""

import os
import requests
import time
import json
from pathlib import Path

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
    print("Using authenticated requests (5000/hour limit)")
else:
    print("WARNING: no GITHUB_TOKEN found — unauthenticated (60/hour limit).")
    print("Copy .env.example to .env and add a token, or this will likely fail.")

# Small/medium/large mix, all public, all with active Actions CI
TEST_REPOS = [
    "tiangolo/fastapi",
    "psf/requests",
]

OUT_DIR = Path(__file__).parent.parent / "data" / "validation"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def get(url, **params):
    resp = requests.get(url, headers=HEADERS, params=params)
    remaining = resp.headers.get("X-RateLimit-Remaining")
    if resp.status_code != 200:
        print(f"  ! {resp.status_code} for {url} — {resp.text[:200]}")
        return None
    if remaining is not None and int(remaining) < 5:
        print(f"  (rate limit low: {remaining} requests left — sleeping 30s)")
        time.sleep(30)
    return resp.json()


def validate_repo(repo: str):
    print(f"\n=== {repo} ===")
    result = {"repo": repo}

    # 1. Can we list workflow runs?
    runs_data = get(f"{API}/repos/{repo}/actions/runs", per_page=5, status="completed")
    if not runs_data or "workflow_runs" not in runs_data:
        print("  FAILED: can't list workflow runs")
        result["runs_accessible"] = False
        return result

    runs = runs_data["workflow_runs"]
    print(f"  OK: pulled {len(runs)} recent completed runs")
    result["runs_accessible"] = True
    result["sample_run_conclusions"] = [r["conclusion"] for r in runs]

    # 2. For one run, can we get PER-JOB pass/fail + runtime?
    sample_run = runs[0]
    run_id = sample_run["id"]
    jobs_data = get(f"{API}/repos/{repo}/actions/runs/{run_id}/jobs")

    if not jobs_data or "jobs" not in jobs_data:
        print("  FAILED: can't get per-job data")
        result["job_level_accessible"] = False
    else:
        jobs = jobs_data["jobs"]
        print(f"  OK: run {run_id} has {len(jobs)} jobs")
        job_summaries = []
        for j in jobs:
            started = j.get("started_at")
            completed = j.get("completed_at")
            runtime_sec = None
            if started and completed:
                from datetime import datetime
                fmt = "%Y-%m-%dT%H:%M:%SZ"
                runtime_sec = (
                    datetime.strptime(completed, fmt) - datetime.strptime(started, fmt)
                ).total_seconds()
            job_summaries.append({
                "name": j["name"],
                "conclusion": j["conclusion"],
                "runtime_sec": runtime_sec,
            })
            print(f"    - {j['name']}: {j['conclusion']} ({runtime_sec}s)")
        result["job_level_accessible"] = True
        result["sample_jobs"] = job_summaries

    # 3. Can we get the triggering commit's diff?
    head_sha = sample_run["head_sha"]
    commit_data = get(f"{API}/repos/{repo}/commits/{head_sha}")

    if not commit_data or "files" not in commit_data:
        print("  FAILED: can't get commit diff (may be a merge commit with too many files, or push event with no single commit)")
        result["diff_accessible"] = False
    else:
        files = commit_data["files"]
        total_patch_chars = sum(len(f.get("patch", "")) for f in files)
        print(f"  OK: commit {head_sha[:8]} touches {len(files)} files, {total_patch_chars} chars of diff")
        result["diff_accessible"] = True
        result["sample_diff_files"] = len(files)
        result["sample_diff_chars"] = total_patch_chars

    return result


if __name__ == "__main__":
    all_results = []
    for repo in TEST_REPOS:
        r = validate_repo(repo)
        all_results.append(r)
        time.sleep(1)

    out_path = OUT_DIR / "access_check.json"
    out_path.write_text(json.dumps(all_results, indent=2))

    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    for r in all_results:
        runs_ok = r.get("runs_accessible")
        jobs_ok = r.get("job_level_accessible")
        diff_ok = r.get("diff_accessible")
        verdict = "VIABLE" if (runs_ok and jobs_ok and diff_ok) else "NEEDS ATTENTION"
        print(f"{r['repo']}: runs={runs_ok} jobs={jobs_ok} diff={diff_ok} -> {verdict}")

    print(f"\nFull results saved to {out_path}")
