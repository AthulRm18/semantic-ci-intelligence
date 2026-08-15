# Semantic CI 

Cross-repository CI failure prediction using code-diff content + pipeline
metadata, with predictions converted into a fail-fast job scheduling policy.

## Status: Step 1 — data access validation

Run `scripts/validate_data_access.py` to confirm we can pull, per repo:
- workflow run history
- per-JOB pass/fail + runtime (not just overall pipeline status)
- the triggering commit's diff

Requires a GitHub personal access token (public_repo scope) — unauthenticated
requests are rate-limited to 60/hour, which isn't enough for real data
collection across multiple repos.

Setup:
```
cp .env.example .env   # then fill in your token
pip install -r requirements.txt
python scripts/validate_data_access.py
```

## Roadmap
1. [ ] Confirm data access (in progress)
2. [ ] Build metadata-only baseline (Model 1)
3. [ ] Add semantic diff encoder, run ablation (Models 2-4)
4. [ ] Cross-repo generalization test
5. [ ] Fail-fast scheduler + demo dashboard
