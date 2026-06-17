# GPT Pro Review Brief

Use this brief when asking GPT Pro or another advanced model to review the project and propose score-improvement directions.

## Objective

Find practical A-board score improvements for AFAC2026 Challenge Group Problem 3 under sparse feedback. Avoid leaderboard probing, target leakage, or methods that rely on hidden test answers.

## Current Best

- Version: `v17`
- Official A-board total: `0.6266`
- Classification: `0.7586`
- Recommendation: `0.4947`
- Rank at report time: `25`
- Known leader reference: total `0.63524`, classification `0.76409`, recommendation `0.50639`

Gap to leader:

- Total: `0.00864`
- Classification: `0.00549`
- Recommendation: `0.01169`

Recommendation remains the larger gap, but small rerank changes have low transfer reliability.

## What To Read

Read these files first:

```text
README.md
AGENTS.md
分数记录.md
上下文维护.md
versions/VERSION_INDEX.json
our_solution/src/recommendation.py
our_solution/src/classification.py
进阶教程.md
赛题.md
```

Optional baseline source:

```text
Baseline源代码-赛题三：稀疏反馈baseline/framework/code/
Baseline源代码-赛题三：稀疏反馈baseline/framework/BASELINE.md
```

The GitHub repository intentionally excludes local data, generated CSV/zip submissions, API keys, checkpoints, and runtime outputs.

## Important Version Lessons

- `v7/v8` classification result is the best stable A1 source: official classification `0.7586`.
- `v11/v12` had slightly higher internal classification accuracy but worse official classification `0.7579`; do not trust tiny internal classification gains.
- `v14` recommendation zero-history 3-seed neural ensemble transferred well.
- `v15` short-history neural tower looked strong offline but failed online: recommendation dropped to `0.4923`.
- `v16` zero-history dropout/seed tweak gave only a small positive transfer.
- `v17` conservative short-history rerank gave a small positive transfer: recommendation `+0.0003` versus v16.
- `v18_probe` was rejected: stronger short-history and zero-history 5-seed variants were not robust over multiple splits.

## Current Guardrail

A future recommendation candidate should be compared against `v17`, not against the original baseline.

Suggested submission threshold:

- Multi-split mean test-weighted NDCG delta versus v17: at least `+0.0012`
- Multi-split minimum delta: at least `+0.0005`
- No negative split if the change touches high-risk short-history users
- For `seq_len=1/2/3`, keep `top1_changed == 0` unless evidence is much stronger
- Do not submit gains around `+0.0001` to `+0.0003`

## Questions For GPT Pro

1. Given `recommendation.py`, what new feature families or model structures are likely to improve zero-history or very-short-history recommendation beyond the current user-feature MLP and rule fusion?
2. Can the validation protocol be improved to better match official A-board short-history users without test-answer probing?
3. Is there a safe way to use item metadata or target prior smoothing that could improve zero-history top10 ranking?
4. Are there classification changes that can beat `0.7586` with a low number of changed rows, or should classification remain frozen?
5. Which concrete v19 experiment should be implemented first, and what exact guard metrics should decide whether to submit it?

## Constraints

- Do not upload or expose `.env` keys.
- Do not rely on hidden labels or reverse-engineering leaderboard feedback.
- Prefer changes that can be reproduced from `A分类/` and `A推荐/` local data only.
- Keep generated submission artifacts out of Git.
