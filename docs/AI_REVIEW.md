# AI Review Brief

Use this brief when asking any capable AI model or human reviewer to inspect the project and propose score-improvement directions.

## Objective

Find practical A-board score improvements for AFAC2026 Challenge Group Problem 3 under sparse feedback. Avoid leaderboard probing, target leakage, hidden-label assumptions, or methods that depend on unreproducible external state.

## Current Best

- Official best version: `v24`
- Official A-board total: `0.6276`
- Classification: `0.7590`
- Recommendation: `0.4962`
- Rank at report time: `27`
- Current root submission: `prediction.zip` rolled back to `v24`
- Latest submitted probe: `v25`, official tie with `v24` at `0.6276`

Known leader reference:

- Team: `非稳态`
- Total: `0.63578`
- Classification: `0.76518`
- Recommendation: `0.50639`

Gap to leader:

- Total: `0.00818`
- Classification: `0.00618`
- Recommendation: `0.01019`

Recommendation remains the larger gap. Recent tiny offline rerank gains around `+0.0001` did not visibly transfer online.

## What To Read First

Read these files before proposing experiments:

```text
README.md
AGENTS.md
分数记录.md
上下文维护.md
versions/VERSION_INDEX.json
docs/DATA_FIRST_IMPROVEMENT_PLAN.md
our_solution/src/recommendation.py
our_solution/src/classification.py
our_solution/src/validation.py
our_solution/tools/v23_long_history_grid_eval.py
```

Then inspect the tracked A-board data:

```text
A分类/A1.npz
A分类/sample_submission.csv
A推荐/train.csv
A推荐/test.csv
A推荐/user.csv
A推荐/item.csv
A推荐/metadata.json
```

Optional historical context:

```text
docs/V19_PLAN.md
Baseline源代码-赛题三：稀疏反馈baseline/framework/code/
Baseline源代码-赛题三：稀疏反馈baseline/framework/BASELINE.md
```

## Important Version Lessons

- `v7/v8/v17` classification was stable at official `0.7586`.
- `v21` pair-gated classification changed only 9 A1 rows and improved official classification to `0.7590`.
- `v22` long-history recommendation tail rerank improved official recommendation from `0.4947` to `0.4958`.
- `v24` expanded the same tail rerank to `seq_len>=21` and reached the current official best recommendation `0.4962`.
- `v25` raised internal recommendation validation slightly but tied `v24` online; do not treat tiny deltas as enough.
- `v15` short-history neural tower looked strong offline but failed online; `seq_len=1/2/3` remains high-risk.
- `v19` count/Bayes short-history probe failed its guard and was not submitted.
- `v26` post-v25 grid found only about `+0.00012` versus `v24`, so no package was generated.

## Current Guardrail

Future candidates should compare against `v24` as the actionable baseline, while also reporting deltas versus `v17/v21` for continuity.

Suggested submission threshold:

- Multi-split mean test-weighted or exact-weighted NDCG delta versus `v24`: at least `+0.0008`.
- Multi-split minimum delta versus `v24`: positive and preferably at least `+0.0003`.
- No negative split for high-risk short-history changes.
- Keep `top1_changed == 0` for short-history users unless evidence is much stronger.
- Do not submit gains around `+0.0001` to `+0.0003` unless backed by a new data signal and strong per-segment diagnostics.

## Current Data-First Questions

1. Which `A推荐/test.csv` length/user-feature segments differ most from the internal validation splits built from `train.csv`?
2. Are there item or user anonymous feature combinations where the current `v24` tail rerank is systematically underusing strong conditional target priors?
3. Can we build a validation split that better matches the public test distribution without using hidden answers?
4. Are repeated-history items, last-item repeats, or count/recency features differently distributed between train-derived validation rows and test rows?
5. Are zero-history users already saturated by the neural/user-feature ensemble, or is there a data-derived cold-start cluster that still has room?
6. Can classification use graph/data diagnostics to identify a very small set of high-confidence label corrections beyond `v21` without repeating the `v20` failure mode?

## Constraints

- Do not upload or expose `.env` keys.
- Do not rely on hidden labels or reverse-engineering leaderboard feedback.
- Keep generated submission artifacts out of Git.
- Prefer reproducible changes based on tracked `A分类/` and `A推荐/` data.
- Explain expected transfer risk by segment before recommending another submission.
