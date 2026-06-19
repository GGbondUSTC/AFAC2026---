# AI Review Brief

Use this brief when asking any capable AI model or human reviewer to inspect the project and propose score-improvement directions.

## Objective

Find practical A-board score improvements for AFAC2026 Challenge Group Problem 3 under sparse feedback. Avoid leaderboard probing, target leakage, hidden-label assumptions, or methods that depend on unreproducible external state.

## Current Best

- Official best version: `v32`
- Official A-board total: `0.6278`
- Classification: `0.7590`
- Recommendation: `0.4966`
- Rank at report time: not recorded
- Current root submission: `prediction.zip` from `v32`, the verified official-best package
- Latest submitted probes: `v33` and `v34`; both were below `v32` and were not retained
- Latest local candidate status: v34 tightened the v33 gate but still scored `0.6277`, below v32

Known leader reference:

- Team: `非稳态`
- Total: `0.63578`
- Classification: `0.76518`
- Recommendation: `0.50639`

Gap to leader:

- Total: `0.00798`
- Classification: `0.00618`
- Recommendation: `0.00979`

Recommendation remains the larger gap. `v32` shows that additional recall can transfer only when its use is tightly gated; future submissions need stronger segment evidence than generic reranking or micro-tuning.

## What To Read First

Read these files before proposing experiments:

```text
README.md
AGENTS.md
分数记录.md
上下文维护.md
versions/VERSION_INDEX.json
docs/DATA_FIRST_IMPROVEMENT_PLAN.md
docs/EXTERNAL_SOLUTION_IDEAS.md
our_solution/src/recommendation.py
our_solution/src/classification.py
our_solution/src/validation.py
our_solution/tools/v23_long_history_grid_eval.py
our_solution/tools/data_audit_rec_segments.py
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
- `v24` expanded the same tail rerank to `seq_len>=21` and reached recommendation `0.4962`.
- `v25` raised internal recommendation validation slightly but tied `v24` online; do not treat tiny deltas as enough.
- `v29` changed the v25 medium/long-history tail rerank to count-only and improved recommendation to `0.4964`.
- `v31` raised seed-42 candidate recall@80 from the v29 reference `0.903206` to `0.963009`, but a broad LightGBM top2-top10 rerank regressed online to recommendation `0.4950`.
- `v32` reused the v31 recall/ranker stack only for an exact-length `>80` insert gate: freeze top5, insert at rank6 or below, maximum two insertions. Its five-split delta versus v29 was mean `+0.000155`, min `0.000000`, and it became the current official best at recommendation `0.4966`, total `0.6278`.
- `v33` widened the `>80` LightGBM blend while freezing top3 and regressed to `0.4963`; `v34` reduced changed rows but reached only `0.4964`. Keep these as negative calibration evidence, not baselines.
- `v15` short-history neural tower looked strong offline but failed online; `seq_len=1/2/3` remains high-risk.
- `v19` count/Bayes short-history probe failed its guard and was not submitted.
- `v26` post-v25 grid found only about `+0.00012` versus `v24`, so no package was generated.
- `v27` local probe added repeat concentration gate plus last/suffix/user-group conditional priors on top of `v24`; default 3-split exact-weighted delta versus `v24` was `-0.000878`, so it was not submitted.
- Validation now reports finer exact bins `0/1/2/3/4-10/11-20/21-30/31-80/>80` and repeat-concentration summaries.

## Current Guardrail

Future candidates should compare against `v32` as the actionable baseline, while also reporting deltas versus `v29`, `v24`, and `v17/v21` for continuity.

Suggested submission threshold:

- Multi-split mean test-weighted or exact-weighted NDCG delta versus `v32`: at least `+0.0005` for a normal submission candidate.
- Multi-split minimum delta versus `v32`: non-negative and preferably at least `+0.0003`.
- A smaller delta requires an explicit, v32-like segment constraint, unchanged top1/top3 where applicable, and clear candidate-quality evidence; it is not enough to tune existing weights.
- No negative split for high-risk short-history changes.
- Keep `top1_changed == 0` for short-history users unless evidence is much stronger.
- Do not submit gains around `+0.0001` to `+0.0003` unless backed by a new data signal and strong per-segment diagnostics.

## Current Data-First Questions

1. Which `A推荐/test.csv` length/user-feature segments differ most from the internal validation splits built from `train.csv`?
2. Which extra v31-style candidates are correct in masked validation but are not safely usable by the current v32 insert gate?
3. Can we build a validation split that better matches the public test distribution without using hidden answers, especially in the `>80` segment?
4. Are repeated-history items, last-item repeats, or count/recency features differently distributed between train-derived validation rows and test rows?
5. Are zero-history users already saturated by the neural/user-feature ensemble, or is there a data-derived cold-start cluster that still has room?
6. Can classification use graph/data diagnostics to identify a very small set of high-confidence label corrections beyond `v21` without repeating the `v20` failure mode?

Current local audit entry point:

```powershell
python .\our_solution\tools\data_audit_rec_segments.py
```

The generated JSON is local-only under `our_solution/output/`; use it as a reproducible diagnostic artifact, not as a submission input.

## Constraints

- Do not upload or expose `.env` keys.
- Do not rely on hidden labels or reverse-engineering leaderboard feedback.
- Keep generated submission artifacts out of Git.
- Prefer reproducible changes based on tracked `A分类/` and `A推荐/` data.
- Explain expected transfer risk by segment before recommending another submission.
