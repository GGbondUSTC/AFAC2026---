# AFAC2026 Sparse Feedback Experiment Challenge

This repository contains an iterative solution for AFAC2026 Challenge Group, Problem 3: sparse-feedback automated experiments.

The GitHub version is intended for code review, strategy analysis, and data-first investigation. It includes the A-board raw data under `A分类/` and `A推荐/`, while still excluding API keys, generated submissions, model checkpoints, and large runtime outputs.

## Current Status

- Official A-board best: `v32`
- Total score: `0.6278`
- Classification score: `0.7590`
- Recommendation score: `0.4966`
- Rank at report time: not recorded
- Current root submission on the local machine: `prediction.zip` from `v32` (ignored by Git; restored after the v33/v34 probes)

Recent diagnosis: `v31` expanded the candidate pool with co-visitation and other sources, but its unconstrained LightGBM rerank regressed online to `0.4950` recommendation. `v32` retained that recall pool but only inserts high-confidence candidates for histories longer than 80: it freezes the top 5 and inserts at most two items from rank 6. This transferred online, lifting recommendation from `0.4964` to `0.4966` and total from `0.6277` to `0.6278`.

The follow-up v33/v34 semi-free rerank probes did not exceed v32, despite preserving their top 3 recommendations. Current direction: treat `v32` as the baseline; improve candidate quality and gate calibration using segment evidence, rather than widening tail reranks or tuning count-only weights. See `docs/DATA_FIRST_IMPROVEMENT_PLAN.md`, `docs/AI_REVIEW.md`, and `docs/EXTERNAL_SOLUTION_IDEAS.md`.

## Repository Layout

```text
our_solution/
  run.py                    Main entry point
  src/
    classification.py       Sparse graph label propagation + linear fallback
    recommendation.py       Hybrid recommender + PyTorch zero/short-history towers
    common.py               Shared utilities
    qwen_client.py          Optional Qwen/DashScope API client
    qwen_agent.py           Optional Qwen-assisted config proposal layer
  tools/
    data_audit_rec_segments.py
                              Local segment-support audit for recommendation data
    v31_recall_ranker_eval.py
    v32_ranker_gate_eval.py   Candidate-recall and conservative-gate evaluations

versions/
  VERSION_INDEX.json        Structured version ledger
  v*/                       Metric summaries kept; submissions and code snapshots ignored

分数记录.md                 Score ledger and official result history
上下文维护.md               Working context and version notes
赛题.md                     Competition/task description
进阶教程.md                 Advanced tutorial notes
AGENTS.md                   Contributor guide
docs/AI_REVIEW.md           AI-agnostic review brief for external model review
docs/DATA_FIRST_IMPROVEMENT_PLAN.md
                            Current data-first improvement plan
docs/V19_PLAN.md            Historical v19 plan and failed-probe context
```

Tracked data paths include `A分类/A1.npz`, `A分类/sample_submission.csv`, and all A-board recommendation files under `A推荐/`.

Ignored local-only paths include `prediction.zip`, `LATEST_VERSION.txt`, `our_solution/output*/`, versioned submission CSV/zip files, `.env`, checkpoints, logs, caches, and baseline bundled data/output.

## Reproducing Locally

The A-board data is tracked in the repository. The expected structure is:

```text
A分类/A1.npz
A分类/sample_submission.csv
A推荐/train.csv
A推荐/test.csv
A推荐/user.csv
A推荐/item.csv
```

Run syntax checks:

```powershell
python -m py_compile .\our_solution\run.py .\our_solution\src\classification.py .\our_solution\src\recommendation.py .\our_solution\src\validation.py
```

Run the current recommendation segment audit:

```powershell
python .\our_solution\tools\data_audit_rec_segments.py
```

Run both tasks:

```powershell
python .\our_solution\run.py --budget 5 --seed 42 --version vN --note "description"
```

Run recommendation only while reusing the current best classification:

```powershell
python .\our_solution\run.py `
  --task 2 `
  --reuse_a1 .\versions\v24\submission\A1.csv `
  --budget 15 `
  --seed 42 `
  --version vN `
  --note "recommendation experiment"
```

Use the local PyTorch environment for neural experiments:

```powershell
& 'E:\desktop\Artificial-Intelligence\Pytorch\.venv\Scripts\python.exe' .\our_solution\run.py ...
```

## Submission Rules

The local root `prediction.zip` is the latest candidate and must contain only:

```text
A1.csv
A2.csv
```

For GitHub, generated submission files are intentionally ignored. Use `versions/VERSION_INDEX.json`, `分数记录.md`, and `上下文维护.md` to understand which version was best and why.

## Review Priorities

The main open problem is recommendation improvement. Classification is currently `0.7590`; the larger remaining gap is recommendation (`0.4966` versus known leader reference `0.50639`).

Recommended review starting points:

1. `docs/AI_REVIEW.md`
2. `docs/DATA_FIRST_IMPROVEMENT_PLAN.md`
3. `分数记录.md`
4. `versions/VERSION_INDEX.json`
5. `our_solution/src/recommendation.py`
6. `our_solution/src/classification.py`
7. `A推荐/train.csv`, `A推荐/test.csv`, `A推荐/user.csv`, `A推荐/item.csv`
8. `A分类/A1.npz`
