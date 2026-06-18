# AFAC2026 Sparse Feedback Experiment Challenge

This repository contains an iterative solution for AFAC2026 Challenge Group, Problem 3: sparse-feedback automated experiments.

The GitHub version is intended for code review, strategy analysis, and data-first investigation. It includes the A-board raw data under `A分类/` and `A推荐/`, while still excluding API keys, generated submissions, model checkpoints, and large runtime outputs.

## Current Status

- Official A-board best: `v24`
- Total score: `0.6276`
- Classification score: `0.7590`
- Recommendation score: `0.4962`
- Rank at report time: `27`
- Current root submission on the local machine: `prediction.zip` from `v24` (ignored by Git)

Recent diagnosis: `v25` improved internal validation slightly by expanding the medium/long-history rerank pool, but its official A-board score tied `v24` exactly at `0.7590 / 0.4962 / 0.6276`. The root submission was rolled back to `v24`.

Current direction: stop spending submissions on tiny rerank deltas around `+0.0001`; use the now-tracked raw data to audit split mismatch, length buckets, item/user feature interactions, and data-derived signals that could produce a larger recommendation gain. See `docs/DATA_FIRST_IMPROVEMENT_PLAN.md` and `docs/AI_REVIEW.md`.

## Repository Layout

```text
our_solution/
  run.py                    Main entry point
  src/
    classification.py       Sparse graph label propagation + linear fallback
    recommendation.py       Hybrid recommender + PyTorch zero/short-history towers
    common.py               Shared utilities
    qwen_client.py          Optional LLM helper, not used in main prediction

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

The main open problem is recommendation improvement. Classification is currently `0.7590`; larger remaining gap is recommendation (`0.4962` versus known leader reference `0.50639`).

Recommended review starting points:

1. `docs/AI_REVIEW.md`
2. `docs/DATA_FIRST_IMPROVEMENT_PLAN.md`
3. `分数记录.md`
4. `versions/VERSION_INDEX.json`
5. `our_solution/src/recommendation.py`
6. `our_solution/src/classification.py`
7. `A推荐/train.csv`, `A推荐/test.csv`, `A推荐/user.csv`, `A推荐/item.csv`
8. `A分类/A1.npz`
