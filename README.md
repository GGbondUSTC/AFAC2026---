# AFAC2026 Sparse Feedback Experiment Challenge

This repository contains an iterative solution for AFAC2026 Challenge Group, Problem 3: sparse-feedback automated experiments.

The public GitHub version is intended for code review and strategy analysis. It excludes local competition data, API keys, generated submissions, model checkpoints, and large runtime outputs.

## Current Status

- Official A-board best: `v17`
- Total score: `0.6266`
- Classification score: `0.7586`
- Recommendation score: `0.4947`
- Rank at report time: `25`
- Current root submission on the local machine: `prediction.zip` from `v17` (ignored by Git)

Recent diagnosis: `v18_probe` tested stronger short-history rerank and zero-history 5-seed neural ensembles, but multi-split validation was not robust enough. It was not submitted and did not replace `v17`.

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
docs/GPT_PRO_REVIEW.md      Suggested prompt/context for GPT Pro review
```

Ignored local-only paths include `A分类/`, `A推荐/`, `prediction.zip`, `our_solution/output*/`, versioned submission CSV/zip files, `.env`, and baseline bundled data/output.

## Reproducing Locally

Place local A-board data in the original structure:

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
python -m py_compile .\our_solution\run.py .\our_solution\src\classification.py .\our_solution\src\recommendation.py
```

Run both tasks:

```powershell
python .\our_solution\run.py --budget 5 --seed 42 --version vN --note "description"
```

Run recommendation only while reusing the current best classification:

```powershell
python .\our_solution\run.py `
  --task 2 `
  --reuse_a1 .\versions\v8\submission\A1.csv `
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

The main open problem is recommendation improvement. Classification has been stable at `0.7586`; the internally better v11/v12 classification variant transferred worse online.

Recommended review starting points:

1. `docs/GPT_PRO_REVIEW.md`
2. `分数记录.md`
3. `versions/VERSION_INDEX.json`
4. `our_solution/src/recommendation.py`
5. `our_solution/src/classification.py`
