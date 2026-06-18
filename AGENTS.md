# Repository Guidelines

## Project Structure & Module Organization

This repository contains the AFAC2026 challenge solution for task 3.

- `our_solution/`: main implementation. `run.py` is the entry point; reusable code lives in `our_solution/src/`.
- `A分类/` and `A推荐/`: tracked A-board data for classification and recommendation. Keep `.env`, generated submissions, checkpoints, and runtime outputs out of Git.
- `versions/vN/`: archived submissions, code snapshots, metrics, and summaries for each version.
- `prediction.zip`: latest root-level submission package. It must contain only `A1.csv` and `A2.csv`.
- `上下文维护.md`, `分数记录.md`, and `versions/VERSION_INDEX.json`: experiment log, score ledger, and structured version index.
- `Baseline源代码-赛题三：稀疏反馈baseline/`: reference baseline; do not edit unless explicitly working on baseline comparisons.

## Build, Test, and Development Commands

Run from the repository root.

```powershell
python .\our_solution\run.py --budget 5 --seed 42 --version vN --note "description"
```

Runs both tasks and archives a version.

```powershell
python .\our_solution\run.py --task 1 --reuse_a2 .\versions\v24\submission\A2.csv --version vN
python .\our_solution\run.py --task 2 --reuse_a1 .\versions\v24\submission\A1.csv --version vN
```

Runs only one task while reusing the other task’s CSV.

```powershell
python -m py_compile .\our_solution\run.py .\our_solution\src\classification.py .\our_solution\src\recommendation.py .\our_solution\src\validation.py
```

Performs a fast syntax check.

Use the PyTorch venv for neural experiments:

```powershell
& 'E:\desktop\Artificial-Intelligence\Pytorch\.venv\Scripts\python.exe' your_script.py
```

## Coding Style & Naming Conventions

Use Python 3, 4-space indentation, type hints where they improve clarity, and concise function names in `snake_case`. Versioned experiment configs should use explicit names such as `v11_label_prop_prior_beta_m025_zero_ridge`. Keep changes scoped; avoid unrelated refactors or metadata churn.

## Testing Guidelines

There is no formal unit-test suite. Validate every submission candidate with:

- syntax check via `py_compile`;
- `prediction.zip` contains only `A1.csv` and `A2.csv`;
- `A1.csv` has `2751` rows;
- `A2.csv` has `10000` rows, 10 unique valid items per row;
- compare A1/A2 diffs against the previous best version.

Only submit versions with meaningful validation gains; tiny recommendation gains below about `+0.001` have not transferred reliably online.

## Commit & Pull Request Guidelines

This workspace is a Git repository with GitHub remote `origin`. Use short imperative commit messages, e.g. `Document v24 results and data audit plan`. PRs should include the version number, command used, internal metrics, official score if available, files changed, and whether raw A-board data was included.

## Security & Configuration Tips

Do not publish `.env` or API keys. The raw A-board data in `A分类/` and `A推荐/` is intended to be tracked for collaborative review. Treat `prediction.zip` in the root as the latest local candidate, but keep it ignored; keep official-best references in `versions/` and update both Markdown ledgers after every submission.
