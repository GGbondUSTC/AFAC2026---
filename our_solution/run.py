#!/usr/bin/env python3
"""Entry point for our AFAC sparse-feedback solution."""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List

from src.classification import run_classification
from src.common import ensure_dir, set_seed, write_json, zip_submission
from src.recommendation import run_recommendation


def default_paths() -> Dict[str, Path]:
    solution_dir = Path(__file__).resolve().parent
    root = solution_dir.parent
    return {
        "cls_data": root / "A分类" / "A1.npz",
        "cls_sample": root / "A分类" / "sample_submission.csv",
        "rec_data": root / "A推荐",
        "output_dir": solution_dir / "output",
    }


def parse_args() -> argparse.Namespace:
    defaults = default_paths()
    parser = argparse.ArgumentParser(description="Sparse-feedback automated experiment solution")
    parser.add_argument("--task", type=int, action="append", choices=[1, 2], help="Task id to run. Repeat for both tasks.")
    parser.add_argument("--cls_data", type=Path, default=defaults["cls_data"], help="Path to classification .npz file.")
    parser.add_argument("--cls_sample", type=Path, default=defaults["cls_sample"], help="Path to classification sample submission.")
    parser.add_argument("--rec_data", type=Path, default=defaults["rec_data"], help="Path to recommendation data directory.")
    parser.add_argument("--output_dir", type=Path, default=defaults["output_dir"], help="Output directory.")
    parser.add_argument("--budget", type=int, default=5, help="Candidate rounds per task.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--val_ratio", type=float, default=0.12, help="Internal validation ratio.")
    parser.add_argument("--time_limit", type=float, default=7200.0, help="Total wall time budget in seconds.")
    parser.add_argument(
        "--zip_name",
        type=str,
        default="prediction.zip",
        help="Submission zip filename, created under output_dir.",
    )
    parser.add_argument(
        "--reuse_a1",
        type=Path,
        default=None,
        help="Reuse an existing A1.csv instead of running task 1. Useful for recommendation-only versions.",
    )
    parser.add_argument(
        "--reuse_a2",
        type=Path,
        default=None,
        help="Reuse an existing A2.csv instead of running task 2.",
    )
    parser.add_argument(
        "--version",
        type=str,
        default="",
        help="Version name such as v1 or v2. If omitted, the next versions/vN folder is used.",
    )
    parser.add_argument(
        "--versions_dir",
        type=Path,
        default=None,
        help="Directory for versioned archives. Defaults to <project_root>/versions.",
    )
    parser.add_argument(
        "--note",
        type=str,
        default="",
        help="Short human note describing what changed in this version.",
    )
    parser.add_argument(
        "--use_qwen_agent",
        action="store_true",
        help="Use Qwen API to propose extra locally validated candidate configs.",
    )
    parser.add_argument(
        "--qwen_env",
        type=Path,
        default=Path(".env"),
        help="Path to Qwen/DashScope .env or raw-key file.",
    )
    parser.add_argument(
        "--qwen_model",
        type=str,
        default=None,
        help="Optional Qwen model override. Defaults to QWEN_MODEL or qwen-plus.",
    )
    parser.add_argument(
        "--qwen_rounds",
        type=int,
        default=3,
        help="Extra Qwen-proposed configs per task when --use_qwen_agent is set.",
    )
    return parser.parse_args()


def next_version_name(versions_dir: Path) -> str:
    max_seen = 0
    if versions_dir.exists():
        for child in versions_dir.iterdir():
            if child.is_dir() and child.name.startswith("v"):
                suffix = child.name[1:]
                if suffix.isdigit():
                    max_seen = max(max_seen, int(suffix))
    return f"v{max_seen + 1}"


def copy_code_snapshot(solution_dir: Path, code_dir: Path) -> None:
    ensure_dir(code_dir)
    for name in ("run.py", "README.md", "requirements.txt", "config.example.yaml"):
        src = solution_dir / name
        if src.exists():
            shutil.copy2(src, code_dir / name)
    src_dir = solution_dir / "src"
    dst_dir = code_dir / "src"
    if src_dir.exists():
        if dst_dir.exists():
            shutil.rmtree(dst_dir)
        ignore = shutil.ignore_patterns("__pycache__", "*.pyc")
        shutil.copytree(src_dir, dst_dir, ignore=ignore)


def reset_submission_dir(submission_dir: Path) -> Path:
    ensure_dir(submission_dir)
    for name in ("A1.csv", "A2.csv", "trajectory_B1.json", "trajectory_B2.json"):
        path = submission_dir / name
        if path.exists():
            path.unlink()
    return submission_dir


def reset_run_outputs(output_dir: Path) -> None:
    for name in (
        "run_summary.json",
        "classification_result.json",
        "recommendation_result.json",
        "trajectory_B1.json",
        "trajectory_B2.json",
        "prediction.zip",
    ):
        path = output_dir / name
        if path.exists():
            path.unlink()


def load_json_if_exists(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def load_reused_task_result(reuse_path: Path, task_key: str) -> Dict[str, Any]:
    task_name = "classification" if task_key == "task1" else "recommendation"
    result: Dict[str, Any] = {
        "task": task_name,
        "reused": True,
        "reused_from": str(reuse_path),
    }

    version_dir = reuse_path.resolve().parent.parent
    summary = load_json_if_exists(version_dir / "version_summary.json")
    summary_results = summary.get("results", {}) if isinstance(summary.get("results"), dict) else {}
    source_task = summary_results.get(task_key, {}) if isinstance(summary_results, dict) else {}
    if isinstance(source_task, dict):
        result.update(source_task)
    if summary.get("version"):
        result["reused_from_version"] = summary.get("version")

    fallback_name = "classification_result.json" if task_key == "task1" else "recommendation_result.json"
    fallback = load_json_if_exists(version_dir / fallback_name)
    for key, value in fallback.items():
        result.setdefault(key, value)

    result["task"] = task_name
    result["reused"] = True
    result["reused_from"] = str(reuse_path)
    return result


def copy_reused_submission_file(reuse_path: Path, submission_dir: Path, filename: str) -> None:
    if not reuse_path.exists():
        raise FileNotFoundError(f"Reuse file not found: {reuse_path}")
    shutil.copy2(reuse_path, submission_dir / filename)


def resolve_qwen_env_path(path: Path, project_root: Path) -> Path:
    if path.is_absolute() or path.exists():
        return path
    return project_root / path


def archive_version(
    solution_dir: Path,
    output_dir: Path,
    submission_dir: Path,
    zip_path: Path,
    args: argparse.Namespace,
    results: Dict[str, object],
) -> Dict[str, str]:
    project_root = solution_dir.parent
    versions_dir = ensure_dir(args.versions_dir or (project_root / "versions"))
    version_name = args.version.strip() or next_version_name(versions_dir)
    version_dir = ensure_dir(versions_dir / version_name)
    for path in (
        version_dir / "prediction.zip",
        version_dir / "run_summary.json",
        version_dir / "classification_result.json",
        version_dir / "recommendation_result.json",
        version_dir / "trajectory_B1.json",
        version_dir / "trajectory_B2.json",
        version_dir / "version_summary.json",
    ):
        if path.exists():
            path.unlink()
    for path in (version_dir / "submission", version_dir / "code"):
        if path.exists():
            shutil.rmtree(path)

    shutil.copy2(zip_path, version_dir / "prediction.zip")
    latest_zip = project_root / "prediction.zip"
    shutil.copy2(zip_path, latest_zip)

    archived_submission = ensure_dir(version_dir / "submission")
    for name in ("A1.csv", "A2.csv", "trajectory_B1.json", "trajectory_B2.json"):
        src = submission_dir / name
        if src.exists():
            shutil.copy2(src, archived_submission / name)

    for name in (
        "run_summary.json",
        "classification_result.json",
        "recommendation_result.json",
        "trajectory_B1.json",
        "trajectory_B2.json",
    ):
        src = output_dir / name
        if src.exists():
            shutil.copy2(src, version_dir / name)

    copy_code_snapshot(solution_dir, version_dir / "code")

    version_summary = {
        "version": version_name,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime()),
        "note": args.note,
        "latest_root_zip": str(latest_zip),
        "version_zip": str(version_dir / "prediction.zip"),
        "source_output_dir": str(output_dir),
        "run_args": {
            "task": args.task,
            "cls_data": str(args.cls_data),
            "cls_sample": str(args.cls_sample),
            "rec_data": str(args.rec_data),
            "reuse_a1": str(args.reuse_a1) if args.reuse_a1 else None,
            "reuse_a2": str(args.reuse_a2) if args.reuse_a2 else None,
            "budget": args.budget,
            "seed": args.seed,
            "val_ratio": args.val_ratio,
            "time_limit": args.time_limit,
            "use_qwen_agent": bool(args.use_qwen_agent),
            "qwen_env": str(args.qwen_env),
            "qwen_model": args.qwen_model,
            "qwen_rounds": args.qwen_rounds,
        },
        "results": results,
    }
    write_json(version_dir / "version_summary.json", version_summary)
    update_version_index(versions_dir, version_name, version_summary)
    (project_root / "LATEST_VERSION.txt").write_text(
        f"{version_name}\n{latest_zip}\n{version_dir}\n",
        encoding="utf-8",
    )
    return {
        "version": version_name,
        "version_dir": str(version_dir),
        "version_zip": str(version_dir / "prediction.zip"),
        "latest_root_zip": str(latest_zip),
    }


def update_version_index(versions_dir: Path, version_name: str, version_summary: Dict[str, object]) -> None:
    index_path = versions_dir / "VERSION_INDEX.json"
    if index_path.exists():
        try:
            with index_path.open("r", encoding="utf-8") as f:
                index = json.load(f)
        except Exception:
            index = {"versions": []}
    else:
        index = {"versions": []}

    existing_versions = index.get("versions", [])
    existing_same = next((v for v in existing_versions if v.get("version") == version_name), {})
    versions = [v for v in existing_versions if v.get("version") != version_name]
    results = version_summary.get("results", {})
    task1 = results.get("task1", {}) if isinstance(results, dict) else {}
    task2 = results.get("task2", {}) if isinstance(results, dict) else {}
    cls_score = task1.get("best_val_acc") if isinstance(task1, dict) else None
    rec_score = task2.get("best_val_ndcg@10") if isinstance(task2, dict) else None
    rec_weighted_score = task2.get("best_test_weighted_ndcg@10") if isinstance(task2, dict) else None
    internal_avg = None
    if isinstance(cls_score, (int, float)) and isinstance(rec_score, (int, float)):
        internal_avg = 0.5 * float(cls_score) + 0.5 * float(rec_score)

    entry = {
        "version": version_name,
        "created_at": version_summary.get("created_at"),
        "note": version_summary.get("note", ""),
        "classification_val_acc": cls_score,
        "recommendation_val_ndcg@10": rec_score,
        "recommendation_test_weighted_ndcg@10": rec_weighted_score,
        "internal_avg": internal_avg,
        "reused_task1": bool(task1.get("reused")) if isinstance(task1, dict) else False,
        "reused_task2": bool(task2.get("reused")) if isinstance(task2, dict) else False,
        "version_zip": version_summary.get("version_zip"),
        "latest_root_zip": version_summary.get("latest_root_zip"),
    }
    for key, value in existing_same.items():
        if key.startswith("official_") and key not in entry:
            entry[key] = value
    versions.append(entry)
    versions.sort(key=lambda item: int(str(item["version"])[1:]) if str(item["version"]).startswith("v") and str(item["version"])[1:].isdigit() else 10**9)
    index["versions"] = versions
    numeric_scores = [v for v in versions if isinstance(v.get("internal_avg"), (int, float))]
    if numeric_scores:
        best = max(numeric_scores, key=lambda item: item["internal_avg"])
        index["best_by_internal_validation"] = best["version"]
    write_json(index_path, index)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    task_ids: List[int] = args.task if args.task else [1, 2]
    if args.reuse_a1 and 1 in task_ids:
        raise ValueError("--reuse_a1 cannot be combined with running task 1.")
    if args.reuse_a2 and 2 in task_ids:
        raise ValueError("--reuse_a2 cannot be combined with running task 2.")
    solution_dir = Path(__file__).resolve().parent
    project_root = solution_dir.parent
    qwen_env_path = resolve_qwen_env_path(args.qwen_env, project_root)
    args.qwen_env = qwen_env_path
    output_dir = ensure_dir(args.output_dir)
    reset_run_outputs(output_dir)
    submission_dir = reset_submission_dir(output_dir / "submission")
    start = time.time()
    results: Dict[str, object] = {}

    if 1 in task_ids:
        remaining = max(60.0, args.time_limit - (time.time() - start))
        results["task1"] = run_classification(
            data_path=args.cls_data,
            sample_path=args.cls_sample,
            output_dir=output_dir,
            budget=args.budget,
            seed=args.seed,
            val_ratio=args.val_ratio,
            time_limit=remaining,
            use_qwen_agent=args.use_qwen_agent,
            qwen_env_path=qwen_env_path,
            qwen_model=args.qwen_model,
            qwen_rounds=args.qwen_rounds,
        )
        traj = output_dir / "trajectory_B1.json"
        if traj.exists():
            shutil.copy2(traj, submission_dir / "trajectory_B1.json")

    if 2 in task_ids:
        remaining = max(60.0, args.time_limit - (time.time() - start))
        results["task2"] = run_recommendation(
            data_dir=args.rec_data,
            output_dir=output_dir,
            budget=args.budget,
            seed=args.seed,
            val_ratio=args.val_ratio,
            time_limit=remaining,
            use_qwen_agent=args.use_qwen_agent,
            qwen_env_path=qwen_env_path,
            qwen_model=args.qwen_model,
            qwen_rounds=args.qwen_rounds,
        )
        traj = output_dir / "trajectory_B2.json"
        if traj.exists():
            shutil.copy2(traj, submission_dir / "trajectory_B2.json")

    if args.reuse_a1:
        copy_reused_submission_file(args.reuse_a1, submission_dir, "A1.csv")
        results["task1"] = load_reused_task_result(args.reuse_a1, "task1")
    if args.reuse_a2:
        copy_reused_submission_file(args.reuse_a2, submission_dir, "A2.csv")
        results["task2"] = load_reused_task_result(args.reuse_a2, "task2")

    missing = [name for name in ("A1.csv", "A2.csv") if not (submission_dir / name).exists()]
    if missing:
        raise FileNotFoundError(
            f"Submission is missing {', '.join(missing)}. Run both tasks or provide --reuse_a1/--reuse_a2."
        )

    zip_path = output_dir / args.zip_name
    zip_submission(submission_dir, zip_path)
    results["submission_dir"] = str(submission_dir)
    results["prediction_zip"] = str(zip_path)
    results["total_duration"] = round(time.time() - start, 4)
    write_json(output_dir / "run_summary.json", results)
    archive_info = archive_version(solution_dir, output_dir, submission_dir, zip_path, args, results)
    results["archive"] = archive_info
    write_json(output_dir / "run_summary.json", results)
    write_json(Path(archive_info["version_dir"]) / "run_summary.json", results)

    print("Run complete")
    print(f"submission_dir={submission_dir}")
    print(f"prediction_zip={zip_path}")
    print(f"latest_root_zip={archive_info['latest_root_zip']}")
    print(f"version_zip={archive_info['version_zip']}")


if __name__ == "__main__":
    main()
