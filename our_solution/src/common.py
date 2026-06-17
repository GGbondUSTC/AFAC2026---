"""Shared utilities for the AFAC sparse-feedback solution."""

from __future__ import annotations

import json
import os
import random
import time
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def ensure_dir(path: os.PathLike[str] | str) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def write_json(path: os.PathLike[str] | str, data: Dict[str, Any]) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def now_ts() -> str:
    # Use local wall time. The trajectory is for reproducibility, not ranking.
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())


class Timer:
    def __init__(self) -> None:
        self.start_time = time.time()

    @property
    def elapsed(self) -> float:
        return time.time() - self.start_time


def stratified_split(
    labels: np.ndarray,
    train_idx: np.ndarray,
    val_ratio: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Create a deterministic stratified train/validation split."""
    rng = np.random.default_rng(seed)
    fit_parts: List[np.ndarray] = []
    val_parts: List[np.ndarray] = []
    y = labels[train_idx]
    for cls in np.unique(y):
        cls_idx = np.asarray(train_idx[y == cls])
        cls_idx = cls_idx[rng.permutation(len(cls_idx))]
        val_size = max(1, int(round(len(cls_idx) * val_ratio)))
        val_parts.append(cls_idx[:val_size])
        fit_parts.append(cls_idx[val_size:])
    fit_idx = np.concatenate(fit_parts)
    val_idx = np.concatenate(val_parts)
    fit_idx = fit_idx[rng.permutation(len(fit_idx))]
    val_idx = val_idx[rng.permutation(len(val_idx))]
    return fit_idx, val_idx


def ndcg_at_k(predictions: Sequence[Sequence[str]], targets: Sequence[str], k: int = 10) -> float:
    scores: List[float] = []
    for pred, target in zip(predictions, targets):
        score = 0.0
        for rank, iid in enumerate(pred[:k], start=1):
            if iid == target:
                score = 1.0 / np.log2(rank + 1)
                break
        scores.append(score)
    return float(np.mean(scores)) if scores else 0.0


def zip_submission(submission_dir: os.PathLike[str] | str, zip_path: os.PathLike[str] | str) -> None:
    submission_dir = Path(submission_dir)
    zip_path = Path(zip_path)
    ensure_dir(zip_path.parent)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for name in ("A1.csv", "A2.csv"):
            file_path = submission_dir / name
            if file_path.exists():
                zf.write(file_path, arcname=name)


@dataclass
class RoundRecord:
    round: int
    config: Dict[str, Any]
    feedback: Dict[str, Any]
    strategy: str
    duration: float
    timestamp: str = field(default_factory=now_ts)


@dataclass
class Trajectory:
    task_id: str
    objective: str
    records: List[RoundRecord] = field(default_factory=list)
    best_round: Optional[int] = None
    selected_config: Dict[str, Any] = field(default_factory=dict)

    def add(
        self,
        round_id: int,
        config: Dict[str, Any],
        feedback: Dict[str, Any],
        strategy: str,
        duration: float,
    ) -> None:
        self.records.append(
            RoundRecord(
                round=round_id,
                config=config,
                feedback=feedback,
                strategy=strategy,
                duration=round(float(duration), 4),
            )
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "objective": self.objective,
            "num_rounds": len(self.records),
            "best_round": self.best_round,
            "selected_config": self.selected_config,
            "records": [
                {
                    "round": r.round,
                    "config": r.config,
                    "feedback": r.feedback,
                    "strategy": r.strategy,
                    "duration": r.duration,
                    "timestamp": r.timestamp,
                }
                for r in self.records
            ],
        }

