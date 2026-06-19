#!/usr/bin/env python3
"""Focused sweep around the v29 count-only medium/long-history rerank."""

from __future__ import annotations

import argparse
import itertools
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import pandas as pd

SOLUTION_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = SOLUTION_DIR.parent
if str(SOLUTION_DIR) not in sys.path:
    sys.path.insert(0, str(SOLUTION_DIR))

from src.common import ensure_dir, set_seed, write_json  # noqa: E402
from src.recommendation import (  # noqa: E402
    EXACT_LENGTH_BINS,
    build_recommender,
    exact_length_bin,
    parse_sequence,
    test_exact_len_weights,
    test_like_recommendation_split,
    v17_conservative_shortseq_config,
    weighted_exact_score,
)
from src.validation import compare_prediction_sets, exact_bins_for_frame, summarize_pairwise_results  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep v29-style count-only rerank variants.")
    parser.add_argument("--rec_data", type=Path, default=PROJECT_ROOT / "A推荐")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--val_ratio", type=float, default=0.12)
    parser.add_argument("--max_base_pool", type=int, default=70)
    parser.add_argument(
        "--output",
        type=Path,
        default=SOLUTION_DIR / "output" / "v30_count_only_grid_eval.json",
    )
    return parser.parse_args()


def load_data(rec_data: Path):
    train_df = pd.read_csv(rec_data / "train.csv").astype({"uid": str, "target_iid": str})
    test_df = pd.read_csv(rec_data / "test.csv").astype({"uid": str})
    user_df = pd.read_csv(rec_data / "user.csv").astype({"uid": str})
    item_df = pd.read_csv(rec_data / "item.csv").astype({"iid": str})
    candidates = item_df["iid"].astype(str).tolist()
    return train_df, test_df, user_df, item_df, candidates


def variants() -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []

    def add(
        name: str,
        min_len: int,
        freeze_top_n: int,
        base_pool: int,
        base_rank_weight: float,
        alpha: float,
        count_weight: float,
        count_transform: str,
    ) -> None:
        out.append(
            {
                "name": name,
                "min_len": int(min_len),
                "freeze_top_n": int(freeze_top_n),
                "base_pool": int(base_pool),
                "base_rank_weight": float(base_rank_weight),
                "alpha": float(alpha),
                "count_weight": float(count_weight),
                "count_transform": count_transform,
            }
        )

    add("v29_count_log_len21_pool25_rank1_alpha032_cw1", 21, 1, 25, 1.0, 0.32, 1.0, "log")

    for min_len, base_pool, alpha in itertools.product(
        (18, 21, 24, 26, 31),
        (20, 25, 30, 35, 40),
        (0.24, 0.28, 0.30, 0.32, 0.34, 0.36, 0.40, 0.45, 0.50),
    ):
        add(
            f"cnt_log_len{min_len}_pool{base_pool}_a{alpha:g}",
            min_len,
            1,
            base_pool,
            1.0,
            alpha,
            1.0,
            "log",
        )

    for rank_weight, alpha in itertools.product((0.75, 0.85, 1.15, 1.30), (0.24, 0.28, 0.32, 0.36, 0.42)):
        add(
            f"cnt_log_len21_pool25_rank{rank_weight:g}_a{alpha:g}",
            21,
            1,
            25,
            rank_weight,
            alpha,
            1.0,
            "log",
        )

    for count_weight, alpha in itertools.product((0.75, 0.9, 1.1, 1.25, 1.5), (0.24, 0.28, 0.32, 0.36, 0.42)):
        add(
            f"cnt_log_len21_pool25_cw{count_weight:g}_a{alpha:g}",
            21,
            1,
            25,
            1.0,
            alpha,
            count_weight,
            "log",
        )

    for transform, alpha in itertools.product(("sqrt", "linear", "binary"), (0.08, 0.12, 0.16, 0.20, 0.28, 0.36)):
        add(
            f"cnt_{transform}_len21_pool25_a{alpha:g}",
            21,
            1,
            25,
            1.0,
            alpha,
            1.0,
            transform,
        )

    for freeze_top_n, alpha in itertools.product((2, 3), (0.24, 0.28, 0.32, 0.40)):
        add(
            f"cnt_log_len21_pool25_freeze{freeze_top_n}_a{alpha:g}",
            21,
            freeze_top_n,
            25,
            1.0,
            alpha,
            1.0,
            "log",
        )

    deduped: Dict[str, Dict[str, Any]] = {}
    for variant in out:
        deduped.setdefault(variant["name"], variant)
    return list(deduped.values())


def row_infos(df: pd.DataFrame) -> List[Dict[str, Any]]:
    infos: List[Dict[str, Any]] = []
    for _, row in df.iterrows():
        hist = parse_sequence(row.get("item_seq_raw", ""))
        counts = Counter(hist)
        recency = {iid: pos for pos, iid in enumerate(hist)}
        hist_len = len(hist)
        max_count = max(counts.values()) if counts else 0
        infos.append(
            {
                "hist": hist,
                "counts": counts,
                "recency": recency,
                "hist_len": hist_len,
                "max_count": max_count,
            }
        )
    return infos


def count_feature(count: int, max_count: int, transform: str) -> float:
    if count <= 0 or max_count <= 0:
        return 0.0
    if transform == "linear":
        return count / max_count
    if transform == "sqrt":
        return math.sqrt(count) / math.sqrt(max_count)
    if transform == "binary":
        return 1.0
    return math.log1p(count) / math.log1p(max_count)


def rerank(base_items: Sequence[str], info: Dict[str, Any], variant: Dict[str, Any], k: int = 10) -> List[str]:
    base_pool = max(k, min(int(variant["base_pool"]), len(base_items)))
    base_items = list(base_items[:base_pool])
    hist = info["hist"]
    if len(hist) < int(variant["min_len"]):
        return base_items[:k]
    freeze_top_n = int(variant["freeze_top_n"])
    if len(base_items) <= freeze_top_n:
        return base_items[:k]
    counts = info["counts"]
    if not counts:
        return base_items[:k]

    max_count = max(1, int(info["max_count"]))
    alpha = float(variant["alpha"])
    count_weight = float(variant["count_weight"])
    base_rank_weight = float(variant["base_rank_weight"])
    transform = str(variant["count_transform"])

    frozen = base_items[:freeze_top_n]
    tail = base_items[freeze_top_n:base_pool]
    tail_pos = {iid: pos for pos, iid in enumerate(tail)}
    scores: Dict[str, float] = {}
    for rank, iid in enumerate(tail, start=freeze_top_n + 1):
        scores[iid] = base_rank_weight / math.log2(rank + 1)
        if iid in counts:
            scores[iid] += alpha * count_weight * count_feature(int(counts[iid]), max_count, transform)
    ordered_tail = sorted(tail, key=lambda iid: (-scores.get(iid, 0.0), tail_pos[iid], iid))
    return (frozen + ordered_tail)[:k]


def evaluate_variant(
    variant: Dict[str, Any],
    seed: int,
    base_masked: List[List[str]],
    base_val: List[List[str]],
    masked_infos: List[Dict[str, Any]],
    val_infos: List[Dict[str, Any]],
    masked_val_df: pd.DataFrame,
    val_df: pd.DataFrame,
    exact_weights: Dict[str, float],
) -> Dict[str, Any]:
    masked_candidate = [rerank(base, info, variant) for base, info in zip(base_masked, masked_infos)]
    val_candidate = [rerank(base, info, variant) for base, info in zip(base_val, val_infos)]
    masked_targets = masked_val_df["target_iid"].astype(str).tolist()
    val_targets = val_df["target_iid"].astype(str).tolist()
    masked = compare_prediction_sets([x[:10] for x in base_masked], masked_candidate, masked_targets, exact_bins_for_frame(masked_val_df))
    natural = compare_prediction_sets([x[:10] for x in base_val], val_candidate, val_targets, exact_bins_for_frame(val_df))
    base_exact = weighted_exact_score({b: masked["by_exact_len"][b]["base"] for b in EXACT_LENGTH_BINS}, exact_weights)
    cand_exact = weighted_exact_score({b: masked["by_exact_len"][b]["candidate"] for b in EXACT_LENGTH_BINS}, exact_weights)
    changed_by_len: Dict[str, int] = {b: 0 for b in EXACT_LENGTH_BINS}
    top1_by_len: Dict[str, int] = {b: 0 for b in EXACT_LENGTH_BINS}
    for base, cand, raw_len in zip(base_masked, masked_candidate, masked_val_df["_masked_raw_len"].astype(int).tolist()):
        bin_name = exact_length_bin(raw_len)
        if list(base[:10]) != list(cand[:10]):
            changed_by_len[bin_name] += 1
        if (list(base[:1]) or [""])[0] != (list(cand[:1]) or [""])[0]:
            top1_by_len[bin_name] += 1
    return {
        "seed": int(seed),
        "candidate": variant["name"],
        "variant": variant,
        "natural": natural,
        "masked": masked,
        "masked_exact_weighted_base": base_exact,
        "masked_exact_weighted_candidate": cand_exact,
        "masked_exact_weighted_delta": cand_exact - base_exact,
        "changed_by_exact_len": changed_by_len,
        "top1_changed_by_exact_len": top1_by_len,
    }


def summarize(results: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for name, split_results in results.items():
        summary = summarize_pairwise_results(split_results)
        rows.append(
            {
                "candidate": name,
                "summary": summary,
                "variant": split_results[0]["variant"] if split_results else {},
                "splits": split_results,
            }
        )
    rows.sort(
        key=lambda row: (
            float(row["summary"]["mean_masked_exact_weighted_delta"]),
            float(row["summary"]["min_masked_exact_weighted_delta"]),
        ),
        reverse=True,
    )
    return rows


def main() -> None:
    args = parse_args()
    train_df, test_df, user_df, item_df, candidates = load_data(args.rec_data)
    exact_weights = test_exact_len_weights(test_df)
    grid = variants()
    max_pool = max(int(args.max_base_pool), max(int(v["base_pool"]) for v in grid))
    all_results: Dict[str, List[Dict[str, Any]]] = {v["name"]: [] for v in grid}

    for seed in args.seeds:
        set_seed(seed)
        fit_df, val_df, masked_val_df, _ = test_like_recommendation_split(
            train_df, test_df, val_ratio=args.val_ratio, seed=seed
        )
        base_model = build_recommender(v17_conservative_shortseq_config(), candidates, user_df, item_df).fit(fit_df)
        base_val = [base_model.predict_row(row, k=max_pool) for _, row in val_df.iterrows()]
        base_masked = [base_model.predict_row(row, k=max_pool) for _, row in masked_val_df.iterrows()]
        val_infos = row_infos(val_df)
        masked_infos = row_infos(masked_val_df)
        for variant in grid:
            result = evaluate_variant(
                variant,
                seed,
                base_masked,
                base_val,
                masked_infos,
                val_infos,
                masked_val_df,
                val_df,
                exact_weights,
            )
            all_results[variant["name"]].append(result)
        partial = summarize(all_results)
        top = partial[0]
        print(
            f"seed={seed} top={top['candidate']} "
            f"mean={top['summary']['mean_masked_exact_weighted_delta']:.6f} "
            f"min={top['summary']['min_masked_exact_weighted_delta']:.6f}",
            flush=True,
        )

    results = summarize(all_results)
    payload = {
        "rec_data": str(args.rec_data),
        "seeds": list(args.seeds),
        "val_ratio": float(args.val_ratio),
        "baseline": "v17_conservative_shortseq_config",
        "exact_weights": exact_weights,
        "variants": grid,
        "results": results,
    }
    ensure_dir(args.output.parent)
    write_json(args.output, payload)
    for row in results[:15]:
        summary = row["summary"]
        print(
            f"{row['candidate']} mean={summary['mean_masked_exact_weighted_delta']:.6f} "
            f"min={summary['min_masked_exact_weighted_delta']:.6f} "
            f"splits={summary['split_deltas']}",
            flush=True,
        )
    print(f"wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
