#!/usr/bin/env python3
"""Probe safer zero-history neural/profile ensemble variants against v24."""

from __future__ import annotations

import argparse
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, DefaultDict, Dict, List, Sequence

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
    parse_sequence,
    test_exact_len_weights,
    test_like_recommendation_split,
    v24_medium_long_history_count_recent_strong_config,
    weighted_exact_score,
)
from src.validation import compare_prediction_sets, exact_bins_for_frame, summarize_pairwise_results  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate zero-history ensemble variants versus v24.")
    parser.add_argument("--rec_data", type=Path, default=PROJECT_ROOT / "A推荐")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--val_ratio", type=float, default=0.12)
    parser.add_argument(
        "--output",
        type=Path,
        default=SOLUTION_DIR / "output" / "v28_zero_ensemble_eval.json",
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
    for alpha in (0.30, 0.40, 0.48, 0.55, 0.62):
        for neural_pool in (10, 15, 20):
            out.append(
                {
                    "name": f"v28_zero_alpha{alpha:g}_npool{neural_pool}",
                    "base_pool": 100,
                    "neural_pool": neural_pool,
                    "neural_rank_weight": alpha,
                    "support_gate": "none",
                }
            )
    for base_pool in (80, 120, 180, 260):
        out.append(
            {
                "name": f"v28_zero_basepool{base_pool}_alpha045",
                "base_pool": base_pool,
                "neural_pool": 15,
                "neural_rank_weight": 0.45,
                "support_gate": "none",
            }
        )
    for gate in ("downweight_low_support", "downweight_high_support", "profile_when_supported"):
        for alpha in (0.40, 0.50, 0.60):
            out.append(
                {
                    "name": f"v28_zero_{gate}_alpha{alpha:g}",
                    "base_pool": 140,
                    "neural_pool": 15,
                    "neural_rank_weight": alpha,
                    "support_gate": gate,
                }
            )
    out.append(
        {
            "name": "v28_zero_seed42_heavy_alpha045",
            "base_pool": 140,
            "neural_pool": 15,
            "neural_rank_weight": 0.45,
            "support_gate": "none",
            "seed_weights": (1.35, 0.85, 0.85),
        }
    )
    out.append(
        {
            "name": "v28_zero_seed_robust_alpha045",
            "base_pool": 140,
            "neural_pool": 15,
            "neural_rank_weight": 0.45,
            "support_gate": "none",
            "seed_weights": (0.9, 1.05, 1.05),
        }
    )
    return out


def support_for_uid(zero_model: Any, uid: str) -> int:
    profile_model = getattr(zero_model, "base_model", None)
    if profile_model is None or getattr(profile_model, "user_lookup", pd.DataFrame()).empty:
        return 0
    user_lookup = profile_model.user_lookup
    if uid not in user_lookup.index:
        return 0
    user_row = user_lookup.loc[uid]
    if isinstance(user_row, pd.DataFrame):
        user_row = user_row.iloc[0]
    supports: List[int] = []
    for cols in getattr(profile_model, "zero_additive_specs", ()):
        if not all(col in user_row.index and pd.notna(user_row[col]) for col in cols):
            continue
        key = tuple(str(user_row[col]) for col in cols)
        supports.append(int(profile_model.zero_additive_totals.get(cols, {}).get(key, 0)))
    return max(supports) if supports else 0


def alpha_with_gate(base_alpha: float, support: int, gate: str) -> float:
    if gate == "downweight_low_support":
        factor = 0.55 + 0.45 * min(1.0, math.log1p(max(support, 0)) / math.log1p(240.0))
        return base_alpha * factor
    if gate == "downweight_high_support":
        factor = 1.0 - 0.35 * min(1.0, math.log1p(max(support, 0)) / math.log1p(240.0))
        return base_alpha * factor
    if gate == "profile_when_supported" and support >= 160:
        return base_alpha * 0.55
    return base_alpha


def fuse_zero(zero_model: Any, uid: str, variant: Dict[str, Any], k: int = 10) -> List[str]:
    profile_model = zero_model.base_model
    base_pool = int(variant.get("base_pool", 100))
    neural_pool = int(variant.get("neural_pool", 15))
    support = support_for_uid(zero_model, uid)
    alpha = alpha_with_gate(
        float(variant.get("neural_rank_weight", 0.55)),
        support,
        str(variant.get("support_gate", "none")),
    )
    base_items = profile_model._zero_additive_prediction(uid, base_pool)
    scores: Dict[str, float] = {}
    for rank, iid in enumerate(base_items, start=1):
        scores[iid] = scores.get(iid, 0.0) + 1.0 / math.log2(rank + 1)
    seed_weights = tuple(float(x) for x in variant.get("seed_weights", ()))
    models = zero_model.models if zero_model.models else ([zero_model.model] if zero_model.model is not None else [])
    for model_idx, model in enumerate(models):
        model_weight = seed_weights[model_idx] if model_idx < len(seed_weights) else 1.0
        for rank, iid in enumerate(zero_model._neural_top_for_model(model, uid, neural_pool), start=1):
            scores[iid] = scores.get(iid, 0.0) + alpha * model_weight / math.log2(rank + 1)
    return sorted(scores, key=lambda iid: (-scores[iid], iid))[:k]


def candidate_predictions(
    frame: pd.DataFrame,
    baseline_predictions: Sequence[Sequence[str]],
    zero_model: Any,
    variant: Dict[str, Any],
) -> List[List[str]]:
    out: List[List[str]] = []
    for (_, row), baseline in zip(frame.iterrows(), baseline_predictions):
        hist = parse_sequence(row.get("item_seq_raw", ""))
        if len(hist) == 0:
            out.append(fuse_zero(zero_model, str(row.get("uid", "")), variant, k=10))
        else:
            out.append(list(baseline[:10]))
    return out


def evaluate_split(
    seed: int,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    user_df: pd.DataFrame,
    item_df: pd.DataFrame,
    candidates: List[str],
    args: argparse.Namespace,
) -> List[Dict[str, Any]]:
    set_seed(seed)
    fit_df, val_df, masked_val_df, _ = test_like_recommendation_split(
        train_df, test_df, val_ratio=args.val_ratio, seed=seed
    )
    v24_config = v24_medium_long_history_count_recent_strong_config()
    v24_model = build_recommender(v24_config, candidates, user_df, item_df).fit(fit_df)
    zero_model = v24_model.base_model
    baseline_masked = [v24_model.predict_row(row, k=10) for _, row in masked_val_df.iterrows()]
    baseline_val = [v24_model.predict_row(row, k=10) for _, row in val_df.iterrows()]
    exact_weights = test_exact_len_weights(test_df)
    split_rows: List[Dict[str, Any]] = []
    for variant in variants():
        masked_candidate = candidate_predictions(masked_val_df, baseline_masked, zero_model, variant)
        val_candidate = candidate_predictions(val_df, baseline_val, zero_model, variant)
        masked = compare_prediction_sets(
            baseline_masked,
            masked_candidate,
            masked_val_df["target_iid"].astype(str).tolist(),
            exact_bins_for_frame(masked_val_df),
        )
        natural = compare_prediction_sets(
            baseline_val,
            val_candidate,
            val_df["target_iid"].astype(str).tolist(),
            exact_bins_for_frame(val_df),
        )
        base_weighted = weighted_exact_score(
            {b: masked["by_exact_len"][b]["base"] for b in EXACT_LENGTH_BINS},
            exact_weights,
        )
        candidate_weighted = weighted_exact_score(
            {b: masked["by_exact_len"][b]["candidate"] for b in EXACT_LENGTH_BINS},
            exact_weights,
        )
        split_rows.append(
            {
                "seed": int(seed),
                "candidate": variant["name"],
                "variant": variant,
                "natural": natural,
                "masked": masked,
                "masked_exact_weighted_base": base_weighted,
                "masked_exact_weighted_candidate": candidate_weighted,
                "masked_exact_weighted_delta": candidate_weighted - base_weighted,
            }
        )
    return split_rows


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
    all_results: DefaultDict[str, List[Dict[str, Any]]] = defaultdict(list)
    for seed in args.seeds:
        split_rows = evaluate_split(seed, train_df, test_df, user_df, item_df, candidates, args)
        for row in split_rows:
            all_results[row["candidate"]].append(row)
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
        "baseline": "v24",
        "results": results,
    }
    ensure_dir(args.output.parent)
    write_json(args.output, payload)
    for row in results[:10]:
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
