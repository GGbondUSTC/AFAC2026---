#!/usr/bin/env python3
"""Fast long-history rerank sweep for v23 probes.

The expensive fitted baseline is shared once per split. Variants only change
the deterministic tail rerank applied on top of v17/v22 base predictions.
"""

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
    LENGTH_BINS,
    build_recommender,
    length_bin,
    parse_sequence,
    test_bin_weights,
    test_exact_len_weights,
    test_like_recommendation_split,
    v17_conservative_shortseq_config,
    weighted_bin_score,
    weighted_exact_score,
)
from src.validation import compare_prediction_sets, exact_bins_for_frame, summarize_pairwise_results  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep v23 long-history rerank variants.")
    parser.add_argument("--rec_data", type=Path, default=PROJECT_ROOT / "A推荐")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--val_ratio", type=float, default=0.12)
    parser.add_argument("--max_base_pool", type=int, default=50)
    parser.add_argument(
        "--output",
        type=Path,
        default=SOLUTION_DIR / "output" / "v23_long_history_grid_eval.json",
    )
    return parser.parse_args()


def load_data(rec_data: Path):
    train_df = pd.read_csv(rec_data / "train.csv").astype({"uid": str, "target_iid": str})
    test_df = pd.read_csv(rec_data / "test.csv").astype({"uid": str})
    user_df = pd.read_csv(rec_data / "user.csv").astype({"uid": str})
    item_df = pd.read_csv(rec_data / "item.csv").astype({"iid": str})
    candidates = item_df["iid"].astype(str).tolist()
    return train_df, test_df, user_df, item_df, candidates


def variant_grid() -> List[Dict[str, Any]]:
    variants: List[Dict[str, Any]] = []

    def add(
        name: str,
        min_len: int = 31,
        freeze_top_n: int = 1,
        base_pool: int = 20,
        base_rank_weight: float = 1.0,
        alpha: float = 0.10,
        count_weight: float = 0.8,
        recency_weight: float = 0.4,
        last_weight: float = 0.1,
    ) -> None:
        variants.append(
            {
                "name": name,
                "min_len": int(min_len),
                "freeze_top_n": int(freeze_top_n),
                "base_pool": int(base_pool),
                "base_rank_weight": float(base_rank_weight),
                "alpha": float(alpha),
                "count_weight": float(count_weight),
                "recency_weight": float(recency_weight),
                "last_weight": float(last_weight),
            }
        )

    add("v22_long_history_count_recent_len31")
    for alpha in (0.04, 0.06, 0.08, 0.12, 0.15, 0.20, 0.25, 0.30, 0.40, 0.60):
        add(f"len31_pool20_alpha{alpha:g}", alpha=alpha)
    for min_len, alpha in itertools.product((11, 21, 31, 41, 61, 81), (0.08, 0.10, 0.12, 0.15, 0.20, 0.25)):
        add(f"len{min_len}_pool20_alpha{alpha:g}", min_len=min_len, alpha=alpha)
    for pool, alpha in itertools.product((10, 12, 15, 20, 30, 50), (0.08, 0.10, 0.12, 0.15, 0.20, 0.25)):
        add(f"len31_pool{pool}_alpha{alpha:g}", base_pool=pool, alpha=alpha)
    weight_sets = {
        "count_only": (1.0, 0.0, 0.0),
        "count_recent": (1.0, 0.25, 0.0),
        "count_recent_last": (0.8, 0.4, 0.1),
        "count_strong": (1.2, 0.2, 0.0),
        "recent_strong": (0.4, 0.8, 0.2),
        "recent_only": (0.0, 1.0, 0.2),
        "last_heavy": (0.8, 0.4, 0.5),
    }
    for label, (cw, rw, lw) in weight_sets.items():
        for alpha in (0.08, 0.10, 0.12, 0.15):
            add(f"len31_{label}_alpha{alpha:g}", alpha=alpha, count_weight=cw, recency_weight=rw, last_weight=lw)
    for rank_weight in (0.7, 0.85, 1.15, 1.3):
        for alpha in (0.08, 0.10, 0.12):
            add(f"len31_rank{rank_weight:g}_alpha{alpha:g}", base_rank_weight=rank_weight, alpha=alpha)
    for freeze_top_n in (2, 3):
        for alpha in (0.10, 0.12, 0.15):
            add(f"len31_freeze{freeze_top_n}_alpha{alpha:g}", freeze_top_n=freeze_top_n, alpha=alpha)

    # v25 probes: refine around the v24 medium-long setting that transferred
    # online, while keeping top1 frozen and avoiding short-history users.
    for min_len, alpha in itertools.product((14, 16, 18, 21, 24, 26, 31), (0.18, 0.22, 0.25, 0.28, 0.32, 0.36)):
        add(f"refine_len{min_len}_pool20_alpha{alpha:g}", min_len=min_len, alpha=alpha)
    for pool, alpha in itertools.product((12, 15, 18, 20, 25, 30, 40, 50), (0.18, 0.22, 0.25, 0.28, 0.32)):
        add(f"refine_len21_pool{pool}_alpha{alpha:g}", min_len=21, base_pool=pool, alpha=alpha)
    refined_weight_sets = {
        "count_only": (1.0, 0.0, 0.0),
        "count_recent": (1.0, 0.25, 0.0),
        "balanced": (0.8, 0.4, 0.1),
        "count_strong": (1.2, 0.2, 0.0),
        "recent_strong": (0.4, 0.8, 0.2),
        "last_heavy": (0.8, 0.4, 0.5),
    }
    for label, (cw, rw, lw) in refined_weight_sets.items():
        for alpha in (0.20, 0.25, 0.30):
            add(
                f"refine_len21_{label}_alpha{alpha:g}",
                min_len=21,
                alpha=alpha,
                count_weight=cw,
                recency_weight=rw,
                last_weight=lw,
            )

    # v26 probes: v25's small offline edge did not move the rounded public
    # score, so search for a larger v24-relative change while keeping top1
    # protected and not touching short histories.
    for rank_weight in (0.55, 0.70, 0.85, 1.00, 1.15, 1.30):
        for pool, alpha in itertools.product((20, 25, 30, 40, 50), (0.22, 0.25, 0.30, 0.35, 0.42)):
            add(
                f"v26_len21_pool{pool}_rank{rank_weight:g}_alpha{alpha:g}",
                min_len=21,
                base_pool=pool,
                base_rank_weight=rank_weight,
                alpha=alpha,
            )
    for freeze_top_n in (2, 3):
        for pool, alpha in itertools.product((20, 25, 30), (0.25, 0.32, 0.40)):
            add(
                f"v26_len21_pool{pool}_freeze{freeze_top_n}_alpha{alpha:g}",
                min_len=21,
                freeze_top_n=freeze_top_n,
                base_pool=pool,
                alpha=alpha,
            )
    for label, (cw, rw, lw) in {
        "count_only": (1.0, 0.0, 0.0),
        "count_strong": (1.35, 0.15, 0.0),
        "count_recent": (1.1, 0.35, 0.0),
        "balanced": (0.8, 0.4, 0.1),
        "recent_strong": (0.35, 0.9, 0.2),
        "last_heavy": (0.8, 0.4, 0.8),
    }.items():
        for pool, alpha in itertools.product((20, 25, 30), (0.25, 0.32, 0.40)):
            add(
                f"v26_len21_pool{pool}_{label}_alpha{alpha:g}",
                min_len=21,
                base_pool=pool,
                alpha=alpha,
                count_weight=cw,
                recency_weight=rw,
                last_weight=lw,
            )

    deduped: Dict[str, Dict[str, Any]] = {}
    for variant in variants:
        deduped.setdefault(variant["name"], variant)
    return list(deduped.values())


def row_infos(df: pd.DataFrame) -> List[Dict[str, Any]]:
    infos: List[Dict[str, Any]] = []
    for _, row in df.iterrows():
        hist = parse_sequence(row.get("item_seq_raw", ""))
        counts = Counter(hist)
        recency = {iid: pos for pos, iid in enumerate(hist)}
        infos.append(
            {
                "hist": hist,
                "counts": counts,
                "recency": recency,
                "max_count": max(counts.values()) if counts else 0,
                "hist_len": len(hist),
            }
        )
    return infos


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
    hist_len = max(1, int(info["hist_len"]))
    recency = info["recency"]
    alpha = float(variant["alpha"])
    count_weight = float(variant["count_weight"])
    recency_weight = float(variant["recency_weight"])
    last_weight = float(variant["last_weight"])
    base_rank_weight = float(variant["base_rank_weight"])

    frozen = base_items[:freeze_top_n]
    tail = base_items[freeze_top_n:base_pool]
    scores: Dict[str, float] = {}
    tail_pos = {iid: pos for pos, iid in enumerate(tail)}
    for rank, iid in enumerate(tail, start=freeze_top_n + 1):
        scores[iid] = base_rank_weight / math.log2(rank + 1)
        if iid not in counts:
            continue
        count_feature = math.log1p(counts[iid]) / math.log1p(max_count)
        recency_feature = (recency[iid] + 1) / hist_len
        last_feature = 1.0 if iid == hist[-1] else 0.0
        scores[iid] += alpha * (
            count_weight * count_feature
            + recency_weight * recency_feature
            + last_weight * last_feature
        )
    ordered_tail = sorted(tail, key=lambda iid: (-scores.get(iid, 0.0), tail_pos[iid], iid))
    return (frozen + ordered_tail)[:k]


def score_by_named_bins(predictions: Sequence[Sequence[str]], targets: Sequence[str], bins: Sequence[str]) -> Dict[str, float]:
    values: Dict[str, List[float]] = {name: [] for name in LENGTH_BINS}
    for pred, target, bin_name in zip(predictions, targets, bins):
        gain = 0.0
        for rank, iid in enumerate(pred[:10], start=1):
            if iid == target:
                gain = 1.0 / math.log2(rank + 1)
                break
        values[str(bin_name)].append(gain)
    return {name: float(np.mean(values[name])) if values[name] else 0.0 for name in LENGTH_BINS}


def changed_by_named_bins(
    base_predictions: Sequence[Sequence[str]],
    candidate_predictions: Sequence[Sequence[str]],
    bins: Sequence[str],
) -> Dict[str, int]:
    changed = {name: 0 for name in LENGTH_BINS}
    top1 = {name: 0 for name in LENGTH_BINS}
    for base, cand, bin_name in zip(base_predictions, candidate_predictions, bins):
        name = str(bin_name)
        if list(base[:10]) != list(cand[:10]):
            changed[name] += 1
        if (list(base[:1]) or [""])[0] != (list(cand[:1]) or [""])[0]:
            top1[name] += 1
    return {f"{name}_changed": changed[name] for name in LENGTH_BINS} | {
        f"{name}_top1_changed": top1[name] for name in LENGTH_BINS
    }


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
    coarse_weights: Dict[str, float],
) -> Dict[str, Any]:
    masked_candidate = [rerank(base, info, variant) for base, info in zip(base_masked, masked_infos)]
    val_candidate = [rerank(base, info, variant) for base, info in zip(base_val, val_infos)]
    masked_targets = masked_val_df["target_iid"].astype(str).tolist()
    val_targets = val_df["target_iid"].astype(str).tolist()
    exact_bins = exact_bins_for_frame(masked_val_df)
    coarse_bins = [
        length_bin(int(x))
        for x in masked_val_df.get("_masked_raw_len", pd.Series([0] * len(masked_val_df))).astype(int).tolist()
    ]
    natural = compare_prediction_sets([x[:10] for x in base_val], val_candidate, val_targets, exact_bins_for_frame(val_df))
    masked = compare_prediction_sets([x[:10] for x in base_masked], masked_candidate, masked_targets, exact_bins)
    base_exact = weighted_exact_score({b: masked["by_exact_len"][b]["base"] for b in EXACT_LENGTH_BINS}, exact_weights)
    cand_exact = weighted_exact_score({b: masked["by_exact_len"][b]["candidate"] for b in EXACT_LENGTH_BINS}, exact_weights)
    base_coarse_by_bin = score_by_named_bins([x[:10] for x in base_masked], masked_targets, coarse_bins)
    cand_coarse_by_bin = score_by_named_bins(masked_candidate, masked_targets, coarse_bins)
    base_coarse = weighted_bin_score(base_coarse_by_bin, coarse_weights)
    cand_coarse = weighted_bin_score(cand_coarse_by_bin, coarse_weights)
    return {
        "seed": int(seed),
        "candidate": variant["name"],
        "natural": natural,
        "masked": masked,
        "masked_exact_weighted_base": base_exact,
        "masked_exact_weighted_candidate": cand_exact,
        "masked_exact_weighted_delta": cand_exact - base_exact,
        "masked_coarse_by_bin_base": base_coarse_by_bin,
        "masked_coarse_by_bin_candidate": cand_coarse_by_bin,
        "masked_coarse_weighted_base": base_coarse,
        "masked_coarse_weighted_candidate": cand_coarse,
        "masked_coarse_weighted_delta": cand_coarse - base_coarse,
        "changed_by_coarse_bin": changed_by_named_bins([x[:10] for x in base_masked], masked_candidate, coarse_bins),
    }


def summarize_by_candidate(results: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    payload: List[Dict[str, Any]] = []
    for name, split_results in results.items():
        exact_summary = summarize_pairwise_results(split_results)
        coarse_deltas = [float(r["masked_coarse_weighted_delta"]) for r in split_results]
        row = {
            "candidate": name,
            "summary": exact_summary,
            "mean_masked_coarse_weighted_delta": float(np.mean(coarse_deltas)) if coarse_deltas else 0.0,
            "min_masked_coarse_weighted_delta": float(np.min(coarse_deltas)) if coarse_deltas else 0.0,
            "max_masked_coarse_weighted_delta": float(np.max(coarse_deltas)) if coarse_deltas else 0.0,
            "split_masked_coarse_weighted_deltas": coarse_deltas,
            "splits": split_results,
        }
        payload.append(row)
    payload.sort(
        key=lambda x: (
            float(x["summary"]["mean_masked_exact_weighted_delta"]),
            float(x["summary"]["min_masked_exact_weighted_delta"]),
            float(x["mean_masked_coarse_weighted_delta"]),
        ),
        reverse=True,
    )
    return payload


def main() -> None:
    args = parse_args()
    train_df, test_df, user_df, item_df, candidates = load_data(args.rec_data)
    variants = variant_grid()
    exact_weights = test_exact_len_weights(test_df)
    coarse_weights = test_bin_weights(test_df)
    all_results: Dict[str, List[Dict[str, Any]]] = {v["name"]: [] for v in variants}
    max_pool = max(int(args.max_base_pool), max(int(v["base_pool"]) for v in variants))

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
        for variant in variants:
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
                coarse_weights,
            )
            all_results[variant["name"]].append(result)
        partial = summarize_by_candidate(all_results)
        top = partial[0]
        print(
            f"seed={seed} top={top['candidate']} "
            f"mean_exact={top['summary']['mean_masked_exact_weighted_delta']:.6f} "
            f"min_exact={top['summary']['min_masked_exact_weighted_delta']:.6f} "
            f"mean_coarse={top['mean_masked_coarse_weighted_delta']:.6f}",
            flush=True,
        )

    results = summarize_by_candidate(all_results)
    payload = {
        "rec_data": str(args.rec_data),
        "seeds": list(args.seeds),
        "val_ratio": float(args.val_ratio),
        "exact_weights": exact_weights,
        "coarse_weights": coarse_weights,
        "variants": variants,
        "results": results,
    }
    ensure_dir(args.output.parent)
    write_json(args.output, payload)
    for row in results[:12]:
        summary = row["summary"]
        print(
            f"{row['candidate']} mean_exact={summary['mean_masked_exact_weighted_delta']:.6f} "
            f"min_exact={summary['min_masked_exact_weighted_delta']:.6f} "
            f"mean_coarse={row['mean_masked_coarse_weighted_delta']:.6f} "
            f"min_coarse={row['min_masked_coarse_weighted_delta']:.6f}",
            flush=True,
        )
    print(f"wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
