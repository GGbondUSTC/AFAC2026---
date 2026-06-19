#!/usr/bin/env python3
"""Evaluate the v31 multi-source recall + LightGBM ranker against v29."""

from __future__ import annotations

import argparse
import json
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
    masked_history_eval,
    recommendation_split,
    test_exact_len_weights,
    test_like_recommendation_split,
    v29_medium_long_count_only_pool25_config,
    v31_covisit_lgbm_ranker_config,
    weighted_exact_score,
)
from src.validation import (  # noqa: E402
    REPEAT_CONCENTRATION_BINS,
    compare_prediction_sets,
    compare_prediction_sets_by_bins,
    exact_bins_for_frame,
    repeat_concentration_bins_for_frame,
    summarize_pairwise_results,
)


POPULARITY_BINS = ("unknown", "tail", "mid", "head")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate v31 recall and ranker guard versus v29.")
    parser.add_argument("--rec_data", type=Path, default=PROJECT_ROOT / "A推荐")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    parser.add_argument("--val_ratio", type=float, default=0.12)
    parser.add_argument(
        "--output",
        type=Path,
        default=SOLUTION_DIR / "output" / "v31_recall_ranker_eval.json",
    )
    return parser.parse_args()


def load_data(rec_data: Path):
    train_df = pd.read_csv(rec_data / "train.csv").astype({"uid": str, "target_iid": str})
    test_df = pd.read_csv(rec_data / "test.csv").astype({"uid": str})
    user_df = pd.read_csv(rec_data / "user.csv").astype({"uid": str})
    item_df = pd.read_csv(rec_data / "item.csv").astype({"iid": str})
    candidates = item_df["iid"].astype(str).tolist()
    return train_df, test_df, user_df, item_df, candidates


def recall_at(predictions: Sequence[Sequence[str]], targets: Sequence[str], k: int) -> float:
    hits = 0
    for pred, target in zip(predictions, targets):
        if str(target) in set(str(x) for x in pred[:k]):
            hits += 1
    return float(hits / max(1, len(targets)))


def target_popularity_bins(targets: Sequence[str], fit_df: pd.DataFrame) -> List[str]:
    counts = Counter(fit_df["target_iid"].astype(str))
    positive_counts = np.asarray([v for v in counts.values() if v > 0], dtype=np.float64)
    if positive_counts.size == 0:
        return ["unknown"] * len(targets)
    q40 = float(np.quantile(positive_counts, 0.40))
    q80 = float(np.quantile(positive_counts, 0.80))
    bins: List[str] = []
    for target in targets:
        count = float(counts.get(str(target), 0))
        if count <= 0:
            bins.append("unknown")
        elif count >= q80:
            bins.append("head")
        elif count >= q40:
            bins.append("mid")
        else:
            bins.append("tail")
    return bins


def weighted_from_compare(compared: Dict[str, Any], side: str, exact_weights: Dict[str, float]) -> float:
    return weighted_exact_score(
        {b: float(compared["by_exact_len"][b][side]) for b in EXACT_LENGTH_BINS},
        exact_weights,
    )


def evaluate_seed(
    seed: int,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    user_df: pd.DataFrame,
    item_df: pd.DataFrame,
    candidates: List[str],
    val_ratio: float,
) -> Dict[str, Any]:
    set_seed(seed)
    fit_df, val_df, masked_val_df, _ = test_like_recommendation_split(
        train_df,
        test_df,
        val_ratio=val_ratio,
        seed=seed,
    )
    exact_weights = test_exact_len_weights(test_df)
    v31_config = v31_covisit_lgbm_ranker_config()
    _, ranker_label_df = recommendation_split(fit_df, val_ratio=val_ratio, seed=seed + 700)
    ranker_train_df = masked_history_eval(ranker_label_df, test_df, seed=seed + 1700)
    ranker_train_df["target_iid"] = ranker_label_df["target_iid"].astype(str).values
    model = build_recommender(v31_config, candidates, user_df, item_df)
    model.fit(fit_df, ranker_train_df=ranker_train_df)
    base_model = getattr(model, "base_model", None)
    if base_model is None:
        base_model = build_recommender(
            v29_medium_long_count_only_pool25_config(),
            candidates,
            user_df,
            item_df,
        ).fit(fit_df)

    masked_targets = masked_val_df["target_iid"].astype(str).tolist()
    masked_base80: List[List[str]] = []
    masked_pool80: List[List[str]] = []
    masked_v31: List[List[str]] = []
    for _, row in masked_val_df.iterrows():
        masked_base80.append(base_model.predict_row(row, k=80))
        masked_pool80.append(model.candidate_pool_for_row(row, k=80))
        masked_v31.append(model.predict_row(row, k=10))

    masked_base10 = [items[:10] for items in masked_base80]
    exact_bins = exact_bins_for_frame(masked_val_df)
    masked = compare_prediction_sets(masked_base10, masked_v31, masked_targets, exact_bins)
    repeat_comp = compare_prediction_sets_by_bins(
        masked_base10,
        masked_v31,
        masked_targets,
        repeat_concentration_bins_for_frame(masked_val_df),
        REPEAT_CONCENTRATION_BINS,
    )
    pop_comp = compare_prediction_sets_by_bins(
        masked_base10,
        masked_v31,
        masked_targets,
        target_popularity_bins(masked_targets, fit_df),
        POPULARITY_BINS,
    )
    base_weighted = weighted_from_compare(masked, "base", exact_weights)
    candidate_weighted = weighted_from_compare(masked, "candidate", exact_weights)

    recall = {
        f"v29_top{k}": recall_at(masked_base80, masked_targets, k)
        for k in (10, 25, 50, 80)
    }
    recall.update(
        {
            f"v31_pool_top{k}": recall_at(masked_pool80, masked_targets, k)
            for k in (10, 25, 50, 80)
        }
    )
    return {
        "seed": int(seed),
        "candidate": v31_config["name"],
        "baseline": "v29_medium_long_count_only_len21_pool25_alpha32",
        "recall": recall,
        "test_exact_len_weights": exact_weights,
        "masked": masked,
        "masked_by_repeat_concentration": repeat_comp,
        "masked_by_target_popularity": pop_comp,
        "masked_exact_weighted_base": base_weighted,
        "masked_exact_weighted_candidate": candidate_weighted,
        "masked_exact_weighted_delta": candidate_weighted - base_weighted,
    }


def guard(summary: Dict[str, Any]) -> Dict[str, Any]:
    deltas = [float(x) for x in summary.get("split_deltas", [])]
    by_exact = summary.get("masked_by_exact_len_summary", {})
    stable_segment = any(
        float(values.get("mean_delta", 0.0)) > 0.0
        and float(values.get("min_delta", 0.0)) >= 0.0
        and float(values.get("mean_changed_rows", 0.0)) > 0.0
        for values in by_exact.values()
    )
    checks = {
        "mean_delta_ge_0.0005": float(summary.get("mean_masked_exact_weighted_delta", 0.0)) >= 0.0005,
        "min_delta_ge_0": float(summary.get("min_masked_exact_weighted_delta", 0.0)) >= 0.0,
        "all_splits_positive": bool(deltas) and all(delta >= 0.0 for delta in deltas),
        "has_stable_positive_segment": stable_segment,
    }
    return {"pass": bool(all(checks.values())), "checks": checks}


def main() -> None:
    args = parse_args()
    train_df, test_df, user_df, item_df, candidates = load_data(args.rec_data)
    results: List[Dict[str, Any]] = []
    for seed in args.seeds:
        result = evaluate_seed(seed, train_df, test_df, user_df, item_df, candidates, args.val_ratio)
        results.append(result)
        print(
            f"seed={seed} delta={result['masked_exact_weighted_delta']:.9f} "
            f"v29_top80_recall={result['recall']['v29_top80']:.6f} "
            f"v31_pool80_recall={result['recall']['v31_pool_top80']:.6f}",
            flush=True,
        )

    summary = summarize_pairwise_results(results)
    recall_summary: Dict[str, float] = {}
    for key in results[0]["recall"]:
        recall_summary[key] = float(np.mean([r["recall"][key] for r in results]))
    summary["recall_summary"] = recall_summary
    summary["guard"] = guard(summary)
    payload = {
        "rec_data": str(args.rec_data),
        "seeds": list(args.seeds),
        "val_ratio": float(args.val_ratio),
        "candidate": v31_covisit_lgbm_ranker_config(),
        "baseline": v29_medium_long_count_only_pool25_config(),
        "summary": summary,
        "splits": results,
    }
    ensure_dir(args.output.parent)
    write_json(args.output, payload)
    print(json.dumps(summary["guard"], ensure_ascii=False), flush=True)
    print(
        "summary "
        f"mean={summary['mean_masked_exact_weighted_delta']:.9f} "
        f"min={summary['min_masked_exact_weighted_delta']:.9f} "
        f"splits={summary['split_deltas']}",
        flush=True,
    )
    print(f"wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
