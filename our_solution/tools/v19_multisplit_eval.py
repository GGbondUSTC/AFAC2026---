#!/usr/bin/env python3
"""Run guarded multi-split v19 recommendation probes without archiving submissions."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

SOLUTION_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = SOLUTION_DIR.parent
if str(SOLUTION_DIR) not in sys.path:
    sys.path.insert(0, str(SOLUTION_DIR))

from src.common import ensure_dir, set_seed, write_json  # noqa: E402
from src.recommendation import (  # noqa: E402
    test_like_recommendation_split,
    v17_conservative_shortseq_config,
    v19a_short_len3_only_alpha04_config,
    v19b_short_len1_3_alpha035_050_config,
)
from src.validation import evaluate_pairwise_recommender, summarize_pairwise_results, v19_guard  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate v19 short-history rerank probes over multiple splits.")
    parser.add_argument("--rec_data", type=Path, default=PROJECT_ROOT / "A推荐", help="Recommendation data directory.")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44], help="Validation split seeds.")
    parser.add_argument(
        "--candidate",
        type=str,
        default="v19a_short_len3_only_alpha04",
        choices=["v19a_short_len3_only_alpha04", "v19b_short_len1_3_alpha035_050", "all"],
        help="Candidate config to evaluate.",
    )
    parser.add_argument("--val_ratio", type=float, default=0.12, help="Validation ratio.")
    parser.add_argument("--output", type=Path, default=None, help="Output JSON path.")
    return parser.parse_args()


def candidate_map() -> Dict[str, Dict[str, Any]]:
    return {
        "v19a_short_len3_only_alpha04": v19a_short_len3_only_alpha04_config(),
        "v19b_short_len1_3_alpha035_050": v19b_short_len1_3_alpha035_050_config(),
    }


def load_data(rec_data: Path):
    train_df = pd.read_csv(rec_data / "train.csv").astype({"uid": str, "target_iid": str})
    test_df = pd.read_csv(rec_data / "test.csv").astype({"uid": str})
    user_df = pd.read_csv(rec_data / "user.csv").astype({"uid": str})
    item_df = pd.read_csv(rec_data / "item.csv").astype({"iid": str})
    candidates = item_df["iid"].astype(str).tolist()
    return train_df, test_df, user_df, item_df, candidates


def evaluate_candidate(args: argparse.Namespace, candidate_name: str, candidate_config: Dict[str, Any]) -> Dict[str, Any]:
    train_df, test_df, user_df, item_df, candidates = load_data(args.rec_data)
    baseline_config = v17_conservative_shortseq_config()
    split_results: List[Dict[str, Any]] = []
    for seed in args.seeds:
        set_seed(seed)
        fit_df, val_df, masked_val_df, _ = test_like_recommendation_split(
            train_df, test_df, val_ratio=args.val_ratio, seed=seed
        )
        result = evaluate_pairwise_recommender(
            fit_df=fit_df,
            val_df=val_df,
            masked_val_df=masked_val_df,
            test_df=test_df,
            user_df=user_df,
            item_df=item_df,
            candidates=candidates,
            candidate_config=candidate_config,
            baseline_config=baseline_config,
            seed=seed,
        )
        split_results.append(result)
        print(
            f"{candidate_name} seed={seed} "
            f"masked_exact_delta={result['masked_exact_weighted_delta']:.6f} "
            f"natural_delta={result['natural']['delta_ndcg@10']:.6f}",
            flush=True,
        )
    summary = summarize_pairwise_results(split_results)
    guard = v19_guard(summary)
    return {
        "candidate": candidate_name,
        "seeds": list(args.seeds),
        "val_ratio": float(args.val_ratio),
        "summary": summary,
        "guard": guard,
        "splits": split_results,
    }


def main() -> None:
    args = parse_args()
    configs = candidate_map()
    selected = list(configs.keys()) if args.candidate == "all" else [args.candidate]
    payload = {
        "rec_data": str(args.rec_data),
        "results": [evaluate_candidate(args, name, configs[name]) for name in selected],
    }
    if args.output is None:
        suffix = args.candidate if args.candidate != "all" else "all"
        args.output = SOLUTION_DIR / "output" / f"v19_multisplit_eval_{suffix}.json"
    ensure_dir(args.output.parent)
    write_json(args.output, payload)
    for result in payload["results"]:
        summary = result["summary"]
        guard = result["guard"]
        print(
            f"{result['candidate']} mean_delta={summary['mean_masked_exact_weighted_delta']:.6f} "
            f"min_delta={summary['min_masked_exact_weighted_delta']:.6f} guard_pass={guard['pass']}",
            flush=True,
        )
    print(f"wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
