#!/usr/bin/env python3
"""Evaluate v34 precise gated blend probes after the v33 online result."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

SOLUTION_DIR = Path(__file__).resolve().parents[1]
TOOLS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SOLUTION_DIR.parent
for path in (SOLUTION_DIR, TOOLS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from src.common import ensure_dir, write_json  # noqa: E402
from src.recommendation import v34_precise_blend_gate_gt80_config  # noqa: E402
from v33_aggressive_ranker_eval import (  # noqa: E402
    evaluate_seed,
    load_data,
    summarize_variant_results,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate v34 precise gated blend probes versus v32.")
    parser.add_argument("--rec_data", type=Path, default=PROJECT_ROOT / "A推荐")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--val_ratio", type=float, default=0.12)
    parser.add_argument("--profile", choices=["quick", "full"], default="quick")
    parser.add_argument("--only", nargs="*", default=None, help="Optional variant names to evaluate.")
    parser.add_argument(
        "--output",
        type=Path,
        default=SOLUTION_DIR / "output" / "v34_precise_blend_eval.json",
    )
    return parser.parse_args()


def variant_configs(profile: str) -> List[Dict[str, Any]]:
    base_grid = [
        (-0.20, 0.00, 7, 4, False),
        (-0.10, 0.05, 7, 3, False),
        (-0.05, 0.10, 6, 3, False),
        (-0.05, 0.20, 5, 2, False),
        (0.00, 0.00, 4, 2, False),
        (0.02, 0.05, 4, 1, False),
        (0.05, 0.10, 3, 1, False),
        (0.08, 0.16, 3, 1, False),
        (0.10, 0.20, 2, 1, False),
        (0.05, 0.15, 3, 0, False),
        (0.10, 0.25, 2, 0, False),
        (0.02, 0.05, 4, 1, True),
        (0.05, 0.10, 3, 1, True),
    ]
    if profile == "full":
        base_grid.extend(
            [
                (0.12, 0.25, 2, 1, False),
                (0.15, 0.35, 2, 1, False),
                (0.20, 0.45, 2, 1, False),
                (0.02, 0.10, 3, 0, False),
                (0.00, 0.05, 4, 0, False),
                (0.08, 0.16, 3, 1, True),
                (0.10, 0.20, 2, 1, True),
            ]
        )
    configs: List[Dict[str, Any]] = []
    for min_position_margin, min_total_margin, max_changed, max_new, require_source_gate in base_grid:
        configs.append(
            v34_precise_blend_gate_gt80_config(
                lambda_weight=8.0,
                min_position_margin=min_position_margin,
                min_total_margin=min_total_margin,
                max_changed_positions=max_changed,
                max_new_items=max_new,
                require_new_item_gate=require_source_gate,
                min_source_count=2 if require_source_gate else 0,
                lgbm_rank_top_n=30 if require_source_gate else 120,
            )
        )
    return configs


def main() -> None:
    args = parse_args()
    configs = variant_configs(args.profile)
    if args.only:
        wanted = set(args.only)
        configs = [config for config in configs if str(config["name"]) in wanted]
        missing = sorted(wanted.difference(str(config["name"]) for config in configs))
        if missing:
            raise ValueError(f"Unknown variants requested by --only: {missing}")
    if not configs:
        raise ValueError("No v34 variants selected.")

    train_df, test_df, user_df, item_df, candidates = load_data(args.rec_data)
    seed_results = [
        evaluate_seed(seed, train_df, test_df, user_df, item_df, candidates, args.val_ratio, configs)
        for seed in args.seeds
    ]

    by_name: Dict[str, List[Dict[str, Any]]] = {str(config["name"]): [] for config in configs}
    for seed_result in seed_results:
        for result in seed_result["variants"]:
            by_name[str(result["candidate"])].append(result)

    summaries = {
        str(config["name"]): summarize_variant_results(by_name[str(config["name"])], test_rows=len(test_df))
        for config in configs
    }
    ranked = sorted(
        summaries.items(),
        key=lambda item: (
            -float(item[1]["mean_delta_vs_v32"]),
            -float(item[1]["min_delta_vs_v32"]),
            abs(float(item[1]["mean_estimated_test_changed_rows_vs_v32"]) - 250.0),
        ),
    )
    print("ranked_variants", flush=True)
    for name, summary in ranked:
        print(
            f"{name} mean_vs_v32={summary['mean_delta_vs_v32']:.9f} "
            f"min_vs_v32={summary['min_delta_vs_v32']:.9f} "
            f"changed_est={summary['mean_estimated_test_changed_rows_vs_v32']:.1f} "
            f"top1={summary['max_top1_changed_vs_v32']:.0f} "
            f"top3={summary['max_top3_changed_vs_v32']:.0f} "
            f"guard={summary['guard']['pass']} "
            f"splits={summary['split_deltas_vs_v32']}",
            flush=True,
        )

    payload = {
        "rec_data": str(args.rec_data),
        "seeds": [int(seed) for seed in args.seeds],
        "val_ratio": float(args.val_ratio),
        "profile": args.profile,
        "intent": "Reduce v33's 530 changed test rows while keeping top3 unchanged.",
        "variants": configs,
        "summaries": summaries,
        "ranked": [{"name": name, **summary} for name, summary in ranked],
        "splits": seed_results,
    }
    ensure_dir(args.output.parent)
    write_json(args.output, payload)
    print(json.dumps({"best": ranked[0][0], "guard": ranked[0][1]["guard"]}, ensure_ascii=False), flush=True)
    print(f"wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
