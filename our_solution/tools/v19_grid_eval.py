#!/usr/bin/env python3
"""Fast v19 grid probe: fit v17+counts once per split, then sweep conservative rerank knobs."""

from __future__ import annotations

import argparse
import copy
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
    EXACT_LENGTH_BINS,
    build_recommender,
    test_exact_len_weights,
    test_like_recommendation_split,
    v19a_short_len3_only_alpha04_config,
    weighted_exact_score,
)
from src.validation import (  # noqa: E402
    compare_prediction_sets,
    exact_bins_for_frame,
    summarize_pairwise_results,
    v19_guard,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep conservative v19 short-history rerank variants.")
    parser.add_argument("--rec_data", type=Path, default=PROJECT_ROOT / "A推荐")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--val_ratio", type=float, default=0.12)
    parser.add_argument(
        "--output",
        type=Path,
        default=SOLUTION_DIR / "output" / "v19_grid_eval_3seed.json",
    )
    return parser.parse_args()


def variants() -> List[Dict[str, Any]]:
    base_components = {
        "last": 0.8,
        "suffix": 0.35,
        "last_group": 1.0,
        "suffix_group": 0.75,
    }
    return [
        {
            "name": "v19_len3_ultra_conservative_a005",
            "target_lengths": (3,),
            "alpha_by_len": {3: 0.005},
            "min_count": 20,
            "min_group_count": 35,
            "min_lift": 1.30,
            "shrink_beta": 90.0,
            "component_weights": base_components,
        },
        {
            "name": "v19_len3_strict_group_a01",
            "target_lengths": (3,),
            "alpha_by_len": {3: 0.010},
            "min_count": 18,
            "min_group_count": 30,
            "min_lift": 1.25,
            "shrink_beta": 80.0,
            "component_weights": {
                "last": 0.0,
                "suffix": 0.0,
                "last_group": 1.0,
                "suffix_group": 0.8,
            },
        },
        {
            "name": "v19_len3_strict_a015",
            "target_lengths": (3,),
            "alpha_by_len": {3: 0.015},
            "min_count": 15,
            "min_group_count": 25,
            "min_lift": 1.20,
            "shrink_beta": 75.0,
            "component_weights": base_components,
        },
        {
            "name": "v19_len1_only_group_a02",
            "target_lengths": (1,),
            "alpha_by_len": {1: 0.020},
            "min_count": 18,
            "min_group_count": 30,
            "min_lift": 1.25,
            "shrink_beta": 80.0,
            "component_weights": {
                "last": 0.0,
                "suffix": 0.0,
                "last_group": 1.0,
                "suffix_group": 0.0,
            },
        },
        {
            "name": "v19_len1_3_strict_combo",
            "target_lengths": (1, 3),
            "alpha_by_len": {1: 0.015, 3: 0.010},
            "min_count": 18,
            "min_group_count": 30,
            "min_lift": 1.25,
            "shrink_beta": 85.0,
            "component_weights": {
                "last": 0.0,
                "suffix": 0.0,
                "last_group": 1.0,
                "suffix_group": 0.8,
            },
        },
    ]


def load_data(rec_data: Path):
    train_df = pd.read_csv(rec_data / "train.csv").astype({"uid": str, "target_iid": str})
    test_df = pd.read_csv(rec_data / "test.csv").astype({"uid": str})
    user_df = pd.read_csv(rec_data / "user.csv").astype({"uid": str})
    item_df = pd.read_csv(rec_data / "item.csv").astype({"iid": str})
    candidates = item_df["iid"].astype(str).tolist()
    return train_df, test_df, user_df, item_df, candidates


def apply_variant(model: Any, variant: Dict[str, Any]) -> None:
    model.target_lengths = {int(x) for x in variant["target_lengths"]}
    model.alpha_by_len = {int(k): float(v) for k, v in variant["alpha_by_len"].items()}
    model.min_count = int(variant["min_count"])
    model.min_group_count = int(variant["min_group_count"])
    model.min_lift = float(variant["min_lift"])
    model.shrink_beta = float(variant["shrink_beta"])
    model.component_weights = dict(variant["component_weights"])


def evaluate_predictions(
    name: str,
    seed: int,
    model: Any,
    base_val: List[List[str]],
    base_masked: List[List[str]],
    val_df: pd.DataFrame,
    masked_val_df: pd.DataFrame,
    test_weights: Dict[str, float],
) -> Dict[str, Any]:
    val_candidate = [model.predict_row(row, k=10) for _, row in val_df.iterrows()]
    masked_candidate = [model.predict_row(row, k=10) for _, row in masked_val_df.iterrows()]
    val_targets = val_df["target_iid"].astype(str).tolist()
    masked_targets = masked_val_df["target_iid"].astype(str).tolist()
    natural = compare_prediction_sets(base_val, val_candidate, val_targets, exact_bins_for_frame(val_df))
    masked = compare_prediction_sets(base_masked, masked_candidate, masked_targets, exact_bins_for_frame(masked_val_df))
    base_weighted = weighted_exact_score(
        {b: masked["by_exact_len"][b]["base"] for b in EXACT_LENGTH_BINS},
        test_weights,
    )
    candidate_weighted = weighted_exact_score(
        {b: masked["by_exact_len"][b]["candidate"] for b in EXACT_LENGTH_BINS},
        test_weights,
    )
    return {
        "seed": int(seed),
        "candidate": name,
        "natural": natural,
        "masked": masked,
        "masked_exact_weighted_base": base_weighted,
        "masked_exact_weighted_candidate": candidate_weighted,
        "masked_exact_weighted_delta": candidate_weighted - base_weighted,
    }


def main() -> None:
    args = parse_args()
    train_df, test_df, user_df, item_df, candidates = load_data(args.rec_data)
    test_weights = test_exact_len_weights(test_df)
    all_results: Dict[str, List[Dict[str, Any]]] = {v["name"]: [] for v in variants()}
    for seed in args.seeds:
        set_seed(seed)
        fit_df, val_df, masked_val_df, _ = test_like_recommendation_split(
            train_df, test_df, val_ratio=args.val_ratio, seed=seed
        )
        model = build_recommender(
            v19a_short_len3_only_alpha04_config(), candidates, user_df, item_df
        ).fit(fit_df)
        base_model = model.base_model
        base_val = [base_model.predict_row(row, k=10) for _, row in val_df.iterrows()]
        base_masked = [base_model.predict_row(row, k=10) for _, row in masked_val_df.iterrows()]
        for variant in variants():
            apply_variant(model, copy.deepcopy(variant))
            result = evaluate_predictions(
                variant["name"], seed, model, base_val, base_masked, val_df, masked_val_df, test_weights
            )
            all_results[variant["name"]].append(result)
            print(
                f"{variant['name']} seed={seed} delta={result['masked_exact_weighted_delta']:.6f} "
                f"natural={result['natural']['delta_ndcg@10']:.6f}",
                flush=True,
            )
    payload = {
        "rec_data": str(args.rec_data),
        "seeds": list(args.seeds),
        "results": [],
    }
    for name, split_results in all_results.items():
        summary = summarize_pairwise_results(split_results)
        payload["results"].append(
            {
                "candidate": name,
                "summary": summary,
                "guard": v19_guard(summary),
                "splits": split_results,
            }
        )
        print(
            f"{name} mean={summary['mean_masked_exact_weighted_delta']:.6f} "
            f"min={summary['min_masked_exact_weighted_delta']:.6f} "
            f"guard={v19_guard(summary)['pass']}",
            flush=True,
        )
    ensure_dir(args.output.parent)
    write_json(args.output, payload)
    print(f"wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
