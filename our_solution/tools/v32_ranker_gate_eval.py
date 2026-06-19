#!/usr/bin/env python3
"""Evaluate conservative v32 LightGBM ranker gates against v29."""

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
    v32_ranker_gate_blend_config,
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
    parser = argparse.ArgumentParser(description="Evaluate v32 ranker gate/blend probes versus v29.")
    parser.add_argument("--rec_data", type=Path, default=PROJECT_ROOT / "A推荐")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--val_ratio", type=float, default=0.12)
    parser.add_argument("--profile", choices=["quick", "full"], default="quick")
    parser.add_argument("--only", nargs="*", default=None, help="Optional variant names to evaluate.")
    parser.add_argument(
        "--output",
        type=Path,
        default=SOLUTION_DIR / "output" / "v32_ranker_gate_eval.json",
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


def variant_configs(profile: str) -> List[Dict[str, Any]]:
    gt80 = v32_ranker_gate_blend_config(
        name="v32_gate_f5_i6_l8_ins2_gt80",
        freeze_top_n=5,
        insert_start_rank=6,
        base_score_weight=8.0,
        max_insertions=2,
        lgbm_rank_top_n=12,
    )
    gt80["prediction_exact_bins"] = (">80",)
    l1120_gt80 = v32_ranker_gate_blend_config(
        name="v32_gate_f5_i6_l8_ins2_1120_gt80",
        freeze_top_n=5,
        insert_start_rank=6,
        base_score_weight=8.0,
        max_insertions=2,
        lgbm_rank_top_n=12,
    )
    l1120_gt80["prediction_exact_bins"] = ("11-20", ">80")

    configs: List[Dict[str, Any]] = [
        v32_ranker_gate_blend_config(
            name="v32_gate_f7_i8_l8_ins1",
            freeze_top_n=7,
            insert_start_rank=8,
            base_score_weight=8.0,
            max_insertions=1,
            lgbm_rank_top_n=10,
        ),
        v32_ranker_gate_blend_config(
            name="v32_gate_f5_i6_l8_ins1",
            freeze_top_n=5,
            insert_start_rank=6,
            base_score_weight=8.0,
            max_insertions=1,
            lgbm_rank_top_n=12,
        ),
        v32_ranker_gate_blend_config(
            name="v32_gate_f5_i6_l8_ins2",
            freeze_top_n=5,
            insert_start_rank=6,
            base_score_weight=8.0,
            max_insertions=2,
            lgbm_rank_top_n=12,
        ),
        v32_ranker_gate_blend_config(
            name="v32_gate_f5_i6_l4_ins2",
            freeze_top_n=5,
            insert_start_rank=6,
            base_score_weight=4.0,
            max_insertions=2,
            lgbm_rank_top_n=15,
        ),
        gt80,
        l1120_gt80,
    ]
    if profile == "full":
        gt80_l4_ins3 = v32_ranker_gate_blend_config(
            name="v32_gate_f5_i6_l4_ins3_gt80",
            freeze_top_n=5,
            insert_start_rank=6,
            base_score_weight=4.0,
            max_insertions=3,
            lgbm_rank_top_n=20,
        )
        gt80_l4_ins3["prediction_exact_bins"] = (">80",)
        gt80_f3 = v32_ranker_gate_blend_config(
            name="v32_gate_f3_i4_l8_ins2_gt80",
            freeze_top_n=3,
            insert_start_rank=4,
            base_score_weight=8.0,
            max_insertions=2,
            lgbm_rank_top_n=12,
        )
        gt80_f3["prediction_exact_bins"] = (">80",)
        gt80_src1 = v32_ranker_gate_blend_config(
            name="v32_gate_f5_i6_l8_ins2_gt80_src1",
            freeze_top_n=5,
            insert_start_rank=6,
            base_score_weight=8.0,
            max_insertions=2,
            min_source_count=1,
            lgbm_rank_top_n=20,
        )
        gt80_src1["prediction_exact_bins"] = (">80",)
        gt80_reorder = v32_ranker_gate_blend_config(
            name="v32_gate_f5_i6_l4_ins2_gt80_reorder",
            freeze_top_n=5,
            insert_start_rank=6,
            base_score_weight=4.0,
            max_insertions=2,
            lgbm_rank_top_n=20,
            allow_base_tail_reorder=True,
        )
        gt80_reorder["prediction_exact_bins"] = (">80",)
        configs.extend(
            [
                v32_ranker_gate_blend_config(
                    name="v32_gate_f3_i4_l8_ins1",
                    freeze_top_n=3,
                    insert_start_rank=4,
                    base_score_weight=8.0,
                    max_insertions=1,
                    lgbm_rank_top_n=10,
                ),
                v32_ranker_gate_blend_config(
                    name="v32_gate_f5_i6_l16_ins1",
                    freeze_top_n=5,
                    insert_start_rank=6,
                    base_score_weight=16.0,
                    max_insertions=1,
                    lgbm_rank_top_n=10,
                ),
                v32_ranker_gate_blend_config(
                    name="v32_gate_f5_i6_l8_ins1_m02",
                    freeze_top_n=5,
                    insert_start_rank=6,
                    base_score_weight=8.0,
                    max_insertions=1,
                    lgbm_rank_top_n=10,
                    min_combined_margin=0.2,
                ),
                gt80_l4_ins3,
                gt80_f3,
                gt80_src1,
                gt80_reorder,
            ]
        )
    return configs


def evaluate_variant(
    model: Any,
    base_model: Any,
    config: Dict[str, Any],
    masked_val_df: pd.DataFrame,
    fit_df: pd.DataFrame,
    exact_weights: Dict[str, float],
) -> Dict[str, Any]:
    model.update_prediction_settings(config)
    masked_targets = masked_val_df["target_iid"].astype(str).tolist()
    masked_base10: List[List[str]] = []
    masked_candidate: List[List[str]] = []
    for _, row in masked_val_df.iterrows():
        masked_base10.append(base_model.predict_row(row, k=10))
        masked_candidate.append(model.predict_row(row, k=10))

    exact_bins = exact_bins_for_frame(masked_val_df)
    masked = compare_prediction_sets(masked_base10, masked_candidate, masked_targets, exact_bins)
    repeat_comp = compare_prediction_sets_by_bins(
        masked_base10,
        masked_candidate,
        masked_targets,
        repeat_concentration_bins_for_frame(masked_val_df),
        REPEAT_CONCENTRATION_BINS,
    )
    pop_comp = compare_prediction_sets_by_bins(
        masked_base10,
        masked_candidate,
        masked_targets,
        target_popularity_bins(masked_targets, fit_df),
        POPULARITY_BINS,
    )
    base_weighted = weighted_from_compare(masked, "base", exact_weights)
    candidate_weighted = weighted_from_compare(masked, "candidate", exact_weights)
    return {
        "candidate": config["name"],
        "config": config,
        "masked": masked,
        "masked_by_repeat_concentration": repeat_comp,
        "masked_by_target_popularity": pop_comp,
        "masked_exact_weighted_base": base_weighted,
        "masked_exact_weighted_candidate": candidate_weighted,
        "masked_exact_weighted_delta": candidate_weighted - base_weighted,
    }


def evaluate_seed(
    seed: int,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    user_df: pd.DataFrame,
    item_df: pd.DataFrame,
    candidates: List[str],
    val_ratio: float,
    configs: List[Dict[str, Any]],
) -> Dict[str, Any]:
    set_seed(seed)
    fit_df, _, masked_val_df, _ = test_like_recommendation_split(
        train_df,
        test_df,
        val_ratio=val_ratio,
        seed=seed,
    )
    exact_weights = test_exact_len_weights(test_df)
    train_config = v31_covisit_lgbm_ranker_config()
    _, ranker_label_df = recommendation_split(fit_df, val_ratio=val_ratio, seed=seed + 700)
    ranker_train_df = masked_history_eval(ranker_label_df, test_df, seed=seed + 1700)
    ranker_train_df["target_iid"] = ranker_label_df["target_iid"].astype(str).values
    model = build_recommender(train_config, candidates, user_df, item_df)
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
    for _, row in masked_val_df.iterrows():
        masked_base80.append(base_model.predict_row(row, k=80))
        masked_pool80.append(model.candidate_pool_for_row(row, k=80))
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

    variants = []
    for config in configs:
        result = evaluate_variant(model, base_model, config, masked_val_df, fit_df, exact_weights)
        result["seed"] = int(seed)
        variants.append(result)
        changed = sum(v["changed_rows"] for v in result["masked"]["by_exact_len"].values())
        print(
            f"seed={seed} {config['name']} delta={result['masked_exact_weighted_delta']:.9f} "
            f"changed={changed}",
            flush=True,
        )
    return {
        "seed": int(seed),
        "recall": recall,
        "test_exact_len_weights": exact_weights,
        "variants": variants,
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
        "mean_delta_gt_0": float(summary.get("mean_masked_exact_weighted_delta", 0.0)) > 0.0,
        "min_delta_ge_-0.0002": float(summary.get("min_masked_exact_weighted_delta", 0.0)) >= -0.0002,
        "has_positive_split": any(delta > 0.0 for delta in deltas),
        "has_stable_positive_segment": stable_segment,
    }
    return {"pass": bool(all(checks.values())), "checks": checks}


def main() -> None:
    args = parse_args()
    configs = variant_configs(args.profile)
    if args.only:
        wanted = set(args.only)
        configs = [config for config in configs if config["name"] in wanted]
        missing = sorted(wanted.difference(config["name"] for config in configs))
        if missing:
            raise ValueError(f"Unknown variants requested by --only: {missing}")
    train_df, test_df, user_df, item_df, candidates = load_data(args.rec_data)
    seed_results = [
        evaluate_seed(seed, train_df, test_df, user_df, item_df, candidates, args.val_ratio, configs)
        for seed in args.seeds
    ]

    by_name: Dict[str, List[Dict[str, Any]]] = {config["name"]: [] for config in configs}
    for seed_result in seed_results:
        for result in seed_result["variants"]:
            by_name[result["candidate"]].append(result)

    summaries: Dict[str, Any] = {}
    for config in configs:
        name = config["name"]
        summary = summarize_pairwise_results(by_name[name])
        summary["guard"] = guard(summary)
        summaries[name] = summary

    ranked = sorted(
        summaries.items(),
        key=lambda item: (
            -float(item[1]["mean_masked_exact_weighted_delta"]),
            -float(item[1]["min_masked_exact_weighted_delta"]),
        ),
    )
    print("ranked_variants", flush=True)
    for name, summary in ranked:
        print(
            f"{name} mean={summary['mean_masked_exact_weighted_delta']:.9f} "
            f"min={summary['min_masked_exact_weighted_delta']:.9f} "
            f"splits={summary['split_deltas']} guard={summary['guard']['pass']}",
            flush=True,
        )

    recall_summary: Dict[str, float] = {}
    for key in seed_results[0]["recall"]:
        recall_summary[key] = float(np.mean([r["recall"][key] for r in seed_results]))

    payload = {
        "rec_data": str(args.rec_data),
        "seeds": list(args.seeds),
        "val_ratio": float(args.val_ratio),
        "profile": args.profile,
        "baseline": v29_medium_long_count_only_pool25_config(),
        "train_config": v31_covisit_lgbm_ranker_config(),
        "variants": configs,
        "recall_summary": recall_summary,
        "summaries": summaries,
        "ranked": [
            {"name": name, **summary}
            for name, summary in ranked
        ],
        "splits": seed_results,
    }
    ensure_dir(args.output.parent)
    write_json(args.output, payload)
    print(f"wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
