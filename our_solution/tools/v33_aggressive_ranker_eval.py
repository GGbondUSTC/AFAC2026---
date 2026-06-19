#!/usr/bin/env python3
"""Evaluate v33 aggressive LightGBM ranker probes against v32 and v29."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

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
    v32_ranker_gate_gt80_config,
    v33a_gt80_ins4_pool120_config,
    v33b_gt80_top3_pool120_config,
    v33c_3180_strict_gt80_aggressive_config,
    v33d_pool200_tail_probe_config,
    v33e_segment_blend_gt80_config,
    v33f_uplift_label_gate_config,
    weighted_exact_score,
)
from src.validation import (  # noqa: E402
    REPEAT_CONCENTRATION_BINS,
    exact_bins_for_frame,
    gain_at_k,
    repeat_concentration_bins_for_frame,
)


POPULARITY_BINS = ("unknown", "tail", "mid", "head")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate v33 aggressive ranker probes versus v32.")
    parser.add_argument("--rec_data", type=Path, default=PROJECT_ROOT / "A推荐")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    parser.add_argument("--val_ratio", type=float, default=0.12)
    parser.add_argument("--profile", choices=["priority", "aggressive"], default="aggressive")
    parser.add_argument("--baseline", choices=["v32", "v29"], default="v32")
    parser.add_argument("--only", nargs="*", default=None, help="Optional variant names to evaluate.")
    parser.add_argument(
        "--output",
        type=Path,
        default=SOLUTION_DIR / "output" / "v33_aggressive_ranker_eval.json",
    )
    return parser.parse_args()


def load_data(rec_data: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, List[str]]:
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


def compare_predictions(
    base_predictions: Sequence[Sequence[str]],
    candidate_predictions: Sequence[Sequence[str]],
    targets: Sequence[str],
    bins: Sequence[str],
    bin_order: Sequence[str],
) -> Dict[str, Any]:
    base_scores: List[float] = []
    candidate_scores: List[float] = []
    rows: Dict[str, List[int]] = defaultdict(list)
    changed: Dict[str, int] = defaultdict(int)
    top1_changed: Dict[str, int] = defaultdict(int)
    top3_changed: Dict[str, int] = defaultdict(int)
    overlap: Dict[str, List[int]] = defaultdict(list)

    total_changed = 0
    total_top1_changed = 0
    total_top3_changed = 0
    total_overlap: List[int] = []
    for idx, (base, candidate, target, bin_name) in enumerate(
        zip(base_predictions, candidate_predictions, targets, bins)
    ):
        base_top = [str(x) for x in base[:10]]
        candidate_top = [str(x) for x in candidate[:10]]
        base_scores.append(gain_at_k(base_top, str(target), k=10))
        candidate_scores.append(gain_at_k(candidate_top, str(target), k=10))
        rows[str(bin_name)].append(idx)
        top_overlap = len(set(base_top).intersection(candidate_top))
        overlap[str(bin_name)].append(top_overlap)
        total_overlap.append(top_overlap)
        if base_top != candidate_top:
            changed[str(bin_name)] += 1
            total_changed += 1
        if (base_top[:1] or [""])[0] != (candidate_top[:1] or [""])[0]:
            top1_changed[str(bin_name)] += 1
            total_top1_changed += 1
        if base_top[:3] != candidate_top[:3]:
            top3_changed[str(bin_name)] += 1
            total_top3_changed += 1

    by_bin: Dict[str, Dict[str, float]] = {}
    for bin_name in bin_order:
        key = str(bin_name)
        idxs = rows.get(key, [])
        if not idxs:
            by_bin[key] = {
                "n": 0,
                "base": 0.0,
                "candidate": 0.0,
                "delta": 0.0,
                "changed_rows": 0,
                "top1_changed": 0,
                "top3_changed": 0,
                "top10_overlap_mean": 0.0,
            }
            continue
        base_mean = float(np.mean([base_scores[i] for i in idxs]))
        candidate_mean = float(np.mean([candidate_scores[i] for i in idxs]))
        by_bin[key] = {
            "n": int(len(idxs)),
            "base": base_mean,
            "candidate": candidate_mean,
            "delta": candidate_mean - base_mean,
            "changed_rows": int(changed.get(key, 0)),
            "top1_changed": int(top1_changed.get(key, 0)),
            "top3_changed": int(top3_changed.get(key, 0)),
            "top10_overlap_mean": float(np.mean(overlap.get(key, [0]))),
        }

    base_mean = float(np.mean(base_scores)) if base_scores else 0.0
    candidate_mean = float(np.mean(candidate_scores)) if candidate_scores else 0.0
    return {
        "base_ndcg@10": base_mean,
        "candidate_ndcg@10": candidate_mean,
        "delta_ndcg@10": candidate_mean - base_mean,
        "total": {
            "n": int(len(targets)),
            "base": base_mean,
            "candidate": candidate_mean,
            "delta": candidate_mean - base_mean,
            "changed_rows": int(total_changed),
            "top1_changed": int(total_top1_changed),
            "top3_changed": int(total_top3_changed),
            "top10_overlap_mean": float(np.mean(total_overlap)) if total_overlap else 0.0,
        },
        "by_bin": by_bin,
        "row_deltas": [float(c - b) for b, c in zip(base_scores, candidate_scores)],
    }


def weighted_from_compare(compared: Dict[str, Any], side: str, exact_weights: Dict[str, float]) -> float:
    return weighted_exact_score(
        {b: float(compared["by_bin"][b][side]) for b in EXACT_LENGTH_BINS},
        exact_weights,
    )


def variant_configs(profile: str) -> List[Dict[str, Any]]:
    configs: List[Dict[str, Any]] = [
        v33a_gt80_ins4_pool120_config(),
        v33b_gt80_top3_pool120_config(),
        v33c_3180_strict_gt80_aggressive_config(),
        v33e_segment_blend_gt80_config(lambda_weight=4.0),
        v33e_segment_blend_gt80_config(lambda_weight=8.0),
        v33e_segment_blend_gt80_config(lambda_weight=16.0),
        v33f_uplift_label_gate_config(),
    ]
    if profile == "aggressive":
        configs.insert(3, v33d_pool200_tail_probe_config())
    return configs


def train_signature(config: Dict[str, Any]) -> Tuple[int, str]:
    return (
        int(config.get("candidate_pool_size", 80)),
        str(config.get("ranker_label_mode", "target")),
    )


def grouped_configs(configs: Sequence[Dict[str, Any]]) -> Dict[Tuple[int, str], List[Dict[str, Any]]]:
    groups: Dict[Tuple[int, str], List[Dict[str, Any]]] = defaultdict(list)
    for config in configs:
        groups[train_signature(config)].append(config)
    return groups


def fit_ranker_model(
    config: Dict[str, Any],
    fit_df: pd.DataFrame,
    ranker_train_df: pd.DataFrame,
    candidates: List[str],
    user_df: pd.DataFrame,
    item_df: pd.DataFrame,
) -> Any:
    model = build_recommender(config, candidates, user_df, item_df)
    return model.fit(fit_df, ranker_train_df=ranker_train_df)


def model_pool_recall(
    model: Any,
    masked_val_df: pd.DataFrame,
    targets: Sequence[str],
    pool_size: int,
) -> Dict[str, float]:
    max_k = min(pool_size, 200)
    predictions = [model.candidate_pool_for_row(row, k=max_k) for _, row in masked_val_df.iterrows()]
    out: Dict[str, float] = {}
    for k in (10, 25, 50, 80, 120, 200):
        if k <= max_k:
            out[f"pool_top{k}"] = recall_at(predictions, targets, k)
    return out


def evaluate_variant(
    model: Any,
    config: Dict[str, Any],
    masked_val_df: pd.DataFrame,
    targets: Sequence[str],
    exact_weights: Dict[str, float],
    exact_bins: Sequence[str],
    repeat_bins: Sequence[str],
    popularity_bins: Sequence[str],
    v32_predictions: Sequence[Sequence[str]],
    v29_predictions: Sequence[Sequence[str]],
) -> Dict[str, Any]:
    model.update_prediction_settings(config)
    candidate_predictions = [model.predict_row(row, k=10) for _, row in masked_val_df.iterrows()]
    vs_v32 = compare_predictions(v32_predictions, candidate_predictions, targets, exact_bins, EXACT_LENGTH_BINS)
    vs_v29 = compare_predictions(v29_predictions, candidate_predictions, targets, exact_bins, EXACT_LENGTH_BINS)
    vs_v32_repeat = compare_predictions(
        v32_predictions,
        candidate_predictions,
        targets,
        repeat_bins,
        REPEAT_CONCENTRATION_BINS,
    )
    vs_v32_popularity = compare_predictions(
        v32_predictions,
        candidate_predictions,
        targets,
        popularity_bins,
        POPULARITY_BINS,
    )
    v32_weighted = weighted_from_compare(vs_v32, "base", exact_weights)
    candidate_weighted_vs_v32 = weighted_from_compare(vs_v32, "candidate", exact_weights)
    v29_weighted = weighted_from_compare(vs_v29, "base", exact_weights)
    candidate_weighted_vs_v29 = weighted_from_compare(vs_v29, "candidate", exact_weights)
    return {
        "candidate": str(config["name"]),
        "config": config,
        "masked_vs_v32": vs_v32,
        "masked_vs_v29": vs_v29,
        "masked_vs_v32_by_repeat_concentration": vs_v32_repeat["by_bin"],
        "masked_vs_v32_by_target_popularity": vs_v32_popularity["by_bin"],
        "masked_exact_weighted_v32": v32_weighted,
        "masked_exact_weighted_v29": v29_weighted,
        "masked_exact_weighted_candidate": candidate_weighted_vs_v32,
        "masked_exact_weighted_delta_vs_v32": candidate_weighted_vs_v32 - v32_weighted,
        "masked_exact_weighted_delta_vs_v29": candidate_weighted_vs_v29 - v29_weighted,
    }


def summarize_segments(
    results: Sequence[Dict[str, Any]],
    compare_key: str,
    segment_key: str,
    bin_order: Sequence[str],
) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for bin_name in bin_order:
        values = [float(r[compare_key][segment_key][str(bin_name)]["delta"]) for r in results]
        changed = [float(r[compare_key][segment_key][str(bin_name)]["changed_rows"]) for r in results]
        top1_changed = [float(r[compare_key][segment_key][str(bin_name)]["top1_changed"]) for r in results]
        top3_changed = [float(r[compare_key][segment_key][str(bin_name)]["top3_changed"]) for r in results]
        overlap = [float(r[compare_key][segment_key][str(bin_name)]["top10_overlap_mean"]) for r in results]
        out[str(bin_name)] = {
            "mean_delta": float(np.mean(values)) if values else 0.0,
            "min_delta": float(np.min(values)) if values else 0.0,
            "mean_changed_rows": float(np.mean(changed)) if changed else 0.0,
            "max_top1_changed": float(np.max(top1_changed)) if top1_changed else 0.0,
            "max_top3_changed": float(np.max(top3_changed)) if top3_changed else 0.0,
            "mean_top10_overlap": float(np.mean(overlap)) if overlap else 0.0,
        }
    return out


def summarize_variant_results(
    results: Sequence[Dict[str, Any]],
    *,
    test_rows: int,
) -> Dict[str, Any]:
    deltas_v32 = [float(r["masked_exact_weighted_delta_vs_v32"]) for r in results]
    deltas_v29 = [float(r["masked_exact_weighted_delta_vs_v29"]) for r in results]
    changed_v32 = [float(r["masked_vs_v32"]["total"]["changed_rows"]) for r in results]
    top1_v32 = [float(r["masked_vs_v32"]["total"]["top1_changed"]) for r in results]
    top3_v32 = [float(r["masked_vs_v32"]["total"]["top3_changed"]) for r in results]
    overlap_v32 = [float(r["masked_vs_v32"]["total"]["top10_overlap_mean"]) for r in results]
    n_rows = [float(r["masked_vs_v32"]["total"]["n"]) for r in results]
    changed_rate = [
        (changed / max(1.0, n))
        for changed, n in zip(changed_v32, n_rows)
    ]
    estimated_test_changed = [rate * float(test_rows) for rate in changed_rate]
    summary: Dict[str, Any] = {
        "num_splits": int(len(results)),
        "mean_delta_vs_v32": float(np.mean(deltas_v32)) if deltas_v32 else 0.0,
        "min_delta_vs_v32": float(np.min(deltas_v32)) if deltas_v32 else 0.0,
        "max_delta_vs_v32": float(np.max(deltas_v32)) if deltas_v32 else 0.0,
        "positive_splits_vs_v32": int(sum(delta > 0.0 for delta in deltas_v32)),
        "split_deltas_vs_v32": deltas_v32,
        "mean_delta_vs_v29": float(np.mean(deltas_v29)) if deltas_v29 else 0.0,
        "min_delta_vs_v29": float(np.min(deltas_v29)) if deltas_v29 else 0.0,
        "split_deltas_vs_v29": deltas_v29,
        "mean_changed_rows_vs_v32": float(np.mean(changed_v32)) if changed_v32 else 0.0,
        "mean_estimated_test_changed_rows_vs_v32": float(np.mean(estimated_test_changed))
        if estimated_test_changed
        else 0.0,
        "max_top1_changed_vs_v32": float(np.max(top1_v32)) if top1_v32 else 0.0,
        "max_top3_changed_vs_v32": float(np.max(top3_v32)) if top3_v32 else 0.0,
        "mean_top10_overlap_vs_v32": float(np.mean(overlap_v32)) if overlap_v32 else 0.0,
        "exact_len_summary_vs_v32": summarize_segments(
            results,
            "masked_vs_v32",
            "by_bin",
            EXACT_LENGTH_BINS,
        ),
    }
    summary["target_popularity_summary_vs_v32"] = {}
    for bin_name in POPULARITY_BINS:
        values = [
            float(r["masked_vs_v32_by_target_popularity"][bin_name]["delta"])
            for r in results
        ]
        changed = [
            float(r["masked_vs_v32_by_target_popularity"][bin_name]["changed_rows"])
            for r in results
        ]
        top1_changed = [
            float(r["masked_vs_v32_by_target_popularity"][bin_name]["top1_changed"])
            for r in results
        ]
        top3_changed = [
            float(r["masked_vs_v32_by_target_popularity"][bin_name]["top3_changed"])
            for r in results
        ]
        summary["target_popularity_summary_vs_v32"][bin_name] = {
            "mean_delta": float(np.mean(values)) if values else 0.0,
            "min_delta": float(np.min(values)) if values else 0.0,
            "mean_changed_rows": float(np.mean(changed)) if changed else 0.0,
            "max_top1_changed": float(np.max(top1_changed)) if top1_changed else 0.0,
            "max_top3_changed": float(np.max(top3_changed)) if top3_changed else 0.0,
        }
    checks = {
        "mean_vs_v32_ge_0.00003": summary["mean_delta_vs_v32"] >= 0.00003,
        "min_vs_v32_ge_-0.00015": summary["min_delta_vs_v32"] >= -0.00015,
        "top1_changed_eq_0": summary["max_top1_changed_vs_v32"] == 0.0,
        "estimated_test_changed_rows_100_800": 100.0
        <= summary["mean_estimated_test_changed_rows_vs_v32"]
        <= 800.0,
    }
    summary["guard"] = {"pass": bool(all(checks.values())), "checks": checks}
    return summary


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
    _, ranker_label_df = recommendation_split(fit_df, val_ratio=val_ratio, seed=seed + 700)
    ranker_train_df = masked_history_eval(ranker_label_df, test_df, seed=seed + 1700)
    ranker_train_df["target_iid"] = ranker_label_df["target_iid"].astype(str).values

    v32_config = v32_ranker_gate_gt80_config()
    v32_model = fit_ranker_model(v32_config, fit_df, ranker_train_df, candidates, user_df, item_df)
    v29_model = getattr(v32_model, "base_model", None)
    if v29_model is None:
        v29_model = build_recommender(
            v29_medium_long_count_only_pool25_config(),
            candidates,
            user_df,
            item_df,
        ).fit(fit_df)

    targets = masked_val_df["target_iid"].astype(str).tolist()
    exact_bins = exact_bins_for_frame(masked_val_df)
    repeat_bins = repeat_concentration_bins_for_frame(masked_val_df)
    popularity_bins = target_popularity_bins(targets, fit_df)
    v29_predictions = [v29_model.predict_row(row, k=10) for _, row in masked_val_df.iterrows()]
    v32_predictions = [v32_model.predict_row(row, k=10) for _, row in masked_val_df.iterrows()]
    v32_vs_v29 = compare_predictions(v29_predictions, v32_predictions, targets, exact_bins, EXACT_LENGTH_BINS)
    v32_weighted = weighted_from_compare(v32_vs_v29, "candidate", exact_weights)
    v29_weighted = weighted_from_compare(v32_vs_v29, "base", exact_weights)

    trained_models: Dict[Tuple[int, str], Any] = {}
    pool_recalls: Dict[str, Dict[str, float]] = {}
    for signature, signature_configs in grouped_configs(configs).items():
        train_config = dict(signature_configs[0])
        train_config["name"] = f"v33_train_pool{signature[0]}_{signature[1]}"
        model = fit_ranker_model(train_config, fit_df, ranker_train_df, candidates, user_df, item_df)
        trained_models[signature] = model
        pool_recalls[f"pool{signature[0]}_{signature[1]}"] = model_pool_recall(
            model,
            masked_val_df,
            targets,
            pool_size=signature[0],
        )

    variants: List[Dict[str, Any]] = []
    for config in configs:
        model = trained_models[train_signature(config)]
        result = evaluate_variant(
            model,
            config,
            masked_val_df,
            targets,
            exact_weights,
            exact_bins,
            repeat_bins,
            popularity_bins,
            v32_predictions,
            v29_predictions,
        )
        result["seed"] = int(seed)
        variants.append(result)
        total = result["masked_vs_v32"]["total"]
        print(
            f"seed={seed} {config['name']} "
            f"delta_vs_v32={result['masked_exact_weighted_delta_vs_v32']:.9f} "
            f"delta_vs_v29={result['masked_exact_weighted_delta_vs_v29']:.9f} "
            f"changed={total['changed_rows']} top1={total['top1_changed']} "
            f"top3={total['top3_changed']} overlap={total['top10_overlap_mean']:.3f}",
            flush=True,
        )

    return {
        "seed": int(seed),
        "fit_rows": int(len(fit_df)),
        "masked_val_rows": int(len(masked_val_df)),
        "test_exact_len_weights": exact_weights,
        "v32_exact_weighted_vs_v29": v32_weighted - v29_weighted,
        "v29_exact_weighted": v29_weighted,
        "v32_exact_weighted": v32_weighted,
        "pool_recall": pool_recalls,
        "variants": variants,
    }


def selected_payload_keys(config: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "name": config.get("name"),
        "candidate_pool_size": config.get("candidate_pool_size"),
        "ranker_label_mode": config.get("ranker_label_mode", "target"),
        "segment_gate_profiles": config.get("segment_gate_profiles", {}),
    }


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
        raise ValueError("No v33 variants selected.")

    train_df, test_df, user_df, item_df, candidates = load_data(args.rec_data)
    seed_results = [
        evaluate_seed(seed, train_df, test_df, user_df, item_df, candidates, args.val_ratio, configs)
        for seed in args.seeds
    ]

    by_name: Dict[str, List[Dict[str, Any]]] = {str(config["name"]): [] for config in configs}
    for seed_result in seed_results:
        for result in seed_result["variants"]:
            by_name[str(result["candidate"])].append(result)

    summaries: Dict[str, Any] = {}
    for config in configs:
        name = str(config["name"])
        summaries[name] = summarize_variant_results(by_name[name], test_rows=len(test_df))

    ranked = sorted(
        summaries.items(),
        key=lambda item: (
            -float(item[1]["mean_delta_vs_v32"]),
            -float(item[1]["min_delta_vs_v32"]),
            float(item[1]["max_top1_changed_vs_v32"]),
        ),
    )
    print("ranked_variants", flush=True)
    for name, summary in ranked:
        print(
            f"{name} mean_vs_v32={summary['mean_delta_vs_v32']:.9f} "
            f"min_vs_v32={summary['min_delta_vs_v32']:.9f} "
            f"mean_vs_v29={summary['mean_delta_vs_v29']:.9f} "
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
        "baseline": args.baseline,
        "official_baseline_note": "v32 is official best: 0.6278 / 0.7590 / 0.4966",
        "v29_config": v29_medium_long_count_only_pool25_config(),
        "v31_train_reference_config": v31_covisit_lgbm_ranker_config(),
        "v32_config": v32_ranker_gate_gt80_config(),
        "variants": [selected_payload_keys(config) for config in configs],
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
