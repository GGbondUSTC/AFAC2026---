"""Pairwise validation helpers for guarded recommendation probes."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from .common import ndcg_at_k
from .recommendation import (
    EXACT_LENGTH_BINS,
    build_recommender,
    exact_length_bin,
    parse_sequence,
    test_exact_len_weights,
    weighted_exact_score,
)

REPEAT_CONCENTRATION_BINS = ("empty", "unique", "low", "medium", "high")


def gain_at_k(prediction: Sequence[str], target: str, k: int = 10) -> float:
    for rank, iid in enumerate(prediction[:k], start=1):
        if iid == target:
            return 1.0 / math.log2(rank + 1)
    return 0.0


def exact_bins_for_frame(df: pd.DataFrame) -> List[str]:
    if "_masked_raw_len" in df.columns:
        lengths = df["_masked_raw_len"].astype(int).tolist()
    else:
        lengths = df["item_seq_raw"].map(lambda x: len(parse_sequence(x))).tolist()
    return [exact_length_bin(int(length)) for length in lengths]


def repeat_concentration_bin(hist: Sequence[str]) -> str:
    hist_len = len(hist)
    if hist_len <= 0:
        return "empty"
    counts = Counter(hist)
    if len(counts) == hist_len:
        return "unique"
    repeat_ratio = 1.0 - (len(counts) / hist_len)
    top_share = max(counts.values()) / hist_len
    if repeat_ratio >= 0.60 or top_share >= 0.45:
        return "high"
    if repeat_ratio >= 0.30 or top_share >= 0.25:
        return "medium"
    return "low"


def repeat_concentration_bins_for_frame(df: pd.DataFrame) -> List[str]:
    return [
        repeat_concentration_bin(parse_sequence(value))
        for value in df.get("item_seq_raw", pd.Series([np.nan] * len(df))).tolist()
    ]


def compare_prediction_sets_by_bins(
    base_predictions: Sequence[Sequence[str]],
    candidate_predictions: Sequence[Sequence[str]],
    targets: Sequence[str],
    bins: Sequence[str],
    bin_order: Sequence[str],
) -> Dict[str, Dict[str, float]]:
    base_scores: List[float] = []
    candidate_scores: List[float] = []
    rows: Dict[str, List[int]] = defaultdict(list)
    changed: Dict[str, int] = defaultdict(int)
    top1_changed: Dict[str, int] = defaultdict(int)
    overlap: Dict[str, List[int]] = defaultdict(list)
    for idx, (base, candidate, target, bin_name) in enumerate(
        zip(base_predictions, candidate_predictions, targets, bins)
    ):
        base_top = list(base[:10])
        candidate_top = list(candidate[:10])
        base_scores.append(gain_at_k(base_top, str(target), k=10))
        candidate_scores.append(gain_at_k(candidate_top, str(target), k=10))
        rows[str(bin_name)].append(idx)
        if base_top != candidate_top:
            changed[str(bin_name)] += 1
        if (base_top[:1] or [""])[0] != (candidate_top[:1] or [""])[0]:
            top1_changed[str(bin_name)] += 1
        overlap[str(bin_name)].append(len(set(base_top).intersection(candidate_top)))

    by_bin: Dict[str, Dict[str, float]] = {}
    for bin_name in bin_order:
        idxs = rows.get(str(bin_name), [])
        if not idxs:
            by_bin[str(bin_name)] = {
                "n": 0,
                "base": 0.0,
                "candidate": 0.0,
                "delta": 0.0,
                "changed_rows": 0,
                "top1_changed": 0,
                "top10_overlap_mean": 0.0,
            }
            continue
        base_mean = float(np.mean([base_scores[i] for i in idxs]))
        candidate_mean = float(np.mean([candidate_scores[i] for i in idxs]))
        by_bin[str(bin_name)] = {
            "n": int(len(idxs)),
            "base": base_mean,
            "candidate": candidate_mean,
            "delta": candidate_mean - base_mean,
            "changed_rows": int(changed.get(str(bin_name), 0)),
            "top1_changed": int(top1_changed.get(str(bin_name), 0)),
            "top10_overlap_mean": float(np.mean(overlap.get(str(bin_name), [0]))),
        }
    return by_bin


def compare_prediction_sets(
    base_predictions: Sequence[Sequence[str]],
    candidate_predictions: Sequence[Sequence[str]],
    targets: Sequence[str],
    bins: Sequence[str],
) -> Dict[str, Any]:
    base_scores: List[float] = []
    candidate_scores: List[float] = []
    rows: Dict[str, List[int]] = defaultdict(list)
    changed: Dict[str, int] = defaultdict(int)
    top1_changed: Dict[str, int] = defaultdict(int)
    overlap: Dict[str, List[int]] = defaultdict(list)
    for idx, (base, candidate, target, bin_name) in enumerate(
        zip(base_predictions, candidate_predictions, targets, bins)
    ):
        base_top = list(base[:10])
        candidate_top = list(candidate[:10])
        base_scores.append(gain_at_k(base_top, str(target), k=10))
        candidate_scores.append(gain_at_k(candidate_top, str(target), k=10))
        rows[str(bin_name)].append(idx)
        if base_top != candidate_top:
            changed[str(bin_name)] += 1
        if (base_top[:1] or [""])[0] != (candidate_top[:1] or [""])[0]:
            top1_changed[str(bin_name)] += 1
        overlap[str(bin_name)].append(len(set(base_top).intersection(candidate_top)))

    by_bin: Dict[str, Dict[str, float]] = {}
    for bin_name in EXACT_LENGTH_BINS:
        idxs = rows.get(bin_name, [])
        if not idxs:
            by_bin[bin_name] = {
                "n": 0,
                "base": 0.0,
                "candidate": 0.0,
                "delta": 0.0,
                "changed_rows": 0,
                "top1_changed": 0,
                "top10_overlap_mean": 0.0,
            }
            continue
        base_mean = float(np.mean([base_scores[i] for i in idxs]))
        candidate_mean = float(np.mean([candidate_scores[i] for i in idxs]))
        by_bin[bin_name] = {
            "n": int(len(idxs)),
            "base": base_mean,
            "candidate": candidate_mean,
            "delta": candidate_mean - base_mean,
            "changed_rows": int(changed.get(bin_name, 0)),
            "top1_changed": int(top1_changed.get(bin_name, 0)),
            "top10_overlap_mean": float(np.mean(overlap.get(bin_name, [0]))),
        }
    return {
        "base_ndcg@10": float(np.mean(base_scores)) if base_scores else 0.0,
        "candidate_ndcg@10": float(np.mean(candidate_scores)) if candidate_scores else 0.0,
        "delta_ndcg@10": float(np.mean(np.asarray(candidate_scores) - np.asarray(base_scores)))
        if base_scores
        else 0.0,
        "by_exact_len": by_bin,
        "row_deltas": [float(c - b) for b, c in zip(base_scores, candidate_scores)],
    }


def bootstrap_mean_ci(values: Sequence[float], seed: int = 2026, rounds: int = 500) -> Dict[str, float]:
    if not values:
        return {"low": 0.0, "high": 0.0}
    rng = np.random.default_rng(seed)
    arr = np.asarray(values, dtype=np.float64)
    means = []
    for _ in range(rounds):
        sample = rng.choice(arr, size=len(arr), replace=True)
        means.append(float(np.mean(sample)))
    return {
        "low": float(np.quantile(means, 0.025)),
        "high": float(np.quantile(means, 0.975)),
    }


def evaluate_pairwise_recommender(
    fit_df: pd.DataFrame,
    val_df: pd.DataFrame,
    masked_val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    user_df: pd.DataFrame,
    item_df: pd.DataFrame,
    candidates: List[str],
    candidate_config: Dict[str, Any],
    baseline_config: Optional[Dict[str, Any]] = None,
    seed: int = 42,
) -> Dict[str, Any]:
    candidate_model = build_recommender(candidate_config, candidates, user_df, item_df).fit(fit_df)
    base_model = getattr(candidate_model, "base_model", None)
    if base_model is None:
        if baseline_config is None:
            raise ValueError("baseline_config is required when candidate has no fitted base_model")
        base_model = build_recommender(baseline_config, candidates, user_df, item_df).fit(fit_df)

    val_base = [base_model.predict_row(row, k=10) for _, row in val_df.iterrows()]
    val_candidate = [candidate_model.predict_row(row, k=10) for _, row in val_df.iterrows()]
    masked_base = [base_model.predict_row(row, k=10) for _, row in masked_val_df.iterrows()]
    masked_candidate = [candidate_model.predict_row(row, k=10) for _, row in masked_val_df.iterrows()]

    val_targets = val_df["target_iid"].astype(str).tolist()
    masked_targets = masked_val_df["target_iid"].astype(str).tolist()
    natural = compare_prediction_sets(val_base, val_candidate, val_targets, exact_bins_for_frame(val_df))
    masked = compare_prediction_sets(masked_base, masked_candidate, masked_targets, exact_bins_for_frame(masked_val_df))
    masked_repeat = compare_prediction_sets_by_bins(
        masked_base,
        masked_candidate,
        masked_targets,
        repeat_concentration_bins_for_frame(masked_val_df),
        REPEAT_CONCENTRATION_BINS,
    )

    exact_weights = test_exact_len_weights(test_df)
    base_weighted = weighted_exact_score(
        {b: masked["by_exact_len"][b]["base"] for b in EXACT_LENGTH_BINS},
        exact_weights,
    )
    candidate_weighted = weighted_exact_score(
        {b: masked["by_exact_len"][b]["candidate"] for b in EXACT_LENGTH_BINS},
        exact_weights,
    )
    return {
        "seed": int(seed),
        "candidate": candidate_config.get("name", ""),
        "baseline": getattr(base_model, "config", baseline_config or {}).get("name", ""),
        "test_exact_len_weights": exact_weights,
        "natural": natural,
        "masked": masked,
        "masked_by_repeat_concentration": masked_repeat,
        "masked_exact_weighted_base": base_weighted,
        "masked_exact_weighted_candidate": candidate_weighted,
        "masked_exact_weighted_delta": candidate_weighted - base_weighted,
        "masked_delta_bootstrap_ci": bootstrap_mean_ci(masked["row_deltas"], seed=seed + 1000),
        "natural_base_ndcg@10": ndcg_at_k(val_base, val_targets, k=10),
        "natural_candidate_ndcg@10": ndcg_at_k(val_candidate, val_targets, k=10),
        "masked_base_ndcg@10": ndcg_at_k(masked_base, masked_targets, k=10),
        "masked_candidate_ndcg@10": ndcg_at_k(masked_candidate, masked_targets, k=10),
    }


def summarize_pairwise_results(results: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    deltas = [float(r["masked_exact_weighted_delta"]) for r in results]
    summary: Dict[str, Any] = {
        "num_splits": int(len(results)),
        "mean_masked_exact_weighted_delta": float(np.mean(deltas)) if deltas else 0.0,
        "min_masked_exact_weighted_delta": float(np.min(deltas)) if deltas else 0.0,
        "max_masked_exact_weighted_delta": float(np.max(deltas)) if deltas else 0.0,
        "split_deltas": deltas,
    }
    by_bin: Dict[str, Dict[str, float]] = {}
    for bin_name in EXACT_LENGTH_BINS:
        values = [float(r["masked"]["by_exact_len"][bin_name]["delta"]) for r in results]
        changed = [float(r["masked"]["by_exact_len"][bin_name]["changed_rows"]) for r in results]
        top1_changed = [float(r["masked"]["by_exact_len"][bin_name]["top1_changed"]) for r in results]
        by_bin[bin_name] = {
            "mean_delta": float(np.mean(values)) if values else 0.0,
            "min_delta": float(np.min(values)) if values else 0.0,
            "mean_changed_rows": float(np.mean(changed)) if changed else 0.0,
            "max_top1_changed": float(np.max(top1_changed)) if top1_changed else 0.0,
        }
    summary["masked_by_exact_len_summary"] = by_bin
    if results and "masked_by_repeat_concentration" in results[0]:
        by_repeat: Dict[str, Dict[str, float]] = {}
        for bin_name in REPEAT_CONCENTRATION_BINS:
            values = [
                float(r["masked_by_repeat_concentration"][bin_name]["delta"])
                for r in results
            ]
            changed = [
                float(r["masked_by_repeat_concentration"][bin_name]["changed_rows"])
                for r in results
            ]
            top1_changed = [
                float(r["masked_by_repeat_concentration"][bin_name]["top1_changed"])
                for r in results
            ]
            by_repeat[bin_name] = {
                "mean_delta": float(np.mean(values)) if values else 0.0,
                "min_delta": float(np.min(values)) if values else 0.0,
                "mean_changed_rows": float(np.mean(changed)) if changed else 0.0,
                "max_top1_changed": float(np.max(top1_changed)) if top1_changed else 0.0,
            }
        summary["masked_by_repeat_concentration_summary"] = by_repeat
    return summary


def v19_guard(summary: Dict[str, Any]) -> Dict[str, Any]:
    by_bin = summary.get("masked_by_exact_len_summary", {})
    checks = {
        "mean_delta_ge_0.0012": float(summary.get("mean_masked_exact_weighted_delta", 0.0)) >= 0.0012,
        "min_delta_ge_0.0005": float(summary.get("min_masked_exact_weighted_delta", 0.0)) >= 0.0005,
        "no_negative_split": all(float(x) >= 0.0 for x in summary.get("split_deltas", [])),
        "len2_unchanged": float(by_bin.get("2", {}).get("mean_changed_rows", 0.0)) == 0.0,
        "len4plus_unchanged": all(
            float(by_bin.get(bin_name, {}).get("mean_changed_rows", 0.0)) == 0.0
            for bin_name in ("4-10", "11-20", "21-30", "31-80", ">80")
        ),
        "top1_unchanged_short": all(
            float(by_bin.get(bin_name, {}).get("max_top1_changed", 0.0)) == 0.0
            for bin_name in ("1", "2", "3")
        ),
    }
    return {"pass": bool(all(checks.values())), "checks": checks}
