#!/usr/bin/env python3
"""Probe a learned medium/long-history tail reranker against v24.

The model intentionally keeps v24's guardrails:
- use the v17 base ranking as the candidate pool;
- apply only to rows with ``seq_len >= 21``;
- freeze top1 and rerank only the tail.

When LightGBM is available, this uses LambdaRank over per-row candidate groups.
Otherwise it falls back to a small sklearn histogram GBDT classifier.
"""

from __future__ import annotations

import argparse
import math
import sys
import warnings
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, DefaultDict, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

warnings.filterwarnings("ignore", message="X does not have valid feature names.*", category=UserWarning)

SOLUTION_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = SOLUTION_DIR.parent
if str(SOLUTION_DIR) not in sys.path:
    sys.path.insert(0, str(SOLUTION_DIR))

from src.common import ensure_dir, set_seed, write_json  # noqa: E402
from src.recommendation import (  # noqa: E402
    EXACT_LENGTH_BINS,
    build_recommender,
    parse_sequence,
    recommendation_split,
    test_exact_len_weights,
    test_like_recommendation_split,
    v24_medium_long_history_count_recent_strong_config,
    weighted_exact_score,
)
from src.validation import compare_prediction_sets, exact_bins_for_frame, summarize_pairwise_results  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a learned v24-compatible tail reranker.")
    parser.add_argument("--rec_data", type=Path, default=PROJECT_ROOT / "A推荐")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--val_ratio", type=float, default=0.12)
    parser.add_argument("--ranker_label_ratio", type=float, default=0.35)
    parser.add_argument("--min_len", type=int, default=21)
    parser.add_argument("--freeze_top_n", type=int, default=1)
    parser.add_argument("--base_pool", type=int, default=20)
    parser.add_argument("--max_train_groups", type=int, default=8000)
    parser.add_argument(
        "--output",
        type=Path,
        default=SOLUTION_DIR / "output" / "v28_learning_rerank_eval.json",
    )
    return parser.parse_args()


def load_data(rec_data: Path):
    train_df = pd.read_csv(rec_data / "train.csv").astype({"uid": str, "target_iid": str})
    test_df = pd.read_csv(rec_data / "test.csv").astype({"uid": str})
    user_df = pd.read_csv(rec_data / "user.csv").astype({"uid": str})
    item_df = pd.read_csv(rec_data / "item.csv").astype({"iid": str})
    candidates = item_df["iid"].astype(str).tolist()
    return train_df, test_df, user_df, item_df, candidates


def safe_norm(value: float, denom: float) -> float:
    if denom <= 0:
        return 0.0
    return float(value) / float(denom)


def user_key(row: pd.Series, cols: Sequence[str]) -> Tuple[str, ...]:
    values: List[str] = []
    for col in cols:
        if col not in row.index or pd.isna(row[col]):
            return ()
        values.append(str(row[col]))
    return tuple(values)


def build_feature_stats(
    fit_df: pd.DataFrame,
    user_df: pd.DataFrame,
    item_df: pd.DataFrame,
) -> Dict[str, Any]:
    target_counts = Counter(fit_df["target_iid"].astype(str))
    max_target_count = max(target_counts.values()) if target_counts else 1
    user_lookup = user_df.set_index("uid") if "uid" in user_df.columns else pd.DataFrame()
    item_lookup = item_df.set_index("iid") if "iid" in item_df.columns else pd.DataFrame()
    group_specs = (
        ("u_cat_01", "u_cat_02"),
        ("u_cat_01", "u_cat_02", "u_cat_06"),
        ("u_cat_01",),
    )
    user_group_counts: Dict[Tuple[str, ...], DefaultDict[Tuple[str, ...], Counter]] = {
        cols: defaultdict(Counter) for cols in group_specs
    }
    user_group_totals: Dict[Tuple[str, ...], Counter] = {cols: Counter() for cols in group_specs}
    merged = fit_df[["uid", "target_iid"]].merge(user_df, on="uid", how="left")
    for row in merged.itertuples(index=False):
        row_dict = row._asdict()
        target = str(row_dict["target_iid"])
        for cols in group_specs:
            if all(col in row_dict and pd.notna(row_dict[col]) for col in cols):
                key = tuple(str(row_dict[col]) for col in cols)
                user_group_counts[cols][key][target] += 1
                user_group_totals[cols][key] += 1

    item_cols = [c for c in item_df.columns if c.startswith("i_cat_") or c.startswith("i_bucket_")]
    item_feature_counts: DefaultDict[Tuple[str, str], Counter] = defaultdict(Counter)
    item_feature_totals: Counter = Counter()
    for iid, count in target_counts.items():
        if iid not in item_lookup.index:
            continue
        item_row = item_lookup.loc[iid]
        for col in item_cols:
            value = item_row.get(col)
            if pd.notna(value):
                key = (col, str(value))
                item_feature_counts[key][iid] += int(count)
                item_feature_totals[key] += int(count)

    return {
        "target_counts": target_counts,
        "max_target_count": max_target_count,
        "user_lookup": user_lookup,
        "item_lookup": item_lookup,
        "group_specs": group_specs,
        "user_group_counts": user_group_counts,
        "user_group_totals": user_group_totals,
        "item_cols": item_cols,
        "item_feature_counts": item_feature_counts,
        "item_feature_totals": item_feature_totals,
    }


def feature_vector(
    row: pd.Series,
    iid: str,
    rank: int,
    hist: Sequence[str],
    stats: Dict[str, Any],
) -> List[float]:
    counts = Counter(hist)
    hist_len = len(hist)
    max_count = max(counts.values()) if counts else 0
    recency = {item: pos for pos, item in enumerate(hist)}
    item_count = counts.get(iid, 0)
    item_recency = recency.get(iid, -1)
    unique_ratio = len(counts) / max(hist_len, 1)
    repeat_ratio = 1.0 - unique_ratio if hist_len else 0.0
    top_share = max_count / max(hist_len, 1)
    base_gain = 1.0 / math.log2(rank + 1)
    count_feature = safe_norm(math.log1p(item_count), math.log1p(max_count))
    recency_feature = safe_norm(item_recency + 1, hist_len) if item_recency >= 0 else 0.0
    target_count = stats["target_counts"].get(iid, 0)
    pop_feature = safe_norm(math.log1p(target_count), math.log1p(stats["max_target_count"]))

    uid = str(row.get("uid", ""))
    user_lookup = stats["user_lookup"]
    group_features: List[float] = []
    if not user_lookup.empty and uid in user_lookup.index:
        user_row = user_lookup.loc[uid]
        if isinstance(user_row, pd.DataFrame):
            user_row = user_row.iloc[0]
        for cols in stats["group_specs"]:
            key = user_key(user_row, cols)
            counter = stats["user_group_counts"].get(cols, {}).get(key)
            denom = math.log1p(max(counter.values())) if counter else 0.0
            group_features.append(safe_norm(math.log1p(counter.get(iid, 0)) if counter else 0.0, denom))
    else:
        group_features = [0.0 for _ in stats["group_specs"]]
    while len(group_features) < len(stats["group_specs"]):
        group_features.append(0.0)

    item_lookup = stats["item_lookup"]
    item_prior = 0.0
    if not item_lookup.empty and iid in item_lookup.index:
        item_row = item_lookup.loc[iid]
        if isinstance(item_row, pd.DataFrame):
            item_row = item_row.iloc[0]
        for col in stats["item_cols"]:
            value = item_row.get(col)
            key = (col, str(value))
            total = stats["item_feature_totals"].get(key, 0)
            if total > 0:
                item_prior += stats["item_feature_counts"].get(key, Counter()).get(iid, 0) / total

    return [
        float(rank),
        base_gain,
        math.log1p(hist_len),
        unique_ratio,
        repeat_ratio,
        top_share,
        float(item_count > 0),
        count_feature,
        recency_feature,
        float(hist[-1] == iid) if hist else 0.0,
        float(item_count),
        pop_feature,
        item_prior,
        *group_features,
    ]


def build_rank_training_data(
    label_df: pd.DataFrame,
    base_model: Any,
    stats: Dict[str, Any],
    min_len: int,
    freeze_top_n: int,
    base_pool: int,
    max_groups: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, List[int], Dict[str, int]]:
    rng = np.random.default_rng(seed)
    rows = label_df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    xs: List[List[float]] = []
    ys: List[int] = []
    groups: List[int] = []
    scanned = 0
    positive_groups = 0
    for _, row in rows.iterrows():
        if positive_groups >= max_groups:
            break
        hist = parse_sequence(row.get("item_seq_raw", ""))
        if len(hist) < min_len:
            continue
        scanned += 1
        target = str(row.get("target_iid", ""))
        base_items = base_model.predict_row(row, k=base_pool)
        tail = list(base_items[freeze_top_n:base_pool])
        if target not in tail:
            continue
        positive_groups += 1
        negatives = [iid for iid in tail if iid != target]
        if len(negatives) > 10:
            keep = set(rng.choice(negatives, size=10, replace=False).tolist())
            tail = [iid for iid in tail if iid == target or iid in keep]
        start = len(xs)
        for rank, iid in enumerate(base_items, start=1):
            if rank <= freeze_top_n or iid not in tail:
                continue
            xs.append(feature_vector(row, iid, rank, hist, stats))
            ys.append(1 if iid == target else 0)
        groups.append(len(xs) - start)
    return (
        np.asarray(xs, dtype=np.float32),
        np.asarray(ys, dtype=np.int32),
        groups,
        {"scanned_rows": int(scanned), "positive_groups": int(positive_groups), "examples": int(len(xs))},
    )


def fit_ranker(x: np.ndarray, y: np.ndarray, groups: List[int], seed: int) -> Tuple[Any, str]:
    if len(np.unique(y)) < 2:
        raise RuntimeError("Learning rerank probe found no positive training examples.")
    try:
        from lightgbm import LGBMRanker

        ranker = LGBMRanker(
            objective="lambdarank",
            metric="ndcg",
            n_estimators=80,
            learning_rate=0.045,
            num_leaves=15,
            min_child_samples=24,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            random_state=seed,
            verbosity=-1,
            force_col_wise=True,
        )
        ranker.fit(x, y, group=groups)
        return ranker, "lightgbm_lambdarank"
    except Exception as exc:  # pragma: no cover - depends on optional package
        print(f"LightGBM unavailable or failed ({exc}); falling back to sklearn HGB.", flush=True)
        sample_weight = np.where(y > 0, max(1.0, float((y == 0).sum()) / max(float((y > 0).sum()), 1.0)), 1.0)
        clf = HistGradientBoostingClassifier(
            max_iter=100,
            max_leaf_nodes=15,
            learning_rate=0.055,
            l2_regularization=0.1,
            random_state=seed,
        )
        clf.fit(x, y, sample_weight=sample_weight)
        return clf, "sklearn_hist_gradient_boosting"


def predict_scores(model: Any, x: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(x)[:, 1]
    return np.asarray(model.predict(x), dtype=np.float64)


def learned_rerank(
    row: pd.Series,
    base_items: Sequence[str],
    model: Any,
    stats: Dict[str, Any],
    variant: Dict[str, Any],
) -> List[str]:
    k = int(variant.get("k", 10))
    min_len = int(variant.get("min_len", 21))
    freeze_top_n = int(variant.get("freeze_top_n", 1))
    base_pool = max(k, min(int(variant.get("base_pool", 20)), len(base_items)))
    base_items = list(base_items[:base_pool])
    hist = parse_sequence(row.get("item_seq_raw", ""))
    if len(hist) < min_len or len(base_items) <= freeze_top_n:
        return base_items[:k]
    frozen = base_items[:freeze_top_n]
    tail = base_items[freeze_top_n:base_pool]
    x = np.asarray(
        [feature_vector(row, iid, rank, hist, stats) for rank, iid in enumerate(base_items, start=1) if iid in tail],
        dtype=np.float32,
    )
    learned = predict_scores(model, x)
    blend = float(variant.get("base_blend", 0.0))
    seen_bonus = float(variant.get("seen_bonus", 0.0))
    counts = Counter(hist)
    scores: Dict[str, float] = {}
    for idx, iid in enumerate(tail):
        rank = freeze_top_n + idx + 1
        score = float(learned[idx]) + blend / math.log2(rank + 1)
        if iid in counts:
            score += seen_bonus
        elif bool(variant.get("seen_only", False)):
            score -= 1e6
        scores[iid] = score
    ordered_tail = sorted(tail, key=lambda iid: (-scores.get(iid, 0.0), tail.index(iid), iid))
    return (frozen + ordered_tail)[:k]


def variants(min_len: int, freeze_top_n: int, base_pool: int) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for blend in (0.0, 0.05, 0.10, 0.20, 0.35, 0.60, 1.00):
        out.append(
            {
                "name": f"v28_lgbm_tail_len{min_len}_pool{base_pool}_blend{blend:g}",
                "min_len": min_len,
                "freeze_top_n": freeze_top_n,
                "base_pool": base_pool,
                "base_blend": blend,
                "seen_only": False,
                "seen_bonus": 0.0,
            }
        )
    for blend in (0.10, 0.20, 0.35):
        out.append(
            {
                "name": f"v28_lgbm_seenonly_len{min_len}_pool{base_pool}_blend{blend:g}",
                "min_len": min_len,
                "freeze_top_n": freeze_top_n,
                "base_pool": base_pool,
                "base_blend": blend,
                "seen_only": True,
                "seen_bonus": 0.0,
            }
        )
    for seen_bonus in (0.02, 0.05, 0.10):
        out.append(
            {
                "name": f"v28_lgbm_seenbonus{seen_bonus:g}_len{min_len}_pool{base_pool}",
                "min_len": min_len,
                "freeze_top_n": freeze_top_n,
                "base_pool": base_pool,
                "base_blend": 0.20,
                "seen_only": False,
                "seen_bonus": seen_bonus,
            }
        )
    return out


def evaluate_split(
    seed: int,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    user_df: pd.DataFrame,
    item_df: pd.DataFrame,
    candidates: List[str],
    args: argparse.Namespace,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    set_seed(seed)
    fit_df, val_df, masked_val_df, _ = test_like_recommendation_split(
        train_df, test_df, val_ratio=args.val_ratio, seed=seed
    )
    _, rank_label_df = recommendation_split(fit_df, val_ratio=args.ranker_label_ratio, seed=seed + 701)
    v24_config = v24_medium_long_history_count_recent_strong_config()
    v24_model = build_recommender(v24_config, candidates, user_df, item_df).fit(fit_df)
    base_model = v24_model.base_model
    stats = build_feature_stats(fit_df, user_df, item_df)
    x, y, groups, train_stats = build_rank_training_data(
        rank_label_df,
        base_model,
        stats,
        min_len=int(args.min_len),
        freeze_top_n=int(args.freeze_top_n),
        base_pool=int(args.base_pool),
        max_groups=int(args.max_train_groups),
        seed=seed + 913,
    )
    ranker, ranker_name = fit_ranker(x, y, groups, seed=seed)
    base_pool = int(args.base_pool)
    baseline_masked = [v24_model.predict_row(row, k=10) for _, row in masked_val_df.iterrows()]
    baseline_val = [v24_model.predict_row(row, k=10) for _, row in val_df.iterrows()]
    base_masked_pool = [base_model.predict_row(row, k=base_pool) for _, row in masked_val_df.iterrows()]
    base_val_pool = [base_model.predict_row(row, k=base_pool) for _, row in val_df.iterrows()]
    exact_weights = test_exact_len_weights(test_df)
    split_rows: List[Dict[str, Any]] = []
    for variant in variants(int(args.min_len), int(args.freeze_top_n), int(args.base_pool)):
        masked_candidate = [
            learned_rerank(row, base_items, ranker, stats, variant)
            for (_, row), base_items in zip(masked_val_df.iterrows(), base_masked_pool)
        ]
        val_candidate = [
            learned_rerank(row, base_items, ranker, stats, variant)
            for (_, row), base_items in zip(val_df.iterrows(), base_val_pool)
        ]
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
                "ranker": ranker_name,
                "train_stats": train_stats,
                "natural": natural,
                "masked": masked,
                "masked_exact_weighted_base": base_weighted,
                "masked_exact_weighted_candidate": candidate_weighted,
                "masked_exact_weighted_delta": candidate_weighted - base_weighted,
            }
        )
    meta = {"seed": int(seed), "ranker": ranker_name, "train_stats": train_stats}
    return split_rows, meta


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
    all_results: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    metas: List[Dict[str, Any]] = []
    for seed in args.seeds:
        split_rows, meta = evaluate_split(seed, train_df, test_df, user_df, item_df, candidates, args)
        metas.append(meta)
        for row in split_rows:
            all_results[row["candidate"]].append(row)
        partial = summarize(all_results)
        top = partial[0]
        print(
            f"seed={seed} top={top['candidate']} "
            f"mean={top['summary']['mean_masked_exact_weighted_delta']:.6f} "
            f"min={top['summary']['min_masked_exact_weighted_delta']:.6f} "
            f"train={meta['train_stats']}",
            flush=True,
        )
    results = summarize(all_results)
    payload = {
        "rec_data": str(args.rec_data),
        "seeds": list(args.seeds),
        "val_ratio": float(args.val_ratio),
        "ranker_label_ratio": float(args.ranker_label_ratio),
        "baseline": "v24_medium_long_history_count_recent_len21_alpha25",
        "metas": metas,
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
