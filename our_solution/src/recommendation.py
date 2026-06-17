"""Recommendation pipeline with test-distribution-aware validation.

v2 keeps the lightweight nature of v1 but changes the selection objective from
plain random validation NDCG to a test-like metric. The public test set is much
shorter than train histories, so candidates are evaluated after masking
validation histories to the observed test length distribution.
"""

from __future__ import annotations

import math
import time
import importlib.util
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, DefaultDict, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .common import Timer, Trajectory, ensure_dir, ndcg_at_k, write_json


LENGTH_BINS = ("<=3", "4-10", "11-30", "31-80", ">80")


def parse_sequence(value: Any) -> List[str]:
    if pd.isna(value):
        return []
    text = str(value).strip()
    if not text:
        return []
    return [x.strip() for x in text.split(",") if x.strip()]


def adjacent_dedup(items: Sequence[str]) -> List[str]:
    result: List[str] = []
    last = None
    for item in items:
        if item != last:
            result.append(item)
            last = item
    return result


def length_bin(length: int) -> str:
    if length <= 3:
        return "<=3"
    if length <= 10:
        return "4-10"
    if length <= 30:
        return "11-30"
    if length <= 80:
        return "31-80"
    return ">80"


def recommendation_split(train_df: pd.DataFrame, val_ratio: float, seed: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Target-stratified split without requiring sklearn."""
    rng = np.random.default_rng(seed)
    fit_parts: List[pd.DataFrame] = []
    val_parts: List[pd.DataFrame] = []
    for _, group in train_df.groupby("target_iid", sort=False):
        indices = group.index.to_numpy()
        indices = indices[rng.permutation(len(indices))]
        val_size = 1 if len(indices) <= 2 else max(1, int(round(len(indices) * val_ratio)))
        val_parts.append(train_df.loc[indices[:val_size]])
        fit_parts.append(train_df.loc[indices[val_size:]])
    fit = pd.concat(fit_parts, axis=0).sample(frac=1.0, random_state=seed)
    val = pd.concat(val_parts, axis=0).sample(frac=1.0, random_state=seed + 1)
    return fit.reset_index(drop=True), val.reset_index(drop=True)


def test_bin_weights(test_df: pd.DataFrame, seq_col: str = "item_seq_raw") -> Dict[str, float]:
    lengths = test_df[seq_col].map(lambda x: len(parse_sequence(x))) if seq_col in test_df.columns else pd.Series([0] * len(test_df))
    bins = lengths.map(length_bin)
    counts = bins.value_counts(normalize=True).to_dict()
    return {b: float(counts.get(b, 0.0)) for b in LENGTH_BINS}


def masked_history_eval(
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    seed: int,
    seq_col: str = "item_seq_raw",
) -> pd.DataFrame:
    """Mask validation histories to match the public test length distribution."""
    rng = np.random.default_rng(seed)
    if seq_col in test_df.columns:
        test_lengths = test_df[seq_col].map(lambda x: len(parse_sequence(x))).to_numpy()
    else:
        test_lengths = np.zeros(len(test_df), dtype=int)
    if len(test_lengths) == 0:
        test_lengths = np.array([0], dtype=int)
    sampled_lengths = rng.choice(test_lengths, size=len(val_df), replace=True)

    masked = val_df.copy()
    raw_values: List[Any] = []
    dedup_values: List[Any] = []
    bins: List[str] = []
    for row, target_len in zip(val_df.itertuples(index=False), sampled_lengths):
        row_dict = row._asdict()
        raw_items = parse_sequence(row_dict.get("item_seq_raw", row_dict.get(seq_col, "")))
        if target_len <= 0:
            truncated: List[str] = []
            raw_values.append(np.nan)
            dedup_values.append(np.nan)
        else:
            truncated = raw_items[-int(target_len):]
            raw_values.append(",".join(truncated))
            dedup_values.append(",".join(adjacent_dedup(truncated)))
        bins.append(length_bin(len(truncated)))
    masked["item_seq_raw"] = raw_values
    masked["item_seq_dedup"] = dedup_values
    masked["_eval_bin"] = bins
    masked["_masked_raw_len"] = [0 if pd.isna(x) else len(parse_sequence(x)) for x in raw_values]
    return masked


def test_like_recommendation_split(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    val_ratio: float,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, float]]:
    fit_df, val_df = recommendation_split(train_df, val_ratio=val_ratio, seed=seed)
    masked_val_df = masked_history_eval(val_df, test_df, seed=seed + 99)
    weights = test_bin_weights(test_df)
    return fit_df, val_df, masked_val_df, weights


def score_by_bin(predictions: Sequence[Sequence[str]], targets: Sequence[str], bins: Sequence[str]) -> Dict[str, float]:
    scores: DefaultDict[str, List[float]] = defaultdict(list)
    for pred, target, bin_name in zip(predictions, targets, bins):
        value = 0.0
        for rank, iid in enumerate(pred[:10], start=1):
            if iid == target:
                value = 1.0 / math.log2(rank + 1)
                break
        scores[str(bin_name)].append(value)
    return {b: float(np.mean(scores[b])) if scores.get(b) else 0.0 for b in LENGTH_BINS}


def weighted_bin_score(by_bin: Dict[str, float], weights: Dict[str, float]) -> float:
    return float(sum(by_bin.get(b, 0.0) * weights.get(b, 0.0) for b in LENGTH_BINS))


def rank_fusion_topk(
    source_scores: Dict[str, np.ndarray],
    weights: Dict[str, float],
    candidates: List[str],
    k: int = 10,
    pool: int = 300,
    tie_scores: Optional[np.ndarray] = None,
) -> List[str]:
    """Borda-like rank fusion over source score vectors."""
    n = len(candidates)
    fused = np.zeros(n, dtype=np.float32)
    max_pool = min(pool, n)
    for name, scores in source_scores.items():
        weight = float(weights.get(name, 0.0))
        if weight <= 0 or scores.size == 0:
            continue
        finite = np.isfinite(scores)
        if not finite.any():
            continue
        clean = np.where(finite, scores, -np.inf)
        if float(np.nanmax(clean) - np.nanmin(clean[finite])) <= 1e-12:
            continue
        order = np.argsort(-clean, kind="mergesort")[:max_pool]
        gains = weight / np.log2(np.arange(2, len(order) + 2, dtype=np.float32))
        fused[order] += gains
    if tie_scores is None:
        tie_scores = np.zeros(n, dtype=np.float32)
    order = np.lexsort((np.arange(n), -tie_scores, -fused))
    return [candidates[i] for i in order[:k]]


@dataclass
class UserFeaturePopularity:
    candidates: List[str]
    user_df: pd.DataFrame
    min_group_count: int = 20

    def __post_init__(self) -> None:
        self.candidate_to_idx = {iid: i for i, iid in enumerate(self.candidates)}
        self.user_lookup = self.user_df.set_index("uid") if "uid" in self.user_df.columns else pd.DataFrame()
        self.feature_counts: DefaultDict[Tuple[str, str], Counter] = defaultdict(Counter)
        self.feature_totals: Counter = Counter()
        self.global_counts: Counter = Counter()
        self.user_cols = [c for c in self.user_df.columns if c.startswith("u_cat_")]

    def fit(self, train_df: pd.DataFrame) -> "UserFeaturePopularity":
        merged = train_df[["uid", "target_iid"]].merge(self.user_df, on="uid", how="left")
        for row in merged.itertuples(index=False):
            row_dict = row._asdict()
            target = str(row_dict["target_iid"])
            self.global_counts[target] += 1
            for col in self.user_cols:
                value = row_dict.get(col)
                if pd.notna(value):
                    key = (col, str(value))
                    self.feature_counts[key][target] += 1
                    self.feature_totals[key] += 1
        return self

    def score_for_uid(self, uid: str, strength: float = 1.0) -> np.ndarray:
        scores = np.zeros(len(self.candidates), dtype=np.float32)
        if self.user_lookup.empty or uid not in self.user_lookup.index:
            return scores
        user_row = self.user_lookup.loc[uid]
        for col in self.user_cols:
            value = user_row.get(col)
            key = (col, str(value))
            if self.feature_totals.get(key, 0) < self.min_group_count:
                continue
            counter = self.feature_counts.get(key)
            if not counter:
                continue
            denom = math.log1p(max(counter.values()))
            if denom <= 0:
                continue
            for iid, count in counter.items():
                idx = self.candidate_to_idx.get(iid)
                if idx is not None:
                    scores[idx] += strength * math.log1p(count) / denom
        return scores


@dataclass
class HybridRecommender:
    """v1-compatible additive hybrid recommender."""

    config: Dict[str, Any]
    candidates: List[str]
    user_df: pd.DataFrame
    item_df: pd.DataFrame

    def __post_init__(self) -> None:
        self.candidate_to_idx = {iid: i for i, iid in enumerate(self.candidates)}
        self.base_scores = np.zeros(len(self.candidates), dtype=np.float32)
        self.transition: DefaultDict[str, Counter] = defaultdict(Counter)
        self.user_feature_counts: DefaultDict[Tuple[str, str], Counter] = defaultdict(Counter)
        self.user_lookup = self.user_df.set_index("uid") if "uid" in self.user_df.columns else pd.DataFrame()
        self.item_lookup = self.item_df.set_index("iid") if "iid" in self.item_df.columns else pd.DataFrame()

    def fit(self, train_df: pd.DataFrame) -> "HybridRecommender":
        target_counts = Counter(train_df["target_iid"].astype(str))
        self._build_base_scores(target_counts)
        self._build_transitions(train_df)
        self._build_user_feature_counts(train_df)
        self._add_item_feature_prior(target_counts)
        return self

    def _build_base_scores(self, target_counts: Counter) -> None:
        pop_weight = float(self.config.get("pop_weight", 1.0))
        max_count = max(target_counts.values()) if target_counts else 1.0
        for iid, count in target_counts.items():
            idx = self.candidate_to_idx.get(str(iid))
            if idx is not None:
                self.base_scores[idx] += pop_weight * math.log1p(count) / math.log1p(max_count)

    def _sequence_column(self, df: pd.DataFrame) -> str:
        preferred = self.config.get("seq_col", "item_seq_dedup")
        if preferred in df.columns:
            return preferred
        for col in ("item_seq_dedup", "item_seq_raw", "item_seq"):
            if col in df.columns:
                return col
        return ""

    def _build_transitions(self, train_df: pd.DataFrame) -> None:
        seq_col = self._sequence_column(train_df)
        if not seq_col or float(self.config.get("transition_weight", 0.0)) <= 0:
            return
        max_hist = int(self.config.get("max_hist", 50))
        decay = float(self.config.get("decay", 0.92))
        transition_weight = float(self.config.get("transition_weight", 1.0))
        use_unique = bool(self.config.get("unique_history", False))
        for row in train_df.itertuples(index=False):
            row_dict = row._asdict()
            target = str(row_dict["target_iid"])
            hist = parse_sequence(row_dict.get(seq_col, ""))
            if use_unique:
                seen = set()
                dedup_rev = []
                for item in reversed(hist):
                    if item not in seen:
                        dedup_rev.append(item)
                        seen.add(item)
                hist = list(reversed(dedup_rev))
            for pos, item in enumerate(reversed(hist[-max_hist:])):
                self.transition[item][target] += transition_weight * (decay ** pos)

    def _build_user_feature_counts(self, train_df: pd.DataFrame) -> None:
        user_weight = float(self.config.get("user_weight", 0.0))
        if user_weight <= 0 or self.user_df.empty:
            return
        user_cols = [c for c in self.user_df.columns if c.startswith("u_cat_")]
        merged = train_df[["uid", "target_iid"]].merge(self.user_df, on="uid", how="left")
        for row in merged.itertuples(index=False):
            row_dict = row._asdict()
            target = str(row_dict["target_iid"])
            for col in user_cols:
                value = row_dict.get(col)
                if pd.notna(value):
                    self.user_feature_counts[(col, str(value))][target] += 1.0

    def _add_item_feature_prior(self, target_counts: Counter) -> None:
        item_weight = float(self.config.get("item_feature_weight", 0.0))
        if item_weight <= 0 or self.item_df.empty:
            return
        item_cols = [c for c in self.item_df.columns if c.startswith("i_cat_") or c.startswith("i_bucket_")]
        feature_counts: DefaultDict[Tuple[str, str], float] = defaultdict(float)
        for iid, count in target_counts.items():
            if iid not in self.item_lookup.index:
                continue
            item_row = self.item_lookup.loc[iid]
            for col in item_cols:
                value = item_row.get(col)
                if pd.notna(value):
                    feature_counts[(col, str(value))] += float(count)
        max_count = max(feature_counts.values()) if feature_counts else 1.0
        for iid, idx in self.candidate_to_idx.items():
            if iid not in self.item_lookup.index:
                continue
            item_row = self.item_lookup.loc[iid]
            prior = 0.0
            for col in item_cols:
                value = item_row.get(col)
                prior += feature_counts.get((col, str(value)), 0.0)
            if prior > 0:
                self.base_scores[idx] += item_weight * math.log1p(prior) / math.log1p(max_count)

    def score_row(self, row: pd.Series) -> np.ndarray:
        scores = self.base_scores.copy()
        seq_col = self._sequence_column(pd.DataFrame([row]))
        hist = parse_sequence(row.get(seq_col, "")) if seq_col else []
        max_hist = int(self.config.get("max_hist", 50))
        decay = float(self.config.get("decay", 0.92))
        for pos, item in enumerate(reversed(hist[-max_hist:])):
            counter = self.transition.get(item)
            if not counter:
                continue
            for target, value in counter.items():
                idx = self.candidate_to_idx.get(target)
                if idx is not None:
                    scores[idx] += (decay ** pos) * math.log1p(value)
        repeat_weight = float(self.config.get("repeat_weight", 0.0))
        if repeat_weight > 0:
            for pos, item in enumerate(reversed(hist[-max_hist:])):
                idx = self.candidate_to_idx.get(item)
                if idx is not None:
                    scores[idx] += repeat_weight * (decay ** pos)
        user_weight = float(self.config.get("user_weight", 0.0))
        uid = str(row.get("uid", ""))
        if user_weight > 0 and uid in self.user_lookup.index:
            user_row = self.user_lookup.loc[uid]
            for col in [c for c in self.user_df.columns if c.startswith("u_cat_")]:
                value = user_row.get(col)
                counter = self.user_feature_counts.get((col, str(value)))
                if not counter:
                    continue
                for target, count in counter.items():
                    idx = self.candidate_to_idx.get(target)
                    if idx is not None:
                        scores[idx] += user_weight * math.log1p(count)
        if bool(self.config.get("exclude_history", False)):
            for item in set(hist):
                idx = self.candidate_to_idx.get(item)
                if idx is not None:
                    scores[idx] = -np.inf
        return scores

    def predict_row(self, row: pd.Series, k: int = 10) -> List[str]:
        scores = self.score_row(row)
        order = np.lexsort((np.arange(len(scores)), -scores))
        return [self.candidates[i] for i in order[:k]]


@dataclass
class ZeroUserGroupHybridRecommender:
    """Use user-feature target popularity only for true cold-start rows."""

    config: Dict[str, Any]
    candidates: List[str]
    user_df: pd.DataFrame
    item_df: pd.DataFrame

    def __post_init__(self) -> None:
        base_config = dict(v2_weight_tuned_config())
        base_config.update(self.config.get("base_config", {}))
        self.base_model = HybridRecommender(
            config=base_config,
            candidates=self.candidates,
            user_df=self.user_df,
            item_df=self.item_df,
        )
        self.user_lookup = self.user_df.set_index("uid") if "uid" in self.user_df.columns else pd.DataFrame()
        self.group_cols = tuple(self.config.get("group_cols", ("u_cat_01", "u_cat_02")))
        self.fallback_cols = tuple(self.config.get("fallback_cols", ("u_cat_01",)))
        self.min_group_count = int(self.config.get("min_group_count", 3))
        self.group_counts: DefaultDict[Tuple[str, ...], Counter] = defaultdict(Counter)
        self.fallback_counts: DefaultDict[Tuple[str, ...], Counter] = defaultdict(Counter)
        self.global_top: List[str] = []

    def fit(self, train_df: pd.DataFrame) -> "ZeroUserGroupHybridRecommender":
        self.base_model.fit(train_df)
        target_counts = Counter(train_df["target_iid"].astype(str))
        self.global_top = [iid for iid, _ in target_counts.most_common(len(self.candidates))]
        merged = train_df[["uid", "target_iid"]].merge(self.user_df, on="uid", how="left")
        for row in merged.itertuples(index=False):
            row_dict = row._asdict()
            target = str(row_dict["target_iid"])
            if all(col in row_dict and pd.notna(row_dict[col]) for col in self.group_cols):
                key = tuple(str(row_dict[col]) for col in self.group_cols)
                self.group_counts[key][target] += 1
            if all(col in row_dict and pd.notna(row_dict[col]) for col in self.fallback_cols):
                key = tuple(str(row_dict[col]) for col in self.fallback_cols)
                self.fallback_counts[key][target] += 1
        return self

    def _top_from_counter(self, counter: Optional[Counter], k: int) -> List[str]:
        out: List[str] = []
        if counter:
            for iid, _ in counter.most_common(200):
                if iid not in out:
                    out.append(iid)
                if len(out) >= k:
                    return out
        for iid in self.global_top:
            if iid not in out:
                out.append(iid)
            if len(out) >= k:
                break
        return out

    def _user_group_prediction(self, uid: str, k: int) -> List[str]:
        if self.user_lookup.empty or uid not in self.user_lookup.index:
            return self._top_from_counter(None, k)
        user_row = self.user_lookup.loc[uid]
        group_key = tuple(str(user_row[col]) for col in self.group_cols)
        group_counter = self.group_counts.get(group_key)
        if group_counter and sum(group_counter.values()) >= self.min_group_count:
            return self._top_from_counter(group_counter, k)
        fallback_key = tuple(str(user_row[col]) for col in self.fallback_cols)
        fallback_counter = self.fallback_counts.get(fallback_key)
        if fallback_counter and sum(fallback_counter.values()) >= self.min_group_count:
            return self._top_from_counter(fallback_counter, k)
        return self._top_from_counter(None, k)

    def predict_row(self, row: pd.Series, k: int = 10) -> List[str]:
        hist = parse_sequence(row.get("item_seq_raw", ""))
        if len(hist) == 0:
            return self._user_group_prediction(str(row.get("uid", "")), k)
        return self.base_model.predict_row(row, k=k)


@dataclass
class ShortHistoryUserPriorHybridRecommender:
    """v6 hybrid: keep v5 cold-start and add weak user priors for short histories."""

    config: Dict[str, Any]
    candidates: List[str]
    user_df: pd.DataFrame
    item_df: pd.DataFrame

    def __post_init__(self) -> None:
        base_config = v5_repeat_tuned_config()
        base_config.update(self.config.get("base_config", {}))
        self.base_model = HybridRecommender(
            config=base_config,
            candidates=self.candidates,
            user_df=self.user_df,
            item_df=self.item_df,
        )
        raw_length_base_configs = self.config.get("length_base_configs", {})
        self.length_base_configs = {
            int(length): dict(model_config)
            for length, model_config in raw_length_base_configs.items()
        }
        self.length_base_models: Dict[int, HybridRecommender] = {}
        self.candidate_to_idx = {iid: i for i, iid in enumerate(self.candidates)}
        self.user_lookup = self.user_df.set_index("uid") if "uid" in self.user_df.columns else pd.DataFrame()
        self.group_cols = tuple(self.config.get("group_cols", ("u_cat_01", "u_cat_02")))
        self.fallback_cols = tuple(self.config.get("fallback_cols", ("u_cat_01",)))
        raw_short_groups = self.config.get(
            "short_group_cols",
            (
                ("u_cat_01", "u_cat_02"),
                ("u_cat_01", "u_cat_06"),
                ("u_cat_02", "u_cat_06"),
            ),
        )
        self.short_group_cols = tuple(tuple(cols) for cols in raw_short_groups)
        self.short_group_weights = tuple(
            float(x) for x in self.config.get("short_group_weights", (1.0, 0.55, 0.55))
        )
        self.min_group_count = int(self.config.get("min_group_count", 3))
        self.min_short_group_count = int(self.config.get("min_short_group_count", 20))
        self.short_user_weight = float(self.config.get("short_user_weight", 0.08))
        self.short_lengths = set(int(x) for x in self.config.get("short_lengths", (1, 2, 3)))
        self.short_user_pool = int(self.config.get("short_user_pool", 120))
        self.group_counts: DefaultDict[Tuple[str, ...], Counter] = defaultdict(Counter)
        self.fallback_counts: DefaultDict[Tuple[str, ...], Counter] = defaultdict(Counter)
        self.short_counts: Dict[Tuple[str, ...], DefaultDict[Tuple[str, ...], Counter]] = {
            cols: defaultdict(Counter) for cols in self.short_group_cols
        }
        raw_zero_specs = self.config.get("zero_additive_specs", ())
        self.zero_additive_specs = tuple(tuple(cols) for cols in raw_zero_specs)
        self.zero_additive_weights = tuple(
            float(x) for x in self.config.get("zero_additive_weights", ())
        )
        self.zero_additive_global_weight = float(self.config.get("zero_additive_global_weight", 0.0))
        self.zero_additive_min_count = int(self.config.get("zero_additive_min_count", 20))
        self.zero_additive_pool = int(self.config.get("zero_additive_pool", 300))
        self.zero_additive_counts: Dict[Tuple[str, ...], DefaultDict[Tuple[str, ...], Counter]] = {
            cols: defaultdict(Counter) for cols in self.zero_additive_specs
        }
        self.zero_additive_totals: Dict[Tuple[str, ...], Counter] = {
            cols: Counter() for cols in self.zero_additive_specs
        }
        self.zero_global_scores = np.zeros(len(self.candidates), dtype=np.float32)
        self.global_top: List[str] = []
        self.hot_penalty_items: List[str] = []

    def fit(self, train_df: pd.DataFrame) -> "ShortHistoryUserPriorHybridRecommender":
        self.base_model.fit(train_df)
        for length, model_config in self.length_base_configs.items():
            self.length_base_models[length] = HybridRecommender(
                config=model_config,
                candidates=self.candidates,
                user_df=self.user_df,
                item_df=self.item_df,
            ).fit(train_df)
        target_counts = Counter(train_df["target_iid"].astype(str))
        self.global_top = [iid for iid, _ in target_counts.most_common(len(self.candidates))]
        denom = math.log1p(max(target_counts.values())) if target_counts else 1.0
        if denom <= 0:
            denom = 1.0
        for iid, count in target_counts.items():
            idx = self.candidate_to_idx.get(iid)
            if idx is not None:
                self.zero_global_scores[idx] = math.log1p(count) / denom
        self.hot_penalty_items = [
            iid for iid, _ in target_counts.most_common(int(self.config.get("hot_penalty_top_n", 0)))
        ]
        merged = train_df[["uid", "target_iid"]].merge(self.user_df, on="uid", how="left")
        for row in merged.itertuples(index=False):
            row_dict = row._asdict()
            target = str(row_dict["target_iid"])
            if all(col in row_dict and pd.notna(row_dict[col]) for col in self.group_cols):
                key = tuple(str(row_dict[col]) for col in self.group_cols)
                self.group_counts[key][target] += 1
            if all(col in row_dict and pd.notna(row_dict[col]) for col in self.fallback_cols):
                key = tuple(str(row_dict[col]) for col in self.fallback_cols)
                self.fallback_counts[key][target] += 1
            for cols in self.short_group_cols:
                if all(col in row_dict and pd.notna(row_dict[col]) for col in cols):
                    key = tuple(str(row_dict[col]) for col in cols)
                    self.short_counts[cols][key][target] += 1
            for cols in self.zero_additive_specs:
                if all(col in row_dict and pd.notna(row_dict[col]) for col in cols):
                    key = tuple(str(row_dict[col]) for col in cols)
                    self.zero_additive_counts[cols][key][target] += 1
                    self.zero_additive_totals[cols][key] += 1
        return self

    def _top_from_counter(self, counter: Optional[Counter], k: int) -> List[str]:
        out: List[str] = []
        if counter:
            for iid, _ in counter.most_common(200):
                if iid not in out:
                    out.append(iid)
                if len(out) >= k:
                    return out
        for iid in self.global_top:
            if iid not in out:
                out.append(iid)
            if len(out) >= k:
                break
        return out

    def _user_group_prediction(self, uid: str, k: int) -> List[str]:
        if self.user_lookup.empty or uid not in self.user_lookup.index:
            return self._top_from_counter(None, k)
        user_row = self.user_lookup.loc[uid]
        group_key = tuple(str(user_row[col]) for col in self.group_cols)
        group_counter = self.group_counts.get(group_key)
        if group_counter and sum(group_counter.values()) >= self.min_group_count:
            return self._top_from_counter(group_counter, k)
        fallback_key = tuple(str(user_row[col]) for col in self.fallback_cols)
        fallback_counter = self.fallback_counts.get(fallback_key)
        if fallback_counter and sum(fallback_counter.values()) >= self.min_group_count:
            return self._top_from_counter(fallback_counter, k)
        return self._top_from_counter(None, k)

    def _add_short_user_prior(self, uid: str, scores: np.ndarray) -> bool:
        if self.short_user_weight <= 0 or self.user_lookup.empty or uid not in self.user_lookup.index:
            return False
        user_row = self.user_lookup.loc[uid]
        applied = False
        for group_idx, cols in enumerate(self.short_group_cols):
            if not all(col in user_row.index and pd.notna(user_row[col]) for col in cols):
                continue
            counter = self.short_counts.get(cols, {}).get(tuple(str(user_row[col]) for col in cols))
            if not counter or sum(counter.values()) < self.min_short_group_count:
                continue
            denom = math.log1p(max(counter.values()))
            if denom <= 0:
                continue
            group_weight = self.short_group_weights[group_idx] if group_idx < len(self.short_group_weights) else 1.0
            for iid, count in counter.most_common(self.short_user_pool):
                idx = self.candidate_to_idx.get(iid)
                if idx is not None:
                    scores[idx] += self.short_user_weight * group_weight * math.log1p(count) / denom
                    applied = True
        return applied

    def _zero_additive_prediction(self, uid: str, k: int) -> List[str]:
        if not self.zero_additive_specs:
            return self._user_group_prediction(uid, k)
        scores = self.zero_additive_global_weight * self.zero_global_scores.copy()
        if not self.user_lookup.empty and uid in self.user_lookup.index:
            user_row = self.user_lookup.loc[uid]
            for group_idx, cols in enumerate(self.zero_additive_specs):
                if not all(col in user_row.index and pd.notna(user_row[col]) for col in cols):
                    continue
                key = tuple(str(user_row[col]) for col in cols)
                if self.zero_additive_totals.get(cols, Counter()).get(key, 0) < self.zero_additive_min_count:
                    continue
                counter = self.zero_additive_counts.get(cols, {}).get(key)
                if not counter:
                    continue
                denom = math.log1p(max(counter.values()))
                if denom <= 0:
                    continue
                group_weight = (
                    self.zero_additive_weights[group_idx]
                    if group_idx < len(self.zero_additive_weights)
                    else 1.0
                )
                for iid, count in counter.most_common(self.zero_additive_pool):
                    idx = self.candidate_to_idx.get(iid)
                    if idx is not None:
                        scores[idx] += group_weight * math.log1p(count) / denom
        order = np.lexsort((np.arange(len(scores)), -scores))
        return [self.candidates[i] for i in order[:k]]

    def _apply_hot_penalty(self, scores: np.ndarray, raw_len: int) -> None:
        penalty = float(self.config.get("hot_penalty_weight", 0.0))
        if penalty <= 0:
            return
        if bool(self.config.get("hot_penalty_short_only", True)) and raw_len not in self.short_lengths:
            return
        for iid in self.hot_penalty_items:
            idx = self.candidate_to_idx.get(iid)
            if idx is not None:
                scores[idx] -= penalty

    def predict_row(self, row: pd.Series, k: int = 10) -> List[str]:
        raw_hist = parse_sequence(row.get("item_seq_raw", ""))
        raw_len = len(raw_hist)
        uid = str(row.get("uid", ""))
        if raw_len == 0:
            if self.zero_additive_specs:
                return self._zero_additive_prediction(uid, k)
            return self._user_group_prediction(uid, k)
        override_model = self.length_base_models.get(raw_len)
        if (
            override_model is None
            and raw_len not in self.short_lengths
            and float(self.config.get("hot_penalty_weight", 0.0)) <= 0
        ):
            return self.base_model.predict_row(row, k=k)
        score_model = override_model or self.base_model
        scores = score_model.score_row(row)
        if raw_len in self.short_lengths:
            self._add_short_user_prior(uid, scores)
        self._apply_hot_penalty(scores, raw_len)
        order = np.lexsort((np.arange(len(scores)), -scores))
        return [self.candidates[i] for i in order[:k]]


@dataclass
class NeuralZeroUserFusionRecommender:
    """v12: fuse a small PyTorch user-feature tower only for zero-history users."""

    config: Dict[str, Any]
    candidates: List[str]
    user_df: pd.DataFrame
    item_df: pd.DataFrame

    def __post_init__(self) -> None:
        base_config = dict(v8_zero_additive_config())
        base_config.update(self.config.get("base_config", {}))
        self.base_model = ShortHistoryUserPriorHybridRecommender(
            config=base_config,
            candidates=self.candidates,
            user_df=self.user_df,
            item_df=self.item_df,
        )
        self.user_cols = [c for c in self.user_df.columns if c.startswith("u_cat_")]
        self.user_lookup = self.user_df.set_index("uid") if "uid" in self.user_df.columns else pd.DataFrame()
        self.candidate_to_idx = {iid: i for i, iid in enumerate(self.candidates)}
        self.vocab_maps: Dict[str, Dict[str, int]] = {}
        self.cardinals: List[int] = []
        self.model: Any = None
        self.models: List[Any] = []
        self.short_seq_models: List[Any] = []
        self.torch: Any = None
        self.device: Any = None

    def _init_torch(self):
        try:
            import torch
            import torch.nn as nn
            from torch.utils.data import DataLoader, Dataset
        except ImportError as exc:
            raise RuntimeError(
                "PyTorch is required for neural_zero_user_fusion. "
                "Run with E:\\desktop\\Artificial-Intelligence\\Pytorch\\.venv\\Scripts\\python.exe"
            ) from exc
        return torch, nn, DataLoader, Dataset

    def _build_vocab(self) -> None:
        self.vocab_maps = {}
        self.cardinals = []
        for col in self.user_cols:
            values = sorted(self.user_df[col].astype(str).unique().tolist())
            mapping = {value: i + 1 for i, value in enumerate(values)}
            self.vocab_maps[col] = mapping
            self.cardinals.append(len(mapping) + 1)

    def _user_feature_indices(self, uid: str) -> List[int]:
        if self.user_lookup.empty or uid not in self.user_lookup.index:
            return [0] * len(self.user_cols)
        row = self.user_lookup.loc[uid]
        return [self.vocab_maps[col].get(str(row[col]), 0) for col in self.user_cols]

    def fit(self, train_df: pd.DataFrame) -> "NeuralZeroUserFusionRecommender":
        self.base_model.fit(train_df)
        torch, nn, DataLoader, Dataset = self._init_torch()
        self.torch = torch
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._build_vocab()
        user_cols = list(self.user_cols)
        vocab_maps = self.vocab_maps
        candidate_to_idx = self.candidate_to_idx
        user_df = self.user_df

        class _UserDataset(Dataset):
            def __init__(self, df: pd.DataFrame) -> None:
                merged = df[["uid", "target_iid"]].merge(user_df, on="uid", how="left")
                xs: List[List[int]] = []
                ys: List[int] = []
                for row in merged.itertuples(index=False):
                    row_dict = row._asdict()
                    target_idx = candidate_to_idx.get(str(row_dict["target_iid"]))
                    if target_idx is None:
                        continue
                    xs.append([vocab_maps[col].get(str(row_dict.get(col)), 0) for col in user_cols])
                    ys.append(target_idx)
                self.x = torch.tensor(xs, dtype=torch.long)
                self.y = torch.tensor(ys, dtype=torch.long)

            def __len__(self) -> int:
                return int(len(self.y))

            def __getitem__(self, idx: int):
                return self.x[idx], self.y[idx]

        class _UserTower(nn.Module):
            def __init__(self, cardinals: List[int], n_items: int) -> None:
                super().__init__()
                emb_dim = int(self_config.get("emb_dim", 16))
                hidden = int(self_config.get("hidden_dim", 384))
                dropout = float(self_config.get("dropout", 0.25))
                self.embs = nn.ModuleList([nn.Embedding(card, emb_dim) for card in cardinals])
                self.net = nn.Sequential(
                    nn.Linear(len(cardinals) * emb_dim, hidden),
                    nn.ReLU(),
                    nn.BatchNorm1d(hidden),
                    nn.Dropout(dropout),
                    nn.Linear(hidden, hidden),
                    nn.ReLU(),
                    nn.BatchNorm1d(hidden),
                    nn.Dropout(dropout),
                    nn.Linear(hidden, n_items),
                )

            def forward(self, x):
                z = torch.cat([emb(x[:, i]) for i, emb in enumerate(self.embs)], dim=1)
                return self.net(z)

        class _ShortSeqDataset(Dataset):
            def __init__(self, df: pd.DataFrame) -> None:
                user_lookup = user_df.set_index("uid") if "uid" in user_df.columns else pd.DataFrame()
                max_len = int(self_config.get("short_seq_max_len", 3))
                aug_lengths = tuple(int(x) for x in self_config.get("short_seq_aug_lengths", (1, 3)))
                xs_user: List[List[int]] = []
                xs_seq: List[List[int]] = []
                ys: List[int] = []
                for row in df.itertuples(index=False):
                    row_dict = row._asdict()
                    target_idx = candidate_to_idx.get(str(row_dict["target_iid"]))
                    hist = parse_sequence(row_dict.get("item_seq_raw", ""))
                    if target_idx is None or not hist:
                        continue
                    uid = str(row_dict["uid"])
                    if not user_lookup.empty and uid in user_lookup.index:
                        user_row = user_lookup.loc[uid]
                        user_features = [
                            vocab_maps[col].get(str(user_row.get(col)), 0)
                            for col in user_cols
                        ]
                    else:
                        user_features = [0] * len(user_cols)

                    lengths: List[int] = []
                    for length in aug_lengths:
                        if len(hist) >= length:
                            lengths.append(length)
                    if len(hist) == 1:
                        lengths.append(1)
                    if len(hist) == 2:
                        lengths.append(2)

                    for length in sorted(set(lengths)):
                        seq = hist[-length:]
                        seq_idx = [
                            candidate_to_idx[item] + 1
                            if item in candidate_to_idx
                            else 0
                            for item in seq
                        ]
                        seq_idx = ([0] * (max_len - len(seq_idx)) + seq_idx)[-max_len:]
                        xs_user.append(user_features)
                        xs_seq.append(seq_idx)
                        ys.append(target_idx)
                self.user_x = torch.tensor(xs_user, dtype=torch.long)
                self.seq_x = torch.tensor(xs_seq, dtype=torch.long)
                self.y = torch.tensor(ys, dtype=torch.long)

            def __len__(self) -> int:
                return int(len(self.y))

            def __getitem__(self, idx: int):
                return self.user_x[idx], self.seq_x[idx], self.y[idx]

        class _ShortSeqTower(nn.Module):
            def __init__(self, cardinals: List[int], n_items: int) -> None:
                super().__init__()
                user_emb_dim = int(self_config.get("short_seq_user_emb_dim", 12))
                item_emb_dim = int(self_config.get("short_seq_item_emb_dim", 48))
                hidden = int(self_config.get("short_seq_hidden_dim", 512))
                dropout = float(self_config.get("short_seq_dropout", 0.3))
                max_len = int(self_config.get("short_seq_max_len", 3))
                self.user_embs = nn.ModuleList([nn.Embedding(card, user_emb_dim) for card in cardinals])
                self.item_emb = nn.Embedding(n_items + 1, item_emb_dim, padding_idx=0)
                self.pos_emb = nn.Embedding(max_len, item_emb_dim)
                input_dim = len(cardinals) * user_emb_dim + max_len * item_emb_dim
                self.net = nn.Sequential(
                    nn.Linear(input_dim, hidden),
                    nn.ReLU(),
                    nn.BatchNorm1d(hidden),
                    nn.Dropout(dropout),
                    nn.Linear(hidden, hidden),
                    nn.ReLU(),
                    nn.BatchNorm1d(hidden),
                    nn.Dropout(dropout),
                    nn.Linear(hidden, n_items),
                )

            def forward(self, user_x, seq_x):
                user_z = torch.cat(
                    [emb(user_x[:, i]) for i, emb in enumerate(self.user_embs)],
                    dim=1,
                )
                positions = torch.arange(seq_x.shape[1], device=seq_x.device).unsqueeze(0)
                seq_z = (self.item_emb(seq_x) + self.pos_emb(positions)).reshape(seq_x.shape[0], -1)
                return self.net(torch.cat([user_z, seq_z], dim=1))

        self_config = self.config
        dataset = _UserDataset(train_df)
        if len(dataset) == 0:
            return self
        loader = DataLoader(
            dataset,
            batch_size=int(self.config.get("batch_size", 1024)),
            shuffle=True,
            num_workers=0,
            pin_memory=bool(torch.cuda.is_available()),
        )
        loss_fn = nn.CrossEntropyLoss()
        epochs = int(self.config.get("epochs", 14))
        seeds = self.config.get("ensemble_seeds")
        if seeds is None:
            seeds = (int(self.config.get("seed", 42)),)
        self.models = []
        for seed_value in seeds:
            seed = int(seed_value)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            model = _UserTower(self.cardinals, len(self.candidates)).to(self.device)
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=float(self.config.get("lr", 2e-3)),
                weight_decay=float(self.config.get("weight_decay", 1e-4)),
            )
            for _ in range(epochs):
                model.train()
                for x, y in loader:
                    x = x.to(self.device, non_blocking=True)
                    y = y.to(self.device, non_blocking=True)
                    optimizer.zero_grad(set_to_none=True)
                    loss = loss_fn(model(x), y)
                    loss.backward()
                    optimizer.step()
            model.eval()
            self.models.append(model)
        self.model = self.models[0] if self.models else None
        self.short_seq_models = []
        if bool(self.config.get("short_seq_enabled", False)):
            short_dataset = _ShortSeqDataset(train_df)
            if len(short_dataset) > 0:
                short_loader = DataLoader(
                    short_dataset,
                    batch_size=int(self.config.get("short_seq_batch_size", 1024)),
                    shuffle=True,
                    num_workers=0,
                    pin_memory=bool(torch.cuda.is_available()),
                )
                short_epochs = int(self.config.get("short_seq_epochs", 12))
                short_seeds = tuple(int(x) for x in self.config.get("short_seq_seeds", (42,)))
                for seed in short_seeds:
                    np.random.seed(seed)
                    torch.manual_seed(seed)
                    if torch.cuda.is_available():
                        torch.cuda.manual_seed_all(seed)
                    short_model = _ShortSeqTower(self.cardinals, len(self.candidates)).to(self.device)
                    short_optimizer = torch.optim.AdamW(
                        short_model.parameters(),
                        lr=float(self.config.get("short_seq_lr", 2e-3)),
                        weight_decay=float(self.config.get("short_seq_weight_decay", 1e-4)),
                    )
                    for _ in range(short_epochs):
                        short_model.train()
                        for user_x, seq_x, y in short_loader:
                            user_x = user_x.to(self.device, non_blocking=True)
                            seq_x = seq_x.to(self.device, non_blocking=True)
                            y = y.to(self.device, non_blocking=True)
                            short_optimizer.zero_grad(set_to_none=True)
                            loss = loss_fn(short_model(user_x, seq_x), y)
                            loss.backward()
                            short_optimizer.step()
                    short_model.eval()
                    self.short_seq_models.append(short_model)
        return self

    def _neural_top_for_model(self, model: Any, uid: str, pool: int) -> List[str]:
        if model is None or self.torch is None:
            return []
        torch = self.torch
        x = torch.tensor([self._user_feature_indices(uid)], dtype=torch.long, device=self.device)
        with torch.no_grad():
            logits = model(x)[0]
            top_idx = torch.topk(logits, k=min(pool, len(self.candidates))).indices.detach().cpu().numpy()
        return [self.candidates[int(i)] for i in top_idx]

    def _neural_top(self, uid: str, pool: int) -> List[str]:
        return self._neural_top_for_model(self.model, uid, pool)

    def _short_sequence_indices(self, hist: Sequence[str]) -> List[int]:
        max_len = int(self.config.get("short_seq_max_len", 3))
        seq_idx = [
            self.candidate_to_idx[item] + 1
            if item in self.candidate_to_idx
            else 0
            for item in hist[-max_len:]
        ]
        return ([0] * (max_len - len(seq_idx)) + seq_idx)[-max_len:]

    def _short_seq_top_for_model(self, model: Any, uid: str, hist: Sequence[str], pool: int) -> List[str]:
        if model is None or self.torch is None:
            return []
        torch = self.torch
        user_x = torch.tensor([self._user_feature_indices(uid)], dtype=torch.long, device=self.device)
        seq_x = torch.tensor([self._short_sequence_indices(hist)], dtype=torch.long, device=self.device)
        with torch.no_grad():
            logits = model(user_x, seq_x)[0]
            top_idx = torch.topk(logits, k=min(pool, len(self.candidates))).indices.detach().cpu().numpy()
        return [self.candidates[int(i)] for i in top_idx]

    def _fuse_zero_predictions(self, uid: str, k: int) -> List[str]:
        base_pool = int(self.config.get("base_pool", 100))
        neural_pool = int(self.config.get("neural_pool", 50))
        alpha = float(self.config.get("neural_rank_weight", 1.2))
        base_items = self.base_model._zero_additive_prediction(uid, base_pool)
        scores: Dict[str, float] = {}
        for rank, iid in enumerate(base_items, start=1):
            scores[iid] = scores.get(iid, 0.0) + 1.0 / math.log2(rank + 1)
        seed_weights = tuple(float(x) for x in self.config.get("ensemble_seed_weights", ()))
        models = self.models if self.models else ([self.model] if self.model is not None else [])
        for model_idx, model in enumerate(models):
            model_weight = seed_weights[model_idx] if model_idx < len(seed_weights) else 1.0
            for rank, iid in enumerate(self._neural_top_for_model(model, uid, neural_pool), start=1):
                scores[iid] = scores.get(iid, 0.0) + alpha * model_weight / math.log2(rank + 1)
        ordered = sorted(scores, key=lambda iid: (-scores[iid], iid))
        return ordered[:k]

    def _fuse_short_predictions(self, row: pd.Series, hist: Sequence[str], k: int) -> List[str]:
        uid = str(row.get("uid", ""))
        base_pool = int(self.config.get("short_seq_base_pool", 100))
        neural_pool = int(self.config.get("short_seq_neural_pool", 30))
        alpha = float(self.config.get("short_seq_rank_weight", 0.65))
        base_items = self.base_model.predict_row(row, k=base_pool)
        scores: Dict[str, float] = {}
        for rank, iid in enumerate(base_items, start=1):
            scores[iid] = scores.get(iid, 0.0) + 1.0 / math.log2(rank + 1)
        seed_weights = tuple(float(x) for x in self.config.get("short_seq_seed_weights", ()))
        for model_idx, model in enumerate(self.short_seq_models):
            model_weight = seed_weights[model_idx] if model_idx < len(seed_weights) else 1.0
            for rank, iid in enumerate(self._short_seq_top_for_model(model, uid, hist, neural_pool), start=1):
                scores[iid] = scores.get(iid, 0.0) + alpha * model_weight / math.log2(rank + 1)
        ordered = sorted(scores, key=lambda iid: (-scores[iid], iid))
        return ordered[:k]

    def predict_row(self, row: pd.Series, k: int = 10) -> List[str]:
        hist = parse_sequence(row.get("item_seq_raw", ""))
        raw_len = len(hist)
        uid = str(row.get("uid", ""))
        if raw_len == 0:
            return self._fuse_zero_predictions(uid, k)
        short_lengths = set(int(x) for x in self.config.get("short_seq_lengths", (1, 2, 3)))
        if self.short_seq_models and raw_len in short_lengths:
            return self._fuse_short_predictions(row, hist, k)
        return self.base_model.predict_row(row, k=k)


@dataclass
class LengthAwareHybridRecommender:
    """Length-aware rank-fusion recommender for v2."""

    config: Dict[str, Any]
    candidates: List[str]
    user_df: pd.DataFrame
    item_df: pd.DataFrame

    def __post_init__(self) -> None:
        self.candidate_to_idx = {iid: i for i, iid in enumerate(self.candidates)}
        self.pop_scores = np.zeros(len(self.candidates), dtype=np.float32)
        self.item_prior_scores = np.zeros(len(self.candidates), dtype=np.float32)
        self.transition_raw: DefaultDict[str, Counter] = defaultdict(Counter)
        self.transition_dedup: DefaultDict[str, Counter] = defaultdict(Counter)
        self.user_pop = UserFeaturePopularity(
            candidates=self.candidates,
            user_df=self.user_df,
            min_group_count=int(self.config.get("min_user_group_count", 20)),
        )
        self.item_lookup = self.item_df.set_index("iid") if "iid" in self.item_df.columns else pd.DataFrame()

    def fit(self, train_df: pd.DataFrame) -> "LengthAwareHybridRecommender":
        target_counts = Counter(train_df["target_iid"].astype(str))
        self._build_popularity(target_counts)
        self._build_item_prior(target_counts)
        self._build_transition_map(train_df, "item_seq_raw", self.transition_raw)
        self._build_transition_map(train_df, "item_seq_dedup", self.transition_dedup)
        self.user_pop.fit(train_df)
        return self

    def _build_popularity(self, target_counts: Counter) -> None:
        denom = math.log1p(max(target_counts.values())) if target_counts else 1.0
        for iid, count in target_counts.items():
            idx = self.candidate_to_idx.get(iid)
            if idx is not None:
                self.pop_scores[idx] = math.log1p(count) / denom

    def _build_item_prior(self, target_counts: Counter) -> None:
        if self.item_df.empty:
            return
        item_cols = [c for c in self.item_df.columns if c.startswith("i_cat_") or c.startswith("i_bucket_")]
        feature_counts: DefaultDict[Tuple[str, str], float] = defaultdict(float)
        for iid, count in target_counts.items():
            if iid not in self.item_lookup.index:
                continue
            row = self.item_lookup.loc[iid]
            for col in item_cols:
                value = row.get(col)
                if pd.notna(value):
                    feature_counts[(col, str(value))] += float(count)
        denom = math.log1p(max(feature_counts.values())) if feature_counts else 1.0
        for iid, idx in self.candidate_to_idx.items():
            if iid not in self.item_lookup.index:
                continue
            row = self.item_lookup.loc[iid]
            score = 0.0
            for col in item_cols:
                value = row.get(col)
                score += feature_counts.get((col, str(value)), 0.0)
            if score > 0:
                self.item_prior_scores[idx] = math.log1p(score) / denom

    def _build_transition_map(self, train_df: pd.DataFrame, seq_col: str, store: DefaultDict[str, Counter]) -> None:
        if seq_col not in train_df.columns:
            return
        max_hist = int(self.config.get("fit_max_hist", 80))
        fit_decay = float(self.config.get("fit_decay", 0.95))
        for row in train_df.itertuples(index=False):
            row_dict = row._asdict()
            target = str(row_dict["target_iid"])
            hist = parse_sequence(row_dict.get(seq_col, ""))
            for pos, item in enumerate(reversed(hist[-max_hist:])):
                store[item][target] += fit_decay ** pos

    def _segment_params(self, raw_len: int) -> Dict[str, float]:
        segments = self.config.get("segments", {})
        if raw_len <= 0:
            key = "zero"
        elif raw_len <= 3:
            key = "short"
        elif raw_len <= 30:
            key = "medium"
        else:
            key = "long"
        params = dict(segments.get("default", {}))
        params.update(segments.get(key, {}))
        return params

    def _transition_scores(
        self,
        hist: List[str],
        transition_map: DefaultDict[str, Counter],
        max_hist: int,
        decay: float,
    ) -> np.ndarray:
        scores = np.zeros(len(self.candidates), dtype=np.float32)
        for pos, item in enumerate(reversed(hist[-max_hist:])):
            counter = transition_map.get(item)
            if not counter:
                continue
            hist_weight = decay ** pos
            for target, value in counter.items():
                idx = self.candidate_to_idx.get(target)
                if idx is not None:
                    scores[idx] += hist_weight * math.log1p(value)
        return scores

    def _repeat_scores(self, hist: List[str], max_hist: int, decay: float, cap: float) -> np.ndarray:
        scores = np.zeros(len(self.candidates), dtype=np.float32)
        for pos, item in enumerate(reversed(hist[-max_hist:])):
            idx = self.candidate_to_idx.get(item)
            if idx is not None:
                scores[idx] = min(cap, scores[idx] + (decay ** pos))
        return scores

    def predict_row(self, row: pd.Series, k: int = 10) -> List[str]:
        raw_hist = parse_sequence(row.get("item_seq_raw", ""))
        dedup_hist = parse_sequence(row.get("item_seq_dedup", ""))
        raw_len = len(raw_hist)
        params = self._segment_params(raw_len)
        max_hist = int(params.get("max_hist", self.config.get("max_hist", 50)))
        decay = float(params.get("decay", self.config.get("decay", 0.92)))
        repeat_cap = float(params.get("repeat_cap", 1.0))
        transition_source = str(params.get("transition_source", self.config.get("transition_source", "raw")))
        if transition_source == "dedup":
            trans_hist = dedup_hist
            transition_map = self.transition_dedup
        else:
            trans_hist = raw_hist
            transition_map = self.transition_raw
        repeat_hist = raw_hist if params.get("repeat_source", "raw") == "raw" else dedup_hist

        source_scores = {
            "pop": self.pop_scores,
            "item": self.item_prior_scores,
            "transition": self._transition_scores(trans_hist, transition_map, max_hist=max_hist, decay=decay),
            "repeat": self._repeat_scores(repeat_hist, max_hist=max_hist, decay=decay, cap=repeat_cap),
            "user": self.user_pop.score_for_uid(str(row.get("uid", "")), strength=1.0),
        }
        weights = {
            "pop": float(params.get("pop", 1.0)),
            "item": float(params.get("item", 0.0)),
            "transition": float(params.get("transition", 0.0)),
            "repeat": float(params.get("repeat", 0.0)),
            "user": float(params.get("user", 0.0)),
        }
        return rank_fusion_topk(
            source_scores,
            weights,
            self.candidates,
            k=k,
            pool=int(params.get("fusion_pool", self.config.get("fusion_pool", 300))),
            tie_scores=self.pop_scores,
        )


def v1_reference_config() -> Dict[str, Any]:
    return {
        "name": "v1_raw_transition_repeat_reference",
        "model": "legacy_hybrid",
        "seq_col": "item_seq_raw",
        "pop_weight": 0.8,
        "transition_weight": 0.8,
        "repeat_weight": 0.45,
        "item_feature_weight": 0.05,
        "max_hist": 50,
        "decay": 0.94,
        "exclude_history": False,
    }


def v2_weight_tuned_config() -> Dict[str, Any]:
    return {
        "name": "v2_weight_tuned_repeat_heavy",
        "model": "legacy_hybrid",
        "seq_col": "item_seq_raw",
        "pop_weight": 0.6,
        "transition_weight": 0.6,
        "repeat_weight": 0.65,
        "item_feature_weight": 0.0,
        "max_hist": 50,
        "decay": 0.94,
        "exclude_history": False,
    }


def v5_repeat_tuned_config() -> Dict[str, Any]:
    return {
        "name": "v5_nonzero_repeat100_trans035",
        "model": "legacy_hybrid",
        "seq_col": "item_seq_raw",
        "pop_weight": 0.7,
        "transition_weight": 0.35,
        "repeat_weight": 1.0,
        "item_feature_weight": 0.0,
        "max_hist": 80,
        "decay": 0.94,
        "exclude_history": False,
    }


def v6_short_user_config(
    name: str,
    max_hist: int,
    decay: float = 0.94,
    pop_weight: float = 0.7,
    transition_weight: float = 0.35,
    repeat_weight: float = 1.0,
    short_user_weight: float = 0.08,
    min_short_group_count: int = 20,
    hot_penalty_weight: float = 0.0,
) -> Dict[str, Any]:
    base_config = v5_repeat_tuned_config()
    base_config.update(
        {
            "name": f"{name}_base",
            "pop_weight": pop_weight,
            "transition_weight": transition_weight,
            "repeat_weight": repeat_weight,
            "max_hist": max_hist,
            "decay": decay,
        }
    )
    return {
        "name": name,
        "model": "short_history_user_prior_hybrid",
        "group_cols": ("u_cat_01", "u_cat_02"),
        "fallback_cols": ("u_cat_01",),
        "min_group_count": 3,
        "short_group_cols": (
            ("u_cat_01", "u_cat_02"),
            ("u_cat_01", "u_cat_06"),
            ("u_cat_02", "u_cat_06"),
        ),
        "short_group_weights": (1.0, 0.55, 0.55),
        "min_short_group_count": min_short_group_count,
        "short_user_weight": short_user_weight,
        "short_lengths": (1, 2, 3),
        "short_user_pool": 120,
        "hot_penalty_top_n": 1 if hot_penalty_weight > 0 else 0,
        "hot_penalty_weight": hot_penalty_weight,
        "hot_penalty_short_only": True,
        "base_config": base_config,
    }


def v8_zero_additive_config() -> Dict[str, Any]:
    config = v6_short_user_config(
        "v8_zero_additive_user_fusion",
        max_hist=120,
        pop_weight=0.55,
        transition_weight=0.1,
        repeat_weight=1.1,
        short_user_weight=0.05,
        min_short_group_count=20,
    )
    config.update(
        {
            "zero_additive_specs": (
                ("u_cat_01", "u_cat_02"),
                ("u_cat_01", "u_cat_02", "u_cat_06"),
                ("u_cat_01",),
            ),
            "zero_additive_weights": (0.25, 0.35, 0.25),
            "zero_additive_global_weight": 0.4,
            "zero_additive_min_count": 20,
            "zero_additive_pool": 300,
        }
    )
    return config


def v9_zero_m10_len2_config() -> Dict[str, Any]:
    config = v8_zero_additive_config()
    config.update(
        {
            "name": "v9_zero_m10_len2_segment",
            "zero_additive_min_count": 10,
            "length_base_configs": {
                2: {
                    "name": "v9_len2_pop11_trans005_repeat0",
                    "model": "legacy_hybrid",
                    "seq_col": "item_seq_raw",
                    "pop_weight": 1.1,
                    "transition_weight": 0.05,
                    "repeat_weight": 0.0,
                    "item_feature_weight": 0.0,
                    "max_hist": 3,
                    "decay": 0.94,
                    "exclude_history": False,
                }
            },
        }
    )
    base_config = dict(config["base_config"])
    base_config["name"] = "v9_zero_m10_len2_segment_base"
    config["base_config"] = base_config
    return config


def v10_zero_m12_short08_config() -> Dict[str, Any]:
    config = v9_zero_m10_len2_config()
    config.update(
        {
            "name": "v10_zero_m12_short08_segment",
            "zero_additive_min_count": 12,
            "short_user_weight": 0.08,
            "min_short_group_count": 30,
        }
    )
    base_config = dict(config["base_config"])
    base_config["name"] = "v10_zero_m12_short08_segment_base"
    config["base_config"] = base_config
    length_base_configs = {
        int(length): dict(model_config)
        for length, model_config in config.get("length_base_configs", {}).items()
    }
    if 2 in length_base_configs:
        length_base_configs[2]["name"] = "v10_len2_pop11_trans005_repeat0"
    config["length_base_configs"] = length_base_configs
    return config


def v12_neural_zero_user_config() -> Dict[str, Any]:
    config = v8_zero_additive_config()
    config.update(
        {
            "name": "v12_neural_zero_user_fusion",
            "model": "neural_zero_user_fusion",
            "seed": 42,
            "emb_dim": 16,
            "hidden_dim": 384,
            "dropout": 0.25,
            "lr": 2e-3,
            "weight_decay": 1e-4,
            "epochs": 14,
            "batch_size": 1024,
            "base_pool": 100,
            "neural_pool": 50,
            "neural_rank_weight": 1.2,
            "base_config": v8_zero_additive_config(),
        }
    )
    return config


def v14_neural_zero_user_ensemble_config() -> Dict[str, Any]:
    config = v12_neural_zero_user_config()
    config.update(
        {
            "name": "v14_neural_zero_user_3seed_ensemble",
            "ensemble_seeds": (42, 123, 11),
            "neural_pool": 20,
            "neural_rank_weight": 0.6,
            "base_pool": 100,
        }
    )
    return config


def v15_neural_short_history_fusion_config() -> Dict[str, Any]:
    config = v14_neural_zero_user_ensemble_config()
    config.update(
        {
            "name": "v15_neural_zero_shortseq_fusion",
            "short_seq_enabled": True,
            "short_seq_lengths": (1, 2, 3),
            "short_seq_seeds": (42,),
            "short_seq_epochs": 12,
            "short_seq_batch_size": 1024,
            "short_seq_max_len": 3,
            "short_seq_aug_lengths": (1, 3),
            "short_seq_user_emb_dim": 12,
            "short_seq_item_emb_dim": 48,
            "short_seq_hidden_dim": 512,
            "short_seq_dropout": 0.3,
            "short_seq_lr": 2e-3,
            "short_seq_weight_decay": 1e-4,
            "short_seq_base_pool": 100,
            "short_seq_neural_pool": 30,
            "short_seq_rank_weight": 0.65,
        }
    )
    return config


def v16_neural_zero_dropout35_config() -> Dict[str, Any]:
    config = v12_neural_zero_user_config()
    config.update(
        {
            "name": "v16_neural_zero_dropout35_3seed",
            "ensemble_seeds": (42, 123, 314),
            "dropout": 0.35,
            "neural_pool": 15,
            "neural_rank_weight": 0.55,
            "base_pool": 100,
        }
    )
    return config


def v17_conservative_shortseq_config() -> Dict[str, Any]:
    config = v16_neural_zero_dropout35_config()
    config.update(
        {
            "name": "v17_neural_zero_shortseq_conservative",
            "short_seq_enabled": True,
            "short_seq_lengths": (1, 2, 3),
            "short_seq_seeds": (42,),
            "short_seq_epochs": 8,
            "short_seq_batch_size": 1024,
            "short_seq_max_len": 3,
            "short_seq_aug_lengths": (1, 3),
            "short_seq_user_emb_dim": 12,
            "short_seq_item_emb_dim": 48,
            "short_seq_hidden_dim": 384,
            "short_seq_dropout": 0.45,
            "short_seq_lr": 1e-3,
            "short_seq_weight_decay": 3e-4,
            "short_seq_base_pool": 20,
            "short_seq_neural_pool": 10,
            "short_seq_rank_weight": 0.08,
        }
    )
    return config


def candidate_configs() -> List[Dict[str, Any]]:
    configs = [
        v1_reference_config(),
        v2_weight_tuned_config(),
        {
            "name": "v4_zero_user_group_v2_nonzero",
            "model": "zero_user_group_hybrid",
            "group_cols": ("u_cat_01", "u_cat_02"),
            "fallback_cols": ("u_cat_01",),
            "min_group_count": 3,
            "base_config": v2_weight_tuned_config(),
        },
        {
            "name": "v5_zero_group_repeat100_trans035",
            "model": "zero_user_group_hybrid",
            "group_cols": ("u_cat_01", "u_cat_02"),
            "fallback_cols": ("u_cat_01",),
            "min_group_count": 3,
            "base_config": v5_repeat_tuned_config(),
        },
        v6_short_user_config(
            "v6_hist100_repeat_user_short",
            max_hist=100,
            short_user_weight=0.06,
            min_short_group_count=20,
        ),
        v6_short_user_config(
            "v6_hist120_repeat_user_short",
            max_hist=120,
            short_user_weight=0.06,
            min_short_group_count=20,
        ),
        v6_short_user_config(
            "v6_hist150_repeat_user_short",
            max_hist=150,
            short_user_weight=0.06,
            min_short_group_count=20,
        ),
        v6_short_user_config(
            "v6_weight_probe_pop055_trans01_rep11",
            max_hist=120,
            pop_weight=0.55,
            transition_weight=0.1,
            repeat_weight=1.1,
            short_user_weight=0.05,
            min_short_group_count=20,
        ),
        v8_zero_additive_config(),
        v9_zero_m10_len2_config(),
        v10_zero_m12_short08_config(),
        v6_short_user_config(
            "v6_weight_probe_pop085_trans05_rep08",
            max_hist=100,
            pop_weight=0.85,
            transition_weight=0.5,
            repeat_weight=0.8,
            short_user_weight=0.05,
            min_short_group_count=20,
        ),
        v6_short_user_config(
            "v6_decay092_repeat_user_short",
            max_hist=120,
            decay=0.92,
            short_user_weight=0.05,
            min_short_group_count=20,
        ),
        v6_short_user_config(
            "v6_decay096_repeat_user_short",
            max_hist=120,
            decay=0.96,
            short_user_weight=0.05,
            min_short_group_count=20,
        ),
        v6_short_user_config(
            "v6_hot_penalty_probe",
            max_hist=100,
            short_user_weight=0.05,
            min_short_group_count=20,
            hot_penalty_weight=0.08,
        ),
        {
            "name": "v2_weight_tuned_longer_history",
            "model": "legacy_hybrid",
            "seq_col": "item_seq_raw",
            "pop_weight": 0.6,
            "transition_weight": 0.6,
            "repeat_weight": 0.65,
            "item_feature_weight": 0.0,
            "max_hist": 80,
            "decay": 0.94,
            "exclude_history": False,
        },
        {
            "name": "cold_user_popularity",
            "model": "length_aware",
            "min_user_group_count": 25,
            "segments": {
                "default": {"pop": 1.0, "item": 0.08, "user": 0.1, "transition": 0.0, "repeat": 0.0},
                "zero": {"pop": 1.0, "item": 0.12, "user": 0.18, "transition": 0.0, "repeat": 0.0},
                "short": {"pop": 0.85, "item": 0.08, "user": 0.08, "transition": 0.65, "repeat": 0.12, "max_hist": 3},
                "medium": {"pop": 0.75, "item": 0.05, "user": 0.04, "transition": 0.9, "repeat": 0.22, "max_hist": 20},
                "long": {"pop": 0.65, "item": 0.04, "user": 0.02, "transition": 1.0, "repeat": 0.32, "max_hist": 50},
            },
        },
        {
            "name": "short_history_transition",
            "model": "length_aware",
            "segments": {
                "default": {"pop": 0.9, "item": 0.04, "user": 0.02, "transition": 0.75, "repeat": 0.12},
                "zero": {"pop": 1.0, "item": 0.05, "user": 0.06, "transition": 0.0, "repeat": 0.0},
                "short": {"pop": 0.72, "item": 0.04, "user": 0.03, "transition": 1.15, "repeat": 0.18, "max_hist": 3, "decay": 0.85},
                "medium": {"pop": 0.68, "item": 0.03, "user": 0.02, "transition": 1.05, "repeat": 0.24, "max_hist": 10, "decay": 0.9},
                "long": {"pop": 0.62, "item": 0.02, "user": 0.01, "transition": 1.05, "repeat": 0.38, "max_hist": 50, "decay": 0.94},
            },
        },
        {
            "name": "length_aware_rank_fusion",
            "model": "length_aware",
            "min_user_group_count": 20,
            "fusion_pool": 400,
            "segments": {
                "default": {"pop": 0.8, "item": 0.05, "user": 0.03, "transition": 0.9, "repeat": 0.2},
                "zero": {"pop": 1.0, "item": 0.1, "user": 0.14, "transition": 0.0, "repeat": 0.0},
                "short": {"pop": 0.78, "item": 0.06, "user": 0.06, "transition": 0.9, "repeat": 0.16, "max_hist": 3},
                "medium": {"pop": 0.7, "item": 0.04, "user": 0.03, "transition": 1.0, "repeat": 0.25, "max_hist": 25},
                "long": {"pop": 0.62, "item": 0.03, "user": 0.02, "transition": 1.1, "repeat": 0.34, "max_hist": 60},
            },
        },
        {
            "name": "length_aware_no_repeat_cap",
            "model": "length_aware",
            "segments": {
                "default": {"pop": 0.85, "item": 0.05, "user": 0.04, "transition": 0.95, "repeat": 0.1, "repeat_cap": 0.5},
                "zero": {"pop": 1.0, "item": 0.08, "user": 0.12, "transition": 0.0, "repeat": 0.0},
                "short": {"pop": 0.82, "item": 0.04, "user": 0.04, "transition": 0.85, "repeat": 0.08, "repeat_cap": 0.35, "max_hist": 3},
                "medium": {"pop": 0.74, "item": 0.03, "user": 0.02, "transition": 0.95, "repeat": 0.14, "repeat_cap": 0.5, "max_hist": 20},
                "long": {"pop": 0.66, "item": 0.02, "user": 0.01, "transition": 1.05, "repeat": 0.22, "repeat_cap": 0.75, "max_hist": 50},
            },
        },
        {
            "name": "length_aware_pop_heavy",
            "model": "length_aware",
            "segments": {
                "default": {"pop": 1.05, "item": 0.08, "user": 0.04, "transition": 0.55, "repeat": 0.08},
                "zero": {"pop": 1.15, "item": 0.12, "user": 0.08, "transition": 0.0, "repeat": 0.0},
                "short": {"pop": 0.95, "item": 0.08, "user": 0.04, "transition": 0.6, "repeat": 0.08, "max_hist": 3},
                "medium": {"pop": 0.82, "item": 0.05, "user": 0.02, "transition": 0.82, "repeat": 0.18, "max_hist": 20},
                "long": {"pop": 0.72, "item": 0.03, "user": 0.01, "transition": 0.98, "repeat": 0.3, "max_hist": 50},
            },
        },
        {
            "name": "dedup_transition_rank_fusion",
            "model": "length_aware",
            "transition_source": "dedup",
            "segments": {
                "default": {"pop": 0.8, "item": 0.04, "user": 0.03, "transition": 0.95, "repeat": 0.18},
                "zero": {"pop": 1.0, "item": 0.1, "user": 0.1, "transition": 0.0, "repeat": 0.0},
                "short": {"pop": 0.82, "item": 0.05, "user": 0.05, "transition": 0.8, "repeat": 0.12, "max_hist": 3},
                "medium": {"pop": 0.7, "item": 0.04, "user": 0.03, "transition": 1.05, "repeat": 0.22, "max_hist": 20},
                "long": {"pop": 0.64, "item": 0.03, "user": 0.02, "transition": 1.15, "repeat": 0.32, "max_hist": 50},
            },
        },
        {
            "name": "user_cold_low_transition",
            "model": "length_aware",
            "min_user_group_count": 10,
            "segments": {
                "default": {"pop": 0.88, "item": 0.05, "user": 0.08, "transition": 0.75, "repeat": 0.12},
                "zero": {"pop": 0.9, "item": 0.1, "user": 0.26, "transition": 0.0, "repeat": 0.0},
                "short": {"pop": 0.82, "item": 0.06, "user": 0.12, "transition": 0.65, "repeat": 0.1, "max_hist": 3},
                "medium": {"pop": 0.76, "item": 0.04, "user": 0.06, "transition": 0.9, "repeat": 0.2, "max_hist": 20},
                "long": {"pop": 0.68, "item": 0.03, "user": 0.03, "transition": 1.0, "repeat": 0.28, "max_hist": 50},
            },
        },
    ]
    if importlib.util.find_spec("torch") is not None:
        configs.insert(11, v12_neural_zero_user_config())
        configs.insert(12, v14_neural_zero_user_ensemble_config())
        # v15 is intentionally not in the default search: it improved internal
        # masked validation but failed official A-board validation.
        configs.insert(13, v16_neural_zero_dropout35_config())
        configs.insert(14, v17_conservative_shortseq_config())
    return configs


def build_recommender(config: Dict[str, Any], candidates: List[str], user_df: pd.DataFrame, item_df: pd.DataFrame):
    if config.get("model") == "zero_user_group_hybrid":
        return ZeroUserGroupHybridRecommender(config=config, candidates=candidates, user_df=user_df, item_df=item_df)
    if config.get("model") == "short_history_user_prior_hybrid":
        return ShortHistoryUserPriorHybridRecommender(config=config, candidates=candidates, user_df=user_df, item_df=item_df)
    if config.get("model") == "neural_zero_user_fusion":
        return NeuralZeroUserFusionRecommender(config=config, candidates=candidates, user_df=user_df, item_df=item_df)
    if config.get("model") == "length_aware":
        return LengthAwareHybridRecommender(config=config, candidates=candidates, user_df=user_df, item_df=item_df)
    return HybridRecommender(config=config, candidates=candidates, user_df=user_df, item_df=item_df)


def evaluate_config(
    fit_df: pd.DataFrame,
    val_df: pd.DataFrame,
    user_df: pd.DataFrame,
    item_df: pd.DataFrame,
    candidates: List[str],
    config: Dict[str, Any],
) -> Tuple[float, List[List[str]]]:
    model = build_recommender(config, candidates, user_df, item_df).fit(fit_df)
    predictions = [model.predict_row(row, k=10) for _, row in val_df.iterrows()]
    score = ndcg_at_k(predictions, val_df["target_iid"].astype(str).tolist(), k=10)
    return score, predictions


def evaluate_config_details(
    fit_df: pd.DataFrame,
    val_df: pd.DataFrame,
    masked_val_df: pd.DataFrame,
    bin_weights: Dict[str, float],
    user_df: pd.DataFrame,
    item_df: pd.DataFrame,
    candidates: List[str],
    config: Dict[str, Any],
) -> Dict[str, Any]:
    model = build_recommender(config, candidates, user_df, item_df).fit(fit_df)
    plain_predictions = [model.predict_row(row, k=10) for _, row in val_df.iterrows()]
    masked_predictions = [model.predict_row(row, k=10) for _, row in masked_val_df.iterrows()]
    plain_ndcg = ndcg_at_k(plain_predictions, val_df["target_iid"].astype(str).tolist(), k=10)
    masked_ndcg = ndcg_at_k(masked_predictions, masked_val_df["target_iid"].astype(str).tolist(), k=10)
    raw_lengths = val_df["item_seq_raw"].map(lambda x: len(parse_sequence(x)))
    natural_short_idx = [
        idx for idx, raw_len in enumerate(raw_lengths.tolist())
        if 1 <= int(raw_len) <= 3
    ]
    natural_short_ndcg = 0.0
    if natural_short_idx:
        natural_short_ndcg = ndcg_at_k(
            [plain_predictions[idx] for idx in natural_short_idx],
            val_df.iloc[natural_short_idx]["target_iid"].astype(str).tolist(),
            k=10,
        )
    by_bin = score_by_bin(
        masked_predictions,
        masked_val_df["target_iid"].astype(str).tolist(),
        masked_val_df["_eval_bin"].astype(str).tolist(),
    )
    test_weighted = weighted_bin_score(by_bin, bin_weights)
    return {
        "plain_val_ndcg@10": plain_ndcg,
        "masked_val_ndcg@10": masked_ndcg,
        "test_weighted_ndcg@10": test_weighted,
        "by_bin_ndcg@10": by_bin,
        "natural_short_val_ndcg@10": natural_short_ndcg,
        "natural_short_val_size": len(natural_short_idx),
    }


def run_recommendation(
    data_dir: str | Path,
    output_dir: str | Path,
    budget: int = 5,
    seed: int = 42,
    val_ratio: float = 0.12,
    time_limit: Optional[float] = None,
) -> Dict[str, Any]:
    timer = Timer()
    data_dir = Path(data_dir)
    output_dir = ensure_dir(output_dir)
    submission_dir = ensure_dir(output_dir / "submission")

    train_df = pd.read_csv(data_dir / "train.csv").astype({"uid": str, "target_iid": str})
    test_df = pd.read_csv(data_dir / "test.csv").astype({"uid": str})
    user_df = pd.read_csv(data_dir / "user.csv").astype({"uid": str})
    item_df = pd.read_csv(data_dir / "item.csv").astype({"iid": str})
    candidates = item_df["iid"].astype(str).tolist()

    fit_df, val_df, masked_val_df, bin_weights = test_like_recommendation_split(
        train_df, test_df, val_ratio=val_ratio, seed=seed
    )
    trajectory = Trajectory(
        task_id="B2",
        objective="Maximize test-distribution-weighted NDCG@10 for product recommendation.",
    )

    all_configs = candidate_configs()
    reference_config = all_configs[0]
    reference_metrics = evaluate_config_details(
        fit_df, val_df, masked_val_df, bin_weights, user_df, item_df, candidates, reference_config
    )
    reference_weighted = float(reference_metrics["test_weighted_ndcg@10"])
    best: Dict[str, Any] = {
        "test_weighted_ndcg@10": reference_weighted,
        "config": dict(reference_config),
        "metrics": reference_metrics,
    }

    configs = all_configs[: max(1, budget)]
    for round_id, config in enumerate(configs, start=1):
        if time_limit is not None and timer.elapsed > time_limit:
            break
        round_start = time.time()
        metrics = evaluate_config_details(
            fit_df, val_df, masked_val_df, bin_weights, user_df, item_df, candidates, config
        )
        feedback = {
            **metrics,
            "fit_size": int(len(fit_df)),
            "val_size": int(len(val_df)),
            "masked_val_size": int(len(masked_val_df)),
            "candidate_items": int(len(candidates)),
            "test_bin_weights": bin_weights,
            "reference_test_weighted_ndcg@10": reference_weighted,
        }
        current_weighted = float(metrics["test_weighted_ndcg@10"])
        if current_weighted >= float(best["test_weighted_ndcg@10"]):
            best = {
                "test_weighted_ndcg@10": current_weighted,
                "config": dict(config),
                "metrics": metrics,
            }
            strategy = "KEEP_AS_BEST; improves or ties v1 under test-distribution-weighted validation."
            trajectory.best_round = round_id
            trajectory.selected_config = dict(config)
        else:
            strategy = "REJECT; test-distribution-weighted NDCG@10 did not improve over current best."
        trajectory.add(round_id, dict(config), feedback, strategy, time.time() - round_start)

    if float(best["test_weighted_ndcg@10"]) + 1e-12 < reference_weighted:
        raise RuntimeError(
            "Best recommendation candidate is below the v1 reference under test-weighted validation; not generating v2."
        )

    final_model = build_recommender(best["config"], candidates, user_df, item_df).fit(train_df)
    predictions = [final_model.predict_row(row, k=10) for _, row in test_df.iterrows()]
    out = pd.DataFrame(
        {
            "uid": test_df["uid"].astype(str),
            "prediction": [",".join(items[:10]) for items in predictions],
        }
    )
    a2_path = submission_dir / "A2.csv"
    out.to_csv(a2_path, index=False)

    result = {
        "task": "recommendation",
        "best_val_ndcg@10": best["metrics"]["plain_val_ndcg@10"],
        "best_masked_val_ndcg@10": best["metrics"]["masked_val_ndcg@10"],
        "best_test_weighted_ndcg@10": best["metrics"]["test_weighted_ndcg@10"],
        "best_by_bin_ndcg@10": best["metrics"]["by_bin_ndcg@10"],
        "best_natural_short_val_ndcg@10": best["metrics"].get("natural_short_val_ndcg@10", 0.0),
        "natural_short_val_size": best["metrics"].get("natural_short_val_size", 0),
        "reference_test_weighted_ndcg@10": reference_weighted,
        "best_config": best["config"],
        "num_rounds": len(trajectory.records),
        "prediction_path": str(a2_path),
        "duration": round(timer.elapsed, 4),
        "zero_history_test_users": int(
            test_df["item_seq_raw"].map(lambda x: len(parse_sequence(x))).eq(0).sum()
            if "item_seq_raw" in test_df.columns
            else 0
        ),
        "test_bin_weights": bin_weights,
    }
    trajectory.selected_config = best["config"]
    write_json(output_dir / "trajectory_B2.json", trajectory.to_dict())
    write_json(output_dir / "recommendation_result.json", result)
    return result
