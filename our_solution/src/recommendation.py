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
from .qwen_agent import ask_qwen_json, compact_json


LENGTH_BINS = ("<=3", "4-10", "11-30", "31-80", ">80")
EXACT_LENGTH_BINS = ("0", "1", "2", "3", "4-10", "11-20", "21-30", "31-80", ">80")


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


def exact_length_bin(length: int) -> str:
    if length <= 0:
        return "0"
    if length <= 3:
        return str(length)
    if length <= 10:
        return "4-10"
    if length <= 20:
        return "11-20"
    if length <= 30:
        return "21-30"
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


def test_exact_len_weights(test_df: pd.DataFrame, seq_col: str = "item_seq_raw") -> Dict[str, float]:
    lengths = (
        test_df[seq_col].map(lambda x: len(parse_sequence(x)))
        if seq_col in test_df.columns
        else pd.Series([0] * len(test_df))
    )
    bins = lengths.map(exact_length_bin)
    counts = bins.value_counts(normalize=True).to_dict()
    return {b: float(counts.get(b, 0.0)) for b in EXACT_LENGTH_BINS}


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


def score_by_exact_len(
    predictions: Sequence[Sequence[str]],
    targets: Sequence[str],
    bins: Sequence[str],
) -> Dict[str, float]:
    scores: DefaultDict[str, List[float]] = defaultdict(list)
    for pred, target, bin_name in zip(predictions, targets, bins):
        value = 0.0
        for rank, iid in enumerate(pred[:10], start=1):
            if iid == target:
                value = 1.0 / math.log2(rank + 1)
                break
        scores[str(bin_name)].append(value)
    return {b: float(np.mean(scores[b])) if scores.get(b) else 0.0 for b in EXACT_LENGTH_BINS}


def weighted_exact_score(by_bin: Dict[str, float], weights: Dict[str, float]) -> float:
    return float(sum(by_bin.get(b, 0.0) * weights.get(b, 0.0) for b in EXACT_LENGTH_BINS))


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


@dataclass
class SegmentBayesShortRerankRecommender:
    """v19 probe: freeze v17 top1 and rerank only selected short-history tails."""

    config: Dict[str, Any]
    candidates: List[str]
    user_df: pd.DataFrame
    item_df: pd.DataFrame

    def __post_init__(self) -> None:
        base_config = dict(self.config.get("base_config") or v17_conservative_shortseq_config())
        self.base_model = build_recommender(base_config, self.candidates, self.user_df, self.item_df)
        self.user_lookup = self.user_df.set_index("uid") if "uid" in self.user_df.columns else pd.DataFrame()
        self.target_lengths = {int(x) for x in self.config.get("target_lengths", (3,))}
        self.freeze_top_n = int(self.config.get("freeze_top_n", 1))
        self.base_rank_weight = float(self.config.get("base_rank_weight", 1.0))
        self.min_count = int(self.config.get("min_count", 8))
        self.min_group_count = int(self.config.get("min_group_count", 12))
        self.shrink_beta = float(self.config.get("shrink_beta", 60.0))
        self.min_lift = float(self.config.get("min_lift", 1.02))
        self.prior_pool = int(self.config.get("prior_pool", 80))
        self.alpha_by_len = {int(k): float(v) for k, v in self.config.get("alpha_by_len", {}).items()}
        self.component_weights = {
            "last": 0.8,
            "suffix": 0.35,
            "last_group": 1.0,
            "suffix_group": 0.75,
            **dict(self.config.get("component_weights", {})),
        }
        raw_group_specs = self.config.get(
            "group_specs",
            (
                ("u_cat_01", "u_cat_02"),
                ("u_cat_01", "u_cat_06"),
                ("u_cat_02", "u_cat_06"),
            ),
        )
        self.group_specs = tuple(tuple(cols) for cols in raw_group_specs)
        self.suffix_orders = tuple(int(x) for x in self.config.get("suffix_orders", (2,)))
        self.global_counts: Counter = Counter()
        self.last_counts: DefaultDict[str, Counter] = defaultdict(Counter)
        self.suffix_counts: Dict[int, DefaultDict[Tuple[str, ...], Counter]] = {
            order: defaultdict(Counter) for order in self.suffix_orders
        }
        self.last_group_counts: Dict[Tuple[str, ...], DefaultDict[Tuple[Tuple[str, ...], str], Counter]] = {
            cols: defaultdict(Counter) for cols in self.group_specs
        }
        self.suffix_group_counts: Dict[
            Tuple[str, ...],
            Dict[int, DefaultDict[Tuple[Tuple[str, ...], Tuple[str, ...]], Counter]],
        ] = {
            cols: {order: defaultdict(Counter) for order in self.suffix_orders}
            for cols in self.group_specs
        }
        self.global_total = 0
        self.candidate_set = set(self.candidates)

    def fit(self, train_df: pd.DataFrame) -> "SegmentBayesShortRerankRecommender":
        self.base_model.fit(train_df)
        for row in train_df.itertuples(index=False):
            row_dict = row._asdict()
            target = str(row_dict["target_iid"])
            if target not in self.candidate_set:
                continue
            uid = str(row_dict.get("uid", ""))
            hist = parse_sequence(row_dict.get("item_seq_raw", ""))
            self.global_counts[target] += 1
            self.global_total += 1
            if not hist:
                continue
            last_item = hist[-1]
            self.last_counts[last_item][target] += 1
            for order in self.suffix_orders:
                if len(hist) >= order:
                    self.suffix_counts[order][tuple(hist[-order:])][target] += 1
            user_row = self._user_row(uid)
            if user_row is None:
                continue
            for cols in self.group_specs:
                group_key = self._group_key(user_row, cols)
                if not group_key:
                    continue
                self.last_group_counts[cols][(group_key, last_item)][target] += 1
                for order in self.suffix_orders:
                    if len(hist) >= order:
                        suffix = tuple(hist[-order:])
                        self.suffix_group_counts[cols][order][(suffix, group_key)][target] += 1
        return self

    def _user_row(self, uid: str) -> Optional[pd.Series]:
        if self.user_lookup.empty or uid not in self.user_lookup.index:
            return None
        row = self.user_lookup.loc[uid]
        if isinstance(row, pd.DataFrame):
            return row.iloc[0]
        return row

    def _group_key(self, user_row: pd.Series, cols: Tuple[str, ...]) -> Tuple[str, ...]:
        if not all(col in user_row.index and pd.notna(user_row[col]) for col in cols):
            return ()
        return tuple(str(user_row[col]) for col in cols)

    def _global_prob(self, iid: str) -> float:
        smooth = float(self.config.get("global_smooth", 1.0))
        return (float(self.global_counts.get(iid, 0)) + smooth) / (
            float(self.global_total) + smooth * max(1, len(self.candidates))
        )

    def _add_counter_scores(
        self,
        scores: Dict[str, float],
        counter: Optional[Counter],
        allowed: set[str],
        alpha: float,
        component: str,
        min_total: int,
    ) -> None:
        if not counter:
            return
        total = float(sum(counter.values()))
        if total < min_total:
            return
        scale = alpha * float(self.component_weights.get(component, 1.0))
        if scale <= 0:
            return
        for iid, count in counter.most_common(self.prior_pool):
            if iid not in allowed:
                continue
            global_prob = self._global_prob(iid)
            posterior = (float(count) + self.shrink_beta * global_prob) / (total + self.shrink_beta)
            lift = posterior / max(global_prob, 1e-12)
            if lift >= self.min_lift:
                scores[iid] = scores.get(iid, 0.0) + scale * math.log(lift)

    def _prior_scores(self, row: pd.Series, hist: List[str], allowed: set[str], alpha: float) -> Dict[str, float]:
        scores: Dict[str, float] = {}
        if not hist:
            return scores
        last_item = hist[-1]
        self._add_counter_scores(
            scores, self.last_counts.get(last_item), allowed, alpha, "last", self.min_count
        )
        for order in self.suffix_orders:
            if len(hist) >= order:
                suffix = tuple(hist[-order:])
                self._add_counter_scores(
                    scores, self.suffix_counts[order].get(suffix), allowed, alpha, "suffix", self.min_count
                )
        user_row = self._user_row(str(row.get("uid", "")))
        if user_row is None:
            return scores
        for cols in self.group_specs:
            group_key = self._group_key(user_row, cols)
            if not group_key:
                continue
            self._add_counter_scores(
                scores,
                self.last_group_counts[cols].get((group_key, last_item)),
                allowed,
                alpha,
                "last_group",
                self.min_group_count,
            )
            for order in self.suffix_orders:
                if len(hist) >= order:
                    suffix = tuple(hist[-order:])
                    self._add_counter_scores(
                        scores,
                        self.suffix_group_counts[cols][order].get((suffix, group_key)),
                        allowed,
                        alpha,
                        "suffix_group",
                        self.min_group_count,
                    )
        return scores

    def predict_row(self, row: pd.Series, k: int = 10) -> List[str]:
        hist = parse_sequence(row.get("item_seq_raw", ""))
        raw_len = len(hist)
        base_items = self.base_model.predict_row(row, k=k)
        if raw_len not in self.target_lengths:
            return base_items[:k]
        if len(base_items) <= self.freeze_top_n:
            return base_items[:k]
        alpha = float(self.alpha_by_len.get(raw_len, self.config.get("alpha", 0.04)))
        if alpha <= 0:
            return base_items[:k]
        frozen = base_items[: self.freeze_top_n]
        tail = base_items[self.freeze_top_n : k]
        allowed = set(tail)
        scores: Dict[str, float] = {}
        for rank, iid in enumerate(tail, start=self.freeze_top_n + 1):
            scores[iid] = self.base_rank_weight / math.log2(rank + 1)
        for iid, value in self._prior_scores(row, hist, allowed, alpha).items():
            scores[iid] = scores.get(iid, 0.0) + value
        ordered_tail = sorted(tail, key=lambda iid: (-scores.get(iid, 0.0), tail.index(iid), iid))
        return (frozen + ordered_tail)[:k]


@dataclass
class LongHistoryCountRerankRecommender:
    """v22 probe: freeze top1 and rerank long-history tails by in-history frequency/recency."""

    config: Dict[str, Any]
    candidates: List[str]
    user_df: pd.DataFrame
    item_df: pd.DataFrame

    def __post_init__(self) -> None:
        base_config = dict(self.config.get("base_config") or v17_conservative_shortseq_config())
        self.base_model = build_recommender(base_config, self.candidates, self.user_df, self.item_df)

    def fit(self, train_df: pd.DataFrame) -> "LongHistoryCountRerankRecommender":
        self.base_model.fit(train_df)
        return self

    def predict_row(self, row: pd.Series, k: int = 10) -> List[str]:
        base_pool = max(k, int(self.config.get("base_pool", 20)))
        base_items = self.base_model.predict_row(row, k=base_pool)
        hist = parse_sequence(row.get("item_seq_raw", ""))
        if len(hist) < int(self.config.get("min_len", 31)):
            return base_items[:k]
        freeze_top_n = int(self.config.get("freeze_top_n", 1))
        if len(base_items) <= freeze_top_n:
            return base_items[:k]

        counts: Dict[str, int] = {}
        recency: Dict[str, int] = {}
        for pos, iid in enumerate(hist):
            counts[iid] = counts.get(iid, 0) + 1
            recency[iid] = pos
        if not counts:
            return base_items[:k]

        max_count = max(counts.values())
        hist_len = max(1, len(hist))
        alpha = float(self.config.get("alpha", 0.10))
        count_weight = float(self.config.get("count_weight", 0.8))
        recency_weight = float(self.config.get("recency_weight", 0.4))
        last_weight = float(self.config.get("last_weight", 0.1))
        base_rank_weight = float(self.config.get("base_rank_weight", 1.0))

        frozen = base_items[:freeze_top_n]
        tail = base_items[freeze_top_n:base_pool]
        scores: Dict[str, float] = {}
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
        ordered_tail = sorted(tail, key=lambda iid: (-scores.get(iid, 0.0), tail.index(iid), iid))
        return (frozen + ordered_tail)[:k]


@dataclass
class MediumLongConditionalRepeatGateRecommender:
    """v27 probe: v24 tail rerank with repeat concentration and high-support priors."""

    config: Dict[str, Any]
    candidates: List[str]
    user_df: pd.DataFrame
    item_df: pd.DataFrame

    def __post_init__(self) -> None:
        base_config = dict(self.config.get("base_config") or v24_medium_long_history_count_recent_strong_config())
        self.base_model = build_recommender(base_config, self.candidates, self.user_df, self.item_df)
        self.user_lookup = self.user_df.set_index("uid") if "uid" in self.user_df.columns else pd.DataFrame()
        self.candidate_set = set(self.candidates)

        self.min_len = int(self.config.get("min_len", 21))
        self.freeze_top_n = int(self.config.get("freeze_top_n", 1))
        self.base_pool = int(self.config.get("base_pool", 30))
        self.base_rank_weight = float(self.config.get("base_rank_weight", 1.0))
        self.alpha_repeat = float(self.config.get("alpha_repeat", 0.18))
        self.alpha_prior = float(self.config.get("alpha_prior", 0.12))
        self.count_weight = float(self.config.get("count_weight", 0.75))
        self.recency_weight = float(self.config.get("recency_weight", 0.25))
        self.last_weight = float(self.config.get("last_weight", 0.10))
        self.last_repeat_min_count = int(self.config.get("last_repeat_min_count", 2))
        self.repeat_gate_floor = float(self.config.get("repeat_gate_floor", 0.5))
        self.repeat_gate_scale = float(self.config.get("repeat_gate_scale", 1.5))

        self.min_count = int(self.config.get("min_count", 20))
        self.min_group_count = int(self.config.get("min_group_count", 40))
        self.shrink_beta = float(self.config.get("shrink_beta", 80.0))
        self.min_lift = float(self.config.get("min_lift", 1.08))
        self.global_smooth = float(self.config.get("global_smooth", 1.0))
        self.suffix_orders = tuple(int(x) for x in self.config.get("suffix_orders", (2,)))
        self.component_weights = {
            "last": 1.0,
            "suffix": 0.7,
            "last_group": 1.0,
            "suffix_group": 0.7,
            **dict(self.config.get("component_weights", {})),
        }

        raw_group_specs = self.config.get("group_specs")
        if raw_group_specs is None:
            group_cols = tuple(self.config.get("group_cols", ("u_cat_01", "u_cat_02", "u_cat_06")))
            raw_group_specs = (group_cols,)
        self.group_specs = tuple(tuple(cols) for cols in raw_group_specs if tuple(cols))

        self.global_counts: Counter = Counter()
        self.global_total = 0
        self.last_counts: DefaultDict[str, Counter] = defaultdict(Counter)
        self.suffix_counts: Dict[int, DefaultDict[Tuple[str, ...], Counter]] = {
            order: defaultdict(Counter) for order in self.suffix_orders
        }
        self.last_group_counts: Dict[Tuple[str, ...], DefaultDict[Tuple[str, Tuple[str, ...]], Counter]] = {
            cols: defaultdict(Counter) for cols in self.group_specs
        }
        self.suffix_group_counts: Dict[
            Tuple[str, ...],
            Dict[int, DefaultDict[Tuple[Tuple[str, ...], Tuple[str, ...]], Counter]],
        ] = {
            cols: {order: defaultdict(Counter) for order in self.suffix_orders}
            for cols in self.group_specs
        }

    def fit(self, train_df: pd.DataFrame) -> "MediumLongConditionalRepeatGateRecommender":
        self.base_model.fit(train_df)
        for row in train_df.itertuples(index=False):
            row_dict = row._asdict()
            target = str(row_dict.get("target_iid", ""))
            if target not in self.candidate_set:
                continue
            hist = parse_sequence(row_dict.get("item_seq_raw", ""))
            self.global_counts[target] += 1
            self.global_total += 1
            if not hist:
                continue

            last_item = hist[-1]
            self.last_counts[last_item][target] += 1
            for order in self.suffix_orders:
                if len(hist) >= order:
                    self.suffix_counts[order][tuple(hist[-order:])][target] += 1

            user_row = self._user_row(str(row_dict.get("uid", "")))
            if user_row is None:
                continue
            for cols in self.group_specs:
                group_key = self._group_key(user_row, cols)
                if not group_key:
                    continue
                self.last_group_counts[cols][(last_item, group_key)][target] += 1
                for order in self.suffix_orders:
                    if len(hist) >= order:
                        suffix = tuple(hist[-order:])
                        self.suffix_group_counts[cols][order][(suffix, group_key)][target] += 1
        return self

    def _user_row(self, uid: str) -> Optional[pd.Series]:
        if self.user_lookup.empty or uid not in self.user_lookup.index:
            return None
        row = self.user_lookup.loc[uid]
        if isinstance(row, pd.DataFrame):
            return row.iloc[0]
        return row

    def _group_key(self, user_row: pd.Series, cols: Tuple[str, ...]) -> Tuple[str, ...]:
        if not all(col in user_row.index and pd.notna(user_row[col]) for col in cols):
            return ()
        return tuple(str(user_row[col]) for col in cols)

    def _global_prob(self, iid: str) -> float:
        return (float(self.global_counts.get(iid, 0)) + self.global_smooth) / (
            float(self.global_total) + self.global_smooth * max(1, len(self.candidates))
        )

    def _counter_log_lift(self, counter: Optional[Counter], iid: str, min_total: int) -> float:
        if not counter:
            return 0.0
        total = float(sum(counter.values()))
        if total < float(min_total):
            return 0.0
        global_prob = self._global_prob(iid)
        posterior = (float(counter.get(iid, 0)) + self.shrink_beta * global_prob) / (
            total + self.shrink_beta
        )
        lift = posterior / max(global_prob, 1e-12)
        if lift < self.min_lift:
            return 0.0
        return math.log(lift)

    def _prior_score(self, row: pd.Series, hist: List[str], iid: str) -> float:
        if not hist or self.alpha_prior <= 0:
            return 0.0
        last_item = hist[-1]
        score = self.component_weights["last"] * self._counter_log_lift(
            self.last_counts.get(last_item), iid, self.min_count
        )
        for order in self.suffix_orders:
            if len(hist) >= order:
                suffix = tuple(hist[-order:])
                score += self.component_weights["suffix"] * self._counter_log_lift(
                    self.suffix_counts[order].get(suffix), iid, self.min_count
                )

        user_row = self._user_row(str(row.get("uid", "")))
        if user_row is None:
            return self.alpha_prior * score
        for cols in self.group_specs:
            group_key = self._group_key(user_row, cols)
            if not group_key:
                continue
            score += self.component_weights["last_group"] * self._counter_log_lift(
                self.last_group_counts[cols].get((last_item, group_key)),
                iid,
                self.min_group_count,
            )
            for order in self.suffix_orders:
                if len(hist) >= order:
                    suffix = tuple(hist[-order:])
                    score += self.component_weights["suffix_group"] * self._counter_log_lift(
                        self.suffix_group_counts[cols][order].get((suffix, group_key)),
                        iid,
                        self.min_group_count,
                    )
        return self.alpha_prior * score

    def _repeat_gate(self, counts: Counter, hist_len: int) -> float:
        if not counts or hist_len <= 0:
            return self.repeat_gate_floor
        max_count = max(counts.values())
        unique_ratio = len(counts) / max(hist_len, 1)
        repeat_ratio = max(0.0, 1.0 - unique_ratio)
        top_share = max_count / max(hist_len, 1)
        concentration = min(1.0, 2.0 * repeat_ratio + top_share)
        return self.repeat_gate_floor + self.repeat_gate_scale * concentration

    def predict_row(self, row: pd.Series, k: int = 10) -> List[str]:
        hist = parse_sequence(row.get("item_seq_raw", ""))
        raw_len = len(hist)
        if raw_len < self.min_len:
            return self.base_model.predict_row(row, k=k)[:k]

        base_pool = max(k, self.base_pool)
        base_items = self.base_model.predict_row(row, k=base_pool)
        if len(base_items) <= self.freeze_top_n:
            return base_items[:k]

        counts = Counter(hist)
        if not counts:
            return base_items[:k]
        recency = {iid: pos for pos, iid in enumerate(hist)}
        max_count = max(counts.values())
        hist_len = max(1, raw_len)
        repeat_gate = self._repeat_gate(counts, hist_len)

        frozen = base_items[: self.freeze_top_n]
        tail = base_items[self.freeze_top_n : base_pool]
        tail_pos = {iid: pos for pos, iid in enumerate(tail)}
        scores: Dict[str, float] = {}
        for rank, iid in enumerate(tail, start=self.freeze_top_n + 1):
            score = self.base_rank_weight / math.log2(rank + 1)
            if iid in counts and self.alpha_repeat > 0:
                count_feature = math.log1p(counts[iid]) / math.log1p(max_count)
                recency_feature = (recency[iid] + 1) / hist_len
                last_feature = 1.0 if iid == hist[-1] and counts[iid] >= self.last_repeat_min_count else 0.0
                score += self.alpha_repeat * repeat_gate * (
                    self.count_weight * count_feature
                    + self.recency_weight * recency_feature
                    + self.last_weight * last_feature
                )
            score += self._prior_score(row, hist, iid)
            scores[iid] = score

        ordered_tail = sorted(tail, key=lambda iid: (-scores.get(iid, 0.0), tail_pos[iid], iid))
        return (frozen + ordered_tail)[:k]


@dataclass
class CandidateLGBMRankerRecommender:
    """v31 probe: top80 multi-source recall plus a conservative LightGBM tail ranker."""

    config: Dict[str, Any]
    candidates: List[str]
    user_df: pd.DataFrame
    item_df: pd.DataFrame

    def __post_init__(self) -> None:
        base_config = dict(self.config.get("base_config") or v29_medium_long_count_only_pool25_config())
        self.base_model = build_recommender(base_config, self.candidates, self.user_df, self.item_df)
        self.candidate_set = set(self.candidates)
        self.user_lookup = self.user_df.set_index("uid") if "uid" in self.user_df.columns else pd.DataFrame()
        self.item_lookup = self.item_df.set_index("iid") if "iid" in self.item_df.columns else pd.DataFrame()
        self.item_cols = [c for c in self.item_df.columns if c.startswith("i_cat_") or c.startswith("i_bucket_")]
        self.group_specs = tuple(
            tuple(cols)
            for cols in self.config.get(
                "group_specs",
                (
                    ("u_cat_01",),
                    ("u_cat_02",),
                    ("u_cat_06",),
                    ("u_cat_01", "u_cat_02"),
                    ("u_cat_01", "u_cat_06"),
                    ("u_cat_02", "u_cat_06"),
                ),
            )
        )
        self.candidate_pool_size = int(self.config.get("candidate_pool_size", 80))
        self.freeze_top_n = int(self.config.get("freeze_top_n", 1))
        self.enable_short_history = bool(self.config.get("enable_short_history", False))
        self.min_rank_len = int(self.config.get("min_rank_len", 4))
        self.counter_top_n = int(self.config.get("counter_top_n", 20))
        self.max_hist_items = int(self.config.get("max_hist_items", 80))
        self.max_train_groups = int(self.config.get("max_train_groups", 8000))
        self.shrink_beta = float(self.config.get("shrink_beta", 80.0))
        self.global_smooth = float(self.config.get("global_smooth", 1.0))
        self.feature_names = self._feature_names()
        self.feature_index = {name: idx for idx, name in enumerate(self.feature_names)}
        self._load_prediction_settings()
        self.booster = None

        self.global_counts: Counter = Counter()
        self.global_total = 0
        self.last_counts: DefaultDict[str, Counter] = defaultdict(Counter)
        self.last_totals: Counter = Counter()
        self.suffix2_counts: DefaultDict[Tuple[str, str], Counter] = defaultdict(Counter)
        self.suffix2_totals: Counter = Counter()
        self.hist_item_counts: DefaultDict[str, Counter] = defaultdict(Counter)
        self.hist_item_totals: Counter = Counter()
        self.user_group_counts: DefaultDict[Tuple[Tuple[str, ...], Tuple[str, ...]], Counter] = defaultdict(Counter)
        self.user_group_totals: Counter = Counter()
        self.hist_feature_counts: DefaultDict[Tuple[str, str], Counter] = defaultdict(Counter)
        self.hist_feature_totals: Counter = Counter()
        self.global_top: List[str] = []

    def _load_prediction_settings(self) -> None:
        self.candidate_pool_size = int(self.config.get("candidate_pool_size", self.candidate_pool_size))
        self.freeze_top_n = int(self.config.get("freeze_top_n", self.freeze_top_n))
        self.enable_short_history = bool(self.config.get("enable_short_history", self.enable_short_history))
        self.min_rank_len = int(self.config.get("min_rank_len", self.min_rank_len))
        self.prediction_mode = str(self.config.get("prediction_mode", "lgbm_rerank"))
        self.base_score_weight = float(self.config.get("base_score_weight", 0.0))
        self.insert_start_rank = int(self.config.get("insert_start_rank", self.freeze_top_n + 1))
        self.max_insertions = int(self.config.get("max_insertions", self.candidate_pool_size))
        self.min_source_count = int(self.config.get("min_source_count", 0))
        self.require_non_base_source = bool(self.config.get("require_non_base_source", False))
        self.lgbm_rank_top_n = int(self.config.get("lgbm_rank_top_n", self.candidate_pool_size))
        self.min_gate_support_log = float(self.config.get("min_gate_support_log", 0.0))
        self.min_gate_log_lift = float(self.config.get("min_gate_log_lift", -1e9))
        self.min_combined_margin = float(self.config.get("min_combined_margin", 0.0))
        self.allow_base_tail_reorder = bool(self.config.get("allow_base_tail_reorder", True))
        self.blend_min_position_margin = float(self.config.get("blend_min_position_margin", 0.0))
        self.blend_min_total_margin = float(self.config.get("blend_min_total_margin", 0.0))
        self.blend_max_changed_positions = int(self.config.get("blend_max_changed_positions", self.candidate_pool_size))
        self.blend_max_new_items = int(self.config.get("blend_max_new_items", self.candidate_pool_size))
        self.blend_require_new_item_gate = bool(self.config.get("blend_require_new_item_gate", False))
        self.prediction_exact_bins = tuple(str(x) for x in self.config.get("prediction_exact_bins", ()))
        self.output_base_k = int(self.config.get("output_base_k", self.candidate_pool_size))
        raw_profiles = self.config.get("segment_gate_profiles", {}) or {}
        self.segment_gate_profiles = {
            str(bin_name): dict(profile)
            for bin_name, profile in raw_profiles.items()
            if isinstance(profile, dict)
        }

    def update_prediction_settings(self, settings: Dict[str, Any]) -> None:
        if "prediction_exact_bins" not in settings:
            self.config.pop("prediction_exact_bins", None)
        if "segment_gate_profiles" not in settings:
            self.config.pop("segment_gate_profiles", None)
        self.config.update(settings)
        self._load_prediction_settings()

    def _prediction_state(self) -> Dict[str, Any]:
        keys = (
            "candidate_pool_size",
            "freeze_top_n",
            "enable_short_history",
            "min_rank_len",
            "prediction_mode",
            "base_score_weight",
            "insert_start_rank",
            "max_insertions",
            "min_source_count",
            "require_non_base_source",
            "lgbm_rank_top_n",
            "min_gate_support_log",
            "min_gate_log_lift",
            "min_combined_margin",
            "allow_base_tail_reorder",
            "blend_min_position_margin",
            "blend_min_total_margin",
            "blend_max_changed_positions",
            "blend_max_new_items",
            "blend_require_new_item_gate",
            "prediction_exact_bins",
            "output_base_k",
        )
        return {key: getattr(self, key) for key in keys}

    def _restore_prediction_state(self, state: Dict[str, Any]) -> None:
        for key, value in state.items():
            setattr(self, key, value)

    def _apply_segment_profile(self, profile: Dict[str, Any]) -> None:
        previous_profiles = self.segment_gate_profiles
        saved_config = dict(self.config)
        try:
            self.config.update(profile)
            self.config.pop("segment_gate_profiles", None)
            self._load_prediction_settings()
            self.segment_gate_profiles = previous_profiles
        finally:
            self.config = saved_config

    def fit(
        self,
        train_df: pd.DataFrame,
        ranker_train_df: Optional[pd.DataFrame] = None,
    ) -> "CandidateLGBMRankerRecommender":
        self.base_model.fit(train_df)
        self._build_statistics(train_df)
        self._fit_ranker(ranker_train_df if ranker_train_df is not None else train_df)
        return self

    def _feature_names(self) -> List[str]:
        return [
            "base_rank",
            "base_rank_score",
            "base_in_top10",
            "base_in_top25",
            "base_in_top50",
            "base_in_top80",
            "hist_len",
            "hist_len_log",
            "len_le3",
            "len_4_10",
            "len_11_20",
            "len_21_30",
            "len_31_80",
            "len_gt80",
            "repeat_ratio",
            "top_share",
            "max_count_log",
            "unique_ratio",
            "cand_count_log",
            "cand_count_norm",
            "cand_recency_norm",
            "cand_is_last",
            "last_support_log",
            "last_posterior",
            "last_log_lift",
            "suffix2_support_log",
            "suffix2_posterior",
            "suffix2_log_lift",
            "hist_item_support_log",
            "hist_item_posterior_max",
            "hist_item_log_lift_max",
            "user_group_support_log",
            "user_group_posterior_max",
            "user_group_log_lift_max",
            "item_feature_support_log",
            "item_feature_posterior_max",
            "item_feature_log_lift_max",
            "global_count_log",
            "global_prob",
            "source_base",
            "source_repeat",
            "source_last",
            "source_suffix2",
            "source_hist_item",
            "source_user_group",
            "source_item_feature",
            "source_global",
        ]

    def _build_statistics(self, train_df: pd.DataFrame) -> None:
        for row in train_df.itertuples(index=False):
            row_dict = row._asdict()
            target = str(row_dict.get("target_iid", ""))
            if target not in self.candidate_set:
                continue
            hist = parse_sequence(row_dict.get("item_seq_raw", ""))
            self.global_counts[target] += 1
            self.global_total += 1

            if hist:
                last_item = hist[-1]
                self.last_counts[last_item][target] += 1
                self.last_totals[last_item] += 1
                if len(hist) >= 2:
                    suffix2 = (hist[-2], hist[-1])
                    self.suffix2_counts[suffix2][target] += 1
                    self.suffix2_totals[suffix2] += 1
                for iid in self._recent_unique(hist):
                    self.hist_item_counts[iid][target] += 1
                    self.hist_item_totals[iid] += 1
                    for key in self._item_feature_keys(iid):
                        self.hist_feature_counts[key][target] += 1
                        self.hist_feature_totals[key] += 1

            user_row = self._user_row(str(row_dict.get("uid", "")))
            if user_row is not None:
                for spec in self.group_specs:
                    group_key = self._group_key(user_row, spec)
                    if not group_key:
                        continue
                    key = (spec, group_key)
                    self.user_group_counts[key][target] += 1
                    self.user_group_totals[key] += 1

        self.global_top = [iid for iid, _ in self.global_counts.most_common(len(self.candidates))]

    def _fit_ranker(self, train_df: pd.DataFrame) -> None:
        try:
            import lightgbm as lgb  # type: ignore
        except Exception:
            self.booster = None
            return

        rank_df = train_df.copy()
        rank_df["_hist_len_for_ranker"] = rank_df["item_seq_raw"].map(lambda x: len(parse_sequence(x)))
        rank_df = rank_df[rank_df["_hist_len_for_ranker"] >= self.min_rank_len]
        if rank_df.empty:
            self.booster = None
            return
        seed = int(self.config.get("seed", 42))
        label_mode = str(self.config.get("ranker_label_mode", "target"))
        rank_df = rank_df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
        if self.max_train_groups > 0 and len(rank_df) > self.max_train_groups:
            rank_df = rank_df.head(self.max_train_groups).copy()

        n_features = len(self.feature_names)
        max_group_size = self.candidate_pool_size + 1
        max_rows = int(len(rank_df) * max_group_size)
        x_values = np.zeros((max_rows, n_features), dtype=np.float32)
        labels = np.zeros(max_rows, dtype=np.float32)
        group_sizes: List[int] = []
        pos = 0

        for _, row in rank_df.iterrows():
            target = str(row.get("target_iid", ""))
            if target not in self.candidate_set:
                continue
            target_label = 1.0
            if label_mode == "uplift":
                base_top = self.base_model.predict_row(row, k=10)
                if target in base_top[:5]:
                    target_label = 0.0
                elif target in base_top[5:10]:
                    target_label = 1.0
                else:
                    target_label = 2.0
                if target_label <= 0.0:
                    continue
            context = self._candidate_context(row, pool_size=self.candidate_pool_size)
            items = list(context["items"])
            if target not in items:
                items.append(target)
            start = pos
            for iid in items:
                if pos >= len(labels):
                    break
                x_values[pos, :] = self._features_for(row, iid, context)
                labels[pos] = float(target_label) if iid == target else 0.0
                pos += 1
            group_size = pos - start
            if group_size > 1 and labels[start:pos].max() > 0:
                group_sizes.append(group_size)
            else:
                pos = start

        if not group_sizes or pos <= 0:
            self.booster = None
            return
        x_values = x_values[:pos]
        labels = labels[:pos]

        valid_groups = max(1, int(round(0.10 * len(group_sizes)))) if len(group_sizes) >= 10 else 0
        train_group_count = len(group_sizes) - valid_groups
        if train_group_count <= 0:
            self.booster = None
            return
        train_rows = int(sum(group_sizes[:train_group_count]))
        train_set = lgb.Dataset(
            x_values[:train_rows],
            label=labels[:train_rows],
            group=group_sizes[:train_group_count],
            feature_name=self.feature_names,
            free_raw_data=False,
        )
        valid_sets = [train_set]
        valid_names = ["train"]
        callbacks = [lgb.log_evaluation(period=0)]
        if valid_groups > 0 and train_rows < len(labels):
            valid_set = lgb.Dataset(
                x_values[train_rows:],
                label=labels[train_rows:],
                group=group_sizes[train_group_count:],
                feature_name=self.feature_names,
                reference=train_set,
                free_raw_data=False,
            )
            valid_sets.append(valid_set)
            valid_names.append("valid")
            callbacks.append(lgb.early_stopping(int(self._lgbm_control("early_stopping_rounds", 30)), verbose=False))

        params = dict(self.config.get("lgbm_params", {}))
        num_boost_round = int(params.pop("num_boost_round", self.config.get("num_boost_round", 300)))
        params.pop("early_stopping_rounds", None)
        params.setdefault("objective", "lambdarank")
        params.setdefault("metric", "ndcg")
        params.setdefault("ndcg_eval_at", [10])
        params.setdefault("learning_rate", 0.03)
        params.setdefault("num_leaves", 31)
        params.setdefault("min_data_in_leaf", 50)
        params.setdefault("verbosity", -1)
        params.setdefault("seed", seed)
        params.setdefault("feature_pre_filter", False)
        self.booster = lgb.train(
            params,
            train_set,
            num_boost_round=num_boost_round,
            valid_sets=valid_sets,
            valid_names=valid_names,
            callbacks=callbacks,
        )

    def _lgbm_control(self, key: str, default: Any) -> Any:
        params = self.config.get("lgbm_params", {})
        if isinstance(params, dict) and key in params:
            return params[key]
        return self.config.get(key, default)

    def _user_row(self, uid: str) -> Optional[pd.Series]:
        if self.user_lookup.empty or uid not in self.user_lookup.index:
            return None
        row = self.user_lookup.loc[uid]
        if isinstance(row, pd.DataFrame):
            return row.iloc[0]
        return row

    def _group_key(self, user_row: pd.Series, spec: Tuple[str, ...]) -> Tuple[str, ...]:
        values: List[str] = []
        for col in spec:
            if col not in user_row.index or pd.isna(user_row[col]):
                return ()
            values.append(str(user_row[col]))
        return tuple(values)

    def _item_feature_keys(self, iid: str) -> List[Tuple[str, str]]:
        if self.item_lookup.empty or iid not in self.item_lookup.index:
            return []
        row = self.item_lookup.loc[iid]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        keys: List[Tuple[str, str]] = []
        for col in self.item_cols:
            value = row.get(col)
            if pd.notna(value):
                keys.append((col, str(value)))
        return keys

    def _recent_unique(self, hist: Sequence[str]) -> List[str]:
        seen = set()
        out: List[str] = []
        for iid in reversed(hist[-self.max_hist_items :]):
            if iid in seen:
                continue
            seen.add(iid)
            out.append(iid)
        return out

    def _global_prob(self, iid: str) -> float:
        return (float(self.global_counts.get(iid, 0)) + self.global_smooth) / (
            float(self.global_total) + self.global_smooth * max(1, len(self.candidates))
        )

    def _counter_stats(self, counter: Optional[Counter], total: float, iid: str) -> Tuple[float, float, float]:
        if not counter or total <= 0:
            prob = self._global_prob(iid)
            return 0.0, prob, 0.0
        support = float(counter.get(iid, 0.0))
        global_prob = self._global_prob(iid)
        posterior = (support + self.shrink_beta * global_prob) / (float(total) + self.shrink_beta)
        lift = posterior / max(global_prob, 1e-12)
        return support, posterior, math.log(max(lift, 1e-12))

    def _best_stats(self, stats: Sequence[Tuple[Optional[Counter], float]], iid: str) -> Tuple[float, float, float]:
        support_sum = 0.0
        posterior_max = 0.0
        lift_max = 0.0
        for counter, total in stats:
            support, posterior, log_lift = self._counter_stats(counter, total, iid)
            support_sum += support
            posterior_max = max(posterior_max, posterior)
            lift_max = max(lift_max, log_lift)
        if not stats:
            posterior_max = self._global_prob(iid)
        return support_sum, posterior_max, lift_max

    def _add_item(
        self,
        out: List[str],
        source_flags: DefaultDict[str, set],
        iid: str,
        source: str,
        limit: int,
    ) -> None:
        if iid not in self.candidate_set:
            return
        source_flags[iid].add(source)
        if iid not in out and len(out) < limit:
            out.append(iid)

    def _add_counter_candidates(
        self,
        out: List[str],
        source_flags: DefaultDict[str, set],
        counter: Optional[Counter],
        source: str,
        limit: int,
        top_n: Optional[int] = None,
    ) -> None:
        if not counter:
            return
        for iid, _ in counter.most_common(top_n or self.counter_top_n):
            self._add_item(out, source_flags, str(iid), source, limit)
            if len(out) >= limit:
                break

    def _candidate_context(self, row: pd.Series, pool_size: Optional[int] = None) -> Dict[str, Any]:
        limit = int(pool_size or self.candidate_pool_size)
        base_items = self.base_model.predict_row(row, k=max(limit, 10))
        source_flags: DefaultDict[str, set] = defaultdict(set)
        base_ranks: Dict[str, int] = {}
        out: List[str] = []
        for rank, iid in enumerate(base_items, start=1):
            base_ranks[str(iid)] = rank
            self._add_item(out, source_flags, str(iid), "base", limit)

        hist = parse_sequence(row.get("item_seq_raw", ""))
        counts = Counter(hist)
        recency = {iid: pos for pos, iid in enumerate(hist)}
        recent_unique = self._recent_unique(hist)
        for iid, _ in sorted(counts.items(), key=lambda item: (-item[1], -recency.get(item[0], -1), item[0])):
            self._add_item(out, source_flags, str(iid), "repeat", limit)

        if hist:
            self._add_counter_candidates(out, source_flags, self.last_counts.get(hist[-1]), "last", limit)
            if len(hist) >= 2:
                self._add_counter_candidates(
                    out,
                    source_flags,
                    self.suffix2_counts.get((hist[-2], hist[-1])),
                    "suffix2",
                    limit,
                )
            hist_counter: Counter = Counter()
            for pos, iid in enumerate(recent_unique):
                counter = self.hist_item_counts.get(iid)
                if not counter:
                    continue
                weight = 1.0 / math.log2(pos + 2)
                for target, value in counter.items():
                    hist_counter[target] += float(value) * weight
            self._add_counter_candidates(out, source_flags, hist_counter, "hist_item", limit)

            feature_counter: Counter = Counter()
            for iid in recent_unique:
                for key in self._item_feature_keys(iid):
                    counter = self.hist_feature_counts.get(key)
                    if not counter:
                        continue
                    for target, value in counter.items():
                        feature_counter[target] += float(value)
            self._add_counter_candidates(out, source_flags, feature_counter, "item_feature", limit)

        hist_item_stats_inputs = [
            (self.hist_item_counts.get(item), float(self.hist_item_totals.get(item, 0)))
            for item in recent_unique
        ]
        item_feature_keys = [
            key
            for item in recent_unique
            for key in self._item_feature_keys(item)
        ]
        item_feature_stats_inputs = [
            (self.hist_feature_counts.get(key), float(self.hist_feature_totals.get(key, 0)))
            for key in item_feature_keys
        ]
        group_stats_inputs: List[Tuple[Optional[Counter], float]] = []
        user_row = self._user_row(str(row.get("uid", "")))
        if user_row is not None:
            group_counter: Counter = Counter()
            for spec in self.group_specs:
                group_key = self._group_key(user_row, spec)
                if not group_key:
                    continue
                key = (spec, group_key)
                counter = self.user_group_counts.get(key)
                group_stats_inputs.append((counter, float(self.user_group_totals.get(key, 0))))
                if not counter:
                    continue
                for target, value in counter.items():
                    group_counter[target] += float(value)
            self._add_counter_candidates(out, source_flags, group_counter, "user_group", limit)

        for iid in self.global_top:
            self._add_item(out, source_flags, str(iid), "global", limit)
            if len(out) >= limit:
                break

        return {
            "items": out[:limit],
            "base_items": base_items,
            "base_ranks": base_ranks,
            "source_flags": source_flags,
            "hist": hist,
            "counts": counts,
            "recency": recency,
            "recent_unique": recent_unique,
            "hist_item_stats_inputs": hist_item_stats_inputs,
            "item_feature_stats_inputs": item_feature_stats_inputs,
            "user_group_stats_inputs": group_stats_inputs,
        }

    def candidate_pool_for_row(self, row: pd.Series, k: int = 80) -> List[str]:
        return list(self._candidate_context(row, pool_size=k)["items"])[:k]

    def _features_for(self, row: pd.Series, iid: str, context: Dict[str, Any]) -> np.ndarray:
        hist: List[str] = context["hist"]
        counts: Counter = context["counts"]
        recency: Dict[str, int] = context["recency"]
        base_rank = int(context["base_ranks"].get(iid, self.candidate_pool_size + 1))
        source_flags = context["source_flags"].get(iid, set())

        hist_len = len(hist)
        unique_count = len(counts)
        max_count = max(counts.values()) if counts else 0
        repeat_ratio = 0.0 if hist_len <= 0 else 1.0 - (unique_count / hist_len)
        top_share = 0.0 if hist_len <= 0 else max_count / hist_len
        candidate_count = float(counts.get(iid, 0))
        candidate_recency = (float(recency.get(iid, -1)) + 1.0) / max(1.0, float(hist_len)) if iid in recency else 0.0

        last_stats = self._counter_stats(
            self.last_counts.get(hist[-1]) if hist else None,
            float(self.last_totals.get(hist[-1], 0)) if hist else 0.0,
            iid,
        )
        suffix2_key = (hist[-2], hist[-1]) if len(hist) >= 2 else None
        suffix2_stats = self._counter_stats(
            self.suffix2_counts.get(suffix2_key) if suffix2_key else None,
            float(self.suffix2_totals.get(suffix2_key, 0)) if suffix2_key else 0.0,
            iid,
        )
        hist_item_stats = self._best_stats(
            context.get("hist_item_stats_inputs", []),
            iid,
        )

        user_group_stats = self._best_stats(context.get("user_group_stats_inputs", []), iid)

        item_feature_stats = self._best_stats(
            context.get("item_feature_stats_inputs", []),
            iid,
        )

        values = [
            float(base_rank),
            0.0 if base_rank > self.candidate_pool_size else 1.0 / math.log2(base_rank + 1),
            1.0 if base_rank <= 10 else 0.0,
            1.0 if base_rank <= 25 else 0.0,
            1.0 if base_rank <= 50 else 0.0,
            1.0 if base_rank <= 80 else 0.0,
            float(hist_len),
            math.log1p(hist_len),
            1.0 if hist_len <= 3 else 0.0,
            1.0 if 4 <= hist_len <= 10 else 0.0,
            1.0 if 11 <= hist_len <= 20 else 0.0,
            1.0 if 21 <= hist_len <= 30 else 0.0,
            1.0 if 31 <= hist_len <= 80 else 0.0,
            1.0 if hist_len > 80 else 0.0,
            repeat_ratio,
            top_share,
            math.log1p(max_count),
            unique_count / max(1.0, float(hist_len)),
            math.log1p(candidate_count),
            candidate_count / max(1.0, float(max_count)),
            candidate_recency,
            1.0 if hist and iid == hist[-1] else 0.0,
            math.log1p(last_stats[0]),
            last_stats[1],
            last_stats[2],
            math.log1p(suffix2_stats[0]),
            suffix2_stats[1],
            suffix2_stats[2],
            math.log1p(hist_item_stats[0]),
            hist_item_stats[1],
            hist_item_stats[2],
            math.log1p(user_group_stats[0]),
            user_group_stats[1],
            user_group_stats[2],
            math.log1p(item_feature_stats[0]),
            item_feature_stats[1],
            item_feature_stats[2],
            math.log1p(float(self.global_counts.get(iid, 0))),
            self._global_prob(iid),
            1.0 if "base" in source_flags else 0.0,
            1.0 if "repeat" in source_flags else 0.0,
            1.0 if "last" in source_flags else 0.0,
            1.0 if "suffix2" in source_flags else 0.0,
            1.0 if "hist_item" in source_flags else 0.0,
            1.0 if "user_group" in source_flags else 0.0,
            1.0 if "item_feature" in source_flags else 0.0,
            1.0 if "global" in source_flags else 0.0,
        ]
        return np.asarray(values, dtype=np.float32)

    def _score_candidates(self, row: pd.Series, items: Sequence[str], context: Dict[str, Any]) -> List[Dict[str, Any]]:
        if self.booster is None or not items:
            return []
        features = [self._features_for(row, iid, context) for iid in items]
        x_values = np.vstack(features)
        lgbm_scores = np.asarray(self.booster.predict(x_values), dtype=np.float64)
        base_ranks = context["base_ranks"]
        scored: List[Dict[str, Any]] = []
        for iid, feature_values, lgbm_score in zip(items, features, lgbm_scores):
            source_flags = set(context["source_flags"].get(iid, set()))
            support_log = max(
                float(feature_values[self.feature_index["last_support_log"]]),
                float(feature_values[self.feature_index["suffix2_support_log"]]),
                float(feature_values[self.feature_index["hist_item_support_log"]]),
                float(feature_values[self.feature_index["user_group_support_log"]]),
                float(feature_values[self.feature_index["item_feature_support_log"]]),
            )
            log_lift = max(
                float(feature_values[self.feature_index["last_log_lift"]]),
                float(feature_values[self.feature_index["suffix2_log_lift"]]),
                float(feature_values[self.feature_index["hist_item_log_lift_max"]]),
                float(feature_values[self.feature_index["user_group_log_lift_max"]]),
                float(feature_values[self.feature_index["item_feature_log_lift_max"]]),
            )
            base_rank_score = float(feature_values[self.feature_index["base_rank_score"]])
            scored.append(
                {
                    "iid": iid,
                    "lgbm_score": float(lgbm_score),
                    "combined_score": float(lgbm_score) + self.base_score_weight * base_rank_score,
                    "base_rank": int(base_ranks.get(iid, 10**6)),
                    "base_rank_score": base_rank_score,
                    "source_flags": source_flags,
                    "source_count": len(source_flags),
                    "support_log": support_log,
                    "log_lift": log_lift,
                }
            )
        by_lgbm = sorted(scored, key=lambda item: (-float(item["lgbm_score"]), int(item["base_rank"]), item["iid"]))
        for rank, item in enumerate(by_lgbm, start=1):
            item["lgbm_rank"] = rank
        return scored

    def _passes_insert_gate(self, scored: Dict[str, Any]) -> bool:
        if self.lgbm_rank_top_n > 0 and int(scored.get("lgbm_rank", 10**6)) > self.lgbm_rank_top_n:
            return False
        if int(scored.get("source_count", 0)) < self.min_source_count:
            return False
        source_flags = set(scored.get("source_flags", set()))
        if self.require_non_base_source and not any(src not in {"base", "global"} for src in source_flags):
            return False
        if float(scored.get("support_log", 0.0)) < self.min_gate_support_log:
            return False
        if float(scored.get("log_lift", -1e9)) < self.min_gate_log_lift:
            return False
        return True

    def _predict_blend_rerank(
        self,
        row: pd.Series,
        base_items: List[str],
        context: Dict[str, Any],
        k: int,
    ) -> List[str]:
        frozen = base_items[: self.freeze_top_n]
        frozen_set = set(frozen)
        tail = [iid for iid in context["items"] if iid not in frozen_set]
        if not tail:
            return base_items[:k]
        scored = self._score_candidates(row, tail, context)
        scored_by_iid = {item["iid"]: item for item in scored}
        ordered_tail = [
            item["iid"]
            for item in sorted(
                scored,
                key=lambda item: (-float(item["combined_score"]), int(item["base_rank"]), item["iid"]),
            )
        ]
        out: List[str] = []
        for iid in frozen + ordered_tail + base_items:
            if iid not in out:
                out.append(iid)
            if len(out) >= k:
                break
        proposal = out[:k]
        if self.prediction_mode == "segment_blend_gate_rerank" and not self._passes_blend_row_gate(
            base_items[:k],
            proposal,
            scored_by_iid,
            k,
        ):
            return base_items[:k]
        return proposal

    def _passes_blend_row_gate(
        self,
        base_top: List[str],
        proposal: List[str],
        scored_by_iid: Dict[str, Dict[str, Any]],
        k: int,
    ) -> bool:
        if base_top[:k] == proposal[:k]:
            return False
        start = min(k, max(0, self.freeze_top_n))
        changed_positions = [
            pos
            for pos in range(start, min(k, len(base_top), len(proposal)))
            if base_top[pos] != proposal[pos]
        ]
        if not changed_positions:
            return False
        if self.blend_max_changed_positions >= 0 and len(changed_positions) > self.blend_max_changed_positions:
            return False

        base_set = set(base_top[:k])
        new_iids = [iid for iid in proposal[start:k] if iid not in base_set]
        if self.blend_max_new_items >= 0 and len(new_iids) > self.blend_max_new_items:
            return False
        if self.blend_require_new_item_gate:
            for iid in new_iids:
                scored = scored_by_iid.get(iid)
                if scored is None or not self._passes_insert_gate(scored):
                    return False

        margins: List[float] = []
        for pos in changed_positions:
            proposal_item = scored_by_iid.get(proposal[pos])
            base_item = scored_by_iid.get(base_top[pos])
            if proposal_item is None or base_item is None:
                return False
            margins.append(float(proposal_item["combined_score"]) - float(base_item["combined_score"]))
        if not margins:
            return False
        if min(margins) < self.blend_min_position_margin:
            return False
        if sum(margins) < self.blend_min_total_margin:
            return False
        return True

    def _predict_insert_gate(
        self,
        row: pd.Series,
        base_items: List[str],
        context: Dict[str, Any],
        k: int,
    ) -> List[str]:
        base_top = list(base_items[:k])
        if len(base_top) < k:
            return base_top
        frozen_n = min(k, max(self.freeze_top_n, self.insert_start_rank - 1))
        frozen = base_top[:frozen_n]
        base_tail = [iid for iid in base_top[frozen_n:k] if iid not in set(frozen)]
        if not base_tail:
            return base_top

        base_top_set = set(base_top)
        candidate_items = [iid for iid in context["items"] if iid not in set(frozen)]
        scored = self._score_candidates(row, candidate_items, context)
        scored_by_iid = {item["iid"]: item for item in scored}
        tail_scores = [scored_by_iid.get(iid) for iid in base_tail if iid in scored_by_iid]
        if not tail_scores:
            return base_top
        cutoff = min(float(item["combined_score"]) for item in tail_scores) + self.min_combined_margin
        promoted = [
            item
            for item in scored
            if item["iid"] not in base_top_set
            and float(item["combined_score"]) > cutoff
            and self._passes_insert_gate(item)
        ]
        promoted = sorted(
            promoted,
            key=lambda item: (-float(item["combined_score"]), int(item["base_rank"]), item["iid"]),
        )[: max(0, min(self.max_insertions, len(base_tail)))]
        if not promoted:
            return base_top

        promoted_iids = [item["iid"] for item in promoted]
        if self.allow_base_tail_reorder:
            eligible_iids = base_tail + promoted_iids
            ordered_tail = [
                item["iid"]
                for item in sorted(
                    [scored_by_iid[iid] for iid in eligible_iids if iid in scored_by_iid],
                    key=lambda item: (-float(item["combined_score"]), int(item["base_rank"]), item["iid"]),
                )
            ][: len(base_tail)]
        else:
            keep_count = max(0, len(base_tail) - len(promoted_iids))
            ordered_tail = base_tail[:keep_count] + promoted_iids[: len(base_tail) - keep_count]

        out: List[str] = []
        for iid in frozen + ordered_tail + base_items:
            if iid not in out:
                out.append(iid)
            if len(out) >= k:
                break
        return out[:k]

    def predict_row(self, row: pd.Series, k: int = 10) -> List[str]:
        base_pool_items = self.base_model.predict_row(row, k=max(self.candidate_pool_size, k))
        if self.output_base_k <= k:
            base_items = self.base_model.predict_row(row, k=k)
        else:
            base_items = base_pool_items
        hist = parse_sequence(row.get("item_seq_raw", ""))
        eval_len = int(row.get("_masked_raw_len", len(hist))) if "_masked_raw_len" in row.index else len(hist)
        exact_bin = exact_length_bin(eval_len)
        if self.booster is None:
            return base_items[:k]

        saved_state: Optional[Dict[str, Any]] = None
        if self.segment_gate_profiles:
            profile = self.segment_gate_profiles.get(exact_bin)
            if profile is None:
                return base_items[:k]
            saved_state = self._prediction_state()
            self._apply_segment_profile(profile)

        if not self.enable_short_history and eval_len < self.min_rank_len:
            if saved_state is not None:
                self._restore_prediction_state(saved_state)
            return base_items[:k]
        if self.prediction_exact_bins and exact_bin not in self.prediction_exact_bins:
            if saved_state is not None:
                self._restore_prediction_state(saved_state)
            return base_items[:k]
        try:
            context = self._candidate_context(row, pool_size=self.candidate_pool_size)
            if self.prediction_mode == "insert_gate":
                return self._predict_insert_gate(row, base_items, context, k)
            if self.prediction_mode in {"blend_rerank", "segment_blend_rerank", "segment_blend_gate_rerank"}:
                return self._predict_blend_rerank(row, base_items, context, k)
            if self.prediction_mode == "uplift_gate":
                return self._predict_insert_gate(row, base_items, context, k)

            frozen = base_items[: self.freeze_top_n]
            tail = [iid for iid in context["items"] if iid not in set(frozen)]
            if not tail:
                return base_items[:k]
            scored = self._score_candidates(row, tail, context)
            ordered_tail = [
                item["iid"]
                for item in sorted(
                    scored,
                    key=lambda item: (-float(item["lgbm_score"]), int(item["base_rank"]), item["iid"]),
                )
            ]
            out: List[str] = []
            for iid in frozen + ordered_tail + base_items:
                if iid not in out:
                    out.append(iid)
                if len(out) >= k:
                    break
            return out[:k]
        finally:
            if saved_state is not None:
                self._restore_prediction_state(saved_state)


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


def v19a_short_len3_only_alpha04_config() -> Dict[str, Any]:
    return {
        "name": "v19a_short_len3_only_alpha04",
        "model": "segment_bayes_short_rerank",
        "base_config": v17_conservative_shortseq_config(),
        "target_lengths": (3,),
        "freeze_top_n": 1,
        "alpha": 0.04,
        "alpha_by_len": {3: 0.04},
        "base_rank_weight": 1.0,
        "min_count": 8,
        "min_group_count": 12,
        "shrink_beta": 60.0,
        "min_lift": 1.02,
        "prior_pool": 80,
        "suffix_orders": (2,),
        "group_specs": (
            ("u_cat_01", "u_cat_02"),
            ("u_cat_01", "u_cat_06"),
            ("u_cat_02", "u_cat_06"),
        ),
        "component_weights": {
            "last": 0.8,
            "suffix": 0.35,
            "last_group": 1.0,
            "suffix_group": 0.75,
        },
    }


def v19b_short_len1_3_alpha035_050_config() -> Dict[str, Any]:
    config = v19a_short_len3_only_alpha04_config()
    config.update(
        {
            "name": "v19b_short_len1_3_alpha035_050",
            "target_lengths": (1, 3),
            "alpha_by_len": {1: 0.035, 3: 0.05},
        }
    )
    return config


def v22_long_history_count_recent_config() -> Dict[str, Any]:
    return {
        "name": "v22_long_history_count_recent_len31",
        "model": "long_history_count_rerank",
        "base_config": v17_conservative_shortseq_config(),
        "min_len": 31,
        "freeze_top_n": 1,
        "base_pool": 20,
        "base_rank_weight": 1.0,
        "alpha": 0.10,
        "count_weight": 0.8,
        "recency_weight": 0.4,
        "last_weight": 0.1,
    }


def v23_long_history_count_recent_strong_config() -> Dict[str, Any]:
    config = v22_long_history_count_recent_config()
    config.update(
        {
            "name": "v23_long_history_count_recent_len31_alpha20",
            "alpha": 0.20,
        }
    )
    return config


def v24_medium_long_history_count_recent_strong_config() -> Dict[str, Any]:
    config = v22_long_history_count_recent_config()
    config.update(
        {
            "name": "v24_medium_long_history_count_recent_len21_alpha25",
            "min_len": 21,
            "alpha": 0.25,
        }
    )
    return config


def v25_medium_long_history_pool25_config() -> Dict[str, Any]:
    config = v22_long_history_count_recent_config()
    config.update(
        {
            "name": "v25_medium_long_history_count_recent_len21_pool25_alpha32",
            "min_len": 21,
            "base_pool": 25,
            "alpha": 0.32,
        }
    )
    return config


def v29_medium_long_count_only_pool25_config() -> Dict[str, Any]:
    config = v22_long_history_count_recent_config()
    config.update(
        {
            "name": "v29_medium_long_count_only_len21_pool25_alpha32",
            "min_len": 21,
            "base_pool": 25,
            "alpha": 0.32,
            "count_weight": 1.0,
            "recency_weight": 0.0,
            "last_weight": 0.0,
        }
    )
    return config


def v30_medium_long_count_only_pool25_cw125_config() -> Dict[str, Any]:
    config = v29_medium_long_count_only_pool25_config()
    config.update(
        {
            "name": "v30_medium_long_count_only_len21_pool25_alpha28_cw125",
            "alpha": 0.28,
            "count_weight": 1.25,
        }
    )
    return config


def v31_covisit_lgbm_ranker_config() -> Dict[str, Any]:
    return {
        "name": "v31_covisit_lgbm_ranker",
        "model": "candidate_lgbm_ranker",
        "base_config": v29_medium_long_count_only_pool25_config(),
        "candidate_pool_size": 80,
        "freeze_top_n": 1,
        "enable_short_history": False,
        "min_rank_len": 4,
        "counter_top_n": 20,
        "max_hist_items": 80,
        "max_train_groups": 8000,
        "shrink_beta": 80.0,
        "global_smooth": 1.0,
        "seed": 42,
        "lgbm_params": {
            "objective": "lambdarank",
            "metric": "ndcg",
            "ndcg_eval_at": [10],
            "learning_rate": 0.03,
            "num_leaves": 31,
            "min_data_in_leaf": 50,
            "num_boost_round": 300,
            "early_stopping_rounds": 30,
        },
    }


def v32_ranker_gate_blend_config(
    name: str = "v32_ranker_gate_blend_f5_i6_l4_ins2",
    *,
    freeze_top_n: int = 5,
    insert_start_rank: int = 6,
    base_score_weight: float = 4.0,
    max_insertions: int = 2,
    min_source_count: int = 2,
    lgbm_rank_top_n: int = 15,
    min_gate_support_log: float = math.log1p(1.0),
    min_gate_log_lift: float = 0.0,
    min_combined_margin: float = 0.0,
    allow_base_tail_reorder: bool = False,
) -> Dict[str, Any]:
    config = v31_covisit_lgbm_ranker_config()
    config.update(
        {
            "name": name,
            "prediction_mode": "insert_gate",
            "freeze_top_n": freeze_top_n,
            "insert_start_rank": insert_start_rank,
            "base_score_weight": base_score_weight,
            "max_insertions": max_insertions,
            "min_source_count": min_source_count,
            "require_non_base_source": True,
            "lgbm_rank_top_n": lgbm_rank_top_n,
            "min_gate_support_log": min_gate_support_log,
            "min_gate_log_lift": min_gate_log_lift,
            "min_combined_margin": min_combined_margin,
            "allow_base_tail_reorder": allow_base_tail_reorder,
            "output_base_k": 10,
        }
    )
    return config


def v32_ranker_gate_gt80_config() -> Dict[str, Any]:
    config = v32_ranker_gate_blend_config(
        name="v32_ranker_gate_gt80_f5_i6_l8_ins2",
        freeze_top_n=5,
        insert_start_rank=6,
        base_score_weight=8.0,
        max_insertions=2,
        min_source_count=2,
        lgbm_rank_top_n=12,
    )
    config["prediction_exact_bins"] = (">80",)
    return config


def _v33_segment_profile(
    *,
    prediction_mode: str = "insert_gate",
    freeze_top_n: int = 5,
    insert_start_rank: int = 6,
    base_score_weight: float = 8.0,
    max_insertions: int = 2,
    min_source_count: int = 2,
    lgbm_rank_top_n: int = 12,
    min_gate_support_log: float = math.log1p(1.0),
    min_gate_log_lift: float = 0.0,
    min_combined_margin: float = 0.0,
    allow_base_tail_reorder: bool = False,
) -> Dict[str, Any]:
    return {
        "prediction_mode": prediction_mode,
        "freeze_top_n": freeze_top_n,
        "insert_start_rank": insert_start_rank,
        "base_score_weight": base_score_weight,
        "max_insertions": max_insertions,
        "min_source_count": min_source_count,
        "require_non_base_source": True,
        "lgbm_rank_top_n": lgbm_rank_top_n,
        "min_gate_support_log": min_gate_support_log,
        "min_gate_log_lift": min_gate_log_lift,
        "min_combined_margin": min_combined_margin,
        "allow_base_tail_reorder": allow_base_tail_reorder,
        "output_base_k": 10,
    }


def _v33_base_config(name: str, candidate_pool_size: int = 120) -> Dict[str, Any]:
    config = v31_covisit_lgbm_ranker_config()
    config.update(
        {
            "name": name,
            "candidate_pool_size": candidate_pool_size,
            "output_base_k": 10,
            "prediction_mode": "insert_gate",
            "freeze_top_n": 5,
            "insert_start_rank": 6,
            "base_score_weight": 8.0,
            "max_insertions": 2,
            "min_source_count": 2,
            "require_non_base_source": True,
            "lgbm_rank_top_n": 12,
            "min_gate_support_log": math.log1p(1.0),
            "min_gate_log_lift": 0.0,
            "min_combined_margin": 0.0,
            "allow_base_tail_reorder": False,
        }
    )
    return config


def v33a_gt80_ins4_pool120_config() -> Dict[str, Any]:
    config = _v33_base_config("v33a_gt80_ins4_pool120", candidate_pool_size=120)
    config["segment_gate_profiles"] = {
        ">80": _v33_segment_profile(
            freeze_top_n=5,
            insert_start_rank=6,
            base_score_weight=8.0,
            max_insertions=4,
            min_source_count=2,
            lgbm_rank_top_n=20,
            min_gate_support_log=math.log1p(1.0),
        )
    }
    return config


def v33b_gt80_top3_pool120_config() -> Dict[str, Any]:
    config = _v33_base_config("v33b_gt80_top3_pool120", candidate_pool_size=120)
    config["segment_gate_profiles"] = {
        ">80": _v33_segment_profile(
            freeze_top_n=3,
            insert_start_rank=4,
            base_score_weight=8.0,
            max_insertions=3,
            min_source_count=2,
            lgbm_rank_top_n=15,
            min_gate_support_log=math.log1p(1.0),
        )
    }
    return config


def v33c_3180_strict_gt80_aggressive_config() -> Dict[str, Any]:
    config = _v33_base_config("v33c_3180_strict_gt80_aggressive", candidate_pool_size=120)
    config["segment_gate_profiles"] = {
        ">80": _v33_segment_profile(
            freeze_top_n=5,
            insert_start_rank=6,
            base_score_weight=8.0,
            max_insertions=4,
            min_source_count=2,
            lgbm_rank_top_n=20,
            min_gate_support_log=math.log1p(1.0),
        ),
        "31-80": _v33_segment_profile(
            freeze_top_n=7,
            insert_start_rank=8,
            base_score_weight=16.0,
            max_insertions=1,
            min_source_count=3,
            lgbm_rank_top_n=8,
            min_gate_support_log=math.log1p(2.0),
            min_gate_log_lift=0.10,
            min_combined_margin=0.10,
        ),
    }
    return config


def v33d_pool200_tail_probe_config() -> Dict[str, Any]:
    config = _v33_base_config("v33d_pool200_tail_probe", candidate_pool_size=200)
    config["segment_gate_profiles"] = {
        ">80": _v33_segment_profile(
            freeze_top_n=5,
            insert_start_rank=6,
            base_score_weight=8.0,
            max_insertions=3,
            min_source_count=2,
            lgbm_rank_top_n=25,
            min_gate_support_log=math.log1p(1.0),
        )
    }
    return config


def v33e_segment_blend_gt80_config(lambda_weight: float = 8.0) -> Dict[str, Any]:
    lambda_label = str(lambda_weight).replace(".", "p")
    config = _v33_base_config(f"v33e_segment_blend_gt80_l{lambda_label}", candidate_pool_size=120)
    config["segment_gate_profiles"] = {
        ">80": _v33_segment_profile(
            prediction_mode="segment_blend_rerank",
            freeze_top_n=3,
            insert_start_rank=4,
            base_score_weight=lambda_weight,
            max_insertions=7,
            min_source_count=0,
            lgbm_rank_top_n=120,
            min_gate_support_log=0.0,
            min_gate_log_lift=-1e9,
            allow_base_tail_reorder=True,
        )
    }
    return config


def v33f_uplift_label_gate_config() -> Dict[str, Any]:
    config = _v33_base_config("v33f_uplift_label_gate", candidate_pool_size=120)
    config["ranker_label_mode"] = "uplift"
    config["segment_gate_profiles"] = {
        ">80": _v33_segment_profile(
            prediction_mode="uplift_gate",
            freeze_top_n=5,
            insert_start_rank=6,
            base_score_weight=8.0,
            max_insertions=3,
            min_source_count=2,
            lgbm_rank_top_n=15,
            min_gate_support_log=math.log1p(1.0),
        )
    }
    return config


def _score_threshold_label(value: float) -> str:
    label = f"{value:.3f}".rstrip("0").rstrip(".")
    if label == "-0":
        label = "0"
    return label.replace("-", "m").replace(".", "p")


def v34_precise_blend_gate_gt80_config(
    *,
    lambda_weight: float = 8.0,
    min_position_margin: float = 0.05,
    min_total_margin: float = 0.10,
    max_changed_positions: int = 3,
    max_new_items: int = 1,
    require_new_item_gate: bool = False,
    min_source_count: int = 0,
    lgbm_rank_top_n: int = 120,
) -> Dict[str, Any]:
    name = (
        "v34_blend_gate_gt80"
        f"_l{_score_threshold_label(lambda_weight)}"
        f"_pm{_score_threshold_label(min_position_margin)}"
        f"_tm{_score_threshold_label(min_total_margin)}"
        f"_c{max_changed_positions}"
        f"_n{max_new_items}"
        f"{'_sg' if require_new_item_gate else ''}"
    )
    config = _v33_base_config(name, candidate_pool_size=120)
    profile = _v33_segment_profile(
        prediction_mode="segment_blend_gate_rerank",
        freeze_top_n=3,
        insert_start_rank=4,
        base_score_weight=lambda_weight,
        max_insertions=7,
        min_source_count=min_source_count,
        lgbm_rank_top_n=lgbm_rank_top_n,
        min_gate_support_log=math.log1p(1.0) if require_new_item_gate else 0.0,
        min_gate_log_lift=0.0 if require_new_item_gate else -1e9,
        allow_base_tail_reorder=True,
    )
    profile.update(
        {
            "blend_min_position_margin": min_position_margin,
            "blend_min_total_margin": min_total_margin,
            "blend_max_changed_positions": max_changed_positions,
            "blend_max_new_items": max_new_items,
            "blend_require_new_item_gate": require_new_item_gate,
        }
    )
    config["segment_gate_profiles"] = {">80": profile}
    return config


def v27_medium_long_conditional_repeat_gate_config() -> Dict[str, Any]:
    return {
        "name": "v27_medium_long_conditional_repeat_gate",
        "model": "medium_long_conditional_repeat_gate",
        "base_config": v24_medium_long_history_count_recent_strong_config(),
        "min_len": 21,
        "freeze_top_n": 1,
        "base_pool": 30,
        "base_rank_weight": 1.0,
        "alpha_repeat": 0.18,
        "alpha_prior": 0.12,
        "count_weight": 0.75,
        "recency_weight": 0.25,
        "last_weight": 0.10,
        "min_count": 20,
        "min_group_count": 40,
        "shrink_beta": 80.0,
        "min_lift": 1.08,
        "group_cols": ("u_cat_01", "u_cat_02", "u_cat_06"),
        "suffix_orders": (2,),
    }


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
        configs.insert(0, v22_long_history_count_recent_config())
        configs.insert(0, v23_long_history_count_recent_strong_config())
        configs.insert(0, v25_medium_long_history_pool25_config())
        configs.insert(0, v24_medium_long_history_count_recent_strong_config())
        configs.insert(0, v29_medium_long_count_only_pool25_config())
        configs.insert(0, v30_medium_long_count_only_pool25_cw125_config())
        # v32 online probe requested after v31 showed useful recall but unsafe
        # free reranking. Keep v29 as the base and only gate insertions for >80.
        configs.insert(0, v32_ranker_gate_gt80_config())
        # v31 ranker remains available as an explicit probe, but is not a
        # default candidate after its official A-board regression.
        configs.insert(1, v27_medium_long_conditional_repeat_gate_config())
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
    if config.get("model") == "segment_bayes_short_rerank":
        return SegmentBayesShortRerankRecommender(config=config, candidates=candidates, user_df=user_df, item_df=item_df)
    if config.get("model") == "long_history_count_rerank":
        return LongHistoryCountRerankRecommender(config=config, candidates=candidates, user_df=user_df, item_df=item_df)
    if config.get("model") == "medium_long_conditional_repeat_gate":
        return MediumLongConditionalRepeatGateRecommender(
            config=config,
            candidates=candidates,
            user_df=user_df,
            item_df=item_df,
        )
    if config.get("model") == "candidate_lgbm_ranker":
        return CandidateLGBMRankerRecommender(config=config, candidates=candidates, user_df=user_df, item_df=item_df)
    return HybridRecommender(config=config, candidates=candidates, user_df=user_df, item_df=item_df)


def fit_recommender_for_config(
    config: Dict[str, Any],
    train_df: pd.DataFrame,
    candidates: List[str],
    user_df: pd.DataFrame,
    item_df: pd.DataFrame,
    *,
    ranker_mask_source_df: Optional[pd.DataFrame] = None,
    val_ratio: float = 0.2,
    seed: int = 42,
):
    model = build_recommender(config, candidates, user_df, item_df)
    if config.get("model") != "candidate_lgbm_ranker":
        return model.fit(train_df)

    if ranker_mask_source_df is None or "target_iid" not in train_df.columns:
        return model.fit(train_df)

    _, ranker_label_df = recommendation_split(train_df, val_ratio=val_ratio, seed=seed + 700)
    if len(ranker_label_df) == 0:
        return model.fit(train_df)

    ranker_train_df = masked_history_eval(ranker_label_df, ranker_mask_source_df, seed=seed + 1700)
    ranker_train_df["target_iid"] = ranker_label_df["target_iid"].astype(str).values
    return model.fit(train_df, ranker_train_df=ranker_train_df)


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
    ranker_mask_source_df: Optional[pd.DataFrame] = None,
    seed: int = 42,
    val_ratio: float = 0.2,
) -> Dict[str, Any]:
    model = fit_recommender_for_config(
        config,
        fit_df,
        candidates,
        user_df,
        item_df,
        ranker_mask_source_df=ranker_mask_source_df,
        seed=seed,
        val_ratio=val_ratio,
    )
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


def _bounded_int(value: Any, default: int, lower: int, upper: int) -> int:
    try:
        parsed = int(round(float(value)))
    except (TypeError, ValueError):
        parsed = default
    return max(lower, min(upper, parsed))


def _bounded_float(value: Any, default: float, lower: float, upper: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        parsed = default
    return max(lower, min(upper, parsed))


def _safe_config_name(prefix: str, raw_name: Any, index: int) -> str:
    raw_text = str(raw_name or "").strip().lower()
    cleaned = "".join(ch if ch.isalnum() else "_" for ch in raw_text)
    cleaned = "_".join(part for part in cleaned.split("_") if part)
    if not cleaned:
        cleaned = f"candidate_{index}"
    if not cleaned.startswith(prefix):
        cleaned = f"{prefix}_{index:02d}_{cleaned}"
    return cleaned[:96]


def _rec_data_summary(train_df: pd.DataFrame, test_df: pd.DataFrame, item_df: pd.DataFrame) -> Dict[str, Any]:
    train_lengths = train_df["item_seq_raw"].map(lambda x: len(parse_sequence(x)))
    test_lengths = test_df["item_seq_raw"].map(lambda x: len(parse_sequence(x)))
    return {
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "candidate_items": int(len(item_df)),
        "train_unique_targets": int(train_df["target_iid"].astype(str).nunique()),
        "test_zero_history_rows": int(test_lengths.eq(0).sum()),
        "test_length_quantiles": {
            str(q): float(test_lengths.quantile(q)) for q in (0.0, 0.25, 0.5, 0.75, 0.9, 1.0)
        },
        "train_length_quantiles": {
            str(q): float(train_lengths.quantile(q)) for q in (0.0, 0.25, 0.5, 0.75, 0.9, 1.0)
        },
    }


def _qwen_rec_base_config() -> Dict[str, Any]:
    if importlib.util.find_spec("torch") is not None:
        return v17_conservative_shortseq_config()
    return v2_weight_tuned_config()


def _sanitize_qwen_rec_config(raw: Any, index: int) -> Tuple[Optional[Dict[str, Any]], str]:
    if not isinstance(raw, dict):
        return None, "suggestion is not an object"
    model_name = str(raw.get("model", "long_history_count_rerank"))
    if model_name != "long_history_count_rerank":
        return None, f"unsupported model: {model_name}"

    freeze_top_n = _bounded_int(raw.get("freeze_top_n"), 1, 0, 3)
    base_pool = _bounded_int(raw.get("base_pool"), 25, 10, 120)
    base_pool = max(base_pool, freeze_top_n + 10)
    config = {
        "name": _safe_config_name("qwen_rec", raw.get("name"), index),
        "model": "long_history_count_rerank",
        "base_config": _qwen_rec_base_config(),
        "min_len": _bounded_int(raw.get("min_len"), 21, 4, 80),
        "freeze_top_n": freeze_top_n,
        "base_pool": base_pool,
        "base_rank_weight": _bounded_float(raw.get("base_rank_weight"), 1.0, 0.2, 3.0),
        "alpha": _bounded_float(raw.get("alpha"), 0.25, 0.0, 1.2),
        "count_weight": _bounded_float(raw.get("count_weight"), 0.8, 0.0, 3.0),
        "recency_weight": _bounded_float(raw.get("recency_weight"), 0.4, 0.0, 3.0),
        "last_weight": _bounded_float(raw.get("last_weight"), 0.1, 0.0, 3.0),
    }
    return config, ""


def qwen_recommendation_configs(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    item_df: pd.DataFrame,
    trial_summaries: List[Dict[str, Any]],
    bin_weights: Dict[str, float],
    requested: int,
    env_path: str | Path,
    model: Optional[str],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    report: Dict[str, Any] = {
        "enabled": True,
        "requested": int(requested),
        "status": "not_called",
        "accepted_count": 0,
        "rejected_count": 0,
        "accepted_names": [],
        "rejections": [],
    }
    if requested <= 0:
        report["status"] = "no_candidates_requested"
        return [], report

    system = (
        "You tune an anonymized sparse-feedback recommender. "
        "Return only valid JSON. Propose configs; do not invent code."
    )
    user = {
        "task": "Suggest recommendation configs for local validation.",
        "allowed_schema": {
            "suggestions": [
                {
                    "name": "qwen_rec_short_name",
                    "model": "long_history_count_rerank",
                    "min_len": "integer 4..80",
                    "freeze_top_n": "integer 0..3",
                    "base_pool": "integer 10..120",
                    "base_rank_weight": "float 0.2..3.0",
                    "alpha": "float 0.0..1.2",
                    "count_weight": "float 0.0..3.0",
                    "recency_weight": "float 0.0..3.0",
                    "last_weight": "float 0.0..3.0",
                    "reason": "brief rationale",
                }
            ]
        },
        "constraints": [
            "Use only model=long_history_count_rerank.",
            "Prefer materially different configs over tiny perturbations.",
            "The local validator will reject configs that do not beat the current best.",
            f"Return at most {requested} suggestions.",
        ],
        "data_summary": _rec_data_summary(train_df, test_df, item_df),
        "test_bin_weights": bin_weights,
        "validated_trials": trial_summaries[-10:],
    }
    parsed, api_report = ask_qwen_json(
        [
            {"role": "system", "content": system},
            {"role": "user", "content": compact_json(user, max_chars=9000)},
        ],
        env_path=env_path,
        model=model,
        temperature=0.25,
        max_tokens=1800,
    )
    report.update(api_report)
    if api_report.get("status") != "ok":
        return [], report
    suggestions = parsed.get("suggestions", parsed) if isinstance(parsed, dict) else parsed
    if not isinstance(suggestions, list):
        report["status"] = "invalid_payload"
        report["rejections"].append("payload does not contain a suggestions list")
        return [], report

    configs: List[Dict[str, Any]] = []
    seen_names = set()
    for index, raw in enumerate(suggestions[:requested], start=1):
        config, error = _sanitize_qwen_rec_config(raw, index)
        if config is None:
            report["rejections"].append(error)
            continue
        name = str(config["name"])
        if name in seen_names:
            report["rejections"].append(f"duplicate config name: {name}")
            continue
        seen_names.add(name)
        configs.append(config)

    report["accepted_count"] = len(configs)
    report["rejected_count"] = len(report["rejections"])
    report["accepted_names"] = [str(config["name"]) for config in configs]
    return configs, report


def run_recommendation(
    data_dir: str | Path,
    output_dir: str | Path,
    budget: int = 5,
    seed: int = 42,
    val_ratio: float = 0.12,
    time_limit: Optional[float] = None,
    use_qwen_agent: bool = False,
    qwen_env_path: str | Path = ".env",
    qwen_model: Optional[str] = None,
    qwen_rounds: int = 3,
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
        fit_df,
        val_df,
        masked_val_df,
        bin_weights,
        user_df,
        item_df,
        candidates,
        reference_config,
        ranker_mask_source_df=test_df,
        seed=seed,
        val_ratio=val_ratio,
    )
    reference_weighted = float(reference_metrics["test_weighted_ndcg@10"])
    best: Dict[str, Any] = {
        "test_weighted_ndcg@10": reference_weighted,
        "config": dict(reference_config),
        "metrics": reference_metrics,
    }
    qwen_agent_report: Dict[str, Any] = {
        "enabled": bool(use_qwen_agent),
        "requested": int(qwen_rounds),
        "status": "disabled",
        "accepted_count": 0,
        "rejected_count": 0,
        "accepted_names": [],
        "rejections": [],
    }
    trial_summaries: List[Dict[str, Any]] = []

    configs = all_configs[: max(1, budget)]
    round_id = 0

    def evaluate_and_record(config: Dict[str, Any], origin: str) -> None:
        nonlocal best, round_id
        round_id += 1
        if time_limit is not None and timer.elapsed > time_limit:
            return
        round_start = time.time()
        metrics = evaluate_config_details(
            fit_df,
            val_df,
            masked_val_df,
            bin_weights,
            user_df,
            item_df,
            candidates,
            config,
            ranker_mask_source_df=test_df,
            seed=seed,
            val_ratio=val_ratio,
        )
        feedback = {
            **metrics,
            "fit_size": int(len(fit_df)),
            "val_size": int(len(val_df)),
            "masked_val_size": int(len(masked_val_df)),
            "candidate_items": int(len(candidates)),
            "test_bin_weights": bin_weights,
            "reference_test_weighted_ndcg@10": reference_weighted,
            "origin": origin,
        }
        current_weighted = float(metrics["test_weighted_ndcg@10"])
        trial_summaries.append(
            {
                "round": round_id,
                "origin": origin,
                "name": config.get("name", ""),
                "model": config.get("model", ""),
                "plain_val_ndcg@10": metrics["plain_val_ndcg@10"],
                "masked_val_ndcg@10": metrics["masked_val_ndcg@10"],
                "test_weighted_ndcg@10": current_weighted,
                "natural_short_val_ndcg@10": metrics.get("natural_short_val_ndcg@10", 0.0),
                "by_bin_ndcg@10": metrics["by_bin_ndcg@10"],
                "config": config,
            }
        )
        if current_weighted >= float(best["test_weighted_ndcg@10"]):
            best = {
                "test_weighted_ndcg@10": current_weighted,
                "config": dict(config),
                "metrics": metrics,
            }
            strategy = (
                f"KEEP_AS_BEST; {origin} candidate improves or ties current best "
                "under test-distribution-weighted validation."
            )
            trajectory.best_round = round_id
            trajectory.selected_config = dict(config)
        else:
            strategy = f"REJECT; {origin} candidate did not improve over current best."
        trajectory.add(round_id, dict(config), feedback, strategy, time.time() - round_start)

    for config in configs:
        if time_limit is not None and timer.elapsed > time_limit:
            break
        evaluate_and_record(config, origin="deterministic")

    if use_qwen_agent and qwen_rounds > 0:
        if time_limit is not None and timer.elapsed > time_limit:
            qwen_agent_report["status"] = "time_limit_exhausted"
        else:
            qwen_configs, qwen_agent_report = qwen_recommendation_configs(
                train_df=train_df,
                test_df=test_df,
                item_df=item_df,
                trial_summaries=trial_summaries,
                bin_weights=bin_weights,
                requested=qwen_rounds,
                env_path=qwen_env_path,
                model=qwen_model,
            )
            for config in qwen_configs:
                if time_limit is not None and timer.elapsed > time_limit:
                    break
                evaluate_and_record(config, origin="qwen_agent")

    if float(best["test_weighted_ndcg@10"]) + 1e-12 < reference_weighted:
        raise RuntimeError(
            "Best recommendation candidate is below the v1 reference under test-weighted validation; not generating v2."
        )

    final_model = fit_recommender_for_config(
        best["config"],
        train_df,
        candidates,
        user_df,
        item_df,
        ranker_mask_source_df=test_df,
        seed=seed,
        val_ratio=val_ratio,
    )
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
        "qwen_agent": qwen_agent_report,
    }
    trajectory.selected_config = best["config"]
    write_json(output_dir / "trajectory_B2.json", trajectory.to_dict())
    write_json(output_dir / "recommendation_result.json", result)
    return result
