#!/usr/bin/env python3
"""Audit recommendation test segments for conditional-prior coverage."""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Sequence, Tuple

import pandas as pd

SOLUTION_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = SOLUTION_DIR.parent
if str(SOLUTION_DIR) not in sys.path:
    sys.path.insert(0, str(SOLUTION_DIR))

from src.common import ensure_dir, write_json  # noqa: E402
from src.recommendation import exact_length_bin, length_bin, parse_sequence  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit recommendation segment support coverage.")
    parser.add_argument("--rec_data", type=Path, default=PROJECT_ROOT / "A推荐")
    parser.add_argument(
        "--group_cols",
        nargs="+",
        default=["u_cat_01", "u_cat_02", "u_cat_06"],
        help="User feature columns used for group support.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=SOLUTION_DIR / "output" / "data_audit_rec_segments.json",
    )
    return parser.parse_args()


def load_data(rec_data: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_df = pd.read_csv(rec_data / "train.csv").astype({"uid": str, "target_iid": str})
    test_df = pd.read_csv(rec_data / "test.csv").astype({"uid": str})
    user_df = pd.read_csv(rec_data / "user.csv").astype({"uid": str})
    return train_df, test_df, user_df


def user_group_key(user_lookup: pd.DataFrame, uid: str, cols: Sequence[str]) -> Tuple[str, ...]:
    if user_lookup.empty or uid not in user_lookup.index:
        return ()
    row = user_lookup.loc[uid]
    if isinstance(row, pd.DataFrame):
        row = row.iloc[0]
    if not all(col in row.index and pd.notna(row[col]) for col in cols):
        return ()
    return tuple(str(row[col]) for col in cols)


def support_counters(
    train_df: pd.DataFrame,
    user_lookup: pd.DataFrame,
    group_cols: Sequence[str],
) -> Tuple[Counter, Counter, Counter]:
    last_support: Counter = Counter()
    suffix2_support: Counter = Counter()
    group_support: Counter = Counter()
    for row in train_df.itertuples(index=False):
        row_dict = row._asdict()
        hist = parse_sequence(row_dict.get("item_seq_raw", ""))
        if hist:
            last_support[hist[-1]] += 1
        if len(hist) >= 2:
            suffix2_support[tuple(hist[-2:])] += 1
        group_key = user_group_key(user_lookup, str(row_dict.get("uid", "")), group_cols)
        if group_key:
            group_support[group_key] += 1
    return last_support, suffix2_support, group_support


def row_features(
    row: pd.Series,
    user_lookup: pd.DataFrame,
    group_cols: Sequence[str],
    last_support: Counter,
    suffix2_support: Counter,
    group_support: Counter,
) -> Dict[str, Any]:
    hist = parse_sequence(row.get("item_seq_raw", ""))
    hist_len = len(hist)
    counts = Counter(hist)
    max_count = max(counts.values()) if counts else 0
    repeat_ratio = 1.0 - (len(counts) / hist_len) if hist_len else 0.0
    top_share = max_count / hist_len if hist_len else 0.0
    suffix2 = tuple(hist[-2:]) if len(hist) >= 2 else ()
    group_key = user_group_key(user_lookup, str(row.get("uid", "")), group_cols)
    return {
        "uid": str(row.get("uid", "")),
        "raw_len": hist_len,
        "coarse_len_bin": length_bin(hist_len),
        "exact_len_bin": exact_length_bin(hist_len),
        "repeat_ratio": repeat_ratio,
        "top_share": top_share,
        "last_item": hist[-1] if hist else "",
        "last_item_support": int(last_support.get(hist[-1], 0)) if hist else 0,
        "suffix2": "|".join(suffix2),
        "suffix2_support": int(suffix2_support.get(suffix2, 0)) if suffix2 else 0,
        "user_group": "|".join(group_key),
        "user_group_support": int(group_support.get(group_key, 0)) if group_key else 0,
    }


def summarize(rows: pd.DataFrame, group_col: str) -> Dict[str, Dict[str, float]]:
    payload: Dict[str, Dict[str, float]] = {}
    thresholds = (1, 5, 20, 40, 80)
    for name, group in rows.groupby(group_col, sort=False):
        item: Dict[str, float] = {
            "n": float(len(group)),
            "repeat_ratio_mean": float(group["repeat_ratio"].mean()) if len(group) else 0.0,
            "top_share_mean": float(group["top_share"].mean()) if len(group) else 0.0,
            "last_item_support_mean": float(group["last_item_support"].mean()) if len(group) else 0.0,
            "suffix2_support_mean": float(group["suffix2_support"].mean()) if len(group) else 0.0,
            "user_group_support_mean": float(group["user_group_support"].mean()) if len(group) else 0.0,
        }
        for threshold in thresholds:
            item[f"last_support_ge_{threshold}_rate"] = float((group["last_item_support"] >= threshold).mean())
            item[f"suffix2_support_ge_{threshold}_rate"] = float((group["suffix2_support"] >= threshold).mean())
            item[f"group_support_ge_{threshold}_rate"] = float((group["user_group_support"] >= threshold).mean())
        payload[str(name)] = item
    return payload


def main() -> None:
    args = parse_args()
    train_df, test_df, user_df = load_data(args.rec_data)
    user_lookup = user_df.set_index("uid") if "uid" in user_df.columns else pd.DataFrame()
    group_cols = tuple(str(col) for col in args.group_cols)
    last_support, suffix2_support, group_support = support_counters(train_df, user_lookup, group_cols)
    rows = pd.DataFrame(
        [
            row_features(row, user_lookup, group_cols, last_support, suffix2_support, group_support)
            for _, row in test_df.iterrows()
        ]
    )
    payload = {
        "rec_data": str(args.rec_data),
        "group_cols": list(group_cols),
        "num_train": int(len(train_df)),
        "num_test": int(len(test_df)),
        "overall": summarize(rows.assign(all="all"), "all").get("all", {}),
        "by_exact_len": summarize(rows, "exact_len_bin"),
        "by_coarse_len": summarize(rows, "coarse_len_bin"),
        "top_last_items": dict(last_support.most_common(20)),
        "top_suffix2": {"|".join(key): int(value) for key, value in suffix2_support.most_common(20)},
        "top_user_groups": {"|".join(key): int(value) for key, value in group_support.most_common(20)},
    }
    ensure_dir(args.output.parent)
    write_json(args.output, payload)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
