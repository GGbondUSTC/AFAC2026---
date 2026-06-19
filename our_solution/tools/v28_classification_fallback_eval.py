#!/usr/bin/env python3
"""Probe stronger fallback models on top of the v21 classification baseline."""

from __future__ import annotations

import argparse
import sys
import time
import warnings
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier

warnings.filterwarnings("ignore", message="X does not have valid feature names.*", category=UserWarning)

SOLUTION_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = SOLUTION_DIR.parent
if str(SOLUTION_DIR) not in sys.path:
    sys.path.insert(0, str(SOLUTION_DIR))

from src.classification import (  # noqa: E402
    build_features,
    candidate_configs,
    label_propagation_scores,
    load_graph_npz,
    predict_label_propagation_pair_gate,
)
from src.common import ensure_dir, stratified_split, write_json  # noqa: E402


SAFE_PAIRS = {
    (4, 9),
    (5, 9),
    (4, 0),
    (1, 7),
    (4, 7),
    (4, 5),
    (1, 2),
    (4, 6),
    (1, 5),
    (8, 3),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate stronger low-confidence classification fallbacks.")
    parser.add_argument("--cls_data", type=Path, default=PROJECT_ROOT / "A分类" / "A1.npz")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    parser.add_argument("--val_ratio", type=float, default=0.12)
    parser.add_argument("--models", nargs="+", default=["lgbm", "extra_trees", "mlp"])
    parser.add_argument(
        "--output",
        type=Path,
        default=SOLUTION_DIR / "output" / "v28_classification_fallback_eval.json",
    )
    return parser.parse_args()


def fallback_feature_config() -> Dict[str, Any]:
    return {
        "name": "attr_1hop_degree_for_stronger_fallback",
        "model": "feature_builder",
        "feature_hops": 1,
        "use_label_prop": False,
        "use_degree": True,
        "symmetrize": True,
        "class_weight": None,
    }


def make_model(model_name: str, seed: int, num_classes: int):
    if model_name == "ridge":
        return RidgeClassifier(alpha=0.4, class_weight=None), False
    if model_name == "extra_trees":
        return (
            ExtraTreesClassifier(
                n_estimators=260,
                max_features="sqrt",
                min_samples_leaf=2,
                class_weight=None,
                random_state=seed,
                n_jobs=-1,
            ),
            False,
        )
    if model_name == "mlp":
        return (
            MLPClassifier(
                hidden_layer_sizes=(160,),
                activation="relu",
                alpha=3e-4,
                batch_size=512,
                learning_rate_init=8e-4,
                max_iter=90,
                early_stopping=True,
                n_iter_no_change=8,
                random_state=seed,
            ),
            True,
        )
    if model_name == "lgbm":
        try:
            from lightgbm import LGBMClassifier
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("LightGBM is not installed in this Python environment.") from exc
        return (
            LGBMClassifier(
                objective="multiclass",
                num_class=int(num_classes),
                n_estimators=160,
                learning_rate=0.045,
                num_leaves=31,
                min_child_samples=24,
                subsample=0.9,
                colsample_bytree=0.9,
                reg_lambda=1.0,
                random_state=seed,
                verbosity=-1,
                force_col_wise=True,
            ),
            False,
        )
    raise ValueError(f"unsupported model: {model_name}")


def masks_from_lp(data: Any, fit_idx: np.ndarray, val_idx: np.ndarray) -> Dict[str, np.ndarray]:
    base_config = candidate_configs()[0]["base_config"]
    scores = label_propagation_scores(
        data=data,
        source_idx=fit_idx,
        steps=int(base_config.get("lp_steps", 5)),
        alpha=float(base_config.get("lp_alpha", 0.95)),
        symmetrize=bool(base_config.get("symmetrize", True)),
        add_self_loop=bool(base_config.get("add_self_loop", True)),
    )
    val_scores = scores[val_idx]
    score_sum = val_scores.sum(axis=1)
    confidence = val_scores.max(axis=1) / (score_sum + 1e-12)
    masks: Dict[str, np.ndarray] = {"zero": score_sum <= 1e-12}
    for threshold in (0.18, 0.22, 0.26, 0.30, 0.35):
        masks[f"conf_le_{threshold:.2f}"] = confidence <= threshold
    return masks


def pair_counts(base_pred: Sequence[int], alt_pred: Sequence[int], mask: np.ndarray) -> Dict[str, int]:
    counts: Counter = Counter()
    for base, alt, keep in zip(base_pred, alt_pred, mask):
        if keep:
            counts[f"{int(base)}->{int(alt)}"] += 1
    return dict(counts)


def evaluate_seed(data: Any, seed: int, val_ratio: float, model_names: Sequence[str]) -> List[Dict[str, Any]]:
    fit_idx, val_idx = stratified_split(data.labels, data.train_idx, val_ratio, seed)
    baseline_config = candidate_configs()[0]
    baseline_pred, baseline_stats = predict_label_propagation_pair_gate(
        data, fit_idx, val_idx, baseline_config, seed
    )
    baseline_acc = float(accuracy_score(data.labels[val_idx], baseline_pred))
    masks = masks_from_lp(data, fit_idx, val_idx)
    x = build_features(data, source_idx=fit_idx, config=fallback_feature_config())
    rows: List[Dict[str, Any]] = [
        {
            "seed": int(seed),
            "candidate": "v21_baseline",
            "model": "baseline",
            "mask": "none",
            "gate": "none",
            "val_acc": baseline_acc,
            "delta_vs_v21": 0.0,
            "changed_rows": 0,
            "pair_counts": {},
            "duration": 0.0,
            "baseline_stats": baseline_stats,
        }
    ]
    for model_idx, model_name in enumerate(model_names):
        start = time.time()
        try:
            model, needs_dense = make_model(model_name, seed + model_idx * 17, data.num_classes)
            train_x = x[fit_idx].toarray() if needs_dense else x[fit_idx]
            val_x = x[val_idx].toarray() if needs_dense else x[val_idx]
            model.fit(train_x, data.labels[fit_idx])
            alt_pred = model.predict(val_x).astype(int)
            fit_duration = time.time() - start
        except Exception as exc:
            rows.append(
                {
                    "seed": int(seed),
                    "candidate": f"v28_cls_{model_name}_failed",
                    "model": model_name,
                    "mask": "none",
                    "gate": "none",
                    "val_acc": 0.0,
                    "delta_vs_v21": -999.0,
                    "changed_rows": 0,
                    "pair_counts": {},
                    "duration": round(time.time() - start, 4),
                    "error": str(exc),
                }
            )
            continue
        for mask_name, base_mask in masks.items():
            for gate in ("none", "safe_pairs"):
                replace_mask = base_mask.copy()
                if gate == "safe_pairs":
                    replace_mask &= np.asarray(
                        [(int(base), int(alt)) in SAFE_PAIRS for base, alt in zip(baseline_pred, alt_pred)],
                        dtype=bool,
                    )
                pred = baseline_pred.copy()
                pred[replace_mask] = alt_pred[replace_mask]
                acc = float(accuracy_score(data.labels[val_idx], pred))
                changed = int(np.sum(pred != baseline_pred))
                rows.append(
                    {
                        "seed": int(seed),
                        "candidate": f"v28_cls_{model_name}_{mask_name}_{gate}",
                        "model": model_name,
                        "mask": mask_name,
                        "gate": gate,
                        "val_acc": acc,
                        "delta_vs_v21": acc - baseline_acc,
                        "changed_rows": changed,
                        "mask_rows": int(base_mask.sum()),
                        "pair_counts": pair_counts(baseline_pred, alt_pred, replace_mask),
                        "duration": round(fit_duration, 4),
                    }
                )
    return rows


def summarize(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: DefaultDict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["candidate"])].append(dict(row))
    summary: List[Dict[str, Any]] = []
    for name, values in grouped.items():
        accs = [float(x["val_acc"]) for x in values if float(x["delta_vs_v21"]) > -100]
        deltas = [float(x["delta_vs_v21"]) for x in values if float(x["delta_vs_v21"]) > -100]
        changed = [float(x.get("changed_rows", 0)) for x in values if float(x["delta_vs_v21"]) > -100]
        summary.append(
            {
                "candidate": name,
                "mean_acc": float(np.mean(accs)) if accs else 0.0,
                "min_acc": float(np.min(accs)) if accs else 0.0,
                "mean_delta_vs_v21": float(np.mean(deltas)) if deltas else -999.0,
                "min_delta_vs_v21": float(np.min(deltas)) if deltas else -999.0,
                "max_delta_vs_v21": float(np.max(deltas)) if deltas else -999.0,
                "mean_changed_rows": float(np.mean(changed)) if changed else 0.0,
                "values": values,
            }
        )
    summary.sort(key=lambda row: (row["mean_delta_vs_v21"], row["min_delta_vs_v21"]), reverse=True)
    return summary


def main() -> None:
    args = parse_args()
    data = load_graph_npz(args.cls_data)
    rows: List[Dict[str, Any]] = []
    for seed in args.seeds:
        seed_rows = evaluate_seed(data, seed, args.val_ratio, args.models)
        rows.extend(seed_rows)
        partial = summarize(rows)
        top = partial[0]
        print(
            f"seed={seed} top={top['candidate']} "
            f"mean_delta={top['mean_delta_vs_v21']:.6f} "
            f"min_delta={top['min_delta_vs_v21']:.6f}",
            flush=True,
        )
    summary = summarize(rows)
    payload = {
        "cls_data": str(args.cls_data),
        "seeds": list(args.seeds),
        "val_ratio": float(args.val_ratio),
        "models": list(args.models),
        "summary": summary,
        "rows": rows,
    }
    ensure_dir(args.output.parent)
    write_json(args.output, payload)
    for row in summary[:12]:
        print(
            f"{row['candidate']} mean_delta={row['mean_delta_vs_v21']:.6f} "
            f"min_delta={row['min_delta_vs_v21']:.6f} "
            f"changed={row['mean_changed_rows']:.1f}",
            flush=True,
        )
    print(f"wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
