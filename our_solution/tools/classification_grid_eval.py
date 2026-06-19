#!/usr/bin/env python3
"""Multi-split classification grid probe without producing A1.csv."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from sklearn.metrics import accuracy_score

SOLUTION_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = SOLUTION_DIR.parent
if str(SOLUTION_DIR) not in sys.path:
    sys.path.insert(0, str(SOLUTION_DIR))

from src.classification import (  # noqa: E402
    build_features,
    candidate_configs,
    load_graph_npz,
    make_model,
    predict_label_propagation,
    predict_label_propagation_with_model_fallback,
)
from src.common import ensure_dir, stratified_split, write_json  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate classification configs over multiple splits.")
    parser.add_argument("--cls_data", type=Path, default=PROJECT_ROOT / "A分类" / "A1.npz")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    parser.add_argument("--val_ratio", type=float, default=0.12)
    parser.add_argument(
        "--output",
        type=Path,
        default=SOLUTION_DIR / "output" / "classification_grid_eval_5seed.json",
    )
    return parser.parse_args()


def ridge_fallback(alpha: float = 0.4, hops: int = 1, class_weight: Any = None) -> Dict[str, Any]:
    return {
        "name": f"ridge_attr_{hops}hop_alpha{alpha}_cw{class_weight}",
        "model": "ridge",
        "alpha": alpha,
        "feature_hops": hops,
        "use_label_prop": False,
        "use_degree": True,
        "symmetrize": True,
        "class_weight": class_weight,
    }


def lp_fallback_config(
    name: str,
    steps: int,
    alpha: float,
    beta: float = 0.0,
    threshold: float | None = None,
    fallback_alpha: float = 0.4,
    fallback_hops: int = 1,
) -> Dict[str, Any]:
    cfg: Dict[str, Any] = {
        "name": name,
        "model": "label_prop_fallback_model",
        "lp_steps": steps,
        "lp_alpha": alpha,
        "symmetrize": True,
        "add_self_loop": True,
        "fallback": "class_prior",
        "fallback_model_config": ridge_fallback(alpha=fallback_alpha, hops=fallback_hops, class_weight=None),
    }
    if abs(beta) > 1e-12:
        cfg["class_prior_beta"] = beta
    if threshold is not None:
        cfg["fallback_confidence_threshold"] = threshold
    return cfg


def probe_configs() -> List[Dict[str, Any]]:
    configs: List[Dict[str, Any]] = []
    known = {cfg["name"]: cfg for cfg in candidate_configs()}
    for name in (
        "v7_label_prop_zero_ridge_attr_1hop_alpha04_nobal",
        "v11_label_prop_prior_beta_m025_zero_ridge",
        "label_prop_zero_ridge_attr_1hop_nobal",
        "label_prop_direct_5_095_prior",
        "label_prop_direct_3_099_prior",
        "label_prop_direct_30_099_prior",
        "ridge_attr_smooth_lp",
    ):
        if name in known:
            configs.append(known[name])

    for beta in (0.0, -0.05, -0.10, -0.15, -0.20, -0.25, -0.30):
        configs.append(
            lp_fallback_config(
                name=f"lp5_a095_beta{beta:+.2f}_zero_ridge04",
                steps=5,
                alpha=0.95,
                beta=beta,
                threshold=None,
                fallback_alpha=0.4,
            )
        )

    for threshold in (0.20, 0.25, 0.30, 0.35, 0.40, 0.45):
        configs.append(
            lp_fallback_config(
                name=f"lp5_a095_thr{threshold:.2f}_ridge04",
                steps=5,
                alpha=0.95,
                threshold=threshold,
                fallback_alpha=0.4,
            )
        )

    for steps, alpha in ((4, 0.95), (6, 0.95), (8, 0.95), (5, 0.90), (5, 0.98), (8, 0.98)):
        configs.append(
            lp_fallback_config(
                name=f"lp{steps}_a{alpha:.2f}_zero_ridge04",
                steps=steps,
                alpha=alpha,
                threshold=None,
                fallback_alpha=0.4,
            )
        )

    for fallback_alpha in (0.2, 0.6, 0.8, 1.2):
        configs.append(
            lp_fallback_config(
                name=f"lp5_a095_zero_ridge{fallback_alpha}",
                steps=5,
                alpha=0.95,
                fallback_alpha=fallback_alpha,
            )
        )
    return configs


def evaluate_config(data: Any, fit_idx: np.ndarray, val_idx: np.ndarray, config: Dict[str, Any], seed: int):
    if config.get("model") == "label_prop":
        pred, stats = predict_label_propagation(data, fit_idx, val_idx, config)
    elif config.get("model") == "label_prop_fallback_model":
        pred, stats = predict_label_propagation_with_model_fallback(data, fit_idx, val_idx, config, seed)
    else:
        x = build_features(data, source_idx=fit_idx, config=config)
        model = make_model(config, seed)
        model.fit(x[fit_idx], data.labels[fit_idx])
        pred = model.predict(x[val_idx])
        stats = {"feature_shape": list(x.shape)}
    return float(accuracy_score(data.labels[val_idx], pred)), stats


def main() -> None:
    args = parse_args()
    data = load_graph_npz(args.cls_data)
    configs = probe_configs()
    rows: List[Dict[str, Any]] = []
    for seed in args.seeds:
        fit_idx, val_idx = stratified_split(data.labels, data.train_idx, args.val_ratio, seed)
        for round_id, config in enumerate(configs, start=1):
            start = time.time()
            acc, stats = evaluate_config(data, fit_idx, val_idx, config, seed + round_id)
            rows.append(
                {
                    "seed": int(seed),
                    "name": str(config["name"]),
                    "val_acc": acc,
                    "duration": round(time.time() - start, 4),
                    "stats": stats,
                }
            )
            print(f"seed={seed} {config['name']} acc={acc:.6f}", flush=True)

    summary: List[Dict[str, Any]] = []
    for config in configs:
        values = [r["val_acc"] for r in rows if r["name"] == config["name"]]
        summary.append(
            {
                "name": str(config["name"]),
                "mean_acc": float(np.mean(values)),
                "min_acc": float(np.min(values)),
                "max_acc": float(np.max(values)),
                "std_acc": float(np.std(values)),
                "values": values,
                "config": config,
            }
        )
    summary.sort(key=lambda x: (x["mean_acc"], x["min_acc"]), reverse=True)
    payload = {
        "cls_data": str(args.cls_data),
        "seeds": list(args.seeds),
        "val_ratio": float(args.val_ratio),
        "summary": summary,
        "rows": rows,
    }
    ensure_dir(args.output.parent)
    write_json(args.output, payload)
    print("top configs:", flush=True)
    for item in summary[:10]:
        print(
            f"{item['name']} mean={item['mean_acc']:.6f} min={item['min_acc']:.6f} std={item['std_acc']:.6f}",
            flush=True,
        )
    print(f"wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
