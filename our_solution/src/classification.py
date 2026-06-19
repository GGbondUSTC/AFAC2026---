"""Lightweight node classification pipeline.

The design intentionally avoids dense full-graph tensors. It combines sparse
node attributes, graph-smoothed attributes, degree statistics, and label
propagation priors, then searches a small set of linear classifiers.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse import csr_matrix, hstack
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import normalize

from .common import Timer, Trajectory, ensure_dir, stratified_split, write_json
from .qwen_agent import ask_qwen_json, compact_json


@dataclass
class GraphData:
    adj: csr_matrix
    features: csr_matrix
    labels: np.ndarray
    train_idx: np.ndarray
    test_idx: np.ndarray
    num_classes: int


def load_graph_npz(path: str | Path) -> GraphData:
    raw = np.load(path, allow_pickle=True)
    adj = csr_matrix(
        (raw["adj_data"], raw["adj_indices"], raw["adj_indptr"]),
        shape=tuple(raw["adj_shape"]),
    ).astype(np.float32)
    features = csr_matrix(
        (raw["attr_data"], raw["attr_indices"], raw["attr_indptr"]),
        shape=tuple(raw["attr_shape"]),
    ).astype(np.float32)
    labels = raw["labels"].astype(np.int64)
    train_idx = raw["train_idx"].astype(np.int64)
    test_idx = raw["test_idx"].astype(np.int64)
    num_classes = int(labels[labels >= 0].max()) + 1
    return GraphData(adj, features, labels, train_idx, test_idx, num_classes)


def normalized_adj(adj: csr_matrix, symmetrize: bool = True, add_self_loop: bool = True) -> csr_matrix:
    a = adj.tocsr(copy=True).astype(np.float32)
    if symmetrize:
        a = a.maximum(a.T)
    if add_self_loop:
        a = a + sp.eye(a.shape[0], dtype=np.float32, format="csr")
    a.data[:] = 1.0
    return normalize(a, norm="l1", axis=1, copy=False)


def label_propagation_features(
    data: GraphData,
    source_idx: np.ndarray,
    steps: int,
    alpha: float,
    symmetrize: bool,
) -> csr_matrix:
    y0 = np.zeros((data.adj.shape[0], data.num_classes), dtype=np.float32)
    y0[source_idx, data.labels[source_idx]] = 1.0
    if steps <= 0:
        return csr_matrix(y0)
    a = normalized_adj(data.adj, symmetrize=symmetrize, add_self_loop=True)
    y = y0.copy()
    for _ in range(steps):
        y = alpha * (a @ y) + (1.0 - alpha) * y0
        # Clamp labeled source nodes so validation labels never leak into their own features.
        y[source_idx] = y0[source_idx]
    return csr_matrix(y)


def label_propagation_scores(
    data: GraphData,
    source_idx: np.ndarray,
    steps: int,
    alpha: float,
    symmetrize: bool,
    add_self_loop: bool = True,
) -> np.ndarray:
    y0 = np.zeros((data.adj.shape[0], data.num_classes), dtype=np.float32)
    y0[source_idx, data.labels[source_idx]] = 1.0
    a = normalized_adj(data.adj, symmetrize=symmetrize, add_self_loop=add_self_loop)
    y = y0.copy()
    for _ in range(max(0, steps)):
        y = alpha * (a @ y) + (1.0 - alpha) * y0
        y[source_idx] = y0[source_idx]
    return y


def predict_label_propagation(
    data: GraphData,
    source_idx: np.ndarray,
    predict_idx: np.ndarray,
    config: Dict[str, Any],
) -> Tuple[np.ndarray, Dict[str, Any]]:
    scores = label_propagation_scores(
        data=data,
        source_idx=source_idx,
        steps=int(config.get("lp_steps", 5)),
        alpha=float(config.get("lp_alpha", 0.95)),
        symmetrize=bool(config.get("symmetrize", True)),
        add_self_loop=bool(config.get("add_self_loop", True)),
    )
    class_prior_beta = float(config.get("class_prior_beta", 0.0))
    if abs(class_prior_beta) > 1e-12:
        counts = np.bincount(data.labels[source_idx], minlength=data.num_classes).astype(np.float32)
        prior = counts / max(float(counts.sum()), 1.0)
        class_weights = np.power(prior + 1e-9, class_prior_beta).astype(np.float32)
        scores = scores * class_weights.reshape(1, -1)
    pred = scores[predict_idx].argmax(axis=1).astype(int)
    score_sum = scores[predict_idx].sum(axis=1)
    zero_mask = score_sum <= 1e-12
    fallback_label = int(np.bincount(data.labels[source_idx], minlength=data.num_classes).argmax())
    if config.get("fallback", "class_prior") == "class_prior" and zero_mask.any():
        pred[zero_mask] = fallback_label
    confidence = scores[predict_idx].max(axis=1) / (score_sum + 1e-12)
    stats = {
        "feature_shape": [int(data.adj.shape[0]), int(data.num_classes)],
        "zero_score_nodes": int(zero_mask.sum()),
        "fallback_label": fallback_label,
        "class_prior_beta": class_prior_beta,
        "mean_confidence": float(np.mean(confidence)),
    }
    return pred, stats


def predict_label_propagation_with_model_fallback(
    data: GraphData,
    source_idx: np.ndarray,
    predict_idx: np.ndarray,
    config: Dict[str, Any],
    seed: int,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    scores = label_propagation_scores(
        data=data,
        source_idx=source_idx,
        steps=int(config.get("lp_steps", 5)),
        alpha=float(config.get("lp_alpha", 0.95)),
        symmetrize=bool(config.get("symmetrize", True)),
        add_self_loop=bool(config.get("add_self_loop", True)),
    )
    class_prior_beta = float(config.get("class_prior_beta", 0.0))
    if abs(class_prior_beta) > 1e-12:
        counts = np.bincount(data.labels[source_idx], minlength=data.num_classes).astype(np.float32)
        prior = counts / max(float(counts.sum()), 1.0)
        class_weights = np.power(prior + 1e-9, class_prior_beta).astype(np.float32)
        scores = scores * class_weights.reshape(1, -1)
    pred = scores[predict_idx].argmax(axis=1).astype(int)
    score_sum = scores[predict_idx].sum(axis=1)
    zero_mask = score_sum <= 1e-12
    fallback_label = int(np.bincount(data.labels[source_idx], minlength=data.num_classes).argmax())
    pred[zero_mask] = fallback_label

    confidence = scores[predict_idx].max(axis=1) / (score_sum + 1e-12)
    replace_mask = zero_mask.copy()
    threshold = config.get("fallback_confidence_threshold")
    if threshold is not None:
        replace_mask |= confidence <= float(threshold)

    fallback_model_config = dict(config.get("fallback_model_config", {}))
    fallback_feature_shape: List[int] = []
    if fallback_model_config and replace_mask.any():
        x = build_features(data, source_idx=source_idx, config=fallback_model_config)
        fallback_feature_shape = list(x.shape)
        model = make_model(fallback_model_config, seed)
        model.fit(x[source_idx], data.labels[source_idx])
        pred[replace_mask] = model.predict(x[predict_idx[replace_mask]]).astype(int)

    stats = {
        "feature_shape": [int(data.adj.shape[0]), int(data.num_classes)],
        "zero_score_nodes": int(zero_mask.sum()),
        "fallback_label": fallback_label,
        "fallback_model_nodes": int(replace_mask.sum()),
        "fallback_model_config": fallback_model_config.get("name", ""),
        "fallback_feature_shape": fallback_feature_shape,
        "class_prior_beta": class_prior_beta,
        "mean_confidence": float(np.mean(confidence)),
    }
    return pred, stats


def predict_label_propagation_pair_gate(
    data: GraphData,
    source_idx: np.ndarray,
    predict_idx: np.ndarray,
    config: Dict[str, Any],
    seed: int,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    base_config = dict(config["base_config"])
    alt_config = dict(config["alt_config"])
    base_pred, base_stats = predict_label_propagation_with_model_fallback(
        data, source_idx, predict_idx, base_config, seed
    )
    alt_pred, alt_stats = predict_label_propagation_with_model_fallback(
        data, source_idx, predict_idx, alt_config, seed
    )
    allowed_pairs = {tuple(int(x) for x in pair) for pair in config.get("allowed_pairs", ())}
    replace_mask = np.array(
        [(int(base), int(alt)) in allowed_pairs for base, alt in zip(base_pred, alt_pred)],
        dtype=bool,
    )
    pred = base_pred.copy()
    pred[replace_mask] = alt_pred[replace_mask]

    pair_counts: Dict[str, int] = {}
    for base, alt, replaced in zip(base_pred, alt_pred, replace_mask):
        if replaced:
            key = f"{int(base)}->{int(alt)}"
            pair_counts[key] = pair_counts.get(key, 0) + 1

    stats = {
        "base_config": base_config.get("name", ""),
        "alt_config": alt_config.get("name", ""),
        "allowed_pairs": [f"{int(a)}->{int(b)}" for a, b in sorted(allowed_pairs)],
        "changed_rows": int(replace_mask.sum()),
        "pair_counts": pair_counts,
        "base_stats": base_stats,
        "alt_stats": alt_stats,
    }
    return pred, stats


def degree_features(adj: csr_matrix) -> csr_matrix:
    out_degree = np.asarray(adj.sum(axis=1)).ravel()
    in_degree = np.asarray(adj.sum(axis=0)).ravel()
    total = out_degree + in_degree
    dense = np.vstack([out_degree, in_degree, total, np.abs(out_degree - in_degree)]).T
    dense = np.log1p(dense).astype(np.float32)
    scale = dense.max(axis=0)
    scale[scale == 0] = 1.0
    dense = dense / scale
    return csr_matrix(dense)


def build_features(data: GraphData, source_idx: np.ndarray, config: Dict[str, Any]) -> csr_matrix:
    x0 = normalize(data.features, norm="l1", axis=1, copy=True).astype(np.float32)
    parts: List[csr_matrix] = [x0]

    feature_hops = int(config.get("feature_hops", 1))
    if feature_hops > 0:
        a = normalized_adj(
            data.adj,
            symmetrize=bool(config.get("symmetrize", True)),
            add_self_loop=True,
        )
        xh = x0
        for _ in range(feature_hops):
            xh = (a @ xh).tocsr().astype(np.float32)
            parts.append(xh)

    if config.get("use_label_prop", True):
        parts.append(
            label_propagation_features(
                data=data,
                source_idx=source_idx,
                steps=int(config.get("lp_steps", 2)),
                alpha=float(config.get("lp_alpha", 0.85)),
                symmetrize=bool(config.get("symmetrize", True)),
            )
        )

    if config.get("use_degree", True):
        parts.append(degree_features(data.adj))

    return hstack(parts, format="csr", dtype=np.float32)


def make_model(config: Dict[str, Any], seed: int):
    model_type = config.get("model", "ridge")
    if model_type == "ridge":
        return RidgeClassifier(
            alpha=float(config.get("alpha", 1.0)),
            class_weight=config.get("class_weight", "balanced"),
        )
    if model_type == "sgd":
        return SGDClassifier(
            loss=config.get("loss", "modified_huber"),
            alpha=float(config.get("alpha", 1e-4)),
            penalty=config.get("penalty", "l2"),
            max_iter=int(config.get("max_iter", 1200)),
            tol=float(config.get("tol", 1e-4)),
            class_weight=config.get("class_weight", "balanced"),
            random_state=seed,
            n_jobs=-1,
        )
    if model_type == "logreg":
        return LogisticRegression(
            C=float(config.get("C", 1.0)),
            solver="saga",
            max_iter=int(config.get("max_iter", 450)),
            class_weight=config.get("class_weight", "balanced"),
            random_state=seed,
        )
    raise ValueError(f"Unsupported classification model: {model_type}")


def candidate_configs() -> List[Dict[str, Any]]:
    return [
        {
            "name": "v21_pair_gated_beta_m010_safe_pairs",
            "model": "label_prop_pair_gate",
            "base_config": {
                "name": "v7_label_prop_zero_ridge_attr_1hop_alpha04_nobal",
                "model": "label_prop_fallback_model",
                "lp_steps": 5,
                "lp_alpha": 0.95,
                "symmetrize": True,
                "add_self_loop": True,
                "fallback": "class_prior",
                "fallback_model_config": {
                    "name": "ridge_attr_1hop_nobal",
                    "model": "ridge",
                    "alpha": 0.4,
                    "feature_hops": 1,
                    "use_label_prop": False,
                    "use_degree": True,
                    "symmetrize": True,
                    "class_weight": None,
                },
            },
            "alt_config": {
                "name": "lp5_a095_beta_m010_zero_ridge04",
                "model": "label_prop_fallback_model",
                "lp_steps": 5,
                "lp_alpha": 0.95,
                "symmetrize": True,
                "add_self_loop": True,
                "class_prior_beta": -0.10,
                "fallback": "class_prior",
                "fallback_model_config": {
                    "name": "ridge_attr_1hop_nobal",
                    "model": "ridge",
                    "alpha": 0.4,
                    "feature_hops": 1,
                    "use_label_prop": False,
                    "use_degree": True,
                    "symmetrize": True,
                    "class_weight": None,
                },
            },
            "allowed_pairs": (
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
            ),
        },
        {
            "name": "v7_label_prop_zero_ridge_attr_1hop_alpha04_nobal",
            "model": "label_prop_fallback_model",
            "lp_steps": 5,
            "lp_alpha": 0.95,
            "symmetrize": True,
            "add_self_loop": True,
            "fallback": "class_prior",
            "fallback_model_config": {
                "name": "ridge_attr_1hop_nobal",
                "model": "ridge",
                "alpha": 0.4,
                "feature_hops": 1,
                "use_label_prop": False,
                "use_degree": True,
                "symmetrize": True,
                "class_weight": None,
            },
        },
        {
            "name": "v11_label_prop_prior_beta_m025_zero_ridge",
            "model": "label_prop_fallback_model",
            "lp_steps": 5,
            "lp_alpha": 0.95,
            "symmetrize": True,
            "add_self_loop": True,
            "class_prior_beta": -0.25,
            "fallback": "class_prior",
            "fallback_model_config": {
                "name": "ridge_attr_1hop_nobal",
                "model": "ridge",
                "alpha": 0.4,
                "feature_hops": 1,
                "use_label_prop": False,
                "use_degree": True,
                "symmetrize": True,
                "class_weight": None,
            },
        },
        {
            "name": "label_prop_zero_ridge_attr_1hop_nobal",
            "model": "label_prop_fallback_model",
            "lp_steps": 5,
            "lp_alpha": 0.95,
            "symmetrize": True,
            "add_self_loop": True,
            "fallback": "class_prior",
            "fallback_model_config": {
                "name": "ridge_attr_1hop_nobal",
                "model": "ridge",
                "alpha": 1.0,
                "feature_hops": 1,
                "use_label_prop": False,
                "use_degree": True,
                "symmetrize": True,
                "class_weight": None,
            },
        },
        {
            "name": "label_prop_direct_5_095_prior",
            "model": "label_prop",
            "lp_steps": 5,
            "lp_alpha": 0.95,
            "symmetrize": True,
            "add_self_loop": True,
            "fallback": "class_prior",
        },
        {
            "name": "label_prop_direct_3_099_prior",
            "model": "label_prop",
            "lp_steps": 3,
            "lp_alpha": 0.99,
            "symmetrize": True,
            "add_self_loop": True,
            "fallback": "class_prior",
        },
        {
            "name": "label_prop_direct_30_099_prior",
            "model": "label_prop",
            "lp_steps": 30,
            "lp_alpha": 0.99,
            "symmetrize": True,
            "add_self_loop": True,
            "fallback": "class_prior",
        },
        {
            "name": "label_prop_direct_20_085_prior",
            "model": "label_prop",
            "lp_steps": 20,
            "lp_alpha": 0.85,
            "symmetrize": True,
            "add_self_loop": True,
            "fallback": "class_prior",
        },
        {
            "name": "ridge_attr_smooth_lp",
            "model": "ridge",
            "alpha": 1.0,
            "feature_hops": 1,
            "use_label_prop": True,
            "lp_steps": 2,
            "lp_alpha": 0.85,
            "symmetrize": True,
            "class_weight": "balanced",
        },
        {
            "name": "ridge_attr_2hop_lp",
            "model": "ridge",
            "alpha": 2.0,
            "feature_hops": 2,
            "use_label_prop": True,
            "lp_steps": 3,
            "lp_alpha": 0.8,
            "symmetrize": True,
            "class_weight": "balanced",
        },
        {
            "name": "sgd_huber_attr_smooth_lp",
            "model": "sgd",
            "loss": "modified_huber",
            "alpha": 2e-5,
            "feature_hops": 1,
            "use_label_prop": True,
            "lp_steps": 2,
            "lp_alpha": 0.85,
            "symmetrize": True,
            "class_weight": "balanced",
        },
        {
            "name": "ridge_attr_only",
            "model": "ridge",
            "alpha": 1.5,
            "feature_hops": 0,
            "use_label_prop": False,
            "use_degree": True,
            "class_weight": "balanced",
        },
        {
            "name": "logreg_sparse_graph",
            "model": "logreg",
            "C": 1.0,
            "feature_hops": 1,
            "use_label_prop": True,
            "lp_steps": 2,
            "lp_alpha": 0.85,
            "symmetrize": True,
            "class_weight": "balanced",
            "max_iter": 350,
        },
    ]


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


def _bounded_bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "y"}:
            return True
        if normalized in {"false", "0", "no", "n"}:
            return False
    return default


def _safe_config_name(prefix: str, raw_name: Any, index: int) -> str:
    raw_text = str(raw_name or "").strip().lower()
    cleaned = "".join(ch if ch.isalnum() else "_" for ch in raw_text)
    cleaned = "_".join(part for part in cleaned.split("_") if part)
    if not cleaned:
        cleaned = f"candidate_{index}"
    if not cleaned.startswith(prefix):
        cleaned = f"{prefix}_{index:02d}_{cleaned}"
    return cleaned[:96]


def _classification_data_summary(data: GraphData) -> Dict[str, Any]:
    labeled = data.labels[data.train_idx]
    counts = np.bincount(labeled, minlength=data.num_classes)
    return {
        "nodes": int(data.adj.shape[0]),
        "edges": int(data.adj.nnz),
        "features": int(data.features.shape[1]),
        "feature_nonzeros": int(data.features.nnz),
        "train_nodes": int(len(data.train_idx)),
        "test_nodes": int(len(data.test_idx)),
        "num_classes": int(data.num_classes),
        "class_counts": {str(i): int(count) for i, count in enumerate(counts.tolist())},
    }


def _sanitize_qwen_classification_config(raw: Any, index: int) -> Tuple[Optional[Dict[str, Any]], str]:
    if not isinstance(raw, dict):
        return None, "suggestion is not an object"
    model_name = str(raw.get("model", "label_prop_fallback_model"))
    if model_name != "label_prop_fallback_model":
        return None, f"unsupported model: {model_name}"

    class_weight = raw.get("fallback_class_weight", raw.get("class_weight", None))
    if str(class_weight).lower() in {"balanced", "balance"}:
        class_weight_value: Optional[str] = "balanced"
    else:
        class_weight_value = None

    config = {
        "name": _safe_config_name("qwen_cls", raw.get("name"), index),
        "model": "label_prop_fallback_model",
        "lp_steps": _bounded_int(raw.get("lp_steps"), 5, 1, 50),
        "lp_alpha": _bounded_float(raw.get("lp_alpha"), 0.95, 0.50, 0.995),
        "symmetrize": _bounded_bool(raw.get("symmetrize"), True),
        "add_self_loop": _bounded_bool(raw.get("add_self_loop"), True),
        "class_prior_beta": _bounded_float(raw.get("class_prior_beta"), 0.0, -0.60, 0.30),
        "fallback": "class_prior",
        "fallback_model_config": {
            "name": "qwen_ridge_fallback",
            "model": "ridge",
            "alpha": _bounded_float(raw.get("fallback_alpha"), 0.4, 0.1, 3.0),
            "feature_hops": _bounded_int(raw.get("fallback_feature_hops"), 1, 0, 2),
            "use_label_prop": False,
            "use_degree": _bounded_bool(raw.get("fallback_use_degree"), True),
            "symmetrize": _bounded_bool(raw.get("fallback_symmetrize"), True),
            "class_weight": class_weight_value,
        },
    }
    return config, ""


def qwen_classification_configs(
    data: GraphData,
    trial_summaries: List[Dict[str, Any]],
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
        "You tune an anonymized sparse graph node classifier. "
        "Return only valid JSON. Propose configs; do not invent code."
    )
    user = {
        "task": "Suggest classification configs for local validation.",
        "allowed_schema": {
            "suggestions": [
                {
                    "name": "qwen_cls_short_name",
                    "model": "label_prop_fallback_model",
                    "lp_steps": "integer 1..50",
                    "lp_alpha": "float 0.50..0.995",
                    "class_prior_beta": "float -0.60..0.30",
                    "symmetrize": "boolean",
                    "add_self_loop": "boolean",
                    "fallback_alpha": "float 0.1..3.0",
                    "fallback_feature_hops": "integer 0..2",
                    "fallback_use_degree": "boolean",
                    "fallback_class_weight": "null or balanced",
                    "reason": "brief rationale",
                }
            ]
        },
        "constraints": [
            "Use only model=label_prop_fallback_model.",
            "Prefer materially different configs over tiny perturbations.",
            "The local validator will reject configs that do not beat the current best.",
            f"Return at most {requested} suggestions.",
        ],
        "data_summary": _classification_data_summary(data),
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
        config, error = _sanitize_qwen_classification_config(raw, index)
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


def run_classification(
    data_path: str | Path,
    output_dir: str | Path,
    budget: int = 5,
    seed: int = 42,
    val_ratio: float = 0.12,
    sample_path: Optional[str | Path] = None,
    time_limit: Optional[float] = None,
    use_qwen_agent: bool = False,
    qwen_env_path: str | Path = ".env",
    qwen_model: Optional[str] = None,
    qwen_rounds: int = 3,
) -> Dict[str, Any]:
    timer = Timer()
    output_dir = ensure_dir(output_dir)
    submission_dir = ensure_dir(output_dir / "submission")
    data = load_graph_npz(data_path)
    fit_idx, val_idx = stratified_split(data.labels, data.train_idx, val_ratio, seed)

    trajectory = Trajectory(
        task_id="B1",
        objective="Maximize internal validation accuracy for product classification.",
    )
    configs = candidate_configs()[: max(1, budget)]
    best: Dict[str, Any] = {"val_acc": -1.0, "config": None, "model": None}
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
    round_id = 0

    def evaluate_and_record(config: Dict[str, Any], origin: str) -> None:
        nonlocal best, round_id
        round_id += 1
        if time_limit is not None and timer.elapsed > time_limit:
            return
        round_start = time.time()
        model = None
        if config.get("model") == "label_prop":
            pred, model_feedback = predict_label_propagation(data, fit_idx, val_idx, config)
        elif config.get("model") == "label_prop_fallback_model":
            pred, model_feedback = predict_label_propagation_with_model_fallback(
                data, fit_idx, val_idx, config, seed + round_id
            )
        elif config.get("model") == "label_prop_pair_gate":
            pred, model_feedback = predict_label_propagation_pair_gate(
                data, fit_idx, val_idx, config, seed + round_id
            )
        else:
            x = build_features(data, source_idx=fit_idx, config=config)
            model = make_model(config, seed + round_id)
            model.fit(x[fit_idx], data.labels[fit_idx])
            pred = model.predict(x[val_idx])
            model_feedback = {"feature_shape": list(x.shape)}
        val_acc = float(accuracy_score(data.labels[val_idx], pred))
        feedback = {
            "val_acc": val_acc,
            "fit_size": int(len(fit_idx)),
            "val_size": int(len(val_idx)),
            "origin": origin,
            **model_feedback,
        }
        trial_summaries.append(
            {
                "round": round_id,
                "origin": origin,
                "name": config.get("name", ""),
                "model": config.get("model", ""),
                "val_acc": val_acc,
                "feedback": model_feedback,
                "config": config,
            }
        )
        if val_acc > best["val_acc"]:
            best = {"val_acc": val_acc, "config": dict(config), "model": model}
            strategy = f"KEEP_AS_BEST; {origin} candidate will be retrained if no later round improves."
            trajectory.best_round = round_id
            trajectory.selected_config = dict(config)
        else:
            strategy = f"REJECT; {origin} candidate did not improve validation accuracy."
        trajectory.add(round_id, dict(config), feedback, strategy, time.time() - round_start)

    for config in configs:
        if time_limit is not None and timer.elapsed > time_limit:
            break
        evaluate_and_record(config, origin="deterministic")

    if use_qwen_agent and qwen_rounds > 0:
        if time_limit is not None and timer.elapsed > time_limit:
            qwen_agent_report["status"] = "time_limit_exhausted"
        else:
            qwen_configs, qwen_agent_report = qwen_classification_configs(
                data=data,
                trial_summaries=trial_summaries,
                requested=qwen_rounds,
                env_path=qwen_env_path,
                model=qwen_model,
            )
            for config in qwen_configs:
                if time_limit is not None and timer.elapsed > time_limit:
                    break
                evaluate_and_record(config, origin="qwen_agent")

    if best["config"] is None:
        raise RuntimeError("No classification configuration was evaluated.")

    final_config = best["config"]
    final_stats: Dict[str, Any] = {}
    if final_config.get("model") == "label_prop":
        test_pred, final_stats = predict_label_propagation(data, data.train_idx, data.test_idx, final_config)
        test_pred = test_pred.astype(int)
    elif final_config.get("model") == "label_prop_fallback_model":
        test_pred, final_stats = predict_label_propagation_with_model_fallback(
            data, data.train_idx, data.test_idx, final_config, seed + 10_000
        )
        test_pred = test_pred.astype(int)
    elif final_config.get("model") == "label_prop_pair_gate":
        test_pred, final_stats = predict_label_propagation_pair_gate(
            data, data.train_idx, data.test_idx, final_config, seed + 10_000
        )
        test_pred = test_pred.astype(int)
    else:
        final_x = build_features(data, source_idx=data.train_idx, config=final_config)
        final_model = make_model(final_config, seed + 10_000)
        final_model.fit(final_x[data.train_idx], data.labels[data.train_idx])
        test_pred = final_model.predict(final_x[data.test_idx]).astype(int)
        final_stats = {"final_feature_shape": list(final_x.shape)}

    pred_map = dict(zip(data.test_idx.tolist(), test_pred.tolist()))
    if sample_path and Path(sample_path).exists():
        sample = pd.read_csv(sample_path)
        out = pd.DataFrame(
            {
                "test_idx": sample["test_idx"].astype(int),
                "label": [pred_map[int(i)] for i in sample["test_idx"]],
            }
        )
    else:
        out = pd.DataFrame({"test_idx": data.test_idx, "label": test_pred})

    a1_path = submission_dir / "A1.csv"
    out.to_csv(a1_path, index=False)

    result = {
        "task": "classification",
        "best_val_acc": best["val_acc"],
        "best_config": final_config,
        "num_rounds": len(trajectory.records),
        "prediction_path": str(a1_path),
        "duration": round(timer.elapsed, 4),
        "class_distribution": out["label"].value_counts().sort_index().to_dict(),
        "final_stats": final_stats,
        "qwen_agent": qwen_agent_report,
    }
    trajectory.selected_config = final_config
    write_json(output_dir / "trajectory_B1.json", trajectory.to_dict())
    write_json(output_dir / "classification_result.json", result)
    return result
