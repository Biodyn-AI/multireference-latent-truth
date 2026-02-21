#!/usr/bin/env python3
"""Evaluate calibration against external anchors derived from blind panel splits."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


Edge = Tuple[str, str]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--posterior",
        type=Path,
        default=Path(
            "subproject_24_multireference_latent_truth_model/implementation/outputs/full_v2_best_fullfit/ltm_family_hb_edge_posteriors.csv"
        ),
    )
    parser.add_argument(
        "--score-col",
        type=str,
        default="posterior",
    )
    parser.add_argument(
        "--score-mode",
        type=str,
        default="raw",
        choices=["raw", "add_context", "context_flag_boost", "min_context_floor"],
        help="How to build ranking score. `raw` uses --score-col from --posterior.",
    )
    parser.add_argument(
        "--governance-edges",
        type=Path,
        default=Path(
            "subproject_24_multireference_latent_truth_model/implementation/outputs/dual_universe_governance_v1/dual_universe_claim_tiers_edges.csv"
        ),
        help="Required for score modes that use context posterior/features.",
    )
    parser.add_argument(
        "--context-weight",
        type=float,
        default=1.0,
        help="Weight used in `add_context` score mode.",
    )
    parser.add_argument(
        "--context-flag-boost",
        type=float,
        default=0.1,
        help="Boost multiplier used in `context_flag_boost` score mode.",
    )
    parser.add_argument(
        "--context-floor-ratio",
        type=float,
        default=0.1,
        help="Floor ratio used in `min_context_floor` score mode.",
    )
    parser.add_argument(
        "--panel-split-summary",
        type=Path,
        default=Path(
            "subproject_24_multireference_latent_truth_model/implementation/outputs/top_tier_readiness_v2/blind_holdout_eval/panel_split_summary.csv"
        ),
    )
    parser.add_argument(
        "--k-values",
        type=str,
        default="500,5000,20000",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "subproject_24_multireference_latent_truth_model/implementation/outputs/top_tier_readiness_v2/external_anchor_calibration"
        ),
    )
    return parser.parse_args()


def _edge_set(path: Path) -> set[Edge]:
    df = pd.read_csv(path, sep="\t", low_memory=False)
    need = {"source", "target"}
    if not need.issubset(df.columns):
        raise ValueError(f"Panel file missing required columns {sorted(need)}: {path}")
    return set(df[["source", "target"]].drop_duplicates().itertuples(index=False, name=None))


def _build_reference_sets(panel_split: pd.DataFrame) -> Dict[str, set[Edge]]:
    refs: Dict[str, set[Edge]] = {
        "train_high_overlap_union": set(),
        "blind_holdout_union": set(),
        "blind_holdout_primary_union": set(),
        "blind_holdout_extra_union": set(),
    }
    for row in panel_split.itertuples(index=False):
        path = Path(row.panel_path)
        edges = _edge_set(path)
        if row.split == "train":
            refs["train_high_overlap_union"] |= edges
        elif row.split == "holdout":
            refs["blind_holdout_union"] |= edges
            if row.panel_group == "primary":
                refs["blind_holdout_primary_union"] |= edges
            elif row.panel_group == "extra":
                refs["blind_holdout_extra_union"] |= edges
    return refs


def _build_score(
    posterior: pd.DataFrame,
    args: argparse.Namespace,
) -> pd.Series:
    if args.score_mode == "raw":
        return posterior[args.score_col].astype(float)

    gov = pd.read_csv(args.governance_edges, low_memory=False)
    need = {"source", "target", "posterior_context", "in_context_universe"}
    if not need.issubset(gov.columns):
        raise ValueError(f"Governance file missing required columns {sorted(need)} for score mode {args.score_mode}")

    merged = posterior.merge(
        gov[["source", "target", "posterior_context", "in_context_universe"]],
        on=["source", "target"],
        how="left",
    )
    p_full = merged[args.score_col].astype(float).fillna(0.0)
    p_ctx = merged["posterior_context"].astype(float).fillna(0.0)
    in_ctx = merged["in_context_universe"].fillna(False).astype(bool)

    if args.score_mode == "add_context":
        return p_full + args.context_weight * p_ctx
    if args.score_mode == "context_flag_boost":
        return p_full * (1.0 + args.context_flag_boost * in_ctx.astype(float))
    if args.score_mode == "min_context_floor":
        return np.minimum(p_full, np.maximum(p_ctx, args.context_floor_ratio * p_full))
    raise ValueError(f"Unsupported score-mode: {args.score_mode}")


def _calibration_metrics(
    labels: np.ndarray,
    scores: np.ndarray,
    k_values: List[int],
) -> Dict[str, float]:
    n_pos = int(labels.sum())
    if n_pos == 0:
        out: Dict[str, float] = {
            "best_threshold": float("nan"),
            "best_precision": float("nan"),
            "best_recall": 0.0,
            "best_f1": float("nan"),
        }
        for k in k_values:
            out[f"precision_at_{k}"] = float("nan")
            out[f"tp_at_{k}"] = 0.0
        return out

    order = np.argsort(-scores)
    y = labels[order].astype(np.float64)
    s = scores[order]

    tp = np.cumsum(y)
    n = np.arange(1, len(y) + 1, dtype=np.float64)
    precision = tp / n
    recall = tp / float(n_pos)
    denom = precision + recall
    f1 = np.where(denom > 0, 2.0 * precision * recall / denom, 0.0)

    best_idx = int(np.nanargmax(f1))
    out = {
        "best_threshold": float(s[best_idx]),
        "best_precision": float(precision[best_idx]),
        "best_recall": float(recall[best_idx]),
        "best_f1": float(f1[best_idx]),
    }
    for k in k_values:
        kk = min(max(int(k), 1), len(y))
        tp_k = float(tp[kk - 1])
        out[f"precision_at_{k}"] = float(tp_k / kk)
        out[f"tp_at_{k}"] = float(tp_k)
    return out


def main() -> None:
    args = _parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    k_values = [int(x.strip()) for x in args.k_values.split(",") if x.strip()]
    if not k_values:
        raise ValueError("No k-values provided")

    posterior = pd.read_csv(args.posterior, low_memory=False)
    need = {"source", "target", args.score_col}
    if not need.issubset(posterior.columns):
        raise ValueError(f"Posterior file missing required columns {sorted(need)}")
    posterior = posterior[["source", "target", args.score_col]].copy()

    panel_split = pd.read_csv(args.panel_split_summary, low_memory=False)
    refs = _build_reference_sets(panel_split)
    full_universe_edges = set(posterior[["source", "target"]].itertuples(index=False, name=None))
    refs = {k: (v & full_universe_edges) for k, v in refs.items()}

    score_series = _build_score(posterior=posterior, args=args)
    scores = score_series.to_numpy(dtype=np.float64)
    edges = list(posterior[["source", "target"]].itertuples(index=False, name=None))

    rows: List[Dict[str, float | str]] = []
    for ref_name, ref_edges in refs.items():
        labels = np.fromiter((1.0 if e in ref_edges else 0.0 for e in edges), dtype=np.float64, count=len(edges))
        metrics = _calibration_metrics(labels=labels, scores=scores, k_values=k_values)
        row: Dict[str, float | str] = {
            "reference_set": ref_name,
            "n_edges": int(len(edges)),
            "n_reference_edges_in_universe": int(len(ref_edges)),
        }
        row.update(metrics)
        rows.append(row)

    summary = pd.DataFrame(rows).sort_values("reference_set").reset_index(drop=True)
    summary["score_mode"] = args.score_mode
    summary["score_col"] = args.score_col
    if args.score_mode == "add_context":
        summary["score_param"] = args.context_weight
    elif args.score_mode == "context_flag_boost":
        summary["score_param"] = args.context_flag_boost
    elif args.score_mode == "min_context_floor":
        summary["score_param"] = args.context_floor_ratio
    else:
        summary["score_param"] = np.nan
    summary.to_csv(args.output_dir / "external_anchor_calibration_summary.csv", index=False)

    # Compatibility file for readiness G03 interface (expects a full_universe row).
    holdout = summary[summary["reference_set"] == "blind_holdout_union"]
    if holdout.empty:
        raise ValueError("blind_holdout_union not found in calibration summary")
    h = holdout.iloc[0]
    compat = pd.DataFrame(
        [
            {
                "universe": "full_universe",
                "n_reference_edges_in_universe": int(h["n_reference_edges_in_universe"]),
                "best_precision": float(h["best_precision"]),
                "best_recall": float(h["best_recall"]),
                "best_f1": float(h["best_f1"]),
                "precision_at_500": float(h.get("precision_at_500", np.nan)),
                "tp_at_500": int(h.get("tp_at_500", np.nan)),
                "precision_at_5000": float(h.get("precision_at_5000", np.nan)),
                "tp_at_5000": int(h.get("tp_at_5000", np.nan)),
                "precision_at_20000": float(h.get("precision_at_20000", np.nan)),
                "tp_at_20000": int(h.get("tp_at_20000", np.nan)),
                "score_mode": args.score_mode,
                "score_col": args.score_col,
            }
        ]
    )
    compat.to_csv(args.output_dir / "calibration_full_vs_context_external_anchor.csv", index=False)

    print("External anchor calibration complete.")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
