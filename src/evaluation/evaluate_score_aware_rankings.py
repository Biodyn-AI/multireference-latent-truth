#!/usr/bin/env python3
"""Evaluate method rankings using continuous scores against posterior truth.

Unlike binary edge-membership metrics, this script uses method score magnitudes
(from score_matrix_wide.csv) and computes soft-label ranking metrics:
- weighted average precision (soft AP),
- weighted ROC-AUC (soft AUC),
- precision at top-K under posterior weighting.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


_EPS = 1e-12


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--score-matrix",
        type=Path,
        default=Path("subproject_24_multireference_latent_truth_model/implementation/outputs/score_matrix_wide.csv"),
    )
    parser.add_argument(
        "--source-meta",
        type=Path,
        default=Path("subproject_24_multireference_latent_truth_model/implementation/outputs/source_metadata.csv"),
    )
    parser.add_argument(
        "--posterior",
        type=Path,
        default=Path("subproject_24_multireference_latent_truth_model/implementation/outputs/ltm_dep_edge_posteriors.csv"),
    )
    parser.add_argument(
        "--posterior-col",
        type=str,
        default="posterior",
    )
    parser.add_argument(
        "--binary-metrics",
        type=Path,
        default=Path("subproject_24_multireference_latent_truth_model/implementation/outputs/dep_posterior_method_metrics.csv"),
        help="Binary-membership metrics for side-by-side comparison.",
    )
    parser.add_argument(
        "--topk-list",
        type=str,
        default="100,1000,5000",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("subproject_24_multireference_latent_truth_model/implementation/outputs"),
    )
    return parser.parse_args()


def _parse_topk_list(s: str) -> List[int]:
    out = []
    for tok in s.split(","):
        tok = tok.strip()
        if not tok:
            continue
        out.append(int(tok))
    return sorted(set(out))


def _weighted_average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
    if labels.size == 0:
        return float("nan")
    rel = labels.astype(float)
    if np.sum(rel) <= _EPS:
        return float("nan")

    order = np.argsort(-scores)
    rel_sorted = rel[order]

    rank = np.arange(1, rel_sorted.size + 1, dtype=float)
    cum_rel = np.cumsum(rel_sorted)
    precision_at_k = cum_rel / rank

    ap = np.sum(precision_at_k * rel_sorted) / np.sum(rel_sorted)
    return float(ap)


def _weighted_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Compute soft-label weighted AUC using grouped ranks for ties."""

    if labels.size == 0:
        return float("nan")

    df = pd.DataFrame({"score": scores.astype(float), "p": labels.astype(float)})
    df["n"] = 1.0 - df["p"]

    total_p = float(df["p"].sum())
    total_n = float(df["n"].sum())
    if total_p <= _EPS or total_n <= _EPS:
        return float("nan")

    grouped = df.groupby("score", as_index=False).agg({"p": "sum", "n": "sum"})
    grouped = grouped.sort_values("score", ascending=True).reset_index(drop=True)

    cum_n_before = np.cumsum(grouped["n"].to_numpy()) - grouped["n"].to_numpy()

    contrib = grouped["p"].to_numpy() * (cum_n_before + 0.5 * grouped["n"].to_numpy())
    auc = float(np.sum(contrib) / (total_p * total_n))
    return auc


def _precision_at_topk(labels: np.ndarray, scores: np.ndarray, k: int) -> float:
    if labels.size == 0:
        return float("nan")
    k = min(k, labels.size)
    if k <= 0:
        return float("nan")
    idx = np.argpartition(scores, -k)[-k:]
    return float(np.mean(labels[idx]))


def main() -> None:
    args = _parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    topk_list = _parse_topk_list(args.topk_list)

    score_matrix = pd.read_csv(args.score_matrix)
    source_meta = pd.read_csv(args.source_meta)
    posterior = pd.read_csv(args.posterior)

    if args.posterior_col not in posterior.columns:
        raise ValueError(
            f"Posterior column '{args.posterior_col}' not found in {args.posterior}; "
            f"available={posterior.columns.tolist()}"
        )

    merged = score_matrix.merge(
        posterior[["source", "target", args.posterior_col]].rename(columns={args.posterior_col: "posterior"}),
        on=["source", "target"],
        how="inner",
    )

    methods = source_meta[source_meta["kind"] == "method"]["source_id"].tolist()
    rows: List[Dict[str, float]] = []

    for method in methods:
        if method not in merged.columns:
            continue

        sub = merged[[method, "posterior"]].dropna()
        if sub.empty:
            continue

        scores = sub[method].to_numpy(dtype=float)
        labels = sub["posterior"].to_numpy(dtype=float)

        row: Dict[str, float] = {
            "method": method,
            "n_eval_edges": float(len(sub)),
            "score_min": float(np.min(scores)),
            "score_max": float(np.max(scores)),
            "score_mean": float(np.mean(scores)),
            "score_std": float(np.std(scores)),
            "soft_ap": _weighted_average_precision(labels, scores),
            "soft_auc": _weighted_auc(labels, scores),
        }

        for k in topk_list:
            row[f"precision_at_{k}"] = _precision_at_topk(labels, scores, k)

        rows.append(row)

    metrics = pd.DataFrame(rows).sort_values("soft_ap", ascending=False).reset_index(drop=True)

    # Side-by-side comparison to binary-membership weighted F1 ranking.
    if args.binary_metrics.exists():
        binary = pd.read_csv(args.binary_metrics)
        comp = metrics.merge(
            binary[["method", "weighted_f1"]],
            on="method",
            how="left",
        )
        comp["rank_soft_ap"] = comp["soft_ap"].rank(ascending=False, method="min")
        comp["rank_weighted_f1"] = comp["weighted_f1"].rank(ascending=False, method="min")
        comp["delta_rank_soft_minus_binary"] = comp["rank_soft_ap"] - comp["rank_weighted_f1"]
    else:
        comp = metrics.copy()

    metrics.to_csv(args.output_dir / "score_aware_method_metrics.csv", index=False)
    comp.to_csv(args.output_dir / "score_aware_vs_binary_comparison.csv", index=False)

    print("Score-aware posterior ranking evaluation complete:")
    print(f"- methods with usable continuous scores: {len(metrics)}")
    print(f"- outputs: {args.output_dir}")


if __name__ == "__main__":
    main()
