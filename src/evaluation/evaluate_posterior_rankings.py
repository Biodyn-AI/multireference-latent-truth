#!/usr/bin/env python3
"""Evaluate method rankings against latent posterior truth and single references."""

from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from ltm_utils import binary_precision_recall_f1, weighted_precision_recall_f1


_EPS = 1e-9


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--matrix",
        type=Path,
        default=Path("subproject_24_multireference_latent_truth_model/implementation/outputs/evidence_matrix_wide.csv"),
    )
    parser.add_argument(
        "--source-meta",
        type=Path,
        default=Path("subproject_24_multireference_latent_truth_model/implementation/outputs/source_metadata.csv"),
    )
    parser.add_argument(
        "--posterior",
        type=Path,
        default=Path("subproject_24_multireference_latent_truth_model/implementation/outputs/ltm_em_edge_posteriors.csv"),
    )
    parser.add_argument(
        "--posterior-col",
        type=str,
        default="posterior",
        help="Posterior probability column name in --posterior file.",
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default="",
        help="Optional prefix for output filenames, e.g. dep_.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("subproject_24_multireference_latent_truth_model/implementation/outputs"),
    )
    return parser.parse_args()


def _pairwise_orders(metric_df: pd.DataFrame, metric_col: str) -> Dict[Tuple[str, str], int]:
    """Encode pairwise method order for one metric table.

    Return values:
    - +1 if method_a > method_b
    - -1 if method_a < method_b
    -  0 if tie
    """

    if metric_df.empty:
        return {}

    scores = metric_df.set_index("method")[metric_col].to_dict()
    methods = sorted(scores.keys())

    orders: Dict[Tuple[str, str], int] = {}
    for a, b in combinations(methods, 2):
        va, vb = scores[a], scores[b]
        if np.isclose(va, vb):
            orders[(a, b)] = 0
        elif va > vb:
            orders[(a, b)] = 1
        else:
            orders[(a, b)] = -1
    return orders


def _disagreement_rate(base: Dict[Tuple[str, str], int], other: Dict[Tuple[str, str], int]) -> float:
    pairs = sorted(set(base.keys()) & set(other.keys()))
    valid = 0
    disagree = 0
    for p in pairs:
        if base[p] == 0 or other[p] == 0:
            continue
        valid += 1
        if base[p] != other[p]:
            disagree += 1
    if valid == 0:
        return float("nan")
    return disagree / float(valid)


def main() -> None:
    args = _parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    matrix = pd.read_csv(args.matrix)
    source_meta = pd.read_csv(args.source_meta)
    posterior = pd.read_csv(args.posterior)

    if args.posterior_col not in posterior.columns:
        raise ValueError(
            f"Posterior column '{args.posterior_col}' not found in {args.posterior}. "
            f"Available columns: {posterior.columns.tolist()}"
        )

    df = matrix.merge(
        posterior[["source", "target", args.posterior_col]].rename(columns={args.posterior_col: "posterior"}),
        on=["source", "target"],
        how="inner",
    )

    methods = source_meta[source_meta["kind"] == "method"]["source_id"].tolist()
    references = source_meta[source_meta["kind"] == "reference"]["source_id"].tolist()

    posterior_rows = []
    for method in methods:
        sub = df[[method, "posterior"]].dropna()
        if sub.empty:
            continue
        y_pred = sub[method].to_numpy(dtype=float)
        p_true = sub["posterior"].to_numpy(dtype=float)

        stats = weighted_precision_recall_f1(y_pred=y_pred, y_prob_true=p_true)
        posterior_rows.append(
            {
                "method": method,
                "n_eval_edges": int(len(sub)),
                "predicted_positive_edges": float(np.sum(y_pred)),
                **stats,
            }
        )

    posterior_metrics = pd.DataFrame(posterior_rows)
    if not posterior_metrics.empty:
        posterior_metrics = posterior_metrics.sort_values("weighted_f1", ascending=False).reset_index(
            drop=True
        )

    ref_rows = []
    for reference in references:
        for method in methods:
            sub = df[[reference, method]].dropna()
            if sub.empty:
                continue
            y_true = sub[reference].to_numpy(dtype=int)
            y_pred = sub[method].to_numpy(dtype=int)
            stats = binary_precision_recall_f1(y_true=y_true, y_pred=y_pred)
            ref_rows.append(
                {
                    "reference": reference,
                    "method": method,
                    "n_eval_edges": int(len(sub)),
                    **stats,
                }
            )

    reference_metrics = pd.DataFrame(ref_rows)

    # Pairwise ranking disagreements relative to posterior-weighted ranking.
    base_orders = _pairwise_orders(posterior_metrics[["method", "weighted_f1"]], "weighted_f1")

    reversal_rows = []
    for reference, ref_df in reference_metrics.groupby("reference"):
        ref_rank = ref_df[["method", "f1"]].copy()
        ref_orders = _pairwise_orders(ref_rank, "f1")
        rev = _disagreement_rate(base_orders, ref_orders)
        reversal_rows.append(
            {
                "reference": reference,
                "pairwise_disagreement_vs_posterior": rev,
                "n_methods": int(ref_rank["method"].nunique()),
            }
        )

    reversal_df = pd.DataFrame(reversal_rows)
    if not reversal_df.empty:
        reversal_df = reversal_df.sort_values(
            "pairwise_disagreement_vs_posterior", ascending=False
        ).reset_index(drop=True)

    # Also summarize cross-reference instability on raw-reference rankings.
    cross_rows = []
    ref_order_map = {}
    for reference, ref_df in reference_metrics.groupby("reference"):
        ref_order_map[reference] = _pairwise_orders(ref_df[["method", "f1"]], "f1")

    ref_names = sorted(ref_order_map.keys())
    for a, b in combinations(ref_names, 2):
        dis = _disagreement_rate(ref_order_map[a], ref_order_map[b])
        cross_rows.append(
            {
                "reference_a": a,
                "reference_b": b,
                "pairwise_disagreement": dis,
            }
        )
    cross_df = pd.DataFrame(cross_rows)
    if not cross_df.empty:
        cross_df = cross_df.sort_values("pairwise_disagreement", ascending=False).reset_index(
            drop=True
        )

    prefix = args.output_prefix
    posterior_metrics.to_csv(args.output_dir / f"{prefix}posterior_method_metrics.csv", index=False)
    reference_metrics.to_csv(args.output_dir / f"{prefix}reference_method_metrics.csv", index=False)
    reversal_df.to_csv(args.output_dir / f"{prefix}ranking_reversal_vs_posterior.csv", index=False)
    cross_df.to_csv(args.output_dir / f"{prefix}reference_pairwise_instability.csv", index=False)

    print("Posterior ranking evaluation complete:")
    print(f"- methods evaluated: {len(posterior_metrics)}")
    print(f"- references evaluated: {reference_metrics['reference'].nunique() if not reference_metrics.empty else 0}")
    print(f"- outputs: {args.output_dir}")


if __name__ == "__main__":
    main()
