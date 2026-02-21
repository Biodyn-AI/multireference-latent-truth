#!/usr/bin/env python3
"""Compare baseline EM latent truth vs dependence-aware latent truth outputs."""

from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("subproject_24_multireference_latent_truth_model/implementation/outputs"),
    )
    return parser.parse_args()


def _pairwise_orders(metric_df: pd.DataFrame, metric_col: str) -> Dict[Tuple[str, str], int]:
    if metric_df.empty:
        return {}
    scores = metric_df.set_index("method")[metric_col].to_dict()
    methods = sorted(scores.keys())
    out: Dict[Tuple[str, str], int] = {}
    for a, b in combinations(methods, 2):
        va, vb = scores[a], scores[b]
        if np.isclose(va, vb):
            out[(a, b)] = 0
        elif va > vb:
            out[(a, b)] = 1
        else:
            out[(a, b)] = -1
    return out


def _disagreement_rate(a: Dict[Tuple[str, str], int], b: Dict[Tuple[str, str], int]) -> float:
    pairs = sorted(set(a.keys()) & set(b.keys()))
    valid = 0
    diff = 0
    for p in pairs:
        if a[p] == 0 or b[p] == 0:
            continue
        valid += 1
        if a[p] != b[p]:
            diff += 1
    if valid == 0:
        return float("nan")
    return diff / float(valid)


def main() -> None:
    args = _parse_args()

    em_post = pd.read_csv(args.output_dir / "ltm_em_edge_posteriors.csv")
    dep_post = pd.read_csv(args.output_dir / "ltm_dep_edge_posteriors.csv")

    em_m = pd.read_csv(args.output_dir / "posterior_method_metrics.csv")
    dep_m = pd.read_csv(args.output_dir / "dep_posterior_method_metrics.csv")

    em_rev = pd.read_csv(args.output_dir / "ranking_reversal_vs_posterior.csv")
    dep_rev = pd.read_csv(args.output_dir / "dep_ranking_reversal_vs_posterior.csv")

    # Edge-level posterior comparison.
    merged_post = em_post[["source", "target", "posterior"]].merge(
        dep_post[["source", "target", "posterior"]],
        on=["source", "target"],
        how="inner",
        suffixes=("_em", "_dep"),
    )

    pearson = float(np.corrcoef(merged_post["posterior_em"], merged_post["posterior_dep"])[0, 1])
    spearman = float(
        pd.Series(merged_post["posterior_em"]).corr(pd.Series(merged_post["posterior_dep"]), method="spearman")
    )

    top_k = max(1, int(0.01 * len(merged_post)))
    em_top = set(
        zip(
            merged_post.nlargest(top_k, "posterior_em")["source"],
            merged_post.nlargest(top_k, "posterior_em")["target"],
        )
    )
    dep_top = set(
        zip(
            merged_post.nlargest(top_k, "posterior_dep")["source"],
            merged_post.nlargest(top_k, "posterior_dep")["target"],
        )
    )
    top_jaccard = len(em_top & dep_top) / float(max(len(em_top | dep_top), 1))

    # Method ranking comparison.
    comp = em_m[["method", "weighted_f1"]].merge(
        dep_m[["method", "weighted_f1"]],
        on="method",
        suffixes=("_em", "_dep"),
    )
    comp["delta_weighted_f1"] = comp["weighted_f1_dep"] - comp["weighted_f1_em"]
    comp["rank_em"] = comp["weighted_f1_em"].rank(ascending=False, method="min")
    comp["rank_dep"] = comp["weighted_f1_dep"].rank(ascending=False, method="min")
    comp["delta_rank"] = comp["rank_dep"] - comp["rank_em"]
    comp = comp.sort_values("weighted_f1_dep", ascending=False).reset_index(drop=True)

    em_orders = _pairwise_orders(em_m[["method", "weighted_f1"]], "weighted_f1")
    dep_orders = _pairwise_orders(dep_m[["method", "weighted_f1"]], "weighted_f1")
    method_rank_disagreement = _disagreement_rate(em_orders, dep_orders)

    # Reference-level disagreement delta.
    rev_comp = em_rev.merge(dep_rev, on="reference", suffixes=("_em", "_dep"))
    rev_comp["delta_disagreement"] = (
        rev_comp["pairwise_disagreement_vs_posterior_dep"]
        - rev_comp["pairwise_disagreement_vs_posterior_em"]
    )

    summary = pd.DataFrame(
        [
            {
                "n_edges": len(merged_post),
                "posterior_pearson": pearson,
                "posterior_spearman": spearman,
                "top1pct_jaccard": top_jaccard,
                "method_pairwise_rank_disagreement_em_vs_dep": method_rank_disagreement,
                "mean_reference_disagreement_em": float(
                    np.nanmean(rev_comp["pairwise_disagreement_vs_posterior_em"])
                ),
                "mean_reference_disagreement_dep": float(
                    np.nanmean(rev_comp["pairwise_disagreement_vs_posterior_dep"])
                ),
                "mean_reference_disagreement_delta_dep_minus_em": float(
                    np.nanmean(rev_comp["delta_disagreement"])
                ),
            }
        ]
    )

    comp.to_csv(args.output_dir / "variant_method_comparison.csv", index=False)
    rev_comp.to_csv(args.output_dir / "variant_reference_disagreement_comparison.csv", index=False)
    summary.to_csv(args.output_dir / "variant_comparison_summary.csv", index=False)

    print("Variant comparison complete:")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
