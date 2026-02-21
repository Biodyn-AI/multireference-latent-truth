#!/usr/bin/env python3
"""Generate edge-level and method-level claim certification tiers.

This script operationalizes claim governance by combining:
- posterior confidence,
- model-variant agreement (EM vs dependence-aware),
- source-dropout sensitivity (method level),
- evidence coverage.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


_EPS = 1e-12


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dep-posterior",
        type=Path,
        default=Path("subproject_24_multireference_latent_truth_model/implementation/outputs/ltm_dep_edge_posteriors.csv"),
    )
    parser.add_argument(
        "--em-posterior",
        type=Path,
        default=Path("subproject_24_multireference_latent_truth_model/implementation/outputs/ltm_em_edge_posteriors.csv"),
    )
    parser.add_argument(
        "--dep-method-metrics",
        type=Path,
        default=Path("subproject_24_multireference_latent_truth_model/implementation/outputs/dep_posterior_method_metrics.csv"),
    )
    parser.add_argument(
        "--em-method-metrics",
        type=Path,
        default=Path("subproject_24_multireference_latent_truth_model/implementation/outputs/posterior_method_metrics.csv"),
    )
    parser.add_argument(
        "--variant-method-comp",
        type=Path,
        default=Path("subproject_24_multireference_latent_truth_model/implementation/outputs/variant_method_comparison.csv"),
    )
    parser.add_argument(
        "--source-dropout-summary",
        type=Path,
        default=Path(
            "subproject_24_multireference_latent_truth_model/implementation/outputs/source_dropout_ablation/source_dropout_summary.csv"
        ),
    )
    parser.add_argument(
        "--source-dropout-dir",
        type=Path,
        default=Path(
            "subproject_24_multireference_latent_truth_model/implementation/outputs/source_dropout_ablation"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("subproject_24_multireference_latent_truth_model/implementation/outputs"),
    )
    return parser.parse_args()


def _edge_tier(row: pd.Series) -> str:
    dep = float(row["posterior_dep"])
    em = float(row["posterior_em"])
    agreement = float(row["agreement"])
    n_obs = int(row["n_observed_sources"])

    if dep >= 0.90 and em >= 0.80 and agreement >= 0.80 and n_obs >= 3:
        return "TIER1_STRONG"
    if dep >= 0.70 and em >= 0.55 and agreement >= 0.60 and n_obs >= 2:
        return "TIER2_MODERATE"
    return "TIER3_EXPLORATORY"


def _method_tier(row: pd.Series) -> str:
    dep_f1 = float(row["weighted_f1_dep"])
    rank_dep = float(row["rank_dep"])
    delta = float(row["delta_weighted_f1_abs"])
    rank_std = float(row.get("ablation_rank_std", np.nan))

    if np.isnan(rank_std):
        rank_std = 999.0

    if rank_dep <= 3 and dep_f1 >= 0.35 and delta <= 0.04 and rank_std <= 1.5:
        return "TIERA_ROBUST"
    if rank_dep <= 6 and dep_f1 >= 0.20 and delta <= 0.08 and rank_std <= 3.0:
        return "TIERB_CONDITIONAL"
    return "TIERC_EXPLORATORY"


def _collect_ablation_method_ranks(source_dropout_dir: Path) -> pd.DataFrame:
    rows: List[Dict[str, float]] = []

    if not source_dropout_dir.exists():
        return pd.DataFrame(columns=["method", "ablation_rank_std", "ablation_rank_mean", "ablation_top1_freq"])

    run_dirs = [p for p in source_dropout_dir.iterdir() if p.is_dir() and p.name.startswith("drop_")]
    for run_dir in run_dirs:
        p = run_dir / "posterior_method_metrics.csv"
        if not p.exists():
            continue
        df = pd.read_csv(p)
        if df.empty:
            continue
        df = df.copy()
        df["rank"] = df["weighted_f1"].rank(ascending=False, method="min")
        top_method = str(df.sort_values("weighted_f1", ascending=False).iloc[0]["method"])
        for _, r in df.iterrows():
            rows.append(
                {
                    "run": run_dir.name,
                    "method": str(r["method"]),
                    "rank": float(r["rank"]),
                    "is_top1": float(str(r["method"]) == top_method),
                }
            )

    if not rows:
        return pd.DataFrame(columns=["method", "ablation_rank_std", "ablation_rank_mean", "ablation_top1_freq"])

    rank_df = pd.DataFrame(rows)
    out = (
        rank_df.groupby("method", as_index=False)
        .agg(
            ablation_rank_std=("rank", "std"),
            ablation_rank_mean=("rank", "mean"),
            ablation_top1_freq=("is_top1", "mean"),
        )
        .fillna(0.0)
    )
    return out


def main() -> None:
    args = _parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    dep_post = pd.read_csv(args.dep_posterior)
    em_post = pd.read_csv(args.em_posterior)

    dep_m = pd.read_csv(args.dep_method_metrics)
    em_m = pd.read_csv(args.em_method_metrics)
    var_m = pd.read_csv(args.variant_method_comp)

    # Edge-level certification.
    edge = dep_post[["source", "target", "posterior", "n_observed_sources"]].merge(
        em_post[["source", "target", "posterior"]],
        on=["source", "target"],
        how="inner",
        suffixes=("_dep", "_em"),
    )

    edge["agreement"] = 1.0 - np.abs(edge["posterior_dep"] - edge["posterior_em"])
    edge["delta_posterior_abs"] = np.abs(edge["posterior_dep"] - edge["posterior_em"])
    edge["claim_tier"] = edge.apply(_edge_tier, axis=1)

    edge_summary = (
        edge.groupby("claim_tier", as_index=False)
        .agg(
            n_edges=("source", "count"),
            mean_posterior_dep=("posterior_dep", "mean"),
            mean_agreement=("agreement", "mean"),
            mean_n_observed_sources=("n_observed_sources", "mean"),
        )
        .sort_values("claim_tier")
        .reset_index(drop=True)
    )

    # Method-level certification.
    method = dep_m[["method", "weighted_f1"]].merge(
        em_m[["method", "weighted_f1"]],
        on="method",
        how="inner",
        suffixes=("_dep", "_em"),
    )
    method = method.merge(var_m[["method", "rank_dep", "rank_em", "delta_weighted_f1"]], on="method", how="left")

    method["delta_weighted_f1_abs"] = np.abs(method["weighted_f1_dep"] - method["weighted_f1_em"])

    ablation = _collect_ablation_method_ranks(args.source_dropout_dir)
    method = method.merge(ablation, on="method", how="left")

    method["claim_tier"] = method.apply(_method_tier, axis=1)
    method = method.sort_values(["rank_dep", "weighted_f1_dep"], ascending=[True, False]).reset_index(drop=True)

    method_summary = (
        method.groupby("claim_tier", as_index=False)
        .agg(
            n_methods=("method", "count"),
            mean_weighted_f1_dep=("weighted_f1_dep", "mean"),
            mean_delta_model=("delta_weighted_f1_abs", "mean"),
            mean_ablation_rank_std=("ablation_rank_std", "mean"),
        )
        .sort_values("claim_tier")
        .reset_index(drop=True)
    )

    # Save artifacts.
    edge.to_csv(args.output_dir / "claim_tiers_edges.csv", index=False)
    edge_summary.to_csv(args.output_dir / "claim_tiers_edges_summary.csv", index=False)

    method.to_csv(args.output_dir / "claim_tiers_methods.csv", index=False)
    method_summary.to_csv(args.output_dir / "claim_tiers_methods_summary.csv", index=False)

    print("Claim-tier certification complete:")
    print(f"- edge rows: {len(edge)}")
    print(f"- method rows: {len(method)}")
    print(f"- outputs: {args.output_dir}")


if __name__ == "__main__":
    main()
