#!/usr/bin/env python3
"""Pareto-based multi-objective promotion policy optimizer."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--policy-summary",
        type=Path,
        default=Path(
            "subproject_24_multireference_latent_truth_model/implementation/outputs/top_tier_readiness_v2/policy_sweep_up_to_conditional_budget842_tinyalpha/deconcentration_policy_summary.csv"
        ),
    )
    parser.add_argument(
        "--blind-summary",
        type=Path,
        default=Path(
            "subproject_24_multireference_latent_truth_model/implementation/outputs/top_tier_readiness_v2/blind_holdout_eval_tinyalpha/blind_holdout_validation_summary.csv"
        ),
    )
    parser.add_argument("--baseline-policy", type=str, default="baseline_alpha0")
    parser.add_argument("--train-reference-set", type=str, default="train_high_overlap_union")
    parser.add_argument("--holdout-reference-set", type=str, default="blind_holdout_union")
    parser.add_argument("--min-joint-delta", type=float, default=-0.003)
    parser.add_argument("--min-train-hit-delta", type=int, default=-1)
    parser.add_argument("--max-top1", type=float, default=0.25)
    parser.add_argument("--max-top2", type=float, default=0.50)
    parser.add_argument("--w-holdout-enrichment", type=float, default=0.35)
    parser.add_argument("--w-train-enrichment", type=float, default=0.25)
    parser.add_argument("--w-top1-reduction", type=float, default=0.20)
    parser.add_argument("--w-joint-retention", type=float, default=0.20)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "subproject_24_multireference_latent_truth_model/implementation/outputs/top_tier_readiness_v2/multiobjective_policy_optimization"
        ),
    )
    return parser.parse_args()


def _norm(series: pd.Series) -> pd.Series:
    lo = float(series.min())
    hi = float(series.max())
    if hi <= lo:
        return pd.Series(np.zeros(len(series)), index=series.index)
    return (series - lo) / (hi - lo)


def _is_pareto_optimal(matrix: np.ndarray) -> np.ndarray:
    """Return boolean mask for points that are not dominated (maximize all columns)."""
    n = matrix.shape[0]
    keep = np.ones(n, dtype=bool)
    for i in range(n):
        if not keep[i]:
            continue
        # dominated by any j: all >= and at least one >
        ge = np.all(matrix >= matrix[i], axis=1)
        gt = np.any(matrix > matrix[i], axis=1)
        dominated = ge & gt
        dominated[i] = False
        if dominated.any():
            keep[i] = False
    return keep


def main() -> None:
    args = _parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    pol = pd.read_csv(args.policy_summary, low_memory=False)
    blind = pd.read_csv(args.blind_summary, low_memory=False)
    blind["policy"] = blind["selection"].astype(str)

    train = blind[blind["reference_set"] == args.train_reference_set].copy()
    hold = blind[blind["reference_set"] == args.holdout_reference_set].copy()
    if train.empty:
        raise ValueError(f"train reference set not found: {args.train_reference_set}")
    if hold.empty:
        raise ValueError(f"holdout reference set not found: {args.holdout_reference_set}")

    train_small = train[["policy", "n_hits", "enrichment_vs_random"]].rename(
        columns={"n_hits": "train_hits", "enrichment_vs_random": "train_enrichment"}
    )
    hold_small = hold[["policy", "n_hits", "enrichment_vs_random"]].rename(
        columns={"n_hits": "holdout_hits", "enrichment_vs_random": "holdout_enrichment"}
    )
    merged = pol.merge(train_small, on="policy", how="left").merge(hold_small, on="policy", how="left")
    if merged.empty:
        raise ValueError("No overlapping policy labels between policy and blind summaries")
    if args.baseline_policy not in set(merged["policy"]):
        raise ValueError(f"Baseline policy missing: {args.baseline_policy}")

    base = merged[merged["policy"] == args.baseline_policy].iloc[0]
    merged["delta_train_hits_vs_baseline"] = merged["train_hits"] - float(base["train_hits"])
    merged["delta_holdout_hits_vs_baseline"] = merged["holdout_hits"] - float(base["holdout_hits"])
    merged["delta_holdout_enrichment_vs_baseline"] = merged["holdout_enrichment"] - float(base["holdout_enrichment"])

    merged["constraint_joint"] = merged["delta_vs_baseline_joint_score_mean"] >= args.min_joint_delta
    merged["constraint_train_hits"] = merged["delta_train_hits_vs_baseline"] >= args.min_train_hit_delta
    merged["constraint_top1"] = merged["top1_source_fraction"] <= args.max_top1
    merged["constraint_top2"] = merged["top2_source_fraction"] <= args.max_top2
    merged["is_feasible"] = merged[["constraint_joint", "constraint_train_hits", "constraint_top1", "constraint_top2"]].all(axis=1)

    # Pareto objectives (maximize):
    # 1) holdout_enrichment
    # 2) train_enrichment
    # 3) top1 reduction vs baseline
    # 4) joint-score retention (negative absolute drop)
    merged["obj_holdout_enrichment"] = merged["holdout_enrichment"].astype(float)
    merged["obj_train_enrichment"] = merged["train_enrichment"].astype(float)
    merged["obj_top1_reduction"] = -merged["delta_vs_baseline_top1_source_fraction"].astype(float)
    merged["obj_joint_retention"] = merged["delta_vs_baseline_joint_score_mean"].astype(float)

    mat = merged[
        ["obj_holdout_enrichment", "obj_train_enrichment", "obj_top1_reduction", "obj_joint_retention"]
    ].to_numpy(dtype=float)
    merged["is_pareto"] = _is_pareto_optimal(mat)

    # Weighted utility for recommendation (on normalized objectives).
    merged["u_holdout"] = _norm(merged["obj_holdout_enrichment"])
    merged["u_train"] = _norm(merged["obj_train_enrichment"])
    merged["u_top1"] = _norm(merged["obj_top1_reduction"])
    merged["u_joint"] = _norm(merged["obj_joint_retention"])
    merged["utility_score"] = (
        args.w_holdout_enrichment * merged["u_holdout"]
        + args.w_train_enrichment * merged["u_train"]
        + args.w_top1_reduction * merged["u_top1"]
        + args.w_joint_retention * merged["u_joint"]
    )

    # Prefer feasible Pareto set first.
    candidate = merged[(merged["is_feasible"]) & (merged["is_pareto"])].copy()
    if candidate.empty:
        candidate = merged[merged["is_pareto"]].copy()
    if candidate.empty:
        candidate = merged.copy()

    rec = candidate.sort_values(
        ["utility_score", "is_feasible", "obj_holdout_enrichment", "obj_joint_retention"],
        ascending=[False, False, False, False],
    ).head(1)
    rec = rec.copy()
    rec["recommendation_reason"] = "best weighted utility on Pareto frontier with feasibility preference"

    merged.to_csv(args.output_dir / "multiobjective_policy_table.csv", index=False)
    merged[merged["is_pareto"]].to_csv(args.output_dir / "multiobjective_pareto_frontier.csv", index=False)
    rec.to_csv(args.output_dir / "multiobjective_recommendation.csv", index=False)

    summary = pd.DataFrame(
        [
            {
                "recommended_policy": str(rec.iloc[0]["policy"]),
                "recommended_alpha": float(rec.iloc[0]["alpha"]),
                "is_feasible": bool(rec.iloc[0]["is_feasible"]),
                "is_pareto": bool(rec.iloc[0]["is_pareto"]),
                "delta_top1_source_fraction": float(rec.iloc[0]["delta_vs_baseline_top1_source_fraction"]),
                "delta_joint_score_mean": float(rec.iloc[0]["delta_vs_baseline_joint_score_mean"]),
                "delta_train_hits": float(rec.iloc[0]["delta_train_hits_vs_baseline"]),
                "delta_holdout_hits": float(rec.iloc[0]["delta_holdout_hits_vs_baseline"]),
                "delta_holdout_enrichment": float(rec.iloc[0]["delta_holdout_enrichment_vs_baseline"]),
                "utility_score": float(rec.iloc[0]["utility_score"]),
            }
        ]
    )
    summary.to_csv(args.output_dir / "multiobjective_summary.csv", index=False)

    print("Multi-objective policy optimization complete.")
    print(rec.to_string(index=False))


if __name__ == "__main__":
    main()
