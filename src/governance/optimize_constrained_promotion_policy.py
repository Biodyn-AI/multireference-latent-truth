#!/usr/bin/env python3
"""Select promotion policy under explicit constraints and blind-holdout objective."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import pandas as pd


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--policy-summary",
        type=Path,
        default=Path(
            "subproject_24_multireference_latent_truth_model/implementation/outputs/"
            "top_tier_readiness_v2/policy_sweep_up_to_conditional_budget842/deconcentration_policy_summary.csv"
        ),
    )
    parser.add_argument(
        "--blind-summary",
        type=Path,
        default=Path(
            "subproject_24_multireference_latent_truth_model/implementation/outputs/"
            "top_tier_readiness_v2/blind_holdout_eval/blind_holdout_validation_summary.csv"
        ),
    )
    parser.add_argument("--baseline-policy", type=str, default="baseline_alpha0")
    parser.add_argument("--train-reference-set", type=str, default="train_high_overlap_union")
    parser.add_argument("--holdout-reference-set", type=str, default="blind_holdout_union")
    parser.add_argument("--max-top1", type=float, default=0.25)
    parser.add_argument("--max-top2", type=float, default=0.50)
    parser.add_argument("--min-joint-delta", type=float, default=-0.01)
    parser.add_argument("--min-train-hit-delta", type=int, default=-2)
    parser.add_argument("--min-train-enrichment", type=float, default=0.20)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "subproject_24_multireference_latent_truth_model/implementation/outputs/"
            "top_tier_readiness_v2/constrained_policy_optimization"
        ),
    )
    return parser.parse_args()


def _policy_from_selection(selection: str) -> str:
    # In this workflow selection labels are produced from policy filenames.
    return selection


def main() -> None:
    args = _parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    pol = pd.read_csv(args.policy_summary, low_memory=False)
    blind = pd.read_csv(args.blind_summary, low_memory=False)

    # Aggregate blind metrics per policy for train and holdout reference sets.
    blind["policy"] = blind["selection"].map(_policy_from_selection)
    train = blind[blind["reference_set"] == args.train_reference_set].copy()
    holdout = blind[blind["reference_set"] == args.holdout_reference_set].copy()
    if train.empty:
        raise ValueError(f"train-reference-set `{args.train_reference_set}` missing in blind summary")
    if holdout.empty:
        raise ValueError(f"holdout-reference-set `{args.holdout_reference_set}` missing in blind summary")

    train_small = train[
        [
            "policy",
            "n_hits",
            "precision_in_selection",
            "enrichment_vs_random",
            "p_value_hypergeom_ge_hits",
        ]
    ].rename(
        columns={
            "n_hits": "train_hits",
            "precision_in_selection": "train_precision",
            "enrichment_vs_random": "train_enrichment",
            "p_value_hypergeom_ge_hits": "train_p_value",
        }
    )
    holdout_small = holdout[
        [
            "policy",
            "n_hits",
            "precision_in_selection",
            "enrichment_vs_random",
            "p_value_hypergeom_ge_hits",
            "n_reference_edges_in_candidate_pool",
        ]
    ].rename(
        columns={
            "n_hits": "holdout_hits",
            "precision_in_selection": "holdout_precision",
            "enrichment_vs_random": "holdout_enrichment",
            "p_value_hypergeom_ge_hits": "holdout_p_value",
            "n_reference_edges_in_candidate_pool": "holdout_candidate_coverage",
        }
    )

    merged = pol.merge(train_small, on="policy", how="left").merge(holdout_small, on="policy", how="left")
    if merged.empty:
        raise ValueError("No overlapping policy labels between policy-summary and blind-summary")

    if args.baseline_policy not in set(merged["policy"]):
        raise ValueError(f"baseline-policy `{args.baseline_policy}` not found")
    baseline = merged[merged["policy"] == args.baseline_policy].iloc[0]

    merged["delta_train_hits_vs_baseline"] = merged["train_hits"] - float(baseline["train_hits"])
    merged["delta_holdout_hits_vs_baseline"] = merged["holdout_hits"] - float(baseline["holdout_hits"])
    merged["delta_holdout_enrichment_vs_baseline"] = merged["holdout_enrichment"] - float(
        baseline["holdout_enrichment"]
    )

    merged["constraint_top1"] = merged["top1_source_fraction"] <= args.max_top1
    merged["constraint_top2"] = merged["top2_source_fraction"] <= args.max_top2
    merged["constraint_joint"] = merged["delta_vs_baseline_joint_score_mean"] >= args.min_joint_delta
    merged["constraint_train_hits"] = merged["delta_train_hits_vs_baseline"] >= args.min_train_hit_delta
    merged["constraint_train_enrichment"] = merged["train_enrichment"] >= args.min_train_enrichment
    constraint_cols = [
        "constraint_top1",
        "constraint_top2",
        "constraint_joint",
        "constraint_train_hits",
        "constraint_train_enrichment",
    ]
    merged["is_feasible"] = merged[constraint_cols].all(axis=1)

    # Lexicographic optimization favors holdout gain while preserving train quality.
    sort_cols: List[str] = [
        "is_feasible",
        "holdout_enrichment",
        "holdout_hits",
        "train_enrichment",
        "top1_source_fraction",
        "joint_score_mean",
    ]
    ascending = [False, False, False, False, True, False]
    ranked = merged.sort_values(sort_cols, ascending=ascending).reset_index(drop=True)

    feasible = ranked[ranked["is_feasible"]].copy()
    if feasible.empty:
        recommendation = ranked.iloc[[0]].copy()
        recommendation["recommendation_reason"] = (
            "no feasible policy met all constraints; selected best unconstrained holdout policy"
        )
    else:
        recommendation = feasible.iloc[[0]].copy()
        recommendation["recommendation_reason"] = "best feasible policy under constrained blind-holdout objective"

    merged.to_csv(args.output_dir / "constrained_policy_merged_metrics.csv", index=False)
    feasible.to_csv(args.output_dir / "constrained_policy_feasible.csv", index=False)
    recommendation.to_csv(args.output_dir / "constrained_policy_recommendation.csv", index=False)

    # Emit one-row summary compatible with checklist-style policy diagnostics.
    rec = recommendation.iloc[0]
    summary = pd.DataFrame(
        [
            {
                "recommended_policy": rec["policy"],
                "recommended_alpha": rec["alpha"],
                "baseline_policy": args.baseline_policy,
                "delta_top1_source_fraction": rec["delta_vs_baseline_top1_source_fraction"],
                "delta_top2_source_fraction": rec["delta_vs_baseline_top2_source_fraction"],
                "delta_joint_score_mean": rec["delta_vs_baseline_joint_score_mean"],
                "delta_train_hits": rec["delta_train_hits_vs_baseline"],
                "delta_holdout_hits": rec["delta_holdout_hits_vs_baseline"],
                "delta_holdout_enrichment": rec["delta_holdout_enrichment_vs_baseline"],
                "holdout_candidate_coverage": rec["holdout_candidate_coverage"],
                "is_feasible": bool(rec["is_feasible"]),
            }
        ]
    )
    summary.to_csv(args.output_dir / "constrained_policy_summary.csv", index=False)

    print("Constrained policy optimization complete.")
    print(recommendation.to_string(index=False))


if __name__ == "__main__":
    main()
