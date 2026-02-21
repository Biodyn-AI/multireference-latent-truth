#!/usr/bin/env python3
"""Joint budget-policy optimizer with strict no-harm feasibility constraints.

This script operationalizes budget-policy selection as a single process:
1) run policy sweeps per budget,
2) run blind-holdout validation per budget,
3) merge metrics and compute baseline-relative deltas,
4) recommend one policy per budget with feasibility-first logic,
5) emit checklist-compatible budget summary for top-tier readiness gate G10.
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
from typing import Dict, List

import pandas as pd


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--budgets", type=str, default="842,1000,1200,1600,2400,3200,4000")
    parser.add_argument("--alphas", type=str, default="0.0,0.0001,0.0002,0.0005,0.001,0.002,0.005,0.01")
    parser.add_argument("--candidate-mode", type=str, default="up_to_conditional")
    parser.add_argument("--baseline-policy", type=str, default="baseline_alpha0")
    parser.add_argument("--train-reference-set", type=str, default="train_high_overlap_union")
    parser.add_argument("--holdout-reference-set", type=str, default="blind_holdout_union")
    parser.add_argument("--min-joint-delta", type=float, default=-0.003)
    parser.add_argument("--min-train-hit-delta", type=int, default=-1)
    parser.add_argument("--min-top1-reduction", type=float, default=0.001)
    parser.add_argument("--min-train-enrichment", type=float, default=0.2)
    parser.add_argument("--min-holdout-coverage", type=int, default=10)
    parser.add_argument("--max-top1", type=float, default=0.25)
    parser.add_argument("--max-top2", type=float, default=0.50)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "subproject_24_multireference_latent_truth_model/implementation/outputs/"
            "top_tier_readiness_v4/joint_budget_policy_optimization"
        ),
    )
    return parser.parse_args()


def _run(cmd: List[str]) -> None:
    proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"Command failed ({proc.returncode}): {' '.join(cmd)}\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
        )


def _parse_int_list(csv_spec: str) -> List[int]:
    values = [int(x.strip()) for x in csv_spec.split(",") if x.strip()]
    if not values:
        raise ValueError("No values parsed")
    return values


def _join_metrics(
    policy_summary: pd.DataFrame,
    blind_summary: pd.DataFrame,
    baseline_policy: str,
    train_reference_set: str,
    holdout_reference_set: str,
) -> pd.DataFrame:
    train = blind_summary[blind_summary["reference_set"] == train_reference_set].copy()
    holdout = blind_summary[blind_summary["reference_set"] == holdout_reference_set].copy()
    if train.empty:
        raise ValueError(f"train reference set missing: {train_reference_set}")
    if holdout.empty:
        raise ValueError(f"holdout reference set missing: {holdout_reference_set}")

    train_small = train[
        [
            "selection",
            "n_hits",
            "enrichment_vs_random",
            "p_value_hypergeom_ge_hits",
        ]
    ].rename(
        columns={
            "selection": "policy",
            "n_hits": "train_hits",
            "enrichment_vs_random": "train_enrichment",
            "p_value_hypergeom_ge_hits": "train_p_value",
        }
    )
    holdout_small = holdout[
        [
            "selection",
            "n_hits",
            "enrichment_vs_random",
            "p_value_hypergeom_ge_hits",
            "n_reference_edges_in_candidate_pool",
        ]
    ].rename(
        columns={
            "selection": "policy",
            "n_hits": "holdout_hits",
            "enrichment_vs_random": "holdout_enrichment",
            "p_value_hypergeom_ge_hits": "holdout_p_value",
            "n_reference_edges_in_candidate_pool": "holdout_candidate_coverage",
        }
    )

    merged = policy_summary.merge(train_small, on="policy", how="left").merge(holdout_small, on="policy", how="left")
    if baseline_policy not in set(merged["policy"]):
        raise ValueError(f"Baseline policy missing in merged table: {baseline_policy}")
    base = merged[merged["policy"] == baseline_policy].iloc[0]

    if "delta_vs_baseline_joint_score_mean" in merged.columns:
        merged["delta_joint_score_mean"] = merged["delta_vs_baseline_joint_score_mean"].astype(float)
    else:
        merged["delta_joint_score_mean"] = merged["joint_score_mean"].astype(float) - float(base["joint_score_mean"])

    if "delta_vs_baseline_top1_source_fraction" in merged.columns:
        merged["delta_top1_source_fraction"] = merged["delta_vs_baseline_top1_source_fraction"].astype(float)
    else:
        merged["delta_top1_source_fraction"] = (
            merged["top1_source_fraction"].astype(float) - float(base["top1_source_fraction"])
        )

    if "delta_vs_baseline_top2_source_fraction" in merged.columns:
        merged["delta_top2_source_fraction"] = merged["delta_vs_baseline_top2_source_fraction"].astype(float)
    else:
        merged["delta_top2_source_fraction"] = (
            merged["top2_source_fraction"].astype(float) - float(base["top2_source_fraction"])
        )

    merged["delta_train_hits"] = merged["train_hits"].astype(float) - float(base["train_hits"])
    merged["delta_holdout_hits"] = merged["holdout_hits"].astype(float) - float(base["holdout_hits"])
    merged["delta_holdout_enrichment"] = (
        merged["holdout_enrichment"].astype(float) - float(base["holdout_enrichment"])
    )
    return merged


def _recommend_per_budget(merged: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    merged = merged.copy()
    merged["is_nonbaseline"] = merged["policy"].astype(str) != args.baseline_policy
    merged["constraint_top1"] = merged["top1_source_fraction"].astype(float) <= args.max_top1
    merged["constraint_top2"] = merged["top2_source_fraction"].astype(float) <= args.max_top2
    merged["constraint_joint"] = merged["delta_joint_score_mean"].astype(float) >= args.min_joint_delta
    merged["constraint_train_hits"] = merged["delta_train_hits"].astype(float) >= float(args.min_train_hit_delta)
    merged["constraint_top1_reduction"] = (
        merged["delta_top1_source_fraction"].astype(float) <= -float(args.min_top1_reduction)
    )
    merged["constraint_train_enrichment"] = merged["train_enrichment"].astype(float) >= args.min_train_enrichment
    merged["constraint_holdout_coverage"] = (
        merged["holdout_candidate_coverage"].astype(float) >= float(args.min_holdout_coverage)
    )

    constraint_cols = [
        "constraint_top1",
        "constraint_top2",
        "constraint_joint",
        "constraint_train_hits",
        "constraint_top1_reduction",
        "constraint_train_enrichment",
        "constraint_holdout_coverage",
    ]
    merged["is_feasible_nonbaseline"] = merged["is_nonbaseline"] & merged[constraint_cols].all(axis=1)

    feasible_nonbaseline = merged[merged["is_feasible_nonbaseline"]].copy()
    if feasible_nonbaseline.empty:
        recommendation = merged[merged["policy"] == args.baseline_policy].head(1).copy()
        recommendation["recommendation_reason"] = "no feasible non-baseline policy; fallback to baseline"
        recommendation["is_feasible"] = True
    else:
        recommendation = feasible_nonbaseline.sort_values(
            [
                "holdout_enrichment",
                "holdout_hits",
                "delta_train_hits",
                "delta_top1_source_fraction",
                "delta_joint_score_mean",
            ],
            ascending=[False, False, False, True, False],
        ).head(1)
        recommendation = recommendation.copy()
        recommendation["recommendation_reason"] = "best feasible non-baseline under strict no-harm constraints"
        recommendation["is_feasible"] = True

    merged["is_feasible"] = merged["policy"] == args.baseline_policy
    merged.loc[merged["is_feasible_nonbaseline"], "is_feasible"] = True
    return merged, recommendation


def main() -> None:
    args = _parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    budgets = _parse_int_list(args.budgets)

    budget_recommendations: List[pd.DataFrame] = []
    merged_tables: List[pd.DataFrame] = []
    feasible_tables: List[pd.DataFrame] = []

    for budget in budgets:
        budget_dir = args.output_dir / f"budget_{budget}"
        policy_dir = budget_dir / "policy"
        blind_dir = budget_dir / "blind"
        policy_dir.mkdir(parents=True, exist_ok=True)
        blind_dir.mkdir(parents=True, exist_ok=True)

        _run(
            [
                "python",
                "subproject_24_multireference_latent_truth_model/implementation/run_deconcentration_policy_sweep.py",
                "--budget",
                str(budget),
                "--alphas",
                args.alphas,
                "--candidate-mode",
                args.candidate_mode,
                "--output-dir",
                str(policy_dir),
            ]
        )
        _run(
            [
                "python",
                "subproject_24_multireference_latent_truth_model/implementation/run_blind_holdout_protocol.py",
                "--selection-dir",
                str(policy_dir),
                "--candidate-mode",
                args.candidate_mode,
                "--comparison-selection",
                args.baseline_policy,
                "--output-dir",
                str(blind_dir),
            ]
        )

        policy_summary = pd.read_csv(policy_dir / "deconcentration_policy_summary.csv", low_memory=False)
        blind_summary = pd.read_csv(blind_dir / "blind_holdout_validation_summary.csv", low_memory=False)
        merged = _join_metrics(
            policy_summary=policy_summary,
            blind_summary=blind_summary,
            baseline_policy=args.baseline_policy,
            train_reference_set=args.train_reference_set,
            holdout_reference_set=args.holdout_reference_set,
        )
        merged["budget"] = int(budget)

        merged, recommendation = _recommend_per_budget(merged=merged, args=args)
        recommendation = recommendation.copy()
        recommendation["budget"] = int(budget)

        merged_tables.append(merged)
        budget_recommendations.append(recommendation)
        feasible_tables.append(merged[merged["is_feasible_nonbaseline"]].copy())

    all_merged = pd.concat(merged_tables, ignore_index=True)
    all_merged.to_csv(args.output_dir / "joint_budget_policy_table.csv", index=False)

    feasible = pd.concat(feasible_tables, ignore_index=True) if feasible_tables else all_merged.head(0).copy()
    feasible.to_csv(args.output_dir / "joint_budget_policy_feasible_nonbaseline.csv", index=False)

    budget_rec = pd.concat(budget_recommendations, ignore_index=True)
    budget_rec = budget_rec.sort_values("budget").reset_index(drop=True)
    budget_rec.to_csv(args.output_dir / "joint_budget_policy_budget_recommendations.csv", index=False)

    summary = pd.DataFrame(
        [
            {
                "budget": int(r["budget"]),
                "recommended_policy": str(r["policy"]),
                "recommended_alpha": float(r["alpha"]),
                "baseline_policy": args.baseline_policy,
                "delta_top1_source_fraction": float(r["delta_top1_source_fraction"]),
                "delta_top2_source_fraction": float(r["delta_top2_source_fraction"]),
                "delta_joint_score_mean": float(r["delta_joint_score_mean"]),
                "delta_train_hits": float(r["delta_train_hits"]),
                "delta_hits_high_overlap_union": float(r["delta_train_hits"]),
                "delta_holdout_hits": float(r["delta_holdout_hits"]),
                "delta_holdout_enrichment": float(r["delta_holdout_enrichment"]),
                "train_enrichment": float(r["train_enrichment"]),
                "holdout_enrichment": float(r["holdout_enrichment"]),
                "holdout_candidate_coverage": float(r["holdout_candidate_coverage"]),
                "is_feasible": bool(r["is_feasible"]),
                "recommendation_reason": str(r["recommendation_reason"]),
            }
            for _, r in budget_rec.iterrows()
        ]
    )
    summary.to_csv(args.output_dir / "budget_sensitivity_summary.csv", index=False)

    active = summary[(summary["recommended_policy"] != args.baseline_policy) & (summary["is_feasible"] == True)]  # noqa: E712
    if active.empty:
        global_rec = summary.sort_values(
            ["holdout_enrichment", "delta_joint_score_mean"],
            ascending=[False, False],
        ).head(1)
    else:
        global_rec = active.sort_values(
            ["holdout_enrichment", "delta_top1_source_fraction", "delta_joint_score_mean"],
            ascending=[False, True, False],
        ).head(1)
    global_rec.to_csv(args.output_dir / "joint_budget_policy_global_recommendation.csv", index=False)

    print("Joint budget-policy optimization complete.")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
