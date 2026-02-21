#!/usr/bin/env python3
"""Build unified summary tables for the consolidated Subproject 24 manuscript."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "subproject_24_multireference_latent_truth_model/implementation/outputs/unified_paper_tables"
        ),
    )
    return parser.parse_args()


def _read(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    return pd.read_csv(path, low_memory=False)


def _safe_div(a: float, b: float) -> float:
    if b == 0:
        return float("nan")
    return float(a / b)


def main() -> None:
    args = _parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # 1) Full-universe robustness regime snapshots from Phase III.
    src_agg = _read(
        Path(
            "subproject_24_multireference_latent_truth_model/implementation/outputs/source_dropout_4way_tuned_regime_aggregate_3way.csv"
        )
    )
    fam_agg = _read(
        Path(
            "subproject_24_multireference_latent_truth_model/implementation/outputs/family_block_dropout_4way_tuned_regime_aggregate_3way.csv"
        )
    )

    source_cols = [
        "regime",
        "mean_rank_disagreement",
        "max_rank_disagreement",
        "mean_abs_delta_pi",
        "max_abs_delta_pi",
        "mean_abs_delta_rho",
    ]
    family_cols = [
        "regime",
        "mean_rank_disagreement",
        "max_rank_disagreement",
        "mean_abs_delta_pi",
        "max_abs_delta_pi",
        "mean_abs_delta_rho",
    ]
    src_small = src_agg[source_cols].copy()
    src_small["stress_protocol"] = "source_dropout_full_universe"
    fam_small = fam_agg[family_cols].copy()
    fam_small["stress_protocol"] = "family_block_dropout_full_universe"
    robustness_progression = pd.concat([src_small, fam_small], ignore_index=True)
    robustness_progression = robustness_progression.sort_values(
        ["stress_protocol", "mean_rank_disagreement"],
        ascending=[True, True],
    ).reset_index(drop=True)
    robustness_progression.to_csv(args.output_dir / "robustness_progression_summary.csv", index=False)

    # 2) Context-vs-full perturbation calibration for aligned hyperparameters.
    context_cal = _read(
        Path(
            "subproject_24_multireference_latent_truth_model/implementation/outputs/context_v2_best_fullfit/perturbation_multipanel/perturbation_multipanel_calibration_summary.csv"
        )
    )
    full_cal = _read(
        Path(
            "subproject_24_multireference_latent_truth_model/implementation/outputs/full_v2_best_fullfit/perturbation_multipanel/perturbation_multipanel_calibration_summary.csv"
        )
    )

    context_row = context_cal[context_cal["reference_set"] == "high_overlap_union"].iloc[0]
    full_row = full_cal[full_cal["reference_set"] == "high_overlap_union"].iloc[0]

    calib_rows: List[dict[str, object]] = [
        {
            "universe": "context_v2",
            "n_reference_edges_in_universe": int(context_row["n_reference_edges_in_universe"]),
            "best_precision": float(context_row["best_precision"]),
            "best_recall": float(context_row["best_recall"]),
            "best_f1": float(context_row["best_f1"]),
            "precision_at_500": float(context_row["precision_at_500"]),
            "tp_at_500": int(context_row["tp_at_500"]),
            "precision_at_5000": float(context_row["precision_at_5000"]),
            "tp_at_5000": int(context_row["tp_at_5000"]),
            "precision_at_20000": float(context_row["precision_at_20000"]),
            "tp_at_20000": int(context_row["tp_at_20000"]),
        },
        {
            "universe": "full_universe",
            "n_reference_edges_in_universe": int(full_row["n_reference_edges_in_universe"]),
            "best_precision": float(full_row["best_precision"]),
            "best_recall": float(full_row["best_recall"]),
            "best_f1": float(full_row["best_f1"]),
            "precision_at_500": float(full_row["precision_at_500"]),
            "tp_at_500": int(full_row["tp_at_500"]),
            "precision_at_5000": float(full_row["precision_at_5000"]),
            "tp_at_5000": int(full_row["tp_at_5000"]),
            "precision_at_20000": float(full_row["precision_at_20000"]),
            "tp_at_20000": int(full_row["tp_at_20000"]),
        },
    ]
    calibration_comparison = pd.DataFrame(calib_rows)
    calibration_comparison.to_csv(args.output_dir / "calibration_full_vs_context.csv", index=False)

    uplift = pd.DataFrame(
        [
            {
                "metric": "precision_at_5000",
                "context_value": float(context_row["precision_at_5000"]),
                "full_value": float(full_row["precision_at_5000"]),
                "context_over_full_ratio": _safe_div(
                    float(context_row["precision_at_5000"]),
                    float(full_row["precision_at_5000"]),
                ),
                "context_minus_full": float(context_row["precision_at_5000"] - full_row["precision_at_5000"]),
            },
            {
                "metric": "best_f1",
                "context_value": float(context_row["best_f1"]),
                "full_value": float(full_row["best_f1"]),
                "context_over_full_ratio": _safe_div(
                    float(context_row["best_f1"]),
                    float(full_row["best_f1"]),
                ),
                "context_minus_full": float(context_row["best_f1"] - full_row["best_f1"]),
            },
            {
                "metric": "best_precision",
                "context_value": float(context_row["best_precision"]),
                "full_value": float(full_row["best_precision"]),
                "context_over_full_ratio": _safe_div(
                    float(context_row["best_precision"]),
                    float(full_row["best_precision"]),
                ),
                "context_minus_full": float(context_row["best_precision"] - full_row["best_precision"]),
            },
        ]
    )
    uplift.to_csv(args.output_dir / "calibration_context_uplift_summary.csv", index=False)

    # 3) Joint objective sweep summary (context v2, multi-seed).
    joint = _read(
        Path(
            "subproject_24_multireference_latent_truth_model/implementation/outputs/family_hb_joint_tuning_context_v2_full/family_hb_joint_tuning_summary.csv"
        )
    )
    joint_cols = [
        "run_id",
        "n_seeds",
        "objective_score_mean",
        "objective_score_ci95_low",
        "objective_score_ci95_high",
        "source_objective_component_mean",
        "block_objective_component_mean",
        "perturbation_reward_component_mean",
        "source_mean_rank_disagreement_mean",
        "source_max_rank_disagreement_mean",
        "block_mean_rank_disagreement_mean",
        "block_max_rank_disagreement_mean",
        "source_baseline_top_weighted_f1_mean",
        "source_baseline_top_weighted_f1_ci95_low",
        "source_baseline_top_weighted_f1_ci95_high",
    ]
    joint_small = joint[joint_cols].copy().sort_values("objective_score_mean", ascending=True).reset_index(drop=True)
    joint_small.to_csv(args.output_dir / "joint_objective_context_sweep_summary.csv", index=False)

    # 4) Dual-universe governance summaries.
    dual_summary = _read(
        Path(
            "subproject_24_multireference_latent_truth_model/implementation/outputs/dual_universe_governance_v1/dual_universe_claim_tiers_summary.csv"
        )
    )
    dual_consistency = _read(
        Path(
            "subproject_24_multireference_latent_truth_model/implementation/outputs/dual_universe_governance_v1/dual_universe_consistency_summary.csv"
        )
    )
    dual_edges = _read(
        Path(
            "subproject_24_multireference_latent_truth_model/implementation/outputs/dual_universe_governance_v1/dual_universe_claim_tiers_edges.csv"
        )
    )
    dual_summary.to_csv(args.output_dir / "dual_universe_claim_tiers_summary.csv", index=False)
    dual_consistency.to_csv(args.output_dir / "dual_universe_consistency_summary.csv", index=False)

    promoted = dual_edges[dual_edges["promotion_allowed"] == True].copy()
    promoted["joint_promoted_score"] = promoted[["posterior_full", "posterior_context"]].min(axis=1)
    promoted_top = promoted.sort_values(
        ["joint_promoted_score", "posterior_context", "posterior_full"],
        ascending=[False, False, False],
    )[
        [
            "source",
            "target",
            "dual_universe_claim_tier",
            "posterior_full",
            "posterior_context",
            "joint_promoted_score",
            "is_reference_full",
            "is_reference_context",
        ]
    ].head(200)
    promoted_top.to_csv(args.output_dir / "dual_universe_promoted_edges_top200.csv", index=False)

    conditional = dual_edges[dual_edges["dual_universe_claim_tier"] == "DUAL_TIER3_CONDITIONAL"].copy()
    conditional["conditional_priority_score"] = (
        conditional["posterior_full"].fillna(0.0) + conditional["posterior_context"].fillna(0.0)
    ) / 2.0
    conditional_priority = conditional.sort_values(
        ["conditional_priority_score", "posterior_full", "posterior_context"],
        ascending=[False, False, False],
    )[
        [
            "source",
            "target",
            "in_context_universe",
            "posterior_full",
            "posterior_context",
            "conditional_priority_score",
            "tier_rank_gap_abs",
            "is_reference_full",
            "is_reference_context",
        ]
    ].head(200)
    conditional_priority.to_csv(args.output_dir / "dual_universe_conditional_priority_top200.csv", index=False)

    promoted_diversity = pd.DataFrame(
        [
            {
                "metric": "n_promoted_edges_total",
                "value": float(promoted.shape[0]),
            },
            {
                "metric": "n_promoted_unique_sources",
                "value": float(promoted["source"].nunique()),
            },
            {
                "metric": "n_promoted_unique_targets",
                "value": float(promoted["target"].nunique()),
            },
            {
                "metric": "top1_source_fraction_promoted",
                "value": _safe_div(
                    float(promoted["source"].value_counts().iloc[0]),
                    float(promoted.shape[0]),
                ),
            },
            {
                "metric": "top2_source_fraction_promoted",
                "value": _safe_div(
                    float(promoted["source"].value_counts().head(2).sum()),
                    float(promoted.shape[0]),
                ),
            },
            {
                "metric": "promoted_reference_fraction_full",
                "value": _safe_div(
                    float(promoted["is_reference_full"].fillna(0).sum()),
                    float(promoted.shape[0]),
                ),
            },
            {
                "metric": "promoted_reference_fraction_context",
                "value": _safe_div(
                    float(promoted["is_reference_context"].fillna(0).sum()),
                    float(promoted.shape[0]),
                ),
            },
        ]
    )
    promoted_diversity.to_csv(args.output_dir / "dual_universe_promoted_diversity_summary.csv", index=False)
    promoted_top_sources = (
        promoted["source"].value_counts().rename_axis("source").reset_index(name="n_promoted_edges").head(25)
    )
    promoted_top_sources["fraction_promoted_edges"] = promoted_top_sources["n_promoted_edges"] / max(promoted.shape[0], 1)
    promoted_top_sources.to_csv(args.output_dir / "dual_universe_promoted_top_sources.csv", index=False)

    conditional_diversity = pd.DataFrame(
        [
            {
                "metric": "n_conditional_edges_total",
                "value": float(conditional.shape[0]),
            },
            {
                "metric": "n_conditional_unique_sources",
                "value": float(conditional["source"].nunique()),
            },
            {
                "metric": "n_conditional_unique_targets",
                "value": float(conditional["target"].nunique()),
            },
            {
                "metric": "conditional_context_coverage_fraction",
                "value": float(conditional["in_context_universe"].mean()),
            },
            {
                "metric": "conditional_missing_context_posterior_fraction",
                "value": float(conditional["posterior_context"].isna().mean()),
            },
            {
                "metric": "conditional_mean_tier_gap_abs",
                "value": float(conditional["tier_rank_gap_abs"].fillna(0.0).mean()),
            },
        ]
    )
    conditional_diversity.to_csv(args.output_dir / "dual_universe_conditional_diversity_summary.csv", index=False)
    conditional_top_sources = (
        conditional["source"]
        .value_counts()
        .rename_axis("source")
        .reset_index(name="n_conditional_edges")
        .head(25)
    )
    conditional_top_sources["fraction_conditional_edges"] = conditional_top_sources["n_conditional_edges"] / max(
        conditional.shape[0], 1
    )
    conditional_top_sources.to_csv(args.output_dir / "dual_universe_conditional_top_sources.csv", index=False)

    # 5) High-level phase timeline table for unified manuscript.
    timeline = pd.DataFrame(
        [
            {
                "phase": "I",
                "theme": "Baseline and dependence-aware latent truth",
                "primary_report": "full_universe_multifactor_dropout_robustness_report.md",
                "status": "completed",
            },
            {
                "phase": "II",
                "theme": "Family-regularized dependence stress tests",
                "primary_report": "family_regularization_phase2_report.md",
                "status": "completed",
            },
            {
                "phase": "III",
                "theme": "Hierarchical family-Bayes robustness",
                "primary_report": "hierarchical_family_phase3_report.md",
                "status": "completed",
            },
            {
                "phase": "IV",
                "theme": "Multipanel perturbation calibration",
                "primary_report": "multipanel_perturbation_phase4_report.md",
                "status": "completed",
            },
            {
                "phase": "V",
                "theme": "Joint objective pilot",
                "primary_report": "joint_objective_phase5_report.md",
                "status": "completed",
            },
            {
                "phase": "VI",
                "theme": "Context-matched uncertainty sweep",
                "primary_report": "joint_objective_uncertainty_context_phase6_report.md",
                "status": "completed",
            },
            {
                "phase": "VII",
                "theme": "Dual-universe claim governance",
                "primary_report": "dual_universe_governance_phase7_report.md",
                "status": "completed",
            },
        ]
    )
    timeline.to_csv(args.output_dir / "phase_timeline_summary.csv", index=False)

    phase_kpis = pd.DataFrame(
        [
            {
                "kpi": "context_over_full_precision_at_5000_ratio",
                "value": _safe_div(float(context_row["precision_at_5000"]), float(full_row["precision_at_5000"])),
            },
            {
                "kpi": "context_over_full_best_f1_ratio",
                "value": _safe_div(float(context_row["best_f1"]), float(full_row["best_f1"])),
            },
            {
                "kpi": "promoted_edges_total",
                "value": float((dual_summary["n_promotion_allowed"]).sum()),
            },
            {
                "kpi": "promoted_edge_fraction_full_universe",
                "value": _safe_div(
                    float((dual_summary["n_promotion_allowed"]).sum()),
                    float(dual_edges.shape[0]),
                ),
            },
            {
                "kpi": "context_edge_fraction_promoted",
                "value": _safe_div(
                    float((dual_summary["n_promotion_allowed"]).sum()),
                    float(dual_consistency.iloc[0]["n_context_edges"]),
                ),
            },
            {
                "kpi": "context_tier_exact_match_fraction",
                "value": float(dual_consistency.iloc[0]["fraction_exact_tier_match"]),
            },
            {
                "kpi": "context_tier_within_one_fraction",
                "value": float(dual_consistency.iloc[0]["fraction_match_within_one_tier"]),
            },
        ]
    )
    phase_kpis.to_csv(args.output_dir / "phase_vii_kpi_summary.csv", index=False)

    # 6) Deconcentration sweep summaries (Phase X), if available.
    deconc_summary_path = Path(
        "subproject_24_multireference_latent_truth_model/implementation/outputs/deconcentration_policy_sweep_v1/deconcentration_policy_summary.csv"
    )
    deconc_reco_path = Path(
        "subproject_24_multireference_latent_truth_model/implementation/outputs/deconcentration_policy_sweep_v1/deconcentration_policy_recommendation.csv"
    )
    deconc_summary_written = False
    deconc_reco_written = False
    if deconc_summary_path.exists():
        deconc_summary = _read(deconc_summary_path)
        deconc_summary.to_csv(args.output_dir / "deconcentration_policy_summary.csv", index=False)
        deconc_summary_written = True
    if deconc_reco_path.exists():
        deconc_reco = _read(deconc_reco_path)
        deconc_reco.to_csv(args.output_dir / "deconcentration_policy_recommendation.csv", index=False)
        deconc_reco_written = True

    budget_sens_path = Path(
        "subproject_24_multireference_latent_truth_model/implementation/outputs/deconcentration_budget_sensitivity_v1/budget_sensitivity_summary.csv"
    )
    budget_sens_written = False
    if budget_sens_path.exists():
        budget_sens = _read(budget_sens_path)
        budget_sens.to_csv(args.output_dir / "deconcentration_budget_sensitivity_summary.csv", index=False)
        budget_sens_written = True

    print("Unified paper tables built:")
    for name in [
        "phase_timeline_summary.csv",
        "robustness_progression_summary.csv",
        "calibration_full_vs_context.csv",
        "calibration_context_uplift_summary.csv",
        "joint_objective_context_sweep_summary.csv",
        "dual_universe_claim_tiers_summary.csv",
        "dual_universe_consistency_summary.csv",
        "dual_universe_promoted_edges_top200.csv",
        "dual_universe_conditional_priority_top200.csv",
        "dual_universe_promoted_diversity_summary.csv",
        "dual_universe_conditional_diversity_summary.csv",
        "dual_universe_promoted_top_sources.csv",
        "dual_universe_conditional_top_sources.csv",
        "phase_vii_kpi_summary.csv",
    ]:
        print(f"- {args.output_dir / name}")
    if deconc_summary_written:
        print(f"- {args.output_dir / 'deconcentration_policy_summary.csv'}")
    if deconc_reco_written:
        print(f"- {args.output_dir / 'deconcentration_policy_recommendation.csv'}")
    if budget_sens_written:
        print(f"- {args.output_dir / 'deconcentration_budget_sensitivity_summary.csv'}")


if __name__ == "__main__":
    main()
