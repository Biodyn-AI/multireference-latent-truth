#!/usr/bin/env python3
"""Build fragility-aware endpoint governance tiers from LOCO/L2O/L3O stress outputs.

Phase XXXI:
1) Merge multilevel endpoint probabilities with first/second/third-order fragility summaries.
2) Compute fragility-penalized endpoint confidence.
3) Export conservative governance tiers and summaries for manuscript integration.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--endpoint-rows",
        type=Path,
        default=Path(
            "subproject_24_multireference_latent_truth_model/implementation/outputs/"
            "top_tier_readiness_v7/multilevel_adaptive_bayes_endpoint_tiering_v1/"
            "multilevel_adaptive_endpoint_rows.csv"
        ),
    )
    parser.add_argument(
        "--loo-summary",
        type=Path,
        default=Path(
            "subproject_24_multireference_latent_truth_model/implementation/outputs/"
            "top_tier_readiness_v7/multilevel_leave_one_cohort_out_v1/"
            "leave_one_cohort_fragility_summary.csv"
        ),
    )
    parser.add_argument(
        "--l2o-summary",
        type=Path,
        default=Path(
            "subproject_24_multireference_latent_truth_model/implementation/outputs/"
            "top_tier_readiness_v7/multilevel_leave_two_cohort_out_v1/"
            "leave_two_cohort_fragility_summary.csv"
        ),
    )
    parser.add_argument(
        "--l3o-summary",
        type=Path,
        default=Path(
            "subproject_24_multireference_latent_truth_model/implementation/outputs/"
            "top_tier_readiness_v7/multilevel_leave_three_cohort_out_v1/"
            "leave_three_cohort_fragility_summary.csv"
        ),
    )
    parser.add_argument("--high-confidence", type=float, default=0.95)
    parser.add_argument("--low-confidence", type=float, default=0.05)
    parser.add_argument("--weight-loo", type=float, default=1.0)
    parser.add_argument("--weight-l2o", type=float, default=1.25)
    parser.add_argument("--weight-l3o", type=float, default=1.5)
    parser.add_argument("--tier-change-level-penalty", type=float, default=0.08)
    parser.add_argument("--tier-change-rate-penalty", type=float, default=0.25)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "subproject_24_multireference_latent_truth_model/implementation/outputs/"
            "top_tier_readiness_v7/fragility_aware_endpoint_governance_v1"
        ),
    )
    return parser.parse_args()


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing required input file: {path}")
    return pd.read_csv(path, low_memory=False)


def _merge_fragility(endpoint: pd.DataFrame, loo: pd.DataFrame, l2o: pd.DataFrame, l3o: pd.DataFrame) -> pd.DataFrame:
    base = endpoint[["cohort", "predictive_prob_joint_pass", "observed_support_eligible", "multilevel_adaptive_joint_tier"]].copy()

    loo_s = loo[
        [
            "cohort",
            "loo_mean_abs_delta_predictive_prob_joint_pass",
            "loo_max_abs_delta_predictive_prob_joint_pass",
            "n_tier_changes_vs_full",
            "n_runs",
        ]
    ].rename(
        columns={
            "loo_mean_abs_delta_predictive_prob_joint_pass": "loo_mean_abs_delta",
            "loo_max_abs_delta_predictive_prob_joint_pass": "loo_max_abs_delta",
            "n_tier_changes_vs_full": "loo_n_tier_changes",
            "n_runs": "loo_n_runs",
        }
    )

    l2o_s = l2o[
        [
            "cohort",
            "l2o_mean_abs_delta_predictive_prob_joint_pass",
            "l2o_max_abs_delta_predictive_prob_joint_pass",
            "n_tier_changes_vs_full",
            "n_runs",
        ]
    ].rename(
        columns={
            "l2o_mean_abs_delta_predictive_prob_joint_pass": "l2o_mean_abs_delta",
            "l2o_max_abs_delta_predictive_prob_joint_pass": "l2o_max_abs_delta",
            "n_tier_changes_vs_full": "l2o_n_tier_changes",
            "n_runs": "l2o_n_runs",
        }
    )

    l3o_s = l3o[
        [
            "cohort",
            "l3o_mean_abs_delta_predictive_prob_joint_pass",
            "l3o_max_abs_delta_predictive_prob_joint_pass",
            "n_tier_changes_vs_full",
            "n_runs",
        ]
    ].rename(
        columns={
            "l3o_mean_abs_delta_predictive_prob_joint_pass": "l3o_mean_abs_delta",
            "l3o_max_abs_delta_predictive_prob_joint_pass": "l3o_max_abs_delta",
            "n_tier_changes_vs_full": "l3o_n_tier_changes",
            "n_runs": "l3o_n_runs",
        }
    )

    out = base.merge(loo_s, on="cohort", how="left").merge(l2o_s, on="cohort", how="left").merge(l3o_s, on="cohort", how="left")
    for col in [
        "loo_mean_abs_delta",
        "loo_max_abs_delta",
        "loo_n_tier_changes",
        "loo_n_runs",
        "l2o_mean_abs_delta",
        "l2o_max_abs_delta",
        "l2o_n_tier_changes",
        "l2o_n_runs",
        "l3o_mean_abs_delta",
        "l3o_max_abs_delta",
        "l3o_n_tier_changes",
        "l3o_n_runs",
    ]:
        out[col] = out[col].fillna(0.0)
    return out


def _fragility_band(max_abs_drift: float, tier_change_levels: int) -> str:
    if max_abs_drift >= 0.20 or tier_change_levels >= 2:
        return "HIGH_FRAGILITY"
    if max_abs_drift >= 0.10 or tier_change_levels == 1:
        return "MODERATE_FRAGILITY"
    return "LOW_FRAGILITY"


def _fragility_aware_tier(
    *,
    support_eligible: bool,
    adjusted_prob: float,
    high_conf: float,
    low_conf: float,
    fragility_band: str,
) -> str:
    if support_eligible:
        if adjusted_prob >= high_conf:
            if fragility_band == "LOW_FRAGILITY":
                return "POWERED_ROBUST_CONFIRMED_PASS"
            return "POWERED_FRAGILITY_SENSITIVE_PASS"
        if adjusted_prob <= low_conf:
            return "POWERED_FRAGILITY_CONFIRMED_FAIL"
        return "POWERED_FRAGILITY_AMBIGUOUS"

    if adjusted_prob >= high_conf:
        return "UNDERPOWERED_FRAGILITY_POSITIVE_SIGNAL"
    if adjusted_prob <= low_conf:
        return "UNDERPOWERED_FRAGILITY_NEGATIVE_SIGNAL"
    return "UNDERPOWERED_FRAGILITY_AMBIGUOUS"


def _plot_fragility_tradeoff(df: pd.DataFrame, out_path: Path) -> None:
    if df.empty:
        return

    color_map = {
        "LOW_FRAGILITY": "#2c7fb8",
        "MODERATE_FRAGILITY": "#fdae61",
        "HIGH_FRAGILITY": "#d7191c",
    }
    colors = [color_map.get(x, "#555555") for x in df["fragility_band"].tolist()]

    fig, ax = plt.subplots(figsize=(9.8, 5.4))
    ax.scatter(
        df["max_abs_fragility_drift"],
        df["fragility_adjusted_joint_prob"],
        c=colors,
        s=95,
        edgecolors="white",
        linewidths=0.9,
    )
    for row in df.itertuples(index=False):
        ax.annotate(str(row.cohort), (float(row.max_abs_fragility_drift), float(row.fragility_adjusted_joint_prob)), fontsize=8)

    ax.axhline(0.95, color="#2166ac", linestyle="--", linewidth=1.0, alpha=0.8)
    ax.axhline(0.05, color="#b2182b", linestyle="--", linewidth=1.0, alpha=0.8)
    ax.set_xlim(-0.005, max(0.27, float(df["max_abs_fragility_drift"].max()) + 0.02))
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("Max endpoint probability drift across LOCO/L2O/L3O")
    ax.set_ylabel("Fragility-adjusted P(joint endpoint pass)")
    ax.set_title("Fragility-Aware Endpoint Governance: Drift vs Adjusted Confidence")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = _parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    endpoint = _read_csv(args.endpoint_rows)
    loo = _read_csv(args.loo_summary)
    l2o = _read_csv(args.l2o_summary)
    l3o = _read_csv(args.l3o_summary)

    out = _merge_fragility(endpoint=endpoint, loo=loo, l2o=l2o, l3o=l3o)

    out["weighted_abs_fragility_drift"] = (
        float(args.weight_loo) * out["loo_mean_abs_delta"]
        + float(args.weight_l2o) * out["l2o_mean_abs_delta"]
        + float(args.weight_l3o) * out["l3o_mean_abs_delta"]
    )
    out["max_abs_fragility_drift"] = out[["loo_max_abs_delta", "l2o_max_abs_delta", "l3o_max_abs_delta"]].max(axis=1)
    out["tier_change_levels"] = (
        (out["loo_n_tier_changes"] > 0).astype(int)
        + (out["l2o_n_tier_changes"] > 0).astype(int)
        + (out["l3o_n_tier_changes"] > 0).astype(int)
    )
    out["tier_change_total"] = out["loo_n_tier_changes"] + out["l2o_n_tier_changes"] + out["l3o_n_tier_changes"]
    out["tier_change_rate"] = out["tier_change_total"] / np.clip(
        out["loo_n_runs"] + out["l2o_n_runs"] + out["l3o_n_runs"],
        1.0,
        None,
    )

    out["fragility_penalty"] = (
        out["weighted_abs_fragility_drift"]
        + float(args.tier_change_level_penalty) * out["tier_change_levels"]
        + float(args.tier_change_rate_penalty) * out["tier_change_rate"]
    )
    out["stability_multiplier"] = (1.0 - out["fragility_penalty"]).clip(lower=0.0, upper=1.0)
    out["fragility_adjusted_joint_prob"] = out["predictive_prob_joint_pass"] * out["stability_multiplier"]
    out["fragility_band"] = [
        _fragility_band(max_abs_drift=float(a), tier_change_levels=int(b))
        for a, b in zip(out["max_abs_fragility_drift"], out["tier_change_levels"])
    ]
    out["fragility_aware_endpoint_tier"] = [
        _fragility_aware_tier(
            support_eligible=bool(s),
            adjusted_prob=float(p),
            high_conf=float(args.high_confidence),
            low_conf=float(args.low_confidence),
            fragility_band=str(fb),
        )
        for s, p, fb in zip(out["observed_support_eligible"], out["fragility_adjusted_joint_prob"], out["fragility_band"])
    ]

    out = out.sort_values("cohort").reset_index(drop=True)
    out.to_csv(args.output_dir / "fragility_aware_endpoint_rows.csv", index=False)

    tier_summary = (
        out.groupby("fragility_aware_endpoint_tier", as_index=False)
        .agg(
            n_cohorts=("cohort", "count"),
            mean_adjusted_joint_prob=("fragility_adjusted_joint_prob", "mean"),
            mean_max_abs_drift=("max_abs_fragility_drift", "mean"),
            mean_tier_change_rate=("tier_change_rate", "mean"),
        )
        .sort_values("fragility_aware_endpoint_tier")
        .reset_index(drop=True)
    )
    tier_summary.to_csv(args.output_dir / "fragility_aware_tier_summary.csv", index=False)

    band_summary = (
        out.groupby("fragility_band", as_index=False)
        .agg(
            n_cohorts=("cohort", "count"),
            mean_adjusted_joint_prob=("fragility_adjusted_joint_prob", "mean"),
            mean_stability_multiplier=("stability_multiplier", "mean"),
            max_abs_drift=("max_abs_fragility_drift", "max"),
        )
        .sort_values("fragility_band")
        .reset_index(drop=True)
    )
    band_summary.to_csv(args.output_dir / "fragility_aware_band_summary.csv", index=False)

    support_summary = (
        out.groupby("observed_support_eligible", as_index=False)
        .agg(
            n_cohorts=("cohort", "count"),
            mean_base_joint_prob=("predictive_prob_joint_pass", "mean"),
            mean_adjusted_joint_prob=("fragility_adjusted_joint_prob", "mean"),
            mean_penalty=("fragility_penalty", "mean"),
            mean_max_abs_drift=("max_abs_fragility_drift", "mean"),
        )
        .sort_values("observed_support_eligible", ascending=False)
        .reset_index(drop=True)
    )
    support_summary.to_csv(args.output_dir / "fragility_aware_support_summary.csv", index=False)

    overall = pd.DataFrame(
        [
            {
                "n_cohorts": int(len(out)),
                "mean_base_joint_prob": float(out["predictive_prob_joint_pass"].mean()),
                "mean_adjusted_joint_prob": float(out["fragility_adjusted_joint_prob"].mean()),
                "mean_stability_multiplier": float(out["stability_multiplier"].mean()),
                "max_abs_fragility_drift": float(out["max_abs_fragility_drift"].max()),
                "n_high_fragility_cohorts": int((out["fragility_band"] == "HIGH_FRAGILITY").sum()),
                "n_moderate_fragility_cohorts": int((out["fragility_band"] == "MODERATE_FRAGILITY").sum()),
                "n_low_fragility_cohorts": int((out["fragility_band"] == "LOW_FRAGILITY").sum()),
            }
        ]
    )
    overall.to_csv(args.output_dir / "fragility_aware_overall_summary.csv", index=False)

    _plot_fragility_tradeoff(out, args.output_dir / "fragility_aware_drift_vs_adjusted_probability.png")

    print("Fragility-aware endpoint governance complete.")
    print(overall.to_string(index=False))


if __name__ == "__main__":
    main()
