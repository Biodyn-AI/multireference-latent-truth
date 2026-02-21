#!/usr/bin/env python3
"""Run a strategic policy-sensitivity audit over fragility-governance sweep outputs.

Phase XXXII:
1) Aggregate cohort-level endpoint decisions across all fragility policy profiles.
2) Quantify how policy choices change endpoint tiers and adjusted confidence.
3) Export strategic operating profiles and cohort-prioritized follow-up actions.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sweep-summary",
        type=Path,
        default=Path(
            "subproject_24_multireference_latent_truth_model/implementation/outputs/"
            "top_tier_readiness_v7/fragility_aware_weight_sweep_v1/policy_sweep_summary.csv"
        ),
    )
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=Path(
            "subproject_24_multireference_latent_truth_model/implementation/outputs/"
            "top_tier_readiness_v7/fragility_aware_weight_sweep_v1/runs"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "subproject_24_multireference_latent_truth_model/implementation/outputs/"
            "top_tier_readiness_v7/fragility_policy_sensitivity_audit_v1"
        ),
    )
    return parser.parse_args()


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    return pd.read_csv(path, low_memory=False)


def _entropy(labels: pd.Series) -> float:
    probs = labels.value_counts(normalize=True).to_numpy(dtype=float)
    probs = probs[probs > 0]
    if probs.size == 0:
        return 0.0
    return float(-np.sum(probs * np.log(probs)))


def _load_policy_long(sweep: pd.DataFrame, runs_dir: Path) -> pd.DataFrame:
    rows: List[pd.DataFrame] = []
    required_cols = [
        "cohort",
        "observed_support_eligible",
        "fragility_adjusted_joint_prob",
        "fragility_penalty",
        "stability_multiplier",
        "fragility_band",
        "fragility_aware_endpoint_tier",
    ]

    for policy in sweep.itertuples(index=False):
        policy_id = str(policy.policy_id)
        run_file = runs_dir / policy_id / "fragility_aware_endpoint_rows.csv"
        run_df = _read_csv(run_file)

        missing = [c for c in required_cols if c not in run_df.columns]
        if missing:
            raise ValueError(f"{run_file} missing columns: {missing}")

        cur = run_df[required_cols].copy()
        cur["policy_id"] = policy_id
        cur["weight_l2o"] = float(policy.weight_l2o)
        cur["weight_l3o"] = float(policy.weight_l3o)
        cur["tier_change_level_penalty"] = float(policy.tier_change_level_penalty)
        cur["tier_change_rate_penalty"] = float(policy.tier_change_rate_penalty)
        cur["policy_mean_adjusted_joint_prob"] = float(policy.mean_adjusted_joint_prob)
        cur["policy_mean_penalty"] = float(policy.mean_penalty)
        cur["is_recommended"] = bool(policy.is_recommended)
        rows.append(cur)

    if not rows:
        raise ValueError("No policy runs were loaded from the sweep directory.")
    return pd.concat(rows, axis=0, ignore_index=True)


def _classify_sensitivity(modal_fraction: float, prob_span: float, tier_entropy: float) -> str:
    # We combine three signals because each catches a different instability mode:
    # categorical switches, confidence spread, and distributional uncertainty.
    if modal_fraction >= 0.95 and prob_span <= 0.05 and tier_entropy <= 0.15:
        return "STABLE"
    if modal_fraction >= 0.75 and prob_span <= 0.15 and tier_entropy <= 0.75:
        return "MODERATE_SENSITIVITY"
    return "HIGH_SENSITIVITY"


def _recommend_action(
    *,
    support_eligible: bool,
    high_fragility_rate: float,
    negative_signal_rate: float,
    robust_pass_rate: float,
    sensitivity_class: str,
) -> str:
    if support_eligible and robust_pass_rate >= 0.9 and sensitivity_class == "STABLE":
        return "Anchor positive claims; focus on biological mechanism depth rather than governance retuning."
    if (not support_eligible) and high_fragility_rate >= 0.5:
        return "Highest priority for new-cohort acquisition and targeted perturbation expansion."
    if negative_signal_rate >= 0.5:
        return "Prioritize falsification-oriented follow-up before any positive claim escalation."
    return "Run focused robustness expansion and collect support edges before tier promotion."


def _compute_cohort_summary(long_df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for cohort, group in long_df.groupby("cohort"):
        tier_counts = group["fragility_aware_endpoint_tier"].value_counts(dropna=False)
        modal_tier = str(tier_counts.index[0])
        modal_fraction = float(tier_counts.iloc[0] / max(1, len(group)))
        prob_min = float(group["fragility_adjusted_joint_prob"].min())
        prob_max = float(group["fragility_adjusted_joint_prob"].max())
        prob_span = prob_max - prob_min
        tier_entropy = _entropy(group["fragility_aware_endpoint_tier"])
        support_eligible = bool(group["observed_support_eligible"].iloc[0])
        high_fragility_rate = float((group["fragility_band"] == "HIGH_FRAGILITY").mean())
        robust_pass_rate = float((group["fragility_aware_endpoint_tier"] == "POWERED_ROBUST_CONFIRMED_PASS").mean())
        negative_signal_rate = float(
            (group["fragility_aware_endpoint_tier"] == "UNDERPOWERED_FRAGILITY_NEGATIVE_SIGNAL").mean()
        )
        sensitivity_class = _classify_sensitivity(
            modal_fraction=modal_fraction,
            prob_span=prob_span,
            tier_entropy=tier_entropy,
        )

        span_norm = min(1.0, prob_span / 0.5)
        priority_score = float(
            0.35 * (0.0 if support_eligible else 1.0)
            + 0.30 * high_fragility_rate
            + 0.20 * (1.0 - modal_fraction)
            + 0.15 * span_norm
        )

        rows.append(
            {
                "cohort": str(cohort),
                "observed_support_eligible": support_eligible,
                "n_policies": int(len(group)),
                "n_unique_tiers": int(group["fragility_aware_endpoint_tier"].nunique()),
                "modal_tier": modal_tier,
                "modal_tier_fraction": modal_fraction,
                "tier_entropy": tier_entropy,
                "adjusted_prob_mean": float(group["fragility_adjusted_joint_prob"].mean()),
                "adjusted_prob_min": prob_min,
                "adjusted_prob_max": prob_max,
                "adjusted_prob_span": prob_span,
                "mean_penalty": float(group["fragility_penalty"].mean()),
                "max_penalty": float(group["fragility_penalty"].max()),
                "high_fragility_rate": high_fragility_rate,
                "robust_pass_rate": robust_pass_rate,
                "negative_signal_rate": negative_signal_rate,
                "policy_sensitivity_class": sensitivity_class,
                "research_priority_score": priority_score,
                "recommended_next_action": _recommend_action(
                    support_eligible=support_eligible,
                    high_fragility_rate=high_fragility_rate,
                    negative_signal_rate=negative_signal_rate,
                    robust_pass_rate=robust_pass_rate,
                    sensitivity_class=sensitivity_class,
                ),
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values(["research_priority_score", "adjusted_prob_span"], ascending=[False, False])
        .reset_index(drop=True)
    )


def _select_policy_profiles(sweep: pd.DataFrame) -> pd.DataFrame:
    feasible = sweep[sweep["n_powered_robust_confirmed_pass"] >= 3].copy()
    if feasible.empty:
        feasible = sweep.copy()

    recommended = feasible[feasible["is_recommended"] == True].copy()
    if recommended.empty:
        recommended = feasible.sort_values(
            ["mean_adjusted_joint_prob", "mean_stability_multiplier", "mean_penalty"],
            ascending=[False, False, True],
        ).head(1)

    max_yield = feasible.sort_values(
        ["mean_adjusted_joint_prob", "mean_penalty"],
        ascending=[False, True],
    ).head(1)
    high_conservatism = feasible.sort_values(
        ["mean_penalty", "mean_adjusted_joint_prob"],
        ascending=[False, False],
    ).head(1)
    strict_negative = feasible.sort_values(
        [
            "n_underpowered_fragility_negative_signal",
            "mean_penalty",
            "mean_adjusted_joint_prob",
        ],
        ascending=[False, False, False],
    ).head(1)

    rows: List[Dict[str, object]] = []
    profile_specs = [
        ("balanced_recommended", "Balanced governance profile selected by sweep rule.", recommended),
        ("max_yield", "Maximizes adjusted confidence while preserving robust-pass count.", max_yield),
        ("high_conservatism", "Maximizes fragility penalties to reduce overclaim risk.", high_conservatism),
        ("strict_negative_signal", "Most willing to call underpowered negative signals.", strict_negative),
    ]

    for profile_name, rationale, df in profile_specs:
        if df.empty:
            continue
        row = df.iloc[0]
        rows.append(
            {
                "profile_name": profile_name,
                "rationale": rationale,
                "policy_id": str(row["policy_id"]),
                "weight_l2o": float(row["weight_l2o"]),
                "weight_l3o": float(row["weight_l3o"]),
                "tier_change_level_penalty": float(row["tier_change_level_penalty"]),
                "tier_change_rate_penalty": float(row["tier_change_rate_penalty"]),
                "mean_adjusted_joint_prob": float(row["mean_adjusted_joint_prob"]),
                "mean_penalty": float(row["mean_penalty"]),
                "n_powered_robust_confirmed_pass": int(row["n_powered_robust_confirmed_pass"]),
                "n_high_fragility_cohorts": int(row["n_high_fragility_cohorts"]),
                "n_underpowered_fragility_ambiguous": int(row["n_underpowered_fragility_ambiguous"]),
                "n_underpowered_fragility_negative_signal": int(row["n_underpowered_fragility_negative_signal"]),
            }
        )

    return pd.DataFrame(rows)


def _build_profile_assignments(profiles: pd.DataFrame, long_df: pd.DataFrame) -> pd.DataFrame:
    rows: List[pd.DataFrame] = []
    for profile in profiles.itertuples(index=False):
        cur = long_df[long_df["policy_id"] == str(profile.policy_id)].copy()
        cur["profile_name"] = str(profile.profile_name)
        rows.append(
            cur[
                [
                    "profile_name",
                    "policy_id",
                    "cohort",
                    "observed_support_eligible",
                    "fragility_adjusted_joint_prob",
                    "fragility_penalty",
                    "fragility_band",
                    "fragility_aware_endpoint_tier",
                ]
            ]
        )
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, axis=0, ignore_index=True).sort_values(["profile_name", "cohort"]).reset_index(drop=True)


def _plot_tier_heatmap(long_df: pd.DataFrame, sweep: pd.DataFrame, out_path: Path) -> None:
    tier_to_code = {
        "UNDERPOWERED_FRAGILITY_NEGATIVE_SIGNAL": 0,
        "UNDERPOWERED_FRAGILITY_AMBIGUOUS": 1,
        "POWERED_FRAGILITY_AMBIGUOUS": 2,
        "POWERED_FRAGILITY_SENSITIVE_PASS": 3,
        "POWERED_ROBUST_CONFIRMED_PASS": 4,
    }
    policy_order = (
        sweep.sort_values(["mean_penalty", "mean_adjusted_joint_prob"], ascending=[True, False])["policy_id"]
        .astype(str)
        .tolist()
    )
    cohorts = sorted(long_df["cohort"].unique().tolist())

    pivot = (
        long_df.assign(tier_code=long_df["fragility_aware_endpoint_tier"].map(tier_to_code).fillna(1.5))
        .pivot_table(index="cohort", columns="policy_id", values="tier_code", aggfunc="first")
        .reindex(index=cohorts, columns=policy_order)
    )
    arr = pivot.to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(14, 4.4))
    im = ax.imshow(arr, aspect="auto", cmap="viridis", vmin=0, vmax=4)
    ax.set_yticks(np.arange(len(cohorts)))
    ax.set_yticklabels(cohorts)
    xticks = np.arange(0, len(policy_order), 10)
    ax.set_xticks(xticks)
    ax.set_xticklabels([policy_order[i] for i in xticks], rotation=90, fontsize=7)
    ax.set_xlabel("Policy profile (sorted by mean penalty)")
    ax.set_ylabel("Cohort")
    ax.set_title("Fragility Tier Sensitivity Across Policy Profiles")
    cbar = plt.colorbar(im, ax=ax, shrink=0.95)
    cbar.set_label("Tier code (0=underpowered negative, 4=powered robust pass)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_sensitivity_frontier(cohort_summary: pd.DataFrame, out_path: Path) -> None:
    if cohort_summary.empty:
        return
    fig, ax = plt.subplots(figsize=(8.4, 5.0))
    colors = cohort_summary["observed_support_eligible"].map({True: "#1b9e77", False: "#d95f02"}).fillna("#7570b3")
    sizes = 120 + 450 * cohort_summary["research_priority_score"].to_numpy(dtype=float)
    ax.scatter(
        cohort_summary["modal_tier_fraction"],
        cohort_summary["adjusted_prob_span"],
        c=colors.tolist(),
        s=sizes,
        alpha=0.85,
        edgecolors="white",
        linewidths=0.9,
    )
    for row in cohort_summary.itertuples(index=False):
        ax.annotate(str(row.cohort), (float(row.modal_tier_fraction), float(row.adjusted_prob_span)), fontsize=8)

    ax.set_xlim(0.45, 1.02)
    ax.set_ylim(-0.005, max(0.30, float(cohort_summary["adjusted_prob_span"].max()) + 0.03))
    ax.set_xlabel("Modal tier fraction across policies (higher = more stable)")
    ax.set_ylabel("Adjusted probability span across policies")
    ax.set_title("Cohort Policy Sensitivity Frontier")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = _parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    sweep = _read_csv(args.sweep_summary).sort_values("policy_id").reset_index(drop=True)
    long_df = _load_policy_long(sweep=sweep, runs_dir=args.runs_dir)
    long_df.to_csv(args.output_dir / "policy_sensitivity_long.csv", index=False)

    cohort_summary = _compute_cohort_summary(long_df)
    cohort_summary.to_csv(args.output_dir / "cohort_policy_sensitivity_summary.csv", index=False)
    cohort_summary.to_csv(args.output_dir / "strategic_research_priorities.csv", index=False)

    profiles = _select_policy_profiles(sweep)
    profiles.to_csv(args.output_dir / "strategic_policy_profiles.csv", index=False)

    assignments = _build_profile_assignments(profiles=profiles, long_df=long_df)
    assignments.to_csv(args.output_dir / "strategic_profile_cohort_assignments.csv", index=False)

    _plot_tier_heatmap(
        long_df=long_df,
        sweep=sweep,
        out_path=args.output_dir / "policy_tier_heatmap.png",
    )
    _plot_sensitivity_frontier(
        cohort_summary=cohort_summary,
        out_path=args.output_dir / "cohort_sensitivity_frontier.png",
    )

    manifest = {
        "n_policies": int(sweep["policy_id"].nunique()),
        "n_cohorts": int(long_df["cohort"].nunique()),
        "n_policy_rows": int(len(long_df)),
        "recommended_policy_id": str(
            sweep.loc[sweep["is_recommended"] == True, "policy_id"].iloc[0]
            if (sweep["is_recommended"] == True).any()
            else profiles.iloc[0]["policy_id"]
        ),
        "top_priority_cohort": str(cohort_summary.iloc[0]["cohort"]) if not cohort_summary.empty else "",
    }
    (args.output_dir / "policy_sensitivity_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    print("Fragility policy sensitivity audit complete.")
    print(f"Policies analyzed: {manifest['n_policies']}")
    print(f"Cohorts analyzed: {manifest['n_cohorts']}")
    print(f"Top-priority cohort: {manifest['top_priority_cohort']}")


if __name__ == "__main__":
    main()
