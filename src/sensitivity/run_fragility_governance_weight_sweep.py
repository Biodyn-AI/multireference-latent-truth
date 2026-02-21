#!/usr/bin/env python3
"""Run strategic weight/penalty sweep for fragility-aware endpoint governance.

Phase XXXII:
1) Sweep governance-policy weights for LOCO/L2O/L3O fragility and tier-change penalties.
2) Build an operating frontier over robustness-vs-yield tradeoffs.
3) Recommend a conservative strategic profile for manuscript-level governance.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--governance-script",
        type=Path,
        default=Path(
            "subproject_24_multireference_latent_truth_model/implementation/"
            "run_fragility_aware_endpoint_governance.py"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "subproject_24_multireference_latent_truth_model/implementation/outputs/"
            "top_tier_readiness_v7/fragility_aware_weight_sweep_v1"
        ),
    )
    return parser.parse_args()


def _run(cmd: List[str]) -> None:
    proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"Command failed ({proc.returncode}): {' '.join(cmd)}\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
        )


def _profiles() -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    policy_id = 0

    # Strategic grid: stress later-order fragility progressively harder.
    for w2 in (1.0, 1.25, 1.5):
        for w3 in (1.25, 1.5, 1.75, 2.0):
            for lvl_pen in (0.06, 0.08, 0.10):
                for rate_pen in (0.20, 0.25, 0.30):
                    if w3 < w2:
                        continue
                    rows.append(
                        {
                            "policy_id": f"policy_{policy_id:03d}",
                            "weight_loo": 1.0,
                            "weight_l2o": float(w2),
                            "weight_l3o": float(w3),
                            "tier_change_level_penalty": float(lvl_pen),
                            "tier_change_rate_penalty": float(rate_pen),
                        }
                    )
                    policy_id += 1
    return rows


def _plot_frontier(df: pd.DataFrame, out_path: Path) -> None:
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(9.7, 5.4))

    # x: conservatism (penalty), y: robust powered yield.
    scatter = ax.scatter(
        df["mean_penalty"].to_numpy(dtype=float),
        df["n_powered_robust_confirmed_pass"].to_numpy(dtype=float),
        c=df["mean_adjusted_joint_prob"].to_numpy(dtype=float),
        cmap="viridis",
        s=80,
        edgecolors="white",
        linewidths=0.7,
    )
    best = df[df["is_recommended"] == True]
    if not best.empty:
        r = best.iloc[0]
        ax.scatter(
            [float(r["mean_penalty"])],
            [float(r["n_powered_robust_confirmed_pass"])],
            s=200,
            marker="*",
            color="#d7301f",
            edgecolors="black",
            linewidths=0.8,
            zorder=4,
        )
        ax.annotate(str(r["policy_id"]), (float(r["mean_penalty"]), float(r["n_powered_robust_confirmed_pass"])), fontsize=8)

    ax.set_xlabel("Mean Fragility Penalty (higher = more conservative)")
    ax.set_ylabel("N Powered Robust Confirmed Pass")
    ax.set_title("Fragility-Governance Operating Frontier")
    ax.grid(alpha=0.25)
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Mean Adjusted Joint Probability")
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = _parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    profiles = _profiles()
    (args.output_dir / "policy_sweep_manifest.json").write_text(json.dumps(profiles, indent=2), encoding="utf-8")

    run_rows: List[Dict[str, object]] = []

    for p in profiles:
        policy_id = str(p["policy_id"])
        run_dir = args.output_dir / "runs" / policy_id
        run_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            "python",
            str(args.governance_script),
            "--weight-loo",
            str(p["weight_loo"]),
            "--weight-l2o",
            str(p["weight_l2o"]),
            "--weight-l3o",
            str(p["weight_l3o"]),
            "--tier-change-level-penalty",
            str(p["tier_change_level_penalty"]),
            "--tier-change-rate-penalty",
            str(p["tier_change_rate_penalty"]),
            "--output-dir",
            str(run_dir),
        ]
        _run(cmd)

        overall = pd.read_csv(run_dir / "fragility_aware_overall_summary.csv", low_memory=False).iloc[0]
        tiers = pd.read_csv(run_dir / "fragility_aware_tier_summary.csv", low_memory=False)
        rows = pd.read_csv(run_dir / "fragility_aware_endpoint_rows.csv", low_memory=False)

        def _count(tier_name: str) -> int:
            sel = tiers[tiers["fragility_aware_endpoint_tier"] == tier_name]
            if sel.empty:
                return 0
            return int(sel["n_cohorts"].iloc[0])

        run_rows.append(
            {
                "policy_id": policy_id,
                "weight_loo": float(p["weight_loo"]),
                "weight_l2o": float(p["weight_l2o"]),
                "weight_l3o": float(p["weight_l3o"]),
                "tier_change_level_penalty": float(p["tier_change_level_penalty"]),
                "tier_change_rate_penalty": float(p["tier_change_rate_penalty"]),
                "mean_base_joint_prob": float(overall["mean_base_joint_prob"]),
                "mean_adjusted_joint_prob": float(overall["mean_adjusted_joint_prob"]),
                "mean_stability_multiplier": float(overall["mean_stability_multiplier"]),
                "max_abs_fragility_drift": float(overall["max_abs_fragility_drift"]),
                "n_high_fragility_cohorts": int(overall["n_high_fragility_cohorts"]),
                "n_moderate_fragility_cohorts": int(overall["n_moderate_fragility_cohorts"]),
                "n_low_fragility_cohorts": int(overall["n_low_fragility_cohorts"]),
                "n_powered_robust_confirmed_pass": _count("POWERED_ROBUST_CONFIRMED_PASS"),
                "n_powered_fragility_ambiguous": _count("POWERED_FRAGILITY_AMBIGUOUS"),
                "n_underpowered_fragility_ambiguous": _count("UNDERPOWERED_FRAGILITY_AMBIGUOUS"),
                "n_underpowered_fragility_negative_signal": _count("UNDERPOWERED_FRAGILITY_NEGATIVE_SIGNAL"),
                "mean_penalty": float(rows["fragility_penalty"].mean()),
                "max_penalty": float(rows["fragility_penalty"].max()),
            }
        )

    sweep = pd.DataFrame(run_rows).sort_values("policy_id").reset_index(drop=True)

    # Strategic recommendation rule:
    # 1) Keep powered robust passes >= 3.
    # 2) Keep high fragility cohorts <= 2.
    # 3) Among feasible, maximize mean adjusted confidence then stability multiplier.
    feasible = sweep[
        (sweep["n_powered_robust_confirmed_pass"] >= 3)
        & (sweep["n_high_fragility_cohorts"] <= 2)
    ].copy()
    if feasible.empty:
        feasible = sweep.copy()

    feasible = feasible.sort_values(
        ["mean_adjusted_joint_prob", "mean_stability_multiplier", "mean_penalty"],
        ascending=[False, False, True],
    ).reset_index(drop=True)
    recommended_id = str(feasible.iloc[0]["policy_id"])
    sweep["is_recommended"] = sweep["policy_id"] == recommended_id
    sweep.to_csv(args.output_dir / "policy_sweep_summary.csv", index=False)

    recommended = sweep[sweep["policy_id"] == recommended_id].copy()
    recommended.to_csv(args.output_dir / "policy_recommended_profile.csv", index=False)

    top_table = (
        sweep.sort_values(
            ["mean_adjusted_joint_prob", "n_powered_robust_confirmed_pass", "mean_penalty"],
            ascending=[False, False, True],
        )
        .head(10)
        .reset_index(drop=True)
    )
    top_table.to_csv(args.output_dir / "policy_top10_profiles.csv", index=False)

    _plot_frontier(sweep, args.output_dir / "policy_operating_frontier.png")

    print("Fragility governance weight sweep complete.")
    print(f"Profiles evaluated: {len(sweep)}")
    print(f"Recommended policy: {recommended_id}")


if __name__ == "__main__":
    main()
