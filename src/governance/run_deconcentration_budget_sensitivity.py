#!/usr/bin/env python3
"""Run budget sensitivity sweep for deconcentration governance policies.

For each budget, runs:
1) deconcentration policy sweep,
2) perturbation validation baseline vs recommended policy,
3) aggregate summary table and figure.
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import pandas as pd


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--budgets", type=str, default="300,500,700,842")
    parser.add_argument("--alphas", type=str, default="0.0,0.05,0.1,0.25,0.5,1.0")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "subproject_24_multireference_latent_truth_model/implementation/outputs/deconcentration_budget_sensitivity_v1"
        ),
    )
    return parser.parse_args()


def _run(cmd: List[str]) -> None:
    proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"Command failed ({proc.returncode}): {' '.join(cmd)}\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
        )


def _select_path_from_policy_dir(policy_dir: Path, policy_name: str) -> Path:
    if policy_name == "baseline_alpha0":
        return policy_dir / "baseline_alpha0_selected_edges.csv"
    return policy_dir / f"{policy_name}_selected_edges.csv"


def main() -> None:
    args = _parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    budgets = [int(x.strip()) for x in args.budgets.split(",") if x.strip()]
    if not budgets:
        raise ValueError("No budgets provided")

    rows: List[Dict[str, object]] = []

    for budget in budgets:
        budget_dir = args.output_dir / f"budget_{budget}"
        policy_dir = budget_dir / "policy"
        validation_dir = budget_dir / "validation"
        policy_dir.mkdir(parents=True, exist_ok=True)
        validation_dir.mkdir(parents=True, exist_ok=True)

        _run(
            [
                "python",
                "subproject_24_multireference_latent_truth_model/implementation/run_deconcentration_policy_sweep.py",
                "--budget",
                str(budget),
                "--alphas",
                args.alphas,
                "--output-dir",
                str(policy_dir),
            ]
        )

        policy_summary = pd.read_csv(policy_dir / "deconcentration_policy_summary.csv")
        policy_reco = pd.read_csv(policy_dir / "deconcentration_policy_recommendation.csv")

        baseline = policy_summary[policy_summary["policy"] == "baseline_alpha0"].iloc[0]
        reco = policy_reco.iloc[0]
        reco_policy = str(reco["policy"])

        baseline_sel = _select_path_from_policy_dir(policy_dir, "baseline_alpha0")
        reco_sel = _select_path_from_policy_dir(policy_dir, reco_policy)

        _run(
            [
                "python",
                "subproject_24_multireference_latent_truth_model/implementation/evaluate_deconcentration_perturbation_validation.py",
                "--selections",
                f"baseline={baseline_sel},recommended={reco_sel}",
                "--output-dir",
                str(validation_dir),
            ]
        )

        val_summary = pd.read_csv(validation_dir / "deconcentration_validation_summary.csv")

        def _get_hit(selection: str, ref_set: str, col: str) -> float:
            rr = val_summary[(val_summary["selection"] == selection) & (val_summary["reference_set"] == ref_set)]
            if rr.empty:
                return float("nan")
            return float(rr.iloc[0][col])

        rows.append(
            {
                "budget": int(budget),
                "recommended_policy": reco_policy,
                "recommended_alpha": float(reco["alpha"]),
                "baseline_top1_source_fraction": float(baseline["top1_source_fraction"]),
                "recommended_top1_source_fraction": float(reco["top1_source_fraction"]),
                "delta_top1_source_fraction": float(reco["top1_source_fraction"] - baseline["top1_source_fraction"]),
                "baseline_top2_source_fraction": float(baseline["top2_source_fraction"]),
                "recommended_top2_source_fraction": float(reco["top2_source_fraction"]),
                "delta_top2_source_fraction": float(reco["top2_source_fraction"] - baseline["top2_source_fraction"]),
                "baseline_joint_score_mean": float(baseline["joint_score_mean"]),
                "recommended_joint_score_mean": float(reco["joint_score_mean"]),
                "delta_joint_score_mean": float(reco["joint_score_mean"] - baseline["joint_score_mean"]),
                "baseline_ref_context_fraction": float(baseline["reference_context_fraction"]),
                "recommended_ref_context_fraction": float(reco["reference_context_fraction"]),
                "delta_ref_context_fraction": float(reco["reference_context_fraction"] - baseline["reference_context_fraction"]),
                "baseline_hits_high_overlap_union": _get_hit("baseline", "high_overlap_union", "n_hits"),
                "recommended_hits_high_overlap_union": _get_hit("recommended", "high_overlap_union", "n_hits"),
                "delta_hits_high_overlap_union": _get_hit("recommended", "high_overlap_union", "n_hits")
                - _get_hit("baseline", "high_overlap_union", "n_hits"),
                "baseline_hits_low_overlap_union": _get_hit("baseline", "low_overlap_union", "n_hits"),
                "recommended_hits_low_overlap_union": _get_hit("recommended", "low_overlap_union", "n_hits"),
                "delta_hits_low_overlap_union": _get_hit("recommended", "low_overlap_union", "n_hits")
                - _get_hit("baseline", "low_overlap_union", "n_hits"),
            }
        )

    out = pd.DataFrame(rows).sort_values("budget").reset_index(drop=True)
    out.to_csv(args.output_dir / "budget_sensitivity_summary.csv", index=False)

    fig, ax = plt.subplots(figsize=(8.8, 5.0))
    ax.plot(out["budget"], out["baseline_top1_source_fraction"], marker="o", label="baseline top1 fraction")
    ax.plot(out["budget"], out["recommended_top1_source_fraction"], marker="o", label="recommended top1 fraction")
    ax.set_xlabel("Promotion budget")
    ax.set_ylabel("Top-1 source fraction")
    ax.set_title("Budget Sensitivity of Deconcentration Policy")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(args.output_dir / "budget_sensitivity_top1_fraction.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    print("Budget sensitivity sweep complete.")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
