#!/usr/bin/env python3
"""Run prior-sensitivity and posterior-predictive stress suite for multilevel endpoint model."""

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
        "--model-script",
        type=Path,
        default=Path(
            "subproject_24_multireference_latent_truth_model/implementation/"
            "compute_multilevel_adaptive_bayes_endpoint_tiers.py"
        ),
    )
    parser.add_argument("--mcmc-steps", type=int, default=22000)
    parser.add_argument("--burn-in", type=int, default=5000)
    parser.add_argument("--thin", type=int, default=6)
    parser.add_argument("--base-seed", type=int, default=131)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "subproject_24_multireference_latent_truth_model/implementation/outputs/"
            "top_tier_readiness_v7/multilevel_prior_sensitivity_v1"
        ),
    )
    return parser.parse_args()


def _run(cmd: List[str]) -> None:
    proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"Command failed ({proc.returncode}): {' '.join(cmd)}\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
        )


def _profile_grid() -> List[Dict[str, object]]:
    return [
        {
            "profile_id": "baseline",
            "prior_sd_beta": 2.5,
            "prior_logsigma_mean": -0.5,
            "prior_logsigma_sd": 0.8,
            "prior_logtau_mean": -0.7,
            "prior_logtau_sd": 0.9,
        },
        {
            "profile_id": "tight_beta",
            "prior_sd_beta": 1.6,
            "prior_logsigma_mean": -0.5,
            "prior_logsigma_sd": 0.8,
            "prior_logtau_mean": -0.7,
            "prior_logtau_sd": 0.9,
        },
        {
            "profile_id": "diffuse_beta",
            "prior_sd_beta": 4.0,
            "prior_logsigma_mean": -0.5,
            "prior_logsigma_sd": 0.8,
            "prior_logtau_mean": -0.7,
            "prior_logtau_sd": 0.9,
        },
        {
            "profile_id": "tight_scales",
            "prior_sd_beta": 2.5,
            "prior_logsigma_mean": -0.7,
            "prior_logsigma_sd": 0.5,
            "prior_logtau_mean": -0.9,
            "prior_logtau_sd": 0.55,
        },
        {
            "profile_id": "diffuse_scales",
            "prior_sd_beta": 2.5,
            "prior_logsigma_mean": -0.4,
            "prior_logsigma_sd": 1.2,
            "prior_logtau_mean": -0.6,
            "prior_logtau_sd": 1.2,
        },
    ]


def _read_one(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing expected output file: {path}")
    return pd.read_csv(path, low_memory=False)


def _plot_profile_cohort_probs(df: pd.DataFrame, out_path: Path) -> None:
    if df.empty:
        return
    pivot = (
        df.pivot_table(
            index="cohort",
            columns="profile_id",
            values="predictive_prob_joint_pass",
            aggfunc="first",
        )
        .sort_index()
    )
    fig, ax = plt.subplots(figsize=(10.2, 5.2))
    x = np.arange(len(pivot.index))
    for profile in pivot.columns:
        ax.plot(x, pivot[profile].to_numpy(dtype=float), marker="o", linewidth=1.4, label=str(profile))
    ax.set_xticks(x)
    ax.set_xticklabels(pivot.index.tolist(), rotation=20, ha="right")
    ax.set_ylabel("Predictive P(joint endpoint pass)")
    ax.set_title("Prior Sensitivity of Multilevel Joint Endpoint Probability")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(alpha=0.25)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = _parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    profiles = _profile_grid()
    (args.output_dir / "sensitivity_profile_manifest.json").write_text(
        json.dumps(profiles, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    all_rows: List[pd.DataFrame] = []
    run_summary_rows: List[Dict[str, object]] = []
    ppc_rows: List[pd.DataFrame] = []
    tier_rows: List[pd.DataFrame] = []

    for idx, profile in enumerate(profiles):
        profile_id = str(profile["profile_id"])
        run_dir = args.output_dir / "runs" / profile_id
        run_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            "python",
            str(args.model_script),
            "--mcmc-steps",
            str(args.mcmc_steps),
            "--burn-in",
            str(args.burn_in),
            "--thin",
            str(args.thin),
            "--seed",
            str(int(args.base_seed + idx)),
            "--prior-sd-beta",
            str(profile["prior_sd_beta"]),
            "--prior-logsigma-mean",
            str(profile["prior_logsigma_mean"]),
            "--prior-logsigma-sd",
            str(profile["prior_logsigma_sd"]),
            "--prior-logtau-mean",
            str(profile["prior_logtau_mean"]),
            "--prior-logtau-sd",
            str(profile["prior_logtau_sd"]),
            "--output-dir",
            str(run_dir),
        ]
        _run(cmd)

        rows = _read_one(run_dir / "multilevel_adaptive_endpoint_rows.csv")
        rows["profile_id"] = profile_id
        rows["prior_sd_beta"] = float(profile["prior_sd_beta"])
        rows["prior_logsigma_mean"] = float(profile["prior_logsigma_mean"])
        rows["prior_logsigma_sd"] = float(profile["prior_logsigma_sd"])
        rows["prior_logtau_mean"] = float(profile["prior_logtau_mean"])
        rows["prior_logtau_sd"] = float(profile["prior_logtau_sd"])
        all_rows.append(rows)

        overall = _read_one(run_dir / "multilevel_adaptive_overall_summary.csv").iloc[0]
        support = _read_one(run_dir / "multilevel_adaptive_support_summary.csv")
        ppc = _read_one(run_dir / "multilevel_adaptive_ppc_summary.csv")
        diag = _read_one(run_dir / "multilevel_adaptive_mcmc_diagnostics.csv").iloc[0]

        ppc["profile_id"] = profile_id
        ppc_rows.append(ppc)

        tier = _read_one(run_dir / "multilevel_adaptive_tier_summary.csv")
        tier["profile_id"] = profile_id
        tier_rows.append(tier)

        support_eligible_row = support[support["observed_support_eligible"] == True]
        support_limited_row = support[support["observed_support_eligible"] == False]

        run_summary_rows.append(
            {
                "profile_id": profile_id,
                "prior_sd_beta": float(profile["prior_sd_beta"]),
                "prior_logsigma_mean": float(profile["prior_logsigma_mean"]),
                "prior_logsigma_sd": float(profile["prior_logsigma_sd"]),
                "prior_logtau_mean": float(profile["prior_logtau_mean"]),
                "prior_logtau_sd": float(profile["prior_logtau_sd"]),
                "n_cohorts": int(overall["n_cohorts"]),
                "observed_pass_rate": float(overall["observed_pass_rate"]),
                "n_powered_confirmed_pass": int(overall["n_powered_confirmed_pass"]),
                "n_underpowered_negative_signal": int(overall["n_underpowered_negative_signal"]),
                "support_eligible_mean_joint_prob": float(
                    support_eligible_row["mean_predictive_prob_joint_pass"].iloc[0]
                    if not support_eligible_row.empty
                    else np.nan
                ),
                "support_limited_mean_joint_prob": float(
                    support_limited_row["mean_predictive_prob_joint_pass"].iloc[0]
                    if not support_limited_row.empty
                    else np.nan
                ),
                "retained_draws": int(diag["retained_draws"]),
                "eta_acceptance_rate": float(diag["eta_acceptance_rate"]),
                "beta_acceptance_rate": float(diag["beta_acceptance_rate"]),
                "family_acceptance_rate": float(diag["family_acceptance_rate"]),
                "logsigma_acceptance_rate": float(diag["logsigma_acceptance_rate"]),
                "logtau_acceptance_rate": float(diag["logtau_acceptance_rate"]),
            }
        )

    all_out = pd.concat(all_rows, ignore_index=True)
    all_out.to_csv(args.output_dir / "sensitivity_all_endpoint_rows.csv", index=False)

    run_summary = pd.DataFrame(run_summary_rows).sort_values("profile_id").reset_index(drop=True)
    run_summary.to_csv(args.output_dir / "sensitivity_run_summary.csv", index=False)

    cohort_var = (
        all_out.groupby(["cohort"], as_index=False)
        .agg(
            mean_joint_prob=("predictive_prob_joint_pass", "mean"),
            sd_joint_prob=("predictive_prob_joint_pass", "std"),
            min_joint_prob=("predictive_prob_joint_pass", "min"),
            max_joint_prob=("predictive_prob_joint_pass", "max"),
            n_distinct_tiers=("multilevel_adaptive_joint_tier", lambda x: int(pd.Series(x).nunique())),
        )
        .sort_values("cohort")
        .reset_index(drop=True)
    )
    cohort_var.to_csv(args.output_dir / "sensitivity_cohort_joint_probability_variance.csv", index=False)

    tier_stability = (
        all_out.groupby(["cohort", "multilevel_adaptive_joint_tier"], as_index=False)
        .agg(n_profiles=("profile_id", "nunique"))
        .sort_values(["cohort", "multilevel_adaptive_joint_tier"])
        .reset_index(drop=True)
    )
    tier_stability.to_csv(args.output_dir / "sensitivity_tier_stability.csv", index=False)

    ppc_all = pd.concat(ppc_rows, ignore_index=True)
    ppc_all.to_csv(args.output_dir / "sensitivity_ppc_all_rows.csv", index=False)
    ppc_summary = (
        ppc_all.groupby("metric", as_index=False)
        .agg(
            mean_tail_prob_ge=("tail_prob_ge_observed", "mean"),
            min_tail_prob_ge=("tail_prob_ge_observed", "min"),
            max_tail_prob_ge=("tail_prob_ge_observed", "max"),
            mean_tail_prob_le=("tail_prob_le_observed", "mean"),
            min_tail_prob_le=("tail_prob_le_observed", "min"),
            max_tail_prob_le=("tail_prob_le_observed", "max"),
        )
        .sort_values("metric")
        .reset_index(drop=True)
    )
    ppc_summary.to_csv(args.output_dir / "posterior_predictive_stress_summary.csv", index=False)

    tier_all = pd.concat(tier_rows, ignore_index=True)
    tier_all.to_csv(args.output_dir / "sensitivity_tier_summary_all_profiles.csv", index=False)

    _plot_profile_cohort_probs(
        df=all_out[["cohort", "profile_id", "predictive_prob_joint_pass"]].copy(),
        out_path=args.output_dir / "sensitivity_cohort_joint_probability_profiles.png",
    )

    print("Multilevel prior-sensitivity suite complete.")
    print(f"Profiles executed: {len(profiles)}")
    print(f"Output directory: {args.output_dir}")


if __name__ == "__main__":
    main()
