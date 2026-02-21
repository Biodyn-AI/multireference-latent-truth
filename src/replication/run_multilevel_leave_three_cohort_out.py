#!/usr/bin/env python3
"""Run leave-three-cohort-out stress analysis for the multilevel adaptive endpoint model.

Phase XXX:
1) Refit the multilevel model while omitting each cohort triplet.
2) Quantify third-order endpoint fragility relative to the full fit.
3) Export triplet-level and cohort-level influence summaries.
"""

from __future__ import annotations

import argparse
import itertools
import json
import subprocess
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

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
    parser.add_argument(
        "--results",
        type=Path,
        default=Path(
            "subproject_24_multireference_latent_truth_model/implementation/outputs/"
            "top_tier_readiness_v6/multicohort_preregistered_replication_v1/"
            "multicohort_replication_results.csv"
        ),
    )
    parser.add_argument(
        "--replication-root",
        type=Path,
        default=Path(
            "subproject_24_multireference_latent_truth_model/implementation/outputs/"
            "top_tier_readiness_v6/multicohort_preregistered_replication_v1"
        ),
    )
    parser.add_argument(
        "--full-fit-rows",
        type=Path,
        default=Path(
            "subproject_24_multireference_latent_truth_model/implementation/outputs/"
            "top_tier_readiness_v7/multilevel_adaptive_bayes_endpoint_tiering_v1/"
            "multilevel_adaptive_endpoint_rows.csv"
        ),
    )
    parser.add_argument("--mcmc-steps", type=int, default=14000)
    parser.add_argument("--burn-in", type=int, default=3000)
    parser.add_argument("--thin", type=int, default=4)
    parser.add_argument("--base-seed", type=int, default=701)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "subproject_24_multireference_latent_truth_model/implementation/outputs/"
            "top_tier_readiness_v7/multilevel_leave_three_cohort_out_v1"
        ),
    )
    return parser.parse_args()


def _run_command(cmd: Sequence[str]) -> None:
    proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"Command failed ({proc.returncode}): {' '.join(cmd)}\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
        )


def _completed_cohorts(results_path: Path) -> List[str]:
    df = pd.read_csv(results_path, low_memory=False)
    done = df[df["status"] == "completed"].copy()
    if done.empty:
        raise ValueError(f"No completed rows in {results_path}")
    return sorted(done["cohort"].astype(str).unique().tolist())


def _ensure_full_fit(args: argparse.Namespace) -> pd.DataFrame:
    if args.full_fit_rows.exists():
        return pd.read_csv(args.full_fit_rows, low_memory=False)

    full_dir = args.output_dir / "full_fit_recomputed"
    full_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "python",
        str(args.model_script),
        "--results",
        str(args.results),
        "--replication-root",
        str(args.replication_root),
        "--mcmc-steps",
        str(args.mcmc_steps),
        "--burn-in",
        str(args.burn_in),
        "--thin",
        str(args.thin),
        "--seed",
        str(int(args.base_seed)),
        "--output-dir",
        str(full_dir),
    ]
    _run_command(cmd)
    out_rows = full_dir / "multilevel_adaptive_endpoint_rows.csv"
    if not out_rows.exists():
        raise FileNotFoundError(f"Missing expected full-fit output: {out_rows}")
    return pd.read_csv(out_rows, low_memory=False)


def _triplet_label(triplet: Tuple[str, str, str]) -> str:
    return f"{triplet[0]}|{triplet[1]}|{triplet[2]}"


def _plot_fragility(df: pd.DataFrame, out_path: Path) -> None:
    if df.empty:
        return
    ordered = df.sort_values("full_predictive_prob_joint_pass", ascending=False).reset_index(drop=True)
    x = np.arange(len(ordered))
    y = ordered["full_predictive_prob_joint_pass"].to_numpy(dtype=float)
    low = np.maximum(0.0, y - ordered["l3o_min_predictive_prob_joint_pass"].to_numpy(dtype=float))
    high = np.maximum(0.0, ordered["l3o_max_predictive_prob_joint_pass"].to_numpy(dtype=float) - y)

    fig, ax = plt.subplots(figsize=(10.8, 5.5))
    ax.errorbar(
        x,
        y,
        yerr=np.vstack([low, high]),
        fmt="o",
        color="#00441b",
        ecolor="#00441b",
        elinewidth=1.6,
        capsize=3,
        markersize=6,
    )
    for i, row in ordered.iterrows():
        ax.annotate(str(row["cohort"]), (x[i], y[i]), fontsize=8, rotation=20, ha="left")

    ax.axhline(0.95, color="#2166ac", linestyle="--", linewidth=1.0, alpha=0.8)
    ax.axhline(0.05, color="#b2182b", linestyle="--", linewidth=1.0, alpha=0.8)
    ax.set_xticks([])
    ax.set_ylim(-0.02, 1.02)
    ax.set_ylabel("P(joint endpoint pass)")
    ax.set_title("Leave-Three-Cohort-Out Fragility Around Full-Fit Endpoint Probability")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = _parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    cohorts = _completed_cohorts(args.results)
    if len(cohorts) < 4:
        raise ValueError("Need at least 4 completed cohorts for leave-three analysis.")
    triplets = list(itertools.combinations(cohorts, 3))

    manifest = {
        "phase": "XXX",
        "cohorts": cohorts,
        "n_cohorts": len(cohorts),
        "n_leave_three_runs": len(triplets),
        "model_script": str(args.model_script),
        "results": str(args.results),
        "replication_root": str(args.replication_root),
        "mcmc_steps": int(args.mcmc_steps),
        "burn_in": int(args.burn_in),
        "thin": int(args.thin),
        "base_seed": int(args.base_seed),
    }
    (args.output_dir / "leave_three_cohort_out_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    full_rows = _ensure_full_fit(args).copy().sort_values("cohort").reset_index(drop=True)
    full_rows.to_csv(args.output_dir / "full_fit_endpoint_rows.csv", index=False)

    run_rows: List[pd.DataFrame] = []
    influence_rows: List[Dict[str, object]] = []
    runs_root = args.output_dir / "runs"
    runs_root.mkdir(parents=True, exist_ok=True)

    base_results = pd.read_csv(args.results, low_memory=False)
    for idx, triplet in enumerate(triplets):
        label = _triplet_label(triplet)
        run_dir = runs_root / f"omit_{triplet[0]}__{triplet[1]}__{triplet[2]}"
        run_dir.mkdir(parents=True, exist_ok=True)
        run_out = run_dir / "multilevel_adaptive_endpoint_rows.csv"

        if not run_out.exists():
            filtered = base_results[~base_results["cohort"].astype(str).isin(triplet)].copy()
            filtered_results = run_dir / "results_leave_three_out.csv"
            filtered.to_csv(filtered_results, index=False)

            cmd = [
                "python",
                str(args.model_script),
                "--results",
                str(filtered_results),
                "--replication-root",
                str(args.replication_root),
                "--mcmc-steps",
                str(args.mcmc_steps),
                "--burn-in",
                str(args.burn_in),
                "--thin",
                str(args.thin),
                "--seed",
                str(int(args.base_seed + idx + 1)),
                "--output-dir",
                str(run_dir),
            ]
            _run_command(cmd)

        if not run_out.exists():
            raise FileNotFoundError(f"Missing expected run output: {run_out}")

        one = pd.read_csv(run_out, low_memory=False)
        one["omitted_triplet"] = label
        one["omitted_cohort_a"] = triplet[0]
        one["omitted_cohort_b"] = triplet[1]
        one["omitted_cohort_c"] = triplet[2]
        run_rows.append(one)

        merged = one.merge(
            full_rows[["cohort", "predictive_prob_joint_pass", "multilevel_adaptive_joint_tier"]].rename(
                columns={
                    "predictive_prob_joint_pass": "full_predictive_prob_joint_pass",
                    "multilevel_adaptive_joint_tier": "full_multilevel_adaptive_joint_tier",
                }
            ),
            on="cohort",
            how="left",
        )
        merged["delta_predictive_prob_joint_pass"] = (
            merged["predictive_prob_joint_pass"] - merged["full_predictive_prob_joint_pass"]
        )
        merged["tier_changed_vs_full"] = (
            merged["multilevel_adaptive_joint_tier"] != merged["full_multilevel_adaptive_joint_tier"]
        )

        for row in merged.itertuples(index=False):
            influence_rows.append(
                {
                    "omitted_triplet": label,
                    "omitted_cohort_a": triplet[0],
                    "omitted_cohort_b": triplet[1],
                    "omitted_cohort_c": triplet[2],
                    "target_cohort": str(row.cohort),
                    "full_predictive_prob_joint_pass": float(row.full_predictive_prob_joint_pass),
                    "l3o_predictive_prob_joint_pass": float(row.predictive_prob_joint_pass),
                    "delta_predictive_prob_joint_pass": float(row.delta_predictive_prob_joint_pass),
                    "abs_delta_predictive_prob_joint_pass": float(abs(row.delta_predictive_prob_joint_pass)),
                    "full_tier": str(row.full_multilevel_adaptive_joint_tier),
                    "l3o_tier": str(row.multilevel_adaptive_joint_tier),
                    "tier_changed_vs_full": bool(row.tier_changed_vs_full),
                }
            )

    all_runs = pd.concat(run_rows, ignore_index=True)
    all_runs.to_csv(args.output_dir / "leave_three_cohort_all_rows.csv", index=False)

    influence = pd.DataFrame(influence_rows)
    influence.to_csv(args.output_dir / "leave_three_cohort_influence_matrix.csv", index=False)

    cohort_fragility = (
        influence.groupby("target_cohort", as_index=False)
        .agg(
            full_predictive_prob_joint_pass=("full_predictive_prob_joint_pass", "first"),
            l3o_mean_predictive_prob_joint_pass=("l3o_predictive_prob_joint_pass", "mean"),
            l3o_sd_predictive_prob_joint_pass=("l3o_predictive_prob_joint_pass", "std"),
            l3o_min_predictive_prob_joint_pass=("l3o_predictive_prob_joint_pass", "min"),
            l3o_max_predictive_prob_joint_pass=("l3o_predictive_prob_joint_pass", "max"),
            l3o_mean_abs_delta_predictive_prob_joint_pass=("abs_delta_predictive_prob_joint_pass", "mean"),
            l3o_max_abs_delta_predictive_prob_joint_pass=("abs_delta_predictive_prob_joint_pass", "max"),
            n_tier_changes_vs_full=("tier_changed_vs_full", "sum"),
            n_runs=("omitted_triplet", "nunique"),
        )
        .rename(columns={"target_cohort": "cohort"})
        .sort_values("cohort")
        .reset_index(drop=True)
    )
    cohort_fragility["tier_stable_across_leave_three_out"] = cohort_fragility["n_tier_changes_vs_full"] == 0
    cohort_fragility["l3o_prob_span"] = (
        cohort_fragility["l3o_max_predictive_prob_joint_pass"]
        - cohort_fragility["l3o_min_predictive_prob_joint_pass"]
    )
    cohort_fragility.to_csv(args.output_dir / "leave_three_cohort_fragility_summary.csv", index=False)

    tier_stability = (
        all_runs.groupby(["cohort", "multilevel_adaptive_joint_tier"], as_index=False)
        .agg(n_runs=("omitted_triplet", "nunique"))
        .sort_values(["cohort", "multilevel_adaptive_joint_tier"])
        .reset_index(drop=True)
    )
    tier_stability.to_csv(args.output_dir / "leave_three_cohort_tier_stability.csv", index=False)

    triplet_influence = (
        influence.groupby("omitted_triplet", as_index=False)
        .agg(
            mean_abs_delta_across_targets=("abs_delta_predictive_prob_joint_pass", "mean"),
            max_abs_delta_across_targets=("abs_delta_predictive_prob_joint_pass", "max"),
            n_targets_with_tier_change=("tier_changed_vs_full", "sum"),
        )
        .sort_values("mean_abs_delta_across_targets", ascending=False)
        .reset_index(drop=True)
    )
    triplet_influence.to_csv(args.output_dir / "leave_three_cohort_triplet_influence_summary.csv", index=False)

    source_rows: List[Dict[str, object]] = []
    for cohort in cohorts:
        involved = influence[
            (influence["omitted_cohort_a"] == cohort)
            | (influence["omitted_cohort_b"] == cohort)
            | (influence["omitted_cohort_c"] == cohort)
        ].copy()
        source_rows.append(
            {
                "cohort": cohort,
                "mean_abs_delta_when_in_triplet_omitted": float(involved["abs_delta_predictive_prob_joint_pass"].mean()),
                "max_abs_delta_when_in_triplet_omitted": float(involved["abs_delta_predictive_prob_joint_pass"].max()),
                "n_targets_with_tier_change_when_in_triplet_omitted": int(involved["tier_changed_vs_full"].sum()),
            }
        )
    source_influence = pd.DataFrame(source_rows).sort_values(
        "mean_abs_delta_when_in_triplet_omitted", ascending=False
    )
    source_influence.to_csv(args.output_dir / "leave_three_cohort_source_influence_summary.csv", index=False)

    overall = pd.DataFrame(
        [
            {
                "n_cohorts": int(len(cohorts)),
                "n_leave_three_out_runs": int(len(triplets)),
                "n_targets_with_any_tier_change": int((cohort_fragility["n_tier_changes_vs_full"] > 0).sum()),
                "max_abs_delta_predictive_prob_joint_pass": float(
                    cohort_fragility["l3o_max_abs_delta_predictive_prob_joint_pass"].max()
                ),
                "mean_abs_delta_predictive_prob_joint_pass": float(
                    cohort_fragility["l3o_mean_abs_delta_predictive_prob_joint_pass"].mean()
                ),
                "median_l3o_prob_span": float(cohort_fragility["l3o_prob_span"].median()),
                "max_l3o_prob_span": float(cohort_fragility["l3o_prob_span"].max()),
            }
        ]
    )
    overall.to_csv(args.output_dir / "leave_three_cohort_overall_summary.csv", index=False)

    _plot_fragility(
        cohort_fragility[
            [
                "cohort",
                "full_predictive_prob_joint_pass",
                "l3o_min_predictive_prob_joint_pass",
                "l3o_max_predictive_prob_joint_pass",
            ]
        ].copy(),
        args.output_dir / "leave_three_cohort_fragility_plot.png",
    )

    print("Multilevel leave-three-cohort-out analysis complete.")
    print(overall.to_string(index=False))


if __name__ == "__main__":
    main()
