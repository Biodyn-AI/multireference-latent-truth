#!/usr/bin/env python3
"""Analyze support-stratified behavior of external endpoint replication.

This script summarizes how endpoint success depends on external reference
support size (`n_reference_edges_in_candidate_pool`) across cohort runs.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results",
        type=Path,
        default=Path(
            "subproject_24_multireference_latent_truth_model/implementation/outputs/"
            "top_tier_readiness_v6/multicohort_preregistered_replication_v1/multicohort_replication_results.csv"
        ),
    )
    parser.add_argument(
        "--support-thresholds",
        type=str,
        default="0,5,10,20,50,100,150,200",
        help="Comma-separated minimum-support thresholds for subset pass-rate sweep.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "subproject_24_multireference_latent_truth_model/implementation/outputs/"
            "top_tier_readiness_v6/multicohort_support_stratification_v1"
        ),
    )
    return parser.parse_args()


def _parse_thresholds(spec: str) -> List[int]:
    vals = [int(x.strip()) for x in spec.split(",") if x.strip()]
    if not vals:
        raise ValueError("No thresholds parsed")
    return sorted(set(vals))


def _bin_label(value: float) -> str:
    v = float(value)
    if v <= 10:
        return "low_support_0_10"
    if v <= 50:
        return "mid_support_11_50"
    return "high_support_gt50"


def main() -> None:
    args = _parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    thresholds = _parse_thresholds(args.support_thresholds)

    df = pd.read_csv(args.results, low_memory=False)
    df = df[df["status"] == "completed"].copy()
    if df.empty:
        raise ValueError("No completed cohort rows found in results file")

    need = {
        "cohort",
        "endpoint_pass",
        "selected_enrichment",
        "selected_p_value",
        "n_reference_edges_in_candidate_pool",
        "n_hits",
    }
    missing = sorted(need - set(df.columns))
    if missing:
        raise ValueError(f"Results file missing required columns: {missing}")

    df["endpoint_pass"] = df["endpoint_pass"].astype(bool)
    df["n_reference_edges_in_candidate_pool"] = df["n_reference_edges_in_candidate_pool"].astype(float)
    df["n_hits"] = df["n_hits"].astype(float)
    df["support_bin"] = df["n_reference_edges_in_candidate_pool"].map(_bin_label)
    df["neglog10_p"] = -np.log10(np.maximum(df["selected_p_value"].astype(float), 1e-300))
    df.to_csv(args.output_dir / "support_cohort_table.csv", index=False)

    # Bin-level summary.
    bin_summary = (
        df.groupby("support_bin", as_index=False)
        .agg(
            n_cohorts=("cohort", "count"),
            n_pass=("endpoint_pass", "sum"),
            pass_rate=("endpoint_pass", "mean"),
            mean_support=("n_reference_edges_in_candidate_pool", "mean"),
            mean_enrichment=("selected_enrichment", "mean"),
            median_enrichment=("selected_enrichment", "median"),
            max_p_value=("selected_p_value", "max"),
            mean_hits=("n_hits", "mean"),
        )
        .sort_values("mean_support")
        .reset_index(drop=True)
    )
    bin_summary.to_csv(args.output_dir / "support_bin_summary.csv", index=False)

    # Threshold sweep: how pass-rate changes with minimum support cutoff.
    sweep_rows = []
    for thr in thresholds:
        sub = df[df["n_reference_edges_in_candidate_pool"] >= thr].copy()
        n = int(len(sub))
        n_pass = int(sub["endpoint_pass"].sum()) if n > 0 else 0
        sweep_rows.append(
            {
                "min_support_threshold": int(thr),
                "n_cohorts": n,
                "n_pass": n_pass,
                "pass_rate": float(n_pass / n) if n > 0 else float("nan"),
                "mean_enrichment": float(sub["selected_enrichment"].mean()) if n > 0 else float("nan"),
                "max_p_value": float(sub["selected_p_value"].max()) if n > 0 else float("nan"),
            }
        )
    sweep = pd.DataFrame(sweep_rows)
    sweep.to_csv(args.output_dir / "support_threshold_sweep.csv", index=False)

    # Monotonic association diagnostics.
    corr = pd.DataFrame(
        [
            {
                "metric_x": "support",
                "metric_y": "selected_enrichment",
                "spearman_rho": float(
                    df["n_reference_edges_in_candidate_pool"].corr(df["selected_enrichment"], method="spearman")
                ),
                "pearson_r": float(df["n_reference_edges_in_candidate_pool"].corr(df["selected_enrichment"])),
            },
            {
                "metric_x": "support",
                "metric_y": "neglog10_p",
                "spearman_rho": float(df["n_reference_edges_in_candidate_pool"].corr(df["neglog10_p"], method="spearman")),
                "pearson_r": float(df["n_reference_edges_in_candidate_pool"].corr(df["neglog10_p"])),
            },
            {
                "metric_x": "support",
                "metric_y": "endpoint_pass",
                "spearman_rho": float(
                    df["n_reference_edges_in_candidate_pool"].corr(df["endpoint_pass"].astype(float), method="spearman")
                ),
                "pearson_r": float(df["n_reference_edges_in_candidate_pool"].corr(df["endpoint_pass"].astype(float))),
            },
        ]
    )
    corr.to_csv(args.output_dir / "support_correlation_summary.csv", index=False)

    # Figure: support vs enrichment with pass/fail color.
    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    colors = df["endpoint_pass"].map({True: "#1b9e77", False: "#d95f02"})
    ax.scatter(
        df["n_reference_edges_in_candidate_pool"],
        df["selected_enrichment"],
        c=colors,
        s=85,
        edgecolors="white",
        linewidths=0.8,
    )
    for _, row in df.iterrows():
        ax.annotate(str(row["cohort"]), (row["n_reference_edges_in_candidate_pool"], row["selected_enrichment"]), fontsize=8)
    ax.axhline(1.5, color="#444444", linestyle="--", linewidth=1.0, alpha=0.7, label="endpoint enrichment threshold")
    ax.set_xlabel("External candidate-space support (n reference edges)")
    ax.set_ylabel("Selected endpoint enrichment")
    ax.set_title("Support-Stratified External Endpoint Behavior Across Cohorts")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(args.output_dir / "support_vs_enrichment.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    # One-row operational summary.
    overall = pd.DataFrame(
        [
            {
                "n_cohorts": int(len(df)),
                "n_pass": int(df["endpoint_pass"].sum()),
                "pass_rate": float(df["endpoint_pass"].mean()),
                "min_support": float(df["n_reference_edges_in_candidate_pool"].min()),
                "max_support": float(df["n_reference_edges_in_candidate_pool"].max()),
                "support_median": float(df["n_reference_edges_in_candidate_pool"].median()),
                "failing_cohorts": ";".join(sorted(df.loc[~df["endpoint_pass"], "cohort"].astype(str).tolist())),
            }
        ]
    )
    overall.to_csv(args.output_dir / "support_stratification_overall_summary.csv", index=False)

    print("External support stratification analysis complete.")
    print(overall.to_string(index=False))


if __name__ == "__main__":
    main()
