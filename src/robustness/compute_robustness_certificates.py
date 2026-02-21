#!/usr/bin/env python3
"""Compute cross-model robustness certificates from full source-dropout runs.

This script aggregates posterior shifts across source-dropout runs for:
1) one-factor dependence-aware latent model, and
2) multi-factor dependence-aware latent model.

It produces edge-level stability certificates and source-level regime comparisons.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dep-ablation-dir",
        type=Path,
        default=Path(
            "subproject_24_multireference_latent_truth_model/implementation/outputs/source_dropout_ablation_full"
        ),
    )
    parser.add_argument(
        "--dep-posterior-file",
        type=str,
        default="ltm_dep_edge_posteriors.csv",
    )
    parser.add_argument(
        "--dep-summary-file",
        type=Path,
        default=Path(
            "subproject_24_multireference_latent_truth_model/implementation/outputs/source_dropout_ablation_full/source_dropout_summary.csv"
        ),
    )
    parser.add_argument(
        "--mf-ablation-dir",
        type=Path,
        default=Path(
            "subproject_24_multireference_latent_truth_model/implementation/outputs/source_dropout_ablation_full_mf"
        ),
    )
    parser.add_argument(
        "--mf-posterior-file",
        type=str,
        default="ltm_mf_edge_posteriors.csv",
    )
    parser.add_argument(
        "--mf-summary-file",
        type=Path,
        default=Path(
            "subproject_24_multireference_latent_truth_model/implementation/outputs/source_dropout_ablation_full_mf/source_dropout_summary.csv"
        ),
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5000,
        help="Number of top robust edges to export.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("subproject_24_multireference_latent_truth_model/implementation/outputs"),
    )
    return parser.parse_args()


def _list_run_dirs(ablation_dir: Path) -> List[Tuple[str, Path]]:
    if not ablation_dir.exists():
        raise FileNotFoundError(f"Ablation directory not found: {ablation_dir}")

    baseline_dir = ablation_dir / "baseline"
    if not baseline_dir.exists():
        raise FileNotFoundError(f"Expected baseline directory at: {baseline_dir}")

    drop_dirs = sorted([p for p in ablation_dir.iterdir() if p.is_dir() and p.name.startswith("drop_")])
    return [("baseline", baseline_dir)] + [(p.name, p) for p in drop_dirs]


def _read_posterior_table(path: Path, include_obs_cols: bool) -> pd.DataFrame:
    cols = ["source", "target", "posterior"]
    if include_obs_cols:
        cols += ["n_observed_sources", "n_positive_sources"]
    df = pd.read_csv(path, usecols=cols)
    return df


def _aggregate_model_stats(ablation_dir: Path, posterior_file: str, prefix: str) -> pd.DataFrame:
    run_dirs = _list_run_dirs(ablation_dir)

    baseline_name, baseline_dir = run_dirs[0]
    baseline_path = baseline_dir / posterior_file
    if not baseline_path.exists():
        raise FileNotFoundError(f"Missing posterior file for {baseline_name}: {baseline_path}")

    baseline_df = _read_posterior_table(baseline_path, include_obs_cols=True)
    key_cols = ["source", "target"]
    base = baseline_df.set_index(key_cols)
    base_post = base["posterior"].astype(float)

    sum_post = base_post.copy()
    sumsq_post = base_post.pow(2)
    min_post = base_post.copy()
    max_post = base_post.copy()

    for run_name, run_dir in run_dirs[1:]:
        run_path = run_dir / posterior_file
        if not run_path.exists():
            raise FileNotFoundError(f"Missing posterior file for {run_name}: {run_path}")

        run_df = _read_posterior_table(run_path, include_obs_cols=False).set_index(key_cols)
        run_post = run_df["posterior"].reindex(base.index)

        if run_post.isna().any():
            n_missing = int(run_post.isna().sum())
            raise ValueError(
                f"Run '{run_name}' has {n_missing} edges not aligned with baseline index."
            )

        sum_post = sum_post + run_post
        sumsq_post = sumsq_post + run_post.pow(2)
        min_post = pd.Series(
            np.minimum(min_post.to_numpy(), run_post.to_numpy()),
            index=min_post.index,
        )
        max_post = pd.Series(
            np.maximum(max_post.to_numpy(), run_post.to_numpy()),
            index=max_post.index,
        )

    n_runs = len(run_dirs)
    mean_post = sum_post / float(n_runs)
    var_post = (sumsq_post / float(n_runs)) - mean_post.pow(2)
    std_post = np.sqrt(np.clip(var_post.to_numpy(), a_min=0.0, a_max=None))
    range_post = max_post - min_post

    out = base.reset_index()[["source", "target", "n_observed_sources", "n_positive_sources"]].copy()
    out[f"{prefix}_posterior_baseline"] = base_post.to_numpy()
    out[f"{prefix}_posterior_mean"] = mean_post.to_numpy()
    out[f"{prefix}_posterior_min"] = min_post.to_numpy()
    out[f"{prefix}_posterior_max"] = max_post.to_numpy()
    out[f"{prefix}_posterior_std"] = std_post
    out[f"{prefix}_posterior_range"] = range_post.to_numpy()
    out[f"{prefix}_n_runs"] = n_runs
    return out


def _stability_tier(row: pd.Series) -> str:
    consensus_min = float(row["consensus_posterior_min"])
    max_range = float(row["max_source_dropout_range"])
    model_gap = float(row["model_gap_baseline_abs"])
    n_obs = int(row["n_observed_sources"])

    if consensus_min >= 0.90 and max_range <= 0.10 and model_gap <= 0.10 and n_obs >= 3:
        return "CERT_A_STRONG"
    if consensus_min >= 0.75 and max_range <= 0.20 and model_gap <= 0.20 and n_obs >= 2:
        return "CERT_B_MODERATE"
    if consensus_min >= 0.60 and max_range <= 0.30 and model_gap <= 0.30:
        return "CERT_C_SUGGESTIVE"
    return "CERT_D_EXPLORATORY"


def _compute_source_regime_comparison(
    dep_summary_file: Path,
    mf_summary_file: Path,
) -> pd.DataFrame:
    dep = pd.read_csv(dep_summary_file)
    mf = pd.read_csv(mf_summary_file)

    dep = dep[dep["dropped_source"] != "NONE_BASELINE"].copy()
    mf = mf[mf["dropped_source"] != "NONE_BASELINE"].copy()

    dep = dep[
        [
            "dropped_source",
            "pairwise_rank_disagreement_vs_baseline",
            "n_disagree_pairs",
            "n_valid_pairs",
            "delta_pi_vs_baseline",
            "delta_rho_vs_baseline",
            "top_method",
        ]
    ].rename(
        columns={
            "pairwise_rank_disagreement_vs_baseline": "dep_rank_disagreement",
            "n_disagree_pairs": "dep_n_disagree_pairs",
            "n_valid_pairs": "dep_n_valid_pairs",
            "delta_pi_vs_baseline": "dep_delta_pi",
            "delta_rho_vs_baseline": "dep_delta_rho",
            "top_method": "dep_top_method",
        }
    )

    mf = mf[
        [
            "dropped_source",
            "pairwise_rank_disagreement_vs_baseline",
            "n_disagree_pairs",
            "n_valid_pairs",
            "delta_pi_vs_baseline",
            "delta_rho_vs_baseline",
            "top_method",
        ]
    ].rename(
        columns={
            "pairwise_rank_disagreement_vs_baseline": "mf_rank_disagreement",
            "n_disagree_pairs": "mf_n_disagree_pairs",
            "n_valid_pairs": "mf_n_valid_pairs",
            "delta_pi_vs_baseline": "mf_delta_pi",
            "delta_rho_vs_baseline": "mf_delta_rho",
            "top_method": "mf_top_method",
        }
    )

    merged = dep.merge(mf, on="dropped_source", how="inner")
    merged["delta_rank_disagreement_mf_minus_dep"] = (
        merged["mf_rank_disagreement"] - merged["dep_rank_disagreement"]
    )
    merged["delta_abs_pi_shift_mf_minus_dep"] = (
        np.abs(merged["mf_delta_pi"]) - np.abs(merged["dep_delta_pi"])
    )
    merged["delta_abs_rho_shift_mf_minus_dep"] = (
        np.abs(merged["mf_delta_rho"]) - np.abs(merged["dep_delta_rho"])
    )
    merged["top_method_match"] = (merged["dep_top_method"] == merged["mf_top_method"]).astype(int)

    merged = merged.sort_values(
        ["dep_rank_disagreement", "mf_rank_disagreement"],
        ascending=False,
    ).reset_index(drop=True)
    return merged


def _priority_tiers_from_score(score: pd.Series) -> pd.Series:
    """Build relative-priority tiers from deterministic descending rank."""

    n = len(score)
    if n == 0:
        return pd.Series(dtype=object)

    order = score.sort_values(ascending=False, kind="mergesort").index
    rank_frac = pd.Series(
        np.arange(1, n + 1, dtype=float) / float(n),
        index=order,
    ).reindex(score.index)

    out = np.where(
        rank_frac <= 0.10,
        "PRIORITY_1_TOP10",
        np.where(
            rank_frac <= 0.25,
            "PRIORITY_2_TOP25",
            np.where(rank_frac <= 0.50, "PRIORITY_3_TOP50", "PRIORITY_4_BOTTOM50"),
        ),
    )
    return pd.Series(out, index=score.index)


def main() -> None:
    args = _parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    dep_stats = _aggregate_model_stats(
        ablation_dir=args.dep_ablation_dir,
        posterior_file=args.dep_posterior_file,
        prefix="dep",
    )
    mf_stats = _aggregate_model_stats(
        ablation_dir=args.mf_ablation_dir,
        posterior_file=args.mf_posterior_file,
        prefix="mf",
    )

    merge_cols = ["source", "target", "n_observed_sources", "n_positive_sources"]
    edge = dep_stats.merge(mf_stats, on=merge_cols, how="inner")

    edge["consensus_posterior_baseline"] = (
        edge["dep_posterior_baseline"] + edge["mf_posterior_baseline"]
    ) / 2.0
    edge["consensus_posterior_mean"] = (edge["dep_posterior_mean"] + edge["mf_posterior_mean"]) / 2.0
    edge["consensus_posterior_min"] = np.minimum(
        edge["dep_posterior_min"].to_numpy(),
        edge["mf_posterior_min"].to_numpy(),
    )
    edge["consensus_posterior_max"] = np.maximum(
        edge["dep_posterior_max"].to_numpy(),
        edge["mf_posterior_max"].to_numpy(),
    )
    edge["max_source_dropout_range"] = np.maximum(
        edge["dep_posterior_range"].to_numpy(),
        edge["mf_posterior_range"].to_numpy(),
    )
    edge["model_gap_baseline_abs"] = np.abs(
        edge["dep_posterior_baseline"] - edge["mf_posterior_baseline"]
    )
    edge["model_gap_min_abs"] = np.abs(edge["dep_posterior_min"] - edge["mf_posterior_min"])
    edge["stability_tier"] = edge.apply(_stability_tier, axis=1)

    # Robust ranking score favors high worst-case posterior, penalizes instability,
    # and down-weights edges that have no direct observed-source support.
    edge["robust_score_raw"] = (
        edge["consensus_posterior_min"]
        - 0.5 * edge["max_source_dropout_range"]
        - 0.5 * edge["model_gap_baseline_abs"]
    )
    edge["coverage_penalty"] = np.where(
        edge["n_observed_sources"] <= 0,
        0.25,
        np.where(edge["n_observed_sources"] == 1, 0.10, 0.0),
    )
    edge["robust_score"] = edge["robust_score_raw"] - edge["coverage_penalty"]
    edge["priority_tier"] = _priority_tiers_from_score(edge["robust_score"])

    summary = (
        edge.groupby("stability_tier", as_index=False)
        .agg(
            n_edges=("source", "count"),
            mean_consensus_baseline=("consensus_posterior_baseline", "mean"),
            mean_consensus_min=("consensus_posterior_min", "mean"),
            median_consensus_min=("consensus_posterior_min", "median"),
            mean_max_dropout_range=("max_source_dropout_range", "mean"),
            median_max_dropout_range=("max_source_dropout_range", "median"),
            mean_model_gap=("model_gap_baseline_abs", "mean"),
            mean_observed_sources=("n_observed_sources", "mean"),
            mean_coverage_penalty=("coverage_penalty", "mean"),
        )
        .sort_values("stability_tier")
        .reset_index(drop=True)
    )
    priority_summary = (
        edge.groupby("priority_tier", as_index=False)
        .agg(
            n_edges=("source", "count"),
            mean_consensus_baseline=("consensus_posterior_baseline", "mean"),
            mean_consensus_min=("consensus_posterior_min", "mean"),
            median_consensus_min=("consensus_posterior_min", "median"),
            mean_max_dropout_range=("max_source_dropout_range", "mean"),
            median_max_dropout_range=("max_source_dropout_range", "median"),
            mean_model_gap=("model_gap_baseline_abs", "mean"),
            mean_observed_sources=("n_observed_sources", "mean"),
            mean_coverage_penalty=("coverage_penalty", "mean"),
            mean_robust_score=("robust_score", "mean"),
            min_robust_score=("robust_score", "min"),
            max_robust_score=("robust_score", "max"),
        )
        .sort_values("priority_tier")
        .reset_index(drop=True)
    )

    tier_rank: Dict[str, int] = {
        "CERT_A_STRONG": 0,
        "CERT_B_MODERATE": 1,
        "CERT_C_SUGGESTIVE": 2,
        "CERT_D_EXPLORATORY": 3,
    }
    top = edge.copy()
    top["tier_rank"] = top["stability_tier"].map(tier_rank).fillna(9).astype(int)
    top = top.sort_values(
        ["tier_rank", "robust_score", "consensus_posterior_min"],
        ascending=[True, False, False],
    ).drop(columns=["tier_rank"])
    top = top.head(max(args.top_k, 1)).reset_index(drop=True)
    top_supported = (
        top[top["n_observed_sources"] >= 2]
        .head(max(args.top_k, 1))
        .reset_index(drop=True)
    )

    source_regime = _compute_source_regime_comparison(
        dep_summary_file=args.dep_summary_file,
        mf_summary_file=args.mf_summary_file,
    )

    edge.to_csv(args.output_dir / "robustness_edge_certificates.csv", index=False)
    summary.to_csv(args.output_dir / "robustness_edge_certificate_summary.csv", index=False)
    priority_summary.to_csv(args.output_dir / "robustness_priority_summary.csv", index=False)
    top.to_csv(args.output_dir / "robustness_top_edges.csv", index=False)
    top_supported.to_csv(args.output_dir / "robustness_top_edges_supported.csv", index=False)
    source_regime.to_csv(args.output_dir / "source_dropout_regime_comparison.csv", index=False)

    print("Robustness certificate computation complete:")
    print(f"- edge rows: {len(edge)}")
    print(f"- summary rows: {len(summary)}")
    print(f"- priority summary rows: {len(priority_summary)}")
    print(f"- top edge rows: {len(top)}")
    print(f"- top supported edge rows: {len(top_supported)}")
    print(f"- source regime rows: {len(source_regime)}")
    print(f"- outputs: {args.output_dir}")


if __name__ == "__main__":
    main()
