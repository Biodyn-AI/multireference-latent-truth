#!/usr/bin/env python3
"""Compute power-aware endpoint tiers for multi-cohort external replication.

This module upgrades fixed-threshold endpoint interpretation with a small
Bayesian model over cohort-level hit rates:
1) reads multicohort replication outputs,
2) estimates posterior probability that enrichment exceeds endpoint threshold,
3) assigns powered vs underpowered endpoint tiers.
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
        help="Root directory containing cohort_<name>/run endpoint comparison files.",
    )
    parser.add_argument("--endpoint-min-enrichment", type=float, default=1.5)
    parser.add_argument("--endpoint-min-support", type=float, default=10.0)
    parser.add_argument("--prior-alpha", type=float, default=1.0)
    parser.add_argument("--prior-beta", type=float, default=1.0)
    parser.add_argument("--mc-samples", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--high-confidence", type=float, default=0.95)
    parser.add_argument("--low-confidence", type=float, default=0.05)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "subproject_24_multireference_latent_truth_model/implementation/outputs/"
            "top_tier_readiness_v6/power_aware_endpoint_tiering_v1"
        ),
    )
    return parser.parse_args()


def _to_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    text = str(value).strip().lower()
    return text in {"1", "true", "t", "yes", "y"}


def _cohort_comparison_row(
    replication_root: Path,
    cohort: str,
    selected_policy: str,
) -> Dict[str, float]:
    comp_path = replication_root / f"cohort_{cohort}" / "run" / "preregistered_external_endpoint_comparison.csv"
    if not comp_path.exists():
        raise FileNotFoundError(f"Missing cohort endpoint comparison: {comp_path}")
    comp = pd.read_csv(comp_path, low_memory=False)
    if comp.empty:
        raise ValueError(f"Empty cohort endpoint comparison: {comp_path}")
    sel = comp[comp["selection"] == str(selected_policy)]
    row = sel.iloc[0] if not sel.empty else comp.iloc[0]
    return {
        "n_selected": float(row["n_selected"]),
        "n_reference_edges_in_candidate_pool": float(row["n_reference_edges_in_candidate_pool"]),
        "n_hits": float(row["n_hits"]),
        "expected_hits_random_from_candidate_pool": float(row["expected_hits_random_from_candidate_pool"]),
        "enrichment_vs_random": float(row["enrichment_vs_random"]),
        "p_value_hypergeom_ge_hits": float(row["p_value_hypergeom_ge_hits"]),
    }


def _bayes_endpoint_probs(
    *,
    hits: int,
    n_selected: int,
    endpoint_hit_rate_threshold: float,
    prior_alpha: float,
    prior_beta: float,
    mc_samples: int,
    seed: int,
) -> Dict[str, float]:
    if n_selected <= 0:
        raise ValueError("n_selected must be positive")
    endpoint_hit_rate_threshold = float(np.clip(endpoint_hit_rate_threshold, 0.0, 1.0))

    post_alpha = float(prior_alpha + hits)
    post_beta = float(prior_beta + (n_selected - hits))
    rng = np.random.default_rng(seed)
    theta = rng.beta(post_alpha, post_beta, size=mc_samples)
    posterior_prob_ge_threshold = float(np.mean(theta >= endpoint_hit_rate_threshold))

    min_hits_for_enrichment = int(np.ceil(endpoint_hit_rate_threshold * n_selected))
    pred_hits = rng.binomial(n_selected, theta)
    predictive_prob_pass = float(np.mean(pred_hits >= min_hits_for_enrichment))

    q025, q50, q975 = np.quantile(theta, [0.025, 0.5, 0.975])
    return {
        "posterior_alpha": post_alpha,
        "posterior_beta": post_beta,
        "posterior_prob_hit_rate_ge_threshold": posterior_prob_ge_threshold,
        "predictive_prob_observed_enrichment_pass": predictive_prob_pass,
        "posterior_hit_rate_q025": float(q025),
        "posterior_hit_rate_q50": float(q50),
        "posterior_hit_rate_q975": float(q975),
        "min_hits_for_enrichment": float(min_hits_for_enrichment),
    }


def _power_tier(
    *,
    support_eligible: bool,
    observed_pass: bool,
    posterior_prob_ge_threshold: float,
    high_conf: float,
    low_conf: float,
) -> str:
    if support_eligible:
        if observed_pass and posterior_prob_ge_threshold >= high_conf:
            return "POWERED_CONFIRMED_PASS"
        if (not observed_pass) and posterior_prob_ge_threshold <= low_conf:
            return "POWERED_CONFIRMED_FAIL"
        if (not observed_pass) and posterior_prob_ge_threshold >= high_conf:
            return "POWERED_CONTRADICTORY_FAIL"
        if observed_pass and posterior_prob_ge_threshold <= low_conf:
            return "POWERED_CONTRADICTORY_PASS"
        return "POWERED_AMBIGUOUS"

    if posterior_prob_ge_threshold >= high_conf:
        return "UNDERPOWERED_POSITIVE_SIGNAL"
    if posterior_prob_ge_threshold <= low_conf:
        return "UNDERPOWERED_NEGATIVE_SIGNAL"
    return "UNDERPOWERED_AMBIGUOUS"


def main() -> None:
    args = _parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.results, low_memory=False)
    df = df[df["status"] == "completed"].copy()
    if df.empty:
        raise ValueError("No completed cohort rows found in results file")

    rows: List[Dict[str, object]] = []
    for idx, row in enumerate(df.sort_values("cohort").itertuples(index=False)):
        cohort = str(row.cohort)
        selected_policy = str(row.selected_policy)
        observed_pass = _to_bool(getattr(row, "endpoint_pass", False))
        support_edges = float(getattr(row, "n_reference_edges_in_candidate_pool"))
        support_eligible = _to_bool(getattr(row, "endpoint_support_eligible", support_edges >= args.endpoint_min_support))

        comp = _cohort_comparison_row(
            replication_root=args.replication_root,
            cohort=cohort,
            selected_policy=selected_policy,
        )
        n_selected = int(round(comp["n_selected"]))
        n_hits = int(round(comp["n_hits"]))
        expected_hits = float(comp["expected_hits_random_from_candidate_pool"])
        random_hit_rate = float(expected_hits / n_selected) if n_selected > 0 else float("nan")
        threshold_hit_rate = float(args.endpoint_min_enrichment * random_hit_rate) if pd.notna(random_hit_rate) else float("nan")

        bayes = _bayes_endpoint_probs(
            hits=n_hits,
            n_selected=n_selected,
            endpoint_hit_rate_threshold=threshold_hit_rate,
            prior_alpha=float(args.prior_alpha),
            prior_beta=float(args.prior_beta),
            mc_samples=int(args.mc_samples),
            seed=int(args.seed + idx),
        )

        if random_hit_rate > 0:
            enrich_q025 = float(bayes["posterior_hit_rate_q025"] / random_hit_rate)
            enrich_q50 = float(bayes["posterior_hit_rate_q50"] / random_hit_rate)
            enrich_q975 = float(bayes["posterior_hit_rate_q975"] / random_hit_rate)
        else:
            enrich_q025 = float("nan")
            enrich_q50 = float("nan")
            enrich_q975 = float("nan")

        power_tier = _power_tier(
            support_eligible=support_eligible,
            observed_pass=observed_pass,
            posterior_prob_ge_threshold=float(bayes["posterior_prob_hit_rate_ge_threshold"]),
            high_conf=float(args.high_confidence),
            low_conf=float(args.low_confidence),
        )

        rows.append(
            {
                "cohort": cohort,
                "selected_policy": selected_policy,
                "observed_endpoint_pass": observed_pass,
                "observed_support_eligible": support_eligible,
                "observed_support_edges": support_edges,
                "observed_hits": n_hits,
                "observed_enrichment": float(comp["enrichment_vs_random"]),
                "observed_p_value": float(comp["p_value_hypergeom_ge_hits"]),
                "n_selected": n_selected,
                "expected_hits_random": expected_hits,
                "random_hit_rate": random_hit_rate,
                "endpoint_min_enrichment": float(args.endpoint_min_enrichment),
                "endpoint_min_support": float(args.endpoint_min_support),
                "endpoint_hit_rate_threshold": threshold_hit_rate,
                "min_hits_for_enrichment": float(bayes["min_hits_for_enrichment"]),
                "posterior_alpha": float(bayes["posterior_alpha"]),
                "posterior_beta": float(bayes["posterior_beta"]),
                "posterior_prob_hit_rate_ge_threshold": float(bayes["posterior_prob_hit_rate_ge_threshold"]),
                "predictive_prob_observed_enrichment_pass": float(bayes["predictive_prob_observed_enrichment_pass"]),
                "posterior_hit_rate_q025": float(bayes["posterior_hit_rate_q025"]),
                "posterior_hit_rate_q50": float(bayes["posterior_hit_rate_q50"]),
                "posterior_hit_rate_q975": float(bayes["posterior_hit_rate_q975"]),
                "posterior_enrichment_q025": enrich_q025,
                "posterior_enrichment_q50": enrich_q50,
                "posterior_enrichment_q975": enrich_q975,
                "power_aware_tier": power_tier,
            }
        )

    out = pd.DataFrame(rows).sort_values("cohort").reset_index(drop=True)
    out.to_csv(args.output_dir / "power_aware_endpoint_rows.csv", index=False)

    support_summary = (
        out.groupby("observed_support_eligible", as_index=False)
        .agg(
            n_cohorts=("cohort", "count"),
            observed_pass_rate=("observed_endpoint_pass", "mean"),
            mean_posterior_prob_ge_threshold=("posterior_prob_hit_rate_ge_threshold", "mean"),
            median_posterior_prob_ge_threshold=("posterior_prob_hit_rate_ge_threshold", "median"),
            mean_predictive_prob_pass=("predictive_prob_observed_enrichment_pass", "mean"),
        )
        .sort_values("observed_support_eligible", ascending=False)
        .reset_index(drop=True)
    )
    support_summary.to_csv(args.output_dir / "power_aware_support_summary.csv", index=False)

    tier_summary = (
        out.groupby("power_aware_tier", as_index=False)
        .agg(
            n_cohorts=("cohort", "count"),
            mean_support_edges=("observed_support_edges", "mean"),
            observed_pass_rate=("observed_endpoint_pass", "mean"),
            mean_posterior_prob_ge_threshold=("posterior_prob_hit_rate_ge_threshold", "mean"),
        )
        .sort_values("power_aware_tier")
        .reset_index(drop=True)
    )
    tier_summary.to_csv(args.output_dir / "power_aware_tier_summary.csv", index=False)

    overall = pd.DataFrame(
        [
            {
                "n_cohorts": int(len(out)),
                "n_observed_pass": int(out["observed_endpoint_pass"].sum()),
                "observed_pass_rate": float(out["observed_endpoint_pass"].mean()),
                "n_powered_confirmed_pass": int((out["power_aware_tier"] == "POWERED_CONFIRMED_PASS").sum()),
                "n_powered_confirmed_fail": int((out["power_aware_tier"] == "POWERED_CONFIRMED_FAIL").sum()),
                "n_underpowered_positive_signal": int((out["power_aware_tier"] == "UNDERPOWERED_POSITIVE_SIGNAL").sum()),
                "n_underpowered_negative_signal": int((out["power_aware_tier"] == "UNDERPOWERED_NEGATIVE_SIGNAL").sum()),
            }
        ]
    )
    overall.to_csv(args.output_dir / "power_aware_overall_summary.csv", index=False)

    # Visualize probability that enrichment clears the endpoint threshold.
    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    colors = out["observed_endpoint_pass"].map({True: "#1b9e77", False: "#d95f02"})
    ax.scatter(
        out["observed_support_edges"],
        out["posterior_prob_hit_rate_ge_threshold"],
        c=colors,
        s=90,
        edgecolors="white",
        linewidths=0.8,
    )
    for _, r in out.iterrows():
        ax.annotate(str(r["cohort"]), (r["observed_support_edges"], r["posterior_prob_hit_rate_ge_threshold"]), fontsize=8)
    ax.axvline(float(args.endpoint_min_support), color="#444444", linestyle="--", linewidth=1.0, alpha=0.75)
    ax.axhline(float(args.high_confidence), color="#2166ac", linestyle="--", linewidth=1.0, alpha=0.75)
    ax.axhline(float(args.low_confidence), color="#b2182b", linestyle="--", linewidth=1.0, alpha=0.75)
    ax.set_xlabel("External candidate-space support (n reference edges)")
    ax.set_ylabel("Posterior P(enrichment >= endpoint threshold)")
    ax.set_title("Power-Aware Endpoint Probability Across Cohorts")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(args.output_dir / "power_aware_support_vs_probability.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    print("Power-aware endpoint tiering complete.")
    print(overall.to_string(index=False))


if __name__ == "__main__":
    main()
