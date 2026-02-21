#!/usr/bin/env python3
"""Compute hierarchical power-aware endpoint tiers for multi-cohort replication.

This script fits empirical-Bayes Beta priors from cohort-level hit rates and
uses them for posterior endpoint-confidence estimates.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

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
    )
    parser.add_argument(
        "--nonhier-rows",
        type=Path,
        default=Path(
            "subproject_24_multireference_latent_truth_model/implementation/outputs/"
            "top_tier_readiness_v6/power_aware_endpoint_tiering_v1/power_aware_endpoint_rows.csv"
        ),
    )
    parser.add_argument("--endpoint-min-enrichment", type=float, default=1.5)
    parser.add_argument("--endpoint-min-support", type=float, default=10.0)
    parser.add_argument(
        "--group-mode",
        type=str,
        choices=["global", "support_stratified"],
        default="support_stratified",
        help="How empirical-Bayes priors are pooled across cohorts.",
    )
    parser.add_argument("--mc-samples", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--high-confidence", type=float, default=0.95)
    parser.add_argument("--low-confidence", type=float, default=0.05)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "subproject_24_multireference_latent_truth_model/implementation/outputs/"
            "top_tier_readiness_v6/hierarchical_power_aware_endpoint_tiering_v1"
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


def _estimate_beta_prior(rates: np.ndarray, n_selected: np.ndarray) -> Tuple[float, float, Dict[str, float]]:
    rates = np.asarray(rates, dtype=float)
    n_selected = np.asarray(n_selected, dtype=float)
    if rates.size == 0:
        return 1.0, 1.0, {"method": "fallback_empty", "mean_rate": float("nan"), "between_var": float("nan")}

    clipped_rates = np.clip(rates, 1e-9, 1.0 - 1e-9)
    weights = np.clip(n_selected, 1.0, None)
    m = float(np.average(clipped_rates, weights=weights))

    if clipped_rates.size == 1:
        # Single-cohort group cannot estimate between-cohort dispersion.
        return 1.0, 1.0, {"method": "fallback_single", "mean_rate": m, "between_var": float("nan")}

    s2 = float(np.var(clipped_rates, ddof=1))
    within = float(np.average(clipped_rates * (1.0 - clipped_rates) / weights, weights=weights))
    between = max(s2 - within, 1e-12)

    max_between = max(m * (1.0 - m) - 1e-12, 1e-12)
    between = min(between, max_between)

    kappa = m * (1.0 - m) / between - 1.0
    if not np.isfinite(kappa) or kappa <= 0:
        return 1.0, 1.0, {
            "method": "fallback_invalid_kappa",
            "mean_rate": m,
            "between_var": between,
        }

    kappa = float(np.clip(kappa, 2.0, 1e8))
    alpha = float(np.clip(m * kappa, 1e-6, 1e9))
    beta = float(np.clip((1.0 - m) * kappa, 1e-6, 1e9))
    return alpha, beta, {
        "method": "moments_between_corrected",
        "mean_rate": m,
        "between_var": between,
        "kappa": kappa,
        "sample_var": s2,
        "within_var": within,
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


def _build_base_rows(args: argparse.Namespace) -> pd.DataFrame:
    results = pd.read_csv(args.results, low_memory=False)
    results = results[results["status"] == "completed"].copy()
    if results.empty:
        raise ValueError("No completed cohort rows found in results file")

    rows: List[Dict[str, object]] = []
    for row in results.sort_values("cohort").itertuples(index=False):
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
            }
        )

    out = pd.DataFrame(rows).sort_values("cohort").reset_index(drop=True)
    return out


def _build_group_priors(base: pd.DataFrame, group_mode: str) -> pd.DataFrame:
    global_alpha, global_beta, global_diag = _estimate_beta_prior(
        rates=base["observed_hits"].to_numpy(dtype=float) / np.clip(base["n_selected"].to_numpy(dtype=float), 1.0, None),
        n_selected=base["n_selected"].to_numpy(dtype=float),
    )

    rows: List[Dict[str, object]] = [
        {
            "group": "global",
            "group_mode": group_mode,
            "n_cohorts": int(len(base)),
            "prior_alpha": global_alpha,
            "prior_beta": global_beta,
            **global_diag,
            "borrowed_from_global": False,
        }
    ]

    if group_mode == "global":
        return pd.DataFrame(rows)

    for support_value in [True, False]:
        sub = base[base["observed_support_eligible"] == support_value].copy()
        group_name = f"support_eligible_{int(support_value)}"
        if len(sub) >= 2:
            alpha, beta, diag = _estimate_beta_prior(
                rates=sub["observed_hits"].to_numpy(dtype=float)
                / np.clip(sub["n_selected"].to_numpy(dtype=float), 1.0, None),
                n_selected=sub["n_selected"].to_numpy(dtype=float),
            )
            borrowed = False
        else:
            alpha, beta = global_alpha, global_beta
            diag = {
                "method": "borrow_global_due_to_small_group",
                "mean_rate": float(sub["observed_hits"].sum() / np.clip(sub["n_selected"].sum(), 1.0, None))
                if len(sub) > 0
                else float("nan"),
                "between_var": float("nan"),
            }
            borrowed = True

        rows.append(
            {
                "group": group_name,
                "group_mode": group_mode,
                "n_cohorts": int(len(sub)),
                "prior_alpha": float(alpha),
                "prior_beta": float(beta),
                **diag,
                "borrowed_from_global": borrowed,
            }
        )

    return pd.DataFrame(rows)


def _select_prior(priors: pd.DataFrame, group_mode: str, support_eligible: bool) -> Tuple[float, float, str]:
    if group_mode == "global":
        row = priors[priors["group"] == "global"].iloc[0]
        return float(row["prior_alpha"]), float(row["prior_beta"]), "global"

    key = f"support_eligible_{int(support_eligible)}"
    sub = priors[priors["group"] == key]
    if sub.empty:
        row = priors[priors["group"] == "global"].iloc[0]
        return float(row["prior_alpha"]), float(row["prior_beta"]), "global_fallback"
    row = sub.iloc[0]
    return float(row["prior_alpha"]), float(row["prior_beta"]), key


def main() -> None:
    args = _parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    base = _build_base_rows(args)
    priors = _build_group_priors(base, group_mode=args.group_mode)
    priors.to_csv(args.output_dir / "hierarchical_prior_summary.csv", index=False)

    rows: List[Dict[str, object]] = []
    for idx, row in enumerate(base.itertuples(index=False)):
        prior_alpha, prior_beta, prior_group = _select_prior(
            priors=priors,
            group_mode=args.group_mode,
            support_eligible=bool(row.observed_support_eligible),
        )
        bayes = _bayes_endpoint_probs(
            hits=int(row.observed_hits),
            n_selected=int(row.n_selected),
            endpoint_hit_rate_threshold=float(row.endpoint_hit_rate_threshold),
            prior_alpha=prior_alpha,
            prior_beta=prior_beta,
            mc_samples=int(args.mc_samples),
            seed=int(args.seed + idx),
        )

        random_hit_rate = float(row.random_hit_rate)
        if random_hit_rate > 0:
            enrich_q025 = float(bayes["posterior_hit_rate_q025"] / random_hit_rate)
            enrich_q50 = float(bayes["posterior_hit_rate_q50"] / random_hit_rate)
            enrich_q975 = float(bayes["posterior_hit_rate_q975"] / random_hit_rate)
        else:
            enrich_q025 = float("nan")
            enrich_q50 = float("nan")
            enrich_q975 = float("nan")

        tier = _power_tier(
            support_eligible=bool(row.observed_support_eligible),
            observed_pass=bool(row.observed_endpoint_pass),
            posterior_prob_ge_threshold=float(bayes["posterior_prob_hit_rate_ge_threshold"]),
            high_conf=float(args.high_confidence),
            low_conf=float(args.low_confidence),
        )

        rows.append(
            {
                **row._asdict(),
                "prior_group": prior_group,
                "prior_alpha": prior_alpha,
                "prior_beta": prior_beta,
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
                "power_aware_tier": tier,
            }
        )

    out = pd.DataFrame(rows).sort_values("cohort").reset_index(drop=True)
    out.to_csv(args.output_dir / "hierarchical_power_aware_endpoint_rows.csv", index=False)

    # Compare against non-hierarchical posterior confidence if available.
    if args.nonhier_rows.exists():
        nonhier = pd.read_csv(args.nonhier_rows, low_memory=False)
        cmp = out.merge(
            nonhier[["cohort", "posterior_prob_hit_rate_ge_threshold", "power_aware_tier"]].rename(
                columns={
                    "posterior_prob_hit_rate_ge_threshold": "posterior_prob_nonhier",
                    "power_aware_tier": "tier_nonhier",
                }
            ),
            on="cohort",
            how="left",
        )
        cmp["posterior_prob_delta_hier_minus_nonhier"] = (
            cmp["posterior_prob_hit_rate_ge_threshold"] - cmp["posterior_prob_nonhier"]
        )
        cmp["tier_changed_vs_nonhier"] = cmp["power_aware_tier"] != cmp["tier_nonhier"]
        cmp.to_csv(args.output_dir / "hierarchical_vs_nonhierarchical_comparison.csv", index=False)

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
    support_summary.to_csv(args.output_dir / "hierarchical_power_support_summary.csv", index=False)

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
    tier_summary.to_csv(args.output_dir / "hierarchical_power_tier_summary.csv", index=False)

    overall = pd.DataFrame(
        [
            {
                "group_mode": args.group_mode,
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
    overall.to_csv(args.output_dir / "hierarchical_power_overall_summary.csv", index=False)

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
    ax.set_ylabel("Hierarchical posterior P(enrichment >= endpoint threshold)")
    ax.set_title("Hierarchical Power-Aware Endpoint Probability Across Cohorts")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(args.output_dir / "hierarchical_power_support_vs_probability.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    print("Hierarchical power-aware endpoint tiering complete.")
    print(overall.to_string(index=False))


if __name__ == "__main__":
    main()
