#!/usr/bin/env python3
"""Compute full hierarchical-Bayes endpoint tiers with null-tail integration.

Compared with empirical-Bayes plug-in priors, this script integrates over
hyperparameter uncertainty via grid-based hyperposteriors within cohort groups.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import betaln, logsumexp
from scipy.stats import beta as beta_dist
from scipy.stats import hypergeom


_EPS = 1e-12


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
    parser.add_argument(
        "--hier-rows",
        type=Path,
        default=Path(
            "subproject_24_multireference_latent_truth_model/implementation/outputs/"
            "top_tier_readiness_v6/hierarchical_power_aware_endpoint_tiering_v1/"
            "hierarchical_power_aware_endpoint_rows.csv"
        ),
    )
    parser.add_argument("--endpoint-min-enrichment", type=float, default=1.5)
    parser.add_argument("--endpoint-max-p", type=float, default=0.05)
    parser.add_argument("--endpoint-min-support", type=float, default=10.0)
    parser.add_argument(
        "--group-mode",
        choices=["global", "support_stratified"],
        default="support_stratified",
    )
    parser.add_argument("--grid-log-alpha-start", type=float, default=-2.0)
    parser.add_argument("--grid-log-alpha-stop", type=float, default=6.0)
    parser.add_argument("--grid-log-alpha-steps", type=int, default=90)
    parser.add_argument("--grid-log-beta-start", type=float, default=-2.0)
    parser.add_argument("--grid-log-beta-stop", type=float, default=10.0)
    parser.add_argument("--grid-log-beta-steps", type=int, default=120)
    parser.add_argument(
        "--log-kappa-prior-sd",
        type=float,
        default=2.0,
        help="Weak regularizer on log(alpha+beta) to avoid extreme-grid dominance.",
    )
    parser.add_argument("--mc-samples", type=int, default=30000)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--high-confidence", type=float, default=0.95)
    parser.add_argument("--low-confidence", type=float, default=0.05)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "subproject_24_multireference_latent_truth_model/implementation/outputs/"
            "top_tier_readiness_v6/full_bayes_endpoint_tiering_v1"
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


def _cohort_comparison_row(replication_root: Path, cohort: str, selected_policy: str) -> Dict[str, float]:
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


def _log_betabinom_mass(h: np.ndarray, n: np.ndarray, alpha: np.ndarray, beta: np.ndarray) -> np.ndarray:
    # Up to a constant in h,n; sufficient for hyperposterior weighting.
    # shape out: (n_grid,)
    # We rely on broadcasting: alpha/beta are (n_grid,), h/n are (n_obs,)
    term = betaln(h[None, :] + alpha[:, None], (n - h)[None, :] + beta[:, None]) - betaln(
        alpha[:, None], beta[:, None]
    )
    return np.sum(term, axis=1)


def _build_hyper_grid(args: argparse.Namespace) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    log_a = np.linspace(args.grid_log_alpha_start, args.grid_log_alpha_stop, args.grid_log_alpha_steps)
    log_b = np.linspace(args.grid_log_beta_start, args.grid_log_beta_stop, args.grid_log_beta_steps)
    a_mesh, b_mesh = np.meshgrid(np.exp(log_a), np.exp(log_b), indexing="ij")
    alpha = a_mesh.reshape(-1)
    beta = b_mesh.reshape(-1)
    log_alpha = np.log(alpha)
    log_beta = np.log(beta)
    return alpha, beta, log_alpha, log_beta, np.log(alpha + beta)


def _fit_hyperposterior(
    *,
    hits: np.ndarray,
    n_selected: np.ndarray,
    alpha_grid: np.ndarray,
    beta_grid: np.ndarray,
    log_kappa_grid: np.ndarray,
    log_kappa_prior_sd: float,
) -> Dict[str, np.ndarray | float]:
    if hits.size == 0:
        raise ValueError("Cannot fit hyperposterior with empty cohort set")

    ll = _log_betabinom_mass(h=hits.astype(float), n=n_selected.astype(float), alpha=alpha_grid, beta=beta_grid)

    # Weak regularizer centered at moderate concentration (log-kappa=3 => kappa~20).
    if log_kappa_prior_sd > 0:
        lp = -0.5 * ((log_kappa_grid - 3.0) / float(log_kappa_prior_sd)) ** 2
    else:
        lp = np.zeros_like(ll)

    log_post = ll + lp
    log_post = log_post - logsumexp(log_post)
    w = np.exp(log_post)

    mean_alpha = float(np.sum(w * alpha_grid))
    mean_beta = float(np.sum(w * beta_grid))
    mean_kappa = float(np.sum(w * (alpha_grid + beta_grid)))
    mean_rate = float(np.sum(w * (alpha_grid / np.maximum(alpha_grid + beta_grid, _EPS))))

    return {
        "weights": w,
        "mean_alpha": mean_alpha,
        "mean_beta": mean_beta,
        "mean_kappa": mean_kappa,
        "mean_rate": mean_rate,
    }


def _build_base_rows(args: argparse.Namespace) -> pd.DataFrame:
    df = pd.read_csv(args.results, low_memory=False)
    df = df[df["status"] == "completed"].copy()
    if df.empty:
        raise ValueError("No completed cohort rows found in results file")

    rows: List[Dict[str, object]] = []
    for row in df.sort_values("cohort").itertuples(index=False):
        cohort = str(row.cohort)
        selected_policy = str(row.selected_policy)
        observed_pass = _to_bool(getattr(row, "endpoint_pass", False))
        support_edges = float(getattr(row, "n_reference_edges_in_candidate_pool"))
        support_eligible = _to_bool(getattr(row, "endpoint_support_eligible", support_edges >= args.endpoint_min_support))

        comp = _cohort_comparison_row(args.replication_root, cohort, selected_policy)
        n_selected = int(round(comp["n_selected"]))
        n_hits = int(round(comp["n_hits"]))
        k_ref = int(round(comp["n_reference_edges_in_candidate_pool"]))
        expected_hits = float(comp["expected_hits_random_from_candidate_pool"])
        random_hit_rate = float(expected_hits / max(n_selected, 1))
        endpoint_hit_rate_threshold = float(args.endpoint_min_enrichment * random_hit_rate)
        min_hits_for_enrichment = int(np.ceil(args.endpoint_min_enrichment * expected_hits))

        # Infer universe size from expectation identity: E = n*K/N.
        if expected_hits <= 0 or k_ref <= 0:
            n_universe = int(max(n_selected, 1))
        else:
            n_universe = int(round((n_selected * k_ref) / expected_hits))
            n_universe = int(max(n_universe, n_selected, k_ref))

        rows.append(
            {
                "cohort": cohort,
                "selected_policy": selected_policy,
                "observed_endpoint_pass": observed_pass,
                "observed_support_eligible": support_eligible,
                "observed_support_edges": float(k_ref),
                "observed_hits": int(n_hits),
                "observed_enrichment": float(comp["enrichment_vs_random"]),
                "observed_p_value": float(comp["p_value_hypergeom_ge_hits"]),
                "n_selected": int(n_selected),
                "n_reference_edges_in_candidate_pool": int(k_ref),
                "n_universe": int(n_universe),
                "expected_hits_random": expected_hits,
                "random_hit_rate": random_hit_rate,
                "endpoint_min_enrichment": float(args.endpoint_min_enrichment),
                "endpoint_max_p": float(args.endpoint_max_p),
                "endpoint_min_support": float(args.endpoint_min_support),
                "endpoint_hit_rate_threshold": endpoint_hit_rate_threshold,
                "min_hits_for_enrichment": int(min_hits_for_enrichment),
            }
        )

    return pd.DataFrame(rows).sort_values("cohort").reset_index(drop=True)


def _group_label(support_eligible: bool, group_mode: str) -> str:
    if group_mode == "global":
        return "global"
    return f"support_eligible_{int(bool(support_eligible))}"


def _fit_group_posteriors(base: pd.DataFrame, args: argparse.Namespace) -> Tuple[Dict[str, Dict[str, np.ndarray | float]], pd.DataFrame, np.ndarray, np.ndarray]:
    alpha_grid, beta_grid, _log_alpha_grid, _log_beta_grid, log_kappa_grid = _build_hyper_grid(args)

    groups = ["global"] if args.group_mode == "global" else ["support_eligible_1", "support_eligible_0"]
    posteriors: Dict[str, Dict[str, np.ndarray | float]] = {}
    summary_rows: List[Dict[str, object]] = []

    # Always fit global fallback.
    global_fit = _fit_hyperposterior(
        hits=base["observed_hits"].to_numpy(dtype=float),
        n_selected=base["n_selected"].to_numpy(dtype=float),
        alpha_grid=alpha_grid,
        beta_grid=beta_grid,
        log_kappa_grid=log_kappa_grid,
        log_kappa_prior_sd=float(args.log_kappa_prior_sd),
    )
    posteriors["global"] = global_fit

    summary_rows.append(
        {
            "group": "global",
            "group_mode": args.group_mode,
            "n_cohorts": int(len(base)),
            "mean_alpha": float(global_fit["mean_alpha"]),
            "mean_beta": float(global_fit["mean_beta"]),
            "mean_kappa": float(global_fit["mean_kappa"]),
            "mean_rate": float(global_fit["mean_rate"]),
            "borrowed_global": False,
        }
    )

    if args.group_mode != "global":
        for g in groups:
            if g == "global":
                continue
            sub = base[base["observed_support_eligible"] == (g.endswith("1"))]
            if len(sub) == 0:
                posteriors[g] = global_fit
                summary_rows.append(
                    {
                        "group": g,
                        "group_mode": args.group_mode,
                        "n_cohorts": 0,
                        "mean_alpha": float(global_fit["mean_alpha"]),
                        "mean_beta": float(global_fit["mean_beta"]),
                        "mean_kappa": float(global_fit["mean_kappa"]),
                        "mean_rate": float(global_fit["mean_rate"]),
                        "borrowed_global": True,
                    }
                )
                continue

            fit = _fit_hyperposterior(
                hits=sub["observed_hits"].to_numpy(dtype=float),
                n_selected=sub["n_selected"].to_numpy(dtype=float),
                alpha_grid=alpha_grid,
                beta_grid=beta_grid,
                log_kappa_grid=log_kappa_grid,
                log_kappa_prior_sd=float(args.log_kappa_prior_sd),
            )
            posteriors[g] = fit
            summary_rows.append(
                {
                    "group": g,
                    "group_mode": args.group_mode,
                    "n_cohorts": int(len(sub)),
                    "mean_alpha": float(fit["mean_alpha"]),
                    "mean_beta": float(fit["mean_beta"]),
                    "mean_kappa": float(fit["mean_kappa"]),
                    "mean_rate": float(fit["mean_rate"]),
                    "borrowed_global": False,
                }
            )

    summary = pd.DataFrame(summary_rows).sort_values("group").reset_index(drop=True)
    return posteriors, summary, alpha_grid, beta_grid


def _tier_from_joint_prob(
    *,
    support_eligible: bool,
    joint_prob: float,
    high_conf: float,
    low_conf: float,
) -> str:
    if support_eligible:
        if joint_prob >= high_conf:
            return "POWERED_JOINT_CONFIRMED_PASS"
        if joint_prob <= low_conf:
            return "POWERED_JOINT_CONFIRMED_FAIL"
        return "POWERED_JOINT_AMBIGUOUS"

    if joint_prob >= high_conf:
        return "UNDERPOWERED_JOINT_POSITIVE_SIGNAL"
    if joint_prob <= low_conf:
        return "UNDERPOWERED_JOINT_NEGATIVE_SIGNAL"
    return "UNDERPOWERED_JOINT_AMBIGUOUS"


def main() -> None:
    args = _parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    base = _build_base_rows(args)
    posteriors, prior_summary, alpha_grid, beta_grid = _fit_group_posteriors(base, args)
    prior_summary.to_csv(args.output_dir / "full_bayes_prior_summary.csv", index=False)

    rng = np.random.default_rng(int(args.seed))
    rows: List[Dict[str, object]] = []

    for idx, row in enumerate(base.itertuples(index=False)):
        group = _group_label(bool(row.observed_support_eligible), args.group_mode)
        post = posteriors.get(group, posteriors["global"])
        weights = np.asarray(post["weights"], dtype=float)

        draw_idx = rng.choice(len(weights), size=int(args.mc_samples), replace=True, p=weights)
        alpha_draw = alpha_grid[draw_idx]
        beta_draw = beta_grid[draw_idx]

        # Cohort-level posterior integrates observed cohort evidence with
        # sampled hyperparameters from the full hyperposterior.
        theta = beta_dist.rvs(
            a=alpha_draw + float(row.observed_hits),
            b=beta_draw + float(row.n_selected - row.observed_hits),
            random_state=np.random.default_rng(int(args.seed + idx + 1)),
        )
        theta = np.clip(theta, _EPS, 1.0 - _EPS)

        posterior_prob_enrich_threshold = float(np.mean(theta >= float(row.endpoint_hit_rate_threshold)))

        n_selected = int(row.n_selected)
        pred_hits = rng.binomial(n=n_selected, p=theta, size=int(args.mc_samples))

        min_hits_for_enrichment = int(row.min_hits_for_enrichment)
        predictive_prob_enrich_pass = float(np.mean(pred_hits >= min_hits_for_enrichment))

        # Exact null-tail integration via hypergeometric survival under random candidate null.
        n_universe = int(row.n_universe)
        k_ref = int(row.n_reference_edges_in_candidate_pool)
        hit_grid = np.arange(n_selected + 1)
        # sf(k-1) gives P[X >= k]
        pvals = hypergeom.sf(hit_grid - 1, M=n_universe, n=k_ref, N=n_selected)
        pred_pvals = pvals[pred_hits]
        predictive_prob_joint_pass = float(
            np.mean((pred_hits >= min_hits_for_enrichment) & (pred_pvals <= float(args.endpoint_max_p)))
        )

        q025, q50, q975 = np.quantile(theta, [0.025, 0.5, 0.975])
        random_hit_rate = float(row.random_hit_rate)
        if random_hit_rate > 0:
            enrich_q025 = float(q025 / random_hit_rate)
            enrich_q50 = float(q50 / random_hit_rate)
            enrich_q975 = float(q975 / random_hit_rate)
        else:
            enrich_q025 = float("nan")
            enrich_q50 = float("nan")
            enrich_q975 = float("nan")

        tier = _tier_from_joint_prob(
            support_eligible=bool(row.observed_support_eligible),
            joint_prob=predictive_prob_joint_pass,
            high_conf=float(args.high_confidence),
            low_conf=float(args.low_confidence),
        )

        rows.append(
            {
                **row._asdict(),
                "prior_group": group,
                "prior_alpha_mean": float(post["mean_alpha"]),
                "prior_beta_mean": float(post["mean_beta"]),
                "posterior_prob_enrichment_threshold": posterior_prob_enrich_threshold,
                "predictive_prob_enrichment_pass": predictive_prob_enrich_pass,
                "predictive_prob_joint_pass": predictive_prob_joint_pass,
                "posterior_hit_rate_q025": float(q025),
                "posterior_hit_rate_q50": float(q50),
                "posterior_hit_rate_q975": float(q975),
                "posterior_enrichment_q025": enrich_q025,
                "posterior_enrichment_q50": enrich_q50,
                "posterior_enrichment_q975": enrich_q975,
                "full_bayes_joint_tier": tier,
            }
        )

    out = pd.DataFrame(rows).sort_values("cohort").reset_index(drop=True)
    out.to_csv(args.output_dir / "full_bayes_endpoint_rows.csv", index=False)

    support_summary = (
        out.groupby("observed_support_eligible", as_index=False)
        .agg(
            n_cohorts=("cohort", "count"),
            observed_pass_rate=("observed_endpoint_pass", "mean"),
            mean_predictive_prob_joint_pass=("predictive_prob_joint_pass", "mean"),
            median_predictive_prob_joint_pass=("predictive_prob_joint_pass", "median"),
            mean_predictive_prob_enrichment_pass=("predictive_prob_enrichment_pass", "mean"),
        )
        .sort_values("observed_support_eligible", ascending=False)
        .reset_index(drop=True)
    )
    support_summary.to_csv(args.output_dir / "full_bayes_support_summary.csv", index=False)

    tier_summary = (
        out.groupby("full_bayes_joint_tier", as_index=False)
        .agg(
            n_cohorts=("cohort", "count"),
            mean_support_edges=("observed_support_edges", "mean"),
            observed_pass_rate=("observed_endpoint_pass", "mean"),
            mean_predictive_prob_joint_pass=("predictive_prob_joint_pass", "mean"),
        )
        .sort_values("full_bayes_joint_tier")
        .reset_index(drop=True)
    )
    tier_summary.to_csv(args.output_dir / "full_bayes_tier_summary.csv", index=False)

    overall = pd.DataFrame(
        [
            {
                "group_mode": args.group_mode,
                "n_cohorts": int(len(out)),
                "n_observed_pass": int(out["observed_endpoint_pass"].sum()),
                "observed_pass_rate": float(out["observed_endpoint_pass"].mean()),
                "n_powered_joint_confirmed_pass": int((out["full_bayes_joint_tier"] == "POWERED_JOINT_CONFIRMED_PASS").sum()),
                "n_powered_joint_confirmed_fail": int((out["full_bayes_joint_tier"] == "POWERED_JOINT_CONFIRMED_FAIL").sum()),
                "n_underpowered_joint_positive_signal": int((out["full_bayes_joint_tier"] == "UNDERPOWERED_JOINT_POSITIVE_SIGNAL").sum()),
                "n_underpowered_joint_negative_signal": int((out["full_bayes_joint_tier"] == "UNDERPOWERED_JOINT_NEGATIVE_SIGNAL").sum()),
            }
        ]
    )
    overall.to_csv(args.output_dir / "full_bayes_overall_summary.csv", index=False)

    if args.hier_rows.exists():
        hier = pd.read_csv(args.hier_rows, low_memory=False)
        cmp_hier = out.merge(
            hier[["cohort", "posterior_prob_hit_rate_ge_threshold", "power_aware_tier"]].rename(
                columns={
                    "posterior_prob_hit_rate_ge_threshold": "hier_posterior_prob_threshold",
                    "power_aware_tier": "hier_tier",
                }
            ),
            on="cohort",
            how="left",
        )
        cmp_hier["delta_fullbayes_joint_minus_hier_threshold_prob"] = (
            cmp_hier["predictive_prob_joint_pass"] - cmp_hier["hier_posterior_prob_threshold"]
        )
        cmp_hier.to_csv(args.output_dir / "full_bayes_vs_hierarchical_comparison.csv", index=False)

    if args.nonhier_rows.exists():
        nonhier = pd.read_csv(args.nonhier_rows, low_memory=False)
        cmp_nonhier = out.merge(
            nonhier[["cohort", "posterior_prob_hit_rate_ge_threshold", "power_aware_tier"]].rename(
                columns={
                    "posterior_prob_hit_rate_ge_threshold": "nonhier_posterior_prob_threshold",
                    "power_aware_tier": "nonhier_tier",
                }
            ),
            on="cohort",
            how="left",
        )
        cmp_nonhier["delta_fullbayes_joint_minus_nonhier_threshold_prob"] = (
            cmp_nonhier["predictive_prob_joint_pass"] - cmp_nonhier["nonhier_posterior_prob_threshold"]
        )
        cmp_nonhier.to_csv(args.output_dir / "full_bayes_vs_nonhierarchical_comparison.csv", index=False)

    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    colors = out["observed_endpoint_pass"].map({True: "#1b9e77", False: "#d95f02"})
    ax.scatter(
        out["observed_support_edges"],
        out["predictive_prob_joint_pass"],
        c=colors,
        s=90,
        edgecolors="white",
        linewidths=0.8,
    )
    for _, r in out.iterrows():
        ax.annotate(str(r["cohort"]), (r["observed_support_edges"], r["predictive_prob_joint_pass"]), fontsize=8)
    ax.axvline(float(args.endpoint_min_support), color="#444444", linestyle="--", linewidth=1.0, alpha=0.75)
    ax.axhline(float(args.high_confidence), color="#2166ac", linestyle="--", linewidth=1.0, alpha=0.75)
    ax.axhline(float(args.low_confidence), color="#b2182b", linestyle="--", linewidth=1.0, alpha=0.75)
    ax.set_xlabel("External candidate-space support (n reference edges)")
    ax.set_ylabel("Full-Bayes predictive P(joint endpoint pass)")
    ax.set_title("Full-Bayes Joint Endpoint Confidence Across Cohorts")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(args.output_dir / "full_bayes_support_vs_joint_probability.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    print("Full hierarchical-Bayes endpoint tiering complete.")
    print(overall.to_string(index=False))


if __name__ == "__main__":
    main()
