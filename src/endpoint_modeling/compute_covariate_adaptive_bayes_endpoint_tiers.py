#!/usr/bin/env python3
"""Compute covariate-aware adaptive Bayesian endpoint tiers for multi-cohort replication.

This phase extends grid-based full-Bayes endpoint modeling with an adaptive MCMC
hierarchical logistic model that includes cohort covariates. The goal is to test
whether support geometry and selection size explain residual endpoint uncertainty.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import expit
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
        "--full-bayes-rows",
        type=Path,
        default=Path(
            "subproject_24_multireference_latent_truth_model/implementation/outputs/"
            "top_tier_readiness_v6/full_bayes_endpoint_tiering_v1/full_bayes_endpoint_rows.csv"
        ),
    )
    parser.add_argument("--endpoint-min-enrichment", type=float, default=1.5)
    parser.add_argument("--endpoint-max-p", type=float, default=0.05)
    parser.add_argument("--endpoint-min-support", type=float, default=10.0)
    parser.add_argument("--prior-sd-beta", type=float, default=2.5)
    parser.add_argument("--prior-logsigma-mean", type=float, default=-0.5)
    parser.add_argument("--prior-logsigma-sd", type=float, default=0.8)
    parser.add_argument("--mcmc-steps", type=int, default=30000)
    parser.add_argument("--burn-in", type=int, default=6000)
    parser.add_argument("--thin", type=int, default=6)
    parser.add_argument("--proposal-sd-eta", type=float, default=0.35)
    parser.add_argument("--proposal-sd-beta", type=float, default=0.08)
    parser.add_argument("--proposal-sd-logsigma", type=float, default=0.07)
    parser.add_argument("--high-confidence", type=float, default=0.95)
    parser.add_argument("--low-confidence", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=73)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "subproject_24_multireference_latent_truth_model/implementation/outputs/"
            "top_tier_readiness_v6/covariate_adaptive_bayes_endpoint_tiering_v1"
        ),
    )
    return parser.parse_args()


def _to_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "t", "yes", "y"}


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


def _safe_zscore(values: np.ndarray) -> Tuple[np.ndarray, float, float]:
    mean = float(np.mean(values))
    std = float(np.std(values, ddof=0))
    if not np.isfinite(std) or std < _EPS:
        std = 1.0
    return (values - mean) / std, mean, std


def _build_design_matrix(base: pd.DataFrame) -> Tuple[np.ndarray, List[str], Dict[str, float]]:
    support_log = np.log1p(base["observed_support_edges"].to_numpy(dtype=float))
    selected_log = np.log1p(base["n_selected"].to_numpy(dtype=float))
    support_log_z, support_log_mean, support_log_std = _safe_zscore(support_log)
    selected_log_z, selected_log_mean, selected_log_std = _safe_zscore(selected_log)
    support_eligible = base["observed_support_eligible"].astype(int).to_numpy(dtype=float)

    cols = [np.ones(len(base), dtype=float)]
    names = ["intercept"]
    if float(np.std(support_eligible, ddof=0)) > _EPS:
        cols.append(support_eligible)
        names.append("support_eligible_binary")
    if float(np.std(support_log_z, ddof=0)) > _EPS:
        cols.append(support_log_z)
        names.append("log1p_support_edges_z")
    if float(np.std(selected_log_z, ddof=0)) > _EPS:
        cols.append(selected_log_z)
        names.append("log1p_n_selected_z")

    X = np.column_stack(cols)
    stats = {
        "support_log_mean": support_log_mean,
        "support_log_std": support_log_std,
        "selected_log_mean": selected_log_mean,
        "selected_log_std": selected_log_std,
        "include_support_eligible_binary": bool("support_eligible_binary" in names),
        "include_log1p_support_edges_z": bool("log1p_support_edges_z" in names),
        "include_log1p_n_selected_z": bool("log1p_n_selected_z" in names),
    }
    return X, names, stats


def _log_binomial_eta(h: np.ndarray, n: np.ndarray, eta: np.ndarray) -> np.ndarray:
    return h * eta - n * np.logaddexp(0.0, eta)


def _log_eta_prior(eta: np.ndarray, mu: np.ndarray, sigma: float) -> float:
    if sigma <= 0:
        return -np.inf
    z = (eta - mu) / sigma
    return float(-0.5 * np.sum(z * z) - len(eta) * np.log(sigma))


def _log_beta_prior(beta: np.ndarray, sd: float) -> float:
    if sd <= 0:
        return -np.inf
    z = beta / sd
    return float(-0.5 * np.sum(z * z))


def _log_logsigma_prior(log_sigma: float, mean: float, sd: float) -> float:
    if sd <= 0:
        return -np.inf
    z = (log_sigma - mean) / sd
    return float(-0.5 * z * z)


def _fit_adaptive_model(
    *,
    hits: np.ndarray,
    n_selected: np.ndarray,
    X: np.ndarray,
    args: argparse.Namespace,
) -> Dict[str, object]:
    rng = np.random.default_rng(int(args.seed))
    n_cohorts, n_coef = X.shape

    empirical_rate = (hits + 0.5) / (n_selected + 1.0)
    eta = np.log(np.clip(empirical_rate, _EPS, 1.0 - _EPS) / np.clip(1.0 - empirical_rate, _EPS, 1.0))
    beta = np.linalg.lstsq(X, eta, rcond=None)[0]
    resid = eta - X @ beta
    sigma = float(np.clip(np.std(resid, ddof=0), 0.15, 3.0))
    log_sigma = float(np.log(sigma))

    keep_eta: List[np.ndarray] = []
    keep_beta: List[np.ndarray] = []
    keep_sigma: List[float] = []

    eta_accept = 0
    eta_total = 0
    beta_accept = 0
    beta_total = 0
    sigma_accept = 0
    sigma_total = 0

    for step in range(int(args.mcmc_steps)):
        sigma = float(np.exp(log_sigma))
        mu = X @ beta

        for idx in range(n_cohorts):
            current_eta = float(eta[idx])
            prop_eta = current_eta + float(rng.normal(scale=float(args.proposal_sd_eta)))

            current_lp = (
                float(_log_binomial_eta(hits[idx : idx + 1], n_selected[idx : idx + 1], np.array([current_eta]))[0])
                + _log_eta_prior(np.array([current_eta]), np.array([mu[idx]]), sigma)
            )
            prop_lp = (
                float(_log_binomial_eta(hits[idx : idx + 1], n_selected[idx : idx + 1], np.array([prop_eta]))[0])
                + _log_eta_prior(np.array([prop_eta]), np.array([mu[idx]]), sigma)
            )
            eta_total += 1
            if np.log(rng.uniform()) < (prop_lp - current_lp):
                eta[idx] = prop_eta
                eta_accept += 1

        prop_beta = beta + rng.normal(scale=float(args.proposal_sd_beta), size=n_coef)
        current_lp = (
            _log_eta_prior(eta, X @ beta, sigma)
            + _log_beta_prior(beta, float(args.prior_sd_beta))
        )
        prop_lp = (
            _log_eta_prior(eta, X @ prop_beta, sigma)
            + _log_beta_prior(prop_beta, float(args.prior_sd_beta))
        )
        beta_total += 1
        if np.log(rng.uniform()) < (prop_lp - current_lp):
            beta = prop_beta
            beta_accept += 1

        prop_log_sigma = log_sigma + float(rng.normal(scale=float(args.proposal_sd_logsigma)))
        prop_sigma = float(np.exp(prop_log_sigma))
        current_lp = (
            _log_eta_prior(eta, X @ beta, sigma)
            + _log_logsigma_prior(log_sigma, float(args.prior_logsigma_mean), float(args.prior_logsigma_sd))
        )
        prop_lp = (
            _log_eta_prior(eta, X @ beta, prop_sigma)
            + _log_logsigma_prior(prop_log_sigma, float(args.prior_logsigma_mean), float(args.prior_logsigma_sd))
        )
        sigma_total += 1
        if np.log(rng.uniform()) < (prop_lp - current_lp):
            log_sigma = prop_log_sigma
            sigma_accept += 1

        after_burn = step >= int(args.burn_in)
        thin_ok = ((step - int(args.burn_in)) % int(args.thin) == 0) if after_burn else False
        if after_burn and thin_ok:
            keep_eta.append(eta.copy())
            keep_beta.append(beta.copy())
            keep_sigma.append(float(np.exp(log_sigma)))

    if not keep_eta:
        raise RuntimeError("MCMC retained zero draws; increase mcmc-steps or reduce burn-in/thin.")

    draws_eta = np.vstack(keep_eta)
    draws_beta = np.vstack(keep_beta)
    draws_sigma = np.asarray(keep_sigma, dtype=float)

    return {
        "eta": draws_eta,
        "beta": draws_beta,
        "sigma": draws_sigma,
        "diagnostics": {
            "mcmc_steps": int(args.mcmc_steps),
            "burn_in": int(args.burn_in),
            "thin": int(args.thin),
            "retained_draws": int(draws_eta.shape[0]),
            "eta_acceptance_rate": float(eta_accept / max(eta_total, 1)),
            "beta_acceptance_rate": float(beta_accept / max(beta_total, 1)),
            "logsigma_acceptance_rate": float(sigma_accept / max(sigma_total, 1)),
        },
    }


def _tier_from_joint_prob(*, support_eligible: bool, joint_prob: float, high_conf: float, low_conf: float) -> str:
    if support_eligible:
        if joint_prob >= high_conf:
            return "POWERED_ADAPTIVE_JOINT_CONFIRMED_PASS"
        if joint_prob <= low_conf:
            return "POWERED_ADAPTIVE_JOINT_CONFIRMED_FAIL"
        return "POWERED_ADAPTIVE_JOINT_AMBIGUOUS"

    if joint_prob >= high_conf:
        return "UNDERPOWERED_ADAPTIVE_JOINT_POSITIVE_SIGNAL"
    if joint_prob <= low_conf:
        return "UNDERPOWERED_ADAPTIVE_JOINT_NEGATIVE_SIGNAL"
    return "UNDERPOWERED_ADAPTIVE_JOINT_AMBIGUOUS"


def _posterior_coeff_summary(beta_draws: np.ndarray, sigma_draws: np.ndarray, coef_names: List[str]) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for idx, name in enumerate(coef_names):
        vals = beta_draws[:, idx]
        q025, q50, q975 = np.quantile(vals, [0.025, 0.5, 0.975])
        rows.append(
            {
                "parameter": name,
                "mean": float(np.mean(vals)),
                "sd": float(np.std(vals, ddof=0)),
                "q025": float(q025),
                "q50": float(q50),
                "q975": float(q975),
                "prob_positive": float(np.mean(vals > 0)),
            }
        )

    q025, q50, q975 = np.quantile(sigma_draws, [0.025, 0.5, 0.975])
    rows.append(
        {
            "parameter": "cohort_random_effect_sigma",
            "mean": float(np.mean(sigma_draws)),
            "sd": float(np.std(sigma_draws, ddof=0)),
            "q025": float(q025),
            "q50": float(q50),
            "q975": float(q975),
            "prob_positive": 1.0,
        }
    )
    return pd.DataFrame(rows)


def _plot_support_vs_joint(out: pd.DataFrame, args: argparse.Namespace) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    colors = out["observed_endpoint_pass"].map({True: "#1b9e77", False: "#d95f02"})
    ax.scatter(
        out["observed_support_edges"],
        out["predictive_prob_joint_pass"],
        c=colors,
        s=95,
        edgecolors="white",
        linewidths=0.8,
    )
    for _, r in out.iterrows():
        ax.annotate(str(r["cohort"]), (r["observed_support_edges"], r["predictive_prob_joint_pass"]), fontsize=8)
    ax.axvline(float(args.endpoint_min_support), color="#444444", linestyle="--", linewidth=1.0, alpha=0.8)
    ax.axhline(float(args.high_confidence), color="#2166ac", linestyle="--", linewidth=1.0, alpha=0.8)
    ax.axhline(float(args.low_confidence), color="#b2182b", linestyle="--", linewidth=1.0, alpha=0.8)
    ax.set_xlabel("External candidate-space support (n reference edges)")
    ax.set_ylabel("Adaptive-Bayes predictive P(joint endpoint pass)")
    ax.set_title("Covariate-Aware Adaptive Bayesian Joint Endpoint Confidence")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(args.output_dir / "adaptive_bayes_support_vs_joint_probability.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_coefficients(coeff: pd.DataFrame, out_path: Path) -> None:
    body = coeff[coeff["parameter"] != "cohort_random_effect_sigma"].copy()
    if body.empty:
        return
    body = body.sort_values("mean").reset_index(drop=True)
    y = np.arange(len(body))
    left = body["mean"].to_numpy(dtype=float) - body["q025"].to_numpy(dtype=float)
    right = body["q975"].to_numpy(dtype=float) - body["mean"].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    ax.errorbar(
        body["mean"].to_numpy(dtype=float),
        y,
        xerr=np.vstack([left, right]),
        fmt="o",
        color="#2b8cbe",
        ecolor="#2b8cbe",
        elinewidth=1.3,
        capsize=3,
    )
    ax.axvline(0.0, color="#444444", linestyle="--", linewidth=1.0, alpha=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(body["parameter"].tolist())
    ax.set_xlabel("Posterior coefficient (95% credible interval)")
    ax.set_title("Adaptive Bayesian Covariate Effects")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = _parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    base = _build_base_rows(args)
    X, coef_names, design_stats = _build_design_matrix(base)
    hits = base["observed_hits"].to_numpy(dtype=float)
    n_selected = base["n_selected"].to_numpy(dtype=float)

    fit = _fit_adaptive_model(hits=hits, n_selected=n_selected, X=X, args=args)
    eta_draws = np.asarray(fit["eta"], dtype=float)
    beta_draws = np.asarray(fit["beta"], dtype=float)
    sigma_draws = np.asarray(fit["sigma"], dtype=float)

    diagnostics = pd.DataFrame(
        [
            {
                **fit["diagnostics"],
                **design_stats,
                "n_cohorts": int(len(base)),
                "n_coefficients": int(X.shape[1]),
            }
        ]
    )
    diagnostics.to_csv(args.output_dir / "adaptive_bayes_mcmc_diagnostics.csv", index=False)

    coeff = _posterior_coeff_summary(beta_draws, sigma_draws, coef_names)
    coeff.to_csv(args.output_dir / "adaptive_bayes_posterior_coefficients.csv", index=False)

    rng = np.random.default_rng(int(args.seed) + 9000)
    rows: List[Dict[str, object]] = []

    for idx, row in enumerate(base.itertuples(index=False)):
        theta = np.clip(expit(eta_draws[:, idx]), _EPS, 1.0 - _EPS)
        n_sel = int(row.n_selected)
        pred_hits = rng.binomial(n=n_sel, p=theta)

        min_hits_for_enrichment = int(row.min_hits_for_enrichment)
        n_universe = int(row.n_universe)
        k_ref = int(row.n_reference_edges_in_candidate_pool)
        hit_grid = np.arange(n_sel + 1)
        pvals = hypergeom.sf(hit_grid - 1, M=n_universe, n=k_ref, N=n_sel)
        pred_pvals = pvals[pred_hits]

        posterior_prob_threshold = float(np.mean(theta >= float(row.endpoint_hit_rate_threshold)))
        predictive_prob_enrich = float(np.mean(pred_hits >= min_hits_for_enrichment))
        predictive_prob_joint = float(
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
            joint_prob=predictive_prob_joint,
            high_conf=float(args.high_confidence),
            low_conf=float(args.low_confidence),
        )

        rows.append(
            {
                **row._asdict(),
                "posterior_prob_enrichment_threshold": posterior_prob_threshold,
                "predictive_prob_enrichment_pass": predictive_prob_enrich,
                "predictive_prob_joint_pass": predictive_prob_joint,
                "posterior_hit_rate_q025": float(q025),
                "posterior_hit_rate_q50": float(q50),
                "posterior_hit_rate_q975": float(q975),
                "posterior_enrichment_q025": enrich_q025,
                "posterior_enrichment_q50": enrich_q50,
                "posterior_enrichment_q975": enrich_q975,
                "adaptive_bayes_joint_tier": tier,
                **{f"x_{coef_names[j]}": float(X[idx, j]) for j in range(len(coef_names))},
            }
        )

    out = pd.DataFrame(rows).sort_values("cohort").reset_index(drop=True)
    out.to_csv(args.output_dir / "adaptive_bayes_endpoint_rows.csv", index=False)

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
    support_summary.to_csv(args.output_dir / "adaptive_bayes_support_summary.csv", index=False)

    tier_summary = (
        out.groupby("adaptive_bayes_joint_tier", as_index=False)
        .agg(
            n_cohorts=("cohort", "count"),
            mean_support_edges=("observed_support_edges", "mean"),
            observed_pass_rate=("observed_endpoint_pass", "mean"),
            mean_predictive_prob_joint_pass=("predictive_prob_joint_pass", "mean"),
        )
        .sort_values("adaptive_bayes_joint_tier")
        .reset_index(drop=True)
    )
    tier_summary.to_csv(args.output_dir / "adaptive_bayes_tier_summary.csv", index=False)

    overall = pd.DataFrame(
        [
            {
                "n_cohorts": int(len(out)),
                "n_observed_pass": int(out["observed_endpoint_pass"].sum()),
                "observed_pass_rate": float(out["observed_endpoint_pass"].mean()),
                "n_powered_confirmed_pass": int((out["adaptive_bayes_joint_tier"] == "POWERED_ADAPTIVE_JOINT_CONFIRMED_PASS").sum()),
                "n_powered_confirmed_fail": int((out["adaptive_bayes_joint_tier"] == "POWERED_ADAPTIVE_JOINT_CONFIRMED_FAIL").sum()),
                "n_underpowered_positive_signal": int((out["adaptive_bayes_joint_tier"] == "UNDERPOWERED_ADAPTIVE_JOINT_POSITIVE_SIGNAL").sum()),
                "n_underpowered_negative_signal": int((out["adaptive_bayes_joint_tier"] == "UNDERPOWERED_ADAPTIVE_JOINT_NEGATIVE_SIGNAL").sum()),
            }
        ]
    )
    overall.to_csv(args.output_dir / "adaptive_bayes_overall_summary.csv", index=False)

    if args.full_bayes_rows.exists():
        full = pd.read_csv(args.full_bayes_rows, low_memory=False)
        cmp = out.merge(
            full[["cohort", "predictive_prob_joint_pass", "full_bayes_joint_tier"]].rename(
                columns={
                    "predictive_prob_joint_pass": "full_bayes_predictive_prob_joint_pass",
                }
            ),
            on="cohort",
            how="left",
        )
        cmp["delta_adaptive_minus_full_bayes_joint_prob"] = (
            cmp["predictive_prob_joint_pass"] - cmp["full_bayes_predictive_prob_joint_pass"]
        )
        cmp.to_csv(args.output_dir / "adaptive_bayes_vs_full_bayes_comparison.csv", index=False)

    _plot_support_vs_joint(out, args)
    _plot_coefficients(coeff, args.output_dir / "adaptive_bayes_coefficient_forest.png")

    print("Covariate-aware adaptive Bayesian endpoint tiering complete.")
    print(overall.to_string(index=False))


if __name__ == "__main__":
    main()
