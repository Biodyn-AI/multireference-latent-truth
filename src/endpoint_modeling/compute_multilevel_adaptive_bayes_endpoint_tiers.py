#!/usr/bin/env python3
"""Compute multilevel adaptive Bayesian endpoint tiers with family partial pooling.

Phase XXVII extension over Phase XXVI:
1) Adds assay-family random effects with hierarchical shrinkage.
2) Adds richer support covariates (nonlinear term) and optional cohort metadata.
3) Exports posterior predictive stress diagnostics.
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
        "--adaptive-rows",
        type=Path,
        default=Path(
            "subproject_24_multireference_latent_truth_model/implementation/outputs/"
            "top_tier_readiness_v6/covariate_adaptive_bayes_endpoint_tiering_v1/"
            "adaptive_bayes_endpoint_rows.csv"
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
    parser.add_argument(
        "--cohort-metadata",
        type=Path,
        default=None,
        help=(
            "Optional cohort metadata CSV with 'cohort' column plus numeric/categorical "
            "covariates and optional 'assay_family'."
        ),
    )
    parser.add_argument("--endpoint-min-enrichment", type=float, default=1.5)
    parser.add_argument("--endpoint-max-p", type=float, default=0.05)
    parser.add_argument("--endpoint-min-support", type=float, default=10.0)
    parser.add_argument("--prior-sd-beta", type=float, default=2.5)
    parser.add_argument("--prior-logsigma-mean", type=float, default=-0.5)
    parser.add_argument("--prior-logsigma-sd", type=float, default=0.8)
    parser.add_argument("--prior-logtau-mean", type=float, default=-0.7)
    parser.add_argument("--prior-logtau-sd", type=float, default=0.9)
    parser.add_argument("--mcmc-steps", type=int, default=36000)
    parser.add_argument("--burn-in", type=int, default=8000)
    parser.add_argument("--thin", type=int, default=7)
    parser.add_argument("--proposal-sd-eta", type=float, default=0.34)
    parser.add_argument("--proposal-sd-beta", type=float, default=0.075)
    parser.add_argument("--proposal-sd-family", type=float, default=0.09)
    parser.add_argument("--proposal-sd-logsigma", type=float, default=0.07)
    parser.add_argument("--proposal-sd-logtau", type=float, default=0.08)
    parser.add_argument("--high-confidence", type=float, default=0.95)
    parser.add_argument("--low-confidence", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=89)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "subproject_24_multireference_latent_truth_model/implementation/outputs/"
            "top_tier_readiness_v7/multilevel_adaptive_bayes_endpoint_tiering_v1"
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


def _infer_assay_family(cohort: str) -> str:
    c = str(cohort).lower()
    if c.startswith("checkpoint_"):
        return "checkpoint_replication"
    if c in {"adamson", "dixit", "dixit7", "shifrut"}:
        return "crispr_perturbseq"
    return "other"


def _infer_cohort_domain(cohort: str) -> str:
    c = str(cohort).lower()
    if c.startswith("checkpoint_"):
        return "checkpoint"
    if c in {"adamson", "dixit", "dixit7", "shifrut"}:
        return "perturbseq"
    return "other"


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
                "assay_family": _infer_assay_family(cohort),
                "cohort_domain": _infer_cohort_domain(cohort),
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

    base = pd.DataFrame(rows).sort_values("cohort").reset_index(drop=True)
    if args.cohort_metadata is not None:
        meta = pd.read_csv(args.cohort_metadata, low_memory=False)
        if "cohort" not in meta.columns:
            raise ValueError(f"Metadata must include 'cohort' column: {args.cohort_metadata}")
        meta = meta.copy()
        meta["cohort"] = meta["cohort"].astype(str)
        base = base.merge(meta, on="cohort", how="left", suffixes=("", "_meta"))
        if "assay_family_meta" in base.columns:
            base["assay_family"] = base["assay_family_meta"].fillna(base["assay_family"]).astype(str)
            base = base.drop(columns=["assay_family_meta"])
        if "assay_family" in base.columns:
            base["assay_family"] = base["assay_family"].fillna(base["cohort"].map(_infer_assay_family)).astype(str)
    return base


def _safe_zscore(values: np.ndarray) -> Tuple[np.ndarray, float, float]:
    mean = float(np.mean(values))
    std = float(np.std(values, ddof=0))
    if not np.isfinite(std) or std < _EPS:
        std = 1.0
    return (values - mean) / std, mean, std


def _build_design_matrix(base: pd.DataFrame) -> Tuple[np.ndarray, List[str], Dict[str, object]]:
    support_log = np.log1p(base["observed_support_edges"].to_numpy(dtype=float))
    support_log_z, support_log_mean, support_log_std = _safe_zscore(support_log)
    support_bin = base["observed_support_eligible"].astype(int).to_numpy(dtype=float)
    support_log_sq = support_log_z * support_log_z

    cols: List[np.ndarray] = [np.ones(len(base), dtype=float)]
    names: List[str] = ["intercept"]

    def _add(col: np.ndarray, name: str) -> None:
        if float(np.std(col, ddof=0)) > _EPS:
            cols.append(col.astype(float))
            names.append(name)

    _add(support_bin, "support_eligible_binary")
    _add(support_log_z, "log1p_support_edges_z")
    _add(support_log_sq, "log1p_support_edges_z_sq")

    known_base_cols = {
        "cohort",
        "selected_policy",
        "assay_family",
        "cohort_domain",
        "observed_endpoint_pass",
        "observed_support_eligible",
        "observed_support_edges",
        "observed_hits",
        "observed_enrichment",
        "observed_p_value",
        "n_selected",
        "n_reference_edges_in_candidate_pool",
        "n_universe",
        "expected_hits_random",
        "random_hit_rate",
        "endpoint_min_enrichment",
        "endpoint_max_p",
        "endpoint_min_support",
        "endpoint_hit_rate_threshold",
        "min_hits_for_enrichment",
    }
    extra_cols = [c for c in base.columns if c not in known_base_cols]

    for col_name in extra_cols:
        s = base[col_name]
        if pd.api.types.is_bool_dtype(s):
            _add(s.astype(int).to_numpy(dtype=float), f"meta_{col_name}")
        elif pd.api.types.is_numeric_dtype(s):
            vals = s.to_numpy(dtype=float)
            if np.isfinite(vals).sum() >= 2:
                fill = np.nanmedian(vals)
            else:
                fill = 0.0
            vals = np.where(np.isfinite(vals), vals, fill)
            vals_z, _, _ = _safe_zscore(vals)
            _add(vals_z, f"meta_{col_name}_z")
        else:
            filled = s.fillna("unknown").astype(str)
            if filled.nunique(dropna=False) <= 1:
                continue
            dummies = pd.get_dummies(filled, prefix=f"meta_{col_name}", drop_first=True)
            for dcol in dummies.columns:
                _add(dummies[dcol].to_numpy(dtype=float), dcol)

    X = np.column_stack(cols)
    stats: Dict[str, object] = {
        "support_log_mean": support_log_mean,
        "support_log_std": support_log_std,
        "n_coefficients": int(len(names)),
    }
    return X, names, stats


def _build_family_index(base: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
    fam = base["assay_family"].fillna("unknown").astype(str)
    labels = sorted(fam.unique().tolist())
    mapping = {name: idx for idx, name in enumerate(labels)}
    idx = fam.map(mapping).to_numpy(dtype=int)
    return idx, labels


def _log_binomial_eta(h: np.ndarray, n: np.ndarray, eta: np.ndarray) -> np.ndarray:
    return h * eta - n * np.logaddexp(0.0, eta)


def _log_norm_centered(x: np.ndarray, mean: np.ndarray, sd: float) -> float:
    if sd <= 0:
        return -np.inf
    z = (x - mean) / sd
    return float(-0.5 * np.sum(z * z) - len(x) * np.log(sd))


def _fit_multilevel_model(
    *,
    hits: np.ndarray,
    n_selected: np.ndarray,
    X: np.ndarray,
    family_idx: np.ndarray,
    family_labels: List[str],
    args: argparse.Namespace,
) -> Dict[str, object]:
    rng = np.random.default_rng(int(args.seed))
    n_cohorts, n_coef = X.shape
    n_families = len(family_labels)

    empirical_rate = (hits + 0.5) / (n_selected + 1.0)
    eta = np.log(np.clip(empirical_rate, _EPS, 1.0 - _EPS) / np.clip(1.0 - empirical_rate, _EPS, 1.0))
    beta = np.linalg.lstsq(X, eta, rcond=None)[0]
    family_effects = np.zeros(n_families, dtype=float)
    resid = eta - X @ beta
    sigma = float(np.clip(np.std(resid, ddof=0), 0.12, 3.0))
    tau = float(np.clip(np.std(resid, ddof=0), 0.10, 2.0))
    log_sigma = float(np.log(sigma))
    log_tau = float(np.log(tau))

    keep_eta: List[np.ndarray] = []
    keep_beta: List[np.ndarray] = []
    keep_family: List[np.ndarray] = []
    keep_sigma: List[float] = []
    keep_tau: List[float] = []

    eta_accept = 0
    eta_total = 0
    beta_accept = 0
    beta_total = 0
    fam_accept = 0
    fam_total = 0
    sigma_accept = 0
    sigma_total = 0
    tau_accept = 0
    tau_total = 0

    for step in range(int(args.mcmc_steps)):
        sigma = float(np.exp(log_sigma))
        tau = float(np.exp(log_tau))
        mu = X @ beta + family_effects[family_idx]

        for i in range(n_cohorts):
            current = float(eta[i])
            prop = current + float(rng.normal(scale=float(args.proposal_sd_eta)))
            curr_lp = (
                float(_log_binomial_eta(hits[i : i + 1], n_selected[i : i + 1], np.array([current]))[0])
                + _log_norm_centered(np.array([current]), np.array([mu[i]]), sigma)
            )
            prop_lp = (
                float(_log_binomial_eta(hits[i : i + 1], n_selected[i : i + 1], np.array([prop]))[0])
                + _log_norm_centered(np.array([prop]), np.array([mu[i]]), sigma)
            )
            eta_total += 1
            if np.log(rng.uniform()) < (prop_lp - curr_lp):
                eta[i] = prop
                eta_accept += 1

        prop_beta = beta + rng.normal(scale=float(args.proposal_sd_beta), size=n_coef)
        curr_lp = (
            _log_norm_centered(eta, X @ beta + family_effects[family_idx], sigma)
            + _log_norm_centered(beta, np.zeros_like(beta), float(args.prior_sd_beta))
        )
        prop_lp = (
            _log_norm_centered(eta, X @ prop_beta + family_effects[family_idx], sigma)
            + _log_norm_centered(prop_beta, np.zeros_like(prop_beta), float(args.prior_sd_beta))
        )
        beta_total += 1
        if np.log(rng.uniform()) < (prop_lp - curr_lp):
            beta = prop_beta
            beta_accept += 1

        for f in range(n_families):
            current = float(family_effects[f])
            prop = current + float(rng.normal(scale=float(args.proposal_sd_family)))
            fam_mask = family_idx == f
            curr_mu = X[fam_mask] @ beta + current
            prop_mu = X[fam_mask] @ beta + prop
            curr_lp = _log_norm_centered(eta[fam_mask], curr_mu, sigma) + _log_norm_centered(
                np.array([current]), np.array([0.0]), tau
            )
            prop_lp = _log_norm_centered(eta[fam_mask], prop_mu, sigma) + _log_norm_centered(
                np.array([prop]), np.array([0.0]), tau
            )
            fam_total += 1
            if np.log(rng.uniform()) < (prop_lp - curr_lp):
                family_effects[f] = prop
                fam_accept += 1

        prop_log_sigma = log_sigma + float(rng.normal(scale=float(args.proposal_sd_logsigma)))
        prop_sigma = float(np.exp(prop_log_sigma))
        curr_lp = _log_norm_centered(eta, X @ beta + family_effects[family_idx], sigma) + _log_norm_centered(
            np.array([log_sigma]), np.array([float(args.prior_logsigma_mean)]), float(args.prior_logsigma_sd)
        )
        prop_lp = _log_norm_centered(eta, X @ beta + family_effects[family_idx], prop_sigma) + _log_norm_centered(
            np.array([prop_log_sigma]), np.array([float(args.prior_logsigma_mean)]), float(args.prior_logsigma_sd)
        )
        sigma_total += 1
        if np.log(rng.uniform()) < (prop_lp - curr_lp):
            log_sigma = prop_log_sigma
            sigma_accept += 1

        prop_log_tau = log_tau + float(rng.normal(scale=float(args.proposal_sd_logtau)))
        prop_tau = float(np.exp(prop_log_tau))
        curr_lp = _log_norm_centered(family_effects, np.zeros_like(family_effects), tau) + _log_norm_centered(
            np.array([log_tau]), np.array([float(args.prior_logtau_mean)]), float(args.prior_logtau_sd)
        )
        prop_lp = _log_norm_centered(family_effects, np.zeros_like(family_effects), prop_tau) + _log_norm_centered(
            np.array([prop_log_tau]), np.array([float(args.prior_logtau_mean)]), float(args.prior_logtau_sd)
        )
        tau_total += 1
        if np.log(rng.uniform()) < (prop_lp - curr_lp):
            log_tau = prop_log_tau
            tau_accept += 1

        after_burn = step >= int(args.burn_in)
        thin_ok = ((step - int(args.burn_in)) % int(args.thin) == 0) if after_burn else False
        if after_burn and thin_ok:
            keep_eta.append(eta.copy())
            keep_beta.append(beta.copy())
            keep_family.append(family_effects.copy())
            keep_sigma.append(float(np.exp(log_sigma)))
            keep_tau.append(float(np.exp(log_tau)))

    if not keep_eta:
        raise RuntimeError("MCMC retained zero draws; adjust mcmc-steps/burn-in/thin.")

    return {
        "eta": np.vstack(keep_eta),
        "beta": np.vstack(keep_beta),
        "family_effects": np.vstack(keep_family),
        "sigma": np.asarray(keep_sigma, dtype=float),
        "tau": np.asarray(keep_tau, dtype=float),
        "diagnostics": {
            "mcmc_steps": int(args.mcmc_steps),
            "burn_in": int(args.burn_in),
            "thin": int(args.thin),
            "retained_draws": int(len(keep_eta)),
            "eta_acceptance_rate": float(eta_accept / max(eta_total, 1)),
            "beta_acceptance_rate": float(beta_accept / max(beta_total, 1)),
            "family_acceptance_rate": float(fam_accept / max(fam_total, 1)),
            "logsigma_acceptance_rate": float(sigma_accept / max(sigma_total, 1)),
            "logtau_acceptance_rate": float(tau_accept / max(tau_total, 1)),
            "n_families": int(n_families),
            "n_coefficients": int(n_coef),
        },
    }


def _posterior_summary(param_draws: np.ndarray, names: List[str], kind: str) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for i, name in enumerate(names):
        vals = param_draws[:, i]
        q025, q50, q975 = np.quantile(vals, [0.025, 0.5, 0.975])
        rows.append(
            {
                "kind": kind,
                "parameter": name,
                "mean": float(np.mean(vals)),
                "sd": float(np.std(vals, ddof=0)),
                "q025": float(q025),
                "q50": float(q50),
                "q975": float(q975),
                "prob_positive": float(np.mean(vals > 0)),
            }
        )
    return pd.DataFrame(rows)


def _tier_from_joint_prob(*, support_eligible: bool, joint_prob: float, high_conf: float, low_conf: float) -> str:
    if support_eligible:
        if joint_prob >= high_conf:
            return "POWERED_MULTILEVEL_JOINT_CONFIRMED_PASS"
        if joint_prob <= low_conf:
            return "POWERED_MULTILEVEL_JOINT_CONFIRMED_FAIL"
        return "POWERED_MULTILEVEL_JOINT_AMBIGUOUS"
    if joint_prob >= high_conf:
        return "UNDERPOWERED_MULTILEVEL_JOINT_POSITIVE_SIGNAL"
    if joint_prob <= low_conf:
        return "UNDERPOWERED_MULTILEVEL_JOINT_NEGATIVE_SIGNAL"
    return "UNDERPOWERED_MULTILEVEL_JOINT_AMBIGUOUS"


def _compute_ppc(
    *,
    eta_draws: np.ndarray,
    base: pd.DataFrame,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(int(seed))
    theta = np.clip(expit(eta_draws), _EPS, 1.0 - _EPS)
    n_sel = base["n_selected"].to_numpy(dtype=int)
    obs_hits = base["observed_hits"].to_numpy(dtype=int)
    min_hits = base["min_hits_for_enrichment"].to_numpy(dtype=int)
    n_universe = base["n_universe"].to_numpy(dtype=int)
    k_ref = base["n_reference_edges_in_candidate_pool"].to_numpy(dtype=int)
    max_p = base["endpoint_max_p"].to_numpy(dtype=float)

    n_draws = theta.shape[0]
    rep_hits = np.zeros((n_draws, len(base)), dtype=int)
    for i in range(len(base)):
        rep_hits[:, i] = rng.binomial(n=n_sel[i], p=theta[:, i])

    def _joint_pass_counts(hit_mat: np.ndarray) -> np.ndarray:
        out = np.zeros(hit_mat.shape[0], dtype=int)
        for i in range(len(base)):
            hit_grid = np.arange(n_sel[i] + 1)
            pvals = hypergeom.sf(hit_grid - 1, M=n_universe[i], n=k_ref[i], N=n_sel[i])
            p_for_draw = pvals[hit_mat[:, i]]
            out += ((hit_mat[:, i] >= min_hits[i]) & (p_for_draw <= max_p[i])).astype(int)
        return out

    obs_rate = obs_hits / np.clip(n_sel, 1, None)
    rep_rate = rep_hits / np.clip(n_sel[None, :], 1, None)

    obs_stats = {
        "mean_hit_rate": float(np.mean(obs_rate)),
        "var_hit_rate": float(np.var(obs_rate, ddof=0)),
        "max_hits": float(np.max(obs_hits)),
        "n_joint_passes": float(np.sum(base["observed_endpoint_pass"].astype(bool))),
    }

    rep_stats = {
        "mean_hit_rate": np.mean(rep_rate, axis=1),
        "var_hit_rate": np.var(rep_rate, axis=1),
        "max_hits": np.max(rep_hits, axis=1).astype(float),
        "n_joint_passes": _joint_pass_counts(rep_hits).astype(float),
    }

    rows: List[Dict[str, object]] = []
    for metric, obs in obs_stats.items():
        vals = rep_stats[metric]
        rows.append(
            {
                "metric": metric,
                "observed_value": float(obs),
                "rep_mean": float(np.mean(vals)),
                "rep_sd": float(np.std(vals, ddof=0)),
                "rep_q025": float(np.quantile(vals, 0.025)),
                "rep_q50": float(np.quantile(vals, 0.5)),
                "rep_q975": float(np.quantile(vals, 0.975)),
                "tail_prob_ge_observed": float(np.mean(vals >= obs)),
                "tail_prob_le_observed": float(np.mean(vals <= obs)),
            }
        )

    # Per-cohort coverage for observed hits against predictive intervals.
    for i, cohort in enumerate(base["cohort"].astype(str).tolist()):
        q025 = float(np.quantile(rep_hits[:, i], 0.025))
        q975 = float(np.quantile(rep_hits[:, i], 0.975))
        obs = float(obs_hits[i])
        rows.append(
            {
                "metric": f"coverage_hits_{cohort}",
                "observed_value": obs,
                "rep_mean": float(np.mean(rep_hits[:, i])),
                "rep_sd": float(np.std(rep_hits[:, i], ddof=0)),
                "rep_q025": q025,
                "rep_q50": float(np.quantile(rep_hits[:, i], 0.5)),
                "rep_q975": q975,
                "tail_prob_ge_observed": float(np.mean(rep_hits[:, i] >= obs)),
                "tail_prob_le_observed": float(np.mean(rep_hits[:, i] <= obs)),
            }
        )

    return pd.DataFrame(rows)


def _plot_support_vs_joint(out: pd.DataFrame, endpoint_min_support: float, high_conf: float, low_conf: float, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.7, 5.1))
    colors = out["observed_endpoint_pass"].map({True: "#1b9e77", False: "#d95f02"})
    ax.scatter(
        out["observed_support_edges"],
        out["predictive_prob_joint_pass"],
        c=colors,
        s=95,
        edgecolors="white",
        linewidths=0.85,
    )
    for _, r in out.iterrows():
        ax.annotate(str(r["cohort"]), (r["observed_support_edges"], r["predictive_prob_joint_pass"]), fontsize=8)
    ax.axvline(float(endpoint_min_support), color="#444444", linestyle="--", linewidth=1.0, alpha=0.8)
    ax.axhline(float(high_conf), color="#2166ac", linestyle="--", linewidth=1.0, alpha=0.8)
    ax.axhline(float(low_conf), color="#b2182b", linestyle="--", linewidth=1.0, alpha=0.8)
    ax.set_xlabel("External candidate-space support (n reference edges)")
    ax.set_ylabel("Multilevel adaptive P(joint endpoint pass)")
    ax.set_title("Multilevel Adaptive Bayesian Joint Endpoint Confidence")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_coefficients(coeff: pd.DataFrame, out_path: Path) -> None:
    body = coeff[coeff["kind"] == "beta"].copy()
    if body.empty:
        return
    body = body.sort_values("mean").reset_index(drop=True)
    y = np.arange(len(body))
    left = body["mean"].to_numpy(dtype=float) - body["q025"].to_numpy(dtype=float)
    right = body["q975"].to_numpy(dtype=float) - body["mean"].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
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
    ax.set_title("Multilevel Adaptive Bayesian Covariate Effects")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = _parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    base = _build_base_rows(args)
    X, coef_names, design_stats = _build_design_matrix(base)
    family_idx, family_labels = _build_family_index(base)

    hits = base["observed_hits"].to_numpy(dtype=float)
    n_selected = base["n_selected"].to_numpy(dtype=float)

    fit = _fit_multilevel_model(
        hits=hits,
        n_selected=n_selected,
        X=X,
        family_idx=family_idx,
        family_labels=family_labels,
        args=args,
    )

    eta_draws = np.asarray(fit["eta"], dtype=float)
    beta_draws = np.asarray(fit["beta"], dtype=float)
    family_draws = np.asarray(fit["family_effects"], dtype=float)
    sigma_draws = np.asarray(fit["sigma"], dtype=float)
    tau_draws = np.asarray(fit["tau"], dtype=float)

    diag = pd.DataFrame([{**fit["diagnostics"], **design_stats, "n_cohorts": int(len(base))}])
    diag.to_csv(args.output_dir / "multilevel_adaptive_mcmc_diagnostics.csv", index=False)

    coeff = pd.concat(
        [
            _posterior_summary(beta_draws, coef_names, kind="beta"),
            _posterior_summary(family_draws, [f"family_effect::{x}" for x in family_labels], kind="family_effect"),
            _posterior_summary(
                np.column_stack([sigma_draws, tau_draws]),
                ["cohort_random_effect_sigma", "family_random_effect_tau"],
                kind="scale",
            ),
        ],
        ignore_index=True,
    )
    coeff.to_csv(args.output_dir / "multilevel_adaptive_posterior_coefficients.csv", index=False)

    fam = coeff[coeff["kind"] == "family_effect"].copy().reset_index(drop=True)
    fam.to_csv(args.output_dir / "multilevel_adaptive_family_effects.csv", index=False)

    theta = np.clip(expit(eta_draws), _EPS, 1.0 - _EPS)
    rng = np.random.default_rng(int(args.seed) + 12000)
    rows: List[Dict[str, object]] = []
    for i, row in enumerate(base.itertuples(index=False)):
        n_sel = int(row.n_selected)
        pred_hits = rng.binomial(n=n_sel, p=theta[:, i])

        min_hits = int(row.min_hits_for_enrichment)
        hit_grid = np.arange(n_sel + 1)
        pvals = hypergeom.sf(hit_grid - 1, M=int(row.n_universe), n=int(row.n_reference_edges_in_candidate_pool), N=n_sel)
        pred_pvals = pvals[pred_hits]

        prob_threshold = float(np.mean(theta[:, i] >= float(row.endpoint_hit_rate_threshold)))
        prob_enrich = float(np.mean(pred_hits >= min_hits))
        prob_joint = float(np.mean((pred_hits >= min_hits) & (pred_pvals <= float(args.endpoint_max_p))))

        q025, q50, q975 = np.quantile(theta[:, i], [0.025, 0.5, 0.975])
        rr = float(row.random_hit_rate)
        if rr > 0:
            eq025, eq50, eq975 = float(q025 / rr), float(q50 / rr), float(q975 / rr)
        else:
            eq025, eq50, eq975 = float("nan"), float("nan"), float("nan")

        tier = _tier_from_joint_prob(
            support_eligible=bool(row.observed_support_eligible),
            joint_prob=prob_joint,
            high_conf=float(args.high_confidence),
            low_conf=float(args.low_confidence),
        )

        rows.append(
            {
                **row._asdict(),
                "posterior_prob_enrichment_threshold": prob_threshold,
                "predictive_prob_enrichment_pass": prob_enrich,
                "predictive_prob_joint_pass": prob_joint,
                "posterior_hit_rate_q025": float(q025),
                "posterior_hit_rate_q50": float(q50),
                "posterior_hit_rate_q975": float(q975),
                "posterior_enrichment_q025": eq025,
                "posterior_enrichment_q50": eq50,
                "posterior_enrichment_q975": eq975,
                "multilevel_adaptive_joint_tier": tier,
                "assay_family_index": int(family_idx[i]),
                **{f"x_{coef_names[j]}": float(X[i, j]) for j in range(len(coef_names))},
            }
        )

    out = pd.DataFrame(rows).sort_values("cohort").reset_index(drop=True)
    out.to_csv(args.output_dir / "multilevel_adaptive_endpoint_rows.csv", index=False)

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
    support_summary.to_csv(args.output_dir / "multilevel_adaptive_support_summary.csv", index=False)

    tier_summary = (
        out.groupby("multilevel_adaptive_joint_tier", as_index=False)
        .agg(
            n_cohorts=("cohort", "count"),
            mean_support_edges=("observed_support_edges", "mean"),
            observed_pass_rate=("observed_endpoint_pass", "mean"),
            mean_predictive_prob_joint_pass=("predictive_prob_joint_pass", "mean"),
        )
        .sort_values("multilevel_adaptive_joint_tier")
        .reset_index(drop=True)
    )
    tier_summary.to_csv(args.output_dir / "multilevel_adaptive_tier_summary.csv", index=False)

    overall = pd.DataFrame(
        [
            {
                "n_cohorts": int(len(out)),
                "n_observed_pass": int(out["observed_endpoint_pass"].sum()),
                "observed_pass_rate": float(out["observed_endpoint_pass"].mean()),
                "n_powered_confirmed_pass": int((out["multilevel_adaptive_joint_tier"] == "POWERED_MULTILEVEL_JOINT_CONFIRMED_PASS").sum()),
                "n_powered_confirmed_fail": int((out["multilevel_adaptive_joint_tier"] == "POWERED_MULTILEVEL_JOINT_CONFIRMED_FAIL").sum()),
                "n_underpowered_positive_signal": int((out["multilevel_adaptive_joint_tier"] == "UNDERPOWERED_MULTILEVEL_JOINT_POSITIVE_SIGNAL").sum()),
                "n_underpowered_negative_signal": int((out["multilevel_adaptive_joint_tier"] == "UNDERPOWERED_MULTILEVEL_JOINT_NEGATIVE_SIGNAL").sum()),
            }
        ]
    )
    overall.to_csv(args.output_dir / "multilevel_adaptive_overall_summary.csv", index=False)

    ppc = _compute_ppc(eta_draws=eta_draws, base=base, seed=int(args.seed) + 17000)
    ppc.to_csv(args.output_dir / "multilevel_adaptive_ppc_summary.csv", index=False)

    if args.adaptive_rows.exists():
        phase26 = pd.read_csv(args.adaptive_rows, low_memory=False)
        cmp26 = out.merge(
            phase26[["cohort", "predictive_prob_joint_pass", "adaptive_bayes_joint_tier"]].rename(
                columns={"predictive_prob_joint_pass": "phase26_predictive_prob_joint_pass"}
            ),
            on="cohort",
            how="left",
        )
        cmp26["delta_multilevel_minus_phase26_joint_prob"] = (
            cmp26["predictive_prob_joint_pass"] - cmp26["phase26_predictive_prob_joint_pass"]
        )
        cmp26.to_csv(args.output_dir / "multilevel_adaptive_vs_phase26_comparison.csv", index=False)

    if args.full_bayes_rows.exists():
        full = pd.read_csv(args.full_bayes_rows, low_memory=False)
        cmp_full = out.merge(
            full[["cohort", "predictive_prob_joint_pass", "full_bayes_joint_tier"]].rename(
                columns={"predictive_prob_joint_pass": "full_bayes_predictive_prob_joint_pass"}
            ),
            on="cohort",
            how="left",
        )
        cmp_full["delta_multilevel_minus_full_bayes_joint_prob"] = (
            cmp_full["predictive_prob_joint_pass"] - cmp_full["full_bayes_predictive_prob_joint_pass"]
        )
        cmp_full.to_csv(args.output_dir / "multilevel_adaptive_vs_full_bayes_comparison.csv", index=False)

    _plot_support_vs_joint(
        out=out,
        endpoint_min_support=float(args.endpoint_min_support),
        high_conf=float(args.high_confidence),
        low_conf=float(args.low_confidence),
        out_path=args.output_dir / "multilevel_adaptive_support_vs_joint_probability.png",
    )
    _plot_coefficients(coeff=coeff, out_path=args.output_dir / "multilevel_adaptive_coefficient_forest.png")

    print("Multilevel adaptive Bayesian endpoint tiering complete.")
    print(overall.to_string(index=False))


if __name__ == "__main__":
    main()
