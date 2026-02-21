#!/usr/bin/env python3
"""Fit a hierarchical-Bayesian family-aware latent-truth model.

Model structure:
- Edge latent state: T (true), N0 (clean negative), N1 (bursty negative)
- Source emissions:
  alpha_j = P(y=1|T), beta0_j = P(y=1|N0), beta1_j = P(y=1|N1)
- Source parameters are regularized via hierarchical priors:
  source -> family mean -> global mean

This provides a more principled alternative to heuristic family shrinkage.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

from ltm_utils import parse_kind_list


_EPS = 1e-9


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--matrix",
        type=Path,
        default=Path("subproject_24_multireference_latent_truth_model/implementation/outputs/evidence_matrix_wide.csv"),
    )
    parser.add_argument(
        "--source-meta",
        type=Path,
        default=Path("subproject_24_multireference_latent_truth_model/implementation/outputs/source_metadata.csv"),
    )
    parser.add_argument("--include-kinds", type=str, default="reference,mechanistic")
    parser.add_argument("--exclude-sources", type=str, default="")
    parser.add_argument(
        "--family-map",
        type=Path,
        default=Path(""),
        help="Optional CSV with columns source_id,family.",
    )
    parser.add_argument("--max-edges", type=int, default=0)
    parser.add_argument(
        "--edge-subsample-seed",
        type=int,
        default=-1,
        help="If >=0 and --max-edges>0, sample edges with this random seed instead of using head().",
    )
    parser.add_argument("--max-iter", type=int, default=250)
    parser.add_argument("--tol", type=float, default=1e-6)
    parser.add_argument(
        "--source-strength-alpha",
        type=float,
        default=12.0,
        help="Prior concentration from family mean to source alpha.",
    )
    parser.add_argument(
        "--source-strength-beta",
        type=float,
        default=24.0,
        help="Prior concentration from family mean to source beta0/beta1.",
    )
    parser.add_argument(
        "--family-strength",
        type=float,
        default=4.0,
        help="Prior concentration from global mean to family mean.",
    )
    parser.add_argument(
        "--global-prior-strength",
        type=float,
        default=2.0,
        help="Weak anchoring concentration for global means toward empirical base.",
    )
    parser.add_argument(
        "--min-youden-clean",
        type=float,
        default=0.01,
        help="Minimum clean Youden margin alpha-beta0.",
    )
    parser.add_argument(
        "--min-burst-gap",
        type=float,
        default=1e-4,
        help="Minimum burst FP gap beta1-beta0.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("subproject_24_multireference_latent_truth_model/implementation/outputs"),
    )
    return parser.parse_args()


def _select_annotators(
    source_meta: pd.DataFrame,
    include_kinds: Sequence[str],
    exclude_sources: Sequence[str],
) -> List[str]:
    sel = source_meta[source_meta["kind"].isin(include_kinds)].copy()
    if exclude_sources:
        sel = sel[~sel["source_id"].isin(exclude_sources)]
    sel = sel.sort_values("source_id").reset_index(drop=True)
    return sel["source_id"].tolist()


def _default_family(source_id: str, kind: str) -> str:
    if source_id.startswith("dorothea_"):
        return "curated_dorothea"
    if source_id == "trrust_human":
        return "curated_trrust"
    if source_id in {"beeline_gsd", "hpn_dream"}:
        return "small_gold_standard"
    if source_id == "timeseries_proxy":
        return "timeseries_proxy"
    if source_id == "dixit_perturb_top500":
        return "perturbation_proxy"
    if source_id == "invariant_top_edges":
        return "mechanistic_invariant"
    return f"{kind}_other"


def _load_family_map(path: Path) -> Dict[str, str]:
    if not path or str(path) == "" or not path.exists() or path.is_dir():
        return {}
    df = pd.read_csv(path)
    need = {"source_id", "family"}
    if not need.issubset(df.columns):
        raise ValueError(f"--family-map must contain columns {sorted(need)}")
    out = (
        df[["source_id", "family"]]
        .dropna()
        .astype(str)
        .drop_duplicates(subset=["source_id"], keep="last")
    )
    return dict(zip(out["source_id"], out["family"]))


def _family_assignments(
    source_meta: pd.DataFrame,
    annotators: Sequence[str],
    family_map: Dict[str, str],
) -> List[str]:
    sub = source_meta[source_meta["source_id"].isin(annotators)][["source_id", "kind"]].copy()
    kind_by_source = dict(zip(sub["source_id"], sub["kind"]))
    families = []
    for source_id in annotators:
        families.append(family_map.get(source_id, _default_family(source_id=source_id, kind=str(kind_by_source[source_id]))))
    return families


def _beta_posterior_mean(
    pos: float,
    total: float,
    prior_mean: float,
    prior_strength: float,
) -> float:
    m = float(np.clip(prior_mean, 0.001, 0.999))
    s = max(float(prior_strength), 0.0)
    a = m * s + 1.0
    b = (1.0 - m) * s + 1.0
    return float(np.clip((pos + a) / (total + a + b + _EPS), 0.001, 0.999))


def _weighted_family_mean(
    values: np.ndarray,
    weights: np.ndarray,
    family_ids: np.ndarray,
    global_mean: float,
    family_strength: float,
) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for fam in sorted(np.unique(family_ids)):
        idx = family_ids == fam
        wf = float(np.sum(weights[idx]))
        if wf <= 0.0:
            fam_mean = global_mean
        else:
            fam_mean = float(np.sum(weights[idx] * values[idx]) / wf)
        out[str(fam)] = float(
            (wf * fam_mean + family_strength * global_mean) / (wf + family_strength + _EPS)
        )
    return out


def _update_global_mean(
    family_means: Dict[str, float],
    family_weight: Dict[str, float],
    base_mean: float,
    global_prior_strength: float,
) -> float:
    num = 0.0
    den = 0.0
    for fam, mu in family_means.items():
        w = max(float(family_weight.get(fam, 0.0)), 0.0)
        num += w * mu
        den += w
    num += global_prior_strength * base_mean
    den += global_prior_strength
    return float(np.clip(num / (den + _EPS), 0.001, 0.999))


def _fit_hierarchical_family_em(
    y: np.ndarray,
    family_ids: Sequence[str],
    max_iter: int,
    tol: float,
    source_strength_alpha: float,
    source_strength_beta: float,
    family_strength: float,
    global_prior_strength: float,
    min_youden_clean: float,
    min_burst_gap: float,
):
    obs = ~np.isnan(y)
    y0 = np.nan_to_num(y, nan=0.0)
    n_edges, n_ann = y.shape
    fam_arr = np.asarray(family_ids, dtype=object)

    edge_pos_rate = np.where(obs.any(axis=1), np.nanmean(y, axis=1), np.nan)
    global_pos = np.nanmean(edge_pos_rate)
    if np.isnan(global_pos):
        global_pos = 0.1

    gamma_t = np.where(np.isnan(edge_pos_rate), global_pos, edge_pos_rate)
    gamma_t = np.clip(gamma_t, 0.01, 0.95)
    neg_mass = np.clip(1.0 - gamma_t, 0.05, 0.99)
    gamma_n1 = 0.10 * neg_mass
    gamma_n0 = neg_mass - gamma_n1

    pi = float(np.clip(np.mean(gamma_t), 0.01, 0.99))
    rho = 0.10

    alpha = np.full(n_ann, 0.80, dtype=float)
    beta0 = np.full(n_ann, 0.08, dtype=float)
    beta1 = np.full(n_ann, 0.25, dtype=float)

    base_global_alpha = float(np.mean(alpha))
    base_global_beta0 = float(np.mean(beta0))
    base_global_beta1 = float(np.mean(beta1))

    g_alpha = base_global_alpha
    g_beta0 = base_global_beta0
    g_beta1 = base_global_beta1

    trace = []

    for it in range(1, max_iter + 1):
        log_t = np.full(n_edges, np.log(pi + _EPS), dtype=float)
        log_n0 = np.full(n_edges, np.log(1.0 - pi + _EPS) + np.log(1.0 - rho + _EPS), dtype=float)
        log_n1 = np.full(n_edges, np.log(1.0 - pi + _EPS) + np.log(rho + _EPS), dtype=float)

        for j in range(n_ann):
            mask = obs[:, j]
            if not np.any(mask):
                continue
            yj = y0[:, j]
            log_t[mask] += yj[mask] * np.log(alpha[j] + _EPS) + (1.0 - yj[mask]) * np.log(1.0 - alpha[j] + _EPS)
            log_n0[mask] += yj[mask] * np.log(beta0[j] + _EPS) + (1.0 - yj[mask]) * np.log(1.0 - beta0[j] + _EPS)
            log_n1[mask] += yj[mask] * np.log(beta1[j] + _EPS) + (1.0 - yj[mask]) * np.log(1.0 - beta1[j] + _EPS)

        max_log = np.maximum(np.maximum(log_t, log_n0), log_n1)
        denom = np.exp(log_t - max_log) + np.exp(log_n0 - max_log) + np.exp(log_n1 - max_log)
        log_denom = max_log + np.log(denom)

        gamma_t = np.exp(log_t - log_denom)
        gamma_n0 = np.exp(log_n0 - log_denom)
        gamma_n1 = np.exp(log_n1 - log_denom)
        gamma_t = np.clip(gamma_t, _EPS, 1.0 - _EPS)
        gamma_n0 = np.clip(gamma_n0, _EPS, 1.0 - _EPS)
        gamma_n1 = np.clip(gamma_n1, _EPS, 1.0 - _EPS)
        norm = gamma_t + gamma_n0 + gamma_n1
        gamma_t /= norm
        gamma_n0 /= norm
        gamma_n1 /= norm

        pi = float(np.clip(np.mean(gamma_t), 0.001, 0.999))
        neg_total = float(np.sum(gamma_n0 + gamma_n1))
        rho = float(np.clip((np.sum(gamma_n1) + 1.0) / (max(neg_total, _EPS) + 2.0), 0.001, 0.999))

        t_pos = np.zeros(n_ann, dtype=float)
        t_tot = np.zeros(n_ann, dtype=float)
        n0_pos = np.zeros(n_ann, dtype=float)
        n0_tot = np.zeros(n_ann, dtype=float)
        n1_pos = np.zeros(n_ann, dtype=float)
        n1_tot = np.zeros(n_ann, dtype=float)

        alpha_mle = np.zeros(n_ann, dtype=float)
        beta0_mle = np.zeros(n_ann, dtype=float)
        beta1_mle = np.zeros(n_ann, dtype=float)

        for j in range(n_ann):
            mask = obs[:, j]
            if not np.any(mask):
                alpha_mle[j] = 0.5
                beta0_mle[j] = 0.5
                beta1_mle[j] = 0.5
                continue

            yj = y0[mask, j]
            gt = gamma_t[mask]
            gn0 = gamma_n0[mask]
            gn1 = gamma_n1[mask]

            t_tot[j] = float(np.sum(gt))
            n0_tot[j] = float(np.sum(gn0))
            n1_tot[j] = float(np.sum(gn1))
            t_pos[j] = float(np.sum(gt * yj))
            n0_pos[j] = float(np.sum(gn0 * yj))
            n1_pos[j] = float(np.sum(gn1 * yj))

            alpha_mle[j] = float(np.clip((t_pos[j] + 1.0) / (t_tot[j] + 2.0), 0.001, 0.999))
            beta0_mle[j] = float(np.clip((n0_pos[j] + 1.0) / (n0_tot[j] + 2.0), 0.001, 0.999))
            beta1_mle[j] = float(np.clip((n1_pos[j] + 1.0) / (n1_tot[j] + 2.0), 0.001, 0.999))

        w_alpha = np.log1p(t_tot)
        w_beta0 = np.log1p(n0_tot)
        w_beta1 = np.log1p(n1_tot)

        fam_alpha = _weighted_family_mean(
            values=alpha_mle,
            weights=w_alpha,
            family_ids=fam_arr,
            global_mean=g_alpha,
            family_strength=family_strength,
        )
        fam_beta0 = _weighted_family_mean(
            values=beta0_mle,
            weights=w_beta0,
            family_ids=fam_arr,
            global_mean=g_beta0,
            family_strength=family_strength,
        )
        fam_beta1 = _weighted_family_mean(
            values=beta1_mle,
            weights=w_beta1,
            family_ids=fam_arr,
            global_mean=g_beta1,
            family_strength=family_strength,
        )

        fam_weight_alpha = {str(f): float(np.sum(w_alpha[fam_arr == f])) for f in np.unique(fam_arr)}
        fam_weight_beta0 = {str(f): float(np.sum(w_beta0[fam_arr == f])) for f in np.unique(fam_arr)}
        fam_weight_beta1 = {str(f): float(np.sum(w_beta1[fam_arr == f])) for f in np.unique(fam_arr)}

        g_alpha = _update_global_mean(fam_alpha, fam_weight_alpha, base_global_alpha, global_prior_strength)
        g_beta0 = _update_global_mean(fam_beta0, fam_weight_beta0, base_global_beta0, global_prior_strength)
        g_beta1 = _update_global_mean(fam_beta1, fam_weight_beta1, base_global_beta1, global_prior_strength)

        for j in range(n_ann):
            fam = str(fam_arr[j])
            alpha[j] = _beta_posterior_mean(
                pos=t_pos[j],
                total=t_tot[j],
                prior_mean=fam_alpha[fam],
                prior_strength=source_strength_alpha,
            )
            beta0[j] = _beta_posterior_mean(
                pos=n0_pos[j],
                total=n0_tot[j],
                prior_mean=fam_beta0[fam],
                prior_strength=source_strength_beta,
            )
            beta1[j] = _beta_posterior_mean(
                pos=n1_pos[j],
                total=n1_tot[j],
                prior_mean=fam_beta1[fam],
                prior_strength=source_strength_beta,
            )

            if beta1[j] < beta0[j] + min_burst_gap:
                beta1[j] = min(0.999, beta0[j] + min_burst_gap)
            if alpha[j] < beta0[j] + min_youden_clean:
                alpha[j] = min(0.999, beta0[j] + min_youden_clean)

        ll = float(np.sum(log_denom))
        trace.append(
            {
                "iter": it,
                "log_likelihood": ll,
                "pi": pi,
                "rho": rho,
                "global_alpha": g_alpha,
                "global_beta0": g_beta0,
                "global_beta1": g_beta1,
            }
        )

        if it > 1 and abs(trace[-1]["log_likelihood"] - trace[-2]["log_likelihood"]) < tol:
            break

    return gamma_t, gamma_n0, gamma_n1, alpha, beta0, beta1, pi, rho, g_alpha, g_beta0, g_beta1, pd.DataFrame(trace)


def main() -> None:
    args = _parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    matrix = pd.read_csv(args.matrix)
    source_meta = pd.read_csv(args.source_meta)

    include_kinds = parse_kind_list(args.include_kinds)
    exclude_sources = parse_kind_list(args.exclude_sources)
    annotators = _select_annotators(source_meta, include_kinds, exclude_sources)
    if not annotators:
        raise ValueError("No annotators selected. Check --include-kinds/--exclude-sources.")

    family_map = _load_family_map(args.family_map)
    families = _family_assignments(source_meta=source_meta, annotators=annotators, family_map=family_map)

    mat = matrix[["source", "target"] + annotators].copy()
    if args.max_edges > 0:
        if args.edge_subsample_seed >= 0:
            take_n = min(args.max_edges, len(mat))
            mat = mat.sample(n=take_n, random_state=int(args.edge_subsample_seed)).reset_index(drop=True)
        else:
            mat = mat.head(args.max_edges).copy()

    y = mat[annotators].to_numpy(dtype=float)
    (
        gamma_t,
        gamma_n0,
        gamma_n1,
        alpha,
        beta0,
        beta1,
        pi,
        rho,
        g_alpha,
        g_beta0,
        g_beta1,
        trace,
    ) = _fit_hierarchical_family_em(
        y=y,
        family_ids=families,
        max_iter=args.max_iter,
        tol=args.tol,
        source_strength_alpha=args.source_strength_alpha,
        source_strength_beta=args.source_strength_beta,
        family_strength=args.family_strength,
        global_prior_strength=args.global_prior_strength,
        min_youden_clean=args.min_youden_clean,
        min_burst_gap=args.min_burst_gap,
    )

    posterior = mat[["source", "target"]].copy()
    posterior["posterior"] = gamma_t
    posterior["posterior_neg_clean"] = gamma_n0
    posterior["posterior_neg_burst"] = gamma_n1
    neg_mass = np.clip(gamma_n0 + gamma_n1, _EPS, 1.0)
    posterior["posterior_burst_given_negative"] = gamma_n1 / neg_mass

    obs = ~np.isnan(y)
    y0 = np.nan_to_num(y, nan=0.0)
    posterior["n_observed_sources"] = obs.sum(axis=1)
    posterior["n_positive_sources"] = (y0 * obs).sum(axis=1)

    rel = pd.DataFrame(
        {
            "source_id": annotators,
            "family": families,
            "alpha_sensitivity": alpha,
            "beta_false_positive_clean": beta0,
            "beta_false_positive_burst": beta1,
            "burst_fp_delta": beta1 - beta0,
            "youden_clean": alpha - beta0,
            "observed_edges": obs.sum(axis=0),
        }
    )
    rel = rel.merge(source_meta[["source_id", "kind"]], on="source_id", how="left")
    rel = rel.sort_values(["family", "youden_clean"], ascending=[True, False]).reset_index(drop=True)

    family_means = (
        rel.groupby("family", as_index=False)
        .agg(
            n_sources=("source_id", "count"),
            mean_alpha=("alpha_sensitivity", "mean"),
            mean_beta0=("beta_false_positive_clean", "mean"),
            mean_beta1=("beta_false_positive_burst", "mean"),
            mean_youden=("youden_clean", "mean"),
        )
        .sort_values("family")
        .reset_index(drop=True)
    )

    fit_summary = pd.DataFrame(
        [
            {
                "n_edges": len(mat),
                "n_annotators": len(annotators),
                "n_families": len(set(families)),
                "pi_prevalence": pi,
                "rho_burst": rho,
                "global_alpha": g_alpha,
                "global_beta0": g_beta0,
                "global_beta1": g_beta1,
                "iterations": int(trace["iter"].max()),
                "final_log_likelihood": float(trace["log_likelihood"].iloc[-1]),
                "source_strength_alpha": float(args.source_strength_alpha),
                "source_strength_beta": float(args.source_strength_beta),
                "family_strength": float(args.family_strength),
                "global_prior_strength": float(args.global_prior_strength),
                "min_youden_clean": float(args.min_youden_clean),
                "min_burst_gap": float(args.min_burst_gap),
                "edge_subsample_seed": int(args.edge_subsample_seed),
                "include_kinds": ",".join(include_kinds),
                "exclude_sources": ",".join(exclude_sources),
            }
        ]
    )

    posterior.to_csv(args.output_dir / "ltm_family_hb_edge_posteriors.csv", index=False)
    rel.to_csv(args.output_dir / "ltm_family_hb_source_reliability.csv", index=False)
    family_means.to_csv(args.output_dir / "ltm_family_hb_family_means.csv", index=False)
    trace.to_csv(args.output_dir / "ltm_family_hb_fit_trace.csv", index=False)
    fit_summary.to_csv(args.output_dir / "ltm_family_hb_fit_summary.csv", index=False)

    print("Hierarchical family-Bayes latent-truth fit complete:")
    print(f"- edges: {len(mat)}")
    print(f"- annotators: {len(annotators)}")
    print(f"- families: {len(set(families))}")
    print(f"- prevalence pi: {pi:.6f}")
    print(f"- burst rho: {rho:.6f}")
    print(f"- iterations: {int(trace['iter'].max())}")
    print(f"- outputs: {args.output_dir}")


if __name__ == "__main__":
    main()
