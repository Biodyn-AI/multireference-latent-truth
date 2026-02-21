#!/usr/bin/env python3
"""Fit a family-regularized dependence-aware latent-truth model.

This extends the one-burst dependence model with source-family shrinkage:
- each source reliability parameter is shrunk toward a family target,
- each family target is shrunk toward a global target.

The goal is to reduce source-family overdominance while preserving the
same output schema used by downstream ranking and ablation scripts.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Sequence

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
    parser.add_argument("--max-iter", type=int, default=250)
    parser.add_argument("--tol", type=float, default=1e-6)
    parser.add_argument(
        "--family-strength-alpha",
        type=float,
        default=4.0,
        help="Shrinkage weight toward family target for alpha (log-count scale).",
    )
    parser.add_argument(
        "--family-strength-beta",
        type=float,
        default=12.0,
        help="Shrinkage weight toward family target for beta0/beta1 (log-count scale).",
    )
    parser.add_argument(
        "--global-strength",
        type=float,
        default=2.0,
        help="Family-target shrinkage toward global target (member-count scale).",
    )
    parser.add_argument(
        "--min-youden-clean",
        type=float,
        default=1e-4,
        help="Minimum enforced alpha-beta0 margin to avoid inverse-clean discrimination.",
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
        if source_id in family_map:
            families.append(family_map[source_id])
        else:
            families.append(_default_family(source_id=source_id, kind=str(kind_by_source[source_id])))
    return families


def _family_targets(
    values: np.ndarray,
    family_ids: np.ndarray,
    global_strength: float,
) -> Dict[str, float]:
    global_mean = float(np.mean(values))
    out: Dict[str, float] = {}
    for fam in sorted(np.unique(family_ids)):
        idx = family_ids == fam
        fam_mean = float(np.mean(values[idx]))
        n_members = float(np.sum(idx))
        tgt = (n_members * fam_mean + global_strength * global_mean) / (n_members + global_strength)
        out[str(fam)] = float(tgt)
    return out


def _fit_family_regularized_em(
    y: np.ndarray,
    family_ids: Sequence[str],
    max_iter: int,
    tol: float,
    family_strength_alpha: float,
    family_strength_beta: float,
    global_strength: float,
    min_youden_clean: float,
):
    obs = ~np.isnan(y)
    y0 = np.nan_to_num(y, nan=0.0)

    n_edges, n_ann = y.shape
    family_ids_arr = np.asarray(family_ids, dtype=object)

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

        alpha_mle = np.zeros(n_ann, dtype=float)
        beta0_mle = np.zeros(n_ann, dtype=float)
        beta1_mle = np.zeros(n_ann, dtype=float)

        eff_alpha = np.zeros(n_ann, dtype=float)
        eff_beta0 = np.zeros(n_ann, dtype=float)
        eff_beta1 = np.zeros(n_ann, dtype=float)

        for j in range(n_ann):
            mask = obs[:, j]
            if not np.any(mask):
                alpha_mle[j] = 0.5
                beta0_mle[j] = 0.5
                beta1_mle[j] = 0.5
                eff_alpha[j] = 0.0
                eff_beta0[j] = 0.0
                eff_beta1[j] = 0.0
                continue

            yj = y0[mask, j]
            gt = gamma_t[mask]
            gn0 = gamma_n0[mask]
            gn1 = gamma_n1[mask]

            s_gt = float(np.sum(gt))
            s_gn0 = float(np.sum(gn0))
            s_gn1 = float(np.sum(gn1))

            alpha_mle[j] = float(np.clip((np.sum(gt * yj) + 1.0) / (s_gt + 2.0), 0.001, 0.999))
            beta0_mle[j] = float(np.clip((np.sum(gn0 * yj) + 1.0) / (s_gn0 + 2.0), 0.001, 0.999))
            beta1_mle[j] = float(np.clip((np.sum(gn1 * yj) + 1.0) / (s_gn1 + 2.0), 0.001, 0.999))

            eff_alpha[j] = s_gt + 2.0
            eff_beta0[j] = s_gn0 + 2.0
            eff_beta1[j] = s_gn1 + 2.0

        alpha_tgt = _family_targets(alpha_mle, family_ids_arr, global_strength=global_strength)
        beta0_tgt = _family_targets(beta0_mle, family_ids_arr, global_strength=global_strength)
        beta1_tgt = _family_targets(beta1_mle, family_ids_arr, global_strength=global_strength)

        for j in range(n_ann):
            fam = str(family_ids_arr[j])

            wa = np.log1p(eff_alpha[j])
            wb0 = np.log1p(eff_beta0[j])
            wb1 = np.log1p(eff_beta1[j])

            alpha[j] = np.clip(
                (wa * alpha_mle[j] + family_strength_alpha * alpha_tgt[fam]) / (wa + family_strength_alpha + _EPS),
                0.001,
                0.999,
            )
            beta0[j] = np.clip(
                (wb0 * beta0_mle[j] + family_strength_beta * beta0_tgt[fam]) / (wb0 + family_strength_beta + _EPS),
                0.001,
                0.999,
            )
            beta1[j] = np.clip(
                (wb1 * beta1_mle[j] + family_strength_beta * beta1_tgt[fam]) / (wb1 + family_strength_beta + _EPS),
                0.001,
                0.999,
            )

            if beta1[j] < beta0[j]:
                beta1[j] = min(0.999, beta0[j] + 1e-4)
            if alpha[j] < beta0[j] + min_youden_clean:
                alpha[j] = min(0.999, beta0[j] + min_youden_clean)

        ll = float(np.sum(log_denom))
        trace.append(
            {
                "iter": it,
                "log_likelihood": ll,
                "pi": pi,
                "rho": rho,
                "mean_alpha": float(np.mean(alpha)),
                "mean_beta0": float(np.mean(beta0)),
                "mean_beta1": float(np.mean(beta1)),
            }
        )

        if it > 1 and abs(trace[-1]["log_likelihood"] - trace[-2]["log_likelihood"]) < tol:
            break

    return gamma_t, gamma_n0, gamma_n1, alpha, beta0, beta1, pi, rho, pd.DataFrame(trace)


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
        mat = mat.head(args.max_edges).copy()

    y = mat[annotators].to_numpy(dtype=float)
    gamma_t, gamma_n0, gamma_n1, alpha, beta0, beta1, pi, rho, trace = _fit_family_regularized_em(
        y=y,
        family_ids=families,
        max_iter=args.max_iter,
        tol=args.tol,
        family_strength_alpha=args.family_strength_alpha,
        family_strength_beta=args.family_strength_beta,
        global_strength=args.global_strength,
        min_youden_clean=args.min_youden_clean,
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

    reliability = pd.DataFrame(
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
    reliability = reliability.merge(source_meta[["source_id", "kind"]], on="source_id", how="left")
    reliability = reliability.sort_values(["family", "youden_clean"], ascending=[True, False]).reset_index(drop=True)

    fit_summary = pd.DataFrame(
        [
            {
                "n_edges": len(mat),
                "n_annotators": len(annotators),
                "n_families": len(set(families)),
                "pi_prevalence": pi,
                "rho_burst": rho,
                "iterations": int(trace["iter"].max()),
                "final_log_likelihood": float(trace["log_likelihood"].iloc[-1]),
                "family_strength_alpha": float(args.family_strength_alpha),
                "family_strength_beta": float(args.family_strength_beta),
                "global_strength": float(args.global_strength),
                "min_youden_clean": float(args.min_youden_clean),
                "include_kinds": ",".join(include_kinds),
                "exclude_sources": ",".join(exclude_sources),
            }
        ]
    )

    posterior.to_csv(args.output_dir / "ltm_family_edge_posteriors.csv", index=False)
    reliability.to_csv(args.output_dir / "ltm_family_source_reliability.csv", index=False)
    trace.to_csv(args.output_dir / "ltm_family_fit_trace.csv", index=False)
    fit_summary.to_csv(args.output_dir / "ltm_family_fit_summary.csv", index=False)

    print("Family-regularized latent-truth fit complete:")
    print(f"- edges: {len(mat)}")
    print(f"- annotators: {len(annotators)}")
    print(f"- families: {len(set(families))}")
    print(f"- prevalence pi: {pi:.6f}")
    print(f"- burst rho: {rho:.6f}")
    print(f"- iterations: {int(trace['iter'].max())}")
    print(f"- outputs: {args.output_dir}")


if __name__ == "__main__":
    main()
