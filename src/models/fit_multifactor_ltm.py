#!/usr/bin/env python3
"""Fit a multi-factor dependence-aware latent-truth model.

States per edge:
- T: true edge
- N0: clean negative
- N1: moderate-burst negative
- N2: high-burst negative

This extends the one-factor dependence model by allowing two distinct
false-positive burst regimes, reducing pressure to explain all source
co-activation with a single negative burst component.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

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
    parser.add_argument("--max-edges", type=int, default=0)
    parser.add_argument("--max-iter", type=int, default=250)
    parser.add_argument("--tol", type=float, default=1e-6)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("subproject_24_multireference_latent_truth_model/implementation/outputs"),
    )
    return parser.parse_args()


def _select_annotators(source_meta: pd.DataFrame, include_kinds: List[str], exclude_sources: List[str]) -> List[str]:
    sel = source_meta[source_meta["kind"].isin(include_kinds)].copy()
    if exclude_sources:
        sel = sel[~sel["source_id"].isin(exclude_sources)]
    sel = sel.sort_values("source_id").reset_index(drop=True)
    return sel["source_id"].tolist()


def _fit_multifactor_em(y: np.ndarray, max_iter: int, tol: float):
    obs = ~np.isnan(y)
    y0 = np.nan_to_num(y, nan=0.0)

    n_edges, n_ann = y.shape

    edge_pos_rate = np.where(obs.any(axis=1), np.nanmean(y, axis=1), np.nan)
    global_pos = np.nanmean(edge_pos_rate)
    if np.isnan(global_pos):
        global_pos = 0.1

    gamma_t = np.where(np.isnan(edge_pos_rate), global_pos, edge_pos_rate)
    gamma_t = np.clip(gamma_t, 0.01, 0.95)

    neg = np.clip(1.0 - gamma_t, 0.05, 0.99)
    gamma_n1 = 0.07 * neg
    gamma_n2 = 0.03 * neg
    gamma_n0 = neg - gamma_n1 - gamma_n2

    pi = float(np.clip(np.mean(gamma_t), 0.01, 0.99))
    rho1 = 0.07
    rho2 = 0.03

    alpha = np.full(n_ann, 0.80, dtype=float)
    beta0 = np.full(n_ann, 0.08, dtype=float)
    beta1 = np.full(n_ann, 0.20, dtype=float)
    beta2 = np.full(n_ann, 0.35, dtype=float)

    trace = []

    for it in range(1, max_iter + 1):
        log_t = np.full(n_edges, np.log(pi + _EPS), dtype=float)
        log_n0 = np.full(
            n_edges,
            np.log(1.0 - pi + _EPS) + np.log(max(1.0 - rho1 - rho2, _EPS)),
            dtype=float,
        )
        log_n1 = np.full(n_edges, np.log(1.0 - pi + _EPS) + np.log(rho1 + _EPS), dtype=float)
        log_n2 = np.full(n_edges, np.log(1.0 - pi + _EPS) + np.log(rho2 + _EPS), dtype=float)

        for j in range(n_ann):
            mask = obs[:, j]
            if not np.any(mask):
                continue

            yj = y0[:, j]
            log_t[mask] += yj[mask] * np.log(alpha[j] + _EPS) + (1.0 - yj[mask]) * np.log(1.0 - alpha[j] + _EPS)
            log_n0[mask] += yj[mask] * np.log(beta0[j] + _EPS) + (1.0 - yj[mask]) * np.log(1.0 - beta0[j] + _EPS)
            log_n1[mask] += yj[mask] * np.log(beta1[j] + _EPS) + (1.0 - yj[mask]) * np.log(1.0 - beta1[j] + _EPS)
            log_n2[mask] += yj[mask] * np.log(beta2[j] + _EPS) + (1.0 - yj[mask]) * np.log(1.0 - beta2[j] + _EPS)

        max_log = np.maximum(np.maximum(log_t, log_n0), np.maximum(log_n1, log_n2))
        denom = (
            np.exp(log_t - max_log)
            + np.exp(log_n0 - max_log)
            + np.exp(log_n1 - max_log)
            + np.exp(log_n2 - max_log)
        )
        log_denom = max_log + np.log(denom)

        gamma_t = np.exp(log_t - log_denom)
        gamma_n0 = np.exp(log_n0 - log_denom)
        gamma_n1 = np.exp(log_n1 - log_denom)
        gamma_n2 = np.exp(log_n2 - log_denom)

        gamma_t = np.clip(gamma_t, _EPS, 1.0 - _EPS)
        gamma_n0 = np.clip(gamma_n0, _EPS, 1.0 - _EPS)
        gamma_n1 = np.clip(gamma_n1, _EPS, 1.0 - _EPS)
        gamma_n2 = np.clip(gamma_n2, _EPS, 1.0 - _EPS)

        norm = gamma_t + gamma_n0 + gamma_n1 + gamma_n2
        gamma_t /= norm
        gamma_n0 /= norm
        gamma_n1 /= norm
        gamma_n2 /= norm

        pi = float(np.clip(np.mean(gamma_t), 0.001, 0.999))

        neg_total = float(np.sum(gamma_n0 + gamma_n1 + gamma_n2))
        rho1 = float(np.clip(np.sum(gamma_n1) / max(neg_total, _EPS), 0.001, 0.999))
        rho2 = float(np.clip(np.sum(gamma_n2) / max(neg_total, _EPS), 0.001, 0.999))

        # Keep rho1+rho2 below 1 by gentle projection.
        if rho1 + rho2 >= 0.98:
            s = rho1 + rho2
            rho1 = 0.98 * rho1 / s
            rho2 = 0.98 * rho2 / s

        for j in range(n_ann):
            mask = obs[:, j]
            if not np.any(mask):
                alpha[j] = 0.5
                beta0[j] = 0.5
                beta1[j] = 0.5
                beta2[j] = 0.5
                continue

            yj = y0[mask, j]
            gt = gamma_t[mask]
            gn0 = gamma_n0[mask]
            gn1 = gamma_n1[mask]
            gn2 = gamma_n2[mask]

            a = np.clip((np.sum(gt * yj) + 1.0) / (np.sum(gt) + 2.0), 0.001, 0.999)
            b0 = np.clip((np.sum(gn0 * yj) + 1.0) / (np.sum(gn0) + 2.0), 0.001, 0.999)
            b1 = np.clip((np.sum(gn1 * yj) + 1.0) / (np.sum(gn1) + 2.0), 0.001, 0.999)
            b2 = np.clip((np.sum(gn2 * yj) + 1.0) / (np.sum(gn2) + 2.0), 0.001, 0.999)

            # Enforce monotone burst intensity: b0 <= b1 <= b2.
            b_sorted = sorted([b0, b1, b2])
            alpha[j] = a
            beta0[j], beta1[j], beta2[j] = b_sorted

        ll = float(np.sum(log_denom))
        trace.append(
            {
                "iter": it,
                "log_likelihood": ll,
                "pi": pi,
                "rho1": rho1,
                "rho2": rho2,
            }
        )

        if it > 1 and abs(trace[-1]["log_likelihood"] - trace[-2]["log_likelihood"]) < tol:
            break

    return gamma_t, gamma_n0, gamma_n1, gamma_n2, alpha, beta0, beta1, beta2, pi, rho1, rho2, pd.DataFrame(trace)


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

    mat = matrix[["source", "target"] + annotators].copy()
    if args.max_edges > 0:
        mat = mat.head(args.max_edges).copy()

    y = mat[annotators].to_numpy(dtype=float)
    (
        gamma_t,
        gamma_n0,
        gamma_n1,
        gamma_n2,
        alpha,
        beta0,
        beta1,
        beta2,
        pi,
        rho1,
        rho2,
        trace,
    ) = _fit_multifactor_em(y=y, max_iter=args.max_iter, tol=args.tol)

    posterior = mat[["source", "target"]].copy()
    posterior["posterior"] = gamma_t
    posterior["posterior_neg_clean"] = gamma_n0
    posterior["posterior_neg_burst1"] = gamma_n1
    posterior["posterior_neg_burst2"] = gamma_n2

    neg_mass = np.clip(gamma_n0 + gamma_n1 + gamma_n2, _EPS, 1.0)
    posterior["posterior_burst_mass_given_negative"] = (gamma_n1 + gamma_n2) / neg_mass

    obs = ~np.isnan(y)
    y0 = np.nan_to_num(y, nan=0.0)
    posterior["n_observed_sources"] = obs.sum(axis=1)
    posterior["n_positive_sources"] = (y0 * obs).sum(axis=1)

    reliability = pd.DataFrame(
        {
            "source_id": annotators,
            "alpha_sensitivity": alpha,
            "beta_false_positive_clean": beta0,
            "beta_false_positive_burst1": beta1,
            "beta_false_positive_burst2": beta2,
            "burst12_minus_clean": (beta1 + beta2) / 2.0 - beta0,
            "youden_clean": alpha - beta0,
            "observed_edges": obs.sum(axis=0),
        }
    )
    reliability = reliability.merge(source_meta[["source_id", "kind"]], on="source_id", how="left")
    reliability = reliability.sort_values("youden_clean", ascending=False).reset_index(drop=True)

    fit_summary = pd.DataFrame(
        [
            {
                "n_edges": len(mat),
                "n_annotators": len(annotators),
                "pi_prevalence": pi,
                "rho_burst1": rho1,
                "rho_burst2": rho2,
                "iterations": int(trace["iter"].max()),
                "final_log_likelihood": float(trace["log_likelihood"].iloc[-1]),
                "include_kinds": ",".join(include_kinds),
                "exclude_sources": ",".join(exclude_sources),
            }
        ]
    )

    posterior.to_csv(args.output_dir / "ltm_mf_edge_posteriors.csv", index=False)
    reliability.to_csv(args.output_dir / "ltm_mf_source_reliability.csv", index=False)
    trace.to_csv(args.output_dir / "ltm_mf_fit_trace.csv", index=False)
    fit_summary.to_csv(args.output_dir / "ltm_mf_fit_summary.csv", index=False)

    print("Multi-factor latent-truth fit complete:")
    print(f"- edges: {len(mat)}")
    print(f"- annotators: {len(annotators)}")
    print(f"- prevalence pi: {pi:.6f}")
    print(f"- burst rho1: {rho1:.6f}")
    print(f"- burst rho2: {rho2:.6f}")
    print(f"- iterations: {int(trace['iter'].max())}")
    print(f"- outputs: {args.output_dir}")


if __name__ == "__main__":
    main()
