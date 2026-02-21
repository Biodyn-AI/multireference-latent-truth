#!/usr/bin/env python3
"""Fit a tissue-aware hierarchical latent truth model on invariant edge panel.

This model uses:
- latent edge truth z_i,
- binary reference annotators (TRRUST, DoRothEA),
- tissue-specific continuous emissions from causal effect magnitudes.

Continuous component:
|effect_mean| per tissue is modeled as Gaussian conditioned on z_i,
with mild shrinkage toward class-specific global means to stabilize
small-sample tissue parameters.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


_EPS = 1e-9


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--edge-scores",
        type=Path,
        default=Path("reports/invariant_causal_edges/artifacts/edge_scores_relaxed.tsv"),
        help="Edge panel with tissue effect columns and reference labels.",
    )
    parser.add_argument("--max-iter", type=int, default=300)
    parser.add_argument("--tol", type=float, default=1e-7)
    parser.add_argument(
        "--shrinkage",
        type=float,
        default=5.0,
        help="Pseudo-count shrinkage weight toward class-global tissue means.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("subproject_24_multireference_latent_truth_model/implementation/outputs"),
    )
    return parser.parse_args()


def _gaussian_logpdf(x: np.ndarray, mu: float, var: float) -> np.ndarray:
    var = max(var, 1e-8)
    return -0.5 * (np.log(2.0 * np.pi * var) + ((x - mu) ** 2) / var)


def _average_precision_binary(y_true: np.ndarray, scores: np.ndarray) -> float:
    y_true = y_true.astype(int)
    if np.sum(y_true) == 0:
        return float("nan")
    order = np.argsort(-scores)
    y = y_true[order]
    tp = np.cumsum(y)
    k = np.arange(1, len(y) + 1)
    precision = tp / k
    return float(np.sum(precision * y) / np.sum(y))


def main() -> None:
    args = _parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.edge_scores, sep="\t")

    tissues = ["kidney", "lung", "immune", "external_lung"]
    x_cols = [f"{t}_effect_mean" for t in tissues]

    needed = ["source", "target", "label_trrust", "label_dorothea", *x_cols]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {args.edge_scores}: {missing}")

    work = df[needed].copy()

    # Convert tissue effects to magnitudes; sign is already handled by invariance metrics.
    for c in x_cols:
        work[c] = work[c].abs().astype(float)

    y = work[["label_trrust", "label_dorothea"]].astype(float).to_numpy()
    x = work[x_cols].to_numpy(dtype=float)

    n_edges = len(work)
    n_ref = y.shape[1]
    n_tissues = x.shape[1]

    # Initialization from observed reference union to avoid arbitrary random starts.
    y_union = (y.max(axis=1) > 0).astype(float)
    gamma = np.clip(0.2 + 0.6 * y_union, 0.05, 0.95)

    pi = float(np.clip(np.mean(gamma), 0.05, 0.95))

    alpha = np.array([0.85, 0.85], dtype=float)
    beta = np.array([0.15, 0.15], dtype=float)

    mu1 = np.nanmean(x, axis=0) + 0.01
    mu0 = np.maximum(np.nanmean(x, axis=0) - 0.01, 1e-4)
    var1 = np.full(n_tissues, np.nanvar(x) + 1e-4, dtype=float)
    var0 = np.full(n_tissues, np.nanvar(x) + 1e-4, dtype=float)

    trace_rows: List[Dict[str, float]] = []

    for it in range(1, args.max_iter + 1):
        # E-step
        log_p1 = np.full(n_edges, np.log(pi + _EPS), dtype=float)
        log_p0 = np.full(n_edges, np.log(1.0 - pi + _EPS), dtype=float)

        for j in range(n_ref):
            yj = y[:, j]
            log_p1 += yj * np.log(alpha[j] + _EPS) + (1.0 - yj) * np.log(1.0 - alpha[j] + _EPS)
            log_p0 += yj * np.log(beta[j] + _EPS) + (1.0 - yj) * np.log(1.0 - beta[j] + _EPS)

        for t in range(n_tissues):
            xt = x[:, t]
            log_p1 += _gaussian_logpdf(xt, mu1[t], var1[t])
            log_p0 += _gaussian_logpdf(xt, mu0[t], var0[t])

        max_log = np.maximum(log_p1, log_p0)
        log_denom = max_log + np.log(np.exp(log_p1 - max_log) + np.exp(log_p0 - max_log))
        gamma = np.exp(log_p1 - log_denom)
        gamma = np.clip(gamma, _EPS, 1.0 - _EPS)

        # M-step
        pi = float(np.clip(np.mean(gamma), 0.001, 0.999))

        for j in range(n_ref):
            yj = y[:, j]
            alpha[j] = np.clip((np.sum(gamma * yj) + 1.0) / (np.sum(gamma) + 2.0), 0.001, 0.999)
            beta[j] = np.clip(
                (np.sum((1.0 - gamma) * yj) + 1.0) / (np.sum(1.0 - gamma) + 2.0),
                0.001,
                0.999,
            )

        gsum = np.sum(gamma)
        ngsum = np.sum(1.0 - gamma)

        # Global class means for shrinkage targets.
        global_mu1 = float(np.sum(gamma[:, None] * x) / max(gsum * n_tissues, _EPS))
        global_mu0 = float(np.sum((1.0 - gamma)[:, None] * x) / max(ngsum * n_tissues, _EPS))

        for t in range(n_tissues):
            xt = x[:, t]

            raw_mu1 = float(np.sum(gamma * xt) / max(gsum, _EPS))
            raw_mu0 = float(np.sum((1.0 - gamma) * xt) / max(ngsum, _EPS))

            # Empirical-Bayes style shrinkage toward class-global means.
            mu1[t] = (gsum * raw_mu1 + args.shrinkage * global_mu1) / (gsum + args.shrinkage)
            mu0[t] = (ngsum * raw_mu0 + args.shrinkage * global_mu0) / (ngsum + args.shrinkage)

            var1[t] = (
                np.sum(gamma * ((xt - mu1[t]) ** 2)) + args.shrinkage * np.nanvar(x)
            ) / max(gsum + args.shrinkage, _EPS)
            var0[t] = (
                np.sum((1.0 - gamma) * ((xt - mu0[t]) ** 2)) + args.shrinkage * np.nanvar(x)
            ) / max(ngsum + args.shrinkage, _EPS)

            var1[t] = max(var1[t], 1e-8)
            var0[t] = max(var0[t], 1e-8)

        ll = float(np.sum(log_denom))
        trace_rows.append({"iter": it, "log_likelihood": ll, "pi": pi})

        if it > 1 and abs(trace_rows[-1]["log_likelihood"] - trace_rows[-2]["log_likelihood"]) < args.tol:
            break

    posterior = work[["source", "target"]].copy()
    posterior["posterior"] = gamma
    posterior["label_union"] = y_union
    posterior["causal_mean_abs_score"] = x.mean(axis=1)

    # Quick discriminative checks against observed union labels (not latent truth).
    ap_posterior = _average_precision_binary(y_union, gamma)
    ap_effect = _average_precision_binary(y_union, posterior["causal_mean_abs_score"].to_numpy())

    ref_params = pd.DataFrame(
        {
            "reference": ["trrust", "dorothea"],
            "alpha_sensitivity": alpha,
            "beta_false_positive": beta,
            "youden_j": alpha - beta,
        }
    )

    tissue_params = pd.DataFrame(
        {
            "tissue": tissues,
            "mu_pos": mu1,
            "mu_neg": mu0,
            "delta_mu": mu1 - mu0,
            "sigma_pos": np.sqrt(var1),
            "sigma_neg": np.sqrt(var0),
        }
    ).sort_values("delta_mu", ascending=False)

    fit_summary = pd.DataFrame(
        [
            {
                "n_edges": n_edges,
                "n_tissues": n_tissues,
                "n_references": n_ref,
                "pi_prevalence": pi,
                "iterations": int(trace_rows[-1]["iter"]),
                "final_log_likelihood": float(trace_rows[-1]["log_likelihood"]),
                "ap_vs_label_union_posterior": ap_posterior,
                "ap_vs_label_union_mean_abs_effect": ap_effect,
                "ap_gain_posterior_minus_effect": ap_posterior - ap_effect,
                "shrinkage": args.shrinkage,
            }
        ]
    )

    out_prefix = args.output_dir / "tissue_hierarchical_ltm"
    posterior.to_csv(f"{out_prefix}_edge_posteriors.csv", index=False)
    ref_params.to_csv(f"{out_prefix}_reference_params.csv", index=False)
    tissue_params.to_csv(f"{out_prefix}_tissue_params.csv", index=False)
    pd.DataFrame(trace_rows).to_csv(f"{out_prefix}_fit_trace.csv", index=False)
    fit_summary.to_csv(f"{out_prefix}_fit_summary.csv", index=False)

    print("Tissue-hierarchical latent-truth fit complete:")
    print(f"- edges: {n_edges}")
    print(f"- prevalence pi: {pi:.6f}")
    print(f"- AP posterior vs label_union: {ap_posterior:.6f}")
    print(f"- AP mean_abs_effect vs label_union: {ap_effect:.6f}")
    print(f"- outputs prefix: {out_prefix}")


if __name__ == "__main__":
    main()
