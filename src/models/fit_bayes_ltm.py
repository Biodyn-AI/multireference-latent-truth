#!/usr/bin/env python3
"""Fit a Bayesian latent-truth model with Gibbs sampling.

This sampler mirrors the EM model but propagates uncertainty over:
- latent edge truth z_i,
- source sensitivity alpha_j,
- source false-positive rate beta_j,
- prevalence pi.

The implementation is intentionally lightweight and CPU-feasible.
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
    parser.add_argument(
        "--include-kinds",
        type=str,
        default="reference,mechanistic",
    )
    parser.add_argument(
        "--exclude-sources",
        type=str,
        default="",
    )
    parser.add_argument("--max-edges", type=int, default=0)
    parser.add_argument("--num-iter", type=int, default=120)
    parser.add_argument("--burn-in", type=int, default=40)
    parser.add_argument("--thin", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("subproject_24_multireference_latent_truth_model/implementation/outputs"),
    )
    return parser.parse_args()


def _select_annotators(
    source_meta: pd.DataFrame,
    include_kinds: List[str],
    exclude_sources: List[str],
) -> List[str]:
    sel = source_meta[source_meta["kind"].isin(include_kinds)].copy()
    if exclude_sources:
        sel = sel[~sel["source_id"].isin(exclude_sources)]
    sel = sel.sort_values("source_id").reset_index(drop=True)
    return sel["source_id"].tolist()


def _sample_beta(rng: np.random.Generator, a: float, b: float) -> float:
    return float(rng.beta(max(a, _EPS), max(b, _EPS)))


def main() -> None:
    args = _parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)

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
    obs = ~np.isnan(y)
    y0 = np.nan_to_num(y, nan=0.0)

    n_edges, n_ann = y.shape

    # Priors are intentionally weakly-informative to keep the sampler data-driven.
    a_pi, b_pi = 1.0, 9.0
    a_alpha, b_alpha = 2.0, 2.0
    a_beta, b_beta = 2.0, 2.0

    # Initialize latent truth from observed edge positive fraction.
    edge_pos_rate = np.where(obs.any(axis=1), np.nanmean(y, axis=1), 0.1)
    z = (edge_pos_rate > np.nanmedian(edge_pos_rate)).astype(int)

    pi = 0.1
    alpha = np.full(n_ann, 0.8, dtype=float)
    beta = np.full(n_ann, 0.1, dtype=float)

    trace_rows = []
    alpha_samples = []
    beta_samples = []
    pi_samples = []

    gamma_sum = np.zeros(n_edges, dtype=float)
    sample_count = 0

    for it in range(1, args.num_iter + 1):
        # Sample prevalence.
        pi = _sample_beta(rng, a_pi + np.sum(z), b_pi + (n_edges - np.sum(z)))

        # Sample source reliabilities.
        for j in range(n_ann):
            mask = obs[:, j]
            if not np.any(mask):
                alpha[j] = 0.5
                beta[j] = 0.5
                continue

            yj = y0[mask, j]
            zj = z[mask]

            alpha_pos = np.sum((zj == 1) & (yj == 1))
            alpha_neg = np.sum((zj == 1) & (yj == 0))
            beta_pos = np.sum((zj == 0) & (yj == 1))
            beta_neg = np.sum((zj == 0) & (yj == 0))

            alpha[j] = _sample_beta(rng, a_alpha + alpha_pos, b_alpha + alpha_neg)
            beta[j] = _sample_beta(rng, a_beta + beta_pos, b_beta + beta_neg)

            alpha[j] = float(np.clip(alpha[j], 0.001, 0.999))
            beta[j] = float(np.clip(beta[j], 0.001, 0.999))

        # Sample latent edge truth z_i.
        logit = np.full(n_edges, np.log(pi + _EPS) - np.log(1.0 - pi + _EPS), dtype=float)

        for j in range(n_ann):
            mask = obs[:, j]
            yj = y0[:, j]
            if not np.any(mask):
                continue

            term1 = yj * (np.log(alpha[j] + _EPS) - np.log(beta[j] + _EPS))
            term0 = (1.0 - yj) * (
                np.log(1.0 - alpha[j] + _EPS) - np.log(1.0 - beta[j] + _EPS)
            )
            update = term1 + term0
            logit[mask] += update[mask]

        p = 1.0 / (1.0 + np.exp(-np.clip(logit, -30.0, 30.0)))
        z = (rng.random(n_edges) < p).astype(int)

        trace_rows.append(
            {
                "iter": it,
                "pi": float(pi),
                "mean_z": float(np.mean(z)),
                "mean_p": float(np.mean(p)),
            }
        )

        if it > args.burn_in and (it - args.burn_in) % max(args.thin, 1) == 0:
            gamma_sum += z
            sample_count += 1
            alpha_samples.append(alpha.copy())
            beta_samples.append(beta.copy())
            pi_samples.append(pi)

    if sample_count == 0:
        raise ValueError("No posterior samples retained. Reduce --burn-in or --thin.")

    posterior_mean = gamma_sum / float(sample_count)

    posterior = mat[["source", "target"]].copy()
    posterior["posterior_mean"] = posterior_mean
    posterior["n_observed_sources"] = obs.sum(axis=1)
    posterior["n_positive_sources"] = (y0 * obs).sum(axis=1)

    alpha_arr = np.vstack(alpha_samples)
    beta_arr = np.vstack(beta_samples)
    pi_arr = np.asarray(pi_samples)

    reliability = pd.DataFrame(
        {
            "source_id": annotators,
            "alpha_mean": alpha_arr.mean(axis=0),
            "alpha_q05": np.quantile(alpha_arr, 0.05, axis=0),
            "alpha_q50": np.quantile(alpha_arr, 0.50, axis=0),
            "alpha_q95": np.quantile(alpha_arr, 0.95, axis=0),
            "beta_mean": beta_arr.mean(axis=0),
            "beta_q05": np.quantile(beta_arr, 0.05, axis=0),
            "beta_q50": np.quantile(beta_arr, 0.50, axis=0),
            "beta_q95": np.quantile(beta_arr, 0.95, axis=0),
        }
    )
    reliability["youden_mean"] = reliability["alpha_mean"] - reliability["beta_mean"]
    reliability = reliability.merge(source_meta[["source_id", "kind"]], on="source_id", how="left")
    reliability = reliability.sort_values("youden_mean", ascending=False).reset_index(drop=True)

    fit_summary = pd.DataFrame(
        [
            {
                "n_edges": n_edges,
                "n_annotators": n_ann,
                "num_iter": args.num_iter,
                "burn_in": args.burn_in,
                "thin": args.thin,
                "retained_samples": sample_count,
                "pi_mean": float(np.mean(pi_arr)),
                "pi_q05": float(np.quantile(pi_arr, 0.05)),
                "pi_q95": float(np.quantile(pi_arr, 0.95)),
                "include_kinds": ",".join(include_kinds),
                "exclude_sources": ",".join(exclude_sources),
            }
        ]
    )

    pd.DataFrame(trace_rows).to_csv(args.output_dir / "ltm_bayes_fit_trace.csv", index=False)
    posterior.to_csv(args.output_dir / "ltm_bayes_edge_posteriors.csv", index=False)
    reliability.to_csv(args.output_dir / "ltm_bayes_source_reliability.csv", index=False)
    fit_summary.to_csv(args.output_dir / "ltm_bayes_fit_summary.csv", index=False)

    print("Bayesian latent-truth fit complete:")
    print(f"- edges: {n_edges}")
    print(f"- annotators: {n_ann}")
    print(f"- retained posterior samples: {sample_count}")
    print(f"- outputs: {args.output_dir}")


if __name__ == "__main__":
    main()
