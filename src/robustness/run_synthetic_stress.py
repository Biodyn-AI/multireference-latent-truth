#!/usr/bin/env python3
"""Synthetic stress tests for the latent-truth estimator.

The suite evaluates whether EM latent-truth recovery remains reliable under:
- independent annotators,
- correlated false-positive bursts,
- coverage skew between positive and negative edges.
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
        "--reliability",
        type=Path,
        default=Path("subproject_24_multireference_latent_truth_model/implementation/outputs/ltm_em_source_reliability.csv"),
        help="EM reliability estimates used to seed synthetic annotator parameters.",
    )
    parser.add_argument(
        "--source-meta",
        type=Path,
        default=Path("subproject_24_multireference_latent_truth_model/implementation/outputs/source_metadata.csv"),
        help="Source metadata with coverage fractions.",
    )
    parser.add_argument("--n-edges", type=int, default=30000)
    parser.add_argument("--repeats", type=int, default=40)
    parser.add_argument("--prevalence", type=float, default=0.12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("subproject_24_multireference_latent_truth_model/implementation/outputs"),
    )
    return parser.parse_args()


def _roc_auc(y_true: np.ndarray, score: np.ndarray) -> float:
    y_true = y_true.astype(int)
    pos = int(np.sum(y_true == 1))
    neg = int(np.sum(y_true == 0))
    if pos == 0 or neg == 0:
        return float("nan")

    ranks = pd.Series(score).rank(method="average").to_numpy(dtype=float)
    sum_ranks_pos = float(np.sum(ranks[y_true == 1]))
    auc = (sum_ranks_pos - pos * (pos + 1) / 2.0) / float(pos * neg)
    return float(auc)


def _em_fit(y: np.ndarray, max_iter: int = 120, tol: float = 1e-6):
    obs = ~np.isnan(y)
    y0 = np.nan_to_num(y, nan=0.0)

    n_edges, n_ann = y.shape

    edge_pos_rate = np.where(obs.any(axis=1), np.nanmean(y, axis=1), 0.1)
    gamma = np.clip(edge_pos_rate, 0.01, 0.99)

    alpha = np.full(n_ann, 0.8, dtype=float)
    beta = np.full(n_ann, 0.1, dtype=float)
    pi = float(np.clip(np.mean(gamma), 0.01, 0.99))

    prev_ll = None
    for _ in range(max_iter):
        log_p1 = np.full(n_edges, np.log(pi + _EPS), dtype=float)
        log_p0 = np.full(n_edges, np.log(1.0 - pi + _EPS), dtype=float)

        for j in range(n_ann):
            mask = obs[:, j]
            if not np.any(mask):
                continue
            yj = y0[:, j]
            log_p1[mask] += yj[mask] * np.log(alpha[j] + _EPS) + (1 - yj[mask]) * np.log(
                1 - alpha[j] + _EPS
            )
            log_p0[mask] += yj[mask] * np.log(beta[j] + _EPS) + (1 - yj[mask]) * np.log(
                1 - beta[j] + _EPS
            )

        max_log = np.maximum(log_p1, log_p0)
        log_denom = max_log + np.log(np.exp(log_p1 - max_log) + np.exp(log_p0 - max_log))
        gamma = np.exp(log_p1 - log_denom)
        gamma = np.clip(gamma, _EPS, 1.0 - _EPS)

        pi = float(np.clip(np.mean(gamma), 0.001, 0.999))

        for j in range(n_ann):
            mask = obs[:, j]
            if not np.any(mask):
                alpha[j] = 0.5
                beta[j] = 0.5
                continue
            yj = y0[mask, j]
            gj = gamma[mask]
            alpha[j] = np.clip((np.sum(gj * yj) + 1.0) / (np.sum(gj) + 2.0), 0.001, 0.999)
            beta[j] = np.clip((np.sum((1.0 - gj) * yj) + 1.0) / (np.sum(1.0 - gj) + 2.0), 0.001, 0.999)

        ll = float(np.sum(log_denom))
        if prev_ll is not None and abs(ll - prev_ll) < tol:
            break
        prev_ll = ll

    return gamma, alpha, beta, pi


def _simulate_annotations(
    rng: np.random.Generator,
    n_edges: int,
    prevalence: float,
    alpha_true: np.ndarray,
    beta_true: np.ndarray,
    coverage: np.ndarray,
    regime: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate latent truth and observed annotations for one synthetic trial."""

    n_ann = len(alpha_true)
    z = (rng.random(n_edges) < prevalence).astype(int)

    y = np.full((n_edges, n_ann), np.nan, dtype=float)

    # Shared burst variable creates dependence in false positives for regime tests.
    if regime == "dependent_fp":
        burst = (rng.random(n_edges) < 0.08).astype(int)
    else:
        burst = np.zeros(n_edges, dtype=int)

    for j in range(n_ann):
        base_cov = coverage[j]

        if regime == "coverage_skew":
            # Positive edges are more likely to be observed than negatives.
            cov_pos = min(0.99, base_cov + 0.20)
            cov_neg = max(0.05, base_cov - 0.20)
            observed = np.where(z == 1, rng.random(n_edges) < cov_pos, rng.random(n_edges) < cov_neg)
        else:
            observed = rng.random(n_edges) < base_cov

        p = np.where(z == 1, alpha_true[j], beta_true[j])

        if regime == "dependent_fp":
            # During burst events, false-positive probability increases jointly.
            p = np.where((z == 0) & (burst == 1), np.minimum(0.95, p + 0.25), p)

        sample = (rng.random(n_edges) < p).astype(float)
        y[observed, j] = sample[observed]

    return z, y


def _build_annotator_params(reliability: pd.DataFrame, source_meta: pd.DataFrame) -> pd.DataFrame:
    merged = reliability.copy()
    if "kind" not in merged.columns:
        merged = merged.merge(source_meta[["source_id", "kind"]], on="source_id", how="left")
    merged = merged.merge(source_meta[["source_id", "coverage_fraction"]], on="source_id", how="left")

    # Keep only non-method sources so we stress-test the truth model under its intended regime.
    merged = merged[merged["kind"].isin(["reference", "mechanistic"])].copy()
    merged = merged.sort_values("source_id").reset_index(drop=True)

    merged["alpha_true"] = merged["alpha_sensitivity"].clip(0.02, 0.98)
    merged["beta_true"] = merged["beta_false_positive"].clip(0.02, 0.98)
    merged["coverage_true"] = merged["coverage_fraction"].fillna(0.5).clip(0.05, 0.99)
    return merged


def main() -> None:
    args = _parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    reliability = pd.read_csv(args.reliability)
    source_meta = pd.read_csv(args.source_meta)
    params = _build_annotator_params(reliability, source_meta)

    if params.empty:
        raise ValueError("No reference/mechanistic annotators available for synthetic stress tests.")

    alpha_true = params["alpha_true"].to_numpy(dtype=float)
    beta_true = params["beta_true"].to_numpy(dtype=float)
    coverage_true = params["coverage_true"].to_numpy(dtype=float)

    rng = np.random.default_rng(args.seed)

    regimes = ["independent", "dependent_fp", "coverage_skew"]
    rows: List[Dict[str, float]] = []

    for regime in regimes:
        for rep in range(args.repeats):
            z, y = _simulate_annotations(
                rng=rng,
                n_edges=args.n_edges,
                prevalence=args.prevalence,
                alpha_true=alpha_true,
                beta_true=beta_true,
                coverage=coverage_true,
                regime=regime,
            )

            gamma, alpha_est, beta_est, pi_est = _em_fit(y)

            auc = _roc_auc(z, gamma)
            brier = float(np.mean((gamma - z) ** 2))
            prevalence_error = float(abs(pi_est - np.mean(z)))
            alpha_mae = float(np.mean(np.abs(alpha_est - alpha_true)))
            beta_mae = float(np.mean(np.abs(beta_est - beta_true)))

            top_k = max(1, int(0.10 * len(gamma)))
            top_idx = np.argpartition(gamma, -top_k)[-top_k:]
            top_precision = float(np.mean(z[top_idx]))

            rows.append(
                {
                    "regime": regime,
                    "repeat": rep,
                    "auc": auc,
                    "brier": brier,
                    "prevalence_error": prevalence_error,
                    "alpha_mae": alpha_mae,
                    "beta_mae": beta_mae,
                    "top10_precision": top_precision,
                }
            )

    trials = pd.DataFrame(rows)
    summary = (
        trials.groupby("regime", as_index=False)
        .agg(
            auc_mean=("auc", "mean"),
            auc_std=("auc", "std"),
            brier_mean=("brier", "mean"),
            prevalence_error_mean=("prevalence_error", "mean"),
            alpha_mae_mean=("alpha_mae", "mean"),
            beta_mae_mean=("beta_mae", "mean"),
            top10_precision_mean=("top10_precision", "mean"),
        )
        .sort_values("regime")
        .reset_index(drop=True)
    )

    params.to_csv(args.output_dir / "synthetic_annotator_params.csv", index=False)
    trials.to_csv(args.output_dir / "synthetic_stress_trials.csv", index=False)
    summary.to_csv(args.output_dir / "synthetic_stress_summary.csv", index=False)

    print("Synthetic stress tests complete:")
    print(f"- annotators: {len(params)}")
    print(f"- trials: {len(trials)}")
    print(f"- outputs: {args.output_dir}")


if __name__ == "__main__":
    main()
