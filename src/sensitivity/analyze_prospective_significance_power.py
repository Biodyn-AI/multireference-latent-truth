#!/usr/bin/env python3
"""Analyze significance gaps and sample-size requirements for prospective tests."""

from __future__ import annotations

import argparse
from math import comb, sqrt
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--validation-summary",
        type=Path,
        default=Path(
            "subproject_24_multireference_latent_truth_model/implementation/outputs/top_tier_readiness_v2/blind_holdout_eval/blind_holdout_validation_summary.csv"
        ),
    )
    parser.add_argument("--alpha", type=float, default=0.05, help="One-sided significance level.")
    parser.add_argument(
        "--target-enrichment",
        type=float,
        default=1.5,
        help="Target enrichment ratio for power/sample-size projection.",
    )
    parser.add_argument("--target-power", type=float, default=0.8, help="Target statistical power.")
    parser.add_argument(
        "--focus-reference-sets",
        type=str,
        default="train_high_overlap_union,blind_holdout_union",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "subproject_24_multireference_latent_truth_model/implementation/outputs/top_tier_readiness_v2/prospective_power_analysis"
        ),
    )
    return parser.parse_args()


def _hypergeom_sf(x: int, N: int, K: int, n: int) -> float:
    if N <= 0 or n < 0 or K < 0 or n > N:
        return float("nan")
    if K == 0:
        return 1.0 if x <= 0 else 0.0
    den = comb(N, n)
    if den == 0:
        return float("nan")
    lo = max(x, 0)
    hi = min(K, n)
    if lo > hi:
        return 0.0
    num = 0
    for i in range(lo, hi + 1):
        num += comb(K, i) * comb(N - K, n - i)
    return float(num / den)


def _min_hits_for_alpha(N: int, K: int, n: int, alpha: float) -> int:
    for h in range(0, n + 1):
        p = _hypergeom_sf(h, N=N, K=K, n=n)
        if p <= alpha:
            return h
    return n + 1


def _required_n_for_power(p0: float, p1: float, alpha: float, power: float) -> float:
    # Normal approximation, one-sided z quantiles (alpha=0.05, power=0.8 defaults).
    # For defaults: z_alpha ~= 1.64485, z_beta ~= 0.84162.
    # We keep a small lookup for common values and fallback to defaults.
    z_lookup = {
        (0.05, 0.8): (1.6448536269514722, 0.8416212335729143),
        (0.05, 0.9): (1.6448536269514722, 1.2815515655446004),
        (0.01, 0.8): (2.3263478740408408, 0.8416212335729143),
    }
    z_alpha, z_beta = z_lookup.get((round(alpha, 2), round(power, 1)), (1.6448536269514722, 0.8416212335729143))
    if not (0 < p0 < 1 and 0 < p1 < 1 and p1 > p0):
        return float("nan")
    num = (z_alpha * sqrt(p0 * (1 - p0)) + z_beta * sqrt(p1 * (1 - p1))) ** 2
    den = (p1 - p0) ** 2
    return float(num / den) if den > 0 else float("nan")


def main() -> None:
    args = _parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    focus_sets = {x.strip() for x in args.focus_reference_sets.split(",") if x.strip()}
    if not focus_sets:
        raise ValueError("No focus reference sets provided")

    val = pd.read_csv(args.validation_summary, low_memory=False)
    cov_col = "n_reference_edges_in_candidate_pool"
    if cov_col not in val.columns:
        if "n_reference_edges_in_promoted_candidates" in val.columns:
            cov_col = "n_reference_edges_in_promoted_candidates"
        else:
            raise ValueError(
                "Validation summary missing coverage column: expected "
                "`n_reference_edges_in_candidate_pool` or `n_reference_edges_in_promoted_candidates`"
            )

    # Infer candidate pool size N from any row with non-zero n and K.
    # expected = n * K / N  => N = n * K / expected
    tmp = val[(val[cov_col] > 0) & (val["n_selected"] > 0) & (val["expected_hits_random_from_candidate_pool"] > 0)].copy()
    if tmp.empty:
        # legacy column name fallback
        if "expected_hits_random_from_promoted_candidates" in val.columns:
            tmp = val[(val[cov_col] > 0) & (val["n_selected"] > 0) & (val["expected_hits_random_from_promoted_candidates"] > 0)].copy()
            exp_col = "expected_hits_random_from_promoted_candidates"
        else:
            raise ValueError("Could not infer candidate pool size from validation summary")
    else:
        exp_col = "expected_hits_random_from_candidate_pool"

    first = tmp.iloc[0]
    N = int(round(float(first["n_selected"]) * float(first[cov_col]) / float(first[exp_col])))
    if N <= 0:
        raise ValueError("Inferred invalid candidate-pool size")

    rows: List[Dict[str, object]] = []
    for row in val.itertuples(index=False):
        ref = str(row.reference_set)
        if ref not in focus_sets:
            continue
        n = int(row.n_selected)
        K = int(getattr(row, cov_col))
        hits = int(row.n_hits)
        expected = float(getattr(row, exp_col))
        enrichment = float(row.enrichment_vs_random) if pd.notna(row.enrichment_vs_random) else float("nan")

        min_hits_sig = _min_hits_for_alpha(N=N, K=K, n=n, alpha=args.alpha) if K > 0 else 0
        required_enrichment = float(min_hits_sig / expected) if expected > 0 else float("nan")
        p0 = float(K / N) if N > 0 else float("nan")
        p1 = min(float(args.target_enrichment * p0), 0.999999) if pd.notna(p0) else float("nan")
        n_required = _required_n_for_power(p0=p0, p1=p1, alpha=args.alpha, power=args.target_power)

        rows.append(
            {
                "selection": str(row.selection),
                "reference_set": ref,
                "N_candidate_pool": N,
                "K_reference_in_pool": K,
                "n_selected": n,
                "observed_hits": hits,
                "expected_hits_random": expected,
                "observed_enrichment": enrichment,
                "observed_p_value": float(row.p_value_hypergeom_ge_hits),
                "min_hits_for_alpha": min_hits_sig,
                "hit_gap_to_significance": int(min_hits_sig - hits),
                "required_enrichment_for_alpha": required_enrichment,
                "p0_random_hit_rate": p0,
                "p1_target_hit_rate": p1,
                "n_required_for_target_enrichment_and_power": n_required,
            }
        )

    out = pd.DataFrame(rows)
    out.to_csv(args.output_dir / "prospective_power_analysis_rows.csv", index=False)

    best = (
        out.sort_values(["reference_set", "observed_enrichment"], ascending=[True, False])
        .groupby("reference_set", as_index=False)
        .first()
    )
    best.to_csv(args.output_dir / "prospective_power_analysis_best_by_reference.csv", index=False)

    print("Prospective power analysis complete.")
    print(best.to_string(index=False))


if __name__ == "__main__":
    main()
