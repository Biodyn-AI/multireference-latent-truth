#!/usr/bin/env python3
"""Enforce dual-universe claim governance for perturbation claim tiers.

This script combines full-universe and context-universe claim tiers for the same
model setting and assigns consolidated governance tiers that require cross-universe
consistency before promoting strong claims.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd


TIER_ORDER: Dict[str, int] = {
    "PERT_TIER1_STRONG": 4,
    "PERT_TIER2_MODERATE": 3,
    "PERT_TIER3_EXPLORATORY": 2,
    "PERT_TIER4_BACKGROUND": 1,
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--full-claims",
        type=Path,
        default=Path(
            "subproject_24_multireference_latent_truth_model/implementation/outputs/full_v2_best_fullfit/perturbation_multipanel/perturbation_multipanel_claim_tiers_edges.csv"
        ),
    )
    parser.add_argument(
        "--context-claims",
        type=Path,
        default=Path(
            "subproject_24_multireference_latent_truth_model/implementation/outputs/context_v2_best_fullfit/perturbation_multipanel/perturbation_multipanel_claim_tiers_edges.csv"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "subproject_24_multireference_latent_truth_model/implementation/outputs/dual_universe_governance_v1"
        ),
    )
    return parser.parse_args()


def _tier_score(series: pd.Series) -> pd.Series:
    return series.map(TIER_ORDER).fillna(0).astype(int)


def _governance_tier(
    full_tier: str | float,
    context_tier: str | float,
    context_available: bool,
) -> str:
    full_rank = int(TIER_ORDER.get(str(full_tier), 0))
    context_rank = int(TIER_ORDER.get(str(context_tier), 0)) if context_available else 0

    if context_available:
        # Strong promotion requires both universes to support at least moderate quality.
        if context_rank >= 4 and full_rank >= 3:
            return "DUAL_TIER1_STRONG"
        if context_rank >= 3 and full_rank >= 3:
            return "DUAL_TIER2_MODERATE"
        # Context strong but full weak -> conditional pending global support.
        if context_rank >= 3 and full_rank < 3:
            return "DUAL_TIER3_CONDITIONAL"
        # Full strong but context weak -> conditional pending context support.
        if full_rank >= 3 and context_rank < 3:
            return "DUAL_TIER3_CONDITIONAL"
        return "DUAL_TIER4_EXPLORATORY"

    # Edges absent from context universe cannot be promoted beyond conditional.
    if full_rank >= 3:
        return "DUAL_TIER3_CONDITIONAL"
    return "DUAL_TIER4_EXPLORATORY"


def main() -> None:
    args = _parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    full = pd.read_csv(args.full_claims)
    context = pd.read_csv(args.context_claims)
    need = {"source", "target", "posterior", "is_reference", "perturb_claim_tier"}
    if not need.issubset(full.columns):
        raise ValueError(f"--full-claims missing columns: {sorted(need - set(full.columns))}")
    if not need.issubset(context.columns):
        raise ValueError(f"--context-claims missing columns: {sorted(need - set(context.columns))}")

    full_small = full[["source", "target", "posterior", "is_reference", "perturb_claim_tier"]].copy()
    full_small = full_small.rename(
        columns={
            "posterior": "posterior_full",
            "is_reference": "is_reference_full",
            "perturb_claim_tier": "perturb_claim_tier_full",
        }
    )

    context_small = context[["source", "target", "posterior", "is_reference", "perturb_claim_tier"]].copy()
    context_small = context_small.rename(
        columns={
            "posterior": "posterior_context",
            "is_reference": "is_reference_context",
            "perturb_claim_tier": "perturb_claim_tier_context",
        }
    )

    merged = full_small.merge(context_small, on=["source", "target"], how="left")
    merged["in_context_universe"] = merged["perturb_claim_tier_context"].notna()
    merged["tier_rank_full"] = _tier_score(merged["perturb_claim_tier_full"])
    merged["tier_rank_context"] = _tier_score(merged["perturb_claim_tier_context"])
    merged["tier_rank_gap_abs"] = np.where(
        merged["in_context_universe"],
        np.abs(merged["tier_rank_full"] - merged["tier_rank_context"]),
        np.nan,
    )

    merged["dual_universe_claim_tier"] = merged.apply(
        lambda r: _governance_tier(
            full_tier=r["perturb_claim_tier_full"],
            context_tier=r["perturb_claim_tier_context"],
            context_available=bool(r["in_context_universe"]),
        ),
        axis=1,
    )

    merged["promotion_allowed"] = merged["dual_universe_claim_tier"].isin(
        ["DUAL_TIER1_STRONG", "DUAL_TIER2_MODERATE"]
    )

    # Tier-level summary with separate precision views.
    summary = (
        merged.groupby("dual_universe_claim_tier", as_index=False)
        .agg(
            n_edges=("source", "count"),
            n_context_edges=("in_context_universe", "sum"),
            n_reference_full=("is_reference_full", "sum"),
            mean_posterior_full=("posterior_full", "mean"),
            n_promotion_allowed=("promotion_allowed", "sum"),
        )
        .sort_values("dual_universe_claim_tier")
        .reset_index(drop=True)
    )
    summary["precision_full_within_tier"] = summary["n_reference_full"] / np.maximum(summary["n_edges"], 1)

    context_only = merged[merged["in_context_universe"]].copy()
    if not context_only.empty:
        context_summary = (
            context_only.groupby("dual_universe_claim_tier", as_index=False)
            .agg(
                n_reference_context=("is_reference_context", "sum"),
                mean_posterior_context=("posterior_context", "mean"),
            )
            .sort_values("dual_universe_claim_tier")
            .reset_index(drop=True)
        )
        summary = summary.merge(context_summary, on="dual_universe_claim_tier", how="left")
        summary["precision_context_within_tier"] = (
            summary["n_reference_context"] / np.maximum(summary["n_context_edges"], 1)
        )
    else:
        summary["n_reference_context"] = 0
        summary["mean_posterior_context"] = np.nan
        summary["precision_context_within_tier"] = np.nan

    # Pairwise tier-consistency diagnostics for context-covered edges.
    if not context_only.empty:
        consistency = pd.DataFrame(
            [
                {
                    "n_context_edges": int(len(context_only)),
                    "mean_abs_tier_gap": float(context_only["tier_rank_gap_abs"].mean()),
                    "fraction_exact_tier_match": float(np.mean(context_only["tier_rank_gap_abs"] == 0)),
                    "fraction_match_within_one_tier": float(np.mean(context_only["tier_rank_gap_abs"] <= 1)),
                }
            ]
        )
    else:
        consistency = pd.DataFrame(
            [
                {
                    "n_context_edges": 0,
                    "mean_abs_tier_gap": float("nan"),
                    "fraction_exact_tier_match": float("nan"),
                    "fraction_match_within_one_tier": float("nan"),
                }
            ]
        )

    merged.to_csv(args.output_dir / "dual_universe_claim_tiers_edges.csv", index=False)
    summary.to_csv(args.output_dir / "dual_universe_claim_tiers_summary.csv", index=False)
    consistency.to_csv(args.output_dir / "dual_universe_consistency_summary.csv", index=False)

    print("Dual-universe governance complete.")
    print(summary.to_string(index=False))
    print(consistency.to_string(index=False))


if __name__ == "__main__":
    main()
