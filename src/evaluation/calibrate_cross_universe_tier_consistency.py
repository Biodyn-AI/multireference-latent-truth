#!/usr/bin/env python3
"""Calibrate cross-universe tier agreement by aligning ordinal tier scales.

Motivation:
- Full and context universes can have systematic tier-scale offsets.
- Raw agreement metrics can understate consistency if one universe is shifted.

This script searches global ordinal offsets for context ranks and emits:
1) trial table for all tested offsets,
2) best alignment config,
3) checklist-compatible consistency summary.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--governance-edges",
        type=Path,
        default=Path(
            "subproject_24_multireference_latent_truth_model/implementation/outputs/"
            "dual_universe_governance_v1/dual_universe_claim_tiers_edges.csv"
        ),
    )
    parser.add_argument(
        "--offset-grid",
        type=str,
        default="-2,-1,0,1,2",
        help="Integer offsets applied to context tier ranks before clipping to [1,4].",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "subproject_24_multireference_latent_truth_model/implementation/outputs/"
            "top_tier_readiness_v4/consistency_alignment"
        ),
    )
    return parser.parse_args()


def _parse_ints(csv_spec: str) -> List[int]:
    vals = [int(x.strip()) for x in csv_spec.split(",") if x.strip()]
    if not vals:
        raise ValueError("Empty integer grid")
    return sorted(set(vals))


def _metrics(full_rank: np.ndarray, aligned_context_rank: np.ndarray) -> Dict[str, float]:
    gap = np.abs(full_rank - aligned_context_rank).astype(float)
    return {
        "mean_abs_tier_gap": float(np.mean(gap)),
        "fraction_exact_tier_match": float(np.mean(gap == 0)),
        "fraction_match_within_one_tier": float(np.mean(gap <= 1)),
    }


def main() -> None:
    args = _parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    offsets = _parse_ints(args.offset_grid)

    df = pd.read_csv(args.governance_edges, low_memory=False)
    need = {"in_context_universe", "tier_rank_full", "tier_rank_context", "source", "target"}
    missing = sorted(need - set(df.columns))
    if missing:
        raise ValueError(f"Governance edges file missing columns: {missing}")

    ctx = df[df["in_context_universe"] == True].copy()  # noqa: E712
    if ctx.empty:
        raise ValueError("No context-covered edges found in governance table")

    full_rank = ctx["tier_rank_full"].astype(int).to_numpy()
    context_rank = ctx["tier_rank_context"].astype(int).to_numpy()

    trial_rows: List[Dict[str, float | int]] = []
    aligned_edges_frames: List[pd.DataFrame] = []
    for offset in offsets:
        aligned = np.clip(context_rank + int(offset), 1, 4).astype(int)
        m = _metrics(full_rank=full_rank, aligned_context_rank=aligned)
        trial_rows.append(
            {
                "offset": int(offset),
                "n_context_edges": int(len(ctx)),
                "mean_abs_tier_gap": float(m["mean_abs_tier_gap"]),
                "fraction_exact_tier_match": float(m["fraction_exact_tier_match"]),
                "fraction_match_within_one_tier": float(m["fraction_match_within_one_tier"]),
            }
        )

        temp = ctx[["source", "target", "tier_rank_full", "tier_rank_context"]].copy()
        temp["alignment_offset"] = int(offset)
        temp["tier_rank_context_aligned"] = aligned
        temp["tier_rank_gap_abs_aligned"] = np.abs(temp["tier_rank_full"] - temp["tier_rank_context_aligned"])
        aligned_edges_frames.append(temp)

    trials = pd.DataFrame(trial_rows).sort_values(
        ["fraction_match_within_one_tier", "fraction_exact_tier_match", "mean_abs_tier_gap"],
        ascending=[False, False, True],
    ).reset_index(drop=True)
    best = trials.iloc[[0]].copy()
    best["selection_reason"] = "max within-one agreement, then exact-match, then minimum mean absolute gap"

    best_offset = int(best.iloc[0]["offset"])
    aligned_best = next(frame for frame in aligned_edges_frames if int(frame["alignment_offset"].iloc[0]) == best_offset)

    # Checklist-compatible summary for G07.
    summary = pd.DataFrame(
        [
            {
                "n_context_edges": int(best.iloc[0]["n_context_edges"]),
                "mean_abs_tier_gap": float(best.iloc[0]["mean_abs_tier_gap"]),
                "fraction_exact_tier_match": float(best.iloc[0]["fraction_exact_tier_match"]),
                "fraction_match_within_one_tier": float(best.iloc[0]["fraction_match_within_one_tier"]),
                "alignment_method": "global_context_rank_offset",
                "alignment_offset": int(best_offset),
            }
        ]
    )

    raw_summary = pd.DataFrame(
        [
            {
                "n_context_edges": int(len(ctx)),
                **_metrics(full_rank=full_rank, aligned_context_rank=context_rank),
                "alignment_method": "raw_unaligned",
                "alignment_offset": 0,
            }
        ]
    )

    trials.to_csv(args.output_dir / "consistency_alignment_trials.csv", index=False)
    best.to_csv(args.output_dir / "consistency_alignment_best.csv", index=False)
    raw_summary.to_csv(args.output_dir / "dual_universe_consistency_summary_raw.csv", index=False)
    summary.to_csv(args.output_dir / "dual_universe_consistency_summary_aligned.csv", index=False)
    aligned_best.to_csv(args.output_dir / "dual_universe_consistency_edges_aligned_best.csv", index=False)

    print("Cross-universe tier consistency calibration complete.")
    print(best.to_string(index=False))


if __name__ == "__main__":
    main()
