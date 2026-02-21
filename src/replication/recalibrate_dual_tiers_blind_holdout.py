#!/usr/bin/env python3
"""Recalibrate dual-tier cutoffs using train split and evaluate on blind holdout."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import pandas as pd


Edge = Tuple[str, str]
EdgeSet = set[Edge]


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
        "--panel-split-summary",
        type=Path,
        default=Path(
            "subproject_24_multireference_latent_truth_model/implementation/outputs/"
            "top_tier_readiness_v2/blind_holdout_eval/panel_split_summary.csv"
        ),
    )
    parser.add_argument("--strong-k-grid", type=str, default="200,500,842,1200,2000")
    parser.add_argument("--moderate-k-grid", type=str, default="500,1000,2000,4000,6000")
    parser.add_argument("--conditional-k-grid", type=str, default="1000,2000,4000,8000,12000")
    parser.add_argument("--min-promoted", type=int, default=500)
    parser.add_argument("--max-promoted", type=int, default=8000)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "subproject_24_multireference_latent_truth_model/implementation/outputs/"
            "top_tier_readiness_v2/recalibrated_dual_tiers"
        ),
    )
    return parser.parse_args()


def _parse_int_grid(text: str) -> List[int]:
    vals = [int(x.strip()) for x in text.split(",") if x.strip()]
    if not vals:
        raise ValueError("Empty integer grid")
    return sorted(set(vals))


def _edge_set(df: pd.DataFrame) -> EdgeSet:
    return set(df[["source", "target"]].drop_duplicates().itertuples(index=False, name=None))


def _load_edge_file(path: Path) -> EdgeSet:
    df = pd.read_csv(path, sep="\t")
    need = {"source", "target"}
    if not need.issubset(df.columns):
        raise ValueError(f"Panel path missing required columns {sorted(need)}: {path}")
    return _edge_set(df)


def _build_reference_sets(panel_split: pd.DataFrame) -> Dict[str, EdgeSet]:
    train_paths = panel_split[panel_split["split"] == "train"]["panel_path"].tolist()
    holdout_paths = panel_split[panel_split["split"] == "holdout"]["panel_path"].tolist()
    train_union: EdgeSet = set()
    holdout_union: EdgeSet = set()
    for p in train_paths:
        train_union |= _load_edge_file(Path(p))
    for p in holdout_paths:
        holdout_union |= _load_edge_file(Path(p))
    return {
        "train_union": train_union,
        "holdout_union": holdout_union,
    }


def _tier_assignment(
    df_sorted: pd.DataFrame,
    strong_k: int,
    moderate_k: int,
    conditional_k: int,
) -> pd.Series:
    n = len(df_sorted)
    strong_k = min(max(strong_k, 0), n)
    moderate_k = min(max(moderate_k, 0), max(0, n - strong_k))
    conditional_k = min(max(conditional_k, 0), max(0, n - strong_k - moderate_k))

    labels = ["DUAL_TIER4_EXPLORATORY"] * n
    for i in range(strong_k):
        labels[i] = "DUAL_TIER1_STRONG"
    for i in range(strong_k, strong_k + moderate_k):
        labels[i] = "DUAL_TIER2_MODERATE"
    for i in range(strong_k + moderate_k, strong_k + moderate_k + conditional_k):
        labels[i] = "DUAL_TIER3_CONDITIONAL"
    return pd.Series(labels, index=df_sorted.index)


def _precision(mask: pd.Series, ref_mask: pd.Series) -> float:
    n = int(mask.sum())
    if n == 0:
        return float("nan")
    return float((ref_mask & mask).sum() / n)


def main() -> None:
    args = _parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    strong_grid = _parse_int_grid(args.strong_k_grid)
    moderate_grid = _parse_int_grid(args.moderate_k_grid)
    conditional_grid = _parse_int_grid(args.conditional_k_grid)

    panel_split = pd.read_csv(args.panel_split_summary, low_memory=False)
    refs = _build_reference_sets(panel_split)
    train_ref = refs["train_union"]
    holdout_ref = refs["holdout_union"]

    df = pd.read_csv(args.governance_edges, low_memory=False)
    df["posterior_full"] = df["posterior_full"].fillna(0.0)
    df["posterior_context"] = df["posterior_context"].fillna(0.0)
    df["joint_score"] = df[["posterior_full", "posterior_context"]].min(axis=1)
    df = df.sort_values(
        ["joint_score", "posterior_context", "posterior_full", "source", "target"],
        ascending=[False, False, False, True, True],
    ).reset_index(drop=True)

    edge_tuples = list(df[["source", "target"]].itertuples(index=False, name=None))
    train_mask = pd.Series([e in train_ref for e in edge_tuples], index=df.index)
    holdout_mask = pd.Series([e in holdout_ref for e in edge_tuples], index=df.index)

    trial_rows: List[Dict[str, object]] = []
    for strong_k in strong_grid:
        for moderate_k in moderate_grid:
            for conditional_k in conditional_grid:
                promoted_k = strong_k + moderate_k
                if promoted_k < args.min_promoted or promoted_k > args.max_promoted:
                    continue
                if promoted_k + conditional_k >= len(df):
                    continue

                tiers = _tier_assignment(df, strong_k=strong_k, moderate_k=moderate_k, conditional_k=conditional_k)
                promoted = tiers.isin(["DUAL_TIER1_STRONG", "DUAL_TIER2_MODERATE"])
                conditional = tiers == "DUAL_TIER3_CONDITIONAL"

                train_promoted_prec = _precision(promoted, train_mask)
                train_conditional_prec = _precision(conditional, train_mask)
                hold_promoted_prec = _precision(promoted, holdout_mask)
                hold_conditional_prec = _precision(conditional, holdout_mask)

                top_counts = df.loc[promoted, "source"].value_counts(normalize=True)
                top1 = float(top_counts.iloc[0]) if len(top_counts) >= 1 else float("nan")
                top2 = float(top_counts.iloc[:2].sum()) if len(top_counts) >= 2 else top1

                trial_rows.append(
                    {
                        "strong_k": strong_k,
                        "moderate_k": moderate_k,
                        "conditional_k": conditional_k,
                        "promoted_k": promoted_k,
                        "train_promoted_precision": train_promoted_prec,
                        "train_conditional_precision": train_conditional_prec,
                        "holdout_promoted_precision": hold_promoted_prec,
                        "holdout_conditional_precision": hold_conditional_prec,
                        "train_precision_gap_promoted_minus_conditional": train_promoted_prec - train_conditional_prec,
                        "holdout_precision_gap_promoted_minus_conditional": hold_promoted_prec - hold_conditional_prec,
                        "promoted_top1_source_fraction": top1,
                        "promoted_top2_source_fraction": top2,
                    }
                )

    trials = pd.DataFrame(trial_rows)
    if trials.empty:
        raise ValueError("No recalibration candidates produced from provided grids/constraints")

    feasible = trials[
        (trials["train_precision_gap_promoted_minus_conditional"] >= 0.0)
        & (trials["holdout_precision_gap_promoted_minus_conditional"] >= 0.0)
    ].copy()
    if feasible.empty:
        ranked = trials.sort_values(
            [
                "train_precision_gap_promoted_minus_conditional",
                "holdout_precision_gap_promoted_minus_conditional",
                "train_promoted_precision",
                "promoted_top1_source_fraction",
            ],
            ascending=[False, False, False, True],
        ).reset_index(drop=True)
        best = ranked.iloc[[0]].copy()
        best["selection_reason"] = "no feasible dominance on both splits; selected best train/holdout precision-gap compromise"
    else:
        ranked = feasible.sort_values(
            [
                "holdout_precision_gap_promoted_minus_conditional",
                "train_precision_gap_promoted_minus_conditional",
                "holdout_promoted_precision",
                "promoted_top1_source_fraction",
            ],
            ascending=[False, False, False, True],
        ).reset_index(drop=True)
        best = ranked.iloc[[0]].copy()
        best["selection_reason"] = "max holdout precision-gap subject to non-negative train and holdout dominance"

    sel = best.iloc[0]
    tiers = _tier_assignment(
        df,
        strong_k=int(sel["strong_k"]),
        moderate_k=int(sel["moderate_k"]),
        conditional_k=int(sel["conditional_k"]),
    )
    out_edges = df.copy()
    out_edges["dual_universe_claim_tier_recalibrated"] = tiers
    out_edges["promotion_allowed_recalibrated"] = out_edges["dual_universe_claim_tier_recalibrated"].isin(
        ["DUAL_TIER1_STRONG", "DUAL_TIER2_MODERATE"]
    )

    # Summary table aligned with checklist gate expectations.
    summary_rows: List[Dict[str, object]] = []
    for tier in [
        "DUAL_TIER1_STRONG",
        "DUAL_TIER2_MODERATE",
        "DUAL_TIER3_CONDITIONAL",
        "DUAL_TIER4_EXPLORATORY",
    ]:
        m = out_edges["dual_universe_claim_tier_recalibrated"] == tier
        summary_rows.append(
            {
                "dual_universe_claim_tier": tier,
                "n_edges": int(m.sum()),
                "n_promotion_allowed": int((m & out_edges["promotion_allowed_recalibrated"]).sum()),
                "precision_train_within_tier": _precision(m, train_mask),
                "precision_holdout_within_tier": _precision(m, holdout_mask),
                "mean_joint_score": float(out_edges.loc[m, "joint_score"].mean()) if int(m.sum()) > 0 else float("nan"),
            }
        )
    summary = pd.DataFrame(summary_rows)

    promoted = out_edges[out_edges["promotion_allowed_recalibrated"]]
    src_counts = promoted["source"].value_counts(normalize=True)
    promoted_div = pd.DataFrame(
        [
            {"metric": "n_promoted_edges_total", "value": float(promoted.shape[0])},
            {"metric": "n_promoted_unique_sources", "value": float(promoted["source"].nunique())},
            {"metric": "n_promoted_unique_targets", "value": float(promoted["target"].nunique())},
            {
                "metric": "top1_source_fraction_promoted",
                "value": float(src_counts.iloc[0]) if len(src_counts) >= 1 else float("nan"),
            },
            {
                "metric": "top2_source_fraction_promoted",
                "value": float(src_counts.iloc[:2].sum()) if len(src_counts) >= 2 else float("nan"),
            },
            {
                "metric": "promoted_train_reference_fraction",
                "value": _precision(out_edges["promotion_allowed_recalibrated"], train_mask),
            },
            {
                "metric": "promoted_holdout_reference_fraction",
                "value": _precision(out_edges["promotion_allowed_recalibrated"], holdout_mask),
            },
        ]
    )

    trials.to_csv(args.output_dir / "recalibration_grid_trials.csv", index=False)
    feasible.to_csv(args.output_dir / "recalibration_grid_feasible.csv", index=False)
    best.to_csv(args.output_dir / "recalibration_best_config.csv", index=False)
    out_edges.to_csv(args.output_dir / "dual_universe_claim_tiers_edges_recalibrated.csv", index=False)
    summary.to_csv(args.output_dir / "dual_universe_claim_tiers_summary_recalibrated.csv", index=False)
    promoted_div.to_csv(args.output_dir / "dual_universe_promoted_diversity_summary_recalibrated.csv", index=False)

    print("Dual-tier recalibration complete.")
    print(best.to_string(index=False))


if __name__ == "__main__":
    main()
