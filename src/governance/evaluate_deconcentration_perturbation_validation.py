#!/usr/bin/env python3
"""Evaluate perturbation validation for deconcentration promotion policies.

Compares selected promoted-edge sets against perturbation reference panels and
computes enrichment relative to random draws from the same promoted-candidate
universe.
"""

from __future__ import annotations

import argparse
from math import comb
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd


Edge = Tuple[str, str]
EdgeSet = set[Edge]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--selections",
        type=str,
        default=(
            "baseline=subproject_24_multireference_latent_truth_model/implementation/outputs/deconcentration_policy_sweep_v1/baseline_alpha0_selected_edges.csv,"
            "deconc_alpha005=subproject_24_multireference_latent_truth_model/implementation/outputs/deconcentration_policy_sweep_v1/penalized_alpha0p05_selected_edges.csv"
        ),
        help="Comma-separated label=selection_csv",
    )
    parser.add_argument(
        "--promoted-candidates",
        type=Path,
        default=Path(
            "subproject_24_multireference_latent_truth_model/implementation/outputs/dual_universe_governance_v1/dual_universe_claim_tiers_edges.csv"
        ),
    )
    parser.add_argument(
        "--full-universe-posterior",
        type=Path,
        default=Path(
            "subproject_24_multireference_latent_truth_model/implementation/outputs/full_v2_best_fullfit/ltm_family_hb_edge_posteriors.csv"
        ),
    )
    parser.add_argument(
        "--panel-glob",
        type=str,
        default="single_cell_mechinterp/outputs/perturb_validation/*/perturbation_edges.tsv",
    )
    parser.add_argument(
        "--min-panel-overlap",
        type=int,
        default=50,
        help="Panels with overlap >= threshold are treated as high-overlap panels.",
    )
    parser.add_argument(
        "--candidate-mode",
        type=str,
        default="promotion_allowed",
        choices=["promotion_allowed", "up_to_conditional", "all"],
        help=(
            "Candidate pool filter for random-expectation baseline: "
            "`promotion_allowed` (legacy), `up_to_conditional` (DUAL_TIER1/2/3), or `all`."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "subproject_24_multireference_latent_truth_model/implementation/outputs/deconcentration_validation_phase11"
        ),
    )
    return parser.parse_args()


def _parse_map(spec: str) -> Dict[str, Path]:
    out: Dict[str, Path] = {}
    for part in [x.strip() for x in spec.split(",") if x.strip()]:
        if "=" not in part:
            raise ValueError(f"Invalid map entry: {part}")
        k, v = part.split("=", 1)
        out[k.strip()] = Path(v.strip())
    if not out:
        raise ValueError("No entries parsed from --selections")
    return out


def _edge_set(df: pd.DataFrame) -> EdgeSet:
    return set(df[["source", "target"]].drop_duplicates().itertuples(index=False, name=None))


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


def _discover_panels(panel_glob: str) -> Dict[str, Path]:
    paths = sorted(Path(".").glob(panel_glob))
    if not paths:
        raise ValueError(f"No panel files found for --panel-glob {panel_glob}")
    return {p.parent.name: p for p in paths}


def _load_panel_edges(panel_map: Dict[str, Path]) -> Dict[str, EdgeSet]:
    out: Dict[str, EdgeSet] = {}
    for label, path in panel_map.items():
        df = pd.read_csv(path, sep="\t")
        need = {"source", "target"}
        if not need.issubset(df.columns):
            raise ValueError(f"Panel {label} missing columns {sorted(need)}")
        out[label] = _edge_set(df)
    return out


def _build_candidate_pool(df: pd.DataFrame, candidate_mode: str) -> pd.DataFrame:
    if candidate_mode == "promotion_allowed":
        if "promotion_allowed" not in df.columns:
            raise ValueError("--promoted-candidates missing 'promotion_allowed'")
        candidates = df[df["promotion_allowed"] == True].copy()  # noqa: E712
    elif candidate_mode == "up_to_conditional":
        need = "dual_universe_claim_tier"
        if need not in df.columns:
            raise ValueError(f"--promoted-candidates missing '{need}' for candidate mode {candidate_mode}")
        candidates = df[
            df["dual_universe_claim_tier"].isin(
                ["DUAL_TIER1_STRONG", "DUAL_TIER2_MODERATE", "DUAL_TIER3_CONDITIONAL"]
            )
        ].copy()
    elif candidate_mode == "all":
        candidates = df.copy()
    else:
        raise ValueError(f"Unsupported candidate_mode: {candidate_mode}")
    return candidates


def _union(sets: Iterable[EdgeSet]) -> EdgeSet:
    out: EdgeSet = set()
    for s in sets:
        out |= s
    return out


def _consensus(edges_by_panel: Dict[str, EdgeSet], panel_labels: Sequence[str], min_count: int = 2) -> EdgeSet:
    counts: Dict[Edge, int] = {}
    for panel in panel_labels:
        for edge in edges_by_panel[panel]:
            counts[edge] = counts.get(edge, 0) + 1
    return {e for e, c in counts.items() if c >= min_count}


def main() -> None:
    args = _parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    selection_map = _parse_map(args.selections)

    promoted_df = pd.read_csv(args.promoted_candidates, low_memory=False)
    promoted_df = _build_candidate_pool(df=promoted_df, candidate_mode=args.candidate_mode)
    promoted_candidates = _edge_set(promoted_df)
    N = len(promoted_candidates)
    if N == 0:
        raise ValueError("No promoted candidates found")

    full_df = pd.read_csv(args.full_universe_posterior)
    full_universe = _edge_set(full_df)

    panel_map = _discover_panels(args.panel_glob)
    panel_edges = _load_panel_edges(panel_map)

    panel_rows: List[Dict[str, object]] = []
    for label, edges in panel_edges.items():
        overlap = len(edges & full_universe)
        panel_rows.append(
            {
                "panel": label,
                "n_panel_edges": int(len(edges)),
                "n_edges_in_full_universe": int(overlap),
                "overlap_fraction_panel": float(overlap / max(len(edges), 1)),
                "is_high_overlap": bool(overlap >= args.min_panel_overlap),
            }
        )
    panel_summary = pd.DataFrame(panel_rows).sort_values("n_edges_in_full_universe", ascending=False).reset_index(
        drop=True
    )
    panel_summary.to_csv(args.output_dir / "panel_overlap_summary.csv", index=False)

    high_panels = panel_summary.loc[panel_summary["is_high_overlap"], "panel"].tolist()
    low_panels = panel_summary.loc[~panel_summary["is_high_overlap"], "panel"].tolist()

    reference_sets: Dict[str, EdgeSet] = {
        "high_overlap_union": _union(panel_edges[p] for p in high_panels) if high_panels else set(),
        "low_overlap_union": _union(panel_edges[p] for p in low_panels) if low_panels else set(),
        "all_union": _union(panel_edges.values()),
        "consensus_2plus_high_overlap": _consensus(panel_edges, high_panels, min_count=2) if high_panels else set(),
        "consensus_2plus_all": _consensus(panel_edges, list(panel_edges.keys()), min_count=2),
    }

    # Restrict references to full universe to match evaluated edge space.
    reference_sets_in_universe = {k: (v & full_universe) for k, v in reference_sets.items()}

    rows: List[Dict[str, object]] = []
    for sel_label, sel_path in selection_map.items():
        sel_df = pd.read_csv(sel_path)
        need = {"source", "target"}
        if not need.issubset(sel_df.columns):
            raise ValueError(f"Selection {sel_label} missing columns {sorted(need)}")
        sel_edges = _edge_set(sel_df)
        n_sel = len(sel_edges)
        outside = len(sel_edges - promoted_candidates)

        for ref_label, ref_edges in reference_sets_in_universe.items():
            hits = len(sel_edges & ref_edges)
            K = len(promoted_candidates & ref_edges)
            expected = n_sel * K / float(N)
            p_ge = _hypergeom_sf(hits, N=N, K=K, n=n_sel)

            rows.append(
                {
                    "selection": sel_label,
                    "selection_file": str(sel_path),
                    "candidate_mode": args.candidate_mode,
                    "reference_set": ref_label,
                    "n_selected": int(n_sel),
                    "n_selected_outside_promoted_candidates": int(outside),
                    "n_reference_edges_in_full_universe": int(len(ref_edges)),
                    "n_reference_edges_in_promoted_candidates": int(K),
                    "n_hits": int(hits),
                    "precision_in_selection": float(hits / max(n_sel, 1)),
                    "recall_in_reference_universe": float(hits / max(len(ref_edges), 1)),
                    "expected_hits_random_from_promoted_candidates": float(expected),
                    "enrichment_vs_random": float(hits / expected) if expected > 0 else float("nan"),
                    "p_value_hypergeom_ge_hits": float(p_ge),
                }
            )

    summary = pd.DataFrame(rows).sort_values(["reference_set", "selection"]).reset_index(drop=True)
    summary.to_csv(args.output_dir / "deconcentration_validation_summary.csv", index=False)

    pivot_cols = [
        "n_hits",
        "precision_in_selection",
        "enrichment_vs_random",
        "p_value_hypergeom_ge_hits",
    ]
    pivot = summary.pivot(index="reference_set", columns="selection", values=pivot_cols)
    pivot.columns = [f"{metric}__{selection}" for metric, selection in pivot.columns]
    pivot = pivot.reset_index()

    # Pairwise delta for two-policy primary comparison if both present.
    if {"baseline", "deconc_alpha005"}.issubset(set(summary["selection"])):
        base = summary[summary["selection"] == "baseline"].set_index("reference_set")
        decon = summary[summary["selection"] == "deconc_alpha005"].set_index("reference_set")
        for metric in [
            "n_hits",
            "precision_in_selection",
            "enrichment_vs_random",
            "p_value_hypergeom_ge_hits",
        ]:
            pivot[f"delta_{metric}__deconc_minus_baseline"] = (
                decon[metric] - base[metric]
            ).reindex(pivot["reference_set"]).to_numpy()

    pivot.to_csv(args.output_dir / "deconcentration_validation_comparison.csv", index=False)

    print("Deconcentration perturbation validation complete.")
    print(f"Promoted candidate universe N={N}")
    print(f"Candidate mode={args.candidate_mode}")
    print(f"High-overlap panels ({len(high_panels)}): {high_panels}")
    print(f"Low-overlap panels ({len(low_panels)}): {low_panels}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
