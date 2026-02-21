#!/usr/bin/env python3
"""Run strict blind-holdout validation for promotion policy selections.

Protocol:
1) build train/holdout panel split from perturbation panel overlap,
2) keep holdout panels fully excluded from tuning-side metrics,
3) evaluate candidate selections against both train and blind holdout sets.
"""

from __future__ import annotations

import argparse
from math import comb
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import pandas as pd


Edge = Tuple[str, str]
EdgeSet = set[Edge]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--selection-dir",
        type=Path,
        default=Path(
            "subproject_24_multireference_latent_truth_model/implementation/outputs/"
            "top_tier_readiness_v1/remediation_up_to_conditional_budget842/policy"
        ),
        help="Directory containing policy selection files named '*_selected_edges.csv'.",
    )
    parser.add_argument(
        "--selection-pattern",
        type=str,
        default="*_selected_edges.csv",
    )
    parser.add_argument(
        "--governance-edges",
        type=Path,
        default=Path(
            "subproject_24_multireference_latent_truth_model/implementation/outputs/"
            "dual_universe_governance_v1/dual_universe_claim_tiers_edges.csv"
        ),
    )
    parser.add_argument(
        "--candidate-mode",
        type=str,
        default="up_to_conditional",
        choices=["promotion_allowed", "up_to_conditional", "all"],
    )
    parser.add_argument(
        "--full-universe-posterior",
        type=Path,
        default=Path(
            "subproject_24_multireference_latent_truth_model/implementation/outputs/"
            "full_v2_best_fullfit/ltm_family_hb_edge_posteriors.csv"
        ),
    )
    parser.add_argument(
        "--primary-panel-glob",
        type=str,
        default="single_cell_mechinterp/outputs/perturb_validation/*/perturbation_edges.tsv",
        help="Panels from this glob are split into train/holdout by overlap threshold.",
    )
    parser.add_argument(
        "--primary-panel-list",
        type=Path,
        default=None,
        help=(
            "Optional newline list of panel paths for the primary panel family. "
            "If provided, --primary-panel-glob is ignored."
        ),
    )
    parser.add_argument(
        "--extra-panel-glob",
        type=str,
        default="single_cell_mechinterp/outputs/head_layer_checkpoint_replication/*/perturbation_edges.tsv",
        help="Panels from this glob are always holdout-only (independent anchors).",
    )
    parser.add_argument(
        "--extra-panel-list",
        type=Path,
        default=None,
        help=(
            "Optional newline list of panel paths for the extra holdout family. "
            "If provided, --extra-panel-glob is ignored."
        ),
    )
    parser.add_argument(
        "--min-train-overlap",
        type=int,
        default=50,
        help="Primary panels with overlap >= threshold are assigned to train split.",
    )
    parser.add_argument(
        "--comparison-selection",
        type=str,
        default="baseline_alpha0",
        help="Selection label used as baseline for pairwise deltas in comparison table.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "subproject_24_multireference_latent_truth_model/implementation/outputs/"
            "blind_holdout_protocol_v1"
        ),
    )
    return parser.parse_args()


def _parse_selection_label(path: Path) -> str:
    name = path.name
    if not name.endswith("_selected_edges.csv"):
        return name
    return name[: -len("_selected_edges.csv")]


def _edge_set(df: pd.DataFrame) -> EdgeSet:
    return set(df[["source", "target"]].drop_duplicates().itertuples(index=False, name=None))


def _union(sets: Iterable[EdgeSet]) -> EdgeSet:
    out: EdgeSet = set()
    for s in sets:
        out |= s
    return out


def _consensus(sets: Sequence[EdgeSet], min_count: int = 2) -> EdgeSet:
    counts: Dict[Edge, int] = {}
    for s in sets:
        for e in s:
            counts[e] = counts.get(e, 0) + 1
    return {e for e, c in counts.items() if c >= min_count}


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


def _build_candidate_pool(df: pd.DataFrame, candidate_mode: str) -> pd.DataFrame:
    if candidate_mode == "promotion_allowed":
        if "promotion_allowed" not in df.columns:
            raise ValueError("governance file missing `promotion_allowed`")
        out = df[df["promotion_allowed"] == True].copy()  # noqa: E712
    elif candidate_mode == "up_to_conditional":
        need = "dual_universe_claim_tier"
        if need not in df.columns:
            raise ValueError("governance file missing `dual_universe_claim_tier`")
        out = df[
            df["dual_universe_claim_tier"].isin(
                ["DUAL_TIER1_STRONG", "DUAL_TIER2_MODERATE", "DUAL_TIER3_CONDITIONAL"]
            )
        ].copy()
    elif candidate_mode == "all":
        out = df.copy()
    else:
        raise ValueError(f"Unsupported candidate-mode: {candidate_mode}")
    return out


def _discover_panel_paths(pattern: str) -> Dict[str, Path]:
    paths = sorted(Path(".").glob(pattern))
    out: Dict[str, Path] = {}
    for path in paths:
        label = path.parent.name
        out[label] = path
    return out


def _read_panel_list(path: Path) -> Dict[str, Path]:
    if not path.exists():
        raise ValueError(f"Panel list path not found: {path}")
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines:
        raise ValueError(f"Panel list is empty: {path}")

    out: Dict[str, Path] = {}
    for raw in lines:
        pp = Path(raw)
        label = pp.parent.name
        if label in out:
            # Deterministic disambiguation if two directories share the same leaf.
            label = str(pp.parent).replace("/", "__")
        out[label] = pp
    return out


def _load_panel_edges(panel_paths: Dict[str, Path], full_universe: EdgeSet) -> Dict[str, EdgeSet]:
    out: Dict[str, EdgeSet] = {}
    for label, path in panel_paths.items():
        df = pd.read_csv(path, sep="\t")
        need = {"source", "target"}
        if not need.issubset(df.columns):
            raise ValueError(f"Panel {label} missing required columns {sorted(need)}")
        out[label] = _edge_set(df) & full_universe
    return out


def main() -> None:
    args = _parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    selection_files = sorted(args.selection_dir.glob(args.selection_pattern))
    if not selection_files:
        raise ValueError(f"No selection files matched {args.selection_pattern} in {args.selection_dir}")
    selections = {_parse_selection_label(p): p for p in selection_files}

    full_df = pd.read_csv(args.full_universe_posterior, low_memory=False)
    full_universe = _edge_set(full_df)

    gov_df = pd.read_csv(args.governance_edges, low_memory=False)
    candidate_pool_df = _build_candidate_pool(gov_df, args.candidate_mode)
    candidate_pool = _edge_set(candidate_pool_df)
    N = len(candidate_pool)
    if N == 0:
        raise ValueError("Candidate pool is empty for selected candidate mode")

    if args.primary_panel_list is not None:
        primary_paths = _read_panel_list(args.primary_panel_list)
    else:
        primary_paths = _discover_panel_paths(args.primary_panel_glob)

    if args.extra_panel_list is not None:
        extra_paths = _read_panel_list(args.extra_panel_list)
    else:
        extra_paths = _discover_panel_paths(args.extra_panel_glob)
    if not primary_paths and not extra_paths:
        raise ValueError("No panel files discovered from primary/extra globs")

    primary_edges = _load_panel_edges(primary_paths, full_universe)
    extra_edges = _load_panel_edges(extra_paths, full_universe)

    panel_rows: List[Dict[str, object]] = []
    train_primary: List[str] = []
    holdout_primary: List[str] = []
    for label, edges in primary_edges.items():
        split = "train" if len(edges) >= args.min_train_overlap else "holdout"
        if split == "train":
            train_primary.append(label)
        else:
            holdout_primary.append(label)
        panel_rows.append(
            {
                "panel": label,
                "panel_group": "primary",
                "split": split,
                "n_edges_in_full_universe": int(len(edges)),
                "panel_path": str(primary_paths[label]),
            }
        )
    for label, edges in extra_edges.items():
        panel_rows.append(
            {
                "panel": label,
                "panel_group": "extra",
                "split": "holdout",
                "n_edges_in_full_universe": int(len(edges)),
                "panel_path": str(extra_paths[label]),
            }
        )

    panel_split = pd.DataFrame(panel_rows).sort_values(
        ["split", "panel_group", "n_edges_in_full_universe", "panel"],
        ascending=[True, True, False, True],
    )
    panel_split.to_csv(args.output_dir / "panel_split_summary.csv", index=False)

    train_sets = [primary_edges[x] for x in train_primary]
    holdout_primary_sets = [primary_edges[x] for x in holdout_primary]
    holdout_extra_sets = list(extra_edges.values())
    holdout_sets = holdout_primary_sets + holdout_extra_sets

    reference_sets: Dict[str, EdgeSet] = {
        "train_high_overlap_union": _union(train_sets),
        "train_high_overlap_consensus_2plus": _consensus(train_sets, min_count=2) if train_sets else set(),
        "blind_holdout_primary_union": _union(holdout_primary_sets),
        "blind_holdout_extra_union": _union(holdout_extra_sets),
        "blind_holdout_union": _union(holdout_sets),
        "blind_holdout_consensus_2plus": _consensus(holdout_sets, min_count=2) if holdout_sets else set(),
    }

    # Candidate-space restricted reference sizes for diagnostics.
    ref_rows: List[Dict[str, object]] = []
    for ref_name, ref_edges in reference_sets.items():
        ref_rows.append(
            {
                "reference_set": ref_name,
                "n_reference_edges_in_full_universe": int(len(ref_edges)),
                "n_reference_edges_in_candidate_pool": int(len(ref_edges & candidate_pool)),
            }
        )
    pd.DataFrame(ref_rows).to_csv(args.output_dir / "reference_set_summary.csv", index=False)

    rows: List[Dict[str, object]] = []
    for selection_label, selection_path in selections.items():
        sel_df = pd.read_csv(selection_path, low_memory=False)
        need = {"source", "target"}
        if not need.issubset(sel_df.columns):
            raise ValueError(f"Selection {selection_label} missing columns {sorted(need)}")
        sel_edges = _edge_set(sel_df)
        n_sel = len(sel_edges)
        outside = len(sel_edges - candidate_pool)

        for ref_name, ref_edges in reference_sets.items():
            hits = len(sel_edges & ref_edges)
            K = len(candidate_pool & ref_edges)
            expected = n_sel * K / float(N)
            p_ge = _hypergeom_sf(hits, N=N, K=K, n=n_sel)
            rows.append(
                {
                    "selection": selection_label,
                    "selection_file": str(selection_path),
                    "candidate_mode": args.candidate_mode,
                    "reference_set": ref_name,
                    "n_selected": int(n_sel),
                    "n_selected_outside_candidate_pool": int(outside),
                    "n_reference_edges_in_full_universe": int(len(ref_edges)),
                    "n_reference_edges_in_candidate_pool": int(K),
                    "n_hits": int(hits),
                    "precision_in_selection": float(hits / max(n_sel, 1)),
                    "recall_in_reference_universe": float(hits / max(len(ref_edges), 1)),
                    "expected_hits_random_from_candidate_pool": float(expected),
                    "enrichment_vs_random": float(hits / expected) if expected > 0 else float("nan"),
                    "p_value_hypergeom_ge_hits": float(p_ge),
                }
            )

    summary = pd.DataFrame(rows).sort_values(["reference_set", "selection"]).reset_index(drop=True)
    summary.to_csv(args.output_dir / "blind_holdout_validation_summary.csv", index=False)

    # Wide comparison with deltas versus baseline selection.
    metrics = ["n_hits", "precision_in_selection", "enrichment_vs_random", "p_value_hypergeom_ge_hits"]
    pivot = summary.pivot(index="reference_set", columns="selection", values=metrics)
    pivot.columns = [f"{m}__{s}" for m, s in pivot.columns]
    pivot = pivot.reset_index()

    if args.comparison_selection in set(summary["selection"]):
        base = summary[summary["selection"] == args.comparison_selection].set_index("reference_set")
        for selection in sorted(set(summary["selection"]) - {args.comparison_selection}):
            cur = summary[summary["selection"] == selection].set_index("reference_set")
            for metric in metrics:
                pivot[f"delta_{metric}__{selection}_minus_{args.comparison_selection}"] = (
                    cur[metric] - base[metric]
                ).reindex(pivot["reference_set"]).to_numpy()

    pivot.to_csv(args.output_dir / "blind_holdout_validation_comparison.csv", index=False)

    print("Blind holdout protocol complete.")
    print(f"Selection files: {len(selections)}")
    print(f"Candidate mode: {args.candidate_mode}; pool size: {N}")
    print(f"Train primary panels: {len(train_primary)}; holdout primary panels: {len(holdout_primary)}")
    print(f"Extra holdout panels: {len(extra_edges)}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
