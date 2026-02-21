#!/usr/bin/env python3
"""Build a context-matched edge universe with panel-balance constraints."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Sequence, Set, Tuple

import numpy as np
import pandas as pd


EdgeSet = Set[Tuple[str, str]]


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
        "--panel-glob",
        type=str,
        default="single_cell_mechinterp/outputs/perturb_validation/*/perturbation_edges.tsv",
    )
    parser.add_argument(
        "--include-kinds",
        type=str,
        default="reference,mechanistic",
        help="Kinds counted for coverage and observed-source filtering.",
    )
    parser.add_argument(
        "--min-observed-sources",
        type=int,
        default=2,
        help="Minimum observed annotators in include-kinds for retained edges.",
    )
    parser.add_argument(
        "--context-mode",
        type=str,
        default="strict",
        choices=["strict", "relaxed"],
        help="strict: source-in-family-source AND target-in-family-target; relaxed: source-in-family-source OR target-in-family-target.",
    )
    parser.add_argument(
        "--dixit-like-families",
        type=str,
        default="dixit,dixit7",
        help="Panel families treated as dominant context group for balancing.",
    )
    parser.add_argument(
        "--max-dixit-like-to-other",
        type=float,
        default=3.0,
        help="Max retained dixit-like-only edges per one retained other-context edge.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("subproject_24_multireference_latent_truth_model/implementation/outputs/context_matched_universe"),
    )
    return parser.parse_args()


def _parse_list(s: str) -> List[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def _panel_family(panel_name: str) -> str:
    name = panel_name.lower()
    if name.startswith("dixit7"):
        return "dixit7"
    if name.startswith("dixit"):
        return "dixit"
    if name.startswith("adamson"):
        return "adamson"
    if name.startswith("shifrut"):
        return "shifrut"
    return name.split("_", 1)[0]


def _edge_set(df: pd.DataFrame) -> EdgeSet:
    return set(df[["source", "target"]].drop_duplicates().itertuples(index=False, name=None))


def _load_panels(panel_glob: str) -> Tuple[Dict[str, EdgeSet], Dict[str, Set[str]], Dict[str, Set[str]]]:
    panel_edges: Dict[str, EdgeSet] = {}
    family_sources: Dict[str, Set[str]] = {}
    family_targets: Dict[str, Set[str]] = {}

    paths = sorted(Path(".").glob(panel_glob))
    if not paths:
        raise ValueError(f"No panels found for glob: {panel_glob}")

    for path in paths:
        panel_name = path.parent.name
        fam = _panel_family(panel_name)
        df = pd.read_csv(path, sep="\t")
        need = {"source", "target"}
        if not need.issubset(df.columns):
            raise ValueError(f"{path}: missing columns {sorted(need)}")
        dedup = df[["source", "target"]].drop_duplicates().copy()
        panel_edges[panel_name] = _edge_set(dedup)
        family_sources.setdefault(fam, set()).update(dedup["source"].astype(str).tolist())
        family_targets.setdefault(fam, set()).update(dedup["target"].astype(str).tolist())

    return panel_edges, family_sources, family_targets


def _recompute_source_meta(
    source_meta: pd.DataFrame,
    matrix: pd.DataFrame,
) -> pd.DataFrame:
    out = source_meta.copy()
    n_edges = len(matrix)
    source_cols = [c for c in out["source_id"].astype(str).tolist() if c in matrix.columns]
    metrics = []
    for source_id in source_cols:
        col = matrix[source_id]
        observed = int(col.notna().sum())
        positive = int(pd.to_numeric(col, errors="coerce").fillna(0.0).sum())
        metrics.append(
            {
                "source_id": source_id,
                "n_covered_edges_context": observed,
                "coverage_fraction_context": observed / float(max(n_edges, 1)),
                "n_positive_edges_context": positive,
                "positive_rate_within_coverage_context": positive / float(max(observed, 1)),
            }
        )
    metrics_df = pd.DataFrame(metrics)
    out = out.merge(metrics_df, on="source_id", how="left")
    return out


def main() -> None:
    args = _parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    matrix = pd.read_csv(args.matrix)
    source_meta = pd.read_csv(args.source_meta)
    include_kinds = set(_parse_list(args.include_kinds))
    dixit_like = set(_parse_list(args.dixit_like_families))
    rng = np.random.default_rng(args.seed)

    panel_edges, family_sources, family_targets = _load_panels(args.panel_glob)
    families = sorted(family_sources.keys())

    annotators = source_meta[source_meta["kind"].isin(include_kinds)]["source_id"].astype(str).tolist()
    annotators = [c for c in annotators if c in matrix.columns]
    if not annotators:
        raise ValueError("No annotator columns found for --include-kinds in matrix.")

    edge = matrix[["source", "target"]].copy()
    edge["edge_key"] = list(zip(edge["source"], edge["target"]))
    edge["n_observed_annotators"] = matrix[annotators].notna().sum(axis=1).astype(int)

    fam_cols: List[str] = []
    for fam in families:
        src_set = family_sources.get(fam, set())
        tgt_set = family_targets.get(fam, set())
        col = f"context_{fam}"
        if args.context_mode == "strict":
            edge[col] = edge["source"].isin(src_set) & edge["target"].isin(tgt_set)
        else:
            edge[col] = edge["source"].isin(src_set) | edge["target"].isin(tgt_set)
        fam_cols.append(col)

    edge["n_context_families"] = edge[fam_cols].sum(axis=1).astype(int)
    dixit_cols = [f"context_{fam}" for fam in families if fam in dixit_like]
    other_cols = [f"context_{fam}" for fam in families if fam not in dixit_like]
    edge["has_dixit_like_context"] = edge[dixit_cols].any(axis=1) if dixit_cols else False
    edge["has_other_context"] = edge[other_cols].any(axis=1) if other_cols else False
    edge["has_any_context"] = edge["n_context_families"] > 0

    candidate = edge["has_any_context"] & (edge["n_observed_annotators"] >= args.min_observed_sources)
    other_candidate = candidate & edge["has_other_context"]
    dixit_only_candidate = candidate & edge["has_dixit_like_context"] & (~edge["has_other_context"])

    other_idx = edge.index[other_candidate].to_numpy()
    dixit_idx = edge.index[dixit_only_candidate].to_numpy()
    n_other = int(len(other_idx))
    if args.max_dixit_like_to_other < 0:
        max_dixit_keep = len(dixit_idx)
    else:
        max_dixit_keep = int(np.floor(args.max_dixit_like_to_other * n_other))
    max_dixit_keep = min(max_dixit_keep, len(dixit_idx))
    if max_dixit_keep < len(dixit_idx):
        keep_dixit_idx = np.sort(rng.choice(dixit_idx, size=max_dixit_keep, replace=False))
    else:
        keep_dixit_idx = np.sort(dixit_idx)

    selected_idx = np.sort(np.concatenate([other_idx, keep_dixit_idx]))
    edge["selected"] = False
    edge.loc[selected_idx, "selected"] = True

    selected_matrix = matrix.loc[selected_idx].copy().reset_index(drop=True)
    selected_edge_set = _edge_set(selected_matrix[["source", "target"]])
    full_edge_set = _edge_set(matrix[["source", "target"]])

    overlap_rows = []
    for panel_name, p_edges in sorted(panel_edges.items()):
        overlap_rows.append(
            {
                "panel": panel_name,
                "family": _panel_family(panel_name),
                "panel_edges": int(len(p_edges)),
                "overlap_full_edges": int(len(p_edges & full_edge_set)),
                "overlap_selected_edges": int(len(p_edges & selected_edge_set)),
                "overlap_selected_fraction_panel": float(len(p_edges & selected_edge_set) / max(len(p_edges), 1)),
            }
        )
    overlap_df = pd.DataFrame(overlap_rows).sort_values("overlap_selected_edges", ascending=False).reset_index(drop=True)

    source_meta_context = _recompute_source_meta(source_meta=source_meta, matrix=selected_matrix)

    summary = pd.DataFrame(
        [
            {
                "n_edges_full": int(len(matrix)),
                "n_edges_candidates": int(candidate.sum()),
                "n_edges_other_candidates": int(other_candidate.sum()),
                "n_edges_dixit_only_candidates": int(dixit_only_candidate.sum()),
                "n_edges_selected": int(len(selected_matrix)),
                "selected_fraction_full": float(len(selected_matrix) / max(len(matrix), 1)),
                "min_observed_sources": int(args.min_observed_sources),
                "max_dixit_like_to_other": float(args.max_dixit_like_to_other),
                "seed": int(args.seed),
                "context_mode": args.context_mode,
                "dixit_like_families": ",".join(sorted(dixit_like)),
                "panel_families": ",".join(families),
            }
        ]
    )

    edge.to_csv(args.output_dir / "context_edge_diagnostics.csv", index=False)
    selected_matrix.to_csv(args.output_dir / "context_matched_evidence_matrix_wide.csv", index=False)
    source_meta_context.to_csv(args.output_dir / "context_matched_source_metadata.csv", index=False)
    overlap_df.to_csv(args.output_dir / "context_panel_overlap_summary.csv", index=False)
    summary.to_csv(args.output_dir / "context_build_summary.csv", index=False)

    print("Context-matched universe build complete:")
    print(summary.to_string(index=False))
    print("Top panel overlaps (selected):")
    print(overlap_df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
