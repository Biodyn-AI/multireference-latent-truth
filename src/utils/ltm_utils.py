#!/usr/bin/env python3
"""Utility helpers for latent-truth modeling on edge annotations.

This module centralizes:
- source catalog loading,
- edge table parsing across heterogeneous formats,
- edge symbol normalization,
- coverage-aware binary matrix construction,
- common metric helpers used across scripts.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


# Numerical clipping avoids log(0) and divide-by-zero issues in downstream estimators.
_EPS = 1e-9


@dataclass(frozen=True)
class SourceSpec:
    """Single evidence source specification loaded from the catalog."""

    source_id: str
    kind: str
    loader: str
    path: Path
    score_col: Optional[str]
    confidence: Optional[Tuple[str, ...]]


def load_source_catalog(catalog_path: Path, repo_root: Path) -> List[SourceSpec]:
    """Load source catalog JSON and resolve source paths against repository root."""

    with catalog_path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)

    specs: List[SourceSpec] = []
    for item in raw:
        specs.append(
            SourceSpec(
                source_id=item["id"],
                kind=item["kind"],
                loader=item["loader"],
                path=repo_root / item["path"],
                score_col=item.get("score_col"),
                confidence=tuple(item["confidence"]) if item.get("confidence") else None,
            )
        )

    return specs


def normalize_gene_symbol(value: Any) -> str:
    """Normalize a symbol to uppercase string with stripped whitespace.

    This is intentionally conservative and does not apply alias remapping,
    because this subproject focuses on a first-pass latent-truth model.
    """

    if pd.isna(value):
        return ""
    return str(value).strip().upper()


def _read_tsv(path: Path, header: Optional[int] = 0) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", header=header)


def _canonicalize_edges(df: pd.DataFrame, source_col: str, target_col: str) -> pd.DataFrame:
    out = pd.DataFrame(
        {
            "source": df[source_col].map(normalize_gene_symbol),
            "target": df[target_col].map(normalize_gene_symbol),
        }
    )
    out = out[(out["source"] != "") & (out["target"] != "")]
    out = out.drop_duplicates(ignore_index=True)
    return out


def load_source_edges(spec: SourceSpec) -> pd.DataFrame:
    """Load a source into canonical edge rows: source, target, score(optional)."""

    if not spec.path.exists():
        raise FileNotFoundError(f"Missing source file for {spec.source_id}: {spec.path}")

    loader = spec.loader

    if loader == "edge_tsv":
        raw = _read_tsv(spec.path)
        edges = _canonicalize_edges(raw, "source", "target")

    elif loader == "scored_edges":
        raw = _read_tsv(spec.path)
        edges = _canonicalize_edges(raw, "source", "target")
        if spec.score_col and spec.score_col in raw.columns:
            score_df = raw[["source", "target", spec.score_col]].copy()
            score_df["source"] = score_df["source"].map(normalize_gene_symbol)
            score_df["target"] = score_df["target"].map(normalize_gene_symbol)
            score_df = score_df[(score_df["source"] != "") & (score_df["target"] != "")]
            # Keep max score on duplicate edges to preserve strongest signal.
            score_df = (
                score_df.groupby(["source", "target"], as_index=False)[spec.score_col]
                .max()
                .rename(columns={spec.score_col: "score"})
            )
            edges = edges.merge(score_df, on=["source", "target"], how="left")

    elif loader == "trrust_no_header":
        # TRRUST export is headerless; first row is a valid edge, not column names.
        raw = _read_tsv(spec.path, header=None)
        if raw.shape[1] < 2:
            raise ValueError(f"Unexpected TRRUST format in {spec.path}")
        raw = raw.rename(columns={0: "source", 1: "target"})
        edges = _canonicalize_edges(raw, "source", "target")

    elif loader == "dorothea_confidence":
        raw = _read_tsv(spec.path)
        if "confidence" not in raw.columns:
            raise ValueError(f"DoRothEA loader requires confidence column: {spec.path}")
        if spec.confidence:
            raw = raw[raw["confidence"].astype(str).isin(spec.confidence)].copy()
        edges = _canonicalize_edges(raw, "source", "target")

    elif loader == "perturbation_edges":
        raw = _read_tsv(spec.path)
        edges = _canonicalize_edges(raw, "source", "target")
        if spec.score_col and spec.score_col in raw.columns:
            score_df = raw[["source", "target", spec.score_col]].copy()
            score_df["source"] = score_df["source"].map(normalize_gene_symbol)
            score_df["target"] = score_df["target"].map(normalize_gene_symbol)
            score_df = score_df[(score_df["source"] != "") & (score_df["target"] != "")]
            score_df = (
                score_df.groupby(["source", "target"], as_index=False)[spec.score_col]
                .max()
                .rename(columns={spec.score_col: "score"})
            )
            edges = edges.merge(score_df, on=["source", "target"], how="left")

    elif loader == "invariant_scores":
        raw = _read_tsv(spec.path)
        edges = _canonicalize_edges(raw, "source", "target")
        if spec.score_col and spec.score_col in raw.columns:
            score_df = raw[["source", "target", spec.score_col]].copy()
            score_df["source"] = score_df["source"].map(normalize_gene_symbol)
            score_df["target"] = score_df["target"].map(normalize_gene_symbol)
            score_df = (
                score_df.groupby(["source", "target"], as_index=False)[spec.score_col]
                .max()
                .rename(columns={spec.score_col: "score"})
            )
            edges = edges.merge(score_df, on=["source", "target"], how="left")

    else:
        raise ValueError(f"Unsupported loader '{loader}' for source {spec.source_id}")

    edges["source_id"] = spec.source_id
    edges["kind"] = spec.kind
    return edges


def build_edge_universe(source_tables: Sequence[pd.DataFrame]) -> pd.DataFrame:
    """Build the canonical edge universe from union of all source tables."""

    universe = pd.concat([t[["source", "target"]] for t in source_tables], ignore_index=True)
    universe = universe.drop_duplicates(ignore_index=True)
    universe = universe.sort_values(["source", "target"]).reset_index(drop=True)
    return universe


def build_coverage_aware_matrix(
    universe: pd.DataFrame,
    source_tables: Dict[str, pd.DataFrame],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Construct binary and score matrices with coverage-aware missingness.

    Rules per source and edge:
    - 1 if edge is explicitly present in source edge list,
    - 0 if both genes are in source node coverage but edge is absent,
    - NaN if at least one gene is outside source node coverage.

    This avoids treating out-of-coverage edges as explicit negatives.
    """

    binary = universe.copy()
    scores = universe.copy()
    summary_rows: List[Dict[str, Any]] = []

    for source_id, table in source_tables.items():
        node_set = set(table["source"].tolist()) | set(table["target"].tolist())

        in_src = universe["source"].isin(node_set)
        in_tgt = universe["target"].isin(node_set)
        covered = in_src & in_tgt

        membership_df = table[["source", "target"]].drop_duplicates().assign(_m=1)
        membership = (
            universe.merge(membership_df, on=["source", "target"], how="left")["_m"]
            .fillna(0)
            .to_numpy(dtype=np.int8)
        )
        col = np.full(len(universe), np.nan)
        col[covered.to_numpy()] = 0.0
        col[membership == 1] = 1.0
        binary[source_id] = col

        score_col = np.full(len(universe), np.nan)
        if "score" in table.columns:
            score_df = (
                table.dropna(subset=["score"])[["source", "target", "score"]]
                .groupby(["source", "target"], as_index=False)["score"]
                .max()
            )
            score_col = universe.merge(score_df, on=["source", "target"], how="left")[
                "score"
            ].to_numpy()
        scores[source_id] = score_col

        positives = int(np.nansum(binary[source_id].to_numpy() == 1.0))
        covered_edges = int(np.sum(~np.isnan(binary[source_id].to_numpy())))
        summary_rows.append(
            {
                "source_id": source_id,
                "n_source_edges": int(len(table)),
                "n_source_nodes": int(len(node_set)),
                "n_positive_edges": positives,
                "n_covered_edges": covered_edges,
                "coverage_fraction": covered_edges / float(len(universe)),
                "positive_rate_within_coverage": positives / float(max(covered_edges, 1)),
                "has_scores": int("score" in table.columns),
            }
        )

    summary = pd.DataFrame(summary_rows).sort_values("source_id").reset_index(drop=True)
    return binary, scores, summary


def parse_kind_list(kind_csv: str) -> List[str]:
    return [x.strip() for x in kind_csv.split(",") if x.strip()]


def safe_log(x: np.ndarray) -> np.ndarray:
    return np.log(np.clip(x, _EPS, 1.0 - _EPS))


def weighted_precision_recall_f1(
    y_pred: np.ndarray,
    y_prob_true: np.ndarray,
) -> Dict[str, float]:
    """Compute precision/recall/F1 where labels are posterior probabilities."""

    y_pred = y_pred.astype(float)
    y_prob_true = y_prob_true.astype(float)

    tp = float(np.sum(y_pred * y_prob_true))
    fp = float(np.sum(y_pred * (1.0 - y_prob_true)))
    fn = float(np.sum((1.0 - y_pred) * y_prob_true))

    precision = tp / max(tp + fp, _EPS)
    recall = tp / max(tp + fn, _EPS)
    f1 = 2.0 * precision * recall / max(precision + recall, _EPS)

    return {
        "weighted_precision": precision,
        "weighted_recall": recall,
        "weighted_f1": f1,
        "weighted_tp": tp,
        "weighted_fp": fp,
        "weighted_fn": fn,
    }


def binary_precision_recall_f1(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Standard binary metrics on fully observed arrays."""

    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, _EPS)

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
    }
