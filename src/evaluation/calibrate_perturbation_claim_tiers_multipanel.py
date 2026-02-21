#!/usr/bin/env python3
"""Calibrate posterior claim tiers against multiple perturbation panels."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd


EdgeSet = set[Tuple[str, str]]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--posterior-files",
        type=str,
        default=(
            "dep=subproject_24_multireference_latent_truth_model/implementation/outputs/ltm_dep_edge_posteriors.csv,"
            "mf=subproject_24_multireference_latent_truth_model/implementation/outputs/ltm_mf_edge_posteriors.csv,"
            "family_hb=subproject_24_multireference_latent_truth_model/implementation/outputs/ltm_family_hb_edge_posteriors.csv"
        ),
        help="Comma-separated label=posterior_csv.",
    )
    parser.add_argument(
        "--primary-model",
        type=str,
        default="family_hb",
        help="Model label used for final edge-tier assignment output.",
    )
    parser.add_argument(
        "--panel-glob",
        type=str,
        default="single_cell_mechinterp/outputs/perturb_validation/*/perturbation_edges.tsv",
        help="Glob used to discover perturbation panel files.",
    )
    parser.add_argument(
        "--panel-map",
        type=str,
        default="",
        help="Optional comma-separated label=path overrides. When set, this is used instead of --panel-glob.",
    )
    parser.add_argument(
        "--min-panel-overlap",
        type=int,
        default=50,
        help="Minimum overlap edges with the primary model universe to count as a high-overlap panel.",
    )
    parser.add_argument(
        "--primary-reference-set",
        type=str,
        default="high_overlap_union",
        help="Reference-set label used for primary-model tier assignment.",
    )
    parser.add_argument(
        "--min-pred-tier1",
        type=int,
        default=50,
    )
    parser.add_argument(
        "--min-pred-tier2",
        type=int,
        default=200,
    )
    parser.add_argument(
        "--min-pred-tier3",
        type=int,
        default=500,
    )
    parser.add_argument("--tier1-k", type=int, default=500)
    parser.add_argument("--tier2-k", type=int, default=5000)
    parser.add_argument("--tier3-k", type=int, default=20000)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("subproject_24_multireference_latent_truth_model/implementation/outputs/perturbation_multipanel"),
    )
    return parser.parse_args()


def _parse_map(spec: str) -> Dict[str, Path]:
    out: Dict[str, Path] = {}
    for part in [x.strip() for x in spec.split(",") if x.strip()]:
        if "=" not in part:
            raise ValueError(f"Invalid map entry: {part}")
        key, value = part.split("=", 1)
        out[key.strip()] = Path(value.strip())
    return out


def _discover_panel_map(args: argparse.Namespace) -> Dict[str, Path]:
    if args.panel_map:
        panel_map = _parse_map(args.panel_map)
    else:
        panel_paths = sorted(Path(".").glob(args.panel_glob))
        panel_map = {p.parent.name: p for p in panel_paths}
    if not panel_map:
        raise ValueError("No perturbation panels found. Check --panel-glob or --panel-map.")
    return panel_map


def _edge_set(df: pd.DataFrame) -> EdgeSet:
    if "edge_key" in df.columns:
        return set(df["edge_key"].tolist())
    return set(df[["source", "target"]].drop_duplicates().itertuples(index=False, name=None))


def _load_panel_edges(panel_map: Dict[str, Path]) -> Dict[str, EdgeSet]:
    out: Dict[str, EdgeSet] = {}
    for label, path in panel_map.items():
        panel_df = pd.read_csv(path, sep="\t")
        required = {"source", "target"}
        if not required.issubset(panel_df.columns):
            raise ValueError(f"{label}: missing columns {sorted(required)} in {path}")
        out[label] = _edge_set(panel_df)
    return out


def _build_reference_sets(
    panel_edges: Dict[str, EdgeSet],
    panel_overlap_summary: pd.DataFrame,
    min_panel_overlap: int,
) -> Dict[str, EdgeSet]:
    # Per-panel references are useful for heterogeneity diagnostics.
    reference_sets: Dict[str, EdgeSet] = {f"panel::{k}": v for k, v in panel_edges.items()}

    all_union: EdgeSet = set()
    for edges in panel_edges.values():
        all_union |= edges
    reference_sets["all_union"] = all_union

    high_overlap_labels = panel_overlap_summary.loc[
        panel_overlap_summary["overlap_edges"] >= min_panel_overlap, "panel"
    ].tolist()
    if not high_overlap_labels:
        high_overlap_labels = panel_overlap_summary["panel"].tolist()

    high_overlap_union: EdgeSet = set()
    for label in high_overlap_labels:
        high_overlap_union |= panel_edges[label]
    reference_sets["high_overlap_union"] = high_overlap_union

    # Consensus sets identify edges repeatedly observed across panels.
    all_counts: Dict[Tuple[str, str], int] = {}
    for edges in panel_edges.values():
        for edge in edges:
            all_counts[edge] = all_counts.get(edge, 0) + 1
    reference_sets["consensus_2plus_all"] = {e for e, c in all_counts.items() if c >= 2}

    high_counts: Dict[Tuple[str, str], int] = {}
    for label in high_overlap_labels:
        for edge in panel_edges[label]:
            high_counts[edge] = high_counts.get(edge, 0) + 1
    reference_sets["consensus_2plus_high_overlap"] = {e for e, c in high_counts.items() if c >= 2}

    return reference_sets


def _compute_curve(df: pd.DataFrame, score_col: str = "calibration_score") -> pd.DataFrame:
    y = df["is_reference"].to_numpy(dtype=int)
    p = df[score_col].to_numpy(dtype=float)

    quantiles = np.linspace(0.50, 0.9995, 180)
    thresholds = np.unique(np.quantile(p, quantiles))
    thresholds = np.sort(thresholds)[::-1]

    rows: List[Dict[str, float | int]] = []
    total_pos = int(np.sum(y))
    total_neg = int(len(y) - total_pos)

    for threshold in thresholds:
        pred = p >= threshold
        tp = int(np.sum(pred & (y == 1)))
        fp = int(np.sum(pred & (y == 0)))
        fn = int(np.sum((~pred) & (y == 1)))
        tn = int(np.sum((~pred) & (y == 0)))

        precision = tp / float(tp + fp) if (tp + fp) > 0 else float("nan")
        recall = tp / float(tp + fn) if (tp + fn) > 0 else float("nan")
        f1 = (
            2.0 * precision * recall / (precision + recall)
            if np.isfinite(precision) and np.isfinite(recall) and (precision + recall) > 0
            else float("nan")
        )

        rows.append(
            {
                "threshold": float(threshold),
                "tp": tp,
                "fp": fp,
                "tn": tn,
                "fn": fn,
                "predicted": int(tp + fp),
                "total_pos": total_pos,
                "total_neg": total_neg,
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
            }
        )

    return pd.DataFrame(rows).sort_values("threshold", ascending=False).reset_index(drop=True)


def _pick_threshold(curve: pd.DataFrame, min_precision: float, min_predicted: int) -> float:
    candidates = curve[(curve["predicted"] >= min_predicted) & (curve["precision"] >= min_precision)].copy()
    if candidates.empty:
        candidates = curve[curve["predicted"] >= min_predicted].copy()
    if candidates.empty:
        candidates = curve.copy()

    # When labels are extremely sparse, many rows have NaN f1/recall; fallback to precision/predicted.
    candidates["f1_fill"] = candidates["f1"].fillna(-1.0)
    candidates["precision_fill"] = candidates["precision"].fillna(-1.0)
    candidates["recall_fill"] = candidates["recall"].fillna(-1.0)
    candidates = candidates.sort_values(
        ["f1_fill", "precision_fill", "recall_fill", "predicted"],
        ascending=[False, False, False, False],
    )
    return float(candidates.iloc[0]["threshold"])


def _precision_recall_at_k(y: np.ndarray, score: np.ndarray, k: int) -> Tuple[float, float, int]:
    if k <= 0:
        return float("nan"), float("nan"), 0
    k = min(k, len(y))
    order = np.argsort(-score, kind="mergesort")
    top = order[:k]
    tp = int(np.sum(y[top] == 1))
    precision = tp / float(k) if k > 0 else float("nan")
    recall = tp / float(np.sum(y == 1)) if np.sum(y == 1) > 0 else float("nan")
    return float(precision), float(recall), tp


def _tier_labels_for_ranks(ranks: np.ndarray, tier1_k: int, tier2_k: int, tier3_k: int) -> np.ndarray:
    return np.where(
        ranks <= tier1_k,
        "PERT_TIER1_STRONG",
        np.where(
            ranks <= tier2_k,
            "PERT_TIER2_MODERATE",
            np.where(ranks <= tier3_k, "PERT_TIER3_EXPLORATORY", "PERT_TIER4_BACKGROUND"),
        ),
    )


def _reference_set_summary(reference_sets: Dict[str, EdgeSet], universe: EdgeSet) -> pd.DataFrame:
    rows: List[Dict[str, float | int | str]] = []
    universe_n = max(len(universe), 1)
    for label, edges in reference_sets.items():
        overlap = len(edges & universe)
        rows.append(
            {
                "reference_set": label,
                "n_edges_total": int(len(edges)),
                "n_edges_in_primary_universe": int(overlap),
                "fraction_primary_universe": float(overlap / universe_n),
            }
        )
    return pd.DataFrame(rows).sort_values("n_edges_in_primary_universe", ascending=False).reset_index(drop=True)


def _panel_overlap_summary(panel_edges: Dict[str, EdgeSet], universe: EdgeSet) -> pd.DataFrame:
    rows: List[Dict[str, float | int | str]] = []
    universe_n = max(len(universe), 1)
    for panel, edges in panel_edges.items():
        overlap = len(edges & universe)
        rows.append(
            {
                "panel": panel,
                "panel_edges": int(len(edges)),
                "overlap_edges": int(overlap),
                "overlap_fraction_panel": float(overlap / max(len(edges), 1)),
                "overlap_fraction_universe": float(overlap / universe_n),
            }
        )
    return pd.DataFrame(rows).sort_values("overlap_edges", ascending=False).reset_index(drop=True)


def _parse_posteriors_map(spec: str) -> Dict[str, Path]:
    post_map = _parse_map(spec)
    if not post_map:
        raise ValueError("No posterior files provided.")
    return post_map


def _load_posteriors(post_map: Dict[str, Path]) -> Dict[str, pd.DataFrame]:
    posteriors: Dict[str, pd.DataFrame] = {}
    required = {"source", "target", "posterior"}
    for label, path in post_map.items():
        post = pd.read_csv(path)
        if not required.issubset(post.columns):
            raise ValueError(f"{label}: missing columns {sorted(required)} in {path}")
        out = post[["source", "target", "posterior"]].copy()
        out["n_observed_sources"] = post["n_observed_sources"] if "n_observed_sources" in post.columns else 0
        out["n_positive_sources"] = post["n_positive_sources"] if "n_positive_sources" in post.columns else 0
        out["edge_key"] = list(zip(out["source"], out["target"]))
        # Tie-break score to stabilize ranking when posterior bins are heavily discretized.
        out["calibration_score"] = (
            out["posterior"]
            + 1e-6 * out["n_observed_sources"].astype(float)
            + 1e-9 * out["n_positive_sources"].astype(float)
        )
        posteriors[label] = out
    return posteriors


def _validate_primary_reference(primary_reference_set: str, available: Iterable[str]) -> None:
    labels = set(available)
    if primary_reference_set not in labels:
        raise ValueError(
            f"--primary-reference-set '{primary_reference_set}' is unavailable. "
            f"Choices: {sorted(labels)}"
        )


def main() -> None:
    args = _parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    post_map = _parse_posteriors_map(args.posterior_files)
    if args.primary_model not in post_map:
        raise ValueError(f"--primary-model '{args.primary_model}' not found in --posterior-files map.")

    posteriors = _load_posteriors(post_map)
    primary_universe = _edge_set(posteriors[args.primary_model][["source", "target"]])

    panel_map = _discover_panel_map(args)
    panel_edges = _load_panel_edges(panel_map)
    panel_overlap_summary = _panel_overlap_summary(panel_edges, primary_universe)
    reference_sets = _build_reference_sets(panel_edges, panel_overlap_summary, args.min_panel_overlap)
    _validate_primary_reference(args.primary_reference_set, reference_sets.keys())

    reference_summary = _reference_set_summary(reference_sets, primary_universe)

    curve_rows: List[pd.DataFrame] = []
    threshold_rows: List[Dict[str, object]] = []
    summary_rows: List[Dict[str, object]] = []
    primary_edges_for_tiers: pd.DataFrame | None = None

    for model_label, edge in posteriors.items():
        for reference_label, ref_edges in reference_sets.items():
            model_df = edge.copy()
            model_df["is_reference"] = model_df["edge_key"].isin(ref_edges).astype(int)

            curve = _compute_curve(model_df, score_col="calibration_score")
            curve["model"] = model_label
            curve["reference_set"] = reference_label
            curve_rows.append(curve)

            best = curve.copy()
            best["f1_fill"] = best["f1"].fillna(-1.0)
            best["precision_fill"] = best["precision"].fillna(-1.0)
            best["recall_fill"] = best["recall"].fillna(-1.0)
            best = best.sort_values(
                ["f1_fill", "precision_fill", "recall_fill", "predicted"],
                ascending=[False, False, False, False],
            ).iloc[0]

            t1 = _pick_threshold(curve, min_precision=0.50, min_predicted=args.min_pred_tier1)
            t2 = _pick_threshold(curve, min_precision=0.30, min_predicted=args.min_pred_tier2)
            t3 = _pick_threshold(curve, min_precision=0.15, min_predicted=args.min_pred_tier3)

            y = model_df["is_reference"].to_numpy(dtype=int)
            s = model_df["calibration_score"].to_numpy(dtype=float)
            p500, r500, tp500 = _precision_recall_at_k(y, s, 500)
            p5k, r5k, tp5k = _precision_recall_at_k(y, s, 5000)
            p20k, r20k, tp20k = _precision_recall_at_k(y, s, 20000)

            threshold_rows.extend(
                [
                    {"model": model_label, "reference_set": reference_label, "tier": "PERT_TIER1_STRONG", "threshold": t1},
                    {"model": model_label, "reference_set": reference_label, "tier": "PERT_TIER2_MODERATE", "threshold": t2},
                    {"model": model_label, "reference_set": reference_label, "tier": "PERT_TIER3_EXPLORATORY", "threshold": t3},
                ]
            )
            summary_rows.append(
                {
                    "model": model_label,
                    "reference_set": reference_label,
                    "n_edges": int(len(model_df)),
                    "n_reference_edges_in_universe": int(model_df["is_reference"].sum()),
                    "best_threshold": float(best["threshold"]),
                    "best_precision": float(best["precision"]) if np.isfinite(best["precision"]) else float("nan"),
                    "best_recall": float(best["recall"]) if np.isfinite(best["recall"]) else float("nan"),
                    "best_f1": float(best["f1"]) if np.isfinite(best["f1"]) else float("nan"),
                    "tier1_threshold": float(t1),
                    "tier2_threshold": float(t2),
                    "tier3_threshold": float(t3),
                    "precision_at_500": p500,
                    "recall_at_500": r500,
                    "tp_at_500": tp500,
                    "precision_at_5000": p5k,
                    "recall_at_5000": r5k,
                    "tp_at_5000": tp5k,
                    "precision_at_20000": p20k,
                    "recall_at_20000": r20k,
                    "tp_at_20000": tp20k,
                }
            )

            if model_label == args.primary_model and reference_label == args.primary_reference_set:
                primary_edges_for_tiers = model_df.copy()

    if primary_edges_for_tiers is None:
        raise RuntimeError("Primary model/reference combination was not evaluated.")

    primary_edges_for_tiers = primary_edges_for_tiers.sort_values(
        ["calibration_score", "posterior", "n_observed_sources", "n_positive_sources"],
        ascending=[False, False, False, False],
        kind="mergesort",
    ).reset_index(drop=True)
    primary_edges_for_tiers["calibration_rank"] = np.arange(1, len(primary_edges_for_tiers) + 1, dtype=int)
    tier1_k = int(max(args.tier1_k, 1))
    tier2_k = int(max(args.tier2_k, tier1_k))
    tier3_k = int(max(args.tier3_k, tier2_k))
    primary_edges_for_tiers["perturb_claim_tier"] = _tier_labels_for_ranks(
        primary_edges_for_tiers["calibration_rank"].to_numpy(dtype=int),
        tier1_k=tier1_k,
        tier2_k=tier2_k,
        tier3_k=tier3_k,
    )
    primary_edges_for_tiers["primary_reference_set"] = args.primary_reference_set

    tier_summary = (
        primary_edges_for_tiers.groupby("perturb_claim_tier", as_index=False)
        .agg(
            n_edges=("source", "count"),
            n_reference_edges=("is_reference", "sum"),
            mean_posterior=("posterior", "mean"),
            mean_observed_sources=("n_observed_sources", "mean"),
        )
        .sort_values("perturb_claim_tier")
        .reset_index(drop=True)
    )
    tier_summary["reference_precision_within_tier"] = (
        tier_summary["n_reference_edges"] / np.maximum(tier_summary["n_edges"], 1)
    )
    tier_summary["primary_reference_set"] = args.primary_reference_set

    curves = pd.concat(curve_rows, ignore_index=True)
    summary = pd.DataFrame(summary_rows).sort_values(
        ["reference_set", "best_f1", "precision_at_500"],
        ascending=[True, False, False],
    ).reset_index(drop=True)
    thresholds = pd.DataFrame(threshold_rows).sort_values(
        ["reference_set", "model", "tier"],
    ).reset_index(drop=True)

    panel_overlap_summary.to_csv(args.output_dir / "perturbation_multipanel_panel_overlap_summary.csv", index=False)
    reference_summary.to_csv(args.output_dir / "perturbation_multipanel_reference_set_summary.csv", index=False)
    curves.to_csv(args.output_dir / "perturbation_multipanel_calibration_curve.csv", index=False)
    summary.to_csv(args.output_dir / "perturbation_multipanel_calibration_summary.csv", index=False)
    thresholds.to_csv(args.output_dir / "perturbation_multipanel_calibration_thresholds.csv", index=False)
    primary_edges_for_tiers.to_csv(args.output_dir / "perturbation_multipanel_claim_tiers_edges.csv", index=False)
    tier_summary.to_csv(args.output_dir / "perturbation_multipanel_claim_tiers_summary.csv", index=False)

    print("Multipanel perturbation calibration complete.")
    print(f"Panels: {len(panel_edges)} | Reference sets: {len(reference_sets)}")
    print(panel_overlap_summary.to_string(index=False))


if __name__ == "__main__":
    main()
