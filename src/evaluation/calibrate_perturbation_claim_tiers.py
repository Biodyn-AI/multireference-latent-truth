#!/usr/bin/env python3
"""Calibrate posterior-based claim tiers using perturbation reference edges."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


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
        "--perturbation-edges",
        type=Path,
        default=Path(
            "single_cell_mechinterp/outputs/perturb_validation/dixit_causal_dixit_targets_tiny_neg/perturbation_edges.tsv"
        ),
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
        default=Path("subproject_24_multireference_latent_truth_model/implementation/outputs"),
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
        raise ValueError("No posterior files provided.")
    return out


def _compute_curve(df: pd.DataFrame, posterior_col: str = "posterior") -> pd.DataFrame:
    y = df["is_perturb"].to_numpy(dtype=int)
    p = df[posterior_col].to_numpy(dtype=float)

    quantiles = np.linspace(0.50, 0.9995, 180)
    thresholds = np.unique(np.quantile(p, quantiles))
    thresholds = np.sort(thresholds)[::-1]

    rows = []
    total_pos = int(np.sum(y))
    total_neg = int(len(y) - total_pos)

    for t in thresholds:
        pred = p >= t
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
                "threshold": float(t),
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

    curve = pd.DataFrame(rows).sort_values("threshold", ascending=False).reset_index(drop=True)
    return curve


def _pick_threshold(
    curve: pd.DataFrame,
    min_precision: float,
    min_predicted: int,
) -> float:
    c = curve[
        (curve["predicted"] >= min_predicted)
        & (curve["precision"] >= min_precision)
    ].copy()
    if c.empty:
        c = curve[curve["predicted"] >= min_predicted].copy()
        if c.empty:
            c = curve.copy()
    c = c.sort_values(["f1", "precision", "recall"], ascending=[False, False, False])
    return float(c.iloc[0]["threshold"])


def _assign_tier(p: float, t1: float, t2: float, t3: float) -> str:
    if p >= t1:
        return "PERT_TIER1_STRONG"
    if p >= t2:
        return "PERT_TIER2_MODERATE"
    if p >= t3:
        return "PERT_TIER3_EXPLORATORY"
    return "PERT_TIER4_BACKGROUND"


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


def main() -> None:
    args = _parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    post_map = _parse_map(args.posterior_files)
    if args.primary_model not in post_map:
        raise ValueError(f"--primary-model '{args.primary_model}' not found in --posterior-files map.")

    pert = pd.read_csv(args.perturbation_edges, sep="\t")[["source", "target"]].drop_duplicates()
    pert["is_perturb"] = 1

    curve_rows: List[pd.DataFrame] = []
    summary_rows: List[Dict[str, object]] = []
    threshold_rows: List[Dict[str, object]] = []

    primary_edge_df: pd.DataFrame | None = None
    primary_t1 = primary_t2 = primary_t3 = None

    for label, path in post_map.items():
        post = pd.read_csv(path)
        need = {"source", "target", "posterior"}
        if not need.issubset(post.columns):
            raise ValueError(f"{label}: missing columns {sorted(need)} in {path}")

        edge = post[["source", "target", "posterior"]].copy()
        if "n_observed_sources" in post.columns:
            edge["n_observed_sources"] = post["n_observed_sources"]
        else:
            edge["n_observed_sources"] = 0
        if "n_positive_sources" in post.columns:
            edge["n_positive_sources"] = post["n_positive_sources"]
        else:
            edge["n_positive_sources"] = 0

        edge = edge.merge(pert, on=["source", "target"], how="left")
        edge["is_perturb"] = edge["is_perturb"].fillna(0).astype(int)
        # Rank-friendly tie-break score for heavily discretized posterior outputs.
        edge["calibration_score"] = (
            edge["posterior"]
            + 1e-6 * edge["n_observed_sources"].astype(float)
            + 1e-9 * edge["n_positive_sources"].astype(float)
        )

        curve = _compute_curve(edge)
        curve["model"] = label
        curve_rows.append(curve)

        best = curve.sort_values(["f1", "precision", "recall"], ascending=[False, False, False]).iloc[0]
        t1 = _pick_threshold(curve, min_precision=0.50, min_predicted=args.min_pred_tier1)
        t2 = _pick_threshold(curve, min_precision=0.30, min_predicted=args.min_pred_tier2)
        t3 = _pick_threshold(curve, min_precision=0.15, min_predicted=args.min_pred_tier3)
        y = edge["is_perturb"].to_numpy(dtype=int)
        s = edge["calibration_score"].to_numpy(dtype=float)
        p500, r500, tp500 = _precision_recall_at_k(y, s, 500)
        p5k, r5k, tp5k = _precision_recall_at_k(y, s, 5000)
        p20k, r20k, tp20k = _precision_recall_at_k(y, s, 20000)

        threshold_rows.extend(
            [
                {"model": label, "tier": "PERT_TIER1_STRONG", "threshold": t1},
                {"model": label, "tier": "PERT_TIER2_MODERATE", "threshold": t2},
                {"model": label, "tier": "PERT_TIER3_EXPLORATORY", "threshold": t3},
            ]
        )

        summary_rows.append(
            {
                "model": label,
                "n_edges": int(len(edge)),
                "n_perturb_edges_in_universe": int(edge["is_perturb"].sum()),
                "best_threshold": float(best["threshold"]),
                "best_precision": float(best["precision"]),
                "best_recall": float(best["recall"]),
                "best_f1": float(best["f1"]),
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

        if label == args.primary_model:
            primary_edge_df = edge.copy()
            primary_t1, primary_t2, primary_t3 = t1, t2, t3

    if primary_edge_df is None:
        raise RuntimeError("Primary model data not loaded.")

    # Use rank-based tiers for primary model to avoid threshold collapse under tied posteriors.
    primary_edge_df = primary_edge_df.sort_values(
        ["calibration_score", "posterior", "n_observed_sources", "n_positive_sources"],
        ascending=[False, False, False, False],
        kind="mergesort",
    ).reset_index(drop=True)
    primary_edge_df["calibration_rank"] = np.arange(1, len(primary_edge_df) + 1, dtype=int)

    tier1_k = int(max(args.tier1_k, 1))
    tier2_k = int(max(args.tier2_k, tier1_k))
    tier3_k = int(max(args.tier3_k, tier2_k))

    primary_edge_df["perturb_claim_tier"] = np.where(
        primary_edge_df["calibration_rank"] <= tier1_k,
        "PERT_TIER1_STRONG",
        np.where(
            primary_edge_df["calibration_rank"] <= tier2_k,
            "PERT_TIER2_MODERATE",
            np.where(
                primary_edge_df["calibration_rank"] <= tier3_k,
                "PERT_TIER3_EXPLORATORY",
                "PERT_TIER4_BACKGROUND",
            ),
        ),
    )

    tier_summary = (
        primary_edge_df.groupby("perturb_claim_tier", as_index=False)
        .agg(
            n_edges=("source", "count"),
            n_perturb_edges=("is_perturb", "sum"),
            mean_posterior=("posterior", "mean"),
            mean_observed_sources=("n_observed_sources", "mean"),
        )
        .sort_values("perturb_claim_tier")
        .reset_index(drop=True)
    )
    tier_summary["perturb_precision_within_tier"] = (
        tier_summary["n_perturb_edges"] / np.maximum(tier_summary["n_edges"], 1)
    )

    curves = pd.concat(curve_rows, ignore_index=True)
    summary = pd.DataFrame(summary_rows).sort_values("best_f1", ascending=False).reset_index(drop=True)
    thresholds = pd.DataFrame(threshold_rows).sort_values(["model", "tier"]).reset_index(drop=True)

    curves.to_csv(args.output_dir / "perturbation_calibration_curve.csv", index=False)
    summary.to_csv(args.output_dir / "perturbation_calibration_summary.csv", index=False)
    thresholds.to_csv(args.output_dir / "perturbation_calibration_thresholds.csv", index=False)
    primary_edge_df.to_csv(args.output_dir / "perturbation_claim_tiers_edges.csv", index=False)
    tier_summary.to_csv(args.output_dir / "perturbation_claim_tiers_summary.csv", index=False)

    print("Perturbation calibration complete:")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
