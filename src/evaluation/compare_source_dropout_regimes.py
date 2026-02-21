#!/usr/bin/env python3
"""Compare source-dropout behavior across latent-truth model regimes."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--inputs",
        type=str,
        default=(
            "dep=subproject_24_multireference_latent_truth_model/implementation/outputs/source_dropout_ablation_full/source_dropout_summary.csv,"
            "mf=subproject_24_multireference_latent_truth_model/implementation/outputs/source_dropout_ablation_full_mf/source_dropout_summary.csv,"
            "family=subproject_24_multireference_latent_truth_model/implementation/outputs/source_dropout_ablation_full_family/source_dropout_summary.csv"
        ),
        help="Comma-separated label=csv_path entries.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("subproject_24_multireference_latent_truth_model/implementation/outputs"),
    )
    parser.add_argument(
        "--id-col",
        type=str,
        default="dropped_source",
        help="Identifier column shared across regime summaries (e.g. dropped_source or dropped_family).",
    )
    parser.add_argument(
        "--baseline-id",
        type=str,
        default="NONE_BASELINE",
        help="Value in --id-col used for baseline row exclusion.",
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default="source_dropout",
        help="Prefix for output artifact names.",
    )
    return parser.parse_args()


def _parse_inputs(spec: str) -> Dict[str, Path]:
    out: Dict[str, Path] = {}
    for part in [x.strip() for x in spec.split(",") if x.strip()]:
        if "=" not in part:
            raise ValueError(f"Invalid input spec entry: {part}")
        label, path = part.split("=", 1)
        out[label.strip()] = Path(path.strip())
    if len(out) < 2:
        raise ValueError("Provide at least 2 labeled inputs.")
    return out


def _load_regime(path: Path, label: str, id_col: str, baseline_id: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if id_col not in df.columns:
        raise ValueError(f"{label}: missing id column '{id_col}' in {path}")
    df = df[df[id_col].astype(str) != baseline_id].copy()
    keep = [
        id_col,
        "pairwise_rank_disagreement_vs_baseline",
        "n_disagree_pairs",
        "n_valid_pairs",
        "delta_pi_vs_baseline",
        "delta_rho_vs_baseline",
        "top_method",
        "top_method_weighted_f1",
    ]
    miss = [c for c in keep if c not in df.columns]
    if miss:
        raise ValueError(f"{label}: missing columns {miss} in {path}")
    df = df[keep].rename(
        columns={
            "pairwise_rank_disagreement_vs_baseline": f"{label}_rank_disagreement",
            "n_disagree_pairs": f"{label}_n_disagree_pairs",
            "n_valid_pairs": f"{label}_n_valid_pairs",
            "delta_pi_vs_baseline": f"{label}_delta_pi",
            "delta_rho_vs_baseline": f"{label}_delta_rho",
            "top_method": f"{label}_top_method",
            "top_method_weighted_f1": f"{label}_top_method_weighted_f1",
        }
    )
    return df


def _regime_aggregate(label: str, merged: pd.DataFrame) -> Dict[str, float]:
    dis = merged[f"{label}_rank_disagreement"].to_numpy(dtype=float)
    dpi = merged[f"{label}_delta_pi"].to_numpy(dtype=float)
    drho = merged[f"{label}_delta_rho"].to_numpy(dtype=float)
    top = merged[f"{label}_top_method"].astype(str)

    baseline_top = "attention_inferred"
    return {
        "regime": label,
        "n_sources": float(len(merged)),
        "mean_rank_disagreement": float(np.nanmean(dis)),
        "max_rank_disagreement": float(np.nanmax(dis)),
        "n_nonzero_rank_disagreement": float(np.sum(dis > 0)),
        "mean_abs_delta_pi": float(np.nanmean(np.abs(dpi))),
        "max_abs_delta_pi": float(np.nanmax(np.abs(dpi))),
        "mean_abs_delta_rho": float(np.nanmean(np.abs(drho))),
        "max_abs_delta_rho": float(np.nanmax(np.abs(drho))),
        "n_top_method_switches": float(np.sum(top != baseline_top)),
    }


def main() -> None:
    args = _parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    input_map = _parse_inputs(args.inputs)
    labels = list(input_map.keys())

    merged: pd.DataFrame | None = None
    for label in labels:
        df = _load_regime(
            path=input_map[label],
            label=label,
            id_col=args.id_col,
            baseline_id=args.baseline_id,
        )
        if merged is None:
            merged = df
        else:
            merged = merged.merge(df, on=args.id_col, how="inner")

    if merged is None:
        raise ValueError("No input data loaded.")

    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            a, b = labels[i], labels[j]
            merged[f"delta_rank_disagreement_{a}_minus_{b}"] = (
                merged[f"{a}_rank_disagreement"] - merged[f"{b}_rank_disagreement"]
            )
            merged[f"delta_abs_pi_{a}_minus_{b}"] = (
                np.abs(merged[f"{a}_delta_pi"]) - np.abs(merged[f"{b}_delta_pi"])
            )
            merged[f"delta_abs_rho_{a}_minus_{b}"] = (
                np.abs(merged[f"{a}_delta_rho"]) - np.abs(merged[f"{b}_delta_rho"])
            )

    merged = merged.sort_values(f"{labels[0]}_rank_disagreement", ascending=False).reset_index(drop=True)

    agg_rows: List[Dict[str, float]] = []
    for label in labels:
        agg_rows.append(_regime_aggregate(label=label, merged=merged))
    aggregate = pd.DataFrame(agg_rows).sort_values("mean_rank_disagreement").reset_index(drop=True)

    merged.to_csv(args.output_dir / f"{args.output_prefix}_regime_comparison_3way.csv", index=False)
    aggregate.to_csv(args.output_dir / f"{args.output_prefix}_regime_aggregate_3way.csv", index=False)

    print(f"{len(labels)}-way source-dropout regime comparison complete:")
    print(aggregate.to_string(index=False))


if __name__ == "__main__":
    main()
