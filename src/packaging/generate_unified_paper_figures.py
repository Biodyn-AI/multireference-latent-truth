#!/usr/bin/env python3
"""Generate manuscript-ready figures from unified paper tables."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tables-dir",
        type=Path,
        default=Path(
            "subproject_24_multireference_latent_truth_model/implementation/outputs/unified_paper_tables"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "subproject_24_multireference_latent_truth_model/implementation/outputs/unified_paper_figures"
        ),
    )
    return parser.parse_args()


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing required table: {path}")
    return pd.read_csv(path, low_memory=False)


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def build_robustness_figure(tables_dir: Path, output_dir: Path) -> Path:
    df = _read_csv(tables_dir / "robustness_progression_summary.csv")
    protocols = ["source_dropout_full_universe", "family_block_dropout_full_universe"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)
    for idx, protocol in enumerate(protocols):
        sub = df[df["stress_protocol"] == protocol].sort_values("mean_rank_disagreement", ascending=True)
        axes[idx].bar(sub["regime"], sub["mean_rank_disagreement"], color="#35618f")
        axes[idx].set_title(protocol.replace("_", " "))
        axes[idx].set_xlabel("Regime")
        axes[idx].tick_params(axis="x", rotation=20)
    axes[0].set_ylabel("Mean Rank Disagreement")

    out = output_dir / "figure_robustness_progression.png"
    _save(fig, out)
    return out


def build_calibration_figure(tables_dir: Path, output_dir: Path) -> Path:
    df = _read_csv(tables_dir / "calibration_full_vs_context.csv")
    metric_cols = ["precision_at_500", "precision_at_5000", "precision_at_20000", "best_f1"]
    metric_labels = ["P@500", "P@5000", "P@20000", "Best F1"]

    context = df[df["universe"] == "context_v2"].iloc[0]
    full = df[df["universe"] == "full_universe"].iloc[0]

    x = np.arange(len(metric_cols))
    w = 0.38

    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    ax.bar(x - w / 2, [context[c] for c in metric_cols], width=w, label="Context v2", color="#2a9d8f")
    ax.bar(x + w / 2, [full[c] for c in metric_cols], width=w, label="Full universe", color="#e76f51")
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels)
    ax.set_ylabel("Metric Value")
    ax.set_title("Calibration Metrics: Context vs Full Universe")
    ax.legend(frameon=False)

    out = output_dir / "figure_calibration_context_vs_full.png"
    _save(fig, out)
    return out


def build_governance_figure(tables_dir: Path, output_dir: Path) -> Path:
    df = _read_csv(tables_dir / "dual_universe_claim_tiers_summary.csv")

    tiers = df["dual_universe_claim_tier"].tolist()
    n_edges = df["n_edges"].to_numpy()
    promoted = df["n_promotion_allowed"].to_numpy()

    x = np.arange(len(tiers))
    w = 0.38

    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    ax.bar(x - w / 2, n_edges, width=w, label="Tier size", color="#264653")
    ax.bar(x + w / 2, promoted, width=w, label="Promotion allowed", color="#f4a261")
    ax.set_xticks(x)
    ax.set_xticklabels(tiers, rotation=18, ha="right")
    ax.set_ylabel("Number of edges")
    ax.set_title("Dual-Universe Governance: Tier Size vs Promotion")
    ax.legend(frameon=False)

    out = output_dir / "figure_dual_governance_tiers.png"
    _save(fig, out)
    return out


def build_source_concentration_figure(tables_dir: Path, output_dir: Path) -> Path:
    promoted = _read_csv(tables_dir / "dual_universe_promoted_top_sources.csv").head(10)
    conditional = _read_csv(tables_dir / "dual_universe_conditional_top_sources.csv").head(10)

    all_sources = sorted(set(promoted["source"]).union(set(conditional["source"])))
    p_map = dict(zip(promoted["source"], promoted["fraction_promoted_edges"]))
    c_map = dict(zip(conditional["source"], conditional["fraction_conditional_edges"]))

    p_vals = [p_map.get(s, 0.0) for s in all_sources]
    c_vals = [c_map.get(s, 0.0) for s in all_sources]

    x = np.arange(len(all_sources))
    w = 0.38

    fig, ax = plt.subplots(figsize=(11.8, 4.8))
    ax.bar(x - w / 2, p_vals, width=w, label="Promoted fraction", color="#8ab17d")
    ax.bar(x + w / 2, c_vals, width=w, label="Conditional fraction", color="#577590")
    ax.set_xticks(x)
    ax.set_xticklabels(all_sources, rotation=35, ha="right")
    ax.set_ylabel("Fraction of tier")
    ax.set_title("Top-Source Concentration in Promoted vs Conditional Sets")
    ax.legend(frameon=False)

    out = output_dir / "figure_source_concentration.png"
    _save(fig, out)
    return out


def main() -> None:
    args = _parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    outputs = [
        build_robustness_figure(args.tables_dir, args.output_dir),
        build_calibration_figure(args.tables_dir, args.output_dir),
        build_governance_figure(args.tables_dir, args.output_dir),
        build_source_concentration_figure(args.tables_dir, args.output_dir),
    ]

    manifest = pd.DataFrame({"figure_path": [str(p) for p in outputs]})
    manifest_path = args.output_dir / "figure_manifest.csv"
    manifest.to_csv(manifest_path, index=False)

    print("Unified paper figures built:")
    for path in outputs:
        print(f"- {path}")
    print(f"- {manifest_path}")


if __name__ == "__main__":
    main()
