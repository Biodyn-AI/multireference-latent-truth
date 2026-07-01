#!/usr/bin/env python3
"""Generate the two new manuscript figures for the revision."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
REV = ROOT / "outputs/revision"
FIGDIR = ROOT / "paper/figures"


def fig_universe_aware():
    depth = pd.read_csv(REV / "enrichment_vs_depth.csv")
    dec = pd.read_csv(REV / "universe_aware_decomposition.csv")
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.3))

    a = ax[0]
    a.axhline(1.0, color="grey", ls=":", lw=1)
    a.plot(depth["depth_fraction"], depth["enrichment_full"], "-o", ms=3,
           color="#b2182b", label="Full universe")
    a.plot(depth["depth_fraction"], depth["enrichment_context"], "-s", ms=3,
           color="#2166ac", label="Context-matched")
    a.set_xlabel("Rank depth (fraction of universe)")
    a.set_ylabel("Enrichment over random (lift)")
    a.set_title("(a) Denominator-invariant signal vs depth")
    a.legend(frameon=False, fontsize=9)
    a.grid(alpha=0.25)

    b = ax[1]
    row = dec[dec["k"] == 5000].iloc[0]
    comps = ["Raw P@5000\nratio", "Base-rate\n(denominator)", "Genuine\nsignal"]
    vals = [row["raw_ratio_context_over_full"], row["base_rate_ratio"], row["signal_ratio_context_over_full"]]
    colors = ["#7f7f7f", "#d6604d", "#4393c3"]
    bars = b.bar(comps, vals, color=colors, width=0.6)
    b.set_ylabel(r"Context/full ratio ($\times$)")
    b.set_title("(b) Decomposition of the apparent uplift")
    b.axhline(1.0, color="grey", ls=":", lw=1)
    for bar, v in zip(bars, vals):
        b.text(bar.get_x() + bar.get_width() / 2, v * 1.02, f"{v:.1f}$\\times$",
               ha="center", va="bottom", fontsize=10)
    b.set_ylim(0, max(vals) * 1.2)
    b.grid(alpha=0.25, axis="y")

    fig.tight_layout()
    fig.savefig(FIGDIR / "fig7_universe_aware.pdf", bbox_inches="tight")
    fig.savefig(FIGDIR / "fig7_universe_aware.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_deconcentration_frontier():
    grid = pd.read_csv(REV / "deconcentration_autotune_grid.csv")
    sel = pd.read_csv(REV / "deconcentration_autotune_selected.csv")
    fig, ax = plt.subplots(figsize=(6.6, 4.8))
    sc = ax.scatter(grid["pielou_evenness"], grid["encode_enrichment"],
                    c=grid["budget"], cmap="viridis", s=28, alpha=0.85, edgecolors="none")
    cb = fig.colorbar(sc); cb.set_label("Promotion budget")

    obj = sel[sel["rule"] == "objective_hmean"].iloc[0]
    base = sel[sel["rule"] == "fixed_842_baseline"].iloc[0]
    ax.scatter([obj["pielou_evenness"]], [obj["encode_enrichment"]], s=210, marker="*",
               color="#d62728", edgecolors="k", zorder=5, label="Auto-selected (b=200, $\\alpha$=0.05)")
    ax.scatter([base["pielou_evenness"]], [base["encode_enrichment"]], s=120, marker="D",
               color="#ff7f0e", edgecolors="k", zorder=5, label="Fixed budget 842")
    ax.set_xlabel("Source diversity (Pielou evenness)")
    ax.set_ylabel("External ENCODE enrichment ($\\times$)")
    ax.set_title("Diversity--precision frontier of promoted edges")
    ax.legend(frameon=False, fontsize=8.5, loc="lower left")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig8_deconcentration_frontier.pdf", bbox_inches="tight")
    fig.savefig(FIGDIR / "fig8_deconcentration_frontier.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    fig_universe_aware()
    fig_deconcentration_frontier()
    print("wrote fig7_universe_aware and fig8_deconcentration_frontier")
