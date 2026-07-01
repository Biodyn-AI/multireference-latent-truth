#!/usr/bin/env python3
"""Distinct validation profile of the Strong vs Moderate promoted tiers.

Combines internal metrics (size, diversity, posterior, perturbation-reference
precision) with the genuinely-external ENCODE/ENETS2 enrichment computed in
external_encode_validation.py, so downstream users can see the risk/return profile
of each tier rather than only the merged 842-edge pool.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
ANN = ROOT / "outputs/revision/governance_edges_annotated.csv"
EXTV = ROOT / "outputs/revision/external_encode_enrichment.csv"
OUTDIR = ROOT / "outputs/revision"

TIERS = {"DUAL_TIER1_STRONG": "Strong", "DUAL_TIER2_MODERATE": "Moderate"}


def main() -> None:
    ann = pd.read_csv(ANN, low_memory=False)
    ext = pd.read_csv(EXTV)
    rows = []
    for key, label in TIERS.items():
        g = ann[ann["dual_universe_claim_tier"] == key]
        vc = g["source"].value_counts()
        row = {
            "tier": label,
            "n_edges": len(g),
            "n_unique_tfs": int(g["source"].nunique()),
            "n_unique_targets": int(g["target"].nunique()),
            "top1_tf_frac": vc.iloc[0] / len(g),
            "top2_tf_frac": vc.iloc[:2].sum() / len(g),
            "mean_posterior_full": g["posterior_full"].mean(),
            "mean_posterior_context": g["posterior_context"].mean(),
            "pert_ref_precision_full": g["is_reference_full"].mean(),
            "pert_ref_precision_context": g["is_reference_context"].mean(),
        }
        for setname in ["encode_tfbs", "enets2_seq"]:
            e = ext[(ext["external_set"] == setname) & (ext["tier"] == label)]
            if len(e):
                row[f"{setname}_enrichment"] = float(e["enrichment_vs_base"].iloc[0])
                row[f"{setname}_p"] = float(e["hypergeom_p"].iloc[0])
        rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(OUTDIR / "strong_vs_moderate_breakdown.csv", index=False)
    with pd.option_context("display.width", 220, "display.max_columns", 30):
        print(out.T.to_string())


if __name__ == "__main__":
    main()
