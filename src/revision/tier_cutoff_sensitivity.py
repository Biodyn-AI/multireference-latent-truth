#!/usr/bin/env python3
"""Sensitivity of the promoted-edge pool to +/-10% perturbation-tier rank cutoffs.

Per-universe perturbation tiers are rank-based: an edge is STRONG if its
calibration rank <= k1, MODERATE if <= k2, EXPLORATORY if <= k3, else BACKGROUND,
with default (k1,k2,k3) = (500, 5000, 20000). These are applied independently in
the full and context universes; the dual-universe tier is their cross-universe
combination. We rescale (k1,k2,k3) by {0.9, 1.0, 1.1} simultaneously and recompute
the promoted pool, its concentration, and its external/perturbation enrichment.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import hypergeom

ROOT = Path(__file__).resolve().parents[2]
FULL = ROOT / "outputs/full_v2_best_fullfit/perturbation_multipanel/perturbation_multipanel_claim_tiers_edges.csv"
CTX = ROOT / "outputs/context_v2_best_fullfit/perturbation_multipanel/perturbation_multipanel_claim_tiers_edges.csv"
ENC = ROOT / "data/core/encode_tf_targets_5celllines_edges.tsv"
OUTDIR = ROOT / "outputs/revision"

RANK = {"PERT_TIER1_STRONG": 4, "PERT_TIER2_MODERATE": 3,
        "PERT_TIER3_EXPLORATORY": 2, "PERT_TIER4_BACKGROUND": 1}


def assign_tier(rank_arr, k1, k2, k3):
    out = np.where(rank_arr <= k1, "PERT_TIER1_STRONG",
          np.where(rank_arr <= k2, "PERT_TIER2_MODERATE",
          np.where(rank_arr <= k3, "PERT_TIER3_EXPLORATORY", "PERT_TIER4_BACKGROUND")))
    return out


def governance(fr, cr, in_ctx):
    # fr, cr integer tier ranks (0 if absent); in_ctx bool array
    strong = in_ctx & (cr >= 4) & (fr >= 3)
    moderate = in_ctx & (~strong) & (cr >= 3) & (fr >= 3)
    return strong, moderate


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    full = pd.read_csv(FULL)[["source", "target", "calibration_rank", "is_reference"]]
    ctx = pd.read_csv(CTX)[["source", "target", "calibration_rank", "is_reference"]]
    full.columns = ["source", "target", "rank_full", "is_ref_full"]
    ctx.columns = ["source", "target", "rank_ctx", "is_ref_ctx"]
    m = full.merge(ctx, on=["source", "target"], how="left")
    m["in_ctx"] = m["rank_ctx"].notna()

    enc = pd.read_csv(ENC, sep="\t")
    enc.columns = ["tf", "target"]
    enc_tfs = set(enc["tf"].astype(str))
    enc_edges = set(zip(enc["tf"].astype(str), enc["target"].astype(str)))
    m["source"] = m["source"].astype(str)
    m["target"] = m["target"].astype(str)
    m["tf_in_enc"] = m["source"].isin(enc_tfs)
    m["in_enc"] = [(s, t) in enc_edges for s, t in zip(m["source"], m["target"])]
    m.loc[~m["tf_in_enc"], "in_enc"] = False
    N_enc = int(m["tf_in_enc"].sum())
    K_enc = int(m.loc[m["tf_in_enc"], "in_enc"].sum())
    enc_base = K_enc / N_enc

    base = (500, 5000, 20000)
    rows = []
    for scale in (0.9, 1.0, 1.1):
        k1, k2, k3 = (int(round(base[0] * scale)), int(round(base[1] * scale)), int(round(base[2] * scale)))
        rf = m["rank_full"].to_numpy()
        rc = m["rank_ctx"].to_numpy()
        tf = assign_tier(rf, k1, k2, k3)
        tc = np.where(m["in_ctx"], assign_tier(np.nan_to_num(rc, nan=1e18), k1, k2, k3), "ABSENT")
        fr = np.array([RANK.get(x, 0) for x in tf])
        cr = np.array([RANK.get(x, 0) for x in tc])
        strong, moderate = governance(fr, cr, m["in_ctx"].to_numpy())
        promoted = strong | moderate
        prom = m[promoted]
        # concentration
        vc = prom["source"].value_counts()
        top1 = vc.iloc[0] / len(prom) if len(prom) else np.nan
        top2 = vc.iloc[:2].sum() / len(prom) if len(prom) else np.nan
        # perturbation precision (union ref in full)
        pert_prec = prom["is_ref_full"].mean()
        # encode enrichment on evaluable subset
        pe = prom[prom["tf_in_enc"]]
        enc_hits = int(pe["in_enc"].sum())
        enc_prec = enc_hits / len(pe) if len(pe) else np.nan
        enc_enr = enc_prec / enc_base if len(pe) else np.nan
        enc_p = float(hypergeom.sf(enc_hits - 1, N_enc, K_enc, len(pe))) if len(pe) else np.nan
        rows.append({
            "scale": scale, "k1": k1, "k2": k2, "k3": k3,
            "n_strong": int(strong.sum()), "n_moderate": int(moderate.sum()),
            "n_promoted": int(promoted.sum()),
            "top1_tf_frac": top1, "top2_tf_frac": top2,
            "n_unique_tfs": int(prom["source"].nunique()),
            "pert_ref_precision": pert_prec,
            "encode_enrichment": enc_enr, "encode_hypergeom_p": enc_p,
        })
    out = pd.DataFrame(rows)
    out.to_csv(OUTDIR / "tier_cutoff_sensitivity.csv", index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
