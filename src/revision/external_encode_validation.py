#!/usr/bin/env python3
"""Genuinely-external validation of governance tiers against ENCODE ChIP-seq.

Two external reference sets, neither used in the latent-truth fusion:
  - encode_tfbs  : ENCODE TFBS clustered V3, 5-cell-line union (permissive; 184 TFs)
  - enets2_seq   : ENCODE ENETS2 sequence-specific edges (strict; 90 TFs)

For each set we restrict the candidate space to edges whose TF is assayed in that
set (universe-aware), take the base rate over that space, and compute the
enrichment (lift over base rate) of each dual-universe governance tier and the
promoted pool. Significance via the hypergeometric survival function.
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np
from scipy.stats import hypergeom

ROOT = Path(__file__).resolve().parents[2]
GOV = ROOT / "outputs/dual_universe_governance_v1/dual_universe_claim_tiers_edges.csv"
EXT = {
    "encode_tfbs": ROOT / "data/core/encode_tf_targets_5celllines_edges.tsv",
    "enets2_seq": ROOT / "data/core/enets2_seqspecific_edges.tsv",
}
OUTDIR = ROOT / "outputs/revision"
TIER_LABEL = {
    "DUAL_TIER1_STRONG": "Strong",
    "DUAL_TIER2_MODERATE": "Moderate",
    "DUAL_TIER3_CONDITIONAL": "Conditional",
    "DUAL_TIER4_EXPLORATORY": "Exploratory",
}


def _load_ext(path: Path):
    df = pd.read_csv(path, sep="\t")
    df.columns = [c.strip().lower() for c in df.columns]
    df = df.rename(columns={df.columns[0]: "tf", df.columns[1]: "target"})
    tfs = set(df["tf"].astype(str))
    edges = set(zip(df["tf"].astype(str), df["target"].astype(str)))
    return tfs, edges


def _hyper_p(hits: int, draws: int, K: int, N: int) -> float:
    # P(X >= hits) for drawing `draws` from N with K successes.
    if draws == 0:
        return float("nan")
    return float(hypergeom.sf(hits - 1, N, K, draws))


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    gov = pd.read_csv(GOV, low_memory=False)
    gov["source"] = gov["source"].astype(str)
    gov["target"] = gov["target"].astype(str)

    rows = []
    for set_name, path in EXT.items():
        tfs, edges = _load_ext(path)
        cand = gov[gov["source"].isin(tfs)].copy()
        cand["in_ext"] = [(s, t) in edges for s, t in zip(cand["source"], cand["target"])]
        N = len(cand)
        K = int(cand["in_ext"].sum())
        base = K / N if N else float("nan")

        groups = []
        for tier in ["DUAL_TIER1_STRONG", "DUAL_TIER2_MODERATE",
                     "DUAL_TIER3_CONDITIONAL", "DUAL_TIER4_EXPLORATORY"]:
            g = cand[cand["dual_universe_claim_tier"] == tier]
            groups.append((TIER_LABEL[tier], g))
        prom = cand[cand["promotion_allowed"] == True]  # noqa: E712
        groups.append(("Promoted(Strong+Moderate)", prom))

        for label, g in groups:
            draws = len(g)
            hits = int(g["in_ext"].sum())
            prec = hits / draws if draws else float("nan")
            enr = prec / base if base and draws else float("nan")
            p = _hyper_p(hits, draws, K, N)
            rows.append({
                "external_set": set_name,
                "tier": label,
                "n_evaluable_edges": draws,
                "n_in_external": hits,
                "precision": prec,
                "candidate_base_rate": base,
                "enrichment_vs_base": enr,
                "hypergeom_p": p,
                "candidate_space_N": N,
                "candidate_space_K": K,
            })

    out = pd.DataFrame(rows)
    out.to_csv(OUTDIR / "external_encode_enrichment.csv", index=False)
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(out.to_string(index=False))


if __name__ == "__main__":
    main()
