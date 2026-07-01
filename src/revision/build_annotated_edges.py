#!/usr/bin/env python3
"""Assemble a single annotated edge table reused by all revision analyses.

Columns: source (TF), target, posterior_full, posterior_context, is_reference_full,
is_reference_context, perturb_claim_tier_full, perturb_claim_tier_context,
in_context_universe, dual_universe_claim_tier, promotion_allowed, tf_in_encode,
in_encode. ENCODE membership is only defined for edges whose TF is among the 184
ENCODE ChIP-seq TFs; `tf_in_encode` marks the ENCODE-evaluable candidate space.
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
GOV = ROOT / "outputs/dual_universe_governance_v1/dual_universe_claim_tiers_edges.csv"
ENCODE = ROOT / "data/core/encode_tf_targets_5celllines_edges.tsv"
OUT = ROOT / "outputs/revision/governance_edges_annotated.csv"


def main() -> None:
    gov = pd.read_csv(GOV)
    enc = pd.read_csv(ENCODE, sep="\t")
    enc.columns = [c.strip().lower() for c in enc.columns]
    enc = enc.rename(columns={enc.columns[0]: "tf", enc.columns[1]: "target"})
    enc_tfs = set(enc["tf"].astype(str))
    enc_edges = set(zip(enc["tf"].astype(str), enc["target"].astype(str)))

    gov["source"] = gov["source"].astype(str)
    gov["target"] = gov["target"].astype(str)
    gov["tf_in_encode"] = gov["source"].isin(enc_tfs)
    gov["in_encode"] = [
        (s, t) in enc_edges for s, t in zip(gov["source"], gov["target"])
    ]
    # ENCODE membership only meaningful where the TF is assayed in ENCODE.
    gov.loc[~gov["tf_in_encode"], "in_encode"] = False

    OUT.parent.mkdir(parents=True, exist_ok=True)
    gov.to_csv(OUT, index=False)

    n = len(gov)
    print(f"annotated edges: {n}")
    print(f"ENCODE TFs: {len(enc_tfs)}  ENCODE edges: {len(enc_edges)}")
    print(f"candidate edges with TF in ENCODE: {int(gov['tf_in_encode'].sum())}")
    print(f"candidate edges present in ENCODE: {int(gov['in_encode'].sum())}")
    prom = gov[gov["promotion_allowed"]]
    print(f"promoted edges: {len(prom)}  promoted with TF in ENCODE: {int(prom['tf_in_encode'].sum())}  promoted in ENCODE: {int(prom['in_encode'].sum())}")


if __name__ == "__main__":
    main()
