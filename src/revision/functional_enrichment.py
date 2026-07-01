#!/usr/bin/env python3
"""Functional/pathway enrichment of the promoted edge set via the Enrichr REST API.

Gene sets analysed (target genes of promoted edges):
  - promoted_all        : all unique targets of the 842 promoted edges
  - egr1_ets1_targets   : targets of the two dominant hubs (EGR1, ETS1)
  - peripheral_targets  : promoted targets after removing the top-3 hub TFs
                          (EGR1, ETS1, E2F4) -> tests for context-specific pathways
Libraries: GO_Biological_Process_2021, MSigDB_Hallmark_2020, KEGG_2021_Human.
Writes the top enriched terms (adjusted p < 0.05) per gene set x library.
"""
from __future__ import annotations
from pathlib import Path
import json
import time
import urllib.request
import uuid
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
ANN = ROOT / "outputs/revision/governance_edges_annotated.csv"
OUTDIR = ROOT / "outputs/revision"
ADDLIST = "https://maayanlab.cloud/Enrichr/addList"
ENRICH = "https://maayanlab.cloud/Enrichr/enrich"
LIBRARIES = ["GO_Biological_Process_2021", "MSigDB_Hallmark_2020", "KEGG_2021_Human"]


def _multipart(genes):
    boundary = "----wb" + uuid.uuid4().hex
    body = (
        f"--{boundary}\r\n"
        'Content-Disposition: form-data; name="list"\r\n\r\n'
        + "\n".join(genes) + "\r\n"
        f"--{boundary}\r\n"
        'Content-Disposition: form-data; name="description"\r\n\r\n'
        "promoted\r\n"
        f"--{boundary}--\r\n"
    ).encode()
    return boundary, body


def add_list(genes):
    boundary, body = _multipart(genes)
    req = urllib.request.Request(ADDLIST, data=body, method="POST")
    req.add_header("Content-Type", f"multipart/form-data; boundary={boundary}")
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read())["userListId"]


def enrich(user_list_id, library):
    url = f"{ENRICH}?userListId={user_list_id}&backgroundType={library}"
    with urllib.request.urlopen(url, timeout=60) as r:
        return json.loads(r.read())[library]


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    ann = pd.read_csv(ANN, low_memory=False)
    prom = ann[ann["promotion_allowed"] == True].copy()  # noqa: E712

    gene_sets = {
        "promoted_all": sorted(prom["target"].astype(str).unique()),
        "egr1_ets1_targets": sorted(prom[prom["source"].isin(["EGR1", "ETS1"])]["target"].astype(str).unique()),
        "peripheral_targets": sorted(prom[~prom["source"].isin(["EGR1", "ETS1", "E2F4"])]["target"].astype(str).unique()),
    }
    for k, v in gene_sets.items():
        print(f"{k}: {len(v)} genes")

    all_rows = []
    for gs_name, genes in gene_sets.items():
        if len(genes) < 5:
            continue
        try:
            uid = add_list(genes)
        except Exception as e:  # pragma: no cover
            print(f"addList FAILED for {gs_name}: {e!r}")
            continue
        time.sleep(0.5)
        for lib in LIBRARIES:
            try:
                res = enrich(uid, lib)
            except Exception as e:  # pragma: no cover
                print(f"enrich FAILED {gs_name}/{lib}: {e!r}")
                continue
            for row in res[:15]:
                # Enrichr row: [rank, term, pval, zscore, combined, genes, adj_p, ...]
                all_rows.append({
                    "gene_set": gs_name, "library": lib, "n_input_genes": len(genes),
                    "rank": row[0], "term": row[1], "p_value": row[2],
                    "combined_score": row[4], "n_overlap_genes": len(row[5]),
                    "adj_p_value": row[6],
                    "overlap_genes": ";".join(row[5][:12]),
                })
            time.sleep(0.4)

    out = pd.DataFrame(all_rows)
    out.to_csv(OUTDIR / "functional_enrichment.csv", index=False)
    print(f"\nwrote {len(out)} enrichment rows")
    if len(out):
        sig = out[out["adj_p_value"] < 0.05]
        for gs in gene_sets:
            for lib in LIBRARIES:
                s = sig[(sig.gene_set == gs) & (sig.library == lib)].head(6)
                if len(s):
                    print(f"\n### {gs} / {lib} (adj p<0.05, top 6)")
                    print(s[["term", "adj_p_value", "n_overlap_genes"]].to_string(index=False))


if __name__ == "__main__":
    main()
