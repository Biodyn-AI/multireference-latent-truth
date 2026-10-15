"""Define the per-tissue gene panel, TF list and candidate universe size (stage D1a).

Panel(t) = genes present in BOTH the scGPT attention matrix (var order of the
processed Tabula Sapiens h5ad: HVG-5000 intersected with the scGPT vocabulary)
and the Geneformer V2-104M attention matrix for tissue t, after HGNC
normalisation; duplicated symbols after normalisation are dropped.
TFs(t) = Panel(t) intersected with Lambert et al. (2018) TFs.
Universe(t) = {(tf, g): tf in TFs(t), g in Panel(t), tf != g}.

Writes outputs/r2/universe/<tissue>/{panel.tsv,tfs.txt,panel_meta.json}
where panel.tsv records, for each panel gene, its row index in each model's
attention matrix.

``--table-only`` rebuilds data/r2/tables/universe_panels.csv from the stored
panel_meta.json files (no attention matrix or h5ad is read; used by
``reproduce_all.py --tables-only`` when the external raw data are not
available) after checking each meta file against its panel.tsv and tfs.txt.
"""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import h5py

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from r2 import common as C  # noqa: E402


def scgpt_genes(tissue):
    p = C.attention_paths("scgpt", tissue)["processed_h5ad"]
    with h5py.File(p, "r") as f:
        var = f["var"]
        idx_key = var.attrs.get("_index", "_index")
        names = var[idx_key][:]
        names = [n.decode() if isinstance(n, bytes) else str(n) for n in names]
        cols = {k: var[k] for k in var.keys()}
    # prefer a feature_name column if the index holds Ensembl ids
    if names and names[0].startswith("ENSG") and "feature_name" in cols:
        fn = cols["feature_name"]
        if isinstance(fn, h5py.Group):  # categorical
            cats = [c.decode() if isinstance(c, bytes) else str(c) for c in fn["categories"][:]]
            names = [cats[i] for i in fn["codes"][:]]
        else:
            names = [n.decode() if isinstance(n, bytes) else str(n) for n in fn[:]]
    return names


def table_from_stored() -> pd.DataFrame:
    """universe_panels.csv from outputs/r2/universe/<tissue>/panel_meta.json (as written by ``main``),
    after checking n_panel, n_tfs and n_universe against panel.tsv and tfs.txt."""
    rows = []
    for t in C.TISSUES:
        od = C.OUT / "universe" / t
        meta = json.loads((od / "panel_meta.json").read_text())
        panel = pd.read_csv(od / "panel.tsv", sep="\t")
        tfs = [x for x in (od / "tfs.txt").read_text().splitlines() if x]
        assert meta["tissue"] == t, (t, meta)
        assert meta["n_panel"] == len(panel) and meta["n_tfs"] == len(tfs), (t, meta, len(panel), len(tfs))
        assert meta["n_universe"] == len(tfs) * (len(panel) - 1), (t, meta)
        rows.append(meta)
    return pd.DataFrame(rows)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--table-only", action="store_true",
                    help="rebuild universe_panels.csv from the stored panel files (no raw data read)")
    args = ap.parse_args(argv)
    if args.table_only:
        C.ensure_dir(C.TABLES)
        df = table_from_stored()
        df.to_csv(C.TABLES / "universe_panels.csv", index=False)
        print(df.to_string(index=False), flush=True)
        return
    rows = []
    for t in C.TISSUES:
        ap_s = C.attention_paths("scgpt", t)
        ap_g = C.attention_paths("geneformer", t)
        s_names = scgpt_genes(t)
        n_s = int(np.sqrt(np.load(ap_s["scores"], mmap_mode="r").shape[0] ** 2))
        A_s = np.load(ap_s["scores"], mmap_mode="r")
        A_g = np.load(ap_g["scores"], mmap_mode="r")
        g_names = [l.rstrip("\n") for l in open(ap_g["gene_names"])]
        assert A_s.shape[0] == len(s_names), (t, A_s.shape, len(s_names))
        if A_g.shape[0] != len(g_names):
            print(f"[{t}] geneformer matrix {A_g.shape} vs {len(g_names)} names", flush=True)
        s_norm = [C.normalize_symbol(x) for x in s_names]
        g_norm = [C.normalize_symbol(x) for x in g_names]
        s_df = pd.DataFrame({"gene": s_norm, "scgpt_idx": range(len(s_norm))})
        g_df = pd.DataFrame({"gene": g_norm, "geneformer_idx": range(len(g_norm))})
        s_df = s_df[s_df.gene != ""].drop_duplicates("gene", keep=False)
        g_df = g_df[g_df.gene != ""].drop_duplicates("gene", keep=False)
        panel = s_df.merge(g_df, on="gene", how="inner").sort_values("gene").reset_index(drop=True)
        tfs = sorted(set(panel.gene) & C.lambert_tfs())
        od = C.ensure_dir(C.OUT / "universe" / t)
        panel.to_csv(od / "panel.tsv", sep="\t", index=False)
        (od / "tfs.txt").write_text("\n".join(tfs) + "\n")
        meta = dict(tissue=t, n_scgpt_genes=len(s_names), n_geneformer_genes=len(g_names),
                    n_geneformer_matrix=int(A_g.shape[0]), n_panel=len(panel), n_tfs=len(tfs),
                    n_universe=len(tfs) * (len(panel) - 1))
        C.write_json(meta, od / "panel_meta.json")
        rows.append(meta)
        print(meta, flush=True)
    C.ensure_dir(C.TABLES)
    pd.DataFrame(rows).to_csv(C.TABLES / "universe_panels.csv", index=False)


if __name__ == "__main__":
    main()
