"""Spot checks of outputs/r2/fm_scores against independent reads of the inputs.

Run: .venv/bin/python -m pytest subproject_24_multireference_latent_truth_model/src/r2/tests/test_fm_scores.py -q
(requires the external drive and a completed build_fm_scores.py run)
"""
import json
import pickle
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from r2 import common as C  # noqa: E402
from r2 import build_fm_scores as B  # noqa: E402

pytestmark = pytest.mark.skipif(not C.EXT.exists(), reason="external drive not mounted")


def _load(t):
    p = C.OUT / "fm_scores" / t / "scores.parquet"
    if not p.exists():
        pytest.skip(f"missing {p}")
    return pd.read_parquet(p), pd.read_csv(C.OUT / "universe" / t / "panel.tsv", sep="\t").set_index("gene")


def _meta(t):
    return json.loads((C.OUT / "fm_scores" / t / "meta.json").read_text())


def _csr_rows(g, rows):
    """List of (indices, values) per requested row of an h5ad CSR group."""
    indptr = g["indptr"][:]
    return [(g["indices"][indptr[r]:indptr[r + 1]], g["data"][indptr[r]:indptr[r + 1]]) for r in rows]


@pytest.mark.parametrize("t", C.TISSUES)
def test_universe_shape(t):
    df, panel = _load(t)
    tfs = [l.strip() for l in open(C.OUT / "universe" / t / "tfs.txt") if l.strip()]
    assert len(df) == len(tfs) * (len(panel) - 1)
    assert list(df.columns) == ["tf", "target"] + B.SCORE_COLS
    assert not (df.tf == df.target).any()
    assert not df.duplicated(["tf", "target"]).any()
    assert set(df.tf) == set(tfs) and set(df.target) == set(panel.index)
    assert (df[B.SCORE_COLS].dtypes == np.float32).all()
    assert df["random"].between(0, 1).all()
    np.testing.assert_allclose(df.pearson_abs, df.pearson.abs(), equal_nan=True)


@pytest.mark.parametrize("t", C.TISSUES)
def test_attention_spot(t):
    df, panel = _load(t)
    rng = np.random.default_rng(0)
    sub = df.iloc[rng.choice(len(df), 400, replace=False)]
    for model in ("scgpt", "geneformer"):
        ap = C.attention_paths(model, t)
        S = np.load(ap["scores"], mmap_mode="r")
        N = np.load(ap["counts"], mmap_mode="r")
        col = f"{model}_idx"
        for _, r in sub.iterrows():
            i, j = panel.loc[r.tf, col], panel.loc[r.target, col]
            for name, (a, b) in ((f"{model}_attn", (i, j)), (f"{model}_attn_T", (j, i))):
                n = int(N[a, b])
                if n == 0:
                    assert np.isnan(r[name])
                    continue
                exp = float(S[a, b]) / n if model == "scgpt" else float(S[a, b])
                assert np.isclose(r[name], exp, rtol=1e-3, atol=1e-7), (t, name, r.tf, r.target)


@pytest.mark.parametrize("t", C.TISSUES)
def test_uniform_and_norm_columns(t):
    df, _ = _load(t)
    co = pd.read_parquet(C.OUT / "fm_scores" / t / "coobs.parquet")
    assert (co.tf.to_numpy() == df.tf.to_numpy()).all() and (co.target.to_numpy() == df.target.to_numpy()).all()
    for model, max_L in (("scgpt", 1000), ("geneformer", 1001)):
        u = df[f"{model}_attn_uniform"].to_numpy(np.float64)
        n = co[f"{model}_coobs"].to_numpy()
        assert np.array_equal(np.isnan(u), n == 0)
        assert np.array_equal(np.isnan(u), np.isnan(df[f"{model}_attn"].to_numpy()))
        fin = np.isfinite(u)
        assert (u[fin] >= 1.0 / max_L - 1e-9).all() and (u[fin] <= 1.0).all()
        for v in ("attn", "attn_T"):
            np.testing.assert_allclose(df[f"{model}_{v}_norm"], df[f"{model}_{v}"].astype(np.float64) / u,
                                       rtol=2e-6, equal_nan=True)


def _independent_uniform(t, model, edges, panel, meta):
    """mean 1/L over cells where both genes are tokens, re-derived without the builder's helpers."""
    ex = meta["extraction_cells"][model]
    if model == "scgpt":
        rows = np.arange(ex["n_cells"])
        with h5py.File(C.SCGPT_PROCESSED_H5AD[t], "r") as f:
            cells = _csr_rows(f["X"], rows)
            n_var = int(f["X"].attrs["shape"][1])
        toks, Ls = [], []
        for ind, val in cells:
            x = np.zeros(n_var, dtype=np.float32)       # dense row, as ScGPTDataset sees it
            x[ind] = val
            nz = np.flatnonzero(x > 0)
            sel = nz[np.argsort(-x[nz])][: ex["max_genes"]]
            toks.append(set(sel.tolist()))
            Ls.append(len(sel))
        idx = panel["scgpt_idx"]
    else:
        rows = B.geneformer_rows(ex["n_obs_source"], ex["max_cells_used"])
        with open(C.GF_DICT_DIR / "token_dictionary_gc104M.pkl", "rb") as fh:
            tok = pickle.load(fh)
        with open(C.GF_DICT_DIR / "gene_median_dictionary_gc104M.pkl", "rb") as fh:
            med = pickle.load(fh)
        with h5py.File(C.TS_SOURCE_H5AD[t], "r") as f:
            ens = [str(x).split(".")[0] for x in B.h5_index(f, "var")]
            cells = _csr_rows(f["X"], rows)
        keep_cols = [j for j, e in enumerate(ens) if e in tok and e in med and tok[e] >= 4]
        local = np.full(len(ens), -1)
        local[keep_cols] = np.arange(len(keep_cols))
        medv = np.array([float(med[ens[j]]) or 1.0 for j in keep_cols], dtype=np.float32)
        toks, Ls = [], []
        for ind, val in cells:
            li = local[ind]
            ok = li >= 0
            x = np.zeros(len(keep_cols), dtype=np.float32)   # dense row, as the script sees it
            x[li[ok]] = val[ok]
            lib = x.sum() if x.sum() > 0 else np.float32(1.0)
            x = (x / lib) * np.float32(1e4) / medv
            nz = np.flatnonzero(x > 0)
            sel = nz[np.argsort(-x[nz])][:1000]
            toks.append(set(sel.tolist()))
            Ls.append(len(sel) + 1 if len(sel) else 0)
        idx = panel["geneformer_idx"]
    out = []
    for a, b in edges:
        i, j = int(idx[a]), int(idx[b])
        w = [1.0 / L for s, L in zip(toks, Ls) if L and i in s and j in s]
        out.append((len(w), np.mean(w) if w else np.nan))
    return out


@pytest.mark.parametrize("t", C.TISSUES)
@pytest.mark.parametrize("model", ["scgpt", "geneformer"])
def test_uniform_spot(t, model):
    df, panel = _load(t)
    co = pd.read_parquet(C.OUT / "fm_scores" / t / "coobs.parquet")
    meta = _meta(t)
    fin = np.flatnonzero(np.isfinite(df[f"{model}_attn_uniform"].to_numpy()))
    rng = np.random.default_rng(5)
    pick = rng.choice(fin, 12, replace=False)
    edges = list(zip(df.tf.to_numpy()[pick], df.target.to_numpy()[pick]))
    got = _independent_uniform(t, model, edges, panel, meta)
    for p, (n, u) in zip(pick, got):
        assert n == co[f"{model}_coobs"].iloc[p]
        assert np.isclose(u, df[f"{model}_attn_uniform"].iloc[p], rtol=1e-5), (t, model, edges)


@pytest.mark.parametrize("t", C.TISSUES)
def test_pearson_spot_on_grnboost2_cells(t):
    df, panel = _load(t)
    meta = _meta(t)
    assert meta["correlation_cells"]["cells_match_grnboost2_rule"]
    names = [l.strip() for l in (C.OUT / "grnboost2" / t / "cells.txt").read_text().splitlines() if l.strip()]
    assert meta["correlation_cells"]["n_cells_used"] == len(names)
    with h5py.File(C.SCGPT_PROCESSED_H5AD[t], "r") as f:
        ens = B._dec(f["var"]["ensembl_id"][:])
    sub = df[np.isfinite(df.pearson)].sample(5, random_state=1)
    with h5py.File(C.TS_SOURCE_H5AD[t], "r") as f:
        obs = B.h5_index(f, "obs")
        opos = {o: k for k, o in enumerate(obs)}
        rows = np.sort([opos[n] for n in names])
        s_ens = B.h5_index(f, "var")
        pos = {e: k for k, e in enumerate(s_ens)}
        for _, r in sub.iterrows():
            cols = [pos[ens[panel.loc[r.tf, "scgpt_idx"]]], pos[ens[panel.loc[r.target, "scgpt_idx"]]]]
            cmap = np.full(len(s_ens), -1)
            cmap[cols] = [0, 1]
            E = B.read_csr_rows(f["X"], rows, len(s_ens), col_map=cmap, n_out=2)
            assert np.isclose(np.corrcoef(E[:, 0], E[:, 1])[0, 1], r.pearson, atol=1e-5)
            rs = pd.Series(E[:, 0]).rank().corr(pd.Series(E[:, 1]).rank())
            assert np.isclose(rs, r.spearman, atol=1e-5)


@pytest.mark.parametrize("t", C.TISSUES)
def test_meta_checks(t):
    meta = _meta(t)
    ad = meta["attention_diagnostics"]
    # scGPT rows are stochastic over the gene tokens (no hidden <cls>): ratio ~ 1 up to float16 error
    lo, med, hi = ad["scgpt"]["row_mass_check_tf_rows"]["ratio_min_median_max"]
    assert 0.97 < lo and hi < 1.03 and abs(med - 1) < 0.01
    # Geneformer gene-key mass < 1 (the rest went to <cls>)
    assert ad["geneformer"]["row_mass_check_tf_rows"]["ratio_min_median_max"][2] <= 1.0 + 1e-4
    for model in ("scgpt", "geneformer"):
        assert meta["extraction_cells"][model]["tokens"]["n_special_tokens"] == B.N_SPECIAL_TOKENS[model]
    bins = ad["scgpt"]["float16_rel_error_bound"]["attn"]["by_coobs_bin"]
    assert sum(b["n_edges"] for b in bins.values()) == ad["scgpt"]["float16_rel_error_bound"]["attn"]["all_finite"]["n_edges"]


@pytest.mark.parametrize("t", C.TISSUES)
def test_diagnostics_use_stored_float32_values(t):
    """meta.json Spearman values reproduce from scores.parquet (float32), incl. the tied Geneformer null."""
    df, _ = _load(t)
    un = {m: _meta(t)["attention_diagnostics"][m]["uniform_null"]["spearman"] for m in ("scgpt", "geneformer")}
    for m in ("scgpt", "geneformer"):
        u = df[f"{m}_attn_uniform"].to_numpy()
        for v in ("attn", "attn_T", "attn_norm", "attn_T_norm"):
            assert np.isclose(un[m][f"{v}~uniform"], B.sp(df[f"{m}_{v}"].to_numpy(), u), atol=1e-12)
    tab = pd.read_csv(C.TABLES / "fm_attention_null.csv")
    tab = tab[(tab.tissue == t) & (tab.variant != "attn_uniform")]
    assert len(tab) == 8 and tab.spearman_with_uniform.notna().all()


@pytest.mark.parametrize("t", C.TISSUES)
def test_min_coobs_sensitivity_sets(t):
    from r2.predicted_sets import fm_coobs_column, topk_per_tf
    df, _ = _load(t)
    cb = pd.read_parquet(C.OUT / "fm_scores" / t / "coobs.parquet")
    assert (cb[["tf", "target"]].values == df[["tf", "target"]].values).all()
    d = pd.concat([df, cb[["scgpt_coobs", "geneformer_coobs"]]], axis=1)
    k, mc = C.PRESPECIFIED["topk_per_tf"], C.PRESPECIFIED["fm_min_coobs_sensitivity"]
    meta = _meta(t)
    assert meta["predicted_set_definitions"]["min_coobs_sensitivity"]["min_coobs"] == mc
    for m in ("scgpt", "geneformer"):
        sec = meta["attention_diagnostics"][m]["min_coobs_sensitivity"]
        assert sec["min_coobs"] == mc and set(sec["variants"]) == set(B.ATTN_VARIANTS)
        for v in ("attn", "attn_norm"):
            c = f"{m}_{v}"
            m0 = topk_per_tf(d, c, k)
            m1 = topk_per_tf(d, c, k, min_coobs=mc, coobs=fm_coobs_column(c))
            n = d[f"{m}_coobs"]
            assert (n[m1] >= mc).all()
            assert not (m0 & (n >= mc) & ~m1).any()
            rec = sec["variants"][v]
            assert rec["n_predicted"] == int(m1.sum()) and rec["n_predicted_unfiltered"] == int(m0.sum())
            assert np.isclose(rec["frac_unfiltered_topk_retained"], (m0 & m1).sum() / m0.sum())


@pytest.mark.parametrize("t", C.TISSUES)
def test_float16_rowsum_check(t):
    rs = _meta(t)["attention_diagnostics"]["scgpt"]["float16_rowsum_check_by_gene_frequency"]
    assert rs["all_genes"]["n_genes"] == sum(b["n_genes"] for b in rs["by_gene_cell_count"].values())
    assert rs["all_genes"]["max_abs_dev_from_1"] < 0.01
    assert 0 < rs["frac_stored_mass_in_entries_coobs_ge500"] < 1


def test_cell_sample_rule_enforced(tmp_path, monkeypatch):
    t = "kidney"
    panel = pd.read_csv(C.OUT / "universe" / t / "panel.tsv", sep="\t")
    names = [l.strip() for l in (C.OUT / "grnboost2" / t / "cells.txt").read_text().splitlines() if l.strip()]
    with h5py.File(C.TS_SOURCE_H5AD[t], "r") as f:
        obs = B.h5_index(f, "obs")
    other = [o for o in obs if o not in set(names)][:1]
    bad = tmp_path / "cells.txt"
    bad.write_text("\n".join(names[1:] + other) + "\n")          # same size, one cell swapped
    monkeypatch.setattr(B, "expression_cells_path", lambda tissue: bad)
    with pytest.raises(RuntimeError, match="sampling rule"):
        B.correlation_scores(t, panel, np.array([0]))
