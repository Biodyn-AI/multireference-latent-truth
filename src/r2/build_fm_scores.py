"""Method scores over the tissue context universes (stage D1b, ``fm_scores``).

For every tissue t the context universe is

    U_ctx(t) = {(tf, target): tf in TFs(t), target in Panel(t), tf != target}

with Panel(t) and TFs(t) taken from ``outputs/r2/universe/<t>/`` (built by
``build_panels.py``).  For every edge this stage records

``scgpt_attn``, ``scgpt_attn_T``  (primary scGPT score, decision log item 5)
    Cell-averaged scGPT (whole-human) attention, mean over layers and heads.
    ``*_attn`` (TF as query) is the primary orientation, as in the original
    pipeline's ``network/infer.py``; ``*_attn_T`` is a reported variant.
    The extraction (``single_cell_mechinterp/src/interpret/attention.py``,
    ``cli.py extract-attention``) accumulates, for every cell, the
    layer/head-mean attention matrix of the genes in that cell's input
    sequence into ``attention_scores.npy`` (a **sum**, float16) and the number
    of cells in which both genes were in the sequence into
    ``attention_counts.npy`` (int32).  The original pipeline's score is
    ``finalize_attention_scores = sum / max(count, 1)``; rows are attention
    *queries* and columns *keys* (torch ``MultiheadAttention`` weights,
    ``(batch, heads, L_query, S_key)``), and ``network/infer.py`` reads a
    ``source -> target`` score as ``scores[source, target]``.  Hence

        scgpt_attn(tf, g)   = sum[tf, g] / count[tf, g]   (TF as query, g as key)
        scgpt_attn_T(tf, g) = sum[g, tf] / count[g, tf]   (g as query, TF as key)

``geneformer_attn``, ``geneformer_attn_T``
    Geneformer V2-104M, same definition.  ``extract_geneformer_attention.py``
    already stores the **mean** (``score_sum / max(count, 1)``, float32) in
    ``attention_scores.npy``, so no further division is applied.  The <cls>
    row/column is dropped by the extraction; rows are queries, columns keys.

    In both models pairs that were never co-observed (count == 0) get NaN.

``scgpt_attn_uniform``, ``geneformer_attn_uniform``  (reported null)
    Mean over the co-observing extraction cells c of 1 / L_c, where L_c is the
    number of keys the softmax of cell c runs over, i.e. the number of
    non-padding input tokens of the cell for that model:

    * scGPT: L_c = number of gene tokens (nonzero HVG genes, top ``max_genes``
      by value); no <cls> token, padding is masked (``src_key_padding_mask``);
    * Geneformer: L_c = 1 + number of gene tokens (<cls> + rank-value genes,
      top 1000; padding masked by ``attention_mask``, no <eos>).

    A layer/head-mean attention row sums to 1 over the L_c keys, so 1 / L_c is
    the attention a query would give every key under uniform attention, and
    ``*_attn_uniform`` is what ``*_attn`` would be if attention were uniform.
    It is computed from expression alone: the per-cell token sets are
    reconstructed from the extraction configs/scripts and verified by
    reproducing the stored co-observation counts of the panel block exactly.
    The co-observing cells are the same for (tf, g) and (g, tf), so the null is
    shared by ``*_attn`` and ``*_attn_T``.  For scGPT the row-stochasticity
    (no hidden keys) is checked on the full stored TF rows
    (sum_j sum[tf, j] == count[tf, tf]); for Geneformer the attention mass on
    gene keys of the TF rows is reported (the rest went to <cls>).

``scgpt_attn_norm``, ``scgpt_attn_T_norm``, ``geneformer_attn_norm``, ``geneformer_attn_T_norm``
    Length-normalised robustness variants: ``*_attn / *_attn_uniform`` and
    ``*_attn_T / *_attn_uniform`` (1 = uniform attention).

``pearson``, ``spearman`` (+ ``*_abs``)
    Signed correlation of TF and target expression across the **same cells
    used by GRNBoost2/GENIE3** (``outputs/r2/grnboost2/<t>/cells.txt``, 3,000
    cells drawn by ``default_rng(PRESPECIFIED['random_seed'])``; the file is
    checked against that rule).  Expression is the single log-normalised
    Tabula Sapiens matrix, log1p(CP10k) of the DecontX-corrected counts
    (``X`` of ``C.TS_SOURCE_H5AD``; genes matched by Ensembl ID through the
    processed var), identical to the GRNBoost2 input.  ``X`` of the processed
    h5ad is NOT used: ``prepare-data`` applied ``normalize_total`` + ``log1p``
    to an already log-normalised matrix, so it is a double-log transform
    (verified and recorded in meta.json).  Spearman uses average ranks.
    Genes with zero variance in the sample give NaN.

``random``
    Seeded uniform(0, 1).

Methods are scored on different cells (scGPT: first 2,000 processed cells;
Geneformer: a seed-42 random draw of 1,200/2,000/1,200 cells; correlations and
GRNBoost2/GENIE3: the shared 3,000-cell sample); all three sets are listed in
``data/r2/tables/fm_extraction_cells.csv``.

Known properties (numbers in meta.json and the tables below)
------------------------------------------------------------
* Raw scGPT mean attention tracks the 1/L null (Spearman ``attn ~
  attn_uniform``; scGPT sequences are short and variable, no cell reaches the
  1,000-token cap), and the raw scGPT top-k sets are enriched for edges seen
  in a single cell.  Geneformer sequences sit at the cap (+<cls>) in almost
  every cell, so its 1/L null is nearly constant and explains little; about a
  quarter of each Geneformer query's attention goes to <cls>
  (``row_mass_check_tf_rows``).
* The association of raw scGPT attention with the number of co-observing
  cells is not of one sign across tissues (negative in immune and lung,
  slightly positive in kidney).  That pairs seen in few cells score higher
  because those cells have shorter sequences (larger 1/L) is a hypothesis
  that the 1/L null supports only in part: after normalisation a weak
  negative association remains in every tissue.
* scGPT sums were accumulated in float16.  ``float16_rel_error_bound``
  (meta.json, ``fm_float16_error.csv``) is a conservative worst-case ceiling
  (every rounding of the running sum in the same direction): it grows with the
  number of co-observing cells and exceeds 25% for a fifth to a half of the
  edges seen in >= 500 cells.  The realised error is much smaller in aggregate:
  every scGPT gene row sums to 1 per cell over its keys, and the stored row
  sums divided by the gene's cell count stay within about 1 +- 1% in every
  gene-frequency bin, including genes present in >= 1,000 extraction cells
  (``float16_rowsum_check_by_gene_frequency``, ``fm_float16_rowsum_check.csv``).
  The row-sum check bounds the error of the attention *mass*, not of single
  small entries, so per-edge errors of frequently co-observed pairs remain a
  stated limitation; the ceiling is reported within every top-k set.
* Top-k predicted sets follow ``r2.predicted_sets.topk_per_tf`` (NaN = not
  predicted, seeded random break of ties at the k-th position).
* Minimum-co-observation sensitivity (decision-log item 9,
  ``PRESPECIFIED['fm_min_coobs_sensitivity']`` = 5): for every attention column
  a second predicted set ``topk_per_tf(..., min_coobs=5, coobs=<model>_coobs)``
  in which only edges co-observed in >= 5 extraction cells are eligible
  (``min_coobs_sensitivity`` in meta.json, ``fm_min_coobs_sensitivity.csv``).
  The normalised variant removes the 1/L rank link but not the excess of
  single-cell edges in the top-k sets; this set addresses the latter.
* All diagnostics (Spearman values, top-k sets, coverage) are computed on the
  stored float32 values, i.e. what downstream stages read.  This matters for the
  Geneformer 1/L null, most of whose values are exactly tied in float32.

Outputs
-------
outputs/r2/fm_scores/<t>/scores.parquet   tf, target, 15 score columns (float32)
outputs/r2/fm_scores/<t>/coobs.parquet    tf, target, scgpt_coobs, geneformer_coobs (int32)
outputs/r2/fm_scores/<t>/meta.json
data/r2/tables/fm_universe_summary.csv    one row per tissue
data/r2/tables/fm_extraction_cells.csv    cells behind every score family
data/r2/tables/fm_attention_null.csv      tissue x model x attention variant vs the 1/L null
data/r2/tables/fm_float16_error.csv       scGPT float16 error bound by co-observation bin
data/r2/tables/fm_float16_rowsum_check.csv scGPT row-sum / cell-count by gene frequency
data/r2/tables/fm_min_coobs_sensitivity.csv  top-k sets with >= 5 co-observing cells

Requires ``outputs/r2/grnboost2/<t>/cells.txt`` (written at the start of
``build_grnboost2.py run``); the run stops if that file does not follow the
GRNBoost2 sampling rule.

Usage
-----
    .venv/bin/python subproject_24_multireference_latent_truth_model/src/r2/build_fm_scores.py
    (option: --tissues immune,kidney,lung)

``--tables-only`` rebuilds the six fm_* tables from the stored
``outputs/r2/fm_scores/<t>/meta.json`` files (the tables are functions of the
meta dictionaries only; no attention matrix or h5ad is read).  Used by
``reproduce_all.py --tables-only`` when the external raw data are not available.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import pickle
import re
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.stats import rankdata, spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from r2 import common as C  # noqa: E402
from r2.predicted_sets import fm_coobs_column, topk_per_tf  # noqa: E402

STAGE = "fm_scores"
SCM = C.EXT / "single_cell_mechinterp"

# Configs actually used for the scGPT extraction (run_phase0_revision.sh).
SCGPT_CONFIG = {
    "kidney": SCM / "configs/eval_bias_scgpt_kidney.yaml",
    "lung": SCM / "configs/eval_bias_scgpt_lung.yaml",
    "immune": SCM / "configs/revision/eval_bias_scgpt_immune.yaml",
}
GENEFORMER_SCRIPT = SCM / "scripts/extract_geneformer_attention.py"
GENEFORMER_LOGS = [SCM / "outputs/eval_bias/geneformer/extract_lung_immune.log",
                   SCM / "outputs/eval_bias/geneformer/extract_kidney.log"]
GENEFORMER_MAX_GENES = 1000
# special (non-gene) tokens inside each cell's softmax: scGPT none (cls_token_id
# null in the configs), Geneformer one <cls> (script line 151; no <eos>)
N_SPECIAL_TOKENS = {"scgpt": 0, "geneformer": 1}

ATTN_VARIANTS = ["attn", "attn_T", "attn_uniform", "attn_norm", "attn_T_norm"]
SCORE_COLS = ([f"scgpt_{v}" for v in ATTN_VARIANTS] + [f"geneformer_{v}" for v in ATTN_VARIANTS]
              + ["pearson", "pearson_abs", "spearman", "spearman_abs", "random"])
# per-tissue RNG stream for the random score: default_rng([seed, tissue_index, 2])
STREAM_RANDOM = 2
# co-observation bins for the float16 error breakdown (inclusive bounds)
COOBS_BINS = [(1, 1), (2, 4), (5, 19), (20, 99), (100, 499), (500, None)]
# gene-frequency bins (cells in which the gene is a token) for the float16 row-sum check
GENE_FREQ_BINS = [(1, 19), (20, 99), (100, 499), (500, 999), (1000, None)]
# minimum-co-observation sensitivity (decision-log item 9)
MIN_COOBS = int(C.PRESPECIFIED["fm_min_coobs_sensitivity"])
MC_TAG = f"|coobs>={MIN_COOBS}"


# ----------------------------------------------------------------------------
# small helpers
# ----------------------------------------------------------------------------
def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _dec(a) -> np.ndarray:
    return np.array([x.decode() if isinstance(x, bytes) else str(x) for x in a], dtype=object)


def file_info(p: Path, md5: bool = False) -> dict:
    p = Path(p)
    if not p.exists():
        return {"path": str(p), "exists": False}
    st = p.stat()
    out = {"path": str(p), "bytes": int(st.st_size),
           "mtime": _dt.datetime.fromtimestamp(st.st_mtime).isoformat(timespec="seconds")}
    if md5:
        out["md5"] = hashlib.md5(p.read_bytes()).hexdigest()
    return out


def rng_for(tissue: str, stream: int) -> np.random.Generator:
    return np.random.default_rng([C.PRESPECIFIED["random_seed"], C.TISSUES.index(tissue), stream])


def h5_obs_column(f: h5py.File, col: str) -> np.ndarray:
    g = f["obs"][col]
    if isinstance(g, h5py.Group):  # categorical
        cats = _dec(g["categories"][:])
        return cats[g["codes"][:]]
    return _dec(g[:])


def h5_index(f: h5py.File, grp: str) -> np.ndarray:
    g = f[grp]
    return _dec(g[g.attrs.get("_index", "_index")][:])


def read_csr_rows(g: h5py.Group, rows: np.ndarray, n_cols: int, col_map: np.ndarray | None = None,
                  n_out: int | None = None, dtype=np.float64) -> np.ndarray:
    """Dense rows of a CSR h5ad matrix.  ``col_map`` maps source column -> output
    column (-1 = drop); without it all ``n_cols`` columns are returned."""
    indptr = g["indptr"][:]
    width = n_cols if col_map is None else int(n_out)
    out = np.zeros((len(rows), width), dtype=dtype)
    data_ds, ind_ds = g["data"], g["indices"]
    for k, r in enumerate(rows):
        a, b = indptr[r], indptr[r + 1]
        if b <= a:
            continue
        d = data_ds[a:b]
        i = ind_ds[a:b]
        if col_map is None:
            out[k, i] = d
        else:
            j = col_map[i]
            keep = j >= 0
            out[k, j[keep]] = d[keep]
    return out


def composition(values: np.ndarray) -> dict:
    u, c = np.unique(values.astype(str), return_counts=True)
    order = np.argsort(-c, kind="stable")
    return {str(u[i]): int(c[i]) for i in order}


def quantiles(x: np.ndarray, qs=(0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0)) -> dict:
    x = np.asarray(x, dtype=float)
    return {str(q): float(np.quantile(x, q)) for q in qs} if x.size else {}


def sp(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    m = np.isfinite(a) & np.isfinite(b)
    return float(spearmanr(a[m], b[m])[0]) if m.sum() > 2 else None


# ----------------------------------------------------------------------------
# universe
# ----------------------------------------------------------------------------
def load_universe(tissue: str):
    d = C.OUT / "universe" / tissue
    panel = pd.read_csv(d / "panel.tsv", sep="\t")
    tfs = [l.strip() for l in open(d / "tfs.txt") if l.strip()]
    pos = {g: i for i, g in enumerate(panel.gene)}
    assert len(pos) == len(panel), "duplicated panel genes"
    tf_pos = np.array([pos[t] for t in tfs], dtype=np.int64)
    P = len(panel)
    # edge order: TFs in tfs.txt order, targets in panel order, self excluded
    e_tf = np.repeat(np.arange(len(tfs)), P)
    e_tg = np.tile(np.arange(P), len(tfs))
    keep = e_tg != tf_pos[e_tf]
    e_tf, e_tg = e_tf[keep], e_tg[keep]
    return panel, tfs, tf_pos, e_tf, e_tg


# ----------------------------------------------------------------------------
# attention
# ----------------------------------------------------------------------------
def read_block(mat: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """mat[np.ix_(idx, idx)] for a (memory-mapped) square matrix, reading whole rows
    in ascending order (one pass over the file)."""
    order = np.argsort(idx, kind="stable")
    s = idx[order]
    block = np.asarray(mat[s])[:, s]
    inv = np.empty_like(order)
    inv[order] = np.arange(len(order))
    return block[np.ix_(inv, inv)]


def attention_block(model: str, tissue: str, idx: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
    """Return (mean attention block with NaN where never co-observed, co-observation
    counts block, diagnostics) for the panel genes ``idx`` (matrix indices)."""
    ap = C.attention_paths(model, tissue)
    S = np.load(ap["scores"], mmap_mode="r")
    N = np.load(ap["counts"], mmap_mode="r")
    assert S.shape == N.shape and S.shape[0] == S.shape[1]
    s_blk = read_block(S, idx).astype(np.float64)
    n_blk = read_block(N, idx).astype(np.int64)
    diag = {"score_dtype": str(S.dtype), "count_dtype": str(N.dtype), "matrix_n": int(S.shape[0]),
            "nonfinite_stored_scores": int((~np.isfinite(s_blk)).sum())}
    seen = n_blk > 0
    if model == "scgpt":
        # stored = float16 SUM over cells -> mean = sum / count (finalize_attention_scores)
        mean = np.full(s_blk.shape, np.nan)
        mean[seen] = s_blk[seen] / n_blk[seen]
        diag["stored_quantity"] = "sum over co-observing cells (float16 accumulator)"
        diag["max_stored_sum"] = float(np.nanmax(s_blk))
    else:
        # stored = MEAN already (score_sum / max(count, 1)); zero where never co-observed
        mean = np.where(seen, s_blk, np.nan)
        diag["stored_quantity"] = "mean over co-observing cells (float32)"
        diag["max_abs_stored_score_where_count0"] = float(np.abs(s_blk[~seen]).max()) if (~seen).any() else 0.0
    off = ~np.eye(len(idx), dtype=bool)
    diag["symmetric_counts"] = bool(np.array_equal(n_blk, n_blk.T))
    diag["max_count_offdiag"] = int(n_blk[off].max())
    return mean, n_blk, diag


def row_mass_check(model: str, tissue: str, tf_idx: np.ndarray, L_cells: np.ndarray) -> dict:
    """Attention mass of the TF query rows on gene keys, from the full stored rows.

    Per cell a query row sums to 1 over its L keys, so summed over the cells that
    contain the TF:  sum_j stored_sum[tf, j] = count[tf, tf] - (mass on special tokens).
    scGPT (no special tokens): the ratio must be 1 up to float16 accumulation error.
    Geneformer (stored mean, <cls> dropped): the ratio is the mean gene-key mass and
    1 - ratio the mean attention on <cls>.
    """
    ap = C.attention_paths(model, tissue)
    S = np.load(ap["scores"], mmap_mode="r")
    N = np.load(ap["counts"], mmap_mode="r")
    idx = np.unique(tf_idx)
    s_rows = np.asarray(S[idx]).astype(np.float64)
    n_rows = np.asarray(N[idx]).astype(np.float64)
    n_self = n_rows[np.arange(len(idx)), idx]
    ok = n_self > 0
    mass = (s_rows if model == "scgpt" else s_rows * n_rows).sum(axis=1)
    ratio = mass[ok] / n_self[ok]
    out = {"n_tf_rows": int(ok.sum()),
           "definition": ("sum_j stored_sum[tf,j] / count[tf,tf]" if model == "scgpt"
                          else "sum_j stored_mean[tf,j]*count[tf,j] / count[tf,tf] (mean gene-key mass)"),
           "ratio_min_median_max": [float(ratio.min()), float(np.median(ratio)), float(ratio.max())]}
    if model == "geneformer":
        Lc = L_cells[L_cells > 0].astype(float)
        out["uniform_expected_gene_key_mass_mean_over_cells"] = float(np.mean((Lc - 1) / Lc))
        out["mean_cls_mass_median_over_tfs"] = float(1 - np.median(ratio))
    return out


# ----------------------------------------------------------------------------
# reconstruction of the extraction cell sets and per-cell token sets
# ----------------------------------------------------------------------------
def scgpt_cell_set(tissue: str) -> dict:
    """Cells of the scGPT extraction per its config: attention.max_cells with the
    default sample_strategy ('head') -> the first max_cells cells of the processed h5ad."""
    txt = SCGPT_CONFIG[tissue].read_text()
    m = re.search(r"max_cells:\s*(\d+)", txt)
    max_cells = int(m.group(1)) if m else None
    strat = re.search(r"sample_strategy:\s*(\w+)", txt)
    strategy = strat.group(1) if strat else "head (default; key absent from config)"
    maxg = int(re.search(r"max_genes:\s*(\d+)", txt).group(1))
    cls = re.search(r"cls_token_id:\s*(\S+)", txt)
    assert cls is None or cls.group(1) == "null", "scGPT extraction config sets a <cls> token"
    with h5py.File(C.SCGPT_PROCESSED_H5AD[tissue], "r") as f:
        n_obs = int(f["X"].attrs["shape"][0])
    n = min(max_cells, n_obs) if max_cells else n_obs
    assert strategy.startswith("head"), strategy
    return {"config": str(SCGPT_CONFIG[tissue]), "max_cells": max_cells, "sample_strategy": strategy,
            "max_genes": maxg, "cls_token_id": None, "n_obs_processed": n_obs, "rows": np.arange(n)}


def scgpt_tokens(tissue: str, rows: np.ndarray, panel_idx: np.ndarray, max_genes: int):
    """ScGPTDataset(include_zero=False, sort_by_expression=True, no CLS): a cell's tokens are
    its genes with X > 0, sorted by value (np.argsort(-x)), truncated to max_genes.
    Returns (B, n_gene_tokens): B[c, p] = 1 if panel gene p is a token of cell c."""
    with h5py.File(C.SCGPT_PROCESSED_H5AD[tissue], "r") as f:
        g = f["X"]
        n_var = int(g.attrs["shape"][1])
        X = read_csr_rows(g, rows, n_var, dtype=np.float32)
    col = np.full(n_var, -1, dtype=np.int64)
    col[panel_idx] = np.arange(len(panel_idx))
    B = np.zeros((len(rows), len(panel_idx)), dtype=np.float64)
    n_tok = np.zeros(len(rows), dtype=np.int64)
    for k in range(len(rows)):
        x = X[k]
        nz = np.where(x > 0)[0]
        ordr = nz[np.argsort(-x[nz])][:max_genes]
        n_tok[k] = len(ordr)
        j = col[ordr]
        B[k, j[j >= 0]] = 1.0
    return B, n_tok


def _gf_parse_log_cells() -> dict:
    """Total cells per tissue from 'cells i/N' progress lines of the Geneformer logs."""
    out = {}
    for p in GENEFORMER_LOGS:
        if not p.exists():
            continue
        cur = None
        for line in p.read_text(errors="replace").splitlines():
            m = re.match(r"=== (\w+) ===", line.strip())
            if m:
                cur = m.group(1)
            m = re.search(r"cells \d+/(\d+)", line)
            if m and cur:
                out[cur] = int(m.group(1))
    return out


def geneformer_tokenizable(tissue: str):
    with open(C.GF_DICT_DIR / "token_dictionary_gc104M.pkl", "rb") as fh:
        token_dict = pickle.load(fh)
    with open(C.GF_DICT_DIR / "gene_median_dictionary_gc104M.pkl", "rb") as fh:
        median_dict = pickle.load(fh)
    with h5py.File(C.TS_SOURCE_H5AD[tissue], "r") as f:
        var_ens = [str(g).split(".")[0] for g in h5_index(f, "var")]
    keep_cols, medians = [], []
    for j, ens in enumerate(var_ens):
        if ens in token_dict and ens in median_dict and token_dict[ens] >= 4:
            keep_cols.append(j)
            medians.append(float(median_dict[ens]))
    medians = np.array(medians, dtype=np.float32)
    medians[medians == 0] = 1.0
    return np.array(keep_cols), medians, len(var_ens)


def geneformer_rows(n_obs: int, n_cells: int) -> np.ndarray:
    """extract_geneformer_attention.py: rng = default_rng(42); sorted choice without replacement."""
    if n_obs <= n_cells:
        return np.arange(n_obs)
    return np.sort(np.random.default_rng(42).choice(n_obs, size=n_cells, replace=False))


def geneformer_tokens(tissue: str, rows: np.ndarray, panel_idx: np.ndarray,
                      keep_cols: np.ndarray, medians: np.ndarray, n_var: int,
                      max_genes: int = GENEFORMER_MAX_GENES, target_sum: float = 1e4):
    """Rank-value tokenisation exactly as the extraction script: X (the TS matrix as stored,
    i.e. log-normalised) on tokenisable genes, / row sum * 1e4, / gene median, nonzero genes
    ranked descending, top max_genes.  Returns (B, n_gene_tokens) as ``scgpt_tokens``."""
    col = np.full(n_var, -1, dtype=np.int64)
    col[keep_cols] = np.arange(len(keep_cols))
    with h5py.File(C.TS_SOURCE_H5AD[tissue], "r") as f:
        X = read_csr_rows(f["X"], rows, n_var, col_map=col, n_out=len(keep_cols), dtype=np.float32)
    lib = X.sum(axis=1, keepdims=True)
    lib[lib == 0] = 1.0
    Xn = (X / lib) * target_sum
    Xn = Xn / medians[None, :]
    pcol = np.full(len(keep_cols), -1, dtype=np.int64)
    pcol[panel_idx] = np.arange(len(panel_idx))
    B = np.zeros((len(rows), len(panel_idx)), dtype=np.float64)
    n_tok = np.zeros(len(rows), dtype=np.int64)
    for k in range(len(rows)):
        v = Xn[k]
        nz = np.where(v > 0)[0]
        if nz.size == 0:
            continue                      # the script skips cells without tokens
        ordr = nz[np.argsort(-v[nz])][:max_genes]
        n_tok[k] = len(ordr)
        j = pcol[ordr]
        B[k, j[j >= 0]] = 1.0
    return B, n_tok


def coobs_and_uniform(B: np.ndarray, L: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """counts[i, j] = #cells with both tokens; uniform[i, j] = mean of 1/L_c over those cells
    (NaN where counts == 0).  Cells with L_c == 0 contribute nothing."""
    w = np.where(L > 0, 1.0 / np.maximum(L, 1), 0.0)
    counts = np.rint(B.T @ B).astype(np.int64)
    usum = (B * w[:, None]).T @ B
    uniform = np.full(counts.shape, np.nan)
    seen = counts > 0
    uniform[seen] = usum[seen] / counts[seen]
    return counts, uniform


def compare_counts(rep: np.ndarray, stored: np.ndarray) -> dict:
    d = np.abs(rep - stored)
    return {"exact_match_frac": float((d == 0).mean()), "max_abs_diff": int(d.max()),
            "n_mismatch": int((d > 0).sum()), "n_pairs": int(d.size)}


def donors_of(rows: np.ndarray, path: Path) -> dict:
    with h5py.File(path, "r") as f:
        donors = h5_obs_column(f, "donor_id")[rows]
        method = h5_obs_column(f, "method")[rows] if "method" in f["obs"] else np.array([])
    return {"n_cells": int(len(rows)), "n_donors": int(len(set(donors))),
            "donors": composition(donors), "assay_method": composition(method) if len(method) else {}}


def token_stats(model: str, n_tok: np.ndarray, max_genes: int) -> dict:
    have = n_tok > 0
    L = n_tok[have] + N_SPECIAL_TOKENS[model]
    return {"n_special_tokens": N_SPECIAL_TOKENS[model], "n_cells_with_tokens": int(have.sum()),
            "softmax_length_L_quantiles": quantiles(L),
            "frac_cells_at_gene_token_cap": float((n_tok[have] >= max_genes).mean()) if have.any() else None,
            "gene_token_cap": int(max_genes)}


def extraction_tokens(tissue: str, model: str, panel: pd.DataFrame, n_blk: np.ndarray):
    """Reconstruct the extraction cells and their token sets, verify them against the
    stored co-observation counts of the panel block (exact match required), and return
    (provenance info, uniform-null block, softmax lengths per cell)."""
    if model == "scgpt":
        sc_set = scgpt_cell_set(tissue)
        info = {k: v for k, v in sc_set.items() if k != "rows"}
        info.update(donors_of(sc_set["rows"], C.SCGPT_PROCESSED_H5AD[tissue]))
        info["cell_selection"] = f"first {len(sc_set['rows'])} cells (obs order) of the processed h5ad"
        info["input_values"] = ("processed X (normalize_total+log1p applied to TS log-normalised X), "
                                "continuous values; tokens = nonzero HVG genes sorted by value, top max_genes")
        B, n_tok = scgpt_tokens(tissue, sc_set["rows"], panel.scgpt_idx.to_numpy(), sc_set["max_genes"])
        L = np.where(n_tok > 0, n_tok + N_SPECIAL_TOKENS[model], 0)
        counts, uniform = coobs_and_uniform(B, L)
        info["verification_coobs_panel_block"] = compare_counts(counts, n_blk)
        info["tokens"] = token_stats(model, n_tok, sc_set["max_genes"])
        if info["verification_coobs_panel_block"]["exact_match_frac"] != 1.0:
            raise RuntimeError(f"[{tissue}] scGPT token sets do not reproduce the stored counts: "
                               f"{info['verification_coobs_panel_block']}")
        return info, uniform, L

    log_cells = _gf_parse_log_cells()
    with h5py.File(C.TS_SOURCE_H5AD[tissue], "r") as f:
        n_obs_src = int(f["X"].attrs["shape"][0])
    info = {"script": str(GENEFORMER_SCRIPT), "source_h5ad": str(C.TS_SOURCE_H5AD[tissue]),
            "n_obs_source": n_obs_src, "sample_seed": 42, "max_genes": GENEFORMER_MAX_GENES,
            "max_cells_in_log": log_cells.get(tissue),
            "input_values": ("TS h5ad X as stored (log1p CP10k, treated by the script as counts) "
                             "on tokenisable genes, /row sum*1e4, /gene median; rank-value, top 1000; "
                             "<cls> prepended")}
    candidates = [log_cells[tissue]] if tissue in log_cells else []
    candidates += [c for c in (1200, 2000) if c not in candidates]
    keep_cols, medians, n_var = geneformer_tokenizable(tissue)
    info["n_tokenisable_genes"] = int(len(keep_cols))
    info["verification_coobs_panel_block"] = {}
    for nc in candidates:
        rows = geneformer_rows(n_obs_src, nc)
        B, n_tok = geneformer_tokens(tissue, rows, panel.geneformer_idx.to_numpy(), keep_cols, medians, n_var)
        L = np.where(n_tok > 0, n_tok + N_SPECIAL_TOKENS[model], 0)
        counts, uniform = coobs_and_uniform(B, L)
        cmp = compare_counts(counts, n_blk)
        info["verification_coobs_panel_block"][str(nc)] = cmp
        log(f"  [{tissue}] geneformer candidate max_cells={nc}: {cmp}")
        if cmp["exact_match_frac"] == 1.0:
            info["max_cells_used"] = nc
            info.update(donors_of(rows, C.TS_SOURCE_H5AD[tissue]))
            info["cell_selection"] = (f"np.random.default_rng(42).choice({n_obs_src}, {nc}, replace=False), "
                                      "sorted, from the source TS h5ad")
            info["tokens"] = token_stats(model, n_tok, GENEFORMER_MAX_GENES)
            return info, uniform, L
    raise RuntimeError(f"[{tissue}] no Geneformer cell-set candidate {candidates} reproduces the stored counts")


# ----------------------------------------------------------------------------
# correlations (on the GRNBoost2 cell sample)
# ----------------------------------------------------------------------------
def expression_cells_path(tissue: str) -> Path:
    return C.OUT / "grnboost2" / tissue / "cells.txt"


def _corr_tf_rows(Z: np.ndarray, tf_pos: np.ndarray) -> np.ndarray:
    """Pearson correlation of columns tf_pos with all columns of Z (cells x genes)."""
    Zc = Z - Z.mean(axis=0, keepdims=True)
    nrm = np.sqrt((Zc ** 2).sum(axis=0))
    num = Zc[:, tf_pos].T @ Zc
    with np.errstate(invalid="ignore", divide="ignore"):
        R = num / np.outer(nrm[tf_pos], nrm)
    R[:, nrm == 0] = np.nan
    R[nrm[tf_pos] == 0, :] = np.nan
    return np.clip(R, -1.0, 1.0)


def correlation_scores(tissue: str, panel: pd.DataFrame, tf_pos: np.ndarray):
    proc = C.SCGPT_PROCESSED_H5AD[tissue]
    src = C.TS_SOURCE_H5AD[tissue]
    cpath = expression_cells_path(tissue)
    if not cpath.exists():
        raise FileNotFoundError(f"{cpath} missing: run build_grnboost2.py run --tissue {tissue} first "
                                "(it writes the expression-baseline cell sample)")
    names = [l.strip() for l in cpath.read_text().splitlines() if l.strip()]
    with h5py.File(proc, "r") as fp, h5py.File(src, "r") as fs:
        obs_p, obs_s = h5_index(fp, "obs"), h5_index(fs, "obs")
        assert np.array_equal(obs_p, obs_s), "processed and source h5ad obs differ"
        pos_obs = {o: i for i, o in enumerate(obs_s)}
        assert len(pos_obs) == len(obs_s), "duplicated obs names"
        rows = np.sort(np.array([pos_obs[n] for n in names], dtype=np.int64))
        assert len(np.unique(rows)) == len(rows) == len(names), "duplicated cells in cells.txt"
        n_obs = len(obs_s)
        rule = (np.arange(n_obs) if len(rows) >= n_obs else
                np.sort(np.random.default_rng(C.PRESPECIFIED["random_seed"]).choice(n_obs, size=len(rows),
                                                                                    replace=False)))
        if not np.array_equal(rows, rule):
            raise RuntimeError(f"[{tissue}] {cpath} does not follow the GRNBoost2 sampling rule "
                               f"(default_rng({C.PRESPECIFIED['random_seed']}).choice(n_obs, {len(rows)}))")
        p_ens = _dec(fp["var"]["ensembl_id"][:])
        s_ens = h5_index(fs, "var")
        n_var_s = len(s_ens)
        s_pos = {e: i for i, e in enumerate(s_ens)}
        panel_ens = p_ens[panel.scgpt_idx.to_numpy()]
        src_cols = np.array([s_pos[e] for e in panel_ens], dtype=np.int64)
        col_map = np.full(n_var_s, -1, dtype=np.int64)
        col_map[src_cols] = np.arange(len(src_cols))
        log(f"  [{tissue}] reading {len(rows)} GRNBoost2 cells x {len(src_cols)} genes (TS log-normalised X)")
        E = read_csr_rows(fs["X"], rows, n_var_s, col_map=col_map, n_out=len(src_cols))
        # diagnostics: is source X log1p(CP10k)?  is processed X a double log?
        full = read_csr_rows(fs["X"], rows[:50], n_var_s, dtype=np.float64)
        cp10k_rowsum = np.expm1(full).sum(axis=1)
        n_var_p = int(fp["X"].attrs["shape"][1])
        Pp = read_csr_rows(fp["X"], rows, n_var_p, dtype=np.float64)[:, panel.scgpt_idx.to_numpy()]
        ratios = []
        for k in range(50):
            m = (E[k] > 0) & (Pp[k] > 0)
            if m.sum() >= 5:
                r = np.expm1(Pp[k, m]) / E[k, m]
                ratios.append(float(r.std() / r.mean()))
        donors = h5_obs_column(fs, "donor_id")[rows]
        method = h5_obs_column(fs, "method")[rows]
    log(f"  [{tissue}] computing pearson/spearman")
    R_p = _corr_tf_rows(E, tf_pos)
    ranks = rankdata(E, axis=0, method="average")
    R_s = _corr_tf_rows(ranks, tf_pos)
    R_proc = _corr_tf_rows(Pp, tf_pos)
    zero_var = E.std(axis=0) == 0
    diag = {
        "expression_source": str(src),
        "expression_definition": "TS X = log1p(CP10k of DecontX counts); genes matched via processed var ensembl_id",
        "cell_sample": "shared with GRNBoost2/GENIE3: outputs/r2/grnboost2/<t>/cells.txt",
        "cells_file": file_info(cpath, md5=True),
        "cells_match_grnboost2_rule": bool(np.array_equal(rows, rule)),
        "cells_rule": f"np.sort(default_rng({C.PRESPECIFIED['random_seed']}).choice(n_obs, n, replace=False))",
        "n_cells_available": int(n_obs), "n_cells_used": int(len(rows)),
        "donors": composition(donors), "n_donors": int(len(set(donors))), "assay_method": composition(method),
        "source_X_expm1_rowsum_first50_min_max": [float(cp10k_rowsum.min()), float(cp10k_rowsum.max())],
        "processed_X_is_log1p_of_rescaled_sourceX_cv_of_ratio_median": float(np.median(ratios)) if ratios else None,
        "n_panel_genes_zero_variance": int(zero_var.sum()),
        "n_tfs_zero_variance": int(zero_var[tf_pos].sum()),
    }
    return R_p, R_s, R_proc, diag, rows


# ----------------------------------------------------------------------------
# diagnostics of the attention scores against the 1/L null
# ----------------------------------------------------------------------------
def f16_bound(mean: np.ndarray, n: np.ndarray) -> np.ndarray:
    """Worst-case relative error of a float16-accumulated mean: each per-cell increment is
    rounded to the float16 grid of the running sum, so |err| / mean <= n * ulp(S) / (2 S)."""
    S = mean * n
    ulp = np.spacing(S.astype(np.float16)).astype(np.float64)
    return n * ulp / (2.0 * np.maximum(S, 1e-300))


def _bound_summary(b: np.ndarray) -> dict:
    if b.size == 0:
        return {"n_edges": 0}
    return {"n_edges": int(b.size), "median": float(np.median(b)), "q95": float(np.quantile(b, 0.95)),
            "frac_gt_0.05": float((b > 0.05).mean()), "frac_gt_0.25": float((b > 0.25).mean())}


def _bin_label(lo, hi) -> str:
    return f"{lo}" if lo == hi else (f">={lo}" if hi is None else f"{lo}-{hi}")


def float16_breakdown(cols: dict, n: np.ndarray, masks: dict) -> dict:
    out = {"definition": "n * ulp_float16(S) / (2 S) per edge (worst case, all roundings same sign); "
                         "S = stored float16 sum, n = co-observing cells"}
    for v in ("attn", "attn_T"):
        mean = cols[f"scgpt_{v}"]
        ok = np.isfinite(mean)
        b = f16_bound(mean[ok], n[ok])
        nn = n[ok]
        by_bin = {}
        for lo, hi in COOBS_BINS:
            sel = (nn >= lo) & (nn <= (hi if hi is not None else np.iinfo(np.int64).max))
            by_bin[_bin_label(lo, hi)] = _bound_summary(b[sel])
        full = np.full(len(mean), np.nan)
        full[ok] = b
        topk = {name: _bound_summary(full[m & ok]) for name, m in masks.items()
                if name.split("|")[0] in (f"scgpt_{v}", f"scgpt_{v}_norm")}
        out[v] = {"all_finite": _bound_summary(b), "by_coobs_bin": by_bin, "within_topk": topk}
    return out


def scgpt_rowsum_by_frequency(tissue: str) -> dict:
    """Empirical check of the float16 accumulation on the full stored scGPT matrix.

    Per cell a query row sums to 1 over its keys, so for every gene g with
    count[g, g] > 0 (cells in which g is a token): sum_j stored_sum[g, j] / count[g, g]
    = 1 up to the accumulated rounding error.  Reported by gene frequency
    (count[g, g]), with the share of all stored attention mass that sits in
    entries co-observed in >= 500 cells.
    """
    ap = C.attention_paths("scgpt", tissue)
    S = np.load(ap["scores"], mmap_mode="r")
    N = np.load(ap["counts"], mmap_mode="r")
    n = S.shape[0]
    rowsum = np.zeros(n)
    total = mass_ge500 = 0.0
    for a in range(0, n, 512):
        s_ = np.asarray(S[a:a + 512]).astype(np.float64)
        c_ = np.asarray(N[a:a + 512])
        rowsum[a:a + 512] = s_.sum(axis=1)
        total += float(s_.sum())
        mass_ge500 += float(s_[c_ >= 500].sum())
    cnt = np.asarray(N[np.arange(n), np.arange(n)]).astype(np.int64)
    ok = cnt > 0
    ratio, cnt = rowsum[ok] / cnt[ok], cnt[ok]

    def summ(r):
        if r.size == 0:
            return {"n_genes": 0}
        return {"n_genes": int(r.size), "min": float(r.min()), "median": float(np.median(r)),
                "max": float(r.max()), "max_abs_dev_from_1": float(np.abs(r - 1).max())}

    by_bin = {}
    for lo, hi in GENE_FREQ_BINS:
        sel = (cnt >= lo) & (cnt <= (hi if hi is not None else np.iinfo(np.int64).max))
        by_bin[_bin_label(lo, hi)] = summ(ratio[sel])
    return {"definition": "sum_j stored_sum[g,j] / count[g,g] over all genes of the stored matrix "
                          "(1 = no accumulated error in the row's attention mass)",
            "all_genes": summ(ratio), "by_gene_cell_count": by_bin,
            "frac_stored_mass_in_entries_coobs_ge500": mass_ge500 / total if total > 0 else None}


def min_coobs_diagnostics(model: str, cols: dict, n: np.ndarray, e_tf: np.ndarray, n_tfs: int,
                          masks: dict, k: int) -> dict:
    """Top-k sets restricted to edges with >= MIN_COOBS co-observing cells vs the unfiltered sets."""
    u = cols[f"{model}_attn_uniform"]
    eligible = n >= MIN_COOBS
    mu = masks[f"{model}_attn_uniform{MC_TAG}"]
    out = {"min_coobs": MIN_COOBS, "coobs_column": f"{model}_coobs",
           "rule": (f"r2.predicted_sets.topk_per_tf(df, col, k={k}, min_coobs={MIN_COOBS}, "
                    f"coobs='{model}_coobs'): edges with fewer co-observing cells are not predicted"),
           "variants": {}}
    for v in ATTN_VARIANTS:
        c = f"{model}_{v}"
        x = cols[c]
        fin = np.isfinite(x)
        el = fin & eligible
        nel = np.bincount(e_tf, weights=el.astype(float), minlength=n_tfs)
        elig_tf = nel >= k
        m0, m1 = masks[c], masks[c + MC_TAG]
        assert not (m1 & ~el).any()
        retained = float((m0 & m1).sum() / m0.sum()) if m0.any() else None
        assert retained is None or np.isclose(retained, (m0 & eligible).sum() / m0.sum())
        inter_u = np.bincount(e_tf, weights=(m1 & mu).astype(float), minlength=n_tfs)
        out["variants"][v] = {
            "n_edges_eligible": int(el.sum()),
            "frac_finite_edges_eligible": float(el.sum() / fin.sum()) if fin.any() else None,
            "n_tfs_any_eligible": int((nel > 0).sum()), f"n_tfs_ge{k}_eligible": int(elig_tf.sum()),
            "n_predicted": int(m1.sum()), "n_predicted_unfiltered": int(m0.sum()),
            "frac_unfiltered_topk_retained": retained,
            f"top{k}_overlap_with_uniform_median_over_tfs": (
                None if v == "attn_uniform" or not elig_tf.any() else float(np.median(inter_u[elig_tf] / k))),
            f"top{k}_overlap_random_expectation_median_over_tfs": (
                float(np.median(k / nel[elig_tf])) if elig_tf.any() else None),
            "spearman_with_uniform_eligible": None if v == "attn_uniform" else sp(x[el], u[el]),
            "spearman_with_coobs_eligible": sp(x[el], n[el]),
            f"top{k}_coobs_quantiles": quantiles(n[m1]),
            f"top{k}_frac_coobs_lt20": float((n[m1] < 20).mean()) if m1.any() else None,
        }
    return out


def null_diagnostics(model: str, cols: dict, n: np.ndarray, e_tf: np.ndarray, n_tfs: int,
                     masks: dict, k: int) -> dict:
    u = cols[f"{model}_attn_uniform"]
    fin = np.isfinite(cols[f"{model}_attn"])
    nfin = np.bincount(e_tf, weights=fin.astype(float), minlength=n_tfs)
    elig = nfin >= k
    mu = masks[f"{model}_attn_uniform"]

    def overlap(m):  # per-TF |top-k(variant) & top-k(uniform)| / k, median over TFs with >= k finite
        inter = np.bincount(e_tf, weights=(m & mu).astype(float), minlength=n_tfs)
        return float(np.median(inter[elig] / k)) if elig.any() else None

    out = {
        "spearman": {
            "attn~uniform": sp(cols[f"{model}_attn"], u),
            "attn_T~uniform": sp(cols[f"{model}_attn_T"], u),
            "attn_norm~uniform": sp(cols[f"{model}_attn_norm"], u),
            "attn_T_norm~uniform": sp(cols[f"{model}_attn_T_norm"], u),
            "attn_norm~attn": sp(cols[f"{model}_attn_norm"], cols[f"{model}_attn"]),
            "attn~coobs": sp(cols[f"{model}_attn"], n),
            "attn_T~coobs": sp(cols[f"{model}_attn_T"], n),
            "attn_norm~coobs": sp(cols[f"{model}_attn_norm"], n),
            "attn_T_norm~coobs": sp(cols[f"{model}_attn_T_norm"], n),
            "uniform~coobs": sp(u, n),
        },
        f"top{k}_overlap_with_uniform_median_over_tfs": {
            v: overlap(masks[f"{model}_{v}"]) for v in ("attn", "attn_T", "attn_norm", "attn_T_norm")},
        f"top{k}_overlap_random_expectation_median_over_tfs": float(np.median(k / nfin[elig])) if elig.any() else None,
        f"n_tfs_ge{k}_finite": int(elig.sum()),
        f"top{k}_frac_coobs_eq1": {v: float((n[masks[f"{model}_{v}"]] == 1).mean()) for v in ATTN_VARIANTS},
        f"top{k}_frac_coobs_le4": {v: float((n[masks[f"{model}_{v}"]] <= 4).mean()) for v in ATTN_VARIANTS},
        "all_finite_frac_coobs_eq1": float((n[fin] == 1).mean()),
        "all_finite_frac_coobs_le4": float((n[fin] <= 4).mean()),
    }
    return out


# ----------------------------------------------------------------------------
# main per tissue
# ----------------------------------------------------------------------------
def build_tissue(tissue: str) -> dict:
    t0 = time.time()
    panel, tfs, tf_pos, e_tf, e_tg = load_universe(tissue)
    n_edges = len(e_tf)
    log(f"[{tissue}] {len(tfs)} TFs x {len(panel)} panel genes -> {n_edges} edges")
    tfp = tf_pos[e_tf]

    cols: dict[str, np.ndarray] = {}
    coobs: dict[str, np.ndarray] = {}
    att_diag, prov = {}, {}
    for model, idx_col in (("scgpt", "scgpt_idx"), ("geneformer", "geneformer_idx")):
        log(f"  [{tissue}] reading {model} attention block")
        idx = panel[idx_col].to_numpy()
        M, Nb, d = attention_block(model, tissue, idx)
        log(f"  [{tissue}] reconstructing {model} extraction cells and token sets")
        info, U, L = extraction_tokens(tissue, model, panel, Nb)
        assert np.allclose(U, U.T, equal_nan=True)
        prov[model] = info
        cols[f"{model}_attn"] = M[tfp, e_tg]
        cols[f"{model}_attn_T"] = M[e_tg, tfp]
        cols[f"{model}_attn_uniform"] = U[tfp, e_tg]
        cols[f"{model}_attn_norm"] = cols[f"{model}_attn"] / cols[f"{model}_attn_uniform"]
        cols[f"{model}_attn_T_norm"] = cols[f"{model}_attn_T"] / cols[f"{model}_attn_uniform"]
        assert np.array_equal(np.isnan(cols[f"{model}_attn"]), np.isnan(cols[f"{model}_attn_uniform"]))
        coobs[f"{model}_coobs"] = Nb[tfp, e_tg]
        n = coobs[f"{model}_coobs"]
        d["universe_coobs_quantiles"] = quantiles(n)
        d["universe_frac_coobs_ge1"] = float((n >= 1).mean())
        d["universe_frac_coobs_ge5"] = float((n >= 5).mean())
        d["universe_frac_coobs_ge20"] = float((n >= 20).mean())
        d["row_mass_check_tf_rows"] = row_mass_check(model, tissue, idx[tf_pos], L)
        att_diag[model] = d

    R_p, R_s, R_proc, corr_diag, corr_rows = correlation_scores(tissue, panel, tf_pos)
    cols["pearson"] = R_p[e_tf, e_tg]          # R_* rows are TFs in tfs order
    cols["pearson_abs"] = np.abs(cols["pearson"])
    cols["spearman"] = R_s[e_tf, e_tg]
    cols["spearman_abs"] = np.abs(cols["spearman"])
    proc_pearson = R_proc[e_tf, e_tg]
    cols["random"] = rng_for(tissue, STREAM_RANDOM).uniform(0.0, 1.0, size=n_edges)

    # ---- write ----
    od = C.ensure_dir(C.OUT / STAGE / tissue)
    tf_names = np.array(tfs, dtype=object)[e_tf]
    tg_names = panel.gene.to_numpy(dtype=object)[e_tg]
    arrays = [pa.array(tf_names, pa.string()), pa.array(tg_names, pa.string())]
    arrays += [pa.array(cols[c].astype(np.float32), pa.float32()) for c in SCORE_COLS]
    pq.write_table(pa.Table.from_arrays(arrays, names=["tf", "target"] + SCORE_COLS),
                   od / "scores.parquet", compression="zstd")
    pq.write_table(pa.Table.from_arrays(
        [pa.array(tf_names, pa.string()), pa.array(tg_names, pa.string()),
         pa.array(coobs["scgpt_coobs"].astype(np.int32)), pa.array(coobs["geneformer_coobs"].astype(np.int32))],
        names=["tf", "target", "scgpt_coobs", "geneformer_coobs"]), od / "coobs.parquet", compression="zstd")

    # ---- diagnostics on the stored float32 values (what downstream stages read) ----
    # every number below uses ``cols32`` (float32 rounded, held in float64)
    k = C.PRESPECIFIED["topk_per_tf"]
    cols32 = {c: cols[c].astype(np.float32).astype(np.float64) for c in SCORE_COLS}
    df = pd.DataFrame({"tf": tf_names, "target": tg_names})
    for c in SCORE_COLS:
        df[c] = cols[c].astype(np.float32)
    for model in ("scgpt", "geneformer"):
        df[f"{model}_coobs"] = coobs[f"{model}_coobs"].astype(np.int32)
    masks = {}
    for m in ("scgpt", "geneformer"):
        for v in ATTN_VARIANTS:
            c = f"{m}_{v}"
            masks[c] = topk_per_tf(df, c, k).to_numpy()
            masks[c + MC_TAG] = topk_per_tf(df, c, k, min_coobs=MIN_COOBS, coobs=fm_coobs_column(c)).to_numpy()
    for model in ("scgpt", "geneformer"):
        att_diag[model]["uniform_null"] = null_diagnostics(model, cols32, coobs[f"{model}_coobs"],
                                                           e_tf, len(tfs), masks, k)
        att_diag[model]["min_coobs_sensitivity"] = min_coobs_diagnostics(model, cols32, coobs[f"{model}_coobs"],
                                                                         e_tf, len(tfs), masks, k)
    att_diag["scgpt"]["float16_rel_error_bound"] = float16_breakdown(cols32, coobs["scgpt_coobs"], masks)
    att_diag["scgpt"]["float16_rel_error_bound"]["note"] = (
        "conservative worst-case ceiling; see float16_rowsum_check_by_gene_frequency for the realised "
        "error of the attention mass")
    log(f"  [{tissue}] float16 row-sum check on the full scGPT matrix")
    att_diag["scgpt"]["float16_rowsum_check_by_gene_frequency"] = scgpt_rowsum_by_frequency(tissue)

    coverage = {c: float(np.isfinite(cols32[c]).mean()) for c in SCORE_COLS}
    per_tf = {}
    for c in SCORE_COLS:
        nfin = np.bincount(e_tf, weights=np.isfinite(cols32[c]).astype(float), minlength=len(tfs))
        per_tf[c] = {"n_tfs_any_finite": int((nfin > 0).sum()), f"n_tfs_ge{k}_finite": int((nfin >= k).sum()),
                     "median_finite_targets_per_tf": float(np.median(nfin))}

    cross = {
        "scgpt_attn~geneformer_attn": sp(cols32["scgpt_attn"], cols32["geneformer_attn"]),
        "scgpt_attn_norm~geneformer_attn_norm": sp(cols32["scgpt_attn_norm"], cols32["geneformer_attn_norm"]),
        "scgpt_attn_uniform~geneformer_attn_uniform": sp(cols32["scgpt_attn_uniform"], cols32["geneformer_attn_uniform"]),
        "scgpt_attn~scgpt_attn_T": sp(cols32["scgpt_attn"], cols32["scgpt_attn_T"]),
        "geneformer_attn~geneformer_attn_T": sp(cols32["geneformer_attn"], cols32["geneformer_attn_T"]),
        "scgpt_attn~scgpt_attn_uniform": sp(cols32["scgpt_attn"], cols32["scgpt_attn_uniform"]),
        "geneformer_attn~geneformer_attn_uniform": sp(cols32["geneformer_attn"], cols32["geneformer_attn_uniform"]),
        "pearson~spearman": sp(cols32["pearson"], cols32["spearman"]),
        "scgpt_attn~pearson_abs": sp(cols32["scgpt_attn"], cols32["pearson_abs"]),
        "geneformer_attn~pearson_abs": sp(cols32["geneformer_attn"], cols32["pearson_abs"]),
        "scgpt_attn_norm~pearson_abs": sp(cols32["scgpt_attn_norm"], cols32["pearson_abs"]),
        "geneformer_attn_norm~pearson_abs": sp(cols32["geneformer_attn_norm"], cols32["pearson_abs"]),
        "scgpt_attn~scgpt_coobs": sp(cols32["scgpt_attn"], coobs["scgpt_coobs"]),
        "geneformer_attn~geneformer_coobs": sp(cols32["geneformer_attn"], coobs["geneformer_coobs"]),
        "pearson(TS log-norm X)~pearson(processed double-log X)": sp(cols32["pearson"],
                                                                     proc_pearson.astype(np.float32)),
    }

    meta = {
        "stage": STAGE, "tissue": tissue, "created": _dt.datetime.now().isoformat(timespec="seconds"),
        "script": str(Path(__file__).resolve()),
        "random_seed": C.PRESPECIFIED["random_seed"],
        "topk_rule": {"k": k, "tie_break": C.PRESPECIFIED["topk_tie_break"],
                      "tie_seed": C.PRESPECIFIED["topk_tie_seed"], "implementation": "r2.predicted_sets.topk_per_tf"},
        "predicted_set_definitions": {
            "primary": "topk_per_tf(scores, <col>, k) for every score column",
            "min_coobs_sensitivity": {
                "columns": [f"{m}_{v}" for m in ("scgpt", "geneformer") for v in ATTN_VARIANTS],
                "min_coobs": MIN_COOBS, "prespecified_key": "fm_min_coobs_sensitivity",
                "coobs_source": "coobs.parquet, column r2.predicted_sets.fm_coobs_column(<col>)",
                "call": f"topk_per_tf(scores.merge(coobs), <col>, k, min_coobs={MIN_COOBS}, "
                        "coobs=fm_coobs_column(<col>))"}},
        "diagnostics_values": "stored float32 scores (as written to scores.parquet)",
        "n_tfs": len(tfs), "n_targets": int(len(panel)), "n_edges": int(n_edges),
        "edge_order": "TFs in tfs.txt order x targets in panel.tsv order, tf != target",
        "score_definitions": {
            "scgpt_attn": "sum[tf,g]/count[tf,g] of scGPT layer/head-mean attention, tf=query row, g=key column; NaN if count==0",
            "scgpt_attn_T": "sum[g,tf]/count[g,tf] (target as query, TF as key); NaN if count==0",
            "geneformer_attn": "stored mean[tf,g] of Geneformer V2-104M layer/head-mean attention (tf=query); NaN if count==0",
            "geneformer_attn_T": "stored mean[g,tf]; NaN if count==0",
            "<model>_attn_uniform": ("mean over co-observing extraction cells of 1/L_c, L_c = softmax length "
                                     "(scGPT: gene tokens; Geneformer: gene tokens + <cls>); NaN if count==0"),
            "<model>_attn_norm": "<model>_attn / <model>_attn_uniform",
            "<model>_attn_T_norm": "<model>_attn_T / <model>_attn_uniform",
            "pearson/spearman": ("signed correlation across the GRNBoost2 cell sample of TS log1p(CP10k, DecontX) "
                                 "expression; NaN if zero variance"),
            "pearson_abs/spearman_abs": "absolute value of the signed correlation",
            "random": f"uniform(0,1), default_rng([{C.PRESPECIFIED['random_seed']}, {C.TISSUES.index(tissue)}, {STREAM_RANDOM}])",
        },
        "coverage": coverage,
        "coverage_per_tf": per_tf,
        "attention_diagnostics": att_diag,
        "correlation_cells": corr_diag,
        "cross_method_spearman": cross,
        "extraction_cells": prov,
        "source_files": {
            "panel": file_info(C.OUT / "universe" / tissue / "panel.tsv"),
            "tfs": file_info(C.OUT / "universe" / tissue / "tfs.txt"),
            "scgpt_scores": file_info(C.attention_paths("scgpt", tissue)["scores"]),
            "scgpt_counts": file_info(C.attention_paths("scgpt", tissue)["counts"]),
            "geneformer_scores": file_info(C.attention_paths("geneformer", tissue)["scores"]),
            "geneformer_counts": file_info(C.attention_paths("geneformer", tissue)["counts"]),
            "geneformer_gene_names": file_info(C.attention_paths("geneformer", tissue)["gene_names"]),
            "processed_h5ad": file_info(C.SCGPT_PROCESSED_H5AD[tissue]),
            "ts_source_h5ad": file_info(C.TS_SOURCE_H5AD[tissue]),
            "scgpt_config": file_info(SCGPT_CONFIG[tissue]),
            "geneformer_script": file_info(GENEFORMER_SCRIPT),
            "geneformer_token_dict": file_info(C.GF_DICT_DIR / "token_dictionary_gc104M.pkl"),
            "geneformer_median_dict": file_info(C.GF_DICT_DIR / "gene_median_dictionary_gc104M.pkl"),
            "expression_cells": file_info(expression_cells_path(tissue), md5=True),
        },
        "outputs": {"scores": file_info(od / "scores.parquet"), "coobs": file_info(od / "coobs.parquet")},
        "runtime_s": round(time.time() - t0, 1),
    }
    C.write_json(meta, od / "meta.json")
    log(f"[{tissue}] done in {meta['runtime_s']} s")
    return meta


# ----------------------------------------------------------------------------
# summary tables
# ----------------------------------------------------------------------------
def _merge_write(rows: list[dict], path: Path, tissues: list[str], sort_cols: list[str]) -> None:
    new = pd.DataFrame(rows)
    if set(tissues) != set(C.TISSUES) and path.exists():   # partial re-run: merge
        old = pd.read_csv(path)
        new = pd.concat([old[~old.tissue.isin(tissues)], new], ignore_index=True)
    new["tissue"] = pd.Categorical(new["tissue"], categories=list(C.TISSUES), ordered=True)
    new.sort_values(sort_cols, kind="stable").to_csv(path, index=False)


def load_stored_meta(tissue: str) -> dict:
    """The meta dictionary ``build_tissue`` returned for ``tissue``, read back from its meta.json.  The tables
    need only this dictionary; a stored score file that is present must have the size meta.json records
    (the score files themselves are read by later stages, e.g. held-out evaluation, not here)."""
    import json
    od = C.OUT / STAGE / tissue
    meta = json.loads((od / "meta.json").read_text())
    for name, rec in (meta.get("outputs") or {}).items():
        f = od / Path(str(rec.get("path", f"{name}.parquet"))).name
        if f.exists() and isinstance(rec.get("bytes"), int) and f.stat().st_size != rec["bytes"]:
            raise ValueError(f"{f}: {f.stat().st_size} bytes, meta.json records {rec['bytes']}")
    return meta


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tissues", default=",".join(C.TISSUES))
    ap.add_argument("--tables-only", action="store_true",
                    help="rebuild the fm_* tables from the stored meta.json files (no raw data read)")
    args = ap.parse_args(argv)
    tissues = [t.strip() for t in args.tissues.split(",") if t.strip()]
    metas = [load_stored_meta(t) if args.tables_only else build_tissue(t) for t in tissues]
    write_tables(metas, tissues)


def write_tables(metas: list[dict], tissues: list[str]) -> None:
    """The six manuscript tables (data/r2/tables/fm_*.csv) from the per-tissue meta dictionaries."""
    k = C.PRESPECIFIED["topk_per_tf"]

    rows, cells, nulls, f16, f16rows, mcs = [], [], [], [], [], []
    for m in metas:
        ad = m["attention_diagnostics"]
        r = {"tissue": m["tissue"], "n_tfs": m["n_tfs"], "n_targets": m["n_targets"], "n_edges": m["n_edges"]}
        r.update({f"coverage_{c}": round(v, 6) for c, v in m["coverage"].items()})
        for model in ("scgpt", "geneformer"):
            un = ad[model]["uniform_null"]
            r[f"frac_{model}_coobs_ge5"] = round(ad[model]["universe_frac_coobs_ge5"], 6)
            r[f"median_{model}_coobs"] = ad[model]["universe_coobs_quantiles"]["0.5"]
            r[f"median_{model}_softmax_length"] = m["extraction_cells"][model]["tokens"]["softmax_length_L_quantiles"]["0.5"]
            r[f"spearman_{model}_attn_uniform"] = round(un["spearman"]["attn~uniform"], 6)
            r[f"spearman_{model}_attn_T_uniform"] = round(un["spearman"]["attn_T~uniform"], 6)
            r[f"top{k}_overlap_{model}_attn_uniform"] = un[f"top{k}_overlap_with_uniform_median_over_tfs"]["attn"]
            r[f"top{k}_overlap_{model}_attn_norm_uniform"] = un[f"top{k}_overlap_with_uniform_median_over_tfs"]["attn_norm"]
            r[f"top{k}_overlap_{model}_random_expectation"] = round(un[f"top{k}_overlap_random_expectation_median_over_tfs"], 6)
        for c in ("scgpt_attn", "geneformer_attn", "pearson"):
            r[f"n_tfs_any_finite_{c}"] = m["coverage_per_tf"][c]["n_tfs_any_finite"]
            r[f"n_tfs_ge{k}_finite_{c}"] = m["coverage_per_tf"][c][f"n_tfs_ge{k}_finite"]
        r["corr_n_cells"] = m["correlation_cells"]["n_cells_used"]
        r["corr_n_donors"] = m["correlation_cells"]["n_donors"]
        r["corr_cells_match_grnboost2"] = m["correlation_cells"]["cells_match_grnboost2_rule"]
        rows.append(r)

        for model in ("scgpt", "geneformer"):
            e = m["extraction_cells"][model]
            ver = e.get("verification_coobs_panel_block")
            if model == "geneformer":
                ver = ver.get(str(e["max_cells_used"]))
            tok = e["tokens"]
            cells.append({"tissue": m["tissue"], "model": model, "used_by": f"{model}_attn*",
                          "n_cells": e.get("n_cells"), "n_donors": e.get("n_donors"),
                          "donors": ";".join(f"{a}:{b}" for a, b in (e.get("donors") or {}).items()),
                          "assay_method": ";".join(f"{a}:{b}" for a, b in (e.get("assay_method") or {}).items()),
                          "cell_selection": e.get("cell_selection"),
                          "coobs_reproduced_exact_frac": ver["exact_match_frac"],
                          "n_special_tokens": tok["n_special_tokens"],
                          "median_softmax_length": tok["softmax_length_L_quantiles"]["0.5"],
                          "frac_cells_at_gene_token_cap": round(tok["frac_cells_at_gene_token_cap"], 6)})
        cc = m["correlation_cells"]
        cells.append({"tissue": m["tissue"], "model": "expression",
                      "used_by": "pearson, spearman, grnboost2, genie3",
                      "n_cells": cc["n_cells_used"], "n_donors": cc["n_donors"],
                      "donors": ";".join(f"{a}:{b}" for a, b in cc["donors"].items()),
                      "assay_method": ";".join(f"{a}:{b}" for a, b in cc["assay_method"].items()),
                      "cell_selection": (f"outputs/r2/grnboost2/{m['tissue']}/cells.txt = {cc['cells_rule']} "
                                         f"over {cc['n_cells_available']} cells (matches: {cc['cells_match_grnboost2_rule']})"),
                      "coobs_reproduced_exact_frac": None, "n_special_tokens": None,
                      "median_softmax_length": None, "frac_cells_at_gene_token_cap": None})

        for model in ("scgpt", "geneformer"):
            un = ad[model]["uniform_null"]
            for v in ATTN_VARIANTS:
                nulls.append({
                    "tissue": m["tissue"], "model": model, "variant": v,
                    "coverage": round(m["coverage"][f"{model}_{v}"], 6),
                    "n_tfs_ge_k_finite": m["coverage_per_tf"][f"{model}_{v}"][f"n_tfs_ge{k}_finite"],
                    "spearman_with_uniform": (None if v == "attn_uniform" else
                                              un["spearman"].get(f"{v}~uniform")),
                    "spearman_with_coobs": un["spearman"].get(f"{v}~coobs") if v != "attn_uniform"
                    else un["spearman"]["uniform~coobs"],
                    "k": k,
                    "topk_overlap_with_uniform_median": un[f"top{k}_overlap_with_uniform_median_over_tfs"].get(v),
                    "topk_overlap_random_expectation": un[f"top{k}_overlap_random_expectation_median_over_tfs"],
                    "topk_frac_coobs_eq1": un[f"top{k}_frac_coobs_eq1"][v],
                    "topk_frac_coobs_le4": un[f"top{k}_frac_coobs_le4"][v],
                    "all_finite_frac_coobs_eq1": un["all_finite_frac_coobs_eq1"],
                    "all_finite_frac_coobs_le4": un["all_finite_frac_coobs_le4"],
                })
        for model in ("scgpt", "geneformer"):
            mc = ad[model]["min_coobs_sensitivity"]
            for v, d in mc["variants"].items():
                mcs.append({
                    "tissue": m["tissue"], "model": model, "variant": v, "min_coobs": mc["min_coobs"], "k": k,
                    "n_edges_eligible": d["n_edges_eligible"],
                    "frac_finite_edges_eligible": d["frac_finite_edges_eligible"],
                    "n_tfs_ge_k_eligible": d[f"n_tfs_ge{k}_eligible"],
                    "n_tfs_ge_k_finite_unfiltered": m["coverage_per_tf"][f"{model}_{v}"][f"n_tfs_ge{k}_finite"],
                    "n_predicted": d["n_predicted"], "n_predicted_unfiltered": d["n_predicted_unfiltered"],
                    "frac_unfiltered_topk_retained": d["frac_unfiltered_topk_retained"],
                    "topk_frac_coobs_eq1_unfiltered": ad[model]["uniform_null"][f"top{k}_frac_coobs_eq1"][v],
                    "topk_median_coobs": d[f"top{k}_coobs_quantiles"].get("0.5"),
                    "topk_frac_coobs_lt20": d[f"top{k}_frac_coobs_lt20"],
                    "topk_overlap_with_uniform_median": d[f"top{k}_overlap_with_uniform_median_over_tfs"],
                    "topk_overlap_with_uniform_median_unfiltered": (
                        None if v == "attn_uniform"
                        else ad[model]["uniform_null"][f"top{k}_overlap_with_uniform_median_over_tfs"][v]),
                    "topk_overlap_random_expectation": d[f"top{k}_overlap_random_expectation_median_over_tfs"],
                    "spearman_with_uniform_eligible": d["spearman_with_uniform_eligible"],
                    "spearman_with_coobs_eligible": d["spearman_with_coobs_eligible"],
                })
        rs = ad["scgpt"]["float16_rowsum_check_by_gene_frequency"]
        for label, d in [("all", rs["all_genes"])] + list(rs["by_gene_cell_count"].items()):
            f16rows.append({"tissue": m["tissue"], "gene_cell_count": label, "n_genes": d["n_genes"],
                            "ratio_min": d.get("min"), "ratio_median": d.get("median"), "ratio_max": d.get("max"),
                            "max_abs_dev_from_1": d.get("max_abs_dev_from_1"),
                            "frac_stored_mass_in_entries_coobs_ge500": rs["frac_stored_mass_in_entries_coobs_ge500"]})
        fb = ad["scgpt"]["float16_rel_error_bound"]
        for v in ("attn", "attn_T"):
            entries = [("all", fb[v]["all_finite"])] + list(fb[v]["by_coobs_bin"].items())
            entries += [(f"top{k}:{name}", s) for name, s in fb[v]["within_topk"].items()]
            for label, s in entries:
                f16.append({"tissue": m["tissue"], "variant": f"scgpt_{v}", "subset": label,
                            "n_edges": s["n_edges"], "median": s.get("median"), "q95": s.get("q95"),
                            "frac_gt_0.05": s.get("frac_gt_0.05"), "frac_gt_0.25": s.get("frac_gt_0.25")})

    C.ensure_dir(C.TABLES)
    _merge_write(rows, C.TABLES / "fm_universe_summary.csv", tissues, ["tissue"])
    _merge_write(cells, C.TABLES / "fm_extraction_cells.csv", tissues, ["tissue", "model"])
    _merge_write(nulls, C.TABLES / "fm_attention_null.csv", tissues, ["tissue"])
    _merge_write(f16, C.TABLES / "fm_float16_error.csv", tissues, ["tissue"])
    _merge_write(f16rows, C.TABLES / "fm_float16_rowsum_check.csv", tissues, ["tissue"])
    _merge_write(mcs, C.TABLES / "fm_min_coobs_sensitivity.csv", tissues, ["tissue"])
    log("wrote fm_universe_summary.csv, fm_extraction_cells.csv, fm_attention_null.csv, fm_float16_error.csv, "
        "fm_float16_rowsum_check.csv, fm_min_coobs_sensitivity.csv")


if __name__ == "__main__":
    main()
