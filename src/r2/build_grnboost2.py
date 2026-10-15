"""Expression-based GRN baselines on the tissue context universes (stage D1, GRNBoost2/GENIE3).

For each tissue t, GRNBoost2 (arboreto 0.1.6, ``arboreto.core.create_graph``
with arboreto's GRNBoost2 profile: ``GradientBoostingRegressor`` with
``SGBM_KWARGS`` and out-of-bag early stopping, window 25) is run with

* regulators  = TF_t  (``outputs/r2/universe/<t>/tfs.txt``),
* candidate targets = Panel_t (``outputs/r2/universe/<t>/panel.tsv``), i.e. the
  expression matrix is restricted to the panel genes so that every returned
  link lies in the tissue context universe U_ctx(t) = TF_t x Panel_t (tf != g),
* cells = a seeded uniform random subsample (without replacement) of
  ``--n-cells`` cells (``PRESPECIFIED['expr_n_cells']`` = 3,000) from the processed Tabula Sapiens
  h5ad ``C.SCGPT_PROCESSED_H5AD[t]``, i.e. the tissue population from which the
  foundation-model attention was extracted.  The attention itself used other
  cells (scGPT: the first 2,000 cells of this file; Geneformer:
  ``default_rng(42)`` samples of 1,200 / 2,000 / 1,200 cells of the source
  h5ad; see ``data/r2/tables/fm_extraction_cells.csv``).  The sample is written
  to ``cells.txt``; a re-run asserts that it is unchanged,
* expression = Tabula Sapiens' single log-normalisation, i.e. the distributed
  ``X`` = log1p(1e4 * decontX / S_c), with S_c Tabula Sapiens' per-cell size
  factor.  S_c is approximately, not exactly, the cell's total decontX count
  over all genes: the ratio S_c / sum_g decontX (recorded per run in
  meta.json) is between 1.000 and about 1.02.  The values are read for the
  sampled cells from the source h5ad ``C.TS_SOURCE_H5AD[t]``.

Why not the processed h5ad's ``X``?  The processed file's ``X`` is *doubly*
log-normalised: the project preprocessing applied ``normalize_total(1e4)`` and
``log1p`` on top of Tabula Sapiens' already log-normalised ``X``
(processed X = log1p(1e4 * y / sum_g y), y = TS X).  That transform is
monotone within a cell (so it does not affect scGPT's per-cell value binning)
but it rescales each cell differently and therefore distorts across-cell
co-variation, which is what GRNBoost2/GENIE3 model.  The script verifies this
relationship on the sampled cells and records the discrepancy in meta.json.

arboreto 0.1.6 needs two runtime fixes on the current environment.  Both live
in ``arboreto_compat.py`` and are installed in the main process *and* on every
dask worker (a WorkerPlugin, verified on each worker before and after the
computation):

1. scikit-learn >= 1.3 changed ``oob_improvement_`` so that it compares losses
   on two different random OOB subsets.  With that definition arboreto's early
   stopping stops after about 25 rounds.  ``SameOOBEarlyStopMonitor`` restores
   the pre-1.3 same-subset improvement and keeps arboreto's window rule
   unchanged.
2. A shim for dask-expr's ``from_delayed([])``, which now raises.

Boosting rounds per target come from arboreto's own meta output
(``include_meta=True``) and are recorded.  The run fails unless all three hold:

* every fitted target has a meta row, i.e. no silent arboreto retry fallback;
* per-target importance sums equal the number of rounds;
* the median number of rounds over targets detected in >= 10% of the sampled
  cells exceeds 100.

Sensitivity variant (``--min-detect-frac 0.01``,
``PRESPECIFIED['grn_detection_filter_sensitivity']``, decision-log items 6 and 9):
the SCENIC-style gene filter (keep genes, TFs included, detected in >= 1% of
the sampled cells) applied before the regressions; written as
``<method>_detect1pct.parquet`` for GRNBoost2 (all tissues) and GENIE3 (immune).
Pairs involving a filtered gene are "not predicted".  The primary runs of both
methods use all panel genes.

Methods disclosure (decision-log item 6; ``validate-es``):
* "arboreto 0.1.6 on scikit-learn 1.5.1 with a same-subset early-stopping
  monitor" (``PRESPECIFIED['grnboost2_early_stopping']``).  The number of
  boosting rounds at which early stopping fires depends on the scikit-learn
  version: arboreto's original monitor under scikit-learn 1.2.2 gives the same
  rounds for 31 of the 36 ``validate-es`` targets (median relative difference 0,
  maximum 28%, importance Spearman median 0.999; ``early_stop_validation.csv``),
  because the fitted trees differ slightly between the two versions (leaf values
  from the first tree on), not because of the monitor: under 1.2.2 the compat
  monitor reproduces arboreto's original rounds for all 36.
* Targets not detected in any sampled cell have an OOB improvement of exactly
  0 and run all 5,000 rounds (arboreto's own behaviour); this costs time only
  and returns no links.

Outputs (``outputs/r2/grnboost2/<t>/``)
  grnboost2.parquet / genie3.parquet  columns tf, target, importance (only the
      pairs returned by the method, i.e. importance > 0; every other pair of
      U_ctx(t) is "not predicted")
  <variant>_rounds.tsv  target, n_cells_detected, frac_detected, n_rounds
      (boosting rounds, or trees for GENIE3), importance_sum, n_links
  <variant>_meta.json  (``meta.json`` is a copy of the primary GRNBoost2 meta);
      includes the boosting rounds per target and their summary
  cells.txt            obs names of the sampled cells (identical for all runs)
  gene_detection.tsv   gene, is_tf, n_cells_detected, frac_detected in the sample
  time_l_<variant>.txt ``/usr/bin/time -l`` output of the whole run (written by
      run_grnboost2.sh).  macOS ``time -l`` reports the rusage of the run *and
      its reaped children*: user/sys CPU include the dask worker processes, and
      "maximum resident set size" is the largest single process (main or one
      worker), not the sum over processes (that is ``peak_rss_total_mib``)
outputs/r2/grnboost2/early_stop_validation.{csv,json}     (``validate-es``)
outputs/r2/grnboost2/summary_rare_targets.csv, summary.json  (``summary``)

Commands (the full sequence is src/r2/run_grnboost2.sh)
  python src/r2/build_grnboost2.py run --tissue immune --method grnboost2
  python src/r2/build_grnboost2.py run --tissue immune --method grnboost2 --min-detect-frac 0.01
  python src/r2/build_grnboost2.py run --tissue immune --method genie3
  python src/r2/build_grnboost2.py run --tissue immune --method genie3 --min-detect-frac 0.01
  python src/r2/build_grnboost2.py validate-es --ref-python <python with scikit-learn 1.2.x>
  python src/r2/build_grnboost2.py table      # -> data/r2/tables/grn_baseline_runtime.csv
  python src/r2/build_grnboost2.py summary    # rare-target and overlap summary
  python src/r2/build_grnboost2.py bench --tissue immune   # per-target timing probe
"""
from __future__ import annotations

import os

# single-threaded numerical libraries in the main process and (inherited) workers
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import platform
import re
import resource
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from collections import Counter
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp

_SRC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_SRC))
from r2 import arboreto_compat as AX  # noqa: E402
from r2 import common as C  # noqa: E402
from r2.predicted_sets import topk_per_tf  # noqa: E402

STAGE = "grnboost2"
OUT_STAGE = C.OUT / STAGE
N_CELLS_DEFAULT = int(C.PRESPECIFIED["expr_n_cells"])   # prespecified cell subsample size (3,000)
N_WORKERS_DEFAULT = 6           # <= 6 dask workers (shared 10-core machine)
# early-stopping guard (GRNBoost2 only): median boosting rounds over targets
# detected in >= ROUNDS_GUARD_MIN_DETECT of the sampled cells must exceed
# ROUNDS_GUARD_MIN_MEDIAN (broken early stopping gives ~25-30)
ROUNDS_GUARD_MIN_DETECT = 0.10
ROUNDS_GUARD_MIN_MEDIAN = 100
DETECT_BINS = [(0.0, 0.01, "lt1pct"), (0.01, 0.10, "1to10pct"), (0.10, 1.01, "ge10pct")]
# "rare" gene: detected in < 1% of the sampled cells (= the sensitivity filter threshold)
RARE_FRAC = float(C.PRESPECIFIED["grn_detection_filter_sensitivity"])
assert AX.EXPECTED_MONITOR.endswith("SameOOBEarlyStopMonitor") and \
    C.PRESPECIFIED["grnboost2_early_stopping"] == "same_subset_oob_monitor"


# ----------------------------------------------------------------------------
# h5ad helpers (h5py, no full load)
# ----------------------------------------------------------------------------


def _strings(ds) -> list[str]:
    return [x.decode() if isinstance(x, bytes) else str(x) for x in ds[:]]


def _obs_names(f: h5py.File) -> list[str]:
    key = f["obs"].attrs.get("_index", "_index")
    return _strings(f["obs"][key])


def _var_names(f: h5py.File) -> list[str]:
    key = f["var"].attrs.get("_index", "_index")
    return _strings(f["var"][key])


def _categorical(f: h5py.File, col: str, rows: np.ndarray) -> list[str]:
    g = f["obs"][col]
    if isinstance(g, h5py.Group):
        cats = _strings(g["categories"])
        codes = g["codes"][:]
        return [cats[c] if c >= 0 else "NA" for c in codes[rows]]
    vals = _strings(g)
    return [vals[r] for r in rows]


def _read_csr_rows(grp: h5py.Group, rows: np.ndarray, n_cols: int, dtype=np.float32) -> sp.csr_matrix:
    """Read the given rows (sorted, ascending) of a CSR matrix stored in an h5ad group."""
    indptr = grp["indptr"][:]
    data_ds, ind_ds = grp["data"], grp["indices"]
    datas, inds, ptr = [], [], [0]
    for r in rows:
        a, b = int(indptr[r]), int(indptr[r + 1])
        datas.append(data_ds[a:b].astype(dtype, copy=False))
        inds.append(ind_ds[a:b])
        ptr.append(ptr[-1] + (b - a))
    data = np.concatenate(datas) if datas else np.zeros(0, dtype)
    ind = np.concatenate(inds) if inds else np.zeros(0, np.int32)
    return sp.csr_matrix((data, ind, np.asarray(ptr, dtype=np.int64)), shape=(len(rows), n_cols))


# ----------------------------------------------------------------------------
# Expression matrix for one tissue
# ----------------------------------------------------------------------------


def load_panel(tissue: str) -> tuple[pd.DataFrame, list[str]]:
    ud = C.OUT / "universe" / tissue
    panel = pd.read_csv(ud / "panel.tsv", sep="\t")
    tfs = [l.strip() for l in open(ud / "tfs.txt") if l.strip()]
    assert panel.gene.is_unique and set(tfs) <= set(panel.gene)
    return panel, tfs


def sample_cells(n_obs: int, n_cells: int, seed: int) -> np.ndarray:
    """Seeded uniform subsample without replacement, returned in ascending order."""
    rng = np.random.default_rng(seed)
    if n_cells >= n_obs:
        return np.arange(n_obs)
    return np.sort(rng.choice(n_obs, size=n_cells, replace=False))


def load_expression(tissue: str, n_cells: int, seed: int):
    """Return (expr, info, tfs, cells): cells x panel-gene single log-normalised
    expression (float32 DataFrame), provenance/check info, TF list, sampled obs names."""
    panel, tfs = load_panel(tissue)
    proc_path = C.SCGPT_PROCESSED_H5AD[tissue]
    src_path = C.TS_SOURCE_H5AD[tissue]
    info: dict = {"processed_h5ad": str(proc_path), "source_h5ad": str(src_path)}

    with h5py.File(proc_path, "r") as g:
        p_obs = _obs_names(g)
        p_var = _var_names(g)
        p_ens = _strings(g["var"]["ensembl_id"])
        n_obs = len(p_obs)
        rows_p = sample_cells(n_obs, n_cells, seed)
        cells = [p_obs[i] for i in rows_p]
        # panel genes -> processed var rows (scgpt_idx) -> Ensembl ids
        pidx = panel.scgpt_idx.to_numpy()
        assert all(C.normalize_symbol(p_var[i]) == gname for i, gname in zip(pidx, panel.gene)), \
            "panel.tsv scgpt_idx does not match processed var order"
        ens = [p_ens[i] for i in pidx]
        # processed X (doubly log-normalised) and decontX counts for the sampled cells
        n_pvar = len(p_var)
        Xp = _read_csr_rows(g["X"], rows_p, n_pvar)[:, pidx].toarray()
        Dp = _read_csr_rows(g["layers"]["decontXcounts"], rows_p, n_pvar)[:, pidx].toarray()
        assay = _categorical(g, "assay", rows_p)
        donor = _categorical(g, "donor_id", rows_p)
        ctype = _categorical(g, "cell_type", rows_p)

    with h5py.File(src_path, "r") as f:
        s_obs = _obs_names(f)
        s_var = _var_names(f)
        pos_obs = {o: i for i, o in enumerate(s_obs)}
        pos_var = {v: i for i, v in enumerate(s_var)}
        rows_s = np.array([pos_obs[c] for c in cells])
        order = np.argsort(rows_s)
        assert np.all(np.diff(rows_s[order]) > 0)
        cols_s = np.array([pos_var[e] for e in ens])
        n_svar = len(s_var)
        Y_full = _read_csr_rows(f["X"], rows_s[order], n_svar)
        D_full = _read_csr_rows(f["layers"]["decontXcounts"], rows_s[order], n_svar, dtype=np.float64)
    inv = np.empty_like(order)
    inv[order] = np.arange(len(order))
    Y_full = Y_full[inv]
    D_full = D_full[inv]
    Y = Y_full[:, cols_s].toarray().astype(np.float32)

    # --- provenance / transform checks on the sampled cells --------------------
    # (1) processed decontX layer == source decontX counts on the panel genes
    info["check_decontx_identical"] = bool(np.array_equal(Dp, D_full[:, cols_s].toarray()))
    # (2) source X is a single log-normalisation on every sampled cell:
    #     expm1(X)/decontX = k_c is constant within the cell; S_c = 1e4 / k_c is
    #     the size factor, compared with the cell's total decontX count
    cvs, k_c = [], []
    for i in range(Y_full.shape[0]):
        a, b = Y_full.indptr[i], Y_full.indptr[i + 1]
        y = Y_full.data[a:b].astype(np.float64)
        d = D_full[i, Y_full.indices[a:b]].toarray().ravel()
        ok = d > 0
        r = np.expm1(y[ok]) / d[ok]
        cvs.append(float(r.std() / r.mean()) if r.size > 1 else 0.0)
        k_c.append(float(np.median(r)) if r.size else np.nan)
    k_c = np.asarray(k_c)
    dsum = np.asarray(D_full.sum(axis=1)).ravel()
    sf_ratio = (1e4 / k_c) / dsum                       # S_c / sum_g decontX
    sf_ratio = sf_ratio[np.isfinite(sf_ratio)]
    info["check_source_X_single_lognorm_max_cv_ratio_all_cells"] = float(np.max(cvs))
    info["check_source_X_size_factor_over_total_decontx_min_median_max"] = [
        float(sf_ratio.min()), float(np.median(sf_ratio)), float(sf_ratio.max())]
    info["check_source_X_rowsum_expm1_median"] = float(np.median(
        np.asarray(Y_full.expm1().sum(axis=1)).ravel()))
    # (3) processed X == log1p(s_c * y) with a per-cell scale s_c = 1e4 / T_c,
    #     T_c ~= sum_g y (double log-normalisation; T_c is the sum over the genes
    #     kept by the earlier min_cells filter, hence ~= not ==).
    Yd = Y.astype(np.float64)
    nz = Yd > 0
    info["check_processed_X_zero_pattern_identical"] = bool(np.array_equal(nz, Xp > 0))
    s_c = np.array([np.median(np.expm1(Xp[i, nz[i]].astype(np.float64)) / Yd[i, nz[i]])
                    if nz[i].any() else np.nan for i in range(Yd.shape[0])])
    ysum = np.asarray(Y_full.sum(axis=1)).ravel().astype(np.float64)
    ok = np.isfinite(s_c)
    pred = np.log1p(s_c[ok, None] * Yd[ok])
    info["check_processed_X_double_lognorm_max_abs_resid_percell_scale"] = float(np.abs(pred - Xp[ok]).max())
    ratio = (1e4 / s_c[ok]) / ysum[ok]
    info["check_processed_X_T_over_sum_y_range"] = [float(ratio.min()), float(np.median(ratio)), float(ratio.max())]
    info["check_processed_X_vs_single_lognorm_max_abs_diff"] = float(np.abs(Yd - Xp).max())
    # across-cell rank agreement per gene (what the double normalisation distorts)
    keep = nz.sum(axis=0) >= 30
    rk_y = pd.DataFrame(Yd[:, keep]).rank().to_numpy()
    rk_x = pd.DataFrame(Xp[:, keep].astype(np.float64)).rank().to_numpy()
    rk_y -= rk_y.mean(0)
    rk_x -= rk_x.mean(0)
    rho = (rk_y * rk_x).sum(0) / np.sqrt((rk_y ** 2).sum(0) * (rk_x ** 2).sum(0))
    info["check_processed_vs_single_across_cell_spearman_median_min"] = [float(np.median(rho)), float(rho.min())]

    expr = pd.DataFrame(Y, index=cells, columns=panel.gene.tolist())
    info.update(
        n_cells_total=n_obs, n_cells=len(cells), cell_rows_processed=rows_p.tolist(),
        n_genes=expr.shape[1], n_tfs=len(tfs),
        n_genes_expressed=int((Y > 0).any(axis=0).sum()),
        n_tfs_expressed=int((expr[tfs].to_numpy() > 0).any(axis=0).sum()),
        frac_nonzero=float((Y > 0).mean()),
        assay_counts=dict(Counter(assay)), donor_counts=dict(Counter(donor)),
        n_cell_types=len(set(ctype)),
    )
    return expr, info, tfs, cells


def gene_detection(expr: pd.DataFrame, tfs: list[str]) -> pd.DataFrame:
    det = pd.DataFrame({"gene": expr.columns, "is_tf": expr.columns.isin(tfs),
                        "n_cells_detected": (expr.to_numpy() > 0).sum(axis=0)})
    det["frac_detected"] = det.n_cells_detected / expr.shape[0]
    return det


# ----------------------------------------------------------------------------
# arboreto with a dask LocalCluster
# ----------------------------------------------------------------------------


class RSSMonitor:
    """Samples the summed RSS of this process and all its descendants (dask
    workers) and the 1-min load average, every ``interval`` seconds."""

    def __init__(self, interval: float = 2.0):
        import psutil
        self.psutil = psutil
        self.proc = psutil.Process()
        self.interval = interval
        self.peak_total = 0
        self.peak_worker = 0
        self.load1 = []
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True)

    def _sample(self):
        total, worker_max = 0, 0
        procs = [self.proc] + self.proc.children(recursive=True)
        for p in procs:
            try:
                r = p.memory_info().rss
            except (self.psutil.NoSuchProcess, self.psutil.AccessDenied):
                continue
            total += r
            if p.pid != self.proc.pid:
                worker_max = max(worker_max, r)
        self.peak_total = max(self.peak_total, total)
        self.peak_worker = max(self.peak_worker, worker_max)
        self.load1.append(os.getloadavg()[0])

    def _run(self):
        while not self._stop.is_set():
            self._sample()
            self._stop.wait(self.interval)

    def __enter__(self):
        self._t.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._t.join()
        self._sample()


def run_arboreto(expr: pd.DataFrame, tfs: list[str], method: str, n_workers: int, seed: int):
    """arboreto (GRNBoost2 / GENIE3 profile) on a LocalCluster with the compat layer
    installed on every worker.  Returns (links, per-target meta, worker status)."""
    AX.install()
    from arboreto.algo import _prepare_input
    from arboreto.core import EARLY_STOP_WINDOW_LENGTH, RF_KWARGS, SGBM_KWARGS, create_graph
    from distributed import Client, LocalCluster

    rtype, kwargs = {"grnboost2": ("GBM", SGBM_KWARGS), "genie3": ("RF", RF_KWARGS)}[method]
    # workers import r2.arboreto_compat (plugin + status checks)
    os.environ["PYTHONPATH"] = os.pathsep.join(
        [str(_SRC)] + [p for p in os.environ.get("PYTHONPATH", "").split(os.pathsep) if p])
    cluster = LocalCluster(n_workers=n_workers, threads_per_worker=1, processes=True,
                           dashboard_address=None, memory_limit="4GiB")
    client = Client(cluster)
    try:
        status_before = AX.install_on_client(client)
        matrix, gene_names, tf_names = _prepare_input(expr, None, tfs)
        links_g, meta_g = create_graph(matrix, gene_names, tf_names, client=client,
                                       regressor_type=rtype, regressor_kwargs=kwargs,
                                       include_meta=True,
                                       early_stop_window_length=EARLY_STOP_WINDOW_LENGTH, seed=seed)
        links, meta = client.compute([links_g, meta_g], sync=True)
        status_after = AX.worker_status(client)
    finally:
        client.close()
        cluster.close()
    workers = {"n_workers_checked_before": len(status_before), "n_workers_checked_after": len(status_after),
               "early_stop_monitor": sorted({s["early_stop_monitor"] for s in status_after.values()}),
               "sklearn": sorted({s["sklearn"] for s in status_after.values()})}
    return links, meta, workers


def tidy_links(net: pd.DataFrame, tfs: list[str], genes: list[str]) -> pd.DataFrame:
    df = net.rename(columns={"TF": "tf"})[["tf", "target", "importance"]].copy()
    df["tf"] = df["tf"].astype(str)
    df["target"] = df["target"].astype(str)
    df["importance"] = df["importance"].astype(np.float64)
    assert set(df.tf) <= set(tfs), "non-TF regulator returned"
    assert set(df.target) <= set(genes), "target outside panel"
    assert (df.tf != df.target).all(), "self-loop returned"
    assert not df.duplicated(["tf", "target"]).any(), "duplicate links"
    assert (df.importance > 0).all()
    # deterministic order: importance desc, then tf, target
    df = df.sort_values(["importance", "tf", "target"], ascending=[False, True, True],
                        kind="mergesort").reset_index(drop=True)
    return df


def rounds_table(meta_df: pd.DataFrame, links: pd.DataFrame, det: pd.DataFrame,
                 genes_fit: list[str], method: str) -> tuple[pd.DataFrame, dict]:
    """Per-target boosting rounds (arboreto meta) with completeness/consistency checks
    and the early-stopping guard.  Raises on failure."""
    m = meta_df.copy()
    m["target"] = m["target"].astype(str)
    if m.target.duplicated().any():
        raise RuntimeError("duplicate targets in arboreto meta output")
    missing = sorted(set(genes_fit) - set(m.target))
    if missing:   # arboreto's retry() swallows exceptions and returns an empty fallback
        raise RuntimeError(f"{len(missing)} targets have no arboreto meta row (failed fits), "
                           f"e.g. {missing[:5]}")
    agg = links.groupby("target").importance.agg(["sum", "size"])
    r = (det.set_index("gene").loc[genes_fit, ["n_cells_detected", "frac_detected"]]
         .rename_axis("target").reset_index())
    r["n_rounds"] = r.target.map(m.set_index("target").n_estimators).astype(int)
    r["importance_sum"] = r.target.map(agg["sum"]).fillna(0.0)
    r["n_links"] = r.target.map(agg["size"]).fillna(0).astype(int)
    summ: dict = {}
    if method == "grnboost2":
        # arboreto de-normalises importances by the number of trees, so the
        # importances of a target with >= 1 split sum to its number of rounds
        w = r.n_links > 0
        dev = float(np.abs(r.importance_sum[w] - r.n_rounds[w]).max()) if w.any() else 0.0
        summ["max_abs_importance_sum_minus_rounds"] = dev
        if dev > 1e-6:
            raise RuntimeError(f"importance sums differ from boosting rounds (max {dev})")
    expressed = r.n_cells_detected > 0
    re_ = r[expressed]
    summ.update(
        n_targets_fit=int(len(r)), n_targets_expressed=int(expressed.sum()),
        median_rounds_expressed=float(re_.n_rounds.median()),
        frac_expressed_targets_exactly_25_rounds=float((re_.n_rounds == 25).mean()),
        frac_expressed_targets_le_50_rounds=float((re_.n_rounds <= 50).mean()),
        n_unexpressed_targets_rounds=sorted(set(r.n_rounds[~expressed].tolist())),
    )
    for lo, hi, lab in DETECT_BINS:
        b = re_[(re_.frac_detected >= lo) & (re_.frac_detected < hi)]
        summ[f"n_targets_{lab}"] = int(len(b))
        summ[f"median_rounds_{lab}"] = float(b.n_rounds.median()) if len(b) else None
    if method == "grnboost2":
        g = re_[re_.frac_detected >= ROUNDS_GUARD_MIN_DETECT]
        med = float(g.n_rounds.median()) if len(g) else float("nan")
        summ["guard"] = {"min_detect_frac": ROUNDS_GUARD_MIN_DETECT, "n_targets": int(len(g)),
                         "median_rounds": med, "required_median_gt": ROUNDS_GUARD_MIN_MEDIAN,
                         "passed": bool(med > ROUNDS_GUARD_MIN_MEDIAN)}
        if not med > ROUNDS_GUARD_MIN_MEDIAN:
            raise RuntimeError(f"early-stopping guard failed: median rounds {med} over {len(g)} targets "
                               f"detected in >= {ROUNDS_GUARD_MIN_DETECT:.0%} of cells "
                               f"(required > {ROUNDS_GUARD_MIN_MEDIAN}); is the compat monitor active?")
    return r, summ


def _versions() -> dict:
    import importlib.metadata as md
    v = {"python": platform.python_version(), "platform": platform.platform(),
         "python_executable": sys.executable}
    for pkg in ("arboreto", "dask", "distributed", "scikit-learn", "numpy", "pandas", "pyarrow", "h5py"):
        try:
            v[pkg] = md.version(pkg)
        except md.PackageNotFoundError:
            v[pkg] = None
    return v


def _check_cells(od: Path, cells: list[str]) -> None:
    p = od / "cells.txt"
    if p.exists():
        prev = [l.strip() for l in p.read_text().splitlines() if l.strip()]
        if prev != cells:
            raise RuntimeError(f"cell sample differs from the existing {p}; refusing to overwrite")
    p.write_text("\n".join(cells) + "\n")


def cmd_run(args) -> None:
    sk = AX.check_sklearn(args.allow_unvalidated_sklearn)
    seed = int(C.PRESPECIFIED["random_seed"])
    t = args.tissue
    od = C.ensure_dir(OUT_STAGE / t)
    t0 = time.time()
    load0 = os.getloadavg()
    expr, info, tfs, cells = load_expression(t, args.n_cells, seed)
    t_load = time.time() - t0
    print(f"[{t}] expression {expr.shape} loaded in {t_load:.1f}s (scikit-learn {sk}); checks: "
          f"decontX identical={info['check_decontx_identical']}, "
          f"source single-lognorm max CV={info['check_source_X_single_lognorm_max_cv_ratio_all_cells']:.2e}, "
          f"S_c/total decontX={info['check_source_X_size_factor_over_total_decontx_min_median_max']}, "
          f"processed double-lognorm resid={info['check_processed_X_double_lognorm_max_abs_resid_percell_scale']:.2e}, "
          f"across-cell rho(processed, single) median/min="
          f"{info['check_processed_vs_single_across_cell_spearman_median_min']}",
          flush=True)
    _check_cells(od, cells)

    # per-gene detection in the sampled cells (label-free; shared by all runs)
    det = gene_detection(expr, tfs)
    det.to_csv(od / "gene_detection.tsv", sep="\t", index=False)

    n_universe = len(tfs) * (expr.shape[1] - 1)       # full U_ctx(t), before any filter
    name = args.method + variant_suffix(args.min_detect_frac)
    expr_fit, tfs_fit = expr, tfs
    if args.min_detect_frac > 0:
        keep = det.gene[det.frac_detected >= args.min_detect_frac].tolist()
        expr_fit = expr[keep]
        tfs_fit = [x for x in tfs if x in set(keep)]
        print(f"[{t}] detection filter >= {args.min_detect_frac:.2%} of cells: "
              f"{len(keep)}/{expr.shape[1]} genes, {len(tfs_fit)}/{len(tfs)} TFs", flush=True)

    with RSSMonitor() as mon:
        t1 = time.time()
        net, meta_df, workers = run_arboreto(expr_fit, tfs_fit, args.method, args.workers, seed)
        t_fit = time.time() - t1
    links = tidy_links(net, tfs_fit, expr_fit.columns.tolist())
    rounds, rsum = rounds_table(meta_df, links, det, expr_fit.columns.tolist(), args.method)
    print(f"[{t}] {name}: rounds summary {json.dumps(rsum)}", flush=True)
    links.to_parquet(od / f"{name}.parquet", index=False)
    rounds.to_csv(od / f"{name}_rounds.tsv", sep="\t", index=False)

    targets_with_links = links.target.nunique()
    expressed = expr_fit.columns[(expr_fit.to_numpy() > 0).any(axis=0)]
    is_gbm = args.method == "grnboost2"
    meta = dict(
        tissue=t, method=args.method, variant=name,
        min_detect_frac=args.min_detect_frac,
        n_genes_fit=int(expr_fit.shape[1]), n_tfs_fit=len(tfs_fit),
        algorithm=("arboreto 0.1.6 create_graph, GRNBoost2 profile (GradientBoostingRegressor, "
                   "SGBM_KWARGS, OOB early stopping, window 25) with the pre-scikit-learn-1.3 "
                   "same-OOB-subset improvement (arboreto_compat.SameOOBEarlyStopMonitor)"
                   if is_gbm else
                   "arboreto 0.1.6 create_graph, GENIE3 profile (RandomForestRegressor, RF_KWARGS)"),
        regressor_kwargs=_regressor_kwargs(args.method),
        early_stop_window_length=25 if is_gbm else None,
        early_stop_monitor=AX.EXPECTED_MONITOR if is_gbm else None,
        dask_workers_patch_check=workers,
        boosting_rounds_summary=rsum,
        boosting_rounds_per_target=dict(zip(rounds.target, rounds.n_rounds.astype(int).tolist())),
        seed=seed, cell_sampling="uniform without replacement, numpy default_rng(seed), sorted",
        expression=("Tabula Sapiens X = log1p(1e4 * decontX / S_c), S_c = TS per-cell size factor "
                    "(~ total decontX; ratio recorded in checks), read from source_h5ad for the "
                    "sampled cells"),
        n_cells=info["n_cells"], n_cells_total=info["n_cells_total"],
        n_genes=int(expr.shape[1]), n_tfs=len(tfs),
        n_genes_expressed=info["n_genes_expressed"], n_tfs_expressed=info["n_tfs_expressed"],
        frac_nonzero=info["frac_nonzero"],
        n_universe_edges=n_universe, n_links=int(len(links)),
        frac_universe_returned=len(links) / n_universe,
        n_targets_with_links=int(targets_with_links),
        n_expressed_targets_without_links=int(len(set(expressed) - set(links.target))),
        n_tfs_with_links=int(links.tf.nunique()),
        runtime_fit_s=round(t_fit, 1), runtime_load_s=round(t_load, 1),
        runtime_total_s=round(time.time() - t0, 1),
        n_workers=args.workers, threads_per_worker=1,
        peak_rss_total_mib=round(mon.peak_total / 2**20, 1),
        peak_rss_worker_mib=round(mon.peak_worker / 2**20, 1),
        peak_rss_main_mib=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, 1),
        rss_note=("MiB = bytes / 2**20.  peak_rss_total_mib = max over 2-s samples of the summed RSS of "
                  "the main process and all dask worker processes (psutil); peak_rss_worker_mib = "
                  "largest single worker; peak_rss_main_mib = getrusage(SELF) maxrss (macOS: bytes)"),
        loadavg_start=load0, loadavg_end=os.getloadavg(), cpu_count=os.cpu_count(),
        loadavg_1min_during_fit_min_median_max=[round(float(np.min(mon.load1)), 2),
                                                round(float(np.median(mon.load1)), 2),
                                                round(float(np.max(mon.load1)), 2)],
        arboreto_patch=("arboreto_compat.install(): EarlyStopMonitor -> SameOOBEarlyStopMonitor "
                        "(main process and every dask worker via WorkerPlugin); from_delayed returns None "
                        "for an empty list (dask-expr raises on from_delayed([]))"),
        versions=_versions(),
        checks={k: v for k, v in info.items() if k.startswith("check_")},
        assay_counts=info["assay_counts"], donor_counts=info["donor_counts"],
        n_cell_types=info["n_cell_types"],
        processed_h5ad=info["processed_h5ad"], source_h5ad=info["source_h5ad"],
        created=time.strftime("%Y-%m-%d %H:%M:%S"),
    )
    C.write_json(meta, od / f"{name}_meta.json")
    if name == "grnboost2":
        shutil.copyfile(od / "grnboost2_meta.json", od / "meta.json")
    print(json.dumps({k: meta[k] for k in ("tissue", "method", "variant", "n_links", "n_targets_with_links",
                                           "runtime_fit_s", "peak_rss_total_mib")}), flush=True)


def variant_suffix(min_detect_frac: float) -> str:
    """'' for the primary run (all panel genes), e.g. '_detect1pct' for a filtered run."""
    if min_detect_frac <= 0:
        return ""
    return f"_detect{min_detect_frac * 100:g}pct"


def _regressor_kwargs(method: str) -> dict:
    from arboreto.core import RF_KWARGS, SGBM_KWARGS
    return dict(SGBM_KWARGS if method == "grnboost2" else RF_KWARGS)


# ----------------------------------------------------------------------------
# Timing probe (serial, a few targets) used to choose feasible settings
# ----------------------------------------------------------------------------


def cmd_bench(args) -> None:
    AX.check_sklearn(args.allow_unvalidated_sklearn)
    AX.install()
    from arboreto.core import RF_KWARGS, SGBM_KWARGS, infer_partial_network, to_tf_matrix
    seed = int(C.PRESPECIFIED["random_seed"])
    expr, info, tfs, _ = load_expression(args.tissue, args.n_cells, seed)
    genes = expr.columns.tolist()
    M = expr.to_numpy()
    tfm, names = to_tf_matrix(M, genes, tfs)
    rng = np.random.default_rng(0)
    expressed = np.where((M > 0).mean(axis=0) > 0.05)[0]
    pick = rng.choice(expressed, size=min(args.n_targets, len(expressed)), replace=False)
    for method, kw, rt in (("grnboost2", SGBM_KWARGS, "GBM"), ("genie3", RF_KWARGS, "RF")):
        if method not in args.methods:
            continue
        ts = []
        for j in pick:
            t0 = time.time()
            infer_partial_network(rt, kw, tfm, names, genes[j], M[:, j], seed=seed)
            ts.append(time.time() - t0)
        print(f"[{args.tissue}] {method}: {len(ts)} targets, mean {np.mean(ts):.2f}s, "
              f"median {np.median(ts):.2f}s, max {np.max(ts):.2f}s -> serial estimate for "
              f"{len(genes)} targets {np.mean(ts) * len(genes) / 60:.1f} min", flush=True)


# ----------------------------------------------------------------------------
# Early-stopping validation: compat monitor (this environment) vs arboreto's
# original monitor under scikit-learn <= 1.2 (reference environment)
# ----------------------------------------------------------------------------


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra = pd.Series(a).rank().to_numpy()
    rb = pd.Series(b).rank().to_numpy()
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def cmd_validate_es(args) -> None:
    import arboreto.core as ACORE
    from arboreto.core import SGBM_KWARGS, clean, to_tf_matrix
    # arboreto's own (unpatched) monitor, even if install() ran in this process
    ArboretoMonitor = getattr(ACORE, "_r2_original_EarlyStopMonitor", ACORE.EarlyStopMonitor)
    from sklearn.ensemble import GradientBoostingRegressor
    sk = AX.check_sklearn(args.allow_unvalidated_sklearn)
    seed = int(C.PRESPECIFIED["random_seed"])
    rng = np.random.default_rng(seed)
    problems, rows = [], []
    for t in C.TISSUES:
        expr, _, tfs, _ = load_expression(t, args.n_cells, seed)
        det = gene_detection(expr, tfs).set_index("gene")
        genes = expr.columns.tolist()
        M = expr.to_numpy()
        tfm, names = to_tf_matrix(M, genes, tfs)
        main_rounds = {}
        mp = OUT_STAGE / t / "grnboost2_rounds.tsv"
        if mp.exists():
            main_rounds = pd.read_csv(mp, sep="\t").set_index("target").n_rounds.to_dict()
        for lo, hi, lab in DETECT_BINS:
            cand = sorted(det.index[(det.frac_detected >= lo) & (det.frac_detected < hi) & (det.n_cells_detected > 0)])
            pick = sorted(rng.choice(cand, size=min(args.n_per_bin, len(cand)), replace=False))
            for g in pick:
                X, tf_names = clean(tfm, names, g)
                y = M[:, genes.index(g)]
                row = dict(tissue=t, target=g, detect_bin=lab, n_cells_detected=int(det.n_cells_detected[g]))
                imps = {}
                for nm, mon in (("original", ArboretoMonitor(25)), ("compat", AX.SameOOBEarlyStopMonitor(25))):
                    reg = GradientBoostingRegressor(random_state=seed, **SGBM_KWARGS)
                    reg.fit(X, y, monitor=mon)
                    r = len(reg.estimators_)
                    row[f"rounds_sk{sk}_{nm}"] = r
                    imps[nm] = reg.feature_importances_ * r
                row["rounds_main_run"] = main_rounds.get(g)
                rows.append(row)
                problems.append((X, y.astype(np.float32), imps["compat"], tf_names))
    df = pd.DataFrame(rows)
    comp = f"rounds_sk{sk}_compat"
    ref_info = None
    if args.ref_python:
        with tempfile.TemporaryDirectory() as td:
            inp, outp = Path(td) / "in.npz", Path(td) / "out.json"
            np.savez(inp, seed=seed, **{f"X_{k}": p[0] for k, p in enumerate(problems)},
                     **{f"y_{k}": p[1] for k, p in enumerate(problems)})
            subprocess.run([args.ref_python, str(Path(__file__).with_name("es_reference_replay.py")),
                            str(inp), str(outp)], check=True)
            ref = json.loads(outp.read_text())
        rv = ref["sklearn"]
        ref_info = {"python": args.ref_python, "sklearn": rv, "numpy": ref["numpy"]}
        for k, pr in enumerate(ref["problems"]):
            df.loc[k, f"rounds_sk{rv}_original"] = pr["original"]["rounds"]
            df.loc[k, f"rounds_sk{rv}_compat"] = pr["compat"]["rounds"]
            df.loc[k, f"sk{rv}_compat_vs_oob_improvement_max_abs_diff"] = pr["compat_vs_sklearn_oob_improvement_max_abs_diff"]
            a = problems[k][2]
            b = np.asarray(pr["original"]["importance"])
            df.loc[k, "importance_spearman_compat_vs_ref_original"] = _spearman(a, b)
            ta = set(np.argsort(-a)[:10][a[np.argsort(-a)[:10]] > 0])
            tb = set(np.argsort(-b)[:10][b[np.argsort(-b)[:10]] > 0])
            df.loc[k, "top10_tf_overlap_compat_vs_ref_original"] = len(ta & tb)
        orig_ref = f"rounds_sk{rv}_original"
        df["rounds_equal_compat_vs_ref_original"] = df[comp] == df[orig_ref]
        df["rounds_rel_diff_compat_vs_ref_original"] = (df[comp] - df[orig_ref]).abs() / df[orig_ref]
    out = OUT_STAGE / "early_stop_validation.csv"
    df.to_csv(out, index=False)
    summ = {"sklearn_here": sk, "reference": ref_info, "n_problems": int(len(df)),
            "median_rounds_original_here": float(df[f"rounds_sk{sk}_original"].median()),
            "median_rounds_compat_here": float(df[comp].median())}
    w = df.rounds_main_run.notna()
    if w.any():
        summ["main_run_rounds_equal_serial_compat"] = int((df.rounds_main_run[w] == df[comp][w]).sum())
        summ["main_run_rounds_compared"] = int(w.sum())
    if ref_info:
        summ.update(
            rounds_equal_compat_vs_ref_original=int(df.rounds_equal_compat_vs_ref_original.sum()),
            median_rounds_ref_original=float(df[orig_ref].median()),
            median_rel_diff_rounds=float(df.rounds_rel_diff_compat_vs_ref_original.median()),
            median_importance_spearman=float(df.importance_spearman_compat_vs_ref_original.median()),
            ref_compat_equals_ref_original_rounds=int((df[f"rounds_sk{rv}_compat"] == df[orig_ref]).sum()),
            ref_compat_vs_sklearn_oob_improvement_max_abs_diff=float(
                df[f"sk{rv}_compat_vs_oob_improvement_max_abs_diff"].max()))
    C.write_json(summ, OUT_STAGE / "early_stop_validation.json")
    print(df.to_string(index=False))
    print(json.dumps(summ, indent=2))
    if w.any() and summ["main_run_rounds_equal_serial_compat"] != summ["main_run_rounds_compared"]:
        raise RuntimeError("dask main-run rounds differ from the serial compat fits")
    if ref_info and summ["ref_compat_equals_ref_original_rounds"] != len(df):
        raise RuntimeError("compat monitor does not reproduce arboreto's original rounds under the reference sklearn")


# ----------------------------------------------------------------------------
# Manuscript-facing runtime table
# ----------------------------------------------------------------------------


def _parse_time_l(path: Path) -> dict:
    if not path.exists():
        return {}
    txt = path.read_text()
    out = {}
    m = re.search(r"([\d.]+)\s+real\s+([\d.]+)\s+user\s+([\d.]+)\s+sys", txt)
    if m:
        out.update(time_l_real_s=float(m.group(1)), time_l_user_s=float(m.group(2)),
                   time_l_sys_s=float(m.group(3)))
    m = re.search(r"(\d+)\s+maximum resident set size", txt)
    if m:   # largest single process (main or one dask worker); macOS reports bytes
        out["time_l_maxrss_largest_process_mib"] = round(int(m.group(1)) / 2**20, 1)
    return out


def cmd_table(args) -> None:
    rows = []
    for t in C.TISSUES:
        for mp in sorted((OUT_STAGE / t).glob("*_meta.json")):
            m = json.loads(mp.read_text())
            row = {k: m[k] for k in (
                "tissue", "method", "variant", "min_detect_frac", "n_genes_fit", "n_tfs_fit", "n_cells",
                "n_cells_total", "n_genes", "n_tfs", "n_genes_expressed", "n_tfs_expressed",
                "n_universe_edges", "n_links", "frac_universe_returned", "n_targets_with_links",
                "n_tfs_with_links", "runtime_fit_s", "runtime_total_s", "n_workers", "peak_rss_total_mib",
                "peak_rss_worker_mib", "peak_rss_main_mib")}
            rs = m["boosting_rounds_summary"]
            row["median_rounds_expressed_targets"] = rs["median_rounds_expressed"]
            for _, _, lab in DETECT_BINS:
                row[f"median_rounds_{lab}"] = rs[f"median_rounds_{lab}"]
            row["frac_expressed_targets_exactly_25_rounds"] = rs["frac_expressed_targets_exactly_25_rounds"]
            row["loadavg_1min_start"] = round(m["loadavg_start"][0], 2)
            lo, med, hi = m["loadavg_1min_during_fit_min_median_max"]
            row.update(loadavg_1min_fit_min=lo, loadavg_1min_fit_median=med, loadavg_1min_fit_max=hi)
            row.update(_parse_time_l(OUT_STAGE / t / f"time_l_{m['variant']}.txt"))
            row["arboreto"] = m["versions"]["arboreto"]
            row["dask"] = m["versions"]["dask"]
            row["scikit_learn"] = m["versions"]["scikit-learn"]
            row["early_stop_monitor"] = m.get("early_stop_monitor")
            rows.append(row)
    df = pd.DataFrame(rows)
    df["primary"] = df.min_detect_frac == 0
    order = {"grnboost2": 0, "genie3": 1}
    df = df.sort_values(["tissue", "min_detect_frac", "method"],
                        key=lambda c: c.map(order) if c.name == "method" else c).reset_index(drop=True)
    C.ensure_dir(C.TABLES)
    df.to_csv(C.TABLES / "grn_baseline_runtime.csv", index=False)
    print(df.to_string(index=False))


# ----------------------------------------------------------------------------
# Rare-target / overlap summary of the stored outputs
# ----------------------------------------------------------------------------


def predicted_topk(links: pd.DataFrame, variant: str, k: int) -> pd.DataFrame:
    """Top-k links per TF by importance under the shared prespecified rule
    (``r2.predicted_sets.topk_per_tf``; seeded random break of ties at the k-th
    position, pairs not returned = not predicted).  The score column is named after
    the variant (e.g. ``grnboost2``, ``genie3_detect1pct``), so different methods get
    independent tie-breaks."""
    d = links.rename(columns={"importance": variant})
    return links[topk_per_tf(d, variant, k).to_numpy()].reset_index(drop=True)


def _n_boundary_ties(links: pd.DataFrame, k: int) -> int:
    n = 0
    for _, g in links.groupby("tf"):
        if len(g) > k:
            v = np.sort(g.importance.to_numpy())[::-1]
            n += int(v[k - 1] == v[k])
    return n


def _jaccard_per_tf(a: pd.DataFrame, b: pd.DataFrame) -> pd.Series:
    sa = a.groupby("tf").target.apply(set)
    sb = b.groupby("tf").target.apply(set)
    common = sa.index.intersection(sb.index)
    return pd.Series({tf: len(sa[tf] & sb[tf]) / len(sa[tf] | sb[tf]) for tf in common}, dtype=float)


def cmd_summary(args) -> None:
    k = int(C.PRESPECIFIED["topk_per_tf"])
    seed = int(C.PRESPECIFIED["topk_tie_seed"])     # used inside r2.predicted_sets.topk_per_tf
    rows, extra = [], {}
    for t in C.TISSUES:
        od = OUT_STAGE / t
        det = pd.read_csv(od / "gene_detection.tsv", sep="\t").set_index("gene")
        rare = det.frac_detected < RARE_FRAC
        tops = {}
        for mp in sorted(od.glob("*_meta.json")):
            m = json.loads(mp.read_text())
            v = m["variant"]
            links = pd.read_parquet(od / f"{v}.parquet")
            top = predicted_topk(links, v, k)
            tops[v] = (links, top)
            per_tf = top.groupby("tf").size()
            g10 = links.head(10)
            rows.append(dict(
                tissue=t, variant=v, n_links=len(links), frac_universe=m["frac_universe_returned"],
                median_rounds_expressed=m["boosting_rounds_summary"]["median_rounds_expressed"],
                **{f"median_rounds_{lab}": m["boosting_rounds_summary"][f"median_rounds_{lab}"]
                   for _, _, lab in DETECT_BINS},
                n_rare_genes=int(rare.sum()), n_rare_tfs=int((rare & det.is_tf).sum()),
                topk=k, n_tfs_topk=int(len(per_tf)), n_links_topk=int(len(top)),
                pct_topk_links_target_rare=round(100 * float(top.target.map(rare).mean()), 1),
                pct_panel_genes_rare=round(100 * float(rare.mean()), 1),
                **{f"pct_topk_links_target_{lab}": round(100 * float(
                    top.target.map(det.frac_detected).between(lo, hi, inclusive="left").mean()), 1)
                   for lo, hi, lab in DETECT_BINS},
                pct_topk_links_target_rare_among_nonrare_tfs=round(
                    100 * float(top[~top.tf.map(rare)].target.map(rare).mean()), 1),
                n_tfs_fewer_than_k_links=int((per_tf < k).sum()),
                n_tfs_tied_at_k=_n_boundary_ties(links, k),
                global_top10="; ".join(f"{a}->{b} ({det.n_cells_detected[a]}/{det.n_cells_detected[b]} cells)"
                                       for a, b in zip(g10.tf, g10.target)),
            ))
        for method in ("grnboost2", "genie3"):
            fv = method + variant_suffix(RARE_FRAC)
            if method in tops and fv in tops:
                j = _jaccard_per_tf(tops[method][1], tops[fv][1])
                key = (f"{t}_jaccard_top{k}_main_vs_detect1pct" if method == "grnboost2"
                       else f"{t}_{method}_jaccard_top{k}_main_vs_detect1pct")
                extra[key] = {"n_tfs": int(len(j)), "median": round(float(j.median()), 3),
                              "mean": round(float(j.mean()), 3)}
        for sfx in ("", variant_suffix(RARE_FRAC)):
            ga, gb = "grnboost2" + sfx, "genie3" + sfx
            if ga in tops and gb in tops:
                a, b = tops[ga], tops[gb]
                j = _jaccard_per_tf(a[1], b[1])
                mg = a[0].merge(b[0], on=["tf", "target"], suffixes=("_gb", "_g3"))
                extra[f"{t}_{ga}_vs_{gb}"] = {
                    "jaccard_topk_n_tfs": int(len(j)), "jaccard_topk_median": round(float(j.median()), 3),
                    "jaccard_topk_mean": round(float(j.mean()), 3),
                    "spearman_shared_pairs": round(_spearman(mg.importance_gb.to_numpy(),
                                                             mg.importance_g3.to_numpy()), 3),
                    "n_shared_pairs": int(len(mg))}
    df = pd.DataFrame(rows)
    df.to_csv(OUT_STAGE / "summary_rare_targets.csv", index=False)
    C.write_json({"topk": k, "tie_seed": seed, "topk_rule": "r2.predicted_sets.topk_per_tf (score column = variant)",
                  "rare_frac": RARE_FRAC, "comparisons": extra},
                 OUT_STAGE / "summary.json")
    with pd.option_context("display.max_colwidth", 400, "display.width", 250):
        print(df.drop(columns="global_top10").to_string(index=False))
        for _, r in df.iterrows():
            print(f"{r.tissue}/{r.variant} global top10: {r.global_top10}")
    print(json.dumps(extra, indent=2))


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="run GRNBoost2 or GENIE3 for one tissue")
    r.add_argument("--tissue", choices=C.TISSUES, required=True)
    r.add_argument("--method", choices=("grnboost2", "genie3"), default="grnboost2")
    r.add_argument("--n-cells", type=int, default=N_CELLS_DEFAULT)
    r.add_argument("--workers", type=int, default=N_WORKERS_DEFAULT)
    r.add_argument("--min-detect-frac", type=float, default=0.0,
                   help="sensitivity only: keep genes detected in >= this fraction of sampled cells "
                        "(SCENIC-style filter, e.g. 0.01); 0 = primary run on all panel genes")
    b = sub.add_parser("bench", help="serial per-target timing probe")
    b.add_argument("--tissue", choices=C.TISSUES, default=C.PRIMARY_TISSUE)
    b.add_argument("--n-cells", type=int, default=N_CELLS_DEFAULT)
    b.add_argument("--n-targets", type=int, default=8)
    b.add_argument("--methods", nargs="+", default=["grnboost2", "genie3"])
    v = sub.add_parser("validate-es", help="compat early stopping vs arboreto's original under sklearn <= 1.2")
    v.add_argument("--n-cells", type=int, default=N_CELLS_DEFAULT)
    v.add_argument("--n-per-bin", type=int, default=4, help="targets per detection bin and tissue")
    v.add_argument("--ref-python", default=os.environ.get("R2_SKLEARN_REF_PYTHON"),
                   help="python executable with scikit-learn <= 1.2 (numpy + sklearn only)")
    for p in (r, b, v):
        p.add_argument("--allow-unvalidated-sklearn", action="store_true",
                       help=f"run with a scikit-learn version outside {AX.SKLEARN_VALIDATED}")
    sub.add_parser("table", help="write data/r2/tables/grn_baseline_runtime.csv")
    sub.add_parser("summary", help="rare-target and overlap summary of the stored outputs")
    args = ap.parse_args(argv)
    if args.cmd == "run":
        assert args.workers <= 6, "at most 6 dask workers on the shared machine"
        cmd_run(args)
    elif args.cmd == "bench":
        cmd_bench(args)
    elif args.cmd == "validate-es":
        cmd_validate_es(args)
    elif args.cmd == "table":
        cmd_table(args)
    else:
        cmd_summary(args)


if __name__ == "__main__":
    main()
