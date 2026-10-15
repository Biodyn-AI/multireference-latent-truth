"""Compute cost and scalability of the pattern-compressed latent-class models (E8; reviewer Q21/Q22).

What this module measures
-------------------------
Synthetic evidence matrices of N edges x K annotators with realistic, TF-blocked sparsity are
generated, compressed to their P unique observation patterns by streaming TF chunks, and the
primary model (``family_hb``, ``lcm.build_model``) is fitted by MAP (10 L-BFGS restarts,
``lcm.fit_map`` settings) and by NUTS (2 chains x 300 warm-up / 300 draws, target acceptance
0.95, the ``lcm.fit_nuts`` kernel).  For every stage the wall time, process CPU time, peak
resident set size (RSS) and the machine load average are recorded.  The real pipeline is
**not** timed here (isolated per-stage timings of the real pipeline are a separate task).

Complexity (derivation in the module report; ``data/r2/tables/scale_complexity.csv``)
-------------------------------------------------------------------------------------
* Encoding each edge's K evidence values: O(N K) time.  Compression by sort-based unique per
  TF chunk of n_c edges + merge of the chunk dictionaries: O(N K + N log n_c + C P_c log(C P_c))
  time (C chunks with P_c <= P patterns each); memory O(n_c K) per chunk, O(N) for the int32
  edge->pattern index (4 bytes/edge) and O(P K) for the pattern table.
* One evaluation of the collapsed log posterior and its gradient: O(P K S) time (two (P, K) x
  (K, S) products and a logsumexp over S classes), memory O(P K) (16 P K bytes for the float64
  observed / reported indicator matrices) + O(P S).  P <= min(N, 3^K) and N enters only as the
  pattern weights, so the cost is independent of N given P.
* MAP: O(R I P K S) for R restarts of I L-BFGS evaluations.  NUTS: O(C D L P K S) for C
  chains, D iterations and L leapfrog steps per iteration (L <= 2^10 - 1); memory
  O(P K + C D d) for d parameters if the per-pattern credibility is not stored per draw
  (``lcm.fit_nuts`` stores it as a deterministic site, adding O(C D P)), plus O(D P) for
  posterior credibility quantiles computed afterwards (``lcm.nuts_credibility``).

Synthetic evidence (``template`` stage reads the real global evidence once)
--------------------------------------------------------------------------
The real primary evidence (global universe, ``prespecified_cc``: 935 TFs x 9,963 genes,
K = 21) has an exact product structure for every literature (LIT) annotator j:
observed(t, g) = cov_j(t) AND node_j(g) (TF coverage x target node set), and a MOTIF/COEXP
annotator observes (t, g) only if t is covered and (t, g) is in the conditioning set (a subset
of the literature-reported edges).  Both generators keep this structure; edges of a synthetic
universe of T TFs x G genes (T, G scaled by sqrt(N / N_real)) are generated TF chunk by TF
chunk:

* ``template`` (primary): each synthetic TF draws the coverage profile of a random real TF
  (all K columns jointly, so the correlation between resources is kept) and each synthetic
  gene draws the node-set profile of a random real gene.
* ``independent`` (sensitivity: no correlation between resources): cov_j(t) ~
  Bernoulli(n_tfs_covered_j / n_tfs) and node_j(g) ~ Bernoulli(n_observed_j /
  (n_tfs_covered_j n_genes)) independently per annotator, from
  ``data/r2/tables/evidence_summary.csv`` (global, prespecified_cc).  MAP only (no NUTS).
* ``stress`` (deliberately unrealistic; times model fitting when compression does not help):
  every cell is observed independently with probability ``STRESS_OBS`` = 0.5 and z ~ w, so
  P approaches min(N, 2^K ...).  Not used for any statement about realistic P.

Reports r ~ Bernoulli(theta_j[z]) on observed cells use the MAP parameters of the real global
``family_hb`` fit.  The latent class is drawn given the edge's observation depth m (number of
LIT annotators observing it): z ~ Categorical(w_m), with w_m the edge-weighted mean posterior
class probabilities of the real patterns of depth m at those parameters (with z ~ w for every
edge, deeply observed edges get too few reports: real data are not missing at random).  The
conditioning set is the literature-reported edges thinned with the real fraction pi_c.  K = 10 keeps the 7 LIT
annotators with the largest real coverage plus the 3 MOTIF/COEXP annotators; K = 42 uses two
independent 21-annotator blocks per TF / gene that share z (families LIT, MOTIF, COEXP, so
S = 5 classes instead of 4).  Per-annotator coverage and positive rates of the real-size
synthetic universes are compared with the real ones in ``scale_generator_check.csv``.

Measurement protocol
--------------------
Every measurement runs in a fresh subprocess (``worker-*`` sub-commands) with OMP/OPENBLAS/MKL
threads = 2, XLA intra-op threads = 2 and ``nice -n 10`` (inherited).  Peak RSS = max(psutil RSS
sampled every 20 ms, ``getrusage`` ru_maxrss), converted to MiB (ru_maxrss is in bytes on macOS
and KiB on Linux); it is a whole-process peak (interpreter, numpy / JAX runtime and data), and a
lower bound of the memory footprint when the machine is short of memory (compressed or swapped
pages are not resident), so the summary reports the largest value over the sessions.  The
1-minute load average is sampled every second.  Other agents share the 10-core laptop, so wall
times are upper bounds; CPU times are reported alongside.  Budget rules fixed before the
MAP/NUTS runs: MAP stops after the restart that crosses ``MAP_BUDGET_S`` (>= 2 restarts) and the
10-restart time is extrapolated from the mean restart; if one restart is projected (largest mean
evaluation count of the completed runs x measured evaluation time) to exceed ``MAP_BUDGET_S``,
only the evaluation timings are measured.  NUTS is run only if its projected time (mean
steps/iteration and time-per-step / value-and-gradient ratio of the measured run with the same K
and the largest P, x the value-and-gradient time of this configuration) is <= ``NUTS_BUDGET_S``,
otherwise it is projected; a NUTS worker is killed after ``NUTS_TIMEOUT_S`` (status 'timeout',
the timeout is a lower bound).  A projection assumes the steps/iteration of its (smaller)
reference run; because steps/iteration grew with N, the tables also give the projection at the
tree-depth cap of 1023 steps (``t_nuts_projected_cap_s``) as an upper bracket.  Every projected number is flagged (``*_status`` = measured /
extrapolated / projected / timeout).

NUTS timing: the timed 2 x 300/300 run (``t_nuts_s``, warm-up + sampling, chains run sequentially
as in ``lcm.fit_nuts``) compiles its own kernels, so ``t_nuts_s`` includes that compilation.  A
probe measured before the timed run estimates it (``t_probe_s``; see ``timed_nuts``), and the
tables give the time per leapfrog step both as ``t_nuts_s`` / steps (upper bound) and as
(``t_nuts_s`` - ``t_probe_s``) / steps.  R-hat / ESS of these short random-start chains
(``lcm.fit_nuts`` site rule) are recorded as context only: they are cost measurements, not
converged fits (the Family-HB posterior is multimodal; see ``posterior_modes.py``).

Sessions
--------
* ``a`` (2026-10-08/09, ``runs/``): the complete grid.  From 22:22 on the MAP and NUTS drivers
  ran concurrently (<= 2 measurement subprocesses), and the machine was heavily loaded by other
  agents (1-minute load average 21-217 over the runs, peaks above 300 on 10 cores).
* ``b`` (2026-10-13, ``runs_b/``; ``replicate`` command): a replicate with one measurement
  subprocess at a time.  It re-runs compression and the per-evaluation timings of every
  configuration, the MAP restarts of the configurations whose 10 restarts were measured within
  ``REPLICATE_MAP_MAX_S`` = 300 s in session a, and NUTS of the configurations measured within
  ``NUTS_BUDGET_S`` in session a (rules fixed before the replicate was started).  It never
  overwrites the stored pattern tables; it checks that the regenerated patterns, the L-BFGS
  paths and the NUTS step counts equal those of session a (``scale_replicate.csv``).  The
  remaining configurations get projections (MAP: session-a evaluation counts x session-b
  evaluation time; NUTS: session-b value-and-gradient time x the steps/iteration and
  step / value-and-gradient ratio of the largest measured run with the same K in either session).
``scale_summary.csv`` reports, per configuration, the most direct record over the sessions
(measured < extrapolated < projected) and, among equally direct ones, the shortest time (the
least disturbed measurement); the session and the load of each reported number are given.
Session b was designed after the project's held-out checkpoint (it reads no held-out data): rows
that use a session-b record carry ``analysis_status`` = post_hoc, session-a rows prespecified.

The template (``outputs/r2/scale/template.*``) is a frozen snapshot of the real evidence
structure and of the generating parameters; ``all`` keeps an existing template.  ``template``
rebuilds it from the currently stored fit, after which every stage must be re-run with
``--force`` (``scale_environment.csv`` records the distance between the snapshot and the
currently stored fit).

Commands
--------
    export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \
        XLA_FLAGS='--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=2'
    nice -n 10 .venv/bin/python subproject_24_multireference_latent_truth_model/src/r2/compute_scaling.py all
    # stages: template | compress | map | nuts [--session a|b] | replicate | tables
    # (all = every stage of session a + tables; replicate = every stage of session b + tables;
    #  existing records are kept unless --force is given)
    .venv/bin/python -m pytest subproject_24_multireference_latent_truth_model/src/r2/tests/test_compute_scaling.py -q

Outputs: ``outputs/r2/scale/`` (template, compressed patterns, one JSON per measurement and
session) and the manuscript tables ``data/r2/tables/scale_*.csv`` (column ``session`` where
both sessions are listed).  No held-out data is read.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import platform
import resource
import subprocess
import sys
import threading
import time
import zlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from r2 import common as C  # noqa: E402

# ----------------------------------------------------------------------------
# Constants (fixed before any scaling run)
# ----------------------------------------------------------------------------
SCALE = C.OUT / "scale"
TEMPLATE_NPZ = SCALE / "template.npz"
TEMPLATE_JSON = SCALE / "template.json"
PAT_DIR = SCALE / "patterns"
RUN_DIR = SCALE / "runs"
PRIMARY_VARIANT = C.PRESPECIFIED["evidence_variant_primary"]
GEN_FIT = C.OUT / "fits" / "global" / PRIMARY_VARIANT / "family_hb"   # generating parameters
SEED = int(C.PRESPECIFIED["random_seed"])

MODEL = "family_hb"                 # prespecified primary estimator
N_RESTARTS = 10
NUTS_CHAINS, NUTS_WARMUP, NUTS_SAMPLES = 2, 300, 300
TARGET_ACCEPT = 0.95                # lcm.fit_nuts default
MAP_BUDGET_S = 1800.0               # per configuration (see module docstring)
NUTS_BUDGET_S = 1800.0
NUTS_TIMEOUT_S = 3 * NUTS_BUDGET_S  # hard stop of one NUTS worker (status 'timeout', lower bound)
NUTS_DEFAULT_STEPS = 127.0          # steps/iteration before any NUTS run with the same K exists
NUTS_MAX_STEPS = 1023               # leapfrog steps per iteration at the numpyro default max tree depth (10)
CHUNK_EDGES = 1 << 20               # TF chunks of whole TFs with <= ~1M edges
TRITS = 39                          # base-3 digits per int64 word (3**39 < 2**63)
N_BENCH = 15                        # repeated evaluations for the per-evaluation timings
K10_N_LIT = 7                       # K = 10: 7 largest LIT annotators + all MOTIF/COEXP
STRESS_OBS = 0.5                    # per-cell observation probability of the stress generator
NUTS_GENERATORS = ("template", "stress")
THREAD_ENV = {
    "OMP_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2", "MKL_NUM_THREADS": "2",
    "VECLIB_MAXIMUM_THREADS": "2",
    "XLA_FLAGS": "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=2",
}
ANALYSIS_STATUS = "prespecified"    # E8 of the plan (section 2.3); uses no held-out data
# The replicate session was designed on 2026-10-13, after the project's held-out checkpoint and
# after the session-a timings were seen.  It reads no held-out data, but by the project rule
# every row that uses a session-b record is labelled post hoc.
SESSION_STATUS = {"a": ANALYSIS_STATUS, "b": "post_hoc"}


def _status_by_session(sessions) -> list:
    return [SESSION_STATUS.get(s, ANALYSIS_STATUS) for s in sessions]
SIZES = ("1e5", "1e6", "1e7", "real", "10xreal")
KS = (10, 21, 42)
# Measurement sessions.  'a' is the complete grid; 'b' is the replicate at a lower machine load
# (see "Sessions" in the module docstring).  Session 'a' keeps the original directory names.
SESSIONS = {
    "a": "complete grid, 2026-10-08/09; the MAP and NUTS drivers ran concurrently from 22:22 on "
         "(<= 2 measurement subprocesses at a time); machine heavily loaded by other agents",
    "b": "replicate, 2026-10-13, one measurement subprocess at a time (machine still shared): compression "
         "and per-evaluation timings of every configuration; MAP restarts of the configurations whose 10 "
         "restarts took <= 300 s in session a; NUTS of the configurations measured within the budget in "
         "session a",
}
PRIMARY_SESSION = "a"
REPLICATE_SESSION = "b"
REPLICATE_MAP_MAX_S = 300.0         # replicate MAP restarts if the primary 10-restart time was measured and <= this
STATUS_RANK = {"measured": 0, "extrapolated": 1, "projected": 2, "timeout": 3}


def run_dir(session: str = PRIMARY_SESSION) -> Path:
    """Directory of the measurement records of a session (``runs`` for 'a', ``runs_<s>`` otherwise)."""
    if session not in SESSIONS:
        raise ValueError(f"unknown session {session!r}")
    return RUN_DIR if session == PRIMARY_SESSION else RUN_DIR.parent / f"{RUN_DIR.name}_{session}"


def log_dir(session: str = PRIMARY_SESSION) -> Path:
    return SCALE / ("logs" if session == PRIMARY_SESSION else f"logs_{session}")


# ----------------------------------------------------------------------------
# Resource measurement
# ----------------------------------------------------------------------------
def maxrss_mib(ru_maxrss: int, system: str | None = None) -> float:
    """``getrusage`` ru_maxrss in MiB: bytes on macOS (Darwin), KiB on Linux."""
    system = system or platform.system()
    return ru_maxrss / 2**20 if system == "Darwin" else ru_maxrss / 2**10


class Monitor:
    """Samples this process's RSS (every ``interval`` s) and the 1-min load average
    (every second) in a background thread; ``stop()`` returns wall/CPU time, the peak RSS
    during the stage and load statistics."""

    def __init__(self, interval: float = 0.02):
        import psutil
        self._p = psutil.Process()
        self.interval = interval
        self._stop = threading.Event()
        self.peak = 0
        self.loads = []

    def _run(self):
        last_load = 0.0
        while not self._stop.is_set():
            try:
                self.peak = max(self.peak, self._p.memory_info().rss)
            except Exception:  # pragma: no cover - process teardown
                pass
            now = time.monotonic()
            if now - last_load >= 1.0:
                self.loads.append(os.getloadavg()[0])
                last_load = now
            self._stop.wait(self.interval)

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *exc):
        self.stop()

    def start(self):
        self.t0 = time.perf_counter()
        self.c0 = time.process_time()
        self.load_start = os.getloadavg()
        self.peak = self._p.memory_info().rss
        self.rss_start = self.peak
        self._th = threading.Thread(target=self._run, daemon=True)
        self._th.start()
        return self

    def stop(self) -> dict:
        self._stop.set()
        self._th.join()
        self.peak = max(self.peak, self._p.memory_info().rss)
        load_end = os.getloadavg()
        loads = self.loads or [load_end[0]]
        self.result = dict(
            wall_s=time.perf_counter() - self.t0, cpu_s=time.process_time() - self.c0,
            rss_start_mib=self.rss_start / 2**20, peak_rss_sampled_mib=self.peak / 2**20,
            load1_start=self.load_start[0], load5_start=self.load_start[1], load15_start=self.load_start[2],
            load1_end=load_end[0], load1_mean=float(np.mean(loads)), load1_max=float(np.max(loads)))
        return self.result


def process_peak_mib() -> float:
    return maxrss_mib(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


def _sh(cmd: list) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return ""


def environment() -> pd.DataFrame:
    """Hardware and software of the measurement machine (scale_environment.csv)."""
    import importlib
    rows = [("tables built (machine and software recorded)", time.strftime("%Y-%m-%d %H:%M:%S %Z")),
            ("cpu", _sh(["sysctl", "-n", "machdep.cpu.brand_string"])),
            ("hw.ncpu", _sh(["sysctl", "-n", "hw.ncpu"])),
            ("hw.physicalcpu", _sh(["sysctl", "-n", "hw.physicalcpu"])),
            ("hw.perflevel0.physicalcpu (performance cores)", _sh(["sysctl", "-n", "hw.perflevel0.physicalcpu"])),
            ("hw.perflevel1.physicalcpu (efficiency cores)", _sh(["sysctl", "-n", "hw.perflevel1.physicalcpu"])),
            ("hw.memsize (bytes)", _sh(["sysctl", "-n", "hw.memsize"])),
            ("os", " ".join(_sh(["sw_vers", f"-{k}"]) for k in ("productName", "productVersion", "buildVersion"))),
            ("platform", platform.platform()),
            ("python", sys.version.split()[0]),
            ("python executable (resolved)", os.path.realpath(sys.executable)),
            ("python prefix", sys.prefix)]
    for mod in ("numpy", "scipy", "pandas", "jax", "jaxlib", "numpyro", "arviz", "psutil"):
        try:
            rows.append((mod, importlib.import_module(mod).__version__))
        except Exception as exc:  # pragma: no cover
            rows.append((mod, f"unavailable ({exc})"))
    for k, v in THREAD_ENV.items():
        rows.append((f"env {k}", v))
    rows.append(("niceness", "10 (nice -n 10, inherited by the measurement subprocesses)"))
    rows.append(("measurement", "every measurement in a fresh subprocess; machine shared with other agents "
                                "(wall times are upper bounds; see the session rows)"))
    df = pd.DataFrame(rows, columns=["key", "value"])
    df["analysis_status"] = ANALYSIS_STATUS
    return df


# ----------------------------------------------------------------------------
# Template from the real global evidence
# ----------------------------------------------------------------------------
def build_template(out_npz: Path = TEMPLATE_NPZ, out_json: Path = TEMPLATE_JSON) -> dict:
    """Coverage / node-set profiles, conditioning fraction and generating parameters from the
    real primary global evidence; checks the LIT product structure exactly."""
    from r2 import evidence as E
    R, n, fam, ann = E.load_patterns("global", PRIMARY_VARIANT)
    ep = np.asarray(E.load_edge_pattern("global", PRIMARY_VARIANT, mmap=False))
    meta = json.loads((E.EV_DIR / "global" / "meta.json").read_text())
    T0, G0 = int(meta["n_tfs"]), int(meta["n_genes"])
    assert len(ep) == T0 * G0
    K = R.shape[1]
    fam = np.asarray(fam)
    is_lit = fam == "LIT"
    valid = (ep >= 0).reshape(T0, G0)
    safe = np.where(ep >= 0, ep, 0)
    cov = np.zeros((T0, K), bool)
    node = np.zeros((G0, K), bool)
    lit_rep = np.zeros((T0, G0), bool)
    obs_cols = {}
    for j in range(K):
        oj = ((R[:, j] >= 0)[safe].reshape(T0, G0)) & valid
        pj = ((R[:, j] == 1)[safe].reshape(T0, G0)) & valid
        cov[:, j] = oj.any(1)
        node[:, j] = oj.any(0)
        if is_lit[j]:
            prod = cov[:, j][:, None] & node[:, j][None, :]
            n_bad = int(((prod != oj) & valid).sum())
            if n_bad:
                raise AssertionError(f"LIT product structure violated for {ann[j]}: {n_bad} cells")
            lit_rep |= pj
        else:
            obs_cols[j] = oj
    # conditioning-set fraction: P(edge in Cond | >=1 LIT report, TF covered by a MOTIF/COEXP annotator)
    cc = np.flatnonzero(~is_lit)
    in_cond = np.zeros((T0, G0), bool)
    for j in cc:
        if (obs_cols[j] & ~lit_rep).any():
            raise AssertionError(f"{ann[j]} observes an edge without a LIT report")
        in_cond |= obs_cols[j]
    cov_any = cov[:, cc].any(1)
    denom = int((lit_rep & cov_any[:, None]).sum())
    pi_c = float(in_cond.sum() / denom)
    # generating parameters: MAP of the real global family_hb fit
    gm = json.loads((GEN_FIT / "meta.json").read_text())
    if list(gm["annotators"]) != list(ann):
        raise AssertionError("annotator order of the generating fit differs from the evidence")
    gp = np.load(GEN_FIT / "params.npz")
    w = np.asarray(gp["w"], float)
    theta = np.asarray(gp["theta"], float)
    # class weights by observation depth m (number of LIT annotators observing the edge):
    # pattern-weighted mean posterior class probabilities of the real patterns with depth m
    obs_r, pos_r = (R >= 0).astype(float), (R == 1).astype(float)
    lj = np.log(w)[None, :] + obs_r * pos_r @ np.log(theta) + (obs_r * (1 - pos_r)) @ np.log1p(-theta)
    post = np.exp(lj - np.logaddexp.reduce(lj, axis=1)[:, None])

    def _w_by_depth(cols):
        m = (R[:, cols] >= 0).sum(1)
        out = np.tile(w, (len(cols) + 1, 1))
        for d in np.unique(m):
            sel = m == d
            out[d] = (post[sel] * n[sel, None]).sum(0) / n[sel].sum()
        return out
    # evidence_summary rates (template of the 'independent' generator)
    es = pd.read_csv(C.TABLES / "evidence_summary.csv")
    es = es[(es["universe"] == "global") & (es["variant"] == PRIMARY_VARIANT)].set_index("annotator")
    es = es.loc[list(ann)]
    c_rate = (es["n_tfs_covered"] / es["n_tfs"]).to_numpy(float)
    o_rate = (es["n_observed"] / (es["n_tfs_covered"] * es["n_genes"])).to_numpy(float)
    # K = 10 subset (rule fixed in advance): 7 LIT with the largest real coverage + MOTIF/COEXP
    lit_order = es[es["family"] == "LIT"].sort_values("n_observed", ascending=False).index[:K10_N_LIT]
    k10 = [a for a in ann if a in set(lit_order) or es.loc[a, "family"] != "LIT"]
    lit_idx = np.flatnonzero(is_lit)
    w_by_depth = _w_by_depth(lit_idx)
    w_by_depth_k10 = _w_by_depth([ann.index(a) for a in k10 if es.loc[a, "family"] == "LIT"])
    # real anchor statistics
    obs_any = (R >= 0).any(1)
    real = dict(N=int(n.sum()), P=int(R.shape[0]), n_masks=int(len(np.unique(R >= 0, axis=0))),
                frac_edges_unobserved=float(n[~obs_any].sum() / n.sum()),
                mean_annotators_observing=float(((R >= 0).sum(1) * n).sum() / n.sum()),
                pos_rate_observed_cells=float(((R == 1).sum(1) * n).sum() / ((R >= 0).sum(1) * n).sum()),
                n_edges_any_report=int(n[(R == 1).any(1)].sum()),
                coverage=((R >= 0).T.astype(float) @ n / n.sum()).tolist(),
                pos_rate=(((R == 1).T.astype(float) @ n) / np.maximum((R >= 0).T.astype(float) @ n, 1)).tolist(),
                n_distinct_tf_profiles=int(len(np.unique(cov, axis=0))),
                n_distinct_gene_profiles_lit=int(len(np.unique(node[:, is_lit], axis=0))))
    C.ensure_dir(out_npz.parent)
    np.savez_compressed(out_npz, cov=cov, node=node, w=w, theta=theta, c_rate=c_rate, o_rate=o_rate,
                        w_by_depth=w_by_depth, w_by_depth_k10=w_by_depth_k10)
    info = dict(T0=T0, G0=G0, K=K, annotators=list(ann), family=fam.tolist(), class_names=gm["class_names"],
                pi_c=pi_c, n_lit_reported=int(lit_rep.sum()), k10_annotators=k10,
                generating_fit=str(GEN_FIT.relative_to(C.SP)), real=real,
                es_coverage=es["coverage"].tolist(), es_pos_rate=es["pos_rate"].tolist(),
                es_n_tfs_covered=es["n_tfs_covered"].astype(int).tolist())
    C.write_json(info, out_json)
    return info


def load_template() -> tuple[dict, dict]:
    z = np.load(TEMPLATE_NPZ)
    return {k: z[k] for k in z.files}, json.loads(TEMPLATE_JSON.read_text())


# ----------------------------------------------------------------------------
# Configurations
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class Config:
    generator: str      # 'template' | 'independent'
    size: str           # '1e5' | '1e6' | '1e7' | 'real' | '10xreal'
    K: int

    @property
    def cid(self) -> str:
        return f"{self.generator}_{self.size}_K{self.K}"

    def shape(self, T0: int, G0: int) -> tuple[int, int]:
        """(T, G): TFs and genes scaled by sqrt(N / N_real); N = T * G."""
        if self.size == "real":
            return T0, G0
        if self.size == "10xreal":
            return int(round(T0 * math.sqrt(10))), int(round(G0 * math.sqrt(10)))
        N = float(self.size)
        T = max(1, int(round(T0 * math.sqrt(N / (T0 * G0)))))
        return T, int(math.ceil(N / T))


def config_grid() -> list[Config]:
    out = [Config("template", s, k) for s in ("1e5", "1e6", "1e7") for k in KS]
    out += [Config("template", "real", 21), Config("template", "10xreal", 21), Config("template", "10xreal", 42)]
    out += [Config("independent", s, k) for s in ("1e5", "1e6", "1e7") for k in KS]
    out += [Config("independent", "real", 21)]
    out += [Config("stress", s, 21) for s in ("1e5", "1e6", "1e7")] + [Config("stress", "1e6", 42)]
    return out


DENSE_GRID = [Config("template", s, 21) for s in ("1e5", "1e6", "1e7", "real")]


def config_from_cid(cid: str) -> Config:
    gen, size, k = cid.split("_")
    return Config(gen, size, int(k[1:]))


# ----------------------------------------------------------------------------
# Synthetic TF-blocked evidence
# ----------------------------------------------------------------------------
class Generator:
    """Streams a synthetic T x G evidence grid (TF-major) in whole-TF chunks."""

    def __init__(self, cfg: Config, tpl: dict, info: dict, seed: int = SEED):
        self.cfg = cfg
        T0, G0 = info["T0"], info["G0"]
        self.T, self.G = cfg.shape(T0, G0)
        self.N = self.T * self.G
        ann = list(info["annotators"])
        fam = list(info["family"])
        if cfg.K == len(ann):
            blocks = [list(range(len(ann)))]
        elif cfg.K == 10:
            blocks = [[ann.index(a) for a in info["k10_annotators"]]]
        elif cfg.K == 2 * len(ann):
            blocks = [list(range(len(ann)))] * 2
        else:
            raise ValueError(f"unsupported K = {cfg.K}")
        if sum(len(b) for b in blocks) != cfg.K:
            raise AssertionError("annotator blocks do not add up to K")
        self.crc = zlib.crc32(cfg.cid.encode())
        self.seed = seed
        rng = np.random.default_rng([seed, self.crc, 0])
        covs, nodes, names, fams, thetas = [], [], [], [], []
        for bi, b in enumerate(blocks):
            if cfg.generator == "template":
                tf_rows = rng.integers(0, T0, size=self.T)
                gene_rows = rng.integers(0, G0, size=self.G)
                covs.append(tpl["cov"][np.ix_(tf_rows, b)])
                nodes.append(tpl["node"][np.ix_(gene_rows, b)])
            elif cfg.generator == "independent":
                covs.append(rng.random((self.T, len(b))) < tpl["c_rate"][b][None, :])
                nodes.append(rng.random((self.G, len(b))) < tpl["o_rate"][b][None, :])
            elif cfg.generator == "stress":
                covs.append(np.ones((self.T, len(b)), bool))
                nodes.append(np.ones((self.G, len(b)), bool))
            else:
                raise ValueError(cfg.generator)
            suffix = "" if len(blocks) == 1 else f"_{'AB'[bi]}"
            names += [ann[j] + suffix for j in b]
            fams += [fam[j] for j in b]
            thetas.append(tpl["theta"][b])
        self.cov = np.ascontiguousarray(np.hstack(covs))
        self.node = np.ascontiguousarray(np.hstack(nodes))
        self.annotators, self.family = names, fams
        self.theta = np.vstack(thetas)
        self.w = np.asarray(tpl["w"], float)
        wd = tpl["w_by_depth_k10"] if cfg.K == 10 else tpl["w_by_depth"]
        self.cum_wd = np.cumsum(wd / wd.sum(1, keepdims=True), axis=1)
        self.n_blocks = len(blocks)
        self.pi_c = float(info["pi_c"])
        self.lit = np.flatnonzero(np.asarray(fams) == "LIT")
        self.cc = np.flatnonzero(np.asarray(fams) != "LIT")
        self.tfs_per_chunk = max(1, CHUNK_EDGES // self.G)

    def chunks(self):
        """Yield (t0, t1, block) with block the (t1 - t0) * G x K int8 evidence (TF-major)."""
        for ci, t0 in enumerate(range(0, self.T, self.tfs_per_chunk)):
            t1 = min(self.T, t0 + self.tfs_per_chunk)
            yield t0, t1, self.block(t0, t1, ci)

    def block(self, t0: int, t1: int, ci: int) -> np.ndarray:
        rng = np.random.default_rng([self.seed, self.crc, 1, ci])
        G, K = self.G, self.cfg.K
        n = (t1 - t0) * G
        if self.cfg.generator == "stress":
            z = np.minimum(np.searchsorted(np.cumsum(self.w / self.w.sum()), rng.random(n), side="right"),
                           len(self.w) - 1)
            M = np.full((n, K), -1, dtype=np.int8)
            for j in range(K):
                idx = np.flatnonzero(rng.random(n) < STRESS_OBS)
                M[idx, j] = rng.random(len(idx)) < self.theta[j, z[idx]]
            return M
        obs_lit = np.empty((n, len(self.lit)), dtype=bool)
        for k, j in enumerate(self.lit):
            obs_lit[:, k] = (self.cov[t0:t1, j][:, None] & self.node[:, j][None, :]).ravel()
        # latent class given the observation depth m (mean over the blocks when K = 42)
        m = obs_lit.sum(1)
        if self.n_blocks > 1:
            m = np.rint(m / self.n_blocks).astype(np.int64)
        u = rng.random(n)
        z = np.minimum((u[:, None] >= self.cum_wd[m]).sum(1), self.cum_wd.shape[1] - 1)
        M = np.full((n, K), -1, dtype=np.int8)
        lit_rep = np.zeros(n, dtype=bool)
        for k, j in enumerate(self.lit):
            idx = np.flatnonzero(obs_lit[:, k])
            rep = rng.random(len(idx)) < self.theta[j, z[idx]]
            M[idx, j] = rep
            lit_rep[idx[rep]] = True
        cond = np.flatnonzero(lit_rep)
        cond = cond[rng.random(len(cond)) < self.pi_c]
        tf_of = cond // G
        for j in self.cc:
            idx = cond[self.cov[t0:t1, j][tf_of]]
            M[idx, j] = rng.random(len(idx)) < self.theta[j, z[idx]]
        return M

    def dense(self) -> np.ndarray:
        M = np.empty((self.N, self.cfg.K), dtype=np.int8)
        for t0, t1, blk in self.chunks():
            M[t0 * self.G:t1 * self.G] = blk
        return M


# ----------------------------------------------------------------------------
# Streaming pattern compression
# ----------------------------------------------------------------------------
def n_words(K: int) -> int:
    return -(-K // TRITS)


def encode(block: np.ndarray) -> np.ndarray:
    """(n, K) {-1,0,1} -> (n, W) int64 base-3 words (column j -> word j // 39, digit j % 39).
    For K <= 39 the single word equals ``evidence._codes`` (same pattern order)."""
    n, K = block.shape
    out = np.zeros((n, n_words(K)), dtype=np.int64)
    for j in range(K):
        w, p = divmod(j, TRITS)
        out[:, w] += (block[:, j].astype(np.int64) + 1) * np.int64(3 ** p)
    return out


def decode(words: np.ndarray, K: int) -> np.ndarray:
    pat = np.empty((len(words), K), dtype=np.int8)
    for w in range(words.shape[1]):
        c = words[:, w].copy()
        for p in range(min(TRITS, K - w * TRITS)):
            pat[:, w * TRITS + p] = (c % 3) - 1
            c //= 3
    return pat


def unique_rows(words: np.ndarray):
    """Sorted unique rows of an (n, W) int64 array with inverse and counts."""
    if words.shape[1] == 1:
        u, inv, cnt = np.unique(words[:, 0], return_inverse=True, return_counts=True)
        return u[:, None], inv.reshape(-1), cnt
    order = np.lexsort(words.T[::-1])
    s = words[order]
    new = np.ones(len(s), dtype=bool)
    new[1:] = (s[1:] != s[:-1]).any(1)
    gid = np.cumsum(new) - 1
    inv = np.empty(len(s), dtype=np.int64)
    inv[order] = gid
    return s[new], inv, np.bincount(gid)


class StreamingCompressor:
    """Chunk-wise pattern compression into a global int32 edge -> pattern index.

    ``add`` stores each chunk's local inverse in the index and keeps only the chunk's unique
    rows; ``finish`` merges the chunk dictionaries, sums the counts and remaps the index."""

    def __init__(self, N: int, K: int):
        self.N, self.K = N, K
        self.ep = np.empty(N, dtype=np.int32)
        self.off = 0
        self.uniq, self.counts, self.spans = [], [], []
        self.t = dict(encode=0.0, unique=0.0, merge=0.0, remap=0.0)
        self.cpu = 0.0

    def add(self, block: np.ndarray) -> None:
        c0 = time.process_time()
        t0 = time.perf_counter()
        words = encode(block)
        t1 = time.perf_counter()
        u, inv, cnt = unique_rows(words)
        n = len(block)
        self.ep[self.off:self.off + n] = inv
        t2 = time.perf_counter()
        self.spans.append((self.off, n, len(u)))
        self.uniq.append(u)
        self.counts.append(cnt)
        self.off += n
        self.t["encode"] += t1 - t0
        self.t["unique"] += t2 - t1
        self.cpu += time.process_time() - c0

    def finish(self):
        if self.off != self.N:
            raise AssertionError(f"{self.off} of {self.N} edges added")
        c0 = time.process_time()
        t0 = time.perf_counter()
        allu = np.concatenate(self.uniq)
        gu, ginv, _ = unique_rows(allu)
        counts = np.bincount(ginv, weights=np.concatenate(self.counts), minlength=len(gu)).astype(np.int64)
        t1 = time.perf_counter()
        base = 0
        for off, n, pc in self.spans:
            m = ginv[base:base + pc].astype(np.int32)
            self.ep[off:off + n] = m[self.ep[off:off + n]]
            base += pc
        t2 = time.perf_counter()
        self.t["merge"] += t1 - t0
        self.t["remap"] += t2 - t1
        self.cpu += time.process_time() - c0
        self.max_chunk_patterns = int(max(s[2] for s in self.spans))
        self.sum_chunk_patterns = int(base)
        return decode(gu, self.K), counts, self.ep


def compress_dense(M: np.ndarray):
    """Whole-matrix compression exactly as ``lcm.PatternData.from_matrix`` does it."""
    uniq, inv, counts = np.unique(M, axis=0, return_inverse=True, return_counts=True)
    return uniq, counts, inv.reshape(-1)


def pattern_stats(R: np.ndarray, n: np.ndarray) -> dict:
    n = np.asarray(n, float)
    obs = R >= 0
    pos = R == 1
    n_obs_cells = float((obs.sum(1) * n).sum())
    return dict(P=int(len(R)), n_masks=int(len(np.unique(obs, axis=0))),
                frac_edges_unobserved=float(n[~obs.any(1)].sum() / n.sum()),
                mean_annotators_observing=float((obs.sum(1) * n).sum() / n.sum()),
                pos_rate_observed_cells=float((pos.sum(1) * n).sum() / max(n_obs_cells, 1.0)),
                n_edges_any_report=int(n[pos.any(1)].sum()),
                coverage=(obs.T.astype(float) @ n / n.sum()).tolist(),
                pos_rate=((pos.T.astype(float) @ n) / np.maximum(obs.T.astype(float) @ n, 1)).tolist())


# ----------------------------------------------------------------------------
# Workers (each runs in a fresh subprocess)
# ----------------------------------------------------------------------------
def _run_json(stage: str, rid: str, session: str = PRIMARY_SESSION) -> Path:
    return run_dir(session) / stage / f"{rid}.json"


def same_patterns(R1: np.ndarray, n1: np.ndarray, R2: np.ndarray, n2: np.ndarray) -> bool:
    """True if two pattern tables hold the same rows with the same counts (any row order)."""
    if R1.shape != R2.shape:
        return False
    w1, w2 = encode(np.asarray(R1, np.int8)), encode(np.asarray(R2, np.int8))
    o1, o2 = np.lexsort(w1.T[::-1]), np.lexsort(w2.T[::-1])
    return bool(np.array_equal(w1[o1], w2[o2]) and
                np.array_equal(np.asarray(n1, np.int64)[o1], np.asarray(n2, np.int64)[o2]))


def worker_compress(cid: str, dense: bool = False, session: str = PRIMARY_SESSION) -> dict:
    """Generate and compress one configuration.  The primary session stores the pattern table
    (``patterns/<cid>.npz``, the input of the MAP / NUTS workers of every session); any other
    session, and every whole-matrix run, compares its result with the stored table
    (``patterns_identical``: same rows and counts) and leaves the stored table untouched."""
    cfg = config_from_cid(cid)
    tpl, info = load_template()
    gen = Generator(cfg, tpl, info)
    rss_base = Monitor().start().stop()["rss_start_mib"]
    out = dict(cid=cid, session=session, method="dense_unique" if dense else "chunked_stream", generator=cfg.generator,
               size=cfg.size, K=cfg.K, T=gen.T, G=gen.G, N=gen.N, chunk_edges_max=gen.tfs_per_chunk * gen.G,
               tfs_per_chunk=gen.tfs_per_chunk, n_chunks=-(-gen.T // gen.tfs_per_chunk),
               rss_baseline_mib=rss_base)
    mon = Monitor().start()
    t_gen = 0.0
    if not dense:
        comp = StreamingCompressor(gen.N, cfg.K)
        first = None
        it = gen.chunks()
        while True:
            t0 = time.perf_counter()
            try:
                c0, c1, blk = next(it)
            except StopIteration:
                break
            t_gen += time.perf_counter() - t0
            if first is None:
                first = (c0, c1, blk.copy())
            comp.add(blk)
            del blk
        R, n, ep = comp.finish()
        # verification: the first chunk is reconstructed exactly; counts add up to N
        c0, c1, blk = first
        assert np.array_equal(R[ep[c0 * gen.G:c1 * gen.G]], blk)
        assert int(n.sum()) == gen.N and int(ep.min()) >= 0
        out.update(t_encode_s=comp.t["encode"], t_unique_s=comp.t["unique"], t_merge_s=comp.t["merge"],
                   t_remap_s=comp.t["remap"], t_compress_s=sum(comp.t.values()), cpu_compress_s=comp.cpu,
                   max_chunk_patterns=comp.max_chunk_patterns, sum_chunk_patterns=comp.sum_chunk_patterns,
                   edge_index_mib=ep.nbytes / 2**20)
    else:
        t0 = time.perf_counter()
        M = gen.dense()
        t_gen = time.perf_counter() - t0
        out["dense_matrix_mib"] = M.nbytes / 2**20
        c0 = time.process_time()
        t0 = time.perf_counter()
        R, n, ep = compress_dense(M)
        out.update(t_compress_s=time.perf_counter() - t0, cpu_compress_s=time.process_time() - c0,
                   edge_index_mib=ep.nbytes / 2**20)
        assert int(n.sum()) == gen.N
    res = mon.stop()
    out.update(t_generate_s=t_gen, wall_total_s=res["wall_s"], cpu_total_s=res["cpu_s"],
               peak_rss_sampled_mib=res["peak_rss_sampled_mib"], ru_maxrss_mib=process_peak_mib(),
               load1_start=res["load1_start"], load5_start=res["load5_start"], load15_start=res["load15_start"],
               load1_mean=res["load1_mean"], load1_max=res["load1_max"], load1_end=res["load1_end"])
    out["peak_rss_mib"] = max(out["peak_rss_sampled_mib"], out["ru_maxrss_mib"])
    out["pattern_table_kib"] = R.nbytes / 2**10
    out.update(pattern_stats(R, n))
    out["annotators"] = gen.annotators
    out["family"] = gen.family
    stored = PAT_DIR / f"{cid}.npz"
    if not dense and session == PRIMARY_SESSION:
        C.ensure_dir(PAT_DIR)
        np.savez_compressed(stored, R=R, n=n, family=np.array(gen.family),
                            annotators=np.array(gen.annotators))
    elif stored.exists():
        # outside the monitored stage: reproducibility of the stored pattern table
        z = np.load(stored)
        out["patterns_identical"] = same_patterns(R, n, z["R"], z["n"])
    return out


def load_pattern_data(cid: str):
    from r2 import lcm
    z = np.load(PAT_DIR / f"{cid}.npz")
    return lcm.PatternData(z["R"], z["n"], list(z["family"]), list(z["annotators"]))


def _times(fn, x_list) -> np.ndarray:
    ts = []
    for x in x_list:
        t0 = time.perf_counter()
        fn(x)
        ts.append(time.perf_counter() - t0)
    return np.asarray(ts)


def _median_time(fn, x_list) -> float:
    return float(np.median(_times(fn, x_list)))


def timed_map(data, model_name: str = MODEL, n_restarts: int = N_RESTARTS, seed: int = SEED,
              budget_s: float | None = None, init_radius: float = 2.0, gtol: float = 1e-6,
              maxiter: int = 20000, n_bench: int = N_BENCH, counts_factor: float | None = 100.0,
              nfev_guess: float | None = None, eval_only: bool = False) -> dict:
    """``lcm.fit_map`` with per-stage instrumentation (identical restarts: same keys, starting
    points, objective closures and L-BFGS options), plus per-evaluation timings.

    Budget rule (``budget_s``): restarts stop after the one that crosses ``budget_s`` (at least
    2 are run) and the 10-restart time is extrapolated; if a single restart is projected to
    exceed ``budget_s`` (``nfev_guess`` evaluations x measured evaluation time) no restart is
    run and the MAP time is projected.  ``eval_only`` forces the last case (per-evaluation
    timings only; needs ``nfev_guess``).

    Per-evaluation timings are medians over ``n_bench`` evaluations at random points; the minima
    (``*_min_s``, the usual estimate of the undisturbed cost on a shared machine) are recorded
    too.  ``counts_invariance_ratio_paired`` is the median over evaluation points of
    time(counts x factor) / time(counts), the two evaluations of a point made back to back."""
    if eval_only and not nfev_guess:
        raise ValueError("eval_only needs nfev_guess")
    import jax
    import jax.numpy as jnp
    import numpyro
    from jax.flatten_util import ravel_pytree
    from numpyro.infer.util import initialize_model
    from scipy.optimize import minimize
    from r2 import lcm

    spec = lcm.MODEL_SPECS[model_name]
    out = dict(model=model_name)
    with Monitor() as m_build:
        model = lcm.build_model(data, spec)
        rng = jax.random.PRNGKey(seed)
        init = initialize_model(rng, model, init_strategy=numpyro.infer.init_to_uniform(radius=init_radius))
    out.update(S=len(model.class_names), class_names=" ".join(model.class_names),
               t_build_init_s=m_build.result["wall_s"])
    potential_fn = init.potential_fn
    postprocess = init.postprocess_fn
    flat0, unravel = ravel_pytree(init.param_info.z)
    out["n_params"] = int(flat0.shape[0])
    pot = jax.jit(lambda x: potential_fn(unravel(x)))
    grad = jax.jit(jax.grad(lambda x: potential_fn(unravel(x))))
    vg = jax.jit(jax.value_and_grad(lambda x: potential_fn(unravel(x))))
    with Monitor() as m_comp:
        x = jnp.asarray(flat0)
        pot(x).block_until_ready()
        grad(x).block_until_ready()
    out["t_compile_s"] = m_comp.result["wall_s"]
    with Monitor() as m_vgc:
        jax.block_until_ready(vg(jnp.asarray(flat0)))
    out["t_compile_vg_s"] = m_vgc.result["wall_s"]
    bk = jax.random.split(jax.random.PRNGKey(seed + 1), n_bench)
    xs = [np.asarray(jax.random.uniform(k, flat0.shape, minval=-init_radius, maxval=init_radius)) for k in bk]
    f = lambda x: float(pot(jnp.asarray(x)))                       # noqa: E731  (as in lcm.fit_map)
    jac = lambda x: np.asarray(grad(jnp.asarray(x)))               # noqa: E731
    with Monitor() as m_bench:
        out["t_pot_s"] = _median_time(f, xs)
        out["t_grad_s"] = _median_time(jac, xs)
        ts = _times(lambda x: (f(x), jac(x)), xs)
        out["t_eval_lbfgs_s"], out["t_eval_lbfgs_min_s"] = float(np.median(ts)), float(ts.min())
        ts = _times(lambda x: jax.block_until_ready(vg(jnp.asarray(x))), xs)
        out["t_vg_s"], out["t_vg_min_s"] = float(np.median(ts)), float(ts.min())
    out["bench_load1_mean"] = m_bench.result["load1_mean"]
    if counts_factor:
        # same patterns, counts x factor (N x factor): the evaluation cost must not change
        d2 = lcm.PatternData(data.R, data.n * counts_factor, data.family, data.annotators)
        init2 = initialize_model(rng, lcm.build_model(d2, spec),
                                 init_strategy=numpyro.infer.init_to_uniform(radius=init_radius))
        f2, unr2 = ravel_pytree(init2.param_info.z)
        vg2 = jax.jit(jax.value_and_grad(lambda x: init2.potential_fn(unr2(x))))
        jax.block_until_ready(vg2(jnp.asarray(f2)))
        out["counts_factor"] = counts_factor
        out["t_vg_counts_x_factor_s"] = _median_time(lambda x: jax.block_until_ready(vg2(jnp.asarray(x))), xs)
        out["t_vg_repeat_s"] = _median_time(lambda x: jax.block_until_ready(vg(jnp.asarray(x))), xs)
        # paired (back-to-back) evaluations: robust to load drift between the two blocks above
        ratios = []
        for x in xs:
            xj = jnp.asarray(x)
            t0 = time.perf_counter()
            jax.block_until_ready(vg(xj))
            t1 = time.perf_counter()
            jax.block_until_ready(vg2(xj))
            ratios.append((time.perf_counter() - t1) / max(t1 - t0, 1e-12))
        out["counts_invariance_ratio_paired"] = float(np.median(ratios))
    keys = jax.random.split(rng, n_restarts)
    restarts = []
    if eval_only or (budget_s is not None and nfev_guess and nfev_guess * out["t_eval_lbfgs_s"] > budget_s):
        out.update(n_restarts_target=n_restarts, n_restarts_run=0, nfev_guess=float(nfev_guess),
                   eval_only=bool(eval_only),
                   t_restart_mean_s=nfev_guess * out["t_eval_lbfgs_s"],
                   t_map_10_s=out["t_build_init_s"] + out["t_compile_s"] + n_restarts * nfev_guess * out["t_eval_lbfgs_s"],
                   t_map_10_status="projected", restarts=[], load1_mean=m_bench.result["load1_mean"],
                   load1_max=m_bench.result["load1_max"], load1_start=m_bench.result["load1_start"])
        return out
    mon = Monitor().start()
    for r in range(n_restarts):
        c0 = time.process_time()
        t0 = time.perf_counter()
        x0 = np.asarray(jax.random.uniform(keys[r], flat0.shape, minval=-init_radius, maxval=init_radius))
        res = minimize(f, x0, jac=jac, method="L-BFGS-B",
                       options=dict(maxiter=maxiter, gtol=gtol, ftol=1e-15, maxcor=30))
        t_opt = time.perf_counter() - t0
        z = unravel(jnp.asarray(res.x))
        sample = {k: np.asarray(v) for k, v in postprocess(z).items()}
        ll, _, _ = lcm._collapsed_loglik_and_cred(data, spec, sample)
        gnorm = float(np.max(np.abs(jac(res.x))))
        restarts.append(dict(restart=r, log_post=-float(res.fun), loglik=ll, converged=bool(res.success),
                             n_iter=int(res.nit), nfev=int(res.nfev), grad_norm=gnorm, t_opt_s=t_opt,
                             t_restart_s=time.perf_counter() - t0, cpu_restart_s=time.process_time() - c0))
        if budget_s is not None and r + 1 >= 2 and r + 1 < n_restarts and \
                time.perf_counter() - mon.t0 > budget_s:
            break
    res_m = mon.stop()
    rs = pd.DataFrame(restarts)
    n_run = len(rs)
    out.update(n_restarts_target=n_restarts, n_restarts_run=n_run,
               t_restarts_wall_s=res_m["wall_s"], cpu_restarts_s=res_m["cpu_s"],
               t_restart_mean_s=float(rs["t_restart_s"].mean()), t_restart_max_s=float(rs["t_restart_s"].max()),
               nfev_mean=float(rs["nfev"].mean()), nfev_max=int(rs["nfev"].max()),
               n_iter_mean=float(rs["n_iter"].mean()), n_converged=int(rs["converged"].sum()),
               t_per_lbfgs_eval_s=float(rs["t_opt_s"].sum() / rs["nfev"].sum()),
               best_log_post=float(rs["log_post"].max()),
               n_within_1e3_of_best=int((rs["log_post"] >= rs["log_post"].max() - 1e-3).sum()),
               load1_mean=res_m["load1_mean"], load1_max=res_m["load1_max"], load1_start=res_m["load1_start"],
               load5_start=res_m["load5_start"], load15_start=res_m["load15_start"],
               peak_rss_restarts_mib=res_m["peak_rss_sampled_mib"])
    if n_run == n_restarts:
        out["t_map_10_s"] = out["t_build_init_s"] + out["t_compile_s"] + res_m["wall_s"]
        out["t_map_10_status"] = "measured"
    else:
        out["t_map_10_s"] = out["t_build_init_s"] + out["t_compile_s"] + n_restarts * out["t_restart_mean_s"]
        out["t_map_10_status"] = "extrapolated"
    out["restarts"] = restarts
    return out


def worker_map(cid: str, budget_s: float | None = MAP_BUDGET_S, nfev_guess: float | None = None,
               eval_only: bool = False, session: str = PRIMARY_SESSION) -> dict:
    data = load_pattern_data(cid)
    out = dict(cid=cid, session=session, P=data.P, K=data.K, N=int(data.n.sum()))
    mon = Monitor().start()
    out.update(timed_map(data, budget_s=budget_s, nfev_guess=nfev_guess, eval_only=eval_only))
    res = mon.stop()
    out.update(wall_total_s=res["wall_s"], cpu_total_s=res["cpu_s"],
               peak_rss_sampled_mib=res["peak_rss_sampled_mib"], ru_maxrss_mib=process_peak_mib())
    out["peak_rss_mib"] = max(out["peak_rss_sampled_mib"], out["ru_maxrss_mib"])
    out["model_arrays_mib"] = 2 * data.P * data.K * 8 / 2**20     # float64 observed + reported matrices
    return out


def nuts_diagnostics(s_chain: dict) -> dict:
    """Worst-case R-hat and bulk / tail ESS over the posterior sites, with the
    ``lcm.fit_nuts`` rule (every site except 'theta' and 'cred')."""
    import arviz as az
    rh, eb, et = [], [], []
    for k, v in s_chain.items():
        if k in ("cred", "theta"):
            continue
        arr = np.asarray(v)
        arr = arr.reshape(arr.shape[0], arr.shape[1], -1)
        ds = az.convert_to_dataset({k: arr})
        rh.append(float(np.nanmax(az.rhat(ds)[k].values)))
        eb.append(float(np.nanmin(az.ess(ds, method="bulk")[k].values)))
        et.append(float(np.nanmin(az.ess(ds, method="tail")[k].values)))
    if not rh:
        return dict(r_hat_max=np.nan, ess_bulk_min=np.nan, ess_tail_min=np.nan)
    return dict(r_hat_max=max(rh), ess_bulk_min=min(eb), ess_tail_min=min(et))


def timed_nuts(data, model_name: str = MODEL, chains: int = NUTS_CHAINS, warmup: int = NUTS_WARMUP,
               samples: int = NUTS_SAMPLES, seed: int = SEED, target_accept: float = TARGET_ACCEPT,
               probe: bool = True) -> dict:
    """NUTS with the ``lcm.fit_nuts`` kernel and settings except that the per-pattern
    credibility deterministic site is hidden (memory O(P K) instead of O(draws P)); the
    credibility posterior is computed afterwards with ``lcm.nuts_credibility``.

    Compile cost: every MCMC object compiles its own warm-up and sampling kernels, so the timed
    run (``t_nuts_s``) includes that compilation.  ``probe`` runs two throw-away MCMC objects
    with the structure of the timed run (same chains, ``warmup()`` then ``run()``) but 1 warm-up
    iteration and 1 draw per chain: the first is the cold start (``t_probe_cold_s``), the second
    (``t_probe_s``) estimates the compile time contained in ``t_nuts_s``
    (``probe_kind`` = 'main_like_warm').  Session-a records were made with an earlier probe
    (1 chain, ``run()`` only, cold; no ``probe_kind`` field)."""
    import jax
    from numpyro.handlers import block
    from numpyro.infer import MCMC, NUTS
    from types import SimpleNamespace
    from r2 import lcm

    spec = lcm.MODEL_SPECS[model_name]
    base = lcm.build_model(data, spec)
    model = block(base, hide=["cred"])
    out = dict(model=model_name, S=len(base.class_names), chains=chains, warmup=warmup, samples=samples,
               target_accept=target_accept, chain_method="sequential")
    if probe:
        t_pr, steps_pr = [], []
        for i in range(2):
            pm = MCMC(NUTS(model, target_accept_prob=target_accept), num_warmup=1, num_samples=1,
                      num_chains=chains, chain_method="sequential", progress_bar=False)
            with Monitor() as m_probe:
                pm.warmup(jax.random.PRNGKey(seed + 7 + i), extra_fields=("num_steps",), collect_warmup=True)
                n_w = int(np.asarray(pm.get_extra_fields()["num_steps"]).sum())
                pm.run(pm.post_warmup_state.rng_key, extra_fields=("num_steps",))
            t_pr.append(m_probe.result["wall_s"])
            steps_pr.append(n_w + int(np.asarray(pm.get_extra_fields()["num_steps"]).sum()))
            del pm
        out.update(t_probe_cold_s=t_pr[0], t_probe_s=t_pr[1], probe_steps=steps_pr[1],
                   probe_kind="main_like_warm")
    mcmc = MCMC(NUTS(model, target_accept_prob=target_accept), num_warmup=warmup, num_samples=samples,
                num_chains=chains, chain_method="sequential", progress_bar=False)
    with Monitor() as m_w:
        mcmc.warmup(jax.random.PRNGKey(seed), extra_fields=("num_steps", "diverging"), collect_warmup=True)
    ws = np.asarray(mcmc.get_extra_fields(group_by_chain=True)["num_steps"])
    with Monitor() as m_s:
        mcmc.run(mcmc.post_warmup_state.rng_key, extra_fields=("num_steps", "diverging"))
    ef = mcmc.get_extra_fields(group_by_chain=True)
    ss = np.asarray(ef["num_steps"])
    draws = mcmc.get_samples()
    if "cred" in draws:
        raise AssertionError("credibility site was not hidden")
    # convergence of the short chains (as lcm.fit_nuts: sampled sites, not theta / cred)
    out.update(nuts_diagnostics(mcmc.get_samples(group_by_chain=True)))
    steps = int(ws.sum() + ss.sum())
    t_total = m_w.result["wall_s"] + m_s.result["wall_s"]
    out.update(t_warmup_s=m_w.result["wall_s"], t_sampling_s=m_s.result["wall_s"], t_nuts_s=t_total,
               cpu_nuts_s=m_w.result["cpu_s"] + m_s.result["cpu_s"],
               steps_warmup=int(ws.sum()), steps_sampling=int(ss.sum()), steps_total=steps,
               steps_per_iter_mean=steps / (chains * (warmup + samples)),
               steps_per_iter_sampling_mean=float(ss.mean()), steps_per_iter_max=int(max(ws.max(), ss.max())),
               n_divergent=int(np.asarray(ef["diverging"]).sum()),
               # t_nuts_s includes the JIT compilation of this MCMC object's kernels (about
               # t_probe_s), so t_nuts_s / steps is an upper bound of the time per leapfrog step;
               # the tables also give (t_nuts_s - t_probe_s) / steps
               t_per_step_s=t_total / steps, t_per_step_sampling_s=m_s.result["wall_s"] / max(int(ss.sum()), 1),
               steps_warmup_by_chain=ws.sum(1).astype(int).tolist(),
               steps_sampling_by_chain=ss.sum(1).astype(int).tolist(),
               peak_rss_nuts_mib=max(m_w.result["peak_rss_sampled_mib"], m_s.result["peak_rss_sampled_mib"]),
               load1_mean=float(np.mean([m_w.result["load1_mean"], m_s.result["load1_mean"]])),
               load1_max=max(m_w.result["load1_max"], m_s.result["load1_max"]),
               load1_start=m_w.result["load1_start"], load5_start=m_w.result["load5_start"],
               load15_start=m_w.result["load15_start"],
               n_draws=int(np.asarray(draws["w"]).shape[0]), n_param_sites=len(draws))
    fit = SimpleNamespace(samples={"w": np.asarray(draws["w"]), "theta": np.asarray(draws["theta"])})
    with Monitor() as m_c:
        cs = lcm.nuts_credibility(fit, data)
    out.update(t_cred_posthoc_s=m_c.result["wall_s"],
               peak_rss_cred_posthoc_mib=m_c.result["peak_rss_sampled_mib"],
               cred_draws_matrix_mib=out["n_draws"] * data.P * 8 / 2**20,
               cred_mean_finite=bool(np.isfinite(cs["mean"]).all()))
    return out


def worker_nuts(cid: str, session: str = PRIMARY_SESSION) -> dict:
    data = load_pattern_data(cid)
    out = dict(cid=cid, session=session, P=data.P, K=data.K, N=int(data.n.sum()))
    mon = Monitor().start()
    out.update(timed_nuts(data))
    res = mon.stop()
    out.update(wall_total_s=res["wall_s"], cpu_total_s=res["cpu_s"],
               peak_rss_sampled_mib=res["peak_rss_sampled_mib"], ru_maxrss_mib=process_peak_mib())
    out["peak_rss_mib"] = max(out["peak_rss_sampled_mib"], out["ru_maxrss_mib"])
    return out


# ----------------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------------
def _spawn(args: list, log: Path, timeout_s: float | None = None) -> int:
    """Run a worker sub-command in a fresh subprocess; returns its exit code (-9 on timeout)."""
    env = dict(os.environ)
    env.update(THREAD_ENV)
    C.ensure_dir(log.parent)
    with open(log, "a") as fh:
        fh.write(f"\n=== {time.strftime('%Y-%m-%d %H:%M:%S')} {' '.join(args)}\n")
        fh.flush()
        try:
            p = subprocess.run([sys.executable, str(Path(__file__).resolve())] + args, env=env, stdout=fh,
                               stderr=subprocess.STDOUT, timeout=timeout_s)
        except subprocess.TimeoutExpired:
            fh.write(f"=== timeout after {timeout_s:.0f} s\n")
            return -9
    return p.returncode


def _read(stage: str, rid: str, session: str = PRIMARY_SESSION) -> dict | None:
    p = _run_json(stage, rid, session)
    return json.loads(p.read_text()) if p.exists() else None


def run_stage_compress(force: bool = False, session: str = PRIMARY_SESSION) -> None:
    jobs = [(c.cid, False) for c in config_grid()] + [(c.cid, True) for c in DENSE_GRID]
    # smallest first
    order = {s: i for i, s in enumerate(SIZES)}
    jobs.sort(key=lambda j: (order[config_from_cid(j[0]).size], config_from_cid(j[0]).K, j[1]))
    for cid, dense in jobs:
        rid = cid + ("__dense" if dense else "")
        if not force and _run_json("compress", rid, session).exists():
            continue
        print(f"[compress:{session}] {rid}", flush=True)
        rc = _spawn(["worker-compress", "--cid", cid, "--session", session] + (["--dense"] if dense else []),
                    log_dir(session) / "compress.log")
        if rc != 0:
            print(f"  failed (rc={rc}); see {log_dir(session).name}/compress.log", flush=True)


def _stored_P() -> dict:
    """P of every configuration with a stored pattern table (compression of the primary session)."""
    out = {}
    for c in config_grid():
        r = _read("compress", c.cid)
        if r:
            out[c.cid] = r["P"]
    return out


def run_stage_map(force: bool = False, session: str = PRIMARY_SESSION) -> None:
    """MAP timings, smallest P first.

    Primary session: ``nfev_guess`` for the single-restart projection is the largest mean number
    of L-BFGS evaluations among completed runs (default 1500).  Replicate session: the restarts
    are re-run for the configurations whose 10 restarts were measured within
    ``REPLICATE_MAP_MAX_S`` in the primary session; the others get the per-evaluation timings only (``--eval-only``) and their MAP time is projected
    with the primary session's mean evaluation count per restart (the optimiser path does not
    depend on the load)."""
    P = _stored_P()
    for cid in sorted(P, key=lambda c: (P[c], c)):
        if not force and _run_json("map", cid, session).exists():
            continue
        args = ["worker-map", "--cid", cid, "--session", session]
        if session == PRIMARY_SESSION:
            done = [r for r in (_read("map", c) for c in P) if r and r.get("n_restarts_run", 0) > 0]
            nfev_guess = max([r["nfev_mean"] for r in done], default=1500.0)
            note = ""
        else:
            ra = _read("map", cid)
            if ra is None:
                print(f"[map:{session}] {cid}: no primary-session MAP record, skipped", flush=True)
                continue
            nfev_guess = ra["nfev_mean"] if ra.get("n_restarts_run", 0) > 0 else ra["nfev_guess"]
            note = ""
            if not (ra.get("t_map_10_status") == "measured" and ra["t_map_10_s"] <= REPLICATE_MAP_MAX_S):
                args.append("--eval-only")
                note = ", evaluations only"
        print(f"[map:{session}] {cid} P={P[cid]} (nfev_guess {nfev_guess:.0f}{note})", flush=True)
        rc = _spawn(args + ["--nfev-guess", f"{nfev_guess:.1f}"], log_dir(session) / "map.log")
        if rc != 0:
            print(f"  failed (rc={rc}); see {log_dir(session).name}/map.log", flush=True)


def nuts_projection(cid: str, done: list[dict], m: dict, steps_known: tuple | None = None) -> dict:
    """Projected NUTS time (2 x 300/300) from the value-and-gradient time of the MAP worker and,
    from the measured NUTS run with the same K and the largest P, the mean steps/iteration and
    the ratio (time per leapfrog step) / (value-and-gradient time).  ``steps_known`` =
    (steps/iteration, source) replaces the assumed steps/iteration (a run of the same
    configuration and seed in another session)."""
    same_k = [d for d in done if d["K"] == m["K"]]
    if same_k:
        ref = max(same_k, key=lambda d: d["P"])
        steps, src = ref["steps_per_iter_mean"], f"{ref['cid']} (P={ref['P']})"
        ratio = float(ref["t_per_step_s"] / ref["t_vg_map_s"])
    else:
        steps, src, ratio = NUTS_DEFAULT_STEPS, "default steps/iter", 1.0
    if steps_known is not None:
        steps, src = float(steps_known[0]), str(steps_known[1])
    iters = NUTS_CHAINS * (NUTS_WARMUP + NUTS_SAMPLES)
    return dict(steps_per_iter_assumed=steps, steps_source=src, step_over_vg_ratio=ratio,
                t_nuts_projected_s=iters * steps * m["t_vg_s"] * ratio + m["t_compile_vg_s"] * 2)


def replicate_nuts_set() -> set:
    """Configurations whose NUTS run is repeated in the replicate session: those measured within
    ``NUTS_BUDGET_S`` in the primary session."""
    out = set()
    for c in config_grid():
        r = _read("nuts", c.cid)
        if r and r.get("status") == "measured" and r["t_nuts_s"] <= NUTS_BUDGET_S:
            out.add(c.cid)
    return out


def run_stage_nuts(force: bool = False, budget_s: float = NUTS_BUDGET_S, session: str = PRIMARY_SESSION) -> None:
    """NUTS timings, smallest P first.  Primary session: run if the projected time is within
    ``budget_s``, else record the projection.  Replicate session: run the configurations of
    ``replicate_nuts_set``; project the others with this session's value-and-gradient time and
    the reference run with the same K and the largest P over both sessions (steps/iteration of
    the primary session's run of the same configuration where one exists)."""
    P = _stored_P()
    done = []
    rep_set = replicate_nuts_set() if session != PRIMARY_SESSION else set()
    # replicate session: the reference for projections may also be a measured run of the primary
    # session (step counts do not depend on the session; its time-per-step / value-and-gradient
    # ratio is taken within its own session), so that the largest measured P is always used
    pool = []
    if session != PRIMARY_SESSION:
        for c in P:
            ra, ma = _read("nuts", c), _read("map", c)
            if ra and ma and ra.get("status") == "measured":
                ra["t_vg_map_s"] = ma["t_vg_s"]
                pool.append(ra)
    for cid in sorted(P, key=lambda c: (P[c], c)):
        if config_from_cid(cid).generator not in NUTS_GENERATORS:
            continue
        m = _read("map", cid, session)
        if m is None:
            print(f"[nuts:{session}] {cid}: no MAP timing, skipped", flush=True)
            continue
        prev = _read("nuts", cid, session)
        if prev is not None and not force:
            if prev.get("status") == "measured":
                prev["t_vg_map_s"] = m["t_vg_s"]
                done.append(prev)
            continue
        known = None
        if session != PRIMARY_SESSION:
            ra = _read("nuts", cid)
            if ra and ra.get("status") == "measured":
                known = (ra["steps_per_iter_mean"], f"session {PRIMARY_SESSION} run of {cid} (same seed)")
        proj = nuts_projection(cid, done + pool, m, known)
        run_it = proj["t_nuts_projected_s"] <= budget_s if session == PRIMARY_SESSION else cid in rep_set
        if not run_it:
            rec = dict(cid=cid, session=session, P=m["P"], K=m["K"], N=m["N"], status="projected",
                       budget_s=budget_s, **proj)
            C.write_json(rec, _run_json("nuts", cid, session))
            print(f"[nuts:{session}] {cid} P={P[cid]} projected {proj['t_nuts_projected_s']:.0f}s: not run",
                  flush=True)
            continue
        print(f"[nuts:{session}] {cid} P={P[cid]} (projected {proj['t_nuts_projected_s']:.0f}s)", flush=True)
        rc = _spawn(["worker-nuts", "--cid", cid, "--session", session], log_dir(session) / "nuts.log",
                    timeout_s=NUTS_TIMEOUT_S)
        if rc == -9:
            rec = dict(cid=cid, session=session, P=m["P"], K=m["K"], N=m["N"], status="timeout",
                       budget_s=budget_s, t_nuts_lower_bound_s=NUTS_TIMEOUT_S, **proj)
            C.write_json(rec, _run_json("nuts", cid, session))
            print(f"  timeout after {NUTS_TIMEOUT_S:.0f}s (recorded as a lower bound)", flush=True)
            continue
        rec = _read("nuts", cid, session)
        if rc != 0 or rec is None:
            print(f"  failed (rc={rc}); see {log_dir(session).name}/nuts.log", flush=True)
            continue
        rec.update(status="measured", **{f"pre_{k}": v for k, v in proj.items()})
        C.write_json(rec, _run_json("nuts", cid, session))
        rec["t_vg_map_s"] = m["t_vg_s"]
        done.append(rec)


# ----------------------------------------------------------------------------
# Tables
# ----------------------------------------------------------------------------
def _loglog_fit(x, y) -> dict:
    """Least-squares line of log10(y) on log10(x) over the finite, positive pairs."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    x, y = np.log10(x[ok]), np.log10(y[ok])
    if len(x) < 3 or np.ptp(x) == 0:
        return dict(n=len(x), slope=np.nan, slope_se=np.nan, intercept=np.nan, r2=np.nan)
    A = np.vstack([np.ones_like(x), x]).T
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    yhat = A @ coef
    dof = len(x) - 2
    s2 = float(((y - yhat) ** 2).sum() / dof)
    cov = s2 * np.linalg.inv(A.T @ A)
    sst = float(((y - y.mean()) ** 2).sum())
    r2 = 1 - ((y - yhat) ** 2).sum() / sst if sst > 0 else np.nan
    return dict(n=len(x), slope=float(coef[1]), slope_se=float(np.sqrt(cov[1, 1])), intercept=float(coef[0]),
                r2=float(r2))


def _linear_fit(x, y) -> dict:
    """Least-squares line y = intercept + slope * x over the finite pairs (memory vs size: the
    intercept is the process baseline, the slope the cost per unit)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < 3 or np.ptp(x) == 0:
        return dict(n=len(x), slope=np.nan, slope_se=np.nan, intercept=np.nan, r2=np.nan)
    A = np.vstack([np.ones_like(x), x]).T
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    res = y - A @ coef
    cov = float((res ** 2).sum() / (len(x) - 2)) * np.linalg.inv(A.T @ A)
    sst = float(((y - y.mean()) ** 2).sum())
    return dict(n=len(x), slope=float(coef[1]), slope_se=float(np.sqrt(cov[1, 1])), intercept=float(coef[0]),
                r2=float(1 - (res ** 2).sum() / sst) if sst > 0 else np.nan)


LOGLOG, LINEAR = "log10(y) = intercept + slope * log10(x)", "y = intercept + slope * x"

COMPLEXITY_ROWS = [
    ("evidence encoding", "O(N K)", "O(n_c K) per chunk",
     "each edge's K values {-1,0,1} are packed into ceil(K/39) int64 base-3 words"),
    ("pattern compression (streaming, by TF chunk)", "O(N K + N log n_c + C P_c log(C P_c))",
     "O(N) int32 edge->pattern index + O(n_c K) chunk + O(P K) table",
     "sort-based unique per chunk of n_c edges, merge of C chunk dictionaries, O(N) remap; "
     "hashing would make it O(N K) expected"),
    ("pattern compression (whole matrix, lcm.PatternData.from_matrix)", "O(N K log N)",
     "O(N K) matrix + O(N) int64 sort/inverse arrays", "np.unique(axis=0) over the dense matrix"),
    ("number of patterns P", "P <= min(N, 3^K, M + N_rep)", "-",
     "M distinct observation masks, N_rep edges with >= 1 report: every unreported edge has the all-zero "
     "pattern of its mask; TF-blocked observation (coverage x target node sets) keeps M small"),
    ("log posterior + gradient (one evaluation)", "O(P K S)", "O(P K) (16 P K bytes float64) + O(P S)",
     "independent of N given P: N enters only through the pattern counts"),
    ("MAP (multi-start L-BFGS)", "O(R I P K S)", "O(P K) + O(R P) stored restart credibilities",
     "R restarts, I objective+gradient evaluations per restart; I is not constant: it grew with N in "
     "the experiments (scale_fits.csv, 'MAP evaluations per restart vs N')"),
    ("NUTS", "O(C D L P K S)", "O(P K + C D d); + O(C D P) if credibility is stored per draw",
     "C chains, D iterations, L leapfrog steps per iteration (<= 1023), d parameters; L grew with N "
     "up to the tree-depth cap in the experiments"),
    ("posterior credibility (lcm.nuts_credibility)", "O(D' P K S)", "O(D' P) (quantiles); O(P) streaming mean/sd",
     "D' retained draws"),
    ("edge-level outputs", "O(N)", "O(N)", "credibility of an edge = credibility of its pattern (index lookup)"),
]

_LOAD_KEYS = ("load1_start", "load1_mean", "load1_max")


def _records(stage: str, session: str, dense: bool = False) -> list[dict]:
    """Stored records of a stage and session in ``config_grid`` order (+ the whole-matrix runs)."""
    rows = []
    for c in config_grid():
        for suffix in (("", "__dense") if dense else ("",)):
            r = _read(stage, c.cid + suffix, session)
            if r:
                r = dict(r)
                r["session"], r["generator"], r["size"] = session, c.generator, c.size
                rows.append(r)
    return rows


def _session_rows(sessions: list) -> list[tuple]:
    """Environment rows describing each measurement session (window and machine load)."""
    rows = []
    for s in sessions:
        files = sorted(run_dir(s).glob("*/*.json"))
        if not files:
            continue
        recs = [json.loads(f.read_text()) for f in files]
        # NUTS records with status 'projected' hold a projection only (nothing was run)
        timed = [x.get("status") != "projected" for x in recs]
        mt = [f.stat().st_mtime for f, ok in zip(files, timed) if ok]
        loads = [x["load1_mean"] for x in recs if x.get("load1_mean") is not None]
        lmax = [x["load1_max"] for x in recs if x.get("load1_max") is not None]
        rows.append((f"session {s}", SESSIONS[s]))
        rows.append((f"session {s}: measurement records", str(sum(timed))))
        rows.append((f"session {s}: projection-only records (NUTS not run)", str(len(timed) - sum(timed))))
        if mt:
            rows.append((f"session {s}: first / last measurement written",
                         f"{time.strftime('%Y-%m-%d %H:%M', time.localtime(min(mt)))} / "
                         f"{time.strftime('%Y-%m-%d %H:%M', time.localtime(max(mt)))}"))
        if loads:
            rows.append((f"session {s}: 1-min load average, per-run means (min / median / max)",
                         f"{np.min(loads):.1f} / {np.median(loads):.1f} / {np.max(loads):.1f} "
                         f"on {os.cpu_count()} cores"))
        if lmax:
            rows.append((f"session {s}: largest 1-min load average seen during a run", f"{np.max(lmax):.1f}"))
    return rows


def _template_rows() -> list[tuple]:
    """Provenance of the frozen template and its distance from the currently stored generating fit."""
    rows = []
    if TEMPLATE_NPZ.exists():
        rows.append(("template snapshot written",
                     time.strftime("%Y-%m-%d %H:%M", time.localtime(TEMPLATE_NPZ.stat().st_mtime))))
        try:
            tpl = np.load(TEMPLATE_NPZ)
            gp = np.load(GEN_FIT / "params.npz")
            rows.append(("template generating parameters vs the currently stored fit (max |dw| / max |dtheta|)",
                         f"{np.abs(tpl['w'] - gp['w']).max():.2e} / {np.abs(tpl['theta'] - gp['theta']).max():.2e} "
                         "(the stored optimum was Newton-polished after the snapshot; 'all' keeps an existing "
                         "template, 'template' rebuilds it and then every stage must be re-run with --force)"))
        except Exception as exc:  # pragma: no cover - fit not available
            rows.append(("template generating parameters vs the currently stored fit", f"not compared ({exc})"))
    return rows


def _compression_table(sessions: list, info: dict) -> pd.DataFrame:
    rows = [r for s in sessions for r in _records("compress", s, dense=True)]
    comp = pd.DataFrame(rows)
    keep = ["cid", "session", "method", "generator", "size", "K", "T", "G", "N", "P", "P_over_N", "n_masks",
            "frac_edges_unobserved", "mean_annotators_observing", "pos_rate_observed_cells", "n_edges_any_report",
            "P_bound_masks_plus_reported", "patterns_identical", "n_chunks", "tfs_per_chunk", "max_chunk_patterns", "t_generate_s", "t_encode_s",
            "t_unique_s", "t_merge_s", "t_remap_s", "t_compress_s", "cpu_compress_s", "compress_ns_per_edge",
            "rss_baseline_mib", "peak_rss_mib", "peak_rss_sampled_mib", "ru_maxrss_mib", "edge_index_mib",
            "dense_matrix_mib", "pattern_table_kib", "load1_start", "load1_mean", "load1_max"]
    if not len(comp):
        return comp
    comp["P_over_N"] = comp["P"] / comp["N"]
    comp["compress_ns_per_edge"] = comp["t_compress_s"] / comp["N"] * 1e9
    # every edge without a report has the all-zero pattern of its observation mask
    comp["P_bound_masks_plus_reported"] = comp["n_masks"] + comp["n_edges_any_report"]
    for k in keep:
        if k not in comp:
            comp[k] = np.nan
    comp = comp[keep].copy()
    comp["analysis_status"] = _status_by_session(comp["session"])
    comp.loc[len(comp)] = {**{k: np.nan for k in comp.columns}, "cid": "REAL_global_prespecified_cc",
                           "session": "", "method": "real evidence (not timed)", "generator": "real",
                           "size": "real", "K": info["K"], "T": info["T0"], "G": info["G0"],
                           "N": info["real"]["N"], "P": info["real"]["P"],
                           "P_over_N": info["real"]["P"] / info["real"]["N"], "n_masks": info["real"]["n_masks"],
                           "frac_edges_unobserved": info["real"]["frac_edges_unobserved"],
                           "mean_annotators_observing": info["real"]["mean_annotators_observing"],
                           "pos_rate_observed_cells": info["real"]["pos_rate_observed_cells"],
                           "n_edges_any_report": info["real"]["n_edges_any_report"],
                           "P_bound_masks_plus_reported": info["real"]["n_masks"] + info["real"]["n_edges_any_report"],
                           "analysis_status": ANALYSIS_STATUS}
    return comp


def _generator_check(info: dict) -> pd.DataFrame:
    gc_rows = []
    for gen in ("template", "independent"):
        r = _read("compress", f"{gen}_real_K21")
        if not r:
            continue
        for j, a in enumerate(info["annotators"]):
            gc_rows.append(dict(generator=gen, annotator=a, family=info["family"][j],
                                coverage_real=info["real"]["coverage"][j], coverage_synthetic=r["coverage"][j],
                                pos_rate_real=info["real"]["pos_rate"][j], pos_rate_synthetic=r["pos_rate"][j],
                                coverage_evidence_summary=info["es_coverage"][j],
                                pos_rate_evidence_summary=info["es_pos_rate"][j]))
        gc_rows.append(dict(generator=gen, annotator="__ALL__", family="",
                            coverage_real=info["real"]["mean_annotators_observing"],
                            coverage_synthetic=r["mean_annotators_observing"],
                            pos_rate_real=info["real"]["pos_rate_observed_cells"],
                            pos_rate_synthetic=r["pos_rate_observed_cells"],
                            P_real=info["real"]["P"], P_synthetic=r["P"],
                            n_masks_real=info["real"]["n_masks"], n_masks_synthetic=r["n_masks"],
                            frac_unobserved_real=info["real"]["frac_edges_unobserved"],
                            frac_unobserved_synthetic=r["frac_edges_unobserved"],
                            n_any_report_real=info["real"]["n_edges_any_report"],
                            n_any_report_synthetic=r["n_edges_any_report"]))
    gcheck = pd.DataFrame(gc_rows)
    if len(gcheck):
        gcheck["coverage_ratio"] = gcheck["coverage_synthetic"] / gcheck["coverage_real"]
        gcheck["pos_rate_ratio"] = gcheck["pos_rate_synthetic"] / gcheck["pos_rate_real"]
        gcheck["analysis_status"] = ANALYSIS_STATUS
    return gcheck


def _map_table(sessions: list) -> pd.DataFrame:
    rows = [{k: v for k, v in r.items() if k != "restarts"} for s in sessions for r in _records("map", s)]
    mp = pd.DataFrame(rows)
    if not len(mp):
        return mp
    mcols = ["cid", "session", "generator", "size", "N", "K", "S", "P", "PKS", "n_params", "t_build_init_s",
             "t_compile_s", "t_pot_s", "t_grad_s", "t_eval_lbfgs_s", "t_eval_lbfgs_min_s", "t_vg_s", "t_vg_min_s",
             "t_vg_ns_per_PKS", "counts_factor", "t_vg_counts_x_factor_s", "t_vg_repeat_s",
             "counts_invariance_ratio", "counts_invariance_ratio_paired", "n_restarts_run", "t_restart_mean_s",
             "t_restart_max_s", "nfev_mean", "nfev_max", "n_iter_mean", "n_converged", "t_per_lbfgs_eval_s",
             "lbfgs_time_over_bench", "t_map_10_s", "t_map_10_status", "t_map_50_projected_s", "cpu_restarts_s", "t_restarts_wall_s",
             "nfev_guess", "eval_only", "peak_rss_mib", "model_arrays_mib", "load1_start", "load1_mean", "load1_max"]
    for k in mcols:
        if k not in mp:
            mp[k] = np.nan
    mp["PKS"] = mp["P"] * mp["K"] * mp["S"]
    mp["t_vg_ns_per_PKS"] = mp["t_vg_s"] / mp["PKS"] * 1e9
    mp["counts_invariance_ratio"] = mp["t_vg_counts_x_factor_s"] / mp["t_vg_repeat_s"]
    # time per evaluation inside the restarts / benchmark time of one objective + gradient call:
    # > 1 (optimiser overhead, line-search bookkeeping), so 'projected' MAP times are optimistic
    mp["lbfgs_time_over_bench"] = mp["t_per_lbfgs_eval_s"] / mp["t_eval_lbfgs_s"]
    # the plan's MAP uses 50 restarts: same build/compile, 50 x the mean restart
    mp["t_map_50_projected_s"] = mp["t_build_init_s"] + mp["t_compile_s"] + 50 * mp["t_restart_mean_s"]
    mp = mp.sort_values(["session", "P", "cid"])[mcols].reset_index(drop=True)
    mp["analysis_status"] = _status_by_session(mp["session"])
    return mp


def _restart_table(sessions: list) -> pd.DataFrame:
    rr = []
    for s in sessions:
        for r in _records("map", s):
            for x in r.get("restarts", []):
                rr.append(dict(cid=r["cid"], session=s, P=r["P"], K=r["K"], **x))
    rdf = pd.DataFrame(rr)
    if len(rdf):
        rdf["analysis_status"] = _status_by_session(rdf["session"])
    return rdf


def _nuts_table(sessions: list) -> pd.DataFrame:
    rows = []
    for s in sessions:
        for r in _records("nuts", s):
            m = _read("map", r["cid"], s)
            if m:
                r["t_vg_map_s"] = m["t_vg_s"]
                r["S"] = r.get("S", m["S"])
            if r.get("status") == "measured" and not r.get("probe_kind"):
                r["probe_kind"] = "single_chain_cold"       # records made before the main-like probe
            rows.append(r)
    nt = pd.DataFrame(rows)
    if not len(nt):
        return nt
    ncols = ["cid", "session", "generator", "size", "N", "K", "S", "P", "PKS", "status", "chains", "warmup",
             "samples", "probe_kind", "t_probe_cold_s", "t_probe_s", "t_warmup_s", "t_sampling_s", "t_nuts_s",
             "t_nuts_excl_compile_s", "t_nuts_2x300_s", "cpu_nuts_s", "steps_total", "steps_per_iter_mean",
             "steps_per_iter_sampling_mean", "steps_per_iter_max", "n_divergent", "r_hat_max", "ess_bulk_min",
             "ess_tail_min", "t_nuts_per_min_ess_bulk_s", "t_per_step_s", "t_per_step_excl_compile_s",
             "t_per_step_sampling_s", "t_vg_map_s", "step_over_vg_measured", "t_per_step_used_s",
             "steps_per_iter_assumed", "steps_source", "step_over_vg_ratio", "t_nuts_projected_s",
             "t_nuts_projected_cap_s", "t_nuts_lower_bound_s", "t_nuts_4x1000_projected_s", "peak_rss_mib", "peak_rss_nuts_mib",
             "t_cred_posthoc_s", "peak_rss_cred_posthoc_mib", "cred_draws_matrix_mib", "load1_start",
             "load1_mean", "load1_max"]
    for k in ncols:
        if k not in nt:
            nt[k] = np.nan
    num = ["t_nuts_s", "t_probe_s", "steps_total", "steps_per_iter_mean", "steps_per_iter_assumed",
           "t_per_step_s", "t_vg_map_s", "step_over_vg_ratio", "t_nuts_projected_s", "ess_bulk_min", "S"]
    nt[num] = nt[num].apply(pd.to_numeric, errors="coerce")
    # measured -> measured time; projected -> projection; timeout -> NaN (lower bound in its own column)
    nt["t_nuts_2x300_s"] = np.where(nt["status"] == "measured", nt["t_nuts_s"],
                                    np.where(nt["status"] == "projected", nt["t_nuts_projected_s"], np.nan))
    nt["t_nuts_per_min_ess_bulk_s"] = nt["t_nuts_s"] / nt["ess_bulk_min"]
    # projections assume the steps/iteration of a smaller reference run, but steps grew with N in
    # the measured runs: the same projection at the tree-depth cap bounds it from above
    nt["t_nuts_projected_cap_s"] = np.where(
        nt["status"] == "projected",
        NUTS_CHAINS * (NUTS_WARMUP + NUTS_SAMPLES) * NUTS_MAX_STEPS * nt["t_vg_map_s"] * nt["step_over_vg_ratio"],
        np.nan)
    # the timed run compiles its own kernels (about t_probe_s): time per leapfrog step without it
    nt["t_nuts_excl_compile_s"] = (nt["t_nuts_s"] - nt["t_probe_s"].fillna(0.0)).clip(lower=0.0)
    nt["t_per_step_excl_compile_s"] = nt["t_nuts_excl_compile_s"] / nt["steps_total"]
    nt["step_over_vg_measured"] = nt["t_per_step_excl_compile_s"] / nt["t_vg_map_s"]
    # plan settings (4 x 1000/1000), naive projection: same steps/iteration and time per step
    spi = nt["steps_per_iter_mean"].fillna(nt["steps_per_iter_assumed"])
    tps = nt["t_per_step_excl_compile_s"].fillna(nt["t_vg_map_s"] * nt["step_over_vg_ratio"].fillna(1.0))
    nt["t_per_step_used_s"] = tps
    nt["t_nuts_4x1000_projected_s"] = 4 * 2000 * spi * tps
    nt["PKS"] = nt["P"] * nt["K"] * nt["S"]
    nt = nt.sort_values(["session", "P", "cid"])[ncols].reset_index(drop=True)
    nt["analysis_status"] = _status_by_session(nt["session"])
    return nt


def _fits_table(comp: pd.DataFrame, mp: pd.DataFrame, nt: pd.DataFrame, sessions: list) -> pd.DataFrame:
    frows = []

    def add(session, quantity, subset, x, y, xv, yv, form=LOGLOG):
        fit = (_loglog_fit if form == LOGLOG else _linear_fit)(xv, yv)
        if fit["n"] > 0:
            frows.append(dict(session=session, quantity=quantity, subset=subset, form=form, x=x, y=y, **fit))

    for s in sessions:
        cs = comp[(comp["session"] == s) & comp["generator"].isin(["template", "independent"])] if len(comp) else comp
        if len(cs):
            if s == PRIMARY_SESSION:                         # P does not depend on the session
                for (gen, method, K), g in cs.groupby(["generator", "method", "K"]):
                    if len(g) >= 3:
                        add(s, "P vs N", f"{gen} {method} K={K}", "N", "P", g["N"], g["P"])
            for method, g in cs.groupby("method"):
                lab = f"{method}, all K and generators"
                add(s, "compression wall time vs N", lab, "N", "t_compress_s", g["N"], g["t_compress_s"])
                add(s, "compression wall time vs N*K", lab, "N*K", "t_compress_s", g["N"] * g["K"],
                    g["t_compress_s"])
                add(s, "peak RSS vs N", lab, "N", "peak_rss_mib", g["N"], g["peak_rss_mib"])
                add(s, "compression peak RSS (bytes) linear in N", lab, "N", "peak_rss_bytes", g["N"],
                    g["peak_rss_mib"] * 2**20, LINEAR)
        ms = mp[mp["session"] == s] if len(mp) else mp
        if len(ms):
            ran = ms[ms["t_map_10_status"].isin(["measured", "extrapolated"])]
            real = ran[ran["generator"].isin(["template", "independent"])]
            for lab, sub, sub_ran in (("all", ms, ran), ("P >= 1e4", ms[ms["P"] >= 1e4], ran[ran["P"] >= 1e4])):
                add(s, "value+gradient time vs P*K*S", lab, "P*K*S", "t_vg_s", sub["PKS"], sub["t_vg_s"])
                add(s, "value+gradient time (minimum) vs P*K*S", lab, "P*K*S", "t_vg_min_s", sub["PKS"],
                    sub["t_vg_min_s"])
                add(s, "L-BFGS evaluation time vs P*K*S", lab, "P*K*S", "t_per_lbfgs_eval_s", sub_ran["PKS"],
                    sub_ran["t_per_lbfgs_eval_s"])
                add(s, "MAP 10-restart time vs P*K*S", lab + ", measured or extrapolated", "P*K*S", "t_map_10_s",
                    sub_ran["PKS"], sub_ran["t_map_10_s"])
                add(s, "MAP peak RSS vs P*K", lab, "P*K", "peak_rss_mib", sub["P"] * sub["K"], sub["peak_rss_mib"])
            add(s, "MAP worker peak RSS (bytes) linear in P*K", "all", "P*K", "peak_rss_bytes", ms["P"] * ms["K"],
                ms["peak_rss_mib"] * 2**20, LINEAR)
            add(s, "MAP evaluations per restart vs N", "template + independent, restarts run", "N", "nfev_mean",
                real["N"], real["nfev_mean"])
            add(s, "MAP evaluations per restart vs P", "template + independent, restarts run", "P", "nfev_mean",
                real["P"], real["nfev_mean"])
        ns = nt[(nt["session"] == s) & (nt["status"] == "measured")] if len(nt) else nt
        if len(ns):
            add(s, "NUTS time per leapfrog step (compile excluded) vs P*K*S", "measured", "P*K*S",
                "t_per_step_excl_compile_s", ns["PKS"], ns["t_per_step_excl_compile_s"])
            add(s, "NUTS 2x300/300 wall time vs P*K*S", "measured", "P*K*S", "t_nuts_s", ns["PKS"], ns["t_nuts_s"])
            add(s, "NUTS leapfrog steps per iteration vs N", "measured", "N", "steps_per_iter_mean", ns["N"],
                ns["steps_per_iter_mean"])
    fits = pd.DataFrame(frows)
    if len(fits):
        fits["analysis_status"] = _status_by_session(fits["session"])
    return fits


def _replicate_table(comp: pd.DataFrame, mp: pd.DataFrame, rdf: pd.DataFrame, nt: pd.DataFrame) -> pd.DataFrame:
    """Primary vs replicate session: deterministic quantities must agree exactly, timings show the
    sensitivity to the machine load (ratio_a_over_b)."""
    a, b = PRIMARY_SESSION, REPLICATE_SESSION
    rows = []

    def pair(df, cid, **sel):
        out = []
        for s in (a, b):
            d = df[(df["cid"] == cid) & (df["session"] == s)]
            for k, v in sel.items():
                d = d[d[k] == v]
            out.append(d.iloc[0] if len(d) else None)
        return out

    def row(stage, ra, rb, det_name, det_a, det_b, identical, t_name, **extra):
        ta, tb = ra.get(t_name, np.nan), rb.get(t_name, np.nan)
        rows.append(dict(cid=ra["cid"], stage=stage, P=ra["P"], K=ra["K"], deterministic_quantity=det_name,
                         value_a=det_a, value_b=det_b, deterministic_identical=identical, time_quantity=t_name,
                         time_a_s=ta, time_b_s=tb, ratio_a_over_b=ta / tb if tb and np.isfinite(tb) else np.nan,
                         peak_rss_a_mib=ra.get("peak_rss_mib", np.nan), peak_rss_b_mib=rb.get("peak_rss_mib", np.nan),
                         load1_mean_a=ra.get("load1_mean", np.nan), load1_mean_b=rb.get("load1_mean", np.nan),
                         **extra))

    cids = [c.cid for c in config_grid()]
    if len(comp):
        for method, stage in (("chunked_stream", "compress"), ("dense_unique", "compress (whole matrix)")):
            for cid in cids:
                ra, rb = pair(comp, cid, method=method)
                if ra is None or rb is None:
                    continue
                ident = bool(rb["patterns_identical"]) if pd.notna(rb["patterns_identical"]) else None
                row(stage, ra, rb, "pattern table (rows and counts) vs the stored table; P shown", ra["P"], rb["P"],
                    ident if ident is not None else bool(ra["P"] == rb["P"]), "t_compress_s")
    if len(mp):
        for cid in cids:
            ra, rb = pair(mp, cid)
            if ra is None or rb is None:
                continue
            row("map evaluation", ra, rb, "number of parameters", ra["n_params"], rb["n_params"],
                bool(ra["n_params"] == rb["n_params"]), "t_vg_s")
            if ra["n_restarts_run"] > 0 and rb["n_restarts_run"] > 0 and len(rdf):
                xa = rdf[(rdf["cid"] == cid) & (rdf["session"] == a)].set_index("restart")
                xb = rdf[(rdf["cid"] == cid) & (rdf["session"] == b)].set_index("restart")
                common = xa.index.intersection(xb.index)
                same = bool((xa.loc[common, "nfev"] == xb.loc[common, "nfev"]).all() and
                            (xa.loc[common, "n_iter"] == xb.loc[common, "n_iter"]).all())
                dlp = float((xa.loc[common, "log_post"] - xb.loc[common, "log_post"]).abs().max())
                row("map restarts", ra, rb, "L-BFGS evaluations summed over the common restarts",
                    int(xa.loc[common, "nfev"].sum()), int(xb.loc[common, "nfev"].sum()), same, "t_restarts_wall_s",
                    n_common_restarts=len(common), max_abs_diff_log_post=dlp,
                    time_note="" if len(xa) == len(xb) else f"restarts run: {len(xa)} vs {len(xb)}")
    if len(nt):
        for cid in cids:
            ra, rb = pair(nt, cid)
            if ra is None or rb is None or ra["status"] != "measured" or rb["status"] != "measured":
                continue
            row("nuts", ra, rb, "leapfrog steps (warm-up + sampling, 2 chains)", ra["steps_total"],
                rb["steps_total"], bool(ra["steps_total"] == rb["steps_total"]), "t_nuts_excl_compile_s",
                time_note="t_nuts_s minus the compile probe (different probe kinds in the two sessions)")
    rep = pd.DataFrame(rows)
    if len(rep):
        for k in ("n_common_restarts", "max_abs_diff_log_post", "time_note"):
            if k not in rep:
                rep[k] = np.nan
        rep["analysis_status"] = SESSION_STATUS[REPLICATE_SESSION]
    return rep


def _best(df: pd.DataFrame, cid: str, time_col: str, status_col: str | None = None, **sel):
    """Most direct record of a configuration over the sessions: best status first (measured <
    extrapolated < projected), then the shortest time (the least disturbed measurement)."""
    d = df[df["cid"] == cid]
    for k, v in sel.items():
        d = d[d[k] == v]
    d = d[d[time_col].notna()]
    if not len(d):
        return None
    rank = d[status_col].map(STATUS_RANK).fillna(9) if status_col else pd.Series(0, index=d.index)
    return d.assign(_rank=rank).sort_values(["_rank", time_col]).iloc[0]


def _peak_rss(df: pd.DataFrame, cid: str, **sel) -> tuple:
    """(largest peak RSS over the sessions, its session).  Resident memory is a lower bound of
    the footprint when the machine is short of memory (compressed / swapped pages are not
    resident), so the largest observed value is the one to plan with."""
    d = df[df["cid"] == cid]
    for k, v in sel.items():
        d = d[d[k] == v]
    d = d[d["peak_rss_mib"].notna()]
    if not len(d):
        return np.nan, None
    r = d.sort_values("peak_rss_mib").iloc[-1]
    return r["peak_rss_mib"], r["session"]


def _summary_table(comp: pd.DataFrame, mp: pd.DataFrame, nt: pd.DataFrame) -> pd.DataFrame:
    """One row per configuration: times from the most direct, then fastest record over the
    sessions (``_best``); peak RSS = the largest over the sessions (``_peak_rss``)."""
    rows = []
    for c in config_grid():
        rc = _best(comp, c.cid, "t_compress_s", method="chunked_stream") if len(comp) else None
        if rc is None:
            continue
        rss, rss_s = _peak_rss(comp, c.cid, method="chunked_stream")
        r = dict(cid=c.cid, generator=c.generator, size=c.size, K=c.K, N=int(rc["N"]), P=int(rc["P"]),
                 P_over_N=rc["P"] / rc["N"], compress_s=rc["t_compress_s"], compress_session=rc["session"],
                 compress_load1_mean=rc["load1_mean"], compress_peak_rss_mib=rss, compress_peak_rss_session=rss_s)
        if len(mp):
            rv = _best(mp, c.cid, "t_vg_s")
            rm = _best(mp, c.cid, "t_map_10_s", "t_map_10_status")
            if rv is not None:
                r.update(S=rv["S"], n_params=rv["n_params"], t_vg_s=rv["t_vg_s"], t_vg_session=rv["session"])
            if rm is not None:
                ran = mp[(mp["cid"] == c.cid) & (mp["n_restarts_run"] > 0)]
                rss, rss_s = _peak_rss(mp, c.cid)
                r.update(map_10_s=rm["t_map_10_s"], map_10_status=rm["t_map_10_status"],
                         map_10_session=rm["session"], map_restarts_run=rm["n_restarts_run"],
                         map_nfev_mean=ran["nfev_mean"].iloc[0] if len(ran) else np.nan,
                         map_50_projected_s=rm["t_map_50_projected_s"], map_load1_mean=rm["load1_mean"],
                         map_peak_rss_mib=rss, map_peak_rss_session=rss_s)
        if len(nt):
            rn = _best(nt, c.cid, "t_nuts_2x300_s", "status")
            if rn is not None:
                rss, rss_s = _peak_rss(nt, c.cid, status="measured")
                r.update(nuts_2x300_s=rn["t_nuts_2x300_s"], nuts_status=rn["status"], nuts_session=rn["session"],
                         nuts_2x300_cap_s=rn["t_nuts_projected_cap_s"],
                         nuts_steps_per_iter=rn["steps_per_iter_mean"] if rn["status"] == "measured"
                         else rn["steps_per_iter_assumed"],
                         nuts_r_hat_max=rn["r_hat_max"], nuts_load1_mean=rn["load1_mean"],
                         nuts_peak_rss_mib=rss, nuts_peak_rss_session=rss_s,
                         nuts_cred_posthoc_s=rn["t_cred_posthoc_s"])
        rows.append(r)
    s = pd.DataFrame(rows)
    if len(s):
        s = s.sort_values(["generator", "K", "N"]).reset_index(drop=True)
        # a row is prespecified only if every number it reports comes from the primary session
        scols = [c for c in s.columns if c.endswith("_session")]
        used_other = s[scols].apply(lambda col: col.notna() & (col != PRIMARY_SESSION)).any(axis=1)
        s["analysis_status"] = np.where(used_other, SESSION_STATUS[REPLICATE_SESSION], ANALYSIS_STATUS)
    return s


def build_tables(tables_dir: Path = C.TABLES) -> dict:
    """Manuscript tables from the stored measurement records of every session."""
    info = json.loads(TEMPLATE_JSON.read_text())
    sessions = [s for s in SESSIONS if run_dir(s).exists()]
    out = {}
    env = environment()
    extra = [(k, v, ANALYSIS_STATUS) for k, v in _template_rows()]
    extra += [(k, v, SESSION_STATUS[k.split()[1].rstrip(":")]) for k, v in _session_rows(sessions)]
    if extra:
        env = pd.concat([env, pd.DataFrame(extra, columns=env.columns)], ignore_index=True)
    out["scale_environment.csv"] = env
    cx = pd.DataFrame(COMPLEXITY_ROWS, columns=["stage", "time", "memory", "notes"])
    cx["analysis_status"] = ANALYSIS_STATUS
    out["scale_complexity.csv"] = cx
    comp = _compression_table(sessions, info)
    out["scale_compression.csv"] = comp
    out["scale_generator_check.csv"] = _generator_check(info)
    mp = _map_table(sessions)
    out["scale_map.csv"] = mp
    rdf = _restart_table(sessions)
    out["scale_map_restarts.csv"] = rdf
    nt = _nuts_table(sessions)
    out["scale_nuts.csv"] = nt
    out["scale_fits.csv"] = _fits_table(comp, mp, nt, sessions)
    if REPLICATE_SESSION in sessions:
        out["scale_replicate.csv"] = _replicate_table(comp, mp, rdf, nt)
    if len(comp) and len(mp):
        out["scale_summary.csv"] = _summary_table(comp, mp, nt)
    C.ensure_dir(tables_dir)
    for name, df in out.items():
        df.to_csv(tables_dir / name, index=False)
    return out


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("template", help="template from the real global evidence (invalidates stored patterns)")
    for name in ("compress", "map", "nuts", "all", "replicate"):
        p = sub.add_parser(name)
        p.add_argument("--force", action="store_true", help="rerun measurements that already exist")
        if name not in ("all", "replicate"):
            p.add_argument("--session", choices=sorted(SESSIONS), default=PRIMARY_SESSION)
    sub.add_parser("tables", help="manuscript tables from stored measurements")
    for name in ("worker-compress", "worker-map", "worker-nuts"):
        p = sub.add_parser(name)
        p.add_argument("--cid", required=True)
        p.add_argument("--session", choices=sorted(SESSIONS), default=PRIMARY_SESSION)
        if name == "worker-compress":
            p.add_argument("--dense", action="store_true")
        if name == "worker-map":
            p.add_argument("--nfev-guess", type=float, default=None)
            p.add_argument("--eval-only", action="store_true")
    a = ap.parse_args(argv)
    if a.cmd == "template":
        info = build_template()
        print(json.dumps({k: v for k, v in info.items() if k not in ("real",)}, indent=1)[:2000])
    elif a.cmd == "worker-compress":
        res = worker_compress(a.cid, dense=a.dense, session=a.session)
        C.write_json(res, _run_json("compress", a.cid + ("__dense" if a.dense else ""), a.session))
        print(json.dumps({k: res.get(k) for k in ("cid", "method", "N", "P", "t_compress_s", "peak_rss_mib",
                                                  "patterns_identical")}))
    elif a.cmd == "worker-map":
        res = worker_map(a.cid, nfev_guess=a.nfev_guess, eval_only=a.eval_only, session=a.session)
        C.write_json(res, _run_json("map", a.cid, a.session))
        print(json.dumps({k: res[k] for k in ("cid", "P", "t_vg_s", "t_map_10_s", "t_map_10_status", "peak_rss_mib")}))
    elif a.cmd == "worker-nuts":
        res = worker_nuts(a.cid, session=a.session)
        C.write_json(res, _run_json("nuts", a.cid, a.session))
        print(json.dumps({k: res[k] for k in ("cid", "P", "t_nuts_s", "steps_total", "peak_rss_mib")}))
    elif a.cmd == "tables":
        for name, df in build_tables().items():
            print(f"{name}: {len(df)} rows")
    else:
        force = getattr(a, "force", False)
        session = REPLICATE_SESSION if a.cmd == "replicate" else getattr(a, "session", PRIMARY_SESSION)
        every = a.cmd in ("all", "replicate")
        if a.cmd == "all" and (force or not TEMPLATE_NPZ.exists()):
            build_template()
        if a.cmd == "compress" or every:
            run_stage_compress(force, session)
        if a.cmd == "map" or every:
            run_stage_map(force, session)
        if a.cmd == "nuts" or every:
            run_stage_nuts(force, session=session)
        if every:
            for name, df in build_tables().items():
                print(f"{name}: {len(df)} rows")


if __name__ == "__main__":
    main()
