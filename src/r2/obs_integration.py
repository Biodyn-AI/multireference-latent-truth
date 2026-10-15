"""Informative-observation credibilities in claim governance and the replication check
(module ``obs_integration``; POST HOC secondary analysis, plan section 8 "Audit + new model").

Question
--------
The informative-observation latent-class models of ``informative_observation.py``
(``family_hb_obs``, ``em_obs``) were fitted after the held-out results had been seen; their MAP
fits are degenerate (the label T is not anchored, see ``outputs/r2/reports/obs_REPORT.txt``).  This
module asks what would happen to the two downstream uses of a credibility in the paper if the
obs credibility replaced the MAR Family-HB credibility:
(a) label-free claim governance (``governance.py``): Bayesian-FDR tiers, held-out tier
    validation (O/E), stratified trend tests, tier concentration and the TF hub over-promotion
    index (TF literature-degree quintiles, as in ``heldout_eval`` degree calibration);
(b) the cross-cohort replication check (``replication.py``): study-level O/E, Mantel-Haenszel
    OR, exact conditional p and the wrong-TF control.
Every row of this module is ``analysis_status = post_hoc`` except rows of the primary
``family_hb_nuts`` that reproduce a row of a prespecified table exactly (same score, set, label,
seed); those carry the analysis_status of the reproduced row and name it in ``reproduces``.
The prespecified tables (``gov_*.csv``, ``rep_*.csv``, ``heldout_*.csv``) are not written.

Credibilities (fit storage contract ``outputs/r2/fits/<universe>/<variant>/<fit_id>/``)
-------------------------------------------------------------------------------------
* ``family_hb_nuts``  prespecified primary: NUTS posterior mean (``cred_mean.npy``), MAR.
* ``family_hb``, ``em``  MAR MAP fits (``cred.npy``): the like-for-like inference comparators.
* ``family_hb_obs``, ``em_obs``  informative-observation MAP fits (``cred.npy``, best optimum),
  variant ``prespecified_cc__elig``.  No NUTS fit of these models exists (``informative_observation``
  decision of 2026-10-13), so MAP is used everywhere.
All are evaluated on the eligibility-coded patterns of the primary variant
(``evidence.load_patterns_elig``), which refine the stored patterns: MAR credibilities map through
``base_pattern``; tiers over whole tie blocks are therefore identical to the stored-pattern tiers.

(a) Governance (definitions of ``governance.py``, reused through its functions)
------------------------------------------------------------------------------
* Global universe: global-fit BFDR levels over U_glob (Strong q = 0.05, Moderate q = 0.20, whole
  tie blocks; Promoted = Strong U Moderate) validated with ``governance.global_validation``.
* Tissue t: the global-fit credibility of a context edge = the global fit's credibility of the same
  edge in U_glob (eligibility patterns of every U_glob edge are identical in both universes,
  checked); edges whose TF has no held-in literature coverage are outside U_glob and are not
  eligible for any annotator (all codes -2), so their global-fit credibility is the credibility of
  that pattern under the global fit (= w_T for every model).  Levels are computed within U_ctx(t)
  for both fits, the dual tier by the primary rule; held-out labels, O/E, TF-cluster bootstrap and
  Mantel trend by ``governance.Context.set_eval`` / ``governance.validation_rows`` on an
  edge-level context (``ObsContext``: each edge its own pattern, weight 1; identical to the
  pattern-weighted computation).
* The minimum-detectable-O/E columns (``mde_*``, ``power_*``) are computed for ``family_hb_nuts``
  only (reproduction); for every other fit they are left empty: they are design properties of the
  set, are not used in any comparison here, and their exact convolution is expensive for sets of
  10^5-10^6 edges.
* Trend tests: Holm within (fit, scope, strata) across the four universes x four references (the
  governance family, one per fit).
* Tier composition (``obs_governance_tiers.csv``): size, share of the universe, credibility range,
  realised BFDR under the fit's own credibilities, evidence status of the tier edges (unobserved /
  observed-not-reported / reported), TF concentration (``governance.concentration`` definitions),
  overlap with the ``family_hb_nuts`` set of the same universe, scope and tier.
* Hub over-promotion (``obs_governance_degree.csv``): the ``heldout_eval.run_degree`` computation
  (TF literature degree = number of U_glob edges of the TF reported by >= 1 held-in LIT annotator,
  quintiles over the 935 global TFs; references encode, enets2, pert_pooled; evaluable sets of
  ``heldout_eval.load_evalset``) applied to the tier sets: global-fit Strong and Promoted (global
  universe), dual Strong and dual Promoted (tissues).  over_promotion = share of the set's
  evaluable edges in the quintile / share of the evaluable held-out positives in the quintile;
  TF-cluster bootstrap (set fixed; stream ``("boot_degree", ref)`` in the global universe as in
  ``heldout_eval``, ``("boot_degree", ref, universe)`` in tissues).  Tissue TFs outside U_glob have
  literature degree 0 and are put in quintile 1 (below every global TF; ``n_tfs_outside_global``).
  The hub over-promotion index of a set is its quintile-5 over_promotion.

(b) Replication (definitions of ``replication.py``, reused through its functions)
-------------------------------------------------------------------------------
Per universe the universe's own fit (as ``replication.load_universe_tiers``), Strong / Moderate =
``replication.pattern_tiers``; evaluable edges, cis exclusion, strata (cohort x TF), comparators
``universe`` / ``observed``, units = studies (``study_id``), ``replication.unit_status``,
``unit_outcome`` (O, E, O/E, MH OR, exact and permutation p, TF-cluster bootstrap CIs, endpoint),
``posthoc_controls`` / ``unit_posthoc`` (responsiveness strata, wrong-TF control) with the seeds of
``replication.run_outcomes``; ``specific_excess`` = O/E / wrong-TF O/E; Holm over evaluable studies
within (universe, fit, comparator).  The step-1 power lock belongs to the primary promoted set and
does not apply to these post hoc sets; no power columns are computed.

Outputs
-------
``data/r2/tables/obs_governance_tiers.csv``, ``obs_governance.csv`` (tier validation),
``obs_governance_trend.csv``, ``obs_governance_degree.csv``, ``obs_replication.csv``;
``outputs/r2/obs_integration/meta.json`` (inputs with md5, settings, checks, reproduction check),
``run.log``.

Command line (subproject root; about 10 min)::

    export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
    caffeinate -i nice -n 10 /Users/ihorkendiukhov/biodyn-work/.venv/bin/python src/r2/obs_integration.py all
    # or: governance | degree | replication | check
    /Users/ihorkendiukhov/biodyn-work/.venv/bin/python -m pytest src/r2/tests/test_obs_integration.py -q

Deterministic: every random quantity uses the seeds of the reused modules (from ``PRESPECIFIED``).
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "2")

import argparse  # noqa: E402
import contextlib  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from r2 import common as C  # noqa: E402
from r2 import evidence as E  # noqa: E402
from r2 import lcm  # noqa: E402
from r2 import governance as GOV  # noqa: E402
from r2 import replication as RP  # noqa: E402

PS = C.PRESPECIFIED
SEED = int(PS["random_seed"])
VARIANT = PS["evidence_variant_primary"]                 # prespecified_cc
ELIG = E.elig_variant(VARIANT)                           # prespecified_cc__elig
FITS = C.OUT / "fits"
MOD = C.OUT / "obs_integration"
UNIVERSES = ("global",) + tuple(C.TISSUES)
PRIMARY = "family_hb_nuts"
OBS_FITS = ("family_hb_obs", "em_obs")
FIT_IDS = (PRIMARY, "family_hb", "em") + OBS_FITS
MAR_PAIR = {"family_hb_obs": "family_hb", "em_obs": "em"}
FIT_INFO = {
    "family_hb_nuts": dict(model="family_hb", observation_model="MAR", inference="NUTS posterior mean",
                           role="prespecified primary"),
    "family_hb": dict(model="family_hb", observation_model="MAR", inference="MAP",
                      role="MAR comparator (MAP, like-for-like with the obs fits)"),
    "em": dict(model="em", observation_model="MAR", inference="MAP", role="MAR comparator of em_obs"),
    "family_hb_obs": dict(model="family_hb_obs", observation_model="informative (LIT observation emission)",
                          inference="MAP (best optimum; no NUTS)", role="post hoc secondary"),
    "em_obs": dict(model="em_obs", observation_model="informative (LIT observation emission)",
                   inference="MAP (best optimum; no NUTS)", role="post hoc secondary"),
}
DEGREE_REFS = ("encode", "enets2", "pert_pooled")        # = heldout_eval.DEGREE_REFERENCES
DEGREE_SETS = {"global": (("global_fit", "Strong"), ("global_fit", "Promoted")),
               "tissue": (("dual", "Strong"), ("dual", "Promoted"))}
STATUS_NAMES = ("unobserved", "observed_negative", "reported")
TABLE = dict(tiers="obs_governance_tiers.csv", validation="obs_governance.csv", trend="obs_governance_trend.csv",
             degree="obs_governance_degree.csv", replication="obs_replication.csv")
REP_LEVEL = "study"
TOL = 1e-9

ANALYSIS_DECISIONS = [
    dict(id="obsint_design", date="2026-10-14", status="post_hoc",
         item="obs_integration: family_hb_obs / em_obs MAP credibilities (no NUTS exists) put through the governance "
              "tiers, tier validation, trend tests, concentration, TF literature-degree over-promotion and the "
              "study-level replication check, next to family_hb_nuts (reproduction), family_hb and em MAP",
         note="designed after all held-out results (including the obs module's) had been seen; companion tables "
              "obs_governance*.csv and obs_replication.csv; prespecified tables untouched"),
    dict(id="obsint_global_cred_outside", date="2026-10-14", status="post_hoc",
         item="global-fit credibility of tissue edges outside U_glob = the global fit's credibility of the "
              "all-not-eligible pattern (= w_T), the analogue of governance's all-unobserved pattern",
         note="eligibility patterns of U_glob edges are identical in the tissue and global universes (checked)"),
    dict(id="obsint_no_mde", date="2026-10-14", status="post_hoc",
         item="minimum-detectable-O/E columns computed for family_hb_nuts (reproduction) only",
         note="design-only quantities, not used in any comparison; exact convolution is costly for 1e5-1e6-edge sets"),
    dict(id="obsint_degree_outside", date="2026-10-14", status="post_hoc",
         item="tissue TFs outside U_glob (literature degree 0) assigned to literature-degree quintile 1",
         note="global quintile 1 spans degrees 1-2; heldout_eval defines quintiles over the 935 global TFs only"),
    dict(id="obsint_rep_scope", date="2026-10-14", status="post_hoc",
         item="replication with the obs promoted sets at study level (comparators universe and observed), every "
              "universe; no power lock (the lock belongs to the primary promoted set)",
         note="seeds of replication.run_outcomes; family_hb_nuts rows reproduce rep_units.csv"),
]


# ----------------------------------------------------------------------------
# Logging and small helpers
# ----------------------------------------------------------------------------
def log(msg: str) -> None:
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    try:
        C.ensure_dir(MOD)
        with open(MOD / "run.log", "a") as fh:
            fh.write(line + "\n")
    except OSError:  # pragma: no cover
        pass


def file_md5(p: Path) -> str:
    h = hashlib.md5()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def _read_list(path: Path) -> list:
    return [x.strip() for x in path.read_text().splitlines() if x.strip()]


@contextlib.contextmanager
def no_power_analysis(active: bool):
    """Replace ``governance.oe_power_analysis`` by an all-NaN stub while ``active`` (post hoc fits)."""
    if not active:
        yield
        return
    orig = GOV.oe_power_analysis

    def stub(N, K, n, oe_targets=None, **kw):
        out = dict(E0=np.nan, n_strata=int(np.sum(np.asarray(n) > 0)), mde_oe=np.nan, mde_odds_ratio=np.nan,
                   max_attainable_oe=np.nan, mde_oe_poisson=np.nan, t_crit=np.nan)
        for k in (oe_targets or {}):
            out[f"power_at_{k}"] = np.nan
        return out

    GOV.oe_power_analysis = stub
    try:
        yield
    finally:
        GOV.oe_power_analysis = orig


def concentration_from_counts(counts: np.ndarray, names, n_tf_universe: int | None = None) -> dict:
    """``governance.concentration`` computed from per-TF edge counts (``names`` = TF names in the
    order of ``counts``); identical output for the same multiset of TFs (tested)."""
    counts = np.asarray(counts, dtype=np.int64)
    names = np.asarray(names, dtype=object)
    keep = counts > 0
    n = int(counts.sum())
    if n == 0:
        return dict(n_edges=0, n_tfs=0, top1_tf="", top1_share=np.nan, top2_share=np.nan,
                    entropy=np.nan, evenness=np.nan, entropy_norm_universe=np.nan, top_tfs="")
    vc = pd.Series(counts[keep], index=names[keep])
    vc = vc.sort_index().sort_values(ascending=False, kind="mergesort")
    p = vc.to_numpy(np.float64) / n
    H = max(0.0, float(-(p * np.log(p)).sum()))
    k = len(vc)
    return dict(n_edges=n, n_tfs=k, top1_tf=str(vc.index[0]), top1_share=float(p[0]),
                top2_share=float(p[:2].sum()), entropy=H,
                evenness=H / math.log(k) if k > 1 else np.nan,
                entropy_norm_universe=(H / math.log(n_tf_universe)) if n_tf_universe and n_tf_universe > 1 else np.nan,
                top_tfs=";".join(f"{t}:{c}" for t, c in vc.head(GOV.TOP_TFS_LISTED).items()))


# ----------------------------------------------------------------------------
# Evidence and credibilities
# ----------------------------------------------------------------------------
class Evidence:
    """Eligibility-coded evidence of a universe: patterns, edge-pattern map, evidence status."""

    def __init__(self, u: str):
        self.u = u
        R, n, fam, ann, base = E.load_patterns_elig(u, VARIANT, with_base=True)
        self.R, self.n, self.fam, self.ann, self.base = R, n.astype(np.float64), list(fam), list(ann), base
        self.ep = np.asarray(E.load_edge_pattern(u, ELIG, mmap=False), dtype=np.int64)
        self.n_plain = len(E.load_patterns(u, VARIANT)[1])
        obs = (R >= 0).any(1)
        rep = (R == 1).any(1)
        self.status_p = np.where(rep, 2, np.where(obs, 1, 0)).astype(np.int8)
        assert np.allclose(np.bincount(self.ep[self.ep >= 0], minlength=len(n)), n)


def fit_dir(u: str, fit_id: str) -> Path:
    return FITS / u / (ELIG if fit_id in OBS_FITS else VARIANT) / fit_id


def cred_file(u: str, fit_id: str) -> Path:
    return fit_dir(u, fit_id) / ("cred_mean.npy" if fit_id.endswith("_nuts") else "cred.npy")


def cred_patterns(ev: Evidence, fit_id: str) -> np.ndarray:
    """Credibility of every eligibility pattern of ``ev.u`` under ``fit_id``."""
    c = np.load(cred_file(ev.u, fit_id)).astype(np.float64)
    if fit_id in OBS_FITS:
        if len(c) != len(ev.n):
            raise RuntimeError(f"{ev.u}/{fit_id}: credibility not aligned with the eligibility patterns")
        return c
    if len(c) != ev.n_plain:
        raise RuntimeError(f"{ev.u}/{fit_id}: credibility not aligned with the stored patterns")
    return c[ev.base]


def obs_params(u: str, fit_id: str) -> dict:
    with np.load(fit_dir(u, fit_id) / "params.npz") as z:
        return dict(w=np.asarray(z["w"]), theta=np.asarray(z["theta"]), eta=np.asarray(z["eta"]))


def cred_unobservable(glob: Evidence, fit_id: str, row: np.ndarray, c_glob: np.ndarray) -> float:
    """Global-fit credibility of the pattern of edges outside U_glob (``row``: every annotator -2 in
    the eligibility coding, i.e. -1 = unobserved in the stored coding).  MAR fits: the stored global
    pattern with every annotator unobserved (as ``governance``); obs fits: the credibility of the
    all-not-eligible pattern under the global MAP parameters (= w_T)."""
    if fit_id in OBS_FITS:
        data = lcm.PatternData(row[None, :].astype(np.int8), np.ones(1), glob.fam, glob.ann, eligibility=True)
        return float(lcm.credibility_at(obs_params("global", fit_id), data)[0])
    cand = np.flatnonzero(np.all(glob.R < 0, axis=1))
    if len(cand) == 0:
        raise RuntimeError("global universe has no all-unobserved pattern")
    vals = np.unique(np.round(c_glob[cand], 15))
    if len(vals) != 1:                    # all-unobserved elig patterns share one base pattern
        raise RuntimeError("all-unobserved patterns of the global fit differ in credibility")
    return float(vals[0])


class TissueMap:
    """Edges of a tissue universe located in U_glob (fit-independent)."""

    def __init__(self, t: str, ev: Evidence, glob: Evidence, gtfs: list, ggenes: list):
        e = pd.read_parquet(C.OUT / "evidence" / t / "evidence.parquet", columns=["tf", "target"])
        self.tf = e["tf"].to_numpy(object)
        self.target = e["target"].to_numpy(object)
        G = len(ggenes)
        i = pd.Series(self.tf).map({x: k for k, x in enumerate(gtfs)}).to_numpy(dtype=float)
        j = pd.Series(self.target).map({x: k for k, x in enumerate(ggenes)}).to_numpy(dtype=float)
        self.in_global = ~np.isnan(i) & ~np.isnan(j)
        gidx = i[self.in_global].astype(np.int64) * G + j[self.in_global].astype(np.int64)
        self.gpat = glob.ep[gidx]
        if (self.gpat < 0).any():
            raise RuntimeError(f"{t}: a context edge maps to a self pair of the global grid")
        cp = ev.ep[self.in_global]
        if not np.array_equal(glob.R[self.gpat], ev.R[cp]):
            raise RuntimeError(f"{t}: eligibility patterns of U_glob edges differ between universes")
        out = np.unique(ev.R[ev.ep[~self.in_global]], axis=0)
        if len(out) > 1 or (len(out) == 1 and not (out == -2).all()):
            raise RuntimeError(f"{t}: edges outside U_glob are not all-not-eligible")
        self.out_row = np.full(ev.R.shape[1], -2, dtype=np.int8)
        self.n_outside = int((~self.in_global).sum())

    def global_cred(self, glob: Evidence, fit_id: str, c_glob: np.ndarray) -> np.ndarray:
        out = np.empty(len(self.tf))
        out[self.in_global] = c_glob[self.gpat]
        if self.n_outside:
            out[~self.in_global] = cred_unobservable(glob, fit_id, self.out_row, c_glob)
        return out


# ----------------------------------------------------------------------------
# (a) Governance
# ----------------------------------------------------------------------------
class ObsContext(GOV.Context):
    """Edge-level stand-in for ``governance.Context`` (each edge its own pattern, weight 1): the
    governance methods ``set_eval``, ``margins``, ``mde_cols``, ``base_rate``, ``attach_labels`` and
    the function ``validation_rows`` apply unchanged."""

    def __init__(self, t: str, tm: TissueMap, ev: Evidence, n_boot: int):    # noqa: D401 (no super())
        self.t, self.n_boot = t, n_boot
        self.tf, self.target = tm.tf, tm.target
        self.N = len(self.tf)
        self.ep = np.arange(self.N, dtype=np.int64)
        self.n = np.ones(self.N)
        self.tf_list = sorted(set(self.tf))
        self.n_tf_universe = len(self.tf_list)
        self.tf_code = pd.Series(self.tf).map({x: i for i, x in enumerate(self.tf_list)}).to_numpy(np.int64)
        self.status = ev.status_p[ev.ep]
        self.in_global = tm.in_global
        self.labels = {}

    def set_fit(self, fit_id: str, cg: np.ndarray, cc: np.ndarray) -> None:
        self.fit_id, self.cg_e, self.cc_e = fit_id, cg, cc
        self.lg = GOV.bfdr_levels(cg, self.n)
        self.lc = GOV.bfdr_levels(cc, self.n)
        self.dual = GOV.dual_tier(self.lg, self.lc, "primary")


def tier_masks(scope_levels: dict) -> dict:
    """(scope, tier) -> (order, boolean edge mask) for the governance tiers of one fit."""
    out = {}
    for scope, lev in scope_levels.items():
        if scope == "dual":
            for T in (3, 2, 1, 0):
                out[(scope, GOV.DUAL_NAMES[T])] = (T, lev == T)
            out[(scope, "Promoted")] = (4, lev >= 2)
        else:
            for L in (2, 1, 0):
                out[(scope, GOV.LEVEL_NAMES[L])] = (L, lev == L)
            out[(scope, "Promoted")] = (4, lev >= 1)
    return out


def tier_rows(universe: str, fit_id: str, masks: dict, cred: dict, tf_code: np.ndarray, tf_names,
              status: np.ndarray, n_tf_universe: int, primary_masks: dict | None) -> list:
    """Tier composition rows (``cred``: scope -> credibility per edge; dual -> both)."""
    rows = []
    N = len(tf_code)
    for (scope, tier), (order, m) in masks.items():
        cnt = np.bincount(tf_code[m], minlength=len(tf_names))
        con = concentration_from_counts(cnt, tf_names, n_tf_universe)
        nm = int(m.sum())
        row = dict(universe=universe, fit=fit_id, scope=scope, tier=tier, tier_order=order, n_edges=nm,
                   share_of_universe=nm / N, n_tfs=con["n_tfs"])
        for s in ("global_fit", "context_fit"):
            if s in cred:
                c = cred[s][m]
                row[f"cred_{s}_min"] = float(c.min()) if nm else np.nan
                row[f"cred_{s}_max"] = float(c.max()) if nm else np.nan
                row[f"bfdr_{s}"] = float(np.mean(1 - c)) if nm else np.nan
        st = status[m]
        for k, name in enumerate(STATUS_NAMES):
            row[f"share_{name}"] = float(np.mean(st == k)) if nm else np.nan
        row.update({k: v for k, v in con.items() if k not in ("n_edges", "n_tfs")})
        if primary_masks is not None and (scope, tier) in primary_masks:
            pm = primary_masks[(scope, tier)][1]
            inter, union = int((m & pm).sum()), int((m | pm).sum())
            row.update(n_primary=int(pm.sum()), n_shared_with_primary=inter,
                       jaccard_with_primary=inter / union if union else np.nan)
        rows.append(row)
    return rows


def global_masks(ev: Evidence, c_p: np.ndarray) -> tuple[np.ndarray, dict]:
    lev_p = GOV.bfdr_levels(c_p, ev.n)
    ok = ev.ep >= 0
    lev = np.full(len(ev.ep), -1, dtype=np.int8)
    lev[ok] = lev_p[ev.ep[ok]]
    m = tier_masks({"global_fit": lev[ok]})
    return lev, m


def run_governance(n_boot: int = GOV.N_BOOT) -> dict:
    t0 = time.time()
    log(f"[governance] fits {list(FIT_IDS)}; universes {list(UNIVERSES)}; n_boot {n_boot}")
    glob = Evidence("global")
    gtfs = _read_list(C.OUT / "evidence" / "global" / "tfs.txt")
    ggenes = _read_list(C.OUT / "evidence" / "global" / "genes.txt")
    pooled = GOV.load_pert_pooled(log_fn=log)
    tier_out, val_out, trend_out = [], [], []
    info = dict(universes={})
    gcred = {f: cred_patterns(glob, f) for f in FIT_IDS}
    okg = glob.ep >= 0
    gtf_code = np.repeat(np.arange(len(gtfs)), len(ggenes))[okg]
    gstatus = glob.status_p[glob.ep[okg]]
    # ---- global universe
    prim = None
    for f in FIT_IDS:
        lev, masks = global_masks(glob, gcred[f])
        if f == PRIMARY:
            prim = masks
        cedge = gcred[f][glob.ep[okg]]
        tier_out += tier_rows("global", f, masks, {"global_fit": cedge}, gtf_code, np.array(gtfs, dtype=object),
                              gstatus, len(gtfs), prim)
        fake = dict(nuts={"mean": gcred[f]}, n=glob.n, ep=glob.ep, genes=ggenes, tfs=gtfs)
        with no_power_analysis(f != PRIMARY):
            v, tr, lab = GOV.global_validation(fake, pooled, n_boot)
        for r in v + tr:
            r["fit"] = f
        val_out += v
        trend_out += tr
        log(f"[governance] global {f}: Strong/Moderate {int(glob.n[GOV.bfdr_levels(gcred[f], glob.n) == 2].sum()):,}/"
            f"{int(glob.n[GOV.bfdr_levels(gcred[f], glob.n) == 1].sum()):,}")
    info["universes"]["global"] = dict(n_edges=int(okg.sum()), labels=lab)
    global_oe = {f: {(r["tier"], r["reference"]): r["oe_tf"] for r in val_out
                     if r["universe"] == "global" and r["fit"] == f} for f in FIT_IDS}
    # ---- tissues
    for t in C.TISSUES:
        ev = Evidence(t)
        if ev.ann != glob.ann:
            raise RuntimeError(f"{t}: annotator order differs from the global universe")
        tm = TissueMap(t, ev, glob, gtfs, ggenes)
        cx = ObsContext(t, tm, ev, n_boot)
        cx.attach_labels(GOV.build_reference_indices(sorted(set(cx.target)), pooled))
        names = np.array(cx.tf_list, dtype=object)
        prim = None
        uinfo = dict(n_edges=cx.N, n_edges_outside_global=tm.n_outside,
                     n_tfs_outside_global=int(len(set(cx.tf[~tm.in_global]))), eligibility_identity_check="passed",
                     global_cred_outside={})
        for f in FIT_IDS:
            cc = cred_patterns(ev, f)[ev.ep]
            cg = tm.global_cred(glob, f, gcred[f])
            if tm.n_outside:
                uinfo["global_cred_outside"][f] = float(cg[~tm.in_global][0])
            cx.set_fit(f, cg, cc)
            masks = tier_masks({"global_fit": cx.lg, "context_fit": cx.lc, "dual": cx.dual})
            if f == PRIMARY:
                prim = masks
            joint = np.minimum(cg, cc)
            trows = tier_rows(t, f, masks, {"global_fit": cg, "context_fit": cc}, cx.tf_code, names, cx.status,
                              cx.n_tf_universe, prim)
            for r in trows:
                if r["scope"] == "dual":
                    m = masks[(r["scope"], r["tier"])][1]
                    r["cred_joint_min"] = float(joint[m].min()) if m.any() else np.nan
            tier_out += trows
            with no_power_analysis(f != PRIMARY):
                v, tr = GOV.validation_rows(cx, global_oe[f])
            for r in v + tr:
                r["fit"] = f
            val_out += v
            trend_out += tr
            log(f"[governance] {t} {f}: dual Strong {int((cx.dual == 3).sum()):,}, promoted "
                f"{int((cx.dual >= 2).sum()):,} of {cx.N:,}")
        info["universes"][t] = uinfo
    tiers = finish(pd.DataFrame(tier_out), "tiers")
    val = finish(pd.DataFrame(val_out), "validation")
    trend = pd.DataFrame(trend_out)
    trend["p_one_sided_holm"] = np.nan
    trend["p_two_sided_holm"] = np.nan
    trend["holm_family_size"] = np.nan
    for _, idx in trend.groupby(["fit", "scope", "strata"]).groups.items():
        trend.loc[idx, "p_one_sided_holm"] = GOV.holm(trend.loc[idx, "p_one_sided"].to_numpy())
        trend.loc[idx, "p_two_sided_holm"] = GOV.holm(trend.loc[idx, "p_two_sided"].to_numpy())
        trend.loc[idx, "holm_family_size"] = int(np.isfinite(trend.loc[idx, "p_one_sided"].to_numpy()).sum())
    trend["holm_family"] = ("Holm within (fit, scope, strata) across all universes (3 tissues + global) x 4 references "
                            "(the governance family, one per fit)")
    trend = finish(trend, "trend")
    info["runtime_s"] = time.time() - t0
    return dict(tiers=tiers, validation=val, trend=trend, info=info)


# ----------------------------------------------------------------------------
# (a') Hub over-promotion by TF literature-degree quintile
# ----------------------------------------------------------------------------
def tissue_dual(t: str, ev: Evidence, tm: TissueMap, glob: Evidence, gcred_f: np.ndarray, fit_id: str) -> np.ndarray:
    cc = cred_patterns(ev, fit_id)[ev.ep]
    cg = tm.global_cred(glob, fit_id, gcred_f)
    return GOV.dual_tier(GOV.bfdr_levels(cg, np.ones(len(cg))), GOV.bfdr_levels(cc, np.ones(len(cc))), "primary")


def degree_rows_for_sets(universe: str, ref: str, es, sets: dict, qmap: dict, bounds: pd.DataFrame,
                         stream_parts: tuple) -> list:
    """``heldout_eval.run_degree`` quantities for the edge sets ``sets`` (name -> (fit, scope, tier,
    weight over the evaluable edges)); one bootstrap matrix per (universe, reference)."""
    from r2 import heldout_eval as HE
    q = np.array([qmap.get(x, 1) for x in es.tf])
    q_tf = np.array([qmap.get(x, 1) for x in es.tfs])
    outside = np.array([x not in qmap for x in es.tfs])
    M = HE.tf_bootstrap_weights(es.n_tf, HE.N_BOOT, HE._stream(*stream_parts))
    P = es.y.sum()
    rows = []
    for (fit_id, scope, tier), w in sets.items():
        wsum = w.sum()
        if wsum <= 0:
            continue
        w_t = np.bincount(es.tf_code, weights=w, minlength=es.n_tf)
        p_t = np.bincount(es.tf_code, weights=es.y.astype(float), minlength=es.n_tf)
        for qq in range(1, 6):
            inq = q == qq
            tq = q_tf == qq
            n_q, pos_q, set_q = int(inq.sum()), float(es.y[inq].sum()), float(w[inq].sum())
            share_pos, share_set = pos_q / P, set_q / wsum
            with np.errstate(invalid="ignore", divide="ignore"):
                b_share_set = (M[:, tq] @ w_t[tq]) / (M @ w_t)
                b_share_pos = (M[:, tq] @ p_t[tq]) / (M @ p_t)
                b_opi = b_share_set / b_share_pos
            row = dict(universe=universe, reference=ref, fit=fit_id, scope=scope, tier=tier,
                       set_size=float(wsum), quintile=qq,
                       degree_min=int(0 if (qq == 1 and outside.any()) else bounds.loc[qq, "min"]),
                       degree_max=int(bounds.loc[qq, "max"]), n_tfs_universe=int(bounds.loc[qq, "size"]),
                       n_tfs_eval=int(tq.sum()), n_tfs_outside_global=int((tq & outside).sum()),
                       n_eval=n_q, n_pos=int(pos_q), share_eval=n_q / es.n, share_pos=share_pos,
                       share_set=share_set, over_promotion=share_set / share_pos if share_pos > 0 else np.nan,
                       pos_rate=pos_q / n_q if n_q else np.nan,
                       set_precision=float((w[inq] * es.y[inq]).sum() / set_q) if set_q > 0 else np.nan,
                       set_precision_all=float((w * es.y).sum() / wsum))
            row["over_promotion_lo"], row["over_promotion_hi"] = HE.percentile_ci(b_opi)
            rows.append(row)
    return rows


def run_degree() -> pd.DataFrame:
    from r2 import heldout_eval as HE
    t0 = time.time()
    deg = HE.tf_literature_degree("global")
    qmap = dict(zip(deg["tf"], deg["quintile"]))
    bounds = deg.groupby("quintile")["lit_degree"].agg(["min", "max", "size"])
    glob = Evidence("global")
    gtfs = _read_list(C.OUT / "evidence" / "global" / "tfs.txt")
    ggenes = _read_list(C.OUT / "evidence" / "global" / "genes.txt")
    gcred = {f: cred_patterns(glob, f) for f in FIT_IDS}
    rows = []
    for u in UNIVERSES:
        if u == "global":
            levels = {}
            for f in FIT_IDS:
                lev_p = GOV.bfdr_levels(gcred[f], glob.n)
                levels[f] = lev_p
            ep = glob.ep
        else:
            ev = Evidence(u)
            tm = TissueMap(u, ev, glob, gtfs, ggenes)
            duals = {f: tissue_dual(u, ev, tm, glob, gcred[f], f) for f in FIT_IDS}
        for ref in DEGREE_REFS:
            es = HE.load_evalset(u, ref)
            if es.n == 0 or es.y.sum() == 0:
                continue
            sets = {}
            for f in FIT_IDS:
                if u == "global":
                    p = ep[es.edge]
                    lv = np.where(p >= 0, levels[f][np.maximum(p, 0)], -1)
                    sets[(f, "global_fit", "Strong")] = (lv == 2).astype(float)
                    sets[(f, "global_fit", "Promoted")] = (lv >= 1).astype(float)
                else:
                    d = duals[f][es.edge]
                    sets[(f, "dual", "Strong")] = (d == 3).astype(float)
                    sets[(f, "dual", "Promoted")] = (d >= 2).astype(float)
            parts = ("boot_degree", ref) if u == "global" else ("boot_degree", ref, u)
            rows += degree_rows_for_sets(u, ref, es, sets, qmap, bounds, parts)
            log(f"[degree] {u}/{ref} done")
    df = pd.DataFrame(rows)
    df["few_tfs"] = df["n_tfs_eval"] < HE.FEW_TFS
    df = finish(df, "degree")
    log(f"[degree] {len(df)} rows in {time.time() - t0:.0f} s")
    return df


# ----------------------------------------------------------------------------
# (b) Replication
# ----------------------------------------------------------------------------
def attach_edges(u: str, de: pd.DataFrame, gtfs: list | None = None, ggenes: list | None = None) -> pd.DataFrame:
    """``replication.attach_patterns`` returning the universe edge id (``edge``) instead of the
    pattern (same rows, same order; self pairs of the global grid dropped)."""
    if de.empty:
        return de.assign(edge=np.zeros(0, dtype=np.int64))
    ep = np.asarray(E.load_edge_pattern(u, VARIANT, mmap=True))
    if u == "global":
        G = len(ggenes)
        ti = pd.Index(gtfs).get_indexer(de["tf"])
        gi = pd.Index(ggenes).get_indexer(de["target"])
        ok = (ti >= 0) & (gi >= 0)
        out = de.loc[ok].copy()
        out["edge"] = ti[ok].astype(np.int64) * G + gi[ok]
    else:
        ev = pd.read_parquet(E.EV_DIR / u / "evidence.parquet", columns=["tf", "target"])
        ev = ev.assign(edge=np.arange(len(ev), dtype=np.int64))
        out = de.merge(ev, on=["tf", "target"], how="inner", validate="one_to_one")
    out = out[np.asarray(ep)[out["edge"].to_numpy()] >= 0]
    return out.reset_index(drop=True)


def replication_edges(u: str, ct: pd.DataFrame, gtfs: list, ggenes: list) -> pd.DataFrame:
    tfs = RP.universe_tfs(u)
    frames = []
    for _, row in ct.iterrows():
        de = RP.read_cohort_de(row["cohort_id"], tfs)
        de = de[~de["cis"]].drop(columns="cis")
        ed = attach_edges(u, de, gtfs, ggenes)
        ed.insert(0, "cohort", row["cohort_id"])
        frames.append(ed)
    return pd.concat(frames, ignore_index=True)


def replication_units(u: str, fit_id: str, ed: pd.DataFrame, units: dict) -> list:
    rows = []
    for comp in RP.COMPARATORS:
        sub = ed if comp == "universe" else ed[ed["observed"]]
        strata = RP.outcome_strata(sub)
        ctrl = {c: RP.posthoc_controls(ec[["tf", "target", "promoted", "positive"]], c)
                for c, ec in sub.groupby("cohort", sort=True)}
        for unit, cohorts in units.items():
            d = strata[strata["cohort"].isin(cohorts)]
            status = RP.unit_status(d)
            N_tot, n_tot, K_tot = int(d["N"].sum()), int(d["n"].sum()), int(d["K"].sum())
            base = dict(universe=u, fit=fit_id, level=REP_LEVEL, unit=unit, comparator=comp,
                        cohorts=";".join(cohorts), status=status, n_strata=int(len(d)),
                        n_strata_informative=int(RP.informative(d["N"], d["n"], d["K"]).sum()) if len(d) else 0,
                        n_evaluable_edges=N_tot, n_promoted_evaluable=n_tot,
                        n_strong_evaluable=int(d["n_strong"].sum()), n_positives=K_tot,
                        positive_rate=K_tot / N_tot if N_tot else np.nan)
            if status == "evaluable":
                base.update(RP.unit_outcome(d, RP.unit_seed("outcome", u, comp, ";".join(sorted(cohorts)))))
                base.update(RP.unit_posthoc([ctrl[c] for c in cohorts if c in ctrl]))
                w = base.get("wrongtf_enrichment", np.nan)
                base["specific_excess"] = base["enrichment"] / w if np.isfinite(w) and w > 0 else np.nan
                w2 = base.get("wrongtf_excl_shared_enrichment", np.nan)
                base["specific_excess_excl_shared"] = base["enrichment"] / w2 if np.isfinite(w2) and w2 > 0 else np.nan
            else:
                base["endpoint_status"] = "non-evaluable"
            rows.append(base)
    return rows


def run_replication() -> pd.DataFrame:
    t0 = time.time()
    ct = RP.cohort_table()
    units = RP.units_of(None, REP_LEVEL)
    gtfs = _read_list(C.OUT / "evidence" / "global" / "tfs.txt")
    ggenes = _read_list(C.OUT / "evidence" / "global" / "genes.txt")
    rows = []
    for u in UNIVERSES:
        ev = Evidence(u)
        ed0 = replication_edges(u, ct, gtfs, ggenes)
        pat = ev.ep[ed0["edge"].to_numpy()]
        for f in FIT_IDS:
            tier_p = RP.pattern_tiers(cred_patterns(ev, f), ev.n)
            ed = ed0.assign(tier=tier_p[pat], observed=ev.status_p[pat] > 0)
            ed["promoted"] = ed["tier"] > 0
            rows += replication_units(u, f, ed, units)
            log(f"[replication] {u} {f}: promoted evaluable {int(ed['promoted'].sum()):,} of {len(ed):,} "
                f"(Strong {int((ed['tier'] == 2).sum()):,})")
    res = pd.DataFrame(rows)
    res["p_holm"] = np.nan
    for _, g in res.groupby(["universe", "fit", "comparator"]):
        e = g[g["status"] == "evaluable"]
        res.loc[e.index, "p_holm"] = RP.holm(e["p_exact"].to_numpy())
    res["endpoint_pass"] = res["endpoint_pass"].astype("boolean")
    res = finish(res, "replication")
    log(f"[replication] {len(res)} rows in {time.time() - t0:.0f} s")
    return res


# ----------------------------------------------------------------------------
# Status, reproduction check and output
# ----------------------------------------------------------------------------
REPRO = {
    "validation": dict(table="gov_tier_validation.csv", keys=["universe", "scope", "tier", "reference"],
                       values=["n_edges", "n_eval", "obs", "exp_tf", "oe_tf", "oe_tf_lo", "oe_tf_hi", "exp_tfdeg",
                               "oe_tfdeg", "oe_tfdeg_lo", "oe_tfdeg_hi", "mde_oe", "power_at_global_oe"]),
    "trend": dict(table="gov_tier_trend.csv", keys=["universe", "scope", "reference", "strata"],
                  values=["n_eval", "n_pos", "T", "E", "V", "z", "p_one_sided", "p_one_sided_holm"]),
    "tiers": dict(table="gov_tiers.csv", keys=["universe", "scope", "tier"], values=["n_edges", "n_tfs"],
                  rename={"tissue": "universe"}),
    "degree": dict(table="heldout_degree_calibration.csv", keys=["reference", "quintile"],
                   values=["set_size", "n_eval", "n_pos", "share_set", "over_promotion", "over_promotion_lo",
                           "over_promotion_hi"],
                   filter=dict(score=PRIMARY, set="strong_bfdr05"), own_filter=dict(universe="global", tier="Strong")),
    "replication": dict(table="rep_units.csv", keys=["universe", "unit", "comparator"],
                        values=["n_evaluable_edges", "n_promoted_evaluable", "n_positives", "enrichment",
                                "mh_odds_ratio", "p_exact", "p_perm", "enrichment_boot_lo", "enrichment_boot_hi",
                                "wrongtf_enrichment", "specific_excess", "resp_strat_enrichment", "p_holm"],
                        filter=dict(level=REP_LEVEL)),
}
_REPRO_RESULTS: dict = {}


def finish(df: pd.DataFrame, kind: str) -> pd.DataFrame:
    """Add fit descriptors, analysis_status (post_hoc unless a family_hb_nuts row reproduces a
    prespecified-table row exactly) and the reproduction record."""
    df = df.copy()
    for k in ("model", "observation_model", "inference", "role"):
        df[k] = df["fit"].map({f: FIT_INFO[f][k] for f in FIT_INFO})
    df["analysis_status"] = "post_hoc"
    df["reproduces"] = ""
    spec = REPRO.get(kind)
    rec = dict(table=spec["table"] if spec else None, n_primary_rows=0, n_matched=0, n_reproduced=0, max_rel_diff=0.0,
               mismatches=[])
    if spec is not None:
        ref = pd.read_csv(C.TABLES / spec["table"]).rename(columns=spec.get("rename", {}))
        for k, v in spec.get("filter", {}).items():
            ref = ref[ref[k] == v]
        own = df["fit"] == PRIMARY
        for k, v in spec.get("own_filter", {}).items():
            own &= df[k] == v
        keys = spec["keys"]
        refi = ref.set_index(keys)
        if refi.index.has_duplicates:
            raise RuntimeError(f"{spec['table']}: duplicated keys {keys}")
        for i in df.index[own]:
            rec["n_primary_rows"] += 1
            key = tuple(df.loc[i, k] for k in keys)
            if key not in refi.index:
                continue
            rec["n_matched"] += 1
            r = refi.loc[key]
            bad = []
            for c in spec["values"]:
                if c not in r.index or c not in df.columns:
                    continue
                a, b = df.loc[i, c], r[c]
                if pd.isna(a) and pd.isna(b):
                    continue
                try:
                    fa, fb = float(a), float(b)
                except (TypeError, ValueError):
                    if str(a) != str(b):
                        bad.append(c)
                    continue
                if np.isinf(fa) and np.isinf(fb) and np.sign(fa) == np.sign(fb):
                    continue
                # most prespecified tables are written with 6 significant digits (rep_units.csv at full
                # precision): equal at 6 significant digits, or within a relative 1e-9
                rel = abs(fa - fb) / abs(fb) if fb != 0 else abs(fa)
                if not (float("%.6g" % fa) == float("%.6g" % fb) or rel <= 1e-9):
                    bad.append(c)
                elif np.isfinite(rel):
                    rec["max_rel_diff"] = max(rec["max_rel_diff"], rel)
            if bad:
                rec["mismatches"].append(dict(key=[str(x) for x in key], columns=bad))
            else:
                rec["n_reproduced"] += 1
                df.loc[i, "analysis_status"] = r.get("analysis_status", "prespecified")
                df.loc[i, "reproduces"] = spec["table"]
    _REPRO_RESULTS[kind] = rec
    return df


def write(df: pd.DataFrame, kind: str) -> Path:
    p = C.ensure_dir(C.TABLES) / TABLE[kind]
    df.to_csv(p, index=False, float_format="%.6g")
    log(f"  wrote {p} ({len(df)} rows)")
    return p


def inputs_record() -> dict:
    rec = {}
    for u in UNIVERSES:
        for f in FIT_IDS:
            p = cred_file(u, f)
            rec[f"{u}/{f}"] = dict(path=str(p.relative_to(C.SP)), md5=file_md5(p))
    for kind in REPRO.values():
        p = C.TABLES / kind["table"]
        rec[kind["table"]] = dict(md5=file_md5(p), mtime=time.strftime("%Y-%m-%d %H:%M:%S",
                                                                        time.localtime(p.stat().st_mtime)))
    return rec


def run_all(stages=("governance", "degree", "replication"), n_boot: int = GOV.N_BOOT) -> dict:
    t0 = time.time()
    C.ensure_dir(MOD)
    meta = dict(module="obs_integration", started=time.strftime("%Y-%m-%d %H:%M:%S"), stages=list(stages),
                fits={f: FIT_INFO[f] for f in FIT_IDS}, variant=VARIANT, elig_variant=ELIG, n_boot=n_boot, seed=SEED,
                q_strong=GOV.Q_STRONG, q_moderate=GOV.Q_MODERATE, degree_references=list(DEGREE_REFS),
                replication_level=REP_LEVEL, analysis_decisions=ANALYSIS_DECISIONS)
    if "governance" in stages:
        g = run_governance(n_boot)
        for k in ("tiers", "validation", "trend"):
            write(g[k], k)
        meta["governance"] = g["info"]
    if "degree" in stages:
        write(run_degree(), "degree")
    if "replication" in stages:
        write(run_replication(), "replication")
    meta["reproduction"] = dict(_REPRO_RESULTS)
    meta["inputs"] = inputs_record()
    meta["runtime_s"] = time.time() - t0
    C.write_json(meta, MOD / "meta.json")
    GOV.record_decisions("obs_integration", ANALYSIS_DECISIONS)
    for k, r in _REPRO_RESULTS.items():
        log(f"[check] {k}: {r['n_reproduced']}/{r['n_matched']} family_hb_nuts rows reproduce {r['table']} "
            f"(of {r['n_primary_rows']}; max rel diff {r['max_rel_diff']:.2g}; mismatches {len(r['mismatches'])})")
    bad = {k: r["mismatches"] for k, r in _REPRO_RESULTS.items() if r["mismatches"]}
    if bad:
        raise RuntimeError(f"family_hb_nuts rows do not reproduce the prespecified tables: {bad}")
    log(f"done in {meta['runtime_s']:.0f} s")
    return meta


def check() -> dict:
    """Freshness check: the credibility files and reproduced tables still have the md5 recorded."""
    meta = json.loads((MOD / "meta.json").read_text())
    now = inputs_record()
    stale = [k for k, v in meta["inputs"].items() if now.get(k, {}).get("md5") != v["md5"]]
    out = dict(checked=time.strftime("%Y-%m-%d %H:%M:%S"), stale_inputs=stale, ok=not stale)
    C.write_json(out, MOD / "output_check.json")
    log(f"[check] {'ok' if not stale else 'STALE: ' + ', '.join(stale)}")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("stage", choices=["all", "governance", "degree", "replication", "check"])
    ap.add_argument("--n-boot", type=int, default=GOV.N_BOOT)
    a = ap.parse_args(argv)
    if a.stage == "check":
        check()
        return
    run_all(("governance", "degree", "replication") if a.stage == "all" else (a.stage,), a.n_boot)


if __name__ == "__main__":
    main()
