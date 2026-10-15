"""Cross-cohort replication check of the promoted set (stage E5; reviewer Q5, Q8, Q17).

Question
--------
Do held-out perturbation cohorts show more differentially expressed targets among the
*promoted* edges of a TF than among its other edges?  This is a descriptive replication
check, not a validation: positives in CRISPR perturbation screens include indirect
(downstream) effects, most cohorts are K562 and the number of independent studies is small.

Promoted set
------------
Global universe (primary; tissue universes are secondary where evaluable), primary evidence
variant (``PRESPECIFIED['evidence_variant_primary']``), Family-HB NUTS posterior-mean
credibility (``outputs/r2/fits/<universe>/<variant>/family_hb_nuts/cred_mean.npy``).
Tiers are label-free Bayesian-FDR sets over whole tie blocks (``bfdr_set``):
Strong = largest prefix of tie blocks (decreasing credibility) with edge-weighted mean
(1 - c) <= ``bfdr_strong``; Moderate = the same at ``bfdr_moderate`` minus Strong.
Promoted = Strong U Moderate.  No perturbation data enter the promoted set.

Evaluable edges and units
-------------------------
For cohort c and TF s (perturbed in c and a TF of the universe), the evaluable edges are the
universe edges (s, g) that c tested, joined on (tf, target) keys, with same-cytoband cis pairs
(``cis_same_band``) removed from positives AND negatives.  Positive = ``positive_primary``.
Comparators: ``universe`` (all evaluable edges; primary) and ``observed`` (only edges observed by
>= 1 annotator; sensitivity, removes the never-observed edges that can never be promoted).
The ``observed`` comparator means "inside at least one annotator's observation mask" (edge
observed, reported or not), not "seen in the literature".
Units: ``cohort`` (cohort_id) and ``study`` (independent unit = ``study_id`` of
``data/r2/tables/cohort_table.csv``; Dixit d7 + d13 are one study, its strata are cohort x TF).
A stratum is a (cohort, TF) pair.  A stratum is *informative* for the within-TF test iff
0 < n_s < N_s and 0 < K_s < N_s (N_s evaluable edges, n_s promoted evaluable edges, K_s positives).
A unit without an informative stratum is *non-evaluable* (reported, never counted as a failure).

Prespecified within-TF test
---------------------------
T = sum_s X_s, X_s = promoted positives of stratum s.  One-sided exact conditional test of
no association given every stratum's margins (null: X_s ~ Hypergeometric(N_s, K_s, n_s)
independently; the null distribution of T is their convolution; this is the exact
Cochran-Mantel-Haenszel test).  p = P0(T >= t_obs).  Effect sizes: TF-stratified enrichment
O/E with O = T and E = sum_s n_s K_s / N_s, and the Mantel-Haenszel odds ratio.  Endpoint
("replication check passed"): O/E >= ``replication_endpoint_enrichment`` and
p <= ``replication_alpha``.

STEP 1 (``power``; written and time-stamped before any outcome is computed)
---------------------------------------------------------------------------
Uses only the design: N_s, n_s per stratum and, as a nuisance, the unit's overall positive
rate pbar (all evaluable edges).  The promoted/non-promoted split of positives is never read.

* ``power``: H1 = true enrichment RR (relative risk, promoted vs non-promoted) =
  ``replication_true_enrichment``; positive probability p0 = pbar N / (N - n + RR n) for
  non-promoted and p1 = min(1, RR p0) for promoted edges (so that the overall rate is pbar),
  homogeneous over strata.  Monte-Carlo power of the exact test at ``replication_alpha`` with
  ``N_POWER_SIMS`` simulated designs (common random numbers over RR).
* ``mde_rr``: smallest RR with power >= ``replication_power`` (bisection in log RR; ``inf`` if
  not reached even when every promoted edge is positive).
* ``floor_edges`` (the power-based support floor): smallest number m of promoted evaluable
  edges for which the exact one-sided Fisher test in a single stratum of the unit's size
  (N evaluable edges, rate pbar) has power >= ``replication_power`` against RR; exact
  (enumeration of the binomial outcomes).  ``floor_expected_positives`` = floor_edges x pbar
  (nearly rate-free for small pbar).  A unit is *adequately powered* iff
  n_promoted_evaluable >= floor_edges; ``power`` of the actual stratified design is reported
  alongside.
* Secondary, ``power_cond``: exact power conditional on the observed per-stratum positive
  counts K_s (margins only), under a common odds ratio equal to RR at pbar.  Shows the loss
  of power when positives concentrate in few TFs.

Outputs of step 1: ``data/r2/tables/rep_power.csv``, ``outputs/r2/replication/strata_design.csv``,
``outputs/r2/replication/power_lock.json`` (UTC time stamp, sha256 of both files) and an
append-only ``power_lock_history.jsonl``.  Steps 2-5 refuse to run if the lock is missing or the
design they recompute differs from the locked design.

STEP 2 (``outcomes``)
--------------------
Per unit: O, E, O/E, unstratified enrichment, MH OR, exact p, within-TF permutation p
(``n_null_draws`` permutations), Holm-adjusted p across evaluable units, TF-cluster bootstrap
95% percentile CIs (``n_bootstrap`` resamples of the TFs with >= 1 promoted evaluable edge), tier
split (Strong / Moderate), endpoint.  Pass rates (Wilson 95% CI) over evaluable units, over
adequately powered units and over non-K562 units.

STEP 3 (``threshold``)
---------------------
The data-derived support threshold of the R1 analysis, re-created to show its instability:
tau* = the support value maximising the pass rate among units with support >= tau (ties:
smallest tau; >= 2 units retained), chosen *from the outcomes*.  Re-estimated after leaving
each unit out (LOO); the left-out unit is then classified by tau_(-u).

STEP 4 (``bayes``)
-----------------
Random-effects logistic model (numpyro NUTS, 4 chains x 1000 warm-up + 1000 draws, target
acceptance 0.99) for unit u and stratum s (strata with >= 1 positive):

    y1_us ~ Binomial(n_us, sigmoid(a_us + theta_u)),  y0_us ~ Binomial(N_us - n_us, sigmoid(a_us))
    a_us ~ Normal(m_u, s_u),  m_u ~ Normal(-4, 3),  s_u ~ HalfNormal(2)          (weak)
    (a_us centred for strata with >= 10 positives, non-centred otherwise; same density)
    theta_u = mu + sigma eta_u,  eta_u ~ Normal(0, 1)                             (non-centred)
    mu ~ Normal(0, mu_sd),  sigma ~ HalfNormal(sigma_sd)

theta_u is a within-TF log odds ratio.  Reported: posterior of mu, sigma, theta_u,
P(exp(theta_u) >= endpoint) per unit and for a new unit (theta_new ~ N(mu, sigma)).
Prior profiles (``PRIOR_PROFILES``) give the prior-sensitivity table.

STEP 5 (``leavek``)
------------------
LkO (k = 1, 2, 3; "leave-k-units-out"): the random-effects model is refitted once for every
subset of k units removed from the evaluable units (all C(U, k) subsets; L1O = leave one
out, L2O = leave two out, L3O = leave three out).  Drift: delta of the posterior mean of mu
and of P(exp(mu) >= endpoint), and |delta| of the per-unit pass probabilities of the units
kept, summarised per k (mean and max).

Revision 2026-10-08 (verifier ``rep``; computed after the step-1 lock, ``post_hoc``)
----------------------------------------------------------------------------------
Columns listed in ``post_hoc_columns`` of ``rep_units.csv`` (rows stay ``analysis_status =
prespecified``; the lock and ``rep_power.csv`` are untouched):

* target responsiveness: resp(s, g) = share of the cohort's other perturbations (all perturbed
  genes, cis pairs excluded, the edge's own TF left out) under which g is DE.  ``resp_mean_*`` =
  within-TF mean responsiveness of promoted / non-promoted targets (weighted by n_s); TF x
  responsiveness-decile strata (deciles of resp over the unit's evaluable edges of each cohort;
  tied values share a bin, so heavily tied cohorts get fewer than 10 bins) give ``resp_strat_*``
  (O/E, exact p, MH OR, endpoint).
* wrong-TF control: each TF s's evaluable targets scored against the DE calls of every other
  universe TF s' perturbed in the same cohort; strata (s, s'), O = promoted-for-s targets DE
  under s', E = sum n K / N (``wrongtf_*``; ``wrongtf_excl_shared_*`` drops pairs whose target is
  also promoted for s').  ``specific_excess`` = enrichment / wrongtf_enrichment.
* ``max_attainable_enrichment`` = sum_s min(n_s, K_s) / E (margins); ``endpoint_attainable`` =
  max >= endpoint; ``endpoint_status`` in {pass, fail, endpoint_unattainable, non-evaluable}.
* ``endpoint_power_rr2``: Monte-Carlo power of the endpoint itself (O/E >= endpoint and p <=
  alpha) at RR = ``replication_true_enrichment`` (same design, seed and simulated tables as the
  locked test power).
* ``floor_edges_min_scan``: the locked ``floor_edges`` is the first crossing of the exponential-
  search / bisection bracket of a saw-toothed exact power curve; this column is the smallest m
  with power >= target found by a linear scan below it.
* bootstrap CIs resample TFs within the unit (cohort x TF strata of the same TF together; was:
  strata) and are not reported with fewer than ``MIN_BOOT_CLUSTERS`` = 5 TFs.
* ``rep_bayes.csv``: per-unit R-hat / ESS of theta_u; ``rep_prior_sensitivity.csv``: a TF-level
  random-slope profile (theta_us = theta_u + tau eps_s, tau ~ HalfNormal(0.5)); ``rep_leavek.csv``:
  ``degenerate_single_unit`` flags fits that keep one unit (mu is then that unit's theta shrunk to
  the prior).

Re-run on the final fits (2026-10-13).  ``fit_primary.py`` replaced the ``family_hb_nuts`` fits of all
four universes by frozen 4 x 1000/1000 runs after the lock was written, which moved a few boundary
edges between tiers (global Strong 4,064 -> 4,060, Moderate 3,270 -> 3,274).  The stratum design
(N_s, n_s, n_strong_s, K_s of every cohort x TF stratum) recomputed from the final fits has the same
digest as the locked design, so the lock of 2026-10-08T16:40:31Z stays valid and ``rep_power.csv`` is
unchanged; ``outcomes`` records this check in ``outputs/r2/replication/lock_check.json`` (tier sizes
at the lock and now) and in the ``promoted_set_changed_since_lock`` column of ``rep_units.csv``.  Had
the design changed, ``power --relock-reason ...`` would have been required (the history file keeps
every lock, and a re-lock states that outcomes of the earlier promoted set had been seen).

Manuscript tables (``data/r2/tables``)
-------------------------------------
``rep_power.csv``, ``rep_units.csv``, ``rep_pass_rates.csv``, ``rep_threshold_loo.csv``,
``rep_bayes.csv``, ``rep_prior_sensitivity.csv``, ``rep_leavek.csv``.

Command line::

    export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
    .venv/bin/python subproject_24_multireference_latent_truth_model/src/r2/replication.py all
    # or step by step (power must run first)
    ... replication.py power | outcomes | threshold | bayes | leavek
    # 'all' keeps an existing power lock and runs steps 2-5 against it; 'power' on an existing lock
    # needs --relock-reason "<why>" (the lock history is append-only)
"""
from __future__ import annotations

import os

os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=2")

import argparse  # noqa: E402
import datetime as _dt  # noqa: E402
import hashlib  # noqa: E402
import itertools  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402
from scipy import stats  # noqa: E402
from scipy.special import gammaln  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from r2 import common as C  # noqa: E402
from r2 import evidence as E  # noqa: E402

try:  # shared analysis helpers (optional; absent when this stage was run, so the module is self-contained)
    from r2 import analysis_common as AC  # noqa: F401,E402
except ImportError:  # pragma: no cover
    AC = None

# ----------------------------------------------------------------------------
# Constants (analysis choices come from common.PRESPECIFIED)
# ----------------------------------------------------------------------------
P = C.PRESPECIFIED
ALPHA = float(P["replication_alpha"])
TARGET_POWER = float(P["replication_power"])
RR_TRUE = float(P["replication_true_enrichment"])
ENDPOINT = float(P["replication_endpoint_enrichment"])
Q_STRONG = float(P["bfdr_strong"])
Q_MODERATE = float(P["bfdr_moderate"])
SEED = int(P["random_seed"])
N_BOOT = int(P["n_bootstrap"])
N_PERM = int(P["n_null_draws"])
VARIANT = P["evidence_variant_primary"]
UNIT_COL = P["cohort_cluster_unit"]            # 'study_id'

UNIVERSES = ("global",) + tuple(C.TISSUES)
PRIMARY_UNIVERSE = "global"
FIT_ID = "family_hb_nuts"
COMPARATORS = ("universe", "observed")
LEVELS = ("cohort", "study")
N_POWER_SIMS = 2000          # Monte-Carlo designs for the power at RR_TRUE (MC se <= 0.011)
N_POWER_SIMS_MDE = 1000      # per bisection step of the minimum detectable effect
MDE_BISECT_STEPS = 14
TRIM = 1e-22                 # stratum null pmf entries below this are dropped in simulations only
TIE_DECIMALS = 12            # credibility tie blocks (same as governance / robustness / heldout_eval)

OUT_DIR = C.OUT / "replication"
LOCK = OUT_DIR / "power_lock.json"
LOCK_HISTORY = OUT_DIR / "power_lock_history.jsonl"
DESIGN_FILE = OUT_DIR / "strata_design.csv"
COHORT_TABLE = C.TABLES / "cohort_table.csv"
FITS_DIR = C.OUT / "fits"

# Bayesian random-effects model
RE_M_PRIOR = (-4.0, 3.0)     # m_u ~ Normal(mean, sd)
RE_S_PRIOR = 2.0             # s_u ~ HalfNormal(sd)
RE_CENTRED_MIN_POS = 10      # strata with >= 10 positives use a centred a_us (see re_model_fn)
# mixed centred / non-centred stratum intercepts (re_model_fn) and target acceptance 0.99: trial fits
# with a fully centred a_us had 15 (study) and 55 (cohort level) divergences out of 4000 draws
NUTS_SETTINGS = dict(num_chains=4, num_warmup=1000, num_samples=1000, target_accept=0.99)
PRIMARY_PROFILE = "primary"
PRIOR_PROFILES = {           # name -> (mu mean, mu sd, sigma HalfNormal sd)
    "primary": (0.0, 1.0, 0.5),
    "mu_sd0.5": (0.0, 0.5, 0.5),
    "mu_sd2": (0.0, 2.0, 0.5),
    "sigma_hn0.25": (0.0, 1.0, 0.25),
    "sigma_hn1": (0.0, 1.0, 1.0),
    "skeptical": (0.0, 0.2, 0.5),
    "skeptical_sigma0.25": (0.0, 0.2, 0.25),
    "diffuse": (0.0, 2.0, 1.0),
}
LEAVE_K = (1, 2, 3)
MIN_BOOT_CLUSTERS = 5        # TF-cluster bootstrap CIs need >= 5 TFs (post hoc rule, verifier minor 4)
RESP_DECILES = 10
TF_SLOPE_PROFILE = "tf_random_slope"
TF_SLOPE_SD = 0.5            # tau ~ HalfNormal(0.5) for the TF-level random slope (post hoc)
POST_HOC_UNIT_COLUMNS = ("resp_mean_promoted;resp_mean_nonpromoted;resp_strat_*;wrongtf_*;wrongtf_excl_shared_*;"
                         "specific_excess*;max_attainable_enrichment;endpoint_attainable;endpoint_status;"
                         "endpoint_power_rr2;floor_edges_min_scan;adequately_powered_min_scan;power_ge_target;"
                         "bootstrap cluster rule (TF within unit, >= 5 TFs)")
ANALYSIS_DECISIONS = [
    dict(id="rep_wrongtf", date="2026-10-08", status="post_hoc",
         item="wrong-TF control and specific excess", note="verifier rep M1; after the 16:40:31Z lock"),
    dict(id="rep_responsiveness", date="2026-10-08", status="post_hoc",
         item="TF x target-responsiveness-decile stratified test", note="verifier rep M1"),
    dict(id="rep_attainable", date="2026-10-08", status="post_hoc",
         item="attainable maximum O/E and endpoint_unattainable classification", note="verifier rep M2"),
    dict(id="rep_endpoint_power", date="2026-10-08", status="post_hoc",
         item="endpoint power (O/E >= 1.5 and p <= 0.05) at RR = 2", note="verifier rep minor 7"),
    dict(id="rep_floor_scan", date="2026-10-08", status="post_hoc",
         item="linear-scan minimum of the support floor", note="verifier rep minor 2; locked floor unchanged"),
    dict(id="rep_boot_clusters", date="2026-10-08", status="post_hoc",
         item="bootstrap resamples TFs within the unit; no CI below 5 TFs", note="verifier rep minor 4"),
    dict(id="rep_tf_slope", date="2026-10-08", status="post_hoc",
         item="TF-level random-slope Bayesian sensitivity profile", note="verifier rep minor 5"),
    dict(id="rep_pass_subsets", date="2026-10-08", status="post_hoc",
         item="pass rates over power_ge_target and endpoint_attainable units", note="verifier rep minors 3 and M2"),
    dict(id="rep_lock_revalidated", date="2026-10-13", status="module_design",
         item="outcomes, threshold, Bayesian and leave-k stages re-run on the final 4 x 1000/1000 family_hb_nuts fits; "
              "the recomputed stratum design equals the locked design (same digest), so the 2026-10-08T16:40:31Z "
              "power lock and rep_power.csv are unchanged (outputs/r2/replication/lock_check.json)",
         note="tier sizes moved by a few boundary edges that are not evaluable in any cohort; no re-lock"),
]


# ----------------------------------------------------------------------------
# Promoted set
# ----------------------------------------------------------------------------
def bfdr_set(cred: np.ndarray, n: np.ndarray, q: float) -> np.ndarray:
    """Largest prefix of whole tie blocks (decreasing credibility) whose edge-weighted mean of
    (1 - cred) is <= q (boolean per pattern).  Tie blocks = equal credibility after rounding to
    ``TIE_DECIMALS`` decimals, as in ``governance.bfdr_levels`` / ``robustness.bfdr_set``; the
    tiers are identical to ``governance.bfdr_levels`` for all four universes (checked)."""
    cred = np.asarray(cred, dtype=float)
    n = np.asarray(n, dtype=float)
    uniq, inv = np.unique(-np.round(cred, TIE_DECIMALS), return_inverse=True)
    inv = inv.reshape(-1)
    bn = np.bincount(inv, weights=n, minlength=len(uniq))
    bfp = np.bincount(inv, weights=n * (1 - cred), minlength=len(uniq))
    cum_n, cum_fp = np.cumsum(bn), np.cumsum(bfp)
    with np.errstate(invalid="ignore", divide="ignore"):
        fdr = np.where(cum_n > 0, cum_fp / cum_n, 0.0)
    ok = np.nonzero(fdr <= q + 1e-15)[0]
    if len(ok) == 0:
        return np.zeros(len(cred), dtype=bool)
    return inv <= ok.max()


def pattern_tiers(cred: np.ndarray, n: np.ndarray) -> np.ndarray:
    """Per-pattern tier code: 2 = Strong, 1 = Moderate, 0 = other."""
    strong = bfdr_set(cred, n, Q_STRONG)
    moderate = bfdr_set(cred, n, Q_MODERATE)
    return np.where(strong, 2, np.where(moderate, 1, 0)).astype(np.int8)


def load_universe_tiers(universe: str) -> dict:
    R, n, fam, ann = E.load_patterns(universe, VARIANT)
    d = FITS_DIR / universe / VARIANT / FIT_ID
    cred = np.load(d / "cred_mean.npy")
    assert cred.shape[0] == R.shape[0], (universe, cred.shape, R.shape)
    tiers = pattern_tiers(cred, n)
    observed = (R >= 0).any(1)
    meta = json.loads((d / "meta.json").read_text())
    return dict(tier=tiers, observed=observed, cred=cred, n=n, fit_dir=str(d.relative_to(C.SP)),
                n_strong=float(n[tiers == 2].sum()), n_moderate=float(n[tiers == 1].sum()),
                fit_meta=dict(n_divergent=meta.get("n_divergent"),
                              r_hat_max=meta.get("convergence", {}).get("r_hat_max")))


# ----------------------------------------------------------------------------
# Evaluable edges
# ----------------------------------------------------------------------------
def cohort_table() -> pd.DataFrame:
    ct = pd.read_csv(COHORT_TABLE)
    return ct[["cohort_id", UNIT_COL, "cell_type_group", "modality"]].copy()


def universe_tfs(universe: str) -> list:
    if universe == "global":
        return [x.strip() for x in (E.EV_DIR / "global" / "tfs.txt").read_text().splitlines() if x.strip()]
    return E.tissue_universe(universe)[0]


def read_cohort_de(cohort: str, tfs) -> pd.DataFrame:
    """DE rows of ``cohort`` whose perturbed gene is in ``tfs`` (tf, target, positive, cis)."""
    path = C.OUT / "cohorts" / cohort / "de.parquet"
    tfs = sorted(set(tfs))
    tb = pq.read_table(path, columns=["perturbed_gene", "target", "positive_primary", "cis_same_band"],
                       filters=[("perturbed_gene", "in", tfs)])
    df = tb.to_pandas()
    out = pd.DataFrame({"tf": df["perturbed_gene"].astype(str).to_numpy(),
                        "target": df["target"].astype(str).to_numpy(),
                        "positive": df["positive_primary"].to_numpy(bool),
                        "cis": df["cis_same_band"].to_numpy(bool)})
    if out.duplicated(["tf", "target"]).any():
        raise ValueError(f"{cohort}: duplicated (perturbed_gene, target) rows")
    return out


def attach_patterns(universe: str, de: pd.DataFrame) -> pd.DataFrame:
    """Join DE rows to universe edges on (tf, target) keys; adds the pattern index.
    Rows whose pair is not a universe edge (or is a self pair) are dropped."""
    if de.empty:
        return de.assign(pattern=np.zeros(0, dtype=np.int64))
    ep = E.load_edge_pattern(universe, VARIANT)
    if universe == "global":
        tfs = universe_tfs("global")
        genes = [x.strip() for x in (E.EV_DIR / "global" / "genes.txt").read_text().splitlines() if x.strip()]
        G = len(genes)
        assert len(ep) == len(tfs) * G
        ti = pd.Index(tfs).get_indexer(de["tf"])
        gi = pd.Index(genes).get_indexer(de["target"])
        ok = (ti >= 0) & (gi >= 0)
        out = de.loc[ok].copy()
        out["pattern"] = np.asarray(ep[ti[ok].astype(np.int64) * G + gi[ok]], dtype=np.int64)
    else:
        ev = pd.read_parquet(E.EV_DIR / universe / "evidence.parquet", columns=["tf", "target"])
        assert len(ev) == len(ep)
        ev = ev.assign(pattern=np.asarray(ep, dtype=np.int64))
        out = de.merge(ev, on=["tf", "target"], how="inner", validate="one_to_one")
    out = out[out["pattern"] >= 0]            # self pairs carry pattern -1 (global grid)
    return out.reset_index(drop=True)


def evaluable_edges(universe: str, tiers: dict, cohorts: pd.DataFrame) -> pd.DataFrame:
    """All evaluable (cohort, tf, target) edges of a universe with tier/observed flags and the
    positive label.  cis pairs removed from positives and negatives."""
    tfs = universe_tfs(universe)
    frames = []
    for _, row in cohorts.iterrows():
        de = read_cohort_de(row["cohort_id"], tfs)
        de = de[~de["cis"]].drop(columns="cis")
        ed = attach_patterns(universe, de)
        ed.insert(0, "cohort", row["cohort_id"])
        frames.append(ed)
    ed = pd.concat(frames, ignore_index=True)
    ed["tier"] = tiers["tier"][ed["pattern"].to_numpy()]
    ed["promoted"] = ed["tier"] > 0
    ed["observed"] = tiers["observed"][ed["pattern"].to_numpy()]
    return ed


# ----------------------------------------------------------------------------
# Design (STEP 1 input): stratum margins only
# ----------------------------------------------------------------------------
def stratum_design(ed: pd.DataFrame) -> pd.DataFrame:
    """Per (cohort, tf): N (evaluable), n (promoted evaluable), n_strong, K (positives).

    K is the positive count over *all* evaluable edges of the stratum; the split of positives
    between promoted and non-promoted edges is not computed here."""
    g = ed.groupby(["cohort", "tf"], sort=True)
    d = pd.DataFrame({"N": g.size(), "n": g["promoted"].sum(),
                      "n_strong": g["tier"].apply(lambda t: int((t == 2).sum())),
                      "K": g["positive"].sum()}).reset_index()
    return d.astype({"N": np.int64, "n": np.int64, "n_strong": np.int64, "K": np.int64})


def all_designs(universes=UNIVERSES) -> tuple[pd.DataFrame, dict]:
    ct = cohort_table()
    rows, info = [], {}
    for u in universes:
        tiers = load_universe_tiers(u)
        ed = evaluable_edges(u, tiers, ct)
        info[u] = dict(n_strong=tiers["n_strong"], n_moderate=tiers["n_moderate"], fit_dir=tiers["fit_dir"],
                       fit_meta=tiers["fit_meta"])
        for comp in COMPARATORS:
            sub = ed if comp == "universe" else ed[ed["observed"]]
            d = stratum_design(sub)
            d.insert(0, "comparator", comp)
            d.insert(0, "universe", u)
            rows.append(d)
    design = pd.concat(rows, ignore_index=True)
    design = design.merge(ct[["cohort_id", UNIT_COL]].rename(columns={"cohort_id": "cohort"}), on="cohort", how="left")
    return design, info


def units_of(design: pd.DataFrame, level: str) -> dict:
    """unit id -> list of cohorts (all cohorts of the cohort table, evaluable or not)."""
    ct = cohort_table()
    if level == "cohort":
        return {c: [c] for c in ct["cohort_id"]}
    out = {}
    for _, r in ct.iterrows():
        out.setdefault(r[UNIT_COL], []).append(r["cohort_id"])
    return out


def informative(N, n, K) -> np.ndarray:
    N, n, K = (np.asarray(x) for x in (N, n, K))
    return (n > 0) & (n < N) & (K > 0) & (K < N)


# ----------------------------------------------------------------------------
# Exact conditional (stratified) test
# ----------------------------------------------------------------------------
class LogFact:
    """log(k!) table, grown on demand."""

    def __init__(self, n: int = 1024):
        self.t = gammaln(np.arange(n + 1) + 1.0)

    def __call__(self, k):
        k = np.asarray(k)
        m = int(np.max(k)) if k.size else 0
        if m >= len(self.t):
            self.t = gammaln(np.arange(2 * m + 1) + 1.0)
        return self.t[k]

    def lchoose(self, a, b):
        return self(a) - self(b) - self(np.asarray(a) - np.asarray(b))


_LF = LogFact()


def hyper_pmf(N: int, K: int, n: int, log_or: float = 0.0, trim: float = 0.0):
    """pmf of X ~ (Fisher noncentral) hypergeometric: X = successes among n draws from N with K
    successes; odds ratio exp(log_or) (0 = central).  Returns (lo, pmf over lo..hi)."""
    lo, hi = max(0, n - (N - K)), min(n, K)
    if hi <= lo:
        return lo, np.ones(1)
    x = np.arange(lo, hi + 1)
    lp = _LF.lchoose(K, x) + _LF.lchoose(N - K, n - x) + log_or * x
    lp -= lp.max()
    p = np.exp(lp)
    p /= p.sum()
    if trim > 0:
        keep = np.nonzero(p > trim)[0]
        a, b = keep[0], keep[-1]
        lo, p = lo + a, p[a:b + 1]
    return lo, p


def sum_distribution(N, K, n, log_or: float = 0.0, trim: float = 0.0):
    """Distribution of T = sum_s X_s (independent strata).  Returns (shift, pmf)."""
    pmf, shift = np.ones(1), 0
    for Ns, Ks, ns in zip(N, K, n):
        if ns == 0 or Ks == 0:
            continue
        lo, p = hyper_pmf(int(Ns), int(Ks), int(ns), log_or, trim)
        shift += lo
        if len(p) > 1:
            pmf = np.convolve(pmf, p)
    return shift, pmf


def exact_pvalue(N, K, n, t: int, trim: float = 0.0) -> float:
    """One-sided exact conditional p = P0(T >= t)."""
    shift, pmf = sum_distribution(N, K, n, 0.0, trim)
    i = int(t) - shift
    if i <= 0:
        return 1.0
    if i >= len(pmf):
        return 0.0
    return float(min(1.0, pmf[i:].sum()))


def mantel_haenszel_or(a, n, K, N) -> float:
    """MH common odds ratio; a = promoted positives, n = promoted, K = positives, N = total."""
    a, n, K, N = (np.asarray(x, dtype=float) for x in (a, n, K, N))
    b = n - a
    c = K - a
    d = N - n - c
    num = np.sum(a * d / N)
    den = np.sum(b * c / N)
    if den == 0:
        return float("inf") if num > 0 else float("nan")
    return float(num / den)


# ----------------------------------------------------------------------------
# STEP 1: design-based power
# ----------------------------------------------------------------------------
def rates_for_rr(pbar: float, N_tot: float, n_tot: float, rr: float) -> tuple[float, float]:
    """Non-promoted / promoted positive probabilities with overall rate pbar and RR = rr."""
    p0 = pbar * N_tot / (N_tot - n_tot + rr * n_tot)
    return p0, min(1.0, rr * p0)


def simulate_power(N, n, pbar: float, rr: float, alpha: float = ALPHA, n_sims: int = N_POWER_SIMS,
                   seed: int = SEED) -> tuple[float, float]:
    """Monte-Carlo power of the exact stratified test for strata (N_s, n_s) with homogeneous
    rates (p0, p1) implied by (pbar, rr).  Common random numbers: the same seed gives the same
    uniforms for every rr."""
    N = np.asarray(N, dtype=np.int64)
    n = np.asarray(n, dtype=np.int64)
    keep = n > 0
    N, n = N[keep], n[keep]
    if len(N) == 0 or pbar <= 0:
        return 0.0, 0.0
    p0, p1 = rates_for_rr(pbar, float(N.sum()), float(n.sum()), rr)
    rng = np.random.default_rng(seed)
    U = rng.random((n_sims, len(N)))
    V = rng.random((n_sims, len(N)))
    X = np.maximum(stats.binom.ppf(U, n, p1), 0).astype(np.int64)
    Y = np.maximum(stats.binom.ppf(V, N - n, p0), 0).astype(np.int64)
    rej = 0
    for b in range(n_sims):
        K = X[b] + Y[b]
        if exact_pvalue(N, K, n, int(X[b].sum()), trim=TRIM) <= alpha:
            rej += 1
    pw = rej / n_sims
    return pw, math.sqrt(max(pw * (1 - pw), 1e-12) / n_sims)


def mde_rr(N, n, pbar: float, alpha: float = ALPHA, target: float = TARGET_POWER,
           n_sims: int = N_POWER_SIMS_MDE, steps: int = MDE_BISECT_STEPS, seed: int = SEED) -> tuple[float, float]:
    """Smallest RR with simulated power >= target (bisection in log RR).  Returns (mde, rr_max);
    mde = inf if the power at rr_max (every promoted edge positive) is below target."""
    N = np.asarray(N, dtype=np.int64)
    n = np.asarray(n, dtype=np.int64)
    if n.sum() == 0 or pbar <= 0:
        return float("inf"), float("nan")
    N_tot, n_tot = float(N[n > 0].sum()), float(n.sum())
    N_all = float(N.sum())
    # rr at which p1 = 1:  rr * pbar * N / (N - n + rr n) = 1  ->  rr = (N - n) / (pbar N - n)
    denom = pbar * N_tot - n_tot
    rr_max = (N_tot - n_tot) / denom if denom > 0 else 1e6
    rr_max = float(min(rr_max, 1e6))
    del N_all
    if simulate_power(N, n, pbar, rr_max, alpha, n_sims, seed)[0] < target:
        return float("inf"), rr_max
    lo, hi = 0.0, math.log(rr_max)
    for _ in range(steps):
        mid = 0.5 * (lo + hi)
        if simulate_power(N, n, pbar, math.exp(mid), alpha, n_sims, seed)[0] >= target:
            hi = mid
        else:
            lo = mid
    return float(math.exp(hi)), rr_max


def critical_values(N: int, m: int, K: np.ndarray, alpha: float = ALPHA) -> np.ndarray:
    """c(K) = smallest x with P(X >= x | K) <= alpha for X ~ Hypergeometric(N, K, m)
    (m + 1 if no such x)."""
    K = np.asarray(K, dtype=np.int64)
    c = stats.hypergeom.isf(alpha, N, K, m)       # smallest x with P(X > x) <= alpha
    c = np.where(np.isnan(c), m, c).astype(np.int64) + 1
    return c


def exact_power_single(N: int, m: int, pbar: float, rr: float, alpha: float = ALPHA, tail: float = 1e-13) -> float:
    """Exact power of the one-sided Fisher test (promoted vs not) in one stratum with N edges,
    m promoted, overall rate pbar and RR = rr (binomial outcomes enumerated)."""
    if m <= 0 or m >= N or pbar <= 0:
        return 0.0
    p0, p1 = rates_for_rr(pbar, float(N), float(m), rr)
    xs = np.arange(m + 1)
    px = stats.binom.pmf(xs, m, p1)
    sel = px > tail
    xs, px = xs[sel], px[sel]
    ylo = int(stats.binom.ppf(tail, N - m, p0))
    yhi = int(stats.binom.isf(tail, N - m, p0)) + 1
    ys = np.arange(max(0, ylo), min(N - m, yhi) + 1)
    py = stats.binom.pmf(ys, N - m, p0)
    Ks = np.arange(xs.min() + ys.min(), xs.max() + ys.max() + 1)
    cK = critical_values(N, m, Ks, alpha)
    Kgrid = xs[:, None] + ys[None, :]
    reject = xs[:, None] >= cK[Kgrid - Ks[0]]
    return float(np.sum(px[:, None] * py[None, :] * reject))


def support_floor(N: int, pbar: float, rr: float = RR_TRUE, alpha: float = ALPHA,
                  target: float = TARGET_POWER) -> float:
    """Support floor of the locked STEP 1: the first m found with exact single-stratum power
    >= target by exponential search + bisection.  Exact tests have a saw-tooth power curve in m,
    so this is the first crossing within the final bracket, not necessarily the smallest m with
    power >= target (see ``support_floor_min_scan``).  inf if not reached for m <= N/2."""
    if pbar <= 0 or N < 2:
        return float("inf")
    m_max = max(1, N // 2)
    hi = 1
    while hi < m_max and exact_power_single(N, hi, pbar, rr, alpha) < target:
        hi *= 2
    hi = min(hi, m_max)
    if exact_power_single(N, hi, pbar, rr, alpha) < target:
        return float("inf")
    lo = hi // 2 if hi > 1 else 0
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if exact_power_single(N, mid, pbar, rr, alpha) >= target:
            hi = mid
        else:
            lo = mid
    return float(hi)


def support_floor_min_scan(N: int, pbar: float, floor: float, rr: float = RR_TRUE, alpha: float = ALPHA,
                           target: float = TARGET_POWER) -> float:
    """Smallest m <= floor with exact single-stratum power >= target (linear scan; post hoc)."""
    if not np.isfinite(floor) or pbar <= 0:
        return float(floor)
    for m in range(1, int(floor) + 1):
        if exact_power_single(N, m, pbar, rr, alpha) >= target:
            return float(m)
    return float(floor)


def simulate_endpoint_power(N, n, pbar: float, rr: float, alpha: float = ALPHA, endpoint: float = ENDPOINT,
                            n_sims: int = N_POWER_SIMS, seed: int = SEED) -> tuple[float, float]:
    """Monte-Carlo power of the replication endpoint (O/E >= endpoint AND exact p <= alpha), with the
    simulated tables of ``simulate_power`` (same seed -> same tables; endpoint power <= test power)."""
    N = np.asarray(N, dtype=np.int64)
    n = np.asarray(n, dtype=np.int64)
    keep = n > 0
    N, n = N[keep], n[keep]
    if len(N) == 0 or pbar <= 0:
        return 0.0, 0.0
    p0, p1 = rates_for_rr(pbar, float(N.sum()), float(n.sum()), rr)
    rng = np.random.default_rng(seed)
    U = rng.random((n_sims, len(N)))
    V = rng.random((n_sims, len(N)))
    X = np.maximum(stats.binom.ppf(U, n, p1), 0).astype(np.int64)
    Y = np.maximum(stats.binom.ppf(V, N - n, p0), 0).astype(np.int64)
    hit = 0
    for b in range(n_sims):
        K = X[b] + Y[b]
        Ex = float(np.sum(n * K / N))
        if Ex <= 0 or X[b].sum() / Ex < endpoint:
            continue
        if exact_pvalue(N, K, n, int(X[b].sum()), trim=TRIM) <= alpha:
            hit += 1
    pw = hit / n_sims
    return pw, math.sqrt(max(pw * (1 - pw), 1e-12) / n_sims)


def conditional_power(N, n, K, log_or: float, alpha: float = ALPHA) -> tuple[float, int]:
    """Exact power given the per-stratum margins (N_s, n_s, K_s) under a common odds ratio
    exp(log_or): P1(T >= t_crit), t_crit = smallest t with P0(T >= t) <= alpha."""
    N, n, K = (np.asarray(x, dtype=np.int64) for x in (N, n, K))
    keep = informative(N, n, K)
    if not keep.any():
        return 0.0, -1
    N, n, K = N[keep], n[keep], K[keep]
    s0, p0 = sum_distribution(N, K, n, 0.0)
    tail0 = np.cumsum(p0[::-1])[::-1]                   # P0(T >= shift + i)
    ok = np.nonzero(tail0 <= alpha)[0]
    if len(ok) == 0:
        return 0.0, -1
    t_crit = s0 + int(ok[0])
    s1, p1 = sum_distribution(N, K, n, log_or)
    i = t_crit - s1
    pw = float(p1[max(i, 0):].sum()) if i < len(p1) else 0.0
    return pw, t_crit


def rr_to_log_or(rr: float, p: float) -> float:
    """log odds ratio equivalent to relative risk rr at baseline probability p (capped)."""
    p1 = min(rr * p, 1 - 1e-9)
    return math.log(p1 / (1 - p1)) - math.log(p / (1 - p))


def conditional_mde_or(N, n, K, alpha: float = ALPHA, target: float = TARGET_POWER, steps: int = 40) -> float:
    lo, hi = 0.0, math.log(1e4)
    if conditional_power(N, n, K, hi, alpha)[0] < target:
        return float("inf")
    for _ in range(steps):
        mid = 0.5 * (lo + hi)
        if conditional_power(N, n, K, mid, alpha)[0] >= target:
            hi = mid
        else:
            lo = mid
    return float(math.exp(hi))


def unit_strata(design: pd.DataFrame, universe: str, comparator: str, cohorts: list) -> pd.DataFrame:
    return design[(design["universe"] == universe) & (design["comparator"] == comparator)
                  & design["cohort"].isin(cohorts)]


def unit_status(d: pd.DataFrame) -> str:
    if d.empty or d["N"].sum() == 0:
        return "non-evaluable: no perturbed TF of the universe tested"
    if d["K"].sum() == 0:
        return "non-evaluable: no evaluable TF positives"
    if d["n"].sum() == 0:
        return "non-evaluable: no promoted evaluable edges"
    if not informative(d["N"], d["n"], d["K"]).any():
        return "non-evaluable: no TF with both promoted edges and positives"
    return "evaluable"


def unit_seed(*parts) -> int:
    """Deterministic per-unit seed derived from PRESPECIFIED['random_seed']."""
    h = hashlib.blake2b("\x1f".join(map(str, (SEED,) + parts)).encode(), digest_size=4).digest()
    return int.from_bytes(h, "little")


def power_table(design: pd.DataFrame) -> pd.DataFrame:
    ct = cohort_table()
    cell = dict(zip(ct["cohort_id"], ct["cell_type_group"]))
    rows = []
    for u in sorted(design["universe"].unique(), key=UNIVERSES.index):
        for level in LEVELS:
            for unit, cohorts in units_of(design, level).items():
                for comp in COMPARATORS:
                    d = unit_strata(design, u, comp, cohorts)
                    status = unit_status(d)
                    N_tot, n_tot, K_tot = int(d["N"].sum()), int(d["n"].sum()), int(d["K"].sum())
                    pbar = K_tot / N_tot if N_tot else float("nan")
                    tf_rate = (d["K"] / d["N"]).to_numpy() if N_tot else np.array([])
                    r = dict(universe=u, level=level, unit=unit, comparator=comp, cohorts=";".join(cohorts),
                             cell_types=";".join(sorted({cell[c] for c in cohorts})),
                             status=status, n_tfs_tested=int(d["tf"].nunique()) if len(d) else 0,
                             n_strata=int(len(d)), n_strata_with_promoted=int((d["n"] > 0).sum()),
                             n_strata_informative=int(informative(d["N"], d["n"], d["K"]).sum()) if len(d) else 0,
                             n_evaluable_edges=N_tot, n_promoted_evaluable=n_tot,
                             n_strong_evaluable=int(d["n_strong"].sum()), n_positives=K_tot,
                             positive_rate=pbar,
                             n_positives_in_strata_with_promoted=int(d.loc[d["n"] > 0, "K"].sum()),
                             median_tf_positive_rate=float(np.median(tf_rate)) if len(tf_rate) else np.nan,
                             max_tf_positive_rate=float(np.max(tf_rate)) if len(tf_rate) else np.nan,
                             share_positives_top_tf=float(d["K"].max() / K_tot) if K_tot else np.nan,
                             expected_positives_promoted_h0=float(n_tot * pbar) if N_tot else np.nan)
                    if status == "evaluable":
                        seed = unit_seed(u, level, unit, comp)
                        t0 = time.time()
                        pw, se = simulate_power(d["N"], d["n"], pbar, RR_TRUE, seed=seed)
                        mde, rr_max = mde_rr(d["N"], d["n"], pbar, seed=seed)
                        floor = support_floor(N_tot, pbar)
                        lor = rr_to_log_or(RR_TRUE, pbar)
                        pc, tcrit = conditional_power(d["N"], d["n"], d["K"], lor)
                        mde_or = conditional_mde_or(d["N"], d["n"], d["K"])
                        r.update(power=pw, power_mc_se=se, mde_rr=mde, rr_max=rr_max,
                                 floor_edges=floor, floor_expected_positives=floor * pbar,
                                 adequately_powered=bool(n_tot >= floor),
                                 power_ge_target=bool(pw >= TARGET_POWER),
                                 power_cond=pc, power_cond_odds_ratio=math.exp(lor), t_crit_cond=tcrit,
                                 mde_cond_or=mde_or, seconds=time.time() - t0)
                    else:
                        r.update(power=0.0 if status.endswith("promoted evaluable edges") else np.nan,
                                 power_mc_se=np.nan, mde_rr=np.inf, rr_max=np.nan,
                                 floor_edges=support_floor(N_tot, pbar) if (N_tot and pbar > 0) else np.inf,
                                 adequately_powered=False, power_ge_target=False,
                                 power_cond=np.nan, power_cond_odds_ratio=np.nan, t_crit_cond=np.nan,
                                 mde_cond_or=np.nan, seconds=0.0)
                        r["floor_expected_positives"] = r["floor_edges"] * pbar if N_tot else np.nan
                    rows.append(r)
                    print(f"[power] {u:7s} {level:6s} {unit:14s} {comp:8s} {status:45s} "
                          f"n={n_tot:6d} N={N_tot:8d} pbar={pbar:.4g} power={r['power']} mde={r['mde_rr']} "
                          f"floor={r['floor_edges']}", flush=True)
    df = pd.DataFrame(rows)
    df.insert(4, "alpha", ALPHA)
    df.insert(5, "true_enrichment", RR_TRUE)
    df.insert(6, "target_power", TARGET_POWER)
    return df


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def design_digest(design: pd.DataFrame) -> str:
    d = design.sort_values(["universe", "comparator", "cohort", "tf"]).reset_index(drop=True)
    return hashlib.sha256(d.to_csv(index=False).encode()).hexdigest()


def run_power(universes=UNIVERSES, relock_reason: str = "") -> None:
    """STEP 1.  ``relock_reason``: required when a lock already exists (e.g. the upstream promoted set
    changed because a fit was re-run); it is written into the new lock, and the append-only history
    keeps every earlier lock."""
    if LOCK.exists() and not relock_reason:
        raise SystemExit("power_lock.json exists; re-locking needs --relock-reason (the history keeps the old lock)")
    C.ensure_dir(OUT_DIR)
    t0 = time.time()
    design, info = all_designs(universes)
    design.to_csv(DESIGN_FILE, index=False)
    pw = power_table(design)
    out = C.TABLES / "rep_power.csv"
    pw.to_csv(out, index=False)
    lock = dict(stage="STEP 1 design-based power", written_at_utc=_dt.datetime.now(_dt.timezone.utc).isoformat(),
                note="computed from stratum margins (N_s, n_s, K_s) and the unit positive rate only; no "
                     "promoted x positive cross-tabulation had been computed by this module when this lock "
                     "was written",
                rep_power_csv=str(out.relative_to(C.SP)), rep_power_sha256=_sha256(out),
                strata_design_csv=str(DESIGN_FILE.relative_to(C.SP)), strata_design_sha256=_sha256(DESIGN_FILE),
                design_digest=design_digest(design), universes=list(universes), promoted_sets=info,
                prespecified={k: P[k] for k in ("replication_true_enrichment", "replication_power",
                                                "replication_alpha", "replication_endpoint_enrichment",
                                                "bfdr_strong", "bfdr_moderate", "random_seed")},
                n_power_sims=N_POWER_SIMS, n_power_sims_mde=N_POWER_SIMS_MDE, runtime_s=time.time() - t0)
    if relock_reason:
        prev = json.loads(LOCK.read_text()) if LOCK.exists() else {}
        lock["relock_reason"] = relock_reason
        lock["previous_lock_written_at_utc"] = prev.get("written_at_utc")
        lock["previous_design_digest"] = prev.get("design_digest")
        lock["note"] += ("; RE-LOCK: outcomes of the previous promoted set had been computed before this lock "
                         "(see relock_reason); this stage still reads margins only")
    C.write_json(lock, LOCK)
    with open(LOCK_HISTORY, "a") as fh:
        fh.write(json.dumps(lock, default=str) + "\n")
    print(f"[power] wrote {out} and {LOCK} ({time.time() - t0:.0f} s)")


def check_lock(design: pd.DataFrame) -> dict:
    if not LOCK.exists():
        raise SystemExit("STEP 1 (power) has not been run: power_lock.json missing")
    lock = json.loads(LOCK.read_text())
    if design_digest(design) != lock["design_digest"]:
        raise SystemExit("design differs from the locked STEP 1 design; re-run the power stage first")
    out = C.TABLES / "rep_power.csv"
    if _sha256(out) != lock["rep_power_sha256"]:
        raise SystemExit("rep_power.csv changed after the lock was written")
    return lock


# ----------------------------------------------------------------------------
# STEP 2: outcomes
# ----------------------------------------------------------------------------
def outcome_strata(ed: pd.DataFrame) -> pd.DataFrame:
    """Per (cohort, tf): N, n, K, a (= promoted positives), and the tier split."""
    ed = ed.assign(pp=ed["promoted"] & ed["positive"], sp=(ed["tier"] == 2) & ed["positive"],
                   strong=ed["tier"] == 2, moderate=ed["tier"] == 1,
                   mp=(ed["tier"] == 1) & ed["positive"])
    g = ed.groupby(["cohort", "tf"], sort=True)
    d = pd.DataFrame({"N": g.size(), "n": g["promoted"].sum(), "K": g["positive"].sum(), "a": g["pp"].sum(),
                      "n_strong": g["strong"].sum(), "a_strong": g["sp"].sum(),
                      "n_moderate": g["moderate"].sum(), "a_moderate": g["mp"].sum()}).reset_index()
    return d.astype({c: np.int64 for c in d.columns if c not in ("cohort", "tf")})


def oe(a, n, K, N) -> tuple[float, float]:
    a, n, K, N = (np.asarray(x, dtype=float) for x in (a, n, K, N))
    O = float(a.sum())
    Ex = float(np.sum(n * K / N))
    return O, Ex


def cluster_bootstrap(d: pd.DataFrame, n_boot: int, seed: int) -> dict:
    """TF-cluster bootstrap of O/E and the MH OR: the TFs with >= 1 promoted evaluable edge are
    resampled with replacement, all (cohort x TF) strata of a TF moving together (a study with two
    cohorts, e.g. Dixit d7 + d13, keeps a TF's strata in one cluster).  No interval is reported
    with fewer than ``MIN_BOOT_CLUSTERS`` TFs (percentile intervals over 2-4 clusters are not
    meaningful).  Strata without a ``tf`` column are their own clusters."""
    s = d[d["n"] > 0]
    nan = dict(oe_lo=np.nan, oe_hi=np.nan, or_lo=np.nan, or_hi=np.nan)
    if len(s) == 0:
        return dict(nan, n_clusters=0)
    labels = s["tf"].astype(str).to_numpy() if "tf" in s.columns else np.arange(len(s)).astype(str)
    codes, uniq = pd.factorize(labels, sort=True)
    G = len(uniq)
    a, n, K, N = (s[c].to_numpy(float) for c in ("a", "n", "K", "N"))
    agg = lambda v: np.bincount(codes, weights=v, minlength=G)  # noqa: E731
    O_c, E_c = agg(a), agg(n * K / N)
    num_c, den_c = agg(a * (N - n - (K - a)) / N), agg((n - a) * (K - a) / N)
    if G < MIN_BOOT_CLUSTERS:
        return dict(nan, n_clusters=int(G))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, G, size=(n_boot, G))
    O, Ex = O_c[idx].sum(1), E_c[idx].sum(1)
    num, den = num_c[idx].sum(1), den_c[idx].sum(1)
    with np.errstate(divide="ignore", invalid="ignore"):
        r_oe = np.where(Ex > 0, O / Ex, np.nan)
        r_or = np.where(den > 0, num / den, np.where(num > 0, np.inf, np.nan))
    q = lambda x, p: float(np.nanquantile(x, p)) if (~np.isnan(x)).any() else np.nan  # noqa: E731
    return dict(oe_lo=q(r_oe, 0.025), oe_hi=q(r_oe, 0.975), or_lo=q(r_or, 0.025), or_hi=q(r_or, 0.975),
                n_clusters=int(G))


def permutation_p(d: pd.DataFrame, t_obs: int, n_perm: int, seed: int) -> float:
    s = d[(d["n"] > 0) & (d["K"] > 0)]
    if len(s) == 0:
        return 1.0
    rng = np.random.default_rng(seed)
    X = rng.hypergeometric(s["K"].to_numpy(), (s["N"] - s["K"]).to_numpy(), s["n"].to_numpy(),
                           size=(n_perm, len(s)))
    T = X.sum(1)
    return float((1 + np.sum(T >= t_obs)) / (n_perm + 1))


def holm(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    out = np.full_like(p, np.nan)
    ok = ~np.isnan(p)
    pv = p[ok]
    m = len(pv)
    if m == 0:
        return out
    order = np.argsort(pv)
    adj = np.empty(m)
    run = 0.0
    for i, j in enumerate(order):
        run = max(run, min(1.0, (m - i) * pv[j]))
        adj[j] = run
    out[ok] = adj
    return out


def wilson(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if n == 0:
        return float("nan"), float("nan")
    ph = k / n
    den = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / den
    h = z * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / den
    lo, hi = c - h, c + h
    lo = 0.0 if k == 0 else lo          # exact bounds at the boundaries (no rounding residue)
    hi = 1.0 if k == n else hi
    return max(0.0, lo), min(1.0, hi)


def unit_outcome(d: pd.DataFrame, seed: int) -> dict:
    N, n, K, a = (d[c].to_numpy(np.int64) for c in ("N", "n", "K", "a"))
    O, Ex = oe(a, n, K, N)
    p_exact = exact_pvalue(N, K, n, int(a.sum()))
    n_tot, N_tot, K_tot = n.sum(), N.sum(), K.sum()
    prom_rate = a.sum() / n_tot if n_tot else np.nan
    non_rate = (K_tot - a.sum()) / (N_tot - n_tot) if N_tot > n_tot else np.nan
    unstrat = (a.sum() / n_tot) / (K_tot / N_tot) if (n_tot and K_tot) else np.nan
    Os, Es = oe(d["a_strong"], d["n_strong"], K, N)
    Om, Em = oe(d["a_moderate"], d["n_moderate"], K, N)
    boot = cluster_bootstrap(d, N_BOOT, seed)
    # influence sensitivity: drop the TF stratum contributing most to O - E
    contrib = d["a"].to_numpy(float) - d["n"].to_numpy(float) * K / N
    top = int(np.argmax(contrib)) if len(d) else -1
    keep = np.arange(len(d)) != top
    O2, E2 = oe(a[keep], n[keep], K[keep], N[keep])
    r = dict(observed_promoted_positives=O, expected_promoted_positives=Ex,
             top_tf=str(d["tf"].iloc[top]) if top >= 0 else "",
             top_tf_excess=float(contrib[top]) if top >= 0 else np.nan,
             enrichment_drop_top_tf=O2 / E2 if E2 > 0 else np.nan,
             p_exact_drop_top_tf=exact_pvalue(N[keep], K[keep], n[keep], int(a[keep].sum())),
             enrichment=O / Ex if Ex > 0 else np.nan,
             enrichment_unstratified=unstrat, mh_odds_ratio=mantel_haenszel_or(a, n, K, N),
             promoted_positive_rate=prom_rate, nonpromoted_positive_rate=non_rate,
             p_exact=p_exact, p_perm=permutation_p(d, int(a.sum()), N_PERM, seed + 1),
             enrichment_boot_lo=boot["oe_lo"], enrichment_boot_hi=boot["oe_hi"],
             mh_or_boot_lo=boot["or_lo"], mh_or_boot_hi=boot["or_hi"],
             n_bootstrap_clusters=boot.get("n_clusters", 0),
             observed_strong=Os, expected_strong=Es, enrichment_strong=Os / Es if Es > 0 else np.nan,
             observed_moderate=Om, expected_moderate=Em, enrichment_moderate=Om / Em if Em > 0 else np.nan)
    r["endpoint_pass"] = bool((r["enrichment"] >= ENDPOINT) and (p_exact <= ALPHA))
    # attainable maximum from the margins (post hoc, verifier M2)
    o_max = float(np.minimum(n, K).sum())
    r["max_attainable_enrichment"] = o_max / Ex if Ex > 0 else np.nan
    r["endpoint_attainable"] = bool(r["max_attainable_enrichment"] >= ENDPOINT) if Ex > 0 else False
    r["endpoint_status"] = ("pass" if r["endpoint_pass"] else
                            ("fail" if r["endpoint_attainable"] else "endpoint_unattainable"))
    return r


# ----------------------------------------------------------------------------
# Post hoc controls (verifier M1): target responsiveness and wrong-TF null
# ----------------------------------------------------------------------------
_DE_SUMMARY: dict = {}


def cohort_de_summary(cohort: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(responsiveness table, Lambert-TF DE rows) of a cohort, cached.

    Responsiveness table: per target, the number of perturbations (every perturbed gene of the
    cohort, cis pairs excluded) under which it was tested (``count``) and DE (``sum``).  TF rows:
    (tf, target, positive) for perturbed Lambert TFs, cis and self pairs excluded."""
    if cohort in _DE_SUMMARY:
        return _DE_SUMMARY[cohort]
    path = C.OUT / "cohorts" / cohort / "de.parquet"
    df = pq.read_table(path, columns=["perturbed_gene", "target", "positive_primary", "cis_same_band"]).to_pandas()
    df = df[~df["cis_same_band"].to_numpy(bool)]
    tot = df.groupby("target", observed=True)["positive_primary"].agg(["sum", "count"])
    tot.index = tot.index.astype(str)
    tfs = sorted(C.lambert_tfs())
    sub = df[df["perturbed_gene"].isin(tfs)]
    tfde = pd.DataFrame({"tf": sub["perturbed_gene"].astype(str).to_numpy(),
                         "target": sub["target"].astype(str).to_numpy(),
                         "positive": sub["positive_primary"].to_numpy(bool)})
    tfde = tfde[tfde["tf"] != tfde["target"]].reset_index(drop=True)
    _DE_SUMMARY[cohort] = (tot.astype(np.int64), tfde)
    return _DE_SUMMARY[cohort]


def responsiveness(ec: pd.DataFrame, tot: pd.DataFrame) -> np.ndarray:
    """Leave-own-TF-out responsiveness of each evaluable edge's target: (DE count of g over all
    perturbations - own label) / (perturbations testing g - 1)."""
    t = tot.reindex(ec["target"].astype(str))
    sm = t["sum"].to_numpy(float) - ec["positive"].to_numpy(float)
    cnt = t["count"].to_numpy(float) - 1
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(cnt > 0, sm / cnt, np.nan)


def wrong_tf_sums(ec: pd.DataFrame, tfde: pd.DataFrame) -> dict:
    """Wrong-TF control of one cohort's evaluable edges ``ec`` (tf, target, promoted, positive):
    for every TF s with promoted edges and every other TF s' of ``ec`` perturbed in the cohort,
    the stratum (s, s') holds s's evaluable targets tested under s'; O = promoted-for-s targets DE
    under s', E = sum n K / N.  ``*_excl``: targets that are also promoted for s' are dropped."""
    tfs = sorted(ec["tf"].unique())
    genes = sorted(ec["target"].unique())
    ti = {t: i for i, t in enumerate(tfs)}
    gi = {g: i for i, g in enumerate(genes)}
    T, G = len(tfs), len(genes)
    Y = np.full((T, G), np.nan)
    d = tfde[tfde["tf"].isin(ti) & tfde["target"].isin(gi)]
    Y[d["tf"].map(ti).to_numpy(), d["target"].map(gi).to_numpy()] = d["positive"].to_numpy(float)
    Ev = np.zeros((T, G), bool)
    Pm = np.zeros((T, G), bool)
    r, c = ec["tf"].map(ti).to_numpy(), ec["target"].map(gi).to_numpy()
    Ev[r, c] = True
    Pm[r, c] = ec["promoted"].to_numpy(bool)
    out = dict(O=0.0, E=0.0, O_excl=0.0, E_excl=0.0, n_pairs=0)
    for s_ in range(T):
        sel = Ev[s_]
        p = Pm[s_, sel]
        if not p.any():
            continue
        Ys = Y[:, sel]
        tested = ~np.isnan(Ys)
        tested[s_] = False
        pos = tested & (Ys == 1)
        for excl in (False, True):
            tt = tested & ~Pm[:, sel] if excl else tested
            pp = pos & tt
            N = tt.sum(1).astype(float)
            n = (tt & p).sum(1).astype(float)
            K = pp.sum(1).astype(float)
            a = (pp & p).sum(1).astype(float)
            ok = N > 0
            key = "_excl" if excl else ""
            out["O" + key] += float(a[ok].sum())
            out["E" + key] += float(np.sum(n[ok] * K[ok] / N[ok]))
            if not excl:
                out["n_pairs"] += int(np.sum(ok & (n > 0)))
    return out


def responsiveness_bins(resp: np.ndarray, n_bins: int = RESP_DECILES) -> pd.Series:
    """Decile bins of responsiveness that keep tied values together: average ranks cut at the
    decile quantiles, duplicate edges dropped (fewer than 10 bins when ties are large, e.g. the
    many never-DE targets share one bin); missing responsiveness is its own bin (-1)."""
    r = pd.Series(np.asarray(resp, dtype=float))
    out = pd.Series(np.full(len(r), -1, dtype=np.int64))
    ok = r.notna().to_numpy()
    if ok.sum() >= n_bins:
        out[ok] = pd.qcut(r[ok].rank(method="average"), n_bins, labels=False, duplicates="drop").astype(np.int64)
    elif ok.any():
        out[ok] = 0
    return out


def posthoc_controls(ec: pd.DataFrame, cohort: str) -> dict:
    """Responsiveness strata, within-TF responsiveness sums and wrong-TF sums of one cohort's
    evaluable edges (one universe x comparator)."""
    tot, tfde = cohort_de_summary(cohort)
    ec = ec.sort_values(["tf", "target"], kind="mergesort").reset_index(drop=True)
    resp = responsiveness(ec, tot)
    dec = responsiveness_bins(resp)
    e2 = ec.assign(resp=resp, decile=dec.to_numpy(), a=ec["promoted"] & ec["positive"])
    g = e2.groupby(["tf", "decile"], sort=True)
    strata = pd.DataFrame({"N": g.size(), "n": g["promoted"].sum(), "K": g["positive"].sum(),
                           "a": g["a"].sum()}).reset_index().assign(cohort=cohort)
    means = []
    for tf, gg in e2.groupby("tf", sort=True):
        pr = gg["promoted"].to_numpy(bool)
        if pr.any() and (~pr).any():
            means.append((float(pr.sum()), float(np.nanmean(gg["resp"][pr])), float(np.nanmean(gg["resp"][~pr]))))
    return dict(strata=strata, means=means, wrong=wrong_tf_sums(ec, tfde))


def unit_posthoc(ctrls: list) -> dict:
    """Aggregate the cohort-level post hoc controls of a unit."""
    out = {}
    if not ctrls:
        return out
    st = pd.concat([c["strata"] for c in ctrls], ignore_index=True)
    N, n, K, a = (st[c].to_numpy(np.int64) for c in ("N", "n", "K", "a"))
    O, Ex = oe(a, n, K, N)
    p = exact_pvalue(N, K, n, int(a.sum()))
    out.update(resp_strat_observed=O, resp_strat_expected=Ex, resp_strat_enrichment=O / Ex if Ex > 0 else np.nan,
               resp_strat_p_exact=p, resp_strat_mh_or=mantel_haenszel_or(a, n, K, N),
               resp_strat_n_strata_informative=int(informative(N, n, K).sum()))
    out["resp_strat_endpoint_pass"] = bool(Ex > 0 and O / Ex >= ENDPOINT and p <= ALPHA)
    m = np.array([x for c in ctrls for x in c["means"]], dtype=float).reshape(-1, 3)
    if len(m):
        w = m[:, 0]
        out["resp_mean_promoted"] = float(np.nansum(w * m[:, 1]) / w.sum())
        out["resp_mean_nonpromoted"] = float(np.nansum(w * m[:, 2]) / w.sum())
    W = {k: sum(c["wrong"][k] for c in ctrls) for k in ("O", "E", "O_excl", "E_excl", "n_pairs")}
    out.update(wrongtf_observed=W["O"], wrongtf_expected=W["E"],
               wrongtf_enrichment=W["O"] / W["E"] if W["E"] > 0 else np.nan, wrongtf_n_strata=int(W["n_pairs"]),
               wrongtf_excl_shared_enrichment=W["O_excl"] / W["E_excl"] if W["E_excl"] > 0 else np.nan)
    return out


def all_edges(universes=UNIVERSES) -> dict:
    ct = cohort_table()
    return {u: evaluable_edges(u, load_universe_tiers(u), ct) for u in universes}


def lock_check(lock: dict, design: pd.DataFrame, universes=UNIVERSES) -> dict:
    """Record of the lock validation of an outcome run: the design digest (equal to the locked one,
    otherwise ``check_lock`` has already stopped the run) and, per universe, the promoted-set sizes
    and NUTS diagnostics stored in the lock next to those of the fits read now.  A fit re-run can
    move boundary edges between tiers without touching any evaluable stratum; then the sizes differ
    but the locked design, and therefore the lock, still hold."""
    out = dict(checked_at_utc=_dt.datetime.now(_dt.timezone.utc).isoformat(),
               lock_written_at_utc=lock.get("written_at_utc"), locked_design_digest=lock.get("design_digest"),
               design_digest_now=design_digest(design),
               design_matches_lock=bool(design_digest(design) == lock.get("design_digest")), universes={})
    for u in universes:
        t = load_universe_tiers(u)
        now = dict(n_strong=t["n_strong"], n_moderate=t["n_moderate"], fit_meta=t["fit_meta"])
        at_lock = {k: lock.get("promoted_sets", {}).get(u, {}).get(k) for k in ("n_strong", "n_moderate", "fit_meta")}
        out["universes"][u] = dict(at_lock=at_lock, now=now,
                                   promoted_set_changed_since_lock=bool(at_lock["n_strong"] != now["n_strong"]
                                                                        or at_lock["n_moderate"] != now["n_moderate"]
                                                                        or at_lock["fit_meta"] != now["fit_meta"]))
    return out


def run_outcomes(universes=UNIVERSES) -> tuple[pd.DataFrame, pd.DataFrame]:
    edges = all_edges(universes)
    design = []
    for u, ed in edges.items():
        for comp in COMPARATORS:
            sub = ed if comp == "universe" else ed[ed["observed"]]
            dd = stratum_design(sub)
            dd.insert(0, "comparator", comp)
            dd.insert(0, "universe", u)
            design.append(dd)
    design = pd.concat(design, ignore_index=True).merge(
        cohort_table()[["cohort_id", UNIT_COL]].rename(columns={"cohort_id": "cohort"}), on="cohort", how="left")
    lock = check_lock(design)
    lchk = lock_check(lock, design, universes)
    C.write_json(lchk, OUT_DIR / "lock_check.json")
    power = pd.read_csv(C.TABLES / "rep_power.csv")
    strata = []
    for u, ed in edges.items():
        for comp in COMPARATORS:
            sub = ed if comp == "universe" else ed[ed["observed"]]
            s = outcome_strata(sub)
            s.insert(0, "comparator", comp)
            s.insert(0, "universe", u)
            strata.append(s)
    strata = pd.concat(strata, ignore_index=True)
    strata.to_csv(OUT_DIR / "strata_outcomes.csv", index=False)
    tf_table(strata).to_csv(C.TABLES / "rep_tf_strata.csv", index=False)
    # post hoc controls per (universe, comparator, cohort) (computed after the lock)
    ctrl = {}
    for u, ed in edges.items():
        for comp in COMPARATORS:
            sub = ed if comp == "universe" else ed[ed["observed"]]
            for c, ec in sub.groupby("cohort", sort=True):
                ctrl[(u, comp, c)] = posthoc_controls(ec[["tf", "target", "promoted", "positive"]], c)
    rows = []
    for _, pr in power.iterrows():
        cohorts = pr["cohorts"].split(";")
        d = strata[(strata["universe"] == pr["universe"]) & (strata["comparator"] == pr["comparator"])
                   & strata["cohort"].isin(cohorts)]
        base = {k: pr[k] for k in ("universe", "level", "unit", "comparator", "cohorts", "cell_types", "status",
                                   "n_strata", "n_strata_informative", "n_evaluable_edges",
                                   "n_promoted_evaluable", "n_positives", "positive_rate", "power", "mde_rr",
                                   "floor_edges", "adequately_powered", "power_cond", "power_ge_target")}
        if pr["status"] == "evaluable":
            # the seed depends on the strata (cohort set), so a study with one cohort reproduces
            # its cohort-level resampling exactly
            base.update(unit_outcome(d, unit_seed("outcome", pr["universe"], pr["comparator"],
                                                  ";".join(sorted(cohorts)))))
            base.update(unit_posthoc([ctrl[(pr["universe"], pr["comparator"], c)] for c in cohorts
                                      if (pr["universe"], pr["comparator"], c) in ctrl]))
            if np.isfinite(base.get("wrongtf_enrichment", np.nan)) and base["wrongtf_enrichment"] > 0:
                base["specific_excess"] = base["enrichment"] / base["wrongtf_enrichment"]
            if np.isfinite(base.get("wrongtf_excl_shared_enrichment", np.nan)) and base["wrongtf_excl_shared_enrichment"] > 0:
                base["specific_excess_excl_shared"] = base["enrichment"] / base["wrongtf_excl_shared_enrichment"]
            # endpoint power with the locked design's simulated tables (same seed as STEP 1)
            pw_seed = unit_seed(pr["universe"], pr["level"], pr["unit"], pr["comparator"])
            base["endpoint_power_rr2"] = simulate_endpoint_power(d["N"], d["n"], float(pr["positive_rate"]), RR_TRUE,
                                                                 seed=pw_seed)[0]
            base["floor_edges_min_scan"] = support_floor_min_scan(int(pr["n_evaluable_edges"]),
                                                                  float(pr["positive_rate"]), float(pr["floor_edges"]))
            base["adequately_powered_min_scan"] = bool(pr["n_promoted_evaluable"] >= base["floor_edges_min_scan"])
        else:
            base["endpoint_status"] = "non-evaluable"
        rows.append(base)
    res = pd.DataFrame(rows)
    res["analysis_status"] = "prespecified"
    res["post_hoc_columns"] = POST_HOC_UNIT_COLUMNS
    res["p_holm"] = np.nan
    for _, g in res.groupby(["universe", "level", "comparator"]):
        ev = g[g["status"] == "evaluable"]
        res.loc[ev.index, "p_holm"] = holm(ev["p_exact"].to_numpy())
    res["endpoint_pass"] = res["endpoint_pass"].astype("boolean")
    res.insert(len(res.columns), "power_lock_written_at_utc", lock["written_at_utc"])
    res["promoted_set_changed_since_lock"] = res["universe"].map(
        {u: v["promoted_set_changed_since_lock"] for u, v in lchk["universes"].items()})
    res["locked_design_reproduced"] = bool(lchk["design_matches_lock"])
    res.to_csv(C.TABLES / "rep_units.csv", index=False)
    rates = pass_rates(res)
    rates.to_csv(C.TABLES / "rep_pass_rates.csv", index=False)
    print(res[res["comparator"] == "universe"][["universe", "level", "unit", "status", "n_promoted_evaluable",
                                                "enrichment", "mh_odds_ratio", "p_exact", "endpoint_pass",
                                                "adequately_powered"]].to_string())
    print(rates.to_string())
    return res, rates


def tf_table(strata: pd.DataFrame) -> pd.DataFrame:
    """Per (universe, cohort, TF) stratum of the primary comparator: margins, promoted positives,
    TF-wise O/E and positive rates (manuscript appendix / interpretation of TF concentration)."""
    s = strata[strata["comparator"] == "universe"].copy()
    s = s.merge(cohort_table()[["cohort_id", UNIT_COL, "cell_type_group"]].rename(columns={"cohort_id": "cohort"}),
                on="cohort", how="left")
    s["expected"] = s["n"] * s["K"] / s["N"]
    s["enrichment"] = np.where(s["expected"] > 0, s["a"] / s["expected"].where(s["expected"] > 0, 1), np.nan)
    s["positive_rate"] = s["K"] / s["N"]
    s["promoted_positive_rate"] = np.where(s["n"] > 0, s["a"] / s["n"].where(s["n"] > 0, 1), np.nan)
    s["informative"] = informative(s["N"], s["n"], s["K"])
    s["share_of_unit_promoted_positives"] = s["a"] / s.groupby(["universe", "cohort"])["a"].transform("sum").replace(0, np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        s["max_attainable_enrichment"] = np.where(s["expected"] > 0, np.minimum(s["n"], s["K"]) / s["expected"], np.nan)
    s["analysis_status"] = "prespecified"
    s["post_hoc_columns"] = "max_attainable_enrichment"
    cols = ["universe", "cohort", UNIT_COL, "cell_type_group", "tf", "N", "n", "n_strong", "n_moderate", "K", "a",
            "a_strong", "a_moderate", "expected", "enrichment", "positive_rate", "promoted_positive_rate",
            "informative", "share_of_unit_promoted_positives", "max_attainable_enrichment", "analysis_status",
            "post_hoc_columns"]
    return s[cols].sort_values(["universe", "cohort", "tf"]).reset_index(drop=True)


def pass_rates(res: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (u, level, comp), g in res.groupby(["universe", "level", "comparator"], sort=False):
        ev = g[g["status"] == "evaluable"]
        subsets = {"all_evaluable": ev, "adequately_powered": ev[ev["adequately_powered"].astype(bool)],
                   "non_K562": ev[~ev["cell_types"].str.contains("K562")]}
        if "power_ge_target" in ev.columns:
            subsets["power_ge_target"] = ev[ev["power_ge_target"].astype(bool)]
        if "endpoint_attainable" in ev.columns:
            subsets["endpoint_attainable"] = ev[ev["endpoint_attainable"].astype(bool)]
            subsets["adequately_powered_and_attainable"] = ev[ev["adequately_powered"].astype(bool)
                                                              & ev["endpoint_attainable"].astype(bool)]
        if "resp_strat_endpoint_pass" in ev.columns:
            subsets["adequately_powered__resp_strat_endpoint"] = ev[ev["adequately_powered"].astype(bool)]
        for name, s in subsets.items():
            col = "resp_strat_endpoint_pass" if name.endswith("__resp_strat_endpoint") else "endpoint_pass"
            k = int(s[col].astype("boolean").fillna(False).astype(bool).sum())
            lo, hi = wilson(k, len(s))
            rows.append(dict(universe=u, level=level, comparator=comp, subset=name, n_units=len(s), n_pass=k,
                             pass_rate=k / len(s) if len(s) else np.nan, wilson_lo=lo, wilson_hi=hi,
                             units=";".join(s["unit"]),
                             units_passing=";".join(s.loc[s[col].astype("boolean").fillna(False).astype(bool), "unit"]),
                             n_non_evaluable=int((g["status"] != "evaluable").sum()),
                             analysis_status=("prespecified" if name in ("all_evaluable", "adequately_powered",
                                                                         "non_K562") else "post_hoc")))
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# STEP 3: data-derived support threshold (R1 approach) and its LOO re-estimation
# ----------------------------------------------------------------------------
SUPPORT_MEASURES = ("n_promoted_evaluable", "expected_promoted_positives", "n_positives")


def data_derived_threshold(support: np.ndarray, passed: np.ndarray, min_units: int = 2) -> tuple[float, int, int]:
    """tau maximising the pass rate among units with support >= tau (candidates = observed support
    values; ties -> smallest tau; at least ``min_units`` units retained).  Returns
    (tau, n_retained, n_pass_retained); tau = nan if no candidate retains min_units units."""
    support = np.asarray(support, dtype=float)
    passed = np.asarray(passed, dtype=bool)
    best = (np.nan, 0, 0, -1.0)
    for tau in np.unique(support):
        keep = support >= tau
        if keep.sum() < min_units:
            continue
        rate = passed[keep].mean()
        if rate > best[3] + 1e-12:
            best = (float(tau), int(keep.sum()), int(passed[keep].sum()), rate)
    return best[0], best[1], best[2]


def run_threshold(res: pd.DataFrame | None = None) -> pd.DataFrame:
    if res is None:
        res = pd.read_csv(C.TABLES / "rep_units.csv")
    rows = []
    power = pd.read_csv(C.TABLES / "rep_power.csv")
    for level in LEVELS:
        ev = res[(res["universe"] == PRIMARY_UNIVERSE) & (res["level"] == level) & (res["comparator"] == "universe")
                 & (res["status"] == "evaluable")].reset_index(drop=True)
        pw = power[(power["universe"] == PRIMARY_UNIVERSE) & (power["level"] == level)
                   & (power["comparator"] == "universe")].set_index("unit")
        passed = ev["endpoint_pass"].astype(bool).to_numpy()
        for meas in SUPPORT_MEASURES:
            sup = ev[meas].to_numpy(float)
            tau, nk, npass = data_derived_threshold(sup, passed)
            base = dict(universe=PRIMARY_UNIVERSE, level=level, support_measure=meas, n_units=len(ev))
            rows.append(dict(base, fit="full", left_out="", tau=tau, n_retained=nk, n_pass_retained=npass,
                             pass_rate_retained=npass / nk if nk else np.nan,
                             units_retained=";".join(ev.loc[sup >= tau, "unit"]) if np.isfinite(tau) else ""))
            taus = []
            for i, unit in enumerate(ev["unit"]):
                keep = np.arange(len(ev)) != i
                t_i, nk_i, np_i = data_derived_threshold(sup[keep], passed[keep])
                taus.append(t_i)
                above = bool(sup[i] >= t_i) if np.isfinite(t_i) else False
                floor_i = pw.loc[unit, "floor_edges"] if meas == "n_promoted_evaluable" else (
                    pw.loc[unit, "floor_expected_positives"] if meas == "expected_promoted_positives" else np.nan)
                rows.append(dict(base, fit="loo", left_out=unit, tau=t_i, n_retained=nk_i, n_pass_retained=np_i,
                                 pass_rate_retained=np_i / nk_i if nk_i else np.nan,
                                 units_retained=";".join(ev.loc[keep & (sup >= t_i), "unit"]) if np.isfinite(t_i) else "",
                                 left_out_support=sup[i], left_out_pass=bool(passed[i]),
                                 left_out_above_tau=above, rule_agrees=bool(above == passed[i]),
                                 left_out_power_floor=floor_i, tau_changed=bool(not np.isclose(t_i, tau, equal_nan=True))))
            t = np.array(taus, dtype=float)
            fin = t[np.isfinite(t)]
            rows.append(dict(base, fit="loo_summary", left_out="", tau=tau,
                             tau_loo_min=float(fin.min()) if len(fin) else np.nan,
                             tau_loo_max=float(fin.max()) if len(fin) else np.nan,
                             n_loo_tau_changed=int(np.sum(~np.isclose(t, tau, equal_nan=True))),
                             n_loo_undefined=int(np.sum(~np.isfinite(t))),
                             loo_rule_agreement=float(np.mean([r["rule_agrees"] for r in rows[-len(ev):]])) if len(ev) else np.nan))
    df = pd.DataFrame(rows)
    df["analysis_status"] = "prespecified"
    df.to_csv(C.TABLES / "rep_threshold_loo.csv", index=False)
    print(df.to_string())
    return df


# ----------------------------------------------------------------------------
# STEP 4/5: Bayesian random-effects model
# ----------------------------------------------------------------------------
def re_data(strata: pd.DataFrame, units: dict, unit_ids: list) -> dict:
    """Binomial counts per stratum (strata with >= 1 positive) for the given units."""
    rows = []
    for ui, u in enumerate(unit_ids):
        d = strata[strata["cohort"].isin(units[u]) & (strata["K"] > 0)]
        for _, r in d.iterrows():
            rows.append((ui, int(r["a"]), int(r["n"]), int(r["K"] - r["a"]), int(r["N"] - r["n"])))
    arr = np.array(rows, dtype=np.int64).reshape(-1, 5)
    return dict(u=arr[:, 0], y1=arr[:, 1], n1=arr[:, 2], y0=arr[:, 3], n0=arr[:, 4], U=len(unit_ids),
                unit_ids=list(unit_ids))


def re_model_fn(mu_mean: float, mu_sd: float, sigma_sd: float, centred: np.ndarray, tf_slope_sd: float | None = None):
    """numpyro model of STEP 4.  ``centred`` (bool per stratum) selects the parametrisation of
    a_us: centred (a_us sampled directly) for strata whose own counts pin a_us down
    (>= ``RE_CENTRED_MIN_POS`` positives), non-centred (a_us = m_u + s_u z_us) otherwise.  Both
    give the same joint density; the mix avoids the funnel of weakly informed strata (Dixit)
    and the slow mixing of a non-centred a for data-rich strata."""
    import jax.numpy as jnp
    import numpyro
    import numpyro.distributions as dist

    ic = np.flatnonzero(centred)
    inc = np.flatnonzero(~centred)
    S = len(centred)

    def model(u, y1, n1, y0, n0, U):
        mu = numpyro.sample("mu", dist.Normal(mu_mean, mu_sd))
        sigma = numpyro.sample("sigma", dist.HalfNormal(sigma_sd))
        with numpyro.plate("units", U):
            eta = numpyro.sample("eta", dist.Normal(0.0, 1.0))
            m = numpyro.sample("m", dist.Normal(RE_M_PRIOR[0], RE_M_PRIOR[1]))
            s = numpyro.sample("s", dist.HalfNormal(RE_S_PRIOR))
        theta = numpyro.deterministic("theta", mu + sigma * eta)
        a = jnp.zeros(S)
        if len(ic):
            with numpyro.plate("strata_c", len(ic)):
                a_c = numpyro.sample("a_c", dist.Normal(m[u[ic]], s[u[ic]]))
            a = a.at[ic].set(a_c)
        if len(inc):
            with numpyro.plate("strata_nc", len(inc)):
                z = numpyro.sample("z_nc", dist.Normal(0.0, 1.0))
            a = a.at[inc].set(m[u[inc]] + s[u[inc]] * z)
        a = numpyro.deterministic("a", a)
        slope = theta[u]
        if tf_slope_sd:      # post hoc sensitivity: TF-level random slope theta_us = theta_u + tau eps_s
            tau = numpyro.sample("tau_tf", dist.HalfNormal(tf_slope_sd))
            with numpyro.plate("strata_slope", S):
                eps = numpyro.sample("eps_tf", dist.Normal(0.0, 1.0))
            slope = slope + tau * eps
        with numpyro.plate("strata", S):
            numpyro.sample("y1", dist.Binomial(total_count=n1, logits=a + slope), obs=y1)
            numpyro.sample("y0", dist.Binomial(total_count=n0, logits=a), obs=y0)
    return model


def fit_re(data: dict, profile: str, seed: int) -> dict:
    import jax
    jax.config.update("jax_enable_x64", True)
    import arviz as az
    import jax.numpy as jnp
    from numpyro.infer import MCMC, NUTS, init_to_value

    slope_sd = TF_SLOPE_SD if profile == TF_SLOPE_PROFILE else None
    mu_mean, mu_sd, sigma_sd = PRIOR_PROFILES[PRIMARY_PROFILE if slope_sd else profile]
    centred = (data["y0"] + data["y1"]) >= RE_CENTRED_MIN_POS
    model = re_model_fn(mu_mean, mu_sd, sigma_sd, centred, slope_sd)
    # start the stratum intercepts at their empirical logits (others at prior-typical values)
    emp = np.log((data["y0"] + 0.5) / (data["n0"] - data["y0"] + 0.5))
    m0 = np.array([emp[data["u"] == i].mean() if np.any(data["u"] == i) else RE_M_PRIOR[0]
                   for i in range(data["U"])])
    init = dict(mu=0.0, sigma=0.1 * sigma_sd, eta=jnp.zeros(data["U"]), m=jnp.asarray(m0), s=jnp.ones(data["U"]))
    if centred.any():
        init["a_c"] = jnp.asarray(emp[centred])
    if (~centred).any():
        init["z_nc"] = jnp.asarray(np.clip(emp[~centred] - m0[data["u"][~centred]], -2, 2))
    if slope_sd:
        init["tau_tf"] = 0.1 * slope_sd
        init["eps_tf"] = jnp.zeros(len(data["u"]))
    kernel = NUTS(model, target_accept_prob=NUTS_SETTINGS["target_accept"], init_strategy=init_to_value(values=init))
    mcmc = MCMC(kernel, num_warmup=NUTS_SETTINGS["num_warmup"], num_samples=NUTS_SETTINGS["num_samples"],
                num_chains=NUTS_SETTINGS["num_chains"], chain_method="sequential", progress_bar=False)
    t0 = time.time()
    args = {k: jnp.asarray(data[k]) for k in ("u", "y1", "n1", "y0", "n0")}
    mcmc.run(jax.random.PRNGKey(seed), U=data["U"], extra_fields=("diverging",), **args)
    sc = mcmc.get_samples(group_by_chain=True)
    div = int(np.asarray(mcmc.get_extra_fields(group_by_chain=True)["diverging"]).sum())
    diag, diag_elem = {}, {}
    for k in ("mu", "sigma", "theta", "eta", "m", "s", "a") + (("tau_tf",) if slope_sd else ()):
        arr = np.asarray(sc[k])
        arr = arr.reshape(arr.shape[0], arr.shape[1], -1)
        ds = az.convert_to_dataset({k: arr})
        rh = np.asarray(az.rhat(ds)[k].values).ravel()
        eb = np.asarray(az.ess(ds, method="bulk")[k].values).ravel()
        et = np.asarray(az.ess(ds, method="tail")[k].values).ravel()
        diag[k] = (float(np.nanmax(rh)), float(np.nanmin(eb)), float(np.nanmin(et)))
        diag_elem[k] = (rh, eb, et)
    flat = {k: np.asarray(v).reshape((-1,) + np.asarray(v).shape[2:]) for k, v in sc.items()}
    rng = np.random.default_rng(seed)
    flat["theta_new"] = flat["mu"] + flat["sigma"] * rng.standard_normal(flat["mu"].shape[0])
    return dict(draws=flat, diag=diag, diag_elem=diag_elem, n_divergent=div, runtime_s=time.time() - t0,
                profile=profile, unit_ids=data["unit_ids"], n_centred=int(centred.sum()),
                n_noncentred=int((~centred).sum()))


def summarize_draws(x: np.ndarray) -> dict:
    thr = math.log(ENDPOINT)
    return dict(mean=float(x.mean()), sd=float(x.std()), q025=float(np.quantile(x, 0.025)),
                q50=float(np.quantile(x, 0.5)), q975=float(np.quantile(x, 0.975)),
                or_median=float(math.exp(np.quantile(x, 0.5))), or_q025=float(math.exp(np.quantile(x, 0.025))),
                or_q975=float(math.exp(np.quantile(x, 0.975))),
                p_gt0=float(np.mean(x > 0)), p_or_ge_endpoint=float(np.mean(x >= thr)))


def fit_rows(fit: dict) -> list:
    rows = []
    D = fit["draws"]
    rmax = max(v[0] for v in fit["diag"].values())
    emin = min(v[1] for v in fit["diag"].values())
    tmin = min(v[2] for v in fit["diag"].values())
    common = dict(r_hat_max_all=rmax, ess_bulk_min_all=emin, ess_tail_min_all=tmin,
                  n_divergent=fit["n_divergent"], runtime_s=fit["runtime_s"])
    for par in ("mu", "sigma", "theta_new") + (("tau_tf",) if "tau_tf" in D else ()):
        dg = fit["diag"].get(par, (np.nan, np.nan, np.nan))
        rows.append(dict(parameter=par, unit="", **summarize_draws(D[par]),
                         r_hat=dg[0], ess_bulk=dg[1], ess_tail=dg[2], **common))
    el = fit.get("diag_elem", {}).get("theta")
    for i, u in enumerate(fit["unit_ids"]):
        # per-unit R-hat / ESS of theta_u (the group maxima/minima are in r_hat_max_all etc.)
        rows.append(dict(parameter="theta", unit=u, **summarize_draws(D["theta"][:, i]),
                         r_hat=float(el[0][i]) if el is not None else fit["diag"]["theta"][0],
                         ess_bulk=float(el[1][i]) if el is not None else fit["diag"]["theta"][1],
                         ess_tail=float(el[2][i]) if el is not None else fit["diag"]["theta"][2], **common))
    return rows


def evaluable_units(res: pd.DataFrame, level: str) -> list:
    ev = res[(res["universe"] == PRIMARY_UNIVERSE) & (res["level"] == level) & (res["comparator"] == "universe")
             & (res["status"] == "evaluable")]
    return list(ev["unit"])


def load_strata(universe: str = PRIMARY_UNIVERSE, comparator: str = "universe") -> pd.DataFrame:
    s = pd.read_csv(OUT_DIR / "strata_outcomes.csv")
    return s[(s["universe"] == universe) & (s["comparator"] == comparator)]


def run_bayes() -> None:
    res = pd.read_csv(C.TABLES / "rep_units.csv")
    strata = load_strata()
    out_main, out_sens = [], []
    fits_dir = C.ensure_dir(OUT_DIR / "bayes")
    for level in LEVELS:
        units = units_of(None, level)
        ids = evaluable_units(res, level)
        data = re_data(strata, units, ids)
        for profile in list(PRIOR_PROFILES) + [TF_SLOPE_PROFILE]:
            seed = unit_seed("bayes", level, profile)
            fit = fit_re(data, profile, seed)
            rows = fit_rows(fit)
            mu_mean, mu_sd, sigma_sd = PRIOR_PROFILES[PRIMARY_PROFILE if profile == TF_SLOPE_PROFILE else profile]
            for r in rows:
                r.update(level=level, profile=profile, prior_mu=f"Normal({mu_mean:g}, {mu_sd:g})",
                         prior_sigma=f"HalfNormal({sigma_sd:g})", n_units=data["U"], n_strata=len(data["u"]),
                         prior_tf_slope=(f"tau ~ HalfNormal({TF_SLOPE_SD:g})" if profile == TF_SLOPE_PROFILE else ""),
                         analysis_status="post_hoc" if profile == TF_SLOPE_PROFILE else "prespecified")
            out_sens.extend(rows)
            if profile == PRIMARY_PROFILE:
                out_main.extend(rows)
                np.savez_compressed(fits_dir / f"{level}_{profile}_draws.npz",
                                    **{k: v for k, v in fit["draws"].items() if k in ("mu", "sigma", "theta", "theta_new")},
                                    unit_ids=np.array(ids))
            print(f"[bayes] {level} {profile}: mu={rows[0]['mean']:.3f} [{rows[0]['q025']:.3f},{rows[0]['q975']:.3f}] "
                  f"rhat={rows[0]['r_hat_max_all']:.3f} ess={rows[0]['ess_bulk_min_all']:.0f} div={fit['n_divergent']} "
                  f"({fit['runtime_s']:.0f}s)", flush=True)
    cols_first = ["level", "profile", "prior_mu", "prior_sigma", "n_units", "n_strata", "parameter", "unit"]
    main = pd.DataFrame(out_main)
    main = main[cols_first + [c for c in main.columns if c not in cols_first]]
    # attach the frequentist per-unit estimates for reference
    ref = res[(res["universe"] == PRIMARY_UNIVERSE) & (res["comparator"] == "universe")][
        ["level", "unit", "observed_promoted_positives", "expected_promoted_positives", "mh_odds_ratio", "enrichment",
         "p_exact"]].rename(
        columns={"observed_promoted_positives": "unit_observed", "expected_promoted_positives": "unit_expected",
                 "mh_odds_ratio": "unit_mh_odds_ratio", "enrichment": "unit_enrichment", "p_exact": "unit_p_exact"})
    main = main.merge(ref, on=["level", "unit"], how="left")
    main["post_hoc_columns"] = "r_hat / ess_bulk / ess_tail of theta rows are per unit (2026-10-08)"
    main.to_csv(C.TABLES / "rep_bayes.csv", index=False)
    sens = pd.DataFrame(out_sens)
    sens = sens[cols_first + [c for c in sens.columns if c not in cols_first]]
    sens.to_csv(C.TABLES / "rep_prior_sensitivity.csv", index=False)


def run_leavek() -> None:
    res = pd.read_csv(C.TABLES / "rep_units.csv")
    strata = load_strata()
    bayes = pd.read_csv(C.TABLES / "rep_bayes.csv")
    rows = []
    thr_name = "p_or_ge_endpoint"
    for level in LEVELS:
        units = units_of(None, level)
        ids = evaluable_units(res, level)
        full = bayes[bayes["level"] == level]
        mu_full = full[full["parameter"] == "mu"].iloc[0]
        p_full = full[full["parameter"] == "theta"].set_index("unit")[thr_name]
        for k in LEAVE_K:
            if k >= len(ids):
                continue
            for out in itertools.combinations(ids, k):
                keep = [u for u in ids if u not in out]
                data = re_data(strata, units, keep)
                seed = unit_seed("leavek", level, *out)
                fit = fit_re(data, PRIMARY_PROFILE, seed)
                fr = fit_rows(fit)
                mu = fr[0]
                th = {r["unit"]: r[thr_name] for r in fr if r["parameter"] == "theta"}
                dp = np.array([th[u] - p_full[u] for u in keep])
                rows.append(dict(level=level, k=k, scheme=f"L{k}O", left_out=";".join(out), units_kept=";".join(keep),
                                 n_units_kept=len(keep), degenerate_single_unit=bool(len(keep) == 1),
                                 mu_mean=mu["mean"], mu_q025=mu["q025"], mu_q975=mu["q975"],
                                 delta_mu_mean=mu["mean"] - mu_full["mean"],
                                 p_mu_or_ge_endpoint=mu[thr_name], delta_p_mu=mu[thr_name] - mu_full[thr_name],
                                 sigma_mean=fr[1]["mean"],
                                 mean_abs_delta_unit_pass_prob=float(np.mean(np.abs(dp))),
                                 max_abs_delta_unit_pass_prob=float(np.max(np.abs(dp))),
                                 unit_pass_probs=";".join(f"{u}={th[u]:.3f}" for u in keep),
                                 r_hat_max_all=mu["r_hat_max_all"], ess_bulk_min_all=mu["ess_bulk_min_all"],
                                 n_divergent=fit["n_divergent"], row_type="fit"))
                print(f"[leavek] {level} L{k}O -{out}: mu={mu['mean']:.3f} dmu={rows[-1]['delta_mu_mean']:+.3f} "
                      f"max|dp|={rows[-1]['max_abs_delta_unit_pass_prob']:.3f} rhat={mu['r_hat_max_all']:.3f} "
                      f"div={fit['n_divergent']}", flush=True)
            g = [r for r in rows if r["level"] == level and r["k"] == k and r["row_type"] == "fit"]
            rows.append(dict(level=level, k=k, scheme=f"L{k}O", left_out="ALL", n_fits=len(g),
                             mu_mean=float(np.mean([r["mu_mean"] for r in g])),
                             mean_abs_delta_mu=float(np.mean([abs(r["delta_mu_mean"]) for r in g])),
                             max_abs_delta_mu=float(np.max([abs(r["delta_mu_mean"]) for r in g])),
                             mean_abs_delta_p_mu=float(np.mean([abs(r["delta_p_mu"]) for r in g])),
                             max_abs_delta_p_mu=float(np.max([abs(r["delta_p_mu"]) for r in g])),
                             mean_abs_delta_unit_pass_prob=float(np.mean([r["mean_abs_delta_unit_pass_prob"] for r in g])),
                             max_abs_delta_unit_pass_prob=float(np.max([r["max_abs_delta_unit_pass_prob"] for r in g])),
                             mu_full=mu_full["mean"], p_mu_full=mu_full[thr_name],
                             r_hat_max_all=float(np.max([r["r_hat_max_all"] for r in g])),
                             n_divergent=int(np.sum([r["n_divergent"] for r in g])), row_type="summary"))
    df = pd.DataFrame(rows)
    df["analysis_status"] = "prespecified"
    df["note"] = ("degenerate_single_unit (post hoc flag): the fit keeps one unit, so mu is that unit's theta shrunk "
                  "towards the prior, not a pooled estimate")
    df.to_csv(C.TABLES / "rep_leavek.csv", index=False)
    print(df[df["row_type"] == "summary"].to_string())


# ----------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("stage", choices=["power", "outcomes", "threshold", "bayes", "leavek", "all"])
    ap.add_argument("--universes", nargs="+", default=list(UNIVERSES))
    ap.add_argument("--relock-reason", default="", help="power: reason for replacing an existing lock")
    a = ap.parse_args(argv)
    C.ensure_dir(OUT_DIR)
    t0 = time.time()
    if a.stage == "all" and LOCK.exists() and not a.relock_reason:
        # an existing lock is never replaced silently: 'all' keeps it (the outcome stage verifies that
        # the recomputed design and rep_power.csv equal the locked ones and stops otherwise)
        print(f"[power] {LOCK.name} exists (written {json.loads(LOCK.read_text()).get('written_at_utc')}); "
              "STEP 1 is not re-run; use 'power --relock-reason ...' to replace the lock")
    elif a.stage in ("power", "all"):
        run_power(tuple(a.universes), a.relock_reason)
    if a.stage in ("outcomes", "all"):
        run_outcomes(tuple(a.universes))
    if a.stage in ("threshold", "all"):
        run_threshold()
    if a.stage in ("bayes", "all"):
        run_bayes()
    if a.stage in ("leavek", "all"):
        run_leavek()
    record_decisions("replication", ANALYSIS_DECISIONS)
    print(f"[replication] {a.stage} done in {time.time() - t0:.0f} s")


def record_decisions(module: str, decisions: list, path: Path | None = None) -> None:
    """Replace this module's rows of the shared dated decision log (``eval_analysis_decisions.csv``)."""
    path = path or (C.TABLES / "eval_analysis_decisions.csv")
    new = pd.DataFrame(decisions).assign(module=module)
    if path.exists():
        old = pd.read_csv(path)
        new = pd.concat([old[old["module"] != module], new], ignore_index=True)
    C.ensure_dir(path.parent)
    new[["module", "id", "date", "status", "item", "note"]].sort_values(["module", "id"], kind="mergesort").to_csv(
        path, index=False)


if __name__ == "__main__":
    main()
