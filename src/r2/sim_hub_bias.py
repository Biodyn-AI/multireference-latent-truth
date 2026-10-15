"""Hub-bias simulation with known truth (reviewer Q9, plan section 2.3 E6).

Question
--------
Does Family-HB (or any other credibility estimator) preferentially promote edges of
popular, well-curated TFs beyond what biology supports?  On real data the share of the
promoted set at high-literature-degree TFs exceeds the share of held-out positives there
for every estimator (``data/r2/tables/heldout_degree_calibration.csv``), but held-out
positives are themselves concentrated at assayed, popular TFs, so the real data cannot say
how much of the excess is estimator behaviour.  Here the truth is known.

Generative design (``SIM_DESIGN``; one replicate = one universe of 400 TFs x 3,000 targets)
-------------------------------------------------------------------------------------------
* TF popularity (study intensity) ``u_t``: ``log u_t ~ N(0, 1.2^2)``.  Popularity deciles
  are equal-count TF deciles of ``u_t`` (decile 10 = most popular).
* Truth ``z_tg ~ Bernoulli(min(p_t h_g, 0.5))`` with gene propensity
  ``h_g ~ logN(-sd^2/2, 0.5^2)`` and TF degree
    - ``null``    : ``p_t = p0 exp(eta_t)`` (independent of popularity),
    - ``bio_hub`` : ``p_t = p0 exp(eta_t) u_t^0.3 / mean(u^0.3)`` (mildly increasing),
    - ``copying`` : as ``null``,
  ``eta_t ~ N(-sd^2/2, 0.5^2)``, ``p0 = 0.01``.
* Literature copying (correlated false positives within LIT): a non-edge is a *copied* edge
  with probability ``c_t``; ``c_t = c_base`` (0.002, popularity-independent) in ``null`` and
  ``bio_hub``; ``c_t = c_base + c_hub u_t / mean(u)`` (c_hub = 0.004) in ``copying``, which
  concentrates copied false positives at popular TFs.
* 12 LIT annotators.  TF coverage ``P(t in cov_j) = sigmoid(logit(kappa_j) + 1.0 log u_t)``;
  every TF is covered by >= 1 LIT annotator (otherwise it is given to the annotator with the
  highest coverage probability: the universe is literature-covered TFs, as on real data).
  Reporting probability for a covered TF: true edge ``sigmoid(logit(alpha_j) + 0.5 log u_t)``,
  copied edge ``sigmoid(logit(gamma_j) + 0.5 log u_t)``, other non-edge ``beta_j``.  The
  popularity slopes are shared by all LIT annotators, so LIT reports are correlated through
  ``u_t`` even given the latent class (the conditional-independence assumption of the
  latent-class models is violated in the realistic direction).
  LIT observation follows the real literal rule (``evidence.py``, ``prespecified`` and the LIT
  part of ``prespecified_cc``): j observes (t, g) iff t is in cov_j and g is a target of >= 1
  report of j (target-node presence; this makes LIT observation depend on j's own reports,
  as on real data).
* 2 MOTIF annotators: a TF has a motif with probability 0.6 (independent of popularity); each
  MOTIF annotator covers a motif TF with probability 0.9.  Reporting: true edge alpha_m,
  promiscuous-motif burst edge gamma_m (shared by both MOTIF annotators = within-family
  correlated false positives), other non-edge beta_m.  Burst edges: non-edges with TF-level
  probability ``m_t = min(0.04 exp(N(-0.5, 1)), 0.3)``, independent of popularity.
* 1 COEXP annotator: covers a TF with probability 0.7; true edge alpha_c, non-edge beta_c.
* Curated-conditional observation (the primary real mask ``prespecified_cc``): a MOTIF/COEXP
  annotator observes (t, g) only if t is covered by it (whole-resource rule, ``PRESPECIFIED
  ['motif_coexp_tf_coverage']``) and (t, g) is in the conditioning set Cond = edges with >= 1
  LIT report (the real Cond is the DoRothEA-channel subset of literature edges; here all LIT
  reports).  Cond depends only on observed LIT reports, so this restriction alone is ignorable
  (missing at random) for the latent-class likelihood; the LIT target-node rule is not.
* Rates are calibrated loosely to the real global Family-HB fit
  (``outputs/r2/fits/global/prespecified_cc/family_hb/params.npz``,
  ``data/r2/tables/fit_source_parameters.csv``): LIT sensitivity 0.1-0.45 at median
  popularity (real 0.19-0.67 conditional on observation), LIT false-positive rate
  0.001-0.006 (real median ~0.004), copy rates 0.03-0.85 (real B_LIT gamma 0.02-0.96), MOTIF
  alpha/beta/gamma 0.09-0.10 / 0.015-0.02 / 0.65-0.75 (real 0.08 / 0.015-0.023 / 0.62-0.76),
  COEXP 0.12 / 0.07 (real 0.108 / 0.075).  ``calibration`` writes the side-by-side table.

Methods and promoted sets
-------------------------
* Latent-class models fitted by multi-start MAP (``lcm.fit_map``): em, dependence,
  family_regularized, family_hb, and family_hb_obs / em_obs when present in
  ``lcm.MODEL_SPECS`` (the informative-observation extension, designed after the held-out
  results were seen).  Restarts: 30 for family_hb, 20 for family_hb_obs, 10 for the others
  (plan section 6 uses 50 on the real data; 60 replicates make that too expensive here, and
  ``sim_hub_modes.csv`` reports how many restarts reach the best optimum).  Realised coverage
  (``REALISED_COVERAGE``, final): every model on all 60 replicates except family_hb_obs, which
  was fitted on 12 (null 0-4, bio_hub 0-3, copying 1-3) and stopped for compute reasons; its
  rows are means over those replicates, and ``n_reps`` in every table says so.  The
  informative-observation models use the eligibility-coded evidence (-1 = TF covered by the LIT
  annotator but target not one of its nodes; -2 = TF not covered, or MOTIF/COEXP not observed).
  Promoted set = largest prefix of whole tie blocks with Bayesian FDR (mean of
  1 - credibility) <= 0.05 (``governance.bfdr_levels``).
* Baselines (``lcm.baseline_scores``): union, majority, vote_fraction, fisher, stouffer.
  Fisher and Stouffer are the mid-p versions implemented there (reported: p = rate_j / 2, not
  reported: p = (1 + rate_j) / 2), as in every other R2 module, not the literal plan-section-6
  wording (p = rate_j / p = 1), under which Stouffer is -infinity for any unreported entry.
  Promoted set = size-matched top set (size = the Family-HB BFDR-0.05 set of the same
  replicate; ties at the boundary broken uniformly at random with a seed derived from
  ``PRESPECIFIED['topk_tie_seed']``).  Every model also gets the size-matched set, so all
  methods can be compared at equal size.  A second size-matched rule (``size_oracle``) uses
  the size of the oracle BFDR-0.05 set (what the evidence supports at 5% FDR when the
  generative model is known); it is larger and shows the size dependence of the index.
* ``oracle``: posterior P(true | reports) under the true generative emission model with the
  true TF-level rates and priors, treating the observation mask as ignorable exactly as the
  MAR models do (unobserved entries are skipped).  It shows what the evidence supports when
  the reporting process is known; its BFDR-0.05 set is the reference for "beyond what the
  evidence supports", whereas the truth itself is the reference for "beyond biology".
  The ``oracle-check`` stage (added 2026-10-13) compares it with the oracle that counts a
  covered-but-unobserved LIT entry as a non-report, which is what such an entry is in this
  generator (``sim_hub_oracle_check.csv``).

Metrics (vs truth)
------------------
precision, recall, realised FDR vs estimated BFDR, AUROC (all edges; observed edges;
within-TF stratified), AP lift; per popularity decile: share of promoted vs share of true
edges, precision, recall, mean credibility vs true-edge rate (calibration).
Over-promotion index (top decile) = (share of promoted edges at top-popularity-decile TFs) /
(share of TRUE edges at those TFs); 1 = proportional to biology.  The same index over
observable literature-degree quintiles (TF-count quintiles of the number of LIT-reported
targets) mirrors the empirical index of ``heldout_degree_calibration.csv``.
Mode diagnostics (every latent-class model): restarts within 1e-3 of the best log posterior,
distinct optima, and label alignment with the truth (AUROC of P(T | pattern) vs the AUROC of
every other class posterior on reported edges; an optimum is ``label_swapped`` when another
class tracks the truth better than T; for a two-class model this is AUROC(T) < 0.5).
``t_class_inflated`` flags an optimum whose class-T weight exceeds ``T_INFLATION_FACTOR`` (10)
times the true prevalence: class T then describes something far more common than true edges
(added on 2026-10-13 after the first informative-observation fits showed such optima; it is a
diagnostic and changes no fit, set or metric).

Analysis status
---------------
Every row of every table carries ``analysis_status = post_hoc``.  The simulation (E6) was
planned before any held-out evaluation (plan section 2.3), and it never reads held-out data,
but its numerical design was fixed in ``SIM_DESIGN`` on 2026-10-08, after the results
checkpoint (plan section 8: the empirical hub over-promotion was known) and before any
simulation fit.  By the project rule (anything designed after held-out outcomes were seen is
post hoc) it is therefore not prespecified.  The column ``status_basis`` separates
``design_fixed_after_checkpoint`` (the core design: MAR ladder, baselines, oracle),
``model_designed_after_heldout`` (family_hb_obs, em_obs) and ``comparison_with_heldout_table``
(``sim_hub_empirical.csv``).  The design was not changed after the first simulation fit.

Diagnostics added on 2026-10-13 (after the first informative-observation fits; they change
no fit, promoted set or metric of the design)
-----------------------------------------------------------------------------------------
* ``restart-check``: 50 fresh restarts for the informative-observation fits whose designed
  fit has no inflated class T (``sim_hub_restart_check.csv``).
* ``oracle-check``: ignorable-mask oracle vs the oracle counting LIT non-reports.
* ``sim_hub_by_mode.csv`` (metrics split by the kind of optimum selected) and
  ``sim_hub_obs_signature.csv`` (parameter signature of the informative-observation fits,
  simulation next to the real-data fits; label-free parameters, read-only).

Commands
--------
    export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \\
        XLA_FLAGS='--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=2'
    PY=/Users/ihorkendiukhov/biodyn-work/.venv/bin/python
    nice -n 10 $PY src/r2/sim_hub_bias.py fit [--scenarios ...] [--reps 0-19] [--models ...]
                                           [--shard i/n] [--no-skip-existing] [--full-design]
    nice -n 10 $PY src/r2/sim_hub_bias.py restart-check [--models em_obs family_hb_obs]
    nice -n 10 $PY src/r2/sim_hub_bias.py oracle-check
    nice -n 10 $PY src/r2/sim_hub_bias.py metrics [--shard i/n] [--only-stale]
    nice -n 10 $PY src/r2/sim_hub_bias.py tables
    nice -n 10 $PY src/r2/sim_hub_bias.py all          # fit + metrics + tables, sequential
Tests: ``$PY -m pytest src/r2/tests/test_sim_hub_bias.py -q``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
import zlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from scipy.stats import rankdata
from scipy.stats import t as student_t

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from r2 import common as C  # noqa: E402

P = C.PRESPECIFIED
SEED = int(P["random_seed"])
TIE_SEED = int(P["topk_tie_seed"])
Q_PROMOTE = float(P["bfdr_strong"])
OUT = C.OUT / "sim_hub"
FITS = OUT / "fits"
METRICS = OUT / "metrics"
TIE_DECIMALS = 12            # as governance.TIE_DECIMALS
NEAR_BEST_TOL = 1e-3         # as fit_primary.NEAR_BEST_TOL
T_INFLATION_FACTOR = 10.0    # mode diagnostic: w_T > factor x true prevalence = inflated class T

# ----------------------------------------------------------------------------------------
# Design (fixed 2026-10-08 before any simulation fit; see module docstring)
# ----------------------------------------------------------------------------------------
SIM_DESIGN = {
    "fixed_on": "2026-10-08",
    "timing": ("E6 planned before held-out evaluation (REVISION_PLAN_R2 2.3); numerical design fixed "
               "after the results checkpoint (plan 8) and before any simulation fit; no held-out data used"),
    "n_tf": 400,
    "n_target": 3000,
    "log_popularity_sd": 1.2,
    "p0": 0.01,
    "tf_degree_sd": 0.5,
    "target_propensity_sd": 0.5,
    "p_max": 0.5,
    "bio_hub_exponent": 0.3,
    "copy_base": 0.002,
    "copy_hub": 0.004,
    "lit_coverage": [0.60, 0.45, 0.35, 0.30, 0.25, 0.20, 0.15, 0.12, 0.10, 0.08, 0.06, 0.05],
    "lit_coverage_slope": 1.0,
    "lit_sensitivity": [0.45, 0.30, 0.40, 0.25, 0.35, 0.20, 0.30, 0.15, 0.25, 0.10, 0.20, 0.15],
    "lit_sensitivity_slope": 0.5,
    "lit_fpr": [0.0020, 0.0010, 0.0030, 0.0015, 0.0040, 0.0020, 0.0050, 0.0030, 0.0020, 0.0040,
                0.0060, 0.0030],
    "lit_copy": [0.85, 0.50, 0.25, 0.20, 0.15, 0.10, 0.10, 0.08, 0.06, 0.05, 0.05, 0.03],
    "motif_tf_has_motif": 0.6,
    "motif_coverage_given_motif": 0.9,
    "motif_sensitivity": [0.10, 0.09],
    "motif_fpr": [0.020, 0.015],
    "motif_burst_report": [0.65, 0.75],
    "motif_burst_mean": 0.04,
    "motif_burst_log_mu": -0.5,
    "motif_burst_log_sd": 1.0,
    "motif_burst_max": 0.3,
    "coexp_coverage": 0.7,
    "coexp_sensitivity": 0.12,
    "coexp_fpr": 0.07,
    "mask": "curated_conditional (MOTIF/COEXP observed only on LIT-reported edges of covered TFs)",
    "scenarios": ["null", "bio_hub", "copying"],
    "n_reps": 20,
    "models": ["em", "dependence", "family_regularized", "family_hb"],
    "obs_models": ["family_hb_obs", "em_obs"],
    "n_restarts": {"family_hb": 30, "family_hb_obs": 20, "default": 10},
    "baselines": ["union", "majority", "vote_fraction", "fisher", "stouffer"],
    "promotion_q": Q_PROMOTE,
    "size_match_reference": "family_hb",
    "set_rules": ["bfdr05", "size_matched", "size_oracle"],
    "n_deciles": 10,
    "n_lit_quintiles": 5,
}
SCENARIOS = tuple(SIM_DESIGN["scenarios"])
# Scenario labels written to CSV tables.  The internal name "null" (kept: it seeds the data and
# names the fit directories) is parsed as a missing value by pandas.read_csv, so tables use
# "null_independent" (truth independent of popularity).
TABLE_SCENARIO_LABEL = {"null": "null_independent"}
N_LIT = len(SIM_DESIGN["lit_coverage"])
N_MOTIF = len(SIM_DESIGN["motif_sensitivity"])
ANNOTATORS = [f"LIT{j + 1:02d}" for j in range(N_LIT)] + [f"MOTIF{j + 1}" for j in range(N_MOTIF)] + ["COEXP1"]
FAMILY = ["LIT"] * N_LIT + ["MOTIF"] * N_MOTIF + ["COEXP"]
POST_HOC_MODELS = set(SIM_DESIGN["obs_models"])


def log(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    try:
        C.ensure_dir(OUT)
        with open(OUT / "run.log", "a") as fh:
            fh.write(line + "\n")
    except OSError:  # pragma: no cover
        pass


def seed_seq(*parts) -> np.random.SeedSequence:
    """Deterministic seed sequence from the prespecified seed and a context label."""
    return np.random.SeedSequence([SEED, zlib.crc32("\x1f".join(map(str, parts)).encode())])


def int_seed(*parts) -> int:
    return int(seed_seq(*parts).generate_state(1)[0] % (2 ** 31 - 1))


STATUS_BASIS = {
    "core": "design_fixed_after_checkpoint",
    "obs": "model_designed_after_heldout",
    "empirical": "comparison_with_heldout_table",
}


def analysis_status(method: str | None = None) -> str:
    """``post_hoc`` for every row: the numerical design was fixed after the results checkpoint
    (see the module docstring, 'Analysis status')."""
    return "post_hoc"


def status_basis(method: str | None = None) -> str:
    """Why a row is post hoc: the informative-observation models were designed after the
    held-out results; everything else only had its numerical design fixed after them."""
    return STATUS_BASIS["obs" if method in POST_HOC_MODELS else "core"]


# ----------------------------------------------------------------------------------------
# Data container and pattern compression
# ----------------------------------------------------------------------------------------
@dataclass
class SimData:
    scenario: str
    rep: int
    data: object                 # lcm.PatternData, plain codes (MAR models, baselines)
    edge_pattern: np.ndarray     # (N,) int32, edge -> pattern row of ``data``
    data_elig: object            # lcm.PatternData, eligibility-coded (informative-observation models)
    edge_pattern_elig: np.ndarray  # (N,) int32, edge -> pattern row of ``data_elig``
    z: np.ndarray                # (N,) bool, true edge
    copied: np.ndarray           # (N,) bool, copied false positive (LIT burst)
    mburst: np.ndarray           # (N,) bool, promiscuous-motif burst non-edge
    log_u: np.ndarray            # (T,) log popularity
    decile: np.ndarray           # (T,) popularity decile 1..10
    lit_degree: np.ndarray       # (T,) number of targets with >= 1 LIT report
    lit_quintile: np.ndarray     # (T,) literature-degree quintile 1..5
    p_edge: np.ndarray           # (N,) true prior edge probability (oracle)
    c_tf: np.ndarray             # (T,) copy probability
    m_tf: np.ndarray             # (T,) motif-burst probability
    rates: dict                  # per-annotator per-TF reporting rates (oracle)
    checksum: str

    @property
    def n_tf(self) -> int:
        return len(self.log_u)

    @property
    def n_target(self) -> int:
        return len(self.edge_pattern) // len(self.log_u)

    @property
    def tf_of_edge(self) -> np.ndarray:
        return np.repeat(np.arange(self.n_tf, dtype=np.int32), self.n_target)


def compress(M: np.ndarray, lowest: int = -1):
    """(N, K) int8 codes in [lowest, 1] -> (unique rows R sorted by their base-b code,
    edge counts, edge -> row index), b = 2 - lowest."""
    K = M.shape[1]
    base = 2 - lowest
    code = np.zeros(M.shape[0], dtype=np.int64)
    for j in range(K):
        code += (M[:, j].astype(np.int64) - lowest) * (base ** j)
    uniq, inv, counts = np.unique(code, return_inverse=True, return_counts=True)
    R = np.empty((len(uniq), K), dtype=np.int8)
    rem = uniq.copy()
    for j in range(K):
        R[:, j] = (rem % base) + lowest
        rem //= base
    return R, counts.astype(np.float64), inv.reshape(-1).astype(np.int32)


def _quantile_groups(x: np.ndarray, n_groups: int) -> np.ndarray:
    """Equal-count groups 1..n_groups of x (ties broken by index order, deterministic)."""
    r = rankdata(x, method="ordinal") - 1
    return (r * n_groups // len(x)).astype(np.int16) + 1


# ----------------------------------------------------------------------------------------
# Generator
# ----------------------------------------------------------------------------------------
def scenario_tf_parameters(scenario: str, log_u: np.ndarray, eta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """TF-level true-edge rate p_t (before the gene propensity) and copy probability c_t."""
    d = SIM_DESIGN
    if scenario not in SCENARIOS:
        raise ValueError(f"unknown scenario {scenario}")
    u = np.exp(log_u)
    p_t = d["p0"] * np.exp(eta)
    if scenario == "bio_hub":
        s = u ** d["bio_hub_exponent"]
        p_t = p_t * s / s.mean()
    c_t = np.full(len(log_u), d["copy_base"])
    if scenario == "copying":
        c_t = c_t + d["copy_hub"] * u / u.mean()
    return p_t, np.minimum(c_t, 0.5)


def simulate_dataset(scenario: str, rep: int, n_tf: int | None = None, n_target: int | None = None) -> SimData:
    """Draw one replicate (deterministic in (scenario, rep, n_tf, n_target))."""
    from r2 import lcm
    d = SIM_DESIGN
    T = int(n_tf or d["n_tf"])
    G = int(n_target or d["n_target"])
    rng = np.random.default_rng(seed_seq("sim_hub_bias", "data", scenario, rep, T, G))
    # popularity, TF degree heterogeneity, gene propensity (the same draw order in every scenario)
    log_u = rng.normal(0.0, d["log_popularity_sd"], T)
    sd_t = d["tf_degree_sd"]
    eta = rng.normal(-sd_t ** 2 / 2, sd_t, T)
    sd_g = d["target_propensity_sd"]
    h = np.exp(rng.normal(-sd_g ** 2 / 2, sd_g, G))
    p_t, c_t = scenario_tf_parameters(scenario, log_u, eta)
    m_t = np.minimum(d["motif_burst_mean"] * np.exp(rng.normal(d["motif_burst_log_mu"], d["motif_burst_log_sd"], T)),
                     d["motif_burst_max"])
    tf = np.repeat(np.arange(T), G)
    gene = np.tile(np.arange(G), T)
    p_edge = np.minimum(p_t[tf] * h[gene], d["p_max"])
    N = T * G
    z = rng.random(N) < p_edge
    copied = (~z) & (rng.random(N) < c_t[tf])
    mburst = (~z) & (rng.random(N) < m_t[tf])
    K = N_LIT + N_MOTIF + 1
    M = np.full((N, K), -2, dtype=np.int8)       # eligibility-coded evidence
    rates = {}
    # ---- LIT
    cov_p = expit(logit(np.asarray(d["lit_coverage"]))[:, None] + d["lit_coverage_slope"] * log_u[None, :])
    cov = rng.random((N_LIT, T)) < cov_p
    none = ~cov.any(0)
    if none.any():
        cov[np.argmax(cov_p[:, none], axis=0), np.nonzero(none)[0]] = True
    any_lit = np.zeros(N, dtype=bool)
    for j in range(N_LIT):
        a_jt = expit(logit(d["lit_sensitivity"][j]) + d["lit_sensitivity_slope"] * log_u)
        g_jt = expit(logit(d["lit_copy"][j]) + d["lit_sensitivity_slope"] * log_u)
        b_j = d["lit_fpr"][j]
        rate = np.where(z, a_jt[tf], np.where(copied, g_jt[tf], b_j))
        rep_j = (rng.random(N) < rate) & cov[j][tf]
        nodes = np.zeros(G, dtype=bool)
        nodes[gene[rep_j]] = True
        obs_j = cov[j][tf] & nodes[gene]
        # eligibility coding: -2 TF not covered, -1 covered but target not a node of j
        M[:, j] = np.where(obs_j, rep_j.astype(np.int8), np.where(cov[j][tf], -1, -2))
        any_lit |= rep_j
        rates[ANNOTATORS[j]] = dict(cov=cov[j], T=a_jt, C=g_jt, N=np.full(T, b_j), M=np.full(T, b_j))
    # ---- MOTIF (curated-conditional)
    has_motif = rng.random(T) < d["motif_tf_has_motif"]
    for k in range(N_MOTIF):
        j = N_LIT + k
        cov_k = has_motif & (rng.random(T) < d["motif_coverage_given_motif"])
        a, b, gm = d["motif_sensitivity"][k], d["motif_fpr"][k], d["motif_burst_report"][k]
        rate = np.where(z, a, np.where(mburst, gm, b))
        rep_k = rng.random(N) < rate
        obs_k = cov_k[tf] & any_lit
        M[:, j] = np.where(obs_k, rep_k.astype(np.int8), -2)   # observation conditioned on
        rates[ANNOTATORS[j]] = dict(cov=cov_k, T=np.full(T, a), C=np.full(T, b), N=np.full(T, b), M=np.full(T, gm))
    # ---- COEXP (curated-conditional)
    j = N_LIT + N_MOTIF
    cov_c = rng.random(T) < d["coexp_coverage"]
    a, b = d["coexp_sensitivity"], d["coexp_fpr"]
    rep_c = rng.random(N) < np.where(z, a, b)
    M[:, j] = np.where(cov_c[tf] & any_lit, rep_c.astype(np.int8), -2)
    rates[ANNOTATORS[j]] = dict(cov=cov_c, T=np.full(T, a), C=np.full(T, b), N=np.full(T, b), M=np.full(T, b))
    # ---- compression and TF strata
    Re, ne, inve = compress(M, lowest=-2)
    data_elig = lcm.PatternData(Re, ne, FAMILY, ANNOTATORS, eligibility=True)
    M[M < -1] = -1                                  # plain codes: not observed = -1
    R, n, inv = compress(M, lowest=-1)
    data = lcm.PatternData(R, n, FAMILY, ANNOTATORS)
    lit_reported = (M[:, :N_LIT] == 1).any(1)
    lit_degree = np.bincount(tf[lit_reported], minlength=T)
    h_ = hashlib.sha256()
    for arr in (R, n, inv, Re, ne, inve, z):
        h_.update(np.ascontiguousarray(arr).tobytes())
    return SimData(scenario=scenario, rep=int(rep), data=data, edge_pattern=inv, data_elig=data_elig,
                   edge_pattern_elig=inve, z=z, copied=copied,
                   mburst=mburst, log_u=log_u, decile=_quantile_groups(log_u, d["n_deciles"]),
                   lit_degree=lit_degree, lit_quintile=_quantile_groups(lit_degree, d["n_lit_quintiles"]),
                   p_edge=p_edge, c_tf=c_t, m_tf=m_t, rates=rates, checksum=h_.hexdigest()[:16])


# ----------------------------------------------------------------------------------------
# Oracle posterior (true generative emission model, observation mask taken as given)
# ----------------------------------------------------------------------------------------
ORACLE_CLASSES = ("T", "N", "C", "M", "CM")   # true; clean non-edge; copied; motif burst; both


def oracle_credibility(sim: SimData, lit_unobserved_as_nonreport: bool = False) -> np.ndarray:
    """Per-edge P(true | observed reports) under the true TF-level rates and priors.

    The default is the *ignorable-mask* oracle used as the reference method ``oracle``: like the
    MAR models it skips unobserved entries.  With ``lit_unobserved_as_nonreport`` an entry of a
    LIT annotator that covers the TF but does not have the target among its nodes (eligibility
    code -1) is counted as a non-report, which is what it is in this generator (the target-node
    rule only hides non-reports).  That is the posterior given everything the LIT annotators
    did and did not report (``oracle-check`` diagnostic)."""
    tf = sim.tf_of_edge
    R = sim.data.R[sim.edge_pattern].copy()              # (N, K) plain codes
    if lit_unobserved_as_nonreport:
        E = sim.data_elig.R[sim.edge_pattern_elig]
        lit = np.array([f == "LIT" for f in FAMILY])
        R[:, lit] = np.where(E[:, lit] == -1, 0, R[:, lit])
    p = sim.p_edge
    c = sim.c_tf[tf]
    m = sim.m_tf[tf]
    with np.errstate(divide="ignore"):
        lp = np.stack([np.log(p), np.log1p(-p) + np.log1p(-c) + np.log1p(-m),
                       np.log1p(-p) + np.log(c) + np.log1p(-m),
                       np.log1p(-p) + np.log1p(-c) + np.log(m),
                       np.log1p(-p) + np.log(c) + np.log(m)], axis=1)
    for j, ann in enumerate(ANNOTATORS):
        r = R[:, j]
        o = r >= 0
        if not o.any():
            continue
        rt = sim.rates[ann]
        # class -> per-TF rate (C and CM use the copy rate for LIT, the clean rate otherwise;
        # M and CM use the burst rate for MOTIF)
        per_class = {"T": rt["T"], "N": rt["N"], "C": rt["C"], "M": rt["M"],
                     "CM": rt["C"] if FAMILY[j] == "LIT" else rt["M"]}
        for s, name in enumerate(ORACLE_CLASSES):
            th = np.clip(per_class[name][tf[o]], 1e-12, 1 - 1e-12)
            lp[o, s] += np.where(r[o] == 1, np.log(th), np.log1p(-th))
    lm = np.logaddexp.reduce(lp, axis=1)
    return np.exp(lp[:, 0] - lm)


# ----------------------------------------------------------------------------------------
# Promoted sets
# ----------------------------------------------------------------------------------------
def tie_blocks(c: np.ndarray) -> tuple[np.ndarray, int]:
    """Block index (0 = highest) after rounding to TIE_DECIMALS (= governance.tie_blocks)."""
    v = np.round(np.asarray(c, dtype=np.float64), TIE_DECIMALS)
    uniq, inv = np.unique(-v, return_inverse=True)
    return inv.reshape(-1), len(uniq)


def bfdr_mask(c: np.ndarray, w: np.ndarray, q: float = Q_PROMOTE) -> np.ndarray:
    """Largest prefix of whole tie blocks (decreasing credibility) with Bayesian FDR
    sum w (1 - c) / sum w <= q (= governance.bfdr_levels(...) == 2 for q = q_strong)."""
    c = np.asarray(c, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    inv, nb = tie_blocks(c)
    bw = np.bincount(inv, weights=w, minlength=nb)
    bl = np.bincount(inv, weights=w * (1.0 - c), minlength=nb)
    cw, cl = np.cumsum(bw), np.cumsum(bl)
    ok = (cw > 0) & (cl <= q * cw * (1 + 1e-12) + 1e-15)
    n_blocks = int(np.argmin(ok)) if not ok.all() else nb
    return inv < n_blocks


def topk_mask(score: np.ndarray, k: int, seed: int) -> np.ndarray:
    """Exactly k edges with the highest score; ties broken uniformly at random (seeded)."""
    s = np.asarray(score, dtype=np.float64)
    k = int(min(max(k, 0), len(s)))
    out = np.zeros(len(s), dtype=bool)
    if k == 0:
        return out
    key = np.random.default_rng(seed).random(len(s))
    s = np.where(np.isnan(s), -np.inf, s)
    order = np.lexsort((key, -s))                 # primary: score desc; secondary: random key
    out[order[:k]] = True
    return out


# ----------------------------------------------------------------------------------------
# Ranking metrics
# ----------------------------------------------------------------------------------------
def auroc(score: np.ndarray, y: np.ndarray) -> float:
    """Mann-Whitney AUROC with midranks for ties (NaN scores rank last)."""
    s = np.where(np.isnan(score), -np.inf, np.asarray(score, dtype=np.float64))
    y = np.asarray(y, dtype=bool)
    n1, n0 = int(y.sum()), int((~y).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = rankdata(s)
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def auroc_within_group(score: np.ndarray, y: np.ndarray, group: np.ndarray) -> float:
    """Stratified Mann-Whitney AUROC: sum_g U_g / sum_g n1_g n0_g over groups (TFs) with both
    classes (= the n1 n0-weighted mean of the per-group AUROCs)."""
    s = np.where(np.isnan(score), -np.inf, np.asarray(score, dtype=np.float64))
    y = np.asarray(y, dtype=bool)
    g = np.asarray(group, dtype=np.int64)
    dense = rankdata(s, method="dense").astype(np.int64)
    key = g * (int(dense.max()) + 1) + dense
    r_all = rankdata(key)                                   # midranks within (group, score) order
    n_g = np.bincount(g)
    start = np.concatenate([[0], np.cumsum(n_g)[:-1]])    # groups occupy consecutive rank ranges
    r = r_all - start[g]
    n1 = np.bincount(g, weights=y.astype(float), minlength=len(n_g))
    n0 = n_g - n1
    u = np.bincount(g, weights=np.where(y, r, 0.0), minlength=len(n_g)) - n1 * (n1 + 1) / 2
    ok = (n1 > 0) & (n0 > 0)
    den = float(np.sum(n1[ok] * n0[ok]))
    return float(np.sum(u[ok]) / den) if den > 0 else float("nan")


def average_precision(score: np.ndarray, y: np.ndarray) -> float:
    """Step-function AP with whole tie blocks: sum_b (tp_b / P) * precision at the end of b."""
    s = np.where(np.isnan(score), -np.inf, np.asarray(score, dtype=np.float64))
    y = np.asarray(y, dtype=bool)
    npos = int(y.sum())
    if npos == 0:
        return float("nan")
    uniq, inv = np.unique(-s, return_inverse=True)
    inv = inv.reshape(-1)
    nb = np.bincount(inv, minlength=len(uniq)).astype(float)
    tb = np.bincount(inv, weights=y.astype(float), minlength=len(uniq))
    prec = np.cumsum(tb) / np.cumsum(nb)
    return float(np.sum(tb / npos * prec))


# ----------------------------------------------------------------------------------------
# Fitting
# ----------------------------------------------------------------------------------------
def n_restarts_for(model: str) -> int:
    r = SIM_DESIGN["n_restarts"]
    return int(r.get(model, r["default"]))


def fit_path(scenario: str, rep: int, model: str) -> Path:
    return FITS / scenario / f"rep{rep:02d}" / model


def model_available(model: str) -> bool:
    from r2 import lcm
    try:
        lcm.MODEL_SPECS[model]
    except KeyError:
        return False
    return True


def is_obs_model(model: str) -> bool:
    from r2 import lcm
    return bool(lcm.MODEL_SPECS[model].get("obs_families"))


def mar_class_posteriors(data, w: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """(P, S) P(z = s | pattern) of a MAR model (= fit_primary.class_posteriors)."""
    obs = (data.R >= 0).astype(float)
    pos = (data.R == 1).astype(float)
    th = np.clip(np.asarray(theta, dtype=np.float64), 1e-12, 1 - 1e-12)
    lj = np.log(np.asarray(w, dtype=np.float64))[None, :] + (obs * pos) @ np.log(th) + (obs * (1 - pos)) @ np.log1p(-th)
    return np.exp(lj - np.logaddexp.reduce(lj, axis=1, keepdims=True))


def class_posteriors_at(params: dict, data, model: str) -> np.ndarray:
    """Class posteriors at fitted parameters; informative-observation models use
    ``lcm.class_posteriors`` (observation and reports), MAR models the MAR formula."""
    from r2 import lcm
    if is_obs_model(model):
        return np.asarray(lcm.class_posteriors(params, data), dtype=np.float64)
    return mar_class_posteriors(data, params["w"], params["theta"])


def lcm_digest() -> str:
    """sha256 (first 16 hex) of src/r2/lcm.py, recorded with every fit (the module is shared)."""
    return hashlib.sha256((Path(__file__).resolve().parent / "lcm.py").read_bytes()).hexdigest()[:16]


# settings of lcm.fit_map (its defaults), needed to re-run one of its restarts
FIT_MAP_DEFAULTS = dict(init_radius=2.0, gtol=1e-6, maxiter=20000, ftol=1e-15, maxcor=30)


def refit_single_restart(data, model: str, seed: int, n_restarts: int, restart: int) -> dict:
    """Re-run restart ``restart`` of ``lcm.fit_map(data, model, n_restarts, seed)`` (same PRNG
    key, start and optimiser settings) and return its constrained parameters, credibility, log
    posterior and log-likelihood.  Needed because ``fit_map`` returns the parameters of one
    restart only (see ``robust_fit_map``)."""
    import jax
    import jax.numpy as jnp
    import numpyro
    from jax.flatten_util import ravel_pytree
    from numpyro.infer.util import initialize_model
    from scipy.optimize import minimize
    from r2 import lcm
    d = FIT_MAP_DEFAULTS
    mdl = lcm.build_model(data, lcm.MODEL_SPECS[model], None)
    rng = jax.random.PRNGKey(seed)
    init = initialize_model(rng, mdl, init_strategy=numpyro.infer.init_to_uniform(radius=d["init_radius"]))
    flat0, unravel = ravel_pytree(init.param_info.z)
    pot = jax.jit(lambda x: init.potential_fn(unravel(x)))
    grad = jax.jit(jax.grad(lambda x: init.potential_fn(unravel(x))))
    key = jax.random.split(rng, n_restarts)[restart]
    x0 = np.asarray(jax.random.uniform(key, flat0.shape, minval=-d["init_radius"], maxval=d["init_radius"]))
    res = minimize(lambda x: float(pot(jnp.asarray(x))), x0, jac=lambda x: np.asarray(grad(jnp.asarray(x))),
                   method="L-BFGS-B", options=dict(maxiter=d["maxiter"], gtol=d["gtol"], ftol=d["ftol"],
                                                   maxcor=d["maxcor"]))
    sample = {k: np.asarray(v) for k, v in init.postprocess_fn(unravel(jnp.asarray(res.x))).items()}
    return dict(params=sample, cred=np.asarray(lcm.credibility_at(sample, data), dtype=np.float64),
                log_post=-float(res.fun), loglik=float(lcm.heldout_loglik(sample, data)))


def robust_fit_map(data, model: str, n_restarts: int, seed: int) -> dict:
    """``lcm.fit_map`` with failed restarts excluded.

    A restart whose optimiser ends at a non-finite log posterior (observed once: an ABNORMAL
    L-BFGS termination after 19 iterations of a family_hb_obs start) is a failed start.
    ``lcm.fit_map`` selects its best restart with ``argmax``, which returns the failed one when
    a NaN is present, and it keeps the parameters of that restart only.  Here the best *finite*
    restart is selected; its parameters are recovered by re-running exactly that restart
    (``refit_single_restart``) and checked against the stored log posterior and credibility.
    Without failed restarts the result is ``lcm.fit_map``'s, unchanged."""
    from r2 import lcm
    fit = lcm.fit_map(data, model, n_restarts=n_restarts, seed=seed)
    lp = np.array([r["log_post"] for r in fit.restarts], dtype=np.float64)
    ok = np.isfinite(lp)
    if not ok.any():
        raise RuntimeError(f"{model}: every restart failed (non-finite log posterior)")
    out = dict(class_names=list(fit.class_names), restarts=fit.restarts, cred_restarts=fit.cred_restarts,
               n_failed_restarts=int((~ok).sum()), recovered_restart=None)
    if ok.all():
        out.update(params={k: np.asarray(v) for k, v in fit.best.items()}, cred=np.asarray(fit.cred),
                   log_post=float(fit.log_post), loglik=float(fit.loglik))
    else:
        ib = int(np.nanargmax(lp))
        one = refit_single_restart(data, model, seed, n_restarts, ib)
        dlp = abs(one["log_post"] - lp[ib])
        dc = float(np.max(np.abs(one["cred"] - np.asarray(fit.cred_restarts[ib], dtype=np.float64))))
        if dlp > 1e-6 * max(1.0, abs(lp[ib])) or dc > 1e-6:
            raise RuntimeError(f"{model}: re-run of restart {ib} does not reproduce it (dlp={dlp:.3g}, dcred={dc:.3g})")
        out.update(params=one["params"], cred=one["cred"], log_post=float(lp[ib]), loglik=one["loglik"],
                   recovered_restart=ib)
    top = ok & (lp >= np.nanmax(lp) - NEAR_BEST_TOL)
    out["summary"] = dict(model=model, n_restarts=int(len(lp)), n_failed_restarts=out["n_failed_restarts"],
                          best_log_post=float(np.nanmax(lp)), n_within_1e3_of_best=int(top.sum()),
                          n_converged=int(sum(bool(r["converged"]) for r in fit.restarts)),
                          log_post_range=float(np.nanmax(lp) - np.nanmin(lp)), loglik=out["loglik"],
                          max_abs_cred_diff_top=float(np.max(np.abs(np.asarray(fit.cred_restarts)[top]
                                                                    - out["cred"][None, :]))),
                          runtime_s=float(fit.runtime_s))
    return out


def fit_one(sim: SimData, model: str, skip_existing: bool = False) -> dict | None:
    from r2 import lcm
    out = fit_path(sim.scenario, sim.rep, model)
    if skip_existing and (out / "meta.json").exists():
        meta = json.loads((out / "meta.json").read_text())
        if meta.get("checksum") == sim.checksum and meta.get("n_restarts") == n_restarts_for(model):
            return None
    obs_model = is_obs_model(model)
    data = sim.data_elig if obs_model else sim.data
    seed = int_seed("fit", sim.scenario, sim.rep, model)
    nr = n_restarts_for(model)
    t0 = time.time()
    fit = robust_fit_map(data, model, nr, seed)
    runtime = time.time() - t0
    params = {k: np.asarray(v) for k, v in fit["params"].items() if k != "cred"}
    post = class_posteriors_at(params, data, model)
    post_check = float(np.max(np.abs(post[:, 0] - fit["cred"])))
    C.ensure_dir(out)
    (out / "meta.json").unlink(missing_ok=True)       # meta.json is written last and marks a complete fit
    np.save(out / "cred.npy", np.asarray(fit["cred"], dtype=np.float64))
    np.save(out / "cred_restarts.npy", np.asarray(fit["cred_restarts"], dtype=np.float32))
    np.save(out / "class_post.npy", post.astype(np.float32))
    np.savez(out / "params.npz", **params)
    pd.DataFrame(fit["restarts"]).to_csv(out / "restarts.csv", index=False)
    s = fit["summary"]
    meta = dict(scenario=sim.scenario, rep=sim.rep, model=model, checksum=sim.checksum,
                representation="eligibility" if obs_model else "plain", n_patterns=int(data.P),
                class_names=list(fit["class_names"]), n_restarts=nr, seed=seed, best_log_post=fit["log_post"],
                loglik=fit["loglik"], summary=s, runtime_s=runtime,
                n_failed_restarts=fit["n_failed_restarts"], recovered_restart=fit["recovered_restart"],
                class_post_vs_cred_max_abs_diff=post_check, analysis_status=analysis_status(model),
                status_basis=status_basis(model),
                lcm_sha256=lcm_digest(),
                spec={k: (list(v) if isinstance(v, tuple) else v) for k, v in lcm.MODEL_SPECS[model].items()})
    C.write_json(meta, out / "meta.json")
    log(f"[fit] {sim.scenario} rep{sim.rep:02d} {model}: P={data.P} log_post={fit['log_post']:.3f} "
        f"near-best={s['n_within_1e3_of_best']}/{nr} w_T={params['w'][0]:.4f} {runtime:.1f}s "
        f"post-check={post_check:.1e}"
        + (f" failed-restarts={fit['n_failed_restarts']} (best finite restart {fit['recovered_restart']} re-run)"
           if fit["n_failed_restarts"] else ""))
    return meta


def parse_reps(spec: str) -> list[int]:
    out = []
    for part in str(spec).split(","):
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        elif part:
            out.append(int(part))
    return out


def shard_items(items: list, shard: str | None) -> list:
    if not shard:
        return items
    i, n = (int(x) for x in shard.split("/"))
    return [it for k, it in enumerate(items) if k % n == i]


# Realised coverage of the fits (final, 2026-10-13).  The design asks for 20 replicates per
# scenario for every model.  family_hb_obs takes 10-35 min per fit on the shared machine and was
# stopped after the replicates below (replicate-first order).  The stop was a compute decision
# (machine saturated by other jobs), but it was taken when the 11 fits finished by then were
# known to select the same inflated-T optimum, so the family_hb_obs rows are not a designed-size
# sample.  copying rep00 is absent because its first fit crashed on a failed restart (see
# ``robust_fit_map``) and the guarded refit was abandoned under load.  ``fit`` honours this
# coverage, so that ``all`` reproduces the reported tables; ``--full-design`` fits the rest.
REALISED_COVERAGE = {
    "family_hb_obs": {"null": [0, 1, 2, 3, 4], "bio_hub": [0, 1, 2, 3], "copying": [1, 2, 3]},
}


def in_realised_coverage(scenario: str, rep: int, model: str, full_design: bool = False) -> bool:
    """False for a (scenario, replicate, model) outside ``REALISED_COVERAGE`` unless
    ``full_design``; models without an entry are fitted on every replicate."""
    cov = REALISED_COVERAGE.get(model)
    return bool(full_design or cov is None or rep in cov.get(scenario, ()))


def run_fits(scenarios, reps, models, shard=None, skip_existing=True, full_design=False) -> None:
    jobs = shard_items([(s, r) for s in scenarios for r in reps], shard)
    for s, r in jobs:
        todo = [m for m in models if in_realised_coverage(s, r, m, full_design)]
        if not todo:
            continue
        sim = simulate_dataset(s, r)
        for m in todo:
            if not model_available(m):
                log(f"[fit] model {m} not in lcm.MODEL_SPECS; skipped")
                continue
            fit_one(sim, m, skip_existing=skip_existing)


# ----------------------------------------------------------------------------------------
# Per-replicate evaluation against the truth
# ----------------------------------------------------------------------------------------
def lcm_models() -> list:
    return list(SIM_DESIGN["models"]) + list(SIM_DESIGN["obs_models"])


def load_fit(sim: SimData, model: str) -> dict | None:
    """Fitted per-pattern credibility etc. of one model, or None if absent / stale."""
    d = fit_path(sim.scenario, sim.rep, model)
    if not (d / "meta.json").exists():
        return None
    meta = json.loads((d / "meta.json").read_text())
    if meta.get("checksum") != sim.checksum:
        raise RuntimeError(f"stale fit {d}: checksum {meta.get('checksum')} != {sim.checksum}")
    return dict(meta=meta, cred=np.load(d / "cred.npy"), cred_restarts=np.load(d / "cred_restarts.npy"),
                class_post=np.load(d / "class_post.npy"), restarts=pd.read_csv(d / "restarts.csv"),
                params=dict(np.load(d / "params.npz")))


def collect_scores(sim: SimData) -> tuple[dict, dict]:
    """Edge-level scores of every method -> (scores, info).  info[m] = dict(kind, cred, fit)."""
    from r2 import lcm
    scores, info = {}, {}
    for m in lcm_models():
        f = load_fit(sim, m)
        if f is None:
            continue
        ep = sim.edge_pattern_elig if f["meta"]["representation"] == "eligibility" else sim.edge_pattern
        scores[m] = f["cred"][ep]
        info[m] = dict(kind="latent_class", cred=True, fit=f, edge_pattern=ep)
    base = lcm.baseline_scores(sim.data)
    for b in SIM_DESIGN["baselines"]:
        scores[b] = np.asarray(base[b], dtype=np.float64)[sim.edge_pattern]
        info[b] = dict(kind="baseline", cred=False)
    scores["oracle"] = oracle_credibility(sim)
    info["oracle"] = dict(kind="oracle", cred=True)
    return scores, info


def promoted_sets(sim: SimData, scores: dict, info: dict) -> dict:
    """{(method, set_rule): edge mask}.  BFDR sets for credibility methods (whole tie blocks;
    latent-class models at pattern level, i.e. whole patterns); size-matched top sets for all."""
    sets = {}
    for m, sc in scores.items():
        if not info[m]["cred"]:
            continue
        if info[m]["kind"] == "latent_class":
            f = info[m]["fit"]
            ep = info[m]["edge_pattern"]
            n_pat = np.bincount(ep, minlength=len(f["cred"])).astype(float)
            sets[(m, "bfdr05")] = bfdr_mask(f["cred"], n_pat)[ep]
        else:
            sets[(m, "bfdr05")] = bfdr_mask(sc, np.ones(len(sc)))
    ref = SIM_DESIGN["size_match_reference"]
    k = int(sets[(ref, "bfdr05")].sum()) if (ref, "bfdr05") in sets else 0
    k_or = int(sets[("oracle", "bfdr05")].sum())
    for m, sc in scores.items():
        sets[(m, "size_matched")] = topk_mask(sc, k, int_seed("tie", TIE_SEED, sim.scenario, sim.rep, m))
        sets[(m, "size_oracle")] = topk_mask(sc, k_or, int_seed("tie_oracle", TIE_SEED, sim.scenario, sim.rep, m))
    return sets


def _ratio(a: float, b: float) -> float:
    return float(a / b) if b > 0 else float("nan")


def evaluate_replicate(sim: SimData) -> dict:
    """All per-replicate tables (lists of dict rows)."""
    scores, info = collect_scores(sim)
    sets = promoted_sets(sim, scores, info)
    z = sim.z
    tf = sim.tf_of_edge
    plain = sim.data.R[sim.edge_pattern]
    observed = (plain >= 0).any(1)
    reported = (plain == 1).any(1)
    strata = {"popularity_decile": sim.decile[tf], "lit_degree_quintile": sim.lit_quintile[tf]}
    top = {"popularity_decile": SIM_DESIGN["n_deciles"], "lit_degree_quintile": SIM_DESIGN["n_lit_quintiles"]}
    n_true = int(z.sum())
    n_true_rep = int((z & reported).sum())
    key = dict(scenario=sim.scenario, rep=sim.rep)
    summary, strata_rows, calib_rows = [], [], []
    score_rows = {}
    for m, sc in scores.items():
        score_rows[m] = dict(auroc_all=auroc(sc, z), auroc_observed=auroc(sc[observed], z[observed]),
                             auroc_reported=auroc(sc[reported], z[reported]),
                             auroc_within_tf=auroc_within_group(sc, z, tf),
                             ap_lift=_ratio(average_precision(sc, z), z.mean()))
        if info[m]["cred"]:
            for sname, sv in strata.items():
                for g in range(1, top[sname] + 1):
                    e = sv == g
                    er = e & reported
                    calib_rows.append(dict(**key, method=m, analysis_status=analysis_status(m),
                                           stratifier=sname, stratum=g, n_edges=int(e.sum()),
                                           true_rate=float(z[e].mean()), mean_cred=float(sc[e].mean()),
                                           n_reported=int(er.sum()),
                                           true_rate_reported=float(z[er].mean()) if er.any() else np.nan,
                                           mean_cred_reported=float(sc[er].mean()) if er.any() else np.nan))
    for (m, rule), S_ in sets.items():
        n_set = int(S_.sum())
        tp = int((S_ & z).sum())
        row = dict(**key, method=m, kind=info[m]["kind"], set_rule=rule, analysis_status=analysis_status(m),
                   set_size=n_set, n_true_in_set=tp, precision=_ratio(tp, n_set), recall=_ratio(tp, n_true),
                   recall_of_reported_truth=_ratio(int((S_ & z & reported).sum()), n_true_rep),
                   fdr=1 - _ratio(tp, n_set) if n_set else np.nan,
                   bfdr_estimated=float(np.mean(1 - scores[m][S_])) if (info[m]["cred"] and n_set) else np.nan,
                   n_copied_in_set=int((S_ & sim.copied).sum()), n_mburst_in_set=int((S_ & sim.mburst).sum()))
        for sname, sv in strata.items():
            t = sv == top[sname]
            share_set = _ratio(int((S_ & t).sum()), n_set)
            share_true = _ratio(int((z & t).sum()), n_true)
            share_true_rep = _ratio(int((z & reported & t).sum()), n_true_rep)
            short = "top_decile" if sname == "popularity_decile" else "lit_q5"
            row[f"share_set_{short}"] = share_set
            row[f"share_true_{short}"] = share_true
            row[f"over_promotion_{short}"] = _ratio(share_set, share_true)
            row[f"over_promotion_vs_reported_truth_{short}"] = _ratio(share_set, share_true_rep)
            fp = S_ & ~z
            row[f"share_fp_{short}"] = _ratio(int((fp & t).sum()), int(fp.sum()))
            row[f"precision_{short}"] = _ratio(int((S_ & z & t).sum()), int((S_ & t).sum()))
            row[f"precision_not_{short}"] = _ratio(int((S_ & z & ~t).sum()), int((S_ & ~t).sum()))
            row[f"recall_{short}"] = _ratio(int((S_ & z & t).sum()), int((z & t).sum()))
            row[f"recall_not_{short}"] = _ratio(int((S_ & z & ~t).sum()), int((z & ~t).sum()))
            for g in range(1, top[sname] + 1):
                e = sv == g
                n_set_g = int((S_ & e).sum())
                tp_g = int((S_ & z & e).sum())
                n_true_g = int((z & e).sum())
                strata_rows.append(dict(**key, method=m, kind=info[m]["kind"], set_rule=rule,
                                        analysis_status=analysis_status(m), stratifier=sname, stratum=g,
                                        n_edges=int(e.sum()), n_true=n_true_g, n_set=n_set_g, n_set_true=tp_g,
                                        share_true=_ratio(n_true_g, n_true),
                                        share_true_reported=_ratio(int((z & reported & e).sum()), n_true_rep),
                                        share_set=_ratio(n_set_g, n_set),
                                        over_promotion=_ratio(_ratio(n_set_g, n_set), _ratio(n_true_g, n_true)),
                                        precision=_ratio(tp_g, n_set_g), recall=_ratio(tp_g, n_true_g),
                                        mean_cred_set=float(scores[m][S_ & e].mean())
                                        if (info[m]["cred"] and n_set_g) else np.nan))
        row.update(score_rows[m])
        summary.append(row)
    modes = [mode_diagnostics(sim, m, info[m], reported) for m in scores if info[m]["kind"] == "latent_class"]
    dataset = dict(**key, checksum=sim.checksum, n_edges=len(z), n_true=n_true, prevalence=float(z.mean()),
                   n_true_reported=n_true_rep, frac_true_reported=_ratio(n_true_rep, n_true),
                   n_copied=int(sim.copied.sum()), n_copied_reported=int((sim.copied & reported).sum()),
                   n_mburst=int(sim.mburst.sum()), frac_observed=float(observed.mean()),
                   n_reported=int(reported.sum()), n_patterns=int(sim.data.P), n_patterns_elig=int(sim.data_elig.P),
                   share_true_top_decile=_ratio(int((z & (strata["popularity_decile"] == 10)).sum()), n_true),
                   share_true_lit_q5=_ratio(int((z & (strata["lit_degree_quintile"] == 5)).sum()), n_true),
                   share_reported_top_decile=_ratio(int((reported & (strata["popularity_decile"] == 10)).sum()),
                                                    int(reported.sum())),
                   frac_true_reported_top_decile=_ratio(int((z & reported & (strata["popularity_decile"] == 10)).sum()),
                                                        int((z & (strata["popularity_decile"] == 10)).sum())),
                   frac_true_reported_bottom_decile=_ratio(int((z & reported & (strata["popularity_decile"] == 1)).sum()),
                                                           int((z & (strata["popularity_decile"] == 1)).sum())),
                   oracle_bfdr05_size=int(sets[("oracle", "bfdr05")].sum()),
                   size_matched_k=int(sets[(SIM_DESIGN["size_match_reference"], "size_matched")].sum())
                   if (SIM_DESIGN["size_match_reference"], "size_matched") in sets else 0)
    return dict(summary=summary, strata=strata_rows, calibration=calib_rows, modes=modes, datasets=[dataset],
                annotators=annotator_rows(sim, plain, info))


def annotator_rows(sim: SimData, plain: np.ndarray, info: dict) -> list:
    """Realised per-annotator rates among observed entries (comparable with fitted rates, which
    are conditional on observation) and the Family-HB fitted rates of the replicate."""
    z, cp, mb = sim.z, sim.copied, sim.mburst
    clean = ~z & ~cp & ~mb
    th = names = None
    if "family_hb" in info:
        th = info["family_hb"]["fit"]["params"]["theta"]
        names = info["family_hb"]["fit"]["meta"]["class_names"]
    rows = []
    for j, ann in enumerate(ANNOTATORS):
        o = plain[:, j] >= 0
        r = plain[:, j] == 1
        burst = cp if FAMILY[j] == "LIT" else (mb if FAMILY[j] == "MOTIF" else np.zeros_like(z))
        row = dict(scenario=sim.scenario, rep=sim.rep, annotator=ann, family=FAMILY[j], n_observed=int(o.sum()),
                   n_reported=int(r.sum()), tpr_observed=_ratio(int((r & z).sum()), int((o & z).sum())),
                   fpr_observed=_ratio(int((r & clean).sum()), int((o & clean).sum())),
                   burst_rate_observed=_ratio(int((r & burst).sum()), int((o & burst).sum())))
        if th is not None:
            row.update(fit_alpha=float(th[j, 0]), fit_beta=float(th[j, 1]))
            cn = f"B_{FAMILY[j]}"
            row["fit_gamma"] = float(th[j, names.index(cn)]) if cn in names else np.nan
        rows.append(row)
    return rows


def mode_diagnostics(sim: SimData, model: str, inf: dict, reported: np.ndarray) -> dict:
    """Restart/optimum structure of one latent-class fit and its label alignment with the truth.

    Alignment is measured on reported edges (where T and the other classes compete):
    AUROC(P(T)) vs AUROC(P(s)) for every other class s at the best optimum (label_swapped if
    another class tracks the truth better; the burst class with the highest AUROC is reported
    separately), and AUROC(P(T)) at every restart (is the best log-posterior optimum also the
    best-aligned one?).  ``t_class_inflated``: w_T > T_INFLATION_FACTOR x true prevalence."""
    f = inf["fit"]
    ep = inf["edge_pattern"]
    z = sim.z
    lp_all = f["restarts"]["log_post"].to_numpy(dtype=np.float64)
    finite = np.flatnonzero(np.isfinite(lp_all))          # failed restarts (non-finite) are ignored
    lp = lp_all[finite]
    order = np.argsort(-lp, kind="mergesort")
    groups, cur = [], [order[0]]
    for i in order[1:]:
        if lp[cur[0]] - lp[i] <= NEAR_BEST_TOL:
            cur.append(i)
        else:
            groups.append(cur)
            cur = [i]
    groups.append(cur)
    idx = np.flatnonzero(reported)
    zr = z[idx]
    epr = ep[idx]
    au_r = np.array([auroc(f["cred_restarts"][r].astype(np.float64)[epr], zr) for r in finite])
    names = f["meta"]["class_names"]
    post = f["class_post"].astype(np.float64)
    au_T = auroc(post[epr, 0], zr)
    au_o = {c: auroc(post[epr, s], zr) for s, c in enumerate(names) if c != "T"}
    au_b = {c: a for c, a in au_o.items() if c != "N"}
    b_best = max(au_b, key=au_b.get) if au_b else None
    o_best = max(au_o, key=au_o.get)
    best = groups[0][0]
    opt_au = [float(np.max(au_r[g])) for g in groups]
    w = f["params"]["w"]
    cred_e = f["cred"][ep]
    return dict(scenario=sim.scenario, rep=sim.rep, model=model, analysis_status=analysis_status(model),
                n_restarts=len(lp_all), n_failed_restarts=int(len(lp_all) - len(lp)),
                n_converged=int(f["restarts"]["converged"].sum()),
                n_at_best=len(groups[0]), n_distinct_optima=len(groups),
                gap_to_next_optimum=float(lp[best] - lp[groups[1][0]]) if len(groups) > 1 else np.nan,
                class_names=" ".join(names), w_T=float(w[0]),
                w_bursts=" ".join(f"{c}={w[s]:.4f}" for s, c in enumerate(names) if c not in ("T", "N")),
                true_prevalence=float(z.mean()),
                expected_true_all=float(cred_e.sum()), n_true=int(z.sum()),
                expected_true_reported=float(cred_e[reported].sum()), n_true_reported=int(z[reported].sum()),
                auroc_T_reported_best=au_T, best_burst_class=b_best,
                auroc_burst_reported_best=float(au_b[b_best]) if b_best else np.nan,
                best_other_class=o_best, auroc_other_reported_best=float(au_o[o_best]),
                label_swapped_best=bool(au_o[o_best] > au_T),
                w_T_over_prevalence=float(w[0] / z.mean()),
                t_class_inflated=bool(w[0] > T_INFLATION_FACTOR * z.mean()),
                auroc_T_all_best=auroc(cred_e, z),
                auroc_T_reported_max_restart=float(np.nanmax(au_r)),
                n_restarts_better_aligned=int(np.sum(au_r > au_T + 0.01)),
                best_optimum_is_best_aligned=bool(au_T >= np.nanmax(opt_au) - 0.01),
                optima_auroc_T_reported=" ".join(f"{a:.3f}" for a in opt_au),
                optima_n_restarts=" ".join(str(len(g)) for g in groups),
                optima_gap=" ".join(f"{lp[best] - lp[g[0]]:.2f}" for g in groups))


def metrics_path(scenario: str, rep: int) -> Path:
    return METRICS / f"{scenario}_rep{rep:02d}.json"


def fitted_models(scenario: str, rep: int) -> list:
    """Latent-class models with a stored fit for this replicate (design order)."""
    return [m for m in lcm_models() if (fit_path(scenario, rep, m) / "meta.json").exists()]


def metrics_up_to_date(scenario: str, rep: int) -> bool:
    """True if the stored metrics of the replicate cover exactly the models fitted so far."""
    f = metrics_path(scenario, rep)
    if not f.exists():
        return False
    try:
        have = sorted(row["model"] for row in json.loads(f.read_text())["modes"])
    except (ValueError, KeyError):
        return False
    return have == sorted(fitted_models(scenario, rep))


def run_metrics(scenarios, reps, shard=None, only_stale: bool = False) -> None:
    jobs = shard_items([(s, r) for s in scenarios for r in reps], shard)
    for s, r in jobs:
        if only_stale and metrics_up_to_date(s, r):
            continue
        t0 = time.time()
        sim = simulate_dataset(s, r)
        res = evaluate_replicate(sim)
        C.write_json({k: v for k, v in res.items()}, metrics_path(s, r))
        log(f"[metrics] {s} rep{r:02d}: {len(res['summary'])} summary rows, "
            f"{len(res['modes'])} fits, {time.time() - t0:.1f}s")


# ----------------------------------------------------------------------------------------
# Oracle check: how much does the ignorable-mask assumption cost? (diagnostic added 2026-10-13)
# ----------------------------------------------------------------------------------------
ORACLE_CHECK = {
    "added_on": "2026-10-13",
    "why": ("the ignorable-mask oracle is over-confident at popular TFs (true rate / mean credibility of reported "
            "edges 0.75-0.87 in the top popularity decile).  The check compares it with the oracle that counts "
            "covered-but-unobserved LIT entries as non-reports, to see whether the target-node observation rule "
            "explains the miscalibration and how much of the hub concentration it accounts for."),
    "reps": "0-19",
}


def oracle_check_path(scenario: str, rep: int) -> Path:
    return OUT / "oracle_check" / f"{scenario}_rep{rep:02d}.json"


def oracle_check_one(sim: SimData) -> list:
    """Rows (one per oracle variant) with the BFDR-0.05 set, ranking and decile calibration."""
    z = sim.z
    tf = sim.tf_of_edge
    dec = sim.decile[tf]
    top = dec == SIM_DESIGN["n_deciles"]
    reported = (sim.data.R[sim.edge_pattern] == 1).any(1)
    rows = []
    for name, flag in (("oracle_ignorable_mask", False), ("oracle_lit_nonreports_counted", True)):
        c = oracle_credibility(sim, lit_unobserved_as_nonreport=flag)
        S_ = bfdr_mask(c, np.ones(len(c)))
        n_set, tp = int(S_.sum()), int((S_ & z).sum())
        row = dict(scenario=sim.scenario, rep=sim.rep, oracle=name, set_size=n_set, precision=_ratio(tp, n_set),
                   recall=_ratio(tp, int(z.sum())), auroc_all=auroc(c, z),
                   auroc_reported=auroc(c[reported], z[reported]),
                   over_promotion_top_decile=_ratio(_ratio(int((S_ & top).sum()), n_set),
                                                    _ratio(int((z & top).sum()), int(z.sum()))),
                   expected_over_true_all=float(c.sum() / z.sum()))
        for g in (1, 5, SIM_DESIGN["n_deciles"]):
            e = (dec == g) & reported
            row[f"ratio_reported_decile{g}"] = _ratio(float(z[e].mean()), float(c[e].mean()))
            ea = dec == g
            row[f"ratio_all_decile{g}"] = _ratio(float(z[ea].mean()), float(c[ea].mean()))
        rows.append(row)
    return rows


def run_oracle_check(scenarios, reps) -> None:
    for s in scenarios:
        for r in reps:
            out = oracle_check_path(s, r)
            if out.exists():
                continue
            t0 = time.time()
            sim = simulate_dataset(s, r)
            C.write_json(dict(checksum=sim.checksum, rows=oracle_check_one(sim)), out)
            log(f"[oracle-check] {s} rep{r:02d}: {time.time() - t0:.1f}s")


def oracle_check_table() -> pd.DataFrame:
    rows = []
    for f in sorted((OUT / "oracle_check").glob("*_rep*.json")):
        rows.extend(json.loads(f.read_text())["rows"])
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    out = []
    metrics = [c for c in df.columns if c not in ("scenario", "rep", "oracle")]
    for (sc, name), g in df.groupby(["scenario", "oracle"]):
        for met in metrics:
            out.append(dict(scenario=sc, oracle=name, metric=met, **mean_ci(g[met])))
    out = pd.DataFrame(out)
    out["definition"] = ("oracle_ignorable_mask = reference method 'oracle' (skips unobserved entries); "
                         "oracle_lit_nonreports_counted = covered-but-unobserved LIT entries counted as non-reports; "
                         "ratio = true-edge rate / mean credibility (1 = calibrated)")
    return out


# ----------------------------------------------------------------------------------------
# Restart check of the informative-observation fits (diagnostic added 2026-10-13)
# ----------------------------------------------------------------------------------------
RESTART_CHECK = {
    "added_on": "2026-10-13",
    "why": ("the first em_obs fits (10 restarts, as designed) selected an optimum with an inflated class T in "
            "most replicates but reached it from only 1-5 of 10 starts; in the other replicates all 10 starts "
            "ended in a truth-aligned optimum.  The check refits those replicates with 50 restarts (the number "
            "used on the real data) to see whether the truth-aligned optimum is the best one or a local one."),
    "models": ["em_obs", "family_hb_obs"],
    "n_restarts": 50,
}


def restart_check_path(scenario: str, rep: int, model: str) -> Path:
    return OUT / "restart_check" / scenario / f"rep{rep:02d}" / f"{model}.json"


def restart_check_one(sim: SimData, model: str, only_uninflated: bool = True) -> dict | None:
    """Refit ``model`` with RESTART_CHECK['n_restarts'] fresh random starts (seed stream
    'restart_check') and compare with the stored design fit.  The design fit is not replaced.
    With ``only_uninflated`` replicates whose design fit already has an inflated class T are
    skipped (the question is whether a truth-aligned design fit is the best optimum)."""
    from r2 import lcm
    stored = load_fit(sim, model)
    if stored is None:
        return None
    prev = float(sim.z.mean())
    w_design = float(stored["params"]["w"][0])
    if only_uninflated and w_design > T_INFLATION_FACTOR * prev:
        return None
    out = restart_check_path(sim.scenario, sim.rep, model)
    nr = int(RESTART_CHECK["n_restarts"])
    if out.exists():
        old = json.loads(out.read_text())
        if old.get("checksum") == sim.checksum and old.get("n_restarts_check") == nr:
            return old
    obs_model = is_obs_model(model)
    data = sim.data_elig if obs_model else sim.data
    ep = sim.edge_pattern_elig if obs_model else sim.edge_pattern
    seed = int_seed("restart_check", sim.scenario, sim.rep, model)
    t0 = time.time()
    fit = robust_fit_map(data, model, nr, seed)
    lp = np.array([r["log_post"] for r in fit["restarts"]], dtype=np.float64)
    lp = lp[np.isfinite(lp)]
    lp_design = float(stored["meta"]["best_log_post"])
    w_check = float(np.asarray(fit["params"]["w"])[0])
    plain = sim.data.R[sim.edge_pattern]
    reported = (plain == 1).any(1)
    res = dict(scenario=sim.scenario, rep=sim.rep, model=model, checksum=sim.checksum,
               n_restarts_design=int(stored["meta"]["n_restarts"]), log_post_design=lp_design,
               w_T_design=w_design, auroc_all_design=auroc(stored["cred"][ep], sim.z),
               auroc_reported_design=auroc(stored["cred"][ep][reported], sim.z[reported]),
               n_restarts_check=nr, seed_check=seed, log_post_check=float(lp.max()),
               log_post_gain=float(lp.max() - lp_design),
               higher_optimum_found=bool(lp.max() > lp_design + NEAR_BEST_TOL),
               n_check_at_check_best=int(np.sum(lp >= lp.max() - NEAR_BEST_TOL)),
               n_check_at_design_optimum=int(np.sum(np.abs(lp - lp_design) <= NEAR_BEST_TOL)),
               w_T_check=w_check, true_prevalence=prev,
               t_class_inflated_design=bool(w_design > T_INFLATION_FACTOR * prev),
               t_class_inflated_check=bool(w_check > T_INFLATION_FACTOR * prev),
               auroc_all_check=auroc(fit["cred"][ep], sim.z),
               auroc_reported_check=auroc(fit["cred"][ep][reported], sim.z[reported]),
               n_failed_restarts_check=fit["n_failed_restarts"],
               runtime_s=time.time() - t0, lcm_sha256=lcm_digest())
    C.write_json(res, out)
    log(f"[restart-check] {sim.scenario} rep{sim.rep:02d} {model}: design lp={lp_design:.3f} w_T={w_design:.4f} | "
        f"{nr} restarts lp={lp.max():.3f} (gain {res['log_post_gain']:.3f}) w_T={w_check:.4f} "
        f"at-best={res['n_check_at_check_best']}/{nr} at-design-optimum={res['n_check_at_design_optimum']}/{nr} "
        f"{res['runtime_s']:.0f}s")
    return res


def run_restart_check(scenarios, reps, models, shard=None, only_uninflated=True) -> None:
    jobs = shard_items([(s, r) for s in scenarios for r in reps], shard)
    for s, r in jobs:
        sim = None
        for m in models:
            if not model_available(m) or not (fit_path(s, r, m) / "meta.json").exists():
                continue
            if only_uninflated:
                # cheap pre-check on the stored class weight (prevalence is ~p0 in every scenario)
                w0 = float(np.load(fit_path(s, r, m) / "params.npz")["w"][0])
                if w0 > T_INFLATION_FACTOR * 2 * SIM_DESIGN["p0"]:
                    continue
            sim = sim or simulate_dataset(s, r)
            restart_check_one(sim, m, only_uninflated=only_uninflated)


def restart_check_table() -> pd.DataFrame:
    rows = [json.loads(f.read_text()) for f in sorted((OUT / "restart_check").glob("*/rep*/*.json"))]
    cols = ["scenario", "rep", "model", "n_restarts_design", "log_post_design", "w_T_design",
            "t_class_inflated_design", "auroc_all_design", "auroc_reported_design", "n_restarts_check",
            "log_post_check", "log_post_gain", "higher_optimum_found", "n_check_at_check_best",
            "n_check_at_design_optimum", "w_T_check", "t_class_inflated_check", "auroc_all_check",
            "auroc_reported_check", "true_prevalence", "runtime_s"]
    df = pd.DataFrame(rows, columns=cols + [c for c in (rows[0] if rows else {}) if c not in cols])
    if len(df):
        df = df[cols].sort_values(["model", "scenario", "rep"]).reset_index(drop=True)
        df["definition"] = ("design fit = stored fit with the designed number of restarts; check = "
                            f"{RESTART_CHECK['n_restarts']} fresh random restarts on the same data; only "
                            "replicates whose design fit has no inflated class T are checked")
    return df


def obs_signature_table() -> pd.DataFrame:
    """Parameter signature of the informative-observation fits: simulation (every stored
    replicate) next to the real-data fits of ``informative_observation.py`` (read-only, label-free
    parameters; no held-out data).  alpha / beta / eta are medians over the LIT annotators."""
    def signature(params: dict, family: list) -> dict:
        w = np.asarray(params["w"], dtype=np.float64)
        th = np.asarray(params["theta"], dtype=np.float64)
        lit = np.array([f == "LIT" for f in family])
        d = dict(w_T=float(w[0]), lit_alpha_median=float(np.median(th[lit, 0])),
                 lit_beta_median=float(np.median(th[lit, 1])),
                 lit_alpha_over_beta_median=float(np.median(th[lit, 0] / th[lit, 1])))
        if params.get("eta") is not None:
            eta = np.asarray(params["eta"], dtype=np.float64)
            ok = np.isfinite(eta[:, 0])
            d.update(eta_T_median=float(np.median(eta[ok, 0])), eta_nonT_median=float(np.median(eta[ok, 1])))
        return d

    rows = []
    for sc in SCENARIOS:
        for m in SIM_DESIGN["obs_models"] + ["family_hb", "em"]:
            sig = []
            for d in sorted((FITS / sc).glob(f"rep*/{m}")):
                # meta.json is written last and marks a complete fit; a directory without it is the
                # remnant of an interrupted or failed fit and must not enter any table
                if (d / "params.npz").exists() and (d / "meta.json").exists():
                    sig.append(signature(dict(np.load(d / "params.npz")), FAMILY))
            if not sig:
                continue
            g = pd.DataFrame(sig)
            infl = g["w_T"] > T_INFLATION_FACTOR * SIM_DESIGN["p0"]
            for label, sel in (("all", np.ones(len(g), dtype=bool)), ("inflated_T", infl.to_numpy()),
                               ("not_inflated", ~infl.to_numpy())):
                if not sel.any():
                    continue
                gg = g[sel]
                rows.append(dict(source="simulation", universe_or_scenario=sc, model=m, subset=label,
                                 n_fits=int(sel.sum()), w_T_min=float(gg["w_T"].min()),
                                 w_T_max=float(gg["w_T"].max()),
                                 **{c: float(gg[c].median()) for c in gg.columns}))
    for u in ("global",) + tuple(C.TISSUES):
        for variant, models in ((C.PRESPECIFIED["evidence_variant_primary"] + "__elig", SIM_DESIGN["obs_models"]),
                                (C.PRESPECIFIED["evidence_variant_primary"], ["family_hb", "em"])):
            for m in models:
                d = C.OUT / "fits" / u / variant / m
                if not (d / "params.npz").exists() or not (d / "meta.json").exists():
                    continue
                try:
                    from r2 import evidence
                    fam = list(evidence.load_meta(u)["variants"][C.PRESPECIFIED["evidence_variant_primary"]]["family"])
                except Exception:  # pragma: no cover - data layer not available
                    continue
                par = dict(np.load(d / "params.npz"))
                if len(fam) != np.asarray(par["theta"]).shape[0]:
                    continue
                sig = signature(par, fam)
                rows.append(dict(source="real_data", universe_or_scenario=u, model=m, subset="all", n_fits=1,
                                 w_T_min=sig["w_T"], w_T_max=sig["w_T"], **sig))
    out = pd.DataFrame(rows)
    if len(out):
        out["definition"] = ("w_T = weight of class T at the selected MAP optimum; LIT medians of sensitivity alpha, "
                             "clean false-positive rate beta, and observation rates eta given T / not T; "
                             f"inflated_T = w_T > {T_INFLATION_FACTOR:g} x p0 (simulation)")
    return out


# ----------------------------------------------------------------------------------------
# Manuscript tables (data/r2/tables/sim_hub_*.csv)
# ----------------------------------------------------------------------------------------
SUMMARY_SET_METRICS = (
    "set_size", "n_true_in_set", "precision", "recall", "recall_of_reported_truth", "fdr", "bfdr_estimated",
    "over_promotion_top_decile", "over_promotion_vs_reported_truth_top_decile", "share_set_top_decile",
    "share_true_top_decile", "share_fp_top_decile", "precision_top_decile", "precision_not_top_decile",
    "recall_top_decile", "recall_not_top_decile", "over_promotion_lit_q5",
    "over_promotion_vs_reported_truth_lit_q5", "share_set_lit_q5", "share_true_lit_q5", "precision_lit_q5",
    "precision_not_lit_q5", "n_copied_in_set")
SCORE_METRICS = ("auroc_all", "auroc_observed", "auroc_reported", "auroc_within_tf", "ap_lift")
CONTRAST_METRICS = ("over_promotion_top_decile", "over_promotion_lit_q5", "share_fp_top_decile",
                    "precision", "recall", "precision_top_decile", "precision_not_top_decile",
                    "recall_top_decile", "recall_not_top_decile")
METHOD_ORDER = ("oracle", "em", "dependence", "family_regularized", "family_hb", "family_hb_obs", "em_obs",
                "union", "majority", "vote_fraction", "fisher", "stouffer")


def mean_ci(x) -> dict:
    """Mean, SD, median and t-based 95% CI of the replicate mean (NaNs dropped)."""
    x = np.asarray(pd.to_numeric(pd.Series(x), errors="coerce"), dtype=np.float64)
    x = x[np.isfinite(x)]
    n = len(x)
    if n == 0:
        return dict(n_reps=0, mean=np.nan, sd=np.nan, median=np.nan, ci_lo=np.nan, ci_hi=np.nan)
    m = float(x.mean())
    sd = float(x.std(ddof=1)) if n > 1 else np.nan
    h = float(student_t.ppf(0.975, n - 1) * sd / math.sqrt(n)) if n > 1 else np.nan
    return dict(n_reps=n, mean=m, sd=sd, median=float(np.median(x)), ci_lo=m - h, ci_hi=m + h)


def _order(df: pd.DataFrame) -> pd.DataFrame:
    if "method" in df:
        df = df.assign(_o=df["method"].map({m: i for i, m in enumerate(METHOD_ORDER)}).fillna(99))
        keys = [k for k in ("scenario", "_o", "set_rule", "stratifier", "stratum", "metric") if k in df]
        df = df.sort_values(keys, kind="mergesort").drop(columns="_o")
    return df.reset_index(drop=True)


def load_metrics() -> dict:
    files = sorted(METRICS.glob("*_rep*.json"))
    if not files:
        raise FileNotFoundError(f"no metrics under {METRICS}; run the 'metrics' stage")
    out = {}
    for f in files:
        d = json.loads(f.read_text())
        for k, rows in d.items():
            out.setdefault(k, []).extend(rows)
    return {k: pd.DataFrame(v) for k, v in out.items()}


def summary_table(sm: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (sc, m, kind, rule, st), g in sm.groupby(["scenario", "method", "kind", "set_rule", "analysis_status"]):
        for met in SUMMARY_SET_METRICS:
            rows.append(dict(scenario=sc, method=m, kind=kind, set_rule=rule, metric=met, analysis_status=st,
                             **mean_ci(g[met])))
    sc_rows = sm[sm["set_rule"] == "bfdr05"].copy()
    sc_rows = pd.concat([sc_rows, sm[(sm["set_rule"] == "size_matched") & (sm["kind"] == "baseline")]])
    for (sc, m, kind, st), g in sc_rows.groupby(["scenario", "method", "kind", "analysis_status"]):
        for met in SCORE_METRICS:
            rows.append(dict(scenario=sc, method=m, kind=kind, set_rule="score", metric=met, analysis_status=st,
                             **mean_ci(g[met])))
    return _order(pd.DataFrame(rows))


def strata_table(st: pd.DataFrame) -> pd.DataFrame:
    rows = []
    keys = ["scenario", "method", "kind", "set_rule", "analysis_status", "stratifier", "stratum"]
    for k, g in st.groupby(keys):
        r = dict(zip(keys, k))
        r["n_reps"] = len(g)
        r["n_true_mean"] = float(g["n_true"].mean())
        r["share_true_mean"] = float(g["share_true"].mean())
        r["share_true_reported_mean"] = float(g["share_true_reported"].mean())
        for met in ("share_set", "over_promotion", "precision", "recall"):
            ci = mean_ci(g[met])
            r.update({f"{met}_mean": ci["mean"], f"{met}_sd": ci["sd"], f"{met}_lo": ci["ci_lo"],
                      f"{met}_hi": ci["ci_hi"]})
        r["mean_cred_set_mean"] = float(pd.to_numeric(g["mean_cred_set"], errors="coerce").mean())
        rows.append(r)
    return _order(pd.DataFrame(rows))


def calibration_table(cb: pd.DataFrame) -> pd.DataFrame:
    rows = []
    keys = ["scenario", "method", "analysis_status", "stratifier", "stratum"]
    cb = cb.assign(ratio_all=cb["true_rate"] / cb["mean_cred"],
                   ratio_reported=cb["true_rate_reported"] / cb["mean_cred_reported"])
    for k, g in cb.groupby(keys):
        r = dict(zip(keys, k))
        r["n_reps"] = len(g)
        for met in ("true_rate", "mean_cred", "true_rate_reported", "mean_cred_reported"):
            r[f"{met}_mean"] = float(pd.to_numeric(g[met], errors="coerce").mean())
        for met in ("ratio_all", "ratio_reported"):
            ci = mean_ci(g[met])
            r.update({f"{met}_mean": ci["mean"], f"{met}_lo": ci["ci_lo"], f"{met}_hi": ci["ci_hi"]})
        rows.append(r)
    out = pd.DataFrame(rows)
    out["definition"] = ("ratio = true-edge rate / mean credibility in the stratum (1 = calibrated; "
                         ">1 = credibility too low); 'reported' = edges with >= 1 report")
    return _order(out)


def modes_tables(md: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    md = md.copy()
    # metrics written before failed restarts were tracked have no such column (no restart failed there)
    md["n_failed_restarts"] = (pd.to_numeric(md["n_failed_restarts"], errors="coerce").fillna(0).astype(int)
                               if "n_failed_restarts" in md else 0)
    rows = []
    for (sc, m, st), g in md.groupby(["scenario", "model", "analysis_status"]):
        r = dict(scenario=sc, model=m, analysis_status=st, n_reps=len(g),
                 n_restarts=int(g["n_restarts"].iloc[0]),
                 frac_restarts_at_best_mean=float((g["n_at_best"] / g["n_restarts"]).mean()),
                 frac_restarts_at_best_min=float((g["n_at_best"] / g["n_restarts"]).min()),
                 n_distinct_optima_mean=float(g["n_distinct_optima"].mean()),
                 n_distinct_optima_max=int(g["n_distinct_optima"].max()),
                 gap_to_next_optimum_median=float(g["gap_to_next_optimum"].median()),
                 n_reps_label_swapped_best=int(g["label_swapped_best"].sum()),
                 n_reps_t_class_inflated=int(g["t_class_inflated"].sum()),
                 n_reps_swapped_or_inflated=int((g["label_swapped_best"].astype(bool)
                                                 | g["t_class_inflated"].astype(bool)).sum()),
                 w_T_median=float(g["w_T"].median()), w_T_min=float(g["w_T"].min()),
                 w_T_max=float(g["w_T"].max()),
                 auroc_T_all_best_mean=float(g["auroc_T_all_best"].mean()),
                 auroc_T_all_best_min=float(g["auroc_T_all_best"].min()),
                 n_reps_best_not_best_aligned=int((~g["best_optimum_is_best_aligned"].astype(bool)).sum()),
                 n_reps_any_restart_better_aligned=int((g["n_restarts_better_aligned"] > 0).sum()),
                 auroc_T_reported_best_mean=float(g["auroc_T_reported_best"].mean()),
                 auroc_T_reported_max_restart_mean=float(g["auroc_T_reported_max_restart"].mean()),
                 auroc_other_reported_best_mean=float(g["auroc_other_reported_best"].mean()),
                 best_other_class_mode=str(g["best_other_class"].mode().iloc[0]),
                 n_failed_restarts_total=int(g["n_failed_restarts"].sum()),
                 w_T_mean=float(g["w_T"].mean()), true_prevalence_mean=float(g["true_prevalence"].mean()),
                 expected_over_true_all_mean=float((g["expected_true_all"] / g["n_true"]).mean()),
                 expected_over_true_reported_mean=float((g["expected_true_reported"] / g["n_true_reported"]).mean()))
        rows.append(r)
    summ = pd.DataFrame(rows)
    summ = summ.assign(_o=summ["model"].map({m: i for i, m in enumerate(METHOD_ORDER)})).sort_values(
        ["scenario", "_o"]).drop(columns="_o").reset_index(drop=True)
    per = md.sort_values(["scenario", "model", "rep"]).reset_index(drop=True)
    return summ, per


BY_MODE_METRICS = ("set_size", "precision", "recall", "fdr", "bfdr_estimated", "over_promotion_top_decile",
                   "over_promotion_lit_q5", "auroc_all", "auroc_reported", "auroc_within_tf")


def by_mode_table(sm: pd.DataFrame, md: pd.DataFrame) -> pd.DataFrame:
    """Latent-class models: set and score metrics split by whether the selected optimum has an
    inflated class T (``t_class_inflated``) or is label-swapped.  Shows what a model delivers
    when multi-start MAP selects a truth-aligned optimum and what it delivers otherwise."""
    flag = md[["scenario", "rep", "model", "t_class_inflated", "label_swapped_best"]].rename(columns={"model": "method"})
    flag["selected_optimum"] = np.where(flag["t_class_inflated"].astype(bool), "inflated_T",
                                        np.where(flag["label_swapped_best"].astype(bool), "label_swapped", "aligned"))
    d = sm.merge(flag[["scenario", "rep", "method", "selected_optimum"]], on=["scenario", "rep", "method"], how="inner")
    rows = []
    for (sc, m, rule, opt), g in d.groupby(["scenario", "method", "set_rule", "selected_optimum"]):
        for met in BY_MODE_METRICS:
            rows.append(dict(scenario=sc, method=m, set_rule=rule, selected_optimum=opt, metric=met, **mean_ci(g[met])))
    out = pd.DataFrame(rows)
    out["definition"] = (f"selected_optimum: inflated_T = w_T > {T_INFLATION_FACTOR:g} x true prevalence; "
                         "label_swapped = another class tracks the truth better than T on reported edges; "
                         "aligned = neither (classification uses the known truth)")
    return _order(out)


def contrasts_table(sm: pd.DataFrame) -> pd.DataFrame:
    """Paired (method - reference) differences per replicate; reference = family_hb or oracle."""
    rows = []
    for ref in ("family_hb", "oracle"):
        for rule in ("size_matched", "size_oracle", "bfdr05"):
            sub = sm[sm["set_rule"] == rule]
            base = sub[sub["method"] == ref].set_index(["scenario", "rep"])
            if base.empty:
                continue
            for (sc, m, st), g in sub[sub["method"] != ref].groupby(["scenario", "method", "analysis_status"]):
                g = g.set_index(["scenario", "rep"])
                common = g.index.intersection(base.index)
                for met in CONTRAST_METRICS:
                    d = (g.loc[common, met].astype(float) - base.loc[common, met].astype(float)).to_numpy()
                    ok = np.isfinite(d)
                    rows.append(dict(scenario=sc, method=m, reference=ref, set_rule=rule, metric=met,
                                     analysis_status=st, n_method_greater=int(np.sum(d[ok] > 0)),
                                     n_method_less=int(np.sum(d[ok] < 0)), **mean_ci(d)))
    out = pd.DataFrame(rows).rename(columns={"mean": "mean_diff", "median": "median_diff"})
    return _order(out)


def datasets_table(ds: pd.DataFrame) -> pd.DataFrame:
    rows = []
    num = [c for c in ds.columns if c not in ("scenario", "rep", "checksum")]
    for sc, g in ds.groupby("scenario"):
        for c in num:
            rows.append(dict(scenario=sc, quantity=c, **mean_ci(g[c])))
    return pd.DataFrame(rows)


def design_table(an: pd.DataFrame | None) -> pd.DataFrame:
    """Design parameters and the calibration of the simulated rates against the real global
    Family-HB fit (rates conditional on observation)."""
    rows = []
    for k, v in SIM_DESIGN.items():
        rows.append(dict(section="design", quantity=k, family="", value=json.dumps(v) if not isinstance(v, str) else v,
                         analysis_status=analysis_status()))
    rows.append(dict(section="design", quantity="table_scenario_labels", family="",
                     value=json.dumps(TABLE_SCENARIO_LABEL), analysis_status=analysis_status()))
    # what was actually fitted (replicates per model and scenario) and any shortfall from the design
    short = []
    for m in lcm_models():
        n_fit = {sc: sum((fit_path(sc, r, m) / "meta.json").exists() for r in range(SIM_DESIGN["n_reps"]))
                 for sc in SCENARIOS}
        rows.append(dict(section="realised", quantity=f"n_replicates_fitted {m}", family="",
                         value=json.dumps({TABLE_SCENARIO_LABEL.get(k, k): v for k, v in n_fit.items()}),
                         analysis_status=analysis_status()))
        if min(n_fit.values()) < SIM_DESIGN["n_reps"]:
            short.append(m)
    rows.append(dict(section="realised", quantity="models_with_fewer_replicates_than_designed", family="",
                     value=json.dumps(short), analysis_status=analysis_status()))
    rows.append(dict(section="realised", quantity="realised_coverage (replicates fitted by 'fit' without --full-design)",
                     family="", analysis_status=analysis_status(),
                     value=json.dumps({m: {TABLE_SCENARIO_LABEL.get(k, k): v for k, v in cov.items()}
                                       for m, cov in REALISED_COVERAGE.items()})))
    rows.append(dict(section="realised", quantity="realised_coverage_reason", family="",
                     analysis_status=analysis_status(),
                     value=("family_hb_obs fits (10-35 min each) were stopped on 2026-10-13 for compute reasons, in "
                            "replicate-first order, when the fits finished by then were known to select the same "
                            "inflated-T optimum; copying rep00 crashed on a failed restart and its guarded refit was "
                            "abandoned under load.  Rows of this model are means over these replicates only.")))
    for name, cfg in (("restart_check", RESTART_CHECK), ("oracle_check", ORACLE_CHECK)):
        rows.append(dict(section="diagnostic", quantity=name, family="", value=json.dumps(cfg),
                         analysis_status=analysis_status()))
    real = None
    f = C.TABLES / "fit_source_parameters.csv"
    if f.exists():
        real = pd.read_csv(f)
        real = real[(real["universe"] == "global") & (real["fit_id"] == "family_hb") & (real["inference"] == "MAP")]
    rp = C.OUT / "fits" / "global" / C.PRESPECIFIED["evidence_variant_primary"] / "family_hb" / "params.npz"
    if rp.exists():
        w = np.load(rp)["w"]
        rows.append(dict(section="calibration", quantity="real_w (T, N, B_LIT, B_MOTIF)", family="",
                         value=" ".join(f"{x:.4f}" for x in w), analysis_status=analysis_status()))
    for fam in ("LIT", "MOTIF", "COEXP"):
        for par, sim_col, fit_col in (("alpha", "tpr_observed", "fit_alpha"), ("beta", "fpr_observed", "fit_beta"),
                                      ("gamma", "burst_rate_observed", "fit_gamma")):
            def rng3(x):
                x = pd.to_numeric(pd.Series(x), errors="coerce").dropna()
                return f"{x.min():.4f} / {x.median():.4f} / {x.max():.4f}" if len(x) else ""
            if real is not None:
                rr = real[(real["family"] == fam) & (real["parameter"] == par)]["value"]
                if len(rr):
                    rows.append(dict(section="calibration", quantity=f"real_fit_{par} (min/median/max)", family=fam,
                                     value=rng3(rr), analysis_status=analysis_status()))
            if an is not None and len(an):
                a = an[(an["family"] == fam) & (an["scenario"] == "null")]
                per_ann = a.groupby("annotator")[[sim_col, fit_col]].mean() if fit_col in a else None
                if per_ann is not None and len(per_ann):
                    rows.append(dict(section="calibration", quantity=f"sim_realised_{par} null (min/median/max)",
                                     family=fam, value=rng3(per_ann[sim_col]), analysis_status=analysis_status()))
                    rows.append(dict(section="calibration", quantity=f"sim_family_hb_fit_{par} null (min/median/max)",
                                     family=fam, value=rng3(per_ann[fit_col]), analysis_status=analysis_status()))
    return pd.DataFrame(rows)


def empirical_table(sm: pd.DataFrame) -> pd.DataFrame:
    """Simulated literature-degree-quintile-5 index next to the empirical one
    (heldout_degree_calibration.csv, quintile 5).  Post hoc: designed after the held-out
    over-promotion had been seen."""
    rows = []
    f = C.TABLES / "heldout_degree_calibration.csv"
    if f.exists():
        e = pd.read_csv(f)
        e = e[(e["quintile"] == e["quintile"].max()) & e["set"].isin(["strong_bfdr05", "top500"])]
        for _, r in e.iterrows():
            rows.append(dict(source="empirical", scenario=r["reference"], method=r["score"], set_rule=r["set"],
                             over_promotion=r["over_promotion"], lo=r["over_promotion_lo"], hi=r["over_promotion_hi"],
                             share_set=r["share_set"], share_positives=r["share_pos"], set_size=r["set_size"],
                             interval="TF-cluster bootstrap 95% (heldout_eval)", analysis_status="post_hoc"))
    for (sc, m, rule), g in sm.groupby(["scenario", "method", "set_rule"]):
        ci = mean_ci(g["over_promotion_lit_q5"])
        rows.append(dict(source="simulation", scenario=sc, method=m, set_rule=rule, over_promotion=ci["mean"],
                         lo=ci["ci_lo"], hi=ci["ci_hi"], share_set=float(g["share_set_lit_q5"].mean()),
                         share_positives=float(g["share_true_lit_q5"].mean()), set_size=float(g["set_size"].mean()),
                         interval="t 95% CI of the mean over replicates", analysis_status="post_hoc"))
    out = pd.DataFrame(rows)
    out["definition"] = ("over_promotion = share of the set at top-literature-degree-quintile TFs / share of "
                         "positives there (empirical: held-out positives; simulation: true edges)")
    return out


def label_scenarios(df: pd.DataFrame) -> pd.DataFrame:
    """Replace internal scenario names by their table labels (TABLE_SCENARIO_LABEL)."""
    df = df.copy()
    for col in ("scenario", "universe_or_scenario"):
        if col in df:
            df[col] = df[col].replace(TABLE_SCENARIO_LABEL)
    return df


def with_status(df: pd.DataFrame, basis: str | None = None) -> pd.DataFrame:
    """Set ``analysis_status`` (post_hoc) and ``status_basis`` on a manuscript table; the basis is
    taken from the method / model column unless a fixed ``basis`` key of STATUS_BASIS is given."""
    df = label_scenarios(df)
    df["analysis_status"] = analysis_status()
    if basis is not None:
        df["status_basis"] = STATUS_BASIS[basis]
    else:
        col = next((c for c in ("method", "model") if c in df), None)
        df["status_basis"] = df[col].map(status_basis) if col else STATUS_BASIS["core"]
    return df


def build_tables(tables_dir: Path = C.TABLES) -> dict:
    M = load_metrics()
    sm = M["summary"]
    C.ensure_dir(tables_dir)
    C.ensure_dir(OUT)
    label_scenarios(sm).to_csv(OUT / "replicates_summary.csv", index=False)
    label_scenarios(M["strata"]).to_csv(OUT / "replicates_strata.csv", index=False)
    out = {
        "sim_hub_summary": summary_table(sm),
        "sim_hub_deciles": strata_table(M["strata"]),
        "sim_hub_calibration": calibration_table(M["calibration"]),
        "sim_hub_contrasts": contrasts_table(sm),
        "sim_hub_datasets": datasets_table(M["datasets"]),
        "sim_hub_design": design_table(M.get("annotators")),
        "sim_hub_empirical": empirical_table(sm),
    }
    out["sim_hub_modes"], out["sim_hub_modes_replicates"] = modes_tables(M["modes"])
    out["sim_hub_by_mode"] = by_mode_table(sm, M["modes"])
    rc = restart_check_table()
    if len(rc):
        out["sim_hub_restart_check"] = rc
    sig = obs_signature_table()
    if len(sig):
        out["sim_hub_obs_signature"] = sig
    oc = oracle_check_table()
    if len(oc):
        out["sim_hub_oracle_check"] = oc
    for name in list(out):
        out[name] = with_status(out[name], "empirical" if name == "sim_hub_empirical" else None)
    for name, df in out.items():
        df.to_csv(tables_dir / f"{name}.csv", index=False, float_format="%.6g")
        log(f"[tables] {name}.csv: {len(df)} rows")
    C.write_json(dict(n_metric_files=len(list(METRICS.glob('*.json'))), design=SIM_DESIGN,
                      lcm_sha256=lcm_digest(), tables=sorted(out)), OUT / "provenance.json")
    return out


# ----------------------------------------------------------------------------------------
# Command line
# ----------------------------------------------------------------------------------------
def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--scenarios", nargs="+", default=list(SCENARIOS))
        p.add_argument("--reps", default=f"0-{SIM_DESIGN['n_reps'] - 1}")
        p.add_argument("--shard", default=None, help="i/n: every n-th (scenario, rep) starting at i")

    p_fit = sub.add_parser("fit")
    common(p_fit)
    p_fit.add_argument("--models", nargs="+", default=SIM_DESIGN["models"] + SIM_DESIGN["obs_models"])
    p_fit.add_argument("--no-skip-existing", action="store_true")
    p_fit.add_argument("--full-design", action="store_true",
                       help="also fit the replicates outside REALISED_COVERAGE (family_hb_obs: 48 fits, 10-35 min each)")
    p_met = sub.add_parser("metrics")
    common(p_met)
    p_met.add_argument("--only-stale", action="store_true",
                       help="recompute only replicates whose stored metrics do not cover the models fitted so far")
    sub.add_parser("tables")
    p_rc = sub.add_parser("restart-check")
    common(p_rc)
    p_rc.add_argument("--models", nargs="+", default=list(RESTART_CHECK["models"]))
    p_rc.add_argument("--all-replicates", action="store_true",
                      help="also check replicates whose design fit already has an inflated class T")
    p_oc = sub.add_parser("oracle-check")
    p_oc.add_argument("--scenarios", nargs="+", default=list(SCENARIOS))
    p_oc.add_argument("--reps", default=ORACLE_CHECK["reps"])
    p_all = sub.add_parser("all")
    common(p_all)
    a = ap.parse_args(argv)
    if a.cmd in ("fit", "all"):
        models = getattr(a, "models", SIM_DESIGN["models"] + SIM_DESIGN["obs_models"])
        run_fits(a.scenarios, parse_reps(a.reps), models, a.shard, skip_existing=not getattr(a, "no_skip_existing", False),
                 full_design=getattr(a, "full_design", False))
    if a.cmd == "oracle-check":
        run_oracle_check(a.scenarios, parse_reps(a.reps))
    if a.cmd == "restart-check":
        run_restart_check(a.scenarios, parse_reps(a.reps), a.models, a.shard, only_uninflated=not a.all_replicates)
    if a.cmd in ("metrics", "all"):
        run_metrics(a.scenarios, parse_reps(a.reps), a.shard, only_stale=getattr(a, "only_stale", False))
    if a.cmd in ("tables", "all"):
        build_tables()


if __name__ == "__main__":
    main()
