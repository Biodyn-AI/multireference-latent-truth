"""Identifiability of Family-HB: posterior-mode masses, mixture credibility and the calibration of
mode selection (module ``posterior_modes``).

Responds to the f1 verifier (M1: the tissue Strong tiers rest on a Laplace-approximated mass of the
dominant mode) and to the f2 verifier (B1: mode selection is not calibrated and the Strong tier fails
in the competing mode; M1: NUTS coverage was computed from chains in different modes; M2: the
sensitivity fits have competing optima a few nats away; M3: stability of the data-driven grouping;
M4: w_T recovery summarised by bias only).

Status.  No held-out data (ChIP references, perturbation cohorts) are read anywhere in this module;
every input is the held-in annotator evidence and fits made from it.  The analyses were nevertheless
designed after the held-out results checkpoint (``REVISION_PLAN_R2.md`` section 8), so every table
row / column created here carries ``analysis_status = post_hoc``; rows and columns inherited from the
prespecified recovery study (plan section 6 (iii)) keep ``prespecified``.

Parts
-----
(1) ``nuts`` + ``bridge`` -- mode masses without a Gaussian assumption.  For every universe (global,
    immune, kidney, lung; primary evidence variant ``prespecified_cc``, model ``family_hb``) and each of
    the top-2 MAP optima m of the primary MAP fit (``fits/<u>/<v>/family_hb/optima_params.npz``,
    polished unconstrained optima written by ``fit_primary.laplace_optima``):
      * NUTS restricted to the mode: 4 chains started from basin-checked draws of the Laplace
        approximation at optimum m (start accepted only if L-BFGS from it returns to optimum m,
        |log p difference| <= ``BASIN_TOL``), 1,000 warm-up + 1,000 draws, target acceptance 0.95, seed
        derived from ``PRESPECIFIED['random_seed']``.  Stored under the fit contract as
        ``fits/<u>/<v>/family_hb_nuts_mode<m>/`` plus ``draws_unconstrained.npz`` (every draw in the
        unconstrained space with its potential energy and chain id).  Tissues: both modes are run
        here (``DEFAULT_NUTS_MODES``); the mode-0 run is an independent replicate of the canonical
        primary fit ``family_hb_nuts`` (same construction, different seed) that keeps all 4,000 draws
        (the canonical fit stores 1,000 thinned draws).  Global: mode 1 only, 4 x 500/500
        (``NUTS_SETTINGS``; one global chain set costs 1.5-3 h on the shared laptop and the 105-nat
        difference between the modes makes the mode probability insensitive to Monte Carlo error by
        tens of orders of magnitude); global mode 0 = the canonical primary fit (4 x 1000/1000,
        1,000 thinned draws).
      * Each chain is verified to stay in its mode: (a) every draw is assigned to the MAP optimum whose
        credibility vector is closest (edge-weighted mean absolute difference over observed patterns),
        (b) L-BFGS is started from ``BASIN_CHECKS_PER_CHAIN`` evenly spaced draws of every chain and the
        optimum it reaches is recorded (basin of attraction of the gradient flow).
      * Marginal likelihood of the mode, Z_m = integral of exp(-U) over the mode, by bridge sampling
        (Meng & Wong 1996, iterative scheme; Gronau et al. 2017 implementation): the draws of every
        chain are split in half; a multivariate normal proposal g is fitted (mean, covariance) to the
        first halves; the second halves (N1, effective size = median bulk ESS over coordinates) and
        N2 = ``N2_MULT`` N1 proposal draws enter the fixed-point iteration
            r <- [mean_i l2_i / (s1 l2_i + s2 r)] / [mean_j 1 / (s1 l1_j + s2 r)],
        l = q / g, q = exp(-U), s1 = N1eff / (N1eff + N2), s2 = N2 / (N1eff + N2) (computed in log
        space).
        COORDINATES (deviation from the plain recipe, fixed 2026-10-13 after the plain recipe failed;
        label-free).  NUTS samples the non-centred coordinates (zf, zj, log omega, log tau).  Because
        the data determine the annotator parameters q_j = mu_f + tau zj sharply, the posterior of
        (zj, log tau) is strongly curved there and a normal proposal fitted in those coordinates
        overlaps it poorly: relative error > 1, iteration not converged in several runs, log Z
        spread of 1-2.5 nats between splits.  The proposal is therefore fitted in the CENTRED
        coordinates y (``centred_coords``: zf -> mu_f, zj -> q_j; exact triangular bijection with
        log |det dy/dx| = sum_q (F log omega_q + K log tau_q)); Z is invariant under the change of
        variables.  The plain recipe (non-centred coordinates, N2 = N1) and a Student-t(5) proposal
        in the centred coordinates are reported as diagnostics next to the estimate (``role``).
        Relative MC error: Fruehwirth-Schnatter (2004) approximation for the primary split, and the
        SD of log Z over ``N_SPLITS`` repeated random splits (fresh proposal draws each); the two are
        combined in quadrature for the interval of the log Z difference.  An estimate from independent
        chains (the canonical primary NUTS fit, mode 0, tissues) is a further check.
      * Mode posterior probabilities p_m = Z_m / sum_l Z_l over the top-2 modes (the other optima of
        kidney / lung have Laplace masses < 1e-10), compared with the Laplace numbers of
        ``fit_map_optima.csv``.
(2) ``governance`` -- mixture credibility c_mix = sum_m p_m c_m (c_0 = the canonical primary
    dominant-mode NUTS mean ``family_hb_nuts``; c_1 = NUTS mean of mode 1) and the sensitivity of the
    BFDR tiers to the mass of the second mode, p2 in {bridge, upper end of the bridge Monte Carlo
    interval, Laplace, 0.01, 0.05, 0.10, 1}.  Tiers are recomputed with the functions of
    ``governance.py`` (``pattern_key_map``, ``bfdr_levels``, ``dual_tier``, ``realized_bfdr``): the
    context-fit credibility is mixed; the global-fit credibility is either kept at its own (bridge)
    estimate (``global_rule = own_estimate``; global p2 ~ 1e-46) or mixed with the same p2
    (``same_p2``, a stress scenario).  A reference row replaces the primary NUTS mean by the
    independent mode-0 rerun (Monte Carlo noise of the tiers).  ``promoted`` = dual Strong or
    Moderate over all edges of the tissue universe (the promotable set; per-method promotion is a
    subset).  The mixture fit at the bridge estimate is stored as ``fits/<u>/<v>/family_hb_mixture/``
    (contract files; credible intervals from the p-weighted mixture of the per-mode draws).
(3) ``simulate`` -- calibration of mode selection.  Data are simulated from the primary ``family_hb``
    MAP fit by one of three simulators (``SIMULATORS``, ``SIM_LABEL_INFO``; part 3b):
      * ``faithful_cc`` (primary since 2026-10-14; labels ``immune_cc``, ``global_cc``,
        ``immune_truthmode1_cc``): real LIT masks, simulated LIT reports, and the curated-conditional
        MOTIF/COEXP masks REGENERATED from the simulated DoRothEA-channel LIT reports (``simulate_cc``);
      * ``faithful_cc_strat`` (labels ``*_ccs``): the same with class weights that depend on the LIT
        observation mask (``stratum_class_weights``); it reproduces the real LIT positive counts and ~90% of
        the informative MOTIF/COEXP observations (``modes_faithful_design.csv``);
      * ``fixed_masks`` (labels ``immune``, ``global``, ``immune_truthmode1``; superseded sensitivity):
        every real observation mask kept (``fit_sensitivity.simulate_fixed_masks``, seed
        ``derived_seed('recovery', u, r)``: replicates r < 20 (immune) / r < 5 (global) are exactly the
        data sets of the stored recovery study).  It decouples the MOTIF/COEXP masks from the LIT reports
        (modes verifier B1), so its gap distribution describes a less informative design.
    The tables carry ``simulator``; ``universe`` holds the table label (immune / global /
    immune_truthmode1) for every simulator.  Family-HB MAP is refitted from 20 restarts (immune:
    10 restarts with the recovery fit seed -- reproducing the stored recovery fit, verified -- plus 10
    with seed ``derived_seed('modes_sim_fit', u, r)``) or 10 restarts (global: the recovery seed only).
    Restarts are grouped into distinct optima (within 1e-3); for the top-2 optima the Laplace mass
    (Newton-polished, exact Hessian) is computed.  The truth mode = the optimum whose credibility has
    the highest edge-weighted Spearman correlation with the oracle credibility (true parameters);
    an L-BFGS run started at the generating parameters (truth-start) is recorded as a second
    criterion.  Per replicate: best / second log posterior and gap, mode correctness, Strong
    (BFDR <= 0.05) and Strong+Moderate (BFDR <= 0.20) set sizes with their claimed BFDR and their
    REALISED FDR against the simulated classes, the same for the oracle, the second optimum, the
    truth-start optimum, the Laplace-weighted mixture of the top-2 optima and the EM fit (10 restarts,
    recovery seed), and 10-bin calibration counts.  Summaries: P(dominant mode = truth mode | gap
    bin), realised FDR by mode correctness, a logistic fit of correctness on the gap, and the position
    of the real-data gaps (immune 7.5, kidney 7.9, lung 11.6, global 108 nats).
    Reverse check (``immune_truthmode1``, 30 replicates, 20 restarts): data generated at the real SECOND
    immune optimum.  For every replicate (both directions) the optimum resembling the OTHER real mode is
    identified (highest Spearman with the credibility at the other real optimum's parameters,
    ``alt_oracle_stats``), and the signed log-posterior difference truth-mode optimum minus that optimum
    is compared with the real gaps: P(other mode beats the truth by >= the real gap) is the simulated
    probability of seeing the real-data ordering if the truth were the other mode.
(3b) design-faithful simulators (``cc_design``, ``simulate_cc``; see the section comment in the code): the
    cc rule applied to the real LIT reports reproduces every real MOTIF/COEXP mask exactly (checked on every
    load); a simulated LIT report is a DoRothEA-channel report by the real per-edge flag where the annotator
    has a real record of the edge, otherwise by its source's channel rule (per-(source, TF) real fraction for
    the four mixed sources); MOTIF/COEXP TF coverage is the real whole-resource coverage.  LIT node-set masks
    are kept (not regenerated; ``lit_node_check``).  The recovery study is repeated under both faithful
    simulators from the first 20 (immune) / 5 (global) replicates (``faithful_recovery_tables``).
(4) ``tables`` -- recovery-table fixes (``fit_recovery*.csv``; old columns and rows are kept as written
    by ``fit_sensitivity.recovery_tables``): realised FDR / set sizes per replicate (from the exact
    replay in part 3), ECE over credibility bins 1-9 (the near-zero bin holds ~99% of the edges),
    w_T (and alpha / beta) bias and RMSE separately by mode, and NUTS coverage only from converged
    single-mode chain sets (chains assigned to the nearest optimum of the replicate; set converged if it
    has >= 2 chains, rank-normalised split R-hat <= ``RHAT_OK`` and bulk ESS >= ``ESS_OK`` over w and
    theta), otherwise ``not_assessable``.  ``fit_sensitivity.build_tables`` calls
    ``augment_recovery_tables`` so a rebuild of that module keeps the new columns.
(5) ``sensmodes`` / ``clusterstab`` -- for every MAP sensitivity fit of ``fit_sensitivity`` (evidence
    variants, priors, groupings; ``family_hb`` and ``em``) in immune and global: distinct optima from
    ``restarts.csv``, gap to the second optimum, Laplace mass of the top-2 optima (best = stored
    ``params.npz``, second = exact replay of its first restart; both Newton-polished) and the
    second optimum's edge-level agreement with the primary fit (BFDR Strong/Moderate Jaccard, top-k
    Jaccard, Spearman; ``fit_sensitivity.compare_views``).  Clustering stability of the data-driven
    grouping: adjusted Rand index (ARI) against the modality families and against the full-data
    partition, across the 5 TF-grouped CV training folds (stored partitions) and 50 TF-bootstrap
    resamples (TFs drawn with replacement, pattern counts recomputed, ``datadriven_families``).

Manuscript tables (``data/r2/tables``)
--------------------------------------
* ``modes_bridge.csv``                 per universe x mode x role (primary estimate and checks): NUTS
                                       diagnostics, chain-in-mode checks, bridge log Z with MC
                                       errors, Laplace log mass, p_m (bridge vs Laplace).
* ``modes_governance_sensitivity.csv`` per tissue x scenario: Strong / Moderate / promoted counts,
                                       Jaccard with the primary tiers, expected FDR of the primary
                                       Strong set under the mixture.
* ``modes_simulation.csv``             summaries (per universe x gap bin x quantity) and
                                       ``modes_simulation_replicates.csv`` (per replicate).
* ``modes_simulation_calibration.csv`` pooled 10-bin calibration of the dominant-mode credibility,
                                       by mode correctness (all three: ``simulator`` column).
* ``modes_faithful_design.csv``        simulators vs the real design: edges with a LIT report, Cond,
                                       MOTIF/COEXP-observed edges (and with a LIT report), per-annotator
                                       observed / reported counts; replicate mean / SD / range and the
                                       analytic expectation (with mixed-channel bounds).
* ``modes_lit_node_check.csv``         why the LIT node-set masks are not regenerated (per LIT annotator).
* ``fit_sensitivity_modes.csv``        per sensitivity fit: optima, gap, Laplace mass, second-mode
                                       agreement with the primary.
* ``fit_grouping_stability.csv``       ARI of the data-driven grouping (folds, bootstraps) and
                                       per-annotator co-assignment stability.
* ``fit_recovery.csv``, ``fit_recovery_calibration.csv``, ``fit_recovery_replicates.csv`` augmented
  (``simulator``: fixed_masks rows = the stored recovery study, faithful_cc / faithful_cc_strat rows = the
  recovery study under the faithful simulators; ``post_hoc_columns``);
  ``fit_recovery_nuts_chainsets.csv`` (recovery-NUTS chains grouped by nearest optimum, convergence under
  the module's lenient rule and the project's frozen flag rule).

Command line (set the thread environment first; at most two compute processes)::

    export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \\
           XLA_FLAGS='--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=2'
    nice -n 10 .venv/bin/python subproject_24_multireference_latent_truth_model/src/r2/posterior_modes.py all
    # or stage by stage (stages are resumable; --force recomputes)
    ... posterior_modes.py nuts --universes immune kidney lung global [--modes 0 1]
    ... posterior_modes.py simulate [--universes immune_cc global_cc immune_truthmode1_cc immune_ccs global_ccs
                                     immune_truthmode1_ccs] [--reps 0 1 ...]   # default: the faithful labels
    ... posterior_modes.py mixture
    ... posterior_modes.py bridge
    ... posterior_modes.py sensmodes --universes immune global
    ... posterior_modes.py clusterstab
    ... posterior_modes.py governance
    ... posterior_modes.py tables

Deterministic: every random quantity is seeded from ``PRESPECIFIED['random_seed']``.
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=2")

import argparse  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from r2 import common as C  # noqa: E402
from r2 import fit_sensitivity as FS  # noqa: E402
from r2 import lcm  # noqa: E402

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpyro  # noqa: E402
from jax.flatten_util import ravel_pytree  # noqa: E402
from numpyro.infer.util import initialize_model, unconstrain_fn  # noqa: E402
from scipy.optimize import minimize  # noqa: E402

# ----------------------------------------------------------------------------------------
# Settings (frozen here; nothing is chosen from wall-clock pilots or from held-out data)
# ----------------------------------------------------------------------------------------
PS = C.PRESPECIFIED
SEED = int(PS["random_seed"])
VARIANT = PS["evidence_variant_primary"]
MODEL = "family_hb"
FITS = C.OUT / "fits"
MODOUT = C.OUT / "posterior_modes"
# Frozen copy of the primary MAP fits (family_hb, em) and of the first-pass primary NUTS fits, taken
# 2026-10-08 21:2x before the fit_primary re-run ("fit_fixes") rewrote the canonical directories;
# MANIFEST.sha256 inside.  The MAP optima (start points of the per-mode NUTS runs), the simulation
# truth (= the parameters the stored recovery study simulated from; verified by exact replay) and
# the sensitivity-fit comparisons read this copy, so they do not change while other modules refit.
# The canonical (Newton-polished) MAP fits agree with it to 1e-7 in the optimum positions, log
# posteriors and Laplace masses and to 3.3e-6 in theta (checked 2026-10-13; ``snapshot_check``).
# The first-pass NUTS copies are superseded by the canonical 4 x 1000/1000 fits and are not used.
SNAP = MODOUT / "primary_map_snapshot"
IDENTIFIED_SITES = ("w", "a", "b", "g", "theta", "cred_observed")
LBFGS_OPTIONS = dict(maxiter=20000, gtol=1e-6, ftol=1e-15, maxcor=30)      # as lcm.fit_map
UNIVERSES = ("global",) + tuple(C.TISSUES)
MODES = (0, 1)
DEFAULT_NUTS_MODES = {"immune": (0, 1), "kidney": (0, 1), "lung": (0, 1), "global": (1,)}   # global mode 0 = primary fit
NUTS_SETTINGS = {  # 4 x 1000/1000 as in plan section 6; global mode 1 4 x 500/500 (frozen 2026-10-08, when the
    # primary global fit had the same settings; that fit was later re-run at 4 x 1000/1000 by fit_fixes): one
    # global chain-set takes 1.5-3 h on the shared laptop and the global mode gap (108 nats) makes the mode
    # probability insensitive to the Monte Carlo error by many orders of magnitude
    "global": dict(num_chains=4, num_warmup=500, num_samples=500),
    "immune": dict(num_chains=4, num_warmup=1000, num_samples=1000),
    "kidney": dict(num_chains=4, num_warmup=1000, num_samples=1000),
    "lung": dict(num_chains=4, num_warmup=1000, num_samples=1000),
}
TARGET_ACCEPT = 0.95
INIT_OVERDISPERSION = 1.0
BASIN_TOL = 0.01                 # nats: an L-BFGS end point belongs to optimum k if |lp - lp_k| <= BASIN_TOL
BASIN_ASSIGN_TOL = 0.05          # nats: a basin-check end point farther than this from every optimum = 'other'
BASIN_CHECKS_PER_CHAIN = {"global": 5, "immune": 10, "kidney": 10, "lung": 10}
MAX_DRAWS_STORED = 1000
N_SPLITS = 20                    # repeated random splits for the bridge-sampling MC error
N_SPLITS_CHECK = 5               # ... for the diagnostic / robustness bridge runs
N2_MULT = 5                      # proposal draws per posterior draw in the estimation half (N2 = N2_MULT * N1)
T_PROPOSAL_DF = 5.0              # degrees of freedom of the Student-t robustness proposal
HIER_TYPES = ("b", "a", "g")     # hierarchical parameter types of lcm.build_model (MAR models)
BRIDGE_TOL = 1e-10
BRIDGE_MAXITER = 1000
P2_GRID = (0.01, 0.05, 0.10)
MODE_TOL = 1e-3                  # restarts within this of a group's best = same optimum
# Simulators (part 3b).  'fixed_masks' keeps every real observation mask and re-draws every report
# (fit_sensitivity.simulate_fixed_masks; superseded, kept as a labelled sensitivity: it decouples the
# curated-conditional MOTIF/COEXP masks from the LIT reports).  'faithful_cc' keeps the real LIT masks and
# REGENERATES the MOTIF/COEXP masks from the simulated LIT reports with the evidence builder's rule
# (``simulate_cc``).  'faithful_cc_strat' is 'faithful_cc' with class weights that depend on the LIT
# observation mask (``stratum_class_weights``), which reproduces the per-annotator LIT positive counts and
# most of the informative MOTIF/COEXP observations of the real data.
SIMULATORS = ("fixed_masks", "faithful_cc", "faithful_cc_strat")
PRIMARY_SIMULATOR = "faithful_cc"
# simulation label (directory sim/<label>) -> (table label written to the ``universe`` column, simulator)
SIM_LABEL_INFO = {
    "immune": ("immune", "fixed_masks"), "global": ("global", "fixed_masks"),
    "immune_truthmode1": ("immune_truthmode1", "fixed_masks"),
    "immune_cc": ("immune", "faithful_cc"), "global_cc": ("global", "faithful_cc"),
    "immune_truthmode1_cc": ("immune_truthmode1", "faithful_cc"),
    "immune_ccs": ("immune", "faithful_cc_strat"), "global_ccs": ("global", "faithful_cc_strat"),
    "immune_truthmode1_ccs": ("immune_truthmode1", "faithful_cc_strat"),
}
N_SIM = {"immune": 100, "global": 10, "immune_truthmode1": 30,
         "immune_cc": 100, "global_cc": 20, "immune_truthmode1_cc": 30,
         "immune_ccs": 100, "global_ccs": 20, "immune_truthmode1_ccs": 30}
N_RESTARTS_SIM = {"immune": (10, 10), "global": (10, 0), "immune_truthmode1": (10, 10),  # (set A, set B) restarts
                  "immune_cc": (10, 10), "global_cc": (10, 0), "immune_truthmode1_cc": (10, 10),
                  "immune_ccs": (10, 10), "global_ccs": (10, 0), "immune_truthmode1_ccs": (10, 10)}
# simulation label -> (data universe, optimum of the real primary MAP fit used as the generating parameters).
# 'immune_truthmode1' (post hoc reverse check): data generated at the real SECOND optimum (mode 1), to see how
# often and by how much the fit then prefers the optimum that resembles the real dominant mode.
SIM_TRUTH = {"immune": ("immune", 0), "global": ("global", 0), "immune_truthmode1": ("immune", 1),
             "immune_cc": ("immune", 0), "global_cc": ("global", 0), "immune_truthmode1_cc": ("immune", 1),
             "immune_ccs": ("immune", 0), "global_ccs": ("global", 0), "immune_truthmode1_ccs": ("immune", 1)}
SIM_ALT = {"immune": 1, "global": 1, "immune_truthmode1": 0,   # the other real mode (alternative oracle)
           "immune_cc": 1, "global_cc": 1, "immune_truthmode1_cc": 0,
           "immune_ccs": 1, "global_ccs": 1, "immune_truthmode1_ccs": 0}
# replicates of the faithful simulations that form the recovery study (as r < 20 / r < 5 of the fixed-mask study)
N_RECOVERY_FROM_SIM = {"immune": 20, "global": 5}
N_RESTARTS_EM_SIM = 10
GAP_BINS = (0.0, 1.0, 2.5, 5.0, 10.0, 20.0, np.inf)
REAL_GAPS = {"immune": 7.506647, "kidney": 7.906500, "lung": 11.621276, "global": 108.337936}
N_CAL_BINS = 10
RHAT_OK = 1.05                   # single-mode chain set: R-hat over w, theta (250 thinned draws per chain)
ESS_OK = 100.0
# the project's frozen NUTS flag rule (plan section 7 item 12): flagged if R-hat > 1.01 or bulk ESS < 400; the
# single-mode coverage is reported under both rules (kind nuts_coverage_single_mode / ..._frozen_rule)
RHAT_FROZEN = 1.01
ESS_FROZEN = 400.0
N_BOOT_CLUSTER = 50
SENS_UNIVERSES = ("immune", "global")
Q_STRONG = float(PS["bfdr_strong"])
Q_MODERATE = float(PS["bfdr_moderate"])
STATUS = "post_hoc"

# dated decision log of this module (shared table ``eval_analysis_decisions.csv``; rows replaced on every
# ``tables`` run).  Everything here is post hoc (designed after the held-out checkpoint) and label-free.
ANALYSIS_DECISIONS = [
    dict(id="modes_design", date="2026-10-08", status="post_hoc",
         item="Family-HB identifiability: NUTS restricted to each of the top-2 MAP optima (basin-checked Laplace "
              "starts, 4 x 1000/1000, TA 0.95), per-mode bridge sampling, mixture credibility, BFDR-tier sensitivity "
              "to the second-mode mass, simulation calibration of mode selection, recovery-table fixes, sensitivity-"
              "fit mode gaps, data-driven grouping stability",
         note="responds to verifiers f1 M1 and f2 B1/M1/M2/M3/M4; reads no held-out data; designed after the held-out "
              "checkpoint (plan section 8) so every new row/column is analysis_status = post_hoc"),
    dict(id="modes_global_mode1_settings", date="2026-10-08", status="post_hoc",
         item="global mode-1 NUTS at 4 x 500/500 (tissues 4 x 1000/1000); global mode 0 = canonical primary NUTS fit",
         note="one global chain set costs 1.5-3 h on the shared laptop; the ~105-nat mode gap makes p2 insensitive to "
              "Monte Carlo error by tens of orders of magnitude; frozen before any bridge estimate"),
    dict(id="modes_bridge_centred", date="2026-10-13", status="post_hoc",
         item="bridge-sampling proposal (multivariate normal) fitted in CENTRED hierarchical coordinates "
              "(mu_f, q_j; exact Jacobian), N2 = 5 N1, 20 repeated splits; raw non-centred recipe and a Student-t(5) "
              "proposal reported as diagnostics (modes_bridge.csv role)",
         note="first-pass estimator (proposal in the non-centred coordinates, N2 = N1) failed its own diagnostics "
              "(relative error 1.0-1.3, iteration not converged in 4/9 runs, split SD 1-2.5 nats); first-pass "
              "records kept in outputs/r2/posterior_modes/bridge_firstpass_superseded/; label-free"),
    dict(id="modes_mode0_draws", date="2026-10-13", status="post_hoc",
         item="mode-0 bridge estimate from this module's mode-0 re-run (all 4,000 draws); the canonical primary NUTS "
              "fit (1,000 thinned draws, independent chains) is an independent check; mixture credibility uses "
              "c_0 = canonical primary NUTS mean and c_1 = mode-1 NUTS mean",
         note="first-pass primary NUTS fits (snapshot) superseded by the fit_fixes re-runs; MAP optima in the snapshot "
              "equal the canonical polished optima to 1e-7 (snapshot_check.csv)"),
    dict(id="modes_simulation", date="2026-10-08", status="post_hoc",
         item="mode-selection simulation: immune 100 replicates (20 restarts), global 10 replicates (10 restarts) "
              "generated at the primary MAP optimum 0 with real masks; reverse check immune_truthmode1 (30 replicates, "
              "generated at the real optimum 1) added 2026-10-13",
         note="task asked for >= 60 immune replicates; replicates < 20 (immune) / < 5 (global) are the stored "
              "recovery-study data sets (exact replay)"),
    dict(id="modes_recovery_fixes", date="2026-10-08", status="post_hoc",
         item="recovery tables: realised FDR per replicate, ECE over credibility bins 1-9, w_T/alpha/beta bias and "
              "RMSE by mode, NUTS coverage only from converged single-mode chain sets (else not_assessable)",
         note="old prespecified rows/columns kept unchanged"),
    dict(id="modes_logistic", date="2026-10-14", status="post_hoc",
         item="logistic fit of P(dominant = truth mode) on the gap by self-contained Newton ML (Wald interval), over "
              "replicates whose restarts found the truth mode; flagged extrapolated when < 5 replicates have gap >= "
              "the real gap",
         note="statsmodels import fails with the installed scipy (the 2026-10-13 table had fit_failed rows)"),
    dict(id="modes_faithful_simulator", date="2026-10-14", status="post_hoc",
         item="design-faithful simulator (simulate_cc): real LIT masks, simulated LIT reports, curated-conditional "
              "MOTIF/COEXP masks REGENERATED from the simulated DoRothEA-channel LIT reports with the evidence builder's "
              "rule (whole-resource TF coverage); validated by exact reconstruction of every real MOTIF/COEXP mask from "
              "the real reports; primary simulator ('faithful_cc') of the mode-selection simulation and of the "
              "recovery study; the fixed-mask results are kept as a labelled sensitivity (simulator column)",
         note="modes verifier B1: the fixed-mask simulator decouples the cc masks from the LIT reports (real data: "
              "795/795 immune, 16,293/16,293 global MOTIF/COEXP-observed edges carry a LIT report; a fixed-mask replicate "
              "about 5%); label-free"),
    dict(id="modes_faithful_channel_rule", date="2026-10-14", status="post_hoc",
         item="channel membership of a simulated LIT report: the real via_dorothea flag where the annotator has a real "
              "record of the edge; else the source's rule (14 of 18 held-in LIT sources pure); for TRRUST, HTRIdb, IntAct "
              "and TFactS the real per-(source, TF) fraction, constant where 0 or 1, else one fixed Bernoulli draw per cell "
              "(seed cc_channel_map, identical in every replicate)",
         note="analytic bounds with every mixed cell off / on are reported in modes_faithful_design.csv"),
    dict(id="modes_faithful_strat", date="2026-10-14", status="post_hoc",
         item="second faithful simulator 'faithful_cc_strat': class weights depend on the edge's LIT observation mask "
              "(stratum), w_k = edge-weighted mean real posterior class probabilities in stratum k at the generating "
              "parameters; oracle = Bayes posterior under the stratified prior",
         note="the plain faithful simulator reproduces only ~60% of the real MOTIF/COEXP-observed edges because the "
              "fitted model under-predicts the DoRothEA-channel LIT positive rates under the outcome-dependent LIT node "
              "masks (fit_ppc z up to 16); the stratified weights reproduce every LIT annotator's expected positives "
              "(M-step identity) and ~90% of the informative observations"),
    dict(id="modes_lit_nodes_not_regenerated", date="2026-10-14", status="post_hoc",
         item="report-dependent LIT node-set masks are not regenerated (f2 minor 7); modes_lit_node_check.csv reports the "
              "cells and prior-predictive reports outside the real node sets and the node genes supported only by "
              "in-universe records",
         note="the MAR rates are rates within the selected (observed) cells; the model defines no report rate outside "
              "the node sets, so regenerating them would need an observation model (the informative-observation "
              "extension, which is degenerate: obs module)"),
    dict(id="modes_faithful_replicates", date="2026-10-14", status="post_hoc",
         item="faithful simulations: immune 100 replicates x 20 restarts, immune reverse check (truth = real optimum 1) "
              "30 x 20, global 20 x 10, for both faithful simulators; recovery study = replicates r < 20 "
              "(immune) / r < 5 (global) generated at optimum 0 (Family-HB = best of the 10 set-A restarts, EM 10 "
              "restarts, truth-start, oracle)",
         note="task: >= 60 immune, 30 reverse, >= 10 global; recovery NUTS (coverage) not repeated under the faithful "
              "simulators (fixed-mask rows kept, labelled)"),
    dict(id="modes_nuts_coverage_rules", date="2026-10-14", status="post_hoc",
         item="recovery-NUTS single-mode coverage reported under the module's lenient rule (R-hat <= 1.05, bulk ESS >= "
              "100; kind nuts_coverage_single_mode) AND the project's frozen flag rule (R-hat <= 1.01, bulk ESS >= 400; "
              "kind nuts_coverage_single_mode_frozen_rule)",
         note="modes verifier m1: under the frozen rule truth-mode coverage is not assessable"),
    dict(id="modes_p2_range", date="2026-10-14", status="post_hoc",
         item="second-mode mass reported with the range across draw sets and proposals (primary, canonical mode-0 "
              "chains, Student-t; modes_bridge.csv p2_range_draw_sets_proposals_lo/hi) next to the within-run interval",
         note="modes verifier m2"),
    dict(id="modes_bridge_diag_flag", date="2026-10-14", status="post_hoc",
         item="modes_bridge.csv diag_flag_all_sites / diag_flag_identified_sites with the project's flag rule "
              "(R-hat > 1.01 or bulk ESS < 400)",
         note="modes verifier m6: global mode-1 NUTS (4 x 500/500) is flagged (identified-site R-hat 1.012)"),
    dict(id="modes_logistic_range", date="2026-10-14", status="post_hoc",
         item="logistic P(correct | real gap): no interval when the real gap exceeds every simulated gap; likelihood-"
              "ratio statistic of the slope reported; kidney / lung rows note that they use the immune calibration",
         note="modes verifier m5"),
    dict(id="modes_recovery_labels", date="2026-10-14", status="post_hoc",
         item="fit_recovery*.csv: simulator and post_hoc_columns columns; fit_recovery_nuts_chainsets.csv: "
              "analysis_status = post_hoc; modes_governance_sensitivity.csv: promoted_definition note",
         note="modes verifier m4 / m7"),
    dict(id="modes_truth_missed", date="2026-10-14", status="post_hoc",
         item="signed truth-mode minus other-mode log posterior: when the restarts miss the truth basin "
              "(truth_mode_found False), the truth-start optimum stands in for the truth mode (truth_vs_alt)",
         note="affects immune replicate 62 only (20 restarts never reached the truth basin; truth-start optimum 0.52 "
              "nats below the best, other-mode, optimum); before the fix it was compared with a degenerate optimum"),
]


def record_decisions(path: Path | None = None) -> None:
    """Replace this module's rows of the shared dated decision log (same writer as ``fit_primary``)."""
    path = path or (C.TABLES / "eval_analysis_decisions.csv")
    new = pd.DataFrame(ANALYSIS_DECISIONS).assign(module="posterior_modes")
    if path.exists():
        old = pd.read_csv(path)
        new = pd.concat([old[old["module"] != "posterior_modes"], new], ignore_index=True)
    C.ensure_dir(path.parent)
    new[["module", "id", "date", "status", "item", "note"]].sort_values(["module", "id"], kind="mergesort").to_csv(
        path, index=False)


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def derived_seed(*parts) -> int:
    return FS.derived_seed(*parts)


def fit_dir(u: str, fit_id: str, variant: str = VARIANT) -> Path:
    return FITS / u / variant / fit_id


def load_data(u: str, variant: str = VARIANT) -> lcm.PatternData:
    return FS.load_data(u, variant)


def observed(data: lcm.PatternData) -> np.ndarray:
    return (data.R >= 0).any(1)


def map_dir(u: str, fit_id: str = MODEL) -> Path:
    """Primary MAP fit directory: the frozen snapshot if present, else the canonical directory."""
    d = SNAP / u / fit_id
    return d if (d / "meta.json").exists() else fit_dir(u, fit_id)


def _jsonable(x):
    if isinstance(x, dict):
        return {k: _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, np.floating):
        return float(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, np.bool_):
        return bool(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    return x


def _md5(path: Path) -> str:
    import hashlib
    return hashlib.md5(Path(path).read_bytes()).hexdigest()


def versions() -> dict:
    import scipy
    return dict(jax=jax.__version__, numpyro=numpyro.__version__, numpy=np.__version__, scipy=scipy.__version__)


def chain_diag(arr: np.ndarray) -> tuple[float, float, float]:
    """R-hat (max), bulk and tail ESS (min) over the trailing dims of (chains, draws, ...)."""
    import arviz as az
    arr = arr.reshape(arr.shape[0], arr.shape[1], -1)
    ds = az.convert_to_dataset({"x": arr})
    return (float(np.nanmax(az.rhat(ds)["x"].values)), float(np.nanmin(az.ess(ds, method="bulk")["x"].values)),
            float(np.nanmin(az.ess(ds, method="tail")["x"].values)))


# ----------------------------------------------------------------------------------------
# Small numerical helpers
# ----------------------------------------------------------------------------------------
def logmeanexp(x: np.ndarray) -> float:
    x = np.asarray(x, float)
    m = np.max(x)
    if not np.isfinite(m):
        return float(m)
    return float(m + np.log(np.mean(np.exp(x - m))))


def mvn_fit(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mean and Cholesky factor of the sample covariance (jitter added if needed)."""
    mu = X.mean(0)
    S = np.cov(X, rowvar=False)
    S = 0.5 * (S + S.T)
    jit = 0.0
    for _ in range(10):
        try:
            L = np.linalg.cholesky(S + jit * np.eye(len(mu)))
            return mu, L
        except np.linalg.LinAlgError:
            jit = max(jit * 10, 1e-10 * np.trace(S) / len(mu))
    raise np.linalg.LinAlgError("covariance not positive definite")


def mvn_logpdf(X: np.ndarray, mu: np.ndarray, L: np.ndarray) -> np.ndarray:
    from scipy.linalg import solve_triangular
    d = len(mu)
    Z = solve_triangular(L, (X - mu).T, lower=True)
    return -0.5 * np.sum(Z ** 2, 0) - np.sum(np.log(np.diag(L))) - 0.5 * d * np.log(2 * np.pi)


def mvt_scale(L: np.ndarray, df: float) -> np.ndarray:
    """Cholesky factor of the scale matrix of a multivariate t with covariance L L' (df > 2)."""
    return L * math.sqrt((df - 2.0) / df)


def mvt_logpdf(X: np.ndarray, mu: np.ndarray, Ls: np.ndarray, df: float) -> np.ndarray:
    """Log density of the multivariate Student-t with location mu, scale matrix Ls Ls' and df."""
    from scipy.linalg import solve_triangular
    from scipy.special import gammaln
    d = len(mu)
    Z = solve_triangular(Ls, (X - mu).T, lower=True)
    maha = np.sum(Z ** 2, 0)
    return (gammaln(0.5 * (df + d)) - gammaln(0.5 * df) - 0.5 * d * np.log(df * np.pi)
            - np.sum(np.log(np.diag(Ls))) - 0.5 * (df + d) * np.log1p(maha / df))


def centred_coords(pt: dict, fam_idx) -> dict | None:
    """Bijection between the unconstrained NON-CENTRED coordinates x of a hierarchical model of
    ``lcm.build_model`` (the coordinates NUTS and the potential use) and its CENTRED coordinates y:

        y = x except   zf_q[f] -> mu^q_f = mu0_q + omega_q zf_q[f]      (family means)
                       zj_q[j] -> q_j    = mu^q_f(j) + tau_q zj_q[j]    (annotator parameters),

    for every parameter type q in ``HIER_TYPES`` present, with omega_q = exp(x[omega_q]) and
    tau_q = exp(x[tau_q]).  The map is triangular, so log |det dy/dx| = sum_q (F log omega_q +
    K n_g log tau_q) and a density transforms as log q_y(y) = log q_x(x(y)) - log |det dy/dx|.

    Why: the data determine the annotator parameters q_j sharply, so in the non-centred coordinates
    the posterior of (zj_q, log tau_q) lies along the curved surface zj = (q_j - mu_f) / tau and a
    multivariate normal fitted there overlaps it poorly (bridge relative error > 1 and a
    non-converging iteration in this application); in the centred coordinates the posterior is close
    to normal.  The normalising constant is invariant under the change of variables.

    Returns ``None`` if ``pt`` has no hierarchical sites (the caller then uses x itself), otherwise
    ``dict(name, to_y, to_x)`` with ``to_y(X) -> (Y, logdet)`` and ``to_x(Y) -> (X, logdet)``; in both
    directions ``logdet`` is log |det dy/dx| at the point."""
    if "unravel" not in pt or "flat0" not in pt:
        return None
    d = len(pt["flat0"])
    tree = pt["unravel"](jnp.arange(d, dtype=pt["flat0"].dtype))
    idx = {k: np.asarray(v).astype(int) for k, v in tree.items()}
    types = [q for q in HIER_TYPES if all(f"{s}_{q}" in idx for s in ("mu0", "omega", "tau", "zf", "zj"))]
    if not types:
        return None
    fam = np.asarray(fam_idx, int)
    blocks = []
    for q in types:
        zj = idx[f"zj_{q}"]
        n_g = 1 if zj.ndim == 1 else zj.shape[1]
        if zj.shape[0] != len(fam):
            raise ValueError(f"zj_{q} has {zj.shape[0]} rows for {len(fam)} annotators")
        blocks.append(dict(mu0=int(idx[f"mu0_{q}"]), om=int(idx[f"omega_{q}"]), tau=int(idx[f"tau_{q}"]),
                           zf=idx[f"zf_{q}"].reshape(-1), zj=zj.reshape(-1), fam=np.repeat(fam, n_g)))

    def to_y(X):
        X = np.atleast_2d(np.asarray(X, float))
        Y, ld = X.copy(), np.zeros(len(X))
        for b in blocks:
            muf = X[:, [b["mu0"]]] + np.exp(X[:, [b["om"]]]) * X[:, b["zf"]]
            Y[:, b["zf"]] = muf
            Y[:, b["zj"]] = muf[:, b["fam"]] + np.exp(X[:, [b["tau"]]]) * X[:, b["zj"]]
            ld += len(b["zf"]) * X[:, b["om"]] + len(b["zj"]) * X[:, b["tau"]]
        return Y, ld

    def to_x(Y):
        Y = np.atleast_2d(np.asarray(Y, float))
        X, ld = Y.copy(), np.zeros(len(Y))
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            for b in blocks:
                muf = Y[:, b["zf"]]
                X[:, b["zf"]] = (muf - Y[:, [b["mu0"]]]) / np.exp(Y[:, [b["om"]]])
                X[:, b["zj"]] = (Y[:, b["zj"]] - muf[:, b["fam"]]) / np.exp(Y[:, [b["tau"]]])
                ld += len(b["zf"]) * Y[:, b["om"]] + len(b["zj"]) * Y[:, b["tau"]]
        return X, ld

    return dict(name="centred", to_y=to_y, to_x=to_x, types=types)


def ess_bulk(arr: np.ndarray) -> float:
    """Bulk ESS (arviz, rank-normalised) of a (chains, draws) array."""
    import arviz as az
    a = np.asarray(arr, float)
    if a.ndim == 1:
        a = a[None, :]
    if not np.all(np.isfinite(a)) or np.std(a) == 0:
        return float("nan")
    return float(az.ess(az.convert_to_dataset({"x": a[:, :, None]}), method="bulk")["x"].values[0])


def bridge_sampling(logq_post: np.ndarray, logg_post: np.ndarray, logq_prop: np.ndarray,
                    logg_prop: np.ndarray, n1_eff: float | None = None, tol: float = BRIDGE_TOL,
                    maxiter: int = BRIDGE_MAXITER) -> dict:
    """Iterative bridge-sampling estimate of log Z (Meng & Wong 1996; Gronau et al. 2017).

    logq = log of the unnormalised target, logg = log of the normalised proposal, at posterior draws
    (``*_post``, N1 of them, effective size ``n1_eff``) and proposal draws (``*_prop``, N2).  Every
    sum is evaluated in log space; non-finite log q (proposal draws outside the support) count as 0."""
    l1 = np.asarray(logq_post, float) - np.asarray(logg_post, float)
    l2 = np.asarray(logq_prop, float) - np.asarray(logg_prop, float)
    l2 = np.where(np.isfinite(l2), l2, -np.inf)
    if not np.all(np.isfinite(l1)):
        raise ValueError("non-finite log q at posterior draws")
    N1 = float(len(l1) if n1_eff is None else min(n1_eff, len(l1)))
    N2 = float(len(l2))
    ls1, ls2 = np.log(N1 / (N1 + N2)), np.log(N2 / (N1 + N2))
    lstar = float(np.median(l1))
    a1, a2 = l1 - lstar, l2 - lstar
    logr, it, conv = 0.0, 0, False
    for it in range(1, maxiter + 1):
        num = logmeanexp(a2 - np.logaddexp(ls1 + a2, ls2 + logr))
        den = logmeanexp(-np.logaddexp(ls1 + a1, ls2 + logr))
        new = num - den
        if abs(new - logr) < tol:
            logr, conv = new, True
            break
        logr = new
    return dict(log_z=float(logr + lstar), n_iter=it, converged=conv, n1=len(l1), n1_eff=N1, n2=len(l2))


def bridge_re2(logq_post, logg_post, logq_prop, logg_prop, log_z: float, n1_eff: float,
               iat_f2: float = 1.0) -> float:
    """Approximate relative mean-squared error of the bridge estimate (Fruehwirth-Schnatter 2004):
    RE^2 = Var_g(f1) / (N2 E_g(f1)^2) + IAT(f2) Var_p(f2) / (N1 E_p(f2)^2) with
    f1 = p/(s1 p + s2 g) at proposal draws, f2 = g/(s1 p + s2 g) at posterior draws, p = q / Z."""
    lp_post = np.asarray(logq_post) - log_z
    lp_prop = np.where(np.isfinite(logq_prop), np.asarray(logq_prop) - log_z, -np.inf)
    N1, N2 = float(len(lp_post)), float(len(lp_prop))
    N1e = min(n1_eff, N1)
    s1, s2 = N1e / (N1e + N2), N2 / (N1e + N2)
    f1 = np.exp(lp_prop - np.logaddexp(np.log(s1) + lp_prop, np.log(s2) + np.asarray(logg_prop)))
    f2 = np.exp(np.asarray(logg_post) - np.logaddexp(np.log(s1) + lp_post, np.log(s2) + np.asarray(logg_post)))
    t1 = np.var(f1, ddof=1) / (N2 * np.mean(f1) ** 2)
    t2 = iat_f2 * np.var(f2, ddof=1) / (N1 * np.mean(f2) ** 2)
    return float(t1 + t2)


# ----------------------------------------------------------------------------------------
# Potential helpers
# ----------------------------------------------------------------------------------------
def potential(data: lcm.PatternData, model: str = MODEL, prior: dict | None = None, seed: int = 0) -> dict:
    """Potential U (negative log posterior in the unconstrained space of ``lcm.build_model``, as
    minimised by ``lcm.fit_map``) with jitted value / gradient / Hessian, a batched (vmapped) value
    and a constrained postprocess."""
    model_fn = lcm.build_model(data, lcm.MODEL_SPECS[model], prior)
    init = initialize_model(jax.random.PRNGKey(seed), model_fn,
                            init_strategy=numpyro.infer.init_to_uniform(radius=2.0))
    flat0, unravel = ravel_pytree(init.param_info.z)
    f = lambda x: init.potential_fn(unravel(x))  # noqa: E731
    pt = dict(model=model_fn, init=init, unravel=unravel, flat0=flat0, names=list(init.param_info.z.keys()),
              pot=jax.jit(f), grad=jax.jit(jax.grad(f)), hess=jax.jit(jax.hessian(f)))
    pt["pot_batch"] = jax.jit(jax.vmap(f))
    pt["post"] = lambda x: {k: np.asarray(v) for k, v in pt["init"].postprocess_fn(pt["unravel"](jnp.asarray(x))).items()}
    return pt


def pot_batch(pt: dict, X: np.ndarray, chunk: int = 250) -> np.ndarray:
    out = []
    for s in range(0, len(X), chunk):
        out.append(np.asarray(pt["pot_batch"](jnp.asarray(X[s:s + chunk]))))
    return np.concatenate(out) if out else np.zeros(0)


def flat_from_constrained(pt: dict, samples: dict, chunk: int = 250) -> np.ndarray:
    """(D, d) unconstrained flat vectors of D constrained samples of the sample sites."""
    names = pt["names"]
    model = pt["model"]

    def one(s):
        z = unconstrain_fn(model, (), {}, s)
        return ravel_pytree(z)[0]

    fb = jax.jit(jax.vmap(one))
    D = len(samples[names[0]])
    out = []
    for s0 in range(0, D, chunk):
        out.append(np.asarray(fb({k: jnp.asarray(samples[k][s0:s0 + chunk]) for k in names})))
    return np.concatenate(out)


def lbfgs(pt: dict, x0: np.ndarray):
    """One L-BFGS run with the settings of ``lcm.fit_map``."""
    return minimize(lambda x: float(pt["pot"](jnp.asarray(x))), np.asarray(x0, float),
                    jac=lambda x: np.asarray(pt["grad"](jnp.asarray(x))), method="L-BFGS-B",
                    options=dict(LBFGS_OPTIONS))


def newton(pt: dict, x: np.ndarray, max_newton: int = 50, gtol: float = 1e-9) -> np.ndarray:
    """Damped Newton descent of U with the exact Hessian (the polish used for the Laplace masses of
    ``fit_map_optima.csv``): stops at max|grad| < gtol, when no step lowers U, or after three steps
    whose decrease is below floating-point resolution."""
    x = np.asarray(x, float)
    f = float(pt["pot"](jnp.asarray(x)))
    stall = 0
    for _ in range(max_newton):
        g = np.asarray(pt["grad"](jnp.asarray(x)))
        if np.max(np.abs(g)) < gtol:
            break
        H = np.asarray(pt["hess"](jnp.asarray(x)))
        H = 0.5 * (H + H.T)
        ev = np.linalg.eigvalsh(H)
        shift = 0.0 if ev.min() > 1e-10 * max(1.0, abs(ev.max())) else (abs(ev.min()) + 1e-8 * max(1.0, abs(ev.max())))
        step = np.linalg.solve(H + shift * np.eye(len(x)), g)
        t, ok = 1.0, False
        for _ in range(40):
            xn = x - t * step
            fn = float(pt["pot"](jnp.asarray(xn)))
            if np.isfinite(fn) and fn <= f:
                ok = True
                break
            t *= 0.5
        if not ok:
            break
        stall = stall + 1 if (f - fn) <= 1e-14 * max(1.0, abs(f)) else 0
        x, f = xn, fn
        if stall >= 3:
            break
    return x


def replay_restart(data: lcm.PatternData, model: str, restart: int, n_restarts: int, seed: int,
                   prior: dict | None = None, pt: dict | None = None) -> np.ndarray:
    """Unconstrained end point of restart ``restart`` of ``lcm.fit_map(data, model, n_restarts, seed)``
    (same PRNG keys, start point and optimiser settings)."""
    pt = potential(data, model, prior) if pt is None else pt
    keys = jax.random.split(jax.random.PRNGKey(seed), n_restarts)
    x0 = np.asarray(jax.random.uniform(keys[restart], pt["flat0"].shape, minval=-2.0, maxval=2.0))
    return np.asarray(lbfgs(pt, x0).x)


def nearest_optimum(chain_cred: np.ndarray, u: str, data: lcm.PatternData) -> tuple[int, float]:
    """Optimum of the primary MAP fit (snapshot ``optima.csv`` / ``cred_restarts.npy``) whose
    credibility is closest (edge-weighted mean absolute difference over observed patterns)."""
    d = map_dir(u)
    opt = pd.read_csv(d / "optima.csv")
    cr = np.load(d / "cred_restarts.npy")
    obs = observed(data)
    dist_ = [np.average(np.abs(chain_cred[obs] - cr[int(str(r).split()[0])][obs]), weights=data.n[obs])
             for r in opt["restarts"]]
    k = int(np.argmin(dist_))
    return k, float(dist_[k])


def laplace_at(pt: dict, x: np.ndarray, polish: bool = True) -> dict:
    """Newton-polished optimum and Laplace log mass log Z = -U + d/2 log 2pi - 1/2 log det H."""
    if polish:
        x = newton(pt, np.asarray(x, float))
    H = np.asarray(pt["hess"](jnp.asarray(x)))
    H = 0.5 * (H + H.T)
    ev = np.linalg.eigvalsh(H)
    ldet = float(np.sum(np.log(ev))) if ev.min() > 0 else float("nan")
    lp = -float(pt["pot"](jnp.asarray(x)))
    return dict(x=np.asarray(x), log_post=lp, log_det=ldet, min_eig=float(ev.min()),
                laplace_log_mass=lp + 0.5 * len(x) * np.log(2 * np.pi) - 0.5 * ldet)


def group_optima(lp: np.ndarray, tol: float = MODE_TOL) -> list:
    """Restart indices grouped into distinct optima (sorted by log posterior, best first)."""
    lp = np.asarray(lp, float)
    order = np.argsort(-lp, kind="mergesort")
    groups, cur = [], [int(order[0])]
    for i in order[1:]:
        if lp[cur[0]] - lp[i] <= tol:
            cur.append(int(i))
        else:
            groups.append(cur)
            cur = [int(i)]
    groups.append(cur)
    return groups


def assign_optimum(lp: float, lp_optima, tol: float = BASIN_ASSIGN_TOL) -> int:
    d = np.abs(np.asarray(lp_optima, float) - lp)
    k = int(np.argmin(d))
    return k if d[k] <= tol else -1


def wmae(c: np.ndarray, ref: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Edge-weighted mean absolute difference of each row of c (D, P) from ref (P,)."""
    return np.abs(np.atleast_2d(c) - ref[None, :]) @ w / w.sum()


# ----------------------------------------------------------------------------------------
# Optima of the primary MAP fit
# ----------------------------------------------------------------------------------------
def primary_optima(u: str, pt: dict | None = None, data: lcm.PatternData | None = None) -> tuple:
    """Polished optima of the primary Family-HB MAP fit (``optima_params.npz``) with log posterior,
    constrained parameters, credibility and the Laplace numbers of ``optima.csv``."""
    data = load_data(u) if data is None else data
    pt = potential(data) if pt is None else pt
    d = map_dir(u)
    opt = pd.read_csv(d / "optima.csv")
    xs = np.load(d / "optima_params.npz")
    out = []
    for k in range(len(opt)):
        x = np.asarray(xs[f"x{k}"], float)
        s = pt["post"](x)
        out.append(dict(k=k, x=x, log_post=-float(pt["pot"](jnp.asarray(x))), sample=s,
                        cred=lcm.credibility_at(s, data), w=np.asarray(s["w"]),
                        laplace_log_mass=float(opt.laplace_log_mass[k]),
                        laplace_rel_mass=float(opt.laplace_rel_mass[k]), log_post_restart=float(opt.log_post[k]),
                        n_restarts=int(opt.n_restarts[k])))
    return data, pt, out


def snapshot_check(tol_x: float = 1e-5, tol_lp: float = 1e-5, tol_theta: float = 1e-4) -> pd.DataFrame:
    """The frozen snapshot of the primary MAP fits against the canonical fits (which ``fit_primary``
    re-polished after the snapshot was taken): positions, log posteriors and Laplace masses of the
    top-2 optima and the best-fit parameters.  Raises if they differ beyond the tolerances, because
    the per-mode NUTS runs, the simulation truth and the sensitivity comparisons would then be stale."""
    rows = []
    for u in UNIVERSES:
        ds, dc = SNAP / u / MODEL, fit_dir(u, MODEL)
        if not (ds / "optima.csv").exists() or not (dc / "optima.csv").exists():
            continue
        os_, oc = pd.read_csv(ds / "optima.csv"), pd.read_csv(dc / "optima.csv")
        xs, xc = np.load(ds / "optima_params.npz"), np.load(dc / "optima_params.npz")
        ps, pc = np.load(ds / "params.npz"), np.load(dc / "params.npz")
        row = dict(universe=u, n_optima_snapshot=len(os_), n_optima_canonical=len(oc),
                   max_abs_dx_top2=float(max(np.max(np.abs(xs[f"x{k}"] - xc[f"x{k}"])) for k in MODES)),
                   max_abs_dlogpost_top2=float(np.max(np.abs(os_.log_post[:2].to_numpy() - oc.log_post[:2].to_numpy()))),
                   max_abs_dlaplace_top2=float(np.max(np.abs(os_.laplace_log_mass[:2].to_numpy()
                                                              - oc.laplace_log_mass[:2].to_numpy()))),
                   max_abs_dtheta_best=float(np.max(np.abs(ps["theta"] - pc["theta"]))),
                   max_abs_dw_best=float(np.max(np.abs(ps["w"] - pc["w"]))),
                   max_abs_dcred_best=float(np.max(np.abs(np.load(ds / "cred.npy") - np.load(dc / "cred.npy")))))
        row["consistent"] = bool(row["n_optima_snapshot"] == row["n_optima_canonical"] and row["max_abs_dx_top2"] <= tol_x
                                 and row["max_abs_dlogpost_top2"] <= tol_lp and row["max_abs_dlaplace_top2"] <= 1e-3
                                 and row["max_abs_dtheta_best"] <= tol_theta)
        rows.append(row)
    df = pd.DataFrame(rows)
    if len(df) and not df.consistent.all():
        raise RuntimeError("primary MAP snapshot differs from the canonical fits:\n" + df.to_string())
    return df


# ----------------------------------------------------------------------------------------
# (1a) NUTS restricted to one mode
# ----------------------------------------------------------------------------------------
def mode_init_params(pt: dict, x_m: np.ndarray, lp_m: float, num_chains: int, seed: int,
                     overdispersion: float = INIT_OVERDISPERSION, max_tries: int = 30) -> tuple[dict, dict]:
    """Basin-checked dispersed starts around optimum x_m: x = x_m + od * L z (L L' = H^-1); a candidate is
    accepted only if L-BFGS from it ends within ``BASIN_TOL`` of the optimum's log posterior."""
    H = np.asarray(pt["hess"](jnp.asarray(x_m)))
    H = 0.5 * (H + H.T)
    ev, V = np.linalg.eigh(H)
    L = V / np.sqrt(np.maximum(ev, 1e-8))[None, :]
    rng = np.random.default_rng(seed)
    X, lp0, tries, ends = [], [], [], []
    for c in range(num_chains):
        for t in range(max_tries):
            x = x_m + overdispersion * L @ rng.standard_normal(len(x_m))
            end = -float(lbfgs(pt, x).fun)
            if abs(end - lp_m) <= BASIN_TOL:
                break
        else:
            raise RuntimeError(f"no start in the basin of the optimum after {max_tries} draws")
        X.append(x)
        lp0.append(-float(pt["pot"](jnp.asarray(x))))
        tries.append(t + 1)
        ends.append(end)
    trees = [pt["unravel"](jnp.asarray(x)) for x in X]
    init = {k: jnp.stack([t_[k] for t_ in trees]) for k in trees[0]}
    rec = dict(init="mode_laplace", overdispersion=overdispersion, optimum_log_post=lp_m, init_log_post=lp0,
               basin_check_end_log_post=ends, draws_per_chain=tries, basin_tol=BASIN_TOL)
    return init, rec


def run_mode_nuts(u: str, m: int, force: bool = False, settings: dict | None = None,
                  root: Path | None = None, basin_checks: int | None = None) -> dict:
    """NUTS restricted to optimum m of the primary MAP fit; contract files + unconstrained draws +
    per-chain mode checks in ``fits/<u>/<variant>/family_hb_nuts_mode<m>/`` (``root`` replaces
    ``outputs/r2/fits`` for tests)."""
    from numpyro.infer import MCMC, NUTS
    out = (root or FITS) / u / VARIANT / f"{MODEL}_nuts_mode{m}"
    kw = dict(settings or NUTS_SETTINGS[u])
    if not force and (out / "meta.json").exists() and (out / "draws_unconstrained.npz").exists():
        meta = json.loads((out / "meta.json").read_text())
        if all(meta.get(k) == v for k, v in kw.items()):
            log(f"[nuts] skip {u} mode{m}")
            return meta
    t0 = time.time()
    data, pt, opts = primary_optima(u)
    # determinism / code-drift guard: the stored optimum reproduces its log posterior
    lp_stored = json.loads((map_dir(u) / "meta.json").read_text())["best_log_post"]
    if abs(opts[0]["log_post"] - lp_stored) > 1e-2:
        raise RuntimeError(f"{u}: polished optimum 0 log posterior {opts[0]['log_post']} vs stored {lp_stored}")
    om = opts[m]
    seed = derived_seed("posterior_modes_nuts", u, m)
    init_params, init_rec = mode_init_params(pt, om["x"], om["log_post"], kw["num_chains"], seed)
    log(f"[nuts] {u} mode{m}: settings {kw}; starts {np.round(init_rec['init_log_post'], 1).tolist()} "
        f"(optimum {om['log_post']:.2f})")
    model = pt["model"]
    kernel = NUTS(model, target_accept_prob=TARGET_ACCEPT)
    mcmc = MCMC(kernel, num_warmup=kw["num_warmup"], num_samples=kw["num_samples"], num_chains=kw["num_chains"],
                chain_method="sequential", progress_bar=False)
    mcmc.run(jax.random.PRNGKey(seed), extra_fields=("diverging", "potential_energy"), init_params=init_params)
    runtime = time.time() - t0
    sc = mcmc.get_samples(group_by_chain=True)
    ef = mcmc.get_extra_fields(group_by_chain=True)
    nc, ns = kw["num_chains"], kw["num_samples"]
    S = {k: np.asarray(v).reshape((nc * ns,) + np.asarray(v).shape[2:]) for k, v in sc.items()}
    pe = np.asarray(ef["potential_energy"]).reshape(-1)
    div = int(np.asarray(ef["diverging"]).sum())
    chain = np.repeat(np.arange(nc), ns)
    X = flat_from_constrained(pt, {k: S[k] for k in pt["names"]})
    U = pot_batch(pt, X)
    max_pe_diff = float(np.max(np.abs(U - pe)))
    if max_pe_diff > 1e-6 * max(1.0, float(np.max(np.abs(pe)))):
        raise RuntimeError(f"unconstrained draws do not reproduce the potential energy ({max_pe_diff})")
    cred = np.asarray(S["cred"])
    obs = observed(data)
    C.ensure_dir(out)
    cm = cred.mean(0)
    np.save(out / "cred.npy", cm)
    np.save(out / "cred_mean.npy", cm)
    np.save(out / "cred_lo.npy", np.quantile(cred, 0.025, axis=0))
    np.save(out / "cred_hi.npy", np.quantile(cred, 0.975, axis=0))
    np.save(out / "cred_sd.npy", cred.std(0))
    D = cred.shape[0]
    stride = max(1, int(math.ceil(D / MAX_DRAWS_STORED)))
    idx = np.arange(0, D, stride)
    np.savez(out / "draws.npz", chain=chain[idx], draw_index=idx,
             **{k: np.asarray(v)[idx] for k, v in S.items() if k != "cred"})
    np.savez(out / "draws_unconstrained.npz", x=X, potential_energy=pe, chain=chain)
    np.savez(out / "params.npz", **{k: np.asarray(v).mean(0) for k, v in S.items() if k != "cred"})
    # diagnostics (sample sites, theta, credibility over observed patterns)
    diag = []
    for k, v in sc.items():
        if k in ("cred", "theta"):
            continue
        arr = np.asarray(v)
        diag.append(dict(site=k, **dict(zip(("r_hat_max", "ess_bulk_min", "ess_tail_min"),
                                            chain_diag(arr.reshape(nc, ns, -1))))))
    diag.append(dict(site="theta", **dict(zip(("r_hat_max", "ess_bulk_min", "ess_tail_min"),
                                              chain_diag(np.asarray(S["theta"]).reshape(nc, ns, -1))))))
    cobs = cred[:, obs]
    ok = cobs.std(0) > 0
    diag.append(dict(site="cred_observed", **dict(zip(("r_hat_max", "ess_bulk_min", "ess_tail_min"),
                                                      chain_diag(cobs[:, ok].reshape(nc, ns, -1))))))
    diag = pd.DataFrame(diag)
    diag.to_csv(out / "diagnostics.csv", index=False)
    ident = diag[diag.site.isin(IDENTIFIED_SITES)]
    # chain-in-mode checks: (a) nearest optimum per draw by credibility, (b) basin of attraction
    lp_opt = [o["log_post"] for o in opts]
    near = np.empty(D, int)
    for s0 in range(0, D, 500):
        dist_ = np.stack([wmae(cred[s0:s0 + 500][:, obs], o["cred"][obs], data.n[obs]) for o in opts], axis=1)
        near[s0:s0 + 500] = np.argmin(dist_, axis=1)
    nb = BASIN_CHECKS_PER_CHAIN[u] if basin_checks is None else basin_checks
    rows = []
    for c in range(nc):
        sl = np.flatnonzero(chain == c)
        pick = sl[np.linspace(0, len(sl) - 1, nb).astype(int)]
        ends = [-float(lbfgs(pt, X[i]).fun) for i in pick]
        basin = [assign_optimum(e, lp_opt) for e in ends]
        cmean = cred[sl].mean(0)
        k_near, d_near = nearest_optimum(cmean, u, data)
        rows.append(dict(chain=c, mean_log_post=float(-pe[sl].mean()), w_mean=S["w"][sl].mean(0).tolist(),
                         w_T_mean=float(S["w"][sl, 0].mean()), frac_draws_nearest_own_mode=float(np.mean(near[sl] == m)),
                         nearest_map_optimum_chain_mean=k_near, dist_nearest_map_optimum=d_near,
                         n_basin_checks=nb, n_basin_own_mode=int(np.sum(np.asarray(basin) == m)),
                         basin_optima=basin, basin_end_log_post=ends))
    chains = pd.DataFrame(rows)
    chains.to_csv(out / "chains.csv", index=False)
    stays = bool((chains.frac_draws_nearest_own_mode == 1.0).all() and (chains.n_basin_own_mode == nb).all())
    meta = dict(fit_id=f"{MODEL}_nuts_mode{m}", model=MODEL, inference="NUTS", universe=u, variant=VARIANT,
                mode=m, class_names=list(pt["model"].class_names), annotators=list(data.annotators),
                family=list(data.family), families=list(data.families),
                seed=seed, target_accept=TARGET_ACCEPT, chain_method="sequential", init=init_rec,
                n_divergent=div, runtime_s=runtime, n_draws=int(D), draws_stored=int(len(idx)), draw_stride=stride,
                max_abs_potential_energy_diff=max_pe_diff,
                convergence=dict(r_hat_max=float(diag.r_hat_max.max()), ess_bulk_min=float(diag.ess_bulk_min.min()),
                                 r_hat_max_identified=float(ident.r_hat_max.max()),
                                 ess_bulk_min_identified=float(ident.ess_bulk_min.min()),
                                 worst_site=str(diag.sort_values("r_hat_max").site.iloc[-1])),
                chains_stay_in_mode=stays, optimum_log_post=om["log_post"],
                optima_log_post=lp_opt, n_patterns=int(data.P), n_patterns_observed=int(obs.sum()),
                map_source=str(map_dir(u).relative_to(C.OUT)),
                versions=versions(), code="src/r2/posterior_modes.py", analysis_status=STATUS,
                created=time.strftime("%Y-%m-%d %H:%M:%S"), **kw)
    C.write_json(_jsonable(meta), out / "meta.json")
    log(f"[nuts] {u} mode{m}: R-hat {meta['convergence']['r_hat_max']:.4f} (identified "
        f"{meta['convergence']['r_hat_max_identified']:.4f}) div {div}; stays in mode {stays}; "
        f"{runtime / 60:.1f} min")
    return meta


# ----------------------------------------------------------------------------------------
# (1b) bridge sampling
# ----------------------------------------------------------------------------------------
def mode_draws(u: str, m: int, pt: dict, source: str, root: Path | None = None) -> dict:
    """Unconstrained draws of the NUTS run restricted to mode m.

    ``source = 'rerun'``: ``family_hb_nuts_mode<m>`` of this module (every draw, with its potential
    energy); ``source = 'primary_canonical'`` (mode 0 only): the canonical primary dominant-mode NUTS
    fit ``family_hb_nuts`` of ``fit_primary`` (Laplace-initialised at optimum 0; 1,000 thinned draws of
    all sample sites); ``source = 'primary_snapshot'`` (mode 0 only): the frozen, superseded first-pass
    copy of that fit (kept for provenance; not used by any table)."""
    if source == "rerun":
        d = (root or FITS) / u / VARIANT / f"{MODEL}_nuts_mode{m}"
        z = np.load(d / "draws_unconstrained.npz")
        X, chain = np.asarray(z["x"]), np.asarray(z["chain"])
        logq = -pot_batch(pt, X)
        if np.max(np.abs(-logq - np.asarray(z["potential_energy"]))) > 1e-6 * max(1.0, float(np.abs(logq).max())):
            raise RuntimeError(f"{u} mode{m}: potential differs from the stored potential energy")
        meta = json.loads((d / "meta.json").read_text())
        return dict(X=X, chain=chain, logq=logq, source=source, dir=str(d.relative_to(C.OUT) if root is None else d),
                    settings=f"{meta['num_chains']}x{meta['num_warmup']}/{meta['num_samples']}", thin=1,
                    n_divergent=meta["n_divergent"], stays=meta["chains_stay_in_mode"])
    if source not in ("primary_canonical", "primary_snapshot") or m != 0:
        raise ValueError(source)
    d = fit_dir(u, "family_hb_nuts") if source == "primary_canonical" else SNAP / u / "family_hb_nuts"
    z = dict(np.load(d / "draws.npz"))
    meta = json.loads((d / "meta.json").read_text())
    X = flat_from_constrained(pt, {k: np.asarray(z[k]) for k in pt["names"]})
    return dict(X=X, chain=np.asarray(z["chain"]), logq=-pot_batch(pt, X), source=source,
                dir=str(d.relative_to(C.OUT)), settings=f"{meta['num_chains']}x{meta['num_warmup']}/{meta['num_samples']}",
                thin=int(meta.get("draw_stride", 1)), n_divergent=meta.get("n_divergent"),
                w=np.asarray(z["w"]), theta=np.asarray(z["theta"]))


def draw_mode_checks(u: str, m: int, pt: dict, md: dict, data: lcm.PatternData, opts: list, nb: int) -> dict:
    """For draws without the checks of ``run_mode_nuts`` (primary snapshot): nearest optimum of every draw
    by credibility and the optimum L-BFGS reaches from ``nb`` evenly spaced draws per chain."""
    obs = observed(data)
    cred = np.stack([lcm.credibility_at(dict(w=md["w"][i], theta=md["theta"][i]), data) for i in range(len(md["w"]))])
    dist_ = np.stack([wmae(cred[:, obs], o["cred"][obs], data.n[obs]) for o in opts], axis=1)
    near = np.argmin(dist_, axis=1)
    lp_opt = [o["log_post"] for o in opts]
    res = dict(frac_draws_nearest_own_mode=float(np.mean(near == m)), n_basin_checks=0, n_basin_own_mode=0,
               chain_w_T_means=[], basin_optima=[])
    for c in np.unique(md["chain"]):
        sl = np.flatnonzero(md["chain"] == c)
        pick = sl[np.linspace(0, len(sl) - 1, nb).astype(int)]
        basin = [assign_optimum(-float(lbfgs(pt, md["X"][i]).fun), lp_opt) for i in pick]
        res["n_basin_checks"] += nb
        res["n_basin_own_mode"] += int(np.sum(np.asarray(basin) == m))
        res["basin_optima"].append(basin)
        res["chain_w_T_means"].append(float(md["w"][sl, 0].mean()))
    res["stays"] = bool(res["frac_draws_nearest_own_mode"] == 1.0 and res["n_basin_own_mode"] == res["n_basin_checks"])
    return res


def bridge_from_draws(u: str, m: int, pt: dict, md: dict, n_splits: int = N_SPLITS, coords: dict | None = None,
                      proposal: str = "mvn", n2_mult: int = 1) -> dict:
    """Bridge-sampling estimate of log Z of the mode from its draws (see module docstring).

    ``coords``: ``None`` = the proposal is fitted in the unconstrained coordinates x of the draws;
    otherwise a bijection (``centred_coords``) and the proposal is fitted in y = to_y(x), the target
    density being q_y(y) = q_x(x(y)) / |det dy/dx|.  ``proposal``: 'mvn' (moment-matched multivariate
    normal) or 'mvt' (multivariate Student-t, ``T_PROPOSAL_DF`` df, same mean and covariance).
    ``n2_mult``: proposal draws per posterior draw of the estimation half."""
    X, chain, logq = md["X"], md["chain"], md["logq"]
    chains = np.unique(chain)
    ns = int(np.sum(chain == chains[0]))
    if not all(int(np.sum(chain == c)) == ns for c in chains):
        raise ValueError("unequal chain lengths")
    order = np.argsort(chain, kind="mergesort")
    X, chain, logq = X[order], chain[order], logq[order]
    if coords is None:
        Y, to_x = X, None
    else:
        Y, ld = coords["to_y"](X)
        logq, to_x = logq - ld, coords["to_x"]
    half = ns // 2
    pos = np.concatenate([np.arange(ns)] * len(chains))
    fit_m, est_m = pos < half, pos >= half
    rng = np.random.default_rng(derived_seed("bridge", u, m, md["source"], "" if coords is None else coords["name"],
                                             proposal, n2_mult))
    d = Y.shape[1]

    def one(fit_mask, est_mask, n1_eff, rng_):
        if fit_mask.sum() < 4 * d:
            raise ValueError(f"{int(fit_mask.sum())} draws to fit a {d}-dimensional proposal")
        mu, L = mvn_fit(Y[fit_mask])
        Ye = Y[est_mask]
        Zp = rng_.standard_normal((int(est_mask.sum()) * int(n2_mult), d))
        if proposal == "mvt":
            Ls = mvt_scale(L, T_PROPOSAL_DF)
            g = rng_.chisquare(T_PROPOSAL_DF, len(Zp)) / T_PROPOSAL_DF
            Yp = mu + (Zp / np.sqrt(g)[:, None]) @ Ls.T
            lge, lgp = mvt_logpdf(Ye, mu, Ls, T_PROPOSAL_DF), mvt_logpdf(Yp, mu, Ls, T_PROPOSAL_DF)
        elif proposal == "mvn":
            Yp = mu + Zp @ L.T
            lge, lgp = mvn_logpdf(Ye, mu, L), mvn_logpdf(Yp, mu, L)
        else:
            raise ValueError(proposal)
        if to_x is None:
            lqp = -pot_batch(pt, Yp)
        else:
            Xp, ldp = to_x(Yp)
            ok = np.all(np.isfinite(Xp), axis=1) & np.isfinite(ldp)
            lqp = np.full(len(Yp), -np.inf)
            if ok.any():
                lqp[ok] = -pot_batch(pt, Xp[ok]) - ldp[ok]
        lqp = np.where(np.isfinite(lqp), lqp, -np.inf)
        return bridge_sampling(logq[est_mask], lge, lqp, lgp, n1_eff=n1_eff), (logq[est_mask], lge, lqp, lgp)

    Ye3 = Y[est_m].reshape(len(chains), ns - half, -1)
    n1_eff = float(np.nanmedian([ess_bulk(Ye3[:, :, j]) for j in range(d)]))
    res, (lq1, lg1, lq2, lg2) = one(fit_m, est_m, n1_eff, rng)
    s1 = min(n1_eff, len(lq1)) / (min(n1_eff, len(lq1)) + len(lq2))
    f2 = np.exp(lg1 - np.logaddexp(np.log(s1) + lq1 - res["log_z"], np.log(1 - s1) + lg1))
    ess_f2 = ess_bulk(f2.reshape(len(chains), ns - half))
    iat = float(len(f2) / ess_f2) if np.isfinite(ess_f2) and ess_f2 > 0 else 1.0
    re2 = bridge_re2(lq1, lg1, lq2, lg2, res["log_z"], n1_eff, iat)
    split_lz, split_conv = [], []
    ratio = n1_eff / est_m.sum()
    for _ in range(n_splits):
        perm = rng.permutation(len(Y))
        fm = np.zeros(len(Y), bool)
        fm[perm[: len(Y) // 2]] = True
        r_ = one(fm, ~fm, ratio * (~fm).sum(), rng)[0]
        split_lz.append(r_["log_z"])
        split_conv.append(r_["converged"])
    split_lz = np.asarray(split_lz)
    sd = float(split_lz.std(ddof=1)) if n_splits > 1 else float("nan")
    return dict(universe=u, mode=m, source=md["source"], draws_dir=md.get("dir", ""),
                nuts_settings=md.get("settings", ""), draw_thinning=md.get("thin", 1),
                coords="unconstrained_noncentred" if coords is None else coords["name"], proposal=proposal,
                n2_mult=int(n2_mult), log_z=res["log_z"], n_iter=res["n_iter"], converged=res["converged"],
                n1=res["n1"], n1_eff=n1_eff, n2=res["n2"], re2_fs=re2, re_fs=float(np.sqrt(re2)), iat_f2=iat,
                frac_proposal_finite=float(np.mean(np.isfinite(lq2))),
                log_z_splits=split_lz.tolist(), n_splits_converged=int(np.sum(split_conv)),
                log_z_split_mean=float(split_lz.mean()) if n_splits else float("nan"),
                log_z_split_median=float(np.median(split_lz)) if n_splits else float("nan"), log_z_split_sd=sd,
                log_z_split_min=float(split_lz.min()) if n_splits else float("nan"),
                log_z_split_max=float(split_lz.max()) if n_splits else float("nan"),
                n_splits=n_splits, d=int(d), n_draws=int(len(Y)),
                max_log_q_post=float(md["logq"].max()), mean_log_q_post=float(md["logq"].mean()))


def bridge_mode(u: str, m: int, pt: dict, n_splits: int = N_SPLITS, root: Path | None = None,
                source: str = "rerun", coords: dict | None = None, proposal: str = "mvn", n2_mult: int = 1) -> dict:
    return bridge_from_draws(u, m, pt, mode_draws(u, m, pt, source, root), n_splits, coords, proposal, n2_mult)


def mode0_source(u: str) -> str:
    """Draws used for the mode-0 bridge estimate: this module's rerun restricted to optimum 0 (every
    draw stored) where it exists, else the 1,000 thinned draws of the canonical primary NUTS fit."""
    return "rerun" if (fit_dir(u, f"{MODEL}_nuts_mode0") / "draws_unconstrained.npz").exists() else "primary_canonical"


def _p2(lz0: float, lz1: float) -> float:
    return float(1.0 / (1.0 + np.exp(np.clip(lz0 - lz1, -700, 700))))


def run_bridge(universes=UNIVERSES, force: bool = False) -> dict:
    """Bridge estimates per universe.  Primary estimator: multivariate normal proposal fitted in the
    centred coordinates, N2 = ``N2_MULT`` N1, ``N_SPLITS`` repeated splits.  Checks (``N_SPLITS_CHECK``
    splits): the same estimator on the canonical primary NUTS draws (mode 0, independent chains),
    a Student-t proposal in the centred coordinates, and the proposal fitted in the raw non-centred
    unconstrained coordinates with N2 = N1 (the textbook recipe; reported because it fails here)."""
    od = C.ensure_dir(MODOUT / "bridge")
    out = {}
    for u in universes:
        f = od / f"{u}.json"
        if f.exists() and not force:
            rec = json.loads(f.read_text())
            if rec.get("schema") == 2:
                out[u] = rec
                continue
        if not (fit_dir(u, f"{MODEL}_nuts_mode1") / "draws_unconstrained.npz").exists():
            log(f"[bridge] {u}: mode-1 NUTS draws missing; skipped")
            continue
        t0 = time.time()
        data, pt, opts = primary_optima(u)
        cc = centred_coords(pt, data.fam_idx)
        src0 = mode0_source(u)
        md = {0: mode_draws(u, 0, pt, src0), 1: mode_draws(u, 1, pt, "rerun")}
        extra = {}
        for m_ in MODES:
            if md[m_]["source"] == "rerun":
                extra[m_] = dict(n_divergent=md[m_]["n_divergent"], chains_stay_in_mode=md[m_]["stays"])
            else:
                chk = draw_mode_checks(u, m_, pt, md[m_], data, opts, nb=BASIN_CHECKS_PER_CHAIN[u] // 2)
                extra[m_] = dict(n_divergent=md[m_]["n_divergent"], chains_stay_in_mode=chk["stays"],
                                 frac_draws_nearest_own_mode=chk["frac_draws_nearest_own_mode"],
                                 n_basin_checks=chk["n_basin_checks"], n_basin_own_mode=chk["n_basin_own_mode"],
                                 chain_w_T_means=chk["chain_w_T_means"], basin_optima=chk["basin_optima"])
        recs = [dict(bridge_from_draws(u, m_, pt, md[m_], N_SPLITS, cc, "mvn", N2_MULT), role="primary_estimate",
                     **extra[m_]) for m_ in MODES]
        checks = []
        if src0 == "rerun":       # independent chains: the canonical primary NUTS fit (thinned draws)
            mdc = mode_draws(u, 0, pt, "primary_canonical")
            chk = draw_mode_checks(u, 0, pt, mdc, data, opts, nb=BASIN_CHECKS_PER_CHAIN[u] // 2)
            checks.append(dict(bridge_from_draws(u, 0, pt, mdc, N_SPLITS_CHECK, cc, "mvn", N2_MULT),
                               role="mode0_canonical_draws", n_divergent=mdc["n_divergent"],
                               chains_stay_in_mode=chk["stays"],
                               frac_draws_nearest_own_mode=chk["frac_draws_nearest_own_mode"],
                               n_basin_checks=chk["n_basin_checks"], n_basin_own_mode=chk["n_basin_own_mode"],
                               chain_w_T_means=chk["chain_w_T_means"], basin_optima=chk["basin_optima"]))
        for m_ in MODES:
            checks.append(dict(bridge_from_draws(u, m_, pt, md[m_], N_SPLITS_CHECK, cc, "mvt", N2_MULT),
                               role="proposal_student_t", **extra[m_]))
            checks.append(dict(bridge_from_draws(u, m_, pt, md[m_], N_SPLITS_CHECK, None, "mvn", 1),
                               role="raw_noncentred_coordinates", **extra[m_]))
        lz = np.array([r["log_z"] for r in recs])
        p = np.exp(lz - np.logaddexp.reduce(lz))
        ll_split = np.array([r["log_z_splits"] for r in recs])
        p2_split = 1.0 / (1.0 + np.exp(ll_split[0] - ll_split[1]))
        # Monte Carlo SD of the log Z difference: Fruehwirth-Schnatter term (posterior and proposal draws, given
        # the proposal) and the spread over repeated splits (proposal fit), combined conservatively
        sd_fs = float(np.sqrt(recs[0]["re2_fs"] + recs[1]["re2_fs"]))
        sd_split = float(np.sqrt(recs[0]["log_z_split_sd"] ** 2 + recs[1]["log_z_split_sd"] ** 2))
        sd_tot = float(np.sqrt(sd_fs ** 2 + sd_split ** 2))
        diff = float(lz[0] - lz[1])
        lap = np.array([opts[m_]["laplace_log_mass"] for m_ in MODES])
        p_lap = np.exp(lap - np.logaddexp.reduce(lap))
        by_role = {}
        for role in sorted({c["role"] for c in checks}):
            cz = {c["mode"]: c["log_z"] for c in checks if c["role"] == role}
            z0, z1 = cz.get(0, lz[0]), cz.get(1, lz[1])
            by_role[role] = dict(log_z_diff=float(z0 - z1), p2=_p2(z0, z1))
        rec = dict(schema=2, universe=u,
                   estimator=dict(coords=cc["name"] if cc else "unconstrained_noncentred", proposal="mvn",
                                  n2_mult=N2_MULT, n_splits=N_SPLITS), mode0_draws=src0,
                   modes=recs, checks=checks, checks_p2=by_role, p_bridge=p.tolist(), p2_bridge=float(p[1]),
                   log_z_diff_bridge=diff, log_z_diff_sd_fs=sd_fs, log_z_diff_split_sd=sd_split,
                   log_z_diff_sd=sd_tot,
                   p2_bridge_lo=_p2(diff + 1.96 * sd_tot, 0.0), p2_bridge_hi=_p2(diff - 1.96 * sd_tot, 0.0),
                   p2_bridge_split_q025=float(np.quantile(p2_split, 0.025)),
                   p2_bridge_split_q975=float(np.quantile(p2_split, 0.975)),
                   p2_bridge_split_min=float(p2_split.min()), p2_bridge_split_max=float(p2_split.max()),
                   laplace_log_mass=lap.tolist(), p_laplace_top2=p_lap.tolist(), p2_laplace=float(p_lap[1]),
                   p_laplace_all_optima=[o["laplace_rel_mass"] for o in opts],
                   log_z_diff_laplace=float(lap[0] - lap[1]),
                   optima_log_post=[o["log_post"] for o in opts], runtime_s=time.time() - t0,
                   created=time.strftime("%Y-%m-%d %H:%M:%S"), analysis_status=STATUS)
        C.write_json(_jsonable(rec), f)
        raw = by_role["raw_noncentred_coordinates"]
        log(f"[bridge] {u}: log Z {np.round(lz, 2).tolist()} (Laplace {np.round(lap, 2).tolist()}); diff "
            f"{diff:.2f} +- {sd_tot:.2f} (Laplace {lap[0] - lap[1]:.2f}); p2 bridge {p[1]:.3g} "
            f"[{rec['p2_bridge_lo']:.3g}, {rec['p2_bridge_hi']:.3g}] vs Laplace {p_lap[1]:.3g}; checks: "
            + "; ".join(f"{k} p2 {v['p2']:.3g}" for k, v in by_role.items())
            + f" ({time.time() - t0:.0f}s)")
        out[u] = rec
    return out


def bridge_table(universes=UNIVERSES) -> pd.DataFrame:
    rows = []
    for u in universes:
        f = MODOUT / "bridge" / f"{u}.json"
        if not f.exists():
            continue
        b = json.loads(f.read_text())
        if b.get("schema") != 2:
            log(f"[bridge] {u}: stale bridge record (first-pass estimator); rerun the bridge stage")
            continue
        prim = {x["mode"]: x for x in b["modes"]}
        for r in b["modes"] + b["checks"]:
            m, role = r["mode"], r["role"]
            row = dict(universe=u, mode=m, role=role, coords=r["coords"], proposal=r["proposal"], n2_mult=r["n2_mult"],
                       draws_source=r["source"], draws_dir=r["draws_dir"],
                       nuts_settings=r["nuts_settings"], draw_thinning=r["draw_thinning"], n_draws=r["n_draws"],
                       optimum_log_post=b["optima_log_post"][m], gap_to_mode0=b["optima_log_post"][0] - b["optima_log_post"][m],
                       n_divergent=r.get("n_divergent"), chains_stay_in_mode=r.get("chains_stay_in_mode"))
            d = C.OUT / r["draws_dir"]
            meta = json.loads((d / "meta.json").read_text())
            cv = meta.get("convergence", {})
            if r["source"] == "rerun":
                ch = pd.read_csv(d / "chains.csv")
                row.update(r_hat_max_all=cv["r_hat_max"], r_hat_max_identified=cv["r_hat_max_identified"],
                           ess_bulk_min_all=cv.get("ess_bulk_min"),
                           ess_bulk_min_identified=cv["ess_bulk_min_identified"], worst_site=cv["worst_site"],
                           runtime_min=meta["runtime_s"] / 60,
                           min_frac_draws_nearest_own_mode=float(ch.frac_draws_nearest_own_mode.min()),
                           n_basin_checks=int(ch.n_basin_checks.sum()), n_basin_own_mode=int(ch.n_basin_own_mode.sum()),
                           chain_w_T_means=";".join(f"{v:.5f}" for v in ch.w_T_mean))
            else:
                row.update(r_hat_max_all=cv.get("r_hat_max"),
                           r_hat_max_identified=np.nanmax([cv.get("r_hat_max_theta", np.nan),
                                                           cv.get("r_hat_max_cred", np.nan)]),
                           ess_bulk_min_all=cv.get("ess_bulk_min"), ess_bulk_min_identified=cv.get("ess_bulk_min_cred"),
                           worst_site=cv.get("worst_site"), runtime_min=meta.get("runtime_s", np.nan) / 60,
                           min_frac_draws_nearest_own_mode=r.get("frac_draws_nearest_own_mode"),
                           n_basin_checks=r.get("n_basin_checks"), n_basin_own_mode=r.get("n_basin_own_mode"),
                           chain_w_T_means=";".join(f"{v:.5f}" for v in r.get("chain_w_T_means", [])))
            row.update(bridge_log_z=r["log_z"], bridge_re_fs=r["re_fs"], bridge_log_z_split_sd=r["log_z_split_sd"],
                       bridge_log_z_split_median=r["log_z_split_median"], bridge_log_z_split_min=r["log_z_split_min"],
                       bridge_log_z_split_max=r["log_z_split_max"], n_splits=r["n_splits"],
                       n_splits_converged=r["n_splits_converged"], bridge_n_iter=r["n_iter"],
                       bridge_converged=r["converged"], bridge_n1=r["n1"], bridge_n1_eff=r["n1_eff"], bridge_n2=r["n2"],
                       frac_proposal_finite=r["frac_proposal_finite"], laplace_log_mass=b["laplace_log_mass"][m],
                       bridge_minus_laplace_log_z=r["log_z"] - b["laplace_log_mass"][m])
            if role == "primary_estimate":
                row.update(p_bridge=b["p_bridge"][m], p_laplace_top2=b["p_laplace_top2"][m],
                           p_laplace_all_optima=b["p_laplace_all_optima"][m],
                           p2_bridge_lo=b["p2_bridge_lo"] if m == 1 else np.nan,
                           p2_bridge_hi=b["p2_bridge_hi"] if m == 1 else np.nan,
                           p2_bridge_split_min=b["p2_bridge_split_min"] if m == 1 else np.nan,
                           p2_bridge_split_max=b["p2_bridge_split_max"] if m == 1 else np.nan,
                           log_z_diff_bridge=b["log_z_diff_bridge"], log_z_diff_laplace=b["log_z_diff_laplace"],
                           log_z_diff_sd=b["log_z_diff_sd"], log_z_diff_sd_fs=b["log_z_diff_sd_fs"],
                           log_z_diff_split_sd=b["log_z_diff_split_sd"])
                if m == 1:
                    # uncertainty across draw sets and proposals (modes verifier m2): the primary estimate, the
                    # independent canonical mode-0 chains and the Student-t proposal (the raw non-centred recipe
                    # failed its own diagnostics and is excluded); wider than the within-run split interval
                    cand = [b["p_bridge"][1]] + [b["checks_p2"][k]["p2"] for k in ("mode0_canonical_draws", "proposal_student_t")
                                                 if k in b["checks_p2"]]
                    row.update(p2_range_draw_sets_proposals_lo=float(min(cand)), p2_range_draw_sets_proposals_hi=float(max(cand)),
                               p2_range_note="min/max over primary estimate, " + ", ".join(
                                   k for k in ("mode0_canonical_draws", "proposal_student_t") if k in b["checks_p2"])
                               + "; the p2_bridge_lo/hi interval covers the within-run split / MC error only")
            else:
                row["log_z_minus_primary_estimate"] = r["log_z"] - prim[m]["log_z"]
                row["log_z_diff_bridge"] = b["checks_p2"][role]["log_z_diff"]
                row["p_bridge"] = b["checks_p2"][role]["p2"] if m == 1 else 1.0 - b["checks_p2"][role]["p2"]
            # the project's frozen NUTS flag rule (fit_primary RHAT_FLAG / ESS_BULK_FLAG): a flag, not a gate
            for scope_, rh_, es_ in (("all_sites", row.get("r_hat_max_all"), row.get("ess_bulk_min_all")),
                                     ("identified_sites", row.get("r_hat_max_identified"), row.get("ess_bulk_min_identified"))):
                rh_ = np.nan if rh_ is None else float(rh_)
                es_ = np.nan if es_ is None else float(es_)
                row[f"diag_flag_{scope_}"] = ("flagged" if (rh_ > RHAT_FROZEN or es_ < ESS_FROZEN) else
                                              ("" if np.isfinite(rh_) and np.isfinite(es_) else "not_available"))
            row["analysis_status"] = STATUS
            rows.append(row)
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------------------
# (2) mixture credibility and governance sensitivity
# ----------------------------------------------------------------------------------------
def p2_estimates(with_upper: bool = False) -> tuple:
    """Second-mode mass per universe: bridge estimate, Laplace estimate (and, optionally, the upper
    end of the bridge Monte Carlo interval).  Stale first-pass bridge records are ignored."""
    pb, pl, ph = {}, {}, {}
    for u in UNIVERSES:
        f = MODOUT / "bridge" / f"{u}.json"
        if f.exists():
            b = json.loads(f.read_text())
            if b.get("schema") != 2:
                continue
            pb[u], pl[u], ph[u] = b["p2_bridge"], b["p2_laplace"], b["p2_bridge_hi"]
    return (pb, pl, ph) if with_upper else (pb, pl)


def laplace_p2_top2(u: str) -> float:
    """Laplace mass of optimum 1 relative to the top-2 optima of the primary MAP fit (snapshot)."""
    opt = pd.read_csv(map_dir(u) / "optima.csv")
    lm = opt["laplace_log_mass"].to_numpy(float)[:2]
    return float(np.exp(lm[1] - np.logaddexp.reduce(lm)))


def map_optimum_cred(u: str, k: int) -> np.ndarray:
    """MAP credibility at optimum k of the primary MAP fit (snapshot ``cred_restarts.npy``)."""
    d = map_dir(u)
    opt = pd.read_csv(d / "optima.csv")
    r = int(str(opt["restarts"].iloc[k]).split()[0])
    return np.load(d / "cred_restarts.npy")[r]


def global_mode1_inputs() -> tuple[np.ndarray, str, float, str]:
    """Global-fit mode-1 credibility and p2: NUTS mode-1 mean and the bridge estimate when available,
    otherwise the MAP optimum-1 credibility and the Laplace estimate (flagged in the table)."""
    f = fit_dir("global", f"{MODEL}_nuts_mode1") / "cred_mean.npy"
    bj = MODOUT / "bridge" / "global.json"
    if f.exists() and (fit_dir("global", f"{MODEL}_nuts_mode1") / "meta.json").exists():
        c1, src = np.load(f), "nuts_mode1_mean"
    else:
        c1, src = map_optimum_cred("global", 1), "map_optimum1_fallback"
    if bj.exists() and json.loads(bj.read_text()).get("schema") == 2:
        b = json.loads(bj.read_text())
        return c1, src, float(b["p2_bridge"]), "bridge"
    return c1, src, laplace_p2_top2("global"), "laplace_fallback"


def weighted_quantiles_mixture(A: np.ndarray, B: np.ndarray, pa: float, qs=(0.025, 0.975), chunk: int = 2000):
    """Per-column quantiles of the mixture pa * empirical(A) + (1 - pa) * empirical(B)."""
    out = np.empty((len(qs), A.shape[1]))
    wa, wb = pa / A.shape[0], (1 - pa) / B.shape[0]
    for s in range(0, A.shape[1], chunk):
        V = np.concatenate([A[:, s:s + chunk], B[:, s:s + chunk]], 0)
        W = np.concatenate([np.full(A.shape[0], wa), np.full(B.shape[0], wb)])
        o = np.argsort(V, axis=0, kind="mergesort")
        Vs = np.take_along_axis(V, o, 0)
        cw = np.cumsum(W[o], axis=0)
        for i, q in enumerate(qs):
            j = np.argmax(cw >= q - 1e-12, axis=0)
            out[i, s:s + chunk] = Vs[j, np.arange(Vs.shape[1])]
    return out


def write_mixture_fit(u: str, p2: float) -> None:
    """Store the full-posterior (mixture over the top-2 modes) credibility as ``family_hb_mixture``.
    Mode 0 = the canonical primary dominant-mode NUTS fit ``family_hb_nuts``; mode 1 = this module's
    NUTS run restricted to optimum 1."""
    data = load_data(u)
    d0, d1 = fit_dir(u, "family_hb_nuts"), fit_dir(u, f"{MODEL}_nuts_mode1")
    c0, c1 = np.load(d0 / "cred_mean.npy"), np.load(d1 / "cred_mean.npy")
    cm = (1 - p2) * c0 + p2 * c1
    out = C.ensure_dir(fit_dir(u, f"{MODEL}_mixture"))
    z0, z1 = np.load(d0 / "draws.npz"), np.load(d1 / "draws.npz")
    A = np.stack([lcm.credibility_at(dict(w=z0["w"][i], theta=z0["theta"][i]), data) for i in range(len(z0["w"]))])
    B = np.stack([lcm.credibility_at(dict(w=z1["w"][i], theta=z1["theta"][i]), data) for i in range(len(z1["w"]))])
    lo, hi = weighted_quantiles_mixture(A, B, 1 - p2)
    sd = np.sqrt((1 - p2) * (A.var(0) + (A.mean(0) - cm) ** 2) + p2 * (B.var(0) + (B.mean(0) - cm) ** 2))
    np.save(out / "cred.npy", cm)
    np.save(out / "cred_mean.npy", cm)
    np.save(out / "cred_lo.npy", lo)
    np.save(out / "cred_hi.npy", hi)
    np.save(out / "cred_sd.npy", sd)
    p0 = dict(np.load(d0 / "params.npz"))
    p1 = dict(np.load(d1 / "params.npz"))
    np.savez(out / "params.npz", w=(1 - p2) * p0["w"] + p2 * p1["w"],
             theta=(1 - p2) * p0["theta"] + p2 * p1["theta"], mode_probabilities=np.array([1 - p2, p2]))
    meta = dict(fit_id=f"{MODEL}_mixture", model=MODEL, inference="NUTS per mode + bridge-sampling mode weights",
                universe=u, variant=VARIANT, mode_probabilities=[1 - p2, p2],
                components=[str(d0.relative_to(C.OUT)), str(d1.relative_to(C.OUT))],
                component_nuts_settings=[
                    "{num_chains}x{num_warmup}/{num_samples}".format(**json.loads((d_ / "meta.json").read_text()))
                    for d_ in (d0, d1)],
                component_cred_md5=[_md5(d_ / "cred_mean.npy") for d_ in (d0, d1)],
                cred_files=dict(cred="mixture posterior mean sum_m p_m E[c | mode m]",
                                cred_lo="2.5% quantile of the p-weighted mixture of stored per-mode draws",
                                cred_hi="97.5% quantile", cred_sd="mixture posterior sd"),
                note="params.npz: p-weighted average of the per-mode posterior means (not a posterior mean of a "
                     "single parameter value; credibility is computed per mode)",
                n_patterns=int(data.P), analysis_status=STATUS, code="src/r2/posterior_modes.py",
                created=time.strftime("%Y-%m-%d %H:%M:%S"))
    C.write_json(_jsonable(meta), out / "meta.json")


def primary_nuts_dir(u: str, source: str) -> Path:
    """Primary dominant-mode NUTS fit: 'canonical' (the current ``family_hb_nuts`` of ``fit_primary``)
    or 'snapshot' (superseded first pass, frozen copy; provenance only)."""
    return SNAP / u / "family_hb_nuts" if source == "snapshot" else fit_dir(u, "family_hb_nuts")


def governance_sensitivity(sources=("canonical",)) -> pd.DataFrame:
    """Tier sensitivity to the second-mode mass (see module docstring, part 2).  Uses only the tier
    functions of ``governance.py`` (``pattern_key_map``, ``bfdr_levels``, ``dual_tier``,
    ``realized_bfdr``); every quantity is a pattern-level computation weighted by edge counts, as in
    ``governance.Context``.  Computed for the canonical primary NUTS fit (``sources``)."""
    from r2 import evidence as E
    from r2 import governance as G
    pb, pl, ph = p2_estimates(with_upper=True)
    if not all(t in pb for t in C.TISSUES):
        log("[governance] tissue bridge estimates missing; skipped")
        return pd.DataFrame()
    # global fit: bridge estimate / NUTS mode-1 mean when available, else Laplace / MAP optimum 1 (flagged)
    cg1_glob, g1_src, pg_bridge, pg_src = global_mode1_inputs()
    pb = dict(pb, **{"global": pg_bridge})
    pl = dict(pl, **{"global": pl.get("global", laplace_p2_top2("global"))})
    Rg, _, _, ann_g = E.load_patterns("global", VARIANT)
    gov_tiers = pd.read_csv(C.TABLES / "gov_tiers.csv") if (C.TABLES / "gov_tiers.csv").exists() else None
    rows, seen = [], {}
    for source in sources:
        dg = primary_nuts_dir("global", source)
        if not (dg / "cred_mean.npy").exists():
            continue
        cg0_glob = np.load(dg / "cred_mean.npy")
        sig = tuple(float(np.load(primary_nuts_dir(u, source) / "cred_mean.npy").sum()) for u in UNIVERSES)
        if sig in seen:
            log(f"[governance] {source} primary NUTS identical to {seen[sig]}; skipped")
            continue
        seen[sig] = source
        meta_g = json.loads((dg / "meta.json").read_text())
        for t in C.TISSUES:
            R, n, _, ann = E.load_patterns(t, VARIANT)
            if list(ann) != list(ann_g):
                raise RuntimeError(f"{t}: annotator order differs from the global fit")
            n = n.astype(np.float64)
            gidx = G.pattern_key_map(R, Rg)
            dc = primary_nuts_dir(t, source)
            meta_c = json.loads((dc / "meta.json").read_text())
            cg0, cc0 = cg0_glob[gidx], np.load(dc / "cred_mean.npy")
            cc1 = np.load(fit_dir(t, f"{MODEL}_nuts_mode1") / "cred_mean.npy")
            f0r = fit_dir(t, f"{MODEL}_nuts_mode0") / "cred_mean.npy"
            cc0r = np.load(f0r) if f0r.exists() else None
            cg1 = cg1_glob[gidx]
            lg0, lc0 = G.bfdr_levels(cg0, n), G.bfdr_levels(cc0, n)
            dual0 = G.dual_tier(lg0, lc0, "primary")
            match = np.nan
            if gov_tiers is not None:
                gt = gov_tiers[(gov_tiers.tissue == t) & (gov_tiers.scope == "dual")].set_index("tier")["n_edges"]
                match = bool(int(gt.get("Strong", -1)) == int(n[dual0 == 3].sum())
                             and int(gt.get("Moderate", -1)) == int(n[dual0 == 2].sum()))

            def jac(a, b):
                un = n[a | b].sum()
                return float(n[a & b].sum() / un) if un > 0 else float("nan")

            def add(scenario, p2_ctx, p2_glob, global_rule, cc, cg):
                lg = G.bfdr_levels(cg, n)
                lc = G.bfdr_levels(cc, n)
                dual = G.dual_tier(lg, lc, "primary")
                rows.append(dict(
                    tissue=t, primary_nuts_source=source, scenario=scenario, global_rule=global_rule,
                    p2_context=p2_ctx, p2_global=p2_glob,
                    context_strong=int(n[lc == 2].sum()), context_moderate=int(n[lc == 1].sum()),
                    context_ge_moderate=int(n[lc >= 1].sum()),
                    global_strong=int(n[lg == 2].sum()), global_moderate=int(n[lg == 1].sum()),
                    dual_strong=int(n[dual == 3].sum()), dual_moderate=int(n[dual == 2].sum()),
                    promoted=int(n[dual >= 2].sum()),
                    promoted_definition=("universe-level dual Strong or Moderate (dual tier >= 2), i.e. the promotable set; "
                                         "not the plan section 6 per-method promotion (top-50 per TF), which is a subset"),
                    jaccard_context_strong=jac(lc == 2, lc0 == 2), jaccard_context_ge_moderate=jac(lc >= 1, lc0 >= 1),
                    jaccard_global_strong=jac(lg == 2, lg0 == 2),
                    jaccard_dual_strong=jac(dual == 3, dual0 == 3), jaccard_promoted=jac(dual >= 2, dual0 >= 2),
                    primary_context_strong_bfdr_under_scenario=G.realized_bfdr(cc, n, lc0 == 2),
                    primary_dual_strong_bfdr_under_scenario_context=G.realized_bfdr(cc, n, dual0 == 3),
                    primary_promoted_bfdr_under_scenario_context=G.realized_bfdr(cc, n, dual0 >= 2),
                    n_context_strong_primary=int(n[lc0 == 2].sum()), n_dual_strong_primary=int(n[dual0 == 3].sum()),
                    n_promoted_primary=int(n[dual0 >= 2].sum()), primary_matches_gov_tiers_csv=match,
                    primary_nuts_settings_context=f"{meta_c.get('num_chains')}x{meta_c.get('num_warmup')}/{meta_c.get('num_samples')}",
                    primary_nuts_settings_global=f"{meta_g.get('num_chains')}x{meta_g.get('num_warmup')}/{meta_g.get('num_samples')}",
                    global_mode1_cred_source=g1_src, global_p2_estimate_source=pg_src,
                    primary_nuts_mtime_context=time.strftime(
                        "%Y-%m-%d %H:%M", time.localtime((dc / "cred_mean.npy").stat().st_mtime)),
                    primary_nuts_mtime_global=time.strftime(
                        "%Y-%m-%d %H:%M", time.localtime((dg / "cred_mean.npy").stat().st_mtime)),
                    analysis_status=STATUS))

            add("primary", 0.0, 0.0, "own_estimate", cc0, cg0)
            if cc0r is not None:     # independent rerun of the mode-0 chains: Monte Carlo noise of the tiers
                add("mode0_rerun_mc_reference", 0.0, 0.0, "own_estimate", cc0r, cg0)
            pg = pb["global"]
            for scen, p2, pgl in (("bridge", pb[t], pb["global"]), ("bridge_upper_mc", ph[t], pb["global"]),
                                  ("laplace", pl[t], pl["global"])):
                add(scen, p2, pgl, "own_estimate", (1 - p2) * cc0 + p2 * cc1, (1 - pgl) * cg0 + pgl * cg1)
            for p2 in P2_GRID:
                add(f"fixed_{p2:g}", p2, pg, "own_estimate", (1 - p2) * cc0 + p2 * cc1, (1 - pg) * cg0 + pg * cg1)
                add(f"fixed_{p2:g}", p2, p2, "same_p2", (1 - p2) * cc0 + p2 * cc1, (1 - p2) * cg0 + p2 * cg1)
            add("mode1_only", 1.0, pg, "own_estimate", cc1, (1 - pg) * cg0 + pg * cg1)
        log(f"[governance] {source}: done")
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------------------
# (3) simulation: calibration of mode selection
# ----------------------------------------------------------------------------------------
def multistart(pt: dict, n_restarts: int, seed: int, init_radius: float = 2.0) -> list:
    """The restarts of ``lcm.fit_map(data, model, n_restarts, seed)`` (identical starting points and
    optimiser settings) with the unconstrained optimum of every restart."""
    keys = jax.random.split(jax.random.PRNGKey(seed), n_restarts)
    out = []
    for r in range(n_restarts):
        x0 = np.asarray(jax.random.uniform(keys[r], pt["flat0"].shape, minval=-init_radius, maxval=init_radius))
        res = lbfgs(pt, x0)
        out.append(dict(restart=r, seed=seed, log_post=-float(res.fun), x=np.asarray(res.x),
                        converged=bool(res.success)))
    return out


def set_stats(cred: np.ndarray, n_p: np.ndarray, t_p: np.ndarray) -> dict:
    """Strong (BFDR <= q_strong) and Strong+Moderate (BFDR <= q_moderate) sets of a credibility vector
    (``governance.bfdr_levels``): size, claimed BFDR and realised FDR against the simulated classes."""
    from r2 import governance as G
    lev = G.bfdr_levels(cred, n_p)
    out = {}
    for name, m in (("strong", lev == 2), ("ge_moderate", lev >= 1)):
        ns = float(n_p[m].sum())
        out[f"{name}_n"] = ns
        out[f"{name}_claimed_bfdr"] = float(np.sum(n_p[m] * (1 - cred[m])) / ns) if ns else np.nan
        out[f"{name}_realised_fdr"] = float(np.sum(n_p[m] - t_p[m]) / ns) if ns else np.nan
        out[f"{name}_n_true"] = float(t_p[m].sum())
    return out


def sim_simulator(u: str) -> str:
    """Simulator of simulation label u (``SIM_LABEL_INFO``)."""
    return SIM_LABEL_INFO[u][1]


def sim_seed(u: str, r: int) -> int:
    """Data seed of replicate r of label u; the fixed-mask labels keep the seeds of the stored recovery study."""
    s = sim_simulator(u)
    return derived_seed("recovery", u, r) if s == "fixed_masks" else derived_seed("modes_sim_data", s, u, r)


_STRAT_W: dict = {}


def strat_weights(u: str) -> tuple[np.ndarray, float]:
    """Stratified class weights of label u (generating parameters of u) and the class-T share of the domain."""
    key = SIM_TRUTH[u]
    if key not in _STRAT_W:
        t = truth_params(u)
        des = cc_design(key[0])
        W = stratum_class_weights(des, t["w"], t["theta"])
        sp, n = des["stratum_of_pattern"], des["n"]
        m = (sp >= 0) & (n > 0)
        nk = np.bincount(sp[m], weights=n[m], minlength=len(W))
        _STRAT_W[key] = (W, float(nk @ W[:, 0] / nk.sum()))
    return _STRAT_W[key]


def simulate_for(u: str, r: int, truth: dict, data: lcm.PatternData) -> tuple:
    """Data set of replicate r of label u: (PatternData, class counts, design statistics, oracle = Bayes
    posterior P(T | pattern) under the generating process, class-T share of the generating process)."""
    s = sim_simulator(u)
    seed = sim_seed(u, r)
    tw, tth = np.asarray(truth["w"], float), np.asarray(truth["theta"], float)
    if s == "fixed_masks":
        sim, cc = FS.simulate_fixed_masks(data, tw, tth, seed=seed)
        return sim, cc, pattern_design_stats(sim), true_credibility(sim, tw, tth), float(tw[0])
    des = cc_design(SIM_TRUTH[u][0])
    W, wT = strat_weights(u) if s == "faithful_cc_strat" else (None, float(tw[0] / tw.sum()))
    sim, cc, st = simulate_cc(des, tw, tth, seed=seed, W=W)
    return sim, cc, st, true_credibility(sim, tw, tth, des, W), wT


def sim_replicate(u: str, r: int, truth: dict, data: lcm.PatternData, alt: dict | None = None) -> dict:
    t0 = time.time()
    sim, cc, dstats, oracle, wT_true = simulate_for(u, r, truth, data)
    n_p, t_p = sim.n, cc[:, 0]
    pt = potential(sim)
    na, nb_ = N_RESTARTS_SIM[u]
    seed_a = derived_seed("recovery_fit", u, r, "family_hb")
    rs = multistart(pt, na, seed_a)
    for x in rs:
        x["set"] = "A"
    if nb_:
        rb = multistart(pt, nb_, derived_seed("modes_sim_fit", u, r))
        for x in rb:
            x["set"] = "B"
        rs += rb
    lp = np.array([x["log_post"] for x in rs])
    groups = group_optima(lp)
    optima = []
    for k, g in enumerate(groups):
        x = rs[g[0]]["x"]
        s = pt["post"](x)
        c = lcm.credibility_at(s, sim)
        o = dict(optimum=k, log_post=float(lp[g[0]]), n_restarts=len(g),
                 n_restarts_set_a=int(sum(rs[i]["set"] == "A" for i in g)), w=np.asarray(s["w"]).tolist(),
                 w_T=float(s["w"][0]), spearman_oracle=FS.weighted_spearman(c, oracle, n_p),
                 auroc=FS.weighted_auroc(c, t_p, n_p - t_p), mae_oracle=float(np.average(np.abs(c - oracle), weights=n_p)),
                 **set_stats(c, n_p, t_p))
        o["_cred"], o["_x"], o["_theta"] = c, x, np.asarray(s["theta"])
        optima.append(o)
    # Laplace masses of the top-2 optima
    for o in optima[:2]:
        la = laplace_at(pt, o["_x"])
        o.update(laplace_log_mass=la["laplace_log_mass"], log_post_polished=la["log_post"], hess_min_eig=la["min_eig"])
    best, sec = optima[0], (optima[1] if len(optima) > 1 else None)
    sp = np.array([o["spearman_oracle"] for o in optima])
    truth_k = int(np.argmax(sp))
    # truth-start
    ts = FS.map_from_start(sim, MODEL, truth["full"])
    ts_k = assign_optimum(ts["log_post"], [o["log_post"] for o in optima], tol=MODE_TOL * 10)
    ts_sp = FS.weighted_spearman(ts["cred"], oracle, n_p)
    # Laplace-weighted mixture of the top-2 optima
    if sec is not None and np.isfinite(best.get("laplace_log_mass", np.nan)) and np.isfinite(sec.get("laplace_log_mass", np.nan)):
        lm = np.array([best["laplace_log_mass"], sec["laplace_log_mass"]])
        p2l = float(np.exp(lm[1] - np.logaddexp.reduce(lm)))
        cmix = (1 - p2l) * best["_cred"] + p2l * sec["_cred"]
    else:
        p2l, cmix = 0.0, best["_cred"]
    # recovery-set (set A) best = the stored recovery fit
    ia = [i for i, x in enumerate(rs) if x["set"] == "A"]
    ib_a = ia[int(np.argmax(lp[ia]))]
    k_a = next(k for k, g in enumerate(groups) if ib_a in g)
    ka_cred = optima[k_a]["_cred"]
    # EM (recovery seed)
    em = lcm.fit_map(sim, "em", n_restarts=N_RESTARTS_EM_SIM, seed=derived_seed("recovery_fit", u, r, "em"))
    rec = dict(
        universe=u, replicate=r, simulator=sim_simulator(u), sim_seed=sim_seed(u, r), n_patterns=int(sim.P),
        n_sim_edges=float(n_p.sum()), n_true=float(t_p.sum()), w_T_true=float(wT_true),
        w_T_realised=float(t_p.sum() / n_p.sum()), design=dstats,
        n_restarts=len(rs), restart_log_post=lp.tolist(), restart_set=[x["set"] for x in rs],
        n_optima=len(optima), best_log_post=best["log_post"],
        second_log_post=sec["log_post"] if sec else np.nan, gap=(best["log_post"] - sec["log_post"]) if sec else np.nan,
        gap_polished=(best["log_post_polished"] - sec["log_post_polished"]) if sec else np.nan,
        laplace_p2=p2l, truth_mode_index=truth_k, dominant_is_truth=bool(truth_k == 0),
        second_is_truth=bool(truth_k == 1),
        truth_spearman_margin=float(sp[truth_k] - np.max(np.delete(sp, truth_k))) if len(sp) > 1 else np.nan,
        truthstart_log_post=ts["log_post"], truthstart_optimum=ts_k, truthstart_spearman_oracle=ts_sp,
        truthstart_minus_best=ts["log_post"] - best["log_post"],
        dominant_is_truthstart=bool(ts_k == 0),
        oracle=dict(auroc=FS.weighted_auroc(oracle, t_p, n_p - t_p), **set_stats(oracle, n_p, t_p)),
        dominant={k: v for k, v in best.items() if not k.startswith("_")},
        second={k: v for k, v in sec.items() if not k.startswith("_")} if sec else None,
        truthstart=dict(w_T=float(ts["w"][0]), **set_stats(ts["cred"], n_p, t_p), spearman_oracle=ts_sp,
                        auroc=FS.weighted_auroc(ts["cred"], t_p, n_p - t_p), log_post=ts["log_post"]),
        mixture_laplace=dict(p2=p2l, **set_stats(cmix, n_p, t_p),
                             spearman_oracle=FS.weighted_spearman(cmix, oracle, n_p)),
        recovery_fit=dict(optimum=k_a, log_post=float(lp[ib_a]), w_T=optima[k_a]["w_T"],
                          spearman_oracle=optima[k_a]["spearman_oracle"], auroc=optima[k_a]["auroc"],
                          is_truth_mode=bool(k_a == truth_k), **set_stats(ka_cred, n_p, t_p)),
        em=dict(log_post=float(em.log_post), w_T=float(em.best["w"][0]),
                spearman_oracle=FS.weighted_spearman(em.cred, oracle, n_p),
                auroc=FS.weighted_auroc(em.cred, t_p, n_p - t_p), **set_stats(em.cred, n_p, t_p)),
        optima=[{k: v for k, v in o.items() if not k.startswith("_")} for o in optima],
        calib={name: FS.calibration_bins(c, n_p, t_p).drop(columns=["bin_lo", "bin_hi"]).to_dict(orient="list")
               for name, c in (("dominant", best["_cred"]), ("oracle", oracle), ("recovery_fit", ka_cred),
                               ("em", em.cred), ("mixture_laplace", cmix), ("truthstart", ts["cred"]))},
        runtime_s=np.nan, analysis_status=STATUS)
    if alt is not None:      # the optimum resembling the OTHER real mode (``alt_oracle_stats``), computed in place
        c_alt = lcm.credibility_at(dict(w=alt["w"], theta=alt["theta"]), sim)
        sp_alt = [FS.weighted_spearman(o["_cred"], c_alt, n_p) for o in optima]
        k_alt = int(np.argmax(sp_alt))
        rec["alt"] = dict(alt_oracle_spearman=sp_alt, alt_mode_index=k_alt,
                          oracle_vs_alt_oracle_spearman=FS.weighted_spearman(oracle, c_alt, n_p),
                          truth_minus_alt_log_post=(float(lp[groups[truth_k][0]] - lp[groups[k_alt][0]])
                                                    if k_alt != truth_k else float("nan")),
                          alt_found_distinct=bool(k_alt != truth_k))
    rec["runtime_s"] = time.time() - t0
    # per-optimum parameters for the NUTS-chain classification of the recovery NUTS replicates
    extra = dict(w=np.stack([np.asarray(o["w"]) for o in optima]), theta=np.stack([o["_theta"] for o in optima]),
                 cred=np.stack([o["_cred"] for o in optima]), oracle=oracle, n=n_p, t=t_p,
                 em_w=np.asarray(em.best["w"]), em_theta=np.asarray(em.best["theta"]),
                 truthstart_w=np.asarray(ts["w"]), truthstart_theta=np.asarray(ts["theta"]))
    return rec, extra


def truth_params(u: str) -> dict:
    """Generating parameters of simulation ``u``: optimum 0 = the primary Family-HB MAP fit (snapshot
    ``params.npz``; the parameters the stored recovery study of ``fit_sensitivity`` simulated from);
    another optimum k = the polished optimum k of ``optima_params.npz`` (``SIM_TRUTH``)."""
    base, k = SIM_TRUTH.get(u, (u, 0))
    return real_mode_params(base, k)


def real_mode_params(base: str, k: int) -> dict:
    if k == 0:
        p = dict(np.load(map_dir(base) / "params.npz"))
    else:
        p = primary_optima(base)[2][k]["sample"]
    return dict(w=np.asarray(p["w"]), theta=np.asarray(p["theta"]), full=p)


FAITHFUL_LABELS = tuple(k for k, v in SIM_LABEL_INFO.items() if v[1] != "fixed_masks")


def run_simulation(universes=FAITHFUL_LABELS, reps=None, force: bool = False) -> None:
    for u in universes:
        od = C.ensure_dir(MODOUT / "sim" / u)
        base = SIM_TRUTH.get(u, (u, 0))[0]
        data = load_data(base)
        truth = truth_params(u)
        alt = real_mode_params(base, SIM_ALT[u]) if sim_simulator(u) != "fixed_masks" else None
        for r in (range(N_SIM[u]) if reps is None else [x for x in reps if x < N_SIM[u]]):
            f = od / f"rep{r:03d}.json"
            if f.exists() and not force:
                continue
            rec, extra = sim_replicate(u, r, truth, data, alt=alt)
            np.savez_compressed(od / f"rep{r:03d}_optima.npz", **extra)
            C.write_json(_jsonable(rec), f)
            d = rec["design"]
            log(f"[sim] {u} rep{r}: {rec['n_optima']} optima, gap {rec['gap']:.3f}, dominant=truth "
                f"{rec['dominant_is_truth']} (truth-start {rec['dominant_is_truthstart']}); Strong FDR "
                f"{rec['dominant']['strong_realised_fdr']:.3f} (oracle {rec['oracle']['strong_realised_fdr']:.3f}); "
                f"cc-observed {d.get('n_edges_cc_observed', np.nan):.0f} (with LIT report "
                f"{d.get('n_edges_cc_observed_with_lit_report', np.nan):.0f}) ({rec['runtime_s']:.0f}s)")


def alt_oracle_stats(u: str, r: int, data: lcm.PatternData, truth: dict, alt: dict) -> dict:
    """Which optimum found in replicate r resembles the OTHER real mode (alternative oracle =
    credibility of the simulated patterns at the parameters of the other real optimum, ``SIM_ALT``),
    and the signed log-posterior difference truth-mode optimum minus alternative-mode optimum.
    Replicates of the faithful simulators store it in the record (``alt``); for the fixed-mask replicates it
    re-simulates the replicate (same seed), checks it against the stored optima and caches the result."""
    rec = json.loads((MODOUT / "sim" / u / f"rep{r:03d}.json").read_text())
    if "alt" in rec:
        return rec["alt"]
    f = MODOUT / "sim" / u / f"rep{r:03d}_alt.json"
    if f.exists():
        return json.loads(f.read_text())
    so = np.load(MODOUT / "sim" / u / f"rep{r:03d}_optima.npz")
    sim = simulate_for(u, r, truth, data)[0]
    if sim.P != so["cred"].shape[1] or not np.allclose(sim.n, so["n"]):
        raise RuntimeError(f"{u} rep{r}: re-simulated data differ from the stored optima")
    c_alt = lcm.credibility_at(dict(w=alt["w"], theta=alt["theta"]), sim)
    sp = [FS.weighted_spearman(c, c_alt, sim.n) for c in so["cred"]]
    k_alt = int(np.argmax(sp))
    k_tr = int(rec["truth_mode_index"])
    lp = [o["log_post"] for o in rec["optima"]]
    out = dict(alt_oracle_spearman=sp, alt_mode_index=k_alt,
               oracle_vs_alt_oracle_spearman=FS.weighted_spearman(so["oracle"], c_alt, sim.n),
               truth_minus_alt_log_post=(float(lp[k_tr] - lp[k_alt]) if k_alt != k_tr else float("nan")),
               alt_found_distinct=bool(k_alt != k_tr))
    C.write_json(_jsonable(out), f)
    return out


def truth_vs_alt(rec: dict, a: dict, truth_mode_found: bool) -> dict:
    """Signed log-posterior difference truth-mode optimum minus the optimum resembling the other real mode.

    Normally the truth-mode optimum is the restart optimum closest to the oracle (``alt_oracle_stats``).
    If the restarts missed the truth basin (``truth_mode_found`` False), that optimum is not the truth
    mode, so the optimum reached from the generating parameters (truth-start) is used instead (fix of
    2026-10-14; before, immune replicate 62 compared the other mode with a degenerate optimum, 78 nats
    instead of 0.5)."""
    if truth_mode_found:
        return dict(alt_found_distinct=bool(a["alt_found_distinct"]),
                    dominant_is_alt=bool(a["alt_mode_index"] == 0 and a["alt_found_distinct"]),
                    truth_minus_alt_log_post=a["truth_minus_alt_log_post"], truth_log_post_source="restart_optimum")
    k_alt = int(a["alt_mode_index"])
    distinct = int(rec["truthstart_optimum"]) != k_alt
    return dict(alt_found_distinct=bool(distinct), dominant_is_alt=bool(k_alt == 0 and distinct),
                truth_minus_alt_log_post=(float(rec["truthstart_log_post"] - rec["optima"][k_alt]["log_post"])
                                          if distinct else float("nan")),
                truth_log_post_source="truthstart_optimum (truth mode not found by the restarts)")


def load_sim(u: str) -> list:
    od = MODOUT / "sim" / u
    recs = [json.loads(f.read_text()) for f in sorted(od.glob("rep[0-9][0-9][0-9].json"))]   # not rep###_alt.json
    return [r for r in recs if r["replicate"] < N_SIM[u]]


SIM_LABELS = tuple(SIM_LABEL_INFO)
DESIGN_COLS = ("n_edges_ge1_lit_report", "n_cond", "n_edges_cc_observed", "n_edges_cc_observed_with_lit_report",
               "n_cc_observations", "n_cc_observations_on_lit_reported")


def sim_replicate_frame(universes=SIM_LABELS) -> pd.DataFrame:
    rows = []
    for u in universes:
        recs = load_sim(u) if (MODOUT / "sim" / u).exists() else []
        if recs:
            base, k = SIM_TRUTH[u]
            sdata, truth, alt = load_data(base), truth_params(u), real_mode_params(base, SIM_ALT[u])
            fixed_stats = fixed_mask_design_stats(u) if sim_simulator(u) == "fixed_masks" else {}
        for r in recs:
            row = dict(universe=SIM_LABEL_INFO[u][0], simulator=sim_simulator(u), sim_label=u,
                       replicate=r["replicate"], n_patterns=r["n_patterns"], n_true=r["n_true"],
                       n_restarts=r["n_restarts"], n_optima=r["n_optima"], best_log_post=r["best_log_post"],
                       second_log_post=r["second_log_post"], gap=r["gap"], gap_polished=r["gap_polished"],
                       laplace_p2=r["laplace_p2"], dominant_is_truth=r["dominant_is_truth"],
                       second_is_truth=r["second_is_truth"], truth_mode_index=r["truth_mode_index"],
                       truth_spearman_margin=r["truth_spearman_margin"],
                       dominant_is_truthstart=r["dominant_is_truthstart"], truthstart_optimum=r["truthstart_optimum"],
                       truthstart_minus_best=r["truthstart_minus_best"],
                       n_restarts_at_best=r["dominant"]["n_restarts"],
                       n_restarts_at_second=r["second"]["n_restarts"] if r["second"] else 0)
            for grp in ("dominant", "second", "oracle", "truthstart", "mixture_laplace", "recovery_fit", "em"):
                g = r.get(grp) or {}
                for k in ("strong_n", "strong_claimed_bfdr", "strong_realised_fdr", "ge_moderate_n",
                          "ge_moderate_claimed_bfdr", "ge_moderate_realised_fdr", "spearman_oracle", "w_T"):
                    if k in g:
                        row[f"{grp}_{k}"] = g[k]
            # the truth mode is identified among the optima found only if one of them is as close to the
            # oracle as the optimum reached from the generating parameters (truth-start)
            sp_found = max(o["spearman_oracle"] for o in r["optima"])
            row["truth_mode_found"] = bool(sp_found >= r["truthstart_spearman_oracle"] - 0.01)
            row["dominant_correct_strict"] = bool(row["dominant_is_truth"] and row["truth_mode_found"])
            row["recovery_fit_log_post"] = r["recovery_fit"]["log_post"]
            row["recovery_fit_is_truth_mode"] = r["recovery_fit"]["is_truth_mode"]
            row["em_log_post"] = r["em"]["log_post"]
            row["w_T_true"] = r["w_T_true"]
            row["data_universe"], row["truth_real_optimum"] = SIM_TRUTH[u]
            a = alt_oracle_stats(u, int(r["replicate"]), sdata, truth, alt)
            row["alt_mode_index"] = a["alt_mode_index"]
            row.update(truth_vs_alt(r, a, row["truth_mode_found"]))
            row["dominant_alt_oracle_spearman"] = a["alt_oracle_spearman"][0]
            row["oracle_vs_alt_oracle_spearman"] = a["oracle_vs_alt_oracle_spearman"]
            d = r.get("design") or fixed_stats.get(int(r["replicate"]), {})
            for c in DESIGN_COLS:
                row[f"design_{c}"] = d.get(c, np.nan)
            row["analysis_status"] = STATUS
            rows.append(row)
    df = pd.DataFrame(rows)
    if len(df):
        df["gap_bin"] = gap_bin_labels(df["gap"].to_numpy(float))
    return df


def gap_bin_labels(gap: np.ndarray) -> np.ndarray:
    """'[lo, hi)' label of every gap (``GAP_BINS``); 'no_second_optimum' when only one optimum was found."""
    out = np.empty(len(gap), dtype=object)
    for i, g in enumerate(gap):
        if not np.isfinite(g):
            out[i] = "no_second_optimum"
            continue
        k = int(np.searchsorted(GAP_BINS, g, side="right")) - 1
        out[i] = f"[{GAP_BINS[k]:g}, {GAP_BINS[k + 1]:g})"
    return out


def _wilson(k: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return c - h, c + h


def logit_fit(x: np.ndarray, y: np.ndarray, max_iter: int = 100, tol: float = 1e-10) -> dict:
    """Maximum-likelihood logistic regression of y (0/1) on [1, x] by Newton-Raphson; covariance =
    inverse Fisher information at the optimum.  Self-contained (no statsmodels dependency, whose
    import breaks with the installed scipy).  Raises RuntimeError on (quasi-)separation, detected as
    a diverging slope or a non-converging iteration.  ``lr_stat`` = likelihood-ratio statistic of the
    slope (2 x log-likelihood gain over the intercept-only model, 1 df)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    X = np.column_stack([np.ones_like(x), x])
    beta = np.zeros(2)
    for it in range(max_iter):
        eta = X @ beta
        p = 1.0 / (1.0 + np.exp(-eta))
        W = p * (1 - p)
        H = X.T @ (X * W[:, None])
        g = X.T @ (y - p)
        step = np.linalg.solve(H, g)
        beta = beta + step
        if np.max(np.abs(beta)) > 50:
            raise RuntimeError("logistic fit diverges (separation)")
        if np.max(np.abs(step)) < tol:
            break
    else:
        raise RuntimeError("logistic fit did not converge")
    p = 1.0 / (1.0 + np.exp(-(X @ beta)))
    H = X.T @ (X * (p * (1 - p))[:, None])
    pc = np.clip(p, 1e-300, 1 - 1e-16)
    ll = float(np.sum(y * np.log(pc) + (1 - y) * np.log1p(-pc)))
    p0 = float(np.clip(y.mean(), 1e-300, 1 - 1e-16))
    ll0 = float(np.sum(y * np.log(p0) + (1 - y) * np.log1p(-p0)))
    return dict(params=beta, cov=np.linalg.inv(H), n_iter=it + 1, loglik=ll, loglik_null=ll0,
                lr_stat=2.0 * (ll - ll0))


def sim_summary(df: pd.DataFrame) -> pd.DataFrame:
    from scipy.stats import chi2
    rows = []
    if "simulator" not in df:
        df = df.assign(simulator="fixed_masks")
    for (u, simr), g in df.groupby(["universe", "simulator"], sort=True):
        start = len(rows)

        def add(stratum, sub, quantity, value, lo=np.nan, hi=np.nan, n=None, **kw):
            rows.append(dict(universe=u, stratum=stratum, quantity=quantity, n_replicates=len(sub) if n is None else n,
                             value=value, ci_lo=lo, ci_hi=hi, **kw, analysis_status=STATUS))
        bins = sorted(g.gap_bin.unique(), key=lambda s: np.inf if s.startswith("no_") else float(s.split(",")[0][1:]))
        strata = [("all", g)] + [(f"gap_bin {b}", g[g.gap_bin == b]) for b in bins]
        strata += [("dominant_is_truth", g[g.dominant_is_truth]), ("dominant_not_truth", g[~g.dominant_is_truth])]
        for name, sub in strata:
            if not len(sub):
                continue
            k = int(sub.dominant_is_truth.sum())
            lo, hi = _wilson(k, len(sub))
            add(name, sub, "p_dominant_is_truth", k / len(sub), lo, hi)
            k2 = int(sub.dominant_is_truthstart.sum())
            lo, hi = _wilson(k2, len(sub))
            add(name, sub, "p_dominant_is_truthstart_optimum", k2 / len(sub), lo, hi)
            k3 = int(sub.dominant_correct_strict.sum())
            lo, hi = _wilson(k3, len(sub))
            add(name, sub, "p_dominant_correct_strict", k3 / len(sub), lo, hi)
            k5 = int(sub.truth_mode_found.sum())
            lo, hi = _wilson(k5, len(sub))
            add(name, sub, "p_truth_mode_found", k5 / len(sub), lo, hi)
            if "dominant_is_alt" in sub:
                k4 = int(sub.dominant_is_alt.sum())
                lo, hi = _wilson(k4, len(sub))
                add(name, sub, "p_dominant_resembles_other_real_mode", k4 / len(sub), lo, hi)
            add(name, sub, "gap_median", float(sub.gap.median()), float(sub.gap.quantile(0.1)), float(sub.gap.quantile(0.9)))
            for grp in ("dominant", "oracle", "second", "truthstart", "mixture_laplace", "em"):
                col = f"{grp}_strong_realised_fdr"
                if col in sub and sub[col].notna().any():
                    v = sub[col].dropna()
                    add(name, sub, f"{grp}_strong_realised_fdr_mean", float(v.mean()), float(v.quantile(0.1)),
                        float(v.quantile(0.9)), n=len(v))
                    add(name, sub, f"{grp}_strong_realised_fdr_gt_0.10", float((v > 0.10).mean()), n=len(v))
                    add(name, sub, f"{grp}_strong_n_mean", float(sub[f"{grp}_strong_n"].mean()), n=len(v))
                    add(name, sub, f"{grp}_strong_claimed_bfdr_mean", float(sub[f"{grp}_strong_claimed_bfdr"].mean()), n=len(v))
                col = f"{grp}_ge_moderate_realised_fdr"
                if col in sub and sub[col].notna().any():
                    v = sub[col].dropna()
                    add(name, sub, f"{grp}_ge_moderate_realised_fdr_mean", float(v.mean()), float(v.quantile(0.1)),
                        float(v.quantile(0.9)), n=len(v))
                    # twice the Moderate claim (0.20), as 0.10 is twice the Strong claim
                    add(name, sub, f"{grp}_ge_moderate_realised_fdr_gt_0.40", float((v > 0.40).mean()), n=len(v))
                    if f"{grp}_ge_moderate_claimed_bfdr" in sub:
                        add(name, sub, f"{grp}_ge_moderate_claimed_bfdr_mean",
                            float(sub[f"{grp}_ge_moderate_claimed_bfdr"].mean()), n=len(v))
            add(name, sub, "laplace_p2_mean", float(sub.laplace_p2.mean()))
            for c in DESIGN_COLS:
                if f"design_{c}" in sub and sub[f"design_{c}"].notna().any():
                    v = sub[f"design_{c}"].dropna()
                    add(name, sub, f"design_{c}_mean", float(v.mean()), float(v.min()), float(v.max()), n=len(v),
                        note="ci_lo / ci_hi = min / max over replicates")
        # gap distribution: quantiles and the position of the real gaps (percentile = fraction of simulated
        # gaps below the real gap; replicates with a second optimum, and those that found the truth mode)
        gf = g.dropna(subset=["gap"])
        if len(gf):
            for q in (0.5, 0.75, 0.9, 0.95):
                add("all", gf, f"gap_q{int(round(q * 100))}", float(gf.gap.quantile(q)))
            add("all", gf, "gap_max", float(gf.gap.max()))
            for tt, gap in REAL_GAPS.items():
                for sname, sub in (("all", gf), ("truth_mode_found", gf[gf.truth_mode_found.astype(bool)])):
                    if len(sub):
                        add(f"real_gap_{tt}", sub, f"frac_sim_gap_below_real_gap_{sname}",
                            float((sub.gap < gap).mean()), real_gap=gap)
        # logistic regression of correctness on the gap, and P(correct | gap >= real gap)
        # fitted on the replicates whose restarts found the truth mode: when the truth basin is missed the best-vs-
        # second gap does not compare the two modes (immune replicate 62: 76 nats between two non-truth optima)
        gg = g.dropna(subset=["gap"])
        gg = gg[gg.truth_mode_found.astype(bool)] if "truth_mode_found" in gg else gg
        if len(gg) >= 10 and gg.dominant_is_truth.nunique() == 2:
            try:
                xg = gg["gap"].to_numpy(float)
                fit = logit_fit(xg, gg["dominant_is_truth"].astype(float).to_numpy())
                for tt, gap in REAL_GAPS.items():
                    xp = np.array([1.0, gap])
                    eta = float(xp @ fit["params"])
                    se = float(np.sqrt(xp @ fit["cov"] @ xp))
                    n_ge = int(np.sum(xg >= gap))
                    beyond = bool(gap > xg.max())
                    note = (f"ML logistic fit of dominant_is_truth on the gap (Newton, Wald interval) over the replicates "
                            f"of the {u} simulation ({simr}) with truth_mode_found")
                    if tt != u.split("_")[0]:
                        note += f"; the {tt} real gap is evaluated with the {u} calibration (no {tt} simulation)"
                    if beyond:
                        note += "; real gap beyond the largest simulated gap: no interval (extrapolation)"
                    elif n_ge < 5:
                        note += "; EXTRAPOLATED: < 5 replicates with gap >= real gap"
                    rows.append(dict(universe=u, stratum=f"logistic_at_real_gap_{tt}", quantity="p_dominant_is_truth",
                                     n_replicates=len(gg), value=1 / (1 + np.exp(-eta)),
                                     ci_lo=np.nan if beyond else 1 / (1 + np.exp(-(eta - 1.96 * se))),
                                     ci_hi=np.nan if beyond else 1 / (1 + np.exp(-(eta + 1.96 * se))),
                                     real_gap=gap, logit_intercept=float(fit["params"][0]),
                                     logit_slope=float(fit["params"][1]), logit_slope_se=float(np.sqrt(fit["cov"][1, 1])),
                                     logit_lr_stat=fit["lr_stat"], logit_lr_p=float(chi2.sf(fit["lr_stat"], 1)),
                                     n_replicates_gap_ge_real=n_ge, max_simulated_gap=float(xg.max()),
                                     # extrapolated: fewer than 5 simulated replicates at or beyond the real gap
                                     extrapolated=bool(n_ge < 5), note=note, analysis_status=STATUS))
            except (np.linalg.LinAlgError, RuntimeError) as exc:  # perfect separation etc.
                rows.append(dict(universe=u, stratum="logistic", quantity="fit_failed", n_replicates=len(gg),
                                 value=np.nan, note=str(exc)[:200], analysis_status=STATUS))
        # signed comparison of the truth-mode optimum with the optimum resembling the other real mode
        if "truth_minus_alt_log_post" in g:
            d = g["truth_minus_alt_log_post"].to_numpy(float)
            fin = np.isfinite(d)
            add("all", g, "p_other_real_mode_found_distinct", float(fin.mean()))
            if fin.any():
                add("all", g, "truth_minus_other_mode_log_post_median", float(np.median(d[fin])),
                    float(np.quantile(d[fin], 0.1)), float(np.quantile(d[fin], 0.9)), n=int(fin.sum()))
                for tt, gap in REAL_GAPS.items():
                    for qn, ev in (("p_other_mode_beats_truth_by_ge_real_gap", d[fin] <= -gap),
                                   ("p_truth_beats_other_mode_by_ge_real_gap", d[fin] >= gap)):
                        lo, hi = _wilson(int(ev.sum()), int(fin.sum()))
                        rows.append(dict(universe=u, stratum=f"real_gap_{tt}", quantity=qn, n_replicates=int(fin.sum()),
                                         value=float(ev.mean()), ci_lo=lo, ci_hi=hi, real_gap=gap,
                                         note="replicates in which both optima were found (distinct)",
                                         analysis_status=STATUS))
        for tt, gap in REAL_GAPS.items():
            sub = g[g.gap >= gap]
            k = int(sub.dominant_is_truth.sum())
            lo, hi = _wilson(k, len(sub))
            rows.append(dict(universe=u, stratum=f"gap_ge_real_{tt}", quantity="p_dominant_is_truth", n_replicates=len(sub),
                             value=k / len(sub) if len(sub) else np.nan, ci_lo=lo, ci_hi=hi, real_gap=gap,
                             frac_replicates_with_gap_ge=float((g.gap >= gap).mean()), analysis_status=STATUS))
        for x in rows[start:]:
            x["simulator"] = simr
    out = pd.DataFrame(rows)
    if len(out):
        out.insert(1, "simulator", out.pop("simulator"))
    return out


def sim_calibration(universes=SIM_LABELS) -> pd.DataFrame:
    rows = []
    for u in universes:
        recs = load_sim(u) if (MODOUT / "sim" / u).exists() else []
        base = dict(universe=SIM_LABEL_INFO[u][0], simulator=sim_simulator(u))
        for name in ("dominant", "oracle", "recovery_fit", "em", "mixture_laplace"):
            for stratum, sel in (("all", lambda r: True), ("dominant_is_truth", lambda r: r["dominant_is_truth"]),
                                 ("dominant_not_truth", lambda r: not r["dominant_is_truth"])):
                rr = [r for r in recs if sel(r)]
                if not rr:
                    continue
                nb = np.sum([r["calib"][name]["n_edges"] for r in rr], 0)
                sp = np.sum([r["calib"][name]["sum_pred"] for r in rr], 0)
                st = np.sum([r["calib"][name]["n_true"] for r in rr], 0)
                for b in range(N_CAL_BINS):
                    rows.append(dict(base, estimator=name, stratum=stratum, bin=b, bin_lo=b / N_CAL_BINS,
                                     bin_hi=(b + 1) / N_CAL_BINS, n_replicates=len(rr), n_edges_total=nb[b],
                                     mean_predicted=sp[b] / nb[b] if nb[b] else np.nan,
                                     empirical_true_rate=st[b] / nb[b] if nb[b] else np.nan, analysis_status=STATUS))
                m = np.arange(N_CAL_BINS) >= 1
                with np.errstate(invalid="ignore", divide="ignore"):
                    mp, tr = sp / nb, st / nb
                ece19 = float(np.nansum(nb[m] * np.abs(mp[m] - tr[m])) / nb[m].sum()) if nb[m].sum() else np.nan
                rows.append(dict(base, estimator=name, stratum=stratum, bin=-1, bin_lo=0.1, bin_hi=1.0,
                                 n_replicates=len(rr), n_edges_total=float(nb[m].sum()), mean_predicted=np.nan,
                                 empirical_true_rate=np.nan, ece_bins1_9_pooled=ece19, analysis_status=STATUS))
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------------------
# (3b) design-faithful simulators: curated-conditional MOTIF/COEXP masks regenerated from the simulated
#      LIT reports (fix of the modes verifier's B1, 2026-10-14; post hoc, label-free)
# ----------------------------------------------------------------------------------------
# Under the primary mask ``prespecified_cc`` a MOTIF/COEXP annotator m observes edge e = (tf, g) iff tf is
# covered by m (whole resource) AND e is in the conditioning set Cond = edges with a held-in LIT record that
# arrived through the DoRothEA channel (``evidence.conditioning_set``; LIT rows with ``via_dorothea``).
# ``fit_sensitivity.simulate_fixed_masks`` keeps the real MOTIF/COEXP masks while re-drawing the LIT reports,
# which decouples them (real data: every MOTIF/COEXP-observed edge carries >= 1 LIT report; a fixed-mask
# replicate: ~5%).  ``simulate_cc`` keeps the real LIT masks (the LIT rule depends on the annotator's whole
# resource only) and rebuilds Cond, and with it every MOTIF/COEXP mask, from the SIMULATED LIT reports:
#   * a simulated report of LIT annotator j on edge e is a DoRothEA-channel report iff chan[e, j]; chan is
#     the REAL per-edge channel membership where j has a real record of e (``via_dorothea``), otherwise the
#     channel rule of j's real records of the same TF: 14 of the 18 held-in LIT sources are pure (all or
#     none of their records via DoRothEA); for TRRUST / HTRIdb / IntAct / TFactS the per-(source, TF) real
#     fraction f is used, constant where f is 0 or 1 and one fixed Bernoulli(f) draw per cell otherwise
#     (seed ``derived_seed('cc_channel_map', u, source)``; the same map in every replicate);
#   * MOTIF/COEXP TF coverage is the real whole-resource coverage (``AnnotatorBank.cov``).
# ``cc_design`` checks that this rule, applied to the REAL LIT reports and channel flags, reproduces every
# real MOTIF/COEXP mask exactly.  Only edges observed by >= 1 LIT annotator (the domain) can be observed at
# all (Cond requires a LIT report), so only they are simulated (as in ``simulate_fixed_masks``).
# LIT node-set masks (f2 minor 7) are NOT regenerated: nodes_j (genes with any record of j in its whole
# resource) is report-dependent, but the MAR parameters are report rates within the selected (observed)
# cells, so the model defines no rate for the cells outside the real node sets; ``lit_node_check`` reports
# the size of the problem (cells and expected reports outside the node sets, node genes supported only by
# in-universe records).  'faithful_cc_strat' instead reproduces the footprint of the report-dependent LIT
# masks on the class composition: the class weights of an edge depend on its LIT observation mask (stratum
# k), w_k = edge-weighted mean posterior class probability of the real edges of stratum k at the generating
# parameters.  This reproduces each LIT annotator's expected positive count exactly (the M-step identity)
# and the real number of informative MOTIF/COEXP observations to within ~10%; its truth (oracle) is the
# Bayes posterior under the stratified prior.
CC_DIR = MODOUT / "cc_design"
MIXED_CHANNEL_SEED = "cc_channel_map"
CHAN_RULES = ("real_record", "source_constant", "tf_constant", "tf_bernoulli")


def _universe_axes(u: str) -> tuple[list, list]:
    from r2 import evidence as E
    if u == "global":
        d = E.EV_DIR / "global"
        tfs = [x.strip() for x in (d / "tfs.txt").read_text().splitlines() if x.strip()]
        genes = [x.strip() for x in (d / "genes.txt").read_text().splitlines() if x.strip()]
        return tfs, genes
    return E.tissue_universe(u)


def _edge_tf_gene(u: str, edges: np.ndarray, tfs: list, genes: list) -> tuple[np.ndarray, np.ndarray]:
    """TF and gene index of stored edges (``edge_pattern`` positions) of universe u."""
    from r2 import evidence as E
    if u == "global":                     # TF-major full grid, self-pairs included (pattern -1)
        G = len(genes)
        return (edges // G).astype(np.int64), (edges % G).astype(np.int64)
    ev = pd.read_parquet(E.EV_DIR / u / "evidence.parquet", columns=["tf", "target"])
    ti = pd.Index(tfs).get_indexer(ev["tf"].to_numpy()[edges])
    gi = pd.Index(genes).get_indexer(ev["target"].to_numpy()[edges])
    if (ti < 0).any() or (gi < 0).any():
        raise RuntimeError(f"{u}: evidence.parquet edges outside the universe axes")
    return ti.astype(np.int64), gi.astype(np.int64)


def lit_mask_codes(R: np.ndarray, lit: np.ndarray) -> np.ndarray:
    """Integer code of the LIT observation mask of every pattern (bit a = LIT annotator lit[a] observes)."""
    return ((R[:, lit] >= 0).astype(np.int64) * (np.int64(2) ** np.arange(len(lit), dtype=np.int64))).sum(1)


_CC_DESIGN_CACHE: dict = {}


def cc_design(u: str) -> dict:
    """Fixed design of the faithful simulators for universe u (primary evidence variant); cached per process.

    Keys: R, n, family, annotators (real patterns), lit / cc (column indices), dom_pattern (real pattern of
    every domain edge = edge observed by >= 1 LIT annotator, in stored edge order), tf_idx, chan (n_dom,
    n_lit) bool channel map, cov (n_dom, n_cc) bool MOTIF/COEXP TF coverage, frac_tf (n_lit, n_tfs) real
    per-(source, TF) channel fraction (NaN: TF not covered by the source), chan_rule_counts, real counts
    and the stratum (LIT observation mask) of every pattern."""
    if u in _CC_DESIGN_CACHE:
        return _CC_DESIGN_CACHE[u]
    from r2 import evidence as E
    if (E.VARIANTS[VARIANT]["mask"] != "prespecified_cc" or E.VARIANTS[VARIANT]["transform"] is not None
            or E.CC_TF_COVERAGE != "whole_resource"):
        raise RuntimeError("cc_design assumes the primary prespecified_cc variant with whole-resource coverage")
    R, n, fam, ann = E.load_patterns(u, VARIANT)
    fam = np.asarray(fam)
    lit = np.flatnonzero(fam == "LIT")
    ccj = np.flatnonzero(np.isin(fam, E.CC_FAMILIES))
    if len(lit) + len(ccj) != len(fam):
        raise RuntimeError(f"{u}: annotators outside LIT/MOTIF/COEXP in the primary variant")
    ep = np.asarray(E.load_edge_pattern(u, VARIANT, mmap=False)).astype(np.int64)
    tfs, genes = _universe_axes(u)
    G = len(genes)
    valid = ep >= 0
    lit_obs_p = (R[:, lit] >= 0).any(1)
    if ((R[:, ccj] >= 0).any(1) & ~lit_obs_p & (n > 0)).any():
        raise RuntimeError(f"{u}: MOTIF/COEXP observation outside the LIT-observed domain")
    dom = np.flatnonzero(valid & lit_obs_p[np.where(valid, ep, 0)])
    pat = ep[dom]
    ti, gi = _edge_tf_gene(u, dom, tfs, genes)
    key = ti * G + gi
    order = np.argsort(key, kind="mergesort")
    skey = key[order]
    ed = E.variant_edges(VARIANT)
    tf_ix, g_ix = pd.Index(tfs), pd.Index(genes)
    n_dom, n_lit = len(dom), len(lit)
    rep_real = np.zeros((n_dom, n_lit), bool)
    via_real = np.zeros((n_dom, n_lit), bool)
    frac_tf = np.full((n_lit, len(tfs)), np.nan)
    for a, j in enumerate(lit):
        e = ed[ed["source_id"] == ann[j]]
        fr = e.groupby("tf")["via_dorothea"].mean()
        ii = tf_ix.get_indexer(fr.index)
        frac_tf[a, ii[ii >= 0]] = fr.to_numpy(float)[ii >= 0]
        t_, g_ = tf_ix.get_indexer(e["tf"]), g_ix.get_indexer(e["target"])
        ok = (t_ >= 0) & (g_ >= 0)
        k = t_[ok].astype(np.int64) * G + g_[ok]
        pos = np.minimum(np.searchsorted(skey, k), len(skey) - 1)
        hit = skey[pos] == k
        idx = order[pos[hit]]
        rep_real[idx, a] = True
        via_real[idx, a] = e["via_dorothea"].to_numpy(bool)[ok][hit]
    Rd_lit = R[pat][:, lit]
    if not np.array_equal(rep_real, Rd_lit == 1):
        raise RuntimeError(f"{u}: real LIT records do not match the stored reports")
    obs_lit = Rd_lit >= 0
    chan = via_real.copy()
    rule_counts = {r_: 0 for r_ in CHAN_RULES}
    rule_counts["real_record"] = int(rep_real.sum())
    for a, j in enumerate(lit):
        f = frac_tf[a][ti]
        cells = obs_lit[:, a] & ~rep_real[:, a]
        if np.isnan(f[cells]).any():
            raise RuntimeError(f"{u}: {ann[j]} observes a TF it has no record of")
        const = cells & ((f == 0) | (f == 1))
        chan[const, a] = f[const] == 1
        pure_src = bool(np.nanmin(frac_tf[a]) == np.nanmax(frac_tf[a]) and np.nanmin(frac_tf[a]) in (0.0, 1.0))
        rule_counts["source_constant" if pure_src else "tf_constant"] += int(const.sum())
        mixed = cells & (f > 0) & (f < 1)
        if mixed.any():
            rng = np.random.default_rng(derived_seed(MIXED_CHANNEL_SEED, u, ann[j]))
            chan[mixed, a] = rng.random(int(mixed.sum())) < f[mixed]
            rule_counts["tf_bernoulli"] += int(mixed.sum())
    chan &= obs_lit
    bank = E.bank_for_variant(VARIANT)
    if list(bank.sources) != list(ann):
        raise RuntimeError(f"{u}: annotator order of the evidence bank differs from the stored patterns")
    cov = np.zeros((n_dom, len(ccj)), bool)
    for b, j in enumerate(ccj):
        cs = bank.cov[bank.sources.index(ann[j])]
        cov[:, b] = np.array([t in cs for t in tfs])[ti]
    cond_real = (rep_real & via_real).any(1)
    if not np.array_equal(cov & cond_real[:, None], R[pat][:, ccj] >= 0):
        raise RuntimeError(f"{u}: the cc rule does not reproduce the real MOTIF/COEXP masks")
    codes = lit_mask_codes(R, lit)
    sc, s_of_p = np.unique(np.where(lit_obs_p, codes, -1), return_inverse=True)
    s_of_p = s_of_p.reshape(-1) - (1 if sc[0] == -1 else 0)      # -1 = pattern without LIT observation
    des = dict(universe=u, R=R, n=n, family=list(fam), annotators=list(ann), lit=lit, cc=ccj, dom_pattern=pat,
               tf_idx=ti, chan=chan, cov=cov, frac_tf=frac_tf, rep_real=rep_real, chan_rule_counts=rule_counts,
               n_dom=int(n_dom), n_cond_real=int(cond_real.sum()),
               n_cc_observed_real=int((cov & cond_real[:, None]).any(1).sum()),
               stratum_codes=sc[sc >= 0], stratum_of_pattern=s_of_p, mask_reconstruction_exact=True)
    _CC_DESIGN_CACHE[u] = des
    return des


def class_log_joint(R: np.ndarray, w, theta, log_prior: np.ndarray | None = None) -> np.ndarray:
    """(P, S) log p(z = s, pattern) under MAR emissions; ``log_prior`` (P, S) replaces log w per pattern."""
    th = np.clip(np.asarray(theta, float), 1e-12, 1 - 1e-12)
    obs = (R >= 0).astype(float)
    pos = (R == 1).astype(float)
    lp = np.log(np.asarray(w, float))[None, :] if log_prior is None else log_prior
    return lp + (obs * pos) @ np.log(th) + (obs * (1 - pos)) @ np.log1p(-th)


def class_posteriors(R: np.ndarray, w, theta, log_prior: np.ndarray | None = None) -> np.ndarray:
    lj = class_log_joint(R, w, theta, log_prior)
    return np.exp(lj - np.logaddexp.reduce(lj, axis=1)[:, None])


def stratum_class_weights(des: dict, w, theta) -> np.ndarray:
    """(n_strata, S) class weights of 'faithful_cc_strat': edge-weighted mean posterior class probabilities
    of the real edges of every LIT-observation-mask stratum at the generating parameters (w, theta)."""
    R, n, sp = des["R"], des["n"], des["stratum_of_pattern"]
    post = class_posteriors(R, w, theta)
    m = (sp >= 0) & (n > 0)
    ns = len(des["stratum_codes"])
    S = post.shape[1]
    num = np.stack([np.bincount(sp[m], weights=n[m] * post[m, s], minlength=ns) for s in range(S)], 1)
    den = np.bincount(sp[m], weights=n[m], minlength=ns)
    if (den <= 0).any():
        raise RuntimeError("empty LIT-mask stratum")
    W = num / den[:, None]
    return W / W.sum(1, keepdims=True)


def pattern_strata(des: dict, R_sim: np.ndarray) -> np.ndarray:
    """Stratum index (row of ``stratum_class_weights``) of simulated patterns (their LIT masks are real)."""
    codes = lit_mask_codes(R_sim, des["lit"])
    sc = des["stratum_codes"]
    k = np.searchsorted(sc, codes)
    if (k >= len(sc)).any() or not np.array_equal(sc[np.minimum(k, len(sc) - 1)], codes):
        raise RuntimeError("simulated LIT mask outside the real strata")
    return k


def true_credibility(sim: lcm.PatternData, w, theta, des: dict | None = None, W: np.ndarray | None = None) -> np.ndarray:
    """Bayes posterior P(T | pattern) under the generating process (stratified prior when W is given)."""
    if W is None:
        return lcm.credibility_at(dict(w=np.asarray(w), theta=np.asarray(theta)), sim)
    lp = np.log(np.clip(W[pattern_strata(des, sim.R)], 1e-300, None))
    return class_posteriors(sim.R, w, theta, log_prior=lp)[:, 0]


def pattern_design_stats(data: lcm.PatternData) -> dict:
    """Design statistics of a (real or simulated) pattern set of the primary variant: edges with >= 1 LIT
    report, edges observed by >= 1 MOTIF/COEXP annotator and how many of them carry >= 1 LIT report, and
    per-annotator observed / reported edge counts."""
    from r2 import evidence as E
    fam = np.asarray(data.family)
    lit, ccj = np.flatnonzero(fam == "LIT"), np.flatnonzero(np.isin(fam, E.CC_FAMILIES))
    R, n = data.R, np.asarray(data.n, float)
    lit_rep = (R[:, lit] == 1).any(1)
    cc_obs = (R[:, ccj] >= 0).any(1)
    return dict(n_edges_observed=float(n[(R >= 0).any(1)].sum()), n_edges_ge1_lit_report=float(n[lit_rep].sum()),
                n_edges_cc_observed=float(n[cc_obs].sum()),
                n_edges_cc_observed_with_lit_report=float(n[cc_obs & lit_rep].sum()),
                n_cc_observations=float(((R[:, ccj] >= 0) * n[:, None]).sum()),
                n_cc_observations_on_lit_reported=float(((R[:, ccj] >= 0) * (n * lit_rep)[:, None]).sum()),
                n_observed=((R >= 0) * n[:, None]).sum(0).tolist(), n_reported=((R == 1) * n[:, None]).sum(0).tolist())


def _accumulate(us: list, cs: list, K: int, S: int, family, annotators) -> tuple:
    u_all, inv2 = np.unique(np.concatenate(us), return_inverse=True)
    cnt = np.bincount(inv2.reshape(-1), weights=np.concatenate(cs))
    code, z = u_all // S, u_all % S
    cu, pinv = np.unique(code, return_inverse=True)
    pinv = pinv.reshape(-1)
    cc = np.zeros((len(cu), S))
    np.add.at(cc, (pinv, z), cnt)
    sim = lcm.PatternData(FS._decode(cu, K), cc.sum(1), list(family), list(annotators))
    return sim, cc


def simulate_cc(des: dict, w, theta, seed: int, W: np.ndarray | None = None, chunk: int = 400_000) -> tuple:
    """Design-faithful simulation (see the section comment): classes and reports for every domain edge with
    the real LIT masks; the MOTIF/COEXP masks rebuilt from the simulated DoRothEA-channel LIT reports.
    ``W`` (n_strata, S): stratified class weights ('faithful_cc_strat'); None = global w ('faithful_cc').
    Returns (PatternData, class counts (P_sim, S), statistics)."""
    w = np.asarray(w, float)
    w = w / w.sum()
    theta = np.asarray(theta, float)
    R, lit, ccj = des["R"], des["lit"], des["cc"]
    K, S = R.shape[1], len(w)
    assert theta.shape == (K, S)
    pat, chan, cov = des["dom_pattern"], des["chan"], des["cov"]
    thT = theta.T
    rng = np.random.default_rng(seed)
    pow3 = 3 ** np.arange(K, dtype=np.int64)
    if W is not None:
        cumW = np.cumsum(W, 1)
        cumW[:, -1] = 1.0
        s_of_e = des["stratum_of_pattern"]
    us, cs = [], []
    st = dict(n_dom=0, n_ge1_lit_report=0, n_cond=0, n_cc_observed=0, n_cc_observed_with_lit_report=0, n_true=0)
    lit_pos_chan = np.zeros(len(lit))
    for s0 in range(0, len(pat), chunk):
        p = pat[s0:s0 + chunk]
        m = len(p)
        if W is None:
            z = rng.choice(S, size=m, p=w)
        else:
            uu = rng.random(m)
            z = np.minimum((uu[:, None] > cumW[s_of_e[p]]).sum(1), S - 1)
        rep = rng.random((m, K)) < thT[z]
        Rl = R[p][:, lit]
        obs_lit = Rl >= 0
        rep_lit = obs_lit & rep[:, lit]
        ch = rep_lit & chan[s0:s0 + chunk]
        cond = ch.any(1)
        obs_cc = cov[s0:s0 + chunk] & cond[:, None]
        M = np.full((m, K), -1, np.int8)
        M[:, lit] = np.where(obs_lit, rep_lit.astype(np.int8), np.int8(-1))
        M[:, ccj] = np.where(obs_cc, rep[:, ccj].astype(np.int8), np.int8(-1))
        code = (M.astype(np.int64) + 1) @ pow3
        u_, c_ = np.unique(code * S + z, return_counts=True)
        us.append(u_)
        cs.append(c_)
        anyrep = rep_lit.any(1)
        cco = obs_cc.any(1)
        st["n_dom"] += m
        st["n_ge1_lit_report"] += int(anyrep.sum())
        st["n_cond"] += int(cond.sum())
        st["n_cc_observed"] += int(cco.sum())
        st["n_cc_observed_with_lit_report"] += int((cco & anyrep).sum())
        st["n_true"] += int((z == 0).sum())
        lit_pos_chan += ch.sum(0)
    sim, cc = _accumulate(us, cs, K, S, des["family"], des["annotators"])
    st["n_lit_reported_dorothea_channel"] = lit_pos_chan.tolist()
    st.update(pattern_design_stats(sim))
    return sim, cc, st


def expected_cc_counts(des: dict, w, theta, W: np.ndarray | None = None, mixed: str = "map",
                       chunk: int = 500_000) -> dict:
    """Analytic expectation of the design statistics of ``simulate_cc``: E[#edges in the simulated Cond],
    E[#MOTIF/COEXP-observed edges], E[#edges with >= 1 LIT report], E[LIT positives per annotator].
    ``mixed``: channel of the mixed-TF cells = the fixed map ('map'), all off ('lo') or all on ('hi')."""
    w = np.asarray(w, float) / np.sum(w)
    theta = np.asarray(theta, float)
    R, lit, pat = des["R"], des["lit"], des["dom_pattern"]
    S = len(w)
    th = theta[lit]                     # (n_lit, S)
    out = dict(n_cond=0.0, n_cc_observed=0.0, n_ge1_lit_report=0.0)
    pos = np.zeros(len(lit))
    for s0 in range(0, len(pat), chunk):
        p = pat[s0:s0 + chunk]
        obs = R[p][:, lit] >= 0
        ch = des["chan"][s0:s0 + chunk].astype(float)
        if mixed != "map":
            f = des["frac_tf"][:, des["tf_idx"][s0:s0 + chunk]].T
            mx = obs & ~des["rep_real"][s0:s0 + chunk] & (f > 0) & (f < 1)
            ch = np.where(mx, 1.0 if mixed == "hi" else 0.0, ch)
        anycov = des["cov"][s0:s0 + chunk].any(1)
        Wz = np.broadcast_to(w, (len(p), S)) if W is None else W[des["stratum_of_pattern"][p]]
        for s in range(S):
            q = np.where(obs, th[:, s][None, :], 0.0)
            pc = 1.0 - np.prod(1.0 - q * ch, 1)
            out["n_cond"] += float((Wz[:, s] * pc).sum())
            out["n_cc_observed"] += float((Wz[:, s] * pc * anycov).sum())
            out["n_ge1_lit_report"] += float((Wz[:, s] * (1.0 - np.prod(1.0 - q, 1))).sum())
            pos += (Wz[:, s][:, None] * q).sum(0)
    out["lit_positives"] = pos.tolist()
    return out


def lit_node_check(u: str, w, theta) -> list:
    """Why the report-dependent LIT node-set masks are not regenerated (f2 minor 7): per LIT annotator, the
    universe cells it would observe without the node condition (tf covered, g not one of its nodes), the
    reports the fitted rates would place there (prior predictive w . theta_j), its real reports, and the node
    genes of the universe whose membership rests only on in-universe records (a simulation would regenerate
    exactly these memberships from simulated reports)."""
    from r2 import evidence as E
    des = cc_design(u)
    R, lit, pat, ann = des["R"], des["lit"], des["dom_pattern"], des["annotators"]
    tfs, genes = _universe_axes(u)
    tfset, gset = set(tfs), set(genes)
    tf_in_genes = np.array([t in gset for t in tfs])
    ed = E.variant_edges(VARIANT)
    bank = E.bank_for_variant(VARIANT)
    w, theta = np.asarray(w, float), np.asarray(theta, float)
    rows = []
    for a, j in enumerate(lit):
        s = bank.sources.index(ann[j])
        cov_u = np.array([t in bank.cov[s] for t in tfs])
        n_cells_cov = int(cov_u.sum() * len(genes) - (cov_u & tf_in_genes).sum())
        n_obs = int((R[pat, j] >= 0).sum())
        rate = float(w @ theta[j] / w.sum())
        e = ed[ed["source_id"] == ann[j]]
        in_u = e["tf"].isin(tfset).to_numpy() & e["target"].isin(gset).to_numpy() & (e["tf"] != e["target"]).to_numpy()
        genes_rec = pd.concat([pd.DataFrame({"g": e["tf"], "in_u": in_u}), pd.DataFrame({"g": e["target"], "in_u": in_u})])
        sup = genes_rec.groupby("g")["in_u"].agg(["all", "size"])
        node_u = sup[sup.index.isin(gset)]
        dep = node_u[node_u["all"]]
        rows.append(dict(universe=u, annotator=ann[j], n_cells_tf_covered=n_cells_cov, n_cells_observed=n_obs,
                         n_cells_off_node=n_cells_cov - n_obs, prior_predictive_rate=rate,
                         expected_reports_off_node=rate * (n_cells_cov - n_obs),
                         real_reports=int((R[pat, j] == 1).sum()), n_node_genes_in_universe=int(len(node_u)),
                         n_node_genes_only_in_universe_records=int(len(dep)),
                         frac_node_genes_report_dependent=float(len(dep) / max(len(node_u), 1))))
    return rows


def fixed_mask_design_stats(u: str) -> dict:
    """Design statistics (``pattern_design_stats``) of the stored fixed-mask replicates of label u, by exact
    re-simulation (cached in ``cc_design/fixed_mask_design_stats_<u>.json``)."""
    f = CC_DIR / f"fixed_mask_design_stats_{u}.json"
    have = json.loads(f.read_text()) if f.exists() else {}
    need = [int(r["replicate"]) for r in load_sim(u) if str(int(r["replicate"])) not in have]
    if need:
        data, truth = load_data(SIM_TRUTH[u][0]), truth_params(u)
        for r in need:
            sim, _ = FS.simulate_fixed_masks(data, truth["w"], truth["theta"], seed=sim_seed(u, r))
            have[str(r)] = pattern_design_stats(sim)
        C.ensure_dir(CC_DIR)
        C.write_json(_jsonable(dict(sorted(have.items(), key=lambda kv: int(kv[0])))), f)
    return {int(k): v for k, v in have.items()}


DESIGN_SCALARS = ("n_edges_observed", "n_edges_ge1_lit_report", "n_cond", "n_edges_cc_observed",
                  "n_edges_cc_observed_with_lit_report", "n_cc_observations", "n_cc_observations_on_lit_reported")


def faithful_design_table(labels=SIM_LABELS) -> pd.DataFrame:
    """Validation of the simulators against the real design (``modes_faithful_design.csv``): per data universe
    and simulation label, the real value, the replicate mean / SD / min / max and (faithful simulators) the
    analytic expectation at the generating parameters of every design statistic, overall and per annotator."""
    rows = []
    for u in sorted({SIM_TRUTH[x][0] for x in labels}):
        des = cc_design(u)
        real = pattern_design_stats(lcm.PatternData(des["R"], des["n"], des["family"], des["annotators"]))
        real["n_cond"] = des["n_cond_real"]
        ann, lit, ccj = des["annotators"], des["lit"], des["cc"]
        b0 = dict(universe=u, sim_label="", simulator="", truth_real_optimum=np.nan, annotator="",
                  analysis_status=STATUS)
        rows.append(dict(b0, quantity="mask_reconstruction_exact", real=float(des["mask_reconstruction_exact"]),
                         note="cc rule applied to the real LIT reports + real via_dorothea flags + whole-resource "
                              "TF coverage reproduces every real MOTIF/COEXP mask"))
        rows.append(dict(b0, quantity="n_domain_edges_lit_observed", real=float(des["n_dom"])))
        rows.append(dict(b0, quantity="n_lit_mask_strata", real=float(len(des["stratum_codes"]))))
        for k_, v_ in des["chan_rule_counts"].items():
            rows.append(dict(b0, quantity=f"n_lit_observed_cells_channel_rule_{k_}", real=float(v_),
                             note="channel membership of a simulated LIT report: real flag / pure-source constant / "
                                  "per-(source, TF) constant / per-(source, TF) Bernoulli (fixed map)"))
        for lab in [x for x in labels if SIM_TRUTH[x][0] == u]:
            recs = load_sim(lab) if (MODOUT / "sim" / lab).exists() else []
            if not recs:
                continue
            simr = sim_simulator(lab)
            stats = ([r["design"] for r in recs] if simr != "fixed_masks"
                     else list(fixed_mask_design_stats(lab).values()))
            truth = truth_params(lab)
            ex = lo = hi = None
            if simr != "fixed_masks":
                W = strat_weights(lab)[0] if simr == "faithful_cc_strat" else None
                ex = expected_cc_counts(des, truth["w"], truth["theta"], W=W)
                lo = expected_cc_counts(des, truth["w"], truth["theta"], W=W, mixed="lo")
                hi = expected_cc_counts(des, truth["w"], truth["theta"], W=W, mixed="hi")
            b = dict(universe=u, sim_label=lab, simulator=simr, truth_real_optimum=SIM_TRUTH[lab][1], analysis_status=STATUS)

            def put(quantity, annotator, rv, vals, e=np.nan, elo=np.nan, ehi=np.nan, note=""):
                v = np.asarray([x for x in vals if x is not None and np.isfinite(x)], float)
                rows.append(dict(b, quantity=quantity, annotator=annotator, real=rv, n_replicates=len(v),
                                 sim_mean=float(v.mean()) if len(v) else np.nan,
                                 sim_sd=float(v.std(ddof=1)) if len(v) > 1 else np.nan,
                                 sim_min=float(v.min()) if len(v) else np.nan, sim_max=float(v.max()) if len(v) else np.nan,
                                 expected=e, expected_mixed_channel_off=elo, expected_mixed_channel_on=ehi,
                                 ratio_sim_mean_to_real=float(v.mean() / rv) if len(v) and rv else np.nan, note=note))
            for q in DESIGN_SCALARS:
                e_ = {"n_cond": "n_cond", "n_edges_cc_observed": "n_cc_observed",
                      "n_edges_ge1_lit_report": "n_ge1_lit_report"}.get(q)
                put(q, "", float(real[q]), [s.get(q, np.nan) for s in stats],
                    ex[e_] if ex and e_ else np.nan, lo[e_] if lo and e_ else np.nan, hi[e_] if hi and e_ else np.nan,
                    note=("not simulated by the fixed-mask simulator" if q == "n_cond" and simr == "fixed_masks" else ""))
            for j, a in enumerate(ann):
                el = ex["lit_positives"][list(lit).index(j)] if (ex and j in lit) else np.nan
                put("n_reported", a, float(real["n_reported"][j]), [s["n_reported"][j] for s in stats], el)
                if j in ccj:
                    put("n_observed", a, float(real["n_observed"][j]), [s["n_observed"][j] for s in stats])
    return pd.DataFrame(rows)


def lit_node_table(universes=("immune", "global")) -> pd.DataFrame:
    """``modes_lit_node_check.csv``: why the LIT node-set masks are not regenerated (``lit_node_check``)."""
    rows = []
    for u in universes:
        t = real_mode_params(u, 0)
        rr = lit_node_check(u, t["w"], t["theta"])
        df = pd.DataFrame(rr)
        tot = df.drop(columns=["universe", "annotator", "prior_predictive_rate", "frac_node_genes_report_dependent"]).sum()
        tot = dict(tot, universe=u, annotator="__ALL__", prior_predictive_rate=np.nan,
                   frac_node_genes_report_dependent=float(tot["n_node_genes_only_in_universe_records"]
                                                          / max(tot["n_node_genes_in_universe"], 1)))
        rows += rr + [tot]
    out = pd.DataFrame(rows)
    out["generating_parameters"] = "primary Family-HB MAP optimum 0"
    out["analysis_status"] = STATUS
    return out


# ----------------------------------------------------------------------------------------
# (4) recovery-table fixes
# ----------------------------------------------------------------------------------------
def ece_bins(g: pd.DataFrame, lo_bin: int = 1) -> float:
    g = g[g["bin"] >= lo_bin]
    nb = g["n_edges"].to_numpy(float)
    if nb.sum() == 0:
        return float("nan")
    mp = g["sum_pred"].to_numpy() / np.maximum(nb, 1)
    tr = g["n_true"].to_numpy() / np.maximum(nb, 1)
    return float(np.sum(nb * np.abs(mp - tr)) / nb.sum())


def recovery_calibration_raw(u: str) -> pd.DataFrame:
    od = FS.MOD / "recovery" / u
    cals = []
    for f in sorted(od.glob("rep*.npz")):
        import io
        z = np.load(f, allow_pickle=False)
        r = int(f.stem[3:])
        cb = pd.read_json(io.StringIO(str(z["calib"])))
        cb["replicate"] = r
        cals.append(cb)
        tsf = od / f"truthstart_rep{r:02d}.json"
        if tsf.exists():
            t = json.loads(tsf.read_text())
            if "calib" in t:
                cb2 = pd.DataFrame(t["calib"])
                cb2["estimator"] = "family_hb_truthstart"
                cb2["replicate"] = r
                cals.append(cb2)
    return pd.concat(cals, ignore_index=True) if cals else pd.DataFrame()


def single_mode_coverage(u: str = FS.NUTS_UNIVERSE) -> tuple[list, list]:
    """NUTS coverage of the recovery-NUTS replicates from converged single-mode chain sets only."""
    import arviz as az
    nd = FS.MOD / "recovery_nuts" / u
    rows, reps = [], []
    for f in sorted(nd.glob("rep*.json")):
        rec = json.loads(f.read_text())
        r = int(rec["replicate"])
        so = MODOUT / "sim" / u / f"rep{r:03d}_optima.npz"
        sj = MODOUT / "sim" / u / f"rep{r:03d}.json"
        if not so.exists():
            continue
        opt = np.load(so)
        srec = json.loads(sj.read_text())
        dz = np.load(nd / f"rep{r:02d}_draws.npz")
        W, TH, ch = np.asarray(dz["w"]), np.asarray(dz["theta"]), np.asarray(dz["chain"])
        n_p = opt["n"]
        oc = opt["cred"]                           # (K_opt, P_sim)
        # chain classification: nearest optimum by mean credibility over <= 50 thinned draws
        data = load_data(u)
        tw, tth = truth_params(u)["w"], truth_params(u)["theta"]
        sim, _ = FS.simulate_fixed_masks(data, tw, tth, seed=derived_seed("recovery", u, r))
        if sim.P != oc.shape[1] or not np.allclose(sim.n, n_p):
            raise RuntimeError(f"recovery-NUTS replicate {r}: simulated data differ from the stored optima")
        cls = {}
        for c_ in np.unique(ch):
            ii = np.flatnonzero(ch == c_)
            ii = ii[np.linspace(0, len(ii) - 1, min(50, len(ii))).astype(int)]
            cm = np.mean([lcm.credibility_at(dict(w=W[i], theta=TH[i]), sim) for i in ii], 0)
            cls[int(c_)] = int(np.argmin(wmae(oc, cm, n_p)))
        truth_k = int(srec["truth_mode_index"])
        for k in sorted(set(cls.values())):
            chains = sorted(c_ for c_, v in cls.items() if v == k)
            m = np.isin(ch, chains)
            base = dict(universe=u, replicate=r, optimum=k, is_truth_mode=bool(k == truth_k), chains=chains,
                        n_chains=len(chains))
            if len(chains) >= 2:
                nmin = min(int((ch == c_).sum()) for c_ in chains)
                arr = np.concatenate([np.stack([W[ch == c_][:nmin] for c_ in chains]),
                                      np.stack([TH[ch == c_][:nmin].reshape(nmin, -1) for c_ in chains])], axis=2)
                ds = az.convert_to_dataset({"x": arr})
                rh = float(np.nanmax(az.rhat(ds)["x"].values))
                es = float(np.nanmin(az.ess(ds, method="bulk")["x"].values))
                conv = bool(rh <= RHAT_OK and es >= ESS_OK)
                conv_f = bool(rh <= RHAT_FROZEN and es >= ESS_FROZEN)
            else:
                rh, es, conv, conv_f = np.nan, np.nan, False, False
            base.update(r_hat_max_w_theta=rh, ess_bulk_min_w_theta=es, converged=conv,
                        status="assessable" if conv else ("single_chain" if len(chains) < 2 else "not_converged"),
                        converged_frozen_rule=conv_f,
                        status_frozen_rule="assessable" if conv_f else (
                            "single_chain" if len(chains) < 2 else "flagged_rhat_gt_1.01_or_ess_lt_400"))
            if conv:
                lo_w, hi_w = np.quantile(W[m], [0.025, 0.975], axis=0)
                lo_t, hi_t = np.quantile(TH[m], [0.025, 0.975], axis=0)
                tt = np.asarray(rec["theta_true"])
                base.update(w_T_lo=float(lo_w[0]), w_T_hi=float(hi_w[0]), w_T_mean=float(W[m, 0].mean()),
                            w_T_covered=bool(lo_w[0] <= tw[0] <= hi_w[0]),
                            alpha_covered=(lo_t[:, 0] <= tt[:, 0]) & (tt[:, 0] <= hi_t[:, 0]),
                            beta_covered=(lo_t[:, 1] <= tt[:, 1]) & (tt[:, 1] <= hi_t[:, 1]),
                            alpha_width=hi_t[:, 0] - lo_t[:, 0], beta_width=hi_t[:, 1] - lo_t[:, 1])
            reps.append(base)
    fam = np.asarray(json.loads(sorted(nd.glob("rep*.json"))[0].read_text())["family"]) if reps else np.array([])
    rules = (("nuts_coverage_single_mode", "converged", RHAT_OK, ESS_OK,
              "lenient rule of this module (2-3 chains of 250 thinned draws); NOT the project's frozen flag rule"),
             ("nuts_coverage_single_mode_frozen_rule", "converged_frozen_rule", RHAT_FROZEN, ESS_FROZEN,
              "project's frozen NUTS flag rule (plan section 7 item 12)"))
    for kind, ckey, rh_ok, es_ok, rule_note in rules:
        for scope in ("truth_mode", "competing_mode"):
            sel = [x for x in reps if x["is_truth_mode"] == (scope == "truth_mode")]
            ok = [x for x in sel if x[ckey]]
            base = dict(universe=u, kind=kind, estimator=f"family_hb_nuts_{scope}", convergence_rule=
                        f"r_hat <= {rh_ok:g} and bulk ESS >= {es_ok:g}",
                        n_replicates=len({x["replicate"] for x in sel}), analysis_status=STATUS)
            if not ok:
                rows.append(dict(base, quantity="w_T_coverage95", family="all", n_values=0, value_mean=np.nan,
                                 status="not_assessable",
                                 note=f"{len(sel)} chain sets; none passes {rule_note} with >= 2 chains in the mode"))
                continue
            note = (f"{len(ok)}/{len(sel)} chain sets assessable (coverage rests on {len(ok)} replicates); "
                    f"R-hat <= {rh_ok}, bulk ESS >= {es_ok:g} over w and theta (250 thinned draws per chain): {rule_note}")
            rows.append(dict(base, quantity="w_T_coverage95", family="all", n_values=len(ok),
                             value_mean=float(np.mean([x["w_T_covered"] for x in ok])),
                             ci_width_mean=float(np.mean([x["w_T_hi"] - x["w_T_lo"] for x in ok])),
                             status="assessable", note=note))
            for par in ("alpha", "beta"):
                cov = np.concatenate([x[f"{par}_covered"] for x in ok])
                wid = np.concatenate([x[f"{par}_width"] for x in ok])
                famv = np.concatenate([fam] * len(ok))
                rows.append(dict(base, quantity=f"{par}_coverage95", family="all", n_values=len(cov),
                                 value_mean=float(cov.mean()), ci_width_mean=float(wid.mean()), status="assessable", note=note))
                for fm in sorted(set(fam)):
                    mm = famv == fm
                    rows.append(dict(base, quantity=f"{par}_coverage95", family=fm, n_values=int(mm.sum()),
                                     value_mean=float(cov[mm].mean()), ci_width_mean=float(wid[mm].mean()),
                                     status="assessable", note=note))
    rep_rows = [{k: (v if not isinstance(v, np.ndarray) else None) for k, v in x.items()} for x in reps]
    for x in rep_rows:
        x["chains"] = ",".join(map(str, x["chains"]))
        for k in ("alpha_covered", "beta_covered", "alpha_width", "beta_width"):
            x.pop(k, None)
    return rows, rep_rows


RECOVERY_POST_HOC_COLUMNS = {
    "fit_recovery": "status;note;mode;simulator;convergence_rule",
    "fit_recovery_calibration": "in_ece_bins1_9;simulator",
    "fit_recovery_replicates": ("ece_bins1_9;replay_matches_stored_fit;mode;gap_to_second_optimum_20restarts;strong_n;"
                                "strong_claimed_bfdr;strong_realised_fdr;ge_moderate_n;ge_moderate_claimed_bfdr;"
                                "ge_moderate_realised_fdr;simulator;competing_by_spearman_rule"),
}


def faithful_recovery_tables(labels=None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """The prespecified recovery study (plan section 6 (iii)) repeated under the faithful simulators: the first
    ``N_RECOVERY_FROM_SIM`` replicates (immune 20, global 5) of each faithful simulation generated at the
    real optimum 0.  Family-HB = best of the 10 set-A restarts (the recovery-study fit), EM 10 restarts,
    truth-start, oracle = Bayes posterior under the generating process.  Same row layout as
    ``fit_sensitivity.recovery_tables`` / ``augment_recovery_tables`` (+ ``simulator``); all post hoc."""
    labels = labels or [x for x in FAITHFUL_LABELS if SIM_TRUTH[x][1] == 0]
    rec_rows, cal_rows, rep_rows = [], [], []
    for lab in labels:
        u, simr = SIM_LABEL_INFO[lab]
        recs = [r for r in (load_sim(lab) if (MODOUT / "sim" / lab).exists() else [])
                if r["replicate"] < N_RECOVERY_FROM_SIM[u]]
        if not recs:
            continue
        truth = truth_params(lab)
        tth = np.asarray(truth["theta"])
        fam = np.asarray(load_data(u).family)
        base = dict(universe=u, simulator=simr, analysis_status=STATUS)
        prm, cals = [], []
        for r in recs:
            rr = int(r["replicate"])
            ex = np.load(MODOUT / "sim" / lab / f"rep{rr:03d}_optima.npz")
            ka = int(r["recovery_fit"]["optimum"])
            mode = "truth" if r["recovery_fit"]["is_truth_mode"] else "competing"
            competing_sp = bool(r["truthstart_spearman_oracle"] - r["recovery_fit"]["spearman_oracle"] > 0.05)
            ests = {"family_hb": (r["recovery_fit"], "recovery_fit", ex["theta"][ka], ex["w"][ka]),
                    "em": (r["em"], "em", ex["em_theta"], ex["em_w"]),
                    "family_hb_truthstart": (r["truthstart"], "truthstart", ex["truthstart_theta"], ex["truthstart_w"]),
                    "oracle": (r["oracle"], "oracle", None, None)}
            for est, (src, cname, th, w_) in ests.items():
                cb = pd.DataFrame(r["calib"][cname])
                cb["bin"] = np.arange(len(cb))
                cb["replicate"], cb["estimator"] = rr, est
                cals.append(cb)
                row = dict(universe=u, replicate=rr, estimator=est, auroc=src.get("auroc"),
                           spearman_oracle=src.get("spearman_oracle", 1.0 if est == "oracle" else np.nan),
                           ece=ece_bins(cb, lo_bin=0), ece_bins1_9=ece_bins(cb), w_T_true=r["w_T_true"],
                           w_T_est=float(w_[0]) if w_ is not None else np.nan, log_post=src.get("log_post", np.nan),
                           **{k: src[k] for k in ("strong_n", "strong_claimed_bfdr", "strong_realised_fdr", "ge_moderate_n",
                                                   "ge_moderate_claimed_bfdr", "ge_moderate_realised_fdr")},
                           simulator=simr, analysis_status=STATUS)
                if est == "family_hb":
                    row.update(mode=mode, gap_to_second_optimum_20restarts=r["gap"], competing_by_spearman_rule=competing_sp,
                               truthstart_log_post_minus_random_best=r["truthstart_log_post"] - r["recovery_fit"]["log_post"],
                               truthstart_spearman_oracle=r["truthstart_spearman_oracle"],
                               truthstart_w_T=r["truthstart"]["w_T"])
                rep_rows.append(row)
                if th is not None:
                    prm.append(dict(estimator=est, replicate=rr, param="w_T", family="all", true=r["w_T_true"],
                                    est=float(w_[0]), mode=mode if est == "family_hb" else "all_modes"))
                    for jj in range(len(fam)):
                        for par, col in (("alpha", 0), ("beta", 1)):
                            prm.append(dict(estimator=est, replicate=rr, param=par, family=fam[jj], true=float(tth[jj, col]),
                                            est=float(th[jj, col]), mode=mode if est == "family_hb" else "all_modes"))
        reps_u = pd.DataFrame([x for x in rep_rows if x["universe"] == u and x["simulator"] == simr])
        n_rep = len(recs)
        hb = reps_u[reps_u.estimator == "family_hb"]
        rec_rows.append(dict(base, kind="identifiability", estimator="family_hb", family="all", n_replicates=n_rep,
                             n_values=n_rep, quantity="n_replicates_map_in_competing_mode",
                             value_mean=float(hb.competing_by_spearman_rule.sum()),
                             note="recovery-study rule: truth-start Spearman vs oracle exceeds the fit's by > 0.05"))
        rec_rows.append(dict(base, kind="identifiability", estimator="family_hb", family="all", n_replicates=n_rep,
                             n_values=n_rep, quantity="n_replicates_fit_not_truth_mode",
                             value_mean=float((hb["mode"] == "competing").sum()),
                             note="mode rule of this module: the fit's optimum is not the optimum closest to the oracle"))
        for est, g in reps_u.groupby("estimator"):
            for q in ("auroc", "spearman_oracle", "ece", "ece_bins1_9"):
                v = g[q].dropna()
                if len(v):
                    rec_rows.append(dict(base, kind="credibility", estimator=est, family="all", quantity=q,
                                         n_replicates=n_rep, n_values=len(v), value_mean=float(v.mean()),
                                         value_sd=float(v.std(ddof=1)) if len(v) > 1 else np.nan,
                                         value_min=float(v.min()), value_max=float(v.max())))
            for q in ("strong_realised_fdr", "ge_moderate_realised_fdr", "strong_n", "ge_moderate_n", "strong_claimed_bfdr"):
                v = g[q].dropna()
                if len(v):
                    rec_rows.append(dict(base, kind="realised_fdr", estimator=est, family="all", quantity=q,
                                         n_replicates=n_rep, n_values=len(v), value_mean=float(v.mean()),
                                         value_sd=float(v.std(ddof=1)) if len(v) > 1 else np.nan,
                                         value_min=float(v.min()), value_max=float(v.max())))
            v = g["strong_realised_fdr"].dropna()
            rec_rows.append(dict(base, kind="realised_fdr", estimator=est, family="all",
                                 quantity="n_replicates_strong_fdr_gt_0.10", n_replicates=n_rep, n_values=len(v),
                                 value_mean=float((v > 0.10).sum())))
        for mode, g in hb.groupby("mode"):
            for q in ("strong_realised_fdr", "ge_moderate_realised_fdr", "strong_n"):
                v = g[q].dropna()
                rec_rows.append(dict(base, kind="realised_fdr_by_mode", estimator="family_hb", family="all", quantity=q,
                                     mode=mode, n_replicates=len(g), n_values=len(v), value_mean=float(v.mean()),
                                     value_sd=float(v.std(ddof=1)) if len(v) > 1 else np.nan,
                                     value_min=float(v.min()), value_max=float(v.max())))
        pr = pd.DataFrame(prm)
        pr["err"] = pr["est"] - pr["true"]
        for (est, mode, par), g in pr.groupby(["estimator", "mode", "param"]):
            for fm, gg in [("all", g)] + ([] if par == "w_T" else list(g.groupby("family"))):
                rec_rows.append(dict(base, kind="parameter_by_mode", estimator=est, quantity=par, family=fm, mode=mode,
                                     n_replicates=gg.replicate.nunique(), n_values=len(gg), true_mean=gg["true"].mean(),
                                     est_mean=gg["est"].mean(), bias=gg["err"].mean(),
                                     rmse=float(np.sqrt(np.mean(gg["err"] ** 2)))))
        g = pr[(pr.estimator == "family_hb")]
        for par, gp in g.groupby("param"):
            rec_rows.append(dict(base, kind="parameter_by_mode", estimator="family_hb", quantity=par, family="all",
                                 mode="all_modes", n_replicates=gp.replicate.nunique(), n_values=len(gp),
                                 true_mean=gp["true"].mean(), est_mean=gp["est"].mean(), bias=gp["err"].mean(),
                                 rmse=float(np.sqrt(np.mean(gp["err"] ** 2)))))
        calc = pd.concat(cals, ignore_index=True)
        for (est, b), g in calc.groupby(["estimator", "bin"]):
            nb = g["n_edges"].sum()
            cal_rows.append(dict(universe=u, estimator=est, bin=int(b), bin_lo=b / N_CAL_BINS, bin_hi=(b + 1) / N_CAL_BINS,
                                 n_replicates=n_rep, n_edges_total=nb, n_edges_per_replicate=nb / n_rep,
                                 mean_predicted=g["sum_pred"].sum() / nb if nb else np.nan,
                                 empirical_true_rate=g["n_true"].sum() / nb if nb else np.nan,
                                 true_rate_sd_across_replicates=float(np.std(
                                     (g["n_true"] / g["n_edges"].replace(0, np.nan)).dropna(), ddof=1))
                                 if (g["n_edges"] > 0).sum() > 1 else np.nan,
                                 analysis_status=STATUS, in_ece_bins1_9=bool(b >= 1), simulator=simr))
        for est in calc.estimator.unique():
            pooled = calc[calc.estimator == est].groupby("bin")[["n_edges", "sum_pred", "n_true"]].sum().reset_index()
            rec_rows.append(dict(base, kind="credibility", estimator=est, family="all", quantity="ece_bins1_9_pooled",
                                 n_replicates=n_rep, n_values=1, value_mean=ece_bins(pooled)))
    return pd.DataFrame(rec_rows), pd.DataFrame(cal_rows), pd.DataFrame(rep_rows)


def augment_recovery_tables(rec: pd.DataFrame, cal: pd.DataFrame, reps: pd.DataFrame) -> tuple:
    """Add the post hoc columns / rows to the tables built by ``fit_sensitivity.recovery_tables``."""
    rec, cal, reps = rec.copy(), cal.copy(), reps.copy()
    rec["analysis_status"] = "prespecified"
    rec["status"] = ""
    rec["note"] = ""
    m_old_nuts = rec["kind"].isin(["nuts_coverage", "nuts_credibility"])
    rec.loc[m_old_nuts, "status"] = "not_valid_mixed_modes"
    rec.loc[m_old_nuts, "note"] = ("chains of these replicates sit in different modes (R-hat 1.53-1.75); "
                                   "superseded by kind = nuts_coverage_single_mode")
    cal["analysis_status"] = "prespecified"
    cal["in_ece_bins1_9"] = cal["bin"] >= 1
    reps["analysis_status"] = "prespecified"
    reps["analysis_status_new_columns"] = STATUS
    new_rows = []
    for u in sorted(reps["universe"].unique()):
        sims = {r["replicate"]: r for r in load_sim(u)}
        raw = recovery_calibration_raw(u)
        # per-replicate ECE over bins 1-9 and realised FDR from the exact replay
        for i, row in reps[reps.universe == u].iterrows():
            r, est = int(row["replicate"]), row["estimator"]
            g = raw[(raw.replicate == r) & (raw.estimator == est)]
            if len(g):
                reps.loc[i, "ece_bins1_9"] = ece_bins(g)
            s = sims.get(r)
            if s is None:
                continue
            src = {"family_hb": s["recovery_fit"], "em": s["em"], "oracle": s["oracle"]}.get(est)
            if est == "family_hb":
                ok = abs(s["recovery_fit"]["log_post"] - float(row["log_post"])) < 1e-6 * max(1.0, abs(float(row["log_post"])))
                reps.loc[i, "replay_matches_stored_fit"] = bool(ok)
                reps.loc[i, "mode"] = "truth" if s["recovery_fit"]["is_truth_mode"] else "competing"
                reps.loc[i, "gap_to_second_optimum_20restarts"] = s["gap"]
            if est == "em":
                ok = abs(s["em"]["log_post"] - float(row["log_post"])) < 1e-6 * max(1.0, abs(float(row["log_post"])))
                reps.loc[i, "replay_matches_stored_fit"] = bool(ok)
            if src is not None:
                for k in ("strong_n", "strong_claimed_bfdr", "strong_realised_fdr", "ge_moderate_n",
                          "ge_moderate_claimed_bfdr", "ge_moderate_realised_fdr"):
                    reps.loc[i, k] = src[k]
        rts = raw[raw.estimator == "family_hb_truthstart"]
        ts_ece = pd.Series({r_: ece_bins(g_) for r_, g_ in rts.groupby("replicate")}, dtype=float)
        sub = reps[reps.universe == u]
        base = dict(universe=u, analysis_status=STATUS, family="all")
        for est, g in sub.groupby("estimator"):
            if "ece_bins1_9" in g and g["ece_bins1_9"].notna().any():
                new_rows.append(dict(base, kind="credibility", estimator=est, quantity="ece_bins1_9",
                                     n_replicates=len(g), n_values=int(g.ece_bins1_9.notna().sum()),
                                     value_mean=g.ece_bins1_9.mean(), value_sd=g.ece_bins1_9.std(ddof=1),
                                     value_min=g.ece_bins1_9.min(), value_max=g.ece_bins1_9.max()))
                rg = raw[raw.estimator == est]
                pooled = rg.groupby("bin")[["n_edges", "sum_pred", "n_true"]].sum().reset_index()
                new_rows.append(dict(base, kind="credibility", estimator=est, quantity="ece_bins1_9_pooled",
                                     n_replicates=len(g), n_values=1, value_mean=ece_bins(pooled)))
            for k in ("strong_realised_fdr", "ge_moderate_realised_fdr", "strong_n", "ge_moderate_n",
                      "strong_claimed_bfdr"):
                if k in g and g[k].notna().any():
                    v = g[k].dropna()
                    new_rows.append(dict(base, kind="realised_fdr", estimator=est, quantity=k, n_replicates=len(g),
                                         n_values=len(v), value_mean=v.mean(), value_sd=v.std(ddof=1),
                                         value_min=v.min(), value_max=v.max()))
            if "strong_realised_fdr" in g and g["strong_realised_fdr"].notna().any():
                v = g["strong_realised_fdr"].dropna()
                new_rows.append(dict(base, kind="realised_fdr", estimator=est, quantity="n_replicates_strong_fdr_gt_0.10",
                                     n_replicates=len(g), n_values=len(v), value_mean=float((v > 0.10).sum())))
        if len(ts_ece):
            new_rows.append(dict(base, kind="credibility", estimator="family_hb_truthstart", quantity="ece_bins1_9",
                                 n_replicates=len(ts_ece), n_values=len(ts_ece), value_mean=ts_ece.mean(),
                                 value_sd=ts_ece.std(ddof=1), value_min=ts_ece.min(), value_max=ts_ece.max()))
        # realised FDR by mode (family_hb)
        hb = sub[(sub.estimator == "family_hb") & sub.get("mode", pd.Series(index=sub.index, dtype=object)).notna()] \
            if "mode" in sub else sub.iloc[0:0]
        for mode, g in hb.groupby("mode"):
            for k in ("strong_realised_fdr", "ge_moderate_realised_fdr", "strong_n"):
                v = g[k].dropna()
                new_rows.append(dict(base, kind="realised_fdr_by_mode", estimator="family_hb", quantity=k, mode=mode,
                                     n_replicates=len(g), n_values=len(v), value_mean=v.mean(),
                                     value_sd=v.std(ddof=1) if len(v) > 1 else np.nan, value_min=v.min(), value_max=v.max()))
        # parameter recovery (w_T, alpha, beta) by mode, from the stored parameter rows
        import io
        prm = []
        for f in sorted((FS.MOD / "recovery" / u).glob("rep*.npz")):
            z = np.load(f, allow_pickle=False)
            p = pd.read_json(io.StringIO(str(z["params"])))
            prm.append(p[p.estimator == "family_hb"])
        if prm and len(hb):
            prm = pd.concat(prm, ignore_index=True)
            mode_of = dict(zip(hb.replicate.astype(int), hb["mode"]))
            prm["mode"] = prm["replicate"].map(mode_of)
            prm["err"] = prm["est"] - prm["true"]
            for (mode, par), g in prm.dropna(subset=["mode"]).groupby(["mode", "param"]):
                if par not in ("w_T", "alpha", "beta"):
                    continue
                nrep = g.replicate.nunique()
                new_rows.append(dict(base, kind="parameter_by_mode", estimator="family_hb", quantity=par, mode=mode,
                                     n_replicates=nrep, n_values=len(g), true_mean=g["true"].mean(),
                                     est_mean=g["est"].mean(), bias=g["err"].mean(),
                                     rmse=float(np.sqrt(np.mean(g["err"] ** 2))),
                                     note=f"mode = {mode}: replicate fits in the {mode} mode"))
            g = prm[prm.param == "w_T"]
            new_rows.append(dict(base, kind="parameter_by_mode", estimator="family_hb", quantity="w_T", mode="all_modes",
                                 n_replicates=g.replicate.nunique(), n_values=len(g), true_mean=g["true"].mean(),
                                 est_mean=g["est"].mean(), bias=g["err"].mean(),
                                 rmse=float(np.sqrt(np.mean(g["err"] ** 2)))))
    cov_rows, cov_reps = single_mode_coverage()
    new_rows += cov_rows
    rec = pd.concat([rec, pd.DataFrame(new_rows)], ignore_index=True)
    # the rows above come from the fixed-mask simulator; the recovery study under the faithful simulators is
    # appended with simulator = faithful_cc / faithful_cc_strat (modes2 fix of the modes verifier's B1 (c))
    for df in (rec, cal, reps):
        df["simulator"] = "fixed_masks"
    f_rec, f_cal, f_reps = faithful_recovery_tables()
    rec = pd.concat([rec, f_rec], ignore_index=True) if len(f_rec) else rec
    cal = pd.concat([cal, f_cal], ignore_index=True) if len(f_cal) else cal
    reps = pd.concat([reps, f_reps], ignore_index=True) if len(f_reps) else reps
    rec["post_hoc_columns"] = RECOVERY_POST_HOC_COLUMNS["fit_recovery"]
    cal["post_hoc_columns"] = RECOVERY_POST_HOC_COLUMNS["fit_recovery_calibration"]
    reps["post_hoc_columns"] = RECOVERY_POST_HOC_COLUMNS["fit_recovery_replicates"]
    cov = pd.DataFrame(cov_reps)
    if len(cov):
        cov["analysis_status"] = STATUS
    return rec, cal, reps, cov


# ----------------------------------------------------------------------------------------
# (5a) mode gaps of the sensitivity fits
# ----------------------------------------------------------------------------------------
def sensitivity_fit_list(u: str) -> list:
    """(variant, fit_id, analysis) of every stored MAP sensitivity fit of ``fit_sensitivity`` (the
    primary fits are read from the snapshot)."""
    items = [(VARIANT, MODEL, "primary"), (VARIANT, "em", "primary")]
    for v in PS["evidence_variants_sensitivity"]:
        for m in (MODEL, "em"):
            if (fit_dir(u, m, v) / "meta.json").exists():
                items.append((v, m, "evidence_variant"))
    for g in FS.GROUPINGS:
        if (fit_dir(u, f"{MODEL}__fam_{g}") / "meta.json").exists():
            items.append((VARIANT, f"{MODEL}__fam_{g}", "family_grouping"))
    for fid in FS.PRIOR_SENS:
        if (fit_dir(u, fid) / "meta.json").exists():
            items.append((VARIANT, fid, "prior"))
    return items


def run_sensmodes(universes=SENS_UNIVERSES, force: bool = False) -> None:
    for u in universes:
        od = C.ensure_dir(MODOUT / "sensmodes" / u)
        for variant, fid, kind in sensitivity_fit_list(u):
            f = od / f"{variant}__{fid}.json"
            if f.exists() and not force:
                continue
            t0 = time.time()
            d = map_dir(u, fid) if kind == "primary" else fit_dir(u, fid, variant)
            meta = json.loads((d / "meta.json").read_text())
            rs = pd.read_csv(d / "restarts.csv")
            lp = rs["log_post"].to_numpy(float)
            groups = group_optima(lp)
            model = meta["model"]
            prior = meta.get("prior_override") or None
            base = load_data(u, variant)
            data = lcm.with_families(base, meta["family"]) if list(meta["family"]) != list(base.family) else base
            rec = dict(universe=u, variant=variant, fit_id=fid, analysis=kind, model=model,
                       n_restarts=int(meta["n_restarts"]), seed=int(meta["seed"]), n_optima=len(groups),
                       optima_log_post=[float(lp[g[0]]) for g in groups], optima_n_restarts=[len(g) for g in groups],
                       best_log_post=float(lp[groups[0][0]]), prior=prior, family=list(meta["family"]))
            if len(groups) >= 2:
                pt = potential(data, model, prior)
                best = dict(np.load(d / "params.npz"))
                zb = unconstrain_fn(pt["model"], (), {}, {k: jnp.asarray(best[k]) for k in pt["names"]})
                xb = np.asarray(ravel_pytree(zb)[0])
                if abs(-float(pt["pot"](jnp.asarray(xb))) - rec["best_log_post"]) > 1e-3:
                    raise RuntimeError(f"{u}/{variant}/{fid}: stored best does not reproduce its log posterior")
                r2 = int(groups[1][0])
                x2 = replay_restart(data, model, r2, int(meta["n_restarts"]), int(meta["seed"]), prior=prior, pt=pt)
                lp2 = -float(pt["pot"](jnp.asarray(x2)))
                la_b, la_2 = laplace_at(pt, xb), laplace_at(pt, x2)
                lm = np.array([la_b["laplace_log_mass"], la_2["laplace_log_mass"]])
                p2 = float(np.exp(lm[1] - np.logaddexp.reduce(lm))) if np.isfinite(lm).all() else np.nan
                s2 = pt["post"](x2)
                c2 = lcm.credibility_at(s2, data)
                np.save(od / f"{variant}__{fid}__second_cred.npy", c2)
                rec.update(second_restart=r2, replay_log_post_diff=lp2 - float(lp[r2]),
                           gap=rec["best_log_post"] - float(lp[groups[1][0]]),
                           gap_polished=la_b["log_post"] - la_2["log_post"],
                           laplace_log_mass_best=la_b["laplace_log_mass"], laplace_log_mass_second=la_2["laplace_log_mass"],
                           laplace_p2_top2=p2, hess_min_eig_best=la_b["min_eig"], hess_min_eig_second=la_2["min_eig"],
                           w_best=np.asarray(best["w"]).tolist(), w_second=np.asarray(s2["w"]).tolist(),
                           class_names=meta["class_names"])
            rec["runtime_s"] = time.time() - t0
            C.write_json(_jsonable(rec), f)
            log(f"[sensmodes] {u} {variant}/{fid}: {len(groups)} optima, gap {rec.get('gap', np.nan):.3f}, "
                f"Laplace p2 {rec.get('laplace_p2_top2', np.nan):.3g} ({rec['runtime_s']:.0f}s)")


def sensmodes_table(universes=SENS_UNIVERSES) -> pd.DataFrame:
    rows = []
    for u in universes:
        od = MODOUT / "sensmodes" / u
        if not od.exists():
            continue
        ep_cache = {}
        ref_cred = np.load(map_dir(u) / "cred.npy")
        v_hb = FS.EdgeView(u, VARIANT, ref_cred, ep_cache)
        tie_key = FS.tie_key_for(u, len(v_hb.ep))
        # top-k Jaccards are reported only for k <= the number of observed edges ranked above the
        # unobserved-pattern tie block (f1 verifier M2: larger k are dominated by unobserved edges)
        unobs_p = ~v_hb.obs_p
        c_unobs = float(ref_cred[unobs_p].max()) if unobs_p.any() else -np.inf
        k_max = int(np.sum(v_hb.obs & (v_hb.c > c_unobs + 1e-12)))
        for f in sorted(od.glob("*.json")):
            r = json.loads(f.read_text())
            row = {k: r.get(k) for k in ("universe", "variant", "fit_id", "analysis", "model", "n_restarts", "n_optima",
                                         "best_log_post", "gap", "gap_polished", "laplace_log_mass_best",
                                         "laplace_log_mass_second", "laplace_p2_top2", "replay_log_post_diff",
                                         "second_restart")}
            fmeta = json.loads(((map_dir(u, r["fit_id"]) if r["analysis"] == "primary"
                                 else fit_dir(u, r["fit_id"], r["variant"])) / "meta.json").read_text())
            row["t_burst_exchangeable"] = bool(FS.exchangeable_t_burst(fmeta))
            row["topk_k_max_observed_above_unobserved"] = k_max
            if r["n_optima"] < 2:
                row["note"] = ("single optimum among the restarts (within 1e-3)" +
                               ("; T and the single burst class are exchangeable, so the restarts split between "
                                "two label copies with identical log posterior" if row["t_burst_exchangeable"] else ""))
            row["optima_n_restarts"] = ";".join(map(str, r["optima_n_restarts"]))
            row["n_restarts_at_best"] = r["optima_n_restarts"][0]
            row["n_restarts_at_second"] = r["optima_n_restarts"][1] if r["n_optima"] > 1 else 0
            row["optima_gaps"] = ";".join(f"{r['best_log_post'] - x:.3f}" for x in r["optima_log_post"])
            if "w_best" in r:
                row["w_T_best"], row["w_T_second"] = r["w_best"][0], r["w_second"][0]
            c2f = od / f"{r['variant']}__{r['fit_id']}__second_cred.npy"
            if c2f.exists():
                c2 = np.load(c2f)
                alt2 = FS.EdgeView(u, r["variant"], c2, ep_cache)
                cmp2 = FS.compare_views(v_hb, alt2, tie_key)
                for k in list(cmp2):
                    if k.startswith("jaccard_top") and int(k[len("jaccard_top"):]) > k_max:
                        cmp2[k] = np.nan
                row.update({f"second_vs_primary_{k}": v for k, v in cmp2.items()
                            if k.startswith(("spearman", "jaccard", "bfdr"))})
                fb = FS.load_fit(map_dir(u, r["fit_id"]) if r["analysis"] == "primary"
                                 else fit_dir(u, r["fit_id"], r["variant"]))
                cmp_b = FS.compare_views(v_hb, FS.EdgeView(u, r["variant"], fb["cred"], ep_cache), tie_key)
                row.update({f"best_vs_primary_{k}": v for k, v in cmp_b.items()
                            if k in ("spearman_observed", "jaccard_bfdr_strong", "jaccard_bfdr_moderate")})
                cmp_bs = FS.compare_views(FS.EdgeView(u, r["variant"], fb["cred"], ep_cache), alt2, tie_key)
                row["second_vs_best_jaccard_bfdr_strong"] = cmp_bs["jaccard_bfdr_strong"]
                row["second_vs_best_spearman_observed"] = cmp_bs["spearman_observed"]
                if r["fit_id"] == MODEL and r["variant"] == VARIANT:
                    row["note"] = "primary fit (50 restarts); compare fit_map_optima.csv"
                if FS.exchangeable_t_burst(fb["meta"]):
                    row["note"] = ("T and the single burst class are exchangeable: the two optima are label "
                                   "copies; the second-mode comparison is with the swapped labelling")
            row["analysis_status"] = STATUS
            rows.append(row)
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------------------
# (5b) stability of the data-driven grouping
# ----------------------------------------------------------------------------------------
def run_clusterstab(universes=SENS_UNIVERSES, n_boot: int = N_BOOT_CLUSTER) -> pd.DataFrame:
    from sklearn.metrics import adjusted_rand_score as ari
    rows = []
    for u in universes:
        data = load_data(u)
        mod = list(data.family)
        full = FS.datadriven_families(data)[0]
        rows.append(dict(universe=u, source="full_data", index=-1, ari_vs_modality=ari(mod, full), ari_vs_full=1.0,
                         partition=";".join(full), equals_modality=bool(ari(mod, full) == 1.0)))
        for p in sorted((FS.MOD / "cv" / u).glob("fold*__family_hb__fam_datadriven.json")):
            r = json.loads(p.read_text())
            part = r["family"]
            rows.append(dict(universe=u, source="cv_training_fold", index=int(r["fold"]),
                             ari_vs_modality=ari(mod, part), ari_vs_full=ari(full, part), partition=";".join(part),
                             equals_modality=bool(ari(mod, part) == 1.0)))
        ep = FS.edge_pattern(u)
        tf_idx, tfs = FS.edge_tf_index(u)
        valid = ep >= 0
        ep_v, tf_v = ep[valid], tf_idx[valid]
        rng = np.random.default_rng(derived_seed("clusterstab", u))
        co = np.zeros((data.K, data.K))
        for b in range(n_boot):
            mult = np.bincount(rng.integers(0, len(tfs), len(tfs)), minlength=len(tfs)).astype(float)
            nb = np.bincount(ep_v, weights=mult[tf_v], minlength=data.P)
            keep = nb > 0
            db = lcm.PatternData(data.R[keep], nb[keep], list(data.family), list(data.annotators))
            part = FS.datadriven_families(db)[0]
            pa = np.asarray(part)
            co += (pa[:, None] == pa[None, :])
            rows.append(dict(universe=u, source="tf_bootstrap", index=b, ari_vs_modality=ari(mod, part),
                             ari_vs_full=ari(full, part), partition=";".join(part),
                             equals_modality=bool(ari(mod, part) == 1.0)))
        co /= n_boot
        fa = np.asarray(full)
        for j, a in enumerate(data.annotators):
            same = fa == fa[j]
            same[j] = False
            other = fa != fa[j]
            rows.append(dict(universe=u, source="annotator_coassignment", index=j, annotator=a, modality=mod[j],
                             full_data_cluster=full[j],
                             mean_coassign_with_full_cluster=float(co[j, same].mean()) if same.any() else np.nan,
                             max_coassign_with_other=float(co[j, other].max()) if other.any() else np.nan))
        log(f"[clusterstab] {u}: done")
    df = pd.DataFrame(rows)
    df["analysis_status"] = STATUS
    return df


def clusterstab_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (u, src), g in df[df.source.isin(["cv_training_fold", "tf_bootstrap", "full_data"])].groupby(["universe", "source"]):
        rows.append(dict(universe=u, source=src, index=-2, n=len(g), ari_vs_modality=g.ari_vs_modality.mean(),
                         ari_vs_modality_min=g.ari_vs_modality.min(), ari_vs_modality_q10=g.ari_vs_modality.quantile(0.1),
                         ari_vs_full=g.ari_vs_full.mean(), ari_vs_full_min=g.ari_vs_full.min(),
                         frac_equals_modality=g.equals_modality.mean(), frac_equals_full=float((g.ari_vs_full == 1.0).mean()),
                         partition="summary", analysis_status=STATUS))
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------------------
# Tables
# ----------------------------------------------------------------------------------------
def _write(df: pd.DataFrame, name: str) -> Path:
    p = C.ensure_dir(C.TABLES) / name
    df.to_csv(p, index=False, float_format="%.6g")
    log(f"wrote {p.relative_to(C.SP)} ({len(df)} rows)")
    return p


def build_tables() -> None:
    sc = snapshot_check()
    if len(sc):
        sc.to_csv(C.ensure_dir(MODOUT) / "snapshot_check.csv", index=False)
        log(f"[snapshot] MAP snapshot = canonical fits (max |dx| {sc.max_abs_dx_top2.max():.2g}, "
            f"max |dtheta| {sc.max_abs_dtheta_best.max():.2g})")
    bt = bridge_table()
    if len(bt):
        _write(bt, "modes_bridge.csv")
    gs = governance_sensitivity()
    if len(gs):
        _write(gs, "modes_governance_sensitivity.csv")
    df = sim_replicate_frame()
    if len(df):
        _write(df, "modes_simulation_replicates.csv")
        _write(sim_summary(df), "modes_simulation.csv")
        _write(sim_calibration(), "modes_simulation_calibration.csv")
        fd = faithful_design_table()
        if len(fd):
            _write(fd, "modes_faithful_design.csv")
    _write(lit_node_table(), "modes_lit_node_check.csv")
    sm = sensmodes_table()
    if len(sm):
        _write(sm, "fit_sensitivity_modes.csv")
    cs_f = MODOUT / "clusterstab.parquet"
    if cs_f.exists():
        cs = pd.read_parquet(cs_f)
        _write(pd.concat([clusterstab_summary(cs), cs], ignore_index=True), "fit_grouping_stability.csv")
    # recovery tables (rebuilt from the stored recovery outputs + the post hoc augmentation)
    rec, cal, reps = FS.recovery_tables(list(FS.UNIVERSES))
    if len(rec):
        rec, cal, reps, cov = augment_recovery_tables(rec, cal, reps)
        FS._write(rec, "recovery")
        FS._write(cal, "recovery_calibration")
        FS._write(reps, "recovery_replicates")
        if len(cov):
            _write(cov, "fit_recovery_nuts_chainsets.csv")
    record_decisions()


def run_mixture_fits() -> None:
    pb, _ = p2_estimates()
    for u, p2 in pb.items():
        write_mixture_fit(u, p2)
        log(f"[mixture] {u}: p2 = {p2:.3g}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("step", choices=["nuts", "bridge", "mixture", "simulate", "sensmodes", "clusterstab",
                                     "governance", "tables", "all"])
    ap.add_argument("--universes", nargs="+", default=None)
    ap.add_argument("--modes", nargs="+", type=int, default=None,
                    help="default: modes 0 and 1 in the tissues, mode 1 in global (DEFAULT_NUTS_MODES); "
                         "global mode 0 of the bridge estimate is the canonical primary NUTS fit")
    ap.add_argument("--reps", nargs="+", type=int, default=None)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args(argv)
    C.ensure_dir(MODOUT)
    t0 = time.time()
    if a.step in ("nuts", "all"):
        for u in (a.universes or ["immune", "kidney", "lung", "global"]):
            for m in (a.modes if a.modes is not None else DEFAULT_NUTS_MODES[u]):
                run_mode_nuts(u, m, force=a.force)
    if a.step in ("bridge", "all"):
        run_bridge(a.universes or UNIVERSES, force=a.force)
    if a.step in ("mixture", "all"):
        run_mixture_fits()
    if a.step in ("simulate", "all"):
        run_simulation(a.universes or FAITHFUL_LABELS, reps=a.reps, force=a.force)
    if a.step in ("sensmodes", "all"):
        run_sensmodes(a.universes or SENS_UNIVERSES, force=a.force)
    if a.step in ("clusterstab", "all"):
        cs = run_clusterstab(a.universes or SENS_UNIVERSES)
        cs.to_parquet(C.ensure_dir(MODOUT) / "clusterstab.parquet", index=False)
    if a.step == "governance":
        gs = governance_sensitivity()
        if len(gs):
            _write(gs, "modes_governance_sensitivity.csv")
    if a.step in ("tables", "all"):
        build_tables()
    log(f"done {a.step} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
