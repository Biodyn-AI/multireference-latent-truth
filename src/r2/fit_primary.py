"""Primary credibility-model fits (reviewer Q1, Q11, Q13): MAP for all five estimators and
the full NUTS posterior of Family-HB, for the global and the three tissue-context universes.

Inputs
------
Pattern-compressed evidence of the primary variant
(``PRESPECIFIED['evidence_variant_primary']`` = ``prespecified_cc``, the curated-conditional
observation model) from ``evidence.load_patterns(universe, variant)``.  No held-out data
(ChIP references, perturbation cohorts) is read by this module.

Fits (``lcm.py`` implements every model; nothing here changes the model)
------------------------------------------------------------------------
* MAP (``map_restarts``: the multi-start L-BFGS of ``lcm.fit_map`` -- same PRNG keys, start
  points and optimiser options -- on the collapsed log posterior in the unconstrained,
  non-centred parametrisation): ``em``, ``dependence``, ``multifactor``,
  ``family_regularized`` with 20 random restarts and ``family_hb`` with 50.  Every restart's
  L-BFGS end point is then *polished* by damped Newton steps with the exact Hessian of the
  potential (``newton_polish``), and the STORED fit is the polished optimum of the best restart.
  MAP values are joint posterior modes in the unconstrained parametrisation (log tau / omega
  with Jacobian, non-centred z); the MAP hierarchical scales tau, omega are therefore
  joint-mode values, not estimates of their marginal posterior (NUTS gives those).  Plan
  section 6 named Gamma(2, 2) scale priors for MAP; the code uses the HalfNormal(1) priors of
  the NUTS model for both (logged deviation).
* MAP convergence (``GRAD_TOL``, plan section 6): a fit is converged if the polished optimum
  has max|gradient| < 1e-6 and a positive-definite Hessian (strict local mode).  Where
  floating-point resolution of the potential stops the polish first (large universes, |log
  posterior| ~ 3.5e5), the relative criterion -- Newton decrement g'H^-1 g / 2 <=
  ``REL_DECREMENT_TOL`` x machine-eps x |log posterior| at a positive-definite Hessian, i.e.
  no further improvement is resolvable -- is reported separately (``converged_relative``).
  ``n_lbfgs_ftol_stop`` counts restarts whose L-BFGS run ended with scipy's success flag
  (relative-reduction stop); ``n_grad_converged`` counts restarts whose polished end point
  meets the gradient criterion.
* Distinct MAP optima: restarts are grouped into optima (``optima.csv``); every optimum is
  recovered exactly by replaying its first restart and its posterior mass is approximated by
  Laplace's method in the unconstrained space (``laplace`` stage; "Laplace-approximated"
  mass -- the Gaussian approximation is poor for the scale parameters, see the report).
* NUTS for ``family_hb``: settings are FROZEN per universe and fit in ``NUTS_SETTINGS``
  (identical copy committed as ``src/r2/configs/fit_primary_nuts.json``; a test checks the
  two agree): 4 chains x 1000 warm-up + 1000 draws, target acceptance 0.95, for every
  universe and both initialisations.  The wall-clock pilot of the first pass (``--settings
  auto``: 1 chain, 100 + 50 iterations, reduced settings if the projection exceeds
  ``--budget-min``) is kept only as a record/option; it is load-dependent and is not used by
  ``all``.  ``NUTS_SETTINGS_HISTORY`` records the earlier choices (global primary run 4 x
  500/500 from the pilot; Laplace start dispersion changed from 2 to 1 after the superseded
  immune run).  Two initialisations:
    - ``family_hb_randinit_nuts``: ``lcm.fit_nuts`` defaults (uniform(-2, 2) in the
      unconstrained space).  The Family-HB posterior has several well-separated local modes
      (label switches between T and a burst class); chains started at random can settle in
      a minor mode and never leave it, so this run is a multimodality diagnostic.
    - ``family_hb_nuts`` (primary posterior summary): chains start from draws of the Laplace
      approximation around the best MAP optimum (scale ``NUTS_INIT_OVERDISPERSION``; each start
      verified by L-BFGS to lie in the basin of that optimum), i.e. the posterior restricted
      to the dominant mode; its share of the posterior mass is the Laplace relative mass in
      ``optima.csv``.  (The Laplace scale already over-disperses: the potential rises much
      faster than quadratically, so the starts lie hundreds of log-posterior units below the
      optimum, far outside the typical set.)  Every chain's mean credibility is assigned to its nearest MAP optimum
      (``chains.csv``), so a chain leaving the dominant mode would be visible.

Seeds: every fit uses ``PRESPECIFIED['random_seed']`` (MAP restarts, NUTS chains and the
NUTS starting points).  ``all`` runs map -> laplace -> nuts (random, then
map_laplace) -> tables -> edge-tables.

Storage (FIT STORAGE CONTRACT)
------------------------------
``outputs/r2/fits/<universe>/<variant>/<fit_id>/``

* ``cred.npy``     float64 per-pattern credibility P(z = T | pattern), aligned with the rows
                   of ``evidence.load_patterns(universe, variant)``.  For ``*_nuts`` fits this is
                   the posterior mean of the credibility.
* ``params.npz``   ``w`` (class weights) and ``theta`` (K x S reporting probabilities, column
                   order = ``class_names``) plus every other constrained site of the model
                   (MAP: values at the optimum; NUTS: posterior means).
* ``meta.json``    model, class names, annotators, families, prior, restarts, best log posterior,
                   log likelihood, convergence summary, runtime, seed, software versions.
* ``restarts.csv`` (MAP) per restart: L-BFGS log posterior / log likelihood / scipy success flag
                   (``lbfgs_ftol_stop``) / max-abs gradient / iterations / message, and the
                   polished end point (``log_post``, ``loglik``, ``grad_norm``, Hessian minimum
                   eigenvalue, Newton steps, ``grad_converged``); ``cred_restarts.npy``
                   (restarts x P) credibility at every restart's polished optimum;
                   ``optima.csv`` distinct optima (restarts grouped within 1e-3 of the polished
                   log posterior) compared with the best.
* NUTS fits add ``cred_mean.npy``, ``cred_lo.npy``, ``cred_hi.npy`` (2.5% / 97.5% quantiles),
  ``cred_sd.npy``, ``draws.npz`` (``w``, ``theta`` and the annotator/hyper-parameter sites,
  thinned to at most 1000 draws, with ``chain`` ids), ``diagnostics.csv`` (site, r_hat_max,
  ess_bulk_min, ess_tail_min; includes the derived ``theta`` and ``cred`` over observed
  patterns), ``chains.csv`` (per-chain means: a direct check for chains in different modes)
  and ``n_divergent`` in ``meta.json``.

Manuscript tables (``data/r2/tables``)
-------------------------------------
* ``fit_convergence.csv``       one row per universe x fit: MAP restart spread, convergence
                                (``n_lbfgs_ftol_stop``, ``n_grad_converged``, polished gradient,
                                ``converged`` / ``converged_relative``), Hessian diagnostics,
                                runtime; NUTS R-hat / ESS / divergences / runtime, over all sites
                                and over the likelihood-identified sites (w, a, b, g, theta,
                                credibility), the frozen settings, and the sites flagged by the
                                conventional thresholds (``sites_flagged``: split R-hat > 1.01 or
                                bulk ESS < 400).
* ``fit_source_parameters.csv`` per universe x fit x annotator x parameter (alpha, beta,
                                gamma[, gamma2], lr_pos = alpha/beta (vs the clean negative N
                                only), lr_neg = (1-beta)/(1-alpha), lr_pos_marginal = alpha /
                                [sum_{s != T} w_s theta_js / (1 - w_T)] (vs ALL non-T classes,
                                the reliability ranking used in the paper, ``rank`` column = rank
                                of the MAP value / posterior mean; NUTS rows also give the 95%
                                posterior interval of the per-draw rank, ``rank_lo`` / ``rank_hi``)
                                and alpha_over_gamma (T vs the annotator's burst class)): MAP value
                                or NUTS mean with 95% credible interval; annotator coverage.
* ``fit_class_weights.csv``     per universe x fit x latent class: mixture weight and expected
                                number of edges in the class among observed edges, among the
                                edges that can inform the class (``n_edges_informative``: B_f ->
                                observed by >= 1 annotator of family f) and among all universe
                                edges incl. unobserved ones (NUTS: mean, 95% CrI).
* ``fit_hyperparameters.csv``   hierarchical fits: global means, family means and scales
                                (logit scale), MAP (joint-mode values, ``estimate_type``) and
                                NUTS mean with 95% CrI; NUTS rows carry the site-level R-hat /
                                ESS and ``diag_flag`` ('flagged' = R-hat > 1.01 or bulk ESS < 400).
* ``fit_map_optima.csv``        per universe x MAP fit x distinct optimum: restarts reaching it,
                                log posterior gap to the best, expected true edges among observed
                                edges and among all edges (incl. unobserved), agreement of its
                                credibility with the best optimum, Laplace-approximated mass.
* ``fit_nuts_chains.csv``       per universe x NUTS fit x chain: mean log likelihood, class weights,
                                nearest MAP optimum (mode membership of each chain).
* ``fit_map_vs_nuts.csv``       Family-HB MAP vs NUTS posterior-mean credibility: edge-weighted
                                Spearman and absolute differences (incl. max) over observed
                                patterns, agreement of BFDR sets and of top-k sets, CrI widths,
                                CrIs straddling 0.5 among all observed edges and among edges with
                                posterior mean > 0.5.
* ``fit_estimator_agreement.csv`` the same comparison for every pair of fits in a universe
                                (EM baseline vs Family-HB etc.).

Every table carries ``analysis_status`` (row level: 'prespecified' = plan section 6 or fixed
label-free in the first pass, before the project's first held-out outcome; 'post_hoc' = rows that
exist only since the fit_fixes round, e.g. the marginal LR+ rows) and ``post_hoc_columns`` (the
columns added or redefined in that round); the dated decisions (``ANALYSIS_DECISIONS``) are written
to ``eval_analysis_decisions.csv``.  No held-out data is read; 'post hoc' only records that the
quantity was designed after held-out results existed in the project.  Rows of prespecified
quantities can still have been RECOMPUTED after the first held-out outcomes: ``values_changed_post_hoc``
/ ``value_change_note`` (robfix 2026-10-14, verifier fitfix minor 11; ``fit_value_provenance``) mark
rows computed from a MAP fit refitted with Newton polishing (credibility change <= 2.2e-5), from a
dominant-mode NUTS fit re-run with the frozen settings (global: chain length 4 x 500/500 -> 4 x
1000/1000; immune re-run 2026-10-08, kidney / lung / global 2026-10-13) or from first-pass random-init
chains relabelled by patch_randinit_layout.py.

Agreement statistics (``compare_creds``) are computed over OBSERVED edges only (patterns seen by
>= 1 annotator): BFDR sets and top-k sets use edge weights restricted to observed patterns, and a
top-k Jaccard is NA when k exceeds the number of observed edges whose credibility is above the
credibility of unobserved edges (= w_T) in either fit (columns ``n_obs_above_unobs_a/b``).

Per-tissue convenience tables: ``outputs/r2/fits/<tissue>/primary_edge_cred.parquet`` (tf,
target in evidence edge order, pattern index, number of observing / reporting annotators,
credibility of every MAP estimator, NUTS mean / 2.5% / 97.5% / sd).

Command line::

    .venv/bin/python subproject_24_multireference_latent_truth_model/src/r2/fit_primary.py all
    # or stage by stage
    ... fit_primary.py map    [--universes global immune kidney lung] [--models ...] [--skip-existing]
    ... fit_primary.py laplace [--universes ...] [--models family_hb ...]
    ... fit_primary.py nuts   [--universes ...] [--init random map_laplace]
                              [--settings frozen|auto|full|reduced] [--budget-min 90]
    ... fit_primary.py tables
    ... fit_primary.py edge-tables
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from r2 import common as C  # noqa: E402
from r2 import evidence as E  # noqa: E402
from r2 import lcm  # noqa: E402

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpyro  # noqa: E402
from jax.flatten_util import ravel_pytree  # noqa: E402
from numpyro.infer.util import initialize_model, unconstrain_fn  # noqa: E402

FITS_DIR = C.OUT / "fits"
UNIVERSES = ("global",) + tuple(C.TISSUES)
VARIANT = C.PRESPECIFIED["evidence_variant_primary"]
MAP_MODELS = ("em", "dependence", "multifactor", "family_regularized", "family_hb")
N_RESTARTS = {"family_hb": 50}
N_RESTARTS_DEFAULT = 20
SEED = int(C.PRESPECIFIED["random_seed"])
NUTS_FULL = dict(num_chains=4, num_warmup=1000, num_samples=1000)
NUTS_REDUCED = dict(num_chains=4, num_warmup=500, num_samples=500)
NUTS_TARGET_ACCEPT = 0.95
NUTS_BUDGET_MIN = 90.0
NUTS_PILOT = dict(num_chains=1, num_warmup=100, num_samples=50)
NUTS_FITS = ("family_hb_nuts", "family_hb_randinit_nuts")
NUTS_INIT_OVERDISPERSION = 1.0
# FROZEN NUTS settings per fit and universe (verifier M5; fixed 2026-10-08).  ``all`` and
# ``nuts`` use these by default (``--settings frozen``); the committed copy
# src/r2/configs/fit_primary_nuts.json must be identical (tests/test_fit_primary.py).
NUTS_CONFIG = Path(__file__).resolve().parent / "configs" / "fit_primary_nuts.json"
_NUTS_PRIMARY = dict(num_chains=4, num_warmup=1000, num_samples=1000, target_accept=NUTS_TARGET_ACCEPT,
                     init="map_laplace", init_overdispersion=NUTS_INIT_OVERDISPERSION, chain_method="sequential")
_NUTS_RANDINIT = dict(num_chains=4, num_warmup=1000, num_samples=1000, target_accept=NUTS_TARGET_ACCEPT,
                      init="random", init_radius=2.0, chain_method="sequential")
NUTS_SETTINGS = {
    "family_hb_nuts": {u: dict(_NUTS_PRIMARY) for u in UNIVERSES},
    "family_hb_randinit_nuts": {u: dict(_NUTS_RANDINIT) for u in UNIVERSES},
}
NUTS_SETTINGS_HISTORY = [
    "2026-10-08 first pass: settings chosen by a wall-clock pilot (load-dependent); global family_hb_nuts "
    "ran 4 x 500/500 (projected 92.97 min > 90), all other runs 4 x 1000/1000.",
    "2026-10-08 first pass: map_laplace start dispersion changed from 2 to 1 after the immune run with "
    "dispersion 2 lost a chain to the second mode during warm-up (superseded log and chains CSV kept in "
    "outputs/r2/fits/).",
    "2026-10-08 fit_fixes: settings frozen as NUTS_SETTINGS (all 4 x 1000/1000, target_accept 0.95); re-runs of "
    "every family_hb_nuts (dominant-mode) fit with the current code from the polished MAP optimum were started; only "
    "the immune re-run finished on 2026-10-08 (kidney, lung and global: see the 2026-10-13 entries).",
    "2026-10-08 fit_fixes: the family_hb_randinit_nuts fits were NOT re-run (load average 70-180; projected global "
    "run > 6 h); they are the first-pass runs, made with settings equal to NUTS_SETTINGS; immune/global were "
    "relabelled by the committed src/r2/patch_randinit_layout.py.",
    "2026-10-13 fit_fixes (resumed): the 2026-10-08 re-runs of the kidney, lung and global family_hb_nuts fits had "
    "been interrupted (only immune had finished); the three were re-run with the frozen settings. The first-pass "
    "fits and tables are kept in outputs/r2/fits/_superseded_firstpass/. patch_randinit_layout.py was re-applied to "
    "immune/global; the random-init chain labels of all four universes agree with the polished MAP optima.",
    "2026-10-13 fit_fixes (resumed): the global family_hb_nuts re-run (4 x 1000/1000, target_accept 0.95) completed at "
    "the third attempt (started 18:54 BST, 87.9 min); the attempts of 2026-10-08 21:33 and 2026-10-13 15:06 were "
    "killed before writing a result (session interruption; laptop sleep). No setting was changed between attempts.",
]
# likelihood-identified quantities (class weights, annotator logits / rates, credibilities); the
# non-centred split of b_j = mu_f + tau z_j into family mean and deviation is prior-identified only
IDENTIFIED_SITES = ("w", "a", "b", "g", "theta", "cred_observed")
# conventional thresholds used to FLAG NUTS sites in the tables (a flag, not a pass/fail gate):
# split R-hat > 1.01 or bulk ESS < 400 (100 per chain with 4 chains)
RHAT_FLAG = 1.01
ESS_BULK_FLAG = 400.0
MAX_DRAWS_STORED = 1000
NEAR_BEST_TOL = 1e-3
GRAD_TOL = 1e-6              # plan section 6: max|gradient| < 1e-6 at the stored (polished) optimum
POLISH_GTOL = 1e-9           # Newton polishing continues to this gradient while the potential allows
NEWTON_MAX = 50
REL_DECREMENT_TOL = 10.0     # relative criterion: Newton decrement <= 10 * eps * max(1, |log posterior|)
LBFGS_OPTIONS = dict(maxiter=20000, gtol=1e-6, ftol=1e-15, maxcor=30)      # as lcm.fit_map
INIT_RADIUS = 2.0                                                          # as lcm.fit_map
BFDR_LEVELS = (C.PRESPECIFIED["bfdr_strong"], C.PRESPECIFIED["bfdr_moderate"])
TOPK = tuple(C.PRESPECIFIED["eval_k"])
PRIOR_DESCRIPTION = {
    "class_weights": "w ~ Dirichlet(1,...,1)",
    "hierarchical": ("q_j ~ N(mu^q_f(j), tau_q^2), mu^q_f ~ N(mu^q_0, omega_q^2), mu^q_0 ~ N(m_q, 2^2), "
                     "tau_q, omega_q ~ HalfNormal(1); m_b = logit(0.01), m_a = m_g = 0; non-centred"),
    "non_hierarchical": "q_j ~ N(m_q, 3^2) independently",
    "map_parametrisation": ("mode of the posterior density in the unconstrained parametrisation of "
                            "lcm.build_model (log tau/omega incl. Jacobian, stick-breaking w, "
                            "non-centred z); the Jacobian term tau*HalfNormal(tau) vanishes at "
                            "tau -> 0, so the MAP scales are boundary-avoiding; MAP tau/omega are "
                            "joint-mode values in these coordinates, not marginal posterior estimates"),
    "deviation_from_plan": ("plan section 6 named Gamma(2,2) MAP scale priors; MAP and NUTS both use "
                            "HalfNormal(1) (boundary avoidance comes from the log-scale Jacobian)"),
}

# ----------------------------------------------------------------------------------------
# Analysis status (prespecified / post hoc) of table rows and columns
# ----------------------------------------------------------------------------------------
# This module reads no held-out data, but the project's first held-out outcomes were computed on
# 2026-10-08 from 16:34 UTC (tuning_optimism, heldout_eval, replication), i.e. AFTER the first pass of this module
# (tables written 16:23 UTC) and BEFORE the fit_fixes round (verifier f1; 2026-10-08 evening,
# completed 2026-10-13).  Everything added or redefined in that round is labelled post hoc:
#   rows     ``analysis_status`` = 'post_hoc' for rows that exist only since that round, else
#            'prespecified' (plan section 6, or fixed label-free in the first pass);
#   columns  ``post_hoc_columns`` = ';'-joined columns of the table that were added or redefined
#            in that round (the same string in every row).
# ``FIRST_PASS_COLUMNS`` are the headers of the first-pass tables (kept in
# outputs/r2/fits/_superseded_firstpass/tables); a column not listed there is post hoc unless it
# is a pure rename (``RENAMED_COLUMNS``); ``REDEFINED_COLUMNS`` kept their name but changed their
# definition.  The dated decision log ``ANALYSIS_DECISIONS`` is written to
# ``eval_analysis_decisions.csv`` next to the tables.
STATUS_COLUMNS = ("analysis_status", "post_hoc_columns")
FIRST_PASS_COLUMNS = {
    "fit_convergence.csv": (
        "universe fit_id model inference n_patterns n_patterns_observed n_edges_observed runtime_s n_params "
        "n_restarts n_converged best_log_post loglik log_post_range n_within_1e3_of_best n_distinct_optima "
        "n_restarts_at_best gap_to_next_optimum second_optimum_n_restarts second_optimum_spearman_w "
        "second_optimum_max_abs_cred_diff expected_true_observed second_optimum_expected_true_observed "
        "max_abs_cred_diff_near_best max_abs_cred_diff_all_restarts grad_norm n_iter_best polish_grad_norm "
        "polish_delta_log_post polish_max_abs_cred_diff hessian_min_eig hessian_max_eig hessian_cond "
        "flattest_direction init n_chains_at_best_map_optimum chain_optima num_chains num_warmup num_samples "
        "settings projected_full_min r_hat_max worst_site ess_bulk_min ess_tail_min r_hat_max_theta r_hat_max_cred "
        "ess_bulk_min_cred ess_tail_min_cred n_divergent cred_ci_width_mean_w cred_ci_width_max "
        "r_hat_max_identified ess_bulk_min_identified ess_tail_min_identified").split(),
    "fit_source_parameters.csv": ("universe fit_id inference annotator family parameter value lo hi n_observed "
                                  "n_reported").split(),
    "fit_class_weights.csv": ("universe fit_id inference class_name w w_lo w_hi expected_edges_observed ee_lo ee_hi "
                              "n_edges_informative").split(),
    "fit_hyperparameters.csv": "universe fit_id inference parameter family data_informed value lo hi".split(),
    "fit_map_optima.csv": ("universe fit_id optimum n_restarts log_post gap_to_best expected_true_observed "
                           "spearman_w_vs_best max_abs_cred_diff_vs_best n_bfdr0_05 jaccard_bfdr0_05_vs_best "
                           "replay_log_post_diff laplace_log_mass laplace_min_eig laplace_log_det n_params "
                           "laplace_rel_mass").split(),
    "fit_nuts_chains.csv": ("universe fit_id chain mean_loglik sd_loglik w_T w_N w_B_LIT w_B_MOTIF "
                            "mean_cred_observed_w nearest_map_optimum dist_nearest_map_optimum").split(),
    "fit_estimator_agreement.csv": (
        "universe fit_a fit_b n_patterns_observed n_edges_observed spearman_w max_abs_diff mean_abs_diff_w "
        "q99_abs_diff_w n_bfdr0.05_a n_bfdr0.05_b jaccard_bfdr0.05 n_bfdr0.2_a n_bfdr0.2_b jaccard_bfdr0.2 "
        "n_top100_a n_top100_b jaccard_top100 n_top500_a n_top500_b jaccard_top500 n_top1000_a n_top1000_b "
        "jaccard_top1000 n_top5000_a n_top5000_b jaccard_top5000").split(),
}
FIRST_PASS_COLUMNS["fit_map_vs_nuts.csv"] = (
    [c for c in FIRST_PASS_COLUMNS["fit_estimator_agreement.csv"] if c not in ("fit_a", "fit_b")]
    + ("frac_edges_map_inside_cri cri_width_mean_w frac_edges_cri_width_gt_0.2 n_edges_cri_straddles_0.5 "
       "n_edges_nuts_mean_gt_0.5 n_edges_map_gt_0.5 cri_width_mean_w_mean_gt_0.5 cri_width_mean_w_bfdr0.2_set "
       "cri_lo_min_bfdr0.05_set w_T_map w_T_nuts_mean").split())
# new name -> first-pass name with the same content (pure renames, not post hoc): the first-pass
# ``grad_norm`` (L-BFGS end point) is now ``grad_norm_lbfgs`` and the first-pass ``polish_grad_norm``
# is now ``grad_norm`` (gradient at the stored, polished optimum)
RENAMED_COLUMNS = {
    "fit_convergence.csv": {"n_lbfgs_ftol_stop": "n_converged", "grad_norm_lbfgs": "grad_norm"},
}
_TOPK_COLUMNS = [f"{p}top{k}{s}" for k in TOPK for p, s in (("n_", "_a"), ("n_", "_b"), ("jaccard_", ""))]
# same name, definition changed in the fit_fixes round
REDEFINED_COLUMNS = {
    "fit_estimator_agreement.csv": _TOPK_COLUMNS,            # observed edges only, NA rule (verifier M2)
    "fit_map_vs_nuts.csv": _TOPK_COLUMNS,
}
POST_HOC_PARAMETERS = ("lr_pos_marginal", "alpha_over_gamma")   # fit_source_parameters.csv rows (verifier M4)
ANALYSIS_DECISIONS = [
    dict(id="fit_first_pass_design", date="2026-10-08", status="module_design",
         item="20 MAP restarts for the four comparison estimators and 50 for Family-HB; optima grouped within 1e-3 "
              "log-posterior units; Laplace-approximated mode masses; dominant-mode NUTS summary (chains started "
              "from basin-checked Laplace draws at the best MAP optimum, dispersion 1 after the superseded "
              "dispersion-2 immune run) with the random-init NUTS run kept as a multimodality diagnostic; "
              "HalfNormal(1) scale priors for MAP instead of the plan's Gamma(2,2); likelihood-identified site list",
         note="fixed label-free in the first pass (tables written 2026-10-08 16:23 UTC), before the project's first "
              "held-out evaluation output (16:34 UTC, tuning_optimism); not in PRESPECIFIED; rows carry "
              "analysis_status = prespecified"),
    dict(id="fit_polish_convergence", date="2026-10-08", status="post_hoc",
         item="stored MAP optimum = Newton-polished optimum of the best restart; convergence = polished "
              "max|gradient| < 1e-6 at a positive-definite Hessian (converged) or Newton decrement <= 10 eps "
              "|log posterior| (converged_relative); n_converged renamed n_lbfgs_ftol_stop, n_grad_converged added",
         note="verifier f1 M3; numerical procedure only, label-free (credibility changes <= 2.2e-5)"),
    dict(id="fit_observed_only_sets", date="2026-10-08", status="post_hoc",
         item="top-k and BFDR-set agreement computed over observed edges only; top-k NA when k exceeds the observed "
              "edges above the credibility of unobserved edges (w_T) in either fit",
         note="verifier f1 M2; brings the code in line with the documented definition; label-free"),
    dict(id="fit_marginal_lr", date="2026-10-08", status="post_hoc",
         item="lr_pos_marginal (report likelihood ratio of T against the mixture of all non-T classes), "
              "alpha_over_gamma and the annotator reliability rank on lr_pos_marginal",
         note="verifier f1 M4; label-free; rows analysis_status = post_hoc"),
    dict(id="fit_nuts_frozen", date="2026-10-08", status="post_hoc",
         item="NUTS settings frozen per universe (4 x 1000/1000, target acceptance 0.95; "
              "src/r2/configs/fit_primary_nuts.json); dominant-mode fits re-run from the polished MAP optimum "
              "(immune 2026-10-08; kidney, lung, global 2026-10-13; global was 4 x 500/500 in the first pass)",
         note="verifier f1 M5 and minor 1; restores the chain length of plan section 6; label-free"),
    dict(id="fit_expected_edges", date="2026-10-08", status="post_hoc",
         item="expected true edges incl. unobserved edges; class-weight expected edges over informative and over all "
              "edges; CrI-straddling counts among edges with posterior mean > 0.5; estimate_type label of MAP "
              "hierarchical scales (joint-mode values)",
         note="verifier f1 minors 2-4; label-free"),
    dict(id="fit_rank_interval", date="2026-10-14", status="post_hoc",
         item="95% posterior interval of the per-draw rank of each annotator's marginal LR+ (rank_lo, rank_hi; "
              "NUTS rows of fit_source_parameters.csv) next to the rank of the posterior mean",
         note="robfix, verifier fitfix minor 6; label-free"),
    dict(id="fit_value_provenance", date="2026-10-14", status="post_hoc",
         item="values_changed_post_hoc / value_change_note per row of every fit_* table: the row comes from a fit "
              "recomputed after the first held-out outcomes (MAP polished refits, frozen-settings NUTS re-runs, "
              "relabelled first-pass random-init chains), although its definition is prespecified",
         note="robfix, verifier fitfix minor 11; bookkeeping only"),
    dict(id="fit_nuts_site_flags", date="2026-10-13", status="post_hoc",
         item="NUTS sites with split R-hat > 1.01 or bulk ESS < 400 are flagged: sites_flagged / n_sites_flagged in "
              "fit_convergence.csv and r_hat_max_site / ess_bulk_min_site / ess_tail_min_site / diag_flag in "
              "fit_hyperparameters.csv",
         note="verifier f1 minor 1 (flag the weakly sampled hyper-parameter rows); thresholds are the conventional "
              "ones, fixed when the global 4 x 1000/1000 re-run had finished; reporting only, label-free"),
]


# ----------------------------------------------------------------------------------------
# Small helpers
# ----------------------------------------------------------------------------------------
def fit_dir(universe: str, fit_id: str, variant: str = VARIANT, root: Path = FITS_DIR) -> Path:
    return root / universe / variant / fit_id


def load_data(universe: str, variant: str = VARIANT) -> lcm.PatternData:
    return lcm.PatternData(*E.load_patterns(universe, variant))


def n_restarts_for(model: str) -> int:
    return N_RESTARTS.get(model, N_RESTARTS_DEFAULT)


def observed_patterns(data: lcm.PatternData) -> np.ndarray:
    """Boolean (P,): pattern observed by at least one annotator."""
    return (data.R >= 0).any(1)


def versions() -> dict:
    import scipy
    return dict(jax=jax.__version__, numpyro=numpyro.__version__, numpy=np.__version__,
                scipy=scipy.__version__)


def _jsonable(x):
    if isinstance(x, dict):
        return {k: _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    return x


def class_posteriors(data: lcm.PatternData, w: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """(P, S) posterior class probabilities P(z = s | pattern) at (w, theta)."""
    obs = (data.R >= 0).astype(float)
    pos = (data.R == 1).astype(float)
    th = np.clip(theta, 1e-12, 1 - 1e-12)
    lj = np.log(w)[None, :] + (obs * pos) @ np.log(th) + (obs * (1 - pos)) @ np.log1p(-th)
    return np.exp(lj - np.logaddexp.reduce(lj, axis=1, keepdims=True))


def annotator_parameters(theta: np.ndarray, class_names: list, family: list, w: np.ndarray | None = None) -> dict:
    """Per-annotator reporting rates from theta (..., K, S).

    alpha = theta[:, T] (sensitivity), beta = theta[:, N] (clean false-positive rate);
    family-burst models: gamma_j = theta[j, B_<family(j)>] (NaN when the annotator's family
    has no burst class); global-burst models: gamma = theta[:, B1], gamma2 = theta[:, B2].
    Also lr_pos = alpha / beta (T vs the clean negative N only), lr_neg = (1 - beta) / (1 - alpha)
    and alpha_over_gamma = alpha / gamma (T vs the annotator's burst class).  With class weights
    w (..., S): lr_pos_marginal = alpha / [sum_{s != T} w_s theta_js / (1 - w_T)], the likelihood
    ratio of a report under T against the mixture of ALL non-T classes (the meaningful
    reliability measure in a model with burst classes)."""
    theta = np.asarray(theta)
    out = {"alpha": theta[..., 0], "beta": theta[..., 1]}
    names = list(class_names)
    if "B1" in names:
        out["gamma"] = theta[..., names.index("B1")]
        if "B2" in names:
            out["gamma2"] = theta[..., names.index("B2")]
    elif any(n.startswith("B_") for n in names):
        g = np.full(theta.shape[:-1], np.nan)
        for j, f in enumerate(family):
            cn = f"B_{f}"
            if cn in names:
                g[..., j] = theta[..., j, names.index(cn)]
        out["gamma"] = g
    out["lr_pos"] = out["alpha"] / out["beta"]
    out["lr_neg"] = (1 - out["beta"]) / (1 - out["alpha"])
    if "gamma" in out:
        out["alpha_over_gamma"] = out["alpha"] / out["gamma"]
    if w is not None:
        w = np.asarray(w, dtype=float)
        non_t = (theta[..., 1:] * w[..., None, 1:]).sum(-1) / (1.0 - w[..., :1])
        out["lr_pos_marginal"] = out["alpha"] / non_t
    return out


def informative_masks(data: lcm.PatternData, class_names: list) -> dict:
    """Boolean (P,) per class: patterns whose likelihood can distinguish the class from the
    clean negative N.  T, N and global bursts -> observed by >= 1 annotator; B_f -> observed by
    >= 1 annotator of family f."""
    obs = data.R >= 0
    out = {}
    for c in class_names:
        if c.startswith("B_"):
            cols = np.array([f == c[2:] for f in data.family])
            out[c] = obs[:, cols].any(1)
        else:
            out[c] = obs.any(1)
    return out


def informative_edges(data: lcm.PatternData, class_names: list) -> dict:
    """Number of edges that can inform each class (``informative_masks``)."""
    return {c: float(data.n[m].sum()) for c, m in informative_masks(data, class_names).items()}


# ----------------------------------------------------------------------------------------
# Comparison utilities (edge-weighted)
# ----------------------------------------------------------------------------------------
def weighted_midranks(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Average ranks of each pattern when every pattern p is expanded into w_p tied edges.

    Equals the midrank that scipy.stats.rankdata assigns to the expanded edge vector."""
    x = np.asarray(x, dtype=float)
    w = np.asarray(w, dtype=float)
    uniq, inv = np.unique(x, return_inverse=True)
    inv = inv.reshape(-1)
    wsum = np.bincount(inv, weights=w, minlength=len(uniq))
    before = np.concatenate([[0.0], np.cumsum(wsum)[:-1]])
    mid = before + (wsum + 1.0) / 2.0
    return mid[inv]


def weighted_spearman(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
    """Spearman correlation of x and y over edges (pattern p repeated w_p times)."""
    w = np.asarray(w, dtype=float)
    rx, ry = weighted_midranks(x, w), weighted_midranks(y, w)
    mx, my = np.average(rx, weights=w), np.average(ry, weights=w)
    cov = np.average((rx - mx) * (ry - my), weights=w)
    vx = np.average((rx - mx) ** 2, weights=w)
    vy = np.average((ry - my) ** 2, weights=w)
    if vx <= 0 or vy <= 0:
        return float("nan")
    return float(cov / math.sqrt(vx * vy))


def bfdr_set(cred: np.ndarray, n: np.ndarray, q: float) -> np.ndarray:
    """Largest set of whole tie blocks, taken in decreasing credibility, whose Bayesian FDR
    (edge-weighted mean of 1 - cred) is <= q.  Returns a boolean pattern mask.

    The cumulative BFDR is non-decreasing along the ordering, so the set is a prefix of
    tie blocks.  A tie block = all patterns with exactly equal credibility."""
    cred = np.asarray(cred, dtype=float)
    n = np.asarray(n, dtype=float)
    uniq, inv = np.unique(-cred, return_inverse=True)       # blocks in decreasing cred
    inv = inv.reshape(-1)
    bn = np.bincount(inv, weights=n, minlength=len(uniq))
    bfp = np.bincount(inv, weights=n * (1 - cred), minlength=len(uniq))
    cum_n, cum_fp = np.cumsum(bn), np.cumsum(bfp)
    with np.errstate(invalid="ignore", divide="ignore"):
        fdr = np.where(cum_n > 0, cum_fp / cum_n, 0.0)
    ok = np.nonzero(fdr <= q + 1e-15)[0]
    if len(ok) == 0:
        return np.zeros(len(cred), dtype=bool)
    last = ok.max()
    return inv <= last


def topk_blocks(cred: np.ndarray, n: np.ndarray, k: float) -> np.ndarray:
    """Smallest prefix of whole tie blocks (decreasing credibility) holding >= k edges."""
    cred = np.asarray(cred, dtype=float)
    n = np.asarray(n, dtype=float)
    uniq, inv = np.unique(-cred, return_inverse=True)
    inv = inv.reshape(-1)
    bn = np.bincount(inv, weights=n, minlength=len(uniq))
    cum = np.cumsum(bn)
    hit = np.nonzero(cum >= k)[0]
    last = hit[0] if len(hit) else len(uniq) - 1
    return inv <= last


def weighted_jaccard(a: np.ndarray, b: np.ndarray, n: np.ndarray) -> float:
    inter = float(n[a & b].sum())
    union = float(n[a | b].sum())
    return inter / union if union > 0 else float("nan")


def n_observed_above_unobserved(c: np.ndarray, data: lcm.PatternData) -> float:
    """Observed edges whose credibility exceeds that of unobserved edges (= w_T; all observed
    edges if every pattern is observed)."""
    obs = observed_patterns(data)
    thr = float(np.max(c[~obs])) if (~obs).any() else -np.inf
    return float(data.n[obs & (c > thr)].sum())


def compare_creds(c1: np.ndarray, c2: np.ndarray, data: lcm.PatternData) -> dict:
    """Edge-weighted agreement of two per-pattern credibility vectors over OBSERVED patterns.

    Every statistic uses observed edges only: Spearman / differences over observed patterns,
    BFDR sets and top-k sets built from edge weights restricted to observed patterns.  The
    unobserved edges all carry credibility w_T and form one tie block that ranks above the
    observed negatives, so including them would make top-k agreement measure that block.  A
    top-k statistic is NA when k exceeds the observed edges above the unobserved credibility
    in either fit (``n_obs_above_unobs_a/b``)."""
    obs = observed_patterns(data)
    n = data.n
    n_obs = np.where(obs, n, 0.0)
    x, y, w = c1[obs], c2[obs], n[obs]
    d = np.abs(x - y)
    out = dict(n_patterns_observed=int(obs.sum()), n_edges_observed=float(w.sum()),
               spearman_w=weighted_spearman(x, y, w), max_abs_diff=float(d.max()),
               mean_abs_diff_w=float(np.average(d, weights=w)),
               q99_abs_diff_w=float(_weighted_quantile(d, w, 0.99)),
               n_obs_above_unobs_a=n_observed_above_unobserved(c1, data),
               n_obs_above_unobs_b=n_observed_above_unobserved(c2, data))
    for q in BFDR_LEVELS:
        a, b = bfdr_set(c1, n_obs, q) & obs, bfdr_set(c2, n_obs, q) & obs
        tag = f"bfdr{q:g}"
        out[f"n_{tag}_a"] = float(n[a].sum())
        out[f"n_{tag}_b"] = float(n[b].sum())
        out[f"jaccard_{tag}"] = weighted_jaccard(a, b, n)
    k_max = min(out["n_obs_above_unobs_a"], out["n_obs_above_unobs_b"])
    for k in TOPK:
        if k > k_max:
            out[f"n_top{k}_a"] = out[f"n_top{k}_b"] = out[f"jaccard_top{k}"] = np.nan
            continue
        a, b = topk_blocks(c1, n_obs, k) & obs, topk_blocks(c2, n_obs, k) & obs
        out[f"n_top{k}_a"] = float(n[a].sum())
        out[f"n_top{k}_b"] = float(n[b].sum())
        out[f"jaccard_top{k}"] = weighted_jaccard(a, b, n)
    return out


def _weighted_quantile(x, w, q):
    o = np.argsort(x)
    cw = np.cumsum(w[o])
    return x[o][np.searchsorted(cw, q * cw[-1])]


# ----------------------------------------------------------------------------------------
# Potential, Newton polishing and convergence criterion
# ----------------------------------------------------------------------------------------
def _potential(data: lcm.PatternData, model_name: str, prior: dict | None = None,
               seed: int = 0, init_radius: float | None = None):
    """Potential (negative log posterior in the unconstrained space of ``lcm.build_model``)
    with jitted value / gradient / Hessian, as used by ``lcm.fit_map``."""
    model = lcm.build_model(data, lcm.MODEL_SPECS[model_name], prior)
    kw = {} if init_radius is None else dict(init_strategy=numpyro.infer.init_to_uniform(radius=init_radius))
    init = initialize_model(jax.random.PRNGKey(seed), model, **kw)
    flat0, unravel = ravel_pytree(init.param_info.z)
    f = lambda x: init.potential_fn(unravel(x))  # noqa: E731
    return dict(model=model, init=init, unravel=unravel, flat0=flat0, names=list(init.param_info.z.keys()),
                pot=jax.jit(f), grad=jax.jit(jax.grad(f)), hess=jax.jit(jax.hessian(f)))


def _constrained(pt: dict, x: np.ndarray) -> dict:
    return {k: np.asarray(v) for k, v in pt["init"].postprocess_fn(pt["unravel"](jnp.asarray(x))).items()}


def _sym_hess(pt: dict, x: np.ndarray) -> np.ndarray:
    H = np.asarray(pt["hess"](jnp.asarray(x)), dtype=float)
    return 0.5 * (H + H.T)


def newton_polish(pt: dict, x: np.ndarray, gtol: float = POLISH_GTOL,
                  max_newton: int = NEWTON_MAX) -> tuple[np.ndarray, dict]:
    """Damped Newton descent of the potential U from x with the exact Hessian.

    Step = (H + shift I)^-1 g (shift only if H is not positive definite), halved up to 40 times.
    A step is accepted if it satisfies the Armijo condition, or if it lowers max|grad| without
    raising U by more than its rounding noise 1e-12 max(1, |U|) (near the optimum, differences
    of U fall below its floating-point resolution; same rule as ``robustness.newton_polish``
    with a tighter noise level).  Stops at max|grad| < gtol, when no step is accepted
    ('no_progress') or after ``max_newton`` steps.

    Returns (x, record): gradient max-abs before/after, steps, status, decrease of U, Hessian
    eigenvalue range at x, Newton decrement g'H^-1 g / 2 and step max|H^-1 g| at x, and the
    convergence flags ``converged`` (max|grad| < GRAD_TOL and positive-definite Hessian) and
    ``converged_relative`` (positive-definite Hessian and Newton decrement <=
    REL_DECREMENT_TOL * eps * max(1, |U|): no further improvement is resolvable)."""
    x = np.asarray(x, dtype=float)
    f = float(pt["pot"](jnp.asarray(x)))
    g = np.asarray(pt["grad"](jnp.asarray(x)), dtype=float)
    gn = float(np.max(np.abs(g)))
    f0, g0 = f, gn
    status, steps = "max_newton", 0
    for _ in range(max_newton):
        if gn < gtol:
            status = "gtol"
            break
        H = _sym_hess(pt, x)
        if not np.all(np.isfinite(H)):
            status = "hessian_not_finite"
            break
        ev = np.linalg.eigvalsh(H)
        shift = 0.0 if ev.min() > 1e-10 * max(1.0, abs(ev.max())) else (abs(ev.min()) + 1e-8 * max(1.0, abs(ev.max())))
        step = np.linalg.solve(H + shift * np.eye(len(x)), g)
        slope = float(g @ step)
        noise = 1e-12 * max(1.0, abs(f))
        t, accepted = 1.0, False
        for _ in range(40):
            xn = x - t * step
            fn = float(pt["pot"](jnp.asarray(xn)))
            if np.isfinite(fn):
                gnew = np.asarray(pt["grad"](jnp.asarray(xn)), dtype=float)
                gnn = float(np.max(np.abs(gnew)))
                if fn <= f - 1e-4 * t * slope or (fn <= f + noise and gnn < gn):
                    accepted = True
                    break
            t *= 0.5
        if not accepted:
            status = "no_progress"
            break
        x, f, g, gn = xn, fn, gnew, gnn
        steps += 1
    H = _sym_hess(pt, x)
    ev = np.linalg.eigvalsh(H) if np.all(np.isfinite(H)) else np.full(len(x), np.nan)
    if np.isfinite(ev[0]) and ev[0] > 0:
        st = np.linalg.solve(H, g)
        decrement, step_inf = float(0.5 * g @ st), float(np.max(np.abs(st)))
    else:
        decrement, step_inf = float("nan"), float("nan")
    rel_tol = REL_DECREMENT_TOL * np.finfo(float).eps * max(1.0, abs(f))
    pd_ok = bool(np.isfinite(ev[0]) and ev[0] > 0)
    rec = dict(n_params=int(len(x)), grad_norm_before=g0, grad_norm_after=gn, newton_steps=steps, status=status,
               delta_log_post=float(f0 - f), hess_min_eig=float(ev[0]), hess_max_eig=float(ev[-1]),
               hess_cond=float(ev[-1] / ev[0]) if pd_ok else float("inf"),
               newton_decrement=decrement, newton_step_inf=step_inf, rel_decrement_tol=float(rel_tol),
               converged=bool(pd_ok and gn < GRAD_TOL),
               converged_relative=bool(pd_ok and np.isfinite(decrement) and decrement <= rel_tol))
    return x, rec


def _flattest_direction(pt: dict, x: np.ndarray, z_template: dict, n_top: int = 4) -> list:
    """Sites loading on the eigenvector of the smallest Hessian eigenvalue at x."""
    labels = []
    for k in pt["names"]:
        size = int(np.size(np.asarray(z_template[k])))
        labels += [f"{k}[{i}]" if size > 1 else k for i in range(size)]
    ev, vec = np.linalg.eigh(_sym_hess(pt, x))
    flat = np.abs(vec[:, 0])
    return [f"{labels[i]}:{vec[i, 0]:+.2f}" for i in np.argsort(-flat)[:n_top]]


def polish_map(data: lcm.PatternData, model_name: str, best: dict, prior: dict | None = None,
               max_newton: int = NEWTON_MAX, gtol: float = POLISH_GTOL) -> dict:
    """Polish a MAP optimum given by its constrained parameters ``best`` (diagnostic helper).

    Returns the ``newton_polish`` record plus the largest absolute change of any pattern's
    credibility (``max_abs_cred_diff``), the reconstruction error of ``best``
    (``recon_max_abs_cred_diff``) and the sites loading on the flattest Hessian direction."""
    pt = _potential(data, model_name, prior)
    z0 = unconstrain_fn(pt["model"], (), {}, {k: jnp.asarray(best[k]) for k in pt["names"]})
    x0, _ = ravel_pytree(z0)
    x0 = np.asarray(x0, dtype=float)
    x, rec = newton_polish(pt, x0, gtol=gtol, max_newton=max_newton)
    c0 = lcm.credibility_at(_constrained(pt, x0), data)
    c1 = lcm.credibility_at(_constrained(pt, x), data)
    rec.update(max_abs_cred_diff=float(np.max(np.abs(c1 - c0))),
               recon_max_abs_cred_diff=float(np.max(np.abs(c0 - lcm.credibility_at(best, data)))),
               flattest_direction=_flattest_direction(pt, x, z0))
    return rec


# ----------------------------------------------------------------------------------------
# MAP fits
# ----------------------------------------------------------------------------------------
def map_restarts(data: lcm.PatternData, model_name: str, n_restarts: int, seed: int = SEED,
                 prior: dict | None = None, init_radius: float = INIT_RADIUS) -> dict:
    """Multi-start MAP: the restarts of ``lcm.fit_map(data, model_name, n_restarts, seed)``
    (same PRNG keys, start points and L-BFGS-B options; tested), each L-BFGS end point
    followed by ``newton_polish``.  The best restart is the one with the highest polished log
    posterior; its polished optimum is the fit.

    Returns dict(pt, restarts (list of per-restart records), xs (polished unconstrained optima),
    cred_restarts (restarts x P, polished), cred_restarts_lbfgs, ib, best (constrained sample),
    best_lbfgs (constrained sample at the best restart's L-BFGS end), polish (record of ib))."""
    from scipy.optimize import minimize
    spec = lcm.MODEL_SPECS[model_name]
    pt = _potential(data, model_name, prior, seed=seed, init_radius=init_radius)
    keys = jax.random.split(jax.random.PRNGKey(seed), n_restarts)
    recs, xs, xl, creds, creds_l, precs = [], [], [], [], [], []
    for r in range(n_restarts):
        x0 = np.asarray(jax.random.uniform(keys[r], pt["flat0"].shape, minval=-init_radius, maxval=init_radius))
        res = minimize(lambda x: float(pt["pot"](jnp.asarray(x))), x0,
                       jac=lambda x: np.asarray(pt["grad"](jnp.asarray(x))),
                       method="L-BFGS-B", options=dict(LBFGS_OPTIONS))
        s_l = _constrained(pt, res.x)
        ll_l, c_l, _ = lcm._collapsed_loglik_and_cred(data, spec, s_l)
        x, prec = newton_polish(pt, np.asarray(res.x))
        s = _constrained(pt, x)
        ll, c, _ = lcm._collapsed_loglik_and_cred(data, spec, s)
        recs.append(dict(restart=r, log_post=-float(pt["pot"](jnp.asarray(x))), loglik=ll,
                         grad_norm=prec["grad_norm_after"], hess_min_eig=prec["hess_min_eig"],
                         newton_steps=prec["newton_steps"], polish_status=prec["status"],
                         grad_converged=prec["converged"], converged_relative=prec["converged_relative"],
                         newton_decrement=prec["newton_decrement"],
                         max_abs_cred_change_polish=float(np.max(np.abs(c - c_l))),
                         lbfgs_log_post=-float(res.fun), lbfgs_loglik=ll_l, lbfgs_ftol_stop=bool(res.success),
                         lbfgs_grad_norm=prec["grad_norm_before"], n_iter=int(res.nit), message=str(res.message)))
        xs.append(x)
        xl.append(np.asarray(res.x))
        precs.append(prec)
        creds.append(c)
        creds_l.append(c_l)
    lp = np.array([r["log_post"] for r in recs])
    ib = int(np.argmax(lp))
    z_t = pt["unravel"](jnp.asarray(xs[ib]))
    pol = dict(precs[ib])
    pol.update(max_abs_cred_diff=float(np.max(np.abs(creds[ib] - creds_l[ib]))),
               flattest_direction=_flattest_direction(pt, xs[ib], z_t))
    return dict(pt=pt, restarts=recs, xs=xs, cred_restarts=np.stack(creds), cred_restarts_lbfgs=np.stack(creds_l),
                ib=ib, best=_constrained(pt, xs[ib]), best_lbfgs=_constrained(pt, xl[ib]), polish=pol)


def optima_table(log_post: np.ndarray, cred_restarts: np.ndarray, data: lcm.PatternData,
                 tol: float = NEAR_BEST_TOL) -> pd.DataFrame:
    """Group restarts into distinct optima and compare each optimum with the best one.

    Restarts are sorted by log posterior; a new optimum starts whenever a restart lies more
    than ``tol`` below the best restart of the current group.  Per optimum: number of restarts,
    log posterior, gap to the best optimum, expected number of true edges among observed edges
    and among all edges incl. unobserved ones (sum of credibility), and edge-weighted Spearman /
    max-abs difference / BFDR-0.05 Jaccard (observed edges) of its credibility against the best
    optimum (credibility of the group's best restart)."""
    lp = np.asarray(log_post, dtype=float)
    order = np.argsort(-lp, kind="mergesort")
    groups, cur = [], [order[0]]
    for i in order[1:]:
        if lp[cur[0]] - lp[i] <= tol:
            cur.append(i)
        else:
            groups.append(cur)
            cur = [i]
    groups.append(cur)
    obs = observed_patterns(data)
    n_obs = np.where(obs, data.n, 0.0)
    c_best = cred_restarts[groups[0][0]]
    s_best = bfdr_set(c_best, n_obs, BFDR_LEVELS[0]) & obs
    rows = []
    for k, g in enumerate(groups):
        c = cred_restarts[g[0]]
        s = bfdr_set(c, n_obs, BFDR_LEVELS[0]) & obs
        rows.append(dict(optimum=k, n_restarts=len(g), log_post=float(lp[g[0]]),
                         gap_to_best=float(lp[groups[0][0]] - lp[g[0]]),
                         expected_true_observed=float(np.sum(data.n[obs] * c[obs])),
                         expected_true_all=float(np.sum(data.n * c)),
                         spearman_w_vs_best=weighted_spearman(c[obs], c_best[obs], data.n[obs]),
                         max_abs_cred_diff_vs_best=float(np.max(np.abs(c[obs] - c_best[obs]))),
                         n_bfdr0_05=float(data.n[s].sum()),
                         jaccard_bfdr0_05_vs_best=weighted_jaccard(s, s_best, data.n),
                         restarts=" ".join(str(int(i)) for i in sorted(g))))
    return pd.DataFrame(rows)


CONVERGENCE_CRITERION = ("converged: polished max|grad| < 1e-6 at a positive-definite Hessian (plan section 6); "
                         "converged_relative: positive-definite Hessian and Newton decrement g'H^-1g/2 <= "
                         "10 * eps * max(1, |log posterior|) (no further improvement resolvable in float64)")


def run_map(universe: str, model: str, variant: str = VARIANT, n_restarts: int | None = None,
            seed: int = SEED, root: Path = FITS_DIR, data: lcm.PatternData | None = None,
            prior: dict | None = None, fit_id: str | None = None) -> dict:
    """Fit one estimator by multi-start MAP (``map_restarts``) and write the contract files.
    The stored optimum (``cred.npy``, ``params.npz``) is the polished optimum of the best
    restart."""
    data = load_data(universe, variant) if data is None else data
    n_restarts = n_restarts_for(model) if n_restarts is None else n_restarts
    fit_id = fit_id or model
    out = C.ensure_dir(fit_dir(universe, fit_id, variant, root))
    t0 = time.time()
    mr = map_restarts(data, model, n_restarts, seed=seed, prior=prior)
    runtime = time.time() - t0
    rs = mr["restarts"]
    ib = mr["ib"]
    best = mr["best"]
    spec = lcm.MODEL_SPECS[model]
    ll, cred, _ = lcm._collapsed_loglik_and_cred(data, spec, best)
    lp = np.array([r["log_post"] for r in rs])
    near = lp >= lp.max() - NEAR_BEST_TOL
    others = lp[~near]
    pol = mr["polish"]
    obs = observed_patterns(data)
    summ = dict(
        model=model, n_restarts=n_restarts, best_log_post=float(lp[ib]), loglik=float(ll),
        n_within_1e3_of_best=int(near.sum()),
        n_lbfgs_ftol_stop=int(sum(r["lbfgs_ftol_stop"] for r in rs)),
        n_grad_converged=int(sum(r["grad_converged"] for r in rs)),
        n_converged_relative=int(sum(r["converged_relative"] for r in rs)),
        converged=bool(rs[ib]["grad_converged"]), converged_relative=bool(rs[ib]["converged_relative"]),
        convergence_criterion=CONVERGENCE_CRITERION,
        log_post_range=float(lp.max() - lp.min()),
        max_abs_cred_diff_top=float(np.max(np.abs(mr["cred_restarts"][near] - cred[None, :]))),
        max_abs_cred_diff_all_restarts=float(np.max(np.abs(mr["cred_restarts"] - cred[None, :]))),
        grad_norm_best=float(rs[ib]["grad_norm"]), grad_norm_best_lbfgs=float(rs[ib]["lbfgs_grad_norm"]),
        max_grad_norm_near_best=float(max(r["grad_norm"] for r, m in zip(rs, near) if m)),
        n_iter_best=int(rs[ib]["n_iter"]),
        gap_to_next_optimum=float(lp.max() - others.max()) if len(others) else float("nan"),
        runtime_s=runtime,
    )
    opt = optima_table(lp, mr["cred_restarts"], data)
    summ.update(n_distinct_optima=int(len(opt)), n_restarts_at_best=int(opt.n_restarts.iloc[0]))
    if len(opt) > 1:
        summ.update(second_optimum_n_restarts=int(opt.n_restarts.iloc[1]),
                    second_optimum_spearman_w=float(opt.spearman_w_vs_best.iloc[1]),
                    second_optimum_max_abs_cred_diff=float(opt.max_abs_cred_diff_vs_best.iloc[1]),
                    second_optimum_expected_true_observed=float(opt.expected_true_observed.iloc[1]),
                    second_optimum_expected_true_all=float(opt.expected_true_all.iloc[1]))
    summ["expected_true_observed"] = float(opt.expected_true_observed.iloc[0])
    summ["expected_true_all"] = float(opt.expected_true_all.iloc[0])
    opt.to_csv(out / "optima.csv", index=False)
    np.save(out / "cred_restarts.npy", np.asarray(mr["cred_restarts"], dtype=np.float64))
    np.save(out / "cred.npy", np.asarray(cred, dtype=np.float64))
    params = {k: np.asarray(v) for k, v in best.items() if k != "cred"}
    np.savez(out / "params.npz", **params)
    pd.DataFrame(rs).to_csv(out / "restarts.csv", index=False)
    meta = dict(
        fit_id=fit_id, model=model, inference="MAP", universe=universe, variant=variant,
        class_names=list(mr["pt"]["model"].class_names), annotators=list(data.annotators), family=list(data.family),
        families=list(data.families), burst_families=[data.families[f] for f in data.burst_families()],
        spec=dict(spec), prior=dict(PRIOR_DESCRIPTION, overrides=prior or {}),
        n_restarts=n_restarts, seed=seed, best_log_post=float(lp[ib]), loglik=float(ll),
        stored_optimum="Newton-polished optimum of the best restart (fit_fixes 2026-10-08; earlier "
                       "fits stored the unpolished L-BFGS end point)",
        convergence=summ, polish=pol, runtime_s=runtime,
        n_patterns=int(data.P), n_patterns_observed=int(obs.sum()), n_edges=float(data.n.sum()),
        n_edges_observed=float(data.n[obs].sum()),
        optimizer=dict(method="L-BFGS-B", **LBFGS_OPTIONS, init=f"uniform(-{INIT_RADIUS:g}, {INIT_RADIUS:g}) in the "
                       "unconstrained space (keys and start points of lcm.fit_map)",
                       polish=f"newton_polish: exact Hessian, gtol {POLISH_GTOL:g}, max {NEWTON_MAX} steps"),
        versions=versions(),
    )
    C.write_json(_jsonable(meta), out / "meta.json")
    print(f"[map] {universe}/{fit_id}: log_post={lp[ib]:.4f} near-best={summ['n_within_1e3_of_best']}/"
          f"{n_restarts} ftol-stop={summ['n_lbfgs_ftol_stop']} grad-conv={summ['n_grad_converged']} "
          f"grad={summ['grad_norm_best']:.2e} (lbfgs {summ['grad_norm_best_lbfgs']:.1e}) "
          f"conv={summ['converged']}/{summ['converged_relative']} polish dcred={pol['max_abs_cred_diff']:.2e} "
          f"{runtime:.1f}s", flush=True)
    return meta


# ----------------------------------------------------------------------------------------
# NUTS fits
# ----------------------------------------------------------------------------------------
def frozen_nuts_settings(fit_id: str, universe: str) -> dict:
    """The frozen settings of ``NUTS_SETTINGS`` for one fit (KeyError for unknown fits)."""
    return dict(NUTS_SETTINGS[fit_id][universe])


def choose_nuts_settings(data: lcm.PatternData, settings: str, budget_min: float, seed: int,
                         prior: dict | None = None, model: str = "family_hb", fit_id: str | None = None,
                         universe: str | None = None) -> tuple[dict, dict]:
    """Return (sampler settings num_chains / num_warmup / num_samples, decision record).

    ``frozen`` (default): ``NUTS_SETTINGS[fit_id][universe]``.  ``full`` / ``reduced``: 4 x
    1000/1000 / 4 x 500/500.  ``auto`` (record/option only; load-dependent, not used by
    ``all``): a short pilot whose wall time is projected linearly to the full run; reduced
    settings if the projection exceeds ``budget_min`` (the first pass showed the projection can
    be 2-3x too optimistic)."""
    rec = dict(requested=settings, budget_min=budget_min)
    if settings == "frozen":
        fz = frozen_nuts_settings(fit_id, universe)
        kw = {k: fz[k] for k in ("num_chains", "num_warmup", "num_samples")}
        return kw, dict(requested="frozen", chosen="frozen", frozen=fz, config=str(NUTS_CONFIG.relative_to(C.SP)),
                        history=list(NUTS_SETTINGS_HISTORY))
    if settings == "full":
        return dict(NUTS_FULL), dict(rec, chosen="full")
    if settings == "reduced":
        return dict(NUTS_REDUCED), dict(rec, chosen="reduced")
    t0 = time.time()
    lcm.fit_nuts(data, model, seed=seed, target_accept=NUTS_TARGET_ACCEPT, prior=prior, **NUTS_PILOT)
    tp = time.time() - t0
    it_pilot = NUTS_PILOT["num_chains"] * (NUTS_PILOT["num_warmup"] + NUTS_PILOT["num_samples"])
    it_full = NUTS_FULL["num_chains"] * (NUTS_FULL["num_warmup"] + NUTS_FULL["num_samples"])
    proj = tp / it_pilot * it_full / 60.0
    chosen = "full" if proj <= budget_min else "reduced"
    rec.update(pilot=NUTS_PILOT, pilot_runtime_s=tp, projected_full_min=proj, chosen=chosen)
    return dict(NUTS_FULL if chosen == "full" else NUTS_REDUCED), rec


def _chain_diag(arr: np.ndarray) -> tuple[float, float, float]:
    """R-hat (max), bulk and tail ESS (min) over the trailing dims of (chains, draws, ...)."""
    import arviz as az
    arr = arr.reshape(arr.shape[0], arr.shape[1], -1)
    ds = az.convert_to_dataset({"x": arr})
    rh = az.rhat(ds)["x"].values
    eb = az.ess(ds, method="bulk")["x"].values
    et = az.ess(ds, method="tail")["x"].values
    return float(np.nanmax(rh)), float(np.nanmin(eb)), float(np.nanmin(et))


def replay_restart(data: lcm.PatternData, model_name: str, restart: int, n_restarts: int,
                   seed: int = SEED, prior: dict | None = None, init_radius: float = 2.0) -> np.ndarray:
    """Re-run restart ``restart`` of ``lcm.fit_map(data, model_name, n_restarts, seed)`` exactly
    (same PRNG keys, start point and optimiser settings) and return the unconstrained optimum.
    Used to recover the parameters of non-best optima, which ``fit_map`` does not return."""
    from scipy.optimize import minimize
    pt = _potential(data, model_name, prior, seed=seed, init_radius=init_radius)
    keys = jax.random.split(jax.random.PRNGKey(seed), n_restarts)
    x0 = np.asarray(jax.random.uniform(keys[restart], pt["flat0"].shape, minval=-init_radius, maxval=init_radius))
    res = minimize(lambda x: float(pt["pot"](jnp.asarray(x))), x0,
                   jac=lambda x: np.asarray(pt["grad"](jnp.asarray(x))),
                   method="L-BFGS-B", options=dict(maxiter=20000, gtol=1e-6, ftol=1e-15, maxcor=30))
    return np.asarray(res.x)


def laplace_optima(universe: str, model: str = "family_hb", variant: str = VARIANT, root: Path = FITS_DIR,
                   data: lcm.PatternData | None = None, prior: dict | None = None, fit_id: str | None = None,
                   seed: int = SEED) -> pd.DataFrame:
    """Laplace approximation of the posterior mass of every distinct MAP optimum.

    For each optimum in ``optima.csv`` the first restart reaching it is replayed
    (``replay_restart``) and polished by Newton steps; with U the potential and H its Hessian
    at the optimum x_k (unconstrained space, Jacobian included, so exp(-U) integrates to the
    marginal likelihood),  log Z_k = -U(x_k) + d/2 log(2 pi) - 1/2 log det H_k.
    The relative mass of optimum k is Z_k / sum_l Z_l over the optima found.  Adds the columns
    to ``optima.csv`` and writes the polished optimum of every optimum to
    ``optima_params.npz`` (unconstrained vectors, key ``x<k>``)."""
    data = load_data(universe, variant) if data is None else data
    fit_id = fit_id or model
    d = fit_dir(universe, fit_id, variant, root)
    opt = pd.read_csv(d / "optima.csv")
    meta = json.loads((d / "meta.json").read_text())
    rs = pd.read_csv(d / "restarts.csv")
    pt = _potential(data, model, prior)
    xs, cols = {}, dict(replay_log_post_diff=[], laplace_log_mass=[], laplace_min_eig=[], laplace_log_det=[],
                        n_params=[])
    for k, row in opt.iterrows():
        r = int(str(row["restarts"]).split()[0])
        x = replay_restart(data, model, r, meta["n_restarts"], seed=meta["seed"], prior=prior)
        lcol = "lbfgs_log_post" if "lbfgs_log_post" in rs.columns else "log_post"   # replay = L-BFGS end point
        cols["replay_log_post_diff"].append(float(-pt["pot"](x)) - float(rs.loc[rs.restart == r, lcol].iloc[0]))
        x, _ = newton_polish(pt, x)
        H = np.asarray(pt["hess"](x))
        H = 0.5 * (H + H.T)
        ev = np.linalg.eigvalsh(H)
        logdet = float(np.sum(np.log(ev))) if ev.min() > 0 else float("nan")
        dim = len(x)
        cols["laplace_log_mass"].append(float(-pt["pot"](x)) + 0.5 * dim * np.log(2 * np.pi) - 0.5 * logdet)
        cols["laplace_min_eig"].append(float(ev.min()))
        cols["laplace_log_det"].append(logdet)
        cols["n_params"].append(dim)
        xs[f"x{k}"] = x
    for c, v in cols.items():
        opt[c] = v
    lm = np.asarray(cols["laplace_log_mass"])
    opt["laplace_rel_mass"] = np.exp(lm - np.logaddexp.reduce(lm[np.isfinite(lm)])) if np.isfinite(lm).any() else np.nan
    opt.to_csv(d / "optima.csv", index=False)
    np.savez(d / "optima_params.npz", **xs)
    print(f"[laplace] {universe}/{fit_id}: rel. mass " +
          ", ".join(f"{m:.3g}" for m in opt["laplace_rel_mass"]) +
          f"; replay diff max {np.nanmax(np.abs(cols['replay_log_post_diff'])):.2e}", flush=True)
    return opt


def laplace_init_params(data: lcm.PatternData, model: str, map_dir: Path, num_chains: int, seed: int,
                        overdispersion: float = NUTS_INIT_OVERDISPERSION, prior: dict | None = None,
                        basin_check: bool = True, max_tries: int = 20) -> tuple[dict, dict]:
    """Dispersed starting points inside the basin of the best MAP optimum.

    Candidate x_c = x_MAP + overdispersion * L z_c, z_c ~ N(0, I), L L' = H^-1 (Laplace
    covariance at the optimum, unconstrained space).  With ``basin_check`` a candidate is
    accepted only if L-BFGS started from it (same settings as ``lcm.fit_map``) returns to the
    best optimum (log posterior within 1e-3); otherwise a new candidate is drawn.  Returns
    numpyro ``init_params`` (leading chain dimension) and a record of the construction."""
    from scipy.optimize import minimize
    best = dict(np.load(map_dir / "params.npz"))
    pt = _potential(data, model, prior)
    z0 = unconstrain_fn(pt["model"], (), {}, {k: jnp.asarray(best[k]) for k in pt["names"]})
    x0, unravel = ravel_pytree(z0)
    x0 = np.asarray(x0)
    lp_map = float(-pt["pot"](x0))
    H = np.asarray(pt["hess"](x0))
    H = 0.5 * (H + H.T)
    ev, V = np.linalg.eigh(H)
    ev = np.maximum(ev, 1e-8)
    L = V / np.sqrt(ev)[None, :]                                # L L' = H^-1
    rng = np.random.default_rng(seed)
    X, lp0, tries, lp_end = [], [], [], []
    for c in range(num_chains):
        for t in range(max_tries):
            x = x0 + overdispersion * L @ rng.standard_normal(len(x0))
            if not basin_check:
                end = float("nan")
                break
            res = minimize(lambda v: float(pt["pot"](jnp.asarray(v))), x,
                           jac=lambda v: np.asarray(pt["grad"](jnp.asarray(v))), method="L-BFGS-B",
                           options=dict(maxiter=20000, gtol=1e-6, ftol=1e-15, maxcor=30))
            end = float(-res.fun)
            if end >= lp_map - NEAR_BEST_TOL:
                break
        else:
            raise RuntimeError(f"no start in the basin of the best optimum after {max_tries} draws")
        X.append(x)
        lp0.append(float(-pt["pot"](x)))
        tries.append(t + 1)
        lp_end.append(end)
    trees = [unravel(jnp.asarray(x)) for x in X]
    init_params = {k: jnp.stack([t[k] for t in trees]) for k in trees[0]}
    rec = dict(init="map_laplace", map_fit=str(map_dir), overdispersion=overdispersion, basin_check=basin_check,
               map_log_post=lp_map, init_log_post=lp0, init_log_post_minus_map=[v - lp_map for v in lp0],
               basin_check_end_log_post=lp_end, draws_per_chain=tries)
    return init_params, rec


def _fit_nuts_init(data: lcm.PatternData, model_name: str, init_params: dict, num_warmup: int,
                   num_samples: int, num_chains: int, seed: int, target_accept: float,
                   prior: dict | None = None) -> lcm.NutsFit:
    """``lcm.fit_nuts`` with explicit per-chain initial values (identical sampler settings and
    diagnostics; ``lcm.fit_nuts`` has no init argument)."""
    import arviz as az
    from numpyro.infer import MCMC, NUTS
    model = lcm.build_model(data, lcm.MODEL_SPECS[model_name], prior)
    t0 = time.time()
    kernel = NUTS(model, target_accept_prob=target_accept)
    mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples, num_chains=num_chains,
                chain_method="sequential", progress_bar=False)
    mcmc.run(jax.random.PRNGKey(seed), extra_fields=("diverging",), init_params=init_params)
    s_chain = mcmc.get_samples(group_by_chain=True)
    div = int(np.asarray(mcmc.get_extra_fields(group_by_chain=True)["diverging"]).sum())
    diags = {}
    for k, v in s_chain.items():
        if k in ("cred", "theta"):
            continue
        arr = np.asarray(v)
        arr = arr.reshape(arr.shape[0], arr.shape[1], -1)
        ds = az.convert_to_dataset({k: arr})
        diags[k] = dict(r_hat_max=float(np.nanmax(az.rhat(ds)[k].values)),
                        ess_bulk_min=float(np.nanmin(az.ess(ds, method="bulk")[k].values)),
                        ess_tail_min=float(np.nanmin(az.ess(ds, method="tail")[k].values)))
    flat = {k: np.asarray(v).reshape((-1,) + np.asarray(v).shape[2:]) for k, v in s_chain.items()}
    return lcm.NutsFit(model_name, model.class_names, flat, diags, div, time.time() - t0)


def nearest_optimum(chain_cred: np.ndarray, map_dir: Path, data: lcm.PatternData) -> tuple[int, float]:
    """Index of the MAP optimum (``optima.csv`` of ``map_dir``) whose credibility is closest
    (edge-weighted mean absolute difference over observed patterns) to a chain's mean credibility."""
    f = map_dir / "optima.csv"
    if not f.exists() or not (map_dir / "cred_restarts.npy").exists():
        return -1, float("nan")
    opt = pd.read_csv(f)
    cr = np.load(map_dir / "cred_restarts.npy")
    obs = observed_patterns(data)
    d = [np.average(np.abs(chain_cred[obs] - cr[int(str(r).split()[0])][obs]), weights=data.n[obs])
         for r in opt["restarts"]]
    k = int(np.argmin(d))
    return k, float(d[k])


def run_nuts(universe: str, variant: str = VARIANT, settings: str = "frozen",
             budget_min: float = NUTS_BUDGET_MIN, seed: int = SEED, root: Path = FITS_DIR,
             data: lcm.PatternData | None = None, prior: dict | None = None,
             model: str = "family_hb", fit_id: str | None = None, nuts_kw: dict | None = None,
             init: str = "map_laplace", map_fit_id: str | None = None, map_root: Path | None = None) -> dict:
    """Full-posterior NUTS fit of ``model`` and the contract files of ``<fit_id>``.

    ``init='random'``: ``lcm.fit_nuts`` defaults (numpyro init_to_uniform(radius=2) in the
    unconstrained space; fit id ``<model>_randinit_nuts``).  ``init='map_laplace'``: chains
    start from basin-checked draws of the Laplace approximation around the best MAP optimum of
    ``map_fit_id`` (default ``model``) -- the posterior restricted to the dominant mode
    (fit id ``<model>_nuts``)."""
    data = load_data(universe, variant) if data is None else data
    if fit_id is None:
        fit_id = f"{model}_nuts" if init == "map_laplace" else f"{model}_randinit_nuts"
    map_dir = fit_dir(universe, map_fit_id or model, variant, map_root or root)
    target_accept, overdispersion = NUTS_TARGET_ACCEPT, NUTS_INIT_OVERDISPERSION
    if nuts_kw is not None:
        kw, decision = dict(nuts_kw), dict(requested="explicit", chosen="explicit")
    else:
        kw, decision = choose_nuts_settings(data, settings, budget_min, seed, prior, model, fit_id, universe)
        if settings == "frozen":
            fz = decision["frozen"]
            if fz["init"] != init:
                raise ValueError(f"frozen settings of {fit_id}/{universe} have init {fz['init']!r}, not {init!r}")
            target_accept = float(fz["target_accept"])
            overdispersion = float(fz.get("init_overdispersion", overdispersion))
    init_rec = dict(init="random", description="numpyro init_to_uniform(radius=2), lcm.fit_nuts default")
    if init == "map_laplace":
        init_params, init_rec = laplace_init_params(data, model, map_dir, kw["num_chains"], seed,
                                                    overdispersion=overdispersion, prior=prior)
        init_rec["overdispersion_history"] = ("2 in the superseded first immune run (a chain left the dominant "
                                              "mode during warm-up), 1 since")
    print(f"[nuts] {universe}/{fit_id}: settings {kw} target_accept {target_accept} ({decision['chosen']}); "
          f"init {init}", flush=True)
    t0 = time.time()
    if init == "map_laplace":
        nf = _fit_nuts_init(data, model, init_params, seed=seed, target_accept=target_accept,
                            prior=prior, **kw)
    else:
        nf = lcm.fit_nuts(data, model, seed=seed, target_accept=target_accept, prior=prior, **kw)
    runtime = time.time() - t0
    out = C.ensure_dir(fit_dir(universe, fit_id, variant, root))
    nc, ns = kw["num_chains"], kw["num_samples"]
    S = nf.samples
    cred = np.asarray(S["cred"])                               # (D, P)
    D = cred.shape[0]
    cs = dict(mean=cred.mean(0), lo=np.quantile(cred, 0.025, axis=0), hi=np.quantile(cred, 0.975, axis=0))
    sd = cred.std(0)
    obs = observed_patterns(data)
    np.save(out / "cred.npy", cs["mean"].astype(np.float64))
    np.save(out / "cred_mean.npy", cs["mean"].astype(np.float64))
    np.save(out / "cred_lo.npy", cs["lo"].astype(np.float64))
    np.save(out / "cred_hi.npy", cs["hi"].astype(np.float64))
    np.save(out / "cred_sd.npy", sd.astype(np.float64))
    # diagnostics: sample sites from lcm, plus derived theta and cred over observed patterns
    diag = [dict(site=k, **v) for k, v in sorted(nf.diagnostics.items())]
    th = np.asarray(S["theta"]).reshape(nc, ns, -1)
    diag.append(dict(site="theta", **dict(zip(("r_hat_max", "ess_bulk_min", "ess_tail_min"), _chain_diag(th)))))
    cobs = cred[:, obs]
    var_ok = cobs.std(0) > 0
    cdiag = _chain_diag(cobs[:, var_ok].reshape(nc, ns, -1)) if var_ok.any() else (np.nan, np.nan, np.nan)
    diag.append(dict(site="cred_observed", **dict(zip(("r_hat_max", "ess_bulk_min", "ess_tail_min"), cdiag))))
    diag_df = pd.DataFrame(diag)
    diag_df.to_csv(out / "diagnostics.csv", index=False)
    sample_sites = [k for k in nf.diagnostics]
    # thinned draws
    stride = max(1, int(math.ceil(D / MAX_DRAWS_STORED)))
    idx = np.arange(0, D, stride)
    chain = np.repeat(np.arange(nc), ns)
    keep = {k: np.asarray(v)[idx] for k, v in S.items() if k != "cred"}
    np.savez(out / "draws.npz", chain=chain[idx], draw_index=idx, **keep)
    # posterior-mean params
    means = {k: np.asarray(v).mean(0) for k, v in S.items() if k != "cred"}
    np.savez(out / "params.npz", **means)
    # per-chain summaries (a chain stuck in another mode shows up here)
    W = np.asarray(S["w"])
    TH = np.asarray(S["theta"])
    ll = np.array([lcm.heldout_loglik({"w": W[d], "theta": TH[d]}, data) for d in range(D)])
    rows = []
    for c in range(nc):
        sl = slice(c * ns, (c + 1) * ns)
        r = dict(chain=c, mean_loglik=float(ll[sl].mean()), sd_loglik=float(ll[sl].std()))
        for s, nm in enumerate(nf.class_names):
            r[f"w_{nm}"] = float(W[sl, s].mean())
        r["mean_cred_observed_w"] = float(np.average(cred[sl][:, obs].mean(0), weights=data.n[obs]))
        r["nearest_map_optimum"], r["dist_nearest_map_optimum"] = nearest_optimum(cred[sl].mean(0), map_dir, data)
        rows.append(r)
    chains_df = pd.DataFrame(rows)
    chains_df.to_csv(out / "chains.csv", index=False)
    smp = diag_df[diag_df.site.isin(sample_sites)]
    summ = dict(
        r_hat_max=float(smp.r_hat_max.max()), ess_bulk_min=float(smp.ess_bulk_min.min()),
        ess_tail_min=float(smp.ess_tail_min.min()),
        r_hat_max_theta=float(diag_df.loc[diag_df.site == "theta", "r_hat_max"].iloc[0]),
        r_hat_max_cred=float(cdiag[0]), ess_bulk_min_cred=float(cdiag[1]), ess_tail_min_cred=float(cdiag[2]),
        worst_site=str(smp.sort_values("r_hat_max").site.iloc[-1]),
        n_divergent=int(nf.n_divergent), n_draws=int(D),
        loglik_mean=float(ll.mean()),
        cred_ci_width_mean_w=float(np.average((cs["hi"] - cs["lo"])[obs], weights=data.n[obs])),
        cred_ci_width_max=float((cs["hi"] - cs["lo"])[obs].max()),
        n_chains_at_best_map_optimum=int((chains_df["nearest_map_optimum"] == 0).sum()),
        chain_optima=chains_df["nearest_map_optimum"].tolist(),
    )
    meta = dict(
        fit_id=fit_id, model=model, inference="NUTS", universe=universe, variant=variant,
        class_names=list(nf.class_names), annotators=list(data.annotators), family=list(data.family),
        families=list(data.families), burst_families=[data.families[f] for f in data.burst_families()],
        spec=lcm.MODEL_SPECS[model], prior=dict(PRIOR_DESCRIPTION, overrides=prior or {}),
        seed=seed, num_chains=nc, num_warmup=kw["num_warmup"], num_samples=ns,
        target_accept=target_accept, chain_method="sequential", settings_decision=decision,
        init=init_rec,
        n_divergent=int(nf.n_divergent), convergence=summ, runtime_s=runtime,
        n_patterns=int(data.P), n_patterns_observed=int(obs.sum()), n_edges=float(data.n.sum()),
        n_edges_observed=float(data.n[obs].sum()), draws_stored=int(len(idx)), draw_stride=stride,
        cred_files=dict(cred="posterior mean", cred_lo="2.5% quantile", cred_hi="97.5% quantile",
                        cred_sd="posterior sd"),
        versions=versions(),
    )
    C.write_json(_jsonable(meta), out / "meta.json")
    print(f"[nuts] {universe}/{fit_id}: R-hat max {summ['r_hat_max']:.4f} (cred {summ['r_hat_max_cred']:.4f}) "
          f"ESS bulk min {summ['ess_bulk_min']:.0f} div {summ['n_divergent']} {runtime / 60:.1f} min", flush=True)
    return meta


# ----------------------------------------------------------------------------------------
# Manuscript tables
# ----------------------------------------------------------------------------------------
def _read_fit(universe: str, fit_id: str, variant: str = VARIANT, root: Path = FITS_DIR):
    d = fit_dir(universe, fit_id, variant, root)
    if not (d / "meta.json").exists():
        return None
    meta = json.loads((d / "meta.json").read_text())
    params = dict(np.load(d / "params.npz"))
    out = dict(meta=meta, params=params, cred=np.load(d / "cred.npy"), dir=d)
    if meta["inference"] == "NUTS":
        out["draws"] = dict(np.load(d / "draws.npz"))
        for s in ("lo", "hi", "sd"):
            out[f"cred_{s}"] = np.load(d / f"cred_{s}.npy")
        out["diagnostics"] = pd.read_csv(d / "diagnostics.csv")
    return out


def site_flags(diag: pd.DataFrame) -> np.ndarray:
    """Boolean per row of a NUTS ``diagnostics.csv``: split R-hat > ``RHAT_FLAG`` or bulk ESS <
    ``ESS_BULK_FLAG`` (site-level worst element)."""
    return ((diag["r_hat_max"] > RHAT_FLAG) | (diag["ess_bulk_min"] < ESS_BULK_FLAG)).to_numpy()


def convergence_table(universes, variant=VARIANT, root=FITS_DIR) -> pd.DataFrame:
    rows = []
    for u in universes:
        for fid in list(MAP_MODELS) + list(NUTS_FITS):
            f = _read_fit(u, fid, variant, root)
            if f is None:
                continue
            m = f["meta"]
            c = m["convergence"]
            r = dict(universe=u, fit_id=fid, model=m["model"], inference=m["inference"],
                     n_patterns=m["n_patterns"], n_patterns_observed=m["n_patterns_observed"],
                     n_edges_observed=m["n_edges_observed"], runtime_s=m["runtime_s"])
            if m["inference"] == "MAP":
                p = m["polish"]
                r.update(n_params=p["n_params"], n_restarts=c["n_restarts"],
                         n_lbfgs_ftol_stop=c.get("n_lbfgs_ftol_stop"), n_grad_converged=c.get("n_grad_converged"),
                         n_converged_relative=c.get("n_converged_relative"),
                         converged=c.get("converged"), converged_relative=c.get("converged_relative"),
                         best_log_post=c["best_log_post"], loglik=c["loglik"],
                         log_post_range=c["log_post_range"], n_within_1e3_of_best=c["n_within_1e3_of_best"],
                         n_distinct_optima=c["n_distinct_optima"], n_restarts_at_best=c.get("n_restarts_at_best"),
                         gap_to_next_optimum=c["gap_to_next_optimum"],
                         second_optimum_n_restarts=c.get("second_optimum_n_restarts"),
                         second_optimum_spearman_w=c.get("second_optimum_spearman_w"),
                         second_optimum_max_abs_cred_diff=c.get("second_optimum_max_abs_cred_diff"),
                         expected_true_observed=c.get("expected_true_observed"),
                         expected_true_all=c.get("expected_true_all"),
                         second_optimum_expected_true_observed=c.get("second_optimum_expected_true_observed"),
                         second_optimum_expected_true_all=c.get("second_optimum_expected_true_all"),
                         max_abs_cred_diff_near_best=c["max_abs_cred_diff_top"],
                         max_abs_cred_diff_all_restarts=c["max_abs_cred_diff_all_restarts"],
                         grad_norm=c["grad_norm_best"], grad_norm_lbfgs=c.get("grad_norm_best_lbfgs"),
                         max_grad_norm_near_best=c.get("max_grad_norm_near_best"), n_iter_best=c["n_iter_best"],
                         polish_status=p.get("status"), polish_newton_steps=p.get("newton_steps"),
                         polish_delta_log_post=p["delta_log_post"],
                         polish_max_abs_cred_diff=p["max_abs_cred_diff"],
                         newton_decrement=p.get("newton_decrement"), rel_decrement_tol=p.get("rel_decrement_tol"),
                         hessian_min_eig=p["hess_min_eig"],
                         hessian_max_eig=p["hess_max_eig"], hessian_cond=p["hess_cond"],
                         flattest_direction=";".join(p["flattest_direction"]))
            else:
                r.update(n_params=None, init=m.get("init", {}).get("init", "random"),
                         n_chains_at_best_map_optimum=c.get("n_chains_at_best_map_optimum"),
                         chain_optima=" ".join(str(x) for x in c.get("chain_optima", [])),
                         num_chains=m["num_chains"], num_warmup=m["num_warmup"],
                         num_samples=m["num_samples"], target_accept=m.get("target_accept"),
                         settings=m["settings_decision"].get("chosen"),
                         init_overdispersion=m.get("init", {}).get("overdispersion"),
                         r_hat_max=c["r_hat_max"], worst_site=c["worst_site"], ess_bulk_min=c["ess_bulk_min"],
                         ess_tail_min=c["ess_tail_min"], r_hat_max_theta=c["r_hat_max_theta"],
                         r_hat_max_cred=c["r_hat_max_cred"], ess_bulk_min_cred=c["ess_bulk_min_cred"],
                         ess_tail_min_cred=c["ess_tail_min_cred"], n_divergent=c["n_divergent"],
                         loglik=c["loglik_mean"], cred_ci_width_mean_w=c["cred_ci_width_mean_w"],
                         cred_ci_width_max=c["cred_ci_width_max"])
                dg = f["diagnostics"]
                ident = dg[dg.site.isin(IDENTIFIED_SITES)]
                r.update(r_hat_max_identified=float(ident.r_hat_max.max()),
                         ess_bulk_min_identified=float(ident.ess_bulk_min.min()),
                         ess_tail_min_identified=float(ident.ess_tail_min.min()))
                flagged = sorted(dg.loc[site_flags(dg), "site"])
                r.update(n_sites_flagged=len(flagged), sites_flagged=";".join(flagged),
                         n_identified_sites_flagged=len([s for s in flagged if s in IDENTIFIED_SITES]))
            rows.append(r)
    return pd.DataFrame(rows)


def _coverage(data: lcm.PatternData) -> tuple[np.ndarray, np.ndarray]:
    obs = data.R >= 0
    pos = data.R == 1
    return (obs * data.n[:, None]).sum(0), (pos * data.n[:, None]).sum(0)


def source_parameter_table(universes, variant=VARIANT, root=FITS_DIR) -> pd.DataFrame:
    rows = []
    for u in universes:
        data = load_data(u, variant)
        n_obs, n_rep = _coverage(data)
        for fid in list(MAP_MODELS) + ["family_hb_nuts"]:
            f = _read_fit(u, fid, variant, root)
            if f is None:
                continue
            m = f["meta"]
            if m["inference"] == "MAP":
                P = annotator_parameters(f["params"]["theta"], m["class_names"], m["family"], w=f["params"]["w"])
                for p, v in P.items():
                    for j, a in enumerate(m["annotators"]):
                        if np.isnan(v[j]):
                            continue
                        rows.append(dict(universe=u, fit_id=fid, inference="MAP", annotator=a,
                                         family=m["family"][j], parameter=p, value=float(v[j]),
                                         lo=np.nan, hi=np.nan, n_observed=float(n_obs[j]),
                                         n_reported=float(n_rep[j])))
            else:
                P = annotator_parameters(f["draws"]["theta"], m["class_names"], m["family"], w=f["draws"]["w"])
                rk = draw_ranks(P["lr_pos_marginal"])
                for p, v in P.items():
                    for j, a in enumerate(m["annotators"]):
                        x = v[:, j]
                        if np.all(np.isnan(x)):
                            continue
                        r = dict(universe=u, fit_id=fid, inference="NUTS", annotator=a,
                                 family=m["family"][j], parameter=p, value=float(x.mean()),
                                 lo=float(np.quantile(x, 0.025)), hi=float(np.quantile(x, 0.975)),
                                 n_observed=float(n_obs[j]), n_reported=float(n_rep[j]))
                        if p == "lr_pos_marginal":
                            r.update(rank_lo=float(np.quantile(rk[:, j], 0.025, method="inverted_cdf")),
                                     rank_hi=float(np.quantile(rk[:, j], 0.975, method="inverted_cdf")))
                        rows.append(r)
    df = pd.DataFrame(rows)
    # annotator reliability ranking (1 = most diagnostic) on the marginal LR+ (T vs all non-T classes):
    # rank of the posterior mean / MAP value; NUTS rows also carry the 95% posterior interval of the
    # per-draw rank (rank_lo, rank_hi; robfix, verifier fitfix minor 6)
    df["rank"] = np.nan
    mk = df["parameter"] == "lr_pos_marginal"
    df.loc[mk, "rank"] = df[mk].groupby(["universe", "fit_id"])["value"].rank(ascending=False, method="min")
    for c in ("rank_lo", "rank_hi"):
        if c not in df.columns:
            df[c] = np.nan
    return df


def draw_ranks(v: np.ndarray) -> np.ndarray:
    """(D, K) per-draw rank of each annotator (1 = largest value, ties 'min'), as the table's ``rank``."""
    v = np.asarray(v, dtype=np.float64)
    return np.stack([pd.Series(row).rank(ascending=False, method="min").to_numpy() for row in v])


def class_weight_table(universes, variant=VARIANT, root=FITS_DIR) -> pd.DataFrame:
    rows = []
    for u in universes:
        data = load_data(u, variant)
        obs = observed_patterns(data)
        nobs = data.n * obs
        for fid in list(MAP_MODELS) + ["family_hb_nuts"]:
            f = _read_fit(u, fid, variant, root)
            if f is None:
                continue
            m = f["meta"]
            names = m["class_names"]
            info = informative_edges(data, names)
            imask = informative_masks(data, names)
            # (P, S) weights: edges among the patterns that can inform each class
            n_inf = np.stack([data.n * imask[c] for c in names], axis=1)

            def expected(post):
                return (nobs @ post, (n_inf * post).sum(0), data.n @ post)

            if m["inference"] == "MAP":
                w = f["params"]["w"]
                eo, ei, ea = expected(class_posteriors(data, w, f["params"]["theta"]))
                for s, c in enumerate(names):
                    rows.append(dict(universe=u, fit_id=fid, inference="MAP", class_name=c, w=float(w[s]),
                                     w_lo=np.nan, w_hi=np.nan, expected_edges_observed=float(eo[s]),
                                     ee_lo=np.nan, ee_hi=np.nan, n_edges_informative=info[c],
                                     expected_edges_informative=float(ei[s]), eei_lo=np.nan, eei_hi=np.nan,
                                     expected_edges_all=float(ea[s]), eea_lo=np.nan, eea_hi=np.nan))
            else:
                W, TH = f["draws"]["w"], f["draws"]["theta"]
                EX = [expected(class_posteriors(data, W[d], TH[d])) for d in range(len(W))]
                EO, EI, EA = (np.stack([e[i] for e in EX]) for i in range(3))
                q = lambda a, s_, p: float(np.quantile(a[:, s_], p))  # noqa: E731
                for s, c in enumerate(names):
                    rows.append(dict(universe=u, fit_id=fid, inference="NUTS", class_name=c,
                                     w=float(W[:, s].mean()), w_lo=q(W, s, 0.025), w_hi=q(W, s, 0.975),
                                     expected_edges_observed=float(EO[:, s].mean()),
                                     ee_lo=q(EO, s, 0.025), ee_hi=q(EO, s, 0.975), n_edges_informative=info[c],
                                     expected_edges_informative=float(EI[:, s].mean()),
                                     eei_lo=q(EI, s, 0.025), eei_hi=q(EI, s, 0.975),
                                     expected_edges_all=float(EA[:, s].mean()),
                                     eea_lo=q(EA, s, 0.025), eea_hi=q(EA, s, 0.975)))
    return pd.DataFrame(rows)


HYPER_SITES = ("mu0", "omega", "tau", "muf")


def hyperparameter_table(universes, variant=VARIANT, root=FITS_DIR) -> pd.DataFrame:
    rows = []
    for u in universes:
        data = load_data(u, variant)
        burst = {data.families[f] for f in data.burst_families()}
        for fid in ("family_regularized", "family_hb", "family_hb_nuts"):
            f = _read_fit(u, fid, variant, root)
            if f is None:
                continue
            m = f["meta"]
            src = f["params"] if m["inference"] == "MAP" else f["draws"]
            for q in ("b", "a", "g"):
                for h in HYPER_SITES:
                    k = f"{h}_{q}"
                    if k not in src:
                        continue
                    v = np.asarray(src[k])
                    labels = [None] if h != "muf" else m["families"]
                    for i, lab in enumerate(labels):
                        x = v if lab is None else v[..., i]
                        identified = not (q == "g" and lab is not None and lab not in burst)
                        r = dict(universe=u, fit_id=fid, inference=m["inference"], parameter=k,
                                 family=lab or "", data_informed=identified)
                        if m["inference"] == "MAP":
                            r.update(value=float(x), lo=np.nan, hi=np.nan,
                                     estimate_type="joint_mode_unconstrained_noncentred",
                                     r_hat_max_site=np.nan, ess_bulk_min_site=np.nan, ess_tail_min_site=np.nan,
                                     diag_flag="")
                        else:
                            r["estimate_type"] = "posterior_mean_95cri"
                            r.update(value=float(np.mean(x)), lo=float(np.quantile(x, 0.025)),
                                     hi=float(np.quantile(x, 0.975)))
                            # site-level sampler diagnostics over all draws (worst element of the site)
                            dg = f["diagnostics"]
                            d = dg[dg.site == k]
                            if len(d):
                                r.update(r_hat_max_site=float(d.r_hat_max.iloc[0]),
                                         ess_bulk_min_site=float(d.ess_bulk_min.iloc[0]),
                                         ess_tail_min_site=float(d.ess_tail_min.iloc[0]),
                                         diag_flag="flagged" if bool(site_flags(d)[0]) else "ok")
                            else:
                                r.update(r_hat_max_site=np.nan, ess_bulk_min_site=np.nan, ess_tail_min_site=np.nan,
                                         diag_flag="")
                        rows.append(r)
    return pd.DataFrame(rows)


def agreement_tables(universes, variant=VARIANT, root=FITS_DIR) -> tuple[pd.DataFrame, pd.DataFrame]:
    """MAP-vs-NUTS table (Family-HB) and all-pairs estimator agreement."""
    mvn, pairs = [], []
    for u in universes:
        data = load_data(u, variant)
        fits = {fid: _read_fit(u, fid, variant, root) for fid in list(MAP_MODELS) + ["family_hb_nuts"]}
        fits = {k: v for k, v in fits.items() if v is not None}
        ids = list(fits)
        for i, a in enumerate(ids):
            for b in ids[i + 1:]:
                r = dict(universe=u, fit_a=a, fit_b=b, **compare_creds(fits[a]["cred"], fits[b]["cred"], data))
                pairs.append(r)
        if "family_hb" in fits and "family_hb_nuts" in fits:
            nf = fits["family_hb_nuts"]
            obs = observed_patterns(data)
            mp = fits["family_hb"]["cred"]
            inside = (mp >= nf["cred_lo"] - 1e-12) & (mp <= nf["cred_hi"] + 1e-12)
            r = dict(universe=u, **compare_creds(mp, nf["cred"], data))
            r["frac_edges_map_inside_cri"] = float(data.n[obs & inside].sum() / data.n[obs].sum())
            width = nf["cred_hi"] - nf["cred_lo"]
            straddle = (nf["cred_lo"] < 0.5) & (nf["cred_hi"] > 0.5)
            r["cri_width_mean_w"] = float(np.average(width[obs], weights=data.n[obs]))
            r["frac_edges_cri_width_gt_0.2"] = float(data.n[obs & (width > 0.2)].sum() / data.n[obs].sum())
            r["n_edges_cri_straddles_0.5"] = float(data.n[obs & straddle].sum())
            r["n_edges_cri_straddles_0.5_mean_gt_0.5"] = float(data.n[obs & straddle & (nf["cred"] > 0.5)].sum())
            r["n_edges_nuts_mean_gt_0.5"] = float(data.n[obs & (nf["cred"] > 0.5)].sum())
            r["n_edges_map_gt_0.5"] = float(data.n[obs & (mp > 0.5)].sum())
            hi_set = obs & (nf["cred"] > 0.5)
            mod_set = bfdr_set(nf["cred"], np.where(obs, data.n, 0.0), BFDR_LEVELS[1]) & obs
            r["cri_width_mean_w_mean_gt_0.5"] = float(np.average(width[hi_set], weights=data.n[hi_set])) if hi_set.any() else np.nan
            r["cri_width_mean_w_bfdr0.2_set"] = float(np.average(width[mod_set], weights=data.n[mod_set])) if mod_set.any() else np.nan
            strong = bfdr_set(nf["cred"], np.where(obs, data.n, 0.0), BFDR_LEVELS[0]) & obs
            r["cri_lo_min_bfdr0.05_set"] = float(nf["cred_lo"][strong].min()) if strong.any() else np.nan
            r["expected_true_observed_nuts"] = float(np.sum(data.n[obs] * nf["cred"][obs]))
            r["expected_true_all_nuts"] = float(np.sum(data.n * nf["cred"]))
            r["expected_true_observed_map"] = float(np.sum(data.n[obs] * mp[obs]))
            r["expected_true_all_map"] = float(np.sum(data.n * mp))
            r["w_T_map"] = float(fits["family_hb"]["params"]["w"][0])
            r["w_T_nuts_mean"] = float(nf["params"]["w"][0])
            mvn.append(r)
    return pd.DataFrame(mvn), pd.DataFrame(pairs)


def fit_value_provenance(universe: str, fit_id: str, variant: str = VARIANT, root: Path = FITS_DIR) -> tuple:
    """(values_changed_post_hoc, value_change_note) of one stored fit, from its meta.json (robfix, verifier
    fitfix minor 11): whether the stored fit was recomputed after the project's first held-out outcomes
    (2026-10-08 16:34 UTC), although the quantity is prespecified.  MAP fits carrying a ``polish`` record
    were refitted in the fit_fixes round (Newton-polished optimum); NUTS fits with ``frozen`` settings were
    re-run (compared with the superseded first-pass fit in ``_superseded_firstpass`` where it was kept);
    first-pass NUTS fits whose chains were relabelled by patch_randinit_layout.py are flagged too."""
    d = fit_dir(universe, fit_id, variant, root)
    if not (d / "meta.json").exists():
        return (False, "")
    m = json.loads((d / "meta.json").read_text())
    if m.get("inference") == "MAP":
        if "polish" in m:
            return (True, "MAP refit in the fit_fixes round: stored optimum = Newton-polished best restart "
                          "(credibility change <= 2.2e-5)")
        return (False, "first-pass MAP fit")
    dec = m.get("settings_decision", {})
    if dec.get("requested") == "frozen":
        note = "NUTS re-run in the fit_fixes round with the frozen settings from the polished MAP optimum"
        old = root / "_superseded_firstpass" / f"{universe}__{variant}__{fit_id}" / "meta.json"
        if old.exists():
            o = json.loads(old.read_text())
            a = (o.get("num_chains"), o.get("num_warmup"), o.get("num_samples"))
            b = (m.get("num_chains"), m.get("num_warmup"), m.get("num_samples"))
            note += (f"; chain length {a[0]} x {a[1]}/{a[2]} -> {b[0]} x {b[1]}/{b[2]}" if a != b
                     else f"; chain length unchanged ({b[0]} x {b[1]}/{b[2]})")
        else:
            note += "; first-pass fit not kept"
        return (True, note)
    if "relabel" in str(m.get("provenance_note", "")):
        return (True, "first-pass NUTS sampling; chain / class labels relabelled post hoc by patch_randinit_layout.py")
    return (False, "first-pass NUTS fit")


VALUE_CHANGE_COLUMNS = ("values_changed_post_hoc", "value_change_note")


def add_value_provenance(name: str, df: pd.DataFrame, variant: str = VARIANT, root: Path = FITS_DIR) -> pd.DataFrame:
    """Add ``VALUE_CHANGE_COLUMNS`` per row: the provenance of the fit(s) the row is computed from
    (``fit_id``; ``fit_a`` and ``fit_b``; Family-HB MAP and NUTS for ``fit_map_vs_nuts.csv``)."""
    if df.empty or "universe" not in df.columns:
        return df
    df = df.drop(columns=[c for c in VALUE_CHANGE_COLUMNS if c in df.columns]).copy()
    cache: dict = {}

    def prov(u, f):
        if (u, f) not in cache:
            cache[(u, f)] = fit_value_provenance(u, f, variant, root)
        return cache[(u, f)]

    if name == "fit_map_vs_nuts.csv":
        fids = [("family_hb", "family_hb_nuts")] * len(df)
    elif "fit_a" in df.columns:
        fids = list(zip(df["fit_a"], df["fit_b"]))
    else:
        fids = [(f,) for f in df["fit_id"]]
    flags, notes = [], []
    for u, fs in zip(df["universe"], fids):
        ps = [prov(u, f) for f in fs]
        flags.append(any(p[0] for p in ps))
        notes.append(" | ".join(f"{f}: {p[1]}" for f, p in zip(fs, ps) if p[1]) if len(fs) > 1 else ps[0][1])
    df["values_changed_post_hoc"] = flags
    df["value_change_note"] = notes
    return df


def post_hoc_columns(name: str, columns) -> list:
    """Columns of table ``name`` added or redefined in the fit_fixes round (see ``FIRST_PASS_COLUMNS``)."""
    first = set(FIRST_PASS_COLUMNS[name])
    renamed = RENAMED_COLUMNS.get(name, {})
    redefined = set(REDEFINED_COLUMNS.get(name, []))
    return [c for c in columns
            if c not in STATUS_COLUMNS and ((c not in first and c not in renamed) or c in redefined)]


def label_analysis_status(name: str, df: pd.DataFrame) -> pd.DataFrame:
    """Add ``analysis_status`` (row level: 'prespecified' / 'post_hoc') and ``post_hoc_columns``
    (';'-joined columns added or redefined after the project's first held-out outcomes)."""
    df = df.drop(columns=[c for c in STATUS_COLUMNS if c in df.columns])
    if df.empty:
        return df
    status = np.full(len(df), "prespecified", dtype=object)
    if name == "fit_source_parameters.csv":
        status[df["parameter"].isin(POST_HOC_PARAMETERS).to_numpy()] = "post_hoc"
    cols = post_hoc_columns(name, df.columns)
    df = df.copy()
    df["analysis_status"] = status
    df["post_hoc_columns"] = ";".join(cols)
    return df


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


def build_tables(universes=UNIVERSES, variant=VARIANT, root=FITS_DIR, tables_dir: Path = C.TABLES) -> dict:
    C.ensure_dir(tables_dir)
    out = {}
    conv = convergence_table(universes, variant, root)
    out["fit_convergence.csv"] = conv
    out["fit_source_parameters.csv"] = source_parameter_table(universes, variant, root)
    out["fit_class_weights.csv"] = class_weight_table(universes, variant, root)
    out["fit_hyperparameters.csv"] = hyperparameter_table(universes, variant, root)
    opt = []
    for u in universes:
        for fid in MAP_MODELS:
            f = fit_dir(u, fid, variant, root) / "optima.csv"
            if f.exists():
                o = pd.read_csv(f)
                o.insert(0, "fit_id", fid)
                o.insert(0, "universe", u)
                opt.append(o.drop(columns=["restarts"]))
    out["fit_map_optima.csv"] = pd.concat(opt, ignore_index=True) if opt else pd.DataFrame()
    ch = []
    for u in universes:
        for fid in NUTS_FITS:
            f = fit_dir(u, fid, variant, root) / "chains.csv"
            if f.exists():
                o = pd.read_csv(f)
                o.insert(0, "fit_id", fid)
                o.insert(0, "universe", u)
                ch.append(o)
    out["fit_nuts_chains.csv"] = pd.concat(ch, ignore_index=True) if ch else pd.DataFrame()
    mvn, pairs = agreement_tables(universes, variant, root)
    out["fit_map_vs_nuts.csv"] = mvn
    out["fit_estimator_agreement.csv"] = pairs
    for name in list(out):
        out[name] = df = label_analysis_status(name, add_value_provenance(name, out[name], variant, root))
        df.to_csv(tables_dir / name, index=False, float_format="%.10g")
        print(f"[tables] wrote {tables_dir / name} ({len(df)} rows)", flush=True)
    record_decisions("fit_primary", ANALYSIS_DECISIONS, Path(tables_dir) / "eval_analysis_decisions.csv")
    return out


def edge_tables(tissues=C.TISSUES, variant=VARIANT, root=FITS_DIR) -> None:
    """Per-tissue edge-level credibility table (evidence edge order, joined by pattern index)."""
    for t in tissues:
        data = load_data(t, variant)
        ev = pd.read_parquet(E.EV_DIR / t / f"evidence{E._suffix(variant)}.parquet", columns=["tf", "target"])
        ep = np.asarray(E.load_edge_pattern(t, variant, mmap=False))
        assert len(ev) == len(ep), (t, len(ev), len(ep))
        df = ev.copy()
        df["pattern"] = ep.astype(np.int32)
        df["n_observing"] = (data.R >= 0).sum(1)[ep].astype(np.int16)
        df["n_reporting"] = (data.R == 1).sum(1)[ep].astype(np.int16)
        for fid in MAP_MODELS:
            f = _read_fit(t, fid, variant, root)
            if f is not None:
                df[f"cred_{fid}"] = f["cred"][ep]
        nf = _read_fit(t, "family_hb_nuts", variant, root)
        if nf is not None:
            df["cred_family_hb_nuts_mean"] = nf["cred"][ep]
            df["cred_family_hb_nuts_lo"] = nf["cred_lo"][ep]
            df["cred_family_hb_nuts_hi"] = nf["cred_hi"][ep]
            df["cred_family_hb_nuts_sd"] = nf["cred_sd"][ep]
        path = C.ensure_dir(root / t) / "primary_edge_cred.parquet"
        df.to_parquet(path, index=False)
        print(f"[edge-tables] wrote {path} ({len(df)} edges)", flush=True)


# ----------------------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--universes", nargs="+", default=list(UNIVERSES), choices=list(UNIVERSES))
        p.add_argument("--variant", default=VARIANT)
        p.add_argument("--root", type=Path, default=FITS_DIR)

    p = sub.add_parser("map", help="MAP fits of the five estimators")
    common(p)
    p.add_argument("--models", nargs="+", default=list(MAP_MODELS), choices=list(MAP_MODELS))
    p.add_argument("--skip-existing", action="store_true")
    p = sub.add_parser("laplace", help="Laplace mass of every distinct MAP optimum (needs map)")
    common(p)
    p.add_argument("--models", nargs="+", default=list(MAP_MODELS), choices=list(MAP_MODELS))
    p = sub.add_parser("nuts", help="NUTS fits of family_hb")
    common(p)
    p.add_argument("--settings", default="frozen", choices=["frozen", "auto", "full", "reduced"])
    p.add_argument("--budget-min", type=float, default=NUTS_BUDGET_MIN)
    p.add_argument("--init", nargs="+", default=["random", "map_laplace"], choices=["random", "map_laplace"])
    p.add_argument("--map-root", type=Path, default=None,
                   help="fits root holding the MAP fits (default: --root); lets a NUTS run write to a staging root")
    p.add_argument("--skip-existing", action="store_true")
    p = sub.add_parser("tables", help="manuscript tables from stored fits")
    common(p)
    p.add_argument("--tables-dir", type=Path, default=C.TABLES)
    p = sub.add_parser("edge-tables", help="per-tissue edge-level credibility tables")
    common(p)
    p = sub.add_parser("all", help="map + nuts + tables + edge-tables")
    common(p)
    p.add_argument("--settings", default="frozen", choices=["frozen", "auto", "full", "reduced"])
    p.add_argument("--budget-min", type=float, default=NUTS_BUDGET_MIN)
    p.add_argument("--init", nargs="+", default=["random", "map_laplace"], choices=["random", "map_laplace"])
    p.add_argument("--skip-existing", action="store_true")
    p.add_argument("--tables-dir", type=Path, default=C.TABLES)
    a = ap.parse_args(argv)

    if a.cmd in ("map", "all"):
        models = getattr(a, "models", MAP_MODELS)
        for u in a.universes:
            data = load_data(u, a.variant)
            for m in models:
                if a.skip_existing and (fit_dir(u, m, a.variant, a.root) / "meta.json").exists():
                    print(f"[map] skip existing {u}/{m}", flush=True)
                    continue
                run_map(u, m, a.variant, root=a.root, data=data)
    if a.cmd in ("laplace", "all"):
        for u in a.universes:
            data = load_data(u, a.variant)
            for m in a.models if a.cmd == "laplace" else MAP_MODELS:
                laplace_optima(u, m, a.variant, a.root, data=data)
    if a.cmd in ("nuts", "all"):
        for init in a.init:
            fid = "family_hb_nuts" if init == "map_laplace" else "family_hb_randinit_nuts"
            for u in a.universes:
                if a.skip_existing and (fit_dir(u, fid, a.variant, a.root) / "meta.json").exists():
                    print(f"[nuts] skip existing {u}/{fid}", flush=True)
                    continue
                run_nuts(u, a.variant, settings=a.settings, budget_min=a.budget_min, root=a.root, init=init,
                         map_root=getattr(a, "map_root", None))
    if a.cmd in ("tables", "all"):
        build_tables(a.universes, a.variant, a.root, a.tables_dir)
    if a.cmd in ("edge-tables", "all"):
        edge_tables([u for u in a.universes if u in C.TISSUES], a.variant, a.root)


if __name__ == "__main__":
    main()
