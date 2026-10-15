"""Robustness of credibility estimators to annotator dropout (evaluation stage E1;
reviewer Q11 and Q13; replaces R1 Table 1 / Fig. 1).

Question
--------
How much do (1) the ranking of GRN-inference methods scored against a credibility
vector and (2) the edge-level credibility itself change when one annotator, one
evidence-modality family or one integrator lineage is removed?  Every estimator is
stressed in the same way: the five latent-class models of ``lcm.py`` (EM /
Dawid-Skene, dependence-aware, multifactor, family-regularized, Family-HB; MAP) and
five pattern-level combination rules (union, majority, vote fraction, Fisher,
Stouffer).  No held-out data (ChIP references, perturbation cohorts) is read.

Design (every choice is fixed here or in ``common.PRESPECIFIED``)
------------------------------------------------------------------
Universes: the tissue context universes immune, kidney and lung (method ranking and
edge level) and the global universe (edge level only), primary evidence variant
``PRESPECIFIED['evidence_variant_primary']``.

Configurations (``configurations``):

* ``full``     all K held-in annotators;
* ``source``   drop each annotator in turn (K configurations);
* ``family``   drop each modality family (LIT, MOTIF, COEXP);
* ``lineage``  drop each integrator lineage (CollecTRI, DoRothEA, Multi-integrator;
               ``data/r2/inputs/annotator_sources.json``), the alternative grouping.
               The model families stay the modality families.

Dropout removes the annotators' columns and re-compresses the patterns
(``lcm.drop_annotators``).  The observation mask of the remaining annotators is kept
as built (under the curated-conditional mask MOTIF/COEXP observe the held-in
literature conditioning set, which is not recomputed without the dropped LIT
sources): the stress test removes reports, not the design.  ``family:COEXP`` and
``source:ARACNe-GTEx`` remove the same single annotator and share one fit.

MAP fitting of the latent-class models (``fit_starts``):

* full data: ``N_RESTARTS_FULL`` = 50 random starts (plan section 6), the same start
  points as ``lcm.fit_map(..., n_restarts=50, seed=PRESPECIFIED['random_seed'])``;
* dropout: a warm start at the full-data optimum restricted to the remaining
  annotators (``restrict_params``: annotator effects of dropped annotators and family
  effects of removed families are deleted, class weights of burst classes that no
  longer exist are removed and the rest renormalised) plus ``N_RANDOM_DROPOUT`` = 5
  random starts, then an ADAPTIVE GLOBAL SEARCH (fit_fixes 2026-10-08, verifier M1):
  further random starts are added one at a time until at least ``MIN_RANDOM_AT_BEST`` = 2
  random starts reach the best mode found (log posterior within ``NEAR_BEST``) or
  ``N_RANDOM_MAX`` = 100 random starts have been run (the cap was 25 in the first run of the
  rule on 2026-10-13; six lineage-dropout fits of the global-burst models then stopped at the
  cap with < 2 random starts at the best mode, two of them at a new, 38-40 units better mode
  reached by a single start, so the cap was raised to 100 for exactly those fits).  Extra start r (r >= 5) is
  ``uniform(fold_in(rng, EXTRA_KEY_OFFSET + r), -2, 2)``, so the first starts are unchanged
  and every start is a deterministic function of (universe, configuration, model, r).
  Fits stored before the rule are completed by ``refit_incomplete_search`` (the affected
  model of a configuration is refitted from scratch with the adaptive rule).  The
  per-configuration diagnostic (``rob_fit_search.csv``): random starts run, random starts at
  the best mode, whether the warm start reached it, ``global_search_ok`` (>= 2 random starts
  at the best mode);
* each start: L-BFGS-B with the options of ``lcm.fit_map`` followed by Newton
  polishing with the exact (jax) Hessian of the negative log posterior until
  max|gradient| <= ``GRAD_TOL`` = 1e-6 (plan section 6).  A start is *converged* if it
  reaches that gradient tolerance at a positive-definite Hessian (strict local mode).
  The fit is the start with the highest log posterior; ``n_within_1e3`` and
  ``cred_spread_near_best`` record how many starts reach that mode and how much their
  credibilities differ.

Baselines (``lcm.baseline_scores``, recomputed on the reduced evidence after dropout)
have a *score* (used by rank-based edge metrics) and a *label* in [0, 1] (used wherever
a credibility is needed).  Fisher and Stouffer use MID-P per-annotator p values (reported:
rate_j / 2; observed negative: (1 + rate_j) / 2; ``lcm.baseline_scores``), a logged deviation
from plan section 6 (p = rate_j / 1), whose rule gives Phi^-1(0) = -inf for every Stouffer
negative:

=============  ===============================  =================================
estimator      score                            label
=============  ===============================  =================================
union          1[>= 1 report]                   = score
majority       1[>= 2 reports]                  = score
vote_fraction  reports / observing annotators   = score
fisher         -log p_Fisher                    1[p_Fisher <= 0.05]
stouffer       Z_Stouffer                       1[Z_Stouffer >= z_0.95]
lcm models     credibility P(T | pattern)       = score
=============  ===============================  =================================

The nominal level ``BASELINE_ALPHA`` = 0.05 is a per-edge combined-test decision rule
(not multiplicity adjusted), fixed before any evaluation.

(1) Method-ranking stability (tissue universes)
-----------------------------------------------
Method panel ``PANEL`` (genie3 for immune only).  A method's predicted set is
``r2.predicted_sets.topk_per_tf(k = PRESPECIFIED['topk_per_tf'])`` over the universe
(NaN / not returned = not predicted).  Under a label vector c over ALL universe edges
(posterior-weighted counts)::

    TP = sum_e pred_e c_e,  FP = sum_e pred_e (1 - c_e),  FN = sum_e (1 - pred_e) c_e
    precision = TP / |pred|,  recall = TP / sum_e c_e,  F1 = 2 TP / (|pred| + sum_e c_e)

Rank disagreement between the full-data and a dropout label vector (R1 Eq.)::

    D = #{a < b: s_ab != 0, s'_ab != 0, s_ab != s'_ab} / #{a < b: s_ab != 0, s'_ab != 0}

with s_ab = sign(F1(a) - F1(b)) (|difference| <= ``TIE_TOL`` counts as a tie).  Per
estimator x protocol: mean and max D over configurations, pooled flips / comparable
pairs with a Wilson 95% interval (treats pairs as independent; descriptive), a
percentile bootstrap interval of the mean D over configurations
(``PRESPECIFIED['n_bootstrap']`` resamples), and the number of configurations whose
top method differs from the full-data top method.  The same statistics are computed
for posterior-weighted precision.  ``tissue = all_tissues`` pools the configurations
of the three tissues.

(2) Edge-level stability (all universes)
----------------------------------------
For each estimator and dropout configuration, against the same estimator on full data:

* ``spearman``: Spearman correlation of the scores over the edges observed by >= 1
  remaining annotator (computed exactly on patterns with edge-count weights and
  mid-ranks for ties, ``weighted_spearman``);
* ``jaccard_bfdr``: Jaccard index of the Bayesian-FDR sets: the largest set of whole
  tie blocks of the label, in decreasing order, whose mean (1 - c) is <=
  ``PRESPECIFIED['bfdr_strong']`` (``bfdr_set``);
* ``jaccard_top``: Jaccard index of the top ``TOP_N`` = 2,000 edges by score, ties
  broken by one seeded uniform key per edge that is shared by all configurations
  (``PRESPECIFIED['topk_tie_seed']``; ``top_n_edges``);
* ``jaccard_top_sizematched`` (verifier M3): the same with n = |S|, the number of edges in
  the full-data Family-HB BFDR <= 0.05 set of the universe (``n_sizematched``), for every
  estimator.  The BFDR set of a 0/1 rule is its positive set (3-10x larger than |S| and,
  under union, a set that dropout can only shrink), so ``jaccard_bfdr`` flatters the hard
  rules; the size-matched comparison removes the size difference.  For 0/1 scores the top-n
  set is a seeded subsample of the positives with keys shared across configurations, so
  part of the hard rules' size-matched stability is still mechanical;
* ``max_abs_dlabel`` (max over all universe edges), ``mean_abs_dlabel_obs`` (edges
  observed by >= 1 remaining annotator), ``abs_dw_T`` = |w_T(dropout) - w_T(full)|
  (latent-class models) and ``abs_dprev`` = |mean label (dropout) - mean label (full)|.

Tie blocks use labels / scores rounded to ``TIE_DECIMALS`` = 12 decimals.

Identification and uninformative configurations (verifier minor 2)
------------------------------------------------------------------
* When one modality family is left (``lineage:DoRothEA``: only LIT), Family-HB's T and B_LIT
  classes are exactly exchangeable (equal log posterior), so its credibility is NOT
  IDENTIFIED: every Family-HB metric of such a configuration is NA (``identified`` = False;
  no label alignment is applied; ``align_exchangeable_labels`` is kept as a diagnostic).
* ``family:LIT`` cannot answer the dropout question: under the curated-conditional mask the
  MOTIF/COEXP observation pattern still encodes LIT membership.
* Family and lineage summaries and contrasts are reported for ``subset`` = ``all`` and
  ``excl_uninformative`` (without ``EXCLUDED_CONFIGS`` = family:LIT, lineage:DoRothEA).

Uncertainty (verifier minor 3)
------------------------------
Bootstrap intervals over configurations are given iid (``*_boot_lo/hi``) and clustered by
configuration name (``*_cluster_lo/hi``: the same source / family / lineage dropped in the
three tissues is one cluster in the pooled ``all_tissues`` scope).  ``significant`` /
``significant_cluster`` = the interval excludes 0.  Family and lineage protocols have 3
configurations per universe: their intervals are DESCRIPTIVE (``ci_status``).  No
multiplicity adjustment is applied across the ~600 contrast rows: every interval is an
UNADJUSTED 95% interval (about 9 estimators x 6 metrics x 5 scopes x 3-4 subsets of contrasts).
``n_boot`` records the resamples of each row (``PRESPECIFIED['n_bootstrap']`` = 1,000 for the
first-pass and fit_fixes rows, ``N_BOOT_POST_HOC`` = 4,000 for the robfix rows below).  The
bootstrap intervals of every summary row were recomputed in the fit_fixes round (the shared
random stream shifted when non-identified values became NA), so they differ from the first pass
in the Monte Carlo digits even where the point values did not change.

Posterior-mode switches (robfix 2026-10-14, verifier fitfix M1; post hoc)
-------------------------------------------------------------------------
A latent-class dropout fit has a MODE SWITCH (``mode_switch``) when its selected optimum is not
the warm-started continuation of the full-data mode, read from the stored start diagnostics: the
polished warm start ends more than ``NEAR_BEST`` below the best start (``fit_warm_gap >
NEAR_BEST``; under the selection rule this is equivalent to ``fit_best_start != 'warm'``, which is
checked) or no warm start exists.  ``fhb_mode_switch`` marks every row of a universe x
configuration in which Family-HB has a mode switch.  The method-ranking and edge-stability
summaries and the paired contrasts are repeated, as post hoc rows, without those configurations:
``subset`` = ``no_fhb_mode_switch`` (every protocol) and ``excl_uninformative_no_fhb_mode_switch``
(family / lineage: also without ``EXCLUDED_CONFIGS``), with ``N_BOOT_POST_HOC`` = 4,000 resamples
(source-clustered in the pooled scope) from random streams of their own, so that no earlier row
changes.  Every summary / contrast row carries ``n_fhb_mode_switch`` (configurations of the row
with a Family-HB switch) and, for the restricted subsets, ``fhb_mode_switch_excluded`` (the
configurations of the parent subset removed, ``universe:config``).  ``rob_mode_switches.csv``
lists every mode switch of every latent-class model with its diagnostics and stability metrics;
``rob_fit_convergence.csv`` counts them per universe x model x protocol.

Further robfix additions (verifier fitfix minors 1, 8, 11; post hoc)
-------------------------------------------------------------------
* ``jaccard_top_obs`` (minor 1): ``jaccard_top`` restricted to configurations in which both
  top-2,000 sets consist of observed edges, i.e. NA when 2,000 exceeds the number of observed
  edges whose score is strictly above the highest score of an unobserved edge (in the full or the
  dropout configuration; ``n_observed_above_unobserved``; the rule of fit_primary verifier f1-M2).
  When it is defined, the top-2,000 set over all edges equals the top-2,000 set over the observed
  edges, so the value equals ``jaccard_top``.  In the tissue universes the first-pass
  ``jaccard_top`` of the latent-class models mostly compares seeded subsamples of the unobserved
  tie block (``n_unobserved_top`` counts its unobserved edges) and should not be interpreted;
  ``jaccard_top_sizematched`` (``n_unobserved_top_sizematched``) is the edge-level set metric.
* ``fit_basin_fraction`` = random starts at the best mode / random starts (minor 8), and an
  independent-seed check (``seedcheck``): every dropout fit whose adaptive search needed extra
  random starts gets ``SEEDCHECK_N_STARTS`` = 30 fresh random starts from an independent key; the
  result (does any fresh start beat the stored optimum?) is merged into ``rob_fit_search.csv``.
  When a fresh start is more than ``NEAR_BEST`` above the stored optimum, its optimum is ADOPTED as
  the fit (``adopt_seedcheck``; the MAP is the best optimum found): the fresh starts join the start
  records as ``seedcheck<r>``, ``fit_seedcheck_adopted`` marks the rows and the superseded fit is kept
  in ``fits/<universe>/_seedcheck_superseded/``.  On 2026-10-14 this happened once (lung
  lineage:DoRothEA, multifactor model: +52.0 log-posterior units).
* ``values_changed_post_hoc`` (minor 11): rows whose point values changed in the fit_fixes round
  although the quantity was prespecified -- detail rows of the fits whose best mode the adaptive
  search changed and the non-identified Family-HB rows (lineage:DoRothEA, now NA), and every
  summary / contrast row that aggregates such a detail row (e.g. the Family-HB lineage ``all``
  rows, which average 6 of 9 configurations).  The fit diagnostics (``fit_*``) of the 39 dropout
  fits with extra random starts also changed.

EM-referenced contrasts (assets2 2026-10-14, verifiers robfix M1(a) and assets_fix M4; post hoc)
-----------------------------------------------------------------------------------------------
Statements of the form "estimator X is less stable than EM" need an interval for X - EM, which the
Family-HB-referenced block does not give.  ``em_reference_contrasts`` repeats ``_contrast_rows`` with
``reference`` = ``EM_REFERENCE`` = 'em' for every metric, scope and protocol and for the subsets ``all`` /
``excl_uninformative`` and the two no-Family-HB-switch subsets, with ``N_BOOT_POST_HOC`` = 4,000 resamples
(source-clustered in the pooled scope) from two random streams of their own (``RNG_STREAM``
em_reference_contrasts / em_reference_switch_contrasts).  The rows are appended to
``rob_estimator_contrasts.csv`` after the Family-HB-referenced rows, which are unchanged, and are all
``analysis_status`` = post_hoc.  Readers of the table must therefore filter on ``reference``.

Outputs
-------
``outputs/r2/robustness/fits/<universe>/{full,drop_<hash>}.npz`` (+ ``.json``
start diagnostics), ``fit_<universe>.log`` and the manuscript tables in
``data/r2/tables``:

* ``rob_method_ranking.csv``    rank disagreement per tissue x estimator x protocol x metric;
* ``rob_edge_stability.csv``    edge-level stability per universe x estimator x protocol;
* ``rob_config_detail.csv``     every universe x estimator x configuration (incl. fit diagnostics);
* ``rob_method_scores_full.csv`` posterior-weighted precision / recall / F1 of each method under
  each estimator's full-data labels, with TF-cluster bootstrap intervals and paired differences
  against random and the 1/L uniform-attention null (the FM benchmarking table);
* ``rob_fit_convergence.csv``   convergence summary per universe x model x protocol (incl. global-search
  counts);
* ``rob_fit_search.csv``        per universe x configuration x model global-search diagnostic;
* ``rob_estimator_contrasts.csv`` paired contrasts of every estimator with Family-HB (mean difference
  in D, Spearman and Jaccard over the same configurations, paired bootstrap interval, win counts),
  followed by the post hoc EM-referenced contrasts (``reference`` = 'em'; see below);
* ``rob_mode_switches.csv``     every latent-class dropout fit with a posterior-mode switch (robfix).

Every table carries ``analysis_status`` (row level: 'prespecified' = plan section 2.3 E1 or fixed
label-free in the first pass, before the project's first held-out outcome; 'post_hoc' = rows that
exist only since the fit_fixes round: ``excl_uninformative`` subsets, size-matched contrasts,
``rob_fit_search.csv``) and ``post_hoc_columns`` (columns added in that round); the dated decisions
(``ANALYSIS_DECISIONS``) are written to ``eval_analysis_decisions.csv``.  No held-out data is read;
'post hoc' only records that the quantity was designed after held-out results existed in the project.

Command line (from the biodyn-work root; set the thread limits of the project rules)::

    .venv/bin/python subproject_24_multireference_latent_truth_model/src/r2/robustness.py fit --universes immune kidney lung
    .venv/bin/python subproject_24_multireference_latent_truth_model/src/r2/robustness.py fit --universes global
    .venv/bin/python subproject_24_multireference_latent_truth_model/src/r2/robustness.py seedcheck --universes immune kidney
    .venv/bin/python subproject_24_multireference_latent_truth_model/src/r2/robustness.py seedcheck --universes lung global
    .venv/bin/python subproject_24_multireference_latent_truth_model/src/r2/robustness.py analyze
    .venv/bin/python subproject_24_multireference_latent_truth_model/src/r2/robustness.py all      # fit + analyze

``fit`` skips configurations whose fit file exists (``--force`` refits) and then completes the
adaptive global search of stored fits (``refit_incomplete_search``; also ``... robustness.py search``).
``seedcheck`` (post hoc, robfix) skips stored checks (``--force`` repeats them); ``analyze`` reads
whatever checks are stored (columns NA otherwise).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import zlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from r2 import common as C  # noqa: E402
from r2 import evidence as E  # noqa: E402
from r2 import lcm  # noqa: E402  (enables jax x64)
from r2.predicted_sets import topk_per_tf  # noqa: E402

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
from jax.flatten_util import ravel_pytree  # noqa: E402
from numpyro import handlers  # noqa: E402
from numpyro.infer import init_to_uniform, init_to_value  # noqa: E402
from numpyro.infer.util import initialize_model  # noqa: E402
from scipy.optimize import minimize  # noqa: E402
from scipy.stats import norm  # noqa: E402

# ----------------------------------------------------------------------------
# Fixed settings
# ----------------------------------------------------------------------------
OUT_DIR = C.OUT / "robustness"
FIT_DIR = OUT_DIR / "fits"
VARIANT = C.PRESPECIFIED["evidence_variant_primary"]
TISSUE_UNIVERSES = tuple(C.TISSUES)
UNIVERSES = TISSUE_UNIVERSES + ("global",)
SEED = int(C.PRESPECIFIED["random_seed"])
N_BOOT = int(C.PRESPECIFIED["n_bootstrap"])
TOPK = int(C.PRESPECIFIED["topk_per_tf"])
BFDR_Q = float(C.PRESPECIFIED["bfdr_strong"])
TIE_SEED = int(C.PRESPECIFIED["topk_tie_seed"])

N_RESTARTS_FULL = 50          # plan section 6
N_RANDOM_DROPOUT = 5          # initial random starts per dropout configuration (+ 1 warm start)
N_RANDOM_MAX = 100            # adaptive global search: at most this many random starts ...
MIN_RANDOM_AT_BEST = 2        # ... stopping once this many random starts reach the best mode
EXTRA_KEY_OFFSET = 1_000_000  # fold_in offset of the adaptive extra starts
EXCLUDED_CONFIGS = ("family:LIT", "lineage:DoRothEA")    # cannot answer the dropout question
INIT_RADIUS = 2.0             # as lcm.fit_map
LBFGS_OPTIONS = dict(maxiter=20000, gtol=1e-6, ftol=1e-15, maxcor=30)   # as lcm.fit_map
GRAD_TOL = 1e-6               # plan section 6: max|gradient| < 1e-6
NEWTON_MAX_ITER = 50
NEAR_BEST = 1e-3              # log-posterior distance counted as "same mode" (as MapFit.summary)
AMBIGUOUS_SPREAD = 0.01       # near-best starts whose credibilities differ by more: mode ambiguous

MODELS = tuple(lcm.MODEL_SPECS)            # em, dependence, multifactor, family_regularized, family_hb
BASELINES = ("union", "majority", "vote_fraction", "fisher", "stouffer")
ESTIMATORS = MODELS + BASELINES
BASELINE_ALPHA = 0.05
TOP_N = 2000
TIE_DECIMALS = 12
TIE_TOL = 1e-12
PROTOCOLS = ("source", "family", "lineage")

# robfix (2026-10-14, verifier fitfix M1 and minors 1 / 8 / 11; post hoc)
N_BOOT_POST_HOC = 4000                    # resamples of the mode-switch-restricted rows
SWITCH_REFERENCE = "family_hb"            # the model whose mode switches define the restricted subsets
NO_SWITCH = "no_fhb_mode_switch"
NO_SWITCH_EXCL = "excl_uninformative_no_fhb_mode_switch"
SWITCH_SUBSETS = (NO_SWITCH, NO_SWITCH_EXCL)
# separate random streams (np.random.default_rng([SEED, stream])) for post hoc rows / columns, so
# that the rows computed with the shared first-pass stream keep their values
RNG_STREAM = dict(edge_post_hoc_metrics=1, contrast_post_hoc_metrics=2, switch_summaries=3,
                  switch_contrasts=4, em_reference_contrasts=5, em_reference_switch_contrasts=6)
EM_REFERENCE = "em"                       # reference of the post hoc EM-referenced contrasts (assets2)
SEEDCHECK_N_STARTS = 30                   # fresh random starts of the independent-seed check
SEEDCHECK_SEED = 20261014                 # independent of SEED (the fits' keys fold SEED)
SEEDCHECK_DIR = OUT_DIR / "seedcheck"

PANEL = ["scgpt_attn", "scgpt_attn_T", "scgpt_attn_norm", "geneformer_attn", "geneformer_attn_T",
         "geneformer_attn_norm", "scgpt_attn_uniform", "grnboost2", "genie3", "pearson_abs",
         "spearman_abs", "random"]
PANEL_TISSUES = {"genie3": ("immune",)}
EXTRA_METHODS = ["geneformer_attn_uniform"]     # scored in the FM table, not ranked
OWN_UNIFORM = {"scgpt": "scgpt_attn_uniform", "geneformer": "geneformer_attn_uniform"}
EXPRESSION_METHODS = ("grnboost2", "genie3")


def panel_for(tissue: str) -> list:
    return [m for m in PANEL if tissue in PANEL_TISSUES.get(m, (tissue,))]


# ----------------------------------------------------------------------------
# Configurations
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class Config:
    protocol: str            # full | source | family | lineage
    name: str
    drop: tuple

    @property
    def cid(self) -> str:
        return "full" if self.protocol == "full" else f"{self.protocol}:{self.name}"

    @property
    def fit_key(self) -> str:
        """File stem shared by configurations that drop the same annotators."""
        if not self.drop:
            return "full"
        h = hashlib.sha1("\x1f".join(sorted(self.drop)).encode()).hexdigest()[:12]
        return f"drop_{h}"


def annotator_lineage(annotators) -> list:
    meta = json.loads(E.INPUT_SOURCES.read_text())["sources"]
    return [meta[a]["lineage"] for a in annotators]


def configurations(annotators, family, lineage) -> list:
    """Full data, then source, family and lineage dropout (sorted, deterministic)."""
    cfgs = [Config("full", "full", ())]
    for a in annotators:
        cfgs.append(Config("source", a, (a,)))
    for grp, labels in (("family", family), ("lineage", lineage)):
        for g in sorted(set(labels)):
            cfgs.append(Config(grp, g, tuple(a for a, x in zip(annotators, labels) if x == g)))
    return cfgs


# ----------------------------------------------------------------------------
# MAP fitting with warm starts and Newton polishing
# ----------------------------------------------------------------------------
def latent_sites(model) -> dict:
    """Latent sample sites of a numpyro model -> shape."""
    tr = handlers.trace(handlers.seed(model, rng_seed=0)).get_trace()
    return {k: tuple(np.shape(v["value"])) for k, v in tr.items()
            if v["type"] == "sample" and not v["is_observed"]}


def restrict_params(full_values: dict, full_classes: list, full_families: list, full_annotators: list,
                    red_data: lcm.PatternData, red_classes: list, red_sites: dict) -> dict:
    """Constrained initial values for the reduced model from a full-data optimum.

    Annotator-level sites (non-hierarchical ``b``/``a``/``g`` and the non-centred
    ``zj_*``) keep the rows of the remaining annotators, family effects ``zf_*`` the
    remaining families, global sites (``mu0_*``, ``omega_*``, ``tau_*``) are copied and
    the class weights ``w`` of the remaining classes are renormalised."""
    keep = np.array([full_annotators.index(a) for a in red_data.annotators])
    fam_keep = np.array([full_families.index(f) for f in red_data.families])
    vals = {}
    for name, shape in red_sites.items():
        v = np.asarray(full_values[name], dtype=np.float64)
        if name == "w":
            idx = [full_classes.index(c) for c in red_classes]
            ww = v[idx]
            out = ww / ww.sum()
        elif name.startswith("zf_"):
            out = v[fam_keep]
        elif name.startswith("zj_") or name in ("b", "a", "g"):
            out = v[keep]
        else:
            out = v
        if tuple(out.shape) != tuple(shape):
            raise ValueError(f"site {name}: restricted shape {out.shape} != model shape {shape}")
        vals[name] = out
    return vals


class _Objective:
    """Negative log posterior of an lcm model in the unconstrained space (as lcm.fit_map)."""

    def __init__(self, data: lcm.PatternData, model_name: str, rng):
        self.data = data
        self.model_name = model_name
        self.spec = lcm.MODEL_SPECS[model_name]
        self.model = lcm.build_model(data, self.spec)
        self.class_names = list(self.model.class_names)
        self.sites = latent_sites(self.model)
        info = initialize_model(rng, self.model, init_strategy=init_to_uniform(radius=INIT_RADIUS))
        self.flat0, self.unravel = ravel_pytree(info.param_info.z)
        self.postprocess = info.postprocess_fn
        f = lambda x: info.potential_fn(self.unravel(x))  # noqa: E731
        self.pot = jax.jit(f)
        self.grad = jax.jit(jax.grad(f))
        self.hess = jax.jit(jax.hessian(f))

    def unconstrain(self, values: dict, rng) -> np.ndarray | None:
        """Flat unconstrained vector of constrained ``values`` (None if not finite)."""
        info = initialize_model(rng, self.model, init_strategy=init_to_value(values=values))
        flat, _ = ravel_pytree(info.param_info.z)
        flat = np.asarray(flat)
        if flat.shape != np.asarray(self.flat0).shape or not np.all(np.isfinite(flat)):
            return None
        return flat

    def f(self, x):
        return float(self.pot(jnp.asarray(x)))

    def g(self, x):
        return np.asarray(self.grad(jnp.asarray(x)))

    def h(self, x):
        H = np.asarray(self.hess(jnp.asarray(x)))
        return 0.5 * (H + H.T)

    def constrained(self, x) -> dict:
        return {k: np.asarray(v) for k, v in self.postprocess(self.unravel(jnp.asarray(x))).items()}


def newton_polish(obj: _Objective, x: np.ndarray, tol: float = GRAD_TOL, max_iter: int = NEWTON_MAX_ITER):
    """Newton iterations with the exact Hessian from an L-BFGS solution.

    A step is accepted if it satisfies the Armijo condition, or if it lowers
    max|gradient| without raising the objective by more than float noise (near the
    mode the objective differences are below its rounding error).  Stops at
    max|grad| <= tol, at a non-positive-definite Hessian or when no step is accepted.
    Returns (x, n_steps, max|grad|, smallest Hessian eigenvalue at x)."""
    x = np.array(x, dtype=np.float64)
    fx, gx = obj.f(x), obj.g(x)
    gn = float(np.max(np.abs(gx)))
    hmin, steps = np.nan, 0
    for it in range(max_iter + 1):
        H = obj.h(x)
        if not np.all(np.isfinite(H)):
            hmin = np.nan
            break
        hmin = float(np.linalg.eigvalsh(H)[0])
        if gn <= tol or it == max_iter or hmin <= 0:
            break
        try:
            step = np.linalg.solve(H, gx)
        except np.linalg.LinAlgError:
            break
        slope = float(gx @ step)
        noise = 1e-10 * max(1.0, abs(fx))
        t, accepted = 1.0, False
        for _ in range(30):
            xn = x - t * step
            fn = obj.f(xn)
            if np.isfinite(fn):
                gnn = obj.g(xn)
                gnn_inf = float(np.max(np.abs(gnn)))
                if fn <= fx - 1e-4 * t * slope or (fn <= fx + noise and gnn_inf < gn):
                    accepted = True
                    break
            t *= 0.5
        if not accepted:
            break
        x, fx, gx, gn = xn, fn, gnn, gnn_inf
        steps += 1
    return x, steps, gn, hmin


def optimise_start(obj: _Objective, x0: np.ndarray) -> dict:
    t0 = time.time()
    res = minimize(obj.f, np.asarray(x0, dtype=np.float64), jac=obj.g, method="L-BFGS-B",
                   options=dict(LBFGS_OPTIONS))
    g_lbfgs = float(np.max(np.abs(obj.g(res.x))))
    x, steps, gn, hmin = newton_polish(obj, res.x)
    lp = -obj.f(x)
    return dict(x=x, log_post=lp, lbfgs_success=bool(res.success), lbfgs_message=str(res.message),
                lbfgs_n_iter=int(res.nit), grad_inf_lbfgs=g_lbfgs, newton_steps=int(steps), grad_inf=gn,
                hess_min_eig=hmin, converged=bool(gn <= GRAD_TOL and np.isfinite(hmin) and hmin > 0),
                runtime_s=time.time() - t0)


def random_start(rng, r: int, n_initial: int, shape) -> np.ndarray:
    """Start point of random start r: ``uniform(split(rng, n_initial)[r], -2, 2)`` for the first
    ``n_initial`` starts (exactly ``lcm.fit_map``), ``uniform(fold_in(rng, EXTRA_KEY_OFFSET + r))``
    for the adaptive extra starts."""
    key = (jax.random.split(rng, n_initial)[r] if r < n_initial
           else jax.random.fold_in(rng, EXTRA_KEY_OFFSET + r))
    return np.asarray(jax.random.uniform(key, shape, minval=-INIT_RADIUS, maxval=INIT_RADIUS))


RANDOM_LABELS = ("random", "seedcheck")     # 'seedcheck<r>': fresh starts merged into a fit by adopt_seedcheck


def random_at_best(recs: list) -> tuple[int, int]:
    """(random starts run, random starts within NEAR_BEST of the best log posterior); the fresh starts
    of an adopted independent-seed check (``seedcheck<r>``) count as random starts."""
    lp = np.array([r["log_post"] for r in recs], dtype=float)
    lp = np.where(np.isfinite(lp), lp, -np.inf)
    best = lp.max()
    rnd = [i for i, r in enumerate(recs) if str(r["start"]).startswith(RANDOM_LABELS)]
    return len(rnd), int(sum(lp[i] >= best - NEAR_BEST for i in rnd))


def select_start(recs: list) -> tuple:
    """Selection rule over start records: the warm start if it reaches the highest mode (within
    NEAR_BEST), which keeps the labelling of the full-data fit when modes are equivalent (e.g. T
    and B_LIT are exchangeable once only the LIT family remains); otherwise the highest log
    posterior.  Returns (selected index, index of the maximum, near-best mask, log posteriors,
    log posteriors with non-finite values as -inf)."""
    lp = np.array([r["log_post"] for r in recs], dtype=float)
    lp_ok = np.where(np.isfinite(lp), lp, -np.inf)
    i_max = int(np.argmax(lp_ok))
    near = lp_ok >= lp_ok[i_max] - NEAR_BEST
    ib = 0 if (recs[0]["start"] == "warm" and near[0]) else i_max
    return ib, i_max, near, lp, lp_ok


def needs_search_refit(info: dict) -> bool:
    """True if a stored dropout fit (json entry with ``starts`` and ``summary``) has to be
    refitted by ``refit_incomplete_search``: the adaptive rule is not met and the cap not reached,
    or extra starts were used but the stored summary predates the search diagnostic fields.  A fit
    whose optimum was adopted from the independent-seed check (``adopt_seedcheck``) is final."""
    if info["summary"].get("seedcheck_adopted"):
        return False
    n_rnd, n_best = random_at_best(info["starts"])
    if n_best < MIN_RANDOM_AT_BEST and n_rnd < N_RANDOM_MAX:
        return True
    return n_rnd > N_RANDOM_DROPOUT and "search_log_post_gain" not in info["summary"]


def fit_starts(data: lcm.PatternData, model_name: str, rng, n_random: int, warm_values: dict | None = None,
               n_random_max: int | None = None, min_random_at_best: int = 0) -> dict:
    """MAP fit from an optional warm start plus ``n_random`` random starts, optionally followed
    by the adaptive global search: extra random starts (``random_start``) until
    ``min_random_at_best`` random starts reach the best mode or ``n_random_max`` random starts.

    Random start r < n_random is ``uniform(split(rng, n_random)[r], -2, 2)`` in the
    unconstrained space, exactly as ``lcm.fit_map(seed=s)`` with ``rng = PRNGKey(s)``.  Returns
    the best start's constrained parameters, credibility and per-start diagnostics."""
    t0 = time.time()
    obj = _Objective(data, model_name, rng)
    shape = np.asarray(obj.flat0).shape
    recs, creds, samples = [], [], []

    def run(label, x0):
        rec = optimise_start(obj, x0)
        sample = obj.constrained(rec.pop("x"))
        cred = lcm.credibility_at(sample, data)
        rec.update(start=label, loglik=lcm.heldout_loglik(sample, data))
        recs.append(rec)
        creds.append(cred)
        samples.append(sample)

    if warm_values is not None:
        xw = obj.unconstrain(warm_values, rng)
        if xw is not None:
            run("warm", xw)
    for r in range(n_random):
        run(f"random{r}", random_start(rng, r, n_random, shape))
    # the fit the initial starts alone would give (warm + n_random random starts): reference of
    # the search diagnostic, a function of the stored starts only (no fitting history)
    ib0 = select_start(recs)[0]
    lp0, cred0, start0 = float(recs[ib0]["log_post"]), creds[ib0], recs[ib0]["start"]
    n_extra = 0
    if n_random_max is not None:
        r = n_random
        while r < n_random_max and random_at_best(recs)[1] < min_random_at_best:
            run(f"random{r}", random_start(rng, r, n_random, shape))
            r += 1
            n_extra += 1
    n_rnd, n_rnd_best = random_at_best(recs)
    ib, i_max, near, lp, lp_ok = select_start(recs)
    spread = float(np.max(np.abs(np.stack(creds)[near] - creds[ib][None, :])))
    warm = [r for r in recs if r["start"] == "warm"]
    best = samples[ib]
    return dict(model=model_name, class_names=obj.class_names, sites=list(obj.sites),
                values={k: best[k] for k in obj.sites}, w=best["w"], theta=best["theta"], cred=creds[ib],
                log_post=float(lp[ib]), loglik=float(recs[ib]["loglik"]), best_start=recs[ib]["start"],
                starts=recs,
                summary=dict(n_starts=len(recs), n_converged=int(sum(r["converged"] for r in recs)),
                             n_lbfgs_success=int(sum(r["lbfgs_success"] for r in recs)),
                             best_start=recs[ib]["start"], best_converged=bool(recs[ib]["converged"]),
                             best_grad_inf=float(recs[ib]["grad_inf"]),
                             best_hess_min_eig=float(recs[ib]["hess_min_eig"]),
                             log_post=float(lp[ib]), loglik=float(recs[ib]["loglik"]),
                             log_post_max=float(lp_ok[i_max]),
                             log_post_range=float(lp_ok.max() - lp_ok[np.isfinite(lp)].min()),
                             n_within_1e3=int(near.sum()), cred_spread_near_best=spread,
                             warm_log_post=float(warm[0]["log_post"]) if warm else np.nan,
                             warm_gap=float(lp_ok[i_max] - warm[0]["log_post"]) if warm else np.nan,
                             warm_converged=bool(warm[0]["converged"]) if warm else None,
                             n_random_starts=n_rnd, n_random_at_best=n_rnd_best, n_extra_random_starts=n_extra,
                             adaptive_search=n_random_max is not None,
                             global_search_ok=bool(n_rnd_best >= MIN_RANDOM_AT_BEST),
                             initial_rule_start=start0, initial_rule_log_post=lp0,
                             search_log_post_gain=float(lp[ib] - lp0),
                             search_changed_best_mode=bool(lp[ib] - lp0 > NEAR_BEST),
                             search_max_abs_cred_change=float(np.max(np.abs(creds[ib] - cred0))),
                             w_T=float(best["w"][0]), runtime_s=time.time() - t0))


def _rng_for(universe: str, cfg: Config, model: str):
    if cfg.protocol == "full":
        return jax.random.PRNGKey(SEED)
    tag = zlib.crc32(f"{universe}|{cfg.fit_key}|{model}".encode()) & 0x7FFFFFFF
    return jax.random.fold_in(jax.random.PRNGKey(SEED), tag)


def _save_fit(path: Path, cfg_ids: list, fits: dict, extra: dict) -> None:
    arrays = {}
    for m, f in fits.items():
        arrays[f"{m}__cred"] = f["cred"]
        arrays[f"{m}__w"] = f["w"]
        arrays[f"{m}__theta"] = f["theta"]
        for k, v in f["values"].items():
            arrays[f"{m}__site__{k}"] = np.asarray(v)
    C.ensure_dir(path.parent)
    np.savez_compressed(path.with_suffix(".npz"), **arrays)
    js = dict(configs=cfg_ids, **extra,
              models={m: dict(class_names=f["class_names"], sites=f["sites"], summary=f["summary"],
                              starts=f["starts"]) for m, f in fits.items()})
    C.write_json(js, path.with_suffix(".json"))


def load_fit(universe: str, fit_key: str) -> dict:
    p = FIT_DIR / universe / fit_key
    js = json.loads(p.with_suffix(".json").read_text())
    z = np.load(p.with_suffix(".npz"))
    out = {}
    for m, info in js["models"].items():
        out[m] = dict(cred=z[f"{m}__cred"], w=z[f"{m}__w"], theta=z[f"{m}__theta"],
                      values={k: z[f"{m}__site__{k}"] for k in info["sites"]}, sites=info["sites"],
                      class_names=info["class_names"], summary=info["summary"], starts=info["starts"])
    out["_meta"] = js
    return out


def run_fits(universe: str, force: bool = False, models=MODELS, n_restarts_full: int = N_RESTARTS_FULL,
             n_random: int = N_RANDOM_DROPOUT, log=print, reverse: bool = False, reselect: bool = True) -> None:
    """Fit the full data and every dropout configuration of ``universe``.

    Stored configurations are skipped (unless ``force``).  A configuration being fitted is
    marked by ``<fit_key>.lock`` (created atomically), so several processes may share one
    universe (``reverse`` walks the configurations backwards); a stale lock from a killed
    process has to be removed by hand.  The result does not depend on which process fits a
    configuration (every start point is a function of universe, configuration and model)."""
    data = lcm.PatternData(*E.load_patterns(universe, VARIANT))
    cfgs = configurations(data.annotators, data.family, annotator_lineage(data.annotators))
    d = C.ensure_dir(FIT_DIR / universe)
    full_path = d / "full"
    if force or not full_path.with_suffix(".npz").exists():
        fits = {}
        for m in models:
            fits[m] = fit_starts(data, m, _rng_for(universe, cfgs[0], m), n_restarts_full)
            s = fits[m]["summary"]
            log(f"[{universe}] full {m}: log_post {s['log_post']:.4f} conv {s['n_converged']}/{s['n_starts']} "
                f"near-best {s['n_within_1e3']} grad {s['best_grad_inf']:.1e} w_T {s['w_T']:.5f} "
                f"{s['runtime_s']:.0f}s")
        _save_fit(full_path, ["full"], fits, dict(universe=universe, variant=VARIANT, drop=[],
                                                  annotators=data.annotators, family=data.family,
                                                  n_patterns=data.P))
    full = load_fit(universe, "full")
    by_key: dict = {}
    for cfg in cfgs[1:]:
        by_key.setdefault(cfg.fit_key, []).append(cfg)
    items = list(by_key.items())
    for key, group in (items[::-1] if reverse else items):
        path = d / key
        if not force and path.with_suffix(".npz").exists():
            continue
        lock = path.with_suffix(".lock")
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, f"{os.getpid()}\n".encode())
            os.close(fd)
        except FileExistsError:
            log(f"[{universe}] {group[0].cid}: locked by another process, skipped")
            continue
        cfg = group[0]
        red, _ = lcm.drop_annotators(data, cfg.drop)
        fits = {}
        for m in models:
            red_model = lcm.build_model(red, lcm.MODEL_SPECS[m])
            warm = restrict_params(full[m]["values"], full[m]["class_names"], data.families, data.annotators,
                                   red, list(red_model.class_names), latent_sites(red_model))
            fits[m] = fit_starts(red, m, _rng_for(universe, cfg, m), n_random, warm_values=warm,
                                 n_random_max=N_RANDOM_MAX, min_random_at_best=MIN_RANDOM_AT_BEST)
            s = fits[m]["summary"]
            log(f"[{universe}] {cfg.cid} {m}: best {s['best_start']} gap_warm {s['warm_gap']:.2e} "
                f"conv {s['n_converged']}/{s['n_starts']} grad {s['best_grad_inf']:.1e} {s['runtime_s']:.0f}s")
        _save_fit(path, [c.cid for c in group], fits,
                  dict(universe=universe, variant=VARIANT, drop=sorted(cfg.drop), annotators=red.annotators,
                       family=red.family, n_patterns=red.P))
        lock.unlink(missing_ok=True)
    if reselect:
        reselect_warm(universe, log=log)
    refit_incomplete_search(universe, models=models, log=log)


def refit_incomplete_search(universe: str, models=MODELS, log=print) -> int:
    """Complete the adaptive global search of stored dropout fits.

    For every stored dropout configuration and latent-class model whose starts do not satisfy
    the rule (``MIN_RANDOM_AT_BEST`` random starts at the best mode, or ``N_RANDOM_MAX`` random
    starts run; ``needs_search_refit``), the model is refitted from scratch with
    ``fit_starts(..., n_random_max, min_random_at_best)`` -- exactly what ``fit --force``
    produces (the first warm + 5 random starts are reproduced, extra starts are added).  Other
    models of the configuration are untouched.  The search diagnostic stored with every fit
    (``initial_rule_log_post``, ``search_log_post_gain``, ``search_changed_best_mode``,
    ``search_max_abs_cred_change``) compares the selected fit with the fit that the warm start +
    5 random starts alone give; it is a function of the starts, not of the fitting history.
    Idempotent; returns the number of models refitted."""
    data = lcm.PatternData(*E.load_patterns(universe, VARIANT))
    cfgs = configurations(data.annotators, data.family, annotator_lineage(data.annotators))
    first = {}
    for cfg in cfgs[1:]:
        first.setdefault(cfg.fit_key, cfg)
    full = None
    n_refit = 0
    for key, cfg in first.items():
        path = FIT_DIR / universe / key
        if not path.with_suffix(".json").exists():
            continue
        js = json.loads(path.with_suffix(".json").read_text())
        todo = [m for m, info in js["models"].items() if m in models and needs_search_refit(info)]
        if not todo:
            continue
        full = full or load_fit(universe, "full")
        fit = load_fit(universe, key)
        red, _ = lcm.drop_annotators(data, cfg.drop)
        for m in todo:
            old = fit[m]["summary"]
            red_model = lcm.build_model(red, lcm.MODEL_SPECS[m])
            warm = restrict_params(full[m]["values"], full[m]["class_names"], data.families, data.annotators,
                                   red, list(red_model.class_names), latent_sites(red_model))
            new = fit_starts(red, m, _rng_for(universe, cfg, m), N_RANDOM_DROPOUT, warm_values=warm,
                             n_random_max=N_RANDOM_MAX, min_random_at_best=MIN_RANDOM_AT_BEST)
            s = new["summary"]
            d_cred = float(np.max(np.abs(new["cred"] - fit[m]["cred"])))
            fit[m] = dict(new)
            n_refit += 1
            log(f"[{universe}] {cfg.cid} {m}: adaptive search {s['n_random_starts']} random starts, "
                f"{s['n_random_at_best']} at best; stored log_post {old['log_post']:.4f} -> {s['log_post']:.4f} "
                f"(max|dcred| vs stored {d_cred:.2e}); gain over warm + {N_RANDOM_DROPOUT} random starts "
                f"{s['search_log_post_gain']:.4f} (max|dcred| {s['search_max_abs_cred_change']:.2e})"
                f"{' BEST MODE CHANGED BY THE SEARCH' if s['search_changed_best_mode'] else ''}")
        meta = fit.pop("_meta")
        extra = {k: v for k, v in meta.items() if k not in ("configs", "models")}
        _save_fit(path, meta["configs"], fit, extra)
    return n_refit


def reselect_warm(universe: str, log=print) -> int:
    """Apply the warm-start selection rule of ``fit_starts`` to stored dropout fits.

    Fits stored before the rule was introduced kept the start with the numerically highest
    log posterior even when the warm start reached the same mode (gap <= NEAR_BEST).  For
    every such model the warm start is re-optimised (deterministic: same start point, same
    optimiser), checked against its stored log posterior and made the selected fit, which is
    what ``fit --force`` would produce.  ``cred_spread_near_best`` becomes the triangle
    bound old spread + max|cred(warm) - cred(old best)|.  Idempotent; returns the number of
    models changed."""
    data = lcm.PatternData(*E.load_patterns(universe, VARIANT))
    cfgs = configurations(data.annotators, data.family, annotator_lineage(data.annotators))
    first = {}
    for cfg in cfgs[1:]:
        first.setdefault(cfg.fit_key, cfg)
    full = None
    n_changed = 0
    for key, cfg in first.items():
        path = FIT_DIR / universe / key
        if not path.with_suffix(".json").exists():
            continue
        js = json.loads(path.with_suffix(".json").read_text())
        todo = [m for m, info in js["models"].items()
                if info["summary"]["best_start"] != "warm" and info["summary"]["warm_gap"] is not None
                and np.isfinite(info["summary"]["warm_gap"]) and info["summary"]["warm_gap"] <= NEAR_BEST]
        if not todo:
            continue
        full = full or load_fit(universe, "full")
        fit = load_fit(universe, key)
        red, _ = lcm.drop_annotators(data, cfg.drop)
        for m in todo:
            red_model = lcm.build_model(red, lcm.MODEL_SPECS[m])
            warm = restrict_params(full[m]["values"], full[m]["class_names"], data.families, data.annotators,
                                   red, list(red_model.class_names), latent_sites(red_model))
            new = fit_starts(red, m, _rng_for(universe, cfg, m), 0, warm_values=warm)
            rec = new["starts"][0]
            if rec["start"] != "warm":
                raise RuntimeError(f"{universe} {cfg.cid} {m}: warm start could not be initialised")
            stored = next(r for r in fit[m]["starts"] if r["start"] == "warm")
            s = dict(fit[m]["summary"])
            lp_max = s.get("log_post_max", s["log_post"])     # old rule: the selected fit had the max
            if abs(rec["log_post"] - stored["log_post"]) > NEAR_BEST or lp_max - rec["log_post"] > NEAR_BEST:
                log(f"[{universe}] {cfg.cid} {m}: re-optimised warm start log_post {rec['log_post']:.6f} differs "
                    f"from stored {stored['log_post']:.6f}; selection kept")
                continue
            d_old = float(np.max(np.abs(new["cred"] - fit[m]["cred"])))
            s.setdefault("log_post_max", s["log_post"])
            s.update(best_start="warm", best_converged=bool(rec["converged"]), best_grad_inf=float(rec["grad_inf"]),
                     best_hess_min_eig=float(rec["hess_min_eig"]), log_post=float(rec["log_post"]),
                     loglik=float(rec["loglik"]), w_T=float(new["w"][0]),
                     cred_spread_near_best=float(s["cred_spread_near_best"]) + d_old,
                     reselected_warm=True, reselect_max_abs_cred_change=d_old,
                     reselect_warm_log_post_rerun_diff=float(rec["log_post"] - stored["log_post"]))
            fit[m].update(cred=new["cred"], w=new["w"], theta=new["theta"], values=new["values"], summary=s)
            n_changed += 1
            log(f"[{universe}] {cfg.cid} {m}: reselected warm start (max|dcred| vs previous {d_old:.2e})")
        meta = fit.pop("_meta")
        extra = {k: v for k, v in meta.items() if k not in ("configs", "models")}
        _save_fit(path, meta["configs"], fit, extra)
    return n_changed


# ----------------------------------------------------------------------------
# Estimator outputs (pattern level)
# ----------------------------------------------------------------------------
def baseline_outputs(data: lcm.PatternData) -> dict:
    """est -> (score, label) per pattern for the combination rules."""
    b = lcm.baseline_scores(data)
    fisher_crit = -np.log(BASELINE_ALPHA)
    z_crit = float(norm.isf(BASELINE_ALPHA))
    return {
        "union": (b["union"], b["union"]),
        "majority": (b["majority"], b["majority"]),
        "vote_fraction": (b["vote_fraction"], b["vote_fraction"]),
        "fisher": (b["fisher"], (b["fisher"] >= fisher_crit).astype(float)),
        "stouffer": (b["stouffer"], (b["stouffer"] >= z_crit).astype(float)),
    }


def estimator_outputs(data: lcm.PatternData, fits: dict) -> dict:
    """est -> dict(score, label, w_T) per pattern of ``data``."""
    out = {}
    for m in MODELS:
        c = np.asarray(fits[m]["cred"], dtype=np.float64)
        out[m] = dict(score=c, label=c, w_T=float(fits[m]["w"][0]))
    for e, (s, lab) in baseline_outputs(data).items():
        out[e] = dict(score=np.asarray(s, dtype=np.float64), label=np.asarray(lab, dtype=np.float64), w_T=np.nan)
    return out


def class_posteriors(data: lcm.PatternData, w: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """(P, S) posterior class probabilities per pattern at parameters (w, theta)."""
    obs = (data.R >= 0).astype(float)
    pos = (data.R == 1).astype(float)
    lj = np.log(w)[None, :] + (obs * pos) @ np.log(theta) + (obs * (1 - pos)) @ np.log1p(-theta)
    return np.exp(lj - np.logaddexp.reduce(lj, axis=1)[:, None])


def family_hb_exchangeable(red: lcm.PatternData, class_names) -> bool:
    """True if Family-HB's T and its only burst class are exactly exchangeable (one modality
    family left): the credibility is then not identified."""
    return len(red.families) == 1 and list(class_names) == ["T", "N", f"B_{red.families[0]}"]


def align_exchangeable_labels(outs: dict, red: lcm.PatternData, fit: dict, inv: np.ndarray,
                              full_cred: np.ndarray, n: np.ndarray, obs: np.ndarray) -> bool:
    """Resolve the T <-> B_f label switch of Family-HB when one family is left.

    With a single modality family the Family-HB likelihood and prior are invariant under
    exchanging T and the family burst class B_f (a <-> g, w_T <-> w_B), so the MAP
    credibility is defined only up to this switch.  The labelling whose credibility has the
    higher edge-level Spearman correlation with the full-data Family-HB credibility (edges
    observed by >= 1 remaining annotator) is kept; ``outs['family_hb']`` is updated in place.
    Returns True if the labels were switched."""
    f = fit["family_hb"]
    if len(red.families) != 1 or list(f["class_names"]) != ["T", "N", f"B_{red.families[0]}"]:
        return False
    post = class_posteriors(red, np.asarray(f["w"]), np.asarray(f["theta"]))
    keep = weighted_spearman(full_cred[obs], post[:, 0][inv][obs], n[obs])
    alt = weighted_spearman(full_cred[obs], post[:, 2][inv][obs], n[obs])
    if not (np.isfinite(alt) and (not np.isfinite(keep) or alt > keep)):
        return False
    outs["family_hb"] = dict(score=post[:, 2], label=post[:, 2], w_T=float(np.asarray(f["w"])[2]))
    return True


# ----------------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------------
def _tie_round(x):
    return np.round(np.asarray(x, dtype=np.float64), TIE_DECIMALS)


def weighted_midranks(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Mid-ranks of the values x repeated w times (edge-level ranks of patterns)."""
    x = np.asarray(x)
    order = np.argsort(x, kind="mergesort")
    xs = x[order]
    new = np.r_[True, xs[1:] != xs[:-1]]
    gid = np.cumsum(new) - 1
    gw = np.bincount(gid, weights=np.asarray(w, dtype=np.float64)[order])
    mid = np.cumsum(gw) - gw + (gw + 1.0) / 2.0
    r = np.empty(len(x), dtype=np.float64)
    r[order] = mid[gid]
    return r


def weighted_spearman(x, y, w) -> float:
    """Spearman correlation over edges, given patterns (x, y) with edge counts w."""
    w = np.asarray(w, dtype=np.float64)
    if w.sum() <= 1:
        return np.nan
    rx, ry = weighted_midranks(_tie_round(x), w), weighted_midranks(_tie_round(y), w)
    mx, my = np.average(rx, weights=w), np.average(ry, weights=w)
    vx, vy = np.average((rx - mx) ** 2, weights=w), np.average((ry - my) ** 2, weights=w)
    if vx <= 0 or vy <= 0:
        return np.nan
    return float(np.average((rx - mx) * (ry - my), weights=w) / np.sqrt(vx * vy))


def bfdr_set(c: np.ndarray, w: np.ndarray, q: float = BFDR_Q) -> np.ndarray:
    """Boolean per pattern: largest set of whole tie blocks of c (decreasing) whose
    edge-weighted mean of (1 - c) is <= q.  Along the decreasing order the running
    mean of (1 - c) is non-decreasing, so the set is a prefix of tie blocks."""
    c = np.asarray(c, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    v = _tie_round(c)
    uniq, inv = np.unique(-v, return_inverse=True)       # block 0 = highest value
    inv = inv.reshape(-1)
    bw = np.bincount(inv, weights=w, minlength=len(uniq))
    bl = np.bincount(inv, weights=w * (1.0 - c), minlength=len(uniq))
    cw, cl = np.cumsum(bw), np.cumsum(bl)
    ok = (cw > 0) & (cl <= q * cw * (1 + 1e-12) + 1e-12)
    n_blocks = int(np.argmin(ok)) if not ok.all() else len(ok)
    return inv < n_blocks


def weighted_jaccard(a: np.ndarray, b: np.ndarray, w: np.ndarray) -> float:
    inter = float(np.sum(w[a & b]))
    union = float(np.sum(w[a | b]))
    return inter / union if union > 0 else np.nan


def top_n_edges(score_edge: np.ndarray, key_edge: np.ndarray, n: int) -> np.ndarray:
    """Sorted indices of the n edges with the highest score, ties broken by key (desc)."""
    s = _tie_round(score_edge)
    n = min(int(n), len(s))
    v = np.partition(s, len(s) - n)[len(s) - n]
    above = np.flatnonzero(s > v)
    tied = np.flatnonzero(s == v)
    need = n - len(above)
    pick = tied[np.argsort(-key_edge[tied], kind="stable")[:need]]
    return np.sort(np.concatenate([above, pick]))


def jaccard_idx(a: np.ndarray, b: np.ndarray) -> float:
    inter = len(np.intersect1d(a, b, assume_unique=True))
    return inter / (len(a) + len(b) - inter) if (len(a) + len(b)) else np.nan


def n_observed_above_unobserved(score: np.ndarray, obs: np.ndarray, w: np.ndarray) -> float:
    """Edges observed by >= 1 annotator whose score is strictly above the highest score of an
    unobserved edge (all observed edges if every edge is observed); patterns with edge counts w.
    A top-n set over all edges consists of observed edges, and equals the top-n set over the
    observed edges, iff n <= this number (verifier fitfix minor 1; fit_primary f1-M2 rule)."""
    s = _tie_round(score)
    obs = np.asarray(obs, dtype=bool)
    w = np.asarray(w, dtype=np.float64)
    unobs = ~obs & (w > 0)
    if not unobs.any():
        return float(w[obs].sum())
    return float(w[obs & (s > s[unobs].max())].sum())


def method_metrics(npred: np.ndarray, label: np.ndarray, n: np.ndarray) -> dict:
    """Posterior-weighted metrics of methods given predicted-edge counts per pattern.

    npred: (M, P) number of predicted edges of each method in each pattern; label (P,)
    in [0, 1]; n (P,) edges per pattern."""
    tp = npred @ label
    npr = npred.sum(1)
    csum = float(n @ label)
    with np.errstate(divide="ignore", invalid="ignore"):
        precision = np.where(npr > 0, tp / npr, np.nan)
        recall = tp / csum if csum > 0 else np.full_like(tp, np.nan)
        f1 = np.where(npr + csum > 0, 2 * tp / (npr + csum), np.nan)
    return dict(tp=tp, n_pred=npr, sum_label=csum, precision=precision, recall=recall, f1=f1)


def _sign(d):
    return np.where(np.abs(d) <= TIE_TOL, 0, np.sign(d)).astype(int)


def rank_disagreement(f: np.ndarray, g: np.ndarray) -> tuple:
    """(flips, comparable pairs) between two score vectors over the same methods (R1 Eq.)."""
    f, g = np.asarray(f, dtype=np.float64), np.asarray(g, dtype=np.float64)
    i, j = np.triu_indices(len(f), 1)
    sf, sg = _sign(f[i] - f[j]), _sign(g[i] - g[j])
    ok = np.isfinite(f[i] - f[j]) & np.isfinite(g[i] - g[j])
    comp = ok & (sf != 0) & (sg != 0)
    return int((comp & (sf != sg)).sum()), int(comp.sum())


def top_method(f: np.ndarray, names: list) -> str:
    f = np.asarray(f, dtype=np.float64)
    best = np.nanmax(f)
    return "|".join(n for n, v in zip(names, f) if abs(v - best) <= TIE_TOL)


def wilson(k: float, n: float, z: float = 1.959963984540054) -> tuple:
    if n <= 0:
        return (np.nan, np.nan)
    p = k / n
    den = 1 + z * z / n
    ctr = (p + z * z / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, ctr - half), min(1.0, ctr + half))


def boot_mean_ci(x, rng, B: int = N_BOOT) -> tuple:
    """Percentile bootstrap interval of the mean of x (resampling its elements)."""
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return (np.nan, np.nan)
    idx = rng.integers(0, len(x), size=(B, len(x)))
    m = x[idx].mean(1)
    return (float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975)))


def cluster_boot_mean_ci(x, clusters, rng, B: int = N_BOOT) -> tuple:
    """Percentile bootstrap interval of the mean of x resampling whole clusters (e.g. the same
    dropped source in the three tissues); each bootstrap mean pools the observations of the
    drawn clusters."""
    x = np.asarray(x, dtype=np.float64)
    cl = np.asarray(clusters)
    ok = np.isfinite(x)
    x, cl = x[ok], cl[ok]
    if len(x) == 0:
        return (np.nan, np.nan)
    codes, uniq = pd.factorize(cl)
    G = len(uniq)
    sums = np.bincount(codes, weights=x, minlength=G)
    cnts = np.bincount(codes, minlength=G).astype(np.float64)
    draw = np.stack([np.bincount(rng.integers(0, G, G), minlength=G) for _ in range(B)]).astype(np.float64)
    m = (draw @ sums) / (draw @ cnts)
    return (float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975)))


def _excludes_zero(lo, hi) -> bool:
    return bool(np.isfinite(lo) and np.isfinite(hi) and (lo > 0 or hi < 0))


# ----------------------------------------------------------------------------
# Method panel (tissue universes)
# ----------------------------------------------------------------------------
def tissue_edges(tissue: str) -> pd.DataFrame:
    """Universe edges in evidence order with the full-data pattern of each edge."""
    ev = pd.read_parquet(E.EV_DIR / tissue / "evidence.parquet", columns=["tf", "target"])
    ep = np.asarray(E.load_edge_pattern(tissue, VARIANT, mmap=False))
    if len(ep) != len(ev):
        raise ValueError(f"{tissue}: edge_pattern length {len(ep)} != evidence rows {len(ev)}")
    ev["pattern"] = ep
    return ev


def method_table(tissue: str, edges: pd.DataFrame) -> pd.DataFrame:
    """Universe edges joined (on tf, target) with every method score; missing = NaN."""
    sc = pd.read_parquet(C.OUT / "fm_scores" / tissue / "scores.parquet")
    need = [m for m in panel_for(tissue) + EXTRA_METHODS if m not in EXPRESSION_METHODS]
    df = edges.merge(sc[["tf", "target"] + need], on=["tf", "target"], how="left", validate="one_to_one")
    for m in EXPRESSION_METHODS:
        if m not in panel_for(tissue):
            continue
        g = pd.read_parquet(C.OUT / "grnboost2" / tissue / f"{m}.parquet").rename(columns={"importance": m})
        n_out = len(g.merge(edges[["tf", "target"]], on=["tf", "target"], how="left", indicator=True)
                    .query("_merge == 'left_only'"))
        if n_out:
            raise ValueError(f"{tissue} {m}: {n_out} links outside the universe")
        df = df.merge(g, on=["tf", "target"], how="left", validate="one_to_one")
    if len(df) != len(edges):
        raise ValueError("join changed the number of universe edges")
    return df


def predicted_masks(df: pd.DataFrame, methods: list, k: int = TOPK) -> dict:
    return {m: topk_per_tf(df, m, k).to_numpy() for m in methods}


def npred_by_pattern(masks: dict, methods: list, pattern: np.ndarray, P: int) -> np.ndarray:
    return np.stack([np.bincount(pattern[masks[m]], minlength=P) for m in methods]).astype(np.float64)


# ----------------------------------------------------------------------------
# Analysis
# ----------------------------------------------------------------------------
def _fit_rows(universe: str, est: str, cfg: Config, fit: dict | None) -> dict:
    if fit is None:
        return {}
    s = fit["summary"]
    n_rnd, n_best = random_at_best(fit["starts"])
    dropout = cfg.protocol != "full"
    n_seed = int(s.get("n_seedcheck_starts", 0))          # fresh starts of an adopted seed check (robfix)
    n_extra = max(0, n_rnd - n_seed - N_RANDOM_DROPOUT) if dropout else 0
    if n_extra and "search_log_post_gain" not in s:
        raise ValueError(f"{universe} {cfg.cid} {est}: stored fit used extra random starts but has no search "
                         "diagnostic; run 'robustness.py search'")
    return dict(fit_n_starts=s["n_starts"], fit_n_converged=s["n_converged"], fit_best_start=s["best_start"],
                fit_best_converged=s["best_converged"], fit_best_grad_inf=s["best_grad_inf"],
                fit_best_hess_min_eig=s["best_hess_min_eig"], fit_log_post=s["log_post"], fit_loglik=s["loglik"],
                fit_n_within_1e3=s["n_within_1e3"], fit_cred_spread_near_best=s["cred_spread_near_best"],
                fit_warm_gap=s["warm_gap"], fit_runtime_s=s["runtime_s"],
                fit_n_random_starts=n_rnd, fit_n_random_at_best=n_best,
                fit_global_search_ok=bool(n_best >= MIN_RANDOM_AT_BEST),
                # adaptive search (dropout fits): extra random starts beyond the initial 5 and the change
                # they made relative to the fit of the warm start + 5 random starts alone
                fit_n_extra_random_starts=n_extra,
                fit_search_log_post_gain=float(s.get("search_log_post_gain", 0.0)) if dropout else np.nan,
                fit_search_changed_best_mode=bool(s.get("search_changed_best_mode", False)),
                fit_search_max_abs_cred_change=(float(s.get("search_max_abs_cred_change", 0.0))
                                                if dropout else np.nan),
                fit_seedcheck_adopted=bool(s.get("seedcheck_adopted", False)))


def analyze_universe(universe: str, log=print) -> tuple:
    """Return (detail rows, method-score rows) for one universe."""
    data = lcm.PatternData(*E.load_patterns(universe, VARIANT))
    lineage = annotator_lineage(data.annotators)
    cfgs = configurations(data.annotators, data.family, lineage)
    n = data.n
    full_fit = load_fit(universe, "full")
    full_out = estimator_outputs(data, full_fit)
    ep_all = np.asarray(E.load_edge_pattern(universe, VARIANT, mmap=False))
    keys_all = np.random.default_rng(TIE_SEED).random(len(ep_all))
    valid = ep_all >= 0
    ep, keys = ep_all[valid], keys_all[valid]
    if not np.array_equal(np.bincount(ep, minlength=data.P), n.astype(np.int64)):
        raise ValueError(f"{universe}: edge_pattern counts do not match the pattern weights")
    n_edges = float(n.sum())

    is_tissue = universe in TISSUE_UNIVERSES
    method_rows = []
    if is_tissue:
        edges = tissue_edges(universe)
        df = method_table(universe, edges)
        panel = panel_for(universe)
        all_methods = panel + EXTRA_METHODS
        masks = predicted_masks(df, all_methods)
        npred_all = npred_by_pattern(masks, all_methods, edges["pattern"].to_numpy(), data.P)
        pidx = [all_methods.index(m) for m in panel]
        npred = npred_all[pidx]
        method_rows = method_scores_full(universe, data, edges, masks, all_methods, panel, full_out)

    full_metrics = {}
    full_top = {}
    full_bfdr = {}
    full_top_sm = {}
    for est, o in full_out.items():
        full_top[est] = top_n_edges(o["score"][ep], keys, TOP_N)
        full_bfdr[est] = bfdr_set(o["label"], n)
        if is_tissue:
            full_metrics[est] = method_metrics(npred, o["label"], n)
    # size-matched sets: n = |S|, the full-data Family-HB BFDR <= 0.05 set of this universe
    n_sm = int(round(float(n[full_bfdr[CONTRAST_REFERENCE]].sum())))
    for est, o in full_out.items():
        full_top_sm[est] = top_n_edges(o["score"][ep], keys, n_sm)
    # robfix (minor 1): observed edges of the full data, unobserved edges in the top sets
    obs_full = (data.R >= 0).any(1)
    obs_full_e = obs_full[ep]
    full_above = {est: n_observed_above_unobserved(o["score"], obs_full, n) for est, o in full_out.items()}

    def unobs_extra(top, top_sm, obs_e, above):
        return dict(n_obs_above_unobserved=above, n_unobserved_top=int((~obs_e[top]).sum()),
                    n_unobserved_top_sizematched=int((~obs_e[top_sm]).sum()))

    rows = []
    for est in ESTIMATORS:
        o = full_out[est]
        r = dict(universe=universe, estimator=est, estimator_type="lcm" if est in MODELS else "baseline",
                 protocol="full", config="full", dropped="", n_annotators=data.K, n_patterns=data.P,
                 w_T=o["w_T"], prevalence=float(n @ o["label"]) / n_edges,
                 n_bfdr=float(n[full_bfdr[est]].sum()), n_sizematched=n_sm, identified=True,
                 excluded_config=False)
        if is_tissue:
            fm = full_metrics[est]
            r.update(top_method_f1=top_method(fm["f1"], panel), top_method_precision=top_method(fm["precision"], panel))
        r.update(_fit_rows(universe, est, cfgs[0], full_fit.get(est)))
        r.update(unobs_extra(full_top[est], full_top_sm[est], obs_full_e, full_above[est]))
        rows.append(r)

    cache: dict = {}
    for cfg in cfgs[1:]:
        red, inv = lcm.drop_annotators(data, cfg.drop)
        if cfg.fit_key not in cache:
            cache[cfg.fit_key] = load_fit(universe, cfg.fit_key)
        fit = cache[cfg.fit_key]
        if fit["_meta"]["annotators"] != red.annotators or fit["_meta"]["n_patterns"] != red.P:
            raise ValueError(f"{universe} {cfg.cid}: stored fit does not match the reduced evidence")
        outs = estimator_outputs(red, fit)
        obs = (red.R >= 0).any(1)[inv]
        # Family-HB with one family left: T <-> B_f exchangeable -> not identified (metrics NA)
        fhb_exch = family_hb_exchangeable(red, fit["family_hb"]["class_names"])
        for est in ESTIMATORS:
            o_f, o_d = full_out[est], outs[est]
            identified = not (est == "family_hb" and fhb_exch)
            lab_d = o_d["label"][inv]
            sc_d = o_d["score"][inv]
            dlab = np.abs(lab_d - o_f["label"])
            top_d = top_n_edges(sc_d[ep], keys, TOP_N)
            top_sm_d = top_n_edges(sc_d[ep], keys, n_sm)
            b_d = bfdr_set(lab_d, n)
            r = dict(universe=universe, estimator=est, estimator_type="lcm" if est in MODELS else "baseline",
                     protocol=cfg.protocol, config=cfg.name, dropped=";".join(cfg.drop),
                     n_annotators=red.K, n_patterns=red.P, w_T=o_d["w_T"],
                     prevalence=float(n @ lab_d) / n_edges, n_bfdr=float(n[b_d].sum()),
                     n_sizematched=n_sm, identified=identified, excluded_config=cfg.cid in EXCLUDED_CONFIGS,
                     n_edges_obs=float(n[obs].sum()),
                     spearman=weighted_spearman(o_f["score"][obs], sc_d[obs], n[obs]),
                     jaccard_bfdr=weighted_jaccard(full_bfdr[est], b_d, n),
                     jaccard_top=jaccard_idx(full_top[est], top_d),
                     jaccard_top_sizematched=jaccard_idx(full_top_sm[est], top_sm_d),
                     max_abs_dlabel=float(dlab.max()),
                     mean_abs_dlabel_obs=float(np.average(dlab[obs], weights=n[obs])) if obs.any() else np.nan,
                     abs_dw_T=abs(o_d["w_T"] - o_f["w_T"]) if est in MODELS else np.nan,
                     abs_dprev=abs(float(n @ lab_d) - float(n @ o_f["label"])) / n_edges)
            # robfix (minor 1): top-2,000 Jaccard only where both sets consist of observed edges
            above_d = n_observed_above_unobserved(sc_d, obs, n)
            r.update(unobs_extra(top_d, top_sm_d, obs[ep], above_d),
                     jaccard_top_obs=(r["jaccard_top"] if min(above_d, full_above[est]) >= TOP_N else np.nan))
            if is_tissue:
                fm_f = full_metrics[est]
                fm_d = method_metrics(npred, lab_d, n)
                for metric in ("f1", "precision"):
                    fl, cp = rank_disagreement(fm_f[metric], fm_d[metric])
                    tm = top_method(fm_d[metric], panel)
                    r.update({f"flips_{metric}": fl, f"comparable_{metric}": cp,
                              f"D_{metric}": fl / cp if cp else np.nan,
                              f"top_method_{metric}": tm,
                              f"top_switch_{metric}": int(tm != top_method(fm_f[metric], panel))})
            if not identified:
                for k in NOT_IDENTIFIED_NA:
                    if k in r:
                        r[k] = None if k.startswith("top_method") else np.nan
            r.update(_fit_rows(universe, est, cfg, fit.get(est)))
            rows.append(r)
        log(f"[{universe}] analysed {cfg.cid}")
    return rows, method_rows


# metrics set to NA for a non-identified Family-HB fit
NOT_IDENTIFIED_NA = ("w_T", "prevalence", "n_bfdr", "spearman", "jaccard_bfdr", "jaccard_top",
                     "jaccard_top_sizematched", "max_abs_dlabel", "mean_abs_dlabel_obs", "abs_dw_T", "abs_dprev",
                     "flips_f1", "comparable_f1", "D_f1", "top_method_f1", "top_switch_f1",
                     "flips_precision", "comparable_precision", "D_precision", "top_method_precision",
                     "top_switch_precision",
                     # robfix
                     "jaccard_top_obs", "n_obs_above_unobserved", "n_unobserved_top", "n_unobserved_top_sizematched")


def method_scores_full(tissue: str, data: lcm.PatternData, edges: pd.DataFrame, masks: dict, methods: list,
                       panel: list, full_out: dict) -> list:
    """FM benchmarking table: every method under every estimator's full-data labels,
    with TF-cluster bootstrap intervals (TFs resampled with replacement, paired across
    methods and estimators)."""
    tf_codes, tf_names = pd.factorize(edges["tf"], sort=True)
    T = len(tf_names)
    pattern = edges["pattern"].to_numpy()
    n_tf = np.bincount(tf_codes, minlength=T).astype(np.float64)
    rng = np.random.default_rng(SEED)
    Wb = np.stack([np.bincount(rng.integers(0, T, T), minlength=T) for _ in range(N_BOOT)]).astype(np.float64)
    NP = np.stack([np.bincount(tf_codes[masks[m]], minlength=T) for m in methods]).astype(np.float64)   # (M, T)
    ntfs = (NP > 0).sum(1)
    NPb = Wb @ NP.T                                                                                    # (B, M)
    rows = []
    for est in ESTIMATORS:
        lab_e = full_out[est]["label"][pattern]
        C_tf = np.bincount(tf_codes, weights=lab_e, minlength=T)
        TP = np.stack([np.bincount(tf_codes[masks[m]], weights=lab_e[masks[m]], minlength=T) for m in methods])
        # expected TP of a TF-matched random predictor: same per-TF predicted counts
        E0 = NP * (C_tf / n_tf)[None, :]
        TPb, Cb, E0b = Wb @ TP.T, Wb @ C_tf, Wb @ E0.T
        with np.errstate(divide="ignore", invalid="ignore"):
            prec = TP.sum(1) / NP.sum(1)
            f1 = 2 * TP.sum(1) / (NP.sum(1) + C_tf.sum())
            lift = TP.sum(1) / E0.sum(1)
            precb = TPb / NPb
            f1b = 2 * TPb / (NPb + Cb[:, None])
            liftb = TPb / E0b
        prev = C_tf.sum() / n_tf.sum()
        pan_idx = [methods.index(m) for m in panel]
        rank_f1 = pd.Series(f1[pan_idx], index=panel).rank(ascending=False, method="min")
        rank_pr = pd.Series(prec[pan_idx], index=panel).rank(ascending=False, method="min")

        def ci(a):
            a = a[np.isfinite(a)]
            return (float(np.quantile(a, 0.025)), float(np.quantile(a, 0.975))) if len(a) else (np.nan, np.nan)

        for i, m in enumerate(methods):
            model = next((mm for mm in OWN_UNIFORM if m.startswith(f"{mm}_attn")), None)
            own_u = OWN_UNIFORM.get(model) if model else None
            r = dict(tissue=tissue, estimator=est, estimator_type="lcm" if est in MODELS else "baseline",
                     method=m, in_ranking_panel=m in panel, n_pred=int(NP[i].sum()), n_tfs_pred=int(ntfs[i]),
                     exp_tp=float(TP[i].sum()), prevalence=float(prev),
                     precision=float(prec[i]), precision_lo=ci(precb[:, i])[0], precision_hi=ci(precb[:, i])[1],
                     recall=float(TP[i].sum() / C_tf.sum()) if C_tf.sum() > 0 else np.nan,
                     f1=float(f1[i]), f1_lo=ci(f1b[:, i])[0], f1_hi=ci(f1b[:, i])[1],
                     rank_f1=float(rank_f1[m]) if m in panel else np.nan,
                     rank_precision=float(rank_pr[m]) if m in panel else np.nan,
                     precision_tf_matched_null=float(E0[i].sum() / NP[i].sum()) if NP[i].sum() else np.nan,
                     lift_tf_matched=float(lift[i]), lift_tf_matched_lo=ci(liftb[:, i])[0],
                     lift_tf_matched_hi=ci(liftb[:, i])[1], own_uniform=own_u or "")
            for ref_name, ref in (("random", "random"), ("own_uniform", own_u)):
                for metric, val, valb in (("f1", f1, f1b), ("precision", prec, precb)):
                    if ref is None or ref == m:
                        d, lo, hi = np.nan, np.nan, np.nan
                    else:
                        j = methods.index(ref)
                        d = float(val[i] - val[j])
                        lo, hi = ci(valb[:, i] - valb[:, j])
                    r[f"delta_{metric}_vs_{ref_name}"] = d
                    r[f"delta_{metric}_vs_{ref_name}_lo"] = lo
                    r[f"delta_{metric}_vs_{ref_name}_hi"] = hi
            rows.append(r)
    return rows


# ----------------------------------------------------------------------------
# Posterior-mode switches and other robfix detail columns (post hoc)
# ----------------------------------------------------------------------------
ROBFIX_DETAIL_COLUMNS = ["jaccard_top_obs", "n_obs_above_unobserved", "n_unobserved_top",
                         "n_unobserved_top_sizematched", "fit_basin_fraction", "fit_seedcheck_adopted", "mode_switch",
                         "fhb_mode_switch", "values_changed_post_hoc"]


def mode_switch_flags(detail: pd.DataFrame) -> pd.Series:
    """Per detail row: True if a latent-class DROPOUT fit's selected optimum is not the warm-started
    continuation of the full-data mode (stored diagnostics: ``fit_warm_gap > NEAR_BEST``, or no
    warm start); NA for full-data and baseline rows.  Raises if the stored selection is inconsistent
    with the rule (``fit_best_start == 'warm'`` iff the warm start reached the best mode)."""
    lc = (detail["estimator"].isin(MODELS) & (detail["protocol"] != "full")).to_numpy()
    gap = pd.to_numeric(detail["fit_warm_gap"], errors="coerce").to_numpy(dtype=float)
    warm_best = (detail["fit_best_start"].astype(str) == "warm").to_numpy()
    switch = ~(np.isfinite(gap) & (gap <= NEAR_BEST))
    bad = lc & (switch == warm_best)
    if bad.any():
        raise ValueError(f"{int(bad.sum())} dropout fits: fit_best_start inconsistent with fit_warm_gap "
                         "(run 'robustness.py reselect')")
    return pd.Series(np.where(lc, switch, None), index=detail.index, dtype=object)


def annotate_detail(detail: pd.DataFrame) -> pd.DataFrame:
    """Add the robfix columns (``ROBFIX_DETAIL_COLUMNS``, at the end) to the detail table:

    * ``fit_basin_fraction``  random starts at the best mode / random starts (latent-class rows);
    * ``mode_switch``         see ``mode_switch_flags``;
    * ``fhb_mode_switch``     Family-HB has a mode switch in this universe x configuration (every row);
    * ``values_changed_post_hoc`` the point values of a prespecified row changed after the first pass:
      latent-class fits whose best mode the adaptive search changed (fit_fixes) or that adopted the optimum
      of the independent-seed check (robfix), and non-identified rows.
    Idempotent."""
    d = detail.copy()
    lc = d["estimator"].isin(MODELS)
    with np.errstate(divide="ignore", invalid="ignore"):
        frac = (pd.to_numeric(d["fit_n_random_at_best"], errors="coerce")
                / pd.to_numeric(d["fit_n_random_starts"], errors="coerce"))
    d["fit_basin_fraction"] = frac.where(lc)
    d["mode_switch"] = mode_switch_flags(d)
    ref = d[(d["estimator"] == SWITCH_REFERENCE) & (d["protocol"] != "full")]
    sw = {(u, p, c) for u, p, c, s in zip(ref["universe"], ref["protocol"], ref["config"], ref["mode_switch"])
          if bool(s)}
    d["fhb_mode_switch"] = [(u, p, c) in sw for u, p, c in zip(d["universe"], d["protocol"], d["config"])]
    changed = (lc & (d["protocol"] != "full") & d["fit_search_changed_best_mode"].eq(True)).to_numpy()
    changed |= d["identified"].eq(False).to_numpy()
    if "fit_seedcheck_adopted" in d.columns:            # optimum adopted from the independent-seed check
        changed |= d["fit_seedcheck_adopted"].eq(True).to_numpy()
    d["values_changed_post_hoc"] = changed
    for c in ROBFIX_DETAIL_COLUMNS:
        if c not in d.columns:
            d[c] = np.nan
    return d[[c for c in d.columns if c not in ROBFIX_DETAIL_COLUMNS] + ROBFIX_DETAIL_COLUMNS]


def _ensure_annotated(detail: pd.DataFrame) -> pd.DataFrame:
    return detail if "fhb_mode_switch" in detail.columns else annotate_detail(detail)


def _cfg_labels(g: pd.DataFrame) -> list:
    """Sorted unique ``universe:config`` labels of rows."""
    return sorted(set(g["universe"].astype(str) + ":" + g["config"].astype(str)))


def _switch_labels(g: pd.DataFrame) -> list:
    return _cfg_labels(g[g["fhb_mode_switch"].astype(bool)])


def _subsets(g: pd.DataFrame, prot: str) -> list:
    """(subset name, rows) of a protocol: 'all' and, for family / lineage dropout,
    'excl_uninformative' (without EXCLUDED_CONFIGS)."""
    out = [("all", g)]
    if prot in ("family", "lineage"):
        cid = g["protocol"] + ":" + g["config"]
        out.append(("excl_uninformative", g[~cid.isin(EXCLUDED_CONFIGS)]))
    return out


def _switch_subsets(g: pd.DataFrame, prot: str) -> list:
    """(subset name, rows) of the post hoc subsets without Family-HB mode switches: 'no_fhb_mode_switch'
    (parent 'all') and, for family / lineage dropout, 'excl_uninformative_no_fhb_mode_switch' (parent
    'excl_uninformative')."""
    keep = ~g["fhb_mode_switch"].astype(bool)
    parents = dict(_subsets(g, prot))
    out = [(NO_SWITCH, g[keep])]
    if prot in ("family", "lineage"):
        par = parents["excl_uninformative"]
        out.append((NO_SWITCH_EXCL, par[~par["fhb_mode_switch"].astype(bool)]))
    return out


SWITCH_PARENT = {NO_SWITCH: "all", NO_SWITCH_EXCL: "excl_uninformative"}


def _switch_columns(g_all: pd.DataFrame, g: pd.DataFrame, prot: str, subset: str) -> dict:
    """``n_fhb_mode_switch`` (configurations of the row with a Family-HB switch) and, for the restricted
    subsets, the configurations of the parent subset that were removed.  ``g_all``: all rows of the
    protocol (same estimator / scope), ``g``: the rows of the subset."""
    out = dict(n_fhb_mode_switch=len(_switch_labels(g)), n_fhb_mode_switch_excluded=0, fhb_mode_switch_excluded="")
    if subset in SWITCH_SUBSETS:
        par = dict(_subsets(g_all, prot))[SWITCH_PARENT[subset]]
        ex = _switch_labels(par)
        out.update(n_fhb_mode_switch_excluded=len(ex), fhb_mode_switch_excluded=";".join(ex))
    return out


def _ci_status(prot: str) -> str:
    return "inferential" if prot == "source" else "descriptive (3 configurations per universe)"


EDGE_METRICS = ("spearman", "jaccard_bfdr", "jaccard_top", "jaccard_top_sizematched")
POST_HOC_EDGE_METRICS = ("jaccard_top_obs",)            # robfix (verifier fitfix minor 1)


def _ranking_rows(detail: pd.DataFrame, rng, subsets=_subsets, n_boot: int = N_BOOT) -> list:
    """Rank-disagreement summary rows per tissue (and pooled) x estimator x protocol x subset x metric."""
    drop = detail[detail.protocol != "full"]
    rank_rows = []
    tis = drop[drop.universe.isin(TISSUE_UNIVERSES)]
    groups = [(t, tis[tis.universe == t]) for t in TISSUE_UNIVERSES] + [("all_tissues", tis)]
    for t, g0 in groups:
        for est in ESTIMATORS:
            for prot in PROTOCOLS:
                g1 = g0[(g0.estimator == est) & (g0.protocol == prot)]
                if g1.empty:
                    continue
                for subset, g in subsets(g1, prot):
                    if g.empty:
                        continue
                    for metric in ("f1", "precision"):
                        D = g[f"D_{metric}"].to_numpy(dtype=float)
                        fl, cp = g[f"flips_{metric}"].sum(), g[f"comparable_{metric}"].sum()
                        lo, hi = boot_mean_ci(D, rng, B=n_boot)
                        clo, chi = cluster_boot_mean_ci(D, g["config"].to_numpy(), rng, B=n_boot)
                        wl, wh = wilson(fl, cp)
                        full_tops = detail[(detail.protocol == "full") & (detail.estimator == est)
                                           & detail.universe.isin([t] if t != "all_tissues" else TISSUE_UNIVERSES)]
                        rank_rows.append(dict(
                            tissue=t, estimator=est, estimator_type="lcm" if est in MODELS else "baseline",
                            protocol=prot, subset=subset, metric=metric, n_configs=len(g),
                            n_configs_identified=int(np.isfinite(D).sum()),
                            mean_D=float(np.nanmean(D)) if np.isfinite(D).any() else np.nan,
                            mean_D_boot_lo=lo, mean_D_boot_hi=hi, mean_D_cluster_lo=clo, mean_D_cluster_hi=chi,
                            ci_status=_ci_status(prot),
                            max_D=float(np.nanmax(D)) if np.isfinite(D).any() else np.nan,
                            n_configs_any_flip=int((g[f"flips_{metric}"] > 0).sum()),
                            pooled_flips=int(fl), pooled_comparable=int(cp),
                            pooled_rate=fl / cp if cp else np.nan, pooled_wilson_lo=wl, pooled_wilson_hi=wh,
                            n_top_switches=int(np.nansum(g[f"top_switch_{metric}"].to_numpy(dtype=float))),
                            full_top_method=";".join(full_tops[f"top_method_{metric}"].astype(str)),
                            # robfix columns
                            n_boot=n_boot, **_switch_columns(g1, g, prot, subset),
                            values_changed_post_hoc=bool(g["values_changed_post_hoc"].astype(bool).any())))
    return rank_rows


def _edge_metric_summary(g: pd.DataFrame, name: str, rng, n_boot: int) -> dict:
    x = g[name].to_numpy(dtype=float)
    lo, hi = boot_mean_ci(x, rng, B=n_boot)
    return {f"{name}_mean": float(np.nanmean(x)) if np.isfinite(x).any() else np.nan,
            f"{name}_mean_boot_lo": lo, f"{name}_mean_boot_hi": hi,
            f"{name}_min": float(np.nanmin(x)) if np.isfinite(x).any() else np.nan}


def _edge_rows(detail: pd.DataFrame, streams: list, subsets=_subsets) -> list:
    """Edge-stability summary rows per universe x estimator x protocol x subset.  ``streams`` =
    [(metrics, rng, n_boot), ...]: the first stream's metrics are summarised in the first-pass column
    position, later streams' metrics (post hoc) after the other columns, each with its own generator."""
    drop = detail[detail.protocol != "full"]
    (metrics0, rng0, nb0), later = streams[0], streams[1:]
    edge_rows = []
    for u in UNIVERSES:
        for est in ESTIMATORS:
            full = detail[(detail.universe == u) & (detail.estimator == est) & (detail.protocol == "full")]
            for prot in PROTOCOLS:
                g1 = drop[(drop.universe == u) & (drop.estimator == est) & (drop.protocol == prot)]
                if g1.empty:
                    continue
                for subset, g in subsets(g1, prot):
                    if g.empty:
                        continue
                    r = dict(universe=u, estimator=est, estimator_type="lcm" if est in MODELS else "baseline",
                             protocol=prot, subset=subset, n_configs=len(g),
                             n_configs_identified=int(g["identified"].astype(bool).sum()),
                             ci_status=_ci_status(prot), n_bfdr_full=float(full.n_bfdr.iloc[0]),
                             n_sizematched=int(full.n_sizematched.iloc[0]),
                             prevalence_full=float(full.prevalence.iloc[0]))
                    for name in metrics0:
                        r.update(_edge_metric_summary(g, name, rng0, nb0))
                    fin = lambda c: g[c].to_numpy(dtype=float)  # noqa: E731
                    r.update(max_abs_dlabel_max=float(np.nanmax(fin("max_abs_dlabel"))) if np.isfinite(fin("max_abs_dlabel")).any() else np.nan,
                             mean_abs_dlabel_obs_mean=float(np.nanmean(fin("mean_abs_dlabel_obs"))) if np.isfinite(fin("mean_abs_dlabel_obs")).any() else np.nan,
                             abs_dw_T_mean=float(np.nanmean(fin("abs_dw_T"))) if est in MODELS and np.isfinite(fin("abs_dw_T")).any() else np.nan,
                             abs_dw_T_max=float(np.nanmax(fin("abs_dw_T"))) if est in MODELS and np.isfinite(fin("abs_dw_T")).any() else np.nan,
                             abs_dprev_mean=float(np.nanmean(fin("abs_dprev"))) if np.isfinite(fin("abs_dprev")).any() else np.nan,
                             abs_dprev_max=float(np.nanmax(fin("abs_dprev"))) if np.isfinite(fin("abs_dprev")).any() else np.nan)
                    # robfix columns
                    for metrics, rng, nb in later:
                        for name in metrics:
                            r.update(_edge_metric_summary(g, name, rng, nb))
                    r.update(jaccard_top_obs_n=int(np.isfinite(g["jaccard_top_obs"].to_numpy(dtype=float)).sum())
                             if "jaccard_top_obs" in g else 0,
                             n_boot=nb0, **_switch_columns(g1, g, prot, subset),
                             values_changed_post_hoc=bool(g["values_changed_post_hoc"].astype(bool).any()))
                    edge_rows.append(r)
    return edge_rows


def _convergence_rows(detail: pd.DataFrame) -> list:
    conv_rows = []
    lc = detail[detail.estimator_type == "lcm"]
    for u in UNIVERSES:
        for m in MODELS:
            for prot in ("full",) + PROTOCOLS:
                g = lc[(lc.universe == u) & (lc.estimator == m) & (lc.protocol == prot)]
                if g.empty:
                    continue
                sw = g[g["mode_switch"].eq(True)] if prot != "full" else g.iloc[:0]
                conv_rows.append(dict(
                    universe=u, model=m, protocol=prot, n_configs=len(g),
                    n_starts_per_config_min=int(g.fit_n_starts.min()), n_starts_per_config_max=int(g.fit_n_starts.max()),
                    n_configs_best_converged=int(g.fit_best_converged.astype(bool).sum()),
                    frac_starts_converged=float(g.fit_n_converged.sum() / g.fit_n_starts.sum()),
                    max_best_grad_inf=float(g.fit_best_grad_inf.max()),
                    min_best_hess_min_eig=float(g.fit_best_hess_min_eig.min()),
                    n_configs_warm_at_best=int((g.fit_warm_gap <= NEAR_BEST).sum()) if prot != "full" else np.nan,
                    max_warm_gap=float(g.fit_warm_gap.max()) if prot != "full" else np.nan,
                    min_n_within_1e3=int(g.fit_n_within_1e3.min()),
                    min_n_random_at_best=int(g.fit_n_random_at_best.min()),
                    n_configs_global_search_ok=int(g.fit_global_search_ok.astype(bool).sum()),
                    n_configs_extra_random_starts=int((g.fit_n_extra_random_starts > 0).sum()),
                    max_n_random_starts=int(g.fit_n_random_starts.max()),
                    n_configs_best_mode_changed_by_search=int(g.fit_search_changed_best_mode.astype(bool).sum()),
                    n_configs_not_identified=int((~g.identified.astype(bool)).sum()) if "identified" in g else 0,
                    n_configs_mode_ambiguous=int((g.fit_cred_spread_near_best > AMBIGUOUS_SPREAD).sum()),
                    max_cred_spread_near_best=float(g.fit_cred_spread_near_best.max()),
                    w_T_full=float(lc[(lc.universe == u) & (lc.estimator == m) & (lc.protocol == "full")].w_T.iloc[0]),
                    total_runtime_s=float(g.fit_runtime_s.sum()),
                    # robfix columns
                    n_configs_mode_switch=int(len(sw)) if prot != "full" else np.nan,
                    mode_switch_configs=";".join(f"{c} ({gap:.2f})" for c, gap in zip(sw["config"], sw["fit_warm_gap"])),
                    min_basin_fraction=float(g.fit_basin_fraction.min())))
    return conv_rows


def summarise(detail: pd.DataFrame) -> tuple:
    """(rob_method_ranking, rob_edge_stability, rob_fit_convergence) tables.

    The first-pass / fit_fixes rows use the shared generator ``default_rng(SEED)`` in the original
    order (ranking rows, then edge rows) with ``N_BOOT`` resamples; the post hoc robfix quantities
    (``POST_HOC_EDGE_METRICS``; the rows of ``SWITCH_SUBSETS`` with ``N_BOOT_POST_HOC`` resamples) use
    generators of their own (``RNG_STREAM``) and are appended, so that no earlier value changes."""
    detail = _ensure_annotated(detail)
    rng = np.random.default_rng(SEED)
    rng_post = np.random.default_rng([SEED, RNG_STREAM["edge_post_hoc_metrics"]])
    rng_sw = np.random.default_rng([SEED, RNG_STREAM["switch_summaries"]])
    rank_rows = _ranking_rows(detail, rng)
    edge_rows = _edge_rows(detail, [(EDGE_METRICS, rng, N_BOOT), (POST_HOC_EDGE_METRICS, rng_post, N_BOOT)])
    rank_rows += _ranking_rows(detail, rng_sw, subsets=_switch_subsets, n_boot=N_BOOT_POST_HOC)
    edge_rows += _edge_rows(detail, [(EDGE_METRICS + POST_HOC_EDGE_METRICS, rng_sw, N_BOOT_POST_HOC)],
                            subsets=_switch_subsets)
    return pd.DataFrame(rank_rows), pd.DataFrame(edge_rows), pd.DataFrame(_convergence_rows(detail))


SEARCH_COLS = ["universe", "protocol", "config", "estimator", "fit_n_starts", "fit_n_random_starts",
               "fit_n_random_at_best", "fit_global_search_ok", "fit_best_start", "fit_warm_gap", "fit_log_post",
               "fit_n_within_1e3", "fit_cred_spread_near_best", "fit_n_extra_random_starts",
               "fit_search_log_post_gain", "fit_search_changed_best_mode", "fit_search_max_abs_cred_change",
               "identified",
               # robfix (verifier fitfix minors 8 / 11, M1)
               "fit_basin_fraction", "mode_switch", "fit_seedcheck_adopted", "values_changed_post_hoc"]
SEEDCHECK_COLS = ["seedcheck_done", "seedcheck_n_starts", "seedcheck_best_log_post", "seedcheck_best_minus_stored",
                  "seedcheck_n_at_stored_mode", "seedcheck_n_better", "seedcheck_better_mode_found",
                  "seedcheck_max_abs_cred_diff_best", "seedcheck_adopted"]


def search_table(detail: pd.DataFrame, seedcheck: pd.DataFrame | None = None) -> pd.DataFrame:
    """Per universe x configuration x latent-class model: global-search diagnostic (random starts
    run, random starts at the best mode and their fraction, warm-start gap, mode switch, extra random
    starts of the adaptive search and what they changed relative to the warm start + 5 random starts
    alone: log-posterior gain, whether the best mode changed, largest credibility change) and, where
    stored, the independent-seed check (``seedcheck``; ``seedcheck_done`` False otherwise)."""
    detail = _ensure_annotated(detail)
    lc = detail[detail.estimator_type == "lcm"]
    out = lc.reindex(columns=SEARCH_COLS).rename(columns={"estimator": "model"}).reset_index(drop=True)
    if seedcheck is None:
        seedcheck = load_seedchecks()
    sc = seedcheck.reindex(columns=["universe", "protocol", "config", "model"] + SEEDCHECK_COLS[1:])
    out = out.merge(sc, on=["universe", "protocol", "config", "model"], how="left", validate="one_to_one")
    out.insert(len(SEARCH_COLS), "seedcheck_done", out["seedcheck_n_starts"].notna())
    return out


SWITCH_TABLE_COLS = ["universe", "protocol", "config", "model", "identified", "excluded_config",
                     "excluded_from_no_switch_subsets", "fit_best_start", "fit_warm_gap", "fit_log_post",
                     "fit_n_random_starts", "fit_n_random_at_best", "fit_basin_fraction", "fit_global_search_ok",
                     "fit_search_changed_best_mode", "w_T", "abs_dw_T", "D_f1", "D_precision", "spearman",
                     "jaccard_bfdr", "jaccard_top_sizematched", "max_abs_dlabel"]


def mode_switch_table(detail: pd.DataFrame) -> pd.DataFrame:
    """``rob_mode_switches.csv``: every latent-class dropout fit whose selected optimum is not the
    warm-started continuation of the full-data mode (``mode_switch``), with its search diagnostics
    and stability metrics; ``excluded_from_no_switch_subsets`` marks the Family-HB switches, whose
    configurations the ``SWITCH_SUBSETS`` rows remove for every estimator."""
    detail = _ensure_annotated(detail)
    sw = detail[detail["mode_switch"].eq(True)].copy()
    sw["excluded_from_no_switch_subsets"] = sw["estimator"] == SWITCH_REFERENCE
    sw = sw.rename(columns={"estimator": "model"})
    order = {p: i for i, p in enumerate(PROTOCOLS)}
    sw = sw.assign(_u=sw["universe"].map({u: i for i, u in enumerate(UNIVERSES)}), _p=sw["protocol"].map(order),
                   _m=sw["model"].map({m: i for i, m in enumerate(MODELS)}))
    sw = sw.sort_values(["_u", "_p", "config", "_m"], kind="mergesort")
    return sw.reindex(columns=SWITCH_TABLE_COLS).reset_index(drop=True)


# ----------------------------------------------------------------------------
# Independent-seed check of the fits that needed extra random starts (robfix, verifier fitfix minor 8)
# ----------------------------------------------------------------------------
def needs_seedcheck(info: dict) -> bool:
    """A stored dropout fit is checked if its adaptive search needed extra random starts (the initial
    warm + ``N_RANDOM_DROPOUT`` random starts did not put ``MIN_RANDOM_AT_BEST`` random starts at the
    best mode); this includes every fit that is not certified."""
    n_rnd, n_best = random_at_best(info["starts"])
    return n_rnd > N_RANDOM_DROPOUT or n_best < MIN_RANDOM_AT_BEST


def seedcheck_rng(universe: str, fit_key: str, model: str):
    tag = zlib.crc32(f"seedcheck|{universe}|{fit_key}|{model}".encode()) & 0x7FFFFFFF
    return jax.random.fold_in(jax.random.PRNGKey(SEEDCHECK_SEED), tag)


def run_seedcheck(universe: str, models=MODELS, n_starts: int = SEEDCHECK_N_STARTS, force: bool = False,
                  log=print) -> int:
    """``SEEDCHECK_N_STARTS`` fresh random starts (no warm start, no adaptive rule; keys from
    ``SEEDCHECK_SEED``, independent of the fits' keys) for every stored dropout fit with
    ``needs_seedcheck``.  Stored per universe in ``SEEDCHECK_DIR/<universe>.json`` (entries keyed
    ``fit_key|model``, kept unless ``force``); a fresh start 'beats' the stored optimum if its log
    posterior is more than ``NEAR_BEST`` above it.  Returns the number of fits checked now."""
    data = lcm.PatternData(*E.load_patterns(universe, VARIANT))
    cfgs = configurations(data.annotators, data.family, annotator_lineage(data.annotators))
    first = {}
    for cfg in cfgs[1:]:
        first.setdefault(cfg.fit_key, cfg)
    path = C.ensure_dir(SEEDCHECK_DIR) / f"{universe}.json"
    store = json.loads(path.read_text()) if path.exists() else {}
    n_done = 0
    for key, cfg in first.items():
        p = FIT_DIR / universe / key
        if not p.with_suffix(".json").exists():
            continue
        js = json.loads(p.with_suffix(".json").read_text())
        todo = [m for m, info in js["models"].items() if m in models and needs_seedcheck(info)
                and (force or f"{key}|{m}" not in store
                     or (store[f"{key}|{m}"]["n_better"] > 0 and not info["summary"].get("seedcheck_adopted")))]
        if not todo:
            continue
        fit = load_fit(universe, key)
        red, _ = lcm.drop_annotators(data, cfg.drop)
        for m in todo:
            t0 = time.time()
            new = fit_starts(red, m, seedcheck_rng(universe, key, m), n_starts)
            stored = float(fit[m]["summary"]["log_post"])
            lp = np.array([r["log_post"] for r in new["starts"]], dtype=float)
            lp = np.where(np.isfinite(lp), lp, -np.inf)
            best = float(lp.max())
            prev = fit[m]["summary"]
            e = dict(
                universe=universe, fit_key=key, configs=[c.cid for c in cfgs[1:] if c.fit_key == key], model=m,
                n_starts=int(n_starts), seed=SEEDCHECK_SEED, stored_log_post=stored, best_log_post=best,
                best_minus_stored=best - stored, n_at_stored_mode=int((np.abs(lp - stored) <= NEAR_BEST).sum()),
                n_better=int((lp > stored + NEAR_BEST).sum()),
                max_abs_cred_diff_best=float(np.max(np.abs(new["cred"] - fit[m]["cred"]))),
                fresh_log_posts=[float(x) for x in lp], runtime_s=time.time() - t0,
                adopted=bool(prev.get("seedcheck_adopted", False)),
                adopted_gain=prev.get("seedcheck_adopted_gain"))
            if e["n_better"] > 0:
                adopt_seedcheck(universe, key, fit, m, new, log=log)
                e.update(adopted=True, adopted_gain=best - stored)
                fit = load_fit(universe, key)
            store[f"{key}|{m}"] = e
            n_done += 1
            log(f"[{universe}] seedcheck {cfg.cid} {m}: stored {stored:.4f} fresh best {best:.4f} "
                f"({e['best_minus_stored']:+.4f}); {e['n_at_stored_mode']}/{n_starts} at the stored mode, "
                f"{e['n_better']} better{'; BETTER MODE ADOPTED' if e['n_better'] > 0 else ''}; {e['runtime_s']:.0f}s")
            C.write_json(store, path)
    return n_done


def adopt_seedcheck(universe: str, key: str, fit: dict, model: str, new: dict, log=print) -> None:
    """Make the best fresh start of an independent-seed check the selected optimum of a stored dropout fit
    when it is more than ``NEAR_BEST`` above the stored optimum (the MAP is the best optimum found; post hoc,
    robfix).  The fresh starts are appended to the start records as ``seedcheck<r>`` (counted as random
    starts by ``random_at_best``), the summary is recomputed in the fields of ``fit_starts`` and marked
    ``seedcheck_adopted``; the superseded files are copied to ``<universe>/_seedcheck_superseded/``."""
    import shutil
    d = FIT_DIR / universe
    bak = C.ensure_dir(d / "_seedcheck_superseded")
    for ext in (".json", ".npz"):
        if not (bak / f"{key}{ext}").exists():
            shutil.copy2((d / key).with_suffix(ext), bak / f"{key}{ext}")
    old = fit[model]
    s0 = old["summary"]
    fresh = [dict(r, start=str(r["start"]).replace("random", "seedcheck", 1)) for r in new["starts"]]
    starts = list(old["starts"]) + fresh
    lp = np.array([r["log_post"] for r in starts], dtype=float)
    lp_ok = np.where(np.isfinite(lp), lp, -np.inf)
    ib = int(np.argmax(lp_ok))
    rb = starts[ib]
    n_rnd, n_rnd_best = random_at_best(starts)
    warm = [r for r in starts if r["start"] == "warm"]
    d_old = float(np.max(np.abs(np.asarray(new["cred"]) - np.asarray(old["cred"]))))
    s = dict(s0)
    s.update(n_starts=len(starts), n_converged=int(sum(r["converged"] for r in starts)),
             n_lbfgs_success=int(sum(r["lbfgs_success"] for r in starts)), best_start=rb["start"],
             best_converged=bool(rb["converged"]), best_grad_inf=float(rb["grad_inf"]),
             best_hess_min_eig=float(rb["hess_min_eig"]), log_post=float(lp[ib]), loglik=float(rb["loglik"]),
             log_post_max=float(lp_ok[ib]), log_post_range=float(lp_ok.max() - lp_ok[np.isfinite(lp)].min()),
             n_within_1e3=int((lp_ok >= lp_ok[ib] - NEAR_BEST).sum()),
             cred_spread_near_best=float(new["summary"]["cred_spread_near_best"]),
             warm_gap=float(lp_ok[ib] - warm[0]["log_post"]) if warm else np.nan,
             n_random_starts=n_rnd, n_random_at_best=n_rnd_best, global_search_ok=bool(n_rnd_best >= MIN_RANDOM_AT_BEST),
             search_log_post_gain=float(lp[ib] - s0.get("initial_rule_log_post", s0["log_post"])),
             search_changed_best_mode=True,
             # vs the initial-rule fit: exact if the stored fit was the initial-rule fit, else a triangle bound
             search_max_abs_cred_change=d_old + (float(s0.get("search_max_abs_cred_change", 0.0))
                                                 if s0.get("search_changed_best_mode") else 0.0),
             w_T=float(np.asarray(new["w"])[0]), runtime_s=float(s0["runtime_s"]) + float(new["summary"]["runtime_s"]),
             seedcheck_adopted=True, seedcheck_adopted_gain=float(lp[ib] - s0["log_post"]),
             seedcheck_adopt_max_abs_cred_change=d_old, n_seedcheck_starts=len(fresh))
    fit[model] = dict(cred=new["cred"], w=new["w"], theta=new["theta"], values=new["values"], sites=old["sites"],
                      class_names=old["class_names"], summary=s, starts=starts)
    meta = fit.pop("_meta")
    extra = {k: v for k, v in meta.items() if k not in ("configs", "models")}
    _save_fit(d / key, meta["configs"], fit, extra)
    log(f"[{universe}] {key} {model}: independent-seed optimum adopted (+{s['seedcheck_adopted_gain']:.4f} log-posterior "
        f"units, max|dcred| {d_old:.3f}); superseded fit kept in {bak}")


def load_seedchecks(universes=UNIVERSES) -> pd.DataFrame:
    """Stored independent-seed checks, one row per universe x configuration x model."""
    rows = []
    for u in universes:
        path = SEEDCHECK_DIR / f"{u}.json"
        if not path.exists():
            continue
        for e in json.loads(path.read_text()).values():
            for cid in e["configs"]:
                prot, name = cid.split(":", 1)
                rows.append(dict(universe=u, protocol=prot, config=name, model=e["model"],
                                 seedcheck_n_starts=e["n_starts"], seedcheck_best_log_post=e["best_log_post"],
                                 seedcheck_best_minus_stored=e["best_minus_stored"],
                                 seedcheck_n_at_stored_mode=e["n_at_stored_mode"], seedcheck_n_better=e["n_better"],
                                 seedcheck_better_mode_found=bool(e["n_better"] > 0 or e.get("adopted", False)),
                                 seedcheck_max_abs_cred_diff_best=e["max_abs_cred_diff_best"],
                                 seedcheck_adopted=bool(e.get("adopted", False))))
    cols = ["universe", "protocol", "config", "model"] + SEEDCHECK_COLS[1:]
    return pd.DataFrame(rows, columns=cols)


CONTRAST_REFERENCE = "family_hb"       # the primary, prespecified estimator
CONTRAST_METRICS = {"D_f1": "lower", "D_precision": "lower", "spearman": "higher",
                    "jaccard_bfdr": "higher", "jaccard_top": "higher", "jaccard_top_sizematched": "higher"}
POST_HOC_CONTRAST_METRICS = {"jaccard_top_obs": "higher"}       # robfix (verifier fitfix minor 1)


def _contrast_rows(detail: pd.DataFrame, reference: str, subsets, streams: list) -> list:
    """Contrast rows; ``streams`` = [(metrics, rng, n_boot), ...] -- every stream's rows are collected
    separately and concatenated in stream order, each stream drawing from its own generator."""
    drop = detail[detail.protocol != "full"]
    scopes = [(u, drop[drop.universe == u]) for u in UNIVERSES]
    scopes.append(("all_tissues", drop[drop.universe.isin(TISSUE_UNIVERSES)]))
    out = [[] for _ in streams]
    for scope, g0 in scopes:
        for prot in PROTOCOLS:
            g2 = g0[g0.protocol == prot]
            for subset, g1 in subsets(g2, prot):
                ref = g1[g1.estimator == reference].set_index(["universe", "config"])
                for est in ESTIMATORS:
                    if est == reference:
                        continue
                    oth = g1[g1.estimator == est].set_index(["universe", "config"])
                    for k, (metrics, rng, n_boot) in enumerate(streams):
                        for metric, better in metrics.items():
                            if metric not in g1.columns:
                                continue
                            j = pd.concat([oth[metric].rename("e"), ref[metric].rename("r")], axis=1).dropna()
                            if j.empty:
                                continue
                            diff = (j.e - j.r).to_numpy(dtype=float)
                            lo, hi = boot_mean_ci(diff, rng, B=n_boot)
                            clusters = j.index.get_level_values("config").to_numpy()
                            clo, chi = cluster_boot_mean_ci(diff, clusters, rng, B=n_boot)
                            sign = -1 if better == "lower" else 1
                            pair_rows = pd.concat([oth.loc[j.index], ref.loc[j.index]])
                            sw = _switch_columns(g2[g2.estimator == reference], ref.loc[j.index].reset_index(),
                                                 prot, subset)
                            out[k].append(dict(
                                scope=scope, protocol=prot, subset=subset, estimator=est, reference=reference,
                                metric=metric, better=better, n_configs=len(j),
                                n_clusters=int(len(np.unique(clusters))),
                                ci_status=_ci_status(prot),
                                mean_estimator=float(j.e.mean()), mean_reference=float(j.r.mean()),
                                mean_diff=float(diff.mean()), mean_diff_boot_lo=lo, mean_diff_boot_hi=hi,
                                mean_diff_cluster_lo=clo, mean_diff_cluster_hi=chi,
                                significant=_excludes_zero(lo, hi),
                                significant_cluster=_excludes_zero(clo, chi),
                                estimator_more_stable_significant=bool(_excludes_zero(clo, chi) and sign * diff.mean() > 0),
                                n_estimator_more_stable=int((sign * diff > TIE_TOL).sum()),
                                n_reference_more_stable=int((sign * diff < -TIE_TOL).sum()),
                                n_equal=int((np.abs(diff) <= TIE_TOL).sum()),
                                # robfix columns
                                n_boot=n_boot, **sw,
                                values_changed_post_hoc=bool(pair_rows["values_changed_post_hoc"].astype(bool).any())))
    return [r for rows in out for r in rows]


def estimator_contrasts(detail: pd.DataFrame, reference: str = CONTRAST_REFERENCE) -> pd.DataFrame:
    """Paired contrasts of every estimator with ``reference`` over the same dropout
    configurations: mean difference (estimator - reference) with a paired percentile
    bootstrap interval over configurations (iid and clustered by configuration name, i.e. by
    source / family / lineage across tissues), significance flags (interval excludes 0) and the
    number of configurations in which each is more stable (lower D, higher Spearman / Jaccard).
    Scopes: each universe and ``all_tissues`` (configurations of the three tissue universes
    pooled); subsets as in ``summarise``.  Non-identified reference fits drop out of the pairs.

    Row blocks: (1) ``CONTRAST_METRICS`` x the subsets of ``_subsets`` with the shared generator
    ``default_rng(SEED)`` and ``N_BOOT`` resamples (first pass / fit_fixes); (2) post hoc
    ``POST_HOC_CONTRAST_METRICS`` for the same subsets (own generator, ``N_BOOT``); (3) post hoc,
    robfix: every metric for ``SWITCH_SUBSETS`` (configurations without a Family-HB mode switch;
    own generator, ``N_BOOT_POST_HOC`` resamples).  Intervals are unadjusted 95% intervals."""
    detail = _ensure_annotated(detail)
    rng = np.random.default_rng(SEED)
    rng_post = np.random.default_rng([SEED, RNG_STREAM["contrast_post_hoc_metrics"]])
    rng_sw = np.random.default_rng([SEED, RNG_STREAM["switch_contrasts"]])
    rows = _contrast_rows(detail, reference, _subsets, [(CONTRAST_METRICS, rng, N_BOOT),
                                                         (POST_HOC_CONTRAST_METRICS, rng_post, N_BOOT)])
    rows += _contrast_rows(detail, reference, _switch_subsets,
                           [({**CONTRAST_METRICS, **POST_HOC_CONTRAST_METRICS}, rng_sw, N_BOOT_POST_HOC)])
    return pd.DataFrame(rows)


def em_reference_contrasts(detail: pd.DataFrame) -> pd.DataFrame:
    """Post hoc (assets2, 2026-10-14): paired contrasts of every estimator with EM (``reference`` = 'em';
    mean difference estimator - EM), every metric, the subsets of ``_subsets`` and ``SWITCH_SUBSETS``,
    ``N_BOOT_POST_HOC`` resamples from the generators ``RNG_STREAM['em_reference_contrasts']`` and
    ``RNG_STREAM['em_reference_switch_contrasts']`` (no earlier row changes).  ``n_fhb_mode_switch`` and
    ``fhb_mode_switch_excluded`` still refer to Family-HB mode switches.  Unadjusted 95% intervals."""
    detail = _ensure_annotated(detail)
    metrics = {**CONTRAST_METRICS, **POST_HOC_CONTRAST_METRICS}
    rng_a = np.random.default_rng([SEED, RNG_STREAM["em_reference_contrasts"]])
    rng_b = np.random.default_rng([SEED, RNG_STREAM["em_reference_switch_contrasts"]])
    rows = _contrast_rows(detail, EM_REFERENCE, _subsets, [(metrics, rng_a, N_BOOT_POST_HOC)])
    rows += _contrast_rows(detail, EM_REFERENCE, _switch_subsets, [(metrics, rng_b, N_BOOT_POST_HOC)])
    return pd.DataFrame(rows)


def all_contrasts(detail: pd.DataFrame) -> pd.DataFrame:
    """``rob_estimator_contrasts.csv``: the Family-HB-referenced rows (``estimator_contrasts``) followed by
    the post hoc EM-referenced rows (``em_reference_contrasts``)."""
    a, b = estimator_contrasts(detail), em_reference_contrasts(detail)
    assert list(a.columns) == list(b.columns)
    return pd.concat([a, b], ignore_index=True)


# ----------------------------------------------------------------------------
# Analysis status (prespecified / post hoc) of table rows and columns
# ----------------------------------------------------------------------------
# This module reads no held-out data, but the project's first held-out outcomes were computed on
# 2026-10-08 from 16:34 UTC (tuning_optimism, heldout_eval, replication), i.e. AFTER the first pass of this module
# (analysis 14:32 UTC) and BEFORE the fit_fixes round (verifier rob).  Everything added or
# redefined in that round is labelled post hoc:
#   rows     ``analysis_status`` = 'post_hoc' for rows that exist only since that round (the
#            ``excl_uninformative`` subsets, the size-matched contrasts, ``rob_fit_search.csv``), else
#            'prespecified' (plan section 2.3 E1, or fixed label-free in the first pass);
#   columns  ``post_hoc_columns`` = ';'-joined columns added in that round (same string in every row;
#            'all' for a table that is new as a whole).
# ``FIRST_PASS_COLUMNS`` are the headers of the first-pass tables (kept in
# outputs/r2/robustness/_superseded_firstpass_tables); ``RENAMED_COLUMNS`` are pure renames.
STATUS_COLUMNS = ("analysis_status", "post_hoc_columns")
FIRST_PASS_COLUMNS = {
    "rob_config_detail.csv": (
        "universe estimator estimator_type protocol config dropped n_annotators n_patterns w_T prevalence n_bfdr "
        "top_method_f1 top_method_precision fit_n_starts fit_n_converged fit_best_start fit_best_converged "
        "fit_best_grad_inf fit_best_hess_min_eig fit_log_post fit_loglik fit_n_within_1e3 fit_cred_spread_near_best "
        "fit_warm_gap fit_runtime_s n_edges_obs spearman jaccard_bfdr jaccard_top max_abs_dlabel "
        "mean_abs_dlabel_obs abs_dw_T abs_dprev flips_f1 comparable_f1 D_f1 top_switch_f1 flips_precision "
        "comparable_precision D_precision top_switch_precision label_swapped").split(),
    "rob_edge_stability.csv": (
        "universe estimator estimator_type protocol n_configs n_bfdr_full prevalence_full spearman_mean "
        "spearman_mean_boot_lo spearman_mean_boot_hi spearman_min jaccard_bfdr_mean jaccard_bfdr_mean_boot_lo "
        "jaccard_bfdr_mean_boot_hi jaccard_bfdr_min jaccard_top_mean jaccard_top_mean_boot_lo "
        "jaccard_top_mean_boot_hi jaccard_top_min max_abs_dlabel_max mean_abs_dlabel_obs_mean abs_dw_T_mean "
        "abs_dw_T_max abs_dprev_mean abs_dprev_max").split(),
    "rob_estimator_contrasts.csv": (
        "scope protocol estimator reference metric better n_configs mean_estimator mean_reference mean_diff "
        "mean_diff_boot_lo mean_diff_boot_hi n_estimator_more_stable n_reference_more_stable n_equal").split(),
    "rob_fit_convergence.csv": (
        "universe model protocol n_configs n_starts_per_config n_configs_best_converged frac_starts_converged "
        "max_best_grad_inf min_best_hess_min_eig n_configs_warm_at_best max_warm_gap min_n_within_1e3 "
        "n_configs_mode_ambiguous max_cred_spread_near_best w_T_full total_runtime_s").split(),
    "rob_method_ranking.csv": (
        "tissue estimator estimator_type protocol metric n_configs mean_D mean_D_boot_lo mean_D_boot_hi max_D "
        "n_configs_any_flip pooled_flips pooled_comparable pooled_rate pooled_wilson_lo pooled_wilson_hi "
        "n_top_switches full_top_method").split(),
    "rob_method_scores_full.csv": None,      # unchanged since the first pass: no post hoc column
    "rob_fit_search.csv": (),                # new table (verifier M1): every row and column post hoc
    "rob_mode_switches.csv": (),             # new table (robfix, verifier fitfix M1): all post hoc
}
RENAMED_COLUMNS = {
    "rob_fit_convergence.csv": {"n_starts_per_config_min": "n_starts_per_config",
                                "n_starts_per_config_max": "n_starts_per_config"},
}
POST_HOC_SUBSETS = ("excl_uninformative",) + SWITCH_SUBSETS
POST_HOC_METRICS = ("jaccard_top_sizematched",) + tuple(POST_HOC_CONTRAST_METRICS)
ANALYSIS_DECISIONS = [
    dict(id="rob_first_pass_design", date="2026-10-08", status="module_design",
         item="integrator-lineage dropout as a third protocol; BFDR <= 0.05 set Jaccard and top-2,000 Jaccard next "
              "to Spearman; Fisher / Stouffer with mid-p per-annotator p values (plan section 6: p = rate for a "
              "report and 1 for a negative, which gives Stouffer -inf) and nominal 0.05 per-edge labels; warm start "
              "+ 5 random starts per dropout configuration, warm start selected among equivalent modes; observation "
              "mask not recomputed after dropout; TF-cluster bootstrap of the method table",
         note="fixed label-free in the first pass (analysis 2026-10-08 14:32 UTC), before the project's first "
              "held-out evaluation output (16:34 UTC, tuning_optimism); not in PRESPECIFIED; rows carry "
              "analysis_status = prespecified"),
    dict(id="rob_adaptive_search", date="2026-10-08", status="post_hoc",
         item="adaptive global search of every dropout MAP fit: random starts added until >= 2 random starts reach "
              "the best mode or 100 random starts were run (cap raised from 25 on 2026-10-13 after six "
              "lineage-dropout fits of the global-burst models stopped at 25 uncertified); stored fits not "
              "meeting the rule refitted (2026-10-13); per-configuration diagnostic rob_fit_search.csv",
         note="verifier rob M1 (lung lineage:CollecTRI Family-HB had been a local mode); label-free"),
    dict(id="rob_size_matched", date="2026-10-08", status="post_hoc",
         item="size-matched top-|S| Jaccard for every estimator, |S| = edges in the full-data Family-HB BFDR <= 0.05 "
              "set of the universe",
         note="verifier rob M3 (BFDR sets of the 0/1 rules are 3-10x larger); label-free"),
    dict(id="rob_not_identified", date="2026-10-08", status="post_hoc",
         item="Family-HB with one modality family left (lineage:DoRothEA) reported as not identified (metrics NA) "
              "instead of label-aligned to the full fit; family / lineage summaries and contrasts also without "
              "family:LIT and lineage:DoRothEA (subset = excl_uninformative)",
         note="verifier rob minor 2; label-free"),
    dict(id="rob_cluster_bootstrap", date="2026-10-08", status="post_hoc",
         item="bootstrap intervals clustered by configuration name (same source / family / lineage across "
              "tissues), significance flags, ci_status = descriptive for family and lineage protocols; no "
              "multiplicity adjustment",
         note="verifier rob minor 3; label-free"),
    dict(id="rob_mode_switch_subsets", date="2026-10-14", status="post_hoc",
         item="Family-HB posterior-mode switch = the selected dropout optimum is not the warm-started continuation "
              "of the full-data mode (stored warm-start gap > 1e-3 log-posterior units); method-ranking, "
              "edge-stability and contrast rows repeated without the configurations with a Family-HB switch "
              "(subsets no_fhb_mode_switch and excl_uninformative_no_fhb_mode_switch) for every estimator, "
              "source-clustered bootstrap with 4,000 resamples from separate random streams; rob_mode_switches.csv "
              "lists every switch of every latent-class model",
         note="robfix, verifier fitfix M1 (two lung source-dropout switches drove the Family-HB vs EM gap); "
              "label-free; the restriction conditions on an outcome of the Family-HB fit, so these rows are a "
              "decomposition, not a replacement of the prespecified rows"),
    dict(id="rob_jaccard_top_observed", date="2026-10-14", status="post_hoc",
         item="jaccard_top_obs: the top-2,000 Jaccard only where both sets consist of observed edges (NA when "
              "2,000 exceeds the observed edges strictly above the highest unobserved score in the full or the "
              "dropout configuration); counts of unobserved edges in the top-2,000 and size-matched sets",
         note="robfix, verifier fitfix minor 1 (same rule as fit_primary f1-M2); the first-pass jaccard_top is "
              "kept unchanged but is dominated by the seeded unobserved tie block in the tissue universes"),
    dict(id="rob_seedcheck", date="2026-10-14", status="post_hoc",
         item="fit_basin_fraction (random starts at the best mode / random starts) and an independent-seed check: "
              "30 fresh random starts (key 20261014) for every dropout fit whose adaptive search needed extra "
              "random starts; a fresh start more than 1e-3 above the stored optimum flags a missed mode, whose "
              "optimum is then adopted as the fit (the MAP is the best optimum found); superseded fit kept",
         note="robfix, verifier fitfix minor 8; label-free; 39 fits checked, one missed mode found and adopted "
              "(lung lineage:DoRothEA multifactor, +52.0 log-posterior units)"),
    dict(id="rob_values_changed_flag", date="2026-10-14", status="post_hoc",
         item="values_changed_post_hoc: prespecified rows whose point values changed in the fit_fixes round "
              "(fits whose best mode the adaptive search changed, non-identified Family-HB rows, and every "
              "summary / contrast row aggregating them); n_boot column",
         note="robfix, verifier fitfix minor 11; bookkeeping only"),
    dict(id="rob_values_changed_flag_seedcheck", date="2026-10-14", status="post_hoc",
         item="values_changed_post_hoc also flags the dropout fit adopted from the independent-seed check (lung "
              "lineage:DoRothEA multifactor) and every summary / contrast row aggregating it; for that fit the "
              "seedcheck_* columns of rob_fit_search.csv describe the check against the superseded optimum",
         note="assets2, verifier robfix m6/m7; clarifies rob_values_changed_flag and rob_seedcheck; bookkeeping only"),
    dict(id="rob_em_reference_contrasts", date="2026-10-14", status="post_hoc",
         item="paired contrasts of every estimator with EM (reference = 'em'; estimator - EM) for every metric, scope "
              "and protocol, subsets all / excl_uninformative and the two no-Family-HB-switch subsets; 4,000 bootstrap "
              "resamples, source-clustered in the pooled scope, from two separate random streams; appended to "
              "rob_estimator_contrasts.csv after the unchanged Family-HB-referenced rows",
         note="assets2, verifiers robfix M1(a) and assets_fix M4: statements that an estimator is more or less stable "
              "than EM need an EM-referenced interval; label-free; unadjusted 95% intervals; readers must filter on "
              "reference"),
]


def post_hoc_columns(name: str, columns) -> list:
    """Columns of table ``name`` added in the fit_fixes round (see ``FIRST_PASS_COLUMNS``)."""
    first = FIRST_PASS_COLUMNS[name]
    if first is None:
        return []
    renamed = RENAMED_COLUMNS.get(name, {})
    return [c for c in columns if c not in STATUS_COLUMNS and c not in first and c not in renamed]


def label_analysis_status(name: str, df: pd.DataFrame) -> pd.DataFrame:
    """Add ``analysis_status`` (row level: 'prespecified' / 'post_hoc') and ``post_hoc_columns``."""
    df = df.drop(columns=[c for c in STATUS_COLUMNS if c in df.columns]).copy()
    if df.empty:
        return df
    post = np.zeros(len(df), dtype=bool)
    if FIRST_PASS_COLUMNS[name] == ():
        post[:] = True
    if "subset" in df.columns:
        post |= df["subset"].isin(POST_HOC_SUBSETS).to_numpy()
    if name == "rob_estimator_contrasts.csv":
        post |= df["metric"].isin(POST_HOC_METRICS).to_numpy()
        if "reference" in df.columns:          # EM-referenced block (assets2): post hoc as a whole
            post |= (df["reference"] != CONTRAST_REFERENCE).to_numpy()
    cols = post_hoc_columns(name, df.columns)
    df["analysis_status"] = np.where(post, "post_hoc", "prespecified")
    # a table that did not exist in the first pass is post hoc as a whole
    df["post_hoc_columns"] = "all" if FIRST_PASS_COLUMNS[name] == () else ";".join(cols)
    return df


def record_decisions(module: str, decisions: list, path: Path | None = None) -> None:
    """Replace this module's rows of the shared dated decision log (``eval_analysis_decisions.csv``);
    other modules' rows are kept unchanged.  The read-modify-write runs under the exclusive ``fcntl`` lock
    that ``heldout_eval.record_decisions`` uses (``outputs/r2/locks/<name>.lock`` for the project table,
    ``<path>.lock`` otherwise) and the table is replaced atomically (verifier heldfix m6)."""
    import fcntl
    import tempfile
    path = Path(path or (C.TABLES / "eval_analysis_decisions.csv"))
    C.ensure_dir(path.parent)
    if path.resolve() == (C.TABLES / "eval_analysis_decisions.csv").resolve():
        lock = C.ensure_dir(C.OUT / "locks") / (path.name + ".lock")
    else:
        lock = path.with_name(path.name + ".lock")
    cols = ["module", "id", "date", "status", "item", "note"]
    with open(lock, "w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        try:
            new = pd.DataFrame(decisions).assign(module=module)
            if path.exists():
                old = pd.read_csv(path)
                new = pd.concat([old[old["module"] != module], new], ignore_index=True)
            fd, tmp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
            try:
                with os.fdopen(fd, "w") as fh:
                    new[cols].sort_values(["module", "id"], kind="mergesort").to_csv(fh, index=False)
                os.chmod(tmp, 0o644)
                os.replace(tmp, path)
            finally:
                if os.path.exists(tmp):
                    os.unlink(tmp)
        finally:
            fcntl.flock(lk, fcntl.LOCK_UN)


def run_analysis(universes=UNIVERSES, tables_dir: Path = C.TABLES, log=print, meta_path: Path | None = None) -> None:
    rows, mrows = [], []
    for u in universes:
        r, m = analyze_universe(u, log=log)
        rows += r
        mrows += m
    detail = annotate_detail(pd.DataFrame(rows))
    ranking, edge, conv = summarise(detail)
    tables_dir = C.ensure_dir(Path(tables_dir))
    tables = {"rob_config_detail.csv": detail, "rob_method_ranking.csv": ranking, "rob_edge_stability.csv": edge,
              "rob_fit_convergence.csv": conv, "rob_fit_search.csv": search_table(detail, load_seedchecks(universes)),
              "rob_method_scores_full.csv": pd.DataFrame(mrows),
              "rob_estimator_contrasts.csv": all_contrasts(detail),
              "rob_mode_switches.csv": mode_switch_table(detail)}
    for name, df in tables.items():
        label_analysis_status(name, df).to_csv(tables_dir / name, index=False)
    record_decisions("robustness", ANALYSIS_DECISIONS, tables_dir / "eval_analysis_decisions.csv")
    C.write_json(dict(variant=VARIANT, seed=SEED, n_bootstrap=N_BOOT, topk_per_tf=TOPK, bfdr_q=BFDR_Q,
                      top_n=TOP_N, tie_seed=TIE_SEED, n_restarts_full=N_RESTARTS_FULL,
                      n_random_dropout=N_RANDOM_DROPOUT, n_random_max=N_RANDOM_MAX,
                      min_random_at_best=MIN_RANDOM_AT_BEST, excluded_configs=list(EXCLUDED_CONFIGS),
                      grad_tol=GRAD_TOL, baseline_alpha=BASELINE_ALPHA,
                      panel=PANEL, extra_methods=EXTRA_METHODS, universes=list(universes),
                      n_boot_post_hoc=N_BOOT_POST_HOC, switch_reference=SWITCH_REFERENCE,
                      switch_subsets=list(SWITCH_SUBSETS), rng_streams=RNG_STREAM,
                      seedcheck_n_starts=SEEDCHECK_N_STARTS, seedcheck_seed=SEEDCHECK_SEED,
                      analysis_decisions=ANALYSIS_DECISIONS),
                 meta_path or (OUT_DIR / "analysis_meta.json"))
    log(f"wrote {len(detail)} detail rows, {len(ranking)} ranking rows, {len(edge)} edge rows")


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------
def _logger(path: Path):
    C.ensure_dir(path.parent)

    def log(msg):
        line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
        print(line, flush=True)
        with open(path, "a") as fh:
            fh.write(line + "\n")
    return log


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fit", help="MAP fits for the full data and every dropout configuration")
    f.add_argument("--universes", nargs="+", default=list(UNIVERSES), choices=list(UNIVERSES))
    f.add_argument("--force", action="store_true")
    f.add_argument("--reverse", action="store_true", help="walk the configurations backwards (second process)")
    f.add_argument("--no-reselect", action="store_true", help="skip the reselect step (see reselect_warm)")
    rs = sub.add_parser("reselect", help="apply the warm-start selection rule to stored fits (reselect_warm)")
    rs.add_argument("--universes", nargs="+", default=list(UNIVERSES), choices=list(UNIVERSES))
    se = sub.add_parser("search", help="complete the adaptive global search of stored fits (refit_incomplete_search)")
    se.add_argument("--universes", nargs="+", default=list(UNIVERSES), choices=list(UNIVERSES))
    sc = sub.add_parser("seedcheck", help="independent-seed check of the fits that needed extra random starts (post hoc)")
    sc.add_argument("--universes", nargs="+", default=list(UNIVERSES), choices=list(UNIVERSES))
    sc.add_argument("--force", action="store_true")
    a = sub.add_parser("analyze", help="metrics and manuscript tables from the stored fits")
    a.add_argument("--universes", nargs="+", default=list(UNIVERSES), choices=list(UNIVERSES))
    a.add_argument("--tables-dir", type=Path, default=C.TABLES)
    al = sub.add_parser("all", help="fit + analyze")
    al.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)
    if args.cmd in ("fit", "all"):
        for u in (args.universes if args.cmd == "fit" else UNIVERSES):
            run_fits(u, force=args.force, log=_logger(OUT_DIR / f"fit_{u}.log"),
                     reverse=getattr(args, "reverse", False), reselect=not getattr(args, "no_reselect", False))
    if args.cmd == "reselect":
        for u in args.universes:
            n = reselect_warm(u, log=_logger(OUT_DIR / f"fit_{u}.log"))
            print(f"{u}: {n} model fits reselected")
    if args.cmd == "search":
        for u in args.universes:
            n = refit_incomplete_search(u, log=_logger(OUT_DIR / f"fit_{u}.log"))
            print(f"{u}: {n} model fits refitted by the adaptive search")
    if args.cmd == "seedcheck":
        for u in args.universes:
            n = run_seedcheck(u, force=args.force, log=_logger(OUT_DIR / f"seedcheck_{u}.log"))
            print(f"{u}: {n} model fits checked with an independent seed")
    if args.cmd in ("analyze", "all"):
        tdir = Path(getattr(args, "tables_dir", C.TABLES))
        meta = None if tdir.resolve() == C.TABLES.resolve() else tdir / "analysis_meta.json"   # robfix m7
        run_analysis(args.universes if args.cmd == "analyze" else UNIVERSES,
                     tables_dir=tdir, log=_logger(OUT_DIR / "analyze.log"), meta_path=meta)


if __name__ == "__main__":
    main()
