"""Label-free claim governance (stage E4; reviewer Q7, Q9, Q13).  Replaces the R1 rank tiers,
the promotion gate and the ENCODE-tuned deconcentration.

Inputs (all label-free unless stated)
-------------------------------------
* Credibility: Family-HB fits of the primary evidence variant
  (``PRESPECIFIED['evidence_variant_primary']``, curated-conditional), FIT STORAGE CONTRACT
  ``outputs/r2/fits/<universe>/<variant>/<fit_id>/``.  Primary = ``family_hb_nuts`` posterior
  mean credibility (``cred_mean.npy``; 95% interval ``cred_lo/hi``); sensitivities = the MAP
  ``family_hb`` and the Dawid-Skene ``em`` MAP credibilities.
* Universes: ``U_ctx(t)`` = ``outputs/r2/evidence/<t>/evidence.parquet`` (tf, target, edge order)
  with ``evidence.load_edge_pattern``; the global universe U_glob (935 TFs x 9,963 genes,
  TF-major grid over ``tfs.txt`` x ``genes.txt``).
* Method predicted sets: ``fm_scores/<t>/scores.parquet`` and ``grnboost2/<t>/grnboost2.parquet``
  joined on (tf, target); ``predicted_sets.topk_per_tf`` (k = ``PRESPECIFIED['topk_per_tf']`` = 50,
  seeded random tie-break, NaN = not predicted).
* Integration baselines (alternative promotion rules): ``annotators/integration_baselines.parquet``
  (held-in DoRothEA level, held-in CollecTRI membership) joined on (tf, target).
* HELD-OUT data (evaluation only, never used for any tier, threshold or operating-point choice):
  ChIP references ``heldout_chip/{encode_tfbs,encode_strict,enets2}`` and the pooled perturbation
  label ``pert_pooled`` from ``cohorts/<cohort>/de.parquet``.

Exact definitions
-----------------
Global-fit credibility of a context edge.  Every edge of U_ctx(t) has the same evidence pattern
in U_ctx(t) and in U_glob (checked at run time); edges whose TF has no held-in literature
coverage are outside U_glob and are unobserved by every annotator, so their global-fit
credibility is that of the all-unobserved pattern of the global fit.  Hence both credibilities
(global fit c^g, context fit c^c) are functions of the context pattern, and every tier below is
computed on patterns weighted by their edge counts (identical to an edge-level computation).

Per-universe Bayesian-FDR tiers (within U_ctx(t), separately for c^g and c^c).  Tie blocks =
edges with equal credibility (rounded to ``TIE_DECIMALS``); blocks are ordered by decreasing
credibility; BFDR(prefix) = edge-weighted mean of (1 - c) over the prefix (non-decreasing along
the order).  Level 2 (Strong) = the largest prefix of whole blocks with BFDR <= q_strong; level 1
(Moderate) = the largest prefix with BFDR <= q_moderate minus level 2; level 0 (Below) otherwise.
q_strong / q_moderate = ``PRESPECIFIED['bfdr_strong' / 'bfdr_moderate']`` (0.05 / 0.20).

Dual tier (rule ``primary``, plan section 6) from the pair (l_g, l_c):
Strong = both 2; Moderate = both >= 1, not both 2; Conditional = >= 1 in exactly one;
Exploratory = both 0.  Promoted = Strong U Moderate.
The +-1 tier shifts (``lenient`` / ``strict``) move the dual-tier boundary by one step on the
lattice of (min, max) level pairs ordered (2,2) > (1,2) > (1,1) ~ (0,2) > (0,1) > (0,0):

==========  ======  =========  ============  ============
pair        primary lenient    strict
==========  ======  =========  ============  ============
(2,2)       Strong  Strong     Strong
(1,2)       Mod.    Strong     Moderate
(1,1)       Mod.    Moderate   Conditional
(0,2)       Cond.   Moderate   Conditional
(0,1)       Cond.   Cond.      Exploratory
(0,0)       Expl.   Expl.      Exploratory
==========  ======  =========  ============  ============

so promoted(strict) c promoted(primary) c promoted(lenient); ``lenient`` is the R1 code rule
(Strong in one universe and >= Moderate in the other is Strong).

Posterior uncertainty of tiers.  For each of the (thinned, 1000) stored NUTS draws of the global
and the context fit (draw d of one paired with draw d of the other; the fits are independent),
the plug-in credibilities c(theta_d) give plug-in tiers; ``gov_tiers.csv`` reports the 2.5/97.5%
quantiles of the tier counts and ``gov_promoted_edges.csv`` the fraction of draws in which an
edge is promoted (``p_promoted_draws``).

Held-out labels (evaluation only).
* ChIP reference r in {encode_tfbs, encode_strict, enets2}: an edge (tf, g) of the evaluation
  universe is evaluable iff tf is an assayed TF of r and g is in r's gene universe (tf != g);
  label = (tf, g) is an edge of r.  Expectations come from
  ``build_heldout_chip.reference_index(edges_r, universe = genes_r & eval_genes, assayed = all
  assayed TFs of r)``: TF-matched = the TF's positive rate over the evaluable genes of the
  evaluation universe (``base``); TF x target-degree matched = the prespecified additive
  logistic null logit P = a_TF + b_bin(leave-pair-out reference in-degree of g)
  (``null_probability``; ``PRESPECIFIED['target_degree_*']``).
* ``pert_pooled``: pairs (tf, g) with tf a Lambert TF perturbed in >= 1 held-out cohort and g
  tested for that perturbation in >= 1 cohort, excluding same-cytoband cis pairs
  (``cis_same_band``) from positives AND negatives; label = ``positive_primary`` in any cohort
  (primary DE of every cohort, i.e. replicate-stratified for Shifrut/Papalexi).  Evaluable = such
  a pair.  Expectations: TF-matched = the TF's pooled positive rate over its evaluable genes of
  the evaluation universe; TF x target-degree = the same additive logistic null fitted on the
  evaluable pairs only (``masked_degree_index``; degree = number of other perturbed TFs for
  which g is a pooled positive, bins by ``degree_bin_edges``).

Observed / expected (O/E) of an edge set S for reference r: O = positives among evaluable edges
of S; E_TF = sum of TF-matched rates; E_TFdeg = sum of null probabilities; O/E with a 95%
TF-cluster bootstrap interval (TFs with >= 1 evaluable edge in S resampled with replacement,
``PRESPECIFIED['n_bootstrap']`` draws, seed from ``PRESPECIFIED['random_seed']`` and the set's
membership, so identical sets reached by different rules get identical intervals; intervals
from < ``MIN_CLUSTERS`` = 10 TFs are flagged ``few_clusters`` as unstable); marginal
enrichment = (O / n_evaluable) / (pooled positive rate over all evaluable edges of the universe).

Stratified trend test across tiers: Mantel's extension test (stratified Cochran-Armitage) of
the label on the ordinal tier score (Exploratory 0 < Conditional 1 < Moderate 2 < Strong 3, or
Below 0 < Moderate 1 < Strong 2 for single-universe levels) over all evaluable edges, strata =
TF, or TF x target-degree bin; conditional on each stratum's positive count the statistic sum
x_i y_i is hypergeometric; z = (T - E)/sqrt(V); one-sided p for an increasing trend and two-sided
p; Holm adjustment within each (scope, strata) group across ALL universes (the three tissues and
the global universe) x the four references (12 tests for ``dual``, 16 for the single-universe
scopes), separately for the one-sided (``p_one_sided_holm``, increasing trend; the reported
test) and two-sided p (``holm_family`` column).

Concentration of an edge set: n edges, n TFs, top-1 / top-2 TF share, Shannon entropy H of the
TF distribution, Pielou evenness J = H / ln(n TFs), normalised entropy H / ln(n TFs of the
universe), and the five largest TFs.

Label-free deconcentration.  Pool = the promoted set of tissue t; joint score s = min(c^g, c^c).
For a budget B and penalty alpha the selected set is the greedy solution of repeatedly taking
the edge with the highest priority logit(s) - alpha * (#edges of its TF already selected); with a
penalty linear in the count this equals the top-B edges by logit(s_(TF, j)) - alpha * j, where j
is the edge's 0-based rank within its TF (ties broken by a seeded per-edge hash key, as in
``predicted_sets``).  (The R1 rule s / (1 + alpha * count) saturated because posteriors differed
by < 0.03; the log-odds penalty keeps alpha on the scale of the credibility differences.)
Grid: B = ``DECONC_BUDGET_FRACTIONS`` x |pool| (rounded), alpha in ``DECONC_ALPHAS``.
Precision axis = posterior-expected precision (mean of s over the selected set); diversity axis
= H / ln(n TFs in the pool) (a fixed scale per tissue, so small sets are not rewarded for
trivially even composition, the R1 J = H / ln(n_edges) bias).  Each axis is min-max normalised
over the grid; the operating point maximises their harmonic mean (ties: larger budget, then
smaller alpha).  No held-out data enter the choice; held-out O/E of the chosen set is then
compared with the undeconcentrated set of the same budget (alpha = 0) and with the whole pool.

Sensitivity grid (Q7): q_strong in {0.01, 0.02, 0.05, 0.10} x q_moderate in {0.10, 0.20, 0.30,
0.40} with q_strong < q_moderate; dual rules primary / lenient / strict; single-universe tiers
(global-fit only, context-fit only: promoted = level >= 1 in that universe); rank-budget rule
(level 2 / 1 = the largest prefix of whole tie blocks holding <= k1 / k2 edges, k1 = 500 and
k2 = 5,000 scaled from the R1 421,521-edge universe to |U_ctx(t)|); MAP instead of NUTS
credibility; the EM (Dawid-Skene) estimator; and promotion rules from baselines: majority vote
(>= 2 reporting annotators), union (>= 1), held-in DoRothEA level A/B
(``dorothea_level_heldin``), held-in CollecTRI membership (``in_collectri_heldin``).
For each: promoted count, edge Jaccard with the primary promoted set, TF-matched held-out O/E
(+ CI) per reference and concentration.

Global universe (supplementary, power): the global-fit BFDR levels over all of U_glob are
validated against the same four references (``universe = 'global'`` rows).

Outputs
-------
``outputs/r2/governance/``: ``<t>/edge_tiers.parquet`` (every U_ctx(t) edge: pattern,
credibilities, levels, dual tier, posterior promotion frequency, held-out labels),
``labels/<universe>__<ref>.parquet`` (evaluable pairs with label and expectations),
``meta.json`` (inputs, settings, checks, runtime), ``run.log``.

Manuscript tables (``data/r2/tables/gov_*.csv``)
-----------------------------------------------
* ``gov_tiers.csv``            tier counts per tissue x scope (global fit in U_ctx, context fit,
                               dual) x tier: edges, TFs, tie blocks, credibility range, realised
                               BFDR, posterior 95% interval of the count.
* ``gov_consistency.csv``      global-fit vs context-fit levels: exact / within-one agreement,
                               linear and quadratic weighted kappa, Spearman of credibilities,
                               3 x 3 cross-tabulation (NUTS and MAP; all edges and edges with
                               level >= 1 in either universe).
* ``gov_promoted_by_method.csv`` promoted edges over U_ctx(t) and within each method's top-k
                               predicted set: counts by dual tier, fraction promoted (TF-cluster
                               CI), TF-matched expected fraction, fraction with >= 1 report.
* ``gov_tier_validation.csv``  per universe x scope x tier x reference: O, E (TF, TF x degree),
                               O/E with TF-cluster CIs, marginal enrichment.
* ``gov_tier_trend.csv``       stratified Mantel trend tests across tiers (TF and TF x degree
                               strata) with Holm-adjusted p.
* ``gov_concentration.csv``    concentration of eligible / observed / reported / tier / promoted /
                               method-promoted / deconcentrated sets.
* ``gov_deconcentration.csv``  label-free operating-point grid (budget x alpha) with the chosen
                               point flagged.
* ``gov_deconcentration_eval.csv`` held-out O/E of the chosen set vs the undeconcentrated set of
                               the same budget and the whole pool.
* ``gov_sensitivity_grid.csv`` Q7 grid (configuration x reference).
* ``gov_promoted_edges.csv``   every promoted edge (primary rule): tier, credibilities with 95%
                               intervals, posterior promotion frequency, held-out labels where
                               evaluable, methods predicting it.

Revision 2026-10-08 (verifier ``gov``; all additions ``analysis_status = post_hoc``)
------------------------------------------------------------------------------------
* Minimum detectable O/E (``gov_tier_validation.csv``; Strong / Moderate / Promoted rows).  For a
  set S and reference r with per-TF margins (N_t evaluable edges, K_t positives of TF t, n_t
  edges of S that are evaluable) the TF-stratified exact conditional test of T = positives in S
  (null: X_t ~ Hypergeometric(N_t, K_t, n_t); H1: Fisher non-central hypergeometric with a common
  odds ratio psi) at one-sided alpha = ``PRESPECIFIED['replication_alpha']``: ``mde_oe`` = E_psi[T] /
  E_0[T] at the smallest psi with power >= ``PRESPECIFIED['replication_power']``; ``mde_oe_poisson``
  = the same from E_0 alone (T ~ Poisson); ``max_attainable_oe`` = sum min(n_t, K_t) / E_0;
  ``power_at_global_oe`` = power at the O/E of the same tier in the global universe (dual Strong /
  Moderate / Promoted vs global-fit Strong / Moderate / Strong U Moderate), and
  ``ci_excludes_global_oe`` = oe_tf_hi < that O/E.  Only margins enter (no S x label cross-tab);
  within-TF dependence and TF heterogeneity are ignored, so the MDE is optimistic.
* Size-matched sensitivity rows (``family = size_matched``): the top-|primary promoted| edges under
  each alternative rule's ranking (seeded tie-break), so O/E is compared at equal set size.
* ``gov_promoted_by_method.csv``: exact test of the promoted count in a method's top-k set against
  the TF-matched hypergeometric expectation (``p_depleted_exact`` / ``p_enriched_exact``), which is
  what "below the random control" has to mean.
* Deconcentration: the operating point under Pielou evenness H / ln(n TFs of the set) as the
  diversity axis (``chosen_pielou``; sensitivity, evaluated like the primary choice).
* ``gov_tiers.csv`` carries the NUTS settings / diagnostics of both fits (``*_nuts_*`` columns, read
  from the fits' ``meta.json``).  The first governance run (2026-10-08) used a global fit with 4 x
  500 + 500 draws; since 2026-10-13 all four universes use the frozen 4 x 1,000 + 1,000 fits of
  ``fit_primary.py`` and every table of this module was regenerated from them.
* ``top_tf_eval`` / ``top_tf_share_eval`` (``gov_tier_validation.csv``, ``gov_deconcentration_eval.csv``;
  added 2026-10-13): the TF holding most evaluable edges of the set and its share, because a
  context O/E can rest on one or two TFs (and, for ``pert_pooled``, on the cell lines in which
  those TFs were perturbed).
* Every table has ``analysis_status``; tables whose prespecified rows carry columns added after
  held-out results existed list them in ``post_hoc_columns``.

Settings not in ``PRESPECIFIED`` (deconcentration grid, score = min of the two credibilities,
logit penalty, harmonic mean, q grid, lenient / strict rules, ``MIN_CLUSTERS``, ``TIE_DECIMALS``) were
fixed in this module when it was written on 2026-10-08 (decision log ``ANALYSIS_DECISIONS``,
``data/r2/tables/eval_analysis_decisions.csv``); they use only credibilities, TF codes and seeded
keys, but other modules' held-out results existed in the project at that time.

Command line::

    .venv/bin/python subproject_24_multireference_latent_truth_model/src/r2/governance.py all
    ... governance.py all [--tissues immune kidney lung] [--no-global] [--n-boot 1000]

Deterministic: every random quantity (bootstrap, tie keys) is seeded from ``PRESPECIFIED``.
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "2")

import argparse  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import resource  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import zlib  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.special import expit, logit, logsumexp  # noqa: E402
from scipy.stats import norm, spearmanr  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from r2 import common as C  # noqa: E402
from r2 import evidence as E  # noqa: E402
from r2 import build_heldout_chip as HC  # noqa: E402
from r2.predicted_sets import tie_break_keys, topk_per_tf  # noqa: E402

try:  # shared analysis helpers (optional; this module is self-contained)
    from r2 import analysis_common as AC  # noqa: F401,E402
except ImportError:  # pragma: no cover
    AC = None

# ----------------------------------------------------------------------------
# Settings (all fixed before any held-out evaluation; held-out data are never used to choose)
# ----------------------------------------------------------------------------
P = C.PRESPECIFIED
VARIANT = P["evidence_variant_primary"]
FITS = C.OUT / "fits"
GOV = C.OUT / "governance"
SEED = int(P["random_seed"])
N_BOOT = int(P["n_bootstrap"])
Q_STRONG = float(P["bfdr_strong"])
Q_MODERATE = float(P["bfdr_moderate"])
TOPK = int(P["topk_per_tf"])
TOPK_SENS = [int(k) for k in P["topk_per_tf_sensitivity"]]
TIE_DECIMALS = 12
PRIMARY_FIT = "family_hb_nuts"
MAP_FIT = "family_hb"
EM_FIT = "em"
CHIP_REFS = ("encode_tfbs", "encode_strict", "enets2")
REFS = CHIP_REFS + ("pert_pooled",)
REF_LABEL = {"encode_tfbs": "encode", "encode_strict": "encode_strict", "enets2": "enets2",
             "pert_pooled": "pert_pooled"}
COHORTS = ("dixit_d13", "dixit_d7", "adamson", "shifrut", "norman", "papalexi", "replogle")
METHODS = ("scgpt_attn", "geneformer_attn", "scgpt_attn_norm", "geneformer_attn_norm",
           "grnboost2", "pearson_abs")
CONTROL_METHODS = ("random",)
DUAL_NAMES = {3: "Strong", 2: "Moderate", 1: "Conditional", 0: "Exploratory"}
LEVEL_NAMES = {2: "Strong", 1: "Moderate", 0: "Below"}
# dual-tier look-up tables indexed [l_global, l_context] (levels 0/1/2)
DUAL_LUT = {
    "primary": np.array([[0, 1, 1], [1, 2, 2], [1, 2, 3]], dtype=np.int8),
    "lenient": np.array([[0, 1, 2], [1, 2, 3], [2, 3, 3]], dtype=np.int8),
    "strict": np.array([[0, 0, 1], [0, 1, 2], [1, 2, 3]], dtype=np.int8),
}
Q_GRID_STRONG = (0.01, 0.02, 0.05, 0.10)
Q_GRID_MODERATE = (0.10, 0.20, 0.30, 0.40)
R1_UNIVERSE = 421_521            # R1 'full' universe size (scoping report)
R1_K1, R1_K2 = 500, 5000
DECONC_BUDGET_FRACTIONS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
DECONC_ALPHAS = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0)
LOGIT_CLIP = 1e-6
MIN_CLUSTERS = 10                # TF-cluster bootstrap intervals with < 10 TFs are flagged (unstable)
TOP_TFS_LISTED = 5
MDE_ALPHA = float(P["replication_alpha"])
MDE_POWER = float(P["replication_power"])
MDE_TIERS = ("Strong", "Moderate", "Promoted")
GLOBAL_TIER_FOR = {"Strong": "Strong", "Moderate": "Moderate", "Promoted": "Promoted"}
ANALYSIS_DECISIONS = [
    dict(id="gov_settings", date="2026-10-08", status="module_design",
         item="deconcentration grid (budget fractions 0.1-1.0 x alpha 0-16), score = min(c_g, c_c), logit penalty, "
              "harmonic mean of min-max normalised axes, diversity H / ln(n TFs in pool); q grid; lenient / strict "
              "rules; MIN_CLUSTERS = 10; TIE_DECIMALS = 12; rank budget scaled from R1",
         note="fixed when governance.py was written (2026-10-08); label-free, but other modules' held-out results "
              "existed in the project at that time; not in PRESPECIFIED"),
    dict(id="gov_mde", date="2026-10-08", status="post_hoc",
         item="minimum detectable O/E, attainable maximum and power at the global-universe O/E (tier validation)",
         note="verifier gov M1; margins only"),
    dict(id="gov_size_matched", date="2026-10-08", status="post_hoc",
         item="size-matched sensitivity rows (top-|primary| edges under each alternative rule)", note="verifier minor 5"),
    dict(id="gov_method_exact", date="2026-10-08", status="post_hoc",
         item="exact hypergeometric test of promoted counts in method top-k sets", note="verifier minor 5"),
    dict(id="gov_pielou", date="2026-10-08", status="post_hoc",
         item="deconcentration operating point with Pielou evenness as the diversity axis", note="verifier minor 6"),
    dict(id="gov_global_promoted", date="2026-10-08", status="post_hoc",
         item="global-universe Promoted (Strong U Moderate) validation rows", note="comparator for the power analysis"),
    dict(id="gov_top_tf_eval", date="2026-10-13", status="post_hoc",
         item="top_tf_eval / top_tf_share_eval of every validated set (TF concentration of the evaluable edges)",
         note="verifier gov M1 (the context perturbation O/E rests on few TFs); descriptive, label-free columns"),
    dict(id="gov_rerun_final_fits", date="2026-10-13", status="module_design",
         item="all gov_* tables regenerated from the frozen 4 x 1000/1000 family_hb_nuts fits of the four universes "
              "(fit_primary re-runs: immune 2026-10-08, kidney / lung / global 2026-10-13)",
         note="no setting of this module changed; tier counts move by at most a few boundary edges"),
]
POST_HOC_COLUMNS = {
    "gov_tier_validation.csv": "mde_oe;mde_odds_ratio;mde_oe_poisson;max_attainable_oe;n_tfs_in_power;global_universe_oe;"
                               "power_at_global_oe;power_note;ci_excludes_global_oe;top_tf_eval;top_tf_share_eval",
    "gov_promoted_by_method.csv": "expected_promoted_tf;p_depleted_exact;p_enriched_exact;ci_excludes_tf_expected",
    "gov_deconcentration.csv": "pielou_norm;harmonic_mean_pielou;chosen_pielou",
    "gov_deconcentration_eval.csv": "top_tf_eval;top_tf_share_eval",
    "gov_tier_trend.csv": "",
    "gov_tiers.csv": "",
    "gov_consistency.csv": "",
    "gov_concentration.csv": "",
    "gov_sensitivity_grid.csv": "",
    "gov_promoted_edges.csv": "",
}


def log(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    try:
        C.ensure_dir(GOV)
        with open(GOV / "run.log", "a") as fh:
            fh.write(line + "\n")
    except OSError:  # pragma: no cover
        pass


def stable_seed(*parts) -> list:
    """Deterministic seed sequence for numpy from the prespecified seed and a context label."""
    return [SEED, zlib.crc32("\x1f".join(map(str, parts)).encode())]


def set_digest(mask: np.ndarray) -> int:
    """CRC32 of a boolean membership mask (content-based bootstrap seeds)."""
    return zlib.crc32(np.packbits(np.asarray(mask, dtype=bool)).tobytes())


# ----------------------------------------------------------------------------
# Tiers
# ----------------------------------------------------------------------------
def tie_blocks(c: np.ndarray) -> tuple[np.ndarray, int]:
    """Block index of every element (0 = highest credibility) and the number of blocks.
    A tie block = all elements with equal credibility after rounding to TIE_DECIMALS."""
    v = np.round(np.asarray(c, dtype=np.float64), TIE_DECIMALS)
    uniq, inv = np.unique(-v, return_inverse=True)
    return inv.reshape(-1), len(uniq)


def bfdr_n_blocks(c: np.ndarray, w: np.ndarray, q: float) -> int:
    """Number of leading tie blocks in the largest whole-block prefix with BFDR <= q.

    BFDR(prefix) = sum w (1 - c) / sum w over the prefix; it is non-decreasing along the
    decreasing-credibility order, so the admissible prefixes are exactly those before the first
    violation."""
    c = np.asarray(c, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    inv, nb = tie_blocks(c)
    bw = np.bincount(inv, weights=w, minlength=nb)
    bl = np.bincount(inv, weights=w * (1.0 - c), minlength=nb)
    cw, cl = np.cumsum(bw), np.cumsum(bl)
    ok = (cw > 0) & (cl <= q * cw * (1 + 1e-12) + 1e-15)
    return int(np.argmin(ok)) if not ok.all() else nb


def bfdr_levels(c: np.ndarray, w: np.ndarray, q_strong: float = Q_STRONG,
                q_moderate: float = Q_MODERATE) -> np.ndarray:
    """Per-element level: 2 = Strong (BFDR <= q_strong prefix), 1 = Moderate (BFDR <= q_moderate
    prefix, not Strong), 0 = Below.  Whole tie blocks only."""
    if not q_strong <= q_moderate:
        raise ValueError("q_strong must be <= q_moderate")
    inv, _ = tie_blocks(c)
    ns, nm = bfdr_n_blocks(c, w, q_strong), bfdr_n_blocks(c, w, q_moderate)
    return np.where(inv < ns, 2, np.where(inv < nm, 1, 0)).astype(np.int8)


def budget_n_blocks(c: np.ndarray, w: np.ndarray, k: float) -> int:
    """Number of leading tie blocks in the largest whole-block prefix holding <= k edges."""
    inv, nb = tie_blocks(c)
    cw = np.cumsum(np.bincount(inv, weights=np.asarray(w, dtype=np.float64), minlength=nb))
    return int(np.searchsorted(cw, k + 1e-9, side="right"))


def rank_budget_levels(c: np.ndarray, w: np.ndarray, k1: float, k2: float) -> np.ndarray:
    """Rank-budget tiers: level 2 = largest whole-block prefix with <= k1 edges, level 1 = with
    <= k2 edges (minus level 2)."""
    inv, _ = tie_blocks(c)
    n1, n2 = budget_n_blocks(c, w, k1), budget_n_blocks(c, w, k2)
    return np.where(inv < n1, 2, np.where(inv < n2, 1, 0)).astype(np.int8)


def dual_tier(lg: np.ndarray, lc: np.ndarray, rule: str = "primary") -> np.ndarray:
    """Dual tier (3 Strong, 2 Moderate, 1 Conditional, 0 Exploratory) from the two levels."""
    return DUAL_LUT[rule][np.asarray(lg, dtype=np.int64), np.asarray(lc, dtype=np.int64)]


def realized_bfdr(c: np.ndarray, w: np.ndarray, mask: np.ndarray) -> float:
    ww = np.asarray(w, dtype=np.float64) * mask
    return float(np.sum(ww * (1 - c)) / ww.sum()) if ww.sum() > 0 else float("nan")


# ----------------------------------------------------------------------------
# Statistics
# ----------------------------------------------------------------------------
def oe_ci(groups: np.ndarray, obs: np.ndarray, exp: np.ndarray, n_boot: int, seed) -> dict:
    """Observed / expected with a cluster-bootstrap 95% interval (clusters = ``groups``)."""
    obs = np.asarray(obs, dtype=np.float64)
    exp = np.asarray(exp, dtype=np.float64)
    o, e = float(obs.sum()), float(exp.sum())
    out = dict(obs=o, exp=e, oe=(o / e) if e > 0 else float("nan"), lo=float("nan"), hi=float("nan"))
    if len(obs) == 0:
        return out
    codes, _ = pd.factorize(np.asarray(groups), sort=True)
    G = int(codes.max()) + 1
    go = np.bincount(codes, weights=obs, minlength=G)
    ge = np.bincount(codes, weights=exp, minlength=G)
    if G < 2 or n_boot <= 0:
        return out
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, G, size=(n_boot, G))
    bo, be = go[idx].sum(1), ge[idx].sum(1)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.where(be > 0, bo / be, np.nan)
    if np.isfinite(r).sum() >= max(10, n_boot // 2):
        out["lo"], out["hi"] = (float(x) for x in np.nanpercentile(r, [2.5, 97.5]))
    return out


def frac_ci(groups: np.ndarray, hit: np.ndarray, n_boot: int, seed) -> tuple[float, float, float]:
    """Fraction of hits with a cluster-bootstrap 95% interval."""
    r = oe_ci(groups, hit, np.ones(len(hit)), n_boot, seed)
    return r["oe"], r["lo"], r["hi"]


def mantel_trend(x: np.ndarray, y: np.ndarray, strata: np.ndarray) -> dict:
    """Mantel extension test for a linear trend of a binary y on scores x, stratified.

    Within stratum s (N_s items, m_s positives) T_s = sum x_i y_i has, under H0 (y exchangeable
    within the stratum), mean m_s xbar_s and variance m_s (N_s - m_s) / (N_s (N_s - 1)) S_xx,s."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    codes, _ = pd.factorize(np.asarray(strata), sort=True)
    G = int(codes.max()) + 1 if len(codes) else 0
    N = np.bincount(codes, minlength=G).astype(np.float64)
    m = np.bincount(codes, weights=y, minlength=G)
    sx = np.bincount(codes, weights=x, minlength=G)
    sxx = np.bincount(codes, weights=x * x, minlength=G)
    T = np.bincount(codes, weights=x * y, minlength=G)
    ok = N > 1
    Ex = np.where(ok, m * sx / np.maximum(N, 1), 0.0)
    Sxx = sxx - sx * sx / np.maximum(N, 1)
    V = np.where(ok, m * (N - m) / (N * np.maximum(N - 1, 1)) * Sxx, 0.0)
    t, e, v = float(T[ok].sum()), float(Ex[ok].sum()), float(V[ok].sum())
    z = (t - e) / math.sqrt(v) if v > 0 else float("nan")
    return dict(T=t, E=e, V=v, z=z,
                p_one_sided=float(norm.sf(z)) if np.isfinite(z) else float("nan"),
                p_two_sided=float(2 * norm.sf(abs(z))) if np.isfinite(z) else float("nan"),
                n_strata_informative=int(np.sum(ok & (m > 0) & (m < N) & (Sxx > 1e-12))))


def oe_power_analysis(N, K, n, oe_targets: dict | None = None, alpha: float = MDE_ALPHA,
                      target: float = MDE_POWER, trim: float = 1e-14) -> dict:
    """Label-free (margins-only) power of the TF-stratified exact conditional test of an O/E.

    Strata t (TFs) with N_t evaluable edges, K_t positives and n_t evaluable edges in the set.
    Null: X_t ~ Hypergeometric(N_t, K_t, n_t) independently; T = sum X_t, E_0 = sum n_t K_t / N_t;
    one-sided level-alpha critical value t_crit from the exact null.  H1: common odds ratio psi
    (Fisher non-central hypergeometric).  Returns ``mde_oe`` (E_psi[T] / E_0 at the smallest psi with
    power >= target; inf if not reached for psi <= 1e4), ``max_attainable_oe`` = sum min(n_t, K_t) / E_0,
    ``mde_oe_poisson`` (from E_0 alone: T ~ Poisson(rho E_0)), and for every name -> O/E in
    ``oe_targets`` the power at that O/E (NaN when the O/E exceeds the attainable maximum)."""
    from scipy.stats import poisson
    from r2 import replication as RP
    N, K, n = (np.asarray(x, dtype=np.int64) for x in (N, K, n))
    use = n > 0
    N, K, n = N[use], K[use], n[use]
    E0 = float(np.sum(n * K / np.maximum(N, 1)))
    out = dict(E0=E0, n_strata=int(len(N)), mde_oe=np.nan, mde_odds_ratio=np.nan, max_attainable_oe=np.nan,
               mde_oe_poisson=np.nan, t_crit=np.nan)
    for k in (oe_targets or {}):
        out[f"power_at_{k}"] = np.nan
    if E0 <= 0:
        return out
    inf_ = (K > 0) & (K < N) & (n < N)
    fixed = float(np.sum((n * K / np.maximum(N, 1))[~inf_]))      # deterministic strata: X_t = n_t K_t / N_t
    Ni, Ki, ni = N[inf_], K[inf_], n[inf_]
    out["max_attainable_oe"] = (fixed + float(np.minimum(ni, Ki).sum())) / E0
    # Poisson MDE from E_0 alone
    c = int(poisson.isf(alpha, E0)) + 1
    lo_r, hi_r = 1.0, 1.0
    while poisson.sf(c - 1, hi_r * E0) < target and hi_r < 1e4:
        hi_r *= 2
    for _ in range(60):
        mid = 0.5 * (lo_r + hi_r)
        if poisson.sf(c - 1, mid * E0) >= target:
            hi_r = mid
        else:
            lo_r = mid
    out["mde_oe_poisson"] = hi_r
    if len(Ni) == 0:
        return out
    s0, p0 = RP.sum_distribution(Ni, Ki, ni, 0.0)
    tail0 = np.cumsum(p0[::-1])[::-1]
    ok = np.nonzero(tail0 <= alpha)[0]
    if len(ok) == 0:
        out["mde_oe"] = np.inf
        return out
    t_crit = s0 + int(ok[0])
    out["t_crit"] = t_crit

    def power(lor):
        s1, p1 = RP.sum_distribution(Ni, Ki, ni, lor, trim)
        i = t_crit - s1
        return float(p1[max(i, 0):].sum()) if i < len(p1) else 0.0

    def mean_t(lor):
        m = 0.0
        for a, b, c_ in zip(Ni, Ki, ni):
            l0, pm = RP.hyper_pmf(int(a), int(b), int(c_), lor)
            m += float(np.dot(np.arange(l0, l0 + len(pm)), pm))
        return m

    def oe_of(lor):
        return (fixed + mean_t(lor)) / E0

    lo, hi = 0.0, math.log(1e4)
    if power(hi) < target:
        out["mde_oe"] = np.inf
    else:
        for _ in range(30):
            mid = 0.5 * (lo + hi)
            if power(mid) >= target:
                hi = mid
            else:
                lo = mid
        out["mde_oe"] = oe_of(hi)
        out["mde_odds_ratio"] = math.exp(hi)
    for name, rho in (oe_targets or {}).items():
        if not np.isfinite(rho) or rho <= 0:
            continue
        if rho >= out["max_attainable_oe"]:
            continue
        if rho <= oe_of(0.0):
            out[f"power_at_{name}"] = power(0.0) if rho >= 1 else np.nan
            continue
        a, b = 0.0, math.log(1e4)
        for _ in range(30):
            mid = 0.5 * (a + b)
            if oe_of(mid) >= rho:
                b = mid
            else:
                a = mid
        out[f"power_at_{name}"] = power(b)
    return out


def holm(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=np.float64)
    out = np.full_like(p, np.nan)
    ok = np.isfinite(p)
    if not ok.any():
        return out
    pv = p[ok]
    o = np.argsort(pv)
    m = len(pv)
    adj = np.maximum.accumulate((m - np.arange(m)) * pv[o])
    res = np.empty(m)
    res[o] = np.minimum(adj, 1.0)
    out[ok] = res
    return out


def weighted_kappa(a: np.ndarray, b: np.ndarray, w: np.ndarray, k: int, weights: str = "linear") -> float:
    """Cohen's weighted kappa of two ordinal ratings 0..k-1 with case weights w."""
    a = np.asarray(a, dtype=np.int64)
    b = np.asarray(b, dtype=np.int64)
    w = np.asarray(w, dtype=np.float64)
    O = np.zeros((k, k))
    np.add.at(O, (a, b), w)
    O /= O.sum()
    Ex = np.outer(O.sum(1), O.sum(0))
    i, j = np.meshgrid(np.arange(k), np.arange(k), indexing="ij")
    D = np.abs(i - j) / (k - 1) if weights == "linear" else ((i - j) / (k - 1)) ** 2
    den = float((D * Ex).sum())
    return 1.0 - float((D * O).sum()) / den if den > 0 else float("nan")


def weighted_spearman(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
    """Spearman correlation of patterns weighted by edge counts (ties -> mid-ranks)."""
    x = np.round(np.asarray(x, dtype=np.float64), TIE_DECIMALS)
    y = np.round(np.asarray(y, dtype=np.float64), TIE_DECIMALS)
    w = np.asarray(w, dtype=np.float64)

    def midranks(v):
        o = np.argsort(v, kind="mergesort")
        vs, ws = v[o], w[o]
        cum = np.cumsum(ws)
        uniq, start = np.unique(vs, return_index=True)
        end = np.r_[start[1:], len(vs)]
        hi = cum[end - 1]
        lo = np.r_[0.0, hi[:-1]]
        mid = (lo + hi + 1) / 2
        r = np.empty(len(v))
        r[o] = np.repeat(mid, end - start)
        return r

    rx, ry = midranks(x), midranks(y)
    mx, my = np.average(rx, weights=w), np.average(ry, weights=w)
    vx, vy = np.average((rx - mx) ** 2, weights=w), np.average((ry - my) ** 2, weights=w)
    if vx <= 0 or vy <= 0:
        return float("nan")
    return float(np.average((rx - mx) * (ry - my), weights=w) / math.sqrt(vx * vy))


def concentration(tfs, n_tf_universe: int | None = None) -> dict:
    """Concentration of an edge set over TFs."""
    s = pd.Series(np.asarray(tfs, dtype=object))
    n = len(s)
    if n == 0:
        return dict(n_edges=0, n_tfs=0, top1_tf="", top1_share=np.nan, top2_share=np.nan,
                    entropy=np.nan, evenness=np.nan, entropy_norm_universe=np.nan, top_tfs="")
    vc = s.value_counts()
    vc = vc.sort_index().sort_values(ascending=False, kind="mergesort")   # deterministic ties
    p = vc.to_numpy(np.float64) / n
    H = max(0.0, float(-(p * np.log(p)).sum()))
    k = len(vc)
    return dict(n_edges=n, n_tfs=k, top1_tf=str(vc.index[0]), top1_share=float(p[0]),
                top2_share=float(p[:2].sum()), entropy=H,
                evenness=H / math.log(k) if k > 1 else np.nan,
                entropy_norm_universe=(H / math.log(n_tf_universe)) if n_tf_universe and n_tf_universe > 1 else np.nan,
                top_tfs=";".join(f"{t}:{c}" for t, c in vc.head(TOP_TFS_LISTED).items()))


# ----------------------------------------------------------------------------
# Deconcentration
# ----------------------------------------------------------------------------
def within_tf_rank(tf_codes: np.ndarray, score: np.ndarray, keys: np.ndarray) -> np.ndarray:
    """0-based rank of every edge within its TF by (score desc, key desc)."""
    o = np.lexsort((~keys, -score, tf_codes))
    tc = tf_codes[o]
    first = np.searchsorted(tc, tc, side="left")
    r = np.empty(len(o), dtype=np.int64)
    r[o] = np.arange(len(o)) - first
    return r


def deconcentrate(tf_codes: np.ndarray, score: np.ndarray, keys: np.ndarray, budget: int,
                  alpha: float) -> np.ndarray:
    """Greedy TF-penalised selection (see module docstring); returns a boolean mask.

    Priority of the j-th edge (0-based, by score) of a TF = logit(score) - alpha * j; the greedy
    algorithm with this penalty selects the top-``budget`` edges by that priority (ties: key)."""
    tf_codes = np.asarray(tf_codes, dtype=np.int64)
    score = np.asarray(score, dtype=np.float64)
    keys = np.asarray(keys, dtype=np.uint64)
    budget = int(min(max(budget, 0), len(score)))
    j = within_tf_rank(tf_codes, score, keys)
    pr = logit(np.clip(score, LOGIT_CLIP, 1 - LOGIT_CLIP)) - alpha * j
    o = np.lexsort((~keys, -pr))
    mask = np.zeros(len(score), dtype=bool)
    mask[o[:budget]] = True
    return mask


def deconcentrate_naive(tf_codes, score, keys, budget, alpha):
    """Reference implementation (literal greedy loop) used by the tests."""
    tf_codes = np.asarray(tf_codes)
    lg = logit(np.clip(np.asarray(score, dtype=np.float64), LOGIT_CLIP, 1 - LOGIT_CLIP))
    keys = np.asarray(keys, dtype=np.uint64)
    chosen = np.zeros(len(lg), dtype=bool)
    count: dict = {}
    for _ in range(int(min(budget, len(lg)))):
        best, best_key = None, None
        for i in np.flatnonzero(~chosen):
            p = lg[i] - alpha * count.get(tf_codes[i], 0)
            if best is None or p > best[0] + 1e-12 or (abs(p - best[0]) <= 1e-12 and keys[i] > best_key):
                best, best_key = (p, i), keys[i]
        chosen[best[1]] = True
        count[tf_codes[best[1]]] = count.get(tf_codes[best[1]], 0) + 1
    return chosen


# ----------------------------------------------------------------------------
# Held-out labels
# ----------------------------------------------------------------------------
def _read_list(path: Path) -> list:
    return [x.strip() for x in path.read_text().splitlines() if x.strip()]


def chip_reference(name: str) -> tuple[pd.DataFrame, list, list]:
    d = C.OUT / "heldout_chip"
    edges = pd.read_parquet(d / f"{name}.parquet", columns=["tf", "target"])
    return edges, _read_list(d / f"{name}_assayed_tfs.txt"), _read_list(d / f"{name}_gene_universe.txt")


def load_pert_pooled(log_fn=log) -> pd.DataFrame:
    """Pooled perturbation label over all held-out cohorts (Lambert TFs perturbed; cis pairs out).

    Returns one row per evaluable (tf, target) pair: ``positive`` (any cohort), ``n_cohorts``
    (cohorts testing the pair), ``n_cohorts_positive`` and the cohorts listing."""
    tfs = C.lambert_tfs()
    parts = []
    for c in COHORTS:
        p = C.OUT / "cohorts" / c / "de.parquet"
        d = pd.read_parquet(p, columns=["perturbed_gene", "target", "positive_primary", "cis_same_band", "is_tf"],
                            filters=[("is_tf", "==", True)])
        for col in ("perturbed_gene", "target"):          # stored as categoricals
            d[col] = d[col].astype(str)
        d = d[d["perturbed_gene"].isin(tfs) & ~d["cis_same_band"].astype(bool)
              & (d["perturbed_gene"] != d["target"])]
        if len(d) == 0:
            log_fn(f"  pert_pooled: {c} has no TF perturbation rows")
            continue
        d = d.rename(columns={"perturbed_gene": "tf"})[["tf", "target", "positive_primary"]]
        d["cohort"] = c
        parts.append(d)
        log_fn(f"  pert_pooled: {c}: {d['tf'].nunique()} TFs, {len(d):,} pairs, {int(d['positive_primary'].sum()):,} positives")
    d = pd.concat(parts, ignore_index=True)
    d["positive_primary"] = d["positive_primary"].astype(bool)
    g = d.groupby(["tf", "target"], sort=True)
    out = g.agg(positive=("positive_primary", "any"), n_cohorts=("cohort", "nunique"),
                n_cohorts_positive=("positive_primary", "sum")).reset_index()
    pos_coh = d[d["positive_primary"]].groupby(["tf", "target"])["cohort"].apply(lambda s: ";".join(sorted(set(s))))
    out = out.merge(pos_coh.rename("cohorts_positive").reset_index(), on=["tf", "target"], how="left")
    out["cohorts_positive"] = out["cohorts_positive"].fillna("")
    return out


def _additive_logit(k: np.ndarray, n: np.ndarray, max_iter: int = 200, gtol: float = 1e-6) -> np.ndarray:
    """ML fit of logit P[i, b] = a_i + c_b (c_0 = 0) to binomial counts k / n over cells n > 0
    (damped Newton, as in ``build_heldout_chip.target_degree_model``).  Returns P (n_tf x n_bin)."""
    n_tf, nb = n.shape
    cells = np.argwhere(n > 0)
    X = np.zeros((len(cells), n_tf + nb - 1))
    X[np.arange(len(cells)), cells[:, 0]] = 1.0
    nz = cells[:, 1] > 0
    X[np.flatnonzero(nz), n_tf + cells[nz, 1] - 1] = 1.0
    y, w = k[cells[:, 0], cells[:, 1]], n[cells[:, 0], cells[:, 1]]
    beta = np.zeros(X.shape[1])
    rate = np.clip(k.sum(1) / np.maximum(n.sum(1), 1), 1e-6, 1 - 1e-6)
    beta[:n_tf] = np.log(rate / (1 - rate))

    def ll(b):
        eta = X @ b
        return float(np.sum(y * eta - w * np.logaddexp(0.0, eta)))

    cur = ll(beta)
    with HC.single_thread_blas():
        for _ in range(max_iter):
            p = expit(X @ beta)
            g = X.T @ (y - w * p)
            if np.max(np.abs(g)) < gtol:
                break
            H = (X * (w * p * (1 - p))[:, None]).T @ X + 1e-9 * np.eye(X.shape[1])
            step, t = np.linalg.solve(H, g), 1.0
            while True:
                cand = beta + t * step
                lc = ll(cand)
                if lc >= cur - 1e-12 * abs(cur) or t < 1e-10:
                    break
                t /= 2
            beta, cur = cand, lc
        else:
            raise RuntimeError("additive logistic null did not converge")
    eta = beta[:n_tf][:, None] + np.concatenate([[0.0], beta[n_tf:]])[None, :]
    return expit(eta)


def masked_degree_index(pairs: pd.DataFrame, eval_genes) -> dict:
    """Target-degree null for a reference whose evaluable pairs are not a full TF x gene grid.

    ``pairs``: evaluable (tf, target) with boolean ``positive``.  Restricted to targets in
    ``eval_genes`` (tf != target).  Degree of g for the pair (TF, g) = number of *other* TFs for
    which g is a positive (leave-pair-out), bins = ``build_heldout_chip.degree_bin_edges``
    (prespecified rule), null = additive logistic a_TF + b_bin fitted on evaluable pairs.
    With a full evaluable grid this reproduces ``build_heldout_chip.target_degree_model``."""
    eg = set(eval_genes)
    d = pairs[pairs["target"].isin(eg) & (pairs["tf"] != pairs["target"])]
    tfs = sorted(set(d["tf"]))
    genes = sorted(set(d["target"]))
    ti = {t: i for i, t in enumerate(tfs)}
    gi = {g: i for i, g in enumerate(genes)}
    Ev = np.zeros((len(tfs), len(genes)), dtype=bool)
    M = np.zeros_like(Ev)
    r, c = d["tf"].map(ti).to_numpy(), d["target"].map(gi).to_numpy()
    Ev[r, c] = True
    M[r, c] = d["positive"].to_numpy(bool)
    deg = M.sum(0).astype(np.int64)
    edges = HC.degree_bin_edges(deg, int(P["target_degree_bins"]), P["target_degree_bin_rule"])
    inner = edges[1:-1]

    def to_bin(x):
        return np.searchsorted(inner, x, side="right")

    nb = len(inner) + 1
    n = np.zeros((len(tfs), nb))
    k = np.zeros((len(tfs), nb))
    for i in range(len(tfs)):
        b = to_bin(deg - M[i])
        e = Ev[i]
        n[i] = np.bincount(b[e], minlength=nb)
        k[i] = np.bincount(b[e], weights=M[i][e].astype(float), minlength=nb)
    Pm = _additive_logit(k, n)
    base = {t: (M[i].sum() / Ev[i].sum()) if Ev[i].sum() else np.nan for i, t in enumerate(tfs)}
    return dict(tfs=tfs, genes=genes, ti=ti, gi=gi, E=Ev, M=M, deg=deg, bin_edges=edges,
                to_bin=to_bin, P=Pm, base=base)


def label_frame_from_index(R: dict, tf, target, masked: bool) -> pd.DataFrame:
    """Labels and expectations of (tf, target) pairs under a reference index.

    Columns: ``evaluable`` (bool), ``label`` (float 1/0, NaN if not evaluable), ``exp_tf``,
    ``exp_deg`` (NaN if not evaluable) and ``deg_bin`` (-1 if not evaluable)."""
    tf = np.asarray(tf, dtype=object)
    target = np.asarray(target, dtype=object)
    t_idx = pd.Series(tf).map(R["ti"]).to_numpy(dtype=float)
    g_idx = pd.Series(target).map(R["gi"]).to_numpy(dtype=float)
    ok = ~np.isnan(t_idx) & ~np.isnan(g_idx) & (tf != target)
    ti_, gi_ = t_idx[ok].astype(np.int64), g_idx[ok].astype(np.int64)
    if masked:
        ok_e = R["E"][ti_, gi_]
        sel = np.flatnonzero(ok)
        ok[sel[~ok_e]] = False
        ti_, gi_ = ti_[ok_e], gi_[ok_e]
    n = len(tf)
    out = pd.DataFrame({"evaluable": ok, "label": np.full(n, np.nan), "exp_tf": np.full(n, np.nan),
                        "exp_deg": np.full(n, np.nan), "deg_bin": np.full(n, -1, dtype=np.int64)})
    lab = R["M"][ti_, gi_].astype(float)
    b = R["to_bin"](R["deg"][gi_] - R["M"][ti_, gi_])
    base = np.array([R["base"][t] for t in np.asarray(R["tfs"], dtype=object)[ti_]], dtype=float)
    out.loc[ok, "label"] = lab
    out.loc[ok, "exp_tf"] = base
    out.loc[ok, "exp_deg"] = R["P"][ti_, b]
    out.loc[ok, "deg_bin"] = b
    return out


def build_reference_indices(eval_genes, pooled: pd.DataFrame) -> dict:
    """Reference indices for an evaluation gene set (ChIP via ``reference_index``)."""
    eg = set(eval_genes)
    out = {}
    for r in CHIP_REFS:
        edges, assayed, universe = chip_reference(r)
        out[r] = (HC.reference_index(edges, set(universe) & eg, assayed=assayed), False)
    out["pert_pooled"] = (masked_degree_index(pooled, eg), True)
    return out


# ----------------------------------------------------------------------------
# Credibility loading
# ----------------------------------------------------------------------------
def fit_path(universe: str, fit_id: str) -> Path:
    return FITS / universe / VARIANT / fit_id


def load_cred(universe: str, fit_id: str) -> dict:
    d = fit_path(universe, fit_id)
    out = {"mean": np.load(d / ("cred_mean.npy" if fit_id.endswith("_nuts") else "cred.npy"))}
    if fit_id.endswith("_nuts"):
        for k in ("lo", "hi", "sd"):
            out[k] = np.load(d / f"cred_{k}.npy")
    out["meta"] = json.loads((d / "meta.json").read_text())
    return out


def draw_credibility(draws: dict, R: np.ndarray) -> np.ndarray:
    """(D, P) plug-in credibility of patterns R under each stored posterior draw (w, theta)."""
    W = np.asarray(draws["w"], dtype=np.float64)
    TH = np.clip(np.asarray(draws["theta"], dtype=np.float64), 1e-12, 1 - 1e-12)
    obs = (R >= 0)
    A = (obs & (R == 1)).astype(np.float64)
    B = (obs & (R == 0)).astype(np.float64)
    lj = np.log(W)[:, None, :] + np.einsum("pk,dks->dps", A, np.log(TH)) \
        + np.einsum("pk,dks->dps", B, np.log1p(-TH))
    return np.exp(lj[:, :, 0] - logsumexp(lj, axis=2))


def pattern_key_map(R_from: np.ndarray, R_to: np.ndarray) -> np.ndarray:
    """Index of each row of R_from among the rows of R_to (exact match; raises if missing)."""
    key = {r.tobytes(): i for i, r in enumerate(np.ascontiguousarray(R_to))}
    idx = np.array([key.get(r.tobytes(), -1) for r in np.ascontiguousarray(R_from)], dtype=np.int64)
    if (idx < 0).any():
        raise RuntimeError(f"{int((idx < 0).sum())} patterns not found in the target pattern set")
    return idx


# ----------------------------------------------------------------------------
# Context analysis
# ----------------------------------------------------------------------------
class Context:
    """Everything needed for one tissue, at pattern level (P) and edge level (N)."""

    def __init__(self, t: str, glob: dict, n_boot: int):
        self.t, self.n_boot = t, n_boot
        ev = pd.read_parquet(C.OUT / "evidence" / t / "evidence.parquet", columns=["tf", "target"])
        self.tf = ev["tf"].to_numpy(object)
        self.target = ev["target"].to_numpy(object)
        self.N = len(ev)
        self.ep = np.asarray(E.load_edge_pattern(t, VARIANT, mmap=False), dtype=np.int64)
        R, n, fam, ann = E.load_patterns(t, VARIANT)
        if list(ann) != list(glob["ann"]):
            raise RuntimeError(f"{t}: annotator order differs from the global fit")
        self.R, self.n = R, n.astype(np.float64)
        assert np.allclose(np.bincount(self.ep, minlength=len(n)), n)
        self.tf_list = sorted(set(self.tf))
        self.n_tf_universe = len(self.tf_list)
        tfc = {x: i for i, x in enumerate(self.tf_list)}
        self.tf_code = pd.Series(self.tf).map(tfc).to_numpy(np.int64)
        # global-fit credibility of the same patterns (verified identical patterns)
        gidx = pattern_key_map(R, glob["R"])
        self.gidx = gidx
        self.check_pattern_identity(glob)
        cg = glob["nuts"]
        self.cg = {k: cg[k][gidx] for k in ("mean", "lo", "hi", "sd")}
        cc = load_cred(t, PRIMARY_FIT)
        self.cc = {k: cc[k] for k in ("mean", "lo", "hi", "sd")}
        self.cc_meta = cc["meta"]
        self.glob_meta = glob["nuts_meta"]
        self.cg_map = glob["map"][gidx]
        self.cc_map = load_cred(t, MAP_FIT)["mean"]
        self.cg_em = glob["em"][gidx]
        self.cc_em = load_cred(t, EM_FIT)["mean"]
        self.n_obs_p = (R >= 0).sum(1)
        self.n_rep_p = (R == 1).sum(1)
        # posterior draws (plug-in credibility per draw)
        dg = np.load(fit_path("global", PRIMARY_FIT) / "draws.npz")
        dc = np.load(fit_path(t, PRIMARY_FIT) / "draws.npz")
        self.draws_g = draw_credibility(dg, R)
        self.draws_c = draw_credibility(dc, R)
        D = min(len(self.draws_g), len(self.draws_c))
        self.draws_g, self.draws_c = self.draws_g[:D], self.draws_c[:D]
        # primary tiers
        self.lg = bfdr_levels(self.cg["mean"], self.n)
        self.lc = bfdr_levels(self.cc["mean"], self.n)
        self.dual = dual_tier(self.lg, self.lc, "primary")
        self.labels: dict = {}

    def check_pattern_identity(self, glob: dict) -> None:
        """Edges in U_glob must have the identical pattern in the context and global universes."""
        gt = {x: i for i, x in enumerate(glob["tfs"])}
        gg = {x: i for i, x in enumerate(glob["genes"])}
        i = pd.Series(self.tf).map(gt).to_numpy(dtype=float)
        j = pd.Series(self.target).map(gg).to_numpy(dtype=float)
        ok = ~np.isnan(i) & ~np.isnan(j)
        self.in_global = ok
        gidx_edge = i[ok].astype(np.int64) * len(glob["genes"]) + j[ok].astype(np.int64)
        gpat = np.asarray(glob["ep"][gidx_edge], dtype=np.int64)
        if not np.array_equal(gpat, self.gidx[self.ep[ok]]):
            raise RuntimeError(f"{self.t}: context and global evidence patterns differ")
        if (self.R[self.ep[~ok]] != -1).any():
            raise RuntimeError(f"{self.t}: an edge outside U_glob has observed evidence")

    # -- helpers at edge level ------------------------------------------------
    def edge(self, pattern_values: np.ndarray) -> np.ndarray:
        return np.asarray(pattern_values)[self.ep]

    def attach_labels(self, indices: dict) -> None:
        for r, (R, masked) in indices.items():
            self.labels[r] = label_frame_from_index(R, self.tf, self.target, masked)

    def set_eval(self, mask_edge: np.ndarray, r: str, seed_tag: str, use: str = "tf") -> dict:
        """O/E of the edge set ``mask_edge`` for reference r."""
        L = self.labels[r]
        m = mask_edge & L["evaluable"].to_numpy()
        lab = L["label"].to_numpy()[m]
        tf = self.tf[m]
        res = dict(n_edges=int(mask_edge.sum()), n_eval=int(m.sum()), n_tfs_eval=int(len(set(tf))),
                   few_clusters=bool(len(set(tf)) < MIN_CLUSTERS), **top_tf_share(tf))
        # the bootstrap seed depends on the evaluated edges only, so identical sets reached by
        # different rules get identical intervals (``seed_tag`` is kept for logging only)
        dig = set_digest(m)
        a = oe_ci(tf, lab, L["exp_tf"].to_numpy()[m], self.n_boot, stable_seed(self.t, r, dig, "tf"))
        b = oe_ci(tf, lab, L["exp_deg"].to_numpy()[m], self.n_boot, stable_seed(self.t, r, dig, "deg"))
        base = self.base_rate(r)
        res.update(obs=a["obs"], exp_tf=a["exp"], oe_tf=a["oe"], oe_tf_lo=a["lo"], oe_tf_hi=a["hi"],
                   exp_tfdeg=b["exp"], oe_tfdeg=b["oe"], oe_tfdeg_lo=b["lo"], oe_tfdeg_hi=b["hi"],
                   base_rate=base,
                   marginal_enrichment=(a["obs"] / m.sum() / base) if m.sum() and base > 0 else np.nan)
        return res

    def margins(self, r: str, mask_edge: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Per-TF (N_t evaluable edges, K_t positives, n_t evaluable edges in ``mask_edge``)."""
        L = self.labels[r]
        e = L["evaluable"].to_numpy()
        lab = np.nan_to_num(L["label"].to_numpy())
        T = self.n_tf_universe
        N_t = np.bincount(self.tf_code[e], minlength=T)
        K_t = np.bincount(self.tf_code[e], weights=lab[e], minlength=T).round().astype(np.int64)
        n_t = (np.bincount(self.tf_code[e & mask_edge], minlength=T) if mask_edge is not None
               else np.zeros(T, dtype=np.int64))
        return N_t, K_t, n_t

    def mde_cols(self, mask_edge: np.ndarray, r: str, global_oe: float | None) -> dict:
        N_t, K_t, n_t = self.margins(r, mask_edge)
        res = oe_power_analysis(N_t, K_t, n_t, {"global_oe": global_oe} if global_oe is not None else None)
        return power_columns(res, global_oe)

    def base_rate(self, r: str) -> float:
        L = self.labels[r]
        e = L["evaluable"].to_numpy()
        return float(L["label"].to_numpy()[e].mean()) if e.any() else float("nan")


def top_tf_share(tf) -> dict:
    """TF holding most edges of a set and its share (ties: alphabetical); empty set -> '' / NaN."""
    tf = np.asarray(tf, dtype=object)
    if len(tf) == 0:
        return dict(top_tf_eval="", top_tf_share_eval=np.nan)
    vc = pd.Series(tf).value_counts().sort_index().sort_values(ascending=False, kind="mergesort")
    return dict(top_tf_eval=str(vc.index[0]), top_tf_share_eval=float(vc.iloc[0] / len(tf)))


def power_columns(res: dict, global_oe: float | None) -> dict:
    return dict(mde_oe=res["mde_oe"], mde_odds_ratio=res["mde_odds_ratio"], mde_oe_poisson=res["mde_oe_poisson"],
                max_attainable_oe=res["max_attainable_oe"], n_tfs_in_power=res["n_strata"],
                global_universe_oe=(global_oe if global_oe is not None else np.nan),
                power_at_global_oe=res.get("power_at_global_oe", np.nan),
                power_note=("TF-stratified exact conditional test, one-sided alpha=%g, power %g; margins only; ignores "
                            "within-TF dependence and TF heterogeneity (optimistic)" % (MDE_ALPHA, MDE_POWER)))


def load_global() -> dict:
    R, n, fam, ann = E.load_patterns("global", VARIANT)
    g = dict(R=R, n=n.astype(np.float64), ann=list(ann), fam=list(fam))
    g["tfs"] = _read_list(C.OUT / "evidence" / "global" / "tfs.txt")
    g["genes"] = _read_list(C.OUT / "evidence" / "global" / "genes.txt")
    g["ep"] = E.load_edge_pattern("global", VARIANT, mmap=True)
    nuts = load_cred("global", PRIMARY_FIT)
    g["nuts"] = {k: nuts[k] for k in ("mean", "lo", "hi", "sd")}
    g["nuts_meta"] = nuts["meta"]
    g["map"] = load_cred("global", MAP_FIT)["mean"]
    g["em"] = load_cred("global", EM_FIT)["mean"]
    if len(g["nuts"]["mean"]) != len(R):
        raise RuntimeError("global credibility not aligned with patterns")
    return g


# ----------------------------------------------------------------------------
# Tables per tissue
# ----------------------------------------------------------------------------
def tier_table(cx: Context) -> list:
    rows = []
    n = cx.n
    D = len(cx.draws_g)
    lg_d = np.stack([bfdr_levels(cx.draws_g[d], n) for d in range(D)])
    lc_d = np.stack([bfdr_levels(cx.draws_c[d], n) for d in range(D)])
    dual_d = DUAL_LUT["primary"][lg_d, lc_d]
    cx.p_promoted_pattern = (dual_d >= 2).mean(0)
    cx.dual_draws = dual_d

    joint = np.minimum(cx.cg["mean"], cx.cc["mean"])

    def add(scope, level_name, order, mask_p, cred, counts_draw, cum_mask=None):
        """cred = the credibility that orders the scope (joint = min of both for dual rows);
        realised BFDR is reported under both fits; ``cum_mask`` = the cumulative set (e.g.
        Strong U Moderate for the Moderate row)."""
        e_mask = cx.edge(mask_p)
        inv, _ = tie_blocks(cred)
        cm = mask_p if cum_mask is None else cum_mask
        rows.append(dict(tissue=cx.t, scope=scope, tier=level_name, tier_order=order,
                         n_edges=int(n[mask_p].sum()), n_tfs=int(len(set(cx.tf[e_mask]))),
                         n_patterns=int(mask_p.sum()), n_tie_blocks=int(len(set(inv[mask_p]))),
                         share_of_universe=float(n[mask_p].sum() / n.sum()),
                         cred_min=float(cred[mask_p].min()) if mask_p.any() else np.nan,
                         cred_max=float(cred[mask_p].max()) if mask_p.any() else np.nan,
                         bfdr_global_fit=realized_bfdr(cx.cg["mean"], n, mask_p),
                         bfdr_context_fit=realized_bfdr(cx.cc["mean"], n, mask_p),
                         cum_n_edges=int(n[cm].sum()),
                         cum_bfdr_global_fit=realized_bfdr(cx.cg["mean"], n, cm),
                         cum_bfdr_context_fit=realized_bfdr(cx.cc["mean"], n, cm),
                         n_edges_draw_lo=float(np.quantile(counts_draw, 0.025)),
                         n_edges_draw_median=float(np.median(counts_draw)),
                         n_edges_draw_hi=float(np.quantile(counts_draw, 0.975))))

    for scope, lev, cred, levd in (("global_fit", cx.lg, cx.cg["mean"], lg_d),
                                   ("context_fit", cx.lc, cx.cc["mean"], lc_d)):
        for L in (2, 1, 0):
            add(scope, LEVEL_NAMES[L], L, lev == L, cred, (n[None, :] * (levd == L)).sum(1),
                cum_mask=(lev >= L))
    for T in (3, 2, 1, 0):
        add("dual", DUAL_NAMES[T], T, cx.dual == T, joint, (n[None, :] * (dual_d == T)).sum(1),
            cum_mask=(cx.dual >= T))
    add("dual", "Promoted", 4, cx.dual >= 2, joint, (n[None, :] * (dual_d >= 2)).sum(1))
    gm, cm = cx.glob_meta, cx.cc_meta
    for r in rows:
        r["q_strong"], r["q_moderate"] = Q_STRONG, Q_MODERATE
        for pre, m in (("global_nuts", gm), ("context_nuts", cm)):
            conv = m.get("convergence", {}) if isinstance(m.get("convergence"), dict) else {}
            r[f"{pre}_chains"] = m.get("num_chains")
            r[f"{pre}_warmup"] = m.get("num_warmup")
            r[f"{pre}_samples"] = m.get("num_samples")
            r[f"{pre}_divergent"] = m.get("n_divergent")
            r[f"{pre}_r_hat_max"] = conv.get("r_hat_max")
            r[f"{pre}_r_hat_max_cred"] = conv.get("r_hat_max_cred")
            r[f"{pre}_ess_bulk_min"] = conv.get("ess_bulk_min")
    return rows


def consistency_rows(cx: Context) -> list:
    rows = []
    for src, cg, cc in (("nuts", cx.cg["mean"], cx.cc["mean"]), ("map", cx.cg_map, cx.cc_map)):
        lg, lc = bfdr_levels(cg, cx.n), bfdr_levels(cc, cx.n)
        for scope, m in (("all_edges", np.ones(len(cx.n), bool)),
                         ("either_ge_moderate", (lg >= 1) | (lc >= 1)),
                         ("observed_edges", cx.n_obs_p > 0)):
            w = cx.n * m
            r = dict(tissue=cx.t, credibility=src, scope=scope, n_edges=int(w.sum()),
                     exact=float(w[lg == lc].sum() / w.sum()),
                     within_one=float(w[np.abs(lg.astype(int) - lc) <= 1].sum() / w.sum()),
                     kappa_linear=weighted_kappa(lg[m], lc[m], cx.n[m], 3, "linear"),
                     kappa_quadratic=weighted_kappa(lg[m], lc[m], cx.n[m], 3, "quadratic"),
                     spearman_cred=weighted_spearman(cg[m], cc[m], cx.n[m]))
            for a in (2, 1, 0):
                for b in (2, 1, 0):
                    r[f"n_global{LEVEL_NAMES[a]}_context{LEVEL_NAMES[b]}"] = int(w[(lg == a) & (lc == b)].sum())
            rows.append(r)
    return rows


def predicted_frame(t: str, cx: Context) -> pd.DataFrame:
    """Score frame aligned with the context edge order (joined on tf, target)."""
    base = pd.DataFrame({"tf": cx.tf, "target": cx.target})
    sc = pd.read_parquet(C.OUT / "fm_scores" / t / "scores.parquet",
                         columns=["tf", "target"] + [m for m in METHODS + CONTROL_METHODS if m != "grnboost2"])
    gb = pd.read_parquet(C.OUT / "grnboost2" / t / "grnboost2.parquet", columns=["tf", "target", "importance"])
    gb = gb.rename(columns={"importance": "grnboost2"})
    if sc.duplicated(["tf", "target"]).any() or gb.duplicated(["tf", "target"]).any():
        raise RuntimeError("duplicated method rows")
    df = base.merge(sc, on=["tf", "target"], how="left", validate="1:1")
    df = df.merge(gb, on=["tf", "target"], how="left", validate="1:1")
    assert (df["tf"].to_numpy() == cx.tf).all() and (df["target"].to_numpy() == cx.target).all()
    return df


def promoted_by_method_rows(cx: Context, scores: pd.DataFrame) -> tuple[list, dict]:
    rows = []
    dual_e = cx.edge(cx.dual)
    prom_e = dual_e >= 2
    rep_e = cx.edge(cx.n_rep_p) > 0
    # per-TF promoted rate over U_ctx (TF-matched expectation of the promoted fraction)
    prom_rate_tf = np.bincount(cx.tf_code, weights=prom_e, minlength=cx.n_tf_universe) / \
        np.bincount(cx.tf_code, minlength=cx.n_tf_universe)
    rows.append(dict(tissue=cx.t, method="all_edges", k=np.nan, n_predicted=cx.N,
                     n_tfs=cx.n_tf_universe,
                     **{f"n_{DUAL_NAMES[T].lower()}": int((dual_e == T).sum()) for T in (3, 2, 1, 0)},
                     n_promoted=int(prom_e.sum()), frac_promoted=float(prom_e.mean()),
                     frac_promoted_lo=np.nan, frac_promoted_hi=np.nan,
                     expected_frac_promoted_tf=float(prom_e.mean()), ratio_to_tf_expected=1.0,
                     n_reported=int(rep_e.sum()), frac_reported=float(rep_e.mean())))
    pred_masks = {}
    N_t = np.bincount(cx.tf_code, minlength=cx.n_tf_universe)
    P_t = np.bincount(cx.tf_code, weights=prom_e.astype(float), minlength=cx.n_tf_universe).round().astype(np.int64)
    for m in METHODS + CONTROL_METHODS:
        for k in [TOPK] + TOPK_SENS:
            mask = topk_per_tf(scores, m, k).to_numpy()
            pred_masks[(m, k)] = mask
            tf = cx.tf[mask]
            f, lo, hi = frac_ci(tf, prom_e[mask].astype(float), cx.n_boot, stable_seed(cx.t, m, k, "frac"))
            exp_f = float(prom_rate_tf[cx.tf_code[mask]].mean()) if mask.any() else np.nan
            ex = exact_promoted_test(N_t, P_t, np.bincount(cx.tf_code[mask], minlength=cx.n_tf_universe),
                                     int(prom_e[mask].sum()))
            rows.append(dict(tissue=cx.t, method=m, k=k, n_predicted=int(mask.sum()), n_tfs=int(len(set(tf))),
                             **{f"n_{DUAL_NAMES[T].lower()}": int((dual_e[mask] == T).sum()) for T in (3, 2, 1, 0)},
                             n_promoted=int(prom_e[mask].sum()), frac_promoted=f,
                             frac_promoted_lo=lo, frac_promoted_hi=hi,
                             expected_frac_promoted_tf=exp_f,
                             ratio_to_tf_expected=(f / exp_f) if exp_f and exp_f > 0 else np.nan,
                             n_reported=int(rep_e[mask].sum()),
                             frac_reported=float(rep_e[mask].mean()) if mask.any() else np.nan,
                             expected_promoted_tf=ex["expected"], p_depleted_exact=ex["p_le"],
                             p_enriched_exact=ex["p_ge"],
                             ci_excludes_tf_expected=(bool(hi < exp_f or lo > exp_f)
                                                      if np.isfinite(lo) and np.isfinite(exp_f) else np.nan),
                             analysis_status="prespecified",
                             note_post_hoc=("expected_promoted_tf, p_*_exact and ci_excludes_tf_expected are post hoc "
                                            "(2026-10-08): exact TF-stratified hypergeometric test of the promoted "
                                            "count (method top-k as a random within-TF subset)")))
    return rows, pred_masks


def exact_promoted_test(N_t, P_t, m_t, obs: int) -> dict:
    """Promoted count of a predicted set under the TF-matched null: within TF t the set's m_t edges
    are a uniformly random subset of the TF's N_t universe edges, P_t of which are promoted
    (X_t ~ Hypergeometric(N_t, P_t, m_t), T = sum X_t, exact convolution).  Returns the expectation
    and the one-sided exact p of depletion (P(T <= obs)) and enrichment (P(T >= obs))."""
    from r2 import replication as RP
    N_t, P_t, m_t = (np.asarray(x, dtype=np.int64) for x in (N_t, P_t, m_t))
    use = m_t > 0
    N_t, P_t, m_t = N_t[use], P_t[use], m_t[use]
    expct = float(np.sum(m_t * P_t / np.maximum(N_t, 1)))
    if len(N_t) == 0:
        return dict(expected=0.0, p_le=np.nan, p_ge=np.nan)
    shift, pmf = RP.sum_distribution(N_t, P_t, m_t, 0.0)
    i = int(obs) - shift
    cdf = np.cumsum(pmf)
    p_le = float(cdf[i]) if 0 <= i < len(pmf) else (0.0 if i < 0 else 1.0)
    p_ge = float(pmf[max(i, 0):].sum()) if i < len(pmf) else 0.0
    return dict(expected=expct, p_le=min(1.0, p_le), p_ge=min(1.0, p_ge))


def validation_rows(cx: Context, global_oe: dict | None = None) -> tuple[list, list]:
    """Tier O/E per reference (+ post hoc power columns for Strong / Moderate / Promoted rows, with
    the global-universe O/E of the same tier, ``global_oe[(tier, reference)]``, as the H1)."""
    rows, trend = [], []
    global_oe = global_oe or {}
    dual_e = cx.edge(cx.dual)
    lev = {"global_fit": cx.edge(cx.lg), "context_fit": cx.edge(cx.lc)}

    def add(scope, tier, order, r, mask, tag):
        row = dict(universe=cx.t, scope=scope, tier=tier, tier_order=order, reference=REF_LABEL[r],
                   **cx.set_eval(mask, r, tag))
        if tier in MDE_TIERS:
            g = global_oe.get((GLOBAL_TIER_FOR[tier], REF_LABEL[r]))
            row.update(cx.mde_cols(mask, r, g))
            row["ci_excludes_global_oe"] = (bool(row["oe_tf_hi"] < g) if (g is not None and np.isfinite(row["oe_tf_hi"]))
                                            else np.nan)
        rows.append(row)

    for r in REFS:
        for T in (3, 2, 1, 0):
            add("dual", DUAL_NAMES[T], T, r, dual_e == T, f"dual{T}")
        add("dual", "Promoted", 4, r, dual_e >= 2, "promoted")
        add("dual", "Not promoted", -1, r, dual_e < 2, "notpromoted")
        for scope, lv in lev.items():
            for L in (2, 1, 0):
                add(scope, LEVEL_NAMES[L], L, r, lv == L, f"{scope}{L}")
        L = cx.labels[r]
        e = L["evaluable"].to_numpy()
        y = L["label"].to_numpy()[e]
        for scope, x in (("dual", dual_e), *lev.items()):
            for strat in ("tf", "tf_x_degree"):
                s = cx.tf[e] if strat == "tf" else pd.Series(cx.tf[e]).astype(str).to_numpy() + "|" + \
                    L["deg_bin"].to_numpy()[e].astype(str)
                mt = mantel_trend(x[e], y, s)
                trend.append(dict(universe=cx.t, scope=scope, reference=REF_LABEL[r], strata=strat,
                                  n_eval=int(e.sum()), n_pos=int(y.sum()),
                                  n_eval_above_bottom=int((x[e] > 0).sum()),
                                  n_pos_above_bottom=int(y[x[e] > 0].sum()), **mt))
    return rows, trend


def concentration_rows(cx: Context, pred_masks: dict, extra: dict) -> list:
    rows = []
    dual_e = cx.edge(cx.dual)
    sets = {
        "eligible_edges": np.ones(cx.N, bool),
        "observed_edges": cx.edge(cx.n_obs_p) > 0,
        "reported_edges": cx.edge(cx.n_rep_p) > 0,
        "global_fit_strong_or_moderate": cx.edge(cx.lg) >= 1,
        "context_fit_strong_or_moderate": cx.edge(cx.lc) >= 1,
        "dual_strong": dual_e == 3,
        "dual_moderate": dual_e == 2,
        "dual_conditional": dual_e == 1,
        "promoted": dual_e >= 2,
    }
    for m in METHODS + CONTROL_METHODS:
        sets[f"predicted_{m}_top{TOPK}"] = pred_masks[(m, TOPK)]
        sets[f"promoted_in_{m}_top{TOPK}"] = pred_masks[(m, TOPK)] & (dual_e >= 2)
    sets.update(extra)
    for name, m in sets.items():
        rows.append(dict(tissue=cx.t, set=name, **concentration(cx.tf[m], cx.n_tf_universe)))
    return rows


def deconcentration(cx: Context) -> tuple[list, list, dict]:
    dual_e = cx.edge(cx.dual)
    pool = np.flatnonzero(dual_e >= 2)
    if len(pool) == 0:
        return [], [], {}
    s = np.minimum(cx.edge(cx.cg["mean"]), cx.edge(cx.cc["mean"]))[pool]
    tfp = cx.tf[pool]
    codes, uniq = pd.factorize(tfp, sort=True)
    keys = tie_break_keys(tfp, cx.target[pool], "governance_deconcentration", int(P["topk_tie_seed"]))
    k_pool = len(uniq)
    grid = []
    sel_cache = {}
    for f in DECONC_BUDGET_FRACTIONS:
        B = max(1, int(round(f * len(pool))))
        for a in DECONC_ALPHAS:
            m = deconcentrate(codes, s, keys, B, a)
            sel_cache[(B, a)] = m
            conc = concentration(tfp[m], cx.n_tf_universe)
            grid.append(dict(tissue=cx.t, pool_size=len(pool), n_tfs_pool=k_pool, budget_fraction=f, budget=B,
                             alpha=a, n_selected=int(m.sum()), mean_cred=float(s[m].mean()),
                             expected_true=float(s[m].sum()), n_tfs=conc["n_tfs"],
                             top1_share=conc["top1_share"], top2_share=conc["top2_share"],
                             evenness_pielou=conc["evenness"],
                             diversity=conc["entropy"] / math.log(k_pool) if k_pool > 1 else np.nan))
    g = pd.DataFrame(grid)

    def mm(x):
        lo, hi = np.nanmin(x), np.nanmax(x)
        return (x - lo) / (hi - lo) if hi > lo else np.ones_like(x)

    g["precision_norm"] = mm(g["mean_cred"].to_numpy())
    g["diversity_norm"] = mm(g["diversity"].fillna(0).to_numpy())
    a, b = g["precision_norm"].to_numpy(), g["diversity_norm"].to_numpy()
    with np.errstate(invalid="ignore", divide="ignore"):
        g["harmonic_mean"] = np.where(a + b > 0, 2 * a * b / (a + b), 0.0)
    best = g.sort_values(["harmonic_mean", "budget", "alpha"], ascending=[False, False, True],
                         kind="mergesort").index[0]
    g["chosen"] = False
    g.loc[best, "chosen"] = True
    # sensitivity (post hoc, verifier minor 6): Pielou evenness H / ln(n TFs of the set) as the diversity axis
    g["pielou_norm"] = mm(g["evenness_pielou"].fillna(0).to_numpy())
    bp = g["pielou_norm"].to_numpy()
    with np.errstate(invalid="ignore", divide="ignore"):
        g["harmonic_mean_pielou"] = np.where(a + bp > 0, 2 * a * bp / (a + bp), 0.0)
    best_p = g.sort_values(["harmonic_mean_pielou", "budget", "alpha"], ascending=[False, False, True],
                           kind="mergesort").index[0]
    g["chosen_pielou"] = False
    g.loc[best_p, "chosen_pielou"] = True
    g["analysis_status"] = "prespecified"
    g["note"] = "chosen = primary label-free rule; chosen_pielou / *_pielou columns = post hoc sensitivity"
    Bc, ac = int(g.loc[best, "budget"]), float(g.loc[best, "alpha"])
    Bp, ap_ = int(g.loc[best_p, "budget"]), float(g.loc[best_p, "alpha"])
    chosen_m = np.zeros(cx.N, bool)
    chosen_m[pool[sel_cache[(Bc, ac)]]] = True
    undec_m = np.zeros(cx.N, bool)
    undec_m[pool[sel_cache[(Bc, 0.0)]]] = True
    pielou_m = np.zeros(cx.N, bool)
    pielou_m[pool[sel_cache[(Bp, ap_)]]] = True
    undec_p = np.zeros(cx.N, bool)
    undec_p[pool[sel_cache[(Bp, 0.0)]]] = True
    pool_m = dual_e >= 2
    ev = []
    for name, m, alpha_, status in (("chosen", chosen_m, ac, "prespecified"),
                                    ("undeconcentrated_same_budget", undec_m, 0.0, "prespecified"),
                                    ("pool", pool_m, 0.0, "prespecified"),
                                    ("chosen_pielou", pielou_m, ap_, "post_hoc"),
                                    ("undeconcentrated_same_budget_pielou", undec_p, 0.0, "post_hoc")):
        conc = concentration(cx.tf[m], cx.n_tf_universe)
        jm = np.minimum(cx.edge(cx.cg["mean"]), cx.edge(cx.cc["mean"]))[m]
        for r in REFS:
            ev.append(dict(tissue=cx.t, set=name, budget=int(m.sum()), alpha=alpha_,
                           mean_cred=float(jm.mean()) if m.any() else np.nan,
                           n_tfs=conc["n_tfs"], top1_share=conc["top1_share"], diversity=conc["entropy"] / math.log(k_pool)
                           if k_pool > 1 and conc["n_edges"] else np.nan, evenness_pielou=conc["evenness"],
                           pool_fraction=float(m.sum() / max(pool_m.sum(), 1)),
                           reference=REF_LABEL[r], **{k: v for k, v in cx.set_eval(m, r, f"deconc_{name}").items()
                                                      if k != "n_edges"}, analysis_status=status))
    extra = {"deconcentrated_chosen": chosen_m, "undeconcentrated_same_budget": undec_m,
             "deconcentrated_chosen_pielou": pielou_m}
    return g.to_dict("records"), ev, extra


def sensitivity_rows(cx: Context, ib: pd.DataFrame) -> list:
    rows = []
    n = cx.n
    prim = cx.edge(cx.dual) >= 2
    configs = []

    def dual_cfg(fam, cfg, qs, qm, rule, lg, lc):
        d = cx.edge(dual_tier(lg, lc, rule))
        configs.append((fam, cfg, qs, qm, rule, d >= 2, int((d == 3).sum())))

    for qs in Q_GRID_STRONG:
        for qm in Q_GRID_MODERATE:
            if qs < qm:
                dual_cfg("q_grid", f"qs{qs:g}_qm{qm:g}", qs, qm, "primary",
                         bfdr_levels(cx.cg["mean"], n, qs, qm), bfdr_levels(cx.cc["mean"], n, qs, qm))
    for rule in ("primary", "lenient", "strict"):
        dual_cfg("dual_rule", rule, Q_STRONG, Q_MODERATE, rule, cx.lg, cx.lc)
    for name, lev in (("global_fit_only", cx.lg), ("context_fit_only", cx.lc)):
        le = cx.edge(lev)
        configs.append(("single_universe", name, Q_STRONG, Q_MODERATE, "single", le >= 1, int((le == 2).sum())))
        configs.append(("single_universe", f"{name}_strong", Q_STRONG, Q_MODERATE, "single", le >= 2,
                        int((le == 2).sum())))
    k1 = R1_K1 / R1_UNIVERSE * cx.N
    k2 = R1_K2 / R1_UNIVERSE * cx.N
    dual_cfg("rank_budget", f"k1_{k1:.0f}_k2_{k2:.0f}", np.nan, np.nan, "primary",
             rank_budget_levels(cx.cg["mean"], n, k1, k2), rank_budget_levels(cx.cc["mean"], n, k1, k2))
    dual_cfg("credibility", "nuts_mean_primary", Q_STRONG, Q_MODERATE, "primary", cx.lg, cx.lc)
    dual_cfg("credibility", "map", Q_STRONG, Q_MODERATE, "primary",
             bfdr_levels(cx.cg_map, n), bfdr_levels(cx.cc_map, n))
    dual_cfg("estimator", "em_map", Q_STRONG, Q_MODERATE, "primary",
             bfdr_levels(cx.cg_em, n), bfdr_levels(cx.cc_em, n))
    nrep = cx.edge(cx.n_rep_p)
    b = pd.DataFrame({"tf": cx.tf, "target": cx.target}).merge(
        ib[["tf", "target", "dorothea_level_heldin", "in_collectri_heldin"]], on=["tf", "target"], how="left",
        validate="1:1")
    for name, m in (("majority_vote_ge2", nrep >= 2), ("union_ge1", nrep >= 1),
                    ("dorothea_heldin_AB", b["dorothea_level_heldin"].isin(["A", "B"]).to_numpy()),
                    ("collectri_heldin", b["in_collectri_heldin"].astype("boolean").fillna(False)
                     .astype(bool).to_numpy())):
        configs.append(("baseline_rule", name, np.nan, np.nan, "baseline", m, np.nan))
    # size-matched comparisons (post hoc): the top-|primary| edges under each alternative ranking
    n_prim = int(prim.sum())
    in_col = b["in_collectri_heldin"].astype("boolean").fillna(False).astype(bool).to_numpy()
    effort = pd.to_numeric(b["collectri_curation_effort_heldin"], errors="coerce").to_numpy(np.float64) \
        if "collectri_curation_effort_heldin" in b else np.full(cx.N, np.nan)
    rank_scores = {
        "joint_nuts": np.minimum(cx.edge(cx.cg["mean"]), cx.edge(cx.cc["mean"])),
        "joint_map": np.minimum(cx.edge(cx.cg_map), cx.edge(cx.cc_map)),
        "joint_em": np.minimum(cx.edge(cx.cg_em), cx.edge(cx.cc_em)),
        "global_fit_nuts": cx.edge(cx.cg["mean"]),
        "context_fit_nuts": cx.edge(cx.cc["mean"]),
        "n_reporting": nrep.astype(np.float64),
        "dorothea_heldin_level": b["dorothea_level_heldin"].map({"A": 4.0, "B": 3.0, "C": 2.0, "D": 1.0})
        .fillna(0.0).to_numpy(np.float64),
        "collectri_heldin_effort": np.where(in_col, np.where(np.isfinite(effort), np.maximum(effort, 0.5), 0.5), 0.0),
    }
    keys = tie_break_keys(cx.tf, cx.target, "governance_size_matched", int(P["topk_tie_seed"]))
    for name, sc in rank_scores.items():
        configs.append(("size_matched", f"{name}_top{n_prim}", np.nan, np.nan, "size_matched",
                        top_n_mask(sc, n_prim, keys), np.nan))
    for fam, cfg, qs, qm, rule, m, n_strong in configs:
        status = "post_hoc" if fam == "size_matched" else "prespecified"
        conc = concentration(cx.tf[m], cx.n_tf_universe)
        inter, union = int((m & prim).sum()), int((m | prim).sum())
        for r in REFS:
            ev = cx.set_eval(m, r, f"sens_{fam}_{cfg}")
            rows.append(dict(tissue=cx.t, family=fam, config=cfg, q_strong=qs, q_moderate=qm, dual_rule=rule,
                             n_promoted=int(m.sum()), n_strong=n_strong, n_shared_with_primary=inter,
                             jaccard_vs_primary=inter / union if union else np.nan,
                             n_tfs=conc["n_tfs"], top1_tf=conc["top1_tf"], top1_share=conc["top1_share"],
                             top2_share=conc["top2_share"], evenness=conc["evenness"],
                             reference=REF_LABEL[r],
                             **{k: ev[k] for k in ("n_eval", "n_tfs_eval", "few_clusters", "obs", "exp_tf", "oe_tf",
                                                   "oe_tf_lo", "oe_tf_hi", "exp_tfdeg", "oe_tfdeg", "oe_tfdeg_lo",
                                                   "oe_tfdeg_hi", "marginal_enrichment")},
                             analysis_status=status))
    return rows


def top_n_mask(score: np.ndarray, n_top: int, keys: np.ndarray) -> np.ndarray:
    """The n_top elements with the highest score (rounded to TIE_DECIMALS; NaN last); ties at the
    boundary are broken by the seeded per-edge keys (larger key first)."""
    sc = np.asarray(score, dtype=np.float64)
    v = np.where(np.isnan(sc), -np.inf, np.round(sc, TIE_DECIMALS))
    o = np.lexsort((~np.asarray(keys, dtype=np.uint64), -v))
    m = np.zeros(len(sc), dtype=bool)
    m[o[:int(min(max(n_top, 0), len(sc)))]] = True
    return m


def edge_tier_frame(cx: Context, pred_masks: dict) -> pd.DataFrame:
    df = pd.DataFrame({"tf": cx.tf, "target": cx.target, "pattern": cx.ep.astype(np.int32),
                       "in_global_universe": cx.in_global,
                       "n_observing": cx.edge(cx.n_obs_p).astype(np.int16),
                       "n_reporting": cx.edge(cx.n_rep_p).astype(np.int16)})
    for k in ("mean", "lo", "hi"):
        df[f"cred_global_{k}"] = cx.edge(cx.cg[k])
    for k in ("mean", "lo", "hi"):
        df[f"cred_context_{k}"] = cx.edge(cx.cc[k])
    df["cred_global_map"] = cx.edge(cx.cg_map)
    df["cred_context_map"] = cx.edge(cx.cc_map)
    df["level_global"] = cx.edge(cx.lg)
    df["level_context"] = cx.edge(cx.lc)
    df["dual_tier"] = pd.Series(cx.edge(cx.dual)).map(DUAL_NAMES).to_numpy()
    df["promoted"] = cx.edge(cx.dual) >= 2
    df["p_promoted_draws"] = cx.edge(cx.p_promoted_pattern)
    for r in REFS:
        df[f"label_{REF_LABEL[r]}"] = cx.labels[r]["label"].to_numpy()
    preds = np.full(cx.N, "", dtype=object)
    for m in METHODS:
        mk = pred_masks[(m, TOPK)]
        preds[mk] = np.where(preds[mk] == "", m, preds[mk] + ";" + m)
    df[f"predicted_by_top{TOPK}"] = preds
    return df


# ----------------------------------------------------------------------------
# Global universe (supplementary tier validation)
# ----------------------------------------------------------------------------
def global_validation(glob: dict, pooled: pd.DataFrame, n_boot: int) -> tuple[list, list, dict]:
    rows, trend = [], []
    lev_p = bfdr_levels(glob["nuts"]["mean"], glob["n"])
    G = len(glob["genes"])
    gi = {g: i for i, g in enumerate(glob["genes"])}
    ti = {t: i for i, t in enumerate(glob["tfs"])}
    indices = build_reference_indices(glob["genes"], pooled)
    info = {}
    for r, (R, masked) in indices.items():
        tfs = [t for t in R["tfs"] if t in ti]
        if masked:
            sub = pooled[pooled["tf"].isin(tfs) & pooled["target"].isin(gi) & (pooled["tf"] != pooled["target"])]
            tf_arr, tg_arr = sub["tf"].to_numpy(object), sub["target"].to_numpy(object)
        else:
            genes = sorted(g for g in R["universe"] if g in gi)   # set -> fixed order (seeds)
            tf_arr = np.repeat(np.array(tfs, dtype=object), len(genes))
            tg_arr = np.tile(np.array(genes, dtype=object), len(tfs))
        L = label_frame_from_index(R, tf_arr, tg_arr, masked)
        e = L["evaluable"].to_numpy()
        tf_e, tg_e = tf_arr[e], tg_arr[e]
        idx = pd.Series(tf_e).map(ti).to_numpy(np.int64) * G + pd.Series(tg_e).map(gi).to_numpy(np.int64)
        lev = lev_p[np.asarray(glob["ep"][idx], dtype=np.int64)]
        y = L["label"].to_numpy()[e]
        et, ed = L["exp_tf"].to_numpy()[e], L["exp_deg"].to_numpy()[e]
        base = float(y.mean()) if len(y) else np.nan
        info[REF_LABEL[r]] = dict(n_eval=int(len(y)), n_pos=int(y.sum()), n_tfs=int(len(set(tf_e))))
        tcode, tuniq = pd.factorize(tf_e, sort=True)
        N_t = np.bincount(tcode, minlength=len(tuniq))
        K_t = np.bincount(tcode, weights=y, minlength=len(tuniq)).round().astype(np.int64)
        for Lv, tier, order in ((2, "Strong", 2), (1, "Moderate", 1), (0, "Below", 0), (None, "Promoted", 4)):
            m = (lev >= 1) if Lv is None else (lev == Lv)
            dig = set_digest(m)
            a = oe_ci(tf_e[m], y[m], et[m], n_boot, stable_seed("global", r, dig, "tf"))
            b = oe_ci(tf_e[m], y[m], ed[m], n_boot, stable_seed("global", r, dig, "deg"))
            row = dict(universe="global", scope="global_fit", tier=tier, tier_order=order,
                       reference=REF_LABEL[r],
                       n_edges=int(glob["n"][(lev_p >= 1) if Lv is None else (lev_p == Lv)].sum()),
                       n_eval=int(m.sum()), n_tfs_eval=int(len(set(tf_e[m]))),
                       few_clusters=bool(len(set(tf_e[m])) < MIN_CLUSTERS), **top_tf_share(tf_e[m]),
                       obs=a["obs"], exp_tf=a["exp"], oe_tf=a["oe"], oe_tf_lo=a["lo"], oe_tf_hi=a["hi"],
                       exp_tfdeg=b["exp"], oe_tfdeg=b["oe"], oe_tfdeg_lo=b["lo"], oe_tfdeg_hi=b["hi"],
                       base_rate=base,
                       marginal_enrichment=(a["obs"] / m.sum() / base) if m.sum() and base > 0 else np.nan,
                       analysis_status="post_hoc" if Lv is None else "prespecified")
            if tier in MDE_TIERS:
                n_t = np.bincount(tcode[m], minlength=len(tuniq))
                row.update(power_columns(oe_power_analysis(N_t, K_t, n_t), None))
            rows.append(row)
        for strat in ("tf", "tf_x_degree"):
            s = tf_e if strat == "tf" else pd.Series(tf_e).astype(str).to_numpy() + "|" + \
                L["deg_bin"].to_numpy()[e].astype(str)
            trend.append(dict(universe="global", scope="global_fit", reference=REF_LABEL[r], strata=strat,
                              n_eval=int(len(y)), n_pos=int(y.sum()), n_eval_above_bottom=int((lev > 0).sum()),
                              n_pos_above_bottom=int(y[lev > 0].sum()), **mantel_trend(lev, y, s)))
    return rows, trend, info


# ----------------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------------
def write_table(df: pd.DataFrame, name: str) -> Path:
    """Write a manuscript table; every table gets ``analysis_status`` (default prespecified) and
    ``post_hoc_columns`` (``POST_HOC_COLUMNS``: columns of prespecified rows that were added after
    held-out results existed)."""
    df = df.copy()
    if "analysis_status" not in df.columns:
        df["analysis_status"] = "prespecified"
    df["post_hoc_columns"] = POST_HOC_COLUMNS.get(name, "")
    p = C.ensure_dir(C.TABLES) / name
    df.to_csv(p, index=False, float_format="%.6g")
    log(f"  wrote {p} ({len(df)} rows)")
    return p


def run_all(tissues, do_global: bool = True, n_boot: int = N_BOOT) -> dict:
    t0 = time.time()
    C.ensure_dir(GOV)
    (GOV / "run.log").write_text("")
    log(f"governance: tissues={list(tissues)} global={do_global} n_boot={n_boot} variant={VARIANT}")
    glob = load_global()
    log(f"global fit loaded: {len(glob['R'])} patterns, NUTS r_hat_max "
        f"{glob['nuts_meta']['convergence']['r_hat_max']:.4f}, divergences {glob['nuts_meta']['n_divergent']}")
    pooled = load_pert_pooled()
    C.ensure_dir(GOV / "labels")
    pooled.to_parquet(GOV / "labels" / "pert_pooled_pairs.parquet", index=False)
    log(f"pert_pooled: {pooled['tf'].nunique()} TFs, {len(pooled):,} evaluable pairs, "
        f"{int(pooled['positive'].sum()):,} positives")
    ib = pd.read_parquet(C.OUT / "annotators" / "integration_baselines.parquet",
                         columns=["tf", "target", "dorothea_level_heldin", "in_collectri_heldin",
                                  "collectri_curation_effort_heldin"])
    out = {k: [] for k in ("tiers", "consistency", "by_method", "validation", "trend", "concentration",
                           "deconc", "deconc_eval", "sensitivity", "promoted")}
    meta = dict(variant=VARIANT, primary_fit=PRIMARY_FIT, q_strong=Q_STRONG, q_moderate=Q_MODERATE,
                n_boot=n_boot, seed=SEED, topk=TOPK, methods=list(METHODS), control_methods=list(CONTROL_METHODS),
                deconc_budget_fractions=list(DECONC_BUDGET_FRACTIONS), deconc_alphas=list(DECONC_ALPHAS),
                q_grid_strong=list(Q_GRID_STRONG), q_grid_moderate=list(Q_GRID_MODERATE),
                rank_budget=dict(r1_universe=R1_UNIVERSE, k1=R1_K1, k2=R1_K2),
                analysis_common_available=AC is not None, tissues={})
    global_oe = {}
    if do_global:
        log("== global universe (supplementary tier validation; first, as the H1 of the context power analysis)")
        v, tr, info = global_validation(glob, pooled, n_boot)
        out["validation"] += v
        out["trend"] += tr
        meta["global_labels"] = info
        global_oe = {(r["tier"], r["reference"]): r["oe_tf"] for r in v}
    for t in tissues:
        log(f"== {t}")
        cx = Context(t, glob, n_boot)
        panel = sorted(set(cx.target))
        indices = build_reference_indices(panel, pooled)
        cx.attach_labels(indices)
        for r in REFS:
            L = cx.labels[r]
            e = L["evaluable"].to_numpy()
            sub = pd.DataFrame({"tf": cx.tf[e], "target": cx.target[e], **{c: L[c].to_numpy()[e]
                                                                          for c in ("label", "exp_tf", "exp_deg", "deg_bin")}})
            sub.to_parquet(GOV / "labels" / f"{t}__{REF_LABEL[r]}.parquet", index=False)
        log(f"  tiers: global-fit S/M {int(cx.n[cx.lg == 2].sum())}/{int(cx.n[cx.lg == 1].sum())}, "
            f"context-fit S/M {int(cx.n[cx.lc == 2].sum())}/{int(cx.n[cx.lc == 1].sum())}, "
            f"promoted {int(cx.n[cx.dual >= 2].sum())}")
        out["tiers"] += tier_table(cx)
        out["consistency"] += consistency_rows(cx)
        scores = predicted_frame(t, cx)
        bm, pred_masks = promoted_by_method_rows(cx, scores)
        out["by_method"] += bm
        v, tr = validation_rows(cx, global_oe)
        out["validation"] += v
        out["trend"] += tr
        grid, dev, extra = deconcentration(cx)
        out["deconc"] += grid
        out["deconc_eval"] += dev
        out["concentration"] += concentration_rows(cx, pred_masks, extra)
        out["sensitivity"] += sensitivity_rows(cx, ib)
        ef = edge_tier_frame(cx, pred_masks)
        C.ensure_dir(GOV / t)
        ef.to_parquet(GOV / t / "edge_tiers.parquet", index=False)
        pe = ef[ef["promoted"]].copy()
        pe.insert(0, "tissue", t)
        out["promoted"].append(pe)
        meta["tissues"][t] = dict(
            n_edges=cx.N, n_patterns=len(cx.n), n_tfs=cx.n_tf_universe,
            n_edges_outside_global_universe=int((~cx.in_global).sum()),
            n_tfs_outside_global_universe=int(len(set(cx.tf[~cx.in_global]))),
            pattern_identity_check="passed", n_draws=int(len(cx.draws_g)),
            context_nuts=dict(r_hat_max=cx.cc_meta["convergence"]["r_hat_max"],
                              n_divergent=cx.cc_meta["n_divergent"]),
            rank_budget_k=[R1_K1 / R1_UNIVERSE * cx.N, R1_K2 / R1_UNIVERSE * cx.N],
            labels={REF_LABEL[r]: dict(n_eval=int(cx.labels[r]["evaluable"].sum()),
                                       n_pos=int(np.nansum(cx.labels[r]["label"])),
                                       n_tfs=int(len(set(cx.tf[cx.labels[r]["evaluable"].to_numpy()]))))
                    for r in REFS})
    # Holm within (scope, strata) across universes x references
    trend = pd.DataFrame(out["trend"])
    trend["p_one_sided_holm"] = np.nan
    trend["p_two_sided_holm"] = np.nan
    trend["holm_family_size"] = np.nan
    for _, idx in trend.groupby(["scope", "strata"]).groups.items():
        trend.loc[idx, "p_one_sided_holm"] = holm(trend.loc[idx, "p_one_sided"].to_numpy())
        trend.loc[idx, "p_two_sided_holm"] = holm(trend.loc[idx, "p_two_sided"].to_numpy())
        trend.loc[idx, "holm_family_size"] = int(np.isfinite(trend.loc[idx, "p_one_sided"].to_numpy()).sum())
    trend["holm_family"] = ("Holm within (scope, strata) across all universes (3 tissues + global) x 4 references; "
                            "p_one_sided_holm = one-sided test for an increasing trend (reported), p_two_sided_holm "
                            "= two-sided")
    trend["analysis_status"] = "prespecified"
    tiers = pd.DataFrame(out["tiers"])
    tiers["analysis_status"] = "prespecified"
    write_table(tiers, "gov_tiers.csv")
    write_table(pd.DataFrame(out["consistency"]).assign(analysis_status="prespecified"), "gov_consistency.csv")
    bm = pd.DataFrame(out["by_method"])
    bm["analysis_status"] = bm["analysis_status"].fillna("prespecified")
    write_table(bm, "gov_promoted_by_method.csv")
    val = pd.DataFrame(out["validation"])
    val["analysis_status"] = val["analysis_status"].fillna("prespecified") if "analysis_status" in val else "prespecified"
    write_table(val, "gov_tier_validation.csv")
    write_table(trend, "gov_tier_trend.csv")
    conc = pd.DataFrame(out["concentration"])
    conc["analysis_status"] = np.where(conc["set"] == "deconcentrated_chosen_pielou", "post_hoc", "prespecified")
    write_table(conc, "gov_concentration.csv")
    write_table(pd.DataFrame(out["deconc"]), "gov_deconcentration.csv")
    write_table(pd.DataFrame(out["deconc_eval"]), "gov_deconcentration_eval.csv")
    write_table(pd.DataFrame(out["sensitivity"]), "gov_sensitivity_grid.csv")
    pe = pd.concat(out["promoted"], ignore_index=True)
    cols = ["tissue", "tf", "target", "dual_tier", "level_global", "level_context",
            "cred_global_mean", "cred_global_lo", "cred_global_hi",
            "cred_context_mean", "cred_context_lo", "cred_context_hi",
            "cred_global_map", "cred_context_map", "p_promoted_draws", "n_observing", "n_reporting",
            *[f"label_{REF_LABEL[r]}" for r in REFS], f"predicted_by_top{TOPK}"]
    pe = pe[cols].sort_values(["tissue", "dual_tier", "cred_context_mean", "tf", "target"],
                              ascending=[True, False, False, True, True], kind="mergesort")
    write_table(pe, "gov_promoted_edges.csv")
    meta["runtime_s"] = time.time() - t0
    meta["peak_rss_gb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9   # bytes on macOS
    meta["versions"] = dict(numpy=np.__version__, pandas=pd.__version__)
    meta["analysis_decisions"] = ANALYSIS_DECISIONS
    C.write_json(meta, GOV / "meta.json")
    record_decisions("governance", ANALYSIS_DECISIONS)
    log(f"done in {meta['runtime_s']:.0f} s")
    return meta


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


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("all", help="compute every governance table")
    a.add_argument("--tissues", nargs="+", default=list(C.TISSUES), choices=list(C.TISSUES))
    a.add_argument("--no-global", action="store_true", help="skip the global-universe validation")
    a.add_argument("--n-boot", type=int, default=N_BOOT)
    args = ap.parse_args(argv)
    if args.cmd == "all":
        run_all(args.tissues, do_global=not args.no_global, n_boot=args.n_boot)


if __name__ == "__main__":
    main()
