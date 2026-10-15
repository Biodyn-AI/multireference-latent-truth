"""Held-out evaluation of credibility estimators, baselines and GRN-inference methods
(evaluation stages E2/E3/E6 of the R2 plan; reviewer Q2, Q4 iii, Q9 (empirical), Q10, Q13,
Q16 and Q20).

Nothing in this module is fitted, tuned or selected on held-out data: every fit, score and
universe is read from the label-free stages (``fits``, ``evidence``, ``fm_scores``,
``grnboost2``, ``annotators``); the held-out ChIP references (``heldout_chip``) and the
perturbation cohorts (``cohorts``) are used only to *evaluate*.

Universes and references
------------------------
Universes: ``global`` (primary for the estimator comparison: 935 literature-covered TFs x
9,963 genes) and the tissue context universes ``immune``, ``kidney``, ``lung``.

References (``REFERENCES``): ``encode`` (ENCODE TFBS, 5-cell-line union; permissive),
``encode_strict`` (>= 2 cell lines), ``enets2`` (Gerstein 2012 proximal network; specific),
every perturbation cohort (``cohort_table.csv``) and ``pert_pooled``.

An edge (tf, g) of universe U is *evaluable* for

* a ChIP reference if tf is an assayed TF of the reference and g lies in the reference gene
  universe (``heldout_chip/<ref>_{assayed_tfs,gene_universe}.txt``), tf != g; positive iff
  (tf, g) is a reference edge.  Every such pair is evaluable (rectangular grid).
* a perturbation cohort if the pair was tested (a row of ``cohorts/<c>/de.parquet`` with
  ``perturbed_gene`` = tf, ``target`` = g) and is not a same-cytoband cis pair
  (``cis_same_band``; removed from positives AND negatives, decision log item 7); positive
  iff ``positive_primary`` (BH FDR < 0.05 and |log2FC| > 0.25 of the cohort's primary DE).
* ``pert_pooled`` if it is evaluable in >= 1 cohort; positive iff it is positive in >= 1
  cohort in which it is evaluable ("any-cohort" rule).  Cohorts without power (Dixit, flagged
  in ``cohort_table.csv``) are included exactly as in the per-cohort rows; the pooled
  reference is clustered by TF like every other reference.

Pairs outside U are ignored; external tables are always joined on (tf, target) keys.

Scores
------
* the five MAP estimators of ``lcm.py`` (``em``, ``dependence``, ``multifactor``,
  ``family_regularized``, ``family_hb``) and the Family-HB NUTS posterior mean
  (``family_hb_nuts``), all fitted on the universe being evaluated (``fits/<U>/<variant>``);
  in the tissue universes additionally ``family_hb_globalfit`` = credibility of the edge's
  evidence pattern under the *global* Family-HB MAP parameters (``lcm.credibility_at``; the
  pattern of an edge does not depend on the universe, see ``evidence.py``);
* pattern baselines ``lcm.baseline_scores``: union, majority, vote_count, vote_fraction,
  n_families_reporting, fisher (-log p), stouffer (Z);
* integration baselines from held-in evidence only (decision log item 3/9):
  ``dorothea_level_heldin`` (A=4 > B=3 > C=2 > D=1 > not in held-in DoRothEA = 0) and
  ``collectri_heldin`` (held-in CollecTRI curation effort; 0 = not in held-in CollecTRI);
* ``random``: global universe a seeded uniform integer in [0, 10^4) per edge of the TF-major
  grid (ties resolved in expectation; 10^4 levels keep the 1M-edge bootstrap/permutation cheap);
  tissue universes the ``random`` column of ``fm_scores`` (the methods' random baseline);
* tissue universes only, the GRN-inference methods (``METHODS``): scGPT / Geneformer
  attention variants incl. the 1/L uniform-attention null, GRNBoost2, GENIE3 (immune),
  Pearson / Spearman (signed and absolute).  Missing / NaN = not predicted (ranks last).

Ties: scores are rounded to ``TIE_DECIMALS`` = 12 decimals; equal values form a tie block,
NaN and -inf form the bottom block (``PRESPECIFIED['nan_rank'] = 'last'``).

Metrics (per universe x reference x score; ``curve_metrics``)
------------------------------------------------------------
Over the N evaluable edges with P positives (base rate b = P / N), scores sorted in
decreasing tie blocks l with t_l edges, p_l positives, cumulative T_l, P_l:

* ``ap``       average precision with whole tie blocks as thresholds
               (sum_l p_l / P * P_l / T_l, the scikit-learn convention); ``ap_lift`` = ap / b;
* ``p_at_k``   precision of the top k edges, a tie block cut by the k-th position contributes
               its positive *rate* (= expectation over uniformly random tie-breaking);
               ``enr_at_k`` = p_at_k / b, for k in ``PRESPECIFIED['eval_k']`` (NaN if k > N);
* ``strat_enr_at_k`` = positives in the top k / expected positives in the top k under the
               reference's TF x target-degree null expit(a_TF + b_bin(d_-TF(g))) (prespecified
               ``PRESPECIFIED['target_degree_*']``).  ChIP: ``build_heldout_chip.reference_index``
               over ALL assayed TFs of the reference x (reference gene universe & U's genes);
               perturbation: the same model on the tested (ragged) pairs of every perturbed
               Lambert TF with target in U's genes (``ragged_target_degree_null``; d_-TF(g) =
               positives of other perturbed TFs at g); identical to ``governance.py``.  1 = no
               enrichment beyond what the TF and the target's reference in-degree explain;
* ``within_tf_auroc`` mean over TFs with >= 1 positive and >= 1 negative of the per-TF AUROC
               (ties count 1/2): a TF-composition-free ranking measure;
* ``ap_lift_comp_adj`` (post hoc, 2026-10-14, verifier ``evalfix`` M1) composition-adjusted AP
               lift = AP / E0[AP], where E0 is the expectation under the within-TF label
               permutation of ``perm_p`` (positives of each TF spread uniformly over its edges; the
               TF's edge and positive counts and the score's tie blocks kept).  E0[AP] is computed in
               closed form (``exact_null_ap``: multivariate-hypergeometric moments per TF, independent
               across TFs), so ``ap_lift_comp_null`` = E0[AP] / b is the exact value of what
               ``perm_ap_lift_null_mean(_5k)`` estimates by Monte Carlo.  E0 already contains every
               gain a score can get from ranking high-prevalence TFs above low-prevalence TFs
               (between-TF composition), so ``ap_lift_comp_adj`` > 1 only for within-TF ranking signal
               (1 = no better than within-TF random); TF-cluster bootstrap CI with E0 recomputed in
               every resample (a TF drawn m times = m independent TF copies).  Simulation check
               (``comp_adj_null_coverage``: score ranking TFs by prevalence, random within TF, 150
               replicates): mean 1.000; the 95% interval contains 1 in 0.95 of replicates with 100
               TFs, 0.90 with 25 and 0.91 with 10 (small-cluster bootstrap), so the intervals of the
               tissue universes (14-24 TFs) are somewhat optimistic; the
               within-TF permutation p (``perm_p``, ``perm_p_5k``) is the calibrated test of
               'no within-TF signal';
* ``best_f1`` / ``best_precision`` max over thresholds (best precision only over thresholds
               with >= ``K_MIN_PRECISION`` = min(eval_k) predicted edges).

Inference (Q13, Q16).  The unit of resampling is the TF: edges of the same TF share the
TF's assay / perturbation and are dependent.

* 95% CIs: TF-cluster percentile bootstrap, ``PRESPECIFIED['n_bootstrap']`` = 1000 resamples
  of the evaluable TFs with replacement (all metrics recomputed, incl. the base rate);
  the same resamples are used for every score of a (universe, reference) (paired).
* ``perm_p``: one-sided within-TF permutation test of AP (H0: within each TF the score is
  unrelated to the reference label; TF composition and the TF's positive count are kept),
  ``PRESPECIFIED['n_null_draws']`` = 1000 permutations, p = (1 + #{AP* >= AP}) / (B + 1);
  Holm-adjusted over the scores of one (universe, reference) (``perm_p_holm``).
* Paired comparison Family-HB vs every other score (``heldout_paired.csv``): difference in
  AP lift and in within-TF AUROC, paired TF-cluster bootstrap 95% CI, two-sided bootstrap
  p = min(1, 2 min(#{D* <= 0} + 1, #{D* >= 0} + 1) / (B + 1)); Holm adjustment over the
  comparisons of one (universe, reference, metric).  The smallest attainable bootstrap p is
  2 / (B + 1) (``p_floor``) and the smallest Holm-adjusted p is m x 2 / (B + 1)
  (``p_holm_floor``); with B = 1000 and m = 15-34 comparisons this floor (0.03-0.07) made most
  contrasts unable to reach 0.05, so (revision 2026-10-08, verifier ``held`` M1):

  - every contrast also carries a bootstrap-SE z-test, p_z = 2 Phi(-|diff| / sd(D*))
    (``p_boot_z``, Holm ``p_z_holm``), which has no resampling floor;
  - CONFIRMATORY family (per universe x reference, ``all_evaluable`` edges, metric AP lift):
    Family-HB vs {EM, Fisher, Stouffer, vote_count, DoRothEA held-in, CollecTRI held-in}
    (``CONFIRMATORY_COMPARATORS``: the EM baseline of the plan's estimator ladder and the plan's
    baselines), 6 contrasts, paired TF-cluster bootstrap with ``N_BOOT_CONFIRM`` = 10,000
    resamples (``p_boot_10k``, floor 2/10001), Holm within the 6 (``p_holm_confirmatory``,
    floor 1.2e-3; ``p_z_holm_confirmatory`` from the 10k z-test); the same scores get a
    within-TF permutation test of AP with ``N_PERM_CONFIRM`` = 5,000 draws (``perm_p_5k``,
    Holm over the 7 scores ``perm_p_5k_holm_confirmatory``).  The family was defined after the
    held-out results had been seen ("prespecified in spirit": the comparators are the plan's,
    the restriction of the multiplicity family is post hoc); every other contrast is
    ``analysis_role = descriptive``.  ``score_a`` of the family is the Family-HB MAP point
    (``family_hb``); the NUTS posterior mean (``family_hb_nuts``, used by governance and
    replication) differs from it by <= 0.001 in global AP lift.  The value ``confirmatory`` of
    ``analysis_role`` is kept for table compatibility; ``multiplicity_family_defined`` records
    that the family restriction is post hoc (verifier ``evalfix`` m9).
  - primary test (verifier ``evalfix`` m3): the percentile bootstrap p (``p_boot``,
    ``p_boot_10k`` and their Holm adjustments); the bootstrap-SE z-test (``p_boot_z*``) is a
    secondary check without resampling floor.  The percentile test is preferred because the
    bootstrap distributions of the differences are skewed; the two disagree on whether 0.05 is
    crossed for global ENETS2 vs DoRothEA and global pooled perturbation vs Fisher.
  - secondary metrics of the confirmatory family (post hoc, 2026-10-14, verifier ``evalfix``
    M1): for the 6 comparators the within-TF AUROC, the unstratified Enr@500 and the
    composition-adjusted AP lift (``metric = ap_lift_comp_adj``) get the same B = 10,000 paired
    resamples and Holm within the 6 (``p_holm_6comparators``); ``confirmatory_comparator`` marks
    these rows.  The composition-adjusted contrasts are also given for the post hoc mechanism
    scores (B = 1000, descriptive).  The confirmatory metric stays AP lift.

Post hoc analyses (``analysis_status = post_hoc`` in every table; decision log
``ANALYSIS_DECISIONS``, written to ``outputs/r2/heldout_eval/meta_*.json`` and
``data/r2/tables/eval_analysis_decisions.csv``): the ``reported_any`` subset, the evidence-status
table, the ``prevalence_matched_ctx_tfs`` and TF-block nulls, the status-mechanism scores
(``status3`` = unobserved < observed-negative < reported, and ``family_hb_unobserved_last`` =
Family-HB MAP credibility with every unobserved edge placed below every observed edge; both
designed after the held-out results showed that the ranking of unobserved edges drives the
estimator comparison; ``fisher_sec6`` = Fisher with the plan section 6 per-annotator p values,
reported p = rate_j and observed-negative p = 1, which ties observed negatives with unobserved
edges, whereas the mid-p rule of ``lcm.baseline_scores`` ranks every observed negative above
every unobserved edge; verifier ``evalfix`` m1), the coverage-aware and matched-null method
metrics, the composition-adjusted AP lift and its paired contrasts, and the posterior-draw
AP-lift intervals.

Limitation (verifier ``evalfix`` m8): the TF is the resampling and permutation unit everywhere.
For ``pert_pooled`` this ignores the dependence between the TFs of one cohort (global universe: Norman 39
TFs, Replogle 56 TFs, measured in shared cells against shared controls), so the TF-cluster
intervals and p values of the pooled perturbation reference are optimistic to an unknown degree.

Coverage (verifier M5).  GRN-inference methods score only part of the universe (NaN = not
predicted, ranked last); AP lift over all evaluable edges therefore mixes coverage with ranking.
Every score with ``frac_scored < 1`` also gets metrics on the evaluable edges it scores
(``*_scored``).  Every method is reported relative to its matched null: the 1/L uniform-
attention null of the same model for scGPT / Geneformer attention variants
(``FM_MATCHED_NULL``), otherwise the coverage indicator (a constant score on the scored edges,
i.e. coverage alone); paired TF-cluster bootstrap CIs of method - null.

Universe decomposition (Q2; ``decomposition``)
----------------------------------------------
For tissue t, reference r in ``DECOMP_REFERENCES`` and the global-fit / context-fit Family-HB
credibility: raw P@k, best F1 and best precision in the context evaluable set vs the global
evaluable set; ratio decomposition raw_ctx / raw_glob = (b_ctx / b_glob) x (enr_ctx / enr_glob);
null: ``n_null_draws`` random subsets of the global evaluable edges of the context size with
exactly the context number of positives (``prevalence_matched``), or of the context size
only (``unmatched``, natural prevalence), ranked by the same credibility.  Reported:
null median / 95% interval, one-sided empirical p in the direction of the context-global gap
and the fraction of the gap that survives the null, (ctx - null median) / (ctx - glob).
The edge-level nulls draw edges independently from ~100 global TFs, although the context sets
come from 14-24 TFs, so their intervals are too narrow for the TF unit used everywhere else
(verifier M4).  Post hoc TF-cluster-aware nulls: ``tf_block_matched`` draws, for each context TF
t (n_t evaluable edges, k_t positives), one global evaluable TF block g (uniformly among global
TFs with >= k_t positives and >= n_t - k_t negatives, without replacement where possible) and
from it exactly k_t positives and n_t - k_t negatives; ``tf_block_size_matched`` matches n_t
only (natural prevalence of the drawn TF).  Context and global values carry TF-cluster bootstrap
CIs (``value_ctx_lo/hi``, ``value_glob_lo/hi``; ``N_BOOT`` resamples of each set's own TFs, drawn
independently for the two sets); the ratio CIs (``ratio_raw_lo/hi``, ``ratio_enrichment_lo/hi``)
combine the two independent resamples, which is conservative when the sets share TFs.  All
empirical p are descriptive (unadjusted).

Degree calibration (Q9; ``degree``)
-----------------------------------
TF literature degree = number of global-universe edges of the TF reported by >= 1 held-in
LIT annotator (label-free); quintiles over the 935 global TFs.  For each estimator /
baseline and reference (encode, enets2, pert_pooled; global universe): share of the top-k
(k in eval_k) and of the Strong set (BFDR <= 0.05, credibility estimators) in each quintile
vs the share of held-out positives in that quintile among evaluable edges;
``over_promotion`` = share_set / share_pos (TF-cluster bootstrap CI, set held fixed); within-
quintile mean credibility vs observed positive rate (reliability).

Sensitivity fits (Q20; ``sensitivity``): AP lift (+ CI) and paired difference to the primary
Family-HB for every fit under ``fits/{global,immune}/<variant>/<fit_id>``.  Fits written by the
post hoc modules (``informative_observation.py``: variant ``*__elig``; ``posterior_modes.py``:
``family_hb_mixture``, ``family_hb_nuts_mode<k>``) are scored too but carry ``kind =
informative_observation_model`` / ``posterior_mode`` and ``analysis_status = post_hoc``; ``fit_mtime``
dates every fit, so a row whose fit predates the final primary fits can be recognised.

Every table has ``analysis_status`` (row level) and ``post_hoc_columns`` (``POST_HOC_COLUMNS``:
columns of prespecified rows that were added after held-out results existed).  All tables were
regenerated on 2026-10-14 (decided 2026-10-13) from the frozen 4 x 1000/1000 ``family_hb_nuts`` fits of ``fit_primary.py``,
and again on 2026-10-14 with the post hoc ``heldfix`` additions (composition-adjusted AP lift,
``fisher_sec6``, sensitivity-fit convergence columns): new rows and columns only; every
pre-existing value is unchanged (checked against the earlier tables; only the ``post_hoc_columns``
lists grew) and ``heldout_methods_context.csv`` is byte-identical.

Sensitivity-fit convergence (verifier ``evalfix`` m6): ``heldout_sensitivity_fits.csv`` carries
``fit_r_hat_max`` and ``fit_init`` from the fit's ``meta.json`` and ``mode_mixing`` = R-hat max >
``MODE_MIXING_RHAT`` (1.1).  The randomly initialised NUTS runs (``family_hb_randinit_nuts`` and
the immune ``__prior_scale*_nuts`` fits) are non-converged mixtures over posterior modes (R-hat
1.56-1.74); their AP-lift shifts reflect mode mixing, not the varied prior or initialisation.

Supplementary: (A) is repeated on the evaluable edges reported by >= 1 held-in annotator
(``--subset reported_any``; ``eval_subset`` column), where the estimators' rankings differ;
``status`` tabulates the held-out positive rate of unobserved / observed-negative / reported
edges against the TF-matched expectation (a direct check of the missing-at-random observation
model of the latent-class estimators).

Outputs
-------
``outputs/r2/heldout_eval/`` (cached evaluable sets ``refs/<universe>/<ref>.parquet``,
bootstrap replicates of the paired statistics, ``meta.json``, logs) and the manuscript tables
``data/r2/tables/heldout_{estimators,paired,universe_decomposition,downsampling_null,
degree_calibration,sensitivity_fits,methods_context,evidence_status}.csv``.

Command line (from the biodyn-work root)::

    export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
    .venv/bin/python subproject_24_multireference_latent_truth_model/src/r2/heldout_eval.py all
    # or stage by stage: refs | estimators [--universes U ...] [--subset reported_any]
    #                    | estimators --merge-partials | decomposition | degree | status | sensitivity
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from r2 import common as C  # noqa: E402

PRESPEC = C.PRESPECIFIED
SEED = int(PRESPEC["random_seed"])
N_BOOT = int(PRESPEC["n_bootstrap"])
N_NULL = int(PRESPEC["n_null_draws"])
EVAL_K = tuple(int(k) for k in PRESPEC["eval_k"])
TOPK = int(PRESPEC["topk_per_tf"])
TOPK_ALL = (TOPK,) + tuple(int(k) for k in PRESPEC["topk_per_tf_sensitivity"])
VARIANT = PRESPEC["evidence_variant_primary"]
BFDR_STRONG = float(PRESPEC["bfdr_strong"])
TIE_DECIMALS = 12
K_MIN_PRECISION = min(EVAL_K)
LOW_SUPPORT_TFS = 10        # as heldout_chip: < 10 evaluable TFs or < 20 positives = low support
LOW_SUPPORT_POS = 20
ALPHA_CI = 0.05
RANDOM_LEVELS = 10_000      # global random score: uniform on 10^4 tie levels (ties resolved in expectation)
# ---- revision 2026-10-08 (verifier held M1-M5, m1-m6; post hoc unless stated)
N_BOOT_CONFIRM = 10_000     # paired bootstrap resamples for the confirmatory family
N_PERM_CONFIRM = 5_000      # within-TF permutations for the confirmatory scores
CONFIRMATORY_COMPARATORS = ("em", "fisher", "stouffer", "vote_count", "dorothea_level_heldin",
                            "collectri_heldin")
CONFIRMATORY_METRIC = "ap_lift"
STATUS_SCORE = "status3"                        # 0 unobserved < 1 observed negative < 2 reported
UNOBS_LAST_SCORE = "family_hb_unobserved_last"  # Family-HB MAP, unobserved edges below all observed
FISHER_SEC6_SCORE = "fisher_sec6"              # Fisher with the plan section 6 p values (verifier evalfix m1)
POST_HOC_SCORES = (STATUS_SCORE, UNOBS_LAST_SCORE, FISHER_SEC6_SCORE)
# ---- revision 2026-10-14 (module heldfix; verifier evalfix M1, m1, m3, m6, m8-m10; post hoc)
COMP_ADJ_METRIC = "ap_lift_comp_adj"            # AP / exact within-TF-permutation expectation of AP
COMP_ADJ_COMPARATORS = CONFIRMATORY_COMPARATORS + POST_HOC_SCORES   # paired composition-adjusted contrasts
MODE_MIXING_RHAT = 1.1                          # sensitivity fits: R-hat max above this = mixture over modes
FM_MATCHED_NULL = {**{f"scgpt_{v}": "scgpt_attn_uniform" for v in ("attn", "attn_T", "attn_norm", "attn_T_norm")},
                   **{f"geneformer_{v}": "geneformer_attn_uniform"
                      for v in ("attn", "attn_T", "attn_norm", "attn_T_norm")}}
COVERAGE_NULL = "coverage_indicator"
N_POSTERIOR_DRAWS = 1000    # NUTS draws propagated into AP lift (all stored draws)
FEW_TFS = 5                 # degree calibration: quintile cells with < 5 evaluable TFs are flagged
ANALYSIS_DECISIONS = [      # dated decision log of this module (also written to meta_*.json)
    dict(id="held_settings", date="2026-10-08", status="module_design",
         item="PAIRED_METRICS (ap_lift, within_tf_auroc, enr_at_500), K_MIN_PRECISION = min(eval_k), "
              "LOW_SUPPORT_TFS = 10 / LOW_SUPPORT_POS = 20, RANDOM_LEVELS = 1e4, TIE_DECIMALS = 12",
         note="fixed in the module at creation (before its outputs existed); not in PRESPECIFIED"),
    dict(id="held_reported_any", date="2026-10-08", status="post_hoc",
         item="reported_any subset of the estimator comparison", note="added after the all-edge results"),
    dict(id="held_evidence_status", date="2026-10-08", status="post_hoc",
         item="heldout_evidence_status.csv", note="added after the all-edge results"),
    dict(id="held_ctx_tf_null", date="2026-10-08", status="post_hoc",
         item="prevalence_matched_ctx_tfs down-sampling null", note="added after the first decomposition"),
    dict(id="held_confirmatory_family", date="2026-10-08", status="post_hoc",
         item="confirmatory family Family-HB vs {EM, Fisher, Stouffer, vote_count, DoRothEA held-in, "
              "CollecTRI held-in} on AP lift, B = 10,000 paired bootstrap, 5,000 within-TF permutations; "
              "bootstrap-SE z-test for all contrasts",
         note="defined after the held-out results (verifier M1); comparators are the plan section 6 baselines"),
    dict(id="held_status_mechanism", date="2026-10-08", status="post_hoc",
         item="status3 and family_hb_unobserved_last scores", note="verifier M2; informed by the outcomes"),
    dict(id="held_tf_block_null", date="2026-10-08", status="post_hoc",
         item="TF-block-matched down-sampling nulls and TF-bootstrap CIs of the decomposition", note="verifier M4"),
    dict(id="held_method_nulls", date="2026-10-08", status="post_hoc",
         item="coverage-aware (scored-subset) metrics and matched-null contrasts of the methods", note="verifier M5"),
    dict(id="held_methods_holm", date="2026-10-08", status="post_hoc",
         item="Holm and BH of methods_context permutation p within universe x reference x k", note="verifier m2"),
    dict(id="held_posterior_draws", date="2026-10-08", status="post_hoc",
         item="AP lift and Enr@500 over the stored NUTS draws (plug-in credibility per draw)", note="verifier m5"),
    dict(id="held_ctx_tf_coverage", date="2026-10-13", status="post_hoc",
         item="decomposition: number / share of context evaluable TFs and edges whose TF is evaluable in the global "
              "universe (n_ctx_tfs_in_glob, frac_ctx_edges_tf_in_glob, frac_ctx_pos_tf_in_glob)",
         note="verifier m7 (context TFs absent from the global universe cannot be in a matched null pool)"),
    dict(id="held_sensitivity_kinds", date="2026-10-13", status="post_hoc",
         item="heldout_sensitivity_fits.csv: fits of the post hoc modules are labelled kind = "
              "informative_observation_model (variant *__elig) / posterior_mode (family_hb_mixture, "
              "family_hb_nuts_mode*) with analysis_status = post_hoc; fit_mtime column",
         note="these fit directories appeared after the first sensitivity run; they were not part of the Q20 design"),
    dict(id="held_rerun_final_fits", date="2026-10-13", status="module_design",
         item="all heldout_* tables regenerated after fit_primary replaced the family_hb_nuts fits of the four "
              "universes by frozen 4 x 1000/1000 runs (immune 2026-10-08; kidney, lung, global 2026-10-13)",
         note="no setting of this module changed; only family_hb_nuts rows and posterior-draw intervals can move"),
    dict(id="held_comp_adj", date="2026-10-14", status="post_hoc",
         item="heldout_estimators.csv: composition-adjusted AP lift ap_lift_comp_adj = AP / E0[AP] (E0 = exact "
              "expectation under the within-TF label permutation of perm_p, closed form; ap_lift_comp_null = E0[AP] / "
              "base rate, of which perm_ap_lift_null_mean(_5k) is the Monte Carlo estimate) for every score, TF-cluster "
              "percentile bootstrap CI with E0 recomputed per resample (B = 1000; *_10k: B = 10,000 for the "
              "confirmatory scores; a TF drawn m times = m independent TF copies)",
         note="verifier evalfix M1 (pooled perturbation contrasts dominated by between-TF composition); defined "
              "after the held-out results; simulation check comp_adj_null_coverage (no within-TF signal): mean 1.000, "
              "95% CI coverage 0.95 / 0.90 / 0.91 with 100 / 25 / 10 TFs"),
    dict(id="held_comp_adj_paired", date="2026-10-14", status="post_hoc",
         item="heldout_paired.csv metric = ap_lift_comp_adj: Family-HB (MAP) minus comparator in composition-"
              "adjusted AP lift, all evaluable edges, every universe x reference; the 6 confirmatory comparators "
              "with B = 10,000 paired resamples and Holm within the 6 (p_holm_6comparators), the post hoc mechanism "
              "scores with B = 1000 (descriptive); not confirmatory (the confirmatory metric stays AP lift)",
         note="verifier evalfix M1"),
    dict(id="held_confirmatory_secondary", date="2026-10-14", status="post_hoc",
         item="confirmatory scores: B = 10,000 CIs of within-TF AUROC and unstratified Enr@500 in heldout_estimators.csv "
              "(*_lo_10k / *_hi_10k); heldout_paired.csv: confirmatory_comparator flag on every metric row of the 6 "
              "comparators (AP lift confirmatory; within-TF AUROC, Enr@500, ap_lift_comp_adj secondary) and "
              "primary_test column",
         note="verifier evalfix M1 (report the within-TF contrasts next to every confirmatory AP-lift contrast)"),
    dict(id="held_fisher_sec6", date="2026-10-14", status="post_hoc",
         item="score fisher_sec6 = Fisher with the plan section 6 per-annotator p (reported p = rate_j, observed "
              "negative p = 1; observed negatives tie with unobserved edges) instead of the mid-p rule of "
              "lcm.baseline_scores; group post_hoc_mechanism (outside every Holm family)",
         note="verifier evalfix m1: the Fisher/Stouffer advantage over Family-HB is built into the mid-p rule; the "
              "mid-p deviation itself was logged and fixed before evaluation"),
    dict(id="held_primary_test", date="2026-10-14", status="post_hoc",
         item="percentile bootstrap p (p_boot, p_boot_10k and their Holm adjustments) named the primary test of every "
              "paired contrast; bootstrap-SE z p (p_boot_z*) secondary",
         note="verifier evalfix m3: the two disagree on crossing 0.05 for global ENETS2 vs DoRothEA and global "
              "pert_pooled vs Fisher; the bootstrap distributions are skewed"),
    dict(id="held_confirmatory_score_a", date="2026-10-14", status="post_hoc",
         item="score_a of the confirmatory family is the Family-HB MAP point (family_hb), stated in "
              "multiplicity_family_defined; analysis_role value 'confirmatory' kept for table compatibility",
         note="verifier evalfix m9; NUTS mean differs by <= 0.001 in global AP lift"),
    dict(id="held_sens_convergence", date="2026-10-14", status="post_hoc",
         item="heldout_sensitivity_fits.csv: fit_r_hat_max (meta.json convergence.r_hat_max, else r_hat_max), "
              "fit_init, mode_mixing = fit_r_hat_max > 1.1 (non-converged mixture over posterior modes)",
         note="verifier evalfix m6: the random-initialisation and immune prior-scale NUTS fits have R-hat 1.56-1.74"),
    dict(id="held_decisions_lock", date="2026-10-14", status="module_design",
         item="record_decisions: exclusive file lock (eval_analysis_decisions.csv.lock) around the read-modify-write "
              "and atomic replace of the shared decision log",
         note="verifier evalfix m10; the other modules' own record_decisions copies are unchanged"),
]
# columns of prespecified rows that were added after held-out results existed (``post_hoc_columns``)
POST_HOC_COLUMNS = {
    "estimators": "perm_p_floor;perm_p_holm_floor;analysis_role;*_10k;*_5k*;*_scored*;matched_null;*_matched_null*;"
                  "*_coverage_null;ap_lift_comp_null;ap_lift_comp_adj*",
    "paired": "p_floor;p_boot_z;p_z_holm;p_holm_floor;degenerate_constant_score;*_10k;p_holm_6comparators*;"
              "p_z_holm_6comparators;p_holm_confirmatory*;p_z_holm_confirmatory;multiplicity_family_defined;analysis_role;"
              "confirmatory_comparator;primary_test",
    "methods_context": "matched_null;null_enr_tf_matched;enr_ratio_to_null*;coverage_null_enr_tf_matched;perm_p_holm;"
                       "perm_p_bh;perm_p_holm_floor",
    "universe_decomposition": "value_ctx_lo/hi;value_glob_lo/hi;base_rate_ctx_lo/hi;ratio_raw_lo/hi;"
                              "ratio_enrichment_lo/hi;n_ctx_tfs_in_glob;frac_ctx_edges_tf_in_glob;"
                              "frac_ctx_pos_tf_in_glob",
    "downsampling_null": "p_emp_two_sided;n_ctx_tfs;n_ctx_tfs_infeasible;mean_reused_blocks;min_eligible_blocks",
    "degree_calibration": "few_tfs",
    "sensitivity_fits": "p_diff_z;n_posterior_draws;ap_lift_post_*;enr_at_500_post_*;fit_mtime;fit_r_hat_max;fit_init;"
                        "mode_mixing;convergence_note",
    "evidence_status": "",
}

OUT_DIR = C.OUT / "heldout_eval"
REF_DIR = OUT_DIR / "refs"
FITS_DIR = C.OUT / "fits"
EV_DIR = C.OUT / "evidence"
CHIP_DIR = C.OUT / "heldout_chip"
COHORT_DIR = C.OUT / "cohorts"
TABLE_PREFIX = "heldout_"

UNIVERSES = ("global",) + tuple(C.TISSUES)
CHIP_REFS = {"encode": "encode_tfbs", "encode_strict": "encode_strict", "enets2": "enets2"}
MAP_MODELS = ("em", "dependence", "multifactor", "family_regularized", "family_hb")
LCM_SCORES = MAP_MODELS + ("family_hb_nuts",)
PRIMARY = "family_hb"
PATTERN_BASELINES = ("union", "majority", "vote_count", "vote_fraction", "n_families_reporting",
                     "fisher", "stouffer")
INTEGRATION_BASELINES = ("dorothea_level_heldin", "collectri_heldin")
FM_ATTENTION = ("scgpt_attn", "scgpt_attn_T", "scgpt_attn_norm", "scgpt_attn_T_norm",
                "scgpt_attn_uniform", "geneformer_attn", "geneformer_attn_T", "geneformer_attn_norm",
                "geneformer_attn_T_norm", "geneformer_attn_uniform")
CORRELATIONS = ("pearson", "pearson_abs", "spearman", "spearman_abs")
EXPRESSION = {"grnboost2": "grnboost2", "genie3": "genie3",
              "grnboost2_detect1pct": "grnboost2_detect1pct", "genie3_detect1pct": "genie3_detect1pct"}
EXPRESSION_TISSUES = {"genie3": ("immune",), "genie3_detect1pct": ("immune",)}
DECOMP_REFERENCES = ("pert_pooled", "encode")
DEGREE_REFERENCES = ("encode", "enets2", "pert_pooled")
SENS_REFERENCES = ("encode", "enets2", "pert_pooled")
SENS_UNIVERSES = ("global", "immune")


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# =========================================================================================
# 1. Metric core (pure numpy; unit-tested in tests/test_heldout_eval.py)
# =========================================================================================
def score_levels(s: np.ndarray, decimals: int = TIE_DECIMALS) -> tuple[np.ndarray, np.ndarray]:
    """Tie blocks of a score in decreasing order.

    Returns (level per element, block values in decreasing order).  NaN and -inf share the
    bottom block (NaN = not predicted ranks last); +inf is an ordinary top value."""
    s = np.asarray(s, dtype=np.float64)
    v = np.where(np.isnan(s), -np.inf, np.round(s, decimals))
    uniq, inv = np.unique(-v, return_inverse=True)
    return inv.reshape(-1).astype(np.int64), -uniq


@dataclass
class LevelCounts:
    """TF x tie-block count matrices of one score on one evaluable set."""

    tot: sp.csr_matrix          # (T, L) edges
    pos: sp.csr_matrix          # (T, L) positives
    nul: sp.csr_matrix          # (T, L) sum of null probabilities
    values: np.ndarray          # (L,) block values (decreasing)

    @property
    def L(self):
        return self.tot.shape[1]

    def totals(self):
        return (np.asarray(self.tot.sum(0)).ravel(), np.asarray(self.pos.sum(0)).ravel(),
                np.asarray(self.nul.sum(0)).ravel())

    def weighted(self, M: np.ndarray):
        """Aggregate with TF weights M (B, T) -> three (B, L) arrays."""
        M = np.asarray(M, dtype=np.float64)
        return ((self.tot.T @ M.T).T, (self.pos.T @ M.T).T, (self.nul.T @ M.T).T)


def level_counts(tf_code: np.ndarray, y: np.ndarray, null_p: np.ndarray, score: np.ndarray,
                 n_tf: int) -> LevelCounts:
    """Aggregate an evaluable set to TF x tie-block counts.  The three matrices share one
    sparsity structure (rows = TFs, sorted block indices within a row)."""
    lev, vals = score_levels(score)
    L = len(vals)
    tf_code = np.asarray(tf_code, dtype=np.int64)
    key = tf_code * L + lev
    uk, inv = np.unique(key, return_inverse=True)
    inv = inv.reshape(-1)
    rows, cols = uk // L, uk % L
    indptr = np.concatenate([[0], np.cumsum(np.bincount(rows, minlength=n_tf))])
    mk = lambda d: sp.csr_matrix((d, cols, indptr), shape=(n_tf, L))  # noqa: E731
    tot = mk(np.bincount(inv, minlength=len(uk)).astype(np.float64))
    pos = mk(np.bincount(inv, weights=np.asarray(y, dtype=np.float64), minlength=len(uk)))
    nul = mk(np.bincount(inv, weights=np.nan_to_num(np.asarray(null_p, dtype=np.float64)),
                         minlength=len(uk)))
    return LevelCounts(tot, pos, nul, vals)


def _at_k(ct, cp, tot, pos, k):
    """Fractional-tie count of positives in the top k (rows of (B, L) cumulative arrays).
    Returns (TP@k, idx) with NaN where k > N."""
    ct = np.atleast_2d(ct)
    n_before = (ct < k).sum(-1)                         # index of the block holding the k-th edge
    N = ct[:, -1]
    ok = N >= k
    i = np.minimum(n_before, ct.shape[1] - 1)
    rows = np.arange(ct.shape[0])
    prev_t = np.where(i > 0, ct[rows, np.maximum(i - 1, 0)], 0.0)
    prev_p = np.where(i > 0, cp[rows, np.maximum(i - 1, 0)], 0.0)
    t_i = np.atleast_2d(tot)[rows, i]
    p_i = np.atleast_2d(pos)[rows, i]
    frac = np.where(t_i > 0, (k - prev_t) / np.where(t_i > 0, t_i, 1), 0.0)
    tp = prev_p + frac * p_i
    return np.where(ok, tp, np.nan), i, frac


def curve_metrics(tot, pos, nul=None, ks=EVAL_K, k_min: int = K_MIN_PRECISION) -> dict:
    """Ranking metrics from tie-block counts (arrays (L,) or (B, L), blocks in decreasing
    score order).  See the module docstring for the definitions."""
    tot = np.atleast_2d(np.asarray(tot, dtype=np.float64))
    pos = np.atleast_2d(np.asarray(pos, dtype=np.float64))
    ct = np.cumsum(tot, -1)
    cp = np.cumsum(pos, -1)
    N = ct[:, -1]
    P = cp[:, -1]
    with np.errstate(invalid="ignore", divide="ignore"):
        prec = np.where(ct > 0, cp / np.where(ct > 0, ct, 1), 0.0)
        ap = (pos * prec).sum(-1) / P
        base = P / N
        out = {"n": N, "n_pos": P, "base_rate": base, "ap": ap, "ap_lift": ap / base}
        f1 = 2 * cp / (ct + P[:, None])
        out["best_f1"] = np.nanmax(np.where(tot > 0, f1, np.nan), -1)
        prec_ok = np.where((ct >= k_min) & (tot > 0), prec, np.nan)
        pk_min, _, _ = _at_k(ct, cp, tot, pos, k_min)
        out["best_precision"] = np.fmax(np.nanmax(np.where(np.isnan(prec_ok).all(-1, keepdims=True),
                                                           -np.inf, prec_ok), -1), pk_min / k_min)
        out["best_precision"] = np.where(np.isfinite(out["best_precision"]), out["best_precision"],
                                         np.nan)
        if nul is not None:
            nul = np.atleast_2d(np.asarray(nul, dtype=np.float64))
            cn = np.cumsum(nul, -1)
        for k in ks:
            tp, i, frac = _at_k(ct, cp, tot, pos, k)
            out[f"p_at_{k}"] = tp / k
            out[f"enr_at_{k}"] = tp / k / base
            if nul is not None:
                rows = np.arange(ct.shape[0])
                prev_n = np.where(i > 0, cn[rows, np.maximum(i - 1, 0)], 0.0)
                en = prev_n + frac * nul[rows, i]
                out[f"exp_null_at_{k}"] = np.where(np.isnan(tp), np.nan, en)
                out[f"strat_enr_at_{k}"] = tp / en
    return out


def per_tf_auroc(lc: LevelCounts) -> np.ndarray:
    """AUROC of the score within each TF (ties count 1/2); NaN for TFs without positives or
    without negatives."""
    T = lc.tot.shape[0]
    out = np.full(T, np.nan)
    ip = lc.tot.indptr
    for t in range(T):
        a, b = ip[t], ip[t + 1]
        if b == a:
            continue
        nt = lc.tot.data[a:b]                      # blocks in decreasing score order
        pt = lc.pos.data[a:b]
        nn = nt - pt
        Pp, Nn = pt.sum(), nn.sum()
        if Pp <= 0 or Nn <= 0:
            continue
        neg_below = Nn - np.cumsum(nn)            # negatives in strictly lower blocks
        out[t] = float((pt * (neg_below + 0.5 * nn)).sum() / (Pp * Nn))
    return out


def tf_bootstrap_weights(n_tf: int, n_boot: int, rng: np.random.Generator) -> np.ndarray:
    """(B, T) multiplicities of a TF-cluster bootstrap (T TFs drawn with replacement)."""
    return rng.multinomial(n_tf, np.full(n_tf, 1.0 / n_tf), size=n_boot).astype(np.float64)


def boot_metrics(lc: LevelCounts, M: np.ndarray, ks=EVAL_K, chunk: int | None = None) -> dict:
    """Curve metrics for each bootstrap replicate (rows of M), chunked to bound memory."""
    B = M.shape[0]
    if chunk is None:
        chunk = max(1, min(B, int(1e7 // max(lc.L, 1))))
    parts = []
    for s in range(0, B, chunk):
        tot, pos, nul = lc.weighted(M[s:s + chunk])
        parts.append(curve_metrics(tot, pos, nul, ks))
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


def boot_light(lc: LevelCounts, M: np.ndarray, k: int = 500, chunk: int | None = None) -> dict:
    """AP lift and Enr@k only, for each bootstrap replicate (rows of M); the large-B
    confirmatory resampling uses this lighter version of ``boot_metrics`` (same definitions)."""
    B = M.shape[0]
    if chunk is None:
        chunk = max(1, min(B, int(1e7 // max(lc.L, 1))))
    ap_l, enr = [], []
    for s in range(0, B, chunk):
        Mc = np.asarray(M[s:s + chunk], dtype=np.float64)
        tot = (lc.tot.T @ Mc.T).T
        pos = (lc.pos.T @ Mc.T).T
        ct = np.cumsum(tot, -1)
        cp = np.cumsum(pos, -1)
        N, P = ct[:, -1], cp[:, -1]
        with np.errstate(invalid="ignore", divide="ignore"):
            prec = np.where(ct > 0, cp / np.where(ct > 0, ct, 1), 0.0)
            base = P / N
            ap_l.append((pos * prec).sum(-1) / P / base)
            tp, _, _ = _at_k(ct, cp, tot, pos, k)
            enr.append(tp / k / base)
    return {"ap_lift": np.concatenate(ap_l), f"enr_at_{k}": np.concatenate(enr)}


def bootstrap_p_z(d: np.ndarray, point: float) -> float:
    """Two-sided bootstrap-SE z-test p = 2 Phi(-|point| / sd(D*)) (no resampling floor)."""
    from scipy.stats import norm
    d = np.asarray(d, dtype=np.float64)
    d = d[np.isfinite(d)]
    if d.size < 2 or not np.isfinite(point):
        return np.nan
    sd = float(np.std(d, ddof=1))
    if sd <= 0:
        return 1.0 if point == 0 else 0.0
    return float(2 * norm.sf(abs(point) / sd))


def bh(p: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values (NaN kept as NaN)."""
    p = np.asarray(p, dtype=np.float64)
    out = np.full_like(p, np.nan)
    ok = np.flatnonzero(np.isfinite(p))
    if ok.size == 0:
        return out
    o = ok[np.argsort(p[ok], kind="mergesort")]
    m = len(o)
    adj = p[o] * m / np.arange(1, m + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    out[o] = np.minimum(adj, 1.0)
    return out


def permuted_ap(lc: LevelCounts, n_perm: int, rng: np.random.Generator,
                chunk: int | None = None) -> np.ndarray:
    """AP under within-TF permutation of the score (equivalently of the labels): for each TF
    the P_t positives are spread over its edges uniformly at random, i.e. the positives per
    tie block follow a multivariate hypergeometric distribution with the TF's block sizes."""
    T, L = lc.tot.shape
    tot_all = np.asarray(lc.tot.sum(0)).ravel()
    P_t = np.rint(np.asarray(lc.pos.sum(1)).ravel()).astype(np.int64)
    if chunk is None:
        chunk = max(1, min(n_perm, int(2.5e7 // max(L, 1))))
    out = []
    for s in range(0, n_perm, chunk):
        b = min(chunk, n_perm - s)
        posp = np.zeros((b, L))
        for t in range(T):
            if P_t[t] == 0:
                continue
            a, e = lc.tot.indptr[t], lc.tot.indptr[t + 1]
            cols = lc.tot.indices[a:e]
            colors = np.rint(lc.tot.data[a:e]).astype(np.int64)
            draws = rng.multivariate_hypergeometric(colors, int(P_t[t]), size=b, method="marginals")
            posp[:, cols] += draws
        out.append(curve_metrics(np.broadcast_to(tot_all, posp.shape), posp, None, ks=())["ap"])
    return np.concatenate(out)


def null_ap_components(lc: LevelCounts) -> tuple[sp.csr_matrix, sp.csr_matrix, np.ndarray]:
    """Per-TF ingredients of the exact within-TF-permutation expectation of AP (``exact_null_ap``).

    Under the permutation of ``permuted_ap`` the positives x_tl of TF t in tie block l are
    multivariate hypergeometric (K_t positives over the TF's N_t edges, n_tl in block l) and
    independent across TFs.  With C_tl = sum_{m <= l} n_tm:
        E[x_tl] = r_t n_tl,  r_t = K_t / N_t,
        Cov(x_tl, sum_{m <= l} x_tm) = f_t (n_tl / N_t)(1 - C_tl / N_t),  f_t = K_t (N_t - K_t) / (N_t - 1).
    Returns (R = r_t n_tl, V = the covariance term, K) as T x L matrices on the sparsity pattern
    of ``lc.tot`` (block indices sorted within rows, as built by ``level_counts``)."""
    tot = lc.tot
    T = tot.shape[0]
    ip = tot.indptr
    cnt = np.diff(ip)
    rows = np.repeat(np.arange(T), cnt)
    if tot.nnz and np.any((np.diff(tot.indices) <= 0) & (np.diff(rows) == 0)):
        raise ValueError("null_ap_components: block indices must be sorted within TF rows")
    n = tot.data.astype(np.float64)
    N_t = np.asarray(tot.sum(1)).ravel()
    K_t = np.asarray(lc.pos.sum(1)).ravel()
    cs = np.cumsum(n)
    before = np.concatenate([[0.0], cs])[ip[:-1]]            # cumulative count before each row
    C = cs - np.repeat(before, cnt)                            # within-row cumulative n_tl
    with np.errstate(invalid="ignore", divide="ignore"):
        r = np.where(N_t > 0, K_t / np.where(N_t > 0, N_t, 1), 0.0)
        f = np.where(N_t > 1, K_t * (N_t - K_t) / np.where(N_t > 1, N_t - 1, 1), 0.0)
        Nr = np.where(N_t > 0, N_t, 1.0)[rows]
        v = f[rows] * (n / Nr) * (1.0 - C / Nr)
    R = sp.csr_matrix((r[rows] * n, tot.indices, ip), shape=tot.shape)
    V = sp.csr_matrix((np.maximum(v, 0.0), tot.indices, ip), shape=tot.shape)
    return R, V, K_t


def exact_null_ap(lc: LevelCounts, M: np.ndarray | None = None, chunk: int | None = None) -> dict:
    """Exact expectation of AP (and AP lift) under the within-TF label permutation of
    ``permuted_ap``, for the full set (``M = None``) or for each TF-bootstrap replicate (rows of
    M; a TF with multiplicity m enters as m independent copies).  AP = sum_l p_l P_l / (P CT_l)
    with P and the cumulative block totals CT_l fixed under the permutation, so
    E[AP] = sum_l (E[p_l] E[P_l] + Cov(p_l, P_l)) / (P CT_l) (``null_ap_components``).
    Returns {"ap": (B,), "ap_lift": (B,)} (ap_lift = E[AP] / base rate of the replicate)."""
    R, V, K = null_ap_components(lc)
    T, L = lc.tot.shape
    if M is None:
        M = np.ones((1, T))
    M = np.asarray(M, dtype=np.float64)
    B = M.shape[0]
    if chunk is None:
        chunk = max(1, min(B, int(5e6 // max(L, 1))))
    ap, lift = [], []
    for s in range(0, B, chunk):
        Mc = M[s:s + chunk]
        tot = (lc.tot.T @ Mc.T).T
        Ep = (R.T @ Mc.T).T
        Vs = (V.T @ Mc.T).T
        ct = np.cumsum(tot, -1)
        EP = np.cumsum(Ep, -1)
        P = Mc @ K
        N = ct[:, -1]
        with np.errstate(invalid="ignore", divide="ignore"):
            e = ((Ep * EP + Vs) / np.where(ct > 0, ct, 1.0)).sum(-1) / P
            ap.append(e)
            lift.append(e / (P / N))
    return {"ap": np.concatenate(ap), "ap_lift": np.concatenate(lift)}


def percentile_ci(x: np.ndarray, alpha: float = ALPHA_CI) -> tuple[float, float]:
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return (np.nan, np.nan)
    return (float(np.quantile(x, alpha / 2)), float(np.quantile(x, 1 - alpha / 2)))


def bootstrap_p_two_sided(d: np.ndarray) -> float:
    """Two-sided bootstrap p for H0: difference = 0 from paired replicates d."""
    d = np.asarray(d, dtype=np.float64)
    d = d[np.isfinite(d)]
    if d.size == 0:
        return np.nan
    B = d.size
    lo = (1 + np.sum(d <= 0)) / (B + 1)
    hi = (1 + np.sum(d >= 0)) / (B + 1)
    return float(min(1.0, 2 * min(lo, hi)))


def holm(p: np.ndarray) -> np.ndarray:
    """Holm step-down adjusted p-values (NaN kept as NaN)."""
    p = np.asarray(p, dtype=np.float64)
    out = np.full_like(p, np.nan)
    ok = np.flatnonzero(np.isfinite(p))
    if ok.size == 0:
        return out
    o = ok[np.argsort(p[ok], kind="mergesort")]
    m = len(o)
    adj = np.maximum.accumulate((m - np.arange(m)) * p[o])
    out[o] = np.minimum(adj, 1.0)
    return out


# -----------------------------------------------------------------------------------------
# target-degree null for ragged (tested-pair) references
# -----------------------------------------------------------------------------------------
def ragged_target_degree_null(tf_code: np.ndarray, gene_code: np.ndarray, y: np.ndarray,
                              n_tf: int, n_gene: int) -> tuple[np.ndarray, dict]:
    """TF x target-degree null of ``build_heldout_chip.target_degree_model`` on a ragged set
    of evaluable pairs (perturbation cohorts test different targets per perturbation).

    d_-TF(g) = number of positives of other TFs with target g among the evaluable pairs
    (leave-pair-out reference in-degree); bins by ``degree_bin_edges`` (prespecified union of
    gene and edge deciles over the genes present in the evaluable set); logit P(y) =
    a_TF + b_bin by maximum likelihood on aggregated (TF, bin) counts (damped Newton).  On a
    rectangular set this reproduces ``target_degree_model`` (tested).  Returns (null
    probability per pair, diagnostics)."""
    from r2.build_heldout_chip import degree_bin_edges, TARGET_DEGREE_BINS
    tf_code = np.asarray(tf_code, dtype=np.int64)
    gene_code = np.asarray(gene_code, dtype=np.int64)
    y = np.asarray(y, dtype=np.float64)
    deg_all = np.bincount(gene_code, weights=y, minlength=n_gene)
    present = np.bincount(gene_code, minlength=n_gene) > 0
    deg = np.rint(deg_all[present]).astype(np.int64)
    edges = degree_bin_edges(deg, TARGET_DEGREE_BINS)
    inner = edges[1:-1]
    d_pair = np.rint(deg_all[gene_code] - y).astype(np.int64)
    b = np.searchsorted(inner, d_pair, side="right")
    nb = len(inner) + 1
    n = np.zeros((n_tf, nb))
    k = np.zeros((n_tf, nb))
    np.add.at(n, (tf_code, b), 1.0)
    np.add.at(k, (tf_code, b), y)
    P, diag = _fit_tf_bin_logit(n, k)
    diag.update(n_bins=int(nb), bin_edges=edges.tolist())
    return P[tf_code, b], diag


def _fit_tf_bin_logit(n: np.ndarray, k: np.ndarray, max_iter: int = 200, tol: float = 1e-6):
    """ML fit of logit p = a_TF + b_bin (first bin = reference) on aggregated counts; TFs
    with no positives get p = 0 (a -> -inf), TFs with all positives p = 1."""
    n_tf, nb = n.shape
    rows_ok = (k.sum(1) > 0) & (k.sum(1) < n.sum(1))
    P = np.zeros((n_tf, nb))
    P[(k.sum(1) >= n.sum(1)) & (n.sum(1) > 0)] = 1.0
    idx = np.flatnonzero(rows_ok)
    if idx.size == 0:
        return P, {"converged": True, "iterations": 0}
    nn, kk = n[idx], k[idx]
    cells = np.argwhere(nn > 0)
    T2 = len(idx)
    X = np.zeros((len(cells), T2 + nb - 1))
    X[np.arange(len(cells)), cells[:, 0]] = 1.0
    nz = cells[:, 1] > 0
    X[np.flatnonzero(nz), T2 + cells[nz, 1] - 1] = 1.0
    yy, w = kk[cells[:, 0], cells[:, 1]], nn[cells[:, 0], cells[:, 1]]
    beta = np.zeros(X.shape[1])
    r = np.clip(kk.sum(1) / nn.sum(1), 1e-6, 1 - 1e-6)
    beta[:T2] = np.log(r / (1 - r))

    def ll(bv):
        eta = X @ bv
        return float(np.sum(yy * eta - w * np.logaddexp(0.0, eta)))

    cur, conv, it = ll(beta), False, 0
    for it in range(max_iter):
        p = 1 / (1 + np.exp(-(X @ beta)))
        g = X.T @ (yy - w * p)
        if np.max(np.abs(g)) < tol:
            conv = True
            break
        H = (X * (w * p * (1 - p))[:, None]).T @ X + 1e-9 * np.eye(X.shape[1])
        step, s = np.linalg.solve(H, g), 1.0
        while True:
            cand = beta + s * step
            lc_ = ll(cand)
            if lc_ >= cur - 1e-12 * abs(cur) or s < 1e-10:
                break
            s /= 2
        beta, cur = cand, lc_
    eta = beta[:T2][:, None] + np.concatenate([[0.0], beta[T2:]])[None, :]
    P[idx] = 1 / (1 + np.exp(-eta))
    return P, {"converged": bool(conv), "iterations": int(it)}


# =========================================================================================
# 2. Universes, evaluable reference sets and scores
# =========================================================================================
def _stream(*parts) -> np.random.Generator:
    """Deterministic random stream for a named analysis step (seed = PRESPECIFIED seed)."""
    import zlib
    return np.random.default_rng([SEED] + [zlib.crc32(str(p).encode()) for p in parts])


@dataclass
class Universe:
    """A TF x gene universe with its edge ids (global: TF-major grid incl. self-pairs;
    tissue: rows of ``evidence/<t>/evidence.parquet``)."""

    name: str
    tfs: np.ndarray
    genes: np.ndarray
    grid: np.ndarray                 # (n_tf, n_gene) edge id, -1 = not an edge of U
    edge_tf: np.ndarray              # (n_edges,) TF index
    edge_gene: np.ndarray            # (n_edges,) gene index
    ti: dict = field(default_factory=dict)
    gi: dict = field(default_factory=dict)

    @property
    def n_edges(self):
        return len(self.edge_tf)

    def edge_ids(self, tf, target) -> np.ndarray:
        """Edge id of each (tf, target) pair; -1 if the pair is not an edge of U."""
        t = pd.Series(np.asarray(tf, dtype=object)).map(self.ti)
        g = pd.Series(np.asarray(target, dtype=object)).map(self.gi)
        ok = (t.notna() & g.notna()).to_numpy()
        out = np.full(len(ok), -1, dtype=np.int64)
        out[ok] = self.grid[t[ok].astype(np.int64).to_numpy(), g[ok].astype(np.int64).to_numpy()]
        return out

    def edge_pattern(self, variant: str = VARIANT) -> np.ndarray:
        from r2 import evidence as E
        return np.asarray(E.load_edge_pattern(self.name, variant, mmap=False))


_UNIVERSE_CACHE: dict = {}


def load_universe(name: str) -> Universe:
    if name in _UNIVERSE_CACHE:
        return _UNIVERSE_CACHE[name]
    if name == "global":
        d = EV_DIR / "global"
        tfs = np.array((d / "tfs.txt").read_text().split())
        genes = np.array((d / "genes.txt").read_text().split())
        G = len(genes)
        grid = np.arange(len(tfs) * G, dtype=np.int64).reshape(len(tfs), G)
        edge_tf = np.repeat(np.arange(len(tfs)), G)
        edge_gene = np.tile(np.arange(G), len(tfs))
        ep = np.load(d / "edge_pattern.npy", mmap_mode="r")
        assert len(ep) == grid.size, "global edge order mismatch"
    else:
        ev = pd.read_parquet(EV_DIR / name / "evidence.parquet", columns=["tf", "target"])
        tfs = np.array(sorted(ev["tf"].unique()))
        genes = np.array(sorted(set(ev["target"]) | set(ev["tf"])))
        ti0 = {t: i for i, t in enumerate(tfs)}
        gi0 = {g: i for i, g in enumerate(genes)}
        edge_tf = ev["tf"].map(ti0).to_numpy(np.int64)
        edge_gene = ev["target"].map(gi0).to_numpy(np.int64)
        grid = np.full((len(tfs), len(genes)), -1, dtype=np.int64)
        grid[edge_tf, edge_gene] = np.arange(len(ev))
        ep = np.load(EV_DIR / name / "edge_pattern.npy", mmap_mode="r")
        assert len(ep) == len(ev), "tissue edge order mismatch"
    U = Universe(name, tfs, genes, grid, edge_tf, edge_gene,
                 {t: i for i, t in enumerate(tfs)}, {g: i for i, g in enumerate(genes)})
    _UNIVERSE_CACHE[name] = U
    return U


@dataclass
class EvalSet:
    """Evaluable edges of one reference in one universe."""

    universe: str
    ref: str
    edge: np.ndarray        # edge ids of the universe
    tf: np.ndarray          # TF names
    target: np.ndarray      # target names
    y: np.ndarray           # bool, reference positive
    null_p: np.ndarray      # TF x target-degree null probability
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        self.tfs, self.tf_code = np.unique(self.tf, return_inverse=True)
        self.tf_code = self.tf_code.reshape(-1)

    @property
    def n(self):
        return len(self.edge)

    @property
    def n_tf(self):
        return len(self.tfs)

    def subset(self, mask: np.ndarray, null_p: np.ndarray | None = None) -> "EvalSet":
        return EvalSet(self.universe, self.ref, self.edge[mask], self.tf[mask], self.target[mask],
                       self.y[mask], (self.null_p[mask] if null_p is None else null_p), dict(self.meta))

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame({"edge": self.edge, "tf": pd.Categorical(self.tf),
                             "target": pd.Categorical(self.target), "y": self.y,
                             "null_p": self.null_p})


def cohort_ids() -> list:
    return pd.read_csv(C.TABLES / "cohort_table.csv")["cohort_id"].tolist()


def reference_names() -> list:
    return list(CHIP_REFS) + cohort_ids() + ["pert_pooled"]


def _chip_evalset(U: Universe, ref: str) -> EvalSet:
    from r2.build_heldout_chip import reference_index, null_probability
    stem = CHIP_REFS[ref]
    edges = pd.read_parquet(CHIP_DIR / f"{stem}.parquet", columns=["tf", "target"])
    assayed = set((CHIP_DIR / f"{stem}_assayed_tfs.txt").read_text().split())
    gu = set((CHIP_DIR / f"{stem}_gene_universe.txt").read_text().split())
    T = sorted(assayed & set(U.tfs))
    G = sorted(gu & set(U.genes))
    if not T or not G:
        return EvalSet(U.name, ref, *(np.array([], dtype=t) for t in (np.int64, object, object, bool, float)))
    # the target-degree null is fitted on ALL assayed TFs of the reference x the evaluable genes
    # (the reference in-degree of a gene is a property of the reference, not of the universe);
    # evaluable pairs are the assayed TFs inside U (as governance.py)
    R = reference_index(edges, G, assayed=sorted(assayed))
    tix = np.array([U.ti[t] for t in T])
    gix = np.array([U.gi[g] for g in G])
    sub = U.grid[np.ix_(tix, gix)]
    tt, gg = np.nonzero(sub >= 0)
    tf = np.asarray(T, dtype=object)[tt]
    target = np.asarray(G, dtype=object)[gg]
    keep = tf != target
    tf, target, eid = tf[keep], target[keep], sub[tt, gg][keep]
    y = R["M"][np.array([R["ti"][t] for t in tf]), np.array([R["gi"][g] for g in target])]
    null_p = null_probability(R, tf, target)
    meta = {"null_model": {k: v for k, v in R["model"].items() if k in ("n_bins", "converged", "iterations",
                                                                          "max_margin_error_tf")},
            "n_assayed_in_universe": len(T), "n_assayed_null_fit": len(assayed), "n_genes_in_universe": len(G)}
    return EvalSet(U.name, ref, eid, tf, target, y.astype(bool), null_p, meta)


def _cohort_pairs(U: Universe, cohort: str) -> pd.DataFrame:
    """Tested, non-cis Lambert-TF perturbation pairs of a cohort with target in U's genes
    (tf, target, y, edge; edge = -1 when (tf, target) is not an edge of U, e.g. a TF outside U).
    All pairs enter the target-degree null; only edge >= 0 pairs are evaluable."""
    import pyarrow.parquet as pq
    f = COHORT_DIR / cohort / "de.parquet"
    tab = pq.read_table(f, columns=["perturbed_gene", "target", "positive_primary", "cis_same_band"],
                        filters=[("perturbed_gene", "in", sorted(C.lambert_tfs()))])
    d = tab.to_pandas()
    if d.empty:
        return pd.DataFrame(columns=["tf", "target", "y", "edge"])
    d = d[~d["cis_same_band"].astype(bool)]
    d = pd.DataFrame({"tf": d["perturbed_gene"].astype(str).to_numpy(),
                      "target": d["target"].astype(str).to_numpy(),
                      "y": d["positive_primary"].astype(bool).to_numpy()})
    d = d[(d["tf"] != d["target"]) & d["target"].isin(set(U.genes))]
    if d.duplicated(["tf", "target"]).any():
        raise ValueError(f"{cohort}: duplicated (perturbed_gene, target) pairs")
    d["edge"] = U.edge_ids(d["tf"].to_numpy(), d["target"].to_numpy())
    return d.reset_index(drop=True)


def _ragged_evalset(U: Universe, ref: str, d: pd.DataFrame, meta: dict) -> EvalSet:
    """Evaluable set of a perturbation reference: the TF x target-degree null is fitted on all
    pairs of ``d`` (every perturbed Lambert TF, targets in U's genes; as governance.py's
    ``masked_degree_index``), the evaluable pairs are those that are edges of U."""
    empty = EvalSet(U.name, ref, *(np.array([], dtype=t) for t in (np.int64, object, object, bool, float)),
                    meta=meta)
    if d.empty or (d["edge"] >= 0).sum() == 0:
        return empty
    tf = d["tf"].to_numpy(object)
    target = d["target"].to_numpy(object)
    tfs, tcode = np.unique(tf, return_inverse=True)
    genes, gcode = np.unique(target, return_inverse=True)
    from r2.build_heldout_chip import single_thread_blas
    with single_thread_blas():
        null_p, diag = ragged_target_degree_null(tcode, gcode, d["y"].to_numpy(), len(tfs), len(genes))
    meta = dict(meta, null_model=diag, n_tfs_null_fit=int(len(tfs)), n_pairs_null_fit=int(len(d)))
    ok = d["edge"].to_numpy() >= 0
    return EvalSet(U.name, ref, d["edge"].to_numpy(np.int64)[ok], tf[ok], target[ok], d["y"].to_numpy(bool)[ok],
                   null_p[ok], meta)


def build_evalset(U: Universe, ref: str) -> EvalSet:
    if ref in CHIP_REFS:
        return _chip_evalset(U, ref)
    if ref == "pert_pooled":
        parts = []
        for c in cohort_ids():
            d = _cohort_pairs(U, c)
            if len(d):
                parts.append(d.assign(cohort=c))
        if not parts:
            return _ragged_evalset(U, ref, pd.DataFrame(), {})
        a = pd.concat(parts, ignore_index=True)
        g = a.groupby(["tf", "target"], sort=True).agg(edge=("edge", "first"), y=("y", "max"),
                                                        n_cohorts=("cohort", "nunique")).reset_index()
        meta = {"rule": "evaluable in >= 1 cohort; positive if positive in >= 1 evaluable cohort",
                "cohorts": sorted(a["cohort"].unique()),
                "n_pairs_in_multiple_cohorts": int(((g["n_cohorts"] > 1) & (g["edge"] >= 0)).sum())}
        return _ragged_evalset(U, ref, g[["tf", "target", "y", "edge"]], meta)
    d = _cohort_pairs(U, ref)
    return _ragged_evalset(U, ref, d, {"cohort": ref})


try:  # shared analysis helpers (optional; this module is self-contained, as governance.py)
    from r2 import analysis_common as AC  # noqa: F401,E402
except ImportError:  # pragma: no cover
    AC = None


def evalset_path(universe: str, ref: str) -> Path:
    return REF_DIR / universe / f"{ref}.parquet"


_MD5_CACHE: dict = {}


def file_md5(f: Path) -> str:
    """md5 of a file, cached per (path, size, mtime) within the process."""
    import hashlib
    st = f.stat()
    key = (str(f), st.st_size, st.st_mtime_ns)
    if key not in _MD5_CACHE:
        h = hashlib.md5()
        with open(f, "rb") as fh:
            for b in iter(lambda: fh.read(1 << 22), b""):
                h.update(b)
        _MD5_CACHE[key] = h.hexdigest()
    return _MD5_CACHE[key]


def evalset_inputs(universe: str, ref: str) -> list:
    """Files an evaluable set is built from (universe definition + reference)."""
    files = [C.TABLES / "cohort_table.csv"]
    if universe == "global":
        files += [EV_DIR / "global" / x for x in ("tfs.txt", "genes.txt")]
    else:
        files += [EV_DIR / universe / "evidence.parquet"]
    if ref in CHIP_REFS:
        stem = CHIP_REFS[ref]
        files += [CHIP_DIR / f"{stem}{x}" for x in (".parquet", "_assayed_tfs.txt", "_gene_universe.txt")]
    else:
        cohorts = cohort_ids() if ref == "pert_pooled" else [ref]
        files += [COHORT_DIR / c / "de.parquet" for c in cohorts]
    return files


def _relname(f: Path) -> str:
    try:
        return str(Path(f).relative_to(C.SP))
    except ValueError:
        return Path(f).name


def evalset_input_digest(universe: str, ref: str) -> dict:
    return {_relname(f): (file_md5(f) if f.exists() else "missing") for f in evalset_inputs(universe, ref)}


def load_evalset(universe: str, ref: str, rebuild: bool = False) -> EvalSet:
    """Cached evaluable set (``outputs/r2/heldout_eval/refs/<universe>/<ref>.parquet``).  The
    cache is reused only if the md5 of every input file (``evalset_inputs``) equals the digest
    stored when it was built; otherwise it is rebuilt (verifier m6)."""
    p = evalset_path(universe, ref)
    if p.exists() and not rebuild:
        meta = json.loads((p.with_suffix(".json")).read_text())
        if meta.get("input_md5") == evalset_input_digest(universe, ref):
            df = pd.read_parquet(p)
            return EvalSet(universe, ref, df["edge"].to_numpy(np.int64), df["tf"].astype(str).to_numpy(object),
                           df["target"].astype(str).to_numpy(object), df["y"].to_numpy(bool),
                           df["null_p"].to_numpy(np.float64), meta)
        log(f"  {universe}/{ref}: cached evaluable set has no / a different input digest; rebuilding")
    es = build_evalset(load_universe(universe), ref)
    C.ensure_dir(p.parent)
    es.to_frame().to_parquet(p, index=False)
    meta = dict(es.meta, n=int(es.n), n_pos=int(es.y.sum()), n_tfs=int(es.n_tf),
                n_tfs_with_pos=int(len(np.unique(es.tf[es.y]))) if es.n else 0,
                input_md5=evalset_input_digest(universe, ref))
    C.write_json(meta, p.with_suffix(".json"))
    es.meta = meta
    return es


# -----------------------------------------------------------------------------------------
# scores
# -----------------------------------------------------------------------------------------
@dataclass
class Score:
    """A score over a universe: per evidence pattern (``kind='pattern'``) or per edge."""

    name: str
    group: str               # lcm | lcm_globalfit | pattern_baseline | integration | random | method
    kind: str                # 'pattern' | 'edge'
    values: np.ndarray
    is_probability: bool = False

    def at(self, edges: np.ndarray, edge_pattern: np.ndarray) -> np.ndarray:
        if self.kind == "pattern":
            p = edge_pattern[edges]
            out = np.full(len(edges), np.nan)
            ok = p >= 0
            out[ok] = self.values[p[ok]]
            return out
        return np.asarray(self.values[edges], dtype=np.float64)


def fit_path(universe: str, fit_id: str, variant: str = VARIANT) -> Path:
    return FITS_DIR / universe / variant / fit_id


def load_cred(universe: str, fit_id: str, variant: str = VARIANT) -> np.ndarray:
    return np.load(fit_path(universe, fit_id, variant) / "cred.npy")


def load_params(universe: str, fit_id: str, variant: str = VARIANT) -> dict:
    z = np.load(fit_path(universe, fit_id, variant) / "params.npz")
    return {"w": np.asarray(z["w"]), "theta": np.asarray(z["theta"])}


def pattern_data(universe: str, variant: str = VARIANT):
    from r2 import evidence as E
    from r2 import lcm
    return lcm.PatternData(*E.load_patterns(universe, variant))


def credibility_at(params: dict, data) -> np.ndarray:
    """P(z = T | pattern) at (w, theta) (as ``lcm.credibility_at``; numpy only)."""
    obs = (data.R >= 0).astype(float)
    pos = (data.R == 1).astype(float)
    th = np.clip(np.asarray(params["theta"]), 1e-12, 1 - 1e-12)
    lj = np.log(params["w"])[None, :] + (obs * pos) @ np.log(th) + (obs * (1 - pos)) @ np.log1p(-th)
    return np.exp(lj[:, 0] - np.logaddexp.reduce(lj, axis=1))


def integration_scores(U: Universe) -> dict:
    """Held-in integration baselines over all edges of U (0 = not in the held-in resource)."""
    ib = pd.read_parquet(C.OUT / "annotators" / "integration_baselines.parquet",
                         columns=["tf", "target", "dorothea_level_heldin_rank", "in_collectri_heldin",
                                  "collectri_curation_effort_heldin"])
    eid = U.edge_ids(ib["tf"].to_numpy(), ib["target"].to_numpy())
    ok = eid >= 0
    dor = np.zeros(U.n_edges, dtype=np.float32)
    rank = ib["dorothea_level_heldin_rank"].to_numpy(dtype=np.float64, na_value=np.nan)
    m = ok & np.isfinite(rank)
    dor[eid[m]] = 5 - rank[m]                                  # A=4, B=3, C=2, D=1
    col = np.zeros(U.n_edges, dtype=np.float32)
    eff = ib["collectri_curation_effort_heldin"].to_numpy(dtype=np.float64, na_value=np.nan)
    m2 = ok & ib["in_collectri_heldin"].to_numpy(bool)
    col[eid[m2]] = np.where(np.isfinite(eff[m2]), np.maximum(eff[m2], 0.5), 0.5)   # member > non-member
    return {"dorothea_level_heldin": dor, "collectri_heldin": col}


def method_scores(U: Universe) -> dict:
    """GRN-inference method scores over the edges of a tissue universe (NaN = not scored)."""
    t = U.name
    sc = pd.read_parquet(C.OUT / "fm_scores" / t / "scores.parquet")
    eid = U.edge_ids(sc["tf"].to_numpy(), sc["target"].to_numpy())
    out = {}
    for c in FM_ATTENTION + CORRELATIONS + ("random",):
        v = np.full(U.n_edges, np.nan)
        ok = eid >= 0
        v[eid[ok]] = sc[c].to_numpy(np.float64)[ok]
        out[c] = v
    for name, stem in EXPRESSION.items():
        if name in EXPRESSION_TISSUES and t not in EXPRESSION_TISSUES[name]:
            continue
        f = C.OUT / "grnboost2" / t / f"{stem}.parquet"
        if not f.exists():
            continue
        g = pd.read_parquet(f)
        e2 = U.edge_ids(g["tf"].to_numpy(), g["target"].to_numpy())
        v = np.full(U.n_edges, np.nan)
        v[e2[e2 >= 0]] = g["importance"].to_numpy(np.float64)[e2 >= 0]
        out[name] = v
    return out


def status_mechanism_scores(data, cred_family_hb: np.ndarray) -> dict:
    """Post hoc status-mechanism scores (verifier M2), per evidence pattern:
    ``status3`` = 0 unobserved < 1 observed by >= 1 annotator but reported by none < 2 reported;
    ``family_hb_unobserved_last`` = Family-HB credibility on observed patterns, -1 (below every
    observed pattern) on the all-unobserved pattern."""
    obs = (data.R >= 0).any(1)
    rep = (data.R == 1).any(1)
    return {STATUS_SCORE: np.where(rep, 2.0, np.where(obs, 1.0, 0.0)),
            UNOBS_LAST_SCORE: np.where(obs, np.asarray(cred_family_hb, dtype=np.float64), -1.0)}


def comp_adj_null_coverage(n_tfs: int, n_rep: int = 150, n_boot: int = 400, seed=(2026, 10, 14)) -> dict:
    """Simulation check of ``ap_lift_comp_adj`` without within-TF signal (post hoc, heldfix): each
    replicate has ``n_tfs`` TFs (100-1500 edges, positive rate ~ Beta(1.2, 6)) and a score that ranks
    the TFs by their positive rate but is random within TF (3 rate + N(0, 0.3^2), 2 decimals).
    Returns the mean and SD of the composition-adjusted AP lift over replicates and the share of
    95% TF-cluster percentile bootstrap intervals (``n_boot`` resamples) that contain 1."""
    rng = np.random.default_rng(list(seed) + [int(n_tfs)])
    vals, cover = [], []
    for _ in range(int(n_rep)):
        sizes = rng.integers(100, 1500, n_tfs)
        rate = rng.beta(1.2, 6, n_tfs)
        tf = np.repeat(np.arange(n_tfs), sizes)
        y = rng.random(len(tf)) < rate[tf]
        s = np.round(3 * rate[tf] + 0.3 * rng.normal(size=len(tf)), 2)
        lc = level_counts(tf, y, np.zeros(len(y)), s, n_tfs)
        vals.append(curve_metrics(*lc.totals(), ks=())["ap"][0] / exact_null_ap(lc)["ap"][0])
        M = tf_bootstrap_weights(n_tfs, n_boot, rng)
        with np.errstate(invalid="ignore", divide="ignore"):
            lo, hi = percentile_ci(boot_light(lc, M, 500)["ap_lift"] / exact_null_ap(lc, M)["ap_lift"])
        cover.append(lo <= 1 <= hi)
    return dict(n_tfs=int(n_tfs), n_rep=int(n_rep), mean=float(np.mean(vals)), sd=float(np.std(vals, ddof=1)),
                coverage=float(np.mean(cover)))


def fisher_sec6_score(data) -> np.ndarray:
    """Post hoc sensitivity score (verifier ``evalfix`` m1), per evidence pattern: Fisher's
    combination with the plan section 6 per-annotator p values, p = rate_j for a report and
    p = 1 for an observed negative (rate_j = pattern-weighted observed positive rate of annotator
    j, as in ``lcm.baseline_scores``); score = -log P(chi2_{2|O|} >= -2 sum log p), 0 for
    unobserved patterns.  Observed-negative-only patterns score 0 and tie with unobserved
    edges; the mid-p rule of ``lcm.baseline_scores`` ((1 + rate_j) / 2 < 1 for a negative) ranks
    them above unobserved edges instead."""
    from scipy.stats import chi2
    obs = data.R >= 0
    pos = data.R == 1
    w = np.asarray(data.n, dtype=np.float64)[:, None]
    rate = (pos * w).sum(0) / np.maximum((obs * w).sum(0), 1)
    n_obs = obs.sum(1)
    logp = np.where(pos, np.log(np.clip(rate, 1e-300, 1.0))[None, :], 0.0)
    X = -2 * logp.sum(1)
    out = np.where(n_obs > 0, -chi2.logsf(X, 2 * np.maximum(n_obs, 1)), 0.0)
    return out + 0.0                                   # -0.0 -> 0.0


def universe_scores(universe: str, variant: str = VARIANT, methods: bool = True, post_hoc: bool = False) -> dict:
    """All scores of (A) for a universe, in table order (``post_hoc``: append the status-
    mechanism scores)."""
    from r2 import lcm
    U = load_universe(universe)
    data = pattern_data(universe, variant)
    S = {}
    for m in MAP_MODELS:
        S[m] = Score(m, "lcm", "pattern", load_cred(universe, m, variant), True)
    S["family_hb_nuts"] = Score("family_hb_nuts", "lcm", "pattern",
                                np.load(fit_path(universe, "family_hb_nuts", variant) / "cred_mean.npy"), True)
    if universe != "global":
        S["family_hb_globalfit"] = Score("family_hb_globalfit", "lcm_globalfit", "pattern",
                                         credibility_at(load_params("global", PRIMARY, variant), data), True)
    for k, v in lcm.baseline_scores(data).items():
        S[k] = Score(k, "pattern_baseline", "pattern", np.asarray(v, dtype=np.float64))
    for k, v in integration_scores(U).items():
        S[k] = Score(k, "integration", "edge", v)
    if universe == "global":
        S["random"] = Score("random", "random", "edge",
                            _stream("random", "global").integers(0, RANDOM_LEVELS, U.n_edges).astype(np.float32))
    if universe != "global":
        ms = method_scores(U)
        S["random"] = Score("random", "random", "edge", ms.pop("random"))
        if methods:
            for k, v in ms.items():
                S[k] = Score(k, "method", "edge", v)
    if post_hoc:
        for k, v in status_mechanism_scores(data, S[PRIMARY].values).items():
            S[k] = Score(k, "post_hoc_mechanism", "pattern", v)
        S[FISHER_SEC6_SCORE] = Score(FISHER_SEC6_SCORE, "post_hoc_mechanism", "pattern", fisher_sec6_score(data))
    return S


# =========================================================================================
# 3. (A) estimator comparison
# =========================================================================================
def support_flag(es: EvalSet) -> bool:
    n_tf_pos = len(np.unique(es.tf[es.y])) if es.n else 0
    return bool(n_tf_pos < LOW_SUPPORT_TFS or es.y.sum() < LOW_SUPPORT_POS)


def _auroc_point_boot(lc: LevelCounts, M: np.ndarray) -> tuple[float, np.ndarray, int]:
    """Mean per-TF AUROC and its bootstrap replicates (TF weights M, per-TF AUROC held fixed)."""
    auc_t = per_tf_auroc(lc)
    okt = np.isfinite(auc_t)
    with np.errstate(invalid="ignore", divide="ignore"):
        auc_b = (M[:, okt] @ auc_t[okt]) / M[:, okt].sum(1)
    return (float(np.nanmean(auc_t)) if okt.any() else np.nan), auc_b, int(okt.sum())


def subset_metrics(es: EvalSet, s: np.ndarray, mask: np.ndarray, M: np.ndarray) -> tuple[dict, dict]:
    """AP lift, Enr@500 and within-TF AUROC of score s on the evaluable edges ``mask`` (point and
    TF-cluster bootstrap replicates with the TF weights M of the full evaluable set)."""
    mask = np.asarray(mask, dtype=bool)
    if mask.sum() == 0 or es.y[mask].sum() == 0:
        nanb = np.full(M.shape[0], np.nan)
        return (dict(n=int(mask.sum()), n_pos=int(es.y[mask].sum()), n_tfs=int(len(np.unique(es.tf_code[mask]))),
                     base_rate=np.nan, ap_lift=np.nan, enr_at_500=np.nan, within_tf_auroc=np.nan),
                dict(ap_lift=nanb, enr_at_500=nanb, within_tf_auroc=nanb))
    lc = level_counts(es.tf_code[mask], es.y[mask], es.null_p[mask], s[mask], es.n_tf)
    pt = {k: float(v[0]) for k, v in curve_metrics(*lc.totals(), ks=(500,)).items()}
    bm = boot_light(lc, M, 500)
    auc, auc_b, _ = _auroc_point_boot(lc, M)
    return (dict(n=int(mask.sum()), n_pos=int(es.y[mask].sum()), n_tfs=int(len(np.unique(es.tf_code[mask]))),
                 base_rate=pt["base_rate"], ap_lift=pt["ap_lift"], enr_at_500=pt["enr_at_500"], within_tf_auroc=auc),
            dict(ap_lift=bm["ap_lift"], enr_at_500=bm["enr_at_500"], within_tf_auroc=auc_b))


def _diff_cols(prefix: str, a: float, b: float, ra: np.ndarray, rb: np.ndarray) -> dict:
    d = np.asarray(ra) - np.asarray(rb)
    lo, hi = percentile_ci(d)
    return {prefix: a - b, f"{prefix}_lo": lo, f"{prefix}_hi": hi, f"{prefix}_p_z": bootstrap_p_two_sided_z(d, a - b)}


def bootstrap_p_two_sided_z(d: np.ndarray, point: float) -> float:
    return bootstrap_p_z(d, point)


def matched_null_name(name: str) -> str:
    return FM_MATCHED_NULL.get(name, COVERAGE_NULL)


def evaluate_scores(es: EvalSet, scores: dict, edge_pattern: np.ndarray, *, perm: bool = True,
                    tag: str = "", confirm: bool = False, comp_adj: bool = False) -> tuple[list, dict]:
    """Point metrics, TF-cluster bootstrap CIs and within-TF permutation p of every score on
    one evaluable set.  Returns (rows, bootstrap replicates per score for paired contrasts).
    The bootstrap resamples are shared by all scores (paired).  ``confirm``: the confirmatory
    scores (Family-HB and ``CONFIRMATORY_COMPARATORS``) also get B = ``N_BOOT_CONFIRM`` paired
    resamples and ``N_PERM_CONFIRM`` within-TF permutations.  Scores with ``frac_scored < 1`` get
    scored-subset metrics; GRN-inference methods get matched-null contrasts (module docstring).
    ``comp_adj`` (post hoc, heldfix): composition-adjusted AP lift ``ap_lift_comp_adj`` with its
    bootstrap CI (and 10k CI for the confirmatory scores); also kept in the replicates for the
    paired contrasts.  It consumes no random numbers, so every other value is unchanged."""
    M = tf_bootstrap_weights(es.n_tf, N_BOOT, _stream("boot", es.universe, es.ref, tag))
    confirm_set = (PRIMARY,) + CONFIRMATORY_COMPARATORS
    M10 = (tf_bootstrap_weights(es.n_tf, N_BOOT_CONFIRM, _stream("boot10k", es.universe, es.ref, tag))
           if confirm else None)
    rows, reps, svals = [], {}, {}
    n_tf_pos = int(len(np.unique(es.tf[es.y])))
    for name, sc in scores.items():
        t0 = time.time()
        s = sc.at(es.edge, edge_pattern)
        svals[name] = s
        lc = level_counts(es.tf_code, es.y, es.null_p, s, es.n_tf)
        pt = {k: float(v[0]) for k, v in curve_metrics(*lc.totals()).items()}
        auc_pt, auc_b, n_auc = _auroc_point_boot(lc, M)
        bm = boot_metrics(lc, M)
        scored = np.isfinite(s)
        post_hoc = bool(sc.group == "post_hoc_mechanism" or tag)
        row = dict(universe=es.universe, reference=es.ref, score=name, group=sc.group,
                   n_eval=int(es.n), n_pos=int(es.y.sum()), n_tfs=int(es.n_tf), n_tfs_with_pos=n_tf_pos,
                   low_support=support_flag(es), frac_scored=float(np.mean(scored)),
                   n_tie_blocks=int(lc.L), base_rate=pt["base_rate"], ap=pt["ap"], ap_lift=pt["ap_lift"])
        row["ap_lift_lo"], row["ap_lift_hi"] = percentile_ci(bm["ap_lift"])
        for k in EVAL_K:
            row[f"enr_at_{k}"] = pt[f"enr_at_{k}"]
            row[f"enr_at_{k}_lo"], row[f"enr_at_{k}_hi"] = percentile_ci(bm[f"enr_at_{k}"])
        for k in EVAL_K:
            row[f"strat_enr_at_{k}"] = pt[f"strat_enr_at_{k}"]
            row[f"strat_enr_at_{k}_lo"], row[f"strat_enr_at_{k}_hi"] = percentile_ci(bm[f"strat_enr_at_{k}"])
        row["within_tf_auroc"] = auc_pt
        row["within_tf_auroc_lo"], row["within_tf_auroc_hi"] = percentile_ci(auc_b)
        row["n_tfs_auroc"] = n_auc
        row["best_f1"], row["best_precision"] = pt["best_f1"], pt["best_precision"]
        if perm:
            ap_null = permuted_ap(lc, N_NULL, _stream("perm", es.universe, es.ref, name, tag))
            row["perm_ap_null_mean"] = float(np.mean(ap_null))
            row["perm_ap_lift_null_mean"] = float(np.mean(ap_null) / pt["base_rate"])
            row["perm_p"] = float((1 + np.sum(ap_null >= pt["ap"] - 1e-12)) / (len(ap_null) + 1))
            row["perm_p_floor"] = 1.0 / (len(ap_null) + 1)
        row["analysis_status"] = "post_hoc" if post_hoc else "prespecified"
        row["analysis_role"] = ("confirmatory" if (confirm and name in confirm_set) else "descriptive")
        reps[name] = {"ap_lift": bm["ap_lift"], "within_tf_auroc": auc_b,
                      **{f"enr_at_{k}": bm[f"enr_at_{k}"] for k in EVAL_K}}
        reps[name]["_point"] = {"ap_lift": pt["ap_lift"], "within_tf_auroc": auc_pt,
                                **{f"enr_at_{k}": pt[f"enr_at_{k}"] for k in EVAL_K}}
        reps[name]["_group"] = sc.group
        reps[name]["_n_tie_blocks"] = int(lc.L)
        if comp_adj:
            e0 = float(exact_null_ap(lc)["ap"][0])
            row["ap_lift_comp_null"] = e0 / pt["base_rate"]
            row[COMP_ADJ_METRIC] = pt["ap"] / e0
            with np.errstate(invalid="ignore", divide="ignore"):
                ca_b = bm["ap_lift"] / exact_null_ap(lc, M)["ap_lift"]
            row[f"{COMP_ADJ_METRIC}_lo"], row[f"{COMP_ADJ_METRIC}_hi"] = percentile_ci(ca_b)
            reps[name][COMP_ADJ_METRIC] = ca_b
            reps[name]["_point"][COMP_ADJ_METRIC] = row[COMP_ADJ_METRIC]
        if confirm and name in confirm_set:
            b10 = boot_light(lc, M10, 500)
            _, auc10, _ = _auroc_point_boot(lc, M10)
            reps[name]["_10k"] = {"ap_lift": b10["ap_lift"], "within_tf_auroc": auc10, "enr_at_500": b10["enr_at_500"]}
            row["ap_lift_lo_10k"], row["ap_lift_hi_10k"] = percentile_ci(b10["ap_lift"])
            if comp_adj:
                row["within_tf_auroc_lo_10k"], row["within_tf_auroc_hi_10k"] = percentile_ci(auc10)
                row["enr_at_500_lo_10k"], row["enr_at_500_hi_10k"] = percentile_ci(b10["enr_at_500"])
                with np.errstate(invalid="ignore", divide="ignore"):
                    ca10 = b10["ap_lift"] / exact_null_ap(lc, M10)["ap_lift"]
                row[f"{COMP_ADJ_METRIC}_lo_10k"], row[f"{COMP_ADJ_METRIC}_hi_10k"] = percentile_ci(ca10)
                reps[name]["_10k"][COMP_ADJ_METRIC] = ca10
            if perm:
                ap5 = permuted_ap(lc, N_PERM_CONFIRM, _stream("perm5k", es.universe, es.ref, name, tag))
                row["perm_ap_lift_null_mean_5k"] = float(np.mean(ap5) / pt["base_rate"])
                row["perm_p_5k"] = float((1 + np.sum(ap5 >= pt["ap"] - 1e-12)) / (len(ap5) + 1))
                row["perm_p_5k_floor"] = 1.0 / (len(ap5) + 1)
        # coverage-aware metrics (evaluable edges the score ranks; NaN / -inf = not scored)
        if row["frac_scored"] < 1:
            sm, sr = subset_metrics(es, s, scored, M)
            row.update(n_eval_scored=sm["n"], n_pos_scored=sm["n_pos"], n_tfs_scored=sm["n_tfs"],
                       base_rate_scored=sm["base_rate"], ap_lift_scored=sm["ap_lift"],
                       enr_at_500_scored=sm["enr_at_500"], within_tf_auroc_scored=sm["within_tf_auroc"])
            row["ap_lift_scored_lo"], row["ap_lift_scored_hi"] = percentile_ci(sr["ap_lift"])
            reps[name]["_scored"] = sr
        rows.append(row)
        log(f"  {es.universe}/{es.ref}/{name}: ap_lift {pt['ap_lift']:.3f} auroc {auc_pt:.3f} "
            f"({time.time() - t0:.1f}s)")
    # matched-null contrasts of the GRN-inference methods (verifier M5)
    cov_cache: dict = {}
    by_name = {r["score"]: r for r in rows}
    for name, sc in scores.items():
        if sc.group != "method":
            continue
        s = svals[name]
        scored = np.isfinite(s)
        key = np.packbits(scored).tobytes()
        if key not in cov_cache:
            ind = np.where(scored, 1.0, np.nan)
            lc = level_counts(es.tf_code, es.y, es.null_p, ind, es.n_tf)
            pt = {k: float(v[0]) for k, v in curve_metrics(*lc.totals(), ks=(500,)).items()}
            bl = boot_light(lc, M, 500)
            auc, auc_b, _ = _auroc_point_boot(lc, M)
            cov_cache[key] = (dict(ap_lift=pt["ap_lift"], within_tf_auroc=auc, enr_at_500=pt["enr_at_500"]),
                              dict(ap_lift=bl["ap_lift"], within_tf_auroc=auc_b, enr_at_500=bl["enr_at_500"]))
        cov_pt, cov_rep = cov_cache[key]
        row = by_name[name]
        null = matched_null_name(name)
        if null != COVERAGE_NULL and null not in reps:
            null = COVERAGE_NULL
        if null == COVERAGE_NULL:
            n_pt, n_rep = cov_pt, cov_rep
        else:
            n_pt = reps[null]["_point"]
            n_rep = reps[null]
        row.update(matched_null=null, ap_lift_coverage_null=cov_pt["ap_lift"],
                   within_tf_auroc_coverage_null=cov_pt["within_tf_auroc"],
                   ap_lift_matched_null=n_pt["ap_lift"], within_tf_auroc_matched_null=n_pt["within_tf_auroc"],
                   enr_at_500_matched_null=n_pt["enr_at_500"])
        row.update(_diff_cols("ap_lift_minus_matched_null", row["ap_lift"], n_pt["ap_lift"],
                              reps[name]["ap_lift"], n_rep["ap_lift"]))
        row.update(_diff_cols("within_tf_auroc_minus_matched_null", row["within_tf_auroc"], n_pt["within_tf_auroc"],
                              reps[name]["within_tf_auroc"], n_rep["within_tf_auroc"]))
        # scored subset: the matched null evaluated on the method's scored evaluable edges
        if "_scored" in reps[name]:
            sr = reps[name]["_scored"]
            if null == COVERAGE_NULL:       # a constant score on the subset: AP lift 1, AUROC 1/2 exactly
                nm = dict(ap_lift=1.0, within_tf_auroc=0.5)
                nr = dict(ap_lift=np.ones(M.shape[0]), within_tf_auroc=np.full(M.shape[0], 0.5))
            else:
                nm, nr = subset_metrics(es, svals[null], scored, M)
            row.update(ap_lift_scored_matched_null=nm["ap_lift"],
                       within_tf_auroc_scored_matched_null=nm["within_tf_auroc"])
            row.update(_diff_cols("ap_lift_scored_minus_matched_null", row["ap_lift_scored"], nm["ap_lift"],
                                  sr["ap_lift"], nr["ap_lift"]))
            row.update(_diff_cols("within_tf_auroc_scored_minus_matched_null", row["within_tf_auroc_scored"],
                                  nm["within_tf_auroc"], sr["within_tf_auroc"], nr["within_tf_auroc"]))
    if perm and rows:
        # descriptive Holm over the prespecified scores (post hoc scores excluded from the family)
        fam = [i for i, r in enumerate(rows) if r["analysis_status"] == "prespecified" or tag]
        fam = [i for i in fam if rows[i]["group"] != "post_hoc_mechanism"]
        p = holm(np.array([rows[i]["perm_p"] for i in fam]))
        for r in rows:
            r["perm_p_holm"] = np.nan
            r["perm_p_holm_floor"] = np.nan
        for i, v in zip(fam, p):
            rows[i]["perm_p_holm"] = float(v)
            rows[i]["perm_p_holm_floor"] = min(1.0, len(fam) / (N_NULL + 1))
        if confirm:
            ci = [i for i, r in enumerate(rows) if r["score"] in confirm_set and "perm_p_5k" in r]
            p5 = holm(np.array([rows[i]["perm_p_5k"] for i in ci]))
            for i, v in zip(ci, p5):
                rows[i]["perm_p_5k_holm_confirmatory"] = float(v)
                rows[i]["perm_p_5k_holm_floor"] = min(1.0, len(ci) / (N_PERM_CONFIRM + 1))
    return rows, reps


PAIRED_METRICS = ("ap_lift", "within_tf_auroc", "enr_at_500")
CONFIRMATORY_FAMILY_NOTE = ("confirmatory family defined 2026-10-08 after the held-out results had been seen "
                            "(verifier held M1); comparators = plan section 6 EM baseline and baselines")


def _paired_block(es: EvalSet, reps: dict, ref_score: str, metric: str, names: list, tag: str = "",
                  post_hoc_metric: bool = False) -> list:
    """Paired contrasts ref_score - name for one metric (see ``paired_rows``).  ``post_hoc_metric``:
    every row is post hoc (the composition-adjusted AP lift), so only the 6-comparator Holm applies."""
    block = []
    for name in names:
        r = reps[name]
        d = reps[ref_score][metric] - r[metric]
        pa, pb = reps[ref_score]["_point"][metric], r["_point"][metric]
        lo, hi = percentile_ci(d)
        nfin = int(np.isfinite(d).sum())
        post_hoc = bool(r.get("_group") == "post_hoc_mechanism" or tag or post_hoc_metric)
        b = dict(universe=es.universe, reference=es.ref, metric=metric, score_a=ref_score,
                 score_b=name, value_a=pa, value_b=pb, diff=pa - pb, diff_lo=lo, diff_hi=hi,
                 p_boot=bootstrap_p_two_sided(d), n_boot_finite=nfin,
                 low_support=support_flag(es), p_floor=2.0 / (nfin + 1) if nfin else np.nan,
                 p_boot_z=bootstrap_p_z(d, pa - pb),
                 degenerate_constant_score=bool(r.get("_n_tie_blocks", 2) <= 1),
                 analysis_status="post_hoc" if post_hoc else "prespecified")
        if ("_10k" in r and "_10k" in reps[ref_score] and metric in r["_10k"]
                and metric in reps[ref_score]["_10k"]):
            d10 = reps[ref_score]["_10k"][metric] - r["_10k"][metric]
            n10 = int(np.isfinite(d10).sum())
            b.update(n_boot_10k=n10, diff_lo_10k=percentile_ci(d10)[0], diff_hi_10k=percentile_ci(d10)[1],
                     p_boot_10k=bootstrap_p_two_sided(d10), p_boot_z_10k=bootstrap_p_z(d10, pa - pb),
                     p_floor_10k=2.0 / (n10 + 1) if n10 else np.nan)
        block.append(b)
    fam = [i for i, b in enumerate(block) if b["analysis_status"] == "prespecified" or tag]
    fam = [i for i in fam if not block[i]["degenerate_constant_score"]
           and reps[block[i]["score_b"]].get("_group") != "post_hoc_mechanism"]
    p = holm(np.array([block[i]["p_boot"] for i in fam]))
    pz = holm(np.array([block[i]["p_boot_z"] for i in fam]))
    for b in block:
        b.update(p_holm=np.nan, p_z_holm=np.nan, p_holm_floor=np.nan)
    for i, v, vz in zip(fam, p, pz):
        block[i].update(p_holm=float(v), p_z_holm=float(vz),
                        p_holm_floor=min(1.0, len(fam) * block[i]["p_floor"]))
    cf = [i for i, b in enumerate(block) if b["score_b"] in CONFIRMATORY_COMPARATORS and "p_boot_10k" in b]
    if cf:
        ph = holm(np.array([block[i]["p_boot_10k"] for i in cf]))
        phz = holm(np.array([block[i]["p_boot_z_10k"] for i in cf]))
        for i, v, vz in zip(cf, ph, phz):
            block[i].update(p_holm_6comparators=float(v), p_z_holm_6comparators=float(vz),
                            p_holm_6comparators_floor=min(1.0, len(cf) * block[i]["p_floor_10k"]))
            if metric == CONFIRMATORY_METRIC and not tag:
                block[i].update(p_holm_confirmatory=float(v), p_z_holm_confirmatory=float(vz),
                                p_holm_confirmatory_floor=min(1.0, len(cf) * block[i]["p_floor_10k"]),
                                multiplicity_family_defined=CONFIRMATORY_FAMILY_NOTE)
    for b in block:
        b["analysis_role"] = "confirmatory" if "p_holm_confirmatory" in b else "descriptive"
        b["test"] = ("paired TF-cluster percentile bootstrap (B=%d), two-sided p = min(1, 2 min(#D*<=0+1, "
                     "#D*>=0+1)/(B+1)); p_boot_z = bootstrap-SE z-test; *_10k: B=%d" % (N_BOOT, N_BOOT_CONFIRM))
        b["unit"] = "TF (edges clustered within TFs)"
        b["multiplicity"] = ("p_holm / p_z_holm: Holm over the prespecified non-degenerate score_b comparisons "
                             "within universe x reference x metric (descriptive); p_holm_confirmatory: Holm "
                             "within the 6 confirmatory comparators (AP lift, all evaluable edges)")
    return block


def paired_rows(es: EvalSet, reps: dict, ref_score: str = PRIMARY, metrics=PAIRED_METRICS, tag: str = "") -> list:
    """Paired TF-cluster bootstrap contrasts ref_score - other.

    * descriptive (all comparisons): percentile p (B = ``N_BOOT``) with ``p_floor``, Holm within
      universe x reference x metric over the prespecified non-degenerate comparators
      (``p_holm``, ``p_holm_floor``), bootstrap-SE z-test ``p_boot_z`` and its Holm ``p_z_holm``;
    * confirmatory (``CONFIRMATORY_COMPARATORS``, B = ``N_BOOT_CONFIRM`` when available):
      ``p_boot_10k``, ``p_boot_z_10k`` and, for each metric, Holm within the 6 comparators
      (``p_holm_6comparators``); ``analysis_role = confirmatory`` and ``p_holm_confirmatory`` only
      for AP lift on all evaluable edges;
    * post hoc (heldfix, verifier ``evalfix`` M1): when the replicates carry the composition-
      adjusted AP lift (``evaluate_scores(comp_adj=True)``) and the 10k resamples (all evaluable
      edges), ``metric = ap_lift_comp_adj`` contrasts against ``COMP_ADJ_COMPARATORS`` (the 6
      confirmatory comparators with 10k resamples and the 6-comparator Holm; the post hoc
      mechanism scores with B = ``N_BOOT``), all ``analysis_status = post_hoc``, descriptive.
    The percentile p is the primary test, the z-test a secondary check (``primary_test``).
    Constant scores on the evaluable set (one tie block, e.g. union on ``reported_any``) are
    degenerate comparators: reported, never in a multiplicity family."""
    if ref_score not in reps:
        return []
    out = []
    names = [n for n in reps if n != ref_score]
    for metric in metrics:
        out.extend(_paired_block(es, reps, ref_score, metric, names, tag))
    if not tag and COMP_ADJ_METRIC in reps[ref_score] and COMP_ADJ_METRIC in reps[ref_score].get("_10k", {}):
        ca = [n for n in COMP_ADJ_COMPARATORS if n in reps and n != ref_score and COMP_ADJ_METRIC in reps[n]]
        if ca:
            out.extend(_paired_block(es, reps, ref_score, COMP_ADJ_METRIC, ca, tag, post_hoc_metric=True))
    return out


def predicted_set_rows(es: EvalSet, pred: dict, U: Universe, null_of: dict | None = None) -> list:
    """Top-k-per-TF predicted sets (masks over all universe edges) evaluated on es: positives vs
    TF-matched expectation (sum of the TF's evaluable base rate) and vs the TF x target-degree
    null; TF-cluster bootstrap CI and within-TF permutation p (hypergeometric per TF), Holm and BH
    within universe x reference x k (``perm_p_holm`` / ``perm_p_bh``, over the prespecified
    method / estimator sets).  ``null_of`` maps a set name to its matched null set (the 1/L
    uniform null of the same model, or the method's coverage null = top-k of a constant score on
    the scored edges with the seeded tie-break): ratio of the TF-matched enrichments with a paired
    TF-cluster bootstrap CI.  Sets whose name starts with ``coverage::`` are null sets (reported,
    not in the multiplicity family)."""
    out = []
    if es.n == 0:
        return out
    null_of = null_of or {}
    T = es.n_tf
    N_t = np.bincount(es.tf_code, minlength=T).astype(float)
    P_t = np.bincount(es.tf_code, weights=es.y.astype(float), minlength=T)
    base_t = P_t / N_t
    M = tf_bootstrap_weights(T, N_BOOT, _stream("boot_pred", es.universe, es.ref))
    boot = {}
    for (name, k), mask in pred.items():
        inset = mask[es.edge]
        n_t = np.bincount(es.tf_code, weights=inset.astype(float), minlength=T)
        tp_t = np.bincount(es.tf_code, weights=(inset & es.y).astype(float), minlength=T)
        en_t = np.bincount(es.tf_code, weights=np.where(inset, np.nan_to_num(es.null_p), 0.0), minlength=T)
        et_t = n_t * base_t
        tp, n_pred, exp_tf, exp_null = tp_t.sum(), n_t.sum(), et_t.sum(), en_t.sum()
        with np.errstate(invalid="ignore", divide="ignore"):
            b_tf = (M @ tp_t) / (M @ et_t)
            b_null = (M @ tp_t) / (M @ en_t)
            b_prec = (M @ tp_t) / (M @ n_t)
        boot[(name, k)] = b_tf
        rng = _stream("perm_pred", es.universe, es.ref, name, k)
        null_tp = np.zeros(N_NULL)
        for t in np.flatnonzero(n_t > 0):
            null_tp += rng.hypergeometric(int(P_t[t]), int(N_t[t] - P_t[t]), int(n_t[t]), size=N_NULL)
        is_null_set = str(name).startswith("coverage::")
        row = dict(universe=es.universe, reference=es.ref, score=name, topk_per_tf=k,
                   n_eval=int(es.n), n_pos=int(es.y.sum()), n_tfs=int(T), low_support=support_flag(es),
                   n_pred_total=int(mask.sum()), n_pred_eval=int(n_pred), n_tfs_pred_eval=int((n_t > 0).sum()),
                   tp=float(tp), precision=float(tp / n_pred) if n_pred else np.nan,
                   base_rate=float(es.y.mean()), exp_tf_matched=float(exp_tf), exp_null=float(exp_null))
        row["enr_tf_matched"] = float(tp / exp_tf) if exp_tf > 0 else np.nan
        row["enr_tf_matched_lo"], row["enr_tf_matched_hi"] = percentile_ci(b_tf)
        row["enr_null"] = float(tp / exp_null) if exp_null > 0 else np.nan
        row["enr_null_lo"], row["enr_null_hi"] = percentile_ci(b_null)
        row["precision_lo"], row["precision_hi"] = percentile_ci(b_prec)
        row["perm_p"] = float((1 + np.sum(null_tp >= tp - 1e-9)) / (N_NULL + 1)) if n_pred else np.nan
        row["analysis_status"] = "post_hoc" if is_null_set else "prespecified"
        out.append(row)
    by = {(r["score"], r["topk_per_tf"]): r for r in out}
    for (name, k), r in by.items():
        null = null_of.get(name)
        r["matched_null"] = null or ""
        if null is None or (null, k) not in by:
            continue
        nr = by[(null, k)]
        r["null_enr_tf_matched"] = nr["enr_tf_matched"]
        with np.errstate(invalid="ignore", divide="ignore"):
            ratio_b = boot[(name, k)] / boot[(null, k)]
            r["enr_ratio_to_null"] = (r["enr_tf_matched"] / nr["enr_tf_matched"]
                                      if nr["enr_tf_matched"] and np.isfinite(nr["enr_tf_matched"]) else np.nan)
        r["enr_ratio_to_null_lo"], r["enr_ratio_to_null_hi"] = percentile_ci(ratio_b)
        cov = f"coverage::{name}" if not str(name).startswith("coverage::") else None
        cov = null_of.get(f"__coverage__{name}", cov)
        if cov and (cov, k) in by:
            r["coverage_null_enr_tf_matched"] = by[(cov, k)]["enr_tf_matched"]
    for k in sorted({r["topk_per_tf"] for r in out}):
        fam = [r for r in out if r["topk_per_tf"] == k and r["analysis_status"] == "prespecified"
               and np.isfinite(r["perm_p"])]
        ph, pb = holm(np.array([r["perm_p"] for r in fam])), bh(np.array([r["perm_p"] for r in fam]))
        for r, a, b in zip(fam, ph, pb):
            r["perm_p_holm"], r["perm_p_bh"] = float(a), float(b)
            r["perm_p_holm_floor"] = min(1.0, len(fam) / (N_NULL + 1))
    return out


def predicted_sets(U: Universe, scores: dict, edge_pattern: np.ndarray, names, ks=TOPK_ALL,
                   coverage_nulls: bool = False) -> tuple[dict, dict]:
    """Prespecified top-k-per-TF predicted sets over all universe edges (``predicted_sets``).

    ``coverage_nulls``: for every GRN-inference method also the top-k of its coverage indicator
    (``coverage::<method>``: constant score on the edges the method scores, NaN elsewhere, so the
    seeded tie-break draws k scored targets per TF at random; post hoc, verifier M5).  Returns
    (sets, null_of) with null_of[method] = its matched null set name (1/L uniform null of the
    same model for attention variants, else the coverage null)."""
    from r2.predicted_sets import topk_per_tf
    df = pd.DataFrame({"tf": U.tfs[U.edge_tf], "target": U.genes[U.edge_gene]})
    out, null_of, cov_seen = {}, {}, {}
    for name in names:
        v = scores[name].at(np.arange(U.n_edges), edge_pattern)
        df["s"] = v
        for k in ks:
            out[(name, k)] = topk_per_tf(df.rename(columns={"s": name}), name, k).to_numpy()
        if coverage_nulls and scores[name].group == "method":
            cname = f"coverage::{name}"
            key = np.packbits(np.isfinite(v)).tobytes()
            if key in cov_seen:            # identical coverage: reuse the sets of the first method
                for k in ks:
                    out[(cname, k)] = out[(cov_seen[key], k)]
            else:
                df["s"] = np.where(np.isfinite(v), 1.0, np.nan)
                for k in ks:
                    out[(cname, k)] = topk_per_tf(df.rename(columns={"s": cname}), cname, k).to_numpy()
                cov_seen[key] = cname
            nm = FM_MATCHED_NULL.get(name)
            null_of[name] = nm if (nm is not None and nm in names) else cname
    return out, null_of


ESTIMATOR_TABLES = ("estimators", "paired", "methods_context")
PARTIAL_DIR = OUT_DIR / "partials"


def reported_mask(universe: str, es: EvalSet, edge_pattern: np.ndarray, variant: str = VARIANT) -> np.ndarray:
    """Evaluable edges reported by >= 1 held-in annotator (label-free subset on which the
    estimators' rankings actually differ)."""
    data = pattern_data(universe, variant)
    rep = (data.R == 1).any(1)
    p = edge_pattern[es.edge]
    return (p >= 0) & rep[np.maximum(p, 0)]


def run_estimators(universes=UNIVERSES, refs=None, merge: bool = True, subset: str = "all_evaluable") -> None:
    """(A): per-universe partial tables (``outputs/r2/heldout_eval/partials``), merged into
    heldout_estimators.csv, heldout_paired.csv and heldout_methods_context.csv.  Universes can
    be run in parallel processes and merged with ``estimators --merge-partials``.

    ``subset='reported_any'`` repeats the estimator comparison on the evaluable edges with >= 1
    held-in report (supplementary post hoc rows, ``eval_subset`` column).  On all evaluable edges
    the confirmatory resampling (``N_BOOT_CONFIRM``, ``N_PERM_CONFIRM``) and the post hoc
    status-mechanism scores are added."""
    refs = reference_names() if refs is None else refs
    primary_subset = subset == "all_evaluable"
    for u in universes:
        log(f"(A) universe {u} [{subset}]")
        est_rows, pair_rows, pred_rows, sizes = [], [], [], []
        U = load_universe(u)
        ep = U.edge_pattern(VARIANT)
        S = universe_scores(u, methods=primary_subset, post_hoc=primary_subset)
        pred, null_of = {}, {}
        if u != "global" and primary_subset:
            pnames = [n for n, s in S.items() if s.group == "method"] + [PRIMARY, "family_hb_globalfit",
                                                                          "family_hb_nuts", "em", "random",
                                                                          "dorothea_level_heldin"]
            pred, null_of = predicted_sets(U, S, ep, pnames, coverage_nulls=True)
        for ref in refs:
            es = load_evalset(u, ref)
            if subset == "reported_any" and es.n:
                es = es.subset(reported_mask(u, es, ep))
            sizes.append(dict(universe=u, reference=ref, eval_subset=subset, n_eval=int(es.n),
                              n_pos=int(es.y.sum()) if es.n else 0, n_tfs=int(es.n_tf) if es.n else 0,
                              evaluated=bool(es.n and es.y.sum())))
            if es.n == 0 or es.y.sum() == 0 or es.y.all():
                log(f"  {u}/{ref}: not evaluable (n={es.n})")
                continue
            tag = "" if primary_subset else subset
            rows, reps = evaluate_scores(es, S, ep, tag=tag, confirm=primary_subset, comp_adj=True)
            est_rows.extend(rows)
            pair_rows.extend(paired_rows(es, reps, tag=tag))
            if pred:
                pred_rows.extend(predicted_set_rows(es, pred, U, null_of))
        C.ensure_dir(PARTIAL_DIR)
        sfx = "" if primary_subset else f"__{subset}"
        for name, rows in zip(ESTIMATOR_TABLES + ("sizes",), (est_rows, pair_rows, pred_rows, sizes)):
            f = PARTIAL_DIR / f"{name}__{u}{sfx}.csv"
            if rows:
                df = pd.DataFrame(rows)
                if name != "sizes":
                    df.insert(2, "eval_subset", subset)
                df.to_csv(f, index=False, float_format="%.10g")
            elif f.exists():
                f.unlink()
    if merge:
        merge_partials()


def _heldfix_columns_last(df: pd.DataFrame, cols) -> pd.DataFrame:
    """Move the columns added by the heldfix revision behind the pre-existing ones (stable layout)."""
    cols = [c for c in cols if c in df.columns]
    return df[[c for c in df.columns if c not in cols] + cols]


def merge_partials() -> None:
    C.ensure_dir(C.TABLES)
    for name in ESTIMATOR_TABLES + ("sizes",):
        files = [PARTIAL_DIR / f"{name}__{u}{sfx}.csv" for sfx in ("", "__reported_any") for u in UNIVERSES]
        parts = []
        for f in files:
            if f.exists():
                d = pd.read_csv(f)
                if name != "sizes" and "eval_subset" not in d.columns:
                    d.insert(2, "eval_subset", "all_evaluable")
                parts.append(d)
        if not parts:
            continue
        df = pd.concat(parts, ignore_index=True)
        if name == "estimators":
            df = _heldfix_columns_last(df, ("ap_lift_comp_null", COMP_ADJ_METRIC, f"{COMP_ADJ_METRIC}_lo",
                                            f"{COMP_ADJ_METRIC}_hi", "within_tf_auroc_lo_10k", "within_tf_auroc_hi_10k",
                                            "enr_at_500_lo_10k", "enr_at_500_hi_10k", f"{COMP_ADJ_METRIC}_lo_10k",
                                            f"{COMP_ADJ_METRIC}_hi_10k"))
            df["test_perm_p"] = ("one-sided within-TF permutation test of AP (B=%d); unit TF; Holm over the "
                                 "prespecified scores within universe x reference (perm_p_holm, descriptive); "
                                 "confirmatory scores: B=%d (perm_p_5k), Holm over the 7 confirmatory scores "
                                 "(perm_p_5k_holm_confirmatory)" % (N_NULL, N_PERM_CONFIRM))
            df["ci"] = ("TF-cluster percentile bootstrap (B=%d; *_10k: B=%d); *_scored = evaluable edges the "
                        "score ranks; *_matched_null = 1/L uniform null of the same model (attention) or the "
                        "coverage indicator (other methods), paired bootstrap" % (N_BOOT, N_BOOT_CONFIRM))
        if name == "paired":
            df["confirmatory_comparator"] = (df["score_b"].isin(CONFIRMATORY_COMPARATORS)
                                             & (df["eval_subset"] == "all_evaluable")
                                             & df.get("p_holm_6comparators", pd.Series(np.nan, index=df.index)).notna())
            df["primary_test"] = ("percentile bootstrap p (p_boot; p_boot_10k / p_holm_6comparators / "
                                  "p_holm_confirmatory for the confirmatory comparators); p_boot_z* = secondary "
                                  "bootstrap-SE z-test")
        if name == "methods_context":
            df["test_perm_p"] = ("one-sided within-TF permutation test (positives per TF hypergeometric, B=%d); "
                                 "unit TF; perm_p unadjusted, perm_p_holm / perm_p_bh within universe x reference "
                                 "x k over the prespecified sets" % N_NULL)
            df["ci"] = ("TF-cluster percentile bootstrap (B=%d); enr_ratio_to_null = enrichment / matched-null "
                        "enrichment (paired bootstrap CI)" % N_BOOT)
        if "analysis_status" in df.columns:
            df.loc[df["eval_subset"] == "reported_any", "analysis_status"] = "post_hoc"
        if name in POST_HOC_COLUMNS:
            df["post_hoc_columns"] = POST_HOC_COLUMNS[name]
        if name == "sizes":
            df.to_csv(OUT_DIR / "evaluable_sets_estimators.csv", index=False)
            continue
        df.to_csv(C.TABLES / f"{TABLE_PREFIX}{name}.csv", index=False, float_format="%.6g")
    log("merged estimator tables")


# =========================================================================================
# 4. (B) universe-aware decomposition and down-sampling nulls (Q2)
# =========================================================================================
DECOMP_METRICS = tuple(f"p_at_{k}" for k in EVAL_K) + ("best_f1", "best_precision")


def block_totals(score: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(edges, positives) per tie block in decreasing score order."""
    lev, vals = score_levels(score)
    return (np.bincount(lev, minlength=len(vals)).astype(np.int64),
            np.bincount(lev, weights=y.astype(float), minlength=len(vals)).round().astype(np.int64))


def raw_metrics(tot, pos) -> dict:
    m = curve_metrics(tot, pos, None, EVAL_K)
    return {k: np.asarray(v) for k, v in m.items()}


def downsample_null(tot: np.ndarray, pos: np.ndarray, n: int, n_pos: int | None, n_draws: int,
                    rng: np.random.Generator) -> dict:
    """Raw metrics of random subsets of n edges drawn without replacement from a ranked
    population with block counts (tot, pos).  ``n_pos`` given: exactly n_pos positives and
    n - n_pos negatives (prevalence-matched); None: natural prevalence (unmatched).  Ranking
    within the subset is inherited from the population score (tie blocks preserved)."""
    neg = tot - pos
    if n_pos is not None:
        if n_pos > pos.sum() or n - n_pos > neg.sum():
            raise ValueError("population too small for the matched null")
        dp = rng.multivariate_hypergeometric(pos, int(n_pos), size=n_draws, method="marginals")
        dn = rng.multivariate_hypergeometric(neg, int(n - n_pos), size=n_draws, method="marginals")
        dt = dp + dn
    else:
        dt = rng.multivariate_hypergeometric(tot, int(n), size=n_draws, method="marginals")
        dp = np.zeros_like(dt)
        nz = np.flatnonzero(tot > 0)
        dp[:, nz] = rng.hypergeometric(np.broadcast_to(pos[nz], dt[:, nz].shape),
                                       np.broadcast_to(neg[nz], dt[:, nz].shape), dt[:, nz])
    return raw_metrics(dt.astype(float), dp.astype(float))


def tf_block_null(g_tf: np.ndarray, g_lev: np.ndarray, g_y: np.ndarray, L: int, ctx_n: np.ndarray,
                  ctx_k: np.ndarray | None, n_draws: int, rng: np.random.Generator) -> tuple[dict, dict]:
    """TF-cluster-aware down-sampling null (post hoc, verifier M4).

    The global evaluable edges are grouped into TF blocks (``g_tf``; tie level ``g_lev`` in 0..L-1
    of the ranking score, label ``g_y``).  One null draw assembles, for every context TF slot j
    (``ctx_n[j]`` evaluable edges, ``ctx_k[j]`` positives), edges of one global TF block g drawn
    uniformly among the eligible blocks not used yet in this draw (eligible: >= k_j positives and
    >= n_j - k_j negatives; ``ctx_k`` None: >= n_j edges, natural prevalence); if every eligible
    block is already used, a used one is drawn again (counted as reuse); a slot without any
    eligible block is skipped (counted as infeasible).  From g, exactly k_j positives and
    n_j - k_j negatives (or n_j edges) are drawn without replacement and keep their global tie
    level.  Slots are filled in a random order per draw.  Returns (raw metrics per draw,
    diagnostics)."""
    g_tf = np.asarray(g_tf, dtype=np.int64)
    g_lev = np.asarray(g_lev, dtype=np.int64)
    g_y = np.asarray(g_y, dtype=bool)
    G = int(g_tf.max()) + 1
    blocks = []
    for g in range(G):
        m = g_tf == g
        lp, cp_ = np.unique(g_lev[m & g_y], return_counts=True)
        ln, cn = np.unique(g_lev[m & ~g_y], return_counts=True)
        blocks.append((lp, cp_.astype(np.int64), ln, cn.astype(np.int64)))
    P_g = np.array([b[1].sum() for b in blocks])
    Q_g = np.array([b[3].sum() for b in blocks])
    ctx_n = np.asarray(ctx_n, dtype=np.int64)
    S = len(ctx_n)
    if ctx_k is not None:
        ctx_k = np.asarray(ctx_k, dtype=np.int64)
        elig = [np.flatnonzero((P_g >= ctx_k[j]) & (Q_g >= ctx_n[j] - ctx_k[j])) for j in range(S)]
    else:
        elig = [np.flatnonzero(P_g + Q_g >= ctx_n[j]) for j in range(S)]
    pos = np.zeros((n_draws, L))
    neg = np.zeros((n_draws, L))
    n_reuse = np.zeros(n_draws)
    for d in range(n_draws):
        used = np.zeros(G, dtype=bool)
        for j in rng.permutation(S):
            e = elig[j]
            if len(e) == 0:
                continue
            free = e[~used[e]]
            if len(free):
                g = int(free[rng.integers(len(free))])
            else:
                g = int(e[rng.integers(len(e))])
                n_reuse[d] += 1
            used[g] = True
            lp, cp_, ln, cn = blocks[g]
            if ctx_k is not None:
                kp, kn = int(ctx_k[j]), int(ctx_n[j] - ctx_k[j])
                if kp:
                    pos[d, lp] += rng.multivariate_hypergeometric(cp_, kp, method="marginals")
                if kn:
                    neg[d, ln] += rng.multivariate_hypergeometric(cn, kn, method="marginals")
            else:
                x = rng.multivariate_hypergeometric(np.concatenate([cp_, cn]), int(ctx_n[j]), method="marginals")
                pos[d, lp] += x[:len(lp)]
                neg[d, ln] += x[len(lp):]
    diag = dict(n_slots=int(S), n_slots_infeasible=int(sum(len(e) == 0 for e in elig)),
                mean_reused_slots=float(n_reuse.mean()), min_eligible_blocks=int(min((len(e) for e in elig), default=0)))
    return raw_metrics(pos + neg, pos), diag


def _null_row(t, ref, fit, metric, null_type, x_all, vc, vg, ec_n, npos, pool_size, pool_pos, base_null, status,
              unit, extra=None) -> dict:
    x = np.asarray(x_all, dtype=np.float64)
    x = x[np.isfinite(x)]
    med = float(np.median(x))
    gap = vc - vg
    p_hi = (1 + np.sum(x >= vc)) / (len(x) + 1)
    p_lo = (1 + np.sum(x <= vc)) / (len(x) + 1)
    p = p_hi if gap >= 0 else p_lo
    row = dict(tissue=t, reference=ref, credibility=fit, metric=metric, null=null_type,
               n_draws=int(len(x)), subset_size=int(ec_n), subset_pos=(int(npos) if npos is not None else np.nan),
               pool_size=int(pool_size), pool_pos=int(pool_pos),
               value_ctx=vc, value_glob=vg, gap=gap, null_median=med,
               null_lo=float(np.quantile(x, 0.025)), null_hi=float(np.quantile(x, 0.975)),
               null_mean_base_rate=float(base_null),
               gap_reproduced_by_null=(med - vg), frac_gap_surviving=((vc - med) / gap if gap else np.nan),
               p_emp_one_sided=float(p), p_emp_two_sided=float(min(1.0, 2 * min(p_hi, p_lo))),
               test=("empirical p of the context value against the null draws: one-sided in the "
                     "direction of the context-global gap, (1 + #{draws at least as extreme}) / (B + 1), "
                     "and two-sided = 2 x the smaller tail; unit = %s; unadjusted (descriptive)" % unit),
               analysis_status=status)
    if extra:
        row.update(extra)
    return row


def ctx_tf_coverage(ec: EvalSet, eg: EvalSet) -> dict:
    """How much of a context evaluable set comes from TFs that are also evaluable in the global
    universe (post hoc, verifier m7).  Context TFs without held-in literature coverage are outside
    the global universe, so no null drawn from the global evaluable set can contain their edges."""
    in_glob = np.isin(ec.tf, eg.tfs)
    return dict(n_ctx_tfs_in_glob=int(np.isin(ec.tfs, eg.tfs).sum()),
                frac_ctx_edges_tf_in_glob=float(in_glob.mean()) if ec.n else np.nan,
                frac_ctx_pos_tf_in_glob=float(in_glob[ec.y].mean()) if ec.y.any() else np.nan)


NULL_STATUS = {"prevalence_matched": "prespecified", "unmatched": "prespecified",
               "prevalence_matched_ctx_tfs": "post_hoc", "tf_block_matched": "post_hoc",
               "tf_block_size_matched": "post_hoc"}


def run_decomposition(tissues=C.TISSUES, refs=DECOMP_REFERENCES) -> None:
    """(B): heldout_universe_decomposition.csv and heldout_downsampling_null.csv (module
    docstring; the TF-block nulls and the TF-bootstrap CIs are post hoc, verifier M4)."""
    Ug = load_universe("global")
    epg = Ug.edge_pattern(VARIANT)
    dg = pattern_data("global")
    pg = load_params("global", PRIMARY)
    cred_g_global = credibility_at(pg, dg)
    stored = load_cred("global", PRIMARY)
    assert np.max(np.abs(cred_g_global - stored)) < 1e-6, "global credibility does not reproduce cred.npy"
    dec_rows, null_rows = [], []
    glob_boot_cache: dict = {}
    for t in tissues:
        Ut = load_universe(t)
        ept = Ut.edge_pattern(VARIANT)
        dt_ = pattern_data(t)
        pt_ = load_params(t, PRIMARY)
        fits = {"global_fit": (credibility_at(pg, dt_), cred_g_global),
                "context_fit": (load_cred(t, PRIMARY), credibility_at(pt_, dg))}
        assert np.max(np.abs(credibility_at(pt_, dt_) - fits["context_fit"][0])) < 1e-6
        for ref in refs:
            ec, eg = load_evalset(t, ref), load_evalset("global", ref)
            if ec.n == 0 or ec.y.sum() == 0:
                continue
            ctx_tfs = set(ec.tfs)
            ctx_n = np.bincount(ec.tf_code, minlength=ec.n_tf)
            ctx_k = np.bincount(ec.tf_code, weights=ec.y.astype(float), minlength=ec.n_tf).round().astype(np.int64)
            Mc = tf_bootstrap_weights(ec.n_tf, N_BOOT, _stream("boot_decomp_ctx", t, ref))
            Mg = tf_bootstrap_weights(eg.n_tf, N_BOOT, _stream("boot_decomp_glob", ref))
            for fit, (cc, cg) in fits.items():
                sc = cc[ept[ec.edge]]
                sg = cg[epg[eg.edge]]
                tc, pc = block_totals(sc, ec.y)
                tg, pgl = block_totals(sg, eg.y)
                mc, mg = raw_metrics(tc, pc), raw_metrics(tg, pgl)
                bc, bg = float(mc["base_rate"][0]), float(mg["base_rate"][0])
                # TF-cluster bootstrap of the context and global values (independent resamples of each
                # set's TFs; ratio CIs combine them independently, conservative when the sets share TFs)
                bcm = boot_metrics(level_counts(ec.tf_code, ec.y, ec.null_p, sc, ec.n_tf), Mc)
                gkey = (ref, fit)
                if gkey not in glob_boot_cache:
                    glob_boot_cache[gkey] = boot_metrics(level_counts(eg.tf_code, eg.y, eg.null_p, sg, eg.n_tf), Mg)
                bgm = glob_boot_cache[gkey]
                cov = ctx_tf_coverage(ec, eg)
                common = dict(tissue=t, reference=ref, credibility=fit, n_ctx=int(ec.n), n_pos_ctx=int(ec.y.sum()),
                              n_tfs_ctx=int(ec.n_tf), n_glob=int(eg.n), n_pos_glob=int(eg.y.sum()),
                              n_tfs_glob=int(eg.n_tf), base_rate_ctx=bc, base_rate_glob=bg, **cov)
                for metric in DECOMP_METRICS:
                    vc, vg = float(mc[metric][0]), float(mg[metric][0])
                    with np.errstate(invalid="ignore", divide="ignore"):
                        rr = bcm[metric] / bgm[metric]
                        re_ = (bcm[metric] / bcm["base_rate"]) / (bgm[metric] / bgm["base_rate"])
                    row = dict(common, metric=metric, value_ctx=vc, value_glob=vg,
                               ratio_raw=vc / vg if vg else np.nan, ratio_base_rate=bc / bg,
                               enrichment_ctx=vc / bc, enrichment_glob=vg / bg,
                               ratio_enrichment=(vc / bc) / (vg / bg) if vg else np.nan)
                    row["value_ctx_lo"], row["value_ctx_hi"] = percentile_ci(bcm[metric])
                    row["value_glob_lo"], row["value_glob_hi"] = percentile_ci(bgm[metric])
                    row["base_rate_ctx_lo"], row["base_rate_ctx_hi"] = percentile_ci(bcm["base_rate"])
                    row["ratio_raw_lo"], row["ratio_raw_hi"] = percentile_ci(rr)
                    row["ratio_enrichment_lo"], row["ratio_enrichment_hi"] = percentile_ci(re_)
                    row["ci"] = ("TF-cluster percentile bootstrap (B=%d) of each set's TFs; ratio CIs combine the two "
                                 "resamples independently (post hoc, verifier M4)" % N_BOOT)
                    dec_rows.append(row)
                # edge-level nulls: random subsets of the global evaluable edges, ranked by the same credibility
                pools = {"prevalence_matched": (tg, pgl, int(ec.y.sum())),
                         "unmatched": (tg, pgl, None)}
                in_ctx_tf = np.isin(eg.tf, list(ctx_tfs))
                if in_ctx_tf.any():
                    tr, prr = block_totals(sg[in_ctx_tf], eg.y[in_ctx_tf])
                    if prr.sum() >= ec.y.sum() and (tr - prr).sum() >= ec.n - ec.y.sum():
                        pools["prevalence_matched_ctx_tfs"] = (tr, prr, int(ec.y.sum()))
                for null_type, (tt, pp, npos) in pools.items():
                    nm = downsample_null(tt, pp, int(ec.n), npos, N_NULL, _stream("downsample", t, ref, fit, null_type))
                    for metric in DECOMP_METRICS:
                        null_rows.append(_null_row(t, ref, fit, metric, null_type, nm[metric], float(mc[metric][0]),
                                                   float(mg[metric][0]), ec.n, npos, tt.sum(), pp.sum(),
                                                   np.mean(nm["base_rate"]), NULL_STATUS[null_type],
                                                   "random edge subset of the global evaluable set"))
                # TF-cluster-aware nulls (TF blocks of the global evaluable set matched to the context TFs)
                lev_g, _ = score_levels(sg)
                for null_type, kk in (("tf_block_matched", ctx_k), ("tf_block_size_matched", None)):
                    nm, diag = tf_block_null(eg.tf_code, lev_g, eg.y, int(lev_g.max()) + 1, ctx_n, kk, N_NULL,
                                             _stream("downsample_tfblock", t, ref, fit, null_type))
                    for metric in DECOMP_METRICS:
                        null_rows.append(_null_row(t, ref, fit, metric, null_type, nm[metric], float(mc[metric][0]),
                                                   float(mg[metric][0]), ec.n, (int(ctx_k.sum()) if kk is not None else None),
                                                   tg.sum(), pgl.sum(), np.mean(nm["base_rate"]), NULL_STATUS[null_type],
                                                   "TF block of the global evaluable set per context TF",
                                                   extra=dict(n_ctx_tfs=diag["n_slots"],
                                                              n_ctx_tfs_infeasible=diag["n_slots_infeasible"],
                                                              mean_reused_blocks=diag["mean_reused_slots"],
                                                              min_eligible_blocks=diag["min_eligible_blocks"])))
                log(f"(B) {t}/{ref}/{fit}: base {bc:.4f} vs {bg:.4f}")
    C.ensure_dir(C.TABLES)
    dec = pd.DataFrame(dec_rows)
    dec["analysis_status"] = "prespecified"
    dec["post_hoc_columns"] = POST_HOC_COLUMNS["universe_decomposition"]
    dec.to_csv(C.TABLES / f"{TABLE_PREFIX}universe_decomposition.csv", index=False, float_format="%.6g")
    nul = pd.DataFrame(null_rows)
    nul["post_hoc_columns"] = POST_HOC_COLUMNS["downsampling_null"]
    nul.to_csv(C.TABLES / f"{TABLE_PREFIX}downsampling_null.csv", index=False, float_format="%.6g")


# =========================================================================================
# 5. (C) degree / hub calibration (Q9 empirical)
# =========================================================================================
def tf_literature_degree(universe: str = "global", variant: str = VARIANT) -> pd.DataFrame:
    """Number of universe edges of each TF reported by >= 1 held-in LIT annotator, and its
    degree quintile over the universe TFs (quintile = floor(5 (midrank - 1/2) / T), so tied
    degrees share a quintile)."""
    U = load_universe(universe)
    data = pattern_data(universe, variant)
    lit = np.array([f == "LIT" for f in data.family])
    lit_rep = (data.R[:, lit] == 1).any(1)
    ep = U.edge_pattern(variant)
    hit = np.zeros(U.n_edges, dtype=bool)
    ok = ep >= 0
    hit[ok] = lit_rep[ep[ok]]
    deg = np.bincount(U.edge_tf, weights=hit.astype(float), minlength=len(U.tfs)).astype(np.int64)
    return pd.DataFrame({"tf": U.tfs, "lit_degree": deg, "quintile": degree_quintiles(deg)})


def degree_quintiles(deg: np.ndarray, n_groups: int = 5) -> np.ndarray:
    """Quantile group (1..n_groups) of each value by its mid-rank, floor(n (midrank - 1/2) / T) + 1;
    tied values always share a group."""
    from scipy.stats import rankdata
    mid = rankdata(deg, method="average")
    return (np.minimum(n_groups - 1, np.floor(n_groups * (mid - 0.5) / len(deg))).astype(int) + 1)


def topk_weights(score: np.ndarray, k: int) -> np.ndarray:
    """Membership weight of each element in the top k (a tie block cut by the k-th position
    gets the fractional weight (k - edges above) / block size)."""
    lev, vals = score_levels(score)
    cnt = np.bincount(lev, minlength=len(vals))
    ct = np.cumsum(cnt)
    if k >= ct[-1]:
        return np.ones(len(score))
    i = int(np.searchsorted(ct, k, side="left"))
    prev = ct[i - 1] if i > 0 else 0
    w = (lev < i).astype(float)
    w[lev == i] = (k - prev) / cnt[i]
    return w


def run_degree(refs=DEGREE_REFERENCES) -> None:
    """(C): heldout_degree_calibration.csv (global universe)."""
    from r2.robustness import bfdr_set
    deg = tf_literature_degree("global")
    qmap = dict(zip(deg["tf"], deg["quintile"]))
    U = load_universe("global")
    ep = U.edge_pattern(VARIANT)
    data = pattern_data("global")
    S = universe_scores("global")
    rows = []
    bounds = deg.groupby("quintile")["lit_degree"].agg(["min", "max", "size"])
    for ref in refs:
        es = load_evalset("global", ref)
        q = np.array([qmap[t] for t in es.tf])
        q_tf = np.array([qmap[t] for t in es.tfs])
        M = tf_bootstrap_weights(es.n_tf, N_BOOT, _stream("boot_degree", ref))
        P = es.y.sum()
        for name, sc in S.items():
            s = sc.at(es.edge, ep)
            sets = {f"top{k}": topk_weights(s, k) for k in EVAL_K if k < es.n}
            if sc.is_probability and sc.kind == "pattern":
                strong = bfdr_set(sc.values, data.n, BFDR_STRONG)
                sets["strong_bfdr05"] = np.where(ep[es.edge] >= 0, strong[np.maximum(ep[es.edge], 0)], False
                                                 ).astype(float)
            for set_name, w in sets.items():
                wsum = w.sum()
                if wsum <= 0:
                    continue
                # per-TF sums for the bootstrap
                w_t = np.bincount(es.tf_code, weights=w, minlength=es.n_tf)
                p_t = np.bincount(es.tf_code, weights=es.y.astype(float), minlength=es.n_tf)
                wp_t = np.bincount(es.tf_code, weights=w * es.y, minlength=es.n_tf)
                for qq in range(1, 6):
                    inq = q == qq
                    tq = q_tf == qq
                    n_q, pos_q, set_q = int(inq.sum()), float(es.y[inq].sum()), float(w[inq].sum())
                    share_pos = pos_q / P
                    share_set = set_q / wsum
                    with np.errstate(invalid="ignore", divide="ignore"):
                        b_share_set = (M[:, tq] @ w_t[tq]) / (M @ w_t)
                        b_share_pos = (M[:, tq] @ p_t[tq]) / (M @ p_t)
                        b_opi = b_share_set / b_share_pos
                    row = dict(reference=ref, score=name, group=sc.group, set=set_name, set_size=float(wsum),
                               quintile=qq, degree_min=int(bounds.loc[qq, "min"]),
                               degree_max=int(bounds.loc[qq, "max"]), n_tfs_universe=int(bounds.loc[qq, "size"]),
                               n_tfs_eval=int(tq.sum()), n_eval=n_q, n_pos=int(pos_q),
                               share_eval=n_q / es.n, share_pos=share_pos, share_set=share_set,
                               over_promotion=share_set / share_pos if share_pos > 0 else np.nan,
                               pos_rate=pos_q / n_q if n_q else np.nan,
                               set_precision=float((w[inq] * es.y[inq]).sum() / set_q) if set_q > 0 else np.nan,
                               set_precision_all=float((w * es.y).sum() / wsum))
                    row["over_promotion_lo"], row["over_promotion_hi"] = percentile_ci(b_opi)
                    if sc.is_probability:
                        row["mean_cred_eval"] = float(np.nanmean(s[inq])) if n_q else np.nan
                        row["mean_cred_set"] = float(np.nansum(w[inq] * s[inq]) / set_q) if set_q > 0 else np.nan
                        row["pos_rate_over_mean_cred"] = (row["pos_rate"] / row["mean_cred_eval"]
                                                          if row["mean_cred_eval"] else np.nan)
                    rows.append(row)
        log(f"(C) {ref} done")
    df = pd.DataFrame(rows)
    df["few_tfs"] = df["n_tfs_eval"] < FEW_TFS
    df["analysis_status"] = "prespecified"
    df["post_hoc_columns"] = POST_HOC_COLUMNS["degree_calibration"]
    df["ci"] = ("TF-cluster percentile bootstrap (B=%d) with the set held fixed; over_promotion = share of the set "
                "in the quintile / share of evaluable held-out positives in the quintile; few_tfs = fewer than %d "
                "evaluable TFs in the quintile (unstable)" % (N_BOOT, FEW_TFS))
    C.ensure_dir(C.TABLES)
    df.to_csv(C.TABLES / f"{TABLE_PREFIX}degree_calibration.csv", index=False, float_format="%.6g")
    deg.to_csv(OUT_DIR / "tf_literature_degree.csv", index=False)


# =========================================================================================
# 6. (D) sensitivity fits (Q20, observation masks, priors)
# =========================================================================================
POST_HOC_FIT_KINDS = ("informative_observation_model", "posterior_mode")


def sensitivity_kind(variant: str, fit_id: str) -> str:
    """Kind of a stored fit.  ``informative_observation_model`` (eligibility-coded pseudo-variant
    ``*__elig``; ``informative_observation.py``) and ``posterior_mode`` (``family_hb_mixture``,
    ``family_hb_nuts_mode<k>``; ``posterior_modes.py``) are fits of post hoc modules
    (``POST_HOC_FIT_KINDS``); the others belong to the label-free Q20 sensitivity design."""
    if variant.endswith("__elig"):
        return "informative_observation_model"
    if fit_id == "family_hb_mixture" or fit_id.startswith("family_hb_nuts_mode"):
        return "posterior_mode"
    if variant != VARIANT:
        return "evidence_variant"
    if "__fam_" in fit_id:
        return "family_grouping"
    if "__prior_" in fit_id:
        return "prior"
    if "randinit" in fit_id:
        return "nuts_initialisation"
    return "primary_variant_estimator"


def run_sensitivity(universes=SENS_UNIVERSES, refs=SENS_REFERENCES) -> None:
    """(D): heldout_sensitivity_fits.csv.  Every fit directory with a cred.npy is scored on the
    evaluable sets of the references; paired bootstrap difference to the primary Family-HB fit
    (same TF resamples).  For the LIT-removing evidence variants the global universe is also
    evaluated without the TFs / genes that only the removed evidence puts into the universe."""
    rows = []
    for u in universes:
        U = load_universe(u)
        scores = {}
        for vdir in sorted(p for p in (FITS_DIR / u).iterdir() if p.is_dir()):
            variant = vdir.name
            ep_v = U.edge_pattern(variant)
            for fdir in sorted(p for p in vdir.iterdir() if p.is_dir() and (p / "cred.npy").exists()):
                key = f"{variant}/{fdir.name}"
                sc = Score(key, "fit", "pattern", np.load(fdir / "cred.npy"), True)
                sc.edge_pattern = ep_v
                scores[key] = sc
        primary_key = f"{VARIANT}/{PRIMARY}"
        excl = {}
        if u == "global":
            for v in ("lit_no_ht", "lit_no_chipderived"):
                f = EV_DIR / "global" / f"lit_only_via_removed__{v}.json"
                if f.exists():
                    j = json.loads(f.read_text())
                    excl[v] = (set(j.get("tfs", [])), set(j.get("genes", [])))
        for ref in refs:
            es = load_evalset(u, ref)
            if es.n == 0:
                continue
            sets = {"all_evaluable": (es, list(scores))}
            for v, (xt, xg) in excl.items():
                m = ~(np.isin(es.tf, list(xt)) | np.isin(es.target, list(xg)))
                keys = [primary_key] + [k for k in scores if k.startswith(v + "/")]
                sets[f"excl_lit_only_via_removed__{v}"] = (es.subset(m), keys)
            for set_name, (e2, keys) in sets.items():
                res, reps = evaluate_sensitivity(e2, {k: scores[k] for k in keys}, tag=set_name)
                for k, r in res.items():
                    variant, fit_id = k.split("/", 1)
                    d = reps[k] - reps[primary_key]
                    lo, hi = percentile_ci(d)
                    row = dict(universe=u, reference=ref, eval_set=set_name, variant=variant, fit_id=fit_id,
                               kind=sensitivity_kind(variant, fit_id), n_eval=int(e2.n), n_pos=int(e2.y.sum()),
                               n_tfs=int(e2.n_tf), low_support=support_flag(e2), **r,
                               diff_vs_primary=r["ap_lift"] - res[primary_key]["ap_lift"],
                               diff_vs_primary_lo=lo, diff_vs_primary_hi=hi,
                               p_diff_boot=bootstrap_p_two_sided(d) if k != primary_key else np.nan,
                               p_diff_z=(bootstrap_p_z(d, r["ap_lift"] - res[primary_key]["ap_lift"])
                                         if k != primary_key else np.nan),
                               analysis_status=("post_hoc" if sensitivity_kind(variant, fit_id) in POST_HOC_FIT_KINDS
                                                else "prespecified"))
                    fdir = FITS_DIR / u / variant / fit_id
                    row["fit_mtime"] = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime((fdir / "cred.npy").stat().st_mtime))
                    if (fdir / "draws.npz").exists():
                        row.update(posterior_draw_metrics(e2, scores[k].edge_pattern, u, variant, fdir))
                    row.update(fit_convergence(fdir))
                    rows.append(row)
            log(f"(D) {u}/{ref} done")
    df = pd.DataFrame(rows)
    df = _heldfix_columns_last(df, ("fit_r_hat_max", "fit_init", "mode_mixing"))
    df["test"] = ("AP lift with TF-cluster percentile bootstrap CI (B=%d); diff_vs_primary = AP lift(fit) - AP lift"
                  "(primary Family-HB MAP), paired bootstrap CI, two-sided bootstrap p (p_diff_boot, floor "
                  "2/(B+1)) and bootstrap-SE z-test p (p_diff_z), unadjusted (sensitivity analysis); NUTS fits: "
                  "ap_lift_post_* / enr_at_500_post_* = quantiles over the stored posterior draws of the plug-in "
                  "credibility (post hoc, verifier m5); fit_mtime = local modification time of the fit's "
                  "cred.npy; kind informative_observation_model / posterior_mode = fits of post hoc modules" % N_BOOT)
    df["convergence_note"] = ("NUTS fits: fit_r_hat_max = R-hat max of the fit's meta.json (convergence.r_hat_max, "
                              "else r_hat_max), fit_init = initialisation; mode_mixing = fit_r_hat_max > %g: a "
                              "non-converged mixture over posterior modes (random initialisation), whose AP-lift "
                              "shift reflects mode mixing, not the varied prior / initialisation (post hoc, "
                              "verifier evalfix m6); empty for MAP fits" % MODE_MIXING_RHAT)
    df["post_hoc_columns"] = POST_HOC_COLUMNS["sensitivity_fits"]
    C.ensure_dir(C.TABLES)
    df.to_csv(C.TABLES / f"{TABLE_PREFIX}sensitivity_fits.csv", index=False, float_format="%.6g")


def fit_convergence(fdir: Path) -> dict:
    """R-hat max, initialisation and mode-mixing flag of a stored NUTS fit (``meta.json``; post hoc,
    verifier ``evalfix`` m6).  MAP fits (no ``draws.npz``) get NaN / empty values."""
    out = dict(fit_r_hat_max=np.nan, fit_init="", mode_mixing="")
    f = fdir / "meta.json"
    if not (fdir / "draws.npz").exists() or not f.exists():
        return out
    try:
        meta = json.loads(f.read_text())
    except (OSError, ValueError):
        return out
    conv = meta.get("convergence") if isinstance(meta.get("convergence"), dict) else {}
    rh = conv.get("r_hat_max", meta.get("r_hat_max"))
    init = meta.get("init")
    init = init.get("init", "") if isinstance(init, dict) else (init or "")
    if rh is not None and np.isfinite(float(rh)):
        out.update(fit_r_hat_max=float(rh), mode_mixing=bool(float(rh) > MODE_MIXING_RHAT))
    out["fit_init"] = str(init)
    return out


def posterior_draw_metrics(es: EvalSet, edge_pattern: np.ndarray, universe: str, variant: str, fdir: Path,
                           n_draws: int = N_POSTERIOR_DRAWS) -> dict:
    """AP lift and Enr@500 of the plug-in credibility of each stored NUTS draw (w_d, theta_d) on
    the evaluable set (pattern-level aggregation; tie blocks over patterns as everywhere): the
    posterior uncertainty of the credibility propagated into the held-out metrics."""
    z = np.load(fdir / "draws.npz")
    W, TH = np.asarray(z["w"]), np.asarray(z["theta"])
    D = min(n_draws, len(W))
    data = pattern_data(universe, variant)
    p = edge_pattern[es.edge]
    ok = p >= 0
    P = data.R.shape[0]
    tot_p = np.bincount(p[ok], minlength=P).astype(float)
    pos_p = np.bincount(p[ok], weights=es.y[ok].astype(float), minlength=P)
    pres = np.flatnonzero(tot_p > 0)
    n_unscored = float((~ok).sum())          # evaluable edges without a pattern rank last (none expected)
    ap, enr = np.full(D, np.nan), np.full(D, np.nan)
    for d in range(D):
        c = credibility_at({"w": W[d], "theta": TH[d]}, data)[pres]
        lev, vals = score_levels(c)
        tot = np.bincount(lev, weights=tot_p[pres], minlength=len(vals))
        pos = np.bincount(lev, weights=pos_p[pres], minlength=len(vals))
        if n_unscored:
            y_un = float(es.y[~ok].sum())
            tot, pos = np.r_[tot, n_unscored], np.r_[pos, y_un]
        m = curve_metrics(tot, pos, None, ks=(500,))
        ap[d], enr[d] = float(m["ap_lift"][0]), float(m["enr_at_500"][0])
    return dict(n_posterior_draws=int(D), ap_lift_post_q025=float(np.quantile(ap, 0.025)),
                ap_lift_post_median=float(np.median(ap)), ap_lift_post_q975=float(np.quantile(ap, 0.975)),
                enr_at_500_post_q025=float(np.quantile(enr, 0.025)), enr_at_500_post_median=float(np.median(enr)),
                enr_at_500_post_q975=float(np.quantile(enr, 0.975)))


def evaluate_sensitivity(es: EvalSet, scores: dict, tag: str) -> tuple[dict, dict]:
    M = tf_bootstrap_weights(es.n_tf, N_BOOT, _stream("boot_sens", es.universe, es.ref, tag))
    res, reps = {}, {}
    for k, sc in scores.items():
        s = sc.at(es.edge, sc.edge_pattern)
        lc = level_counts(es.tf_code, es.y, es.null_p, s, es.n_tf)
        pt = curve_metrics(*lc.totals())
        bm = boot_metrics(lc, M, ks=(500,))
        auc_t = per_tf_auroc(lc)
        lo, hi = percentile_ci(bm["ap_lift"])
        res[k] = dict(ap=float(pt["ap"][0]), ap_lift=float(pt["ap_lift"][0]), ap_lift_lo=lo, ap_lift_hi=hi,
                      enr_at_500=float(pt["enr_at_500"][0]) if 500 in EVAL_K else np.nan,
                      strat_enr_at_500=float(pt["strat_enr_at_500"][0]) if 500 in EVAL_K else np.nan,
                      within_tf_auroc=float(np.nanmean(auc_t)) if np.isfinite(auc_t).any() else np.nan)
        reps[k] = bm["ap_lift"]
    return res, reps


# =========================================================================================
# 6b. evidence status vs held-out positives (interpretation of the MAR observation model)
# =========================================================================================
STATUS_REFERENCES = ("encode", "encode_strict", "enets2", "norman", "papalexi", "replogle", "pert_pooled")


def run_evidence_status(universes=UNIVERSES, refs=STATUS_REFERENCES) -> None:
    """heldout_evidence_status.csv: held-out positive rate of evaluable edges by evidence status
    (``unobserved`` = no held-in annotator observes the edge; ``observed_neg`` = observed by >= 1,
    reported by none; ``reported`` = >= 1 report) vs the TF-matched expectation (mean over the
    status class of the TF's evaluable positive rate), O/E with TF-cluster bootstrap CI, and the
    mean Family-HB MAP credibility of the class.  The latent-class models treat unobserved
    entries as missing at random, so an unobserved edge keeps the prior credibility while an
    observed negative is down-weighted; this table checks that ordering against held-out data."""
    rows = []
    for u in universes:
        U = load_universe(u)
        ep = U.edge_pattern(VARIANT)
        data = pattern_data(u)
        obs = (data.R >= 0).any(1)
        rep = (data.R == 1).any(1)
        cred = load_cred(u, PRIMARY)
        for ref in refs:
            es = load_evalset(u, ref)
            if es.n == 0 or es.y.sum() == 0:
                continue
            pp = ep[es.edge]
            st = np.where(rep[pp], 2, np.where(obs[pp], 1, 0))
            T = es.n_tf
            N_t = np.bincount(es.tf_code, minlength=T).astype(float)
            P_t = np.bincount(es.tf_code, weights=es.y.astype(float), minlength=T)
            base_e = (P_t / N_t)[es.tf_code]
            M = tf_bootstrap_weights(T, N_BOOT, _stream("boot_status", u, ref))
            for k, name in enumerate(("unobserved", "observed_neg", "reported")):
                m = st == k
                if not m.any():
                    continue
                o_t = np.bincount(es.tf_code[m], weights=es.y[m].astype(float), minlength=T)
                e_t = np.bincount(es.tf_code[m], weights=base_e[m], minlength=T)
                with np.errstate(invalid="ignore", divide="ignore"):
                    b = (M @ o_t) / (M @ e_t)
                lo, hi = percentile_ci(b)
                rows.append(dict(universe=u, reference=ref, status=name, n_eval=int(m.sum()),
                                 n_tfs=int(len(np.unique(es.tf_code[m]))), n_pos=int(es.y[m].sum()),
                                 pos_rate=float(es.y[m].mean()), exp_tf_matched_rate=float(base_e[m].mean()),
                                 oe_tf_matched=float(o_t.sum() / e_t.sum()) if e_t.sum() > 0 else np.nan,
                                 oe_lo=lo, oe_hi=hi, mean_cred_family_hb=float(np.mean(cred[pp[m]])),
                                 low_support=support_flag(es)))
        log(f"(status) {u} done")
    df = pd.DataFrame(rows)
    df["analysis_status"] = "post_hoc"
    df["post_hoc_columns"] = POST_HOC_COLUMNS["evidence_status"]
    df["ci"] = "TF-cluster percentile bootstrap (B=%d) of O/E (TF-matched expectation)" % N_BOOT
    C.ensure_dir(C.TABLES)
    df.to_csv(C.TABLES / f"{TABLE_PREFIX}evidence_status.csv", index=False, float_format="%.6g")


# =========================================================================================
# 7. Command line
# =========================================================================================
def build_refs(universes=UNIVERSES, refs=None) -> None:
    refs = reference_names() if refs is None else refs
    rows = []
    for u in universes:
        for r in refs:
            es = load_evalset(u, r, rebuild=True)
            rows.append(dict(universe=u, reference=r, **{k: v for k, v in es.meta.items()
                                                       if k in ("n", "n_pos", "n_tfs", "n_tfs_with_pos")}))
            log(f"refs {u}/{r}: n={es.n} pos={int(es.y.sum()) if es.n else 0} tfs={es.n_tf if es.n else 0}")
    pd.DataFrame(rows).to_csv(OUT_DIR / "evaluable_sets.csv", index=False)


def write_run_meta(stage: str, runtime_s: float) -> None:
    """``outputs/r2/heldout_eval/meta_<stage>.json``: settings, the decision log and md5 of every
    input read, including the sensitivity stage's variant patterns / fits (verifier m6)."""
    inputs = [C.TABLES / "cohort_table.csv", C.OUT / "annotators" / "integration_baselines.parquet"]
    inputs += [CHIP_DIR / f"{v}{x}" for v in CHIP_REFS.values()
               for x in (".parquet", "_assayed_tfs.txt", "_gene_universe.txt")]
    inputs += [COHORT_DIR / c / "de.parquet" for c in cohort_ids()]
    for u in UNIVERSES:
        inputs += sorted((FITS_DIR / u).glob("*/*/cred*.npy"))          # every variant and fit (sensitivity)
        inputs += sorted((FITS_DIR / u).glob("*/*/params.npz"))
        inputs += sorted((FITS_DIR / u).glob("*/*/draws.npz"))
        inputs += sorted((EV_DIR / u).glob("patterns*.parquet")) + sorted((EV_DIR / u).glob("edge_pattern*.npy"))
        inputs += sorted((EV_DIR / u).glob("lit_only_via_removed__*.json"))
        if u != "global":
            inputs += [EV_DIR / u / "evidence.parquet"]
            inputs += [C.OUT / "fm_scores" / u / "scores.parquet"] + sorted((C.OUT / "grnboost2" / u).glob("*.parquet"))
        else:
            inputs += [EV_DIR / u / "tfs.txt", EV_DIR / u / "genes.txt"]
    meta = dict(stage=stage, runtime_s=round(runtime_s, 1), created=time.strftime("%Y-%m-%dT%H:%M:%S"),
                settings=dict(seed=SEED, n_bootstrap=N_BOOT, n_null_draws=N_NULL, eval_k=EVAL_K, topk=TOPK_ALL,
                              variant=VARIANT, tie_decimals=TIE_DECIMALS, k_min_precision=K_MIN_PRECISION,
                              random_levels_global=RANDOM_LEVELS, low_support=(LOW_SUPPORT_TFS, LOW_SUPPORT_POS),
                              n_boot_confirm=N_BOOT_CONFIRM, n_perm_confirm=N_PERM_CONFIRM,
                              confirmatory_comparators=list(CONFIRMATORY_COMPARATORS),
                              confirmatory_metric=CONFIRMATORY_METRIC, fm_matched_null=FM_MATCHED_NULL,
                              target_degree=(PRESPEC["target_degree_bins"], PRESPEC["target_degree_bin_rule"],
                                             PRESPEC["target_degree_definition"])),
                analysis_decisions=ANALYSIS_DECISIONS,
                analysis_common_available=AC is not None,
                inputs={_relname(f): file_md5(f) for f in dict.fromkeys(inputs) if f.exists()},
                versions=dict(numpy=np.__version__, pandas=pd.__version__, scipy=__import__("scipy").__version__))
    C.write_json(meta, OUT_DIR / f"meta_{stage}.json")


DECISIONS_TABLE = C.TABLES / "eval_analysis_decisions.csv"


def record_decisions(module: str, decisions: list, path: Path = DECISIONS_TABLE) -> None:
    """Replace this module's rows of the shared dated decision log (``eval_analysis_decisions.csv``).

    The read-modify-write runs under an exclusive ``fcntl`` lock (``outputs/r2/locks/<name>.lock``
    for the project table, ``<path>.lock`` otherwise) and the table is replaced atomically
    (temporary file + ``os.replace``), so concurrent writers that use this function cannot drop
    each other's rows (verifier ``evalfix`` m10)."""
    import fcntl
    import os
    import tempfile
    cols = ["module", "id", "date", "status", "item", "note"]
    C.ensure_dir(path.parent)
    path = Path(path)
    if path.resolve() == DECISIONS_TABLE.resolve():
        lock = C.ensure_dir(C.OUT / "locks") / (path.name + ".lock")
    else:
        lock = path.with_name(path.name + ".lock")
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


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("stage", choices=["refs", "estimators", "decomposition", "degree", "status", "sensitivity",
                                      "all"])
    ap.add_argument("--universes", nargs="+", default=list(UNIVERSES))
    ap.add_argument("--refs", nargs="+", default=None)
    ap.add_argument("--subset", choices=["all_evaluable", "reported_any"], default="all_evaluable",
                    help="estimators: evaluate on all evaluable edges (primary) or on those with >= 1 held-in report")
    ap.add_argument("--merge-partials", action="store_true",
                    help="estimators: merge per-universe partial tables written by parallel runs")
    a = ap.parse_args(argv)
    C.ensure_dir(OUT_DIR)
    t0 = time.time()
    if a.stage in ("refs", "all"):
        build_refs(a.universes, a.refs)
    if a.stage == "all":
        run_estimators(a.universes, a.refs, merge=False)
        run_estimators(a.universes, a.refs, subset="reported_any")
    elif a.stage == "estimators":
        if a.merge_partials:
            merge_partials()
        else:
            run_estimators(a.universes, a.refs, subset=a.subset)
    if a.stage in ("decomposition", "all"):
        run_decomposition([u for u in a.universes if u != "global"] or C.TISSUES)
    if a.stage in ("degree", "all"):
        run_degree()
    if a.stage in ("status", "all"):
        run_evidence_status()
    if a.stage in ("sensitivity", "all"):
        run_sensitivity()
    write_run_meta(a.stage if not (a.stage == "estimators" and a.subset != "all_evaluable") else
                   f"estimators_{a.subset}", time.time() - t0)
    record_decisions("heldout_eval", ANALYSIS_DECISIONS)
    log(f"done {a.stage} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
