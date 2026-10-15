"""Optimism of label-based configuration tuning (reviewer Q6; global universe).

Question
--------
In R1 the Family-HB hyperparameters were chosen with a joint objective whose perturbation
term was computed on the perturbation panels that were later reported as validation.  How
optimistic is that practice?  If a configuration is chosen because it ranks perturbation
targets best on some labels, how much of its advantage survives on labels that were not
used for the choice, and does it beat the label-free choices of the rebuilt study (the
prespecified Family-HB, and selection by held-out marginal likelihood) out of sample?

This is the only R2 module that uses held-out perturbation labels for a *choice*.  It does
so deliberately, to measure the optimism of that choice; nothing it selects is used
anywhere else in the pipeline.

Configuration grid (``config_grid``; global universe, MAP fits of the fit-storage contract
``outputs/r2/fits/global/<variant>/<fit_id>``; a missing fit is fitted here by MAP with
``N_RESTARTS_MISSING`` = 10 restarts and stored per the contract)
------------------------------------------------------------------------------------
* ``estimator``         em, dependence, multifactor, family_regularized, family_hb
                        (primary evidence variant, primary prior);
* ``evidence_variant``  family_hb under every ``PRESPECIFIED['evidence_variants_sensitivity']``
                        variant (config id ``family_hb@<variant>``);
* ``prior``             family_hb__prior_scale0.5 / _scale2 / _dir0.5 / _dir2;
* ``grouping``          family_hb__fam_singleton / _lineage / _datadriven.

Score of an edge under a configuration = its credibility P(T | pattern) (``cred.npy``,
indexed by the variant's ``edge_pattern``).  NaN would rank last (none occur).

Perturbation labels (``build_labels``)
--------------------------------------
For every cohort of ``data/r2/tables/cohort_table.csv`` (``outputs/r2/cohorts/<c>/de.parquet``)
a tested pair (perturbed_gene, target) is *evaluable* for the global universe if the
perturbed gene is a universe TF, the target is a universe gene, they differ and the pair is
not ``cis_same_band`` (cis pairs are dropped from positives and negatives).  Its label is
``positive_primary``.  ``pert_pooled``: an edge is evaluable if it is evaluable in >= 1 cohort
and positive if it is positive in >= 1 cohort.  Study-level label sets (LOSO) apply the same
pooling within ``PRESPECIFIED['cohort_cluster_unit']`` (study_id: Dixit d7 + d13 are one study).

Metric
------
AP = sum_b (R_b - R_{b-1}) P_b over the tie blocks b of the score in decreasing order
(precision P_b and recall R_b after including the whole block; the step-function average
precision of scikit-learn, every tie block one threshold; scores rounded to ``TIE_DECIMALS``).
AP lift = AP / prevalence of positives among the evaluable edges.  Two tuning criteria:

* ``pooled_ap_lift``   AP lift of one ranking of all evaluable edges of the selected TFs
                       (primary; the R1-style reward);
* ``tf_geomean_ap_lift`` geometric mean over TFs with >= 1 positive of the TF's own AP lift
                       (within-TF ranking, every TF weighted equally; removes TF-composition
                       effects; sensitivity).  The geometric mean is used because per-TF lifts
                       are heavy-tailed (a TF with 2 positives among ~4,000 edges reaches a
                       lift of several hundred), so an arithmetic mean is driven by one or two TFs.

Revision 2026-10-08 (verifier ``tune``; ``analysis_status = post_hoc`` in the tables)
------------------------------------------------------------------------------------
* Composition-matched criteria.  A pooled AP lift mixes ranking within TFs with TF composition,
  and the per-TF geometric mean is not centred at 1 under a random within-TF order (its null
  depends on how many tie blocks a configuration has).  ``N_NULL_TUNE`` within-TF label
  permutations (the same permutations for every configuration, common random numbers) give
  ``pooled_ap_lift_nullnorm`` = pooled AP lift / its permutation-null mean on the same edges, and
  ``tf_geomean_ap_lift_nullnorm`` = 2 ^ mean over TFs of (log2 lift_t - E_null[log2 lift_t]);
  both are tuning criteria (selection is made on them) next to the raw ones.
* ``random_tf_half_stratified``: 50 halves stratified by each TF's dominant study (the study with
  most of its evaluable edges), so tuning and held-out halves have similar study mixes.
* LOSO ``fold_set``: all folds (as before), ``excl_low_support`` (held-out studies whose cohorts
  carry the decision-log item 7 flag "essentially no positives", i.e. Dixit, are not left out),
  ``weighted_by_heldout_positives`` and every single study (``study:<id>``).
* ``family_hb@no_motif_coexp`` is excluded from the grid (``EXCLUDED_CONFIGS``: its true-edge and
  burst classes are exchangeable, so its credibility is not identified); 19 configurations.
* ``tune_selection_frequency.csv``: full-label difference of each configuration to the primary
  with a TF-cluster bootstrap CI (pooled AP lift; per-TF null-normalised geometric mean).
* ``tune_status_decomposition.csv``: TF-matched expected positives and O/E per evidence status, and
  the within-TF Mantel-Haenszel odds ratio of each status against unobserved edges.
* Part 3: selection-change summaries also with em and family_regularized merged (their
  credibilities have Spearman 0.999996, so flips between them are not real changes).
* The switch of the per-TF criterion from the arithmetic to the geometric mean was made after
  outcomes were seen (SRF lift 342), so ``tf_geomean_ap_lift`` is labelled post hoc.
* LOSO folds (added 2026-10-13): ``heldout_pooled_diff_vs_primary`` = pooled AP lift of the selected
  configuration minus that of the primary on the held-out study, with a TF-cluster percentile
  bootstrap CI (``PRESPECIFIED['n_bootstrap']`` resamples of the held-out study's TFs; the selection
  is held fixed, so the interval describes the held-out contrast, not the selection step) and a
  bootstrap-SE z-test; in ``tune_optimism.csv`` on the single-study rows (``fold_set = study:<id>``).
* Every table has ``analysis_status`` and ``post_hoc_columns``.  The label-free CV table and the
  label-set table use no label-based choice (``prespecified``); part 3 as a whole re-creates the R1
  objective with a design fixed before its outputs existed (``prespecified``; merged columns post hoc).

Label-based tuning (part 1)
---------------------------
* ``random_tf_half``: ``N_SPLITS`` = 50 seeded random halves of the perturbed TFs that have >= 1
  evaluable edge (tuning TFs = floor(n/2), held-out TFs = the rest; ``PRESPECIFIED['random_seed']``).
  The criterion is computed on the tuning TFs for every configuration, the best one is selected
  (ties: first in grid order), and it is evaluated on the held-out TFs.
* ``loso``: leave one study out; tune on the pooled labels of the other studies, evaluate on
  the left-out study's labels (studies with >= 1 evaluable positive are left out in turn).
  TFs may overlap between studies, so LOSO measures cross-study transfer.

Two grids: ``all`` (the 20 configurations) and ``primary_evidence`` (the 12 configurations fitted to
the primary evidence: estimators, priors and family groupings, i.e. the hyperparameter-type choices R1
tuned; evidence variants excluded).

Per split: in-sample criterion of the selected configuration (tuning labels), out-of-sample
criterion (held-out labels), optimism = in - out, excess optimism = optimism minus the same
gap of the fixed primary configuration on the same split (removes half-to-half difficulty
differences; for random halves both have the same expectation), the retained-gain ratio
mean(out - out_primary) / mean(in - in_primary) (share of the apparent in-sample advantage over the
primary that survives on held-out labels; 1 - ratio is the share that is optimism), and the Spearman
correlation of the configuration criteria between the tuning and the held-out labels.

Label-free comparators (part 2), evaluated on the same held-out labels
-----------------------------------------------------------------------
* ``label_free_cvll``: the configuration with the best mean 5-fold TF-grouped held-out
  marginal log-likelihood per observed edge (``data/r2/tables/fit_cv_loglik.csv``, global; the
  four prior variants, missing from that table, are cross-validated here with the same folds,
  restarts and seed: ``run_prior_cv``).  Evidence variants change the data, so their
  likelihoods are not comparable and they are not candidates.
* ``primary_family_hb``: the prespecified primary.
* ``oracle_heldout`` (best configuration on the held-out labels; an upper bound) and
  ``config_average`` (mean over all configurations; a random pick) for scale.

R1 joint-objective weight sensitivity (part 3)
----------------------------------------------
Candidates are the estimators with robustness results (``rob_method_ranking.csv``,
``tissue = all_tissues``): ``lcm5`` (the five latent-class models) and ``all10`` (plus the five
combination rules).  Components, min-max normalised over the candidates so that 1 = best:
Rob_source = mean D under single-annotator dropout (``protocol = source``), Rob_block = mean D
under modality-family dropout (``protocol = family``), Pert = pooled AP lift on pert_pooled.
J(w) = w_s Rob_source + w_b Rob_block + w_p Pert on the simplex grid with step 0.1 (66 points);
the selected configuration is argmax J.  ``full``: Pert on all labels (R1 practice);
``crossfit``: Pert on the tuning TFs of each random half, the selection evaluated on the
held-out TFs.  Robustness metric: F1 (R1) and posterior-weighted precision.

Outputs
-------
``outputs/r2/tuning_optimism/``: ``cv/global/fold*__<prior id>.json`` (prior-variant CV folds),
``labels_global.npz`` (evaluable edges and per-study labels), ``tf_splits.csv``,
``split_detail.csv`` (every split x criterion x configuration), ``config_scores_full.csv``,
``weight_crossfit_detail.csv``, ``run_meta.json``, ``tuning_optimism.log``.
Manuscript tables (``data/r2/tables``):

* ``tune_optimism.csv``            in-sample / out-of-sample criterion, optimism and
                                   comparators per split type x criterion x selector;
* ``tune_selection_frequency.csv`` per configuration: full-label criterion, held-out
                                   log-likelihood, selection and held-out-best frequencies;
* ``tune_weight_sensitivity.csv``  R1 joint objective: selected configuration per weight point;
* ``tune_weight_summary.csv``      how often the selection changes across the weight grid and
                                   across label splits;
* ``tune_label_sets.csv``          perturbation label sets (studies, TFs, edges, positives);
* ``tune_labelfree_cv.csv``        held-out log-likelihood of the label-free candidates;
* ``tune_status_decomposition.csv`` perturbation-positive rate and mean credibility of pert_pooled
                                   edges by evidence status (reported / observed only / unobserved)
                                   per configuration (diagnostic).

Command line (from the biodyn-work root, with the thread limits of the project rules)::

    .venv/bin/python subproject_24_multireference_latent_truth_model/src/r2/tuning_optimism.py cv        # prior-variant CV
    .venv/bin/python subproject_24_multireference_latent_truth_model/src/r2/tuning_optimism.py analyze   # parts 1-3 + tables
    .venv/bin/python subproject_24_multireference_latent_truth_model/src/r2/tuning_optimism.py all       # cv + analyze

``cv`` and missing-fit fitting reuse existing results (``--force`` recomputes).
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=2")

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from dataclasses import dataclass  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from r2 import common as C  # noqa: E402
from r2 import evidence as E  # noqa: E402
from r2 import lcm  # noqa: E402
from r2 import fit_sensitivity as FS  # noqa: E402

try:  # shared evaluation helpers of the R2 analysis stage (optional; see run_meta.json)
    from r2 import analysis_common as AC  # noqa: E402,F401
except ImportError:  # pragma: no cover - module not present in this checkout
    AC = None

# ----------------------------------------------------------------------------
# Fixed settings
# ----------------------------------------------------------------------------
PS = C.PRESPECIFIED
UNIVERSE = "global"
PRIMARY = PS["evidence_variant_primary"]
SEED = int(PS["random_seed"])
STUDY_COL = PS["cohort_cluster_unit"]          # 'study_id'
FITS = C.OUT / "fits"
OUT_DIR = C.OUT / "tuning_optimism"
CV_DIR = OUT_DIR / "cv" / UNIVERSE
FS_CV_DIR = C.OUT / "fit_sensitivity" / "cv" / UNIVERSE
TABLES = C.TABLES
PREFIX = "tune_"

N_SPLITS = 50                    # random TF halves
N_NULL_TUNE = 100                # within-TF label permutations for the null-normalised criteria (post hoc)
LOW_SUPPORT_FLAG = "essentially no positives"     # cohort_table flag of decision-log item 7 (Dixit)
EXCLUDED_CONFIGS = {"family_hb@no_motif_coexp": "true-edge and burst classes exchangeable (credibility not identified)"}
POST_HOC_CRITERIA = ("tf_geomean_ap_lift", "pooled_ap_lift_nullnorm", "tf_geomean_ap_lift_nullnorm")
POST_HOC_SPLITS = ("random_tf_half_stratified",)
MERGE_EQUIVALENT = {"family_regularized": "em|family_regularized", "em": "em|family_regularized"}
ANALYSIS_DECISIONS = [
    dict(id="tune_geomean", date="2026-10-08", status="post_hoc",
         item="per-TF criterion as the geometric mean of per-TF AP lifts (was arithmetic)",
         note="switched after SRF reached lift 342 (verifier tune minor 9)"),
    dict(id="tune_nullnorm", date="2026-10-08", status="post_hoc",
         item="null-normalised criteria (within-TF label permutation, %d common permutations)" % 100,
         note="verifier tune M3/M4"),
    dict(id="tune_loso_subsets", date="2026-10-08", status="post_hoc",
         item="LOSO without low-support studies (decision-log item 7 flag), weighted by positives, per study",
         note="verifier tune M1"),
    dict(id="tune_stratified_halves", date="2026-10-08", status="post_hoc",
         item="random TF halves stratified by dominant study", note="verifier tune minor 7"),
    dict(id="tune_exclude_no_motif_coexp", date="2026-10-08", status="post_hoc",
         item="family_hb@no_motif_coexp excluded from the tuning grid", note="verifier tune minor 9"),
    dict(id="tune_merge_equivalent", date="2026-10-08", status="post_hoc",
         item="part 3 selection summaries with em and family_regularized merged", note="verifier tune minor 5"),
    dict(id="tune_status_tf", date="2026-10-08", status="post_hoc",
         item="TF-stratified evidence-status decomposition (TF-matched O/E, MH OR vs unobserved)",
         note="verifier tune M3"),
    dict(id="tune_loso_boot_ci", date="2026-10-13", status="post_hoc",
         item="LOSO: TF-cluster bootstrap CI and bootstrap-SE z-test of the pooled AP lift of each selected "
              "configuration minus the primary on the held-out study (heldout_pooled_diff_vs_primary*; selection "
              "held fixed)",
         note="verifier tune M2 (paired tuned - primary differences need a TF-cluster interval)"),
    dict(id="tune_rerun_final_inputs", date="2026-10-13", status="module_design",
         item="analyze re-run after robustness.py regenerated rob_method_ranking.csv with the adaptive global search "
              "(2026-10-13; part 3 reads the subset = all rows); parts 1-2 read MAP fits and CV folds only",
         note="no NUTS fit is read by this module; prior-variant CV folds reused (evidence byte-identical)"),
]
# columns of prespecified rows / whole tables added after held-out results existed
POST_HOC_COLUMNS = {
    "optimism": "retained_gain_ratio (name kept; definition unchanged);heldout_pooled_diff_vs_primary*",
    "selection_frequency": "full_*_nullnorm;full_pooled_diff_vs_primary*;full_tf_nullnorm_ratio_vs_primary*;"
                           "full_*_null_mean",
    "status_decomposition": "",
    "label_sets": "",
    "labelfree_cv": "",
    "weight_sensitivity": "",
    "weight_summary": "*_merged",
}
N_BOOT_LOSO = int(PS["n_bootstrap"])
N_RESTARTS_MISSING = 10          # MAP restarts for a grid fit that does not exist yet
N_RESTARTS_CV = FS.N_RESTARTS    # 20, as in fit_cv_loglik.csv
TIE_DECIMALS = 12
PRIMARY_CONFIG = "family_hb"
CRITERIA = ("pooled_ap_lift", "tf_geomean_ap_lift", "pooled_ap_lift_nullnorm", "tf_geomean_ap_lift_nullnorm")
RAW_CRITERIA = ("pooled_ap_lift", "tf_geomean_ap_lift")
WEIGHT_STEP = 0.1
ROB_TISSUE = "all_tissues"
ROB_METRICS = ("f1", "precision")
ROB_WAIT_MIN = 60
BASELINES = ("union", "majority", "vote_fraction", "fisher", "stouffer")
# configuration grids compared by label-based tuning: all 20, and the 12 configurations fitted to the
# primary evidence (estimators, priors, family groupings: the hyperparameter-type choices R1 tuned)
GRIDS = {"all": lambda c: True, "primary_evidence": lambda c: c.group != "evidence_variant"}
CANDIDATE_SETS = {"lcm5": tuple(lcm.MODEL_SPECS), "all10": tuple(lcm.MODEL_SPECS) + BASELINES}

_LOG_FH = None


def log(*a):
    msg = " ".join(str(x) for x in a)
    print(msg, flush=True)
    if _LOG_FH is not None:
        _LOG_FH.write(time.strftime("%H:%M:%S ") + msg + "\n")
        _LOG_FH.flush()


# ----------------------------------------------------------------------------
# Configuration grid and fits
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class Config:
    config_id: str
    group: str                 # estimator | evidence_variant | prior | grouping
    variant: str
    fit_id: str
    model: str
    grouping: str = "modality"
    prior: tuple = ()          # sorted (key, value) overrides of lcm.DEFAULT_PRIOR

    @property
    def prior_dict(self):
        return dict(self.prior) or None

    @property
    def fit_path(self) -> Path:
        return FITS / UNIVERSE / self.variant / self.fit_id


def config_grid() -> list:
    """The configurations compared by label-based tuning (fixed order = tie-break order): 20 minus
    ``EXCLUDED_CONFIGS`` (19)."""
    cfgs = [Config(m, "estimator", PRIMARY, m, m) for m in lcm.MODEL_SPECS]
    cfgs += [Config(f"family_hb@{v}", "evidence_variant", v, "family_hb", "family_hb")
             for v in PS["evidence_variants_sensitivity"]]
    cfgs += [Config(fid, "prior", PRIMARY, fid, "family_hb", prior=tuple(sorted(pr.items())))
             for fid, pr in FS.PRIOR_SENS.items()]
    cfgs += [Config(f"family_hb__fam_{g}", "grouping", PRIMARY, f"family_hb__fam_{g}", "family_hb", grouping=g)
             for g in FS.GROUPINGS]
    return [c for c in cfgs if c.config_id not in EXCLUDED_CONFIGS]


def ensure_fit(cfg: Config, force: bool = False) -> dict:
    """Load the stored MAP fit of a configuration; fit and store it (contract) if missing."""
    d = cfg.fit_path
    if not force and (d / "cred.npy").exists() and (d / "meta.json").exists():
        return dict(cred=np.load(d / "cred.npy"), meta=json.loads((d / "meta.json").read_text()),
                    fitted_here=False)
    log(f"[fit] {cfg.config_id}: no stored fit at {d}; MAP with {N_RESTARTS_MISSING} restarts")
    data = lcm.PatternData(*E.load_patterns(UNIVERSE, cfg.variant))
    data = lcm.with_families(data, FS.families_for(cfg.grouping, data))
    f = FS.run_map(data, cfg.model, d, fit_id=cfg.fit_id, universe=UNIVERSE, variant=cfg.variant,
                   prior=cfg.prior_dict, n_restarts=N_RESTARTS_MISSING, seed=SEED, force=True,
                   extra=dict(created_by="src/r2/tuning_optimism.py"))
    return dict(cred=f["cred"], meta=f["meta"], fitted_here=True)


# ----------------------------------------------------------------------------
# Perturbation labels
# ----------------------------------------------------------------------------
def universe_axes() -> tuple:
    d = E.EV_DIR / UNIVERSE
    tfs = [x.strip() for x in (d / "tfs.txt").read_text().splitlines() if x.strip()]
    genes = [x.strip() for x in (d / "genes.txt").read_text().splitlines() if x.strip()]
    return tfs, genes


def evaluable_pairs(de: pd.DataFrame, tf_index: pd.Index, gene_index: pd.Index) -> pd.DataFrame:
    """Evaluable (tf, gene) index pairs of one DE table with their primary label.

    Kept: perturbed gene in ``tf_index``, target in ``gene_index``, perturbed gene != target and
    not ``cis_same_band`` (cis pairs leave both positives and negatives)."""
    pg = de["perturbed_gene"].astype(str).to_numpy()
    tg = de["target"].astype(str).to_numpy()
    ti = tf_index.get_indexer(pg)
    gi = gene_index.get_indexer(tg)
    keep = (ti >= 0) & (gi >= 0) & (pg != tg) & ~de["cis_same_band"].to_numpy(bool)
    return pd.DataFrame(dict(tf_idx=ti[keep].astype(np.int64), gene_idx=gi[keep].astype(np.int64),
                             positive=de["positive_primary"].to_numpy(bool)[keep]))


@dataclass
class Labels:
    """Evaluable edges of the union of all cohorts with per-study test / positive flags."""
    edge: np.ndarray        # (E,) global edge index tf_idx * n_genes + gene_idx (sorted, unique)
    tf: np.ndarray          # (E,) TF index into ``tfs``
    tested: np.ndarray      # (E, S) bool: evaluable in study s
    pos: np.ndarray         # (E, S) bool: positive in study s
    studies: list
    tfs: list

    def select(self, studies=None) -> tuple:
        """(mask, y) of the pooled label set over ``studies`` (default: all = pert_pooled)."""
        cols = list(range(len(self.studies))) if studies is None else [self.studies.index(s) for s in studies]
        return self.tested[:, cols].any(1), self.pos[:, cols].any(1)


def pool_labels(pairs: pd.DataFrame, studies: list, n_genes: int, tfs: list) -> Labels:
    """Pool cohort-level evaluable pairs (columns tf_idx, gene_idx, positive, study) per study."""
    edge = pairs["tf_idx"].to_numpy(np.int64) * n_genes + pairs["gene_idx"].to_numpy(np.int64)
    uniq, inv = np.unique(edge, return_inverse=True)
    inv = inv.reshape(-1)
    s_idx = pd.Index(studies).get_indexer(pairs["study"].astype(str))
    assert (s_idx >= 0).all()
    tested = np.zeros((len(uniq), len(studies)), bool)
    pos = np.zeros((len(uniq), len(studies)), bool)
    tested[inv, s_idx] = True
    np.logical_or.at(pos, (inv, s_idx), pairs["positive"].to_numpy(bool))
    return Labels(uniq, (uniq // n_genes).astype(np.int64), tested, pos, list(studies), list(tfs))


def build_labels() -> tuple:
    """pert_pooled and study label sets over the global universe; returns (Labels, summary df)."""
    tfs, genes = universe_axes()
    tix, gix = pd.Index(tfs), pd.Index(genes)
    ct = pd.read_csv(TABLES / "cohort_table.csv")
    parts, rows = [], []
    for _, r in ct.iterrows():
        de = pd.read_parquet(C.OUT / "cohorts" / r["cohort_id"] / "de.parquet",
                             columns=["perturbed_gene", "target", "positive_primary", "cis_same_band"])
        p = evaluable_pairs(de, tix, gix)
        p["study"] = str(r[STUDY_COL])
        p["cohort"] = r["cohort_id"]
        parts.append(p)
        rows.append(dict(label_set=f"cohort:{r['cohort_id']}", study_id=str(r[STUDY_COL]),
                         n_tfs=int(p["tf_idx"].nunique()), n_edges=int(len(p)), n_pos=int(p["positive"].sum())))
    pairs = pd.concat(parts, ignore_index=True)
    studies = sorted(pairs["study"].unique()) if len(pairs) else []
    all_studies = sorted(ct[STUDY_COL].astype(str).unique())
    lab = pool_labels(pairs, all_studies, len(genes), tfs)
    for s in all_studies:
        m, y = lab.select([s])
        rows.append(dict(label_set=f"study:{s}", study_id=s, n_tfs=int(len(np.unique(lab.tf[m]))),
                         n_edges=int(m.sum()), n_pos=int(y[m].sum())))
    m, y = lab.select()
    rows.append(dict(label_set="pert_pooled", study_id=";".join(studies), n_tfs=int(len(np.unique(lab.tf[m]))),
                     n_edges=int(m.sum()), n_pos=int(y[m].sum())))
    summ = pd.DataFrame(rows)
    summ["prevalence"] = summ["n_pos"] / summ["n_edges"].where(summ["n_edges"] > 0)
    tfs_pos = pd.Series(lab.pos.any(1)).groupby(lab.tf).any()
    summ.loc[summ["label_set"] == "pert_pooled", "n_tfs_with_pos"] = int(tfs_pos.sum())
    for s in all_studies:
        ms, ys = lab.select([s])
        summ.loc[summ["label_set"] == f"study:{s}", "n_tfs_with_pos"] = int(
            pd.Series(ys[ms]).groupby(lab.tf[ms]).any().sum()) if ms.any() else 0
    return lab, summ


# ----------------------------------------------------------------------------
# Scores and the tie-aware average precision
# ----------------------------------------------------------------------------
def score_blocks(score: np.ndarray) -> np.ndarray:
    """Tie-block index of each score, 0 = highest (scores rounded to TIE_DECIMALS; NaN last)."""
    s = np.asarray(score, dtype=np.float64)
    s = np.where(np.isnan(s), -np.inf, np.round(s, TIE_DECIMALS))
    _, inv = np.unique(-s, return_inverse=True)
    return inv.reshape(-1).astype(np.int64)


def ap_lift(block: np.ndarray, y: np.ndarray) -> dict:
    """Average precision (tie blocks = single thresholds) and AP lift of one ranking."""
    block = np.asarray(block, np.int64)
    y = np.asarray(y, np.float64)
    if len(block) == 0:
        return dict(ap=np.nan, lift=np.nan, n_pos=0.0, n=0.0)
    nb = int(block.max()) + 1
    tot = np.bincount(block, minlength=nb).astype(np.float64)
    pos = np.bincount(block, weights=y, minlength=nb)
    keep = tot > 0
    tot, pos = tot[keep], pos[keep]
    P, N = pos.sum(), tot.sum()
    if P <= 0:
        return dict(ap=np.nan, lift=np.nan, n_pos=float(P), n=float(N))
    ap = float(np.sum(pos * np.cumsum(pos) / np.cumsum(tot)) / P)
    return dict(ap=ap, lift=ap / (P / N), n_pos=float(P), n=float(N))


def ap_lift_by_group(block: np.ndarray, y: np.ndarray, group: np.ndarray, n_groups: int) -> dict:
    """``ap_lift`` within each group (arrays of length n_groups; NaN where a group has no positive)."""
    block = np.asarray(block, np.int64)
    group = np.asarray(group, np.int64)
    y = np.asarray(y, np.float64)
    out = dict(ap=np.full(n_groups, np.nan), lift=np.full(n_groups, np.nan),
               n_pos=np.zeros(n_groups), n=np.zeros(n_groups))
    if len(block) == 0:
        return out
    nb = int(block.max()) + 1
    key = group * nb + block
    uk, inv = np.unique(key, return_inverse=True)
    inv = inv.reshape(-1)
    tot = np.bincount(inv).astype(np.float64)
    pos = np.bincount(inv, weights=y)
    g = uk // nb
    first = np.r_[True, g[1:] != g[:-1]]
    seg = np.cumsum(first) - 1
    ctot, cpos = np.cumsum(tot), np.cumsum(pos)
    start = np.flatnonzero(first)
    ctot = ctot - (ctot - tot)[start][seg]
    cpos = cpos - (cpos - pos)[start][seg]
    contrib = pos * cpos / ctot
    P = np.bincount(g, weights=pos, minlength=n_groups)
    N = np.bincount(g, weights=tot, minlength=n_groups)
    S = np.bincount(g, weights=contrib, minlength=n_groups)
    with np.errstate(divide="ignore", invalid="ignore"):
        ap = np.where(P > 0, S / P, np.nan)
        lift = np.where(P > 0, ap / (P / np.where(N > 0, N, 1)), np.nan)
    out.update(ap=ap, lift=lift, n_pos=P, n=N)
    return out


def geomean(v) -> float:
    """Geometric mean of positive values (NaN when empty)."""
    v = np.asarray(v, dtype=np.float64)
    v = v[np.isfinite(v)]
    return float(2.0 ** np.mean(np.log2(v))) if len(v) else np.nan


def criterion(block: np.ndarray, lab: Labels, mask: np.ndarray, y: np.ndarray, which: str) -> float:
    """Tuning criterion of one configuration on the evaluable edges ``mask`` with labels ``y``."""
    if which == "pooled_ap_lift":
        return ap_lift(block[mask], y[mask])["lift"]
    if which == "tf_geomean_ap_lift":
        r = ap_lift_by_group(block[mask], y[mask], lab.tf[mask], len(lab.tfs))
        v = r["lift"][r["n_pos"] > 0]
        return geomean(v)
    raise ValueError(which)


# ----------------------------------------------------------------------------
# Within-TF permutation null (common random numbers across configurations; post hoc)
# ----------------------------------------------------------------------------
def within_tf_permutations(y: np.ndarray, tf: np.ndarray, n_perm: int, rng: np.random.Generator):
    """(n_perm x n) sparse 0/1 matrix: in each row the positives of every TF are placed on a
    uniformly random subset of that TF's edges (within-TF label permutation)."""
    import scipy.sparse as sp
    y = np.asarray(y, dtype=bool)
    tf = np.asarray(tf, dtype=np.int64)
    order = np.argsort(tf, kind="stable")
    ts = tf[order]
    bounds = np.r_[0, np.flatnonzero(np.diff(ts)) + 1, len(ts)]
    kpos = [int(y[order[a:b]].sum()) for a, b in zip(bounds[:-1], bounds[1:])]
    rows, cols = [], []
    for r in range(n_perm):
        for (a, b), k in zip(zip(bounds[:-1], bounds[1:]), kpos):
            if k == 0:
                continue
            pick = order[a + rng.choice(b - a, size=k, replace=False)]
            cols.append(pick)
            rows.append(np.full(k, r, dtype=np.int64))
    if not cols:
        return sp.csr_matrix((n_perm, len(y)))
    rr, cc = np.concatenate(rows), np.concatenate(cols)
    return sp.csr_matrix((np.ones(len(rr)), (rr, cc)), shape=(n_perm, len(y)))


def ap_lift_perm(block: np.ndarray, Y) -> np.ndarray:
    """Pooled AP lift of one ranking (tie blocks ``block``) for every row of the label matrix Y
    (sparse n_perm x n; same definition as ``ap_lift``)."""
    import scipy.sparse as sp
    block = np.asarray(block, np.int64)
    n = len(block)
    if n == 0:
        return np.full(Y.shape[0], np.nan)
    nb = int(block.max()) + 1
    tot = np.bincount(block, minlength=nb).astype(np.float64)
    oh = sp.csr_matrix((np.ones(n), (np.arange(n), block)), shape=(n, nb))
    pos = np.asarray((Y @ oh).todense(), dtype=np.float64)
    keep = tot > 0
    tot, pos = tot[keep], pos[:, keep]
    ct, cp = np.cumsum(tot), np.cumsum(pos, 1)
    P = cp[:, -1]
    with np.errstate(invalid="ignore", divide="ignore"):
        ap = (pos * cp / ct).sum(1) / P
        return ap / (P / n)


def ap_lift_by_group_perm(block: np.ndarray, Y, group: np.ndarray, n_groups: int, chunk: int = 25) -> np.ndarray:
    """Per-group AP lift (``ap_lift_by_group``) for every row of Y: (n_perm, n_groups), NaN where a
    group has no positive in that row."""
    import scipy.sparse as sp
    block = np.asarray(block, np.int64)
    group = np.asarray(group, np.int64)
    n = len(block)
    out = np.full((Y.shape[0], n_groups), np.nan)
    if n == 0:
        return out
    nb = int(block.max()) + 1
    key = group * nb + block
    uk, inv = np.unique(key, return_inverse=True)
    inv = inv.reshape(-1)
    nk = len(uk)
    tot = np.bincount(inv, minlength=nk).astype(np.float64)
    g = uk // nb
    first = np.r_[True, g[1:] != g[:-1]]
    start = np.flatnonzero(first)
    seg = np.cumsum(first) - 1
    ctot = np.cumsum(tot)
    ctot = ctot - (ctot - tot)[start][seg]
    Ng = np.add.reduceat(tot, start)
    oh = sp.csr_matrix((np.ones(n), (np.arange(n), inv)), shape=(n, nk))
    for a in range(0, Y.shape[0], chunk):
        pos = np.asarray((Y[a:a + chunk] @ oh).todense(), dtype=np.float64)
        cpos = np.cumsum(pos, 1)
        cpos = cpos - (cpos - pos)[:, start][:, seg]
        S = np.add.reduceat(pos * cpos / ctot, start, axis=1)
        P = np.add.reduceat(pos, start, axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            lift = np.where(P > 0, (S / P) / (P / Ng), np.nan)
        out[a:a + chunk][:, g[start]] = lift
    return out


def perm_null_per_tf(blocks: list, Y, tf: np.ndarray, n_tfs: int) -> np.ndarray:
    """E_null[log2 per-TF AP lift] per configuration (n_cfg, n_tfs) over the permutations Y."""
    out = np.full((len(blocks), n_tfs), np.nan)
    for i, b in enumerate(blocks):
        L = ap_lift_by_group_perm(b, Y, tf, n_tfs)
        with np.errstate(divide="ignore", invalid="ignore"):
            out[i] = np.nanmean(np.log2(L), 0)
    return out


def dominant_study(lab: "Labels", mask: np.ndarray) -> pd.Series:
    """Study holding most evaluable edges of each TF (ties: first in sorted study order)."""
    idx = np.flatnonzero(mask)
    cnt = pd.DataFrame(lab.tested[idx].astype(int), columns=lab.studies).groupby(lab.tf[idx]).sum()
    return cnt.idxmax(axis=1)


def stratified_tf_splits(tf_pool: np.ndarray, strata: pd.Series, n_splits: int = N_SPLITS, seed: int = SEED) -> list:
    """Seeded random halves of the TF pool stratified by ``strata`` (TF -> label): within each
    stratum half of the TFs (rounding alternates between halves over strata) go to the tuning half."""
    rng = np.random.default_rng([seed, 7])
    tf_pool = np.sort(np.asarray(tf_pool))
    lab = strata.reindex(tf_pool).fillna("none").to_numpy()
    groups = {k: tf_pool[lab == k] for k in sorted(set(lab))}
    out = []
    for _ in range(n_splits):
        tune, flip = [], 0
        for k, members in groups.items():
            perm = rng.permutation(members)
            h = len(perm) // 2 + (len(perm) % 2) * flip
            flip ^= len(perm) % 2
            tune.extend(perm[:h])
        tune = np.sort(np.asarray(tune, dtype=tf_pool.dtype))
        out.append((tune, np.setdiff1d(tf_pool, tune)))
    return out


# ----------------------------------------------------------------------------
# Splits and selection
# ----------------------------------------------------------------------------
def random_tf_splits(tf_pool: np.ndarray, n_splits: int = N_SPLITS, seed: int = SEED) -> list:
    """Seeded random halves of the TF pool: list of (tuning TFs, held-out TFs), both sorted."""
    rng = np.random.default_rng(seed)
    tf_pool = np.sort(np.asarray(tf_pool))
    h = len(tf_pool) // 2
    out = []
    for _ in range(n_splits):
        perm = rng.permutation(len(tf_pool))
        out.append((np.sort(tf_pool[perm[:h]]), np.sort(tf_pool[perm[h:]])))
    return out


def select_best(values) -> tuple:
    """Index of the maximum (ties within 1e-12: first in grid order) and the number tied."""
    v = np.asarray(values, dtype=np.float64)
    if not np.isfinite(v).any():
        return -1, 0
    best = np.nanmax(v)
    tied = np.flatnonzero(np.isfinite(v) & (v >= best - 1e-12))
    return int(tied[0]), int(len(tied))


def rank_desc(v) -> np.ndarray:
    """Rank 1 = best (average ranks for ties)."""
    return pd.Series(np.asarray(v, float)).rank(ascending=False, method="average").to_numpy()


def split_records(in_vals: np.ndarray, out_vals: np.ndarray, cfg_ids: list, fixed: dict) -> dict:
    """Selector results for one split.  ``fixed`` maps selector -> configuration index."""
    sel, n_tied = select_best(in_vals)
    orc, _ = select_best(out_vals)
    ok = np.isfinite(in_vals) & np.isfinite(out_vals)
    rho = spearmanr(in_vals[ok], out_vals[ok]).correlation if ok.sum() > 2 else np.nan
    rec = {"label_tuned": dict(config=cfg_ids[sel], in_value=in_vals[sel], out_value=out_vals[sel], n_tied=n_tied),
           "oracle_heldout": dict(config=cfg_ids[orc], in_value=in_vals[orc], out_value=out_vals[orc], n_tied=1),
           "config_average": dict(config="mean_of_grid", in_value=float(np.nanmean(in_vals)),
                                  out_value=float(np.nanmean(out_vals)), n_tied=0)}
    for name, i in fixed.items():
        rec[name] = dict(config=cfg_ids[i], in_value=in_vals[i], out_value=out_vals[i], n_tied=0)
    for r in rec.values():
        r["gap"] = r["in_value"] - r["out_value"]
    prim = rec["primary_family_hb"]
    for r in rec.values():
        r["excess_gap"] = r["gap"] - prim["gap"]
        r["in_minus_primary"] = r["in_value"] - prim["in_value"]
        r["out_minus_primary"] = r["out_value"] - prim["out_value"]
        r["spearman_halves"] = rho
    return rec


# ----------------------------------------------------------------------------
# Label-free selection: held-out marginal log-likelihood (TF-grouped 5-fold CV)
# ----------------------------------------------------------------------------
def _cv_record_path(fold: int, cid: str) -> Path:
    return CV_DIR / f"fold{fold}__{cid}.json"


def run_prior_cv(force: bool = False, folds=None) -> None:
    """5-fold TF-grouped held-out log-likelihood of the four prior variants (the configurations of
    ``FS.PRIOR_SENS``), with the folds, restarts and seed of ``fit_sensitivity.run_cv``."""
    data = FS.load_data(UNIVERSE)
    ep = FS.edge_pattern(UNIVERSE)
    tf_idx, tfs = FS.edge_tf_index(UNIVERSE)
    fold_of_tf = FS.tf_folds(len(tfs))
    ref = FS_CV_DIR / "tf_folds.csv"
    if ref.exists():
        rf = pd.read_csv(ref)
        assert rf["tf"].tolist() == tfs and (rf["fold"].to_numpy() == fold_of_tf).all(), "fold mismatch"
    C.ensure_dir(CV_DIR)
    pd.DataFrame(dict(tf=tfs, fold=fold_of_tf)).to_csv(CV_DIR / "tf_folds.csv", index=False)
    counts = FS.fold_pattern_counts(ep, fold_of_tf[tf_idx], data.P)
    for f in (range(FS.N_FOLDS) if folds is None else folds):
        n_test = counts[f]
        n_train = counts.sum(0) - n_test
        tr, te = n_train > 0, n_test > 0
        train = lcm.PatternData(data.R[tr], n_train[tr], list(data.family), list(data.annotators))
        test = lcm.PatternData(data.R[te], n_test[te], list(data.family), list(data.annotators))
        n_test_obs = FS._n_observed(test)
        for cid, prior in FS.PRIOR_SENS.items():
            out = _cv_record_path(f, cid)
            if out.exists() and not force:
                continue
            t = time.time()
            fit = lcm.fit_map(train, "family_hb", n_restarts=N_RESTARTS_CV, seed=SEED, prior=prior)
            ll_test = lcm.heldout_loglik(fit.best, test)
            cchk = FS.conditional_annotator_check(test, fit.best["w"], fit.best["theta"])
            s = fit.summary()
            rec = dict(universe=UNIVERSE, fold=f, config=cid, model="family_hb", grouping="modality",
                       prior=prior, n_train_edges=float(train.n.sum()), n_train_observed=FS._n_observed(train),
                       n_test_edges=float(test.n.sum()), n_test_observed=n_test_obs,
                       n_test_tfs=int((fold_of_tf == f).sum()), train_loglik=float(fit.loglik),
                       train_log_post=float(fit.log_post), test_loglik=float(ll_test),
                       test_loglik_per_obs_edge=float(ll_test / n_test_obs),
                       test_cond_loglik=float(cchk["cll"].sum()), test_n_obs_cells=float(cchk["n_obs"].sum()),
                       test_cond_loglik_per_cell=float(cchk["cll"].sum() / cchk["n_obs"].sum()),
                       n_restarts=N_RESTARTS_CV, n_within_1e3=s["n_within_1e3_of_best"],
                       n_converged=s["n_converged"], w=np.asarray(fit.best["w"]).tolist(),
                       class_names=fit.class_names, runtime_s=time.time() - t,
                       code="src/r2/tuning_optimism.py::run_prior_cv")
            C.write_json(rec, out)
            log(f"  cv fold{f} {cid}: test ll/edge={rec['test_loglik_per_obs_edge']:.7f} "
                f"within1e-3={rec['n_within_1e3']}/{N_RESTARTS_CV} ({rec['runtime_s']:.0f}s)")


def labelfree_cv_table() -> pd.DataFrame:
    """Held-out log-likelihood summary over the label-free candidates (fit_sensitivity folds of the
    estimators / groupings plus this module's prior-variant folds), computed as in
    ``fit_sensitivity.cv_tables``; differences are paired by fold against family_hb."""
    recs = []
    for d, src in ((FS_CV_DIR, "fit_sensitivity"), (CV_DIR, "tuning_optimism")):
        for p in sorted(d.glob("fold*__*.json")):
            r = json.loads(p.read_text())
            recs.append(dict(fold=r["fold"], config=r["config"], test_loglik=r["test_loglik"],
                             n_test_observed=r["n_test_observed"],
                             test_loglik_per_obs_edge=r["test_loglik_per_obs_edge"],
                             test_cond_loglik=r.get("test_cond_loglik"), test_n_obs_cells=r.get("test_n_obs_cells"),
                             test_cond_loglik_per_cell=r.get("test_cond_loglik_per_cell"),
                             n_within_1e3=r.get("n_within_1e3"), n_restarts=r.get("n_restarts"), source=src))
    folds = pd.DataFrame(recs)
    if folds.empty:
        return folds
    ref = folds[folds["config"] == PRIMARY_CONFIG].set_index("fold")
    rows = []
    for cid, g in folds.groupby("config"):
        g = g.set_index("fold").sort_index()
        k = len(g)
        x = g["test_loglik_per_obs_edge"]
        common = g.index.intersection(ref.index)
        dpe = (g.loc[common, "test_loglik"] - ref.loc[common, "test_loglik"]) / g.loc[common, "n_test_observed"]
        dtot = g.loc[common, "test_loglik"] - ref.loc[common, "test_loglik"]
        xc = g["test_cond_loglik_per_cell"].astype(float)
        rows.append(dict(universe=UNIVERSE, config=cid, source=g["source"].iloc[0], n_folds=k,
                         complete=bool(k == FS.N_FOLDS),
                         heldout_ll_per_edge_mean=x.mean(), heldout_ll_per_edge_se=x.std(ddof=1) / np.sqrt(k),
                         diff_vs_family_hb_per_edge_mean=dpe.mean(),
                         diff_vs_family_hb_per_edge_se=dpe.std(ddof=1) / np.sqrt(len(dpe)) if len(dpe) > 1 else np.nan,
                         diff_vs_family_hb_total_nats=float(dtot.sum()),
                         n_folds_better_than_family_hb=int((dtot > 0).sum()),
                         heldout_cond_ll_per_cell_mean=xc.mean(),
                         n_restarts=int(g["n_restarts"].min()) if g["n_restarts"].notna().all() else np.nan,
                         min_restarts_within_1e3=int(g["n_within_1e3"].min())))
    out = pd.DataFrame(rows).sort_values("heldout_ll_per_edge_mean", ascending=False).reset_index(drop=True)
    out["rank"] = np.arange(1, len(out) + 1)
    return out


def check_cv_against_table(cv: pd.DataFrame) -> float:
    """Max |difference| between the recomputed held-out log-likelihood means and fit_cv_loglik.csv."""
    p = TABLES / "fit_cv_loglik.csv"
    if not p.exists():
        return np.nan
    t = pd.read_csv(p)
    t = t[t["universe"] == UNIVERSE].set_index("config")["heldout_ll_per_edge_mean"]
    m = cv.set_index("config")["heldout_ll_per_edge_mean"]
    common = t.index.intersection(m.index)
    return float(np.max(np.abs(t.loc[common] - m.loc[common]))) if len(common) else np.nan


def label_free_choice(cv: pd.DataFrame, cfg_ids: list) -> tuple:
    """Configuration with the best mean held-out log-likelihood among complete grid candidates."""
    cand = cv[cv["complete"] & cv["config"].isin(cfg_ids)]
    if cand.empty:
        return PRIMARY_CONFIG, []
    best = cand.sort_values("heldout_ll_per_edge_mean", ascending=False)["config"].iloc[0]
    return best, sorted(cand["config"])


# ----------------------------------------------------------------------------
# Part 1 + 2: label-based tuning vs label-free choices
# ----------------------------------------------------------------------------
def config_blocks(cfgs: list, lab: Labels) -> tuple:
    """Tie-block index per evaluable edge for every configuration, plus fit metadata rows."""
    eps, n_pat, blocks, meta_rows = {}, {}, [], []
    for cfg in cfgs:
        f = ensure_fit(cfg)
        if cfg.variant not in eps:
            eps[cfg.variant] = np.asarray(E.load_edge_pattern(UNIVERSE, cfg.variant))[lab.edge].astype(np.int64)
            n_pat[cfg.variant] = len(E.load_patterns(UNIVERSE, cfg.variant)[1])
        ep = eps[cfg.variant]
        assert len(f["cred"]) == n_pat[cfg.variant], (cfg.config_id, len(f["cred"]), n_pat[cfg.variant])
        assert (ep >= 0).all()
        m = f["meta"]
        assert list(m["class_names"])[0] == "T", (cfg.config_id, m["class_names"])
        blocks.append(score_blocks(f["cred"][ep]))
        conv = m.get("convergence", {})
        with np.load(cfg.fit_path / "params.npz") as z:
            w_t = float(np.asarray(z["w"])[0])
        meta_rows.append(dict(config_id=cfg.config_id, group=cfg.group, variant=cfg.variant, fit_id=cfg.fit_id,
                              model=cfg.model, grouping=cfg.grouping,
                              prior=json.dumps(cfg.prior_dict) if cfg.prior_dict else "",
                              n_restarts=m.get("n_restarts"), n_within_1e3_of_best=conv.get("n_within_1e3_of_best"),
                              fitted_here=f["fitted_here"],
                              t_burst_exchangeable=FS.exchangeable_t_burst(m), w_T=w_t,
                              best_log_post=m.get("best_log_post")))
    return blocks, pd.DataFrame(meta_rows)


def status_decomposition(cfgs: list, lab: Labels) -> pd.DataFrame:
    """Perturbation-positive rate and mean credibility of pert_pooled edges by evidence status under
    each configuration's observation model: reported by >= 1 annotator, observed but not reported, not
    observed by any annotator (diagnostic: how a configuration orders the three groups explains most
    of its pooled AP lift).  TF-stratified columns (post hoc, verifier M3): TF-matched expected
    positives (sum of each edge's TF positive rate) and O/E per status, and the within-TF Mantel-
    Haenszel odds ratio of the status against unobserved edges (TFs with both statuses)."""
    m, y = lab.select()
    tf = lab.tf[m]
    yy = y[m]
    T = len(lab.tfs)
    N_t = np.bincount(tf, minlength=T).astype(float)
    P_t = np.bincount(tf, weights=yy.astype(float), minlength=T)
    base = np.where(N_t > 0, P_t / np.maximum(N_t, 1), 0.0)[tf]
    rows, cache = [], {}
    for cfg in cfgs:
        if cfg.variant not in cache:
            R = E.load_patterns(UNIVERSE, cfg.variant)[0]
            ep = np.asarray(E.load_edge_pattern(UNIVERSE, cfg.variant))[lab.edge[m]].astype(np.int64)
            st = np.where((R == 1).any(1), "reported", np.where((R >= 0).any(1), "observed_not_reported",
                                                                "unobserved"))
            cache[cfg.variant] = (ep, st[ep])
        ep, status = cache[cfg.variant]
        cred = np.load(cfg.fit_path / "cred.npy")[ep]
        un = status == "unobserved"
        for name in ("reported", "observed_not_reported", "unobserved"):
            k = status == name
            exp_ = float(base[k].sum())
            row = dict(config_id=cfg.config_id, variant=cfg.variant, status=name, n_edges=int(k.sum()),
                       frac_edges=float(k.mean()), n_pos=int(yy[k].sum()),
                       pos_rate=float(yy[k].mean()) if k.any() else np.nan,
                       pos_rate_all=float(yy.mean()),
                       cred_mean=float(cred[k].mean()) if k.any() else np.nan,
                       cred_median=float(np.median(cred[k])) if k.any() else np.nan,
                       exp_tf_matched=exp_, oe_tf_matched=float(yy[k].sum() / exp_) if exp_ > 0 else np.nan)
            if name != "unobserved":
                a_ = np.bincount(tf[k], weights=yy[k].astype(float), minlength=T)
                b_ = np.bincount(tf[k], weights=(~yy[k]).astype(float), minlength=T)
                c_ = np.bincount(tf[un], weights=yy[un].astype(float), minlength=T)
                d_ = np.bincount(tf[un], weights=(~yy[un]).astype(float), minlength=T)
                n_ = a_ + b_ + c_ + d_
                both = ((a_ + b_) > 0) & ((c_ + d_) > 0) & ((a_ + c_) > 0) & (n_ > 0)
                num = float(np.sum((a_ * d_ / np.maximum(n_, 1))[both]))
                den = float(np.sum((b_ * c_ / np.maximum(n_, 1))[both]))
                row.update(mh_or_vs_unobserved=num / den if den > 0 else np.nan, n_tfs_mh=int(both.sum()))
            rows.append(row)
    out = pd.DataFrame(rows)
    out["analysis_status"] = "post_hoc"
    return out


def run_tuning(blocks: list, lab: Labels, cfg_ids: list, fixed: dict, extra_splits: bool = True,
               grids: dict | None = None) -> tuple:
    """Random TF halves, study-stratified halves (post hoc) and LOSO: per split x criterion x
    configuration values and selector records, for the raw and the null-normalised criteria.

    ``grids``: {grid name: (configuration indices, fixed selectors)}; the criterion values are
    computed once for all configurations and every grid selects among its own subset (default:
    one grid of all configurations with ``fixed``).  Rows carry ``grid`` / ``n_configs``."""
    grids = grids or {"all": (list(range(len(cfg_ids))), fixed)}
    detail, sel_rows, split_rows = [], [], []

    def _collect_grids(vals, info):
        for gname, (gidx, gfixed) in grids.items():
            gi = dict(grid=gname, n_configs=len(gidx), **info)
            _collect({c: (v[0][gidx], v[1][gidx]) for c, v in vals.items()}, gi, [cfg_ids[i] for i in gidx],
                     gfixed, detail, sel_rows)

    m_all, y_all = lab.select()
    tf_pool = np.unique(lab.tf[m_all])
    idx_all = np.flatnonzero(m_all)
    tf_m, y_m = lab.tf[idx_all], y_all[idx_all]
    blocks_m = [b[idx_all] for b in blocks]
    # per-TF log2 AP lift on pert_pooled (fixed across random halves) and its within-TF null
    per_tf = np.stack([np.log2(ap_lift_by_group(b, y_m, tf_m, len(lab.tfs))["lift"]) for b in blocks_m])
    Yfull = within_tf_permutations(y_m, tf_m, N_NULL_TUNE, np.random.default_rng([SEED, 11]))
    per_tf_null = perm_null_per_tf(blocks_m, Yfull, tf_m, len(lab.tfs))
    per_tf_nn = per_tf - per_tf_null
    tf_has_pos = pd.Series(y_m).groupby(tf_m).any().reindex(range(len(lab.tfs)), fill_value=False).to_numpy()

    def half_values(tt):
        """criteria of every configuration on the pert_pooled edges of the TFs tt."""
        pos = np.flatnonzero(np.isin(tf_m, tt))
        Ys = Yfull[:, pos]
        it = np.isin(np.arange(len(lab.tfs)), tt) & tf_has_pos
        raw = np.array([ap_lift(b[pos], y_m[pos])["lift"] for b in blocks_m])
        null = np.array([np.nanmean(ap_lift_perm(b[pos], Ys)) for b in blocks_m])
        return {"pooled_ap_lift": raw, "tf_geomean_ap_lift": 2.0 ** np.nanmean(per_tf[:, it], 1),
                "pooled_ap_lift_nullnorm": raw / null,
                "tf_geomean_ap_lift_nullnorm": 2.0 ** np.nanmean(per_tf_nn[:, it], 1)}, len(pos), int(y_m[pos].sum())

    split_sets = [("random_tf_half", random_tf_splits(tf_pool))]
    if extra_splits:
        split_sets.append(("random_tf_half_stratified",
                           stratified_tf_splits(tf_pool, dominant_study(lab, m_all))))
    for split_type, splits in split_sets:
        for s, (tune, hold) in enumerate(splits):
            for half, tt in (("tuning", tune), ("heldout", hold)):
                split_rows += [dict(split_type=split_type, split_id=s, half=half, tf=lab.tfs[t]) for t in tt]
            vt, nt, pt = half_values(tune)
            vh, nh, ph = half_values(hold)
            info = dict(split_type=split_type, split_id=s, heldout_study="",
                        n_tfs_tuning=len(tune), n_tfs_heldout=len(hold),
                        n_pos_tuning=pt, n_pos_heldout=ph, n_edges_tuning=nt, n_edges_heldout=nh)
            _collect_grids({c: (vt[c], vh[c]) for c in CRITERIA}, info)
    # leave one study out (labels differ between sides, so each side gets its own permutation null)
    low = low_support_studies()
    for s_ in lab.studies:
        mh, yh = lab.select([s_])
        if not yh[mh].any():
            continue
        others = [x for x in lab.studies if x != s_]
        mt, yt = lab.select(others)
        info = dict(split_type="loso", split_id=lab.studies.index(s_), heldout_study=s_,
                    n_tfs_tuning=int(len(np.unique(lab.tf[mt]))), n_tfs_heldout=int(len(np.unique(lab.tf[mh]))),
                    n_pos_tuning=int(yt[mt].sum()), n_pos_heldout=int(yh[mh].sum()),
                    n_edges_tuning=int(mt.sum()), n_edges_heldout=int(mh.sum()),
                    heldout_low_support=bool(s_ in low))
        vals = {}
        for side, (m_, y_) in (("in", (mt, yt)), ("out", (mh, yh))):
            ii = np.flatnonzero(m_)
            tf_s, y_s = lab.tf[ii], y_[ii]
            Ys = within_tf_permutations(y_s, tf_s, N_NULL_TUNE, np.random.default_rng([SEED, 13, lab.studies.index(s_),
                                                                                        int(side == "out")]))
            bs = [b[ii] for b in blocks]
            raw_p = np.array([ap_lift(b, y_s)["lift"] for b in bs])
            null_p = np.array([np.nanmean(ap_lift_perm(b, Ys)) for b in bs])
            pt_ = np.stack([np.log2(ap_lift_by_group(b, y_s, tf_s, len(lab.tfs))["lift"]) for b in bs])
            pn_ = perm_null_per_tf(bs, Ys, tf_s, len(lab.tfs))
            has = pd.Series(y_s).groupby(tf_s).any().reindex(range(len(lab.tfs)), fill_value=False).to_numpy()
            vals[side] = {"pooled_ap_lift": raw_p, "tf_geomean_ap_lift": 2.0 ** np.nanmean(pt_[:, has], 1),
                          "pooled_ap_lift_nullnorm": raw_p / null_p,
                          "tf_geomean_ap_lift_nullnorm": 2.0 ** np.nanmean((pt_ - pn_)[:, has], 1)}
        n_before = len(sel_rows)
        _collect_grids({c: (vals["in"][c], vals["out"][c]) for c in CRITERIA}, info)
        # held-out pooled AP lift of every selected configuration minus the primary: TF-cluster bootstrap
        ih = np.flatnonzero(mh)
        add_heldout_contrast(sel_rows[n_before:], [b[ih] for b in blocks], yh[ih], lab.tf[ih], cfg_ids,
                             np.random.default_rng([SEED, 23, lab.studies.index(s_)]))
    null_info = dict(per_tf=per_tf, per_tf_null=per_tf_null, tf_has_pos=tf_has_pos, Yfull=Yfull, idx_all=idx_all)
    return pd.DataFrame(detail), pd.DataFrame(sel_rows), pd.DataFrame(split_rows), null_info


def add_heldout_contrast(rows: list, blocks_h: list, y_h: np.ndarray, tf_h: np.ndarray, cfg_ids: list,
                         rng: np.random.Generator, n_boot: int | None = None) -> None:
    """Add to the selector records ``rows`` of one held-out label set the pooled AP lift of the record's
    configuration minus that of the primary on the held-out edges, with a paired TF-cluster percentile
    bootstrap CI and a bootstrap-SE z-test (``heldout_pooled_diff_vs_primary[_lo|_hi|_p_z]``,
    ``heldout_n_tfs_boot``).  ``blocks_h``: tie blocks of every configuration on the held-out edges
    (order of ``cfg_ids``).  The selection is held fixed.  Records without a single configuration
    (``config_average``) get NaN."""
    from r2 import heldout_eval as H
    n_boot = N_BOOT_LOSO if n_boot is None else n_boot
    tcode, tuniq = pd.factorize(np.asarray(tf_h), sort=True)
    M = H.tf_bootstrap_weights(len(tuniq), n_boot, rng)
    cache: dict = {}

    def reps(cid):
        if cid not in cache:
            b = np.asarray(blocks_h[cfg_ids.index(cid)])
            lc = H.level_counts(tcode, y_h, np.zeros(len(b)), -b.astype(np.float64), len(tuniq))
            cache[cid] = (float(H.curve_metrics(*lc.totals(), ks=())["ap_lift"][0]), H.boot_light(lc, M, 500)["ap_lift"])
        return cache[cid]

    p0, r0 = reps(PRIMARY_CONFIG)
    for r in rows:
        r.update(heldout_pooled_diff_vs_primary=np.nan, heldout_pooled_diff_vs_primary_lo=np.nan,
                 heldout_pooled_diff_vs_primary_hi=np.nan, heldout_pooled_diff_vs_primary_p_z=np.nan,
                 heldout_n_tfs_boot=int(len(tuniq)))
        if r["config"] not in cfg_ids:
            continue
        pc, rc = reps(r["config"])
        d = rc - r0
        lo, hi = H.percentile_ci(d)
        r.update(heldout_pooled_diff_vs_primary=pc - p0, heldout_pooled_diff_vs_primary_lo=lo,
                 heldout_pooled_diff_vs_primary_hi=hi,
                 heldout_pooled_diff_vs_primary_p_z=(H.bootstrap_p_z(d, pc - p0) if r["config"] != PRIMARY_CONFIG
                                                     else np.nan))


def low_support_studies() -> set:
    """Studies whose cohorts carry the decision-log item 7 flag ("essentially no positives")."""
    ct = pd.read_csv(TABLES / "cohort_table.csv")
    f = ct["flag"].astype(str).str.contains(LOW_SUPPORT_FLAG, regex=False) if "flag" in ct else np.zeros(len(ct), bool)
    return set(ct.loc[f, STUDY_COL].astype(str))


def _collect(vals: dict, info: dict, cfg_ids: list, fixed: dict, detail: list, sel_rows: list) -> None:
    for crit, (vin, vout) in vals.items():
        rin, rout = rank_desc(vin), rank_desc(vout)
        sel, _ = select_best(vin)
        for i, cid in enumerate(cfg_ids):
            detail.append(dict(**info, criterion=crit, config_id=cid, in_value=vin[i], out_value=vout[i],
                               rank_tuning=rin[i], rank_heldout=rout[i], selected=bool(i == sel)))
        for name, r in split_records(vin, vout, cfg_ids, fixed).items():
            sel_rows.append(dict(**info, criterion=crit, selector=name, **r))


def _q(x, q):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return float(np.quantile(x, q)) if len(x) else np.nan


def _opt_row(g: pd.DataFrame, prim: pd.Series, lf: pd.Series, w: np.ndarray | None = None) -> dict:
    """Aggregates of one selector over a set of splits (``w``: split weights, None = equal)."""
    gi = g.set_index("split_id")
    w = np.ones(len(g)) if w is None else np.asarray(w, dtype=float)
    wm = lambda x: float(np.average(np.asarray(x, float), weights=w)) if np.isfinite(np.asarray(x, float)).all() \
        else float(np.nanmean(x))  # noqa: E731
    vc = g["config"].value_counts()
    in_m, out_m = wm(g["in_minus_primary"]), wm(g["out_minus_primary"])
    return dict(
        n_splits=len(g),
        selected_config_modal=vc.index[0], selected_config_modal_freq=float(vc.iloc[0] / len(g)),
        n_distinct_selected=int(g["config"].nunique()),
        in_mean=wm(g["in_value"]), in_sd=g["in_value"].std(ddof=1),
        out_mean=wm(g["out_value"]), out_sd=g["out_value"].std(ddof=1),
        out_q025=_q(g["out_value"], 0.025), out_q975=_q(g["out_value"], 0.975),
        gap_mean=wm(g["gap"]), gap_sd=g["gap"].std(ddof=1),
        gap_q025=_q(g["gap"], 0.025), gap_q975=_q(g["gap"], 0.975),
        gap_rel_mean=float(np.nanmean(g["gap"] / g["out_value"])),
        excess_gap_mean=wm(g["excess_gap"]),
        excess_gap_q025=_q(g["excess_gap"], 0.025), excess_gap_q975=_q(g["excess_gap"], 0.975),
        in_minus_primary_mean=in_m, out_minus_primary_mean=out_m,
        retained_gain_ratio=(out_m / in_m if abs(in_m) > 1e-12 else np.nan),
        out_minus_primary_q025=_q(g["out_minus_primary"], 0.025),
        out_minus_primary_q975=_q(g["out_minus_primary"], 0.975),
        frac_out_gt_primary=float(np.mean(gi["out_value"] > prim.reindex(gi.index) + 1e-12)),
        frac_out_gt_label_free=float(np.mean(gi["out_value"] > lf.reindex(gi.index) + 1e-12)),
        frac_in_gt_out=float(np.mean(g["in_value"] > g["out_value"])),
        spearman_halves_mean=g["spearman_halves"].mean(),
        spearman_halves_q025=_q(g["spearman_halves"], 0.025),
        spearman_halves_q975=_q(g["spearman_halves"], 0.975),
        **_single_fold_contrast(g))


HELDOUT_CONTRAST_COLS = ("heldout_pooled_diff_vs_primary", "heldout_pooled_diff_vs_primary_lo",
                         "heldout_pooled_diff_vs_primary_hi", "heldout_pooled_diff_vs_primary_p_z", "heldout_n_tfs_boot")


def _single_fold_contrast(g: pd.DataFrame) -> dict:
    """Held-out pooled contrast with its TF-bootstrap CI for a single-fold row (LOSO ``study:<id>``);
    NaN for aggregates over several folds or splits."""
    if len(g) == 1 and HELDOUT_CONTRAST_COLS[0] in g.columns:
        return {c: g[c].iloc[0] for c in HELDOUT_CONTRAST_COLS}
    return {c: np.nan for c in HELDOUT_CONTRAST_COLS}


def optimism_table(sel: pd.DataFrame, labelfree_candidates: list) -> pd.DataFrame:
    """Selector aggregates per grid x split type x criterion (``fold_set = all_folds``); LOSO also per
    fold set (without low-support held-out studies, weighted by held-out positives, single study)."""
    order = ["label_tuned", "label_free_cvll", "primary_family_hb", "oracle_heldout", "config_average"]
    rows = []
    for (grid, st, crit, selector), g in sel.groupby(["grid", "split_type", "criterion", "selector"]):
        same = (sel["grid"] == grid) & (sel["split_type"] == st) & (sel["criterion"] == crit)
        prim = sel[same & (sel["selector"] == "primary_family_hb")].set_index("split_id")["out_value"]
        lf = sel[same & (sel["selector"] == "label_free_cvll")].set_index("split_id")["out_value"]
        fold_sets = [("all_folds", g, None)]
        if st == "loso":
            low = g["heldout_low_support"].astype(bool) if "heldout_low_support" in g else np.zeros(len(g), bool)
            fold_sets.append(("excl_low_support", g[~low], None))
            fold_sets.append(("weighted_by_heldout_positives", g, g["n_pos_heldout"].to_numpy(float)))
            fold_sets += [(f"study:{h}", g[g["heldout_study"] == h], None) for h in sorted(g["heldout_study"].unique())]
        for fs, gg, w in fold_sets:
            if len(gg) == 0:
                continue
            status = ("prespecified" if (crit == "pooled_ap_lift" and st in ("random_tf_half", "loso")
                                         and fs == "all_folds") else "post_hoc")
            rows.append(dict(grid=grid, n_configs=int(g["n_configs"].iloc[0]), split_type=st, criterion=crit,
                             selector=selector, fold_set=fs, **_opt_row(gg, prim, lf, w),
                             label_free_candidates=";".join(labelfree_candidates) if selector == "label_free_cvll" else "",
                             analysis_status=status))
    out = pd.DataFrame(rows)
    out["_o"] = out["selector"].map({s: i for i, s in enumerate(order)})
    out["_f"] = out["fold_set"].map(lambda f: 0 if f == "all_folds" else (1 if f == "excl_low_support" else
                                                                       (2 if f.startswith("weighted") else 3)))
    out["_c"] = out["criterion"].map({c: i for i, c in enumerate(CRITERIA)})
    return out.sort_values(["grid", "split_type", "_c", "_f", "fold_set", "_o"],
                           ascending=[True, False, True, True, True, True]).drop(columns=["_o", "_f", "_c"])


def full_label_contrasts(blocks: list, cfg_ids: list, lab: Labels, null_info: dict, n_boot: int = None) -> pd.DataFrame:
    """Full-label (pert_pooled) contrasts of every configuration with the primary, with TF-cluster
    bootstrap CIs (B = ``PRESPECIFIED['n_bootstrap']``; TFs resampled with replacement, paired over
    configurations): pooled AP lift difference, and the ratio of the per-TF null-normalised
    geometric means (post hoc, verifier M2/M3)."""
    from r2 import heldout_eval as H
    n_boot = int(PS["n_bootstrap"]) if n_boot is None else n_boot
    idx = null_info["idx_all"]
    tf_m = lab.tf[idx]
    y_m = lab.select()[1][idx]
    tcode, tuniq = pd.factorize(tf_m, sort=True)
    M = H.tf_bootstrap_weights(len(tuniq), n_boot, np.random.default_rng([SEED, 17]))
    pooled = {}
    for cid, b in zip(cfg_ids, blocks):
        bb = b[idx]
        lc = H.level_counts(tcode, y_m, np.zeros(len(bb)), -bb.astype(np.float64), len(tuniq))
        pooled[cid] = (float(H.curve_metrics(*lc.totals(), ks=())["ap_lift"][0]), H.boot_light(lc, M, 500)["ap_lift"])
    # per-TF null-normalised log2 lifts (TFs with >= 1 positive); bootstrap over those TFs
    nn = null_info["per_tf"] - null_info["per_tf_null"]
    has = null_info["tf_has_pos"]
    T_pos = np.flatnonzero(has)
    Mt = H.tf_bootstrap_weights(len(T_pos), n_boot, np.random.default_rng([SEED, 19]))
    i0 = cfg_ids.index(PRIMARY_CONFIG)
    rows = []
    for i, cid in enumerate(cfg_ids):
        d_p = pooled[cid][1] - pooled[PRIMARY_CONFIG][1]
        lo, hi = H.percentile_ci(d_p)
        dt = nn[i, T_pos] - nn[i0, T_pos]
        ok = np.isfinite(dt)
        with np.errstate(invalid="ignore", divide="ignore"):
            rb = 2.0 ** ((Mt[:, ok] @ dt[ok]) / Mt[:, ok].sum(1))
        rlo, rhi = H.percentile_ci(rb)
        rows.append(dict(config_id=cid,
                         full_pooled_diff_vs_primary=pooled[cid][0] - pooled[PRIMARY_CONFIG][0],
                         full_pooled_diff_vs_primary_lo=lo, full_pooled_diff_vs_primary_hi=hi,
                         full_pooled_diff_vs_primary_p_z=H.bootstrap_p_z(d_p, pooled[cid][0] - pooled[PRIMARY_CONFIG][0]),
                         full_tf_nullnorm_ratio_vs_primary=float(2.0 ** np.mean(dt[ok])) if ok.any() else np.nan,
                         full_tf_nullnorm_ratio_vs_primary_lo=rlo, full_tf_nullnorm_ratio_vs_primary_hi=rhi))
    return pd.DataFrame(rows)


def selection_table(detail: pd.DataFrame, full: pd.DataFrame, meta: pd.DataFrame, cv: pd.DataFrame,
                    extra: pd.DataFrame | None = None) -> pd.DataFrame:
    rows = []
    for (grid, st, crit), g in detail.groupby(["grid", "split_type", "criterion"]):
        n_split = g["split_id"].nunique()
        best_out = g.loc[g.groupby("split_id")["out_value"].idxmax()]
        for cid, gc in g.groupby("config_id", sort=False):
            rows.append(dict(grid=grid, split_type=st, criterion=crit, config_id=cid, n_splits=n_split,
                             n_selected=int(gc["selected"].sum()), freq_selected=float(gc["selected"].mean()),
                             n_best_heldout=int((best_out["config_id"] == cid).sum()),
                             freq_best_heldout=float((best_out["config_id"] == cid).mean()),
                             mean_rank_tuning=gc["rank_tuning"].mean(), mean_rank_heldout=gc["rank_heldout"].mean(),
                             mean_in_value=gc["in_value"].mean(), mean_out_value=gc["out_value"].mean()))
    out = pd.DataFrame(rows)
    fw = full.pivot_table(index="config_id", columns="criterion", values="value")
    fw.columns = [f"full_{c}" for c in fw.columns]
    out = out.merge(meta[["config_id", "group", "variant", "fit_id", "t_burst_exchangeable"]], on="config_id",
                    how="left").merge(fw, left_on="config_id", right_index=True, how="left")
    if not cv.empty:
        cvx = cv[cv["complete"]].set_index("config")[["heldout_ll_per_edge_mean", "heldout_ll_per_edge_se",
                                       "diff_vs_family_hb_per_edge_mean", "diff_vs_family_hb_per_edge_se"]]
        out = out.merge(cvx, left_on="config_id", right_index=True, how="left")
    if extra is not None:
        out = out.merge(extra, on="config_id", how="left")
    out["analysis_status"] = np.where(out["criterion"].isin(POST_HOC_CRITERIA) | out["split_type"].isin(POST_HOC_SPLITS),
                                      "post_hoc", "prespecified")
    order = {c: i for i, c in enumerate(meta["config_id"])}
    out["_o"] = out["config_id"].map(order)
    return out.sort_values(["grid", "split_type", "criterion", "_o"],
                           ascending=[True, False, True, True]).drop(columns="_o")


# ----------------------------------------------------------------------------
# Part 3: R1 joint-objective weight sensitivity
# ----------------------------------------------------------------------------
def weight_grid(step: float = WEIGHT_STEP) -> np.ndarray:
    """(w_s, w_b, w_p) on the probability simplex with the given step (66 points at 0.1)."""
    n = int(round(1 / step))
    pts = [(i, j, n - i - j) for i in range(n + 1) for j in range(n + 1 - i)]
    return np.array(pts, dtype=np.float64) / n


def minmax_good(x, higher_better: bool = True) -> np.ndarray:
    """Min-max normalisation to [0, 1] with 1 = best (all-equal -> 0: the term cannot discriminate)."""
    x = np.asarray(x, dtype=np.float64)
    lo, hi = np.nanmin(x), np.nanmax(x)
    if not np.isfinite(hi - lo) or hi - lo <= 0:
        return np.zeros_like(x)
    z = (x - lo) / (hi - lo)
    return z if higher_better else 1.0 - z


def joint_select(W: np.ndarray, d_source, d_block, pert) -> np.ndarray:
    """Selected candidate index per weight row of W under the normalised R1 objective."""
    comp = np.stack([minmax_good(d_source, False), minmax_good(d_block, False), minmax_good(pert, True)])  # (3, M)
    J = W @ comp                                                                                            # (G, M)
    return np.array([select_best(j)[0] for j in J]), J


def wait_for(path: Path, minutes: int = ROB_WAIT_MIN) -> bool:
    t0 = time.time()
    while not path.exists():
        if time.time() - t0 > minutes * 60:
            return False
        log(f"waiting for {path.name} ...")
        time.sleep(60)
    return True


def robustness_components(metric: str) -> pd.DataFrame:
    rob = pd.read_csv(TABLES / "rob_method_ranking.csv")
    if "subset" in rob.columns:      # robustness.py (fit_fixes): family / lineage rows also exist for the
        rob = rob[rob["subset"] == "all"]   # 'excl_uninformative' subset; the R1 block dropout uses all configurations
    r = rob[(rob["tissue"] == ROB_TISSUE) & (rob["metric"] == metric)]
    piv = r.pivot_table(index="estimator", columns="protocol", values="mean_D")
    return piv.rename(columns={"source": "D_source", "family": "D_block"})[["D_source", "D_block"]]


def estimator_blocks(lab: Labels, lcm_blocks: dict) -> dict:
    """Tie blocks of the five LCM estimators (given) and the five combination rules on pert edges."""
    data = lcm.PatternData(*E.load_patterns(UNIVERSE, PRIMARY))
    ep = np.asarray(E.load_edge_pattern(UNIVERSE, PRIMARY))[lab.edge].astype(np.int64)
    bs = lcm.baseline_scores(data)
    out = dict(lcm_blocks)
    for b in BASELINES:
        out[b] = score_blocks(bs[b][ep])
    return out


def run_weights(est_blocks: dict, lab: Labels, splits: list) -> tuple:
    W = weight_grid()
    m_all, y_all = lab.select()
    full_pert = {e: ap_lift(b[m_all], y_all[m_all])["lift"] for e, b in est_blocks.items()}
    split_pert = []
    for tune, hold in splits:
        mt = m_all & np.isin(lab.tf, tune)
        mh = m_all & np.isin(lab.tf, hold)
        split_pert.append({e: (ap_lift(b[mt], y_all[mt])["lift"], ap_lift(b[mh], y_all[mh])["lift"])
                           for e, b in est_blocks.items()})
    grid_rows, summ_rows, cf_rows = [], [], []
    for metric in ROB_METRICS:
        rob = robustness_components(metric)
        for cset, cands in CANDIDATE_SETS.items():
            cands = [c for c in cands if c in rob.index and c in est_blocks]
            ds, db = rob.loc[cands, "D_source"].to_numpy(), rob.loc[cands, "D_block"].to_numpy()
            pf = np.array([full_pert[c] for c in cands])
            sel_full, J = joint_select(W, ds, db, pf)
            # cross-fitted selections
            sel_cf = np.zeros((len(splits), len(W)), np.int64)
            in_cf = np.zeros((len(splits), len(W)))
            out_cf = np.zeros((len(splits), len(W)))
            for s, sp in enumerate(split_pert):
                pin = np.array([sp[c][0] for c in cands])
                pout = np.array([sp[c][1] for c in cands])
                ss, _ = joint_select(W, ds, db, pin)
                sel_cf[s], in_cf[s], out_cf[s] = ss, pin[ss], pout[ss]
                for g, (w, k) in enumerate(zip(W, ss)):
                    cf_rows.append(dict(rob_metric=metric, candidate_set=cset, split_id=s, w_s=w[0], w_b=w[1],
                                        w_p=w[2], selected=cands[k], in_ap_lift=pin[k], out_ap_lift=pout[k]))
            for g, w in enumerate(W):
                Jg = J[g]
                order = np.argsort(-Jg, kind="stable")
                vc = pd.Series([cands[k] for k in sel_cf[:, g]]).value_counts()
                grid_rows.append(dict(
                    rob_metric=metric, candidate_set=cset, w_s=w[0], w_b=w[1], w_p=w[2],
                    selected_full=cands[sel_full[g]], J_selected=Jg[order[0]],
                    runner_up_full=cands[order[1]], J_margin=Jg[order[0]] - Jg[order[1]],
                    full_ap_lift_selected=pf[sel_full[g]],
                    full_ap_lift_family_hb=full_pert[PRIMARY_CONFIG],
                    crossfit_modal_selected=vc.index[0], crossfit_modal_freq=float(vc.iloc[0] / len(splits)),
                    crossfit_n_distinct=int(len(vc)),
                    crossfit_freq_family_hb=float(np.mean([cands[k] == PRIMARY_CONFIG for k in sel_cf[:, g]])),
                    crossfit_in_ap_lift_mean=in_cf[:, g].mean(), crossfit_out_ap_lift_mean=out_cf[:, g].mean(),
                    crossfit_gap_mean=(in_cf[:, g] - out_cf[:, g]).mean(),
                    crossfit_out_family_hb_mean=float(np.mean([sp[PRIMARY_CONFIG][1] for sp in split_pert])),
                ))
            # summary of selection changes
            names = [cands[k] for k in sel_full]
            adj = _adjacent_pairs(W)
            changes = float(np.mean([names[a] != names[b] for a, b in adj]))
            instab = float(np.mean([1 - pd.Series(sel_cf[:, g]).value_counts().iloc[0] / len(splits)
                                    for g in range(len(W))]))
            # the same with the effectively identical em / family_regularized merged (post hoc)
            mg = lambda c: MERGE_EQUIVALENT.get(c, c)  # noqa: E731
            names_m = [mg(x) for x in names]
            changes_m = float(np.mean([names_m[a] != names_m[b] for a, b in adj]))
            instab_m = float(np.mean([1 - pd.Series([mg(cands[k]) for k in sel_cf[:, g]]).value_counts().iloc[0]
                                      / len(splits) for g in range(len(W))]))
            for c in cands:
                summ_rows.append(dict(
                    rob_metric=metric, candidate_set=cset, config_id=c, n_grid=len(W),
                    D_source=float(rob.loc[c, "D_source"]), D_block=float(rob.loc[c, "D_block"]),
                    full_pooled_ap_lift=float(full_pert[c]),
                    share_grid_selected_full=float(np.mean([n == c for n in names])),
                    share_grid_selected_crossfit=float(np.mean([cands[k] == c for k in sel_cf.ravel()])),
                    n_distinct_selected_full=int(len(set(names))),
                    frac_adjacent_weight_pairs_changed=changes,
                    mean_crossfit_selection_instability=instab,
                    frac_adjacent_weight_pairs_changed_merged=changes_m,
                    mean_crossfit_selection_instability_merged=instab_m,
                    n_distinct_selected_full_merged=int(len(set(names_m))),
                    selected_at_wp0=_sel_at(W, names, wp=0.0), selected_at_wp1=_sel_at(W, names, wp=1.0),
                    frac_grid_not_family_hb=float(np.mean([n != PRIMARY_CONFIG for n in names]))))
    return pd.DataFrame(grid_rows), pd.DataFrame(summ_rows), pd.DataFrame(cf_rows), full_pert


def _adjacent_pairs(W: np.ndarray) -> list:
    """Pairs of grid points that differ by one step moved between two weights."""
    d = np.abs(W[:, None, :] - W[None, :, :]).sum(-1)
    i, j = np.where(np.triu(np.isclose(d, 2 * WEIGHT_STEP), 1))
    return list(zip(i.tolist(), j.tolist()))


def _sel_at(W, names, wp) -> str:
    """Selections at w_p = wp (several weight points when wp < 1), joined with ';'."""
    idx = np.flatnonzero(np.isclose(W[:, 2], wp))
    return ";".join(sorted(set(names[i] for i in idx)))


# ----------------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------------
def _write(df: pd.DataFrame, name: str) -> Path:
    """Write a manuscript table with ``analysis_status`` (default prespecified) and ``post_hoc_columns``."""
    df = df.copy()
    if "analysis_status" not in df.columns:
        df["analysis_status"] = "prespecified"
    df["post_hoc_columns"] = POST_HOC_COLUMNS.get(name, "")
    p = TABLES / f"{PREFIX}{name}.csv"
    C.ensure_dir(p.parent)
    df.to_csv(p, index=False, float_format="%.6g")
    log(f"wrote {p.relative_to(C.SP)} ({len(df)} rows)")
    return p


def analyze() -> dict:
    t0 = time.time()
    C.ensure_dir(OUT_DIR)
    cfgs = config_grid()
    cfg_ids = [c.config_id for c in cfgs]
    lab, lab_summ = build_labels()
    np.savez_compressed(OUT_DIR / "labels_global.npz", edge=lab.edge, tf=lab.tf, tested=lab.tested, pos=lab.pos,
                        studies=np.array(lab.studies), tfs=np.array(lab.tfs))
    _write(lab_summ, "label_sets")
    log(f"labels: {lab_summ[lab_summ.label_set == 'pert_pooled'].to_dict('records')}")
    blocks, meta = config_blocks(cfgs, lab)
    # label-free candidates
    cv = labelfree_cv_table()
    cv_check = check_cv_against_table(cv)
    lf_cfg, lf_cands = label_free_choice(cv, cfg_ids)
    if not cv.empty:
        _write(cv, "labelfree_cv")
    log(f"label-free choice: {lf_cfg} among {lf_cands} (max |diff| vs fit_cv_loglik.csv = {cv_check:.2e})")
    m_all, y_all = lab.select()
    grids = {}
    for gname, keep in GRIDS.items():
        idx = [i for i, c in enumerate(cfgs) if keep(c)]
        ids = [cfg_ids[i] for i in idx]
        grids[gname] = (idx, {"label_free_cvll": ids.index(lf_cfg), "primary_family_hb": ids.index(PRIMARY_CONFIG)})
        log(f"grid {gname}: {len(ids)} configurations")
    detail, sel, split_tfs, null_all = run_tuning(blocks, lab, cfg_ids, {}, grids=grids)
    lead = ["grid", "n_configs"]
    detail = detail[lead + [c for c in detail.columns if c not in lead]]
    sel = sel[lead + [c for c in sel.columns if c not in lead]]
    # full-label criteria (descriptive), raw and null-normalised (grid 'all' covers every configuration)
    pooled_null = np.array([np.nanmean(ap_lift_perm(b[null_all["idx_all"]], null_all["Yfull"])) for b in blocks])
    nn = null_all["per_tf"] - null_all["per_tf_null"]
    has = null_all["tf_has_pos"]
    full_rows = []
    for i, (cid, b) in enumerate(zip(cfg_ids, blocks)):
        raw_p = criterion(b, lab, m_all, y_all, "pooled_ap_lift")
        full_rows += [dict(config_id=cid, criterion="pooled_ap_lift", value=raw_p, null_mean=pooled_null[i]),
                      dict(config_id=cid, criterion="tf_geomean_ap_lift",
                           value=criterion(b, lab, m_all, y_all, "tf_geomean_ap_lift"),
                           null_mean=float(2.0 ** np.nanmean(null_all["per_tf_null"][i, has]))),
                      dict(config_id=cid, criterion="pooled_ap_lift_nullnorm", value=raw_p / pooled_null[i],
                           null_mean=1.0),
                      dict(config_id=cid, criterion="tf_geomean_ap_lift_nullnorm",
                           value=float(2.0 ** np.nanmean(nn[i, has])), null_mean=1.0)]
    full = pd.DataFrame(full_rows)
    full.merge(meta, on="config_id").to_csv(OUT_DIR / "config_scores_full.csv", index=False)
    contrasts = full_label_contrasts(blocks, cfg_ids, lab, null_all)
    nullw = full.pivot_table(index="config_id", columns="criterion", values="null_mean")[
        ["pooled_ap_lift", "tf_geomean_ap_lift"]].rename(columns=lambda c: f"full_{c}_null_mean").reset_index()
    detail.to_csv(OUT_DIR / "split_detail.csv", index=False, float_format="%.8g")
    sel.to_csv(OUT_DIR / "selector_detail.csv", index=False, float_format="%.8g")
    split_tfs.to_csv(OUT_DIR / "tf_splits.csv", index=False)
    _write(optimism_table(sel, lf_cands), "optimism")
    _write(selection_table(detail, full, meta, cv, extra=contrasts.merge(nullw, on="config_id")), "selection_frequency")
    _write(status_decomposition(cfgs, lab), "status_decomposition")
    # part 3
    have_rob = wait_for(TABLES / "rob_method_ranking.csv")
    wmeta = {}
    if have_rob:
        lcm_blocks = {c.config_id: b for c, b in zip(cfgs, blocks) if c.group == "estimator"}
        est_blocks = estimator_blocks(lab, lcm_blocks)
        tf_pool = np.unique(lab.tf[m_all])
        grid, summ, cf, full_pert = run_weights(est_blocks, lab, random_tf_splits(tf_pool))
        _write(grid, "weight_sensitivity")
        _write(summ, "weight_summary")
        cf.to_csv(OUT_DIR / "weight_crossfit_detail.csv", index=False, float_format="%.8g")
        wmeta = dict(estimator_full_pooled_ap_lift=full_pert)
    else:
        log("rob_method_ranking.csv not available after waiting; part 3 skipped")
    run_meta = dict(created=time.strftime("%Y-%m-%d %H:%M:%S"), runtime_s=time.time() - t0, seed=SEED,
                    n_splits=N_SPLITS, criteria=list(CRITERIA), post_hoc_criteria=list(POST_HOC_CRITERIA),
                    n_null_permutations=N_NULL_TUNE, excluded_configs=EXCLUDED_CONFIGS,
                    low_support_studies=sorted(low_support_studies()), analysis_decisions=ANALYSIS_DECISIONS,
                    criterion_note=("tf_geomean_ap_lift: geometric mean adopted after the arithmetic mean was seen to "
                                    "be dominated by SRF (lift 342 from 2 positives); post hoc"),
                    configs=meta.to_dict("records"),
                    label_free_choice=lf_cfg, label_free_candidates=lf_cands,
                    cv_recompute_max_abs_diff_vs_fit_cv_loglik=cv_check,
                    analysis_common_available=AC is not None, rob_table_used=have_rob, **wmeta,
                    inputs=dict(cohort_table=str(TABLES / "cohort_table.csv"),
                                rob_method_ranking=str(TABLES / "rob_method_ranking.csv"),
                                fit_cv_loglik=str(TABLES / "fit_cv_loglik.csv")))
    C.write_json(run_meta, OUT_DIR / "run_meta.json")
    record_decisions("tuning_optimism", ANALYSIS_DECISIONS)
    log(f"analysis done in {time.time() - t0:.0f}s")
    return run_meta


def record_decisions(module: str, decisions: list, path: Path | None = None) -> None:
    """Replace this module's rows of the shared dated decision log (``eval_analysis_decisions.csv``)."""
    path = path or (TABLES / "eval_analysis_decisions.csv")
    new = pd.DataFrame(decisions).assign(module=module)
    if path.exists():
        old = pd.read_csv(path)
        new = pd.concat([old[old["module"] != module], new], ignore_index=True)
    C.ensure_dir(path.parent)
    new[["module", "id", "date", "status", "item", "note"]].sort_values(["module", "id"], kind="mergesort").to_csv(
        path, index=False)


def main(argv=None):
    global _LOG_FH
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("command", choices=["cv", "analyze", "all"])
    ap.add_argument("--force", action="store_true", help="recompute existing prior-variant CV folds")
    ap.add_argument("--folds", type=int, nargs="*", default=None, help="CV folds to run (default: all)")
    a = ap.parse_args(argv)
    C.ensure_dir(OUT_DIR)
    _LOG_FH = open(OUT_DIR / "tuning_optimism.log", "a")
    log(f"== tuning_optimism {a.command} ({time.strftime('%Y-%m-%d %H:%M:%S')})")
    if a.command in ("cv", "all"):
        run_prior_cv(force=a.force, folds=a.folds)
    if a.command in ("analyze", "all"):
        analyze()


if __name__ == "__main__":
    main()
