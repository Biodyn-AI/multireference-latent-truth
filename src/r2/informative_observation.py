"""Informative-observation latent-class models: a secondary, post hoc extension (module
``informative_observation``; plan section 8, "Audit + new model").

Why (label-free motivation)
---------------------------
The posterior predictive checks of the MAR estimators (``data/r2/tables/fit_ppc.csv``) show that
the rate at which an edge is reported rises far more steeply with the number of literature (LIT)
annotators observing it than any MAR latent-class model predicts: observation is not ignorable.
The held-out evaluation agrees (unobserved edges are depleted of held-out positives, yet MAR
models rank them above observed negatives).  This module fits the selection-model extension of
``lcm.py`` in which observation of a LIT annotator is itself an emission of the latent class:

    LIT annotator j is ELIGIBLE for edge (t, g) iff t is covered by j (TF-level design) and
    OBSERVES it iff eligible and g is a node of j (the primary curated-conditional mask);
    o_ij | z_i = s, eligible ~ Bernoulli(eta_js), eta_jT for s = T, eta_jNonT for N and bursts;
    r_ij | o_ij = 1, z_i = s ~ Bernoulli(theta_js)  (unchanged).

``family_hb_obs`` = Family-HB + informative observation (eta logits pooled within LIT, logit-
normal, HalfNormal(1) scales); ``em_obs`` = Dawid-Skene + informative observation (independent
N(0, 3) logits).  No ordering between eta_jT and eta_jNonT.  MOTIF/COEXP observation is
conditioned on: under the curated-conditional mask a MOTIF/COEXP annotator observes (t, g) only if
(t, g) carries a held-in DoRothEA-channel literature record, i.e. its observation is a
deterministic function of the held-in LIT data (records and their DoRothEA channel flag).  Given
those data it carries no further information on z (missing at random given the observed data),
so it is treated as fixed design and coded -2 when absent; only the LIT observation process,
which is not a function of other observed quantities, is modelled.

Status: the extension was designed AFTER the held-out results had been seen.  Every row this
module writes carries ``analysis_status`` = ``post_hoc`` except rows that reproduce prespecified
quantities of the MAR models (same score, metric, data and resamples as the prespecified
tables), which carry ``prespecified``.  Family-HB (MAR) remains the prespecified primary model.

Data
----
Eligibility-coded evidence of the primary variant (``evidence.build_elig_patterns``;
``evidence.load_patterns(u, 'prespecified_cc__elig')``): codes 1 reported, 0 observed not reported,
-1 eligible not observed (LIT), -2 not eligible / not modelled.  It refines the stored evidence
(collapsing -2 to -1 gives the stored pattern of every edge), so MAR credibilities map onto it
through the ``base_pattern`` column.

Stages (``python src/r2/informative_observation.py <stage>``)
------------------------------------------------------------
``evidence``  build / verify the ``__elig`` evidence files of global, immune, kidney, lung.
``map``       MAP of ``family_hb_obs`` and ``em_obs`` (``lcm.fit_map``, 50 L-BFGS restarts, seed
              ``PRESPECIFIED['random_seed']``) through ``fit_primary.run_map`` (Newton polish,
              Hessian, distinct optima), then ``fit_primary.laplace_optima`` (Laplace mass of each
              distinct optimum).  Stored under the fit contract
              ``outputs/r2/fits/<u>/prespecified_cc__elig/<fit_id>/`` (``cred.npy`` aligned with
              ``evidence.load_patterns(u, 'prespecified_cc__elig')``).
``map_tfw``   sensitivity (post hoc): the same two models with the observation log-likelihood of LIT
              annotator j weighted by 1 / (number of universe TFs eligible for j) (``lcm`` prior option
              ``obs_weight``; fit ids ``family_hb_obs__tfw``, ``em_obs__tfw``; MAP, 20 restarts; every
              universe).  Eligibility is TF-level and observation gene-level, so the node indicator of a
              gene is replicated over every eligible TF; the weight counts it once per gene and annotator
              (composite likelihood).  It tests whether the generative model's behaviour is driven by
              this replication.
``map_cond``  EXPLORATORY, not run by ``all`` and not part of the reported results: the same two models
              fitted by the conditional likelihood sum_p n_p log p(r_p | o_p) + log prior (``lcm`` prior
              option ``obs_conditional``; fit ids ``family_hb_obs__cond``, ``em_obs__cond``).  The
              observation pattern only shapes the class prior P(z | o) (latent-class model with the
              observation pattern as concomitant variable).  A check on immune (2026-10-13,
              ``logs/cond_exploration_immune.log``) found the objective strongly multimodal with
              unanchored labels (every start of ``em_obs`` ended in a different proper optimum; the best
              ``family_hb_obs`` optimum had baseline w_T = 0.997), so no fit was stored.  If such fits are
              produced with this stage, the ppc / cv / heldout / tables stages include them.
``nuts``      NOT RUN and NOT part of ``all`` (decision of 2026-10-13): no ``family_hb_obs_nuts`` fit
              exists and every reported quantity of this module comes from the MAP fits, with
              Laplace intervals (Gaussian at the MAP optimum) for eta and w_T.  Reasons: (i) cost --
              in the non-centred coordinates of the MAR fits the sampler saturates the tree depth
              (immune pilot: 1,023 leapfrogs per iteration, step 1e-3; projected 4.7 h for immune
              and about 14 h for global at 4 x 1,000 + 1,000; a dense Laplace metric does not help);
              (ii) it would not change a conclusion -- the MAP analysis shows a degenerate model
              (two label-swapped basins, 187 to 68,764 log-posterior units apart, each reached by
              about half of the restarts), and a chain started in one basin only quantifies the
              within-basin spread, which the Laplace intervals already show to be negligible
              (global ``family_hb_obs``: w_T 0.6776, 95% interval 0.6771 to 0.6781).
              The stage is kept for completeness: NUTS of ``family_hb_obs`` (target acceptance 0.95,
              settings ``NUTS_KW``), chains started from basin-checked Laplace draws around the best
              MAP optimum (``fit_primary.run_nuts(init='map_laplace')``), in partially centred
              coordinates (``NUTS_PRIOR``; same posterior; pilot with these coordinates: mean 450
              leapfrogs, projected 3.8 h for immune), start centre = the MAP mode re-optimised in
              those coordinates (``centered_mode``).  The auxiliary directories
              ``family_hb_obs__cmode`` (global, immune) left by the pilots hold only that start
              centre; no stage, table or report reads them.  If a ``family_hb_obs_nuts`` fit is ever
              produced, the heldout / tables stages add it together with its MAR counterpart
              ``family_hb_nuts`` (otherwise ``family_hb_nuts`` is not read, so the tables do not
              depend on re-runs of the primary NUTS fits).
``ppc``       posterior predictive checks with ``fit_sensitivity.ppc_statistics`` (same statistics,
              full data and LIT block), 20 replicates per fit, for the MAR fits ``family_hb`` / ``em``
              (canonical primary MAP) and ``family_hb_obs`` / ``em_obs``:
              * ``conditional`` (all four): the real observation masks are kept; z is drawn from
                P(z | o) (= w for MAR fits, so this is the MAR check of ``fit_sensitivity``) and the
                reports from theta.  Answers: given which annotators observe an edge, does the model
                reproduce how often (and by how many) it is reported?
              * ``joint`` (informative-observation fits; MAR fits with class-independent observation
                rates eta_j = empirical rate, ``*+ignorable_obs``): the LIT observation indicators of
                every eligible entry are simulated too (MOTIF/COEXP masks fixed); adds the per-
                annotator observation rate and the distribution of the number of observing LIT
                annotators among edges with >= 1 eligible LIT annotator.
``cv``        5-fold TF-grouped cross-validation (the TF folds of ``fit_sensitivity``; global and
              immune): ``family_hb_obs`` / ``em_obs`` refitted on the training TFs (MAP; one warm
              start from every distinct optimum of the full-data fit plus ``N_RESTARTS_CV`` random
              restarts, best training log posterior; ``cv_fit``); MAR ``family_hb`` / ``em`` from the
              stored fold fits of ``fit_sensitivity`` (20 restarts).  Per fold: conditional held-out log-likelihood of
              the reports given the observation pattern, log p(r | o) per observed test edge
              (comparable across all four; for MAR fits it is the ordinary held-out marginal
              likelihood), and the joint log p(o, r) per test edge for the obs fits and for the MAR
              fits with class-independent observation rates estimated on the training edges.
``heldout``   post hoc held-out evaluation with ``heldout_eval`` functions (the prespecified tables of
              that module are not touched): references encode, encode_strict, enets2, pert_pooled in
              every universe; AP lift (TF-cluster bootstrap CI), within-TF AUROC, Enr@k and TF x
              target-degree stratified enrichment, within-TF permutation p; paired TF-cluster
              bootstrap differences of each informative-observation fit vs Family-HB, EM and Fisher;
              evidence-status O/E (unobserved / observed negative / reported), with the unobserved
              class split by the number of eligible-but-unobserving LIT annotators; the label-free
              3-level status score (unobserved < observed negative < reported).
``tables``    ``obs_fit_summary.csv``, ``obs_map_optima.csv``, ``obs_params.csv``, ``obs_ppc.csv``, ``obs_cv.csv`` (the
              ``heldout`` stage writes ``obs_heldout.csv``, ``obs_heldout_paired.csv``,
              ``obs_heldout_status.csv``), then ``check``.
``check``     freshness of every output against the stored fits (``check_outputs``): PPC / CV files
              and the tables must be newer than the fits they were computed from, the credibility
              files behind the held-out scores must have the md5 recorded by the ``heldout`` stage.
              Writes ``outputs/r2/informative_observation/output_check.json``; fails if stale.
``all``       evidence, map, map_tfw, ppc, cv, heldout, tables (every reported result; not ``nuts``,
              not ``map_cond``).

Command line (from the subproject root; thread limits as in the other modules)::

    export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \
           XLA_FLAGS='--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=2'
    /Users/ihorkendiukhov/biodyn-work/.venv/bin/python src/r2/informative_observation.py all
    # or: evidence | map [--universes ...] [--models ...] | map_tfw | ppc | cv [--folds 0 1 ...]
    #     | heldout | tables | check        (not in ``all``: nuts [--universes ..], map_cond)
    /Users/ihorkendiukhov/biodyn-work/.venv/bin/python -m pytest src/r2/tests/test_informative_observation.py -q
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
import zlib  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from r2 import common as C  # noqa: E402
from r2 import evidence as E  # noqa: E402
from r2 import lcm  # noqa: E402

PS = C.PRESPECIFIED
SEED = int(PS["random_seed"])
VARIANT = PS["evidence_variant_primary"]                 # prespecified_cc
ELIG_VARIANT = E.elig_variant(VARIANT)                   # prespecified_cc__elig
UNIVERSES = ("global",) + tuple(C.TISSUES)
NUTS_UNIVERSES = ("global", "immune")
PPC_UNIVERSES = ("global", "immune")
CV_UNIVERSES = ("global", "immune")
OBS_FITS = ("family_hb_obs", "em_obs")
NUTS_FIT = "family_hb_obs_nuts"
# sensitivity (post hoc): observation log-likelihood of LIT annotator j weighted by 1 / (number of
# universe TFs eligible for j), so that each gene's node indicator counts once per annotator
TFW_FITS = ("family_hb_obs__tfw", "em_obs__tfw")
TFW_UNIVERSES = UNIVERSES
N_RESTARTS_TFW = 20
# conditional-likelihood variant (EXPLORATORY; stage ``map_cond`` only, not in ``all``; no stored fits,
# see the module docstring): the same models fitted by the conditional likelihood of the reports given
# the observation pattern (``lcm`` prior option ``obs_conditional``)
COND_FITS = ("family_hb_obs__cond", "em_obs__cond")
COND_PRIOR = {"obs_conditional": True}
N_RESTARTS_COND = 50
MAR_FITS = ("family_hb", "em")
N_RESTARTS = 50
N_RESTARTS_CV = 4          # random restarts per CV fold fit, in addition to one warm start from every
#                           distinct optimum of the full-data fit (see ``cv_fit``)
# start points of the CV fold fits per universe.  Immune: every distinct full-data optimum + 4 random
# starts.  Global (one start costs ~10x the immune one): warm starts from the best optimum and the
# alternative optimum (``alt_optimum``: label-swapped / second) only -- in immune the warm starts
# reached the best training optimum found by the 4 additional random starts in every fold fit
# (|difference| < 1e-6 nats; column ``n_folds_random_start_beats_warm`` of obs_cv.csv).
CV_STARTS = {"immune": dict(warm="all", n_random=N_RESTARTS_CV), "global": dict(warm="best_alt", n_random=0)}
N_PPC = 20
N_FOLDS = 5
# NUTS settings per universe -- for the stage ``nuts``, which was NOT RUN (see the module docstring; the
# results are MAP + Laplace only) and is not part of ``all``.  Plan section 6: 4 chains x 1,000 warm-up +
# 1,000 draws (immune).  Global: 4 x 300 + 300, chosen from the measured cost: one gradient of the
# 115,973-pattern global likelihood costs ~3x the immune one (pilot: ~8 ms vs ~3 ms on the shared machine),
# so at the pilot's ~450 leapfrogs per iteration the plan setting projects to ~8 h and this setting to
# ~2.5 h; the immune pilot with 300 warm-up iterations mixed (bulk ESS 70-240 per 200 draws).
NUTS_KW = {"immune": dict(num_chains=4, num_warmup=1000, num_samples=1000),
           "global": dict(num_chains=4, num_warmup=300, num_samples=300)}
# NUTS coordinates: the annotator-level parameters the data determine sharply (clean false-positive
# logits b and the observation logits eT, eN) are sampled centred, q_j ~ N(mu_f, tau); a, g and the
# family level stay non-centred (``lcm.build_model(prior={'centered': ...})``; same posterior).  With
# the all-non-centred coordinates of the MAR fits the sampler saturates at tree depth 10 with step
# size 1e-3 (pilot, immune: 1,023 leapfrogs per iteration, ESS(w) 2 per 100 draws, also with a
# dense Laplace metric); centring everything is not possible (a_j is weakly determined: the centred
# density is unbounded as tau_a -> 0).  Pilot with these coordinates (immune, 300 + 200): mean 450
# leapfrogs, step 1.2e-2, bulk ESS 70-240 per 200 draws.  Logs: outputs/r2/informative_observation/logs/nuts_pilot*.log
NUTS_CENTERED = ("b", "eT", "eN")
NUTS_PRIOR = {"centered": NUTS_CENTERED}
CMODE_FIT = "family_hb_obs__cmode"       # auxiliary: the best MAP mode re-optimised in the NUTS coordinates
HELDOUT_REFS = ("encode", "encode_strict", "enets2", "pert_pooled")
PAIRED_VS = {"family_hb_obs": ("family_hb", "em", "fisher", "status3"),
             "em_obs": ("em", "family_hb", "fisher", "status3"),
             "family_hb_obs_swap": ("family_hb", "em", "fisher", "status3"),
             "em_obs_swap": ("em", "family_hb", "fisher", "status3"),
             "family_hb_obs__tfw": ("family_hb", "em", "fisher", "status3", "family_hb_unobserved_last"),
             "em_obs__tfw": ("em", "family_hb", "fisher", "status3", "family_hb_unobserved_last"),
             "family_hb_obs__tfw_opt2": ("family_hb", "em", "fisher", "status3"),
             "family_hb_obs__cond": ("family_hb", "em", "fisher", "status3", "family_hb_unobserved_last"),
             "em_obs__cond": ("em", "family_hb", "fisher", "status3", "family_hb_unobserved_last"),
             "family_hb_obs__cond_opt2": ("family_hb", "em", "fisher", "status3"),
             "em_obs__cond_opt2": ("em", "family_hb", "fisher", "status3"),
             "family_hb_obs_nuts": ("family_hb_nuts", "family_hb", "em", "fisher", "status3")}
NREP_CAP = 6
NOBS_LIT_BINS = ((0, 0), (1, 1), (2, 2), (3, 4), (5, 7), (8, 99))
UNOBS_ELIG_BINS = ((0, 0), (1, 2), (3, 5), (6, 99))

FITS = C.OUT / "fits"
MOD = C.OUT / "informative_observation"
TABLES = C.TABLES
FIT_SENS_CV = C.OUT / "fit_sensitivity" / "cv"


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def derived_seed(*parts) -> int:
    """Deterministic seed of a sub-task (same rule as ``fit_sensitivity.derived_seed``)."""
    return int((SEED + zlib.crc32("/".join(map(str, parts)).encode())) % (2 ** 31 - 1))


def _jsonable(x):
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return _jsonable(x.tolist())
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    return x


# ----------------------------------------------------------------------------------------
# Data and fits
# ----------------------------------------------------------------------------------------
def load_elig(universe: str, with_base: bool = False):
    """Eligibility-coded PatternData of the primary variant (and the base-pattern map)."""
    R, n, fam, ann, base = E.load_patterns_elig(universe, VARIANT, with_base=True)
    data = lcm.PatternData(R, n, fam, ann, eligibility=True)
    return (data, base) if with_base else data


def load_plain(universe: str) -> lcm.PatternData:
    return lcm.PatternData(*E.load_patterns(universe, VARIANT))


def edge_pattern_elig(universe: str) -> np.ndarray:
    return np.asarray(E.load_edge_pattern(universe, ELIG_VARIANT, mmap=False)).astype(np.int64)


def fit_dir(universe: str, fit_id: str) -> Path:
    """Contract directory of a fit: MAR fits under the primary variant, obs fits under
    the eligibility-coded pseudo-variant."""
    variant = ELIG_VARIANT if fit_id.startswith(OBS_FITS) else VARIANT
    return FITS / universe / variant / fit_id


def read_fit(universe: str, fit_id: str) -> dict | None:
    d = fit_dir(universe, fit_id)
    if not (d / "meta.json").exists():
        return None
    meta = json.loads((d / "meta.json").read_text())
    with np.load(d / "params.npz") as z:
        params = {k: np.asarray(z[k]) for k in z.files}
    out = dict(meta=meta, params=params, cred=np.load(d / "cred.npy"), dir=d)
    if meta.get("inference") == "NUTS":
        with np.load(d / "draws.npz") as z:
            out["draws"] = {k: np.asarray(z[k]) for k in z.files}
        for s in ("lo", "hi", "sd"):
            out[f"cred_{s}"] = np.load(d / f"cred_{s}.npy")
        out["diagnostics"] = pd.read_csv(d / "diagnostics.csv")
    return out


def mar_params(params: dict) -> dict:
    return {"w": np.asarray(params["w"]), "theta": np.asarray(params["theta"])}


def obs_params(params: dict) -> dict:
    out = {"w": np.asarray(params["w"]), "theta": np.asarray(params["theta"]), "eta": np.asarray(params["eta"])}
    if "obs_weight" in params:
        out["obs_weight"] = np.asarray(params["obs_weight"])
    return out


def _unweighted(params: dict) -> dict:
    """The generative (unweighted) parameter set of a fit (drops ``obs_weight``)."""
    return {k: v for k, v in params.items() if k != "obs_weight"}


def all_obs_fits() -> tuple:
    return OBS_FITS + TFW_FITS + COND_FITS


def mar_comparators(universe: str) -> tuple:
    """MAR fits read next to the informative-observation fits: the MAP fits ``family_hb`` and
    ``em`` always; ``family_hb_nuts`` only if a ``family_hb_obs_nuts`` fit exists in the universe
    (it is the counterpart of that fit alone; the NUTS stage was not run, so the tables of this
    module do not depend on the primary NUTS fits, which ``fit_primary`` may re-run)."""
    has_nuts = (fit_dir(universe, NUTS_FIT) / "meta.json").exists()
    return MAR_FITS + (("family_hb_nuts",) if has_nuts else ())


def tf_replication_weights(universe: str, data: lcm.PatternData) -> np.ndarray:
    """obs_weight_j = 1 / (number of universe TFs eligible for LIT annotator j): the node
    indicator of a gene is replicated over every eligible TF of the annotator (eligibility is
    TF-level, observation gene-level), so this weight counts it once per gene and annotator
    (composite likelihood); 1 for the other annotators (no observation term)."""
    m = json.loads(E._elig_paths(universe, VARIANT)["meta"].read_text())
    ntf = {r["annotator"]: r["n_tfs_eligible"] for r in m["per_annotator"]}
    w = np.ones(data.K)
    for j in data.obs_columns(("LIT",)):
        w[j] = 1.0 / max(ntf[data.annotators[j]], 1)
    return w


def cred_on_elig(universe: str, fit_id: str, data: lcm.PatternData, base: np.ndarray) -> np.ndarray:
    """Credibility of a stored fit on the eligibility-coded patterns (MAR fits mapped through
    the base-pattern index; obs fits as stored)."""
    f = read_fit(universe, fit_id)
    if f is None:
        raise FileNotFoundError(f"{fit_dir(universe, fit_id)} missing")
    if fit_id.startswith(OBS_FITS):
        assert len(f["cred"]) == data.P
        return f["cred"]
    return np.asarray(f["cred"])[base]


def file_md5(p: Path) -> str:
    import hashlib
    h = hashlib.md5()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def evidence_record(universe: str) -> dict:
    p = E._elig_paths(universe, VARIANT)
    return {"patterns": str(p["patterns"].relative_to(C.SP)), "patterns_md5": file_md5(p["patterns"]),
            "edge_pattern_md5": file_md5(p["edge_pattern"]),
            "base_patterns_md5": file_md5(p["base_patterns"])}


# ----------------------------------------------------------------------------------------
# Stage: evidence
# ----------------------------------------------------------------------------------------
def run_evidence(universes=UNIVERSES) -> None:
    for u in universes:
        p = E.ensure_elig_patterns(u, VARIANT)
        m = json.loads(p["meta"].read_text())
        log(f"[evidence] {u}: {m['n_patterns']:,} eligibility patterns ({m['n_base_patterns']:,} stored), "
            f"{m['n_edges']:,} edges, {len(m['modelled_annotators'])} modelled annotators")


# ----------------------------------------------------------------------------------------
# Stage: map (+ Laplace mass of every distinct optimum)
# ----------------------------------------------------------------------------------------
def _annotate_meta(d: Path, universe: str, extra: dict) -> None:
    meta = json.loads((d / "meta.json").read_text())
    meta.update(extra)
    meta["evidence"] = evidence_record(universe)
    meta["analysis_status"] = "post_hoc"
    meta["analysis_note"] = ("informative-observation extension designed after the held-out results were seen; "
                             "secondary analysis (plan section 8)")
    C.write_json(_jsonable(meta), d / "meta.json")


def run_map(universes=UNIVERSES, models=OBS_FITS, n_restarts=N_RESTARTS, force=False) -> None:
    from r2 import fit_primary as FP
    for u in universes:
        data = load_elig(u)
        for m in models:
            d = fit_dir(u, m)
            if (d / "meta.json").exists() and (d / "optima_params.npz").exists() and not force:
                log(f"[map] {u}/{m}: exists")
                continue
            t0 = time.time()
            FP.run_map(u, m, variant=ELIG_VARIANT, n_restarts=n_restarts, seed=SEED, data=data, fit_id=m)
            FP.laplace_optima(u, model=m, variant=ELIG_VARIANT, data=data, fit_id=m)
            _annotate_meta(d, u, {"mar_counterpart": lcm.MAR_COUNTERPART[m],
                                  "pattern_rows": f"evidence.load_patterns('{u}', '{ELIG_VARIANT}')"})
            log(f"[map] {u}/{m}: done ({time.time() - t0:.0f}s)")


def run_map_tfw(universes=TFW_UNIVERSES, models=OBS_FITS, n_restarts=N_RESTARTS_TFW, force=False) -> None:
    """Sensitivity fits ``<model>__tfw`` (obs_weight = TF-replication weights)."""
    from r2 import fit_primary as FP
    for u in universes:
        data = load_elig(u)
        prior = {"obs_weight": tf_replication_weights(u, data)}
        for m in models:
            fid = f"{m}__tfw"
            d = fit_dir(u, fid)
            if (d / "meta.json").exists() and (d / "optima_params.npz").exists() and not force:
                log(f"[map_tfw] {u}/{fid}: exists")
                continue
            t0 = time.time()
            FP.run_map(u, m, variant=ELIG_VARIANT, n_restarts=n_restarts, seed=SEED, data=data, fit_id=fid,
                       prior=prior)
            FP.laplace_optima(u, model=m, variant=ELIG_VARIANT, data=data, fit_id=fid, prior=prior)
            _annotate_meta(d, u, {"mar_counterpart": lcm.MAR_COUNTERPART[m],
                                  "obs_weight_rule": "1 / number of universe TFs eligible for the LIT annotator",
                                  "pattern_rows": f"evidence.load_patterns('{u}', '{ELIG_VARIANT}')"})
            log(f"[map_tfw] {u}/{fid}: done ({time.time() - t0:.0f}s)")


def run_map_cond(universes=UNIVERSES, models=OBS_FITS, n_restarts=N_RESTARTS_COND, force=False) -> None:
    """Conditional-likelihood fits ``<model>__cond`` (``COND_PRIOR``): MAP of the same parameters
    under sum_p n_p log p(r_p | o_p) + log prior."""
    from r2 import fit_primary as FP
    for u in universes:
        data = load_elig(u)
        for m in models:
            fid = f"{m}__cond"
            d = fit_dir(u, fid)
            if (d / "meta.json").exists() and (d / "optima_params.npz").exists() and not force:
                log(f"[map_cond] {u}/{fid}: exists")
                continue
            t0 = time.time()
            FP.run_map(u, m, variant=ELIG_VARIANT, n_restarts=n_restarts, seed=SEED, data=data, fit_id=fid,
                       prior=dict(COND_PRIOR))
            FP.laplace_optima(u, model=m, variant=ELIG_VARIANT, data=data, fit_id=fid, prior=dict(COND_PRIOR))
            _annotate_meta(d, u, {"mar_counterpart": lcm.MAR_COUNTERPART[m],
                                  "objective": "conditional likelihood of the reports given the observation pattern, "
                                               "log p(r | o), + log prior; meta 'loglik' is the joint log p(o, r) "
                                               "at the fitted parameters",
                                  "pattern_rows": f"evidence.load_patterns('{u}', '{ELIG_VARIANT}')"})
            log(f"[map_cond] {u}/{fid}: done ({time.time() - t0:.0f}s)")


# ----------------------------------------------------------------------------------------
# Stage: nuts
# ----------------------------------------------------------------------------------------
def chain_loglik_columns(d: Path, data: lcm.PatternData) -> pd.DataFrame:
    """Add the joint log p(o, r) and the conditional log p(r | o) per chain to ``chains.csv``
    (``fit_primary.run_nuts`` records the report-only marginal likelihood)."""
    with np.load(d / "draws.npz") as z:
        W, TH, ETA, chain = z["w"], z["theta"], z["eta"], z["chain"]
    jl = np.array([lcm.heldout_loglik({"w": W[i], "theta": TH[i], "eta": ETA[i]}, data) for i in range(len(W))])
    cl = np.array([lcm.conditional_loglik({"w": W[i], "theta": TH[i], "eta": ETA[i]}, data) for i in range(len(W))])
    ch = pd.read_csv(d / "chains.csv")
    ch["mean_joint_loglik"] = [float(jl[chain == c].mean()) for c in ch["chain"]]
    ch["mean_cond_loglik_r_given_o"] = [float(cl[chain == c].mean()) for c in ch["chain"]]
    ch["mean_loglik_note"] = "mean_loglik = report-only marginal at (w, theta) (fit_primary); joint uses eta"
    ch.to_csv(d / "chains.csv", index=False)
    return ch


def centered_mode(u: str, data: lcm.PatternData, force: bool = False) -> Path:
    """The best MAP optimum of ``family_hb_obs`` re-optimised in the NUTS coordinates
    (``NUTS_PRIOR``): L-BFGS (options of ``lcm.fit_map``) started at the stored MAP parameters,
    then Newton polish.  A posterior mode depends on the coordinates (Jacobian of q = mu + tau z),
    so the centred mode differs slightly from the stored MAP; it lies in the same basin (the
    record gives the largest credibility difference and the Spearman correlation with the MAP).
    Written as the auxiliary fit ``family_hb_obs__cmode`` (``params.npz``, ``cred.npy``,
    ``meta.json``; ``optima.csv`` / ``cred_restarts.npy`` are those of the MAP fit so that
    ``fit_primary.run_nuts`` assigns every chain to the nearest MAP optimum).  It only provides
    the centre of the basin-checked Laplace starts of the NUTS chains."""
    import shutil
    import jax.numpy as jnp
    from jax.flatten_util import ravel_pytree
    from numpyro.infer.util import unconstrain_fn
    from scipy.optimize import minimize
    from r2 import fit_primary as FP
    d_map, d = fit_dir(u, "family_hb_obs"), fit_dir(u, CMODE_FIT)
    if (d / "meta.json").exists() and not force:
        return d
    t0 = time.time()
    pt = FP._potential(data, "family_hb_obs", dict(NUTS_PRIOR))
    with np.load(d_map / "params.npz") as z:
        best = {k: np.asarray(z[k]) for k in z.files}
    z0 = unconstrain_fn(pt["model"], (), {}, {k: jnp.asarray(best[k]) for k in pt["names"]})
    x0 = np.asarray(ravel_pytree(z0)[0], dtype=float)
    res = minimize(lambda x: float(pt["pot"](jnp.asarray(x))), x0,
                   jac=lambda x: np.asarray(pt["grad"](jnp.asarray(x))),
                   method="L-BFGS-B", options=dict(FP.LBFGS_OPTIONS))
    x, prec = FP.newton_polish(pt, np.asarray(res.x))
    smp = FP._constrained(pt, x)
    cred = lcm.credibility_at(obs_params(smp), data)
    cred_map = np.load(d_map / "cred.npy")
    C.ensure_dir(d)
    np.savez(d / "params.npz", **{k: np.asarray(v) for k, v in smp.items() if k != "cred"})
    np.save(d / "cred.npy", cred)
    for f in ("optima.csv", "cred_restarts.npy"):
        shutil.copy2(d_map / f, d / f)
    meta = dict(fit_id=CMODE_FIT, model="family_hb_obs", inference="MAP (NUTS coordinates; auxiliary)", universe=u,
                variant=ELIG_VARIANT, prior_overrides=dict(NUTS_PRIOR), started_from=str(d_map.relative_to(C.SP)),
                log_post_at_map_parameters=-float(pt["pot"](jnp.asarray(x0))), log_post=-float(pt["pot"](jnp.asarray(x))),
                lbfgs_iterations=int(res.nit), polish=prec, w=np.asarray(smp["w"]),
                max_abs_cred_diff_vs_map=float(np.max(np.abs(cred - cred_map))),
                spearman_w_vs_map=FP.weighted_spearman(cred, cred_map, data.n),
                note="optima.csv and cred_restarts.npy are copies of the family_hb_obs MAP fit",
                analysis_status="post_hoc", runtime_s=time.time() - t0)
    C.write_json(_jsonable(meta), d / "meta.json")
    log(f"[nuts] {u}: mode in the NUTS coordinates: log post {meta['log_post']:.3f}, polish {prec['status']} "
        f"grad {prec['grad_norm_after']:.1e} min eig {prec['hess_min_eig']:.3g}, max|dcred| vs MAP "
        f"{meta['max_abs_cred_diff_vs_map']:.3g}, Spearman {meta['spearman_w_vs_map']:.6f}")
    if not prec["hess_min_eig"] > 0:
        raise RuntimeError(f"[{u}] no proper mode in the NUTS coordinates (Hessian not positive definite)")
    return d


def run_nuts(universes=NUTS_UNIVERSES, force=False) -> None:
    from r2 import fit_primary as FP
    for u in universes:
        d = fit_dir(u, NUTS_FIT)
        if (d / "meta.json").exists() and not force:
            log(f"[nuts] {u}: exists")
            continue
        data = load_elig(u)
        cm = centered_mode(u, data, force=force)
        FP.run_nuts(u, variant=ELIG_VARIANT, data=data, model="family_hb_obs", fit_id=NUTS_FIT,
                    nuts_kw=dict(NUTS_KW[u]), init="map_laplace", map_fit_id=CMODE_FIT, prior=dict(NUTS_PRIOR),
                    seed=SEED)
        chain_loglik_columns(d, data)
        cmeta = json.loads((cm / "meta.json").read_text())
        _annotate_meta(d, u, {
            "mar_counterpart": "family_hb_nuts", "pattern_rows": f"evidence.load_patterns('{u}', '{ELIG_VARIANT}')",
            "nuts_coordinates": dict(
                centered=list(NUTS_CENTERED),
                description="annotator-level b, eT, eN sampled centred (q_j ~ N(mu_f, tau)); a, g and the family "
                            "level non-centred; same posterior as the non-centred model of the MAP fit",
                reason="the non-centred sampler of fit_primary saturates at tree depth 10 (pilot logs)",
                init_centre=CMODE_FIT, init_centre_max_abs_cred_diff_vs_map=cmeta["max_abs_cred_diff_vs_map"],
                init_centre_spearman_vs_map=cmeta["spearman_w_vs_map"]),
            "nuts_settings_note": ("plan section 6 settings (4 x 1000/1000)" if NUTS_KW[u]["num_samples"] == 1000 else
                                   "reduced from 4 x 1000/1000 for runtime, fixed before the run (see NUTS_KW)")})


# ----------------------------------------------------------------------------------------
# Simulation for the posterior predictive checks
# ----------------------------------------------------------------------------------------
def _decode3(codes: np.ndarray, K: int) -> np.ndarray:
    out = np.empty((len(codes), K), np.int8)
    c = codes.copy()
    for j in range(K):
        out[:, j] = (c % 3) - 1
        c //= 3
    return out


def _compress(us: list, cs: list, K: int, data: lcm.PatternData) -> lcm.PatternData:
    u_all, inv = np.unique(np.concatenate(us), return_inverse=True)
    cnt = np.bincount(inv.reshape(-1), weights=np.concatenate(cs))
    return lcm.PatternData(_decode3(u_all, K), cnt, list(data.family), list(data.annotators))


def _draw_classes(rng, probs: np.ndarray) -> np.ndarray:
    """One categorical draw per row of ``probs`` (inverse CDF)."""
    cdf = np.cumsum(probs, axis=1)
    u = rng.random(len(probs)) * cdf[:, -1]
    return np.minimum((u[:, None] > cdf).sum(1), probs.shape[1] - 1)


def simulate_conditional(data: lcm.PatternData, params: dict, seed: int, chunk: int = 400_000) -> lcm.PatternData:
    """Replicate reports on the REAL observation masks of the edges observed by >= 1 annotator;
    z ~ P(z | o) (``lcm.class_posteriors(given='obs')``; = w for MAR parameters).  Returns plain
    (collapsed) PatternData of the simulated edges."""
    rng = np.random.default_rng(seed)
    obs = data.R >= 0
    keep = obs.any(1) & (data.n > 0)
    probs = lcm.class_posteriors(params, data, given="obs")[keep]
    th = np.clip(np.asarray(params["theta"], float), 0, 1)
    ok = obs[keep]
    cnt = np.rint(data.n[keep]).astype(np.int64)
    idx = np.repeat(np.arange(len(cnt)), cnt)
    K = data.K
    pow3 = 3 ** np.arange(K, dtype=np.int64)
    us, cs = [], []
    for s0 in range(0, len(idx), chunk):
        ii = idx[s0:s0 + chunk]
        z = _draw_classes(rng, probs[ii])
        rep = rng.random((len(ii), K)) < th.T[z]
        M = np.where(ok[ii], rep.astype(np.int8), np.int8(-1))
        u, c = np.unique((M.astype(np.int64) + 1) @ pow3, return_counts=True)
        us.append(u)
        cs.append(c)
    return _compress(us, cs, K, data)


def ignorable_eta(data: lcm.PatternData, obs_families=("LIT",)) -> np.ndarray:
    """Class-independent observation rates (K, 2): empirical eligible -> observed rate of every
    modelled annotator (NaN rows for the others); the MAR model + ignorable observation."""
    eta = np.full((data.K, 2), np.nan)
    for j in data.obs_columns(obs_families):
        el = data.n[data.R[:, j] >= -1].sum()
        ob = data.n[data.R[:, j] >= 0].sum()
        eta[j] = ob / el if el > 0 else np.nan
    return eta


def simulate_joint(data: lcm.PatternData, params: dict, seed: int, chunk: int = 400_000):
    """Replicate observation AND reports for every edge: the design (eligibility of the modelled
    annotators, observation of the others) is the real one; z ~ w; o ~ Bernoulli(eta_jz) for
    eligible modelled entries; r ~ Bernoulli(theta_jz) if observed.  ``params['eta']`` may have
    2 columns (class-independent rates: both equal).  Returns (plain PatternData of the edges
    observed by >= 1 annotator, LIT observation statistics dict)."""
    rng = np.random.default_rng(seed)
    w = np.asarray(params["w"], float)
    w = w / w.sum()
    th = np.clip(np.asarray(params["theta"], float), 0, 1)
    eta = np.asarray(params["eta"], float)
    S = len(w)
    if eta.shape[1] != S:          # class-independent rates
        eta = np.repeat(eta[:, :1], S, axis=1)
    mod = np.isfinite(eta[:, 0])
    K = data.K
    design = np.where(mod[None, :], data.R >= -1, data.R >= 0)
    keep = design.any(1) & (data.n > 0)
    D = design[keep]
    cnt = np.rint(data.n[keep]).astype(np.int64)
    idx = np.repeat(np.arange(len(cnt)), cnt)
    etaT = np.where(mod[:, None], np.nan_to_num(eta), 1.0).T          # (S, K)
    pow3 = 3 ** np.arange(K, dtype=np.int64)
    lit = np.flatnonzero(mod)
    us, cs = [], []
    n_elig = np.zeros(K)
    n_seen = np.zeros(K)
    nobs_hist = np.zeros(len(NOBS_LIT_BINS))
    for s0 in range(0, len(idx), chunk):
        ii = idx[s0:s0 + chunk]
        z = rng.choice(S, size=len(ii), p=w)
        Di = D[ii]
        o = (rng.random((len(ii), K)) < etaT[z]) & Di
        rep = rng.random((len(ii), K)) < th.T[z]
        M = np.where(o, rep.astype(np.int8), np.int8(-1))
        anyo = o.any(1)
        u, c = np.unique((M[anyo].astype(np.int64) + 1) @ pow3, return_counts=True)
        us.append(u)
        cs.append(c)
        n_elig += Di.sum(0)
        n_seen += o.sum(0)
        el_any = Di[:, lit].any(1)
        nl = o[:, lit].sum(1)[el_any]
        for b, (lo, hi) in enumerate(NOBS_LIT_BINS):
            nobs_hist[b] += np.sum((nl >= lo) & (nl <= hi))
    stats = dict(obs_rate=np.where(mod, n_seen / np.maximum(n_elig, 1), np.nan),
                 nobs_lit_frac=nobs_hist / max(nobs_hist.sum(), 1))
    return _compress(us, cs, K, data), stats


def real_observation_stats(data: lcm.PatternData) -> dict:
    """The observation statistics of ``simulate_joint`` on the real data."""
    mod = np.zeros(data.K, bool)
    mod[data.obs_columns(("LIT",))] = True
    n = data.n
    el = (data.R >= -1) & mod[None, :]
    seen = (data.R >= 0) & mod[None, :]
    rate = np.where(mod, (seen * n[:, None]).sum(0) / np.maximum((el * n[:, None]).sum(0), 1), np.nan)
    el_any = el.any(1)
    nl = seen.sum(1)
    hist = np.array([n[el_any & (nl >= lo) & (nl <= hi)].sum() for lo, hi in NOBS_LIT_BINS])
    return dict(obs_rate=rate, nobs_lit_frac=hist / max(hist.sum(), 1))


# ----------------------------------------------------------------------------------------
# Stage: ppc
# ----------------------------------------------------------------------------------------
PPC_KEYS = ("nrep_frac", "rate", "lor_pairs", "anyrep_by_nobs")


def ppc_fits(u: str, data: lcm.PatternData) -> dict:
    """{label: (params, kind)} of the fits checked in universe u."""
    out = {}
    for m in MAR_FITS:
        f = read_fit(u, m)
        if f is not None:
            out[m] = (mar_params(f["params"]), "mar")
    for m in all_obs_fits():
        f = read_fit(u, m)
        if f is not None:
            out[m] = (obs_params(f["params"]), "obs")
    return out


def run_ppc(universes=PPC_UNIVERSES, n_rep=N_PPC, force=False) -> None:
    from r2 import fit_sensitivity as FS
    for u in universes:
        od = C.ensure_dir(MOD / "ppc" / u)
        data = load_elig(u)
        plain = data.marginal()
        pairs = FS.ppc_pairs(plain)
        real = FS.ppc_statistics(plain, pairs)
        lc = FS.lit_columns(plain)
        pairs_lit = FS.ppc_pairs(FS.subset_annotators(plain, lc))
        real_lit = FS.ppc_statistics(FS.subset_annotators(plain, lc), pairs_lit)
        robs = real_observation_stats(data)
        np.savez(od / "real.npz", **{k: real[k] for k in PPC_KEYS}, pairs=pairs,
                 **{f"lit_{k}": real_lit[k] for k in PPC_KEYS}, lit_pairs=pairs_lit, lit_cols=np.array(lc),
                 **{f"obs_{k}": v for k, v in robs.items()})
        fits = ppc_fits(u, data)
        jobs = [(lab, "conditional", p) for lab, (p, kind) in fits.items()]
        for lab, (p, kind) in fits.items():
            if lab in COND_FITS:
                continue        # eta of a conditional-likelihood fit shapes P(z | o); it is not a fitted observation rate
            if kind == "obs":
                jobs.append((lab, "joint", p))
            else:
                jobs.append((f"{lab}+ignorable_obs", "joint", dict(p, eta=ignorable_eta(data))))
        for lab, ptype, p in jobs:
            out = od / f"{ptype}__{lab}.npz"
            if out.exists() and not force:
                continue
            t = time.time()
            reps = {k: [] for k in PPC_KEYS}
            reps.update({f"lit_{k}": [] for k in PPC_KEYS})
            reps.update({"obs_obs_rate": [], "obs_nobs_lit_frac": []})
            for r in range(n_rep):
                sd = derived_seed("obs_ppc", u, ptype, lab, r)
                if ptype == "conditional":
                    sim = simulate_conditional(data, p, sd)
                    ost = None
                else:
                    sim, ost = simulate_joint(data, p, sd)
                st = FS.ppc_statistics(sim, pairs)
                st_lit = FS.ppc_statistics(FS.subset_annotators(sim, lc), pairs_lit)
                for k in PPC_KEYS:
                    reps[k].append(st[k])
                    reps[f"lit_{k}"].append(st_lit[k])
                if ost is not None:
                    reps["obs_obs_rate"].append(ost["obs_rate"])
                    reps["obs_nobs_lit_frac"].append(ost["nobs_lit_frac"])
            np.savez(out, **{k: np.stack(v) for k, v in reps.items() if v})
            log(f"[ppc] {u} {ptype} {lab}: {n_rep} replicates ({time.time() - t:.0f}s)")


def ppc_table(universes=PPC_UNIVERSES) -> pd.DataFrame:
    from r2 import fit_sensitivity as FS
    rows = []
    for u in universes:
        od = MOD / "ppc" / u
        if not (od / "real.npz").exists():
            continue
        data = load_elig(u)
        real = dict(np.load(od / "real.npz"))
        pairs = real["pairs"]
        fam = np.array(data.family)
        within = fam[pairs[:, 0]] == fam[pairs[:, 1]]
        for f in sorted(od.glob("*__*.npz")):
            ptype, lab = f.stem.split("__", 1)
            rep = dict(np.load(f))
            base_model = lab.split("+")[0]
            status = "prespecified" if (ptype == "conditional" and base_model in MAR_FITS) else "post_hoc"
            obs_model = ("informative_conditional_likelihood" if base_model in COND_FITS else
                         "informative_tf_weighted" if base_model in TFW_FITS else
                         "informative" if base_model in OBS_FITS else
                         "ignorable_rates" if "+ignorable" in lab else "MAR_fixed_masks")

            def add(stat, stratum, tobs, trep):
                trep = np.asarray(trep, float)
                m, s = float(np.mean(trep)), float(np.std(trep, ddof=1))
                rows.append(dict(universe=u, ppc_type=ptype, fit=lab, model=base_model,
                                 observation_model=obs_model,
                                 statistic=stat, stratum=stratum, kind="statistic", observed=float(tobs),
                                 rep_mean=m, rep_sd=s, z=(float(tobs) - m) / s if s > 1e-12 else np.nan,
                                 ppp_upper=float(np.mean(trep >= tobs)), n_rep=len(trep), analysis_status=status))

            def summ(stat, z):
                z = np.asarray(z, float)
                z = z[np.isfinite(z)]
                rows.append(dict(universe=u, ppc_type=ptype, fit=lab, model=base_model,
                                 observation_model=obs_model,
                                 statistic=stat, stratum="summary", kind="summary_abs_z",
                                 z=float(np.mean(np.abs(z))) if len(z) else np.nan,
                                 max_abs_z=float(np.max(np.abs(z))) if len(z) else np.nan,
                                 ppp_upper=float(np.mean(np.abs(z) > 2)) if len(z) else np.nan,
                                 n_elements=int(len(z)), n_rep=int(rep["rate"].shape[0]), analysis_status=status))

            def zs(o_, r_):
                sd_ = r_.std(0, ddof=1)
                return np.where(sd_ > 1e-12, (o_ - r_.mean(0)) / np.where(sd_ > 1e-12, sd_, 1.0), np.nan)

            lab_bins = [f"{lo}" if lo == hi else (f"{lo}-{hi}" if hi < 99 else f">={lo}") for lo, hi in FS.NOBS_BINS]
            for b in range(NREP_CAP + 1):
                add("n_reporting_frac", f"{b}" if b < NREP_CAP else f">={NREP_CAP}", real["nrep_frac"][b],
                    rep["nrep_frac"][:, b])
            for b, lb in enumerate(lab_bins):
                add("anyreport_frac_by_n_observing", lb, real["anyrep_by_nobs"][b], rep["anyrep_by_nobs"][:, b])
            Lr, Ls = real["lor_pairs"], rep["lor_pairs"]
            add("lor_mean", "within_family", Lr[within].mean(), Ls[:, within].mean(1))
            add("lor_mean", "between_family", Lr[~within].mean(), Ls[:, ~within].mean(1))
            for j, a in enumerate(data.annotators):
                add("positive_rate", a, real["rate"][j], rep["rate"][:, j])
            for b in range(NREP_CAP + 1):
                add("litblock_n_reporting_frac", f"{b}" if b < NREP_CAP else f">={NREP_CAP}",
                    real["lit_nrep_frac"][b], rep["lit_nrep_frac"][:, b])
            for b, lb in enumerate(lab_bins):
                add("litblock_anyreport_frac_by_n_observing", lb, real["lit_anyrep_by_nobs"][b],
                    rep["lit_anyrep_by_nobs"][:, b])
            add("litblock_lor_mean", "LIT-LIT", real["lit_lor_pairs"].mean(), rep["lit_lor_pairs"].mean(1))
            summ("n_reporting_frac", zs(real["nrep_frac"], rep["nrep_frac"]))
            summ("anyreport_frac_by_n_observing", zs(real["anyrep_by_nobs"], rep["anyrep_by_nobs"]))
            summ("positive_rate", zs(real["rate"], rep["rate"]))
            summ("lor_pair_within", zs(Lr, Ls)[within])
            summ("lor_pair_between", zs(Lr, Ls)[~within])
            summ("litblock_n_reporting_frac", zs(real["lit_nrep_frac"], rep["lit_nrep_frac"]))
            summ("litblock_anyreport_frac_by_n_observing", zs(real["lit_anyrep_by_nobs"], rep["lit_anyrep_by_nobs"]))
            summ("litblock_lor_pair", zs(real["lit_lor_pairs"], rep["lit_lor_pairs"]))
            if "obs_obs_rate" in rep:
                mod = np.isfinite(real["obs_obs_rate"])
                for j in np.flatnonzero(mod):
                    add("lit_observation_rate", data.annotators[j], real["obs_obs_rate"][j], rep["obs_obs_rate"][:, j])
                for b, (lo, hi) in enumerate(NOBS_LIT_BINS):
                    add("n_observing_lit_frac", f"{lo}" if lo == hi else (f"{lo}-{hi}" if hi < 99 else f">={lo}"),
                        real["obs_nobs_lit_frac"][b], rep["obs_nobs_lit_frac"][:, b])
                summ("lit_observation_rate", zs(real["obs_obs_rate"][mod], rep["obs_obs_rate"][:, mod]))
                summ("n_observing_lit_frac", zs(real["obs_nobs_lit_frac"], rep["obs_nobs_lit_frac"]))
    df = pd.DataFrame(rows)
    if len(df):
        df["definition"] = np.where(
            df["ppc_type"] == "conditional",
            "real observation masks kept; z ~ P(z | o) (= w for MAR fits); reports ~ theta; statistics over observed edges",
            "LIT observation simulated for eligible entries (o ~ eta_z; MAR fits: class-independent empirical rates), "
            "MOTIF/COEXP masks fixed; z ~ w; statistics over simulated observed edges")
    return df


# ----------------------------------------------------------------------------------------
# Stage: cv (TF-grouped 5-fold)
# ----------------------------------------------------------------------------------------
def tf_fold_index(u: str):
    """Per-edge fold label (TF folds of ``fit_sensitivity``) and the TF list."""
    from r2 import fit_sensitivity as FS
    tfi, tfs = E.universe_edge_tfs(u)
    fold_of_tf = FS.tf_folds(len(tfs))
    ref = MOD.parent / "fit_sensitivity" / "cv" / u / "tf_folds.csv"
    if ref.exists():
        r = pd.read_csv(ref)
        if list(r["tf"]) != list(tfs) or not np.array_equal(r["fold"].to_numpy(), fold_of_tf):
            raise AssertionError(f"[{u}] TF folds differ from fit_sensitivity")
    return fold_of_tf[tfi], tfs, fold_of_tf


def _n_elig_any(data: lcm.PatternData, n: np.ndarray) -> float:
    cols = data.obs_columns(("LIT",))
    return float(n[(data.R[:, cols] >= -1).any(1)].sum())


def cv_fit(train: lcm.PatternData, model: str, universe: str, n_random: int = N_RESTARTS_CV,
           seed: int = SEED, prior: dict | None = None, fit_id: str | None = None,
           warm_optima: list | None = None) -> dict:
    """MAP of an informative-observation model on the training edges of one CV fold
    (``prior``: e.g. the ``obs_weight`` of the ``__tfw`` fits; ``fit_id``: the full-data fit
    whose optima give the warm starts, default ``model``).

    Start points: (i) one warm start from every distinct optimum of the full-data fit
    (``optima_params.npz``, unconstrained vectors; the parameter layout does not depend on the
    edge set; ``warm_optima``: only these optimum indices), so that the basins known from the
    multi-start full-data fit are explored on the training data; (ii) ``n_random`` random starts (the PRNG keys / start points of
    ``lcm.fit_map(train, model, n_restarts=n_random, seed)``) as a check that the warm starts do
    not miss a better training optimum.  L-BFGS-B with the options of ``lcm.fit_map``.  The fit
    is the end point with the highest TRAINING log posterior (no test data enters the choice).
    Returns dict(best (constrained sample), log_post, loglik, starts (records), class_names,
    n_within_1e3, best_start, random_minus_warm (best random - best warm log posterior),
    gap_to_next)."""
    import jax
    import jax.numpy as jnp
    from scipy.optimize import minimize
    from r2 import fit_primary as FP
    pt = FP._potential(train, model, prior, seed=seed, init_radius=FP.INIT_RADIUS)
    starts = []
    f = fit_dir(universe, fit_id or model) / "optima_params.npz"
    if f.exists():
        with np.load(f) as z:
            for k in sorted(z.files, key=lambda s: int(s[1:])):
                if warm_optima is None or int(k[1:]) in warm_optima:
                    starts.append((f"warm_opt{k[1:]}", np.asarray(z[k], dtype=float)))
    keys = jax.random.split(jax.random.PRNGKey(seed), max(n_random, 1))
    for r in range(n_random):
        starts.append((f"random{r}", np.asarray(jax.random.uniform(keys[r], pt["flat0"].shape,
                                                                   minval=-FP.INIT_RADIUS, maxval=FP.INIT_RADIUS))))
    recs, xs = [], []
    for kind, x0 in starts:
        t = time.time()
        res = minimize(lambda x: float(pt["pot"](jnp.asarray(x))), x0,
                       jac=lambda x: np.asarray(pt["grad"](jnp.asarray(x))),
                       method="L-BFGS-B", options=dict(FP.LBFGS_OPTIONS))
        g = float(np.max(np.abs(np.asarray(pt["grad"](jnp.asarray(res.x))))))
        recs.append(dict(start=kind, log_post=-float(res.fun), n_iter=int(res.nit), grad_norm=g,
                         lbfgs_success=bool(res.success), runtime_s=time.time() - t))
        xs.append(np.asarray(res.x))
    lp = np.array([r["log_post"] for r in recs])
    ib = int(np.argmax(lp))
    best = FP._constrained(pt, xs[ib])
    warm = np.array([r["start"].startswith("warm") for r in recs])
    below = lp[lp < lp.max() - 1e-3]
    return dict(best=best, log_post=float(lp[ib]), loglik=lcm.heldout_loglik(obs_params(best), train),
                starts=recs, class_names=list(pt["model"].class_names),
                n_within_1e3=int(np.sum(lp >= lp.max() - 1e-3)), best_start=recs[ib]["start"],
                random_minus_warm=(float(lp[~warm].max() - lp[warm].max()) if warm.any() and (~warm).any() else np.nan),
                gap_to_next=float(lp.max() - below.max()) if len(below) else np.nan)


def run_cv(universes=CV_UNIVERSES, folds=None, n_restarts=None, force=False) -> None:
    from r2 import fit_sensitivity as FS
    for u in universes:
        cs = CV_STARTS.get(u, dict(warm="all", n_random=N_RESTARTS_CV))
        n_restarts_u = cs["n_random"] if n_restarts is None else n_restarts
        data = load_elig(u)
        ep = edge_pattern_elig(u)
        edge_fold, tfs, fold_of_tf = tf_fold_index(u)
        assert len(edge_fold) == len(ep)
        counts = FS.fold_pattern_counts(ep, edge_fold, data.P, N_FOLDS)
        od = C.ensure_dir(MOD / "cv" / u)
        for f in (range(N_FOLDS) if folds is None else folds):
            n_test = counts[f]
            n_train = counts.sum(0) - n_test
            tr, te = n_train > 0, n_test > 0
            train = lcm.PatternData(data.R[tr], n_train[tr], list(data.family), list(data.annotators), True)
            test = lcm.PatternData(data.R[te], n_test[te], list(data.family), list(data.annotators), True)
            n_test_obs = float(test.n[(test.R >= 0).any(1)].sum())
            common = dict(universe=u, fold=f, n_train_edges=float(train.n.sum()), n_test_edges=float(test.n.sum()),
                          n_test_observed=n_test_obs, n_test_lit_eligible=_n_elig_any(test, test.n),
                          n_test_tfs=int((fold_of_tf == f).sum()))
            eta_ign = ignorable_eta(train)
            tfw = {"obs_weight": tf_replication_weights(u, data)}     # full-universe design weights
            for fid in all_obs_fits():
                m = fid.split("__")[0]
                if read_fit(u, fid) is None:
                    continue
                out = od / f"fold{f}__{fid}.json"
                if out.exists() and not force:
                    continue
                t = time.time()
                n_restarts = n_restarts_u
                warm = None if cs["warm"] == "all" else [0] + [k for k in (alt_optimum(u, fid)[0],) if k is not None]
                fit = cv_fit(train, m, u, n_random=n_restarts, seed=SEED,
                             prior=tfw if fid in TFW_FITS else dict(COND_PRIOR) if fid in COND_FITS else None,
                             fit_id=fid, warm_optima=warm)
                p = obs_params(fit["best"])
                eta = np.asarray(p["eta"])
                lit = np.isfinite(eta[:, 0])
                rec = dict(common, fit=fid, model=m, source="this_module", n_restarts=len(fit["starts"]),
                           n_random_restarts=n_restarts, n_warm_starts=len(fit["starts"]) - n_restarts,
                           n_within_1e3=fit["n_within_1e3"], best_start=fit["best_start"],
                           best_random_minus_best_warm_log_post=fit["random_minus_warm"],
                           gap_to_next=fit["gap_to_next"], starts=fit["starts"],
                           train_log_post=fit["log_post"], train_joint_loglik=float(fit["loglik"]),
                           # joint / observation log-likelihood of the GENERATIVE model at the fitted
                           # parameters (for the __tfw fits: the composite weights are dropped);
                           # conditional log p(r | o) with P(z | o) as the fit defines it (weighted)
                           test_joint_loglik=lcm.heldout_loglik(_unweighted(p), test),
                           test_cond_loglik=lcm.conditional_loglik(p, test),
                           test_obs_loglik=lcm.observation_loglik(_unweighted(p), test),
                           test_cond_loglik_generative=lcm.conditional_loglik(_unweighted(p), test),
                           joint_definition=("informative-observation model" if fid in OBS_FITS else
                                             "generative (unweighted) model at the composite-likelihood estimate"
                                             if fid in TFW_FITS else
                                             "generative model at the conditional-likelihood estimate (eta not "
                                             "fitted to the observation indicators)"),
                           w=np.asarray(p["w"]), theta=np.asarray(p["theta"]), eta=eta,
                           n_lit_eta_T_gt_NonT=int(np.sum(eta[lit, 0] > eta[lit, 1])),
                           class_names=fit["class_names"], runtime_s=time.time() - t)
                C.write_json(_jsonable(rec), out)
                log(f"[cv] {u} fold{f} {fid}: cond/obs-edge {rec['test_cond_loglik'] / n_test_obs:.6f} "
                    f"joint/edge {rec['test_joint_loglik'] / rec['n_test_edges']:.6f} best start {fit['best_start']} "
                    f"({rec['runtime_s']:.0f}s)")
            for m in MAR_FITS:
                out = od / f"fold{f}__{m}.json"
                if out.exists() and not force:
                    continue
                src = FIT_SENS_CV / u / f"fold{f}__{m}.json"
                if not src.exists():
                    raise FileNotFoundError(f"{src}: run fit_sensitivity.py cv first")
                j = json.loads(src.read_text())
                p = {"w": np.asarray(j["w"]), "theta": np.asarray(j["theta"])}
                cond = lcm.conditional_loglik(p, test)
                if not np.isclose(cond, j["test_loglik"], rtol=1e-9, atol=1e-6):
                    raise AssertionError(f"[{u} fold{f} {m}] conditional log-lik {cond} != fit_sensitivity {j['test_loglik']}")
                pi = dict(p, eta=eta_ign)
                # MAR + class-independent observation rates: log p(o) does not depend on z
                obs_ll = lcm.observation_loglik({"w": np.array([1.0]), "theta": p["theta"][:, :1],
                                                 "eta": eta_ign[:, :1]}, test)
                rec = dict(common, fit=m, model=m, source=str(src.relative_to(C.SP)), n_restarts=j["n_restarts"],
                           n_within_1e3=j.get("n_within_1e3"), train_log_post=j["train_log_post"],
                           test_cond_loglik=cond, test_obs_loglik=obs_ll, test_joint_loglik=cond + obs_ll,
                           joint_definition="MAR model x class-independent LIT observation rates (training-edge MLE)",
                           w=p["w"], theta=p["theta"], eta_ignorable=pi["eta"])
                C.write_json(_jsonable(rec), out)
                log(f"[cv] {u} fold{f} {m}: cond/obs-edge {cond / n_test_obs:.6f}")


def cv_table(universes=CV_UNIVERSES) -> pd.DataFrame:
    rows = []
    for u in universes:
        d = MOD / "cv" / u
        recs = [json.loads(p.read_text()) for p in sorted(d.glob("fold*__*.json"))]
        if not recs:
            continue
        df = pd.DataFrame(recs)
        df["cond_per_obs_edge"] = df["test_cond_loglik"] / df["n_test_observed"]
        df["joint_per_edge"] = df["test_joint_loglik"] / df["n_test_edges"]
        df["obs_per_edge"] = df["test_obs_loglik"] / df["n_test_edges"]
        for metric in ("cond_per_obs_edge", "joint_per_edge", "obs_per_edge"):
            ref = df[df["fit"] == "family_hb"].set_index("fold")[metric]
            for fit, g in df.groupby("fit"):
                g = g.set_index("fold").sort_index()
                common = g.index.intersection(ref.index)
                diff = g.loc[common, metric] - ref.loc[common]
                cp = lcm.MAR_COUNTERPART.get(fit.split("__")[0])
                cref = df[df["fit"] == cp].set_index("fold")[metric] if cp else None
                cdiff = (g.loc[g.index.intersection(cref.index), metric] - cref.loc[g.index.intersection(cref.index)]
                         if cref is not None and len(cref) else pd.Series(dtype=float))
                tot = (g["test_cond_loglik"] if metric.startswith("cond") else
                       g["test_joint_loglik"] if metric.startswith("joint") else g["test_obs_loglik"]).sum()
                rows.append(dict(
                    universe=u, fit=fit, metric=metric, n_folds=len(g), mean=float(g[metric].mean()),
                    se=float(g[metric].std(ddof=1) / np.sqrt(len(g))) if len(g) > 1 else np.nan,
                    total_loglik=float(tot),
                    diff_vs_family_hb=float(diff.mean()) if len(common) else np.nan,
                    diff_se=float(diff.std(ddof=1) / np.sqrt(len(diff))) if len(diff) > 1 else np.nan,
                    n_folds_better_than_family_hb=int((diff > 0).sum()) if fit != "family_hb" else np.nan,
                    mar_counterpart=cp if cp else "",
                    diff_vs_mar_counterpart=float(cdiff.mean()) if len(cdiff) else np.nan,
                    diff_vs_mar_counterpart_se=(float(cdiff.std(ddof=1) / np.sqrt(len(cdiff))) if len(cdiff) > 1
                                                else np.nan),
                    n_folds_better_than_mar_counterpart=int((cdiff > 0).sum()) if len(cdiff) else np.nan,
                    n_restarts=int(g["n_restarts"].iloc[0]),
                    n_warm_starts=(int(g["n_warm_starts"].iloc[0]) if "n_warm_starts" in g and
                                   pd.notna(g["n_warm_starts"].iloc[0]) else 0),
                    min_restarts_at_best=int(g["n_within_1e3"].min()) if g["n_within_1e3"].notna().all() else np.nan,
                    n_folds_random_start_beats_warm=(
                        int((g["best_random_minus_best_warm_log_post"] > 1e-3).sum())
                        if "best_random_minus_best_warm_log_post" in g and
                        g["best_random_minus_best_warm_log_post"].notna().any() else np.nan),
                    n_folds_majority_eta_T_gt_NonT=(
                        int((g["n_lit_eta_T_gt_NonT"] > 9).sum())
                        if "n_lit_eta_T_gt_NonT" in g and g["n_lit_eta_T_gt_NonT"].notna().any() else np.nan),
                    joint_definition=(g["joint_definition"].iloc[0] if "joint_definition" in g and
                                      isinstance(g["joint_definition"].iloc[0], str) else "informative-observation model"),
                    analysis_status=("prespecified" if (fit in MAR_FITS and metric == "cond_per_obs_edge") else "post_hoc"),
                    per_fold=" ".join(f"{v:.6g}" for v in g[metric].to_numpy())))
    out = pd.DataFrame(rows)
    if len(out):
        out["definition"] = out["metric"].map({
            "cond_per_obs_edge": "held-out log p(r | o) per test edge observed by >= 1 annotator (MAR: marginal likelihood)",
            "joint_per_edge": "held-out log p(o, r) per test edge (MAR: x class-independent observation rates)",
            "obs_per_edge": "held-out log p(o) per test edge"})
    return out


# ----------------------------------------------------------------------------------------
# Stage: heldout (post hoc)
# ----------------------------------------------------------------------------------------
def status_score(data: lcm.PatternData) -> np.ndarray:
    """Label-free 3-level evidence status: 0 unobserved < 1 observed negative < 2 reported."""
    obs = (data.R >= 0).any(1)
    rep = (data.R == 1).any(1)
    return np.where(rep, 2.0, np.where(obs, 1.0, 0.0))


def heldout_scores(u: str, data: lcm.PatternData, base: np.ndarray) -> tuple[dict, dict]:
    """Scores over the eligibility patterns (and edge-level random), with their analysis status."""
    from r2 import heldout_eval as HE
    S, status = {}, {}
    plain = load_plain(u)
    for m in all_obs_fits():
        f = read_fit(u, m)
        if f is not None:
            S[m] = HE.Score(m, "lcm_obs", "pattern", f["cred"], True)
            status[m] = "post_hoc"
    f = read_fit(u, NUTS_FIT)
    if f is not None:
        S[NUTS_FIT] = HE.Score(NUTS_FIT, "lcm_obs", "pattern", np.load(fit_dir(u, NUTS_FIT) / "cred_mean.npy"), True)
        status[NUTS_FIT] = "post_hoc"
    # alternative MAP optimum of each informative-observation fit (mode-structure diagnostic):
    # the label-swapped optimum of the generative fits, the second optimum of the __tfw fits
    for m in all_obs_fits():
        k, role = alt_optimum(u, m)
        if k is not None:
            x2 = optimum_params(u, m, data, k)
            name = m + alt_suffix(role)
            S[name] = HE.Score(name, "lcm_obs", "pattern", lcm.credibility_at(x2, data), True)
            status[name] = "post_hoc"
    for m in mar_comparators(u):
        fd = FITS / u / VARIANT / m
        if (fd / "cred.npy").exists():
            c = np.load(fd / ("cred_mean.npy" if m.endswith("_nuts") else "cred.npy"))
            S[m] = HE.Score(m, "lcm", "pattern", np.asarray(c)[base], True)
            status[m] = "prespecified"
    bs = lcm.baseline_scores(plain)
    S["fisher"] = HE.Score("fisher", "pattern_baseline", "pattern", np.asarray(bs["fisher"])[base])
    status["fisher"] = "prespecified"
    # post hoc status-mechanism scores, defined as in heldout_eval (status3; Family-HB with every
    # unobserved edge ranked below every observed edge)
    st = status_score(data)
    S["status3"] = HE.Score("status3", "status_mechanism", "pattern", st)
    status["status3"] = "post_hoc"
    if "family_hb" in S:
        S["family_hb_unobserved_last"] = HE.Score("family_hb_unobserved_last", "status_mechanism", "pattern",
                                                  np.where(st > 0, S["family_hb"].values, -1.0), True)
        status["family_hb_unobserved_last"] = "post_hoc"
        if hasattr(HE, "status_mechanism_scores"):
            ref = HE.status_mechanism_scores(data, S["family_hb"].values)
            assert np.array_equal(ref["status3"], S["status3"].values)
            assert np.array_equal(ref["family_hb_unobserved_last"], S["family_hb_unobserved_last"].values)
    U = HE.load_universe(u)
    if u == "global":
        S["random"] = HE.Score("random", "random", "edge",
                               HE._stream("random", "global").integers(0, HE.RANDOM_LEVELS, U.n_edges).astype(np.float32))
    else:
        S["random"] = HE.Score("random", "random", "edge", HE.method_scores(U)["random"])
    status["random"] = "prespecified"
    return S, status


def fit_prior(meta: dict) -> dict | None:
    """Prior overrides recorded by ``fit_primary.run_map`` (obs_weight back to an array)."""
    ov = (meta.get("prior") or {}).get("overrides") or {}
    if not ov:
        return None
    return {k: (np.asarray(v, dtype=float) if k == "obs_weight" else v) for k, v in ov.items()}


def alt_optimum(u: str, fit_id: str) -> tuple[int | None, str]:
    """(index in ``optima.csv``, role) of the alternative MAP optimum reported next to the best
    one.  Generative fits: the best optimum with the opposite labelling of the two observation
    classes (credibility negatively rank-correlated with the best optimum; role
    ``label_swapped``, score suffix ``_swap``).  ``__tfw`` / ``__cond`` fits: the second optimum (role
    ``second_optimum``, suffix ``_opt2``).  Only optima whose Newton polish reached a positive-
    definite Hessian (finite Laplace mass) qualify.  (None, '') if there is none."""
    f = fit_dir(u, fit_id) / "optima.csv"
    if not f.exists():
        return None, ""
    opt = pd.read_csv(f)
    ok = np.isfinite(opt.get("laplace_log_mass", pd.Series(np.nan, index=opt.index)).to_numpy(dtype=float))
    if fit_id in TFW_FITS + COND_FITS:
        return (1, "second_optimum") if len(opt) > 1 and ok[1] else (None, "")
    for k in range(1, len(opt)):
        if ok[k] and opt["spearman_w_vs_best"].iloc[k] < 0:
            return k, "label_swapped"
    return None, ""


def alt_suffix(role: str) -> str:
    return {"label_swapped": "_swap", "second_optimum": "_opt2"}[role]


def optimum_params(u: str, fit_id: str, data: lcm.PatternData, k: int) -> dict | None:
    """Constrained (w, theta, eta[, obs_weight]) at the k-th distinct MAP optimum
    (``optima_params.npz``, Newton-polished) of a stored informative-observation fit."""
    from r2 import fit_primary as FP
    import jax.numpy as jnp
    d = fit_dir(u, fit_id)
    z = np.load(d / "optima_params.npz")
    if f"x{k}" not in z.files:
        return None
    meta = json.loads((d / "meta.json").read_text())
    pt = FP._potential(data, meta["model"], fit_prior(meta))
    s = {k_: np.asarray(v) for k_, v in pt["init"].postprocess_fn(pt["unravel"](jnp.asarray(z[f"x{k}"]))).items()}
    return obs_params(s)


def score_provenance(u: str, names) -> dict:
    """md5 / modification time / inference settings of the stored credibility file behind every
    fit-based score (the MAR fits are owned by ``fit_primary`` and may be re-run by it)."""
    out = {}
    for m in names:
        d = fit_dir(u, m) if m.startswith(OBS_FITS) else FITS / u / VARIANT / m
        f = d / ("cred_mean.npy" if m.endswith("_nuts") else "cred.npy")
        if not f.exists():
            continue
        meta = json.loads((d / "meta.json").read_text()) if (d / "meta.json").exists() else {}
        out[m] = dict(file=str(f.relative_to(C.SP)), md5=file_md5(f),
                      mtime=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(f.stat().st_mtime)),
                      inference=meta.get("inference"), n_restarts=meta.get("n_restarts"),
                      num_warmup=meta.get("num_warmup"), num_samples=meta.get("num_samples"))
    return out


def run_heldout(universes=UNIVERSES, refs=HELDOUT_REFS) -> None:
    from r2 import heldout_eval as HE
    est_rows, pair_rows, status_rows = [], [], []
    prov = {}
    for u in universes:
        data, base = load_elig(u, with_base=True)
        ep = edge_pattern_elig(u)
        S, status = heldout_scores(u, data, base)
        prov[u] = score_provenance(u, [k for k in S if k in all_obs_fits() + (NUTS_FIT,) + MAR_FITS + ("family_hb_nuts",)])
        obs_any = (data.R >= 0).any(1)
        rep_any = (data.R == 1).any(1)
        lit = data.obs_columns(("LIT",))
        n_elig_unobs = (data.R[:, lit] == -1).sum(1)
        creds = {k: S[k].values for k in ("family_hb", "em") + all_obs_fits() + (NUTS_FIT,) if k in S}
        for ref in refs:
            es = HE.load_evalset(u, ref)
            if es.n == 0 or es.y.sum() == 0 or es.y.all():
                continue
            log(f"[heldout] {u}/{ref}: n={es.n:,} pos={int(es.y.sum()):,}")
            rows, reps = HE.evaluate_scores(es, S, ep)       # tag '' = the resamples of heldout_eval
            for r in rows:
                r["analysis_status"] = status[r["score"]]
            est_rows.extend(rows)
            for a, others in PAIRED_VS.items():
                if a not in reps:
                    continue
                sub = {k: reps[k] for k in (a,) + others if k in reps}
                for r in HE.paired_rows(es, sub, ref_score=a):
                    r["analysis_status"] = "post_hoc"
                    r["multiplicity"] = ("p_holm: Holm over the %d comparisons of score_a within universe x reference "
                                         "x metric (this table; post hoc)" % (len(sub) - 1))
                    r.pop("analysis_role", None)
                    pair_rows.append(r)
            # evidence status O/E (+ unobserved split by eligible-unobserving LIT annotators)
            pp = ep[es.edge]
            st = np.where(rep_any[pp], 2, np.where(obs_any[pp], 1, 0))
            T = es.n_tf
            N_t = np.bincount(es.tf_code, minlength=T).astype(float)
            P_t = np.bincount(es.tf_code, weights=es.y.astype(float), minlength=T)
            base_e = (P_t / N_t)[es.tf_code]
            M = HE.tf_bootstrap_weights(T, HE.N_BOOT, HE._stream("boot_status", u, ref))
            classes = [("unobserved", st == 0, "prespecified"), ("observed_neg", st == 1, "prespecified"),
                       ("reported", st == 2, "prespecified")]
            ne = n_elig_unobs[pp]
            for lo, hi in UNOBS_ELIG_BINS:
                lab = f"{lo}" if lo == hi else (f"{lo}-{hi}" if hi < 99 else f">={lo}")
                classes.append((f"unobserved__n_eligible_unobserving_lit_{lab}", (st == 0) & (ne >= lo) & (ne <= hi),
                                "post_hoc"))
            for name, m, ast in classes:
                if not m.any():
                    continue
                o_t = np.bincount(es.tf_code[m], weights=es.y[m].astype(float), minlength=T)
                e_t = np.bincount(es.tf_code[m], weights=base_e[m], minlength=T)
                with np.errstate(invalid="ignore", divide="ignore"):
                    b = (M @ o_t) / (M @ e_t)
                lo_, hi_ = HE.percentile_ci(b)
                row = dict(universe=u, reference=ref, status=name, n_eval=int(m.sum()),
                           n_tfs=int(len(np.unique(es.tf_code[m]))), n_pos=int(es.y[m].sum()),
                           pos_rate=float(es.y[m].mean()), exp_tf_matched_rate=float(base_e[m].mean()),
                           oe_tf_matched=float(o_t.sum() / e_t.sum()) if e_t.sum() > 0 else np.nan,
                           oe_lo=lo_, oe_hi=hi_, low_support=HE.support_flag(es), analysis_status=ast)
                for k, c in creds.items():
                    row[f"mean_cred_{k}"] = float(np.mean(c[pp[m]]))
                status_rows.append(row)
        log(f"[heldout] {u} done")
    est = pd.DataFrame(est_rows)
    est["test_perm_p"] = ("one-sided within-TF permutation test of AP (B=%d); unit TF; Holm over the scores of this "
                          "table within universe x reference (perm_p_holm)" % HE.N_NULL)
    est["ci"] = "TF-cluster percentile bootstrap (B=%d), the resamples of heldout_eval" % HE.N_BOOT
    est.to_csv(TABLES / "obs_heldout.csv", index=False, float_format="%.6g")
    pdf = pd.DataFrame(pair_rows)
    pdf.to_csv(TABLES / "obs_heldout_paired.csv", index=False, float_format="%.6g")
    sdf = pd.DataFrame(status_rows)
    sdf["ci"] = "TF-cluster percentile bootstrap (B=%d) of O/E (TF-matched expectation)" % HE.N_BOOT
    sdf.to_csv(TABLES / "obs_heldout_status.csv", index=False, float_format="%.6g")
    C.write_json(dict(written=time.strftime("%Y-%m-%d %H:%M:%S"), universes=list(universes), references=list(refs),
                      scores=prov), MOD / "heldout_provenance.json")
    log("[heldout] wrote obs_heldout.csv, obs_heldout_paired.csv, obs_heldout_status.csv")


# ----------------------------------------------------------------------------------------
# Stage: tables (fit summary, parameters)
# ----------------------------------------------------------------------------------------
def _logit(p: float) -> float:
    p = min(max(float(p), 1e-12), 1 - 1e-12)
    return float(np.log(p) - np.log1p(-p))


def _expit(x: float) -> float:
    return float(1.0 / (1.0 + np.exp(-x)))


def _bfdr_n(cred, n, q):
    from r2 import fit_primary as FP
    return float(n[FP.bfdr_set(cred, n, q)].sum())


def replication_diagnostics(u: str, data: lcm.PatternData) -> dict:
    """Label-free description of the LIT observation process of a universe: eligibility is TF-level
    and observation gene-level, so every gene's node indicator for annotator j is repeated over
    the TFs eligible for j.  Returns the mean number of eligible TFs per LIT annotator (the
    replication factor), eligible entries, distinct gene-level node profiles over the LIT
    annotators, and the edge-weighted Spearman correlation between the number of observing LIT
    annotators and the number of eligible ones (how much of the observation count is TF design)."""
    from r2 import fit_primary as FP
    m = json.loads(E._elig_paths(u, VARIANT)["meta"].read_text())
    per = pd.DataFrame(m["per_annotator"])
    per = per[per["modelled"]]
    bank = E.bank_for_variant(VARIANT)
    if u == "global":
        genes = (E.EV_DIR / "global" / "genes.txt").read_text().split()
    else:
        genes = E.tissue_universe(u)[1]
    lit = [j for j, f in enumerate(bank.family) if f == "LIT"]
    gi = bank.target_index(genes)
    prof = np.zeros((len(genes), len(lit)), bool)
    ok = gi >= 0
    for c, j in enumerate(lit):
        prof[ok, c] = bank.node_mask[j, gi[ok]]
    cols = data.obs_columns(("LIT",))
    n_obs_lit = (data.R[:, cols] >= 0).sum(1)
    n_el_lit = (data.R[:, cols] >= -1).sum(1)
    return dict(replication_mean_eligible_tfs_per_lit_annotator=float(per["n_tfs_eligible"].mean()),
                n_lit_eligible_entries=float(per["n_eligible"].sum()),
                n_lit_observed_entries=float(per["n_observed"].sum()),
                n_genes=len(genes), n_distinct_gene_lit_node_profiles=int(len(np.unique(prof, axis=0))),
                spearman_n_observing_vs_n_eligible_lit=FP.weighted_spearman(n_obs_lit, n_el_lit, data.n))


def universe_edge_targets(universe: str) -> tuple[np.ndarray, list]:
    """Target-gene index of every stored edge of a universe (``edge_pattern`` order; the
    companion of ``evidence.universe_edge_tfs``) and the gene list."""
    if universe == "global":
        d = E.EV_DIR / "global"
        tfs = (d / "tfs.txt").read_text().split()
        genes = (d / "genes.txt").read_text().split()
        return np.tile(np.arange(len(genes), dtype=np.int64), len(tfs)), genes
    tfs, genes = E.tissue_universe(universe)
    gi = {g: i for i, g in enumerate(genes)}
    allg = np.arange(len(genes), dtype=np.int64)
    return np.concatenate([np.delete(allg, gi[t]) if t in gi else allg for t in tfs]), genes


def between_group_share(values: np.ndarray, group: np.ndarray) -> float:
    """Share of the variance of ``values`` that lies between the groups (one-way ANOVA
    R^2 = SS_between / SS_total; 1 = the value is a function of the group)."""
    v = np.asarray(values, dtype=np.float64)
    m = v.mean()
    sst = float(np.sum((v - m) ** 2))
    if sst <= 0:
        return float("nan")
    cnt = np.bincount(group)
    gm = np.bincount(group, weights=v) / np.maximum(cnt, 1)
    return float(np.sum(cnt * (gm - m) ** 2) / sst)


def gene_level_diagnostics(universe: str, cred: np.ndarray, ep: np.ndarray, groups: dict | None = None) -> dict:
    """Label-free check of WHAT the latent class tracks: share of the edge-level variance of the
    credibility that lies between target genes and between TFs (all universe edges).  A LIT
    annotator's observation indicator of (t, g) is the node indicator of g repeated over the
    eligible TFs, so a class driven by the observation emission is a target-gene attribute."""
    if groups is None:
        groups = {"target": universe_edge_targets(universe)[0], "tf": E.universe_edge_tfs(universe)[0]}
    ok = ep >= 0
    c = np.asarray(cred, dtype=np.float64)[ep[ok]]
    return {f"cred_var_share_between_{k}s": between_group_share(c, g[ok]) for k, g in groups.items()}


def _alt_columns(u: str, fid: str, opt: pd.DataFrame) -> dict:
    """Fit-summary columns of the alternative optimum (``alt_optimum``) of an obs MAP fit."""
    if not fid.startswith(OBS_FITS):
        return {}
    k, role = alt_optimum(u, fid)
    if k is None:
        return dict(alt_optimum=np.nan, alt_optimum_role="none")
    r = opt.iloc[k]
    return dict(alt_optimum=int(k), alt_optimum_role=role, alt_optimum_gap=float(r["gap_to_best"]),
                alt_optimum_n_restarts=int(r["n_restarts"]), alt_optimum_spearman_w=float(r["spearman_w_vs_best"]),
                alt_optimum_jaccard_bfdr0_05=float(r["jaccard_bfdr0_05_vs_best"]),
                alt_optimum_laplace_rel_mass=float(r["laplace_rel_mass"]) if "laplace_rel_mass" in r else np.nan)


def fit_summary_table(universes=UNIVERSES) -> pd.DataFrame:
    from r2 import fit_primary as FP
    rows = []
    for u in universes:
        data, base = load_elig(u, with_base=True)
        ep = edge_pattern_elig(u)
        groups = {"target": universe_edge_targets(u)[0], "tf": E.universe_edge_tfs(u)[0]}
        assert len(groups["target"]) == len(ep) == len(groups["tf"])
        n = data.n
        obs = (data.R >= 0).any(1)
        rep = (data.R == 1).any(1)
        st = status_score(data)
        rdiag = replication_diagnostics(u, data)
        n_obs_lit = (data.R[:, data.obs_columns(("LIT",))] >= 0).sum(1)
        for fid in all_obs_fits() + (NUTS_FIT,) + mar_comparators(u):
            f = read_fit(u, fid)
            if f is None:
                continue
            meta = f["meta"]
            is_obs = fid.startswith(OBS_FITS)
            cred = f["cred"] if is_obs else np.asarray(f["cred"])[base]
            p = f["params"]
            pr = obs_params(p) if is_obs else mar_params(p)
            conv = meta.get("convergence", {})
            r = dict(universe=u, fit_id=fid, model=meta.get("model"), inference=meta.get("inference"),
                     variant=meta.get("variant"), analysis_status="post_hoc" if is_obs else "prespecified",
                     n_patterns=int(data.P), n_edges=float(n.sum()), class_names=" ".join(meta.get("class_names", [])),
                     w=" ".join(f"{v:.6g}" for v in np.asarray(p["w"])), w_T=float(np.asarray(p["w"])[0]),
                     joint_loglik=lcm.heldout_loglik(_unweighted(pr), data) if is_obs else np.nan,
                     cond_loglik_r_given_o=lcm.conditional_loglik(pr, data),
                     obs_loglik=lcm.observation_loglik(_unweighted(pr), data) if is_obs else np.nan,
                     composite_loglik=lcm.heldout_loglik(pr, data) if "obs_weight" in pr else np.nan,
                     best_log_post=meta.get("best_log_post"),
                     expected_true_all_edges=float(np.sum(n * cred)),
                     expected_true_observed=float(np.sum(n[obs] * cred[obs])),
                     n_bfdr_0_05=_bfdr_n(cred, n, 0.05), n_bfdr_0_20=_bfdr_n(cred, n, 0.20),
                     mean_cred_unobserved=float(np.average(cred[st == 0], weights=n[st == 0])),
                     mean_cred_observed_neg=float(np.average(cred[st == 1], weights=n[st == 1])),
                     mean_cred_reported=float(np.average(cred[st == 2], weights=n[st == 2])),
                     spearman_cred_vs_n_lit_observing=FP.weighted_spearman(cred, n_obs_lit, n),
                     spearman_cred_vs_n_lit_observing_reported=FP.weighted_spearman(cred[rep], n_obs_lit[rep], n[rep]),
                     **gene_level_diagnostics(u, cred, ep, groups), **rdiag)
            if is_obs:
                jl, ol = r["joint_loglik"], r["obs_loglik"]
                r.update(obs_share_of_joint_loglik=float(ol / jl) if jl != 0 else np.nan)
            if meta.get("inference") == "MAP":
                opt = pd.read_csv(f["dir"] / "optima.csv")
                r.update(n_restarts=meta.get("n_restarts"), n_restarts_at_best=int(opt.n_restarts.iloc[0]),
                         n_distinct_optima=int(len(opt)),
                         gap_to_next_optimum=float(opt.gap_to_best.iloc[1]) if len(opt) > 1 else np.nan,
                         second_optimum_n_restarts=int(opt.n_restarts.iloc[1]) if len(opt) > 1 else np.nan,
                         second_optimum_spearman_w=float(opt.spearman_w_vs_best.iloc[1]) if len(opt) > 1 else np.nan,
                         second_optimum_jaccard_bfdr0_05=(float(opt.jaccard_bfdr0_05_vs_best.iloc[1])
                                                          if len(opt) > 1 else np.nan),
                         laplace_rel_mass_best=(float(opt.laplace_rel_mass.iloc[0]) if "laplace_rel_mass" in opt else np.nan),
                         **_alt_columns(u, fid, opt),
                         n_lbfgs_success=conv.get("n_converged"), grad_norm_best=conv.get("grad_norm_best"),
                         polish_grad_after=meta.get("polish", {}).get("grad_norm_after"),
                         polish_max_abs_cred_diff=meta.get("polish", {}).get("max_abs_cred_diff"),
                         hess_min_eig=meta.get("polish", {}).get("hess_min_eig"),
                         runtime_s=meta.get("runtime_s"))
            else:
                W = f["draws"]["w"]
                r.update(w_T_lo=float(np.quantile(W[:, 0], 0.025)), w_T_hi=float(np.quantile(W[:, 0], 0.975)),
                         r_hat_max=conv.get("r_hat_max"), ess_bulk_min=conv.get("ess_bulk_min"),
                         ess_tail_min=conv.get("ess_tail_min"), r_hat_max_cred=conv.get("r_hat_max_cred"),
                         n_divergent=meta.get("n_divergent"),
                         n_chains_at_best_map_optimum=conv.get("n_chains_at_best_map_optimum"),
                         num_chains=meta.get("num_chains"), num_warmup=meta.get("num_warmup"),
                         num_samples=meta.get("num_samples"), runtime_s=meta.get("runtime_s"),
                         cred_ci_width_mean_observed=conv.get("cred_ci_width_mean_w"))
            if is_obs:
                eta = np.asarray(p["eta"])
                lit = np.isfinite(eta[:, 0])
                r.update(median_eta_T=float(np.median(eta[lit, 0])), median_eta_NonT=float(np.median(eta[lit, 1])),
                         n_lit_eta_T_gt_NonT=int(np.sum(eta[lit, 0] > eta[lit, 1])), n_lit=int(lit.sum()))
                mar = lcm.MAR_COUNTERPART.get(fid.split("__")[0], "family_hb_nuts" if fid == NUTS_FIT else None)
                fm = read_fit(u, mar) if mar else None
                if fm is not None:
                    cm = np.asarray(fm["cred"])[base]
                    r.update(mar_counterpart=mar,
                             spearman_w_vs_mar_observed=FP.weighted_spearman(cred[obs], cm[obs], n[obs]),
                             spearman_w_vs_mar_all=FP.weighted_spearman(cred, cm, n),
                             jaccard_bfdr0_05_vs_mar=FP.weighted_jaccard(FP.bfdr_set(cred, n, 0.05),
                                                                         FP.bfdr_set(cm, n, 0.05), n))
            rows.append(r)
    return pd.DataFrame(rows)


def laplace_eta_sd(u: str, fit_id: str, data: lcm.PatternData, optimum: int = 0) -> dict | None:
    """Laplace (Gaussian in the unconstrained space) standard deviations of logit eta_T,
    logit eta_NonT and their difference for every modelled annotator (keys ``sd_T``, ``sd_NonT``,
    ``sd_diff``; length-K arrays, NaN for the other annotators) and of logit w_T (``sd_logit_w_T``,
    scalar) at a stored MAP optimum
    (``optima_params.npz`` key ``x<optimum>``): delta method, cov = J H^-1 J' with H the Hessian
    of the potential.  None if the optimum is not stored or H is not positive definite.  The
    intervals are conditional on the model's edge-level independence of the observation
    indicators (which are replicated over TFs), so they understate the uncertainty; for the
    ``__tfw`` fits H is the Hessian of the composite objective (not a calibrated posterior)."""
    import jax
    import jax.numpy as jnp
    from r2 import fit_primary as FP
    d = fit_dir(u, fit_id)
    if not (d / "optima_params.npz").exists():
        return None
    with np.load(d / "optima_params.npz") as z:
        if f"x{optimum}" not in z.files:
            return None
        x = np.asarray(z[f"x{optimum}"], dtype=float)
    meta = json.loads((d / "meta.json").read_text())
    pt = FP._potential(data, meta["model"], fit_prior(meta))
    H = FP._sym_hess(pt, x)
    if not np.all(np.isfinite(H)) or np.linalg.eigvalsh(H).min() <= 0:
        return None
    cols = data.obs_columns(("LIT",))
    jc = jnp.asarray(cols)

    def f(xx):
        s = pt["init"].postprocess_fn(pt["unravel"](xx))
        e = s["eta"][jc][:, :2]
        lg = jnp.log(e) - jnp.log1p(-e)
        wt = s["w"][0]
        return jnp.concatenate([lg[:, 0], lg[:, 1], lg[:, 0] - lg[:, 1], (jnp.log(wt) - jnp.log1p(-wt))[None]])

    J = np.asarray(jax.jacfwd(f)(jnp.asarray(x)), dtype=float)
    sd = np.sqrt(np.clip(np.einsum("ij,ji->i", J, np.linalg.solve(H, J.T)), 0, None))
    k = len(cols)
    out = {name: np.full(data.K, np.nan) for name in ("sd_T", "sd_NonT", "sd_diff")}
    out["sd_T"][cols], out["sd_NonT"][cols], out["sd_diff"][cols] = sd[:k], sd[k:2 * k], sd[2 * k:3 * k]
    out["sd_logit_w_T"] = float(sd[3 * k])        # scalar: logit of the class-T weight
    return out


def params_table(universes=UNIVERSES) -> pd.DataFrame:
    """Per universe x obs fit (best optimum, alternative optimum; NUTS only if such a fit exists --
    none does) x annotator: eta_T, eta_NonT (MAP value with Laplace 95% intervals), logit
    difference, alpha, beta, gamma, the annotator's eligible / observed / reported edges, and the
    class-T weight of the optimum with its Laplace 95% interval (``w_T``, ``w_T_lo``, ``w_T_hi``;
    repeated on every row of the optimum).  The Laplace intervals describe the curvature of one
    posterior basin under the model's independent-edge likelihood; they say nothing about the
    other (label-swapped) basin and understate the uncertainty (replicated node indicators)."""
    from r2 import fit_primary as FP
    rows = []
    for u in universes:
        data = load_elig(u)
        n = data.n
        m_el = ((data.R >= -1) * n[:, None]).sum(0)
        m_ob = ((data.R >= 0) * n[:, None]).sum(0)
        m_rp = ((data.R == 1) * n[:, None]).sum(0)
        lit = np.zeros(data.K, bool)
        lit[data.obs_columns(("LIT",))] = True
        sets = []
        for fid in all_obs_fits():
            f = read_fit(u, fid)
            if f is None:
                continue
            opt_tab = pd.read_csv(fit_dir(u, fid) / "optima.csv")
            sets.append((fid, "MAP", 0, f["params"], None, f["meta"]["class_names"], "best", opt_tab))
            k, role = alt_optimum(u, fid)
            if k is not None:
                sets.append((fid, "MAP", k, optimum_params(u, fid, data, k), None, f["meta"]["class_names"], role,
                             opt_tab))
        f = read_fit(u, NUTS_FIT)
        if f is not None:
            sets.append((NUTS_FIT, "NUTS", 0, f["params"], f["draws"], f["meta"]["class_names"],
                         "posterior_at_best_mode", None))
        for fid, inf, opt, p, draws, cn, role, opt_tab in sets:
            ap = FP.annotator_parameters(np.asarray(p["theta"]), cn, data.family)
            eta = np.asarray(p["eta"])
            lap = laplace_eta_sd(u, fid, data, opt) if inf == "MAP" else None
            q = 1.959964
            w_T = float(np.asarray(p["w"])[0])
            w_ci = ((_expit(_logit(w_T) - q * lap["sd_logit_w_T"]), _expit(_logit(w_T) + q * lap["sd_logit_w_T"]))
                    if lap is not None else
                    (float(np.quantile(draws["w"][:, 0], 0.025)), float(np.quantile(draws["w"][:, 0], 0.975)))
                    if draws is not None else (np.nan, np.nan))
            for j, a in enumerate(data.annotators):
                r = dict(universe=u, fit_id=fid, inference=inf, optimum=opt, optimum_role=role,
                         optimum_gap_to_best=(float(opt_tab["gap_to_best"].iloc[opt]) if opt_tab is not None else np.nan),
                         optimum_n_restarts=(int(opt_tab["n_restarts"].iloc[opt]) if opt_tab is not None else np.nan),
                         w_T=w_T, w_T_lo=w_ci[0], w_T_hi=w_ci[1], annotator=a, family=data.family[j],
                         observation_modelled=bool(lit[j]), n_eligible=float(m_el[j]) if lit[j] else np.nan,
                         n_observed=float(m_ob[j]), n_reported=float(m_rp[j]),
                         empirical_obs_rate=float(m_ob[j] / m_el[j]) if lit[j] and m_el[j] > 0 else np.nan,
                         eta_T=float(eta[j, 0]), eta_NonT=float(eta[j, 1]),
                         logit_eta_T_minus_NonT=float(np.log(eta[j, 0] / (1 - eta[j, 0])) - np.log(eta[j, 1] / (1 - eta[j, 1])))
                         if lit[j] else np.nan,
                         alpha=float(ap["alpha"][j]), beta=float(ap["beta"][j]),
                         gamma=float(ap["gamma"][j]) if "gamma" in ap else np.nan, analysis_status="post_hoc")
                if lap is not None and lit[j]:
                    lt, ln = _logit(eta[j, 0]), _logit(eta[j, 1])
                    r.update(eta_T_lo=_expit(lt - q * lap["sd_T"][j]), eta_T_hi=_expit(lt + q * lap["sd_T"][j]),
                             eta_NonT_lo=_expit(ln - q * lap["sd_NonT"][j]),
                             eta_NonT_hi=_expit(ln + q * lap["sd_NonT"][j]),
                             logit_diff_lo=float(lt - ln - q * lap["sd_diff"][j]),
                             logit_diff_hi=float(lt - ln + q * lap["sd_diff"][j]),
                             interval="Laplace (Gaussian in the unconstrained space, delta method) 95% interval "
                                      "at the MAP optimum; assumes independent edges")
                if draws is not None and lit[j]:
                    r["interval"] = "NUTS 95% credible interval (2.5%, 97.5% quantiles); assumes independent edges"
                    E_ = draws["eta"][:, j, :]
                    d_ = np.log(E_[:, 0] / (1 - E_[:, 0])) - np.log(E_[:, 1] / (1 - E_[:, 1]))
                    r.update(eta_T_lo=float(np.quantile(E_[:, 0], 0.025)), eta_T_hi=float(np.quantile(E_[:, 0], 0.975)),
                             eta_NonT_lo=float(np.quantile(E_[:, 1], 0.025)),
                             eta_NonT_hi=float(np.quantile(E_[:, 1], 0.975)),
                             logit_diff_lo=float(np.quantile(d_, 0.025)), logit_diff_hi=float(np.quantile(d_, 0.975)),
                             prob_eta_T_gt_NonT=float(np.mean(E_[:, 0] > E_[:, 1])))
                if draws is not None:
                    TH = draws["theta"]
                    apd = FP.annotator_parameters(TH, cn, data.family)
                    for k in ("alpha", "beta"):
                        r[f"{k}_lo"] = float(np.quantile(apd[k][:, j], 0.025))
                        r[f"{k}_hi"] = float(np.quantile(apd[k][:, j], 0.975))
                rows.append(r)
    return pd.DataFrame(rows)


def optima_table(universes=UNIVERSES) -> pd.DataFrame:
    """Mode structure of every informative-observation MAP fit (as ``fit_map_optima.csv`` of
    ``fit_primary``): one row per distinct optimum with its restarts, gap, Laplace mass, agreement
    with the best optimum, and the orientation of the observation classes at that optimum (w_T,
    median eta_T / eta_NonT, number of LIT annotators with eta_T > eta_NonT)."""
    rows = []
    for u in universes:
        data = load_elig(u)
        lit = data.obs_columns(("LIT",))
        for fid in all_obs_fits():
            d = fit_dir(u, fid)
            if not (d / "optima.csv").exists():
                continue
            opt = pd.read_csv(d / "optima.csv")
            meta = json.loads((d / "meta.json").read_text())
            k_alt, role_alt = alt_optimum(u, fid)
            for k, r in opt.iterrows():
                row = dict(universe=u, fit_id=fid, model=meta["model"], n_restarts_total=meta["n_restarts"])
                row.update({c: r[c] for c in opt.columns if c != "restarts"})
                conv = bool(np.isfinite(r.get("laplace_log_mass", np.nan)))
                row["polished_positive_definite"] = conv
                row["role"] = ("best" if k == 0 else role_alt if k == k_alt else
                               "other" if conv else "not_converged_restart")
                pk = optimum_params(u, fid, data, int(k)) if conv else None
                if pk is not None:
                    eta = np.asarray(pk["eta"])[lit]
                    row.update(w_T=float(np.asarray(pk["w"])[0]), w=" ".join(f"{v:.6g}" for v in np.asarray(pk["w"])),
                               median_eta_T=float(np.median(eta[:, 0])), median_eta_NonT=float(np.median(eta[:, 1])),
                               n_lit_eta_T_gt_NonT=int(np.sum(eta[:, 0] > eta[:, 1])), n_lit=int(len(lit)))
                row["analysis_status"] = "post_hoc"
                rows.append(row)
    return pd.DataFrame(rows)


# dated decision log of this module (shared table ``eval_analysis_decisions.csv``); everything is post hoc
ANALYSIS_DECISIONS = [
    dict(id="obs_design", date="2026-10-08", status="post_hoc",
         item="informative-observation latent-class models family_hb_obs / em_obs (LIT observation as a class-"
              "dependent emission; eta_T, eta_NonT per annotator)",
         note="designed after the held-out results were seen (plan section 8); secondary analysis; Family-HB stays primary"),
    dict(id="obs_tfw", date="2026-10-08", status="post_hoc",
         item="TF-replication-weighted (composite-likelihood) sensitivity fits *__tfw: observation log-likelihood of "
              "LIT annotator j weighted by 1 / number of universe TFs eligible for j",
         note="defined before any informative-observation fit was run; motivated by the design (eligibility TF-level, "
              "observation gene-level); fitted 2026-10-13"),
    dict(id="obs_alt_optimum", date="2026-10-13", status="post_hoc",
         item="label-swapped MAP optimum of the generative fits (*_swap) and second optimum of the weighted fits "
              "(*_opt2) reported next to the best optimum in parameters and held-out tables",
         note="added after the MAP fits showed that family_hb_obs and em_obs select opposite labellings"),
    dict(id="obs_gene_share", date="2026-10-13", status="post_hoc",
         item="share of the variance of the credibility between target genes / TFs (obs_fit_summary.csv)",
         note="label-free diagnostic added after the MAP parameters were inspected"),
    dict(id="obs_nuts", date="2026-10-13", status="post_hoc",
         item="NUTS of family_hb_obs NOT run: the informative-observation models are reported from MAP fits with "
              "Laplace intervals only (no family_hb_obs_nuts fit; stage 'nuts' excluded from 'all')",
         note="pilots (immune): non-centred sampler saturates tree depth 10 (1,023 leapfrogs per iteration; projected "
              "4.7 h immune, ~14 h global at 4 x 1000/1000), partially centred coordinates 450 leapfrogs (3.8 h immune); "
              "the MAP analysis already shows two label-swapped basins with negligible within-basin spread, so "
              "sampling one basin would not change a conclusion; no held-out data involved in the decision"),
    dict(id="obs_no_nuts_comparator", date="2026-10-13", status="post_hoc",
         item="family_hb_nuts is not read by this module unless a family_hb_obs_nuts fit exists",
         note="keeps the obs_* tables independent of re-runs of the primary NUTS fits by fit_primary"),
    dict(id="obs_laplace_w_T", date="2026-10-13", status="post_hoc",
         item="Laplace 95% interval of the class-T weight of every informative-observation MAP optimum "
              "(obs_params.csv, obs_fit_summary.csv)",
         note="added with the MAP-only decision; describes one basin under the independent-edge likelihood"),
    dict(id="obs_cv_starts", date="2026-10-13", status="post_hoc",
         item="CV fold fits: warm starts from the distinct full-data optima (+ 4 random starts in immune; warm starts "
              "only in global)",
         note="training log posterior selects the fit; fixed before the CV results were read"),
    dict(id="obs_status_split", date="2026-10-08", status="post_hoc",
         item="unobserved evidence-status class split by the number of eligible but unobserving LIT annotators; "
              "label-free 3-level status score and Family-HB with unobserved edges last as comparators",
         note="status scores as defined in heldout_eval (verifier held M2)"),
    dict(id="obs_cond_explored", date="2026-10-13", status="post_hoc",
         item="conditional-likelihood variant (lcm obs_conditional) explored on immune and not pursued",
         note="strongly multimodal with unanchored labels (logs/cond_exploration_immune.log); no fits or tables"),
]


W_T_LAPLACE = ("Laplace 95% interval (Gaussian in the unconstrained space, delta method on logit w_T) at the MAP "
               "optimum; one posterior basin; assumes independent edges")


def add_w_T_intervals(summary: pd.DataFrame, params: pd.DataFrame) -> pd.DataFrame:
    """Fill ``w_T_lo`` / ``w_T_hi`` of the informative-observation MAP rows of the fit summary with
    the Laplace interval of the best optimum (rows ``optimum == 0`` of the parameter table) and
    label the kind of interval in ``w_T_interval`` (MAR rows: empty; their intervals belong to
    the tables of ``fit_primary``)."""
    out = summary.copy()
    for c in ("w_T_lo", "w_T_hi"):
        if c not in out:
            out[c] = np.nan
    out["w_T_interval"] = np.where(out["inference"].eq("NUTS") & out["fit_id"].eq(NUTS_FIT),
                                   "NUTS 95% credible interval", "")
    if len(params):
        best = (params[(params["inference"] == "MAP") & (params["optimum"] == 0)]
                .drop_duplicates(["universe", "fit_id"]).set_index(["universe", "fit_id"]))
        for i, r in out.iterrows():
            key = (r["universe"], r["fit_id"])
            if r["inference"] == "MAP" and key in best.index and np.isfinite(best.loc[key, "w_T_lo"]):
                if not np.isclose(best.loc[key, "w_T"], r["w_T"], rtol=1e-9):
                    raise AssertionError(f"{key}: w_T of the parameter table differs from the fit summary")
                out.loc[i, "w_T_lo"], out.loc[i, "w_T_hi"] = float(best.loc[key, "w_T_lo"]), float(best.loc[key, "w_T_hi"])
                out.loc[i, "w_T_interval"] = W_T_LAPLACE
    return out


def older_than(outputs: dict, inputs: dict) -> list:
    """Problems of a set of derived files: {name: mtime or None}.  An output is reported if it is
    missing (None) or older than the newest input; a missing input is reported too."""
    probs = [f"missing input: {k}" for k, t in inputs.items() if t is None]
    newest = max([t for t in inputs.values() if t is not None], default=None)
    for k, t in outputs.items():
        if t is None:
            probs.append(f"missing output: {k}")
        elif newest is not None and t < newest:
            probs.append(f"stale output (older than its inputs): {k}")
    return probs


def _mtime(p: Path):
    return p.stat().st_mtime if p.exists() else None


def fit_inputs(universe: str, fit_id: str) -> dict:
    """{relative path: mtime} of the files that define a stored fit for this module
    (``params.npz``, ``cred.npy``; ``optima_params.npz`` for the informative-observation fits,
    whose optima give the CV warm starts and the alternative optima)."""
    d = fit_dir(universe, fit_id)
    names = ("params.npz", "cred.npy") + (("optima_params.npz",) if fit_id.startswith(OBS_FITS) else ())
    return {str((d / f).relative_to(C.SP)): _mtime(d / f) for f in names}


def check_outputs(universes=UNIVERSES, raise_on_problem: bool = True) -> dict:
    """Freshness of every stored output of this module against the fits it was computed from.

    * every reported fit exists (``family_hb_obs``, ``em_obs`` and the ``__tfw`` fits, 4 universes);
    * PPC replicate files and CV fold files are newer than the fit(s) they used (CV of the MAR
      fits: newer than the ``fit_sensitivity`` fold file they copy);
    * the credibility file behind every held-out score still has the md5 recorded by the
      ``heldout`` stage (``heldout_provenance.json``), and every fit is in that record;
    * each table is newer than its inputs.
    Writes ``output_check.json`` (problems, md5 of every credibility file) and raises if a
    problem is found.  File times are compared, so copying the outputs without preserving the
    modification times makes the check fail (re-run the stage in that case)."""
    probs = []
    fits = OBS_FITS + TFW_FITS + tuple(f for f in COND_FITS if any((fit_dir(u, f) / "meta.json").exists()
                                                                    for u in universes))
    all_in = {}
    creds = {}
    for u in universes:
        for fid in fits + mar_comparators(u):
            ins = fit_inputs(u, fid)
            probs += [f"missing fit file: {k}" for k, t in ins.items() if t is None]
            all_in.update({k: t for k, t in ins.items() if t is not None})
            c = fit_dir(u, fid) / ("cred_mean.npy" if fid.endswith("_nuts") else "cred.npy")
            if c.exists():
                creds[f"{u}/{fid}"] = file_md5(c)
    # PPC and CV files
    for u in [x for x in universes if x in PPC_UNIVERSES]:
        od = MOD / "ppc" / u
        for fid in fits + MAR_FITS:
            if fid in fits and not (fit_dir(u, fid) / "meta.json").exists():
                continue
            lab = fid if fid in fits else f"{fid}+ignorable_obs"
            want = [od / f"conditional__{fid}.npz"] + ([] if fid in COND_FITS else [od / f"joint__{lab}.npz"])
            probs += older_than({str(f.relative_to(C.SP)): _mtime(f) for f in want}, fit_inputs(u, fid))
    for u in [x for x in universes if x in CV_UNIVERSES]:
        od = MOD / "cv" / u
        for f in range(N_FOLDS):
            for fid in fits + MAR_FITS:
                out = od / f"fold{f}__{fid}.json"
                ins = (fit_inputs(u, fid) if fid in fits else
                       {str((FIT_SENS_CV / u / out.name).relative_to(C.SP)): _mtime(FIT_SENS_CV / u / out.name)})
                probs += older_than({str(out.relative_to(C.SP)): _mtime(out)}, ins)
    # held-out provenance
    pf = MOD / "heldout_provenance.json"
    if not pf.exists():
        probs.append("missing output: heldout_provenance.json (run the heldout stage)")
    else:
        prov = json.loads(pf.read_text())["scores"]
        for u in universes:
            for fid in fits + mar_comparators(u):
                if not (fit_dir(u, fid) / "meta.json").exists():
                    continue
                rec = prov.get(u, {}).get(fid)
                if rec is None:
                    probs.append(f"held-out tables lack {u}/{fid} (re-run the heldout stage)")
                elif rec["md5"] != creds.get(f"{u}/{fid}"):
                    probs.append(f"held-out tables used another version of {u}/{fid} (re-run the heldout stage)")
            extra = set(prov.get(u, {})) - set(fits + mar_comparators(u))
            probs += [f"held-out tables contain {u}/{k}, which this module no longer reads" for k in sorted(extra)]
    # tables
    tab = lambda n: {f"data/r2/tables/{n}.csv": _mtime(TABLES / f"{n}.csv")}      # noqa: E731
    for n in ("obs_fit_summary", "obs_map_optima", "obs_params", "obs_heldout", "obs_heldout_paired",
              "obs_heldout_status"):
        probs += older_than(tab(n), all_in)
    ppc_files = {str(f.relative_to(C.SP)): _mtime(f) for u in universes for f in (MOD / "ppc" / u).glob("*__*.npz")}
    cv_files = {str(f.relative_to(C.SP)): _mtime(f) for u in universes for f in (MOD / "cv" / u).glob("fold*__*.json")}
    probs += older_than(tab("obs_ppc"), ppc_files) + older_than(tab("obs_cv"), cv_files)
    probs = sorted(set(probs))
    rec = dict(checked=time.strftime("%Y-%m-%d %H:%M:%S"), universes=list(universes), fits=list(fits),
               mar_comparators={u: list(mar_comparators(u)) for u in universes},
               nuts_fit_present={u: (fit_dir(u, NUTS_FIT) / "meta.json").exists() for u in universes},
               n_ppc_files=len(ppc_files), n_cv_files=len(cv_files), n_problems=len(probs), problems=probs,
               cred_md5=creds)
    C.write_json(_jsonable(rec), C.ensure_dir(MOD) / "output_check.json")
    log(f"[check] {len(creds)} fits, {len(ppc_files)} PPC files, {len(cv_files)} CV files: {len(probs)} problem(s)")
    for p in probs:
        log(f"[check]   {p}")
    if probs and raise_on_problem:
        raise RuntimeError(f"{len(probs)} stale or missing output(s); see {MOD / 'output_check.json'}")
    return rec


def build_tables(check: bool = True) -> None:
    C.ensure_dir(TABLES)
    from r2 import heldout_eval as HE
    HE.record_decisions("informative_observation", ANALYSIS_DECISIONS)
    params = params_table()
    for name, df in (("obs_fit_summary", add_w_T_intervals(fit_summary_table(), params)),
                     ("obs_map_optima", optima_table()), ("obs_params", params),
                     ("obs_ppc", ppc_table()), ("obs_cv", cv_table())):
        if len(df):
            df.to_csv(TABLES / f"{name}.csv", index=False, float_format="%.6g")
            log(f"[tables] {name}.csv: {len(df)} rows")
    if check:
        check_outputs()


# ----------------------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("stage", choices=["evidence", "map", "map_tfw", "map_cond", "nuts", "ppc", "cv", "heldout",
                                      "tables", "check", "all"])
    ap.add_argument("--universes", nargs="+", default=None)
    ap.add_argument("--models", nargs="+", default=list(OBS_FITS))
    ap.add_argument("--folds", nargs="+", type=int, default=None)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args(argv)
    C.ensure_dir(MOD)
    t0 = time.time()
    if a.stage in ("evidence", "all"):
        run_evidence(a.universes or UNIVERSES)
    if a.stage in ("map", "all"):
        run_map(a.universes or UNIVERSES, a.models, force=a.force)
    if a.stage in ("map_tfw", "all"):
        run_map_tfw(a.universes or TFW_UNIVERSES, a.models, force=a.force)
    if a.stage == "map_cond":            # exploratory; not part of ``all`` (see the module docstring)
        run_map_cond(a.universes or UNIVERSES, a.models, force=a.force)
    if a.stage == "nuts":                # NOT run for the reported results and not part of ``all`` (module docstring)
        run_nuts(a.universes or NUTS_UNIVERSES, force=a.force)
    if a.stage in ("ppc", "all"):
        run_ppc(a.universes or PPC_UNIVERSES, force=a.force)
    if a.stage in ("cv", "all"):
        run_cv(a.universes or CV_UNIVERSES, a.folds, force=a.force)
    if a.stage in ("heldout", "all"):
        run_heldout(a.universes or UNIVERSES)
    if a.stage in ("tables", "all"):
        build_tables()
    if a.stage == "check":
        check_outputs()
    log(f"done {a.stage} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
