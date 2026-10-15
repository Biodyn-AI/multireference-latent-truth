"""Label-free sensitivity analyses and model checks of the credibility models (module ``fits_sensitivity``).

Reviewer points Q1 (iv: inference checks), Q20 (family definitions) and the label-free part of
Q6.  Nothing in this module reads held-out data (ChIP references, perturbation cohorts); every
fit uses only the stored annotator evidence of ``evidence.py``.  Universes: ``global`` and
``immune``; evidence variant ``PRESPECIFIED['evidence_variant_primary']`` unless stated.

Parts
-----
(a) ``fits`` -- evidence variants.  For every variant in
    ``PRESPECIFIED['evidence_variants_sensitivity']``: MAP fits ``family_hb`` and ``em``
    (20 L-BFGS restarts) stored under ``fits/<universe>/<variant>/``.
(b) ``fits`` -- family groupings (Q20), primary variant, ``family_hb`` MAP (20 restarts):
      ``family_hb__fam_singleton``   every annotator its own family: no family bursts and no
                                     pooling across annotators below the global level;
      ``family_hb__fam_lineage``     integrator lineage (``annotator_sources.json``:
                                     CollecTRI / DoRothEA / Multi-integrator);
      ``family_hb__fam_datadriven``  average-linkage clustering of annotators on
                                     d_jk = 1 - phi_jk, phi_jk = phi coefficient of reporting
                                     among the edges co-observed by j and k (pattern-weighted),
                                     cut into 3 clusters (= number of modality families).
                                     A pair without co-observed edges, or with a constant
                                     margin, has phi = 0.
(c) ``fits`` / ``nuts`` -- priors: ``family_hb__prior_scale0.5`` / ``__prior_scale2``
    (HalfNormal sd of the hierarchical scales tau, omega), ``__prior_dir0.5`` / ``__prior_dir2``
    (Dirichlet concentration of the class weights); MAP 20 restarts in both universes; NUTS
    (4 chains x 1,000 warm-up + 1,000 draws) for scale0.5 and scale2 on immune.
(d) ``cv`` -- label-free model comparison: 5-fold cross-validation with folds grouped by TF
    (TFs permuted with ``PRESPECIFIED['random_seed']`` and split into 5 near-equal groups).  Each
    of the five estimators and the three family groupings is fitted (MAP, 20 restarts) on the
    training-TF edges (pattern counts recomputed from the per-edge pattern index; the
    data-driven grouping is re-clustered on the training edges) and scored by the marginal
    log-likelihood of the test-TF edges, divided by the number of test edges observed by >= 1
    annotator (edges observed by no annotator contribute exactly 0).
(e) ``ppc`` -- plug-in posterior predictive checks: for every estimator (and grouping), 20
    replicate data sets from its MAP fit with the real observation masks; statistics
    (observed edges only): distribution of the number of reporting annotators (0..5, >=6),
    pairwise co-reporting log-odds ratios among co-observed edges (0.5-corrected; pairs with
    co-observed edges and >= 1 report of each annotator among them in the real data;
    within-/between-modality-family averages and per family pair), per-annotator positive
    rates, and the fraction of edges with >= 1 report by number of observing annotators.
    Standardised discrepancy z = (T_obs - mean T_rep) / sd T_rep; ppp = P(T_rep >= T_obs).
    Under the curated-conditional masks a MOTIF/COEXP annotator observes an edge only if a
    held-in literature resource reports it, so re-simulating reports on fixed masks breaks that
    constraint and distorts every statistic involving MOTIF/COEXP.  Two checks are therefore
    added: (i) the same statistics on the LIT block only (``litblock_*``; LIT masks do not depend
    on other annotators' reports and the LIT margin of the model is the same latent-class model
    with the other columns integrated out), and (ii) an analytic leave-one-annotator-out
    conditional check (``conditional_annotator_check``) that conditions on the other reports.
(f) ``recovery`` -- simulation-based recovery and calibration: data simulated from the
    primary ``family_hb`` MAP fit with the real observation masks and N (immune 20 replicates,
    global 5); refits ``family_hb`` (10 restarts) and ``em`` (10 restarts); bias / RMSE of
    w_T, alpha_j, beta_j (and gamma_j, class weights) by modality family; AUROC of credibility
    against the true class; calibration (10 equal-width credibility bins: mean predicted vs
    empirical true-T rate; ECE); weighted Spearman against the oracle credibility (true
    parameters).  ``recovery_nuts``: NUTS (4 x 500 warm-up / 500 draws) on 5 immune replicates for
    the 95% credible-interval coverage of w_T, alpha_j, beta_j.  ``truthstart``: identifiability
    diagnostic -- one more ``family_hb`` L-BFGS run per replicate started at the generating
    parameters (``map_from_start``); its optimum is compared with the best random-start optimum
    (log posterior, Spearman vs oracle), separating optimiser failures from competing modes; its
    credibility enters the calibration table as ``family_hb_truthstart``.  The NUTS replicates
    report coverage for the pooled chains and for the chains of the label-free selected mode
    (``select_mode_chains``: chains within 2 nats of the best chain-mean log-likelihood).
(g) ``modes`` -- posterior modes of the primary Family-HB fit: the 50 restarts of the primary MAP
    fit (same seed, hence the same starting points) are grouped into distinct optima; for each,
    log posterior, Laplace log mass log p(x*) + d/2 log 2pi - 1/2 log det H(x*) in the
    unconstrained space (``map_optima``), its Laplace mode probability, class weights, BFDR-0.05 set
    and agreement with the best optimum.  Every stored NUTS chain (primary NUTS of the primary-fit
    module and the prior-sensitivity NUTS fits) is assigned to the nearest optimum by class
    weights (``fit_sensitivity_nuts_chains.csv``; until 2026-10-14 this table was written as
    ``fit_nuts_chains.csv``, the name of a different ``fit_primary.py`` table, which it overwrote
    whenever the two table stages ran in this order), and ``fit_nuts_prior_sensitivity.csv`` repeats each NUTS
    summary over the chains in the best optimum's mode.

Simulation with the real observation masks
------------------------------------------
``simulate_fixed_masks`` draws the latent class and the reports for exactly the observed edges
of the real data (each observation mask is used as often as it occurs), whereas
``lcm.simulate`` re-samples masks i.i.d. from their empirical distribution and needs an
(N x K) float matrix (1.6 GB for the 9.3M-edge global universe).  The generative model is the
same; edges observed by no annotator are not simulated because they carry no likelihood and
their credibility is the prior weight w_T.

Fit storage (contract shared by all modules)
--------------------------------------------
``outputs/r2/fits/<universe>/<variant>/<fit_id>/`` with ``cred.npy`` (per-pattern
credibility aligned to ``evidence.load_patterns(universe, variant)``), ``params.npz`` (w, theta
and the other constrained sites), ``meta.json``, ``restarts.csv``; NUTS fits (``*_nuts``) add
``cred_mean/lo/hi/sd.npy``, ``draws.npz`` (w, theta; 1,000 evenly thinned draws),
``diagnostics.csv`` and ``n_divergent`` in ``meta.json``.  This module writes only the
sensitivity fit ids and the evidence-variant directories.  The primary MAP/NUTS fits
(``fits/<u>/<primary>/<model>``) are read when they exist (written by the primary-fit module);
otherwise they are fitted here with the same settings (20 restarts; NUTS 4 x 1,000/1,000) and
stored under ``outputs/r2/fit_sensitivity/primary_fallback/`` so that the canonical
directories are never written by two modules.  Every table records which source was used.

Robustness summaries (``data/r2/tables/fit_sensitivity.csv``)
-------------------------------------------------------------
Every sensitivity fit is compared with the primary ``family_hb`` fit (and ``em`` variant fits
also with the primary ``em``) at the edge level over the universe:
* weighted Spearman of credibility over edges observed by >= 1 annotator under either fit, and
  over edges with >= 1 report under either fit;
* mean absolute credibility difference over observed edges;
* Jaccard of the top-K edge sets, K in ``PRESPECIFIED['eval_k']``, ties at the K-th position
  broken by one fixed random key per edge (``PRESPECIFIED['topk_tie_seed']``), identical for
  all fits so that identical tie blocks select identical edges;
* Jaccard of the label-free Bayesian-FDR sets (largest set of whole credibility tie blocks
  with mean(1 - c) <= q; q = ``bfdr_strong`` 0.05 and ``bfdr_moderate`` 0.20).

Manuscript tables (``data/r2/tables``, prefix ``fit_``)
-----------------------------------------------------
* ``fit_sensitivity.csv``            every sensitivity fit vs the primary fit (see above).
* ``fit_sensitivity_params.csv``     class weights, hyper-parameters and family-mean alpha/beta of
                                     every Family-HB sensitivity fit.
* ``fit_family_groupings.csv``       annotator membership under modality / lineage / data-driven /
                                     singleton groupings.
* ``fit_coreporting_assoc.csv``      pairwise co-reporting counts, phi and log-odds ratio (the
                                     data-driven clustering input).
* ``fit_nuts_prior_sensitivity.csv`` NUTS fits under the primary and the two scale priors.
* ``fit_cv_loglik.csv``              TF-grouped 5-fold held-out log-likelihood per observed edge
                                     (mean, SE over folds, difference vs Family-HB);
                                     ``fit_cv_loglik_folds.csv`` per fold.
* ``fit_ppc.csv``                    posterior predictive checks (per statistic / stratum and
                                     summaries of |z|).
* ``fit_modes.csv``                  distinct MAP optima of the primary Family-HB fit with Laplace mass.
* ``fit_sensitivity_nuts_chains.csv`` per NUTS chain: class weights, nearest optimum, Spearman vs best
                                     (renamed 2026-10-14; ``fit_nuts_chains.csv`` belongs to ``fit_primary.py``).
* ``fit_recovery.csv``               parameter bias / RMSE, credibility AUROC / Spearman vs oracle /
                                     ECE, NUTS 95% coverage, identifiability (truth-start) summary;
                                     ``fit_recovery_calibration.csv`` pooled 10-bin calibration;
                                     ``fit_recovery_replicates.csv`` per replicate.

Command line::

    .venv/bin/python subproject_24_multireference_latent_truth_model/src/r2/fit_sensitivity.py all
    # or step by step (the --folds / --reps / --fit-ids options split work over processes)
    ... fit_sensitivity.py fits --universes global immune
    ... fit_sensitivity.py nuts [--fit-ids family_hb__prior_scale0.5 family_hb__prior_scale2]
    ... fit_sensitivity.py cv --universes global immune [--folds 0 1 2 3 4]
    ... fit_sensitivity.py ppc --universes global immune
    ... fit_sensitivity.py recovery --universes global immune [--reps ...]
    ... fit_sensitivity.py recovery_nuts [--reps 0 1 2 3 4]
    ... fit_sensitivity.py truthstart --universes global immune
    ... fit_sensitivity.py modes --universes global immune
    ... fit_sensitivity.py tables

Existing fits / fold results are reused (``--force`` refits).  Set OMP/OPENBLAS/MKL threads and
``XLA_FLAGS`` before running (defaults to 2 threads are applied below).
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=2")

import argparse  # noqa: E402
import io  # noqa: E402
import json  # noqa: E402
import shutil  # noqa: E402
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
PRIMARY = PS["evidence_variant_primary"]
SEED = int(PS["random_seed"])
TIE_SEED = int(PS["topk_tie_seed"])
EVAL_K = list(PS["eval_k"])
BFDR_Q = {"strong": float(PS["bfdr_strong"]), "moderate": float(PS["bfdr_moderate"])}

FITS = C.OUT / "fits"
MOD = C.OUT / "fit_sensitivity"
FALLBACK = MOD / "primary_fallback"
TABLES = C.TABLES
PREFIX = "fit_"

UNIVERSES = ("global", "immune")
MODELS = tuple(lcm.MODEL_SPECS)            # em, dependence, multifactor, family_regularized, family_hb
N_RESTARTS = 20
N_RESTARTS_RECOVERY = 10
N_FOLDS = 5
N_PPC = 20
N_RECOVERY = {"immune": 20, "global": 5}
N_RECOVERY_NUTS = 5
NUTS_MAIN = dict(num_warmup=1000, num_samples=1000, num_chains=4)
NUTS_RECOVERY = dict(num_warmup=500, num_samples=500, num_chains=4)
MAX_DRAWS_STORED = 1000
N_DD_CLUSTERS = 3
NREP_CAP = 6                               # number-of-reporting-annotators bins 0..5, >=6
NOBS_BINS = ((1, 1), (2, 2), (3, 4), (5, 7), (8, 99))   # number of observing annotators (PPC strata)
N_CAL_BINS = 10

GROUPINGS = ("singleton", "lineage", "datadriven")
PRIOR_SENS = {
    "family_hb__prior_scale0.5": {"scale_sd": 0.5},
    "family_hb__prior_scale2": {"scale_sd": 2.0},
    "family_hb__prior_dir0.5": {"dirichlet": 0.5},
    "family_hb__prior_dir2": {"dirichlet": 2.0},
}
NUTS_PRIOR_IDS = ("family_hb__prior_scale0.5", "family_hb__prior_scale2")
NUTS_UNIVERSE = "immune"
# label-free CV / PPC configurations: (config id, model, family grouping)
CONFIGS = [(m, m, "modality") for m in MODELS] + [
    (f"family_hb__fam_{g}", "family_hb", g) for g in GROUPINGS]


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def derived_seed(*parts) -> int:
    """Deterministic seed for a sub-task (simulation replicate, fold, ...)."""
    return int((SEED + zlib.crc32("/".join(map(str, parts)).encode())) % (2 ** 31 - 1))


# ----------------------------------------------------------------------------------------
# Data and families
# ----------------------------------------------------------------------------------------
def load_data(universe: str, variant: str = PRIMARY) -> lcm.PatternData:
    return lcm.PatternData(*E.load_patterns(universe, variant))


def edge_pattern(universe: str, variant: str = PRIMARY) -> np.ndarray:
    """Per-edge pattern index (int64); -1 marks global self-pairs (not edges)."""
    return np.asarray(E.load_edge_pattern(universe, variant, mmap=False)).astype(np.int64)


def edge_tf_index(universe: str) -> tuple[np.ndarray, list]:
    """TF index of every edge (aligned with ``edge_pattern``) and the TF list."""
    if universe == "global":
        d = E.EV_DIR / "global"
        tfs = [x.strip() for x in (d / "tfs.txt").read_text().splitlines() if x.strip()]
        genes = [x.strip() for x in (d / "genes.txt").read_text().splitlines() if x.strip()]
        idx = np.repeat(np.arange(len(tfs), dtype=np.int64), len(genes))
        return idx, tfs
    ev = pd.read_parquet(E.EV_DIR / universe / "evidence.parquet", columns=["tf"])
    tfs = sorted(ev["tf"].unique())
    pos = {t: i for i, t in enumerate(tfs)}
    return ev["tf"].map(pos).to_numpy(np.int64), tfs


def source_meta() -> dict:
    return json.loads(E.INPUT_SOURCES.read_text())["sources"]


def lineage_families(annotators) -> list:
    src = source_meta()
    return [src[a]["lineage"] for a in annotators]


def coreport_counts(data: lcm.PatternData) -> dict:
    """Pattern-weighted 2x2 co-reporting counts for every annotator pair among co-observed edges.

    n11[j,k] = edges reported by j and k; n10 = reported by j, observed-not-reported by k;
    n01 = the reverse; n00 = observed by both, reported by neither; nco = n11+n10+n01+n00."""
    obs = (data.R >= 0).astype(float)
    pos = (data.R == 1).astype(float)
    w = data.n[:, None]
    nco = obs.T @ (obs * w)
    n11 = pos.T @ (pos * w)
    n1o = pos.T @ (obs * w)            # j reported, k observed
    n10 = n1o - n11
    n01 = n1o.T - n11
    n00 = nco - n11 - n10 - n01
    return dict(nco=nco, n11=n11, n10=n10, n01=n01, n00=n00)


def phi_and_lor(cc: dict) -> tuple[np.ndarray, np.ndarray]:
    n11, n10, n01, n00 = cc["n11"], cc["n10"], cc["n01"], cc["n00"]
    den = np.sqrt((n11 + n10) * (n01 + n00) * (n11 + n01) * (n10 + n00))
    with np.errstate(invalid="ignore", divide="ignore"):
        phi = np.where(den > 0, (n11 * n00 - n10 * n01) / np.where(den > 0, den, 1.0), 0.0)
    lor = np.log((n11 + 0.5) * (n00 + 0.5) / ((n10 + 0.5) * (n01 + 0.5)))
    lor = np.where(cc["nco"] > 0, lor, np.nan)
    return phi, lor


def datadriven_families(data: lcm.PatternData, n_clusters: int = N_DD_CLUSTERS) -> tuple[list, dict]:
    """Average-linkage clustering on 1 - phi (co-reporting among co-observed edges).

    Clusters are named DD1.. by decreasing size (ties: alphabetically first member)."""
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform
    phi, lor = phi_and_lor(coreport_counts(data))
    D = 1.0 - phi
    np.fill_diagonal(D, 0.0)
    D = (D + D.T) / 2
    Z = linkage(squareform(D, checks=False), method="average")
    lab = fcluster(Z, t=n_clusters, criterion="maxclust")
    groups = {}
    for a, l in zip(data.annotators, lab):
        groups.setdefault(int(l), []).append(a)
    order = sorted(groups.values(), key=lambda g: (-len(g), sorted(g)[0]))
    name = {}
    for i, g in enumerate(order):
        for a in g:
            name[a] = f"DD{i + 1}"
    return [name[a] for a in data.annotators], dict(phi=phi, lor=lor, linkage=Z.tolist())


def families_for(grouping: str, data: lcm.PatternData) -> list:
    if grouping == "modality":
        return list(data.family)
    if grouping == "singleton":
        return list(data.annotators)
    if grouping == "lineage":
        return lineage_families(data.annotators)
    if grouping == "datadriven":
        return datadriven_families(data)[0]
    raise ValueError(grouping)


# ----------------------------------------------------------------------------------------
# Fit storage
# ----------------------------------------------------------------------------------------
def fit_dir(universe: str, variant: str, fit_id: str, root: Path = FITS) -> Path:
    return root / universe / variant / fit_id


def _prior_full(prior: dict | None) -> dict:
    p = dict(lcm.DEFAULT_PRIOR)
    if prior:
        p.update(prior)
    return p


def _atomic_dir(d: Path) -> Path:
    tmp = d.with_name(d.name + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    return tmp


def _commit_dir(tmp: Path, d: Path) -> None:
    if d.exists():
        shutil.rmtree(d)
    tmp.rename(d)


def _n_observed(data: lcm.PatternData) -> float:
    return float(data.n[(data.R >= 0).any(1)].sum())


def save_map_fit(fit: lcm.MapFit, data: lcm.PatternData, d: Path, *, fit_id: str, universe: str,
                 variant: str, prior: dict | None, n_restarts: int, seed: int, extra: dict | None = None):
    tmp = _atomic_dir(d)
    np.save(tmp / "cred.npy", np.asarray(fit.cred, dtype=np.float64))
    np.savez(tmp / "params.npz", **{k: np.asarray(v) for k, v in fit.best.items() if k != "cred"})
    pd.DataFrame(fit.restarts).to_csv(tmp / "restarts.csv", index=False)
    spec = lcm.MODEL_SPECS[fit.model_name]
    meta = dict(fit_id=fit_id, universe=universe, variant=variant, model=fit.model_name, kind="map",
                class_names=list(fit.class_names), annotators=list(data.annotators), family=list(data.family),
                families=list(data.families),
                burst_families=[data.families[f] for f in data.burst_families()] if spec["family_burst"] else [],
                prior=_prior_full(prior), prior_override=prior or {}, n_restarts=n_restarts, seed=seed,
                best_log_post=float(fit.log_post), loglik=float(fit.loglik), convergence=fit.summary(),
                runtime_s=float(fit.runtime_s), n_patterns=int(data.P), n_edges=float(data.n.sum()),
                n_observed_edges=_n_observed(data), w=np.asarray(fit.best["w"]).tolist(),
                code="src/r2/fit_sensitivity.py", created=time.strftime("%Y-%m-%d %H:%M:%S"))
    if extra:
        meta.update(extra)
    C.write_json(meta, tmp / "meta.json")
    _commit_dir(tmp, d)


def save_nuts_fit(nf: lcm.NutsFit, data: lcm.PatternData, d: Path, *, fit_id: str, universe: str, variant: str,
                  prior: dict | None, settings: dict, seed: int, extra: dict | None = None):
    tmp = _atomic_dir(d)
    c = np.asarray(nf.samples["cred"])
    cm = c.mean(0)
    np.save(tmp / "cred.npy", cm)
    np.save(tmp / "cred_mean.npy", cm)
    np.save(tmp / "cred_lo.npy", np.quantile(c, 0.025, axis=0))
    np.save(tmp / "cred_hi.npy", np.quantile(c, 0.975, axis=0))
    np.save(tmp / "cred_sd.npy", c.std(0))
    nd = c.shape[0]
    idx = np.linspace(0, nd - 1, min(MAX_DRAWS_STORED, nd)).astype(int)
    # params.npz of a NUTS fit holds posterior means (w, theta); the draws are in draws.npz
    np.savez(tmp / "params.npz", w=np.asarray(nf.samples["w"]).mean(0), theta=np.asarray(nf.samples["theta"]).mean(0))
    np.savez(tmp / "draws.npz", w=np.asarray(nf.samples["w"])[idx], theta=np.asarray(nf.samples["theta"])[idx],
             draw_index=idx)
    diag = pd.DataFrame([dict(site=k, **v) for k, v in nf.diagnostics.items()])
    diag.to_csv(tmp / "diagnostics.csv", index=False)
    meta = dict(fit_id=fit_id, universe=universe, variant=variant, model=nf.model_name, kind="nuts",
                class_names=list(nf.class_names), annotators=list(data.annotators), family=list(data.family),
                families=list(data.families), prior=_prior_full(prior), prior_override=prior or {},
                settings=settings, seed=seed, n_divergent=int(nf.n_divergent), n_draws_total=int(nd),
                n_draws_stored=int(len(idx)), r_hat_max=float(diag["r_hat_max"].max()),
                ess_bulk_min=float(diag["ess_bulk_min"].min()), ess_tail_min=float(diag["ess_tail_min"].min()),
                w_mean=np.asarray(nf.samples["w"]).mean(0).tolist(),
                w_q025=np.quantile(np.asarray(nf.samples["w"]), 0.025, axis=0).tolist(),
                w_q975=np.quantile(np.asarray(nf.samples["w"]), 0.975, axis=0).tolist(),
                runtime_s=float(nf.runtime_s), n_patterns=int(data.P), code="src/r2/fit_sensitivity.py",
                init=dict(init="random", description="numpyro init_to_uniform(radius=2) in the unconstrained "
                                                      "space (lcm.fit_nuts default)"),
                created=time.strftime("%Y-%m-%d %H:%M:%S"))
    if extra:
        meta.update(extra)
    C.write_json(meta, tmp / "meta.json")
    _commit_dir(tmp, d)


def load_fit(d: Path) -> dict:
    meta = json.loads((d / "meta.json").read_text())
    with np.load(d / "params.npz") as z:
        params = {k: np.asarray(z[k]) for k in z.files}
    out = dict(meta=meta, params=params, cred=np.load(d / "cred.npy"), dir=str(d))
    if (d / "draws.npz").exists():                      # NUTS fit (this module or canonical)
        for k in ("cred_lo", "cred_hi", "cred_sd"):
            out[k] = np.load(d / f"{k}.npy")
        with np.load(d / "draws.npz") as z:
            Wd = np.asarray(z["w"])
        out["nuts"] = nuts_summary(d, Wd, meta)
    return out


def nuts_summary(d: Path, W: np.ndarray, meta: dict) -> dict:
    """Uniform NUTS summary from the stored draws and diagnostics (works for fits written by
    this module and by the primary-fit module)."""
    diag = pd.read_csv(d / "diagnostics.csv")
    return dict(w_mean=W.mean(0).tolist(), w_q025=np.quantile(W, 0.025, axis=0).tolist(),
                w_q975=np.quantile(W, 0.975, axis=0).tolist(), r_hat_max=float(diag["r_hat_max"].max()),
                ess_bulk_min=float(diag["ess_bulk_min"].min()), ess_tail_min=float(diag["ess_tail_min"].min()),
                n_divergent=meta.get("n_divergent"), runtime_s=meta.get("runtime_s"),
                prior=meta.get("prior") if isinstance(meta.get("prior"), dict) else {})


def fit_wT(f: dict) -> float:
    return float(f["nuts"]["w_mean"][0]) if "nuts" in f else float(np.asarray(f["params"]["w"])[0])


def fit_exists(d: Path, n_restarts: int | None = None, prior: dict | None = None, family: list | None = None) -> bool:
    if not (d / "meta.json").exists() or not (d / "cred.npy").exists():
        return False
    meta = json.loads((d / "meta.json").read_text())
    if n_restarts is not None and meta.get("n_restarts") != n_restarts:
        return False
    if prior is not None and meta.get("prior") != _prior_full(prior):
        return False
    if family is not None and meta.get("family") != list(family):
        return False
    return True


def run_map(data: lcm.PatternData, model: str, d: Path, *, fit_id: str, universe: str, variant: str,
            prior: dict | None = None, n_restarts: int = N_RESTARTS, seed: int = SEED, force: bool = False,
            extra: dict | None = None) -> dict:
    if not force and fit_exists(d, n_restarts, prior, list(data.family)):
        log(f"  [skip] {d.relative_to(C.OUT)}")
        return load_fit(d)
    t = time.time()
    fit = lcm.fit_map(data, model, n_restarts=n_restarts, seed=seed, prior=prior)
    s = fit.summary()
    save_map_fit(fit, data, d, fit_id=fit_id, universe=universe, variant=variant, prior=prior,
                 n_restarts=n_restarts, seed=seed, extra=extra)
    log(f"  [fit] {d.relative_to(C.OUT)}  log_post={fit.log_post:.3f} within1e-3={s['n_within_1e3_of_best']}"
        f"/{n_restarts} conv={s['n_converged']} w_T={float(fit.best['w'][0]):.5f} ({time.time() - t:.0f}s)")
    return load_fit(d)


def primary_fit(universe: str, model: str, force: bool = False) -> dict:
    """Primary-variant MAP fit: canonical ``fits/`` directory if present, else a local fallback."""
    data = load_data(universe)
    canon = fit_dir(universe, PRIMARY, model)
    if (canon / "meta.json").exists() and (canon / "params.npz").exists():
        f = load_fit(canon)
        if f["meta"].get("annotators") == list(data.annotators) and len(f["cred"]) == data.P:
            f["source"] = "canonical"
            return f
    d = FALLBACK / universe / PRIMARY / model
    f = run_map(data, model, d, fit_id=model, universe=universe, variant=PRIMARY, force=force,
                extra=dict(note="fallback primary fit (canonical fits/ directory absent when run)"))
    f["source"] = "fallback"
    return f


def primary_nuts(universe: str, force: bool = False, allow_fit: bool = False) -> dict | None:
    """Primary Family-HB NUTS fit: canonical directory if complete, else the local fallback if it
    exists; a missing fit is run here only with ``allow_fit`` (never while building tables)."""
    data = load_data(universe)
    canon = fit_dir(universe, PRIMARY, "family_hb_nuts")
    if (canon / "meta.json").exists() and (canon / "cred_mean.npy").exists() and (canon / "draws.npz").exists():
        f = load_fit(canon)
        if len(f["cred"]) == data.P:
            f["source"] = "canonical"
            return f
    d = FALLBACK / universe / PRIMARY / "family_hb_nuts"
    if not allow_fit and not force:
        if (d / "meta.json").exists() and (d / "draws.npz").exists():
            f = load_fit(d)
            f["source"] = "fallback"
            return f
        log(f"primary NUTS fit for {universe} not available (canonical {canon} incomplete)")
        return None
    f = run_nuts(data, d, fit_id="family_hb_nuts", universe=universe, prior=None, force=force,
                 extra=dict(note="fallback primary NUTS fit (canonical fits/ directory absent when run)"))
    f["source"] = "fallback"
    return f


def run_nuts(data: lcm.PatternData, d: Path, *, fit_id: str, universe: str, prior: dict | None,
             force: bool = False, extra: dict | None = None, settings: dict = NUTS_MAIN) -> dict:
    if not force and (d / "meta.json").exists() and (d / "cred_mean.npy").exists():
        meta = json.loads((d / "meta.json").read_text())
        if meta.get("prior") == _prior_full(prior) and meta.get("settings") == settings:
            log(f"  [skip] {d}")
            return load_fit(d)
    t = time.time()
    nf = lcm.fit_nuts(data, "family_hb", seed=SEED, prior=prior, **settings)
    save_nuts_fit(nf, data, d, fit_id=fit_id, universe=universe, variant=PRIMARY, prior=prior,
                  settings=settings, seed=SEED, extra=extra)
    f = load_fit(d)
    log(f"  [nuts] {d}  div={nf.n_divergent} rhat_max={f['meta']['r_hat_max']:.3f} "
        f"ess_bulk_min={f['meta']['ess_bulk_min']:.0f} ({time.time() - t:.0f}s)")
    return f


# ----------------------------------------------------------------------------------------
# (a)-(c) MAP sensitivity fits
# ----------------------------------------------------------------------------------------
def run_fits(universes, force=False, n_restarts=N_RESTARTS):
    for u in universes:
        log(f"[fits] {u}")
        # (a) evidence variants
        for v in PS["evidence_variants_sensitivity"]:
            data = load_data(u, v)
            for m in ("family_hb", "em"):
                run_map(data, m, fit_dir(u, v, m), fit_id=m, universe=u, variant=v, force=force,
                        n_restarts=n_restarts)
        base = load_data(u)
        # (b) family groupings
        for g in GROUPINGS:
            fam = families_for(g, base)
            data = lcm.with_families(base, fam)
            fid = f"family_hb__fam_{g}"
            run_map(data, "family_hb", fit_dir(u, PRIMARY, fid), fit_id=fid, universe=u, variant=PRIMARY,
                    force=force, n_restarts=n_restarts, extra=dict(grouping=g))
        # (c) priors
        for fid, pr in PRIOR_SENS.items():
            run_map(base, "family_hb", fit_dir(u, PRIMARY, fid), fit_id=fid, universe=u, variant=PRIMARY,
                    prior=pr, force=force, n_restarts=n_restarts)


def run_nuts_priors(force=False, fit_ids=None):
    u = NUTS_UNIVERSE
    data = load_data(u)
    for fid in (fit_ids or NUTS_PRIOR_IDS):
        log(f"[nuts] {u} {fid}")
        run_nuts(data, fit_dir(u, PRIMARY, fid + "_nuts"), fit_id=fid + "_nuts", universe=u,
                 prior=PRIOR_SENS[fid], force=force)


# ----------------------------------------------------------------------------------------
# Edge-level comparison utilities
# ----------------------------------------------------------------------------------------
def weighted_midranks(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Mid-ranks of x when value x_i stands for w_i tied observations."""
    vals, inv = np.unique(x, return_inverse=True)
    wv = np.bincount(inv.reshape(-1), weights=w, minlength=len(vals))
    below = np.concatenate([[0.0], np.cumsum(wv)[:-1]])
    return (below + (wv + 1.0) / 2.0)[inv.reshape(-1)]


def weighted_spearman(x, y, w) -> float:
    x, y, w = np.asarray(x, float), np.asarray(y, float), np.asarray(w, float)
    m = w > 0
    x, y, w = x[m], y[m], w[m]
    if w.sum() <= 1:
        return float("nan")
    rx, ry = weighted_midranks(x, w), weighted_midranks(y, w)
    mx, my = np.average(rx, weights=w), np.average(ry, weights=w)
    cov = np.average((rx - mx) * (ry - my), weights=w)
    vx = np.average((rx - mx) ** 2, weights=w)
    vy = np.average((ry - my) ** 2, weights=w)
    return float(cov / np.sqrt(vx * vy)) if vx > 0 and vy > 0 else float("nan")


def weighted_auroc(score, n_pos, n_neg) -> float:
    """AUROC (ties count 1/2) for groups with a common score and n_pos / n_neg members."""
    s, npos, nneg = np.asarray(score, float), np.asarray(n_pos, float), np.asarray(n_neg, float)
    vals, inv = np.unique(s, return_inverse=True)
    inv = inv.reshape(-1)
    P_ = np.bincount(inv, weights=npos, minlength=len(vals))
    N_ = np.bincount(inv, weights=nneg, minlength=len(vals))
    below = np.concatenate([[0.0], np.cumsum(N_)[:-1]])
    tp, tn = P_.sum(), N_.sum()
    if tp == 0 or tn == 0:
        return float("nan")
    return float(np.sum(P_ * (below + 0.5 * N_)) / (tp * tn))


def bfdr_select(cred: np.ndarray, n: np.ndarray, q: float) -> np.ndarray:
    """Pattern mask of the largest set of whole credibility tie blocks with mean(1 - c) <= q."""
    cred = np.asarray(cred, float)
    vals, inv = np.unique(cred, return_inverse=True)
    inv = inv.reshape(-1)
    wn = np.bincount(inv, weights=n, minlength=len(vals))[::-1]
    wf = np.bincount(inv, weights=n * (1 - cred), minlength=len(vals))[::-1]
    cn, cf = np.cumsum(wn), np.cumsum(wf)
    with np.errstate(invalid="ignore", divide="ignore"):
        bf = np.where(cn > 0, cf / np.where(cn > 0, cn, 1), 0.0)
    ok = np.flatnonzero((bf <= q + 1e-12) & (cn > 0))
    if len(ok) == 0:
        return np.zeros(len(cred), bool)
    thr = vals[::-1][ok.max()]
    return cred >= thr


def topk_edges(c: np.ndarray, key: np.ndarray, K: int) -> np.ndarray:
    """Indices of the top-K edges by (credibility desc, fixed random key desc)."""
    K = min(K, len(c))
    thr = np.partition(c, len(c) - K)[len(c) - K]
    above = np.flatnonzero(c > thr)
    tie = np.flatnonzero(c == thr)
    need = K - len(above)
    if need > 0:
        tie = tie[np.argsort(-key[tie], kind="stable")[:need]]
        return np.concatenate([above, tie])
    return above


def jaccard_idx(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) == 0 and len(b) == 0:
        return float("nan")
    inter = len(np.intersect1d(a, b, assume_unique=True))
    return inter / (len(a) + len(b) - inter)


class EdgeView:
    """Edge-level view of a fit: credibility per edge via the variant's pattern index."""

    def __init__(self, universe: str, variant: str, cred: np.ndarray, ep_cache: dict):
        key = (universe, variant)
        if key not in ep_cache:
            R, n, _, _ = E.load_patterns(universe, variant)
            ep_cache[key] = dict(ep=edge_pattern(universe, variant), obs=(R >= 0).any(1), rep=(R == 1).any(1),
                                 n=n.astype(float))
        e = ep_cache[key]
        self.ep, self.obs_p, self.rep_p, self.n = e["ep"], e["obs"], e["rep"], e["n"]
        self.valid = self.ep >= 0
        self.cred_p = np.asarray(cred, float)
        assert len(self.cred_p) == len(self.n), (universe, variant, len(self.cred_p), len(self.n))
        epv = np.where(self.valid, self.ep, 0)
        self.c = np.where(self.valid, self.cred_p[epv], -np.inf)
        self.obs = self.valid & self.obs_p[epv]
        self.rep = self.valid & self.rep_p[epv]


def compare_views(ref: EdgeView, alt: EdgeView, tie_key: np.ndarray) -> dict:
    out = {}
    for name, mask in (("observed", ref.obs | alt.obs), ("anyreport", ref.rep | alt.rep)):
        pr = ref.ep[mask]
        pa = alt.ep[mask]
        key = pr * (len(alt.cred_p) + 1) + pa
        uk, cnt = np.unique(key, return_counts=True)
        a_, b_ = uk // (len(alt.cred_p) + 1), uk % (len(alt.cred_p) + 1)
        x, y = ref.cred_p[a_], alt.cred_p[b_]
        out[f"spearman_{name}"] = weighted_spearman(x, y, cnt)
        out[f"n_edges_{name}"] = int(cnt.sum())
        if name == "observed":
            out["mean_abs_diff_observed"] = float(np.average(np.abs(x - y), weights=cnt))
            out["max_abs_diff_observed"] = float(np.max(np.abs(x - y)))
    for K in EVAL_K:
        out[f"jaccard_top{K}"] = jaccard_idx(topk_edges(ref.c, tie_key, K), topk_edges(alt.c, tie_key, K))
    for lab, q in BFDR_Q.items():
        sr = bfdr_select(ref.cred_p, ref.n, q)
        sa = bfdr_select(alt.cred_p, alt.n, q)
        er = ref.valid & sr[np.where(ref.valid, ref.ep, 0)]
        ea = alt.valid & sa[np.where(alt.valid, alt.ep, 0)]
        inter = int(np.sum(er & ea))
        un = int(np.sum(er | ea))
        out[f"bfdr_{lab}_n_ref"] = int(er.sum())
        out[f"bfdr_{lab}_n_alt"] = int(ea.sum())
        out[f"jaccard_bfdr_{lab}"] = inter / un if un else float("nan")
    return out


def tie_key_for(universe: str, n_edges: int) -> np.ndarray:
    return np.random.default_rng([TIE_SEED, zlib.crc32(universe.encode())]).random(n_edges)


# ----------------------------------------------------------------------------------------
# Simulation with fixed (real) observation masks
# ----------------------------------------------------------------------------------------
def _decode(codes: np.ndarray, K: int) -> np.ndarray:
    out = np.empty((len(codes), K), np.int8)
    c = codes.copy()
    for j in range(K):
        out[:, j] = (c % 3) - 1
        c //= 3
    return out


def simulate_fixed_masks(data: lcm.PatternData, w, theta, seed: int, chunk: int = 400_000):
    """Simulate classes and reports for exactly the observed edges of ``data``.

    Returns (PatternData of simulated patterns, class_counts (P_sim, S)); only edges observed by
    >= 1 annotator are simulated."""
    w = np.asarray(w, float)
    w = w / w.sum()
    theta = np.asarray(theta, float)
    K, S = data.K, len(w)
    assert theta.shape == (K, S)
    obs = data.R >= 0
    keep = obs.any(1) & (data.n > 0)
    mk, inv = np.unique(obs[keep], axis=0, return_inverse=True)
    mc = np.rint(np.bincount(inv.reshape(-1), weights=data.n[keep])).astype(np.int64)
    idx = np.repeat(np.arange(len(mk)), mc)
    rng = np.random.default_rng(seed)
    pow3 = 3 ** np.arange(K, dtype=np.int64)
    thT = theta.T
    us, cs = [], []
    for s0 in range(0, len(idx), chunk):
        ii = idx[s0:s0 + chunk]
        z = rng.choice(S, size=len(ii), p=w)
        rep = rng.random((len(ii), K)) < thT[z]
        M = np.where(mk[ii], rep.astype(np.int8), np.int8(-1))
        code = (M.astype(np.int64) + 1) @ pow3
        u, c = np.unique(code * S + z, return_counts=True)
        us.append(u)
        cs.append(c)
    u_all, inv2 = np.unique(np.concatenate(us), return_inverse=True)
    cnt = np.bincount(inv2.reshape(-1), weights=np.concatenate(cs))
    code, z = u_all // S, u_all % S
    cu, pinv = np.unique(code, return_inverse=True)
    pinv = pinv.reshape(-1)
    cc = np.zeros((len(cu), S))
    np.add.at(cc, (pinv, z), cnt)
    sim = lcm.PatternData(_decode(cu, K), cc.sum(1), list(data.family), list(data.annotators))
    return sim, cc


def conditional_annotator_check(data: lcm.PatternData, w, theta) -> dict:
    """Leave-one-annotator-out conditional predictive check at plug-in parameters.

    For every pattern and annotator j observing it, pi_j = P(r_j = 1 | r_-j, Theta)
    = sum_s P(z = s | r_-j) theta_js (class posterior computed without j).  Because the
    prediction conditions on the other annotators' observed reports, the check is unaffected
    by observation masks that depend on those reports (the curated-conditional MOTIF/COEXP
    mask); masks that depend on the annotator's own reports still bias it.
    Returns per-annotator observed count O_j, expected count E_j = sum pi_j, variance
    V_j = sum pi_j (1 - pi_j) (independent-Bernoulli approximation), and the conditional
    log score CLL_j = sum [r log pi + (1 - r) log(1 - pi)] (all pattern-weighted)."""
    w = np.asarray(w, float)
    th = np.clip(np.asarray(theta, float), 1e-12, 1 - 1e-12)
    obs = (data.R >= 0)
    pos = (data.R == 1)
    lt, l1t = np.log(th), np.log1p(-th)
    lp = np.log(w)[None, :] + (obs & pos).astype(float) @ lt + (obs & ~pos).astype(float) @ l1t
    K = data.K
    O, Ex, V, CLL, N = (np.zeros(K) for _ in range(5))
    for j in range(K):
        m = obs[:, j]
        if not m.any():
            continue
        r = pos[m, j]
        lj = lp[m] - np.where(r[:, None], lt[j][None, :], l1t[j][None, :])
        post = np.exp(lj - np.logaddexp.reduce(lj, axis=1)[:, None])
        pi = np.clip(post @ th[j], 1e-12, 1 - 1e-12)
        n = data.n[m]
        O[j] = np.sum(n * r)
        Ex[j] = np.sum(n * pi)
        V[j] = np.sum(n * pi * (1 - pi))
        CLL[j] = np.sum(n * np.where(r, np.log(pi), np.log1p(-pi)))
        N[j] = n.sum()
    return dict(observed=O, expected=Ex, var=V, cll=CLL, n_obs=N)


# ----------------------------------------------------------------------------------------
# (d) TF-grouped cross-validation
# ----------------------------------------------------------------------------------------
def tf_folds(n_tfs: int, n_folds: int = N_FOLDS, seed: int = SEED) -> np.ndarray:
    """Fold label of each TF index (random permutation, near-equal groups)."""
    perm = np.random.default_rng(seed).permutation(n_tfs)
    fold = np.empty(n_tfs, np.int64)
    for f, grp in enumerate(np.array_split(perm, n_folds)):
        fold[grp] = f
    return fold


def fold_pattern_counts(ep: np.ndarray, edge_fold: np.ndarray, P: int, n_folds: int = N_FOLDS) -> np.ndarray:
    """(n_folds, P) edge counts of each pattern within each fold (self-pairs excluded)."""
    v = ep >= 0
    out = np.zeros((n_folds, P))
    for f in range(n_folds):
        m = v & (edge_fold == f)
        out[f] = np.bincount(ep[m], minlength=P)
    return out


def run_cv(universes, force=False, n_restarts=N_RESTARTS, folds=None):
    for u in universes:
        log(f"[cv] {u}")
        data = load_data(u)
        ep = edge_pattern(u)
        tf_idx, tfs = edge_tf_index(u)
        assert len(tf_idx) == len(ep), (len(tf_idx), len(ep))
        fold_of_tf = tf_folds(len(tfs))
        od = C.ensure_dir(MOD / "cv" / u)
        pd.DataFrame(dict(tf=tfs, fold=fold_of_tf)).to_csv(od / "tf_folds.csv", index=False)
        counts = fold_pattern_counts(ep, fold_of_tf[tf_idx], data.P)
        for f in (range(N_FOLDS) if folds is None else folds):
            n_test = counts[f]
            n_train = counts.sum(0) - n_test
            tr = n_train > 0
            te = n_test > 0
            train = lcm.PatternData(data.R[tr], n_train[tr], list(data.family), list(data.annotators))
            test = lcm.PatternData(data.R[te], n_test[te], list(data.family), list(data.annotators))
            n_test_obs = _n_observed(test)
            for cid, model, grouping in CONFIGS:
                out = od / f"fold{f}__{cid}.json"
                if out.exists() and not force:
                    continue
                t = time.time()
                fam = families_for(grouping, train)
                trd = lcm.with_families(train, fam)
                fit = lcm.fit_map(trd, model, n_restarts=n_restarts, seed=SEED)
                ll_test = lcm.heldout_loglik(fit.best, test)
                cchk = conditional_annotator_check(test, fit.best["w"], fit.best["theta"])
                s = fit.summary()
                rec = dict(universe=u, fold=f, config=cid, model=model, grouping=grouping, family=fam,
                           n_train_edges=float(train.n.sum()), n_train_observed=_n_observed(train),
                           n_test_edges=float(test.n.sum()), n_test_observed=n_test_obs,
                           n_test_tfs=int((fold_of_tf == f).sum()), train_loglik=float(fit.loglik),
                           train_log_post=float(fit.log_post), test_loglik=float(ll_test),
                           test_loglik_per_obs_edge=float(ll_test / n_test_obs),
                           test_cond_loglik=float(cchk["cll"].sum()), test_n_obs_cells=float(cchk["n_obs"].sum()),
                           test_cond_loglik_per_cell=float(cchk["cll"].sum() / cchk["n_obs"].sum()),
                           n_restarts=n_restarts,
                           n_within_1e3=s["n_within_1e3_of_best"], n_converged=s["n_converged"],
                           w=np.asarray(fit.best["w"]).tolist(), theta=np.asarray(fit.best["theta"]).tolist(),
                           class_names=fit.class_names, runtime_s=time.time() - t)
                C.write_json(rec, out)
                log(f"  {u} fold{f} {cid}: test ll/edge={rec['test_loglik_per_obs_edge']:.6f} "
                    f"({rec['runtime_s']:.0f}s)")


# ----------------------------------------------------------------------------------------
# (e) posterior predictive checks
# ----------------------------------------------------------------------------------------
def ppc_statistics(data: lcm.PatternData, pairs: np.ndarray | None = None) -> dict:
    """Pattern-weighted PPC statistics over observed edges."""
    obs = data.R >= 0
    pos = data.R == 1
    n = data.n
    oa = obs.any(1)
    npos = np.minimum(pos.sum(1), NREP_CAP)
    dist = np.bincount(npos[oa], weights=n[oa], minlength=NREP_CAP + 1)
    frac = dist / max(dist.sum(), 1)
    rate = (pos * n[:, None]).sum(0) / np.maximum((obs * n[:, None]).sum(0), 1)
    nobs = obs.sum(1)
    anyrep = pos.any(1)
    by_nobs = np.array([np.sum(n[(nobs >= lo) & (nobs <= hi)] * anyrep[(nobs >= lo) & (nobs <= hi)])
                        / max(np.sum(n[(nobs >= lo) & (nobs <= hi)]), 1) for lo, hi in NOBS_BINS])
    cc = coreport_counts(data)
    _, lor = phi_and_lor(cc)
    out = dict(nrep_frac=frac, rate=rate, anyrep_by_nobs=by_nobs, lor_matrix=lor, cc=cc)
    if pairs is not None:
        out["lor_pairs"] = lor[pairs[:, 0], pairs[:, 1]]
    return out


def ppc_pairs(data: lcm.PatternData) -> np.ndarray:
    """Annotator pairs j<k with co-observed edges and >= 1 report of each among them."""
    cc = coreport_counts(data)
    n1o = cc["n11"] + cc["n10"]          # j reported among co-observed
    no1 = cc["n11"] + cc["n01"]
    K = data.K
    return np.array([(j, k) for j in range(K) for k in range(j + 1, K)
                     if cc["nco"][j, k] > 0 and n1o[j, k] > 0 and no1[j, k] > 0], dtype=int).reshape(-1, 2)


def _ppc_config_fit(u: str, cid: str) -> tuple[dict, str]:
    if cid in MODELS:
        f = primary_fit(u, cid)
        return f, f["source"]
    d = fit_dir(u, PRIMARY, cid)
    if not (d / "meta.json").exists():
        raise FileNotFoundError(f"{d}: run the 'fits' step first")
    return load_fit(d), "this_module"


def subset_annotators(data: lcm.PatternData, cols) -> lcm.PatternData:
    """The same edges restricted to annotator columns ``cols`` (patterns are not re-compressed;
    every statistic used here is a pattern-weighted sum, so duplicate rows are harmless)."""
    cols = list(cols)
    return lcm.PatternData(data.R[:, cols], data.n, [data.family[j] for j in cols],
                           [data.annotators[j] for j in cols])


def lit_columns(data: lcm.PatternData) -> list:
    """Columns of the literature-curated (LIT) annotators.  Their observation masks depend only
    on TF coverage / node sets, not on other annotators' reports, so the block PPC restricted to
    them is not distorted by the report-dependent MOTIF/COEXP mask of the ``_cc`` variants."""
    return [j for j, f in enumerate(data.family) if f == "LIT"]


PPC_KEYS = ("nrep_frac", "rate", "lor_pairs", "anyrep_by_nobs")


def run_ppc(universes, n_rep=N_PPC, force=False):
    for u in universes:
        od = C.ensure_dir(MOD / "ppc" / u)
        data = load_data(u)
        pairs = ppc_pairs(data)
        real = ppc_statistics(data, pairs)
        lc = lit_columns(data)
        dlit = subset_annotators(data, lc)
        pairs_lit = ppc_pairs(dlit)
        real_lit = ppc_statistics(dlit, pairs_lit)
        np.savez(od / "real.npz", **{k: real[k] for k in PPC_KEYS}, pairs=pairs,
                 **{f"lit_{k}": real_lit[k] for k in PPC_KEYS}, lit_pairs=pairs_lit, lit_cols=np.array(lc))
        for cid, _, _ in CONFIGS:
            out = od / f"{cid}.npz"
            if out.exists() and not force:
                continue
            try:
                fit, src = _ppc_config_fit(u, cid)
            except FileNotFoundError as exc:
                log(f"[ppc] {u} {cid}: skipped ({exc})")
                continue
            w, th = fit["params"]["w"], fit["params"]["theta"]
            t = time.time()
            reps = {k: [] for k in PPC_KEYS}
            reps.update({f"lit_{k}": [] for k in PPC_KEYS})
            for r in range(n_rep):
                sim, _ = simulate_fixed_masks(data, w, th, seed=derived_seed("ppc", u, cid, r))
                st = ppc_statistics(sim, pairs)
                st_lit = ppc_statistics(subset_annotators(sim, lc), pairs_lit)
                for k in PPC_KEYS:
                    reps[k].append(st[k])
                    reps[f"lit_{k}"].append(st_lit[k])
            cchk = conditional_annotator_check(data, w, th)
            np.savez(out, **{k: np.stack(v) for k, v in reps.items()}, source=np.array(src),
                     fit_log_post=np.array(float(fit["meta"].get("best_log_post", np.nan))),
                     **{f"cond_{k}": v for k, v in cchk.items()})
            log(f"[ppc] {u} {cid}: {n_rep} replicates ({time.time() - t:.0f}s, fit source {src})")


def ppc_table(universes) -> pd.DataFrame:
    rows = []
    for u in universes:
        od = MOD / "ppc" / u
        if not (od / "real.npz").exists():
            continue
        data = load_data(u)
        real = dict(np.load(od / "real.npz"))
        pairs = real["pairs"]
        fam = np.array(data.family)
        fj, fk = fam[pairs[:, 0]], fam[pairs[:, 1]]
        within = fj == fk
        ptype = np.array(["-".join(sorted([a, b])) for a, b in zip(fj, fk)])
        for cid, model, grouping in CONFIGS:
            p = od / f"{cid}.npz"
            if not p.exists():
                continue
            rep = dict(np.load(p))
            src = str(rep.pop("source"))

            def add(stat, stratum, tobs, trep, kind="statistic"):
                trep = np.asarray(trep, float)
                m, s = float(np.mean(trep)), float(np.std(trep, ddof=1))
                rows.append(dict(universe=u, estimator=cid, model=model, grouping=grouping, statistic=stat,
                                 stratum=stratum, kind=kind, observed=float(tobs), rep_mean=m, rep_sd=s,
                                 z=(float(tobs) - m) / s if s > 1e-9 else np.nan,
                                 ppp_upper=float(np.mean(trep >= tobs)), n_rep=len(trep), fit_source=src))

            for b in range(NREP_CAP + 1):
                add("n_reporting_frac", f"{b}" if b < NREP_CAP else f">={NREP_CAP}",
                    real["nrep_frac"][b], rep["nrep_frac"][:, b])
            Lr, Ls = real["lor_pairs"], rep["lor_pairs"]
            add("lor_mean", "within_family", Lr[within].mean(), Ls[:, within].mean(1))
            add("lor_mean", "between_family", Lr[~within].mean(), Ls[:, ~within].mean(1))
            for pt in sorted(set(ptype)):
                m_ = ptype == pt
                add("lor_mean", f"pair_{pt}", Lr[m_].mean(), Ls[:, m_].mean(1))
            for b, (lo, hi) in enumerate(NOBS_BINS):
                add("anyreport_frac_by_n_observing", f"{lo}" if lo == hi else (f"{lo}-{hi}" if hi < 99 else f">={lo}"),
                    real["anyrep_by_nobs"][b], rep["anyrep_by_nobs"][:, b])
            for j, a in enumerate(data.annotators):
                add("positive_rate", a, real["rate"][j], rep["rate"][:, j])
            # leave-one-annotator-out conditional check (analytic, plug-in MAP)
            zc = []
            for j, a in enumerate(data.annotators):
                O_, E_, V_ = rep["cond_observed"][j], rep["cond_expected"][j], rep["cond_var"][j]
                z_ = (O_ - E_) / np.sqrt(V_) if V_ > 0 else np.nan
                zc.append(z_)
                rows.append(dict(universe=u, estimator=cid, model=model, grouping=grouping,
                                 statistic="conditional_positive_count", stratum=a, kind="conditional_loo",
                                 observed=O_, rep_mean=E_, rep_sd=np.sqrt(V_), z=z_, ppp_upper=np.nan, n_rep=0,
                                 fit_source=src, n_elements=int(rep["cond_n_obs"][j]),
                                 cond_loglik=rep["cond_cll"][j]))
            zc = np.array(zc)
            zc = zc[np.isfinite(zc)]
            rows.append(dict(universe=u, estimator=cid, model=model, grouping=grouping,
                             statistic="conditional_positive_count", stratum="summary", kind="summary_abs_z",
                             observed=np.nan, rep_mean=np.nan, rep_sd=np.nan, z=float(np.mean(np.abs(zc))),
                             ppp_upper=float(np.mean(np.abs(zc) > 2)), n_rep=0, fit_source=src,
                             n_elements=int(len(zc)), max_abs_z=float(np.max(np.abs(zc)))))
            rows.append(dict(universe=u, estimator=cid, model=model, grouping=grouping,
                             statistic="conditional_loglik_per_cell", stratum="all", kind="conditional_score",
                             observed=float(rep["cond_cll"].sum() / rep["cond_n_obs"].sum()), rep_mean=np.nan,
                             rep_sd=np.nan, z=np.nan, ppp_upper=np.nan, n_rep=0, fit_source=src,
                             n_elements=int(rep["cond_n_obs"].sum()), cond_loglik=float(rep["cond_cll"].sum())))
            # summaries of per-element standardized discrepancies
            def zs(obs_, reps_):
                sd_ = reps_.std(0, ddof=1)
                return np.where(sd_ > 1e-9, (obs_ - reps_.mean(0)) / np.where(sd_ > 1e-9, sd_, 1.0), np.nan)
            zr = zs(real["rate"], rep["rate"])
            zl = zs(Lr, Ls)
            zn = zs(real["nrep_frac"], rep["nrep_frac"])
            zo = zs(real["anyrep_by_nobs"], rep["anyrep_by_nobs"])
            blocks = [("positive_rate", zr), ("lor_pair_within", zl[within]),
                      ("lor_pair_between", zl[~within]), ("n_reporting_frac", zn),
                      ("anyreport_frac_by_n_observing", zo)]
            if "lit_nrep_frac" in rep and "lit_nrep_frac" in real:
                # LIT block (annotators whose masks do not depend on other annotators' reports)
                for b in range(NREP_CAP + 1):
                    add("litblock_n_reporting_frac", f"{b}" if b < NREP_CAP else f">={NREP_CAP}",
                        real["lit_nrep_frac"][b], rep["lit_nrep_frac"][:, b])
                Llr, Lls = real["lit_lor_pairs"], rep["lit_lor_pairs"]
                add("litblock_lor_mean", "LIT-LIT", Llr.mean(), Lls.mean(1))
                for b, (lo, hi) in enumerate(NOBS_BINS):
                    add("litblock_anyreport_frac_by_n_observing",
                        f"{lo}" if lo == hi else (f"{lo}-{hi}" if hi < 99 else f">={lo}"),
                        real["lit_anyrep_by_nobs"][b], rep["lit_anyrep_by_nobs"][:, b])
                blocks += [("litblock_n_reporting_frac", zs(real["lit_nrep_frac"], rep["lit_nrep_frac"])),
                           ("litblock_lor_pair", zs(Llr, Lls)),
                           ("litblock_anyreport_frac_by_n_observing",
                            zs(real["lit_anyrep_by_nobs"], rep["lit_anyrep_by_nobs"]))]
            for stat, z in blocks:
                z = z[np.isfinite(z)]
                rows.append(dict(universe=u, estimator=cid, model=model, grouping=grouping, statistic=stat,
                                 stratum="summary", kind="summary_abs_z", observed=np.nan,
                                 rep_mean=np.nan, rep_sd=np.nan, z=float(np.mean(np.abs(z))) if len(z) else np.nan,
                                 ppp_upper=float(np.mean(np.abs(z) > 2)) if len(z) else np.nan,
                                 n_rep=int(rep["rate"].shape[0]), fit_source=src,
                                 n_elements=int(len(z)), max_abs_z=float(np.max(np.abs(z))) if len(z) else np.nan))
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------------------
# (f) simulation-based recovery and calibration
# ----------------------------------------------------------------------------------------
def calibration_bins(cred_p: np.ndarray, n_p: np.ndarray, t_p: np.ndarray, n_bins: int = N_CAL_BINS) -> pd.DataFrame:
    b = np.minimum((np.asarray(cred_p) * n_bins).astype(int), n_bins - 1)
    nb = np.bincount(b, weights=n_p, minlength=n_bins)
    sp = np.bincount(b, weights=n_p * cred_p, minlength=n_bins)
    st = np.bincount(b, weights=t_p, minlength=n_bins)
    return pd.DataFrame(dict(bin=np.arange(n_bins), bin_lo=np.arange(n_bins) / n_bins,
                             bin_hi=(np.arange(n_bins) + 1) / n_bins, n_edges=nb, sum_pred=sp, n_true=st))


def _param_rows(u, est, rep, truth_theta, truth_w, est_theta, est_w, fam, est_class_names, true_class_names):
    rows = []
    fam = np.asarray(fam)
    rows.append(dict(universe=u, estimator=est, replicate=rep, param="w_T", family="all", annotator_index=-1,
                     true=float(truth_w[0]), est=float(est_w[0])))
    for jj in range(len(fam)):
        rows.append(dict(universe=u, estimator=est, replicate=rep, param="alpha", family=fam[jj], annotator_index=int(jj),
                         true=float(truth_theta[jj, 0]), est=float(est_theta[jj, 0])))
        rows.append(dict(universe=u, estimator=est, replicate=rep, param="beta", family=fam[jj], annotator_index=int(jj),
                         true=float(truth_theta[jj, 1]), est=float(est_theta[jj, 1])))
    if list(est_class_names) == list(true_class_names):
        for s, cname in enumerate(true_class_names[2:], start=2):
            rows.append(dict(universe=u, estimator=est, replicate=rep, param="w_class", family=cname, annotator_index=-1,
                             true=float(truth_w[s]), est=float(est_w[s])))
            bf = cname[2:]
            for jj in np.flatnonzero(fam == bf):
                rows.append(dict(universe=u, estimator=est, replicate=rep, param="gamma", family=bf, annotator_index=int(jj),
                                 true=float(truth_theta[jj, s]), est=float(est_theta[jj, s])))
    return rows


def run_recovery(universes, force=False, n_rep=None, n_restarts=N_RESTARTS_RECOVERY, reps=None):
    for u in universes:
        od = C.ensure_dir(MOD / "recovery" / u)
        data = load_data(u)
        truth = primary_fit(u, "family_hb")
        tw, tth = truth["params"]["w"], truth["params"]["theta"]
        tnames = truth["meta"]["class_names"]
        R_ = N_RECOVERY[u] if n_rep is None else n_rep
        for r in (range(R_) if reps is None else [r for r in reps if r < R_]):
            out = od / f"rep{r:02d}.npz"
            if out.exists() and not force:
                continue
            t = time.time()
            sim, cc = simulate_fixed_masks(data, tw, tth, seed=derived_seed("recovery", u, r))
            n_p, t_p = sim.n, cc[:, 0]
            oracle = lcm.credibility_at(dict(w=tw, theta=tth), sim)
            ll_truth = lcm.heldout_loglik(dict(w=tw, theta=tth), sim)
            res, prow, cal = {}, [], []
            for est in ("family_hb", "em"):
                fit = lcm.fit_map(sim, est, n_restarts=n_restarts, seed=derived_seed("recovery_fit", u, r, est))
                s = fit.summary()
                c = fit.cred
                res[est] = dict(auroc=weighted_auroc(c, t_p, n_p - t_p),
                                spearman_oracle=weighted_spearman(c, oracle, n_p),
                                mae_oracle=float(np.average(np.abs(c - oracle), weights=n_p)),
                                n_within_1e3=s["n_within_1e3_of_best"], n_converged=s["n_converged"],
                                log_post=float(fit.log_post), loglik_fit_minus_truth=float(fit.loglik - ll_truth),
                                runtime_s=s["runtime_s"])
                cb = calibration_bins(c, n_p, t_p)
                cb["estimator"] = est
                cal.append(cb)
                prow += _param_rows(u, est, r, tth, tw, np.asarray(fit.best["theta"]), np.asarray(fit.best["w"]),
                                    data.family, fit.class_names, tnames)
            res["oracle"] = dict(auroc=weighted_auroc(oracle, t_p, n_p - t_p), spearman_oracle=1.0, mae_oracle=0.0)
            cb = calibration_bins(oracle, n_p, t_p)
            cb["estimator"] = "oracle"
            cal.append(cb)
            calc = pd.concat(cal)
            np.savez(out, perf=json.dumps(res), params=pd.DataFrame(prow).to_json(orient="records"),
                     calib=calc.to_json(orient="records"), n_sim_edges=float(n_p.sum()),
                     n_true=float(t_p.sum()), n_patterns=sim.P)
            log(f"[recovery] {u} rep{r}: AUROC hb={res['family_hb']['auroc']:.4f} em={res['em']['auroc']:.4f} "
                f"oracle={res['oracle']['auroc']:.4f} ({time.time() - t:.0f}s)")


def map_from_start(data: lcm.PatternData, model: str, start: dict, prior: dict | None = None) -> dict:
    """One L-BFGS run of the ``lcm.fit_map`` objective started at given constrained parameters.

    ``start`` must hold every sample site of the model (e.g. the stored ``params.npz`` of a MAP
    fit of the same model and annotator layout).  Same potential, optimiser and tolerances as
    ``lcm.fit_map``, so the returned log posterior is directly comparable with its restarts."""
    import jax
    import jax.numpy as jnp
    import numpyro
    from jax.flatten_util import ravel_pytree
    from numpyro.infer.util import initialize_model, unconstrain_fn
    from scipy.optimize import minimize
    model_fn = lcm.build_model(data, lcm.MODEL_SPECS[model], prior)
    init = initialize_model(jax.random.PRNGKey(0), model_fn,
                            init_strategy=numpyro.infer.init_to_uniform(radius=2.0))
    names = list(init.param_info.z.keys())
    z0 = unconstrain_fn(model_fn, (), {}, {k: jnp.asarray(start[k]) for k in names})
    x0, unravel = ravel_pytree(z0)
    pot = jax.jit(lambda x: init.potential_fn(unravel(x)))
    grad = jax.jit(jax.grad(lambda x: init.potential_fn(unravel(x))))
    lp_start = -float(pot(x0))
    res = minimize(lambda x: float(pot(jnp.asarray(x))), np.asarray(x0, float),
                   jac=lambda x: np.asarray(grad(jnp.asarray(x))), method="L-BFGS-B",
                   options=dict(maxiter=20000, gtol=1e-6, ftol=1e-15, maxcor=30))
    sample = {k: np.asarray(v) for k, v in init.postprocess_fn(unravel(jnp.asarray(res.x))).items()}
    cred = lcm.credibility_at(sample, data)
    return dict(log_post=-float(res.fun), log_post_at_start=lp_start, converged=bool(res.success),
                n_iter=int(res.nit), w=np.asarray(sample["w"]), theta=np.asarray(sample["theta"]), cred=cred)


def run_recovery_truthstart(universes, force=False):
    """Identifiability diagnostic for the recovery replicates: refit ``family_hb`` on each simulated
    data set once more, starting from the *true* (generating) parameters, and compare its optimum
    with the best of the random restarts.  truth-start log posterior > random-start best means the
    random restarts missed the mode nearest the truth (optimisation problem); truth-start <=
    random-start best with a different credibility means the data support a competing mode at
    least as well (weak identification)."""
    for u in universes:
        od = MOD / "recovery" / u
        files = sorted(od.glob("rep*.npz"))
        if not files:
            continue
        data = load_data(u)
        truth = primary_fit(u, "family_hb")
        tp = truth["params"]
        tw, tth = tp["w"], tp["theta"]
        for f in files:
            r = int(f.stem[3:])
            out = od / f"truthstart_rep{r:02d}.json"
            if out.exists() and not force:
                continue
            t = time.time()
            sim, cc = simulate_fixed_masks(data, tw, tth, seed=derived_seed("recovery", u, r))
            n_p, t_p = sim.n, cc[:, 0]
            oracle = lcm.credibility_at(dict(w=tw, theta=tth), sim)
            ts = map_from_start(sim, "family_hb", tp)
            perf = json.loads(str(np.load(f)["perf"]))["family_hb"]
            rec = dict(universe=u, replicate=r, truth_fit_log_post=float(truth["meta"].get("best_log_post", np.nan)),
                       random_best_log_post=perf["log_post"], truthstart_log_post=ts["log_post"],
                       truthstart_log_post_at_truth=ts["log_post_at_start"],
                       truthstart_minus_random_best=ts["log_post"] - perf["log_post"],
                       truthstart_converged=ts["converged"], truthstart_n_iter=ts["n_iter"],
                       truthstart_w=ts["w"].tolist(), w_true=np.asarray(tw).tolist(),
                       truthstart_auroc=weighted_auroc(ts["cred"], t_p, n_p - t_p),
                       truthstart_spearman_oracle=weighted_spearman(ts["cred"], oracle, n_p),
                       random_best_auroc=perf["auroc"], random_best_spearman_oracle=perf["spearman_oracle"],
                       calib=calibration_bins(ts["cred"], n_p, t_p).to_dict(orient="list"),
                       runtime_s=time.time() - t)
            C.write_json(rec, out)
            log(f"[truthstart] {u} rep{r}: lp truth-start {ts['log_post']:.2f} vs random best {perf['log_post']:.2f} "
                f"(diff {rec['truthstart_minus_random_best']:+.2f}); spearman_oracle {rec['truthstart_spearman_oracle']:.3f} "
                f"vs {perf['spearman_oracle']:.3f} ({rec['runtime_s']:.0f}s)")


# ----------------------------------------------------------------------------------------
# Posterior modes of the primary Family-HB fit (Laplace mass of each MAP optimum)
# ----------------------------------------------------------------------------------------
N_RESTARTS_MODES = 50          # = the primary fit's restarts (same seed -> same starting points)
MODE_TOL = 1e-3


def map_optima(data: lcm.PatternData, model: str, n_restarts: int, seed: int, prior: dict | None = None):
    """The restarts of ``lcm.fit_map`` (identical starting points for the same seed) with every
    restart's optimum kept, plus the Laplace approximation of the posterior mass around it.

    Returns a list of dicts (restart, log_post, converged, sample, log_det_hessian, min_eig,
    laplace_log_mass) where laplace_log_mass = log p(x*) + d/2 log(2 pi) - 1/2 log det H(x*) in the
    unconstrained parametrisation of ``lcm.build_model`` (the potential includes the Jacobian)."""
    import jax
    import jax.numpy as jnp
    import numpyro
    from jax.flatten_util import ravel_pytree
    from numpyro.infer.util import initialize_model
    from scipy.optimize import minimize
    model_fn = lcm.build_model(data, lcm.MODEL_SPECS[model], prior)
    init_radius = 2.0
    rng = jax.random.PRNGKey(seed)
    init = initialize_model(rng, model_fn, init_strategy=numpyro.infer.init_to_uniform(radius=init_radius))
    flat0, unravel = ravel_pytree(init.param_info.z)
    pot = jax.jit(lambda x: init.potential_fn(unravel(x)))
    grad = jax.jit(jax.grad(lambda x: init.potential_fn(unravel(x))))
    hess = jax.jit(jax.hessian(lambda x: init.potential_fn(unravel(x))))
    keys = jax.random.split(rng, n_restarts)
    out = []
    for r in range(n_restarts):
        x0 = np.asarray(jax.random.uniform(keys[r], flat0.shape, minval=-init_radius, maxval=init_radius))
        res = minimize(lambda x: float(pot(jnp.asarray(x))), x0, jac=lambda x: np.asarray(grad(jnp.asarray(x))),
                       method="L-BFGS-B", options=dict(maxiter=20000, gtol=1e-6, ftol=1e-15, maxcor=30))
        sample = {k: np.asarray(v) for k, v in init.postprocess_fn(unravel(jnp.asarray(res.x))).items()}
        out.append(dict(restart=r, log_post=-float(res.fun), converged=bool(res.success), x=np.asarray(res.x),
                        sample=sample))
    # Laplace mass at the best restart of every distinct optimum only (Hessian is the costly part)
    order = np.argsort([-o["log_post"] for o in out], kind="mergesort")
    groups, cur = [], [int(order[0])]
    for i in order[1:]:
        if out[cur[0]]["log_post"] - out[i]["log_post"] <= MODE_TOL:
            cur.append(int(i))
        else:
            groups.append(cur)
            cur = [int(i)]
    groups.append(cur)
    d = len(flat0)
    optima = []
    for k, g in enumerate(groups):
        o = out[g[0]]
        H = np.asarray(hess(jnp.asarray(o["x"])))
        H = 0.5 * (H + H.T)
        ev = np.linalg.eigvalsh(H)
        ldet = float(np.sum(np.log(ev))) if ev.min() > 0 else float("nan")
        optima.append(dict(optimum=k, restarts=sorted(g), n_restarts=len(g), log_post=o["log_post"],
                           sample=o["sample"], n_params=d, hess_min_eig=float(ev.min()),
                           log_det_hessian=ldet,
                           laplace_log_mass=o["log_post"] + 0.5 * d * np.log(2 * np.pi) - 0.5 * ldet))
    return optima


def chain_ids(d: Path, n_draws_stored: int, meta: dict) -> np.ndarray | None:
    """Chain id of every stored NUTS draw (draws.npz 'chain' if present, else from draw_index)."""
    with np.load(d / "draws.npz") as z:
        if "chain" in z.files:
            return np.asarray(z["chain"]).astype(int)
        if "draw_index" in z.files:
            ns = (meta.get("settings") or {}).get("num_samples")
            if ns:
                return (np.asarray(z["draw_index"]) // int(ns)).astype(int)
    return None


def run_modes(universes, force=False):
    """Distinct MAP optima of the primary Family-HB fit with Laplace posterior mass, and the
    assignment of every stored NUTS chain (primary and prior-sensitivity NUTS fits) to the
    nearest optimum (by class weights)."""
    for u in universes:
        od = C.ensure_dir(MOD / "modes" / u)
        out = od / "optima.json"
        if out.exists() and not force:
            continue
        t = time.time()
        data = load_data(u)
        opt = map_optima(data, "family_hb", N_RESTARTS_MODES, SEED)
        canon = primary_fit(u, "family_hb")
        obs = (data.R >= 0).any(1)
        best_c = lcm.credibility_at(opt[0]["sample"], data)
        s_best = bfdr_select(best_c, data.n, BFDR_Q["strong"])
        lm = np.array([o["laplace_log_mass"] for o in opt])
        ok = np.isfinite(lm)
        prob = np.full(len(opt), np.nan)
        if ok.any():
            prob[ok] = np.exp(lm[ok] - np.logaddexp.reduce(lm[ok]))
        recs = []
        for o, pr in zip(opt, prob):
            c = lcm.credibility_at(o["sample"], data)
            sb = bfdr_select(c, data.n, BFDR_Q["strong"])
            inter = data.n[sb & s_best].sum()
            union = data.n[sb | s_best].sum()
            np.savez(od / f"optimum{o['optimum']}.npz", **{k: v for k, v in o["sample"].items() if k != "cred"},
                     cred=c)
            recs.append(dict(universe=u, optimum=o["optimum"], n_restarts=o["n_restarts"],
                             restarts=" ".join(map(str, o["restarts"])), log_post=o["log_post"],
                             gap_to_best=opt[0]["log_post"] - o["log_post"], n_params=o["n_params"],
                             hess_min_eig=o["hess_min_eig"], log_det_hessian=o["log_det_hessian"],
                             laplace_log_mass=o["laplace_log_mass"], laplace_mode_prob=float(pr),
                             w=np.asarray(o["sample"]["w"]).tolist(), w_T=float(o["sample"]["w"][0]),
                             expected_true_observed=float(np.sum(data.n[obs] * c[obs])),
                             n_bfdr_strong=float(data.n[sb].sum()),
                             jaccard_bfdr_strong_vs_best=float(inter / union) if union else np.nan,
                             spearman_vs_best_observed=weighted_spearman(c[obs], best_c[obs], data.n[obs]),
                             max_abs_cred_diff_vs_canonical=float(np.max(np.abs(c - canon["cred"])))
                             if o["optimum"] == 0 else np.nan))
        C.write_json(dict(optima=recs, runtime_s=time.time() - t,
                          canonical_best_log_post=float(canon["meta"].get("best_log_post", np.nan))), out)
        log(f"[modes] {u}: {len(opt)} optima; log_post {[round(o['log_post'], 2) for o in opt]}; "
            f"Laplace probs {np.round(prob, 4).tolist()} ({time.time() - t:.0f}s)")


def nuts_chain_rows(u: str) -> list:
    """Every stored NUTS chain of the primary and prior-sensitivity Family-HB fits: chain-mean class
    weights, nearest MAP optimum (L1 distance of log class weights) and the weighted Spearman of the
    chain's mean credibility (over <= 50 thinned draws) with the best MAP optimum."""
    od = MOD / "modes" / u
    files = sorted(od.glob("optimum*.npz"), key=lambda f: int(f.stem[7:]))
    if not files:
        return []
    data = load_data(u)
    obs = (data.R >= 0).any(1)
    W_opt = np.stack([np.load(f)["w"] for f in files])
    best_c = np.load(files[0])["cred"]
    rows = []
    nuts_dirs = [("family_hb_nuts", fit_dir(u, PRIMARY, "family_hb_nuts"))] + \
        [(fid + "_nuts", fit_dir(u, PRIMARY, fid + "_nuts")) for fid in NUTS_PRIOR_IDS]
    for fid, d in nuts_dirs:
        if not ((d / "draws.npz").exists() and (d / "meta.json").exists()):
            continue
        meta = json.loads((d / "meta.json").read_text())
        with np.load(d / "draws.npz") as z:
            Wd, THd = np.asarray(z["w"]), np.asarray(z["theta"])
        ch = chain_ids(d, len(Wd), meta)
        if ch is None or Wd.shape[1] != W_opt.shape[1]:
            continue
        for c_ in np.unique(ch):
            m = ch == c_
            wm = Wd[m].mean(0)
            near = int(np.argmin(np.abs(np.log(W_opt) - np.log(wm)[None, :]).sum(1)))
            sel = np.flatnonzero(m)[::max(1, int(m.sum()) // 50)]
            cm = np.mean([lcm.credibility_at(dict(w=Wd[i], theta=THd[i]), data) for i in sel], axis=0)
            rows.append(dict(universe=u, fit_id=fid, chain=int(c_), n_draws=int(m.sum()), w_mean=wm.tolist(),
                             w_T_mean=float(wm[0]), nearest_optimum=near,
                             spearman_vs_best_observed=weighted_spearman(cm[obs], best_c[obs], data.n[obs])))
    return rows


def modes_tables(universes) -> tuple[pd.DataFrame, pd.DataFrame]:
    opt, ch = [], []
    for u in universes:
        p = MOD / "modes" / u / "optima.json"
        if p.exists():
            z = json.loads(p.read_text())
            for r in z["optima"]:
                r = dict(r)
                w = r.pop("w")
                r.update({f"w_{i}": v for i, v in enumerate(w)})
                opt.append(r)
        for r in nuts_chain_rows(u):
            w = r.pop("w_mean")
            r.update({f"w_mean_{i}": v for i, v in enumerate(w)})
            ch.append(r)
    return pd.DataFrame(opt), pd.DataFrame(ch)


def select_mode_chains(W: np.ndarray, TH: np.ndarray, chain: np.ndarray, data: lcm.PatternData,
                       n_eval: int = 50, tol: float = 2.0) -> tuple[list, dict]:
    """Label-free choice of the chains that sampled the highest-density mode.

    Per chain: mean collapsed log-likelihood over <= ``n_eval`` thinned draws.  Chains within
    ``tol`` nats of the best chain's mean are kept (chains in the same mode differ by Monte Carlo
    noise only, separate modes by much more).  Uses no truth, so it mimics what an analyst
    would do with a non-mixing posterior (keep the dominant mode)."""
    ll = {}
    for c_ in np.unique(chain):
        idx = np.flatnonzero(chain == c_)
        idx = idx[np.linspace(0, len(idx) - 1, min(n_eval, len(idx))).astype(int)]
        ll[int(c_)] = float(np.mean([lcm.heldout_loglik(dict(w=W[i], theta=TH[i]), data) for i in idx]))
    best = max(ll.values())
    return sorted(c_ for c_, v in ll.items() if v >= best - tol), ll


def run_recovery_nuts(force=False, n_rep=N_RECOVERY_NUTS, reps=None):
    u = NUTS_UNIVERSE
    od = C.ensure_dir(MOD / "recovery_nuts" / u)
    data = load_data(u)
    truth = primary_fit(u, "family_hb")
    tw, tth = truth["params"]["w"], truth["params"]["theta"]
    for r in (range(n_rep) if reps is None else [r for r in reps if r < n_rep]):
        out = od / f"rep{r:02d}.json"
        if out.exists() and not force:
            continue
        t = time.time()
        sim, cc = simulate_fixed_masks(data, tw, tth, seed=derived_seed("recovery", u, r))   # = MAP replicate r
        n_p, t_p = sim.n, cc[:, 0]
        oracle = lcm.credibility_at(dict(w=tw, theta=tth), sim)
        nf = lcm.fit_nuts(sim, "family_hb", seed=derived_seed("recovery_nuts", u, r), **NUTS_RECOVERY)
        W, TH = np.asarray(nf.samples["w"]), np.asarray(nf.samples["theta"])
        chain = np.repeat(np.arange(NUTS_RECOVERY["num_chains"]), NUTS_RECOVERY["num_samples"])
        keep, ll_chain = select_mode_chains(W, TH, chain, sim)
        sel = np.isin(chain, keep)

        def summ(m):
            lo_w, hi_w = np.quantile(W[m], [0.025, 0.975], axis=0)
            lo_t, hi_t = np.quantile(TH[m], [0.025, 0.975], axis=0)
            ii = np.flatnonzero(m)
            ii = ii[np.linspace(0, len(ii) - 1, min(200, len(ii))).astype(int)]
            cm = np.mean([lcm.credibility_at(dict(w=W[i], theta=TH[i]), sim) for i in ii], axis=0)
            return dict(w_lo=lo_w.tolist(), w_hi=hi_w.tolist(), w_mean=W[m].mean(0).tolist(),
                        theta_lo=lo_t.tolist(), theta_hi=hi_t.tolist(), theta_mean=TH[m].mean(0).tolist(),
                        auroc=weighted_auroc(cm, t_p, n_p - t_p), spearman_oracle=weighted_spearman(cm, oracle, n_p))

        rec = dict(universe=u, replicate=r, n_divergent=int(nf.n_divergent),
                   r_hat_max=float(max(v["r_hat_max"] for v in nf.diagnostics.values())),
                   ess_bulk_min=float(min(v["ess_bulk_min"] for v in nf.diagnostics.values())),
                   class_names=nf.class_names, family=list(data.family),
                   w_true=np.asarray(tw).tolist(), theta_true=np.asarray(tth).tolist(), settings=NUTS_RECOVERY,
                   chain_loglik_mean=ll_chain, chain_w_T_mean={int(c_): float(W[chain == c_, 0].mean())
                                                              for c_ in np.unique(chain)},
                   mode_chains=keep, **summ(np.ones(len(W), bool)),
                   mode={k: v for k, v in summ(sel).items()}, runtime_s=time.time() - t)
        idx = np.linspace(0, len(W) - 1, min(MAX_DRAWS_STORED, len(W))).astype(int)
        np.savez(od / f"rep{r:02d}_draws.npz", w=W[idx], theta=TH[idx], chain=chain[idx])
        C.write_json(rec, out)
        log(f"[recovery-nuts] {u} rep{r}: div={rec['n_divergent']} rhat={rec['r_hat_max']:.3f} mode chains {keep} "
            f"w_T in CrI pooled={rec['w_lo'][0] <= tw[0] <= rec['w_hi'][0]} "
            f"mode={rec['mode']['w_lo'][0] <= tw[0] <= rec['mode']['w_hi'][0]} ({rec['runtime_s']:.0f}s)")


def recovery_tables(universes) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows, cal_rows, rep_rows = [], [], []
    for u in universes:
        od = MOD / "recovery" / u
        files = sorted(od.glob("rep*.npz"))
        if not files:
            continue
        perf, params, cals = [], [], []
        for f in files:
            z = np.load(f, allow_pickle=False)
            r = int(f.stem[3:])
            for est, d in json.loads(str(z["perf"])).items():
                perf.append(dict(estimator=est, replicate=r, **d))
            params.append(pd.read_json(io.StringIO(str(z["params"]))))
            cb = pd.read_json(io.StringIO(str(z["calib"])))
            cb["replicate"] = r
            cals.append(cb)
            tsf = od / f"truthstart_rep{r:02d}.json"
            if tsf.exists():
                tsr = json.loads(tsf.read_text())
                if "calib" in tsr:
                    cb2 = pd.DataFrame(tsr["calib"])
                    cb2["estimator"] = "family_hb_truthstart"
                    cb2["replicate"] = r
                    cals.append(cb2)
        perf = pd.DataFrame(perf)
        params = pd.concat(params, ignore_index=True)
        cals = pd.concat(cals, ignore_index=True)
        nrep = len(files)
        # per-replicate table (+ truth-start identifiability diagnostic when available)
        wt = params[params["param"] == "w_T"].set_index(["estimator", "replicate"])["est"]
        ts = {}
        for f in sorted(od.glob("truthstart_rep*.json")):
            rec = json.loads(f.read_text())
            ts[int(rec["replicate"])] = rec
        for _, pr in perf.iterrows():
            r = int(pr["replicate"])
            row = dict(universe=u, replicate=r, estimator=pr["estimator"], auroc=pr.get("auroc"),
                       spearman_oracle=pr.get("spearman_oracle"), mae_oracle=pr.get("mae_oracle"),
                       ece=np.nan, w_T_est=float(wt.get((pr["estimator"], r), np.nan)),
                       w_T_true=float(params.loc[params["param"] == "w_T", "true"].iloc[0]),
                       log_post=pr.get("log_post"), n_within_1e3=pr.get("n_within_1e3"),
                       loglik_fit_minus_truth=pr.get("loglik_fit_minus_truth"))
            if pr["estimator"] == "family_hb" and r in ts:
                t_ = ts[r]
                row.update(truthstart_log_post_minus_random_best=t_["truthstart_minus_random_best"],
                           truthstart_spearman_oracle=t_["truthstart_spearman_oracle"],
                           truthstart_auroc=t_["truthstart_auroc"], truthstart_w_T=t_["truthstart_w"][0])
            rep_rows.append(row)
        if ts:
            d_lp = np.array([ts[r]["truthstart_minus_random_best"] for r in sorted(ts)])
            d_sp = np.array([ts[r]["truthstart_spearman_oracle"] - ts[r]["random_best_spearman_oracle"]
                             for r in sorted(ts)])
            other = d_sp > 0.05          # random-start MAP is a different mode than the truth-start one
            base = dict(universe=u, kind="identifiability", estimator="family_hb", family="all",
                        n_replicates=len(ts), n_values=len(ts))
            rows.append(dict(base, quantity="n_replicates_map_in_competing_mode", value_mean=float(other.sum()),
                             value_sd=np.nan))
            rows.append(dict(base, quantity="truthstart_minus_random_best_log_post_competing",
                             value_mean=float(d_lp[other].mean()) if other.any() else np.nan,
                             value_sd=float(d_lp[other].std(ddof=1)) if other.sum() > 1 else np.nan))
            rows.append(dict(base, quantity="n_replicates_truthstart_higher_log_post",
                             value_mean=float((d_lp > 1e-3).sum()), value_sd=np.nan))
            rows.append(dict(base, quantity="spearman_oracle_truthstart_mode",
                             value_mean=float(np.mean([ts[r]["truthstart_spearman_oracle"] for r in ts])),
                             value_sd=float(np.std([ts[r]["truthstart_spearman_oracle"] for r in ts], ddof=1))
                             if len(ts) > 1 else np.nan))
            if other.any():
                rows.append(dict(base, quantity="spearman_oracle_competing_mode",
                                 value_mean=float(np.mean([ts[r]["random_best_spearman_oracle"]
                                                           for r, o in zip(sorted(ts), other) if o])),
                                 value_sd=np.nan))
                rows.append(dict(base, quantity="w_T_competing_mode",
                                 value_mean=float(np.mean([wt.get(("family_hb", r), np.nan)
                                                           for r, o in zip(sorted(ts), other) if o])),
                                 value_sd=np.nan))
        # parameter recovery by family
        params["err"] = params["est"] - params["true"]
        lg = lambda x: np.log(np.clip(x, 1e-12, 1 - 1e-12) / (1 - np.clip(x, 1e-12, 1 - 1e-12)))  # noqa: E731
        params["err_logit"] = lg(params["est"]) - lg(params["true"])
        for (est, par, fam), g in params.groupby(["estimator", "param", "family"]):
            rows.append(dict(universe=u, kind="parameter", estimator=est, quantity=par, family=fam,
                             n_replicates=nrep, n_values=len(g), true_mean=g["true"].mean(), est_mean=g["est"].mean(),
                             bias=g["err"].mean(), rmse=float(np.sqrt(np.mean(g["err"] ** 2))),
                             bias_logit=g["err_logit"].mean(), rmse_logit=float(np.sqrt(np.mean(g["err_logit"] ** 2))),
                             value_mean=np.nan, value_sd=np.nan))
        for par in ("alpha", "beta", "gamma"):
            g0 = params[params["param"] == par]
            for est, g in g0.groupby("estimator"):
                rows.append(dict(universe=u, kind="parameter", estimator=est, quantity=par, family="all",
                                 n_replicates=nrep, n_values=len(g), true_mean=g["true"].mean(),
                                 est_mean=g["est"].mean(), bias=g["err"].mean(),
                                 rmse=float(np.sqrt(np.mean(g["err"] ** 2))), bias_logit=g["err_logit"].mean(),
                                 rmse_logit=float(np.sqrt(np.mean(g["err_logit"] ** 2))),
                                 value_mean=np.nan, value_sd=np.nan))
        # credibility performance
        ece = {}
        for (est, r), g in cals.groupby(["estimator", "replicate"]):
            nb = g["n_edges"].to_numpy()
            mp = g["sum_pred"].to_numpy() / np.maximum(nb, 1)
            tr = g["n_true"].to_numpy() / np.maximum(nb, 1)
            ece[(est, r)] = float(np.sum(nb * np.abs(mp - tr)) / nb.sum())
        perf["ece"] = [ece[(e, r)] for e, r in zip(perf["estimator"], perf["replicate"])]
        for row in rep_rows:
            if row["universe"] == u:
                row["ece"] = ece.get((row["estimator"], row["replicate"]), np.nan)
        for est, g in perf.groupby("estimator"):
            for q in ("auroc", "spearman_oracle", "mae_oracle", "ece", "n_within_1e3", "n_converged",
                      "loglik_fit_minus_truth"):
                if q not in g or g[q].isna().all():
                    continue
                rows.append(dict(universe=u, kind="credibility", estimator=est, quantity=q, family="all",
                                 n_replicates=nrep, n_values=int(g[q].notna().sum()), value_mean=g[q].mean(),
                                 value_sd=g[q].std(ddof=1)))
        # pooled calibration curve
        for (est, b), g in cals.groupby(["estimator", "bin"]):
            nb = g["n_edges"].sum()
            cal_rows.append(dict(universe=u, estimator=est, bin=int(b), bin_lo=g["bin_lo"].iloc[0],
                                 bin_hi=g["bin_hi"].iloc[0], n_replicates=nrep, n_edges_total=nb,
                                 n_edges_per_replicate=nb / nrep,
                                 mean_predicted=g["sum_pred"].sum() / nb if nb else np.nan,
                                 empirical_true_rate=g["n_true"].sum() / nb if nb else np.nan,
                                 true_rate_sd_across_replicates=float(np.std(
                                     (g["n_true"] / g["n_edges"].replace(0, np.nan)).dropna(), ddof=1))
                                 if (g["n_edges"] > 0).sum() > 1 else np.nan))
    # NUTS coverage (pooled chains, and the chains of the label-free selected mode)
    nd = MOD / "recovery_nuts" / NUTS_UNIVERSE
    files = sorted(nd.glob("rep*.json")) if nd.exists() else []
    recs = [json.loads(f.read_text()) for f in files]
    recs = [r_ for r_ in recs if "mode" in r_]          # current format only
    if recs:
        fam = np.array(recs[0]["family"])
        for scope in ("pooled", "selected_mode"):
            cov = {"w_T": [], "alpha": {}, "beta": {}}
            width = {"w_T": [], "alpha": {}, "beta": {}}
            for rec in recs:
                src = rec if scope == "pooled" else rec["mode"]
                wt, wl, wh = rec["w_true"][0], src["w_lo"][0], src["w_hi"][0]
                cov["w_T"].append(wl <= wt <= wh)
                width["w_T"].append(wh - wl)
                th = np.array(rec["theta_true"])
                lo, hi = np.array(src["theta_lo"]), np.array(src["theta_hi"])
                for par, col in (("alpha", 0), ("beta", 1)):
                    for jj in range(len(fam)):
                        cov[par].setdefault(fam[jj], []).append(lo[jj, col] <= th[jj, col] <= hi[jj, col])
                        width[par].setdefault(fam[jj], []).append(hi[jj, col] - lo[jj, col])
            base = dict(universe=NUTS_UNIVERSE, kind="nuts_coverage", estimator=f"family_hb_nuts_{scope}",
                        n_replicates=len(recs))
            rows.append(dict(base, quantity="w_T_coverage95", family="all", n_values=len(cov["w_T"]),
                             value_mean=float(np.mean(cov["w_T"])), value_sd=np.nan,
                             ci_width_mean=float(np.mean(width["w_T"]))))
            for par in ("alpha", "beta"):
                allc, allw = [], []
                for fm in sorted(cov[par]):
                    allc += cov[par][fm]
                    allw += width[par][fm]
                    rows.append(dict(base, quantity=f"{par}_coverage95", family=fm, n_values=len(cov[par][fm]),
                                     value_mean=float(np.mean(cov[par][fm])), value_sd=np.nan,
                                     ci_width_mean=float(np.mean(width[par][fm]))))
                rows.append(dict(base, quantity=f"{par}_coverage95", family="all", n_values=len(allc),
                                 value_mean=float(np.mean(allc)), value_sd=np.nan, ci_width_mean=float(np.mean(allw))))
            for q in ("auroc", "spearman_oracle"):
                v = [(rec if scope == "pooled" else rec["mode"])[q] for rec in recs]
                rows.append(dict(base, kind="nuts_credibility", quantity=q, family="all", n_values=len(v),
                                 value_mean=float(np.mean(v)), value_sd=float(np.std(v, ddof=1)) if len(v) > 1 else np.nan))
        base = dict(universe=NUTS_UNIVERSE, kind="nuts_diagnostics", estimator="family_hb_nuts", n_replicates=len(recs))
        for q in ("n_divergent", "r_hat_max", "ess_bulk_min"):
            v = [rec[q] for rec in recs]
            rows.append(dict(base, quantity=q, family="all", n_values=len(v), value_mean=float(np.mean(v)),
                             value_sd=np.nan, value_min=float(np.min(v)), value_max=float(np.max(v))))
        v = [len(rec["mode_chains"]) for rec in recs]
        rows.append(dict(base, quantity="n_chains_selected_mode", family="all", n_values=len(v),
                         value_mean=float(np.mean(v)), value_sd=np.nan, value_min=float(np.min(v)),
                         value_max=float(np.max(v))))
    return pd.DataFrame(rows), pd.DataFrame(cal_rows), pd.DataFrame(rep_rows)


# ----------------------------------------------------------------------------------------
# Tables
# ----------------------------------------------------------------------------------------
def _write(df: pd.DataFrame, name: str) -> Path:
    p = TABLES / f"{PREFIX}{name}.csv"
    C.ensure_dir(p.parent)
    df.to_csv(p, index=False, float_format="%.6g")
    log(f"wrote {p.relative_to(C.SP)} ({len(df)} rows)")
    return p


def exchangeable_t_burst(meta: dict) -> bool:
    """True when T and a family-burst class are exchangeable under the model: a Family-HB fit with
    a single family (every annotator in it) has classes T, N, B_f in which T and B_f both raise
    reporting of every annotator with identical priors, so the likelihood and prior are invariant
    to swapping them (the restarts then split between the two labellings)."""
    return (meta.get("model") == "family_hb" and len(meta.get("families", [])) == 1
            and len(meta.get("class_names", [])) == 3)


def class_posterior(params: dict, data: lcm.PatternData, s: int) -> np.ndarray:
    """Per-pattern posterior probability of latent class ``s`` at given (w, theta)."""
    w = np.asarray(params["w"], float)
    th = np.clip(np.asarray(params["theta"], float), 1e-12, 1 - 1e-12)
    obs = (data.R >= 0).astype(float)
    pos = (data.R == 1).astype(float)
    lj = np.log(w)[None, :] + (obs * pos) @ np.log(th) + (obs * (1 - pos)) @ np.log1p(-th)
    return np.exp(lj[:, s] - np.logaddexp.reduce(lj, axis=1))


def sensitivity_rows(universes) -> pd.DataFrame:
    rows = []
    for u in universes:
        ep_cache = {}
        ref_hb = primary_fit(u, "family_hb")
        ref_em = primary_fit(u, "em")
        v_hb = EdgeView(u, PRIMARY, ref_hb["cred"], ep_cache)
        v_em = EdgeView(u, PRIMARY, ref_em["cred"], ep_cache)
        tie_key = tie_key_for(u, len(v_hb.ep))
        items = []   # (variant, fit_id, kind, fitdict)
        for m in MODELS:
            if m == "family_hb":
                continue
            f = primary_fit(u, m)
            items.append((PRIMARY, m, "primary_estimator", f))
        for v in PS["evidence_variants_sensitivity"]:
            for m in ("family_hb", "em"):
                d = fit_dir(u, v, m)
                if (d / "meta.json").exists():
                    items.append((v, m, "evidence_variant", load_fit(d)))
        for g in GROUPINGS:
            d = fit_dir(u, PRIMARY, f"family_hb__fam_{g}")
            if (d / "meta.json").exists():
                items.append((PRIMARY, f"family_hb__fam_{g}", "family_grouping", load_fit(d)))
        for fid in PRIOR_SENS:
            d = fit_dir(u, PRIMARY, fid)
            if (d / "meta.json").exists():
                items.append((PRIMARY, fid, "prior", load_fit(d)))
        if u == NUTS_UNIVERSE:
            pn = None
            for fid in NUTS_PRIOR_IDS:
                d = fit_dir(u, PRIMARY, fid + "_nuts")
                if (d / "meta.json").exists():
                    items.append((PRIMARY, fid + "_nuts", "prior_nuts", load_fit(d)))
            try:
                pn = primary_nuts(u) if any(i[2] == "prior_nuts" for i in items) else None
            except Exception as exc:  # pragma: no cover
                log(f"primary NUTS unavailable: {exc}")
            if pn is not None:
                items.append((PRIMARY, "family_hb_nuts", "primary_nuts", pn))
        for variant, fid, kind, f in items:
            alt = EdgeView(u, variant, f["cred"], ep_cache)
            refs = [("family_hb", v_hb, ref_hb["source"])]
            if fid == "em" and kind == "evidence_variant":
                refs.append(("em", v_em, ref_em["source"]))
            if kind == "prior_nuts" and pn is not None:
                refs.append(("family_hb_nuts", EdgeView(u, PRIMARY, pn["cred"], ep_cache), pn["source"]))
            meta = f["meta"]
            conv = meta.get("convergence", {})
            for rname, rv, rsrc in refs:
                cmp_ = compare_views(rv, alt, tie_key)
                rows.append(dict(universe=u, variant=variant, fit_id=fid, analysis=kind, reference=rname,
                                 reference_source=rsrc, model=meta.get("model"), n_annotators=len(meta["annotators"]),
                                 families=";".join(meta.get("families", [])),
                                 classes=";".join(meta.get("class_names", [])),
                                 best_log_post=meta.get("best_log_post"), loglik=meta.get("loglik"),
                                 n_restarts=meta.get("n_restarts"),
                                 n_within_1e3_of_best=conv.get("n_within_1e3_of_best"),
                                 # fit_primary renamed n_converged to n_lbfgs_ftol_stop (same count; its
                                 # fit_convergence.csv keeps the column name n_converged)
                                 n_converged=conv.get("n_converged", conv.get("n_lbfgs_ftol_stop")),
                                 log_post_range=conv.get("log_post_range"),
                                 w_T=fit_wT(f),
                                 w_T_ref=fit_wT(ref_hb) if rname == "family_hb" else
                                 (fit_wT(ref_em) if rname == "em" else fit_wT(pn)),
                                 n_divergent=f.get("nuts", {}).get("n_divergent"),
                                 r_hat_max=f.get("nuts", {}).get("r_hat_max"),
                                 ess_bulk_min=f.get("nuts", {}).get("ess_bulk_min"),
                                 max_abs_cred_diff_top_restarts=conv.get("max_abs_cred_diff_top"),
                                 t_burst_exchangeable=exchangeable_t_burst(meta), **cmp_))
                if exchangeable_t_burst(meta) and rname == "family_hb":
                    # T and the single burst class are exchangeable: compare the other labelling too
                    data_v = load_data(u, variant)
                    cs = class_posterior(f["params"], data_v, 2)
                    cmp_s = compare_views(rv, EdgeView(u, variant, cs, ep_cache), tie_key)
                    rows[-1].update({f"swapped_{k}": v for k, v in cmp_s.items()
                                     if k.startswith(("spearman", "jaccard"))})
        log(f"[tables] sensitivity comparisons done for {u}")
    return pd.DataFrame(rows)


def grouping_tables(universes) -> tuple[pd.DataFrame, pd.DataFrame]:
    mem, assoc = [], []
    for u in universes:
        data = load_data(u)
        dd, info = datadriven_families(data)
        lin = lineage_families(data.annotators)
        cc = coreport_counts(data)
        for j, a in enumerate(data.annotators):
            mem.append(dict(universe=u, annotator=a, modality=data.family[j], lineage=lin[j], datadriven=dd[j],
                            singleton=a))
        phi, lor = info["phi"], info["lor"]
        for j in range(data.K):
            for k in range(j + 1, data.K):
                assoc.append(dict(universe=u, annotator_j=data.annotators[j], annotator_k=data.annotators[k],
                                  modality_j=data.family[j], modality_k=data.family[k],
                                  n_coobserved=cc["nco"][j, k], n11=cc["n11"][j, k], n10=cc["n10"][j, k],
                                  n01=cc["n01"][j, k], phi=phi[j, k], log_odds_ratio=lor[j, k],
                                  distance=1 - phi[j, k]))
    return pd.DataFrame(mem), pd.DataFrame(assoc)


def prior_param_table(universes) -> pd.DataFrame:
    rows = []
    for u in universes:
        fits = [("family_hb", primary_fit(u, "family_hb"))]
        for fid in list(PRIOR_SENS) + [f"family_hb__fam_{g}" for g in GROUPINGS]:
            d = fit_dir(u, PRIMARY, fid)
            if (d / "meta.json").exists():
                fits.append((fid, load_fit(d)))
        for v in PS["evidence_variants_sensitivity"]:
            d = fit_dir(u, v, "family_hb")
            if (d / "meta.json").exists():
                fits.append((f"{v}/family_hb", load_fit(d)))
        for fid, f in fits:
            p = f["params"]
            names = f["meta"]["class_names"]
            for s, cname in enumerate(names):
                rows.append(dict(universe=u, fit=fid, param=f"w_{cname}", family="", value=float(p["w"][s])))
            for q in ("b", "a", "g"):
                for h in ("tau", "omega", "mu0"):
                    k = f"{h}_{q}"
                    if k in p:
                        rows.append(dict(universe=u, fit=fid, param=k, family="", value=float(p[k])))
                if f"muf_{q}" in p:
                    for i, fam in enumerate(f["meta"]["families"]):
                        rows.append(dict(universe=u, fit=fid, param=f"muf_{q}", family=fam,
                                         value=float(np.asarray(p[f"muf_{q}"])[i])))
            th = np.asarray(p["theta"])
            fam = np.asarray(f["meta"]["family"])
            for fm in sorted(set(fam)):
                m = fam == fm
                rows.append(dict(universe=u, fit=fid, param="alpha_mean", family=fm, value=float(th[m, 0].mean())))
                rows.append(dict(universe=u, fit=fid, param="beta_mean", family=fm, value=float(th[m, 1].mean())))
    return pd.DataFrame(rows)


def _cond_cv_cols(g: pd.DataFrame, ref: pd.DataFrame, common) -> dict:
    """Held-out leave-one-annotator-out conditional log score (pseudo-likelihood) columns."""
    if "test_cond_loglik_per_cell" not in g:
        return {}
    x = g["test_cond_loglik_per_cell"]
    d = (g.loc[common, "test_cond_loglik"] - ref.loc[common, "test_cond_loglik"]) / g.loc[common, "test_n_obs_cells"]
    k = len(x)
    return dict(heldout_cond_ll_per_cell_mean=x.mean(), heldout_cond_ll_per_cell_se=x.std(ddof=1) / np.sqrt(k),
                cond_diff_vs_family_hb_per_cell_mean=d.mean(),
                cond_diff_vs_family_hb_per_cell_se=d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else np.nan,
                cond_diff_vs_family_hb_total_nats=float((g.loc[common, "test_cond_loglik"]
                                                         - ref.loc[common, "test_cond_loglik"]).sum()))


def cv_tables(universes) -> tuple[pd.DataFrame, pd.DataFrame]:
    recs = []
    for u in universes:
        for p in sorted((MOD / "cv" / u).glob("fold*__*.json")):
            r = json.loads(p.read_text())
            r.pop("family", None)
            r["w_T"] = r.pop("w")[0]
            r.pop("class_names", None)
            r.pop("theta", None)
            recs.append(r)
    folds = pd.DataFrame(recs)
    if folds.empty:
        return folds, folds
    rows = []
    for u, gu in folds.groupby("universe"):
        ref = gu[gu["config"] == "family_hb"].set_index("fold")
        for cid, g in gu.groupby("config"):
            g = g.set_index("fold").sort_index()
            x = g["test_loglik_per_obs_edge"]
            common = g.index.intersection(ref.index)
            dpe = (g.loc[common, "test_loglik"] - ref.loc[common, "test_loglik"]) / g.loc[common, "n_test_observed"]
            dtot = g.loc[common, "test_loglik"] - ref.loc[common, "test_loglik"]
            k = len(x)
            rows.append(dict(universe=u, config=cid, model=g["model"].iloc[0], grouping=g["grouping"].iloc[0],
                             n_folds=k, heldout_ll_per_edge_mean=x.mean(), heldout_ll_per_edge_se=x.std(ddof=1) / np.sqrt(k),
                             diff_vs_family_hb_per_edge_mean=dpe.mean(),
                             diff_vs_family_hb_per_edge_se=dpe.std(ddof=1) / np.sqrt(len(dpe)) if len(dpe) > 1 else np.nan,
                             diff_vs_family_hb_total_nats=dtot.sum(),
                             n_folds_better_than_family_hb=int((dtot > 0).sum()),
                             heldout_ll_total=g["test_loglik"].sum(), n_test_observed_total=g["n_test_observed"].sum(),
                             train_ll_per_edge_mean=(g["train_loglik"] / g["n_train_observed"]).mean(),
                             **_cond_cv_cols(g, ref, common),
                             min_restarts_within_1e3=int(g["n_within_1e3"].min()), w_T_mean=g["w_T"].mean()))
    summ = pd.DataFrame(rows).sort_values(["universe", "heldout_ll_per_edge_mean"], ascending=[True, False])
    keep = ["universe", "fold", "config", "model", "grouping", "n_test_tfs", "n_train_observed", "n_test_observed",
            "train_loglik", "test_loglik", "test_loglik_per_obs_edge", "test_cond_loglik",
            "test_cond_loglik_per_cell", "n_within_1e3", "n_converged", "w_T"]
    return summ, folds[keep].sort_values(["universe", "config", "fold"])


def nuts_table() -> pd.DataFrame:
    rows = []
    u = NUTS_UNIVERSE
    ids = [(fid + "_nuts", fit_dir(u, PRIMARY, fid + "_nuts")) for fid in NUTS_PRIOR_IDS]
    pn = primary_nuts(u)
    if pn is not None:
        ids = [("family_hb_nuts", Path(pn["dir"]))] + ids
    data = load_data(u)
    hb = primary_fit(u, "family_hb")
    obs = (data.R >= 0).any(1)
    rep = (data.R == 1).any(1)
    for fid, d in ids:
        if not (d / "meta.json").exists():
            continue
        f = load_fit(d)
        m = f["nuts"]
        pr = _prior_full(PRIOR_SENS.get(fid[:-len("_nuts")], None))
        width = f["cred_hi"] - f["cred_lo"]
        rows.append(dict(universe=u, fit_id=fid, source=str(d.relative_to(C.OUT)), scale_sd=pr["scale_sd"],
                         dirichlet=pr["dirichlet"], n_divergent=m["n_divergent"], r_hat_max=m["r_hat_max"],
                         ess_bulk_min=m["ess_bulk_min"], ess_tail_min=m["ess_tail_min"],
                         w_T_mean=m["w_mean"][0], w_T_q025=m["w_q025"][0], w_T_q975=m["w_q975"][0],
                         w_T_map_primary=float(hb["params"]["w"][0]),
                         cri_width_mean_anyreport_edges=float(np.average(width[rep], weights=data.n[rep])),
                         cri_width_mean_observed_edges=float(np.average(width[obs], weights=data.n[obs])),
                         spearman_vs_map_anyreport=weighted_spearman(f["cred"][rep], hb["cred"][rep], data.n[rep]),
                         mean_abs_diff_vs_map_observed=float(np.average(np.abs(f["cred"][obs] - hb["cred"][obs]),
                                                                       weights=data.n[obs])),
                         runtime_s=m["runtime_s"], chains="all"))
        # the same summaries over the chains that sampled the best MAP optimum's mode only
        mode_rows = [r for r in nuts_chain_rows(u) if r["fit_id"] == fid]
        if not mode_rows:
            continue
        keep = sorted(r["chain"] for r in mode_rows if r["nearest_optimum"] == 0)
        with np.load(d / "draws.npz") as z:
            Wd, THd = np.asarray(z["w"]), np.asarray(z["theta"])
        ch = chain_ids(d, len(Wd), f["meta"])
        sel = np.isin(ch, keep)
        base = dict(rows[-1])
        base.update(chains="map_mode:" + ",".join(map(str, keep)), n_chains_map_mode=len(keep),
                    n_chains_total=len(mode_rows))
        if sel.sum() == 0:
            for k in ("r_hat_max", "ess_bulk_min", "ess_tail_min", "w_T_mean", "w_T_q025", "w_T_q975",
                      "cri_width_mean_anyreport_edges", "cri_width_mean_observed_edges",
                      "spearman_vs_map_anyreport", "mean_abs_diff_vs_map_observed"):
                base[k] = np.nan
            rows.append(base)
            continue
        Cd = np.stack([lcm.credibility_at(dict(w=Wd[i], theta=THd[i]), data) for i in np.flatnonzero(sel)])
        cmean = Cd.mean(0)
        wdt = np.quantile(Cd, 0.975, axis=0) - np.quantile(Cd, 0.025, axis=0)
        rh = np.nan
        if len(keep) >= 2:
            import arviz as az
            nmin = min(int((ch == c_).sum()) for c_ in keep)
            arr = np.stack([Wd[ch == c_][:nmin] for c_ in keep])          # (chains, draws, S)
            rh = float(np.nanmax(az.rhat(az.convert_to_dataset({"w": arr}))["w"].values))
        base.update(r_hat_max=np.nan, ess_bulk_min=np.nan, ess_tail_min=np.nan, r_hat_w_map_mode_thinned=rh,
                    w_T_mean=float(Wd[sel, 0].mean()), w_T_q025=float(np.quantile(Wd[sel, 0], 0.025)),
                    w_T_q975=float(np.quantile(Wd[sel, 0], 0.975)),
                    cri_width_mean_anyreport_edges=float(np.average(wdt[rep], weights=data.n[rep])),
                    cri_width_mean_observed_edges=float(np.average(wdt[obs], weights=data.n[obs])),
                    spearman_vs_map_anyreport=weighted_spearman(cmean[rep], hb["cred"][rep], data.n[rep]),
                    mean_abs_diff_vs_map_observed=float(np.average(np.abs(cmean[obs] - hb["cred"][obs]),
                                                                  weights=data.n[obs])))
        rows.append(base)
    return pd.DataFrame(rows)


def build_tables(universes):
    mem, assoc = grouping_tables(universes)
    _write(mem, "family_groupings")
    _write(assoc, "coreporting_assoc")
    _write(sensitivity_rows(universes), "sensitivity")
    _write(prior_param_table(universes), "sensitivity_params")
    nt = nuts_table()
    if len(nt):
        _write(nt, "nuts_prior_sensitivity")
    summ, folds = cv_tables(universes)
    if len(summ):
        _write(summ, "cv_loglik")
        _write(folds, "cv_loglik_folds")
    ppc = ppc_table(universes)
    if len(ppc):
        _write(ppc, "ppc")
    mo, mc = modes_tables(universes)
    if len(mo):
        _write(mo, "modes")
    if len(mc):
        _write(mc, "sensitivity_nuts_chains")   # not "nuts_chains": that is fit_primary's table
    rec, cal, reps = recovery_tables(universes)
    if len(rec):
        # post hoc columns / rows of the identifiability module (realised FDR, ECE over bins 1-9,
        # w_T by mode, single-mode NUTS coverage) are kept when this module rebuilds its tables
        from r2 import posterior_modes as PM
        if (PM.MODOUT / "sim").exists():
            rec, cal, reps, cov = PM.augment_recovery_tables(rec, cal, reps)
            if len(cov):
                PM._write(cov, "fit_recovery_nuts_chainsets.csv")
        _write(rec, "recovery")
        _write(cal, "recovery_calibration")
        _write(reps, "recovery_replicates")


# ----------------------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("step", choices=["fits", "nuts", "cv", "ppc", "recovery", "recovery_nuts", "truthstart",
                                     "modes", "tables", "all"])
    ap.add_argument("--universes", nargs="+", default=list(UNIVERSES), choices=list(UNIVERSES))
    ap.add_argument("--restarts", type=int, default=N_RESTARTS)
    ap.add_argument("--nuts", action="store_true", help="recovery: also run the NUTS coverage replicates")
    ap.add_argument("--force", action="store_true", help="refit even if stored results exist")
    ap.add_argument("--folds", nargs="+", type=int, default=None, help="cv: only these folds (parallel runs)")
    ap.add_argument("--reps", nargs="+", type=int, default=None,
                    help="recovery / recovery_nuts: only these replicate indices (parallel runs)")
    ap.add_argument("--fit-ids", nargs="+", default=None, choices=list(NUTS_PRIOR_IDS),
                    help="nuts: only these prior-sensitivity fits (parallel runs)")
    a = ap.parse_args(argv)
    C.ensure_dir(MOD)
    t0 = time.time()
    if a.step in ("fits", "all"):
        run_fits(a.universes, force=a.force, n_restarts=a.restarts)
    if a.step in ("nuts", "all"):
        run_nuts_priors(force=a.force, fit_ids=a.fit_ids)
    if a.step in ("cv", "all"):
        run_cv(a.universes, force=a.force, n_restarts=a.restarts, folds=a.folds)
    if a.step in ("ppc", "all"):
        run_ppc(a.universes, force=a.force)
    if a.step in ("recovery", "all"):
        run_recovery(a.universes, force=a.force, reps=a.reps)
        if a.nuts or a.step == "all":
            run_recovery_nuts(force=a.force)
    if a.step == "recovery_nuts":
        run_recovery_nuts(force=a.force, reps=a.reps)
    if a.step in ("truthstart", "all"):
        run_recovery_truthstart(a.universes, force=a.force)
    if a.step in ("modes", "all"):
        run_modes(a.universes, force=a.force)
    if a.step in ("tables", "all"):
        build_tables(list(UNIVERSES))
    log(f"done {a.step} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
