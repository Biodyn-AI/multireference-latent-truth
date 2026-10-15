"""Latent-class credibility models over pattern-compressed annotator evidence.

All five estimators of the paper are members of one model family and share a
single implementation:

========================  ===================  ===========  =====================
name                      latent classes       hierarchy    burst structure
========================  ===================  ===========  =====================
``em``                    T, N                 no           none (Dawid-Skene)
``dependence``            T, N, B              no           one global burst
``multifactor``           T, N, B1, B2         no           two global bursts
``family_regularized``    T, N                 yes          none
``family_hb``             T, N, B_f            yes          one burst per family
========================  ===================  ===========  =====================

Generative model (``family_hb``; the other models are special cases)::

    z_i ~ Categorical(w),             w ~ Dirichlet(1, ..., 1)
    r_ij | z_i = s ~ Bernoulli(theta_js)   for annotators j observing edge i
    theta_jN   = beta_j = sigmoid(b_j)                      (clean false-positive rate)
    theta_jT   = alpha_j = beta_j + (1 - beta_j) sigmoid(a_j)   (sensitivity > beta_j)
    (global-burst models instead order beta_j < gamma_jm < alpha_j)
    theta_jB_f = gamma_j = beta_j + (1 - beta_j) sigmoid(g_j)   if f(j) == f else beta_j
    q_j ~ Normal(mu^q_{f(j)}, tau_q),  mu^q_f ~ Normal(mu^q_0, omega_q),
    mu^q_0 ~ Normal(m_q, 2),  tau_q, omega_q ~ HalfNormal(1),   q in {b, a, g}

Non-hierarchical models place independent ``Normal(m_q, 3)`` priors on q_j.
The latent class is marginalised analytically, and because every quantity is
a function of the observation pattern only, the likelihood is evaluated over
the P unique patterns weighted by their edge counts.

Credibility of an edge = P(z = T | pattern, parameters).

Informative observation (secondary extension; ``family_hb_obs``, ``em_obs``)
---------------------------------------------------------------------------
The five models above treat an unobserved entry as missing at random (MAR).
The two ``*_obs`` models (designed after the held-out evaluation; reported as
a secondary, post hoc analysis) make *observation* itself an emission for the
annotators of the families in ``spec['obs_families']`` (literature, LIT).  They
need eligibility-coded evidence (``PatternData(..., eligibility=True)``):

    R_ij = 1 reported, 0 observed but not reported,
          -1 eligible but not observed, -2 not eligible / observation not modelled

A LIT annotator j is *eligible* for edge (t, g) iff t is covered by j and
*observes* it iff eligible and g is a node of j (curated-conditional mask).
For eligible entries of modelled annotators

    o_ij | z_i = s ~ Bernoulli(eta_js),  eta_jT (s = T),  eta_jNonT (s = N or a burst class)
    r_ij | o_ij = 1, z_i = s ~ Bernoulli(theta_js)   (as above)

with logit eta_jT = eT_j, logit eta_jNonT = eN_j pooled like the other rates
(``family_hb_obs``: eT_j ~ N(mu^eT_f(j), tau_eT), mu^eT_f ~ N(mu^eT_0, omega_eT),
mu^eT_0 ~ N(0, 2), HalfNormal(1) scales, same for eN; ``em_obs``: independent
N(0, 3)); no ordering between eta_jT and eta_jNonT.  Observation of the other
annotators (MOTIF/COEXP, deterministic given the LIT data under the
curated-conditional mask) is conditioned on and coded -2 when absent.  Their
-2 entries and all ineligible entries contribute nothing.  With
eta_jT = eta_jNonT the observation factor does not depend on z and the model
reduces to its MAR counterpart (tested).  Credibility = P(z = T | o_i, r_i).
``conditional_loglik`` gives log p(r | o) (comparable with the MAR models, for
which it is the ordinary marginal likelihood), ``observation_loglik`` log p(o)
and ``heldout_loglik`` the joint log p(o, r).

``MODEL_SPECS`` holds every model; iterating over it (``tuple(MODEL_SPECS)``,
``for m in MODEL_SPECS``) lists only the five MAR models of the prespecified
estimator ladder (``MAR_MODELS``), so that modules looping over the ladder are
unchanged; the extension models are reached by key and listed in ``OBS_MODELS``.

Inference
---------
* ``fit_map``: posterior mode in the unconstrained parametrisation (log scales,
  stick-breaking weights; the Jacobian makes it boundary-avoiding for the
  hierarchical scales), found by L-BFGS from many random starts.
* ``fit_nuts``: full posterior by the No-U-Turn sampler (numpyro), with
  non-centred hierarchical parametrisation.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp

jax.config.update("jax_enable_x64", True)

import numpyro  # noqa: E402
import numpyro.distributions as dist  # noqa: E402
from numpyro.infer import MCMC, NUTS  # noqa: E402
from numpyro.infer.util import initialize_model  # noqa: E402
from scipy.optimize import minimize  # noqa: E402
from jax.flatten_util import ravel_pytree  # noqa: E402

class _ModelSpecs(dict):
    """Model specifications.  Key lookup (``[]``, ``get``, ``in``) covers every model;
    iteration, ``keys``/``values``/``items`` and ``len`` cover only the MAR estimator
    ladder (``MAR_MODELS``) so that existing loops over ``MODEL_SPECS`` are unchanged by the
    informative-observation extension (``OBS_MODELS``)."""

    def _ladder(self):
        return [k for k in dict.keys(self) if not dict.__getitem__(self, k).get("obs_families")]

    def __iter__(self):
        return iter(self._ladder())

    def keys(self):
        return self._ladder()

    def values(self):
        return [dict.__getitem__(self, k) for k in self._ladder()]

    def items(self):
        return [(k, dict.__getitem__(self, k)) for k in self._ladder()]

    def __len__(self):
        return len(self._ladder())

    def all_names(self):
        return list(dict.keys(self))


MODEL_SPECS = _ModelSpecs({
    "em": dict(n_global_burst=0, family_burst=False, hierarchical=False),
    "dependence": dict(n_global_burst=1, family_burst=False, hierarchical=False),
    "multifactor": dict(n_global_burst=2, family_burst=False, hierarchical=False),
    "family_regularized": dict(n_global_burst=0, family_burst=False, hierarchical=True),
    "family_hb": dict(n_global_burst=0, family_burst=True, hierarchical=True),
    # informative-observation extension (secondary, post hoc; see module docstring)
    "family_hb_obs": dict(n_global_burst=0, family_burst=True, hierarchical=True, obs_families=("LIT",)),
    "em_obs": dict(n_global_burst=0, family_burst=False, hierarchical=False, obs_families=("LIT",)),
})
MAR_MODELS = tuple(MODEL_SPECS)                                   # the prespecified ladder
OBS_MODELS = tuple(k for k in MODEL_SPECS.all_names() if k not in MAR_MODELS)
# MAR counterpart of each informative-observation model (eta_T = eta_NonT reduces to it)
MAR_COUNTERPART = {"family_hb_obs": "family_hb", "em_obs": "em"}
PRIOR_MEAN = {"b": float(np.log(0.01 / 0.99)), "a": 0.0, "g": 0.0, "eT": 0.0, "eN": 0.0}
OBS_CODE_UNOBSERVED = -1     # eligible, not observed (eligibility-coded data)
OBS_CODE_INELIGIBLE = -2     # not eligible / observation not modelled
FLAT_SD = 3.0       # sd of the independent priors of non-hierarchical models
GLOBAL_SD = 2.0     # sd of the global hyper-means of hierarchical models


# ----------------------------------------------------------------------------
# Data container
# ----------------------------------------------------------------------------
@dataclass
class PatternData:
    """Pattern-compressed evidence.

    R: (P, K) int8 with 1 = reported, 0 = observed but not reported,
       -1 = not observed.
    n: (P,) number of edges with each pattern.
    family: length-K family label of each annotator.
    annotators: length-K annotator ids.
    eligibility: optional; True for eligibility-coded evidence, where -1 = eligible but
       not observed and -2 = not eligible (or observation not modelled).  Every MAR
       quantity uses only R >= 0 (observed) and R == 1 (reported), so MAR models give the
       same results on eligibility-coded and on plain evidence; the ``*_obs`` models need it.
    """

    R: np.ndarray
    n: np.ndarray
    family: list
    annotators: list
    eligibility: bool = False
    families: list = field(init=False)
    fam_idx: np.ndarray = field(init=False)

    def __post_init__(self):
        self.R = np.asarray(self.R, dtype=np.int8)
        self.n = np.asarray(self.n, dtype=np.float64)
        assert self.R.shape[0] == self.n.shape[0]
        assert self.R.shape[1] == len(self.family) == len(self.annotators)
        self.eligibility = bool(self.eligibility)
        if self.R.size:
            lo = -2 if self.eligibility else -1
            if self.R.min() < lo or self.R.max() > 1:
                raise ValueError(f"evidence codes must lie in [{lo}, 1]")
        self.families = sorted(set(self.family))
        self.fam_idx = np.array([self.families.index(f) for f in self.family])

    @property
    def K(self):
        return self.R.shape[1]

    @property
    def P(self):
        return self.R.shape[0]

    def burst_families(self):
        """Families with >= 2 annotators (a family burst state is only
        identifiable when at least two annotators can co-report it)."""
        counts = np.bincount(self.fam_idx, minlength=len(self.families))
        return [f for f in range(len(self.families)) if counts[f] >= 2]

    @staticmethod
    def from_matrix(M: np.ndarray, family, annotators, eligibility: bool = False):
        """Compress a dense (edges x annotators) {1,0,-1} (eligibility-coded: {1,0,-1,-2})
        matrix to patterns.

        Returns (PatternData, edge_to_pattern index)."""
        M = np.asarray(M, dtype=np.int8)
        uniq, inv, counts = np.unique(M, axis=0, return_inverse=True, return_counts=True)
        return PatternData(uniq, counts, list(family), list(annotators), eligibility), inv.reshape(-1)

    def obs_columns(self, obs_families) -> np.ndarray:
        """Indices of the annotators whose observation is modelled (families ``obs_families``)."""
        return np.flatnonzero(np.isin(np.asarray(self.family, dtype=object), list(obs_families)))

    def marginal(self) -> "PatternData":
        """The same evidence with eligibility codes collapsed (-2 -> -1) and patterns
        re-compressed: the plain representation used by the MAR models."""
        R = np.where(self.R < -1, np.int8(-1), self.R).astype(np.int8)
        uniq, inv = np.unique(R, axis=0, return_inverse=True)
        n = np.bincount(inv.reshape(-1), weights=self.n, minlength=len(uniq))
        return PatternData(uniq, n, list(self.family), list(self.annotators))


# ----------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------
def _class_layout(data: PatternData, spec: dict):
    """Return the class names and, for each burst class, the annotator mask of
    elevated reporting (K-vector of 0/1)."""
    names = ["T", "N"]
    masks = []
    K = data.K
    for m in range(spec["n_global_burst"]):
        names.append(f"B{m + 1}")
        masks.append(np.ones(K))
    if spec["family_burst"]:
        for f in data.burst_families():
            names.append(f"B_{data.families[f]}")
            masks.append((data.fam_idx == f).astype(float))
    return names, (np.stack(masks) if masks else np.zeros((0, K)))


def _n_g(data, spec):
    """Number of burst-rate parameters per annotator and their index map."""
    if spec["family_burst"]:
        return 1
    return spec["n_global_burst"]


# NB: other modules compare stored fit metadata with dict(DEFAULT_PRIOR) (fit_sensitivity), so the
# optional keys 'obs_weight', 'centered' and 'obs_conditional' (absent = off) are NOT listed here.
DEFAULT_PRIOR = dict(scale_sd=1.0, dirichlet=1.0, global_sd=GLOBAL_SD, flat_sd=FLAT_SD)


def build_model(data: PatternData, spec: dict, prior: dict | None = None):
    """Return a numpyro model function for the given data and spec.

    ``prior`` may override ``scale_sd`` (HalfNormal sd of tau/omega),
    ``dirichlet`` (class-weight concentration), ``global_sd`` and ``flat_sd``.
    Informative-observation models also accept ``obs_weight`` (scalar or length-K vector, read
    for the modelled annotators): the observation log-likelihood of annotator j is multiplied
    by obs_weight_j (a composite / power likelihood; 1 = the generative model).  The weights are
    recorded as the deterministic site ``obs_weight`` and used by every numpy evaluation.
    ``centered`` (hierarchical models; default False = the non-centred form of every existing
    fit): True, or a collection of parameter types among 'b', 'a', 'g', 'eT', 'eN'.  The
    annotator-level parameters of those types are sampled directly, q_j ~ N(mu^q_f(j), tau_q),
    instead of through z_j with q_j = mu^q_f(j) + tau_q z_j.  The joint density of the model
    parameters is the same (tested); only the sampling coordinates change, which matters for
    NUTS when the data determine q_j sharply (the family level stays non-centred).  Types
    whose q_j the data do not determine should stay non-centred (a centred density is
    unbounded as tau_q -> 0).
    ``obs_conditional=True`` (informative-observation models): the likelihood factor is the
    CONDITIONAL likelihood of the reports given the observation pattern,
    sum_p n_p log p(r_p | o_p) = sum_p n_p [log sum_s w_s p(o_p | s) p(r_p | s) - log sum_s w_s p(o_p | s)],
    instead of the joint log p(o, r).  The model and its parameters (w, theta, eta) are the
    same; the observation pattern only shapes the class prior P(z | o) (a latent-class model
    with the observation pattern as concomitant variable) and the fit is not rewarded for
    explaining the observation indicators themselves.  Credibility = P(z = T | o, r) as before."""
    pr = dict(DEFAULT_PRIOR)
    if prior:
        pr.update(prior)
    names, burst_masks = _class_layout(data, spec)
    S = len(names)
    K = data.K
    F = len(data.families)
    cen = pr.get("centered")
    centered = (tuple(PRIOR_MEAN) if cen is True else tuple(cen) if cen else ())
    if any(q not in PRIOR_MEAN for q in centered):
        raise ValueError(f"centered must be a bool or a collection of {tuple(PRIOR_MEAN)}")
    fam_idx = jnp.asarray(data.fam_idx)
    obs = jnp.asarray((data.R >= 0).astype(float))
    pos = jnp.asarray((data.R == 1).astype(float))
    n = jnp.asarray(data.n)
    burst_masks = jnp.asarray(burst_masks)
    n_g = _n_g(data, spec)
    hier = spec["hierarchical"]
    # which family each family-burst class belongs to (for picking gamma)
    if spec["family_burst"]:
        burst_fams = data.burst_families()
    # informative observation: annotators whose observation is an emission
    obs_fams = tuple(spec.get("obs_families", ()))
    if obs_fams:
        if not data.eligibility:
            raise ValueError("informative-observation models need eligibility-coded evidence "
                             "(PatternData(..., eligibility=True))")
        ocols = data.obs_columns(obs_fams)
        if len(ocols) == 0:
            raise ValueError(f"no annotator of the families {obs_fams}")
        Ko = len(ocols)
        ofams = sorted(set(data.family[j] for j in ocols))
        ofam_idx = jnp.asarray([ofams.index(data.family[j]) for j in ocols])
        Ro = data.R[:, ocols]
        elig_o = jnp.asarray((Ro >= -1).astype(float))           # eligible
        seen_o = jnp.asarray((Ro >= 0).astype(float))            # eligible and observed
        miss_o = elig_o - seen_o                                  # eligible, not observed
        eta_rows = np.full(K, -1, dtype=np.int64)
        eta_rows[ocols] = np.arange(Ko)
        not_modelled = jnp.asarray(eta_rows < 0)
        eta_take = jnp.asarray(np.maximum(eta_rows, 0))
        ow = pr.get("obs_weight")
        if ow is not None:
            ow = np.broadcast_to(np.asarray(ow, dtype=np.float64), (K,)).copy()
            if not np.all(np.isfinite(ow[ocols])) or np.any(ow[ocols] < 0):
                raise ValueError("obs_weight must be finite and >= 0 for the modelled annotators")
            ow_full = np.where(eta_rows >= 0, ow, np.nan)
            seen_o = seen_o * jnp.asarray(ow[ocols])[None, :]
            miss_o = miss_o * jnp.asarray(ow[ocols])[None, :]

    def annotator_params(q, n_ann=None, fidx=None, n_fam=None):
        """Sample the annotator-level parameters of type q (shape (K,) or (K, n_g)); for the
        observation rates (q in {'eT', 'eN'}) over the ``n_ann`` modelled annotators with
        family index ``fidx`` among ``n_fam`` families."""
        n_ann = K if n_ann is None else n_ann
        fidx = fam_idx if fidx is None else fidx
        n_fam = F if n_fam is None else n_fam
        shape = (n_ann,) if (q != "g" or n_g == 1) else (n_ann, n_g)
        if not hier:
            return numpyro.sample(q, dist.Normal(PRIOR_MEAN[q], pr["flat_sd"]).expand(shape).to_event(len(shape)))
        mu0 = numpyro.sample(f"mu0_{q}", dist.Normal(PRIOR_MEAN[q], pr["global_sd"]))
        omega = numpyro.sample(f"omega_{q}", dist.HalfNormal(pr["scale_sd"]))
        tau = numpyro.sample(f"tau_{q}", dist.HalfNormal(pr["scale_sd"]))
        with numpyro.plate(f"fam_{q}", n_fam):
            z_f = numpyro.sample(f"zf_{q}", dist.Normal(0.0, 1.0))
        mu_f = numpyro.deterministic(f"muf_{q}", mu0 + omega * z_f)
        if q in centered:
            loc = mu_f[fidx] if len(shape) == 1 else mu_f[fidx][:, None]
            return numpyro.sample(q, dist.Normal(loc, tau).expand(shape).to_event(len(shape)))
        with numpyro.plate(f"ann_{q}", n_ann):
            z_j = numpyro.sample(f"zj_{q}", dist.Normal(0.0, 1.0))
        return numpyro.deterministic(f"{q}", mu_f[fidx] + tau * z_j)

    def model():
        w = numpyro.sample("w", dist.Dirichlet(pr["dirichlet"] * jnp.ones(S)))
        b = annotator_params("b")
        a = annotator_params("a")
        beta = jax.nn.sigmoid(b)
        if spec["n_global_burst"] > 0:
            # global-burst models: ordered reporting rates per annotator,
            # beta < gamma_1 < ... < gamma_M < alpha, which identifies T as the
            # class reported most often by every annotator
            g = annotator_params("g")
            g2 = g if g.ndim == 2 else g[:, None]
            gammas, lower = [], beta
            for m in range(spec["n_global_burst"]):
                lower = lower + (1 - lower) * jax.nn.sigmoid(g2[:, m])
                gammas.append(lower)
            alpha = lower + (1 - lower) * jax.nn.sigmoid(a)
            cols = [alpha, beta] + gammas
        else:
            alpha = beta + (1 - beta) * jax.nn.sigmoid(a)
            cols = [alpha, beta]
            if spec["family_burst"] and S > 2:
                # family bursts: T is identified because it raises reporting in
                # every family, whereas B_f raises it only within family f
                g = annotator_params("g")
                gamma = beta + (1 - beta) * jax.nn.sigmoid(g)
                for k, f in enumerate(burst_fams):
                    m = burst_masks[k]
                    cols.append(m * gamma + (1 - m) * beta)
        theta = jnp.clip(jnp.stack(cols, axis=1), 1e-12, 1 - 1e-12)  # (K, S)
        ll_ps = (obs * pos) @ jnp.log(theta) + (obs * (1 - pos)) @ jnp.log1p(-theta)
        if obs_fams:
            # observation emission of the modelled annotators: eta_T for class T,
            # eta_NonT shared by N and the burst classes (no ordering constraint)
            eT = annotator_params("eT", Ko, ofam_idx, len(ofams))
            eN = annotator_params("eN", Ko, ofam_idx, len(ofams))
            eta_T = jnp.clip(jax.nn.sigmoid(eT), 1e-12, 1 - 1e-12)
            eta_N = jnp.clip(jax.nn.sigmoid(eN), 1e-12, 1 - 1e-12)
            eta = jnp.concatenate([eta_T[:, None], jnp.repeat(eta_N[:, None], S - 1, axis=1)], axis=1)  # (Ko, S)
            ll_obs = seen_o @ jnp.log(eta) + miss_o @ jnp.log1p(-eta)
            ll_ps = ll_ps + ll_obs
            # (K, S) with NaN rows for annotators whose observation is not modelled
            numpyro.deterministic("eta", jnp.where(not_modelled[:, None], jnp.nan, eta[eta_take]))
            if ow is not None:
                numpyro.deterministic("obs_weight", jnp.asarray(ow_full))
        log_joint = jnp.log(w)[None, :] + ll_ps
        log_mix = logsumexp(log_joint, axis=1)
        if obs_fams and pr.get("obs_conditional"):
            # conditional likelihood of the reports given the observation pattern
            numpyro.factor("loglik", jnp.sum(n * (log_mix - logsumexp(jnp.log(w)[None, :] + ll_obs, axis=1))))
        else:
            numpyro.factor("loglik", jnp.sum(n * log_mix))
        numpyro.deterministic("cred", jnp.exp(log_joint[:, 0] - log_mix))
        numpyro.deterministic("theta", theta)

    model.class_names = names
    return model


def _collapsed_loglik_and_cred(data, spec, sample):
    """Evaluate log-likelihood and per-pattern credibility at a parameter sample
    (dict of constrained sites incl. deterministic 'theta' and 'w')."""
    theta = np.asarray(sample["theta"])
    w = np.asarray(sample["w"])
    obs = (data.R >= 0).astype(float)
    pos = (data.R == 1).astype(float)
    ll_ps = (obs * pos) @ np.log(theta) + (obs * (1 - pos)) @ np.log1p(-theta)
    if sample.get("eta") is not None:
        ll_ps = ll_ps + _observation_term(data, sample["eta"], sample.get("obs_weight"))
    lj = np.log(w)[None, :] + ll_ps
    lm = np.logaddexp.reduce(lj, axis=1)
    return float(np.sum(data.n * lm)), np.exp(lj[:, 0] - lm), lm


def _observation_term(data: PatternData, eta, obs_weight=None) -> np.ndarray:
    """(P, S) log p(o_p | z = s) of the modelled annotators (rows of ``eta`` (K, S) that are
    finite), each annotator's term multiplied by ``obs_weight`` (K,) if given (composite
    likelihood; see ``build_model``); requires eligibility-coded data."""
    eta = np.asarray(eta, dtype=np.float64)
    cols = np.flatnonzero(np.isfinite(eta[:, 0]))
    if len(cols) == 0:
        return np.zeros((data.P, eta.shape[1]))
    if not data.eligibility:
        raise ValueError("parameters with an observation emission (eta) need eligibility-coded evidence")
    Ro = data.R[:, cols]
    e = np.clip(eta[cols], 1e-12, 1 - 1e-12)
    seen = (Ro >= 0).astype(float)
    miss = ((Ro == -1)).astype(float)
    if obs_weight is not None:
        wt = np.broadcast_to(np.asarray(obs_weight, dtype=np.float64), (data.K,))[cols]
        seen, miss = seen * wt[None, :], miss * wt[None, :]
    return seen @ np.log(e) + miss @ np.log1p(-e)


def _log_joint_parts(data: PatternData, sample: dict):
    """(log w, report term (P, S), observation term (P, S)) at a parameter sample."""
    theta = np.clip(np.asarray(sample["theta"], dtype=np.float64), 1e-12, 1 - 1e-12)
    w = np.asarray(sample["w"], dtype=np.float64)
    obs = (data.R >= 0).astype(float)
    pos = (data.R == 1).astype(float)
    rep = (obs * pos) @ np.log(theta) + (obs * (1 - pos)) @ np.log1p(-theta)
    ob = (_observation_term(data, sample["eta"], sample.get("obs_weight")) if sample.get("eta") is not None
          else np.zeros_like(rep))
    return np.log(w), rep, ob


def observation_loglik(params: dict, data: PatternData, per_pattern: bool = False):
    """log p(o) = sum_p n_p log sum_s w_s p(o_p | s) (0 for MAR parameters without eta)."""
    lw, _, ob = _log_joint_parts(data, params)
    lo = np.logaddexp.reduce(lw[None, :] + ob, axis=1)
    return lo if per_pattern else float(np.sum(data.n * lo))


def conditional_loglik(params: dict, data: PatternData, per_pattern: bool = False):
    """log p(r | o) = sum_p n_p log sum_s P(z = s | o_p) p(r_p | s): the report likelihood given
    the observation pattern.  For MAR parameters P(z | o) = w and this is the ordinary marginal
    likelihood (``heldout_loglik``), so it compares MAR and informative-observation models on
    the same quantity.  Patterns observed by no annotator contribute exactly 0."""
    lw, rep, ob = _log_joint_parts(data, params)
    a = lw[None, :] + ob
    post_o = a - np.logaddexp.reduce(a, axis=1, keepdims=True)
    lc = np.logaddexp.reduce(post_o + rep, axis=1)
    return lc if per_pattern else float(np.sum(data.n * lc))


def class_posteriors(params: dict, data: PatternData, given: str = "all") -> np.ndarray:
    """(P, S) P(z = s | ...) with ``given`` = 'all' (o and r), 'obs' (o only) or 'prior' (w)."""
    lw, rep, ob = _log_joint_parts(data, params)
    a = lw[None, :] + {"all": rep + ob, "obs": ob, "prior": 0.0 * rep}[given]
    return np.exp(a - np.logaddexp.reduce(a, axis=1, keepdims=True))


# ----------------------------------------------------------------------------
# MAP by multi-start L-BFGS
# ----------------------------------------------------------------------------
@dataclass
class MapFit:
    model_name: str
    class_names: list
    best: dict                   # constrained parameter values at the best optimum
    cred: np.ndarray             # (P,) credibility at the best optimum
    loglik: float
    log_post: float
    restarts: list               # per-restart dicts (log_post, loglik, converged, grad_norm, n_iter)
    cred_restarts: np.ndarray    # (n_restarts, P)
    runtime_s: float

    def summary(self):
        lp = np.array([r["log_post"] for r in self.restarts])
        conv = np.array([r["converged"] for r in self.restarts])
        top = lp >= lp.max() - 1e-3
        return dict(model=self.model_name, n_restarts=len(lp), best_log_post=float(lp.max()),
                    n_within_1e3_of_best=int(top.sum()), n_converged=int(conv.sum()),
                    log_post_range=float(lp.max() - lp.min()), loglik=self.loglik,
                    max_abs_cred_diff_top=float(np.max(np.abs(self.cred_restarts[top] - self.cred[None, :]))),
                    runtime_s=self.runtime_s)


def fit_map(data: PatternData, model_name: str, n_restarts: int = 50, seed: int = 0,
            init_radius: float = 2.0, gtol: float = 1e-6, maxiter: int = 20000,
            prior: dict | None = None) -> MapFit:
    spec = MODEL_SPECS[model_name]
    model = build_model(data, spec, prior)
    t0 = time.time()
    rng = jax.random.PRNGKey(seed)
    init = initialize_model(rng, model, init_strategy=numpyro.infer.init_to_uniform(radius=init_radius))
    potential_fn = init.potential_fn
    postprocess = init.postprocess_fn
    flat0, unravel = ravel_pytree(init.param_info.z)
    pot = jax.jit(lambda x: potential_fn(unravel(x)))
    grad = jax.jit(jax.grad(lambda x: potential_fn(unravel(x))))
    restarts, creds, bests = [], [], []
    keys = jax.random.split(rng, n_restarts)
    for r in range(n_restarts):
        x0 = np.asarray(jax.random.uniform(keys[r], flat0.shape, minval=-init_radius, maxval=init_radius))
        res = minimize(lambda x: float(pot(jnp.asarray(x))), x0,
                       jac=lambda x: np.asarray(grad(jnp.asarray(x))),
                       method="L-BFGS-B", options=dict(maxiter=maxiter, gtol=gtol, ftol=1e-15, maxcor=30))
        z = unravel(jnp.asarray(res.x))
        const = postprocess(z)
        sample = {k: np.asarray(v) for k, v in const.items()}
        ll, cred, _ = _collapsed_loglik_and_cred(data, spec, sample)
        gnorm = float(np.max(np.abs(np.asarray(grad(jnp.asarray(res.x))))))
        restarts.append(dict(restart=r, log_post=-float(res.fun), loglik=ll, converged=bool(res.success),
                             grad_norm=gnorm, n_iter=int(res.nit), message=str(res.message)))
        creds.append(cred)
        bests.append(sample)
    lp = np.array([r["log_post"] for r in restarts])
    ib = int(np.argmax(lp))
    return MapFit(model_name, model.class_names, bests[ib], creds[ib], restarts[ib]["loglik"],
                  float(lp[ib]), restarts, np.stack(creds), time.time() - t0)


# ----------------------------------------------------------------------------
# Full posterior by NUTS
# ----------------------------------------------------------------------------
@dataclass
class NutsFit:
    model_name: str
    class_names: list
    samples: dict            # site -> (n_draws, ...) arrays (incl. 'cred')
    diagnostics: dict        # site -> dict(r_hat_max, ess_bulk_min, ess_tail_min)
    n_divergent: int
    runtime_s: float

    def cred_summary(self):
        c = self.samples["cred"]
        return dict(mean=c.mean(0), lo=np.quantile(c, 0.025, axis=0), hi=np.quantile(c, 0.975, axis=0))


def fit_nuts(data: PatternData, model_name: str = "family_hb", num_warmup: int = 1000,
             num_samples: int = 1000, num_chains: int = 4, seed: int = 0,
             target_accept: float = 0.95, prior: dict | None = None) -> NutsFit:
    import arviz as az
    spec = MODEL_SPECS[model_name]
    model = build_model(data, spec, prior)
    t0 = time.time()
    kernel = NUTS(model, target_accept_prob=target_accept)
    mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples, num_chains=num_chains,
                chain_method="sequential", progress_bar=False)
    mcmc.run(jax.random.PRNGKey(seed), extra_fields=("diverging",))
    s_chain = mcmc.get_samples(group_by_chain=True)
    div = int(np.asarray(mcmc.get_extra_fields(group_by_chain=True)["diverging"]).sum())
    diags = {}
    for k, v in s_chain.items():
        if k in ("cred", "theta"):
            continue
        arr = np.asarray(v)
        arr = arr.reshape(arr.shape[0], arr.shape[1], -1)
        idata = az.convert_to_dataset({k: arr})
        rh = az.rhat(idata)[k].values
        eb = az.ess(idata, method="bulk")[k].values
        et = az.ess(idata, method="tail")[k].values
        diags[k] = dict(r_hat_max=float(np.nanmax(rh)), ess_bulk_min=float(np.nanmin(eb)),
                        ess_tail_min=float(np.nanmin(et)))
    flat = {k: np.asarray(v).reshape((-1,) + np.asarray(v).shape[2:]) for k, v in s_chain.items()}
    return NutsFit(model_name, model.class_names, flat, diags, div, time.time() - t0)


# ----------------------------------------------------------------------------
# Simulation from a fitted model (parameter recovery / calibration)
# ----------------------------------------------------------------------------
def simulate(data: PatternData, params: dict, seed: int = 0, n_edges: int | None = None):
    """Simulate edge-level evidence with the observation masks of ``data``.

    Edges are drawn with the empirical distribution of observation masks
    (which annotators observe the edge); the latent class and the reports are
    then drawn from the model with parameters ``params`` (needs 'w', 'theta').
    Returns (dense matrix, true class index per edge)."""
    rng = np.random.default_rng(seed)
    obs_mask = (data.R >= 0)
    mkeys, minv = np.unique(obs_mask, axis=0, return_inverse=True)
    mcounts = np.bincount(minv.reshape(-1), weights=data.n)
    N = int(data.n.sum()) if n_edges is None else int(n_edges)
    which = rng.choice(len(mkeys), size=N, p=mcounts / mcounts.sum())
    w = np.asarray(params["w"])
    theta = np.asarray(params["theta"])
    z = rng.choice(len(w), size=N, p=w / w.sum())
    U = rng.random((N, data.K))
    rep = (U < theta.T[z]).astype(np.int8)          # (N, K)
    M = np.where(mkeys[which], rep, -1).astype(np.int8)
    return M, z


def simulate_informative(data: PatternData, params: dict, seed: int = 0, n_edges: int | None = None):
    """Simulate eligibility-coded evidence from an informative-observation model.

    The *design* of an edge is the eligibility of the modelled annotators (rows of
    ``params['eta']`` that are finite) together with the observation mask of the other
    annotators (conditioned on, not modelled); designs are drawn with their empirical
    distribution in ``data``.  Then z ~ Categorical(w); for eligible modelled annotators
    o ~ Bernoulli(eta_jz); for observed entries r ~ Bernoulli(theta_jz).  Returns
    (dense eligibility-coded matrix (N, K), true class index per edge)."""
    if not data.eligibility:
        raise ValueError("simulate_informative needs eligibility-coded evidence")
    rng = np.random.default_rng(seed)
    eta = np.asarray(params["eta"], dtype=np.float64)
    mod = np.isfinite(eta[:, 0])
    design = np.where(mod[None, :], data.R >= -1, data.R >= 0)       # eligible / observed
    dkeys, dinv = np.unique(design, axis=0, return_inverse=True)
    dcounts = np.bincount(dinv.reshape(-1), weights=data.n)
    N = int(data.n.sum()) if n_edges is None else int(n_edges)
    which = rng.choice(len(dkeys), size=N, p=dcounts / dcounts.sum())
    w = np.asarray(params["w"], dtype=np.float64)
    theta = np.asarray(params["theta"], dtype=np.float64)
    z = rng.choice(len(w), size=N, p=w / w.sum())
    D = dkeys[which]
    o = (rng.random((N, data.K)) < np.where(mod[None, :], np.nan_to_num(eta.T[z]), 1.0)) & D
    rep = (rng.random((N, data.K)) < theta.T[z]).astype(np.int8)
    M = np.where(o, rep, np.where(mod[None, :] & D, np.int8(-1), np.int8(-2))).astype(np.int8)
    return M, z


# ----------------------------------------------------------------------------
# Pattern-level baseline scores (meta-analytic combination rules)
# ----------------------------------------------------------------------------
def baseline_scores(data: PatternData) -> dict:
    """Deterministic pattern-level scores of simple combination rules.

    rate_j = observed positive rate of annotator j (pattern-weighted).
    Fisher / Stouffer use mid-p values of the one-sided binary test of each
    annotator with null reporting probability rate_j:
        reported:      p = rate_j / 2
        not reported:  p = (1 + rate_j) / 2
    """
    from scipy.stats import chi2, norm
    obs = (data.R >= 0)
    pos = (data.R == 1)
    w = data.n[:, None]
    rate = (pos * w).sum(0) / np.maximum((obs * w).sum(0), 1)
    n_obs = obs.sum(1)
    n_pos = pos.sum(1)
    p_mid = np.where(pos, rate[None, :] / 2, (1 + rate[None, :]) / 2)
    logp = np.where(obs, np.log(p_mid), 0.0)
    X = -2 * logp.sum(1)
    fisher = np.where(n_obs > 0, -chi2.logsf(X, 2 * np.maximum(n_obs, 1)), 0.0)  # -log p (higher = stronger)
    zsc = np.where(obs, norm.isf(p_mid), 0.0)
    stouffer = np.where(n_obs > 0, zsc.sum(1) / np.sqrt(np.maximum(n_obs, 1)), -np.inf)
    fam = data.fam_idx
    n_fam_pos = np.zeros(data.P)
    for f in range(len(data.families)):
        n_fam_pos += pos[:, fam == f].any(1)
    return dict(
        union=(n_pos > 0).astype(float),
        majority=(n_pos >= 2).astype(float),
        vote_count=n_pos.astype(float),
        vote_fraction=np.where(n_obs > 0, n_pos / np.maximum(n_obs, 1), 0.0),
        n_families_reporting=n_fam_pos,
        fisher=fisher,
        stouffer=stouffer,
    )


# ----------------------------------------------------------------------------
# Helpers for sensitivity analyses and label-free model comparison
# ----------------------------------------------------------------------------
def with_families(data: PatternData, family) -> PatternData:
    """Same evidence with an alternative annotator -> family grouping."""
    return PatternData(data.R, data.n, list(family), list(data.annotators), data.eligibility)


def drop_annotators(data: PatternData, drop) -> tuple:
    """Remove annotators (by id) and re-compress patterns.

    Returns (PatternData, old_pattern -> new_pattern index)."""
    keep = [j for j, a in enumerate(data.annotators) if a not in set(drop)]
    R = data.R[:, keep]
    uniq, inv = np.unique(R, axis=0, return_inverse=True)
    inv = inv.reshape(-1)
    n = np.bincount(inv, weights=data.n, minlength=len(uniq))
    fam = [data.family[j] for j in keep]
    ann = [data.annotators[j] for j in keep]
    return PatternData(uniq, n, fam, ann, data.eligibility), inv


def heldout_loglik(fit_params: dict, data_test: PatternData) -> float:
    """Marginal log-likelihood of test patterns at fitted parameters
    (needs 'w' and 'theta'; annotator order must match).  With an observation emission
    ('eta') it is the joint log p(o, r) of the informative-observation model."""
    ll, _, _ = _collapsed_loglik_and_cred(data_test, None, fit_params)
    return ll


def credibility_at(fit_params: dict, data: PatternData) -> np.ndarray:
    """Per-pattern credibility P(T | pattern) at given parameters."""
    _, cred, _ = _collapsed_loglik_and_cred(data, None, fit_params)
    return cred


def nuts_credibility(nuts: "NutsFit", data: PatternData, max_draws: int | None = None) -> dict:
    """Posterior summary of per-pattern credibility for (possibly new) patterns
    ``data`` using the posterior draws of 'w' and 'theta' (annotator order must
    match the fitted data)."""
    W = nuts.samples["w"]
    TH = nuts.samples["theta"]
    ETA = nuts.samples.get("eta")
    OW = nuts.samples.get("obs_weight")
    idx = np.arange(W.shape[0])
    if max_draws and len(idx) > max_draws:
        idx = np.linspace(0, len(idx) - 1, max_draws).astype(int)
    obs = (data.R >= 0).astype(float)
    pos = (data.R == 1).astype(float)
    C_ = np.empty((len(idx), data.P))
    for r, d in enumerate(idx):
        th = TH[d]
        lj = np.log(W[d])[None, :] + (obs * pos) @ np.log(th) + (obs * (1 - pos)) @ np.log1p(-th)
        if ETA is not None:
            lj = lj + _observation_term(data, ETA[d], None if OW is None else OW[d])
        C_[r] = np.exp(lj[:, 0] - np.logaddexp.reduce(lj, axis=1))
    return dict(mean=C_.mean(0), lo=np.quantile(C_, 0.025, 0), hi=np.quantile(C_, 0.975, 0),
                sd=C_.std(0))
