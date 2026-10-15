"""Tests of the informative-observation extension (lcm ``family_hb_obs`` / ``em_obs``,
``evidence`` eligibility coding, ``informative_observation`` helpers)."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from r2 import lcm  # noqa: E402

FAM = ["LIT"] * 4 + ["MOTIF"] * 2 + ["COEXP"]
ANN = [f"s{j}" for j in range(len(FAM))]


def _truth(eta_T, eta_N, S=4):
    """Family-HB-shaped truth (classes T, N, B_LIT, B_MOTIF) with LIT observation rates."""
    fam = np.array(FAM)
    beta = np.array([0.01, 0.02, 0.005, 0.01, 0.05, 0.08, 0.03])
    alpha = np.array([0.45, 0.35, 0.25, 0.40, 0.45, 0.50, 0.30])
    gamma = np.array([0.30, 0.25, 0.20, 0.30, 0.60, 0.55, 0.03])
    theta = np.stack([alpha, beta, np.where(fam == "LIT", gamma, beta), np.where(fam == "MOTIF", gamma, beta)], axis=1)
    w = np.array([0.06, 0.84, 0.06, 0.04])
    eta = np.full((len(FAM), S), np.nan)
    lit = fam == "LIT"
    eta[lit, 0] = eta_T
    eta[lit, 1:] = np.asarray(eta_N)[:, None]
    return dict(w=w[:S] / w[:S].sum(), theta=theta[:, :S], eta=eta)


def _toy(truth, N=40000, seed=0, elig_p=(0.8, 0.6, 0.5, 0.7), motif_p=(0.4, 0.4, 0.5)):
    """Edge-level eligibility-coded data generated from the extended model."""
    rng = np.random.default_rng(seed)
    K = len(FAM)
    R0 = np.full((N, K), -2, dtype=np.int8)
    R0[:, :4] = np.where(rng.random((N, 4)) < np.array(elig_p), -1, -2)       # LIT eligible
    R0[:, 4:] = np.where(rng.random((N, 3)) < np.array(motif_p), 0, -2)       # MOTIF/COEXP observed (fixed)
    data0, _ = lcm.PatternData.from_matrix(R0, FAM, ANN, eligibility=True)
    M, z = lcm.simulate_informative(data0, truth, seed=seed + 1, n_edges=N)
    data, inv = lcm.PatternData.from_matrix(M, FAM, ANN, eligibility=True)
    return data, inv, z, M


def _edge_level_loglik(M, truth):
    """Direct per-edge joint log-likelihood (loops over classes; no pattern compression)."""
    w, th, eta = truth["w"], truth["theta"], truth["eta"]
    N, K = M.shape
    S = len(w)
    out = np.zeros((N, S))
    for s in range(S):
        lp = np.full(N, np.log(w[s]))
        for j in range(K):
            r = M[:, j]
            if np.isfinite(eta[j, 0]):
                lp += np.where(r >= 0, np.log(eta[j, s]), 0.0) + np.where(r == -1, np.log1p(-eta[j, s]), 0.0)
            lp += np.where(r == 1, np.log(th[j, s]), 0.0) + np.where(r == 0, np.log1p(-th[j, s]), 0.0)
        out[:, s] = lp
    lm = np.logaddexp.reduce(out, axis=1)
    return lm.sum(), np.exp(out[:, 0] - lm)


def test_model_specs_iteration_unchanged():
    assert tuple(lcm.MODEL_SPECS) == ("em", "dependence", "multifactor", "family_regularized", "family_hb")
    assert len(lcm.MODEL_SPECS) == 5 and [k for k, _ in lcm.MODEL_SPECS.items()] == list(lcm.MAR_MODELS)
    assert "family_hb_obs" in lcm.MODEL_SPECS and "em_obs" in lcm.MODEL_SPECS
    assert lcm.MODEL_SPECS["family_hb_obs"]["obs_families"] == ("LIT",)
    assert lcm.OBS_MODELS == ("family_hb_obs", "em_obs")


def test_pattern_loglik_equals_edge_level():
    truth = _truth([0.6, 0.5, 0.4, 0.7], [0.2, 0.1, 0.15, 0.3])
    data, inv, z, M = _toy(truth, N=8000)
    ll, cred, _ = lcm._collapsed_loglik_and_cred(data, None, truth)
    ll_e, cred_e = _edge_level_loglik(M, truth)
    assert np.isclose(ll, ll_e, rtol=1e-10)
    assert np.allclose(cred[inv], cred_e, atol=1e-12)
    # joint = log p(o) + log p(r | o)
    assert np.isclose(ll, lcm.observation_loglik(truth, data) + lcm.conditional_loglik(truth, data), rtol=1e-10)


def test_numpyro_model_matches_numpy():
    """The jax model's likelihood factor equals the numpy evaluation at the same parameters."""
    import jax
    from numpyro.handlers import seed, trace
    truth = _truth([0.6, 0.5, 0.4, 0.7], [0.2, 0.1, 0.15, 0.3])
    data, _, _, _ = _toy(truth, N=3000, seed=3)
    for name in ("em_obs", "family_hb_obs"):
        model = lcm.build_model(data, lcm.MODEL_SPECS[name])
        tr = trace(seed(model, jax.random.PRNGKey(4))).get_trace()
        site = {k: np.asarray(v["value"]) for k, v in tr.items()}
        sample = dict(w=site["w"], theta=site["theta"], eta=site["eta"])
        ll_np, cred_np, _ = lcm._collapsed_loglik_and_cred(data, None, sample)
        ll_jax = float(np.asarray(tr["loglik"]["fn"].log_factor))
        assert np.isclose(ll_jax, ll_np, rtol=1e-8)
        assert np.allclose(site["cred"], cred_np, atol=1e-10)
        # eta rows of non-LIT annotators are NaN; LIT rows: column T free, other columns equal
        assert np.isnan(site["eta"][4:]).all()
        assert np.allclose(site["eta"][:4, 1:], site["eta"][:4, 1:2])


def test_reduces_to_mar_when_eta_equal():
    truth = _truth([0.3, 0.2, 0.5, 0.4], [0.3, 0.2, 0.5, 0.4])            # eta_T = eta_NonT
    data, inv, z, M = _toy(truth, N=6000, seed=5)
    mar = dict(w=truth["w"], theta=truth["theta"])
    c_obs = lcm.credibility_at(truth, data)
    c_mar = lcm.credibility_at(mar, data)
    assert np.allclose(c_obs, c_mar, atol=1e-12)
    # conditional log p(r | o) equals the MAR marginal likelihood; joint adds a z-free term
    assert np.isclose(lcm.conditional_loglik(truth, data), lcm.heldout_loglik(mar, data), rtol=1e-12)
    lit = np.isfinite(truth["eta"][:, 0])
    e = truth["eta"][lit, 0]
    Ro = data.R[:, lit]
    log_o = np.sum(data.n[:, None] * ((Ro >= 0) * np.log(e) + (Ro == -1) * np.log1p(-e)))
    assert np.isclose(lcm.heldout_loglik(truth, data), lcm.heldout_loglik(mar, data) + log_o, rtol=1e-12)
    # MAR models give the same result on eligibility-coded and on collapsed evidence
    plain = data.marginal()
    assert np.isclose(lcm.heldout_loglik(mar, plain), lcm.heldout_loglik(mar, data), rtol=1e-12)


def test_obs_model_requires_eligibility():
    truth = _truth([0.6, 0.5, 0.4, 0.7], [0.2, 0.1, 0.15, 0.3])
    data, _, _, _ = _toy(truth, N=2000)
    with pytest.raises(ValueError):
        lcm.build_model(data.marginal(), lcm.MODEL_SPECS["family_hb_obs"])
    with pytest.raises(ValueError):
        lcm.PatternData(data.R, data.n, FAM, ANN)              # -2 codes without the flag


def test_em_obs_recovers_truth():
    """Recovery on synthetic data generated from the extended model (two classes)."""
    truth = _truth([0.7, 0.6, 0.5, 0.8], [0.15, 0.1, 0.2, 0.25], S=2)
    data, inv, z, M = _toy(truth, N=40000, seed=11)
    fit = lcm.fit_map(data, "em_obs", n_restarts=4, seed=2)
    eta = fit.best["eta"]
    assert np.max(np.abs(eta[:4, 0] - truth["eta"][:4, 0])) < 0.06
    assert np.max(np.abs(eta[:4, 1] - truth["eta"][:4, 1])) < 0.02
    assert abs(fit.best["w"][0] - truth["w"][0]) < 0.01
    _, c_true, _ = lcm._collapsed_loglik_and_cred(data, None, truth)
    assert np.average(np.abs(fit.cred - c_true), weights=data.n) < 0.01


def test_family_hb_obs_recovers_truth():
    truth = _truth([0.7, 0.6, 0.5, 0.8], [0.15, 0.1, 0.2, 0.25])
    data, inv, z, M = _toy(truth, N=60000, seed=13)
    fit = lcm.fit_map(data, "family_hb_obs", n_restarts=4, seed=3)
    _, c_true, _ = lcm._collapsed_loglik_and_cred(data, None, truth)
    assert np.average(np.abs(fit.cred - c_true), weights=data.n) < 0.02
    assert abs(fit.best["w"][0] - truth["w"][0]) < 0.015
    assert np.max(np.abs(fit.best["eta"][:4, 0] - truth["eta"][:4, 0])) < 0.08
    assert np.max(np.abs(fit.best["eta"][:4, 1] - truth["eta"][:4, 1])) < 0.03
    ll_true, _, _ = lcm._collapsed_loglik_and_cred(data, None, truth)
    assert fit.loglik >= ll_true - 5.0


def test_simulate_informative_rates():
    truth = _truth([0.6, 0.5, 0.4, 0.7], [0.2, 0.1, 0.15, 0.3])
    data, inv, z, M = _toy(truth, N=200000, seed=21)
    for j in range(4):
        el = M[:, j] >= -1
        for s, rate in ((0, truth["eta"][j, 0]), (1, truth["eta"][j, 1])):
            m = el & (z == s)
            assert abs((M[m, j] >= 0).mean() - rate) < 0.02
    assert ((M[:, 4:] == -1).sum() == 0)                  # not modelled: unobserved coded -2


def test_conditional_class_posterior_is_prior_for_mar():
    truth = _truth([0.6, 0.5, 0.4, 0.7], [0.2, 0.1, 0.15, 0.3])
    data, _, _, _ = _toy(truth, N=3000)
    mar = dict(w=truth["w"], theta=truth["theta"])
    P = lcm.class_posteriors(mar, data, given="obs")
    assert np.allclose(P, truth["w"][None, :])
    Pall = lcm.class_posteriors(truth, data, given="all")
    assert np.allclose(Pall[:, 0], lcm.credibility_at(truth, data))


def test_obs_weight_interpolates_between_mar_and_full():
    import jax
    from numpyro.handlers import seed, trace
    truth = _truth([0.6, 0.5, 0.4, 0.7], [0.2, 0.1, 0.15, 0.3])
    data, _, _, _ = _toy(truth, N=4000, seed=8)
    mar = dict(w=truth["w"], theta=truth["theta"])
    # weight 1 = generative model; weight 0 = MAR credibility
    assert np.allclose(lcm.credibility_at(dict(truth, obs_weight=np.ones(7)), data), lcm.credibility_at(truth, data))
    assert np.allclose(lcm.credibility_at(dict(truth, obs_weight=np.zeros(7)), data), lcm.credibility_at(mar, data))
    # the jax model with weights matches the numpy evaluation
    ow = np.array([0.5, 0.1, 0.2, 0.3, 1.0, 1.0, 1.0])
    model = lcm.build_model(data, lcm.MODEL_SPECS["family_hb_obs"], prior={"obs_weight": ow})
    tr = trace(seed(model, jax.random.PRNGKey(1))).get_trace()
    site = {k: np.asarray(v["value"]) for k, v in tr.items()}
    assert np.allclose(site["obs_weight"][:4], ow[:4]) and np.isnan(site["obs_weight"][4:]).all()
    smp = dict(w=site["w"], theta=site["theta"], eta=site["eta"], obs_weight=site["obs_weight"])
    ll_np, cred_np, _ = lcm._collapsed_loglik_and_cred(data, None, smp)
    assert np.isclose(float(np.asarray(tr["loglik"]["fn"].log_factor)), ll_np, rtol=1e-8)
    assert np.allclose(site["cred"], cred_np, atol=1e-10)


# ----------------------------------------------------------------------------------------
# module helpers (synthetic data)
# ----------------------------------------------------------------------------------------
def test_simulation_helpers():
    from r2 import informative_observation as IO
    truth = _truth([0.6, 0.5, 0.4, 0.7], [0.2, 0.1, 0.15, 0.3])
    data, _, _, _ = _toy(truth, N=30000, seed=9)
    mar = dict(w=truth["w"], theta=truth["theta"])
    # conditional simulation keeps the real observation masks and the observed-edge count
    sim = IO.simulate_conditional(data, mar, seed=1)
    real = data.marginal()
    o_real = (real.R >= 0)
    keys_r, cnt_r = np.unique(o_real, axis=0, return_counts=False), None
    obs_sim = ((sim.R >= 0) * sim.n[:, None]).sum(0)
    obs_real = ((real.R >= 0) * real.n[:, None]).sum(0)
    assert np.allclose(obs_sim, obs_real)
    # joint simulation from the truth reproduces the observation rates of the real data
    sim2, st = IO.simulate_joint(data, truth, seed=2)
    ro = IO.real_observation_stats(data)
    assert np.nanmax(np.abs(st["obs_rate"] - ro["obs_rate"])) < 0.02
    assert np.allclose(st["nobs_lit_frac"].sum(), 1.0)
    # class-independent rates equal the empirical rates
    ei = IO.ignorable_eta(data)
    assert np.allclose(ei[:4, 0], ro["obs_rate"][:4]) and np.isnan(ei[4:]).all()
    # 3-level status score
    s = IO.status_score(data)
    assert set(np.unique(s)) <= {0.0, 1.0, 2.0}
    assert np.all((s == 2) == (data.R == 1).any(1))


# ----------------------------------------------------------------------------------------
# eligibility-coded evidence (real stored evidence; skipped when it is not built)
# ----------------------------------------------------------------------------------------
def _have_evidence(u="immune"):
    from r2 import evidence as E
    return (E.EV_DIR / u / "patterns.parquet").exists()


@pytest.mark.skipif(not _have_evidence(), reason="stored evidence not built")
def test_elig_patterns_refine_stored_evidence():
    from r2 import evidence as E
    out = E.build_elig_patterns("immune", write=False)
    R, n, fam, ann = E.load_patterns("immune")
    ep = np.asarray(E.load_edge_pattern("immune", mmap=False))
    # per edge: collapsing -2 -> -1 gives the stored pattern
    e2 = out["edge_pattern"]
    assert (e2 >= 0).sum() == (ep >= 0).sum()
    coll = np.where(out["R"] < -1, -1, out["R"])
    idx = np.flatnonzero(ep >= 0)
    assert np.array_equal(coll[e2[idx]], R[ep[idx]])
    assert np.array_equal(out["base_pattern"][e2[idx]], ep[idx])
    # codes: -1 only for LIT, observed entries of LIT always eligible
    lit = np.array([f == "LIT" for f in fam])
    assert not (out["R"][:, ~lit] == -1).any()


@pytest.mark.skipif(not _have_evidence(), reason="stored evidence not built")
def test_elig_matches_direct_bank_computation():
    """Eligibility (TF covered by the LIT annotator) and observation recomputed edge by edge
    from the annotator bank for a sample of TFs agree with the stored eligibility coding."""
    from r2 import evidence as E
    bank = E.bank_for_variant("prespecified_cc")
    tfs, genes = E.tissue_universe("immune")
    out = E.build_elig_patterns("immune", write=False)
    tfi, tf_list = E.universe_edge_tfs("immune")
    gs = set(genes)
    rng = np.random.default_rng(0)
    pick = rng.choice(len(tfs), size=12, replace=False)
    start = np.concatenate([[0], np.cumsum([len(genes) - (t in gs) for t in tfs])])
    for i in pick:
        t = tfs[i]
        targets = [g for g in genes if g != t]
        dense = E.evidence_for_edges(bank, [t] * len(targets), targets, mask="prespecified_cc")
        rows = out["R"][out["edge_pattern"][start[i]:start[i + 1]]]
        for j, f in enumerate(bank.family):
            if f == "LIT":
                elig = t in bank.cov[j]
                exp = np.where(dense[:, j] >= 0, dense[:, j], -1 if elig else -2)
            else:
                exp = np.where(dense[:, j] >= 0, dense[:, j], -2)
            assert np.array_equal(rows[:, j], exp), (t, bank.sources[j])


# ----------------------------------------------------------------------------------------
# CV fold fits (warm + random starts) and the gene-level diagnostic
# ----------------------------------------------------------------------------------------
def test_between_group_share():
    from r2 import informative_observation as IO
    rng = np.random.default_rng(0)
    g = rng.integers(0, 50, 20000)
    assert np.isclose(IO.between_group_share(g.astype(float) * 0.01, g), 1.0)       # a function of the group
    assert IO.between_group_share(rng.random(20000), g) < 0.01                        # independent of the group
    v = (g % 2) + rng.normal(0, 1.0, 20000)                                           # half/half
    assert 0.15 < IO.between_group_share(v, g) < 0.25
    assert np.isnan(IO.between_group_share(np.ones(10), np.arange(10)))


def test_cv_fit_warm_and_random_starts(tmp_path, monkeypatch):
    """``cv_fit`` reaches the optimum of ``lcm.fit_map`` on the same data; a warm start placed at
    that optimum is recognised as (one of) the best start(s); the weighted (composite) variant
    runs through the same code path."""
    import jax.numpy as jnp
    from jax.flatten_util import ravel_pytree
    from numpyro.infer.util import unconstrain_fn
    from r2 import fit_primary as FP
    from r2 import informative_observation as IO
    truth = _truth([0.7, 0.6, 0.5, 0.8], [0.15, 0.1, 0.2, 0.25], S=2)
    data, _, _, _ = _toy(truth, N=20000, seed=17)
    ref = lcm.fit_map(data, "em_obs", n_restarts=3, seed=IO.SEED)
    pt = FP._potential(data, "em_obs")
    z0 = unconstrain_fn(pt["model"], (), {}, {k: jnp.asarray(ref.best[k]) for k in pt["names"]})
    x0 = np.asarray(ravel_pytree(z0)[0], dtype=float)
    monkeypatch.setattr(IO, "fit_dir", lambda u, fid: tmp_path / fid)
    (tmp_path / "em_obs").mkdir()
    np.savez(tmp_path / "em_obs" / "optima_params.npz", x0=x0, x1=x0 + 0.5)
    fit = IO.cv_fit(data, "em_obs", "toy", n_random=2)
    assert [r["start"] for r in fit["starts"]] == ["warm_opt0", "warm_opt1", "random0", "random1"]
    assert abs(fit["log_post"] - ref.log_post) < 1e-3
    assert fit["n_within_1e3"] >= 2 and abs(fit["random_minus_warm"]) < 1e-3
    assert np.allclose(fit["best"]["eta"][:4], ref.best["eta"][:4], atol=1e-3)
    # the random starts are those of lcm.fit_map(n_restarts=n_random, seed): same end points
    ref2 = lcm.fit_map(data, "em_obs", n_restarts=2, seed=IO.SEED)
    lp_rand = [r["log_post"] for r in fit["starts"] if r["start"].startswith("random")]
    assert np.allclose(lp_rand, [r["log_post"] for r in ref2.restarts], atol=1e-6)
    # composite-likelihood variant: no stored optima for this fit id -> random starts only
    fw = IO.cv_fit(data, "em_obs", "toy", n_random=2, prior={"obs_weight": np.full(7, 0.05)}, fit_id="em_obs__tfw")
    assert [r["start"] for r in fw["starts"]] == ["random0", "random1"]
    assert "obs_weight" in fw["best"] and np.isnan(fw["random_minus_warm"])


@pytest.mark.skipif(not _have_evidence(), reason="stored evidence not built")
def test_universe_edge_targets_match_stored_edges():
    import pandas as pd
    from r2 import evidence as E
    from r2 import informative_observation as IO
    ev = pd.read_parquet(E.EV_DIR / "immune" / "evidence.parquet", columns=["tf", "target"])
    gi, genes = IO.universe_edge_targets("immune")
    ti, tfs = E.universe_edge_tfs("immune")
    assert len(gi) == len(ti) == len(ev)
    assert np.array_equal(np.asarray(genes, dtype=object)[gi], ev["target"].to_numpy(dtype=object))
    assert np.array_equal(np.asarray(tfs, dtype=object)[ti], ev["tf"].to_numpy(dtype=object))


def test_centered_parametrisation_same_density():
    """Centred and non-centred annotator-level coordinates define the same joint density of
    the model parameters: log p_centred(q, ...) = log p_noncentred(z, ...) - n_ann log tau_q
    per centred type (change of variables q = mu_f + tau z); deterministic outputs agree."""
    import jax
    from numpyro.handlers import seed, trace
    from numpyro.infer.util import log_density
    truth = _truth([0.6, 0.5, 0.4, 0.7], [0.2, 0.1, 0.15, 0.3])
    data, _, _, _ = _toy(truth, N=3000, seed=4)
    spec = lcm.MODEL_SPECS["family_hb_obs"]
    m_nc = lcm.build_model(data, spec)
    tr = trace(seed(m_nc, jax.random.PRNGKey(7))).get_trace()
    vals_nc = {k: v["value"] for k, v in tr.items() if v["type"] == "sample" and not v.get("is_observed")}
    lp_nc, _ = log_density(m_nc, (), {}, vals_nc)
    for cen in (("b", "eT", "eN"), True):
        types = ("b", "a", "g", "eT", "eN") if cen is True else cen
        m_c = lcm.build_model(data, spec, {"centered": cen})
        vals_c = {k: v for k, v in vals_nc.items() if k not in {f"zj_{q}" for q in types}}
        for q in types:
            vals_c[q] = tr[q]["value"]
        lp_c, tr_c = log_density(m_c, (), {}, vals_c)
        jac = sum(np.size(tr[q]["value"]) * np.log(float(tr[f"tau_{q}"]["value"])) for q in types)
        assert np.isclose(float(lp_c), float(lp_nc) - jac, rtol=1e-10, atol=1e-8)
        for k in ("theta", "eta", "cred"):
            assert np.allclose(np.asarray(tr_c[k]["value"]), np.asarray(tr[k]["value"]), equal_nan=True, atol=1e-12)
        assert not any(f"zj_{q}" in tr_c for q in types)
    with pytest.raises(ValueError):
        lcm.build_model(data, spec, {"centered": ("x",)})
    # default (non-centred) model unchanged: the same sample sites as before the option existed
    assert {"zj_b", "zj_a", "zj_g", "zj_eT", "zj_eN"} <= set(vals_nc)


def _store_toy_fit(tmp_path, data, fit, fit_id="em_obs"):
    """Write the files of a stored MAP fit that the module helpers read (toy data)."""
    import json
    import jax.numpy as jnp
    from jax.flatten_util import ravel_pytree
    from numpyro.infer.util import unconstrain_fn
    from r2 import fit_primary as FP
    pt = FP._potential(data, "em_obs")
    z0 = unconstrain_fn(pt["model"], (), {}, {k: jnp.asarray(fit.best[k]) for k in pt["names"]})
    x0, _ = FP.newton_polish(pt, np.asarray(ravel_pytree(z0)[0], dtype=float))
    d = tmp_path / fit_id
    d.mkdir(exist_ok=True)
    np.savez(d / "optima_params.npz", x0=x0)
    (d / "meta.json").write_text(json.dumps({"model": "em_obs", "prior": {"overrides": {}}}))
    return d


def test_laplace_eta_sd(tmp_path, monkeypatch):
    """Laplace standard deviations of the observation logits: finite for the modelled (LIT)
    annotators only, consistent with a covariance (triangle inequality for the difference) and
    shrinking like 1 / sqrt(n) when every edge count is multiplied by 4."""
    from r2 import informative_observation as IO
    truth = _truth([0.7, 0.6, 0.5, 0.8], [0.15, 0.1, 0.2, 0.25], S=2)
    data, _, _, _ = _toy(truth, N=20000, seed=23)
    big = lcm.PatternData(data.R, data.n * 4, FAM, ANN, True)
    out, w_hat = {}, {}
    for name, dd in (("n1", data), ("n4", big)):
        fit = lcm.fit_map(dd, "em_obs", n_restarts=2, seed=1)
        w_hat[name] = float(np.asarray(fit.best["w"])[0])
        d = _store_toy_fit(tmp_path, dd, fit, fit_id=f"em_obs_{name}")
        monkeypatch.setattr(IO, "fit_dir", lambda u, fid, d=d: d)
        out[name] = IO.laplace_eta_sd("toy", "em_obs", dd, 0)
    sd = out["n1"]
    assert np.all(np.isfinite(sd["sd_T"][:4])) and np.all(sd["sd_T"][:4] > 0) and np.isnan(sd["sd_T"][4:]).all()
    assert np.all(sd["sd_diff"][:4] <= sd["sd_T"][:4] + sd["sd_NonT"][:4] + 1e-12)
    assert np.all(sd["sd_diff"][:4] >= np.abs(sd["sd_T"][:4] - sd["sd_NonT"][:4]) - 1e-12)
    ratio = out["n4"]["sd_NonT"][:4] / sd["sd_NonT"][:4]
    assert np.all((ratio > 0.45) & (ratio < 0.55))
    # class-T weight: a finite scalar, not below the binomial standard error of the logit of a
    # proportion with known class labels (the classes are latent, so the information is smaller)
    # and of the same order; shrinking like 1 / sqrt(n)
    wT = w_hat["n1"]
    se_binom = 1.0 / np.sqrt(data.n.sum() * wT * (1 - wT))
    assert isinstance(sd["sd_logit_w_T"], float) and 0.98 * se_binom < sd["sd_logit_w_T"] < 5 * se_binom
    assert 0.45 < out["n4"]["sd_logit_w_T"] / sd["sd_logit_w_T"] < 0.55
    monkeypatch.setattr(IO, "fit_dir", lambda u, fid: tmp_path / "missing")
    assert IO.laplace_eta_sd("toy", "em_obs", data, 0) is None


def test_add_w_T_intervals_and_freshness_helpers(tmp_path, monkeypatch):
    """The Laplace interval of w_T reaches the MAP informative-observation rows of the fit
    summary only; ``older_than`` flags missing and stale outputs; ``family_hb_nuts`` is read
    only when a ``family_hb_obs_nuts`` fit exists; ``all`` does not run NUTS."""
    import pandas as pd
    from r2 import informative_observation as IO
    summ = pd.DataFrame(dict(universe=["u", "u", "u"], fit_id=["em_obs", "em", "em_obs__tfw"],
                             inference=["MAP", "MAP", "MAP"], w_T=[0.3, 0.01, 0.02]))
    par = pd.DataFrame(dict(universe=["u"] * 4, fit_id=["em_obs", "em_obs", "em_obs", "em_obs__tfw"],
                            inference=["MAP"] * 4, optimum=[0, 0, 1, 0], w_T=[0.3, 0.3, 0.7, 0.02],
                            w_T_lo=[0.29, 0.29, 0.69, np.nan], w_T_hi=[0.31, 0.31, 0.71, np.nan]))
    out = IO.add_w_T_intervals(summ, par).set_index("fit_id")
    assert (out.loc["em_obs", "w_T_lo"], out.loc["em_obs", "w_T_hi"]) == (0.29, 0.31)       # best optimum, not the swap
    assert out.loc["em_obs", "w_T_interval"] == IO.W_T_LAPLACE
    assert np.isnan(out.loc["em", "w_T_lo"]) and out.loc["em", "w_T_interval"] == ""         # MAR row untouched
    assert np.isnan(out.loc["em_obs__tfw", "w_T_lo"]) and out.loc["em_obs__tfw", "w_T_interval"] == ""   # no Hessian
    with pytest.raises(AssertionError):
        IO.add_w_T_intervals(summ.assign(w_T=[0.31, 0.01, 0.02]), par)
    assert IO.older_than({"a": 5.0}, {"x": 2.0, "y": 4.0}) == []
    assert IO.older_than({"a": 3.0, "b": None}, {"x": 2.0, "y": 4.0, "z": None}) == [
        "missing input: z", "stale output (older than its inputs): a", "missing output: b"]
    monkeypatch.setattr(IO, "fit_dir", lambda u, fid: tmp_path / u / fid)
    assert IO.mar_comparators("u") == ("family_hb", "em")
    (tmp_path / "u" / IO.NUTS_FIT).mkdir(parents=True)
    (tmp_path / "u" / IO.NUTS_FIT / "meta.json").write_text("{}")
    assert IO.mar_comparators("u") == ("family_hb", "em", "family_hb_nuts")
    # 'all' = every reported stage, without NUTS and without the exploratory conditional fits
    called = []
    for name in ("run_evidence", "run_map", "run_map_tfw", "run_map_cond", "run_nuts", "run_ppc", "run_cv",
                 "run_heldout", "build_tables", "check_outputs"):
        monkeypatch.setattr(IO, name, lambda *a, _n=name, **k: called.append(_n))
    monkeypatch.setattr(IO, "MOD", tmp_path / "mod")
    IO.main(["all"])
    assert called == ["run_evidence", "run_map", "run_map_tfw", "run_ppc", "run_cv", "run_heldout", "build_tables"]


def test_alt_optimum_rules(tmp_path, monkeypatch):
    import pandas as pd
    from r2 import informative_observation as IO
    monkeypatch.setattr(IO, "fit_dir", lambda u, fid: tmp_path / fid)
    for fid in ("family_hb_obs", "family_hb_obs__tfw", "em_obs"):
        (tmp_path / fid).mkdir()
    # generative fit: optimum 1 is a near-duplicate of the best, 2 the label-swapped one, 3 not converged
    pd.DataFrame(dict(optimum=[0, 1, 2, 3], spearman_w_vs_best=[1.0, 0.9999, -0.99, -0.98],
                      laplace_log_mass=[-10.0, -56.0, -500.0, np.nan])).to_csv(tmp_path / "family_hb_obs" / "optima.csv",
                                                                             index=False)
    assert IO.alt_optimum("u", "family_hb_obs") == (2, "label_swapped")
    assert IO.alt_suffix("label_swapped") == "_swap"
    # weighted fit: the second optimum, whatever its correlation
    pd.DataFrame(dict(optimum=[0, 1], spearman_w_vs_best=[1.0, 0.8],
                      laplace_log_mass=[-10.0, -21.0])).to_csv(tmp_path / "family_hb_obs__tfw" / "optima.csv", index=False)
    assert IO.alt_optimum("u", "family_hb_obs__tfw") == (1, "second_optimum")
    # a single optimum, or none stored
    pd.DataFrame(dict(optimum=[0], spearman_w_vs_best=[1.0], laplace_log_mass=[-10.0])).to_csv(
        tmp_path / "em_obs" / "optima.csv", index=False)
    assert IO.alt_optimum("u", "em_obs") == (None, "")
    assert IO.alt_optimum("u", "em_obs__tfw") == (None, "")


def test_obs_conditional_likelihood_factor_and_recovery():
    """``obs_conditional``: the likelihood factor of the jax model equals the numpy
    ``conditional_loglik`` (= joint - log p(o)); the conditional-likelihood MAP recovers the
    credibility of data generated from the informative-observation model; with eta_T = eta_NonT
    the conditional likelihood equals the MAR marginal likelihood."""
    import jax
    from numpyro.handlers import seed, trace
    truth = _truth([0.6, 0.5, 0.4, 0.7], [0.2, 0.1, 0.15, 0.3])
    data, _, _, _ = _toy(truth, N=3000, seed=31)
    for name in ("em_obs", "family_hb_obs"):
        model = lcm.build_model(data, lcm.MODEL_SPECS[name], {"obs_conditional": True})
        tr = trace(seed(model, jax.random.PRNGKey(5))).get_trace()
        site = {k: np.asarray(v["value"]) for k, v in tr.items()}
        smp = dict(w=site["w"], theta=site["theta"], eta=site["eta"])
        ll_jax = float(np.asarray(tr["loglik"]["fn"].log_factor))
        assert np.isclose(ll_jax, lcm.conditional_loglik(smp, data), rtol=1e-8)
        assert np.isclose(ll_jax, lcm.heldout_loglik(smp, data) - lcm.observation_loglik(smp, data), rtol=1e-8)
        assert np.allclose(site["cred"], lcm.credibility_at(smp, data), atol=1e-10)      # cred = P(T | o, r)
    truth2 = _truth([0.7, 0.6, 0.5, 0.8], [0.15, 0.1, 0.2, 0.25], S=2)
    data2, _, _, _ = _toy(truth2, N=40000, seed=37)
    fit = lcm.fit_map(data2, "em_obs", n_restarts=4, seed=2, prior={"obs_conditional": True})
    c_true = lcm.credibility_at(truth2, data2)
    obs = (data2.R >= 0).any(1)
    assert np.average(np.abs(fit.cred - c_true)[obs], weights=data2.n[obs]) < 0.02
    assert lcm.conditional_loglik(fit.best, data2) >= lcm.conditional_loglik(truth2, data2) - 5.0
    assert np.all(fit.best["eta"][:4, 0] > fit.best["eta"][:4, 1])                      # direction recovered
    eq = _truth([0.3, 0.2, 0.5, 0.4], [0.3, 0.2, 0.5, 0.4])
    d3, _, _, _ = _toy(eq, N=4000, seed=41)
    assert np.isclose(lcm.conditional_loglik(eq, d3), lcm.heldout_loglik(dict(w=eq["w"], theta=eq["theta"]), d3))
