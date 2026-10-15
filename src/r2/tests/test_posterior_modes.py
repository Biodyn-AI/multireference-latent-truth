"""Tests for the Family-HB identifiability module (src/r2/posterior_modes.py)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.stats import multivariate_normal, multivariate_t

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from r2 import lcm  # noqa: E402
from r2 import posterior_modes as PM  # noqa: E402


def _toy(seed=0, N=20000):
    """Two planted annotator families with family-specific burst classes."""
    rng = np.random.default_rng(seed)
    fam = ["A"] * 3 + ["B"] * 3
    K = len(fam)
    fa = np.array(fam)
    beta, alpha, gam = np.full(K, 0.02), np.full(K, 0.6), np.full(K, 0.5)
    w = np.array([0.05, 0.85, 0.05, 0.05])
    theta = np.stack([alpha, beta, np.where(fa == "A", gam, beta), np.where(fa == "B", gam, beta)], axis=1)
    cover = rng.random((N, K)) < 0.8
    z = rng.choice(4, size=N, p=w)
    rep = (rng.random((N, K)) < theta.T[z]).astype(np.int8)
    M = np.where(cover, rep, -1).astype(np.int8)
    data, _ = lcm.PatternData.from_matrix(M, fam, [f"s{j}" for j in range(K)])
    return data, dict(w=w, theta=theta)


def _bridge(logq, draws_fit, draws_est, rng, n1_eff=None):
    mu, L = PM.mvn_fit(draws_fit)
    prop = mu + rng.standard_normal((len(draws_est), len(mu))) @ L.T
    lge, lgp = PM.mvn_logpdf(draws_est, mu, L), PM.mvn_logpdf(prop, mu, L)
    return PM.bridge_sampling(logq(draws_est), lge, logq(prop), lgp, n1_eff=n1_eff), (lge, lgp, prop)


def test_mvn_logpdf_matches_scipy():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((500, 4)) @ np.diag([1, 2, 0.5, 3]) + 1.0
    mu, L = PM.mvn_fit(X)
    Y = rng.standard_normal((20, 4))
    assert np.allclose(PM.mvn_logpdf(Y, mu, L), multivariate_normal(mu, L @ L.T).logpdf(Y))


@pytest.mark.parametrize("kind", ["gauss", "student"])
def test_bridge_sampling_recovers_known_log_z(kind):
    rng = np.random.default_rng(1)
    d, lz = 8, -1234.5
    A = rng.standard_normal((d, d))
    S = A @ A.T / d + np.eye(d)
    mu = rng.standard_normal(d)
    dist_ = multivariate_normal(mu, S) if kind == "gauss" else multivariate_t(mu, S, df=6)
    logq = lambda X: dist_.logpdf(X) + lz  # noqa: E731
    post = dist_.rvs(6000, random_state=2)
    res, (lge, lgp, prop) = _bridge(logq, post[:3000], post[3000:], rng)
    assert res["converged"]
    assert abs(res["log_z"] - lz) < 0.05
    re = np.sqrt(PM.bridge_re2(logq(post[3000:]), lge, logq(prop), lgp, res["log_z"], 3000))
    assert 0 < re < 0.05


def test_bridge_sampling_two_separated_modes_gives_mode_probabilities():
    """Unnormalised density = p0 N(m0, I) + p1 N(m1, I) far apart: bridge sampling on draws from each
    component (restricted to its mode) recovers log(p_m) + const and hence the mode probabilities."""
    rng = np.random.default_rng(3)
    d, c = 5, 50.0
    m0, m1 = np.zeros(d), np.full(d, 30.0)
    p = np.array([0.97, 0.03])

    def logq(X):
        return c + np.logaddexp(np.log(p[0]) + multivariate_normal(m0).logpdf(X),
                                np.log(p[1]) + multivariate_normal(m1).logpdf(X))

    lz = []
    for m in (m0, m1):
        post = m + rng.standard_normal((4000, d))
        res, _ = _bridge(logq, post[:2000], post[2000:], rng)
        lz.append(res["log_z"])
    lz = np.array(lz)
    pm = np.exp(lz - np.logaddexp.reduce(lz))
    assert np.allclose(lz, c + np.log(p), atol=0.05)
    assert abs(pm[1] - 0.03) < 0.002


def test_bridge_handles_proposal_draws_outside_support():
    rng = np.random.default_rng(4)
    post = rng.standard_normal((4000, 3))
    logq = lambda X: np.where(np.abs(X).max(1) < 10, multivariate_normal(np.zeros(3)).logpdf(X), -np.inf)  # noqa: E731
    res, _ = _bridge(logq, post[:2000], post[2000:], rng)
    assert abs(res["log_z"]) < 0.05


def test_group_optima_and_assign():
    lp = np.array([-10.0, -10.0005, -12.0, -10.0009, -12.0002, -30.0])
    g = PM.group_optima(lp)
    assert [sorted(x) for x in g] == [[0, 1, 3], [2, 4], [5]]
    assert PM.assign_optimum(-12.01, [-10.0, -12.0], tol=0.05) == 1
    assert PM.assign_optimum(-11.0, [-10.0, -12.0], tol=0.05) == -1


def test_set_stats_realised_fdr():
    cred = np.array([0.99, 0.98, 0.9, 0.5, 0.01])
    n = np.array([10.0, 10.0, 10.0, 10.0, 100.0])
    t = np.array([9.0, 10.0, 5.0, 5.0, 1.0])            # true edges per pattern
    s = PM.set_stats(cred, n, t)
    # Strong = first two patterns (BFDR (0.1+0.2)/20 = 0.015; adding 0.9 -> 1.3/30 = 0.043; adding 0.5 -> 6.3/40)
    assert s["strong_n"] == 30 and np.isclose(s["strong_claimed_bfdr"], 1.3 / 30)
    assert np.isclose(s["strong_realised_fdr"], (1 + 0 + 5) / 30)
    assert s["ge_moderate_n"] == 40 and np.isclose(s["ge_moderate_realised_fdr"], 11 / 40)


def test_weighted_quantiles_mixture_matches_bruteforce():
    rng = np.random.default_rng(5)
    A = rng.normal(0, 1, (400, 7))
    B = rng.normal(3, 1, (100, 7))
    q = PM.weighted_quantiles_mixture(A, B, 0.8, qs=(0.025, 0.5, 0.975), chunk=3)
    # brute force: replicate A rows 1x and B rows so that the per-draw weights match (0.8/400 vs 0.2/100)
    V = np.concatenate([A, B], 0)
    W = np.concatenate([np.full(400, 0.8 / 400), np.full(100, 0.2 / 100)])
    for j in range(7):
        o = np.argsort(V[:, j], kind="mergesort")
        cw = np.cumsum(W[o])
        for i, qq in enumerate((0.025, 0.5, 0.975)):
            assert q[i, j] == V[o, j][np.argmax(cw >= qq - 1e-12)]


def test_multistart_reproduces_fit_map_restarts():
    data, _ = _toy()
    fit = lcm.fit_map(data, "family_hb", n_restarts=3, seed=11)
    pt = PM.potential(data)
    ms = PM.multistart(pt, 3, 11)
    lp_fit = [r["log_post"] for r in fit.restarts]
    assert np.allclose([r["log_post"] for r in ms], lp_fit, rtol=0, atol=1e-9)
    best = ms[int(np.argmax([r["log_post"] for r in ms]))]
    c = lcm.credibility_at(pt["post"](best["x"]), data)
    assert np.allclose(c, fit.cred, atol=1e-10)


def test_flat_from_constrained_roundtrip_and_potential():
    data, _ = _toy(N=5000)
    pt = PM.potential(data)
    rng = np.random.default_rng(6)
    X = rng.uniform(-1, 1, (5, len(pt["flat0"])))
    samples = {}
    for x in X:
        s = pt["post"](x)
        for k in pt["names"]:
            samples.setdefault(k, []).append(s[k])
    samples = {k: np.stack(v) for k, v in samples.items()}
    Xb = PM.flat_from_constrained(pt, samples)
    assert np.allclose(Xb, X, atol=1e-8)
    assert np.allclose(PM.pot_batch(pt, X), [float(pt["pot"](x)) for x in X])


def test_laplace_at_matches_fit_primary_formula():
    data, _ = _toy(N=8000)
    pt = PM.potential(data)
    ms = PM.multistart(pt, 2, 3)
    x = ms[int(np.argmax([r["log_post"] for r in ms]))]["x"]
    la = PM.laplace_at(pt, x)
    H = np.asarray(pt["hess"](la["x"]))
    ld = np.linalg.slogdet(0.5 * (H + H.T))[1]
    assert np.isclose(la["laplace_log_mass"], la["log_post"] + 0.5 * len(x) * np.log(2 * np.pi) - 0.5 * ld)
    assert la["log_post"] >= max(r["log_post"] for r in ms) - 1e-9


def test_mode_init_params_starts_in_requested_basin():
    data, _ = _toy(N=8000)
    pt = PM.potential(data)
    ms = PM.multistart(pt, 2, 3)
    x = PM.laplace_at(pt, ms[int(np.argmax([r["log_post"] for r in ms]))]["x"])["x"]
    lp = -float(pt["pot"](x))
    init, rec = PM.mode_init_params(pt, x, lp, num_chains=2, seed=1)
    assert all(abs(e - lp) <= PM.BASIN_TOL for e in rec["basin_check_end_log_post"])
    assert all(v.shape[0] == 2 for v in init.values())


def test_ece_bins_excludes_near_zero_bin():
    g = pd.DataFrame(dict(bin=[0, 1, 9], n_edges=[1000, 10, 10], sum_pred=[10.0, 1.5, 9.5], n_true=[0, 1, 5]))
    # bins 1-9: |0.15 - 0.1| * 10 + |0.95 - 0.5| * 10 over 20 = 0.25
    assert np.isclose(PM.ece_bins(g), 0.25)
    assert PM.ece_bins(g, lo_bin=0) < 0.25


def test_wilson_interval():
    lo, hi = PM._wilson(8, 10)
    assert 0.48 < lo < 0.50 and 0.93 < hi < 0.95
    assert np.isnan(PM._wilson(0, 0)[0])


def test_gap_bin_labels():
    lab = PM.gap_bin_labels(np.array([0.0, 0.5, 2.5, 7.5, 11.6, 108.0, np.nan]))
    assert lab.tolist() == ["[0, 1)", "[0, 1)", "[2.5, 5)", "[5, 10)", "[10, 20)", "[20, inf)", "no_second_optimum"]


def test_replay_restart_matches_multistart():
    data, _ = _toy(N=8000)
    pt = PM.potential(data)
    ms = PM.multistart(pt, 3, 21)
    x = PM.replay_restart(data, "family_hb", 2, 3, 21, pt=pt)
    assert np.allclose(x, ms[2]["x"], atol=1e-12)


def test_snapshot_dir_fallback(tmp_path, monkeypatch):
    monkeypatch.setattr(PM, "SNAP", tmp_path / "snap")
    assert PM.map_dir("immune") == PM.fit_dir("immune", "family_hb")
    d = tmp_path / "snap" / "immune" / "family_hb"
    d.mkdir(parents=True)
    (d / "meta.json").write_text("{}")
    assert PM.map_dir("immune") == d


def test_sim_summary_counts_and_bins():
    df = pd.DataFrame(dict(universe="immune", replicate=range(6), gap=[0.1, 0.2, 3.0, 8.0, 9.0, np.nan],
                           dominant_is_truth=[False, True, True, True, True, True],
                           dominant_is_truthstart=[False, True, True, True, True, True],
                           truth_mode_found=True, dominant_correct_strict=[False, True, True, True, True, True],
                           laplace_p2=0.1, dominant_strong_realised_fdr=[0.9, 0.05, 0.04, 0.03, 0.05, 0.04],
                           dominant_strong_n=100.0, dominant_strong_claimed_bfdr=0.05))
    df["gap_bin"] = PM.gap_bin_labels(df.gap.to_numpy(float))
    s = PM.sim_summary(df)
    v = s[(s.stratum == "gap_bin [0, 1)") & (s.quantity == "p_dominant_is_truth")]
    assert float(v.value.iloc[0]) == 0.5 and int(v.n_replicates.iloc[0]) == 2
    v = s[(s.stratum == "gap_ge_real_immune") & (s.quantity == "p_dominant_is_truth")]
    assert float(v.value.iloc[0]) == 1.0 and int(v.n_replicates.iloc[0]) == 2
    v = s[(s.stratum == "dominant_not_truth") & (s.quantity == "dominant_strong_realised_fdr_mean")]
    assert np.isclose(float(v.value.iloc[0]), 0.9)


def test_bridge_from_draws_recovers_log_z_with_chain_structure():
    """``bridge_from_draws`` (first-half/second-half split per chain, ESS-based N1, repeated random
    splits) on 4 chains of exact draws from a correlated Gaussian with known normaliser."""
    import jax.numpy as jnp
    rng = np.random.default_rng(7)
    d, lz = 6, -321.0
    A = rng.standard_normal((d, d))
    S = A @ A.T / d + 0.5 * np.eye(d)
    mu = rng.standard_normal(d)
    dist_ = multivariate_normal(mu, S)
    pt = {"pot_batch": lambda X: -(jnp.asarray(dist_.logpdf(np.asarray(X))) + lz)}
    X = dist_.rvs(4 * 600, random_state=8)
    md = dict(X=X, chain=np.repeat(np.arange(4), 600), logq=dist_.logpdf(X) + lz, source="test")
    res = PM.bridge_from_draws("toy", 0, pt, md, n_splits=5)
    assert abs(res["log_z"] - lz) < 0.05
    assert res["log_z_split_sd"] < 0.05 and 0 < res["re_fs"] < 0.05
    assert len(res["log_z_splits"]) == 5 and res["frac_proposal_finite"] == 1.0


def test_laplace_p2_and_map_optimum_cred_from_snapshot(tmp_path, monkeypatch):
    monkeypatch.setattr(PM, "SNAP", tmp_path / "snap")
    d = tmp_path / "snap" / "immune" / "family_hb"
    d.mkdir(parents=True)
    (d / "meta.json").write_text("{}")
    pd.DataFrame(dict(optimum=[0, 1, 2], restarts=["3 0", "1 2", "4"],
                      laplace_log_mass=[-100.0, -100.0 - np.log(99.0), -150.0])).to_csv(d / "optima.csv", index=False)
    cr = np.arange(5 * 4, dtype=float).reshape(5, 4)
    np.save(d / "cred_restarts.npy", cr)
    assert np.isclose(PM.laplace_p2_top2("immune"), 0.01)
    assert np.array_equal(PM.map_optimum_cred("immune", 0), cr[3])
    assert np.array_equal(PM.map_optimum_cred("immune", 1), cr[1])


def test_mvt_logpdf_matches_scipy():
    rng = np.random.default_rng(9)
    X = rng.standard_normal((300, 4)) @ np.diag([1, 2, 0.5, 3]) - 0.5
    mu, L = PM.mvn_fit(X)
    Ls = PM.mvt_scale(L, 5.0)
    Y = rng.standard_normal((20, 4)) * 2
    assert np.allclose(PM.mvt_logpdf(Y, mu, Ls, 5.0), multivariate_t(mu, Ls @ Ls.T, df=5).logpdf(Y))
    # the scale matrix reproduces the covariance L L'
    assert np.allclose(Ls @ Ls.T * 5.0 / 3.0, L @ L.T)


def test_centred_coords_roundtrip_values_and_jacobian():
    """y holds the model's own centred quantities (deterministic sites muf_q and q), the inverse map
    returns x, and log |det dy/dx| equals the log determinant of a finite-difference Jacobian."""
    data, _ = _toy(N=3000)
    pt = PM.potential(data)
    cc = PM.centred_coords(pt, data.fam_idx)
    assert cc["name"] == "centred" and cc["types"] == ["b", "a", "g"]
    rng = np.random.default_rng(10)
    d = len(pt["flat0"])
    X = rng.uniform(-1, 1, (4, d))
    Y, ld = cc["to_y"](X)
    Xb, ldb = cc["to_x"](Y)
    assert np.allclose(Xb, X, atol=1e-10) and np.allclose(ld, ldb)
    idx = {k: np.asarray(v).astype(int) for k, v in pt["unravel"](np.arange(d, dtype=float)).items()}
    for i in range(len(X)):
        s = pt["post"](X[i])
        for q in "bag":
            assert np.allclose(Y[i, idx[f"zf_{q}"]], s[f"muf_{q}"], atol=1e-9)
            assert np.allclose(Y[i, idx[f"zj_{q}"]], s[q], atol=1e-9)
            assert np.isclose(Y[i, idx[f"tau_{q}"]], np.log(s[f"tau_{q}"]))       # scales stay on the log scale
    eps = 1e-6
    J = np.stack([(cc["to_y"](X[0] + eps * e)[0][0] - cc["to_y"](X[0] - eps * e)[0][0]) / (2 * eps) for e in np.eye(d)], axis=1)
    sign, logdet = np.linalg.slogdet(J)
    assert sign > 0 and np.isclose(logdet, ld[0], atol=1e-6)
    assert PM.centred_coords({"pot_batch": None}, data.fam_idx) is None             # no hierarchical sites: identity


def test_bridge_centred_coordinates_recover_known_log_z_where_raw_coordinates_are_poor():
    """Target that is exactly normal in the centred coordinates (known normaliser): the centred bridge
    estimate recovers log Z; the same draws in the non-centred coordinates are far from normal and the
    plain recipe has a much larger error."""
    import jax.numpy as jnp
    data, _ = _toy(N=3000)
    pt0 = PM.potential(data)
    cc = PM.centred_coords(pt0, data.fam_idx)
    d, lz = len(pt0["flat0"]), -777.0
    rng = np.random.default_rng(11)
    idx = {k: np.asarray(v).astype(int) for k, v in pt0["unravel"](np.arange(d, dtype=float)).items()}
    mu, sd = rng.normal(0, 0.2, d), np.full(d, 0.2)
    for q in "bag":                         # annotator parameters 2 units from their family mean, tau uncertain:
        mu[idx[f"zj_{q}"]] += 2.0           # zj = (q_j - mu_f) / tau is then far from normal
        sd[idx[f"tau_{q}"]] = 0.6
    A = rng.standard_normal((d, d))
    Rm = 0.3 * (A @ A.T / d) + 0.7 * np.eye(d)
    S = Rm * np.outer(sd, sd)
    dist_ = multivariate_normal(mu, S)

    def pot_batch(X):                       # U(x) = -[log N(y(x)) + log |det dy/dx| + log Z]
        Y, ld = cc["to_y"](np.asarray(X))
        return jnp.asarray(-(dist_.logpdf(Y) + ld + lz))

    pt = dict(pot_batch=pot_batch, unravel=pt0["unravel"], flat0=pt0["flat0"])
    Yd = dist_.rvs(4 * 800, random_state=12)
    X, _ = cc["to_x"](Yd)
    md = dict(X=X, chain=np.repeat(np.arange(4), 800), logq=-np.asarray(pot_batch(X)), source="test")
    res = PM.bridge_from_draws("toy", 0, pt, md, n_splits=4, coords=cc, proposal="mvn", n2_mult=3)
    assert res["coords"] == "centred" and res["converged"] and res["n2"] == 3 * res["n1"]
    assert abs(res["log_z"] - lz) < 0.05 and res["re_fs"] < 0.05 and res["log_z_split_sd"] < 0.05
    rt = PM.bridge_from_draws("toy", 0, pt, md, n_splits=2, coords=cc, proposal="mvt", n2_mult=3)
    assert abs(rt["log_z"] - lz) < 0.2
    raw = PM.bridge_from_draws("toy", 0, pt, md, n_splits=2, coords=None, proposal="mvn", n2_mult=1)
    assert raw["coords"] == "unconstrained_noncentred" and raw["re_fs"] > 5 * res["re_fs"]


def test_p2_and_snapshot_check(tmp_path, monkeypatch):
    assert np.isclose(PM._p2(0.0, 0.0), 0.5) and np.isclose(PM._p2(np.log(99.0), 0.0), 0.01)
    assert PM._p2(1e4, 0.0) < 1e-300 or PM._p2(1e4, 0.0) == 0.0
    monkeypatch.setattr(PM, "SNAP", tmp_path / "snap")
    monkeypatch.setattr(PM, "FITS", tmp_path / "fits")
    monkeypatch.setattr(PM, "UNIVERSES", ("immune",))
    rng = np.random.default_rng(13)
    x0, x1, th = rng.standard_normal(5), rng.standard_normal(5), rng.random((3, 2))
    for root, shift in ((tmp_path / "snap" / "immune" / "family_hb", 0.0),
                        (tmp_path / "fits" / "immune" / PM.VARIANT / "family_hb", 1e-9)):
        root.mkdir(parents=True)
        pd.DataFrame(dict(log_post=[-10.0, -17.5 + shift], laplace_log_mass=[-60.0, -66.0])).to_csv(root / "optima.csv", index=False)
        np.savez(root / "optima_params.npz", x0=x0 + shift, x1=x1)
        np.savez(root / "params.npz", w=np.array([0.1, 0.9]), theta=th + shift)
        np.save(root / "cred.npy", np.array([0.2, 0.3]))
    df = PM.snapshot_check()
    assert bool(df.consistent.iloc[0]) and df.max_abs_dx_top2.iloc[0] < 1e-8
    np.savez(tmp_path / "fits" / "immune" / PM.VARIANT / "family_hb" / "optima_params.npz", x0=x0 + 0.1, x1=x1)
    with pytest.raises(RuntimeError):
        PM.snapshot_check()


def test_load_sim_ignores_cached_alt_files(tmp_path, monkeypatch):
    import json
    monkeypatch.setattr(PM, "MODOUT", tmp_path)
    od = tmp_path / "sim" / "immune"
    od.mkdir(parents=True)
    for r in (0, 1, 150):
        (od / f"rep{r:03d}.json").write_text(json.dumps(dict(replicate=r)))
    (od / "rep000_alt.json").write_text(json.dumps(dict(alt_mode_index=1)))
    assert [r["replicate"] for r in PM.load_sim("immune")] == [0, 1]      # rep 150 >= N_SIM, alt file skipped


def test_logit_fit_matches_known_mle_and_flags_separation():
    rng = np.random.default_rng(3)
    x = rng.normal(0, 2, 4000)
    y = (rng.random(4000) < 1 / (1 + np.exp(-(0.5 + 0.8 * x)))).astype(float)
    fit = PM.logit_fit(x, y)
    assert np.allclose(fit["params"], [0.5, 0.8], atol=0.12)
    # score equations hold at the optimum
    p = 1 / (1 + np.exp(-(fit["params"][0] + fit["params"][1] * x)))
    assert abs(np.sum(y - p)) < 1e-6 and abs(np.sum(x * (y - p))) < 1e-6
    se = np.sqrt(np.diag(fit["cov"]))
    assert np.all(se > 0) and se[1] < 0.1
    with pytest.raises(RuntimeError):
        PM.logit_fit(np.arange(10.0), (np.arange(10) >= 5).astype(float))   # perfectly separated


def test_sim_summary_logistic_rows():
    rng = np.random.default_rng(5)
    gap = rng.exponential(2.0, 60)
    ok = rng.random(60) < 1 / (1 + np.exp(-(0.2 + 0.6 * gap)))
    df = pd.DataFrame(dict(universe="immune", replicate=range(60), gap=gap, dominant_is_truth=ok,
                           dominant_is_truthstart=ok, truth_mode_found=True, dominant_correct_strict=ok,
                           laplace_p2=0.1))
    df["gap_bin"] = PM.gap_bin_labels(df.gap.to_numpy(float))
    s = PM.sim_summary(df)
    lr = s[s.stratum == "logistic_at_real_gap_immune"]
    assert len(lr) == 1 and 0 < float(lr.value.iloc[0]) < 1
    if PM.REAL_GAPS["immune"] <= gap.max():
        assert float(lr.ci_lo.iloc[0]) <= float(lr.value.iloc[0]) <= float(lr.ci_hi.iloc[0])
    else:                                   # real gap beyond every simulated gap: no Wald interval (verifier m5)
        assert np.isnan(float(lr.ci_lo.iloc[0])) and np.isnan(float(lr.ci_hi.iloc[0]))
    assert int(lr.n_replicates_gap_ge_real.iloc[0]) == int(np.sum(gap >= PM.REAL_GAPS["immune"]))
    assert not (s.quantity == "fit_failed").any()


def test_record_decisions_replaces_only_own_rows(tmp_path):
    p = tmp_path / "eval_analysis_decisions.csv"
    pd.DataFrame([dict(module="governance", id="g1", date="2026-10-08", status="post_hoc", item="x", note="y"),
                  dict(module="posterior_modes", id="old", date="2026-10-01", status="post_hoc", item="z", note="")]
                 ).to_csv(p, index=False)
    PM.record_decisions(p)
    PM.record_decisions(p)          # idempotent
    d = pd.read_csv(p)
    assert (d.module == "governance").sum() == 1
    own = d[d.module == "posterior_modes"]
    assert len(own) == len(PM.ANALYSIS_DECISIONS) and "old" not in set(own.id)
    assert set(own.status) == {"post_hoc"}


def test_truth_vs_alt_uses_truthstart_when_truth_mode_missed():
    rec = dict(truthstart_optimum=-1, truthstart_log_post=-100.5,
               optima=[dict(log_post=-100.0), dict(log_post=-150.0), dict(log_post=-170.0)])
    a = dict(alt_mode_index=0, alt_found_distinct=True, truth_minus_alt_log_post=-70.0)
    found = PM.truth_vs_alt(rec, a, True)
    assert found["truth_minus_alt_log_post"] == -70.0 and found["dominant_is_alt"]
    missed = PM.truth_vs_alt(rec, a, False)
    assert np.isclose(missed["truth_minus_alt_log_post"], -0.5) and missed["dominant_is_alt"]
    assert missed["truth_log_post_source"].startswith("truthstart")
    rec2 = dict(rec, truthstart_optimum=0)           # truth-start lands on the other-mode optimum: not distinct
    m2 = PM.truth_vs_alt(rec2, a, False)
    assert np.isnan(m2["truth_minus_alt_log_post"]) and not m2["alt_found_distinct"] and not m2["dominant_is_alt"]


# ----------------------------------------------------------------------------------------
# design-faithful simulators (modes2: fix of the modes verifier's B1)
# ----------------------------------------------------------------------------------------
def _toy_cc_design(seed=0, N=4000, n_tf=8):
    """Small synthetic curated-conditional design: 3 LIT annotators with random fixed masks, 2 MOTIF/COEXP
    annotators with TF coverage, a random channel map on the LIT-observed cells."""
    rng = np.random.default_rng(seed)
    fam = ["LIT"] * 3 + ["MOTIF", "COEXP"]
    lit, cc = np.arange(3), np.array([3, 4])
    obs = rng.random((N, 3)) < 0.6
    obs[~obs.any(1), 0] = True
    tf = rng.integers(0, n_tf, N)
    cov_tf = rng.random((n_tf, 2)) < 0.7
    M = np.full((N, 5), -1, np.int8)
    M[:, :3] = np.where(obs, 0, -1)
    codes = (M.astype(np.int64) + 1) @ (3 ** np.arange(5))
    uc, inv = np.unique(codes, return_inverse=True)
    R = np.stack([M[np.flatnonzero(inv == k)[0]] for k in range(len(uc))])
    n = np.bincount(inv, minlength=len(uc)).astype(float)
    sc_codes = PM.lit_mask_codes(R, lit)
    strata, s_of_p = np.unique(sc_codes, return_inverse=True)
    des = dict(universe="toy", R=R, n=n, family=fam, annotators=[f"a{j}" for j in range(5)], lit=lit, cc=cc,
               dom_pattern=inv.astype(np.int64), tf_idx=tf, chan=obs & (rng.random((N, 3)) < 0.5),
               cov=cov_tf[tf], frac_tf=np.full((3, n_tf), 0.5), rep_real=np.zeros((N, 3), bool),
               stratum_codes=strata, stratum_of_pattern=s_of_p.reshape(-1))
    w = np.array([0.1, 0.7, 0.1, 0.1])
    theta = np.array([[0.7, 0.05, 0.5, 0.05]] * 3 + [[0.6, 0.1, 0.1, 0.5]] * 2)
    return des, w, theta


def test_simulate_cc_regenerates_cc_masks_from_simulated_lit_reports():
    des, w, theta = _toy_cc_design()
    sim, cc, st = PM.simulate_cc(des, w, theta, seed=3)
    R = sim.R
    lit_rep = (R[:, :3] == 1).any(1)
    cc_obs = (R[:, 3:] >= 0).any(1)
    # every MOTIF/COEXP-observed edge carries >= 1 LIT report, and the LIT masks are the real ones
    assert np.all(lit_rep[cc_obs])
    assert st["n_cc_observed"] == st["n_cc_observed_with_lit_report"] == sim.n[cc_obs].sum()
    real_codes = np.sort(np.unique(PM.lit_mask_codes(des["R"], des["lit"])))
    assert set(PM.lit_mask_codes(R, des["lit"])) <= set(real_codes)
    assert sim.n.sum() == len(des["dom_pattern"]) == cc.sum() and st["n_true"] == cc[:, 0].sum()
    # with certain LIT reporting and full channel membership, Cond = every LIT-observed edge
    th1 = theta.copy()
    th1[:3] = 1.0
    d2 = dict(des, chan=des["R"][des["dom_pattern"]][:, :3] >= 0)
    _, _, st2 = PM.simulate_cc(d2, w, th1, seed=1)
    assert st2["n_cond"] == len(des["dom_pattern"])
    assert st2["n_cc_observed"] == int(des["cov"].any(1).sum())
    # deterministic in the seed
    sim_b, cc_b, _ = PM.simulate_cc(des, w, theta, seed=3)
    assert np.array_equal(sim_b.R, sim.R) and np.array_equal(cc_b, cc)


def test_expected_cc_counts_match_simulation_mean():
    des, w, theta = _toy_cc_design(seed=2)
    ex = PM.expected_cc_counts(des, w, theta, chunk=777)
    sims = [PM.simulate_cc(des, w, theta, seed=s)[2] for s in range(60)]
    for k_sim, k_ex in (("n_cond", "n_cond"), ("n_cc_observed", "n_cc_observed"), ("n_ge1_lit_report", "n_ge1_lit_report")):
        v = np.array([s[k_sim] for s in sims], float)
        assert abs(v.mean() - ex[k_ex]) < 4 * v.std(ddof=1) / np.sqrt(len(v)) + 1e-9, k_sim
    pos = np.array([s["n_reported"][:3] for s in sims], float).mean(0)
    assert np.allclose(pos, ex["lit_positives"], rtol=0.06)
    lo = PM.expected_cc_counts(des, w, theta, mixed="lo")["n_cc_observed"]
    hi = PM.expected_cc_counts(des, w, theta, mixed="hi")["n_cc_observed"]
    assert lo <= ex["n_cc_observed"] <= hi


def test_stratum_class_weights_reproduce_lit_positives_and_reduce_to_global_weights():
    des, w, theta = _toy_cc_design(seed=4)
    # 'real' reports simulated from the model, then the stratified weights at the generating parameters
    sim, _, _ = PM.simulate_cc(des, w, theta, seed=9)
    W = PM.stratum_class_weights(des, w, theta)
    assert W.shape == (len(des["stratum_codes"]), len(w)) and np.allclose(W.sum(1), 1)
    # with uninformative annotators (theta equal across classes) the posterior is the prior: W = w everywhere
    th0 = np.repeat(theta[:, 1:2], 4, axis=1)
    assert np.allclose(PM.stratum_class_weights(des, w, th0), w[None, :])
    # true credibility: stratified prior equal to w gives the MAR credibility
    c_mar = lcm.credibility_at(dict(w=w, theta=theta), sim)
    c_str = PM.true_credibility(sim, w, theta, des, np.tile(w, (len(W), 1)))
    assert np.allclose(c_mar, c_str, atol=1e-10)
    # M-step identity: expected LIT positives under W = sum over edges of posterior-weighted rates
    d_real = dict(des, R=des["R"])
    post = PM.class_posteriors(des["R"], w, theta)
    lit = des["lit"]
    m = des["R"][:, lit] >= 0
    target = ((des["n"][:, None] * m) * (post @ theta[lit].T)).sum(0)
    ex = PM.expected_cc_counts(d_real, w, theta, W=W)["lit_positives"]
    assert np.allclose(ex, target, rtol=1e-8)


def test_cc_design_reconstructs_the_real_immune_masks():
    des = PM.cc_design("immune")
    assert des["mask_reconstruction_exact"]
    R, p, cc, lit = des["R"], des["dom_pattern"], des["cc"], des["lit"]
    assert des["n_cc_observed_real"] == int(((R[p][:, cc] >= 0).any(1)).sum())
    assert not (des["chan"] & ~(R[p][:, lit] >= 0)).any()           # channel only on LIT-observed cells
    real = PM.pattern_design_stats(lcm.PatternData(R, des["n"], des["family"], des["annotators"]))
    assert real["n_edges_cc_observed"] == real["n_edges_cc_observed_with_lit_report"] == des["n_cc_observed_real"]
    assert sum(des["chan_rule_counts"].values()) == int((R[p][:, lit] >= 0).sum())


def test_simulate_for_fixed_masks_replays_the_recovery_seed():
    data = PM.load_data("immune")
    truth = PM.truth_params("immune")
    sim, cc, st, oracle, wT = PM.simulate_for("immune", 3, truth, data)
    ref, cref = PM.FS.simulate_fixed_masks(data, truth["w"], truth["theta"], seed=PM.derived_seed("recovery", "immune", 3))
    assert np.array_equal(sim.R, ref.R) and np.array_equal(cc, cref)
    assert np.allclose(oracle, lcm.credibility_at(dict(w=truth["w"], theta=truth["theta"]), ref))
    assert PM.sim_seed("immune_cc", 3) != PM.sim_seed("immune_ccs", 3) != PM.sim_seed("immune", 3)
    assert {PM.sim_simulator(x) for x in PM.SIM_LABELS} == set(PM.SIMULATORS)


def test_sim_summary_separates_simulators_and_flags_gaps_beyond_range():
    rng = np.random.default_rng(11)
    frames = []
    for simr in ("fixed_masks", "faithful_cc"):
        gap = rng.exponential(2.0, 40)
        ok = rng.random(40) < 1 / (1 + np.exp(-(0.3 + 0.7 * gap)))
        frames.append(pd.DataFrame(dict(universe="immune", simulator=simr, replicate=range(40), gap=gap,
                                        dominant_is_truth=ok, dominant_is_truthstart=ok, truth_mode_found=True,
                                        dominant_correct_strict=ok, laplace_p2=0.1)))
    df = pd.concat(frames, ignore_index=True)
    df["gap_bin"] = PM.gap_bin_labels(df.gap.to_numpy(float))
    s = PM.sim_summary(df)
    a = s[(s.stratum == "all") & (s.quantity == "p_dominant_is_truth")]
    assert set(a.simulator) == {"fixed_masks", "faithful_cc"} and (a.n_replicates == 40).all()
    g = s[(s.stratum == "logistic_at_real_gap_global")]
    assert g.ci_lo.isna().all() and g.ci_hi.isna().all()          # 108 nats beyond every simulated gap
    k = s[(s.stratum == "logistic_at_real_gap_kidney")]
    assert k.note.str.contains("immune calibration").all()
    assert (s[s.stratum.str.startswith("logistic_at")].logit_lr_stat > 0).all()
    pr = s[(s.stratum == "real_gap_immune") & (s.quantity == "frac_sim_gap_below_real_gap_all")]
    for simr, v in zip(pr.simulator, pr.value):
        assert np.isclose(v, (df[df.simulator == simr].gap < PM.REAL_GAPS["immune"]).mean())


def test_logit_fit_lr_statistic():
    rng = np.random.default_rng(8)
    x = rng.normal(0, 1, 500)
    y = (rng.random(500) < 1 / (1 + np.exp(-(0.2 + 1.0 * x)))).astype(float)
    f = PM.logit_fit(x, y)
    p = 1 / (1 + np.exp(-(f["params"][0] + f["params"][1] * x)))
    ll = np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))
    ll0 = np.sum(y * np.log(y.mean()) + (1 - y) * np.log(1 - y.mean()))
    assert np.isclose(f["lr_stat"], 2 * (ll - ll0)) and f["lr_stat"] > 20
