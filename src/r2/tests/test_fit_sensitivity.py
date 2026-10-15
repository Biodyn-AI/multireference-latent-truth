"""Tests for the label-free sensitivity / model-check module (src/r2/fit_sensitivity.py)."""
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from r2 import fit_sensitivity as FS  # noqa: E402
from r2 import lcm  # noqa: E402


def _toy(seed=0, N=30000):
    """Three planted annotator blocks with block-specific burst classes."""
    rng = np.random.default_rng(seed)
    fam = ["A"] * 3 + ["B"] * 3 + ["C"] * 2
    K = len(fam)
    beta = np.full(K, 0.02)
    alpha = np.full(K, 0.6)
    gam = np.full(K, 0.5)
    w = np.array([0.05, 0.80, 0.08, 0.07])        # T, N, B_A, B_B
    fa = np.array(fam)
    theta = np.stack([alpha, beta, np.where(fa == "A", gam, beta), np.where(fa == "B", gam, beta)], axis=1)
    cover = rng.random((N, K)) < 0.8
    z = rng.choice(4, size=N, p=w)
    rep = (rng.random((N, K)) < theta.T[z]).astype(np.int8)
    M = np.where(cover, rep, -1).astype(np.int8)
    data, inv = lcm.PatternData.from_matrix(M, fam, [f"s{j}" for j in range(K)])
    return data, inv, M, z, dict(w=w, theta=theta)


def test_weighted_spearman_matches_expanded():
    rng = np.random.default_rng(1)
    x = rng.integers(0, 5, 40).astype(float)
    y = x + rng.integers(0, 3, 40)
    w = rng.integers(1, 4, 40)
    xe, ye = np.repeat(x, w), np.repeat(y, w)
    assert np.isclose(FS.weighted_spearman(x, y, w), spearmanr(xe, ye).correlation)


def test_weighted_auroc_matches_sklearn():
    rng = np.random.default_rng(2)
    s = rng.integers(0, 6, 30).astype(float)
    npos = rng.integers(0, 4, 30)
    nneg = rng.integers(0, 4, 30)
    yt = np.concatenate([np.ones(npos.sum()), np.zeros(nneg.sum())])
    ys = np.concatenate([np.repeat(s, npos), np.repeat(s, nneg)])
    assert np.isclose(FS.weighted_auroc(s, npos, nneg), roc_auc_score(yt, ys))


def test_bfdr_select_whole_tie_blocks():
    cred = np.array([0.99, 0.97, 0.97, 0.6, 0.2])
    n = np.array([10, 5, 5, 10, 100])
    sel = FS.bfdr_select(cred, n, 0.05)
    # 0.99 block: 0.01; + 0.97 block (both patterns): (0.1+0.3)/20 = 0.02; + 0.6: (0.4+4)/30 = 0.147
    assert sel.tolist() == [True, True, True, False, False]
    sel2 = FS.bfdr_select(cred, n, 0.20)
    assert sel2.tolist() == [True, True, True, True, False]
    assert not FS.bfdr_select(np.array([0.5]), np.array([3.0]), 0.05).any()


def test_topk_tie_break_and_jaccard():
    c = np.array([5, 4, 4, 4, 1, 0], float)
    key = np.array([0.1, 0.9, 0.3, 0.5, 0.2, 0.4])
    top = FS.topk_edges(c, key, 3)
    assert sorted(top.tolist()) == [0, 1, 3]
    assert FS.jaccard_idx(np.array([0, 1, 3]), np.array([0, 2, 3])) == pytest.approx(0.5)


def test_simulate_fixed_masks_keeps_masks_and_rates():
    data, inv, M, z, truth = _toy()
    sim, cc = FS.simulate_fixed_masks(data, truth["w"], truth["theta"], seed=3, chunk=7000)
    # observation masks are exactly the real ones (only observed edges are simulated)
    obs_real = data.R >= 0
    keep = obs_real.any(1)
    mk_r, ir = np.unique(obs_real[keep], axis=0, return_inverse=True)
    cnt_r = np.bincount(ir.reshape(-1), weights=data.n[keep])
    mk_s, is_ = np.unique(sim.R >= 0, axis=0, return_inverse=True)
    cnt_s = np.bincount(is_.reshape(-1), weights=sim.n)
    assert np.array_equal(mk_r, mk_s) and np.allclose(cnt_r, cnt_s)
    assert np.isclose(cc.sum(), data.n[keep].sum())
    # class frequencies and per-annotator positive rates agree with the model
    assert np.allclose(cc.sum(0) / cc.sum(), truth["w"], atol=0.01)
    obs = sim.R >= 0
    pos = sim.R == 1
    rate = (pos * sim.n[:, None]).sum(0) / (obs * sim.n[:, None]).sum(0)
    expect = truth["theta"] @ truth["w"]
    assert np.allclose(rate, expect, atol=0.012)
    # deterministic in the seed
    sim2, cc2 = FS.simulate_fixed_masks(data, truth["w"], truth["theta"], seed=3, chunk=7000)
    assert np.array_equal(sim.R, sim2.R) and np.array_equal(sim.n, sim2.n)


def test_coreport_counts_match_edge_level():
    data, inv, M, z, truth = _toy(N=4000)
    cc = FS.coreport_counts(data)
    j, k = 0, 4
    both = (M[:, j] >= 0) & (M[:, k] >= 0)
    assert cc["nco"][j, k] == both.sum()
    assert cc["n11"][j, k] == np.sum(both & (M[:, j] == 1) & (M[:, k] == 1))
    assert cc["n10"][j, k] == np.sum(both & (M[:, j] == 1) & (M[:, k] == 0))
    assert cc["n01"][j, k] == np.sum(both & (M[:, j] == 0) & (M[:, k] == 1))
    assert cc["n00"][j, k] == np.sum(both & (M[:, j] == 0) & (M[:, k] == 0))
    phi, lor = FS.phi_and_lor(cc)
    a = M[both, j].astype(float)
    b = M[both, k].astype(float)
    assert np.isclose(phi[j, k], np.corrcoef(a, b)[0, 1])


def test_datadriven_clusters_recover_planted_blocks():
    data, *_ = _toy()
    fam, info = FS.datadriven_families(data, 3)
    # annotators of the same planted block share a cluster and blocks are separated
    groups = {}
    for f_true, f_dd in zip(data.family, fam):
        groups.setdefault(f_true, set()).add(f_dd)
    assert all(len(v) == 1 for v in groups.values())
    assert len({next(iter(v)) for v in groups.values()}) == 3


def test_tf_folds_grouped_and_counts_partition():
    data, inv, M, z, truth = _toy(N=5000)
    rng = np.random.default_rng(4)
    tf_of_edge = rng.integers(0, 37, len(inv))
    ep = inv.astype(np.int64).copy()
    ep[:3] = -1                                   # self-pairs are ignored
    fold_of_tf = FS.tf_folds(37, 5, seed=11)
    assert sorted(np.bincount(fold_of_tf).tolist()) == [7, 7, 7, 8, 8]
    counts = FS.fold_pattern_counts(ep, fold_of_tf[tf_of_edge], data.P, 5)
    assert np.allclose(counts.sum(0), np.bincount(ep[ep >= 0], minlength=data.P))
    for f in range(5):
        tfs_f = set(tf_of_edge[(fold_of_tf[tf_of_edge] == f)])
        assert all(fold_of_tf[t] == f for t in tfs_f)


def test_ppc_statistics_and_pairs():
    data, inv, M, z, truth = _toy(N=6000)
    pairs = FS.ppc_pairs(data)
    st = FS.ppc_statistics(data, pairs)
    obs = (M >= 0).any(1)
    npos = np.minimum((M == 1).sum(1), FS.NREP_CAP)
    expect = np.bincount(npos[obs], minlength=FS.NREP_CAP + 1) / obs.sum()
    assert np.allclose(st["nrep_frac"], expect)
    assert st["lor_pairs"].shape == (len(pairs),)
    # within-block co-reporting is stronger than between-block co-reporting
    fam = np.array(data.family)
    within = fam[pairs[:, 0]] == fam[pairs[:, 1]]
    assert st["lor_pairs"][within].mean() > st["lor_pairs"][~within].mean()


def test_calibration_bins():
    cred = np.array([0.05, 0.15, 0.95, 1.0])
    n = np.array([10, 10, 4, 1])
    t = np.array([1, 2, 4, 1])
    cb = FS.calibration_bins(cred, n, t, 10)
    assert cb["n_edges"].tolist()[0] == 10 and cb["n_edges"].tolist()[9] == 5
    assert np.isclose(cb.loc[9, "sum_pred"], 0.95 * 4 + 1.0)
    assert cb.loc[1, "n_true"] == 2


def test_derived_seed_deterministic():
    assert FS.derived_seed("a", 1) == FS.derived_seed("a", 1)
    assert FS.derived_seed("a", 1) != FS.derived_seed("a", 2)


def test_conditional_annotator_check_bruteforce():
    data, inv, M, z, truth = _toy(N=3000)
    out = FS.conditional_annotator_check(data, truth["w"], truth["theta"])
    w, th = truth["w"], truth["theta"]
    j = 2
    O = E_ = 0.0
    for p in range(data.P):
        r = data.R[p]
        if r[j] < 0:
            continue
        lp = np.log(w).copy()
        for k in range(data.K):
            if k == j or r[k] < 0:
                continue
            lp += np.log(th[k]) if r[k] == 1 else np.log1p(-th[k])
        post = np.exp(lp - np.logaddexp.reduce(lp))
        pi = post @ th[j]
        O += data.n[p] * (r[j] == 1)
        E_ += data.n[p] * pi
    assert np.isclose(out["observed"][j], O) and np.isclose(out["expected"][j], E_)
    # at the true parameters the conditional check is well calibrated
    zj = (out["observed"] - out["expected"]) / np.sqrt(out["var"])
    assert np.all(np.abs(zj) < 4)


def test_lit_block_subset_matches_edge_level():
    data, inv, M, z, truth = _toy(N=5000)
    fam = ["LIT"] * 5 + ["MOTIF"] * 2 + ["COEXP"]
    d2 = lcm.with_families(data, fam)
    cols = FS.lit_columns(d2)
    assert cols == [0, 1, 2, 3, 4]
    sub = FS.subset_annotators(d2, cols)
    assert sub.K == 5 and sub.family == ["LIT"] * 5 and np.isclose(sub.n.sum(), data.n.sum())
    st = FS.ppc_statistics(sub)
    Ml = M[:, cols]
    obs = (Ml >= 0).any(1)
    npos = np.minimum((Ml == 1).sum(1), FS.NREP_CAP)
    expect = np.bincount(npos[obs], minlength=FS.NREP_CAP + 1) / obs.sum()
    assert np.allclose(st["nrep_frac"], expect)


def test_map_from_start_reproduces_map_optimum():
    data, inv, M, z, truth = _toy(N=8000)
    fit = lcm.fit_map(data, "family_hb", n_restarts=3, seed=5)
    ts = FS.map_from_start(data, "family_hb", fit.best)
    # starting at the optimum: the start value equals the stored optimum and L-BFGS stays there
    assert np.isclose(ts["log_post_at_start"], fit.log_post, atol=1e-6)
    assert ts["log_post"] >= fit.log_post - 1e-6
    assert np.max(np.abs(ts["cred"] - fit.cred)) < 1e-4


def test_map_optima_reproduces_fit_map_restarts_and_laplace():
    data, inv, M, z, truth = _toy(N=6000)
    fit = lcm.fit_map(data, "family_hb", n_restarts=4, seed=9)
    opt = FS.map_optima(data, "family_hb", n_restarts=4, seed=9)
    # same starting points -> same best optimum; every restart is assigned to exactly one optimum
    assert np.isclose(opt[0]["log_post"], fit.log_post, atol=1e-6)
    assert sorted(r for o in opt for r in o["restarts"]) == [0, 1, 2, 3]
    lp_fit = sorted([r["log_post"] for r in fit.restarts], reverse=True)
    assert np.isclose(max(lp_fit), opt[0]["log_post"], atol=1e-6)
    # Laplace mass: finite at a strict local maximum (positive-definite Hessian)
    assert opt[0]["hess_min_eig"] > 0 and np.isfinite(opt[0]["laplace_log_mass"])
    assert np.isclose(opt[0]["laplace_log_mass"],
                      opt[0]["log_post"] + 0.5 * opt[0]["n_params"] * np.log(2 * np.pi) - 0.5 * opt[0]["log_det_hessian"])


def test_select_mode_chains_keeps_highest_likelihood_mode():
    data, inv, M, z, truth = _toy(N=6000)
    w, th = truth["w"], truth["theta"]
    # a worse-fitting parameter point (a pure label permutation would have the same likelihood)
    w2, th2 = w.copy(), th.copy()
    th2[:, 0] = 0.3
    W = np.stack([w] * 10 + [w2] * 10 + [w] * 10)
    TH = np.stack([th] * 10 + [th2] * 10 + [th] * 10)
    chain = np.repeat([0, 1, 2], 10)
    keep, ll = FS.select_mode_chains(W, TH, chain, data, n_eval=5)
    assert keep == [0, 2] and ll[0] > ll[1]
