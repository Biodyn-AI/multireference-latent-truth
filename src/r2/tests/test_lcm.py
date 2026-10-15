"""Tests for the latent-class credibility models (src/r2/lcm.py)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from r2 import lcm  # noqa: E402


def _toy(seed=0, N=60000):
    rng = np.random.default_rng(seed)
    fam = ["LIT"] * 4 + ["MOTIF"] * 2 + ["COEXP"]
    K = len(fam)
    beta = np.array([0.01, 0.02, 0.005, 0.01, 0.05, 0.08, 0.03])
    alpha = np.array([0.35, 0.25, 0.15, 0.30, 0.45, 0.50, 0.30])
    gamma = np.array([0.30, 0.25, 0.20, 0.30, 0.60, 0.55, 0.03])
    w = np.array([0.06, 0.84, 0.06, 0.04])          # T, N, B_LIT, B_MOTIF
    theta = np.stack([alpha, beta,
                      np.where(np.array(fam) == "LIT", gamma, beta),
                      np.where(np.array(fam) == "MOTIF", gamma, beta)], axis=1)
    cover = rng.random((N, K)) < np.array([0.7, 0.5, 0.4, 0.6, 0.9, 0.9, 0.8])
    z = rng.choice(4, size=N, p=w)
    rep = (rng.random((N, K)) < theta.T[z]).astype(np.int8)
    M = np.where(cover, rep, -1).astype(np.int8)
    data, inv = lcm.PatternData.from_matrix(M, fam, [f"s{j}" for j in range(K)])
    return data, inv, z, dict(w=w, theta=theta)


def test_pattern_loglik_matches_edge_level():
    data, inv, z, truth = _toy(N=5000)
    spec = lcm.MODEL_SPECS["family_hb"]
    ll, cred, lm = lcm._collapsed_loglik_and_cred(data, spec, truth)
    # edge-level recomputation
    obs = data.R[inv] >= 0
    pos = data.R[inv] == 1
    th = truth["theta"]
    lj = np.log(truth["w"])[None, :] + (obs * pos) @ np.log(th) + (obs * ~pos) @ np.log1p(-th)
    assert np.isclose(ll, np.logaddexp.reduce(lj, axis=1).sum())


def test_family_hb_recovers_truth():
    data, inv, z, truth = _toy()
    fit = lcm.fit_map(data, "family_hb", n_restarts=4, seed=1)
    _, cred_true, _ = lcm._collapsed_loglik_and_cred(data, lcm.MODEL_SPECS["family_hb"], truth)
    # credibility close to the oracle posterior, pattern-weighted
    err = np.average(np.abs(fit.cred - cred_true), weights=data.n)
    assert err < 0.03, err
    assert abs(fit.best["w"][0] - truth["w"][0]) < 0.02
    # MAP has to dominate the truth in log posterior terms only approximately; check loglik
    ll_true, _, _ = lcm._collapsed_loglik_and_cred(data, lcm.MODEL_SPECS["family_hb"], truth)
    assert fit.loglik >= ll_true - 5.0


if __name__ == "__main__":
    import time
    t = time.time(); test_pattern_loglik_matches_edge_level(); print("ok1", time.time() - t)
    data, inv, z, truth = _toy()
    print("P patterns", data.P)
    for m in lcm.MODEL_SPECS:
        t = time.time()
        f = lcm.fit_map(data, m, n_restarts=4, seed=1)
        _, ct, _ = lcm._collapsed_loglik_and_cred(data, lcm.MODEL_SPECS["family_hb"], truth)
        err = np.average(np.abs(f.cred - ct), weights=data.n)
        auc_true = None
        c_edge = f.cred[inv]
        from sklearn.metrics import roc_auc_score
        print(m, f.summary()["n_converged"], round(f.loglik, 1), "w_T", round(float(f.best["w"][0]), 4),
              "MAE vs oracle", round(err, 4), "AUROC(z==T)", round(roc_auc_score(z == 0, c_edge), 4),
              f"{time.time()-t:.1f}s")
    t = time.time(); test_family_hb_recovers_truth(); print("ok2", time.time() - t)
