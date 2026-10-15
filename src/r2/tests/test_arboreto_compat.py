"""Tests of the arboreto early-stopping compatibility layer (src/r2/arboreto_compat.py).

Run: .venv/bin/python -m pytest subproject_24_multireference_latent_truth_model/src/r2/tests/test_arboreto_compat.py -q
The cross-version test needs a python with scikit-learn <= 1.2 (numpy + sklearn
only), given by R2_SKLEARN_REF_PYTHON (default: ~/anaconda3/envs/econ/bin/python);
it is skipped when that interpreter is missing.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SRC = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SRC))
from r2 import arboreto_compat as AX  # noqa: E402

from sklearn.ensemble import GradientBoostingRegressor  # noqa: E402

SGBM = {"learning_rate": 0.01, "n_estimators": 5000, "max_features": 0.1, "subsample": 0.9}
SEED = 20261007
REF_PY = os.environ.get("R2_SKLEARN_REF_PYTHON",
                        str(Path.home() / "anaconda3/envs/econ/bin/python"))


def _problems():
    rng = np.random.default_rng(0)
    n, p = 1500, 30
    X = rng.normal(size=(n, p)).astype(np.float32)
    ys = [(np.sin(X[:, k]) * 2 + X[:, k + 1] * X[:, k + 2] + rng.normal(scale=1.0 + k, size=n)).astype(np.float32)
          for k in range(3)]
    ys.append(np.where(rng.random(n) < 0.01, rng.exponential(3, size=n), 0).astype(np.float32))  # rare target
    ys.append(np.zeros(n, dtype=np.float32))                                                    # unexpressed
    return X, ys


class _Reference:
    """Independent same-subset improvement: prediction before round i is
    reconstructed as F_i - lr * tree_i(X) (no stored copies)."""

    def __init__(self):
        self.impr = []

    def __call__(self, i, reg, loc):
        oob = ~loc["sample_mask"]
        after = loc["raw_predictions"][:, 0]
        before = after - reg.learning_rate * reg.estimators_[i, 0].predict(loc["X"])
        y = loc["y"][oob]
        self.impr.append(np.mean((y - before[oob]) ** 2) - np.mean((y - after[oob]) ** 2))
        return False


def test_improvement_equals_independent_reference():
    X, ys = _problems()
    for y in ys[:3]:
        mon = AX.SameOOBEarlyStopMonitor(25)
        reg = GradientBoostingRegressor(random_state=SEED, **SGBM).fit(X, y, monitor=mon)
        r = len(reg.estimators_)
        ref = _Reference()
        GradientBoostingRegressor(random_state=SEED, **{**SGBM, "n_estimators": r}).fit(X, y, monitor=ref)
        np.testing.assert_allclose(mon.oob_improvement, ref.impr, rtol=1e-6, atol=1e-10)
        # stopping round = first round >= 24 whose 25-round mean improvement is negative
        w = np.convolve(ref.impr, np.ones(25) / 25, "valid")
        assert r == 25 + int(np.argmax(w < 0)) and w[-1] < 0


def test_monitor_does_not_change_the_fit():
    X, ys = _problems()
    y = ys[0]
    mon = AX.SameOOBEarlyStopMonitor(25)
    reg = GradientBoostingRegressor(random_state=SEED, **SGBM).fit(X, y, monitor=mon)
    r = len(reg.estimators_)
    plain = GradientBoostingRegressor(random_state=SEED, **{**SGBM, "n_estimators": r}).fit(X, y)
    assert np.array_equal(reg.feature_importances_, plain.feature_importances_)
    assert np.array_equal(reg.predict(X), plain.predict(X))


def test_rounds_learnable_rare_and_constant_targets():
    X, ys = _problems()
    rounds = []
    for y in ys:
        reg = GradientBoostingRegressor(random_state=SEED, **SGBM).fit(X, y, monitor=AX.SameOOBEarlyStopMonitor(25))
        rounds.append(len(reg.estimators_))
    assert min(rounds[:3]) > 150          # learnable targets boost well past the window
    assert rounds[3] < 60                 # near-constant rare target stops early
    assert rounds[4] == SGBM["n_estimators"]   # constant target: improvement exactly 0, never < 0


@pytest.mark.skipif(AX.sklearn_version() < "1.3", reason="only sklearn >= 1.3 has the cross-subset definition")
def test_original_monitor_is_broken_here():
    from arboreto.core import EarlyStopMonitor
    import arboreto.core as ACORE
    orig = getattr(ACORE, "_r2_original_EarlyStopMonitor", EarlyStopMonitor)
    X, ys = _problems()
    reg = GradientBoostingRegressor(random_state=SEED, **SGBM).fit(X, ys[0], monitor=orig(25))
    assert len(reg.estimators_) < 60


def test_install_patches_arboreto_core():
    import arboreto.core as ACORE
    AX.install()
    AX.install()                                   # idempotent
    assert ACORE.EarlyStopMonitor is AX.SameOOBEarlyStopMonitor
    assert ACORE._r2_original_EarlyStopMonitor is not AX.SameOOBEarlyStopMonitor
    X, ys = _problems()
    names = [f"g{k}" for k in range(X.shape[1])]
    links, meta = ACORE.infer_partial_network("GBM", ACORE.SGBM_KWARGS, X, names, "target", ys[0],
                                              include_meta=True, seed=SEED)
    reg = GradientBoostingRegressor(random_state=SEED, **ACORE.SGBM_KWARGS).fit(
        X, ys[0], monitor=AX.SameOOBEarlyStopMonitor(25))
    assert int(meta.n_estimators.iloc[0]) == len(reg.estimators_)
    assert abs(links.importance.sum() - len(reg.estimators_)) < 1e-8


def test_dask_workers_use_compat_monitor():
    """create_graph on a 2-worker LocalCluster: every worker is patched and the
    per-target rounds equal serial compat fits."""
    from distributed import Client, LocalCluster
    import arboreto.core as ACORE
    AX.install()
    os.environ["PYTHONPATH"] = os.pathsep.join([str(SRC)] + [p for p in os.environ.get("PYTHONPATH", "").split(os.pathsep) if p])
    X, ys = _problems()
    M = np.column_stack([X] + ys[:3])
    genes = [f"g{k}" for k in range(X.shape[1])] + ["t0", "t1", "t2"]
    with LocalCluster(n_workers=2, threads_per_worker=1, processes=True, dashboard_address=None) as cl, \
            Client(cl) as client:
        st = AX.install_on_client(client)
        assert len(st) == 2
        lg, mg = ACORE.create_graph(M, genes, genes[:X.shape[1]], client=client, regressor_type="GBM",
                                    regressor_kwargs=ACORE.SGBM_KWARGS, target_genes=["t0", "t1", "t2"],
                                    include_meta=True, seed=SEED)
        links, meta = client.compute([lg, mg], sync=True)
        AX.worker_status(client)
    got = meta.set_index("target").n_estimators.to_dict()
    for k, g in enumerate(["t0", "t1", "t2"]):
        reg = GradientBoostingRegressor(random_state=SEED, **ACORE.SGBM_KWARGS).fit(
            X, ys[k], monitor=AX.SameOOBEarlyStopMonitor(25))
        assert got[g] == len(reg.estimators_) > 150


@pytest.mark.skipif(not Path(REF_PY).exists(), reason="no scikit-learn <= 1.2 interpreter")
def test_reference_sklearn122_compat_equals_arboreto_original(tmp_path):
    """Under scikit-learn <= 1.2 the compat monitor must reproduce sklearn's own
    oob_improvement_ and arboreto's original stopping round exactly."""
    X, ys = _problems()
    inp, out = tmp_path / "in.npz", tmp_path / "out.json"
    np.savez(inp, seed=SEED, **{f"X_{k}": X for k in range(len(ys))}, **{f"y_{k}": y for k, y in enumerate(ys)})
    subprocess.run([REF_PY, str(SRC / "r2" / "es_reference_replay.py"), str(inp), str(out)], check=True)
    ref = json.loads(out.read_text())
    assert ref["sklearn"] < "1.3"
    for pr in ref["problems"]:
        assert pr["compat"]["rounds"] == pr["original"]["rounds"]
        assert pr["compat_vs_sklearn_oob_improvement_max_abs_diff"] < 1e-12
    # versus this environment: the boosting paths agree to float rounding until a
    # near-tie split flips, so stopping rounds agree for most but not all problems
    here = []
    for y in ys:
        reg = GradientBoostingRegressor(random_state=SEED, **SGBM).fit(X, y, monitor=AX.SameOOBEarlyStopMonitor(25))
        here.append(len(reg.estimators_))
    there = [pr["original"]["rounds"] for pr in ref["problems"]]
    rel = np.abs(np.array(here) - np.array(there)) / np.array(there)
    assert (np.array(here) == np.array(there)).sum() >= len(ys) - 1 and rel.max() < 0.25
