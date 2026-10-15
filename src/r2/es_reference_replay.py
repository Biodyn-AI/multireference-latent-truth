"""Reference replay of arboreto's GRNBoost2 regressions under scikit-learn <= 1.2.

Runs in a *separate* Python environment with scikit-learn 1.2.x (numpy + sklearn
only; arboreto need not be installed).  For each problem in the input .npz it
fits ``GradientBoostingRegressor(random_state=seed, **SGBM_KWARGS)`` with

* ``original``: arboreto 0.1.6's ``EarlyStopMonitor`` (copied verbatim below),
  i.e. the intended GRNBoost2 behaviour, and
* ``compat``:   ``r2.arboreto_compat.SameOOBEarlyStopMonitor``, checking that its
  improvements equal scikit-learn <= 1.2's own ``oob_improvement_`` (which is
  the same-OOB-subset quantity in these versions).

Called by ``build_grnboost2.py validate-es`` and tests/test_arboreto_compat.py:
    <py-sklearn-1.2> es_reference_replay.py IN.npz OUT.json
IN.npz holds X_<k>, y_<k> (k = 0..n-1) and seed; OUT.json gets, per problem,
rounds and importances (feature_importances_ * rounds) of both monitors.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import sklearn
from sklearn.ensemble import GradientBoostingRegressor

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from r2.arboreto_compat import SameOOBEarlyStopMonitor  # noqa: E402

SGBM_KWARGS = {"learning_rate": 0.01, "n_estimators": 5000, "max_features": 0.1, "subsample": 0.9}


class EarlyStopMonitor:  # verbatim from arboreto 0.1.6 (arboreto/core.py)

    def __init__(self, window_length=25):
        self.window_length = window_length

    def window_boundaries(self, current_round):
        lo = max(0, current_round - self.window_length + 1)
        hi = current_round + 1
        return lo, hi

    def __call__(self, current_round, regressor, _):
        if current_round >= self.window_length - 1:
            lo, hi = self.window_boundaries(current_round)
            return np.mean(regressor.oob_improvement_[lo: hi]) < 0
        else:
            return False


def main(inp: str, out: str) -> None:
    d = np.load(inp)
    seed = int(d["seed"])
    n = len([k for k in d.files if k.startswith("X_")])
    res = {"sklearn": sklearn.__version__, "numpy": np.__version__, "problems": []}
    for k in range(n):
        X, y = d[f"X_{k}"], d[f"y_{k}"]
        row = {}
        for name in ("original", "compat"):
            mon = EarlyStopMonitor(25) if name == "original" else SameOOBEarlyStopMonitor(25)
            reg = GradientBoostingRegressor(random_state=seed, **SGBM_KWARGS)
            reg.fit(X, y, monitor=mon)
            r = len(reg.estimators_)
            row[name] = {"rounds": r, "importance": (reg.feature_importances_ * r).tolist()}
            if name == "compat":
                a = np.asarray(mon.oob_improvement)
                b = reg.oob_improvement_[: len(a)]
                row["compat_vs_sklearn_oob_improvement_max_abs_diff"] = float(np.abs(a - b).max())
        res["problems"].append(row)
    Path(out).write_text(json.dumps(res))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
