"""Runtime compatibility layer for arboreto 0.1.6 (GRNBoost2 / GENIE3) on the R2 environment.

Two problems, both fixed at runtime without touching site-packages:

1. **GRNBoost2 early stopping (scikit-learn >= 1.3).**  arboreto's
   ``EarlyStopMonitor`` stops boosting at round i (i >= w-1, w = 25) when the
   mean of ``regressor.oob_improvement_`` over the last w rounds is negative.
   It was written for scikit-learn <= 1.2, where ``oob_improvement_[i]`` is the
   loss decrease of round i measured on round i's out-of-bag (OOB) subset, i.e.
   the *same* subset before and after the round.  From scikit-learn 1.3 on,
   ``oob_improvement_[i] = oob_scores_[i-1] - oob_scores_[i]`` compares losses
   on two *different* random OOB subsets; the windowed mean then telescopes to
   a difference of two noisy subset losses and boosting stops after ~25 rounds
   for almost every target.  ``SameOOBEarlyStopMonitor`` recomputes the
   pre-1.3 quantity exactly,

       impr_i = L(y_oob_i, F_{i-1}(x_oob_i)) - L(y_oob_i, F_i(x_oob_i)),
       L(y, f; w) = sum(w * (y - f)^2) / sum(w)     (sklearn <= 1.2 LeastSquaresError),

   with F_{i-1} the raw predictions before round i (a copy kept from the
   previous call; for i = 0 the init estimator's predictions) and applies
   arboreto's unchanged window rule.  It reads only ``locals()`` of
   ``BaseGradientBoosting._fit_stages`` (``y``, ``X``, ``sample_mask``,
   ``sample_weight``, ``raw_predictions``) and never consumes random numbers,
   so the fitted trees are those of an uninterrupted fit truncated at the
   pre-1.3 stopping round.  On every round it cross-checks its post-round OOB
   loss against scikit-learn's own ``oob_scores_[i]`` (and, at i = 0, its
   improvement against ``oob_improvement_[0]``, which is same-subset in every
   version); a mismatch raises, which makes arboreto drop the target and the
   caller's completeness check fail loudly.

2. **dask >= 2025 (dask-expr).**  arboreto's ``create_graph`` calls
   ``from_delayed`` on the (empty) list of meta frames when
   ``include_meta=False``, which now raises; the shim returns ``None`` for an
   empty list (the meta frame is unused in that case).

Install with ``install()`` in the main process and ``install_on_client(client)``
for every dask worker (a ``WorkerPlugin``, so restarted workers are patched
too); ``worker_status(client)`` verifies the patch on each worker.
"""
from __future__ import annotations

import os

import numpy as np

EARLY_STOP_WINDOW_LENGTH = 25            # arboreto.core.EARLY_STOP_WINDOW_LENGTH
# scikit-learn versions on which the monitor has been validated (unit tests in
# tests/test_arboreto_compat.py and the cross-version replay
# ``build_grnboost2.py validate-es`` against scikit-learn 1.2.2)
SKLEARN_VALIDATED = ("1.5.1",)
_REL_TOL = 1e-9                          # tolerance of the oob_scores_ cross-check
PLUGIN_NAME = "r2-arboreto-compat"


def _ls_loss(y: np.ndarray, raw: np.ndarray, w: np.ndarray) -> float:
    """scikit-learn <= 1.2 ``LeastSquaresError.__call__`` with sample weights."""
    return float(1 / w.sum() * np.sum(w * ((y - raw.ravel()) ** 2)))


class SameOOBEarlyStopMonitor:
    """arboreto's early-stopping rule on the pre-1.3 (same-OOB-subset) improvement."""

    def __init__(self, window_length: int = EARLY_STOP_WINDOW_LENGTH):
        self.window_length = window_length
        self.oob_improvement: list[float] = []
        self._prev = None

    def window_boundaries(self, current_round: int) -> tuple[int, int]:
        lo = max(0, current_round - self.window_length + 1)
        hi = current_round + 1
        return lo, hi

    def __call__(self, current_round: int, regressor, local_vars: dict) -> bool:
        i = current_round
        if i != len(self.oob_improvement):
            raise RuntimeError(f"monitor saw round {i} after {len(self.oob_improvement)} rounds "
                               "(warm start is not supported)")
        if regressor.subsample >= 1.0:
            raise RuntimeError("OOB early stopping requires subsample < 1")
        y = local_vars["y"]
        mask = local_vars["sample_mask"]
        sw = local_vars["sample_weight"]
        after = local_vars["raw_predictions"]
        before = regressor._raw_predict_init(local_vars["X"]) if i == 0 else self._prev
        oob = ~mask
        w = sw[oob]
        loss_after = _ls_loss(y[oob], after[oob], w)
        impr = _ls_loss(y[oob], before[oob], w) - loss_after
        # cross-checks against scikit-learn's own bookkeeping
        scores = getattr(regressor, "oob_scores_", None)            # sklearn >= 1.3
        if scores is not None and abs(scores[i] - loss_after) > _REL_TOL * (1.0 + abs(loss_after)):
            raise RuntimeError(f"OOB loss mismatch at round {i}: {loss_after} vs oob_scores_ {scores[i]}")
        if i == 0 and abs(regressor.oob_improvement_[0] - impr) > _REL_TOL * (1.0 + abs(impr)):
            raise RuntimeError(f"round-0 OOB improvement mismatch: {impr} vs {regressor.oob_improvement_[0]}")
        self.oob_improvement.append(impr)
        self._prev = after.copy()        # raw predictions are updated in place by sklearn
        if i >= self.window_length - 1:
            lo, hi = self.window_boundaries(i)
            return bool(np.mean(self.oob_improvement[lo:hi]) < 0)
        return False


def sklearn_version() -> str:
    import sklearn
    return sklearn.__version__


def check_sklearn(allow_unvalidated: bool = False) -> str:
    v = sklearn_version()
    if v not in SKLEARN_VALIDATED and not allow_unvalidated:
        raise RuntimeError(f"scikit-learn {v} is not in the validated set {SKLEARN_VALIDATED}; "
                           "re-run tests/test_arboreto_compat.py and build_grnboost2.py validate-es "
                           "before using it")
    return v


def install() -> None:
    """Patch ``arboreto.core`` in the current process (idempotent)."""
    import arboreto.core as AC
    if getattr(AC, "_r2_compat_installed", False):
        return
    AC._r2_original_EarlyStopMonitor = AC.EarlyStopMonitor
    AC.EarlyStopMonitor = SameOOBEarlyStopMonitor
    orig = AC.from_delayed

    def from_delayed_safe(dfs, meta=None, **kw):
        return orig(dfs, meta=meta, **kw) if len(dfs) else None

    AC.from_delayed = from_delayed_safe
    AC._r2_compat_installed = True


def _status() -> dict:
    import arboreto.core as AC
    m = AC.EarlyStopMonitor
    return {"early_stop_monitor": f"{m.__module__}.{m.__qualname__}",
            "from_delayed_shim": bool(getattr(AC, "_r2_compat_installed", False)),
            "sklearn": sklearn_version(), "pid": os.getpid()}


EXPECTED_MONITOR = f"{SameOOBEarlyStopMonitor.__module__}.{SameOOBEarlyStopMonitor.__qualname__}"


def install_on_client(client) -> dict:
    """Register a WorkerPlugin that installs the patch on every (current and future)
    worker, then verify it on each worker; returns the per-worker status."""
    from distributed.diagnostics.plugin import WorkerPlugin

    class _Plugin(WorkerPlugin):
        name = PLUGIN_NAME

        def setup(self, worker):
            install()

    client.register_plugin(_Plugin(), name=PLUGIN_NAME)
    return worker_status(client)


def worker_status(client) -> dict:
    """Per-worker patch status; raises unless every worker uses the compat monitor
    and the same scikit-learn version as the main process."""
    st = client.run(_status)
    main_v = sklearn_version()
    bad = {w: s for w, s in st.items()
           if s["early_stop_monitor"] != EXPECTED_MONITOR or not s["from_delayed_shim"]
           or s["sklearn"] != main_v}
    if not st or bad:
        raise RuntimeError(f"arboreto compat patch missing or inconsistent on workers: {bad or st}")
    return st
