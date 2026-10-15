"""Tests for the cross-cohort replication stage (src/r2/replication.py).

Run: .venv/bin/python -m pytest subproject_24_multireference_latent_truth_model/src/r2/tests/test_replication.py -q
"""
import itertools
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from r2 import replication as RP  # noqa: E402


# ----------------------------------------------------------------------------
# exact conditional test
# ----------------------------------------------------------------------------
def test_exact_pvalue_single_stratum_equals_fisher_greater():
    for (a, n, K, N) in [(5, 20, 30, 400), (0, 10, 5, 100), (7, 7, 12, 50), (3, 40, 3, 1000)]:
        table = [[a, n - a], [K - a, N - n - (K - a)]]
        ref = stats.fisher_exact(table, alternative="greater")[1]
        assert np.isclose(RP.exact_pvalue([N], [K], [n], a), ref, rtol=1e-9, atol=1e-15)


def test_exact_pvalue_multi_stratum_brute_force():
    N, K, n = [12, 9, 15], [4, 3, 6], [5, 2, 4]
    pmfs = []
    for Ns, Ks, ns in zip(N, K, n):
        xs = np.arange(0, ns + 1)
        pmfs.append({int(x): stats.hypergeom.pmf(x, Ns, Ks, ns) for x in xs})
    for t in range(0, 12):
        ref = sum(np.prod([pmfs[i][x[i]] for i in range(3)])
                  for x in itertools.product(*[list(p) for p in pmfs]) if sum(x) >= t)
        assert np.isclose(RP.exact_pvalue(N, K, n, t), ref, atol=1e-12)


def test_strata_without_promoted_or_positives_do_not_change_p():
    base = RP.exact_pvalue([100, 80], [10, 6], [8, 5], 4)
    more = RP.exact_pvalue([100, 80, 50, 70], [10, 6, 0, 9], [8, 5, 6, 0], 4)
    assert np.isclose(base, more)


def test_hyper_pmf_noncentral_matches_scipy():
    lo, p = RP.hyper_pmf(60, 14, 9, log_or=math.log(3.0))
    ref = stats.nchypergeom_fisher.pmf(np.arange(lo, lo + len(p)), 60, 14, 9, 3.0)
    assert np.allclose(p, ref, atol=1e-12)


def test_mantel_haenszel_matches_statsmodels():
    from statsmodels.stats.contingency_tables import StratifiedTable
    a, n, K, N = np.array([5, 2, 9]), np.array([20, 10, 30]), np.array([40, 12, 60]), np.array([400, 150, 700])
    tables = [np.array([[a[i], n[i] - a[i]], [K[i] - a[i], N[i] - n[i] - K[i] + a[i]]]) for i in range(3)]
    ref = StratifiedTable(tables).oddsratio_pooled
    assert np.isclose(RP.mantel_haenszel_or(a, n, K, N), ref)


# ----------------------------------------------------------------------------
# power
# ----------------------------------------------------------------------------
def test_rates_for_rr_preserve_overall_rate():
    p0, p1 = RP.rates_for_rr(0.05, 1000.0, 100.0, 2.0)
    assert np.isclose(p1, 2 * p0)
    assert np.isclose((100 * p1 + 900 * p0) / 1000, 0.05)


def test_critical_values_property():
    N, m = 500, 30
    Ks = np.array([0, 5, 20, 60, 200])
    c = RP.critical_values(N, m, Ks, 0.05)
    for K, ck in zip(Ks, c):
        sf = lambda x: stats.hypergeom.sf(x - 1, N, K, m)   # P(X >= x)  # noqa: E731
        assert sf(ck) <= 0.05 + 1e-12
        if ck - 1 >= 0 and ck - 1 <= min(m, K):
            assert sf(ck - 1) > 0.05


def test_exact_single_power_matches_simulation():
    N, m, pbar = 3000, 60, 0.05
    ex = RP.exact_power_single(N, m, pbar, 2.0)
    sim, se = RP.simulate_power([N], [m], pbar, 2.0, n_sims=3000, seed=3)
    assert abs(ex - sim) < 4 * se + 0.01
    # size under the null is <= alpha (exact test, conservative)
    assert RP.exact_power_single(N, m, pbar, 1.0) <= 0.05 + 1e-9


def test_power_increases_with_effect_and_floor_reaches_target():
    N, pbar = 20000, 0.02
    f = RP.support_floor(N, pbar)
    assert np.isfinite(f)
    assert RP.exact_power_single(N, int(f), pbar, 2.0) >= 0.8
    assert RP.exact_power_single(N, max(1, int(f) // 2), pbar, 2.0) < 0.8
    # expected null positives at the floor is in the Poisson range (rate-free for small pbar)
    assert 3 < f * pbar < 20
    p1 = RP.simulate_power([5000, 4000], [40, 30], 0.03, 1.5, n_sims=400, seed=1)[0]
    p2 = RP.simulate_power([5000, 4000], [40, 30], 0.03, 3.0, n_sims=400, seed=1)[0]
    assert p2 > p1


def test_conditional_power_size_and_monotone():
    N, n, K = [800, 600, 900], [20, 15, 30], [40, 25, 60]
    p_null, t = RP.conditional_power(N, n, K, 0.0)
    assert p_null <= 0.05 + 1e-12 and t > 0
    p_alt = RP.conditional_power(N, n, K, math.log(2.0))[0]
    p_big = RP.conditional_power(N, n, K, math.log(4.0))[0]
    assert p_null < p_alt < p_big
    mde = RP.conditional_mde_or(N, n, K)
    assert RP.conditional_power(N, n, K, math.log(mde))[0] >= 0.8 - 1e-9


def test_unit_status_classification():
    d = pd.DataFrame(dict(N=[100, 50], n=[5, 0], K=[0, 3]))
    assert RP.unit_status(d).startswith("non-evaluable: no TF with both")
    d = pd.DataFrame(dict(N=[100], n=[5], K=[0]))
    assert RP.unit_status(d) == "non-evaluable: no evaluable TF positives"
    d = pd.DataFrame(dict(N=[100], n=[5], K=[4]))
    assert RP.unit_status(d) == "evaluable"
    assert RP.unit_status(pd.DataFrame(dict(N=[], n=[], K=[]))).startswith("non-evaluable: no perturbed TF")


# ----------------------------------------------------------------------------
# promoted set, outcome helpers
# ----------------------------------------------------------------------------
def test_bfdr_set_matches_fit_primary():
    from r2 import fit_primary as FP
    rng = np.random.default_rng(0)
    c = np.round(rng.beta(0.3, 3, 500), 3)
    n = rng.integers(1, 50, 500).astype(float)
    for q in (0.05, 0.2):
        assert np.array_equal(RP.bfdr_set(c, n, q), FP.bfdr_set(c, n, q))
    t = RP.pattern_tiers(c, n)
    assert set(np.unique(t)) <= {0, 1, 2}
    # Strong is nested in Strong U Moderate and whole tie blocks are kept together
    for v in np.unique(c):
        assert len(np.unique(t[c == v])) == 1


def _toy_edges(seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for c in ("c1", "c2"):
        for tf in ("A", "B", "C"):
            for g in range(200):
                tier = rng.choice([0, 0, 0, 0, 1, 2])
                pos = rng.random() < (0.25 if tier else 0.1)
                rows.append((c, tf, f"g{g}", bool(pos), int(tier)))
    ed = pd.DataFrame(rows, columns=["cohort", "tf", "target", "positive", "tier"])
    ed["promoted"] = ed["tier"] > 0
    ed["observed"] = True
    return ed


def test_design_and_outcome_strata_agree_and_outcome_is_consistent():
    ed = _toy_edges()
    d1 = RP.stratum_design(ed)
    d2 = RP.outcome_strata(ed)
    for col in ("N", "n", "K", "n_strong"):
        assert np.array_equal(d1[col].to_numpy(), d2[col].to_numpy())
    assert np.array_equal(d2["a"], d2["a_strong"] + d2["a_moderate"])
    r = RP.unit_outcome(d2, seed=1)
    a, n, K, N = (d2[c].to_numpy(float) for c in ("a", "n", "K", "N"))
    assert np.isclose(r["observed_promoted_positives"], a.sum())
    assert np.isclose(r["expected_promoted_positives"], np.sum(n * K / N))
    # 3 TFs: the TF-cluster bootstrap reports no interval below MIN_BOOT_CLUSTERS TFs
    assert r["n_bootstrap_clusters"] == 3 and np.isnan(r["enrichment_boot_lo"])
    assert abs(r["p_perm"] - r["p_exact"]) < 0.05 or (r["p_exact"] < 0.001 and r["p_perm"] < 0.01)
    assert r["endpoint_pass"] == ((r["enrichment"] >= RP.ENDPOINT) and (r["p_exact"] <= RP.ALPHA))


def test_wilson_and_holm_match_statsmodels():
    from statsmodels.stats.multitest import multipletests
    from statsmodels.stats.proportion import proportion_confint
    lo, hi = RP.wilson(3, 7)
    ref = proportion_confint(3, 7, method="wilson")
    assert np.allclose((lo, hi), ref)
    p = np.array([0.01, 0.04, 0.03, 0.2, np.nan])
    ref = multipletests(p[:4], method="holm")[1]
    assert np.allclose(RP.holm(p)[:4], ref) and np.isnan(RP.holm(p)[4])


def test_data_derived_threshold_rule():
    support = np.array([5, 50, 200, 800])
    passed = np.array([False, True, False, True])
    tau, nk, npass = RP.data_derived_threshold(support, passed)
    # tau=5: 2/4, tau=50: 2/3, tau=200: 1/2, tau=800: 1 unit only (< min_units) -> tau = 50
    assert (tau, nk, npass) == (50.0, 3, 2)
    tau, nk, npass = RP.data_derived_threshold(np.array([5, 50]), np.array([False, True]))
    assert (tau, nk, npass) == (5.0, 2, 1)


def test_re_data_counts():
    strata = pd.DataFrame(dict(cohort=["c1", "c1", "c2", "c3"], tf=["A", "B", "A", "A"], N=[100, 80, 50, 60],
                               n=[10, 5, 4, 3], K=[20, 0, 6, 9], a=[5, 0, 2, 1]))
    units = {"u1": ["c1", "c2"], "u2": ["c3"]}
    d = RP.re_data(strata, units, ["u1", "u2"])
    assert d["U"] == 2 and len(d["u"]) == 3                      # zero-positive stratum dropped
    assert d["y1"].tolist() == [5, 2, 1] and d["n0"].tolist() == [90, 46, 57]
    assert d["y0"].tolist() == [15, 4, 8]


def test_re_model_recovers_strong_effect(monkeypatch):
    monkeypatch.setattr(RP, "NUTS_SETTINGS", dict(num_chains=1, num_warmup=300, num_samples=300, target_accept=0.9))
    rng = np.random.default_rng(5)
    rows = []
    for c in ("c1", "c2"):
        for s in range(8):
            N, n = 3000, 200
            a0 = rng.normal(-3, 0.3)
            p0 = 1 / (1 + np.exp(-a0))
            p1 = 1 / (1 + np.exp(-(a0 + math.log(3.0))))
            y1 = rng.binomial(n, p1)
            y0 = rng.binomial(N - n, p0)
            rows.append((c, f"t{s}", N, n, y1 + y0, y1))
    strata = pd.DataFrame(rows, columns=["cohort", "tf", "N", "n", "K", "a"])
    data = RP.re_data(strata, {"c1": ["c1"], "c2": ["c2"]}, ["c1", "c2"])
    fit = RP.fit_re(data, "primary", seed=1)
    th = fit["draws"]["theta"].mean(0)
    assert np.all(np.abs(th - math.log(3.0)) < 0.35)
    rows = RP.fit_rows(fit)
    assert rows[0]["parameter"] == "mu" and rows[-1]["parameter"] == "theta"


# ----------------------------------------------------------------------------
# key joins against the stored evidence (integration; skipped without data)
# ----------------------------------------------------------------------------
@pytest.mark.skipif(not (RP.E.EV_DIR / "global" / "edge_pattern.npy").exists(), reason="evidence not built")
def test_global_and_tissue_joins_give_identical_evidence_rows():
    ev = pd.read_parquet(RP.E.EV_DIR / "immune" / "evidence.parquet", columns=["tf", "target"])
    sub = ev.sample(300, random_state=0).reset_index(drop=True)
    de = sub.assign(positive=False)
    g = RP.attach_patterns("global", de)
    t = RP.attach_patterns("immune", de)
    m = g.merge(t, on=["tf", "target"], suffixes=("_g", "_t"))
    assert len(m) > 250
    Rg = RP.E.load_patterns("global")[0]
    Rt = RP.E.load_patterns("immune")[0]
    assert np.array_equal(Rg[m["pattern_g"].to_numpy()], Rt[m["pattern_t"].to_numpy()])


def test_bfdr_tiers_match_governance_levels():
    pytest.importorskip("r2.governance")
    from r2 import governance as G
    rng = np.random.default_rng(2)
    c = rng.beta(0.3, 3, 2000)
    c[:50] = c[50:100]                                   # exact ties
    c[100:110] = c[110:120] + 1e-14                      # near-ties within the rounding tolerance
    n = rng.integers(1, 30, 2000).astype(float)
    assert np.array_equal(RP.pattern_tiers(c, n), G.bfdr_levels(c, n, RP.Q_STRONG, RP.Q_MODERATE))


def test_cluster_bootstrap_degenerate_with_one_tf():
    d = pd.DataFrame(dict(N=[100, 50], n=[10, 0], K=[20, 5], a=[5, 0]))
    b = RP.cluster_bootstrap(d, 200, 1)
    assert np.isnan(b["oe_lo"]) and b["n_clusters"] == 1


def test_wilson_boundaries_exact():
    lo, hi = RP.wilson(0, 3)
    assert lo == 0.0 and 0 < hi < 1
    lo, hi = RP.wilson(4, 4)
    assert hi == 1.0 and 0 < lo < 1


def test_tf_table_and_lock(tmp_path, monkeypatch):
    ed = _toy_edges(3)
    s = RP.outcome_strata(ed).assign(universe="global", comparator="universe")
    monkeypatch.setattr(RP, "cohort_table", lambda: pd.DataFrame(dict(
        cohort_id=["c1", "c2"], study_id=["s1", "s1"], cell_type_group=["K562", "K562"], modality=["x", "x"])))
    t = RP.tf_table(s)
    assert len(t) == 6 and np.allclose(t["expected"], t["n"] * t["K"] / t["N"])
    assert np.allclose(t.groupby("cohort")["share_of_unit_promoted_positives"].sum(), 1.0)
    # the outcome stage refuses to run without, or against a different, locked design
    design = RP.stratum_design(ed).assign(universe="global", comparator="universe")
    monkeypatch.setattr(RP, "LOCK", tmp_path / "lock.json")
    with pytest.raises(SystemExit):
        RP.check_lock(design)
    (tmp_path / "rep_power.csv").write_text("x\n")
    monkeypatch.setattr(RP.C, "TABLES", tmp_path)
    RP.C.write_json(dict(design_digest=RP.design_digest(design), rep_power_sha256=RP._sha256(tmp_path / "rep_power.csv")),
                    tmp_path / "lock.json")
    assert RP.check_lock(design)["design_digest"] == RP.design_digest(design)
    changed = design.copy()
    changed.loc[0, "n"] += 1
    with pytest.raises(SystemExit):
        RP.check_lock(changed)


# ----------------------------------------------------------------------------
# revision 2026-10-08 (verifier rep): post hoc controls, attainable maximum, floors
# ----------------------------------------------------------------------------
def test_cluster_bootstrap_resamples_tfs_and_needs_five():
    d = pd.DataFrame(dict(cohort=["c1", "c2"] * 6, tf=np.repeat(list("ABCDEF"), 2), N=100, n=10, K=20,
                          a=np.arange(12) % 5))
    b = RP.cluster_bootstrap(d, 300, 1)
    assert b["n_clusters"] == 6 and b["oe_lo"] <= RP.oe(d["a"], d["n"], d["K"], d["N"])[0] / \
        RP.oe(d["a"], d["n"], d["K"], d["N"])[1] <= b["oe_hi"]
    b4 = RP.cluster_bootstrap(d[d["tf"].isin(list("ABCD"))], 300, 1)
    assert b4["n_clusters"] == 4 and np.isnan(b4["oe_lo"])


def test_unit_outcome_attainable_maximum_and_status():
    # one TF with 70% positives and every promoted edge positive: O/E at its ceiling 1/0.7 < 1.5
    d = pd.DataFrame(dict(cohort="c", tf="GATA1", N=[300], n=[11], K=[210], a=[11], n_strong=[11], a_strong=[11],
                          n_moderate=[0], a_moderate=[0]))
    r = RP.unit_outcome(d, seed=1)
    assert np.isclose(r["max_attainable_enrichment"], 11 / (11 * 210 / 300))
    assert np.isclose(r["enrichment"], r["max_attainable_enrichment"])
    assert r["endpoint_attainable"] is False and r["endpoint_status"] == "endpoint_unattainable"


def test_wrong_tf_sums_brute_force():
    ec = pd.DataFrame(dict(tf=["A"] * 4 + ["B"] * 4, target=list("wxyz") * 2,
                           promoted=[True, True, False, False, False, True, False, False],
                           positive=[True, False, False, True, False, True, True, False]))
    tfde = ec.rename(columns={}).assign()[["tf", "target", "positive"]]
    w = RP.wrong_tf_sums(ec, tfde)
    # A's promoted targets w, x under B: B positives on (w, x, y, z) = (F, T, T, F); N=4, n=2, K=2, a=1
    # B's promoted target x under A: A positives (T, F, F, T); N=4, n=1, K=2, a=0
    assert np.isclose(w["O"], 1.0) and np.isclose(w["E"], 2 * 2 / 4 + 1 * 2 / 4)
    # excluding targets promoted for the other TF: under B drop x (promoted for B): N=3, n=1, K=1, a=0;
    # under A drop w, x (promoted for A): N=2, n=0 -> 0
    assert np.isclose(w["O_excl"], 0.0) and np.isclose(w["E_excl"], 1 * 1 / 3)


def test_responsiveness_leave_own_out_and_tied_bins():
    tot = pd.DataFrame({"sum": [5, 0, 2], "count": [11, 11, 3]}, index=["g1", "g2", "g3"])
    ec = pd.DataFrame(dict(target=["g1", "g2", "g3"], positive=[True, False, False]))
    r = RP.responsiveness(ec, tot)
    assert np.allclose(r, [4 / 10, 0.0, 2 / 2])
    resp = np.r_[np.zeros(60), np.linspace(0.1, 1, 40)]
    b = RP.responsiveness_bins(resp)
    assert len(set(b[:60])) == 1 and b[60:].max() > b[0]


def test_floor_scan_and_endpoint_power():
    N, pbar = 2000, 0.05
    f = RP.support_floor(N, pbar)
    m = RP.support_floor_min_scan(N, pbar, f)
    assert m <= f and RP.exact_power_single(N, int(m), pbar, RP.RR_TRUE) >= RP.TARGET_POWER
    assert all(RP.exact_power_single(N, k, pbar, RP.RR_TRUE) < RP.TARGET_POWER for k in range(1, int(m)))
    Ns, ns = np.array([500, 800, 700]), np.array([30, 40, 25])
    pw, _ = RP.simulate_power(Ns, ns, 0.1, 2.0, n_sims=300, seed=4)
    pe, _ = RP.simulate_endpoint_power(Ns, ns, 0.1, 2.0, n_sims=300, seed=4)
    assert pe <= pw + 1e-12


def test_re_model_tf_slope_profile_runs(monkeypatch):
    monkeypatch.setattr(RP, "NUTS_SETTINGS", dict(num_chains=2, num_warmup=150, num_samples=150, target_accept=0.9))
    rng = np.random.default_rng(6)
    rows = []
    for c in ("c1", "c2"):
        for s in range(5):
            N, n = 2000, 150
            p0 = 0.05
            y1 = rng.binomial(n, 0.12)
            y0 = rng.binomial(N - n, p0)
            rows.append((c, f"t{s}", N, n, y1 + y0, y1))
    strata = pd.DataFrame(rows, columns=["cohort", "tf", "N", "n", "K", "a"])
    data = RP.re_data(strata, {"c1": ["c1"], "c2": ["c2"]}, ["c1", "c2"])
    fit = RP.fit_re(data, RP.TF_SLOPE_PROFILE, seed=2)
    rows = RP.fit_rows(fit)
    assert "tau_tf" in fit["draws"] and any(r["parameter"] == "tau_tf" for r in rows)
    th = [r for r in rows if r["parameter"] == "theta"]
    assert len(th) == 2 and all(np.isfinite(r["r_hat"]) for r in th)


def test_lock_check_records_fit_changes_without_design_change(monkeypatch):
    """A fit re-run that moves tier sizes but no evaluable stratum keeps the lock valid; the record
    says that the promoted set changed and that the locked design is reproduced."""
    ed = _toy_edges(3)
    design = RP.stratum_design(ed).assign(universe="global", comparator="universe")
    lock = dict(written_at_utc="2026-10-08T16:40:31+00:00", design_digest=RP.design_digest(design),
                promoted_sets={"global": dict(n_strong=10.0, n_moderate=5.0,
                                              fit_meta=dict(n_divergent=5, r_hat_max=1.03))})
    now = dict(n_strong=9.0, n_moderate=6.0, fit_meta=dict(n_divergent=0, r_hat_max=1.01))
    monkeypatch.setattr(RP, "load_universe_tiers", lambda u: dict(now))
    r = RP.lock_check(lock, design, universes=("global",))
    assert r["design_matches_lock"] and r["universes"]["global"]["promoted_set_changed_since_lock"]
    assert r["universes"]["global"]["at_lock"]["n_strong"] == 10.0 and r["universes"]["global"]["now"]["n_strong"] == 9.0
    monkeypatch.setattr(RP, "load_universe_tiers", lambda u: dict(lock["promoted_sets"]["global"]))
    assert not RP.lock_check(lock, design, universes=("global",))["universes"]["global"]["promoted_set_changed_since_lock"]
    changed = design.copy()
    changed.loc[0, "n"] += 1
    assert not RP.lock_check(lock, changed, universes=("global",))["design_matches_lock"]


def test_all_keeps_an_existing_lock(tmp_path, monkeypatch):
    """`replication.py all` must not replace an existing power lock (re-locking needs an explicit reason)."""
    called = []
    monkeypatch.setattr(RP, "OUT_DIR", tmp_path)
    monkeypatch.setattr(RP, "LOCK", tmp_path / "power_lock.json")
    for name in ("run_power", "run_outcomes", "run_threshold", "run_bayes", "run_leavek"):
        monkeypatch.setattr(RP, name, lambda *a, _n=name, **k: called.append(_n))
    monkeypatch.setattr(RP, "record_decisions", lambda *a, **k: None)
    RP.main(["all"])
    assert called[0] == "run_power"                      # no lock yet: STEP 1 runs first
    called.clear()
    (tmp_path / "power_lock.json").write_text('{"written_at_utc": "x"}')
    RP.main(["all"])
    assert "run_power" not in called and called == ["run_outcomes", "run_threshold", "run_bayes", "run_leavek"]
    called.clear()
    RP.main(["power", "--relock-reason", "promoted set changed"])
    assert called == ["run_power"]
