"""Tests for the held-out evaluation module (src/r2/heldout_eval.py)."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from r2 import heldout_eval as H  # noqa: E402


def _toy(seed=0, n=4000, T=6, decimals=1):
    rng = np.random.default_rng(seed)
    tf = rng.integers(0, T, n)
    y = rng.random(n) < 0.08 + 0.04 * tf
    s = np.round(rng.normal(size=n) + 0.7 * y, decimals)
    s[rng.random(n) < 0.05] = np.nan
    return tf, y, s, T


def test_score_levels_nan_and_ties_last():
    lev, vals = H.score_levels(np.array([0.5, np.nan, 2.0, 0.5, -np.inf, np.inf]))
    assert list(lev) == [2, 3, 1, 2, 3, 0]
    assert vals[0] == np.inf and vals[-1] == -np.inf


def test_ap_matches_sklearn_with_ties():
    from sklearn.metrics import average_precision_score
    tf, y, s, T = _toy()
    lc = H.level_counts(tf, y, np.full(len(y), 0.1), s, T)
    m = H.curve_metrics(*lc.totals())
    s2 = np.where(np.isnan(s), -1e9, s)
    assert np.isclose(m["ap"][0], average_precision_score(y, s2))
    assert np.isclose(m["base_rate"][0], y.mean())


def test_p_at_k_is_expectation_over_random_tie_breaks():
    tf, y, s, T = _toy(decimals=0)            # heavy ties
    lc = H.level_counts(tf, y, np.zeros(len(y)), s, T)
    m = H.curve_metrics(*lc.totals(), ks=(100, 333))
    v = np.where(np.isnan(s), -np.inf, s)
    for k in (100, 333):
        thr = np.sort(v)[::-1][k - 1]
        above = v > thr
        tie = v == thr
        expected = (y[above].sum() + (k - above.sum()) * y[tie].mean()) / k
        assert np.isclose(m[f"p_at_{k}"][0], expected)


def test_strat_enrichment_uses_null_probabilities():
    tf, y, s, T = _toy()
    null = np.full(len(y), 0.05)
    lc = H.level_counts(tf, y, null, s, T)
    m = H.curve_metrics(*lc.totals(), ks=(200,))
    assert np.isclose(m["exp_null_at_200"][0], 200 * 0.05)
    assert np.isclose(m["strat_enr_at_200"][0], m["p_at_200"][0] * 200 / 10.0)


def test_per_tf_auroc_matches_sklearn():
    from sklearn.metrics import roc_auc_score
    tf, y, s, T = _toy()
    lc = H.level_counts(tf, y, np.zeros(len(y)), s, T)
    au = H.per_tf_auroc(lc)
    s2 = np.where(np.isnan(s), -1e9, s)
    for t in range(T):
        assert np.isclose(au[t], roc_auc_score(y[tf == t], s2[tf == t]))


def test_bootstrap_with_unit_weights_reproduces_point_estimate():
    tf, y, s, T = _toy()
    lc = H.level_counts(tf, y, np.full(len(y), 0.1), s, T)
    pt = H.curve_metrics(*lc.totals())
    bm = H.boot_metrics(lc, np.ones((3, T)))
    for k in ("ap", "ap_lift", "enr_at_100", "strat_enr_at_500", "best_f1"):
        assert np.allclose(bm[k], pt[k][0], equal_nan=True)


def test_within_tf_permutation_keeps_tf_positive_counts_and_is_null():
    tf, y, s, T = _toy(n=3000)
    lc = H.level_counts(tf, y, np.zeros(len(y)), s, T)
    ap_null = H.permuted_ap(lc, 300, np.random.default_rng(1))
    # under H0 the score is unrelated to the label: the observed AP (signal 0.7 sd) is extreme
    ap = H.curve_metrics(*lc.totals())["ap"][0]
    assert ap > np.quantile(ap_null, 0.99)
    # a random score is not extreme
    s_rand = np.random.default_rng(2).random(len(y))
    lcr = H.level_counts(tf, y, np.zeros(len(y)), s_rand, T)
    ap_r = H.curve_metrics(*lcr.totals())["ap"][0]
    null_r = H.permuted_ap(lcr, 300, np.random.default_rng(3))
    assert np.quantile(null_r, 0.005) < ap_r < np.quantile(null_r, 0.995)


def test_holm_and_bootstrap_p():
    p = np.array([0.01, 0.04, 0.03, np.nan, 0.5])
    assert np.allclose(H.holm(p), [0.04, 0.09, 0.09, np.nan, 0.5], equal_nan=True)
    assert H.bootstrap_p_two_sided(np.ones(999)) == pytest.approx(2 / 1000)
    assert H.bootstrap_p_two_sided(np.r_[np.ones(500), -np.ones(500)]) == 1.0


def test_ragged_null_reproduces_rectangular_target_degree_model():
    from r2.build_heldout_chip import target_degree_model
    rng = np.random.default_rng(5)
    n_tf, n_g = 12, 400
    hub = rng.random(n_g) ** 3
    M = rng.random((n_tf, n_g)) < (0.02 + 0.5 * hub)[None, :] * rng.uniform(0.3, 1, n_tf)[:, None]
    self_col = np.full(n_tf, -1)
    ref = target_degree_model(M, self_col)
    tt, gg = np.meshgrid(np.arange(n_tf), np.arange(n_g), indexing="ij")
    p, diag = H.ragged_target_degree_null(tt.ravel(), gg.ravel(), M.ravel(), n_tf, n_g)
    deg = M.sum(0)
    b = ref["to_bin"](deg[gg.ravel()] - M.ravel())
    assert diag["converged"]
    assert np.allclose(p, ref["P"][tt.ravel(), b], atol=1e-6)


def test_downsample_null_exact_counts():
    rng = np.random.default_rng(0)
    tot = np.array([50, 200, 1000, 3000])
    pos = np.array([30, 60, 100, 60])
    nm = H.downsample_null(tot, pos, 400, 40, 50, rng)
    assert np.allclose(nm["n"], 400) and np.allclose(nm["n_pos"], 40)
    nu = H.downsample_null(tot, pos, 400, None, 50, rng)
    assert np.allclose(nu["n"], 400)
    assert abs(nu["base_rate"].mean() - pos.sum() / tot.sum()) < 0.02


def test_topk_weights_fractional_ties():
    s = np.array([5, 4, 4, 4, 3, 3, 1.0])
    w = H.topk_weights(s, 3)
    assert np.isclose(w.sum(), 3)
    assert w[0] == 1 and np.allclose(w[1:4], 2 / 3) and np.all(w[4:] == 0)


def test_predicted_set_perm_and_tf_matched_expectation():
    """Within-TF permutation keeps the TF-matched expectation: a predicted set that is random
    within each TF has enrichment ~1 and a large permutation p."""
    rng = np.random.default_rng(3)
    n, T = 6000, 5
    tf = np.repeat(np.arange(T), n // T).astype(object).astype(str)
    y = rng.random(n) < np.repeat(np.linspace(0.05, 0.4, T), n // T)
    es = H.EvalSet("u", "r", np.arange(n), tf, np.arange(n).astype(str), y, np.full(n, 0.2))
    mask = rng.random(n) < 0.1
    rows = H.predicted_set_rows(es, {("rand", 50): mask}, None)
    r = rows[0]
    assert abs(r["enr_tf_matched"] - 1) < 0.15
    assert r["perm_p"] > 0.01


@pytest.mark.skipif(not (H.EV_DIR / "global" / "tfs.txt").exists(), reason="evidence stage not built")
def test_global_universe_edge_ids_are_tf_major():
    U = H.load_universe("global")
    G = len(U.genes)
    e = U.edge_ids([U.tfs[3], U.tfs[10], "NOT_A_GENE"], [U.genes[7], U.genes[0], U.genes[0]])
    assert list(e) == [3 * G + 7, 10 * G, -1]


@pytest.mark.skipif(not (H.EV_DIR / "immune" / "evidence.parquet").exists(), reason="evidence stage not built")
def test_tissue_universe_edge_ids_join_on_keys():
    import pandas as pd
    U = H.load_universe("immune")
    ev = pd.read_parquet(H.EV_DIR / "immune" / "evidence.parquet", columns=["tf", "target"])
    idx = np.random.default_rng(0).choice(len(ev), 50, replace=False)
    assert np.array_equal(U.edge_ids(ev["tf"].to_numpy()[idx], ev["target"].to_numpy()[idx]), idx)


def test_degree_quintiles_ties_share_group():
    deg = np.array([1, 1, 1, 1, 2, 3, 4, 5, 6, 100])
    q = H.degree_quintiles(deg)
    assert len(set(q[:4])) == 1
    assert q.min() >= 1 and q.max() == 5 and np.all(np.diff(q[np.argsort(deg, kind="stable")]) >= 0)
    assert np.array_equal(H.degree_quintiles(np.arange(10)), np.repeat(np.arange(1, 6), 2))


def test_matched_null_reproduces_prevalence_driven_best_f1():
    """Best F1 of an uninformative ranking is the predict-all F1 2b/(1+b), so a prevalence-matched
    subset reproduces it whatever the subset size."""
    rng = np.random.default_rng(1)
    tot = np.full(50, 200)
    pos = rng.binomial(200, 0.1, size=50)
    nm = H.downsample_null(tot, pos, 1000, 250, 200, rng)
    b = 0.25
    assert np.all(nm["best_f1"] >= 2 * b / (1 + b) - 1e-12)
    assert abs(np.median(nm["best_f1"]) - 2 * b / (1 + b)) < 0.02


# ----------------------------------------------------------------------------------------
# revision 2026-10-08 (verifier held M1-M5): confirmatory resampling, nulls, coverage
# ----------------------------------------------------------------------------------------
def test_boot_light_equals_boot_metrics():
    tf, y, s, T = _toy(seed=4)
    lc = H.level_counts(tf, y, np.full(len(y), 0.1), s, T)
    M = H.tf_bootstrap_weights(T, 40, np.random.default_rng(1))
    a = H.boot_metrics(lc, M, ks=(500,))
    b = H.boot_light(lc, M, 500, chunk=7)
    assert np.allclose(a["ap_lift"], b["ap_lift"], equal_nan=True)
    assert np.allclose(a["enr_at_500"], b["enr_at_500"], equal_nan=True)


def test_bootstrap_z_and_bh():
    from statsmodels.stats.multitest import multipletests
    rng = np.random.default_rng(0)
    d = rng.normal(0.3, 0.1, 5000)
    assert H.bootstrap_p_z(d, 0.3) == pytest.approx(2 * __import__("scipy").stats.norm.sf(3.0), rel=0.05)
    assert H.bootstrap_p_z(np.zeros(10), 0.0) == 1.0
    p = np.array([0.01, 0.04, 0.03, np.nan, 0.5, 0.002])
    ok = np.isfinite(p)
    assert np.allclose(H.bh(p)[ok], multipletests(p[ok], method="fdr_bh")[1])


def _fake_reps(names, B=2000, seed=0, groups=None):
    rng = np.random.default_rng(seed)
    reps = {}
    for i, n in enumerate(names):
        r = {m: rng.normal(1 + 0.1 * i, 0.02, B) for m in H.PAIRED_METRICS}
        r["_point"] = {m: 1 + 0.1 * i for m in H.PAIRED_METRICS}
        r["_group"] = (groups or {}).get(n, "lcm")
        r["_n_tie_blocks"] = 5
        r["_10k"] = {m: rng.normal(1 + 0.1 * i, 0.02, 10_000) for m in H.PAIRED_METRICS}
        reps[n] = r
    return reps


def test_paired_rows_confirmatory_family_and_floors():
    names = [H.PRIMARY] + list(H.CONFIRMATORY_COMPARATORS) + ["union", H.STATUS_SCORE]
    reps = _fake_reps(names, groups={H.STATUS_SCORE: "post_hoc_mechanism"})
    reps[names[0]]["_10k"] = reps[names[1]]["_10k"]        # no difference vs EM in the 10k replicates
    es = H.EvalSet("u", "r", np.arange(30), np.array(["a"] * 15 + ["b"] * 15, dtype=object),
                   np.arange(30).astype(str).astype(object), np.r_[np.ones(25, bool), np.zeros(5, bool)], np.zeros(30))
    rows = H.paired_rows(es, reps)
    ap = [r for r in rows if r["metric"] == "ap_lift"]
    conf = [r for r in ap if r["analysis_role"] == "confirmatory"]
    assert {r["score_b"] for r in conf} == set(H.CONFIRMATORY_COMPARATORS)
    assert all(np.isclose(r["p_holm_confirmatory_floor"], 6 * 2 / 10_001) for r in conf)
    # the descriptive Holm family excludes the post hoc score; floors are reported
    ph = [r for r in ap if r["score_b"] == H.STATUS_SCORE][0]
    assert ph["analysis_status"] == "post_hoc" and np.isnan(ph["p_holm"])
    fam = [r for r in ap if np.isfinite(r["p_holm"])]
    assert len(fam) == len(H.CONFIRMATORY_COMPARATORS) + 1
    assert all(np.isclose(r["p_holm_floor"], min(1, len(fam) * r["p_floor"])) for r in fam)
    # secondary metrics carry the 6-comparator Holm but are not confirmatory
    sec = [r for r in rows if r["metric"] == "within_tf_auroc" and r["score_b"] == "fisher"][0]
    assert sec["analysis_role"] == "descriptive" and np.isfinite(sec["p_holm_6comparators"])


def test_status_mechanism_scores_order():
    from types import SimpleNamespace
    R = np.array([[-1, -1], [0, -1], [0, 0], [1, 0], [0, 1]])
    cred = np.array([0.30, 0.10, 0.05, 0.9, 0.7])
    sc = H.status_mechanism_scores(SimpleNamespace(R=R), cred)
    assert list(sc[H.STATUS_SCORE]) == [0, 1, 1, 2, 2]
    u = sc[H.UNOBS_LAST_SCORE]
    assert u[0] < u[1:].min() and np.allclose(u[1:], cred[1:])


def test_tf_block_null_matches_context_tf_margins():
    rng = np.random.default_rng(1)
    G, per = 15, 300
    g_tf = np.repeat(np.arange(G), per)
    g_y = rng.random(G * per) < np.repeat(rng.uniform(0.05, 0.5, G), per)
    g_lev = rng.integers(0, 40, G * per)
    ctx_n = np.array([100, 80, 50, 30])
    ctx_k = np.array([20, 5, 25, 3])
    nm, diag = H.tf_block_null(g_tf, g_lev, g_y, 40, ctx_n, ctx_k, 30, np.random.default_rng(2))
    assert np.allclose(nm["n"], ctx_n.sum()) and np.allclose(nm["n_pos"], ctx_k.sum())
    assert diag["n_slots_infeasible"] == 0 and diag["mean_reused_slots"] == 0
    nu, _ = H.tf_block_null(g_tf, g_lev, g_y, 40, ctx_n, None, 30, np.random.default_rng(3))
    assert np.allclose(nu["n"], ctx_n.sum()) and nu["n_pos"].std() > 0
    # an infeasible slot (more positives than any block has) is skipped and counted
    _, d2 = H.tf_block_null(g_tf, g_lev, g_y, 40, np.array([290]), np.array([289]), 5, np.random.default_rng(4))
    assert d2["n_slots_infeasible"] == 1


def test_subset_metrics_equals_evaluating_the_subset():
    tf, y, s, T = _toy(seed=6)
    es = H.EvalSet("u", "r", np.arange(len(y)), tf.astype(str).astype(object), np.arange(len(y)).astype(str).astype(object),
                   y, np.full(len(y), 0.1))
    M = H.tf_bootstrap_weights(es.n_tf, 20, np.random.default_rng(0))
    mask = np.isfinite(s)
    pt, rep = H.subset_metrics(es, s, mask, M)
    sub = es.subset(mask)
    lc = H.level_counts(sub.tf_code, sub.y, sub.null_p, s[mask], sub.n_tf)
    m = H.curve_metrics(*lc.totals())
    assert np.isclose(pt["ap_lift"], m["ap_lift"][0]) and np.isclose(pt["enr_at_500"], m["enr_at_500"][0])
    assert np.isclose(pt["within_tf_auroc"], np.nanmean(H.per_tf_auroc(lc)))
    assert rep["ap_lift"].shape == (20,)


def test_predicted_set_rows_matched_null_ratio_and_holm():
    rng = np.random.default_rng(5)
    n, T = 6000, 6
    tf = np.repeat(np.arange(T), n // T).astype(str).astype(object)
    y = rng.random(n) < 0.2
    es = H.EvalSet("u", "r", np.arange(n), tf, np.arange(n).astype(str).astype(object), y, np.full(n, 0.2))
    good = y & (rng.random(n) < 0.5) | (rng.random(n) < 0.05)
    null = rng.random(n) < 0.12
    pred = {("m", 50): good, ("coverage::m", 50): null, ("e", 50): rng.random(n) < 0.1}
    rows = H.predicted_set_rows(es, pred, None, {"m": "coverage::m"})
    r = {x["score"]: x for x in rows}
    assert np.isclose(r["m"]["enr_ratio_to_null"], r["m"]["enr_tf_matched"] / r["coverage::m"]["enr_tf_matched"])
    assert r["m"]["enr_ratio_to_null_lo"] <= r["m"]["enr_ratio_to_null"] <= r["m"]["enr_ratio_to_null_hi"]
    assert r["coverage::m"]["analysis_status"] == "post_hoc" and "perm_p_holm" not in r["coverage::m"]
    assert np.isclose(r["m"]["perm_p_holm"], min(1.0, 2 * min(r["m"]["perm_p"], r["e"]["perm_p"]))) or \
        r["m"]["perm_p_holm"] >= r["m"]["perm_p"]


# ----------------------------------------------------------------------------
# additions 2026-10-13 (eval_fixes: provenance labels and context-TF coverage)
# ----------------------------------------------------------------------------
def test_ctx_tf_coverage_counts_tfs_missing_from_the_global_set():
    obj = lambda a: np.asarray(a, dtype=object)  # noqa: E731
    ec = H.EvalSet("kidney", "pert_pooled", np.arange(6), obj(["A", "A", "A", "B", "Z", "Z"]), obj(list("abcdef")),
                   np.array([1, 0, 0, 1, 1, 0], bool), np.full(6, 0.3))
    eg = H.EvalSet("global", "pert_pooled", np.arange(4), obj(["A", "A", "B", "C"]), obj(list("abcd")),
                   np.array([1, 0, 1, 0], bool), np.full(4, 0.3))
    c = H.ctx_tf_coverage(ec, eg)
    assert c["n_ctx_tfs_in_glob"] == 2                       # A and B; Z is not evaluable in the global set
    assert c["frac_ctx_edges_tf_in_glob"] == pytest.approx(4 / 6)
    assert c["frac_ctx_pos_tf_in_glob"] == pytest.approx(2 / 3)


def test_sensitivity_kind_labels_post_hoc_module_fits():
    k = H.sensitivity_kind
    assert k(H.VARIANT + "__elig", "family_hb_obs") == "informative_observation_model"
    assert k(H.VARIANT, "family_hb_mixture") == "posterior_mode" and k(H.VARIANT, "family_hb_nuts_mode1") == "posterior_mode"
    assert set(H.POST_HOC_FIT_KINDS) == {"informative_observation_model", "posterior_mode"}
    # the label-free Q20 design keeps its kinds
    assert k("lit_no_ht", "family_hb") == "evidence_variant" and k(H.VARIANT, "family_hb__fam_lineage") == "family_grouping"
    assert k(H.VARIANT, "family_hb__prior_scale2_nuts") == "prior" and k(H.VARIANT, "family_hb_randinit_nuts") == "nuts_initialisation"
    assert k(H.VARIANT, "family_hb_nuts") == "primary_variant_estimator"


def test_every_table_has_a_post_hoc_columns_entry():
    for name in ("estimators", "paired", "methods_context", "universe_decomposition", "downsampling_null",
                 "degree_calibration", "sensitivity_fits", "evidence_status"):
        assert name in H.POST_HOC_COLUMNS
    assert {d["status"] for d in H.ANALYSIS_DECISIONS} <= {"post_hoc", "module_design", "prespecified"}
    assert len({d["id"] for d in H.ANALYSIS_DECISIONS}) == len(H.ANALYSIS_DECISIONS)


# ----------------------------------------------------------------------------
# additions 2026-10-14 (heldfix: composition-adjusted AP lift, verifier evalfix M1, m1, m6, m10)
# ----------------------------------------------------------------------------
def _enumerate_null_ap(tf, y, s, T):
    """Mean AP over every within-TF arrangement of the positives (brute force)."""
    import itertools
    groups = [np.flatnonzero(tf == t) for t in range(T)]
    ks = [int(y[g].sum()) for g in groups]
    combos = [list(itertools.combinations(g, k)) for g, k in zip(groups, ks)]
    aps = []
    for choice in itertools.product(*combos):
        yy = np.zeros(len(y), bool)
        for c in choice:
            yy[list(c)] = True
        lc = H.level_counts(tf, yy, np.zeros(len(y)), s, T)
        aps.append(H.curve_metrics(*lc.totals(), ks=())["ap"][0])
    return float(np.mean(aps))


def test_exact_null_ap_matches_enumeration_with_ties():
    tf = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2])
    y = np.array([1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0], bool)
    s = np.array([3, 3, 2, 1, np.nan, 3, 2, 2, 0, 1, 1, 2.0])        # ties within and across TFs, NaN last
    lc = H.level_counts(tf, y, np.zeros(len(y)), s, 3)
    e = H.exact_null_ap(lc)
    assert e["ap"][0] == pytest.approx(_enumerate_null_ap(tf, y, s, 3), abs=1e-12)
    assert e["ap_lift"][0] == pytest.approx(e["ap"][0] / y.mean())


def test_exact_null_ap_matches_permutation_mean():
    tf, y, s, T = _toy(seed=8, n=3000)
    lc = H.level_counts(tf, y, np.zeros(len(y)), s, T)
    ap_null = H.permuted_ap(lc, 4000, np.random.default_rng(4))
    e = H.exact_null_ap(lc)["ap"][0]
    assert abs(ap_null.mean() - e) < 4 * ap_null.std(ddof=1) / np.sqrt(len(ap_null))


def test_exact_null_ap_bootstrap_weights_equal_independent_tf_copies():
    tf, y, s, T = _toy(seed=9, n=1500, T=4)
    lc = H.level_counts(tf, y, np.zeros(len(y)), s, T)
    w = np.array([2.0, 0.0, 1.0, 3.0])
    got = H.exact_null_ap(lc, w[None, :], chunk=1)["ap"][0]
    # the replicate as a data set in which every copy of a TF is a separate TF
    parts, code = [], 0
    for t in range(T):
        for _ in range(int(w[t])):
            m = tf == t
            parts.append((np.full(m.sum(), code), y[m], s[m]))
            code += 1
    tf2, y2, s2 = (np.concatenate(x) for x in zip(*parts))
    lc2 = H.level_counts(tf2, y2, np.zeros(len(y2)), s2, code)
    assert got == pytest.approx(H.exact_null_ap(lc2)["ap"][0], rel=1e-12)
    ones = H.exact_null_ap(lc, np.ones((3, T)))["ap"]
    assert np.allclose(ones, H.exact_null_ap(lc)["ap"][0])


def test_comp_adj_is_one_for_a_tf_level_score():
    """A score that is constant within each TF (it only orders TFs, e.g. by prevalence) gains AP lift
    from TF composition alone; its composition-adjusted AP lift is exactly 1."""
    rng = np.random.default_rng(2)
    T, per = 6, 400
    tf = np.repeat(np.arange(T), per)
    rate = np.linspace(0.02, 0.5, T)
    y = rng.random(T * per) < rate[tf]
    s = rate[tf]                                   # ranks high-prevalence TFs first
    lc = H.level_counts(tf, y, np.zeros(len(y)), s, T)
    m = H.curve_metrics(*lc.totals(), ks=())
    e = H.exact_null_ap(lc)
    assert m["ap_lift"][0] > 1.3
    assert m["ap"][0] / e["ap"][0] == pytest.approx(1.0, abs=1e-12)


def _toy_evalset_and_scores(seed=11, T=8, per=250):
    rng = np.random.default_rng(seed)
    n = T * per
    tf = np.repeat(np.arange(T), per)
    rate = np.linspace(0.03, 0.4, T)
    y = rng.random(n) < rate[tf]
    es = H.EvalSet("toy", "ref", np.arange(n), tf.astype(str).astype(object), np.arange(n).astype(str).astype(object),
                   y, np.full(n, 0.1))
    signal = {H.PRIMARY: 0.6, "em": 0.5, "fisher": 0.0, "stouffer": 0.2, "vote_count": 0.3,
              "dorothea_level_heldin": 0.1, "collectri_heldin": 0.4}
    scores = {}
    for k, (name, a) in enumerate(signal.items()):
        if name == "fisher":                      # between-TF composition only, no within-TF signal
            v = np.round(0.3 * rng.normal(size=n) + 4.0 * rate[tf], 1)
        else:
            v = np.round(rng.normal(size=n) + a * y, 1)
        scores[name] = H.Score(name, "lcm" if name in ("family_hb", "em") else "pattern_baseline", "edge", v)
    scores[H.STATUS_SCORE] = H.Score(H.STATUS_SCORE, "post_hoc_mechanism", "edge",
                                     np.round(rng.normal(size=n) + 3.0 * rate[tf], 1))
    return es, scores


def test_evaluate_scores_comp_adj_columns_and_paired_contrasts(monkeypatch):
    monkeypatch.setattr(H, "N_BOOT", 200)
    monkeypatch.setattr(H, "N_NULL", 200)
    monkeypatch.setattr(H, "N_BOOT_CONFIRM", 600)
    monkeypatch.setattr(H, "N_PERM_CONFIRM", 3000)
    monkeypatch.setattr(H, "log", lambda msg: None)
    es, scores = _toy_evalset_and_scores()
    ep = np.zeros(es.n, dtype=np.int64)
    rows0, reps0 = H.evaluate_scores(es, scores, ep, confirm=True)
    rows, reps = H.evaluate_scores(es, scores, ep, confirm=True, comp_adj=True)
    # comp_adj consumes no random numbers: every pre-existing value is identical
    for a, b in zip(rows0, rows):
        for k, v in a.items():
            assert (v == b[k]) or (isinstance(v, float) and np.isnan(v) and np.isnan(b[k])), k
    r = {x["score"]: x for x in rows}
    for name, x in r.items():
        assert x["ap_lift_comp_adj"] == pytest.approx(x["ap_lift"] / x["ap_lift_comp_null"])
        assert x["ap_lift_comp_adj_lo"] <= x["ap_lift_comp_adj"] <= x["ap_lift_comp_adj_hi"]
        # the exact null equals the Monte Carlo permutation mean up to Monte Carlo error
        assert x["ap_lift_comp_null"] == pytest.approx(x["perm_ap_lift_null_mean"], rel=0.01)
    for name in (H.PRIMARY,) + H.CONFIRMATORY_COMPARATORS:
        x = r[name]
        assert x["ap_lift_comp_null"] == pytest.approx(x["perm_ap_lift_null_mean_5k"], rel=0.005)
        assert x["ap_lift_comp_adj_lo_10k"] <= x["ap_lift_comp_adj"] <= x["ap_lift_comp_adj_hi_10k"]
        assert x["within_tf_auroc_lo_10k"] <= x["within_tf_auroc"] <= x["within_tf_auroc_hi_10k"]
        assert np.isfinite(x["enr_at_500_lo_10k"]) and np.isfinite(x["enr_at_500_hi_10k"])
    assert "ap_lift_comp_adj" not in r[H.STATUS_SCORE] or np.isfinite(r[H.STATUS_SCORE]["ap_lift_comp_adj"])
    # the TF-composition-driven score has a large pooled lift but no within-TF signal
    fi, hb = r["fisher"], r[H.PRIMARY]
    assert fi["ap_lift"] > 1.3 and fi["ap_lift_comp_adj"] < fi["ap_lift"] - 0.3 and fi["perm_p_5k"] > 0.01
    assert hb["ap_lift_comp_adj_lo_10k"] > 1 and hb["perm_p_5k"] < 0.01
    pr = H.paired_rows(es, reps)
    ca = [p for p in pr if p["metric"] == H.COMP_ADJ_METRIC]
    assert {p["score_b"] for p in ca} == set(H.CONFIRMATORY_COMPARATORS) | {H.STATUS_SCORE}
    assert all(p["analysis_status"] == "post_hoc" and p["analysis_role"] == "descriptive" for p in ca)
    assert all(np.isnan(p["p_holm"]) for p in ca)
    conf = [p for p in ca if p["score_b"] in H.CONFIRMATORY_COMPARATORS]
    assert all(np.isfinite(p["p_holm_6comparators"]) and p["n_boot_10k"] == 600 for p in conf)
    st = [p for p in ca if p["score_b"] == H.STATUS_SCORE][0]
    assert "p_holm_6comparators" not in st and st["n_boot_finite"] == 200
    for p in ca:
        assert p["diff"] == pytest.approx(r[H.PRIMARY]["ap_lift_comp_adj"] - r[p["score_b"]]["ap_lift_comp_adj"])
    # the pre-existing paired rows are unchanged by comp_adj
    pr0 = H.paired_rows(es, reps0)
    assert len(pr) == len(pr0) + len(ca)
    for a, b in zip(pr0, pr):
        assert a.keys() == b.keys()
        for k in a:
            assert (a[k] == b[k]) or (isinstance(a[k], float) and np.isnan(a[k]) and np.isnan(b[k])), k
    # reported_any-style (tagged) runs get no composition-adjusted paired rows
    assert not [p for p in H.paired_rows(es, reps, tag="reported_any") if p["metric"] == H.COMP_ADJ_METRIC]


def test_fisher_sec6_ties_observed_negatives_with_unobserved():
    from types import SimpleNamespace
    from scipy.stats import chi2
    R = np.array([[-1, -1, -1], [0, -1, -1], [0, 0, 0], [1, -1, -1], [1, 0, 0], [1, 1, -1]])
    n = np.array([100.0, 50, 20, 10, 5, 2])
    sc = H.fisher_sec6_score(SimpleNamespace(R=R, n=n))
    assert sc[0] == 0 and sc[1] == 0 and sc[2] == 0             # observed negatives tie with unobserved
    assert np.all(sc[3:] > 0) and sc[5] > sc[3] > sc[4]        # extra observed negatives lower the score
    rate = ((R == 1) * n[:, None]).sum(0) / ((R >= 0) * n[:, None]).sum(0)
    assert sc[4] == pytest.approx(-chi2.logsf(-2 * np.log(rate[0]), 6))
    assert not np.signbit(sc).any()


def test_fit_convergence_flags_mode_mixing(tmp_path):
    import json
    a = tmp_path / "a"
    a.mkdir()
    (a / "draws.npz").write_bytes(b"")
    (a / "meta.json").write_text(json.dumps({"convergence": {"r_hat_max": 1.56}, "init": {"init": "random"}}))
    b = tmp_path / "b"
    b.mkdir()
    (b / "draws.npz").write_bytes(b"")
    (b / "meta.json").write_text(json.dumps({"r_hat_max": 1.004, "init": {"init": "map_laplace"}}))
    c = tmp_path / "c"
    c.mkdir()
    (c / "meta.json").write_text(json.dumps({"r_hat_max": 2.0}))
    fa, fb, fc = H.fit_convergence(a), H.fit_convergence(b), H.fit_convergence(c)
    assert fa["mode_mixing"] is True and fa["fit_r_hat_max"] == 1.56 and fa["fit_init"] == "random"
    assert fb["mode_mixing"] is False and fb["fit_init"] == "map_laplace"
    assert np.isnan(fc["fit_r_hat_max"]) and fc["mode_mixing"] == ""          # MAP fit: not applicable


def test_record_decisions_keeps_rows_of_concurrent_writers(tmp_path):
    import threading
    import pandas as pd
    path = tmp_path / "decisions.csv"
    mods = [f"m{i}" for i in range(8)]

    def write(m):
        for rep in range(5):
            H.record_decisions(m, [dict(id=f"{m}_x", date="2026-10-14", status="post_hoc", item=str(rep), note="")],
                               path=path)

    th = [threading.Thread(target=write, args=(m,)) for m in mods]
    for t in th:
        t.start()
    for t in th:
        t.join()
    d = pd.read_csv(path)
    assert sorted(d["module"]) == mods and set(d["item"].astype(str)) == {"4"}
    assert oct(path.stat().st_mode & 0o777) == "0o644"


def test_comp_adj_is_unbiased_and_its_tf_bootstrap_ci_covers_without_within_tf_signal():
    """Simulation check of the post hoc estimand (``comp_adj_null_coverage``): a score that ranks TFs
    by prevalence but is random within TF has mean composition-adjusted AP lift 1, and the 95%
    TF-bootstrap interval covers 1 in most data sets (25 TFs; small-cluster under-coverage ~0.9)."""
    r = H.comp_adj_null_coverage(25, n_rep=60, n_boot=300)
    assert abs(r["mean"] - 1) < 3 * r["sd"] / np.sqrt(r["n_rep"])
    assert r["coverage"] >= 0.8
