"""Tests for r2.governance (label-free tiers, dual rules, statistics, deconcentration)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from r2 import governance as G  # noqa: E402
from r2 import build_heldout_chip as HC  # noqa: E402


# ----------------------------------------------------------------------------- tiers
def test_bfdr_levels_hand_example():
    c = np.array([0.99, 0.99, 0.9, 0.8, 0.5, 0.1])
    w = np.ones(6)
    # cumulative BFDR: .01, .01, .04, .08, .164, .3
    lev = G.bfdr_levels(c, w, 0.05, 0.20)
    assert lev.tolist() == [2, 2, 2, 1, 1, 0]


def test_bfdr_whole_tie_blocks_never_split():
    c = np.array([0.99, 0.9, 0.9, 0.9, 0.2])
    w = np.ones(5)
    # adding the 0.9 block gives (0.01 + 0.3) / 4 = 0.0775 > 0.05: the whole block is excluded
    lev = G.bfdr_levels(c, w, 0.05, 0.20)
    assert lev.tolist() == [2, 1, 1, 1, 0]
    # every tie block has a single level
    for v in np.unique(c):
        assert len(set(lev[c == v])) == 1


def test_bfdr_weighted_patterns_equal_expanded_edges():
    rng = np.random.default_rng(1)
    c = np.round(rng.random(40), 2)            # many ties
    w = rng.integers(1, 6, size=40).astype(float)
    lev_p = G.bfdr_levels(c, w, 0.1, 0.3)
    ce = np.repeat(c, w.astype(int))
    lev_e = G.bfdr_levels(ce, np.ones(len(ce)), 0.1, 0.3)
    assert np.array_equal(np.repeat(lev_p, w.astype(int)), lev_e)
    # realised BFDR of the Strong set is <= q and adding the next block would exceed it
    s = lev_p == 2
    assert G.realized_bfdr(c, w, s) <= 0.1 + 1e-12


def test_bfdr_requires_ordered_q():
    with pytest.raises(ValueError):
        G.bfdr_levels(np.array([0.5]), np.ones(1), 0.3, 0.1)


def test_rank_budget_levels_whole_blocks():
    c = np.array([0.9, 0.8, 0.8, 0.8, 0.5, 0.4])
    w = np.ones(6)
    lev = G.rank_budget_levels(c, w, k1=2, k2=5)
    # k1 = 2 cannot include the 3-edge 0.8 block -> only the first edge; k2 = 5 includes it
    assert lev.tolist() == [2, 1, 1, 1, 1, 0]


def test_dual_rules_match_plan_and_are_nested():
    lv = np.array([0, 1, 2])
    lg, lc = np.meshgrid(lv, lv, indexing="ij")
    prim = G.dual_tier(lg, lc, "primary")
    for a in lv:
        for b in lv:
            t = prim[a, b]
            if a == 2 and b == 2:
                assert t == 3
            elif min(a, b) >= 1:
                assert t == 2
            elif max(a, b) >= 1:
                assert t == 1
            else:
                assert t == 0
    for rule in G.DUAL_LUT:
        assert np.array_equal(G.DUAL_LUT[rule], G.DUAL_LUT[rule].T)       # symmetric in the universes
    p = {r: G.dual_tier(lg, lc, r) >= 2 for r in G.DUAL_LUT}
    assert np.all(p["strict"] <= p["primary"]) and np.all(p["primary"] <= p["lenient"])
    assert G.DUAL_LUT["lenient"][1, 2] == 3 and G.DUAL_LUT["lenient"][0, 2] == 2
    assert G.DUAL_LUT["strict"][1, 1] == 1 and G.DUAL_LUT["strict"][0, 1] == 0


# ----------------------------------------------------------------------------- statistics
def test_mantel_trend_moments_match_within_stratum_permutation():
    rng = np.random.default_rng(3)
    n = 300
    strata = rng.integers(0, 6, size=n)
    x = rng.integers(0, 4, size=n).astype(float)
    y = (rng.random(n) < 0.2 + 0.05 * x).astype(float)
    res = G.mantel_trend(x, y, strata)
    T = []
    for _ in range(4000):
        yp = y.copy()
        for s in np.unique(strata):
            i = np.flatnonzero(strata == s)
            yp[i] = rng.permutation(y[i])
        T.append(float(x @ yp))
    T = np.array(T)
    assert abs(T.mean() - res["E"]) < 0.05 * np.sqrt(res["V"]) + 0.2
    assert abs(T.var() / res["V"] - 1) < 0.1
    assert res["z"] > 0


def test_mantel_trend_null_when_no_variation():
    x = np.zeros(10)
    y = np.array([1, 0] * 5, dtype=float)
    r = G.mantel_trend(x, y, np.zeros(10))
    assert np.isnan(r["z"])


def test_oe_ci_deterministic_and_correct_ratio():
    g = np.array(["a", "a", "b", "b", "c", "c", "d", "e"])
    obs = np.array([1, 0, 1, 1, 0, 0, 1, 1], dtype=float)
    exp = np.full(8, 0.5)
    r1 = G.oe_ci(g, obs, exp, 500, [1, 2])
    r2 = G.oe_ci(g, obs, exp, 500, [1, 2])
    assert r1 == r2
    assert r1["oe"] == pytest.approx(5 / 4)
    assert r1["lo"] <= r1["oe"] <= r1["hi"]


def test_weighted_kappa_matches_sklearn():
    sk = pytest.importorskip("sklearn.metrics")
    rng = np.random.default_rng(5)
    a = rng.integers(0, 3, size=200)
    b = np.clip(a + rng.integers(-1, 2, size=200), 0, 2)
    w = rng.integers(1, 4, size=200).astype(float)
    for wt in ("linear", "quadratic"):
        ref = sk.cohen_kappa_score(a, b, weights=wt, sample_weight=w)
        assert G.weighted_kappa(a, b, w, 3, wt) == pytest.approx(ref, abs=1e-10)


def test_weighted_spearman_matches_expanded_scipy():
    from scipy.stats import spearmanr
    rng = np.random.default_rng(6)
    x = rng.integers(0, 10, size=30).astype(float)          # exact ties (no float-rounding merges)
    y = x + rng.integers(0, 10, size=30)
    w = rng.integers(1, 5, size=30)
    ref = spearmanr(np.repeat(x, w), np.repeat(y, w)).statistic
    assert G.weighted_spearman(x, y, w.astype(float)) == pytest.approx(ref, abs=1e-10)


def test_holm():
    p = np.array([0.01, 0.04, 0.03, np.nan])
    out = G.holm(p)
    assert out[0] == pytest.approx(0.03)
    assert out[2] == pytest.approx(0.06)
    assert out[1] == pytest.approx(0.06)
    assert np.isnan(out[3])


def test_concentration_metrics():
    tfs = ["A"] * 6 + ["B"] * 3 + ["C"]
    r = G.concentration(tfs, n_tf_universe=4)
    assert r["n_edges"] == 10 and r["n_tfs"] == 3
    assert r["top1_tf"] == "A" and r["top1_share"] == pytest.approx(0.6)
    assert r["top2_share"] == pytest.approx(0.9)
    p = np.array([0.6, 0.3, 0.1])
    H = -(p * np.log(p)).sum()
    assert r["evenness"] == pytest.approx(H / np.log(3))
    assert r["entropy_norm_universe"] == pytest.approx(H / np.log(4))
    assert r["top_tfs"].startswith("A:6;B:3")
    single = G.concentration(["A", "A"])
    assert single["entropy"] == 0.0 and np.isnan(single["evenness"])


# ----------------------------------------------------------------------------- deconcentration
@pytest.mark.parametrize("alpha", [0.0, 0.5, 2.0, 16.0])
def test_deconcentrate_equals_literal_greedy(alpha):
    rng = np.random.default_rng(7)
    n = 60
    tf = rng.integers(0, 5, size=n)
    s = np.round(rng.uniform(0.4, 1.0, size=n), 2)       # ties within and across TFs
    keys = rng.integers(0, 2 ** 63, size=n, dtype=np.uint64)
    for B in (1, 7, 20, 60):
        fast = G.deconcentrate(tf, s, keys, B, alpha)
        slow = G.deconcentrate_naive(tf, s, keys, B, alpha)
        assert fast.sum() == B
        assert np.array_equal(fast, slow)


def test_deconcentrate_alpha_spreads_tfs():
    tf = np.array([0] * 10 + [1, 2, 3])
    s = np.array([0.99] * 10 + [0.8, 0.8, 0.8])
    keys = np.arange(13, dtype=np.uint64)
    assert set(tf[G.deconcentrate(tf, s, keys, 4, 0.0)]) == {0}
    assert set(tf[G.deconcentrate(tf, s, keys, 4, 16.0)]) == {0, 1, 2, 3}


# ----------------------------------------------------------------------------- held-out null
def test_masked_degree_index_reproduces_full_grid_null():
    rng = np.random.default_rng(8)
    tfs = [f"T{i}" for i in range(6)]
    genes = tfs[:3] + [f"G{i}" for i in range(80)]
    rows = []
    for i, t in enumerate(tfs):
        for g in genes:
            if g != t:
                rows.append((t, g, bool(rng.random() < 0.1 + 0.05 * i)))
    pairs = pd.DataFrame(rows, columns=["tf", "target", "positive"])
    edges = pairs[pairs["positive"]][["tf", "target"]]
    R = HC.reference_index(edges, genes, assayed=tfs)
    Rm = G.masked_degree_index(pairs, genes)
    # same TFs and genes; same base rates and fitted null
    assert Rm["tfs"] == R["tfs"]
    order = [Rm["gi"][g] for g in sorted(R["gi"], key=R["gi"].get)]
    assert np.array_equal(Rm["deg"][order], R["deg"])
    for t in tfs:
        assert Rm["base"][t] == pytest.approx(R["base"][t])
    assert np.allclose(Rm["P"], R["P"], atol=1e-6)
    lf_full = G.label_frame_from_index(R, pairs["tf"], pairs["target"], masked=False)
    lf_mask = G.label_frame_from_index(Rm, pairs["tf"], pairs["target"], masked=True)
    assert np.array_equal(lf_full["label"].to_numpy(), lf_mask["label"].to_numpy())
    assert np.allclose(lf_full["exp_deg"].to_numpy(), lf_mask["exp_deg"].to_numpy(), atol=1e-6)


def test_masked_labels_non_evaluable_pairs_are_nan():
    pairs = pd.DataFrame({"tf": ["A", "A", "B", "B"], "target": ["x", "y", "x", "z"],
                          "positive": [True, False, False, True]})
    Rm = G.masked_degree_index(pairs, ["x", "y", "z"])
    lf = G.label_frame_from_index(Rm, ["A", "A", "B", "C", "A"], ["x", "z", "z", "x", "A"], masked=True)
    assert lf["evaluable"].tolist() == [True, False, True, False, False]
    assert lf["label"].iloc[0] == 1.0 and lf["label"].iloc[2] == 1.0
    assert np.isnan(lf["label"].iloc[1]) and lf["deg_bin"].iloc[1] == -1


# ----------------------------------------------------------------------------- credibility per draw
def test_draw_credibility_matches_lcm():
    from r2 import lcm
    rng = np.random.default_rng(9)
    K, S, D = 4, 3, 5
    R = rng.integers(-1, 2, size=(12, K)).astype(np.int8)
    W = rng.dirichlet(np.ones(S), size=D)
    TH = rng.uniform(0.05, 0.9, size=(D, K, S))
    out = G.draw_credibility({"w": W, "theta": TH}, R)
    data = lcm.PatternData(R, np.ones(12), ["F"] * K, [f"a{j}" for j in range(K)])
    for d in range(D):
        ref = lcm.credibility_at({"w": W[d], "theta": TH[d]}, data)
        assert np.allclose(out[d], ref)


def test_pattern_key_map_exact_and_raises():
    A = np.array([[1, 0], [0, -1], [1, 1]], dtype=np.int8)
    B = np.array([[0, -1], [1, 1], [1, 0], [-1, -1]], dtype=np.int8)
    assert G.pattern_key_map(A, B).tolist() == [2, 0, 1]
    with pytest.raises(RuntimeError):
        G.pattern_key_map(np.array([[0, 0]], dtype=np.int8), B)


# ----------------------------------------------------------------------------
# revision 2026-10-08 (verifier gov): power / MDE, exact method test, size matching,
# label invariance of the label-free choices
# ----------------------------------------------------------------------------
def test_oe_power_analysis_mde_and_attainable():
    from r2 import replication as RP
    N, K, n = np.array([400, 300, 200, 150]), np.array([60, 30, 50, 10]), np.array([20, 15, 10, 8])
    r = G.oe_power_analysis(N, K, n, {"g": 1.8})
    E0 = np.sum(n * K / N)
    assert np.isclose(r["E0"], E0)
    assert np.isclose(r["max_attainable_oe"], np.minimum(n, K).sum() / E0)
    assert 1.0 < r["mde_oe"] < r["max_attainable_oe"]
    # power at the MDE odds ratio reaches the target; Monte-Carlo check of the exact power
    lor = np.log(r["mde_odds_ratio"])
    s1, p1 = RP.sum_distribution(N, K, n, lor)
    assert p1[max(int(r["t_crit"]) - s1, 0):].sum() >= G.MDE_POWER - 1e-9
    rng = np.random.default_rng(0)
    T = np.zeros(4000)
    for a, b, c in zip(N, K, n):
        lo, pm = RP.hyper_pmf(int(a), int(b), int(c), lor)
        T += lo + rng.choice(len(pm), size=4000, p=pm)
    assert abs(np.mean(T >= r["t_crit"]) - G.MDE_POWER) < 0.03
    assert 0 <= r["power_at_g"] <= 1
    assert r["mde_oe_poisson"] >= 1.0
    # a target above the attainable maximum has no power value
    assert np.isnan(G.oe_power_analysis(N, K, n, {"g": 50.0})["power_at_g"])


def test_exact_promoted_test_matches_simulation():
    N_t, P_t, m_t = np.array([100, 80, 60]), np.array([10, 30, 0]), np.array([20, 10, 5])
    r = G.exact_promoted_test(N_t, P_t, m_t, obs=1)
    rng = np.random.default_rng(1)
    T = sum(rng.hypergeometric(P, N - P, m, size=200_000) for N, P, m in zip(N_t, P_t, m_t))
    assert np.isclose(r["expected"], np.sum(m_t * P_t / N_t))
    assert abs(r["p_le"] - np.mean(T <= 1)) < 0.003 and abs(r["p_ge"] - np.mean(T >= 1)) < 0.003


def test_top_n_mask_size_and_ties():
    keys = np.arange(10, dtype=np.uint64)
    s = np.array([5, 4, 4, 4, 3, np.nan, 2, 2, 1, 0], dtype=float)
    m = G.top_n_mask(s, 3, keys)
    assert m.sum() == 3 and m[0] and m[3] and m[2] and not m[1]      # tie block cut by key (larger first)
    assert G.top_n_mask(s, 20, keys).sum() == 10


def _fake_context(label_seed=1, seed=0, P=80, n_tf=12):
    rng = np.random.default_rng(seed)
    n = rng.integers(1, 30, P).astype(float)
    ep = np.repeat(np.arange(P), n.astype(int))
    N = len(ep)
    cg = np.clip(rng.beta(0.6, 2, P), 1e-4, 1 - 1e-4)
    cg[:8] = rng.uniform(0.97, 0.999, 8)
    cc = np.clip(cg + rng.normal(0, 0.02, P), 1e-4, 1 - 1e-4)
    cx = object.__new__(G.Context)
    cx.t, cx.n_boot, cx.N, cx.ep, cx.n = "toy", 50, N, ep, n
    cx.tf = np.array([f"T{i}" for i in rng.integers(0, n_tf, N)], dtype=object)
    cx.target = np.array([f"g{i}" for i in range(N)], dtype=object)
    cx.tf_list = sorted(set(cx.tf))
    cx.n_tf_universe = len(cx.tf_list)
    cx.tf_code = pd.Series(cx.tf).map({x: i for i, x in enumerate(cx.tf_list)}).to_numpy(np.int64)
    cx.n_obs_p = rng.integers(0, 3, P)
    cx.n_rep_p = np.minimum(cx.n_obs_p, rng.integers(0, 2, P))
    cx.cg = dict(mean=cg, lo=cg * 0.9, hi=np.minimum(cg * 1.05, 1), sd=cg * 0 + 0.01)
    cx.cc = dict(mean=cc, lo=cc * 0.9, hi=np.minimum(cc * 1.05, 1), sd=cc * 0 + 0.01)
    cx.cg_map, cx.cc_map, cx.cg_em, cx.cc_em = cg, cc, np.sqrt(cg), np.sqrt(cc)
    cx.lg, cx.lc = G.bfdr_levels(cg, n), G.bfdr_levels(cc, n)
    cx.dual = G.dual_tier(cx.lg, cx.lc, "primary")
    lr = np.random.default_rng(label_seed)
    cx.labels = {}
    for r in G.REFS:
        ev = lr.random(N) < 0.6
        rate = lr.uniform(0.05, 0.5, cx.n_tf_universe)[cx.tf_code]
        lab = np.where(ev, (lr.random(N) < rate).astype(float), np.nan)
        base = pd.Series(lab).groupby(cx.tf_code).transform("mean").to_numpy()
        cx.labels[r] = pd.DataFrame(dict(evaluable=ev, label=lab, exp_tf=np.where(ev, base, np.nan),
                                         exp_deg=np.where(ev, base, np.nan), deg_bin=np.where(ev, 0, -1)))
    return cx


def test_label_free_choices_do_not_depend_on_heldout_labels():
    a, b = _fake_context(label_seed=1), _fake_context(label_seed=99)
    assert np.array_equal(a.dual, b.dual)
    ga, eva, xa = G.deconcentration(a)
    gb, evb, xb = G.deconcentration(b)
    cols = ["budget", "alpha", "n_selected", "mean_cred", "diversity", "harmonic_mean", "chosen", "chosen_pielou"]
    assert pd.DataFrame(ga)[cols].equals(pd.DataFrame(gb)[cols])
    for k in xa:
        assert np.array_equal(xa[k], xb[k])
    # held-out evaluation differs (labels differ) but set sizes do not
    da, db = pd.DataFrame(eva), pd.DataFrame(evb)
    assert (da["budget"].to_numpy() == db["budget"].to_numpy()).all()
    assert not np.allclose(da["oe_tf"].to_numpy(float), db["oe_tf"].to_numpy(float), equal_nan=True)


def test_sensitivity_rows_size_matched_and_validation_power_columns():
    cx = _fake_context()
    ib = pd.DataFrame(dict(tf=cx.tf[:50], target=cx.target[:50], dorothea_level_heldin=["A", "B", "C", "D", None] * 10,
                           in_collectri_heldin=[True, False] * 25, collectri_curation_effort_heldin=np.arange(50.0)))
    rows = pd.DataFrame(G.sensitivity_rows(cx, ib))
    n_prim = int((cx.edge(cx.dual) >= 2).sum())
    sm = rows[rows["family"] == "size_matched"]
    assert len(sm) and (sm["n_promoted"] == n_prim).all() and (sm["analysis_status"] == "post_hoc").all()
    assert (rows.loc[rows["family"] != "size_matched", "analysis_status"] == "prespecified").all()
    jn = sm[sm["config"].str.startswith("joint_nuts")]
    assert len(jn) == len(G.REFS)
    v, tr = G.validation_rows(cx, {("Promoted", "encode"): 1.7})
    v = pd.DataFrame(v)
    p = v[(v["tier"] == "Promoted") & (v["reference"] == "encode")].iloc[0]
    assert p["global_universe_oe"] == 1.7 and 0 <= p["power_at_global_oe"] <= 1 or np.isnan(p["power_at_global_oe"])
    assert "mde_oe" in v.columns and v.loc[v["tier"] == "Exploratory", "mde_oe"].isna().all()


def test_tier_table_counts_and_draw_intervals():
    cx = _fake_context()
    rng = np.random.default_rng(3)
    D = 40
    cx.draws_g = np.clip(cx.cg["mean"][None, :] + rng.normal(0, 0.005, (D, len(cx.n))), 1e-6, 1 - 1e-6)
    cx.draws_c = np.clip(cx.cc["mean"][None, :] + rng.normal(0, 0.005, (D, len(cx.n))), 1e-6, 1 - 1e-6)
    cx.glob_meta = dict(num_chains=4, num_warmup=500, num_samples=500, n_divergent=5,
                        convergence=dict(r_hat_max=1.03, r_hat_max_cred=1.01, ess_bulk_min=132.0))
    cx.cc_meta = dict(num_chains=4, num_warmup=1000, num_samples=1000, n_divergent=0, convergence=dict(r_hat_max=1.0))
    t = pd.DataFrame(G.tier_table(cx))
    dual = t[t["scope"] == "dual"].set_index("tier")
    n_e = cx.n
    for T_, name in G.DUAL_NAMES.items():
        assert dual.loc[name, "n_edges"] == int(n_e[cx.dual == T_].sum())
    assert dual.loc["Promoted", "n_edges"] == dual.loc["Strong", "n_edges"] + dual.loc["Moderate", "n_edges"]
    assert (t["n_edges_draw_lo"] <= t["n_edges_draw_hi"]).all()
    assert (t["global_nuts_warmup"] == 500).all() and (t["context_nuts_warmup"] == 1000).all()
    assert cx.p_promoted_pattern.shape == (len(n_e),)


def test_top_tf_share_and_set_eval_columns():
    r = G.top_tf_share(np.array(["B", "A", "B", "C", "A", "B"], dtype=object))
    assert r["top_tf_eval"] == "B" and abs(r["top_tf_share_eval"] - 0.5) < 1e-12
    tie = G.top_tf_share(np.array(["B", "A"], dtype=object))
    assert tie["top_tf_eval"] == "A" and tie["top_tf_share_eval"] == 0.5        # ties: alphabetical
    empty = G.top_tf_share(np.array([], dtype=object))
    assert empty["top_tf_eval"] == "" and np.isnan(empty["top_tf_share_eval"])
    cx = _fake_context()
    mask = cx.edge(cx.dual) >= 2
    ev = cx.set_eval(mask, G.REFS[0], "x")
    m = mask & cx.labels[G.REFS[0]]["evaluable"].to_numpy()
    vc = pd.Series(cx.tf[m]).value_counts()
    assert ev["top_tf_share_eval"] == pytest.approx(vc.max() / m.sum()) and vc[ev["top_tf_eval"]] == vc.max()


def test_write_table_adds_status_and_post_hoc_columns(tmp_path, monkeypatch):
    monkeypatch.setattr(G.C, "TABLES", tmp_path)
    monkeypatch.setattr(G, "GOV", tmp_path / "gov")
    G.write_table(pd.DataFrame(dict(a=[1, 2])), "gov_promoted_edges.csv")
    G.write_table(pd.DataFrame(dict(a=[1, 2], analysis_status=["post_hoc", "prespecified"])), "gov_tier_validation.csv")
    e = pd.read_csv(tmp_path / "gov_promoted_edges.csv")
    v = pd.read_csv(tmp_path / "gov_tier_validation.csv")
    assert (e["analysis_status"] == "prespecified").all() and e["post_hoc_columns"].isna().all()
    assert v["analysis_status"].tolist() == ["post_hoc", "prespecified"]
    assert "mde_oe" in v["post_hoc_columns"].iloc[0] and "top_tf_share_eval" in v["post_hoc_columns"].iloc[0]
    assert set(G.POST_HOC_COLUMNS) >= {"gov_tier_validation.csv", "gov_deconcentration.csv", "gov_promoted_by_method.csv"}


def test_global_validation_end_to_end_on_a_toy_universe(monkeypatch):
    """global_validation: tier rows of a toy global universe against a toy pooled perturbation
    reference; O/E recomputed directly, Promoted = Strong U Moderate, labels do not move tiers."""
    rng = np.random.default_rng(11)
    tfs = [f"T{i}" for i in range(8)]
    genes = [f"g{i}" for i in range(60)]
    N = len(tfs) * len(genes)
    Pn = 30
    ep = rng.integers(0, Pn, N)
    cred = np.clip(rng.beta(0.5, 3, Pn), 1e-4, 1 - 1e-4)
    cred[:4] = [0.999, 0.99, 0.9, 0.8]
    glob = dict(tfs=tfs, genes=genes, ep=ep, n=np.bincount(ep, minlength=Pn).astype(float),
                nuts=dict(mean=cred))

    def pooled(seed):
        r = np.random.default_rng(seed)
        tf = np.repeat(tfs[:6], len(genes))
        tg = np.tile(genes, 6)
        keep = r.random(len(tf)) < 0.8                       # ragged: not every pair tested
        return pd.DataFrame(dict(tf=tf[keep], target=tg[keep], positive=r.random(keep.sum()) < 0.25))

    monkeypatch.setattr(G, "build_reference_indices",
                        lambda eval_genes, p: {"pert_pooled": (G.masked_degree_index(p, eval_genes), True)})
    pl = pooled(1)
    rows, trend, info = G.global_validation(glob, pl, n_boot=50)
    v = pd.DataFrame(rows).set_index("tier")
    assert set(v.index) == {"Strong", "Moderate", "Below", "Promoted"}
    assert v.loc["Promoted", "obs"] == v.loc["Strong", "obs"] + v.loc["Moderate", "obs"]
    assert v.loc["Promoted", "n_eval"] == v.loc["Strong", "n_eval"] + v.loc["Moderate", "n_eval"]
    assert v["n_eval"].drop("Promoted").sum() == info["pert_pooled"]["n_eval"] == len(pl)
    assert v.loc["Promoted", "analysis_status"] == "post_hoc" and v.loc["Strong", "analysis_status"] == "prespecified"
    # direct recomputation of the Strong row
    lev = G.bfdr_levels(cred, glob["n"])
    ti = {t: i for i, t in enumerate(tfs)}
    gi = {g: i for i, g in enumerate(genes)}
    idx = pl["tf"].map(ti).to_numpy() * len(genes) + pl["target"].map(gi).to_numpy()
    strong = lev[ep[idx]] == 2
    base = pl.groupby("tf")["positive"].mean()
    exp = base.reindex(pl["tf"]).to_numpy()[strong].sum()
    assert v.loc["Strong", "obs"] == pl["positive"].to_numpy()[strong].sum()
    assert v.loc["Strong", "exp_tf"] == pytest.approx(exp)
    assert v.loc["Strong", "oe_tf"] == pytest.approx(pl["positive"].to_numpy()[strong].sum() / exp)
    assert {"mde_oe", "max_attainable_oe", "top_tf_eval", "top_tf_share_eval"} <= set(v.columns)
    assert len(trend) == 2 and {t["strata"] for t in trend} == {"tf", "tf_x_degree"}
    # other labels: tier membership (n_edges, n_eval per tier given the same tested pairs) is label-free
    pl2 = pl.assign(positive=np.random.default_rng(5).random(len(pl)) < 0.25)
    v2 = pd.DataFrame(G.global_validation(glob, pl2, n_boot=50)[0]).set_index("tier")
    assert (v2["n_edges"] == v["n_edges"]).all() and (v2["n_eval"] == v["n_eval"]).all()
