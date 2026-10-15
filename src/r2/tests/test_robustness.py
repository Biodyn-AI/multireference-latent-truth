"""Tests for the dropout-robustness stage (src/r2/robustness.py).

Run: .venv/bin/python -m pytest subproject_24_multireference_latent_truth_model/src/r2/tests/test_robustness.py -q
"""
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import chi2, norm, spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from r2 import lcm  # noqa: E402
from r2 import robustness as RB  # noqa: E402

import jax  # noqa: E402


def _toy(seed=0, N=20000):
    """Synthetic evidence with LIT (4), MOTIF (2) and COEXP (1) annotators."""
    rng = np.random.default_rng(seed)
    fam = ["LIT"] * 4 + ["MOTIF"] * 2 + ["COEXP"]
    K = len(fam)
    beta = np.array([0.01, 0.02, 0.005, 0.01, 0.05, 0.08, 0.03])
    alpha = np.array([0.35, 0.25, 0.15, 0.30, 0.45, 0.50, 0.30])
    gamma = np.array([0.30, 0.25, 0.20, 0.30, 0.60, 0.55, 0.03])
    w = np.array([0.06, 0.84, 0.06, 0.04])
    theta = np.stack([alpha, beta, np.where(np.array(fam) == "LIT", gamma, beta),
                      np.where(np.array(fam) == "MOTIF", gamma, beta)], axis=1)
    cover = rng.random((N, K)) < np.array([0.7, 0.5, 0.4, 0.6, 0.9, 0.9, 0.8])
    z = rng.choice(4, size=N, p=w)
    rep = (rng.random((N, K)) < theta.T[z]).astype(np.int8)
    M = np.where(cover, rep, -1).astype(np.int8)
    data, inv = lcm.PatternData.from_matrix(M, fam, [f"s{j}" for j in range(K)])
    return data, inv


# ----------------------------------------------------------------------------
# metric helpers
# ----------------------------------------------------------------------------
def test_weighted_spearman_equals_edge_level_spearman():
    rng = np.random.default_rng(1)
    x = rng.integers(0, 6, 40).astype(float) / 5          # many ties
    y = x + rng.normal(0, 0.3, 40)
    y[::4] = y[1::4][: len(y[::4])]                        # ties in y as well
    w = rng.integers(1, 9, 40)
    ref = spearmanr(np.repeat(x, w), np.repeat(y, w)).correlation
    assert np.isclose(RB.weighted_spearman(x, y, w), ref, atol=1e-12)


def test_weighted_midranks_ties():
    r = RB.weighted_midranks(np.array([2.0, 1.0, 2.0]), np.array([1, 2, 3]))
    # expanded: 1,1,2,2,2,2 -> ranks of 1 = 1.5, of 2 = 4.5
    assert np.allclose(r, [4.5, 1.5, 4.5])


def test_bfdr_set_prefix_and_whole_tie_blocks():
    c = np.array([0.99, 0.99, 0.9, 0.5])
    w = np.array([10, 5, 3, 100.0])
    assert RB.bfdr_set(c, w, 0.05).tolist() == [True, True, True, False]
    # including only one of the two 0.9 edges would give BFDR exactly 0.05, but the
    # whole tie block gives 0.2 / 3 > 0.05 -> neither is included
    c2 = np.array([1.0, 0.9, 0.9])
    assert RB.bfdr_set(c2, np.ones(3), 0.05).tolist() == [True, False, False]
    # everything qualifies
    assert RB.bfdr_set(np.array([1.0, 1.0]), np.ones(2), 0.05).all()
    # nothing qualifies
    assert not RB.bfdr_set(np.array([0.5, 0.2]), np.ones(2), 0.05).any()


def test_bfdr_set_binary_labels_is_the_positive_set():
    lab = np.array([1.0, 0.0, 1.0, 0.0])
    assert RB.bfdr_set(lab, np.array([3, 100, 2, 50.0])).tolist() == [True, False, True, False]


def test_top_n_edges_matches_lexsort_and_is_consistent():
    rng = np.random.default_rng(2)
    s = rng.integers(0, 5, 200).astype(float)
    key = rng.random(200)
    for n in (1, 7, 50, 199, 200):
        got = RB.top_n_edges(s, key, n)
        ref = np.sort(np.lexsort((-key, -s))[:n])
        assert np.array_equal(got, ref)
    # a tie block cut at the boundary is filled by the same keys in every configuration
    a = RB.top_n_edges(s, key, 50)
    s2 = s.copy()
    s2[s2 == 0] = -1                                       # changes only edges below the cut
    assert np.array_equal(a, RB.top_n_edges(s2, key, 50))


def test_rank_disagreement_definition():
    assert RB.rank_disagreement([3, 2, 1], [1, 2, 3]) == (3, 3)
    assert RB.rank_disagreement([3, 2, 1], [3, 2, 1]) == (0, 3)
    # pairs tied in either vector are not comparable
    assert RB.rank_disagreement([1, 1, 2], [2, 1, 1]) == (1, 1)
    assert RB.rank_disagreement([1, 1], [1, 2]) == (0, 0)


def test_method_metrics_binary_labels_match_standard_f1():
    rng = np.random.default_rng(3)
    P = 30
    n = rng.integers(1, 20, P).astype(float)
    lab = (rng.random(P) < 0.3).astype(float)
    npred = np.minimum(rng.integers(0, 5, (2, P)), n).astype(float)
    m = RB.method_metrics(npred, lab, n)
    for i in range(2):
        tp = npred[i] @ lab
        fp = npred[i].sum() - tp
        fn = n @ lab - tp
        assert np.isclose(m["f1"][i], 2 * tp / (2 * tp + fp + fn))
        assert np.isclose(m["precision"][i], tp / (tp + fp))
        assert np.isclose(m["recall"][i], tp / (tp + fn))


def test_wilson_interval():
    lo, hi = RB.wilson(5, 10)
    assert np.isclose(lo, 0.2366, atol=1e-4) and np.isclose(hi, 0.7634, atol=1e-4)
    lo0, hi0 = RB.wilson(0, 20)
    assert lo0 == 0.0 and 0.15 < hi0 < 0.17


def test_baseline_labels_are_nominal_level_decisions():
    data, _ = _toy(N=3000)
    out = RB.baseline_outputs(data)
    b = lcm.baseline_scores(data)
    p_f = np.exp(-b["fisher"])
    assert np.array_equal(out["fisher"][1] == 1, p_f <= 0.05 + 1e-15)
    assert np.array_equal(out["stouffer"][1] == 1, b["stouffer"] >= norm.isf(0.05))
    assert np.array_equal(out["union"][1], b["union"])
    assert np.array_equal(out["vote_fraction"][0], b["vote_fraction"])
    # chi-square identity used by lcm.baseline_scores
    assert np.isfinite(chi2.isf(0.05, 2))


def test_configurations_and_shared_fit_keys():
    ann = ["a", "b", "c", "d"]
    fam = ["LIT", "LIT", "MOTIF", "COEXP"]
    lin = ["X", "Y", "Y", "Y"]
    cfgs = RB.configurations(ann, fam, lin)
    ids = [c.cid for c in cfgs]
    assert ids[0] == "full" and len(cfgs) == 1 + 4 + 3 + 2
    by = {c.cid: c for c in cfgs}
    assert by["family:LIT"].drop == ("a", "b")
    assert by["lineage:Y"].drop == ("b", "c", "d")
    # family COEXP and source d drop the same annotator -> one fit
    assert by["family:COEXP"].fit_key == by["source:d"].fit_key
    assert by["family:LIT"].fit_key != by["source:a"].fit_key


# ----------------------------------------------------------------------------
# fitting
# ----------------------------------------------------------------------------
def test_fit_starts_reproduces_lcm_fit_map_and_converges():
    data, _ = _toy(N=8000)
    ours = RB.fit_starts(data, "em", jax.random.PRNGKey(5), 2)
    ref = lcm.fit_map(data, "em", n_restarts=2, seed=5)
    assert abs(ours["log_post"] - ref.log_post) < 1e-5
    assert np.max(np.abs(ours["cred"] - ref.cred)) < 1e-4
    assert ours["summary"]["best_converged"]
    assert ours["summary"]["best_grad_inf"] <= RB.GRAD_TOL


@pytest.mark.parametrize("drop", [("s4",), ("s6",), ("s0", "s1", "s2", "s3")])
def test_restrict_params_and_warm_start(drop):
    """Dropping a MOTIF annotator removes the B_MOTIF class; dropping COEXP or LIT
    removes a family.  The restricted values must round-trip through the model's
    transforms and the warm start must reach a converged mode."""
    data, _ = _toy(N=12000)
    full = RB.fit_starts(data, "family_hb", jax.random.PRNGKey(0), 2)
    red, inv = lcm.drop_annotators(data, drop)
    model = lcm.build_model(red, lcm.MODEL_SPECS["family_hb"])
    sites = RB.latent_sites(model)
    vals = RB.restrict_params(full["values"], full["class_names"], data.families, data.annotators,
                              red, list(model.class_names), sites)
    assert set(vals) == set(sites)
    if drop == ("s4",):
        assert "B_MOTIF" not in model.class_names and len(vals["w"]) == 3
    if drop == ("s6",):
        assert vals["zf_b"].shape == (2,)
    assert np.isclose(vals["w"].sum(), 1.0)
    obj = RB._Objective(red, "family_hb", jax.random.PRNGKey(1))
    x = obj.unconstrain(vals, jax.random.PRNGKey(1))
    back = obj.constrained(x)
    for k, v in vals.items():
        assert np.allclose(back[k], v, atol=1e-8), k
    # annotator-level parameters of the kept annotators are unchanged
    keep = [data.annotators.index(a) for a in red.annotators]
    full_b = np.asarray(full["values"]["zj_b"])[keep]
    assert np.allclose(back["zj_b"], full_b)
    fit = RB.fit_starts(red, "family_hb", jax.random.PRNGKey(2), 1, warm_values=vals)
    assert fit["starts"][0]["start"] == "warm"
    assert fit["summary"]["best_converged"]
    assert np.isfinite(fit["summary"]["warm_gap"]) and fit["summary"]["warm_gap"] >= 0


def test_estimator_outputs_dropout_mapping():
    """Labels of a dropout configuration mapped back through the pattern index agree
    with a direct edge-level recomputation."""
    data, inv_edges = _toy(N=5000)
    red, inv = lcm.drop_annotators(data, ("s0",))
    fits = {m: dict(cred=np.linspace(0, 1, red.P), w=np.array([0.1, 0.9])) for m in RB.MODELS}
    out = RB.estimator_outputs(red, fits)
    full_cols = [j for j, a in enumerate(data.annotators) if a != "s0"]
    edge_R = data.R[inv_edges][:, full_cols]
    # recompute union per edge directly
    union_edge = (edge_R == 1).any(1).astype(float)
    assert np.array_equal(out["union"]["label"][inv][inv_edges], union_edge)


def test_warm_start_is_selected_among_equivalent_modes():
    """With only the LIT family left, T and B_LIT of Family-HB are exchangeable (equal
    log posterior); the selection keeps the warm start, i.e. the full-data labelling."""
    data, _ = _toy(N=12000)
    full = RB.fit_starts(data, "family_hb", jax.random.PRNGKey(0), 2)
    red, inv = lcm.drop_annotators(data, ("s4", "s5", "s6"))
    model = lcm.build_model(red, lcm.MODEL_SPECS["family_hb"])
    assert model.class_names == ["T", "N", "B_LIT"]
    vals = RB.restrict_params(full["values"], full["class_names"], data.families, data.annotators,
                              red, list(model.class_names), RB.latent_sites(model))
    fit = RB.fit_starts(red, "family_hb", jax.random.PRNGKey(3), 4, warm_values=vals)
    s = fit["summary"]
    if s["warm_gap"] <= RB.NEAR_BEST:
        assert s["best_start"] == "warm"
    # the selected labelling agrees with the full-data credibility
    w = data.n
    assert RB.weighted_spearman(full["cred"], fit["cred"][inv], w) > 0.5


def test_family_hb_single_family_label_switch_is_exact_and_aligned():
    """With one family the swapped labelling has the same log posterior, and the alignment
    picks the labelling that agrees with the reference credibility."""
    data, _ = _toy(N=8000)
    red, inv = lcm.drop_annotators(data, ("s4", "s5", "s6"))
    fit = RB.fit_starts(red, "family_hb", jax.random.PRNGKey(4), 2)
    obj = RB._Objective(red, "family_hb", jax.random.PRNGKey(4))
    v = {k: np.asarray(x) for k, x in fit["values"].items()}
    sw = dict(v)
    sw["w"] = v["w"][[2, 1, 0]]
    for q in ("mu0", "omega", "tau", "zf", "zj"):
        sw[f"{q}_a"], sw[f"{q}_g"] = v[f"{q}_g"], v[f"{q}_a"]
    x0, x1 = obj.unconstrain(v, jax.random.PRNGKey(0)), obj.unconstrain(sw, jax.random.PRNGKey(0))
    assert abs(obj.f(x0) - obj.f(x1)) < 1e-6
    post = RB.class_posteriors(red, np.asarray(fit["w"]), np.asarray(fit["theta"]))
    assert np.allclose(post[:, 0], fit["cred"], atol=1e-10)
    n = data.n
    obs = (red.R >= 0).any(1)[inv]
    # reference = the burst-class posterior -> the alignment must switch to it
    outs = {"family_hb": dict(score=fit["cred"], label=fit["cred"], w_T=float(fit["w"][0]))}
    fits = {"family_hb": dict(w=fit["w"], theta=fit["theta"], class_names=fit["class_names"])}
    assert RB.align_exchangeable_labels(outs, red, fits, inv, post[:, 2][inv], n, obs)
    assert np.allclose(outs["family_hb"]["label"], post[:, 2])
    outs2 = {"family_hb": dict(score=fit["cred"], label=fit["cred"], w_T=float(fit["w"][0]))}
    assert not RB.align_exchangeable_labels(outs2, red, fits, inv, fit["cred"][inv], n, obs)


# ----------------------------------------------------------------------------
# adaptive global search, identification, size-matched sets, uncertainty (fit_fixes)
# ----------------------------------------------------------------------------
def test_random_start_keys_extend_without_changing_first_starts():
    rng = jax.random.PRNGKey(7)
    shape = (5,)
    keys = jax.random.split(rng, 3)
    for r in range(3):
        ref = np.asarray(jax.random.uniform(keys[r], shape, minval=-2, maxval=2))
        assert np.array_equal(RB.random_start(rng, r, 3, shape), ref)
    extra = [RB.random_start(rng, r, 3, shape) for r in range(3, 6)]
    firsts = [RB.random_start(rng, r, 3, shape) for r in range(3)]
    for e in extra:
        assert np.all(np.abs(e) <= 2) and not any(np.allclose(e, f) for f in firsts)


def test_random_at_best_counts():
    recs = [dict(start="warm", log_post=-10.0), dict(start="random0", log_post=-10.0005),
            dict(start="random1", log_post=-12.0), dict(start="random2", log_post=float("nan"))]
    assert RB.random_at_best(recs) == (3, 1)


def test_adaptive_search_stops_by_rule():
    data, _ = _toy(N=6000)
    red, _ = lcm.drop_annotators(data, ("s0",))
    rng = jax.random.PRNGKey(3)
    base = RB.fit_starts(red, "family_hb", rng, 1)
    ad = RB.fit_starts(red, "family_hb", rng, 1, n_random_max=6, min_random_at_best=2)
    # the initial start is reproduced exactly
    assert abs(base["starts"][0]["log_post"] - ad["starts"][0]["log_post"]) < 1e-9
    s = ad["summary"]
    n_rnd, n_best = RB.random_at_best(ad["starts"])
    assert (s["n_random_starts"], s["n_random_at_best"]) == (n_rnd, n_best)
    assert n_best >= 2 or n_rnd == 6
    # it stopped as soon as the rule was met
    if n_best >= 2 and n_rnd > 1:
        assert RB.random_at_best(ad["starts"][:-1])[1] < 2
    assert s["global_search_ok"] == (n_best >= RB.MIN_RANDOM_AT_BEST)
    assert ad["log_post"] >= base["log_post"] - RB.NEAR_BEST
    # search diagnostic: the reference is the fit of the initial starts alone (history-free)
    assert s["n_extra_random_starts"] == n_rnd - 1
    assert abs(s["initial_rule_log_post"] - base["log_post"]) < 1e-9
    assert np.isclose(s["search_log_post_gain"], ad["log_post"] - base["log_post"], atol=1e-9)
    assert s["search_changed_best_mode"] == (s["search_log_post_gain"] > RB.NEAR_BEST)
    assert np.isclose(s["search_max_abs_cred_change"], np.max(np.abs(ad["cred"] - base["cred"])), atol=1e-9)
    b = base["summary"]
    assert b["search_log_post_gain"] == 0.0 and b["search_max_abs_cred_change"] == 0.0


def test_select_start_and_needs_search_refit():
    recs = [dict(start="warm", log_post=-10.0004), dict(start="random0", log_post=-10.0),
            dict(start="random1", log_post=-13.0)]
    ib, i_max, near, lp, lp_ok = RB.select_start(recs)
    assert (ib, i_max) == (0, 1) and near.tolist() == [True, True, False]      # warm start kept within NEAR_BEST
    recs[0]["log_post"] = -10.5
    assert RB.select_start(recs)[0] == 1
    recs[1]["log_post"] = float("nan")
    assert RB.select_start(recs)[0] == 0 and RB.select_start(recs)[4][1] == -np.inf

    def info(n_random, n_at_best, with_fields=False):
        st = [dict(start="warm", log_post=-5.0)]
        st += [dict(start=f"random{r}", log_post=-1.0 if r < n_at_best else -3.0) for r in range(n_random)]
        if n_at_best == 0:
            st[0]["log_post"] = -1.0
        return dict(starts=st, summary=dict(search_log_post_gain=0.0) if with_fields else {})

    assert not RB.needs_search_refit(info(RB.N_RANDOM_DROPOUT, 2))               # rule met by the initial starts
    assert RB.needs_search_refit(info(RB.N_RANDOM_DROPOUT, 1))                   # rule not met, cap not reached
    assert RB.needs_search_refit(info(RB.N_RANDOM_DROPOUT, 0))
    assert not RB.needs_search_refit(info(RB.N_RANDOM_MAX, 1, with_fields=True)) # cap reached
    assert RB.needs_search_refit(info(9, 2))                                     # extra starts, old summary
    assert not RB.needs_search_refit(info(9, 2, with_fields=True))


def test_family_hb_exchangeable_flag():
    data, _ = _toy(N=3000)
    red1, _ = lcm.drop_annotators(data, ("s4", "s5", "s6"))
    m1 = lcm.build_model(red1, lcm.MODEL_SPECS["family_hb"])
    assert RB.family_hb_exchangeable(red1, m1.class_names)
    red2, _ = lcm.drop_annotators(data, ("s0",))
    m2 = lcm.build_model(red2, lcm.MODEL_SPECS["family_hb"])
    assert not RB.family_hb_exchangeable(red2, m2.class_names)


def test_cluster_bootstrap():
    rng = np.random.default_rng(0)
    lo, hi = RB.cluster_boot_mean_ci(np.full(9, 0.3), np.repeat(["a", "b", "c"], 3), rng, B=200)
    assert np.isclose(lo, 0.3) and np.isclose(hi, 0.3)
    # strong cluster effects: the clustered interval is wider than the iid one
    eff = {"a": -1.0, "b": 0.0, "c": 1.0, "d": 0.5, "e": -0.5}
    cl = np.repeat(list(eff), 3)
    x = np.array([eff[c] for c in cl]) + rng.normal(0, 0.01, len(cl))
    ilo, ihi = RB.boot_mean_ci(x, np.random.default_rng(1), B=2000)
    clo, chi = RB.cluster_boot_mean_ci(x, cl, np.random.default_rng(1), B=2000)
    assert (chi - clo) > 1.3 * (ihi - ilo)
    # singleton clusters = iid resampling (same mean, similar width)
    slo, shi = RB.cluster_boot_mean_ci(x, np.arange(len(x)), np.random.default_rng(2), B=4000)
    assert abs((shi - slo) - (ihi - ilo)) < 0.25 * (ihi - ilo)


def _fake_detail(fhb_switch=()):
    """Synthetic detail table; ``fhb_switch`` = (universe, protocol, config) triples in which Family-HB
    gets a posterior-mode switch (warm gap 2.5, best start random3) with large D / low Jaccard."""
    rows = []
    rng = np.random.default_rng(5)
    for u in ("immune", "kidney", "lung"):
        for est in RB.ESTIMATORS:
            base = dict(universe=u, estimator=est, estimator_type="lcm" if est in RB.MODELS else "baseline",
                        identified=True, n_bfdr=100.0, n_sizematched=50, prevalence=0.01)
            rows.append(dict(base, protocol="full", config="full", top_method_f1="m1", top_method_precision="m1"))
            for prot, cfgs in (("source", ["A", "B", "C", "D"]), ("family", ["LIT", "MOTIF", "COEXP"]),
                               ("lineage", ["CollecTRI", "DoRothEA", "Multi-integrator"])):
                for c in cfgs:
                    ident = not (est == "family_hb" and prot == "lineage" and c == "DoRothEA")
                    shift = 0.1 if est == "family_hb" else 0.0
                    r = dict(base, protocol=prot, config=c, identified=ident,
                             D_f1=rng.random() * 0.1 + shift, D_precision=rng.random() * 0.1,
                             flips_f1=1, comparable_f1=10, flips_precision=0, comparable_precision=10,
                             top_switch_f1=0, top_switch_precision=0,
                             spearman=0.9 - shift, jaccard_bfdr=0.8 - shift, jaccard_top=0.85,
                             jaccard_top_sizematched=0.9 - shift, max_abs_dlabel=0.1, mean_abs_dlabel_obs=0.01,
                             abs_dw_T=0.001 if est in RB.MODELS else np.nan, abs_dprev=0.001)
                    if not ident:
                        for k in RB.NOT_IDENTIFIED_NA:
                            if k in r:
                                r[k] = None if k.startswith("top_method") else np.nan
                    rows.append(r)
    import pandas as pd
    d = pd.DataFrame(rows)
    for k in ("fit_n_starts", "fit_n_converged", "fit_best_grad_inf", "fit_best_hess_min_eig",
              "fit_n_within_1e3", "fit_n_random_at_best", "fit_cred_spread_near_best", "fit_runtime_s", "w_T",
              "fit_n_random_starts", "fit_n_extra_random_starts"):
        d[k] = 1.0
    d["fit_warm_gap"] = 0.0
    d["fit_best_start"] = "warm"
    for k in ("fit_best_converged", "fit_global_search_ok"):
        d[k] = True
    d["fit_search_changed_best_mode"] = False
    d["jaccard_top_obs"] = d["jaccard_top"]
    for u, prot, c in fhb_switch:
        m = (d.universe == u) & (d.protocol == prot) & (d.config == c) & (d.estimator == "family_hb")
        assert m.sum() == 1
        d.loc[m, ["fit_warm_gap", "fit_best_start", "D_f1", "jaccard_top_sizematched"]] = [2.5, "random3", 0.9, 0.1]
    return d


def test_summaries_subsets_and_contrasts_on_synthetic_detail():
    d = _fake_detail()
    ranking, edge, conv = RB.summarise(d)
    assert set(ranking.subset) == {"all", "excl_uninformative", RB.NO_SWITCH, RB.NO_SWITCH_EXCL}
    assert set(ranking[ranking.protocol == "source"].subset) == {"all", RB.NO_SWITCH}
    r = ranking[(ranking.tissue == "immune") & (ranking.estimator == "family_hb") & (ranking.protocol == "lineage")
                & (ranking.metric == "f1")]
    assert r[r.subset == "all"].n_configs_identified.iloc[0] == 2          # DoRothEA not identified
    assert r[r.subset == "excl_uninformative"].n_configs.iloc[0] == 2
    assert {"mean_D_cluster_lo", "ci_status"} <= set(ranking.columns)
    assert (ranking[ranking.protocol != "source"].ci_status.str.startswith("descriptive")).all()
    assert "jaccard_top_sizematched_mean" in edge.columns
    c = RB.estimator_contrasts(d)
    assert "jaccard_top_sizematched" in set(c.metric)
    sub = c[(c.scope == "all_tissues") & (c.protocol == "source") & (c.metric == "jaccard_top_sizematched")
            & (c.estimator == "em")]
    # em is 0.1 more stable than family_hb everywhere -> significant under both bootstraps
    assert np.isclose(sub.mean_diff.iloc[0], 0.1) and sub.significant.iloc[0] and sub.significant_cluster.iloc[0]
    assert sub.n_clusters.iloc[0] == 4 and sub.n_configs.iloc[0] == 12
    lin = c[(c.scope == "immune") & (c.protocol == "lineage") & (c.subset == "all") & (c.metric == "D_f1")
            & (c.estimator == "em")]
    assert lin.n_configs.iloc[0] == 2                                       # non-identified pair dropped
    st = RB.search_table(d)
    assert set(st.model) == set(RB.MODELS) and "fit_global_search_ok" in st.columns
    assert {"n_configs_extra_random_starts", "n_configs_best_mode_changed_by_search",
            "max_n_random_starts"} <= set(conv.columns)


def test_edge_metrics_match_edge_level_bruteforce():
    """Pattern-level stability metrics mapped through the dropout pattern index agree with a
    brute-force computation on the expanded edge vectors."""
    data, inv_edges = _toy(N=4000)
    red, inv = lcm.drop_annotators(data, ("s1",))
    rng = np.random.default_rng(0)
    c_full = np.round(rng.random(data.P), 3)
    c_red = np.round(rng.random(red.P), 3)
    n = data.n
    edge_full = c_full[inv_edges]
    edge_red = c_red[inv][inv_edges]
    # Spearman over edges observed by >= 1 remaining annotator
    obs = (red.R >= 0).any(1)[inv]
    from scipy.stats import spearmanr
    ref = spearmanr(edge_full[obs[inv_edges]], edge_red[obs[inv_edges]]).correlation
    assert np.isclose(RB.weighted_spearman(c_full[obs], c_red[inv][obs], n[obs]), ref, atol=1e-10)
    # BFDR set: the largest prefix of whole tie blocks of the edge vector with mean(1 - c) <= q
    q = 0.3
    order = np.sort(np.unique(edge_red))[::-1]
    best = 0
    for t in order:
        sel = edge_red >= t
        if (1 - edge_red[sel]).mean() <= q:
            best = sel.sum()
    assert RB.bfdr_set(c_red[inv], n, q)[inv_edges].sum() == best
    # size-matched top-n with shared keys = lexsort on the edge vector
    keys = rng.random(len(inv_edges))
    m = 137
    got = RB.top_n_edges(edge_red, keys, m)
    assert np.array_equal(got, np.sort(np.lexsort((-keys, -edge_red))[:m]))
    a, b = RB.top_n_edges(edge_full, keys, m), got
    assert np.isclose(RB.jaccard_idx(a, b), len(set(a) & set(b)) / len(set(a) | set(b)))


def test_analysis_status_labels(tmp_path):
    import pandas as pd
    names = {"rob_config_detail.csv", "rob_edge_stability.csv", "rob_estimator_contrasts.csv",
             "rob_fit_convergence.csv", "rob_method_ranking.csv", "rob_method_scores_full.csv", "rob_fit_search.csv",
             "rob_mode_switches.csv"}
    assert set(RB.FIRST_PASS_COLUMNS) == names
    con = pd.DataFrame(dict(scope="immune", protocol=["source", "family", "family", "source"],
                            subset=["all", "all", "excl_uninformative", "all"], estimator="em", reference="family_hb",
                            metric=["D_f1", "D_f1", "D_f1", "jaccard_top_sizematched"], mean_diff=0.0,
                            mean_diff_cluster_lo=-1.0, mean_diff_cluster_hi=1.0))
    out = RB.label_analysis_status("rob_estimator_contrasts.csv", con)
    assert out["analysis_status"].tolist() == ["prespecified", "prespecified", "post_hoc", "post_hoc"]
    assert set(out["post_hoc_columns"]) == {"subset;mean_diff_cluster_lo;mean_diff_cluster_hi"}
    assert RB.label_analysis_status("rob_estimator_contrasts.csv", out).equals(out)        # idempotent
    rk = pd.DataFrame(dict(tissue="immune", estimator="em", protocol=["lineage", "lineage"],
                           subset=["all", "excl_uninformative"], metric="f1", mean_D=0.1))
    assert RB.label_analysis_status("rob_method_ranking.csv", rk)["analysis_status"].tolist() == ["prespecified",
                                                                                                "post_hoc"]
    # the search diagnostic is a new table; the method table is unchanged since the first pass
    se = pd.DataFrame(dict(universe=["lung"], protocol=["lineage"], config=["CollecTRI"], model=["family_hb"]))
    lab = RB.label_analysis_status("rob_fit_search.csv", se)
    assert lab["analysis_status"].tolist() == ["post_hoc"]
    assert lab["post_hoc_columns"].iloc[0] == "all"
    ms = pd.DataFrame(dict(tissue=["immune"], estimator=["em"], method=["random"], f1=[0.1]))
    lab = RB.label_analysis_status("rob_method_scores_full.csv", ms)
    assert lab["analysis_status"].tolist() == ["prespecified"] and lab["post_hoc_columns"].iloc[0] == ""
    # renamed columns are not post hoc
    assert RB.post_hoc_columns("rob_fit_convergence.csv", ["universe", "n_starts_per_config_min",
                                                           "min_n_random_at_best"]) == ["min_n_random_at_best"]
    p = tmp_path / "eval_analysis_decisions.csv"
    pd.DataFrame([dict(module="other", id="x", date="2026-10-08", status="post_hoc", item="i", note="n")]
                 ).to_csv(p, index=False)
    RB.record_decisions("robustness", RB.ANALYSIS_DECISIONS, p)
    log = pd.read_csv(p)
    assert (log["module"] == "other").sum() == 1
    assert (log["module"] == "robustness").sum() == len(RB.ANALYSIS_DECISIONS)
    assert set(log["status"]) <= {"post_hoc", "module_design"}


def test_method_scores_full_matches_bruteforce():
    """FM table: precision, F1, TF-matched lift and paired deltas equal an edge-level computation."""
    import pandas as pd
    rng = np.random.default_rng(3)
    T, G, P = 12, 40, 30
    tf = np.repeat([f"TF{i:02d}" for i in range(T)], G)
    pattern = rng.integers(0, P, T * G)
    edges = pd.DataFrame(dict(tf=tf, target=np.tile([f"g{j}" for j in range(G)], T), pattern=pattern))
    lab = rng.random(P)
    full_out = {e: dict(score=lab, label=lab if e in RB.MODELS else (lab > 0.5).astype(float), w_T=np.nan)
                for e in RB.ESTIMATORS}
    methods = ["scgpt_attn", "scgpt_attn_uniform", "geneformer_attn", "geneformer_attn_uniform", "random"]
    masks = {m: rng.random(T * G) < p for m, p in zip(methods, (0.2, 0.15, 0.25, 0.2, 0.1))}
    masks["scgpt_attn"][: 3 * G] = False                         # no prediction for three TFs
    panel = ["scgpt_attn", "scgpt_attn_uniform", "geneformer_attn", "random"]
    rows = pd.DataFrame(RB.method_scores_full("toy", None, edges, masks, methods, panel, full_out))
    assert len(rows) == len(RB.ESTIMATORS) * len(methods)
    for est in ("family_hb", "union"):
        le = full_out[est]["label"][pattern]
        ref = {}
        for m in methods:
            mk = masks[m]
            tp = le[mk].sum()
            e0 = sum(mk[tf == t].sum() * le[tf == t].mean() for t in np.unique(tf))
            ref[m] = dict(precision=tp / mk.sum(), f1=2 * tp / (mk.sum() + le.sum()), lift=tp / e0,
                          n_tfs=len(np.unique(tf[mk])))
        g = rows[rows.estimator == est].set_index("method")
        for m in methods:
            r = g.loc[m]
            assert np.isclose(r["precision"], ref[m]["precision"]) and np.isclose(r["f1"], ref[m]["f1"])
            assert np.isclose(r["lift_tf_matched"], ref[m]["lift"]) and r["n_tfs_pred"] == ref[m]["n_tfs"]
            assert r["precision_lo"] <= r["precision_hi"] and r["lift_tf_matched_lo"] <= r["lift_tf_matched_hi"]
            assert r["in_ranking_panel"] == (m in panel)
            if m != "random":
                assert np.isclose(r["delta_f1_vs_random"], ref[m]["f1"] - ref["random"]["f1"])
                assert r["delta_f1_vs_random_lo"] <= r["delta_f1_vs_random_hi"]
        assert g.loc["scgpt_attn", "own_uniform"] == "scgpt_attn_uniform"
        assert np.isclose(g.loc["geneformer_attn", "delta_precision_vs_own_uniform"],
                          ref["geneformer_attn"]["precision"] - ref["geneformer_attn_uniform"]["precision"])
        assert np.isnan(g.loc["random", "delta_f1_vs_random"]) and np.isnan(g.loc["random", "delta_f1_vs_own_uniform"])
        # ranks over the panel only (1 = best F1)
        f1p = pd.Series({m: ref[m]["f1"] for m in panel}).rank(ascending=False, method="min")
        assert g.loc[panel, "rank_f1"].tolist() == f1p.tolist() and np.isnan(g.loc["geneformer_attn_uniform", "rank_f1"])


# ----------------------------------------------------------------------------
# robfix (2026-10-14): mode-switch subsets, observed-edge top sets, seed check, flags
# ----------------------------------------------------------------------------
def test_n_observed_above_unobserved_and_observed_top_sets():
    """n <= n_observed_above_unobserved  <=>  the top-n set over all edges is the top-n set over the observed
    edges (and contains no unobserved edge); the next n reaches the unobserved tie block or its level."""
    rng = np.random.default_rng(11)
    for rep in range(20):
        N = 400
        obs = rng.random(N) < 0.6
        s = np.round(rng.random(N), 2)
        u = 0.3
        s[~obs] = u                                           # unobserved edges share the prior score
        w = np.ones(N)
        key = rng.random(N)
        n_above = RB.n_observed_above_unobserved(s, obs, w)
        assert n_above == float((obs & (s > u)).sum())
        idx = np.flatnonzero(obs)
        for n in (1, int(n_above) // 2 + 1, int(n_above)):
            top = RB.top_n_edges(s, key, n)
            assert obs[top].all()
            assert np.array_equal(top, idx[RB.top_n_edges(s[idx], key[idx], n)])
        top = RB.top_n_edges(s, key, int(n_above) + 1)
        assert (s[top] <= u).any()
    # pattern weights count edges; no unobserved edge -> all observed edges
    assert RB.n_observed_above_unobserved(np.array([0.9, 0.1]), np.array([True, True]), np.array([3, 2])) == 5.0
    assert RB.n_observed_above_unobserved(np.array([0.9, 0.1, 0.5]), np.array([True, True, False]),
                                          np.array([3, 2, 7])) == 3.0


def test_mode_switch_flags_and_annotate_detail():
    d = _fake_detail(fhb_switch=[("lung", "source", "A"), ("kidney", "lineage", "CollecTRI")])
    # a non-Family-HB switch does not define the restricted subsets
    m = (d.universe == "immune") & (d.protocol == "source") & (d.config == "B") & (d.estimator == "em")
    d.loc[m, ["fit_warm_gap", "fit_best_start"]] = [0.5, "random1"]
    d.loc[(d.universe == "immune") & (d.protocol == "source") & (d.config == "C") & (d.estimator == "dependence"),
          "fit_search_changed_best_mode"] = True
    a = RB.annotate_detail(d)
    assert list(a.columns[-len(RB.ROBFIX_DETAIL_COLUMNS):]) == RB.ROBFIX_DETAIL_COLUMNS
    assert RB.annotate_detail(a).equals(a)                                     # idempotent
    sw = a[a.mode_switch.eq(True)]
    assert set(zip(sw.universe, sw.config, sw.estimator)) == {("lung", "A", "family_hb"),
                                                             ("kidney", "CollecTRI", "family_hb"), ("immune", "B", "em")}
    assert a.loc[(a.protocol == "full") | ~a.estimator.isin(RB.MODELS), "mode_switch"].isna().all()
    fhb = a[a.fhb_mode_switch]
    assert set(zip(fhb.universe, fhb.protocol, fhb.config)) == {("lung", "source", "A"), ("kidney", "lineage", "CollecTRI")}
    assert len(fhb) == 2 * len(RB.ESTIMATORS)                                 # every estimator of the configuration
    # values changed post hoc: search-changed fits and non-identified Family-HB rows
    ch = a[a.values_changed_post_hoc]
    assert (("immune", "source", "C", "dependence") in set(zip(ch.universe, ch.protocol, ch.config, ch.estimator)))
    assert set(ch[ch.estimator == "family_hb"].config) == {"DoRothEA"} and (~ch.identified).sum() == 3
    assert np.allclose(a[a.estimator.isin(RB.MODELS)].fit_basin_fraction, 1.0)
    assert a[~a.estimator.isin(RB.MODELS)].fit_basin_fraction.isna().all()
    # an inconsistent stored selection is an error
    bad = d.copy()
    bad.loc[m, "fit_best_start"] = "warm"
    with pytest.raises(ValueError):
        RB.annotate_detail(bad)


def test_switch_subsets_do_not_change_earlier_rows_and_match_bruteforce():
    import pandas as pd
    d0 = _fake_detail()
    d1 = _fake_detail(fhb_switch=[("lung", "source", "A"), ("immune", "family", "MOTIF")])
    # same metric values in both (only the switch flags differ); restore the Family-HB values of d0
    for c in ("D_f1", "jaccard_top_sizematched"):
        d1[c] = d0[c]
    r0, e0, _ = RB.summarise(d0)
    r1, e1, conv1 = RB.summarise(d1)
    c0, c1 = RB.estimator_contrasts(d0), RB.estimator_contrasts(d1)
    sw_cols = ["n_fhb_mode_switch", "n_fhb_mode_switch_excluded", "fhb_mode_switch_excluded"]
    for a, b in ((r0, r1), (e0, e1), (c0, c1)):
        old_a = a[~a.subset.isin(RB.SWITCH_SUBSETS)].drop(columns=sw_cols)
        old_b = b[~b.subset.isin(RB.SWITCH_SUBSETS)].drop(columns=sw_cols)
        pd.testing.assert_frame_equal(old_a.reset_index(drop=True), old_b.reset_index(drop=True))
        assert (a[a.subset.isin(RB.SWITCH_SUBSETS)].n_boot == RB.N_BOOT_POST_HOC).all()
        assert (a[~a.subset.isin(RB.SWITCH_SUBSETS)].n_boot == RB.N_BOOT).all()
    # restricted rows: the switch configuration is removed for every estimator
    q = r1[(r1.tissue == "all_tissues") & (r1.protocol == "source") & (r1.metric == "f1")]
    for est in ("em", "family_hb", "union"):
        rr = q[q.estimator == est].set_index("subset")
        assert rr.loc["all", "n_configs"] == 12 and rr.loc[RB.NO_SWITCH, "n_configs"] == 11
        assert rr.loc["all", "n_fhb_mode_switch"] == 1 and rr.loc[RB.NO_SWITCH, "n_fhb_mode_switch"] == 0
        assert rr.loc[RB.NO_SWITCH, "fhb_mode_switch_excluded"] == "lung:A"
        g = d1[(d1.protocol == "source") & (d1.estimator == est) & ~((d1.universe == "lung") & (d1.config == "A"))]
        assert np.isclose(rr.loc[RB.NO_SWITCH, "mean_D"], g.D_f1.mean())
    fam = r1[(r1.tissue == "immune") & (r1.protocol == "family") & (r1.metric == "f1") & (r1.estimator == "em")]
    fam = fam.set_index("subset")
    assert fam.loc[RB.NO_SWITCH, "n_configs"] == 2 and fam.loc[RB.NO_SWITCH_EXCL, "n_configs"] == 1   # MOTIF, LIT out
    assert fam.loc[RB.NO_SWITCH_EXCL, "fhb_mode_switch_excluded"] == "immune:MOTIF"
    # contrasts over the configurations without a switch = brute-force paired mean
    cc = c1[(c1.scope == "all_tissues") & (c1.protocol == "source") & (c1.subset == RB.NO_SWITCH)
            & (c1.estimator == "em") & (c1.metric == "D_f1")].iloc[0]
    src = d1[(d1.protocol == "source") & ~((d1.universe == "lung") & (d1.config == "A"))]
    piv = src.pivot_table(index=["universe", "config"], columns="estimator", values="D_f1")
    assert np.isclose(cc.mean_diff, (piv["em"] - piv["family_hb"]).mean()) and cc.n_configs == 11
    assert cc.n_clusters == 4 and cc.fhb_mode_switch_excluded == "lung:A" and cc.n_boot == RB.N_BOOT_POST_HOC
    assert "jaccard_top_obs" in set(c1.metric) and "jaccard_top_obs_mean" in e1.columns
    # the convergence table counts the switches
    cv = conv1[(conv1.universe == "lung") & (conv1.model == "family_hb") & (conv1.protocol == "source")].iloc[0]
    assert cv.n_configs_mode_switch == 1 and cv.mode_switch_configs == "A (2.50)"
    st = RB.mode_switch_table(d1)
    assert len(st) == 2 and st.excluded_from_no_switch_subsets.all() and set(st.config) == {"A", "MOTIF"}


def test_seedcheck_selection_keys_and_merge(tmp_path, monkeypatch):
    import json
    import pandas as pd

    def info(n_random, n_at_best):
        st = [dict(start="warm", log_post=-1.0)]
        st += [dict(start=f"random{r}", log_post=-1.0 if r < n_at_best else -3.0) for r in range(n_random)]
        return dict(starts=st)

    assert not RB.needs_seedcheck(info(RB.N_RANDOM_DROPOUT, 2))
    assert RB.needs_seedcheck(info(RB.N_RANDOM_DROPOUT + 1, 2))           # the search needed an extra start
    assert RB.needs_seedcheck(info(RB.N_RANDOM_MAX, 1))                   # not certified
    k1 = np.asarray(jax.random.key_data(RB.seedcheck_rng("lung", "drop_x", "family_hb")))
    k2 = np.asarray(jax.random.key_data(RB.seedcheck_rng("lung", "drop_x", "em")))
    cfg = RB.Config("source", "x", ("x",))
    k3 = np.asarray(jax.random.key_data(RB._rng_for("lung", cfg, "family_hb")))
    assert not np.array_equal(k1, k2) and not np.array_equal(k1, k3)
    monkeypatch.setattr(RB, "SEEDCHECK_DIR", tmp_path)
    (tmp_path / "lung.json").write_text(json.dumps({"drop_x|family_hb": dict(
        universe="lung", fit_key="drop_x", configs=["source:A", "family:COEXP"], model="family_hb", n_starts=30,
        best_log_post=-10.0, best_minus_stored=-0.5, n_at_stored_mode=0, n_better=0, max_abs_cred_diff_best=0.2)}))
    sc = RB.load_seedchecks()
    assert len(sc) == 2 and set(sc.config) == {"A", "COEXP"} and not sc.seedcheck_better_mode_found.any()
    d = _fake_detail()
    d.loc[(d.universe == "lung") & (d.config == "COEXP"), "config"] = "COEXP"
    st = RB.search_table(d, sc)
    hit = st[(st.universe == "lung") & (st.model == "family_hb") & st.config.isin(["A", "COEXP"])]
    assert hit.seedcheck_done.all() and (hit.seedcheck_n_starts == 30).all()
    assert st.seedcheck_done.sum() == 2 and st.seedcheck_n_better.isna().sum() == len(st) - 2
    assert RB.search_table(d, pd.DataFrame(columns=sc.columns)).seedcheck_done.sum() == 0


def test_robfix_analysis_status_labels():
    import pandas as pd
    rk = pd.DataFrame(dict(tissue="immune", estimator="em", protocol="source", subset=["all", RB.NO_SWITCH],
                           metric="f1", mean_D=0.1, n_boot=[RB.N_BOOT, RB.N_BOOT_POST_HOC]))
    lab = RB.label_analysis_status("rob_method_ranking.csv", rk)
    assert lab["analysis_status"].tolist() == ["prespecified", "post_hoc"]
    assert "n_boot" in lab["post_hoc_columns"].iloc[0].split(";")
    con = pd.DataFrame(dict(scope="immune", protocol="source", subset=["all", "all", RB.NO_SWITCH], estimator="em",
                            metric=["D_f1", "jaccard_top_obs", "D_f1"]))
    assert RB.label_analysis_status("rob_estimator_contrasts.csv", con)["analysis_status"].tolist() == [
        "prespecified", "post_hoc", "post_hoc"]
    ms = RB.label_analysis_status("rob_mode_switches.csv", pd.DataFrame(dict(universe=["lung"], model=["family_hb"])))
    assert ms["analysis_status"].tolist() == ["post_hoc"] and ms["post_hoc_columns"].iloc[0] == "all"
    ids = {d["id"] for d in RB.ANALYSIS_DECISIONS}
    assert {"rob_mode_switch_subsets", "rob_jaccard_top_observed", "rob_seedcheck", "rob_values_changed_flag"} <= ids


def test_adopt_seedcheck_merges_starts_and_is_final(tmp_path, monkeypatch):
    """A better fresh optimum becomes the fit; the summary follows fit_starts' fields, the search treats the
    fit as final, the superseded files are kept and the detail columns count the extra starts apart."""
    data, _ = _toy(N=6000)
    red, _ = lcm.drop_annotators(data, ("s0",))
    monkeypatch.setattr(RB, "FIT_DIR", tmp_path)
    old = RB.fit_starts(red, "em", jax.random.PRNGKey(1), 2)
    for r in old["starts"]:                       # pretend the stored fit is a worse local mode
        r["log_post"] -= 5.0
    old["summary"].update(log_post=old["log_post"] - 5.0, initial_rule_log_post=old["log_post"] - 5.0,
                          search_changed_best_mode=False, search_max_abs_cred_change=0.0)
    RB._save_fit(tmp_path / "toy" / "drop_x", ["source:s0"], {"em": old}, dict(universe="toy", drop=["s0"]))
    fit = RB.load_fit("toy", "drop_x")
    new = RB.fit_starts(red, "em", RB.seedcheck_rng("toy", "drop_x", "em"), 3)
    RB.adopt_seedcheck("toy", "drop_x", fit, "em", new, log=lambda *a: None)
    got = RB.load_fit("toy", "drop_x")["em"]
    s = got["summary"]
    assert s["seedcheck_adopted"] and s["best_start"].startswith("seedcheck") and s["n_seedcheck_starts"] == 3
    assert s["n_starts"] == 5 and np.isclose(s["log_post"], new["log_post"])
    assert np.isclose(s["seedcheck_adopted_gain"], new["log_post"] - (old["log_post"] - 5.0))
    assert np.allclose(got["cred"], new["cred"]) and s["search_changed_best_mode"]
    assert RB.random_at_best(got["starts"])[0] == 5
    assert not RB.needs_search_refit(dict(starts=got["starts"], summary=s))
    assert (tmp_path / "toy" / "_seedcheck_superseded" / "drop_x.json").exists()
    rows = RB._fit_rows("toy", "em", RB.Config("source", "s0", ("s0",)), got)
    assert rows["fit_seedcheck_adopted"] and rows["fit_n_random_starts"] == 5 and rows["fit_n_extra_random_starts"] == 0


def test_em_reference_contrasts_are_post_hoc_appended_and_match_bruteforce():
    """assets2 (post hoc): EM-referenced contrasts are appended after the unchanged Family-HB block."""
    import pandas as pd
    d = _fake_detail(fhb_switch=[("lung", "source", "A")])
    c_hb = RB.estimator_contrasts(d)
    c_em = RB.em_reference_contrasts(d)
    allc = RB.all_contrasts(d)
    pd.testing.assert_frame_equal(allc.iloc[:len(c_hb)].reset_index(drop=True), c_hb.reset_index(drop=True))
    assert len(allc) == len(c_hb) + len(c_em) and (allc.iloc[len(c_hb):].reference == RB.EM_REFERENCE).all()
    assert "em" not in set(c_em.estimator) and (c_em.n_boot == RB.N_BOOT_POST_HOC).all()
    assert set(c_em.subset) == set(c_hb.subset) and set(c_em.metric) == set(c_hb.metric)
    excl = lambda x: x[~((x.universe == "lung") & (x.config == "A"))]  # noqa: E731
    for subset, keep in (("all", lambda x: x), (RB.NO_SWITCH, excl)):
        r = c_em[(c_em.scope == "all_tissues") & (c_em.protocol == "source") & (c_em.subset == subset)
                 & (c_em.metric == "D_f1") & (c_em.estimator == "family_hb")].iloc[0]
        piv = keep(d[d.protocol == "source"]).pivot_table(index=["universe", "config"], columns="estimator",
                                                           values="D_f1")
        assert np.isclose(r.mean_diff, (piv["family_hb"] - piv["em"]).mean())
        assert r.mean_diff_cluster_lo <= r.mean_diff <= r.mean_diff_cluster_hi and r.n_clusters == 4
    # the switch columns still refer to Family-HB mode switches
    r = c_em[(c_em.scope == "all_tissues") & (c_em.protocol == "source") & (c_em.subset == RB.NO_SWITCH)
             & (c_em.metric == "D_f1") & (c_em.estimator == "union")].iloc[0]
    assert r.fhb_mode_switch_excluded == "lung:A" and r.n_configs == 11
    # labels: every EM-referenced row is post hoc; the Family-HB block keeps its labels
    lab = RB.label_analysis_status("rob_estimator_contrasts.csv", allc)
    assert (lab.iloc[len(c_hb):].analysis_status == "post_hoc").all()
    pd.testing.assert_frame_equal(lab.iloc[:len(c_hb)].reset_index(drop=True),
                                  RB.label_analysis_status("rob_estimator_contrasts.csv", c_hb).reset_index(drop=True))
    pd.testing.assert_frame_equal(c_em, RB.em_reference_contrasts(d))       # own random streams: reproducible
    assert "rob_em_reference_contrasts" in {x["id"] for x in RB.ANALYSIS_DECISIONS}


def test_record_decisions_keeps_other_modules_rows(tmp_path):
    import pandas as pd
    p = tmp_path / "eval_analysis_decisions.csv"
    other = pd.DataFrame([dict(module="aa", id="x1", date="2026-10-01", status="post_hoc", item="i, with comma", note="n"),
                          dict(module="zz", id="y1", date="2026-10-02", status="prespecified", item="j", note="")])
    other.to_csv(p, index=False)
    RB.record_decisions("robustness", RB.ANALYSIS_DECISIONS, p)
    out = pd.read_csv(p)
    assert (out.module == "robustness").sum() == len(RB.ANALYSIS_DECISIONS)
    kept = out[out.module != "robustness"].reset_index(drop=True)
    pd.testing.assert_frame_equal(kept.fillna(""), other.fillna("")[kept.columns])
    before = p.read_bytes()
    RB.record_decisions("robustness", RB.ANALYSIS_DECISIONS, p)                 # idempotent
    assert p.read_bytes() == before
