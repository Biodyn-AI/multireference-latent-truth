"""Tests for the hub-bias simulation (src/r2/sim_hub_bias.py)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from r2 import sim_hub_bias as S  # noqa: E402

SMALL = dict(n_tf=40, n_target=150)


@pytest.fixture(scope="module")
def small_sim():
    return S.simulate_dataset("copying", 0, **SMALL)


def test_compress_roundtrip():
    rng = np.random.default_rng(0)
    for lowest in (-1, -2):
        M = rng.integers(lowest, 2, size=(5000, 15)).astype(np.int8)
        R, n, inv = S.compress(M, lowest=lowest)
        assert np.array_equal(R[inv], M)
        assert n.sum() == len(M)
        assert len(np.unique(R, axis=0)) == len(R)


def test_generator_deterministic_and_consistent(small_sim):
    again = S.simulate_dataset("copying", 0, **SMALL)
    assert again.checksum == small_sim.checksum
    assert S.simulate_dataset("copying", 1, **SMALL).checksum != small_sim.checksum
    sim = small_sim
    N = SMALL["n_tf"] * SMALL["n_target"]
    assert sim.data.n.sum() == N and sim.data_elig.n.sum() == N
    plain = sim.data.R[sim.edge_pattern]
    elig = sim.data_elig.R[sim.edge_pattern_elig]
    # plain codes = eligibility codes with -2 collapsed to -1
    assert np.array_equal(plain, np.where(elig < -1, -1, elig))
    # copied and motif-burst edges are non-edges
    assert not (sim.copied & sim.z).any() and not (sim.mburst & sim.z).any()
    # MOTIF/COEXP observe only LIT-reported edges (curated-conditional mask)
    lit_rep = (plain[:, :S.N_LIT] == 1).any(1)
    assert not (plain[:, S.N_LIT:] >= 0)[~lit_rep].any()
    # LIT target-node rule: an observed LIT entry has a reported target for that annotator
    tf = sim.tf_of_edge
    gene = np.tile(np.arange(SMALL["n_target"]), SMALL["n_tf"])
    for j in range(S.N_LIT):
        rep_genes = set(gene[plain[:, j] == 1])
        assert set(gene[plain[:, j] >= 0]) <= rep_genes
        # eligible-but-unobserved (-1) only at covered TFs
        cov_tf = set(tf[elig[:, j] >= -1])
        assert cov_tf == set(np.flatnonzero(sim.rates[S.ANNOTATORS[j]]["cov"]))
    # every TF is covered by >= 1 LIT annotator
    cov = np.stack([sim.rates[a]["cov"] for a in S.ANNOTATORS[:S.N_LIT]])
    assert cov.any(0).all()
    # strata: equal-count deciles of popularity
    assert np.bincount(sim.decile)[1:].tolist() == [SMALL["n_tf"] // 10] * 10
    order = np.argsort(sim.log_u)
    assert np.all(np.diff(sim.decile[order]) >= 0)


def test_scenario_parameters():
    rng = np.random.default_rng(1)
    log_u = rng.normal(0, 1.2, 400)
    eta = rng.normal(0, 0.5, 400)
    p_null, c_null = S.scenario_tf_parameters("null", log_u, eta)
    p_hub, _ = S.scenario_tf_parameters("bio_hub", log_u, eta)
    p_copy, c_copy = S.scenario_tf_parameters("copying", log_u, eta)
    assert np.allclose(p_null, S.SIM_DESIGN["p0"] * np.exp(eta))
    assert np.allclose(p_copy, p_null)
    ratio = p_hub / p_null
    assert np.allclose(np.log(ratio) - 0.3 * log_u, (np.log(ratio) - 0.3 * log_u)[0])   # proportional to u^0.3
    assert np.isclose(np.mean(np.exp(0.3 * log_u) * ratio[0] / np.exp(0.3 * log_u[0])), np.mean(ratio))
    assert np.allclose(c_null, S.SIM_DESIGN["copy_base"])
    assert np.all(np.diff(c_copy[np.argsort(log_u)]) >= 0) and c_copy.max() > 3 * c_null.max()
    with pytest.raises(ValueError):
        S.scenario_tf_parameters("other", log_u, eta)


def test_oracle_matches_bruteforce(small_sim):
    sim = small_sim
    oc = S.oracle_credibility(sim)
    plain = sim.data.R[sim.edge_pattern]
    unobs = ~(plain >= 0).any(1)
    assert np.allclose(oc[unobs], sim.p_edge[unobs])          # no evidence -> prior
    tf = sim.tf_of_edge
    rng = np.random.default_rng(2)
    for e in rng.choice(np.flatnonzero((plain == 1).any(1)), 25, replace=False):
        t = tf[e]
        p, c, m = sim.p_edge[e], sim.c_tf[t], sim.m_tf[t]
        priors = {"T": p, "N": (1 - p) * (1 - c) * (1 - m), "C": (1 - p) * c * (1 - m),
                  "M": (1 - p) * (1 - c) * m, "CM": (1 - p) * c * m}
        lik = {}
        for cl in priors:
            L = 1.0
            for j, a in enumerate(S.ANNOTATORS):
                if plain[e, j] < 0:
                    continue
                rt = sim.rates[a]
                key = cl if cl != "CM" else ("C" if S.FAMILY[j] == "LIT" else "M")
                th = rt[key][t]
                L *= th if plain[e, j] == 1 else 1 - th
            lik[cl] = priors[cl] * L
        assert np.isclose(oc[e], lik["T"] / sum(lik.values()), rtol=1e-8)


def test_oracle_counting_lit_nonreports(small_sim):
    sim = small_sim
    base = S.oracle_credibility(sim)
    full = S.oracle_credibility(sim, lit_unobserved_as_nonreport=True)
    elig = sim.data_elig.R[sim.edge_pattern_elig]
    hidden = (elig[:, :S.N_LIT] == -1).any(1)
    # identical where no covered-but-unobserved LIT entry exists.  A counted non-report is
    # evidence against a true edge relative to a clean non-edge (sensitivity > false-positive
    # rate), so the expected number of true edges drops (single edges can gain when the
    # competing class is a copied edge, whose reporting rate can exceed the sensitivity).
    assert np.allclose(full[~hidden], base[~hidden])
    assert np.any(np.abs(full[hidden] - base[hidden]) > 1e-6) and full.sum() < base.sum()
    # brute force for one edge with hidden entries
    e = int(np.flatnonzero(hidden & (elig == 1).any(1))[0]) if (hidden & (elig == 1).any(1)).any() \
        else int(np.flatnonzero(hidden)[0])
    t_ = sim.tf_of_edge[e]
    p, c, m = sim.p_edge[e], sim.c_tf[t_], sim.m_tf[t_]
    lik = {"T": p, "N": (1 - p) * (1 - c) * (1 - m), "C": (1 - p) * c * (1 - m),
           "M": (1 - p) * (1 - c) * m, "CM": (1 - p) * c * m}
    for j, a in enumerate(S.ANNOTATORS):
        code = elig[e, j]
        if code == -2 or (code == -1 and S.FAMILY[j] != "LIT"):
            continue
        for cl in lik:
            key = cl if cl != "CM" else ("C" if S.FAMILY[j] == "LIT" else "M")
            th = sim.rates[a][key][t_]
            lik[cl] *= th if code == 1 else 1 - th
    assert np.isclose(full[e], lik["T"] / sum(lik.values()), rtol=1e-8)
    rows = S.oracle_check_one(sim)
    assert [r["oracle"] for r in rows] == ["oracle_ignorable_mask", "oracle_lit_nonreports_counted"]
    assert all(0 <= r["precision"] <= 1 for r in rows if r["set_size"])


def test_bfdr_mask_matches_governance_rule():
    rng = np.random.default_rng(3)
    c = np.round(rng.beta(0.3, 2.0, 3000), 2)                  # many ties
    w = rng.integers(1, 20, 3000).astype(float)
    mask = S.bfdr_mask(c, w, 0.05)
    if mask.any():
        assert np.sum(w[mask] * (1 - c[mask])) / np.sum(w[mask]) <= 0.05 + 1e-12
        # whole tie blocks: every element with credibility >= the smallest selected one is in
        assert mask[c >= c[mask].min()].all()
    # adding the next block would exceed q
    nxt = c < (c[mask].min() if mask.any() else np.inf)
    if nxt.any():
        cut = c[nxt].max()
        m2 = c >= cut
        assert np.sum(w[m2] * (1 - c[m2])) / np.sum(w[m2]) > 0.05
    try:
        from r2 import governance as G
    except Exception:  # pragma: no cover - governance needs the full data layer
        return
    assert np.array_equal(mask, G.bfdr_levels(c, w, 0.05, 0.2) == 2)


def test_topk_mask_exact_size_and_order():
    s = np.array([5.0, 3.0, 3.0, 3.0, 1.0, np.nan])
    for seed in range(5):
        m = S.topk_mask(s, 3, seed)
        assert m.sum() == 3 and m[0] and not m[4] and not m[5]
    assert S.topk_mask(s, 0, 0).sum() == 0
    assert S.topk_mask(s, 10, 0).sum() == len(s)


def test_ranking_metrics_against_sklearn():
    from sklearn.metrics import average_precision_score, roc_auc_score
    rng = np.random.default_rng(4)
    y = rng.random(4000) < 0.1
    s = rng.normal(size=4000) + y
    s_tied = np.round(s, 1)
    assert np.isclose(S.auroc(s_tied, y), roc_auc_score(y, s_tied))
    assert np.isclose(S.average_precision(s, y), average_precision_score(y, s))
    g = rng.integers(0, 7, 4000)
    num = den = 0.0
    for k in range(7):
        sel = g == k
        n1, n0 = y[sel].sum(), (~y[sel]).sum()
        num += roc_auc_score(y[sel], s_tied[sel]) * n1 * n0
        den += n1 * n0
    assert np.isclose(S.auroc_within_group(s_tied, y, g), num / den)


def _fake_fit(post, cred_restarts, lp, names):
    restarts = pd.DataFrame(dict(restart=range(len(lp)), log_post=lp, converged=True))
    return dict(meta=dict(class_names=names), cred=post[:, 0], cred_restarts=cred_restarts,
                class_post=post, restarts=restarts, params=dict(w=np.full(len(names), 1 / len(names))))


def test_mode_diagnostics_flags_label_swap(small_sim):
    sim = small_sim
    N = len(sim.z)
    ep = np.arange(N)
    reported = (sim.data.R[sim.edge_pattern] == 1).any(1)
    rng = np.random.default_rng(5)
    good = np.clip(sim.z * 0.8 + rng.random(N) * 0.2, 0, 1)
    bad = rng.random(N) * 0.5
    names = ["T", "N", "B_LIT"]
    # best optimum: T column uninformative, burst column tracks the truth -> swapped
    post = np.stack([bad, 1 - bad - 0.5 * good, 0.5 * good], axis=1)
    fit = _fake_fit(post, np.stack([bad, good]), np.array([-10.0, -12.0]), names)
    d = S.mode_diagnostics(sim, "family_hb", dict(fit=fit, edge_pattern=ep), reported)
    assert d["label_swapped_best"] and d["n_distinct_optima"] == 2
    assert not d["best_optimum_is_best_aligned"] and d["n_restarts_better_aligned"] == 1
    fit2 = _fake_fit(np.stack([good, 1 - good - 0.01, np.full(N, 0.01)], axis=1), np.stack([good, good]),
                     np.array([-10.0, -10.0]), names)
    d2 = S.mode_diagnostics(sim, "family_hb", dict(fit=fit2, edge_pattern=ep), reported)
    assert not d2["label_swapped_best"] and d2["best_optimum_is_best_aligned"] and d2["n_at_best"] == 2
    # the fake class weights (1/3 each) are far above the true prevalence -> inflated class T
    assert d2["t_class_inflated"] and np.isclose(d2["w_T_over_prevalence"], (1 / 3) / sim.z.mean())
    # two-class model: T below chance on reported edges = N tracks the truth better -> swapped
    two = _fake_fit(np.stack([1 - good, good], axis=1), np.stack([1 - good, good]), np.array([-5.0, -9.0]),
                    ["T", "N"])
    two["params"] = dict(w=np.array([sim.z.mean(), 1 - sim.z.mean()]))
    d3 = S.mode_diagnostics(sim, "em_obs", dict(fit=two, edge_pattern=ep), reported)
    assert d3["label_swapped_best"] and d3["best_other_class"] == "N" and d3["best_burst_class"] is None
    assert not d3["t_class_inflated"] and d3["auroc_T_all_best"] < 0.5


def test_end_to_end_small(tmp_path, monkeypatch):
    monkeypatch.setattr(S, "FITS", tmp_path / "fits")
    monkeypatch.setattr(S, "METRICS", tmp_path / "metrics")
    monkeypatch.setattr(S, "OUT", tmp_path)
    monkeypatch.setitem(S.SIM_DESIGN, "n_restarts", {"default": 2})
    sim = S.simulate_dataset("null", 0, n_tf=40, n_target=300)
    for m in ("em", "family_hb"):
        meta = S.fit_one(sim, m)
        assert meta["class_post_vs_cred_max_abs_diff"] < 1e-8
    assert S.fit_one(sim, "em", skip_existing=True) is None
    assert S.fitted_models("null", 0) == ["em", "family_hb"]
    res = S.evaluate_replicate(sim)
    # stale-metrics detection: metrics are current only if they cover exactly the stored fits
    assert not S.metrics_up_to_date("null", 0)
    S.C.write_json({"modes": [{"model": "em"}]}, S.metrics_path("null", 0))
    assert not S.metrics_up_to_date("null", 0)
    S.C.write_json({"modes": res["modes"]}, S.metrics_path("null", 0))
    assert S.metrics_up_to_date("null", 0)
    sm = pd.DataFrame(res["summary"])
    assert set(sm["set_rule"]) == {"bfdr05", "size_matched", "size_oracle"}
    hb = sm[(sm.method == "family_hb") & (sm.set_rule == "bfdr05")].iloc[0]
    k = hb["set_size"]
    assert (sm[sm.set_rule == "size_matched"]["set_size"] == k).all()
    ok = sm["share_true_top_decile"] > 0
    assert np.allclose(sm.loc[ok, "over_promotion_top_decile"],
                       sm.loc[ok, "share_set_top_decile"] / sm.loc[ok, "share_true_top_decile"], equal_nan=True)
    st = pd.DataFrame(res["strata"])
    g = st[(st.method == "oracle") & (st.set_rule == "size_oracle") & (st.stratifier == "popularity_decile")]
    assert np.isclose(g["share_true"].sum(), 1.0) and len(g) == 10
    assert set(pd.DataFrame(res["modes"])["model"]) == {"em", "family_hb"}
    # every row is post hoc (numerical design fixed after the results checkpoint); the basis
    # separates the core design from the informative-observation models
    assert (sm["analysis_status"] == "post_hoc").all()
    assert S.status_basis("family_hb") == "design_fixed_after_checkpoint"
    assert S.status_basis("family_hb_obs") == "model_designed_after_heldout"
    # the internal scenario name "null" would be read back as a missing value by pandas
    lab = S.with_status(pd.DataFrame(dict(scenario=["null", "bio_hub"], method=["em", "em"])))
    assert lab["scenario"].tolist() == ["null_independent", "bio_hub"]
    lab.to_csv(tmp_path / "lab.csv", index=False)
    assert pd.read_csv(tmp_path / "lab.csv")["scenario"].notna().all()
    t = S.with_status(pd.DataFrame(dict(method=["em", "em_obs"], analysis_status=["x", "y"])))
    assert t["analysis_status"].tolist() == ["post_hoc"] * 2
    assert t["status_basis"].tolist() == ["design_fixed_after_checkpoint", "model_designed_after_heldout"]


def test_end_to_end_informative_observation_models(tmp_path, monkeypatch):
    """The informative-observation models are fitted on the eligibility-coded patterns and
    mapped back to edges through their own edge -> pattern index."""
    if not (S.model_available("em_obs") and S.model_available("family_hb_obs")):
        pytest.skip("informative-observation models not in lcm.MODEL_SPECS")
    monkeypatch.setattr(S, "FITS", tmp_path / "fits")
    monkeypatch.setattr(S, "METRICS", tmp_path / "metrics")
    monkeypatch.setattr(S, "OUT", tmp_path)
    monkeypatch.setitem(S.SIM_DESIGN, "n_restarts", {"default": 2})
    sim = S.simulate_dataset("copying", 1, n_tf=40, n_target=300)
    # eligibility codes: LIT -1 only at covered TFs, MOTIF/COEXP never -1
    elig = sim.data_elig.R[sim.edge_pattern_elig]
    assert not (elig[:, S.N_LIT:] == -1).any()
    assert (elig[:, :S.N_LIT] == -1).any() and (elig[:, :S.N_LIT] == -2).any()
    for m in ("em", "family_hb", "em_obs", "family_hb_obs"):
        meta = S.fit_one(sim, m)
        assert meta["class_post_vs_cred_max_abs_diff"] < 1e-8
        assert meta["representation"] == ("eligibility" if m.endswith("_obs") else "plain")
        assert meta["n_patterns"] == (sim.data_elig.P if m.endswith("_obs") else sim.data.P)
    scores, info = S.collect_scores(sim)
    f = info["em_obs"]["fit"]
    assert np.array_equal(scores["em_obs"], f["cred"][sim.edge_pattern_elig])
    assert all(len(v) == len(sim.z) and np.isfinite(v[np.isfinite(v)]).all() for v in scores.values())
    res = S.evaluate_replicate(sim)
    sm = pd.DataFrame(res["summary"])
    assert {"em_obs", "family_hb_obs"} <= set(sm["method"])
    # a BFDR set of a latent-class model is a union of whole patterns of its own representation
    sets = S.promoted_sets(sim, scores, info)
    mask = sets[("em_obs", "bfdr05")]
    per_pat = np.bincount(sim.edge_pattern_elig, weights=mask.astype(float), minlength=sim.data_elig.P)
    assert np.all((per_pat == 0) | (per_pat == sim.data_elig.n))
    md = pd.DataFrame(res["modes"]).set_index("model")
    assert md.loc["family_hb_obs", "class_names"].split()[:2] == ["T", "N"]
    # restart check: fresh restarts (a different seed stream) on the same data; it reports the
    # gain over the design fit and leaves the design fit untouched
    monkeypatch.setitem(S.RESTART_CHECK, "n_restarts", 3)
    before = (S.fit_path("copying", 1, "em_obs") / "meta.json").read_text()
    rc = S.restart_check_one(sim, "em_obs", only_uninflated=False)
    assert rc["n_restarts_check"] == 3 and rc["n_restarts_design"] == 2
    assert np.isclose(rc["log_post_gain"], rc["log_post_check"] - rc["log_post_design"])
    assert rc["higher_optimum_found"] == (rc["log_post_gain"] > S.NEAR_BEST_TOL)
    assert 1 <= rc["n_check_at_check_best"] <= 3
    assert (S.fit_path("copying", 1, "em_obs") / "meta.json").read_text() == before
    assert S.restart_check_one(sim, "em_obs", only_uninflated=False) == rc          # cached
    bm = S.by_mode_table(sm, pd.DataFrame(res["modes"]))
    assert set(bm["selected_optimum"]) <= {"aligned", "inflated_T", "label_swapped"}
    one = bm[(bm.method == "em") & (bm.set_rule == "bfdr05") & (bm.metric == "precision")]
    assert len(one) == 1 and one["n_reps"].iloc[0] == 1
    assert np.isclose(one["mean"].iloc[0], sm[(sm.method == "em") & (sm.set_rule == "bfdr05")]["precision"].iloc[0],
                      equal_nan=True)
    tab = S.restart_check_table()
    assert len(tab) == 1 and tab.loc[0, "model"] == "em_obs"
    sig = S.obs_signature_table()
    sim_rows = sig[(sig.source == "simulation") & (sig.subset == "all")]
    assert {"em_obs", "family_hb_obs", "em", "family_hb"} == set(sim_rows["model"])
    assert sim_rows.set_index("model").loc["em_obs", "eta_T_median"] > 0
    assert np.isnan(sim_rows.set_index("model").loc["em", "eta_T_median"])
    assert (sim_rows["n_fits"] == 1).all()
    # the remnant of an interrupted fit (files but no meta.json) enters no table and no metric
    import shutil
    remnant = S.fit_path("copying", 7, "family_hb_obs")
    shutil.copytree(S.fit_path("copying", 1, "family_hb_obs"), remnant)
    (remnant / "meta.json").unlink()
    sig2 = S.obs_signature_table()
    assert sig2[(sig2.source == "simulation") & (sig2.subset == "all")]["n_fits"].tolist() == sim_rows["n_fits"].tolist()
    assert S.fitted_models("copying", 7) == []


def test_realised_coverage_limits_fit_jobs(monkeypatch):
    """``fit`` reproduces the reported coverage: family_hb_obs only on the 12 realised replicates
    unless --full-design; every other model on every replicate."""
    cov = S.REALISED_COVERAGE["family_hb_obs"]
    assert set(cov) == set(S.SCENARIOS)
    assert sum(len(v) for v in cov.values()) == 12
    assert S.in_realised_coverage("null", 4, "family_hb_obs") and not S.in_realised_coverage("null", 5, "family_hb_obs")
    assert not S.in_realised_coverage("copying", 0, "family_hb_obs")
    assert S.in_realised_coverage("copying", 0, "family_hb_obs", full_design=True)
    assert all(S.in_realised_coverage(s, r, m) for s in S.SCENARIOS for r in range(S.SIM_DESIGN["n_reps"])
               for m in S.SIM_DESIGN["models"] + ["em_obs"])
    calls = []
    monkeypatch.setattr(S, "simulate_dataset", lambda s, r: (s, r))
    monkeypatch.setattr(S, "model_available", lambda m: True)
    monkeypatch.setattr(S, "fit_one", lambda sim, m, skip_existing=False: calls.append((sim, m)))
    S.run_fits(["copying"], [0, 1], ["em", "family_hb_obs"])
    assert calls == [(("copying", 0), "em"), (("copying", 1), "em"), (("copying", 1), "family_hb_obs")]
    calls.clear()
    S.run_fits(["copying"], [0], ["family_hb_obs"])
    assert calls == []
    S.run_fits(["copying"], [0], ["family_hb_obs"], full_design=True)
    assert calls == [(("copying", 0), "family_hb_obs")]


def test_robust_fit_map_excludes_failed_restart(monkeypatch):
    """A restart with a non-finite log posterior is excluded; the best finite restart is
    recovered by re-running it and equals what lcm.fit_map returns without the failure."""
    import copy
    from r2 import lcm
    sim = S.simulate_dataset("null", 2, n_tf=30, n_target=200)
    seed = 11
    clean = lcm.fit_map(sim.data, "family_hb", n_restarts=3, seed=seed)
    lp = np.array([r["log_post"] for r in clean.restarts])
    ib = int(np.argmax(lp))
    # no failure: identical to lcm.fit_map
    same = S.robust_fit_map(sim.data, "family_hb", 3, seed)
    assert same["n_failed_restarts"] == 0 and same["recovered_restart"] is None
    assert np.array_equal(same["cred"], clean.cred) and same["log_post"] == clean.log_post
    # re-running a single restart reproduces it
    one = S.refit_single_restart(sim.data, "family_hb", seed, 3, ib)
    assert np.isclose(one["log_post"], lp[ib], rtol=0, atol=1e-6)
    assert np.max(np.abs(one["cred"] - clean.cred_restarts[ib])) < 1e-8
    # emulate what fit_map returns when another restart fails: NaN log posterior, and
    # argmax(NaN) makes that restart the returned "best"
    bad = (ib + 1) % 3
    broken = copy.deepcopy(clean)
    broken.restarts[bad]["log_post"] = float("nan")
    broken.best = {k: np.full_like(np.asarray(v, dtype=float), 0.5) for k, v in clean.best.items()}
    broken.cred = np.full_like(clean.cred, 0.5)
    broken.log_post = float("nan")
    monkeypatch.setattr(lcm, "fit_map", lambda *a, **k: broken)
    rec = S.robust_fit_map(sim.data, "family_hb", 3, seed)
    assert rec["n_failed_restarts"] == 1 and rec["recovered_restart"] == ib
    assert np.isclose(rec["log_post"], clean.log_post) and np.max(np.abs(rec["cred"] - clean.cred)) < 1e-8
    assert np.allclose(rec["params"]["w"], clean.best["w"], atol=1e-8)
    assert rec["summary"]["n_within_1e3_of_best"] >= 1 and np.isfinite(rec["summary"]["log_post_range"])
