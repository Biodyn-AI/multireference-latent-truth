"""Tests for the primary-fit stage (src/r2/fit_primary.py)."""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from r2 import fit_primary as F  # noqa: E402
from r2 import lcm  # noqa: E402


def _toy(seed=0, N=20000):
    rng = np.random.default_rng(seed)
    fam = ["LIT"] * 4 + ["MOTIF"] * 2 + ["COEXP"]
    K = len(fam)
    beta = np.array([0.01, 0.02, 0.005, 0.01, 0.05, 0.08, 0.03])
    alpha = np.array([0.35, 0.25, 0.15, 0.30, 0.45, 0.50, 0.30])
    gamma = np.array([0.30, 0.25, 0.20, 0.30, 0.60, 0.55, 0.03])
    w = np.array([0.06, 0.84, 0.06, 0.04])
    theta = np.stack([alpha, beta,
                      np.where(np.array(fam) == "LIT", gamma, beta),
                      np.where(np.array(fam) == "MOTIF", gamma, beta)], axis=1)
    cover = rng.random((N, K)) < np.array([0.7, 0.5, 0.4, 0.6, 0.9, 0.9, 0.8])
    cover[: N // 10] = False                                   # some unobserved edges
    z = rng.choice(4, size=N, p=w)
    rep = (rng.random((N, K)) < theta.T[z]).astype(np.int8)
    M = np.where(cover, rep, -1).astype(np.int8)
    data, inv = lcm.PatternData.from_matrix(M, fam, [f"s{j}" for j in range(K)])
    return data, inv, dict(w=w, theta=theta)


def test_weighted_spearman_matches_expanded():
    rng = np.random.default_rng(1)
    x = np.round(rng.random(40), 1)          # many ties
    y = x + rng.normal(0, 0.2, 40)
    w = rng.integers(1, 6, 40)
    xe, ye = np.repeat(x, w), np.repeat(y, w)
    assert np.isclose(F.weighted_spearman(x, y, w), spearmanr(xe, ye).correlation)
    from scipy.stats import rankdata
    r = F.weighted_midranks(x, w)
    assert np.allclose(np.repeat(r, w), rankdata(xe))


def test_bfdr_set_brute_force():
    rng = np.random.default_rng(2)
    cred = np.round(rng.random(30), 2)
    cred[:5] = 0.97                                 # a tie block
    n = rng.integers(1, 20, 30).astype(float)
    for q in (0.05, 0.2, 0.5):
        m = F.bfdr_set(cred, n, q)
        # whole tie blocks only
        for c in np.unique(cred):
            assert m[cred == c].all() or not m[cred == c].any()
        # brute force: thresholds t, set = cred >= t
        best = 0.0
        for t in np.unique(cred):
            s = cred >= t
            if (n[s] * (1 - cred[s])).sum() / n[s].sum() <= q:
                best = max(best, n[s].sum())
        assert np.isclose(n[m].sum(), best)


def test_topk_blocks():
    cred = np.array([0.9, 0.9, 0.5, 0.4, 0.1])
    n = np.array([3.0, 2.0, 4.0, 1.0, 10.0])
    assert F.topk_blocks(cred, n, 4).tolist() == [True, True, False, False, False]
    assert F.topk_blocks(cred, n, 6).tolist() == [True, True, True, False, False]
    assert F.topk_blocks(cred, n, 100).all()


def test_annotator_parameters_layouts():
    fam = ["LIT", "LIT", "MOTIF", "MOTIF", "COEXP"]
    K = len(fam)
    alpha, beta = np.linspace(0.3, 0.7, K), np.linspace(0.01, 0.05, K)
    gL, gM = np.full(K, 0.2), np.full(K, 0.25)
    th = np.stack([alpha, beta, np.where(np.array(fam) == "LIT", gL, beta),
                   np.where(np.array(fam) == "MOTIF", gM, beta)], axis=1)
    P = F.annotator_parameters(th, ["T", "N", "B_LIT", "B_MOTIF"], fam)
    assert np.allclose(P["alpha"], alpha) and np.allclose(P["beta"], beta)
    assert np.allclose(P["gamma"][:2], 0.2) and np.allclose(P["gamma"][2:4], 0.25)
    assert np.isnan(P["gamma"][4])                     # COEXP has no burst class
    assert np.allclose(P["lr_pos"], alpha / beta)
    assert np.allclose(P["alpha_over_gamma"][:2], alpha[:2] / 0.2) and np.isnan(P["alpha_over_gamma"][4])
    # marginal LR+ against the mixture of all non-T classes
    w = np.array([0.1, 0.6, 0.2, 0.1])
    Pm = F.annotator_parameters(th, ["T", "N", "B_LIT", "B_MOTIF"], fam, w=w)
    for j in range(K):
        den = (w[1] * th[j, 1] + w[2] * th[j, 2] + w[3] * th[j, 3]) / (1 - w[0])
        assert np.isclose(Pm["lr_pos_marginal"][j], alpha[j] / den)
    # a burst class reporting more than T makes the marginal LR+ far smaller than alpha/beta
    assert (Pm["lr_pos_marginal"][:2] < Pm["lr_pos"][:2]).all()
    # global-burst layout, with a leading draw axis
    th2 = np.stack([alpha, beta, gL, gM], axis=1)[None].repeat(3, 0)
    P2 = F.annotator_parameters(th2, ["T", "N", "B1", "B2"], fam, w=np.tile(w, (3, 1)))
    assert P2["gamma"].shape == (3, K) and np.allclose(P2["gamma2"], 0.25)
    assert P2["lr_pos_marginal"].shape == (3, K)


def test_class_posteriors_consistent_with_lcm():
    data, inv, truth = _toy(N=3000)
    cp = F.class_posteriors(data, truth["w"], truth["theta"])
    assert np.allclose(cp.sum(1), 1)
    assert np.allclose(cp[:, 0], lcm.credibility_at(truth, data))
    info = F.informative_edges(data, ["T", "N", "B_LIT", "B_MOTIF"])
    obs = (data.R >= 0).any(1)
    assert info["T"] == data.n[obs].sum()
    assert info["B_MOTIF"] <= info["T"]


def test_compare_creds_identity():
    data, inv, truth = _toy(N=3000)
    c = lcm.credibility_at(truth, data)
    r = F.compare_creds(c, c, data)
    assert np.isclose(r["spearman_w"], 1.0) and r["max_abs_diff"] == 0.0
    jac = {k: r[k] for k in r if k.startswith("jaccard")}
    assert all(np.isnan(v) or np.isclose(v, 1.0) for v in jac.values())
    assert sum(np.isfinite(v) for v in jac.values()) >= 3          # BFDR sets + small k are defined


def test_run_map_writes_contract(tmp_path):
    data, inv, truth = _toy()
    meta = F.run_map("toy", "family_hb", variant="v", n_restarts=3, seed=5, root=tmp_path, data=data)
    d = tmp_path / "toy" / "v" / "family_hb"
    cred = np.load(d / "cred.npy")
    assert cred.dtype == np.float64 and cred.shape == (data.P,)
    p = np.load(d / "params.npz")
    assert p["theta"].shape == (data.K, 4) and np.isclose(p["w"].sum(), 1)
    assert np.allclose(cred, lcm.credibility_at({"w": p["w"], "theta": p["theta"]}, data))
    m = json.loads((d / "meta.json").read_text())
    assert m["class_names"] == ["T", "N", "B_LIT", "B_MOTIF"] and m["n_restarts"] == 3
    assert (d / "restarts.csv").exists()
    # polishing the optimum changes nothing material; the stored optimum is the polished one
    assert m["polish"]["max_abs_cred_diff"] < 1e-4
    assert m["polish"]["hess_min_eig"] > 0
    c = m["convergence"]
    assert c["converged"] and c["grad_norm_best"] < F.GRAD_TOL and c["n_grad_converged"] >= 1
    assert "n_converged" not in c and "n_lbfgs_ftol_stop" in c
    import pandas as pd
    rs = pd.read_csv(d / "restarts.csv")
    assert {"lbfgs_log_post", "lbfgs_ftol_stop", "grad_converged", "grad_norm"} <= set(rs.columns)
    assert (rs.log_post >= rs.lbfgs_log_post - 1e-9).all()
    # credibility close to the oracle posterior on this well-specified toy
    oracle = lcm.credibility_at(truth, data)
    assert np.average(np.abs(cred - oracle), weights=data.n) < 0.05


def test_run_nuts_writes_contract(tmp_path):
    data, inv, truth = _toy(N=5000)
    F.run_nuts("toy", variant="v", root=tmp_path, data=data, seed=3, init="random",
               nuts_kw=dict(num_chains=2, num_warmup=40, num_samples=30))
    d = tmp_path / "toy" / "v" / "family_hb_randinit_nuts"
    mean, lo, hi = (np.load(d / f"cred_{s}.npy") for s in ("mean", "lo", "hi"))
    assert np.allclose(np.load(d / "cred.npy"), mean)
    assert (lo <= mean + 1e-12).all() and (mean <= hi + 1e-12).all()
    dr = np.load(d / "draws.npz")
    assert dr["w"].shape[0] == 60 and dr["theta"].shape[1:] == (data.K, 4)
    diag = (d / "diagnostics.csv").read_text()
    assert "cred_observed" in diag and "theta" in diag
    m = json.loads((d / "meta.json").read_text())
    assert "n_divergent" in m and m["num_chains"] == 2


def test_optima_table_groups_restarts():
    data, inv, truth = _toy(N=3000)
    c = lcm.credibility_at(truth, data)
    lp = np.array([-10.0, -10.0005, -12.0, -10.0002, -12.0004])
    creds = np.stack([c, c, 1 - c, c, 1 - c])
    o = F.optima_table(lp, creds, data)
    assert o.n_restarts.tolist() == [3, 2]
    assert np.isclose(o.gap_to_best.iloc[1], 2.0)
    assert np.isclose(o.spearman_w_vs_best.iloc[0], 1.0) and o.spearman_w_vs_best.iloc[1] < 0


def test_replay_and_laplace(tmp_path):
    data, inv, truth = _toy(N=8000)
    F.run_map("toy", "family_hb", variant="v", n_restarts=3, seed=5, root=tmp_path, data=data)
    opt = F.laplace_optima("toy", "family_hb", variant="v", root=tmp_path, data=data)
    # replaying a restart reproduces its optimum exactly (deterministic optimiser)
    assert np.all(np.abs(opt["replay_log_post_diff"]) < 1e-6)
    assert np.isclose(opt["laplace_rel_mass"].sum(), 1.0)
    assert (opt["laplace_min_eig"] > 0).all()
    assert (tmp_path / "toy" / "v" / "family_hb" / "optima_params.npz").exists()


def test_run_nuts_map_laplace_init(tmp_path):
    data, inv, truth = _toy(N=5000)
    F.run_map("toy", "family_hb", variant="v", n_restarts=2, seed=5, root=tmp_path, data=data)
    meta = F.run_nuts("toy", variant="v", root=tmp_path, data=data, seed=3, init="map_laplace",
                      nuts_kw=dict(num_chains=2, num_warmup=40, num_samples=30))
    d = tmp_path / "toy" / "v" / "family_hb_nuts"
    assert meta["fit_id"] == "family_hb_nuts" and meta["init"]["init"] == "map_laplace"
    ch = (d / "chains.csv").read_text()
    assert "nearest_map_optimum" in ch
    assert meta["convergence"]["n_chains_at_best_map_optimum"] == 2


def test_map_restarts_reproduce_lcm_fit_map():
    """map_restarts runs exactly the restarts of lcm.fit_map (same keys, starts, optimiser)."""
    data, inv, truth = _toy(N=4000)
    for model in ("em", "family_hb"):
        ref = lcm.fit_map(data, model, n_restarts=3, seed=11)
        mr = F.map_restarts(data, model, 3, seed=11)
        lp_ref = np.array([r["log_post"] for r in ref.restarts])
        lp_l = np.array([r["lbfgs_log_post"] for r in mr["restarts"]])
        assert np.allclose(lp_l, lp_ref, rtol=0, atol=1e-8)
        assert [r["n_iter"] for r in mr["restarts"]] == [r["n_iter"] for r in ref.restarts]
        assert np.allclose(mr["cred_restarts_lbfgs"], ref.cred_restarts, atol=1e-10)
        # polishing never lowers the log posterior and moves credibility very little
        lp_p = np.array([r["log_post"] for r in mr["restarts"]])
        assert (lp_p >= lp_l - 1e-9).all()
        assert np.max(np.abs(mr["cred_restarts"] - mr["cred_restarts_lbfgs"])) < 1e-3


def test_newton_polish_reaches_gradient_tolerance():
    data, inv, truth = _toy(N=4000)
    pt = F._potential(data, "em", seed=0, init_radius=2.0)
    from scipy.optimize import minimize
    import jax.numpy as jnp
    x0 = np.zeros(len(np.asarray(pt["flat0"])))
    res = minimize(lambda x: float(pt["pot"](jnp.asarray(x))), x0, jac=lambda x: np.asarray(pt["grad"](jnp.asarray(x))),
                   method="L-BFGS-B", options=dict(maxiter=50, gtol=1e-3))          # deliberately loose
    x, rec = F.newton_polish(pt, res.x)
    assert rec["converged"] and rec["grad_norm_after"] < F.GRAD_TOL <= rec["grad_norm_before"]
    assert rec["hess_min_eig"] > 0 and rec["newton_decrement"] >= 0 and rec["converged_relative"]


def test_compare_creds_observed_edges_only():
    data, inv, truth = _toy(N=3000)
    c = lcm.credibility_at(truth, data)
    obs = F.observed_patterns(data)
    assert (~obs).any()
    n_above = F.n_observed_above_unobserved(c, data)
    thr = c[~obs].max()
    assert n_above == data.n[obs & (c > thr)].sum()
    rng = np.random.default_rng(0)
    c2 = np.clip(c + rng.normal(0, 0.01, len(c)), 0, 1)
    c2[~obs] = c[~obs]
    r = F.compare_creds(c, c2, data)
    for k in F.TOPK:
        if k > min(r["n_obs_above_unobs_a"], r["n_obs_above_unobs_b"]):
            assert np.isnan(r[f"jaccard_top{k}"])
        else:
            assert r[f"n_top{k}_a"] <= data.n[obs].sum()
    # set sizes never include unobserved edges
    for q in F.BFDR_LEVELS:
        assert r[f"n_bfdr{q:g}_a"] <= data.n[obs].sum()


def test_frozen_nuts_settings_config():
    cfg = json.loads(F.NUTS_CONFIG.read_text())
    assert cfg["settings"] == F.NUTS_SETTINGS
    assert cfg["history"] == F.NUTS_SETTINGS_HISTORY
    for fid in F.NUTS_FITS:
        for u in F.UNIVERSES:
            s = F.frozen_nuts_settings(fid, u)
            assert (s["num_chains"], s["num_warmup"], s["num_samples"]) == (4, 1000, 1000)
            assert s["target_accept"] == 0.95
    kw, dec = F.choose_nuts_settings(None, "frozen", 90.0, 0, fit_id="family_hb_nuts", universe="global")
    assert kw == dict(num_chains=4, num_warmup=1000, num_samples=1000) and dec["chosen"] == "frozen"


def test_analysis_status_labels(tmp_path):
    import pandas as pd
    assert set(F.FIRST_PASS_COLUMNS) == {
        "fit_convergence.csv", "fit_source_parameters.csv", "fit_class_weights.csv", "fit_hyperparameters.csv",
        "fit_map_optima.csv", "fit_nuts_chains.csv", "fit_map_vs_nuts.csv", "fit_estimator_agreement.csv"}
    par = ["alpha", "lr_pos", "lr_pos_marginal", "alpha_over_gamma"]
    df = pd.DataFrame(dict(universe="u", fit_id="family_hb", inference="MAP", annotator="a", family="LIT",
                           parameter=par, value=1.0, lo=np.nan, hi=np.nan, n_observed=1.0, n_reported=1.0,
                           rank=[np.nan, np.nan, 1.0, np.nan]))
    out = F.label_analysis_status("fit_source_parameters.csv", df)
    assert out["analysis_status"].tolist() == ["prespecified", "prespecified", "post_hoc", "post_hoc"]
    assert set(out["post_hoc_columns"]) == {"rank"}
    # idempotent, and the status columns never list themselves
    again = F.label_analysis_status("fit_source_parameters.csv", out)
    assert again.equals(out)
    # renamed columns are not post hoc; redefined top-k columns and new columns are
    cols = ["universe", "fit_id", "n_lbfgs_ftol_stop", "grad_norm_lbfgs", "n_grad_converged", "grad_norm"]
    assert F.post_hoc_columns("fit_convergence.csv", cols) == ["n_grad_converged"]
    cols = ["universe", "fit_a", "fit_b", "spearman_w", "jaccard_bfdr0.05", "jaccard_top100", "n_top100_a",
            "n_obs_above_unobs_a"]
    assert F.post_hoc_columns("fit_estimator_agreement.csv", cols) == ["jaccard_top100", "n_top100_a",
                                                                       "n_obs_above_unobs_a"]
    # decision log: only this module's rows are replaced; statuses are from the shared vocabulary
    p = tmp_path / "eval_analysis_decisions.csv"
    pd.DataFrame([dict(module="other", id="x", date="2026-10-08", status="post_hoc", item="i", note="n"),
                  dict(module="fit_primary", id="stale", date="2026-10-08", status="post_hoc", item="i", note="n")]
                 ).to_csv(p, index=False)
    F.record_decisions("fit_primary", F.ANALYSIS_DECISIONS, p)
    log = pd.read_csv(p)
    assert (log["module"] == "other").sum() == 1 and "stale" not in set(log["id"])
    mine = log[log["module"] == "fit_primary"]
    assert len(mine) == len(F.ANALYSIS_DECISIONS) and set(mine["status"]) <= {"post_hoc", "module_design"}


def test_build_tables_end_to_end(tmp_path, monkeypatch):
    """Tables from stored toy fits: every table has the status columns, post hoc columns exist, the
    marginal LR+ rows are post hoc and ranked, and top-k agreement is NA beyond the observed edges."""
    import pandas as pd
    data, inv, truth = _toy(N=5000)
    monkeypatch.setattr(F, "load_data", lambda universe, variant=F.VARIANT: data)
    for m in ("em", "family_hb"):
        F.run_map("toy", m, variant="v", n_restarts=2, seed=5, root=tmp_path, data=data)
        F.laplace_optima("toy", m, variant="v", root=tmp_path, data=data)
    F.run_nuts("toy", variant="v", root=tmp_path, data=data, seed=3, init="map_laplace",
               nuts_kw=dict(num_chains=2, num_warmup=40, num_samples=30))
    td = tmp_path / "tables"
    out = F.build_tables(("toy",), "v", tmp_path, td)
    assert set(out) == set(F.FIRST_PASS_COLUMNS)
    for name, df in out.items():
        assert len(df) and set(df["analysis_status"]) <= {"prespecified", "post_hoc"}
        listed = [c for c in df["post_hoc_columns"].iloc[0].split(";") if c]
        assert set(listed) <= set(df.columns) and not set(listed) & set(F.STATUS_COLUMNS)
        assert pd.read_csv(td / name).shape == df.shape
    sp = out["fit_source_parameters.csv"]
    post = sp[sp["analysis_status"] == "post_hoc"]
    assert set(post["parameter"]) == set(F.POST_HOC_PARAMETERS)
    ranked = sp[sp["parameter"] == "lr_pos_marginal"]
    assert ranked["rank"].notna().all() and sp.loc[sp["parameter"] != "lr_pos_marginal", "rank"].isna().all()
    for (_, fid), g in ranked.groupby(["universe", "fit_id"]):
        assert g.sort_values("value", ascending=False)["rank"].tolist() == sorted(g["rank"])
    # marginal LR+ = alpha / mixture of the non-T classes (MAP fit recomputed from the stored parameters)
    p = np.load(tmp_path / "toy" / "v" / "family_hb" / "params.npz")
    w, th = p["w"], p["theta"]
    ref = th[:, 0] / ((th[:, 1:] * w[None, 1:]).sum(1) / (1 - w[0]))
    got = ranked[(ranked["fit_id"] == "family_hb")].set_index("annotator")["value"].reindex(data.annotators)
    assert np.allclose(got.to_numpy(), ref)
    ag = out["fit_estimator_agreement.csv"]
    k_max = ag[["n_obs_above_unobs_a", "n_obs_above_unobs_b"]].min(axis=1)
    for k in F.TOPK:
        assert ag.loc[k_max < k, f"jaccard_top{k}"].isna().all()
        assert ag.loc[k_max >= k, f"jaccard_top{k}"].notna().all()
    cw = out["fit_class_weights.csv"]
    assert (cw["expected_edges_informative"] <= cw["n_edges_informative"] + 1e-6).all()
    assert (cw["expected_edges_observed"] <= cw["expected_edges_all"] + 1e-6).all()
    mv = out["fit_map_vs_nuts.csv"].iloc[0]
    assert mv["n_edges_cri_straddles_0.5_mean_gt_0.5"] <= mv["n_edges_cri_straddles_0.5"]
    assert mv["expected_true_observed_nuts"] <= mv["expected_true_all_nuts"] + 1e-9
    log = pd.read_csv(td / "eval_analysis_decisions.csv")
    assert set(log["module"]) == {"fit_primary"}
    # NUTS site flags (split R-hat > 1.01 or bulk ESS < 400) in the convergence and hyper-parameter tables
    dg = pd.read_csv(tmp_path / "toy" / "v" / "family_hb_nuts" / "diagnostics.csv")
    want = sorted(dg.loc[(dg.r_hat_max > F.RHAT_FLAG) | (dg.ess_bulk_min < F.ESS_BULK_FLAG), "site"])
    conv = out["fit_convergence.csv"]
    nuts = conv[conv["inference"] == "NUTS"].iloc[0]
    assert nuts["n_sites_flagged"] == len(want) and (nuts["sites_flagged"] or "") == ";".join(want)
    assert nuts["n_identified_sites_flagged"] == len([x for x in want if x in F.IDENTIFIED_SITES])
    hp = out["fit_hyperparameters.csv"]
    hn = hp[hp["inference"] == "NUTS"]
    assert len(hn) and set(hn["diag_flag"]) <= {"flagged", "ok"} and hn["r_hat_max_site"].notna().all()
    flagged = (hn["r_hat_max_site"] > F.RHAT_FLAG) | (hn["ess_bulk_min_site"] < F.ESS_BULK_FLAG)
    assert ((hn["diag_flag"] == "flagged") == flagged).all()
    for _, r in hn.iterrows():
        assert r["r_hat_max_site"] == dg.loc[dg.site == r["parameter"], "r_hat_max"].iloc[0]
    hm = hp[hp["inference"] == "MAP"]
    assert hm["r_hat_max_site"].isna().all() and (hm["diag_flag"] == "").all()
    assert {"diag_flag", "r_hat_max_site"} <= set(hp["post_hoc_columns"].iloc[0].split(";"))
    assert "sites_flagged" in conv["post_hoc_columns"].iloc[0].split(";")
    # robfix: posterior rank interval of the marginal LR+ (NUTS rows) and the value-provenance flags
    rn = ranked[ranked["inference"] == "NUTS"]
    K = len(data.annotators)
    assert len(rn) == K and rn[["rank_lo", "rank_hi"]].notna().all().all()
    assert ((1 <= rn["rank_lo"]) & (rn["rank_lo"] <= rn["rank_hi"]) & (rn["rank_hi"] <= K)).all()
    assert ranked.loc[ranked["inference"] == "MAP", ["rank_lo", "rank_hi"]].isna().all().all()
    assert sp.loc[sp["parameter"] != "lr_pos_marginal", "rank_lo"].isna().all()
    for name, df in out.items():
        assert set(F.VALUE_CHANGE_COLUMNS) <= set(df.columns), name
        assert set(F.VALUE_CHANGE_COLUMNS) <= set(df["post_hoc_columns"].iloc[0].split(";")), name


def test_site_flags_thresholds():
    import pandas as pd
    dg = pd.DataFrame(dict(site=["a", "b", "c", "d"], r_hat_max=[1.009, 1.011, 1.0, 1.02],
                           ess_bulk_min=[400.0, 5000.0, 399.9, 100.0], ess_tail_min=[1.0, 1.0, 1.0, 1.0]))
    assert F.site_flags(dg).tolist() == [False, True, True, True]


def _fake_nuts_dir(d, cred, lo, hi, w_T, num=1000, r_hat=1.01):
    """Minimal stored NUTS fit (the files nuts_rerun_check reads)."""
    import pandas as pd
    d.mkdir(parents=True)
    for k, v in (("cred", cred), ("cred_lo", lo), ("cred_hi", hi), ("cred_sd", (hi - lo) / 4)):
        np.save(d / f"{k}.npy", v)
    np.savez(d / "params.npz", w=np.array([w_T, 1 - w_T]))
    pd.DataFrame(dict(site=["w", "theta", "cred_observed", "muf_b"], r_hat_max=[1.001, 1.002, 1.003, r_hat],
                      ess_bulk_min=[900.0, 800.0, 700.0, 150.0], ess_tail_min=[950.0, 850.0, 750.0, 200.0])
                 ).to_csv(d / "diagnostics.csv", index=False)
    conv = dict(r_hat_max=r_hat, worst_site="muf_b", ess_bulk_min=150.0, ess_tail_min=200.0, n_divergent=2,
                n_chains_at_best_map_optimum=4)
    (d / "meta.json").write_text(json.dumps(dict(num_chains=4, num_warmup=num, num_samples=num, target_accept=0.95,
                                                 runtime_s=120.0, convergence=conv)))


def test_nuts_rerun_check_matches_bruteforce(tmp_path, monkeypatch):
    """First-pass vs re-run comparison: identical runs give zero change; a perturbed re-run reproduces an
    edge-level brute-force computation of the Strong-set churn and of the moved-edge counts."""
    from r2 import nuts_rerun_check as RC
    data, inv, truth = _toy(N=4000)
    monkeypatch.setattr(F, "load_data", lambda universe, variant=F.VARIANT: data)
    c1 = lcm.credibility_at(truth, data)
    rng = np.random.default_rng(1)
    obs = F.observed_patterns(data)
    c2 = np.clip(c1 + rng.normal(0, 0.03, len(c1)), 0, 1)
    c2[~obs] = c1[~obs]
    lo1, hi1 = np.clip(c1 - 0.05, 0, 1), np.clip(c1 + 0.05, 0, 1)
    lo2, hi2 = np.clip(c2 - 0.02, 0, 1), np.clip(c2 + 0.02, 0, 1)
    _fake_nuts_dir(RC.superseded_dir("toy", "v", tmp_path), c1, lo1, hi1, 0.06, num=500, r_hat=1.03)
    _fake_nuts_dir(F.fit_dir("toy", RC.FIT_ID, "v", tmp_path), c2, lo2, hi2, 0.061)
    # a universe without a kept first-pass fit is skipped
    _fake_nuts_dir(F.fit_dir("other", RC.FIT_ID, "v", tmp_path), c2, lo2, hi2, 0.061)
    df = RC.build(("toy", "other"), "v", tmp_path)
    assert df["universe"].tolist() == ["toy"] and set(df["analysis_status"]) == {"post_hoc"}
    assert (tmp_path / RC.SUPERSEDED / RC.OUT_NAME).exists()
    r = df.iloc[0]
    assert (r["num_samples_a"], r["num_samples_b"]) == (500, 1000)
    assert r["r_hat_max_a"] == 1.03 and r["r_hat_max_identified_a"] == 1.003 and r["ess_bulk_min_identified_b"] == 700.0
    # brute force over expanded edges (observed edges only)
    e1, e2, eo = c1[inv], c2[inv], obs[inv]
    e1, e2 = e1[eo], e2[eo]

    def strong(c, q=F.BFDR_LEVELS[0]):
        best = np.zeros(len(c), dtype=bool)
        for v in np.unique(c)[::-1]:                 # whole tie blocks in decreasing credibility
            s = c >= v
            if np.mean(1 - c[s]) <= q + 1e-15:
                best = s
        return best
    s1, s2 = strong(e1), strong(e2)
    assert r["n_bfdr0.05_a"] == s1.sum() and r["n_bfdr0.05_b"] == s2.sum()
    assert r["n_strong_left"] == (s1 & ~s2).sum() and r["n_strong_entered"] == (s2 & ~s1).sum()
    assert np.isclose(r["jaccard_bfdr0.05"], (s1 & s2).sum() / (s1 | s2).sum())
    d = np.abs(e1 - e2)
    assert r["n_edges_abs_diff_gt_0.01"] == (d > 0.01).sum() and r["n_edges_abs_diff_gt_0.05"] == (d > 0.05).sum()
    assert np.isclose(r["max_abs_diff"], d.max()) and np.isclose(r["max_abs_diff_strong_union"], d[s1 | s2].max())
    assert np.isclose(r["frac_edges_b_mean_inside_a_cri"], np.mean(np.abs(e2 - e1) <= 0.05 + 1e-12), atol=2e-3)
    assert np.isclose(r["expected_true_observed_b"], e2.sum()) and r["w_T_mean_b"] == 0.061
    # identical runs: no change
    same = RC.compare_runs(RC.read_nuts_dir(F.fit_dir("toy", RC.FIT_ID, "v", tmp_path)),
                           RC.read_nuts_dir(F.fit_dir("toy", RC.FIT_ID, "v", tmp_path)), data)
    assert same["max_abs_diff"] == 0 and same["jaccard_bfdr0.05"] == 1 and same["n_strong_left"] == 0
    assert same["frac_edges_b_mean_inside_a_cri"] == 1


def test_draw_ranks_and_value_provenance(tmp_path):
    """robfix: per-draw ranks equal the table's rank rule; fit provenance is read from meta.json."""
    import pandas as pd
    v = np.array([[3.0, 1.0, 2.0], [1.0, 1.0, 5.0]])
    assert F.draw_ranks(v).tolist() == [[1.0, 3.0, 2.0], [2.0, 2.0, 1.0]]

    def meta(fid, **kw):
        d = tmp_path / "u" / "v" / fid
        d.mkdir(parents=True, exist_ok=True)
        (d / "meta.json").write_text(json.dumps(kw))

    meta("em", inference="MAP", polish={"status": "ok"})
    meta("old_map", inference="MAP")
    meta("family_hb_nuts", inference="NUTS", settings_decision={"requested": "frozen"}, num_chains=4,
         num_warmup=1000, num_samples=1000)
    meta("family_hb_randinit_nuts", inference="NUTS", settings_decision={"requested": "auto"},
         provenance_note="first-pass run; relabelled by src/r2/patch_randinit_layout.py")
    meta("first_nuts", inference="NUTS", settings_decision={"requested": "auto"})
    sup = tmp_path / "_superseded_firstpass" / "u__v__family_hb_nuts"
    sup.mkdir(parents=True)
    (sup / "meta.json").write_text(json.dumps(dict(num_chains=4, num_warmup=500, num_samples=500)))
    P = lambda f: F.fit_value_provenance("u", f, "v", tmp_path)  # noqa: E731
    assert P("em")[0] and "Newton-polished" in P("em")[1]
    assert P("old_map") == (False, "first-pass MAP fit")
    assert P("family_hb_nuts")[0] and "4 x 500/500 -> 4 x 1000/1000" in P("family_hb_nuts")[1]
    assert P("family_hb_randinit_nuts")[0] and P("first_nuts") == (False, "first-pass NUTS fit")
    assert P("missing") == (False, "")
    df = pd.DataFrame(dict(universe="u", fit_a=["em", "old_map"], fit_b=["old_map", "first_nuts"]))
    out = F.add_value_provenance("fit_estimator_agreement.csv", df, "v", tmp_path)
    assert out["values_changed_post_hoc"].tolist() == [True, False]
    assert out["value_change_note"].iloc[0].startswith("em: MAP refit")
    one = F.add_value_provenance("fit_convergence.csv", pd.DataFrame(dict(universe=["u"], fit_id=["family_hb_nuts"])),
                                 "v", tmp_path)
    assert one["values_changed_post_hoc"].tolist() == [True]
    mv = F.add_value_provenance("fit_map_vs_nuts.csv", pd.DataFrame(dict(universe=["u"])), "v", tmp_path)
    assert mv["values_changed_post_hoc"].tolist() == [True] and "family_hb_nuts:" in mv["value_change_note"].iloc[0]
    assert F.add_value_provenance("fit_convergence.csv", out, "v", tmp_path).shape[1] == out.shape[1]  # idempotent
