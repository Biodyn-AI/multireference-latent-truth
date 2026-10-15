"""Tests of the compute-scaling module (src/r2/compute_scaling.py).

Run: .venv/bin/python -m pytest subproject_24_multireference_latent_truth_model/src/r2/tests/test_compute_scaling.py -q
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from r2 import compute_scaling as CS  # noqa: E402
from r2 import evidence as E  # noqa: E402


def _random_evidence(n, K, seed=0, p_obs=0.4, p_pos=0.2):
    rng = np.random.default_rng(seed)
    obs = rng.random((n, K)) < p_obs
    pos = rng.random((n, K)) < p_pos
    return np.where(obs, pos.astype(np.int8), np.int8(-1)).astype(np.int8)


def test_encode_matches_evidence_codes_and_roundtrips():
    M = _random_evidence(3000, 21, seed=1)
    words = CS.encode(M)
    assert words.shape == (3000, 1)
    assert np.array_equal(words[:, 0], E._codes(M, 3 ** np.arange(21, dtype=np.int64)))
    assert np.array_equal(CS.decode(words, 21), M)
    M42 = _random_evidence(3000, 42, seed=2)
    w42 = CS.encode(M42)
    assert w42.shape == (3000, 2)
    assert np.array_equal(CS.decode(w42, 42), M42)


def test_rss_units():
    assert CS.maxrss_mib(2**20, "Darwin") == 1.0      # bytes on macOS
    assert CS.maxrss_mib(1024, "Linux") == 1.0        # KiB on Linux


@pytest.mark.parametrize("K", [7, 42])
def test_streaming_compression_equals_dense(K):
    M = _random_evidence(6000, K, seed=K, p_obs=0.3, p_pos=0.3)
    sc = CS.StreamingCompressor(len(M), K)
    cuts = [0, 1, 500, 2311, 2312, 6000]
    for a, b in zip(cuts[:-1], cuts[1:]):
        sc.add(M[a:b])
    R, n, ep = sc.finish()
    assert np.array_equal(R[ep], M)                       # exact edge -> pattern index
    assert int(n.sum()) == len(M) and len(np.unique(R, axis=0)) == len(R)
    Rd, nd, epd = CS.compress_dense(M)                    # = lcm.PatternData.from_matrix
    assert len(Rd) == len(R)
    a = {tuple(r): c for r, c in zip(R.tolist(), n.tolist())}
    b = {tuple(r): c for r, c in zip(Rd.tolist(), nd.tolist())}
    assert a == b
    from r2 import lcm
    data, inv = lcm.PatternData.from_matrix(M, ["LIT"] * K, [f"a{j}" for j in range(K)])
    assert np.array_equal(data.R, Rd) and np.array_equal(inv, epd)


def _toy_template(seed=0):
    rng = np.random.default_rng(seed)
    ann = ["L1", "L2", "L3", "M1"]
    fam = ["LIT", "LIT", "LIT", "MOTIF"]
    T0, G0 = 7, 9
    tpl = dict(cov=rng.random((T0, 4)) < 0.6, node=rng.random((G0, 4)) < 0.5,
               w=np.array([0.2, 0.6, 0.15, 0.05]),
               theta=np.array([[0.6, 0.05, 0.4, 0.05], [0.5, 0.02, 0.5, 0.02],
                               [0.7, 0.1, 0.3, 0.1], [0.4, 0.05, 0.05, 0.6]]),
               c_rate=np.full(4, 0.5), o_rate=np.full(4, 0.5))
    tpl["w_by_depth"] = np.tile(tpl["w"], (4, 1))
    tpl["w_by_depth"][3] = [0.6, 0.2, 0.15, 0.05]
    tpl["w_by_depth_k10"] = tpl["w_by_depth"]
    info = dict(T0=T0, G0=G0, annotators=ann, family=fam, pi_c=0.7, k10_annotators=ann)
    return tpl, info


@pytest.mark.parametrize("K", [4, 8])
def test_generator_observation_structure(K):
    tpl, info = _toy_template()
    gen = CS.Generator(CS.Config("template", "real", K), tpl, info)
    gen.tfs_per_chunk = 2                                  # several chunks
    M = gen.dense()
    assert M.shape == (7 * 9, K)
    blk = M.reshape(7, 9, K)
    fam = np.asarray(gen.family)
    lit_rep = (blk[:, :, fam == "LIT"] == 1).any(2)
    for j in range(K):
        obs = blk[:, :, j] >= 0
        if fam[j] == "LIT":                                # product structure: TF coverage x node set
            assert np.array_equal(obs, gen.cov[:, j][:, None] & gen.node[:, j][None, :])
        else:                                              # MOTIF: covered TF and a LIT-reported edge
            assert not (obs & ~lit_rep).any()
            assert not (obs & ~gen.cov[:, j][:, None]).any()
    # determinism and chunk-wise streaming == dense
    gen2 = CS.Generator(CS.Config("template", "real", K), tpl, info)
    gen2.tfs_per_chunk = 2
    assert np.array_equal(gen2.dense(), M)
    sc = CS.StreamingCompressor(gen2.N, K)
    for _, _, b in gen2.chunks():
        sc.add(b)
    R, n, ep = sc.finish()
    assert np.array_equal(R[ep], M)


def test_pattern_count_bound():
    """P <= (number of observation masks) + (edges with >= 1 report), for sparse and dense reports."""
    for p_pos in (0.01, 0.5):
        M = _random_evidence(5000, 9, seed=3, p_obs=0.3, p_pos=p_pos)
        R, n, _ = CS.compress_dense(M)
        st = CS.pattern_stats(R, n)
        assert st["P"] == len(R) <= st["n_masks"] + st["n_edges_any_report"]
        assert st["n_edges_any_report"] == int((M == 1).any(1).sum())
        assert st["n_masks"] == len(np.unique(M >= 0, axis=0))


def test_linear_fit():
    x = np.array([1.0, 2.0, 3.0, 4.0, np.nan])
    f = CS._linear_fit(x, 5.0 + 2.0 * x)
    assert f["n"] == 4 and abs(f["slope"] - 2) < 1e-9 and abs(f["intercept"] - 5) < 1e-9


def test_stress_generator_density():
    tpl, info = _toy_template()
    gen = CS.Generator(CS.Config("stress", "real", 4), tpl, info)
    M = gen.dense()
    assert abs((M >= 0).mean() - CS.STRESS_OBS) < 0.15


def test_config_shapes():
    T0, G0 = 935, 9963
    for size, N in (("1e5", 1e5), ("1e6", 1e6), ("1e7", 1e7)):
        T, G = CS.Config("template", size, 21).shape(T0, G0)
        assert abs(T * G / N - 1) < 0.01
    T, G = CS.Config("template", "10xreal", 21).shape(T0, G0)
    assert abs(T * G / (T0 * G0) - 10) < 0.01
    assert CS.config_from_cid("independent_1e6_K42") == CS.Config("independent", "1e6", 42)


def test_loglog_fit_recovers_slope():
    x = np.array([1e3, 1e4, 1e5, 1e6])
    f = CS._loglog_fit(x, 3e-7 * x)
    assert abs(f["slope"] - 1) < 1e-9 and f["r2"] > 0.999999
    # missing / non-positive values are dropped, not propagated
    g = CS._loglog_fit(np.append(x, [1e7, 1e8]), np.append(3e-7 * x, [np.nan, 0.0]))
    assert g["n"] == 4 and abs(g["slope"] - 1) < 1e-9
    assert np.isnan(CS._loglog_fit([1.0, 2.0], [1.0, 2.0])["slope"])       # fewer than 3 points


def test_same_patterns_ignores_row_order_only():
    M = _random_evidence(4000, 42, seed=5, p_obs=0.2, p_pos=0.3)
    sc = CS.StreamingCompressor(len(M), 42)
    sc.add(M[:1500])
    sc.add(M[1500:])
    R, n, _ = sc.finish()
    Rd, nd, _ = CS.compress_dense(M)                      # different row order
    assert not np.array_equal(R, Rd) and CS.same_patterns(R, n, Rd, nd)
    n2 = n.copy()
    n2[0] += 1
    assert not CS.same_patterns(R, n2, Rd, nd)            # counts matter
    R2 = R.copy()
    R2[0, 0] = 1 if R2[0, 0] != 1 else 0
    assert not CS.same_patterns(R2, n, Rd, nd)            # rows matter
    assert not CS.same_patterns(R[:-1], n[:-1], Rd, nd)


def test_session_directories(monkeypatch, tmp_path):
    monkeypatch.setattr(CS, "RUN_DIR", tmp_path / "runs")
    assert CS.run_dir("a") == tmp_path / "runs" and CS.run_dir("b") == tmp_path / "runs_b"
    assert CS._run_json("map", "x", "b") == tmp_path / "runs_b" / "map" / "x.json"
    assert CS.log_dir("a").name == "logs" and CS.log_dir("b").name == "logs_b"
    with pytest.raises(ValueError):
        CS.run_dir("z")


def _toy_lcm_data(N=4000, seed=0):
    from r2 import lcm
    rng = np.random.default_rng(seed)
    fam = ["LIT"] * 3 + ["MOTIF"] * 2
    theta = np.array([[0.5, 0.01, 0.4, 0.01], [0.4, 0.02, 0.5, 0.02], [0.6, 0.01, 0.3, 0.01],
                      [0.5, 0.05, 0.05, 0.6], [0.4, 0.04, 0.04, 0.5]])
    w = np.array([0.08, 0.8, 0.08, 0.04])
    z = rng.choice(4, size=N, p=w)
    rep = (rng.random((N, 5)) < theta.T[z]).astype(np.int8)
    M = np.where(rng.random((N, 5)) < 0.6, rep, -1).astype(np.int8)
    data, _ = lcm.PatternData.from_matrix(M, fam, [f"a{j}" for j in range(5)])
    return data


def test_timed_map_reproduces_lcm_fit_map():
    from r2 import lcm
    data = _toy_lcm_data()
    ref = lcm.fit_map(data, "family_hb", n_restarts=2, seed=3)
    got = CS.timed_map(data, "family_hb", n_restarts=2, seed=3, n_bench=2, counts_factor=2.0)
    for a, b in zip(ref.restarts, got["restarts"]):
        assert a["n_iter"] == b["n_iter"]
        assert np.isclose(a["log_post"], b["log_post"], rtol=0, atol=1e-9)
    assert got["t_map_10_status"] == "measured" and got["n_restarts_run"] == 2
    assert got["S"] == 4 and got["t_vg_s"] > 0 and got["t_vg_counts_x_factor_s"] > 0


def test_timed_map_budget_extrapolates():
    data = _toy_lcm_data(N=2000, seed=1)
    got = CS.timed_map(data, "em", n_restarts=5, seed=1, budget_s=0.0, n_bench=1, counts_factor=None)
    assert got["n_restarts_run"] == 2 and got["t_map_10_status"] == "extrapolated"


def test_timed_map_eval_only_projects():
    data = _toy_lcm_data(N=2000, seed=1)
    got = CS.timed_map(data, "em", n_restarts=10, seed=1, n_bench=3, counts_factor=3.0, nfev_guess=200.0,
                       eval_only=True)
    assert got["n_restarts_run"] == 0 and got["t_map_10_status"] == "projected" and got["eval_only"]
    assert got["restarts"] == [] and 0 < got["t_vg_min_s"] <= got["t_vg_s"]
    assert got["counts_invariance_ratio_paired"] > 0
    expect = got["t_build_init_s"] + got["t_compile_s"] + 10 * 200.0 * got["t_eval_lbfgs_s"]
    assert np.isclose(got["t_map_10_s"], expect)
    with pytest.raises(ValueError):
        CS.timed_map(data, "em", eval_only=True)


def test_timed_nuts_hides_credibility():
    data = _toy_lcm_data(N=2000, seed=2)
    out = CS.timed_nuts(data, "family_hb", chains=2, warmup=15, samples=10, probe=False)
    assert out["steps_total"] >= 2 * 25 and out["n_draws"] == 20
    assert out["cred_mean_finite"] and out["t_per_step_s"] > 0
    assert np.isfinite(out["r_hat_max"]) and out["ess_bulk_min"] > 0
    assert sum(out["steps_warmup_by_chain"]) + sum(out["steps_sampling_by_chain"]) == out["steps_total"]
    assert "t_probe_s" not in out
    # the compile probe does not change the timed run (same seed -> same leapfrog steps)
    out2 = CS.timed_nuts(data, "family_hb", chains=2, warmup=15, samples=10, probe=True)
    assert out2["steps_total"] == out["steps_total"] and out2["probe_kind"] == "main_like_warm"
    assert out2["t_probe_s"] > 0 and out2["t_probe_cold_s"] > 0 and out2["probe_steps"] >= 4


def test_nuts_diagnostics_excludes_deterministic_sites():
    rng = np.random.default_rng(0)
    s = {"w": rng.normal(size=(2, 200, 3)), "theta": np.zeros((2, 200, 4, 2)),
         "cred": np.zeros((2, 200, 5))}
    d = CS.nuts_diagnostics(s)
    assert 0.98 < d["r_hat_max"] < 1.05 and d["ess_bulk_min"] > 100   # theta / cred (constant) ignored
    s["w"][1] += 5.0                                                     # chains disagree
    assert CS.nuts_diagnostics(s)["r_hat_max"] > 1.5


def test_nuts_projection_uses_largest_P_reference():
    done = [dict(cid="a", K=21, P=100, steps_per_iter_mean=50.0, t_per_step_s=2e-4, t_vg_map_s=1e-3),
            dict(cid="b", K=21, P=5000, steps_per_iter_mean=120.0, t_per_step_s=3e-3, t_vg_map_s=2e-3),
            dict(cid="c", K=42, P=9000, steps_per_iter_mean=300.0, t_per_step_s=1.0, t_vg_map_s=1.0)]
    m = dict(K=21, t_vg_s=0.01, t_compile_vg_s=1.0)
    pr = CS.nuts_projection("x", done, m)
    assert pr["steps_per_iter_assumed"] == 120.0 and np.isclose(pr["step_over_vg_ratio"], 1.5)
    iters = CS.NUTS_CHAINS * (CS.NUTS_WARMUP + CS.NUTS_SAMPLES)
    assert np.isclose(pr["t_nuts_projected_s"], iters * 120.0 * 0.01 * 1.5 + 2.0)
    pr0 = CS.nuts_projection("x", [], m)                     # no reference: default steps, ratio 1
    assert pr0["steps_per_iter_assumed"] == CS.NUTS_DEFAULT_STEPS and pr0["step_over_vg_ratio"] == 1.0
    prk = CS.nuts_projection("x", done, m, steps_known=(700.0, "session a"))   # steps of an earlier run
    assert prk["steps_per_iter_assumed"] == 700.0 and prk["steps_source"] == "session a"
    assert np.isclose(prk["t_nuts_projected_s"], iters * 700.0 * 0.01 * 1.5 + 2.0)


def test_spawn_timeout(monkeypatch, tmp_path):
    # a worker that runs past the timeout returns -9 and the log records it
    exe = tmp_path / "slow.sh"
    exe.write_text("#!/bin/sh\nsleep 5\n")
    exe.chmod(0o755)
    monkeypatch.setattr(CS.sys, "executable", str(exe))
    log = tmp_path / "x.log"
    assert CS._spawn([], log, timeout_s=0.5) == -9
    assert "timeout" in log.read_text()


@pytest.mark.skipif(not CS.TEMPLATE_NPZ.exists(), reason="template not built")
def test_stored_template():
    tpl, info = CS.load_template()
    assert tpl["cov"].shape == (info["T0"], info["K"]) and tpl["node"].shape == (info["G0"], info["K"])
    assert 0 < info["pi_c"] < 1
    assert np.allclose(tpl["w_by_depth"].sum(1), 1) and np.allclose(tpl["w_by_depth_k10"].sum(1), 1)
    assert len(info["k10_annotators"]) == 10
    assert info["real"]["P"] == len(E.load_patterns("global")[0])


def test_build_tables_status_handling(monkeypatch, tmp_path):
    """Tables from fake measurement records: measured / projected / timeout NUTS rows are
    reported correctly and every table carries analysis_status."""
    import json
    run = tmp_path / "runs"
    for st in ("compress", "map", "nuts"):
        (run / st).mkdir(parents=True)
    K, ann = 2, ["a1", "a2"]
    info = dict(K=K, T0=3, G0=4, annotators=ann, family=["LIT", "LIT"], es_coverage=[0.5, 0.5],
                es_pos_rate=[0.1, 0.1],
                real=dict(N=12, P=5, n_masks=3, frac_edges_unobserved=0.2, mean_annotators_observing=1.0,
                          pos_rate_observed_cells=0.1, n_edges_any_report=2, coverage=[0.5, 0.5],
                          pos_rate=[0.1, 0.1]))
    tj = tmp_path / "template.json"
    tj.write_text(json.dumps(info))
    monkeypatch.setattr(CS, "RUN_DIR", run)
    monkeypatch.setattr(CS, "TEMPLATE_JSON", tj)
    cids = ["template_1e5_K21", "template_1e6_K21", "template_1e7_K21", "template_real_K21"]
    for i, cid in enumerate(cids):
        P = 10 ** (i + 2)
        (run / "compress" / f"{cid}.json").write_text(json.dumps(dict(
            cid=cid, method="chunked_stream", generator="template", size=cid.split("_")[1], K=21, N=P * 100,
            P=P, t_compress_s=0.1 * (i + 1), peak_rss_mib=100.0 + i, coverage=[0.4, 0.6], pos_rate=[0.1, 0.2],
            mean_annotators_observing=1.0, pos_rate_observed_cells=0.1, n_masks=3, frac_edges_unobserved=0.2,
            n_edges_any_report=2, load1_mean=5.0)))
        (run / "map" / f"{cid}.json").write_text(json.dumps(dict(
            cid=cid, P=P, K=21, S=4, N=P * 100, t_vg_s=1e-6 * P, t_vg_counts_x_factor_s=1.0, t_vg_repeat_s=1.0,
            t_map_10_s=1e-3 * P, t_map_10_status="measured", t_per_lbfgs_eval_s=1e-6 * P, peak_rss_mib=200.0,
            load1_mean=5.0, restarts=[dict(restart=0, log_post=-1.0, nfev=10)])))
    nuts = {cids[0]: dict(status="measured", t_nuts_s=10.0, steps_per_iter_mean=100.0, t_per_step_s=1e-4,
                          ess_bulk_min=50.0, chains=2, warmup=300, samples=300),
            cids[1]: dict(status="projected", t_nuts_projected_s=5000.0, steps_per_iter_assumed=100.0,
                          step_over_vg_ratio=1.0),
            cids[2]: dict(status="timeout", t_nuts_lower_bound_s=5400.0, t_nuts_projected_s=1000.0,
                          steps_per_iter_assumed=100.0, step_over_vg_ratio=1.0)}
    for i, (cid, r) in enumerate(nuts.items()):
        (run / "nuts" / f"{cid}.json").write_text(json.dumps(dict(cid=cid, P=10 ** (i + 2), K=21, N=1, **r)))
    out = CS.build_tables(tables_dir=tmp_path / "tables")
    nt = out["scale_nuts.csv"].set_index("cid")
    assert nt.loc[cids[0], "t_nuts_2x300_s"] == 10.0 and nt.loc[cids[0], "t_nuts_per_min_ess_bulk_s"] == 0.2
    assert nt.loc[cids[1], "t_nuts_2x300_s"] == 5000.0
    assert np.isnan(nt.loc[cids[2], "t_nuts_2x300_s"]) and nt.loc[cids[2], "t_nuts_lower_bound_s"] == 5400.0
    assert np.isclose(nt.loc[cids[0], "t_nuts_4x1000_projected_s"], 4 * 2000 * 100 * 1e-4)
    # projection at the tree-depth cap (1023 steps/iteration), projected rows only: t_vg = 1e-6 * P, P = 1000
    assert np.isclose(nt.loc[cids[1], "t_nuts_projected_cap_s"], 1200 * 1023 * 1e-3 * 1.0)
    assert np.isnan(nt.loc[cids[0], "t_nuts_projected_cap_s"])
    for name, df in out.items():
        assert (tmp_path / "tables" / name).exists()
        assert "analysis_status" in df.columns and set(df["analysis_status"]) <= {"prespecified", "post_hoc"}, name
    fits = out["scale_fits.csv"]
    row = fits[(fits["quantity"] == "value+gradient time vs P*K*S") & (fits["subset"] == "all")].iloc[0]
    assert abs(row["slope"] - 1) < 1e-9                                   # t_vg = 1e-6 P


def _fake_session(run, cids, t_scale, load, session, restarts=True, nuts_steps=1000):
    import json
    for st in ("compress", "map", "nuts"):
        (run / st).mkdir(parents=True, exist_ok=True)
    for i, cid in enumerate(cids):
        P = 10 ** (i + 2)
        (run / "compress" / f"{cid}.json").write_text(json.dumps(dict(
            cid=cid, session=session, method="chunked_stream", K=21, N=P * 100, P=P, t_compress_s=0.1 * t_scale,
            peak_rss_mib=100.0 + load, coverage=[0.4, 0.6], pos_rate=[0.1, 0.2], mean_annotators_observing=1.0,
            pos_rate_observed_cells=0.1, n_masks=3, frac_edges_unobserved=0.2, n_edges_any_report=2,
            load1_mean=load, patterns_identical=True)))
        rs = [dict(restart=r, log_post=-1.0 - r, nfev=10 + r, n_iter=5 + r, t_restart_s=t_scale) for r in range(3)]
        full = restarts or i == 0
        (run / "map" / f"{cid}.json").write_text(json.dumps(dict(
            cid=cid, session=session, P=P, K=21, S=4, N=P * 100, n_params=84, t_vg_s=1e-6 * P * t_scale,
            t_vg_counts_x_factor_s=1.0, t_vg_repeat_s=1.0, t_build_init_s=1.0, t_compile_s=1.0,
            t_restart_mean_s=t_scale, t_map_10_s=10.0 * t_scale, n_restarts_run=3 if full else 0,
            t_map_10_status="measured" if full else "projected", nfev_mean=11.0 if full else None,
            t_per_lbfgs_eval_s=1e-6 * P * t_scale if full else None, t_restarts_wall_s=3 * t_scale if full else None,
            peak_rss_mib=200.0, load1_mean=load, restarts=rs if full else [])))
    (run / "nuts" / f"{cids[0]}.json").write_text(json.dumps(dict(
        cid=cids[0], session=session, P=100, K=21, N=10000, status="measured", t_nuts_s=10.0 * t_scale + 2.0,
        t_probe_s=2.0, steps_total=nuts_steps, steps_per_iter_mean=nuts_steps / 1200, t_per_step_s=0.01,
        ess_bulk_min=50.0, r_hat_max=1.01, chains=2, warmup=300, samples=300, peak_rss_mib=700.0,
        load1_mean=load)))


def test_build_tables_two_sessions(monkeypatch, tmp_path):
    """Replicate session: deterministic quantities compared, the summary reports the most direct
    and then the fastest record, session-a NUTS records get the old probe kind."""
    import json
    info = dict(K=2, T0=3, G0=4, annotators=["a1", "a2"], family=["LIT", "LIT"], es_coverage=[0.5, 0.5],
                es_pos_rate=[0.1, 0.1],
                real=dict(N=12, P=5, n_masks=3, frac_edges_unobserved=0.2, mean_annotators_observing=1.0,
                          pos_rate_observed_cells=0.1, n_edges_any_report=2, coverage=[0.5, 0.5],
                          pos_rate=[0.1, 0.1]))
    tj = tmp_path / "template.json"
    tj.write_text(json.dumps(info))
    monkeypatch.setattr(CS, "RUN_DIR", tmp_path / "runs")
    monkeypatch.setattr(CS, "TEMPLATE_JSON", tj)
    cids = ["template_1e5_K21", "template_1e6_K21", "template_1e7_K21"]
    _fake_session(tmp_path / "runs", cids, t_scale=4.0, load=100.0, session="a")
    # session b: twice as fast; restarts only for the first configuration; one NUTS step count differs
    _fake_session(tmp_path / "runs_b", cids, t_scale=2.0, load=10.0, session="b", restarts=False, nuts_steps=1001)
    out = CS.build_tables(tables_dir=tmp_path / "tables")
    assert len(out["scale_generator_check.csv"]) == 0             # no real-size configuration in this fake grid
    for name, df in out.items():
        if len(df):
            assert "analysis_status" in df.columns and set(df["analysis_status"]) <= {"prespecified", "post_hoc"}, name
    assert set(out["scale_map.csv"]["session"]) == {"a", "b"} and len(out["scale_map.csv"]) == 6
    rep = out["scale_replicate.csv"]
    cr = rep[rep["stage"] == "compress"]
    assert len(cr) == 3 and cr["deterministic_identical"].all() and np.allclose(cr["ratio_a_over_b"], 2.0)
    mr = rep[rep["stage"] == "map restarts"]
    assert list(mr["cid"]) == [cids[0]] and bool(mr["deterministic_identical"].iloc[0])
    assert mr["n_common_restarts"].iloc[0] == 3 and mr["max_abs_diff_log_post"].iloc[0] == 0.0
    assert len(rep[rep["stage"] == "map evaluation"]) == 3
    nr = rep[rep["stage"] == "nuts"].iloc[0]
    assert not nr["deterministic_identical"] and nr["value_a"] == 1000 and nr["value_b"] == 1001
    assert np.isclose(nr["time_a_s"], 40.0) and np.isclose(nr["time_b_s"], 20.0)     # probe subtracted
    nt = out["scale_nuts.csv"].set_index(["cid", "session"])
    assert nt.loc[(cids[0], "a"), "probe_kind"] == "single_chain_cold"
    assert np.isclose(nt.loc[(cids[0], "b"), "t_per_step_excl_compile_s"], 20.0 / 1001)
    sm = out["scale_summary.csv"].set_index("cid")
    assert (sm["compress_session"] == "b").all() and (sm["t_vg_session"] == "b").all()
    assert sm.loc[cids[0], "map_10_session"] == "b" and sm.loc[cids[0], "map_10_s"] == 20.0   # both measured: faster
    # measured in a beats the faster projection of b
    assert sm.loc[cids[1], "map_10_session"] == "a" and sm.loc[cids[1], "map_10_status"] == "measured"
    assert sm.loc[cids[0], "nuts_session"] == "b" and sm.loc[cids[0], "nuts_status"] == "measured"
    # peak RSS: the largest over the sessions (resident memory is a lower bound under memory pressure)
    assert (sm["compress_peak_rss_session"] == "a").all() and (sm["compress_peak_rss_mib"] == 200.0).all()
    # rows built from replicate-session records are post hoc; primary-session rows stay prespecified
    assert (sm["analysis_status"] == "post_hoc").all() and (rep["analysis_status"] == "post_hoc").all()
    for name in ("scale_compression.csv", "scale_map.csv", "scale_map_restarts.csv", "scale_nuts.csv",
                 "scale_fits.csv"):
        df = out[name]
        assert (df.loc[df["session"] == "b", "analysis_status"] == "post_hoc").all(), name
        assert (df.loc[df["session"] != "b", "analysis_status"] == "prespecified").all(), name
    envs = out["scale_environment.csv"].set_index("key")["analysis_status"]
    assert envs["session b"] == "post_hoc" and envs["session a"] == "prespecified" and envs["cpu"] == "prespecified"
    env = out["scale_environment.csv"].set_index("key")["value"]
    assert env["session b: measurement records"] == "7" and "10.0" in env[
        "session b: 1-min load average, per-run means (min / median / max)"]
    fits = out["scale_fits.csv"]
    assert set(fits["session"]) == {"a", "b"}
    row = fits[(fits["quantity"] == "value+gradient time vs P*K*S") & (fits["session"] == "b")].iloc[0]
    assert abs(row["slope"] - 1) < 1e-9


def test_replicate_rules(monkeypatch, tmp_path):
    import json
    monkeypatch.setattr(CS, "RUN_DIR", tmp_path / "runs")
    (tmp_path / "runs" / "nuts").mkdir(parents=True)
    recs = {"template_1e5_K21": dict(status="measured", t_nuts_s=CS.NUTS_BUDGET_S - 1),
            "template_1e6_K21": dict(status="measured", t_nuts_s=CS.NUTS_BUDGET_S + 1),
            "template_1e7_K21": dict(status="projected", t_nuts_projected_s=10.0)}
    for cid, r in recs.items():
        (tmp_path / "runs" / "nuts" / f"{cid}.json").write_text(json.dumps(dict(cid=cid, **r)))
    assert CS.replicate_nuts_set() == {"template_1e5_K21"}


def test_replicate_nuts_projection_uses_primary_reference(monkeypatch, tmp_path):
    """Session-b projections take the largest measured same-K run of either session as reference
    and the primary session's step count of the same configuration where it exists; nothing
    outside the replicate set is run."""
    import json
    monkeypatch.setattr(CS, "RUN_DIR", tmp_path / "runs")
    for sess in ("a", "b"):
        for st in ("compress", "map", "nuts"):
            CS._run_json(st, "x", sess).parent.mkdir(parents=True, exist_ok=True)
    cids = {"template_1e5_K21": 100, "template_1e6_K21": 1000, "template_1e7_K21": 10000}
    for cid, P in cids.items():
        CS._run_json("compress", cid, "a").write_text(json.dumps(dict(cid=cid, P=P)))
        for sess, scale in (("a", 2.0), ("b", 1.0)):
            CS._run_json("map", cid, sess).write_text(json.dumps(dict(
                cid=cid, P=P, K=21, N=P * 100, S=4, t_vg_s=1e-6 * P * scale, t_compile_vg_s=0.5)))
    slow = CS.NUTS_BUDGET_S + 1.0                       # over budget in session a -> not replicated
    CS._run_json("nuts", "template_1e5_K21", "a").write_text(json.dumps(dict(
        cid="template_1e5_K21", P=100, K=21, status="measured", t_nuts_s=slow, steps_per_iter_mean=200.0,
        t_per_step_s=4e-4)))                            # ratio 4e-4 / 2e-4 = 2
    CS._run_json("nuts", "template_1e6_K21", "a").write_text(json.dumps(dict(
        cid="template_1e6_K21", P=1000, K=21, status="measured", t_nuts_s=slow, steps_per_iter_mean=600.0,
        t_per_step_s=6e-3)))                            # largest measured P: ratio 6e-3 / 2e-3 = 3
    CS._run_json("nuts", "template_1e7_K21", "a").write_text(json.dumps(dict(
        cid="template_1e7_K21", P=10000, K=21, status="projected", t_nuts_projected_s=1e5)))

    def no_spawn(*a, **k):
        raise AssertionError("no worker may be started")
    monkeypatch.setattr(CS, "_spawn", no_spawn)
    assert CS.replicate_nuts_set() == set()
    CS.run_stage_nuts(session="b")
    iters = CS.NUTS_CHAINS * (CS.NUTS_WARMUP + CS.NUTS_SAMPLES)
    r = {c: json.loads(CS._run_json("nuts", c, "b").read_text()) for c in cids}
    assert all(v["status"] == "projected" and v["session"] == "b" for v in r.values())
    # never measured: steps and ratio of the largest measured run (session a, P = 1000)
    assert r["template_1e7_K21"]["steps_per_iter_assumed"] == 600.0
    assert np.isclose(r["template_1e7_K21"]["t_nuts_projected_s"], iters * 600.0 * 1e-2 * 3.0 + 1.0)
    # measured in session a: its own step count, ratio of the largest measured run
    assert r["template_1e5_K21"]["steps_per_iter_assumed"] == 200.0
    assert "session a" in r["template_1e5_K21"]["steps_source"]
    assert np.isclose(r["template_1e5_K21"]["t_nuts_projected_s"], iters * 200.0 * 1e-4 * 3.0 + 1.0)
