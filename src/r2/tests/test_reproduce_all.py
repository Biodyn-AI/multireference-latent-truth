"""Tests of the pipeline entry point ``src/r2/reproduce_all.py`` (stage registry, measurement,
timing table, manifest, requirements).  Nothing here runs an analysis stage."""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from r2 import reproduce_all as RA  # noqa: E402

PY = sys.executable


# ----------------------------------------------------------------------------- registry
def test_stage_names_unique_and_steps_exist():
    names = [s.name for s in RA.STAGES]
    assert len(names) == len(set(names))
    for s in RA.STAGES:
        assert s.layer in {"data", "model", "evaluation", "assets"}
        assert s.steps_for(False), f"{s.name}: no full step"
        assert len({st.name for st in s.steps}) == len(s.steps), s.name
        for st in s.steps:
            scripts = [a for a in st.argv if a.endswith((".py", ".sh"))]
            assert scripts, (s.name, st.name)
            assert all(Path(a).exists() for a in scripts), (s.name, st.name, scripts)


def test_every_stage_with_tables_has_a_tables_only_path():
    for s in RA.STAGES:
        if s.outputs:
            assert s.steps_for(True), f"{s.name} writes tables but has no tables-only step"
    # model stages that only fit (their tables come from later stages) have none
    assert not RA.STAGE_BY_NAME["obs_models"].steps_for(True)
    assert not RA.STAGE_BY_NAME["posterior_modes"].steps_for(True)


def test_each_table_has_one_owner_and_none_is_undeclared():
    owners: dict[str, list[str]] = {}
    for s in RA.STAGES:
        for t in s.outputs:
            owners.setdefault(t, []).append(s.name)
    assert {t: o for t, o in owners.items() if len(o) > 1} == {}
    assert RA.unclaimed_tables() == []
    # the two modules that used to write the same file now write different tables
    assert "fit_nuts_chains" in RA.STAGE_BY_NAME["fit_primary"].outputs
    assert "fit_sensitivity_nuts_chains" in RA.STAGE_BY_NAME["fit_sensitivity"].outputs


def test_dependency_order():
    idx = {s.name: i for i, s in enumerate(RA.STAGES)}
    for s in RA.STAGES:
        for r in s.needs + s.needs_tables:
            if r.kind == "upstream" and r.producer:
                assert idx[r.producer] <= idx[s.name], (s.name, r.producer)
    before = [("panels", "evidence"), ("annotators", "evidence"), ("grnboost2", "fm_scores"),
              ("evidence", "heldout_chip"), ("fit_primary", "fit_sensitivity"), ("fit_sensitivity", "modes_tables"),
              ("governance", "modes_tables"), ("robustness", "tuning"), ("heldout_eval", "sim_hub"),
              ("heldout_eval", "obs_eval"), ("governance", "obs_integration"), ("replication", "obs_integration"),
              ("heldout_eval", "obs_integration"), ("obs_eval", "obs_integration"), ("posterior_modes", "heldout_eval"),
              ("obs_models", "heldout_eval")]
    for a, b in before:
        assert idx[a] < idx[b], (a, b)
    assert RA.STAGES[-1].name == "paper_assets"


def test_select_stages():
    class A:
        only = None
        from_ = None
    a = A()
    assert [s.name for s in RA.select_stages(a)] == [s.name for s in RA.STAGES]
    a.from_ = "governance"
    sel = [s.name for s in RA.select_stages(a)]
    assert sel[0] == "governance" and sel[-1] == "paper_assets" and "heldout_eval" not in sel
    a.from_, a.only = None, ["paper_assets", "panels"]
    assert [s.name for s in RA.select_stages(a)] == ["panels", "paper_assets"]     # pipeline order
    a.only = ["nope"]
    with pytest.raises(SystemExit):
        RA.select_stages(a)


def test_tables_only_never_refits():
    """The tables-only path contains no MAP / NUTS / fit / simulation step."""
    bad = {"map", "laplace", "nuts", "fits", "cv", "ppc", "recovery", "truthstart", "modes", "map_tfw",
           "evidence_elig", "bridge", "mixture", "simulate", "sensmodes", "clusterstab", "fit-tissues", "fit-global",
           "search", "fit-shard0", "fit-shard1", "restart-check", "oracle-check", "metrics", "session-a", "session-b",
           "run"}
    for s in RA.STAGES:
        for st in s.steps_for(True):
            assert st.name not in bad, (s.name, st.name)


def test_dry_run_runs_nothing(monkeypatch, capsys):
    def boom(*a, **k):
        raise AssertionError("dry run executed a step")
    monkeypatch.setattr(RA, "run_step", boom)
    assert RA.main(["--tables-only", "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "heldout_eval.py estimators --subset reported_any" in out
    assert "fit_primary.py tables" in out and "fit_primary.py map" not in out
    assert RA.main(["--dry-run", "--only", "fit_primary"]) == 0
    out = capsys.readouterr().out
    assert "fit_primary.py map" in out and "fit_primary.py nuts --init random map_laplace" in out


# ----------------------------------------------------------------------------- measurement
def test_run_step_measures_memory_time_and_exit(tmp_path):
    code = "import time; b = bytearray(300 * 2**20); b[::4096] = b'x' * len(b[::4096]); time.sleep(0.6)"
    r = RA.run_step([PY, "-c", code], tmp_path / "a.log")
    assert r["exit_code"] == 0
    assert r["wall_time_s"] >= 0.6
    assert 250 <= r["peak_rss_mib"] <= 2000
    assert r["peak_rss_tree_mib"] >= r["peak_rss_mib"] * 0.5
    assert (tmp_path / "a.log").read_text().startswith("# ")
    r = RA.run_step([PY, "-c", "import sys; print('x'); sys.exit(3)"], tmp_path / "b.log")
    assert r["exit_code"] == 3
    assert "x" in (tmp_path / "b.log").read_text()


def test_child_env_threads(monkeypatch):
    monkeypatch.delenv("OMP_NUM_THREADS", raising=False)
    env = RA.child_env()
    assert env["OMP_NUM_THREADS"] == "2" and "intra_op_parallelism_threads=2" in env["XLA_FLAGS"]
    assert env["PYTHONPATH"].split(":")[0].endswith("/src")
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    assert RA.child_env()["OMP_NUM_THREADS"] == "1"          # an explicit setting is kept


# ----------------------------------------------------------------------------- timing table
def _row(**kw):
    base = dict(stage="evidence", layer="data", step="build", mode="full+tables", scope="all",
                timing_source="measured_isolated", wall_time_s=10.0, peak_rss_mib=100.0, exit_code=0)
    base.update(kw)
    return base


def test_write_timings_upsert_select_and_idempotent(tmp_path):
    p = tmp_path / "t.csv"
    RA.write_timings([_row(timing_source="recorded_meta", wall_time_s=50.0),
                      _row(stage="fit_primary", layer="model", step="map", mode="full", wall_time_s=7.0),
                      _row(stage="fit_primary", layer="model", step="map", mode="full", timing_source="recorded_meta",
                           wall_time_s=9.0)], p)
    RA.write_timings([_row(wall_time_s=12.0)], p)                    # measured row for evidence
    RA.write_timings([_row(wall_time_s=13.0)], p)                    # replaces it (same key)
    rows = list(csv.DictReader(open(p)))
    ev = [r for r in rows if r["stage"] == "evidence"]
    assert len(ev) == 2
    sel = [r for r in ev if r["selected"] == "True"]
    assert len(sel) == 1 and sel[0]["timing_source"] == "measured_isolated" and float(sel[0]["wall_time_s"]) == 13.0
    assert [r["stage"] for r in rows] == ["evidence", "evidence", "fit_primary", "fit_primary"]   # pipeline order
    assert all(r["analysis_status"] == "prespecified" for r in rows)
    before = p.read_bytes()
    RA.write_timings([], p)
    assert p.read_bytes() == before
    # a failed measurement is kept but never selected
    RA.write_timings([_row(stage="governance", layer="evaluation", step="all", exit_code=1)], p)
    g = [r for r in csv.DictReader(open(p)) if r["stage"] == "governance"]
    assert g[0]["selected"] == "False"


def test_stage_summary_sums_selected_steps(tmp_path):
    p = tmp_path / "t.csv"
    RA.write_timings([_row(stage="heldout_eval", layer="evaluation", step="refs", wall_time_s=10.0, peak_rss_mib=50),
                      _row(stage="heldout_eval", layer="evaluation", step="estimators", wall_time_s=100.0,
                           peak_rss_mib=900),
                      _row(stage="fit_primary", layer="model", step="map", mode="full", wall_time_s=60.0),
                      _row(stage="fit_primary", layer="model", step="tables", wall_time_s=5.0)], p)
    summ = {r["stage"]: r for r in RA.stage_summary(RA.read_timings(p))}
    he = summ["heldout_eval"]
    assert he["full_wall_s"] == 110.0 and he["tables_wall_s"] == 110.0 and he["full_peak_rss_mib"] == 900
    assert "estimators-reported_any" in he["full_steps_missing"]
    fp = summ["fit_primary"]
    assert fp["full_wall_s"] == 65.0 and fp["tables_wall_s"] == 5.0
    assert summ["panels"]["full_wall_s"] is None


def test_committed_timing_table_schema():
    if not RA.TIMINGS_CSV.exists():
        pytest.skip("no timing table yet")
    rows = RA.read_timings()
    assert rows and list(rows[0]) == RA.TIMING_COLUMNS
    names = {s.name: {st.name for st in s.steps} for s in RA.STAGES}
    for r in rows:
        assert r["stage"] in names and r["step"] in names[r["stage"]], (r["stage"], r["step"])
        assert r["timing_source"] in RA.SOURCE_PRIORITY
        assert r["analysis_status"] in {"prespecified", "post_hoc"}
        if r["selected"] == "True":
            assert str(r["exit_code"]) in {"0", "0.0"}
            assert float(r["wall_time_s"]) >= 0


# ----------------------------------------------------------------------------- environment / manifest
def test_requirements_pin_imported_packages(tmp_path):
    d = RA.imported_distributions()
    for k in ("numpy", "pandas", "scipy", "jax", "jaxlib", "numpyro", "arviz", "pyarrow"):
        assert k in d, k
    p = RA.write_requirements(tmp_path / "req.txt")
    lines = [ln for ln in p.read_text().splitlines() if ln and not ln.startswith("#")]
    assert lines and all("==" in ln.split("#")[0] for ln in lines)
    # the lock next to it pins the direct requirements and their dependencies (pandas needs python-dateutil)
    lock = [ln for ln in (tmp_path / "req.lock").read_text().splitlines() if ln and not ln.startswith("#")]
    names = {RA._canon(ln.split("==")[0]) for ln in lock}
    assert {"numpy", "pandas", "python-dateutil"} <= names and len(lock) > len(lines)
    assert all("==" in ln for ln in lock)


def test_environment_rows():
    keys = dict(RA.environment_rows())
    assert "python" in keys and "package numpy" in keys and "env OMP_NUM_THREADS (children)" in keys
    if sys.platform == "darwin":
        assert keys["cpu"] and int(keys["hw.ncpu"]) >= 1 and int(keys["hw.memsize (bytes)"]) > 0


def test_manifest_detects_changes(tmp_path, monkeypatch):
    data = tmp_path / "data"
    (data / "inputs" / "sub").mkdir(parents=True)
    (data / "tables").mkdir()
    (data / "inputs" / "sub" / "a.txt").write_text("a")
    (data / "tables" / "t.csv").write_text("x,y\n1,2\n")
    monkeypatch.setattr(RA, "DATA", data)
    monkeypatch.setattr(RA, "INPUTS", data / "inputs")
    monkeypatch.setattr(RA, "TABLES", data / "tables")
    m = RA.write_manifest(data / "MANIFEST.sha256")
    txt = m.read_text().splitlines()
    assert [ln.split("  ")[1] for ln in txt] == ["inputs/sub/a.txt", "tables/t.csv"]
    assert RA.check_manifest(m) == []
    (data / "tables" / "t.csv").write_text("x,y\n1,3\n")
    (data / "tables" / "new.csv").write_text("z\n")
    (data / "inputs" / "sub" / "a.txt").unlink()
    probs = RA.check_manifest(m)
    assert any(p.startswith("checksum differs: tables/t.csv") for p in probs)
    assert any(p.startswith("not in manifest: tables/new.csv") for p in probs)
    assert any(p.startswith("missing: inputs/sub/a.txt") for p in probs)


# ----------------------------------------------------------------------------- orchestration (fake stages)
@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    tables = tmp_path / "tables"
    tables.mkdir()
    out = tmp_path / "out"
    monkeypatch.setattr(RA, "TABLES", tables)
    monkeypatch.setattr(RA, "TIMINGS_CSV", tables / "pipeline_stage_timings.csv")
    monkeypatch.setattr(RA, "ENV_CSV", tables / "pipeline_environment.csv")
    monkeypatch.setattr(RA, "PIPE_OUT", out)
    monkeypatch.setattr(RA, "MANIFEST", tmp_path / "MANIFEST.sha256")
    monkeypatch.setattr(RA, "REQUIREMENTS", tmp_path / "req.txt")
    monkeypatch.setattr(RA, "REQUIREMENTS_LOCK", tmp_path / "req.lock")
    monkeypatch.setattr(RA, "DATA", tmp_path)
    monkeypatch.setattr(RA, "INPUTS", tmp_path / "inputs")
    (tmp_path / "inputs").mkdir()
    write = f"open({str(tables / 'toy.csv')!r}, 'w').write('a,b\\n1,2\\n')"
    stages = [
        RA.Stage("toy", "data", "toy stage", [RA.Step("make", [PY, "-c", write], True, True)], outputs=["toy"]),
        RA.Stage("ext", "data", "needs external data", [RA.Step("x", [PY, "-c", "pass"], True, True)],
                 needs=[RA.Req("external", "raw data", str(tmp_path / "absent.h5ad"))]),
        RA.Stage("later", "evaluation", "after toy", [RA.Step("y", [PY, "-c", "pass"], True, True)]),
    ]
    monkeypatch.setattr(RA, "STAGES", stages)
    monkeypatch.setattr(RA, "STAGE_BY_NAME", {s.name: s for s in stages})
    return tmp_path


def test_run_records_skips_external_and_checks_determinism(sandbox):
    rc = RA.main(["--tables-only", "--repeat-check"])
    assert rc == 0
    rows = list(csv.DictReader(open(RA.TIMINGS_CSV)))
    assert {r["stage"] for r in rows} == {"toy", "later"}             # 'ext' skipped (external data missing)
    assert all(r["timing_source"] == "measured_isolated" and r["selected"] == "True" for r in rows)
    det = list(csv.DictReader(open(RA.PIPE_OUT / "determinism_check.csv")))
    assert [(d["stage"], d["table"], d["identical"]) for d in det] == [("toy", "toy", "True")]
    assert RA.MANIFEST.exists() and RA.check_manifest(RA.MANIFEST) == []
    assert "SKIPPED: needs external data" in (RA.PIPE_OUT / "run.log").read_text()
    env = dict(r[:2] for r in csv.reader(open(RA.ENV_CSV)))
    assert "python" in env


def test_run_blocks_on_missing_upstream(sandbox):
    RA.STAGES[2].needs = [RA.Req("upstream", "toy output", str(sandbox / "nothing" / "*.parquet"), "toy")]
    assert RA.main(["--tables-only", "--no-record"]) == 1
    assert "BLOCKED" in (RA.PIPE_OUT / "run.log").read_text()
    assert not RA.TIMINGS_CSV.exists()


def test_failed_step_stops_the_run(sandbox):
    RA.STAGES[0].steps[0].argv = [PY, "-c", "raise SystemExit(2)"]
    assert RA.main(["--tables-only"]) == 1
    rows = list(csv.DictReader(open(RA.TIMINGS_CSV)))
    assert [(r["stage"], r["exit_code"], r["selected"]) for r in rows] == [("toy", "2", "False")]
    assert not RA.MANIFEST.exists()


# ----------------------------------------------------------------------------- fresh clone (pipeline_final)
def test_tables_only_path_needs_no_external_data():
    """Every tables-only step reads stored outputs or committed inputs, never the external raw data, so the
    tables-only run works on a fresh clone with the supporting archive."""
    for s in RA.STAGES:
        if s.steps_for(True):
            assert all(r.kind != "external" for r in s.reqs_for(True)), s.name
    for name, flag in (("panels", "--table-only"), ("fm_scores", "--tables-only"), ("heldout_chip", "--table-only")):
        st = RA.STAGE_BY_NAME[name]
        assert [x.name for x in st.steps_for(True)] == ["table"] and flag in st.steps_for(True)[0].argv
        assert [x.name for x in st.steps_for(False)] == ["build"]


def test_skip_and_select():
    class A:
        only = None
        from_ = None
        skip = ["paper_assets"]
    names = [s.name for s in RA.select_stages(A())]
    assert "paper_assets" not in names and names[0] == "panels" and len(names) == len(RA.STAGES) - 1


def test_write_timings_whole_scope_supersedes_per_scope_rows(tmp_path):
    p = tmp_path / "t.csv"
    nuts = dict(stage="fit_primary", layer="model", step="nuts", mode="full")
    RA.write_timings([_row(**nuts, scope="kidney, map_laplace init", timing_source="recorded_meta", wall_time_s=100.0),
                      _row(**nuts, scope="lung, map_laplace init", timing_source="recorded_meta", wall_time_s=200.0)], p)
    summ = {r["stage"]: r for r in RA.stage_summary(RA.read_timings(p))}
    assert summ["fit_primary"]["full_wall_s"] == 300.0
    RA.write_timings([_row(**nuts, scope="all", wall_time_s=250.0)], p)     # a later whole-step measurement
    rows = RA.read_timings(p)
    assert [(r["scope"], r["selected"]) for r in rows if r["step"] == "nuts"] == [
        ("all", "True"), ("kidney, map_laplace init", "False"), ("lung, map_laplace init", "False")]
    summ = {r["stage"]: r for r in RA.stage_summary(rows)}
    assert summ["fit_primary"]["full_wall_s"] == 250.0                       # not 550


def test_write_timings_relabels_steps_that_left_the_tables_only_path(tmp_path):
    p = tmp_path / "t.csv"
    RA.write_timings([_row(stage="panels", step="build", mode="full+tables", wall_time_s=2.0)], p)
    r = RA.read_timings(p)
    assert [(x["step"], x["mode"]) for x in r] == [("build", "full")]       # build is a full-run step now
    summ = {x["stage"]: x for x in RA.stage_summary(r)}
    assert summ["panels"]["full_wall_s"] == 2.0 and summ["panels"]["tables_wall_s"] is None


def test_record_writers_flags_undeclared_and_conflicting_writers(tmp_path, monkeypatch):
    monkeypatch.setattr(RA, "TABLES", tmp_path)
    a, b = RA.STAGE_BY_NAME["fit_sensitivity"], RA.STAGE_BY_NAME["modes_tables"]
    st_a, st_b = a.steps[-1], b.steps[-1]
    w, last = [], {}
    f = tmp_path / "fit_recovery.csv"
    s0 = RA.table_state()
    f.write_text("x\n1\n")
    assert RA.record_writers(a, st_a, s0, RA.table_state(), w, last) == 0     # also_writes: declared
    assert w[-1]["status"] == "declared" and w[-1]["content_changed"]
    s1 = RA.table_state()
    f.write_text("x\n1\n")                                                  # same content again: no conflict
    import os
    os.utime(f, ns=(s1["fit_recovery"][1] + 10 ** 9, s1["fit_recovery"][1] + 10 ** 9))
    assert RA.record_writers(b, st_b, s1, RA.table_state(), w, last) == 0
    assert not w[-1]["content_changed"]
    s2 = RA.table_state()
    f.write_text("x\n2\n")                                                  # different content: conflict
    assert RA.record_writers(a, st_a, s2, RA.table_state(), w, last) == 1
    assert w[-1]["conflict_with"] == "modes_tables/tables"
    s3 = RA.table_state()
    (tmp_path / "gov_tiers.csv").write_text("y\n")                          # owned by governance
    RA.record_writers(a, st_a, s3, RA.table_state(), w, last)
    assert w[-1]["table"] == "gov_tiers" and w[-1]["status"] == "UNDECLARED"


def test_traced_inputs_keeps_reads_before_writes(tmp_path, monkeypatch):
    out = tmp_path / "outputs" / "r2"
    (out / "fits").mkdir(parents=True)
    for n in ("a.npy", "b.json", "c.json"):
        (out / "fits" / n).write_text("1")
    monkeypatch.setattr(RA, "OUT", out)
    monkeypatch.setattr(RA, "SP", tmp_path)
    run = tmp_path / "run"
    (run / "trace").mkdir(parents=True)
    # evidence (earlier stage) writes c.json; fit_primary reads a, b, c; c was produced by the run itself
    (run / "trace" / "evidence__build.txt").write_text(f"W\t1\t{out / 'fits' / 'c.json'}\n")
    (run / "trace" / "fit_primary__tables.txt").write_text(
        f"R\t2\t{out / 'fits' / 'a.npy'}\nL\t2\t{out / 'fits'}\nR\t2\t{out / 'fits' / 'c.json'}\n"
        f"R\t2\t{out / 'fits' / 'b.json'}\nR\t2\t/nonexistent/elsewhere.csv\n")
    tr = RA.traced_inputs(run)
    assert sorted(Path(p).name for p in tr["needed"]) == ["a.npy", "b.json"]
    assert str(out / "fits") in tr["listed"] and tr["outside"] == {}


def test_trace_hook_records_reads_and_writes(tmp_path):
    import subprocess
    trace = tmp_path / "trace.txt"
    target = tmp_path / "x.csv"
    target.write_text("a\n1\n")
    code = (f"import pandas as pd; pd.read_csv({str(target)!r}); "
            f"pd.DataFrame({{'b': [1]}}).to_csv({str(tmp_path / 'y.csv')!r})")
    env = RA.child_env(trace)
    subprocess.run([PY, "-c", code], env=env, check=True, cwd=tmp_path)
    lines = [ln.split("\t") for ln in trace.read_text().splitlines()]
    assert ["R", str(target)] in [[k, p] for k, _, p in lines]
    assert ["W", str(tmp_path / "y.csv")] in [[k, p] for k, _, p in lines]
    assert "R2_TRACE_INPUTS" not in RA.child_env()


def test_archive_readme_and_extra_patterns(tmp_path, monkeypatch):
    for k, name in (("ARCHIVE_README", "README.md"), ("ARCHIVE", "a.tar.gz"), ("ARCHIVE_SUM", "a.tar.gz.sha256"),
                    ("ARCHIVE_LIST", "files.sha256")):
        monkeypatch.setattr(RA, k, tmp_path / name)
    info = dict(n_files=12, n_traced=10, extra_patterns=list(RA.ARCHIVE_EXTRA), bytes_uncompressed=2 ** 30,
                bytes_archive=2 ** 29, sha256="0" * 64, by_dir={"outputs/r2/fits": [10, 2 ** 20]}, outside={},
                outside_env={}, run_dir="x")
    txt = RA.write_archive_readme(info).read_text()
    assert "12 files" in txt and "(10 files)" in txt and "(2 files:" in txt and "| `outputs/r2/fits/` | 10 |" in txt
    assert "tar -xzf" in txt and "SUPPORTING_OUTPUTS.sha256" in txt
    # the existence-checked simulation files are under outputs/r2 and are globs relative to the subproject
    assert all(p.startswith("outputs/r2/") and "*" in p for p in RA.ARCHIVE_EXTRA)
    if (RA.OUT / "sim_hub" / "fits").is_dir():
        assert all(list(RA.SP.glob(p)) for p in RA.ARCHIVE_EXTRA)
