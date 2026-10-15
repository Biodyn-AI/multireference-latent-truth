"""Tests for the manuscript tables (r2.paper_assets.tables / tables_supp / textable).

* every main and supplementary table renders (booktabs, caption + label, provenance comment,
  no vertical rules, cell counts match the column specification);
* every printed number round-trips to its CSV: the registered CSV cell (file, row, column) is
  re-read independently with pandas, formatted with the registered format and must equal the
  printed text, which must occur in the .tex; derived numbers are checked against their source
  cells and re-parsed;
* the formatters follow the conventions (ratios 2 decimals, p-values 2 significant digits in
  scientific notation, counts with {,} thousands separators);
* post hoc daggers come with their note; stale-source marking works;
* (if pdflatex is available) every table compiles with the R1 preamble without errors,
  overfull boxes wider than 1 pt or floats that do not fit the page.

Run: ``.venv/bin/python -m pytest subproject_24_multireference_latent_truth_model/src/r2/tests/test_paper_assets_tables.py -q``
"""
from __future__ import annotations

import math
import re
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from r2.paper_assets import style as S  # noqa: E402
from r2.paper_assets import tables as T  # noqa: E402
from r2.paper_assets import textable as X  # noqa: E402


@pytest.fixture(scope="module")
def built():
    X.STALE_PREFIXES[:] = []
    return T.build(write=False)


@pytest.fixture(scope="module")
def csvs():
    cache = {}

    def get(name):
        if name not in cache:
            cache[name] = pd.read_csv(S.TABLES / name, low_memory=False)
        return cache[name]
    return get


# ----------------------------------------------------------------------------- formats
def test_formats():
    assert X.fmt(1.23456, "ratio") == "1.23"
    assert X.fmt(0.0123456, "p") == r"$1.2\times 10^{-2}$"
    assert X.fmt(3.1e-16, "p") == r"$3.1\times 10^{-16}$"
    assert X.fmt(1.0, "p") == "1"
    assert X.fmt(1234567, "count") == "1{,}234{,}567"
    assert X.fmt(0.0044, "prob") == "0.0044"
    assert X.fmt(1.8e-46, "prob") == r"$1.8\times 10^{-46}$"
    assert X.fmt(0.68, "prob") == "0.68"
    assert X.fmt(-0.0061, "ratio3") == "$-$0.006"
    assert X.fmt(float("nan"), "ratio") == X.DASH
    assert X.fmt(float("inf"), "count") == r"$\infty$"
    assert X.fmt(0.00996, "sig2") == "0.010"
    for kind, x in [("ratio", 0.9587), ("p", 2.4e-3), ("count", 15662.4), ("prob", 0.00435),
                    ("sig2", 7.3e-05), ("pct", 0.3566), ("rhat", 1.01624)]:
        assert math.isclose(X.parse_number(X.fmt(x, kind)), float(f"{X.parse_number(X.fmt(x, kind)):.12g}"))


def test_escape_and_ncols():
    assert X.tex("a_b & 50%") == r"a\_b \& 50\%"
    assert X.tex("≥2") == r"$\geq$2"
    assert X._ncols("@{}lrrp{3cm}c@{}") == 5
    assert X._ncols("@{}l" + "rcc" * 3 + "@{}") == 10


# ----------------------------------------------------------------------------- rendering
def test_registry_complete(built):
    names = set(built)
    for nm in T.MAIN:
        assert nm in names
    assert all(n.startswith("tab_") for n in T.MAIN)
    assert len([n for n in names if n.startswith("tabS_")]) >= 20


def test_every_table_renders(built):
    for nm, t in built.items():
        tx = t.tex
        assert tx.startswith("% provenance:"), nm
        assert "data/r2/tables/" in tx.splitlines()[0], nm
        assert tx.count("\\begin{table}") == tx.count("\\end{table}") >= 1, nm
        assert "\\caption{" in tx and f"\\label{{{t.label}}}" in tx, nm
        assert "\\toprule" in tx and "\\midrule" in tx and "\\bottomrule" in tx, nm
        for spec in re.findall(r"\\begin\{tabular\}\{([^\n]*)\}\n", tx):
            assert "|" not in spec, (nm, spec)
        assert "\\small" in tx, nm
        assert not re.search(r"(?<!\\)nan\b|NaN", tx.replace("\\nan", "")), nm
        # every provenance source exists
        for ln in tx.splitlines():
            m = re.match(r"%\s+(\S+\.csv) \(modified", ln)
            if m:
                assert (S.TABLES / m.group(1)).exists(), (nm, m.group(1))


def test_posthoc_note(built):
    for nm, t in built.items():
        body = "\n".join(ln for ln in t.tex.splitlines() if not ln.startswith("%"))
        if X.DAG in body.replace(X.POSTHOC_NOTE, ""):
            assert "Post hoc: defined after the held-out data had been examined" in body, nm


# ----------------------------------------------------------------------------- round trip
TEXT_ONLY = {"tabS_complexity"}   # complexity orders only (text table)
_TOL = {"ratio": 0.005, "ratio3": 0.0005, "num1": 0.05, "num0": 0.5, "num4": 0.00005, "count": 0.5,
        "pct": 0.05, "pct0": 0.5, "pct2": 0.005, "rhat": 0.0005}


def _close(kind, text, value):
    parsed = X.parse_number(text)
    if math.isnan(parsed):
        return False
    if kind in ("p", "prob", "sig2", "sig3", "sci2"):
        if kind == "p" and text == "1":
            return value >= 0.995
        return math.isclose(parsed, value, rel_tol=0.051, abs_tol=0.0051 if abs(value) >= 0.01 else 0)
    target = 100 * value if kind.startswith("pct") else value
    if math.isinf(value):
        return math.isinf(parsed)
    return abs(parsed - target) <= _TOL[kind] * (1 + 1e-9) + 1e-12


def test_numbers_round_trip_to_csv(built, csvs):
    n = 0
    for nm, t in built.items():
        assert t.cells or nm in TEXT_ONLY, f"{nm}: no registered numbers"
        for c in t.cells:
            assert c.text in t.tex, (nm, c.text)
            if c.derived:
                for csv, row, col, val in c.sources:
                    v = csvs(csv).iloc[row][col]
                    if val is not None and not (isinstance(v, str)):
                        assert np.isclose(float(v), val, equal_nan=True), (nm, csv, row, col)
            else:
                v = csvs(c.csv).iloc[c.row][c.column]
                assert X.fmt(v, c.kind) == c.text, (nm, c.csv, c.row, c.column, v, c.text)
                assert c.value is not None and (np.isclose(float(v), c.value) or
                                                (math.isinf(float(v)) and math.isinf(c.value))), (nm, c.column)
            if c.value is not None and math.isfinite(c.value):
                assert _close(c.kind, c.text, c.value), (nm, c.column, c.kind, c.text, c.value)
            n += 1
    assert n > 3000


def test_specific_values(built, csvs):
    """Spot checks against independent pandas look-ups."""
    he = csvs("heldout_estimators.csv")
    r = he[(he.universe == "global") & (he.reference == "encode") & (he.eval_subset == "all_evaluable")
           & (he.score == "fisher")].iloc[0]
    assert f"{r.ap_lift:.2f} [{r.ap_lift_lo:.2f}, {r.ap_lift_hi:.2f}]" in built["tab_heldout"].tex
    ev = csvs("evidence_summary.csv")
    g = ev[(ev.universe == "global") & (ev.variant == "prespecified_cc") & (ev.annotator == "__ALL__")].iloc[0]
    assert f"{int(g.n_edges):,}".replace(",", "{,}") in built["tab_universes"].tex
    fc = csvs("fit_convergence.csv")
    n = fc[(fc.universe == "global") & (fc.fit_id == "family_hb_nuts")].iloc[0]
    assert f"{n.r_hat_max:.3f}" in built["tab_fit"].tex


def test_stale_marking():
    X.STALE_PREFIXES[:] = ["heldout_"]
    try:
        t = T.tab_heldout()
        assert "% WARNING: built from source tables that were being regenerated" in t.tex
        assert "heldout_estimators.csv" in t.stale_sources
        t2 = T.tab_universes()
        assert not t2.stale_sources
    finally:
        X.STALE_PREFIXES[:] = []


def test_split_body_keeps_rows():
    body = [("panel", "A")] + [["x", "1"]] * 30 + [("panel", "B")] + [["y", "2"]] * 30
    parts = X.split_body(body, 25)
    assert len(parts) >= 3
    flat = [it for p in parts for it in p if not isinstance(it, tuple)]
    assert len(flat) == 60


# ----------------------------------------------------------------------------- LaTeX
@pytest.mark.skipif(shutil.which("pdflatex") is None, reason="pdflatex not available")
def test_tables_compile(built, tmp_path):
    out = tmp_path / "tables"
    for nm, t in built.items():
        X.write_table(t.tex, out, nm)
    res = T.check_latex(out, list(built), workdir=tmp_path / "latex")
    bad = {k: v for k, v in res.items() if not v["ok"] or v["overfull_pt"] > 1.0 or v["float_too_large"]}
    assert not bad, bad


def test_written_tables_are_current(built):
    """The .tex files in paper/r2/tables match a fresh build (no stale output after a CSV change)."""
    missing = [nm for nm in built if not (S.TAB_DIR / f"{nm}.tex").exists()]
    if len(missing) == len(built):
        pytest.skip("tables not generated yet")
    assert not missing, missing
    for nm, t in built.items():
        disk = (S.TAB_DIR / f"{nm}.tex").read_text()
        strip = lambda s: "\n".join(ln for ln in s.splitlines() if not ln.startswith("% provenance:")
                                    and "(modified" not in ln)
        assert strip(disk) == strip(t.tex), f"{nm}: paper/r2/tables/{nm}.tex is stale; rebuild the tables"


# ----------------------------------------------------------------------------- assets_fix additions
def _f(x, kind):
    return X.fmt(x, kind)


def test_heldout_labels_and_new_columns(built, csvs):
    tx = built["tab_heldout"].tex
    assert "Family-HB (MAP)" in tx and "Family-HB (NUTS)" in tx
    assert "Composition-adjusted AP lift" in tx and "the NUTS fit was not tested" in tx
    he = csvs("heldout_estimators.csv")
    r = he[(he.universe == "global") & (he.reference == "enets2") & (he.eval_subset == "all_evaluable")
           & (he.score == "family_hb")].iloc[0]
    assert f"{_f(r.enr_at_500, 'ratio')} [{_f(r.enr_at_500_lo, 'ratio')}, {_f(r.enr_at_500_hi, 'ratio')}]" in tx
    assert f"{_f(r.ap_lift_comp_adj, 'ratio')} [{_f(r.ap_lift_comp_adj_lo, 'ratio')}" in tx
    hp = csvs("heldout_paired.csv")
    conf = hp[(hp.universe == "global") & (hp.metric == "ap_lift") & (hp.analysis_role == "confirmatory")]
    assert set(conf.score_a) == {"family_hb"}          # every confirmatory test is the MAP fit


def test_robustness_clustered_and_no_switch(built, csvs):
    tx = built["tab_robustness"].tex
    rk = csvs("rob_method_ranking.csv")
    for sub in ("all", "no_fhb_mode_switch"):
        r = rk[(rk.tissue == "all_tissues") & (rk.estimator == "family_hb") & (rk.protocol == "source")
               & (rk.subset == sub) & (rk.metric == "f1")].iloc[0]
        assert f"[{_f(r.mean_D_cluster_lo, 'ratio3')}, {_f(r.mean_D_cluster_hi, 'ratio3')}]" in tx, sub
    assert "Without mode switches" in tx and "lung, SIGNOR and TRRUST dropped" in tx
    # the unclustered interval of the EM source row is no longer printed
    r = rk[(rk.tissue == "all_tissues") & (rk.estimator == "em") & (rk.protocol == "source") & (rk.subset == "all")
           & (rk.metric == "f1")].iloc[0]
    assert f"[{_f(r.mean_D_boot_lo, 'ratio3')}, {_f(r.mean_D_boot_hi, 'ratio3')}]" not in tx


def test_methods_promoted_has_all_edges_row(built, csvs):
    tx = built["tab_methods_promoted"].tex
    gp = csvs("gov_promoted_by_method.csv")
    a = gp[(gp.method == "all_edges") & (gp.tissue == "immune")].iloc[0]
    assert "All edges &" in tx and _f(a.n_predicted, "count") in tx


def test_compute_two_floats_and_projection_ranges(built, csvs):
    tx = built["tab_compute"].tex
    assert "\\label{tab:compute}" in tx and "\\label{tab:compute-scaling}" in tx
    assert tx.count("\\begin{table}") == 2
    sn = csvs("scale_nuts.csv")
    pr = sn[(sn.cid == "template_10xreal_K42") & (sn.status == "projected")]
    lo, hi = pr.t_nuts_projected_s.min(), pr.t_nuts_projected_cap_s.max()
    assert f"{_f(lo, 'count')}--{_f(hi, 'count')}$^{{n}}$" in tx
    assert "Wall times are upper bounds" not in tx         # the bound wording is per entry type now


def test_modes_sim_simulators_and_extrapolation(built):
    tx = built["tabS_modes_sim"].tex
    for lab in ("Design-faithful simulator (primary)", "stratum-specific class weights", "Fixed observation masks"):
        assert lab in tx, lab
    assert "truth = mode 1" not in tx and "second mode" in tx
    assert "[1.00, 1.00]" not in tx                         # no degenerate Wald interval is printed


def test_signed_zero_interval_bound():
    r = pd.Series({"lo": -0.0001, "hi": 0.002, "_csv": "x.csv", "_row": 0})
    t = X.Table("t")
    assert t.ci(r, "lo", "hi", "ratio") == "[$-$0.00, 0.00]"           # one more decimal still rounds to 0
    assert t.ci(r, "lo", "hi", "ratio3") == "[$-$0.0001, 0.002]"      # one more decimal shows the sign
    r2 = pd.Series({"lo": -0.0009, "hi": 0.002, "_csv": "x.csv", "_row": 0})
    assert t.ci(r2, "lo", "hi", "ratio") == "[$-$0.001, 0.00]"
