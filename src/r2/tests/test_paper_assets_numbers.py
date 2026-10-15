"""Tests for the prose number macros (r2.paper_assets.prose_numbers / prose_numbers_spec).

* every macro specification evaluates (no missing CSV, row, column or ambiguous filter), names are
  ``\\n`` + letters only and unique;
* ``paper/r2/numbers.tex`` is current (byte-identical to a fresh build);
* every CSV-backed macro value is re-read independently with pandas and formatted again;
* the manuscript prose contains no literal result number, every macro used is defined and every
  non-helper macro is used; no ``\\pending`` placeholder is left;
* the checker flags literal numbers in text and math and ignores references, citations, years in a date or
  author-year context, equation environments, allow-listed identifiers and macros; it flags bare four-digit
  numbers, ``\\texttt`` numbers, digits glued to other letters and lone math integers, and words that refer to a
  review or revision history;
* every wording condition (``require``) holds, and a failing condition is reported;
* the formats (sign, powers of ten, safe-side rounding, number words) behave as documented.

Run: ``.venv/bin/python -m pytest subproject_24_multireference_latent_truth_model/src/r2/tests/test_paper_assets_numbers.py -q``
"""
from __future__ import annotations

import math
import re
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from r2.paper_assets import prose_numbers as PN  # noqa: E402
from r2.paper_assets import prose_numbers_spec as SPEC  # noqa: E402
from r2.paper_assets import style as S  # noqa: E402


@pytest.fixture(scope="module")
def specs():
    return PN.evaluate(SPEC.specs())


def test_specs_evaluate_and_names(specs):
    assert len(specs) > 500
    assert PN.validate_names(specs) == []
    errs = [f"{n.name}: {n.error}" for n in specs if n.error]
    assert errs == []
    for n in specs:
        assert sum(x is not None for x in (n.csv, n.const, n.expr)) == 1, n.name
        if n.csv:
            assert n.col and (S.TABLES / n.csv).exists(), n.name


def test_numbers_tex_current(specs):
    assert PN.NUMBERS_TEX.exists()
    assert PN.NUMBERS_TEX.read_text() == PN.render(specs), "run make_paper_assets.py numbers"


def test_every_macro_has_provenance_comment(specs):
    lines = PN.NUMBERS_TEX.read_text().split("\n")
    for i, ln in enumerate(lines):
        if ln.startswith("\\newcommand"):
            assert lines[i - 1].startswith("% ") and len(lines[i - 1]) > 4, ln


def _independent(n: PN.Num, raw: dict):
    df = pd.read_csv(S.TABLES / n.csv, low_memory=False)
    q = re.sub(r"\{(n[A-Za-z]+)\}", lambda m: repr(float(raw[m.group(1)])), n.query) if n.query else None
    sel = df.query(q, engine="python") if q else df
    v = sel[n.col]
    if n.agg is None:
        assert v.nunique(dropna=False) == 1, n.name
        x = v.iloc[0]
    elif n.agg == "count":
        x = len(sel)
    elif n.agg == "nunique":
        x = v.nunique()
    elif n.agg == "same":
        x = float(v.iloc[0])
    elif n.agg.startswith("abs"):
        x = getattr(v.astype(float).abs(), n.agg[3:])()
    else:
        x = getattr(v.astype(float), n.agg)()
    if n.regex:
        x = float(re.search(n.regex, str(x)).group(1))
    if n.scale is not None:
        x = float(x) * n.scale
    return x


def test_csv_values_reread_independently(specs):
    raw = {n.name: n.raw for n in specs}
    for n in specs:
        if not n.csv:
            continue
        x = _independent(n, raw)
        if isinstance(x, str):
            assert x == n.raw, n.name
        else:
            assert math.isclose(float(x), float(n.raw), rel_tol=1e-12, abs_tol=1e-15), n.name
        assert PN.fmt_tex(x, n.fmt) == n.tex, n.name


def test_constants_match_prespecified(specs):
    from r2.common import PRESPECIFIED
    by = {n.name: n for n in specs}
    assert by["nTopK"].tex == str(PRESPECIFIED["topk_per_tf"])
    assert by["nBfdrStrong"].raw == PRESPECIFIED["bfdr_strong"]
    assert by["nBfdrModerate"].raw == PRESPECIFIED["bfdr_moderate"]
    assert by["nRepEndpoint"].raw == PRESPECIFIED["replication_endpoint_enrichment"]
    assert by["nEvalK"].raw in PRESPECIFIED["eval_k"]


def test_prose_is_bound(specs):
    res = PN.check(specs)
    assert res["violations"] == [], res["violations"][:10]
    assert res["undefined"] == []
    assert res["unused"] == []
    for f, _, _ in PN.PROSE:
        txt = (S.PAPER / f).read_text()
        assert "\\pending{" not in txt, f
    assert "numbers.tex" in (S.PAPER / "manuscript.tex").read_text()


def test_audit_csv(specs):
    assert PN.AUDIT_CSV.exists()
    a = pd.read_csv(PN.AUDIT_CSV, keep_default_na=False)
    for c in ("macro", "value", "csv", "filter", "column", "used_in"):
        assert c in a.columns
    assert len(a) == len(specs)
    nonhelper = a[a["helper"].astype(str) == "False"]
    assert (nonhelper["used_in"] != "").all()


def test_checker_flags_and_ignores():
    text = "\n".join([
        r"Family-HB has an AP lift of 0.96 [0.94, 0.98] and $p=1.2\times10^{-3}$.",     # 3 + 2 literals (mantissa, base)
        r"Bound: \nApHbEnc{} [\nApHbEncLo{}, \nApHbEncHi{}] (\Cref{tab:heldout}, \citet{dixit2016perturb}).",
        r"Dixit 2016 studies and Q20, L1O, scGPT, $F_1$, $x_{ij}\in\{0,1\}$ and 95\% intervals.",
        r"\begin{equation} a = 0.25 \label{eq:x} \end{equation}",
        r"% a comment with 0.77",
        r"difference $-0.11$ at 12.6-fold",
    ])
    viol, uses = PN.scan_text("sec_results.tex", text)
    lits = [v["literal"] for v in viol]
    # exponent 3 is a 0-3 math constant; "Q20" (digits glued to letters, not an allow-listed identifier) is flagged
    assert lits == ["0.96", "0.94", "0.98", "1.2", "10", "Q20", "0.11", "12.6"], lits
    assert {u["macro"] for u in uses} == {"nApHbEnc", "nApHbEncLo", "nApHbEncHi"}


def test_formats():
    f = PN.fmt_tex
    assert f(-0.0081, "r3") == r"\ensuremath{-0.008}"
    assert f(-0.0001, "r3") == r"\ensuremath{-0.000}"
    assert f(1.8e-46, "e2") == r"\ensuremath{1.8\times10^{-46}}"
    assert f(1e-6, "pow10") == r"\ensuremath{10^{-6}}"
    assert f(6.3e-5, "lgfloor") == r"\ensuremath{10^{-5}}" and f(4.4e-3, "lgceil") == r"\ensuremath{10^{-2}}"
    assert f(0.946, "floor2") == "0.94" and f(0.0024, "ceil3") == "0.003" and f(0.95, "floor2") == "0.95"
    assert f(1234567, "int") == "1{,}234{,}567" and f(19005, "approx1") == "20{,}000"
    assert f(12, "word") == "twelve" and f(3, "Word") == "Three" and f(6, "ord") == "sixth"
    assert f(0.84, "ordpct") == "84th" and f(0.81, "ordpct") == "81st" and f(0.12, "ordpct") == "12th"
    assert f(2.0, "fold") == "twofold" and f(0.5, "frac") == "half" and f(5.28, "wfloor") == "five"
    assert f(0.4542, "pct0") == "45" and f("NFKB1", "str") == "NFKB1"
    with pytest.raises(ValueError):
        f(2.5, "word")
    with pytest.raises(ValueError):
        f(float("nan"), "r2")


def test_checker_years_dates_identifiers_and_lone_math():
    """Years only in a date or author-year context; digits glued to letters only for allow-listed identifiers;
    \\texttt arguments are checked; a lone integer 0-3 forming a whole math span is a count, not a constant."""
    text = "\n".join([
        r"We used 2000 cells, accessed 2026-01-08, in October 2026, as in Garcia-Alonso et al. 2019 and (2019).",
        r"Values \texttt{0.37} and x0.37 and AP1.05 on ENETS2 in K562 cells (L1O, GRNBoost2, TRRUST~v2, THP-1, V2-104M).",
        r"We found $3$ optima and $x_{1}$, $\{0,1\}$, $2^{k}$; question Q20.",
    ])
    viol, _ = PN.scan_text("sec_results.tex", text)
    assert sorted(v["literal"] for v in viol) == sorted(["2000", "0.37", "x0.37", "AP1.05", "3", "Q20"])


def test_artefact_scan():
    text = ("As the reviewer asked, we now report the corrected R2 values (Q6); see \\citep{review2019x}.\n"
            "In the previous version this was updated. A rule \\ref{sec:revision} and DoRothEA levels are fine.")
    lits = [a["literal"] for a in PN.scan_artefacts("sec_results.tex", text)]
    assert lits == ["reviewer", "we now", "corrected", "R2", "(Q6)", "previous version", "updated"]


def test_prose_has_no_artefacts(specs):
    assert PN.check(specs)["artefacts"] == []


def test_wording_conditions_hold(specs):
    """Every wording condition of the specification holds for the current tables (build fails otherwise)."""
    assert PN.check_requires(specs) == []
    assert sum(1 for n in specs if n.require) >= 40


def test_wording_condition_failure_is_reported():
    a = PN.Num("nAa", expr="1", fmt="int")
    b = PN.Num("nBb", expr="2", fmt="int", require="{nAa} == {nBb}", require_note="'all'")
    c = PN.Num("nCc", expr="3", fmt="int", require="{nZz} > 0", require_note="unknown macro")
    specs = PN.evaluate([a, b, c])
    errs = PN.check_requires(specs)
    assert len(errs) == 2
    assert "nBb" in errs[0] and "FALSE" in errs[0] and "'all'" in errs[0]
    assert "nCc" in errs[1] and "nZz" in errs[1]
    assert PN.render(specs).count("requires (") == 2
