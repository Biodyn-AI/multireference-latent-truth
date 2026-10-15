"""Tests for r2.paper_assets (shared style and the main figures fig1..fig8).

* the fixed colour mapping uses only Okabe-Ito colours;
* number formatters follow the manuscript conventions;
* every figure's data function reads values that round-trip to the CSVs;
* every figure renders to PDF + 300-dpi PNG at the 16 cm text width, with a
  caption file that carries a provenance header;
* every number printed in a caption is a value of the figure's data (rounded as
  printed), so no number is typed by hand.

Run: ``.venv/bin/python -m pytest src/r2/tests/test_paper_assets.py -q``
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from r2.paper_assets import figures as F  # noqa: E402
from r2.paper_assets import style as S  # noqa: E402

OKABE = {c.upper() for c in S.OKABE_ITO.values()}


# ----------------------------------------------------------------------------- style
def test_palette_is_okabe_ito_everywhere():
    cols = [s.color for s in S.SCORE_STYLE.values()]
    cols += list(S.REFERENCE_COLOR.values()) + list(S.FAMILY_COLOR.values()) + list(S.TIER_COLOR.values())
    cols += list(S.UNIVERSE_COLOR.values()) + [v["color"] for v in S.STATUS_STYLE.values()]
    assert {c.upper() for c in cols} <= OKABE


def test_estimators_have_distinct_colours():
    cols = [S.SCORE_STYLE[e].color for e in S.ESTIMATOR_ORDER]
    assert len(set(cols)) == len(cols)
    fams = [S.FAMILY_COLOR[f] for f in ("LIT", "MOTIF", "COEXP", "CHIP")]
    assert len(set(fams)) == 4


def test_formatters():
    assert S.fmt_ratio(1.2345) == "1.23"
    assert S.fmt_ratio(np.nan) == "--"
    assert S.fmt_p(0.000123) == "$1.2\\times 10^{-4}$"
    assert S.fmt_p(0.05, latex=False) == "5.0e-02"
    assert S.fmt_count(9314634) == "9{,}314{,}634"
    assert S.fmt_count(9314634, latex=False) == "9,314,634"
    assert S.fmt_pct(0.9619) == "96"
    assert S.fmt_sig(0.00011617, 2) == "0.00012"


def test_provenance_records_sources(tmp_path):
    P = S.Provenance("x")
    df = P.read("universe_panels", "all rows")
    assert len(df) > 0
    txt = P.text()
    assert "universe_panels.csv" in txt and "all rows" in txt
    assert P.tex_comment().splitlines()[0].startswith("% provenance:")


# ----------------------------------------------------------------------------- data round trip
def _csv(name):
    return pd.read_csv(S.TABLES / f"{name}.csv")


def test_fig2_alpha_beta_round_trip():
    fd = F.data_fig2_model()
    sp = _csv("fit_source_parameters")
    sp = sp[(sp.universe == "global") & (sp.fit_id == "family_hb_nuts")]
    for r in fd.d["ab"].itertuples():
        a = sp[(sp.annotator == r.annotator) & (sp.parameter == "alpha")].iloc[0]
        b = sp[(sp.annotator == r.annotator) & (sp.parameter == "beta")].iloc[0]
        assert r.alpha == a.value and r.alpha_lo == a.lo and r.alpha_hi == a.hi
        assert r.beta == b.value and r.beta_lo == b.lo and r.beta_hi == b.hi
    src = _csv("source_table")
    assert len(fd.d["ab"]) == int((~src.held_out.map(S.as_bool)).sum())


def test_fig4_ap_lift_round_trip():
    fd = F.data_fig4_heldout()
    h = _csv("heldout_estimators")
    for r in fd.d["h"].itertuples():
        c = h[(h.universe == "global") & (h.eval_subset == "all_evaluable") & (h.reference == r.reference)
              & (h.score == r.score)]
        assert len(c) == 1
        assert c.ap_lift.iloc[0] == r.ap_lift and c.ap_lift_lo.iloc[0] == r.ap_lift_lo


def test_fig5_decomposition_identity():
    """raw ratio = base-rate ratio x enrichment ratio in the plotted data."""
    m = F.data_fig5_universe().d["m"]
    assert np.allclose(m.ratio_raw, m.ratio_base_rate * m.ratio_enrichment, rtol=1e-3)
    assert (m.null_ratio_lo <= m.null_ratio_hi).all()


def test_fig6_counts_round_trip():
    d = F.data_fig6_governance().d
    gt = _csv("gov_tiers")
    for r in d["tiers"].itertuples():
        c = gt[(gt.tissue == r.tissue) & (gt.scope == "dual") & (gt.tier == r.tier)]
        assert int(c.n_edges.iloc[0]) == int(r.n_edges)
    assert (d["pm"].k == F._PRE["topk_per_tf"]).all()


def test_fig8_units_round_trip():
    d = F.data_fig8_replication().d
    u = _csv("rep_units")
    for r in d["u"].itertuples():
        c = u[(u.universe == r.universe) & (u.level == "study") & (u.comparator == "universe") & (u.unit == r.unit)]
        assert c.enrichment.iloc[0] == r.enrichment or (np.isnan(c.enrichment.iloc[0]) and np.isnan(r.enrichment))


# ----------------------------------------------------------------------------- rendering + captions
NUM = re.compile(r"(?<![\w.$\\{])(\d+(?:\{,\}\d{3})*(?:\.\d+)?)(?![\w}])")
SCI = re.compile(r"(\d\.\d)\\times 10\^\{(-?\d+)\}")
ALLOWED = {"95", "90", "10--90", "1.96"}


def _values(obj, out):
    if isinstance(obj, pd.DataFrame):
        for c in obj.columns:
            v = pd.to_numeric(obj[c], errors="coerce").to_numpy(dtype=float)
            out.update(v[np.isfinite(v)].tolist())
    elif isinstance(obj, pd.Series):
        _values(obj.to_frame().T, out)
    elif isinstance(obj, dict):
        for v in obj.values():
            _values(v, out)
    elif isinstance(obj, (list, tuple, set)):
        for v in obj:
            _values(v, out)
    elif isinstance(obj, (int, float, np.integer, np.floating)) and np.isfinite(obj):
        out.add(float(obj))
    return out


def _matches(token: str, vals: np.ndarray) -> bool:
    t = token.replace("{,}", "")
    x = float(t)
    nd = len(t.split(".")[1]) if "." in t else 0
    for scale in (1.0, 100.0):
        r = np.round(vals * scale, nd)
        if np.any(np.isclose(r, x, atol=0.5 * 10 ** (-nd) + 1e-12)):
            return True
    return False


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    out = tmp_path_factory.mktemp("figs")
    res = {}
    for name in F.FIGURES:
        S.apply_style()
        data_fn, plot_fn, cap_fn = F.FIGURES[name]
        fd = data_fn()
        fig = plot_fn(fd)
        cap = cap_fn(fd)
        paths = F.save_figure(fig, fd, cap, out)
        res[name] = (fd, cap, paths)
    return res


@pytest.mark.parametrize("name", list(F.FIGURES))
def test_figure_renders(built, name):
    from PIL import Image

    fd, cap, paths = built[name]
    assert paths["pdf"].stat().st_size > 5_000
    assert paths["pdf"].read_bytes()[:4] == b"%PDF"
    with Image.open(paths["png"]) as im:
        w, h = im.size
        dpi = im.info.get("dpi", (0, 0))[0]
    assert abs(w - round(S.FULL_W * S.DPI_PNG)) <= 2
    assert round(dpi) == S.DPI_PNG
    txt = paths["caption"].read_text()
    assert txt.startswith("% provenance:")
    assert "data/r2/tables" in txt.splitlines()[0]


@pytest.mark.parametrize("name", list(F.FIGURES))
def test_caption_numbers_come_from_data(built, name):
    fd, cap, _ = built[name]
    assert not re.search(r"\bnan\b", cap.lower()) and "None" not in cap and "--]" not in cap and "[--" not in cap
    vals = np.array(sorted(_values(fd.d, set())), dtype=float)
    # scientific-notation p-values / probabilities
    for mant, exp in SCI.findall(cap):
        x = float(mant) * 10 ** int(exp)
        assert np.any(np.isclose(vals, x, rtol=0.051)), (name, mant, exp)
    body = SCI.sub(" ", cap)
    body = re.sub(r"\\(eqref|ref|label|cref|Cref)\{[^}]*\}", " ", body)
    for lab in sorted(S.REFERENCE_LABEL.values(), key=len, reverse=True):   # study names carry years
        body = body.replace(lab, " ")
    body = re.sub(r"\$[^$]*\$", lambda m: m.group(0) if "\\leq" in m.group(0) or "=" in m.group(0) else " ", body)
    bad = []
    for tok in NUM.findall(body):
        if tok in ALLOWED:
            continue
        t = tok.replace("{,}", "")
        if "." not in t and float(t) <= 10:      # panel counts, mode numbers, Q1..Q5
            continue
        if not _matches(tok, vals):
            bad.append(tok)
    assert not bad, f"{name}: caption numbers not found in the figure data: {bad}"


def test_cli_lists_subcommands():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import importlib

    m = importlib.import_module("r2.make_paper_assets")
    assert set(m.GROUPS) == {"tables", "figures", "digest", "numbers"}
    assert m.main(["--help"]) == 0
