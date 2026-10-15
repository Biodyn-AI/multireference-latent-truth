"""Tests for r2.paper_assets.figures_supp (supplementary figures) and r2.paper_assets.digest.

Run: .venv/bin/python -m pytest subproject_24_multireference_latent_truth_model/src/r2/tests/test_paper_assets_suppfig_digest.py -q
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from r2.paper_assets import digest as D  # noqa: E402
from r2.paper_assets import figures_supp as FS  # noqa: E402
from r2.paper_assets import style as S  # noqa: E402


# ----------------------------------------------------------------------------- supplementary figures
@pytest.fixture(scope="module")
def built_figs(tmp_path_factory):
    out = tmp_path_factory.mktemp("figs")
    return out, FS.build(out_dir=out)


def test_every_supplementary_figure_renders(built_figs):
    out, res = built_figs
    assert set(res) == set(FS.FIGURES)
    for name in FS.FIGURES:
        for ext in (".pdf", ".png"):
            p = out / f"{name}{ext}"
            assert p.exists() and p.stat().st_size > 5_000, p
        cap = (out / f"{name}_caption.txt").read_text()
        assert cap.startswith("% provenance:")
        # every declared source CSV is named in the provenance
        for csv in FS.FIGURE_SOURCES[name]:
            assert csv in cap, (name, csv)
        # the LaTeX float wrapper: one figure float, the PDF, a LaTeX-safe caption and the fixed label
        flt = (out / f"{name}_float.tex").read_text()
        body = "\n".join(ln for ln in flt.splitlines() if not ln.startswith("%"))
        assert body.count("\\begin{figure}") == 1 and body.count("\\end{figure}") == 1
        assert f"{{{name}.pdf}}" in body
        assert f"\\label{{{S.FIG_LABEL[name]}}}" in body
        caption = body.split("\\caption{", 1)[1].rsplit("}\n  \\label", 1)[0]
        assert not S.latex_caption_problems(caption), (name, S.latex_caption_problems(caption))


def test_latex_caption_checker():
    assert not S.latex_caption_problems("95\\% interval, $\\alpha \\in \\{0, 1\\}$")
    assert S.latex_caption_problems("95% interval")
    assert S.latex_caption_problems("penalty α")
    assert S.latex_caption_problems("$x$ and $y")
    assert S.latex_caption_problems("{unbalanced")


def test_figure_labels_cover_every_figure():
    from r2.paper_assets import figures as F
    assert set(S.FIG_LABEL) == set(F.FIGURES) | set(FS.FIGURES)
    assert len(set(S.FIG_LABEL.values())) == len(S.FIG_LABEL)


def test_png_resolution_and_width(built_figs):
    from PIL import Image

    out, _ = built_figs
    for name in FS.FIGURES:
        im = Image.open(out / f"{name}.png")
        dpi = im.info.get("dpi", (0, 0))[0]
        assert abs(dpi - S.DPI_PNG) < 1, (name, dpi)
        width_cm = im.size[0] / S.DPI_PNG * 2.54
        assert width_cm <= 16.05, (name, width_cm)          # never wider than the text block
        assert width_cm >= 7.7, (name, width_cm)            # at least half width


def test_pdf_is_vector_with_provenance(built_figs):
    out, _ = built_figs
    for name in FS.FIGURES:
        raw = (out / f"{name}.pdf").read_bytes()
        assert raw.startswith(b"%PDF")
        assert b"/Subject" in raw                           # provenance embedded in the metadata
        assert b"/Subtype /Image" not in raw or name == "figS4_gov_grid"   # only the heat map embeds a raster


def test_gov_grid_q_strong_invariance_matches_csv():
    """The caption states that O/E does not depend on q_strong; check it on the CSV."""
    g = pd.read_csv(S.TABLES / "gov_sensitivity_grid.csv")
    g = g[g.family == "q_grid"]
    spread = g.groupby(["tissue", "q_moderate", "reference"]).oe_tf.agg(lambda s: s.max() - s.min())
    assert float(spread.max()) == pytest.approx(0.0, abs=1e-12)


def test_recovery_calibration_bins_are_bins_1_to_9():
    d = pd.read_csv(S.TABLES / "fit_recovery_calibration.csv")
    sel = d[d.in_ece_bins1_9.map(S.as_bool)]
    assert set(sel.bin) == set(range(1, 10))


# ----------------------------------------------------------------------------- digest
@pytest.fixture(scope="module")
def built_digest(tmp_path_factory):
    out = tmp_path_factory.mktemp("digest")
    res = D.build(out_dir=out)
    return out, res


def test_digest_has_no_missing_fact(built_digest):
    out, res = built_digest
    missing = [f"{f.id}: {f.error}" for f in res["facts"].values() if f.error]
    assert not missing, missing
    md = (out / "results_digest.md").read_text()
    assert "MISSING" not in md
    for sec, title, _ in D.SECTIONS:
        assert f"## {sec} {title}" in md


def _num(s: str) -> float:
    return float(s.replace(",", "").replace("%", "").strip())


def test_scalar_facts_round_trip_to_csv(built_digest):
    """Single-row facts: the printed value equals the CSV value at the printed precision."""
    _, res = built_digest
    checked = 0
    for f in res["facts"].values():
        if f.by or f.agg or f.derive is not None or f.fmt in ("str", "p") or f.fmt.startswith(("e", "sig")):
            continue
        df = pd.read_csv(S.TABLES / f.csv, low_memory=False)
        sel = df.query(f.query, engine="python") if f.query else df
        x = float(sel[f.col].iloc[0]) * (f.mult if f.mult else 1.0)
        if f.fmt == "int":
            assert _num(f.value) == round(x), f.id
        elif f.fmt.startswith("pct"):
            assert _num(f.value) == pytest.approx(100 * x, abs=10 ** -int(f.fmt[3:]) / 2 + 1e-12), f.id
        elif f.fmt.startswith("r"):
            d = int(f.fmt[1:])
            assert _num(f.value) == pytest.approx(x, abs=0.5 * 10 ** -d + 1e-12), f.id
        checked += 1
    assert checked >= 30


def test_grouped_facts_round_trip(built_digest):
    """Grouped facts: every group's printed value equals the CSV value."""
    _, res = built_digest
    for fid in ["hb_gap", "nuts_rhat_id", "gov_prom", "ho_ap_hb", "rep_enr", "sim_hb_or"]:
        f = res["facts"][fid]
        df = pd.read_csv(S.TABLES / f.csv, low_memory=False)
        sel = df.query(f.query, engine="python") if f.query else df
        by = [f.by] if isinstance(f.by, str) else f.by
        for _, row in sel.iterrows():
            lab = "/".join(D._disp(str(row[c])) for c in by)
            exp = D._fmt(row[f.col], f.fmt)
            assert f"{lab} {exp}" in f.value, (fid, lab, exp, f.value)


def test_prose_has_no_hand_typed_numbers():
    """Interpretive templates may contain numbers only inside fact placeholders or in
    metric/assay/identifier names (Enr@500, P@500, 1/L, top-50, K562, THP-1, F1, R1, Q-numbers)."""
    allowed = re.compile(
        r"\{[VC]\[[A-Za-z0-9_]+\]\}|Enr@500|P@500|1/L|top-500|top-50|top-1|K562|THP-1|F1|R1|ENETS2|"
        r"sim-M1|O\(PKS\)|L[123]O|\b5\.\d+\b|Q\d+|10×|GRNBoost2")
    for sec, pr in D.PROSE.items():
        for txt in pr.get("interp", []) + pr.get("caveats", []):
            stripped = allowed.sub("", txt)
            digits = re.findall(r"\d", stripped)
            assert not digits, (sec, stripped)


def test_prose_placeholders_refer_to_existing_facts():
    ids = {f.id for lst in D._facts().values() for f in lst}
    for sec, pr in D.PROSE.items():
        for txt in pr.get("interp", []) + pr.get("caveats", []):
            for fid in re.findall(r"\{[VC]\[([A-Za-z0-9_]+)\]\}", txt):
                assert fid in ids, (sec, fid)


def test_reviewer_points_cover_q1_to_q24(built_digest):
    out, res = built_digest
    md = (out / "reviewer_points.md").read_text()
    for i in range(1, 25):
        assert re.search(rf"^## Q{i}: ", md, re.M), i
    for q, (secs, fids, _) in D.REVIEWER_MAP.items():
        for fid in fids:
            assert fid in res["facts"], (q, fid)
        for s in secs:
            assert s in dict((a, b) for a, b, _ in D.SECTIONS), (q, s)


def test_freshness_flags_regenerated_tables():
    rows = D.freshness({"heldout_estimators.csv", "fit_convergence.csv"})
    d = {r["csv"]: r for r in rows}
    assert d["heldout_estimators.csv"]["regenerated"] is True
    assert d["fit_convergence.csv"]["regenerated"] is False


def test_fact_formatting():
    assert D._fmt(1234.4, "int") == "1,234"
    assert D._fmt(0.12345, "r2") == "0.12"
    assert D._fmt(0.0123, "pct1") == "1.2%"
    assert D._fmt(np.nan, "r2") == "NA"
    assert D._fmt(3.1e-16, "p") == "3.1e-16"


def test_reviewer_notes_have_no_hand_typed_numbers():
    allowed = re.compile(r"figS\d+|\+/-1|±1|R1|L[123]O|Q\d+|5\.\d+|F1|GRNBoost2|GENIE3|ENETS2|K562|Table S\d+|src/r2")
    for q, (_, _, note) in D.REVIEWER_MAP.items():
        stripped = allowed.sub("", note)
        assert not re.findall(r"\d", stripped), (q, stripped)


def test_derived_mode_selection_facts(built_digest):
    """Derived facts recomputed from the replicate table: gap-threshold counts, largest wrong-selection gap,
    bound-based likelihood ratio and misselection rates."""
    _, res = built_digest
    f = res["facts"]
    r = pd.read_csv(S.TABLES / "modes_simulation_replicates.csv", low_memory=False)
    r = r[r.simulator.isin(["faithful_cc", "faithful_cc_strat"])]
    ok = r.dominant_is_truth.map(S.as_bool)
    ge = r.gap.astype(float) >= D.GAP_SAFE
    assert f["msim_ge_thr"].value == f"{int((ge & ok).sum())} of {int(ge.sum())}"
    assert f["msim_nfaith"].value == f"{len(r):,}"
    assert float(f["msim_wrong_maxgap"].value) == pytest.approx(r.loc[~ok, "gap"].max(), abs=0.005)
    # every wrong selection lies below the threshold (the statement the digest makes)
    assert r.loc[~ok, "gap"].max() < D.GAP_SAFE
    m = pd.read_csv(S.TABLES / "modes_simulation.csv", low_memory=False)
    q = m[(m.stratum == "real_gap_immune") & (m.simulator == "faithful_cc")]
    a = q[(q.universe == "immune") & (q.quantity == "p_truth_beats_other_mode_by_ge_real_gap")].value.iloc[0]
    b = q[(q.universe == "immune_truthmode1") & (q.quantity == "p_other_mode_beats_truth_by_ge_real_gap")].ci_hi.iloc[0]
    assert f"design-faithful {a / b:.1f}" in f["msim_lr"].value
    pd_ = m[(m.stratum == "all") & (m.quantity == "p_dominant_is_truth") & (m.simulator == "faithful_cc")]
    for u, lab in (("immune", "Immune"), ("immune_truthmode1", "Immune (truth = real second mode)")):
        x = 1.0 - pd_[pd_.universe == u].value.iloc[0]
        assert f"{lab} {x:.2f}" in f["msim_miss"].value


def test_digest_robustness_em_reference_and_lineage(built_digest):
    """EM-referenced contrasts come from the reference == 'em' rows; the lineage switch-only mean equals the
    mean over the identified switch configurations of rob_config_detail."""
    _, res = built_digest
    f = res["facts"]
    c = pd.read_csv(S.TABLES / "rob_estimator_contrasts.csv", low_memory=False)
    r = c[(c.scope == "all_tissues") & (c.protocol == "source") & (c.metric == "D_f1") & (c.reference == "em")
          & (c.subset == "all") & (c.estimator == "family_hb")].iloc[0]
    assert f["rob_em_fhb"].full == (f"{r.mean_diff:.4f} [{r.mean_diff_cluster_lo:.4f}, {r.mean_diff_cluster_hi:.4f}]")
    # every Family-HB-referenced fact filters on the reference column (the table now holds EM-referenced rows)
    for fact in f.values():
        if fact.csv == "rob_estimator_contrasts.csv":
            assert "reference ==" in (fact.query or ""), fact.id
    d = pd.read_csv(S.TABLES / "rob_config_detail.csv", low_memory=False)
    ident = d[(d.protocol == "lineage") & (d.estimator == "family_hb") & d.identified.map(S.as_bool)][["universe", "config"]]
    lin = d[(d.protocol == "lineage")].merge(ident, on=["universe", "config"])
    sw = lin[lin.fhb_mode_switch.map(S.as_bool)]
    for est, lab in (("family_hb", "Family-HB"), ("em", "EM baseline")):
        x = sw[sw.estimator == est].D_f1.astype(float).mean()
        assert f"{lab} {x:.3f}" in f["rob_lin_swD"].value, (est, x, f["rob_lin_swD"].value)


def test_digest_nuts_projection_matches_scaling_table(built_digest):
    """The digest's projected NUTS range is the range of tab:compute-scaling (min step projection, max cap)."""
    _, res = built_digest
    n = pd.read_csv(S.TABLES / "scale_nuts.csv", low_memory=False)
    n = n[n.status == "projected"]
    for cid, lab in (("template_10xreal_K21", "10x size, K=21"), ("template_10xreal_K42", "10x size, K=42")):
        lo = n[n.cid == cid].t_nuts_projected_s.min()
        hi = n[n.cid == cid].t_nuts_projected_cap_s.max()
        assert f"{lab} {int(round(lo)):,}" in res["facts"]["sc_nuts_lo"].value
        assert f"{lab} {int(round(hi)):,}" in res["facts"]["sc_nuts_hi"].value
