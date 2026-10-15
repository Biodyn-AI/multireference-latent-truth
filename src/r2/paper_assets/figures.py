"""Main figures fig1..fig8 of the SAGMB R2 manuscript.

Every plotted or printed number is read from ``data/r2/tables/*.csv`` (design
constants such as the top-k size and the BFDR thresholds come from
``r2.common.PRESPECIFIED`` or from the CSVs).  Each figure has

* ``data_<name>()``  -- returns a dict of tidy DataFrames read from the CSVs
  (used by the plot function, the caption and the tests), and records the
  source files and row filters in a :class:`style.Provenance`;
* ``plot_<name>(data)`` -- draws the figure;
* ``caption_<name>(data)`` -- the caption draft with numbers from the data.

``make(name)`` writes ``paper/r2/figures/<name>.pdf`` (vector), ``<name>.png``
(300 dpi), ``<name>_caption.txt`` (provenance comment + caption) and the LaTeX
float wrapper ``<name>_float.tex`` (``style.write_float``; fixed label of
``style.FIG_LABEL``).

CLI: ``python -m r2.paper_assets.figures [names...]`` (from ``src/``) or
``python src/r2/make_paper_assets.py figures``.
"""
from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle  # noqa: E402
from matplotlib.ticker import FixedLocator, NullFormatter, NullLocator  # noqa: E402

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from r2.paper_assets import style as S  # type: ignore
else:
    from . import style as S

try:  # design constants (top-k, BFDR thresholds, replication endpoint)
    from r2.common import PRESPECIFIED as _PRE
except Exception:  # pragma: no cover - package layout fallback
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from r2.common import PRESPECIFIED as _PRE

OI = S.OI
FS = S.FONT_SIZES


def _module_constant(module: str, name: str):
    """A design constant of an analysis module (read, never edited, by the asset scripts)."""
    import importlib
    return getattr(importlib.import_module(f"r2.{module}"), name)


@dataclass
class FigureData:
    name: str
    prov: S.Provenance
    d: dict


# ============================================================================ helpers
def _fig(width_in: float, height_cm: float):
    return plt.figure(figsize=(width_in, height_cm * S.CM))


def panel_label(ax, letter: str, x: float = -0.02, y: float = 1.0, fig=None, **kw):
    """Bold lower-case panel letter at the top-left corner of ``ax`` (axes fraction)."""
    kw.setdefault("ha", "right")
    kw.setdefault("va", "bottom")
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=FS["panel"], fontweight="bold", **kw)


def fig_label(fig, letter: str, x: float, y: float):
    """Panel letter at figure coordinates."""
    fig.text(x, y, letter, fontsize=FS["panel"], fontweight="bold", ha="left", va="top")


def _err(v, lo, hi):
    v, lo, hi = (np.asarray(a, dtype=float) for a in (v, lo, hi))
    lo_e = np.where(np.isfinite(lo), np.clip(v - lo, 0, None), 0.0)
    hi_e = np.where(np.isfinite(hi), np.clip(hi - v, 0, None), 0.0)
    return np.vstack([lo_e, hi_e])


def point_ci(ax, x, y, lo=None, hi=None, *, color="k", marker="o", filled=True, ms=3.6,
             horizontal=True, lw=0.8, zorder=3, alpha=1.0, mew=0.8, label=None):
    """A point estimate with its CI as an error bar (horizontal CI by default)."""
    x = np.atleast_1d(np.asarray(x, dtype=float))
    y = np.atleast_1d(np.asarray(y, dtype=float))
    if lo is not None and hi is not None:
        if horizontal:
            e = _err(x, lo, hi)
            ax.errorbar(x, y, xerr=e, fmt="none", ecolor=color, elinewidth=lw, alpha=alpha,
                        zorder=zorder - 1, capsize=0)
        else:
            e = _err(y, lo, hi)
            ax.errorbar(x, y, yerr=e, fmt="none", ecolor=color, elinewidth=lw, alpha=alpha,
                        zorder=zorder - 1, capsize=0)
    ax.plot(x, y, ls="none", marker=marker, ms=ms, mec=color, mew=mew,
            mfc=color if filled else "white", alpha=alpha, zorder=zorder, label=label)


def ref_line(ax, value=1.0, orientation="v", **kw):
    kw.setdefault("color", OI["grey"])
    kw.setdefault("lw", 0.6)
    kw.setdefault("ls", "-")
    kw.setdefault("zorder", 0.5)
    if orientation == "v":
        ax.axvline(value, **kw)
    else:
        ax.axhline(value, **kw)


def _fmt_tick(v: float) -> str:
    if v >= 1 and abs(v - round(v)) < 1e-9:
        return f"{int(round(v))}"
    s = f"{v:.3g}"
    return s


def set_log_ticks(ax, ticks, axis="x", labels=None):
    ticks = list(ticks)
    labels = labels or [_fmt_tick(t) for t in ticks]
    if axis == "x":
        ax.set_xscale("log")
        ax.xaxis.set_major_locator(FixedLocator(ticks))
        ax.xaxis.set_minor_locator(NullLocator())
        ax.set_xticklabels(labels)
    else:
        ax.set_yscale("log")
        ax.yaxis.set_major_locator(FixedLocator(ticks))
        ax.yaxis.set_minor_locator(NullLocator())
        ax.set_yticklabels(labels)


def nice_log_ticks(lo: float, hi: float, candidates=(0.01, 0.02, 0.05, 0.1, 0.2, 0.25, 0.5, 1, 2, 4, 5,
                                                       10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000)):
    t = [c for c in candidates if lo <= c <= hi]
    return t


def _row_axes_labels(ax, ypos, labels, fontsize=None, colors=None, styles=None):
    ax.set_yticks(ypos)
    ax.set_yticklabels(labels, fontsize=fontsize or FS["small"] + 0.5)
    if colors or styles:
        for i, t in enumerate(ax.get_yticklabels()):
            if colors and colors[i]:
                t.set_color(colors[i])
            if styles and styles[i]:
                t.set_fontstyle(styles[i])
    ax.tick_params(axis="y", length=0)


def _hide_y(ax):
    ax.tick_params(axis="y", left=False, labelleft=False)
    ax.spines["left"].set_visible(False)


def _num(x, d=2):
    return S.fmt_ratio(x, d)


def _ci(lo, hi, d=2):
    return f"[{S.fmt_ratio(lo, d)}, {S.fmt_ratio(hi, d)}]"


def _one(df: pd.DataFrame, **conds) -> pd.Series:
    m = np.ones(len(df), dtype=bool)
    for k, v in conds.items():
        m &= (df[k] == v).to_numpy()
    sub = df[m]
    if len(sub) != 1:
        raise KeyError(f"expected exactly one row for {conds}, got {len(sub)}")
    return sub.iloc[0]


def _maybe(df: pd.DataFrame, **conds):
    try:
        return _one(df, **conds)
    except KeyError:
        return None


def save_figure(fig, fd: FigureData, caption: str, out_dir: Path | None = None) -> dict:
    out_dir = Path(out_dir or S.FIG_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf = out_dir / f"{fd.name}.pdf"
    png = out_dir / f"{fd.name}.png"
    prov_txt = fd.prov.text()
    fig.savefig(pdf, metadata={"Title": fd.name, "Subject": prov_txt,
                               "Creator": "src/r2/paper_assets/figures.py", "CreationDate": None})
    fig.savefig(png, dpi=S.DPI_PNG, metadata={"Title": fd.name, "Description": prov_txt,
                                               "Software": "src/r2/paper_assets/figures.py"})
    width_in = float(fig.get_size_inches()[0])
    plt.close(fig)
    cap = out_dir / f"{fd.name}_caption.txt"
    cap.write_text(fd.prov.tex_comment() + caption.strip() + "\n")
    flt = S.write_float(fd.name, caption, fd.prov, width_in, out_dir)
    return {"pdf": pdf, "png": png, "caption": cap, "float": flt}


# ============================================================================ fig1 design
def data_fig1_design() -> FigureData:
    P = S.Provenance("fig1_design")
    src = P.read("source_table", "held_out (False = held-in annotator; True = held-out ChIP); family")
    ev = P.read("evidence_summary", "variant = mask = prespecified_cc, annotator = __ALL__")
    ev = ev[(ev.variant == "prespecified_cc") & (ev["mask"] == "prespecified_cc") & (ev.annotator == "__ALL__")]
    panels = P.read("universe_panels", "all rows (tissue universes)")
    chip = P.read("heldout_chip_table", "all rows")
    coh = P.read("cohort_table", "all rows")
    tiers = P.read("gov_tiers", "scope = dual, tier = Promoted; q_strong, q_moderate")
    tv = P.read("gov_tier_validation", "universe = global, reference = encode, tier in {Strong, Moderate, Promoted}")
    held_in = src[~src.held_out.map(S.as_bool)]
    held_out = src[src.held_out.map(S.as_bool)]
    fam_counts = held_in.groupby("family").size().to_dict()
    glob = _one(ev, universe="global")
    gtv = tv[(tv.universe == "global") & (tv.reference == "encode")]
    d = dict(
        n_held_in=len(held_in), fam_counts=fam_counts, n_chip=len(held_out),
        chip_names=list(held_out.source_id),
        burst_families=[f for f, n in fam_counts.items() if n >= 2],
        glob=dict(n_tfs=int(glob.n_tfs), n_genes=int(glob.n_genes), n_edges=int(glob.n_edges),
                  n_patterns=int(glob.n_patterns), n_reported=int(glob.n_pos),
                  coverage=float(glob.coverage)),
        tissues={r.tissue: dict(n_tfs=int(r.n_tfs), n_genes=int(r.n_panel), n_edges=int(r.n_universe))
                 for r in panels.itertuples()},
        chip={r.key: dict(n_tfs=int(r.n_tfs), density=float(r.density)) for r in chip.itertuples()},
        n_cohorts=len(coh), n_studies=coh.study_id.nunique(),
        cohorts=coh[["cohort_id", "study_id", "modality", "cell_type_group", "n_perturbed_tfs"]].copy(),
        q_strong=float(tiers.q_strong.iloc[0]), q_moderate=float(tiers.q_moderate.iloc[0]),
        promoted_ctx={r.tissue: int(r.n_edges) for r in tiers[(tiers.scope == "dual") & (tiers.tier == "Promoted")].itertuples()},
        glob_tiers={t: int(_one(gtv, tier=t).n_edges) for t in ("Strong", "Moderate", "Promoted")},
        topk=int(_PRE["topk_per_tf"]),
    )
    P.note("(constants)", "topk_per_tf from r2.common.PRESPECIFIED")
    return FigureData("fig1_design", P, d)


def _box(ax, x, y, w, h, text, *, fc="white", ec="k", lw=0.7, ls="-", fs=None, ha="center",
         va="center", weight="normal", round_=0.12, tc="k", style="normal", zorder=2, pad_x=0.12):
    p = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={round_}", fc=fc, ec=ec,
                       lw=lw, ls=ls, zorder=zorder)
    ax.add_patch(p)
    if text:
        tx = x + w / 2 if ha == "center" else (x + pad_x if ha == "left" else x + w - pad_x)
        ty = y + h / 2 if va == "center" else (y + h - 0.12 if va == "top" else y + 0.12)
        ax.text(tx, ty, text, ha=ha, va=va, fontsize=fs or FS["small"] + 0.5, color=tc,
                fontweight=weight, fontstyle=style, zorder=zorder + 1, linespacing=1.15)
    return p


def _arrow(ax, x0, y0, x1, y1, *, color="k", lw=0.8, ls="-", style="-|>", ms=6, zorder=1,
           connectionstyle="arc3,rad=0"):
    a = FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style, mutation_scale=ms, color=color, lw=lw,
                        ls=ls, zorder=zorder, shrinkA=0, shrinkB=0, connectionstyle=connectionstyle)
    ax.add_patch(a)
    return a


LINE_H = 0.33      # cm per 7-pt text line (line spacing 1.2)
TITLE_H = 0.40


def _textbox(ax, x, ytop, w, title, lines, *, ec="k", fc="white", ls="-", lw=0.7, italic_from=None,
             min_h=None, title_color="k"):
    """Box whose height follows its content; returns (ybottom, ycentre)."""
    n = len(lines)
    h = 0.14 + (TITLE_H if title else 0) + n * LINE_H + 0.10
    if min_h:
        h = max(h, min_h)
    _box(ax, x, ytop - h, w, h, "", ec=ec, fc=fc, ls=ls, lw=lw)
    y = ytop - 0.12
    fs = FS["small"] + 0.5
    if title:
        ax.text(x + 0.14, y, title, ha="left", va="top", fontsize=fs, fontweight="bold", color=title_color)
        y -= TITLE_H
    for i, ln in enumerate(lines):
        st = "italic" if (italic_from is not None and i >= italic_from) else "normal"
        ax.text(x + 0.14, y, ln, ha="left", va="top", fontsize=fs, fontstyle=st)
        y -= LINE_H
    return ytop - h, ytop - h / 2


def plot_fig1_design(fd: FigureData):
    d = fd.d
    W, H = S.TEXT_WIDTH_CM, 7.45
    fig = _fig(S.FULL_W, H)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")
    c = S.fmt_count_plain
    cols = [(0.10, 3.45), (3.95, 3.55), (7.95, 3.95), (12.3, 3.45)]
    heads = ["Held-in evidence", "Candidate universes", "Model and claim governance", "Held-out data"]
    for (x, w), t in zip(cols, heads):
        ax.text(x, H - 0.25, t, ha="left", va="center", fontsize=FS["title"], fontweight="bold")
    top = H - 0.6

    # column 1: annotators by family + held-out ChIP family
    x, w = cols[0]
    fc = d["fam_counts"]
    fam_desc = {"LIT": "curated, text-mined", "MOTIF": "binding-site scans", "COEXP": "GTEx regulons"}
    fbh = 0.70
    hh = 0.14 + TITLE_H + 3 * (fbh + 0.1) + 0.02
    _box(ax, x, top - hh, w, hh, "", ec="k")
    ax.text(x + 0.14, top - 0.12, f"{d['n_held_in']} annotators", ha="left", va="top", fontsize=FS["small"] + 0.5,
            fontweight="bold")
    yb = top - 0.14 - TITLE_H
    for f in ("LIT", "MOTIF", "COEXP"):
        _box(ax, x + 0.12, yb - fbh, w - 0.24, fbh, "", fc=S.FAMILY_COLOR[f] + "22", ec=S.FAMILY_COLOR[f], lw=0.9)
        ax.text(x + 0.25, yb - 0.08, f"{S.FAMILY_LABEL[f]}: {fc.get(f, 0)}", ha="left", va="top",
                fontsize=FS["small"] + 0.5, fontweight="bold")
        ax.text(x + 0.25, yb - 0.08 - LINE_H, fam_desc[f], ha="left", va="top", fontsize=FS["small"] + 0.5)
        yb -= fbh + 0.1
    y_ann_bottom = top - hh
    ychip_b, ychip_c = _textbox(ax, x, y_ann_bottom - 0.3, w, f"ChIP: {d['n_chip']} resources",
                                ["held out of every", "primary fit"], ec=S.FAMILY_COLOR["CHIP"],
                                fc=S.FAMILY_COLOR["CHIP"] + "18", ls="--", lw=0.9)

    # column 2: universes
    x2, w2 = cols[1]
    g = d["glob"]
    gb, gc = _textbox(ax, x2, top, w2, "Global universe",
                      [f"{c(g['n_tfs'])} TFs $\\times$ {c(g['n_genes'])} genes",
                       f"{c(g['n_edges'])} ordered pairs",
                       f"{c(g['n_patterns'])} observation patterns",
                       f"{c(g['n_reported'])} reported edges"])
    tl = [f"{S.UNIVERSE_LABEL[t]}: {c(d['tissues'][t]['n_tfs'])} TFs $\\times$ {c(d['tissues'][t]['n_genes'])} genes"
          for t in ("immune", "kidney", "lung")]
    tb, tc = _textbox(ax, x2, gb - 0.3, w2, "Tissue-context universes",
                      tl + [f"top-{d['topk']} predicted sets:", "scGPT, Geneformer attention,",
                            "GRNBoost2, correlation"], italic_from=3)
    # column 3: model chain
    x3, w3 = cols[2]
    chain = [
        ("Family-HB latent-class model",
         ["classes T, N, " + ", ".join("B$_{\\mathrm{%s}}$" % f for f in d["burst_families"])], OI["vermillion"]),
        ("Credibility", ["$c_i=P(z_i=\\mathrm{T}\\mid\\mathrm{data})$"], "k"),
        ("BFDR tiers", [f"Strong $\\leq$ {S.fmt_ratio(d['q_strong'])}, Moderate $\\leq$ {S.fmt_ratio(d['q_moderate'])}"], "k"),
        ("Dual-universe rule", ["global fit $\\times$ context fit"], "k"),
    ]
    y = top
    cens = []
    for i, (t, ls_, ec) in enumerate(chain):
        yb3, yc3 = _textbox(ax, x3, y, w3, t, ls_, ec=ec, lw=0.9 if ec != "k" else 0.7)
        cens.append((y, yb3, yc3))
        if i:
            _arrow(ax, x3 + w3 / 2, cens[i - 1][1], x3 + w3 / 2, y)
        y = yb3 - 0.28
    pr = d["promoted_ctx"]
    pb, pc = _textbox(ax, x3, y, w3, "Promoted claims",
                      ["Strong or Moderate edges:",
                       f"global {c(d['glob_tiers']['Promoted'])}; immune {c(pr['immune'])};",
                       f"kidney {c(pr['kidney'])}; lung {c(pr['lung'])}"],
                      ec=OI["vermillion"], fc=OI["vermillion"] + "1A", lw=0.9)
    _arrow(ax, x3 + w3 / 2, cens[-1][1], x3 + w3 / 2, y)

    # arrows: evidence -> universes -> model
    _arrow(ax, x + w, gc, x2, gc)
    _arrow(ax, x + w, y_ann_bottom + 0.35, x2, tc + 0.4)
    _arrow(ax, x2 + w2, gc, x3, cens[0][2])
    _arrow(ax, x2 + w2, tc, x3, cens[0][2] - 0.15)

    # column 4: held-out references and evaluation
    x4, w4 = cols[3]
    ch = d["chip"]
    cb, cc = _textbox(ax, x4, top, w4, "ChIP-seq references",
                      [f"ENCODE TFBS: {ch['encode_tfbs']['n_tfs']} TFs",
                       f"ENCODE-strict: {ch['encode_strict']['n_tfs']} TFs",
                       f"ENETS2: {ch['enets2']['n_tfs']} TFs"], ls="--")
    kb, kc = _textbox(ax, x4, cb - 0.3, w4, "CRISPR screens",
                      [f"{d['n_cohorts']} cohorts, {d['n_studies']} studies", "single-cell readout"], ls="--")
    eb, ec_ = _textbox(ax, x4, kb - 0.45, w4, "Evaluation only",
                       ["AP lift, Enr@$k$,", "within-TF AUROC,", "TF-matched O/E,", "replication check"],
                       fc="#F2F2F2", ec=OI["grey"])
    xb = x4 + w4 + 0.07
    for yy_ in (cc, kc):
        ax.plot([x4 + w4, xb], [yy_, yy_], color=OI["grey"], lw=0.8, ls="--")
    ax.plot([xb, xb], [cc, ec_], color=OI["grey"], lw=0.8, ls="--")
    _arrow(ax, xb, ec_, x4 + w4, ec_, ls="--", color=OI["grey"])
    _arrow(ax, x3 + w3, pc, x4, ec_, ls="--", color=OI["grey"])
    # nothing flows from held-out data into the fit: note
    ax.text(x4 + w4 / 2, eb - 0.15, "no held-out data enter a\nprimary fit or universe", ha="center", va="top",
            fontsize=FS["small"] + 0.5, fontstyle="italic", color="k", linespacing=1.15)
    return fig



def caption_fig1_design(fd: FigureData) -> str:
    d = fd.d
    c = S.fmt_count
    g = d["glob"]
    fc = d["fam_counts"]
    t = d["tissues"]
    return (
        "\\textbf{Study design.} "
        f"{d['n_held_in']} held-in annotators in three evidence families (literature {fc.get('LIT', 0)}, "
        f"motif {fc.get('MOTIF', 0)}, coexpression {fc.get('COEXP', 0)}) report TF--target edges; the ChIP family "
        f"({d['n_chip']} resources) is held out of every primary fit. "
        f"The global universe contains {c(g['n_tfs'])} literature-covered TFs $\\times$ {c(g['n_genes'])} genes "
        f"({c(g['n_edges'])} ordered pairs, {c(g['n_patterns'])} distinct observation patterns, {c(g['n_reported'])} edges "
        "with at least one report); the tissue-context universes contain the pairs both foundation models can score "
        f"(immune {c(t['immune']['n_edges'])}, kidney {c(t['kidney']['n_edges'])}, lung {c(t['lung']['n_edges'])} pairs), "
        f"in which scGPT and Geneformer attention, GRNBoost2 and correlation define predicted sets (top {d['topk']} targets per TF). "
        "Family-HB assigns every edge a credibility; BFDR tiers "
        f"(Strong: BFDR $\\leq {S.fmt_ratio(d['q_strong'])}$; Moderate: $\\leq {S.fmt_ratio(d['q_moderate'])}$) and, in the tissue "
        "universes, the dual-universe rule turn credibilities into graded claims; Strong and Moderate edges are promoted "
        f"(global {c(d['glob_tiers']['Promoted'])}; immune {c(d['promoted_ctx']['immune'])}, kidney {c(d['promoted_ctx']['kidney'])}, "
        f"lung {c(d['promoted_ctx']['lung'])}). "
        f"Held-out ChIP-seq references (ENCODE TFBS, {d['chip']['encode_tfbs']['n_tfs']} TFs; ENCODE-strict, "
        f"{d['chip']['encode_strict']['n_tfs']}; ENETS2, {d['chip']['enets2']['n_tfs']}) and {d['n_cohorts']} CRISPR perturbation "
        f"cohorts from {d['n_studies']} studies enter no primary fit, universe definition or choice of the primary analysis (dashed: "
        "evaluation only); the tuning-optimism analysis uses perturbation labels only to measure what tuning on them "
        "would gain, and nothing it selects is used elsewhere."
    )


# ============================================================================ fig2 model
def data_fig2_model() -> FigureData:
    P = S.Provenance("fig2_model")
    sp = P.read("fit_source_parameters", "universe = global, fit_id = family_hb_nuts, parameter in {alpha, beta}")
    sp = sp[(sp.universe == "global") & (sp.fit_id == "family_hb_nuts") & sp.parameter.isin(["alpha", "beta"])]
    a = sp[sp.parameter == "alpha"].set_index("annotator")
    b = sp[sp.parameter == "beta"].set_index("annotator")
    ab = pd.DataFrame({
        "family": a.family, "alpha": a.value, "alpha_lo": a.lo, "alpha_hi": a.hi,
        "beta": b.value.reindex(a.index), "beta_lo": b.lo.reindex(a.index), "beta_hi": b.hi.reindex(a.index),
        "n_observed": a.n_observed, "n_reported": a.n_reported,
    }).reset_index()
    br = P.read("modes_bridge", "role = primary_estimate, mode = 1 (second mode); coords = centred, proposal = mvn")
    br = br[(br.role == "primary_estimate") & (br["mode"] == 1)]
    rng_cols = [c for c in ("p2_range_draw_sets_proposals_lo", "p2_range_draw_sets_proposals_hi") if c in br]
    modes = br[["universe", "gap_to_mode0", "p_bridge", "p2_bridge_lo", "p2_bridge_hi", "p_laplace_top2",
                "log_z_diff_bridge", "log_z_diff_laplace", "log_z_diff_sd", "n_draws", "nuts_settings"]
               + rng_cols].copy()
    modes["order"] = modes.universe.map({u: i for i, u in enumerate(S.UNIVERSE_ORDER)})
    modes = modes.sort_values("order").reset_index(drop=True)
    rep = P.read("modes_simulation_replicates", "simulator = faithful_cc (design-faithful); dominant_is_truth, "
                 "dominant_strong_realised_fdr")
    rep = rep[rep.simulator == "faithful_cc"] if "simulator" in rep else rep
    ms0 = P.read("modes_simulation", "stratum in {dominant_is_truth, dominant_not_truth, all}; "
                 "quantity in {dominant_strong_realised_fdr_mean, dominant_strong_claimed_bfdr_mean, p_dominant_is_truth}; "
                 "simulator = faithful_cc (primary); faithful_cc_strat and fixed_masks (sensitivity, caption only)")
    ms0 = ms0[ms0.quantity.isin(["dominant_strong_realised_fdr_mean", "dominant_strong_claimed_bfdr_mean",
                                 "p_dominant_is_truth", "oracle_strong_realised_fdr_mean"])
              & ms0.stratum.isin(["all", "dominant_is_truth", "dominant_not_truth"])]
    ms = ms0[ms0.simulator == "faithful_cc"] if "simulator" in ms0 else ms0
    sens = ms0[(ms0.simulator != "faithful_cc") & (ms0.stratum == "all") & (ms0.quantity == "p_dominant_is_truth")] \
        if "simulator" in ms0 else ms0.iloc[0:0]
    fd = P.read("modes_faithful_design", "quantity = n_edges_cc_observed, sim_label in {immune_cc, global_cc}: "
                "ratio_sim_mean_to_real (share of the real motif/coexpression-observed edges)")
    fd = fd[(fd.quantity == "n_edges_cc_observed") & fd.sim_label.isin(["immune_cc", "global_cc"])]
    cc_share = {str(r.universe): float(r.ratio_sim_mean_to_real) for r in fd.itertuples()}
    fm = P.read("fit_modes", "w_T and w_2 (= class B_LIT; class order T, N, B_LIT, B_MOTIF as in fit_class_weights) "
                "of optimum 0 and 1")
    swap = {}
    for u in fm.universe.unique():
        a, b = fm[(fm.universe == u) & (fm.optimum == 0)], fm[(fm.universe == u) & (fm.optimum == 1)]
        if len(a) and len(b):
            a, b = a.iloc[0], b.iloc[0]
            swap[u] = bool(abs(b.w_T - a.w_2) / a.w_2 < 0.05 and abs(b.w_2 - a.w_T) / a.w_T < 0.05)
    ph_b = bool((br.analysis_status == "post_hoc").all()) if "analysis_status" in br else False
    ph_c = bool((ms.analysis_status == "post_hoc").all()) if "analysis_status" in ms else False
    rep = rep[["universe", "replicate", "dominant_is_truth", "dominant_strong_realised_fdr",
               "dominant_strong_claimed_bfdr", "dominant_strong_n", "gap"]].copy()
    rep["dominant_is_truth"] = rep.dominant_is_truth.map(S.as_bool)
    return FigureData("fig2_model", P, dict(ab=ab, modes=modes, rep=rep, ms=ms, sens=sens, swap=swap, ph_b=ph_b,
                                            ph_c=ph_c, q_strong=float(_PRE["bfdr_strong"]), cc_share=cc_share))


SIM_UNIVERSE_LABEL = {"immune": "Immune\nmode 1", "immune_truthmode1": "Immune\nmode 2",
                      "global": "Global\nmode 1"}
SIM_UNIVERSE_TEXT = {"immune": "immune universe, truth = mode 1", "immune_truthmode1": "immune universe, truth = mode 2",
                     "global": "global universe, truth = mode 1"}


def _sim_universe_order(us):
    pref = ["global", "immune", "immune_truthmode1"]
    return [u for u in pref if u in us] + sorted(u for u in us if u not in pref)


# label offsets (points) for the annotators labelled in fig2a; others default to the right
FIG2_DISPLAY = {"LIT_other": "Other\nliterature"}
FIG2_LABEL_OFFSETS = {"TRRUST": (-5, 0, "right", "center"), "TRED": (0, 5, "center", "bottom"),
                      "LIT_other": (0, 9, "center", "bottom"), "NTNU.Curated": (0, -6, "center", "top"),
                      "GOA": (5, 0, "left", "center"), "JASPAR": (-9, 2, "right", "center"),
                      "HOCOMOCO": (2, -6, "left", "top"), "ARACNe-GTEx": (5, 0, "left", "center")}


def plot_fig2_model(fd: FigureData):
    d = fd.d
    fig = _fig(S.FULL_W, 7.2)
    fsm = FS["small"] + 0.5
    # a: alpha vs beta
    axa = fig.add_axes([0.075, 0.13, 0.30, 0.82])
    ab = d["ab"]
    for f in ["LIT", "MOTIF", "COEXP"]:
        s = ab[ab.family == f]
        if s.empty:
            continue
        col = S.FAMILY_COLOR[f]
        axa.errorbar(s.beta, s.alpha, xerr=_err(s.beta, s.beta_lo, s.beta_hi),
                     yerr=_err(s.alpha, s.alpha_lo, s.alpha_hi), fmt="none", ecolor=col, elinewidth=0.6, zorder=2)
        axa.plot(s.beta, s.alpha, ls="none", marker=S.FAMILY_MARKER[f], ms=3.6, mfc=col, mec="white", mew=0.4,
                 zorder=3, label=f"{S.FAMILY_LABEL[f]} ({len(s)})")
    xlo = 10 ** math.floor(math.log10(ab.beta_lo.min()))
    ylo = 10 ** (math.floor(math.log10(ab.alpha_lo.min()) * 2) / 2)
    axa.plot([xlo, 1], [xlo, 1], color=OI["grey"], lw=0.6, ls="--", zorder=1)
    xt = 0.35
    axa.text(xt, xt * 1.25, "$\\alpha_j=\\beta_j$", fontsize=fsm, color=OI["grey"], ha="right", va="bottom",
             rotation=0)
    axa.set_xlim(xlo, 1)
    axa.set_ylim(ylo, 1)
    xt_ = [10.0 ** e for e in range(int(round(math.log10(xlo))), 1)]
    axa.set_xscale("log")
    axa.xaxis.set_major_locator(FixedLocator(xt_))
    axa.xaxis.set_minor_locator(NullLocator())
    axa.set_xticklabels([f"$10^{{{int(round(math.log10(t)))}}}$" if t < 1 else "1" for t in xt_])
    set_log_ticks(axa, nice_log_ticks(ylo, 1, (0.02, 0.05, 0.1, 0.2, 0.5, 1)), "y")
    axa.set_xlabel("Clean false-positive rate $\\beta_j$ (log scale)")
    axa.set_ylabel("Sensitivity $\\alpha_j$ (log scale)")
    lit = ab[ab.family == "LIT"]
    pick = set(ab[ab.family != "LIT"].annotator)
    pick |= set(lit.nsmallest(1, "beta").annotator) | set(lit.nlargest(1, "beta").annotator)
    pick |= set(lit.nlargest(2, "alpha").annotator) | set(lit.nsmallest(1, "alpha").annotator)
    for r in ab[ab.annotator.isin(pick)].itertuples():
        dx, dy, ha, va = FIG2_LABEL_OFFSETS.get(r.annotator, (5, 0, "left", "center"))
        axa.annotate(FIG2_DISPLAY.get(r.annotator, r.annotator), (r.beta, r.alpha), xytext=(dx, dy), textcoords="offset points",
                     fontsize=fsm, ha=ha, va=va, color="k")
    axa.legend(loc="lower left", title="Family (annotators)", title_fontsize=fsm, handletextpad=0.2,
               borderaxespad=0.3, alignment="left")
    panel_label(axa, "a", x=-0.17, y=1.0)

    # b: second-mode probability (top) and differences between the modes (bottom)
    m = d["modes"]
    ylab = [S.UNIVERSE_LABEL[u] for u in m.universe]
    yp = np.arange(len(m))[::-1].astype(float)
    xb, wb = 0.49, 0.21
    axb1 = fig.add_axes([xb, 0.57, wb, 0.24])
    axb2 = fig.add_axes([xb, 0.13, wb, 0.30])
    off = 0.15
    point_ci(axb1, m.p_laplace_top2, yp + off, color=OI["black"], marker="s", filled=False, ms=3.4)
    point_ci(axb1, m.p_bridge, yp - off, m.p2_bridge_lo, m.p2_bridge_hi, color=OI["vermillion"], marker="o")
    pmin = float(np.nanmin(np.r_[m.p_laplace_top2, m.p_bridge]))
    emin = int(math.floor(math.log10(pmin)))
    step = 15 if emin < -30 else (5 if emin < -10 else 1)
    e0 = -step * int(math.floor(-emin / step))
    ticks = [10.0 ** e for e in range(e0, 1, step)]
    axb1.set_xscale("log")
    axb1.set_xlim(10.0 ** (emin - step / 5), 10 ** (step / 6))
    axb1.xaxis.set_major_locator(FixedLocator(ticks))
    axb1.xaxis.set_minor_locator(NullLocator())
    axb1.set_xticklabels([f"$10^{{{int(round(math.log10(t)))}}}$" if t < 1 else "1" for t in ticks])
    _row_axes_labels(axb1, yp, ylab)
    axb1.set_ylim(-0.6, len(m) - 0.4)
    axb1.set_xlabel("Posterior probability of mode 2")

    sd = m.log_z_diff_sd.to_numpy(dtype=float)
    point_ci(axb2, m.gap_to_mode0, yp + off, color=OI["black"], marker="D", filled=False, ms=3.2)
    point_ci(axb2, m.log_z_diff_laplace, yp, color=OI["black"], marker="s", filled=False, ms=3.4)
    point_ci(axb2, m.log_z_diff_bridge, yp - off, m.log_z_diff_bridge - 1.96 * sd, m.log_z_diff_bridge + 1.96 * sd,
             color=OI["vermillion"], marker="o")
    vv = np.r_[m.gap_to_mode0, m.log_z_diff_laplace, m.log_z_diff_bridge].astype(float)
    vmin, vmax = float(np.nanmin(vv)), float(np.nanmax(vv))
    set_log_ticks(axb2, nice_log_ticks(vmin * 0.6, vmax * 1.6, (1, 2, 5, 10, 20, 50, 100, 200)), "x")
    axb2.set_xlim(vmin * 0.6, vmax * 1.6)
    _row_axes_labels(axb2, yp, ylab)
    axb2.set_ylim(-0.6, len(m) - 0.4)
    axb2.set_xlabel("Mode 1 minus mode 2 (nats, log scale)")
    hb = [Line2D([], [], ls="none", marker="o", mfc=OI["vermillion"], mec=OI["vermillion"], ms=3.4,
                 label="Bridge sampling (95% interval)"),
          Line2D([], [], ls="none", marker="s", mfc="white", mec="k", ms=3.4, label="Laplace approximation"),
          Line2D([], [], ls="none", marker="D", mfc="white", mec="k", ms=3.2, label="Log-posterior gap of the optima")]
    fig.legend(handles=hb, loc="upper left", bbox_to_anchor=(xb - 0.09, 0.995), borderaxespad=0.1,
               handletextpad=0.3)
    fig_label(fig, "b", xb - 0.115, 0.995)

    # c: realised FDR of the Strong set by dominant-mode status
    axc = fig.add_axes([0.805, 0.16, 0.19, 0.79])
    rep = d["rep"]
    us = _sim_universe_order(rep.universe.unique())
    rng = np.random.default_rng(20261014)
    xt, xl = [], []
    for i, u in enumerate(us):
        for j, truth in enumerate([True, False]):
            s = rep[(rep.universe == u) & (rep.dominant_is_truth == truth)]
            x = i * 1.0 + (j - 0.5) * 0.42
            col = OI["blue"] if truth else OI["vermillion"]
            if len(s):
                jit = rng.uniform(-0.12, 0.12, len(s))
                axc.plot(x + jit, s.dominant_strong_realised_fdr, ls="none", marker="o", ms=1.6, mfc=col,
                         mec="none", alpha=0.5, zorder=2)
                strat = "dominant_is_truth" if truth else "dominant_not_truth"
                r = _maybe(d["ms"], universe=u, stratum=strat, quantity="dominant_strong_realised_fdr_mean")
                if r is not None:
                    point_ci(axc, [x], [r.value], [r.ci_lo], [r.ci_hi], color="k", marker="D", ms=2.6,
                             horizontal=False, zorder=4, lw=0.7, filled=False)
        xt.append(i * 1.0)
        xl.append(SIM_UNIVERSE_LABEL.get(u, u))
    ref_line(axc, d["q_strong"], "h", color="k", ls="--", lw=0.7)
    axc.set_xticks(xt)
    axc.set_xticklabels(xl, fontsize=fsm, linespacing=1.0)
    axc.tick_params(axis="x", length=0)
    axc.set_ylim(0, 1.0)
    axc.set_xlim(-0.5, len(us) - 0.5)
    axc.set_xlabel("Simulated from", labelpad=1)
    axc.set_ylabel("Realised FDR of the Strong set")
    hc = [Line2D([], [], ls="none", marker="o", mfc=OI["blue"], mec="none", ms=3.0, label="Dominant = truth"),
          Line2D([], [], ls="none", marker="o", mfc=OI["vermillion"], mec="none", ms=3.0, label="Dominant $\\neq$ truth"),
          Line2D([], [], ls="none", marker="D", mfc="white", mec="k", ms=2.8, label="Mean, inter-decile range"),
          Line2D([], [], color="k", lw=0.7, ls="--", label=f"Claimed BFDR {S.fmt_ratio(d['q_strong'])}")]
    axc.legend(handles=hc, loc="center left", bbox_to_anchor=(-0.03, 0.55), borderaxespad=0.1, handletextpad=0.2)
    panel_label(axc, "c", x=-0.40, y=1.0)
    return fig



def caption_fig2_model(fd: FigureData) -> str:
    d = fd.d
    ab = d["ab"]
    m = d["modes"].set_index("universe")
    ms = d["ms"]
    nlit = int((ab.family == "LIT").sum())
    parts = [
        "\\textbf{Family-HB parameters and posterior modes.} "
        "(a) Annotator sensitivity $\\alpha_j$ against clean false-positive rate $\\beta_j$ "
        f"(posterior means with 95\\% credible intervals, global universe, NUTS) for the {len(ab)} held-in annotators "
        f"({nlit} literature); the dashed diagonal is $\\alpha_j=\\beta_j$, which the parametrisation excludes. "
        f"Sensitivities range from {_num(ab.alpha.min())} to {_num(ab.alpha.max())} and false-positive rates from "
        f"{S.fmt_sig(ab.beta.min(), 2)} to {S.fmt_sig(ab.beta.max(), 2)}. "
    ]
    sw = [u for u, v in d["swap"].items() if v]
    sw_txt = ""
    if sw:
        sw_txt = (" (in the " + " and ".join(S.UNIVERSE_LABEL[u].lower() for u in S.UNIVERSE_ORDER if u in sw)
                  + " fits, whose class weights are stored for both modes, mode 2 approximately exchanges the weights "
                  "of the true class and the literature-burst class)")
    parts.append("(b) Mode 1 is the dominant posterior mode of each universe fit and mode 2 the second" + sw_txt + ". "
                 "Top: posterior probability of mode 2 by bridge sampling (filled; Monte Carlo 95\\% interval) and by "
                 "Laplace's method (open squares). Bottom: mode 1 minus mode 2 in log posterior mass (bridge sampling, "
                 "filled; Laplace's method, open squares) and the log-posterior gap of the two optima (diamonds). "
                 "Values and ranges in Table~\\ref{tab:fit}. ")
    rep = d["rep"]
    seg = []
    for u in _sim_universe_order(rep.universe.unique()):
        t = _maybe(ms, universe=u, stratum="dominant_is_truth", quantity="dominant_strong_realised_fdr_mean")
        nt = _maybe(ms, universe=u, stratum="dominant_not_truth", quantity="dominant_strong_realised_fdr_mean")
        pt = _maybe(ms, universe=u, stratum="all", quantity="p_dominant_is_truth")
        if t is None:
            continue
        if nt is None:
            seg.append(f"{SIM_UNIVERSE_TEXT.get(u, u)}: dominant mode is the truth mode in all "
                       f"{int(t.n_replicates)} replicates, mean realised FDR {_num(t.value)}")
            continue
        seg.append(f"{SIM_UNIVERSE_TEXT.get(u, u)}: dominant mode is the truth mode in "
                   f"{int(t.n_replicates)} of {int(pt.n_replicates) if pt is not None else int(t.n_replicates + nt.n_replicates)} "
                   f"replicates, mean realised FDR {_num(t.value)} when it is and {_num(nt.value)} when it is not")
    shr = d.get("cc_share", {})
    shr_txt = ""
    if shr:
        shr_txt = ("; the simulated data contain " + " and ".join(
            f"{S.fmt_pct(v)}\\% ({S.UNIVERSE_LABEL.get(u, u).lower()})" for u, v in sorted(shr.items(), key=lambda kv: kv[0] != "immune"))
            + " of the real number of motif- or coexpression-observed edges")
    parts.append("(c) Mode-selection simulation from the fitted model with the design-faithful simulator (data "
                 "simulated at the parameters of the indicated mode with the real literature observation masks and the "
                 "motif/coexpression masks rebuilt from the simulated literature reports, as in the evidence builder"
                 + shr_txt + "; refitted by multi-start MAP): realised false discovery rate of the Strong set for "
                 "replicates whose dominant mode is or is not the truth mode (points: replicates; diamonds: mean with "
                 f"inter-decile range over replicates; dashed line: claimed BFDR {S.fmt_ratio(d['q_strong'])}); "
                 + "; ".join(seg) + ".")
    sens = d["sens"]
    sl = {"fixed_masks": "with every observation mask fixed at the real one",
          "faithful_cc_strat": "with class weights that depend on the literature-mask stratum"}
    st = []
    for sim in ("fixed_masks", "faithful_cc_strat"):
        ss = sens[sens.simulator == sim] if len(sens) else sens
        if not len(ss):
            continue
        items = []
        for u in _sim_universe_order(list(ss.universe.unique())):
            r = ss[ss.universe == u].iloc[0]
            items.append(SIM_UNIVERSE_TEXT.get(u, u).split(",")[0]
                         + (" (truth = mode 2)" if u == "immune_truthmode1" else "") + f" {_num(r.value)}")
        st.append(f"{sl[sim]}: " + ", ".join(items))
    if st:
        parts.append(" Sensitivity, probability that the dominant mode is the truth mode " + "; ".join(st)
                     + " (Supplementary Table~\\ref{tab:S-modes-sim}).")
    ph = [p for p, f in (("(b)", d["ph_b"]), ("(c)", d["ph_c"])) if f]
    if ph:
        what = []
        if d["ph_b"]:
            what.append("the bridge-sampling estimates in (b)")
        if d["ph_c"]:
            what.append("panel (c)")
        parts.append(" " + " and ".join(what).capitalize() + (" are" if len(what) > 1 else " is")
                     + " post hoc (defined after the held-out data had been examined; no held-out data are used).")
    return "".join(parts)


# ============================================================================ fig3 robustness
ROB_ORDER = ["em", "dependence", "multifactor", "family_regularized", "family_hb",
             "union", "majority", "vote_fraction", "fisher", "stouffer"]


def data_fig3_robustness() -> FigureData:
    P = S.Provenance("fig3_robustness")
    mr0 = P.read("rob_method_ranking", "tissue = all_tissues, metric = f1, subset in {all, no_fhb_mode_switch (post hoc)}; "
                 "lineage: subset = excl_uninformative (post hoc; the configurations in which Family-HB is identified, "
                 "the same for every estimator); source-clustered intervals mean_D_cluster_lo/hi")
    mr = mr0[(mr0.tissue == "all_tissues") & (mr0.metric == "f1") & (mr0.subset == "all")].copy()
    mr_lin = mr0[(mr0.tissue == "all_tissues") & (mr0.metric == "f1") & (mr0.subset == "excl_uninformative")
                 & (mr0.protocol == "lineage")].copy()
    mr_ns = mr0[(mr0.tissue == "all_tissues") & (mr0.metric == "f1") & (mr0.subset == "no_fhb_mode_switch")].copy()
    es = P.read("rob_edge_stability", "protocol = source, subset = all; jaccard_bfdr_mean and jaccard_top_sizematched_mean "
                "(post hoc) with bootstrap intervals over configurations")
    es = es[(es.protocol == "source") & (es.subset == "all")].copy()
    ct = P.read("rob_estimator_contrasts", "reference = family_hb, subset = all, scope = all_tissues (D_f1) or "
                "universe (jaccard_bfdr)")
    ct = ct[(ct.subset == "all") & (ct.reference == "family_hb")].copy()
    P.note("(constants)", "bfdr_strong from r2.common.PRESPECIFIED")
    h = mr[(mr.estimator == "family_hb") & (mr.protocol == "source")].iloc[0]
    hn = mr_ns[(mr_ns.estimator == "family_hb") & (mr_ns.protocol == "source")]
    hl = mr_lin[mr_lin.estimator == "family_hb"]
    return FigureData("fig3_robustness", P, dict(
        mr=mr, mr_ns=mr_ns, mr_lin=mr_lin, es=es, ct=ct, q=float(_PRE["bfdr_strong"]),
        ph_lin=bool(len(hl) and (hl.analysis_status == "post_hoc").all()),
        ph_ident=_is_posthoc_col(h, "n_configs_identified"), ph_clus=_is_posthoc_col(h, "mean_D_cluster_lo"),
        ph_ns=bool(len(hn) and (hn.analysis_status == "post_hoc").all()),
        ph_sm=bool(es.post_hoc_columns.astype(str).str.contains("jaccard_top_sizematched").any()),
        switch=str(hn.fhb_mode_switch_excluded.iloc[0]) if len(hn) else "",
        n_switch=int(h.n_fhb_mode_switch) if S.finite(h.get("n_fhb_mode_switch")) else 0))


def _is_posthoc_col(r, col: str) -> bool:
    """True if ``col`` is listed (shell wildcards allowed) in the row's ``post_hoc_columns``."""
    import fnmatch
    v = r.get("post_hoc_columns") if hasattr(r, "get") else None
    if not isinstance(v, str):
        return False
    pats = [p.strip().split(" (")[0] for p in v.split(";") if p.strip()]
    return any(fnmatch.fnmatchcase(col, p) for p in pats)


def _switch_text(text: str) -> str:
    by: dict[str, list[str]] = {}
    for tok in str(text).split(";"):
        if ":" in tok:
            u, c = tok.strip().split(":", 1)
            by.setdefault(u, []).append(c)
    out = []
    for u, cs in by.items():
        out.append(S.UNIVERSE_LABEL.get(u, u).lower() + ", " + (cs[0] if len(cs) == 1 else ", ".join(cs[:-1]) + " and " + cs[-1])
                   + " dropped")
    return "; ".join(out)


FADED = 0.35          # alpha of the BFDR-set markers of fig3b


def plot_fig3_robustness(fd: FigureData):
    d = fd.d
    fig = _fig(S.FULL_W, 11.6)
    order = [e for e in ROB_ORDER if e in set(d["mr"].estimator)]
    n = len(order)
    yp = np.arange(n)[::-1].astype(float)
    # visual gap between latent-class models and rules
    ngrp = sum(1 for e in order if S.SCORE_STYLE[e].group == "lcm")
    yp = np.array([y + (0.5 if i < ngrp else 0.0) for i, y in enumerate(yp)])
    labels = [S.score_label(e) for e in order]
    protos = ["source", "family", "lineage"]
    left, right = 0.155, 0.985
    wtot = right - left
    gapw = 0.035
    wa = (wtot - gapw * 2) / 3
    off = 0.2
    axes_a = []
    for k, p in enumerate(protos):
        ax = fig.add_axes([left + k * (wa + gapw), 0.60, wa, 0.33])
        src_a = d["mr_lin"] if (p == "lineage" and len(d["mr_lin"])) else d["mr"]
        s = src_a[src_a.protocol == p].set_index("estimator")
        sn = d["mr_ns"][d["mr_ns"].protocol == p].set_index("estimator")
        for i, e in enumerate(order):
            if e not in s.index:
                continue
            r = s.loc[e]
            st = S.SCORE_STYLE[e]
            y0 = yp[i] + (off if p == "source" else 0.0)
            point_ci(ax, [r.mean_D], [y0], [r.mean_D_cluster_lo], [r.mean_D_cluster_hi], color=st.color,
                     marker=st.marker, ms=3.6)
            if p == "source" and e in sn.index:
                rn = sn.loc[e]
                point_ci(ax, [rn.mean_D], [yp[i] - off], [rn.mean_D_cluster_lo], [rn.mean_D_cluster_hi], color=st.color,
                         marker=st.marker, ms=3.2, filled=False)
            if r.n_configs_identified < r.n_configs:
                ax.annotate(f"{int(r.n_configs_identified)}/{int(r.n_configs)} identified", (r.mean_D_cluster_hi, yp[i]),
                            xytext=(3, 0), textcoords="offset points", fontsize=FS["small"] + 0.5, va="center")
        ttl = S.PROTOCOL_LABEL[p]
        if p == "lineage" and len(d["mr_lin"]):
            nl = int(d["mr_lin"].n_configs.iloc[0])
            ttl += f" ({nl} identified" + (", post hoc)" if d["ph_lin"] else ")")
        ax.set_title(ttl, fontsize=FS["title"])
        ax.set_xlim(left=0)
        ax.axhline(yp[ngrp - 1] - 0.75, color=OI["grey"], lw=0.4, ls=":")
        if k == 0:
            _row_axes_labels(ax, yp, labels)
        else:
            _hide_y(ax)
        ax.set_ylim(yp.min() - 0.7, yp.max() + 0.7)
        axes_a.append(ax)
    xmax = max(a.get_xlim()[1] for a in axes_a[1:])
    for a in axes_a[1:]:
        a.set_xlim(0, xmax)
    axes_a[1].set_xlabel("Mean rank disagreement $D$ of the tissue-method ranking (posterior-weighted $F_1$)")
    ha = [Line2D([], [], ls="none", marker="o", mfc="k", mec="k", ms=3.4, label="All configurations"),
          Line2D([], [], ls="none", marker="o", mfc="white", mec="k", ms=3.2,
                 label="Without Family-HB mode switches" + (" (post hoc)" if d["ph_ns"] else ""))]
    fig.legend(handles=ha, loc="upper right", bbox_to_anchor=(0.995, 1.0), ncol=2, borderaxespad=0.1)
    fig_label(fig, "a", 0.005, 0.995)

    # b: set Jaccard under source dropout per universe (BFDR set filled, size-matched open)
    univs = [u for u in S.UNIVERSE_ORDER if u in set(d["es"].universe)]
    wb = (wtot - gapw * (len(univs) - 1)) / len(univs)
    axes_b = []
    for k, u in enumerate(univs):
        ax = fig.add_axes([left + k * (wb + gapw), 0.085, wb, 0.36])
        s = d["es"][d["es"].universe == u].set_index("estimator")
        for i, e in enumerate(order):
            if e not in s.index:
                continue
            r = s.loc[e]
            st = S.SCORE_STYLE[e]
            # faded: BFDR sets (sizes differ); filled: size-matched sets (open markers are reserved for panel a)
            point_ci(ax, [r.jaccard_bfdr_mean], [yp[i] + off], [r.jaccard_bfdr_mean_boot_lo], [r.jaccard_bfdr_mean_boot_hi],
                     color=st.color, marker=st.marker, ms=3.4, alpha=FADED)
            point_ci(ax, [r.jaccard_top_sizematched_mean], [yp[i] - off], [r.jaccard_top_sizematched_mean_boot_lo],
                     [r.jaccard_top_sizematched_mean_boot_hi], color=st.color, marker=st.marker, ms=3.6)
        ax.set_title(S.UNIVERSE_LABEL[u] + " universe", fontsize=FS["title"])
        ax.axhline(yp[ngrp - 1] - 0.75, color=OI["grey"], lw=0.4, ls=":")
        if k == 0:
            _row_axes_labels(ax, yp, labels)
        else:
            _hide_y(ax)
        ax.set_ylim(yp.min() - 0.7, yp.max() + 0.7)
        axes_b.append(ax)
    lo = min(a.get_xlim()[0] for a in axes_b)
    for a in axes_b:
        a.set_xlim(max(0.0, math.floor(lo * 10) / 10), 1.0)
    hb = [Line2D([], [], ls="none", marker="o", mfc="k", mec="k", ms=3.4, alpha=FADED,
                 label=f"BFDR $\\leq$ {S.fmt_ratio(d['q'])} set (sizes differ)"),
          Line2D([], [], ls="none", marker="o", mfc="k", mec="k", ms=3.6,
                 label="Top $n$ edges, $n$ = size of the Family-HB set" + (" (post hoc)" if d["ph_sm"] else ""))]
    fig.legend(handles=hb, loc="upper right", bbox_to_anchor=(0.995, 0.505), ncol=2, borderaxespad=0.1)
    fig.text(left + wtot / 2, 0.012, "Mean Jaccard index of the set, full data vs single-source dropout",
             ha="center", va="bottom", fontsize=FS["label"])
    fig_label(fig, "b", 0.005, 0.50)
    return fig


def caption_fig3_robustness(fd: FigureData) -> str:
    d = fd.d
    mr = d["mr"].set_index(["estimator", "protocol"])
    mn = d["mr_ns"].set_index(["estimator", "protocol"])
    es = d["es"].set_index(["universe", "estimator"])

    def dval(e, p, m=mr):
        r = m.loc[(e, p)]
        return f"{S.fmt_ratio(r.mean_D, 3)} {_ci(r.mean_D_cluster_lo, r.mean_D_cluster_hi, 3)}"

    def jval(u, e, col="jaccard_bfdr_mean", lo="jaccard_bfdr_mean_boot_lo", hi="jaccard_bfdr_mean_boot_hi"):
        r = es.loc[(u, e)]
        return f"{_num(r[col])} {_ci(r[lo], r[hi])}"

    def sval(u, e):
        return jval(u, e, "jaccard_top_sizematched_mean", "jaccard_top_sizematched_mean_boot_lo",
                    "jaccard_top_sizematched_mean_boot_hi")

    hs = mr.loc[("family_hb", "source")]
    hf = mr.loc[("family_hb", "family")]
    ml = d["mr_lin"].set_index(["estimator", "protocol"]) if len(d["mr_lin"]) else mr
    fhb_l = ml.loc[("family_hb", "lineage")]
    n_ns = int(mn.loc[("family_hb", "source")].n_configs) if ("family_hb", "source") in mn.index else None
    sz = {e: int(es.loc[("global", e)].n_bfdr_full) for e in ("family_hb", "em", "union") if ("global", e) in es.index}
    q = S.fmt_ratio(d["q"])
    ph = " (post hoc)"
    txt = (
        "\\textbf{Robustness to annotator dropout.} "
        "(a) Mean rank disagreement $D$ \\eqref{eq:rankdisagreement} between the method ranking under the full-data "
        "credibility and under dropout of one source, one modality family or one integrator lineage, pooled over the "
        f"three tissue universes ({int(hs.n_configs)} source, {int(hf.n_configs)} family and {int(fhb_l.n_configs)} lineage "
        "configurations), with 95\\% "
        "bootstrap intervals clustered by the dropped source, family or lineage" + (ph if d["ph_clus"] else "")
        + "; family and lineage intervals are descriptive (at most three configurations per universe). "
        f"Source dropout: Family-HB {dval('family_hb', 'source')}, EM {dval('em', 'source')}, union {dval('union', 'source')}"
    )
    if n_ns is not None:
        txt += (f"; open markers{ph if d['ph_ns'] else ''}: the {n_ns} configurations left after removing the "
                f"{d['n_switch']} in which Family-HB switches posterior mode ({_switch_text(d['switch'])}), Family-HB "
                f"{dval('family_hb', 'source', mn)}, EM {dval('em', 'source', mn)}")
    fhb_la = mr.loc[("family_hb", "lineage")]
    nsw_l = int(fhb_l.n_fhb_mode_switch) if S.finite(fhb_l.get("n_fhb_mode_switch")) else None
    txt += (f". Lineage dropout{ph if d['ph_lin'] else ''}: the {int(fhb_l.n_configs)} configurations in which Family-HB "
            "is identified, for every estimator (dropping the DoRothEA lineage leaves one modality family, so Family-HB is "
            f"not identified in {int(fhb_la.n_configs - fhb_la.n_configs_identified)} of the {int(fhb_la.n_configs)} lineage "
            f"configurations{'; post hoc identification rule' if d['ph_ident'] else ''}): Family-HB "
            f"{dval('family_hb', 'lineage', ml)}, EM {dval('em', 'lineage', ml)}"
            + (f"; Family-HB switches posterior mode in {nsw_l} of them" if nsw_l else "") + ". "
            f"(b) Mean Jaccard index between the full-data and the single-source dropout sets (faded markers: BFDR "
            f"$\\leq {q}$ set; filled markers{ph if d['ph_sm'] else ''}: top $n$ edges, $n$ the size of the Family-HB set), "
            "with 95\\% bootstrap intervals over the dropout configurations (one per source). For a 0/1 rule the BFDR "
            "set is its set of positives, several times larger than the Family-HB set")
    if sz:
        txt += (" (global universe: Family-HB " + S.fmt_count(sz.get("family_hb")) + ", EM " + S.fmt_count(sz.get("em"))
                + ", union " + S.fmt_count(sz.get("union")) + " edges)")
    txt += (", which inflates its Jaccard index. Global universe, BFDR set: Family-HB "
            f"{jval('global', 'family_hb')}, EM {jval('global', 'em')}, union {jval('global', 'union')}; size-matched: "
            f"Family-HB {sval('global', 'family_hb')}, EM {sval('global', 'em')}, union {sval('global', 'union')}.")
    if d.get("switch"):
        sw_u = [u for u in S.UNIVERSE_ORDER if u in {tok.split(":", 1)[0] for tok in str(d["switch"]).split(";") if ":" in tok}]
        if sw_u:
            txt += (" The Family-HB values of the " + " and ".join(S.UNIVERSE_LABEL.get(u, u).lower() for u in sw_u)
                    + " universe" + ("s" if len(sw_u) > 1 else "") + " include the source configurations in which "
                    "Family-HB switches posterior mode (" + _switch_text(d["switch"]) + "; size-matched Family-HB "
                    + ", ".join(f"{S.UNIVERSE_LABEL.get(u, u).lower()} {sval(u, 'family_hb')}" for u in sw_u
                                if (u, "family_hb") in es.index) + ").")
    return txt


# ============================================================================ fig4 held-out
FIG4_GROUPS = [
    ("Latent-class estimators", ["em", "dependence", "multifactor", "family_regularized", "family_hb"]),
    ("Combination rules", ["union", "majority", "vote_count", "vote_fraction", "n_families_reporting", "fisher",
                           "stouffer"]),
    ("Integration schemes", ["dorothea_level_heldin", "collectri_heldin"]),
    ("", ["random"]),
    ("Post hoc", ["status3", "family_hb_unobserved_last"]),
]
FIG4_REFS = ["encode", "enets2", "pert_pooled"]
FIG4_LABEL = {"family_hb": "Family-HB (MAP)"}
FIG4_STATUS_REFS = ["encode", "enets2", "pert_pooled"]


def data_fig4_heldout() -> FigureData:
    P = S.Provenance("fig4_heldout")
    h = P.read("heldout_estimators", "universe = global, eval_subset = all_evaluable, reference in "
               "{encode, enets2, pert_pooled}; score in the plotted list")
    scores = [s for _, g in FIG4_GROUPS for s in g]
    h = h[(h.universe == "global") & (h.eval_subset == "all_evaluable") & h.reference.isin(FIG4_REFS)
          & h.score.isin(scores)].copy()
    st = P.read("heldout_evidence_status", "reference in {encode, enets2, pert_pooled}")
    st = st[st.reference.isin(FIG4_STATUS_REFS)].copy()
    st["low_support"] = st.low_support.map(S.as_bool)
    ob = P.read("obs_heldout_status", "universe = global, reference = encode, status in {unobserved, observed_neg}: "
                "mean_cred_em (EM credibility by evidence status)")
    ob = ob[(ob.universe == "global") & (ob.reference == "encode") & ob.status.isin(["unobserved", "observed_neg"])]
    P.note("(constants)", "LOW_SUPPORT_TFS, LOW_SUPPORT_POS from r2.heldout_eval")
    return FigureData("fig4_heldout", P, dict(h=h, st=st, ob=ob,
                                              ph_b=bool((st.analysis_status == "post_hoc").all()),
                                              low_tfs=int(_module_constant("heldout_eval", "LOW_SUPPORT_TFS")),
                                              low_pos=int(_module_constant("heldout_eval", "LOW_SUPPORT_POS"))))


def _fig4_rows(present):
    rows, ypos, labels, styles, seps = [], [], [], [], []
    y = 0.0
    for gi, (gname, members) in enumerate(FIG4_GROUPS):
        mem = [m for m in members if m in present]
        if not mem:
            continue
        if gi > 0:
            seps.append(y - 0.35)
            y -= 0.7
        for m in mem:
            rows.append(m)
            ypos.append(y)
            labels.append(FIG4_LABEL.get(m, S.score_label(m)))
            styles.append("italic" if gname == "Post hoc" else None)
            y -= 1.0
    return rows, np.array(ypos), labels, styles, seps


def plot_fig4_heldout(fd: FigureData):
    d = fd.d
    h = d["h"]
    fig = _fig(S.FULL_W, 12.4)
    rows, yp, labels, styles, seps = _fig4_rows(set(h.score))
    left, right = 0.20, 0.985
    gapw = 0.025
    wa = (right - left - 2 * gapw) / 3
    axes = []
    for k, ref in enumerate(FIG4_REFS):
        ax = fig.add_axes([left + k * (wa + gapw), 0.43, wa, 0.49])
        s = h[h.reference == ref].set_index("score")
        for i, sc in enumerate(rows):
            if sc not in s.index:
                continue
            r = s.loc[sc]
            stl = S.SCORE_STYLE[sc]
            point_ci(ax, [r.ap_lift], [yp[i]], [r.ap_lift_lo], [r.ap_lift_hi], color=stl.color, marker=stl.marker,
                     ms=3.6, filled=stl.group != "post_hoc_mechanism")
        ref_line(ax, 1.0)
        for sp in seps:
            ax.axhline(sp, color=OI["grey"], lw=0.4, ls=":")
        first = s.iloc[0]
        ax.set_title(f"{S.reference_label(ref)}\n({S.fmt_count_plain(first.n_tfs)} TFs, base rate "
                     f"{S.fmt_ratio(first.base_rate, 3)})", fontsize=FS["title"], linespacing=1.1)
        if k == 0:
            _row_axes_labels(ax, yp, labels, styles=styles)
        else:
            _hide_y(ax)
        ax.set_ylim(yp.min() - 0.7, yp.max() + 0.7)
        axes.append(ax)
    axes[1].set_xlabel("AP lift (average precision / base rate), global universe, all evaluable edges")
    fig_label(fig, "a", 0.005, 0.995)

    # b: evidence status O/E
    st = d["st"]
    univs = [u for u in S.UNIVERSE_ORDER if u in set(st.universe)]
    left_b = 0.075
    wb = (right - left_b - gapw * (len(univs) - 1)) / len(univs)
    axes_b = []
    refs = [r for r in FIG4_STATUS_REFS if r in set(st.reference)]
    xr = np.arange(len(refs), dtype=float)
    offs = {"unobserved": -0.22, "observed_neg": 0.0, "reported": 0.22}
    vals = st[["oe_tf_matched", "oe_lo", "oe_hi"]].to_numpy(dtype=float)
    vpos = vals[vals > 0]
    ylo, yhi = max(vpos.min() * 0.85, 0.3), vals.max() * 1.15
    for k, u in enumerate(univs):
        ax = fig.add_axes([left_b + k * (wb + gapw), 0.10, wb, 0.22])
        s = st[st.universe == u]
        for status in S.STATUS_ORDER:
            ss = s[s.status == status].set_index("reference")
            sty = S.STATUS_STYLE[status]
            for j, ref in enumerate(refs):
                if ref not in ss.index:
                    continue
                r = ss.loc[ref]
                point_ci(ax, [xr[j] + offs[status]], [r.oe_tf_matched], [r.oe_lo], [r.oe_hi], color=sty["color"],
                         marker=sty["marker"], filled=sty["filled"], horizontal=False, ms=3.2,
                         alpha=0.45 if r.low_support else 1.0)
        ref_line(ax, 1.0, "h")
        set_log_ticks(ax, nice_log_ticks(ylo, yhi, (0.5, 0.7, 1, 1.5, 2, 3, 5, 7)), "y")
        ax.set_ylim(ylo, yhi)
        ax.set_xticks(xr)
        ax.set_xticklabels([{"encode": "ENCODE", "encode_strict": "ENC.-strict", "enets2": "ENETS2",
                             "pert_pooled": "Pert."}[r] for r in refs], fontsize=FS["small"] + 0.5)
        ax.tick_params(axis="x", length=0)
        ax.set_xlim(-0.5, len(refs) - 0.5)
        ax.set_title(S.UNIVERSE_LABEL[u] + " universe", fontsize=FS["title"])
        if k == 0:
            ax.set_ylabel("O/E (TF-matched)")
        else:
            ax.tick_params(axis="y", labelleft=False)
        axes_b.append(ax)
    hs = [Line2D([], [], ls="none", marker=S.STATUS_STYLE[s]["marker"], ms=3.4, mec=S.STATUS_STYLE[s]["color"],
                 mfc=S.STATUS_STYLE[s]["color"] if S.STATUS_STYLE[s]["filled"] else "white",
                 label=S.STATUS_LABEL[s]) for s in S.STATUS_ORDER]
    fig.legend(handles=hs, loc="lower center", bbox_to_anchor=((left + right) / 2, 0.0), ncol=3, borderaxespad=0.2,
               handletextpad=0.2, title="Evidence status of the edge:", title_fontsize=FS["small"] + 0.5)
    fig._suptitle = None
    fig_label(fig, "b", 0.005, 0.37)
    if d.get("ph_b"):
        fig.text(0.03, 0.366, "(post hoc)", fontsize=FS["small"] + 0.5, fontstyle="italic", ha="left", va="top")
    return fig


def caption_fig4_heldout(fd: FigureData) -> str:
    d = fd.d
    h = d["h"].set_index(["reference", "score"])
    st = d["st"]

    def ap(ref, sc):
        r = h.loc[(ref, sc)]
        return f"{_num(r.ap_lift)} {_ci(r.ap_lift_lo, r.ap_lift_hi)}"

    g = st[st.universe == "global"].set_index(["reference", "status"])

    def oe(ref, s):
        r = g.loc[(ref, s)]
        return f"{_num(r.oe_tf_matched)} {_ci(r.oe_lo, r.oe_hi)}"

    tis = st[(st.universe != "global") & (st.status == "unobserved") & ~st.low_support]
    ge = g.loc[("encode", "unobserved")], g.loc[("encode", "observed_neg")]
    ob = d["ob"].set_index("status")
    cred = (f"Family-HB {S.fmt_sig(ge[0].mean_cred_family_hb, 2)} vs {S.fmt_sig(ge[1].mean_cred_family_hb, 2)}")
    if "unobserved" in ob.index and "observed_neg" in ob.index:
        cred += (f"; EM {S.fmt_sig(ob.loc['unobserved'].mean_cred_em, 2)} vs "
                 f"{S.fmt_sig(ob.loc['observed_neg'].mean_cred_em, 2)}")
    higher = bool(ge[0].mean_cred_family_hb > ge[1].mean_cred_family_hb)
    return (
        "\\textbf{Held-out evaluation in the global universe.} "
        "(a) AP lift of every score on all evaluable edges, with 95\\% TF-bootstrap intervals; the vertical line marks a "
        "random ranking. Open italic rows are post hoc mechanism scores (the 3-level evidence status and Family-HB with "
        "unobserved edges ranked last); Family-HB (MAP) is the posterior-mode fit used in the paired tests. "
        f"Family-HB: ENCODE {ap('encode', 'family_hb')}, ENETS2 {ap('enets2', 'family_hb')}, pooled perturbation "
        f"{ap('pert_pooled', 'family_hb')}; Fisher: {ap('encode', 'fisher')}, {ap('enets2', 'fisher')}, "
        f"{ap('pert_pooled', 'fisher')}; EM: {ap('encode', 'em')}, {ap('enets2', 'em')}, {ap('pert_pooled', 'em')}. "
        "(b) Positives among unobserved, observed but unreported, and reported edges relative to the TF-matched "
        f"expectation (95\\% TF-bootstrap intervals; faded: fewer than {d['low_tfs']} TFs with positives or fewer than "
        f"{d['low_pos']} positives)" + ("; the whole panel is post hoc" if d["ph_b"] else "") + ". "
        f"Global universe, unobserved edges: ENCODE {oe('encode', 'unobserved')}, ENETS2 {oe('enets2', 'unobserved')}, "
        f"pooled perturbation {oe('pert_pooled', 'unobserved')}; observed negatives {oe('encode', 'observed_neg')} "
        f"(ENCODE). Unobserved edges are depleted of positives in every universe (O/E {_num(tis.oe_tf_matched.min())} to "
        f"{_num(tis.oe_tf_matched.max())} in the tissue universes)"
        + (", whereas the missing-at-random latent-class estimators give them a higher mean credibility than observed "
           f"negatives (ENCODE-evaluable edges: {cred}): coverage is informative (inference: not missing at random)."
           if higher else ".")
    )


# ============================================================================ fig5 universe decomposition
# the decomposition is shown at the evaluation depth used for Enr@k throughout the paper (one of the
# prespecified evaluation depths; checked here so that the figure cannot use an unevaluated k)
FIG5_K = 500
assert FIG5_K in _PRE["eval_k"]


def _p_text(p) -> str:
    """p in decimal notation from 0.01 up, scientific below (LaTeX math content, no $)."""
    if not S.finite(p):
        return "--"
    p = float(p)
    if p >= 0.01:
        return f"{p:.2f}"
    return S.fmt_p(p)[1:-1]


def data_fig5_universe() -> FigureData:
    P = S.Provenance("fig5_universe")
    k = FIG5_K
    dec = P.read("heldout_universe_decomposition", f"metric = p_at_{k}")
    dec = dec[dec.metric == f"p_at_{k}"].copy()
    nul0 = P.read("heldout_downsampling_null", f"metric = p_at_{k}, null in {{prevalence_matched (edge sampling), "
                  "tf_block_matched (whole TF blocks; post hoc)}")
    cols = ["tissue", "reference", "credibility", "null_median", "null_lo", "null_hi", "n_draws", "p_emp_two_sided",
            "frac_gap_surviving", "analysis_status"]
    nul = nul0[(nul0.metric == f"p_at_{k}") & (nul0.null == "prevalence_matched")][cols]
    tfb = nul0[(nul0.metric == f"p_at_{k}") & (nul0.null == "tf_block_matched")][cols]
    tfb = tfb.rename(columns={c: "tfb_" + c for c in cols[3:]})
    m = dec.merge(nul, on=["tissue", "reference", "credibility"], how="left", validate="one_to_one")
    m = m.merge(tfb, on=["tissue", "reference", "credibility"], how="left", validate="one_to_one")
    ph_ci = bool(dec.post_hoc_columns.astype(str).str.contains("ratio_raw_lo").any())
    ph_p = bool(nul0.post_hoc_columns.astype(str).str.contains("p_emp_two_sided").any()) \
        if "post_hoc_columns" in nul0 else False
    for pre in ("", "tfb_"):
        m[pre + "null_ratio_lo"] = m[pre + "null_lo"] / m.value_glob
        m[pre + "null_ratio_hi"] = m[pre + "null_hi"] / m.value_glob
        m[pre + "null_ratio_median"] = m[pre + "null_median"] / m.value_glob
    P.note("(constants)", "FIG5_K (one of r2.common.PRESPECIFIED['eval_k'])")
    return FigureData("fig5_universe", P, dict(m=m, k=k, ph_ci=ph_ci, ph_p=ph_p))


def plot_fig5_universe(fd: FigureData):
    d = fd.d
    m = d["m"]
    fig = _fig(S.FULL_W, 7.0)
    tissues = [t for t in ("immune", "kidney", "lung") if t in set(m.tissue)]
    refs = [r for r in ("encode", "pert_pooled") if r in set(m.reference)]
    qty = [("ratio_raw", "ratio_raw_lo", "ratio_raw_hi", f"P@{d['k']} ratio (raw)"),
           ("ratio_base_rate", None, None, "Base-rate ratio"),
           ("ratio_enrichment", "ratio_enrichment_lo", "ratio_enrichment_hi", f"Enr@{d['k']} ratio")]
    ypos, labels, keys = [], [], []
    y = 0.0
    for ri, ref in enumerate(refs):
        if ri:
            y -= 1.1
        for q in qty:
            ypos.append(y)
            labels.append(q[3])
            keys.append((ref, q))
            y -= 1.0
    ypos = np.array(ypos)
    left, right = 0.17, 0.985
    gapw = 0.03
    w = (right - left - gapw * (len(tissues) - 1)) / len(tissues)
    vals = np.r_[m.ratio_raw_lo, m.ratio_raw_hi, m.ratio_base_rate, m.ratio_enrichment_lo, m.ratio_enrichment_hi,
                 m.null_ratio_lo, m.null_ratio_hi, m.tfb_null_ratio_lo, m.tfb_null_ratio_hi]
    xlo, xhi = np.nanmin(vals) * 0.8, np.nanmax(vals) * 1.4
    off = {"global_fit": 0.14, "context_fit": -0.14}
    axes = []
    for k, t in enumerate(tissues):
        ax = fig.add_axes([left + k * (w + gapw), 0.20, w, 0.72])
        for i, (ref, (c, lo, hi, _)) in enumerate(keys):
            col = S.REFERENCE_COLOR[ref]
            for cred in ("global_fit", "context_fit"):
                r = _maybe(m, tissue=t, reference=ref, credibility=cred)
                if r is None:
                    continue
                yy = ypos[i] + off[cred]
                if c == "ratio_raw" and S.finite(r.tfb_null_ratio_lo):
                    ax.add_patch(Rectangle((r.tfb_null_ratio_lo, yy - 0.11), r.tfb_null_ratio_hi - r.tfb_null_ratio_lo,
                                           0.22, fc="none", ec=OI["grey"], lw=0.6, ls=(0, (2, 1)), zorder=1))
                if c == "ratio_raw" and S.finite(r.null_ratio_lo):
                    ax.add_patch(Rectangle((r.null_ratio_lo, yy - 0.11), r.null_ratio_hi - r.null_ratio_lo, 0.22,
                                           fc=OI["grey"], ec="none", alpha=0.45, zorder=1))
                point_ci(ax, [r[c]], [yy], [r[lo]] if lo else None, [r[hi]] if hi else None, color=col,
                         marker=S.REFERENCE_MARKER[ref], filled=(cred == "global_fit"), ms=3.4)
        ref_line(ax, 1.0)
        set_log_ticks(ax, nice_log_ticks(xlo, xhi, (0.25, 0.5, 1, 2, 4)), "x")
        ax.set_xlim(xlo, xhi)
        ax.set_title(S.UNIVERSE_LABEL[t] + " vs global", fontsize=FS["title"])
        if k == 0:
            _row_axes_labels(ax, ypos, labels)
            for ri, ref in enumerate(refs):
                yy = ypos[ri * 3]
                ax.text(-0.03, yy + 0.62, S.reference_label(ref), transform=ax.get_yaxis_transform(), ha="right",
                        va="center", fontsize=FS["small"] + 0.5, color=S.REFERENCE_COLOR[ref], fontweight="bold")
        else:
            _hide_y(ax)
        ax.set_ylim(ypos.min() - 0.6, ypos.max() + 0.9)
        axes.append(ax)
    axes[len(axes) // 2].set_xlabel("Context universe / global universe (log scale)")
    hl = [Line2D([], [], ls="none", marker="o", mfc="k", mec="k", ms=3.4, label="Global-fit credibility"),
          Line2D([], [], ls="none", marker="o", mfc="white", mec="k", ms=3.4, label="Context-fit credibility"),
          Rectangle((0, 0), 1, 1, fc=OI["grey"], alpha=0.45, ec="none",
                    label="Prevalence-matched null, edges (95%)"),
          Rectangle((0, 0), 1, 1, fc="none", ec=OI["grey"], lw=0.6, ls=(0, (2, 1)),
                    label="TF-block null (95%)")]
    fig.legend(handles=hl, loc="lower center", ncol=4, bbox_to_anchor=(0.57, 0.0), borderaxespad=0.2,
               columnspacing=0.9)
    return fig


def _where(p, alpha=0.05) -> str:
    """'outside' when a two-sided empirical p is below alpha, else 'inside'."""
    return "outside" if (S.finite(p) and float(p) < alpha) else "inside"


def caption_fig5_universe(fd: FigureData) -> str:
    d = fd.d
    m = d["m"]
    k = d["k"]
    seg = []
    for t in ("immune", "kidney", "lung"):
        r = _maybe(m, tissue=t, reference="pert_pooled", credibility="global_fit")
        if r is None:
            continue
        seg.append(f"{t}: raw {_num(r.ratio_raw)} = base rate {_num(r.ratio_base_rate)} $\\times$ enrichment "
                   f"{_num(r.ratio_enrichment)} {_ci(r.ratio_enrichment_lo, r.ratio_enrichment_hi)}")
    e = _maybe(m, tissue="immune", reference="encode", credibility="global_fit")
    nd = int(m.n_draws.dropna().iloc[0]) if m.n_draws.notna().any() else None
    return (
        "\\textbf{Universe-aware decomposition of context-versus-global differences.} "
        f"For each tissue and reference, the ratio of precision among the {k} highest-credibility evaluable edges "
        f"in the context universe to that in the global universe (raw P@{k} ratio) is the product of the base-rate ratio "
        f"and the Enr@{k} ratio \\eqref{{eq:decomp}}; 95\\% intervals" + (" (post hoc)" if d["ph_ci"] else "")
        + " combine independent TF bootstraps of the two universes. "
        "Filled markers: Family-HB credibility from the global fit; open markers: from the context fit. "
        f"Grey bars: central 95\\% of the raw ratio under the prevalence-matched null ({S.fmt_count(nd)} random sub-universes of the global "
        "universe with the context's numbers of evaluable edges and positives, sampled as edges); dashed boxes: the same null "
        "sampled as whole TF blocks matched to the context's per-TF sizes (post hoc), which keeps the within-TF dependence. "
        + "Pooled perturbation, global-fit credibility: " + "; ".join(seg) + ". "
        + (f"On ENCODE the raw immune ratio is {_num(e.ratio_raw)} although the enrichment ratio is {_num(e.ratio_enrichment)}, "
           f"because the context base rate is lower ({_num(e.base_rate_ctx)} vs {_num(e.base_rate_glob)}); the context P@{k} "
           f"({_num(e.value_ctx)}) lies {_where(e.p_emp_two_sided)} the edge-level null distribution (median "
           f"{_num(e.null_median)}, two-sided empirical $p={_p_text(e.p_emp_two_sided)}$"
           + (", post hoc" if d["ph_p"] else "") + f") and {_where(e.tfb_p_emp_two_sided)} "
           f"the TF-block null distribution (median {_num(e.tfb_null_median)}, $p={_p_text(e.tfb_p_emp_two_sided)}$); the "
           "edge-level null ignores the clustering of edges within TFs and is anti-conservative."
           if e is not None else "")
    )


# ============================================================================ fig6 governance
FIG6_METHODS = ["scgpt_attn", "scgpt_attn_norm", "geneformer_attn", "geneformer_attn_norm", "grnboost2",
                "pearson_abs", "random"]
FIG6_REFS = ["encode", "enets2", "pert_pooled"]
FIG6_METHOD_LABEL = {"scgpt_attn_norm": "scGPT, normalised", "geneformer_attn_norm": "Geneformer, normalised"}


def data_fig6_governance() -> FigureData:
    P = S.Provenance("fig6_governance")
    tiers = P.read("gov_tiers", "scope = dual, tier in {Strong, Moderate, Conditional}; draw interval columns")
    tiers = tiers[(tiers.scope == "dual") & tiers.tier.isin(["Strong", "Moderate", "Conditional", "Exploratory"])].copy()
    tv = P.read("gov_tier_validation", "universe = global (scope global_fit) or tissues (scope dual); "
                "tier in {Strong, Moderate, Conditional}; reference in {encode, enets2, pert_pooled}")
    gl = tv[(tv.universe == "global") & (tv.scope == "global_fit")]
    glob_counts = gl[(gl.reference == "encode") & gl.tier.isin(["Strong", "Moderate"])][["tier", "n_edges", "n_tfs_eval"]]
    tv = tv[(((tv.universe == "global") & (tv.scope == "global_fit")) | ((tv.universe != "global") & (tv.scope == "dual")))
            & tv.tier.isin(["Strong", "Moderate", "Conditional"]) & tv.reference.isin(FIG6_REFS)].copy()
    tv["few_clusters"] = tv.few_clusters.map(S.as_bool)
    k = int(_PRE["topk_per_tf"])
    pm = P.read("gov_promoted_by_method", f"k = {k} (method top-k per TF) and method = all_edges")
    allm = pm[pm.method == "all_edges"].copy()
    pm = pm[(pm.k == k) & pm.method.isin(FIG6_METHODS)].copy()
    P.note("(constants)", "topk_per_tf from r2.common.PRESPECIFIED; MIN_CLUSTERS from r2.governance")
    return FigureData("fig6_governance", P, dict(tiers=tiers, glob_counts=glob_counts, tv=tv, pm=pm, allm=allm, k=k,
                                                 min_clusters=int(_module_constant("governance", "MIN_CLUSTERS"))))


def plot_fig6_governance(fd: FigureData):
    d = fd.d
    fig = _fig(S.FULL_W, 13.0)
    # a: tier counts
    axa = fig.add_axes([0.085, 0.60, 0.25, 0.34])
    groups = ["global", "immune", "kidney", "lung"]
    tiers3 = ["Strong", "Moderate", "Conditional"]
    offs = {"Strong": -0.22, "Moderate": 0.0, "Conditional": 0.22}
    allv = []
    for gi, u in enumerate(groups):
        for t in tiers3:
            if u == "global":
                r = _maybe(d["glob_counts"], tier=t)
                if r is None:
                    continue
                v, lo, hi = r.n_edges, np.nan, np.nan
            else:
                r = _maybe(d["tiers"], tissue=u, tier=t)
                if r is None:
                    continue
                v, lo, hi = r.n_edges, r.n_edges_draw_lo, r.n_edges_draw_hi
            allv += [v, lo, hi]
            point_ci(axa, [gi + offs[t]], [v], [lo], [hi], color=S.TIER_COLOR[t], marker="o", horizontal=False, ms=3.6)
    av = np.array(allv, dtype=float)
    av = av[np.isfinite(av) & (av > 0)]
    set_log_ticks(axa, nice_log_ticks(av.min() * 0.6, av.max() * 1.6, (20, 50, 100, 200, 500, 1000, 2000, 5000)), "y")
    axa.set_ylim(av.min() * 0.6, av.max() * 1.6)
    axa.set_xticks(range(len(groups)))
    axa.set_xticklabels([S.UNIVERSE_LABEL[g] + ("\n(single fit)" if g == "global" else "\n(dual)") for g in groups],
                        fontsize=FS["small"] + 0.5)
    axa.tick_params(axis="x", length=0)
    axa.set_xlim(-0.5, len(groups) - 0.5)
    axa.set_ylabel("Edges in tier (log scale)")
    ht = [Line2D([], [], ls="none", marker="o", mfc=S.TIER_COLOR[t], mec=S.TIER_COLOR[t], ms=3.6, label=t)
          for t in tiers3]
    axa.legend(handles=ht, loc="upper right", borderaxespad=0.1, handletextpad=0.2)
    fig_label(fig, "a", 0.005, 0.975)

    # b: TF-matched O/E by tier
    tv = d["tv"]
    univs = [u for u in S.UNIVERSE_ORDER if u in set(tv.universe)]
    left_b, right_b = 0.42, 0.985
    gapw = 0.012
    wb = (right_b - left_b - gapw * (len(univs) - 1)) / len(univs)
    vals = tv[["oe_tf", "oe_tf_lo", "oe_tf_hi"]].to_numpy(dtype=float)
    vpos = vals[np.isfinite(vals) & (vals > 0)]
    ylo, yhi = max(vpos.min() * 0.7, 0.2), vpos.max() * 1.3
    roff = {"encode": -0.24, "enets2": 0.0, "pert_pooled": 0.24}
    axes_b = []
    for k, u in enumerate(univs):
        ax = fig.add_axes([left_b + k * (wb + gapw), 0.60, wb, 0.34])
        s = tv[tv.universe == u]
        tl = [t for t in tiers3 if t in set(s.tier)]
        for j, t in enumerate(tl):
            for ref in FIG6_REFS:
                r = _maybe(s, tier=t, reference=ref)
                if r is None:
                    continue
                col = S.REFERENCE_COLOR[ref]
                x = j + roff[ref]
                if not (r.oe_tf > 0):
                    ax.plot([x], [ylo * 1.06], ls="none", marker="v", ms=3.4, mfc="white", mec=col, mew=0.8, zorder=3)
                    continue
                lo = r.oe_tf_lo if r.oe_tf_lo > 0 else ylo
                point_ci(ax, [x], [r.oe_tf], [lo], [r.oe_tf_hi], color=col, marker=S.REFERENCE_MARKER[ref],
                         filled=not r.few_clusters, horizontal=False, ms=3.2)
        ref_line(ax, 1.0, "h")
        set_log_ticks(ax, nice_log_ticks(ylo, yhi, (0.25, 0.5, 1, 2, 5, 10, 20)), "y")
        ax.set_ylim(ylo, yhi)
        ax.set_xticks(range(len(tl)))
        ax.set_xticklabels([{"Strong": "Strong", "Moderate": "Mod.", "Conditional": "Cond."}[t] for t in tl],
                           fontsize=FS["small"] + 0.5)
        ax.tick_params(axis="x", length=0)
        ax.set_xlim(-0.55, len(tl) - 0.45)
        ax.set_title(S.UNIVERSE_LABEL[u], fontsize=FS["title"])
        if k == 0:
            ax.set_ylabel("O/E (TF-matched)")
        else:
            ax.tick_params(axis="y", labelleft=False)
        axes_b.append(ax)
    hr = [Line2D([], [], ls="none", marker=S.REFERENCE_MARKER[r], mfc=S.REFERENCE_COLOR[r], mec=S.REFERENCE_COLOR[r],
                 ms=3.4, label=S.reference_label(r)) for r in FIG6_REFS]
    hr.append(Line2D([], [], ls="none", marker="o", mfc="white", mec="k", ms=3.4, label=f"< {d['min_clusters']} TFs"))
    hr.append(Line2D([], [], ls="none", marker="v", mfc="white", mec="k", ms=3.4, label="O/E = 0"))
    fig.legend(handles=hr, loc="upper center", bbox_to_anchor=((left_b + right_b) / 2, 0.515), ncol=5,
               borderaxespad=0.1, handletextpad=0.1, columnspacing=0.8)
    fig_label(fig, "b", left_b - 0.075, 0.975)

    # c: promoted share of top-k predictions
    pm = d["pm"]
    tissues = [t for t in ("immune", "kidney", "lung") if t in set(pm.tissue)]
    meth = [m for m in FIG6_METHODS if m in set(pm.method)]
    yp = np.arange(len(meth))[::-1].astype(float)
    left_c = 0.165
    wc = (0.985 - left_c - 0.03 * (len(tissues) - 1)) / len(tissues)
    xmax = float(np.nanmax(pm.frac_promoted_hi)) * 100 * 1.08
    axes_c = []
    for k, t in enumerate(tissues):
        ax = fig.add_axes([left_c + k * (wc + 0.03), 0.075, wc, 0.35])
        s = pm[pm.tissue == t].set_index("method")
        al = _maybe(d["allm"], tissue=t)
        if al is not None:
            ax.axvline(al.frac_promoted * 100, color=OI["grey"], lw=0.7, ls="--", zorder=0.5)
        for i, mth in enumerate(meth):
            if mth not in s.index:
                continue
            r = s.loc[mth]
            stl = S.SCORE_STYLE[mth]
            ax.plot([r.expected_frac_promoted_tf * 100], [yp[i]], ls="none", marker="|", ms=7, mew=1.2,
                    color=OI["black"], zorder=4)
            point_ci(ax, [r.frac_promoted * 100], [yp[i]], [r.frac_promoted_lo * 100], [r.frac_promoted_hi * 100],
                     color=stl.color, marker=stl.marker, ms=3.6)
        ax.set_xlim(-0.035 * xmax, xmax)
        ax.set_title(S.UNIVERSE_LABEL[t], fontsize=FS["title"])
        if k == 0:
            _row_axes_labels(ax, yp, [FIG6_METHOD_LABEL.get(m_, S.score_label(m_)) for m_ in meth])
        else:
            _hide_y(ax)
        ax.set_ylim(yp.min() - 0.6, yp.max() + 0.6)
        axes_c.append(ax)
    axes_c[len(axes_c) // 2].set_xlabel(f"Promoted (dual Strong or Moderate) share of the top-{d['k']} predictions per TF (%)")
    hc = [Line2D([], [], ls="none", marker="|", ms=7, mew=1.2, color="k", label="TF-matched expectation"),
          Line2D([], [], ls="--", lw=0.7, color=OI["grey"], label="All universe edges")]
    axes_c[-1].legend(handles=hc, loc="lower right", borderaxespad=0.1)
    fig_label(fig, "c", 0.005, 0.455)
    return fig


def caption_fig6_governance(fd: FigureData) -> str:
    d = fd.d
    c = S.fmt_count
    tiers = d["tiers"]
    tv = d["tv"]
    pm = d["pm"]
    gc = d["glob_counts"].set_index("tier")
    seg_a = []
    for t in ("immune", "kidney", "lung"):
        s = tiers[tiers.tissue == t].set_index("tier")
        seg_a.append(f"{t} {c(s.loc['Strong'].n_edges)}/{c(s.loc['Moderate'].n_edges)}/{c(s.loc['Conditional'].n_edges)}")
    g = tv[tv.universe == "global"].set_index(["tier", "reference"])

    def oe(t, ref):
        r = g.loc[(t, ref)]
        return f"{_num(r.oe_tf)} {_ci(r.oe_tf_lo, r.oe_tf_hi)}"

    def share(t, m):
        r = _one(pm, tissue=t, method=m)
        return (f"{S.fmt_ratio(r.frac_promoted * 100, 2)}\\% vs {S.fmt_ratio(r.expected_frac_promoted_tf * 100, 2)}\\% "
                f"expected")

    return (
        "\\textbf{Claim governance.} "
        "(a) Number of edges per tier: single-fit BFDR tiers in the global universe (Strong "
        f"{c(gc.loc['Strong'].n_edges)}, Moderate {c(gc.loc['Moderate'].n_edges)}) and dual-universe tiers in the tissue "
        "universes (Strong/Moderate/Conditional: " + "; ".join(seg_a) + "), with 95\\% intervals over posterior draws "
        "of the credibilities (tissue universes). "
        "(b) Held-out positives in each tier relative to the TF-matched expectation, with 95\\% TF-bootstrap intervals "
        f"(open markers: fewer than {d['min_clusters']} evaluable TFs; triangles: no positive observed; intervals reaching 0 "
        "are cut at the axis). Global universe, Strong tier: "
        f"ENCODE {oe('Strong', 'encode')}, ENETS2 {oe('Strong', 'enets2')}, pooled perturbation {oe('Strong', 'pert_pooled')}; "
        f"Moderate tier: {oe('Moderate', 'encode')}, {oe('Moderate', 'enets2')}, {oe('Moderate', 'pert_pooled')}. "
        f"(c) Share of each method's top-{d['k']} predictions per TF that are promoted (95\\% TF-bootstrap intervals), against "
        "the share expected if the method picked random targets of the same TFs (bars) and the share among all "
        f"universe edges (dashed). scGPT attention: immune {share('immune', 'scgpt_attn')}; Geneformer attention: immune "
        f"{share('immune', 'geneformer_attn')}; GRNBoost2: immune {share('immune', 'grnboost2')}."
    )


# ============================================================================ fig7 hubs
FIG7_SCORES = [("family_hb", "strong_bfdr05"), ("em", "top500"), ("fisher", "top500"),
               ("dorothea_level_heldin", "top500"), ("random", "top500")]
SCENARIO_TICK = {"null_independent": "Independent", "bio_hub": "Biological\nhubs", "copying": "Hub\ncopying"}
FIG7_SIM_METHODS = ["oracle", "family_hb", "em", "dependence", "fisher", "stouffer", "majority", "union",
                    "vote_fraction"]


def data_fig7_hubs() -> FigureData:
    P = S.Provenance("fig7_hubs")
    dc = P.read("heldout_degree_calibration", "reference in {encode, enets2, pert_pooled}; (score, set) in "
                + ", ".join(f"({a}, {b})" for a, b in FIG7_SCORES))
    keep = pd.Series(False, index=dc.index)
    for sc, st in FIG7_SCORES:
        keep |= (dc.score == sc) & (dc.set == st)
    dc = dc[keep & dc.reference.isin(["encode", "enets2", "pert_pooled"])].copy()
    dc["few_tfs"] = dc.few_tfs.map(S.as_bool)
    sm = P.read("sim_hub_summary", "set_rule = size_matched, metric in {over_promotion_lit_q5, "
                "over_promotion_top_decile}; methods " + ", ".join(FIG7_SIM_METHODS))
    sm = sm[(sm.set_rule == "size_matched") & sm.metric.isin(["over_promotion_lit_q5", "over_promotion_top_decile"])
            & sm.method.isin(FIG7_SIM_METHODS + ["family_regularized"])].copy()
    sz = P.read("sim_hub_summary", "set_rule = size_matched, metric = set_size, method = family_hb")
    sz = sz[(sz.set_rule == "size_matched") & (sz.metric == "set_size") & (sz.method == "family_hb")]
    P.note("(constants)", "FEW_TFS from r2.heldout_eval")
    # largest attainable index in the top quintile: 1 / share of positives there (derived from share_pos)
    q5 = dc[(dc.score == "family_hb") & (dc.set == "strong_bfdr05") & (dc.quintile == 5)]
    ceil = {r.reference: 1.0 / r.share_pos for r in q5.itertuples() if r.share_pos > 0}
    return FigureData("fig7_hubs", P, dict(dc=dc, sm=sm, sz=sz, ceil=ceil,
                                           few_tfs=int(_module_constant("heldout_eval", "FEW_TFS"))))


def _set_q(st: str) -> str:
    """'strong_bfdr05' -> '0.05' (threshold encoded in the set name of heldout_degree_calibration)."""
    import re as _re
    m = _re.search(r"bfdr(\d+)$", st)          # two digits = hundredths (bfdr05 -> 0.05)
    return f"{int(m.group(1)) / 100:.2f}" if m else ""


def _fig7_label(sc, st):
    base = S.score_label(sc)
    if st.startswith("strong_bfdr"):
        return base + (" (MAP)" if sc == "family_hb" else "") + f", BFDR $\\leq$ {_set_q(st)} set"
    return base + f", top {st.replace('top', '')}"


def plot_fig7_hubs(fd: FigureData):
    d = fd.d
    fig = _fig(S.FULL_W, 12.6)
    dc = d["dc"]
    refs = [r for r in ("encode", "enets2", "pert_pooled") if r in set(dc.reference)]
    left, right = 0.075, 0.70
    gapw = 0.02
    wa = (right - left - gapw * (len(refs) - 1)) / len(refs)
    series = [(sc, st) for sc, st in FIG7_SCORES if ((dc.score == sc) & (dc.set == st)).any()]
    ns = len(series)
    offs = np.linspace(-0.24, 0.24, ns)
    ytop = 3.0
    axes = []
    for k, ref in enumerate(refs):
        ax = fig.add_axes([left + k * (wa + gapw), 0.58, wa, 0.36])
        for si, (sc, st) in enumerate(series):
            s = dc[(dc.reference == ref) & (dc.score == sc) & (dc.set == st)].sort_values("quintile")
            stl = S.SCORE_STYLE[sc]
            filled = st != "strong_bfdr05"
            x = s.quintile.to_numpy(dtype=float) + offs[si]
            ax.plot(x, s.over_promotion, color=stl.color, lw=0.5, alpha=0.6, zorder=2,
                    ls="--" if st == "strong_bfdr05" else "-")
            for few, alpha in ((False, 1.0), (True, 0.4)):
                mm = (s.few_tfs == few).to_numpy()
                if mm.any():
                    point_ci(ax, x[mm], s.over_promotion[mm], s.over_promotion_lo[mm], s.over_promotion_hi[mm],
                             color=stl.color, marker=stl.marker, filled=filled, horizontal=False, ms=3.0, lw=0.6,
                             alpha=alpha)
        ref_line(ax, 1.0, "h")
        if ref in d["ceil"]:
            ax.plot([4.6, 5.4], [d["ceil"][ref]] * 2, color="k", lw=1.0, ls=(0, (1, 1)), zorder=5)
        ax.set_ylim(0, ytop)
        ax.set_xticks([1, 2, 3, 4, 5])
        q = dc[(dc.reference == ref) & (dc.score == series[0][0]) & (dc.set == series[0][1])].sort_values("quintile")
        ax.set_xticklabels([f"Q{int(r.quintile)}" for r in q.itertuples()], fontsize=FS["small"] + 0.5)
        ax.set_xlim(0.5, 5.5)
        ax.set_title(S.reference_label(ref), fontsize=FS["title"])
        if k == 0:
            ax.set_ylabel("Over-promotion index")
        else:
            ax.tick_params(axis="y", labelleft=False)
        axes.append(ax)
    axes[1].set_xlabel("TF literature-degree quintile (Q5: most literature edges per TF)")
    hl = [Line2D([], [], color=S.score_color(sc), lw=0.5, ls="--" if st == "strong_bfdr05" else "-",
                 marker=S.score_marker(sc), ms=3.2, mec=S.score_color(sc),
                 mfc="white" if st == "strong_bfdr05" else S.score_color(sc), label=_fig7_label(sc, st))
          for sc, st in series]
    hl.append(Line2D([], [], color="k", lw=1.0, ls=(0, (1, 1)), label="Largest possible index in Q5"))
    fig.legend(handles=hl, loc="upper left", bbox_to_anchor=(right + 0.01, 0.95), borderaxespad=0.1,
               title="Held-out data (global universe)", title_fontsize=FS["small"] + 0.5, alignment="left")
    fig_label(fig, "a", 0.005, 0.975)

    # b, c: simulation with known truth
    sm = d["sm"]
    scen = [s for s in S.SCENARIO_ORDER if s in set(sm.scenario)]
    meths = [m for m in FIG7_SIM_METHODS if m in set(sm.method)]   # family_regularized: caption only
    offs2 = np.linspace(-0.36, 0.36, len(meths))
    for pi, (metric, title, lab) in enumerate([
            ("over_promotion_lit_q5", "Top literature-degree quintile", "b"),
            ("over_promotion_top_decile", "Top popularity decile", "c")]):
        ax = fig.add_axes([0.075 + pi * 0.355, 0.075, 0.27, 0.34])
        for mi, mth in enumerate(meths):
            stl = S.SCORE_STYLE[mth]
            for si, sc in enumerate(scen):
                r = _maybe(sm, scenario=sc, method=mth, metric=metric)
                if r is None:
                    continue
                point_ci(ax, [si + offs2[mi]], [r["mean"]], [r.ci_lo], [r.ci_hi], color=stl.color, marker=stl.marker,
                         horizontal=False, ms=3.4 if mth != "oracle" else 5.0)
        ref_line(ax, 1.0, "h")
        ax.set_xticks(range(len(scen)))
        ax.set_xticklabels([SCENARIO_TICK.get(s, s) for s in scen], fontsize=FS["small"] + 0.5, linespacing=1.0)
        ax.tick_params(axis="x", length=0)
        ax.set_xlim(-0.55, len(scen) - 0.45)
        ax.set_ylim(0, None)
        ax.set_title(title, fontsize=FS["title"])
        ax.set_ylabel("Over-promotion index (simulation)")
        fig_label(fig, lab, 0.005 + pi * 0.355, 0.475)
    hs = [Line2D([], [], ls="none", marker=S.score_marker(m), ms=3.4 if m != "oracle" else 5.0, mec=S.score_color(m),
                 mfc=S.score_color(m), label=S.score_label(m)) for m in meths]
    fig.legend(handles=hs, loc="lower left", bbox_to_anchor=(0.73, 0.075), borderaxespad=0.1,
               title="Simulation (equal set size)", title_fontsize=FS["small"] + 0.5, alignment="left")
    return fig


def caption_fig7_hubs(fd: FigureData) -> str:
    d = fd.d
    dc = d["dc"]
    sm = d["sm"]

    def q5(ref, sc, st):
        r = _one(dc, reference=ref, score=sc, set=st, quintile=5)
        return f"{_num(r.over_promotion)} {_ci(r.over_promotion_lo, r.over_promotion_hi)}"

    r5 = _one(dc, reference="encode", score="family_hb", set="strong_bfdr05", quintile=5)
    qq = dc[(dc.reference == "encode") & (dc.score == "family_hb") & (dc.set == "strong_bfdr05")].sort_values("quintile")
    qr = ", ".join(f"{S.fmt_count(r.degree_min)}--{S.fmt_count(r.degree_max)}" for r in qq.itertuples())

    def sim(sc, mth, metric="over_promotion_lit_q5"):
        r = _one(sm, scenario=sc, method=mth, metric=metric)
        return f"{_num(r['mean'])} {_ci(r.ci_lo, r.ci_hi)}"

    q = _set_q("strong_bfdr05")
    topn = next((st.replace("top", "") for _, st in FIG7_SCORES if st.startswith("top")), "")
    ceil = "; ".join(f"{S.reference_label(ref)} {_num(d['ceil'][ref])}" for ref in ("encode", "enets2", "pert_pooled")
                     if ref in d["ceil"])
    ph_few = bool(dc.post_hoc_columns.astype(str).str.contains("few_tfs").any()) if "post_hoc_columns" in dc else False

    nrep = int(sm[sm.method == "family_hb"].n_reps.min())
    a = sm[sm.method == "family_regularized"].set_index(["scenario", "metric"])["mean"]
    b = sm[sm.method == "em"].set_index(["scenario", "metric"])["mean"]
    same = len(a) and len(a) == len(b) and bool((a.round(2) == b.reindex(a.index).round(2)).all())
    fr_txt = ("Family-regularized coincides with EM to two decimals and is not shown. " if same
              else "Family-regularized is not shown. ")
    return (
        "\\textbf{Over-promotion of well-studied TFs.} "
        "The over-promotion index of a selected set for a group of TFs is the share of the set in the group divided by the "
        "share of positives (a) or of true edges (b, c) in the group (1: proportional to the positives; in (b, c) the "
        "oracle posterior shows the level the evidence supports). "
        f"(a) Held-out data, global universe: index by quintile of TF literature degree for the Family-HB (MAP) BFDR "
        f"$\\leq {q}$ set and for the top {topn} edges of selected scores, with 95\\% TF-bootstrap intervals holding the set "
        f"fixed (faded{' (post hoc)' if ph_few else ''}: fewer than {d['few_tfs']} evaluable TFs in the quintile). Literature "
        "edges per TF in quintiles Q1--Q5: " + qr + ". "
        "The top quintile holds "
        f"{S.fmt_pct(r5.share_set)}\\% of the Family-HB set but {S.fmt_pct(r5.share_pos)}\\% of ENCODE positives; "
        f"index of the Family-HB set in the top quintile: ENCODE {q5('encode', 'family_hb', 'strong_bfdr05')}, ENETS2 "
        f"{q5('enets2', 'family_hb', 'strong_bfdr05')}, pooled perturbation {q5('pert_pooled', 'family_hb', 'strong_bfdr05')}; "
        f"random top {topn} on ENCODE: {q5('encode', 'random', 'top' + topn)}. In the top quintile the index cannot exceed "
        f"1/(share of positives) (dotted; {ceil}), so every informative score is near this ceiling and panel (a) does not separate "
        "estimators; the sets also differ in size. "
        f"(b, c) Simulation with known truth ({nrep} replicates per scenario; mean with 95\\% interval over replicates): "
        f"index of sets of the size of the Family-HB BFDR $\\leq {q}$ set at TFs in the top literature-degree quintile "
        "(b) and in the top popularity decile (c), relative to the true edges. Under hub-concentrated copying (b): "
        f"oracle posterior {sim('copying', 'oracle')}, Family-HB {sim('copying', 'family_hb')}, EM {sim('copying', 'em')}, "
        f"Fisher {sim('copying', 'fisher')}. " + fr_txt + "All simulation rows are post hoc."
    )


# ============================================================================ fig8 replication
FIG8_STUDIES = ["dixit2016", "norman2019", "papalexi2021", "replogle2022", "shifrut2018", "adamson2016"]
FIG8_PROFILE_ORDER = ["primary", "mu_sd0.5", "mu_sd2", "sigma_hn0.25", "sigma_hn1", "diffuse", "skeptical",
                      "skeptical_sigma0.25", "tf_random_slope"]


def data_fig8_replication() -> FigureData:
    P = S.Provenance("fig8_replication")
    u = P.read("rep_units", "level = study, comparator = universe, status = evaluable")
    u = u[(u.level == "study") & (u.comparator == "universe")].copy()
    u_eval = u[u.status == "evaluable"].copy()
    ps = P.read("rep_prior_sensitivity", "level = study, parameter in {mu, theta_new}")
    ps = ps[(ps.level == "study") & ps.parameter.isin(["mu", "theta_new"])].copy()
    pr = P.read("rep_pass_rates", "level = study, comparator = universe, subset in {all_evaluable, adequately_powered}")
    pr = pr[(pr.level == "study") & (pr.comparator == "universe") & pr.subset.isin(["all_evaluable", "adequately_powered"])]
    pw = P.read("rep_power", "level = study, comparator = universe: alpha, true_enrichment, target_power")
    pw = pw[(pw.level == "study") & (pw.comparator == "universe")]
    dl = P.read("eval_analysis_decisions", "module = replication: status of rep_attainable, rep_boot_clusters")
    dst = {i: str(st) for i, st in zip(dl.id, dl.status)}
    P.note("(constants)", "replication_endpoint_enrichment from r2.common.PRESPECIFIED; MIN_BOOT_CLUSTERS from r2.replication")
    return FigureData("fig8_replication", P, dict(
        u=u_eval, u_all=u, ps=ps, pr=pr, endpoint=float(_PRE["replication_endpoint_enrichment"]),
        ph_att=dst.get("rep_attainable") == "post_hoc", ph_boot=dst.get("rep_boot_clusters") == "post_hoc",
        alpha=float(pw.alpha.dropna().iloc[0]), rr=float(pw.true_enrichment.dropna().iloc[0]),
        target_power=float(pw.target_power.dropna().iloc[0]),
        min_boot=int(_module_constant("replication", "MIN_BOOT_CLUSTERS"))))


def _power_class(r) -> str:
    if r.endpoint_status == "endpoint_unattainable":
        return "unattainable"
    return "powered" if S.as_bool(r.adequately_powered) else "underpowered"


POWER_MARKER = {"powered": dict(marker="o", filled=True), "underpowered": dict(marker="o", filled=False),
                "unattainable": dict(marker="X", filled=True)}
POWER_LABEL = {"powered": "Adequately powered", "underpowered": "Below support floor",
               "unattainable": "Endpoint unattainable"}


def plot_fig8_replication(fd: FigureData):
    d = fd.d
    u = d["u"]
    fsm = FS["small"] + 0.5
    fig = _fig(S.FULL_W, 8.6)
    studies = [s for s in FIG8_STUDIES if s in set(u.unit)]
    rows, ypos, heads = [], [], []
    y = 0.0
    for s in studies:
        rr = [_maybe(u, unit=s, universe=un) for un in S.UNIVERSE_ORDER]
        rr = [r for r in rr if r is not None]
        if not rr:
            continue
        heads.append((y, S.reference_label(s)))
        y -= 0.85
        for r in rr:
            rows.append(r)
            ypos.append(y)
            y -= 0.85
        y -= 0.25
    ypos = np.array(ypos)
    ulabels = [S.UNIVERSE_LABEL[r.universe] for r in rows]
    ytop, ybot = heads[0][0] + 0.5, ypos.min() - 0.6
    ax_b, ax_h = 0.12, 0.70

    # a: O/E per study x universe
    axa = fig.add_axes([0.105, ax_b, 0.33, ax_h])
    oe_all = np.r_[[r.enrichment for r in rows], [r.enrichment_boot_lo for r in rows],
                   [r.enrichment_boot_hi for r in rows]].astype(float)
    pos = oe_all[np.isfinite(oe_all) & (oe_all > 0)]
    xlo, xhi = min(pos.min() * 0.7, 0.4), pos.max() * 1.4
    for i, r in enumerate(rows):
        pc = _power_class(r)
        col = S.UNIVERSE_COLOR[r.universe]
        mk = POWER_MARKER[pc]
        if not (r.enrichment > 0):
            axa.plot([xlo * 1.06], [ypos[i]], ls="none", marker="<", ms=3.6, mfc="white" if not mk["filled"] else col,
                     mec=col, mew=0.8)
            axa.annotate(f"O = {S.fmt_count_plain(r.observed_promoted_positives)}, E = "
                         f"{S.fmt_ratio(r.expected_promoted_positives, 1)}", (xlo * 1.06, ypos[i]), xytext=(5, 0),
                         textcoords="offset points", va="center", fontsize=fsm)
            continue
        lo = r.enrichment_boot_lo if S.finite(r.enrichment_boot_lo) and r.enrichment_boot_lo > 0 else np.nan
        point_ci(axa, [r.enrichment], [ypos[i]], [lo], [r.enrichment_boot_hi], color=col, marker=mk["marker"],
                 filled=mk["filled"], ms=3.8)
    ref_line(axa, 1.0)
    axa.axvline(d["endpoint"], color="k", lw=0.7, ls="--", zorder=0.6)
    set_log_ticks(axa, nice_log_ticks(xlo, xhi, (0.25, 0.5, 1, 1.5, 2, 4, 8)), "x")
    axa.set_xlim(xlo, xhi)
    axa.set_yticks(ypos)
    axa.set_yticklabels(ulabels, fontsize=fsm)
    axa.tick_params(axis="y", length=0)
    for yh, lab in heads:
        axa.text(-0.02, yh, lab, transform=axa.get_yaxis_transform(), ha="right", va="center", fontsize=fsm,
                 fontweight="bold")
    axa.set_ylim(ybot, ytop)
    axa.set_xlabel("Stratified enrichment O/E of promoted edges")
    fig_label(fig, "a", 0.005, 0.995)

    # b: minimum detectable enrichment
    axb = fig.add_axes([0.455, ax_b, 0.13, ax_h])
    mde = np.array([r.mde_rr for r in rows], dtype=float)
    fin = mde[np.isfinite(mde)]
    for i, r in enumerate(rows):
        col = S.UNIVERSE_COLOR[r.universe]
        if np.isfinite(mde[i]):
            axb.plot([mde[i]], [ypos[i]], ls="none", marker="s", ms=3.4, mfc=col, mec=col)
    axb.axvline(d["endpoint"], color="k", lw=0.7, ls="--", zorder=0.6)
    axb.axvline(d["rr"], color=OI["grey"], lw=0.7, ls=":", zorder=0.6)
    set_log_ticks(axb, nice_log_ticks(fin.min() * 0.8, fin.max() * 1.3, (1, 2, 4, 8, 16, 32)), "x")
    axb.set_xlim(min(fin.min() * 0.8, 1.2), fin.max() * 1.3)
    _hide_y(axb)
    axb.set_ylim(ybot, ytop)
    axb.set_xlabel(f"MDE at {S.fmt_pct(d['target_power'])}% power")
    fig_label(fig, "b", 0.445, 0.995)

    # shared legend for a and b (power class = marker, universe = colour)
    hp = [Line2D([], [], ls="none", marker=POWER_MARKER[p]["marker"], ms=3.6, mec="k",
                 mfc="k" if POWER_MARKER[p]["filled"] else "white", label=POWER_LABEL[p])
          for p in ("powered", "underpowered", "unattainable") if any(_power_class(r) == p for r in rows)]
    hp += [Line2D([], [], ls="none", marker="s", ms=3.4, mfc=S.UNIVERSE_COLOR[x], mec=S.UNIVERSE_COLOR[x],
                  label=S.UNIVERSE_LABEL[x]) for x in S.UNIVERSE_ORDER if any(r.universe == x for r in rows)]
    hp.append(Line2D([], [], color="k", lw=0.7, ls="--", label=f"Endpoint {S.fmt_ratio(d['endpoint'], 1)}"))
    hp.append(Line2D([], [], color=OI["grey"], lw=0.7, ls=":", label=f"Design enrichment {S.fmt_ratio(d['rr'], 0)}"))
    fig.legend(handles=hp, loc="upper left", bbox_to_anchor=(0.02, 0.97), ncol=5, borderaxespad=0.1,
               columnspacing=0.8, handletextpad=0.2)

    # c: pooled Bayesian effect under prior profiles
    ps = d["ps"]
    profs = [p for p in FIG8_PROFILE_ORDER if p in set(ps.profile)]
    yc = np.arange(len(profs))[::-1].astype(float)
    axc = fig.add_axes([0.775, ax_b, 0.21, ax_h])
    for i, p in enumerate(profs):
        r = _maybe(ps, profile=p, parameter="mu")
        rn = _maybe(ps, profile=p, parameter="theta_new")
        if rn is not None:
            point_ci(axc, [rn.or_median], [yc[i] - 0.18], [rn.or_q025], [rn.or_q975], color=OI["grey"], marker="o",
                     filled=False, ms=3.2)
        if r is not None:
            point_ci(axc, [r.or_median], [yc[i] + 0.14], [r.or_q025], [r.or_q975],
                     color=OI["vermillion"] if p == "primary" else "k", marker="D", ms=3.2)
    ref_line(axc, 1.0)
    axc.axvline(d["endpoint"], color="k", lw=0.7, ls="--", zorder=0.6)
    vals = ps[["or_q025", "or_q975"]].to_numpy(dtype=float)
    set_log_ticks(axc, nice_log_ticks(vals.min() * 0.9, vals.max() * 1.1, (0.25, 0.5, 1, 2, 4, 8)), "x")
    axc.set_xlim(vals.min() * 0.85, vals.max() * 1.15)
    _row_axes_labels(axc, yc, [_prior_label(ps, p) for p in profs])
    axc.set_ylim(yc.min() - 0.6, yc.max() + 0.6)
    axc.set_xlabel("Odds ratio (promoted vs other)")
    hcl = [Line2D([], [], ls="none", marker="D", ms=3.2, mfc="k", mec="k", label="Pooled $e^{\\mu}$"),
           Line2D([], [], ls="none", marker="o", ms=3.2, mfc="white", mec=OI["grey"], label="New study $e^{\\theta}$")]
    axc.legend(handles=hcl, loc="lower left", bbox_to_anchor=(-0.02, 1.0), ncol=2, borderaxespad=0.1,
               columnspacing=0.6, handletextpad=0.2)
    fig_label(fig, "c", 0.63, 0.885)
    return fig



def _prior_text(x: str) -> str:
    """'Normal(0, 0.5)' -> 'N(0,0.5^2)', 'HalfNormal(0.25)' -> 'HN(0.25)' (prior strings of rep_prior_sensitivity,
    whose second Normal argument is the standard deviation)."""
    import re as _re
    x = str(x).replace(" ", "")
    m = _re.fullmatch(r"Normal\(([-0-9.]+),([0-9.]+)\)", x)
    if m:
        return f"N({m.group(1)},{m.group(2)}^2)"
    return x.replace("HalfNormal", "HN").replace("Normal", "N")


def _prior_label(ps: pd.DataFrame, profile: str) -> str:
    """Axis label of a prior profile, built from its prior_mu / prior_sigma columns (no typed numbers)."""
    if profile == "primary":
        return "Primary"
    if profile == "tf_random_slope":
        return "TF random slope$^{\\dagger}$"
    r = ps[(ps.profile == profile) & (ps.parameter == "mu")]
    pr = ps[(ps.profile == "primary") & (ps.parameter == "mu")]
    if not len(r) or not len(pr):
        return profile
    r, prim = r.iloc[0], pr.iloc[0]
    parts = []
    if r.prior_mu != prim.prior_mu:
        parts.append("$\\mu\\sim " + _prior_text(r.prior_mu) + "$")
    if r.prior_sigma != prim.prior_sigma:
        parts.append("$\\sigma\\sim " + _prior_text(r.prior_sigma) + "$")
    return ",\n".join(parts) if parts else profile


def caption_fig8_replication(fd: FigureData) -> str:
    d = fd.d
    u = d["u"]
    ps = d["ps"]
    g = u[u.universe == "global"]
    seg = []
    for r in g.itertuples():
        if r.enrichment > 0:
            seg.append(f"{S.reference_label(r.unit)} {_num(r.enrichment)} {_ci(r.enrichment_boot_lo, r.enrichment_boot_hi)} "
                       f"(power {_num(r.power)}, MDE {_num(r.mde_rr)})")
        else:
            seg.append(f"{S.reference_label(r.unit)}: {S.fmt_count_plain(r.observed_promoted_positives)} observed vs "
                       f"{S.fmt_ratio(r.expected_promoted_positives, 1)} expected promoted positives (power {_num(r.power)}, "
                       f"MDE {S.fmt_ratio(r.mde_rr, 1)})")
    pr = d["pr"]
    ga = _maybe(pr, universe="global", subset="all_evaluable")
    gp = _maybe(pr, universe="global", subset="adequately_powered")
    mu = _one(ps, profile="primary", parameter="mu")
    sk = _maybe(ps, profile="skeptical", parameter="mu")
    # alternative PRIORS only; the TF random-slope profile changes the model, not the prior
    pmin = ps[(ps.parameter == "mu") & ~ps.profile.isin(["primary", "tf_random_slope"])]
    nprof = int(len(pmin))
    rs = _maybe(ps, profile="tf_random_slope", parameter="mu")
    na = "non-evaluable studies omitted"
    return (
        "\\textbf{Cross-cohort replication check.} "
        "(a) Stratified enrichment (observed over TF-stratified expected perturbation positives among promoted evaluable "
        f"edges) per study and universe (global universe primary, tissue universes secondary), with 95\\% TF-bootstrap "
        f"intervals where at least {d['min_boot']} TFs contribute" + (" (post hoc rule)" if d["ph_boot"] else "")
        + "; intervals reaching 0 are cut at the axis. Marker shape gives the power class locked before any replication outcome was "
        "computed (filled: adequately powered; open: below the support floor)"
        + ("; the endpoint-unattainable class (X; the positive-rate margins cannot reach the endpoint) was added post hoc"
           if d["ph_att"] else "")
        + f"; dashed line: endpoint {S.fmt_ratio(d['endpoint'], 1)} ({na}). Global universe: " + "; ".join(seg) + ". "
        + (f"Pass rate {ga.n_pass}/{ga.n_units} evaluable studies (a study below the support floor is reported as "
            f"underpowered but counts as not passing) and {gp.n_pass}/{gp.n_units} adequately powered studies. "
           if ga is not None and gp is not None else "")
        + f"(b) Minimum detectable enrichment at {S.fmt_pct(d['target_power'])}\\% power and one-sided $\\alpha={S.fmt_ratio(d['alpha'])}$ "
        f"from the design alone (dotted: design enrichment {S.fmt_ratio(d['rr'], 0)}). "
        "(c) Pooled within-TF odds ratio $e^{\\mu}$ and the predicted odds ratio of a new study in the Bayesian "
        f"random-effects model under the primary and {nprof} alternative prior specifications"
        + (" and a model with a TF-level random slope$^{\\dagger}$" if rs is not None else "")
        + " (median and 95\\% credible interval; N, normal; HN, half-normal" + ("; $^{\\dagger}$post hoc" if rs is not None else "") + "). "
        f"Primary prior: {_num(mu.or_median)} {_ci(mu.or_q025, mu.or_q975)}, "
        f"$P(e^{{\\mu}}\\geq {S.fmt_ratio(d['endpoint'], 1)})={_num(mu.p_or_ge_endpoint)}$"
        + (f"; sceptical prior $\\mu\\sim {_prior_text(sk.prior_mu)}$: {_num(sk.or_median)} {_ci(sk.or_q025, sk.or_q975)}, "
           f"$P={_num(sk.p_or_ge_endpoint)}$" if sk is not None else "")
        + f"; across the alternative priors $P(e^{{\\mu}}\\geq {S.fmt_ratio(d['endpoint'], 1)})$ ranges from "
        f"{_num(pmin.p_or_ge_endpoint.min())} to {_num(pmin.p_or_ge_endpoint.max())}"
        + (f"; TF random slope: {_num(rs.or_median)} {_ci(rs.or_q025, rs.or_q975)}, $P={_num(rs.p_or_ge_endpoint)}$"
           if rs is not None else "") + "."
    )


# ============================================================================ registry / CLI
FIGURES = {
    "fig1_design": (data_fig1_design, plot_fig1_design, caption_fig1_design),
    "fig2_model": (data_fig2_model, plot_fig2_model, caption_fig2_model),
    "fig3_robustness": (data_fig3_robustness, plot_fig3_robustness, caption_fig3_robustness),
    "fig4_heldout": (data_fig4_heldout, plot_fig4_heldout, caption_fig4_heldout),
    "fig5_universe": (data_fig5_universe, plot_fig5_universe, caption_fig5_universe),
    "fig6_governance": (data_fig6_governance, plot_fig6_governance, caption_fig6_governance),
    "fig7_hubs": (data_fig7_hubs, plot_fig7_hubs, caption_fig7_hubs),
    "fig8_replication": (data_fig8_replication, plot_fig8_replication, caption_fig8_replication),
}

# tables each figure depends on (for the staleness check of the final report)
FIGURE_SOURCES = {
    "fig1_design": ["source_table", "evidence_summary", "universe_panels", "heldout_chip_table", "cohort_table",
                    "gov_tiers", "gov_tier_validation"],
    "fig2_model": ["fit_source_parameters", "modes_bridge", "modes_simulation", "modes_simulation_replicates",
                   "modes_faithful_design", "fit_modes"],
    "fig3_robustness": ["rob_method_ranking", "rob_edge_stability", "rob_estimator_contrasts"],
    "fig4_heldout": ["heldout_estimators", "heldout_evidence_status"],
    "fig5_universe": ["heldout_universe_decomposition", "heldout_downsampling_null"],
    "fig6_governance": ["gov_tiers", "gov_tier_validation", "gov_promoted_by_method"],
    "fig7_hubs": ["heldout_degree_calibration", "sim_hub_summary"],
    "fig8_replication": ["rep_units", "rep_prior_sensitivity", "rep_pass_rates", "rep_power"],
}


def make(name: str, out_dir: Path | None = None) -> dict:
    S.apply_style()
    data_fn, plot_fn, cap_fn = FIGURES[name]
    fd = data_fn()
    fig = plot_fn(fd)
    caption = cap_fn(fd)
    return save_figure(fig, fd, caption, out_dir)


def make_all(names=None, out_dir: Path | None = None) -> dict:
    out = {}
    for n in names or FIGURES:
        out[n] = make(n, out_dir)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("names", nargs="*", help="figure names (default: all)")
    ap.add_argument("--out-dir", default=None)
    a = ap.parse_args(argv)
    res = make_all(a.names or None, Path(a.out_dir) if a.out_dir else None)
    for n, r in res.items():
        print(f"{n}: {r['pdf']}")


if __name__ == "__main__":
    main()
