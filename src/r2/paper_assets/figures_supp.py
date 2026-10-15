"""Supplementary figures figS1..figS8 of the SAGMB R2 paper.

Every plotted value is read from ``data/r2/tables/*.csv`` (via
``style.Provenance``); nothing is typed by hand.  Each figure is written as a
vector PDF and a 300-dpi PNG to ``paper/r2/figures/`` together with
``<name>_caption.txt`` (``% provenance:`` comment with the source CSV files, their
modification times and the row filters used, then the LaTeX caption) and the
float wrapper ``<name>_float.tex`` (``style.write_float``).  The provenance text is
also embedded in the PDF (``Subject``) and PNG (``Description``) metadata.

Figures
-------
figS1_ppc               posterior predictive check: any-report rate by number of
                        observing literature annotators (fit_ppc.csv)
figS2_obs_model         informative-observation model diagnostics (obs_params.csv,
                        obs_fit_summary.csv)
figS3_deconcentration   label-free deconcentration frontier (gov_deconcentration.csv)
figS4_gov_grid          governance sensitivity grid (gov_sensitivity_grid.csv)
figS5_compute_scaling   compute scaling, log-log (scale_summary.csv)
figS6_fm_attention_null foundation-model attention vs its 1/L uniform null
                        (fm_attention_null.csv, heldout_methods_context.csv)
figS7_recovery_calibration  simulation recovery: credibility calibration, bins 1-9
                        (fit_recovery_calibration.csv)
figS8_replication_leavek    replication check leave-k-out drift (rep_leavek.csv,
                        rep_bayes.csv)

CLI: ``python -m r2.paper_assets.figures_supp [--only figS1_ppc ...]`` (with
``src`` on ``sys.path``) or ``python src/r2/make_paper_assets.py figures``.
"""
from __future__ import annotations

import argparse
import math
import re
import sys
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from r2.paper_assets import style as S  # type: ignore
else:
    from . import style as S
from r2.common import PRESPECIFIED  # noqa: E402

K_PRED = int(PRESPECIFIED["topk_per_tf"])        # predicted set = top-k targets per TF

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import FixedLocator, LogLocator, NullFormatter  # noqa: E402

CM = S.CM
OI = S.OI
FS = S.FONT_SIZES

# ordinal ramps (one hue light -> dark) for ordered, non-entity categories
ORDINAL_BLUES = ["#9ecae1", "#56B4E9", "#2b8cbe", "#0072B2", "#08519c", "#08306b",
                 "#041f45", "#000000", "#000000", "#000000"]
K_COLOR = {10: "#56B4E9", 21: "#0072B2", 42: "#000000"}      # number of annotators (ordinal)
GEN_MARKER = {"template": "o", "independent": "s", "stress": "^"}
GEN_LABEL = {"template": "real-data template", "independent": "independent annotators",
             "stress": "stress (no pattern repeats)"}


# ----------------------------------------------------------------------------- helpers
def _panel(ax, letter: str, dx: float = -0.16, dy: float = 1.04) -> None:
    ax.text(dx, dy, letter, transform=ax.transAxes, fontsize=FS["panel"],
            fontweight="bold", va="bottom", ha="left")


def _refline(ax, y: float = 1.0, **kw) -> None:
    kw.setdefault("color", OI["grey"])
    kw.setdefault("lw", 0.6)
    kw.setdefault("ls", "--")
    kw.setdefault("zorder", 0)
    ax.axhline(y, **kw)


def _plain_log(ax, axis: str = "y", subs=(1.0, 2.0, 5.0)) -> None:
    """Log axis with plain decimal tick labels at 1-2-5 steps (no 10^k notation)."""
    from matplotlib.ticker import FuncFormatter

    def f(v, _):
        if v <= 0:
            return ""
        return f"{v:g}" if (1e-4 <= v < 1e5) else f"{v:.0e}"

    a = ax.yaxis if axis == "y" else ax.xaxis
    a.set_major_locator(LogLocator(base=10, subs=subs))
    a.set_major_formatter(FuncFormatter(f))
    a.set_minor_formatter(NullFormatter())


def _need(df: pd.DataFrame, what: str) -> pd.DataFrame:
    if df is None or len(df) == 0:
        raise ValueError(f"no rows for {what}")
    return df


def save_figure(fig, name: str, prov: S.Provenance, caption: str,
                out_dir: Path | None = None) -> dict:
    """Write PDF + 300-dpi PNG + caption text (provenance comment + LaTeX caption) + the LaTeX float wrapper
    ``<name>_float.tex``; return the paths."""
    out_dir = Path(out_dir) if out_dir is not None else S.FIG_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    ptxt = prov.text()
    pdf = out_dir / f"{name}.pdf"
    png = out_dir / f"{name}.png"
    fig.savefig(pdf, metadata={"Title": name, "Subject": ptxt.replace("\n", " | "),
                               "Creator": "src/r2/paper_assets/figures_supp.py"})
    fig.savefig(png, dpi=S.DPI_PNG, metadata={"Title": name, "Description": ptxt,
                                              "Software": "src/r2/paper_assets/figures_supp.py"})
    width_in = float(fig.get_size_inches()[0])
    cap = out_dir / f"{name}_caption.txt"
    cap.write_text(prov.tex_comment() + caption.strip() + "\n")
    plt.close(fig)
    flt = S.write_float(name, caption, prov, width_in, out_dir)
    return {"pdf": pdf, "png": png, "caption": cap, "float": flt}


def _fmt(x, d=2):
    return S.fmt_ratio(x, d)


# ----------------------------------------------------------------------------- figS1
PPC_STAT = "litblock_anyreport_frac_by_n_observing"
PPC_STRATA = ["1", "2", "3-4", "5-7", ">=8"]


def fig_ppc():
    prov = S.Provenance("figS1_ppc")
    d = prov.read("fit_ppc.csv",
                  f"statistic == '{PPC_STAT}', kind == 'statistic', grouping == 'modality', "
                  f"estimator in the five-model ladder, universe in (global, immune)")
    d = d[(d.statistic == PPC_STAT) & (d.kind == "statistic") & (d.grouping == "modality")
          & d.estimator.isin(S.ESTIMATOR_ORDER)].copy()
    _need(d, "PPC any-report rows")
    universes = [u for u in S.UNIVERSE_ORDER if u in set(d.universe)]
    n_rep = sorted(set(d.n_rep.astype(int)))
    fig, axes = plt.subplots(2, len(universes), figsize=(S.FULL_W, 8.6 * CM),
                             sharex=True, gridspec_kw={"height_ratios": [1.25, 1.0]})
    axes = np.atleast_2d(axes)
    x0 = np.arange(len(PPC_STRATA), dtype=float)
    ests = [e for e in S.ESTIMATOR_ORDER if e in set(d.estimator)]
    offs = np.linspace(-0.22, 0.22, len(ests))
    for j, u in enumerate(universes):
        du = d[d.universe == u]
        ax, axr = axes[0, j], axes[1, j]
        obs = du[du.estimator == ests[0]].set_index("stratum").reindex(PPC_STRATA)["observed"]
        ax.plot(x0, obs.values, color=OI["black"], lw=1.2, marker="o", ms=4,
                label="Observed", zorder=5)
        for e, o in zip(ests, offs):
            r = du[du.estimator == e].set_index("stratum").reindex(PPC_STRATA)
            lo = r.rep_mean - 1.96 * r.rep_sd
            hi = r.rep_mean + 1.96 * r.rep_sd
            ax.errorbar(x0 + o, r.rep_mean, yerr=[r.rep_mean - lo, hi - r.rep_mean], fmt=S.score_marker(e),
                        color=S.score_color(e), ms=3.2, lw=0.8, mfc="white" if e != "family_hb" else S.score_color(e),
                        mew=0.8, label=S.score_label(e), zorder=4)
            ratio = r.observed / r.rep_mean
            rlo = r.observed / hi.clip(lower=1e-12)
            rhi = r.observed / lo.clip(lower=1e-12)
            axr.errorbar(x0 + o, ratio, yerr=[ratio - rlo, rhi - ratio], fmt=S.score_marker(e),
                         color=S.score_color(e), ms=3.2, lw=0.8,
                         mfc="white" if e != "family_hb" else S.score_color(e), mew=0.8, zorder=4)
        ax.set_yscale("log")
        axr.set_yscale("log")
        _plain_log(ax, "y", subs=(1.0, 3.0))
        _plain_log(axr, "y", subs=(1.0, 2.0, 5.0))
        _refline(axr, 1.0)
        ax.set_title(S.UNIVERSE_LABEL[u] + " universe")
        axr.set_xticks(x0)
        axr.set_xticklabels([s.replace(">=", "$\\geq$") for s in PPC_STRATA])
        axr.set_xlabel("Literature annotators observing the edge")
        if j == 0:
            ax.set_ylabel("Edges with $\\geq$1 literature report\n(fraction of observed edges)")
            axr.set_ylabel("Observed / replicated")
        _panel(ax, "abcd"[j], dx=-0.2)
    h, lab = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, lab, loc="upper center", ncol=len(lab), bbox_to_anchor=(0.5, 1.0),
               columnspacing=1.0)
    fig.tight_layout(rect=(0, 0, 1, 0.93), h_pad=0.6, w_pad=1.2)
    lo_s, hi_s = PPC_STRATA[0], PPC_STRATA[-1]
    rat = {st: (d[d.stratum == st].observed / d[d.stratum == st].rep_mean) for st in (lo_s, hi_s)}
    under = bool((rat[hi_s] > 1).all())
    over = bool((rat[lo_s] < 1).all())
    claim = ""
    if under and over:
        claim = (" Every estimator, including Family-HB, under-predicts the reporting rate of edges observed by many "
                 "annotators and over-predicts it for edges observed by few (inference: observation is not ignorable).")
    caption = (
        "\\textbf{Posterior predictive check of the literature block.} For edges observed by a given number of "
        "literature annotators (x axis), the fraction with at least one literature report: observed "
        f"(black) and the mean over {', '.join(str(n) for n in n_rep)} data sets replicated from each "
        "fitted estimator on the real observation masks (coloured; bars: mean $\\pm 1.96$ replicate SD). "
        "Lower panels: observed divided by replicated (reference line at 1)." + claim
    )
    return fig, prov, caption


# ----------------------------------------------------------------------------- figS2
def fig_obs_model():
    prov = S.Provenance("figS2_obs_model")
    p = prov.read("obs_params.csv", "universe == 'global', optimum_role == 'best', "
                                    "fit_id in (family_hb_obs, em_obs), observation_modelled == True")
    p = p[(p.universe == "global") & (p.optimum_role == "best") & p.fit_id.isin(["family_hb_obs", "em_obs"])
          & (p.observation_modelled.map(S.as_bool))].copy()
    _need(p, "obs_params global best")
    fs = prov.read("obs_fit_summary.csv", "fit_id in (family_hb, em, family_hb_obs, em_obs); all universes; "
                                          "columns mean_cred_unobserved / _observed_neg / _reported")
    fs = fs[fs.fit_id.isin(["family_hb", "em", "family_hb_obs", "em_obs"])].copy()
    fig, axes = plt.subplots(1, 2, figsize=(S.FULL_W, 6.4 * CM),
                             gridspec_kw={"width_ratios": [1.0, 1.25]})
    ax = axes[0]
    for fid in ["family_hb_obs", "em_obs"]:
        q = p[p.fit_id == fid]
        xe = [q.eta_NonT - q.eta_NonT_lo, q.eta_NonT_hi - q.eta_NonT]
        ye = [q.eta_T - q.eta_T_lo, q.eta_T_hi - q.eta_T]
        ax.errorbar(q.eta_NonT, q.eta_T, xerr=xe, yerr=ye, fmt=S.score_marker(fid), ms=3.6,
                    color=S.score_color(fid), mfc=S.score_color(fid) if fid == "family_hb_obs" else "white",
                    mew=0.8, lw=0.6, ls="none", label=S.score_label(fid))
    lim = [min(p.eta_T.min(), p.eta_NonT.min()) * 0.5, 1.2]
    ax.plot(lim, lim, color=OI["grey"], lw=0.6, ls="--", zorder=0)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel(r"Observation rate if not true, $\eta_{j\bar{\mathrm{T}}}$")
    ax.set_ylabel(r"Observation rate if true, $\eta_{j\mathrm{T}}$")
    _plain_log(ax, "x", subs=(1.0,))
    _plain_log(ax, "y", subs=(1.0,))
    # legend above the axes: every region inside the axes holds annotators of one of the two models
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, borderaxespad=0.3,
              title="Literature annotators, global universe", title_fontsize=FS["small"] + 0.5)
    n_ann = p.groupby("fit_id").annotator.nunique().to_dict()
    _panel(ax, "a", dx=-0.25)

    ax = axes[1]
    status = [("mean_cred_unobserved", "Unobserved"), ("mean_cred_observed_neg", "Observed,\nnot reported"),
              ("mean_cred_reported", "Reported")]
    fits = ["em", "family_hb", "em_obs", "family_hb_obs"]
    offs = np.linspace(-0.27, 0.27, len(fits))
    for k, (col, _) in enumerate(status):
        for fid, o in zip(fits, offs):
            r = fs[fs.fit_id == fid]
            g = r[r.universe == "global"]
            t = r[r.universe != "global"]
            filled = fid in ("family_hb", "family_hb_obs")
            ax.plot(np.full(len(t), k + o), t[col], ls="none", marker=S.score_marker(fid), ms=2.6,
                    color=S.score_color(fid), mfc="none", mew=0.6, alpha=0.9, zorder=3)
            ax.plot([k + o], g[col], ls="none", marker=S.score_marker(fid), ms=4.6,
                    color=S.score_color(fid), mfc=S.score_color(fid) if filled else "white", mew=0.9,
                    zorder=4, label=S.score_label(fid) if k == 0 else None)
    ax.set_yscale("log")
    yv = fs[[c for c, _ in status]].to_numpy(dtype=float)
    yv = yv[np.isfinite(yv) & (yv > 0)]
    ax.set_ylim(yv.min() / 60.0, min(yv.max() * 3.0, 2.0))      # room for the legends below the data
    _plain_log(ax, "y", subs=(1.0,))
    ax.set_xticks(range(len(status)))
    ax.set_xticklabels([s for _, s in status])
    ax.set_xlim(-0.6, len(status) - 0.4)
    ax.set_ylabel("Mean credibility of the edges")
    ax.set_xlabel("Evidence status of the edge")
    ax.set_title("Credibility by evidence status")
    leg = ax.legend(loc="lower right", ncol=1)
    ax.add_artist(leg)
    ax.legend(handles=[Line2D([], [], ls="none", marker="o", ms=4.6, color=OI["black"], mfc=OI["black"],
                              label="global universe"),
                       Line2D([], [], ls="none", marker="o", ms=2.6, color=OI["black"], mfc="none",
                              label="tissue universes")], loc="lower left")
    _panel(ax, "b", dx=-0.17)
    fig.tight_layout(w_pad=1.6)
    nt = sorted(set(n_ann.values()))
    above = {fid: (int((p[p.fit_id == fid].eta_T > p[p.fit_id == fid].eta_NonT).sum()), int((p.fit_id == fid).sum()))
             for fid in ["family_hb_obs", "em_obs"]}
    maj = {fid: a / n > 0.5 for fid, (a, n) in above.items() if n}
    opp = len(maj) == 2 and maj["family_hb_obs"] != maj["em_obs"]
    g = fs[fs.universe == "global"].set_index("fit_id")
    order_txt = ""
    if {"family_hb_obs", "em_obs"} <= set(g.index):
        h, e = g.loc["family_hb_obs"], g.loc["em_obs"]
        if h.mean_cred_unobserved > max(h.mean_cred_observed_neg, h.mean_cred_reported) and \
                e.mean_cred_unobserved < e.mean_cred_observed_neg:
            order_txt = (" In the global universe Family-HB + observation gives unobserved edges the highest mean "
                         f"credibility ({S.fmt_sig(h.mean_cred_unobserved, 2)}); EM + observation ranks them below "
                         f"observed-but-unreported edges ({S.fmt_sig(e.mean_cred_unobserved, 2)} vs "
                         f"{S.fmt_sig(e.mean_cred_observed_neg, 2)}).")
    caption = (
        "\\textbf{Informative-observation extension} (secondary, post hoc; MAP fits with Laplace 95\\% intervals). "
        f"(a) Observation rates of the {', '.join(str(x) for x in nt)} literature annotators under the "
        "true class and under the other classes at the best optimum of Family-HB + observation and "
        "EM + observation (global universe; dashed: equal rates, i.e. ignorable observation)."
        + (f" The two models place most of their annotators on opposite sides of the diagonal (above it: "
           f"{above['family_hb_obs'][0]} of {above['family_hb_obs'][1]} for Family-HB + observation, "
           f"{above['em_obs'][0]} of {above['em_obs'][1]} for EM + observation): the label T is not anchored."
           if opp else "")
        + " (b) Mean credibility of unobserved, observed-but-unreported and reported edges for the "
        "ignorable models (EM, Family-HB) and the extensions (large markers: global universe; small "
        "open markers: immune, kidney and lung)." + order_txt
    )
    return fig, prov, caption


# ----------------------------------------------------------------------------- figS3
def fig_deconcentration():
    prov = S.Provenance("figS3_deconcentration")
    d = prov.read("gov_deconcentration.csv", "all rows (budget fraction x alpha grid per tissue); "
                                             "chosen = primary label-free rule; chosen_pielou = post hoc")
    tissues = [t for t in S.UNIVERSE_ORDER if t in set(d.tissue)]
    fig, axes = plt.subplots(1, len(tissues), figsize=(S.FULL_W, 6.0 * CM), sharey=True)
    fr = sorted(d.budget_fraction.unique())
    cmap = plt.get_cmap("Blues")
    cols = {f: cmap(0.3 + 0.7 * i / max(len(fr) - 1, 1)) for i, f in enumerate(fr)}
    for j, t in enumerate(tissues):
        ax = axes[j]
        dt = d[d.tissue == t]
        for f in fr:
            q = dt[dt.budget_fraction == f].sort_values("alpha")
            ax.plot(q.diversity, q.mean_cred, color=cols[f], lw=0.8, marker="o", ms=1.8, zorder=2)
        c = dt[dt.chosen.map(S.as_bool)]
        cp = dt[dt.chosen_pielou.map(S.as_bool)]
        a0 = dt[(dt.budget_fraction.isin(c.budget_fraction)) & (dt.alpha == 0)]
        ax.plot(a0.diversity, a0.mean_cred, ls="none", marker="s", ms=4.5, mfc="white", mec=OI["black"],
                mew=0.8, zorder=5, label="same budget, no penalty ($\\lambda$ = 0)")
        ax.plot(c.diversity, c.mean_cred, ls="none", marker="*", ms=9, color=OI["vermillion"], mec=OI["black"],
                mew=0.4, zorder=6, label="chosen point (label-free rule)")
        ax.plot(cp.diversity, cp.mean_cred, ls="none", marker="D", ms=4.2, mfc="none", mec=OI["vermillion"],
                mew=0.9, zorder=6, label="Pielou-evenness variant (post hoc)")
        pool = dt.pool_size.iloc[0]
        ntf = dt.n_tfs_pool.iloc[0]
        ax.set_title(f"{S.UNIVERSE_LABEL[t]} (pool: {S.fmt_count_plain(pool)} edges, "
                     f"{S.fmt_count_plain(ntf)} TFs)", fontsize=FS["title"] - 0.5)
        ax.set_xlabel("TF diversity, $H/\\ln K_\\mathrm{pool}$")
        if j == 0:
            ax.set_ylabel("Mean credibility of the subset")
        cc = c.iloc[0]
        ax.annotate(f"chosen: {S.fmt_count_plain(cc.budget)} edges,\n$\\lambda$ = {cc.alpha:g}",
                    (cc.diversity, cc.mean_cred), xytext=(0.05, 0.08), textcoords="axes fraction",
                    ha="left", va="bottom", fontsize=FS["small"] + 0.5,
                    arrowprops=dict(arrowstyle="-", lw=0.5, color=OI["grey"], shrinkB=5))
        _panel(ax, "abc"[j], dx=-0.12 if j else -0.24)
    cax = fig.add_axes([0.915, 0.2, 0.012, 0.58])
    sm = plt.cm.ScalarMappable(cmap=matplotlib.colors.ListedColormap([cols[f] for f in fr]),
                               norm=matplotlib.colors.BoundaryNorm(
                                   np.arange(len(fr) + 1) - 0.5, len(fr)))
    cb = fig.colorbar(sm, cax=cax, ticks=range(len(fr)))
    cb.ax.set_yticklabels([f"{100 * f:.0f}" for f in fr])
    cb.ax.tick_params(length=0)
    cb.set_label("Budget (% of promoted pool)")
    cb.outline.set_linewidth(0.4)
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="upper center", ncol=3, bbox_to_anchor=(0.47, 1.0))
    fig.subplots_adjust(left=0.08, right=0.895, bottom=0.17, top=0.8, wspace=0.12)
    alphas = sorted(d.alpha.unique())
    caption = (
        "\\textbf{Label-free deconcentration of the promoted set.} Each line is one budget (share of the promoted "
        "pool, colour) traced over the TF penalty $\\lambda \\in \\{"
        + ", ".join(f"{a:g}" for a in alphas)
        + "\\}$; x: TF diversity (Shannon entropy over TFs divided by the log number of TFs in the pool), "
        "y: mean credibility of the selected subset (min of global-fit and context-fit credibility). "
        "Star: operating point chosen by the label-free rule of Section~\\ref{sec:governance} (harmonic mean of "
        "normalised credibility and diversity; fixed after held-out results of other analyses existed, but no "
        "held-out data enter the choice); open square: the same budget without penalty; open diamond: the "
        "post hoc variant with Pielou evenness."
    )
    return fig, prov, caption


# ----------------------------------------------------------------------------- figS4
def fig_gov_grid():
    prov = S.Provenance("figS4_gov_grid")
    d = prov.read("gov_sensitivity_grid.csv",
                  "family == 'q_grid' (dual_rule primary) and family == 'dual_rule' (primary, lenient, strict); "
                  "reference in (encode, enets2, pert_pooled); columns n_promoted, n_strong, oe_tf, oe_tf_lo, oe_tf_hi")
    tissues = [t for t in S.UNIVERSE_ORDER if t in set(d.tissue)]
    g = d[d.family == "q_grid"].copy()
    qs = sorted(g.q_strong.unique())
    qm = sorted(g.q_moderate.unique())
    refs = [r for r in ["encode", "enets2", "pert_pooled"] if r in set(d.reference)]
    fig = plt.figure(figsize=(S.FULL_W, 10.6 * CM))
    gs = fig.add_gridspec(2, len(tissues), height_ratios=[1.0, 1.15], hspace=0.55, wspace=0.32)
    cmap = plt.get_cmap("Blues")
    vmax = g.n_promoted.max()
    vmin = g.n_promoted.min()
    plotted = d[d.family.isin(["q_grid", "dual_rule"]) & d.reference.isin(refs)]
    vals = pd.concat([plotted.oe_tf, plotted.oe_tf_lo, plotted.oe_tf_hi])
    floor = float(vals[vals > 0].min()) / 1.4
    ceil_ = float(vals.max()) * 1.3

    def _eb(ax2, xpos, r, mk, col, few):
        lo = max(r.oe_tf_lo, floor)
        ax2.errorbar(xpos, r.oe_tf, yerr=[[r.oe_tf - lo], [r.oe_tf_hi - r.oe_tf]], fmt=mk, color=col, ms=3.4,
                     lw=0.8, mfc="white" if few else col, mew=0.8)
        if r.oe_tf_lo < floor:
            ax2.plot(xpos, floor, marker="v", ms=3.0, color=col, clip_on=False, zorder=5)
    for j, t in enumerate(tissues):
        ax = fig.add_subplot(gs[0, j])
        gt = g[(g.tissue == t)].drop_duplicates(["q_strong", "q_moderate"]).set_index(["q_strong", "q_moderate"])
        M = np.full((len(qs), len(qm)), np.nan)
        for a, s in enumerate(qs):
            for b, m in enumerate(qm):
                if (s, m) in gt.index:
                    M[a, b] = gt.loc[(s, m), "n_promoted"]
        ax.imshow(M, cmap=cmap, vmin=vmin * 0.6, vmax=vmax, aspect="auto", origin="lower")
        prim = d[(d.tissue == t) & (d.family == "dual_rule") & (d.config == "primary")].iloc[0]
        for a, s in enumerate(qs):
            for b, m in enumerate(qm):
                if (s, m) not in gt.index:
                    ax.text(b, a, "n/a", ha="center", va="center", fontsize=FS["small"] + 0.5, color=OI["grey"])
                    continue
                r = gt.loc[(s, m)]
                dark = (r.n_promoted - vmin * 0.6) / (vmax - vmin * 0.6) > 0.55
                ax.text(b, a + 0.12, S.fmt_count_plain(r.n_promoted), ha="center", va="center",
                        fontsize=FS["small"] + 0.5, color="white" if dark else OI["black"])
                ax.text(b, a - 0.24, f"S {S.fmt_count_plain(r.n_strong)}", ha="center", va="center",
                        fontsize=FS["small"] + 0.5, color="white" if dark else OI["black"])
                if np.isclose(s, prim.q_strong) and np.isclose(m, prim.q_moderate):
                    ax.add_patch(plt.Rectangle((b - 0.5, a - 0.5), 1, 1, fill=False, ec=OI["vermillion"], lw=1.4))
        ax.set_xticks(range(len(qm)))
        ax.set_xticklabels([f"{m:g}" for m in qm])
        ax.set_yticks(range(len(qs)))
        ax.set_yticklabels([f"{s:g}" for s in qs])
        ax.set_xlabel("$q_\\mathrm{moderate}$ (BFDR threshold)")
        if j == 0:
            ax.set_ylabel("$q_\\mathrm{strong}$ (BFDR threshold)")
        ax.set_title(S.UNIVERSE_LABEL[t], fontsize=FS["title"])
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.tick_params(length=0)
        _panel(ax, "abc"[j], dx=-0.32, dy=1.02)

        # lower row: O/E (TF-matched) vs q_moderate, plus +-1-tier rules at the primary q
        ax2 = fig.add_subplot(gs[1, j])
        xs = list(range(len(qm)))
        rules = ["strict", "lenient"]
        xr = [len(qm) + 0.6 + i for i in range(len(rules))]
        offs = np.linspace(-0.22, 0.22, len(refs))
        for rf, o in zip(refs, offs):
            q = g[(g.tissue == t) & (g.reference == rf) & np.isclose(g.q_strong, prim.q_strong)].set_index("q_moderate").reindex(qm)
            col = S.REFERENCE_COLOR.get(rf, OI["black"])
            mk = S.REFERENCE_MARKER.get(rf, "o")
            few = q.few_clusters.map(S.as_bool).values
            for xi, (_, r), fw in zip(xs, q.iterrows(), few):
                _eb(ax2, xi + o, r, mk, col, fw)
            for xi, rule in zip(xr, rules):
                rr = d[(d.tissue == t) & (d.family == "dual_rule") & (d.config == rule) & (d.reference == rf)]
                if len(rr):
                    r = rr.iloc[0]
                    _eb(ax2, xi + o, r, mk, col, S.as_bool(r.few_clusters))
        _refline(ax2, 1.0)
        ax2.axvline(len(qm) - 0.2 + 0.0, color=OI["grey"], lw=0.4, ls=":")
        ax2.set_yscale("log")
        ax2.set_ylim(floor, ceil_)
        _plain_log(ax2, "y", subs=(1.0, 2.0, 5.0))
        ax2.set_xticks(xs + xr)
        ax2.set_xticklabels([f"{m:g}" for m in qm] + [r for r in rules], fontsize=FS["small"] + 0.5)
        for tl in ax2.get_xticklabels()[len(qm):]:
            tl.set_rotation(35)
            tl.set_ha("right")
            tl.set_rotation_mode("anchor")
        ax2.set_xlabel("$q_\\mathrm{moderate}$ | rule shifted by one tier")
        if j == 0:
            ax2.set_ylabel("Promoted O/E, held out\n(TF-matched)")
        _panel(ax2, "def"[j], dx=-0.32, dy=1.02)
    hand = [Line2D([], [], ls="none", marker=S.REFERENCE_MARKER.get(r, "o"), color=S.REFERENCE_COLOR.get(r),
                   ms=3.6, label=S.reference_label(r)) for r in refs]
    hand.append(Line2D([], [], ls="none", marker="o", color=OI["black"], mfc="white", ms=3.6,
                       label="fewer than ten evaluable TFs"))
    fig.legend(handles=hand, loc="lower center", ncol=len(hand), bbox_to_anchor=(0.5, 0.0))
    fig.subplots_adjust(left=0.1, right=0.99, top=0.94, bottom=0.19)
    # q_strong never changes the promoted set under the primary rule (verified from the CSV)
    inv = (g.groupby(["tissue", "q_moderate", "reference"]).oe_tf.agg(lambda s: s.max() - s.min()).max())
    caption = (
        "\\textbf{Sensitivity of claim governance to the BFDR thresholds} (dual-universe rule). (a--c) Number of "
        "promoted edges (Strong $\\cup$ Moderate; colour and upper number) and of Strong edges (S) over the "
        "grid of $q_\\mathrm{strong} \\times q_\\mathrm{moderate}$; the prespecified point is outlined. "
        "Under the primary rule $q_\\mathrm{strong}$ only moves edges between Strong and Moderate, so "
        f"the promoted set and its enrichment do not depend on it (largest change of O/E across "
        f"$q_\\mathrm{{strong}}$: {_fmt(inv, 2)}). (d--f) Observed over TF-matched expected held-out "
        "positives in the promoted set, by $q_\\mathrm{moderate}$ at the prespecified $q_\\mathrm{strong}$, "
        "and for the rules shifted by one tier (strict, lenient) at the prespecified thresholds; bars: "
        "95\\% TF-cluster bootstrap intervals (triangles: lower limit below the axis range); open markers: "
        "intervals resting on fewer than ten evaluable TFs; reference line at 1."
    )
    return fig, prov, caption


# ----------------------------------------------------------------------------- figS5
def fig_compute_scaling():
    prov = S.Provenance("figS5_compute_scaling")
    d = prov.read("scale_summary.csv", "all 26 configurations; status columns map_10_status / nuts_status "
                                       "(measured = filled, extrapolated/projected = open)")
    sm = prov.read("scale_map.csv", "n_restarts_run of the measured MAP records (restart count shown in panel e)")
    sn = prov.read("scale_nuts.csv", "chains / warmup / samples of the measured NUTS records (panel f); projected "
                                     "configurations: min t_nuts_projected_s and max t_nuts_projected_cap_s over the "
                                     "measurement sessions (the range of Table tab:compute-scaling)")
    n_restarts = int(sm.loc[sm.t_map_10_status == "measured", "n_restarts_run"].mode().iloc[0])
    sn_m = sn[sn.status == "measured"]
    n_ch, n_wu, n_sa = (int(sn_m[c].mode().iloc[0]) for c in ("chains", "warmup", "samples"))
    fig, axes = plt.subplots(2, 3, figsize=(S.FULL_W, 9.8 * CM))
    axes = axes.ravel()
    real = d[(d.generator == "template") & (d["size"] == "real")]
    n_real = float(real.N.iloc[0]) if len(real) else None

    def scatter(ax, x, y, status=None, sub=None, ylab="", xlab="", title=""):
        q = d if sub is None else sub
        for (gen, K), r in q.groupby(["generator", "K"]):
            r = r.sort_values(x)
            ok = r[y].notna()
            r = r[ok]
            if not len(r):
                continue
            meas = np.ones(len(r), bool) if status is None else (r[status] == "measured").values
            ax.plot(r[x], r[y], color=K_COLOR.get(int(K), OI["grey"]), lw=0.6,
                    ls="-" if gen == "template" else (":" if gen == "independent" else "--"), zorder=2)
            ax.plot(r[x][meas], r[y][meas], ls="none", marker=GEN_MARKER[gen], ms=3.4,
                    color=K_COLOR.get(int(K)), zorder=3)
            ax.plot(r[x][~meas], r[y][~meas], ls="none", marker=GEN_MARKER[gen], ms=3.4, mfc="white",
                    color=K_COLOR.get(int(K)), mew=0.8, zorder=3)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(xlab)
        ax.set_ylabel(ylab)
        ax.set_title(title, fontsize=FS["title"] - 0.5)
        if x == "N" and n_real:
            ax.axvline(n_real, color=OI["grey"], lw=0.5, ls="--", zorder=0)

    scatter(axes[0], "N", "P", ylab="Distinct observation patterns $P$", xlab="Edges $N$",
            title="Pattern compression")
    axes[0].plot([d.N.min(), d.N.max()], [d.N.min(), d.N.max()], color=OI["grey"], lw=0.5, ls=":", zorder=0)
    axes[0].text(d.N.max() * 0.5, d.N.max() * 0.08, "$P=N$", fontsize=FS["small"] + 0.5, color=OI["grey"],
                 ha="right")
    scatter(axes[1], "N", "compress_s", ylab="Wall time (s)", xlab="Edges $N$", title="Compression time")
    scatter(axes[2], "N", "compress_peak_rss_mib", ylab="Peak memory (MiB)", xlab="Edges $N$",
            title="Compression memory")
    scatter(axes[3], "P", "t_vg_s", ylab="Wall time (s)", xlab="Patterns $P$",
            title="One log-posterior + gradient")
    scatter(axes[4], "P", "map_10_s", status="map_10_status", ylab="Wall time (s)", xlab="Patterns $P$",
            title=f"MAP, {n_restarts} random restarts")
    nuts = d[d.nuts_2x300_s.notna()].copy()
    # projected configurations: the step-count projection (lowest over sessions) to the tree-depth-cap projection
    # (highest over sessions), as printed in tab:compute-scaling (verifier assets_fix M3)
    proj = sn[sn.status == "projected"].groupby("cid").agg(lo=("t_nuts_projected_s", "min"),
                                                          hi=("t_nuts_projected_cap_s", "max"))
    pm = (nuts.nuts_status == "projected") & nuts.cid.isin(proj.index)
    nuts.loc[pm, "nuts_2x300_s"] = nuts.loc[pm, "cid"].map(proj["lo"]).astype(float)
    nuts.loc[pm, "nuts_2x300_cap_s"] = nuts.loc[pm, "cid"].map(proj["hi"]).astype(float)
    scatter(axes[5], "P", "nuts_2x300_s", status="nuts_status", sub=nuts, ylab="Wall time (s)",
            xlab="Patterns $P$", title=f"NUTS, {n_ch} chains $\\times$ {n_wu}/{n_sa}")
    cap = nuts[nuts.nuts_2x300_cap_s.notna()]
    for _, r in cap.iterrows():
        axes[5].plot([r.P, r.P], [r.nuts_2x300_s, r.nuts_2x300_cap_s], color=K_COLOR.get(int(r.K)), lw=0.8,
                     zorder=1)
    for ax, l in zip(axes, "abcdef"):
        _panel(ax, l, dx=-0.3)
        ax.yaxis.set_minor_formatter(NullFormatter())
    _plain_log(axes[2], "y", subs=(1.0, 2.0, 5.0))
    hand = [Line2D([], [], color=K_COLOR[k], lw=1.0, label=f"K = {k} annotators") for k in sorted(K_COLOR)
            if k in set(d.K.astype(int))]
    gls = {"template": "-", "independent": ":", "stress": "--"}
    hand += [Line2D([], [], ls=gls[g], lw=0.6, marker=GEN_MARKER[g], color=OI["black"], ms=3.4, label=GEN_LABEL[g])
             for g in ["template", "independent", "stress"] if g in set(d.generator)]
    hand += [Line2D([], [], ls="none", marker="o", color=OI["black"], mfc="white", ms=3.4,
                    label="projected / extrapolated")]
    fig.legend(handles=hand, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.9), h_pad=0.8, w_pad=0.8)
    hw = prov.read("scale_environment.csv", "key in (cpu, hw.ncpu, hw.memsize (bytes))").set_index("key")["value"]
    machine = (f"{hw.get('cpu', '?')}, {hw.get('hw.ncpu', '?')} cores, "
               f"{int(float(hw.get('hw.memsize (bytes)', 'nan')) / 2 ** 30) if 'hw.memsize (bytes)' in hw else '?'} GiB")
    caption = (
        "\\textbf{Computational scaling on synthetic evidence} (no held-out data). (a) Number of distinct "
        "observation patterns $P$ against the number of edges $N$ for evidence generated from the real "
        "observation design (template), from independent annotators and without repeated patterns "
        "(stress); dashed vertical line: size of the global universe. (b, c) Wall time and peak memory "
        "of the streaming pattern compression. (d) One evaluation of the collapsed log posterior and its "
        f"gradient. (e) MAP with {n_restarts} random restarts (open: extrapolated from the restarts that finished, or "
        "projected from evaluation counts, a lower bound). (f) NUTS with "
        f"{n_ch} chains of {n_wu} warm-up and {n_sa} draws from random starts, a cost benchmark rather than a converged "
        "fit; open markers are projections from the steps per iteration of the largest measured run (lowest over the "
        "two measurement sessions), which are lower bounds because the steps per iteration grow with $N$, and the "
        "vertical bars reach the projection at the tree-depth cap (highest over the sessions), the better estimate at "
        "ten times the real size; the same ranges as in Table~\\ref{tab:compute-scaling}. Measured times are "
        "wall-clock on a shared "
        f"laptop ({machine}) and are upper bounds of the undisturbed cost."
    )
    return fig, prov, caption


# ----------------------------------------------------------------------------- figS6
def fig_fm_attention_null():
    prov = S.Provenance("figS6_fm_attention_null")
    a = prov.read("fm_attention_null.csv", f"variant in (attn, attn_T, attn_norm); k == {K_PRED}")
    a = a[a.variant.isin(["attn", "attn_T", "attn_norm"]) & (a.k == K_PRED)].copy()
    h = prov.read("heldout_methods_context.csv",
                  f"eval_subset == 'all_evaluable', topk_per_tf == {K_PRED}, score in (scgpt_attn, geneformer_attn), "
                  "reference in (encode, enets2, pert_pooled); columns enr_ratio_to_null(_lo,_hi), matched_null")
    h = h[(h.eval_subset == "all_evaluable") & (h.topk_per_tf == K_PRED)
          & h.score.isin(["scgpt_attn", "geneformer_attn"]) & h.reference.isin(["encode", "enets2", "pert_pooled"])].copy()
    _need(h, "methods_context attention rows")
    tissues = [t for t in S.UNIVERSE_ORDER if t in set(a.tissue)]
    fig, axes = plt.subplots(1, 2, figsize=(S.FULL_W, 6.2 * CM), gridspec_kw={"width_ratios": [1.0, 1.35]})
    ax = axes[0]
    models = [("scgpt", "scgpt_attn"), ("geneformer", "geneformer_attn")]
    variants = [("attn", "raw"), ("attn_T", "transposed"), ("attn_norm", "normalised")]
    xt, xl = [], []
    x = 0.0
    for t in tissues:
        for m, key in models:
            for v, vl in variants:
                r = a[(a.tissue == t) & (a.model == m) & (a.variant == v)]
                if not len(r):
                    continue
                r = r.iloc[0]
                hatch = {"attn": None, "attn_T": "////", "attn_norm": "...."}[v]
                ax.bar(x, r.topk_overlap_with_uniform_median, width=0.8, color=S.score_color(key) if v == "attn" else "white",
                       edgecolor=S.score_color(key), hatch=hatch, lw=0.7)
                ax.plot([x - 0.4, x + 0.4], [r.topk_overlap_random_expectation] * 2, color=OI["black"], lw=0.9)
                x += 1
            x += 0.4
        xt.append(x - 0.4 - len(models) * len(variants) / 2 - 0.2)
        xl.append(S.UNIVERSE_LABEL[t])
        x += 0.8
    ax.set_xticks(xt)
    ax.set_xticklabels(xl)
    ax.set_ylabel(f"Top-{K_PRED} overlap with the 1/L null\n(median over TFs)")
    ax.set_ylim(0, 1)
    ax.set_title("Attention ranking vs uniform-attention null")
    hand = [matplotlib.patches.Patch(facecolor=S.score_color("scgpt_attn"), edgecolor=S.score_color("scgpt_attn"), label="scGPT"),
            matplotlib.patches.Patch(facecolor=S.score_color("geneformer_attn"), edgecolor=S.score_color("geneformer_attn"), label="Geneformer"),
            matplotlib.patches.Patch(facecolor="white", edgecolor=OI["black"], hatch="////", label="transposed"),
            matplotlib.patches.Patch(facecolor="white", edgecolor=OI["black"], hatch="....", label="normalised"),
            Line2D([], [], color=OI["black"], lw=0.9, label="random expectation")]
    ax.legend(handles=hand, loc="upper right", ncol=2)
    _panel(ax, "a", dx=-0.22)

    ax = axes[1]
    refs = ["encode", "enets2", "pert_pooled"]
    xt, xl = [], []
    x = 0.0
    pos = pd.concat([h.enr_ratio_to_null, h.enr_ratio_to_null_lo[h.enr_ratio_to_null_lo > 0]])
    floor = float(pos[pos > 0].min()) / 1.8      # axis floor below the smallest positive value / bound
    for t in tissues:
        start = x
        for rf in refs:
            for key, o in zip(["scgpt_attn", "geneformer_attn"], [-0.17, 0.17]):
                r = h[(h.universe == t) & (h.reference == rf) & (h.score == key)]
                if not len(r):
                    continue
                r = r.iloc[0]
                low = S.as_bool(r.low_support)
                lo = max(r.enr_ratio_to_null_lo, floor)
                ax.errorbar(x + o, r.enr_ratio_to_null, yerr=[[r.enr_ratio_to_null - lo],
                                                              [r.enr_ratio_to_null_hi - r.enr_ratio_to_null]],
                            fmt=S.REFERENCE_MARKER.get(rf, "o"), color=S.score_color(key), ms=3.6, lw=0.8,
                            mfc="white" if low else S.score_color(key), mew=0.8)
                if r.enr_ratio_to_null_lo < floor:
                    ax.plot(x + o, floor, marker="v", ms=3.0, color=S.score_color(key), clip_on=False, zorder=5)
            x += 1
        xt.append((start + x - 1) / 2)
        xl.append(S.UNIVERSE_LABEL[t])
        ax.axvline(x - 0.5 + 0.2, color=OI["grey"], lw=0.3, ls=":") if t != tissues[-1] else None
        x += 0.4
    _refline(ax, 1.0)
    ax.set_yscale("log")
    ax.set_ylim(bottom=floor)
    _plain_log(ax, "y", subs=(1.0, 2.0, 5.0))
    ax.set_xticks(xt)
    ax.set_xticklabels(xl)
    ax.set_ylabel(f"Enrichment of top-{K_PRED} attention\n/ enrichment of its 1/L null")
    ax.set_title("Held-out enrichment relative to the matched null")
    hand = [Line2D([], [], ls="none", marker=S.REFERENCE_MARKER.get(r, "o"), color=OI["black"], ms=3.6,
                   label=S.reference_label(r)) for r in refs]
    hand += [Line2D([], [], ls="-", lw=2.0, color=S.score_color(k), label=S.score_label(k))
             for k in ["scgpt_attn", "geneformer_attn"]]
    hand += [Line2D([], [], ls="none", marker="o", color=OI["black"], mfc="white", ms=3.6, label="low support")]
    ax.legend(handles=hand, loc="upper left", ncol=2)
    _panel(ax, "b", dx=-0.16)
    ax.set_ylim(top=ax.get_ylim()[1] * 2.5)
    fig.tight_layout(w_pad=1.5)
    k = int(a.k.iloc[0])
    raw = a[(a.variant == "attn")]
    rs, rg = raw[raw.model == "scgpt"], raw[raw.model == "geneformer"]
    ov = ""
    if len(rs):
        ov = (f" The {k} top targets of raw scGPT attention overlap with those of its null by "
              f"{S.fmt_ratio(rs.topk_overlap_with_uniform_median.min())}--{S.fmt_ratio(rs.topk_overlap_with_uniform_median.max())} "
              f"(random ranking {S.fmt_ratio(rs.topk_overlap_random_expectation.min())}--"
              f"{S.fmt_ratio(rs.topk_overlap_random_expectation.max())})"
              + (f", those of raw Geneformer attention by {S.fmt_ratio(rg.topk_overlap_with_uniform_median.min())}--"
                 f"{S.fmt_ratio(rg.topk_overlap_with_uniform_median.max())}" if len(rg) else "") + ".")
    caption = (
        "\\textbf{Foundation-model attention against its uniform-attention null.} (a) Median over TFs of the "
        f"overlap between the {k} top-ranked targets by attention and by the $1/L$ null (the attention an "
        "edge would receive if each cell spread attention uniformly over its $L$ input genes); filled: raw "
        "attention, hatched: transposed and normalised variants; black ticks: overlap expected for a random "
        "ranking." + ov + " (b) TF-matched held-out "
        f"enrichment of each model's top-{k}-per-TF predicted set divided by that of its own $1/L$ null, with "
        "paired 95\\% TF-cluster bootstrap intervals (reference line at 1; triangles mark lower limits "
        "below the axis range; open markers: references flagged as low support)."
    )
    return fig, prov, caption


# ----------------------------------------------------------------------------- figS7
OVERCONF_TOL = 0.01     # figS7: a bin counts as over-confident if the true-edge fraction is this much below the mean


def fig_recovery_calibration():
    prov = S.Provenance("figS7_recovery_calibration")
    d = prov.read("fit_recovery_calibration.csv", "in_ece_bins1_9 == True (bins 1-9); all estimators; "
                  "simulator = faithful_cc (design-faithful)")
    d = d[d.in_ece_bins1_9.map(S.as_bool)].copy()
    if "simulator" in d:
        d = d[d.simulator == "faithful_cc"].copy()
    universes = [u for u in ["immune", "global"] if u in set(d.universe)]
    sty = {"oracle": ("oracle", "-", True), "family_hb_truthstart": ("family_hb", "--", False),
           "family_hb": ("family_hb", "-", True), "em": ("em", "-", False)}
    lab = {"oracle": S.score_label("oracle"), "family_hb": "Family-HB (MAP)",
           "family_hb_truthstart": "Family-HB (started at the truth)", "em": S.score_label("em")}
    fig, axes = plt.subplots(1, len(universes), figsize=(S.FULL_W * 0.75, 6.4 * CM), sharey=True)
    axes = np.atleast_1d(axes)
    ests = [e for e in ["oracle", "family_hb_truthstart", "family_hb", "em"] if e in set(d.estimator)]
    offs = dict(zip(ests, np.linspace(-0.008, 0.008, len(ests))))
    for j, u in enumerate(universes):
        ax = axes[j]
        ax.plot([0.1, 1], [0.1, 1], color=OI["grey"], lw=0.6, ls="--", zorder=0)
        for e in ests:
            r = d[(d.universe == u) & (d.estimator == e)].sort_values("bin")
            key, ls, filled = sty[e]
            col = S.score_color(key)
            ax.errorbar(r.mean_predicted + offs[e], r.empirical_true_rate, yerr=r.true_rate_sd_across_replicates,
                        fmt=S.score_marker(key) if e != "family_hb_truthstart" else "s", ls=ls, color=col, ms=3.2,
                        lw=0.8, mfc=col if filled else "white", mew=0.8, label=lab[e])
        nrep = int(d[d.universe == u].n_replicates.max())
        ax.set_title(f"{S.UNIVERSE_LABEL[u]} design, {nrep} simulated data sets", fontsize=FS["title"] - 0.5)
        ax.set_xlim(0.08, 1.02)
        ax.set_ylim(-0.02, 1.08)
        ax.set_xlabel("Mean credibility in the bin")
        if j == 0:
            ax.set_ylabel("Fraction of true edges in the bin")
        _panel(ax, "ab"[j], dx=-0.18)
    axes[0].legend(loc="upper left")
    fig.tight_layout(w_pad=1.0)
    blo, bhi = d.bin_lo.min(), d.bin_hi.max()
    rc = prov.read("fit_recovery.csv", "simulator = faithful_cc: analysis_status of the truth-start estimator and of "
                                       "quantity ece_bins1_9 (post hoc flags)")
    rc = rc[rc.simulator == "faithful_cc"] if "simulator" in rc else rc
    ph_ts = bool((rc[rc.estimator == "family_hb_truthstart"].analysis_status == "post_hoc").any())
    ph_bins = bool((rc[rc.quantity.astype(str).str.startswith("ece_bins1_9")].analysis_status == "post_hoc").any())
    em = d[d.estimator == "em"]
    n_over = int((em.empirical_true_rate < em.mean_predicted - OVERCONF_TOL).sum())
    em_txt = (f"the EM baseline is over-confident (fraction of true edges more than {OVERCONF_TOL:g} below the mean "
              f"credibility) in {n_over} of its {len(em)} universe $\\times$ bin cells" if len(em) else "")
    ph_sim = bool(len(d)) and "analysis_status" in d and bool((d.analysis_status == "post_hoc").all())
    fdz = prov.read("modes_faithful_design.csv", "quantity = n_edges_cc_observed, sim_label in {immune_cc, global_cc}: "
                                                 "ratio_sim_mean_to_real")
    fdz = fdz[(fdz.quantity == "n_edges_cc_observed") & fdz.sim_label.isin(["immune_cc", "global_cc"])]
    shr = {str(r.universe): float(r.ratio_sim_mean_to_real) for r in fdz.itertuples()}
    shr_txt = ("; the simulated data contain " + " and ".join(
        f"{S.fmt_pct(shr[u])}\\% ({S.UNIVERSE_LABEL.get(u, u).lower()})" for u in ("immune", "global") if u in shr)
        + " of the real number of motif- or coexpression-observed edges") if shr else ""
    comp = rc[(rc.kind == "identifiability") & (rc.quantity == "n_replicates_map_in_competing_mode")]
    comp_txt = ", ".join(f"{S.UNIVERSE_LABEL.get(r.universe, r.universe).lower()} {int(r.value_mean)} of "
                         f"{int(r.n_replicates)}" for r in comp.itertuples())
    ph = [] if ph_sim else [x for x, f in (("the truth-start fit", ph_ts), ("the calibration over the bins shown", ph_bins))
                            if f]
    caption = (
        "\\textbf{Calibration of credibility in simulation.} Data simulated from the fitted Family-HB model with the "
        "real literature observation masks and the motif/coexpression masks rebuilt from the simulated literature "
        f"reports (design-faithful simulator{shr_txt}). Edges are binned by credibility (bins from {blo:g} to {bhi:g}; the bin "
        f"below {blo:g}, which holds almost all edges, is omitted); points: pooled fraction of truly regulatory "
        "edges against the mean credibility in the bin; bars: $\\pm 1$ SD of that fraction across simulated "
        "data sets; dashed line: perfect calibration. Started at the true parameters, Family-HB is close "
        "to the oracle posterior; multi-start MAP selects the competing mode in some data sets"
        + (f" ({comp_txt})" if comp_txt else "") + ", which pulls the upper bins down and widens their spread"
        + (f"; {em_txt}" if em_txt else "") + "."
        + (" Post hoc: " + " and ".join(ph) + "." if ph else "")
        + (" The design-faithful simulation is post hoc (defined after the held-out data had been examined; it uses no held-out "
           "data)." if ph_sim else "")
    )
    return fig, prov, caption


# ----------------------------------------------------------------------------- figS8
def _unit_short(u: str) -> str:
    """Display name of a replication unit without the publication year (e.g. 'Norman 2019' -> 'Norman')."""
    lab = S.reference_label(u)
    return re.sub(r"\s+(19|20)\d\d$", "", lab)


def fig_replication_leavek():
    prov = S.Provenance("figS8_replication_leavek")
    lk = prov.read("rep_leavek.csv", "row_type == 'fit'; levels study and cohort; columns k, left_out, units_kept, "
                                     "mu_mean, mu_q025, mu_q975, degenerate_single_unit")
    lk = lk[lk.row_type == "fit"].copy()
    rb = prov.read("rep_bayes.csv", "profile == 'primary', parameter == 'mu' (full fit with all units)")
    rb = rb[(rb.profile == "primary") & (rb.parameter == "mu")]
    levels = [lv for lv in ["study", "cohort"] if lv in set(lk.level)]
    nrows = {lv: len(lk[lk.level == lv]) + 1 for lv in levels}
    fig, axes = plt.subplots(1, len(levels), figsize=(S.FULL_W, max(nrows.values()) * 0.36 * CM + 2.6 * CM))
    axes = np.atleast_1d(axes)
    allx = np.exp(pd.concat([lk.mu_q025, lk.mu_q975, rb.q025, rb.q975]))
    xlim = (allx.min() / 1.25, allx.max() * 1.25)
    kcol = {0: OI["black"], 1: OI["blue"], 2: OI["skyblue"], 3: OI["grey"]}
    for j, lv in enumerate(levels):
        ax = axes[j]
        q = lk[lk.level == lv].sort_values(["k", "left_out"])
        full = rb[rb.level == lv].iloc[0]
        rows = [("all units", full["mean"], full.q025, full.q975, 0, False)]
        for _, r in q.iterrows():
            deg = S.as_bool(r.degenerate_single_unit)
            if deg and isinstance(r.units_kept, str):
                lab = "only " + _unit_short(r.units_kept)
            else:
                lab = "without " + ", ".join(_unit_short(u) for u in str(r.left_out).split(";"))
            rows.append((lab, r.mu_mean, r.mu_q025, r.mu_q975, int(r.k), deg))
        y = np.arange(len(rows))[::-1].astype(float)
        for yi, (lab, m, lo, hi, k, deg) in zip(y, rows):
            col = kcol.get(k, OI["grey"])
            ax.errorbar(np.exp(m), yi, xerr=[[np.exp(m) - np.exp(lo)], [np.exp(hi) - np.exp(m)]], fmt="o",
                        color=col, ms=3.4, lw=0.8, mfc="white" if deg else col, mew=0.8)
        ax.axvline(1.0, color=OI["grey"], lw=0.6, ls="--", zorder=0)
        ax.axvline(np.exp(full["mean"]), color=OI["black"], lw=0.4, ls=":", zorder=0)
        ax.set_yticks(y)
        ax.set_yticklabels([r[0] for r in rows])
        ax.set_ylim(-0.7, len(rows) - 0.3)
        ax.set_xscale("log")
        ax.set_xlim(*xlim)
        _plain_log(ax, "x", subs=(1.0, 2.0, 5.0))
        ax.set_xlabel("Pooled within-TF odds ratio,\nexp(posterior mean of $\\mu$)")
        ax.set_title(f"Units = {lv.replace('study', 'studies').replace('cohort', 'cohorts')}",
                     fontsize=FS["title"])
        ax.text(-0.02, 1.0, "ab"[j], transform=ax.transAxes, fontsize=FS["panel"], fontweight="bold",
                ha="right", va="bottom")
    hand = [Line2D([], [], ls="none", marker="o", color=c, ms=3.4, label=l) for c, l in
            [(OI["black"], "full fit"), (OI["blue"], "L1O"), (OI["skyblue"], "L2O"), (OI["grey"], "L3O")]]
    hand.append(Line2D([], [], ls="none", marker="o", color=OI["black"], mfc="white", ms=3.4,
                       label="single unit kept (not pooled)"))
    fig.legend(handles=hand, loc="lower center", ncol=len(hand), bbox_to_anchor=(0.5, 0.0))
    fig.subplots_adjust(left=0.2, right=0.98, top=0.93, bottom=0.17, wspace=0.95)
    prof = rb.iloc[0]
    from .figures import _prior_text
    caption = (
        "\\textbf{Leave-$k$-out drift of the pooled replication effect.} Points: $\\exp$ of the posterior mean of "
        "the pooled within-TF log odds ratio $\\mu$ of the Bayesian random-effects model, with the exponentiated 95\\% "
        f"credible limits (prior $\\mu\\sim {_prior_text(prof.prior_mu)}$, $\\sigma\\sim {_prior_text(prof.prior_sigma)}$), "
        "for the full data and refitted after removing every subset of one (L1O), two (L2O) or three (L3O) units, with "
        "studies (a) or cohorts (b) as units; the main text reports the posterior median of $\\exp(\\mu)$, which is "
        f"{S.fmt_ratio(rb[rb.level == 'study'].iloc[0]['or_median'])} at study level against "
        f"{S.fmt_ratio(np.exp(rb[rb.level == 'study'].iloc[0]['mean']))} for the mean shown here. Open markers: "
        "fits that keep a single unit, whose $\\mu$ is that unit's effect shrunk towards the prior rather than "
        "a pooled estimate. Dashed line: no enrichment; dotted line: full-data posterior mean."
    )
    return fig, prov, caption


# ----------------------------------------------------------------------------- registry / CLI
FIGURES: dict[str, Callable] = {
    "figS1_ppc": fig_ppc,
    "figS2_obs_model": fig_obs_model,
    "figS3_deconcentration": fig_deconcentration,
    "figS4_gov_grid": fig_gov_grid,
    "figS5_compute_scaling": fig_compute_scaling,
    "figS6_fm_attention_null": fig_fm_attention_null,
    "figS7_recovery_calibration": fig_recovery_calibration,
    "figS8_replication_leavek": fig_replication_leavek,
}

# CSV files each supplementary figure depends on (used by the digest and the stale-table check)
FIGURE_SOURCES = {
    "figS1_ppc": ["fit_ppc.csv"],
    "figS2_obs_model": ["obs_params.csv", "obs_fit_summary.csv"],
    "figS3_deconcentration": ["gov_deconcentration.csv"],
    "figS4_gov_grid": ["gov_sensitivity_grid.csv"],
    "figS5_compute_scaling": ["scale_summary.csv", "scale_map.csv", "scale_nuts.csv", "scale_environment.csv"],
    "figS6_fm_attention_null": ["fm_attention_null.csv", "heldout_methods_context.csv"],
    "figS7_recovery_calibration": ["fit_recovery_calibration.csv", "fit_recovery.csv", "modes_faithful_design.csv"],
    "figS8_replication_leavek": ["rep_leavek.csv", "rep_bayes.csv"],
}


def build(only: list[str] | None = None, out_dir: Path | None = None) -> dict:
    S.apply_style()
    out = {}
    for name, fn in FIGURES.items():
        if only and name not in only:
            continue
        fig, prov, caption = fn()
        out[name] = save_figure(fig, name, prov, caption, out_dir=out_dir)
        out[name]["sources"] = sorted(prov.sources)
        print(f"wrote {name}: {', '.join(str(p.name) for k, p in out[name].items() if k != 'sources')}")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--only", nargs="*", default=None, help="figure names to build")
    ap.add_argument("--out-dir", default=None)
    a = ap.parse_args(argv)
    build(a.only, Path(a.out_dir) if a.out_dir else None)
    return 0


if __name__ == "__main__":
    main()
