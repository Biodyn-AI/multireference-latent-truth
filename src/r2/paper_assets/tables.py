"""Main manuscript tables (``paper/r2/tables/tab_*.tex``) built from ``data/r2/tables/*.csv``.

Every number is read from a CSV by the builders below (see ``textable.Table``); each
generated file starts with a provenance comment (source CSVs, modification times, row
filters).  Supplementary tables (``tabS_*``) are built by ``tables_supp``.

CLI::

    python -m r2.paper_assets.tables [--out DIR] [--only NAME ...] [--stale PREFIX ...]
                                     [--check-latex]

or ``python src/r2/make_paper_assets.py tables``.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from . import style as S
from . import textable as X
from .textable import DASH, Table, first_row, rows, stacked, tex
from ..common import PRESPECIFIED as PRE

# ----------------------------------------------------------------------------- labels
TISSUES = ["immune", "kidney", "lung"]
UNIVERSES = ["global", "immune", "kidney", "lung"]
STUDY_CITE = {
    "dixit2016": "dixit2016perturb", "adamson2016": "adamson2016multiplexed",
    "shifrut2018": "shifrut2018genome", "norman2019": "norman2019exploring",
    "papalexi2021": "papalexi2021characterizing", "replogle2022": "replogle2022mapping",
}
CHIP_KEY = {"encode_tfbs": "encode", "encode_strict": "encode_strict", "enets2": "enets2"}
LCM5 = ["em", "dependence", "multifactor", "family_regularized", "family_hb"]
RULES5 = ["union", "majority", "vote_fraction", "fisher", "stouffer"]


def U(u: str) -> str:
    return S.UNIVERSE_LABEL.get(u, u)


def SL(key: str) -> str:
    return tex(S.score_label(key))


def SHORT(key: str) -> str:
    st = S.SCORE_STYLE.get(key)
    return tex(st.short) if st and st.short else SL(key)


def RL(key: str) -> str:
    return tex(S.reference_label(CHIP_KEY.get(key, key)))


def q_strong(t: Table) -> str:
    """The Strong BFDR threshold as printed, read from gov_tiers.csv (registered as a cell).
    Only for quantities computed by the governance module from gov_tiers' threshold."""
    g = t.read("gov_tiers.csv", "q_strong of the first row (threshold definition)")
    return t.v(g.iloc[0], "q_strong", "ratio") if len(g) else DASH


def q_pre(key: str = "bfdr_strong") -> str:
    """A BFDR threshold of ``r2.common.PRESPECIFIED`` (the constant the analysis modules read)."""
    return X.fmt(PRE[key], "ratio")


def q_from_name(name: str) -> str:
    """The BFDR threshold encoded in a fixed column or set-rule name (``n_bfdr_0_05`` -> 0.05,
    ``bfdr05`` -> 0.05), so that a caption cannot drift from the data it describes."""
    m = re.search(r"(\d+)_(\d+)$", name)
    if m:
        return X.fmt(float(f"{m.group(1)}.{m.group(2)}"), "ratio")
    m = re.search(r"bfdr(\d+)$", name)        # two digits = hundredths (bfdr05 -> 0.05)
    if m:
        return X.fmt(int(m.group(1)) / 100.0, "ratio")
    raise ValueError(f"no threshold in {name!r}")


def _module_constant(module: str, name: str):
    """A design constant of an analysis module (read, never edited, by the asset scripts)."""
    import importlib
    return getattr(importlib.import_module(f"r2.{module}"), name)


def mc(n: int, text: str, align: str = "c") -> str:
    return f"\\multicolumn{{{n}}}{{{align}}}{{{text}}}"


def cmid(spec: str) -> str:
    return f"\\cmidrule(lr){{{spec}}}"


# ============================================================================= MAIN TABLES
def tab_universes() -> Table:
    t = Table("tab_universes", "tab:universes")
    ev = t.read("evidence_summary.csv", "variant == prespecified_cc & annotator == __ALL__")
    fm = t.read("fm_universe_summary.csv", "all tissues")
    gb = t.read("grn_baseline_runtime.csv", "variant == grnboost2 (primary)")
    ann = rows(ev, variant="prespecified_cc", universe="global")
    ann = ann[(ann.annotator != "__ALL__") & (ann.family != "CHIP")]
    t.note_source("evidence_summary.csv", "number of held-in annotators: variant == prespecified_cc & "
                  "universe == global & annotator != __ALL__")
    k = t.d(ann.annotator.nunique(), "count", [])
    t.caption = ("Candidate universes. Edges are ordered TF--target pairs; observation patterns are the "
                 f"distinct (observed, reported) vectors over the {k} held-in annotators under the "
                 "curated-conditional observation model; method coverage is the share of universe "
                 "edges a method scores.")
    body = []
    for u in TISSUES + ["global"]:
        r = first_row(ev, universe=u, variant="prespecified_cc", annotator="__ALL__")
        f = first_row(fm, tissue=u)
        g = first_row(gb, tissue=u, variant="grnboost2")
        if u == "global":
            body.append(("midrule",))
        body.append([U(u), t.v(r, "n_tfs", "count"), t.v(r, "n_genes", "count"),
                     t.v(r, "n_edges", "count"), t.v(r, "n_patterns", "count"),
                     t.v(r, "coverage", "pct"), t.v(r, "n_pos", "count"),
                     t.v(f, "coverage_scgpt_attn", "pct") if f is not None else DASH,
                     t.v(f, "coverage_geneformer_attn", "pct") if f is not None else DASH,
                     t.v(f, "coverage_pearson", "pct") if f is not None else DASH,
                     t.v(g, "frac_universe_returned", "pct") if g is not None else DASH])
    t.note("Observed: share of edges observed by at least one held-in annotator. "
           "Reported: edges with at least one held-in report. Correlation: Pearson and Spearman "
           "correlation (identical coverage). GRNBoost2: share of universe edges returned "
           "by the unfiltered primary run. Method scores exist only for the tissue-context "
           "universes (--).")
    header = ["Universe & TFs & Genes & Edges & Patterns & Observed & Reported & "
              + mc(4, "Method coverage (\\%)"),
              cmid("8-11"),
              " & & & & $P$ & (\\%) & edges & scGPT & Geneformer & Correlation & GRNBoost2"]
    t.tex = t.render("@{}lrrrrrrrrrr@{}", header, body, tabcolsep=3)
    return t


def tab_sources() -> Table:
    t = Table("tab_sources", "tab:sources",
              "Evidence annotators summarised by modality family (full list with versions, access "
              "dates and citations in Supplementary Table~\\ref{tab:S-sources}).")
    st = t.read("source_table.csv", "all rows, grouped by family")
    body = []
    fam_order = [f for f in ["LIT", "MOTIF", "COEXP", "CHIP"] if f in set(st.family)]
    lineage_abbr = {"CollecTRI": "C", "DoRothEA": "D", "Multi-integrator": "C+D",
                    "OmniPath-tf_target": "O"}
    for fam in fam_order:
        g = st[st.family == fam]
        names = ", ".join(tex("other literature" if s == "LIT_other" else s) for s in g.source_id)
        n = t.d(len(g), "count", [])
        edges = t.d(g.n_edges.sum(), "count", [(r, "n_edges") for _, r in g.iterrows()])
        mv = float(np.median(g.n_tfs))
        med = t.d(mv, "num0" if mv == round(mv) else "num1", [(r, "n_tfs") for _, r in g.iterrows()])
        lo = t.d(g.n_tfs.min(), "count", [(r, "n_tfs") for _, r in g.iterrows()])
        hi = t.d(g.n_tfs.max(), "count", [(r, "n_tfs") for _, r in g.iterrows()])
        tfs = f"{med} ({lo}--{hi})" if len(g) > 1 else med
        lin = []
        for lv, ab in lineage_abbr.items():
            k = int((g.lineage == lv).sum())
            if k:
                lin.append(f"{ab}\\,{t.d(k, 'count', [])}")
        held = g.held_out.astype(str).str.lower().eq("true")
        role = ("held out" if held.all() else "held in" if (~held).all() else "mixed")
        body.append([tex(S.FAMILY_LABEL.get(fam, fam)), n, edges, tfs, ", ".join(lin), role,
                     f"\\parbox[t]{{4.1cm}}{{\\raggedright {names}}}"])
    t.note("Edge records are summed over the annotators of a family (an edge reported by two "
           "annotators counts twice). TFs: median (range) over annotators. Lineage: integrator "
           "through which the records reached us (C, CollecTRI; D, DoRothEA; C+D, both; "
           "O, OmniPath TF--target interactions); TRRUST records were also read from its own file. Other "
           "literature: the DoRothEA curated reviews and NFIRegulomeDB, merged because each has few records. "
           "The ChIP family is held out of every primary fit because the validation references are ChIP-seq data.")
    header = ["Family & $n$ & Edge records & TFs per annotator & Lineage & Role & Annotators"]
    t.tex = t.render("@{}lrrrllp{4.1cm}@{}", header, body, tabcolsep=4)
    return t


def tab_chip() -> Table:
    t = Table("tab_chip", "tab:chip",
              "Held-out ChIP-seq references and the edges they can label (evaluable edges) in each "
              "universe.")
    ref = t.read("heldout_chip_table.csv", "all rows")
    cov = t.read("heldout_chip_coverage.csv", "all rows")
    body = []
    for _, r in ref.iterrows():
        key = r["key"]
        first = True
        second = False
        for u in UNIVERSES:
            c = first_row(cov, universe=u, reference=key)
            if c is None:
                continue
            if first:
                lead = [RL(key), t.v(r, "n_tfs", "count"), t.v(r, "n_edges", "count"),
                        t.v(r, "density", "pct")]
                second = True
            elif second:
                lead = [f"{{\\footnotesize ({tex(r['specificity'])})}}", "", "", ""]
                second = False
            else:
                lead = ["", "", "", ""]
            body.append(lead + [U(u), t.v(c, "evaluable_tfs", "count"),
                                t.v(c, "evaluable_edges", "count"), t.v(c, "positives", "count"),
                                t.v(c, "density", "pct")])
            first = False
        body.append(("addlinespace",))
    if body and body[-1] == ("addlinespace",):
        body.pop()
    t.note("Density: positives as a percentage of the reference's TF $\\times$ gene grid "
           "(reference) or of the evaluable edges (universe). An edge is evaluable if the "
           "reference assayed its TF and its target lies in the reference gene universe.")
    srcs = []
    for _, r in ref.iterrows():
        if str(r["key"]) == "encode_strict":
            continue
        srcs.append(f"{RL(r['key'])}: \\url{{{r['url']}}} ({tex(r['version'])})")
    t.note("Sources. " + "; ".join(srcs) + ". ENCODE-strict is derived from the ENCODE TFBS file.")
    header = [mc(4, "Reference", "c") + " & " + mc(5, "Evaluable in universe", "c"),
              cmid("1-4") + cmid("5-9"),
              " & TFs & Edges & Density (\\%) & Universe & TFs & Edges & Positives & Density (\\%)"]
    t.tex = t.render("@{}lrrrlrrrr@{}", header, body, tabcolsep=4)
    return t


DE_LABEL = {"pooled": "pooled $t$", "stratified": "stratified $t$", "pseudobulk_z": "pseudobulk $z$"}
DE_NOTE = {"pooled": "Welch $t$-test of the perturbed cells against the non-targeting (or intergenic-guide) controls",
           "stratified": "the same test stratified by replicate, or by donor $\\times$ stimulation",
           "pseudobulk_z": "approximate $z$-test on the published pseudobulk $z$-scores against controls of the same "
                           "GEM group (no replicate structure available)"}


def _accession(src) -> str:
    """Accession of a cohort for the table: 'GEO GSE90063 (scPerturb h5ad)' -> 'GSE90063'."""
    import re as _re
    txt = str(src)
    ids = _re.findall(r"(GSE\d+|GSM\d+)", txt)
    if not ids:
        m = _re.search(r"figshare plus (\d+)", txt)
        ids = ["figshare", m.group(1)] if m else [txt]
    if len(ids) == 1:
        return "{\\footnotesize " + tex(ids[0]) + "}"
    return "{\\footnotesize\\begin{tabular}[t]{@{}r@{}}" + "\\\\".join(tex(i) for i in ids) + "\\end{tabular}}"


def tab_cohorts() -> Table:
    t = Table("tab_cohorts", "tab:cohorts", "")
    ct = t.read("cohort_table.csv", "all cohorts")
    ov = t.read("cohort_universe_overlap.csv", "universe == global")
    order = list(ct.cohort_id)
    cites = []
    for sid in ct.study_id.drop_duplicates():
        cohorts = ", ".join(tex(S.reference_label(c)) for c in ct[ct.study_id == sid].cohort_id)
        cites.append(f"{cohorts}, \\citet{{{STUDY_CITE.get(sid, sid)}}}")
    t.caption = ("Perturbation cohorts (" + "; ".join(cites) + "). Positives: |log$_2$ fold change| "
                 "and adjusted $p$ thresholds of the primary differential-expression design "
                 "(Section~\\ref{sec:data-heldout}); same-band cis pairs excluded.")

    def two(text: str) -> str:
        parts = str(text).rsplit(" ", 1)
        if len(parts) == 1 or len(text) <= 9:
            return tex(text)
        return f"\\begin{{tabular}}[t]{{@{{}}r@{{}}}}{tex(parts[0])}\\\\{tex(parts[1])}\\end{{tabular}}"

    def line(label, fn):
        return [label] + [fn(first_row(ct, cohort_id=c), first_row(ov, cohort_id=c, universe="global"))
                          for c in order]

    body = [
        line("Cell type", lambda r, o: two(r["cell_type_group"])),
        line("Modality", lambda r, o: tex(r["modality"])),
        line("Accession", lambda r, o: _accession(r["accession_source"])),
        line("DE design", lambda r, o: DE_LABEL.get(str(r["primary_de"]), tex(r["primary_de"]))),
        line("Cells", lambda r, o: t.v(r, "n_cells", "count")),
        line("Control cells", lambda r, o: t.v(r, "n_controls", "count")),
        line("Perturbations", lambda r, o: t.v(r, "n_perturbations_tested", "count")),
        line("Perturbed TFs", lambda r, o: t.v(r, "n_perturbed_tfs", "count")),
        line("TF positive edges", lambda r, o: t.v(r, "n_positive_tf_edges_primary", "count")),
        ("panel", "Global universe"),
        line("Perturbed TFs", lambda r, o: t.v(o, "n_perturbed_tfs_in_universe", "count")),
        line("Evaluable pairs", lambda r, o: t.v(o, "n_pairs_evaluable", "count")),
        line("Positive pairs", lambda r, o: t.v(o, "n_positive_primary", "count")),
        line("TFs with a positive", lambda r, o: t.v(o, "n_tfs_with_positive", "count")),
    ]
    flags: dict[str, list[str]] = {}
    nonrobust: list[str] = []
    crit = ""
    for c in order:
        r_ = first_row(ct, cohort_id=c)
        fl = r_["flag"]
        if not (isinstance(fl, str) and fl.strip()):
            continue
        key = fl.split(";")[0].split(" (")[0].strip()
        if key.startswith("TF perturbations with non-robust DE"):
            # name the flagged TFs and how many of the cohort's perturbed TFs they are
            m_ = re.search(r"\(([^)]*Jaccard[^)]*)\)", fl)
            crit = m_.group(1) if m_ else crit
            tfs = str(r_["nonrobust_tf_perturbations"]).replace(";", ", ")
            nonrobust.append(f"{tex(S.reference_label(c))} ({tex(tfs)}; {int(r_['n_nonrobust_tf_perturbations'])} of "
                             f"{int(r_['n_perturbed_tfs'])} perturbed TFs)")
            continue
        flags.setdefault(key, []).append(tex(S.reference_label(c)))
    for key, cs in flags.items():
        txt = tex(key).replace("|log2FC|", "$|\\log_2\\mathrm{FC}|$")
        t.note(f"{', '.join(cs)}: {txt}.")
    if nonrobust:
        crit_txt = (tex(crit).replace("non-cis pooled-vs-stratified Jaccard", "Jaccard index of the non-cis positives "
                                      "of the pooled and the stratified test") if crit else "")
        t.note(f"TF perturbations whose positives are not robust to the DE design"
               + (f" ({crit_txt})" if crit_txt else "") + ": " + " and ".join(nonrobust) + ".")
    t.note("Accession: GEO series (GSE) or sample (GSM), or figshare record; the Adamson labels were re-derived "
           "from the GEO cell identities, and the Replogle data are the published pseudobulk profiles. "
           "GEM group: one 10x Genomics library (gel bead-in-emulsion batch).")
    used = [d for d in DE_LABEL if d in set(ct.primary_de.astype(str))]
    t.note("DE design: " + "; ".join(f"{DE_LABEL[d]}, {DE_NOTE[d]}" for d in used) + ".")
    header = [" & " + " & ".join(tex(S.reference_label(c)) for c in order)]
    t.tex = t.render("@{}l" + "r" * len(order) + "@{}", header, body, tabcolsep=1.8)
    return t


def tab_fit() -> Table:
    t = Table("tab_fit", "tab:fit",
              "Family-HB fit diagnostics per universe: posterior-mode search, mass of the second "
              "optimum, convergence of the dominant-mode NUTS run and the class weight of true "
              "edges.")
    conv = t.read("fit_convergence.csv", "fit_id in {family_hb, family_hb_nuts}")
    opt = t.read("fit_map_optima.csv", "fit_id == family_hb & optimum == 1")
    br = t.read("modes_bridge.csv", "role == primary_estimate & mode == 1")
    cw = t.read("fit_class_weights.csv", "fit_id == family_hb_nuts & class_name == T")
    cols = UNIVERSES
    m = {u: first_row(conv, universe=u, fit_id="family_hb") for u in cols}
    n = {u: first_row(conv, universe=u, fit_id="family_hb_nuts") for u in cols}
    o2 = {u: first_row(opt, universe=u, fit_id="family_hb", optimum=1) for u in cols}
    b2 = {u: first_row(br, universe=u, role="primary_estimate", mode=1) for u in cols}
    w = {u: first_row(cw, universe=u, fit_id="family_hb_nuts", class_name="T") for u in cols}

    def line(label, fn):
        return [label] + [fn(u) for u in cols]

    bridge_ph = any(X.is_posthoc_row(b2[u]) for u in cols if b2[u] is not None)

    def _p2_range(u):
        r = b2[u]
        if r is None or "p2_range_draw_sets_proposals_lo" not in r.index \
                or not X._isnum(r["p2_range_draw_sets_proposals_lo"]):
            return DASH
        # two lines (lowest-- / highest), so that the column keeps the width of the single estimates
        return ("\\begin{tabular}[t]{@{}r@{}}" + t.v(r, "p2_range_draw_sets_proposals_lo", "prob") + "--\\\\"
                + t.v(r, "p2_range_draw_sets_proposals_hi", "prob") + "\\end{tabular}")
    eea_ph = any(X.is_posthoc_col(w[u], "expected_edges_all") for u in cols if w[u] is not None)
    n0 = next((n[u] for u in cols if n[u] is not None), None)
    nuts_title = "Full posterior (NUTS, dominant mode"
    if n0 is not None:
        nuts_title += f"; {t.v(n0, 'num_chains', 'count')} chains $\\times$ {t.v(n0, 'num_samples', 'count')} draws"
    nuts_title += ")"
    body = [
        ("panel", "Posterior mode (multi-start MAP)"),
        line("Observation patterns $P$", lambda u: t.v(m[u], "n_patterns", "count")),
        line("Restarts", lambda u: t.v(m[u], "n_restarts", "count")),
        line("Distinct optima", lambda u: t.v(m[u], "n_distinct_optima", "count")),
        line("Restarts at the best optimum", lambda u: t.v(m[u], "n_restarts_at_best", "count")),
        line("Gap to second optimum (log post.)", lambda u: t.v(m[u], "gap_to_next_optimum", "num1")),
        line("Second-mode prob., Laplace", lambda u: t.v(o2[u], "laplace_rel_mass", "prob")),
        line("Second-mode prob., bridge sampling" + t.ph_if(bridge_ph),
             lambda u: t.est_ci(b2[u], "p_bridge", "p2_bridge_lo", "p2_bridge_hi", "prob", stack=True)
             if b2[u] is not None else DASH),
        line("\\quad range (draw sets, proposals)" + t.ph_if(bridge_ph), _p2_range),
        ("panel", nuts_title),
        line("Max.\\ $\\widehat{R}$, all sites", lambda u: t.v(n[u], "r_hat_max", "rhat")),
        line("Max.\\ $\\widehat{R}$, identified sites",
             lambda u: t.v(n[u], "r_hat_max_identified", "rhat")),
        line("Min.\\ bulk ESS, all sites", lambda u: t.v(n[u], "ess_bulk_min", "count")),
        line("Min.\\ bulk ESS, identified sites", lambda u: t.v(n[u], "ess_bulk_min_identified", "count")),
        line("Divergent transitions", lambda u: t.v(n[u], "n_divergent", "count")),
        line("Class weight $w_{\\mathrm{T}}$ [95\\% CrI]",
             lambda u: t.est_ci(w[u], "w", "w_lo", "w_hi", "prob", stack=True)),
        line("Expected true edges (observed)",
             lambda u: t.est_ci(w[u], "expected_edges_observed", "ee_lo", "ee_hi", "count", stack=True)),
        line("Expected true edges (all)" + t.ph_if(eea_ph),
             lambda u: t.est_ci(w[u], "expected_edges_all", "eea_lo", "eea_hi", "count", stack=True)),
    ]
    missing = [U(u) for u in cols if b2[u] is None]
    if missing:
        t.note("Bridge sampling not available for " + ", ".join(missing) + " (--).")
    t.note("Second-mode probability: approximate posterior mass of the second-best optimum relative "
           "to all optima (Laplace) or to the two best modes (bridge sampling on per-mode NUTS draws; "
           "Monte-Carlo 95\\% interval of the bridge estimate, which covers the within-run error only; range: "
           "smallest to largest bridge estimate over the draw sets and proposals that were run, i.e. Gaussian and "
           "Student-$t$ proposals and, in the tissue universes, a second set of dominant-mode draws). "
           "Likelihood-identified sites: class weights, "
           "annotator rates and credibilities. Expected true edges: posterior class weight $w_{\\mathrm{T}}$ times "
           "the number of edges observed by at least one annotator (observed) or of all universe edges (all), with "
           "95\\% credible interval.")
    nc = t.read("modes_bridge.csv", "role == raw_noncentred_coordinates (bridge check in the non-centred coordinates)")
    ncr = nc[nc.role == "raw_noncentred_coordinates"]
    if len(ncr):
        t.note("A further bridge-sampling check in the raw non-centred sampling coordinates" + t.ph() + " was too "
               "imprecise to be informative (relative error of every estimate at least "
               + t.d(float(np.floor(ncr.bridge_re_fs.astype(float).min() * 10) / 10), "num1",
                     [(ncr.loc[ncr.bridge_re_fs.astype(float).idxmin()], "bridge_re_fs")])
               + ") and is not shown.")
    header = [" & " + " & ".join(U(u) for u in cols)]
    t.tex = t.render("@{}lrrrr@{}", header, body, tabcolsep=4)
    return t


def _switch_list(text) -> str:
    """'lung:SIGNOR;lung:TRRUST' -> 'lung, SIGNOR and TRRUST dropped' (grouped by universe)."""
    if not isinstance(text, str) or not text.strip():
        return ""
    by: dict[str, list[str]] = {}
    for tok in text.split(";"):
        if ":" in tok:
            u, c = tok.strip().split(":", 1)
            by.setdefault(u, []).append(tex(c))
    parts = []
    for u, cs in by.items():
        cs_txt = cs[0] if len(cs) == 1 else ", ".join(cs[:-1]) + " and " + cs[-1]
        parts.append(f"{U(u).lower()}, {cs_txt} dropped")
    return "; ".join(parts)


def tab_robustness() -> Table:
    t = Table("tab_robustness", "tab:robustness")
    qs = q_pre("bfdr_strong")
    near = _module_constant("robustness", "NEAR_BEST")
    rk = t.read("rob_method_ranking.csv", "tissue == all_tissues & metric == f1 & subset in {all, no_fhb_mode_switch}; "
                                          "lineage: subset == excl_uninformative (configurations in which Family-HB is "
                                          "identified, the same for every estimator)")
    ct = t.read("rob_estimator_contrasts.csv", "scope == all_tissues & metric == D_f1 & reference == family_hb & "
                                                "protocol == source & subset in {all, no_fhb_mode_switch}")
    es = t.read("rob_edge_stability.csv", "universe == global & protocol == source & subset == all")
    t.note_source("robustness.py", "NEAR_BEST (log-posterior distance counted as the same mode)")

    def rkr(e, prot, sub="all"):
        return first_row(rk, tissue="all_tissues", estimator=e, protocol=prot, subset=sub, metric="f1")

    def ctr(e, sub):
        return first_row(ct, scope="all_tissues", protocol="source", subset=sub, estimator=e, reference="family_hb",
                         metric="D_f1")

    def dcell(r):
        return t.est_ci(r, "mean_D", "mean_D_cluster_lo", "mean_D_cluster_hi", "ratio3", stack=True)

    def ccell(r):
        return t.est_ci(r, "mean_diff", "mean_diff_cluster_lo", "mean_diff_cluster_hi", "ratio3", stack=True)

    h_all, h_ns = rkr("family_hb", "source"), rkr("family_hb", "source", "no_fhb_mode_switch")
    ns_ph = X.is_posthoc_row(h_ns)
    ident_ph = X.is_posthoc_col(h_all, "n_configs_identified")
    clus_ph = X.is_posthoc_col(h_all, "mean_D_cluster_lo")
    # lineage dropout: the configurations in which Family-HB is identified, for every estimator (equal sets)
    LIN_SUB = "excl_uninformative"
    body_a, notes_ident, changed = [], [], []

    def mark(txt, m):
        return txt.replace("\\\\[-1pt]", m + "\\\\[-1pt]", 1) if "\\\\[-1pt]" in txt else txt + m

    for grp, ests in (("Latent-class estimators", LCM5), ("Combination rules", RULES5)):
        body_a.append(("panel", grp))
        for e in ests:
            cells = [SL(e), dcell(rkr(e, "source")),
                     DASH if e == "family_hb" else ccell(ctr(e, "all")),
                     dcell(rkr(e, "source", "no_fhb_mode_switch")),
                     DASH if e == "family_hb" else ccell(ctr(e, "no_fhb_mode_switch"))]
            for prot in ("family", "lineage"):
                r = rkr(e, prot, LIN_SUB if prot == "lineage" else "all")
                txt = dcell(r)
                if r is not None and r["n_configs_identified"] < r["n_configs"]:
                    txt = mark(txt, "$^{a}$")
                    notes_ident.append((e, prot, r))
                if r is not None and S.as_bool(r.get("values_changed_post_hoc", False)):
                    txt = mark(txt, "$^{b}$")
                    changed.append(e)
                cells.append(txt)
            body_a.append(cells)
    for e, prot, r in notes_ident:
        t.note(f"$^{{a}}$\\,{SL(e)}, {prot} dropout: identified in {t.v(r, 'n_configs_identified', 'count')} "
               f"of {t.v(r, 'n_configs', 'count')} configurations{t.ph_if(ident_ph)} (one modality family left: true and "
               "literature-burst classes exchangeable); mean over the identified ones.")
    lin_h = rkr("family_hb", "lineage", LIN_SUB)
    lin_all = rkr("family_hb", "lineage")
    lin_ph = X.is_posthoc_row(lin_h)
    lin_txt = ""
    if lin_h is not None and lin_all is not None:
        lin_txt = (f"Lineage dropout{t.ph_if(lin_ph)}: the {t.v(lin_h, 'n_configs', 'count')} configurations in which "
                   "Family-HB is identified, for every estimator (the other "
                   f"{t.d(float(lin_all['n_configs']) - float(lin_all['n_configs_identified']), 'count', [(lin_all, 'n_configs'), (lin_all, 'n_configs_identified')])} "
                   f"of {t.v(lin_all, 'n_configs', 'count')} leave one modality family); Family-HB switches posterior "
                   f"mode in {t.v(lin_h, 'n_fhb_mode_switch', 'count')} of them.")
    if changed:
        lin_txt += (" " if lin_txt else "") + ("$^{b}$\\,Includes dropout fits whose optimum was found only by the "
                                              "extended start search or the independent-seed check (post hoc).")
    if lin_txt:
        t.note(lin_txt)
    nsw = h_all["n_fhb_mode_switch"] if h_all is not None and "n_fhb_mode_switch" in h_all.index else None
    excl = _switch_list(h_ns["fhb_mode_switch_excluded"]) if h_ns is not None else ""
    sw_txt = ""
    if h_all is not None and h_ns is not None:
        sw_txt = (f" Mode switches{t.ph_if(ns_ph)}: in {t.v(h_all, 'n_fhb_mode_switch', 'count')} of "
                  f"{t.v(h_all, 'n_configs', 'count')} source configurations ({excl}) the selected Family-HB dropout "
                  "optimum is not the continuation of the full-data mode (warm-started fit more than "
                  f"{X.fmt(near, 'sci2')} log-posterior units below it); the right-hand source columns drop them for "
                  "every estimator, which conditions on the Family-HB fit and decomposes, but does not replace, the "
                  "prespecified columns.")
    fam0 = rkr("em", "family")
    t.note("$\\Delta D$: paired difference to Family-HB (negative: more stable). Intervals: unadjusted 95\\% bootstrap "
           "intervals over configurations clustered by the dropped source, family or lineage (one cluster across the "
           f"three tissues{t.ph_if(clus_ph)})"
           + ("; family and lineage dropout (three families or lineages per tissue): descriptive."
              if fam0 is not None else ".") + sw_txt)

    # (b) edge-level stability, global universe
    e0 = first_row(es, universe="global", estimator="family_hb", protocol="source", subset="all")
    sm_ph = any(X.is_posthoc_col(r, "jaccard_top_sizematched_mean") for _, r in es.iterrows()) or \
        X.decision_status(t, "robustness", "rob_size_matched") == "post_hoc"
    body_b = []
    for grp, ests in (("Latent-class estimators", LCM5), ("Combination rules", RULES5)):
        body_b.append(("panel", grp))
        for e in ests:
            r = first_row(es, universe="global", estimator=e, protocol="source", subset="all")
            body_b.append([SL(e), t.v(r, "n_bfdr_full", "count"),
                           t.est_ci(r, "jaccard_bfdr_mean", "jaccard_bfdr_mean_boot_lo", "jaccard_bfdr_mean_boot_hi"),
                           t.est_ci(r, "jaccard_top_sizematched_mean", "jaccard_top_sizematched_mean_boot_lo",
                                    "jaccard_top_sizematched_mean_boot_hi"),
                           t.v(r, "spearman_mean", "ratio3")])
    no_sw = e0 is not None and X._isnum(e0.get("n_fhb_mode_switch")) and float(e0["n_fhb_mode_switch"]) == 0
    t.note(f"(b) $|S|$: size of the full-data posterior-mode BFDR\\,$\\le$\\,{qs} set (for a 0/1 rule its positives, several times "
           "the Family-HB set, which favours the rules' Jaccard index); size-matched: top $n$ edges, $n$ the size of "
           "the Family-HB set; Spearman: scores of observed edges. One configuration per source, so the intervals are "
           "source-clustered" + ("; no Family-HB mode switch." if no_sw else "."))
    n_src = t.v(h_all, "n_configs", "count") if h_all is not None else ""
    n_ns = t.v(h_ns, "n_configs", "count") if h_ns is not None else ""
    fam_h = rkr("family_hb", "family")
    n_fam = t.v(fam_h, "n_configs", "count") if fam_h is not None else ""
    n_lin = t.v(lin_h, "n_configs", "count") if lin_h is not None else ""
    head_a = [" & " + mc(2, f"Source dropout ({n_src})") + " & " + mc(2, f"Without mode switches ({n_ns}){t.ph_if(ns_ph)}")
              + f" & Family ({n_fam}) & Lineage ({n_lin}){t.ph_if(lin_ph)}",
              cmid("2-3") + cmid("4-5"),
              "Estimator & $D$ & $\\Delta D$ & $D$ & $\\Delta D$ & $D$ & $D$"]
    nb = t.v(e0, "n_configs", "count") if e0 is not None else ""
    head_b = [" & & " + mc(2, "Jaccard index [95\\% CI]") + " & ",
              cmid("3-4"),
              f"Estimator & $|S|$ & BFDR\\,$\\le$\\,{qs} set & Size-matched{t.ph_if(sm_ph)} & Spearman"]
    t.caption = ("Robustness to annotator dropout. (a) Rank disagreement $D$ of the tissue-method ranking "
                 "(posterior-weighted $F_1$), three tissue universes pooled, with and without the configurations in "
                 "which Family-HB switches posterior mode. (b) Edge-level stability in the global universe under "
                 f"single-source dropout ({nb} configurations): full-data sets against dropout sets.")
    t.tex = t.render_blocks([
        dict(title="(a) Method ranking, tissue universes", colspec="@{}lcccccc@{}", header=head_a, body=body_a,
             tabcolsep=4, size="footnotesize"),
        dict(title="(b) Edges, global universe", colspec="@{}lrccc@{}", header=head_b, body=body_b, tabcolsep=5,
             size="footnotesize"),
    ])
    return t


HELD_SCORES = (["family_hb_nuts", "family_hb", "em", "dependence", "multifactor", "family_regularized"],
               ["union", "majority", "vote_count", "fisher", "stouffer"],
               ["dorothea_level_heldin", "collectri_heldin"],
               ["random"])
HELD_LABEL = {"family_hb": "Family-HB (MAP)"}


def tab_heldout() -> Table:
    t = Table("tab_heldout", "tab:heldout")
    he = t.read("heldout_estimators.csv", "universe == global & eval_subset == all_evaluable")
    hp = t.read("heldout_paired.csv", "universe == global & eval_subset == all_evaluable & "
                                      "{metric == ap_lift & analysis_role == confirmatory; "
                                      "metric == ap_lift_comp_adj & confirmatory_comparator}")
    t.note_source("heldout_estimators.csv", "columns ap_lift, ap_lift_comp_adj (+ _lo/_hi), within_tf_auroc, "
                                            "enr_at_500 (+ _lo/_hi), strat_enr_at_500")
    body = []
    score_a, n_comp = set(), set()
    ph_adj = False
    for ref in ("encode", "enets2", "pert_pooled"):
        r0 = first_row(he, universe="global", reference=ref, eval_subset="all_evaluable", score="family_hb")
        if r0 is None:
            continue
        title = (f"{RL(ref)}: {t.v(r0, 'n_eval', 'count')} evaluable edges, "
                 f"{t.v(r0, 'n_pos', 'count')} positives, {t.v(r0, 'n_tfs', 'count')} TFs")
        body.append(("panel", title))
        for gi, grp in enumerate(HELD_SCORES):
            if gi:
                body.append(("addlinespace",))
            for s in grp:
                r = first_row(he, universe="global", reference=ref, eval_subset="all_evaluable", score=s)
                if r is None:
                    continue
                pr = first_row(hp, universe="global", reference=ref, eval_subset="all_evaluable",
                               metric="ap_lift", score_b=s, analysis_role="confirmatory")
                pa = first_row(hp, universe="global", reference=ref, eval_subset="all_evaluable",
                               metric="ap_lift_comp_adj", score_b=s, confirmatory_comparator=True)
                if pr is not None:
                    score_a.add(pr["score_a"])
                    n_comp.add((ref, s))
                if pa is not None:
                    score_a.add(pa["score_a"])
                    ph_adj = ph_adj or X.is_posthoc_row(pa)
                pcell = t.p(pr, "p_holm_confirmatory", "p_holm_confirmatory_floor") if pr is not None else DASH
                acell = t.p(pa, "p_holm_6comparators", "p_holm_6comparators_floor") if pa is not None else DASH
                lab = tex(HELD_LABEL.get(s, S.score_label(s))) + t.ph_if(X.is_posthoc_row(r))
                body.append([lab, t.est_ci(r, "ap_lift", "ap_lift_lo", "ap_lift_hi", "ratio"), pcell,
                             t.est_ci(r, "ap_lift_comp_adj", "ap_lift_comp_adj_lo", "ap_lift_comp_adj_hi", "ratio"),
                             acell, t.v(r, "within_tf_auroc", "ratio3"),
                             t.est_ci(r, "enr_at_500", "enr_at_500_lo", "enr_at_500_hi", "ratio"),
                             t.v(r, "strat_enr_at_500", "ratio")])
    if score_a != {"family_hb"}:
        raise ValueError(f"tab_heldout: paired tests are not all Family-HB (MAP) minus comparator: {score_a}")
    pr_any = first_row(hp, universe="global", metric="ap_lift", analysis_role="confirmatory")
    nb = t.v(pr_any, "n_boot_10k", "count") if pr_any is not None and "n_boot_10k" in pr_any.index else ""
    per_ref = max((sum(1 for rr, _ in n_comp if rr == ref) for ref in ("encode", "enets2", "pert_pooled")), default=0)
    r_any = first_row(he, universe="global", reference="encode", eval_subset="all_evaluable", score="family_hb")
    adj_col_ph = X.is_posthoc_col(r_any, "ap_lift_comp_adj")
    ph_p = any(X.is_posthoc_col(r, "p_holm_confirmatory") for _, r in
               rows(hp, universe="global", metric="ap_lift", analysis_role="confirmatory").iterrows())
    t.caption = ("Held-out evaluation in the global universe (all evaluable edges): AP lift and composition-adjusted AP "
                 "lift (95\\% TF-cluster bootstrap intervals; Holm $p$ of Family-HB (MAP) against each comparator of the "
                 "primary comparison family), within-TF AUROC, and enrichment of the top 500 edges, unstratified and stratified.")
    t.note(f"Holm $p$: Family-HB (MAP) minus comparator, paired TF-cluster bootstrap"
           + (f" ({nb} resamples)" if nb and nb != DASH else "")
           + f", Holm over the {t.d(per_ref, 'count', [])} comparators of the primary comparison family (post hoc) of each reference ($\\leq$: "
           "resampling floor; --: not tested; the NUTS fit was not tested). Composition-adjusted AP lift: AP over its exact "
           "expectation under permutation of the labels within TFs (1: no ranking signal within TFs); its paired tests "
           "are descriptive.")
    t.note("All lifts and enrichments equal 1, within-TF AUROC 0.5, for a random ranking. Enr@500: positives among the "
           "500 highest-ranked edges over the base-rate expectation (the metric of the paired Enr@500 contrasts); "
           "Strat.: within TF $\\times$ target-degree strata. All scores and references: Supplementary "
           "Tables~\\ref{tab:S-heldout-chip}--\\ref{tab:S-heldout-pert-enr}.")
    d_adj = t.ph_if(adj_col_ph or ph_adj)
    head = [" & " + mc(2, "AP lift") + " & " + mc(2, "Composition-adjusted AP lift" + d_adj) + " & Within-TF & "
            + mc(2, "Enr@500"),
            cmid("2-3") + cmid("4-5") + cmid("7-8"),
            "Score & [95\\% CI] & Holm $p$" + t.ph_if(ph_p) + " & [95\\% CI] & Holm $p$" + d_adj
            + " & AUROC & [95\\% CI] & Strat."]
    t.tex = t.render_blocks([dict(colspec="@{}lccccccc@{}", header=head, body=body, tabcolsep=2.4,
                                  size="footnotesize")], arraystretch=0.93)
    return t


def tab_status() -> Table:
    t = Table("tab_status", "tab:status", "")
    st = t.read("heldout_evidence_status.csv", "reference in {encode, enets2, pert_pooled}")
    ph = all(X.is_posthoc_row(r) for _, r in st.iterrows())
    t.caption = ("Held-out positive rate by evidence status" + (" (post hoc analysis" + t.ph() + ")" if ph else "")
                 + ": observed over TF-matched expected "
                 "positives (O/E, 95\\% TF-cluster bootstrap interval) and mean Family-HB credibility "
                 "$\\bar c$ of evaluable edges that no annotator observes, that are observed but not "
                 "reported, or that carry at least one report.")
    body = []
    low = False
    for ref in ("encode", "enets2", "pert_pooled"):
        sub = rows(st, reference=ref)
        if not len(sub):
            continue
        body.append(("panel", RL(ref)))
        for u in UNIVERSES:
            cells = [U(u)]
            any_low = False
            for status in ("unobserved", "observed_neg", "reported"):
                r = first_row(st, universe=u, reference=ref, status=status)
                if r is not None and S.as_bool(r["low_support"]):
                    any_low = True
                cells += [t.est_ci(r, "oe_tf_matched", "oe_lo", "oe_hi", "ratio", stack=True),
                          t.v(r, "mean_cred_family_hb", "sig2")]
            if any_low:
                cells[0] += "$^{*}$"
                low = True
            body.append(cells)
    if low:
        t.note("$^{*}$\\,At least one status group has low support (few positives or TFs).")
    t.note("O/E = 1: positives occur at the rate expected from the TF's own positive rate. "
           "$\\bar c$: mean posterior-mode credibility of the group.")
    header = ["Universe & " + mc(2, "Unobserved") + " & " + mc(2, "Observed, not reported") + " & " +
              mc(2, "Reported"),
              cmid("2-3") + cmid("4-5") + cmid("6-7"),
              " & O/E & $\\bar c$ & O/E & $\\bar c$ & O/E & $\\bar c$"]
    t.tex = t.render("@{}lcccccc@{}", header, body)
    return t


def tab_decomp(credibility: str = "global_fit", name: str = "tab_decomp", label: str = "tab:decomp") -> Table:
    cred_lab = {"global_fit": "global-fit", "context_fit": "context-fit"}[credibility]
    t = Table(name, label,
              "Universe-aware decomposition of the context--global difference in raw precision of "
              f"the top 500 edges ({cred_lab} Family-HB credibility): raw ratio = base-rate ratio "
              "$\\times$ enrichment ratio, and comparison with prevalence-matched random "
              "sub-universes of the global universe drawn edge by edge or as whole TF blocks.")
    dc = t.read("heldout_universe_decomposition.csv", f"metric == p_at_500 & credibility == {credibility}")
    nl = t.read("heldout_downsampling_null.csv",
                f"metric == p_at_500 & credibility == {credibility} & null in {{prevalence_matched, tf_block_matched}}")
    body = []
    tfb_ph = False
    for ref in [r for r in ("encode", "pert_pooled", "encode_strict", "enets2") if r in set(dc.reference)]:
        r0 = first_row(dc, reference=ref, metric="p_at_500", credibility=credibility)
        body.append(("panel", f"{RL(ref)} (global universe: P@500 = {t.v(r0, 'value_glob', 'ratio')}, "
                              f"base rate {t.v(r0, 'base_rate_glob', 'ratio')})"))
        for tis in TISSUES:
            r = first_row(dc, tissue=tis, reference=ref, metric="p_at_500", credibility=credibility)
            if r is None:
                continue
            e = first_row(nl, tissue=tis, reference=ref, metric="p_at_500", credibility=credibility,
                          null="prevalence_matched")
            b = first_row(nl, tissue=tis, reference=ref, metric="p_at_500", credibility=credibility,
                          null="tf_block_matched")
            tfb_ph = tfb_ph or X.is_posthoc_row(b)
            body.append([U(tis), t.v(r, "base_rate_ctx", "ratio"), t.v(r, "ratio_base_rate", "ratio"),
                         t.v(r, "value_ctx", "ratio"),
                         t.est_ci(r, "ratio_raw", "ratio_raw_lo", "ratio_raw_hi", "ratio", stack=True),
                         t.est_ci(r, "ratio_enrichment", "ratio_enrichment_lo", "ratio_enrichment_hi",
                                  "ratio", stack=True),
                         t.est_ci(e, "null_median", "null_lo", "null_hi", "ratio", stack=True),
                         t.v(e, "frac_gap_surviving", "ratio"),
                         t.est_ci(b, "null_median", "null_lo", "null_hi", "ratio", stack=True),
                         t.v(b, "frac_gap_surviving", "ratio")])
    t.note("Ratios are context over global. Null: median and 95\\% interval of P@500 over random "
           "sub-universes with the context's number of evaluable edges and exactly its number of "
           "positives, drawn as edges or as whole TF blocks matched to the context's per-TF sizes "
           "and positive counts. Surviving: fraction of the context--global gap not reproduced by "
           "the null, (context $-$ null median)/(context $-$ global); negative values mean that the "
           "context value exceeds its null although raw precision falls. The edge-level null ignores the "
           "clustering of edges within TFs and is anti-conservative; the TF-block null keeps it.")
    r_any = first_row(dc, credibility=credibility)
    ci_ph = X.is_posthoc_col(r_any, "ratio_raw_lo")
    if ci_ph:
        t.note("Intervals: TF-cluster bootstrap of the context and the global set drawn independently "
               "(conservative when the sets share TFs)" + t.ph() + ".")
    d = t.ph_if(tfb_ph)
    header = [" & " + mc(2, "Base rate") + " & " + mc(3, "P@500") + " & " +
              mc(2, "Edge null") + " & " + mc(2, "TF-block null" + d),
              cmid("2-3") + cmid("4-6") + cmid("7-8") + cmid("9-10"),
              "Tissue & Context & Ratio & Context & Raw ratio & Enrichment ratio & Median & "
              "Surviving & Median & Surviving"]
    t.tex = t.render("@{}lccccccccc@{}", header, body, tabcolsep=4)
    return t


def tab_governance() -> Table:
    t = Table("tab_governance", "tab:governance",
              "Claim governance. Dual-universe tiers in the tissue-context universes (95\\% interval "
              "over posterior draws), their posterior expected FDR under the global and the context "
              "fit, agreement of the two fits, and held-out enrichment of promoted edges after TF "
              "matching (O/E, 95\\% TF-cluster bootstrap interval). Global universe: single-fit tiers.")
    gt = t.read("gov_tiers.csv", "scope == dual")
    gc = t.read("gov_consistency.csv", "credibility == nuts & scope in {all_edges, either_ge_moderate}")
    gv = t.read("gov_tier_validation.csv",
                "tissues: scope == dual & tier == Promoted; global: scope == global_fit & tier in {Strong, Moderate, Promoted}")
    cols = TISSUES + ["global"]

    def tier(u, name):
        if u == "global":
            r = first_row(gv, universe="global", scope="global_fit", tier=name, reference="encode")
            if r is None:
                return DASH
            return t.v(r, "n_edges", "count") + t.ph_if(X.is_posthoc_row(r))
        r = first_row(gt, tissue=u, scope="dual", tier=name)
        return t.est_ci(r, "n_edges", "n_edges_draw_lo", "n_edges_draw_hi", "count", stack=True)

    def bfdr(u, name):
        if u == "global":
            return DASH
        r = first_row(gt, tissue=u, scope="dual", tier=name)
        return f"{t.v(r, 'bfdr_global_fit', 'ratio3')} / {t.v(r, 'bfdr_context_fit', 'ratio3')}"

    def kappa(u, scope):
        if u == "global":
            return DASH
        r = first_row(gc, tissue=u, credibility="nuts", scope=scope)
        return t.v(r, "kappa_linear", "ratio")

    few = []

    def oe(u, ref):
        if u == "global":
            r = first_row(gv, universe="global", scope="global_fit", tier="Promoted", reference=ref)
        else:
            r = first_row(gv, universe=u, scope="dual", tier="Promoted", reference=ref)
        if r is None:
            return DASH
        txt = t.est_ci(r, "oe_tf", "oe_tf_lo", "oe_tf_hi", "ratio", stack=True)
        mark = ""
        if S.as_bool(r["few_clusters"]):
            mark += "$^{*}$"
            few.append(1)
        mark += t.ph_if(X.is_posthoc_row(r))
        return txt.replace("\\\\[-1pt]", mark + "\\\\[-1pt]", 1) if "\\\\[-1pt]" in txt else txt + mark

    def line(label, fn):
        return [label] + [fn(u) for u in cols]

    body = [
        ("panel", "Tier size (edges)"),
        line("Strong", lambda u: tier(u, "Strong")),
        line("Moderate", lambda u: tier(u, "Moderate")),
        line("Conditional", lambda u: tier(u, "Conditional") if u != "global" else DASH),
        line("Promoted (Strong $\\cup$ Moderate)", lambda u: tier(u, "Promoted")),
        ("panel", "Posterior expected FDR (global fit / context fit)"),
        line("Strong", lambda u: bfdr(u, "Strong")),
        line("Promoted", lambda u: bfdr(u, "Promoted")),
        ("panel", "Agreement of global and context fit (linearly weighted $\\kappa$)"),
        line("All edges", lambda u: kappa(u, "all_edges")),
        line("Edges $\\geq$ Moderate in either fit", lambda u: kappa(u, "either_ge_moderate")),
        ("panel", "Held-out O/E of promoted edges, TF-matched"),
        line(RL("encode"), lambda u: oe(u, "encode")),
        line(RL("enets2"), lambda u: oe(u, "enets2")),
        line(RL("pert_pooled"), lambda u: oe(u, "pert_pooled")),
    ]
    if few:
        t.note("$^{*}$\\,Few TF clusters among the evaluable promoted edges; interval unreliable.")
    q0 = first_row(gt, tissue="immune", scope="dual", tier="Strong")
    qs = t.v(q0, "q_strong", "ratio") if q0 is not None else DASH
    qm = t.v(q0, "q_moderate", "ratio") if q0 is not None else DASH
    t.note(f"Single-fit levels: Strong, the largest prefix of whole tie blocks with posterior expected FDR "
           f"$\\leq$ {qs}; Moderate, the largest prefix with $\\leq$ {qm}, minus Strong. Dual tiers: Strong if Strong "
           "under both fits (the intersection of the two single-fit Strong sets); Moderate if at least Moderate under "
           "both, not Strong; Conditional if at least Moderate under exactly one fit. Tiers use posterior-mean "
           "credibilities. The global universe has a single fit, so it has no Conditional tier "
           "and no cross-universe agreement (--).")
    header = [" & " + " & ".join(U(u) for u in cols)]
    t.tex = t.render("@{}lcccc@{}", header, body)
    return t


METHOD_ORDER = ["all_edges", "scgpt_attn", "scgpt_attn_norm", "geneformer_attn", "geneformer_attn_norm",
                "grnboost2", "pearson_abs", "random"]


def tab_methods_promoted() -> Table:
    t = Table("tab_methods_promoted", "tab:methods-promoted", "")
    k0 = float(PRE["topk_per_tf"])
    gp = t.read("gov_promoted_by_method.csv", f"k == {int(k0)}; method == all_edges (k empty)")
    r0 = first_row(gp, k=k0)
    k_txt = t.v(r0, "k", "count") if r0 is not None else "50"
    t.caption = (f"Promoted share of each method's predicted set (top {k_txt} targets per TF) in the "
                 "tissue-context universes, against the share expected from the TF composition of the "
                 "set (TF-matched expectation); exact one-sided $p$ for enrichment and depletion.")
    ph_p = X.is_posthoc_col(r0, "p_enriched_exact") or X.is_posthoc_col(r0, "p_depleted_exact")
    body = []
    for tis in TISSUES:
        body.append(("panel", U(tis)))
        for mth in METHOD_ORDER:
            r = (first_row(gp, tissue=tis, method=mth, k=None) if mth == "all_edges"
                 else first_row(gp, tissue=tis, method=mth, k=k0))
            if r is None:
                continue
            body.append([SHORT(mth), t.v(r, "n_predicted", "count"), t.v(r, "n_promoted", "count"),
                         t.est_ci(r, "frac_promoted", "frac_promoted_lo", "frac_promoted_hi", "pct2"),
                         t.v(r, "expected_frac_promoted_tf", "pct2"), t.v(r, "ratio_to_tf_expected", "ratio"),
                         t.p(r, "p_enriched_exact"), t.p(r, "p_depleted_exact")])
    t.note("All universe edges: the whole context universe (reference row; its promoted share is the share of "
           "promoted edges in the universe and equals its TF-matched expectation by definition). Interval: 95\\% "
           "TF-cluster bootstrap interval of the promoted share. Exact $p$: conditional on, for every TF, its numbers of "
           "universe, promoted and predicted edges; the test treats the edges of a TF as exchangeable and is "
           "anti-conservative when they are dependent.")
    header = ["Method & Predicted & " + mc(2, "Promoted") + " & Expected & Ratio & "
              + mc(2, "Exact $p$" + t.ph_if(ph_p)),
              cmid("3-4") + cmid("7-8"),
              " & edges & $n$ & \\% [95\\% CI] & \\% & & enriched & depleted"]
    t.tex = t.render("@{}lrrcrccc@{}", header, body, tabcolsep=4)
    return t


def tab_replication() -> Table:
    t = Table("tab_replication", "tab:replication", "")
    ru = t.read("rep_units.csv", "universe == global & level == study & comparator == universe")
    rp = t.read("rep_power.csv", "universe == global & level == study & comparator == universe")
    pr = t.read("rep_pass_rates.csv", "universe == global & level == study & comparator == universe")
    p0 = first_row(rp, universe="global", level="study", comparator="universe")
    te = t.v(p0, "true_enrichment", "num0") if p0 is not None else ""
    tp = t.v(p0, "target_power", "pct0") if p0 is not None else ""
    cites = "; ".join(f"{tex(S.reference_label(u))}, \\citet{{{STUDY_CITE.get(u, u)}}}"
                      for u in rp[(rp.universe == "global") & (rp.level == "study")].unit.drop_duplicates())
    t.caption = ("Cross-cohort replication check in the global universe, per independent study (" + cites + "). Design "
                 "quantities were computed and locked before any replication outcome: power of the exact "
                 f"stratified test against a {te}-fold enrichment, minimum detectable enrichment (MDE) "
                 f"and support floor (promoted evaluable edges for {tp}\\% power). Outcome: stratified "
                 "O/E with 95\\% TF-cluster bootstrap interval, Mantel--Haenszel odds ratio and exact "
                 "one-sided $p$.")
    ph_att = X.decision_status(t, "replication", "rep_attainable") == "post_hoc"
    ph_wt = X.decision_status(t, "replication", "rep_wrongtf") == "post_hoc"
    ph_rs = X.decision_status(t, "replication", "rep_responsiveness") == "post_hoc"
    ph_bt = X.decision_status(t, "replication", "rep_boot_clusters") == "post_hoc"
    min_boot = _module_constant("replication", "MIN_BOOT_CLUSTERS")
    t.note_source("replication.py", "MIN_BOOT_CLUSTERS (TFs needed for a bootstrap interval)")
    undefined = []
    body = []
    order = list(ru[(ru.universe == "global") & (ru.level == "study") & (ru.comparator == "universe")].unit)
    for unit in order:
        r = first_row(ru, universe="global", level="study", comparator="universe", unit=unit)
        lab = tex(S.reference_label(unit))
        if not str(r["status"]).startswith("evaluable"):
            body.append([lab, t.v(r, "n_promoted_evaluable", "count"),
                         mc(10, "\\emph{" + tex(str(r["status"])) + "}", "l")])
            continue
        lo_, hi_ = r.get("enrichment_boot_lo"), r.get("enrichment_boot_hi")
        if X._isnum(lo_) and X._isnum(hi_) and float(lo_) < float(hi_):
            oe = t.est_ci(r, "enrichment", "enrichment_boot_lo", "enrichment_boot_hi", "ratio", stack=True)
        else:                       # no interval (too few TFs) or a degenerate one (every resample equal)
            oe = stacked(t.v(r, "enrichment", "ratio"), DASH)
            undefined.append(lab)
        if S.as_bool(r["endpoint_pass"]):
            status = "pass"
        elif not S.as_bool(r["adequately_powered"]):
            status = "\\begin{tabular}[t]{@{}l@{}}under-\\\\powered\\end{tabular}"
        else:
            status = "fail"
        if str(r["endpoint_status"]) == "endpoint_unattainable":
            status += " (unattainable" + t.ph_if(X.is_posthoc_col(r, "endpoint_status")) + ")"
        body.append([lab, t.v(r, "n_promoted_evaluable", "count"), t.v(r, "power", "ratio"),
                     t.v(r, "mde_rr", "ratio"), t.v(r, "floor_edges", "count"),
                     t.v(r, "max_attainable_enrichment", "ratio"), oe, t.v(r, "mh_odds_ratio", "ratio"),
                     t.p(r, "p_exact"), status, t.v(r, "wrongtf_enrichment", "ratio"),
                     t.v(r, "resp_strat_enrichment", "ratio")])
    a = first_row(pr, universe="global", level="study", comparator="universe", subset="all_evaluable")
    b = first_row(pr, universe="global", level="study", comparator="universe", subset="adequately_powered")
    if a is not None:
        al = t.v(p0, "alpha", "ratio") if p0 is not None else DASH
        txt = (f"Pass (O/E $\\geq$ 1.5 and $p \\leq$ {al}): {t.v(a, 'n_pass', 'count')} of "
               f"{t.v(a, 'n_units', 'count')} evaluable studies (Wilson 95\\% CI "
               f"{t.v(a, 'wilson_lo', 'ratio')}--{t.v(a, 'wilson_hi', 'ratio')}; a study below its support floor is "
               "reported as underpowered, not as a failure, but counts as not passing in this rate)")
        if b is not None:
            txt += (f"; {t.v(b, 'n_pass', 'count')} of {t.v(b, 'n_units', 'count')} adequately powered "
                    f"studies ({t.v(b, 'wilson_lo', 'ratio')}--{t.v(b, 'wilson_hi', 'ratio')})")
        t.note(txt + ".")
    t.note(f"Intervals{t.ph_if(ph_bt)}: TF-cluster bootstrap within the study, given only for studies with at least "
           f"{X.fmt(min_boot, 'count')} TFs"
           + (f"; --, no interval: fewer TFs, or every resample gives the same O/E ({', '.join(undefined)})"
              if undefined else "") + ".")
    t.note("Exact $p$: one-sided exact test conditional on the margins of every (cohort, TF) stratum; it treats the "
           "edges of a stratum as exchangeable and is anti-conservative when targets of one TF are dependent, so "
           "the TF-cluster bootstrap intervals carry the inference.")
    t.note("Floor $\\infty$: no number of promoted edges reaches the target power. Max.\\ O/E: largest "
           "O/E the margins allow. Wrong-TF: O/E when each promoted edge is scored against the "
           "perturbation of a different TF of the same cohort (specificity control). Resp.-strat.: "
           "O/E with strata additionally defined by target responsiveness (share of perturbations "
           "in which the target responds).")
    header = ["Study & Promoted & " + mc(3, "Design (locked)") + " & Margins & " + mc(4, "Outcome") + " & " +
              mc(2, "Controls (O/E)"),
              cmid("3-5") + cmid("6-6") + cmid("7-10") + cmid("11-12"),
              " & evaluable & Power & MDE & Floor & Max. & O/E & MH & $p$ & Result & Wrong & Resp.-",
              " & edges & & & & O/E" + t.ph_if(ph_att) + " & [95\\% CI]" + t.ph_if(ph_bt) + " & OR & & & TF"
              + t.ph_if(ph_wt) +
              " & strat." + t.ph_if(ph_rs)]
    t.tex = t.render("@{}lrrrrrcrclrr@{}", header, body, tabcolsep=2.4)
    return t


def _machine(t: Table, env_csv: str) -> str:
    env = t.read(env_csv, "keys cpu, hw.ncpu, hw.memsize (bytes)")
    cpu, ncpu, mem = (first_row(env, key=k) for k in ("cpu", "hw.ncpu", "hw.memsize (bytes)"))
    if cpu is None or ncpu is None or mem is None:
        return "one laptop"
    return (f"one laptop ({tex(cpu['value'])}, {t.d(float(ncpu['value']), 'count', [(ncpu, 'value')])} "
            f"cores, {t.d(float(mem['value']) / 2 ** 30, 'num0', [(mem, 'value')])}\\,GiB)")


def _scaling_float(t: Table) -> tuple[str, str, list, list]:
    """Synthetic compute-scaling block (scale_summary.csv, scale_nuts.csv): caption, colspec, header, body.
    Adds its notes to ``t``."""
    sc = t.read("scale_summary.csv", "generator in {template, independent, stress}, selected configurations")
    nt = t.read("scale_nuts.csv", "cid == template_real_K21 (run settings); status == projected (range over sessions)")
    fc = t.read("fit_convergence.csv", "universe == global & fit_id == family_hb_nuts (production NUTS run)")
    machine = _machine(t, "scale_environment.csv")
    ref = first_row(sc, generator="template", size="real", K=21)
    nr = first_row(nt, cid="template_real_K21", status="measured")
    n_map = t.v(ref, "map_restarts_run", "count") if ref is not None else DASH
    nuts_set = (f"{t.v(nr, 'chains', 'count')} chains $\\times$ ({t.v(nr, 'warmup', 'count')} warm-up + "
                f"{t.v(nr, 'samples', 'count')} draws)") if nr is not None else "short chains"
    cap = ("Compute scaling on synthetic evidence (no held-out data). Wall time on "
           f"{machine} for pattern compression, one log-posterior and gradient evaluation, {n_map} MAP restarts and "
           f"a short NUTS run ({nuts_set}, random starts), with peak resident memory.")
    sel = [("template", "1e5", 21), ("template", "1e6", 21), ("template", "real", 21),
           ("template", "1e7", 21), ("template", "10xreal", 21), ("template", "10xreal", 42),
           ("independent", "real", 21), ("stress", "1e6", 21), ("stress", "1e7", 21)]
    gen_label = {"template": "Real-data template", "independent": "Independent annotators",
                 "stress": "Stress (no repeated patterns)"}
    size_label = {"real": "real size", "10xreal": "10 $\\times$ real"}
    used = set()
    body, last_gen = [], None
    sel_rows = [first_row(sc, generator=g_, size=s_, K=k_) for g_, s_, k_ in sel]
    all_ph = all(r_ is not None and X.is_posthoc_row(r_) for r_ in sel_rows)
    if all_ph:          # one dagger on the caption instead of one per row
        cap = cap.rstrip(".") + " (post hoc measurement session" + t.ph() + ")."
    for gen, size, k in sel:
        r = first_row(sc, generator=gen, size=size, K=k)
        if r is None:
            continue
        if gen != last_gen:
            body.append(("panel", gen_label.get(gen, gen)))
            last_gen = gen
        ph = "" if all_ph else t.ph_if(X.is_posthoc_row(r))
        mst = str(r["map_10_status"])
        m_mark = {"extrapolated": "$^{e}$", "projected": "$^{m}$"}.get(mst, "")
        used.add(m_mark)
        tvg = t.d(float(r["t_vg_s"]) * 1000.0, "sig2", [(r, "t_vg_s")])
        if not X._isnum(r["nuts_2x300_s"]):
            ncell = DASH
        elif str(r["nuts_status"]) == "projected":
            pr = rows(nt, cid=r["cid"], status="projected")
            lo_r = pr.loc[pr["t_nuts_projected_s"].astype(float).idxmin()]
            hi_r = pr.loc[pr["t_nuts_projected_cap_s"].astype(float).idxmax()]
            ncell = (t.d(float(lo_r["t_nuts_projected_s"]), "count", [(lo_r, "t_nuts_projected_s")]) + "--"
                     + t.d(float(hi_r["t_nuts_projected_cap_s"]), "count", [(hi_r, "t_nuts_projected_cap_s")])
                     + "$^{n}$")
            used.add("$^{n}$")
        else:
            ncell = t.v(r, "nuts_2x300_s", "count")
        body.append([size_label.get(size, f"$N \\approx 10^{{{size[-1]}}}$") + ("\\," + ph if ph else ""),
                     t.v(r, "N", "count"), t.v(r, "K", "count"), t.v(r, "P", "count"),
                     t.v(r, "compress_s", "sig2"), t.v(r, "compress_peak_rss_mib", "count"), tvg,
                     t.v(r, "map_10_s", "count") + m_mark, t.v(r, "map_peak_rss_mib", "count"), ncell])
    if any(sz == "real" for _, sz, _ in sel):
        t.note("Real size: a synthetic universe with as many TFs and genes as the global universe, so $N$ differs from "
               "the real number of edges only by the TF self-pairs, which the real universe excludes; its number of "
               "patterns $P$ differs from the real one because its observations and reports are simulated.")
    if "$^{e}$" in used:
        t.note("$^{e}$\\,MAP: extrapolated from the mean time of the restarts that finished within the time budget.")
    if "$^{m}$" in used:
        t.note("$^{m}$\\,MAP not run: evaluations per restart of the first session $\\times$ measured time per "
               "evaluation; a lower bound (optimiser overhead not included).")
    if "$^{n}$" in used:
        cap_steps = _module_constant("compute_scaling", "NUTS_MAX_STEPS")
        t.note("$^{n}$\\,NUTS not run (time budget); range over the two measurement sessions from the projection with "
               "the leapfrog steps per iteration of the largest measured run with the same $K$ (lower end: steps per "
               "iteration grew with $N$, so this is a lower bound) to the projection at the tree-depth cap of "
               f"{X.fmt(cap_steps, 'count')} steps per iteration (upper end, the better estimate for "
               "$N \\geq 10^7$).")
    g = first_row(fc, universe="global", fit_id="family_hb_nuts")
    rh = t.v(nr, "r_hat_max", "ratio") if nr is not None else DASH
    prod = ""
    if g is not None:
        prod = (f" The MAP-initialised production fit of the global universe ({t.v(g, 'num_chains', 'count')} chains "
                f"$\\times$ ({t.v(g, 'num_warmup', 'count')} + {t.v(g, 'num_samples', 'count')})) took "
                f"{t.v(g, 'runtime_s', 'count')}\\,s.")
    t.note("$N$: edges; $K$: annotators; $P$: distinct observation patterns. Measured wall times are upper bounds of the "
           "undisturbed cost (machine shared with other jobs). Real-size template: observation design and fitted "
           "parameters of the global universe. The NUTS benchmark starts chains at random and is a cost measurement, "
           f"not a converged fit (maximum $\\widehat R$ {rh} at real size: chains in different modes)." + prod)
    header = [" & $N$ & $K$ & $P$ & " + mc(2, "Compression") + " & Evaluation & " + mc(2, "MAP restarts")
              + " & NUTS",
              cmid("5-6") + cmid("8-9"),
              "Data set & & & & s & MiB & ms & s & MiB & s"]
    return cap, "@{}lrrrrrrrrr@{}", header, body


def tab_compute() -> Table:
    """Computation (reviewer Q21/Q22).  With ``pipeline_stage_timings.csv`` present: two floats, the
    per-stage wall time and peak memory of the real pipeline (label ``tab:compute``) and the synthetic
    scaling experiment (label ``tab:compute-scaling``); otherwise the scaling table alone."""
    t = Table("tab_compute", "tab:compute")
    if not (S.TABLES / "pipeline_stage_timings.csv").exists():
        cap, spec, head, body = _scaling_float(t)
        t.caption = cap
        t.tex = t.render(spec, head, body, tabcolsep=4)
        return t
    cap2, spec2, head2, body2 = _scaling_float(t)
    t.caption, t.label = cap2, "tab:compute-scaling"
    tex2 = t.render(spec2, head2, body2, tabcolsep=4)
    tex2 = "\n".join(ln for ln in tex2.splitlines() if not ln.startswith("%")) + "\n"
    t.notes, t.uses_posthoc, t.label = [], False, "tab:compute"
    head1, body1, cap1 = _pipeline_into(t)
    t.caption = cap1 + " Synthetic scaling: Table~\\ref{tab:compute-scaling}."
    tex1 = t.render("@{}lrrlrr@{}", head1, body1, tabcolsep=5)
    t.tex = tex1 + "\n" + tex2
    return t


PIPELINE_LABEL = {
    "panels": "Gene panels", "annotators": "Annotators (committed inputs)", "evidence": "Evidence patterns",
    "grnboost2": "GRNBoost2 / GENIE3", "fm_scores": "Method scores", "cohorts": "Perturbation cohorts",
    "heldout_chip": "Held-out ChIP", "fit_primary": "Primary fits (MAP, NUTS)",
    "fit_sensitivity": "Sensitivity, CV, PPC, recovery", "obs_models": "Informative-observation fits",
    "posterior_modes": "Posterior modes (fits, bridge, simulation)", "robustness": "Dropout robustness",
    "heldout_eval": "Held-out evaluation", "governance": "Claim governance",
    "modes_tables": "Posterior-mode tables, tier sensitivity", "replication": "Replication check",
    "tuning": "Tuning optimism", "obs_eval": "Informative-observation evaluation", "sim_hub": "Hub-bias simulation",
    "compute_scaling": "Compute scaling", "obs_integration": "Informative-observation governance",
    "paper_assets": "Manuscript assets",
}
STEP_LABEL = {"laplace": "Laplace approximation", "evidence": "evidence encoding", "seedcheck": "independent-seed check",
              "oracle-check": "oracle check", "restart-check": "restart check", "map": "posterior-mode fits",
              "nuts": "NUTS fits", "search": "extended start search", "bridge": "bridge sampling",
              "simulate": "simulation", "tables": "table step"}
PIPELINE_LAYER = {"data": "Data layer", "model": "Model layer", "evaluation": "Evaluation", "assets": "Assets"}


def _pipeline_into(t: Table) -> tuple[list, list, str]:
    """Per-stage wall time and peak memory of the real pipeline (pipeline_stage_timings.csv) into ``t``
    (cells, notes); returns (header, body, caption).

    Per stage and mode the selected rows of the timing table are summed (wall time) and maximised
    (peak RSS of the largest process).  Full run = every step of the stage; tables-only run = the
    steps that regenerate the tables from stored outputs (``src/r2/reproduce_all.py --tables-only``)."""
    from .. import reproduce_all as RA
    df = t.read("pipeline_stage_timings.csv", "selected == True; summed per stage and mode")
    df = df[df["selected"].astype(str) == "True"]
    machine = _machine(t, "pipeline_environment.csv")
    mark = {"measured_isolated": "M", "recorded_meta": "R", "recorded_table": "R", "recorded_log": "L"}
    body, last_layer = [], None
    tot = {"full": [0.0, []], "tables": [0.0, []]}
    untimed, missing_steps, rss_partial, used_nm = [], [], False, False

    def fmin(x, srcs):
        """Minutes: 2 significant digits, thousands separator from 1,000 on."""
        return t.d(x, "count" if abs(x) >= 1000 else "sig2", srcs)
    for st in RA.STAGES:
        cells = []
        for mode, modes in (("full", ("full", "full+tables")), ("tables", ("tables_only", "full+tables"))):
            want = [x.name for x in st.steps if (x.full if mode == "full" else x.tables)]
            sel = df[(df["stage"] == st.name) & df["mode"].isin(modes) & df["step"].isin(want)]
            if not want:
                cells += [DASH, DASH] + ([DASH] if mode == "full" else [])
                continue
            if not len(sel):
                used_nm = True
                cells += ["n.m.", DASH] + ([DASH] if mode == "full" else [])
                if mode == "full":
                    untimed.append(PIPELINE_LABEL.get(st.name, st.name))
                continue
            srcs = [(r, "wall_time_s") for _, r in sel.iterrows()]
            wall = float(sel["wall_time_s"].astype(float).sum())
            tot[mode][0] += wall
            tot[mode][1] += srcs
            rss = sel["peak_rss_mib"].astype(float)
            r_rss = sel.loc[rss.idxmax()] if rss.notna().any() else None
            cells.append(fmin(wall / 60.0, srcs))
            w_rss = float(sel.loc[rss.notna(), "wall_time_s"].astype(float).sum())
            part = r_rss is not None and wall > 0 and w_rss < 0.5 * wall
            rss_partial |= part
            cells.append((t.d(float(r_rss["peak_rss_mib"]) / 1024.0, "sig2", [(r_rss, "peak_rss_mib")])
                          + ("$^{b}$" if part else "")) if r_rss is not None else DASH)
            if mode == "full":
                code = "+".join(sorted({mark.get(x, "?") for x in sel["timing_source"]}))
                missing = [x for x in want if x not in set(sel["step"])]
                if missing:
                    code += "$^{a}$"
                    missing_steps.append(f"{tex(PIPELINE_LABEL.get(st.name, st.name))}: "
                                         + ", ".join(tex(STEP_LABEL.get(m_, m_.replace("_", " ").replace("-", " ")))
                                                     for m_ in missing))
                cells.append(code)
        if st.layer != last_layer:
            body.append(("panel", PIPELINE_LAYER.get(st.layer, st.layer)))
            last_layer = st.layer
        body.append([tex(PIPELINE_LABEL.get(st.name, st.name))] + cells)
    if missing_steps:
        t.note("$^{a}$\\,Not included in the total: no runtime is recorded for the steps " + "; ".join(missing_steps)
               + ".")
    if rss_partial:
        t.note("$^{b}$\\,Memory is known only for steps that make up less than half of the stage's wall time "
               "(the long fitting steps did not record their peak memory).")
    body.append(("midrule",))
    body.append(["Total"] + [fmin(tot["full"][0] / 60.0, tot["full"][1]), "", "",
                             fmin(tot["tables"][0] / 60.0, tot["tables"][1]), ""])
    meas = df[df["timing_source"] == "measured_isolated"]
    la = ""
    if len(meas):
        lo = meas.loc[meas["loadavg_1m_start"].astype(float).idxmin()]
        hi = meas.loc[meas["loadavg_1m_max"].astype(float).idxmax()]
        la = (f" The 1-min load average of the machine (other users' jobs included) ranged "
              f"{t.d(float(lo['loadavg_1m_start']), 'sig2', [(lo, 'loadavg_1m_start')])}--"
              f"{t.d(float(hi['loadavg_1m_max']), 'sig2', [(hi, 'loadavg_1m_max')])} during the measured steps.")
    # share of the full-run time that is recorded (R, L) rather than measured in isolation (M): the same rows
    # as the full-run total
    w_m = sum(float(r_["wall_time_s"]) for r_, _ in tot["full"][1] if r_["timing_source"] == "measured_isolated")
    share_m = w_m / tot["full"][0] if tot["full"][0] > 0 else float("nan")
    # recorded over isolated wall time of the same steps (MAP fits; kidney NUTS from the MAP start): how far the
    # recorded runtimes of the contended original runs exceed the cost on a quiet machine
    full_df = t.read("pipeline_stage_timings.csv", "fit_primary map and kidney NUTS: recorded vs measured wall time")
    ratios = []
    for step_, scope_ in (("map", "all"), ("nuts", "kidney, map_laplace init")):
        g = full_df[(full_df["stage"] == "fit_primary") & (full_df["step"] == step_) & (full_df["scope"] == scope_)]
        rec = g[g["timing_source"].astype(str).str.startswith("recorded")]
        mea = g[g["timing_source"] == "measured_isolated"]
        if len(rec) and len(mea) and float(mea.iloc[0]["wall_time_s"]) > 0:
            ratios.append((float(rec.iloc[0]["wall_time_s"]) / float(mea.iloc[0]["wall_time_s"]),
                           [(rec.iloc[0], "wall_time_s"), (mea.iloc[0], "wall_time_s")]))
    ratio_txt = ""
    if ratios:
        lo_, hi_ = min(ratios, key=lambda x: x[0]), max(ratios, key=lambda x: x[0])
        ratio_txt = (f"; isolated re-runs of the primary MAP fits and of the kidney NUTS fit took "
                     f"{t.d(lo_[0], 'sig2', lo_[1])}--{t.d(hi_[0], 'sig2', hi_[1])} times less wall time than "
                     "recorded, so the total describes the runs as made rather than the cost on an idle machine")
    cap = ("Wall time and peak resident memory of each stage of the analysis pipeline on the real data on "
           f"{machine}, run through the single entry point \\texttt{{reproduce\\_all.py}}. Full run: every step of the stage; "
           "tables-only: the steps that regenerate the tables from the stored fits and outputs. Steps marked M were "
           "measured as isolated runs, one step process at a time with two threads; steps marked R or L are the "
           "runtimes recorded when the long fits originally ran, partly in parallel and with other thread settings on "
           "the shared machine"
           + (f", and make up {t.d(100.0 * (1.0 - share_m), 'num0', tot['full'][1])}\\% of the full-run time"
              if share_m == share_m else "")
           + "." + la)
    t.note("Timing: M, measured as an isolated run; R, runtime recorded by the analysis script when the step "
           "originally ran (summed over its fits or replicates; no peak memory unless the script stored it); L, from "
           "the time stamps in the script's logs" + ("; n.m., not measured" if used_nm else "") + ". All times come from a machine "
           "shared with other jobs; the recorded R and L times come from more heavily loaded runs and are the least "
           "precise. Peak memory: largest single process (GRNBoost2: summed over its dask workers). "
           f"Totals: full run {t.d(tot['full'][0] / 3600.0, 'sig2', tot['full'][1])}\\,h of compute as recorded on the "
           "shared machine" + ratio_txt + "; "
           f"tables-only run {t.d(tot['tables'][0] / 60.0, 'sig2', tot['tables'][1])}\\,min.")
    if untimed:
        t.note("No runtime is available for: " + ", ".join(tex(x) for x in untimed) + ".")
    header = [" & " + mc(3, "Full run") + " & " + mc(2, "Tables only"),
              cmid("2-4") + cmid("5-6"),
              "Stage & min & GiB & timing & min & GiB"]
    return header, body, cap


# Retired table names: still accepted by ``build`` / ``--only`` (``reproduce_all.refresh_assets`` asks for
# tab_pipeline) but not written; a stale file of that name is removed (verifier assets_fix L12: the stand-alone
# copy of the stage-timing table was \input nowhere and duplicated the first float of tab_compute).
RETIRED = {"tab_pipeline": "the per-stage timing table is the first float of tab_compute (label tab:compute)"}


# ============================================================================= registry / CLI
MAIN: dict[str, Callable[[], Table]] = {
    "tab_universes": tab_universes,
    "tab_sources": tab_sources,
    "tab_chip": tab_chip,
    "tab_cohorts": tab_cohorts,
    "tab_fit": tab_fit,
    "tab_robustness": tab_robustness,
    "tab_heldout": tab_heldout,
    "tab_status": tab_status,
    "tab_decomp": tab_decomp,
    "tab_governance": tab_governance,
    "tab_methods_promoted": tab_methods_promoted,
    "tab_replication": tab_replication,
    "tab_compute": tab_compute,
}


def registry() -> dict[str, Callable[[], Table]]:
    from . import tables_supp
    reg = dict(MAIN)
    reg.update(tables_supp.SUPP)
    return reg


def build(names=None, out_dir: Path = S.TAB_DIR, write: bool = True) -> dict[str, Table]:
    reg = registry()
    names = [n for n in (list(names) if names else list(reg)) if n not in RETIRED]
    if write:
        for n in RETIRED:
            (Path(out_dir) / f"{n}.tex").unlink(missing_ok=True)
    built = {}
    for nm in names:
        tb = reg[nm]()
        tabs = tb if isinstance(tb, list) else [tb]
        for tt in tabs:
            built[tt.name] = tt
            if write:
                X.write_table(tt.tex, out_dir, tt.name)
    if write:
        _write_cells(built, out_dir)
    return built


def _write_cells(built: dict[str, Table], out_dir: Path = S.TAB_DIR) -> Path:
    """Cell registry of every printed number (merged with the registry of tables not rebuilt)."""
    out = (S.SP / "outputs" / "r2" / "paper_assets") if out_dir.resolve() == S.TAB_DIR.resolve() else out_dir
    out.mkdir(parents=True, exist_ok=True)
    recs = []
    for nm, tt in built.items():
        for c in tt.cells:
            recs.append(dict(table=nm, csv=c.csv, row=c.row, column=c.column, value=c.value, kind=c.kind,
                             text=c.text, derived=c.derived,
                             sources=json.dumps(c.sources) if c.derived else ""))
    p = out / "tables_cells.csv"
    new = pd.DataFrame(recs)
    if p.exists():
        old = pd.read_csv(p)
        old = old[~old.table.isin(list(built) + list(RETIRED))]
        new = pd.concat([old, new], ignore_index=True)
    new.to_csv(p, index=False)
    sp = out / "tables_stale_sources.json"
    stale = json.loads(sp.read_text()) if sp.exists() else {}
    for nm in RETIRED:
        stale.pop(nm, None)
    for nm, tt in built.items():
        stale.pop(nm, None)
        if tt.stale_sources:
            stale[nm] = sorted(tt.stale_sources)
    sp.write_text(json.dumps(stale, indent=1))
    return p


PREAMBLE = r"""\documentclass[11pt,a4paper]{article}
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage{mathptmx}
\usepackage[margin=2.5cm]{geometry}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{amsmath,amssymb}
\usepackage{xcolor}
\usepackage{hyperref}
\usepackage{cleveref}
\usepackage[authoryear,round]{natbib}
\usepackage{caption}
\usepackage{subcaption}
\usepackage{multirow}
\usepackage{float}
\usepackage{url}
\captionsetup{font=small, labelfont=bf}
\hfuzz=0.5pt
% citations are not resolved in the check document; give them a realistic width
\renewcommand{\citet}[1]{Abcdefgh et al.\ (2019)}
\renewcommand{\citep}[1]{(Abcdefgh et al., 2019)}
\begin{document}
"""


def check_latex(tex_dir: Path = S.TAB_DIR, names=None, workdir: Path | None = None) -> dict:
    """Compile every table with the R1 preamble; report errors and overfull boxes per table."""
    workdir = workdir or (S.SP / "outputs" / "r2" / "paper_assets" / "latex_check")
    workdir.mkdir(parents=True, exist_ok=True)
    files = sorted(tex_dir.glob("*.tex")) if names is None else [tex_dir / f"{n}.tex" for n in names]
    result = {}
    for f in files:
        doc = (PREAMBLE + f"\\input{{{f.resolve()}}}\n\\clearpage\n"
               "\\section{Data}\\label{sec:data-heldout}\\label{sec:res-fit}\n\\end{document}\n")
        src = workdir / f"check_{f.stem}.tex"
        src.write_text(doc)
        pr = subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", src.name],
                            cwd=workdir, capture_output=True, text=True, timeout=180)
        log = (workdir / f"check_{f.stem}.log").read_text(errors="replace")
        over = [float(m) for m in re.findall(r"Overfull \\hbox \(([0-9.]+)pt too wide\)", log)]
        errs = re.findall(r"^! (.*)$", log, flags=re.M)
        pages = re.search(r"Output written on .*?\((\d+) page", log)
        result[f.stem] = dict(ok=pr.returncode == 0 and not errs, errors=errs[:3],
                              overfull_pt=max(over) if over else 0.0,
                              pages=int(pages.group(1)) if pages else None,
                              float_too_large="Float too large" in log)
    return result


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=S.TAB_DIR)
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--stale", nargs="*", default=[],
                    help="CSV name prefixes still being regenerated (marked in the outputs)")
    ap.add_argument("--check-latex", action="store_true")
    a = ap.parse_args(argv)
    X.STALE_PREFIXES[:] = a.stale
    built = build(a.only, a.out)
    print(f"wrote {len(built)} tables to {a.out}")
    for nm, tt in built.items():
        flag = f"  STALE: {', '.join(sorted(tt.stale_sources))}" if tt.stale_sources else ""
        print(f"  {nm}: {len(tt.cells)} cells{flag}")
    if a.check_latex:
        res = check_latex(a.out, list(built))
        bad = {k: v for k, v in res.items() if not v["ok"] or v["overfull_pt"] > 1.0 or v["float_too_large"]}
        for k, v in res.items():
            print(f"  latex {k}: ok={v['ok']} overfull={v['overfull_pt']:.1f}pt pages={v['pages']}"
                  f"{' FLOAT TOO LARGE' if v['float_too_large'] else ''} {v['errors'] if v['errors'] else ''}")
        return 1 if bad else 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
