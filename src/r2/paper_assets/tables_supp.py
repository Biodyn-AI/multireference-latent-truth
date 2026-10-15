"""Supplementary manuscript tables (``paper/r2/tables/tabS_*.tex``).

Same conventions as ``tables`` (every number read from ``data/r2/tables/*.csv`` through
``textable.Table``; provenance comment; post hoc rows / columns marked with a dagger).
Long tables are split into several ``table`` floats in one file (``render_split``),
because the R1 preamble has no longtable package.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import style as S
from . import textable as X
from .textable import DASH, Table, breakable, first_row, render_split, rows, stacked, tex
from .tables import (LCM5, RL, RULES5, SHORT, SL, STUDY_CITE, TISSUES, U, UNIVERSES, _module_constant, cmid, mc,
                     q_from_name, q_pre, q_strong, tab_decomp)

# ----------------------------------------------------------------------------- labels
VARIANT_LABEL = {
    "prespecified_cc": "Primary (curated-conditional)",
    "prespecified": "Literal observation mask",
    "both_endpoints": "Both endpoints required",
    "both_endpoints_cc": "Both endpoints, curated-cond.",
    "chip_prespecified_cc": "ChIP family included",
    "lit_merged_extri_ntnu": "ExTRI and NTNU merged",
    "lit_no_ht": "Without very large regulons",
    "lit_no_chipderived": "Without ChIP-derived LIT",
    "no_motif_coexp": "Literature only",
}
FIT_LABEL = {
    "family_hb": "Family-HB",
    "family_hb_nuts": "Family-HB (NUTS)",
    "family_hb_randinit_nuts": "Family-HB (NUTS, random init.)",
    "family_hb__fam_singleton": "Singleton families",
    "family_hb__fam_lineage": "Lineage families",
    "family_hb__fam_datadriven": "Data-driven families",
    "family_hb__prior_scale0.5": "Scale priors halved",
    "family_hb__prior_scale2": "Scale priors doubled",
    "family_hb__prior_dir0.5": "Dirichlet(0.5)",
    "family_hb__prior_dir2": "Dirichlet(2)",
    "family_hb__prior_scale0.5_nuts": "Scale priors halved (NUTS)",
    "family_hb__prior_scale2_nuts": "Scale priors doubled (NUTS)",
}
ANN_LABEL = {"LIT_other": "Other literature"}
LINEAGE_ABBR = {"CollecTRI": "C", "DoRothEA": "D", "Multi-integrator": "C+D", "OmniPath-tf_target": "O"}
SENS_ORDER = ([("evidence_variant", v, "family_hb") for v in
               ["prespecified", "both_endpoints", "both_endpoints_cc", "chip_prespecified_cc",
                "lit_merged_extri_ntnu", "lit_no_ht", "lit_no_chipderived", "no_motif_coexp"]]
              + [("family_grouping", "prespecified_cc", f) for f in
                 ["family_hb__fam_singleton", "family_hb__fam_lineage", "family_hb__fam_datadriven"]]
              + [("prior", "prespecified_cc", f) for f in
                 ["family_hb__prior_scale0.5", "family_hb__prior_scale2", "family_hb__prior_dir0.5",
                  "family_hb__prior_dir2"]])
SENS_GROUP = {"evidence_variant": "Evidence variants (Family-HB)",
              "family_grouping": "Family groupings", "prior": "Priors"}


def config_label(cid: str) -> str:
    """Readable name of a tuning configuration id (``family_hb@variant``, ``family_hb__x``, ...)."""
    if cid == "mean_of_grid":
        return "grid average"
    if "@" in cid:
        f, v = cid.split("@", 1)
        lab = VARIANT_LABEL.get(v, v)
        lab = lab[0].lower() + lab[1:] if lab[1:2].islower() else lab
        return f"{S.score_label(f)}, {lab}"
    if cid in FIT_LABEL:
        return FIT_LABEL[cid]
    return S.score_label(cid)


def sens_label(variant: str, fit_id: str) -> str:
    return tex(VARIANT_LABEL.get(variant, variant)) if fit_id == "family_hb" else tex(FIT_LABEL.get(fit_id, fit_id))


# ============================================================================= tables
# bibliography keys (paper/r2/references.tex) of the primary citation of each resource
SOURCE_CITE = {
    "ExTRI": ["vazquez2022extri"], "NTNU.Curated": ["mullerdott2023collectri"], "GEREDB": ["huang2019geredb"],
    "TRRUST": ["han2018trrust"], "TRED": ["jiang2007tred"], "TFactS": ["essaghir2010tfacts"],
    "HTRIdb": ["bovolenta2012htridb"], "SIGNOR": ["perfetto2016signor"], "Pavlidis2021": ["chu2021curation"],
    "PAZAR": ["portales2009pazar"], "KEGG": ["kanehisa2012kegg"], "CytReg": ["carrascopro2018cytreg"],
    "IntAct": ["orchard2014intact"], "GOA": ["huntley2015goa"], "ORegAnno": ["lesurf2016oreganno"],
    "TFe": ["yusuf2012tfe"], "FANTOM4": ["fantom2009transcriptional", "kawaji2011fantom"],
    "LIT_other": ["garcia2019benchmark", "gronostajski2011nfi"],
    "HOCOMOCO": ["kulakovskiy2018hocomoco"], "JASPAR": ["khan2018jaspar"],
    "ARACNe-GTEx": ["margolin2006reverse", "garcia2019benchmark"], "PAZAR-HT": ["portales2009pazar"],
    "ReMap": ["cheneby2018remap"], "ENCODE-proximal": ["gerstein2012architecture"],
    "ENCODE-distal": ["gerstein2012architecture"],
}
CHANNEL_ABBR = {"CollecTRI": "C", "DoRothEA": "D", "TRRUST-file": "T", "OmniPath-tf_target": "O"}


# Published resource URLs that no longer resolve although the resource is online elsewhere (checked 2026-10-15);
# the source table keeps the published URL, the supplementary table prints the current one and says so in a note.
URL_MOVED = {"ORegAnno": ("https://oreganno.org", "http://www.oreganno.org", "2026-10-15",
                          "leads to the UCSC Genome Browser ORegAnno track")}


def _source_url(text: str) -> tuple[str, bool]:
    """First URL of a resource-URL cell and whether the resource was offline when accessed."""
    import re as _re
    m = _re.search(r"https?://[^\s;,)]+", str(text))
    off = bool(_re.search(r"offline|unreachable", str(text)))
    return (m.group(0) if m else ""), off


def _short_version(text: str) -> str:
    """Version or access text without the file-integrity details (kept in the source table)."""
    v = str(text).split(";")[0].strip().replace("tf_target", "TF--target")
    return v.replace(" (DoRothEA 2019)", ", via DoRothEA (2019)").replace(" (DoRothEA)", ", via DoRothEA")


def tabS_sources() -> Table:
    t = Table("tabS_sources", "tab:S-sources",
              "Evidence annotators: primary resource, modality family, channel through which the records were "
              "obtained, version, citation, resource URL and size (edges, TFs and targets after normalisation to "
              "approved HGNC symbols and Lambert TFs).")
    st = t.read("source_table.csv", "all rows")
    body = []
    missing = []
    for fam in ["LIT", "MOTIF", "COEXP", "CHIP"]:
        g = st[st.family == fam]
        if not len(g):
            continue
        body.append(("panel", tex(S.FAMILY_LABEL.get(fam, fam)) + (" (held out)" if fam == "CHIP" else "")))
        for _, r in g.iterrows():
            sid = str(r["source_id"])
            keys = SOURCE_CITE.get(sid)
            if not keys:
                missing.append(sid)
                keys = []
            ch = "+".join(CHANNEL_ABBR.get(c.strip(), tex(c.strip())) for c in str(r["integrator_channels"]).split(";"))
            url, off = _source_url(r["resource_url"])
            if sid in URL_MOVED and url == URL_MOVED[sid][1]:
                url = URL_MOVED[sid][0]
            ver = _short_version(r["version_or_access_date"])
            # an offline resource is printed without a live link: its old domain may now serve unrelated content
            url_cell = (f"\\parbox[t]{{3.1cm}}{{\\raggedright\\scriptsize "
                        + (f"\\nolinkurl{{{url}}} (offline)" if off else f"\\url{{{url}}}") + "}") if url else "--"
            nl = max(1.0, 0.75 * max(np.ceil(len(ver) / 28), np.ceil(len(url) / 30), 1))
            body.append(("row", [tex(sid if sid != "LIT_other" else "Other literature"), ch,
                                 f"\\parbox[t]{{2.8cm}}{{\\raggedright\\scriptsize {breakable(ver)}}}",
                                 "\\parbox[t]{2.3cm}{\\raggedright\\scriptsize " + "; ".join(f"\\citet{{{k}}}" for k in keys) + "}",
                                 url_cell,
                                 t.v(r, "n_edges", "count"), t.v(r, "n_tfs", "count"), t.v(r, "n_targets", "count")],
                         nl))
    if missing:
        raise ValueError(f"tabS_sources: no citation key for {missing}")
    import re as _re
    how = " ".join(str(x) for x in st.how_obtained.dropna().unique())
    acc_c = _re.search(r"CollecTRI query \(accessed ([0-9-]+)\)", how)
    acc_d = _re.search(r"DoRothEA A-D query \([^)]*accessed ([0-9-]+)\)", how)
    t.note("Channel: how the records were obtained, through OmniPath queries of CollecTRI (C"
           + (f", accessed {acc_c.group(1)}" if acc_c else "") + ") or of "
           "DoRothEA confidence levels A--D (D, omnipath Python client"
           + (f", accessed {acc_d.group(1)}" if acc_d else "") + "), the TRRUST~v2 human file "
           "(T), or OmniPath TF--target evidence attached to DoRothEA and CollecTRI records (O). "
           "Other literature: the DoRothEA curated reviews and NFIRegulomeDB, which have too few records to model "
           "separately. Offline: the resource URL did not respond when the data were accessed (printed without a link); "
           "the records come from the integrators. "
           + " ".join(f"{tex(k)}: the published URL \\nolinkurl{{{old}}} no longer resolves (checked {d}); "
                      f"\\nolinkurl{{{new}}} {what}." for k, (new, old, d, what) in URL_MOVED.items()
                      if k in set(st.source_id.astype(str)))
           + " DOIs, family-assignment criteria and processing notes are in the source table of the code repository, "
           "and the checksums of the downloaded integrator and TRRUST files in its input manifest.")
    header = ["Resource & Channel & Version & Citation & Resource URL & Edges & TFs & Targets"]
    t.tex = render_split(t, "@{}llp{2.8cm}p{2.3cm}p{3.1cm}rrr@{}", header, body, max_lines=40, tabcolsep=2.0)
    return t


def tabS_annotator_coverage() -> Table:
    t = Table("tabS_annotator_coverage", "tab:S-annotator-coverage",
              "Coverage of the held-in annotators under the curated-conditional observation model in each "
              "universe: TFs covered, edges observed (share of the universe) and reports (share of observed "
              "edges).")
    ev = t.read("evidence_summary.csv", "variant == prespecified_cc & annotator != __ALL__")
    sub = ev[(ev.variant == "prespecified_cc") & (ev.annotator != "__ALL__")]
    order = list(sub[sub.universe == "global"].sort_values(["family", "n_observed"], ascending=[False, False]).annotator)
    fam_order = {"LIT": 0, "MOTIF": 1, "COEXP": 2}
    order = sorted(order, key=lambda a: (fam_order.get(first_row(sub, universe="global", annotator=a)["family"], 9),
                                         -first_row(sub, universe="global", annotator=a)["n_observed"]))
    body = []
    for a in order:
        g = first_row(sub, universe="global", annotator=a)
        cells = [tex(ANN_LABEL.get(a, a)), tex(g["family"]), t.v(g, "n_tfs_covered", "count"), t.v(g, "n_observed", "count"),
                 t.v(g, "coverage", "pct"), t.v(g, "n_pos", "count"), t.v(g, "pos_rate", "pct")]
        for tis in TISSUES:
            r = first_row(sub, universe=tis, annotator=a)
            cells += [t.v(r, "coverage", "pct"), t.v(r, "pos_rate", "pct")]
        body.append(cells)
    tot = first_row(ev, universe="global", variant="prespecified_cc", annotator="__ALL__")
    if tot is not None:
        body.append(("midrule",))
        cells = ["Any annotator", "", t.v(tot, "n_tfs_covered", "count"), t.v(tot, "n_observed", "count"),
                 t.v(tot, "coverage", "pct"), t.v(tot, "n_pos", "count"), t.v(tot, "pos_rate", "pct")]
        for tis in TISSUES:
            r = first_row(ev, universe=tis, variant="prespecified_cc", annotator="__ALL__")
            cells += [t.v(r, "coverage", "pct"), t.v(r, "pos_rate", "pct")]
        body.append(cells)
    t.note("Observed (\\%): share of universe edges the annotator observes; reported (\\%): share of its "
           "observed edges it reports. Any annotator: edges observed by at least one annotator, and edges "
           "with at least one report.")
    header = [" & & " + mc(5, "Global universe") + " & " + " & ".join(mc(2, U(tis)) for tis in TISSUES),
              cmid("3-7") + "".join(cmid(f"{8 + 2 * i}-{9 + 2 * i}") for i in range(len(TISSUES))),
              "Annotator & Family & TFs & Observed & \\% & Reports & \\% rep. & obs.\\,\\% & rep.\\,\\% & "
              "obs.\\,\\% & rep.\\,\\% & obs.\\,\\% & rep.\\,\\%"]
    t.tex = t.render("@{}llrrrrr" + "rr" * len(TISSUES) + "@{}", header, body, tabcolsep=2.6)
    return t


def tabS_tune_status() -> Table:
    t = Table("tabS_tune_status", "tab:S-tune-status",
              "Perturbation positives and credibility by evidence status for every tuning configuration "
              "(pooled perturbation reference, global universe): observed over TF-matched expected positives "
              "and mean credibility of reported, observed but unreported, and unobserved edges.")
    ts = t.read("tune_status_decomposition.csv", "all rows")
    ph = all(X.is_posthoc_row(r) for _, r in ts.iterrows())
    statuses = [("reported", "Reported"), ("observed_not_reported", "Observed, not reported"), ("unobserved", "Unobserved")]
    body = []
    for cfg in ts.config_id.drop_duplicates():
        cells = [tex(config_label(str(cfg)))]
        for st, _ in statuses:
            r = first_row(ts, config_id=cfg, status=st)
            cells += [t.v(r, "oe_tf_matched", "ratio"), t.v(r, "cred_mean", "sig2")]
        body.append(cells)
    r0 = first_row(ts, status="reported")
    if r0 is not None:
        t.note(f"Positive rate over all evaluable edges: {t.v(r0, 'pos_rate_all', 'ratio3')}. The status classes of "
               "the evidence variants differ because their observation rules differ.")
    if ph:
        t.caption += t.ph()
    header = ["Configuration & " + " & ".join(mc(2, lab) for _, lab in statuses),
              cmid("2-3") + cmid("4-5") + cmid("6-7"),
              " & O/E & $\\bar c$ & O/E & $\\bar c$ & O/E & $\\bar c$"]
    t.tex = t.render("@{}lcccccc@{}", header, body, tabcolsep=4)
    return t


def tabS_annotator_params() -> Table:
    t = Table("tabS_annotator_params", "tab:S-annotator-params",
              "Annotator parameters of Family-HB in the global universe (dominant-mode NUTS posterior "
              "means with 95\\% credible intervals): sensitivity $\\alpha_j$, false-positive rate "
              "$\\beta_j$, burst reporting rate $\\gamma_j$ and the marginal positive likelihood ratio "
              "(a report under T against the mixture of all non-T classes), ranked by the latter.")
    fp = t.read("fit_source_parameters.csv", "universe == global & fit_id == family_hb_nuts")
    g = fp[(fp.universe == "global") & (fp.fit_id == "family_hb_nuts")]
    lr = g[g.parameter == "lr_pos_marginal"].sort_values("rank")
    ph_lr = any(X.is_posthoc_row(r) for _, r in lr.iterrows())
    body = []
    for _, r in lr.iterrows():
        a = first_row(g, annotator=r["annotator"], parameter="alpha")
        b = first_row(g, annotator=r["annotator"], parameter="beta")
        c = first_row(g, annotator=r["annotator"], parameter="gamma")
        body.append([t.v(r, "rank", "count"), tex(ANN_LABEL.get(r["annotator"], r["annotator"])), tex(r["family"]),
                     t.v(r, "n_observed", "count"), t.v(r, "n_reported", "count"),
                     t.est_ci(a, "value", "lo", "hi", "sig2", stack=True),
                     t.est_ci(b, "value", "lo", "hi", "sig2", stack=True),
                     t.est_ci(c, "value", "lo", "hi", "sig2", stack=True) if c is not None else DASH,
                     t.est_ci(r, "value", "lo", "hi", "sig3", stack=True)])
    t.note("Observed: edges the annotator observes; reports: edges it reports. $\\gamma_j$ is the "
           "reporting rate under the burst class of the annotator's own family (no burst class for "
           "the single coexpression annotator, --).")
    ph_rank = any(X.is_posthoc_col(r, "rank") for _, r in lr.iterrows())
    header = ["Rank" + t.ph_if(ph_rank) + " & Annotator & Family & Observed & Reports & $\\alpha_j$ & $\\beta_j$ & $\\gamma_j$ & "
              "LR$^{+}$" + t.ph_if(ph_lr)]
    t.tex = render_split(t, "@{}rllrrcccc@{}", header, body, max_lines=44, tabcolsep=3)
    return t


def tabS_sens_fit() -> Table:
    t = Table("tabS_sens_fit", "tab:S-sens-fit")
    t.caption = ("Sensitivity of the Family-HB fit (posterior mode) to the evidence variant, the family "
                 "grouping and the priors, against the primary fit: class weight $w_{\\mathrm T}$, Spearman "
                 "correlation of credibilities over observed edges, size and Jaccard index of the posterior-mode "
                 f"BFDR\\,$\\le$\\,{q_pre()} set, and the posterior-mode structure (distinct optima among the "
                 "restarts, gap to the second optimum and its Laplace probability).")
    fs = t.read("fit_sensitivity.csv", "reference == family_hb, Family-HB sensitivity fits")
    fm = t.read("fit_sensitivity_modes.csv", "all rows")
    ph_modes = any(X.is_posthoc_row(r) for _, r in fm.iterrows())
    body = []
    underflow = False
    for u in ["global", "immune"]:
        prim = first_row(fs, universe=u, reference="family_hb")
        body.append(("panel", f"{U(u)} universe (primary: $w_{{\\mathrm T}}$ = "
                              f"{t.v(prim, 'w_T_ref', 'sig2')}, BFDR set {t.v(prim, 'bfdr_strong_n_ref', 'count')} edges)"))
        last = None
        for grp, var, fit in SENS_ORDER:
            r = first_row(fs, universe=u, variant=var, fit_id=fit, reference="family_hb")
            if r is None:
                continue
            if last is not None and grp != last:
                body.append(("addlinespace",))
            last = grp
            m = first_row(fm, universe=u, variant=var, fit_id=fit)
            exch = S.as_bool(r["t_burst_exchangeable"])
            lab = sens_label(var, fit) + ("$^{a}$" if exch else "")
            p2 = DASH
            if m is not None:
                if X._isnum(m["laplace_p2_top2"]) and float(m["laplace_p2_top2"]) == 0.0 and X._isnum(m["gap"]):
                    p2 = "$<10^{-300}$\\,$^{u}$"
                    underflow = True
                else:
                    p2 = t.v(m, "laplace_p2_top2", "sig2")
            body.append([lab, t.v(r, "w_T", "sig2"), t.v(r, "spearman_observed", "ratio3"),
                         t.v(r, "bfdr_strong_n_alt", "count"), t.v(r, "jaccard_bfdr_strong", "ratio"),
                         t.v(m, "n_optima", "count") if m is not None else DASH,
                         t.v(m, "gap", "num1") if m is not None else DASH, p2])
    t.note("$^{a}$\\,With one modality family the true and literature-burst classes are exchangeable; "
           "the labelling is not identified. Gap: log-posterior units; -- where all restarts reached "
           "one optimum. BFDR set: posterior-mode set (the Strong tier of the governance tables uses NUTS "
           "credibility).")
    if underflow:
        t.note("$^{u}$\\,Below the smallest double-precision number (the Laplace mass underflows at gaps of "
               "thousands of nats).")
    header = ["Fit & $w_{\\mathrm T}$ & Spearman & BFDR set $n$ & Jaccard & Optima & Gap" + t.ph_if(ph_modes)
              + " & $P$(2nd)" + t.ph_if(ph_modes)]
    t.tex = render_split(t, "@{}lccrcrrc@{}", header, body, max_lines=46, tabcolsep=4)
    return t


def tabS_sens_heldout() -> Table:
    t = Table("tabS_sens_heldout", "tab:S-sens-heldout",
              "Held-out AP lift of the sensitivity fits and its paired difference to the primary "
              "Family-HB (MAP) fit (95\\% TF-cluster bootstrap interval), all evaluable edges.")
    hs = t.read("heldout_sensitivity_fits.csv", "eval_set == all_evaluable")
    refs = ["encode", "enets2", "pert_pooled"]
    body = []
    he = t.read("heldout_estimators.csv", "eval_subset == all_evaluable & score == family_hb (primary AP lift, as in "
                                          "the main held-out table)")
    extra = [("primary_variant_estimator", "prespecified_cc", "family_hb_nuts"),
             ("nuts_initialisation", "prespecified_cc", "family_hb_randinit_nuts")]
    prior_nuts = [("prior", "prespecified_cc", "family_hb__prior_scale0.5_nuts"),
                  ("prior", "prespecified_cc", "family_hb__prior_scale2_nuts")]
    mix = []
    for u in ["global", "immune"]:
        body.append(("panel", f"{U(u)} universe"))
        prim = [first_row(he, universe=u, reference=ref, eval_subset="all_evaluable", score="family_hb") for ref in refs]
        body.append(["Family-HB (MAP), primary fit: AP lift"] + [t.est_ci(p, "ap_lift", "ap_lift_lo", "ap_lift_hi", "ratio", stack=True)
                                                       if p is not None else DASH for p in prim])
        last = None
        for grp, var, fit in extra + SENS_ORDER + prior_nuts:
            cells = []
            any_row = False
            for ref in refs:
                r = first_row(hs, universe=u, reference=ref, eval_set="all_evaluable", variant=var, fit_id=fit)
                any_row = any_row or r is not None
                cells.append(t.est_ci(r, "diff_vs_primary", "diff_vs_primary_lo", "diff_vs_primary_hi",
                                      "ratio3", stack=True) if r is not None else DASH)
            if not any_row:
                continue
            if last is not None and grp != last:
                body.append(("addlinespace",))
            last = grp
            rr = [first_row(hs, universe=u, reference=ref, eval_set="all_evaluable", variant=var, fit_id=fit)
                  for ref in refs]
            phr = any(X.is_posthoc_row(r) for r in rr)
            mm = next((r for r in rr if r is not None and S.as_bool(r.get("mode_mixing"))), None)
            lab = sens_label(var, fit) + t.ph_if(phr)
            if mm is not None:
                lab += "$^{m}$"
                mix.append(mm)
            body.append([lab] + cells)
    t.note("First row: AP lift of the primary Family-HB (MAP) fit with its 95\\% TF-cluster bootstrap interval, as in the main "
           "held-out table. Other rows: AP lift of the fit minus AP lift of the primary fit on the same evaluable "
           "edges (paired TF-cluster bootstrap of its own).")
    if mix:
        lo = min(float(r["fit_r_hat_max"]) for r in mix)
        hi = max(float(r["fit_r_hat_max"]) for r in mix)
        src = [(r, "fit_r_hat_max") for r in mix]
        t.note(f"$^{{m}}$\\,NUTS from random starts whose chains mix posterior modes (maximum $\\widehat R$ "
               f"{t.d(lo, 'rhat', src)}--{t.d(hi, 'rhat', src)}): the difference reflects mode mixing, not the prior or "
               "the initialisation.")
    header = ["Fit & " + " & ".join(RL(r) for r in refs)]
    t.tex = render_split(t, "@{}lccc@{}", header, body, max_lines=44)
    return t


def tabS_cv() -> Table:
    t = Table("tabS_cv", "tab:S-cv")
    cv = t.read("fit_cv_loglik.csv", "all rows")
    nf = t.v(cv.iloc[0], "n_folds", "count") if len(cv) else DASH
    t.caption = (f"Label-free model comparison: {nf}-fold cross-validated held-out log-likelihood per "
                 "observed edge (folds formed by TF), and the difference to Family-HB per edge and in "
                 "total (nats).")
    body = []
    for u in ["global", "immune"]:
        body.append(("panel", f"{U(u)} universe"))
        sub = cv[cv.universe == u].sort_values("heldout_ll_per_edge_mean", ascending=False)
        for _, r in sub.iterrows():
            lab = SL(r["model"]) if r["config"] == r["model"] else tex(FIT_LABEL.get(r["config"], r["config"]))
            body.append([lab, f"{t.v(r, 'heldout_ll_per_edge_mean', 'num4')} ({t.v(r, 'heldout_ll_per_edge_se', 'num4')})",
                         f"{t.v(r, 'diff_vs_family_hb_per_edge_mean', 'sig2')} ({t.v(r, 'diff_vs_family_hb_per_edge_se', 'sig2')})",
                         t.v(r, "diff_vs_family_hb_total_nats", "num1"),
                         f"{t.v(r, 'n_folds_better_than_family_hb', 'count')}/{t.v(r, 'n_folds', 'count')}"])
    t.note("Standard errors in parentheses (over folds). Folds better: folds in which the configuration "
           "has a higher held-out log-likelihood than Family-HB.")
    header = ["Configuration & Held-out log-lik.\\ per edge (SE) & Difference per edge (SE) & Total (nats) & Folds better"]
    t.tex = t.render("@{}lcccc@{}", header, body, tabcolsep=4)
    return t


PPC_STATS = [("n_reporting_frac", "count"), ("anyreport_frac_by_n_observing", "any report"),
             ("positive_rate", "rates"), ("conditional_positive_count", "cond.\\ count"),
             ("lor_pair_within", "within"), ("lor_pair_between", "between")]


def tabS_ppc() -> Table:
    t = Table("tabS_ppc", "tab:S-ppc",
              "Posterior predictive checks: largest absolute $z$-score (observed minus replicated mean, "
              "divided by the replicated standard deviation) over the elements of each statistic, and the "
              "replicated share of edges with a literature report among the edges observed by the most "
              "literature annotators (top stratum), against the observed share.")
    pp = t.read("fit_ppc.csv", "kind == summary_abs_z; statistic == litblock_anyreport_frac_by_n_observing & "
                               "kind == statistic & stratum == top")
    s = pp[pp.kind == "summary_abs_z"]
    order = LCM5 + ["family_hb__fam_singleton", "family_hb__fam_lineage", "family_hb__fam_datadriven"]
    body = []
    r0 = first_row(s, statistic="n_reporting_frac")
    nrep = first_row(pp, kind="statistic")
    lb = pp[(pp.statistic == "litblock_anyreport_frac_by_n_observing") & (pp.kind == "statistic")]
    top = str(lb.stratum.iloc[-1]) if len(lb) else None
    for u in ["global", "immune"]:
        o = first_row(lb, universe=u, stratum=top) if top else None
        obs = f"; top stratum ({tex(top.replace('>=', '≥'))} literature annotators) observed {t.v(o, 'observed', 'sig2')}" if o is not None else ""
        body.append(("panel", f"{U(u)} universe{obs}"))
        for e in order:
            cells = [SL(e) if e in LCM5 else tex(FIT_LABEL.get(e, e))]
            for stat, _ in PPC_STATS:
                r = first_row(s, universe=u, estimator=e, statistic=stat)
                cells.append(t.v(r, "max_abs_z", "num1"))
            rt = first_row(lb, universe=u, estimator=e, stratum=top) if top else None
            cells.append(t.v(rt, "rep_mean", "sig2"))
            if all(c == DASH for c in cells[1:]):
                continue
            body.append(cells)
    t.note("Reporting count: share of edges with $k$ reporting annotators; any report by coverage: share "
           "of edges with at least one report by number of observing annotators; reporting rates and "
           "conditional count: per annotator; co-reporting: pairwise log odds ratios of reports within "
           "and between families."
           + (f" Replicates per fit: {t.v(nrep, 'n_rep', 'count')}; the $|z|$ values carry Monte Carlo error from "
              "these few replicates (fits with nearly identical parameters can differ markedly), so the replicated "
              "rates are the better summary." if nrep is not None else ""))
    header = [" & " + mc(2, "Reports per edge") + " & " + mc(2, "Per annotator") + " & " + mc(2, "Co-reporting log OR")
              + " & Top stratum",
              cmid("2-3") + cmid("4-5") + cmid("6-7") + cmid("8-8"),
              "Estimator & " + " & ".join(lab for _, lab in PPC_STATS) + " & replicated"]
    t.tex = t.render("@{}l" + "r" * (len(PPC_STATS) + 1) + "@{}", header, body, tabcolsep=4)
    return t


REC_SIM = {"faithful_cc": "Design-faithful (primary)",
           "faithful_cc_strat": "Stratum-specific weights (sensitivity)",
           "fixed_masks": "Fixed observation masks (sensitivity)"}


def tabS_recovery() -> Table:
    t = Table("tabS_recovery", "tab:S-recovery",
              "Simulation-based recovery at the fitted Family-HB parameters (data simulated from the fitted model "
              "and refitted; design-faithful simulator primary, two simulators of the observation design as "
              "sensitivity analyses): parameter recovery (mean over annotators), ranking and calibration "
              "of credibility, and the realised false discovery rate of the Strong set; Family-HB also by the "
              "posterior mode its multi-start MAP selected (truth start: Family-HB started at the true parameters).")
    rc = t.read("fit_recovery.csv", "simulator in {faithful_cc, faithful_cc_strat, fixed_masks}; kinds parameter, "
                                    "parameter_by_mode, credibility, realised_fdr, realised_fdr_by_mode, identifiability; "
                                    "family == all")
    body = []
    ph_q: set = set()
    ph_rows = False
    sims = [s_ for s_ in ["faithful_cc", "faithful_cc_strat", "fixed_masks"] if s_ in set(rc.simulator.dropna())]
    for sim in sims:
        for u in ["global", "immune"]:
            sub = rc[(rc.universe == u) & (rc.simulator == sim)]
            if not len(sub):
                continue
            idr = first_row(sub, kind="identifiability", estimator="family_hb", quantity="n_replicates_map_in_competing_mode")
            nrep = t.v(idr, "n_replicates", "count") if idr is not None else DASH
            comp = t.v(idr, "value_mean", "count") if idr is not None else DASH
            # a simulator panel whose rows are all post hoc carries one dagger in its header (verifier assets_fix M1)
            panel_ph = bool(len(sub)) and all(X.is_posthoc_row(r) for _, r in sub.iterrows())
            body.append(("panel", f"{REC_SIM.get(sim, tex(sim))}, {U(u).lower()} universe: {nrep} replicates, "
                                  f"competing mode in {comp}{t.ph_if(panel_ph)}"))

            def q(kind, est, quantity, col="value_mean", kindfmt="ratio3", mode=None):
                cond = dict(kind=kind, estimator=est, family="all", quantity=quantity)
                if mode is not None:
                    cond["mode"] = mode
                r = first_row(sub, **cond)
                if r is None and kind == "parameter" and mode is None:
                    r = first_row(sub, kind="parameter_by_mode", estimator=est, family="all", quantity=quantity,
                                  mode="all_modes")
                if r is not None and X.is_posthoc_row(r) and not panel_ph:
                    ph_q.add(quantity)
                return t.v(r, col, kindfmt) if r is not None else DASH

            for est, lab in (("oracle", SL("oracle")), ("family_hb_truthstart", "Family-HB, truth start"),
                             ("family_hb", "Family-HB (MAP)"), ("em", SL("em"))):
                if not len(sub[sub.estimator == est]):
                    continue
                ph = (not panel_ph and est == "family_hb_truthstart"
                      and any(X.is_posthoc_row(r) for _, r in sub[sub.estimator == est].iterrows()))
                row = [lab + t.ph_if(ph),
                             q("parameter", est, "w_T", "bias", "sig2"), q("parameter", est, "alpha", "rmse", "ratio3"),
                             q("parameter", est, "beta", "rmse", "sig2"),
                             q("credibility", est, "auroc"), q("credibility", est, "spearman_oracle"),
                             q("credibility", est, "ece_bins1_9_pooled"),
                             q("realised_fdr", est, "strong_claimed_bfdr"),
                             q("realised_fdr", est, "strong_realised_fdr"),
                             q("realised_fdr", est, "strong_n", kindfmt="count")]
                if all(c == DASH for c in row[1:]):
                    continue
                ph_rows |= ph
                body.append(row)
                if est == "family_hb":
                    for mode, mlab in (("truth", "truth mode"), ("competing", "competing mode")):
                        r = first_row(sub, kind="realised_fdr_by_mode", estimator="family_hb", quantity="strong_realised_fdr",
                                      mode=mode)
                        if r is None:
                            continue
                        body.append([f"\\quad {mlab} ($n$ = {t.v(r, 'n_replicates', 'count')})",
                                     q("parameter_by_mode", est, "w_T", "bias", "sig2", mode),
                                     q("parameter_by_mode", est, "alpha", "rmse", "ratio3", mode),
                                     q("parameter_by_mode", est, "beta", "rmse", "sig2", mode), DASH, DASH, DASH, DASH,
                                     t.v(r, "value_mean", "ratio3"),
                                     q("realised_fdr_by_mode", est, "strong_n", kindfmt="count", mode=mode)])
                        if X.is_posthoc_row(r) and not panel_ph:
                            ph_q.add("by_mode")
    t.note("Bias of $w_{\\mathrm T}$ and root-mean-square error (RMSE) of the annotator rates "
           "$\\alpha$ and $\\beta$ on the probability scale (oracle: true parameters, --). ECE: "
           "expected calibration error over credibility bins 0.1--0.9, pooled over replicates. Strong set: "
           f"BFDR\\,$\\le$\\,{q_pre()}; claimed: its posterior expected FDR; realised: share of simulated non-edges in it. "
           "Competing mode: replicates whose selected optimum is not the truth-aligned one; the by-mode rows have few "
           "replicates. " + sim_note(t))
    d_ece = t.ph_if("ece_bins1_9_pooled" in ph_q)
    d_set = t.ph_if(bool(ph_q & {"strong_claimed_bfdr", "strong_realised_fdr", "strong_n"}))
    header = [" & $w_{\\mathrm T}$ & $\\alpha$ & $\\beta$ & " + mc(3, "Credibility") + " & " + mc(3, "Strong set" + d_set),
              cmid("5-7") + cmid("8-10"),
              "Estimator & bias & RMSE & RMSE & AUROC & Spearman & ECE" + d_ece + " & claimed & realised & $n$"]
    t.tex = render_split(t, "@{}lccccccccr@{}", header, body, max_lines=46, tabcolsep=2.4)
    return t


# (quantity, label, format, interval kind: 'w' Wilson 95%, 'q' 10th-90th percentile over replicates, None)
MODE_ROWS = [("p_dominant_is_truth", "P(dominant optimum = truth)", "ratio", "w"),
             ("p_truth_mode_found", "P(truth optimum found)", "ratio", "w"),
             ("gap_median", "Median gap between the two optima (nats)", "num1", "q"),
             ("laplace_p2_mean", "Mean Laplace probability of the other optimum", "sig2", None),
             ("dominant_strong_realised_fdr_mean", "Strong realised FDR, dominant optimum", "ratio3", "q"),
             ("dominant_strong_realised_fdr_gt_0.10", "P(realised FDR $>$ 0.10), dominant", "ratio", None),
             ("truthstart_strong_realised_fdr_mean", "Strong realised FDR, truth-start", "ratio3", "q"),
             ("second_strong_realised_fdr_mean", "Strong realised FDR, other optimum", "ratio3", "q"),
             ("mixture_laplace_strong_realised_fdr_mean", "Strong realised FDR, Laplace mixture", "ratio3", "q"),
             ("oracle_strong_realised_fdr_mean", "Strong realised FDR, oracle", "ratio3", "q"),
             ("em_strong_realised_fdr_mean", "Strong realised FDR, EM", "ratio3", "q")]
MODE_ROWS_SENS = ["p_dominant_is_truth", "gap_median", "dominant_strong_realised_fdr_mean",
                  "second_strong_realised_fdr_mean", "oracle_strong_realised_fdr_mean"]
SIM_PANEL = {"faithful_cc": "Design-faithful simulator (primary)",
             "faithful_cc_strat": "Design-faithful, stratum-specific class weights (sensitivity)",
             "fixed_masks": "Fixed observation masks (sensitivity)"}


def sim_note(t: Table) -> str:
    """Description of the three simulators of the observation design, with the share of the real
    motif/coexpression-observed edges that the design-faithful simulator reproduces (modes_faithful_design)."""
    fd = t.read("modes_faithful_design.csv", "quantity == n_edges_cc_observed & sim_label in {immune_cc, global_cc}: "
                                             "ratio_sim_mean_to_real")
    parts = []
    for lab, u in (("immune_cc", "immune"), ("global_cc", "global")):
        r = first_row(fd, sim_label=lab, quantity="n_edges_cc_observed")
        if r is not None and X._isnum(r["ratio_sim_mean_to_real"]):
            parts.append(f"{t.v(r, 'ratio_sim_mean_to_real', 'pct0')}\\% ({u})")
    share = (" Its data contain " + " and ".join(parts) + " of the real number of motif- or coexpression-observed "
             "edges, i.e. fewer informative observations than the real data.") if parts else ""
    return ("Design-faithful: real literature observation masks, motif and coexpression masks rebuilt from the "
            "simulated literature reports as in the evidence builder." + share + " Stratum-specific: the same with "
            "class weights that depend on the literature-mask stratum, which reproduces the real literature report "
            "counts. Fixed masks: every observation mask fixed at the real one, so that motif and coexpression "
            "observations do not depend on the simulated literature reports.")


def tabS_modes_sim() -> Table:
    t = Table("tabS_modes_sim", "tab:S-modes-sim",
              "Mode-selection simulation: data simulated from the dominant Family-HB optimum (truth = dominant mode) "
              "or, in the immune universe, from the second optimum (truth = second mode), refitted by multi-start MAP; "
              "how often the highest-posterior optimum is the truth, how the gap between the two optima compares with "
              "the real gaps, and what the Strong set of each optimum realises. Rows by simulator of the observation "
              "design.")
    ms = t.read("modes_simulation.csv", "simulator in {faithful_cc, faithful_cc_strat, fixed_masks}; stratum in {all, "
                                        "real_gap_<u>, gap_ge_real_<u>, logistic_at_real_gap_<u>}")
    unis = [u for u in ["global", "immune", "immune_truthmode1"] if u in set(ms.universe)]
    sims = [s for s in ["faithful_cc", "faithful_cc_strat", "fixed_masks"] if s in set(ms.simulator)]
    ph = all(X.is_posthoc_row(r) for _, r in ms.iterrows())
    head_lab = {"global": "Global", "immune": "Immune", "immune_truthmode1": "Immune, truth ="}
    head_lab2 = {"global": "", "immune": "", "immune_truthmode1": "second mode"}
    real = {}
    for u in UNIVERSES:
        r = first_row(ms, stratum=f"real_gap_{u}", quantity="frac_sim_gap_below_real_gap_all")
        if r is not None:
            real[u] = r

    def cell(r, kind, ci):
        if r is None:
            return DASH
        if ci and X._isnum(r["ci_lo"]):
            return t.est_ci(r, "value", "ci_lo", "ci_hi", kind, stack=True)
        return t.v(r, "value", kind)

    # where the real gap falls among the simulated gaps: the real gap of the universe whose design was simulated
    gap_ref = {"global": "global", "immune": "immune", "immune_truthmode1": "immune"}
    rep_ = t.read("modes_simulation_replicates.csv", "simulator == faithful_cc: replicates whose Laplace-mixture Strong "
                                                    "set is empty (mixture_laplace_strong_n == 0) and dominant_is_truth")
    body, used_e, mix_n = [], False, False
    for sim in sims:
        body.append(("panel", SIM_PANEL.get(sim, tex(sim))))
        sub = ms[ms.simulator == sim]
        nr = ["Replicates"]
        n_all = {}
        for u in unis:
            r0 = first_row(sub, universe=u, stratum="all", quantity="p_dominant_is_truth")
            n_all[u] = float(r0["n_replicates"]) if r0 is not None else None
            nr.append(t.v(r0, "n_replicates", "count"))
        body.append(nr)
        rows_ = MODE_ROWS if sim == "faithful_cc" else [m for m in MODE_ROWS if m[0] in MODE_ROWS_SENS]
        for q, lab, kind, ci in rows_:
            cells_ = []
            for u in unis:
                r = first_row(sub, universe=u, stratum="all", quantity=q)
                c = cell(r, kind, ci)
                # rows computed on a subset of the replicates print their n (Laplace mixture: empty sets omitted)
                if r is not None and n_all[u] is not None and X._isnum(r["n_replicates"]) \
                        and float(r["n_replicates"]) != n_all[u] and c != DASH:
                    n_txt = f"$n$ = {t.v(r, 'n_replicates', 'count')}"
                    c = (c.replace("}\\end{tabular}", " " + n_txt + "}\\end{tabular}", 1) if "\\end{tabular}" in c
                         else stacked(c, n_txt))
                    mix_n |= q.startswith("mixture_laplace")
                cells_.append(c)
            body.append([lab] + cells_)
            if q == "gap_median":
                gl = []
                for u in unis:
                    r = first_row(sub, universe=u, stratum=f"real_gap_{gap_ref.get(u, u)}",
                                  quantity="frac_sim_gap_below_real_gap_all")
                    gl.append(t.v(r, "value", "ratio") if r is not None else DASH)
                body.append(["Share of simulated gaps below the real gap$^{g}$"] + gl)
        # comparison with the real immune gap (both directions)
        ri = real.get("immune")
        if ri is not None:
            g = t.v(ri, "real_gap", "num1")
            for q, lab in (("p_truth_beats_other_mode_by_ge_real_gap", f"P(truth ahead by $\\geq$ {g} nats, immune)"),
                           ("p_other_mode_beats_truth_by_ge_real_gap", f"P(other ahead by $\\geq$ {g} nats, immune)")):
                body.append([lab] + [cell(first_row(sub, universe=u, stratum="real_gap_immune", quantity=q), "ratio", "w")
                                     for u in unis])
        # gap-conditional correctness and the logistic fit at each real gap
        for ru in [u for u in UNIVERSES if u in real]:
            g = t.v(real[ru], "real_gap", "num1")
            cells = [f"P(dominant = truth $\\mid$ gap $\\geq$ {g}, {U(ru).lower()})"]
            for u in unis:
                r = first_row(sub, universe=u, stratum=f"gap_ge_real_{ru}", quantity="p_dominant_is_truth")
                if r is None or not X._isnum(r["value"]):
                    cells.append(DASH)
                    continue
                cells.append(stacked(t.v(r, "value", "ratio"), f"{t.ci(r, 'ci_lo', 'ci_hi', 'ratio')} $n$ = "
                                                               f"{t.v(r, 'n_replicates', 'count')}"))
            lr = [first_row(sub, universe=u, stratum=f"logistic_at_real_gap_{ru}", quantity="p_dominant_is_truth")
                  for u in unis]
            lr = [r if (r is not None and X._isnum(r["value"])) else None for r in lr]
            if any(c != DASH for c in cells[1:]):
                body.append(cells)
            # a logistic row is printed only if at least one cell is not extrapolated (cells are registered only then)
            if any(r is not None and not S.as_bool(r["extrapolated"]) for r in lr):
                lg = []
                for r in lr:
                    if r is None:
                        lg.append(DASH)
                    elif S.as_bool(r["extrapolated"]):
                        lg.append(f"({t.v(r, 'value', 'ratio')})$^{{e}}$")
                        used_e = True
                    else:
                        lg.append(cell(r, "ratio", "w"))
                body.append([f"Logistic P(dominant = truth) at {g}, {U(ru).lower()}"] + lg)
    t.note(sim_note(t))
    gtxt = [f"{t.v(real[u], 'real_gap', 'num1')} nats ({U(u).lower()})" for u in ("global", "immune") if u in real]
    if gtxt:
        t.note("$^{g}$\\,Real gap of the simulated universe: " + ", ".join(gtxt) + "; the immune columns use the "
               "immune gap.")
    if mix_n:
        mt = []
        for lab, col in (("immune_cc", "immune"), ("immune_truthmode1_cc", "truth = second mode")):
            g = rep_[rep_.sim_label == lab]
            if not len(g):
                continue
            emp = g["mixture_laplace_strong_n"].astype(float) == 0
            wrong = ~g["dominant_is_truth"].map(S.as_bool)
            srcs = [(r_, "mixture_laplace_strong_n") for _, r_ in g.iterrows()]
            srcw = srcs + [(r_, "dominant_is_truth") for _, r_ in g.iterrows()]
            fdr = g.loc[~emp, "dominant_strong_realised_fdr"].astype(float)
            srcf = [(r_, "dominant_strong_realised_fdr") for _, r_ in g[~emp].iterrows()] + srcs
            mt.append(f"{col}, {t.d(float(emp.sum()), 'count', srcs)} of {t.d(float(len(g)), 'count', srcs)} "
                      f"omitted, among them {t.d(float((emp & wrong).sum()), 'count', srcw)} of the "
                      f"{t.d(float(wrong.sum()), 'count', srcw)} replicates in which the wrong optimum is dominant; "
                      f"dominant optimum on the replicates used {t.d(float(fdr.mean()), 'ratio3', srcf)}")
        t.note("Laplace mixture: replicates whose mixture Strong set is empty are omitted ($n$: replicates used; "
               + "; ".join(mt) + "): the mixture abstains rather than errs in the omitted replicates, so its realised "
               "FDR is not comparable with the dominant-optimum row over all replicates.")
    t.note("Intervals: probabilities, Wilson 95\\% interval (logistic rows: Wald 95\\%); medians and means, "
           "10th--90th percentile over replicates. Gaps: real log-posterior gaps of the fitted universes "
           "(global, immune, kidney, lung). Truth-start: fit started at the true parameters; other optimum: the "
           "next-best optimum of the same fit; Laplace mixture: credibility averaged over the optima with Laplace "
           "weights.")
    if used_e:
        t.note("$^{e}$\\,Extrapolated (flagged in the source table): too few replicates reach that gap, so the logistic "
               "curve is evaluated beyond the simulated gaps (no interval); rows in which every cell is extrapolated "
               "are omitted.")
    if ph:
        t.caption += t.ph()
    header = [" & " + " & ".join(head_lab[u] for u in unis), " & " + " & ".join(head_lab2[u] for u in unis)]
    t.tex = render_split(t, "@{}l" + "c" * len(unis) + "@{}", header, body, max_lines=44, at_panels_only=True)
    return t


def tabS_modes_gov() -> Table:
    t = Table("tabS_modes_gov", "tab:S-modes-gov",
              "Governance under a two-mode mixture posterior: credibility averaged over the dominant and "
              "the second Family-HB mode with second-mode weight $p_2$ (bridge-sampling estimate, its upper "
              "Monte-Carlo bound, the Laplace estimate, or fixed values), tier sizes, Jaccard index with the "
              "primary sets, and the posterior expected FDR of the primary dual Strong and promoted sets "
              "under the mixture.")
    mg = t.read("modes_governance_sensitivity.csv", "all rows")
    ph = all(X.is_posthoc_row(r) for _, r in mg.iterrows())
    lab = {"primary": "dominant mode (primary)", "mode0_rerun_mc_reference": "dominant mode, re-run",
           "bridge": "bridge $p_2$", "bridge_upper_mc": "bridge, upper bound",
           "laplace": "Laplace $p_2$", "mode1_only": "second mode only"}
    body = []
    for tis in TISSUES:
        sub = mg[mg.tissue == tis]
        if not len(sub):
            continue
        body.append(("panel", U(tis)))
        for _, r in sub.iterrows():
            sc = str(r["scenario"])
            name = lab.get(sc, "fixed $p_2$") + (", both fits" if r["global_rule"] == "same_p2" else "")
            body.append([name, t.v(r, "p2_context", "prob"), t.v(r, "p2_global", "prob"), t.v(r, "context_strong", "count"),
                         t.v(r, "dual_strong", "count"), t.v(r, "promoted", "count"),
                         t.v(r, "jaccard_dual_strong", "ratio"), t.v(r, "jaccard_promoted", "ratio"),
                         t.v(r, "primary_dual_strong_bfdr_under_scenario_context", "ratio3"),
                         t.v(r, "primary_promoted_bfdr_under_scenario_context", "ratio3")])
    t.note("$p_2$: weight of the second mode in the context and the global fit (the global fit keeps its own "
           "estimate unless marked ``both fits''). Ctx.\\ Strong: single-fit Strong set of the context fit; "
           "Strong and prom.: dual Strong and promoted sets. "
           "FDR: posterior expected FDR of the primary sets under the mixture credibility of the context fit.")
    if ph:
        t.caption += t.ph()
    header = [" & " + mc(2, "$p_2$") + " & " + mc(3, "Tier size (dual)") + " & " + mc(2, "Jaccard") + " & "
              + mc(2, "FDR of primary"),
              cmid("2-3") + cmid("4-6") + cmid("7-8") + cmid("9-10"),
              "Scenario & ctx. & glob. & ctx.\\ Strong & Strong & prom. & Strong & prom. & Strong & prom."]
    t.tex = render_split(t, "@{}lccrrrcccc@{}", header, body, max_lines=46, tabcolsep=3)
    return t


GOV_FAMILY_LABEL = {"q_grid": "Threshold grid", "dual_rule": "Dual-universe rule", "single_universe": "Single fit",
                    "rank_budget": "Rank budget", "credibility": "Credibility", "estimator": "Estimator",
                    "baseline_rule": "Simple rules", "size_matched": "Size-matched to primary"}


def _gov_cfg_label(r, t: Table | None = None) -> str:
    fam, cfg = r["family"], str(r["config"])
    if fam == "q_grid":
        return f"$q_M$ = {t.v(r, 'q_moderate', 'ratio') if t is not None else X.fmt(r['q_moderate'], 'ratio')}"
    lab = {"primary": "primary", "lenient": "lenient (+1 tier)", "strict": "strict ($-$1 tier)",
           "global_fit_only": "global fit only", "global_fit_only_strong": "global fit, Strong only",
           "context_fit_only": "context fit only", "context_fit_only_strong": "context fit, Strong only",
           "nuts_mean_primary": "NUTS mean (primary)", "map": "posterior mode", "em_map": "EM baseline",
           "majority_vote_ge2": "majority ($\\geq$2 reports)", "union_ge1": "union ($\\geq$1 report)",
           "dorothea_heldin_AB": "DoRothEA A/B", "collectri_heldin": "CollecTRI member"}.get(cfg)
    if lab:
        return lab
    if fam == "rank_budget":
        return "rank budget"
    if fam == "size_matched":
        base = cfg.rsplit("_top", 1)[0]
        return {"joint_nuts": "joint, NUTS", "joint_map": "joint, mode", "joint_em": "joint, EM",
                "global_fit_nuts": "global fit, NUTS", "context_fit_nuts": "context fit, NUTS",
                "n_reporting": "number of reports", "dorothea_heldin_level": "DoRothEA level",
                "collectri_heldin_effort": "CollecTRI effort"}.get(base, tex(base))
    return tex(cfg)


def _tab_gov_grid(name: str, ref: str) -> Table:
    t = Table(name, f"tab:S-{name.split('_', 1)[1].replace('_', '-')}",
              f"Sensitivity of the promoted set to the governance rules: size, Jaccard index with the primary "
              f"promoted set, and TF-matched held-out O/E ({RL(ref)}, 95\\% TF-cluster bootstrap interval).")
    gg = t.read("gov_sensitivity_grid.csv", f"reference == {ref}; q_grid rows with q_strong == primary q_strong")
    sub = gg[gg.reference == ref]
    q_s = first_row(sub, family="dual_rule", config="primary")
    qs = q_s["q_strong"] if q_s is not None else None
    keys = []
    for _, r in sub[sub.tissue == "immune"].iterrows():
        if r["family"] == "q_grid" and qs is not None and r["q_strong"] != qs:
            continue
        keys.append((r["family"], r["config"]))
    body = []
    last = None
    for fam, cfg in keys:
        if fam != last:
            body.append(("panel", GOV_FAMILY_LABEL.get(fam, tex(fam))))
            last = fam
        r0 = first_row(sub, tissue="immune", family=fam, config=cfg)
        cells = [_gov_cfg_label(r0, t) + t.ph_if(X.is_posthoc_row(r0))]
        for tis in TISSUES:
            r = first_row(sub, tissue=tis, family=fam, config=cfg)
            oe = t.est_ci(r, "oe_tf", "oe_tf_lo", "oe_tf_hi", "ratio", stack=True) if r is not None else DASH
            if r is not None and S.as_bool(r["few_clusters"]):
                oe = oe.replace("\\\\[-1pt]", "$^{*}$\\\\[-1pt]", 1)
            cells += [t.v(r, "n_promoted", "count"), t.v(r, "jaccard_vs_primary", "ratio"), oe]
        body.append(cells)
    if qs is not None:
        t.note(f"Threshold grid: moderate threshold $q_M$ at the primary strong threshold $q_S$ = "
               f"{t.v(q_s, 'q_strong', 'ratio')} (the promoted set does not depend on $q_S$). "
               "$^{*}$\\,Few TF clusters.")
    header = [" & " + " & ".join(mc(3, U(tis)) for tis in TISSUES),
              cmid("2-4") + cmid("5-7") + cmid("8-10"),
              "Rule" + " & $n$ & Jaccard & O/E" * len(TISSUES)]
    t.tex = render_split(t, "@{}l" + "rcc" * len(TISSUES) + "@{}", header, body, max_lines=44, tabcolsep=3)
    return t


def tabS_gov_grid() -> Table:
    return _tab_gov_grid("tabS_gov_grid", "encode")


def tabS_gov_grid_pert() -> Table:
    return _tab_gov_grid("tabS_gov_grid_pert", "pert_pooled")


def tabS_deconc() -> Table:
    t = Table("tabS_deconc", "tab:S-deconc",
              "Deconcentration of the promoted set: the label-free chosen subset, an undeconcentrated "
              "subset of the same size (highest credibility), and the full promoted pool; TF "
              "concentration and TF-matched held-out O/E (95\\% TF-cluster bootstrap interval).")
    de = t.read("gov_deconcentration_eval.csv", "all rows")
    sets = ["chosen", "undeconcentrated_same_budget", "pool", "chosen_pielou", "undeconcentrated_same_budget_pielou"]
    lab = {"chosen": "chosen", "undeconcentrated_same_budget": "same size, by credibility", "pool": "promoted pool",
           "chosen_pielou": "chosen (Pielou criterion)", "undeconcentrated_same_budget_pielou": "same size (Pielou)"}
    body = []
    for tis in TISSUES:
        body.append(("panel", U(tis)))
        for s in sets:
            r = first_row(de, tissue=tis, set=s, reference="encode")
            if r is None:
                continue
            p = first_row(de, tissue=tis, set=s, reference="pert_pooled")
            e2 = first_row(de, tissue=tis, set=s, reference="enets2")
            body.append([tex(lab[s]) + t.ph_if(X.is_posthoc_row(r)), t.v(r, "budget", "count"), t.v(r, "alpha", "num0"),
                         t.v(r, "mean_cred", "ratio3"), t.v(r, "n_tfs", "count"), t.v(r, "top1_share", "ratio"),
                         t.v(r, "evenness_pielou", "ratio"),
                         t.est_ci(r, "oe_tf", "oe_tf_lo", "oe_tf_hi", "ratio", stack=True),
                         t.est_ci(e2, "oe_tf", "oe_tf_lo", "oe_tf_hi", "ratio", stack=True),
                         t.est_ci(p, "oe_tf", "oe_tf_lo", "oe_tf_hi", "ratio", stack=True)])
    t.note("$\\alpha$: TF penalty of the greedy selection (0: no deconcentration). Top TF: share of the "
           "set's edges belonging to its most frequent TF; evenness: Pielou's evenness of the TF counts.")
    header = ["Set & Size & $\\alpha$ & Mean $\\bar c$ & TFs & Top TF & Evenness & " + RL("encode") + " & "
              + RL("enets2") + " & Pert."]
    t.tex = render_split(t, "@{}lrrrrrrccc@{}", header, body, max_lines=44, tabcolsep=3)
    return t


def tabS_rep_prior() -> Table:
    t = Table("tabS_rep_prior", "tab:S-rep-prior",
              "Prior sensitivity of the pooled replication model (study level): pooled odds ratio "
              "$\\exp(\\mu)$ and odds ratio of a new study $\\exp(\\theta_{\\mathrm{new}})$ (posterior "
              "median, 95\\% CrI), between-study SD $\\sigma$, posterior probability that the pooled odds "
              "ratio reaches the endpoint, and each study's posterior probability of passing.")
    rp = t.read("rep_prior_sensitivity.csv", "level == study")
    s = rp[rp.level == "study"]
    units = [u for u in s[s.parameter == "theta"].unit.dropna().unique()]
    body = []
    for prof in s.profile.drop_duplicates():
        mu = first_row(s, profile=prof, parameter="mu")
        sg = first_row(s, profile=prof, parameter="sigma")
        tn = first_row(s, profile=prof, parameter="theta_new")
        if mu is None:
            continue
        prior = f"{tex(mu['prior_mu']).replace('Normal', 'N')}, {tex(mu['prior_sigma']).replace('HalfNormal', 'HN')}"
        cells = [stacked(tex(prof).replace("\\_", " ") + t.ph_if(X.is_posthoc_row(mu)), prior, align="l"),
                 t.est_ci(mu, "or_median", "or_q025", "or_q975", "ratio", stack=True), t.v(mu, "p_or_ge_endpoint", "ratio"),
                 t.est_ci(sg, "mean", "q025", "q975", "ratio", stack=True),
                 t.est_ci(tn, "or_median", "or_q025", "or_q975", "ratio", stack=True)]
        for un in units:
            th = first_row(s, profile=prof, parameter="theta", unit=un)
            cells.append(t.v(th, "p_or_ge_endpoint", "ratio"))
        body.append(cells)
    r_any = first_row(s, profile="primary", parameter="mu")
    t.note("N, HN: normal and half-normal priors of $\\mu$ and $\\sigma$. Pass probability: posterior "
           "probability that the study's odds ratio reaches the endpoint. tf random slope: adds a TF-level random "
           "slope." + (f" Max.\\ $\\widehat R$ {t.v(r_any, 'r_hat_max_all', 'rhat')} under the primary prior."
                                        if r_any is not None else ""))
    header = ["Profile & $\\exp(\\mu)$ & $P$(end- & $\\sigma$ & $\\exp(\\theta_{\\mathrm{new}})$ & "
              + mc(len(units), "Pass probability"),
              cmid(f"6-{5 + len(units)}"),
              "($\\mu$, $\\sigma$ priors) & & point) & & & " + " & ".join(tex(S.reference_label(u).split(" ")[0]) for u in units)]
    t.tex = t.render("@{}lcccc" + "c" * len(units) + "@{}", header, body, tabcolsep=3.5)
    return t


def tabS_rep_leavek() -> Table:
    t = Table("tabS_rep_leavek", "tab:S-rep-leavek",
              "Leave-$k$-out refits of the pooled replication model: changes in the pooled log odds ratio "
              "$\\mu$, in the posterior probability that the pooled odds ratio reaches the endpoint, and in "
              "the per-unit pass probabilities, over all subsets of $k$ units left out.")
    lk = t.read("rep_leavek.csv", "row_type == summary")
    s = lk[lk.row_type == "summary"]
    body = []
    for lvl in ["study", "cohort"]:
        sub = s[s.level == lvl]
        if not len(sub):
            continue
        r0 = sub.iloc[0]
        body.append(("panel", f"{tex(lvl.capitalize())} level (full data: $\\mu$ = {t.v(r0, 'mu_full', 'ratio')}, "
                              f"$P$ = {t.v(r0, 'p_mu_full', 'ratio')})"))
        for _, r in sub.sort_values("k").iterrows():
            body.append([tex(r["scheme"]), t.v(r, "n_fits", "count"), t.v(r, "mean_abs_delta_mu", "ratio"),
                         t.v(r, "max_abs_delta_mu", "ratio"), t.v(r, "mean_abs_delta_p_mu", "ratio"),
                         t.v(r, "max_abs_delta_p_mu", "ratio"), t.v(r, "mean_abs_delta_unit_pass_prob", "ratio"),
                         t.v(r, "max_abs_delta_unit_pass_prob", "ratio")])
    header = [" & & " + mc(2, "$|\\Delta\\mu|$") + " & " + mc(2, "$|\\Delta P|$") + " & " + mc(2, "$|\\Delta$ pass prob.$|$"),
              cmid("3-4") + cmid("5-6") + cmid("7-8"),
              "Scheme & Fits & mean & max & mean & max & mean & max"]
    t.tex = t.render("@{}lrrrrrrr@{}", header, body)
    return t


def tabS_rep_bayes() -> Table:
    t = Table("tabS_rep_bayes", "tab:S-rep-bayes",
              "Pooled replication model under the primary prior (study level): per-study odds ratio "
              "$\\exp(\\theta_u)$ (posterior median, 95\\% CrI) and pass probability, beside the "
              "per-study frequentist estimates.")
    rb = t.read("rep_bayes.csv", "level == study")
    s = rb[rb.level == "study"]
    body = []
    for _, r in s[s.parameter == "theta"].iterrows():
        body.append([tex(S.reference_label(r["unit"])), t.est_ci(r, "or_median", "or_q025", "or_q975", "ratio"),
                     t.v(r, "p_or_ge_endpoint", "ratio"), t.v(r, "unit_enrichment", "ratio"),
                     t.v(r, "unit_mh_odds_ratio", "ratio"), t.v(r, "unit_observed", "count"),
                     t.v(r, "unit_expected", "sig2")])
    body.append(("midrule",))
    for par, lab in (("mu", "Pooled, $\\exp(\\mu)$"), ("theta_new", "New study, $\\exp(\\theta_{\\mathrm{new}})$")):
        r = first_row(s, parameter=par)
        if r is not None:
            body.append([lab, t.est_ci(r, "or_median", "or_q025", "or_q975", "ratio"), t.v(r, "p_or_ge_endpoint", "ratio"),
                         DASH, DASH, DASH, DASH])
    r = first_row(s, parameter="sigma")
    if r is not None:
        t.note(f"Between-study SD $\\sigma$: {t.est_ci(r, 'mean', 'q025', 'q975', 'ratio')}; max.\\ "
               f"$\\widehat R$ {t.v(r, 'r_hat_max_all', 'rhat')}, min.\\ bulk ESS {t.v(r, 'ess_bulk_min_all', 'count')}, "
               f"{t.v(r, 'n_divergent', 'count')} divergent transitions.")
    header = ["Study & OR [95\\% CrI] & $P$(pass) & O/E & MH OR & Observed & Expected"]
    t.tex = t.render("@{}lccrrrr@{}", header, body)
    return t


def tabS_tuning() -> Table:
    t = Table("tabS_tuning", "tab:S-tuning",
              "Optimism of tuning the configuration on perturbation labels (pooled perturbation "
              "reference, global universe): configuration selected on one half of the TFs (or all but one "
              "study) and scored on the other, against the label-free choice (best cross-validated "
              "log-likelihood) and the fixed primary Family-HB.")
    to = t.read("tune_optimism.csv", "fold_set == all_folds; split_type in {random_tf_half, loso}; "
                                     "criterion in {pooled_ap_lift, tf_geomean_ap_lift}")
    sub = to[(to.fold_set == "all_folds") & to.split_type.isin(["random_tf_half", "loso"])
             & to.criterion.isin(["pooled_ap_lift", "tf_geomean_ap_lift"])]
    sel_lab = {"label_tuned": "tuned on labels", "label_free_cvll": "label-free (CV)", "primary_family_hb": "primary",
               "oracle_heldout": "best on held-out half"}
    crit_lab = {"pooled_ap_lift": "pooled AP lift", "tf_geomean_ap_lift": "per-TF AP lift (geom.\\ mean)"}
    split_lab = {"random_tf_half": "random TF halves", "loso": "leave one study out"}
    grid_lab = {"all": "all configurations", "primary_evidence": "primary evidence only"}
    body = []
    for grid in ["all", "primary_evidence"]:
        for split in ["random_tf_half", "loso"]:
            g = sub[(sub.grid == grid) & (sub.split_type == split)]
            if not len(g):
                continue
            r0 = g.iloc[0]
            body.append(("panel", f"{grid_lab[grid]} ({t.v(r0, 'n_configs', 'count')} configurations), "
                                  f"{split_lab[split]} ({t.v(r0, 'n_splits', 'count')} splits)"))
            for crit in ["pooled_ap_lift", "tf_geomean_ap_lift"]:
                body.append(("raw", f"\\multicolumn{{8}}{{@{{}}l}}{{\\quad Criterion: {crit_lab[crit]}}} \\\\"))
                for sel in ["label_tuned", "label_free_cvll", "primary_family_hb"]:
                    r = first_row(g, criterion=crit, selector=sel)
                    if r is None:
                        continue
                    cfg = f"{tex(config_label(str(r['selected_config_modal'])))} ({t.v(r, 'selected_config_modal_freq', 'ratio')})"
                    body.append(("row", ["\\quad " + tex(sel_lab[sel]) + t.ph_if(X.is_posthoc_row(r)),
                                         f"\\parbox[t]{{3.6cm}}{{\\raggedright\\footnotesize {cfg}}}",
                                         t.v(r, "n_distinct_selected", "count"), t.v(r, "in_mean", "ratio3"),
                                         t.v(r, "out_mean", "ratio3"), t.v(r, "gap_mean", "ratio3"),
                                         t.v(r, "out_minus_primary_mean", "ratio3"), t.v(r, "frac_out_gt_primary", "ratio")],
                                 1.0 if len(cfg) < 40 else 1.8))
    t.note("Selected: most frequently selected configuration (share of splits); distinct: number of "
           "different configurations selected. In / out: criterion on the tuning / held-out part; optimism: "
           "in $-$ out; out $-$ primary: held-out gain over Family-HB; $P$: share of splits with a held-out "
           "gain.")
    header = ["Selection & Selected (share) & Distinct & In & Out & Optimism & Out $-$ prim. & $P$"]
    t.tex = render_split(t, "@{}lp{3.6cm}rrrrrr@{}", header, body, max_lines=44, tabcolsep=3.5)
    return t


def tabS_tune_weights() -> Table:
    t = Table("tabS_tune_weights", "tab:S-tune-weights",
              "A joint selection objective that weighs dropout robustness against the perturbation reward, "
              "$J = w_s\\,\\mathrm{Rob}_{\\mathrm{source}} + "
              "w_b\\,\\mathrm{Rob}_{\\mathrm{block}} + w_p\\,\\mathrm{Pert}$ over the weight simplex: share of "
              "weight points at which each candidate is selected, for two robustness metrics and two "
              "candidate sets, with the inputs of the objective.")
    tw = t.read("tune_weight_summary.csv", "all rows")
    combos = [("f1", "lcm5"), ("f1", "all10"), ("precision", "lcm5"), ("precision", "all10")]
    body = []
    for cfg in LCM5 + RULES5:
        r = first_row(tw, config_id=cfg, rob_metric="f1", candidate_set="all10")
        if r is None:
            continue
        cells = [SL(cfg), t.v(r, "D_source", "ratio3"), t.v(r, "D_block", "ratio3"), t.v(r, "full_pooled_ap_lift", "ratio3")]
        for m, c in combos:
            rr = first_row(tw, config_id=cfg, rob_metric=m, candidate_set=c)
            cells.append(t.v(rr, "share_grid_selected_full", "ratio") if rr is not None else DASH)
        body.append(cells)
    r0 = first_row(tw, rob_metric="f1", candidate_set="lcm5")
    t.note("$D$: mean rank disagreement under source and block (family and lineage) dropout ($F_1$ "
           "metric); Pert: pooled perturbation AP lift. Candidate sets: the five latent-class estimators "
           "(LC) or these and the five combination rules (all)."
           + (f" Weight points: {t.v(r0, 'n_grid', 'count')}." if r0 is not None else ""))
    header = [" & " + mc(3, "Objective inputs") + " & " + mc(4, "Share of weight points selected"),
              cmid("2-4") + cmid("5-8"),
              "Candidate & $D$ source & $D$ block & Pert & $F_1$, LC & $F_1$, all & prec., LC & prec., all"]
    t.tex = t.render("@{}lrrrcccc@{}", header, body, tabcolsep=4)
    return t


SIM_METHODS = ["oracle", "family_hb", "em", "dependence", "family_regularized", "union", "majority",
               "vote_fraction", "fisher", "stouffer", "family_hb_obs", "em_obs"]
SCEN_LABEL = {"null_independent": "Truth independent of popularity", "bio_hub": "Biological hubs",
              "copying": "Hub-concentrated literature copying"}


def tabS_simhub() -> Table:
    t = Table("tabS_simhub", "tab:S-simhub",
              "Hub-bias simulation with known truth: BFDR\\,$\\le$\\,QS set size and realised FDR, "
              "precision and over-promotion index of top-popularity-decile TFs at a set size matched to the "
              "Family-HB set, and AUROC over all edges (mean over replicates, 95\\% $t$ interval).")
    sh = t.read("sim_hub_summary.csv", "set_rule in {bfdr05, size_matched, score}")
    qs = q_from_name("bfdr05")
    t.caption = t.caption.replace("QS", qs)
    body = []
    trunc = [False]
    ph_all = all(X.is_posthoc_row(r) for _, r in sh.iterrows())
    for sc in ["null_independent", "bio_hub", "copying"]:
        if sc not in set(sh.scenario):
            continue
        body.append(("panel", SCEN_LABEL.get(sc, tex(sc))))
        for m in SIM_METHODS:
            def g(rule, metric):
                return first_row(sh, scenario=sc, method=m, set_rule=rule, metric=metric)
            sz, fdr = g("bfdr05", "set_size"), g("bfdr05", "fdr")
            pr, ix = g("size_matched", "precision"), g("size_matched", "over_promotion_top_decile")
            au = g("score", "auroc_all")
            if all(x is None for x in (sz, fdr, pr, ix, au)):
                continue
            nrep = next((x for x in (sz, pr, au) if x is not None), None)
            lab = SL(m)
            if nrep is not None and nrep["n_reps"] < sh.n_reps.max():
                lab += f" ($n$ = {t.v(nrep, 'n_reps', 'count')})"
            if pr is not None and X._isnum(pr["ci_lo"]) and float(pr["ci_lo"]) < 0:
                trunc[0] = True     # a precision cannot be negative: the t interval is truncated at 0
                pr_cell = (f"{t.v(pr, 'mean', 'ratio')} [{t.d(0.0, 'ratio', [(pr, 'ci_lo')])}, "
                           f"{t.v(pr, 'ci_hi', 'ratio')}]")
            else:
                pr_cell = t.est_ci(pr, "mean", "ci_lo", "ci_hi", "ratio")
            body.append([lab, t.v(sz, "mean", "count"), t.v(fdr, "mean", "ratio"),
                         pr_cell, t.est_ci(ix, "mean", "ci_lo", "ci_hi", "ratio"),
                         t.v(au, "mean", "ratio3")])
    t.note("Over-promotion index: share of the set at top-popularity-decile TFs divided by the share of "
           "true edges there (1 = proportional to biology). $n$: replicates behind a row when fewer than "
           "the design." + (" Precision intervals are truncated at 0." if trunc[0] else ""))
    if ph_all:
        t.caption += t.ph()
    header = [" & " + mc(2, f"BFDR\\,$\\le$\\,{qs} set") + " & " + mc(2, "Size-matched set") + " & ",
              cmid("2-3") + cmid("4-5"),
              "Method & Size & FDR & Precision [95\\% CI] & Index [95\\% CI] & AUROC"]
    t.tex = render_split(t, "@{}lrrccc@{}", header, body, max_lines=46, tabcolsep=4)
    return t


def tabS_simhub_modes() -> Table:
    t = Table("tabS_simhub_modes", "tab:S-simhub-modes",
              "Posterior-mode structure of the latent-class fits in the hub-bias simulation: restarts "
              "reaching the best optimum, distinct optima, median gap to the next optimum, selected "
              "optima that are label-swapped or have an inflated true class, and the fitted class weight.")
    sm = t.read("sim_hub_modes.csv", "all rows")
    body = []
    ph_all = all(X.is_posthoc_row(r) for _, r in sm.iterrows())
    for sc in ["null_independent", "bio_hub", "copying"]:
        sub = sm[sm.scenario == sc]
        if not len(sub):
            continue
        body.append(("panel", SCEN_LABEL.get(sc, tex(sc))))
        for m in ["em", "dependence", "family_regularized", "family_hb", "em_obs", "family_hb_obs"]:
            r = first_row(sub, model=m)
            if r is None:
                continue
            body.append([SL(m), t.v(r, "n_reps", "count"), t.v(r, "n_restarts", "count"),
                         t.v(r, "frac_restarts_at_best_mean", "ratio"), t.v(r, "n_distinct_optima_mean", "num1"),
                         t.v(r, "gap_to_next_optimum_median", "num1"), t.v(r, "n_reps_label_swapped_best", "count"),
                         t.v(r, "n_reps_t_class_inflated", "count"), t.v(r, "w_T_median", "prob"),
                         t.v(r, "true_prevalence_mean", "prob")])
    if ph_all:
        t.caption += t.ph()
    header = ["Model & Reps & Restarts & At best & Optima & Gap & Swapped & Inflated & $w_{\\mathrm T}$ & Truth"]
    t.tex = t.render("@{}lrrrrrrrrr@{}", header, body, tabcolsep=3.5)
    return t


OBS_FITS = ["family_hb", "family_hb_obs", "family_hb_obs__tfw", "em", "em_obs", "em_obs__tfw"]
OBS_SHORT = {"family_hb": "Family-HB", "family_hb_obs": "\\quad + observation", "family_hb_obs__tfw": "\\quad + obs., TF-w.",
             "em": "EM baseline", "em_obs": "\\quad + observation", "em_obs__tfw": "\\quad + obs., TF-w."}
ROLE_SHORT = {"label_swapped": "swapped", "second_optimum": "2nd optimum", "none": "none"}
OBS_LABEL = {"family_hb": "Family-HB", "family_hb_obs": "Family-HB + obs.", "family_hb_obs__tfw": "Family-HB + obs., TF-weighted",
             "em": "EM baseline", "em_obs": "EM + obs.", "em_obs__tfw": "EM + obs., TF-weighted",
             "family_hb_obs_swap": "Family-HB + obs., swapped labels", "em_obs_swap": "EM + obs., swapped labels",
             "family_hb_obs__tfw_opt2": "Family-HB + obs., TF-w., 2nd optimum", "fisher": "Fisher",
             "status3": "3-level status", "family_hb_unobserved_last": "Family-HB, unobserved last", "random": "Random"}


def tabS_obs_fit() -> Table:
    t = Table("tabS_obs_fit", "tab:S-obs-fit",
              "Informative-observation extension (posterior mode, Laplace intervals) against the ignorable "
              "models: class weight of T, share of the joint log-likelihood due to observation, size of the "
              "BFDR\\,$\\le$\\,0.05 set, mean credibility by evidence status, literature annotators whose "
              "observation rate is higher under T, and agreement with the ignorable counterpart.")
    t.caption = t.caption.replace("BFDR\\,$\\le$\\,0.05", f"BFDR\\,$\\le$\\,{q_from_name('n_bfdr_0_05')}")
    of = t.read("obs_fit_summary.csv", "all rows")
    body = []
    for u in UNIVERSES:
        sub = of[of.universe == u]
        if not len(sub):
            continue
        body.append(("panel", f"{U(u)} universe"))
        for f in OBS_FITS:
            r = first_row(sub, fit_id=f)
            if r is None:
                continue
            eta = (f"{t.v(r, 'n_lit_eta_T_gt_NonT', 'count')}/{t.v(r, 'n_lit', 'count')}"
                   if X._isnum(r["n_lit"]) else DASH)
            body.append([OBS_SHORT.get(f, tex(f)) + t.ph_if(X.is_posthoc_row(r)),
                         t.est_ci(r, "w_T", "w_T_lo", "w_T_hi", "sig2", stack=True) if X._isnum(r["w_T_lo"]) else t.v(r, "w_T", "sig2"),
                         t.v(r, "obs_share_of_joint_loglik", "ratio"), t.v(r, "n_bfdr_0_05", "count"),
                         t.v(r, "mean_cred_unobserved", "sig2"), t.v(r, "mean_cred_observed_neg", "sig2"),
                         t.v(r, "mean_cred_reported", "sig2"), eta,
                         t.v(r, "spearman_w_vs_mar_observed", "ratio"),
                         tex(ROLE_SHORT.get(str(r["alt_optimum_role"]), str(r["alt_optimum_role"])))
                         if isinstance(r["alt_optimum_role"], str) else DASH])
    t.note("Obs.\\ share: observation term over the joint log-likelihood. $\\eta_{\\mathrm T}>\\eta_{\\bar{\\mathrm T}}$: "
           "literature annotators that observe true edges more often than non-edges. Spearman: credibility of "
           "observed edges against the ignorable counterpart. Alt.\\ optimum: role of the next optimum "
           "(label-swapped: T and burst classes exchanged).")
    header = [" & & Obs. & BFDR & " + mc(3, "Mean credibility") +
              " & $\\eta_{\\mathrm T}>$ & & Alt.",
              cmid("5-7"),
              "Fit & $w_{\\mathrm T}$ & share & set $n$ & unobs. & obs.\\,neg. & reported & $\\eta_{\\bar{\\mathrm T}}$ & Spearman & optimum"]
    t.tex = render_split(t, "@{}lccrccccrl@{}", header, body, max_lines=46, tabcolsep=2.2)
    return t


def tabS_obs_heldout() -> Table:
    t = Table("tabS_obs_heldout", "tab:S-obs-heldout",
              "Held-out evaluation of the informative-observation extension in the global universe (all "
              "evaluable edges): AP lift [95\\% TF-cluster bootstrap interval], within-TF AUROC and "
              "stratified Enr@500, with the primary estimators, Fisher, the post hoc status scores and random.")
    oh = t.read("obs_heldout.csv", "universe == global")
    scores = ["family_hb_obs", "em_obs", "family_hb_obs__tfw", "em_obs__tfw", "family_hb_obs_swap", "em_obs_swap",
              "family_hb", "em", "fisher", "status3", "family_hb_unobserved_last", "random"]
    body = []
    for ref in ["encode", "enets2", "pert_pooled"]:
        sub = oh[(oh.universe == "global") & (oh.reference == ref)]
        if not len(sub):
            continue
        body.append(("panel", RL(ref)))
        for s in scores:
            r = first_row(sub, score=s)
            if r is None:
                continue
            body.append([tex(OBS_LABEL.get(s, s)) + t.ph_if(X.is_posthoc_row(r)),
                         t.est_ci(r, "ap_lift", "ap_lift_lo", "ap_lift_hi", "ratio"),
                         t.v(r, "within_tf_auroc", "ratio3"), t.v(r, "strat_enr_at_500", "ratio")])
    header = ["Score & AP lift [95\\% CI] & Within-TF AUROC & Strat.\\ Enr@500"]
    t.tex = render_split(t, "@{}lccc@{}", header, body, max_lines=46)
    return t


def tabS_obs_cv() -> Table:
    t = Table("tabS_obs_cv", "tab:S-obs-cv")
    oc = t.read("obs_cv.csv", "all rows")
    nf = t.v(oc.iloc[0], "n_folds", "count") if len(oc) else DASH
    t.caption = (f"Cross-validated held-out log-likelihood ({nf} folds by TF) of the informative-observation "
                 "models and their ignorable counterparts: of the reports given the observation pattern "
                 "(per observed edge), of the observation indicators and of both (per edge).")
    metrics = [("cond_per_obs_edge", "Reports $|$ observation"), ("obs_per_edge", "Observation"),
               ("joint_per_edge", "Joint")]
    body = []
    for u in ["global", "immune"]:
        sub = oc[oc.universe == u]
        if not len(sub):
            continue
        body.append(("panel", f"{U(u)} universe"))
        for f in OBS_FITS:
            cells = [tex(OBS_LABEL.get(f, f))]
            for m, _ in metrics:
                r = first_row(sub, fit=f, metric=m)
                if r is None:
                    cells.append(DASH)
                    continue
                d = (f" ({t.v(r, 'diff_vs_mar_counterpart', 'sig2')})"
                     if X._isnum(r["diff_vs_mar_counterpart"]) else "")
                cells.append(t.v(r, "mean", "num4") + d + t.ph_if(X.is_posthoc_row(r)))
            body.append(cells)
    t.note("In parentheses: difference to the ignorable counterpart (Family-HB or EM) on the same folds. "
           "For the ignorable models the observation term is a class-independent Bernoulli model of the "
           "literature observation indicators.")
    header = ["Model & " + " & ".join(lab for _, lab in metrics)]
    t.tex = t.render("@{}lccc@{}", header, body)
    return t


def tabS_fm_null() -> Table:
    t = Table("tabS_fm_null", "tab:S-fm-null",
              "Foundation-model attention against its uniform-attention (1/$L$) null in each tissue: "
              "coverage, Spearman correlation with the null, median overlap of the top-$k$ targets per TF "
              "with the null's (and its expectation under random ranking), and the share of top-$k$ pairs "
              "co-observed in only one cell or in at most four cells.")
    fa = t.read("fm_attention_null.csv", "all rows")
    vlab = {"attn": "raw", "attn_T": "raw, transposed", "attn_uniform": "1/$L$ null", "attn_norm": "normalised",
            "attn_T_norm": "normalised, transposed"}
    body = []
    for tis in TISSUES:
        body.append(("panel", U(tis)))
        for model in ["scgpt", "geneformer"]:
            for var in ["attn", "attn_T", "attn_norm", "attn_T_norm", "attn_uniform"]:
                r = first_row(fa, tissue=tis, model=model, variant=var)
                if r is None:
                    continue
                body.append([("scGPT" if model == "scgpt" else "Geneformer") + ", " + vlab[var],
                             t.v(r, "coverage", "pct"), t.v(r, "spearman_with_uniform", "ratio"),
                             t.v(r, "topk_overlap_with_uniform_median", "ratio"),
                             t.v(r, "topk_overlap_random_expectation", "ratio"),
                             t.v(r, "topk_frac_coobs_eq1", "ratio"), t.v(r, "topk_frac_coobs_le4", "ratio")])
    r0 = first_row(fa, tissue="immune")
    if r0 is not None:
        t.note(f"$k$ = {t.v(r0, 'k', 'count')} targets per TF.")
    header = [" & Coverage & Spearman & " + mc(2, "Top-$k$ overlap") + " & " + mc(2, "Top-$k$ co-observed in"),
              cmid("4-5") + cmid("6-7"),
              "Score & (\\%) & with null & median & random & 1 cell & $\\leq$4 cells"]
    t.tex = render_split(t, "@{}lrrrrrr@{}", header, body, max_lines=46, tabcolsep=4)
    return t


FM_METHODS = ["scgpt_attn", "scgpt_attn_norm", "scgpt_attn_uniform", "geneformer_attn", "geneformer_attn_norm",
              "geneformer_attn_uniform", "grnboost2", "genie3", "pearson_abs", "spearman_abs", "family_hb_nuts",
              "random"]


def tabS_fm_heldout() -> Table:
    t = Table("tabS_fm_heldout", "tab:S-fm-heldout",
              "Network-inference methods in the tissue-context universes: TF-matched enrichment of each "
              "method's predicted set (top 50 targets per TF) among held-out positives, and its ratio to the "
              "method's matched null (the 1/$L$ uniform-attention null of the same model for attention scores, "
              "otherwise the coverage indicator); 95\\% TF-cluster bootstrap intervals.")
    hm = t.read("heldout_methods_context.csv", "topk_per_tf == 50 & eval_subset == all_evaluable")
    refs = ["encode", "enets2", "pert_pooled"]
    body = []
    low = False
    for tis in TISSUES:
        body.append(("panel", U(tis)))
        for m in FM_METHODS:
            cells = [SHORT(m)]
            hit = False
            for ref in refs:
                r = first_row(hm, universe=tis, reference=ref, score=m, topk_per_tf=50)
                if r is None:
                    cells += [DASH, DASH]
                    continue
                hit = True
                e = t.est_ci(r, "enr_tf_matched", "enr_tf_matched_lo", "enr_tf_matched_hi", "ratio", stack=True)
                if S.as_bool(r["low_support"]):
                    e = e.replace("\\\\[-1pt]", "$^{*}$\\\\[-1pt]", 1)
                    low = True
                cells += [e, t.est_ci(r, "enr_ratio_to_null", "enr_ratio_to_null_lo", "enr_ratio_to_null_hi", "ratio", stack=True)
                          if X._isnum(r["enr_ratio_to_null"]) else DASH]
            if hit:
                body.append(cells)
    if low:
        t.note("$^{*}$\\,Low support (few evaluable TFs or positives).")
    t.note("Enrichment 1: as many positives as expected from the TF composition of the set. Ratio to null "
           "1: no better than the matched null.")
    ph_null = any(X.is_posthoc_col(r, "enr_ratio_to_null") for _, r in hm.head(50).iterrows())
    header = [" & " + " & ".join(mc(2, RL(r)) for r in refs),
              cmid("2-3") + cmid("4-5") + cmid("6-7"),
              "Method" + (" & Enrichment & vs null" + t.ph_if(ph_null)) * len(refs)]
    t.tex = render_split(t, "@{}lcccccc@{}", header, body, max_lines=40, tabcolsep=3)
    return t


FM_SENS_METHODS = ["scgpt_attn", "geneformer_attn", "grnboost2", "pearson_abs", "family_hb_nuts", "random"]


def tabS_fm_sens() -> Table:
    """Sensitivity of the predicted sets: number of targets per TF, and attention restricted to co-observed pairs."""
    from r2.common import PRESPECIFIED as _PRE
    ks = sorted({int(_PRE["topk_per_tf"]), *[int(k) for k in _PRE["topk_per_tf_sensitivity"]]})
    t = Table("tabS_fm_sens", "tab:S-fm-sens",
              "Sensitivity of the predicted sets in the tissue-context universes. (a) TF-matched enrichment among "
              "held-out ChIP positives of each method's predicted set of " + ", ".join(str(k) for k in ks[:-1])
              + f" or {ks[-1]} targets per TF (point estimates). (b) Attention restricted to pairs co-observed in at "
              f"least {int(_PRE['fm_min_coobs_sensitivity'])} cells: share of the unrestricted top-{int(_PRE['topk_per_tf'])} "
              "set that is retained, and overlap of the top set with that of the uniform-attention ($1/L$) null.")
    hm = t.read("heldout_methods_context.csv", "eval_subset == all_evaluable & reference in {encode, enets2} & "
                "score in FM_SENS_METHODS")
    fc = t.read("fm_min_coobs_sensitivity.csv", "variant == attn & k == topk_per_tf")
    refs = ["encode", "enets2"]
    body = []
    for tis in TISSUES:
        body.append(("panel", U(tis)))
        for m in FM_SENS_METHODS:
            cells = [SHORT(m)]
            hit = False
            for ref in refs:
                for k in ks:
                    r = first_row(hm, universe=tis, reference=ref, score=m, topk_per_tf=k, eval_subset="all_evaluable")
                    cells.append(t.v(r, "enr_tf_matched", "ratio") if r is not None else DASH)
                    hit |= r is not None
            if hit:
                body.append(cells)
    head = [" & " + " & ".join(mc(len(ks), RL(r)) for r in refs),
            cmid(f"2-{1 + len(ks)}") + cmid(f"{2 + len(ks)}-{1 + 2 * len(ks)}"),
            "Method" + (" & " + " & ".join(f"$k={k}$" for k in ks)) * len(refs)]
    blocks = [dict(title="(a) Predicted-set size", colspec="@{}l" + "c" * (2 * len(ks)) + "@{}", header=head,
                   body=body, size="footnotesize", tabcolsep=4)]
    body2 = []
    k0 = int(_PRE["topk_per_tf"])
    for tis in TISSUES:
        for mdl in ("scgpt", "geneformer"):
            r = first_row(fc, tissue=tis, model=mdl, variant="attn", k=k0)
            if r is None:
                continue
            body2.append([U(tis), "scGPT" if mdl == "scgpt" else "Geneformer",
                          t.v(r, "frac_unfiltered_topk_retained", "pct0"),
                          t.v(r, "topk_overlap_with_uniform_median_unfiltered", "pct0"),
                          t.v(r, "topk_overlap_with_uniform_median", "pct0"),
                          t.v(r, "topk_overlap_random_expectation", "pct0")])
    head2 = [" & & Retained & " + mc(3, "Overlap with the $1/L$ null (\\%)"),
             cmid("4-6"),
             "Universe & Model & (\\%) & unrestricted & restricted & random"]
    blocks.append(dict(title="(b) Co-observation restriction", colspec="@{}llrccc@{}", header=head2, body=body2,
                       size="footnotesize", tabcolsep=5))
    t.note("Overlap: median over TFs of the share of a TF's top targets that are also among the top targets of the "
           "uniform-attention null; random: the overlap expected for a random ranking.")
    t.tex = t.render_blocks(blocks)
    return t


def _scaling_text(x: str) -> str:
    """Readable LaTeX for the quantity / subset labels of scale_fits.csv (no code identifiers)."""
    import re as _re
    t_ = str(x)
    t_ = t_.replace("chunked_stream", "streaming compression").replace("dense_unique", "whole-matrix compression")
    t_ = t_.replace("template + independent", "template and independent generators")
    t_ = _re.sub(r"\bK=(\d+)", r"$K=\1$", t_)
    t_ = _re.sub(r"P >= 1e(\d+)", r"$P\\geq 10^{\1}$", t_)
    t_ = t_.replace("NUTS 2x300/300", "NUTS, 2 chains $\\times$ (300 + 300)")
    t_ = t_.replace("value+gradient", "log posterior and gradient").replace("MAP 10-restart", "MAP, 10 restarts,")
    t_ = t_.replace("(bytes)", "in bytes")
    for a, b in (("P*K*S", "$PKS$"), ("N*K", "$NK$"), ("P*K", "$PK$")):
        t_ = t_.replace(a, b)
    t_ = _re.sub(r"\bvs ([NP])\b", r"vs $\1$", t_)
    t_ = _re.sub(r"^P vs", r"$P$ vs", t_)
    t_ = _re.sub(r"linear in ([NP])\b", r"linear in $\1$", t_)
    return t_


def tabS_scaling() -> Table:
    t = Table("tabS_scaling", "tab:S-scaling",
              "Empirical scaling exponents from the compute-scaling experiment: least-squares fits of "
              "$\\log_{10} y = a + b \\log_{10} x$ over the synthetic configurations (slope $b$ with "
              "standard error, $R^2$, number of configurations).")
    sf = t.read("scale_fits.csv", "form == log-log")
    sub = sf[sf.form.str.startswith("log10")]
    body = []
    for sess in sub.session.drop_duplicates():
        g = sub[sub.session == sess]
        ph = all(X.is_posthoc_row(r) for _, r in g.iterrows())
        body.append(("panel", f"Measurement session {tex(sess)}" + (t.ph() if ph else "")))
        for _, r in g.iterrows():
            q_, sb = _scaling_text(r["quantity"]), _scaling_text(r["subset"])
            body.append(("row", [f"\\parbox[t]{{5.2cm}}{{\\raggedright {q_}\\strut}}",
                                 f"\\parbox[t]{{4.2cm}}{{\\raggedright\\footnotesize {sb}\\strut}}",
                                 t.v(r, "n", "count"), f"{t.v(r, 'slope', 'ratio')} ({t.v(r, 'slope_se', 'ratio')})",
                                 t.v(r, "r2", "ratio")], 1.0 if len(str(r["subset"])) < 36 else 1.7))
    t.note("Session b replicated session a on the same machine with one measurement process at a time. Streaming "
           "compression: patterns found TF chunk by TF chunk and merged; whole-matrix compression: one pass over the "
           "full evidence matrix.")
    header = ["Quantity $y$ vs $x$ & Configurations & $n$ & Slope (SE) & $R^2$"]
    t.tex = render_split(t, "@{}p{5.2cm}p{4.2cm}rcr@{}", header, body, max_lines=36, tabcolsep=3)
    return t


# LaTeX rendering of scale_complexity.csv (stage -> stage label, time, memory, notes); the CSV rows are the
# derivation of compute_scaling.py, typeset here without code identifiers
COMPLEXITY_TEX = {
    "evidence encoding": ("Evidence encoding", "$O(NK)$", "$O(n_c K)$ per chunk",
                          "the $K$ values of an edge (unobserved, unreported, reported) are packed into base-3 words"),
    "pattern compression (streaming, by TF chunk)": (
        "Pattern compression, streaming by TF chunk", "$O(NK + N\\log n_c + C P_c\\log(C P_c))$",
        "$O(N)$ edge-to-pattern index $+\\ O(n_c K)$ chunk $+\\ O(PK)$ table",
        "sort-based unique per chunk of $n_c$ edges, merge of the $C$ chunk dictionaries, $O(N)$ re-indexing; "
        "hashing would give $O(NK)$ expected time"),
    "pattern compression (whole matrix, lcm.PatternData.from_matrix)": (
        "Pattern compression, whole matrix", "$O(NK\\log N)$", "$O(NK)$ matrix $+\\ O(N)$ sort arrays",
        "row-wise unique of the full evidence matrix"),
    "number of patterns P": (
        "Number of patterns $P$", "$P\\leq\\min(N,\\,3^K,\\,M+N_{\\mathrm{rep}})$", "--",
        "$M$ distinct observation masks, $N_{\\mathrm{rep}}$ edges with at least one report: every unreported edge "
        "has the all-zero pattern of its mask; TF-blocked observation keeps $M$ small"),
    "log posterior + gradient (one evaluation)": (
        "Log posterior and gradient, one evaluation", "$O(PKS)$", "$O(PK)+O(PS)$",
        "independent of $N$ given $P$: $N$ enters only through the pattern counts"),
    "MAP (multi-start L-BFGS)": (
        "MAP, multi-start L-BFGS", "$O(RIPKS)$", "$O(PK)+O(RP)$",
        "$R$ restarts, $I$ evaluations per restart; $I$ grew with $N$ in the experiments (Table~\\ref{tab:S-scaling})"),
    "NUTS": ("NUTS", "$O(CDLPKS)$", "$O(PK + CDd)$, $+\\ O(CDP)$ if credibility is stored per draw",
             "$C$ chains, $D$ iterations, $L$ leapfrog steps per iteration (at most $2^{10}-1$), $d$ parameters; "
             "$L$ grew with $N$ up to the tree-depth cap"),
    "posterior credibility (lcm.nuts_credibility)": (
        "Posterior credibility summaries", "$O(D'PKS)$", "$O(D'P)$ (quantiles); $O(P)$ (streaming mean and SD)",
        "$D'$ retained draws"),
    "edge-level outputs": ("Edge-level outputs", "$O(N)$", "$O(N)$",
                           "the credibility of an edge is that of its pattern (index lookup)"),
}


def tabS_complexity() -> Table:
    t = Table("tabS_complexity", "tab:S-complexity",
              "Time and memory complexity of each stage ($N$ edges, $K$ annotators, $P$ observation patterns, "
              "$S$ latent classes; further symbols in the notes column).")
    sc = t.read("scale_complexity.csv", "all rows")
    body = []
    unknown = [str(x) for x in sc.stage if str(x) not in COMPLEXITY_TEX]
    if unknown:
        raise ValueError(f"tabS_complexity: no LaTeX rendering for stages {unknown}")
    for _, r in sc.iterrows():
        lab, tm, mem, nt = COMPLEXITY_TEX[str(r["stage"])]
        body.append([f"\\parbox[t]{{2.8cm}}{{\\raggedright {lab}\\strut}}",
                     f"\\parbox[t]{{3.2cm}}{{\\raggedright\\footnotesize {tm}\\strut}}",
                     f"\\parbox[t]{{3.3cm}}{{\\raggedright\\footnotesize {mem}\\strut}}",
                     f"\\parbox[t]{{4.6cm}}{{\\raggedright\\scriptsize {nt}\\strut}}"])
    header = ["Stage & Time & Memory & Notes"]
    t.tex = t.render("@{}p{2.8cm}p{3.2cm}p{3.3cm}p{4.6cm}@{}", header, body, tabcolsep=3)
    return t


HELD_FULL_GROUPS = [["family_hb_nuts", "family_hb", "em", "dependence", "multifactor", "family_regularized"],
                    ["union", "majority", "vote_count", "vote_fraction", "n_families_reporting", "fisher", "stouffer"],
                    ["dorothea_level_heldin", "collectri_heldin"], ["status3", "family_hb_unobserved_last"], ["random"]]
HELD_METRICS = {"ap_lift": ("AP lift", "ap_lift_lo", "ap_lift_hi", "ratio", True),
                "within_tf_auroc": ("AUROC", None, None, "ratio3", False),
                "strat_enr_at_500": ("Enr@500", "strat_enr_at_500_lo", "strat_enr_at_500_hi", "ratio", True)}


def _heldout_full(name: str, refs: list[str], metrics: list[str], what: str, omit: tuple = ()) -> Table:
    t = Table(name, f"tab:S-{name.split('_', 1)[1].replace('_', '-')}",
              f"Held-out evaluation in the global universe, all scores (all evaluable edges): {what}.")
    he = t.read("heldout_estimators.csv", "universe == global & eval_subset == all_evaluable & reference in {"
                + ", ".join(refs) + "}")
    g = he[(he.universe == "global") & (he.eval_subset == "all_evaluable")]
    refs = [r for r in refs if r in set(g.reference)]
    body = []
    low_refs = set()
    for gi, grp in enumerate(HELD_FULL_GROUPS):
        if gi:
            body.append(("addlinespace",))
        for s in grp:
            rr = [first_row(g, reference=ref, score=s) for ref in refs]
            if all(r is None for r in rr):
                continue
            ph = any(X.is_posthoc_row(r) for r in rr if r is not None)
            cells = [SHORT(s) + t.ph_if(ph)]
            for ref, r in zip(refs, rr):
                if r is not None and S.as_bool(r["low_support"]):
                    low_refs.add(ref)
                for m in metrics:
                    _, lo, hi, kind, ci = HELD_METRICS[m]
                    if r is None:
                        cells.append(DASH)
                    elif ci:
                        cells.append(t.est_ci(r, m, lo, hi, kind, stack=True))
                    else:
                        cells.append(t.v(r, m, kind))
            body.append(cells)
    if low_refs:
        t.note(f"$^{{*}}$\\,Low support (fewer than {_module_constant('heldout_eval', 'LOW_SUPPORT_TFS')} TFs with "
               f"positives or fewer than {_module_constant('heldout_eval', 'LOW_SUPPORT_POS')} positives): "
               + ", ".join(RL(r) for r in refs if r in low_refs) + ".")
    omitted = [r for r in omit if r in set(g.reference)]
    if omitted:
        t.note("Not shown: " + ", ".join(RL(r) for r in omitted) + " (almost no positives at the fold-change "
               "threshold, Table~\\ref{tab:cohorts}); their results are in the analysis tables of the code repository.")
    t.note("AP lift: 95\\% TF-cluster bootstrap interval; AUROC: within-TF AUROC; Enr@500: stratified "
           "enrichment of the top 500 edges.")
    nm = len(metrics)
    head1 = " & " + " & ".join(mc(nm, RL(r) + ("$^{*}$" if r in low_refs else "")) for r in refs)
    cm = "".join(cmid(f"{2 + i * nm}-{1 + (i + 1) * nm}") for i in range(len(refs)))
    head2 = "Score" + (" & " + " & ".join(HELD_METRICS[m][0] for m in metrics)) * len(refs)
    t.tex = render_split(t, "@{}l" + "c" * (nm * len(refs)) + "@{}", [head1, cm, head2], body, max_lines=44,
                         tabcolsep=1.8)
    return t


def tabS_heldout_chip() -> Table:
    return _heldout_full("tabS_heldout_chip", ["encode", "encode_strict", "enets2"],
                         ["ap_lift", "within_tf_auroc", "strat_enr_at_500"], "ChIP-seq references")


def tabS_heldout_pert() -> Table:
    return _heldout_full("tabS_heldout_pert", ["norman", "papalexi", "replogle", "pert_pooled"],
                         ["ap_lift", "within_tf_auroc"], "perturbation cohorts, AP lift and within-TF AUROC",
                         omit=("dixit_d7", "dixit_d13"))


def tabS_heldout_pert_enr() -> Table:
    return _heldout_full("tabS_heldout_pert_enr", ["norman", "papalexi", "replogle", "pert_pooled"],
                         ["strat_enr_at_500"], "perturbation cohorts, stratified enrichment of the top 500 edges",
                         omit=("dixit_d7", "dixit_d13"))


def tabS_decomp_context() -> Table:
    t = tab_decomp(credibility="context_fit", name="tabS_decomp_context", label="tab:S-decomp-context")
    return t


SUPP = {
    "tabS_sources": tabS_sources,
    "tabS_annotator_coverage": tabS_annotator_coverage,
    "tabS_annotator_params": tabS_annotator_params,
    "tabS_sens_fit": tabS_sens_fit,
    "tabS_sens_heldout": tabS_sens_heldout,
    "tabS_cv": tabS_cv,
    "tabS_ppc": tabS_ppc,
    "tabS_recovery": tabS_recovery,
    "tabS_modes_sim": tabS_modes_sim,
    "tabS_modes_gov": tabS_modes_gov,
    "tabS_heldout_chip": tabS_heldout_chip,
    "tabS_heldout_pert": tabS_heldout_pert,
    "tabS_heldout_pert_enr": tabS_heldout_pert_enr,
    "tabS_decomp_context": tabS_decomp_context,
    "tabS_gov_grid": tabS_gov_grid,
    "tabS_gov_grid_pert": tabS_gov_grid_pert,
    "tabS_deconc": tabS_deconc,
    "tabS_rep_bayes": tabS_rep_bayes,
    "tabS_rep_prior": tabS_rep_prior,
    "tabS_rep_leavek": tabS_rep_leavek,
    "tabS_tuning": tabS_tuning,
    "tabS_tune_weights": tabS_tune_weights,
    "tabS_tune_status": tabS_tune_status,
    "tabS_simhub": tabS_simhub,
    "tabS_simhub_modes": tabS_simhub_modes,
    "tabS_obs_fit": tabS_obs_fit,
    "tabS_obs_heldout": tabS_obs_heldout,
    "tabS_obs_cv": tabS_obs_cv,
    "tabS_fm_null": tabS_fm_null,
    "tabS_fm_heldout": tabS_fm_heldout,
    "tabS_fm_sens": tabS_fm_sens,
    "tabS_scaling": tabS_scaling,
    "tabS_complexity": tabS_complexity,
}


def _here(fn):
    """Supplement floats may stand where they are input ([htbp]): each section holds its own floats
    (placeins in supplement.tex), so a float need not wait for the next page."""
    def wrapped():
        t = fn()
        t.tex = t.tex.replace("\\begin{table}[tbp]", "\\begin{table}[htbp]")
        return t
    wrapped.__name__ = fn.__name__
    return wrapped


SUPP = {k: _here(f) for k, f in SUPP.items()}


# ----------------------------------------------------------------------------- supplement bibliography
SUPP_TEX = "supplement.tex"
SUPP_REFS = "references_supp.tex"
SUPP_REFS_KEYS = "references_supp_keys.tex"


def _inputs(path, seen=None) -> list:
    """``path`` and every file it \\input's (recursively; .tex added when missing)."""
    import re as _re
    seen = seen if seen is not None else []
    if path in seen or not path.exists():
        return seen
    seen.append(path)
    for m in _re.finditer(r"\\input\{([^}]+)\}", path.read_text()):
        f = m.group(1)
        f = f if f.endswith(".tex") else f + ".tex"
        _inputs(S.PAPER / f, seen)
    return seen


def write_supp_references() -> list[str]:
    """Write ``paper/r2/references_supp.tex``: the entries of ``references.tex`` cited in the supplement
    (supplement.tex and every table or figure file it inputs), in the order of ``references.tex``."""
    import re as _re
    keys = []
    for f in _inputs(S.PAPER / SUPP_TEX):
        if f.name == SUPP_REFS:
            continue
        txt = "\n".join(ln for ln in f.read_text().splitlines() if not ln.lstrip().startswith("%"))
        for m in _re.finditer(r"\\cite[tp]?\*?(?:\[[^\]]*\])*\{([^}]+)\}", txt):
            keys += [k.strip() for k in m.group(1).split(",") if k.strip()]
    ref = (S.PAPER / "references.tex").read_text()
    items = _re.split(r"(?=\\bibitem)", ref.split("\\end{thebibliography}")[0])[1:]
    by_key = {_re.search(r"\]\{([^}]+)\}", it).group(1): it.strip() for it in items}
    missing = sorted(set(keys) - set(by_key))
    if missing:
        raise ValueError(f"supplement cites keys missing from references.tex: {missing}")
    used = [it for k, it in by_key.items() if k in set(keys)]
    out = ("% generated by src/r2/paper_assets/tables_supp.py (write_supp_references): entries of references.tex\n"
           "% cited in the supplement; do not edit by hand\n\\begin{thebibliography}{99}\n\n"
           + "\n\n".join(used) + "\n\n\\end{thebibliography}\n")
    (S.PAPER / SUPP_REFS).write_text(out)
    # xr-hyper imports the main text's \\bibcite entries together with its labels; natbib would then report every
    # entry of the supplement bibliography as multiply defined.  This file, input right after \\externaldocument,
    # forgets the imported citations of the keys the supplement lists itself.
    clear = ("% generated by src/r2/paper_assets/tables_supp.py (write_supp_references); do not edit by hand\n"
             "\\makeatletter\n"
             + "".join(f"\\expandafter\\let\\csname b@{k}\\endcsname\\relax\n" for k in sorted(set(keys)))
             + "\\makeatother\n")
    (S.PAPER / SUPP_REFS_KEYS).write_text(clear)
    return sorted(set(keys))


def main(argv=None) -> int:
    """``make_paper_assets.py tables`` runs ``tables.main`` (every table) and then this: the supplement bibliography."""
    keys = write_supp_references()
    print(f"[tables_supp] {SUPP_REFS}: {len(keys)} cited entries")
    return 0
