"""Macro specifications for ``paper/r2/numbers.tex`` (see ``prose_numbers.py`` for the rules).

One entry per number printed in the manuscript prose.  Every entry names its source: a CSV of
``data/r2/tables`` with a row filter and a column (and, where needed, an aggregation over the selected
rows), a prespecified constant (``r2.common.PRESPECIFIED``), a frozen configuration value or code
constant, or a documented arithmetic combination of earlier macros.  The order follows the manuscript.
``was`` records the literal that the prose printed before the binding (2026-10-14), so that the
binding audit can list every number the binding changed.

Naming: ``\\n`` + topic + qualifier, letters only; universes ``Glob Imm Kid Lung``; references
``Enc`` (ENCODE TFBS), ``Enet`` (ENETS2), ``Pert`` (pooled perturbation); intervals ``Lo``/``Hi``;
ranges over rows ``Min``/``Max``.
"""
from __future__ import annotations

if __package__ in (None, ""):
    from r2.paper_assets.prose_numbers import Num  # type: ignore
else:
    from .prose_numbers import Num

U = {"global": "Glob", "immune": "Imm", "kidney": "Kid", "lung": "Lung"}
TIS = {"immune": "Imm", "kidney": "Kid", "lung": "Lung"}
REF = {"encode": "Enc", "enets2": "Enet", "pert_pooled": "Pert"}


def N(name, csv, query, col, fmt="r2", **kw) -> Num:
    return Num(name, csv=csv, query=query, col=col, fmt=fmt, **kw)


def C(name, const, fmt, **kw) -> Num:
    return Num(name, const=const, fmt=fmt, **kw)


def E(name, expr, fmt, **kw) -> Num:
    return Num(name, expr=expr, fmt=fmt, **kw)


def CI(name, csv, query, col, lo, hi, fmt="r2", was=(None, None, None), note="", **kw) -> list[Num]:
    """Value and interval: macros name, nameLo, nameHi."""
    return [N(name, csv, query, col, fmt, was=was[0], note=note, **kw),
            N(name + "Lo", csv, query, lo, fmt, was=was[1], note=(note + "; lower bound").strip("; "), **kw),
            N(name + "Hi", csv, query, hi, fmt, was=was[2], note=(note + "; upper bound").strip("; "), **kw)]


def RNG(name, csv, query, col, fmt="r2", was=(None, None), note="", fmt_hi=None, **kw) -> list[Num]:
    """Range over the selected rows: macros nameMin, nameMax."""
    return [N(name + "Min", csv, query, col, fmt, agg="min", was=was[0], note=note, **kw),
            N(name + "Max", csv, query, col, fmt_hi or fmt, agg="max", was=was[1], note=note, **kw)]


# ------------------------------------------------------------------------------------------- queries
HBN = "fit_id == 'family_hb_nuts'"
GLOB_HO = "universe == 'global' and eval_subset == 'all_evaluable'"
ROB_SRC = ("scope == 'all_tissues' and protocol == 'source' and subset == '{sub}' and metric == '{met}' and "
           "estimator == '{est}' and reference == 'family_hb'")
ROB_EM = ("scope == '{sc}' and protocol == 'source' and subset == '{sub}' and metric == '{met}' and "
          "estimator {est} and reference == 'em'")
RANK_SRC = "tissue == 'all_tissues' and protocol == 'source' and subset == '{sub}' and metric == 'f1' and estimator == '{est}'"
GOV_P = "scope == 'dual' and tier == 'Promoted' and reference == '{ref}' and universe == '{u}'"
REP_U = "universe == 'global' and level == 'study' and comparator == 'universe' and unit == '{u}'"
MS = "universe == '{u}' and simulator == '{sim}' and stratum == '{st}' and quantity == '{q}'"
FAITHFUL = "simulator in ['faithful_cc', 'faithful_cc_strat']"


def specs() -> list[Num]:
    S: list[Num] = []
    add = S.extend

    # ======================================================================================= design
    add([
        C("nTopK", "PRESPECIFIED:topk_per_tf", "int", note="predicted set: top-k targets per TF", was="50"),
        C("nEvalK", "PRESPECIFIED:eval_k[1]", "int", note="k of Enr@k / P@k reported in the text", was="500"),
        C("nBfdrStrong", "PRESPECIFIED:bfdr_strong", "r2", note="Strong tier BFDR threshold", was="0.05"),
        C("nBfdrModerate", "PRESPECIFIED:bfdr_moderate", "r2", note="Moderate tier BFDR threshold", was="0.20"),
        C("nRepEndpoint", "PRESPECIFIED:replication_endpoint_enrichment", "r1", was="1.5"),
        C("nRepAlpha", "PRESPECIFIED:replication_alpha", "r2", was="0.05"),
        C("nRepTrueEnrFold", "PRESPECIFIED:replication_true_enrichment", "fold", was="twofold"),
        C("nRepPowerPct", "PRESPECIFIED:replication_power", "pct0", was="80"),
        C("nDeMinCells", "PRESPECIFIED:de_min_cells", "int", was="20"),
        C("nDeMinFracPct", "PRESPECIFIED:de_min_target_frac_expressed", "pct0", was="5"),
        C("nDeFdr", "PRESPECIFIED:de_fdr", "r2", was="0.05"),
        C("nDeMinLfc", "PRESPECIFIED:de_min_abs_log2fc", "r2", was="0.25"),
        C("nExprCells", "PRESPECIFIED:expr_n_cells", "int", was="3,000"),
        C("nFmMinCoobs", "PRESPECIFIED:fm_min_coobs_sensitivity", "word", was="five"),
        C("nTopKSensLo", "PRESPECIFIED:topk_per_tf_sensitivity[0]", "int", note="smaller predicted-set size (sensitivity)"),
        C("nTopKSensHi", "PRESPECIFIED:topk_per_tf_sensitivity[1]", "int", note="larger predicted-set size (sensitivity)"),
        C("nEncStrictMinLines", "PRESPECIFIED:encode_strict_min_cell_lines", "word", was="two"),
        C("nGradTol", "code:src/r2/fit_primary.py:GRAD_TOL", "pow10", note="MAP convergence: max|gradient| threshold",
          was="10^-6"),
        C("nRhatFlag", "code:src/r2/fit_primary.py:RHAT_FLAG", "r2", note="NUTS site flag: split R-hat above this value",
          was="1.01"),
        N("nMapRestartsHb", "fit_convergence.csv", "fit_id == 'family_hb'", "n_restarts", "int", was="50",
          note="same in every universe"),
        N("nMapRestartsOther", "fit_convergence.csv", "inference == 'MAP' and fit_id != 'family_hb'", "n_restarts",
          "int", was="20", note="the four comparison estimators, every universe"),
        N("nNutsChains", "fit_convergence.csv", HBN, "num_chains", "word", was="four"),
        N("nNutsWarmup", "fit_convergence.csv", HBN, "num_warmup", "int", was="1,000"),
        N("nNutsDraws", "fit_convergence.csv", HBN, "num_samples", "int", was="1,000"),
        N("nNutsTargetAccept", "fit_convergence.csv", HBN, "target_accept", "r2", was="0.95"),
        N("nCvFolds", "fit_cv_loglik.csv", "config == 'em'", "n_folds", "word", was="five"),
        N("nRepNPriors", "rep_prior_sensitivity.csv",
          "level == 'study' and parameter == 'mu' and analysis_status == 'prespecified'", "profile", "word",
          agg="count", was="eight", note="prespecified prior profiles of the pooled replication model"),
    ])

    # ======================================================================================= data
    add([
        N("nNAnnotHeldin", "source_table.csv", "held_out == False", "source_id", "int", agg="count", was="21"),
        N("nNAnnotLit", "source_table.csv", "family == 'LIT'", "source_id", "int", agg="count", was="18"),
        N("nCohN", "cohort_table.csv", None, "cohort_id", "word", agg="count", was="seven"),
        N("nCohNCap", "cohort_table.csv", None, "cohort_id", "Word", agg="count", was="Seven"),
        N("nCohStudies", "cohort_table.csv", None, "study_id", "word", agg="nunique", was="six"),
        N("nCohStudiesCap", "cohort_table.csv", None, "study_id", "Word", agg="nunique", was="Six"),
    ])
    for t, lab, was in (("immune", "Imm", "20,000"), ("kidney", "Kid", "11,376"), ("lung", "Lung", "65,847")):
        add([N(f"nTsCells{lab}", "fm_extraction_cells.csv", f"tissue == '{t}' and model == 'expression'",
               "cell_selection", "int", regex=r"over (\d+) cells", was=was,
               note="cells of the source Tabula Sapiens file (parsed from the documented sampling rule)")])
    add([
        N("nTsDonorsKid", "fm_extraction_cells.csv", "tissue == 'kidney' and model == 'expression'", "n_donors",
          "word", was="one"),
    ])
    GA = "universe == 'global' and variant == 'prespecified_cc' and annotator == '__ALL__'"
    add([
        N("nGlobTfs", "evidence_summary.csv", GA, "n_tfs", "int", was="935"),
        N("nGlobGenes", "evidence_summary.csv", GA, "n_genes", "int", was="9,963"),
        N("nGlobEdges", "evidence_summary.csv", GA, "n_edges", "int", was="9,314,634"),
        N("nGlobEdgesMio", "evidence_summary.csv", GA, "n_edges", "r1", scale=1e-6, note="millions", was="9.3"),
        N("nGlobCoverage", "evidence_summary.csv", GA, "coverage", "r3", helper=True,
          note="share of global edges observed by at least one annotator"),
        E("nGlobUnobsPct", "1 - {nGlobCoverage}", "pct0", note="share observed by no annotator", was="45"),
        N("nGlobReported", "evidence_summary.csv", GA, "n_pos", "int", was="43,444"),
        N("nGlobPatterns", "evidence_summary.csv", GA, "n_patterns", "int", was="20,278"),
        N("nScgptCells", "fm_extraction_cells.csv", "model == 'scgpt'", "n_cells", "int", agg="same", was="2,000"),
        N("nGfCellsImmLung", "fm_extraction_cells.csv", "model == 'geneformer' and tissue in ['immune', 'lung']",
          "n_cells", "int", agg="same", was="1,200"),
        N("nGfCellsKid", "fm_extraction_cells.csv", "model == 'geneformer' and tissue == 'kidney'", "n_cells", "int",
          was="2,000"),
        N("nScgptSmartImm", "fm_extraction_cells.csv", "model == 'scgpt' and tissue == 'immune'", "assay_method",
          "int", regex=r"smartseq:(\d+)", helper=True),
        N("nScgptSmartKid", "fm_extraction_cells.csv", "model == 'scgpt' and tissue == 'kidney'", "assay_method",
          "int", regex=r"smartseq:(\d+)", helper=True),
        N("nScgptSmartLung", "fm_extraction_cells.csv", "model == 'scgpt' and tissue == 'lung'", "assay_method",
          "int", regex=r"smartseq:(\d+)", helper=True),
        E("nScgptSmartPctImm", "{nScgptSmartImm} / {nScgptCells}", "pct0", note="share of Smart-seq2 cells, scGPT, immune"),
        E("nScgptSmartPctKid", "{nScgptSmartKid} / {nScgptCells}", "pct0", note="share of Smart-seq2 cells, scGPT, kidney"),
        E("nScgptSmartPctLung", "{nScgptSmartLung} / {nScgptCells}", "pct0", note="share of Smart-seq2 cells, scGPT, lung"),
        N("nScgptDonorsLung", "fm_extraction_cells.csv", "model == 'scgpt' and tissue == 'lung'", "n_donors", "word"),
        N("nSmartGfImm", "fm_extraction_cells.csv", "model == 'geneformer' and tissue == 'immune'", "assay_method", "int",
          regex=r"smartseq:(\d+)", helper=True),
        N("nCellsGfImm", "fm_extraction_cells.csv", "model == 'geneformer' and tissue == 'immune'", "n_cells", "int", helper=True),
        N("nSmartGfKid", "fm_extraction_cells.csv", "model == 'geneformer' and tissue == 'kidney'", "assay_method", "int",
          regex=r"smartseq:(\d+)", helper=True),
        N("nCellsGfKid", "fm_extraction_cells.csv", "model == 'geneformer' and tissue == 'kidney'", "n_cells", "int", helper=True),
        N("nSmartGfLung", "fm_extraction_cells.csv", "model == 'geneformer' and tissue == 'lung'", "assay_method", "int",
          regex=r"smartseq:(\d+)", helper=True),
        N("nCellsGfLung", "fm_extraction_cells.csv", "model == 'geneformer' and tissue == 'lung'", "n_cells", "int", helper=True),
        N("nSmartExImm", "fm_extraction_cells.csv", "model == 'expression' and tissue == 'immune'", "assay_method", "int",
          regex=r"smartseq:(\d+)", helper=True),
        N("nCellsExImm", "fm_extraction_cells.csv", "model == 'expression' and tissue == 'immune'", "n_cells", "int", helper=True),
        N("nSmartExKid", "fm_extraction_cells.csv", "model == 'expression' and tissue == 'kidney'", "assay_method", "int",
          regex=r"smartseq:(\d+)", helper=True),
        N("nCellsExKid", "fm_extraction_cells.csv", "model == 'expression' and tissue == 'kidney'", "n_cells", "int", helper=True),
        N("nSmartExLung", "fm_extraction_cells.csv", "model == 'expression' and tissue == 'lung'", "assay_method", "int",
          regex=r"smartseq:(\d+)", helper=True),
        N("nCellsExLung", "fm_extraction_cells.csv", "model == 'expression' and tissue == 'lung'", "n_cells", "int", helper=True),
        E("nOtherSmartPctMax", "max({nSmartGfImm} / {nCellsGfImm}, {nSmartGfKid} / {nCellsGfKid}, {nSmartGfLung} / {nCellsGfLung}, {nSmartExImm} / {nCellsExImm}, {nSmartExKid} / {nCellsExKid}, {nSmartExLung} / {nCellsExLung})", "ceil0", scale=100.0,
          note="largest Smart-seq2 share among the randomly sampled Geneformer and expression cells (percent, rounded up)"),
        N("nCoobsScMin", "fm_universe_summary.csv", None, "median_scgpt_coobs", "int", agg="min", was="1"),
        N("nCoobsScMax", "fm_universe_summary.csv", None, "median_scgpt_coobs", "int", agg="max", was="7"),
        N("nCoobsGfMax", "fm_universe_summary.csv", None, "median_geneformer_coobs", "int", agg="max",
          was="below 1", note="median co-observing cells, Geneformer, largest over tissues"),
        *RNG("nCoobsFiveScPct", "fm_universe_summary.csv", None, "frac_scgpt_coobs_ge5", "pct0",
             note="share of all universe pairs co-observed in at least five cells, scGPT",
             require="{nFmMinCoobs} == 5", require_note="column frac_*_coobs_ge5 is the co-observation threshold"),
        *RNG("nCoobsFiveGfPct", "fm_universe_summary.csv", None, "frac_geneformer_coobs_ge5", "pct0",
             note="share of all universe pairs co-observed in at least five cells, Geneformer"),
        N("nEncTfs", "heldout_chip_table.csv", "key == 'encode_tfbs'", "n_tfs", "int", was="114"),
        N("nEncDensityPct", "heldout_chip_table.csv", "key == 'encode_tfbs'", "density", "pct0", was="36"),
        N("nEnetTfs", "heldout_chip_table.csv", "key == 'enets2'", "n_tfs", "int", was="86"),
        N("nEnetDensityPct", "heldout_chip_table.csv", "key == 'enets2'", "density", "pct1", was="1.3"),
        N("nPythonVersion", "pipeline_environment.csv", "key == 'python'", "value", "r2", regex=r"^(\d+\.\d+)",
          was="3.11"),
        N("nHwCores", "pipeline_environment.csv", "key == 'hw.ncpu'", "value", "int", was="10"),
        N("nHwRamGiB", "pipeline_environment.csv", "key == 'RAM (GiB)'", "value", "int", was="32"),
    ])

    # ======================================================================================= 5.1 fit
    add([
        N("nMapGradMax", "fit_convergence.csv", "inference == 'MAP'", "grad_norm", "e2", agg="max", was="7.3e-10"),
        N("nMapHessMin", "fit_convergence.csv", "inference == 'MAP'", "hessian_min_eig", "r3", agg="min", was="0.104"),
        *RNG("nNutsRhatId", "fit_convergence.csv", HBN, "r_hat_max_identified", "r3", was=("1.003", "1.006")),
        *RNG("nNutsEssId", "fit_convergence.csv", HBN, "ess_bulk_min_identified", "int", was=("1,676", "1,951")),
        N("nNutsDivMax", "fit_convergence.csv", HBN, "n_divergent", "word", agg="max", was="one"),
        N("nNutsRhatAllGlob", "fit_convergence.csv", f"universe == 'global' and {HBN}", "r_hat_max", "r3", was="1.016"),
        N("nNutsEssAllGlob", "fit_convergence.csv", f"universe == 'global' and {HBN}", "ess_bulk_min", "int",
          was="300"),
        N("nMvnInsideMin", "fit_map_vs_nuts.csv", None, "frac_edges_map_inside_cri", "r3", agg="min", helper=True,
          require="{nMvnInsideMin} == 1",
          require_note="'the posterior-mode credibility of every edge lies inside its 95% credible interval'"),
        N("nMvnSpMin", "fit_map_vs_nuts.csv", None, "spearman_w", "floor3", agg="min", was="0.999",
          note="'at least': rounded down"),
        *RNG("nMvnJac", "fit_map_vs_nuts.csv", None, "jaccard_bfdr0.05", "r3", was=("0.947", "0.998")),
        N("nNutsStradGlob", "fit_map_vs_nuts.csv", "universe == 'global'", "n_edges_cri_straddles_0.5_mean_gt_0.5",
          "int", was="602"),
        N("nNutsGtHalfGlob", "fit_map_vs_nuts.csv", "universe == 'global'", "n_edges_nuts_mean_gt_0.5", "int",
          was="6,816"),
        N("nNutsStradImm", "fit_map_vs_nuts.csv", "universe == 'immune'", "n_edges_cri_straddles_0.5_mean_gt_0.5",
          "int", was="96"),
        N("nNutsGtHalfImm", "fit_map_vs_nuts.csv", "universe == 'immune'", "n_edges_nuts_mean_gt_0.5", "int",
          was="482"),
        # two modes
        N("nHbNoptGlobImm", "fit_convergence.csv", "fit_id == 'family_hb' and universe in ['global', 'immune']",
          "n_distinct_optima", "word", was="two"),
        N("nHbNoptKid", "fit_convergence.csv", "fit_id == 'family_hb' and universe == 'kidney'", "n_distinct_optima",
          "word", was="three"),
        N("nHbNoptLung", "fit_convergence.csv", "fit_id == 'family_hb' and universe == 'lung'", "n_distinct_optima",
          "word", was="four"),
        *RNG("nHbSpTwo", "fit_convergence.csv", "fit_id == 'family_hb'", "second_optimum_spearman_w", "r2",
             was=("0.76", "0.87")),
        N("nHbJacTwoMax", "fit_map_optima.csv", "fit_id == 'family_hb' and optimum == 1", "jaccard_bfdr0_05_vs_best",
          "ceil3", agg="max", was="0.002", note="'at most': rounded up"),
        N("nHbGapGlob", "fit_convergence.csv", "fit_id == 'family_hb' and universe == 'global'",
          "gap_to_next_optimum", "r0", was="108"),
        N("nHbGapImm", "fit_convergence.csv", "fit_id == 'family_hb' and universe == 'immune'",
          "gap_to_next_optimum", "r1"),
        *RNG("nHbGapTis", "fit_convergence.csv", "fit_id == 'family_hb' and universe != 'global'",
             "gap_to_next_optimum", "r1", was=("7.5", "11.6")),
    ])
    MB = "role == 'primary_estimate' and mode == 1 and universe == '{u}'"
    add([
        N("nBridgePGlob", "modes_bridge.csv", MB.format(u="global"), "p_bridge", "e2", was="1.8e-46"),
        N("nBridgePImm", "modes_bridge.csv", MB.format(u="immune"), "p_bridge", "e2", was="4.4e-3"),
        N("nBridgePImmLo", "modes_bridge.csv", MB.format(u="immune"), "p2_bridge_lo", "e2", was="2.8e-3",
          note="Monte Carlo 95% interval of the bridge estimate"),
        N("nBridgePImmHi", "modes_bridge.csv", MB.format(u="immune"), "p2_bridge_hi", "e2", was="6.9e-3",
          note="Monte Carlo 95% interval of the bridge estimate"),
        N("nBridgePImmRangeLo", "modes_bridge.csv", MB.format(u="immune"), "p2_range_draw_sets_proposals_lo", "e2",
          note="range over posterior draw sets and bridge proposals (added: modes2_VERIFY m4)", helper=True),
        N("nBridgePImmRangeHi", "modes_bridge.csv", MB.format(u="immune"), "p2_range_draw_sets_proposals_hi", "e2",
          note="range over posterior draw sets and bridge proposals (added: modes2_VERIFY m4)", helper=True),
        N("nBridgePKid", "modes_bridge.csv", MB.format(u="kidney"), "p_bridge", "e2", was="2.9e-3"),
        N("nBridgePKidRangeLo", "modes_bridge.csv", MB.format(u="kidney"), "p2_range_draw_sets_proposals_lo", "e2",
          note="range over posterior draw sets and bridge proposals", helper=True),
        N("nBridgePKidRangeHi", "modes_bridge.csv", MB.format(u="kidney"), "p2_range_draw_sets_proposals_hi", "e2",
          note="range over posterior draw sets and bridge proposals", helper=True),
        N("nBridgePLung", "modes_bridge.csv", MB.format(u="lung"), "p_bridge", "e2", was="6.3e-5"),
        N("nBridgeNcReMin", "modes_bridge.csv", "role == 'raw_noncentred_coordinates'", "bridge_re_fs", "floor1",
          agg="min", note="smallest relative error of the bridge estimate in the raw non-centred coordinates (post hoc check)"),
        N("nBridgePTisLgMin", "modes_bridge.csv", "role == 'primary_estimate' and mode == 1 and universe != 'global'",
          "p_bridge", "lgfloor", agg="min", was="10^-5", note="order of magnitude, rounded down"),
        N("nBridgePTisLgMax", "modes_bridge.csv", "role == 'primary_estimate' and mode == 1 and universe != 'global'",
          "p_bridge", "lgceil", agg="max", was="10^-2", note="order of magnitude, rounded up"),
        *RNG("nRandRhat", "fit_convergence.csv", "fit_id == 'family_hb_randinit_nuts'", "r_hat_max", "r2",
             was=("1.56", "1.74")),
    ])
    MG = "scenario == '{sc}' and global_rule == 'own_estimate'"
    add([
        N("nMgJacBridgeMin", "modes_governance_sensitivity.csv", MG.format(sc="bridge"), "jaccard_context_strong",
          "floor3", agg="min", was="0.976", note="'at least': rounded down"),
        *RNG("nMgStrongFive", "modes_governance_sensitivity.csv", MG.format(sc="fixed_0.05"), "context_strong", "int",
             was=("26", "36")),
        N("nMgPFive", "modes_governance_sensitivity.csv", MG.format(sc="fixed_0.05"), "p2_context", "r2", agg="same",
          was="0.05"),
        N("nMgPromFiveMin", "modes_governance_sensitivity.csv", MG.format(sc="fixed_0.05"), "jaccard_promoted",
          "floor2", agg="min", was="0.95", note="'at least': rounded down (0.946)"),
        N("nMgPromFiveSameMin", "modes_governance_sensitivity.csv",
          "scenario == 'fixed_0.05' and global_rule == 'same_p2'", "jaccard_promoted", "floor2", agg="min",
          note="'at least', rounded down: the global fit mixed at the same second-mode probability"),
    ])
    # design-faithful mode-selection simulation (modes2; post hoc)
    MSA = "universe == '{u}' and simulator == 'faithful_cc' and stratum == '{st}' and quantity == '{q}'"
    add([
        *CI("nSimPdomImm", "modes_simulation.csv",
            MS.format(u="immune", sim="faithful_cc", st="all", q="p_dominant_is_truth"), "value", "ci_lo", "ci_hi",
            "r2", note="Wilson 95%"),
        N("nSimNImm", "modes_simulation.csv", MS.format(u="immune", sim="faithful_cc", st="all",
                                                         q="p_dominant_is_truth"), "n_replicates", "int"),
        E("nSimMissFwd", "1 - {nSimPdomImm}", "r2", note="misselection rate, truth = real dominant optimum"),
        N("nSimLapFwd", "modes_simulation.csv", MS.format(u="immune", sim="faithful_cc", st="all", q="laplace_p2_mean"),
          "value", "r2", note="mean Laplace second-mode probability, truth = real dominant optimum",
          require="{nSimLapFwd} > {nSimMissFwd}", require_note="'Laplace slightly overstates the risk' (forward)"),
        N("nSimPdomGlob", "modes_simulation.csv", MS.format(u="global", sim="faithful_cc", st="all",
                                                             q="p_dominant_is_truth"), "value", "r2", helper=True),
        N("nSimNGlob", "modes_simulation.csv", MS.format(u="global", sim="faithful_cc", st="all",
                                                          q="p_dominant_is_truth"), "n_replicates", "int"),
        E("nSimCorrectGlob", "{nSimPdomGlob} * {nSimNGlob}", "int",
          note="replicates whose dominant optimum is the truth (probability x replicates)"),
        *CI("nSimPdomRev", "modes_simulation.csv",
            MS.format(u="immune_truthmode1", sim="faithful_cc", st="all", q="p_dominant_is_truth"), "value", "ci_lo",
            "ci_hi", "r2", note="truth = real second optimum; Wilson 95%"),
        N("nSimNRev", "modes_simulation.csv", MS.format(u="immune_truthmode1", sim="faithful_cc", st="all",
                                                         q="p_dominant_is_truth"), "n_replicates", "int"),
        E("nSimMissRev", "1 - {nSimPdomRev}", "r2", note="misselection rate, truth = real second optimum"),
        N("nSimLapRev", "modes_simulation_replicates.csv", "universe == 'immune_truthmode1' and simulator == 'faithful_cc'",
          "laplace_p2", "r2", agg="mean", note="mean Laplace second-mode probability over the replicates",
          require="{nSimLapRev} < {nSimMissRev}", require_note="'Laplace understates the risk' (reverse)"),
        N("nSimNFaithful", "modes_simulation_replicates.csv", FAITHFUL, "replicate", "int", agg="count",
          note="replicates of both design-faithful simulators, both truth directions"),
        N("nSimWrongGapMax", "modes_simulation_replicates.csv", f"{FAITHFUL} and dominant_is_truth == False", "gap",
          "r1", agg="max", require="{nSimWrongGapMax} < {nHbGapTisMin}",
          require_note="'smaller than every real gap'"),
        N("nSimFixedWrongGapMax", "modes_simulation_replicates.csv",
          "simulator == 'fixed_masks' and dominant_is_truth == False and truth_mode_found == False", "gap", "r0",
          agg="max", require="{nSimFixedWrongGapMax} > {nHbGapImm}",
          require_note="fixed-mask simulator: a wrong selection at a gap larger than the real immune gap"),
        N("nSimFixedWrongN", "modes_simulation_replicates.csv",
          "simulator == 'fixed_masks' and dominant_is_truth == False and gap > {nHbGapImm}", "replicate", "word",
          agg="count"),
        N("nSimNGeReal", "modes_simulation_replicates.csv", FAITHFUL + " and gap >= {nHbGapImm}", "replicate", "int",
          agg="count", note="replicates with a gap at least the real immune gap"),
        N("nSimNGeRealCorrect", "modes_simulation_replicates.csv",
          FAITHFUL + " and gap >= {nHbGapImm} and dominant_is_truth == True", "replicate", "int", agg="count",
          helper=True, require="{nSimNGeRealCorrect} == {nSimNGeReal}",
          require_note="'all selected the true optimum'"),
        N("nSimNGeRealImmCc", "modes_simulation.csv", MSA.format(u="immune", st="gap_ge_real_immune",
                                                                  q="p_dominant_is_truth"), "n_replicates", "int",
          note="primary simulator, truth = dominant optimum: replicates with gap >= real immune gap",
          require="{nSimNGeRealImmCc} + {nSimNGeRealRevCc} < 0.5 * {nSimNGeReal}",
          require_note="'most of these come from the global design or the stratum-weighted variant'"),
        N("nSimNGeRealRevCc", "modes_simulation.csv", MSA.format(u="immune_truthmode1", st="gap_ge_real_immune",
                                                                  q="p_dominant_is_truth"), "n_replicates", "int",
          helper=True, require="{nSimNGeRealRevCc} == 0",
          require_note="'none of the replicates generated from the second optimum reached the real immune gap'"),
        *CI("nSimFwdGe", "modes_simulation.csv", MSA.format(u="immune", st="real_gap_immune",
                                                             q="p_truth_beats_other_mode_by_ge_real_gap"),
            "value", "ci_lo", "ci_hi", "r2", note="truth = dominant optimum: truth wins by >= real immune gap; Wilson"),
        N("nSimRevGeHi", "modes_simulation.csv", MSA.format(u="immune_truthmode1", st="real_gap_immune",
                                                             q="p_other_mode_beats_truth_by_ge_real_gap"),
          "ci_hi", "r2", note="truth = second optimum: Wilson upper bound of P(dominant-like optimum wins by >= real gap)",
          require="{nSimFwdGeLo} < {nSimRevGeHi} < {nSimFwdGeHi}",
          require_note="'the upper bound lies inside the forward interval' (weak evidence)"),
        N("nSimGapPctImm", "modes_simulation.csv",
          MS.format(u="immune", sim="faithful_cc", st="real_gap_immune", q="frac_sim_gap_below_real_gap_all"),
          "value", "ordpct", note="percentile of the real immune gap among simulated immune-design gaps"),
        N("nSimGapPctImmStrat", "modes_simulation.csv",
          MS.format(u="immune", sim="faithful_cc_strat", st="real_gap_immune", q="frac_sim_gap_below_real_gap_all"),
          "value", "ordpct", note="the same under stratum-dependent class weights"),
        N("nSimFdrWrong", "modes_simulation.csv",
          "stratum == 'dominant_not_truth' and quantity == 'dominant_strong_realised_fdr_mean' and "
          "simulator == 'faithful_cc'", "value", "r2", agg="same", was="0.93--0.95",
          note="design-faithful simulator, both truth directions (print the same)"),
        *RNG("nSimFdrRight", "modes_simulation.csv",
             "stratum == 'dominant_is_truth' and quantity == 'dominant_strong_realised_fdr_mean' and "
             "simulator == 'faithful_cc'", "value", "r2", note="right mode, three simulated designs"),
        N("nSimClaimed", "modes_simulation.csv",
          "stratum == 'dominant_is_truth' and quantity == 'dominant_strong_claimed_bfdr_mean' and "
          "simulator in ['faithful_cc', 'faithful_cc_strat']", "value", "r2", agg="same",
          note="claimed BFDR of the Strong tier (print the same)"),
        N("nSimFdrStratImm", "modes_simulation.csv",
          MS.format(u="immune", sim="faithful_cc_strat", st="dominant_is_truth", q="dominant_strong_realised_fdr_mean"),
          "value", "r3", note="right mode, stratum-dependent class weights, immune design"),
        N("nSimFdrStratGlob", "modes_simulation.csv",
          MS.format(u="global", sim="faithful_cc_strat", st="dominant_is_truth", q="dominant_strong_realised_fdr_mean"),
          "value", "r3", note="right mode, stratum-dependent class weights, global design"),
    ])
    LR = "universe == 'global' and fit_id == 'family_hb_nuts' and parameter == '{p}' and annotator == '{a}'"
    for a, lab, was in (("SIGNOR", "Signor", ("562", "498", "635")), ("TRRUST", "Trrust", ("472", "443", "504")),
                        ("HTRIdb", "Htridb", ("250", "226", "278")), ("NTNU.Curated", "Ntnu", ("139", "128", "149"))):
        add(CI(f"nLr{lab}", "fit_source_parameters.csv", LR.format(p="lr_pos_marginal", a=a), "value", "lo", "hi",
               "r0", was=was, note="marginal LR+ (global NUTS), 95% CrI"))
    add([
        N("nLrNtnuRank", "fit_source_parameters.csv", LR.format(p="lr_pos_marginal", a="NTNU.Curated"), "rank", "ord",
          was="sixth"),
        *CI("nAgNtnu", "fit_source_parameters.csv", LR.format(p="alpha_over_gamma", a="NTNU.Curated"), "value", "lo",
            "hi", "r2", was=("0.56", "0.52", "0.60")),
        N("nNtnuExtriBoth", "fit_coreporting_assoc.csv",
          "universe == 'global' and annotator_j == 'ExTRI' and annotator_k == 'NTNU.Curated'", "n11", "int",
          helper=True),
        N("nNtnuOnly", "fit_coreporting_assoc.csv",
          "universe == 'global' and annotator_j == 'ExTRI' and annotator_k == 'NTNU.Curated'", "n01", "int",
          helper=True),
        E("nNtnuInExtriPct", "{nNtnuExtriBoth} / ({nNtnuExtriBoth} + {nNtnuOnly})", "pct0", was="90",
          note="share of NTNU.Curated reports (co-observed edges, global) also reported by ExTRI"),
        N("nStrongEmGlob", "fit_map_optima.csv", "fit_id == 'em' and optimum == 0 and universe == 'global'",
          "n_bfdr0_05", "int", was="18,873"),
        N("nStrongHbGlob", "fit_map_optima.csv", "fit_id == 'family_hb' and optimum == 0 and universe == 'global'",
          "n_bfdr0_05", "int", was="4,064"),
        E("nStrongRatioGlob", "{nStrongEmGlob} / {nStrongHbGlob}", "r1", was="4.6"),
    ])
    REC = "kind == 'realised_fdr' and estimator == '{e}' and quantity == 'strong_realised_fdr' and simulator == 'faithful_cc'"
    RECM = ("kind == 'realised_fdr_by_mode' and estimator == 'family_hb' and quantity == 'strong_realised_fdr' and "
            "simulator == 'faithful_cc' and mode == '{m}'")
    RECN = ("kind == 'realised_fdr' and estimator == 'family_hb' and quantity == 'strong_realised_fdr' and "
            "simulator == 'faithful_cc' and universe == '{u}'")
    add([
        *RNG("nRecFdrEm", "fit_recovery.csv", REC.format(e="em"), "value_mean", "r2", was=("0.57", "0.60")),
        N("nRecFdrHbTruth", "fit_recovery.csv", RECM.format(m="truth"), "value_mean", "r2", agg="same",
          note="Family-HB, replicates whose fit selected the true mode (immune and global print the same)"),
        N("nRecFdrHbImm", "fit_recovery.csv", RECN.format(u="immune"), "value_mean", "r2",
          note="Family-HB, all immune-design replicates"),
        N("nRecNCompImm", "fit_recovery.csv", RECM.format(m="competing") + " and universe == 'immune'", "n_replicates",
          "int", note="immune-design replicates whose fit selected the competing mode"),
        N("nRecNImm", "fit_recovery.csv", RECN.format(u="immune"), "n_replicates", "int"),
        N("nRecNGlob", "fit_recovery.csv", RECN.format(u="global"), "n_replicates", "int"),
        N("nRecNTruthGlob", "fit_recovery.csv", RECM.format(m="truth") + " and universe == 'global'", "n_replicates",
          "int", helper=True, require="{nRecNTruthGlob} == {nRecNGlob}",
          require_note="'the global value has no competing-mode replicate'"),
    ])
    # simulator validation (Methods): informative observations carried by the simulated data
    DES = ("quantity == 'n_edges_cc_observed_with_lit_report' and simulator == '{s}' and annotator.isnull()")
    add([
        *RNG("nSimInfoPct", "modes_faithful_design.csv", DES.format(s="faithful_cc"), "ratio_sim_mean_to_real", "pct0",
             note="share of the real motif/coexpression-observed edges with a literature report, design-faithful"),
        *RNG("nSimInfoStratPct", "modes_faithful_design.csv", DES.format(s="faithful_cc_strat"),
             "ratio_sim_mean_to_real", "pct0", note="the same with stratum-dependent class weights"),
    ])
    CV = "config == 'em' and universe == '{u}'"
    add([
        N("nCvEmGlob", "fit_cv_loglik.csv", CV.format(u="global"), "diff_vs_family_hb_per_edge_mean", "e2", scale=-1.0,
          was="6.3e-4", note="EM minus Family-HB, sign reversed ('lower by')"),
        N("nCvEmGlobSe", "fit_cv_loglik.csv", CV.format(u="global"), "diff_vs_family_hb_per_edge_se", "e2",
          was="5.7e-5"),
        N("nCvEmImm", "fit_cv_loglik.csv", CV.format(u="immune"), "diff_vs_family_hb_per_edge_mean", "e2", scale=-1.0,
          was="1.9e-3", note="EM minus Family-HB, sign reversed ('lower by')"),
        N("nCvEmImmSe", "fit_cv_loglik.csv", CV.format(u="immune"), "diff_vs_family_hb_per_edge_se", "e2",
          was="5.2e-4"),
        N("nCvFoldsTotal", "fit_cv_loglik.csv", "config == 'em'", "n_folds", "int", agg="sum", helper=True),
        N("nCvEmFoldsEmBetter", "fit_cv_loglik.csv", "config == 'em'", "n_folds_better_than_family_hb", "int",
          agg="sum", helper=True),
        E("nCvFoldsHbBetter", "{nCvFoldsTotal} - {nCvEmFoldsEmBetter}", "word", was="ten",
          note="folds (both universes) in which Family-HB predicts held-out reports better than EM",
          require="{nCvEmFoldsEmBetter} == 0", require_note="'in all ... folds'"),
    ])
    PPC = ("universe == 'global' and estimator == 'family_hb' and statistic == "
           "'litblock_anyreport_frac_by_n_observing' and stratum == '{s}' and grouping == 'modality' and "
           "kind == 'statistic'")
    add([
        N("nPpcObsOne", "fit_ppc.csv", PPC.format(s="1"), "observed", "sig2", was="0.0024"),
        N("nPpcRepOne", "fit_ppc.csv", PPC.format(s="1"), "rep_mean", "sig2", was="0.0059"),
        N("nPpcObsEight", "fit_ppc.csv", PPC.format(s=">=8"), "observed", "sig2", was="0.14"),
        N("nPpcRepEight", "fit_ppc.csv", PPC.format(s=">=8"), "rep_mean", "sig2", was="0.031"),
    ])

    # ======================================================================================= 5.2 robustness
    MSF = "in_ranking_panel == True and estimator == 'family_hb' and tissue {t}"
    add([
        N("nRobNMethodsImm", "rob_method_scores_full.csv", MSF.format(t="== 'immune'"), "method", "word",
          agg="nunique", was="twelve", note="immune ranking panel"),
        N("nRobNMethodsKidLung", "rob_method_scores_full.csv", MSF.format(t="in ['kidney', 'lung']"), "method",
          "word", agg="nunique", note="kidney and lung ranking panels (no GENIE3)",
          require="{nRobNMethodsKidLung} == {nRobNMethodsImm} - 1",
          require_note="'twelve (immune) or eleven (kidney, lung)': the two smaller panels have one method fewer"),
        N("nRobNconf", "rob_method_ranking.csv", RANK_SRC.format(sub="all", est="family_hb"), "n_configs", "int",
          was="63"),
    ])
    for est, lab, was in (("union", "Union", "0.005"), ("vote_fraction", "Vote", "0.007"), ("em", "Em", "0.008"),
                          ("majority", "Maj", "0.014"), ("family_hb", "Hb", "0.017"), ("dependence", "Dep", "0.017"),
                          ("stouffer", "Sto", "0.019"), ("fisher", "Fis", "0.022"), ("multifactor", "Mf", "0.058")):
        add([N(f"nRobD{lab}", "rob_method_ranking.csv", RANK_SRC.format(sub="all", est=est), "mean_D", "r3", was=was)])
    # paired against EM (post hoc, source-clustered unadjusted 95% intervals; module assets2)
    EMD = dict(sc="all_tissues", met="D_f1")
    add([
        # Family-HB minus EM is printed as the negated EM-minus-Family-HB contrast of tab_robustness (the
        # prespecified Family-HB-referenced row), so that the prose and Table 7 report one interval
        N("nRobCtHbEm", "rob_estimator_contrasts.csv", ROB_SRC.format(sub="all", met="D_f1", est="em"),
          "mean_diff", "r3", scale=-1.0,
          note="Family-HB minus EM, D, 63 source configurations (= -(EM minus Family-HB) of tab_robustness)"),
        N("nRobCtHbEmLo", "rob_estimator_contrasts.csv", ROB_SRC.format(sub="all", met="D_f1", est="em"),
          "mean_diff_cluster_hi", "sig1", scale=-1.0,
          note="lower bound = -(upper bound of EM minus Family-HB); one significant digit: it is close to 0",
          require="{nRobCtHbEmLo} > 0", require_note="'Family-HB is slightly less stable than EM' over all configurations"),
        N("nRobCtHbEmHi", "rob_estimator_contrasts.csv", ROB_SRC.format(sub="all", met="D_f1", est="em"),
          "mean_diff_cluster_lo", "r3", scale=-1.0, note="upper bound = -(lower bound of EM minus Family-HB)"),
        N("nRobCtHbEmRaw", "rob_estimator_contrasts.csv", ROB_EM.format(sub="all", est="== 'family_hb'", **EMD),
          "mean_diff", "r4", helper=True),
        *CI("nRobCtDepEm", "rob_estimator_contrasts.csv", ROB_EM.format(sub="all", est="== 'dependence'", **EMD),
            "mean_diff", "mean_diff_cluster_lo", "mean_diff_cluster_hi", "r3",
            require="{nRobCtDepEmLo} > 0", require_note="'the dependence-aware model is less stable than EM'"),
        *CI("nRobCtMfEm", "rob_estimator_contrasts.csv", ROB_EM.format(sub="all", est="== 'multifactor'", **EMD),
            "mean_diff", "mean_diff_cluster_lo", "mean_diff_cluster_hi", "r3",
            require="{nRobCtMfEmLo} > 0", require_note="'the multifactor model is less stable than EM'"),
        *RNG("nRobCtRuleEm", "rob_estimator_contrasts.csv",
             ROB_EM.format(sub="all", est="in ['fisher', 'stouffer', 'majority']", **EMD), "mean_diff", "r3",
             note="Fisher, Stouffer and majority minus EM, D"),
        N("nRobCtRuleEmLoMin", "rob_estimator_contrasts.csv",
          ROB_EM.format(sub="all", est="in ['fisher', 'stouffer', 'majority']", **EMD), "mean_diff_cluster_lo",
          "r3", agg="min", helper=True,
          require="{nRobCtRuleEmLoMin} > 0", require_note="'Fisher, Stouffer and majority are also less stable than EM'"),
        N("nRobCtUvEmLoMax", "rob_estimator_contrasts.csv",
          ROB_EM.format(sub="all", est="in ['union', 'vote_fraction']", **EMD), "mean_diff_cluster_lo", "r3",
          agg="max", helper=True),
        N("nRobCtUvEmHiMin", "rob_estimator_contrasts.csv",
          ROB_EM.format(sub="all", est="in ['union', 'vote_fraction']", **EMD), "mean_diff_cluster_hi", "r3",
          agg="min", helper=True,
          require="{nRobCtUvEmLoMax} < 0 < {nRobCtUvEmHiMin}",
          require_note="'union and vote fraction are not distinguishable from EM' (both intervals contain 0)"),
    ])
    # paired against Family-HB (prespecified contrast family of tab_robustness)
    for est, lab, was in (("union", "Union", ("-0.012", "-0.021", "-0.003")),
                          ("vote_fraction", "Vote", ("-0.009", "-0.018", "-0.002"))):
        add(CI(f"nRobCt{lab}Ns", "rob_estimator_contrasts.csv", ROB_SRC.format(sub="no_fhb_mode_switch", met="D_f1", est=est),
               "mean_diff", "mean_diff_cluster_lo", "mean_diff_cluster_hi", "r3",
               note="minus Family-HB, D, without the Family-HB mode-switch configurations (post hoc)",
               require=f"{{nRobCt{lab}NsHi}} < 0",
               require_note="'union and vote fraction remain more stable than Family-HB on D'"))
    RCD = "protocol == 'source' and estimator == 'family_hb' and universe != 'global' and fhb_mode_switch == {b}"
    add([
        N("nRobNswitch", "rob_method_ranking.csv", RANK_SRC.format(sub="all", est="family_hb"), "n_fhb_mode_switch",
          "word", was="two"),
        N("nRobCtHbEmNsRaw", "rob_estimator_contrasts.csv",
          ROB_EM.format(sub="no_fhb_mode_switch", est="== 'family_hb'", **EMD), "mean_diff", "r4", helper=True),
        E("nRobGapSharePct", "1 - {nRobCtHbEmNsRaw} / {nRobCtHbEmRaw}", "pct0",
          note="share of the Family-HB - EM difference in D removed by excluding the switch configurations",
          require="0.4 <= {nRobGapSharePct} <= 0.6", require_note="'About half'"),
        *RNG("nRobDSwitch", "rob_config_detail.csv", RCD.format(b="True"), "D_f1", "r3",
             note="Family-HB D in the two mode-switch configurations"),
        N("nRobDNoSwitchMax", "rob_config_detail.csv", RCD.format(b="False"), "D_f1", "r3", agg="max",
          require="{nRobDNoSwitchMax} < {nRobDSwitchMin}",
          require_note="'the two with the largest disagreement for Family-HB'"),
        N("nRobNconfNs", "rob_method_ranking.csv", RANK_SRC.format(sub="no_fhb_mode_switch", est="family_hb"),
          "n_configs", "int"),
        N("nRobDHbNs", "rob_method_ranking.csv", RANK_SRC.format(sub="no_fhb_mode_switch", est="family_hb"), "mean_D",
          "r3"),
        N("nRobDEmNs", "rob_method_ranking.csv", RANK_SRC.format(sub="no_fhb_mode_switch", est="em"), "mean_D", "r3"),
        N("nRobDDepNs", "rob_method_ranking.csv", RANK_SRC.format(sub="no_fhb_mode_switch", est="dependence"),
          "mean_D", "r3"),
        N("nRobCtHbEmNs", "rob_estimator_contrasts.csv", ROB_SRC.format(sub="no_fhb_mode_switch", met="D_f1", est="em"),
          "mean_diff", "r3", scale=-1.0,
          note="Family-HB minus EM without the switch configurations (= -(EM minus Family-HB) of tab_robustness)"),
        N("nRobCtHbEmNsLo", "rob_estimator_contrasts.csv",
          ROB_SRC.format(sub="no_fhb_mode_switch", met="D_f1", est="em"), "mean_diff_cluster_hi", "r3", scale=-1.0,
          note="lower bound = -(upper bound of EM minus Family-HB)"),
        N("nRobCtHbEmNsHi", "rob_estimator_contrasts.csv",
          ROB_SRC.format(sub="no_fhb_mode_switch", met="D_f1", est="em"), "mean_diff_cluster_lo", "r3", scale=-1.0,
          note="upper bound = -(lower bound of EM minus Family-HB)",
            require="{nRobCtHbEmNsLo} < 0 < {nRobCtHbEmNsHi}",
            require_note="'Family-HB is not distinguishable from EM' without the switch configurations"),
        *CI("nRobCtDepEmNs", "rob_estimator_contrasts.csv",
            ROB_EM.format(sub="no_fhb_mode_switch", est="== 'dependence'", **EMD),
            "mean_diff", "mean_diff_cluster_lo", "mean_diff_cluster_hi", "r3",
            require="{nRobCtDepEmNsLo} > 0",
            require_note="'the dependence-aware model remains less stable than EM' without the switch configurations"),
        N("nRobCtFsHbNsHiMin", "rob_estimator_contrasts.csv",
          "scope == 'all_tissues' and protocol == 'source' and subset == 'no_fhb_mode_switch' and metric == 'D_f1' "
          "and estimator in ['fisher', 'stouffer'] and reference == 'family_hb'", "mean_diff_cluster_hi", "r3",
          agg="min", helper=True, require="{nRobCtFsHbNsHiMin} > 0",
          require_note="'Fisher and Stouffer are not more stable than Family-HB' (no interval below 0)"),
    ])
    ES = "universe == 'global' and protocol == 'source' and subset == 'all' and estimator {e}"
    add([
        N("nRobNEst", "rob_edge_stability.csv", "universe == 'global' and protocol == 'source' and subset == 'all'",
          "estimator", "word", agg="nunique", note="latent-class estimators and combination rules in the robustness study"),
        *RNG("nRobSp", "rob_edge_stability.csv", "universe == 'global' and protocol == 'source' and subset == 'all'",
             "spearman_mean", "r2",
             was=("0.98", "0.98"), note="all ten estimators and rules, global universe"),
        N("nRobJacBfdrEm", "rob_edge_stability.csv", ES.format(e="== 'em'"), "jaccard_bfdr_mean", "r2", was="0.92"),
        N("nRobJacBfdrUnion", "rob_edge_stability.csv", ES.format(e="== 'union'"), "jaccard_bfdr_mean", "r2",
          was="0.97"),
        N("nRobJacBfdrHb", "rob_edge_stability.csv", ES.format(e="== 'family_hb'"), "jaccard_bfdr_mean", "r2",
          was="0.85"),
    ])
    SM = dict(sc="all_tissues", met="jaccard_top_sizematched")
    add([
        N("nRobSmHb", "rob_estimator_contrasts.csv", ROB_EM.format(sub="all", est="== 'family_hb'", **SM),
          "mean_estimator", "r2", was="0.90", note="size-matched Jaccard, pooled over tissues (post hoc column)"),
        N("nRobSmEm", "rob_estimator_contrasts.csv", ROB_EM.format(sub="all", est="== 'family_hb'", **SM),
          "mean_reference", "r2", was="0.89"),
        *CI("nRobSmCtHbEm", "rob_estimator_contrasts.csv", ROB_EM.format(sub="all", est="== 'family_hb'", **SM),
            "mean_diff", "mean_diff_cluster_lo", "mean_diff_cluster_hi", "r3",
            require="{nRobSmCtHbEmLo} < 0 < {nRobSmCtHbEmHi}",
            require_note="'Family-HB is not distinguishable from EM' at equal set size"),
        N("nRobSmUnion", "rob_estimator_contrasts.csv", ROB_EM.format(sub="all", est="== 'union'", **SM),
          "mean_estimator", "r2", was="0.96"),
        *CI("nRobSmCtUnionEm", "rob_estimator_contrasts.csv", ROB_EM.format(sub="all", est="== 'union'", **SM),
            "mean_diff", "mean_diff_cluster_lo", "mean_diff_cluster_hi", "r3",
            require="{nRobSmCtUnionEmLo} > 0", require_note="'union is more stable than EM' at equal set size"),
        N("nRobSmMaj", "rob_estimator_contrasts.csv", ROB_EM.format(sub="all", est="== 'majority'", **SM),
          "mean_estimator", "r2"),
        *CI("nRobSmCtMajEm", "rob_estimator_contrasts.csv", ROB_EM.format(sub="all", est="== 'majority'", **SM),
            "mean_diff", "mean_diff_cluster_lo", "mean_diff_cluster_hi", "r3",
            require="{nRobSmCtMajEmLo} > 0", require_note="'majority is more stable than EM' at equal set size"),
        N("nRobSmOthLoMax", "rob_estimator_contrasts.csv",
          ROB_EM.format(sub="all", est="in ['dependence', 'vote_fraction', 'fisher', 'stouffer', "
                                      "'family_regularized', 'multifactor']", **SM), "mean_diff_cluster_lo", "r3",
          agg="max", helper=True, require="{nRobSmOthLoMax} <= 0",
          require_note="'only union and majority are more stable than EM' at equal size (no other interval above 0)"),
        N("nRobSmDep", "rob_estimator_contrasts.csv", ROB_EM.format(sub="all", est="== 'dependence'", **SM),
          "mean_estimator", "r2", was="0.91"),
        *CI("nRobSmCtDepHb", "rob_estimator_contrasts.csv",
            ROB_SRC.format(sub="all", met="jaccard_top_sizematched", est="dependence"),
            "mean_diff", "mean_diff_cluster_lo", "mean_diff_cluster_hi", "r3",
            require="{nRobSmCtDepHbLo} > 0", require_note="'the dependence-aware model is more stable than Family-HB'"),
        N("nRobSmCtDepEmLo", "rob_estimator_contrasts.csv", ROB_EM.format(sub="all", est="== 'dependence'", **SM),
          "mean_diff_cluster_lo", "r3", helper=True),
        N("nRobSmCtDepEmHi", "rob_estimator_contrasts.csv", ROB_EM.format(sub="all", est="== 'dependence'", **SM),
          "mean_diff_cluster_hi", "r3", helper=True, require="{nRobSmCtDepEmLo} < 0 < {nRobSmCtDepEmHi}",
          require_note="'the dependence-aware model ... is not distinguishable from EM' at equal set size"),
        N("nRobSmHbNs", "rob_estimator_contrasts.csv", ROB_EM.format(sub="no_fhb_mode_switch", est="== 'family_hb'", **SM),
          "mean_estimator", "r3"),
        N("nRobSmEmNs", "rob_estimator_contrasts.csv", ROB_EM.format(sub="no_fhb_mode_switch", est="== 'family_hb'", **SM),
          "mean_reference", "r3"),
        N("nRobSmCtHbEmNsLo", "rob_estimator_contrasts.csv",
          ROB_EM.format(sub="no_fhb_mode_switch", est="== 'family_hb'", **SM), "mean_diff_cluster_lo", "r3", helper=True),
        N("nRobSmCtHbEmNsHi", "rob_estimator_contrasts.csv",
          ROB_EM.format(sub="no_fhb_mode_switch", est="== 'family_hb'", **SM), "mean_diff_cluster_hi", "r3", helper=True,
          require="{nRobSmCtHbEmNsLo} < 0 < {nRobSmCtHbEmNsHi}",
          require_note="'not distinguishable at equal size' without the switch configurations"),
        N("nRobGlobSwitch", "rob_edge_stability.csv", ES.format(e="== 'family_hb'"), "n_fhb_mode_switch", "int",
          helper=True, require="{nRobGlobSwitch} == 0",
          require_note="'in the global universe, where Family-HB never switches mode under source dropout'"),
        *CI("nRobSmCtHbEmGlob", "rob_estimator_contrasts.csv",
            ROB_EM.format(sc="global", sub="all", met="jaccard_top_sizematched", est="== 'family_hb'"),
            "mean_diff", "mean_diff_cluster_lo", "mean_diff_cluster_hi", "r3",
            require="{nRobSmCtHbEmGlobLo} < 0 < {nRobSmCtHbEmGlobHi}",
            require_note="global size-matched contrast 'not distinguishable from zero'"),
        N("nRobSmGlobNEm", "rob_estimator_contrasts.csv",
          ROB_EM.format(sc="global", sub="all", met="jaccard_top_sizematched", est="== 'family_hb'"),
          "n_reference_more_stable", "int", note="configurations in which EM is more stable at equal size"),
        N("nRobSmGlobN", "rob_estimator_contrasts.csv",
          ROB_EM.format(sc="global", sub="all", met="jaccard_top_sizematched", est="== 'family_hb'"),
          "n_configs", "int"),
    ])
    LIN = "tissue == 'all_tissues' and protocol == 'lineage' and subset == '{s}' and metric == 'f1' and estimator == '{e}'"
    add([
        N("nRobLinN", "rob_method_ranking.csv", LIN.format(s="excl_uninformative", e="family_hb"), "n_configs", "word",
          note="identified lineage configurations (tissues)"),
        N("nRobLinHb", "rob_method_ranking.csv", LIN.format(s="excl_uninformative", e="family_hb"), "mean_D", "r3",
          was="0.120"),
        N("nRobLinEm", "rob_method_ranking.csv", LIN.format(s="excl_uninformative", e="em"), "mean_D", "r3", was="0.065"),
        N("nRobLinUnion", "rob_method_ranking.csv", LIN.format(s="excl_uninformative", e="union"), "mean_D", "r3",
          was="0.035"),
        N("nRobLinNs", "rob_method_ranking.csv", LIN.format(s="excl_uninformative_no_fhb_mode_switch", e="family_hb"),
          "n_configs", "word", note="identified lineage configurations without a Family-HB switch"),
        N("nRobLinHbNs", "rob_method_ranking.csv", LIN.format(s="excl_uninformative_no_fhb_mode_switch", e="family_hb"),
          "mean_D", "r3"),
        N("nRobLinEmNs", "rob_method_ranking.csv", LIN.format(s="excl_uninformative_no_fhb_mode_switch", e="em"),
          "mean_D", "r3", require="{nRobLinEmNs} > {nRobLinHbNs}",
          require_note="'in the other configurations the order reverses'"),
        E("nRobLinNSw", "{nRobLinN} - {nRobLinNs}", "word", note="lineage configurations with a Family-HB switch"),
        E("nRobLinHbSw", "({nRobLinN} * {nRobLinHb} - {nRobLinNs} * {nRobLinHbNs}) / ({nRobLinN} - {nRobLinNs})", "r3",
          note="Family-HB mean D over the switch configurations (from the two subset means)"),
        E("nRobLinEmSw", "({nRobLinN} * {nRobLinEm} - {nRobLinNs} * {nRobLinEmNs}) / ({nRobLinN} - {nRobLinNs})", "r3",
          note="EM mean D over the same configurations",
          require="{nRobLinHbSw} > {nRobLinEmSw}", require_note="'the separation lies in the switch configurations'"),
        *RNG("nWtPct", "fit_class_weights.csv", f"{HBN} and class_name == 'T'", "w", "pct1", was=("0.3", "0.7"),
             note="w_T, percent of edges"),
    ])

    # ======================================================================================= 5.3 held-out
    for r, lab, was in (("encode", "Enc", "1.04"), ("enets2", "Enet", "0.76"), ("pert_pooled", "Pert", "0.48")):
        add([N(f"nHoNeval{lab}Mio", "heldout_estimators.csv", f"{GLOB_HO} and score == 'family_hb' and reference == '{r}'",
               "n_eval", "r2", scale=1e-6, was=was, note="evaluable edges, millions")])
    AP = GLOB_HO + " and score == '{s}' and reference == '{r}'"
    for r, rl in REF.items():
        was = {"encode": ("0.96", "0.94", "0.98"), "enets2": ("1.59", "1.31", "2.00"),
               "pert_pooled": ("0.94", "0.90", "0.98")}[r]
        add(CI(f"nApHb{rl}", "heldout_estimators.csv", AP.format(s="family_hb", r=r), "ap_lift", "ap_lift_lo",
               "ap_lift_hi", "r2", was=was, note="TF-bootstrap 95%"))
    for s, sl, refs in (("em", "Em", {"encode": "0.96", "enets2": "1.49", "pert_pooled": "0.94"}),
                        ("fisher", "Fi", {"encode": "1.07", "enets2": "1.80", "pert_pooled": "1.01"}),
                        ("stouffer", "St", {"encode": "1.04", "enets2": "1.80"}),
                        ("vote_count", "Vc", {"pert_pooled": "1.02"}),
                        ("dorothea_level_heldin", "Dor", {"pert_pooled": "1.01"}),
                        ("collectri_heldin", "Col", {"pert_pooled": "1.02"}),
                        ("status3", "Status", {"encode": "1.05", "enets2": "1.38", "pert_pooled": "1.06"}),
                        ("family_hb_unobserved_last", "Ul", {"encode": "1.07", "enets2": "1.79", "pert_pooled": "1.03"})):
        for r, was in refs.items():
            add([N(f"nAp{sl}{REF[r]}", "heldout_estimators.csv", AP.format(s=s, r=r), "ap_lift", "r2", was=was)])
    PA = GLOB_HO + " and metric == 'ap_lift' and score_b == '{b}' and reference == '{r}'"
    add([
        N("nPairFiEnc", "heldout_paired.csv", PA.format(b="fisher", r="encode"), "diff", "r2", was="-0.11"),
        N("nPairFiEnet", "heldout_paired.csv", PA.format(b="fisher", r="enets2"), "diff", "r2", was="-0.21"),
        N("nPairFiChipP", "heldout_paired.csv",
          GLOB_HO + " and metric == 'ap_lift' and score_b == 'fisher' and reference in ['encode', 'enets2']",
          "p_holm_confirmatory", "e2", agg="same", was="1.2e-3", note="Holm, confirmatory family"),
        *CI("nPairDorEnet", "heldout_paired.csv", PA.format(b="dorothea_level_heldin", r="enets2"), "diff", "diff_lo",
            "diff_hi", "r2", was=("0.21", "0.07", "0.41")),
        N("nPairDorEnetP", "heldout_paired.csv", PA.format(b="dorothea_level_heldin", r="enets2"),
          "p_holm_confirmatory", "e2", was="2.4e-3"),
        *CI("nPairEmEnc", "heldout_paired.csv", PA.format(b="em", r="encode"), "diff", "diff_lo", "diff_hi", "r3",
            note="Family-HB minus EM, ENCODE AP lift (confirmatory)",
            require="{nPairEmEncHi} < 0", require_note="'EM is slightly better on ENCODE'"),
        N("nPairEmEncP", "heldout_paired.csv", PA.format(b="em", r="encode"), "p_holm_confirmatory", "e2"),
        N("nPairEmOtherPMin", "heldout_paired.csv",
          GLOB_HO + " and metric == 'ap_lift' and score_b == 'em' and reference in ['enets2', 'pert_pooled']",
          "p_holm_confirmatory", "r2", agg="min", helper=True, require="{nPairEmOtherPMin} > 0.05",
          require_note="'EM is not distinguishable on ENETS2 and perturbation data'"),
        N("nPairChipStoPMax", "heldout_paired.csv",
          GLOB_HO + " and metric == 'ap_lift' and score_b == 'stouffer' and reference in ['encode', 'enets2']",
          "p_holm_confirmatory", "r2", agg="max", helper=True, require="{nPairChipStoPMax} < 0.05",
          require_note="'and so is Stouffer's' (better on both ChIP references)"),
    ])
    ADJ = GLOB_HO + " and reference == 'pert_pooled' and score {s}"
    PADJ = (GLOB_HO + " and metric == 'ap_lift_comp_adj' and confirmatory_comparator == True and "
            "reference == 'pert_pooled' and score_b {b}")
    add([
        N("nCompNullHb", "heldout_estimators.csv", ADJ.format(s="== 'family_hb'"), "ap_lift_comp_null", "r3",
          note="within-TF permutation expectation of AP lift (post hoc)"),
        *[N(f"nCompNull{lab}", "heldout_estimators.csv", ADJ.format(s=f"== '{sc}'"), "ap_lift_comp_null", "r3",
            helper=True) for sc, lab in (("fisher", "Fi"), ("stouffer", "St"), ("vote_count", "Vc"),
                                         ("dorothea_level_heldin", "Dor"), ("collectri_heldin", "Col"))],
        N("nApStPertRaw", "heldout_estimators.csv", ADJ.format(s="== 'stouffer'"), "ap_lift", "r3", helper=True),
        *RNG("nCompNullCmp", "heldout_estimators.csv",
             ADJ.format(s="in ['fisher', 'stouffer', 'vote_count', 'dorothea_level_heldin', 'collectri_heldin']"),
             "ap_lift_comp_null", "r2", note="the five non-EM confirmatory comparators"),
        E("nCompShareMin",
          "min(({nCompNullFi} - {nCompNullHb}) / ({nApFiPert} - {nApHbPert}), "
          "({nCompNullSt} - {nCompNullHb}) / ({nApStPertRaw} - {nApHbPert}), "
          "({nCompNullVc} - {nCompNullHb}) / ({nApVcPert} - {nApHbPert}), "
          "({nCompNullDor} - {nCompNullHb}) / ({nApDorPert} - {nApHbPert}), "
          "({nCompNullCol} - {nCompNullHb}) / ({nApColPert} - {nApHbPert}))", "r2", helper=True,
          note="smallest between-TF composition share of the pooled perturbation AP-lift gap over the five comparators",
          require="{nCompShareMin} > 0.5", require_note="'most of the pooled perturbation difference reflects TF composition'"),
        *CI("nAdjHb", "heldout_estimators.csv", ADJ.format(s="== 'family_hb'"), "ap_lift_comp_adj",
            "ap_lift_comp_adj_lo", "ap_lift_comp_adj_hi", "r3", note="composition-adjusted AP lift (post hoc)"),
        N("nPairAdjFi", "heldout_paired.csv", PADJ.format(b="== 'fisher'"), "diff", "r3", scale=-1.0,
          note="Fisher minus Family-HB (MAP), composition-adjusted AP lift"),
        N("nPairAdjFiP", "heldout_paired.csv", PADJ.format(b="== 'fisher'"), "p_holm_6comparators", "r2",
          note="Holm over the six comparators (descriptive)"),
        N("nPairAdjVcCol", "heldout_paired.csv", PADJ.format(b="in ['vote_count', 'collectri_heldin']"), "diff", "r3",
          agg="same", scale=-1.0, note="vote count and CollecTRI minus Family-HB (print the same)"),
        N("nPairAdjVcColP", "heldout_paired.csv", PADJ.format(b="in ['vote_count', 'collectri_heldin']"),
          "p_holm_6comparators", "e2", agg="same"),
        N("nPairAdjFloor", "heldout_paired.csv", PADJ.format(b="in ['vote_count', 'collectri_heldin']"),
          "p_holm_6comparators_floor", "e2", agg="same", helper=True,
          require="abs({nPairAdjVcColP} - {nPairAdjFloor}) < 1e-12",
          require_note="'the smallest attainable value'"),
        N("nPairAdjFiSec", "heldout_paired.csv",
          GLOB_HO + " and metric == 'ap_lift_comp_adj' and reference == 'pert_pooled' and score_b == 'fisher_sec6'",
          "diff", "r3", scale=-1.0, note="Fisher (no mid-p correction) minus Family-HB, composition-adjusted (post hoc)"),
        N("nPairAdjFiSecLo", "heldout_paired.csv",
          GLOB_HO + " and metric == 'ap_lift_comp_adj' and reference == 'pert_pooled' and score_b == 'fisher_sec6'",
          "diff_hi", "r3", scale=-1.0, note="lower bound (sign reversed: from diff_hi)"),
        N("nPairAdjFiSecHi", "heldout_paired.csv",
          GLOB_HO + " and metric == 'ap_lift_comp_adj' and reference == 'pert_pooled' and score_b == 'fisher_sec6'",
          "diff_lo", "r3", scale=-1.0, note="upper bound (sign reversed: from diff_lo)",
          require="{nPairAdjFiSecLo} > 0", require_note="'below the Fisher variant'"),
        *CI("nAucHbPert", "heldout_estimators.csv", AP.format(s="family_hb", r="pert_pooled"), "within_tf_auroc",
            "within_tf_auroc_lo", "within_tf_auroc_hi", "r3",
            require="{nAucHbPertHi} < 0.5", require_note="'ranks below random within TFs' (interval below 0.5)"),
        N("nPermHbPert", "heldout_estimators.csv", AP.format(s="family_hb", r="pert_pooled"), "perm_p_5k", "r1",
          require="{nPermHbPert} == 1", require_note="'below that of every within-TF permutation'"),
        N("nAdjHbCovers", "heldout_estimators.csv", ADJ.format(s="== 'family_hb'"), "ap_lift_comp_adj_lo", "r3",
          helper=True, require="{nAdjHbLo} < 1 < {nAdjHbHi}",
          require_note="'the TF-bootstrap interval of the composition-adjusted AP lift includes 1'"),
        N("nAdjHbEnc", "heldout_estimators.csv", AP.format(s="family_hb", r="encode"), "ap_lift_comp_adj", "r3"),
        N("nAdjFiEnc", "heldout_estimators.csv", AP.format(s="fisher", r="encode"), "ap_lift_comp_adj", "r3"),
        *CI("nAucHbEnc", "heldout_estimators.csv", AP.format(s="family_hb", r="encode"), "within_tf_auroc",
            "within_tf_auroc_lo", "within_tf_auroc_hi", "r3",
            require="{nAucHbEncHi} < 0.5", require_note="ENCODE: 'within-TF deficit in AUROC'"),
        *CI("nPairAucFiEnet", "heldout_paired.csv",
            GLOB_HO + " and metric == 'within_tf_auroc' and score_a == 'family_hb' and score_b == 'fisher' and "
            "reference == 'enets2'", "diff", "diff_lo", "diff_hi", "r3",
            require="{nPairAucFiEnetLo} < 0 < {nPairAucFiEnetHi}",
            require_note="ENETS2: 'within-TF deficit in AP only' (AUROC difference to Fisher not distinguishable)"),
        *CI("nPairAucEmEnet", "heldout_paired.csv",
            GLOB_HO + " and metric == 'within_tf_auroc' and score_a == 'family_hb' and score_b == 'em' and "
            "reference == 'enets2'", "diff", "diff_lo", "diff_hi", "r3",
            note="Family-HB minus EM, within-TF AUROC, ENETS2 (paired TF bootstrap)",
            require="{nPairAucEmEnetHi} < 0", require_note="'its within-TF AUROC is slightly below EM's'"),
        N("nPairAucEmEnetP", "heldout_paired.csv",
          GLOB_HO + " and metric == 'within_tf_auroc' and score_a == 'family_hb' and score_b == 'em' and "
          "reference == 'enets2'", "p_holm", "ceil2", note="Holm-adjusted bootstrap p over the estimator comparisons "
          "('p <=': at the resampling floor, rounded up)",
          require="abs({nPairAucEmEnetP} - {nPairAucEmEnetPFloor}) < 1e-12",
          require_note="'p <= ..., the resampling floor'"),
        N("nPairAucEmEnetPFloor", "heldout_paired.csv",
          GLOB_HO + " and metric == 'within_tf_auroc' and score_a == 'family_hb' and score_b == 'em' and "
          "reference == 'enets2'", "p_holm_floor", "r4", helper=True, note="smallest attainable Holm-adjusted p"),
        N("nEnetAdjFsHiMax", "heldout_paired.csv",
          GLOB_HO + " and metric == 'ap_lift_comp_adj' and confirmatory_comparator == True and reference == 'enets2' "
          "and score_b in ['fisher', 'stouffer']", "diff_hi", "r3", agg="max", helper=True,
          require="{nEnetAdjFsHiMax} < 0", require_note="ENETS2: within-TF AP deficit against Fisher and Stouffer"),
        N("nEnetAdjOthHiMin", "heldout_paired.csv",
          GLOB_HO + " and metric == 'ap_lift_comp_adj' and confirmatory_comparator == True and reference == 'enets2' "
          "and score_b not in ['fisher', 'stouffer']", "diff_hi", "r3", agg="min", helper=True,
          require="{nEnetAdjOthHiMin} > 0", require_note="ENETS2: 'only against mid-p Fisher and Stouffer'"),
        *RNG("nAdjHbTisPert", "heldout_estimators.csv",
             "universe != 'global' and eval_subset == 'all_evaluable' and reference == 'pert_pooled' and "
             "score == 'family_hb'", "ap_lift_comp_adj", "r2", note="tissue universes, composition-adjusted (post hoc)"),
        *RNG("nAdjUlTisPert", "heldout_estimators.csv",
             "universe != 'global' and eval_subset == 'all_evaluable' and reference == 'pert_pooled' and "
             "score == 'family_hb_unobserved_last'", "ap_lift_comp_adj", "r2",
             require="{nAdjUlTisPertMin} > 1 and {nAdjUlTisPertMin} > {nAdjHbTisPertMax}",
             require_note="'the within-TF deficits disappear when uncovered edges are placed last'"),
        N("nAucUlMinusHbTisMin", "heldout_paired.csv",
          "universe != 'global' and eval_subset == 'all_evaluable' and metric == 'within_tf_auroc' and "
          "reference == 'pert_pooled' and score_a == 'family_hb' and score_b == 'family_hb_unobserved_last'",
          "diff_hi", "r3", agg="max", helper=True, require="{nAucUlMinusHbTisMin} < 0",
          require_note="tissues: 'within-TF AUROC moves the same way' (Family-HB below its uncovered-last variant)"),
        *RNG("nTfsPosTis", "heldout_estimators.csv",
             "universe != 'global' and eval_subset == 'all_evaluable' and reference == 'pert_pooled' and "
             "score == 'family_hb'", "n_tfs_with_pos", "int", note="TFs with at least one positive"),
        N("nAdjDefTisHiMax", "heldout_paired.csv",
          "universe != 'global' and eval_subset == 'all_evaluable' and metric == 'ap_lift_comp_adj' and "
          "reference == 'pert_pooled' and score_b == 'family_hb_unobserved_last'", "diff_hi", "r3", agg="max",
          helper=True, require="{nAdjDefTisHiMax} < 0",
          require_note="tissues: Family-HB below its uncovered-last variant within TF (paired intervals below 0)"),
        N("nAucUlPert", "heldout_estimators.csv", AP.format(s="family_hb_unobserved_last", r="pert_pooled"),
          "within_tf_auroc", "r3"),
        N("nAdjUlPert", "heldout_estimators.csv", AP.format(s="family_hb_unobserved_last", r="pert_pooled"),
          "ap_lift_comp_adj", "r3"),
        *RNG("nAucLcm", "heldout_estimators.csv", GLOB_HO + " and group == 'lcm' and reference in ['encode', 'pert_pooled']",
             "within_tf_auroc", "r3", was=("0.468", "0.489"),
             note="latent-class estimators, ENCODE and pooled perturbation"),
        N("nAucLcmHiMax", "heldout_estimators.csv",
          GLOB_HO + " and group == 'lcm' and reference in ['encode', 'pert_pooled']", "within_tf_auroc_hi", "r3",
          agg="max", helper=True, require="{nAucLcmHiMax} < 0.5",
          require_note="'every latent-class estimator ranks worse than random within TFs' (all intervals below 0.5)"),
    ])
    ST = "universe == 'global' and reference == '{r}' and status == '{s}'"
    for r, rl in REF.items():
        was = {"encode": ("0.89", "0.88", "0.91"), "enets2": ("0.85", "0.78", "0.90"),
               "pert_pooled": ("0.97", "0.95", "0.98")}[r]
        add(CI(f"nStUnobs{rl}", "heldout_evidence_status.csv", ST.format(r=r, s="unobserved"), "oe_tf_matched", "oe_lo",
               "oe_hi", "r2", was=was))
        add([N(f"nStObsneg{rl}", "heldout_evidence_status.csv", ST.format(r=r, s="observed_neg"), "oe_tf_matched", "r2",
               was={"encode": "1.05", "enets2": "0.98", "pert_pooled": "1.01"}[r]),
             N(f"nStRep{rl}", "heldout_evidence_status.csv", ST.format(r=r, s="reported"), "oe_tf_matched", "r2",
               was={"encode": "1.17", "enets2": "3.41", "pert_pooled": "1.36"}[r])])
    add([
        N("nStCredUnobs", "heldout_evidence_status.csv", ST.format(r="encode", s="unobserved"), "mean_cred_family_hb",
          "sig2", was="0.0031"),
        N("nStCredObsneg", "heldout_evidence_status.csv", ST.format(r="encode", s="observed_neg"),
          "mean_cred_family_hb", "sig2", was="0.00099"),
    ])
    for r, rl in REF.items():
        fmt = "r1" if r == "enets2" else "r2"
        was = {"encode": ("1.62", "1.34", "1.92"), "enets2": ("12.6", "8.8", "17.5"),
               "pert_pooled": ("2.10", "1.38", "3.56")}[r]
        add(CI(f"nEnrHb{rl}", "heldout_estimators.csv", AP.format(s="family_hb", r=r), "enr_at_500", "enr_at_500_lo",
               "enr_at_500_hi", fmt, was=was))
        add([N(f"nSenrHb{rl}", "heldout_estimators.csv", AP.format(s="family_hb", r=r), "strat_enr_at_500", "r2",
               was={"encode": "1.32", "enets2": "6.79", "pert_pooled": "1.53"}[r])])
    add([
        *RNG("nChipTfsTis", "heldout_chip_coverage.csv", "universe != 'global' and reference == 'encode_tfbs'",
             "evaluable_tfs", "int", was=("17", "24")),
        *RNG("nTisEncD", "heldout_paired.csv",
             "universe != 'global' and eval_subset == 'all_evaluable' and reference == 'encode' and metric == 'ap_lift'"
             " and analysis_role == 'confirmatory' and score_b != 'em'", "diff", "r2", was=("-0.11", "-0.03")),
    ])
    # predicted sets (top-k per TF) of Family-HB and of expression-based network inference in the tissue universes
    MC = "universe == '{t}' and reference == '{r}' and eval_subset == 'all_evaluable' and topk_per_tf == 50 and score == '{s}'"
    for t, tl in TIS.items():
        for r, rl in (("encode", "Enc"), ("enets2", "Enet")):
            add([N(f"nPs{s_l}{rl}{tl}", "heldout_methods_context.csv", MC.format(t=t, r=r, s=s_), "enr_tf_matched", "r2",
                   helper=(s_l == "Rand" and t != "kidney") or (s_l in ("Gb", "Pe", "Hb") and r == "enets2" and t == "lung"),
                   note=f"TF-matched enrichment of the top-50-per-TF predicted set, {s_} ({r}, {t})")
                 for s_, s_l in (("family_hb_nuts", "Hb"), ("grnboost2", "Gb"), ("pearson_abs", "Pe"), ("random", "Rand"))])
    add([
        N("nPsExprBeatsHbEnc", "heldout_methods_context.csv",
          "eval_subset == 'all_evaluable' and topk_per_tf == 50 and reference == 'encode' and score == 'family_hb_nuts'",
          "enr_tf_matched", "r2", agg="max", helper=True,
          require=("min({nPsGbEncImm} - {nPsHbEncImm}, {nPsGbEncKid} - {nPsHbEncKid}, {nPsGbEncLung} - {nPsHbEncLung}, "
                   "{nPsPeEncImm} - {nPsHbEncImm}, {nPsPeEncKid} - {nPsHbEncKid}, {nPsPeEncLung} - {nPsHbEncLung}) > 0"),
          require_note="'GRNBoost2 and |Pearson| are more enriched for ENCODE than Family-HB in every tissue'"),
        N("nPsExprBeatsHbEnet", "heldout_methods_context.csv",
          "eval_subset == 'all_evaluable' and topk_per_tf == 50 and reference == 'enets2' and score == 'family_hb_nuts'",
          "enr_tf_matched", "r2", agg="max", helper=True,
          require=("min({nPsGbEnetImm} - {nPsHbEnetImm}, {nPsGbEnetKid} - {nPsHbEnetKid}, "
                   "{nPsPeEnetImm} - {nPsHbEnetImm}, {nPsPeEnetKid} - {nPsHbEnetKid}) > 0 and "
                   "{nPsGbEnetLung} < {nPsHbEnetLung}"),
          require_note="'on ENETS2 in immune and kidney' (not in lung, where GRNBoost2 is below Family-HB)"),
    ])
    for r, rl in (("encode", "Enc"), ("enets2", "Enet")):
        q = MC.format(t="kidney", r=r, s="family_hb_nuts")
        add([N(f"nPsHb{rl}KidLo", "heldout_methods_context.csv", q, "enr_tf_matched_lo", "r2", helper=True),
             N(f"nPsHb{rl}KidHi", "heldout_methods_context.csv", q, "enr_tf_matched_hi", "r2", helper=True,
               require=f"{{nPsHb{rl}KidLo}} < {{nPsRand{rl}Kid}} < {{nPsHb{rl}KidHi}} and {{nPsHb{rl}Kid}} <= {{nPsRand{rl}Kid}}",
               require_note="kidney: Family-HB 'not above the random score' (random inside the Family-HB interval)")])
    EVT = "universe == '{t}' and variant == 'prespecified_cc' and annotator == '__ALL__'"
    for t, tl in TIS.items():
        add([N(f"nEvRep{tl}", "evidence_summary.csv", EVT.format(t=t), "n_pos", "int", helper=True),
             N(f"nEvTfs{tl}", "evidence_summary.csv", EVT.format(t=t), "n_tfs", "int", helper=True)])
    add([
        E("nCtxRepPerTfMin", "min({nEvRepImm} / {nEvTfsImm}, {nEvRepKid} / {nEvTfsKid}, {nEvRepLung} / {nEvTfsLung})",
          "floor0", note="reported edges per TF in the tissue-context universes (smallest)"),
        E("nCtxRepPerTfMax", "max({nEvRepImm} / {nEvTfsImm}, {nEvRepKid} / {nEvTfsKid}, {nEvRepLung} / {nEvTfsLung})",
          "ceil0", note="reported edges per TF in the tissue-context universes (largest)",
          require="{nCtxRepPerTfMax} < {nTopK}", require_note="'fewer reported edges per TF than the 50 predicted'"),
        *RNG("nPsHbEncLoK", "heldout_methods_context.csv",
             "eval_subset == 'all_evaluable' and topk_per_tf == {nTopKSensLo} and reference == 'encode' and "
             "score == 'family_hb_nuts'", "enr_tf_matched", "r2",
             note="Family-HB predicted sets of the smaller size: TF-matched ENCODE enrichment over tissues"),
        N("nPsHbEncLoKGain", "heldout_methods_context.csv",
          "eval_subset == 'all_evaluable' and topk_per_tf == {nTopKSensLo} and reference == 'encode' and "
          "score == 'family_hb_nuts'", "enr_tf_matched", "r2", agg="min", helper=True,
          require="{nPsHbEncLoKMin} > {nPsHbEncKid} and {nPsHbEncLoKMax} > {nPsHbEncLung}",
          require_note="'the Family-HB deficit shrinks for smaller predicted sets'"),
        N("nRandEncMinAllK", "heldout_methods_context.csv",
          "eval_subset == 'all_evaluable' and reference == 'encode' and score == 'random'", "enr_tf_matched", "r2",
          agg="min", helper=True),
        N("nScRandBelowAllK", "heldout_methods_context.csv",
          "eval_subset == 'all_evaluable' and reference == 'encode' and score == 'scgpt_attn'", "enr_tf_matched", "r2",
          agg="max", helper=True, require="{nScRandBelowAllK} < {nRandEncMinAllK}",
          require_note="'scGPT attention stays far below a random score on ENCODE at every predicted-set size'"),
        *RNG("nFmCoobsRetPct", "fm_min_coobs_sensitivity.csv", "model == 'scgpt' and variant == 'attn' and k == 50",
             "frac_unfiltered_topk_retained", "pct0", note="share of the scGPT top-50 set retained after the restriction"),
        *RNG("nFmCoobsOvPct", "fm_min_coobs_sensitivity.csv", "model == 'scgpt' and variant == 'attn' and k == 50",
             "topk_overlap_with_uniform_median", "pct0", note="overlap of the restricted scGPT top set with the 1/L null"),
        N("nFmCoobsOvRandPctMax", "fm_min_coobs_sensitivity.csv", "model == 'scgpt' and variant == 'attn' and k == 50",
          "topk_overlap_random_expectation", "pct0", agg="max", note="random expectation of that overlap"),
    ])

    # ======================================================================================= 5.4 informative observation
    OF = "universe == 'global' and fit_id {f}"
    OH = "universe == 'global' and reference == '{r}' and score == '{s}'"
    add([
        N("nObsShare", "obs_fit_summary.csv", OF.format(f="in ['family_hb_obs', 'em_obs']"),
          "obs_share_of_joint_loglik", "pct0", agg="same", was="98"),
        N("nObsVar", "obs_fit_summary.csv", OF.format(f="in ['family_hb_obs', 'em_obs']"),
          "cred_var_share_between_targets", "pct0", agg="same", was="83"),
        N("nObsVarHb", "obs_fit_summary.csv", OF.format(f="== 'family_hb'"), "cred_var_share_between_targets", "pct1",
          was="0.6"),
        N("nObsWtHbObs", "obs_fit_summary.csv", OF.format(f="== 'family_hb_obs'"), "w_T", "r2", was="0.68"),
        N("nObsWtEmObs", "obs_fit_summary.csv", OF.format(f="== 'em_obs'"), "w_T", "r2", was="0.32"),
        N("nObsApHbObs", "obs_heldout.csv", OH.format(r="encode", s="family_hb_obs"), "ap_lift", "r2", was="0.97"),
        *CI("nObsApEmObs", "obs_heldout.csv", OH.format(r="encode", s="em_obs"), "ap_lift", "ap_lift_lo",
            "ap_lift_hi", "r2", was=("1.02", "1.01", "1.03")),
        N("nObsApStatus", "obs_heldout.csv", OH.format(r="encode", s="status3"), "ap_lift", "r2", was="1.05"),
        N("nObsApFisher", "obs_heldout.csv", OH.format(r="encode", s="fisher"), "ap_lift", "r2", was="1.07"),
        N("nObsSenrEmObs", "obs_heldout.csv", OH.format(r="encode", s="em_obs"), "strat_enr_at_500", "r2", was="1.07"),
        N("nObsTfwSp", "obs_fit_summary.csv", OF.format(f="== 'family_hb_obs__tfw'"), "spearman_w_vs_mar_observed",
          "r2", was="0.98"),
        N("nObsSimN", "sim_hub_modes.csv", "model == 'family_hb_obs'", "n_reps", "word", agg="sum", was="twelve"),
        N("nObsSimInfl", "sim_hub_modes.csv", "model == 'family_hb_obs'", "n_reps_t_class_inflated", "word",
          agg="sum", helper=True, note="equals nObsSimN ('all')", require="{nObsSimInfl} == {nObsSimN}",
          require_note="'all ... fits inflate the true class'"),
        *RNG("nObsSimWtPct", "sim_hub_modes.csv", "model == 'family_hb_obs'", "w_T_median", "pct0",
             was=("37", "50"), note="median fitted w_T by scenario, percent"),
        N("nObsSimWtAllPctMin", "sim_hub_modes.csv", "model == 'family_hb_obs'", "w_T_min", "pct0", agg="min",
          note="smallest fitted w_T over all replicates, percent"),
        N("nObsSimWtAllPctMax", "sim_hub_modes.csv", "model == 'family_hb_obs'", "w_T_max", "pct0", agg="max",
          note="largest fitted w_T over all replicates, percent"),
        N("nObsSimTruePct", "sim_hub_modes.csv", "model == 'family_hb_obs'", "true_prevalence_mean", "r0", agg="max",
          scale=100.0, note="true share of edges in the simulated universes (percent; 'about')"),
    ])

    # ======================================================================================= 5.5 universes
    DEC = "tissue == '{t}' and reference == 'pert_pooled' and credibility == 'global_fit' and metric == 'p_at_500'"
    DN = DEC + " and null == '{n}'"
    add([
        N("nUniBaseImm", "heldout_universe_decomposition.csv", DEC.format(t="immune"), "base_rate_ctx", "r3",
          was="0.249"),
        N("nUniBaseGlob", "heldout_universe_decomposition.csv", DEC.format(t="immune"), "base_rate_glob", "r3",
          was="0.097"),
        N("nUniBr", "heldout_universe_decomposition.csv", DEC.format(t="immune"), "ratio_base_rate", "r2", was="2.57"),
        *CI("nUniRaw", "heldout_universe_decomposition.csv", DEC.format(t="immune"), "ratio_raw", "ratio_raw_lo",
            "ratio_raw_hi", "r2", was=("1.28", "0.63", "2.51")),
        *CI("nUniEnr", "heldout_universe_decomposition.csv", DEC.format(t="immune"), "ratio_enrichment",
            "ratio_enrichment_lo", "ratio_enrichment_hi", "r2", was=("0.50", "0.30", "0.80")),
        *CI("nUniNull", "heldout_downsampling_null.csv", DN.format(t="immune", n="prevalence_matched"), "null_median",
            "null_lo", "null_hi", "r3", was=("0.246", "0.225", "0.266")),
        N("nUniCtx", "heldout_downsampling_null.csv", DN.format(t="immune", n="prevalence_matched"), "value_ctx", "r3",
          was="0.261"),
        N("nUniSurvEdge", "heldout_downsampling_null.csv", DN.format(t="immune", n="prevalence_matched"),
          "frac_gap_surviving", "pct0", was="27"),
        N("nUniSurvBlock", "heldout_downsampling_null.csv", DN.format(t="immune", n="tf_block_matched"),
          "frac_gap_surviving", "pct0", was="12"),
        N("nUniPEdge", "heldout_downsampling_null.csv", DN.format(t="immune", n="prevalence_matched"),
          "p_emp_two_sided", "r2", was="0.15"),
        N("nUniPBlock", "heldout_downsampling_null.csv", DN.format(t="immune", n="tf_block_matched"),
          "p_emp_two_sided", "r2", was="0.73"),
        N("nUniFOneSurv", "heldout_downsampling_null.csv",
          "tissue == 'immune' and reference == 'pert_pooled' and credibility == 'global_fit' and metric == 'best_f1'"
          " and null == 'prevalence_matched'", "frac_gap_surviving", "pct1", was="0.4"),
        N("nUniEnrKid", "heldout_universe_decomposition.csv", DEC.format(t="kidney"), "ratio_enrichment", "r2",
          was="0.45"),
        N("nUniEnrLung", "heldout_universe_decomposition.csv", DEC.format(t="lung"), "ratio_enrichment", "r2",
          was="0.47"),
        *CI("nUniRawKid", "heldout_universe_decomposition.csv", DEC.format(t="kidney"), "ratio_raw", "ratio_raw_lo",
            "ratio_raw_hi", "r2", require="{nUniRawKid} < 1", require_note="'but lower in kidney'"),
        *CI("nUniRawLung", "heldout_universe_decomposition.csv", DEC.format(t="lung"), "ratio_raw", "ratio_raw_lo",
            "ratio_raw_hi", "r2", require="{nUniRawLungLo} < 1 < {nUniRawLungHi} and {nUniRawLo} < 1 < {nUniRawHi}",
            require_note="'nominally higher in immune and lung (intervals include 1)'"),
        N("nUniBrMin", "heldout_universe_decomposition.csv",
          "reference == 'pert_pooled' and credibility == 'global_fit' and metric == 'p_at_500'", "ratio_base_rate", "r2",
          agg="min", helper=True, require="{nUniBrMin} > 1",
          require_note="'The perturbation positive rate is higher in every tissue universe'"),
    ])
    for t, tl in TIS.items():
        add([N(f"nUniEncP{tl}", "heldout_downsampling_null.csv",
               f"tissue == '{t}' and reference == 'encode' and credibility == 'global_fit' and metric == 'p_at_500' "
               "and null == 'tf_block_matched'", "p_emp_two_sided", "r2",
               was={"immune": "0.21", "kidney": "0.92", "lung": "0.06"}[t])])

    # ======================================================================================= 5.6 governance
    GT = "scope == 'dual' and tier == '{tier}' and tissue == '{t}'"
    add([
        N("nGovPromImm", "gov_tiers.csv", GT.format(tier="Promoted", t="immune"), "n_edges", "int", was="493"),
        N("nGovStrongImm", "gov_tiers.csv", GT.format(tier="Strong", t="immune"), "n_edges", "int", was="297"),
        N("nGovModImm", "gov_tiers.csv", GT.format(tier="Moderate", t="immune"), "n_edges", "int", was="196"),
        N("nGovPromKid", "gov_tiers.csv", GT.format(tier="Promoted", t="kidney"), "n_edges", "int", was="276"),
        N("nGovPromLung", "gov_tiers.csv", GT.format(tier="Promoted", t="lung"), "n_edges", "int", was="412"),
        *RNG("nGovShare", "gov_tiers.csv", "scope == 'dual' and tier == 'Promoted'", "share_of_universe", "pct2",
             was=("0.12", "0.15")),
    ])
    for t, tl in TIS.items():
        lo, hi = {"immune": ("456", "512"), "kidney": ("237", "289"), "lung": ("344", "434")}[t]
        add([N(f"nGovProm{tl}Lo", "gov_tiers.csv", GT.format(tier="Promoted", t=t), "n_edges_draw_lo", "int", was=lo),
             N(f"nGovProm{tl}Hi", "gov_tiers.csv", GT.format(tier="Promoted", t=t), "n_edges_draw_hi", "int", was=hi)])
    add([
        *RNG("nGovBfdrProm", "gov_tiers.csv", "scope == 'dual' and tier == 'Promoted'", "bfdr_context_fit", "r2",
             was=("0.16", "0.17")),
        *RNG("nGovBfdrMod", "gov_tiers.csv", "scope == 'dual' and tier == 'Moderate'", "bfdr_context_fit", "r2",
             was=("0.29", "0.35")),
        *RNG("nGovBfdrModGlob", "gov_tiers.csv", "scope == 'dual' and tier == 'Moderate'", "bfdr_global_fit", "r2",
             note="BFDR of the dual Moderate tier under the global fit"),
        *RNG("nGovKappa", "gov_consistency.csv", "credibility == 'nuts' and scope == 'either_ge_moderate'",
             "kappa_linear", "r2", was=("0.58", "0.67")),
    ])
    for t, tl in TIS.items():
        add(CI(f"nGovOeEnc{tl}", "gov_tier_validation.csv", GOV_P.format(ref="encode", u=t), "oe_tf", "oe_tf_lo",
               "oe_tf_hi", "r2", was={"immune": ("1.67", "1.35", "2.38"), "kidney": ("1.24", "1.03", "1.81"),
                                     "lung": ("2.27", "1.78", "3.37")}[t]))
        add([N(f"nGovOeEnet{tl}", "gov_tier_validation.csv", GOV_P.format(ref="enets2", u=t), "oe_tf", "r2",
               was={"immune": "5.95", "kidney": "3.66", "lung": "6.74"}[t])])
        add(CI(f"nGovOePert{tl}", "gov_tier_validation.csv", GOV_P.format(ref="pert_pooled", u=t), "oe_tf", "oe_tf_lo",
               "oe_tf_hi", "r2", was={"immune": ("1.20", "0.82", "1.64"), "kidney": ("0.66", "0.00", "1.00"),
                                     "lung": ("1.29", "1.13", "1.57")}[t]))
    add([
        *RNG("nGovExplEnc", "gov_tier_validation.csv", "scope == 'dual' and tier == 'Exploratory' and reference == 'encode'",
             "oe_tf", "r2", note="Exploratory tier, TF-matched O/E, ENCODE, range over tissues"),
        *RNG("nGovExplEnet", "gov_tier_validation.csv", "scope == 'dual' and tier == 'Exploratory' and reference == 'enets2'",
             "oe_tf", "r2", note="Exploratory tier, TF-matched O/E, ENETS2, range over tissues"),
        N("nGovModEnetKidNeval", "gov_tier_validation.csv",
          "scope == 'dual' and tier == 'Moderate' and reference == 'enets2' and universe == 'kidney'", "n_eval", "int",
          require="{nGovModEnetKidObs} == 0", require_note="'none of the ... evaluable Moderate edges is a positive'"),
        N("nGovModEnetKidObs", "gov_tier_validation.csv",
          "scope == 'dual' and tier == 'Moderate' and reference == 'enets2' and universe == 'kidney'", "obs", "int",
          helper=True),
        *RNG("nGovOeEnet", "gov_tier_validation.csv", "scope == 'dual' and tier == 'Promoted' and reference == 'enets2'",
             "oe_tf", "r1", was=("3.7", "6.7"), note="abstract: range over tissues"),
        N("nGovModEncImm", "gov_tier_validation.csv", "scope == 'dual' and tier == 'Moderate' and reference == 'encode' "
          "and universe == 'immune'", "oe_tf", "r2", helper=True),
        N("nGovStrongEncImm", "gov_tier_validation.csv", "scope == 'dual' and tier == 'Strong' and reference == 'encode' "
          "and universe == 'immune'", "oe_tf", "r2", helper=True, require="{nGovModEncImm} >= {nGovStrongEncImm}",
          require_note="'not strictly ordered: in immune the Moderate tier is as enriched for ENCODE binding as Strong'"),
        N("nGovTrendChipMax", "gov_tier_trend.csv",
          "scope == 'dual' and strata == 'tf_x_degree' and reference != 'pert_pooled'", "p_one_sided_holm", "ceil3",
          agg="max", was="0.015", note="'p <=': rounded up"),
        N("nGovTrendTfKidEnc", "gov_tier_trend.csv",
          "scope == 'dual' and strata == 'tf' and reference == 'encode' and universe == 'kidney'", "p_one_sided_holm",
          "r3", note="trend test stratified by TF alone, kidney, ENCODE"),
        N("nGovTrendTfKidStrict", "gov_tier_trend.csv",
          "scope == 'dual' and strata == 'tf' and reference == 'encode_strict' and universe == 'kidney'",
          "p_one_sided_holm", "r2", note="trend test stratified by TF alone, kidney, ENCODE-strict"),
        N("nGovTrendTfOtherMax", "gov_tier_trend.csv",
          "scope == 'dual' and strata == 'tf' and reference != 'pert_pooled' and not (universe == 'kidney' and "
          "reference in ['encode', 'encode_strict'])", "p_one_sided_holm", "ceil3", agg="max",
          note="largest Holm p of the TF-stratified trend tests outside the two kidney ENCODE tests"),
    ])
    GG = "universe == 'global' and reference == '{r}' and tier == '{t}'"
    add([
        *CI("nGovGlobPertStrong", "gov_tier_validation.csv", GG.format(r="pert_pooled", t="Strong"), "oe_tf",
            "oe_tf_lo", "oe_tf_hi", "r2", was=("1.85", "1.49", "2.43")),
        *CI("nGovGlobPertMod", "gov_tier_validation.csv", GG.format(r="pert_pooled", t="Moderate"), "oe_tf",
            "oe_tf_lo", "oe_tf_hi", "r2", was=("1.47", "1.21", "1.81")),
        N("nGovGlobPertBelow", "gov_tier_validation.csv", GG.format(r="pert_pooled", t="Below"), "oe_tf", "r2",
          was="1.00"),
        N("nGovGlobEnetStrong", "gov_tier_validation.csv", GG.format(r="enets2", t="Strong"), "oe_tf", "r2",
          was="6.67"),
        N("nGovGlobEnetMod", "gov_tier_validation.csv", GG.format(r="enets2", t="Moderate"), "oe_tf", "r2",
          was="3.16"),
        N("nGovGlobEnetBelow", "gov_tier_validation.csv", GG.format(r="enets2", t="Below"), "oe_tf", "r2",
          was="0.98"),
        *[N(f"nGovPertNtf{tl}", "gov_tier_validation.csv", GOV_P.format(ref="pert_pooled", u=t), "n_tfs_eval", "int",
            was={"immune": "10", "kidney": None, "lung": "4"}[t]) for t, tl in TIS.items()],
        N("nGovPertFewKL", "gov_tier_validation.csv",
          "scope == 'dual' and tier == 'Promoted' and reference == 'pert_pooled' and universe in ['kidney', 'lung']",
          "few_clusters", "int", agg="min", helper=True, require="{nGovPertFewKL} == 1",
          require_note="kidney and lung: 'too few TFs for reliable bootstrap intervals' (few_clusters flag)"),
        N("nGovPertImmExcl", "gov_tier_validation.csv", GOV_P.format(ref="pert_pooled", u="immune"),
          "ci_excludes_global_oe", "int", agg="sum", helper=True, require="{nGovPertImmExcl} == 1",
          require_note="'the immune interval excludes the global-universe effect'"),
        N("nGovPertGlobOe", "gov_tier_validation.csv",
          "scope == 'dual' and tier == 'Promoted' and reference == 'pert_pooled'", "global_universe_oe", "r2",
          agg="same", was="1.71"),
        N("nGovPertPowImm", "gov_tier_validation.csv", GOV_P.format(ref="pert_pooled", u="immune"),
          "power_at_global_oe", "r2", was="0.98"),
        *RNG("nGovPertTopTfPct", "gov_tier_validation.csv",
             "scope == 'dual' and tier == 'Promoted' and reference == 'pert_pooled' and universe in ['immune', 'lung']",
             "top_tf_share_eval", "pct0", was=("39", "48"), note="SPI1 in both tissues"),
    ])
    GM = "k == 50 and method == '{m}' and tissue == '{t}'"
    for t, tl in TIS.items():
        add([
            N(f"nMethNp{tl}", "gov_promoted_by_method.csv", GM.format(m="scgpt_attn", t=t), "n_predicted", "int",
              was={"immune": "7,900", "kidney": "5,300", "lung": "7,700"}[t]),
            N(f"nMethN{tl}", "gov_promoted_by_method.csv", GM.format(m="scgpt_attn", t=t), "n_promoted", "int",
              was={"immune": "2", "kidney": "1", "lung": "4"}[t]),
            N(f"nMethExp{tl}", "gov_promoted_by_method.csv", GM.format(m="scgpt_attn", t=t), "expected_promoted_tf",
              "r1", was={"immune": "12.3", "kidney": "8.0", "lung": "9.3"}[t]),
            N(f"nMethP{tl}", "gov_promoted_by_method.csv", GM.format(m="scgpt_attn", t=t), "p_depleted_exact",
              "r3" if t == "lung" else "e2", was={"immune": "3.5e-4", "kidney": "2.7e-3", "lung": "0.045"}[t]),
        ])
    add([
        N("nMethNRandKid", "gov_promoted_by_method.csv", GM.format(m="random", t="kidney"), "n_promoted", "int"),
        N("nMethExpRandKid", "gov_promoted_by_method.csv", GM.format(m="random", t="kidney"), "expected_promoted_tf",
          "r1"),
        N("nMethPRandKid", "gov_promoted_by_method.csv", GM.format(m="random", t="kidney"), "p_depleted_exact", "r3",
          require="{nMethPRandKid} < 0.05", require_note="'the random control is itself depleted in kidney'"),
        N("nMethPRandImmLung", "gov_promoted_by_method.csv", "k == 50 and method == 'random' and tissue != 'kidney'",
          "p_depleted_exact", "r2", agg="min", helper=True, require="{nMethPRandImmLung} > 0.05",
          require_note="random control depleted only in kidney"),
    ])
    GMF = "k == 50 and method in ['geneformer_attn', 'grnboost2']"
    add([
        N("nMethRatioMinW", "gov_promoted_by_method.csv", GMF, "ratio_to_tf_expected", "wfloor", agg="min",
          was="two", note="Geneformer attention and GRNBoost2, rounded down"),
        N("nMethRatioMaxW", "gov_promoted_by_method.csv", GMF, "ratio_to_tf_expected", "wfloor", agg="max",
          was="five", note="Geneformer attention and GRNBoost2, rounded down"),
        N("nMethFracBound", "gov_promoted_by_method.csv", GMF, "frac_promoted", "ceil0", agg="max", scale=100.0,
          was="1", note="percent, rounded up ('below')"),
        N("nMethFracBoundCorr", "gov_promoted_by_method.csv",
          "k == 50 and method in ['geneformer_attn', 'grnboost2', 'pearson_abs']", "frac_promoted",
          "ceil0", agg="max", scale=100.0, was="1",
          note="Geneformer attention, GRNBoost2 and correlations; percent, rounded up ('fewer than')"),
        *RNG("nFmOverlapPct", "fm_attention_null.csv", "variant == 'attn' and k == 50 and model == 'scgpt'",
             "topk_overlap_with_uniform_median", "pct0", was=("42", "48")),
        *RNG("nFmOverlapRandPct", "fm_attention_null.csv", "variant == 'attn' and k == 50 and model == 'scgpt'",
             "topk_overlap_random_expectation", "pct0", was=("3", "5")),
        *RNG("nFmEncRatioSc", "heldout_methods_context.csv",
             "eval_subset == 'all_evaluable' and topk_per_tf == 50 and reference == 'encode' and score == 'scgpt_attn'",
             "enr_ratio_to_null", "r2", was=("0.75", "0.99")),
        N("nFmEncRatioScLoMax", "heldout_methods_context.csv",
          "eval_subset == 'all_evaluable' and topk_per_tf == 50 and reference == 'encode' and score == 'scgpt_attn'",
          "enr_ratio_to_null_lo", "r2", agg="max", helper=True, require="{nFmEncRatioScLoMax} <= 1",
          require_note="'on held-out ENCODE scGPT attention is not enriched beyond the null' (no interval above 1)"),
        *CI("nFmStrictRatioScKid", "heldout_methods_context.csv",
            "eval_subset == 'all_evaluable' and topk_per_tf == 50 and reference == 'encode_strict' and "
            "score == 'scgpt_attn' and universe == 'kidney'", "enr_ratio_to_null", "enr_ratio_to_null_lo",
            "enr_ratio_to_null_hi", "r2", note="post hoc column; stricter ENCODE subset, kidney",
            require="{nFmStrictRatioScKidLo} > 1",
            require_note="'on the stricter ENCODE subset it exceeds the null in kidney'"),
        N("nFmStrictScKidTfs", "heldout_methods_context.csv",
          "eval_subset == 'all_evaluable' and topk_per_tf == 50 and reference == 'encode_strict' and "
          "score == 'scgpt_attn' and universe == 'kidney'", "n_tfs", "int", note="evaluable TFs"),
        N("nFmStrictScOthLoMax", "heldout_methods_context.csv",
          "eval_subset == 'all_evaluable' and topk_per_tf == 50 and reference == 'encode_strict' and "
          "score == 'scgpt_attn' and universe != 'kidney'", "enr_ratio_to_null_lo", "r2", agg="max", helper=True,
          require="{nFmStrictScOthLoMax} <= 1", require_note="'only in kidney' (stricter ENCODE subset)"),
        N("nFmGfChipOthLoMax", "heldout_methods_context.csv",
          "eval_subset == 'all_evaluable' and topk_per_tf == 50 and score == 'geneformer_attn' and "
          "reference in ['encode', 'encode_strict', 'pert_pooled']", "enr_ratio_to_null_lo", "r2", agg="max",
          helper=True, require="{nFmGfChipOthLoMax} <= 1",
          require_note="'on the ChIP references and the pooled perturbation reference Geneformer attention exceeds "
                       "its null only on ENETS2'"),
        *CI("nFmEnetRatioGfImm", "heldout_methods_context.csv",
            "eval_subset == 'all_evaluable' and topk_per_tf == 50 and reference == 'enets2' and "
            "score == 'geneformer_attn' and universe == 'immune'", "enr_ratio_to_null", "enr_ratio_to_null_lo",
            "enr_ratio_to_null_hi", "r2", was=("2.90", "2.09", "4.20")),
        N("nFmEnetRatioGfKid", "heldout_methods_context.csv",
          "eval_subset == 'all_evaluable' and topk_per_tf == 50 and reference == 'enets2' and "
          "score == 'geneformer_attn' and universe == 'kidney'", "enr_ratio_to_null", "r2", was="1.94"),
    ])
    GS = "family == '{f}' and reference == 'encode' and tissue == '{t}'"
    add([
        *RNG("nGovQmImm", "gov_sensitivity_grid.csv", GS.format(f="q_grid", t="immune"), "n_promoted", "int",
             was=("363", "755")),
        *RNG("nGovQmGrid", "gov_sensitivity_grid.csv", GS.format(f="q_grid", t="immune"), "q_moderate", "r2",
             was=("0.10", "0.40")),
        *RNG("nGovQmOeImm", "gov_sensitivity_grid.csv", GS.format(f="q_grid", t="immune"), "oe_tf", "r2",
             was=("1.58", "1.67")),
    ])
    for t, tl in TIS.items():
        add([N(f"nGovStrict{tl}", "gov_sensitivity_grid.csv",
               f"family == 'dual_rule' and config == 'strict' and reference == 'encode' and tissue == '{t}'",
               "n_promoted", "int", was={"immune": "341", "kidney": "186", "lung": "248"}[t])])
    add([
        *RNG("nGovMapJ", "gov_sensitivity_grid.csv", "family == 'credibility' and config == 'map' and reference == 'encode'",
             "jaccard_vs_primary", "r2", was=("0.97", "0.99")),
        *RNG("nGovEmN", "gov_sensitivity_grid.csv", "family == 'estimator' and config == 'em_map' and reference == 'encode'",
             "n_promoted", "int", was=("1,051", "1,659")),
        *RNG("nGovSmEncImm", "gov_sensitivity_grid.csv",
             "family == 'size_matched' and reference == 'encode' and tissue == 'immune' and "
             "(config.str.startswith('joint_nuts') or config.str.startswith('joint_em') or "
             "config.str.startswith('n_reporting') or config.str.startswith('dorothea'))", "oe_tf", "r2",
             was=("1.68", "1.77")),
    ])
    for t, tl in TIS.items():
        add([
            N(f"nGovTop{tl}Pct", "gov_concentration.csv", f"set == 'promoted' and tissue == '{t}'", "top1_share", "pct0",
              was={"immune": "15", "kidney": "14", "lung": "8"}[t]),
            N(f"nGovTopTf{tl}", "gov_concentration.csv", f"set == 'promoted' and tissue == '{t}'", "top1_tf", "str",
              was={"immune": "NFKB1", "kidney": "JUN", "lung": "EGR1"}[t]),
            N(f"nDecBudget{tl}", "gov_deconcentration.csv", f"chosen == True and tissue == '{t}'", "budget", "int",
              was={"immune": "49", "kidney": "55", "lung": "82"}[t]),
            N(f"nDecAlpha{tl}", "gov_deconcentration.csv", f"chosen == True and tissue == '{t}'", "alpha", "r0",
              was={"immune": "8", "kidney": "2", "lung": "2"}[t]),
        ])
    add([
        N("nGovTopTieLung", "gov_concentration.csv",
          "set == 'promoted' and tissue == 'lung' and top_tfs.str.split(';').str[1].str.startswith('SPI1:')",
          "top2_share", "r3", helper=True, require="abs({nGovTopTieLung} - 2 * {nGovTopLungPct}) < 1e-5",
          require_note="'EGR1 and SPI1, tied' (second TF of the lung promoted set is SPI1 with the same count)"),
        N("nGovTopEligBound", "gov_concentration.csv", "set == 'eligible_edges'", "top1_share", "ceil0", agg="max",
          scale=100.0, was="1", note="percent, rounded up ('at most')"),
        *RNG("nDecTopPct", "gov_deconcentration.csv", "chosen == True", "top1_share", "pct0", was=("4", "5")),
        *RNG("nDecCred", "gov_deconcentration.csv", "chosen == True", "mean_cred", "r2", was=("0.97", "0.98")),
        *RNG("nDecOe", "gov_deconcentration_eval.csv", "set == 'chosen' and reference == 'encode'", "oe_tf", "r2",
             was=("2.02", "2.88")),
        *RNG("nDecNeval", "gov_deconcentration_eval.csv", "set == 'chosen' and reference == 'encode'", "n_eval", "int",
             was=("17", "24")),
    ])

    # ======================================================================================= 5.7 hubs
    HQ = "score == 'family_hb' and set == 'strong_bfdr05' and quintile == 5"
    add([
        *RNG("nHubSharePct", "heldout_degree_calibration.csv", HQ, "share_set", "pct0", was=("95", "96")),
        *RNG("nHubPosPct", "heldout_degree_calibration.csv", HQ, "share_pos", "pct0", was=("47", "70")),
    ])
    for r, rl in REF.items():
        add(CI(f"nHubIdx{rl}", "heldout_degree_calibration.csv", HQ + f" and reference == '{r}'", "over_promotion",
               "over_promotion_lo", "over_promotion_hi", "r2",
               was={"encode": ("1.56", "1.34", "1.84"), "enets2": ("1.38", "1.20", "1.65"),
                    "pert_pooled": ("2.03", "1.55", "3.11")}[r]))
    add([
        *RNG("nHubAll", "heldout_degree_calibration.csv", "set == 'top500' and quintile == 5 and score != 'random'",
             "over_promotion", "r2", was=("1.33", "2.08")),
        *RNG("nHubRand", "heldout_degree_calibration.csv", "set == 'top500' and quintile == 5 and score == 'random'",
             "over_promotion", "r2", was=("0.90", "1.00")),
        N("nSimHubTfs", "sim_hub_design.csv", "section == 'design' and quantity == 'n_tf'", "value", "int"),
        N("nSimHubGenes", "sim_hub_design.csv", "section == 'design' and quantity == 'n_target'", "value", "int"),
        N("nSimHubLogPopSd", "sim_hub_design.csv", "section == 'design' and quantity == 'log_popularity_sd'", "value", "r1"),
        N("nSimHubNrep", "sim_hub_summary.csv", "method == 'family_hb' and set_rule == 'bfdr05' and metric == 'precision'",
          "n_reps", "int", agg="same", was="20"),
        *RNG("nSimOrIdx", "sim_hub_summary.csv",
             "method == 'oracle' and set_rule == 'size_matched' and metric == 'over_promotion_top_decile'", "mean", "r2",
             was=("2.36", "3.30"),
             note="oracle posterior, sets of the size of the Family-HB BFDR set, top popularity decile (Fig. 7c, Table S34)"),
        *RNG("nSimOrIdxQ", "sim_hub_summary.csv",
             "method == 'oracle' and set_rule == 'size_matched' and metric == 'over_promotion_lit_q5'", "mean", "r2",
             note="oracle posterior, equal set size, top literature-degree quintile (Fig. 7b)"),
    ])
    HQ5 = ("method == 'family_hb' and reference == 'oracle' and set_rule == 'size_matched' and "
           "metric == 'over_promotion_lit_q5' and scenario == '{s}'")
    for s_, sl in (("bio_hub", "Bio"), ("null_independent", "Null"), ("copying", "Copy")):
        add(CI(f"nSimHbOrQ{sl}", "sim_hub_contrasts.csv", HQ5.format(s=s_), "mean_diff", "ci_lo", "ci_hi", "r2",
               note="Family-HB minus oracle over-promotion index, equal size, top literature-degree quintile; t 95% CI"))
    HO = ("method == 'family_hb' and reference == 'oracle' and set_rule == 'size_matched' and "
          "metric == 'over_promotion_top_decile' and scenario == '{s}'")
    for s, sl, was in (("bio_hub", "Bio", ("-0.38", "-0.61", "-0.14")), ("null_independent", "Null", ("0.44", "0.17", "0.70")),
                       ("copying", "Copy", ("2.64", "2.07", "3.22"))):
        add(CI(f"nSimHbOr{sl}", "sim_hub_contrasts.csv", HO.format(s=s), "mean_diff", "ci_lo", "ci_hi", "r2", was=was,
               note="Family-HB minus oracle over-promotion index, equal size; t 95% CI"))
    add([
        *CI("nSimHbFdrCopy", "sim_hub_summary.csv",
            "method == 'family_hb' and set_rule == 'bfdr05' and metric == 'fdr' and scenario == 'copying'", "mean",
            "ci_lo", "ci_hi", "r2", was=("0.23", "0.16", "0.30")),
        *RNG("nSimHbFdrNoCopy", "sim_hub_summary.csv",
             "method == 'family_hb' and set_rule == 'bfdr05' and metric == 'fdr' and scenario != 'copying'", "mean",
             "r2", note="realised Strong FDR, hub scenarios without copying",
             require="{nSimHbFdrNoCopyMin} > {nSimClaimed}",
             require_note="'somewhat anti-conservative in the hub simulations without copying'"),
        N("nSimHbFpTopCopyPct", "sim_hub_summary.csv",
          "method == 'family_hb' and set_rule == 'bfdr05' and metric == 'share_fp_top_decile' and scenario == 'copying'",
          "mean", "pct0", was="70"),
        *RNG("nSimEmHb", "sim_hub_contrasts.csv",
             "method == 'em' and reference == 'family_hb' and set_rule == 'size_matched' and "
             "metric == 'over_promotion_top_decile'", "mean_diff", "r2", was=("0.96", "3.67")),
        N("nSimCiLessN", "sim_hub_contrasts.csv",
          "method in ['em', 'fisher'] and reference == 'family_hb' and set_rule == 'size_matched' and "
          "metric == 'over_promotion_top_decile'", "n_method_less", "int", agg="sum", helper=True,
          require="{nSimCiLessN} == 0", require_note="'more hub-concentrated than Family-HB in every scenario and replicate'"),
        *RNG("nSimFiHb", "sim_hub_contrasts.csv",
             "method == 'fisher' and reference == 'family_hb' and set_rule == 'size_matched' and "
             "metric == 'over_promotion_top_decile'", "mean_diff", "r2", was=("0.81", "3.26")),
        N("nSimHubRepAll", "sim_hub_contrasts.csv",
          "method == 'em' and reference == 'family_hb' and set_rule == 'size_matched' and "
          "metric == 'over_promotion_top_decile'", "n_reps", "int", agg="sum",
          note="replicates over the three scenarios", require="{nSimHubRepAll} == 3 * {nSimHubNrep}",
          require_note="'all ... replicates' of the three scenarios"),
        *RNG("nSimStHb", "sim_hub_contrasts.csv",
             "method == 'stouffer' and reference == 'family_hb' and set_rule == 'size_matched' and "
             "metric == 'over_promotion_top_decile'", "mean_diff", "r2"),
        N("nSimStGtN", "sim_hub_contrasts.csv",
          "method == 'stouffer' and reference == 'family_hb' and set_rule == 'size_matched' and "
          "metric == 'over_promotion_top_decile'", "n_method_greater", "int", agg="sum",
          note="replicates in which Stouffer is more hub-concentrated than Family-HB"),
        N("nSimStRepN", "sim_hub_contrasts.csv",
          "method == 'stouffer' and reference == 'family_hb' and set_rule == 'size_matched' and "
          "metric == 'over_promotion_top_decile'", "n_reps", "int", agg="sum", helper=True,
          require="{nSimStRepN} == {nSimHubRepAll} and {nSimStGtN} < {nSimStRepN}",
          require_note="'Stouffer in N of 60 replicates' (not all)"),
        *RNG("nSimCountHb", "sim_hub_contrasts.csv",
             "method in ['union', 'majority', 'vote_fraction'] and reference == 'family_hb' and set_rule == 'size_matched' "
             "and metric == 'over_promotion_top_decile'", "mean_diff", "r2",
             note="counting rules minus Family-HB, top-decile index, range over rules and scenarios"),
        N("nSimCountHbHiMax", "sim_hub_contrasts.csv",
          "method in ['union', 'majority', 'vote_fraction'] and reference == 'family_hb' and set_rule == 'size_matched' "
          "and metric == 'over_promotion_top_decile'", "ci_hi", "r2", agg="max", helper=True,
          require="{nSimCountHbHiMax} < 0", require_note="'less hub-concentrated than Family-HB in every scenario'"),
        N("nSimCountPrecMin", "sim_hub_contrasts.csv",
          "method in ['union', 'majority', 'vote_fraction'] and reference == 'family_hb' and set_rule == 'size_matched' "
          "and metric == 'precision'", "mean_diff", "r2", agg="min", helper=True),
        N("nSimCountPrecMax", "sim_hub_contrasts.csv",
          "method in ['union', 'majority', 'vote_fraction'] and reference == 'family_hb' and set_rule == 'size_matched' "
          "and metric == 'precision'", "mean_diff", "r2", agg="max", helper=True),
        N("nSimCountPrecHiMax", "sim_hub_contrasts.csv",
          "method in ['union', 'majority', 'vote_fraction'] and reference == 'family_hb' and set_rule == 'size_matched' "
          "and metric == 'precision'", "ci_hi", "r2", agg="max", helper=True,
          require="{nSimCountPrecHiMax} < 0", require_note="'at a precision lower by ...' in every scenario"),
        E("nSimCountPrecLossMin", "-{nSimCountPrecMax}", "r2", note="smallest precision loss of the counting rules"),
        E("nSimCountPrecLossMax", "-{nSimCountPrecMin}", "r2", note="largest precision loss of the counting rules"),
    ])

    # ======================================================================================= 5.8 replication
    RP = "universe == 'global' and level == 'study' and comparator == 'universe' and subset == '{s}'"
    add([
        N("nRepNEvalCap", "rep_pass_rates.csv", RP.format(s="all_evaluable"), "n_units", "Word", was="Four"),
        N("nRepPassAllD", "rep_pass_rates.csv", RP.format(s="all_evaluable"), "n_units", "word", was="four"),
        N("nRepPassAllNCap", "rep_pass_rates.csv", RP.format(s="all_evaluable"), "n_pass", "Word", was="Three"),
        N("nRepPassPowN", "rep_pass_rates.csv", RP.format(s="adequately_powered"), "n_pass", "word", was="three"),
        N("nRepPassPowD", "rep_pass_rates.csv", RP.format(s="adequately_powered"), "n_units", "word", was="three"),
        N("nRepPassAllLo", "rep_pass_rates.csv", RP.format(s="all_evaluable"), "wilson_lo", "r2", was="0.30"),
        N("nRepPassAllHi", "rep_pass_rates.csv", RP.format(s="all_evaluable"), "wilson_hi", "r2", was="0.95"),
        N("nRepPassPowLo", "rep_pass_rates.csv", RP.format(s="adequately_powered"), "wilson_lo", "r2", was="0.44"),
        N("nRepPassPowHi", "rep_pass_rates.csv", RP.format(s="adequately_powered"), "wilson_hi", "r2", was="1.00"),
        *RNG("nRepPow", "rep_power.csv",
             "universe == 'global' and level == 'study' and comparator == 'universe' and adequately_powered == True",
             "power", "r2", was=("0.99", "1.00")),
        *RNG("nRepMde", "rep_power.csv",
             "universe == 'global' and level == 'study' and comparator == 'universe' and adequately_powered == True",
             "mde_rr", "r2", was=("1.39", "1.56")),
        N("nRepDixitNpos", "rep_power.csv", REP_U.format(u="dixit2016"), "n_positives", "int", was="30"),
        N("nRepDixitPow", "rep_power.csv", REP_U.format(u="dixit2016"), "power", "r2", was="0.03"),
    ])
    for u, ul in (("norman2019", "Norman"), ("papalexi2021", "Papalexi"), ("replogle2022", "Replogle")):
        add(CI(f"nRepEnr{ul}", "rep_units.csv", REP_U.format(u=u), "enrichment", "enrichment_boot_lo",
               "enrichment_boot_hi", "r2",
               was={"Norman": ("1.95", "1.64", "2.45"), "Papalexi": ("2.24", "1.72", "4.86"),
                    "Replogle": ("1.57", "1.12", "4.86")}[ul]))
        add([
            N(f"nRepP{ul}", "rep_units.csv", REP_U.format(u=u), "p_exact", "e2",
              was={"Norman": "1.9e-14", "Papalexi": "3.1e-16", "Replogle": "7.1e-4"}[ul]),
            N(f"nRepResp{ul}", "rep_units.csv", REP_U.format(u=u), "resp_strat_enrichment", "r2",
              was={"Norman": "1.25", "Papalexi": "1.41", "Replogle": "1.34"}[ul]),
            N(f"nRepWrong{ul}", "rep_units.csv", REP_U.format(u=u), "wrongtf_enrichment", "r2",
              was={"Norman": "1.55", "Papalexi": "1.40", "Replogle": "1.18"}[ul]),
        ])
    add([
        N("nRepRespPMax", "rep_units.csv",
          "universe == 'global' and level == 'study' and comparator == 'universe' and endpoint_status == 'pass'",
          "resp_strat_p_exact", "ceil3", agg="max", require="{nRepRespPMax} <= {nRepAlpha}",
          require_note="'all remain significant' after the response stratification"),
        N("nRepRespEnrMax", "rep_units.csv",
          "universe == 'global' and level == 'study' and comparator == 'universe' and endpoint_status == 'pass'",
          "resp_strat_enrichment", "r2", agg="max", helper=True, require="{nRepRespEnrMax} < {nRepEndpoint}",
          require_note="'none reaches the endpoint' after the response stratification"),
    ])
    RB = "level == 'study' and profile == '{p}' and parameter == '{q}'"
    LK = "level == 'study' and row_type == 'summary' and scheme == '{s}'"
    add([
        *RNG("nRepDropTop", "rep_units.csv",
             "universe == 'global' and level == 'study' and comparator == 'universe' and endpoint_status == 'pass'",
             "enrichment_drop_top_tf", "r2", was=("1.80", "1.91"), note="passing studies"),
        *RNG("nRepSpec", "rep_units.csv",
             "universe == 'global' and level == 'study' and comparator == 'universe' and status == 'evaluable' and "
             "unit != 'dixit2016'", "specific_excess", "r2", was=("1.25", "1.60")),
        *CI("nRepMu", "rep_bayes.csv", RB.format(p="primary", q="mu"), "or_median", "or_q025", "or_q975", "r2",
            was=("2.82", "1.42", "4.09")),
        N("nRepPmu", "rep_bayes.csv", RB.format(p="primary", q="mu"), "p_or_ge_endpoint", "r2", was="0.97"),
        N("nRepNewLo", "rep_bayes.csv", RB.format(p="primary", q="theta_new"), "or_q025", "r2", was="0.92"),
        N("nRepNewHi", "rep_bayes.csv", RB.format(p="primary", q="theta_new"), "or_q975", "r2", was="6.56"),
        *CI("nRepSkept", "rep_prior_sensitivity.csv", RB.format(p="skeptical", q="mu"), "or_median", "or_q025",
            "or_q975", "r2", was=("1.21", "0.81", "1.81")),
        N("nRepSkeptP", "rep_prior_sensitivity.csv", RB.format(p="skeptical", q="mu"), "p_or_ge_endpoint", "r2",
          was="0.15"),
        N("nRepLkMeanOne", "rep_leavek.csv", LK.format(s="L1O"), "mean_abs_delta_mu", "r2", was="0.10"),
        N("nRepLkMeanTwo", "rep_leavek.csv", LK.format(s="L2O"), "mean_abs_delta_mu", "r2", was="0.16"),
        N("nRepLkMeanThree", "rep_leavek.csv", LK.format(s="L3O"), "mean_abs_delta_mu", "r2", was="0.45"),
        N("nRepLkMaxOne", "rep_leavek.csv", LK.format(s="L1O"), "max_abs_delta_mu", "r2", was="0.15"),
        N("nRepLkMaxTwo", "rep_leavek.csv", LK.format(s="L2O"), "max_abs_delta_mu", "r2", was="0.45"),
        N("nRepLkMaxThree", "rep_leavek.csv", LK.format(s="L3O"), "max_abs_delta_mu", "r2", was="1.24"),
        N("nRepLkThreeKept", "rep_leavek.csv", "level == 'study' and scheme == 'L3O' and row_type == 'fit'",
          "n_units_kept", "int", agg="max", helper=True, require="{nRepLkThreeKept} == 1",
          require_note="'with three studies removed a single study remains'"),
        N("nRepTau", "rep_threshold_loo.csv",
          "level == 'study' and support_measure == 'n_promoted_evaluable' and fit == 'loo_summary'",
          "loo_rule_agreement", "frac", was="half"),
    ])

    # ======================================================================================= 5.9 sensitivity
    SV = "analysis == 'evidence_variant' and fit_id == 'family_hb' and variant in {v}"
    HS = "universe == 'global' and reference == 'enets2' and eval_set == 'all_evaluable'"
    add([
        N("nSensPriorMin", "fit_sensitivity.csv", "analysis == 'prior'", "jaccard_bfdr_strong", "floor3", agg="min",
          was="0.988", note="'at least': rounded down"),
        *RNG("nSensVarKeep", "fit_sensitivity.csv",
             SV.format(v="['both_endpoints_cc', 'lit_no_chipderived', 'lit_no_ht', 'no_motif_coexp']"),
             "jaccard_bfdr_strong", "r2", was=("0.77", "0.99"),
             note="curated-conditional variants without the ChIP family, except the merged ExTRI/NTNU.Curated panel "
                  "(reported separately)"),
        N("nSensVarMergedGlob", "fit_sensitivity.csv",
          SV.format(v="['lit_merged_extri_ntnu']") + " and universe == 'global'", "jaccard_bfdr_strong", "r2"),
        N("nSensVarMergedImm", "fit_sensitivity.csv",
          SV.format(v="['lit_merged_extri_ntnu']") + " and universe == 'immune'", "jaccard_bfdr_strong", "r2"),
        *RNG("nSensVarChip", "fit_sensitivity.csv", SV.format(v="['chip_prespecified_cc']"), "jaccard_bfdr_strong", "r2",
             note="ChIP family added to the curated-conditional model"),
        *RNG("nSensVarLeak", "fit_sensitivity.csv", SV.format(v="['prespecified', 'both_endpoints']"),
             "jaccard_bfdr_strong", "r2", was=("0.58", "0.70"),
             note="motif/coexpression genome-wide (prespecified mask) or both endpoints, without the literature condition"),
        *RNG("nHsVarMask", "heldout_sensitivity_fits.csv",
             HS + " and kind == 'evidence_variant' and fit_id == 'family_hb' and variant in ['prespecified', 'both_endpoints']",
             "diff_vs_primary", "r2", was=("0.72", "0.79")),
        *RNG("nSensGrpLinJac", "fit_sensitivity.csv", "analysis == 'family_grouping' and fit_id == 'family_hb__fam_lineage'",
             "jaccard_bfdr_strong", "r2", was=("44", "53"),
             note="Strong-set Jaccard index of the lineage grouping with the primary (printed earlier as a percentage)"),
        *RNG("nSensGrpSing", "fit_sensitivity.csv", "analysis == 'family_grouping' and fit_id == 'family_hb__fam_singleton'",
             "jaccard_bfdr_strong", "r2", was=("0.21", "0.26")),
        *RNG("nSensGrpDd", "fit_sensitivity.csv", "analysis == 'family_grouping' and fit_id == 'family_hb__fam_datadriven'",
             "jaccard_bfdr_strong", "r2", was=("0.99", "1.00")),
        N("nGrpLinJGlob", "fit_sensitivity.csv", "analysis == 'family_grouping' and fit_id == 'family_hb__fam_lineage' and universe == 'global'", "jaccard_bfdr_strong", "r3", helper=True),
        N("nGrpLinRefGlob", "fit_sensitivity.csv", "analysis == 'family_grouping' and fit_id == 'family_hb__fam_lineage' and universe == 'global'", "bfdr_strong_n_ref", "int", helper=True),
        N("nGrpLinAltGlob", "fit_sensitivity.csv", "analysis == 'family_grouping' and fit_id == 'family_hb__fam_lineage' and universe == 'global'", "bfdr_strong_n_alt", "int", helper=True),
        E("nGrpLinKeepGlob", "{nGrpLinJGlob} * ({nGrpLinRefGlob} + {nGrpLinAltGlob}) / (1 + {nGrpLinJGlob}) / {nGrpLinRefGlob}",
          "r3", helper=True, note="share of the primary Strong edges in the lineage Strong set: |A and B| = J (|A| + |B|) / (1 + J)"),
        N("nGrpLinJImm", "fit_sensitivity.csv", "analysis == 'family_grouping' and fit_id == 'family_hb__fam_lineage' and universe == 'immune'", "jaccard_bfdr_strong", "r3", helper=True),
        N("nGrpLinRefImm", "fit_sensitivity.csv", "analysis == 'family_grouping' and fit_id == 'family_hb__fam_lineage' and universe == 'immune'", "bfdr_strong_n_ref", "int", helper=True),
        N("nGrpLinAltImm", "fit_sensitivity.csv", "analysis == 'family_grouping' and fit_id == 'family_hb__fam_lineage' and universe == 'immune'", "bfdr_strong_n_alt", "int", helper=True),
        E("nGrpLinKeepImm", "{nGrpLinJImm} * ({nGrpLinRefImm} + {nGrpLinAltImm}) / (1 + {nGrpLinJImm}) / {nGrpLinRefImm}",
          "r3", helper=True, note="share of the primary Strong edges in the lineage Strong set: |A and B| = J (|A| + |B|) / (1 + J)"),
        E("nSensGrpLinKeepPctMin", "min({nGrpLinKeepGlob}, {nGrpLinKeepImm})", "pct0",
          note="share of the primary Strong edges kept by the lineage grouping, smaller over global and immune"),
        E("nSensGrpLinKeepPctMax", "max({nGrpLinKeepGlob}, {nGrpLinKeepImm})", "pct0"),
        E("nSensGrpLinSizeMin", "min({nGrpLinAltGlob} / {nGrpLinRefGlob}, {nGrpLinAltImm} / {nGrpLinRefImm})", "r1",
          note="size of the lineage Strong set relative to the primary"),
        E("nSensGrpLinSizeMax", "max({nGrpLinAltGlob} / {nGrpLinRefGlob}, {nGrpLinAltImm} / {nGrpLinRefImm})", "r1"),
        N("nGrpSingJGlob", "fit_sensitivity.csv", "analysis == 'family_grouping' and fit_id == 'family_hb__fam_singleton' and universe == 'global'", "jaccard_bfdr_strong", "r3", helper=True),
        N("nGrpSingRefGlob", "fit_sensitivity.csv", "analysis == 'family_grouping' and fit_id == 'family_hb__fam_singleton' and universe == 'global'", "bfdr_strong_n_ref", "int", helper=True),
        N("nGrpSingAltGlob", "fit_sensitivity.csv", "analysis == 'family_grouping' and fit_id == 'family_hb__fam_singleton' and universe == 'global'", "bfdr_strong_n_alt", "int", helper=True),
        E("nGrpSingKeepGlob", "{nGrpSingJGlob} * ({nGrpSingRefGlob} + {nGrpSingAltGlob}) / (1 + {nGrpSingJGlob}) / {nGrpSingRefGlob}",
          "r3", helper=True, note="share of the primary Strong edges in the singleton Strong set: |A and B| = J (|A| + |B|) / (1 + J)"),
        N("nGrpSingJImm", "fit_sensitivity.csv", "analysis == 'family_grouping' and fit_id == 'family_hb__fam_singleton' and universe == 'immune'", "jaccard_bfdr_strong", "r3", helper=True),
        N("nGrpSingRefImm", "fit_sensitivity.csv", "analysis == 'family_grouping' and fit_id == 'family_hb__fam_singleton' and universe == 'immune'", "bfdr_strong_n_ref", "int", helper=True),
        N("nGrpSingAltImm", "fit_sensitivity.csv", "analysis == 'family_grouping' and fit_id == 'family_hb__fam_singleton' and universe == 'immune'", "bfdr_strong_n_alt", "int", helper=True),
        E("nGrpSingKeepImm", "{nGrpSingJImm} * ({nGrpSingRefImm} + {nGrpSingAltImm}) / (1 + {nGrpSingJImm}) / {nGrpSingRefImm}",
          "r3", helper=True, note="share of the primary Strong edges in the singleton Strong set: |A and B| = J (|A| + |B|) / (1 + J)"),
        E("nSensGrpSingKeepPctMin", "min({nGrpSingKeepGlob}, {nGrpSingKeepImm})", "pct0",
          note="share of the primary Strong edges kept by the singleton grouping, smaller over global and immune"),
        E("nSensGrpSingKeepPctMax", "max({nGrpSingKeepGlob}, {nGrpSingKeepImm})", "pct0"),
        E("nSensGrpSingSizeMin", "min({nGrpSingAltGlob} / {nGrpSingRefGlob}, {nGrpSingAltImm} / {nGrpSingRefImm})", "r1",
          note="size of the singleton Strong set relative to the primary"),
        E("nSensGrpSingSizeMax", "max({nGrpSingAltGlob} / {nGrpSingRefGlob}, {nGrpSingAltImm} / {nGrpSingRefImm})", "r1"),
        *RNG("nGrpAri", "fit_grouping_stability.csv",
             "partition == 'summary' and source in ['cv_training_fold', 'tf_bootstrap']", "ari_vs_modality", "r2",
             was=("0.52", "0.83")),
        N("nGrpAriFullGlob", "fit_grouping_stability.csv",
          "partition == 'summary' and source == 'full_data' and universe == 'global'", "ari_vs_modality", "r2",
          note="data-driven full-data partition against the modality partition, global"),
        N("nGrpAriFullImm", "fit_grouping_stability.csv",
          "partition == 'summary' and source == 'full_data' and universe == 'immune'", "ari_vs_modality", "r2",
          helper=True, require="{nGrpAriFullImm} == 1", require_note="'recovers the modality families exactly in immune'"),
        N("nGrpOregMotifGlob", "fit_family_groupings.csv",
          "universe == 'global' and annotator == 'ORegAnno' and datadriven == 'DD2'", "annotator", "int", agg="count",
          helper=True, require="{nGrpOregMotifGlob} == 1 and {nGrpLitOffGlob} == 1 and {nGrpMotifDdGlob} == 2",
          require_note="'ORegAnno, the only literature annotator outside the literature cluster, joins the motif cluster'"),
        N("nGrpLitOffGlob", "fit_family_groupings.csv",
          "universe == 'global' and modality == 'LIT' and datadriven != 'DD1'", "annotator", "int", agg="count",
          helper=True),
        N("nGrpMotifDdGlob", "fit_family_groupings.csv",
          "universe == 'global' and modality == 'MOTIF' and datadriven == 'DD2'", "annotator", "int", agg="count",
          helper=True),
        N("nHsMaxNonLeak", "heldout_sensitivity_fits.csv",
          HS + " and kind in ['prior', 'evidence_variant', 'family_grouping'] and fit_id.str.startswith('family_hb')"
          " and variant not in ['prespecified', 'both_endpoints', 'chip_prespecified_cc']", "diff_vs_primary", "ceil2",
          agg="absmax", was="0.09",
          note="largest |change| of global ENETS2 AP lift over the Family-HB prior, evidence-variant and grouping fits "
               "without the leaking variants; 'not more than': rounded up"),
    ])
    TO = ("grid == '{g}' and split_type == '{s}' and criterion == 'pooled_ap_lift' and selector == '{sel}' and "
          "fold_set == '{f}'")
    add([
        N("nTuneNConfigs", "tune_optimism.csv", TO.format(g="all", s="random_tf_half", sel="label_tuned", f="all_folds"),
          "n_configs", "int", was="20"),
        N("nTuneNwin", "tune_optimism.csv",
          TO.format(g="primary_evidence", s="random_tf_half", sel="label_tuned", f="all_folds"), "n_distinct_selected",
          "word", was="nine"),
        N("nTuneIn", "tune_optimism.csv",
          TO.format(g="primary_evidence", s="random_tf_half", sel="label_tuned", f="all_folds"),
          "in_minus_primary_mean", "r3", was="0.005"),
        *CI("nTuneOut", "tune_optimism.csv",
            TO.format(g="primary_evidence", s="random_tf_half", sel="label_tuned", f="all_folds"),
            "out_minus_primary_mean", "out_minus_primary_q025", "out_minus_primary_q975", "r3",
            was=("-0.001", "-0.013", "0.008"), note="2.5-97.5% over splits"),
        N("nTuneAllFreqPct", "tune_optimism.csv", TO.format(g="all", s="random_tf_half", sel="label_tuned", f="all_folds"),
          "selected_config_modal_freq", "pct0", was="74"),
        N("nTuneAllRetPct", "tune_optimism.csv", TO.format(g="all", s="random_tf_half", sel="label_tuned", f="all_folds"),
          "retained_gain_ratio", "pct0", was="69"),
        N("nTuneLosoPct", "tune_optimism.csv", TO.format(g="all", s="loso", sel="label_tuned", f="all_folds"),
          "retained_gain_ratio", "pct0", was="29"),
        N("nTuneLosoExcl", "tune_optimism.csv", TO.format(g="all", s="loso", sel="label_tuned", f="excl_low_support"),
          "retained_gain_ratio", "r2", helper=True, require="{nTuneLosoExcl} >= 1",
          require_note="'or all of it once the uninformative Dixit fold is dropped'"),
        N("nTuneLosoHbOut", "tune_optimism.csv",
          TO.format(g="primary_evidence", s="loso", sel="label_tuned", f="all_folds"), "out_minus_primary_mean", "r3",
          note="hyperparameter grid, leave one study out: held-out AP lift of the tuned choice minus Family-HB (mean over folds)"),
        N("nTuneLosoHbFrac", "tune_optimism.csv",
          TO.format(g="primary_evidence", s="loso", sel="label_tuned", f="all_folds"), "frac_out_gt_primary", "r2",
          helper=True, require="{nTuneLosoHbFrac} == 1", require_note="'positive in all four folds'"),
        N("nTuneLosoHbNfold", "tune_optimism.csv",
          TO.format(g="primary_evidence", s="loso", sel="label_tuned", f="all_folds"), "n_splits", "word"),
        N("nTuneLosoHbExcl", "tune_optimism.csv",
          TO.format(g="primary_evidence", s="loso", sel="label_tuned", f="excl_low_support"), "out_minus_primary_mean",
          "r3", note="the same without the low-support Dixit fold (post hoc subset)"),
        N("nTuneNullOut", "tune_optimism.csv",
          "grid == 'primary_evidence' and split_type == 'random_tf_half' and criterion == 'pooled_ap_lift_nullnorm' "
          "and selector == 'label_tuned' and fold_set == 'all_folds'", "out_minus_primary_mean", "r3",
          note="held-out gain with the TF-composition-normalised criterion (post hoc)"),
        N("nTuneNullFracPct", "tune_optimism.csv",
          "grid == 'primary_evidence' and split_type == 'random_tf_half' and criterion == 'pooled_ap_lift_nullnorm' "
          "and selector == 'label_tuned' and fold_set == 'all_folds'", "frac_out_gt_primary", "pct0",
          note="share of halves in which the tuned choice beats Family-HB on the held-out TFs"),
        N("nTuneJointHbPrecPct", "tune_weight_summary.csv",
          "rob_metric == 'precision' and candidate_set == 'lcm5' and config_id == 'family_hb'",
          "share_grid_selected_full", "pct0", note="share of weight points at which the joint objective selects Family-HB"),
        N("nTuneJointHbOther", "tune_weight_summary.csv",
          "config_id == 'family_hb' and not (rob_metric == 'precision' and candidate_set == 'lcm5')",
          "share_grid_selected_full", "r2", agg="max", helper=True, require="{nTuneJointHbOther} == 0",
          require_note="'never with the F1-based metric or when the combination rules are candidates'"),
        N("nTuneLf", "tune_optimism.csv",
          TO.format(g="primary_evidence", s="random_tf_half", sel="label_free_cvll", f="all_folds"),
          "out_minus_primary_mean", "r4", was="0.0002"),
    ])

    # ======================================================================================= 5.10 computation
    SC = "cid == '{c}'"
    PT = "selected == True and mode in {m}"
    add([
        N("nScNRealMio", "scale_summary.csv", SC.format(c="template_real_K21"), "N", "r1", scale=1e-6, was="9.3"),
        N("nScPRealApprox", "scale_summary.csv", SC.format(c="template_real_K21"), "P", "approx1", was="20,000",
          note="rounded to one significant figure ('about')"),
        N("nScCompRealW", "scale_summary.csv", SC.format(c="template_real_K21"), "compress_s", "wround", was="one",
          note="seconds, rounded ('about')"),
        N("nScVgRealMs", "scale_summary.csv", SC.format(c="template_real_K21"), "t_vg_s", "r1", scale=1000.0,
          was="1.5"),
        N("nPipeTablesMin", "pipeline_stage_timings.csv", PT.format(m="['tables_only', 'full+tables']"), "wall_time_s",
          "r0", agg="sum", scale=1 / 60, note="minutes, tables-only run"),
        N("nPipeTablesPeakGiB", "pipeline_stage_timings.csv", PT.format(m="['tables_only', 'full+tables']"),
          "peak_rss_mib", "r1", agg="max", scale=1 / 1024, note="GiB, largest single step of the tables-only run"),
        N("nPipeFullHours", "pipeline_stage_timings.csv", PT.format(m="['full', 'full+tables']"), "wall_time_s", "r0",
          agg="sum", scale=1 / 3600,
          note="hours of compute, full run, as recorded on a shared machine (mostly runtimes of the original, "
               "contended runs; isolated re-runs of the dominant steps are 3-4 times faster: not a bound)"),
        N("nPipeFullRecS", "pipeline_stage_timings.csv",
          PT.format(m="['full', 'full+tables']") + " and timing_source.str.startswith('recorded')", "wall_time_s", "r0",
          agg="sum", helper=True),
        N("nPipeFullS", "pipeline_stage_timings.csv", PT.format(m="['full', 'full+tables']"), "wall_time_s", "r0",
          agg="sum", helper=True),
        E("nPipeRecordedPct", "{nPipeFullRecS} / {nPipeFullS}", "pct0",
          note="share of the full-run seconds that are recorded runtimes of the original runs"),
        N("nPipeMapRecS", "pipeline_stage_timings.csv",
          "stage == 'fit_primary' and step == 'map' and timing_source.str.startswith('recorded')", "wall_time_s", "r1",
          helper=True),
        N("nPipeMapIsoS", "pipeline_stage_timings.csv",
          "stage == 'fit_primary' and step == 'map' and timing_source.str.startswith('measured') and selected == True", "wall_time_s", "int",
          note="all posterior-mode fits (five estimators, four universes) re-run one at a time"),
        N("nPipeNutsKidRecS", "pipeline_stage_timings.csv",
          "stage == 'fit_primary' and step == 'nuts' and scope == 'kidney, map_laplace init' and "
          "timing_source.str.startswith('recorded')", "wall_time_s", "r1", helper=True),
        N("nPipeNutsKidIsoMin", "pipeline_stage_timings.csv",
          "stage == 'fit_primary' and step == 'nuts' and scope == 'kidney, map_laplace init' and "
          "timing_source.str.startswith('measured') and selected == True", "wall_time_s", "r1", scale=1 / 60, note="minutes"),
        E("nPipeIsoRatioMin", "min({nPipeMapRecS} / {nPipeMapIsoS}, {nPipeNutsKidRecS} / ({nPipeNutsKidIsoMin} * 60))",
          "r1", note="recorded over isolated wall time, posterior-mode fits and kidney NUTS (smaller ratio)"),
        E("nPipeIsoRatioMax", "max({nPipeMapRecS} / {nPipeMapIsoS}, {nPipeNutsKidRecS} / ({nPipeNutsKidIsoMin} * 60))",
          "r1", note="recorded over isolated wall time (larger ratio)"),
        N("nArchiveGiB", "pipeline_release_check.csv", "check == 'archive' and key == 'gib_compressed'", "value", "r2",
          agg="same", note="GiB, compressed supporting archive of the stored outputs the tables-only run reads "
                           "(src/r2/release_check.py)"),
        N("nRtMapGlob", "fit_convergence.csv", "fit_id == 'family_hb' and universe == 'global'", "runtime_s", "int",
          was="408"),
        *RNG("nRtMapTis", "fit_convergence.csv", "fit_id == 'family_hb' and universe != 'global'", "runtime_s", "int",
             was=("80", "97")),
        N("nRtNutsGlobMinutes", "fit_convergence.csv", f"{HBN} and universe == 'global'", "runtime_s", "r0", scale=1 / 60,
          was="88", note="minutes"),
        *RNG("nRtNutsTisMinutes", "fit_convergence.csv", f"{HBN} and universe != 'global'", "runtime_s", "r0", scale=1 / 60,
             was=("21", "52"), note="minutes"),
        N("nScPN", "scale_fits.csv", "session == 'a' and quantity == 'P vs N' and subset == 'template chunked_stream K=21'",
          "slope", "r2", was="0.63"),
        N("nScMapTenKOne", "scale_summary.csv", SC.format(c="template_10xreal_K21"), "map_10_s", "int", was="685"),
        N("nScKOne", "scale_summary.csv", SC.format(c="template_10xreal_K21"), "K", "int", was="21"),
        N("nScMapTenKTwoHours", "scale_summary.csv", SC.format(c="template_10xreal_K42"), "map_10_s", "r1",
          scale=1 / 3600, was="1.5", note="hours (extrapolated)"),
        N("nScKTwo", "scale_summary.csv", SC.format(c="template_10xreal_K42"), "K", "int", was="42"),
        N("nScNutsProjMinHours", "scale_nuts.csv", "size == '10xreal' and status == 'projected'", "t_nuts_projected_s",
          "r0", agg="min", scale=1 / 3600, was="2", helper=True,
          note="hours; smallest step-count projection over both sessions (lower bound)"),
        N("nScNutsProjMaxHours", "scale_nuts.csv", "size == '10xreal' and status == 'projected'",
          "t_nuts_projected_cap_s", "r0", agg="max", scale=1 / 3600, was="12", helper=True,
          note="hours; largest projection at the tree-depth cap over both sessions"),
        N("nScNutsKOneLoHours", "scale_nuts.csv", "size == '10xreal' and status == 'projected' and K == {nScKOne}",
          "t_nuts_projected_s", "r1", agg="min", scale=1 / 3600,
          note="hours, K = nScKOne: smallest step-count projection over both sessions (lower bound)"),
        N("nScNutsKOneHiHours", "scale_nuts.csv", "size == '10xreal' and status == 'projected' and K == {nScKOne}",
          "t_nuts_projected_cap_s", "r1", agg="max", scale=1 / 3600,
          note="hours, K = nScKOne: largest projection at the tree-depth cap over both sessions"),
        N("nScNutsKTwoLoHours", "scale_nuts.csv", "size == '10xreal' and status == 'projected' and K == {nScKTwo}",
          "t_nuts_projected_s", "r1", agg="min", scale=1 / 3600,
          note="hours, K = nScKTwo: smallest step-count projection over both sessions (lower bound)"),
        N("nScNutsKTwoHiHours", "scale_nuts.csv", "size == '10xreal' and status == 'projected' and K == {nScKTwo}",
          "t_nuts_projected_cap_s", "r1", agg="max", scale=1 / 3600,
          note="hours, K = nScKTwo: largest projection at the tree-depth cap over both sessions"),
        N("nScMapRestarts", "scale_summary.csv", SC.format(c="template_10xreal_K21"), "map_restarts_run", "word",
          note="MAP restarts of the scaling runs"),
        N("nScMapFiftyKOneHours", "scale_summary.csv", SC.format(c="template_10xreal_K21"), "map_50_projected_s", "r1",
          scale=1 / 3600, note="hours, projected for the 50 restarts of the Family-HB fits"),
        N("nScMapFiftyKTwoHours", "scale_summary.csv", SC.format(c="template_10xreal_K42"), "map_50_projected_s", "r0",
          scale=1 / 3600, note="hours, projected for the 50 restarts of the Family-HB fits"),
        N("nScNutsChains", "scale_nuts.csv", "status == 'measured'", "chains", "word", agg="same"),
        N("nScNutsWarm", "scale_nuts.csv", "status == 'measured'", "warmup", "int", agg="same"),
        N("nScNutsDraw", "scale_nuts.csv", "status == 'measured'", "samples", "int", agg="same"),
        N("nScNutsProdMinHours", "scale_nuts.csv", "size == '10xreal' and status == 'projected'",
          "t_nuts_4x1000_projected_s", "floor0", agg="min", scale=1 / 3600,
          note="hours; smallest projection for 4 chains x (1,000 + 1,000) iterations (step-count projection: lower bound)"),
    ])

    # ======================================================================================= discussion
    SR = "in_ranking_panel == True and method == 'scgpt_attn' and tissue == '{t}'"
    add([
        *[N(f"nScRankBest{tl}", "rob_method_scores_full.csv", SR.format(t=t), "rank_f1", "int", agg="min", helper=True,
            note="best rank of raw scGPT attention over the ten estimators and rules") for t, tl in TIS.items()],
        E("nScBottom", "max({nRobNMethodsImm} - {nScRankBestImm} + 1, {nRobNMethodsKidLung} - {nScRankBestKid} + 1, "
                       "{nRobNMethodsKidLung} - {nScRankBestLung} + 1)", "word",
          note="raw scGPT attention is among this many lowest-ranked methods of every tissue panel under every estimator"),
    ])
    add([
        *RNG("nFmCells", "fm_extraction_cells.csv", "model in ['scgpt', 'geneformer']", "n_cells", "int",
             was=("1,200", "2,000")),
    ])
    return S
