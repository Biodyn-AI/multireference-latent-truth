"""Dated decisions of the manuscript-asset scripts (module ``paper_assets``) for the shared decision log
``data/r2/tables/eval_analysis_decisions.csv``.

These are presentation decisions taken after the held-out results had been seen (status ``post_hoc``);
none changes an analysis table.  ``python -m r2.paper_assets.decisions`` (re)writes the module's rows
with the locked, atomic writer of ``r2.heldout_eval`` (idempotent: the rows are replaced, not appended).
"""
from __future__ import annotations

import sys
from pathlib import Path

DATE = "2026-10-14"
MODULE = "paper_assets"

DECISIONS = [
    dict(id="assets_rob_intervals", date=DATE, status="post_hoc",
         item="tab_robustness and fig3 print one interval type: bootstrap over configurations clustered by the dropped "
              "source / family / lineage (rob_method_ranking and rob_estimator_contrasts *_cluster_lo/hi); in the global "
              "universe every source configuration is its own cluster, so the per-configuration bootstrap of "
              "rob_edge_stability is the same type",
         note="assets_tables_VERIFY M-T1, assets_figures_VERIFY F4; the unclustered *_boot_lo/hi columns of the "
              "method ranking are no longer printed"),
    dict(id="assets_rob_noswitch", date=DATE, status="post_hoc",
         item="tab_robustness shows D and the paired difference to Family-HB under source dropout with and without the "
              "configurations in which Family-HB switches posterior mode (subset no_fhb_mode_switch), and fig3a the "
              "no-switch D as open markers; the main table replaces the top-2,000 Jaccard (artefact in the tissue "
              "universes, robustness minor 1) by the size-matched Jaccard with interval and the Spearman correlation",
         note="robfix (fitfix_VERIFY M1); the no-switch rows are a decomposition and do not replace the prespecified "
              "rows"),
    dict(id="assets_heldout_columns", date=DATE, status="post_hoc",
         item="tab_heldout adds the composition-adjusted AP lift (ap_lift_comp_adj, 95% TF-cluster bootstrap) with "
              "its descriptive Holm p over the six confirmatory comparators (p_holm_6comparators) and the unstratified "
              "Enr@500 with interval; Family-HB rows are labelled (MAP) and (NUTS) and every paired test is Family-HB "
              "(MAP) minus comparator (checked at build time)",
         note="assets_tables_VERIFY M-T3/M-T4; heldfix composition adjustment"),
    dict(id="assets_compute_tables", date=DATE, status="post_hoc",
         item="tab_compute holds two floats: per-stage wall time and peak memory of the real pipeline "
              "(pipeline_stage_timings.csv; label tab:compute) and the synthetic scaling experiment (label "
              "tab:compute-scaling); projected NUTS cells print the range from the smallest step-count projection "
              "(lower bound) to the largest tree-depth-cap projection over the two sessions (scale_nuts.csv); MAP "
              "projections (lower bounds) and extrapolations have their own footnotes",
         note="assets_tables_VERIFY M-T2, scale_VERIFY M1/M3; tab_pipeline kept as a stand-alone copy of the first "
              "float"),
    dict(id="assets_modes_sim_primary", date=DATE, status="post_hoc",
         item="tabS_modes_sim, tabS_recovery, fig2c and the digest use the design-faithful simulator (simulator = "
              "faithful_cc) as primary and show the stratum-weight and fixed-mask simulators as sensitivity; "
              "logistic P(dominant = truth) cells flagged extrapolated are marked and rows extrapolated in every "
              "column are omitted; interval types stated per row (Wilson for probabilities, 10th-90th percentile "
              "over replicates for medians and means)",
         note="assets_tables_VERIFY M-T5, modes2"),
    dict(id="assets_figure_floats", date=DATE, status="post_hoc",
         item="every figure is written with a generated LaTeX float wrapper <name>_float.tex (fixed label "
              "fig:design ... fig:replication, fig:S-ppc ... fig:S-leavek; caption checked for non-ASCII characters, "
              "unescaped %, unbalanced $ and braces)",
         note="assets_suppfig_VERIFY m7"),
    dict(id="assets_digest_interpretation", date=DATE, status="post_hoc",
         item="results digest interpretation and caveats rewritten for the verifier findings (assets_tables D1-D12, "
              "assets_suppfig H1-H2, M1-M9, m1-m12, assets_figures X1-X2) and the modes2, robfix and heldfix results; "
              "every number is still a fact read from a CSV",
         note="no analysis table changed"),
]


# Binding of the manuscript prose to the analysis tables (paper_assets/prose_numbers.py; module "binding").
MODULE_BINDING = "paper_assets_binding"
BINDING_DECISIONS = [
    dict(id="binding_number_macros", date=DATE, status="post_hoc",
         item="every number of the manuscript prose (sec_*.tex, manuscript front/back matter, supplement prose) is a "
              "macro of paper/r2/numbers.tex written by make_paper_assets.py numbers from one CSV cell (row filter + "
              "column, or a stated aggregation over the selected rows), a PRESPECIFIED constant, a frozen config / "
              "code constant (fit_primary GRAD_TOL, RHAT_FLAG) or a documented arithmetic of earlier macros; the "
              "checker fails on any remaining literal result number (allow-list with reasons: equation constants, "
              "years, interval level 95%, probability threshold 0.5, cell-line and model names, address, the "
              "upstream HVG setting 5,000 and the archive size)",
         note="binding module; audit outputs/r2/reports/binding_number_audit.csv; no analysis table changed"),
    dict(id="binding_safe_rounding", date=DATE, status="post_hoc",
         item="statements of the form 'at least', 'at most', 'below', 'fewer than', 'p <=' and 'not more than' print "
              "their bound rounded towards the safe side (floor / ceil formats); this changed four printed values: "
              "promoted-set Jaccard at second-mode mass 0.05 0.95 -> 0.94 (0.946), prior-variant Strong-set Jaccard "
              "0.988 -> 0.987, runner-up Strong-set Jaccard 0.002 -> 0.003 (0.0024), largest non-leaking held-out "
              "AP-lift change 0.09 -> 0.10 (0.091)",
         note="modes2_VERIFY m3 (promoted-set Jaccard)"),
    dict(id="binding_corrections", date=DATE, status="post_hoc",
         item="prose numbers that did not match the current tables were replaced by the CSV values: wrong-mode Strong "
              "FDR 0.93 -> 0.94 (lower end), EM Strong realised FDR in the recovery simulation 0.57-0.60 -> 0.58-0.62 "
              "(design-faithful simulator), NTNU.Curated reports also in ExTRI about 90% -> 93%, latent-class within-TF "
              "AUROC lower end 0.468 -> 0.473 (ENCODE and pooled perturbation), number of tuning configurations "
              "20 -> 19, NUTS projection at ten times the real size up to 12 h -> 21 h (tree-depth cap, both "
              "sessions), Geneformer median co-observing cells 'below 1' -> 0",
         note="each listed with old and new value in the binding report and the audit CSV (prose_before_binding)"),
    dict(id="binding_rob_scope", date=DATE, status="post_hoc",
         item="robustness paragraph: the size-matched Jaccards are the tissue-pooled values of rob_estimator_contrasts "
              "(Family-HB 0.87, EM 0.89, Fisher 0.86, union 0.96, dependence-aware 0.90; were the immune values 0.90, "
              "0.89, 0.87, 0.96, 0.91 of an earlier table) so that they match the paired contrasts; the BFDR-set "
              "Jaccards are labelled as global-universe values",
         note="robfix_VERIFY m4"),
    dict(id="binding_pending_fills", date=DATE, status="post_hoc",
         item="the five [PENDING] placeholders of sec_results were filled from the final tables with the verifier "
              "corrections: design-faithful mode-selection simulation incl. the reverse direction and the largest "
              "wrong-selection gap (modes2, modes2_VERIFY M1), recovery FDR (design-faithful), mode-switch "
              "decomposition of the robustness contrasts (robfix, robfix_VERIFY M1 / m1-m5; no EM-referenced "
              "contrast is claimed because none is tabulated), composition-adjusted AP lift (heldfix, heldfix_VERIFY "
              "M1 / m1-m2, global universe), per-stage pipeline cost (pipeline_stage_timings)",
         note="the \\pending macro was removed from manuscript.tex and supplement.tex (unused)"),
    dict(id="binding_bridge_interval", date=DATE, status="post_hoc",
         item="immune second-mode probability printed with its Monte Carlo interval and, in addition, the range over "
              "posterior draw sets and bridge proposals (modes_bridge p2_range_draw_sets_proposals_lo/hi)",
         note="modes2_VERIFY m4 (the split interval excludes the canonical-chain estimate)"),
]


# Final polish of the manuscript assets and prose (module "paper_assets_fixer"); presentation only.
MODULE_FIXER = "paper_assets_fixer"
FIXER_DECISIONS = [
    dict(id="fixer_hb_em_one_interval", date=DATE, status="post_hoc",
         item="the prose cites the Family-HB minus EM rank-disagreement contrast as the negated EM-minus-Family-HB "
              "contrast of tab_robustness (rob_estimator_contrasts reference = family_hb, cluster bootstrap), so that "
              "prose and table report one interval; the EM-referenced rows remain the source of the other contrasts "
              "with EM (dependence-aware, multifactor, rules)",
         note="rev_r1 m1; no analysis table changed"),
    dict(id="fixer_replication_underpowered", date=DATE, status="post_hoc",
         item="tab_replication prints 'underpowered' (not 'fail') for an evaluable study that does not pass and is "
              "below its locked support floor (Dixit); it stays in the denominator of the all-evaluable pass rate",
         note="rev_r1 M1; rep_units endpoint_status unchanged"),
    dict(id="fixer_fm_sens_table", date=DATE, status="post_hoc",
         item="new supplementary table tabS_fm_sens: TF-matched ChIP enrichment of predicted sets of 20, 50 and 100 "
              "targets per TF (heldout_methods_context) and the co-observation restriction of attention "
              "(fm_min_coobs_sensitivity: retained share, overlap with the 1/L null)",
         note="rev_r1 M2; both source tables existed; no analysis table changed"),
    dict(id="fixer_expression_vs_hb_predicted_sets", date=DATE, status="post_hoc",
         item="sec_results 5.3 compares the top-50 predicted sets of GRNBoost2 and |Pearson| with that of Family-HB "
              "(NUTS, tissue fit) on ENCODE and ENETS2 (point estimates of heldout_methods_context enr_tf_matched)",
         note="rev_r1 M4"),
    dict(id="fixer_hub_sim_metrics", date=DATE, status="post_hoc",
         item="hub-simulation prose: oracle index at equal set size (size_matched) for the top popularity decile and "
              "the top literature-degree quintile, Family-HB minus oracle on both scales; previously the oracle index "
              "of its own BFDR set was quoted next to equal-size contrasts",
         note="rev_r1 m2, rev_stats M3"),
    dict(id="fixer_tables_presentation", date=DATE, status="post_hoc",
         item="table presentation: p >= 0.1 printed with two decimals; continued table parts keep the number of the "
              "first part (ContinuedFloat); one wording of the post hoc note; accession row in tab_cohorts and source "
              "URLs in tab_chip and tabS_sources; code identifiers replaced by plain text (complexity and scaling "
              "tables, step names of tab_compute)",
         note="rev_edit M2, M3, M5, M6, m6; rev_r1 M5"),
]


# Final verification round (module "paper_assets_final", 2026-10-15); presentation and release only.
MODULE_FINAL = "paper_assets_final"
FINAL_DECISIONS = [
    dict(id="final_release_file_list", date="2026-10-15", status="post_hoc",
         item="the release file list (git ls-files -co --exclude-standard, used by release_check.py) excludes planning/, "
              "REVISION_PLAN.md and LaTeX build by-products (.gitignore; paper/main_clean.aux kept for build_marked.py); "
              "the reviewer digest takes the point titles from paper/r2/letter_quotes.tex instead of the planning file, "
              "and letter_numbers.py keeps the committed quotes when planning/ is absent",
         note="fv_letter B1 (the decision letter in planning/ holds a personal submission-system link); procedural, "
              "no analysis table changed"),
    dict(id="final_prose_scope", date="2026-10-15", status="post_hoc",
         item="prose wording matched to the tables: data-driven family partition against the modality partition "
              "(fit_grouping_stability full_data ARI), Strong-set containment and size ratio of the lineage and "
              "singleton groupings (fit_sensitivity: intersection = J(a+b)/(1+J)), hub-simulation contrasts named per "
              "score (sim_hub_contrasts size_matched: EM, Fisher, Stouffer, union, majority, vote fraction), per-replicate "
              "class-weight range of the informative-observation hub fits (sim_hub_modes w_T_min/max), co-observation "
              "medians over all universe pairs with the share co-observed in at least five cells (fm_universe_summary), "
              "projected NUTS cost per annotator count (scale_nuts), Exploratory-tier enrichment (gov_tier_validation), "
              "Holm p at the resampling floor printed as an upper bound",
         note="fv_claims M1, m1-m5, m10, m12; fv_letter M3, m5; no analysis table changed"),
    dict(id="final_table_notes", date="2026-10-15", status="post_hoc",
         item="tab_cohorts names the TF perturbations with non-robust DE and the criterion (cohort_table "
              "nonrobust_tf_perturbations); tabS_sources prints offline resources without a live link; wrapped "
              "first-column cells of tabS_scaling and tabS_complexity end with a strut",
         note="fv_claims m11; fv_refs 2, 3, 6; no analysis table changed"),
]


def record(path: Path | None = None) -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from r2 import heldout_eval as HE
    HE.record_decisions(MODULE, DECISIONS, path or HE.DECISIONS_TABLE)
    HE.record_decisions(MODULE_BINDING, BINDING_DECISIONS, path or HE.DECISIONS_TABLE)
    HE.record_decisions(MODULE_FIXER, FIXER_DECISIONS, path or HE.DECISIONS_TABLE)
    HE.record_decisions(MODULE_FINAL, FINAL_DECISIONS, path or HE.DECISIONS_TABLE)


def main(argv=None) -> int:
    record()
    print(f"recorded {len(DECISIONS)} {MODULE}, {len(BINDING_DECISIONS)} {MODULE_BINDING} and "
          f"{len(FIXER_DECISIONS)} {MODULE_FIXER} and {len(FINAL_DECISIONS)} {MODULE_FINAL} decisions")
    return 0


if __name__ == "__main__":
    sys.exit(main())
