#!/usr/bin/env python3
"""Evaluate top-tier manuscript readiness gates for Subproject 24."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable, Dict, List

import pandas as pd


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--label",
        type=str,
        default="v1",
        help="Label embedded into the markdown report title.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "subproject_24_multireference_latent_truth_model/implementation/outputs/top_tier_readiness_v1"
        ),
    )

    # Artifact paths (overridable for alternative protocol runs).
    parser.add_argument(
        "--tables-dir",
        type=Path,
        default=Path(
            "subproject_24_multireference_latent_truth_model/implementation/outputs/unified_paper_tables"
        ),
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=Path(
            "subproject_24_multireference_latent_truth_model/implementation/outputs/unified_paper_figures"
        ),
    )
    parser.add_argument(
        "--figure-manifest",
        type=Path,
        default=Path(
            "subproject_24_multireference_latent_truth_model/implementation/outputs/unified_paper_figures/figure_manifest.csv"
        ),
    )
    parser.add_argument(
        "--calibration-summary",
        type=Path,
        default=Path(
            "subproject_24_multireference_latent_truth_model/implementation/outputs/unified_paper_tables/calibration_full_vs_context.csv"
        ),
    )
    parser.add_argument(
        "--validation-summary",
        type=Path,
        default=Path(
            "subproject_24_multireference_latent_truth_model/implementation/outputs/deconcentration_validation_phase11/deconcentration_validation_summary.csv"
        ),
    )
    parser.add_argument(
        "--validation-high-reference-set",
        type=str,
        default="high_overlap_union",
    )
    parser.add_argument(
        "--validation-holdout-reference-set",
        type=str,
        default="low_overlap_union",
    )
    parser.add_argument(
        "--promoted-diversity-summary",
        type=Path,
        default=Path(
            "subproject_24_multireference_latent_truth_model/implementation/outputs/unified_paper_tables/dual_universe_promoted_diversity_summary.csv"
        ),
    )
    parser.add_argument(
        "--tier-consistency-summary",
        type=Path,
        default=Path(
            "subproject_24_multireference_latent_truth_model/implementation/outputs/unified_paper_tables/dual_universe_consistency_summary.csv"
        ),
    )
    parser.add_argument(
        "--tier-summary",
        type=Path,
        default=Path(
            "subproject_24_multireference_latent_truth_model/implementation/outputs/unified_paper_tables/dual_universe_claim_tiers_summary.csv"
        ),
    )
    parser.add_argument(
        "--tier-precision-column",
        type=str,
        default="precision_full_within_tier",
        help="Precision column used in --tier-summary for gate G08.",
    )
    parser.add_argument(
        "--budget-sensitivity-summary",
        type=Path,
        default=Path(
            "subproject_24_multireference_latent_truth_model/implementation/outputs/deconcentration_budget_sensitivity_v1/budget_sensitivity_summary.csv"
        ),
    )
    parser.add_argument(
        "--leaderboard",
        type=Path,
        default=Path(
            "subproject_24_multireference_latent_truth_model/implementation/outputs/variant_method_leaderboard.csv"
        ),
    )
    return parser.parse_args()


def _as_float(value: object) -> float:
    return float(value) if value is not None else float("nan")


def _build_blocked_gate(
    gate_id: str,
    category: str,
    gate_name: str,
    severity: str,
    threshold: str,
    missing_files: List[Path],
    recommendation: str,
) -> Dict[str, object]:
    return {
        "gate_id": gate_id,
        "category": category,
        "gate_name": gate_name,
        "status": "BLOCKED",
        "severity": severity,
        "threshold": threshold,
        "metric_value": "",
        "detail": f"Missing required files: {', '.join(str(p) for p in missing_files)}",
        "recommendation": recommendation,
    }


def _build_gate(
    gate_id: str,
    category: str,
    gate_name: str,
    severity: str,
    threshold: str,
    status: str,
    metric_value: str,
    detail: str,
    recommendation: str,
) -> Dict[str, object]:
    return {
        "gate_id": gate_id,
        "category": category,
        "gate_name": gate_name,
        "status": status,
        "severity": severity,
        "threshold": threshold,
        "metric_value": metric_value,
        "detail": detail,
        "recommendation": recommendation,
    }


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False)


def _check_required_files(paths: List[Path]) -> List[Path]:
    return [path for path in paths if not path.exists()]


def _choose_col(df: pd.DataFrame, candidates: List[str]) -> str:
    for col in candidates:
        if col in df.columns:
            return col
    raise ValueError(f"None of the required columns present: {candidates}")


def _gate_reproducibility(root: Path, args: argparse.Namespace) -> Dict[str, object]:
    required = [
        root / "subproject_24_multireference_latent_truth_model/paper/unified_multireference_latent_truth_full_paper.tex",
        root / "subproject_24_multireference_latent_truth_model/paper/unified_multireference_latent_truth_full_paper.pdf",
        root / "subproject_24_multireference_latent_truth_model/handoff/run_log.md",
        root / args.calibration_summary,
        root / args.figure_manifest,
    ]
    missing = _check_required_files(required)
    if missing:
        return _build_blocked_gate(
            gate_id="G01",
            category="reproducibility",
            gate_name="Submission artifact bundle completeness",
            severity="LOW",
            threshold="All critical files exist, >=15 table CSVs, >=4 figure PNGs",
            missing_files=missing,
            recommendation="Regenerate missing artifacts via table/figure packaging and paper build commands.",
        )

    tables_dir = root / args.tables_dir
    figures_dir = root / args.figures_dir
    n_table_csv = len(list(tables_dir.glob("*.csv")))
    n_figure_png = len(list(figures_dir.glob("*.png")))
    passed = n_table_csv >= 15 and n_figure_png >= 4
    status = "PASS" if passed else "FAIL"
    return _build_gate(
        gate_id="G01",
        category="reproducibility",
        gate_name="Submission artifact bundle completeness",
        severity="LOW",
        threshold="All critical files exist, >=15 table CSVs, >=4 figure PNGs",
        status=status,
        metric_value=f"table_csv={n_table_csv}; figure_png={n_figure_png}",
        detail="Core manuscript, run log, unified tables, and unified figures are present.",
        recommendation="If failed, regenerate packaging outputs and rerun LaTeX assembly.",
    )


def _gate_robustness_leadership(root: Path, _: argparse.Namespace) -> Dict[str, object]:
    src_path = root / (
        "subproject_24_multireference_latent_truth_model/implementation/outputs/"
        "source_dropout_4way_tuned_regime_aggregate_3way.csv"
    )
    fam_path = root / (
        "subproject_24_multireference_latent_truth_model/implementation/outputs/"
        "family_block_dropout_4way_tuned_regime_aggregate_3way.csv"
    )
    missing = _check_required_files([src_path, fam_path])
    if missing:
        return _build_blocked_gate(
            gate_id="G02",
            category="robustness",
            gate_name="Primary model robustness leadership",
            severity="MEDIUM",
            threshold="family_hb_tuned has lowest mean rank disagreement in both stress protocols",
            missing_files=missing,
            recommendation="Re-run source and family-block dropout regime aggregations.",
        )

    src = _read_csv(src_path)
    fam = _read_csv(fam_path)
    src_best = str(src.sort_values("mean_rank_disagreement").iloc[0]["regime"])
    fam_best = str(fam.sort_values("mean_rank_disagreement").iloc[0]["regime"])
    passed = src_best == "family_hb_tuned" and fam_best == "family_hb_tuned"
    status = "PASS" if passed else "FAIL"
    return _build_gate(
        gate_id="G02",
        category="robustness",
        gate_name="Primary model robustness leadership",
        severity="MEDIUM",
        threshold="family_hb_tuned has lowest mean rank disagreement in both stress protocols",
        status=status,
        metric_value=f"source_best={src_best}; family_best={fam_best}",
        detail="Uses full-universe source-dropout and family-block-dropout aggregate tables.",
        recommendation="If failed, retune model priors/objective for robustness-first selection.",
    )


def _gate_external_precision_floor(root: Path, args: argparse.Namespace) -> Dict[str, object]:
    cal_path = root / args.calibration_summary
    missing = _check_required_files([cal_path])
    if missing:
        return _build_blocked_gate(
            gate_id="G03",
            category="biological_validation",
            gate_name="Full-universe external precision floor",
            severity="CRITICAL",
            threshold="full precision@5000 >= 0.01 AND full best_f1 >= 0.02",
            missing_files=missing,
            recommendation="Regenerate calibration tables for context/full matched settings.",
        )

    cal = _read_csv(cal_path)
    full = cal[cal["universe"] == "full_universe"]
    if full.empty:
        return _build_gate(
            gate_id="G03",
            category="biological_validation",
            gate_name="Full-universe external precision floor",
            severity="CRITICAL",
            threshold="full precision@5000 >= 0.01 AND full best_f1 >= 0.02",
            status="BLOCKED",
            metric_value="",
            detail="Row `full_universe` is missing from calibration table.",
            recommendation="Fix table assembly so full-universe calibration row is emitted.",
        )

    full_row = full.iloc[0]
    p5000 = _as_float(full_row["precision_at_5000"])
    best_f1 = _as_float(full_row["best_f1"])
    passed = p5000 >= 0.01 and best_f1 >= 0.02
    status = "PASS" if passed else "FAIL"
    return _build_gate(
        gate_id="G03",
        category="biological_validation",
        gate_name="Full-universe external precision floor",
        severity="CRITICAL",
        threshold="full precision@5000 >= 0.01 AND full best_f1 >= 0.02",
        status=status,
        metric_value=f"precision@5000={p5000:.6f}; best_f1={best_f1:.6f}",
        detail="Evaluates whether global (not context-only) perturbation alignment is submission-grade.",
        recommendation="Expand high-quality external anchors and re-optimize for global calibration, not only context slices.",
    )


def _gate_prospective_significance(root: Path, args: argparse.Namespace) -> Dict[str, object]:
    val_path = root / args.validation_summary
    missing = _check_required_files([val_path])
    if missing:
        return _build_blocked_gate(
            gate_id="G04",
            category="biological_validation",
            gate_name="Prospective external lift significance",
            severity="CRITICAL",
            threshold="best high-overlap enrichment >= 1.5 AND p_value <= 0.05",
            missing_files=missing,
            recommendation="Run prospective validation and retain per-selection statistics.",
        )

    val = _read_csv(val_path)
    high = val[val["reference_set"] == args.validation_high_reference_set].copy()
    if high.empty:
        return _build_gate(
            gate_id="G04",
            category="biological_validation",
            gate_name="Prospective external lift significance",
            severity="CRITICAL",
            threshold="best high-overlap enrichment >= 1.5 AND p_value <= 0.05",
            status="BLOCKED",
            metric_value="",
            detail=f"No `{args.validation_high_reference_set}` rows found in validation summary.",
            recommendation="Ensure high-overlap training reference set is included in validation output.",
        )

    high = high.sort_values(["enrichment_vs_random", "n_hits"], ascending=[False, False]).reset_index(drop=True)
    best = high.iloc[0]
    enrichment = _as_float(best["enrichment_vs_random"])
    p_value = _as_float(best["p_value_hypergeom_ge_hits"])
    selection = str(best["selection"])
    passed = enrichment >= 1.5 and p_value <= 0.05
    status = "PASS" if passed else "FAIL"
    return _build_gate(
        gate_id="G04",
        category="biological_validation",
        gate_name="Prospective external lift significance",
        severity="CRITICAL",
        threshold="best high-overlap enrichment >= 1.5 AND p_value <= 0.05",
        status=status,
        metric_value=f"selection={selection}; enrichment={enrichment:.4f}; p={p_value:.4f}",
        detail="Uses the configured high-overlap reference set for prospective significance.",
        recommendation="Increase candidate-overlap with independent perturbation references and rerun prospective tests.",
    )


def _gate_low_overlap_holdout(root: Path, args: argparse.Namespace) -> Dict[str, object]:
    val_path = root / args.validation_summary
    missing = _check_required_files([val_path])
    if missing:
        return _build_blocked_gate(
            gate_id="G05",
            category="biological_validation",
            gate_name="Low-overlap holdout operational coverage",
            severity="CRITICAL",
            threshold="n_reference_edges_in_candidate_pool >= 10 AND n_hits >= 1",
            missing_files=missing,
            recommendation="Run validation summary with blind holdout reference partition included.",
        )

    val = _read_csv(val_path)
    low = val[val["reference_set"] == args.validation_holdout_reference_set].copy()
    if low.empty:
        return _build_gate(
            gate_id="G05",
            category="biological_validation",
            gate_name="Low-overlap holdout operational coverage",
            severity="CRITICAL",
            threshold="n_reference_edges_in_candidate_pool >= 10 AND n_hits >= 1",
            status="BLOCKED",
            metric_value="",
            detail=f"No `{args.validation_holdout_reference_set}` rows found in validation summary.",
            recommendation="Add explicit blind holdout split and evaluate candidate coverage.",
        )

    cov_col = _choose_col(
        low,
        ["n_reference_edges_in_candidate_pool", "n_reference_edges_in_promoted_candidates"],
    )
    max_cov = int(low[cov_col].max())
    max_hits = int(low["n_hits"].max())
    passed = max_cov >= 10 and max_hits >= 1
    status = "PASS" if passed else "FAIL"
    return _build_gate(
        gate_id="G05",
        category="biological_validation",
        gate_name="Low-overlap holdout operational coverage",
        severity="CRITICAL",
        threshold="n_reference_edges_in_candidate_pool >= 10 AND n_hits >= 1",
        status=status,
        metric_value=f"max_holdout_coverage={max_cov}; max_hits={max_hits}",
        detail="Checks whether holdout references can discriminate candidate policies.",
        recommendation="Rebuild candidate pool to include blind-holdout references while preserving training precision.",
    )


def _gate_promotion_concentration(root: Path, args: argparse.Namespace) -> Dict[str, object]:
    div_path = root / args.promoted_diversity_summary
    missing = _check_required_files([div_path])
    if missing:
        return _build_blocked_gate(
            gate_id="G06",
            category="governance",
            gate_name="Promotion concentration ceiling",
            severity="HIGH",
            threshold="top1_source_fraction_promoted <= 0.25 AND top2_source_fraction_promoted <= 0.50",
            missing_files=missing,
            recommendation="Regenerate promoted diversity summary from governance outputs.",
        )

    div = _read_csv(div_path)
    value_map = {str(row["metric"]): _as_float(row["value"]) for _, row in div.iterrows()}
    top1 = value_map.get("top1_source_fraction_promoted", float("nan"))
    top2 = value_map.get("top2_source_fraction_promoted", float("nan"))
    passed = top1 <= 0.25 and top2 <= 0.50
    status = "PASS" if passed else "FAIL"
    return _build_gate(
        gate_id="G06",
        category="governance",
        gate_name="Promotion concentration ceiling",
        severity="HIGH",
        threshold="top1_source_fraction_promoted <= 0.25 AND top2_source_fraction_promoted <= 0.50",
        status=status,
        metric_value=f"top1={top1:.4f}; top2={top2:.4f}",
        detail="Concentration gate on promoted set.",
        recommendation="Apply constrained promotion optimization with source-balance penalties.",
    )


def _gate_tier_agreement(root: Path, args: argparse.Namespace) -> Dict[str, object]:
    cons_path = root / args.tier_consistency_summary
    missing = _check_required_files([cons_path])
    if missing:
        return _build_blocked_gate(
            gate_id="G07",
            category="governance",
            gate_name="Cross-universe tier agreement floor",
            severity="HIGH",
            threshold="exact_match >= 0.10 AND within_one_tier >= 0.75",
            missing_files=missing,
            recommendation="Regenerate dual-universe consistency summary and inspect tier mappings.",
        )

    cons = _read_csv(cons_path).iloc[0]
    exact = _as_float(cons["fraction_exact_tier_match"])
    within_one = _as_float(cons["fraction_match_within_one_tier"])
    passed = exact >= 0.10 and within_one >= 0.75
    status = "PASS" if passed else "FAIL"
    return _build_gate(
        gate_id="G07",
        category="governance",
        gate_name="Cross-universe tier agreement floor",
        severity="HIGH",
        threshold="exact_match >= 0.10 AND within_one_tier >= 0.75",
        status=status,
        metric_value=f"exact={exact:.4f}; within_one={within_one:.4f}",
        detail="Measures tier stability between universes.",
        recommendation="Refine tier mapping with uncertainty-aware calibration and context-coverage correction.",
    )


def _gate_promoted_precision(root: Path, args: argparse.Namespace) -> Dict[str, object]:
    tiers_path = root / args.tier_summary
    missing = _check_required_files([tiers_path])
    if missing:
        return _build_blocked_gate(
            gate_id="G08",
            category="governance",
            gate_name="Promoted-tier precision dominance",
            severity="CRITICAL",
            threshold="weighted promoted precision >= conditional precision",
            missing_files=missing,
            recommendation="Regenerate dual-universe tier precision summaries.",
        )

    tiers = _read_csv(tiers_path)
    if args.tier_precision_column not in tiers.columns:
        return _build_gate(
            gate_id="G08",
            category="governance",
            gate_name="Promoted-tier precision dominance",
            severity="CRITICAL",
            threshold="weighted promoted precision >= conditional precision",
            status="BLOCKED",
            metric_value="",
            detail=f"Tier precision column `{args.tier_precision_column}` missing.",
            recommendation="Set --tier-precision-column to an existing precision column in tier summary.",
        )

    promoted = tiers[
        tiers["dual_universe_claim_tier"].isin(["DUAL_TIER1_STRONG", "DUAL_TIER2_MODERATE"])
    ].copy()
    conditional = tiers[tiers["dual_universe_claim_tier"] == "DUAL_TIER3_CONDITIONAL"].copy()
    if promoted.empty or conditional.empty:
        return _build_gate(
            gate_id="G08",
            category="governance",
            gate_name="Promoted-tier precision dominance",
            severity="CRITICAL",
            threshold="weighted promoted precision >= conditional precision",
            status="BLOCKED",
            metric_value="",
            detail="Required promoted or conditional tier rows are missing.",
            recommendation="Ensure all four dual tiers are present in summary export.",
        )

    promoted_weight = promoted["n_edges"].astype(float).sum()
    promoted_precision = (
        (promoted["n_edges"].astype(float) * promoted[args.tier_precision_column].astype(float)).sum()
        / max(promoted_weight, 1.0)
    )
    conditional_precision = _as_float(conditional.iloc[0][args.tier_precision_column])
    passed = promoted_precision >= conditional_precision
    status = "PASS" if passed else "FAIL"
    return _build_gate(
        gate_id="G08",
        category="governance",
        gate_name="Promoted-tier precision dominance",
        severity="CRITICAL",
        threshold="weighted promoted precision >= conditional precision",
        status=status,
        metric_value=f"promoted_precision={promoted_precision:.6f}; conditional_precision={conditional_precision:.6f}",
        detail="Checks whether promoted tiers carry stronger precision than conditional tier.",
        recommendation="Retune tier boundaries to enforce promoted-over-conditional precision dominance.",
    )


def _gate_baseline_breadth(root: Path, args: argparse.Namespace) -> Dict[str, object]:
    leaderboard_path = root / args.leaderboard
    missing = _check_required_files([leaderboard_path])
    if missing:
        return _build_blocked_gate(
            gate_id="G09",
            category="benchmarking",
            gate_name="Baseline breadth and controls",
            severity="MEDIUM",
            threshold=">=10 methods and includes baseline_random_3950 + baseline_trrust",
            missing_files=missing,
            recommendation="Regenerate variant leaderboard from multi-model evaluation.",
        )

    leaderboard = _read_csv(leaderboard_path)
    methods = set(leaderboard["method"].astype(str).tolist())
    n_methods = len(methods)
    has_controls = {"baseline_random_3950", "baseline_trrust"}.issubset(methods)
    passed = n_methods >= 10 and has_controls
    status = "PASS" if passed else "FAIL"
    return _build_gate(
        gate_id="G09",
        category="benchmarking",
        gate_name="Baseline breadth and controls",
        severity="MEDIUM",
        threshold=">=10 methods and includes baseline_random_3950 + baseline_trrust",
        status=status,
        metric_value=f"n_methods={n_methods}; has_controls={has_controls}",
        detail="Ensures evaluation includes enough competing methods plus sanity controls.",
        recommendation="Add additional external baselines if this gate fails.",
    )


def _gate_budget_policy_stability(root: Path, args: argparse.Namespace) -> Dict[str, object]:
    budget_path = root / args.budget_sensitivity_summary
    missing = _check_required_files([budget_path])
    if missing:
        return _build_blocked_gate(
            gate_id="G10",
            category="policy",
            gate_name="Budget-conditioned policy stability",
            severity="HIGH",
            threshold=(
                "For non-baseline recommendations: min delta_hits_high_overlap_union >= -1 "
                "AND min delta_joint_score_mean >= -0.003"
            ),
            missing_files=missing,
            recommendation="Run budget/policy sweep to map policy utility across regimes.",
        )

    budget = _read_csv(budget_path)
    if "recommended_policy" not in budget.columns:
        return _build_gate(
            gate_id="G10",
            category="policy",
            gate_name="Budget-conditioned policy stability",
            severity="HIGH",
            threshold=(
                "For non-baseline recommendations: min delta_hits_high_overlap_union >= -1 "
                "AND min delta_joint_score_mean >= -0.003"
            ),
            status="BLOCKED",
            metric_value="",
            detail="Budget summary missing `recommended_policy` column.",
            recommendation="Provide a budget summary table with recommended policy labels and delta metrics.",
        )

    active = budget[budget["recommended_policy"] != "baseline_alpha0"].copy()
    if active.empty:
        return _build_gate(
            gate_id="G10",
            category="policy",
            gate_name="Budget-conditioned policy stability",
            severity="HIGH",
            threshold=(
                "For non-baseline recommendations: min delta_hits_high_overlap_union >= -1 "
                "AND min delta_joint_score_mean >= -0.003"
            ),
            status="BLOCKED",
            metric_value="",
            detail="No budgets selected a non-baseline policy; cannot assess deconcentration stability.",
            recommendation="Expand budget/alpha grid until deconcentration and baseline regimes can be compared.",
        )

    hits_col = _choose_col(active, ["delta_hits_high_overlap_union", "delta_train_hits"])
    min_delta_hits = _as_float(active[hits_col].min())
    min_delta_score = _as_float(active["delta_joint_score_mean"].min())
    passed = min_delta_hits >= -1.0 and min_delta_score >= -0.003
    status = "PASS" if passed else "FAIL"
    return _build_gate(
        gate_id="G10",
        category="policy",
        gate_name="Budget-conditioned policy stability",
        severity="HIGH",
        threshold=(
            "For non-baseline recommendations: min delta_hits_high_overlap_union >= -1 "
            "AND min delta_joint_score_mean >= -0.003"
        ),
        status=status,
        metric_value=f"min_delta_hits={min_delta_hits:.3f}; min_delta_score={min_delta_score:.6f}",
        detail="Checks whether deconcentration remains non-destructive in active recommendation regimes.",
        recommendation="Constrain policy optimizer with explicit hit-retention guardrails.",
    )


def _write_markdown_report(
    gate_results: pd.DataFrame,
    summary: pd.DataFrame,
    out_path: Path,
    label: str,
) -> None:
    overall = summary.iloc[0]
    lines: List[str] = []
    lines.append(f"# Top-Tier Readiness Execution Report ({label})")
    lines.append("")
    lines.append("## Overall status")
    lines.append("")
    lines.append(f"- `top_tier_ready`: `{bool(overall['top_tier_ready'])}`")
    lines.append(
        f"- `readiness_score`: `{overall['readiness_score']:.3f}` "
        f"(`{int(overall['n_pass'])}` pass / `{int(overall['n_total'])}` total gates)"
    )
    lines.append(f"- `critical_fail_count`: `{int(overall['n_critical_fail'])}`")
    lines.append(f"- `blocked_count`: `{int(overall['n_blocked'])}`")
    lines.append("")
    lines.append("## Gate outcomes")
    lines.append("")
    lines.append("| Gate | Status | Severity | Metric | Threshold |")
    lines.append("|---|---|---|---|---|")
    for _, row in gate_results.iterrows():
        lines.append(
            f"| {row['gate_id']} {row['gate_name']} | {row['status']} | {row['severity']} | "
            f"{row['metric_value']} | {row['threshold']} |"
        )
    lines.append("")

    failed = gate_results[gate_results["status"] == "FAIL"].copy()
    blocked = gate_results[gate_results["status"] == "BLOCKED"].copy()
    lines.append("## Priority actions from failed gates")
    lines.append("")
    if failed.empty and blocked.empty:
        lines.append("- All gates passed; no immediate corrective action required.")
    else:
        for _, row in pd.concat([failed, blocked], ignore_index=True).iterrows():
            lines.append(f"- `{row['gate_id']}` ({row['severity']}): {row['recommendation']}")
    lines.append("")
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = _parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    root = Path(".").resolve()

    checks: List[Callable[[Path, argparse.Namespace], Dict[str, object]]] = [
        _gate_reproducibility,
        _gate_robustness_leadership,
        _gate_external_precision_floor,
        _gate_prospective_significance,
        _gate_low_overlap_holdout,
        _gate_promotion_concentration,
        _gate_tier_agreement,
        _gate_promoted_precision,
        _gate_baseline_breadth,
        _gate_budget_policy_stability,
    ]
    rows = [check(root, args) for check in checks]
    gate_results = pd.DataFrame(rows)
    gate_results.to_csv(args.output_dir / "gate_results.csv", index=False)

    n_total = int(gate_results.shape[0])
    n_pass = int((gate_results["status"] == "PASS").sum())
    n_fail = int((gate_results["status"] == "FAIL").sum())
    n_blocked = int((gate_results["status"] == "BLOCKED").sum())
    critical_fail = int(
        ((gate_results["status"] == "FAIL") & (gate_results["severity"] == "CRITICAL")).sum()
    )
    readiness_score = float(n_pass / max(n_total, 1))
    top_tier_ready = bool(readiness_score >= 0.8 and critical_fail == 0 and n_blocked == 0)

    summary = pd.DataFrame(
        [
            {
                "n_total": n_total,
                "n_pass": n_pass,
                "n_fail": n_fail,
                "n_blocked": n_blocked,
                "n_critical_fail": critical_fail,
                "readiness_score": readiness_score,
                "top_tier_ready": top_tier_ready,
            }
        ]
    )
    summary.to_csv(args.output_dir / "readiness_summary.csv", index=False)

    _write_markdown_report(
        gate_results=gate_results,
        summary=summary,
        out_path=args.output_dir / "readiness_report.md",
        label=args.label,
    )

    print("Top-tier readiness evaluation complete.")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
