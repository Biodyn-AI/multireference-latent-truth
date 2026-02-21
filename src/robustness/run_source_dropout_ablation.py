#!/usr/bin/env python3
"""Run source-dropout ablations for dependence-aware latent truth modeling.

This script quantifies how removing each annotator source changes:
- prevalence and burst estimates,
- posterior-aware method rankings,
- pairwise ranking disagreement vs baseline.
"""

from __future__ import annotations

import argparse
import math
import shlex
import subprocess
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path("."),
    )
    parser.add_argument(
        "--matrix",
        type=Path,
        default=Path("subproject_24_multireference_latent_truth_model/implementation/outputs/evidence_matrix_wide.csv"),
    )
    parser.add_argument(
        "--source-meta",
        type=Path,
        default=Path("subproject_24_multireference_latent_truth_model/implementation/outputs/source_metadata.csv"),
    )
    parser.add_argument("--include-kinds", type=str, default="reference,mechanistic")
    parser.add_argument("--max-edges", type=int, default=120000)
    parser.add_argument("--max-iter", type=int, default=180)
    parser.add_argument(
        "--fit-script",
        type=str,
        default="fit_dependent_ltm.py",
        help="Model-fit script under implementation/, e.g. fit_dependent_ltm.py.",
    )
    parser.add_argument(
        "--posterior-file",
        type=str,
        default="ltm_dep_edge_posteriors.csv",
        help="Posterior filename produced by --fit-script.",
    )
    parser.add_argument(
        "--fit-summary-file",
        type=str,
        default="ltm_dep_fit_summary.csv",
        help="Fit summary filename produced by --fit-script.",
    )
    parser.add_argument(
        "--fit-extra-args",
        type=str,
        default="",
        help="Extra args appended to fit-script invocation.",
    )
    parser.add_argument(
        "--ci-alpha",
        type=float,
        default=0.05,
        help="Two-sided alpha for disagreement confidence intervals (currently supports 0.05).",
    )
    parser.add_argument(
        "--ablation-dir",
        type=Path,
        default=Path("subproject_24_multireference_latent_truth_model/implementation/outputs/source_dropout_ablation"),
    )
    return parser.parse_args()


def _parse_csv_list(s: str) -> List[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def _run(cmd: List[str], cwd: Path) -> None:
    subprocess.run(cmd, cwd=str(cwd), check=True)


def _pairwise_orders(metric_df: pd.DataFrame, metric_col: str) -> Dict[Tuple[str, str], int]:
    if metric_df.empty:
        return {}
    scores = metric_df.set_index("method")[metric_col].to_dict()
    methods = sorted(scores.keys())
    out: Dict[Tuple[str, str], int] = {}
    for a, b in combinations(methods, 2):
        va, vb = scores[a], scores[b]
        if np.isclose(va, vb):
            out[(a, b)] = 0
        elif va > vb:
            out[(a, b)] = 1
        else:
            out[(a, b)] = -1
    return out


def _disagreement_stats(a: Dict[Tuple[str, str], int], b: Dict[Tuple[str, str], int]) -> Tuple[float, int, int]:
    pairs = sorted(set(a.keys()) & set(b.keys()))
    valid = 0
    diff = 0
    for p in pairs:
        if a[p] == 0 or b[p] == 0:
            continue
        valid += 1
        if a[p] != b[p]:
            diff += 1
    if valid == 0:
        return float("nan"), 0, 0
    return diff / float(valid), diff, valid


def _wilson_interval(k: int, n: int, alpha: float) -> Tuple[float, float]:
    """Wilson interval for binomial proportion.

    For now we support alpha=0.05 only (z=1.96), which is sufficient for this workflow.
    """

    if n <= 0:
        return float("nan"), float("nan")
    if alpha != 0.05:
        raise ValueError("Current implementation supports --ci-alpha 0.05 only.")

    z = 1.959963984540054
    phat = k / float(n)
    denom = 1.0 + (z * z) / n
    center = (phat + (z * z) / (2.0 * n)) / denom
    half = (
        z
        * math.sqrt((phat * (1.0 - phat) / n) + ((z * z) / (4.0 * n * n)))
        / denom
    )
    return max(0.0, center - half), min(1.0, center + half)


def _safe_name(source_id: str) -> str:
    return source_id.replace("/", "_").replace(" ", "_")


def _extract_burst_terms(fit_row: pd.Series) -> Dict[str, float]:
    """Return comparable burst terms across one-factor and multi-factor summaries."""

    rho = fit_row.get("rho_burst", np.nan)
    rho1 = fit_row.get("rho_burst1", np.nan)
    rho2 = fit_row.get("rho_burst2", np.nan)

    has_rho = not pd.isna(rho)
    has_rho1 = not pd.isna(rho1)
    has_rho2 = not pd.isna(rho2)

    if has_rho:
        total = float(rho)
    elif has_rho1 and has_rho2:
        total = float(rho1 + rho2)
    else:
        total = float("nan")

    return {
        "rho_burst": float(total),  # Backward-compatible alias.
        "rho_burst_total": float(total),
        "rho_burst1": float(rho1) if has_rho1 else float("nan"),
        "rho_burst2": float(rho2) if has_rho2 else float("nan"),
    }


def _delta_or_nan(value: float, baseline: float) -> float:
    if pd.isna(value) or pd.isna(baseline):
        return float("nan")
    return float(value - baseline)


def main() -> None:
    args = _parse_args()
    args.ablation_dir.mkdir(parents=True, exist_ok=True)

    source_meta = pd.read_csv(args.source_meta)
    include_kinds = _parse_csv_list(args.include_kinds)
    annotators = (
        source_meta[source_meta["kind"].isin(include_kinds)]["source_id"].sort_values().tolist()
    )
    if not annotators:
        raise ValueError("No annotators selected for ablation.")

    impl = "subproject_24_multireference_latent_truth_model/implementation"
    fit_script_path = f"{impl}/{args.fit_script}"
    fit_extra_args = shlex.split(args.fit_extra_args) if args.fit_extra_args else []

    # Baseline run on the same subset for fair comparison.
    baseline_dir = args.ablation_dir / "baseline"
    baseline_dir.mkdir(parents=True, exist_ok=True)

    _run(
        [
            "python",
            fit_script_path,
            "--matrix",
            str(args.matrix),
            "--source-meta",
            str(args.source_meta),
            "--include-kinds",
            args.include_kinds,
            "--max-edges",
            str(args.max_edges),
            "--max-iter",
            str(args.max_iter),
            *fit_extra_args,
            "--output-dir",
            str(baseline_dir),
        ],
        cwd=args.repo_root,
    )

    _run(
        [
            "python",
            f"{impl}/evaluate_posterior_rankings.py",
            "--matrix",
            str(args.matrix),
            "--source-meta",
            str(args.source_meta),
            "--posterior",
            str(baseline_dir / args.posterior_file),
            "--posterior-col",
            "posterior",
            "--output-dir",
            str(baseline_dir),
        ],
        cwd=args.repo_root,
    )

    baseline_fit = pd.read_csv(baseline_dir / args.fit_summary_file).iloc[0]
    baseline_metrics = pd.read_csv(baseline_dir / "posterior_method_metrics.csv")
    baseline_orders = _pairwise_orders(baseline_metrics[["method", "weighted_f1"]], "weighted_f1")
    baseline_burst = _extract_burst_terms(baseline_fit)

    rows = []

    for source in annotators:
        run_dir = args.ablation_dir / f"drop_{_safe_name(source)}"
        run_dir.mkdir(parents=True, exist_ok=True)

        _run(
            [
                "python",
                fit_script_path,
                "--matrix",
                str(args.matrix),
                "--source-meta",
                str(args.source_meta),
                "--include-kinds",
                args.include_kinds,
                "--exclude-sources",
                source,
                "--max-edges",
                str(args.max_edges),
                "--max-iter",
                str(args.max_iter),
                *fit_extra_args,
                "--output-dir",
                str(run_dir),
            ],
            cwd=args.repo_root,
        )

        _run(
            [
                "python",
                f"{impl}/evaluate_posterior_rankings.py",
                "--matrix",
                str(args.matrix),
                "--source-meta",
                str(args.source_meta),
                "--posterior",
                str(run_dir / args.posterior_file),
                "--posterior-col",
                "posterior",
                "--output-dir",
                str(run_dir),
            ],
            cwd=args.repo_root,
        )

        fit = pd.read_csv(run_dir / args.fit_summary_file).iloc[0]
        metrics = pd.read_csv(run_dir / "posterior_method_metrics.csv")
        orders = _pairwise_orders(metrics[["method", "weighted_f1"]], "weighted_f1")
        disagreement, n_diff, n_valid = _disagreement_stats(baseline_orders, orders)
        ci_lo, ci_hi = _wilson_interval(n_diff, n_valid, alpha=args.ci_alpha)
        burst = _extract_burst_terms(fit)

        top_row = metrics.sort_values("weighted_f1", ascending=False).iloc[0]
        rows.append(
            {
                "dropped_source": source,
                "n_edges": int(fit["n_edges"]),
                "pi_prevalence": float(fit["pi_prevalence"]),
                "rho_burst": burst["rho_burst"],
                "rho_burst_total": burst["rho_burst_total"],
                "rho_burst1": burst["rho_burst1"],
                "rho_burst2": burst["rho_burst2"],
                "delta_pi_vs_baseline": float(fit["pi_prevalence"] - baseline_fit["pi_prevalence"]),
                "delta_rho_vs_baseline": _delta_or_nan(
                    burst["rho_burst"],
                    baseline_burst["rho_burst"],
                ),
                "delta_rho_total_vs_baseline": _delta_or_nan(
                    burst["rho_burst_total"],
                    baseline_burst["rho_burst_total"],
                ),
                "delta_rho1_vs_baseline": _delta_or_nan(
                    burst["rho_burst1"],
                    baseline_burst["rho_burst1"],
                ),
                "delta_rho2_vs_baseline": _delta_or_nan(
                    burst["rho_burst2"],
                    baseline_burst["rho_burst2"],
                ),
                "top_method": str(top_row["method"]),
                "top_method_weighted_f1": float(top_row["weighted_f1"]),
                "pairwise_rank_disagreement_vs_baseline": disagreement,
                "n_disagree_pairs": int(n_diff),
                "n_valid_pairs": int(n_valid),
                "pairwise_rank_disagreement_ci_low": ci_lo,
                "pairwise_rank_disagreement_ci_high": ci_hi,
            }
        )

    summary = pd.DataFrame(rows).sort_values(
        "pairwise_rank_disagreement_vs_baseline", ascending=False
    ).reset_index(drop=True)

    baseline_row = pd.DataFrame(
        [
            {
                "dropped_source": "NONE_BASELINE",
                "n_edges": int(baseline_fit["n_edges"]),
                "pi_prevalence": float(baseline_fit["pi_prevalence"]),
                "rho_burst": baseline_burst["rho_burst"],
                "rho_burst_total": baseline_burst["rho_burst_total"],
                "rho_burst1": baseline_burst["rho_burst1"],
                "rho_burst2": baseline_burst["rho_burst2"],
                "delta_pi_vs_baseline": 0.0,
                "delta_rho_vs_baseline": 0.0,
                "delta_rho_total_vs_baseline": 0.0,
                "delta_rho1_vs_baseline": 0.0,
                "delta_rho2_vs_baseline": 0.0,
                "top_method": str(
                    baseline_metrics.sort_values("weighted_f1", ascending=False).iloc[0]["method"]
                ),
                "top_method_weighted_f1": float(
                    baseline_metrics.sort_values("weighted_f1", ascending=False).iloc[0]["weighted_f1"]
                ),
                "pairwise_rank_disagreement_vs_baseline": 0.0,
                "n_disagree_pairs": 0,
                "n_valid_pairs": int(
                    _disagreement_stats(baseline_orders, baseline_orders)[2]
                ),
                "pairwise_rank_disagreement_ci_low": 0.0,
                "pairwise_rank_disagreement_ci_high": 0.0,
            }
        ]
    )

    full = pd.concat([baseline_row, summary], ignore_index=True)
    full.to_csv(args.ablation_dir / "source_dropout_summary.csv", index=False)

    print("Source-dropout ablation complete:")
    print(full.to_string(index=False))


if __name__ == "__main__":
    main()
