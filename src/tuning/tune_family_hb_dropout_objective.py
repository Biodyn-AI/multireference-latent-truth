#!/usr/bin/env python3
"""Tune hierarchical family model hyperparameters using dropout robustness objective."""

from __future__ import annotations

import argparse
import itertools
import subprocess
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument(
        "--source-strength-alpha-grid",
        type=str,
        default="8,12",
    )
    parser.add_argument(
        "--source-strength-beta-grid",
        type=str,
        default="16,24",
    )
    parser.add_argument(
        "--family-strength-grid",
        type=str,
        default="3,5",
    )
    parser.add_argument(
        "--global-prior-strength-grid",
        type=str,
        default="1,2",
    )
    parser.add_argument("--min-youden-clean", type=float, default=0.01)
    parser.add_argument("--max-edges", type=int, default=100000)
    parser.add_argument("--max-iter", type=int, default=80)
    parser.add_argument(
        "--f1-target",
        type=float,
        default=0.38,
        help="Penalty target for baseline top weighted-F1.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("subproject_24_multireference_latent_truth_model/implementation/outputs/family_hb_tuning"),
    )
    return parser.parse_args()


def _parse_float_grid(s: str) -> List[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def _run(cmd: List[str], cwd: Path) -> None:
    subprocess.run(cmd, cwd=str(cwd), check=True)


def main() -> None:
    args = _parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    a_grid = _parse_float_grid(args.source_strength_alpha_grid)
    b_grid = _parse_float_grid(args.source_strength_beta_grid)
    f_grid = _parse_float_grid(args.family_strength_grid)
    g_grid = _parse_float_grid(args.global_prior_strength_grid)

    impl = "subproject_24_multireference_latent_truth_model/implementation"
    results = []

    for i, (sa, sb, fs, gs) in enumerate(itertools.product(a_grid, b_grid, f_grid, g_grid), start=1):
        run_id = f"sa{sa:g}_sb{sb:g}_fs{fs:g}_gs{gs:g}"
        ablation_dir = args.output_dir / run_id
        fit_extra = (
            f"--source-strength-alpha {sa} "
            f"--source-strength-beta {sb} "
            f"--family-strength {fs} "
            f"--global-prior-strength {gs} "
            f"--min-youden-clean {args.min_youden_clean}"
        )

        cmd = [
            "python",
            f"{impl}/run_source_dropout_ablation.py",
            "--fit-script",
            "fit_family_hierarchical_bayes_ltm.py",
            "--posterior-file",
            "ltm_family_hb_edge_posteriors.csv",
            "--fit-summary-file",
            "ltm_family_hb_fit_summary.csv",
            "--max-edges",
            str(args.max_edges),
            "--max-iter",
            str(args.max_iter),
            "--fit-extra-args",
            fit_extra,
            "--ablation-dir",
            str(ablation_dir),
        ]
        print(f"[{i}] Running {run_id}")
        _run(cmd, cwd=args.repo_root)

        summary_path = ablation_dir / "source_dropout_summary.csv"
        df = pd.read_csv(summary_path)
        base = df[df["dropped_source"] == "NONE_BASELINE"].iloc[0]
        nonbase = df[df["dropped_source"] != "NONE_BASELINE"].copy()

        mean_dis = float(nonbase["pairwise_rank_disagreement_vs_baseline"].mean())
        max_dis = float(nonbase["pairwise_rank_disagreement_vs_baseline"].max())
        mean_abs_dpi = float(np.mean(np.abs(nonbase["delta_pi_vs_baseline"])))
        max_abs_dpi = float(np.max(np.abs(nonbase["delta_pi_vs_baseline"])))
        mean_abs_drho = float(np.mean(np.abs(nonbase["delta_rho_vs_baseline"])))
        top_f1 = float(base["top_method_weighted_f1"])
        f1_penalty = max(0.0, args.f1_target - top_f1)

        objective = (
            mean_dis
            + 0.5 * max_dis
            + 0.5 * mean_abs_dpi
            + 0.2 * mean_abs_drho
            + 0.5 * f1_penalty
        )

        results.append(
            {
                "run_id": run_id,
                "source_strength_alpha": sa,
                "source_strength_beta": sb,
                "family_strength": fs,
                "global_prior_strength": gs,
                "baseline_pi": float(base["pi_prevalence"]),
                "baseline_rho": float(base["rho_burst"]),
                "baseline_top_method": str(base["top_method"]),
                "baseline_top_weighted_f1": top_f1,
                "mean_rank_disagreement": mean_dis,
                "max_rank_disagreement": max_dis,
                "mean_abs_delta_pi": mean_abs_dpi,
                "max_abs_delta_pi": max_abs_dpi,
                "mean_abs_delta_rho": mean_abs_drho,
                "f1_penalty": f1_penalty,
                "objective_score": objective,
                "ablation_dir": str(ablation_dir),
            }
        )

    out = pd.DataFrame(results).sort_values(
        ["objective_score", "mean_rank_disagreement", "mean_abs_delta_pi"],
        ascending=[True, True, True],
    ).reset_index(drop=True)

    out.to_csv(args.output_dir / "family_hb_tuning_summary.csv", index=False)
    best = out.iloc[0]
    pd.DataFrame([best]).to_csv(args.output_dir / "family_hb_tuning_best.csv", index=False)

    print("Family-HB tuning complete:")
    print(out.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
