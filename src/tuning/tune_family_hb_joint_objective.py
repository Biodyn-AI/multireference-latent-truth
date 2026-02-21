#!/usr/bin/env python3
"""Tune Family-HB hyperparameters with joint robustness + perturbation objective.

This tuner extends the Phase V pilot by:
1. multi-seed execution (seeded edge subsampling),
2. multi-threshold perturbation utility,
3. uncertainty summaries over seeds.
"""

from __future__ import annotations

import argparse
import itertools
import math
import subprocess
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd


EdgeSet = set[Tuple[str, str]]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
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
    parser.add_argument(
        "--seeds",
        type=str,
        default="1,2,3",
        help="Comma-separated seeds for edge subsampling uncertainty runs.",
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
        "--panel-glob",
        type=str,
        default="single_cell_mechinterp/outputs/perturb_validation/*/perturbation_edges.tsv",
    )
    parser.add_argument(
        "--min-panel-overlap",
        type=int,
        default=50,
        help="Minimum panel overlap with candidate universe for high-overlap union.",
    )
    parser.add_argument(
        "--perturbation-k-list",
        type=str,
        default="500,5000,20000",
        help="Comma-separated top-k cutoffs for perturbation utility.",
    )
    parser.add_argument(
        "--perturbation-k-weights",
        type=str,
        default="0.2,0.5,0.3",
        help="Comma-separated utility weights aligned with --perturbation-k-list.",
    )
    parser.add_argument(
        "--perturbation-weight",
        type=float,
        default=20.0,
        help="Global scale for perturbation utility in the objective.",
    )
    parser.add_argument(
        "--background-penalty-weight",
        type=float,
        default=0.01,
        help="Penalty weight on (1 - recall@max_k) to discourage background concentration.",
    )
    parser.add_argument(
        "--family-block-weight",
        type=float,
        default=1.0,
        help="Scaling factor for family-block robustness terms.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("subproject_24_multireference_latent_truth_model/implementation/outputs/family_hb_joint_tuning"),
    )
    return parser.parse_args()


def _parse_float_grid(s: str) -> List[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def _parse_int_list(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def _parse_seed_list(s: str) -> List[int]:
    seeds = _parse_int_list(s)
    if not seeds:
        raise ValueError("At least one seed is required.")
    return seeds


def _parse_weight_list(s: str) -> List[float]:
    weights = _parse_float_grid(s)
    if not weights:
        raise ValueError("At least one perturbation utility weight is required.")
    if any(w < 0.0 for w in weights):
        raise ValueError("Perturbation utility weights must be non-negative.")
    total = sum(weights)
    if total <= 0.0:
        raise ValueError("Perturbation utility weights must sum to a positive value.")
    return [w / total for w in weights]


def _run(cmd: List[str], cwd: Path) -> None:
    subprocess.run(cmd, cwd=str(cwd), check=True)


def _edge_set(df: pd.DataFrame) -> EdgeSet:
    return set(df[["source", "target"]].drop_duplicates().itertuples(index=False, name=None))


def _load_panel_edges(repo_root: Path, panel_glob: str) -> Dict[str, EdgeSet]:
    panel_edges: Dict[str, EdgeSet] = {}
    for path in sorted((repo_root / ".").glob(panel_glob)):
        panel = pd.read_csv(path, sep="\t")
        required = {"source", "target"}
        if not required.issubset(panel.columns):
            raise ValueError(f"{path}: missing columns {sorted(required)}")
        panel_edges[path.parent.name] = _edge_set(panel)
    if not panel_edges:
        raise ValueError(f"No panel files found for glob: {panel_glob}")
    return panel_edges


def _build_high_overlap_union(
    panel_edges: Dict[str, EdgeSet],
    universe: EdgeSet,
    min_panel_overlap: int,
) -> Tuple[EdgeSet, List[str]]:
    overlaps = {label: len(edges & universe) for label, edges in panel_edges.items()}
    high_labels = [label for label, ov in overlaps.items() if ov >= min_panel_overlap]
    if not high_labels:
        high_labels = list(panel_edges.keys())
    union_edges: EdgeSet = set()
    for label in high_labels:
        union_edges |= panel_edges[label]
    return union_edges, high_labels


def _precision_recall_at_k(y: np.ndarray, score: np.ndarray, k: int) -> Tuple[float, float, int]:
    if k <= 0:
        return float("nan"), float("nan"), 0
    k = min(k, len(y))
    order = np.argsort(-score, kind="mergesort")
    top = order[:k]
    tp = int(np.sum(y[top] == 1))
    precision = tp / float(k) if k > 0 else float("nan")
    recall = tp / float(np.sum(y == 1)) if np.sum(y == 1) > 0 else float("nan")
    return float(precision), float(recall), tp


def _summarize_source_dropout(path: Path, f1_target: float) -> Dict[str, float | str]:
    df = pd.read_csv(path)
    baseline = df[df["dropped_source"] == "NONE_BASELINE"].iloc[0]
    nonbase = df[df["dropped_source"] != "NONE_BASELINE"].copy()

    top_f1 = float(baseline["top_method_weighted_f1"])
    return {
        "source_baseline_pi": float(baseline["pi_prevalence"]),
        "source_baseline_rho": float(baseline["rho_burst"]),
        "source_baseline_top_method": str(baseline["top_method"]),
        "source_baseline_top_weighted_f1": top_f1,
        "source_mean_rank_disagreement": float(nonbase["pairwise_rank_disagreement_vs_baseline"].mean()),
        "source_max_rank_disagreement": float(nonbase["pairwise_rank_disagreement_vs_baseline"].max()),
        "source_mean_abs_delta_pi": float(np.mean(np.abs(nonbase["delta_pi_vs_baseline"]))),
        "source_max_abs_delta_pi": float(np.max(np.abs(nonbase["delta_pi_vs_baseline"]))),
        "source_mean_abs_delta_rho": float(np.mean(np.abs(nonbase["delta_rho_vs_baseline"]))),
        "f1_penalty": max(0.0, f1_target - top_f1),
    }


def _summarize_family_block(path: Path) -> Dict[str, float]:
    df = pd.read_csv(path)
    nonbase = df[df["dropped_family"] != "NONE_BASELINE"].copy()
    return {
        "block_mean_rank_disagreement": float(nonbase["pairwise_rank_disagreement_vs_baseline"].mean()),
        "block_max_rank_disagreement": float(nonbase["pairwise_rank_disagreement_vs_baseline"].max()),
        "block_mean_abs_delta_pi": float(np.mean(np.abs(nonbase["delta_pi_vs_baseline"]))),
        "block_max_abs_delta_pi": float(np.max(np.abs(nonbase["delta_pi_vs_baseline"]))),
        "block_mean_abs_delta_rho": float(np.mean(np.abs(nonbase["delta_rho_vs_baseline"]))),
    }


def _perturbation_metrics(
    posterior_path: Path,
    panel_edges: Dict[str, EdgeSet],
    min_panel_overlap: int,
    k_list: Sequence[int],
) -> Dict[str, float | int | str]:
    post = pd.read_csv(posterior_path)
    required = {"source", "target", "posterior"}
    if not required.issubset(post.columns):
        raise ValueError(f"{posterior_path}: missing columns {sorted(required)}")

    edge = post[["source", "target", "posterior"]].copy()
    edge["n_observed_sources"] = post["n_observed_sources"] if "n_observed_sources" in post.columns else 0
    edge["n_positive_sources"] = post["n_positive_sources"] if "n_positive_sources" in post.columns else 0
    edge["calibration_score"] = (
        edge["posterior"]
        + 1e-6 * edge["n_observed_sources"].astype(float)
        + 1e-9 * edge["n_positive_sources"].astype(float)
    )
    edge["edge_key"] = list(zip(edge["source"], edge["target"]))

    universe = _edge_set(edge)
    ref_edges, high_labels = _build_high_overlap_union(panel_edges, universe, min_panel_overlap)
    edge["is_reference"] = edge["edge_key"].isin(ref_edges).astype(int)

    y = edge["is_reference"].to_numpy(dtype=int)
    s = edge["calibration_score"].to_numpy(dtype=float)
    prevalence = float(np.mean(y))

    out: Dict[str, float | int | str] = {
        "perturb_reference_set": "high_overlap_union",
        "perturb_high_overlap_panels": ";".join(sorted(high_labels)),
        "perturb_n_reference_edges_in_universe": int(np.sum(y)),
        "perturb_reference_prevalence": prevalence,
    }
    for k in k_list:
        p, r, tp = _precision_recall_at_k(y, s, k)
        out[f"perturb_precision_at_{k}"] = p
        out[f"perturb_recall_at_{k}"] = r
        out[f"perturb_tp_at_{k}"] = tp
    return out


def _ci95(mean: float, std: float, n: int) -> Tuple[float, float]:
    if n <= 1 or np.isnan(std):
        return float(mean), float(mean)
    half = 1.959963984540054 * float(std) / math.sqrt(float(n))
    return float(mean - half), float(mean + half)


def _aggregate_seed_runs(seed_runs: pd.DataFrame, group_cols: Sequence[str]) -> pd.DataFrame:
    numeric_cols = [
        c
        for c in seed_runs.columns
        if c not in set(group_cols) | {"seed"} and pd.api.types.is_numeric_dtype(seed_runs[c])
    ]
    string_cols = [
        c
        for c in seed_runs.columns
        if c not in set(group_cols) | set(numeric_cols) | {"seed"}
    ]

    g = seed_runs.groupby(list(group_cols), as_index=False)
    mean_df = g[numeric_cols].mean().rename(columns={c: f"{c}_mean" for c in numeric_cols})
    std_df = g[numeric_cols].std().rename(columns={c: f"{c}_std" for c in numeric_cols})
    n_df = g["seed"].count().rename(columns={"seed": "n_seeds"})

    out = mean_df.merge(std_df, on=list(group_cols), how="left").merge(n_df, on=list(group_cols), how="left")

    if string_cols:
        first_df = g[string_cols].first()
        out = out.merge(first_df, on=list(group_cols), how="left")

    for c in numeric_cols:
        m_col = f"{c}_mean"
        s_col = f"{c}_std"
        lo_col = f"{c}_ci95_low"
        hi_col = f"{c}_ci95_high"
        ci = out.apply(
            lambda r: _ci95(
                mean=float(r[m_col]),
                std=float(r[s_col]) if not pd.isna(r[s_col]) else float("nan"),
                n=int(r["n_seeds"]),
            ),
            axis=1,
        )
        out[lo_col] = [x[0] for x in ci]
        out[hi_col] = [x[1] for x in ci]

    return out


def main() -> None:
    args = _parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    a_grid = _parse_float_grid(args.source_strength_alpha_grid)
    b_grid = _parse_float_grid(args.source_strength_beta_grid)
    f_grid = _parse_float_grid(args.family_strength_grid)
    g_grid = _parse_float_grid(args.global_prior_strength_grid)
    seeds = _parse_seed_list(args.seeds)
    k_list = _parse_int_list(args.perturbation_k_list)
    k_weights = _parse_weight_list(args.perturbation_k_weights)

    if len(k_list) != len(k_weights):
        raise ValueError("--perturbation-k-list and --perturbation-k-weights must have the same length.")

    panel_edges = _load_panel_edges(args.repo_root, args.panel_glob)
    max_k = max(k_list)
    impl = "subproject_24_multireference_latent_truth_model/implementation"
    seed_rows: List[Dict[str, object]] = []

    combos = list(itertools.product(a_grid, b_grid, f_grid, g_grid))
    run_idx = 0
    for sa, sb, fs, gs in combos:
        run_id = f"sa{sa:g}_sb{sb:g}_fs{fs:g}_gs{gs:g}"
        for seed in seeds:
            run_idx += 1
            print(f"[{run_idx}/{len(combos) * len(seeds)}] Running {run_id} seed={seed}")

            base_dir = args.output_dir / run_id / f"seed_{seed}"
            source_dir = base_dir / "source_dropout"
            block_dir = base_dir / "family_block_dropout"

            fit_extra = (
                f"--source-strength-alpha {sa} "
                f"--source-strength-beta {sb} "
                f"--family-strength {fs} "
                f"--global-prior-strength {gs} "
                f"--min-youden-clean {args.min_youden_clean} "
                f"--edge-subsample-seed {seed}"
            )

            source_cmd = [
                "python",
                f"{impl}/run_source_dropout_ablation.py",
                "--matrix",
                str(args.matrix),
                "--source-meta",
                str(args.source_meta),
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
                str(source_dir),
            ]
            block_cmd = [
                "python",
                f"{impl}/run_family_block_dropout_ablation.py",
                "--matrix",
                str(args.matrix),
                "--source-meta",
                str(args.source_meta),
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
                str(block_dir),
            ]

            _run(source_cmd, cwd=args.repo_root)
            _run(block_cmd, cwd=args.repo_root)

            source_summary = _summarize_source_dropout(source_dir / "source_dropout_summary.csv", f1_target=args.f1_target)
            block_summary = _summarize_family_block(block_dir / "family_block_dropout_summary.csv")
            perturb_summary = _perturbation_metrics(
                source_dir / "baseline" / "ltm_family_hb_edge_posteriors.csv",
                panel_edges=panel_edges,
                min_panel_overlap=args.min_panel_overlap,
                k_list=k_list,
            )

            source_objective = (
                float(source_summary["source_mean_rank_disagreement"])
                + 0.5 * float(source_summary["source_max_rank_disagreement"])
                + 0.5 * float(source_summary["source_mean_abs_delta_pi"])
                + 0.2 * float(source_summary["source_mean_abs_delta_rho"])
                + 0.5 * float(source_summary["f1_penalty"])
            )
            block_objective = args.family_block_weight * (
                float(block_summary["block_mean_rank_disagreement"])
                + 0.5 * float(block_summary["block_max_rank_disagreement"])
                + 0.5 * float(block_summary["block_mean_abs_delta_pi"])
                + 0.2 * float(block_summary["block_mean_abs_delta_rho"])
            )

            weighted_precision = 0.0
            for k, w in zip(k_list, k_weights):
                weighted_precision += w * float(perturb_summary[f"perturb_precision_at_{k}"])
            prevalence = float(perturb_summary["perturb_reference_prevalence"])
            precision_lift = weighted_precision - prevalence
            recall_max = float(perturb_summary[f"perturb_recall_at_{max_k}"])
            background_penalty = args.background_penalty_weight * (1.0 - recall_max)
            perturb_raw_utility = precision_lift - background_penalty
            perturb_reward = args.perturbation_weight * perturb_raw_utility

            objective = source_objective + block_objective - perturb_reward

            row: Dict[str, object] = {
                "run_id": run_id,
                "seed": int(seed),
                "source_strength_alpha": sa,
                "source_strength_beta": sb,
                "family_strength": fs,
                "global_prior_strength": gs,
                "source_objective_component": source_objective,
                "block_objective_component": block_objective,
                "perturb_weighted_precision": weighted_precision,
                "perturb_precision_lift": precision_lift,
                "perturb_background_penalty": background_penalty,
                "perturb_raw_utility": perturb_raw_utility,
                "perturbation_reward_component": perturb_reward,
                "objective_score": objective,
                "source_dropout_dir": str(source_dir),
                "family_block_dropout_dir": str(block_dir),
                "matrix": str(args.matrix),
                "source_meta": str(args.source_meta),
                "k_list": ",".join(str(k) for k in k_list),
                "k_weights": ",".join(f"{w:.6f}" for w in k_weights),
            }
            row.update(source_summary)
            row.update(block_summary)
            row.update(perturb_summary)
            seed_rows.append(row)

    seed_runs = pd.DataFrame(seed_rows).sort_values(
        ["run_id", "seed"],
        ascending=[True, True],
    ).reset_index(drop=True)
    seed_runs.to_csv(args.output_dir / "family_hb_joint_tuning_seed_runs.csv", index=False)

    group_cols = [
        "run_id",
        "source_strength_alpha",
        "source_strength_beta",
        "family_strength",
        "global_prior_strength",
        "matrix",
        "source_meta",
        "k_list",
        "k_weights",
    ]
    summary = _aggregate_seed_runs(seed_runs, group_cols=group_cols).sort_values(
        ["objective_score_mean", "source_mean_rank_disagreement_mean", "block_mean_rank_disagreement_mean"],
        ascending=[True, True, True],
    ).reset_index(drop=True)
    summary.to_csv(args.output_dir / "family_hb_joint_tuning_summary.csv", index=False)
    pd.DataFrame([summary.iloc[0]]).to_csv(args.output_dir / "family_hb_joint_tuning_best.csv", index=False)

    print("Joint Family-HB tuning complete:")
    print(summary.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
