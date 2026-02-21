#!/usr/bin/env python3
"""Run synthetic missingness stress tests across latent-truth regimes."""

from __future__ import annotations

import argparse
import subprocess
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

from ltm_utils import parse_kind_list


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
    parser.add_argument("--include-kinds", type=str, default="reference,mechanistic")
    parser.add_argument("--models", type=str, default="dep,mf,family_hb")
    parser.add_argument("--levels", type=str, default="0.2,0.4")
    parser.add_argument(
        "--scenarios",
        type=str,
        default="global_random,curated_dorothea,timeseries_proxy",
        help="Comma-separated scenarios: global_random or family IDs.",
    )
    parser.add_argument("--max-edges", type=int, default=120000)
    parser.add_argument("--max-iter", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("subproject_24_multireference_latent_truth_model/implementation/outputs/synthetic_missingness_stress"),
    )
    return parser.parse_args()


def _parse_csv_list(s: str) -> List[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def _default_family(source_id: str, kind: str) -> str:
    if source_id.startswith("dorothea_"):
        return "curated_dorothea"
    if source_id == "trrust_human":
        return "curated_trrust"
    if source_id in {"beeline_gsd", "hpn_dream"}:
        return "small_gold_standard"
    if source_id == "timeseries_proxy":
        return "timeseries_proxy"
    if source_id == "dixit_perturb_top500":
        return "perturbation_proxy"
    if source_id == "invariant_top_edges":
        return "mechanistic_invariant"
    return f"{kind}_other"


def _annotators_and_families(source_meta: pd.DataFrame, include_kinds: Sequence[str]) -> Tuple[List[str], Dict[str, str]]:
    sub = source_meta[source_meta["kind"].isin(include_kinds)].copy()
    sub = sub.sort_values("source_id").reset_index(drop=True)
    ann = sub["source_id"].tolist()
    fam = {r["source_id"]: _default_family(str(r["source_id"]), str(r["kind"])) for _, r in sub.iterrows()}
    return ann, fam


def _model_specs(model_id: str) -> Tuple[str, str, str]:
    impl = "subproject_24_multireference_latent_truth_model/implementation"
    mapping = {
        "dep": (f"{impl}/fit_dependent_ltm.py", "ltm_dep_edge_posteriors.csv", "ltm_dep_fit_summary.csv"),
        "mf": (f"{impl}/fit_multifactor_ltm.py", "ltm_mf_edge_posteriors.csv", "ltm_mf_fit_summary.csv"),
        "family_hb": (
            f"{impl}/fit_family_hierarchical_bayes_ltm.py",
            "ltm_family_hb_edge_posteriors.csv",
            "ltm_family_hb_fit_summary.csv",
        ),
    }
    if model_id not in mapping:
        raise ValueError(f"Unsupported model id '{model_id}'.")
    return mapping[model_id]


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


def _disagreement(a: Dict[Tuple[str, str], int], b: Dict[Tuple[str, str], int]) -> Tuple[float, int, int]:
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
        return float("nan"), diff, valid
    return diff / float(valid), diff, valid


def _extract_rho_total(row: pd.Series) -> float:
    if "rho_burst" in row and not pd.isna(row["rho_burst"]):
        return float(row["rho_burst"])
    if "rho_burst1" in row and "rho_burst2" in row:
        if not pd.isna(row["rho_burst1"]) and not pd.isna(row["rho_burst2"]):
            return float(row["rho_burst1"] + row["rho_burst2"])
    return float("nan")


def _corrupt_matrix(
    matrix: pd.DataFrame,
    annotators: Sequence[str],
    families: Dict[str, str],
    scenario: str,
    level: float,
    rng: np.random.Generator,
) -> pd.DataFrame:
    out = matrix.copy()
    if scenario == "global_random":
        cols = list(annotators)
    else:
        cols = [a for a in annotators if families[a] == scenario]
        if not cols:
            raise ValueError(f"Scenario '{scenario}' matched no annotator columns.")

    for c in cols:
        arr = out[c].to_numpy(dtype=float)
        observed = ~np.isnan(arr)
        if not np.any(observed):
            continue
        mask = rng.random(size=arr.shape[0]) < level
        drop = observed & mask
        arr[drop] = np.nan
        out[c] = arr

    return out


def main() -> None:
    args = _parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "matrices").mkdir(parents=True, exist_ok=True)

    matrix = pd.read_csv(args.matrix)
    if args.max_edges > 0:
        matrix = matrix.head(args.max_edges).copy()

    source_meta = pd.read_csv(args.source_meta)
    include_kinds = parse_kind_list(args.include_kinds)
    annotators, families = _annotators_and_families(source_meta, include_kinds=include_kinds)
    if not annotators:
        raise ValueError("No annotators selected.")

    # Keep all matrix columns so method-evaluation scripts can access method outputs.
    # Fit scripts only consume their required annotator subset internally.
    matrix = matrix.copy()

    models = _parse_csv_list(args.models)
    levels = [float(x) for x in _parse_csv_list(args.levels)]
    scenarios = _parse_csv_list(args.scenarios)

    rng = np.random.default_rng(args.seed)

    impl = "subproject_24_multireference_latent_truth_model/implementation"
    baseline_ctx: Dict[str, Dict[str, object]] = {}

    # Baseline fits for each model on the same subset.
    for model in models:
        fit_script, posterior_file, fit_summary_file = _model_specs(model)
        run_dir = args.output_dir / f"baseline_{model}"
        run_dir.mkdir(parents=True, exist_ok=True)

        baseline_matrix_path = args.output_dir / "matrices" / "baseline_matrix.csv"
        if not baseline_matrix_path.exists():
            matrix.to_csv(baseline_matrix_path, index=False)

        _run(
            [
                "python",
                fit_script,
                "--matrix",
                str(baseline_matrix_path),
                "--source-meta",
                str(args.source_meta),
                "--include-kinds",
                args.include_kinds,
                "--max-edges",
                "0",
                "--max-iter",
                str(args.max_iter),
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
                str(baseline_matrix_path),
                "--source-meta",
                str(args.source_meta),
                "--posterior",
                str(run_dir / posterior_file),
                "--posterior-col",
                "posterior",
                "--output-dir",
                str(run_dir),
            ],
            cwd=args.repo_root,
        )

        base_fit = pd.read_csv(run_dir / fit_summary_file).iloc[0]
        base_metrics = pd.read_csv(run_dir / "posterior_method_metrics.csv")
        base_orders = _pairwise_orders(base_metrics[["method", "weighted_f1"]], "weighted_f1")
        base_post = pd.read_csv(run_dir / posterior_file)[["source", "target", "posterior"]]
        baseline_ctx[model] = {
            "fit": base_fit,
            "metrics": base_metrics,
            "orders": base_orders,
            "posterior": base_post,
            "posterior_file": posterior_file,
            "fit_summary_file": fit_summary_file,
        }

    rows: List[Dict[str, object]] = []
    scenario_rows: List[Dict[str, object]] = []

    for scenario in scenarios:
        for level in levels:
            corrupted = _corrupt_matrix(
                matrix=matrix,
                annotators=annotators,
                families=families,
                scenario=scenario,
                level=level,
                rng=rng,
            )
            missing_fraction = float(
                np.mean(np.isnan(corrupted[annotators].to_numpy(dtype=float)))
            )
            label = f"{scenario}_p{level:.2f}".replace(".", "p")
            matrix_path = args.output_dir / "matrices" / f"{label}.csv"
            corrupted.to_csv(matrix_path, index=False)

            for model in models:
                fit_script, posterior_file, fit_summary_file = _model_specs(model)
                run_dir = args.output_dir / f"{label}_{model}"
                run_dir.mkdir(parents=True, exist_ok=True)

                _run(
                    [
                        "python",
                        fit_script,
                        "--matrix",
                        str(matrix_path),
                        "--source-meta",
                        str(args.source_meta),
                        "--include-kinds",
                        args.include_kinds,
                        "--max-edges",
                        "0",
                        "--max-iter",
                        str(args.max_iter),
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
                        str(matrix_path),
                        "--source-meta",
                        str(args.source_meta),
                        "--posterior",
                        str(run_dir / posterior_file),
                        "--posterior-col",
                        "posterior",
                        "--output-dir",
                        str(run_dir),
                    ],
                    cwd=args.repo_root,
                )

                fit = pd.read_csv(run_dir / fit_summary_file).iloc[0]
                metrics = pd.read_csv(run_dir / "posterior_method_metrics.csv")
                orders = _pairwise_orders(metrics[["method", "weighted_f1"]], "weighted_f1")
                dis, ndiff, nvalid = _disagreement(
                    baseline_ctx[model]["orders"],  # type: ignore[index]
                    orders,
                )

                top_method = str(metrics.sort_values("weighted_f1", ascending=False).iloc[0]["method"])
                top_wf1 = float(metrics.sort_values("weighted_f1", ascending=False).iloc[0]["weighted_f1"])
                base_top = str(
                    baseline_ctx[model]["metrics"]  # type: ignore[index]
                    .sort_values("weighted_f1", ascending=False)
                    .iloc[0]["method"]
                )

                post = pd.read_csv(run_dir / posterior_file)[["source", "target", "posterior"]]
                merged = baseline_ctx[model]["posterior"].merge(  # type: ignore[index]
                    post,
                    on=["source", "target"],
                    how="inner",
                    suffixes=("_base", "_stress"),
                )
                pearson = float(np.corrcoef(merged["posterior_base"], merged["posterior_stress"])[0, 1])
                spearman = float(
                    pd.Series(merged["posterior_base"]).corr(
                        pd.Series(merged["posterior_stress"]),
                        method="spearman",
                    )
                )

                base_fit = baseline_ctx[model]["fit"]  # type: ignore[index]
                delta_pi = float(fit["pi_prevalence"] - base_fit["pi_prevalence"])
                delta_rho = float(_extract_rho_total(fit) - _extract_rho_total(base_fit))

                row = {
                    "model": model,
                    "scenario": scenario,
                    "level": level,
                    "stress_label": label,
                    "missing_fraction": missing_fraction,
                    "n_edges": int(fit["n_edges"]),
                    "pi_prevalence": float(fit["pi_prevalence"]),
                    "delta_pi_vs_baseline": delta_pi,
                    "rho_total": float(_extract_rho_total(fit)),
                    "delta_rho_total_vs_baseline": delta_rho,
                    "top_method": top_method,
                    "top_method_weighted_f1": top_wf1,
                    "top_method_changed": int(top_method != base_top),
                    "pairwise_rank_disagreement_vs_baseline": dis,
                    "n_disagree_pairs": int(ndiff),
                    "n_valid_pairs": int(nvalid),
                    "posterior_pearson_vs_baseline": pearson,
                    "posterior_spearman_vs_baseline": spearman,
                }
                rows.append(row)
                scenario_rows.append(row)

    detail = pd.DataFrame(rows).sort_values(
        ["scenario", "level", "model"]
    ).reset_index(drop=True)
    summary = (
        pd.DataFrame(scenario_rows)
        .groupby(["model", "scenario"], as_index=False)
        .agg(
            n_runs=("stress_label", "count"),
            mean_missing_fraction=("missing_fraction", "mean"),
            mean_rank_disagreement=("pairwise_rank_disagreement_vs_baseline", "mean"),
            max_rank_disagreement=("pairwise_rank_disagreement_vs_baseline", "max"),
            mean_abs_delta_pi=("delta_pi_vs_baseline", lambda x: float(np.mean(np.abs(x)))),
            max_abs_delta_pi=("delta_pi_vs_baseline", lambda x: float(np.max(np.abs(x)))),
            mean_abs_delta_rho=("delta_rho_total_vs_baseline", lambda x: float(np.mean(np.abs(x)))),
            mean_posterior_pearson=("posterior_pearson_vs_baseline", "mean"),
            mean_posterior_spearman=("posterior_spearman_vs_baseline", "mean"),
            top_method_change_rate=("top_method_changed", "mean"),
        )
        .sort_values(["scenario", "mean_rank_disagreement", "mean_abs_delta_pi"])
        .reset_index(drop=True)
    )

    overall = (
        pd.DataFrame(scenario_rows)
        .groupby("model", as_index=False)
        .agg(
            n_runs=("stress_label", "count"),
            mean_rank_disagreement=("pairwise_rank_disagreement_vs_baseline", "mean"),
            max_rank_disagreement=("pairwise_rank_disagreement_vs_baseline", "max"),
            mean_abs_delta_pi=("delta_pi_vs_baseline", lambda x: float(np.mean(np.abs(x)))),
            max_abs_delta_pi=("delta_pi_vs_baseline", lambda x: float(np.max(np.abs(x)))),
            mean_abs_delta_rho=("delta_rho_total_vs_baseline", lambda x: float(np.mean(np.abs(x)))),
            mean_posterior_pearson=("posterior_pearson_vs_baseline", "mean"),
            mean_posterior_spearman=("posterior_spearman_vs_baseline", "mean"),
            top_method_change_rate=("top_method_changed", "mean"),
        )
        .sort_values("mean_rank_disagreement")
        .reset_index(drop=True)
    )

    detail.to_csv(args.output_dir / "synthetic_missingness_stress_detail.csv", index=False)
    summary.to_csv(args.output_dir / "synthetic_missingness_stress_summary.csv", index=False)
    overall.to_csv(args.output_dir / "synthetic_missingness_stress_overall.csv", index=False)

    print("Synthetic missingness stress complete:")
    print(overall.to_string(index=False))


if __name__ == "__main__":
    main()
