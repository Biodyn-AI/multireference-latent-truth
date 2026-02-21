#!/usr/bin/env python3
"""Run leave-one-cohort-out pre-registered external replications.

This script treats each discovered panel cohort as reserved external once,
while all other cohorts form the primary pool for policy selection.
"""

from __future__ import annotations

import argparse
import math
import subprocess
from pathlib import Path
from typing import Dict, List

import pandas as pd


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--budget", type=int, default=4000)
    parser.add_argument("--alphas", type=str, default="0.0,0.0001,0.0002,0.0005,0.001,0.002,0.005,0.01")
    parser.add_argument("--candidate-mode", type=str, default="up_to_conditional")
    parser.add_argument("--endpoint-min-enrichment", type=float, default=1.5)
    parser.add_argument("--endpoint-max-p", type=float, default=0.05)
    parser.add_argument("--endpoint-min-support", type=float, default=10.0)
    parser.add_argument("--min-primary-panels", type=int, default=4)
    parser.add_argument("--min-external-panels", type=int, default=1)
    parser.add_argument(
        "--panel-globs",
        type=str,
        default=(
            "single_cell_mechinterp/outputs/perturb_validation/*/perturbation_edges.tsv,"
            "single_cell_mechinterp/outputs/head_layer_checkpoint_replication/*/perturbation_edges.tsv"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "subproject_24_multireference_latent_truth_model/implementation/outputs/"
            "top_tier_readiness_v6/multicohort_preregistered_replication_v1"
        ),
    )
    return parser.parse_args()


def _run(cmd: List[str]) -> None:
    proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"Command failed ({proc.returncode}): {' '.join(cmd)}\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
        )


def _discover(panel_globs: str) -> List[Path]:
    out: List[Path] = []
    for pat in [x.strip() for x in panel_globs.split(",") if x.strip()]:
        out.extend(sorted(Path(".").glob(pat)))
    uniq = sorted(set(out))
    if not uniq:
        raise ValueError("No panel files discovered")
    return uniq


def _cohort_name(path: Path) -> str:
    label = path.parent.name
    if label.startswith("adamson"):
        return "adamson"
    if label.startswith("shifrut"):
        return "shifrut"
    if label.startswith("dixit7"):
        return "dixit7"
    if label.startswith("dixit"):
        return "dixit"
    if label.startswith("perturb_immune"):
        return "checkpoint_immune"
    if label.startswith("perturb_metabolic"):
        return "checkpoint_metabolic"
    return "other"


def _write_list(path: Path, entries: List[Path]) -> None:
    path.write_text("\n".join(str(p) for p in entries), encoding="utf-8")


def _wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n <= 0:
        return (float("nan"), float("nan"))
    phat = k / n
    denom = 1.0 + (z * z) / n
    center = (phat + (z * z) / (2 * n)) / denom
    margin = (z / denom) * math.sqrt((phat * (1 - phat) / n) + ((z * z) / (4 * n * n)))
    return (center - margin, center + margin)


def _to_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if math.isnan(float(value)):
            return False
        return bool(int(value))
    text = str(value).strip().lower()
    return text in {"1", "true", "t", "yes", "y"}


def _support_tier(
    support_edges: float,
    endpoint_pass: bool,
    min_support: float,
) -> str:
    support_eligible = bool(not math.isnan(float(support_edges)) and float(support_edges) >= float(min_support))
    if support_eligible and endpoint_pass:
        return "PASS_SUPPORT_ELIGIBLE"
    if support_eligible and not endpoint_pass:
        return "FAIL_SUPPORT_ELIGIBLE"
    if endpoint_pass:
        return "PASS_SUPPORT_LIMITED"
    return "FAIL_SUPPORT_LIMITED"


def main() -> None:
    args = _parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    panels = _discover(args.panel_globs)
    panel_df = pd.DataFrame(
        [{"panel_path": str(p), "panel_label": p.parent.name, "cohort": _cohort_name(p)} for p in panels]
    )
    panel_df.to_csv(args.output_dir / "panel_inventory_with_cohorts.csv", index=False)

    # Shared policy sweep reused across cohorts.
    shared_policy_dir = args.output_dir / "shared_policy"
    if not (shared_policy_dir / "deconcentration_policy_summary.csv").exists():
        _run(
            [
                "python",
                "subproject_24_multireference_latent_truth_model/implementation/run_deconcentration_policy_sweep.py",
                "--budget",
                str(args.budget),
                "--alphas",
                args.alphas,
                "--candidate-mode",
                args.candidate_mode,
                "--output-dir",
                str(shared_policy_dir),
            ]
        )

    cohort_rows: List[Dict[str, object]] = []
    for cohort, sub in panel_df.groupby("cohort", sort=True):
        external_panels = [Path(x) for x in sub["panel_path"].tolist()]
        primary_panels = [Path(x) for x in panel_df[panel_df["cohort"] != cohort]["panel_path"].tolist()]

        if len(external_panels) < args.min_external_panels or len(primary_panels) < args.min_primary_panels:
            cohort_rows.append(
                {
                    "cohort": cohort,
                    "status": "skipped",
                    "reason": "insufficient primary/external panel counts",
                    "n_primary_panels": len(primary_panels),
                    "n_external_panels": len(external_panels),
                }
            )
            continue

        cohort_dir = args.output_dir / f"cohort_{cohort}"
        cohort_dir.mkdir(parents=True, exist_ok=True)
        primary_list = cohort_dir / "primary_panel_list.txt"
        external_list = cohort_dir / "external_panel_list.txt"
        _write_list(primary_list, primary_panels)
        _write_list(external_list, external_panels)

        _run(
            [
                "python",
                "subproject_24_multireference_latent_truth_model/implementation/run_preregistered_external_validation.py",
                "--budget",
                str(args.budget),
                "--alphas",
                args.alphas,
                "--candidate-mode",
                args.candidate_mode,
                "--primary-panel-list",
                str(primary_list),
                "--external-panel-list",
                str(external_list),
                "--reuse-policy-dir",
                str(shared_policy_dir),
                "--endpoint-min-enrichment",
                str(args.endpoint_min_enrichment),
                "--endpoint-max-p",
                str(args.endpoint_max_p),
                "--endpoint-min-support",
                str(args.endpoint_min_support),
                "--output-dir",
                str(cohort_dir / "run"),
            ]
        )

        endpoint = pd.read_csv(cohort_dir / "run" / "preregistered_external_endpoint_summary.csv").iloc[0]
        comp = pd.read_csv(cohort_dir / "run" / "preregistered_external_endpoint_comparison.csv")
        sel = comp[comp["selection"] == str(endpoint["selected_policy"])]
        endpoint_pass = _to_bool(endpoint.get("endpoint_pass"))
        endpoint_support_eligible = _to_bool(endpoint.get("endpoint_support_eligible"))
        endpoint_min_support = float(endpoint.get("endpoint_min_support", args.endpoint_min_support))
        endpoint_support_tier = str(endpoint.get("endpoint_support_tier", ""))
        ref_cov = float(endpoint.get("selected_n_reference_edges_in_candidate_pool", float("nan")))
        hits = float(endpoint.get("selected_n_hits", float("nan")))
        if (math.isnan(ref_cov) or math.isnan(hits)) and not sel.empty:
            ref_cov = float(sel.iloc[0]["n_reference_edges_in_candidate_pool"])
            hits = float(sel.iloc[0]["n_hits"])
        if not endpoint_support_tier:
            endpoint_support_tier = _support_tier(
                support_edges=ref_cov,
                endpoint_pass=endpoint_pass,
                min_support=endpoint_min_support,
            )
        cohort_rows.append(
            {
                "cohort": cohort,
                "status": "completed",
                "reason": "",
                "n_primary_panels": len(primary_panels),
                "n_external_panels": len(external_panels),
                "selected_policy": str(endpoint["selected_policy"]),
                "selected_enrichment": float(endpoint["selected_enrichment"]),
                "selected_p_value": float(endpoint["selected_p_value"]),
                "endpoint_pass": endpoint_pass,
                "endpoint_min_support": endpoint_min_support,
                "endpoint_support_eligible": endpoint_support_eligible,
                "endpoint_support_tier": endpoint_support_tier,
                "n_reference_edges_in_candidate_pool": ref_cov,
                "n_hits": hits,
            }
        )

    results = pd.DataFrame(cohort_rows).sort_values(["status", "cohort"]).reset_index(drop=True)
    results.to_csv(args.output_dir / "multicohort_replication_results.csv", index=False)

    completed = results[results["status"] == "completed"].copy()
    n = int(len(completed))
    k = int(completed["endpoint_pass"].sum()) if n > 0 else 0
    lo, hi = _wilson_ci(k, n)
    completed["endpoint_support_eligible"] = completed["endpoint_support_eligible"].map(_to_bool)
    eligible = completed[completed["endpoint_support_eligible"]].copy()
    limited = completed[~completed["endpoint_support_eligible"]].copy()
    n_eligible = int(len(eligible))
    k_eligible = int(eligible["endpoint_pass"].sum()) if n_eligible > 0 else 0
    lo_eligible, hi_eligible = _wilson_ci(k_eligible, n_eligible)
    n_limited = int(len(limited))
    k_limited = int(limited["endpoint_pass"].sum()) if n_limited > 0 else 0
    lo_limited, hi_limited = _wilson_ci(k_limited, n_limited)
    summary = pd.DataFrame(
        [
            {
                "n_cohorts_completed": n,
                "n_endpoint_pass": k,
                "endpoint_pass_rate": float(k / n) if n > 0 else float("nan"),
                "endpoint_pass_rate_wilson_ci_low": lo,
                "endpoint_pass_rate_wilson_ci_high": hi,
                "endpoint_min_support": float(args.endpoint_min_support),
                "n_support_eligible_cohorts": n_eligible,
                "n_support_eligible_pass": k_eligible,
                "support_eligible_pass_rate": float(k_eligible / n_eligible) if n_eligible > 0 else float("nan"),
                "support_eligible_pass_rate_wilson_ci_low": lo_eligible,
                "support_eligible_pass_rate_wilson_ci_high": hi_eligible,
                "n_support_limited_cohorts": n_limited,
                "n_support_limited_pass": k_limited,
                "support_limited_pass_rate": float(k_limited / n_limited) if n_limited > 0 else float("nan"),
                "support_limited_pass_rate_wilson_ci_low": lo_limited,
                "support_limited_pass_rate_wilson_ci_high": hi_limited,
                "mean_selected_enrichment": float(completed["selected_enrichment"].mean()) if n > 0 else float("nan"),
                "median_selected_enrichment": float(completed["selected_enrichment"].median()) if n > 0 else float("nan"),
                "max_selected_p_value": float(completed["selected_p_value"].max()) if n > 0 else float("nan"),
            }
        ]
    )
    summary.to_csv(args.output_dir / "multicohort_replication_summary.csv", index=False)

    tier_summary = (
        completed.groupby("endpoint_support_tier", as_index=False)
        .agg(
            n_cohorts=("cohort", "count"),
            n_pass=("endpoint_pass", "sum"),
            pass_rate=("endpoint_pass", "mean"),
            mean_support=("n_reference_edges_in_candidate_pool", "mean"),
            mean_enrichment=("selected_enrichment", "mean"),
            max_selected_p_value=("selected_p_value", "max"),
        )
        .sort_values("endpoint_support_tier")
        .reset_index(drop=True)
    )
    tier_summary.to_csv(args.output_dir / "multicohort_support_tier_summary.csv", index=False)

    print("Multi-cohort pre-registered replication complete.")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
