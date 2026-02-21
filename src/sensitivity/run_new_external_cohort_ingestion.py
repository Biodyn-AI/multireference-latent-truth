#!/usr/bin/env python3
"""Run strict new-cohort external ingestion under a frozen manifest.

This script is designed for Phase XXVII-style expansion where new external
cohorts must be evaluated without touching the existing cohort registry.
If no new cohorts are discovered, it writes an explicit blocker summary.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from datetime import datetime, timezone
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
        "--panel-list",
        type=Path,
        default=None,
        help="Optional explicit panel list; if provided, --panel-globs is ignored.",
    )
    parser.add_argument(
        "--known-cohorts-csv",
        type=Path,
        default=Path(
            "subproject_24_multireference_latent_truth_model/implementation/outputs/"
            "top_tier_readiness_v6/multicohort_preregistered_replication_v1/"
            "multicohort_replication_results.csv"
        ),
    )
    parser.add_argument("--known-cohort-column", type=str, default="cohort")
    parser.add_argument("--known-status-column", type=str, default="status")
    parser.add_argument("--known-completed-value", type=str, default="completed")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "subproject_24_multireference_latent_truth_model/implementation/outputs/"
            "top_tier_readiness_v7/new_external_cohort_ingestion_v1"
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
        raise ValueError("No panel files discovered from --panel-globs")
    return uniq


def _read_panel_list(path: Path) -> List[Path]:
    if not path.exists():
        raise ValueError(f"Panel list path not found: {path}")
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines:
        raise ValueError(f"Panel list is empty: {path}")
    return sorted(Path(x) for x in lines)


def _infer_cohort(path: Path) -> str:
    label = path.parent.name.lower()
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
    # Common public perturb-seq cohort labels if added later.
    if label.startswith("norman"):
        return "norman"
    if label.startswith("replogle"):
        return "replogle"
    if label.startswith("frangieh"):
        return "frangieh"
    return path.parent.name


def _write_list(path: Path, entries: List[Path]) -> None:
    path.write_text("\n".join(str(p) for p in entries), encoding="utf-8")


def _to_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    text = str(value).strip().lower()
    return text in {"1", "true", "t", "yes", "y"}


def _load_known_cohorts(args: argparse.Namespace) -> List[str]:
    if not args.known_cohorts_csv.exists():
        raise ValueError(f"Known cohort registry not found: {args.known_cohorts_csv}")
    df = pd.read_csv(args.known_cohorts_csv, low_memory=False)
    if args.known_cohort_column not in df.columns:
        raise ValueError(f"Missing known cohort column: {args.known_cohort_column}")
    if args.known_status_column in df.columns:
        df = df[df[args.known_status_column].astype(str) == str(args.known_completed_value)]
    return sorted(set(df[args.known_cohort_column].astype(str).tolist()))


def _extract_endpoint_row(run_dir: Path) -> Dict[str, object]:
    endpoint = pd.read_csv(run_dir / "preregistered_external_endpoint_summary.csv").iloc[0]
    endpoint_pass = _to_bool(endpoint.get("endpoint_pass"))
    return {
        "selected_policy": str(endpoint["selected_policy"]),
        "selected_enrichment": float(endpoint["selected_enrichment"]),
        "selected_p_value": float(endpoint["selected_p_value"]),
        "endpoint_pass": endpoint_pass,
        "endpoint_min_support": float(endpoint.get("endpoint_min_support", float("nan"))),
        "endpoint_support_eligible": _to_bool(endpoint.get("endpoint_support_eligible")),
        "endpoint_support_tier": str(endpoint.get("endpoint_support_tier", "")),
        "n_reference_edges_in_candidate_pool": float(endpoint.get("selected_n_reference_edges_in_candidate_pool", float("nan"))),
        "n_hits": float(endpoint.get("selected_n_hits", float("nan"))),
    }


def _wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n <= 0:
        return (float("nan"), float("nan"))
    phat = k / n
    denom = 1.0 + (z * z) / n
    center = (phat + (z * z) / (2 * n)) / denom
    margin = (z / denom) * math.sqrt((phat * (1 - phat) / n) + ((z * z) / (4 * n * n)))
    return (center - margin, center + margin)


def main() -> None:
    args = _parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    panels = _read_panel_list(args.panel_list) if args.panel_list is not None else _discover(args.panel_globs)
    known = _load_known_cohorts(args)
    known_set = set(known)

    inventory_rows: List[Dict[str, object]] = []
    for p in panels:
        cohort = _infer_cohort(p)
        inventory_rows.append(
            {
                "panel_path": str(p),
                "panel_label": p.parent.name,
                "cohort": cohort,
                "is_known_cohort": cohort in known_set,
            }
        )
    inventory = pd.DataFrame(inventory_rows).sort_values(["cohort", "panel_label"]).reset_index(drop=True)
    inventory.to_csv(args.output_dir / "new_cohort_panel_inventory.csv", index=False)

    discovered = sorted(set(inventory["cohort"].astype(str).tolist()))
    new_cohorts = sorted([c for c in discovered if c not in known_set])

    manifest = {
        "protocol_id": "new_external_cohort_ingestion_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "budget": int(args.budget),
        "alphas": args.alphas,
        "candidate_mode": args.candidate_mode,
        "endpoint_min_enrichment": float(args.endpoint_min_enrichment),
        "endpoint_max_p": float(args.endpoint_max_p),
        "endpoint_min_support": float(args.endpoint_min_support),
        "known_cohort_registry": str(args.known_cohorts_csv),
        "known_cohorts": known,
        "n_known_cohorts": len(known),
        "n_discovered_cohorts": len(discovered),
        "discovered_cohorts": discovered,
        "n_new_cohorts": len(new_cohorts),
        "new_cohorts": new_cohorts,
        "panel_list": str(args.panel_list) if args.panel_list is not None else "",
        "panel_globs": args.panel_globs,
    }
    (args.output_dir / "new_cohort_ingestion_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    if not new_cohorts:
        blocker = pd.DataFrame(
            [
                {
                    "status": "blocked_no_new_cohorts",
                    "n_known_cohorts": len(known),
                    "n_discovered_cohorts": len(discovered),
                    "n_new_cohorts": 0,
                    "message": "No truly new cohorts discovered under current panel inventory.",
                }
            ]
        )
        blocker.to_csv(args.output_dir / "new_cohort_ingestion_summary.csv", index=False)
        pd.DataFrame(
            [{"cohort": c, "status": "known_existing"} for c in discovered]
        ).to_csv(args.output_dir / "new_cohort_ingestion_results.csv", index=False)
        print("New-cohort ingestion blocked: no unseen cohorts found.")
        return

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

    results_rows: List[Dict[str, object]] = []
    for cohort in new_cohorts:
        cohort_panels = [Path(x) for x in inventory[inventory["cohort"] == cohort]["panel_path"].tolist()]
        primary_panels = [Path(x) for x in inventory[inventory["cohort"] != cohort]["panel_path"].tolist()]
        if len(cohort_panels) < args.min_external_panels or len(primary_panels) < args.min_primary_panels:
            results_rows.append(
                {
                    "cohort": cohort,
                    "status": "skipped",
                    "reason": "insufficient primary/external panels",
                    "n_primary_panels": len(primary_panels),
                    "n_external_panels": len(cohort_panels),
                }
            )
            continue

        cohort_dir = args.output_dir / f"cohort_{cohort}"
        cohort_dir.mkdir(parents=True, exist_ok=True)
        primary_list = cohort_dir / "primary_panel_list.txt"
        external_list = cohort_dir / "external_panel_list.txt"
        _write_list(primary_list, primary_panels)
        _write_list(external_list, cohort_panels)

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

        endpoint_row = _extract_endpoint_row(cohort_dir / "run")
        results_rows.append(
            {
                "cohort": cohort,
                "status": "completed",
                "reason": "",
                "n_primary_panels": len(primary_panels),
                "n_external_panels": len(cohort_panels),
                **endpoint_row,
            }
        )

    results = pd.DataFrame(results_rows).sort_values(["status", "cohort"]).reset_index(drop=True)
    results.to_csv(args.output_dir / "new_cohort_ingestion_results.csv", index=False)

    completed = results[results["status"] == "completed"].copy()
    n = int(len(completed))
    k = int(completed["endpoint_pass"].map(_to_bool).sum()) if n > 0 else 0
    lo, hi = _wilson_ci(k, n)
    pd.DataFrame(
        [
            {
                "status": "completed_new_cohort_ingestion",
                "n_new_cohorts_requested": len(new_cohorts),
                "n_completed": n,
                "n_endpoint_pass": k,
                "endpoint_pass_rate": float(k / n) if n > 0 else float("nan"),
                "endpoint_pass_rate_wilson_ci_low": lo,
                "endpoint_pass_rate_wilson_ci_high": hi,
            }
        ]
    ).to_csv(args.output_dir / "new_cohort_ingestion_summary.csv", index=False)

    print("New external cohort ingestion complete.")
    print(f"Discovered cohorts: {len(discovered)}")
    print(f"New cohorts: {len(new_cohorts)}")
    print(f"Completed cohort ingestions: {n}")


if __name__ == "__main__":
    main()
