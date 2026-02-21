#!/usr/bin/env python3
"""Run leakage-safe pre-registered external validation protocol.

Protocol design:
1) Lock policy-search knobs in a manifest before evaluation.
2) Build candidate policies from governance edges at fixed budget.
3) Split perturbation panels into:
   - primary panels (train + tuning holdout),
   - reserved external panels (final evaluation only).
4) Select one policy using train + primary-holdout metrics only.
5) Evaluate the locked policy on reserved external panels and report endpoint pass/fail.
"""

from __future__ import annotations

import argparse
import json
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
    parser.add_argument("--baseline-policy", type=str, default="baseline_alpha0")
    parser.add_argument(
        "--primary-panel-glob",
        type=str,
        default="single_cell_mechinterp/outputs/perturb_validation/*/perturbation_edges.tsv",
    )
    parser.add_argument(
        "--primary-panel-list",
        type=Path,
        default=None,
        help=(
            "Optional newline list of primary panel paths. "
            "If provided, --primary-panel-glob is ignored."
        ),
    )
    parser.add_argument(
        "--external-panel-glob",
        type=str,
        default="single_cell_mechinterp/outputs/head_layer_checkpoint_replication/*/perturbation_edges.tsv",
    )
    parser.add_argument(
        "--external-panel-list",
        type=Path,
        default=None,
        help=(
            "Optional newline list of reserved external panel paths. "
            "If provided, --external-panel-glob is ignored."
        ),
    )
    parser.add_argument("--min-train-overlap", type=int, default=50)
    parser.add_argument(
        "--selection-train-reference-set",
        type=str,
        default="train_high_overlap_union",
        help="Reference set used as train objective in constrained selection.",
    )
    parser.add_argument(
        "--selection-holdout-reference-set",
        type=str,
        default="blind_holdout_primary_union",
        help="Reference set used for policy selection. External set is excluded by design.",
    )
    parser.add_argument(
        "--external-reference-set",
        type=str,
        default="blind_holdout_extra_union",
        help="Reserved external endpoint reference set.",
    )
    parser.add_argument("--max-top1", type=float, default=0.25)
    parser.add_argument("--max-top2", type=float, default=0.50)
    parser.add_argument("--min-joint-delta", type=float, default=-0.003)
    parser.add_argument("--min-train-hit-delta", type=int, default=-1)
    parser.add_argument("--min-train-enrichment", type=float, default=0.2)
    parser.add_argument("--endpoint-min-enrichment", type=float, default=1.5)
    parser.add_argument("--endpoint-max-p", type=float, default=0.05)
    parser.add_argument(
        "--endpoint-min-support",
        type=float,
        default=10.0,
        help=(
            "Minimum external candidate-space reference-edge support required "
            "to classify the endpoint as support-eligible."
        ),
    )
    parser.add_argument(
        "--reuse-policy-dir",
        type=Path,
        default=None,
        help=(
            "Optional directory with precomputed policy outputs from "
            "run_deconcentration_policy_sweep.py. If set, policy sweep is skipped."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "subproject_24_multireference_latent_truth_model/implementation/outputs/"
            "top_tier_readiness_v6/preregistered_external_protocol_v1"
        ),
    )
    return parser.parse_args()


def _run(cmd: List[str]) -> None:
    proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"Command failed ({proc.returncode}): {' '.join(cmd)}\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
        )


def _discover(pattern: str) -> List[str]:
    return sorted(str(p) for p in Path(".").glob(pattern))


def _read_panel_list(path: Path) -> List[str]:
    if not path.exists():
        raise ValueError(f"Panel list path not found: {path}")
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines:
        raise ValueError(f"Panel list is empty: {path}")
    return sorted(lines)


def _endpoint_pass(
    enrichment: float,
    p_value: float,
    min_enrichment: float,
    max_p: float,
) -> bool:
    return bool(float(enrichment) >= float(min_enrichment) and float(p_value) <= float(max_p))


def _support_tier(
    support_edges: float,
    endpoint_pass: bool,
    min_support: float,
) -> str:
    support_eligible = bool(pd.notna(support_edges) and float(support_edges) >= float(min_support))
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

    policy_dir = args.output_dir / "policy"
    blind_dir = args.output_dir / "blind"
    optimization_dir = args.output_dir / "selection_optimization"
    policy_dir.mkdir(parents=True, exist_ok=True)
    blind_dir.mkdir(parents=True, exist_ok=True)
    optimization_dir.mkdir(parents=True, exist_ok=True)

    if args.primary_panel_list is not None:
        primary_panels = _read_panel_list(args.primary_panel_list)
    else:
        primary_panels = _discover(args.primary_panel_glob)

    if args.external_panel_list is not None:
        external_panels = _read_panel_list(args.external_panel_list)
    else:
        external_panels = _discover(args.external_panel_glob)
    if not primary_panels:
        raise ValueError(f"No primary panels found for glob: {args.primary_panel_glob}")
    if not external_panels:
        raise ValueError(f"No external panels found for glob: {args.external_panel_glob}")

    # Write a locked manifest before the run. This is the protocol contract.
    manifest = {
        "protocol_id": "preregistered_external_protocol_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "budget": int(args.budget),
        "alphas": args.alphas,
        "candidate_mode": args.candidate_mode,
        "baseline_policy": args.baseline_policy,
        "primary_panel_glob": args.primary_panel_glob,
        "primary_panel_list": str(args.primary_panel_list) if args.primary_panel_list is not None else "",
        "external_panel_glob": args.external_panel_glob,
        "external_panel_list": str(args.external_panel_list) if args.external_panel_list is not None else "",
        "n_primary_panels": len(primary_panels),
        "n_external_panels": len(external_panels),
        "selection_train_reference_set": args.selection_train_reference_set,
        "selection_holdout_reference_set": args.selection_holdout_reference_set,
        "external_reference_set": args.external_reference_set,
        "selection_constraints": {
            "max_top1": args.max_top1,
            "max_top2": args.max_top2,
            "min_joint_delta": args.min_joint_delta,
            "min_train_hit_delta": args.min_train_hit_delta,
            "min_train_enrichment": args.min_train_enrichment,
        },
        "primary_endpoint": {
            "reference_set": args.external_reference_set,
            "min_enrichment": args.endpoint_min_enrichment,
            "max_p_value": args.endpoint_max_p,
            "min_support": args.endpoint_min_support,
        },
    }
    (args.output_dir / "protocol_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    pd.DataFrame(
        [{"panel_path": p, "panel_group": "primary"} for p in primary_panels]
        + [{"panel_path": p, "panel_group": "external_reserved"} for p in external_panels]
    ).to_csv(args.output_dir / "protocol_panel_inventory.csv", index=False)

    # 1) Generate policy selections from fixed governance setup.
    if args.reuse_policy_dir is not None:
        policy_dir = args.reuse_policy_dir
    else:
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
                str(policy_dir),
            ]
        )

    # 2) Build train/holdout split with reserved external panels held out.
    blind_cmd = [
        "python",
        "subproject_24_multireference_latent_truth_model/implementation/run_blind_holdout_protocol.py",
        "--selection-dir",
        str(policy_dir),
        "--candidate-mode",
        args.candidate_mode,
        "--min-train-overlap",
        str(args.min_train_overlap),
        "--comparison-selection",
        args.baseline_policy,
        "--output-dir",
        str(blind_dir),
    ]
    if args.primary_panel_list is not None:
        blind_cmd.extend(["--primary-panel-list", str(args.primary_panel_list)])
    else:
        blind_cmd.extend(["--primary-panel-glob", args.primary_panel_glob])
    if args.external_panel_list is not None:
        blind_cmd.extend(["--extra-panel-list", str(args.external_panel_list)])
    else:
        blind_cmd.extend(["--extra-panel-glob", args.external_panel_glob])
    _run(blind_cmd)

    # 3) Select policy using train + primary holdout only.
    _run(
        [
            "python",
            "subproject_24_multireference_latent_truth_model/implementation/optimize_constrained_promotion_policy.py",
            "--policy-summary",
            str(policy_dir / "deconcentration_policy_summary.csv"),
            "--blind-summary",
            str(blind_dir / "blind_holdout_validation_summary.csv"),
            "--baseline-policy",
            args.baseline_policy,
            "--train-reference-set",
            args.selection_train_reference_set,
            "--holdout-reference-set",
            args.selection_holdout_reference_set,
            "--max-top1",
            str(args.max_top1),
            "--max-top2",
            str(args.max_top2),
            "--min-joint-delta",
            str(args.min_joint_delta),
            "--min-train-hit-delta",
            str(args.min_train_hit_delta),
            "--min-train-enrichment",
            str(args.min_train_enrichment),
            "--output-dir",
            str(optimization_dir),
        ]
    )

    blind_summary = pd.read_csv(blind_dir / "blind_holdout_validation_summary.csv", low_memory=False)
    recommendation = pd.read_csv(optimization_dir / "constrained_policy_recommendation.csv", low_memory=False).iloc[0]
    recommended_policy = str(recommendation["policy"])
    compared = [args.baseline_policy, recommended_policy]

    endpoint_rows = blind_summary[
        (blind_summary["reference_set"] == args.external_reference_set)
        & (blind_summary["selection"].isin(compared))
    ].copy()
    if endpoint_rows.empty:
        raise ValueError(f"No endpoint rows found for reference set: {args.external_reference_set}")

    endpoint_rows["enrichment_vs_random"] = pd.to_numeric(endpoint_rows["enrichment_vs_random"], errors="coerce")
    endpoint_rows["p_value_hypergeom_ge_hits"] = pd.to_numeric(
        endpoint_rows["p_value_hypergeom_ge_hits"], errors="coerce"
    )
    endpoint_rows["n_reference_edges_in_candidate_pool"] = pd.to_numeric(
        endpoint_rows["n_reference_edges_in_candidate_pool"], errors="coerce"
    )
    endpoint_rows["n_hits"] = pd.to_numeric(endpoint_rows["n_hits"], errors="coerce")
    endpoint_rows["endpoint_pass_standard"] = endpoint_rows.apply(
        lambda r: _endpoint_pass(
            enrichment=float(r["enrichment_vs_random"]),
            p_value=float(r["p_value_hypergeom_ge_hits"]),
            min_enrichment=float(args.endpoint_min_enrichment),
            max_p=float(args.endpoint_max_p),
        ),
        axis=1,
    )
    endpoint_rows["endpoint_support_eligible"] = (
        endpoint_rows["n_reference_edges_in_candidate_pool"] >= float(args.endpoint_min_support)
    )
    endpoint_rows["endpoint_support_tier"] = endpoint_rows.apply(
        lambda r: _support_tier(
            support_edges=float(r["n_reference_edges_in_candidate_pool"]),
            endpoint_pass=bool(r["endpoint_pass_standard"]),
            min_support=float(args.endpoint_min_support),
        ),
        axis=1,
    )

    # Endpoint decision is defined on the locked recommended policy.
    rec_endpoint = endpoint_rows[endpoint_rows["selection"] == recommended_policy]
    if rec_endpoint.empty:
        raise ValueError(f"Recommended policy `{recommended_policy}` missing in endpoint rows")
    rec_endpoint_row = rec_endpoint.iloc[0]
    endpoint_pass = bool(rec_endpoint_row["endpoint_pass_standard"])
    support_eligible = bool(rec_endpoint_row["endpoint_support_eligible"])
    selected_support = float(rec_endpoint_row["n_reference_edges_in_candidate_pool"])
    selected_hits = float(rec_endpoint_row["n_hits"])
    endpoint_support_tier = str(rec_endpoint_row["endpoint_support_tier"])

    endpoint_summary = pd.DataFrame(
        [
            {
                "selected_policy": recommended_policy,
                "baseline_policy": args.baseline_policy,
                "external_reference_set": args.external_reference_set,
                "selected_enrichment": float(rec_endpoint_row["enrichment_vs_random"]),
                "selected_p_value": float(rec_endpoint_row["p_value_hypergeom_ge_hits"]),
                "selected_n_reference_edges_in_candidate_pool": selected_support,
                "selected_n_hits": selected_hits,
                "endpoint_min_enrichment": float(args.endpoint_min_enrichment),
                "endpoint_max_p_value": float(args.endpoint_max_p),
                "endpoint_min_support": float(args.endpoint_min_support),
                "endpoint_support_eligible": support_eligible,
                "endpoint_support_tier": endpoint_support_tier,
                "endpoint_pass": endpoint_pass,
            }
        ]
    )
    endpoint_rows.to_csv(args.output_dir / "preregistered_external_endpoint_comparison.csv", index=False)
    endpoint_summary.to_csv(args.output_dir / "preregistered_external_endpoint_summary.csv", index=False)

    # Auxiliary dashboard for transparency across key reference sets.
    key_refs = [
        args.selection_train_reference_set,
        args.selection_holdout_reference_set,
        args.external_reference_set,
        "blind_holdout_union",
    ]
    aux = blind_summary[
        (blind_summary["reference_set"].isin(key_refs))
        & (blind_summary["selection"].isin(compared))
    ].copy()
    aux.to_csv(args.output_dir / "preregistered_external_key_reference_metrics.csv", index=False)

    # Final run manifest with realized selection and endpoint.
    realized_manifest = {
        **manifest,
        "realized_selection": {
            "recommended_policy": recommended_policy,
            "recommended_alpha": float(recommendation["alpha"]),
            "recommendation_reason": str(recommendation.get("recommendation_reason", "")),
        },
        "realized_endpoint": endpoint_summary.iloc[0].to_dict(),
    }
    (args.output_dir / "protocol_manifest_realized.json").write_text(
        json.dumps(realized_manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    print("Pre-registered external validation protocol complete.")
    print(endpoint_summary.to_string(index=False))


if __name__ == "__main__":
    main()
