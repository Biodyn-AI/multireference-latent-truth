#!/usr/bin/env python3
"""Build a LNCS-style camera-ready package and supplement from frozen bundle artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Dict, List

import pandas as pd


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bundle-root",
        type=Path,
        default=Path("subproject_24_multireference_latent_truth_model/paper/submission_bundle"),
    )
    parser.add_argument(
        "--bundle-prefix",
        type=str,
        default="subproject24_submission_bundle_v6_",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("subproject_24_multireference_latent_truth_model/paper/camera_ready_lnncs_v1"),
    )
    return parser.parse_args()


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _latest_bundle_dir(bundle_root: Path, prefix: str) -> Path:
    candidates = sorted([p for p in bundle_root.glob(f"{prefix}*") if p.is_dir()])
    if not candidates:
        raise FileNotFoundError(f"No bundle directories found under {bundle_root} with prefix {prefix}")
    return candidates[-1]


def _copy(src: Path, dst: Path) -> None:
    if not src.exists():
        raise FileNotFoundError(f"Missing source file: {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def _render_camera_ready_tex(stats: Dict[str, object]) -> str:
    bundle_name_tex = str(stats["bundle_name"]).replace("_", r"\_")
    return rf"""\documentclass[runningheads]{{llncs}}
\usepackage[T1]{{fontenc}}
\usepackage{{graphicx}}
\usepackage{{booktabs}}
\usepackage{{amsmath}}
\usepackage{{xcolor}}
\usepackage{{hyperref}}

\begin{{document}}

\title{{Latent Truth Without a Gold Standard: Multi-Reference Credibility Modeling for Mechanistic GRN Claims}}
\titlerunning{{Latent Truth for Mechanistic GRN Claims}}
\author{{Ihor Kendiukhov}}
\authorrunning{{I. Kendiukhov}}
\institute{{University of Tuebingen, Germany\\\email{{kenduhov.ig@gmail.com}}}}
\maketitle

\begin{{abstract}}
We present a low-compute, high-scope research program for mechanistic GRN claim governance under imperfect references. Across Phases I--XXVII, the framework integrates latent-truth modeling, robustness stress tests, dual-universe governance, pre-registered external validation, multi-cohort replication, and hierarchical Bayesian endpoint confidence layers. The current profile is checklist-complete (10/10 gates), with multi-cohort endpoint pass rate {stats["pass_rate"]:.3f} ({int(stats["n_pass"])}/{int(stats["n_cohorts"])}). Covariate-aware multilevel Bayesian endpoint modeling confirms strong support-driven confidence separation (support-eligible mean joint pass probability {stats["multilevel_support_eligible"]:.3f} vs support-limited {stats["multilevel_support_limited"]:.3f}) while preserving conservative interpretation in sparse-support cohorts. A frozen checksum bundle and deterministic packaging pipeline support submission-grade reproducibility.
\keywords{{Mechanistic Interpretability \and Gene Regulatory Networks \and Bayesian Hierarchical Modeling \and Evaluation Robustness}}
\end{{abstract}}

\section{{Introduction}}
Benchmarking in mechanistic GRN inference is often treated as a single-reference truth problem, despite incompleteness and context bias across curated and perturbation-derived sources. This produces unstable rankings and overconfident biological claims. Our program addresses this with explicit uncertainty propagation and policy-level claim governance.

\section{{Methods Overview}}
The unified method stack includes:
\begin{{enumerate}}
\item Multi-source latent truth modeling and dependence-aware robustness stress.
\item Context-vs-full calibration with perturbation-backed references.
\item Dual-universe claim-tier governance and deconcentration policy controls.
\item Pre-registered external protocol with frozen manifest and held-out endpoint.
\item Multi-cohort replication with support-aware endpoint interpretation.
\item Power-aware, full-Bayes, adaptive, and multilevel Bayesian endpoint tiers.
\end{{enumerate}}

\section{{Core Results}}
\subsection{{Submission-readiness and replication status}}
Checklist readiness is complete (10/10 gates). Pre-registered multi-cohort replication passes in 5/6 cohorts, with failure concentrated in sparse-support external geometry.

\subsection{{Support geometry remains the dominant reliability axis}}
Support-stratified confidence is stable across Bayesian layers. In the current multilevel model, support-eligible cohorts remain near-certain while support-limited cohorts stay substantially less certain.

\subsection{{Multilevel adaptive Bayesian extension improves uncertainty allocation}}
The Phase XXVII multilevel model adds assay-family partial pooling and nonlinear support covariates. Posterior support-magnitude effects are consistently positive, and sparse-support cohorts shift toward uncertainty-aware interpretation instead of extreme overconfident tails.

\begin{{figure}}[t]
\centering
\includegraphics[width=0.95\textwidth]{{figures/multilevel_adaptive_support_vs_joint_probability.png}}
\caption{{Multilevel adaptive endpoint joint-confidence profile across cohorts.}}
\end{{figure}}

\begin{{figure}}[t]
\centering
\includegraphics[width=0.95\textwidth]{{figures/sensitivity_cohort_joint_probability_profiles.png}}
\caption{{Prior-sensitivity profiles for cohort-level joint endpoint probabilities.}}
\end{{figure}}

\section{{Reproducibility and Artifact Freeze}}
The camera-ready package is derived from frozen bundle \texttt{{{bundle_name_tex}}} with {int(stats["bundle_n_files"])} files and checksum manifesting. The package includes manuscript, key diagnostics, and supplementary artifact traceability.

\section{{Limitations and Next Validation Steps}}
Current limits are cohort count and external cohort novelty in this checkout. The next validation stage is ingestion of truly new cohorts under frozen manifests and re-execution of the same protocol without retuning.

\begin{{thebibliography}}{{8}}
\bibitem{{cui2024}} Cui, H., Wang, C., Maan, H., et al.: scGPT: Towards a foundation model for single-cell multi-omics. Nature Methods (2024)
\bibitem{{han2018}} Han, H., Cho, J.-W., Lee, S., et al.: TRRUST v2. Nucleic Acids Research (2018)
\bibitem{{garcia2019}} Garcia-Alonso, L., Holland, C.H., Ibrahim, M.M., et al.: DoRothEA benchmarking. Genome Research (2019)
\bibitem{{subproject24}} Kendiukhov, I.: Unified Multi-Reference Latent Truth Model artifacts (Phases I--XXVII), 2026.
\end{{thebibliography}}

\end{{document}}
"""


def _render_supplement_tex(stats: Dict[str, object]) -> str:
    bundle_name_tex = str(stats["bundle_name"]).replace("_", r"\_")
    readiness_tex = str(stats["readiness_profile"]).replace("_", r"\_")
    external_tex = str(stats["external_profile"]).replace("_", r"\_")
    return rf"""\documentclass[11pt]{{article}}
\usepackage[margin=1in]{{geometry}}
\usepackage[T1]{{fontenc}}
\usepackage{{graphicx}}
\usepackage{{booktabs}}
\usepackage{{longtable}}
\usepackage{{hyperref}}

\title{{Supplementary Material: Latent Truth Without a Gold Standard}}
\author{{Ihor Kendiukhov}}
\date{{}}
\begin{{document}}
\maketitle

\section*{{S1. Frozen Artifact Context}}
This supplement is built from frozen bundle \texttt{{{bundle_name_tex}}} with {int(stats["bundle_n_files"])} files.
Bundle summary fields:
\begin{{itemize}}
\item readiness profile: \texttt{{{readiness_tex}}}
\item external protocol profile: \texttt{{{external_tex}}}
\item total bundle bytes: {int(stats["bundle_total_bytes"])}
\end{{itemize}}

\section*{{S2. Phase XXVII Additions}}
\begin{{itemize}}
\item strict new-cohort ingestion workflow with frozen manifesting;
\item multilevel adaptive Bayesian endpoint model with assay-family random effects;
\item prior-sensitivity and posterior-predictive stress suite.
\end{{itemize}}

\section*{{S3. Key Diagnostic Figures}}
\begin{{figure}}[h]
\centering
\includegraphics[width=0.86\textwidth]{{figures/multilevel_adaptive_coefficient_forest.png}}
\caption{{Multilevel adaptive coefficient posterior intervals.}}
\end{{figure}}

\begin{{figure}}[h]
\centering
\includegraphics[width=0.86\textwidth]{{figures/multilevel_adaptive_support_vs_joint_probability.png}}
\caption{{Support vs multilevel joint pass probability.}}
\end{{figure}}

\begin{{figure}}[h]
\centering
\includegraphics[width=0.86\textwidth]{{figures/sensitivity_cohort_joint_probability_profiles.png}}
\caption{{Prior-sensitivity cohort joint-probability profiles.}}
\end{{figure}}

\section*{{S4. Reproducibility Notes}}
All commands used in the latest extension phase are captured in:
\begin{{itemize}}
\item \url{{subproject_24_multireference_latent_truth_model/handoff/run_log.md}}
\item \url{{subproject_24_multireference_latent_truth_model/implementation/README.md}}
\end{{itemize}}

\end{{document}}
"""


def main() -> None:
    args = _parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = args.output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    bundle_dir = _latest_bundle_dir(args.bundle_root, args.bundle_prefix)
    bundle_summary = json.loads((bundle_dir / "bundle_summary.json").read_text(encoding="utf-8"))
    bundle_manifest = pd.read_csv(bundle_dir / "bundle_manifest.csv", low_memory=False)

    # Pull stats from current outputs.
    multicohort = pd.read_csv(
        "subproject_24_multireference_latent_truth_model/implementation/outputs/"
        "top_tier_readiness_v6/multicohort_preregistered_replication_v1/multicohort_replication_summary.csv",
        low_memory=False,
    ).iloc[0]
    multilevel_support = pd.read_csv(
        "subproject_24_multireference_latent_truth_model/implementation/outputs/"
        "top_tier_readiness_v7/multilevel_adaptive_bayes_endpoint_tiering_v1/multilevel_adaptive_support_summary.csv",
        low_memory=False,
    )
    eligible = multilevel_support[multilevel_support["observed_support_eligible"] == True]
    limited = multilevel_support[multilevel_support["observed_support_eligible"] == False]

    stats = {
        "bundle_name": str(bundle_summary["bundle_name"]),
        "bundle_n_files": int(bundle_summary["n_files"]),
        "bundle_total_bytes": int(bundle_summary["total_size_bytes"]),
        "readiness_profile": str(bundle_summary["readiness_profile"]),
        "external_profile": str(bundle_summary["external_protocol_profile"]),
        "n_cohorts": int(multicohort["n_cohorts_completed"]),
        "n_pass": int(multicohort["n_endpoint_pass"]),
        "pass_rate": float(multicohort["endpoint_pass_rate"]),
        "multilevel_support_eligible": float(eligible["mean_predictive_prob_joint_pass"].iloc[0]) if not eligible.empty else float("nan"),
        "multilevel_support_limited": float(limited["mean_predictive_prob_joint_pass"].iloc[0]) if not limited.empty else float("nan"),
    }

    # Copy selected figures from frozen/current outputs.
    figure_sources = {
        "figure_robustness_progression.png": Path(
            "subproject_24_multireference_latent_truth_model/implementation/outputs/unified_paper_figures/figure_robustness_progression.png"
        ),
        "figure_calibration_context_vs_full.png": Path(
            "subproject_24_multireference_latent_truth_model/implementation/outputs/unified_paper_figures/figure_calibration_context_vs_full.png"
        ),
        "figure_dual_governance_tiers.png": Path(
            "subproject_24_multireference_latent_truth_model/implementation/outputs/unified_paper_figures/figure_dual_governance_tiers.png"
        ),
        "multilevel_adaptive_support_vs_joint_probability.png": Path(
            "subproject_24_multireference_latent_truth_model/implementation/outputs/top_tier_readiness_v7/multilevel_adaptive_bayes_endpoint_tiering_v1/multilevel_adaptive_support_vs_joint_probability.png"
        ),
        "multilevel_adaptive_coefficient_forest.png": Path(
            "subproject_24_multireference_latent_truth_model/implementation/outputs/top_tier_readiness_v7/multilevel_adaptive_bayes_endpoint_tiering_v1/multilevel_adaptive_coefficient_forest.png"
        ),
        "sensitivity_cohort_joint_probability_profiles.png": Path(
            "subproject_24_multireference_latent_truth_model/implementation/outputs/top_tier_readiness_v7/multilevel_prior_sensitivity_v1/sensitivity_cohort_joint_probability_profiles.png"
        ),
    }
    copied_rows: List[Dict[str, object]] = []
    for name, src in figure_sources.items():
        dst = figures_dir / name
        _copy(src, dst)
        copied_rows.append(
            {
                "source_path": src.as_posix(),
                "camera_ready_path": dst.relative_to(args.output_dir).as_posix(),
                "size_bytes": int(dst.stat().st_size),
                "sha256": _sha256(dst),
            }
        )

    camera_tex = args.output_dir / "camera_ready_lnncs.tex"
    supp_tex = args.output_dir / "supplementary_appendix.tex"
    camera_tex.write_text(_render_camera_ready_tex(stats), encoding="utf-8")
    supp_tex.write_text(_render_supplement_tex(stats), encoding="utf-8")

    # Also copy the frozen bundle summary + manifest snapshots for traceability.
    _copy(bundle_dir / "bundle_summary.json", args.output_dir / "frozen_bundle_summary.json")
    _copy(bundle_dir / "bundle_manifest.csv", args.output_dir / "frozen_bundle_manifest.csv")

    pd.DataFrame(copied_rows).sort_values("camera_ready_path").to_csv(
        args.output_dir / "camera_ready_figure_manifest.csv",
        index=False,
    )

    package_summary = {
        "camera_ready_template": "llncs",
        "bundle_source_dir": bundle_dir.as_posix(),
        "bundle_name": stats["bundle_name"],
        "bundle_n_files": stats["bundle_n_files"],
        "multicohort_pass_rate": stats["pass_rate"],
        "multilevel_support_eligible_mean_joint_prob": stats["multilevel_support_eligible"],
        "multilevel_support_limited_mean_joint_prob": stats["multilevel_support_limited"],
    }
    (args.output_dir / "camera_ready_package_summary.json").write_text(
        json.dumps(package_summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    print("Camera-ready LNCS package created.")
    print(f"Output directory: {args.output_dir}")
    print(f"Bundle source: {bundle_dir}")
    print(f"Copied figures: {len(copied_rows)}")
    print(f"Frozen bundle entries: {len(bundle_manifest)}")


if __name__ == "__main__":
    main()
