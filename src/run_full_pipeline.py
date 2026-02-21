#!/usr/bin/env python3
"""Run the full Subproject 24 baseline pipeline end-to-end.

This orchestrator keeps commands explicit and reproducible while remaining laptop-feasible.
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def _run(cmd: list[str], cwd: Path) -> None:
    print("[run]", " ".join(cmd))
    subprocess.run(cmd, cwd=str(cwd), check=True)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path("."),
        help="Repository root.",
    )
    parser.add_argument(
        "--max-edges",
        type=int,
        default=0,
        help="Optional edge cap passed to EM/Bayes fits (0 = full).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    repo = args.repo_root

    impl = "subproject_24_multireference_latent_truth_model/implementation"

    _run(["python", f"{impl}/build_evidence_matrix.py"], cwd=repo)

    em_cmd = ["python", f"{impl}/fit_em_ltm.py"]
    if args.max_edges > 0:
        em_cmd += ["--max-edges", str(args.max_edges)]
    _run(em_cmd, cwd=repo)

    bayes_cmd = ["python", f"{impl}/fit_bayes_ltm.py"]
    if args.max_edges > 0:
        bayes_cmd += ["--max-edges", str(args.max_edges)]
    _run(bayes_cmd, cwd=repo)

    _run(["python", f"{impl}/run_synthetic_stress.py"], cwd=repo)
    _run(["python", f"{impl}/evaluate_posterior_rankings.py"], cwd=repo)

    print("Pipeline complete.")


if __name__ == "__main__":
    main()
