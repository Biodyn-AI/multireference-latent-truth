#!/usr/bin/env python3
"""Run the dependence-aware next-step bundle for Subproject 24.

Bundle includes:
1) dependence-aware latent-truth fit,
2) posterior ranking evaluation with prefixed outputs,
3) baseline-vs-dependence variant comparison.
"""

from __future__ import annotations

import subprocess
from pathlib import Path


def _run(cmd: list[str], cwd: Path) -> None:
    print("[run]", " ".join(cmd))
    subprocess.run(cmd, cwd=str(cwd), check=True)


def main() -> None:
    repo = Path(".")
    impl = "subproject_24_multireference_latent_truth_model/implementation"

    _run(["python", f"{impl}/fit_dependent_ltm.py"], cwd=repo)
    _run(
        [
            "python",
            f"{impl}/evaluate_posterior_rankings.py",
            "--posterior",
            "subproject_24_multireference_latent_truth_model/implementation/outputs/ltm_dep_edge_posteriors.csv",
            "--posterior-col",
            "posterior",
            "--output-prefix",
            "dep_",
        ],
        cwd=repo,
    )
    _run(["python", f"{impl}/compare_ltm_variants.py"], cwd=repo)

    print("Dependence-aware bundle complete.")


if __name__ == "__main__":
    main()
