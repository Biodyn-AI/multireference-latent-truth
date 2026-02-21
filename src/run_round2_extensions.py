#!/usr/bin/env python3
"""Run round-2 Subproject 24 extensions in sequence.

Includes:
1) tissue-hierarchical fit,
2) score-aware ranking evaluation,
3) claim-tier certification,
4) source-dropout ablation (subset, configurable).
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
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--ablation-max-edges", type=int, default=100000)
    parser.add_argument("--ablation-max-iter", type=int, default=140)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    repo = args.repo_root
    impl = "subproject_24_multireference_latent_truth_model/implementation"

    _run(["python", f"{impl}/fit_tissue_hierarchical_ltm.py"], cwd=repo)
    _run(["python", f"{impl}/evaluate_score_aware_rankings.py"], cwd=repo)
    _run(["python", f"{impl}/certify_claim_tiers.py"], cwd=repo)
    _run(
        [
            "python",
            f"{impl}/run_source_dropout_ablation.py",
            "--max-edges",
            str(args.ablation_max_edges),
            "--max-iter",
            str(args.ablation_max_iter),
        ],
        cwd=repo,
    )

    print("Round-2 extensions complete.")


if __name__ == "__main__":
    main()
