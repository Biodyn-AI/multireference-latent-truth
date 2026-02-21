#!/usr/bin/env python3
"""Build coverage-aware edge-by-source evidence matrices.

This script is the first executable step of Subproject 24.
It ingests heterogeneous edge resources listed in source_catalog.json,
normalizes symbols, and emits:
- canonical edge universe,
- binary evidence matrix with coverage-aware missingness,
- score matrix for sources that expose continuous scores,
- source-level coverage/positivity summary,
- expanded catalog with source metadata.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import pandas as pd

from ltm_utils import (
    SourceSpec,
    build_coverage_aware_matrix,
    build_edge_universe,
    load_source_catalog,
    load_source_edges,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--catalog",
        type=Path,
        default=Path("subproject_24_multireference_latent_truth_model/implementation/source_catalog.json"),
        help="Path to source catalog JSON.",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path("."),
        help="Repository root used to resolve catalog-relative paths.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("subproject_24_multireference_latent_truth_model/implementation/outputs"),
        help="Directory for generated evidence artifacts.",
    )
    parser.add_argument(
        "--write-long",
        action="store_true",
        help="Also write long-format evidence table (large file).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    specs: List[SourceSpec] = load_source_catalog(args.catalog, args.repo_root)

    source_tables: Dict[str, pd.DataFrame] = {}
    catalog_rows = []

    for spec in specs:
        table = load_source_edges(spec)
        source_tables[spec.source_id] = table

        catalog_rows.append(
            {
                "source_id": spec.source_id,
                "kind": spec.kind,
                "loader": spec.loader,
                "path": str(spec.path),
                "exists": int(spec.path.exists()),
                "n_edges_loaded": int(len(table)),
                "score_col": spec.score_col if spec.score_col else "",
                "confidence_filter": ",".join(spec.confidence) if spec.confidence else "",
            }
        )

    catalog_df = pd.DataFrame(catalog_rows).sort_values("source_id").reset_index(drop=True)
    catalog_df.to_csv(output_dir / "source_catalog_expanded.csv", index=False)

    universe = build_edge_universe(list(source_tables.values()))
    universe.to_csv(output_dir / "edge_universe.csv", index=False)

    binary, scores, source_summary = build_coverage_aware_matrix(universe, source_tables)

    binary.to_csv(output_dir / "evidence_matrix_wide.csv", index=False)
    scores.to_csv(output_dir / "score_matrix_wide.csv", index=False)
    source_summary.to_csv(output_dir / "source_summary.csv", index=False)

    # Keep a compact metadata file that downstream scripts can join quickly.
    source_meta = catalog_df.merge(source_summary, on="source_id", how="left")
    source_meta.to_csv(output_dir / "source_metadata.csv", index=False)

    if args.write_long:
        long_df = binary.melt(
            id_vars=["source", "target"],
            var_name="source_id",
            value_name="label",
        )
        long_df = long_df.dropna(subset=["label"]).reset_index(drop=True)
        long_df.to_csv(output_dir / "evidence_matrix_long.csv", index=False)

    # Provide a minimal execution summary for quick human sanity-checking.
    summary = {
        "n_sources": len(specs),
        "n_edges_universe": int(len(universe)),
        "n_reference_sources": int((source_meta["kind"] == "reference").sum()),
        "n_method_sources": int((source_meta["kind"] == "method").sum()),
        "n_mechanistic_sources": int((source_meta["kind"] == "mechanistic").sum()),
    }
    pd.DataFrame([summary]).to_csv(output_dir / "build_summary.csv", index=False)

    print("Built evidence matrix artifacts:")
    for name in [
        "build_summary.csv",
        "edge_universe.csv",
        "evidence_matrix_wide.csv",
        "score_matrix_wide.csv",
        "source_summary.csv",
        "source_metadata.csv",
        "source_catalog_expanded.csv",
    ]:
        print(f"- {output_dir / name}")


if __name__ == "__main__":
    main()
