#!/usr/bin/env python3
"""Compare multiple latent-truth model variants on posteriors and rankings."""

from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("subproject_24_multireference_latent_truth_model/implementation/outputs"),
    )
    parser.add_argument(
        "--variants",
        type=str,
        default="em,dep,mf",
        help="Comma-separated variant IDs. Supported: em, dep, mf, family, family_hb",
    )
    parser.add_argument("--top-frac", type=float, default=0.01)
    return parser.parse_args()


def _parse_list(s: str) -> List[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def _posterior_file(output_dir: Path, variant: str) -> Path:
    mapping = {
        "em": output_dir / "ltm_em_edge_posteriors.csv",
        "dep": output_dir / "ltm_dep_edge_posteriors.csv",
        "mf": output_dir / "ltm_mf_edge_posteriors.csv",
        "family": output_dir / "ltm_family_edge_posteriors.csv",
        "family_hb": output_dir / "ltm_family_hb_edge_posteriors.csv",
    }
    if variant not in mapping:
        raise ValueError(f"Unsupported variant '{variant}'.")
    return mapping[variant]


def _method_metric_file(output_dir: Path, variant: str) -> Path:
    mapping = {
        "em": output_dir / "posterior_method_metrics.csv",
        "dep": output_dir / "dep_posterior_method_metrics.csv",
        "mf": output_dir / "mf_posterior_method_metrics.csv",
        "family": output_dir / "family_posterior_method_metrics.csv",
        "family_hb": output_dir / "family_hb_posterior_method_metrics.csv",
    }
    if variant not in mapping:
        raise ValueError(f"Unsupported variant '{variant}'.")
    return mapping[variant]


def _reference_dis_file(output_dir: Path, variant: str) -> Path:
    mapping = {
        "em": output_dir / "ranking_reversal_vs_posterior.csv",
        "dep": output_dir / "dep_ranking_reversal_vs_posterior.csv",
        "mf": output_dir / "mf_ranking_reversal_vs_posterior.csv",
        "family": output_dir / "family_ranking_reversal_vs_posterior.csv",
        "family_hb": output_dir / "family_hb_ranking_reversal_vs_posterior.csv",
    }
    if variant not in mapping:
        raise ValueError(f"Unsupported variant '{variant}'.")
    return mapping[variant]


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


def main() -> None:
    args = _parse_args()
    variants = _parse_list(args.variants)

    post: Dict[str, pd.DataFrame] = {}
    method: Dict[str, pd.DataFrame] = {}
    refdis: Dict[str, pd.DataFrame] = {}

    for v in variants:
        post[v] = pd.read_csv(_posterior_file(args.output_dir, v))[["source", "target", "posterior"]]
        method[v] = pd.read_csv(_method_metric_file(args.output_dir, v))[["method", "weighted_f1"]]
        refdis[v] = pd.read_csv(_reference_dis_file(args.output_dir, v))

    pair_rows = []

    for a, b in combinations(variants, 2):
        pa = post[a].rename(columns={"posterior": f"posterior_{a}"})
        pb = post[b].rename(columns={"posterior": f"posterior_{b}"})
        m = pa.merge(pb, on=["source", "target"], how="inner")

        pearson = float(np.corrcoef(m[f"posterior_{a}"], m[f"posterior_{b}"])[0, 1])
        spearman = float(
            pd.Series(m[f"posterior_{a}"]).corr(pd.Series(m[f"posterior_{b}"]), method="spearman")
        )

        top_k = max(1, int(args.top_frac * len(m)))
        sa = set(zip(m.nlargest(top_k, f"posterior_{a}")["source"], m.nlargest(top_k, f"posterior_{a}")["target"]))
        sb = set(zip(m.nlargest(top_k, f"posterior_{b}")["source"], m.nlargest(top_k, f"posterior_{b}")["target"]))
        top_jacc = len(sa & sb) / float(max(len(sa | sb), 1))

        oa = _pairwise_orders(method[a], "weighted_f1")
        ob = _pairwise_orders(method[b], "weighted_f1")
        r, diff, valid = _disagreement(oa, ob)

        ra = refdis[a]["pairwise_disagreement_vs_posterior"].to_numpy(dtype=float)
        rb = refdis[b]["pairwise_disagreement_vs_posterior"].to_numpy(dtype=float)
        mean_ref_a = float(np.nanmean(ra))
        mean_ref_b = float(np.nanmean(rb))

        pair_rows.append(
            {
                "variant_a": a,
                "variant_b": b,
                "n_edges": len(m),
                "posterior_pearson": pearson,
                "posterior_spearman": spearman,
                "topfrac": args.top_frac,
                "topfrac_jaccard": top_jacc,
                "method_pairwise_disagreement": r,
                "method_pairwise_n_diff": diff,
                "method_pairwise_n_valid": valid,
                "mean_ref_disagreement_a": mean_ref_a,
                "mean_ref_disagreement_b": mean_ref_b,
                "delta_ref_disagreement_b_minus_a": mean_ref_b - mean_ref_a,
            }
        )

    grid = pd.DataFrame(pair_rows)
    grid.to_csv(args.output_dir / "variant_grid_comparison.csv", index=False)

    # Also generate a compact leaderboard table across variants.
    method_rows = []
    for v in variants:
        m = method[v].copy()
        m["variant"] = v
        m["rank"] = m["weighted_f1"].rank(ascending=False, method="min")
        method_rows.append(m)
    leaderboard = pd.concat(method_rows, ignore_index=True)
    pivot = leaderboard.pivot_table(index="method", columns="variant", values=["weighted_f1", "rank"])
    pivot.columns = [f"{a}_{b}" for a, b in pivot.columns]
    pivot = pivot.reset_index()
    pivot.to_csv(args.output_dir / "variant_method_leaderboard.csv", index=False)

    print("Variant-grid comparison complete:")
    print(grid.to_string(index=False))


if __name__ == "__main__":
    main()
