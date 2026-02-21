#!/usr/bin/env python3
"""Run governance-aware deconcentration policy sweep on promoted edges.

This script keeps the promotion budget fixed and explores source-penalized
selection policies to reduce source concentration while tracking precision
proxies and score retention.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(
            "subproject_24_multireference_latent_truth_model/implementation/outputs/dual_universe_governance_v1/dual_universe_claim_tiers_edges.csv"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "subproject_24_multireference_latent_truth_model/implementation/outputs/deconcentration_policy_sweep_v1"
        ),
    )
    parser.add_argument(
        "--alphas",
        type=str,
        default="0.0,0.05,0.1,0.25,0.5,1.0",
        help="Comma-separated source-concentration penalties. 0.0 reproduces baseline.",
    )
    parser.add_argument(
        "--budget",
        type=int,
        default=500,
        help="Selection budget. 0 means use all promotion-eligible edges.",
    )
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=0,
        help="Optional cap on input promoted candidates for smoke tests; 0 uses all.",
    )
    parser.add_argument(
        "--top1-threshold",
        type=float,
        default=0.23,
        help="Heuristic top-1-source fraction threshold for recommendation.",
    )
    parser.add_argument(
        "--candidate-mode",
        type=str,
        default="promotion_allowed",
        choices=["promotion_allowed", "up_to_conditional", "all"],
        help=(
            "Candidate pool filter: `promotion_allowed` reproduces prior phases; "
            "`up_to_conditional` includes DUAL_TIER1/2/3; `all` uses full edge table."
        ),
    )
    return parser.parse_args()


def _safe_div(a: float, b: float) -> float:
    return float(a / b) if b != 0 else float("nan")


def _source_stats(sources: Iterable[str]) -> Dict[str, float]:
    counts = Counter(sources)
    n = float(sum(counts.values()))
    if n <= 0:
        return {
            "n_unique_sources": 0.0,
            "top1_source_fraction": float("nan"),
            "top2_source_fraction": float("nan"),
            "source_entropy": float("nan"),
            "source_hhi": float("nan"),
            "effective_source_count": float("nan"),
        }

    freqs = np.array(sorted(counts.values(), reverse=True), dtype=float) / n
    top1 = freqs[0] if len(freqs) >= 1 else float("nan")
    top2 = np.sum(freqs[:2]) if len(freqs) >= 2 else top1
    entropy = float(-np.sum(freqs * np.log(freqs + 1e-12)))
    hhi = float(np.sum(freqs**2))
    effective = float(1.0 / hhi) if hhi > 0 else float("nan")
    return {
        "n_unique_sources": float(len(counts)),
        "top1_source_fraction": float(top1),
        "top2_source_fraction": float(top2),
        "source_entropy": entropy,
        "source_hhi": hhi,
        "effective_source_count": effective,
    }


def _build_policy_name(alpha: float) -> str:
    if abs(alpha) < 1e-12:
        return "baseline_alpha0"
    txt = str(alpha).replace(".", "p")
    return f"penalized_alpha{txt}"


def _prepare_promoted(df: pd.DataFrame, candidate_mode: str) -> pd.DataFrame:
    if candidate_mode == "promotion_allowed":
        candidates = df[df["promotion_allowed"] == True].copy()  # noqa: E712
    elif candidate_mode == "up_to_conditional":
        need = "dual_universe_claim_tier"
        if need not in df.columns:
            raise ValueError(f"Input missing required column for {candidate_mode}: {need}")
        candidates = df[
            df["dual_universe_claim_tier"].isin(
                ["DUAL_TIER1_STRONG", "DUAL_TIER2_MODERATE", "DUAL_TIER3_CONDITIONAL"]
            )
        ].copy()
    elif candidate_mode == "all":
        candidates = df.copy()
    else:
        raise ValueError(f"Unsupported candidate_mode: {candidate_mode}")

    if candidates.empty:
        raise ValueError("No candidate edges found for the selected candidate mode.")

    candidates["posterior_full"] = candidates["posterior_full"].fillna(0.0)
    candidates["posterior_context"] = candidates["posterior_context"].fillna(0.0)
    candidates["joint_score"] = candidates[["posterior_full", "posterior_context"]].min(axis=1)
    candidates = candidates.sort_values(
        ["joint_score", "posterior_context", "posterior_full", "source", "target"],
        ascending=[False, False, False, True, True],
    ).reset_index(drop=True)
    return candidates


def _select_with_source_penalty(promoted: pd.DataFrame, alpha: float, budget: int) -> pd.DataFrame:
    remaining = promoted.copy()
    source_counts: Dict[str, int] = defaultdict(int)
    chosen_rows: List[pd.Series] = []

    for _ in range(min(budget, len(remaining))):
        penalties = remaining["source"].map(lambda s: 1.0 + alpha * source_counts[s]).astype(float)
        adjusted = remaining["joint_score"].astype(float) / penalties
        best_pos = int(adjusted.values.argmax())
        best_row = remaining.iloc[best_pos]
        chosen_rows.append(best_row)
        source_counts[str(best_row["source"])] += 1
        remaining = remaining.drop(remaining.index[best_pos]).reset_index(drop=True)

    if not chosen_rows:
        return promoted.head(0).copy()

    out = pd.DataFrame(chosen_rows).reset_index(drop=True)
    out["selection_alpha"] = alpha
    out["selection_policy"] = _build_policy_name(alpha)
    out = out.sort_values(
        ["joint_score", "posterior_context", "posterior_full", "source", "target"],
        ascending=[False, False, False, True, True],
    ).reset_index(drop=True)
    return out


def _policy_metrics(
    name: str,
    alpha: float,
    sel: pd.DataFrame,
    budget: int,
    candidate_mode: str,
    n_candidates: int,
) -> Dict[str, float | str]:
    n = int(len(sel))
    source_stats = _source_stats(sel["source"].astype(str).tolist())

    row: Dict[str, float | str] = {
        "policy": name,
        "alpha": float(alpha),
        "budget": int(budget),
        "candidate_mode": candidate_mode,
        "n_candidates": int(n_candidates),
        "n_selected": n,
        "n_unique_targets": int(sel["target"].nunique()),
        "joint_score_mean": float(sel["joint_score"].mean()) if n else float("nan"),
        "joint_score_min": float(sel["joint_score"].min()) if n else float("nan"),
        "posterior_full_mean": float(sel["posterior_full"].mean()) if n else float("nan"),
        "posterior_context_mean": float(sel["posterior_context"].mean()) if n else float("nan"),
        "reference_full_fraction": float(sel["is_reference_full"].fillna(0).mean()) if n else float("nan"),
        "reference_context_fraction": float(sel["is_reference_context"].fillna(0).mean()) if n else float("nan"),
    }
    row.update(source_stats)
    row["deconcentration_objective"] = float(
        row["reference_context_fraction"] - 0.1 * row["top1_source_fraction"]
    )
    return row


def _source_distribution(policy: str, sel: pd.DataFrame) -> pd.DataFrame:
    dist = (
        sel["source"].astype(str).value_counts().rename_axis("source").reset_index(name="n_edges")
    )
    dist["policy"] = policy
    dist["fraction"] = dist["n_edges"] / max(len(sel), 1)
    return dist[["policy", "source", "n_edges", "fraction"]]


def _recommend(summary: pd.DataFrame, top1_threshold: float) -> pd.DataFrame:
    feasible = summary[summary["top1_source_fraction"] <= top1_threshold].copy()
    if feasible.empty:
        feasible = summary.copy()
    feasible = feasible.sort_values(
        ["reference_context_fraction", "deconcentration_objective", "joint_score_mean", "top1_source_fraction"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)
    best = feasible.iloc[[0]].copy()
    best["recommendation_reason"] = (
        "max reference_context_fraction within deconcentration constraint"
        if not summary[summary["top1_source_fraction"] <= top1_threshold].empty
        else "no policy met top1 threshold; selected best precision-proxy policy"
    )
    return best


def _plot_tradeoff(summary: pd.DataFrame, recommended_policy: str, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    x = summary["top1_source_fraction"].to_numpy(dtype=float)
    y = summary["reference_context_fraction"].to_numpy(dtype=float)
    s = 30 + 8 * summary["n_unique_sources"].to_numpy(dtype=float)

    colors = ["#2a9d8f" if p == recommended_policy else "#35618f" for p in summary["policy"]]
    ax.scatter(x, y, s=s, c=colors, alpha=0.9, edgecolors="white", linewidths=0.7)

    for _, r in summary.iterrows():
        ax.annotate(str(r["policy"]), (r["top1_source_fraction"], r["reference_context_fraction"]), fontsize=8)

    ax.set_xlabel("Top-1 source fraction (lower is better deconcentration)")
    ax.set_ylabel("Reference context fraction (higher is better precision proxy)")
    ax.set_title("Deconcentration Policy Tradeoff at Fixed Promotion Budget")
    ax.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = _parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input, low_memory=False)
    need_cols = {
        "source",
        "target",
        "posterior_full",
        "posterior_context",
        "is_reference_full",
        "is_reference_context",
        "promotion_allowed",
    }
    missing = sorted(need_cols - set(df.columns))
    if missing:
        raise ValueError(f"Input missing required columns: {missing}")

    promoted = _prepare_promoted(df=df, candidate_mode=args.candidate_mode)
    if args.max_candidates > 0:
        promoted = promoted.head(args.max_candidates).copy()

    budget = int(args.budget) if args.budget > 0 else int(len(promoted))
    alphas = [float(x.strip()) for x in args.alphas.split(",") if x.strip()]
    if not alphas:
        raise ValueError("No alphas provided")

    summary_rows: List[Dict[str, float | str]] = []
    dist_frames: List[pd.DataFrame] = []

    for alpha in alphas:
        policy = _build_policy_name(alpha)
        selected = _select_with_source_penalty(promoted=promoted, alpha=alpha, budget=budget)
        selected.to_csv(args.output_dir / f"{policy}_selected_edges.csv", index=False)
        summary_rows.append(
            _policy_metrics(
                policy,
                alpha,
                selected,
                budget=budget,
                candidate_mode=args.candidate_mode,
                n_candidates=len(promoted),
            )
        )
        dist_frames.append(_source_distribution(policy, selected))

    summary = pd.DataFrame(summary_rows).sort_values("alpha", ascending=True).reset_index(drop=True)
    baseline = summary[summary["alpha"] == 0.0].iloc[0]
    for col in [
        "top1_source_fraction",
        "top2_source_fraction",
        "n_unique_sources",
        "n_unique_targets",
        "reference_context_fraction",
        "joint_score_mean",
        "source_hhi",
    ]:
        summary[f"delta_vs_baseline_{col}"] = summary[col] - float(baseline[col])

    summary.to_csv(args.output_dir / "deconcentration_policy_summary.csv", index=False)
    pd.concat(dist_frames, ignore_index=True).to_csv(
        args.output_dir / "deconcentration_policy_source_distribution.csv", index=False
    )

    recommendation = _recommend(summary, top1_threshold=float(args.top1_threshold))
    recommendation.to_csv(args.output_dir / "deconcentration_policy_recommendation.csv", index=False)

    _plot_tradeoff(
        summary=summary,
        recommended_policy=str(recommendation.iloc[0]["policy"]),
        output=args.output_dir / "deconcentration_policy_tradeoff.png",
    )

    print("Deconcentration sweep complete.")
    print(f"Promoted candidates: {len(promoted)}")
    print(f"Candidate mode: {args.candidate_mode}")
    print(f"Budget: {budget}")
    print(summary.to_string(index=False))
    print("Recommendation:")
    print(recommendation.to_string(index=False))


if __name__ == "__main__":
    main()
