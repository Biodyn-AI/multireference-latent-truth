#!/usr/bin/env python3
"""Automated deconcentration operating point via a diversity-precision objective.

Removes reliance on a hand-picked promotion budget. We sweep a 2-D grid of
(budget, source-penalty alpha), and for each operating point measure:
  - source diversity: Pielou evenness J = H / ln(n_selected), H = Shannon entropy of
    the promoted TF distribution (in [0,1], 1 = perfectly even);
  - validation precision: enrichment of the promoted set over the ENCODE-evaluable
    candidate base rate (genuinely external), plus the perturbation reference fraction.
We then min-max-normalise both criteria across the grid and select the operating
point maximising their harmonic mean (a diversity-precision F-score). The knee of
the diversity-vs-precision frontier is reported as a second, objective-free rule.
"""
from __future__ import annotations
from collections import defaultdict, Counter
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import hypergeom

ROOT = Path(__file__).resolve().parents[2]
ANN = ROOT / "outputs/revision/governance_edges_annotated.csv"
OUTDIR = ROOT / "outputs/revision"


def select(pool: pd.DataFrame, alpha: float, budget: int) -> pd.DataFrame:
    remaining = pool.reset_index(drop=True).copy()
    counts: dict = defaultdict(int)
    chosen = []
    js = remaining["joint_score"].to_numpy(dtype=float)
    src = remaining["source"].to_numpy()
    taken = np.zeros(len(remaining), dtype=bool)
    for _ in range(min(budget, len(remaining))):
        pen = np.array([1.0 + alpha * counts[s] for s in src])
        adj = np.where(taken, -np.inf, js / pen)
        best = int(adj.argmax())
        chosen.append(best)
        counts[src[best]] += 1
        taken[best] = True
    return remaining.iloc[chosen]


def diversity_precision(sel: pd.DataFrame, enc_base, N_enc, K_enc):
    n = len(sel)
    counts = Counter(sel["source"].astype(str))
    freqs = np.array(list(counts.values()), dtype=float) / n
    H = float(-np.sum(freqs * np.log(freqs + 1e-12)))
    even = H / np.log(n) if n > 1 else 0.0
    pe = sel[sel["tf_in_encode"]]
    hits = int(pe["in_encode"].sum())
    enc_prec = hits / len(pe) if len(pe) else np.nan
    enc_enr = enc_prec / enc_base if len(pe) else np.nan
    enc_p = float(hypergeom.sf(hits - 1, N_enc, K_enc, len(pe))) if len(pe) else np.nan
    return {
        "n_selected": n, "n_unique_tfs": len(counts),
        "shannon_H": H, "pielou_evenness": even,
        "top1_tf_frac": max(freqs), "effective_tfs": float(1.0 / np.sum(freqs**2)),
        "encode_enrichment": enc_enr, "encode_hypergeom_p": enc_p,
        "pert_ref_frac_full": float(sel["is_reference_full"].fillna(0).mean()),
    }


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    ann = pd.read_csv(ANN, low_memory=False)
    pool = ann[ann["promotion_allowed"] == True].copy()  # noqa: E712
    pool["posterior_full"] = pool["posterior_full"].fillna(0.0)
    pool["posterior_context"] = pool["posterior_context"].fillna(0.0)
    pool["joint_score"] = pool[["posterior_full", "posterior_context"]].min(axis=1)
    pool = pool.sort_values(["joint_score", "posterior_context", "posterior_full", "source", "target"],
                            ascending=[False, False, False, True, True]).reset_index(drop=True)

    N_enc = int(ann["tf_in_encode"].sum())
    K_enc = int(ann.loc[ann["tf_in_encode"], "in_encode"].sum())
    enc_base = K_enc / N_enc

    budgets = list(range(100, len(pool) + 1, 50)) + [len(pool)]
    budgets = sorted(set(b for b in budgets if b <= len(pool)))
    alphas = [0.0, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0]

    rows = []
    for b in budgets:
        for a in alphas:
            sel = select(pool, a, b)
            m = diversity_precision(sel, enc_base, N_enc, K_enc)
            m.update({"budget": b, "alpha": a})
            rows.append(m)
    grid = pd.DataFrame(rows)

    # Normalise the two criteria across the grid and take their harmonic mean.
    def mm(x):
        x = x.astype(float)
        lo, hi = np.nanmin(x), np.nanmax(x)
        return (x - lo) / (hi - lo) if hi > lo else np.zeros_like(x)
    grid["div_norm"] = mm(grid["pielou_evenness"])
    grid["prec_norm"] = mm(grid["encode_enrichment"])
    grid["objective_hmean"] = 2 * grid["div_norm"] * grid["prec_norm"] / (
        grid["div_norm"] + grid["prec_norm"] + 1e-12)
    grid.to_csv(OUTDIR / "deconcentration_autotune_grid.csv", index=False)

    best = grid.sort_values("objective_hmean", ascending=False).iloc[0]

    # Knee of the (evenness, enrichment) frontier: Pareto-front then max distance to
    # the chord between its extreme points.
    front = grid.sort_values("pielou_evenness")
    pareto = []
    best_prec = -np.inf
    for _, r in front[::-1].iterrows():  # from high evenness to low
        if r["encode_enrichment"] > best_prec:
            pareto.append(r); best_prec = r["encode_enrichment"]
    pf = pd.DataFrame(pareto).sort_values("pielou_evenness").reset_index(drop=True)
    if len(pf) >= 3:
        x = mm(pf["pielou_evenness"]); y = mm(pf["encode_enrichment"])
        x0, y0, x1, y1 = x.iloc[0], y.iloc[0], x.iloc[-1], y.iloc[-1]
        d = np.abs((y1 - y0) * x - (x1 - x0) * y + x1 * y0 - y1 * x0) / (np.hypot(y1 - y0, x1 - x0) + 1e-12)
        knee = pf.iloc[int(np.asarray(d).argmax())]
    else:
        knee = best

    sel_summary = pd.DataFrame([
        {"rule": "objective_hmean", **best[["budget", "alpha", "n_unique_tfs", "pielou_evenness",
         "top1_tf_frac", "encode_enrichment", "encode_hypergeom_p", "pert_ref_frac_full"]].to_dict()},
        {"rule": "frontier_knee", **knee[["budget", "alpha", "n_unique_tfs", "pielou_evenness",
         "top1_tf_frac", "encode_enrichment", "encode_hypergeom_p", "pert_ref_frac_full"]].to_dict()},
        {"rule": "fixed_842_baseline", **grid[(grid.budget == len(pool)) & (grid.alpha == 0.0)].iloc[0][
         ["budget", "alpha", "n_unique_tfs", "pielou_evenness", "top1_tf_frac", "encode_enrichment",
          "encode_hypergeom_p", "pert_ref_frac_full"]].to_dict()},
    ])
    sel_summary.to_csv(OUTDIR / "deconcentration_autotune_selected.csv", index=False)
    print("Grid points:", len(grid))
    print(sel_summary.to_string(index=False))


if __name__ == "__main__":
    main()
