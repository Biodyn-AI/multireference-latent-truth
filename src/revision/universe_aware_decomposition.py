#!/usr/bin/env python3
"""Universe-aware decomposition of the full-vs-context calibration 'uplift'.

The raw precision@k ratio between the context-matched and full universes conflates
two effects: (i) a base-rate (denominator) effect, because the context universe has a
much higher density of evaluable perturbation references, and (ii) a genuine ranking
signal, measured denominator-free as enrichment = precision / base_rate (lift over
random). We decompose

    P@k(context) / P@k(full) = [base(context)/base(full)] * [lift@k(context)/lift@k(full)]

and report the enrichment-vs-rank-depth profile in each universe.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
FULL = ROOT / "outputs/full_v2_best_fullfit/perturbation_multipanel/perturbation_multipanel_claim_tiers_edges.csv"
CTX = ROOT / "outputs/context_v2_best_fullfit/perturbation_multipanel/perturbation_multipanel_claim_tiers_edges.csv"
OUTDIR = ROOT / "outputs/revision"


def _ranked(df: pd.DataFrame) -> np.ndarray:
    d = df.sort_values("calibration_score", ascending=False)
    return d["is_reference"].to_numpy(dtype=float)


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    full = pd.read_csv(FULL)
    ctx = pd.read_csv(CTX)

    yf, yc = _ranked(full), _ranked(ctx)
    nf, nc = len(yf), len(yc)
    bf, bc = yf.mean(), yc.mean()

    # Decomposition table at shared k values (capped to each universe).
    ks = [100, 200, 500, 1000, 2000, 5000, 7536]
    rows = []
    for k in ks:
        kf, kc = min(k, nf), min(k, nc)
        pf = yf[:kf].sum() / kf
        pc = yc[:kc].sum() / kc
        lift_f = pf / bf
        lift_c = pc / bc
        rows.append({
            "k": k,
            "P_at_k_full": pf,
            "P_at_k_context": pc,
            "raw_ratio_context_over_full": pc / pf if pf > 0 else np.nan,
            "lift_full": lift_f,
            "lift_context": lift_c,
            "base_rate_ratio": bc / bf,
            "signal_ratio_context_over_full": lift_c / lift_f if lift_f > 0 else np.nan,
        })
    dec = pd.DataFrame(rows)
    dec.to_csv(OUTDIR / "universe_aware_decomposition.csv", index=False)

    # Enrichment vs rank depth (fraction of universe) for the SNR-vs-depth question.
    fracs = np.linspace(0.01, 1.0, 40)
    depth_rows = []
    for fr in fracs:
        kf = max(1, int(round(fr * nf)))
        kc = max(1, int(round(fr * nc)))
        depth_rows.append({
            "depth_fraction": fr,
            "enrichment_full": (yf[:kf].mean()) / bf,
            "enrichment_context": (yc[:kc].mean()) / bc,
            "recall_full": yf[:kf].sum() / yf.sum(),
            "recall_context": yc[:kc].sum() / yc.sum(),
        })
    depth = pd.DataFrame(depth_rows)
    depth.to_csv(OUTDIR / "enrichment_vs_depth.csv", index=False)

    summary = pd.DataFrame([{
        "n_full": nf, "n_context": nc,
        "ref_full": int(yf.sum()), "ref_context": int(yc.sum()),
        "base_rate_full": bf, "base_rate_context": bc,
        "base_rate_ratio_context_over_full": bc / bf,
        "best_lift_full": depth["enrichment_full"].max(),
        "best_lift_context": depth["enrichment_context"].max(),
    }])
    summary.to_csv(OUTDIR / "universe_aware_summary.csv", index=False)

    print("== base rates ==")
    print(summary.to_string(index=False))
    print("\n== decomposition (raw ratio = base_rate_ratio x signal_ratio) ==")
    print(dec.to_string(index=False))


if __name__ == "__main__":
    main()
