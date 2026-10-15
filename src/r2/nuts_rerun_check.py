"""First-pass vs re-run check of the dominant-mode Family-HB NUTS fits (fit_fixes round).

The fit_fixes round (verifier f1, finding M5 and minor 1) froze the NUTS settings at
4 chains x 1000 warm-up + 1000 draws, target acceptance 0.95 (``fit_primary.NUTS_SETTINGS``)
and re-ran every ``family_hb_nuts`` fit from the polished MAP optimum.  The first-pass fits
that were replaced are kept in

    outputs/r2/fits/_superseded_firstpass/<universe>__<variant>__family_hb_nuts/

(kidney, lung and global; the global first pass was 4 x 500/500, chosen by the load-dependent
wall-clock pilot).  This script quantifies how much the stored posterior summary moved between
the two runs, so that the effect of the re-run on the credibilities and on the Strong
(Bayesian FDR <= 0.05) and Moderate (<= 0.2) sets is on record:

* sampler settings and diagnostics of both runs (R-hat, bulk / tail ESS, divergences, over all
  sites and over the likelihood-identified sites ``fit_primary.IDENTIFIED_SITES``);
* ``fit_primary.compare_creds(first pass, re-run)`` over observed edges (edge-weighted Spearman,
  absolute differences, BFDR-set and top-k agreement; suffix ``_a`` = first pass, ``_b`` = re-run);
* churn of the Strong set (edges leaving / entering), the largest credibility change inside the
  union of the two Strong sets, numbers of observed edges whose posterior-mean credibility moved
  by more than 0.01 / 0.05, the class weight w_T, the expected number of true edges and the
  mean 95% CrI width of both runs.

The comparison is a numerical-procedure check: it reads no held-out data.  It was designed in
the fit_fixes round, after the project's first held-out outcomes existed, so every row carries
``analysis_status = post_hoc``.  A universe without a kept first-pass fit is skipped (immune:
its first-pass dominant-mode fit already used 4 x 1000/1000 and was overwritten by the
2026-10-08 re-run before copies were kept; the first-pass tables in
``_superseded_firstpass/tables`` hold its summary numbers).

Output: ``outputs/r2/fits/_superseded_firstpass/nuts_rerun_vs_firstpass.csv`` (one row per
universe).  Deterministic (pure function of the stored fits).

Command line::

    .venv/bin/python subproject_24_multireference_latent_truth_model/src/r2/nuts_rerun_check.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from r2 import fit_primary as F  # noqa: E402

FIT_ID = "family_hb_nuts"
SUPERSEDED = "_superseded_firstpass"
OUT_NAME = "nuts_rerun_vs_firstpass.csv"
MOVE_THRESHOLDS = (0.01, 0.05)


def superseded_dir(universe: str, variant: str = F.VARIANT, root: Path = F.FITS_DIR) -> Path:
    return Path(root) / SUPERSEDED / f"{universe}__{variant}__{FIT_ID}"


def read_nuts_dir(d: Path) -> dict | None:
    """Stored NUTS fit of directory ``d`` (None if it has no ``meta.json``)."""
    d = Path(d)
    if not (d / "meta.json").exists():
        return None
    out = dict(meta=json.loads((d / "meta.json").read_text()), dir=d,
               diagnostics=pd.read_csv(d / "diagnostics.csv"), params=dict(np.load(d / "params.npz")))
    for k in ("cred", "cred_lo", "cred_hi", "cred_sd"):
        out[k] = np.load(d / f"{k}.npy")
    return out


def _run_summary(fit: dict, tag: str) -> dict:
    m, c = fit["meta"], fit["meta"]["convergence"]
    dg = fit["diagnostics"]
    ident = dg[dg.site.isin(F.IDENTIFIED_SITES)]
    return {
        f"num_chains_{tag}": m["num_chains"], f"num_warmup_{tag}": m["num_warmup"],
        f"num_samples_{tag}": m["num_samples"], f"target_accept_{tag}": m.get("target_accept"),
        f"runtime_min_{tag}": m["runtime_s"] / 60.0,
        f"r_hat_max_{tag}": c["r_hat_max"], f"worst_site_{tag}": c["worst_site"],
        f"ess_bulk_min_{tag}": c["ess_bulk_min"], f"ess_tail_min_{tag}": c["ess_tail_min"],
        f"r_hat_max_identified_{tag}": float(ident.r_hat_max.max()),
        f"ess_bulk_min_identified_{tag}": float(ident.ess_bulk_min.min()),
        f"ess_tail_min_identified_{tag}": float(ident.ess_tail_min.min()),
        f"n_divergent_{tag}": c["n_divergent"],
        f"n_chains_at_best_map_optimum_{tag}": c.get("n_chains_at_best_map_optimum"),
    }


def compare_runs(first: dict, rerun: dict, data) -> dict:
    """Row of the comparison table for one universe (``first`` / ``rerun`` from ``read_nuts_dir``)."""
    c1, c2 = first["cred"], rerun["cred"]
    if c1.shape != c2.shape or c1.shape != (data.P,):
        raise ValueError(f"credibility vectors do not match the evidence patterns: {c1.shape} {c2.shape} {data.P}")
    obs = F.observed_patterns(data)
    n = data.n
    n_obs = np.where(obs, n, 0.0)
    row = dict(_run_summary(first, "a"), **_run_summary(rerun, "b"))
    row.update(F.compare_creds(c1, c2, data))
    d = np.abs(c1 - c2)
    for thr in MOVE_THRESHOLDS:
        row[f"n_edges_abs_diff_gt_{thr:g}"] = float(n[obs & (d > thr)].sum())
    for q, name in zip(F.BFDR_LEVELS, ("strong", "moderate")):
        a, b = F.bfdr_set(c1, n_obs, q) & obs, F.bfdr_set(c2, n_obs, q) & obs
        row[f"n_{name}_left"] = float(n[a & ~b].sum())          # in the first-pass set only
        row[f"n_{name}_entered"] = float(n[b & ~a].sum())       # in the re-run set only
        u = a | b
        row[f"max_abs_diff_{name}_union"] = float(d[u].max()) if u.any() else np.nan
        sym = a ^ b
        # credibility (re-run) of the edges that changed sides: all sit at the set boundary
        row[f"cred_b_min_{name}_changed"] = float(c2[sym].min()) if sym.any() else np.nan
        row[f"cred_b_max_{name}_changed"] = float(c2[sym].max()) if sym.any() else np.nan
    for tag, f in (("a", first), ("b", rerun)):
        width = f["cred_hi"] - f["cred_lo"]
        row[f"cri_width_mean_w_{tag}"] = float(np.average(width[obs], weights=n[obs]))
        row[f"w_T_mean_{tag}"] = float(f["params"]["w"][0])
        row[f"expected_true_observed_{tag}"] = float(np.sum(n[obs] * f["cred"][obs]))
        row[f"expected_true_all_{tag}"] = float(np.sum(n * f["cred"]))
        row[f"n_edges_mean_gt_0.5_{tag}"] = float(n[obs & (f["cred"] > 0.5)].sum())
    # the re-run mean inside the first-pass 95% CrI and vice versa (edge-weighted, observed edges)
    in_a = (c2 >= first["cred_lo"] - 1e-12) & (c2 <= first["cred_hi"] + 1e-12)
    in_b = (c1 >= rerun["cred_lo"] - 1e-12) & (c1 <= rerun["cred_hi"] + 1e-12)
    row["frac_edges_b_mean_inside_a_cri"] = float(n[obs & in_a].sum() / n[obs].sum())
    row["frac_edges_a_mean_inside_b_cri"] = float(n[obs & in_b].sum() / n[obs].sum())
    return row


def build(universes=F.UNIVERSES, variant: str = F.VARIANT, root: Path = F.FITS_DIR,
          out: Path | None = None) -> pd.DataFrame:
    rows = []
    for u in universes:
        first = read_nuts_dir(superseded_dir(u, variant, root))
        rerun = read_nuts_dir(F.fit_dir(u, FIT_ID, variant, root))
        if first is None or rerun is None:
            print(f"[rerun-check] {u}: no kept first-pass fit or no current fit; skipped", flush=True)
            continue
        row = dict(universe=u, fit_id=FIT_ID, **compare_runs(first, rerun, F.load_data(u, variant)))
        row["analysis_status"] = "post_hoc"
        rows.append(row)
        print(f"[rerun-check] {u}: Spearman {row['spearman_w']:.6f} max|diff| {row['max_abs_diff']:.4f} "
              f"Strong {row['n_bfdr0.05_a']:.0f} -> {row['n_bfdr0.05_b']:.0f} (Jaccard {row['jaccard_bfdr0.05']:.4f})",
              flush=True)
    df = pd.DataFrame(rows)
    out = Path(out) if out is not None else Path(root) / SUPERSEDED / OUT_NAME
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False, float_format="%.10g")
    print(f"[rerun-check] wrote {out} ({len(df)} rows)", flush=True)
    return df


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--universes", nargs="+", default=list(F.UNIVERSES), choices=list(F.UNIVERSES))
    ap.add_argument("--variant", default=F.VARIANT)
    ap.add_argument("--root", type=Path, default=F.FITS_DIR)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args(argv)
    build(a.universes, a.variant, a.root, a.out)


if __name__ == "__main__":
    main()
