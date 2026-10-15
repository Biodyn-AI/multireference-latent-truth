"""Provenance record: one-off relabelling of the first-pass random-init Family-HB NUTS runs.

On 2026-10-08 the immune and global random-init NUTS runs (4 chains x 1000 warm-up + 1000 draws,
target acceptance 0.95, ``lcm.fit_nuts`` defaults, seed ``PRESPECIFIED['random_seed']`` -- i.e. the
settings now frozen in ``fit_primary.NUTS_SETTINGS['family_hb_randinit_nuts']``) were started by
a version of ``fit_primary.py`` that wrote them to ``family_hb_nuts`` and had no ``init`` record.
This script (previously in a scratch directory, verifier finding f1 M5) moved them to
``family_hb_randinit_nuts``, added the ``init`` record and assigned every chain to its nearest MAP
optimum from the stored thinned draws (250 per chain).  No draw was changed.  The kidney and lung
random-init runs were written by ``fit_primary.py nuts --init random`` directly.

The random-init runs were not re-run in the fit_fixes round (machine load average 70-180 made
the projected global run > 6 h); ``fit_primary.py nuts --init random`` (frozen settings) is the
command that regenerates them.  Re-running this script refreshes the chain labels against the
current MAP optima (``optima.csv`` / ``cred_restarts.npy`` of ``family_hb``); it is idempotent.

Usage::

    .venv/bin/python subproject_24_multireference_latent_truth_model/src/r2/patch_randinit_layout.py immune
    .venv/bin/python subproject_24_multireference_latent_truth_model/src/r2/patch_randinit_layout.py global
"""
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from r2 import fit_primary as F  # noqa: E402
from r2 import lcm  # noqa: E402


def relabel(u: str) -> pd.DataFrame:
    base = F.FITS_DIR / u / F.VARIANT
    src, dst = base / "family_hb_nuts", base / "family_hb_randinit_nuts"
    if src.exists() and not dst.exists():
        m = json.loads((src / "meta.json").read_text())
        assert "init" not in m or m["init"].get("init") == "random", m.get("init")
        shutil.move(str(src), str(dst))
    data = F.load_data(u)
    m = json.loads((dst / "meta.json").read_text())
    dr = np.load(dst / "draws.npz")
    ch = pd.read_csv(dst / "chains.csv")
    near, dist = [], []
    for c in ch["chain"]:
        sel = np.nonzero(dr["chain"] == c)[0]
        cm = np.mean([lcm.credibility_at({"w": dr["w"][i], "theta": dr["theta"][i]}, data) for i in sel], axis=0)
        k, d = F.nearest_optimum(cm, base / "family_hb", data)
        near.append(k)
        dist.append(d)
    ch["nearest_map_optimum"] = near
    ch["dist_nearest_map_optimum"] = dist
    ch.to_csv(dst / "chains.csv", index=False)
    m["fit_id"] = "family_hb_randinit_nuts"
    m.setdefault("init", dict(init="random", description="numpyro init_to_uniform(radius=2), lcm.fit_nuts default"))
    m["init"]["chain_labels"] = "nearest MAP optimum from the stored thinned draws (patch_randinit_layout.py)"
    m["convergence"]["n_chains_at_best_map_optimum"] = int(sum(k == 0 for k in near))
    m["convergence"]["chain_optima"] = near
    m["provenance_note"] = ("first-pass run (2026-10-08) with settings equal to fit_primary.NUTS_SETTINGS; "
                            "relabelled by src/r2/patch_randinit_layout.py")
    (dst / "meta.json").write_text(json.dumps(m, indent=2, default=str))
    return ch


if __name__ == "__main__":
    for u in sys.argv[1:]:
        print(u, relabel(u).to_string(), flush=True)
