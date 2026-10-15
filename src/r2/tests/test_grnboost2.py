"""Tests of the GRNBoost2/GENIE3 stage summary helpers and stored outputs (src/r2/build_grnboost2.py).

Run: .venv/bin/python -m pytest subproject_24_multireference_latent_truth_model/src/r2/tests/test_grnboost2.py -q
(the output checks are skipped when outputs/r2/grnboost2 has not been built)
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from r2 import build_grnboost2 as G  # noqa: E402
from r2 import common as C  # noqa: E402
from r2.predicted_sets import topk_per_tf  # noqa: E402


def _links(n_tf=4, n_tg=80, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({"tf": np.repeat([f"T{i}" for i in range(n_tf)], n_tg),
                       "target": [f"G{j}" for _ in range(n_tf) for j in range(n_tg)]})
    df["importance"] = rng.integers(1, 6, size=len(df)).astype(float)    # heavy ties
    return df


def test_prespecified_constants_used():
    assert G.N_CELLS_DEFAULT == C.PRESPECIFIED["expr_n_cells"] == 3000
    assert G.RARE_FRAC == C.PRESPECIFIED["grn_detection_filter_sensitivity"] == 0.01
    assert G.variant_suffix(G.RARE_FRAC) == "_detect1pct"


def test_predicted_topk_is_the_shared_rule():
    links = _links()
    top = G.predicted_topk(links, "grnboost2", 20)
    ref = links[topk_per_tf(links.rename(columns={"importance": "grnboost2"}), "grnboost2", 20).to_numpy()]
    assert set(zip(top.tf, top.target)) == set(zip(ref.tf, ref.target))
    assert (top.groupby("tf").size() == 20).all()
    # row order does not matter
    shuf = links.sample(frac=1.0, random_state=3)
    assert set(zip(*G.predicted_topk(shuf, "grnboost2", 20)[["tf", "target"]].T.values)) == \
        set(zip(top.tf, top.target))


def test_methods_get_independent_tie_breaks():
    links = _links()
    links["importance"] = 1.0                                   # every edge tied
    a = G.predicted_topk(links, "grnboost2", 20)
    b = G.predicted_topk(links, "genie3", 20)
    j = G._jaccard_per_tf(a, b)
    assert (j < 1).all() and j.median() < 0.6                   # same seed, different column -> independent


def test_parse_time_l(tmp_path):
    p = tmp_path / "t.txt"
    p.write_text("      247.00 real      1276.73 user        33.57 sys\n"
                 "          1020502016  maximum resident set size\n")
    out = G._parse_time_l(p)
    assert out == {"time_l_real_s": 247.0, "time_l_user_s": 1276.73, "time_l_sys_s": 33.57,
                   "time_l_maxrss_largest_process_mib": 973.2}


def _variants(t):
    od = G.OUT_STAGE / t
    if not (od / "gene_detection.tsv").exists():
        pytest.skip(f"missing {od}")
    return od, [json.loads(p.read_text())["variant"] for p in sorted(od.glob("*_meta.json"))]


@pytest.mark.parametrize("t", C.TISSUES)
def test_stored_links_and_filter(t):
    od, variants = _variants(t)
    det = pd.read_csv(od / "gene_detection.tsv", sep="\t").set_index("gene")
    tfs = [l.strip() for l in open(C.OUT / "universe" / t / "tfs.txt") if l.strip()]
    for v in variants:
        links = pd.read_parquet(od / f"{v}.parquet")
        assert set(links.tf) <= set(tfs) and set(links.target) <= set(det.index)
        assert (links.tf != links.target).all() and (links.importance > 0).all()
        assert not links.duplicated(["tf", "target"]).any()
        if v.endswith("_detect1pct"):
            genes = set(links.tf) | set(links.target)
            assert (det.loc[sorted(genes), "frac_detected"] >= G.RARE_FRAC).all()


def test_genie3_filtered_run_and_summary():
    od, variants = _variants("immune")
    if "genie3_detect1pct" not in variants:
        pytest.skip("GENIE3 detect1pct not built")
    meta = json.loads((od / "genie3_detect1pct_meta.json").read_text())
    assert meta["min_detect_frac"] == 0.01 and meta["method"] == "genie3"
    summ = json.loads((G.OUT_STAGE / "summary.json").read_text())
    k = C.PRESPECIFIED["topk_per_tf"]
    assert summ["topk"] == k
    a = G.predicted_topk(pd.read_parquet(od / "genie3.parquet"), "genie3", k)
    b = G.predicted_topk(pd.read_parquet(od / "genie3_detect1pct.parquet"), "genie3_detect1pct", k)
    j = G._jaccard_per_tf(a, b)
    rec = summ["comparisons"][f"immune_genie3_jaccard_top{k}_main_vs_detect1pct"]
    assert rec["n_tfs"] == len(j) and np.isclose(rec["median"], round(float(j.median()), 3))
    assert "immune_grnboost2_detect1pct_vs_genie3_detect1pct" in summ["comparisons"]
