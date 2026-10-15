"""Unit tests of the prespecified top-k-per-TF rule in ``r2.predicted_sets``.

Run: .venv/bin/python -m pytest subproject_24_multireference_latent_truth_model/src/r2/tests/test_predicted_sets.py -q
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from r2 import common as C  # noqa: E402
from r2.predicted_sets import tie_break_keys, topk_per_tf  # noqa: E402


def _edges(scores: dict) -> pd.DataFrame:
    rows = [(tf, f"G{j}", v) for tf, vals in scores.items() for j, v in enumerate(vals)]
    return pd.DataFrame(rows, columns=["tf", "target", "s"])


def _brute(df: pd.DataFrame, k: int, seed: int, col: str = "s") -> set:
    """Reference implementation: per TF, sort by (score desc, tie key desc) and take k."""
    out = set()
    for tf, g in df[df[col].notna()].groupby("tf"):
        keys = tie_break_keys(g.tf.to_numpy(), g.target.to_numpy(), col, seed)
        o = sorted(zip(g[col].astype(float), keys, g.target), key=lambda x: (-x[0], -int(x[1]), x[2]))
        out |= {(tf, t) for _, _, t in o[:k]}
    return out


def _sel(df, mask) -> set:
    return set(zip(df.tf[mask], df.target[mask]))


def test_no_ties_takes_top_k():
    df = _edges({"A": [5, 1, 4, 2, 3], "B": [0.1, 0.9, 0.5]})
    m = topk_per_tf(df, "s", 2)
    assert m.dtype == bool and m.index.equals(df.index)
    assert _sel(df, m) == {("A", "G0"), ("A", "G2"), ("B", "G1"), ("B", "G2")}


def test_nan_not_predicted_and_short_tfs_keep_all_finite():
    df = _edges({"A": [np.nan, 3, np.nan, 1], "B": [np.nan, np.nan], "C": [2, 1, 0, -1]})
    m = topk_per_tf(df, "s", 3)
    assert _sel(df, m) == {("A", "G1"), ("A", "G3"), ("C", "G0"), ("C", "G1"), ("C", "G2")}
    assert not m[df.s.isna()].any()


def test_infinite_values_are_ordinary_scores():
    df = _edges({"A": [np.inf, 1, -np.inf, 2]})
    assert _sel(df, topk_per_tf(df, "s", 2)) == {("A", "G0"), ("A", "G3")}
    assert _sel(df, topk_per_tf(df, "s", 4)) == {("A", f"G{j}") for j in range(4)}


def test_tie_at_kth_position_exact_size_and_membership():
    # values above the tie: 9, 8; tie block of five 5's; k = 4 -> two of the five
    df = _edges({"A": [9, 5, 5, 8, 5, 5, 1, 5]})
    m = topk_per_tf(df, "s", 4, seed=123)
    sel = _sel(df, m)
    assert len(sel) == 4
    assert {("A", "G0"), ("A", "G3")} <= sel
    assert all(df.set_index("target").loc[t, "s"] == 5 for _, t in sel - {("A", "G0"), ("A", "G3")})
    assert sel == _brute(df, 4, 123)


def test_tie_block_fully_inside_top_k_is_kept():
    df = _edges({"A": [7, 7, 7, 1, 0]})
    assert _sel(df, topk_per_tf(df, "s", 3)) == {("A", "G0"), ("A", "G1"), ("A", "G2")}
    assert _sel(df, topk_per_tf(df, "s", 4)) == {("A", "G0"), ("A", "G1"), ("A", "G2"), ("A", "G3")}


def test_default_seed_is_prespecified():
    df = _edges({"A": [1.0] * 30})
    a = topk_per_tf(df, "s", 7)
    b = topk_per_tf(df, "s", 7, seed=C.PRESPECIFIED["topk_tie_seed"])
    assert a.equals(b)
    assert _sel(df, a) == _brute(df, 7, C.PRESPECIFIED["topk_tie_seed"])


def test_deterministic_and_seed_dependent():
    df = _edges({"A": [1.0] * 40})
    sels = {frozenset(_sel(df, topk_per_tf(df, "s", 10, seed=s))) for s in range(20)}
    assert frozenset(_sel(df, topk_per_tf(df, "s", 10, seed=3))) in sels
    assert len(sels) > 15                      # different seeds give different draws
    assert topk_per_tf(df, "s", 10, seed=3).equals(topk_per_tf(df, "s", 10, seed=3))


def test_uniform_over_tie_block():
    # 2 strictly-above edges + tie block of 10, k = 5 -> 3 of 10 chosen; P = 0.3 each
    vals = [3.0, 2.0] + [1.0] * 10
    df = _edges({"A": vals})
    n = 4000
    hits = np.zeros(len(vals))
    for s in range(n):
        hits += topk_per_tf(df, "s", 5, seed=s).to_numpy()
    assert hits[0] == n and hits[1] == n
    freq = hits[2:] / n
    # binomial sd = sqrt(.3*.7/4000) ~ 0.0072; allow ~4.5 sd
    assert np.all(np.abs(freq - 0.3) < 0.033), freq
    assert hits.sum() == 5 * n


def test_row_order_nan_rows_and_other_tfs_do_not_matter():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"tf": np.repeat([f"T{i}" for i in range(6)], 80),
                       "target": [f"G{j}" for _ in range(6) for j in range(80)]})
    df["s"] = rng.integers(0, 6, size=len(df)).astype(np.float32)   # heavy ties
    df.loc[rng.random(len(df)) < 0.2, "s"] = np.nan
    ref = _sel(df, topk_per_tf(df, "s", 10))
    shuf = df.sample(frac=1.0, random_state=1).reset_index(drop=True)
    assert _sel(shuf, topk_per_tf(shuf, "s", 10)) == ref
    fin = df.dropna(subset=["s"])
    assert _sel(fin, topk_per_tf(fin, "s", 10)) == ref
    one = df[df.tf == "T2"]
    assert _sel(one, topk_per_tf(one, "s", 10)) == {e for e in ref if e[0] == "T2"}
    assert ref == _brute(df, 10, C.PRESPECIFIED["topk_tie_seed"])
    per_tf = pd.Series([e[0] for e in ref]).value_counts()
    assert (per_tf == 10).all() and len(per_tf) == 6


def test_nested_in_k():
    rng = np.random.default_rng(1)
    df = pd.DataFrame({"tf": np.repeat(["A", "B", "C"], 200),
                       "target": [f"G{j}" for _ in range(3) for j in range(200)],
                       "s": rng.integers(0, 15, size=600).astype(float)})
    m20, m50, m100 = (topk_per_tf(df, "s", k) for k in (20, 50, 100))
    assert not (m20 & ~m50).any() and not (m50 & ~m100).any()
    assert m20.sum() == 60 and m50.sum() == 150 and m100.sum() == 300


def test_tie_breaks_independent_across_columns():
    df = _edges({"A": [1.0] * 200})
    df["s2"] = df["s"]
    a, b = topk_per_tf(df, "s", 50), topk_per_tf(df, "s2", 50)
    overlap = (a & b).sum()
    assert overlap < 50                         # same scores, different column -> independent draws
    assert 2 <= overlap <= 25                   # hypergeometric mean 12.5


def test_float32_exact_ties_and_index_alignment():
    df = _edges({"A": np.float32([0.1, 0.1, 0.1, 0.2])})
    df["s"] = df["s"].astype(np.float32)
    df.index = [10, 3, 7, 99]
    m = topk_per_tf(df, "s", 2)
    assert list(m.index) == [10, 3, 7, 99]
    assert m.loc[99] and m.sum() == 2


def test_input_validation():
    df = _edges({"A": [1, 2]})
    with pytest.raises(ValueError):
        topk_per_tf(df, "s", 0)
    with pytest.raises(KeyError):
        topk_per_tf(df, "missing", 1)
    with pytest.raises(ValueError):
        topk_per_tf(pd.concat([df, df]), "s", 1)
    empty = _edges({"A": [np.nan, np.nan]})
    assert not topk_per_tf(empty, "s", 1).any()


# ---------------------------------------------------------------------------
# optional co-observation eligibility filter (PRESPECIFIED fm_min_coobs_sensitivity)
# ---------------------------------------------------------------------------


def _with_coobs(seed=2, n_tf=5, n_tg=60):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({"tf": np.repeat([f"T{i}" for i in range(n_tf)], n_tg),
                       "target": [f"G{j}" for _ in range(n_tf) for j in range(n_tg)]})
    df["s"] = rng.integers(0, 8, size=len(df)).astype(np.float32)          # heavy ties
    df.loc[rng.random(len(df)) < 0.1, "s"] = np.nan
    df["n"] = rng.integers(0, 12, size=len(df)).astype(np.int32)
    return df


def test_min_coobs_prespecified_constant():
    v = C.PRESPECIFIED["fm_min_coobs_sensitivity"]
    assert isinstance(v, int) and v >= 2


def test_min_coobs_equals_masking_low_counts_as_nan():
    df = _with_coobs()
    mc = C.PRESPECIFIED["fm_min_coobs_sensitivity"]
    m = topk_per_tf(df, "s", 10, min_coobs=mc, coobs="n")
    ref = df.copy()
    ref.loc[ref.n < mc, "s"] = np.nan
    assert m.equals(topk_per_tf(ref, "s", 10))
    assert not m[df.n < mc].any()                          # ineligible edges never predicted
    assert _sel(df, m) == _brute(ref, 10, C.PRESPECIFIED["topk_tie_seed"])
    # same as running the plain rule on the eligible rows only (tie keys do not move)
    elig = df[df.n >= mc]
    assert _sel(df, m) == _sel(elig, topk_per_tf(elig, "s", 10))


def test_min_coobs_contains_eligible_part_of_unfiltered_set_and_keeps_short_tfs():
    df = _with_coobs(seed=5)
    mc = 5
    m0 = topk_per_tf(df, "s", 10)
    m5 = topk_per_tf(df, "s", 10, min_coobs=mc, coobs="n")
    assert not (m0 & (df.n >= mc) & ~m5).any()
    # a TF with fewer than k eligible finite edges keeps all of them
    df2 = _edges({"A": [5, 4, 3, 2, 1, 0]})
    df2["n"] = [1, 9, 0, 7, 4, 5]
    assert _sel(df2, topk_per_tf(df2, "s", 4, min_coobs=5, coobs="n")) == {("A", "G1"), ("A", "G3"), ("A", "G5")}


def test_min_coobs_accepts_series_array_and_treats_missing_counts_as_ineligible():
    df = _with_coobs(seed=7)
    df.index = np.arange(len(df))[::-1] * 3                # non-default index
    a = topk_per_tf(df, "s", 8, min_coobs=5, coobs="n")
    b = topk_per_tf(df, "s", 8, min_coobs=5, coobs=df["n"])
    c = topk_per_tf(df, "s", 8, min_coobs=5, coobs=df["n"].to_numpy())
    assert a.equals(b) and a.equals(c) and a.index.equals(df.index)
    nf = df["n"].astype(float)
    nf.iloc[: len(df) // 2] = np.nan
    d = topk_per_tf(df, "s", 8, min_coobs=5, coobs=nf)
    assert not d.iloc[: len(df) // 2].any()


def test_min_coobs_input_validation():
    df = _with_coobs()
    with pytest.raises(ValueError):
        topk_per_tf(df, "s", 5, min_coobs=5)                # count missing
    with pytest.raises(ValueError):
        topk_per_tf(df, "s", 5, coobs="n")                  # threshold missing
    with pytest.raises(KeyError):
        topk_per_tf(df, "s", 5, min_coobs=5, coobs="missing")
    with pytest.raises(ValueError):
        topk_per_tf(df, "s", 5, min_coobs=5, coobs=np.ones(3))
    with pytest.raises(ValueError):
        topk_per_tf(df, "s", 5, min_coobs=5, coobs=df["n"].reset_index(drop=True).iloc[::-1])
    with pytest.raises(ValueError):
        topk_per_tf(df, "s", 5, min_coobs=np.nan, coobs="n")


def test_fm_coobs_column():
    from r2.predicted_sets import fm_coobs_column
    assert fm_coobs_column("scgpt_attn") == "scgpt_coobs"
    assert fm_coobs_column("scgpt_attn_T_norm") == "scgpt_coobs"
    assert fm_coobs_column("geneformer_attn_uniform") == "geneformer_coobs"
    assert fm_coobs_column("pearson") is None and fm_coobs_column("random") is None
