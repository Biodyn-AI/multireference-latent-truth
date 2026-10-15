"""Prespecified rule for a method's *predicted* edge set: top-k targets per TF.

Rule (``common.PRESPECIFIED``: ``topk_per_tf``, ``topk_tie_break``,
``topk_tie_seed``), fixed before any held-out evaluation:

* Scores are ranked within each TF, higher = more confident.
* A NaN score means "not predicted": NaN edges are never in the predicted set
  (and rank below every finite score).  ``+inf``/``-inf`` are ordinary values.
* The predicted set of a TF is its k highest-scoring edges.  Every edge whose
  score is strictly above the k-th largest value is included; if the k-th
  position falls inside a block of exactly tied scores, the remaining places are
  filled by a uniformly random subset of that tie block, drawn with the fixed
  seed.
* A TF with fewer than k non-NaN scores keeps all of them.

Implementation of the random tie-break.  Each edge in a tie block that has to be
split receives a pseudo-random 64-bit key ``blake2b(seed, score_col, tf, target)``
and the tie block is filled in decreasing key order.  For distinct edges the
keys behave as i.i.d. uniform draws, so the chosen subset is uniform over all
subsets of the required size.  Because a key depends only on the edge, the
column name and the seed, the result is

* invariant to row order and to dropping NaN rows or other TFs;
* nested in k (top-20 is contained in top-50, which is contained in top-100);
* independent across score columns (two methods tied on the same edges do
  not receive correlated tie-breaks, which would inflate their overlap).

Equivalently, the predicted set is the first k edges of the total order
(score descending, key descending).

Optional eligibility filter (``min_coobs`` / ``coobs``; sensitivity analysis
``PRESPECIFIED['fm_min_coobs_sensitivity']`` for the foundation-model attention
columns, decision-log item 9).  An edge whose co-observation count is below
``min_coobs`` (or missing) is treated exactly like a NaN score: it is never
predicted, and the top-k is taken among the remaining edges.  Because the tie
keys do not depend on which edges are present, the filtered set is the first k
*eligible* edges of the same total order; in particular it contains every
eligible edge of the unfiltered top-k set.  The co-observation column of an
attention score is given by ``fm_coobs_column`` (``coobs.parquet`` of the
``fm_scores`` stage).

Usage::

    from r2.predicted_sets import topk_per_tf
    mask = topk_per_tf(scores, "scgpt_attn", k=50)        # boolean Series
    predicted = scores[mask]
    # sensitivity: only edges co-observed in >= 5 extraction cells are eligible
    df = scores.merge(coobs, on=["tf", "target"])          # fm_scores/<t>/coobs.parquet
    mask5 = topk_per_tf(df, "scgpt_attn", k=50, min_coobs=C.PRESPECIFIED["fm_min_coobs_sensitivity"],
                        coobs=fm_coobs_column("scgpt_attn"))
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from r2 import common as C  # noqa: E402


def tie_break_keys(tf, target, score_col: str, seed: int) -> np.ndarray:
    """Pseudo-random uint64 key per edge, a function of (seed, score_col, tf, target) only."""
    salt = f"{int(seed)}\x1f{score_col}\x1f".encode()
    keys = np.empty(len(tf), dtype=np.uint64)
    for i, (a, b) in enumerate(zip(tf, target)):
        h = hashlib.blake2b(salt + f"{a}\x1f{b}".encode(), digest_size=8).digest()
        keys[i] = int.from_bytes(h, "little")
    return keys


def fm_coobs_column(score_col: str) -> str | None:
    """Co-observation column (``fm_scores`` ``coobs.parquet``) of an attention score column:
    ``scgpt_*`` -> ``scgpt_coobs``, ``geneformer_*`` -> ``geneformer_coobs``, else None."""
    for model in C.MODELS:
        if score_col.startswith(f"{model}_attn"):
            return f"{model}_coobs"
    return None


def _coobs_values(df: pd.DataFrame, coobs) -> np.ndarray:
    if isinstance(coobs, str):
        if coobs not in df.columns:
            raise KeyError(coobs)
        v = df[coobs]
    elif isinstance(coobs, pd.Series):
        if not coobs.index.equals(df.index):
            raise ValueError("coobs Series index must equal df.index")
        v = coobs
    else:
        v = pd.Series(np.asarray(coobs))
        if len(v) != len(df):
            raise ValueError(f"coobs has length {len(v)}, df has {len(df)} rows")
    return pd.to_numeric(v, errors="raise").to_numpy(dtype=np.float64, na_value=np.nan)


def topk_per_tf(df: pd.DataFrame, score_col: str, k: int,
                seed: int = C.PRESPECIFIED["topk_tie_seed"],
                tf_col: str = "tf", target_col: str = "target",
                min_coobs: float | None = None, coobs=None) -> pd.Series:
    """Boolean mask (aligned with ``df.index``) of the top-``k`` edges per TF by ``score_col``.

    See the module docstring for the prespecified rule.  ``df`` must contain one
    row per (tf, target) edge.

    ``min_coobs`` (optional, with ``coobs``): only edges with ``coobs >= min_coobs``
    are eligible; the others (and edges with a missing count) are treated as NaN
    scores.  ``coobs`` is a column name of ``df``, a Series indexed like ``df`` or
    an array of ``len(df)`` counts.
    """
    k = int(k)
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    for c in (tf_col, target_col, score_col):
        if c not in df.columns:
            raise KeyError(c)
    if df.duplicated([tf_col, target_col]).any():
        raise ValueError("duplicated (tf, target) rows")
    if (min_coobs is None) != (coobs is None):
        raise ValueError("min_coobs and coobs must be given together")

    s = pd.to_numeric(df[score_col], errors="raise").to_numpy(dtype=np.float64, na_value=np.nan)
    if min_coobs is not None:
        if not np.isfinite(float(min_coobs)):
            raise ValueError(f"min_coobs must be finite, got {min_coobs}")
        eligible = _coobs_values(df, coobs) >= float(min_coobs)     # NaN count -> False
        s = np.where(eligible, s, np.nan)
    out = np.zeros(len(df), dtype=bool)
    idx = np.flatnonzero(~np.isnan(s))
    if idx.size == 0:
        return pd.Series(out, index=df.index, name="predicted")

    tf_all = df[tf_col].to_numpy()
    codes, _ = pd.factorize(tf_all[idx], sort=True)
    sv = s[idx]
    order = np.lexsort((-sv, codes))           # by TF, then score descending
    oc, os_, oi = codes[order], sv[order], idx[order]
    first = np.searchsorted(oc, oc, side="left")
    size = np.searchsorted(oc, oc, side="right") - first
    take = np.minimum(k, size)                 # number of places per TF
    vk = os_[first + take - 1]                 # k-th largest value (or the minimum)

    above = os_ > vk
    tied = os_ == vk
    n_groups = int(oc.max()) + 1
    n_above = np.bincount(oc, weights=above.astype(np.float64), minlength=n_groups).astype(np.int64)[oc]
    n_tied = np.bincount(oc, weights=tied.astype(np.float64), minlength=n_groups).astype(np.int64)[oc]
    need = take - n_above                      # places left for the tie block

    sel = above | (tied & (n_tied <= need))
    split = tied & (n_tied > need)             # tie blocks cut by the k-th position
    if split.any():
        js = np.flatnonzero(split)
        rows = oi[js]
        keys = tie_break_keys(tf_all[rows], df[target_col].to_numpy()[rows], score_col, seed)
        tgt = df[target_col].to_numpy()[rows].astype(str)
        # within each TF: key descending (bitwise NOT turns ascending into descending);
        # target name breaks the (astronomically unlikely) 64-bit key collisions
        o2 = np.lexsort((tgt, ~keys, oc[js]))
        g = oc[js][o2]
        rank = np.arange(len(o2)) - np.searchsorted(g, g, side="left")
        chosen = np.zeros(len(js), dtype=bool)
        chosen[o2] = rank < need[js][o2]
        sel[js] = chosen
    out[oi[sel]] = True
    return pd.Series(out, index=df.index, name="predicted")
