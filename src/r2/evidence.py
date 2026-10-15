"""Observation-masked annotator evidence over TF x gene universes (stage D2b, ``outputs/r2/evidence``).

Inputs are the committed annotator files written by ``build_annotators.py``:
``data/r2/inputs/annotator_edges.parquet`` (source_id, family, lineage, tf, target,
via_dorothea) and ``data/r2/inputs/annotator_sources.json`` (held-in list, ChIP
sensitivity sources, sensitivity definitions); tissue universes come from
``outputs/r2/universe/<tissue>`` (``build_panels.py``).

For an edge e = (tf, g) and annotator j the evidence value is

    r_ej =  1   if (tf, g) is an edge of j                         (reported)
    r_ej =  0   if j observes (tf, g) but does not report it      (observed negative)
    r_ej = -1   if j does not observe (tf, g)                       (missing)

Observation masks
-----------------
cov_j   = TFs with >= 1 edge in j (whole resource, not universe-restricted)
nodes_j = genes appearing as TF or target of any edge in j
genome-wide families = MOTIF, COEXP, CHIP (scan / inference / assay covers every gene)

* ``prespecified``   (plan section 2.0, literal):
      j observes (tf, g)  iff  tf in cov_j  and  (family_j genome-wide  or  g in nodes_j)
* ``both_endpoints`` (R1 rule, sensitivity):
      j observes (tf, g)  iff  tf in nodes_j  and  g in nodes_j
* ``*_cc`` ("curated-conditional") variants apply the same rule to LIT (and CHIP)
  annotators, but a MOTIF or COEXP annotator observes (tf, g) only if, in addition,
  (tf, g) is in the conditioning set Cond: edges with DoRothEA-channel evidence from a
  held-in literature resource (LIT rows with ``via_dorothea``).  MOTIF/COEXP reports outside
  that set are -1.  Reason: HOCOMOCO, JASPAR and ARACNe-GTEx are only available as DoRothEA
  A-D evidence, and DoRothEA releases a TFBS/GTEx-supported edge only if it also has
  curated or ChIP-seq support; a MOTIF/COEXP report without literature support therefore
  implies held-out ChIP support (~90% of these reports).  Inside the conditioning set the
  edge is released regardless of TFBS/GTEx support, so the reports are complete there,
  and the restriction depends only on held-in evidence.  ``prespecified_cc`` is primary
  (decision log 2026-10-07, item 1; paper wording: "curated-conditional observation model").

  MOTIF/COEXP TF coverage under the ``_cc`` masks: ``PRESPECIFIED['motif_coexp_tf_coverage']
  = 'whole_resource'`` (decision log items 10-11).  cov_j and nodes_j are read from all of j's
  released records, exactly as for the other masks, so the masks are

      prespecified_cc   : tf in cov_j                      and (tf, g) in Cond
      both_endpoints_cc : tf in nodes_j and g in nodes_j   and (tf, g) in Cond   (R1 node rule)

  Every stored output (``meta.json`` key ``motif_coexp_tf_coverage``) and every fit uses this
  rule; ``load_patterns`` refuses stored evidence built under another rule.  The residual
  TF-level dependence on DoRothEA's export rule (a TF's motif/coexpression records are released
  only with curated or ChIP support) is a disclosed limitation; within-TF metrics are not
  affected by TF-level selection.

  Not used: ``cc_tf_coverage='within_conditioning_set'`` (decision log item 10, REVERTED by
  item 11 on 2026-10-08 before any held-out result was inspected).  It counts a TF as covered
  by a MOTIF/COEXP annotator only if that annotator reports >= 1 edge of the TF inside Cond
  (cov_j^cc).  This makes observation depend on the annotator's own reports in the very cells
  being masked (every covered TF has a positive by construction, apparent positive rates rise
  by 2-18%), which violates the missing-at-random assumption of the latent-class models.  It is
  kept only as a non-default option of ``AnnotatorBank`` / ``bank_for_variant`` and for the
  optional before/after table (``build(..., compare_cc_rules=True)``, CLI
  ``--compare-cc-coverage``, written to ``evidence_cc_tf_coverage.csv`` in ``--tables-dir``).
  Under that option only the TF condition changes (``both_endpoints_cc``: tf in cov_j^cc); the
  target condition always uses the whole-resource node set nodes_j (the change of the
  ``both_endpoints_cc`` target rule made in the same attempt is reverted too).

Universes
---------
* tissue t: sorted ``tfs.txt`` x sorted ``panel.tsv`` genes, self-pairs excluded.
* global (decision log item 2): TFs with >= 1 edge in a held-in LIT annotator x (targets of
  held-in LIT annotators U all tissue panel genes); stored as a full TF-major grid whose
  self-pairs carry pattern -1.  It does not depend on MOTIF/COEXP/CHIP edges and is the
  same for every evidence variant.

Evidence variants (all universes)
---------------------------------
=======================  ================  ====================================================
variant                  mask              annotators
=======================  ================  ====================================================
prespecified_cc          prespecified_cc   held-in LIT + MOTIF + COEXP (primary; unsuffixed files)
prespecified             prespecified      held-in
both_endpoints           both_endpoints    held-in
both_endpoints_cc        both_endpoints_cc held-in
chip_prespecified_cc     prespecified_cc   held-in + ReMap + PAZAR-HT (ChIP-inclusive)
lit_merged_extri_ntnu    prespecified_cc   held-in, ExTRI and NTNU.Curated merged into one source
lit_no_ht                prespecified_cc   held-in without exactly 9 LIT regulons
                                           (data/r2/tables/lit_ht_like_regulons.csv): 5 named
                                           high-throughput-like (PAZAR HIF1A; ORegAnno CTCF/YY1/
                                           STAT1/GATA2) + 4 large (> 500 targets: ExTRI SP1/TP53,
                                           TRED MYC/E2F1); FANTOM4, the other ORegAnno regulons,
                                           NFIRegulomeDB and HTRIdb stay in; conditioning set
                                           recomputed without the 9 regulons
lit_no_chipderived       prespecified_cc   held-in without the whole LIT sources
                                           PRESPECIFIED['lit_no_chipderived_drop'] (FANTOM4,
                                           HTRIdb, ORegAnno, LIT_other; decision log item 9);
                                           conditioning set recomputed without them
no_motif_coexp           prespecified_cc   held-in LIT only (cc is then identical to prespecified)
=======================  ================  ====================================================

The universes are the same for every variant (decision log item 2).  For the variants that
remove LIT evidence (``lit_no_ht``, ``lit_no_chipderived``) ``meta.json`` records, per
universe, the TFs and genes that are covered / targeted only by the removed LIT evidence
(``variants.<v>.universe_dependence``); for the global universe the lists are written to
``lit_only_via_removed__<v>.json`` so that held-out metrics can be reported with these
genes excluded (the global universe recomputed without the removed evidence would lack them).

Summary columns (``data/r2/tables/evidence_summary.csv``)
--------------------------------------------------------
``n_tfs_in_source`` = universe TFs with >= 1 edge in the annotator (cov_j, whole resource);
``n_tfs_covered`` = universe TFs with >= 1 observed (non-self) cell under the variant's mask;
``n_tfs_reported`` = universe TFs with >= 1 report in the universe.  Under ``both_endpoints``
a TF observes cells as soon as it is a node (TF or target) of the annotator, so
``n_tfs_covered`` can exceed ``n_tfs_in_source`` (e.g. immune ``__ALL__``: 133 vs 148).

Optional before/after table (not part of the canonical build; ``--compare-cc-coverage``):
``evidence_cc_tf_coverage.csv`` per universe, ``_cc``-mask variant with MOTIF/COEXP annotators,
and MOTIF/COEXP annotator (+ ``__ALL__``): TF coverage, observed cells and positive rate under
the prespecified whole-resource rule and under the reverted within-conditioning-set rule
(``*_within_cond``), plus pattern counts, mean observing annotators per edge and edges
unobserved by every annotator under both rules.

Pattern compression
-------------------
Each edge's evidence vector over the K annotators is encoded as the base-3 integer
sum_j (r_ej + 1) 3^j; ``patterns`` lists the distinct vectors (sorted by code) with their
edge counts and ``edge_pattern`` maps each edge to its pattern row.

Per universe and variant the stage writes ``patterns{sfx}.parquet``, ``edge_pattern{sfx}.npy``,
``tf_coverage{sfx}.parquet`` (per TF and annotator: -1 no observed cell, 0 observed without a
report, 1 >= 1 report) and, for tissue universes, the dense ``evidence{sfx}.parquet``;
``sfx`` is empty for the primary variant and ``__<variant>`` otherwise.

Eligibility-coded evidence (informative-observation extension, additive; section
"Eligibility-coded evidence" below): ``build_elig_patterns`` / ``load_patterns_elig`` derive, from
the stored files without changing them, patterns with codes 1 / 0 / -1 (LIT annotator covers the
TF but the target is not one of its nodes) / -2 (not eligible or not modelled), written to
``patterns{sfx}__elig.parquet``, ``edge_pattern{sfx}__elig.npy`` and ``meta{sfx}__elig.json``;
``load_patterns(u, '<variant>__elig')`` / ``load_edge_pattern(u, '<variant>__elig')`` read them.

Library use::

    from r2 import evidence as E
    bank = E.bank_for_variant("prespecified_cc")
    dense = E.evidence_for_edges(bank, tfs, targets, mask="prespecified_cc")   # int8 (n, K)
    ps = E.evidence_patterns(bank, tf_list, gene_list, mask="both_endpoints")  # PatternSet
    R, n, fam, ann = E.load_patterns("immune")                # stored primary patterns
    R, n, fam, ann = E.load_patterns_elig("immune")           # eligibility-coded (lcm eligibility=True)

Command line (builds all universes and ``data/r2/tables/evidence_summary.csv``)::

    .venv/bin/python subproject_24_multireference_latent_truth_model/src/r2/evidence.py [--universes immune kidney lung global]

``--out-dir DIR --tables-dir DIR`` write the same files elsewhere (used to check that two
builds are byte-identical); the defaults are ``outputs/r2/evidence`` and ``data/r2/tables``.
``--compare-cc-coverage`` additionally writes the optional before/after table (use it only
with a scratch ``--tables-dir``; the comparison does not change any other output).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from r2 import common as C  # noqa: E402

INPUT_EDGES = C.INPUTS / "annotator_edges.parquet"
INPUT_SOURCES = C.INPUTS / "annotator_sources.json"
EV_DIR = C.OUT / "evidence"
FAMILY_ORDER = ("LIT", "MOTIF", "COEXP", "CHIP")
GENOME_WIDE = ("MOTIF", "COEXP", "CHIP")
CC_FAMILIES = ("MOTIF", "COEXP")
MASKS = ("prespecified", "prespecified_cc", "both_endpoints", "both_endpoints_cc")
PRIMARY_MASK = "prespecified_cc"
# stored variants: mask, annotator set ('heldin', 'heldin+chip', 'lit') and edge transform
VARIANTS = {
    "prespecified_cc": dict(mask="prespecified_cc", sources="heldin", transform=None),   # primary
    "prespecified": dict(mask="prespecified", sources="heldin", transform=None),
    "both_endpoints": dict(mask="both_endpoints", sources="heldin", transform=None),
    "both_endpoints_cc": dict(mask="both_endpoints_cc", sources="heldin", transform=None),
    "chip_prespecified_cc": dict(mask="prespecified_cc", sources="heldin+chip", transform=None),
    "lit_merged_extri_ntnu": dict(mask="prespecified_cc", sources="heldin", transform="merge_sets"),
    "lit_no_ht": dict(mask="prespecified_cc", sources="heldin", transform="drop_lit_ht_like"),
    "lit_no_chipderived": dict(mask="prespecified_cc", sources="heldin", transform="drop_lit_chipderived"),
    "no_motif_coexp": dict(mask="prespecified_cc", sources="lit", transform=None),
}
PRIMARY_VARIANT = C.PRESPECIFIED["evidence_variant_primary"]
SENSITIVITY_VARIANTS = [v for v in VARIANTS if v != PRIMARY_VARIANT]
LIT_REMOVING_VARIANTS = ("lit_no_ht", "lit_no_chipderived")
MAX_K = 39  # 3**39 < 2**63
# MOTIF/COEXP TF coverage under the _cc masks.  The default is PRESPECIFIED
# ('whole_resource', decision log items 10-11); 'within_conditioning_set' is the reverted
# decision 10, kept only as a non-default option (outcome-dependent observation, see docstring)
CC_TF_COVERAGE_RULES = ("whole_resource", "within_conditioning_set")
CC_TF_COVERAGE = C.PRESPECIFIED["motif_coexp_tf_coverage"]
CC_TF_COVERAGE_ALTERNATIVE = "within_conditioning_set"
CC_TABLE = "evidence_cc_tf_coverage.csv"
SUMMARY_TABLE = "evidence_summary.csv"
assert PRIMARY_VARIANT in VARIANTS
assert set(C.PRESPECIFIED["evidence_variants_sensitivity"]) <= set(SENSITIVITY_VARIANTS)
assert CC_TF_COVERAGE in CC_TF_COVERAGE_RULES


# --------------------------------------------------------------------------------------
# Inputs and variant edge tables
# --------------------------------------------------------------------------------------
@lru_cache(maxsize=1)
def load_annotator_inputs() -> tuple[pd.DataFrame, dict]:
    """Committed annotator edges (all sources incl. held-out CHIP) and source metadata."""
    ed = pd.read_parquet(INPUT_EDGES, columns=["source_id", "family", "lineage", "tf", "target", "via_dorothea"])
    meta = json.loads(INPUT_SOURCES.read_text())
    return ed, meta


def transform_edges(ed: pd.DataFrame, meta: dict, transform: str | None) -> pd.DataFrame:
    """Apply a sensitivity transform to the annotator edge table."""
    if transform is None:
        return ed
    if transform == "merge_sets":
        out = ed
        for name, members in meta["merge_sets"].items():
            m = out["source_id"].isin(members)
            merged = (out[m].assign(source_id=name)
                      .groupby(["source_id", "tf", "target"], sort=True)
                      .agg(family=("family", "first"), lineage=("lineage", "first"), via_dorothea=("via_dorothea", "any"))
                      .reset_index())
            if merged["family"].nunique() > 1:
                raise ValueError(f"merge set {name} spans several families")
            out = pd.concat([out[~m], merged[out.columns]], ignore_index=True)
        return out.sort_values(["source_id", "tf", "target"]).reset_index(drop=True)
    if transform == "drop_lit_ht_like":
        key = pd.MultiIndex.from_tuples([tuple(x) for x in meta["lit_ht_like_regulons"]], names=["source_id", "tf"])
        drop = pd.MultiIndex.from_frame(ed[["source_id", "tf"]]).isin(key)
        return ed[~drop].reset_index(drop=True)
    if transform == "drop_lit_chipderived":
        drop_src = list(meta["lit_no_chipderived_drop"])
        fam = ed.groupby("source_id")["family"].first()
        bad = [x for x in drop_src if fam.get(x) != "LIT"]
        if bad:
            raise KeyError(f"lit_no_chipderived_drop: not LIT sources of the edge table: {bad}")
        return ed[~ed["source_id"].isin(drop_src)].reset_index(drop=True)
    raise ValueError(f"unknown transform {transform}")


def conditioning_set(ed: pd.DataFrame) -> pd.DataFrame:
    """Edges with DoRothEA-channel evidence from a held-in literature resource."""
    m = (ed["family"] == "LIT") & ed["via_dorothea"].astype(bool)
    return ed.loc[m, ["tf", "target"]].drop_duplicates().sort_values(["tf", "target"]).reset_index(drop=True)


def variant_sources(ed: pd.DataFrame, meta: dict, which: str) -> list:
    fam = ed.groupby("source_id")["family"].first()
    heldin = [s for s in fam.index if fam[s] != "CHIP"]
    if which == "heldin":
        return heldin
    if which == "heldin+chip":
        return heldin + list(meta["chip_sensitivity_sources"])
    if which == "lit":
        return [s for s in heldin if fam[s] == "LIT"]
    raise ValueError(which)


def _check_registered(meta: dict) -> None:
    """The committed sensitivity definitions must match the prespecified constants."""
    if list(meta.get("lit_no_chipderived_drop", [])) != list(C.PRESPECIFIED["lit_no_chipderived_drop"]):
        raise ValueError("annotator_sources.json lit_no_chipderived_drop differs from "
                         "PRESPECIFIED['lit_no_chipderived_drop']: rebuild the annotators")


def variant_edges(variant: str) -> pd.DataFrame:
    """Annotator edge table (all families) after the variant's transform."""
    ed, meta = load_annotator_inputs()
    _check_registered(meta)
    return transform_edges(ed, meta, VARIANTS[variant]["transform"])


def bank_for_variant(variant: str, cc_tf_coverage: str = CC_TF_COVERAGE) -> "AnnotatorBank":
    spec = VARIANTS[variant]
    _, meta = load_annotator_inputs()
    ed = variant_edges(variant)
    return AnnotatorBank(ed, sources=variant_sources(ed, meta, spec["sources"]), cond=conditioning_set(ed),
                         cc_tf_coverage=cc_tf_coverage)


# --------------------------------------------------------------------------------------
# Annotator bank
# --------------------------------------------------------------------------------------
class AnnotatorBank:
    """Edges, coverage and node sets of a chosen list of annotators, indexed for fast lookup.

    Parameters
    ----------
    edges   : DataFrame with source_id, family, tf, target (and via_dorothea if ``cond`` is None).
    sources : annotators to use (default: every non-CHIP source in ``edges``); ordered by
              family then name.
    cond    : conditioning set (tf, target) of the curated-conditional masks (default:
              ``conditioning_set(edges)``, i.e. derived from all LIT rows of ``edges``).
    cc_tf_coverage : TF coverage of MOTIF/COEXP annotators under the ``_cc`` masks.
              'whole_resource' (default = ``PRESPECIFIED['motif_coexp_tf_coverage']``):
              cov_j from all of the annotator's records, as for every other mask.
              'within_conditioning_set' (NOT USED; reverted decision 10): cov_j^cc = TFs with
              >= 1 record of the annotator inside ``cond``.  Not used because observation then
              depends on the annotator's own reports (outcome-dependent missingness, every
              covered TF has a positive by construction; decision log item 11).  Only the TF
              condition changes; target conditions always use the whole-resource nodes_j.

    Attributes ``cov`` / ``node_mask`` hold the whole-resource TF coverage and node sets;
    ``cov_cond`` the TFs with >= 1 record inside ``cond`` (diagnostic); ``cov_cc`` the TF
    coverage used by the ``_cc`` masks (equal to ``cov`` under 'whole_resource').
    """

    def __init__(self, edges: pd.DataFrame, sources: list | None = None, cond: pd.DataFrame | None = None,
                 cc_tf_coverage: str = CC_TF_COVERAGE):
        if cc_tf_coverage not in CC_TF_COVERAGE_RULES:
            raise ValueError(f"cc_tf_coverage must be one of {CC_TF_COVERAGE_RULES}")
        self.cc_tf_coverage = cc_tf_coverage
        fam = edges.groupby("source_id")["family"].first().to_dict()
        if sources is None:
            sources = [s for s in fam if fam[s] != "CHIP"]
        unknown = sorted(set(sources) - set(fam))
        if unknown:
            raise KeyError(f"sources not in edge table: {unknown}")
        sources = sorted(set(sources), key=lambda s: (FAMILY_ORDER.index(fam[s]), s))
        if len(sources) > MAX_K:
            raise ValueError(f"at most {MAX_K} annotators supported")
        if cond is None:
            cond = conditioning_set(edges)
        self.sources = sources
        self.family = [fam[s] for s in sources]
        self.K = len(sources)
        self.genome_wide = np.array([f in GENOME_WIDE for f in self.family])
        self.is_cc = np.array([f in CC_FAMILIES for f in self.family])
        self.n_cond = int(len(cond))

        ed = edges[edges["source_id"].isin(sources)]
        vocab = pd.Index(sorted(set(ed["tf"]) | set(ed["target"]) | set(cond["tf"]) | set(cond["target"])))
        self.vocab = vocab
        self.V = len(vocab)
        self.node_mask = np.zeros((self.K, self.V), dtype=bool)
        self.cov: list[set] = []
        self.pos: list[dict] = []
        for j, s in enumerate(sources):
            e = ed[ed["source_id"] == s]
            ti = vocab.get_indexer(e["tf"])
            gi = vocab.get_indexer(e["target"])
            self.node_mask[j, ti] = True
            self.node_mask[j, gi] = True
            self.cov.append(set(e["tf"]))
            order = np.lexsort((gi, ti))
            ti, gi = ti[order], gi[order]
            tfs_sorted = e["tf"].values[order]
            d = {}
            if len(ti):
                cut = np.flatnonzero(np.diff(ti)) + 1
                for a, b in zip(np.r_[0, cut], np.r_[cut, len(ti)]):
                    d[tfs_sorted[a]] = gi[a:b].astype(np.int64)
            self.pos.append(d)
        ci = vocab.get_indexer(cond["target"])
        self.cond = {tf: np.sort(ci[idx]).astype(np.int64)
                     for tf, idx in cond.groupby("tf").indices.items()}
        self._empty = np.zeros(0, dtype=np.int64)

        # TFs with >= 1 record inside the conditioning set (diagnostic; TF coverage of
        # MOTIF/COEXP annotators only under the non-default 'within_conditioning_set' option)
        cond_key = pd.MultiIndex.from_frame(cond[["tf", "target"]])
        self.cov_cond: list[set] = []
        for j, s in enumerate(sources):
            e = ed[ed["source_id"] == s]
            e = e[pd.MultiIndex.from_frame(e[["tf", "target"]]).isin(cond_key)]
            self.cov_cond.append(set(e["tf"]))
        self.cov_cc = list(self.cov)
        if cc_tf_coverage == "within_conditioning_set":
            for j in np.flatnonzero(self.is_cc):
                self.cov_cc[j] = self.cov_cond[j]

    @classmethod
    def default(cls, include_chip: bool = False) -> "AnnotatorBank":
        return bank_for_variant("chip_prespecified_cc" if include_chip else PRIMARY_VARIANT)

    # ----------------------------------------------------------------------------------
    def describe(self) -> pd.DataFrame:
        return pd.DataFrame({"source_id": self.sources, "family": self.family,
                             "genome_wide": self.genome_wide, "n_cov_tfs": [len(c) for c in self.cov],
                             "n_cov_tfs_cond": [len(c) for c in self.cov_cond],
                             "n_cov_tfs_cc_mask": [len(c) for c in self.cov_cc],
                             "n_nodes": self.node_mask.sum(1),
                             "n_edges": [sum(len(v) for v in d.values()) for d in self.pos]})

    def target_index(self, genes) -> np.ndarray:
        """Vocabulary index of each gene (-1 if the gene is in no selected annotator)."""
        return self.vocab.get_indexer(pd.Index(genes))

    def tf_block(self, tf: str, tgt_idx: np.ndarray, mask: str, node_obs: np.ndarray | None = None,
                 inv: np.ndarray | None = None) -> np.ndarray:
        """Evidence (n_targets, K) int8 for one TF against targets given as vocabulary indices.

        node_obs : optional precomputed (K, n) bool = target in the annotator's node set
                   (``self.node_mask``; grid speed-up).
        inv      : optional (V,) int map vocabulary index -> position in the target vector
                   (requires unique targets; grid speed-up).
        """
        if mask not in MASKS:
            raise ValueError(f"mask must be one of {MASKS}")
        n = len(tgt_idx)
        out = np.full((n, self.K), -1, dtype=np.int8)
        valid = tgt_idx >= 0
        safe = np.where(valid, tgt_idx, 0)
        both = mask.startswith("both_endpoints")
        cc = mask.endswith("_cc")
        within = self.cc_tf_coverage == "within_conditioning_set"
        nmask = self.node_mask
        cov = self.cov_cc if cc else self.cov
        tfi = self.vocab.get_indexer([tf])[0]
        cond_pos = None
        for j in range(self.K):
            if both:
                if cc and within and self.is_cc[j]:
                    if tf not in cov[j]:          # non-default TF-coverage option (reverted item 10)
                        continue
                elif tfi < 0 or not nmask[j, tfi]:
                    continue
                obs = node_obs[j].copy() if node_obs is not None else valid & nmask[j, safe]
            else:
                if tf not in cov[j]:
                    continue
                if self.genome_wide[j]:
                    obs = np.ones(n, dtype=bool)
                else:
                    obs = node_obs[j].copy() if node_obs is not None else valid & nmask[j, safe]
            pos = self._member(self.pos[j].get(tf, self._empty), tgt_idx, valid, inv)
            if cc and self.is_cc[j]:
                if cond_pos is None:
                    cond_pos = self._member(self.cond.get(tf, self._empty), tgt_idx, valid, inv)
                obs &= cond_pos
                pos &= cond_pos
            if (pos & ~obs).any():   # positives are always observed under every mask
                raise AssertionError(f"reported edge not observed: {self.sources[j]} {tf}")
            col = np.where(obs, np.int8(0), np.int8(-1))
            col[pos] = 1
            out[:, j] = col
        return out

    @staticmethod
    def _member(arr: np.ndarray, tgt_idx: np.ndarray, valid: np.ndarray, inv: np.ndarray | None) -> np.ndarray:
        n = len(tgt_idx)
        if len(arr) == 0:
            return np.zeros(n, dtype=bool)
        if inv is not None:
            m = np.zeros(n, dtype=bool)
            p = inv[arr]
            m[p[p >= 0]] = True
            return m
        return valid & np.isin(tgt_idx, arr)


# --------------------------------------------------------------------------------------
# Dense evidence for arbitrary edge lists (small universes)
# --------------------------------------------------------------------------------------
def evidence_for_edges(bank: AnnotatorBank, tfs, targets, mask: str = PRIMARY_MASK) -> np.ndarray:
    """Dense int8 matrix (n_edges, K) for an arbitrary list of (tf, target) edges (input order kept)."""
    tfs = np.asarray(tfs, dtype=object)
    targets = np.asarray(targets, dtype=object)
    out = np.empty((len(tfs), bank.K), dtype=np.int8)
    tidx = bank.target_index(targets)
    for tf, idx in pd.Series(np.arange(len(tfs))).groupby(tfs, sort=False).indices.items():
        out[idx] = bank.tf_block(tf, tidx[idx], mask)
    return out


# --------------------------------------------------------------------------------------
# Pattern compression
# --------------------------------------------------------------------------------------
@dataclass
class PatternSet:
    patterns: np.ndarray      # (P, K) int8 in {-1, 0, 1}
    counts: np.ndarray        # (P,) int64 edges per pattern
    edge_pattern: np.ndarray  # (n_edges,) int32 pattern row per edge (-1 = not an edge)
    sources: list
    family: list
    mask: str
    tfs: list | None = None
    tf_observed: np.ndarray | None = None   # (n_tfs, K) bool: >= 1 observed (non-self) cell
    tf_reported: np.ndarray | None = None   # (n_tfs, K) bool: >= 1 report

    def to_frame(self) -> pd.DataFrame:
        df = pd.DataFrame(self.patterns, columns=self.sources)
        df.insert(0, "pattern_id", np.arange(len(df), dtype=np.int32))
        df["n_edges"] = self.counts
        return df

    def tf_coverage_frame(self) -> pd.DataFrame:
        """Per TF and annotator: -1 no observed cell, 0 observed without a report, 1 >= 1 report."""
        v = np.where(self.tf_reported, 1, np.where(self.tf_observed, 0, -1)).astype(np.int8)
        df = pd.DataFrame(v, columns=self.sources)
        df.insert(0, "tf", self.tfs)
        return df

    def summary(self) -> pd.DataFrame:
        """Per-annotator coverage and positive rate (from the compressed representation)."""
        n_edges = int(self.counts.sum())
        obs = (self.patterns >= 0).astype(np.int64).T @ self.counts
        pos = (self.patterns == 1).astype(np.int64).T @ self.counts
        df = pd.DataFrame({"annotator": self.sources, "family": self.family,
                           "n_edges": n_edges, "n_observed": obs, "coverage": obs / max(n_edges, 1),
                           "n_pos": pos, "pos_rate": np.where(obs > 0, pos / np.maximum(obs, 1), np.nan)})
        if self.tf_observed is not None:
            df["n_tfs_covered"] = self.tf_observed.sum(0)
            df["n_tfs_reported"] = self.tf_reported.sum(0)
        return df


def _codes(block: np.ndarray, pow3: np.ndarray) -> np.ndarray:
    return (block.astype(np.int64) + 1) @ pow3


def _decode(codes: np.ndarray, K: int) -> np.ndarray:
    pat = np.empty((len(codes), K), dtype=np.int8)
    c = codes.copy()
    for j in range(K):
        pat[:, j] = (c % 3) - 1
        c //= 3
    return pat


def evidence_patterns(bank: AnnotatorBank, tfs, genes, mask: str = PRIMARY_MASK, grid: bool = True,
                      return_dense: bool = False, verbose: bool = False):
    """Pattern-compressed evidence for the universe tfs x genes (TF-major, input orders kept).

    grid=True  : edge_pattern has length len(tfs) * len(genes); self-pairs get -1.
    grid=False : self-pairs are dropped (edge order = TF-major over non-self pairs).
    return_dense=True additionally returns the dense int8 matrix (only for small universes).
    Memory: one (len(genes), K) block per TF plus an int64 code per edge.
    """
    tfs = list(tfs)
    genes = pd.Index(list(genes))
    if genes.has_duplicates or len(set(tfs)) != len(tfs):
        raise ValueError("tfs and genes must be unique")
    K, G = bank.K, len(genes)
    pow3 = 3 ** np.arange(K, dtype=np.int64)
    tgt_idx = bank.target_index(genes)
    valid = tgt_idx >= 0
    node_obs = valid[None, :] & bank.node_mask[:, np.where(valid, tgt_idx, 0)]
    inv = np.full(bank.V, -1, dtype=np.int64)
    inv[tgt_idx[valid]] = np.flatnonzero(valid)
    codes = np.empty(len(tfs) * G, dtype=np.int64)
    dense = np.empty((len(tfs) * G, K), dtype=np.int8) if return_dense else None
    tf_obs = np.zeros((len(tfs), K), dtype=bool)
    tf_rep = np.zeros((len(tfs), K), dtype=bool)
    self_pos = genes.get_indexer(tfs)
    t0 = time.time()
    for i, tf in enumerate(tfs):
        blk = bank.tf_block(tf, tgt_idx, mask, node_obs=node_obs, inv=inv)
        c = _codes(blk, pow3)
        if self_pos[i] >= 0:
            c[self_pos[i]] = -1
            rows = np.ones(G, dtype=bool)
            rows[self_pos[i]] = False
            b2 = blk[rows]
        else:
            b2 = blk
        tf_obs[i] = (b2 >= 0).any(0)
        tf_rep[i] = (b2 == 1).any(0)
        codes[i * G:(i + 1) * G] = c
        if return_dense:
            dense[i * G:(i + 1) * G] = blk
        if verbose and (i + 1) % 200 == 0:
            print(f"    {i + 1}/{len(tfs)} TFs ({time.time() - t0:.0f}s)", flush=True)
    keep = codes >= 0
    uniq, inv_codes, counts = np.unique(codes[keep], return_inverse=True, return_counts=True)
    if grid:
        ep = np.full(len(codes), -1, dtype=np.int32)
        ep[keep] = inv_codes.astype(np.int32)
    else:
        ep = inv_codes.astype(np.int32)
        if return_dense:
            dense = dense[keep]
    ps = PatternSet(_decode(uniq, K), counts.astype(np.int64), ep, list(bank.sources), list(bank.family), mask,
                    tfs=tfs, tf_observed=tf_obs, tf_reported=tf_rep)
    return (ps, dense) if return_dense else ps


# --------------------------------------------------------------------------------------
# Loading stored evidence
# --------------------------------------------------------------------------------------
def _suffix(variant: str) -> str:
    return "" if variant == PRIMARY_VARIANT else f"__{variant}"


def check_coverage_rule(meta: dict, where: str = "") -> None:
    """Stored evidence must have been built under ``PRESPECIFIED['motif_coexp_tf_coverage']``.

    ``meta`` is a universe's ``meta.json``; a missing key or another rule raises ValueError, so
    no fit can silently use evidence built under the reverted decision 10."""
    want = C.PRESPECIFIED["motif_coexp_tf_coverage"]
    got = meta.get("motif_coexp_tf_coverage")
    if got != want:
        raise ValueError(f"stored evidence {where} was built with motif_coexp_tf_coverage={got!r}, "
                         f"but PRESPECIFIED['motif_coexp_tf_coverage'] = {want!r}: rebuild with evidence.py")


def load_meta(universe: str, ev_dir: Path | None = None) -> dict:
    """``meta.json`` of a stored universe, checked against the prespecified coverage rule."""
    d = Path(ev_dir or EV_DIR) / universe
    meta = json.loads((d / "meta.json").read_text())
    check_coverage_rule(meta, str(d))
    return meta


def load_patterns(universe: str, variant: str = PRIMARY_VARIANT, ev_dir: Path | None = None):
    """Return (R (P,K) int8, n (P,) int64, family list, annotator list) for a stored universe.

    Usable directly as ``lcm.PatternData(R, n, family, annotators)``.  Raises ValueError if the
    stored ``meta.json`` does not record ``PRESPECIFIED['motif_coexp_tf_coverage']``.
    An eligibility-coded pseudo-variant ``'<variant>__elig'`` returns ``load_patterns_elig``
    (use ``lcm.PatternData(..., eligibility=True)``).
    """
    base_variant, is_elig = split_elig_variant(variant)
    if is_elig:
        return load_patterns_elig(universe, base_variant, ev_dir)
    d = Path(ev_dir or EV_DIR) / universe
    meta = load_meta(universe, ev_dir)
    df = pd.read_parquet(d / f"patterns{_suffix(variant)}.parquet")
    ann = meta["variants"][variant]["sources"]
    fam = meta["variants"][variant]["family"]
    return df[ann].to_numpy(np.int8), df["n_edges"].to_numpy(np.int64), fam, ann


def load_edge_pattern(universe: str, variant: str = PRIMARY_VARIANT, mmap: bool = True,
                      ev_dir: Path | None = None) -> np.ndarray:
    base_variant, is_elig = split_elig_variant(variant)
    if is_elig:
        return load_edge_pattern_elig(universe, base_variant, mmap, ev_dir)
    load_meta(universe, ev_dir)
    return np.load(Path(ev_dir or EV_DIR) / universe / f"edge_pattern{_suffix(variant)}.npy",
                   mmap_mode="r" if mmap else None)


# --------------------------------------------------------------------------------------
# Universes
# --------------------------------------------------------------------------------------
def tissue_universe(t: str) -> tuple[list, list]:
    ud = C.OUT / "universe" / t
    tfs = sorted(x.strip() for x in (ud / "tfs.txt").read_text().splitlines() if x.strip())
    genes = sorted(pd.read_csv(ud / "panel.tsv", sep="\t")["gene"].astype(str))
    return tfs, genes


def global_universe(edges: pd.DataFrame, panel_genes) -> tuple[list, list, dict]:
    """TFs of held-in LIT annotators x (targets of held-in LIT annotators U panel genes).

    Only LIT rows are read, so the universe cannot depend on MOTIF/COEXP reports (visible
    only through ChIP support) or on CHIP sources.
    """
    lit = edges[edges["family"] == "LIT"]
    tfs = sorted(set(lit["tf"]))
    lit_targets = set(lit["target"])
    genes = sorted(lit_targets | set(panel_genes))
    info = {"definition": "TFs with >= 1 edge in a held-in LIT annotator x (targets of held-in LIT annotators "
                          "U all tissue panel genes)",
            "n_lit_tfs": len(tfs), "n_lit_targets": len(lit_targets),
            "n_panel_genes": len(set(panel_genes)), "n_panel_genes_not_lit_target": len(set(panel_genes) - lit_targets)}
    return tfs, genes, info


def universe_dependence(base_lit: pd.DataFrame, var_lit: pd.DataFrame, tfs, genes, panel_genes, grid: bool) -> tuple[dict, dict]:
    """Universe TFs / genes whose LIT coverage comes only from evidence a variant removes.

    tfs_lit_only_via_removed   : universe TFs with a LIT edge in the base table but none in the variant
    genes_lit_only_via_removed : universe genes that are a LIT target in the base table but not in the variant
    For the global universe (grid=True) the 'entering' sets are those that a universe recomputed from the
    variant's LIT evidence would lose (genes additionally not in any tissue panel).
    """
    tfset, gset, panel = set(tfs), set(genes), set(panel_genes)
    tf_rm = sorted(tfset & (set(base_lit["tf"]) - set(var_lit["tf"])))
    g_rm = sorted(gset & (set(base_lit["target"]) - set(var_lit["target"])))
    info = {"n_tfs_lit_only_via_removed": len(tf_rm), "n_genes_lit_target_only_via_removed": len(g_rm)}
    lists = {}
    if grid:
        g_enter = sorted(set(g_rm) - panel)
        T, G = len(tfs), len(genes)
        rt, rg = set(tf_rm), set(g_enter)
        n_cells = len(rt) * G + T * len(rg) - len(rt) * len(rg)
        n_self = len({x for x in tfset & gset if x in rt or x in rg})
        info.update({"n_tfs_entering_only_via_removed": len(tf_rm),
                     "n_genes_entering_only_via_removed": len(g_enter),
                     "frac_genes_entering_only_via_removed": len(g_enter) / max(G, 1),
                     "n_edges_with_tf_or_gene_entering_only_via_removed": int(n_cells - n_self),
                     "definition": ("TFs with no LIT edge and genes that are no LIT target and no tissue-panel gene once "
                                    "the variant's LIT evidence is removed; a global universe recomputed from the "
                                    "variant would not contain them (the stored universe is shared by all variants)")})
        lists = {"tfs": tf_rm, "genes": g_enter, "definition": info["definition"]}
    return info, lists


# --------------------------------------------------------------------------------------
# Build all universes
# --------------------------------------------------------------------------------------
SUMMARY_COLS = ["universe", "variant", "mask", "annotator", "family", "n_tfs", "n_genes", "n_patterns", "n_edges",
                "n_tfs_in_source", "n_tfs_covered", "n_tfs_reported", "n_observed", "coverage",
                "n_pos", "pos_rate", "mean_annotators_observing"]
CC_COLS = ["universe", "variant", "mask", "annotator", "family", "n_tfs", "n_genes", "n_edges",
           "n_tfs_in_source", "n_tfs_in_source_cond",
           "n_tfs_covered", "n_tfs_covered_within_cond", "n_tfs_reported",
           "n_observed", "n_observed_within_cond", "n_observed_lost_within_cond", "n_pos",
           "pos_rate", "pos_rate_within_cond",
           "n_patterns", "n_patterns_within_cond",
           "mean_annotators_observing", "mean_annotators_observing_within_cond",
           "n_edges_unobserved_by_all", "n_edges_unobserved_by_all_within_cond", "n_edges_any_report"]


def _pattern_stats(ps: PatternSet) -> dict:
    """Universe-level counts of a PatternSet (edges, reports, observation)."""
    n_e = int(ps.counts.sum())
    n_obs_edge = (ps.patterns >= 0).sum(1)
    any_obs = int(ps.counts[n_obs_edge > 0].sum())
    return {"n_edges": n_e, "n_patterns": int(len(ps.counts)),
            "n_edges_any_report": int(ps.counts[(ps.patterns == 1).any(1)].sum()),
            "n_edges_observed_by_any": any_obs, "n_edges_unobserved_by_all": n_e - any_obs,
            "n_tfs_covered_by_any": int(ps.tf_observed.any(1).sum()),
            "n_tfs_reported_by_any": int(ps.tf_reported.any(1).sum()),
            "mean_annotators_observing": float((n_obs_edge * ps.counts).sum() / max(n_e, 1))}


def cc_rule_comparison(u: str, v: str, bank: AnnotatorBank, ps: PatternSet, alt: AnnotatorBank, ps_alt: PatternSet,
                       tfs: list, n_genes: int, dense: np.ndarray | None = None,
                       dense_alt: np.ndarray | None = None) -> pd.DataFrame:
    """Before/after rows: prespecified whole-resource coverage (``bank``/``ps``) vs the reverted
    within-conditioning-set option (``alt``/``ps_alt``, columns ``*_within_cond``).

    Checks that the alternative only turns observed negatives of MOTIF/COEXP annotators into
    missing values: reports and every LIT/CHIP column are unchanged (edge level when dense
    matrices are given, otherwise per annotator)."""
    if ps.sources != ps_alt.sources:
        raise AssertionError("annotator order differs between coverage rules")
    cc_cols = np.flatnonzero(bank.is_cc)
    other = np.flatnonzero(~bank.is_cc)
    sm, sm_alt = ps.summary(), ps_alt.summary()
    if not (sm["n_pos"].values == sm_alt["n_pos"].values).all():
        raise AssertionError(f"[{u}/{v}] reports differ between coverage rules")
    if not (sm["n_observed"].values[other] == sm_alt["n_observed"].values[other]).all():
        raise AssertionError(f"[{u}/{v}] LIT/CHIP observation differs between coverage rules")
    if (sm_alt["n_observed"].values[cc_cols] > sm["n_observed"].values[cc_cols]).any():
        raise AssertionError(f"[{u}/{v}] within-conditioning-set option observes more cells")
    if dense is not None:
        if not (dense[:, other] == dense_alt[:, other]).all():
            raise AssertionError(f"[{u}/{v}] LIT/CHIP cells differ between coverage rules")
        d, a = dense[:, cc_cols], dense_alt[:, cc_cols]
        diff = d != a
        if not ((d[diff] == 0) & (a[diff] == -1)).all():
            raise AssertionError(f"[{u}/{v}] coverage option changed cells other than observed negatives")
    st, st_alt = _pattern_stats(ps), _pattern_stats(ps_alt)
    if st["n_edges_any_report"] != st_alt["n_edges_any_report"]:
        raise AssertionError(f"[{u}/{v}] edges with a report differ between coverage rules")
    tfset = set(tfs)
    common = {"universe": u, "variant": v, "mask": ps.mask, "n_tfs": len(tfs), "n_genes": n_genes,
              "n_edges": st["n_edges"], "n_patterns": st["n_patterns"],
              "n_patterns_within_cond": st_alt["n_patterns"],
              "mean_annotators_observing": st["mean_annotators_observing"],
              "mean_annotators_observing_within_cond": st_alt["mean_annotators_observing"],
              "n_edges_unobserved_by_all": st["n_edges_unobserved_by_all"],
              "n_edges_unobserved_by_all_within_cond": st_alt["n_edges_unobserved_by_all"],
              "n_edges_any_report": st["n_edges_any_report"]}
    rows = []
    for j in cc_cols:
        rows.append({**common, "annotator": bank.sources[j], "family": bank.family[j],
                     "n_tfs_in_source": len(tfset & bank.cov[j]),
                     "n_tfs_in_source_cond": len(tfset & bank.cov_cond[j]),
                     "n_tfs_covered": int(ps.tf_observed[:, j].sum()),
                     "n_tfs_covered_within_cond": int(ps_alt.tf_observed[:, j].sum()),
                     "n_tfs_reported": int(ps.tf_reported[:, j].sum()),
                     "n_observed": int(sm["n_observed"].iloc[j]),
                     "n_observed_within_cond": int(sm_alt["n_observed"].iloc[j]),
                     "n_observed_lost_within_cond": int(sm["n_observed"].iloc[j] - sm_alt["n_observed"].iloc[j]),
                     "n_pos": int(sm["n_pos"].iloc[j]),
                     "pos_rate": sm["pos_rate"].iloc[j], "pos_rate_within_cond": sm_alt["pos_rate"].iloc[j]})
    rows.append({**common, "annotator": "__ALL__", "family": "",
                 "n_tfs_in_source": len(tfset & set().union(*bank.cov)),
                 "n_tfs_in_source_cond": len(tfset & set().union(*bank.cov_cond)),
                 "n_tfs_covered": st["n_tfs_covered_by_any"],
                 "n_tfs_covered_within_cond": st_alt["n_tfs_covered_by_any"],
                 "n_tfs_reported": st["n_tfs_reported_by_any"],
                 "n_observed": st["n_edges_observed_by_any"],
                 "n_observed_within_cond": st_alt["n_edges_observed_by_any"],
                 "n_observed_lost_within_cond": st["n_edges_observed_by_any"] - st_alt["n_edges_observed_by_any"],
                 "n_pos": st["n_edges_any_report"],
                 "pos_rate": st["n_edges_any_report"] / max(st["n_edges_observed_by_any"], 1),
                 "pos_rate_within_cond": st["n_edges_any_report"] / max(st_alt["n_edges_observed_by_any"], 1)})
    return pd.DataFrame(rows)


def _cc_compare_variants(banks: dict) -> list:
    """Variants whose mask is curated-conditional and that contain MOTIF/COEXP annotators."""
    return [v for v, spec in VARIANTS.items() if spec["mask"].endswith("_cc") and banks[v].is_cc.any()]


def _write_table(df: pd.DataFrame, path: Path, cols: list, universes: list) -> None:
    """Write a tidy table; a partial rebuild keeps the stored rows of the other universes."""
    if path.exists() and set(universes) != {"immune", "kidney", "lung", "global"}:
        old = pd.read_csv(path)
        df = pd.concat([old[~old["universe"].isin(universes)], df], ignore_index=True)
    df.reindex(columns=cols).to_csv(path, index=False, float_format="%.6g")
    print(f"wrote {path}", flush=True)


def build(universes: list, out_dir: Path = EV_DIR, tables_dir: Path = C.TABLES,
          compare_cc_rules: bool = False) -> None:
    """Build every universe x variant under the prespecified coverage rule.

    ``compare_cc_rules=True`` additionally evaluates the reverted within-conditioning-set
    option and writes the before/after table ``CC_TABLE`` to ``tables_dir``; no other output
    (patterns, meta.json, summary table) depends on this flag."""
    out_dir, tables_dir = Path(out_dir), Path(tables_dir)
    C.ensure_dir(out_dir)
    C.ensure_dir(tables_dir)
    ed0, meta0 = load_annotator_inputs()
    banks = {v: bank_for_variant(v) for v in VARIANTS}
    alt_banks = ({v: bank_for_variant(v, cc_tf_coverage=CC_TF_COVERAGE_ALTERNATIVE)
                  for v in _cc_compare_variants(banks)} if compare_cc_rules else {})
    base_lit = ed0[ed0["family"] == "LIT"]
    var_lit = {}
    for v in LIT_REMOVING_VARIANTS:
        e = variant_edges(v)
        var_lit[v] = e[e["family"] == "LIT"]
    summary_rows, cc_rows = [], []
    for u in universes:
        t0 = time.time()
        od = C.ensure_dir(out_dir / u)
        uinfo = {}
        if u == "global":
            panel = set().union(*(tissue_universe(t)[1] for t in C.TISSUES))
            tfs, genes, uinfo = global_universe(ed0[ed0["family"] != "CHIP"], panel)
            (od / "tfs.txt").write_text("\n".join(tfs) + "\n")
            (od / "genes.txt").write_text("\n".join(genes) + "\n")
            grid = True
        else:
            tfs, genes = tissue_universe(u)
            panel = set(genes)
            grid = False
        n_pairs = len(tfs) * len(genes)
        n_self = len(set(tfs) & set(genes))
        meta = {"universe": u, "n_tfs": len(tfs), "n_genes": len(genes), "n_edges": n_pairs - n_self,
                "n_self_pairs": n_self, "primary_variant": PRIMARY_VARIANT, **uinfo,
                "edge_order": ("TF-major full grid over sorted tfs.txt x sorted genes.txt; self-pairs have "
                               "pattern -1" if grid else
                               "rows of evidence.parquet: TF-major over sorted TFs x sorted panel genes, "
                               "self-pairs excluded"),
                "motif_coexp_tf_coverage": CC_TF_COVERAGE,
                "variants": {}}
        print(f"[{u}] {len(tfs)} TFs x {len(genes)} genes = {n_pairs - n_self:,} edges", flush=True)
        edges_df = None
        if not grid:
            tf_col = np.repeat(np.array(tfs, dtype=object), len(genes))
            g_col = np.tile(np.array(genes, dtype=object), len(tfs))
            keep = tf_col != g_col
            edges_df = pd.DataFrame({"tf": tf_col[keep], "target": g_col[keep]})
        tfset = set(tfs)
        for v, spec in VARIANTS.items():
            bank = banks[v]
            mask = spec["mask"]
            res = evidence_patterns(bank, tfs, genes, mask=mask, grid=grid, return_dense=not grid)
            ps, dense = res if not grid else (res, None)
            sfx = _suffix(v)
            ps.to_frame().to_parquet(od / f"patterns{sfx}.parquet", index=False)
            np.save(od / f"edge_pattern{sfx}.npy", ps.edge_pattern)
            ps.tf_coverage_frame().to_parquet(od / f"tf_coverage{sfx}.parquet", index=False)
            if dense is not None:
                ev = edges_df.copy()
                for j, s in enumerate(ps.sources):
                    ev[s] = dense[:, j]
                ev.to_parquet(od / f"evidence{sfx}.parquet", index=False)
            sm = ps.summary()
            st = _pattern_stats(ps)
            meta["variants"][v] = {"mask": mask, "annotator_set": spec["sources"], "transform": spec["transform"],
                                   "sources": ps.sources, "family": ps.family,
                                   "n_conditioning_edges": bank.n_cond,
                                   **{k: st[k] for k in ("n_patterns", "n_edges", "n_edges_any_report",
                                                         "n_edges_observed_by_any", "n_edges_unobserved_by_all",
                                                         "n_tfs_covered_by_any", "n_tfs_reported_by_any",
                                                         "mean_annotators_observing")},
                                   "files": [f"patterns{sfx}.parquet", f"edge_pattern{sfx}.npy", f"tf_coverage{sfx}.parquet"] +
                                            ([f"evidence{sfx}.parquet"] if dense is not None else [])}
            if v in LIT_REMOVING_VARIANTS:
                dep, lists = universe_dependence(base_lit, var_lit[v], tfs, genes, panel, grid)
                meta["variants"][v]["universe_dependence"] = dep
                if lists:
                    fn = f"lit_only_via_removed{sfx}.json"
                    C.write_json(lists, od / fn)
                    meta["variants"][v]["files"].append(fn)
            if v in alt_banks:
                res_alt = evidence_patterns(alt_banks[v], tfs, genes, mask=mask, grid=grid, return_dense=not grid)
                ps_alt, dense_alt = res_alt if not grid else (res_alt, None)
                cc_rows.append(cc_rule_comparison(u, v, bank, ps, alt_banks[v], ps_alt, tfs, len(genes),
                                                  dense, dense_alt))
                del res_alt, ps_alt, dense_alt
            sm.insert(0, "mask", mask)
            sm.insert(0, "variant", v)
            sm.insert(0, "universe", u)
            sm["n_tfs_in_source"] = [len(tfset & bank.cov[j]) for j in range(bank.K)]
            sm["n_tfs"], sm["n_genes"], sm["n_patterns"] = len(tfs), len(genes), st["n_patterns"]
            summary_rows.append(sm)
            summary_rows.append(pd.DataFrame([{
                "universe": u, "variant": v, "mask": mask, "annotator": "__ALL__", "family": "", "n_edges": st["n_edges"],
                "n_tfs_in_source": len(tfset & set().union(*bank.cov)),
                "n_tfs_covered": st["n_tfs_covered_by_any"],
                "n_tfs_reported": st["n_tfs_reported_by_any"],
                "n_observed": st["n_edges_observed_by_any"],
                "coverage": st["n_edges_observed_by_any"] / max(st["n_edges"], 1),
                "n_pos": st["n_edges_any_report"],
                "pos_rate": st["n_edges_any_report"] / max(st["n_edges_observed_by_any"], 1),
                "n_patterns": st["n_patterns"], "n_tfs": len(tfs), "n_genes": len(genes),
                "mean_annotators_observing": st["mean_annotators_observing"]}]))
            print(f"  [{u}/{v}] K={ps.patterns.shape[1]} patterns={st['n_patterns']:,} "
                  f"any-report={st['n_edges_any_report']:,} unobserved-by-all={st['n_edges_unobserved_by_all']:,} "
                  f"TFs covered={st['n_tfs_covered_by_any']} ({time.time() - t0:.0f}s)", flush=True)
            del res, ps, dense
        C.write_json(meta, od / "meta.json")
    _write_table(pd.concat(summary_rows, ignore_index=True), tables_dir / SUMMARY_TABLE, SUMMARY_COLS, universes)
    if compare_cc_rules:
        _write_table(pd.concat(cc_rows, ignore_index=True), tables_dir / CC_TABLE, CC_COLS, universes)


# --------------------------------------------------------------------------------------
# Eligibility-coded evidence (informative-observation extension; additive)
# --------------------------------------------------------------------------------------
# The latent-class models with an observation emission (``lcm.MODEL_SPECS['family_hb_obs']``,
# ``['em_obs']``) need to know, for every modelled annotator, whether an unobserved entry was
# *eligible* (the annotator covers the TF but the target is not one of its nodes) or not.
# Codes (``lcm.PatternData(..., eligibility=True)``):
#
#     1 reported, 0 observed but not reported,
#    -1 eligible but not observed                      (modelled annotators only)
#    -2 not eligible, or not observed by an annotator whose observation is not modelled
#
# Modelled annotators = families ``OBS_MODELLED_FAMILIES`` (LIT) whose observation is node based.
# Eligibility of LIT annotator j for edge (tf, g):  ``prespecified*`` masks: tf in cov_j (TF-level
# design: j covers the TF);  ``both_endpoints*`` masks: tf in nodes_j.  j observes (tf, g) iff
# eligible and g in nodes_j, i.e. exactly the stored mask, so the eligibility-coded evidence is a
# refinement of the stored evidence: collapsing -2 to -1 gives back the stored pattern of every
# edge (checked when building).  MOTIF/COEXP observation is not modelled (under the
# curated-conditional masks it is a deterministic function of the held-in literature evidence,
# so it is conditioned on); their unobserved entries are -2.
#
# Files (the stored evidence is never changed; ``sfx`` = ``_suffix(variant) + '__elig'``):
#     patterns{sfx}.parquet     pattern_id, annotator columns (codes above), n_edges, base_pattern
#                               (row of the stored ``patterns{_suffix(variant)}.parquet``)
#     edge_pattern{sfx}.npy     per-edge eligibility pattern (same edge order; -1 = not an edge)
#     meta{sfx}.json            definition, counts per annotator, md5 of the stored base files
# The pseudo-variant ``'<variant>__elig'`` is accepted by ``load_patterns`` / ``load_edge_pattern``
# so that fits stored under ``fits/<universe>/<variant>__elig/<fit_id>`` follow the fit contract
# (``cred.npy`` aligned with ``load_patterns(universe, '<variant>__elig')`` rows).
ELIG_SUFFIX = "__elig"
OBS_MODELLED_FAMILIES = ("LIT",)


def elig_variant(variant: str = PRIMARY_VARIANT) -> str:
    """Name of the eligibility-coded pseudo-variant of ``variant``."""
    return f"{variant}{ELIG_SUFFIX}"


def split_elig_variant(variant: str) -> tuple[str, bool]:
    """('<base variant>', True) for an eligibility-coded pseudo-variant, else (variant, False)."""
    if variant.endswith(ELIG_SUFFIX):
        return variant[: -len(ELIG_SUFFIX)], True
    return variant, False


def _elig_suffix(variant: str) -> str:
    return _suffix(variant) + ELIG_SUFFIX


def _md5(path: Path) -> str:
    import hashlib
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def universe_edge_tfs(universe: str, ev_dir: Path | None = None) -> tuple[np.ndarray, list]:
    """TF index of every stored edge of a universe (``edge_pattern`` order) and the TF list.

    global: TF-major grid over ``tfs.txt`` x ``genes.txt`` (self-pairs included, pattern -1);
    tissue: TF-major over the sorted universe TFs x sorted panel genes, self-pairs excluded."""
    d = Path(ev_dir or EV_DIR) / universe
    if universe == "global":
        tfs = (d / "tfs.txt").read_text().split()
        genes = (d / "genes.txt").read_text().split()
        return np.repeat(np.arange(len(tfs), dtype=np.int64), len(genes)), tfs
    tfs, genes = tissue_universe(universe)
    gs = set(genes)
    return np.repeat(np.arange(len(tfs), dtype=np.int64), [len(genes) - (t in gs) for t in tfs]), tfs


def eligibility_matrix(bank: AnnotatorBank, tfs, mask: str,
                       families: tuple = OBS_MODELLED_FAMILIES) -> tuple[np.ndarray, np.ndarray]:
    """(E (n_tfs, K) bool, modelled (K,) bool): E[t, j] = TF t is eligible for modelled annotator j.

    Modelled = annotators of ``families`` with node-based observation (not genome-wide).
    ``prespecified*`` masks: t in cov_j; ``both_endpoints*`` masks: t in the node set of j."""
    if mask not in MASKS:
        raise ValueError(f"mask must be one of {MASKS}")
    modelled = np.array([(f in families) and not gw for f, gw in zip(bank.family, bank.genome_wide)])
    E = np.zeros((len(tfs), bank.K), dtype=bool)
    if mask.startswith("both_endpoints"):
        nm = bank.node_mask_for(mask)
        ti = bank.target_index(list(tfs))
        ok = ti >= 0
        for j in np.flatnonzero(modelled):
            E[ok, j] = nm[j, ti[ok]]
    else:
        cov = bank.cov_cc if mask.endswith("_cc") else bank.cov
        for j in np.flatnonzero(modelled):
            E[:, j] = [t in cov[j] for t in tfs]
    return E, modelled


def _elig_paths(universe: str, variant: str, ev_dir: Path | None = None) -> dict:
    d = Path(ev_dir or EV_DIR) / universe
    s = _elig_suffix(variant)
    return {"patterns": d / f"patterns{s}.parquet", "edge_pattern": d / f"edge_pattern{s}.npy",
            "meta": d / f"meta{s}.json",
            "base_patterns": d / f"patterns{_suffix(variant)}.parquet",
            "base_edge_pattern": d / f"edge_pattern{_suffix(variant)}.npy"}


def build_elig_patterns(universe: str, variant: str = PRIMARY_VARIANT, ev_dir: Path | None = None,
                        write: bool = True, families: tuple = OBS_MODELLED_FAMILIES) -> dict:
    """Eligibility-coded patterns of a stored universe / variant (see the section comment).

    Derived from the stored base patterns and the per-TF eligibility of the variant's annotator
    bank; the stored files are only read.  Returns dict(R, n, family, annotators, edge_pattern,
    base_pattern, meta) and, with ``write``, writes the three ``__elig`` files."""
    if split_elig_variant(variant)[1]:
        raise ValueError("pass the base variant")
    t0 = time.time()
    paths = _elig_paths(universe, variant, ev_dir)
    R, n, fam, ann = load_patterns(universe, variant, ev_dir)
    ep = np.asarray(load_edge_pattern(universe, variant, mmap=False, ev_dir=ev_dir)).astype(np.int64)
    tfi, tfs = universe_edge_tfs(universe, ev_dir)
    if len(tfi) != len(ep):
        raise AssertionError(f"[{universe}] edge order mismatch: {len(tfi)} vs {len(ep)} edges")
    bank = bank_for_variant(variant)
    if list(bank.sources) != list(ann):
        raise AssertionError(f"[{universe}/{variant}] annotator order differs from the stored evidence")
    mask = VARIANTS[variant]["mask"]
    E, modelled = eligibility_matrix(bank, tfs, mask, families)
    ev, ev_id = np.unique(E[:, modelled], axis=0, return_inverse=True)
    ev_id = ev_id.reshape(-1)
    valid = ep >= 0
    key = ep[valid] * len(ev) + ev_id[tfi[valid]]
    ukey, kinv = np.unique(key, return_inverse=True)
    base, e = ukey // len(ev), ukey % len(ev)
    El = np.zeros((len(ukey), len(ann)), dtype=bool)
    El[:, modelled] = ev[e]
    Rb = R[base]
    if ((Rb >= 0) & modelled[None, :] & ~El).any():
        raise AssertionError(f"[{universe}/{variant}] an observed entry of a modelled annotator is not eligible")
    Rk = np.where(Rb >= 0, Rb, np.where(modelled[None, :] & El, np.int8(-1), np.int8(-2))).astype(np.int8)
    Rn, rinv = np.unique(Rk, axis=0, return_inverse=True)            # canonical (sorted) rows
    rinv = rinv.reshape(-1)
    base_of = np.full(len(Rn), -1, dtype=np.int64)
    base_of[rinv] = base
    if not (base_of[rinv] == base).all():
        raise AssertionError("an eligibility pattern maps to two stored patterns")
    edge_pattern = np.full(len(ep), -1, dtype=np.int32)
    edge_pattern[valid] = rinv[kinv.reshape(-1)].astype(np.int32)
    counts = np.bincount(edge_pattern[valid], minlength=len(Rn)).astype(np.int64)
    # refinement checks: collapsing -2 -> -1 gives the stored pattern; counts add up per base pattern
    if not (np.where(Rn < -1, -1, Rn) == R[base_of]).all():
        raise AssertionError("collapsed eligibility patterns differ from the stored patterns")
    if not np.array_equal(np.bincount(base_of, weights=counts, minlength=len(R)), np.asarray(n, dtype=float)):
        raise AssertionError("eligibility pattern counts do not add up to the stored counts")
    w = counts[:, None]
    per = pd.DataFrame({"annotator": ann, "family": fam, "modelled": modelled,
                        "n_eligible": ((Rn >= -1) * w).sum(0), "n_observed": ((Rn >= 0) * w).sum(0),
                        "n_eligible_unobserved": ((Rn == -1) * w).sum(0), "n_reported": ((Rn == 1) * w).sum(0),
                        "n_tfs_eligible": E.sum(0)})
    per.loc[~per["modelled"], ["n_eligible", "n_eligible_unobserved", "n_tfs_eligible"]] = 0
    meta = {"universe": universe, "variant": variant, "elig_variant": elig_variant(variant), "mask": mask,
            "codes": {"1": "reported", "0": "observed, not reported", "-1": "eligible, not observed (modelled annotators)",
                      "-2": "not eligible, or unobserved entry of an annotator whose observation is not modelled"},
            "modelled_families": list(families), "modelled_annotators": [a for a, m in zip(ann, modelled) if m],
            "eligibility_rule": ("tf in the node set of j" if mask.startswith("both_endpoints")
                                 else "tf covered by j (>= 1 record of the TF in the resource)"),
            "observation_rule": "eligible and target in the node set of j (the stored mask)",
            "n_patterns": int(len(Rn)), "n_base_patterns": int(len(R)), "n_edges": int(counts.sum()),
            "n_eligibility_vectors": int(len(ev)),
            "per_annotator": per.to_dict(orient="records"),
            "base_md5": {"patterns": _md5(paths["base_patterns"]), "edge_pattern": _md5(paths["base_edge_pattern"])},
            "motif_coexp_tf_coverage": CC_TF_COVERAGE, "runtime_s": round(time.time() - t0, 2)}
    out = dict(R=Rn, n=counts, family=list(fam), annotators=list(ann), edge_pattern=edge_pattern,
               base_pattern=base_of, meta=meta)
    if write:
        df = pd.DataFrame(Rn, columns=ann)
        df.insert(0, "pattern_id", np.arange(len(df), dtype=np.int32))
        df["n_edges"] = counts
        df["base_pattern"] = base_of.astype(np.int32)
        df.to_parquet(paths["patterns"], index=False)
        np.save(paths["edge_pattern"], edge_pattern)
        meta["md5"] = {"patterns": _md5(paths["patterns"]), "edge_pattern": _md5(paths["edge_pattern"])}
        C.write_json(meta, paths["meta"])
    return out


def _elig_current(universe: str, variant: str, ev_dir: Path | None = None) -> bool:
    """True if the stored ``__elig`` files exist and were derived from the current base files."""
    p = _elig_paths(universe, variant, ev_dir)
    if not (p["patterns"].exists() and p["edge_pattern"].exists() and p["meta"].exists()):
        return False
    m = json.loads(p["meta"].read_text())
    return (m.get("base_md5", {}).get("patterns") == _md5(p["base_patterns"])
            and m.get("base_md5", {}).get("edge_pattern") == _md5(p["base_edge_pattern"])
            and m.get("md5", {}).get("patterns") == _md5(p["patterns"]))


def ensure_elig_patterns(universe: str, variant: str = PRIMARY_VARIANT, ev_dir: Path | None = None,
                         build_missing: bool = True) -> dict:
    """Paths of up-to-date ``__elig`` files (built if missing or stale and ``build_missing``)."""
    load_meta(universe, ev_dir)
    if not _elig_current(universe, variant, ev_dir):
        if not build_missing:
            raise FileNotFoundError(f"eligibility-coded evidence for {universe}/{variant} missing or stale: "
                                    "run build_elig_patterns")
        build_elig_patterns(universe, variant, ev_dir)
    return _elig_paths(universe, variant, ev_dir)


def load_patterns_elig(universe: str, variant: str = PRIMARY_VARIANT, ev_dir: Path | None = None,
                       build_missing: bool = True, with_base: bool = False):
    """Eligibility-coded (R, n, family, annotators) for ``lcm.PatternData(..., eligibility=True)``
    (``with_base``: also the stored base-pattern row of every eligibility pattern)."""
    p = ensure_elig_patterns(universe, variant, ev_dir, build_missing)
    meta = load_meta(universe, ev_dir)
    df = pd.read_parquet(p["patterns"])
    ann = meta["variants"][variant]["sources"]
    fam = meta["variants"][variant]["family"]
    out = (df[ann].to_numpy(np.int8), df["n_edges"].to_numpy(np.int64), fam, ann)
    return out + (df["base_pattern"].to_numpy(np.int64),) if with_base else out


def load_edge_pattern_elig(universe: str, variant: str = PRIMARY_VARIANT, mmap: bool = True,
                           ev_dir: Path | None = None, build_missing: bool = True) -> np.ndarray:
    p = ensure_elig_patterns(universe, variant, ev_dir, build_missing)
    return np.load(p["edge_pattern"], mmap_mode="r" if mmap else None)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--universes", nargs="+", default=["immune", "kidney", "lung", "global"],
                    choices=["immune", "kidney", "lung", "global"])
    ap.add_argument("--out-dir", type=Path, default=EV_DIR, help="evidence output directory (default outputs/r2/evidence)")
    ap.add_argument("--tables-dir", type=Path, default=C.TABLES, help="summary table directory (default data/r2/tables)")
    ap.add_argument("--compare-cc-coverage", action="store_true",
                    help="also write the before/after table of the reverted within-conditioning-set option")
    args = ap.parse_args()
    build(args.universes, out_dir=args.out_dir, tables_dir=args.tables_dir, compare_cc_rules=args.compare_cc_coverage)


if __name__ == "__main__":
    main()
