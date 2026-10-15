"""Held-out ChIP references with provenance (stage D3, ``outputs/r2/heldout_chip``).

The rebuilt study keeps three ChIP-seq-derived TF-target references *out of
every fit, tuning step and universe definition* and uses them only for
validation (REVISION_PLAN_R2.md section 7, decision 8):

1. **ENCODE TFBS, five cell lines** (``encode_tfbs``, the permissive reference).
   Source: Harmonizome (Rouillard et al. 2016, Database baw100), dataset
   "ENCODE Transcription Factor Binding Site Profiles" (download folder
   ``encodetf``), crisp gene-set library ``gene_set_library_crisp.gmt.gz``
   (1,679 ChIP-seq profiles named ``Factor_CellLine_Genome_Replicate``; the
   dataset page lists 2015-04-06 as last update, the file is served with
   Last-Modified 2020-08-03).  Target assignment: a gene is a target of a
   profile when the factor binds near it.  Harmonizome's published processing
   scripts (a 2016 version) keep peaks within 5 kb of the nearest gene, weight
   them by distance (half-maximum at 2 kb), then standardise and threshold; the
   served library still contains per-replicate and histone profiles, which
   those scripts would merge or drop, so the exact rule used for this file
   version cannot be verified.  We keep the hg19 profiles from the five ENCODE
   Tier-1/2 cell lines (GM12878, H1-hESC, HeLa-S3, HepG2, K562) and take, per
   factor, the union over cell lines and replicates.

2. **ENCODE-strict** (``encode_strict``): the subset of (1) supported in
   ``>= C.PRESPECIFIED['encode_strict_min_cell_lines']`` of the five cell lines,
   restricted to TFs assayed in at least that many lines (so that a TF is never
   penalised for having been assayed in a single line).  Same gene universe as (1).

3. **ENETS2** (``enets2``, the specific reference): the filtered proximal
   TF-target network of Gerstein et al. (2012, Nature 489:91-100), file
   ``enets2.Proximal_filtered.txt`` from http://encodenets.gersteinlab.org/
   (Last-Modified 2012-03-06).  Binding within +/-2.5 kb of an annotated TSS
   (raw network ``enets1.Proximal_raw``) filtered with the TIP probabilistic
   model (Cheng et al. 2011) to 26,070 high-confidence edges; 119 factors and
   458 ChIP-seq experiments of the ENCODE January-2011 freeze, mainly in K562,
   GM12878, HepG2, HeLa-S3 and H1-hESC plus other lines for some factors
   (e.g. ESR1, GATA3, NR3C1; see ``provenance.json``).
   Two factor labels of the distributed file are corrected
   (``ENETS2_RELABEL``): "NFKB1" is the ENCODE "NFKB" ChIP-seq whose antibody
   (Santa Cruz sc-372) targets NF-kB p65 = RELA, and "TCF4" is the old alias of
   TCF7L2 (ENCODE antibody term "TCF4", deprecated as "same as TCF7L2").  The
   relabel is applied only if it is confirmed by (i) target overlap: the most
   significant Harmonizome hg19 profile (hypergeometric test within the shared
   gene universe) belongs to the proposed factor and Harmonizome has no profile
   of the distributed label, and (ii) ENCODE antibody metadata: the UCSC ENCODE
   controlled-vocabulary entry of the antibody names the proposed factor and
   no ENCODE uniform-TFBS experiment targets the distributed label.  An
   unconfirmed relabel drops the factor instead.  Gerstein et al.'s
   supplementary table S4A is checked as well (it lists TCF7L2, not TCF4,
   with the same number of targets).

Cleaning (all references): factor names are stripped of epitope-tag prefixes
(``eGFP-``, ``HA-``), TF and target symbols are HGNC-normalised with
``common.normalize_series``, only factors in the Lambert et al. (2018) TF list
are kept, self-loops are dropped and edges are de-duplicated.

Evaluable TFs (``<ref>_assayed_tfs.txt`` and ``assayed_tfs.tsv``): ENCODE =
Lambert TFs with >= 1 non-histone profile in the five lines; ENCODE-strict =
those assayed in >= 2 of the lines; ENETS2 = Lambert TFs with >= 1 edge in
the *filtered* network (after relabelling).  Lambert TFs that appear only in
the raw proximal network (JUNB, NR4A1, THAP1, ZNF143) have no filtered
release and are therefore not evaluable; they are listed in
``dropped_factors.tsv``.  Every dropped, excluded or relabelled factor is
listed there with its reason.

Gene universes (genes that could have been called a target,
``<ref>_gene_universe.txt``): ENCODE and ENCODE-strict = genes bound by >= 1
non-histone profile of the five cell lines; ENETS2 = genes with >= 1 proximal
peak of any factor in the raw network (``enets1.Proximal_raw``) or in the
filtered network (16% of ENETS2 edges are not in enets1).

Target-degree null (prespecified, ``C.PRESPECIFIED['target_degree_*']``):
``target_degree_definition = 'leave_pair_out_ref_indegree'`` -- the degree of
gene g for the pair (TF, g) is the number of *other* assayed reference TFs
binding g; ``target_degree_bin_rule = 'union_gene_edge_deciles'`` with
``target_degree_bins = 10`` -- bin boundaries are the 10-quantiles of that
in-degree over universe genes united with the 10-quantiles over reference
edges; the null is the additive logistic model logit P = a_TF + b_bin fitted
to all assayed-TF x universe-gene pairs.  ``reference_index`` and
``target_degree_model`` (plus ``degree_bin`` / ``null_probability``) are the
single implementation; the held-out evaluation stage imports them.

The script also (a) reproduces the legacy files that the R1 analysis used
(``common.ENCODE_TFBS`` and ``common.ENETS2``) from the official downloads and
records whether they match exactly; (b) quantifies the overlap of every
held-in annotator with each held-out reference (``annotator_overlap.csv``)
against three expectations: TF-matched (each TF's own target density),
target-matched (the target-degree null above) and TF-matched inside the
plan's gene set G (``outputs/r2/evidence/global/genes.txt``); pooled rows
cover each modality family, all held-in sources, the two literature
sensitivity groups ``all_held_in_no_ht`` (the 9 high-throughput-like LIT
regulons of the annotator stage removed, as in evidence variant ``lit_no_ht``)
and ``all_held_in_no_chipderived`` (every record of
``C.PRESPECIFIED['lit_no_chipderived_drop']`` removed, as in evidence variant
``lit_no_chipderived``), each split by co-support from held-out ChIP sources;
rows are flagged when the annotator, or the selection that defines the row,
shares ENCODE experiments with the reference, and when support is too small
for a stable lift (``low_support``); the same is done per OmniPath primary
resource (``omnipath_source_overlap.csv``); (c) counts evaluable TFs,
evaluable edges and positives of each reference inside the three tissue
context universes and the global universe (``heldout_chip_coverage.csv``);
and (d) checks every ENETS2 regulon against the Harmonizome profiles of the
same factor (``enets2_label_concordance.tsv``; documented, nothing dropped).

Usage::

    .venv/bin/python subproject_24_multireference_latent_truth_model/src/r2/build_heldout_chip.py \
        [--download] [--wait-annotators-min 0] [--n-boot 1000] [--allow-partial]

Every upstream input (annotator edges and sources.json, the three tissue
universes, the global universe) is required; ``--allow-partial`` downgrades a
missing input to a warning (exploratory runs only).

``--table-only`` rewrites the three manuscript tables (heldout_chip_table,
heldout_chip_coverage, heldout_chip_annotator_overlap) from the stored
``provenance.json`` and ``annotator_overlap.csv`` of the last full run, reading
no raw file (``reproduce_all.py --tables-only`` without the external data).

All raw inputs live under ``C.EXT/external_data/heldout_chip`` (downloaded with
``--download`` when missing; every download is checked against an expected
md5).  Deterministic: bootstrap seeded with ``C.PRESPECIFIED['random_seed']``,
BLAS limited to one thread during model fits (``threadpoolctl``) and floats
written with 12 significant digits, so reruns are byte-identical also when the
module is imported after numpy.
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")   # small dense solves only; set before numpy is imported

import argparse  # noqa: E402
import ast
import contextlib
import gzip
import hashlib
import json
import re
import struct
import sys
import time
import urllib.request
import warnings
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

try:   # pins BLAS to one thread even when numpy was imported before this module
    from threadpoolctl import threadpool_limits as _threadpool_limits
except ImportError:  # pragma: no cover - threadpoolctl ships with scikit-learn
    _threadpool_limits = None

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from r2 import common as C  # noqa: E402

# ----------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------
RAW_DIR = C.EXT / "external_data" / "heldout_chip"
OUT_DIR = C.OUT / "heldout_chip"
ANNOTATOR_DIR = C.OUT / "annotators"
ANNOTATOR_EDGES = ANNOTATOR_DIR / "annotator_edges.parquet"
ANNOTATOR_SOURCES = ANNOTATOR_DIR / "sources.json"        # family / held_out per source
ANNOTATOR_PROVENANCE = ANNOTATOR_DIR / "provenance.json"  # OmniPath cache used, PAZAR split
GLOBAL_DIR = C.OUT / "evidence" / "global"                # global universe: tfs.txt x genes.txt (= G)
LEGACY_SP53 = C.EXT / "subproject_53_scgpt_gpl_replication"
LEGACY_SP53_DIR = LEGACY_SP53 / "data" / "encode"
LEGACY_SP53_SCRIPTS = {   # related 'sequence-specific' filters of the GPL-paper replication
    "phase1b_enets2_evaluation.py": "CHROMATIN_REGULATORS",
    "phase1c_enets2_original.py": "EXCLUDE_TFS",
}

HARMONIZOME_BASE = "https://maayanlab.cloud/static/hdfs/harmonizome/data/encodetf/"
ENCODENETS_BASE = "http://encodenets.gersteinlab.org/"
UCSC_CV = "https://genome.ucsc.edu/cgi-bin/hgEncodeVocab?ra=encode/cv.ra&type=Antibody&term="
# expected_md5 is asserted for every file; CGI pages carry a per-request nonce, so for
# them the md5 of the extracted vocabulary-table text ('text_md5') is asserted instead
RAW_FILES = {
    "harmonizome_encode_tfbs_gmt": {
        "url": HARMONIZOME_BASE + "gene_set_library_crisp.gmt.gz",
        "path": RAW_DIR / "harmonizome_encodetf" / "gene_set_library_crisp.gmt.gz",
        "server_last_modified": "Mon, 03 Aug 2020 19:33:31 GMT",
        "expected_md5": "4f5f0ad8eaa1c04a70fe170cc79b6da7",
    },
    "harmonizome_encode_tfbs_attribute_list": {
        "url": HARMONIZOME_BASE + "attribute_list_entries.txt.gz",
        "path": RAW_DIR / "harmonizome_encodetf" / "attribute_list_entries.txt.gz",
        "server_last_modified": "Mon, 03 Aug 2020 19:33:31 GMT",
        "expected_md5": "9bd806060236926f2eb1d46839f733df",
    },
    "harmonizome_encode_tfbs_processing_scripts": {
        "url": HARMONIZOME_BASE + "processing_scripts.tar.gz",
        "path": RAW_DIR / "harmonizome_encodetf" / "processing_scripts.tar.gz",
        "server_last_modified": "Mon, 03 Aug 2020 19:34:02 GMT",
        "expected_md5": "06a0627db1bb9e146503989797a13aab",
    },
    "enets2_proximal_filtered": {
        "url": ENCODENETS_BASE + "enets2.Proximal_filtered.txt",
        "path": RAW_DIR / "encodenets_gerstein2012" / "enets2.Proximal_filtered.txt",
        "server_last_modified": "Tue, 06 Mar 2012 23:55:27 GMT",
        "expected_md5": "ca9c6e580577fb14e7158c2a7c9bc526",
    },
    "enets1_proximal_raw": {
        "url": ENCODENETS_BASE + "enets1.Proximal_raw.txt",
        "path": RAW_DIR / "encodenets_gerstein2012" / "enets1.Proximal_raw.txt",
        "server_last_modified": None,
        "expected_md5": "ed88f4a2cb40bcd3c6ae953c868561c8",
    },
    "gerstein2012_table_S4": {   # supplementary table S4 (factor-level statistics)
        "url": ENCODENETS_BASE + "enets13.S4_Excel_tables.zip",
        "path": RAW_DIR / "encodenets_gerstein2012" / "enets13.S4_Excel_tables.zip",
        "server_last_modified": "Mon, 14 May 2012 14:07:39 GMT",
        "expected_md5": "d43a5421ff6dda7fc47df147f9b4690f",
    },
    "ucsc_encode_awg_tfbs_inputs": {   # ENCODE uniform TFBS: experiment -> antibody -> HGNC target
        "url": "http://hgdownload.soe.ucsc.edu/goldenPath/hg19/encodeDCC/wgEncodeRegTfbsClustered/"
               "wgEncodeRegTfbsClusteredInputsV3.tab.gz",
        "path": RAW_DIR / "ucsc_encode" / "wgEncodeRegTfbsClusteredInputsV3.tab.gz",
        "server_last_modified": "Fri, 07 Jun 2013 23:58:36 GMT",
        "expected_md5": "d151e39d961c256bda35bfe99439ce81",
    },
    "ucsc_encode_cv_antibody_NFKB": {
        "url": UCSC_CV + "NFKB",
        "path": RAW_DIR / "ucsc_encode" / "cv_antibody_NFKB.html",
        "server_last_modified": None, "expected_md5": None,
        "expected_text_md5": "f465d5f0c4128b1a130703c3f976615b",
    },
    "ucsc_encode_cv_antibody_TCF4": {
        "url": UCSC_CV + "TCF4",
        "path": RAW_DIR / "ucsc_encode" / "cv_antibody_TCF4.html",
        "server_last_modified": None, "expected_md5": None,
        "expected_text_md5": "49078aaebdd64cc5bf297826495a316f",
    },
}

FIVE_CELL_LINES = ("GM12878", "H1-hESC", "HeLa-S3", "HepG2", "K562")
TAG_PREFIXES = ("eGFP-", "HA-")
# histone marks present as "factors" in the Harmonizome ENCODE library
HISTONE_PREFIXES = ("H2A", "H2B", "H3K", "H4K", "H3AC", "H4AC")
ENCODE_STRICT_K = int(C.PRESPECIFIED["encode_strict_min_cell_lines"])

# label errors in the distributed ENETS2 file (REVISION_PLAN_R2.md section 7, decision 8):
# distributed label -> (factor actually assayed, ENCODE antibody term of the experiment)
ENETS2_RELABEL = {
    "NFKB1": {"to": "RELA", "encode_antibody": "NFKB"},
    "TCF4": {"to": "TCF7L2", "encode_antibody": "TCF4"},
}
RELABEL_TOP_PROFILES = 5          # profiles recorded per relabel check

# target-matched expectation (prespecified before any held-out evaluation, common.PRESPECIFIED):
# additive logistic model logit p = a_TF + b_bin(target degree), target degree = number of other
# assayed reference TFs binding the gene; bin boundaries = quantiles of the in-degree over genes
# united with the quantiles over reference edges
TARGET_DEGREE_BINS = int(C.PRESPECIFIED["target_degree_bins"])
TARGET_DEGREE_BIN_RULE = str(C.PRESPECIFIED["target_degree_bin_rule"])
TARGET_DEGREE_DEFINITION = str(C.PRESPECIFIED["target_degree_definition"])
SUPPORTED_BIN_RULES = ("union_gene_edge_deciles",)
SUPPORTED_DEGREE_DEFINITIONS = ("leave_pair_out_ref_indegree",)
MODEL_MAX_ITER = 500
MODEL_GRAD_TOL = 1e-6            # max |observed - fitted| margin (edge counts) at convergence

# overlap-table rows with fewer evaluable TFs or reference positives than this are marked
# 'low_support' (bootstrap CIs over a handful of TFs are unstable); descriptive only
MIN_SUPPORT_TFS = 10
MIN_SUPPORT_POSITIVES = 20
CSV_FLOAT_FORMAT = "%.12g"       # 12 significant digits: stable across BLAS thread counts

# ENETS2 regulon-label concordance (documented, nothing dropped): a regulon is flagged when
# no Harmonizome profile of the same factor overlaps it at Bonferroni-adjusted p < 0.05
LABEL_CHECK_ALPHA = 0.05

REFERENCES = {
    "encode_tfbs": {
        "label": "ENCODE TFBS (Harmonizome, 5 cell lines)",
        "provenance": (
            "Harmonizome 'ENCODE Transcription Factor Binding Site Profiles' crisp gene-set "
            "library (encodetf/gene_set_library_crisp.gmt.gz); hg19 profiles of GM12878, "
            "H1-hESC, HeLa-S3, HepG2, K562; union over cell lines and replicates; target = gene "
            "with a nearby binding peak as assigned by Harmonizome (its published 2016 processing "
            "scripts use nearest gene within 5 kb, distance weighting, standardisation and a "
            "threshold; the exact rule of the served 2015 file version is not verifiable)"),
        "url": HARMONIZOME_BASE + "gene_set_library_crisp.gmt.gz",
        "version": "Harmonizome dataset updated 2015-04-06 (file Last-Modified 2020-08-03); accessed 2026-10-07",
        "citation": "ENCODE Project Consortium (2011) PLoS Biol 9:e1001046; Rouillard et al. (2016) Database baw100",
        "cell_lines": ";".join(FIVE_CELL_LINES),
        "specificity": "permissive",
    },
    "encode_strict": {
        "label": f"ENCODE-strict (>= {ENCODE_STRICT_K} of 5 cell lines)",
        "provenance": (
            f"ENCODE TFBS (above) restricted to edges bound in >= {ENCODE_STRICT_K} of the five cell "
            f"lines and to TFs assayed in >= {ENCODE_STRICT_K} of them"),
        "url": HARMONIZOME_BASE + "gene_set_library_crisp.gmt.gz",
        "version": "derived from the ENCODE TFBS file above",
        "citation": "ENCODE Project Consortium (2011) PLoS Biol 9:e1001046; Rouillard et al. (2016) Database baw100",
        "cell_lines": ";".join(FIVE_CELL_LINES),
        "specificity": "intermediate",
    },
    "enets2": {
        "label": "ENETS2 (Gerstein et al. 2012 proximal filtered)",
        "provenance": (
            "encodenets.gersteinlab.org enets2.Proximal_filtered.txt: TF binding within "
            "+/-2.5 kb of annotated TSSs (raw proximal network, enets1) filtered with the "
            "TIP probabilistic model; labels NFKB1->RELA and TCF4->TCF7L2 corrected"),
        "url": ENCODENETS_BASE + "enets2.Proximal_filtered.txt",
        "version": "file Last-Modified 2012-03-06 (Nature 2012 supplementary data); accessed 2026-10-07",
        "citation": "Gerstein et al. (2012) Nature 489:91-100; Cheng et al. (2011) Bioinformatics 27:3221-3227 (TIP)",
        "cell_lines": None,   # filled from the ENCODE uniform-TFBS metadata at run time
        "specificity": "specific",
    },
}

# OmniPath primary-resource label -> evidence family for the OmniPath fallback table; labels
# that are annotator source ids take the family recorded in the annotator stage's sources.json
OMNIPATH_FAMILY = {
    "ReMap": "CHIP", "ENCODE-proximal": "CHIP", "ENCODE-distal": "CHIP", "PAZAR-HT": "CHIP",
    "HOCOMOCO": "MOTIF", "JASPAR": "MOTIF",
    "ARACNe-GTEx": "COEXP",
    "DoRothEA": "AGGREGATE", "DoRothEA-A": "AGGREGATE", "CollecTRI": "AGGREGATE",
}
PAZAR_HT_MIN_TARGETS = 1000   # annotator stage: PAZAR regulons > 1000 targets = PAZAR-HT (CHIP)

# held-out ChIP sources whose experiments overlap the ENCODE experiments behind the references:
# ReMap aggregates public ChIP-seq including ENCODE; OmniPath's ENCODE-proximal/-distal are the
# Gerstein et al. (2012) ENCODE networks (ENETS2 itself = ENCODE-proximal)
ENCODE_SHARING_SOURCES = ("ReMap", "ENCODE-proximal", "ENCODE-distal")
GERSTEIN_SOURCES = ("ENCODE-proximal", "ENCODE-distal")
# co-support splits of the held-in pooled rows: tag -> held-out CHIP sources used for selection
# (None = every held-out CHIP source); 'chip_remap_pazarht' leaves out the Gerstein networks
CHIP_SPLITS = {"chip_any": None, "chip_remap_pazarht": ("ReMap", "PAZAR-HT")}
CONSERVATIVE_HEADLINE = "all_held_in|without_chip_remap_pazarht"
# OmniPath aggregate resources whose records include held-out ChIP evidence (DoRothEA levels
# combine curated, ReMap ChIP-seq, motif and coexpression support)
AGGREGATES_WITH_CHIP = ("DoRothEA", "DoRothEA-A")
FLAG_UNINFORMATIVE = ("uninformative by construction: OmniPath ENCODE-proximal = ENETS2 membership "
                      "(the without-split has no ENETS2 positives)")
FLAG_SAME_NETWORK = "same source network (Gerstein et al. 2012); distributed labels NFKB1/TCF4 not corrected"
FLAG_SHARES = "shares ENCODE ChIP-seq experiments with the reference"
FLAG_SELECTED = ("selected on co-support from held-out ChIP ({}) that shares ENCODE ChIP-seq experiments "
                 "with the reference; not independent validation")
FLAG_AGGREGATE = ("aggregate includes held-out ChIP-seq evidence (ReMap) that shares ENCODE ChIP-seq "
                  "experiments with the reference")
FLAG_LOW = (f"low support (< {MIN_SUPPORT_TFS} evaluable TFs or < {MIN_SUPPORT_POSITIVES} reference "
            "positives); lift and CI unstable")


# ----------------------------------------------------------------------------
# Raw files and provenance
# ----------------------------------------------------------------------------


def md5sum(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def sha256sum(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def cv_table_cells(html: str) -> list:
    """Text of the <TD> cells of a UCSC hgEncodeVocab page (stable across requests)."""
    cells = re.findall(r"<TD[^>]*>(.*?)</TD>", html, flags=re.S | re.I)
    out = []
    for c in cells:
        t = re.sub(r"<[^>]+>", " ", c)
        t = re.sub(r"\s+", " ", t.replace("&nbsp;", " ")).strip()
        out.append(t)
    return out


def text_md5(path: Path) -> str:
    return hashlib.md5("\n".join(cv_table_cells(path.read_text(errors="replace"))).encode()).hexdigest()


def ensure_raw(download: bool) -> dict:
    """Check (and optionally download) every raw file against its expected md5."""
    log_path = RAW_DIR / "download_log.json"
    log = json.loads(log_path.read_text()) if log_path.exists() else {}
    records = {}
    for key, spec in RAW_FILES.items():
        p = spec["path"]
        if not p.exists():
            if not download:
                raise FileNotFoundError(f"{p} missing; rerun with --download")
            C.ensure_dir(p.parent)
            print(f"[download] {spec['url']}")
            req = urllib.request.Request(spec["url"], headers={"User-Agent": "biodyn-r2/1.0"})
            with urllib.request.urlopen(req, timeout=600) as r, open(p, "wb") as fh:
                lm = r.headers.get("Last-Modified")
                fh.write(r.read())
            log[key] = {"retrieved_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                        "server_last_modified": lm}
        if key not in log:
            mtime = datetime.fromtimestamp(p.stat().st_mtime, timezone.utc)
            log[key] = {"retrieved_utc": mtime.isoformat(timespec="seconds"),
                        "server_last_modified": spec["server_last_modified"]}
        rec = {"url": spec["url"], "path": str(p), "bytes": p.stat().st_size, "md5": md5sum(p), **log[key]}
        if spec.get("expected_md5"):
            rec["expected_md5"] = spec["expected_md5"]
            if rec["md5"] != spec["expected_md5"]:
                raise RuntimeError(f"{p}: md5 {rec['md5']} != expected {spec['expected_md5']}")
        if spec.get("expected_text_md5"):
            rec["text_md5"] = text_md5(p)
            rec["expected_text_md5"] = spec["expected_text_md5"]
            rec["note"] = "dynamic CGI page (per-request nonce); md5 of the vocabulary-table text is checked"
            if rec["text_md5"] != spec["expected_text_md5"]:
                raise RuntimeError(f"{p}: table-text md5 {rec['text_md5']} != expected "
                                   f"{spec['expected_text_md5']}")
        records[key] = rec
    C.write_json(log, log_path)
    return records


# ----------------------------------------------------------------------------
# Minimal BIFF8 (.xls) reader for the Gerstein et al. supplementary tables
# (no xlrd in the analysis environment; verified cell-by-cell against xlrd 2.0.1)
# ----------------------------------------------------------------------------


def _cfb_stream(data: bytes, name: str) -> bytes:
    """One stream of an OLE2 compound file (streams stored in the regular FAT only)."""
    if data[:8] != bytes.fromhex("D0CF11E0A1B11AE1"):
        raise ValueError("not an OLE2 compound file")
    ssz = 1 << struct.unpack_from("<H", data, 0x1E)[0]
    n_fat, dir_start = struct.unpack_from("<II", data, 0x2C)
    cutoff = struct.unpack_from("<I", data, 0x38)[0]
    dif_start, n_dif = struct.unpack_from("<II", data, 0x44)
    fat_secs = list(struct.unpack_from("<109I", data, 0x4C))
    s = dif_start
    for _ in range(n_dif):
        vals = struct.unpack_from(f"<{ssz // 4}I", data, (s + 1) * ssz)
        fat_secs += vals[:-1]
        s = vals[-1]
    fat = []
    for fs in fat_secs[:n_fat]:
        fat += struct.unpack_from(f"<{ssz // 4}I", data, (fs + 1) * ssz)

    def chain(start):
        out, s, seen = [], start, set()
        while s < 0xFFFFFFFA and s not in seen:
            seen.add(s)
            out.append(data[(s + 1) * ssz:(s + 2) * ssz])
            s = fat[s]
        return b"".join(out)

    d = chain(dir_start)
    for off in range(0, len(d), 128):
        nlen = struct.unpack_from("<H", d, off + 0x40)[0]
        if d[off:off + max(nlen - 2, 0)].decode("utf-16-le") == name:
            start, size = struct.unpack_from("<II", d, off + 0x74)
            if size < cutoff:
                raise NotImplementedError("mini-stream storage not supported")
            return chain(start)[:size]
    raise KeyError(name)


def _rk(v: int) -> float:
    if v & 2:
        x = float((v >> 2) - (1 << 30) if v & 0x80000000 else v >> 2)
    else:
        x = struct.unpack("<d", struct.pack("<Q", (v & 0xFFFFFFFC) << 32))[0]
    return x / 100 if v & 1 else x


def read_xls_sheet(data: bytes, sheet: int = 0) -> dict:
    """Cells {(row, col): str | float} of one worksheet of a BIFF8 workbook."""
    wb = _cfb_stream(data, "Workbook")
    recs, pos = [], 0
    while pos + 4 <= len(wb):
        rt, ln = struct.unpack_from("<HH", wb, pos)
        recs.append((rt, wb[pos + 4:pos + 4 + ln]))
        pos += 4 + ln
    sst = []
    i = next((j for j, r in enumerate(recs) if r[0] == 0x00FC), None)
    if i is not None:                        # shared strings: SST + CONTINUE records
        segs, j = [recs[i][1]], i + 1
        while j < len(recs) and recs[j][0] == 0x003C:
            segs.append(recs[j][1])
            j += 1
        st = {"seg": 0, "p": 8}

        def take(n):
            b = segs[st["seg"]][st["p"]:st["p"] + n]
            st["p"] += n
            return b

        for _ in range(struct.unpack_from("<I", segs[0], 4)[0]):
            if st["p"] >= len(segs[st["seg"]]):
                st["seg"], st["p"] = st["seg"] + 1, 0
            cch = struct.unpack("<H", take(2))[0]
            flags = take(1)[0]
            n_run = struct.unpack("<H", take(2))[0] if flags & 0x08 else 0
            n_ext = struct.unpack("<I", take(4))[0] if flags & 0x04 else 0
            chars, wide, left = [], flags & 0x01, cch
            while left:
                avail = (len(segs[st["seg"]]) - st["p"]) // (2 if wide else 1)
                if avail == 0:                 # string continues in the next CONTINUE record
                    st["seg"], st["p"] = st["seg"] + 1, 0
                    wide = take(1)[0] & 0x01
                    continue
                k = min(left, avail)
                chars.append(take(k * (2 if wide else 1)).decode("utf-16-le" if wide else "latin-1"))
                left -= k
            skip = 4 * n_run + n_ext
            while skip:
                k = min(skip, len(segs[st["seg"]]) - st["p"])
                if k == 0:
                    st["seg"], st["p"] = st["seg"] + 1, 0
                    continue
                take(k)
                skip -= k
            sst.append("".join(chars))
    cells, n_ws, cur = {}, -1, None
    for rt, b in recs:
        if rt == 0x0809:                       # BOF: worksheet substreams have type 0x0010
            if struct.unpack_from("<H", b, 2)[0] == 0x0010:
                n_ws += 1
                cur = n_ws
            else:
                cur = None
            continue
        if rt == 0x000A:                       # EOF
            cur = None
            continue
        if cur != sheet:
            continue
        if rt == 0x00FD:                       # LABELSST
            r, c, _, k = struct.unpack_from("<HHHI", b)
            cells[(r, c)] = sst[k]
        elif rt == 0x0203:                     # NUMBER
            r, c, _, x = struct.unpack_from("<HHHd", b)
            cells[(r, c)] = x
        elif rt == 0x027E:                     # RK
            r, c, _, v = struct.unpack_from("<HHHI", b)
            cells[(r, c)] = _rk(v)
        elif rt == 0x00BD:                     # MULRK
            r, c0 = struct.unpack_from("<HH", b)
            for q in range((len(b) - 6) // 6):
                cells[(r, c0 + q)] = _rk(struct.unpack_from("<I", b, 4 + 6 * q + 2)[0])
    return cells


def read_gerstein_s4a(zip_path: Path) -> pd.DataFrame:
    """Gerstein et al. (2012) table S4A: factor -> number of (proximal) targets."""
    with zipfile.ZipFile(zip_path) as z:
        cells = read_xls_sheet(z.read("S4A.xls"))
    hdr = next(r for (r, c), v in sorted(cells.items()) if c == 0 and v == "Factor")
    col = next(c for (r, c), v in cells.items() if r == hdr and v == "Targets")
    rows = sorted({r for r, _ in cells if r > hdr})
    out = [(str(cells[(r, 0)]).strip(), cells.get((r, col))) for r in rows
           if isinstance(cells.get((r, 0)), str) and cells[(r, 0)].strip()]
    return pd.DataFrame(out, columns=["factor", "n_targets"])


# ----------------------------------------------------------------------------
# Parsers
# ----------------------------------------------------------------------------


def read_harmonizome_gmt(path: Path) -> pd.DataFrame:
    """One row per profile: profile, factor_raw, cell_line, genome, replicate, genes (list).

    Profile names follow Harmonizome's ``Factor_CellLine_Genome_Replicate``.
    """
    rows = []
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt") as fh:
        for line in fh:
            f = line.rstrip("\n").split("\t")
            term, factor, genes = f[0], f[1], [g for g in f[2:] if g]
            parts = term.split("_")
            rows.append((term, factor, "_".join(parts[1:-2]), parts[-2], parts[-1], genes))
    return pd.DataFrame(rows, columns=["profile", "factor_raw", "cell_line", "genome",
                                       "replicate", "genes"])


def explode_profiles(prof: pd.DataFrame) -> pd.DataFrame:
    """Long format (one row per profile x gene) with categorical columns."""
    n = prof["genes"].map(len).to_numpy()
    idx = np.repeat(np.arange(len(prof)), n)
    genes = np.concatenate([np.asarray(g, dtype=object) for g in prof["genes"]])
    out = pd.DataFrame({c: pd.Categorical(prof[c].to_numpy()[idx])
                        for c in ("profile", "factor_raw", "cell_line")})
    out["target_raw"] = pd.Categorical(genes)
    return out


def normalize_categorical(s: pd.Series) -> pd.Series:
    """HGNC-normalise a categorical/object series via its unique values."""
    u = pd.Series(pd.unique(s.astype(str)))
    m = dict(zip(u, C.normalize_series(u)))
    return s.astype(str).map(m)


def read_encodenets(path: Path) -> pd.DataFrame:
    # keep_default_na=False: the literal symbol 'NA' (one raw row) must not become NaN
    return pd.read_csv(path, sep=r"\s+", header=None, names=["factor_raw", "edge_type", "target_raw"],
                       dtype=str, engine="python", keep_default_na=False, na_values=[])


def read_awg_inputs(path: Path) -> pd.DataFrame:
    """ENCODE uniform TFBS (AWG, Jan-2011 freeze) inputs: experiment, HGNC target, antibody, cell."""
    df = pd.read_csv(path, sep="\t", header=None, dtype=str, keep_default_na=False,
                     names=["table", "label", "target", "antibody", "cell_line", "treatment", "lab"])
    df["target_norm"] = C.normalize_series(df["target"])
    return df


def parse_cv_antibody(path: Path) -> dict:
    """UCSC ENCODE controlled-vocabulary antibody entry: term, description, target, vendor."""
    html = path.read_text(errors="replace")
    cells = cv_table_cells(html)
    gene = re.findall(r"GeneCard:([A-Za-z0-9\-]+)", " ".join(cells))
    desc = next((c for c in cells if "Antibody Target" in c), "")
    vendor = next((c for c in cells if re.search(r"(Santa Cruz|Cell Signaling|Abcam|Millipore)", c)), "")
    return {"description": desc, "genecard_target": gene[0].upper() if gene else "",
            "vendor": vendor}


def clean_factor(name: str) -> str:
    s = str(name).strip()
    for p in TAG_PREFIXES:
        if s.startswith(p):
            s = s[len(p):]
    return C.normalize_symbol(s)


def is_histone(name: str) -> bool:
    u = str(name).upper()
    return u.startswith(HISTONE_PREFIXES) or u in {"H2AFZ", "H2AZ1"}


# ----------------------------------------------------------------------------
# Builders
# ----------------------------------------------------------------------------


def build_encode(gmt: pd.DataFrame, tfs: frozenset) -> dict:
    """Clean ENCODE TFBS edges (union over the five cell lines) from the profile table."""
    sel = (gmt["genome"] == "hg19") & gmt["cell_line"].isin(FIVE_CELL_LINES)
    prof = gmt[sel].reset_index(drop=True)
    steps = {"profiles_total": int(len(gmt)),
             "profiles_hg19": int((gmt["genome"] == "hg19").sum()),
             "profiles_mm9": int((gmt["genome"] == "mm9").sum()),
             "profiles_5celllines": int(len(prof)),
             "profiles_by_cell_line": {k: int(v) for k, v in prof["cell_line"].value_counts().sort_index().items()},
             "factors_5celllines_raw": int(prof["factor_raw"].nunique()),
             "associations_5celllines": int(prof["genes"].map(len).sum())}
    five = explode_profiles(prof)
    # legacy reconstruction: raw factor names, union over the 5 cell lines
    legacy = five[["factor_raw", "target_raw"]].astype(str).drop_duplicates()
    steps["edges_raw_union"] = int(len(legacy))

    fmap = {f: clean_factor(f) for f in prof["factor_raw"].unique()}
    hist = {f: is_histone(f) for f in prof["factor_raw"].unique()}
    five["tf"] = five["factor_raw"].astype(str).map(fmap)
    five["histone"] = five["factor_raw"].astype(str).map(hist)
    five["target"] = normalize_categorical(five["target_raw"])

    universe = sorted(set(five.loc[~five["histone"], "target"]) - {""})
    dropped = []
    for raw, norm in sorted(fmap.items()):
        if hist[raw]:
            reason = "histone mark"
        elif norm not in tfs:
            reason = "not a Lambert-2018 TF"
        else:
            continue
        dropped.append({"reference": "encode_tfbs", "factor_raw": raw, "factor_normalised": norm,
                        "action": "dropped", "factor_used": "", "reason": reason})
    keep = five[five["tf"].isin(tfs) & ~five["histone"]]
    selfl = keep["tf"] == keep["target"]
    steps["self_loop_pairs_dropped"] = int(keep.loc[selfl, "tf"].nunique())
    keep = keep[~selfl]
    cell_bit = {c: 1 << i for i, c in enumerate(FIVE_CELL_LINES)}
    kc = keep[["tf", "target", "cell_line"]].astype(str).drop_duplicates()
    kc["bit"] = kc["cell_line"].map(cell_bit)
    agg = kc.groupby(["tf", "target"], sort=True).agg(n_cell_lines=("bit", "size"),
                                                      mask=("bit", "sum")).reset_index()
    npf = (keep[["tf", "target", "profile"]].astype(str).drop_duplicates()
               .groupby(["tf", "target"], sort=True).size())
    agg["n_profiles"] = npf.reindex(pd.MultiIndex.from_frame(agg[["tf", "target"]])).to_numpy()
    mask_str = {m: ";".join(c for c in FIVE_CELL_LINES if m & cell_bit[c]) for m in range(1, 32)}
    agg["cell_lines"] = agg["mask"].map(mask_str)
    agg = agg[["tf", "target", "n_cell_lines", "cell_lines", "n_profiles"]]
    agg["n_cell_lines"] = agg["n_cell_lines"].astype("int16")
    agg["n_profiles"] = agg["n_profiles"].astype("int16")
    agg["cell_lines"] = agg["cell_lines"].astype("category")
    kp = keep[["tf", "factor_raw", "profile", "cell_line"]].astype(str).drop_duplicates()
    tf_info = (kp.groupby("tf")
                 .agg(raw_names=("factor_raw", lambda s: ";".join(sorted(set(s)))),
                      n_profiles=("profile", "nunique"),
                      n_cell_lines=("cell_line", "nunique"),
                      cell_lines=("cell_line", lambda s: ";".join(sorted(set(s)))))
                 .reset_index())
    return {"edges": agg.reset_index(drop=True), "legacy": legacy, "universe": universe,
            "dropped": dropped, "tf_info": tf_info, "steps": steps}


def build_encode_strict(enc: dict, k: int) -> dict:
    """ENCODE edges bound in >= k of the five cell lines, for TFs assayed in >= k lines."""
    ti = enc["tf_info"]
    keep_tfs = set(ti.loc[ti["n_cell_lines"] >= k, "tf"])
    e = enc["edges"]
    edges = e[e["tf"].isin(keep_tfs) & (e["n_cell_lines"] >= k)].reset_index(drop=True)
    excluded = [{"reference": "encode_strict", "factor_raw": r.raw_names, "factor_normalised": r.tf,
                 "action": "excluded", "factor_used": "",
                 "reason": f"assayed in {int(r.n_cell_lines)} of the five cell lines (< {k})"}
                for r in ti[~ti["tf"].isin(keep_tfs)].sort_values("tf").itertuples()]
    tf_info = ti[ti["tf"].isin(keep_tfs)].reset_index(drop=True)
    steps = {"min_cell_lines": k, "tfs_encode": int(len(ti)), "tfs_kept": len(keep_tfs),
             "edges_encode_kept_tfs": int(e["tf"].isin(keep_tfs).sum()), "edges_strict": int(len(edges)),
             "tfs_with_strict_edges": int(edges["tf"].nunique())}
    return {"edges": edges, "universe": enc["universe"], "dropped": excluded, "tf_info": tf_info,
            "steps": steps}


def harmonizome_profile_index(raw2: pd.DataFrame, raw1: pd.DataFrame, gmt: pd.DataFrame) -> dict:
    """Harmonizome hg19 non-histone profiles restricted to the genes shared with ENETS
    (raw + filtered proximal targets): profile table, cleaned factor per profile, gene
    universe U, gene index and per-profile gene-index arrays."""
    hg = gmt[(gmt["genome"] == "hg19") & ~gmt["factor_raw"].map(is_histone)].reset_index(drop=True)
    hg_factor = hg["factor_raw"].map(clean_factor)
    raw_genes = pd.unique(np.concatenate([np.asarray(g, dtype=object) for g in hg["genes"]]))
    gmap = dict(zip(raw_genes, C.normalize_series(pd.Series(raw_genes))))
    u_en = (set(C.normalize_series(raw1["target_raw"])) | set(C.normalize_series(raw2["target_raw"]))) - {""}
    u_hz = {gmap[g] for g in raw_genes} - {""}
    U = sorted(u_en & u_hz)
    gi = {g: i for i, g in enumerate(U)}
    prof_idx = [np.unique(np.fromiter((gi[gmap[g]] for g in gs if gmap[g] in gi), dtype=np.int64))
                for gs in hg["genes"]]
    return {"hg": hg, "hg_factor": hg_factor, "U": U, "gi": gi, "prof_idx": prof_idx,
            "sizes": np.array([len(x) for x in prof_idx])}


def confirm_enets2_relabels(raw2: pd.DataFrame, raw1: pd.DataFrame, gmt: pd.DataFrame,
                            awg: pd.DataFrame, cv: dict, s4a: pd.DataFrame, hidx: dict | None = None) -> dict:
    """Evidence for each ENETS2_RELABEL entry; ``confirmed`` = target overlap AND antibody metadata."""
    hidx = hidx or harmonizome_profile_index(raw2, raw1, gmt)
    hg, hg_factor, U, gi = hidx["hg"], hidx["hg_factor"], hidx["U"], hidx["gi"]
    prof_idx, sizes = hidx["prof_idx"], hidx["sizes"]
    s4 = dict(zip(s4a["factor"], s4a["n_targets"]))
    raw2_counts = raw2.groupby("factor_raw").size()
    out = {}
    for src, spec in ENETS2_RELABEL.items():
        new, ab = spec["to"], spec["encode_antibody"]
        T = np.zeros(len(U), dtype=bool)
        tg = [gi[g] for g in set(C.normalize_series(raw2.loc[raw2["factor_raw"] == src, "target_raw"])) if g in gi]
        T[tg] = True
        nT = int(T.sum())
        k = np.array([int(T[x].sum()) for x in prof_idx])
        logp = stats.hypergeom.logsf(k - 1, len(U), sizes, nT)
        lift = np.where(sizes > 0, (k / max(nT, 1)) / np.maximum(sizes / len(U), 1e-300), np.nan)
        res = pd.DataFrame({"profile": hg["profile"], "factor": hg_factor, "shared_targets": k,
                            "profile_size": sizes, "lift": lift, "log10_p": logp / np.log(10)})
        res = res[res["profile_size"] > 0].sort_values(["log10_p", "lift", "profile"],
                                                       ascending=[True, False, True])
        top = res.head(RELABEL_TOP_PROFILES)
        new_rows = res[res["factor"] == new]
        overlap_ok = bool(len(res) and res.iloc[0]["factor"] == new and not (res["factor"] == src).any())
        awg_ab = awg[awg["antibody"].str.upper() == ab.upper()]
        cvx = cv.get(ab, {})
        antibody_ok = bool(cvx.get("genecard_target") == new
                           and not (awg["target_norm"] == src).any()
                           and (awg_ab.empty or set(awg_ab["target_norm"]) == {new}))
        out[src] = {
            "relabel_to": new, "encode_antibody_term": ab,
            "n_enets2_edges_raw": int(raw2_counts.get(src, 0)),
            "n_targets_in_shared_universe": nT, "shared_universe_genes": len(U),
            "top_profiles_by_hypergeometric_p": top.round(4).to_dict("records"),
            "best_profile_of_new_factor": (new_rows.head(1).round(4).to_dict("records")[0]
                                           if len(new_rows) else None),
            "rank_of_best_new_factor_profile": (int(np.flatnonzero(res["factor"].to_numpy() == new)[0]) + 1
                                                if len(new_rows) else None),
            "harmonizome_profiles_of_source_label": int((res["factor"] == src).sum()),
            "ucsc_cv_antibody": cvx,
            "awg_uniform_experiments_with_antibody": awg_ab[["table", "target", "cell_line", "lab"]]
                                                       .to_dict("records"),
            "awg_uniform_experiments_targeting_source_label": int((awg["target_norm"] == src).sum()),
            "gerstein_S4A": {"source_label_listed": src in s4, "new_label_listed": new in s4,
                             "new_label_n_targets": s4.get(new), "source_label_n_targets": s4.get(src)},
            "check_target_overlap": overlap_ok, "check_antibody_metadata": antibody_ok,
            "confirmed": overlap_ok and antibody_ok,
        }
    return out


def build_enets2(raw2: pd.DataFrame, raw1: pd.DataFrame, tfs: frozenset, relabel: dict) -> dict:
    """Clean ENETS2 edges; ``relabel`` maps distributed label -> evidence dict (with 'confirmed')."""
    steps = {"edges_raw": int(len(raw2)), "factors_raw": int(raw2["factor_raw"].nunique()),
             "targets_raw": int(raw2["target_raw"].nunique()),
             "edge_types": sorted(raw2["edge_type"].unique().tolist()),
             "enets1_edges_raw": int(len(raw1)), "enets1_factors_raw": int(raw1["factor_raw"].nunique()),
             "enets1_targets_raw": int(raw1["target_raw"].nunique())}
    e2 = set(zip(raw2.factor_raw, raw2.target_raw))
    e1 = set(zip(raw1.factor_raw, raw1.target_raw))
    steps["enets2_subset_of_enets1_frac"] = float(len(e2 & e1) / len(e2))
    df = raw2.copy()
    fmap = {f: clean_factor(f) for f in df["factor_raw"].unique()}
    df["tf_source_label"] = df["factor_raw"].map(fmap)
    df["target"] = C.normalize_series(df["target_raw"])
    # edges under the distributed labels (for the OmniPath identity checks)
    src = df[df["tf_source_label"].isin(tfs) & (df["tf_source_label"] != df["target"])]
    edges_source_labels = (src.rename(columns={"tf_source_label": "tf"})[["tf", "target"]]
                              .drop_duplicates().reset_index(drop=True))
    dropped, fused = [], dict(fmap)
    for r, ev in sorted(relabel.items()):
        if ev["confirmed"]:
            fused[r] = ev["relabel_to"]
            dropped.append({"reference": "enets2", "factor_raw": r, "factor_normalised": fmap[r],
                            "action": "relabelled", "factor_used": ev["relabel_to"],
                            "reason": f"label error in the distributed file: ENCODE antibody "
                                      f"'{ev['encode_antibody_term']}' targets {ev['relabel_to']} "
                                      f"(UCSC ENCODE vocabulary; target overlap with Harmonizome "
                                      f"{ev['relabel_to']} profiles)"})
        else:
            fused[r] = ""
            dropped.append({"reference": "enets2", "factor_raw": r, "factor_normalised": fmap[r],
                            "action": "dropped", "factor_used": "",
                            "reason": f"suspected label error (proposed {ev['relabel_to']}) not confirmed"})
    df["tf"] = df["factor_raw"].map(fused)
    dropped += [{"reference": "enets2", "factor_raw": r, "factor_normalised": n, "action": "dropped",
                 "factor_used": "", "reason": "not a Lambert-2018 TF" if n else "missing/invalid symbol"}
                for r, n in sorted(fmap.items()) if r not in relabel and n not in tfs]
    keep = df[df["tf"].isin(tfs)]
    steps["self_loops_dropped"] = int((keep["tf"] == keep["target"]).sum())
    keep = keep[keep["tf"] != keep["target"]]
    edges = keep[["tf", "target"]].drop_duplicates().sort_values(["tf", "target"]).reset_index(drop=True)
    steps["duplicates_after_normalisation"] = int(len(keep) - len(edges))
    steps["relabelled"] = {r: fused[r] for r in relabel}
    # factors of the raw proximal network without any filtered edge: assayed, not evaluable
    raw_only = sorted(set(raw1["factor_raw"]) - set(raw2["factor_raw"]))
    for r in raw_only:
        n = clean_factor(r)
        dropped.append({"reference": "enets2", "factor_raw": r, "factor_normalised": n,
                        "action": "not_evaluable" if n in tfs else "dropped", "factor_used": "",
                        "reason": ("in the raw proximal network (enets1) but absent from the filtered "
                                   "network: no filtered release, not evaluable") if n in tfs else
                                  ("raw-network-only factor; " + ("not a Lambert-2018 TF" if n else
                                                                  "missing/invalid symbol"))})
    steps["raw_only_factors"] = raw_only
    # genes that could be called a target: any proximal target in the raw network
    # (enets1) or in the filtered network (16% of enets2 edges are not in enets1)
    universe = sorted((set(C.normalize_series(raw1["target_raw"])) | set(df["target"])) - {""})
    tf_info = (keep.groupby("tf").agg(raw_names=("factor_raw", lambda s: ";".join(sorted(set(s)))))
                   .reset_index())
    tf_info["n_profiles"] = np.nan
    tf_info["n_cell_lines"] = np.nan
    tf_info["cell_lines"] = ""
    return {"edges": edges, "edges_source_labels": edges_source_labels, "universe": universe,
            "dropped": dropped, "tf_info": tf_info, "steps": steps}


def _band_region(loc: str) -> str:
    """'22q11.21' -> '22q11' (chromosome, arm, region and band); '' when unknown."""
    m = re.match(r"^(\d+|X|Y)([pq])(\d{1,2})", str(loc))
    return "".join(m.groups()) if m else ""


def enets2_label_concordance(edges: pd.DataFrame, hidx: dict, universes: dict) -> pd.DataFrame:
    """Per ENETS2 regulon: overlap with the Harmonizome hg19 profiles of the same factor.

    For each TF, the hypergeometric p of its targets against every profile (shared ENETS /
    Harmonizome gene universe) is computed; the regulon is ``concordant`` when its best
    same-factor profile reaches Bonferroni-adjusted p < ``LABEL_CHECK_ALPHA`` (over all
    profiles).  The most frequent cytogenetic region of the targets (e.g. '22q11') and its
    share flag positional artefacts such as amplicon-driven regulons.  Diagnostic only:
    no regulon is dropped (no prespecified rule)."""
    hg_factor = hidx["hg_factor"].to_numpy()
    gi, prof_idx, sizes, nU = hidx["gi"], hidx["prof_idx"], hidx["sizes"], len(hidx["U"])
    n_prof = int((sizes > 0).sum())
    thr = np.log10(LABEL_CHECK_ALPHA / max(n_prof, 1))
    band = C.symbol_cytoband()
    rows = []
    for tf, g in edges.groupby("tf", sort=True):
        T = np.zeros(nU, dtype=bool)
        T[[gi[x] for x in set(g["target"]) if x in gi]] = True
        nT = int(T.sum())
        k = np.array([int(T[x].sum()) for x in prof_idx])
        lp = np.where(sizes > 0, stats.hypergeom.logsf(k - 1, nU, sizes, nT) / np.log(10), np.nan)
        order = np.lexsort((-k, np.nan_to_num(lp, nan=np.inf)))
        same = np.flatnonzero(hg_factor == tf)
        same = same[sizes[same] > 0]
        best_same = same[np.argmin(lp[same])] if len(same) else None
        reg = pd.Series([_band_region(band.get(x, "")) for x in g["target"]])
        reg = reg[reg != ""]
        top_reg = reg.value_counts().sort_index().sort_values(ascending=False, kind="stable")
        row = {"tf": tf, "n_targets": int(len(g)), "n_targets_in_shared_universe": nT,
               "n_same_factor_profiles": int(len(same)),
               "best_same_factor_profile": hidx["hg"]["profile"].iat[best_same] if best_same is not None else "",
               "best_same_factor_shared_targets": int(k[best_same]) if best_same is not None else 0,
               "best_same_factor_log10_p": float(f"{lp[best_same]:.4g}") if best_same is not None else np.nan,
               "best_same_factor_rank": (int(np.flatnonzero(order == best_same)[0]) + 1
                                         if best_same is not None else np.nan),
               "top_profile": hidx["hg"]["profile"].iat[order[0]], "top_profile_factor": hg_factor[order[0]],
               "top_profile_log10_p": float(f"{lp[order[0]]:.4g}"),
               "bonferroni_log10_threshold": float(f"{thr:.4g}"),
               "top_cytoband_region": top_reg.index[0] if len(top_reg) else "",
               "top_cytoband_region_frac": float(f"{top_reg.iloc[0] / len(reg):.4g}") if len(top_reg) else np.nan}
        row["concordant"] = bool(best_same is not None and lp[best_same] <= thr)
        row["label_check"] = ("concordant" if row["concordant"] else
                              "no Harmonizome profile of the same factor" if best_same is None else
                              "same-factor overlap not significant")
        for u, (utfs, _) in universes.items():
            row[f"in_universe_{u}"] = tf in utfs
        rows.append(row)
    return pd.DataFrame(rows)


def enets2_cell_lines(edges: pd.DataFrame, awg: pd.DataFrame) -> dict:
    """ENCODE uniform-TFBS (AWG, same Jan-2011 freeze) cell lines of each ENETS2 TF."""
    per = {t: sorted(set(awg.loc[awg["target_norm"] == t, "cell_line"])) for t in sorted(edges["tf"].unique())}
    five = set(FIVE_CELL_LINES)
    other_only = {t: c for t, c in per.items() if c and not (set(c) & five)}
    not_in_awg = sorted(t for t, c in per.items() if not c)
    others = sorted({c for t in other_only for c in other_only[t]})
    desc = ("mainly " + ";".join(FIVE_CELL_LINES) + "; " + ", ".join(sorted(other_only)) +
            " assayed only in other ENCODE lines (" + ";".join(others) + ")")
    return {"description": desc, "other_lines_only": other_only, "not_in_awg_uniform_set": not_in_awg}


def _literal_set(script: Path, var: str):
    tree = ast.parse(script.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == var for t in node.targets):
            return set(ast.literal_eval(node.value))
    return None


def legacy_checks(enc_legacy: pd.DataFrame, raw2: pd.DataFrame) -> dict:
    """Compare the legacy R1 files with what the official downloads give."""
    out = {}
    leg = pd.read_csv(C.ENCODE_TFBS, sep="\t", dtype=str, keep_default_na=False)
    a = set(zip(leg.iloc[:, 0], leg.iloc[:, 1]))
    b = set(zip(enc_legacy.factor_raw, enc_legacy.target_raw))
    out["encode_tfbs"] = {
        "legacy_file": str(C.ENCODE_TFBS), "legacy_md5": md5sum(C.ENCODE_TFBS),
        "legacy_edges": len(a), "legacy_factors": int(leg.iloc[:, 0].nunique()),
        "reconstructed_edges": len(b), "identical_edge_set": a == b,
        "recipe": "Harmonizome encodetf crisp library; hg19 profiles of the 5 cell lines; union per "
                  "raw factor name over cell lines and replicates; no symbol normalisation; self-loops kept",
        "legacy_self_loops": int(sum(1 for t, g in a if t == g)),
        "legacy_nonTF_factors_examples": sorted({t for t, _ in a if not (clean_factor(t) in C.lambert_tfs())})[:12],
    }
    sp53_lib = LEGACY_SP53_DIR / "tfbs_gene_set_library_crisp.gmt"
    if sp53_lib.exists():
        with gzip.open(RAW_FILES["harmonizome_encode_tfbs_gmt"]["path"], "rb") as fh:
            h = hashlib.md5(fh.read()).hexdigest()
        out["encode_tfbs"]["subproject53_library_md5"] = md5sum(sp53_lib)
        out["encode_tfbs"]["official_library_md5_uncompressed"] = h
        out["encode_tfbs"]["subproject53_library_identical_to_official"] = (md5sum(sp53_lib) == h)

    leg2 = pd.read_csv(C.ENETS2, sep="\t", dtype=str, keep_default_na=False)
    a2 = set(zip(leg2.tf, leg2.target))
    raw_pairs = set(zip(raw2.factor_raw, raw2.target_raw))
    kept = set(leg2.tf)
    raw_kept = {(t, g) for t, g in raw_pairs if t in kept}
    removed = sorted(set(raw2.factor_raw) - kept)
    out["enets2"] = {
        "legacy_file": str(C.ENETS2), "legacy_md5": md5sum(C.ENETS2),
        "legacy_edges": len(a2), "legacy_factors": len(kept),
        "legacy_subset_of_official": a2 <= raw_pairs,
        "legacy_equals_official_restricted_to_its_factors": a2 == raw_kept,
        "factors_removed_in_legacy": removed,
        "nonLambert_factors_kept_in_legacy": sorted(t for t in kept if clean_factor(t) not in C.lambert_tfs()),
        "recipe": (f"official enets2.Proximal_filtered.txt with a hand-made list of {len(removed)} factors "
                   "removed ('sequence-specific' filter); no script reproduces the list (related "
                   "exclusion lists of the subproject_53 GPL-paper replication are cited below)"),
    }
    related = {}
    for name, var in LEGACY_SP53_SCRIPTS.items():
        p = LEGACY_SP53 / "scripts" / name
        if not p.exists():
            related[name] = {"path": str(p), "found": False}
            continue
        excl = _literal_set(p, var)
        related[name] = {"path": str(p), "found": True, "md5": md5sum(p), "variable": var,
                         "n_excluded": len(excl) if excl is not None else None,
                         "reproduces_legacy_removed_list": (excl is not None
                                                            and (excl & set(raw2.factor_raw)) == set(removed)),
                         "n_shared_with_legacy_removed_list": (len(excl & set(removed))
                                                               if excl is not None else None)}
    out["enets2"]["subproject53_related_scripts"] = related
    sp53_dl = LEGACY_SP53_DIR / "enets2_proximal_filtered_download.txt"
    if sp53_dl.exists():
        out["enets2"]["subproject53_download_identical_to_official"] = (
            md5sum(sp53_dl) == md5sum(RAW_FILES["enets2_proximal_filtered"]["path"]))
    return out


# ----------------------------------------------------------------------------
# Base rates, target-degree model and overlap with annotators
# ----------------------------------------------------------------------------


def _check_null_settings(bin_rule: str, degree_definition: str) -> None:
    if bin_rule not in SUPPORTED_BIN_RULES:
        raise ValueError(f"target_degree_bin_rule {bin_rule!r} not implemented ({SUPPORTED_BIN_RULES})")
    if degree_definition not in SUPPORTED_DEGREE_DEFINITIONS:
        raise ValueError(f"target_degree_definition {degree_definition!r} not implemented "
                         f"({SUPPORTED_DEGREE_DEFINITIONS})")


_check_null_settings(TARGET_DEGREE_BIN_RULE, TARGET_DEGREE_DEFINITION)


def single_thread_blas():
    """Context manager limiting BLAS/OpenMP to one thread (bit-stable dense algebra)."""
    if _threadpool_limits is None:  # pragma: no cover
        return contextlib.nullcontext()
    return _threadpool_limits(limits=1)


def reference_index(edges: pd.DataFrame, universe, assayed=None, *, n_bins: int | None = None,
                    bin_rule: str | None = None, degree_definition: str | None = None) -> dict:
    """Index a held-out reference and fit its target-degree null.

    Parameters
    ----------
    edges : DataFrame with columns ``tf``, ``target`` (HGNC symbols; one row per edge).
    universe : iterable of genes that could have been called a target (the reference's
        gene universe, optionally intersected with an evaluation gene set).  Edges whose
        target is outside it are ignored.
    assayed : iterable of evaluable (assayed) TFs.  ``None`` = TFs with >= 1 edge in the
        universe.  Assayed TFs without edges keep an all-zero row (they are evaluable and
        every pair is a negative).
    n_bins, bin_rule, degree_definition : target-degree null settings; default to
        ``C.PRESPECIFIED['target_degree_bins' / 'target_degree_bin_rule' /
        'target_degree_definition']``.  Only the prespecified rule and definition are
        implemented; anything else raises ``ValueError``.

    Returns
    -------
    dict with
        ``tfs`` (sorted assayed TFs), ``ti`` / ``gi`` (TF / gene -> row / column),
        ``M`` (bool TF x gene incidence), ``targets`` (TF -> set of targets in the
        universe), ``base`` (TF -> TF-matched base rate = targets / (|universe| - self)),
        ``universe``, ``n_universe``, ``keys`` (``tf\\ttarget`` edge keys), ``self_col``
        (column of each TF's own gene or -1), and the fitted null of
        :func:`target_degree_model` (``deg``, ``bin_edges``, ``to_bin``, ``P``, ``model``).
        Use :func:`degree_bin` and :func:`null_probability` to query it for arbitrary pairs.
    """
    U = sorted(set(universe))
    gi = {g: i for i, g in enumerate(U)}
    e = edges[edges["target"].isin(gi)]
    tfs = sorted(set(e["tf"]) if assayed is None else set(assayed))
    ti = {t: i for i, t in enumerate(tfs)}
    e = e[e["tf"].isin(ti)]
    M = np.zeros((len(tfs), len(U)), dtype=bool)
    M[e["tf"].map(ti).to_numpy(), e["target"].map(gi).to_numpy()] = True
    self_col = np.array([gi.get(t, -1) for t in tfs], dtype=np.int64)
    M[np.flatnonzero(self_col >= 0), self_col[self_col >= 0]] = False   # self-pairs are never edges
    n_u = len(U)
    n_t = n_u - (self_col >= 0)
    tsets = {t: set(np.asarray(U)[M[i]]) for i, t in enumerate(tfs)}
    base = {t: M[i].sum() / n_t[i] for i, t in enumerate(tfs)}
    keep = e[e["tf"] != e["target"]]
    R = {"targets": tsets, "base": base, "universe": set(U), "n_universe": n_u, "tfs": tfs,
         "keys": set(C.edge_key(keep["tf"], keep["target"])), "gi": gi, "ti": ti, "M": M,
         "self_col": self_col}
    R.update(target_degree_model(M, self_col, n_bins=n_bins, bin_rule=bin_rule,
                                 degree_definition=degree_definition))
    return R


def degree_bin_edges(deg: np.ndarray, n_bins: int, bin_rule: str = TARGET_DEGREE_BIN_RULE) -> np.ndarray:
    """Bin boundaries of the target-degree null.

    ``union_gene_edge_deciles``: the ``n_bins``-quantiles (``method='lower'``) of the
    reference in-degree over universe genes (resolves the sparse bulk) united with the
    ``n_bins``-quantiles over reference edges, i.e. each gene weighted by its in-degree
    (resolves the hub tail where positives concentrate).  Returns the sorted unique
    boundaries; bin b holds degrees in [inner[b-1], inner[b]) with inner = edges[1:-1].
    """
    _check_null_settings(bin_rule, TARGET_DEGREE_DEFINITION)
    deg = np.asarray(deg, dtype=np.int64)
    q = np.linspace(0, 1, int(n_bins) + 1)
    by_edge = np.repeat(deg, deg)
    parts = [np.quantile(deg, q, method="lower")]
    if len(by_edge):
        parts.append(np.quantile(by_edge, q, method="lower"))
    return np.unique(np.concatenate(parts))


def target_degree_model(M: np.ndarray, self_col: np.ndarray, n_bins: int | None = None, *,
                        bin_rule: str | None = None, degree_definition: str | None = None) -> dict:
    """Fit the prespecified target-degree null on a TF x gene incidence matrix.

    Model: logit P(edge TF->g) = a_TF + b_bin(d_-TF(g)) over all assayed-TF x universe-gene
    pairs except self-pairs, where (``leave_pair_out_ref_indegree``) d_-TF(g) = number of
    *other* assayed TFs binding g = ``M[:, g].sum() - M[TF, g]``; bins follow
    :func:`degree_bin_edges` (``union_gene_edge_deciles``).  Leaving the pair out keeps a
    positive from raising its own expected probability.  Fitted by maximum likelihood on
    the (TF, bin) aggregated counts with damped Newton-Raphson; with the canonical link the
    fitted counts reproduce the observed per-TF and per-bin margins (``max_margin_error_*``,
    convergence tolerance ``MODEL_GRAD_TOL`` edges).  BLAS is pinned to one thread so the
    fit is bit-stable.

    Parameters
    ----------
    M : bool array (n_tf, n_gene); ``M[i, g]`` = TF i binds gene g in the reference.
    self_col : int array (n_tf,); column of TF i's own gene, or -1 when not in the universe.
    n_bins, bin_rule, degree_definition : default to the ``C.PRESPECIFIED`` values.

    Returns
    -------
    dict with ``deg`` (in-degree per gene), ``bin_edges``, ``to_bin`` (degree -> bin index),
    ``P`` (n_tf x n_bin fitted probabilities) and ``model`` (settings, convergence and fit
    diagnostics; JSON-serialisable).  Raises ``RuntimeError`` if the fit does not converge.
    """
    n_bins = TARGET_DEGREE_BINS if n_bins is None else int(n_bins)
    bin_rule = TARGET_DEGREE_BIN_RULE if bin_rule is None else bin_rule
    degree_definition = TARGET_DEGREE_DEFINITION if degree_definition is None else degree_definition
    _check_null_settings(bin_rule, degree_definition)
    M = np.asarray(M, dtype=bool)
    self_col = np.asarray(self_col, dtype=np.int64)
    n_tf, n_g = M.shape
    deg = M.sum(0).astype(np.int64)
    edges = degree_bin_edges(deg, n_bins, bin_rule)
    inner = edges[1:-1]

    def to_bin(d):
        return np.searchsorted(inner, d, side="right")

    nb = len(inner) + 1
    n = np.zeros((n_tf, nb))
    k = np.zeros((n_tf, nb))
    for i in range(n_tf):
        b = to_bin(deg - M[i])
        ok = np.ones(n_g, dtype=bool)
        if self_col[i] >= 0:
            ok[self_col[i]] = False
        n[i] = np.bincount(b[ok], minlength=nb)
        k[i] = np.bincount(b[ok], weights=M[i][ok], minlength=nb)
    # design over non-empty cells: TF one-hot (all) + bin one-hot (first bin = reference)
    cells = np.argwhere(n > 0)
    X = np.zeros((len(cells), n_tf + nb - 1))
    X[np.arange(len(cells)), cells[:, 0]] = 1.0
    nz = cells[:, 1] > 0
    X[np.flatnonzero(nz), n_tf + cells[nz, 1] - 1] = 1.0
    y, w = k[cells[:, 0], cells[:, 1]], n[cells[:, 0], cells[:, 1]]
    beta = np.zeros(X.shape[1])
    tf_rate = np.clip(k.sum(1) / np.maximum(n.sum(1), 1), 1e-6, 1 - 1e-6)
    beta[:n_tf] = np.log(tf_rate / (1 - tf_rate))

    def loglik(b):
        eta = X @ b
        return float(np.sum(y * eta - w * np.logaddexp(0.0, eta)))

    # damped Newton-Raphson (step halving keeps the log-likelihood monotone)
    with single_thread_blas():
        ll, converged, it = loglik(beta), False, 0
        for it in range(MODEL_MAX_ITER):
            p = 1 / (1 + np.exp(-(X @ beta)))
            g = X.T @ (y - w * p)                 # = observed - fitted margins
            if np.max(np.abs(g)) < MODEL_GRAD_TOL:
                converged = True
                break
            H = (X * (w * p * (1 - p))[:, None]).T @ X + 1e-9 * np.eye(X.shape[1])
            step, t = np.linalg.solve(H, g), 1.0
            while True:
                cand = beta + t * step
                ll_c = loglik(cand)
                if ll_c >= ll - 1e-12 * abs(ll) or t < 1e-10:
                    break
                t /= 2
            beta, ll = cand, ll_c
    if not converged:
        raise RuntimeError("target-degree model did not converge")
    eta_all = beta[:n_tf][:, None] + np.concatenate([[0.0], beta[n_tf:]])[None, :]
    P = 1 / (1 + np.exp(-eta_all))
    fitted = P * n
    return {"deg": deg, "bin_edges": edges, "to_bin": to_bin, "P": P,
            "model": {"n_bins_prespecified": n_bins, "bin_rule": bin_rule,
                      "degree_definition": degree_definition,
                      "n_bins": int(nb), "bin_edges": edges.tolist(), "iterations": it,
                      "converged": converged, "loglik": float(f"{ll:.12g}"),
                      "max_margin_error_tf": float(f"{np.max(np.abs(fitted.sum(1) - k.sum(1))):.3g}"),
                      "max_margin_error_bin": float(f"{np.max(np.abs(fitted.sum(0) - k.sum(0))):.3g}"),
                      "bin_logodds": np.concatenate([[0.0], beta[n_tf:]]).round(4).tolist(),
                      "bin_n_pairs": n.sum(0).astype(np.int64).tolist(),
                      "bin_positive_rate": (k.sum(0) / np.maximum(n.sum(0), 1)).round(5).tolist()}}


def _pair_index(R: dict, tf, target):
    tf = pd.Series(np.asarray(tf, dtype=object))
    target = pd.Series(np.asarray(target, dtype=object))
    t_idx = tf.map(R["ti"]).to_numpy(dtype=float)
    g_idx = target.map(R["gi"]).to_numpy(dtype=float)
    ok = ~np.isnan(t_idx) & ~np.isnan(g_idx) & (tf.to_numpy() != target.to_numpy())
    return t_idx, g_idx, ok


def degree_bin(R: dict, tf, target) -> np.ndarray:
    """Target-degree bin of each (tf, target) pair under :func:`reference_index` ``R``.

    The degree is leave-pair-out (the pair's own reference edge is not counted), so a
    stratified null draws, for each evaluated pair, from pairs of the same TF (or any TF)
    and the same bin.  Pairs with a non-assayed TF, a target outside the universe or
    tf == target get -1."""
    t_idx, g_idx, ok = _pair_index(R, tf, target)
    out = np.full(len(ok), -1, dtype=np.int64)
    ti, gj = t_idx[ok].astype(np.int64), g_idx[ok].astype(np.int64)
    out[ok] = R["to_bin"](R["deg"][gj] - R["M"][ti, gj])
    return out


def null_probability(R: dict, tf, target) -> np.ndarray:
    """Target-matched null probability P(edge) = expit(a_TF + b_bin) of each pair.

    NaN for pairs that are not evaluable (TF not assayed, target outside the universe,
    tf == target).  Summing it over a predicted edge set gives the expected number of
    reference positives under the target-degree null."""
    t_idx, g_idx, ok = _pair_index(R, tf, target)
    out = np.full(len(ok), np.nan)
    b = degree_bin(R, np.asarray(tf, dtype=object)[ok], np.asarray(target, dtype=object)[ok])
    out[ok] = R["P"][t_idx[ok].astype(np.int64), b]
    return out


def _per_tf(e: pd.DataFrame, R: dict, chip_keys: set | None) -> pd.DataFrame | None:
    """Evaluable annotator edges -> per-TF observed, TF-matched and target-matched expectations."""
    s = e[e["tf"].isin(R["ti"]) & e["target"].isin(R["gi"]) & (e["tf"] != e["target"])]
    if len(s) == 0:
        return None
    t_idx = s["tf"].map(R["ti"]).to_numpy()
    g_idx = s["target"].map(R["gi"]).to_numpy()
    y = R["M"][t_idx, g_idx]
    d = pd.DataFrame({"tf": s["tf"].to_numpy(), "pos": y.astype(float),
                      "exp_tf": s["tf"].map(R["base"]).to_numpy(float),
                      "exp_tgt": null_probability(R, s["tf"].to_numpy(), s["target"].to_numpy())})
    if chip_keys is not None:
        d["chip"] = C.edge_key(s["tf"], s["target"]).isin(chip_keys).to_numpy().astype(float)
    return d.groupby("tf", sort=True).agg(n=("pos", "size"), pos=("pos", "sum"), exp_tf=("exp_tf", "sum"),
                                          exp_tgt=("exp_tgt", "sum"),
                                          **({"chip": ("chip", "sum")} if chip_keys is not None else {}))


def _boot_ci(num: np.ndarray, den: np.ndarray, seed: int, n_boot: int):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(num), size=(n_boot, len(num)))
    with np.errstate(divide="ignore", invalid="ignore"):
        boot = num[idx].sum(1) / den[idx].sum(1)
    return tuple(np.nanpercentile(boot, [2.5, 97.5]))


def overlap_table(ann: pd.DataFrame, group_col: str, refs: dict, rng_seed: int, n_boot: int,
                  extra_cols: list | None = None, chip_keys: set | None = None) -> pd.DataFrame:
    """For every annotator x reference: precision of the annotator's evaluable edges against
    the TF-matched, target-matched (TF x target-degree) and pooled reference base rates, plus
    the TF-matched lift inside the gene set G (``refs[name]['in_G']``)."""
    rows = []
    extra_cols = extra_cols or []
    for ref_name, R in refs.items():
        RG = R.get("in_G")
        for name, g in ann.groupby(group_col, sort=True):
            meta = {c: g[c].iloc[0] for c in extra_cols}
            e = g.drop_duplicates(["tf", "target"])
            row = {"reference": ref_name, "annotator": name, **meta, "n_edges_total": int(len(e))}
            per = _per_tf(e, R, chip_keys)
            if per is None:
                rows.append({**row, "n_edges_evaluable": 0, "n_tfs_evaluable": 0})
                continue
            n, k = per["n"].sum(), per["pos"].sum()
            e_tf, e_tg = per["exp_tf"].sum(), per["exp_tgt"].sum()
            ref_pos = sum(len(R["targets"][t]) for t in per.index)
            pooled_base = ref_pos / sum(R["n_universe"] - (1 if t in R["universe"] else 0) for t in per.index)
            pos_a = per["pos"].to_numpy(float)
            lo, hi = _boot_ci(pos_a, per["exp_tf"].to_numpy(float), rng_seed, n_boot)
            lo2, hi2 = _boot_ci(pos_a, per["exp_tgt"].to_numpy(float), rng_seed, n_boot)
            row.update({
                "n_edges_evaluable": int(n), "n_tfs_evaluable": int(len(per)),
                "n_reference_positive": int(k), "precision": k / n,
                "tf_matched_base_rate": e_tf / n, "target_matched_base_rate": e_tg / n,
                "pooled_base_rate": pooled_base,
                "lift_tf_matched": (k / e_tf) if e_tf > 0 else np.nan,
                "lift_tf_matched_ci_lo": lo, "lift_tf_matched_ci_hi": hi,
                "lift_target_matched": (k / e_tg) if e_tg > 0 else np.nan,
                "lift_target_matched_ci_lo": lo2, "lift_target_matched_ci_hi": hi2,
                "lift_pooled": (k / n) / pooled_base if pooled_base > 0 else np.nan,
                "reference_recall": k / ref_pos,
            })
            if chip_keys is not None:
                row["frac_edges_remap_pazarht_supported"] = per["chip"].sum() / n
            if RG is not None:
                pg = _per_tf(e, RG, None)
                if pg is not None and pg["exp_tf"].sum() > 0:
                    lo3, hi3 = _boot_ci(pg["pos"].to_numpy(float), pg["exp_tf"].to_numpy(float), rng_seed, n_boot)
                    row.update({"n_edges_evaluable_in_G": int(pg["n"].sum()),
                                "lift_tf_matched_in_G": pg["pos"].sum() / pg["exp_tf"].sum(),
                                "lift_tf_matched_in_G_ci_lo": lo3, "lift_tf_matched_in_G_ci_hi": hi3})
            rows.append(row)
    return pd.DataFrame(rows)


def sensitivity_groups(ann: pd.DataFrame, sources_json: dict) -> tuple[dict, dict]:
    """Held-in records of the two literature sensitivity groups and exact definitions.

    ``all_held_in_no_ht`` = held-in records minus the LIT regulons that evidence variant
    ``lit_no_ht`` removes (``sources.json['lit_ht_like_regulons']``, (source, TF) pairs);
    ``all_held_in_no_chipderived`` = held-in records minus every record of the sources in
    ``C.PRESPECIFIED['lit_no_chipderived_drop']`` (evidence variant ``lit_no_chipderived``).
    A pooled edge stays in a group while any remaining source supports it.  Raises if a
    listed source or regulon is absent from the annotator edges (no silent no-op)."""
    heldin = ann[~ann["held_out"]]
    srcs = set(heldin["annotator"])
    drop = list(C.PRESPECIFIED["lit_no_chipderived_drop"])
    if "lit_no_chipderived_drop" in sources_json and list(sources_json["lit_no_chipderived_drop"]) != drop:
        raise ValueError("sources.json lit_no_chipderived_drop differs from common.PRESPECIFIED")
    if missing := [s for s in drop if s not in srcs]:
        raise ValueError(f"lit_no_chipderived_drop sources absent from held-in annotators: {missing}")
    ht = [tuple(x) for x in sources_json.get("lit_ht_like_regulons", [])]
    if not ht:
        raise ValueError("sources.json has no 'lit_ht_like_regulons'")
    present = set(map(tuple, heldin[["annotator", "tf"]].drop_duplicates().to_numpy()))
    if missing := [r for r in ht if r not in present]:
        raise ValueError(f"lit_ht_like_regulons absent from held-in annotators: {missing}")
    is_ht = pd.MultiIndex.from_frame(heldin[["annotator", "tf"]]).isin(ht)
    groups = {"all_held_in_no_ht": heldin[~is_ht],
              "all_held_in_no_chipderived": heldin[~heldin["annotator"].isin(drop)]}
    all_keys = set(C.edge_key(heldin["tf"], heldin["target"]))
    n_src = heldin.drop_duplicates(["annotator", "tf", "target"]).groupby("annotator").size()
    watch = sorted(set(drop) | {s for s, _ in ht})
    defs = {}
    for name, kept in groups.items():
        kept_src = kept.drop_duplicates(["annotator", "tf", "target"]).groupby("annotator").size()
        removed = {s: int(n_src[s] - kept_src.get(s, 0)) for s in sorted(n_src.index)
                   if n_src[s] != kept_src.get(s, 0)}
        defs[name] = {
            "evidence_variant": "lit_no_ht" if name.endswith("no_ht") else "lit_no_chipderived",
            "removes": ({"regulons": [list(r) for r in sorted(ht)]} if name.endswith("no_ht")
                        else {"sources": drop,
                              "lit_other_components": sources_json.get("lit_other_components")}),
            "records_removed_by_source": removed,
            "records_kept_of_listed_sources": {s: f"{int(kept_src.get(s, 0))} of {int(n_src[s])}" for s in watch},
            "pooled_held_in_edges": len(all_keys),
            "pooled_edges_removed": len(all_keys - set(C.edge_key(kept["tf"], kept["target"]))),
        }
    return groups, defs


def overlap_flags(ov: pd.DataFrame, chip_sources: list) -> pd.DataFrame:
    """Add consistent dependence/support flags to an overlap table.

    Columns added: ``low_support`` (< ``MIN_SUPPORT_TFS`` evaluable TFs or <
    ``MIN_SUPPORT_POSITIVES`` reference positives), ``shares_encode_experiments`` (the
    row's edges come from, or were selected by co-support from, a held-out ChIP source that
    contains ENCODE experiments: ReMap and OmniPath ENCODE-proximal/-distal against every
    reference, since ENETS2 is built from the ENCODE Jan-2011 experiments),
    ``conservative_headline`` (``CONSERVATIVE_HEADLINE`` rows) and ``flag`` ('; '-joined
    texts).  ``chip_sources`` = every held-out CHIP source id (used for 'family:CHIP' and
    the ``chip_any`` split)."""
    ov = ov.copy()
    n_tf = ov["n_tfs_evaluable"].fillna(0)
    n_pos = ov.get("n_reference_positive", pd.Series(0, index=ov.index)).fillna(0)
    ov["low_support"] = (n_tf < MIN_SUPPORT_TFS) | (n_pos < MIN_SUPPORT_POSITIVES)
    flags, shares = [], []
    for ref, ann, low in zip(ov["reference"], ov["annotator"], ov["low_support"]):
        base, _, split = str(ann).partition("|")
        f = []
        if ref == "enets2" and split.endswith("_chip_any"):
            f.append(FLAG_UNINFORMATIVE)
        members = list(chip_sources) if base == "family:CHIP" else [base]
        sh = False
        if ref == "enets2" and set(members) & set(GERSTEIN_SOURCES):
            f.append(FLAG_SAME_NETWORK)
            sh = True
        if set(members) & (set(ENCODE_SHARING_SOURCES) - (set(GERSTEIN_SOURCES) if ref == "enets2" else set())):
            f.append(FLAG_SHARES)
            sh = True
        if base in AGGREGATES_WITH_CHIP:
            f.append(FLAG_AGGREGATE)
            sh = True
        if split.startswith("with_"):
            tag = split[len("with_"):]
            sel = chip_sources if CHIP_SPLITS[tag] is None else CHIP_SPLITS[tag]
            enc = [s for s in ENCODE_SHARING_SOURCES if s in sel]
            if enc:
                f.append(FLAG_SELECTED.format(", ".join(enc)))
                sh = True
        if low:
            f.append(FLAG_LOW)
        flags.append("; ".join(f))
        shares.append(sh)
    ov["shares_encode_experiments"] = shares
    ov["conservative_headline"] = ov["annotator"].eq(CONSERVATIVE_HEADLINE)
    ov["flag"] = flags
    return ov


def _read_annotator_file(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    cols = {c.lower(): c for c in df.columns}
    tf_c = next(cols[c] for c in ("tf", "regulator", "source") if c in cols)
    tg_c = next(cols[c] for c in ("target", "gene") if c in cols)
    an_c = next(cols[c] for c in ("annotator", "source_id", "resource", "source_name") if c in cols)
    fam_c = next((cols[c] for c in ("family", "modality", "modality_family") if c in cols), None)
    out = pd.DataFrame({"tf": C.normalize_series(df[tf_c]), "target": C.normalize_series(df[tg_c]),
                        "annotator": df[an_c].astype(str)})
    out["family"] = df[fam_c].astype(str).values if fam_c else ""
    meta = source_meta()
    out["held_out"] = out["annotator"].map(
        lambda a: bool(meta[a]["held_out"]) if a in meta else None)
    miss = out["held_out"].isna()
    out.loc[miss, "held_out"] = out.loc[miss, "family"].str.upper().eq("CHIP")
    out["held_out"] = out["held_out"].astype(bool)
    if not fam_c:
        out["family"] = out["annotator"].map(lambda a: meta.get(a, {}).get("family", ""))
    return out


def require_input(ok: bool, what: str, allow_partial: bool) -> bool:
    """True if an upstream input is present; otherwise raise, or warn with ``allow_partial``."""
    if ok:
        return True
    if allow_partial:
        warnings.warn(f"{what} missing; continuing without it (--allow-partial)")
        return False
    raise FileNotFoundError(f"{what} missing: run the upstream stage first (or pass --allow-partial)")


def sources_json() -> dict:
    """The annotator stage's sources.json ({} when absent)."""
    return json.loads(ANNOTATOR_SOURCES.read_text()) if ANNOTATOR_SOURCES.exists() else {}


def source_meta() -> dict:
    return sources_json().get("sources", {})


def load_annotators(wait_min: float, allow_partial: bool = False) -> pd.DataFrame | None:
    """Annotator edges of the annotator stage; ``held_out`` from its sources.json (CHIP family).

    Polls up to ``wait_min`` minutes for the edge file; a missing edge file or sources.json
    raises unless ``allow_partial``."""
    t0 = time.time()
    while not ANNOTATOR_EDGES.exists() and (time.time() - t0) < wait_min * 60:
        print(f"[wait] {ANNOTATOR_EDGES} not found; sleeping 60 s")
        time.sleep(60)
    ok = require_input(ANNOTATOR_EDGES.exists(), str(ANNOTATOR_EDGES), allow_partial)
    ok = require_input(ANNOTATOR_SOURCES.exists(), str(ANNOTATOR_SOURCES), allow_partial) and ok
    return _read_annotator_file(ANNOTATOR_EDGES) if ok else None


def omnipath_pickle() -> tuple[Path, dict]:
    """The OmniPath DoRothEA pickle used by the annotator stage (its provenance.json), else the
    newest cached pickle with per-record 'sources'; returns the path and its checksums."""
    p, expected_sha = None, None
    if ANNOTATOR_PROVENANCE.exists():
        dc = json.loads(ANNOTATOR_PROVENANCE.read_text()).get("dorothea_cache", {})
        if dc.get("path") and Path(dc["path"]).exists():
            p, expected_sha = Path(dc["path"]), dc.get("sha256")
    if p is None:
        cands = []
        for q in sorted(C.OMNIPATH_CACHE.glob("*.pickle")):
            try:
                df = pd.read_pickle(q)
            except Exception:  # pragma: no cover - unreadable cache entries are ignored
                continue
            if isinstance(df, pd.DataFrame) and {"source_genesymbol", "target_genesymbol", "sources"} <= set(df.columns):
                cands.append((len(df.columns), len(df), q.name, q))
        if not cands:
            raise FileNotFoundError("no OmniPath TF-target pickle in " + str(C.OMNIPATH_CACHE))
        p = sorted(cands)[-1][3]
    rec = {"path": str(p), "md5": md5sum(p), "sha256": sha256sum(p), "bytes": p.stat().st_size,
           "mtime": datetime.fromtimestamp(p.stat().st_mtime, timezone.utc).isoformat(timespec="seconds")}
    if expected_sha:
        rec["annotator_stage_sha256"] = expected_sha
        if rec["sha256"] != expected_sha:
            raise RuntimeError(f"{p}: sha256 differs from the annotator stage's record")
    return p, rec


def omnipath_by_resource(pickle_path: Path) -> tuple[pd.DataFrame, dict]:
    """Explode the cached OmniPath TF-target table into (primary resource, tf, target).

    PAZAR is split as in the annotator stage: regulons with > PAZAR_HT_MIN_TARGETS targets
    (binding-experiment regulons) form 'PAZAR-HT' (CHIP, held out)."""
    df = pd.read_pickle(pickle_path)
    df = df[["source_genesymbol", "target_genesymbol", "sources"]].copy()
    df = df[~df["source_genesymbol"].astype(str).str.contains("_|COMPLEX", regex=True)]
    df["tag"] = df["sources"].astype(str).str.split(";")
    df = df.explode("tag")
    df["annotator"] = df["tag"].str.split("_").str[0]
    pz = df[df["annotator"] == "PAZAR"]
    size = pz.groupby("source_genesymbol")["target_genesymbol"].nunique()
    big = set(size[size > PAZAR_HT_MIN_TARGETS].index)
    info = {"threshold_targets": PAZAR_HT_MIN_TARGETS, "pazar_ht_tfs": sorted(big)}
    if ANNOTATOR_PROVENANCE.exists():
        ref = json.loads(ANNOTATOR_PROVENANCE.read_text()).get("pazar_split", {}).get("raw_tfs_gt")
        if ref is not None:
            info["identical_to_annotator_stage_split"] = set(ref) == big
    df.loc[(df["annotator"] == "PAZAR") & df["source_genesymbol"].isin(big), "annotator"] = "PAZAR-HT"
    df["tf"] = C.normalize_series(df["source_genesymbol"])
    df["target"] = C.normalize_series(df["target_genesymbol"])
    df = df[df["tf"].isin(C.lambert_tfs())]
    df = df[["annotator", "tf", "target"]].drop_duplicates()
    meta = source_meta()
    fam = {a: (OMNIPATH_FAMILY.get(a) or meta.get(a, {}).get("family") or "LIT") for a in df["annotator"].unique()}
    df["family"] = df["annotator"].map(fam)
    df["held_out"] = df["family"].eq("CHIP")
    return df, info


# ----------------------------------------------------------------------------
# Coverage of the context and global universes
# ----------------------------------------------------------------------------


def universe_sets(allow_partial: bool = False) -> dict:
    """{universe: (TFs, genes)} for the three tissue context universes and the global
    universe; a missing universe raises unless ``allow_partial``."""
    out = {}
    for t in C.TISSUES:
        d = C.OUT / "universe" / t
        if require_input((d / "tfs.txt").exists() and (d / "panel.tsv").exists(),
                         f"tissue universe {d}/{{tfs.txt,panel.tsv}}", allow_partial):
            out[t] = (set((d / "tfs.txt").read_text().split()),
                      set(pd.read_csv(d / "panel.tsv", sep="\t")["gene"]))
    if require_input((GLOBAL_DIR / "tfs.txt").exists() and (GLOBAL_DIR / "genes.txt").exists(),
                     f"global universe {GLOBAL_DIR}/{{tfs.txt,genes.txt}}", allow_partial):
        out["global"] = (set((GLOBAL_DIR / "tfs.txt").read_text().split()),
                         set((GLOBAL_DIR / "genes.txt").read_text().split()))
    return out


def coverage_rows(built: dict, refs: dict, universes: dict) -> list:
    rows = []
    for u, (utfs, ugenes) in universes.items():
        for name, B in built.items():
            R = refs[name]
            e = B["edges"]
            tf_eval = sorted(utfs & set(R["tfs"]))
            ein = e[e["tf"].isin(tf_eval) & e["target"].isin(ugenes) & e["target"].isin(R["universe"])
                    & (e["tf"] != e["target"])]
            ev = sum(len((ugenes & R["universe"]) - {t}) for t in tf_eval)
            rows.append({"universe": u, "reference": name, "universe_tfs": len(utfs),
                         "universe_genes": len(ugenes), "evaluable_tfs": len(tf_eval),
                         "evaluable_edges": int(ev), "positives": int(len(ein)),
                         "density": (len(ein) / ev) if ev else np.nan,
                         "median_positives_per_tf": (float(ein.groupby("tf").size()
                                                           .reindex(tf_eval, fill_value=0).median())
                                                     if tf_eval else np.nan),
                         "tfs_with_positives": int(ein["tf"].nunique())})
    return rows


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------


def file_record(p: Path) -> dict:
    return {"path": str(p), "md5": md5sum(p),
            "mtime": datetime.fromtimestamp(p.stat().st_mtime, timezone.utc).isoformat(timespec="seconds")}


def _write_csv(df: pd.DataFrame, path: Path, **kw) -> None:
    df.to_csv(path, index=False, float_format=CSV_FLOAT_FORMAT, **kw)


def dependence_notes(ov: pd.DataFrame | None, group_defs: dict, sj: dict) -> list:
    """Dependence notes for provenance.json, with the counts of this run."""
    drop = list(C.PRESPECIFIED["lit_no_chipderived_drop"])
    ht = sj.get("lit_ht_like_regulons", [])
    nh = group_defs.get("all_held_in_no_ht", {})
    nc = group_defs.get("all_held_in_no_chipderived", {})
    kept = nh.get("records_kept_of_listed_sources", {})
    nfi = (sj.get("duplicate_drops") or {}).get("NFIRegulomeDB", {}).get("n_kept", "?")
    frac, low = "", "n/a"
    if ov is not None:
        lo = ov[ov["low_support"]]
        low = f"{len(lo)} rows, e.g. " + ", ".join(f"{a} vs {r}" for r, a in
                                                   zip(lo["reference"].head(3), lo["annotator"].head(3)))
        r = ov[(ov["reference"] == "enets2") & ov["annotator"].isin(drop)].set_index("annotator")
        frac = "; ".join(f"{a} {r.at[a, 'frac_edges_remap_pazarht_supported']:.2f}" for a in drop if a in r.index)
    return [
        ("Held-in literature sources that may contain records derived from binding experiments: ORegAnno "
         "(community-curated regulatory regions; CTCF/YY1/STAT1/GATA2 regulons look high-throughput), FANTOM4 "
         "(described by the resource as a literature gold standard; binding-experiment records not excluded), HTRIdb "
         "(DoRothEA excluded its large-scale ChIP-chip/ChIP-seq records; the CollecTRI copy may include them) and "
         "NFIRegulomeDB (" + str(nfi) + " non-duplicate edges in LIT_other = " + "+".join(sj.get("lit_other_components") or []) +
         "). Two sensitivity variants address them. lit_no_ht removes ONLY the " + str(len(ht)) + " regulons " +
         ", ".join(f"{s} {t}" for s, t in ht) + " (" + str(nh.get("pooled_edges_removed", "?")) +
         " pooled held-in edges lose all support); it leaves in place FANTOM4 (" + kept.get("FANTOM4", "?") +
         " records), HTRIdb (" + kept.get("HTRIdb", "?") + "), ORegAnno (" + kept.get("ORegAnno", "?") +
         ") and LIT_other (" + kept.get("LIT_other", "?") + "), so it covers these sources only partially. "
         "lit_no_chipderived (PRESPECIFIED['lit_no_chipderived_drop'] = " + ", ".join(drop) + ") removes every "
         "record of these four sources (" + str(nc.get("pooled_edges_removed", "?")) + " pooled held-in edges lose "
         "all support). annotator_overlap.csv rows all_held_in_no_ht and all_held_in_no_chipderived (with their "
         "co-support splits) quantify both; per-source frac_edges_remap_pazarht_supported against ENETS2: " +
         (frac or "n/a") + "."),
        ("Held-out ChIP sources sharing ENCODE experiments with the references: ReMap aggregates public ChIP-seq "
         "including ENCODE (shared with ENCODE TFBS, ENCODE-strict and ENETS2, which is built from the ENCODE "
         "Jan-2011 experiments), and OmniPath's ENCODE-proximal/-distal are the Gerstein et al. (2012) networks "
         "(ENCODE-proximal = ENETS2). Every overlap row whose edges come from, or were selected by co-support from, "
         "these sources is flagged (shares_encode_experiments); a ChIP-inclusive sensitivity fit (ReMap, PAZAR-HT) "
         "must not be presented as independently validated on ENCODE or ENETS2. The conservative headline for "
         "held-in agreement is the '" + CONSERVATIVE_HEADLINE + "' row (conservative_headline = True): held-in "
         "edges without ReMap/PAZAR-HT co-support; its lift is biased downward because co-support correlates with "
         "being a reference positive."),
        ("DoRothEA exports motif (HOCOMOCO, JASPAR) and coexpression (ARACNe-GTEx) records only when curated or ChIP "
         "support exists, so held-in MOTIF/COEXP agreement with held-out ChIP is partly ChIP-selected; the "
         "ENETS2 *_chip_any splits are uninformative by construction (ENCODE-proximal = ENETS2 membership)."),
        (f"Rows with < {MIN_SUPPORT_TFS} evaluable TFs or < {MIN_SUPPORT_POSITIVES} reference positives are marked "
         "low_support (" + low + "); their lifts and bootstrap CIs are not interpretable."),
    ]


SMALL_COLS = ["reference", "annotator", "family", "held_out", "n_edges_evaluable", "n_tfs_evaluable",
              "n_reference_positive", "precision", "tf_matched_base_rate", "lift_tf_matched",
              "lift_tf_matched_ci_lo", "lift_tf_matched_ci_hi", "target_matched_base_rate",
              "lift_target_matched", "lift_target_matched_ci_lo", "lift_target_matched_ci_hi",
              "lift_tf_matched_in_G", "lift_pooled", "reference_recall",
              "frac_edges_remap_pazarht_supported", "low_support", "shares_encode_experiments",
              "conservative_headline", "flag"]


def reference_table(references: dict, summary: dict) -> pd.DataFrame:
    """heldout_chip_table.csv: one row per reference (REFERENCES metadata + build summary)."""
    tab = []
    for name, meta in references.items():
        s = summary[name]
        tab.append({"reference": meta["label"], "key": name, "specificity": meta["specificity"],
                    "provenance": meta["provenance"], "url": meta["url"], "version": meta["version"],
                    "n_tfs": s["n_tfs"], "n_edges": s["n_edges"], "n_targets": s["n_targets"],
                    "n_gene_universe": s["n_gene_universe"], "density": round(s["density_in_universe"], 4),
                    "cell_lines": meta["cell_lines"], "citation": meta["citation"],
                    "role": "held-out validation only"})
    return pd.DataFrame(tab)


def annotator_overlap_small(src_tab: pd.DataFrame) -> pd.DataFrame:
    """heldout_chip_annotator_overlap.csv: the overlap rows that are informative, manuscript columns."""
    src_tab = src_tab[~src_tab["flag"].fillna("").astype(str).str.contains("uninformative by construction",
                                                                              regex=False)]
    return src_tab.reindex(columns=SMALL_COLS).round(4)


def tables_from_stored() -> None:
    """The three manuscript tables from the stored outputs of the last full run (provenance.json,
    annotator_overlap.csv or, if the annotators were unavailable then, omnipath_source_overlap.csv)."""
    prov = json.loads((OUT_DIR / "provenance.json").read_text())
    summary = prov["summary"]
    for name in prov["references"]:
        f = OUT_DIR / f"{name}.parquet"
        if not f.exists():
            raise FileNotFoundError(f"{f}: the stored reference of the last full run is missing")
    C.ensure_dir(C.TABLES)
    _write_csv(pd.DataFrame(summary["coverage"]), C.TABLES / "heldout_chip_coverage.csv")
    _write_csv(reference_table(prov["references"], summary), C.TABLES / "heldout_chip_table.csv")
    src = OUT_DIR / ("omnipath_source_overlap.csv" if summary.get("annotator_source") == "unavailable"
                     else "annotator_overlap.csv")
    ov = pd.read_csv(src)
    ov["flag"] = ov["flag"].fillna("")
    _write_csv(annotator_overlap_small(ov), C.TABLES / "heldout_chip_annotator_overlap.csv")
    print(f"wrote heldout_chip_coverage.csv, heldout_chip_table.csv, heldout_chip_annotator_overlap.csv "
          f"from {OUT_DIR / 'provenance.json'} and {src}", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--table-only", action="store_true",
                    help="rewrite the three manuscript tables from the stored outputs (no raw file read)")
    ap.add_argument("--download", action="store_true", help="download missing raw files")
    ap.add_argument("--wait-annotators-min", type=float, default=0.0,
                    help="poll this many minutes for annotator_edges.parquet")
    ap.add_argument("--n-boot", type=int, default=C.PRESPECIFIED["n_bootstrap"])
    ap.add_argument("--allow-partial", action="store_true",
                    help="warn instead of failing when an upstream input (annotators, universes) is missing")
    args = ap.parse_args(argv)
    if args.table_only:
        tables_from_stored()
        return
    seed = C.PRESPECIFIED["random_seed"]
    C.ensure_dir(OUT_DIR)
    C.ensure_dir(C.TABLES)
    tfs = C.lambert_tfs()

    prov = {"raw_files": ensure_raw(args.download)}

    gmt = read_harmonizome_gmt(RAW_FILES["harmonizome_encode_tfbs_gmt"]["path"])
    enc = build_encode(gmt, tfs)
    strict = build_encode_strict(enc, ENCODE_STRICT_K)
    raw2 = read_encodenets(RAW_FILES["enets2_proximal_filtered"]["path"])
    raw1 = read_encodenets(RAW_FILES["enets1_proximal_raw"]["path"])
    awg = read_awg_inputs(RAW_FILES["ucsc_encode_awg_tfbs_inputs"]["path"])
    cv = {ab: parse_cv_antibody(RAW_FILES[f"ucsc_encode_cv_antibody_{ab}"]["path"])
          for ab in sorted({s["encode_antibody"] for s in ENETS2_RELABEL.values()})}
    s4a = read_gerstein_s4a(RAW_FILES["gerstein2012_table_S4"]["path"])
    hidx = harmonizome_profile_index(raw2, raw1, gmt)
    relabel = confirm_enets2_relabels(raw2, raw1, gmt, awg, cv, s4a, hidx)
    en2 = build_enets2(raw2, raw1, tfs, relabel)
    cl = enets2_cell_lines(en2["edges"], awg)
    REFERENCES["enets2"]["cell_lines"] = cl["description"]
    s4_counts = dict(zip(s4a["factor"], s4a["n_targets"]))
    raw2_counts = raw2.groupby("factor_raw").size()
    shared = sorted(set(s4_counts) & set(raw2_counts.index))
    prov["enets2_relabels"] = relabel
    prov["enets2_cell_lines"] = cl
    prov["gerstein_S4A_vs_enets2"] = {
        "s4a_factors": len(s4_counts), "enets2_factors": int(len(raw2_counts)),
        "shared_labels": len(shared),
        "shared_labels_with_equal_target_count": int(sum(s4_counts[f] == raw2_counts[f] for f in shared)),
        "enets2_labels_not_in_s4a": sorted(set(raw2_counts.index) - set(s4_counts)),
        "s4a_labels_not_in_enets2": sorted(set(s4_counts) - set(raw2_counts.index))}
    prov["legacy_checks"] = legacy_checks(enc["legacy"], raw2)
    prov["build_steps"] = {"encode_tfbs": enc["steps"], "encode_strict": strict["steps"], "enets2": en2["steps"]}

    # ---- upstream universes (required unless --allow-partial)
    universes = universe_sets(args.allow_partial)
    G = universes["global"][1] if "global" in universes else None

    # ---- ENETS2 regulon-label concordance with Harmonizome (diagnostic, nothing dropped)
    conc = enets2_label_concordance(en2["edges"], hidx, universes)
    _write_csv(conc, OUT_DIR / "enets2_label_concordance.tsv", sep="\t")
    disc = conc[~conc["concordant"]]
    prov["enets2_label_concordance"] = {
        "rule": (f"best Harmonizome hg19 profile of the same factor reaches Bonferroni-adjusted hypergeometric "
                 f"p < {LABEL_CHECK_ALPHA} over all profiles (shared ENETS/Harmonizome universe)"),
        "n_tfs": int(len(conc)), "n_concordant": int(conc["concordant"].sum()),
        "not_concordant": disc[["tf", "n_targets", "label_check", "best_same_factor_shared_targets",
                                "top_profile_factor", "top_cytoband_region", "top_cytoband_region_frac"]
                               + [c for c in conc.columns if c.startswith("in_universe_")]].to_dict("records"),
        "action": "documented only; no prespecified rule drops a regulon",
    }

    # ---- write edge tables, TF lists, universes
    built = {"encode_tfbs": enc, "encode_strict": strict, "enets2": en2}
    refs, tf_rows, summary = {}, [], {}
    for name, B in built.items():
        B["edges"].to_parquet(OUT_DIR / f"{name}.parquet", index=False)
        (OUT_DIR / f"{name}_gene_universe.txt").write_text("\n".join(B["universe"]) + "\n")
        R = reference_index(B["edges"], B["universe"], assayed=B["tf_info"]["tf"])
        if G is not None:
            R["in_G"] = reference_index(B["edges"], sorted(set(B["universe"]) & G), assayed=B["tf_info"]["tf"])
        refs[name] = R
        ti = B["tf_info"].copy()
        ti["n_targets"] = ti["tf"].map(B["edges"].groupby("tf").size()).fillna(0).astype(int)
        ti["n_targets_in_universe"] = ti["tf"].map(lambda t: len(R["targets"].get(t, ())))
        ti["base_rate"] = ti["tf"].map(R["base"])
        ti["label_check"] = (ti["tf"].map(dict(zip(conc["tf"], conc["label_check"]))).fillna("")
                             if name == "enets2" else "")
        ti.insert(0, "reference", name)
        tf_rows.append(ti)
        (OUT_DIR / f"{name}_assayed_tfs.txt").write_text("\n".join(sorted(ti["tf"])) + "\n")
        e = B["edges"]
        summary[name] = {"n_tfs": int(len(R["tfs"])), "n_tfs_with_edges": int(e["tf"].nunique()),
                         "n_edges": int(len(e)), "n_targets": int(e["target"].nunique()),
                         "n_gene_universe": len(B["universe"]),
                         "edges_outside_universe": int((~e["target"].isin(R["universe"])).sum()),
                         "density_in_universe": float(len(R["keys"]) / sum(
                             R["n_universe"] - (1 if t in R["universe"] else 0) for t in R["tfs"])),
                         "median_targets_per_tf": float(e.groupby("tf").size().median()),
                         "target_degree_model": R["model"]}
        if "in_G" in R:
            summary[name]["in_G"] = {"n_genes": R["in_G"]["n_universe"],
                                     "density": float(len(R["in_G"]["keys"]) / sum(
                                         R["in_G"]["n_universe"] - (1 if t in R["in_G"]["universe"] else 0)
                                         for t in R["in_G"]["tfs"])),
                                     "target_degree_model": R["in_G"]["model"]}
    tf_df = pd.concat(tf_rows, ignore_index=True)
    _write_csv(tf_df, OUT_DIR / "assayed_tfs.tsv", sep="\t")
    _write_csv(pd.DataFrame(enc["dropped"] + strict["dropped"] + en2["dropped"]),
               OUT_DIR / "dropped_factors.tsv", sep="\t")
    e5 = enc["edges"]
    summary["encode_tfbs"]["edges_by_n_cell_lines"] = {
        int(k): int(v) for k, v in e5["n_cell_lines"].value_counts().sort_index().items()}
    summary["encode_tfbs"]["tfs_by_n_cell_lines"] = {
        int(k): int(v) for k, v in enc["tf_info"]["n_cell_lines"].value_counts().sort_index().items()}
    a, b = set(refs["encode_tfbs"]["tfs"]), set(refs["enets2"]["tfs"])
    en2e = en2["edges"][en2["edges"].tf.isin(a)]
    summary["cross_reference"] = {
        "shared_tfs": len(a & b), "encode_only_tfs": len(a - b), "enets2_only_tfs": len(b - a),
        "enets2_edges_in_encode_frac_shared_tfs": float(np.mean(
            C.edge_key(en2e["tf"], en2e["target"]).isin(refs["encode_tfbs"]["keys"])))}

    # ---- coverage of the tissue context universes and the global universe
    cov = pd.DataFrame(coverage_rows(built, refs, universes))
    _write_csv(cov, C.TABLES / "heldout_chip_coverage.csv")
    summary["coverage"] = cov.to_dict("records")
    prov["universe_files"] = {u: [file_record(p) for p in (
        [GLOBAL_DIR / "tfs.txt", GLOBAL_DIR / "genes.txt"] if u == "global" else
        [C.OUT / "universe" / u / "tfs.txt", C.OUT / "universe" / u / "panel.tsv"])] for u in universes}

    # ---- overlap with held-in annotators (primary) and OmniPath resources (secondary)
    ann = load_annotators(args.wait_annotators_min, args.allow_partial)
    summary["annotator_source"] = str(ANNOTATOR_EDGES) if ann is not None else "unavailable"
    ov, group_defs, sj = None, {}, sources_json()
    if ann is not None:
        prov["annotator_files"] = [file_record(p) for p in (ANNOTATOR_EDGES, ANNOTATOR_SOURCES)
                                   if p.exists()]
        chip_sources = sorted(ann.loc[ann["held_out"], "annotator"].unique())
        # pooled rows: each modality family (union of its sources), all held-in sources and the
        # two literature sensitivity groups (evidence variants lit_no_ht / lit_no_chipderived)
        fam = ann.assign(annotator="family:" + ann["family"])
        heldin = ann[~ann["held_out"]].assign(annotator="all_held_in", family="ALL")
        groups, group_defs = sensitivity_groups(ann, sj)
        sens = [g.assign(annotator=k, family="ALL") for k, g in groups.items()]
        pooled = pd.concat([fam, heldin, *sens], ignore_index=True).drop_duplicates(["annotator", "tf", "target"])
        # split held-in pooled rows by co-support from a held-out CHIP source (dependence channel:
        # DoRothEA's motif/coexpression records are only exported when curated or ChIP support exists)
        # (a) any held-out CHIP source; (b) only the peak-aggregation sources ReMap / PAZAR-HT,
        # because OmniPath's ENCODE-proximal/-distal tags are the Gerstein-2012 networks themselves
        chip = ann[ann["held_out"]]
        splits = {tag: (chip if srcs is None else chip[chip["annotator"].isin(srcs)])
                  for tag, srcs in CHIP_SPLITS.items()}
        chip_keys = set(C.edge_key(splits["chip_remap_pazarht"]["tf"], splits["chip_remap_pazarht"]["target"]))
        base = pooled[~pooled["held_out"]]
        for tag, ch in splits.items():
            keys = set(C.edge_key(ch["tf"], ch["target"]))
            hp = base.copy()
            sup = C.edge_key(hp["tf"], hp["target"]).isin(keys).to_numpy()
            hp["annotator"] = hp["annotator"] + np.where(sup, f"|with_{tag}", f"|without_{tag}")
            pooled = pd.concat([pooled, hp], ignore_index=True)
        ov = pd.concat([overlap_table(ann, "annotator", refs, seed, args.n_boot, ["family", "held_out"], chip_keys),
                        overlap_table(pooled, "annotator", refs, seed, args.n_boot, ["family", "held_out"],
                                      chip_keys)],
                       ignore_index=True)
        ov = overlap_flags(ov, chip_sources)
        ov = ov.sort_values(["reference", "held_out", "family", "annotator"]).reset_index(drop=True)
        _write_csv(ov, OUT_DIR / "annotator_overlap.csv")
        prov["sensitivity_groups"] = group_defs
    pk, pk_rec = omnipath_pickle()
    prov["omnipath_pickle"] = pk_rec
    om, pazar_info = omnipath_by_resource(pk)
    prov["omnipath_pazar_split"] = pazar_info
    # OmniPath's 'ENCODE-proximal' resource vs the official ENETS2 file (distributed labels)
    raw2n = set(C.edge_key(en2["edges_source_labels"]["tf"], en2["edges_source_labels"]["target"]))
    relab_keys = set(C.edge_key(en2["edges"]["tf"], en2["edges"]["target"]))
    ep = om[om["annotator"] == "ENCODE-proximal"]
    ep_keys = set(C.edge_key(ep["tf"], ep["target"]))
    om_keys = set(C.edge_key(om["tf"], om["target"]))
    inter = om_keys & raw2n
    summary["omnipath_encode_proximal_vs_enets2"] = {
        "n_encode_proximal_edges_lambert": len(ep_keys),
        "n_enets2_clean_edges_distributed_labels": len(raw2n),
        "frac_encode_proximal_in_enets2": len(ep_keys & raw2n) / max(len(ep_keys), 1),
        "n_omnipath_tf_target_edges_in_enets2": len(inter),
        "frac_of_those_tagged_encode_proximal": len(inter & ep_keys) / max(len(inter), 1),
        "frac_enets2_clean_edges_in_omnipath": len(inter) / max(len(raw2n), 1),
        "frac_enets2_clean_edges_in_omnipath_after_relabel": len(om_keys & relab_keys) / max(len(relab_keys), 1),
    }
    ovo = overlap_table(om, "annotator", refs, seed, args.n_boot, ["family", "held_out"])
    ovo = overlap_flags(ovo, sorted(om.loc[om["held_out"], "annotator"].unique()))
    _write_csv(ovo, OUT_DIR / "omnipath_source_overlap.csv")

    # ---- manuscript-facing tables
    _write_csv(reference_table(REFERENCES, summary), C.TABLES / "heldout_chip_table.csv")
    src_tab = ov if ov is not None else ovo
    _write_csv(annotator_overlap_small(src_tab), C.TABLES / "heldout_chip_annotator_overlap.csv")

    prov["summary"] = summary
    prov["references"] = REFERENCES
    prov["n_bootstrap"] = args.n_boot
    prov["random_seed"] = seed
    prov["target_degree_null"] = {
        "target_degree_bins": TARGET_DEGREE_BINS, "target_degree_bin_rule": TARGET_DEGREE_BIN_RULE,
        "target_degree_definition": TARGET_DEGREE_DEFINITION, "source": "common.PRESPECIFIED",
        "implementation": "src/r2/build_heldout_chip.py::reference_index / target_degree_model "
                          "(degree_bin, null_probability for arbitrary pairs)"}
    prov["overlap_support_rule"] = {"min_tfs_evaluable": MIN_SUPPORT_TFS,
                                    "min_reference_positives": MIN_SUPPORT_POSITIVES,
                                    "action": "marked (low_support), not suppressed"}
    prov["allow_partial"] = bool(args.allow_partial)
    prov["dependence_notes"] = dependence_notes(ov, group_defs, sj)
    C.write_json(prov, OUT_DIR / "provenance.json")
    print(json.dumps(summary, indent=2, default=str))
    print(json.dumps({k: {kk: v[kk] for kk in ("relabel_to", "check_target_overlap", "check_antibody_metadata",
                                                "confirmed")} for k, v in relabel.items()}, indent=2))


if __name__ == "__main__":
    main()
