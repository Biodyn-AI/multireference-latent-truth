"""Public evidence annotators split by primary resource (stage D2, ``outputs/r2/annotators``).

Inputs (frozen copies in ``outputs/r2/annotators/raw/``, listed with sha256 in ``raw/provenance.json``)
-----------------------------------------------------------------------------------------------------
1. **OmniPath DoRothEA A-D query** (``DOROTHEA_URL``): 278,677 records with per-record
   ``evidences`` JSON and the DoRothEA flag columns.  Every record carries a DoRothEA
   evidence, so the table is the complete DoRothEA A-D release as served by OmniPath;
   evidences of other OmniPath datasets attached to these records (``tf_target``,
   ``collectri``) are only visible on DoRothEA edges and are therefore *truncated* (not used
   for held-in annotators, see below).  The frozen copy is the omnipath Python client
   (1.0.12) cache entry of that exact request (the client's cache key is the md5 of the
   request URL, which identifies the query), accessed 2026-01-08 (file mtime).
2. **CollecTRI** from the OmniPath web service (``COLLECTRI_URL``), raw TSV downloaded
   2026-10-07 with the HTTP headers and the server ``/about`` string.  Same truncation
   logic: only ``collectri``-dataset evidences are complete in this download.
3. **TRRUST v2** raw human file (``TRRUST_URL``; local copy of ``C.TRRUST_RAW``).

Every run reads the files named in ``raw/provenance.json`` and checks their sha256; there
is no search of the OmniPath cache.  ``--freeze-legacy`` (run once) created the frozen
copies from the exact cache entry / file used originally; ``--refresh dorothea collectri
trrust`` re-downloads chosen inputs into new dated files and repoints the manifest.

Portable committed copies (fresh clone): ``outputs/`` is gitignored and the DoRothEA input
is a pandas pickle, so ``--export-portable`` (run once after freezing/refreshing) writes
zstd parquet copies of the DoRothEA and CollecTRI tables and a byte copy of the TRRUST
file to ``data/r2/inputs/annotator_raw/`` and records them (sha256, rows) under
``portable`` in the manifest.  A run reads the frozen ``raw/`` file when present and
otherwise the portable copy (``--from-portable`` forces the copies); when both exist the
build checks that they load to identical tables (``provenance.json``,
``portable_inputs_equivalent``).  The manifest is read from ``raw/provenance.json`` or, if
absent, from its committed copy ``data/r2/inputs/annotator_raw_inputs.json``.

Primary-resource split
----------------------
An edge (tf, target) belongs to primary resource *r* if any record of the edge carries an
evidence ``{resource: r, dataset: d, via: I}`` with ``(d, I)`` = ``('dorothea', 'DoRothEA')``
(DoRothEA channel) or ``('collectri', 'CollecTRI')`` (CollecTRI channel); TRRUST also from
the raw file.  The same primary resource seen through both integrators (TRRUST, HTRIdb
[= 'HTRI' in CollecTRI], TFactS, IntAct) is merged (edge union).  'TRED' and 'RegNetwork'
in the DoRothEA channel are identical edge sets (DoRothEA retrieved TRED via RegNetwork)
and are merged into 'TRED'.  Integrator tags ('DoRothEA', 'CollecTRI') and the derived
'DoRothEA-A' set (defined by agreement among other resources, including ChIP-seq) are not
annotators.  PAZAR is split by regulon size: regulons with > 1000 targets (24 TFs; Garcia-
Alonso et al. 2019 attribute these unusually large regulons to interactions extracted, at
least partly, from binding experiments) form 'PAZAR-HT' (CHIP family, held out); the
small regulons stay 'PAZAR' (LIT).

Duplicate evidence (decision log 2026-10-07, item 4): TRRD edges duplicated in TFactS
(all of them; DoRothEA obtained TRRD through the TFactS REF field) and NFIRegulomeDB edges
duplicated in PAZAR are dropped before LIT resources with < ``LIT_MIN_EDGES`` edges are
merged into 'LIT_other'.

``tf_target``-dataset evidences (TRRUST, HTRIdb, PAZAR, SIGNOR, ORegAnno, KEGG,
KEGG-MEDICUS, ENCODE-proximal, ENCODE-distal) reach us only attached to DoRothEA /
CollecTRI records; they are not used for held-in annotators.  ENCODE-proximal/-distal
(Gerstein et al. 2012) are saved with the held-out ChIP edges, flagged incomplete, and
never fitted (the held-out reference ENETS2 comes from the same ENCODE network).

Cleaning: complex endpoints (``COMPLEX:...``) dropped; symbols HGNC-normalised
(``C.normalize_series``); endpoints that are not HGNC-approved symbols dropped; tf must be
a Lambert et al. (2018) TF; self-loops dropped; edges de-duplicated per source.

Sensitivity definitions recorded here (used by ``evidence.py``)
--------------------------------------------------------------
* ``lit_ht_like_regulons`` (variant ``lit_no_ht``): exactly the 9 LIT regulons listed in
  ``data/r2/tables/lit_ht_like_regulons.csv`` and nothing else.  (a) 5 named regulons
  that look high-throughput: PAZAR HIF1A and ORegAnno CTCF/YY1/STAT1/GATA2 (decision log
  item 4).  They were named during the first data-layer verification, partly because
  ORegAnno overlaps the ENCODE tags (a held-out ChIP family) at 23% against 1-11% for the
  other LIT sources; this selection used held-out information and is disclosed.  (b) 4 large
  regulons with > ``LIT_HT_MAX_TARGETS`` targets, a rule added on 2026-10-07 in the
  phase-1b fix: ExTRI SP1/TP53 (text-mined, so "large", not high-throughput) and TRED
  MYC/E2F1 (PAZAR HIF1A also exceeds it).  ``lit_no_ht`` does NOT remove FANTOM4, the
  other ORegAnno regulons, NFIRegulomeDB (in LIT_other) or HTRIdb; those resources are
  removed in ``lit_no_chipderived``.
* ``lit_no_chipderived_drop`` (variant ``lit_no_chipderived``, decision log item 9): the
  whole held-in LIT sources ``C.PRESPECIFIED['lit_no_chipderived_drop']`` (FANTOM4, HTRIdb,
  ORegAnno, LIT_other) are removed, i.e. every held-in literature source that may contain
  records derived from binding experiments (decision log item 8), including the 7
  NFIRegulomeDB edges and the DoRothEA-reviews edges that make up LIT_other.  The
  conditioning set of the curated-conditional masks is recomputed without them.
* ``merge_sets``: ExTRI and NTNU.Curated as one source (NTNU.Curated is a manually curated
  subset of ExTRI sentences, Mueller-Dott et al. 2023).

Integration baselines (held-in evidence only; decision log item 3)
------------------------------------------------------------------
``integration_baselines.parquet`` keeps the released DoRothEA level and flags, adds ChIP
flags (``remap``, ``pazar_ht``, ``encode_tag``), held-in support flags, the DoRothEA level
recomputed from held-in evidence (``dorothea_level_heldin``) and CollecTRI counts without
DoRothEA-A.

Held-in support (decision log item 9): ``heldin_support_lit`` = a held-in LITERATURE
annotator reports the edge.  This is the only held-in support flag for filtering baselines.
Held-in MOTIF/COEXP reports are not support: they are visible only on DoRothEA A-D edges,
i.e. edges with curated or ChIP-seq support, so a MOTIF/COEXP report without literature
support is ChIP-conditioned (all such edges carry ReMap or PAZAR-HT support).  They are
flagged by ``motif_coexp_only_support`` (diagnostic; never a filter).  The former
``heldin_support`` (any held-in family) is removed.  ``dorothea_level_heldin`` /
``in_dorothea_heldin`` require held-in curated DoRothEA evidence *and*
``heldin_support_lit`` (the second condition is implied by the first and is asserted).
``collectri_curation_effort_heldin`` / ``collectri_n_references_heldin`` are the released
CollecTRI values restricted to ``in_collectri_heldin`` edges (missing for the DoRothEA-A-only
CollecTRI edges, whose released values come from non-held-in support).

DoRothEA level rule (Garcia-Alonso et al. 2019: A = all four evidence lines,
reviews, or >= 2 curated resources; B-D = curated and/or ChIP-seq with decreasing
additional evidence).  The B-D split is not spelled out in the paper; the rule below is the
one that reproduces the released levels from the evidence flags (agreement reported in
``provenance.json``)::

    A  reviews  or  n_curated >= 2  or  (curated & chip & tfbs & gtex)
    B  (curated & chip)  or  ((curated | chip) & tfbs & gtex)
    C  ((curated | chip) & tfbs)  or  (chip & gtex)
    D  curated | chip                  (curated only, ChIP only, curated & GTEx)
    -  otherwise (level E, not distributed)

n_curated counts distinct DoRothEA-channel curated resources, with RegNetwork = TRED and
TRRD = TFactS.  Held-in version: chip = False, PAZAR-HT is not curated, NFIRegulomeDB
duplicates count as PAZAR, and the level is missing unless the edge has held-in curated
DoRothEA evidence (TFBS/GTEx-only edges would be level E, which DoRothEA does not
distribute; in this release they are visible only through held-out ChIP support).

Outputs (``outputs/r2/annotators``; committed copies in ``data/r2/inputs``)
-----------------------------------------------------------------------
annotator_edges.parquet           source_id, family, lineage, tf, target, via_dorothea (+ 'annotator'
                                  alias) for ALL sources incl. held-out CHIP; held-in = family != CHIP
                                  -> committed copy data/r2/inputs/annotator_edges.parquet (no alias)
heldout_chip_edges.parquet        CHIP-family edges (+ 'complete', 'use_in_sensitivity' flags)
annotator_edges_by_channel.parquet  pre-merge provenance: label, channel, primary_resource, final_id,
                                  drop_reason, tf, target
dorothea_heldin_curated_edges.parquet  edges with >= 1 DoRothEA-channel evidence from a held-in LIT
                                  resource (= LIT rows with via_dorothea; conditioning set of the
                                  curated-conditional masks)
integration_baselines.parquet     -> committed copy data/r2/inputs/integration_baselines.parquet
sources.json                      per-source metadata, held-in list, sensitivity definitions
                                  -> committed copy data/r2/inputs/annotator_sources.json
provenance.json                   input manifest copy, access dates, filter counts, diagnostics
data/r2/inputs/annotator_raw_inputs.json   copy of raw/provenance.json (hashes of the raw inputs)
data/r2/tables/source_table.csv   manuscript source table (Q18/Q20)
data/r2/tables/source_components.csv  every OmniPath/TRRUST label -> source_id mapping with counts
data/r2/tables/lit_ht_like_regulons.csv  the 9 LIT regulons removed in the 'lit_no_ht' sensitivity
data/r2/inputs/annotator_raw/     portable copies of the three raw inputs (--export-portable)

Usage::

    .venv/bin/python subproject_24_multireference_latent_truth_model/src/r2/build_annotators.py
    .venv/bin/python subproject_24_multireference_latent_truth_model/src/r2/build_annotators.py --from-portable
    .venv/bin/python subproject_24_multireference_latent_truth_model/src/r2/build_annotators.py --refresh collectri --export-portable

Deterministic (no randomness); outputs depend only on the frozen raw files.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import pickle
import shutil
import sys
import urllib.request
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from r2 import common as C  # noqa: E402

OUT_DIR = C.OUT / "annotators"
RAW_DIR = OUT_DIR / "raw"
RAW_MANIFEST = RAW_DIR / "provenance.json"
COMMITTED_EDGES = C.INPUTS / "annotator_edges.parquet"
COMMITTED_BASELINES = C.INPUTS / "integration_baselines.parquet"
COMMITTED_SOURCES = C.INPUTS / "annotator_sources.json"
COMMITTED_RAW_MANIFEST = C.INPUTS / "annotator_raw_inputs.json"
PORTABLE_DIR = C.INPUTS / "annotator_raw"      # committed portable copies of the raw inputs

# The DoRothEA query exactly as issued by the omnipath Python client 1.0.12
# (``Dorothea.get(dorothea_levels=A-D, genesymbols=True, fields=[...])``, academic licence,
# human); md5(DOROTHEA_URL) is the client's cache key of the frozen copy.
DOROTHEA_URL = ("https://omnipathdb.org/interactions?datasets=dorothea&dorothea_levels=A%2CB%2CC%2CD"
                "&fields=curation_effort%2Cdorothea_chipseq%2Cdorothea_coexp%2Cdorothea_curated"
                "%2Cdorothea_level%2Cdorothea_tfbs%2Cevidences%2Creferences%2Csources"
                "&format=tsv&genesymbols=1&license=academic&organisms=9606")
COLLECTRI_URL = ("https://omnipathdb.org/interactions?datasets=collectri&genesymbols=1"
                 "&fields=sources,references,curation_effort,evidences,extra_attrs"
                 "&license=academic&organisms=9606")
TRRUST_URL = "https://www.grnpedia.org/trrust/data/trrust_rawdata.human.tsv"
DOROTHEA_REQUIRED = {"source", "target", "source_genesymbol", "target_genesymbol", "evidences",
                     "dorothea_level", "dorothea_curated", "dorothea_chipseq", "dorothea_tfbs",
                     "dorothea_coexp", "sources", "references", "curation_effort"}
DOROTHEA_FLAGS = ("dorothea_curated", "dorothea_chipseq", "dorothea_tfbs", "dorothea_coexp")

LIT_MIN_EDGES = 300           # LIT resources below this size are merged into LIT_other
PAZAR_HT_MIN_TARGETS = 1000   # Garcia-Alonso et al. 2019: only PAZAR has regulons > 1000 targets
GENOME_WIDE_FAMILIES = ("MOTIF", "COEXP", "CHIP")
# CHIP sources that may enter the ChIP-inclusive sensitivity fit (complete edge sets and
# not the publication the held-out ENETS2 reference comes from)
CHIP_SENSITIVITY_SOURCES = ("ReMap", "PAZAR-HT")
# duplicate evidence dropped from LIT_other components: component -> container source
DUPLICATE_DROPS = {"TRRD": "TFactS", "NFIRegulomeDB": "PAZAR"}
# LIT regulons removed in the 'lit_no_ht' sensitivity: 5 named high-throughput-like regulons
# (decision log item 4) + every LIT regulon with > LIT_HT_MAX_TARGETS targets (rule added
# 2026-10-07, phase-1b fix); 9 regulons in total
LIT_HT_NAMED = (("PAZAR", "HIF1A"), ("ORegAnno", "CTCF"), ("ORegAnno", "YY1"),
                ("ORegAnno", "STAT1"), ("ORegAnno", "GATA2"))
LIT_HT_MAX_TARGETS = 500
LIT_HT_DESCRIPTION = (
    "lit_no_ht removes exactly the 9 LIT regulons in data/r2/tables/lit_ht_like_regulons.csv: 5 named "
    "high-throughput-like regulons (PAZAR HIF1A; ORegAnno CTCF, YY1, STAT1, GATA2; decision log item 4; named "
    "during the first verification partly from ORegAnno's 23% overlap with the held-out ENCODE tags, disclosed) and "
    "4 further regulons with > 500 targets (ExTRI SP1, TP53 [text-mined: large, not high-throughput]; TRED MYC, "
    "E2F1; rule added 2026-10-07). It does not remove FANTOM4, the other ORegAnno regulons, NFIRegulomeDB "
    "(LIT_other) or HTRIdb; these are removed in lit_no_chipderived.")
# whole held-in LIT sources removed in the 'lit_no_chipderived' sensitivity (decision log item 9)
LIT_NO_CHIPDERIVED_DROP = tuple(C.PRESPECIFIED["lit_no_chipderived_drop"])
LIT_NO_CHIPDERIVED_DESCRIPTION = (
    "lit_no_chipderived removes the whole held-in LIT sources " + ", ".join(LIT_NO_CHIPDERIVED_DROP) +
    " (PRESPECIFIED['lit_no_chipderived_drop']; decision log item 9): every held-in literature source that may "
    "contain records derived from binding experiments (FANTOM4, HTRIdb incl. its CollecTRI copy, ORegAnno, and "
    "LIT_other = DoRothEA-reviews + the 7 non-duplicate NFIRegulomeDB edges); the curated conditioning set is "
    "recomputed without them.")
MERGE_SETS = {"ExTRI+NTNU.Curated": ["ExTRI", "NTNU.Curated"]}

# OmniPath resource label (per channel) -> primary source id
LABEL_TO_SOURCE = {
    "HTRI": "HTRIdb", "HTRIdb": "HTRIdb", "RegNetwork": "TRED", "TRED": "TRED",
    "TFactS": "TFactS", "TfactS": "TFactS",
}
INTEGRATOR_LABELS = {"DoRothEA", "CollecTRI", "CollecTRI2"}
DERIVED_LABELS = {"DoRothEA-A"}
# DoRothEA-channel labels that are not curated resources (evidence lines ChIP, TFBS, GTEx)
DOROTHEA_NONCURATED = {"ReMap", "HOCOMOCO", "JASPAR", "ARACNe-GTEx"}
# curated-resource identity for the DoRothEA n_curated count (released levels / held-in levels)
CURATED_COUNT_MAP = {"RegNetwork": "TRED", "TRRD": "TFactS"}
CURATED_COUNT_MAP_HELDIN = {"RegNetwork": "TRED", "TRRD": "TFactS", "NFIRegulomeDB": "PAZAR"}
LEVEL_RANK = {"A": 1, "B": 2, "C": 3, "D": 4}

DOROTHEA_REF = ("Garcia-Alonso et al. 2019", "10.1101/gr.240663.118")
COLLECTRI_REF = ("Muller-Dott et al. 2023", "10.1093/nar/gkad841")
COLLECTRI_REPO = "https://github.com/saezlab/CollecTRI"

# --------------------------------------------------------------------------------------
# Per-resource metadata: full name, family + criterion, primary citation, URL, version,
# notes.  Family = evidence modality of the resource's interactions as described by the
# resource / integrator publications.  URLs are those given in the publications (checked
# 2026-10-07; unreachable ones are marked).
# --------------------------------------------------------------------------------------
SOURCE_META = {
    # ---------------- literature-curated / text-mined (LIT) ----------------
    "TRRUST": dict(full_name="TRRUST v2: Transcriptional Regulatory Relationships Unravelled by Sentence-based Text mining (human)",
                   family="LIT", criterion="manual curation of small-scale experimental evidence from PubMed abstracts pre-selected by sentence-based text mining",
                   citation="Han et al. 2018", doi="10.1093/nar/gkx1013", resource_url="https://www.grnpedia.org/trrust/",
                   version="TRRUST v2 (2018); also via DoRothEA and CollecTRI", notes=""),
    "HTRIdb": dict(full_name="HTRIdb: Human Transcriptional Regulation Interactions database",
                   family="LIT", criterion="literature-curated experimentally verified interactions; DoRothEA excluded large-scale ChIP-chip/ChIP-seq records (HTRIdb v052016); 'HTRI' in CollecTRI",
                   citation="Bovolenta et al. 2012", doi="10.1186/1471-2164-13-405", resource_url="http://www.lbbc.ibb.unesp.br/htri (offline 2026-10-07)",
                   version="v052016 (DoRothEA); CollecTRI v1 copy",
                   notes="the CollecTRI copy may include records from binding experiments (disclosed; decision log item 8); whole source removed in the lit_no_chipderived sensitivity"),
    "TFactS": dict(full_name="TFactS: catalogue of TF-target regulations",
                   family="LIT", criterion="catalogue of TF-target regulations compiled manually from the literature",
                   citation="Essaghir et al. 2010", doi="10.1093/nar/gkq149", resource_url="http://www.tfacts.org (offline 2026-10-07)",
                   version="as integrated in DoRothEA (2019) and CollecTRI v1 (2023)",
                   notes="contains all TRRD edges (TRRD dropped as a duplicate)"),
    "IntAct": dict(full_name="IntAct molecular interaction database (protein-DNA interactions of TFs)",
                   family="LIT", criterion="expert curation of published protein-DNA interaction experiments",
                   citation="Orchard et al. 2014", doi="10.1093/nar/gkt1115", resource_url="https://www.ebi.ac.uk/intact/",
                   version="as integrated in DoRothEA (2019) and CollecTRI v1 (2023)", notes=""),
    "TRED": dict(full_name="TRED: Transcriptional Regulatory Element Database (retrieved via RegNetwork)",
                 family="LIT", criterion="curated TF-target relations (experimentally supported); obtained by DoRothEA via RegNetwork (Liu et al. 2015); OmniPath labels 'TRED' and 'RegNetwork' are identical edge sets",
                 citation="Jiang et al. 2007", doi="10.1093/nar/gkl1041", resource_url="http://rulai.cshl.edu/TRED (offline 2026-10-07)",
                 version="TRED 2007 via RegNetwork 2015 (DoRothEA)", notes="MYC/E2F1 regulons (> 500 targets) removed in lit_no_ht"),
    "KEGG": dict(full_name="KEGG pathway gene-expression relations (GErel)",
                 family="LIT", criterion="regulatory relations drawn manually in KEGG pathway maps from the literature (GErel, retrieved by DoRothEA with KEGGREST)",
                 citation="Kanehisa et al. 2012", doi="10.1093/nar/gkr988", resource_url="https://www.kegg.jp/",
                 version="as integrated in DoRothEA (2019)", notes=""),
    "PAZAR": dict(full_name="PAZAR regulatory sequence annotation database, small regulons (<= 1000 targets)",
                  family="LIT", criterion="curated TF / regulatory-sequence annotations (regulons <= 1000 targets)",
                  citation="Portales-Casamar et al. 2009", doi="10.1093/nar/gkn783", resource_url="http://www.pazar.info (offline 2026-10-07)",
                  version="as integrated in DoRothEA (2019)",
                  notes="HIF1A regulon (635 targets, about half of the edges) treated as high-throughput-like; removed in the lit_no_ht sensitivity"),
    "FANTOM4": dict(full_name="FANTOM4 literature gold-standard TF-target edges (edge.GoldStd_TF)",
                    family="LIT", criterion="literature-derived gold-standard TF-target edges of the FANTOM4 EdgeExpressDB",
                    citation="Kawaji et al. 2011", doi="10.1093/nar/gkq1112", resource_url="https://fantom.gsc.riken.jp/4/",
                    version="FANTOM4 (DoRothEA)",
                    notes="may include records derived from binding experiments (disclosed; decision log item 8); not touched by lit_no_ht; whole source removed in the lit_no_chipderived sensitivity"),
    "ORegAnno": dict(full_name="ORegAnno 3.0: Open Regulatory Annotation database (excl. PAZAR / NFIRegulomeDB records)",
                     family="LIT", criterion="community-curated regulatory annotations from the literature",
                     citation="Lesurf et al. 2016", doi="10.1093/nar/gkv1203", resource_url="http://www.oreganno.org",
                     version="ORegAnno 3.0 (DoRothEA)",
                     notes="CTCF/YY1/STAT1/GATA2 regulons (298 of 464 edges) look high-throughput (named partly from ORegAnno's overlap with the held-out ENCODE tags); these 4 regulons are removed in lit_no_ht, the whole source in lit_no_chipderived"),
    "TFe": dict(full_name="TFe: Transcription Factor Encyclopedia (expert-curated targets)",
                family="LIT", criterion="targets curated by TF experts from the literature",
                citation="Yusuf et al. 2012", doi="10.1186/gb-2012-13-3-r24", resource_url="http://www.cisreg.ca/tfe (offline 2026-10-07)",
                version="as integrated in DoRothEA (2019)", notes=""),
    "TRRD": dict(full_name="TRRD: Transcription Regulatory Regions Database",
                 family="LIT", criterion="manual annotation of experimentally characterised regulatory regions from the literature",
                 citation="Kolchanov et al. 2002", doi="10.1093/nar/30.1.312", resource_url="(no current URL; obtained by DoRothEA through the TFactS REF field)",
                 version="as integrated in DoRothEA (2019, via TFactS REF field)", notes="dropped: every edge duplicated in TFactS"),
    "DoRothEA-reviews": dict(full_name="DoRothEA manual curation of targets from TF-centric papers and reviews",
                             family="LIT", criterion="targets manually curated from TF-centric publications by the DoRothEA authors",
                             citation=DOROTHEA_REF[0], doi=DOROTHEA_REF[1], resource_url="https://saezlab.github.io/dorothea/",
                             version="DoRothEA (2019)", notes="in LIT_other (removed in lit_no_chipderived)"),
    "NFIRegulomeDB": dict(full_name="NFI-Regulome Database",
                          family="LIT", criterion="classified by DoRothEA as a literature-curated resource (NFI-regulated control regions)",
                          citation="Gronostajski et al. 2011", doi="10.1186/2043-9113-1-4", resource_url="(no current URL)",
                          version="as integrated in DoRothEA (2019)", notes="edges duplicated in PAZAR dropped; the remaining 7 edges are in LIT_other (removed in lit_no_chipderived)"),
    "ExTRI": dict(full_name="ExTRI: Extraction of Transcription Regulation Interactions from literature",
                  family="LIT", criterion="machine-learning text mining of PubMed abstracts (literature-derived, not experimental data)",
                  citation="Vazquez et al. 2022", doi="10.1016/j.bbagrm.2021.194778", resource_url="http://www.extri.org (as published; unreachable 2026-10-07)",
                  version="as integrated in CollecTRI v1 (2023)",
                  notes="NTNU.Curated is a curated subset of ExTRI sentences; merged in the lit_merged_extri_ntnu sensitivity; SP1/TP53 regulons (> 500 targets; text-mined, large rather than high-throughput) removed in lit_no_ht"),
    "NTNU.Curated": dict(full_name="NTNU manual sentence-level curation of DbTF-target interactions (Laegreid lab)",
                         family="LIT", criterion="manual validation of ExTRI sentences for validity of the TF-gene interaction and mode of regulation (about 20K sentences)",
                         citation=COLLECTRI_REF[0] + " (resource in preparation)", doi=COLLECTRI_REF[1],
                         resource_url=f"(not released separately; distributed within CollecTRI, {COLLECTRI_REPO})",
                         version="as integrated in CollecTRI v1 (2023)", notes="derived from ExTRI sentences"),
    "GEREDB": dict(full_name="GEREDB: Gene Expression Regulation Database",
                   family="LIT", criterion="gene-regulation relations curated by mining PubMed abstracts (subset with TFClass TF regulators)",
                   citation="Huang et al. 2019", doi="10.1142/S0219720019500240", resource_url="http://www.thua45.cn/geredb (as published; unreachable 2026-10-07)",
                   version="as integrated in CollecTRI v1 (2023)", notes=""),
    "SIGNOR": dict(full_name="SIGNOR: SIGnaling Network Open Resource (transcriptional regulation relations)",
                   family="LIT", criterion="manually curated causal relations from the literature",
                   citation="Perfetto et al. 2016", doi="10.1093/nar/gkv1048", resource_url="https://signor.uniroma2.it/",
                   version="as integrated in CollecTRI v1 (2023)", notes=""),
    "CytReg": dict(full_name="CytReg: cytokine transcriptional regulation database",
                   family="LIT", criterion="literature-curated TF-cytokine gene regulation",
                   citation="Carrasco Pro et al. 2018", doi="10.1093/nar/gky787", resource_url="https://cytreg.bu.edu",
                   version="as integrated in CollecTRI v1 (2023)", notes=""),
    "Pavlidis2021": dict(full_name="Experiment-level curation of TF-target interactions (Chu, Pavlidis et al.)",
                         family="LIT", criterion="experiment-level manual curation of published low-throughput perturbation, binding and reporter experiments (supplementary data)",
                         citation="Chu et al. 2021", doi="10.1371/journal.pcbi.1009484",
                         resource_url="https://github.com/PavlidisLab/dtri-curation; https://doi.org/10.5683/SP2/XLLWRL",
                         version="as integrated in CollecTRI v1 (2023)", notes=""),
    "GOA": dict(full_name="Gene Ontology Annotation database (TF regulation targets)",
                family="LIT", criterion="curated GO annotations with regulation-target extensions (literature-based)",
                citation="Huntley et al. 2015", doi="10.1093/nar/gku1113", resource_url="https://www.ebi.ac.uk/GOA/",
                version="as integrated in CollecTRI v1 (2023)", notes=""),
    # ---------------- motif scans (MOTIF) ----------------
    "HOCOMOCO": dict(full_name="HOCOMOCO v11 PWM scan of promoters (DoRothEA TFBS evidence; top 500 targets per TF)",
                     family="MOTIF", criterion="FIMO scan of promoters (-1000/+200 bp around the TSS) with HOCOMOCO v11 motifs; top 500 targets per TF (Garcia-Alonso et al. 2019)",
                     citation="Kulakovskiy et al. 2018", doi="10.1093/nar/gkx1106", resource_url="https://hocomoco11.autosome.org/",
                     version="HOCOMOCO v11 (DoRothEA 2019)",
                     notes="visible only on edges with curated or ChIP-seq support (DoRothEA A-D); curated-conditional observation"),
    "JASPAR": dict(full_name="JASPAR 2018 PWM scan of promoters (DoRothEA TFBS evidence; top 500 targets per TF)",
                   family="MOTIF", criterion="FIMO scan of promoters (-1000/+200 bp around the TSS) with JASPAR 2018 motifs; top 500 targets per TF (Garcia-Alonso et al. 2019)",
                   citation="Khan et al. 2018", doi="10.1093/nar/gkx1126", resource_url="https://jaspar.elixir.no/",
                   version="JASPAR 2018 (DoRothEA 2019)",
                   notes="visible only on edges with curated or ChIP-seq support (DoRothEA A-D); curated-conditional observation"),
    # ---------------- expression inference (COEXP) ----------------
    "ARACNe-GTEx": dict(full_name="ARACNe regulons inferred from GTEx v6 tissue expression (DoRothEA; interactions in >= 3 tissues)",
                        family="COEXP", criterion="ARACNe (mutual information) regulons inferred per GTEx v6 tissue; interactions present in >= 3 GTEx tissues (Garcia-Alonso et al. 2019)",
                        citation="Margolin et al. 2006; " + DOROTHEA_REF[0], doi="10.1038/nprot.2006.106", resource_url="https://gtexportal.org",
                        version="GTEx v6 (DoRothEA 2019)",
                        notes="visible only on edges with curated or ChIP-seq support (DoRothEA A-D); curated-conditional observation"),
    # ---------------- ChIP-derived (CHIP; held out) ----------------
    "ReMap": dict(full_name="ReMap 2018 ChIP-seq peaks scored by distance to the TSS (DoRothEA; top 500 targets per TF)",
                  family="CHIP", criterion="ChIP-seq binding (ENCODE + public ChIP-seq, uniformly reprocessed); peaks scored by TSS distance, top 500 targets per TF (Garcia-Alonso et al. 2019)",
                  citation="Cheneby et al. 2018", doi="10.1093/nar/gkx1092", resource_url="https://remap.univ-amu.fr/",
                  version="ReMap 2018 (DoRothEA 2019)", notes="contains ENCODE data"),
    "PAZAR-HT": dict(full_name="PAZAR regulons with > 1000 targets (binding-experiment-like)",
                     family="CHIP", criterion="the 24 PAZAR regulons with > 1000 targets; Garcia-Alonso et al. 2019 attribute these unusually large regulons to interactions extracted, at least partly, from binding experiments (assay type inferred from regulon size)",
                     citation="Portales-Casamar et al. 2009", doi="10.1093/nar/gkn783", resource_url="http://www.pazar.info (offline 2026-10-07)",
                     version="as integrated in DoRothEA (2019)", notes=""),
    "ENCODE-proximal": dict(full_name="ENCODE TF-target network, proximal edges (Gerstein et al. 2012), OmniPath tf_target",
                            family="CHIP", criterion="ENCODE ChIP-seq binding within promoter-proximal windows",
                            citation="Gerstein et al. 2012", doi="10.1038/nature11245", resource_url="http://encodenets.gersteinlab.org",
                            version="OmniPath tf_target attachment (truncated to DoRothEA/CollecTRI edges)", notes=""),
    "ENCODE-distal": dict(full_name="ENCODE TF-target network, distal edges (Gerstein et al. 2012), OmniPath tf_target",
                          family="CHIP", criterion="ENCODE ChIP-seq binding at distal regulatory elements",
                          citation="Gerstein et al. 2012", doi="10.1038/nature11245", resource_url="http://encodenets.gersteinlab.org",
                          version="OmniPath tf_target attachment (truncated to DoRothEA/CollecTRI edges)", notes=""),
}
TRUNCATED_CHIP = ("ENCODE-proximal", "ENCODE-distal")


# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------
def sha256(path: Path, block: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(block), b""):
            h.update(chunk)
    return h.hexdigest()


def mtime_iso(path: Path) -> str:
    return dt.datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")


def edge_set(df: pd.DataFrame) -> set:
    return set(zip(df["tf"], df["target"]))


def _to_bool(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s
    return s.astype(str).str.strip().str.lower().isin(["true", "1"])


# --------------------------------------------------------------------------------------
# raw-input manifest (freeze / refresh / load)
# --------------------------------------------------------------------------------------
def _read_manifest() -> dict:
    """raw/provenance.json, else its committed copy (fresh clone without outputs/)."""
    for p in (RAW_MANIFEST, COMMITTED_RAW_MANIFEST):
        if p.exists():
            return json.loads(p.read_text())
    raise FileNotFoundError(f"{RAW_MANIFEST} and {COMMITTED_RAW_MANIFEST} missing: run once with --freeze-legacy "
                            "or --refresh")


def _write_manifest(man: dict) -> None:
    man["inputs"] = dict(sorted(man["inputs"].items()))
    C.write_json(man, RAW_MANIFEST)


def _file_entry(path: Path, **kw) -> dict:
    return dict(file=path.name, sha256=sha256(path), bytes=path.stat().st_size, **kw)


def _http_head(url: str) -> dict:
    try:
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req, timeout=60) as r:
            return {"checked": dt.date.today().isoformat(), "status": r.status,
                    "last_modified": r.headers.get("Last-Modified"), "content_length": r.headers.get("Content-Length")}
    except Exception as e:  # pragma: no cover - network optional
        return {"checked": dt.date.today().isoformat(), "error": repr(e)}


def freeze_legacy() -> None:
    """One-time copy of the exact inputs of the original build into raw/ (no cache search)."""
    import omnipath
    C.ensure_dir(RAW_DIR)
    man = json.loads(RAW_MANIFEST.read_text()) if RAW_MANIFEST.exists() else {"inputs": {}}
    key = hashlib.md5(DOROTHEA_URL.encode("utf-8")).hexdigest()
    src = C.OMNIPATH_CACHE / f"{key}.pickle"            # the client's cache entry for DOROTHEA_URL
    date = mtime_iso(src)[:10]
    dst = RAW_DIR / f"dorothea_omnipath_AD_{date}.pickle"
    shutil.copy2(src, dst)
    man["inputs"]["dorothea"] = _file_entry(
        dst, format="pickle", url=DOROTHEA_URL, access_date=date,
        obtained=(f"omnipath Python client {omnipath.__version__} cache entry {src.name} "
                  "(cache key = md5 of the request URL, verified)"),
        origin_path=str(src), origin_mtime=mtime_iso(src))
    craw = RAW_DIR / "collectri_omnipath_2026-10-07.tsv"
    man["inputs"]["collectri"] = _collectri_entry(craw, "2026-10-07")
    dst = RAW_DIR / "trrust_rawdata.human.tsv"
    shutil.copy2(C.TRRUST_RAW, dst)
    man["inputs"]["trrust"] = _file_entry(
        dst, format="tsv", url=TRRUST_URL, access_date=f"on or before {mtime_iso(C.TRRUST_RAW)[:10]} (file mtime)",
        obtained="local copy of the TRRUST v2 raw human file used throughout the project",
        origin_path=str(C.TRRUST_RAW), origin_mtime=mtime_iso(C.TRRUST_RAW), remote_head=_http_head(TRRUST_URL))
    man["frozen"] = dt.datetime.now().isoformat(timespec="seconds")
    _write_manifest(man)
    print(json.dumps(man, indent=1), flush=True)


def _collectri_entry(path: Path, date: str) -> dict:
    e = _file_entry(path, format="tsv", url=COLLECTRI_URL, access_date=date, obtained="HTTP GET (urllib)")
    hdr = RAW_DIR / f"collectri_headers_{date}.txt"
    about = RAW_DIR / f"omnipath_about_{date}.txt"
    if hdr.exists():
        e["http_headers_file"] = hdr.name
    if about.exists():
        e["omnipath_server"] = about.read_text().strip()
    return e


def _download(url: str, out: Path) -> str:
    with urllib.request.urlopen(url, timeout=1800) as r:
        data = r.read()
        headers = "\n".join(f"{k}: {v}" for k, v in r.headers.items())
    out.write_bytes(data)
    return f"GET {url}\n{headers}\n"


def refresh(which: list) -> None:
    """Re-download the chosen inputs into new dated files and repoint the manifest."""
    C.ensure_dir(RAW_DIR)
    man = json.loads(RAW_MANIFEST.read_text()) if RAW_MANIFEST.exists() else {"inputs": {}}
    today = dt.date.today().isoformat()
    if "collectri" in which:
        out = RAW_DIR / f"collectri_omnipath_{today}.tsv"
        (RAW_DIR / f"collectri_headers_{today}.txt").write_text(_download(COLLECTRI_URL, out))
        with urllib.request.urlopen("https://omnipathdb.org/about", timeout=60) as r:
            (RAW_DIR / f"omnipath_about_{today}.txt").write_bytes(r.read())
        man["inputs"]["collectri"] = _collectri_entry(out, today)
    if "dorothea" in which:
        out = RAW_DIR / f"dorothea_omnipath_AD_{today}.tsv"
        (RAW_DIR / f"dorothea_headers_{today}.txt").write_text(_download(DOROTHEA_URL, out))
        man["inputs"]["dorothea"] = _file_entry(out, format="tsv", url=DOROTHEA_URL, access_date=today,
                                                obtained="HTTP GET (urllib)",
                                                http_headers_file=f"dorothea_headers_{today}.txt")
    if "trrust" in which:
        out = RAW_DIR / f"trrust_rawdata.human_{today}.tsv"
        (RAW_DIR / f"trrust_headers_{today}.txt").write_text(_download(TRRUST_URL, out))
        man["inputs"]["trrust"] = _file_entry(out, format="tsv", url=TRRUST_URL, access_date=today,
                                              obtained="HTTP GET (urllib)",
                                              http_headers_file=f"trrust_headers_{today}.txt")
    man["refreshed"] = {**man.get("refreshed", {}), **{w: today for w in which}}
    _write_manifest(man)


def _read_table(path: Path, fmt: str, key: str) -> pd.DataFrame:
    """Read one raw input; object-column nulls are normalised to NaN so that the frozen file
    and its portable parquet copy (which stores None) load to identical tables."""
    if key == "trrust":
        return pd.read_csv(path, sep="\t", header=None, names=["tf_raw", "target_raw", "mode", "pmids"], dtype=str)
    if fmt == "pickle":
        with open(path, "rb") as fh:
            df = pickle.load(fh)
    elif fmt == "parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path, sep="\t", low_memory=False)
    df = pd.DataFrame(df).reset_index(drop=True)
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].where(df[c].notna(), np.nan)
    return df


def _checked(path: Path, expected: str) -> Path:
    h = sha256(path)
    if h != expected:
        raise ValueError(f"sha256 mismatch for {path}: {h} != {expected}")
    return path


def _resolve_input(key: str, e: dict, from_portable: bool) -> tuple[Path, str, str]:
    """(path, format, 'raw' | 'portable') of one input; sha256 checked."""
    raw = RAW_DIR / e["file"]
    port = e.get("portable")
    if not from_portable and raw.exists():
        return _checked(raw, e["sha256"]), e["format"], "raw"
    if port and (C.SP / port["path"]).exists():
        return _checked(C.SP / port["path"], port["sha256"]), port["format"], "portable"
    raise FileNotFoundError(f"{key}: neither {raw} nor a portable copy is available "
                            "(--freeze-legacy / --refresh, then --export-portable)")


def load_inputs(from_portable: bool = False) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict, dict]:
    """Read the three raw inputs named in the manifest after checking their sha256.

    Returns (dorothea, collectri, trrust, manifest, load_info).  When the frozen raw file is
    used and a portable copy exists, the copy is loaded too and must give an identical table.
    """
    man = _read_manifest()
    tabs, info = {}, {}
    for k in ("dorothea", "collectri", "trrust"):
        e = man["inputs"][k]
        p, fmt, origin = _resolve_input(k, e, from_portable)
        tabs[k] = _read_table(p, fmt, k)
        info[k] = {"loaded_from": origin, "relpath": str(p.relative_to(C.SP)), "format": fmt}
        port = e.get("portable")
        if origin == "raw" and port and (C.SP / port["path"]).exists():
            q = _checked(C.SP / port["path"], port["sha256"])
            other = _read_table(q, port["format"], k)
            pd.testing.assert_frame_equal(tabs[k], other, check_dtype=True)
            info[k]["portable_copy_identical"] = True
    dor = tabs["dorothea"]
    missing = DOROTHEA_REQUIRED - set(dor.columns)
    if missing:
        raise KeyError(f"DoRothEA input lacks columns {sorted(missing)}")
    for f in DOROTHEA_FLAGS:
        dor[f] = _to_bool(dor[f])
    return dor, tabs["collectri"], tabs["trrust"], man, info


def export_portable() -> None:
    """Write portable copies of the frozen raw inputs to data/r2/inputs/annotator_raw/ and record
    them (path relative to the subproject root, sha256, rows) under 'portable' in the manifest."""
    import pyarrow
    if not RAW_MANIFEST.exists():
        raise FileNotFoundError(f"{RAW_MANIFEST} missing: the portable copies are exported from the frozen raw files")
    man = json.loads(RAW_MANIFEST.read_text())
    C.ensure_dir(PORTABLE_DIR)
    keep = set()
    for k, e in sorted(man["inputs"].items()):
        src = _checked(RAW_DIR / e["file"], e["sha256"])
        if k == "trrust":
            dst = PORTABLE_DIR / e["file"]
            shutil.copyfile(src, dst)
            fmt, n = "tsv", len(_read_table(dst, "tsv", k))
            note = "byte copy of the frozen file"
        else:
            df = _read_table(src, e["format"], k)
            dst = PORTABLE_DIR / (Path(e["file"]).stem + ".parquet")
            df.to_parquet(dst, index=False, compression="zstd")
            fmt, n = "parquet", len(df)
            note = (f"zstd parquet export of the frozen {e['format']} file (pandas {pd.__version__}, pyarrow "
                    f"{pyarrow.__version__}); loads to the identical table (object nulls as NaN)")
        keep.add(dst.name)
        e["portable"] = {"path": str(dst.relative_to(C.SP)), "format": fmt, "sha256": sha256(dst),
                         "bytes": dst.stat().st_size, "n_rows": int(n), "note": note}
    for q in PORTABLE_DIR.iterdir():
        if q.is_file() and q.name not in keep:
            q.unlink()
    _write_manifest(man)
    C.write_json(man, COMMITTED_RAW_MANIFEST)
    print(json.dumps({k: v["portable"] for k, v in man["inputs"].items()}, indent=1), flush=True)


def explode_evidences(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (record, resource, dataset, via) from the OmniPath evidences JSON.

    All four evidence categories (positive / negative sign, directed, undirected)
    are pooled: membership of an edge in a resource does not depend on the sign.
    """
    rows = []
    for i, s in enumerate(df["evidences"].values):
        e = json.loads(s)
        seen = set()
        for cat in ("positive", "negative", "directed", "undirected"):
            for ev in e.get(cat, []) or []:
                key = (ev["resource"], ev["dataset"], ev["via"] or "")
                if key not in seen:
                    seen.add(key)
                    rows.append((i, *key, ";".join(ev.get("references") or [])))
    ev = pd.DataFrame(rows, columns=["rec", "resource", "dataset", "via", "refs"])
    meta = df[["source", "target", "source_genesymbol", "target_genesymbol"]].reset_index(drop=True)
    return ev.join(meta, on="rec")


class Cleaner:
    """Symbol cleaning shared by all channels; keeps per-step drop counts."""

    def __init__(self):
        amap = C._alias_map()
        self.approved = frozenset(amap.values())
        self.tfs = C.lambert_tfs()

    def clean(self, df: pd.DataFrame, tf_sym: str, tg_sym: str, tf_id: str | None = None,
              tg_id: str | None = None) -> tuple[pd.DataFrame, dict]:
        n0 = len(df)
        cnt = {"n_records": n0}
        if tf_id is not None:
            cx = df[tf_id].astype(str).str.startswith("COMPLEX") | df[tg_id].astype(str).str.startswith("COMPLEX")
            df = df[~cx]
            cnt["drop_complex"] = int(cx.sum())
        out = df.copy()
        out["tf"] = C.normalize_series(out[tf_sym].fillna(""))
        out["target"] = C.normalize_series(out[tg_sym].fillna(""))
        ng = ~(out["tf"].isin(self.approved) & out["target"].isin(self.approved))
        cnt["drop_non_hgnc"] = int(ng.sum())
        out = out[~ng]
        ntf = ~out["tf"].isin(self.tfs)
        cnt["drop_tf_not_lambert"] = int(ntf.sum())
        out = out[~ntf]
        sl = out["tf"] == out["target"]
        cnt["drop_self_loop"] = int(sl.sum())
        out = out[~sl]
        cnt["n_kept_records"] = int(len(out))
        return out, cnt


# --------------------------------------------------------------------------------------
# DoRothEA confidence-level rule
# --------------------------------------------------------------------------------------
def dorothea_level_rule(n_curated, reviews, chip, tfbs, coexp) -> np.ndarray:
    """DoRothEA level (object array of 'A'-'D' or None) from per-edge evidence (see module doc)."""
    n_cur = np.asarray(n_curated, dtype=np.int64)
    rev = np.asarray(reviews, dtype=bool)
    chip = np.asarray(chip, dtype=bool)
    tfbs = np.asarray(tfbs, dtype=bool)
    coexp = np.asarray(coexp, dtype=bool)
    cur = n_cur >= 1
    a = rev | (n_cur >= 2) | (cur & chip & tfbs & coexp)
    b = (cur & chip) | ((cur | chip) & tfbs & coexp)
    c = ((cur | chip) & tfbs) | (chip & coexp)
    d = cur | chip
    out = np.full(len(n_cur), None, dtype=object)
    for lev, m in (("D", d), ("C", c), ("B", b), ("A", a)):   # best level wins
        out[m] = lev
    return out


# --------------------------------------------------------------------------------------
# main build
# --------------------------------------------------------------------------------------
def build(from_portable: bool = False) -> None:
    C.ensure_dir(OUT_DIR)
    C.ensure_dir(C.TABLES)
    C.ensure_dir(C.INPUTS)
    cleaner = Cleaner()
    dor, col, tr, man, load_info = load_inputs(from_portable)
    prov: dict = {"raw_inputs": man, "inputs_loaded": load_info,
                  "portable_inputs_equivalent": (True if all(v.get("portable_copy_identical") for v in load_info.values())
                                                 else "not checked (portable copies loaded or absent)")}
    d_in, c_in = man["inputs"]["dorothea"], man["inputs"]["collectri"]
    prov["dorothea"] = {"n_records": int(len(dor)), "access_date": d_in["access_date"], "url": d_in["url"]}
    # compatibility key read by build_heldout_chip.py: the frozen DoRothEA pickle (not the cache).
    # 'path' is absolute on the machine that ran the build; 'relpath' is relative to the subproject root.
    prov["dorothea_cache"] = {"path": str(RAW_DIR / d_in["file"]), "relpath": str((RAW_DIR / d_in["file"]).relative_to(C.SP)),
                              "relative_to": "subproject root", "sha256": d_in["sha256"],
                              "mtime_access_date": d_in["access_date"], "format": d_in["format"],
                              "portable_relpath": d_in.get("portable", {}).get("path")}
    prov["collectri"] = {"n_records": int(len(col)), "access_date": c_in["access_date"]}
    print(f"[inputs] DoRothEA {len(dor):,} records; CollecTRI {len(col):,} records; TRRUST {len(tr):,} rows", flush=True)

    ev_d = explode_evidences(dor)
    ev_c = explode_evidences(col)
    # reconciliation of the evidences JSON with the 'sources' column of the DoRothEA table
    tok = defaultdict(int)
    for s in dor["sources"].astype(str):
        for t in s.split(";"):
            tok[t] += 1
    ev_tok = (ev_d["resource"] + np.where(ev_d["via"] != "", "_" + ev_d["via"], "")).value_counts()
    prov["dorothea_sources_vs_evidences"] = {
        t: {"sources_column": int(tok.get(t, 0)), "evidences_json": int(ev_tok.get(t, 0))}
        for t in sorted(set(tok) | set(ev_tok.index))}

    # ---------------- channel records ----------------
    frames = []
    comp_rows = []    # component table rows (label x channel)
    pazar_big: set = set()

    def add_channel(ev: pd.DataFrame, dataset: str, via: str, channel: str):
        sub = ev[(ev["dataset"] == dataset) & (ev["via"] == via)]
        for lab, g in sub.groupby("resource", sort=True):
            if lab in INTEGRATOR_LABELS:
                continue
            raw_edges = g[["source_genesymbol", "target_genesymbol"]].drop_duplicates()
            if lab in DERIVED_LABELS:
                comp_rows.append(dict(label=lab, channel=channel, source_id="", status="excluded: derived integration set",
                                      n_raw_edges=len(raw_edges), n_clean_edges=0))
                continue
            sid = LABEL_TO_SOURCE.get(lab, lab)
            g = g.assign(source_id=sid)
            raw_edges = raw_edges.assign(source_id=sid)
            if lab == "PAZAR" and channel == "DoRothEA":
                # regulon size on the raw (as distributed) PAZAR regulons
                size = raw_edges.groupby("source_genesymbol")["target_genesymbol"].nunique()
                big = set(size[size > PAZAR_HT_MIN_TARGETS].index)
                pazar_big.update(big)
                g.loc[g["source_genesymbol"].isin(big), "source_id"] = "PAZAR-HT"
                raw_edges.loc[raw_edges["source_genesymbol"].isin(big), "source_id"] = "PAZAR-HT"
                prov["pazar_split"] = {"threshold_targets": PAZAR_HT_MIN_TARGETS,
                                       "n_raw_regulons_gt": len(big), "raw_tfs_gt": sorted(big),
                                       "n_raw_edges_gt": int(raw_edges["source_genesymbol"].isin(big).sum()),
                                       "n_raw_edges_le": int((~raw_edges["source_genesymbol"].isin(big)).sum())}
            for s2, g2 in g.groupby("source_id"):
                cl, cnt = cleaner.clean(g2, "source_genesymbol", "target_genesymbol", "source", "target")
                cl = cl.assign(label=lab, channel=channel)
                comp_rows.append(dict(label=lab, channel=channel, source_id=s2, status="used",
                                      n_raw_edges=int((raw_edges["source_id"] == s2).sum()),
                                      n_clean_edges=len(cl[["tf", "target"]].drop_duplicates()), **cnt))
                frames.append(cl[["label", "channel", "source_id", "tf", "target"]])

    add_channel(ev_d, "dorothea", "DoRothEA", "DoRothEA")
    add_channel(ev_c, "collectri", "CollecTRI", "CollecTRI")

    # TRRUST raw file
    cl, cnt = cleaner.clean(tr, "tf_raw", "target_raw")
    cl = cl.assign(label="TRRUST", channel="TRRUST-file", source_id="TRRUST")
    frames.append(cl[["label", "channel", "source_id", "tf", "target"]])
    comp_rows.append(dict(label="TRRUST", channel="TRRUST-file", source_id="TRRUST", status="used",
                          n_raw_edges=len(tr[["tf_raw", "target_raw"]].drop_duplicates()),
                          n_clean_edges=len(cl[["tf", "target"]].drop_duplicates()), **cnt))

    # tf_target attachments (truncated): ENCODE kept as held-out CHIP (flagged), rest excluded
    tt = pd.concat([ev_d[(ev_d["dataset"] == "tf_target")], ev_c[(ev_c["dataset"] == "tf_target")]])
    for lab, g in tt.groupby("resource", sort=True):
        raw_edges = g[["source_genesymbol", "target_genesymbol"]].drop_duplicates()
        if lab in TRUNCATED_CHIP:
            cl, cnt = cleaner.clean(g, "source_genesymbol", "target_genesymbol", "source", "target")
            cl = cl.assign(label=lab, channel="OmniPath-tf_target", source_id=lab)
            frames.append(cl[["label", "channel", "source_id", "tf", "target"]])
            comp_rows.append(dict(label=lab, channel="OmniPath-tf_target", source_id=lab,
                                  status="held-out CHIP; truncated to DoRothEA/CollecTRI edges; never fitted",
                                  n_raw_edges=len(raw_edges), n_clean_edges=len(cl[["tf", "target"]].drop_duplicates()), **cnt))
        else:
            sid = LABEL_TO_SOURCE.get(lab, lab)
            why = ("excluded: truncated attachment (complete copy used via integrator channel)"
                   if sid in {"TRRUST", "HTRIdb", "PAZAR", "SIGNOR", "ORegAnno", "KEGG"}
                   else "excluded: truncated attachment, no complete copy available")
            comp_rows.append(dict(label=lab, channel="OmniPath-tf_target", source_id="", status=why,
                                  n_raw_edges=len(raw_edges), n_clean_edges=0))

    long = pd.concat(frames, ignore_index=True).drop_duplicates(["label", "channel", "source_id", "tf", "target"])
    long = long.sort_values(["source_id", "channel", "label", "tf", "target"]).reset_index(drop=True)
    missing = sorted(set(long["source_id"]) - set(SOURCE_META))
    if missing:
        raise KeyError(f"no metadata for sources {missing}")
    fam = {s: SOURCE_META[s]["family"] for s in long["source_id"].unique()}

    # ---------------- duplicate evidence (decision log item 4) ----------------
    long["drop_reason"] = ""
    dup_info = {}
    for comp_src, container in DUPLICATE_DROPS.items():
        cont = edge_set(long[long["source_id"] == container])
        rows = long["source_id"] == comp_src
        e = list(zip(long.loc[rows, "tf"], long.loc[rows, "target"]))
        dup = np.array([x in cont for x in e], dtype=bool)
        idx = long.index[rows][dup]
        long.loc[idx, "drop_reason"] = f"duplicate of {container}"
        n_e = len(set(e))
        n_dup = len({x for x, d in zip(e, dup) if d})
        dup_info[comp_src] = {"container": container, "n_edges": n_e, "n_duplicated_dropped": n_dup,
                              "n_kept": n_e - n_dup}
    prov["duplicate_drops"] = dup_info
    kept = long[long["drop_reason"] == ""]

    # ---------------- families, LIT_other merge, lineage ----------------
    size = kept.drop_duplicates(["source_id", "tf", "target"]).groupby("source_id").size()
    small_lit = sorted(s for s in size.index if fam[s] == "LIT" and size[s] < LIT_MIN_EDGES)
    long["final_id"] = np.where(long["drop_reason"] != "", "",
                                np.where(long["source_id"].isin(small_lit), "LIT_other", long["source_id"]))
    fam["LIT_other"] = "LIT"
    kept = long[long["drop_reason"] == ""]

    def lineage_of(channels: set) -> str:
        ch = {c for c in channels if c in ("DoRothEA", "CollecTRI")}
        if "TRRUST-file" in channels:
            ch.add("TRRUST-file")
        if channels == {"OmniPath-tf_target"}:
            return "OmniPath-tf_target"
        if ch == {"DoRothEA"}:
            return "DoRothEA"
        if ch == {"CollecTRI"}:
            return "CollecTRI"
        return "Multi-integrator"

    lineage = {s: lineage_of(set(g["channel"])) for s, g in kept.groupby("final_id")}
    long["family"] = long["final_id"].map(fam)
    long["lineage"] = long["final_id"].map(lineage)
    long.rename(columns={"source_id": "primary_resource"})[
        ["label", "channel", "primary_resource", "final_id", "drop_reason", "family", "lineage", "tf", "target"]
    ].to_parquet(OUT_DIR / "annotator_edges_by_channel.parquet", index=False)

    kept = long[long["drop_reason"] == ""].assign(via_d=lambda x: x["channel"] == "DoRothEA")
    edges = (kept.groupby(["final_id", "tf", "target"], sort=True)
             .agg(family=("family", "first"), lineage=("lineage", "first"), via_dorothea=("via_d", "any"))
             .reset_index().rename(columns={"final_id": "source_id"}))
    edges = edges[["source_id", "family", "lineage", "tf", "target", "via_dorothea"]]
    edges.to_parquet(COMMITTED_EDGES, index=False, compression="zstd")
    edges.assign(annotator=edges["source_id"]).to_parquet(OUT_DIR / "annotator_edges.parquet", index=False)

    chip = edges[edges["family"] == "CHIP"].drop(columns=["via_dorothea"]).copy()
    chip["complete"] = ~chip["source_id"].isin(TRUNCATED_CHIP)
    chip["use_in_sensitivity"] = chip["source_id"].isin(CHIP_SENSITIVITY_SOURCES)
    chip.to_parquet(OUT_DIR / "heldout_chip_edges.parquet", index=False)

    # conditioning set for the curated-conditional masks: edges with DoRothEA-channel evidence
    # from a held-in LIT resource (DoRothEA releases them irrespective of TFBS/GTEx support)
    dcur = (edges.loc[(edges["family"] == "LIT") & edges["via_dorothea"], ["tf", "target"]]
            .drop_duplicates().sort_values(["tf", "target"]).reset_index(drop=True))
    dcur_long = kept[(kept["channel"] == "DoRothEA") & (kept["family"] == "LIT")][["tf", "target"]].drop_duplicates()
    assert edge_set(dcur) == edge_set(dcur_long)
    dcur.to_parquet(OUT_DIR / "dorothea_heldin_curated_edges.parquet", index=False)

    # ---------------- sensitivity definitions ----------------
    lit = edges[edges["family"] == "LIT"]
    reg = lit.groupby(["source_id", "tf"]).size().rename("n_targets").reset_index()
    named = set(LIT_HT_NAMED)
    reg["named"] = [(s, t) in named for s, t in zip(reg["source_id"], reg["tf"])]
    reg["large"] = reg["n_targets"] > LIT_HT_MAX_TARGETS
    ht = reg[reg["named"] | reg["large"]].copy()
    ht["reason"] = np.where(ht["named"] & ht["large"], f"named high-throughput-like; > {LIT_HT_MAX_TARGETS} targets",
                            np.where(ht["named"], "named high-throughput-like regulon",
                                     f"large regulon (> {LIT_HT_MAX_TARGETS} targets)"))
    ht["n_source_edges"] = ht["source_id"].map(lit.groupby("source_id").size())
    ht["frac_of_source"] = ht["n_targets"] / ht["n_source_edges"]
    ht = ht.sort_values(["source_id", "tf"])[["source_id", "tf", "n_targets", "n_source_edges", "frac_of_source", "reason"]]
    missing_named = named - set(zip(ht["source_id"], ht["tf"]))
    if missing_named:
        raise KeyError(f"named HT-like regulons not found: {sorted(missing_named)}")
    ht.to_csv(C.TABLES / "lit_ht_like_regulons.csv", index=False, float_format="%.4g")

    # lit_no_chipderived: whole held-in LIT sources removed (decision log item 9)
    srcs_fam = edges.groupby("source_id")["family"].first()
    bad = [x for x in LIT_NO_CHIPDERIVED_DROP if srcs_fam.get(x) != "LIT"]
    if bad:
        raise KeyError(f"lit_no_chipderived_drop names sources that are not held-in LIT sources: {bad}")

    def _variant_info(keep_lit: pd.DataFrame) -> dict:
        cond_v = edge_set(keep_lit[keep_lit["via_dorothea"]])
        lit_e, keep_e = edge_set(lit), edge_set(keep_lit)
        return {"n_lit_rows_removed": int(len(lit) - len(keep_lit)),
                "n_lit_edges_union": len(keep_e), "n_lit_edges_lost": len(lit_e - keep_e),
                "n_lit_tfs": int(keep_lit["tf"].nunique()), "n_lit_tfs_lost": len(set(lit["tf"]) - set(keep_lit["tf"])),
                "n_lit_targets": int(keep_lit["target"].nunique()),
                "n_lit_targets_lost": len(set(lit["target"]) - set(keep_lit["target"])),
                "n_conditioning_edges": len(cond_v), "n_conditioning_edges_lost": len(edge_set(dcur) - cond_v)}

    ht_key = pd.MultiIndex.from_frame(ht[["source_id", "tf"]])
    sens_info = {
        "lit_no_ht": {"description": LIT_HT_DESCRIPTION, "n_regulons": int(len(ht)),
                      "n_named": int(ht["reason"].str.startswith("named").sum()),
                      **_variant_info(lit[~pd.MultiIndex.from_frame(lit[["source_id", "tf"]]).isin(ht_key)])},
        "lit_no_chipderived": {"description": LIT_NO_CHIPDERIVED_DESCRIPTION, "dropped_sources": list(LIT_NO_CHIPDERIVED_DROP),
                               "n_edges_per_dropped_source": {x: int((lit["source_id"] == x).sum())
                                                              for x in LIT_NO_CHIPDERIVED_DROP},
                               **_variant_info(lit[~lit["source_id"].isin(LIT_NO_CHIPDERIVED_DROP)])},
    }
    prov["sensitivity_variants"] = sens_info
    merge_info = {}
    for name, members in MERGE_SETS.items():
        sets = {m: edge_set(edges[edges["source_id"] == m]) for m in members}
        u = set().union(*sets.values())
        merge_info[name] = {"members": members, "n_union": len(u),
                            "n_member": {m: len(s) for m, s in sets.items()},
                            "frac_member_in_other": {m: len(s & set().union(*(sets[o] for o in members if o != m))) / max(len(s), 1)
                                                     for m, s in sets.items()}}
    prov["merge_sets"] = merge_info

    # ---------------- per-source metadata, tables ----------------
    held_in = sorted(s for s in lineage if fam[s] != "CHIP")
    sources = {}
    rows = []
    for s, g in edges.groupby("source_id"):
        comps = sorted(kept.loc[kept["final_id"] == s, "source_id"].unique())
        chans = sorted(kept.loc[kept["final_id"] == s, "channel"].unique())
        if s == "LIT_other":
            meta = dict(full_name="Merged small literature resources (< %d edges each): %s" % (LIT_MIN_EDGES, ", ".join(comps)),
                        criterion="union of LIT resources with < %d cleaned edges" % LIT_MIN_EDGES,
                        citation="; ".join(SOURCE_META[c]["citation"] for c in comps),
                        doi="; ".join(SOURCE_META[c]["doi"] for c in comps),
                        resource_url="; ".join(f"{c}: {SOURCE_META[c]['resource_url']}" for c in comps),
                        version="; ".join(f"{c}: {SOURCE_META[c]['version']}" for c in comps),
                        notes="; ".join(
                            [f"{c}: {SOURCE_META[c]['notes']}" for c in comps if SOURCE_META[c]["notes"]] +
                            [f"TRRD dropped ({dup_info['TRRD']['n_duplicated_dropped']} of {dup_info['TRRD']['n_edges']} edges duplicated in TFactS)"]))
        else:
            meta = SOURCE_META[s]
        how = []
        if "DoRothEA" in chans:
            how.append(f"OmniPath DoRothEA A-D query (omnipath Python client; accessed {d_in['access_date']})")
        if "CollecTRI" in chans:
            how.append(f"OmniPath CollecTRI query (accessed {c_in['access_date']})")
        if "TRRUST-file" in chans:
            how.append("TRRUST v2 raw human file")
        if "OmniPath-tf_target" in chans:
            how.append("OmniPath tf_target evidences attached to DoRothEA/CollecTRI records (truncated)")
        obtained_url = []
        if "DoRothEA" in chans or "OmniPath-tf_target" in chans:
            obtained_url.append(d_in["url"])
        if "CollecTRI" in chans or "OmniPath-tf_target" in chans:
            obtained_url.append(c_in["url"])
        if "TRRUST-file" in chans:
            obtained_url.append(TRRUST_URL)
        version = meta["version"]
        if s == "TRRUST":
            t_in = man["inputs"]["trrust"]
            head = t_in.get("remote_head", {})
            version += (f"; raw file accessed {t_in['access_date']}, sha256 {t_in['sha256'][:12]}..."
                        + (f"; served file Last-Modified {head['last_modified']}, Content-Length {head['content_length']} "
                           f"= local size {t_in['bytes']} (HTTP HEAD {head['checked']})" if head.get("last_modified") else ""))
        held_out = fam[s] == "CHIP"
        sources[s] = dict(family=fam[s], lineage=lineage[s], genome_wide=fam[s] in GENOME_WIDE_FAMILIES,
                          held_out=held_out, components=comps, channels=chans,
                          complete=s not in TRUNCATED_CHIP,
                          use_in_chip_sensitivity=s in CHIP_SENSITIVITY_SOURCES)
        rows.append(dict(
            source_id=s, full_name=meta["full_name"], family=fam[s], lineage=lineage[s],
            how_obtained="; ".join(how), integrator_channels="; ".join(chans),
            version_or_access_date=version, obtained_url=" | ".join(obtained_url),
            resource_url=meta["resource_url"], primary_citation=meta["citation"], doi=meta["doi"],
            family_criterion=meta["criterion"],
            n_edges=len(g), n_tfs=g["tf"].nunique(), n_targets=g["target"].nunique(),
            n_nodes=len(set(g["tf"]) | set(g["target"])),
            held_out=held_out,
            role=("held out of every fit (incomplete; same ENCODE network as the ENETS2 validation reference)"
                  if s in TRUNCATED_CHIP else
                  "held out of the primary fit; ChIP-inclusive sensitivity fit only" if held_out else "annotator (primary fit)"),
            components=", ".join(comps), notes=meta["notes"]))
    st = pd.DataFrame(rows)
    order = {"LIT": 0, "MOTIF": 1, "COEXP": 2, "CHIP": 3}
    st = st.sort_values(by=["family", "n_edges"], key=lambda c: c.map(order) if c.name == "family" else -c).reset_index(drop=True)
    st.to_csv(C.TABLES / "source_table.csv", index=False)

    comp = pd.DataFrame(comp_rows)
    comp["final_source_id"] = comp["source_id"].map(lambda s: "LIT_other" if s in small_lit else s)
    for comp_src, info in dup_info.items():
        m = comp["source_id"] == comp_src
        comp.loc[m, "status"] = (f"used; {info['n_duplicated_dropped']} of {info['n_edges']} edges dropped as duplicates of "
                                 f"{info['container']}" if info["n_kept"] else
                                 f"dropped: all {info['n_edges']} edges duplicated in {info['container']}")
        if not info["n_kept"]:
            comp.loc[m, "final_source_id"] = ""
    comp.to_csv(C.TABLES / "source_components.csv", index=False)

    src_json = {"sources": sources, "held_in": held_in, "lit_other_components": small_lit,
                "genome_wide_families": list(GENOME_WIDE_FAMILIES),
                "chip_sensitivity_sources": list(CHIP_SENSITIVITY_SOURCES),
                "lit_min_edges": LIT_MIN_EDGES, "pazar_ht_min_targets": PAZAR_HT_MIN_TARGETS,
                "duplicate_drops": dup_info,
                "lit_ht_like_regulons": [[s, t] for s, t in zip(ht["source_id"], ht["tf"])],
                "lit_ht_max_targets": LIT_HT_MAX_TARGETS,
                "lit_ht_like_description": LIT_HT_DESCRIPTION,
                "lit_no_chipderived_drop": list(LIT_NO_CHIPDERIVED_DROP),
                "lit_no_chipderived_description": LIT_NO_CHIPDERIVED_DESCRIPTION,
                "merge_sets": MERGE_SETS,
                "conditioning_set": "edges with family == 'LIT' and via_dorothea (DoRothEA-channel held-in literature evidence)"}
    C.write_json(src_json, OUT_DIR / "sources.json")
    C.write_json(src_json, COMMITTED_SOURCES)
    C.write_json(man, COMMITTED_RAW_MANIFEST)

    # ---------------- integration baselines ----------------
    ib, ib_prov = integration_baselines(dor, col, ev_d, ev_c, cleaner, edges, pazar_big)
    prov["integration_baselines"] = ib_prov
    ib.to_parquet(OUT_DIR / "integration_baselines.parquet", index=False)
    ib.to_parquet(COMMITTED_BASELINES, index=False, compression="zstd")

    # ---------------- diagnostics recorded in provenance ----------------
    heldin_edges = edges[edges["family"] != "CHIP"]
    union_lit = edge_set(heldin_edges[heldin_edges["family"] == "LIT"])
    chip_pairs = edge_set(chip[chip["use_in_sensitivity"]])
    cond = edge_set(dcur)
    diag = {}
    for s in ("HOCOMOCO", "JASPAR", "ARACNe-GTEx"):
        es = edges[edges["source_id"] == s]
        e = edge_set(es)
        no_lit = e - union_lit
        tfs_all = set(es["tf"])
        tfs_cond = {tf for tf, tg in e & cond}
        diag[s] = {"n_edges": len(e), "n_without_heldin_LIT_support": len(no_lit),
                   "n_without_LIT_but_with_heldout_CHIP": len(no_lit & chip_pairs),
                   "n_covered_tfs": len(tfs_all), "n_covered_tfs_without_conditioning_set_report": len(tfs_all - tfs_cond),
                   "n_conditioning_cells_of_those_tfs": int(dcur["tf"].isin(tfs_all - tfs_cond).sum())}
    f = dor[list(DOROTHEA_FLAGS)].astype(bool)
    diag["dorothea_records_without_curated_or_chipseq"] = int((~f.dorothea_curated & ~f.dorothea_chipseq).sum())
    diag["dorothea_tfbs_records"] = int(f.dorothea_tfbs.sum())
    diag["dorothea_tfbs_records_chipseq_only"] = int((f.dorothea_tfbs & ~f.dorothea_curated & f.dorothea_chipseq).sum())
    diag["dorothea_coexp_records"] = int(f.dorothea_coexp.sum())
    diag["dorothea_coexp_records_chipseq_only"] = int((f.dorothea_coexp & ~f.dorothea_curated & f.dorothea_chipseq).sum())
    tred = long[(long["label"].isin(["TRED", "RegNetwork"])) & (long["channel"] == "DoRothEA")]
    diag["tred_equals_regnetwork"] = bool(edge_set(tred[tred["label"] == "TRED"]) == edge_set(tred[tred["label"] == "RegNetwork"]))
    prov["diagnostics"] = diag
    prov["lit_other_components"] = small_lit
    prov["lit_ht_like_regulons"] = ht.to_dict(orient="records")
    prov["n_edges_total"] = int(len(edges))
    prov["n_conditioning_edges"] = int(len(dcur))
    prov["n_heldin_edges_union"] = int(len(heldin_edges[["tf", "target"]].drop_duplicates()))
    prov["n_heldin_tfs"] = int(heldin_edges["tf"].nunique())
    prov["n_heldin_targets"] = int(heldin_edges["target"].nunique())
    litd = heldin_edges[heldin_edges["family"] == "LIT"]
    prov["n_heldin_lit_edges_union"] = int(len(litd[["tf", "target"]].drop_duplicates()))
    prov["n_heldin_lit_tfs"] = int(litd["tf"].nunique())
    prov["n_heldin_lit_targets"] = int(litd["target"].nunique())
    prov["committed_inputs"] = {p.name: {"sha256": sha256(p), "bytes": p.stat().st_size}
                                for p in (COMMITTED_EDGES, COMMITTED_BASELINES, COMMITTED_SOURCES, COMMITTED_RAW_MANIFEST)}
    C.write_json(prov, OUT_DIR / "provenance.json")
    print(st[["source_id", "family", "lineage", "n_edges", "n_tfs", "n_targets", "held_out"]].to_string(), flush=True)
    print(ht.to_string(), flush=True)
    print(json.dumps({k: prov[k] for k in ("duplicate_drops", "merge_sets", "diagnostics", "n_heldin_lit_tfs",
                                           "n_heldin_lit_targets", "n_conditioning_edges")}, indent=1), flush=True)
    print(json.dumps(ib_prov, indent=1, default=str), flush=True)


def _record_labels(ev: pd.DataFrame, dataset: str, via: str) -> pd.DataFrame:
    sub = ev[(ev["dataset"] == dataset) & (ev["via"] == via)]
    sub = sub[~sub["resource"].isin(INTEGRATOR_LABELS)]
    return sub[["rec", "resource"]].drop_duplicates()


def _edge_label_counts(lab: pd.DataFrame, rec_edge: pd.DataFrame, name: str) -> pd.Series:
    """Number of distinct labels per cleaned edge (union over the edge's records)."""
    m = lab.merge(rec_edge, on="rec")[["tf", "target", "resource"]].drop_duplicates()
    return m.groupby(["tf", "target"]).size().rename(name)


def integration_baselines(dor: pd.DataFrame, col: pd.DataFrame, ev_d: pd.DataFrame, ev_c: pd.DataFrame,
                          cleaner: Cleaner, edges: pd.DataFrame, pazar_big: set) -> tuple[pd.DataFrame, dict]:
    """Per-edge DoRothEA level/flags (released and held-in-only), CollecTRI support, ChIP/held-in flags."""
    prov: dict = {}
    # ---------------- DoRothEA ----------------
    d = dor[["source", "target", "source_genesymbol", "target_genesymbol", "dorothea_level", *DOROTHEA_FLAGS]].copy()
    d["rec"] = np.arange(len(d))
    d, cnt_d = cleaner.clean(d, "source_genesymbol", "target_genesymbol", "source", "target")
    d["level_rank"] = d["dorothea_level"].astype(str).map(
        lambda s: min(LEVEL_RANK.get(x.strip(), 9) for x in s.split(";")))
    rec_edge = d[["rec", "tf", "target"]]
    lab = _record_labels(ev_d, "dorothea", "DoRothEA")
    lab = lab[~lab["resource"].isin(DERIVED_LABELS)]
    cur = lab[~lab["resource"].isin(DOROTHEA_NONCURATED)]
    # released-level identity (validation of the rule)
    cur_full = cur.assign(resource=cur["resource"].map(lambda r: CURATED_COUNT_MAP.get(r, r)))
    # held-in identity: PAZAR records of the PAZAR-HT TFs are held-out ChIP-like evidence
    raw_tf = dor["source_genesymbol"].astype(str).values
    is_ht = (cur["resource"] == "PAZAR").values & np.isin(raw_tf[cur["rec"].values], list(pazar_big))
    cur_h = cur[~is_ht]
    cur_h = cur_h.assign(resource=cur_h["resource"].map(lambda r: CURATED_COUNT_MAP_HELDIN.get(r, r)))
    n_full = _edge_label_counts(cur_full, rec_edge, "dorothea_n_curated")
    n_h = _edge_label_counts(cur_h, rec_edge, "dorothea_n_curated_heldin")
    rev = lab[lab["resource"] == "DoRothEA-reviews"].merge(rec_edge, on="rec")[["tf", "target"]].drop_duplicates()
    g = d.groupby(["tf", "target"], sort=True)
    dd = g.agg(dorothea_level_rank=("level_rank", "min"),
               **{f: (f, "max") for f in DOROTHEA_FLAGS}).reset_index()
    dd = dd.merge(n_full.reset_index(), on=["tf", "target"], how="left").merge(n_h.reset_index(), on=["tf", "target"], how="left")
    for k in ("dorothea_n_curated", "dorothea_n_curated_heldin"):
        dd[k] = dd[k].fillna(0).astype(int)
    dd["dorothea_reviews"] = pd.MultiIndex.from_frame(dd[["tf", "target"]]).isin(pd.MultiIndex.from_frame(rev))
    inv = {v: k for k, v in LEVEL_RANK.items()}
    dd["dorothea_level"] = dd["dorothea_level_rank"].map(inv)
    dd["dorothea_level_rule"] = dorothea_level_rule(dd["dorothea_n_curated"], dd["dorothea_reviews"],
                                                    dd["dorothea_chipseq"], dd["dorothea_tfbs"], dd["dorothea_coexp"])
    dd["dorothea_curated_heldin"] = dd["dorothea_n_curated_heldin"] >= 1
    dd["dorothea_level_heldin"] = dorothea_level_rule(dd["dorothea_n_curated_heldin"], dd["dorothea_reviews"],
                                                      np.zeros(len(dd), bool), dd["dorothea_tfbs"], dd["dorothea_coexp"])
    dd.loc[~dd["dorothea_curated_heldin"], "dorothea_level_heldin"] = None
    dd["dorothea_level_heldin_rank"] = dd["dorothea_level_heldin"].map(LEVEL_RANK)
    agree = (dd["dorothea_level_rule"] == dd["dorothea_level"])
    prov["dorothea_clean"] = cnt_d
    prov["dorothea_rule_agreement_with_released"] = {
        "n_edges": int(len(dd)), "n_agree": int(agree.sum()), "frac_agree": float(agree.mean()),
        # edges whose evidences JSON lists a curated resource although no record has the curated flag
        "n_curated_label_without_curated_flag": int(((dd["dorothea_n_curated"] >= 1) & ~dd["dorothea_curated"].astype(bool)).sum()),
        "n_curated_flag_without_curated_label": int(((dd["dorothea_n_curated"] == 0) & dd["dorothea_curated"].astype(bool)).sum()),
        "confusion_released_x_rule": pd.crosstab(dd["dorothea_level"], dd["dorothea_level_rule"].fillna("none")).to_dict()}

    # ---------------- CollecTRI ----------------
    c = col[["source", "target", "source_genesymbol", "target_genesymbol", "curation_effort",
             "is_stimulation", "is_inhibition"]].reset_index(drop=True).copy()
    c["rec"] = np.arange(len(c))
    evc = ev_c[ev_c["dataset"] == "collectri"]
    refs = evc[(evc["resource"] == "CollecTRI") & (evc["via"] == "")].groupby("rec")["refs"].first()
    c["pmids"] = c["rec"].map(refs).fillna("")
    c, cnt_c = cleaner.clean(c, "source_genesymbol", "target_genesymbol", "source", "target")
    c["is_stimulation"] = _to_bool(c["is_stimulation"])
    c["is_inhibition"] = _to_bool(c["is_inhibition"])
    crec = c[["rec", "tf", "target"]]
    clab = _record_labels(ev_c, "collectri", "CollecTRI")
    n_res = _edge_label_counts(clab, crec, "collectri_n_resources")
    n_res_h = _edge_label_counts(clab[~clab["resource"].isin(DERIVED_LABELS)], crec, "collectri_n_resources_heldin")
    dora = clab[clab["resource"].isin(DERIVED_LABELS)].merge(crec, on="rec")[["tf", "target"]].drop_duplicates()
    gc = c.groupby(["tf", "target"], sort=True)
    cc = gc.agg(collectri_curation_effort=("curation_effort", "max"),
                collectri_is_stimulation=("is_stimulation", "max"),
                collectri_is_inhibition=("is_inhibition", "max")).reset_index()
    pm = gc["pmids"].agg(lambda x: len({p for s in x for p in s.split(";") if p})).rename("collectri_n_references")
    cc = cc.merge(pm.reset_index(), on=["tf", "target"])
    cc = cc.merge(n_res.reset_index(), on=["tf", "target"], how="left").merge(n_res_h.reset_index(), on=["tf", "target"], how="left")
    for k in ("collectri_n_resources", "collectri_n_resources_heldin"):
        cc[k] = cc[k].fillna(0).astype(int)
    cc["collectri_dorothea_a"] = pd.MultiIndex.from_frame(cc[["tf", "target"]]).isin(pd.MultiIndex.from_frame(dora))
    cc["in_collectri"] = True
    cc["in_collectri_heldin"] = cc["collectri_n_resources_heldin"] >= 1

    out = dd.merge(cc, on=["tf", "target"], how="outer")
    out["in_dorothea"] = out["dorothea_level"].notna()
    out["in_collectri"] = out["in_collectri"].astype("boolean").fillna(False).astype(bool)
    out["in_collectri_heldin"] = out["in_collectri_heldin"].astype("boolean").fillna(False).astype(bool)
    out["in_dorothea_heldin"] = out["dorothea_level_heldin"].notna()

    # ---------------- ChIP and held-in flags from the annotator edges ----------------
    key = pd.MultiIndex.from_frame(out[["tf", "target"]])

    def has(mask) -> np.ndarray:
        return key.isin(pd.MultiIndex.from_frame(edges.loc[mask, ["tf", "target"]].drop_duplicates()))

    # held-in support = a held-in LITERATURE annotator reports the edge (decision log item 9);
    # held-in MOTIF/COEXP reports without literature support are ChIP-conditioned (diagnostic flag)
    out["heldin_support_lit"] = has(edges["family"] == "LIT")
    out["motif_coexp_only_support"] = has(edges["family"].isin(["MOTIF", "COEXP"])) & ~out["heldin_support_lit"]
    out["remap"] = has(edges["source_id"] == "ReMap")
    out["pazar_ht"] = has(edges["source_id"] == "PAZAR-HT")
    out["encode_tag"] = has(edges["source_id"].isin(TRUNCATED_CHIP))

    # held-in DoRothEA level requires literature support (implied by held-in curated DoRothEA
    # evidence, so the count removed here is expected to be 0; recorded in provenance and tested)
    lvl_wo_lit = out["dorothea_level_heldin"].notna() & ~out["heldin_support_lit"]
    n_lvl_wo_lit = int(lvl_wo_lit.sum())
    out.loc[lvl_wo_lit, ["dorothea_level_heldin", "dorothea_level_heldin_rank"]] = None
    out["in_dorothea_heldin"] = out["dorothea_level_heldin"].notna()
    # released CollecTRI effort / references restricted to edges with held-in CollecTRI support
    hc = out["in_collectri_heldin"].to_numpy(bool)
    out["collectri_curation_effort_heldin"] = out["collectri_curation_effort"].where(hc)
    out["collectri_n_references_heldin"] = out["collectri_n_references"].where(hc)

    for k in ("dorothea_curated", "dorothea_chipseq", "dorothea_tfbs", "dorothea_coexp", "dorothea_reviews",
              "dorothea_curated_heldin", "collectri_is_stimulation", "collectri_is_inhibition", "collectri_dorothea_a"):
        out[k] = out[k].astype("boolean")
    for k in ("collectri_curation_effort", "collectri_n_resources", "collectri_n_resources_heldin",
              "collectri_n_references", "collectri_curation_effort_heldin", "collectri_n_references_heldin",
              "dorothea_level_rank", "dorothea_level_heldin_rank",
              "dorothea_n_curated", "dorothea_n_curated_heldin"):
        out[k] = out[k].astype("Int64")
    for k in ("dorothea_level", "dorothea_level_rule", "dorothea_level_heldin"):
        out[k] = out[k].astype(object).where(out[k].notna(), None)
    cols = ["tf", "target",
            "in_dorothea", "dorothea_level", "dorothea_level_rank", "dorothea_curated", "dorothea_chipseq",
            "dorothea_tfbs", "dorothea_coexp", "dorothea_n_curated", "dorothea_reviews", "dorothea_level_rule",
            "dorothea_n_curated_heldin", "dorothea_curated_heldin", "in_dorothea_heldin",
            "dorothea_level_heldin", "dorothea_level_heldin_rank",
            "in_collectri", "collectri_n_references", "collectri_n_resources", "collectri_dorothea_a",
            "collectri_n_resources_heldin", "in_collectri_heldin", "collectri_curation_effort",
            "collectri_curation_effort_heldin", "collectri_n_references_heldin",
            "collectri_is_stimulation", "collectri_is_inhibition",
            "heldin_support_lit", "motif_coexp_only_support", "remap", "pazar_ht", "encode_tag"]
    out = out[cols].sort_values(["tf", "target"]).reset_index(drop=True)
    prov.update({
        "collectri_clean": cnt_c, "n_edges": int(len(out)), "n_dorothea": int(out["in_dorothea"].sum()),
        "n_collectri": int(out["in_collectri"].sum()),
        "dorothea_level_counts": out["dorothea_level"].value_counts().sort_index().to_dict(),
        "dorothea_level_heldin_counts": out["dorothea_level_heldin"].value_counts().sort_index().to_dict(),
        "n_dorothea_heldin": int(out["in_dorothea_heldin"].sum()),
        "n_dorothea_without_heldin_support_lit": int((out["in_dorothea"] & ~out["heldin_support_lit"]).sum()),
        "n_dorothea_motif_coexp_only_support": int((out["in_dorothea"] & out["motif_coexp_only_support"]).sum()),
        "n_motif_coexp_only_support_with_heldout_chip": int((out["motif_coexp_only_support"]
                                                             & (out["remap"] | out["pazar_ht"])).sum()),
        "n_motif_coexp_only_support_dorothea_curated_flag": int((out["motif_coexp_only_support"]
                                                                 & out["dorothea_curated"].fillna(False).astype(bool)).sum()),
        "n_dorothea_without_heldin_curated": int((out["in_dorothea"] & ~out["in_dorothea_heldin"]).sum()),
        "n_level_heldin_without_lit_support": n_lvl_wo_lit,
        "released_x_heldin_level": pd.crosstab(out["dorothea_level"].fillna("none"),
                                               out["dorothea_level_heldin"].fillna("none")).to_dict(),
        "n_collectri_heldin": int(out["in_collectri_heldin"].sum()),
        "n_collectri_dorothea_a_only": int((out["in_collectri"] & ~out["in_collectri_heldin"]).sum()),
        "n_collectri_effort_heldin_missing": int((out["in_collectri"] & out["collectri_curation_effort_heldin"].isna()).sum()),
        "n_with_remap": int(out["remap"].sum()), "n_with_pazar_ht": int(out["pazar_ht"].sum()),
        "n_with_encode_tag": int(out["encode_tag"].sum()),
        "level_heldin_rule": ("A: reviews or >= 2 held-in curated resources; B: curated & TFBS & GTEx; C: curated & TFBS; "
                              "D: curated (incl. curated & GTEx); missing: no held-in curated DoRothEA evidence or no "
                              "held-in LIT support (heldin_support_lit)"),
        "support_flags": ("heldin_support_lit = a held-in LIT annotator reports the edge (the held-in support flag for "
                          "baselines); motif_coexp_only_support = held-in MOTIF/COEXP report without LIT support "
                          "(ChIP-conditioned, diagnostic only); the former 'heldin_support' (any held-in family) is removed"),
    })
    return out, prov


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--refresh", nargs="+", choices=["dorothea", "collectri", "trrust"],
                    help="re-download these raw inputs into new dated files in raw/ and use them")
    ap.add_argument("--download", action="store_true", help="deprecated alias of --refresh collectri")
    ap.add_argument("--freeze-legacy", action="store_true",
                    help="(one-time) copy the exact original inputs into raw/ and write raw/provenance.json")
    ap.add_argument("--export-portable", action="store_true",
                    help="write portable copies of the frozen inputs to data/r2/inputs/annotator_raw/ (after "
                         "--freeze-legacy / --refresh)")
    ap.add_argument("--from-portable", action="store_true",
                    help="build from the committed portable copies even if the frozen raw files exist")
    args = ap.parse_args()
    if args.freeze_legacy:
        freeze_legacy()
    which = list(args.refresh or []) + (["collectri"] if args.download else [])
    if which:
        refresh(sorted(set(which)))
    if args.export_portable:
        export_portable()
    build(from_portable=args.from_portable)


if __name__ == "__main__":
    main()
