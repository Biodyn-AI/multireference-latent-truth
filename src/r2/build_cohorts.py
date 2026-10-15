"""Held-out perturbation validation cohorts with per-perturbation differential expression (stage D4).

The cohorts built here are used ONLY for held-out evaluation. Nothing in this
module writes any universe, annotator or method score, and no other stage may
use these outputs for fitting, tuning or universe construction. The tissue and
global universes are read only for the descriptive coverage table
(``cohort_universe_overlap.csv``) and for the descriptive gene-mapping report.

Cohorts
-------
====================  ==========================================================
``dixit_d13``         Dixit et al. 2016, K562 TF Perturb-seq, day 13 (CRISPR KO)
``dixit_d7``          Dixit et al. 2016, K562 TF Perturb-seq, day 7 (CRISPR KO);
                      same study, guides and TFs as ``dixit_d13`` (study_id
                      ``dixit2016``; one independent unit)
``adamson``           Adamson et al. 2016, K562 UPR Perturb-seq screen (CRISPRi)
``shifrut``           Shifrut et al. 2018, primary human T cells (CRISPR KO)
``norman``            Norman et al. 2019, K562 CRISPRa, single-gene perturbations
``papalexi``          Papalexi et al. 2021, THP-1 ECCITE-seq (CRISPR KO)
``replogle``          Replogle et al. 2022, K562 essential Perturb-seq (CRISPRi),
                      pseudobulk files only
====================  ==========================================================

Gene identity (decision log items 7 and 9; ``PRESPECIFIED['symbol_mapping']``)
-----------------------------------------------------------------------------
Rule ``approved_symbol_first_then_ensembl``: matrix columns and perturbation
labels are mapped to the canonical symbol space of the whole pipeline
(``common.normalize_symbol``, i.e. approved symbols of ``HGNC_TABLE``).
Priority per column: (1) the column name itself if it is an approved symbol
(source ``symbol_ensembl_agrees`` / ``symbol``, or ``symbol_ensembl_differs`` when the column's Ensembl id
now maps to another gene, e.g. Replogle's second HSPA14 column
ENSG00000284024 = MSANTD7, Dixit/Adamson ZNF670 whose id now maps to the
readthrough ZNF670-ZNF695); (2) for a non-approved name, the Ensembl-derived
symbol (``common.ensembl_to_symbol``) if it is approved (``ensembl``; e.g.
Norman's RP5-862P8.2 = MAP3K21). ``HGNC_TABLE`` mixes releases and still lists
some renamed or withdrawn symbols as approved; a name that is no longer an
approved symbol in ``HGNC_FULL`` is therefore not treated as approved and its
Ensembl id decides (``ensembl_outdated_name``; e.g. Dixit/Adamson CXorf22 ->
CFAP47, Norman/Shifrut TMEM155 -> SMIM43); (3) the HGNC-normalised name (previous/alias
symbol; ``symbol_normalized``); (4)/(5) unapproved Ensembl-derived symbol /
normalised name. ``ensembl_to_symbol`` reads a newer HGNC release
(``HGNC_FULL``) than ``normalize_symbol``; an Ensembl-derived symbol that is not
approved in the older release (e.g. CDC50A, SACK1B) is mapped back through its
HGNC previous symbols (TMEM30A, FAM83B). When several columns map to one
symbol the column with the better source (``SOURCE_RANK``; e.g. of Replogle's two
TBCE columns the one whose Ensembl id HGNC assigns to TBCE) and then the larger
total count is kept. Perturbation labels follow the same rule: an approved label is kept; a
non-approved label takes the symbol of the matrix column with the same name
(the Ensembl route; e.g. Adamson 'QARS' -> QARS1) or, for Replogle, of the
perturbation row's Ensembl id; otherwise its HGNC-normalised form.
Gene-id sources: Dixit ``var['gene_id']``, Adamson ``var['ensembl_id']``,
Norman ``var['ensemble_id']``, Papalexi ``var['ensembl_id']`` (Ensembl for
14,246 of 18,649 columns, symbols otherwise), Replogle ``var`` index, Shifrut
none in the file: its 33,694 var names are identical, in order, to Norman's
(same 10x GRCh38-1.2.0 reference), so Norman's ids are used after an
element-wise identity check. The perturbed gene's own column(s) are excluded
from the target family and used for the on-target check: columns whose
symbol is the mapped label, the canonical form of any raw label, or (Replogle)
the Ensembl-derived symbol of the perturbation row's id, and columns carrying
an HGNC Ensembl id of the perturbed gene or the row's own id (fixes the Norman
MAP3K21 / RP5-862P8.2 and Replogle HSPA14 / MSANTD7 self-edges).
``self_pair_check`` tests targets against all of these labels.

Differential expression (single-cell cohorts; prespecified in ``common.PRESPECIFIED``)
-------------------------------------------------------------------------------------
1. Raw UMI counts are detected by inspection (all stored values integer). They
   are library-size normalised over all genes to 10,000 counts per cell (CP10k)
   and log1p transformed (scanpy ``normalize_total(1e4)`` + ``log1p``). A matrix
   already stored as log1p(CP10k) (Adamson/GEARS) is detected by verifying that
   expm1(X) divided by its smallest non-zero value per cell is integer for every
   entry, i.e. X = log1p(1e4 * count / library_size); it is then used as is.
2. Columns are mapped to canonical symbols as described above.
3. Guides are collapsed to the perturbed gene. Control cells carry
   non-targeting / intergenic guides (cohort-specific labels below).
4. For every perturbed gene with >= ``de_min_cells`` cells and every gene
   expressed (count > 0) in >= ``de_min_target_frac_expressed`` of the cells of
   either group: Welch two-sample t-test (perturbed vs. control) on log1p(CP10k)
   expression, two-sided p-value from the Welch-Satterthwaite t distribution
   ("pooled" test).
5. Multi-replicate designs (``PRESPECIFIED['replicate_stratified_de']``:
   Shifrut and Papalexi; decision log items 7 and 9): the primary test is the
   replicate-stratified two-sample t-test (``de_stratified_one``) over the
   design's strata (Shifrut: donor x stimulation; Papalexi: the four
   IFN-gamma-treated HTO replicates), because the pooled test is confounded by
   the stratum composition of the perturbed vs. control cells (Shifrut CD3D
   cells 48% stimulated vs. 56% of controls; Papalexi SPI1 total-variation
   distance 0.17). The pooled Welch test is the sensitivity analysis
   (``de_pooled.parquet``). A tested TF perturbation whose non-cis positive
   set under the two tests has Jaccard < ``NONROBUST_JACCARD`` (0.3) is flagged
   ``de_nonrobust_tf`` (e.g. Papalexi SPI1).
6. log2FC = log2((mean CP10k in perturbed + c) / (mean CP10k in control + c)),
   with pseudocount c = ``LOG2FC_PSEUDOCOUNT_CP10K`` (0.1 CP10k), fixed here
   (stratified test: control means post-stratified to the perturbed cells'
   stratum composition).
7. Benjamini-Hochberg FDR within each perturbation, over the genes tested for
   that perturbation (the perturbed gene's own columns are excluded).
8. Positive edge before cis exclusion (``positive_incl_cis``; not for
   evaluation) iff padj < ``de_fdr`` and |log2FC| > ``de_min_abs_log2fc``.
9. ``cis_same_band``: target and perturbed gene share an HGNC cytogenetic band
   at the finest resolution both are annotated to (``parse_location``): bands
   are compared as intervals on the chromosome arm, so ranges
   ('2q32.2-q32.3' contains '2q32.2') and coarser annotations ('19q13.1'
   contains '19q13.11') match, while distinct sub-bands do not ('17q21.32' vs.
   '17q21.31'). Bands of canonical symbols missing from ``HGNC_FULL`` are looked
   up through a unique HGNC previous symbol (e.g. CTAGE5 -> MIA2, 14q13.2).
   Locations that are not a chromosome band ('mitochondria', 'reserved',
   chromosome-only '15') never match. ``positive_primary = positive_incl_cis &
   ~cis_same_band``. Downstream evaluation uses ``positive_primary`` and
   excludes cis_same_band pairs from both positives and negatives
   (``PRESPECIFIED['cis_exclusion']``). BH families include cis pairs.
10. On-target check: the same statistics (primary test) for the perturbed gene
   itself (regardless of the expression filter); expected direction is down for
   CRISPR KO / CRISPRi and up for CRISPRa. "On-target pass" = change in the
   expected direction with p < 0.05 and |log2FC| > ``de_min_abs_log2fc``.
   For Cas9 knockout the mRNA readout is not a measure of knockout efficacy
   (frameshift alleles can be stable transcripts), so a failed check is not
   evidence of a failed knockout.

Replogle 2022 (pseudobulk only; see ``build_replogle``)
------------------------------------------------------
The per-cell data are 10.7 GB; the figshare pseudobulk files are used instead
and are downloaded with md5 verification if missing (``ensure_replogle_files``;
a ``download_log.json`` next to the files records the download time, otherwise
the file modification time is reported).
``K562_essential_normalized_bulk_01.h5ad`` holds, per perturbation row, the mean
over cells of gene expression z-normalised against control cells of the same
GEM group; ``num_cells_filtered`` gives the number of cells. The test statistic
is the approximate z-test Z = mean_z * sqrt(n_cells), two-sided normal p,
BH within perturbation. This assumes that, under the null, per-cell z-scores of
perturbed cells have mean 0 and unit variance (the control mean and SD are
estimated from ~10^4 cells and treated as known) and that cells are
independent; it is anti-conservative if a perturbation inflates cell-to-cell
variance. log2FC is computed from ``K562_essential_raw_bulk_01.h5ad`` (mean raw
UMI per cell per perturbation): pseudobulk profiles are scaled to 10,000
counts and compared with the pooled core non-targeting control profile, with
the same pseudocount. Rows for the same gene (alternative promoters/
transcripts; each row pools several guides) are pooled with cell-count
weights; ``n_pseudobulk_rows`` counts these rows (the number of guides is not
in the files, so ``n_guides`` is missing). The expression filter cannot be
applied per cell; the authors' expressed-gene set (8,563 genes) is used. Their
``var['mean']`` is not the control mean; the lowest mean raw UMI per pooled
core-control cell among these genes is recorded in ``meta.json`` (about 0.05,
i.e. ~5% expected detection under Poisson sampling).

Outputs (``<out-root>/<cohort>/``, default ``outputs/r2/cohorts``)
-----------------------------------------------------------------
* ``de.parquet`` (primary test): perturbed_gene, target, log2fc, t, pval, padj,
  positive_incl_cis, cis_same_band, positive_primary, n_pert_cells,
  n_ctrl_cells, mean_pert, mean_ctrl (CP10k), frac_pert, frac_ctrl, is_tf
  (perturbed gene is a Lambert TF). One row per tested (perturbation, target)
  pair. Evaluate with ``positive_primary`` and drop ``cis_same_band`` rows.
* ``de_pooled.parquet`` (Shifrut, Papalexi): the pooled-test sensitivity
  analysis, same schema and the same tested pairs.
* ``perturbations.csv``: one row per labelled perturbed gene (tested or not),
  with cell counts, guides, positives (primary test), cis removals, the
  sensitivity test's positives, the pooled-vs-stratified Jaccard and
  ``de_nonrobust_tf`` flag, and the on-target check.
* ``meta.json``: inputs (path, size, md5; HGNC files with md5/sha256 and
  release proxy), labels, gene mapping, filters, counts, parameters, stratum
  composition, self-checks.
* ``universe_overlap.json``: provenance of the overlap table.
* ``<tables-dir>/cohort_table.csv``, ``cohort_perturbations.csv``,
  ``cohort_universe_overlap.csv`` (default ``data/r2/tables``; with a
  non-default ``--out-root`` the default is ``<out-root>/tables``).

Usage
-----
    python src/r2/build_cohorts.py                       # all cohorts + tables
    python src/r2/build_cohorts.py --cohorts norman shifrut
    python src/r2/build_cohorts.py --table-only          # tables + overlap from existing outputs
    python src/r2/build_cohorts.py --out-root /tmp/x     # scratch rebuild (tables in /tmp/x/tables)
    python src/r2/build_cohorts.py --no-download         # never download Replogle files
"""
from __future__ import annotations

import argparse
import datetime as dt
import gc
import hashlib
import json
import re
import resource
import shutil
import sys
import time
import urllib.request
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from r2 import common as C  # noqa: E402

P = C.PRESPECIFIED
LOG2FC_PSEUDOCOUNT_CP10K = 0.1   # pseudocount (CP10k units) for log2FC; fixed a priori
TARGET_SUM = 1e4                 # normalize_total target
ONTARGET_P = 0.05                # nominal p for the on-target check (single test per perturbation)
STRATUM_MIN_CELLS = 3            # min cells per group within a stratum (stratified test)
OUT_ROOT = C.OUT / "cohorts"
UNIVERSE_DIR = C.OUT / "universe"                 # tissue panels (build_panels.py)
GLOBAL_UNIVERSE_DIR = C.OUT / "evidence" / "global"   # global universe (evidence.py)
CONTROL = "__CONTROL__"

# stratified test (strata column description); cohorts listed here get both tests
STRATIFIED = {
    "shifrut": "obs['sample'] (donor x stimulation: D1/D2 x stim/nostim)",
    "papalexi": "obs['hto'] (replicate: rep1-tx..rep4-tx)",
}
# primary test per cohort (default 'pooled'); fixed in common.PRESPECIFIED
# ('replicate_stratified_de': multi-replicate designs use the replicate-stratified
# test as primary; 'shifrut_de' is the earlier, consistent Shifrut-only entry)
PRIMARY_DE = {cid: "stratified" for cid in P["replicate_stratified_de"]}
assert set(PRIMARY_DE) <= set(STRATIFIED), "replicate_stratified_de cohort without strata"
assert PRIMARY_DE.get("shifrut") == P["shifrut_de"], "PRESPECIFIED shifrut_de / replicate_stratified_de disagree"
PRIMARY_DE_RATIONALE = {
    "shifrut": "decision log items 7 and 9 / PRESPECIFIED['replicate_stratified_de']: pooled test "
               "confounded by donor x stimulation composition (stimulation is a strong biological "
               "contrast); stratified test is primary, pooled kept as sensitivity (de_pooled.parquet)",
    "papalexi": "decision log item 9 / PRESPECIFIED['replicate_stratified_de']: multi-replicate design "
                "(4 IFN-gamma-treated HTO replicates); the pooled test is confounded by replicate "
                "composition (SPI1 TVD 0.17; pooled-vs-stratified Jaccard 0.06 for SPI1 although 0.83 "
                "overall); stratified test is primary, pooled kept as sensitivity (de_pooled.parquet)",
}
# pooled-vs-stratified robustness flag for TF perturbations (decision round 2,
# fixed before any held-out evaluation): Jaccard of the non-cis positive sets
NONROBUST_JACCARD = 0.3
# DE design per cohort (manuscript cohort table, column 'de_design')
DE_DESIGN = {
    "dixit_d13": "single condition; pooled Welch t-test vs. intergenic-guide controls",
    "dixit_d7": "single condition; pooled Welch t-test vs. intergenic-guide controls",
    "adamson": "single condition; pooled Welch t-test vs. non-targeting controls",
    "shifrut": "multi-replicate (2 donors x stimulated/unstimulated, 4 strata): replicate-stratified "
               "t-test primary; pooled Welch t-test as sensitivity",
    "norman": "single condition (GEM groups not modelled); pooled Welch t-test vs. non-targeting controls",
    "papalexi": "multi-replicate (4 IFN-gamma-treated HTO replicates): replicate-stratified t-test "
                "primary; pooled Welch t-test as sensitivity",
    "replogle": "pseudobulk z-scores (vs. same-GEM-group controls); approximate z-test, no replicate "
                "structure available",
}
# short design notes carried into the manuscript cohort table
DESIGN_NOTES = {
    "dixit_d13": "Cas9 knockout; guides collapsed to gene; intergenic-guide controls; same study, "
                 "guides and TFs as dixit_d7",
    "dixit_d7": "Cas9 knockout; guides collapsed to gene; intergenic-guide controls; same study, "
                "guides and TFs as dixit_d13",
    "adamson": "UPR screen; GEARS gene subset (4,888 genes); labels re-derived from GEO",
    "shifrut": "2 donors x stimulated/unstimulated; primary DE stratified by donor x stimulation; "
               "pooled DE reported as sensitivity",
    "norman": "single-gene CRISPRa only; confident single guide identity",
    "papalexi": "IFN-gamma-stimulated replicates; primary DE stratified by replicate; no escaper "
                "removal; pooled DE reported as sensitivity",
    "replogle": "pseudobulk only; approximate z-test (mean z x sqrt(n)); promoter variants pooled",
}
ONTARGET_NOTES = {
    "CRISPR KO": "Cas9 knockout acts on DNA; mRNA reduction (e.g. via nonsense-mediated decay) is an "
                 "incomplete readout of knockout efficacy, so a failed on-target mRNA check is not "
                 "evidence of a failed knockout",
    "CRISPRi": "CRISPRi represses transcription; on-target mRNA knockdown is the expected readout",
    "CRISPRa": "CRISPRa activates transcription; on-target mRNA induction is the expected readout",
}

REPLOGLE_FILES = {
    "normalized_bulk": {
        "name": "K562_essential_normalized_bulk_01.h5ad",
        "url": "https://ndownloader.figshare.com/files/35780870",
        "figshare_file_id": 35780870,
        "md5": "30496767641cd2e660ee6ecb5baee132",
    },
    "raw_bulk": {
        "name": "K562_essential_raw_bulk_01.h5ad",
        "url": "https://ndownloader.figshare.com/files/35773070",
        "figshare_file_id": 35773070,
        "md5": "8321d5d3ffc99db2a5c71edca4189735",
    },
}
REPLOGLE_DOWNLOAD_LOG = "download_log.json"


@dataclass
class CohortSpec:
    cohort_id: str
    full_name: str
    accession: str
    cell_type: str
    modality: str          # "CRISPR KO" | "CRISPRi" | "CRISPRa"
    path: Path
    study_id: str          # dependence unit (PRESPECIFIED['cohort_cluster_unit'])
    cell_type_group: str   # "K562" | "primary T cell" | "THP-1 monocyte"
    flag: str = ""         # prespecified caveat key ('' = none)
    notes: list = field(default_factory=list)

    @property
    def expected_direction(self) -> str:
        return "up" if self.modality == "CRISPRa" else "down"


PD = C.PERTURB_DIR
COHORTS = {
    "dixit_d13": CohortSpec(
        "dixit_d13", "Dixit et al. 2016 (K562 TFs, day 13)",
        "GEO GSE90063 (scPerturb h5ad)", "K562 (CML cell line)", "CRISPR KO",
        PD / "dixit" / "DixitRegev2016_K562_TFs_13_days.h5ad", "dixit2016", "K562", flag="dixit"),
    "dixit_d7": CohortSpec(
        "dixit_d7", "Dixit et al. 2016 (K562 TFs, day 7)",
        "GEO GSE90063 (scPerturb h5ad)", "K562 (CML cell line)", "CRISPR KO",
        PD / "dixit_7_days" / "DixitRegev2016_K562_TFs_7_days.h5ad", "dixit2016", "K562", flag="dixit"),
    "adamson": CohortSpec(
        "adamson", "Adamson et al. 2016 (K562 UPR screen)",
        "GEO GSE90546, GSM2406681 (GEARS h5ad, labels re-derived from GEO cell identities)",
        "K562 (CML cell line)", "CRISPRi",
        PD / "adamson" / "perturb_processed_corrected_upr_symbols.h5ad", "adamson2016", "K562"),
    "shifrut": CohortSpec(
        "shifrut", "Shifrut et al. 2018 (primary human T cells)",
        "GEO GSE119450 (scPerturb h5ad)",
        "Primary human T cells (2 donors; TCR-stimulated and unstimulated)", "CRISPR KO",
        PD / "shifrut" / "ShifrutMarson2018.h5ad", "shifrut2018", "primary T cell"),
    "norman": CohortSpec(
        "norman", "Norman et al. 2019 (K562 CRISPRa, single genes)",
        "GEO GSE133344 (scPerturb h5ad)", "K562 (CML cell line)", "CRISPRa",
        PD / "norman" / "NormanWeissman2019_filtered.h5ad", "norman2019", "K562"),
    "papalexi": CohortSpec(
        "papalexi", "Papalexi et al. 2021 (THP-1 ECCITE-seq)",
        "GEO GSE153056 (scPerturb h5ad)", "THP-1 (monocytic leukemia, IFN-gamma stimulated)",
        "CRISPR KO",
        PD / "papalexi" / "PapalexiSatija2021_eccite_RNA.h5ad", "papalexi2021", "THP-1 monocyte"),
    "replogle": CohortSpec(
        "replogle", "Replogle et al. 2022 (K562 essential, pseudobulk)",
        "figshare plus 20029387 (doi:10.25452/figshare.plus.20029387)",
        "K562 (CML cell line)", "CRISPRi",
        C.REPLOGLE_DIR / REPLOGLE_FILES["normalized_bulk"]["name"], "replogle2022", "K562"),
}


# ----------------------------------------------------------------------------
# Generic helpers
# ----------------------------------------------------------------------------

def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def file_md5(path: Path, chunk: int = 1 << 24) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def file_record(path: Path) -> dict:
    return {"path": str(path), "size_bytes": path.stat().st_size, "md5": file_md5(path)}


def peak_rss_gb() -> float:
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / 1e9 if sys.platform == "darwin" else r / 1e6


def bh(p: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values (monotone, capped at 1)."""
    p = np.asarray(p, dtype=float)
    n = p.size
    if n == 0:
        return p
    order = np.argsort(p, kind="mergesort")
    ranked = p[order] * n / np.arange(1, n + 1)
    adj = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.minimum(adj, 1.0)
    return out


def dedup_symbols(raw_symbols, totals) -> tuple[np.ndarray, np.ndarray]:
    """Legacy name-only mapping (R2 first build): HGNC-normalise names, keep the
    largest-total column per symbol. Used only to report what the Ensembl-based
    mapping changes.
    """
    sym = C.normalize_series(pd.Series(list(raw_symbols))).to_numpy()
    df = pd.DataFrame({"sym": sym, "tot": np.asarray(totals, float), "col": np.arange(len(sym))})
    df = df[df["sym"].str.len() > 0]
    df = df.sort_values(["sym", "tot", "col"], ascending=[True, False, True], kind="mergesort")
    df = df.drop_duplicates("sym", keep="first").sort_values("col")
    return df["col"].to_numpy(), df["sym"].to_numpy()


def clean_ensembl(values) -> np.ndarray:
    """Ensembl gene ids without version; '' where the value is not an ENSG id."""
    s = pd.Series(list(values), dtype=object).astype(str).str.strip().str.split(".").str[0]
    return np.where(s.str.match(r"^ENSG\d+$"), s, "").astype(object)


# ----------------------------------------------------------------------------
# Gene identity: approved-symbol-first mapping into the canonical symbol space,
# cytoband intervals
# ----------------------------------------------------------------------------

assert P["symbol_mapping"] == "approved_symbol_first_then_ensembl", P["symbol_mapping"]
# approved name whose Ensembl id HGNC confirms > approved name without HGNC-known id >
# approved name whose id HGNC now assigns elsewhere > Ensembl-derived symbol > ...
SOURCE_RANK = {"symbol_ensembl_agrees": 0, "symbol": 1, "symbol_ensembl_differs": 2, "ensembl": 3,
               "ensembl_outdated_name": 3, "symbol_normalized": 4, "ensembl_unapproved": 5,
               "symbol_unapproved": 6}
# R2 phase-1b Ensembl-first rule; used only to report what the prespecified rule changes
LEGACY_SOURCE_RANK = {"ensembl": 0, "symbol": 1, "ensembl_unapproved": 2, "symbol_unapproved": 3}

_LOC_SUFFIX_RE = re.compile(r"\s+(?:not on reference assembly|alternate reference locus|"
                            r"not on current reference|GRCh\d+ patch|unplaced)$")
_LOC_END_RE = re.compile(r"^(\d{1,2}|X|Y)?([pq])(ter|\d+(?:\.\d+)?)$")
BAND_DIGITS = 6


def _band_bounds(band: str) -> tuple:
    """Interval of a band's digit string (each position one digit) at BAND_DIGITS resolution."""
    if band == "ter":
        return 10 ** BAND_DIGITS - 1, 10 ** BAND_DIGITS - 1
    d = band.replace(".", "")
    if len(d) > BAND_DIGITS:
        raise ValueError(f"band finer than {BAND_DIGITS} digits: {band}")
    return int(d.ljust(BAND_DIGITS, "0")), int(d.ljust(BAND_DIGITS, "9"))


@lru_cache(maxsize=None)
def parse_location(loc: str) -> tuple:
    """HGNC location string -> tuple of (chromosome arm, lo, hi) band intervals.

    A band's digits (region, band, sub-bands; one digit each, e.g. q21.32 ->
    2132) define the interval of all finer bands it contains, so '19q13.1'
    contains '19q13.11' while '17q21.32' and '17q21.31' are disjoint. A range
    ('2q32.2-q32.3', '1q41-qter') is the union interval on its arm; a range
    across the centromere ('1pter-q21') gives one interval per arm. 'Xp22.33
    and Yp11.2' / '... or ...' give several intervals. Annotation suffixes
    ('alternate reference locus', 'not on reference assembly', ...) are
    ignored; chromosome-only ('15'), 'mitochondria', 'reserved', 'unplaced' and
    centromere-only locations give no interval (never cis).
    """
    s = _LOC_SUFFIX_RE.sub("", str(loc or "").strip())
    out = []
    for part in re.split(r"\s+(?:and|or)\s+", s):
        ends = part.strip().split("-")
        if len(ends) > 2:
            continue
        m = [_LOC_END_RE.match(e.strip()) for e in ends]
        if not all(m) or not m[0].group(1):
            continue
        chrom = m[0].group(1)
        if len(m) == 2 and m[1].group(1) not in (None, chrom):
            continue
        b = [(x.group(2),) + _band_bounds(x.group(3)) for x in m]
        if len({a for a, _, _ in b}) == 1:
            out.append((chrom + b[0][0], min(lo for _, lo, _ in b), max(hi for _, _, hi in b)))
        else:                       # across the centromere: each endpoint to the centromere
            out.extend((chrom + a, 0, hi) for a, _, hi in b)
    return tuple(out)


def intervals_overlap(a: tuple, b: tuple) -> bool:
    return any(ka == kb and la <= hb and lb <= ha for ka, la, ha in a for kb, lb, hb in b)


class GeneMapper:
    """Map (name, Ensembl id) pairs to the canonical symbols of ``common.normalize_symbol``.

    Rule ``PRESPECIFIED['symbol_mapping'] == 'approved_symbol_first_then_ensembl'``
    (see the module docstring). ``common.ensembl_to_symbol`` / ``symbol_cytoband``
    come from a newer HGNC release (``HGNC_FULL``) than the alias table behind
    ``normalize_symbol`` (``HGNC_TABLE``); ``new2canon`` maps every newer
    approved symbol into the canonical space (directly, or through a unique HGNC
    previous symbol).
    """

    def __init__(self):
        hg = pd.read_csv(C.HGNC_FULL, sep="\t", dtype=str,
                         usecols=["symbol", "prev_symbol", "ensembl_gene_id", "location"]).fillna("")
        self.approved = frozenset(C._alias_map().values())
        # symbols approved in the newer release; HGNC_TABLE mixes releases and still
        # lists some renamed/withdrawn symbols as approved (CXorf22 = CFAP47, TMEM155)
        self.current = frozenset(hg["symbol"].str.upper())
        self.e2s = C.ensembl_to_symbol()
        self.cyto = C.symbol_cytoband()
        new2canon = {}
        for sym, prev in zip(hg["symbol"].str.upper(), hg["prev_symbol"]):
            s = C.normalize_symbol(sym)
            if s not in self.approved:
                cands = {C.normalize_symbol(x) for x in prev.split("|") if x.strip()} & self.approved
                if len(cands) == 1:
                    s = next(iter(cands))
            new2canon[sym] = s
        self.new2canon = new2canon
        inv: dict[str, set] = {}
        for new, can in new2canon.items():
            inv.setdefault(can, set()).add(new)
        # canonical -> newer symbol(s); used for cytoband and Ensembl lookups
        self.canon2new = {c: (c if c in v else sorted(v)[0]) for c, v in inv.items() if len(v) == 1 or c in v}
        sym2ens: dict[str, set] = {}
        for sym, ens in zip(hg["symbol"].str.upper(), hg["ensembl_gene_id"].str.split(".").str[0]):
            if ens:
                sym2ens.setdefault(sym, set()).add(ens)
        self.sym2ens = sym2ens
        # previous symbol (newer release) -> location, where unambiguous; band
        # fallback for canonical symbols withdrawn/merged in HGNC_FULL (CTAGE5 -> MIA2)
        prev_loc: dict[str, set] = {}
        for loc, prev in zip(hg["location"], hg["prev_symbol"]):
            for x in prev.split("|"):
                x = x.strip().upper()
                if x and loc:
                    prev_loc.setdefault(x, set()).add(loc)
        self.prev_loc = {k: next(iter(v)) for k, v in prev_loc.items() if len(v) == 1}

    # -- symbols ------------------------------------------------------------
    def canon(self, sym) -> str:
        """Canonical symbol for a label (perturbation names, column names)."""
        s = C.normalize_symbol(sym)
        if s and s not in self.approved and s in self.new2canon:
            s = self.new2canon[s]
        return s

    def from_ensembl(self, ens: str) -> str:
        new = self.e2s.get(ens) if ens else None
        return self.new2canon.get(new, C.normalize_symbol(new)) if new else ""

    def is_approved_name(self, u: str) -> bool:
        """Approved in the canonical table and still approved in the newer HGNC release."""
        return u in self.approved and u in self.current

    def resolve(self, name, ens) -> tuple:
        """(canonical symbol, source) for one column/perturbation (approved name first).

        A name that is approved in the canonical table but no longer approved in
        HGNC_FULL (renamed or withdrawn, e.g. CXorf22 -> CFAP47, TMEM155 ->
        SMIM43) is not an approved HGNC symbol: its Ensembl id decides if it maps
        to an approved symbol (source ``ensembl_outdated_name``).
        """
        u = "" if name is None else str(name).strip().upper()
        se = self.from_ensembl(ens)
        if u and u in self.approved:
            if not se:
                return u, "symbol"
            if se == u:
                return u, "symbol_ensembl_agrees"
            if u in self.current or se not in self.approved:
                return u, "symbol_ensembl_differs"
            return se, "ensembl_outdated_name"
        if se and se in self.approved:
            return se, "ensembl"
        sn = self.canon(name)
        if sn and sn in self.approved:
            return sn, "symbol_normalized"
        if se:
            return se, "ensembl_unapproved"
        return sn, "symbol_unapproved"

    def resolve_ensembl_first(self, name, ens) -> tuple:
        """Legacy R2 phase-1b rule (Ensembl id first); reporting only."""
        se = self.from_ensembl(ens)
        sn = self.canon(name)
        if se and se in self.approved:
            return se, "ensembl"
        if sn and sn in self.approved:
            return sn, "symbol"
        if se:
            return se, "ensembl_unapproved"
        return sn, "symbol_unapproved"

    def ensembl_of(self, canon_sym: str) -> set:
        """HGNC Ensembl ids of a canonical symbol (via the newer HGNC release); empty if
        the symbol has no unique counterpart there."""
        new = self.canon2new.get(canon_sym)
        return set(self.sym2ens.get(new, set())) if new else set()

    # -- columns ------------------------------------------------------------
    def map_columns(self, names, ens_ids, totals, legacy: bool = False) -> dict:
        """Map matrix columns; keep one column per canonical symbol.

        Returns dict with keep_cols, genes (canonical symbols of kept columns),
        gene_ens (Ensembl id of each kept column, '' if none), all_syms/all_src
        (per original column) and source counts. ``legacy=True`` applies the
        phase-1b Ensembl-first rule (reporting only).
        """
        names = list(names)
        ens_ids = clean_ensembl(ens_ids) if ens_ids is not None else np.array([""] * len(names), object)
        resolve, rank = ((self.resolve_ensembl_first, LEGACY_SOURCE_RANK) if legacy
                         else (self.resolve, SOURCE_RANK))
        res = [resolve(n, e) for n, e in zip(names, ens_ids)]
        sym = np.array([r[0] for r in res], dtype=object)
        src = np.array([r[1] for r in res], dtype=object)
        df = pd.DataFrame({"sym": sym, "rank": [rank[s] for s in src],
                           "tot": np.asarray(totals, float), "col": np.arange(len(sym))})
        df = df[df["sym"].str.len() > 0]
        df = df.sort_values(["sym", "rank", "tot", "col"], ascending=[True, True, False, True],
                            kind="mergesort")
        df = df.drop_duplicates("sym", keep="first").sort_values("col")
        keep = df["col"].to_numpy()
        return {"keep_cols": keep, "genes": df["sym"].to_numpy(), "gene_ens": ens_ids[keep],
                "all_syms": sym, "all_src": src, "all_ens": ens_ids,
                "source_counts_kept": {k: int(v) for k, v in pd.Series(src[keep]).value_counts().items()}}

    # -- cytobands ----------------------------------------------------------
    def location(self, canon_sym: str) -> str:
        """HGNC location of a canonical symbol: own entry, newer-release symbol, or a
        unique previous symbol in HGNC_FULL ('' if none)."""
        loc = self.cyto.get(canon_sym)
        if not loc:
            loc = self.cyto.get(self.canon2new.get(canon_sym, ""), "")
        if not loc:
            loc = self.prev_loc.get(canon_sym, "")
        return loc or ""

    def intervals(self, canon_sym: str) -> tuple:
        return parse_location(self.location(canon_sym))

    def band(self, canon_sym: str) -> str:
        """Location string if it contains at least one chromosome band, else ''."""
        loc = self.location(canon_sym)
        return loc if parse_location(loc) else ""

    def is_cis(self, a: str, b: str) -> bool:
        return intervals_overlap(self.intervals(a), self.intervals(b))


@lru_cache(maxsize=1)
def gene_mapper() -> GeneMapper:
    return GeneMapper()


def hgnc_provenance() -> dict:
    """md5/sha256, size and a release proxy (latest date_modified) of the HGNC inputs."""
    out = {}
    for key, path in (("HGNC_TABLE", C.HGNC_TABLE), ("HGNC_FULL", C.HGNC_FULL)):
        rec = file_record(path)
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            for b in iter(lambda: fh.read(1 << 24), b""):
                h.update(b)
        rec["sha256"] = h.hexdigest()
        if key == "HGNC_FULL":
            d = pd.read_csv(path, sep="\t", dtype=str, usecols=["date_modified", "status"])
            rec.update({
                "n_rows": int(len(d)),
                "release_proxy_latest_date_modified": str(d["date_modified"].dropna().max()),
                "source": "HGNC complete set TSV (https://storage.googleapis.com/public-download-files/"
                          "hgnc/tsv/tsv/hgnc_complete_set.txt); byte-identical (same md5/sha256) to "
                          "subproject_38_genomic_proximity_bias_audit/data/hgnc_complete_set.tsv, "
                          "downloaded by that project's ensure_hgnc_table() (file dated 2026-02-17)",
                "used_for": "Ensembl id -> symbol (common.ensembl_to_symbol), cytogenetic bands "
                            "(common.symbol_cytoband), previous-symbol fallbacks"})
        else:
            d = pd.read_csv(path, sep="\t", dtype=str, usecols=["Approved symbol"])
            rec.update({"n_rows": int(len(d)),
                        "used_for": "canonical symbol space (common.normalize_symbol)",
                        "note": "mixes HGNC releases (e.g. both KIAA0125 and FAM30A approved); the "
                                "canonical space of all R2 stages is defined by this file"})
        out[key] = rec
    return out


def universe_gene_sets() -> dict:
    """Tissue-panel and global-universe gene sets (descriptive mapping report only)."""
    out = {"tissue_panels": set(), "global": set()}
    for t in C.TISSUES:
        f = UNIVERSE_DIR / t / "panel.tsv"
        if f.exists():
            out["tissue_panels"] |= set(pd.read_csv(f, sep="\t")["gene"].astype(str))
    g = GLOBAL_UNIVERSE_DIR / "genes.txt"
    if g.exists():
        out["global"] = set(g.read_text().split())
    return out


def mapping_report(mapper: GeneMapper, M: dict, M_legacy: dict, names, totals, universe: dict) -> dict:
    """Describe the prespecified mapping against the name-only and the Ensembl-first mappings."""
    tfs = C.lambert_tfs()
    _, name_only = dedup_symbols(names, totals)
    new, old, ef = set(M["genes"]), set(name_only), set(M_legacy["genes"])
    kept = M["keep_cols"]
    differs = [(str(names[c]), mapper.from_ensembl(M["all_ens"][c])) for c in kept
               if M["all_src"][c] == "symbol_ensembl_differs"]
    by_ens = [(str(names[c]), str(M["all_syms"][c])) for c in kept
              if M["all_src"][c] == "ensembl" and mapper.canon(names[c]) != M["all_syms"][c]]
    rep = {
        "rule": P["symbol_mapping"],
        "n_columns": int(len(M["all_syms"])),
        "n_columns_with_ensembl_id": int(sum(1 for e in M["all_ens"] if e)),
        "n_columns_ensembl_in_hgnc": int(sum(1 for e in M["all_ens"] if e and e in mapper.e2s)),
        "n_genes_kept": int(len(new)),
        "source_counts_kept": M["source_counts_kept"],
        "n_lambert_tfs": int(len(new & tfs)),
        "n_approved_names_kept_although_ensembl_id_maps_elsewhere": len(differs),
        "approved_names_kept_examples_(name,ensembl_symbol)": differs[:25],
        "n_nonapproved_names_resolved_by_ensembl": len(by_ens),
        "nonapproved_names_resolved_by_ensembl_examples": by_ens[:25],
        "n_genes_legacy_name_only": int(len(old)),
        "n_genes_recovered_vs_name_only": int(len(new - old)),
        "n_genes_lost_vs_name_only": int(len(old - new)),
        "tfs_recovered_vs_name_only": sorted(new & tfs - old),
        "tfs_lost_vs_name_only": sorted(old & tfs - new),
        "n_genes_phase1b_ensembl_first": int(len(ef)),
        "n_genes_recovered_vs_ensembl_first": int(len(new - ef)),
        "n_genes_lost_vs_ensembl_first": int(len(ef - new)),
        "tfs_recovered_vs_ensembl_first": sorted(new & tfs - ef),
        "tfs_lost_vs_ensembl_first": sorted(ef & tfs - new),
    }
    for uname, genes in universe.items():
        rep[f"n_{uname}_genes"] = int(len(new & genes))
        rep[f"{uname}_genes_recovered_vs_name_only"] = sorted((new - old) & genes)
        rep[f"{uname}_genes_lost_vs_name_only"] = sorted((old - new) & genes)
        rep[f"{uname}_genes_recovered_vs_ensembl_first"] = sorted((new - ef) & genes)
        rep[f"{uname}_genes_lost_vs_ensembl_first"] = sorted((ef - new) & genes)
    return rep


def resolve_labels(mapper: GeneMapper, labels, raw_names, M: dict) -> dict:
    """Canonical symbol of each perturbation label (approved label first).

    An approved label (``GeneMapper.is_approved_name``) is used as is. Any other
    label (e.g. 'QARS', an ambiguous previous symbol of QARS1 and EPRS1, or an
    outdated canonical name) takes the symbol of the matrix column with the same
    name in the same dataset (so label and column follow the same rule), if
    exactly one such symbol exists; otherwise its canonical (HGNC-normalised)
    form, or the symbol of the unique column whose normalised name equals it.
    """
    by_raw: dict[str, set] = {}
    by_norm: dict[str, set] = {}
    for raw, s in zip(list(raw_names), M["all_syms"]):
        if s:
            by_raw.setdefault(str(raw).strip().upper(), set()).add(s)
            by_norm.setdefault(C.normalize_symbol(raw), set()).add(s)
    out = {}
    for x in labels:
        u = str(x).strip().upper()
        if mapper.is_approved_name(u):
            out[x] = u
            continue
        cands = by_raw.get(u, set())
        if len(cands) == 1:
            out[x] = next(iter(cands))
            continue
        if u in mapper.approved:
            out[x] = u
            continue
        s = mapper.canon(x)
        if s not in mapper.approved:
            cands = by_norm.get(C.normalize_symbol(x), set()) | by_norm.get(s, set())
            if len(cands) == 1:
                s = next(iter(cands))
        out[x] = s
    return out


def label_aliases(mapper: GeneMapper, raw_labels, ens_ids=()) -> set:
    """Symbols that denote the perturbed gene besides its mapped label: each raw
    label (if approved) and its canonical form, and the Ensembl-derived symbol of
    the perturbation's own gene id(s) (Replogle rows)."""
    out = set()
    for r in raw_labels:
        u = str(r).strip().upper()
        if u in mapper.approved:
            out.add(u)
        c = mapper.canon(r)
        if c:
            out.add(c)
    for e in ens_ids:
        s = mapper.from_ensembl(e)
        if s:
            out.add(s)
    return out


def annotate_cis(de: pd.DataFrame, mapper: GeneMapper) -> pd.DataFrame:
    """Add cis_same_band (overlapping HGNC band intervals) and positive_primary."""
    if len(de) == 0:
        de["cis_same_band"] = pd.Series(dtype=bool)
        de["positive_primary"] = pd.Series(dtype=bool)
        return de
    pg = de["perturbed_gene"].astype("category")
    tg = de["target"].astype("category")
    pcats = [str(x) for x in pg.cat.categories]
    tcats = [str(x) for x in tg.cat.categories]
    t_iv = [mapper.intervals(t) for t in tcats]
    p_iv = [mapper.intervals(p) for p in pcats]
    arms = sorted({k for iv in t_iv + p_iv for k, _, _ in iv})
    aid = {a: i for i, a in enumerate(arms)}
    owner = np.array([i for i, iv in enumerate(t_iv) for _ in iv], dtype=np.int64)
    t_arm = np.array([aid[k] for iv in t_iv for k, _, _ in iv], dtype=np.int64)
    t_lo = np.array([lo for iv in t_iv for _, lo, _ in iv], dtype=np.int64)
    t_hi = np.array([hi for iv in t_iv for _, _, hi in iv], dtype=np.int64)
    cis = np.zeros((len(pcats), len(tcats)), dtype=bool)
    for i, iv in enumerate(p_iv):
        for k, lo, hi in iv:
            m = (t_arm == aid[k]) & (t_lo <= hi) & (lo <= t_hi)
            cis[i, owner[m]] = True
    de["cis_same_band"] = cis[pg.cat.codes.to_numpy(), tg.cat.codes.to_numpy()]
    de["positive_primary"] = de["positive_incl_cis"].to_numpy() & ~de["cis_same_band"].to_numpy()
    return de


DE_COLS = ["perturbed_gene", "target", "log2fc", "t", "pval", "padj", "positive_incl_cis",
           "cis_same_band", "positive_primary", "n_pert_cells", "n_ctrl_cells",
           "mean_pert", "mean_ctrl", "frac_pert", "frac_ctrl", "is_tf"]


def finalize_de(parts, tfs, mapper) -> pd.DataFrame:
    base = ["perturbed_gene", "target", "log2fc", "t", "pval", "padj", "positive_incl_cis",
            "n_pert_cells", "n_ctrl_cells", "mean_pert", "mean_ctrl", "frac_pert", "frac_ctrl"]
    if not parts:
        return pd.DataFrame(columns=DE_COLS)
    de = pd.concat(parts, ignore_index=True)[base]
    de["is_tf"] = de["perturbed_gene"].isin(tfs)
    de["perturbed_gene"] = de["perturbed_gene"].astype("category")
    de["target"] = de["target"].astype("category")
    de = annotate_cis(de, mapper)
    return de[DE_COLS]


def per_perturbation_counts(de: pd.DataFrame, suffix: str = "") -> pd.DataFrame:
    """Per-perturbation positives (non-cis and cis-inclusive), cis removals and FDR-only counts."""
    if len(de) == 0:
        return pd.DataFrame(columns=["perturbed_gene"])
    lfc = de["log2fc"].to_numpy(dtype=float)
    posp = de["positive_primary"].to_numpy()
    g = pd.DataFrame({
        "perturbed_gene": de["perturbed_gene"].astype(str).to_numpy(),
        "pos": de["positive_incl_cis"].to_numpy(), "posp": posp,
        "up": posp & (lfc > 0), "down": posp & (lfc < 0),
        "cis": de["cis_same_band"].to_numpy(),
        "cispos": (de["positive_incl_cis"] & de["cis_same_band"]).to_numpy(),
        "fdr": (de["padj"] < P["de_fdr"]).to_numpy(),
        "alfc": np.abs(lfc)}).groupby("perturbed_gene", sort=True)
    out = pd.DataFrame({
        f"n_targets_tested{suffix}": g.size().astype(int),
        f"n_positive_primary{suffix}": g["posp"].sum().astype(int),
        f"n_positive_primary_up{suffix}": g["up"].sum().astype(int),
        f"n_positive_primary_down{suffix}": g["down"].sum().astype(int),
        f"n_positive_incl_cis{suffix}": g["pos"].sum().astype(int),
        f"n_cis_tested{suffix}": g["cis"].sum().astype(int),
        f"n_cis_positive_removed{suffix}": g["cispos"].sum().astype(int),
        f"n_fdr_significant_any_lfc{suffix}": g["fdr"].sum().astype(int),
        f"q99_abs_log2fc{suffix}": g["alfc"].quantile(0.99)})
    return out.reset_index()


def robustness_table(de_primary: pd.DataFrame, de_sens: pd.DataFrame, primary: str,
                     sens: str) -> pd.DataFrame:
    """Per-perturbation agreement of the non-cis positive sets of the two DE tests.

    ``de_nonrobust_tf``: tested TF perturbation with Jaccard < NONROBUST_JACCARD
    (a perturbation with no positive under either test has Jaccard NaN and is
    not flagged).
    """
    def sets(de):
        d = de.loc[de["positive_primary"].to_numpy(), ["perturbed_gene", "target"]]
        return d.groupby(d["perturbed_gene"].astype(str), observed=True)["target"].apply(
            lambda s: set(s.astype(str))).to_dict()
    a, b = sets(de_primary), sets(de_sens)
    perts = sorted(set(de_primary["perturbed_gene"].astype(str)) | set(de_sens["perturbed_gene"].astype(str)))
    tf = set(de_primary.loc[de_primary["is_tf"].to_numpy(), "perturbed_gene"].astype(str))
    rows = []
    for p in perts:
        x, y = a.get(p, set()), b.get(p, set())
        u = x | y
        j = len(x & y) / len(u) if u else np.nan
        rows.append({"perturbed_gene": p, f"n_positive_primary_{primary}": len(x),
                     f"n_positive_primary_{sens}": len(y), "n_positive_primary_both_tests": len(x & y),
                     "jaccard_pooled_vs_stratified": j,
                     "de_nonrobust_tf": bool(p in tf and np.isfinite(j) and j < NONROBUST_JACCARD)})
    return pd.DataFrame(rows)


def cis_summary(de: pd.DataFrame, mapper: GeneMapper, n_examples: int = 10) -> dict:
    cp = de[de["positive_incl_cis"] & de["cis_same_band"]]
    top = (cp.groupby(cp["perturbed_gene"].astype(str), observed=True)["target"]
           .apply(lambda s: sorted(s.astype(str))).to_dict()) if len(cp) else {}
    order = sorted(top, key=lambda k: (-len(top[k]), k))[:n_examples]
    # the phase-1b rule (identical full location strings) for comparison
    pos = de[de["positive_incl_cis"]]
    pl = pos["perturbed_gene"].astype(str).map(mapper.band)
    tl = pos["target"].astype(str).map(mapper.band)
    exact = (pl.to_numpy() != "") & (pl.to_numpy() == tl.to_numpy())
    added = pos[pos["cis_same_band"].to_numpy() & ~exact]
    return {"rule": "target and perturbed gene share an HGNC band at the finest common resolution "
                    "(band intervals overlap on the same chromosome arm; ranges and coarser "
                    "annotations contain finer bands; previous-symbol fallback for the lookup)",
            "n_cis_pairs_tested": int(de["cis_same_band"].sum()),
            "n_cis_positive_removed": int(len(cp)),
            "n_cis_positive_removed_tf": int(cp["is_tf"].sum()),
            "n_perturbations_with_cis_positive": int(len(top)),
            "n_cis_positive_removed_exact_string_rule": int(exact.sum()),
            "n_cis_positive_added_by_interval_rule": int(len(added)),
            "n_cis_positive_added_by_interval_rule_tf": int(added["is_tf"].sum()),
            "added_examples": [(str(p), mapper.band(str(p)), str(t), mapper.band(str(t)))
                               for p, t in zip(added["perturbed_gene"].head(10), added["target"].head(10))],
            "examples": {k: {"band": mapper.band(k), "targets": top[k]} for k in order}}


def self_pair_check(de: pd.DataFrame, mapper: GeneMapper, gene_ens_of: dict, raw_name_of: dict,
                    aliases_of: dict | None = None, extra_ens_of: dict | None = None) -> dict:
    """No target may be the perturbed gene.

    Checks, per perturbation, targets equal to (i) the mapped label, (ii) any
    alias of it (raw labels' canonical forms, the Ensembl-derived symbol of the
    perturbation's own id; ``label_aliases``), (iii) a target column carrying an
    HGNC Ensembl id of the label or an alias, or the perturbation row's own id,
    (iv) a target whose raw column name normalises to the label or an alias.
    Any hit fails the check.
    """
    aliases_of = aliases_of or {}
    extra_ens_of = extra_ens_of or {}
    pgc = de["perturbed_gene"].astype("category")
    tgc = de["target"].astype("category")
    tcats = list(tgc.cat.categories)
    tcode = {t: i for i, t in enumerate(tcats)}
    ens_codes: dict[str, list] = {}
    name_codes: dict[str, list] = {}
    for t, i in tcode.items():
        e = gene_ens_of.get(t, "")
        if e:
            ens_codes.setdefault(e, []).append(i)
        name_codes.setdefault(mapper.canon(raw_name_of.get(t, "")), []).append(i)
    tcodes = tgc.cat.codes.to_numpy()
    groups = pd.Series(np.arange(len(de))).groupby(pgc.cat.codes.to_numpy()).indices
    hits = {"symbol": [], "raw_label": [], "ensembl": [], "raw_name": []}
    for code, pos in groups.items():
        pg = str(pgc.cat.categories[code])
        al = set(aliases_of.get(pg, ())) | {pg}
        ens = {e for a in al for e in mapper.ensembl_of(a)} | set(extra_ens_of.get(pg, ()))
        bad = {"symbol": [tcode[pg]] if pg in tcode else [],
               "raw_label": [tcode[a] for a in sorted(al - {pg}) if a in tcode],
               "ensembl": [c for e in sorted(ens) for c in ens_codes.get(e, [])],
               "raw_name": [c for a in sorted(al) for c in name_codes.get(a, [])]}
        for kind, b in bad.items():
            if b:
                tc = tcodes[pos]
                hits[kind] += [(pg, tcats[c]) for c in tc[np.isin(tc, b)]]
    return {"n_self_pairs_by_symbol": len(hits["symbol"]),
            "n_self_pairs_by_raw_label_alias": len(hits["raw_label"]),
            "n_self_pairs_by_ensembl": len(hits["ensembl"]),
            "n_targets_whose_raw_name_is_the_perturbed_gene": len(hits["raw_name"]),
            "examples": {k: v[:10] for k, v in hits.items() if v},
            "passed": not any(hits.values())}


# ----------------------------------------------------------------------------
# Expression matrix: detection, normalisation and group statistics
# ----------------------------------------------------------------------------

@dataclass
class ExprMatrix:
    """Cells x genes CSR matrix with log1p(CP10k) values and parallel CP10k values."""
    indptr: np.ndarray
    indices: np.ndarray
    logv: np.ndarray       # log1p(CP10k) per stored entry
    normv: np.ndarray      # CP10k per stored entry
    n_genes: int
    scale_info: dict

    def positions(self, rows: np.ndarray) -> np.ndarray:
        starts = self.indptr[rows]
        lens = self.indptr[rows + 1] - starts
        tot = int(lens.sum())
        if tot == 0:
            return np.zeros(0, dtype=np.int64)
        offs = np.repeat(starts - np.concatenate([[0], np.cumsum(lens)[:-1]]), lens)
        return offs + np.arange(tot)

    def group_stats(self, rows: np.ndarray) -> dict:
        rows = np.asarray(rows, dtype=np.int64)
        pos = self.positions(rows)
        idx = self.indices[pos]
        lv = self.logv[pos].astype(np.float64)
        G = self.n_genes
        n = len(rows)
        s1 = np.bincount(idx, weights=lv, minlength=G)
        s2 = np.bincount(idx, weights=lv * lv, minlength=G)
        sn = np.bincount(idx, weights=self.normv[pos].astype(np.float64), minlength=G)
        nz = np.bincount(idx, minlength=G)
        mean = s1 / n
        var = np.clip((s2 - n * mean * mean) / max(n - 1, 1), 0.0, None)
        return {"n": n, "mean": mean, "var": var, "mean_norm": sn / n, "frac": nz / n}


def classify_scale(X: sp.csr_matrix, n_check: int = 2000, seed: int = 0) -> dict:
    """Decide whether X holds raw counts or log1p(CP10k) values (by inspection)."""
    rng = np.random.default_rng(seed)
    rows = np.sort(rng.choice(X.shape[0], size=min(n_check, X.shape[0]), replace=False))
    sub = X[rows]
    d = sub.data.astype(np.float64)
    if d.size and np.all(np.abs(d - np.round(d)) < 1e-6) and d.max() > 20:
        return {"scale": "raw_counts", "checked_cells": int(len(rows)),
                "max_value_checked": float(d.max())}
    # log1p(CP10k) check: expm1(x) / min-nonzero-per-cell must be integer
    e = sub.copy()
    e.data = np.expm1(e.data.astype(np.float64))
    rowmin = np.array([e.data[e.indptr[i]:e.indptr[i + 1]].min() if e.indptr[i + 1] > e.indptr[i]
                       else np.nan for i in range(e.shape[0])])
    ratio = e.data / np.repeat(rowmin, np.diff(e.indptr))
    int_frac = float(np.mean(np.abs(ratio - np.round(ratio)) < 1e-2))
    if int_frac > 0.999:
        lib = TARGET_SUM / rowmin            # implied total counts over ALL genes
        stored = np.asarray(e.sum(1)).ravel()
        return {"scale": "log1p_cp10k", "checked_cells": int(len(rows)),
                "integer_reconstruction_fraction": int_frac,
                "implied_library_size_median": float(np.nanmedian(lib)),
                "stored_genes_fraction_of_cp10k_median": float(np.nanmedian(stored / TARGET_SUM))}
    raise ValueError(f"Cannot classify expression scale (integer fraction {int_frac:.3f})")


def build_expr(X: sp.csr_matrix, scale: dict) -> ExprMatrix:
    X = X.tocsr()
    X.sort_indices()
    if scale["scale"] == "raw_counts":
        data = X.data.astype(np.float64)
        lib = np.asarray(X.sum(1)).ravel().astype(np.float64)
        if np.any(lib <= 0):
            raise ValueError("cells with zero library size")
        rowfac = np.repeat(TARGET_SUM / lib, np.diff(X.indptr))
        normv = (data * rowfac).astype(np.float32)
        logv = np.log1p(normv.astype(np.float64)).astype(np.float32)
        info = dict(scale, normalisation="normalize_total(target_sum=1e4) over all genes + log1p",
                    library_size_median=float(np.median(lib)))
    else:
        logv = X.data.astype(np.float32)
        normv = np.expm1(X.data.astype(np.float64)).astype(np.float32)
        info = dict(scale, normalisation="stored values already log1p(CP10k) over all genes; used as is")
    return ExprMatrix(X.indptr.astype(np.int64), X.indices.astype(np.int32), logv, normv,
                      X.shape[1], info)


# ----------------------------------------------------------------------------
# Differential expression
# ----------------------------------------------------------------------------

def welch(g1: dict, g0: dict, cols=None):
    """Welch t statistic and two-sided p (vectorised over genes)."""
    def pick(a):
        return a if cols is None else a[cols]
    n1, n0 = g1["n"], g0["n"]
    m1, m0 = pick(g1["mean"]), pick(g0["mean"])
    a1, a0 = pick(g1["var"]) / n1, pick(g0["var"]) / n0
    se2 = a1 + a0
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (m1 - m0) / np.sqrt(se2)
        df = se2 ** 2 / (a1 ** 2 / max(n1 - 1, 1) + a0 ** 2 / max(n0 - 1, 1))
    bad = ~(se2 > 0)
    t = np.where(bad, 0.0, t)
    df = np.where(bad | ~np.isfinite(df), 1.0, df)
    p = 2.0 * stats.t.sf(np.abs(t), df)
    p = np.where(bad, 1.0, p)
    return t, p


def log2fc(mean_pert, mean_ctrl):
    c = LOG2FC_PSEUDOCOUNT_CP10K
    return np.log2((mean_pert + c) / (mean_ctrl + c))


def _ontarget_from(lfc, t, p, frac1, frac0, m1, m0, direction, in_matrix=True) -> dict:
    if not in_matrix:
        return {"ontarget_in_matrix": False, "ontarget_log2fc": np.nan, "ontarget_t": np.nan,
                "ontarget_pval": np.nan, "ontarget_frac_pert": np.nan,
                "ontarget_frac_ctrl": np.nan, "ontarget_mean_pert": np.nan,
                "ontarget_mean_ctrl": np.nan, "ontarget_pass": False}
    sign_ok = lfc < 0 if direction == "down" else lfc > 0
    return {"ontarget_in_matrix": True, "ontarget_log2fc": float(lfc), "ontarget_t": float(t),
            "ontarget_pval": float(p), "ontarget_frac_pert": float(frac1),
            "ontarget_frac_ctrl": float(frac0), "ontarget_mean_pert": float(m1),
            "ontarget_mean_ctrl": float(m0),
            "ontarget_pass": bool(sign_ok and p < ONTARGET_P and abs(lfc) > P["de_min_abs_log2fc"])}


def ontarget_record(gi, g1, g0, direction) -> dict:
    """On-target statistics of the pooled (Welch) test."""
    if gi is None:
        return _ontarget_from(np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, direction, False)
    t, p = welch(g1, g0, np.array([gi]))
    lfc = float(log2fc(g1["mean_norm"][gi], g0["mean_norm"][gi]))
    return _ontarget_from(lfc, t[0], p[0], g1["frac"][gi], g0["frac"][gi],
                          g1["mean_norm"][gi], g0["mean_norm"][gi], direction)


def _test_cols(g1, g0, exclude) -> np.ndarray:
    thr = P["de_min_target_frac_expressed"]
    keep = (g1["frac"] >= thr) | (g0["frac"] >= thr)
    for gi in exclude:
        keep[gi] = False
    return np.flatnonzero(keep)


def de_one(pert_gene, genes, g1, g0, exclude):
    """Pooled Welch DE table for one perturbation (own columns excluded; expression filter)."""
    cols = _test_cols(g1, g0, exclude)
    t, p = welch(g1, g0, cols)
    lfc = log2fc(g1["mean_norm"][cols], g0["mean_norm"][cols])
    padj = bh(p)
    pos = (padj < P["de_fdr"]) & (np.abs(lfc) > P["de_min_abs_log2fc"])
    return pd.DataFrame({
        "perturbed_gene": pert_gene, "target": genes[cols],
        "log2fc": lfc.astype(np.float32), "t": t.astype(np.float32),
        "pval": p, "padj": padj, "positive_incl_cis": pos,
        "n_pert_cells": np.int32(g1["n"]), "n_ctrl_cells": np.int32(g0["n"]),
        "mean_pert": g1["mean_norm"][cols].astype(np.float32),
        "mean_ctrl": g0["mean_norm"][cols].astype(np.float32),
        "frac_pert": g1["frac"][cols].astype(np.float32),
        "frac_ctrl": g0["frac"][cols].astype(np.float32),
    })


def stratified_stats(rows1, rows0, strata, E, cols) -> dict:
    """Stratified two-sample t-test (stratum-specific means, common variance) for ``cols``.

    Model per gene: y = mu_{stratum, group} + e, e ~ N(0, sigma^2). The
    perturbation effect is the weighted mean of within-stratum differences
    d_s = mean_pert_s - mean_ctrl_s with weights w_s = n1_s n0_s / (n1_s + n0_s)
    (the Mantel-Haenszel/van Elteren-type weights of a blocked comparison);
    Var = sigma^2 / sum_s w_s with sigma^2 the pooled within-cell variance on
    df = N - 2 S (equals the OLS cell-means contrast). Strata with
    < STRATUM_MIN_CELLS cells in either group are dropped. CP10k means: perturbed
    cells of the used strata, and control means post-stratified to the
    perturbed cells' stratum composition.
    """
    cols = np.asarray(cols)
    sw, swd, sse = 0.0, np.zeros(cols.size), np.zeros(cols.size)
    ctrl_adj, pert_sum = np.zeros(cols.size), np.zeros(cols.size)
    used = []
    for s in np.unique(strata[rows1]):
        r1 = rows1[strata[rows1] == s]
        r0 = rows0[strata[rows0] == s]
        if len(r1) < STRATUM_MIN_CELLS or len(r0) < STRATUM_MIN_CELLS:
            continue
        a, b = E.group_stats(r1), E.group_stats(r0)
        w = a["n"] * b["n"] / (a["n"] + b["n"])
        sw += w
        swd += w * (a["mean"][cols] - b["mean"][cols])
        sse += (a["n"] - 1) * a["var"][cols] + (b["n"] - 1) * b["var"][cols]
        ctrl_adj += a["n"] * b["mean_norm"][cols]
        pert_sum += a["n"] * a["mean_norm"][cols]
        used.append((s, len(r1), len(r0)))
    n1u = sum(u[1] for u in used)
    n0u = sum(u[2] for u in used)
    df = n1u + n0u - 2 * len(used)
    if not used or df < 1:
        return {"t": np.zeros(cols.size), "p": np.ones(cols.size), "mean_pert": None,
                "mean_ctrl": None, "n1": n1u, "n0": n0u, "used": used, "df": df}
    sigma2 = sse / df
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (swd / sw) * np.sqrt(sw) / np.sqrt(sigma2)
    bad = ~(sigma2 > 0)
    t = np.where(bad, 0.0, t)
    p = np.where(bad, 1.0, 2.0 * stats.t.sf(np.abs(t), df))
    return {"t": t, "p": p, "mean_pert": pert_sum / n1u, "mean_ctrl": ctrl_adj / n1u,
            "n1": n1u, "n0": n0u, "used": used, "df": df}


def de_stratified_one(pert_gene, genes, rows1, rows0, strata, E, exclude, g1=None, g0=None):
    """Stratified DE table for one perturbation (see ``stratified_stats``).

    The expression filter uses the pooled groups, as in the pooled test.
    """
    g1 = E.group_stats(rows1) if g1 is None else g1
    g0 = E.group_stats(rows0) if g0 is None else g0
    cols = _test_cols(g1, g0, exclude)
    r = stratified_stats(rows1, rows0, strata, E, cols)
    mp = g1["mean_norm"][cols] if r["mean_pert"] is None else r["mean_pert"]
    mc = g0["mean_norm"][cols] if r["mean_ctrl"] is None else r["mean_ctrl"]
    lfc = log2fc(mp, mc)
    padj = bh(r["p"])
    pos = (padj < P["de_fdr"]) & (np.abs(lfc) > P["de_min_abs_log2fc"])
    return pd.DataFrame({
        "perturbed_gene": pert_gene, "target": genes[cols],
        "log2fc": lfc.astype(np.float32), "t": r["t"].astype(np.float32),
        "pval": r["p"], "padj": padj, "positive_incl_cis": pos,
        "n_pert_cells": np.int32(r["n1"]), "n_ctrl_cells": np.int32(r["n0"]),
        "mean_pert": mp.astype(np.float32), "mean_ctrl": mc.astype(np.float32),
        "frac_pert": g1["frac"][cols].astype(np.float32),
        "frac_ctrl": g0["frac"][cols].astype(np.float32),
    }), r["used"]


def ontarget_stratified(gi, rows1, rows0, strata, E, g1, g0, direction) -> dict:
    if gi is None:
        return ontarget_record(None, g1, g0, direction)
    r = stratified_stats(rows1, rows0, strata, E, np.array([gi]))
    if r["mean_pert"] is None:
        return _ontarget_from(np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, direction, False)
    lfc = float(log2fc(r["mean_pert"][0], r["mean_ctrl"][0]))
    return _ontarget_from(lfc, r["t"][0], r["p"][0], g1["frac"][gi], g0["frac"][gi],
                          r["mean_pert"][0], r["mean_ctrl"][0], direction)


# ----------------------------------------------------------------------------
# Cohort loaders: return counts/log matrix, raw gene symbols, gene ids, labels
# ----------------------------------------------------------------------------

@dataclass
class Loaded:
    X: sp.csr_matrix
    raw_symbols: list
    gene_ids: object           # per-column Ensembl ids (or None)
    gene_id_source: str
    labels: pd.DataFrame       # index = kept cells; columns: pert_gene_raw, guide, stratum
    label_notes: dict
    raw_counts_probe: object = None
    extra_inputs: dict = field(default_factory=dict)


def _read(path: Path) -> ad.AnnData:
    a = ad.read_h5ad(path)
    if not sp.issparse(a.X):
        a.X = sp.csr_matrix(a.X)
    return a


def _finish(a: ad.AnnData, keep: np.ndarray, labels: pd.DataFrame, notes: dict,
            gene_ids, gene_id_source: str, symbols=None, extra_inputs=None) -> Loaded:
    keep = np.asarray(keep, bool)
    X = a.X
    if sp.isspmatrix_csc(X) or isinstance(X, sp.csc_array):
        X = X.tocsr()
    X = X[np.flatnonzero(keep)]
    labels = labels.loc[keep].copy()
    sym = list(a.var_names) if symbols is None else list(symbols)
    return Loaded(sp.csr_matrix(X), sym, gene_ids, gene_id_source, labels, notes,
                  extra_inputs=extra_inputs or {})


def load_dixit(spec: CohortSpec) -> Loaded:
    a = _read(spec.path)
    o = a.obs
    pert = o["perturbation"].astype(str)
    is_ctrl = pert.eq("control")
    is_pert = ~is_ctrl & ~pert.eq("nan") & o["target"].notna()
    lab = pd.DataFrame(index=o.index)
    lab["pert_gene_raw"] = np.where(is_ctrl, CONTROL, o["target"].astype(str))
    lab["guide"] = o["guide_id"].astype(str)
    lab["stratum"] = "all"
    notes = {"control_definition": "perturbation == 'control' (4 INTERGENIC guides)",
             "excluded": f"{int((~is_ctrl & ~is_pert).sum())} cells with no assigned guide "
                         "(perturbation == 'nan')",
             "guide_collapse": "obs['target'] (gene) from guides p-sg<GENE>-<k>"}
    return _finish(a, (is_ctrl | is_pert).to_numpy(), lab, notes,
                   a.var["gene_id"].astype(str).to_numpy(), "var['gene_id']")


def load_adamson(spec: CohortSpec) -> Loaded:
    a = _read(spec.path)
    o = a.obs
    cond = o["condition_corrected"].astype(str)
    collide = o["potential_collision"].astype(bool) | o["barcode_collision"].astype(bool)
    is_ctrl = cond.eq("control")
    keep = ~collide
    lab = pd.DataFrame(index=o.index)
    lab["pert_gene_raw"] = np.where(is_ctrl, CONTROL, cond)
    lab["guide"] = o["guide_identity"].astype(str)
    lab["stratum"] = "all"
    meta = a.uns.get("adamson_metadata_correction", {})
    notes = {"control_definition": "condition_corrected == 'control' (non-targeting guide 63(mod)_pBA580)",
             "label_source": "condition_corrected: GEARS 'adamson' cells restricted to the UPR "
                             "screen (GSM2406681) with labels re-derived from GEO cell-identity "
                             "files (pfizer-opensource/perturb_seq update_GEARS_Adamson_metadata); "
                             "expression values unmodified from GEARS",
             "correction_metadata": {k: str(v) for k, v in dict(meta).items()},
             "excluded": f"{int(collide.sum())} cells flagged potential_collision/barcode_collision",
             "gene_space": "GEARS highly-variable subset only (stored genes); "
                           "values are log1p(CP10k) normalised over all genes"}
    return _finish(a, keep.to_numpy(), lab, notes, a.var["ensembl_id"].astype(str).to_numpy(),
                   "var['ensembl_id']", symbols=a.var["gene_name"].astype(str))


def _shifrut_gene_ids(a: ad.AnnData):
    """Shifrut has no gene ids; reuse Norman's when the var names are identical in order."""
    ref_path = COHORTS["norman"].path
    ref = ad.read_h5ad(ref_path, backed="r")
    try:
        same = len(ref.var_names) == len(a.var_names) and bool(
            np.all(np.asarray(ref.var_names) == np.asarray(a.var_names)))
        ids = ref.var["ensemble_id"].astype(str).to_numpy() if same else None
    finally:
        ref.file.close()
    if ids is None:
        return None, "none (var names differ from the Norman 10x reference; symbols only)", {}
    return ids, ("Norman 2019 var['ensemble_id']: the 33,694 Shifrut var names are identical, in "
                 "order, to Norman's (10x GRCh38-1.2.0 reference); checked element-wise"), \
        {"gene_id_reference": file_record(ref_path)}


def load_shifrut(spec: CohortSpec) -> Loaded:
    a = _read(spec.path)
    o = a.obs
    pert = o["perturbation"].astype(str)
    is_ctrl = pert.eq("control")
    is_pert = ~is_ctrl & o["perturbation"].notna() & ~pert.eq("nan")
    lab = pd.DataFrame(index=o.index)
    lab["pert_gene_raw"] = np.where(is_ctrl, CONTROL, pert)
    lab["guide"] = o["guide_id"].astype(str)
    lab["stratum"] = o["sample"].astype(str)
    notes = {"control_definition": "perturbation == 'control' (7 NonTarget guides)",
             "excluded": f"{int((~is_ctrl & ~is_pert).sum())} cells with no single assigned guide",
             "strata": "donors D1, D2 x {stim, nostim}; primary DE stratified by sample "
                       "(PRESPECIFIED['replicate_stratified_de']); pooled DE in de_pooled.parquet"}
    ids, src, extra = _shifrut_gene_ids(a)
    return _finish(a, (is_ctrl | is_pert).to_numpy(), lab, notes, ids, src, extra_inputs=extra)


def load_norman(spec: CohortSpec) -> Loaded:
    a = _read(spec.path)
    o = a.obs
    pert = o["perturbation"].astype(str)
    qc = o["good_coverage"].astype(bool) & o["number_of_cells"].eq(1)
    is_ctrl = pert.eq("control") & o["nperts"].eq(0)
    is_single = o["nperts"].eq(1) & ~pert.str.contains("_")
    keep = qc & (is_ctrl | is_single)
    lab = pd.DataFrame(index=o.index)
    lab["pert_gene_raw"] = np.where(is_ctrl, CONTROL, pert)
    lab["guide"] = o["guide_id"].astype(str).str.split(";").str[0]
    lab["stratum"] = o["gemgroup"].astype(str)
    notes = {"control_definition": "perturbation == 'control' (NegCtrl guide pairs), nperts == 0",
             "single_gene": "nperts == 1 (gene paired with NegCtrl0); dual-gene perturbations excluded",
             "qc": "good_coverage == True and number_of_cells == 1 (single confident guide identity)",
             "excluded_qc": int(((is_ctrl | is_single) & ~qc).sum()),
             "excluded_dual": int(o["nperts"].eq(2).sum())}
    return _finish(a, keep.to_numpy(), lab, notes, a.var["ensemble_id"].astype(str).to_numpy(),
                   "var['ensemble_id']")


def load_papalexi(spec: CohortSpec) -> Loaded:
    a = _read(spec.path)
    o = a.obs
    pert = o["perturbation"].astype(str)
    is_ctrl = pert.eq("control")
    hto = o["hto"].astype(str)
    keep_hto = hto.str.endswith("-tx")
    gene = pert.str.replace(r"g\d+$", "", regex=True)
    lab = pd.DataFrame(index=o.index)
    lab["pert_gene_raw"] = np.where(is_ctrl, CONTROL, gene)
    lab["guide"] = o["guide_id"].astype(str)
    lab["stratum"] = hto
    notes = {"control_definition": "perturbation == 'control' (NT guides NTg1..NTg10)",
             "guide_collapse": "guides <GENE>g1..g4 collapsed to <GENE>",
             "excluded": f"{int((~keep_hto).sum())} cells with HTO 'rep2-ctrl' (labelled as a "
                         "control condition, unlike the IFN-gamma-treated '-tx' replicates)",
             "escapers": "no mixscape-style removal of non-perturbed (escaping) cells",
             "strata": "4 IFN-gamma-treated replicates (HTO rep1-tx..rep4-tx); primary DE stratified "
                       "by replicate (PRESPECIFIED['replicate_stratified_de']); pooled DE in "
                       "de_pooled.parquet"}
    return _finish(a, keep_hto.to_numpy(), lab, notes, a.var["ensembl_id"].astype(str).to_numpy(),
                   "var['ensembl_id'] (Ensembl id where it is an ENSG id, otherwise a symbol)")


LOADERS = {"dixit_d13": load_dixit, "dixit_d7": load_dixit, "adamson": load_adamson,
           "shifrut": load_shifrut, "norman": load_norman, "papalexi": load_papalexi}


# ----------------------------------------------------------------------------
# Single-cell cohort driver
# ----------------------------------------------------------------------------

def _own_columns(pg, gene_index, ens_to_col, mapper, aliases=(), extra_ens=()) -> list:
    """Kept columns that are the perturbed gene: by mapped label or alias symbol
    (``label_aliases``), by HGNC Ensembl id of either, or by the perturbation's own
    gene id (``extra_ens``)."""
    syms = {pg} | set(aliases)
    own = {gene_index[s] for s in syms if s in gene_index}
    for e in {e for s in syms for e in mapper.ensembl_of(s)} | set(extra_ens):
        if e in ens_to_col:
            own.add(ens_to_col[e])
    return sorted(own)


def _strata_tvd(strata_p: pd.Series, strata_c: pd.Series) -> float:
    a = strata_p.value_counts(normalize=True)
    b = strata_c.value_counts(normalize=True)
    idx = a.index.union(b.index)
    return float(0.5 * np.abs(a.reindex(idx, fill_value=0) - b.reindex(idx, fill_value=0)).sum())


def build_singlecell(spec: CohortSpec, out_dir: Path) -> dict:
    t0 = time.time()
    tfs = C.lambert_tfs()
    mapper = gene_mapper()
    cid = spec.cohort_id
    stratified = cid in STRATIFIED
    primary = PRIMARY_DE.get(cid, "pooled")
    log(f"{cid}: reading {spec.path}")
    L = LOADERS[cid](spec)
    scale = classify_scale(L.X, seed=P["random_seed"])
    log(f"{cid}: {L.X.shape[0]} cells x {L.X.shape[1]} genes; scale={scale['scale']}")

    totals = np.asarray(L.X.sum(0)).ravel()
    probe_rows = np.arange(min(50, L.X.shape[0]))
    L.raw_counts_probe = (probe_rows, L.X[probe_rows].copy()) if scale["scale"] == "raw_counts" else None
    E = build_expr(L.X, scale)        # normalisation uses ALL stored genes
    del L.X
    gc.collect()
    M = mapper.map_columns(L.raw_symbols, L.gene_ids, totals)
    M_legacy = mapper.map_columns(L.raw_symbols, L.gene_ids, totals, legacy=True)
    keep_cols, genes, gene_ens = M["keep_cols"], M["genes"], M["gene_ens"]
    gene_index = {g: i for i, g in enumerate(genes)}
    ens_to_col = {e: i for i, e in enumerate(gene_ens) if e}
    map_rep = mapping_report(mapper, M, M_legacy, L.raw_symbols, totals, universe_gene_sets())
    map_rep["gene_id_source"] = L.gene_id_source
    log(f"{cid}: genes {map_rep['n_genes_kept']} (name-only {map_rep['n_genes_legacy_name_only']}; "
        f"+{map_rep['n_genes_recovered_vs_name_only']} -{map_rep['n_genes_lost_vs_name_only']}; "
        f"vs Ensembl-first +{map_rep['n_genes_recovered_vs_ensembl_first']} "
        f"-{map_rep['n_genes_lost_vs_ensembl_first']}); TFs recovered vs Ensembl-first "
        f"{map_rep['tfs_recovered_vs_ensembl_first']}")
    # statistics are computed over all stored columns and then restricted to the
    # kept, canonically named columns
    EV = _ColumnView(E, keep_cols)

    lab = L.labels
    label_map = resolve_labels(mapper, sorted(set(lab["pert_gene_raw"]) - {CONTROL}), L.raw_symbols, M)
    map_rep["labels_resolved_via_column_name"] = {k: v for k, v in label_map.items()
                                                  if v != mapper.canon(k)}
    lab["pert_gene"] = lab["pert_gene_raw"].map(lambda x: CONTROL if x == CONTROL else label_map[x])
    rows_all = np.arange(len(lab))
    pg_arr = lab["pert_gene"].to_numpy()
    ctrl_rows = rows_all[pg_arr == CONTROL]
    g0 = EV.group_stats(ctrl_rows)

    strata = lab["stratum"].to_numpy()
    multi = lab["stratum"].nunique() > 1
    pert_genes = sorted(set(pg_arr) - {CONTROL})
    pooled_parts, strat_parts, prow, aliases_of = [], [], [], {}
    other = "stratified" if primary == "pooled" else "pooled"
    for pg in pert_genes:
        r1 = rows_all[pg_arr == pg]
        guides = sorted(lab["guide"].iloc[r1].unique().tolist())
        raw_labels = sorted(lab["pert_gene_raw"].iloc[r1].unique())
        aliases_of[pg] = label_aliases(mapper, raw_labels) - {pg}
        own = _own_columns(pg, gene_index, ens_to_col, mapper, aliases_of[pg])
        gi = own[0] if own else None
        if pg in gene_index:
            gi = gene_index[pg]
        rec = {"perturbed_gene": pg, "perturbed_gene_raw": ";".join(raw_labels), "is_tf": pg in tfs,
               "n_pert_cells": int(len(r1)), "n_ctrl_cells": int(len(ctrl_rows)),
               "n_guides": len(guides), "guides": ";".join(guides),
               "n_pseudobulk_rows": np.nan,
               "tested": len(r1) >= P["de_min_cells"], "primary_de": primary,
               "n_own_columns_excluded": len(own)}
        if multi:
            comp1 = lab["stratum"].iloc[r1].value_counts()
            rec["strata_counts"] = json.dumps({str(k): int(v) for k, v in sorted(comp1.items())})
            rec["strata_tvd_vs_ctrl"] = _strata_tvd(lab["stratum"].iloc[r1], lab["stratum"].iloc[ctrl_rows])
        if cid == "shifrut":
            rec["frac_stim_pert"] = float(np.mean([str(s).endswith("_stim") for s in strata[r1]]))
            rec["frac_stim_ctrl"] = float(np.mean([str(s).endswith("_stim") for s in strata[ctrl_rows]]))
        g1 = EV.group_stats(r1)
        ot_pooled = ontarget_record(gi, g1, g0, spec.expected_direction)
        if stratified:
            ot_strat = ontarget_stratified(gi, r1, ctrl_rows, strata, EV, g1, g0, spec.expected_direction)
            ot_primary, ot_other = (ot_strat, ot_pooled) if primary == "stratified" else (ot_pooled, ot_strat)
            rec.update(ot_primary)
            rec.update({f"{k}_{other}": v for k, v in ot_other.items()
                        if k in ("ontarget_log2fc", "ontarget_pval", "ontarget_pass")})
        else:
            rec.update(ot_pooled)
        npos = "-"
        if rec["tested"]:
            df = de_one(pg, genes, g1, g0, own)
            pooled_parts.append(df)
            npos = int(df["positive_incl_cis"].sum())
            if stratified:
                sdf, used = de_stratified_one(pg, genes, r1, ctrl_rows, strata, EV, own, g1, g0)
                rec["strata_used"] = json.dumps([[str(s), int(a), int(b)] for s, a, b in used])
                strat_parts.append(sdf)
                if primary == "stratified":
                    npos = int(sdf["positive_incl_cis"].sum())
        prow.append(rec)
        log(f"{cid}: {pg:>10s} n={len(r1):5d} tested={rec['tested']} "
            f"pos(incl. cis, {primary})={npos} ontarget_lfc={rec['ontarget_log2fc']:.2f}")

    pert_df = pd.DataFrame(prow)
    de_pooled = finalize_de(pooled_parts, tfs, mapper)
    de_strat = finalize_de(strat_parts, tfs, mapper) if stratified else None
    de = de_strat if primary == "stratified" else de_pooled
    de_other = (de_pooled if primary == "stratified" else de_strat) if stratified else None
    pert_df = pert_df.merge(per_perturbation_counts(de), on="perturbed_gene", how="left")
    if de_other is not None:
        pert_df = pert_df.merge(robustness_table(de, de_other, primary, other),
                                on="perturbed_gene", how="left")
        pert_df["de_nonrobust_tf"] = pert_df["de_nonrobust_tf"].fillna(False).astype(bool)

    C.ensure_dir(out_dir)
    de.to_parquet(out_dir / "de.parquet", index=False, compression="zstd")
    for variant in ("pooled", "stratified"):          # remove stale non-primary files
        f = out_dir / f"de_{variant}.parquet"
        if f.exists() and not (de_other is not None and variant == other):
            f.unlink()
    if de_other is not None:
        de_other.to_parquet(out_dir / f"de_{other}.parquet", index=False, compression="zstd")
    pert_df.to_csv(out_dir / "perturbations.csv", index=False)

    extra = {"primary_de": primary, "de_design": DE_DESIGN[cid],
             "primary_de_rationale": PRIMARY_DE_RATIONALE.get(cid, "pooled Welch test (default)")}
    if stratified:
        def pairs(d, col):
            m = d[col].to_numpy()
            return set(zip(d.loc[m, "perturbed_gene"].astype(str), d.loc[m, "target"].astype(str)))
        a, b = pairs(de_pooled, "positive_primary"), pairs(de_strat, "positive_primary")
        ai, bi = pairs(de_pooled, "positive_incl_cis"), pairs(de_strat, "positive_incl_cis")
        nonrobust = pert_df[pert_df["de_nonrobust_tf"]]
        extra["stratified_test"] = {
            "strata": STRATIFIED[cid],
            "role": "primary" if primary == "stratified" else "sensitivity",
            "method": "stratified two-sample t-test on log1p(CP10k): weighted mean of within-"
                      "stratum differences, weights n1*n0/(n1+n0), pooled within-cell variance "
                      "(df = N - 2S); BH within perturbation; log2FC vs. control CP10k means "
                      "post-stratified to the perturbed cells' stratum composition",
            "min_cells_per_group_per_stratum": STRATUM_MIN_CELLS,
            "n_positive_primary_pooled": len(a), "n_positive_primary_stratified": len(b),
            "n_positive_primary_both": len(a & b),
            "jaccard_overall_non_cis": (len(a & b) / len(a | b)) if (a | b) else None,
            "jaccard_overall_incl_cis": (len(ai & bi) / len(ai | bi)) if (ai | bi) else None,
            "median_jaccard_per_perturbation_non_cis":
                float(np.nanmedian(pert_df["jaccard_pooled_vs_stratified"]))
                if pert_df["jaccard_pooled_vs_stratified"].notna().any() else None,
            "nonrobust_rule": f"tested TF perturbation with non-cis pooled-vs-stratified Jaccard < "
                              f"{NONROBUST_JACCARD}",
            "nonrobust_tf_perturbations": {
                str(r.perturbed_gene): {"jaccard": float(r.jaccard_pooled_vs_stratified),
                                        "n_positive_primary_stratified": int(r.n_positive_primary_stratified),
                                        "n_positive_primary_pooled": int(r.n_positive_primary_pooled)}
                for r in nonrobust.itertuples()},
        }
        ctrl_comp = lab["stratum"].iloc[ctrl_rows].value_counts(normalize=True)
        extra["control_strata_fraction"] = {str(k): float(v) for k, v in sorted(ctrl_comp.items())}
    if multi:
        tv = pert_df.set_index("perturbed_gene")["strata_tvd_vs_ctrl"]
        extra["strata_composition"] = {
            "statistic": "total-variation distance between the stratum composition of the "
                         "perturbed cells and of the control cells",
            "median_tvd": float(tv.median()), "max_tvd": float(tv.max()),
            "max_tvd_perturbation": str(tv.idxmax())}
        extra["strata_cell_counts"] = {
            "control": {str(k): int(v) for k, v in lab["stratum"].iloc[ctrl_rows].value_counts().items()},
            "all_labelled": {str(k): int(v) for k, v in lab["stratum"].value_counts().items()}}
        if cid == "shifrut":
            fs = pert_df["frac_stim_pert"]
            extra["strata_composition"]["frac_stim_ctrl"] = float(pert_df["frac_stim_ctrl"].iloc[0])
            extra["strata_composition"]["frac_stim_pert_range"] = [float(fs.min()), float(fs.max())]

    meta = _cohort_meta(spec, L, E, scale, genes, lab, ctrl_rows, pert_df, de, t0)
    meta["gene_mapping"] = map_rep
    meta.update(extra)
    meta["cis_exclusion"] = cis_summary(de, mapper)
    raw_name_of = {g: str(L.raw_symbols[c]) for g, c in zip(genes, keep_cols)}
    meta["self_pairs"] = self_pair_check(de, mapper, dict(zip(genes, gene_ens)), raw_name_of, aliases_of)
    if de_other is not None:
        meta["self_pairs_sensitivity_de"] = self_pair_check(de_other, mapper, dict(zip(genes, gene_ens)),
                                                            raw_name_of, aliases_of)
    meta["selfcheck"] = selfcheck(E, keep_cols, gene_index, lab, ctrl_rows, de_pooled, L.raw_counts_probe)
    if stratified:
        meta["selfcheck_stratified"] = selfcheck_stratified(E, keep_cols, gene_index, lab, ctrl_rows,
                                                            de_strat, strata)
    C.write_json(meta, out_dir / "meta.json")
    log(f"{cid}: done in {time.time() - t0:.0f}s; {len(de)} DE rows; "
        f"{int(de.positive_incl_cis.sum())} positives incl. cis ({int(de.positive_primary.sum())} non-cis); "
        f"peak RSS {peak_rss_gb():.1f} GB")
    return meta


def _csr_logmatrix(E: ExprMatrix) -> sp.csr_matrix:
    return sp.csr_matrix((E.logv, E.indices, E.indptr), shape=(len(E.indptr) - 1, E.n_genes))


def selfcheck(E: ExprMatrix, keep_cols, gene_index, lab, ctrl_rows, de, probe) -> dict:
    """Verify the vectorised pooled statistics against reference implementations.

    * Welch t / p against ``scipy.stats.ttest_ind(equal_var=False)`` on dense
      vectors for up to 3 perturbations x 300 tested targets (seeded draw);
    * BH against ``statsmodels`` ``fdr_bh`` for the same perturbations;
    * log1p(CP10k) against scanpy ``normalize_total`` + ``log1p`` on a probe of
      raw-count cells (raw-count cohorts only).
    """
    from statsmodels.stats.multitest import multipletests
    rng = np.random.default_rng(P["random_seed"])
    M = _csr_logmatrix(E)
    out = {"test": "pooled Welch", "max_rel_t_diff": 0.0, "max_abs_log10p_diff": 0.0,
           "max_abs_padj_diff": 0.0, "n_checked": 0}
    perts = sorted(de["perturbed_gene"].astype(str).unique())[:3]
    pg_arr = lab["pert_gene"].to_numpy()
    for pg in perts:
        sub = de[de["perturbed_gene"].astype(str) == pg]
        k = min(300, len(sub))
        pick = sub.iloc[np.sort(rng.choice(len(sub), size=k, replace=False))]
        cols = keep_cols[[gene_index[g] for g in pick["target"].astype(str)]]
        r1 = np.flatnonzero(pg_arr == pg)
        a = M[r1][:, cols].toarray().astype(np.float64)
        b = M[ctrl_rows][:, cols].toarray().astype(np.float64)
        tt = stats.ttest_ind(a, b, axis=0, equal_var=False)
        ok = np.isfinite(tt.statistic)
        tm = pick["t"].to_numpy().astype(np.float64)[ok]      # stored as float32
        out["max_rel_t_diff"] = max(out["max_rel_t_diff"], float(np.max(
            np.abs(tt.statistic[ok] - tm) / np.maximum(1.0, np.abs(tm)))))
        lp_ref = np.log10(np.maximum(tt.pvalue[ok], 1e-300))
        lp_me = np.log10(np.maximum(pick["pval"].to_numpy()[ok], 1e-300))
        out["max_abs_log10p_diff"] = max(out["max_abs_log10p_diff"], float(np.max(np.abs(lp_ref - lp_me))))
        ref = multipletests(sub["pval"].to_numpy(), method="fdr_bh")[1]
        out["max_abs_padj_diff"] = max(out["max_abs_padj_diff"], float(np.max(np.abs(
            ref - sub["padj"].to_numpy()))))
        out["n_checked"] += int(ok.sum())
    if probe is not None:
        import scanpy as sc
        rows, Xraw = probe
        a = ad.AnnData(Xraw.astype(np.float32).copy())
        sc.pp.normalize_total(a, target_sum=TARGET_SUM)
        sc.pp.log1p(a)
        mine = M[rows].toarray()
        ref = a.X.toarray() if sp.issparse(a.X) else np.asarray(a.X)
        out["max_abs_lognorm_diff_vs_scanpy"] = float(np.max(np.abs(mine - ref)))
    out["passed"] = bool(out["max_rel_t_diff"] < 1e-5 and out["max_abs_log10p_diff"] < 1e-3
                         and out["max_abs_padj_diff"] < 1e-9
                         and out.get("max_abs_lognorm_diff_vs_scanpy", 0.0) < 1e-4)
    log(f"selfcheck: {out}")
    return out


def ols_cellmeans_contrast(y: np.ndarray, stratum: np.ndarray, group: np.ndarray):
    """Reference implementation: OLS with one mean per (stratum, group) cell.

    Returns (t, df) of the contrast sum_s w_s (mu_s1 - mu_s0) / sum_s w_s,
    w_s = n1_s n0_s / (n1_s + n0_s), with the residual variance of the full
    cell-means model. y is cells x genes.
    """
    cells = sorted(set(zip(stratum.tolist(), group.tolist())))
    X = np.zeros((len(y), len(cells)))
    for j, (s, g) in enumerate(cells):
        X[(stratum == s) & (group == g), j] = 1.0
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    df = len(y) - len(cells)
    sigma2 = (resid ** 2).sum(0) / df
    c = np.zeros(len(cells))
    for s in sorted(set(stratum.tolist())):
        n1 = int(((stratum == s) & (group == 1)).sum())
        n0 = int(((stratum == s) & (group == 0)).sum())
        w = n1 * n0 / (n1 + n0)
        c[cells.index((s, 1))] += w
        c[cells.index((s, 0))] -= w
    c /= c[c > 0].sum()
    xtx_inv = np.linalg.inv(X.T @ X)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (c @ beta) / np.sqrt(sigma2 * (c @ xtx_inv @ c))
    return t, df


def selfcheck_stratified(E, keep_cols, gene_index, lab, ctrl_rows, sde, strata) -> dict:
    """Stratified t against an explicit OLS cell-means fit (2 perturbations x 100 targets)."""
    rng = np.random.default_rng(P["random_seed"] + 1)
    M = _csr_logmatrix(E)
    pg_arr = lab["pert_gene"].to_numpy()
    out = {"test": "stratified (vs. OLS cell-means contrast)", "max_rel_t_diff": 0.0,
           "max_abs_padj_diff": 0.0, "n_checked": 0}
    from statsmodels.stats.multitest import multipletests
    for pg in sorted(sde["perturbed_gene"].astype(str).unique())[:2]:
        sub = sde[sde["perturbed_gene"].astype(str) == pg]
        pick = sub.iloc[np.sort(rng.choice(len(sub), size=min(100, len(sub)), replace=False))]
        cols = keep_cols[[gene_index[g] for g in pick["target"].astype(str)]]
        r1 = np.flatnonzero(pg_arr == pg)
        used = [s for s in np.unique(strata[r1])
                if (strata[r1] == s).sum() >= STRATUM_MIN_CELLS and (strata[ctrl_rows] == s).sum() >= STRATUM_MIN_CELLS]
        r1 = r1[np.isin(strata[r1], used)]
        r0 = ctrl_rows[np.isin(strata[ctrl_rows], used)]
        rows = np.concatenate([r1, r0])
        y = M[rows][:, cols].toarray().astype(np.float64)
        grp = np.concatenate([np.ones(len(r1), int), np.zeros(len(r0), int)])
        t, df = ols_cellmeans_contrast(y, strata[rows], grp)
        ok = np.isfinite(t)
        tm = pick["t"].to_numpy().astype(np.float64)[ok]
        out["max_rel_t_diff"] = max(out["max_rel_t_diff"], float(np.max(
            np.abs(t[ok] - tm) / np.maximum(1.0, np.abs(tm)))))
        ref = multipletests(sub["pval"].to_numpy(), method="fdr_bh")[1]
        out["max_abs_padj_diff"] = max(out["max_abs_padj_diff"], float(np.max(np.abs(ref - sub["padj"].to_numpy()))))
        out["n_checked"] += int(ok.sum())
    out["passed"] = bool(out["max_rel_t_diff"] < 1e-4 and out["max_abs_padj_diff"] < 1e-9)
    log(f"selfcheck_stratified: {out}")
    return out


class _ColumnView:
    """Wrap an ExprMatrix so that group statistics are restricted to kept columns."""

    def __init__(self, E: ExprMatrix, keep_cols: np.ndarray):
        self.E, self.keep = E, keep_cols

    def group_stats(self, rows):
        g = self.E.group_stats(rows)
        return {k: (v[self.keep] if isinstance(v, np.ndarray) else v) for k, v in g.items()}


def _ontarget_summary(pert_df: pd.DataFrame, direction: str) -> dict:
    t = pert_df[pert_df["tested"]]
    ev = t[t["ontarget_in_matrix"].astype(bool)]
    return {"expected_direction": direction,
            "n_tested": int(len(t)), "n_evaluable": int(len(ev)),
            "n_pass": int(ev["ontarget_pass"].astype(bool).sum()),
            "median_log2fc": float(ev["ontarget_log2fc"].median()) if len(ev) else None,
            "n_sign_consistent": int(((ev["ontarget_log2fc"] < 0) if direction == "down"
                                      else (ev["ontarget_log2fc"] > 0)).sum()),
            "not_evaluable": t.loc[~t["ontarget_in_matrix"].astype(bool), "perturbed_gene"].tolist()}


def _counts_meta(pert_df: pd.DataFrame, de: pd.DataFrame) -> dict:
    tested = pert_df[pert_df["tested"]]
    tf_t = tested[tested["is_tf"]]
    q99 = tested["q99_abs_log2fc"].dropna()
    return {
        "n_perturbations_labelled": int(len(pert_df)),
        "n_perturbations_tested": int(len(tested)),
        "n_perturbed_tfs_tested": int(tested["is_tf"].sum()),
        "perturbations_not_tested": pert_df.loc[~pert_df["tested"], "perturbed_gene"].tolist(),
        "n_de_rows": int(len(de)),
        "n_positive_edges_incl_cis": int(de["positive_incl_cis"].sum()),
        "n_positive_edges_primary": int(de["positive_primary"].sum()),
        "n_positive_tf_edges_incl_cis": int(de.loc[de["is_tf"], "positive_incl_cis"].sum()),
        "n_positive_tf_edges_primary": int(de.loc[de["is_tf"], "positive_primary"].sum()),
        "n_fdr_significant_any_lfc": int((de["padj"] < P["de_fdr"]).sum()),
        "q99_abs_log2fc_per_perturbation_range": [float(q99.min()), float(q99.max())] if len(q99) else None,
        "median_targets_tested": float(tested["n_targets_tested"].median()) if len(tested) else None,
        "median_positives_primary_per_perturbation":
            float(tested["n_positive_primary"].median()) if len(tested) else None,
        "median_positives_primary_per_tf_perturbation":
            float(tf_t["n_positive_primary"].median()) if len(tf_t) else None,
    }


def _cohort_meta(spec, L, E, scale, genes, lab, ctrl_rows, pert_df, de, t0) -> dict:
    m = {
        "cohort_id": spec.cohort_id, "study_id": spec.study_id,
        "cell_type_group": spec.cell_type_group,
        "full_name": spec.full_name, "accession": spec.accession,
        "cell_type": spec.cell_type, "modality": spec.modality,
        "input": file_record(spec.path) | L.extra_inputs,
        "gene_identity_inputs": hgnc_provenance(),
        "labels": L.label_notes,
        "expression": E.scale_info,
        "n_genes_after_symbol_dedup": int(len(genes)),
        "n_cells_used": int(len(lab)), "n_control_cells": int(len(ctrl_rows)),
        "n_perturbed_cells": int(len(lab) - len(ctrl_rows)),
    }
    m.update(_counts_meta(pert_df, de))
    m.update({
        "ontarget": _ontarget_summary(pert_df, spec.expected_direction),
        "parameters": {k: P[k] for k in ("de_min_cells", "de_fdr", "de_min_abs_log2fc",
                                         "de_min_target_frac_expressed", "random_seed",
                                         "cis_exclusion", "symbol_mapping", "replicate_stratified_de")}
        | {"log2fc_pseudocount_cp10k": LOG2FC_PSEUDOCOUNT_CP10K, "nonrobust_jaccard": NONROBUST_JACCARD,
           "test": "Welch t (two-sided)" if PRIMARY_DE.get(spec.cohort_id, "pooled") == "pooled"
           else "stratified two-sample t (two-sided)",
           "multiple_testing": "BH within perturbation", "ontarget_p": ONTARGET_P,
           "control_subsampling": "none (sufficient statistics; all controls used)"},
        "wall_time_s": round(time.time() - t0, 1), "peak_rss_gb_process": round(peak_rss_gb(), 2),
    })
    return m


# ----------------------------------------------------------------------------
# Replogle 2022 (pseudobulk)
# ----------------------------------------------------------------------------

def ensure_replogle_files(download: bool = True) -> dict:
    """Return md5-verified file records; download missing files (md5-checked) if allowed.

    The download time comes from ``download_log.json`` in ``C.REPLOGLE_DIR``
    (written when this function downloads a file); for files obtained otherwise
    the file modification time is reported.
    """
    C.ensure_dir(C.REPLOGLE_DIR)
    log_path = C.REPLOGLE_DIR / REPLOGLE_DOWNLOAD_LOG
    dl = json.loads(log_path.read_text()) if log_path.exists() else {}
    recs = {}
    for k, f in REPLOGLE_FILES.items():
        p = C.REPLOGLE_DIR / f["name"]
        if not p.exists():
            if not download:
                raise FileNotFoundError(f"{p} missing; rerun without --no-download or fetch {f['url']}")
            tmp = p.with_name(p.name + ".part")
            log(f"replogle: downloading {f['url']} -> {p}")
            with urllib.request.urlopen(f["url"]) as r, open(tmp, "wb") as fh:
                shutil.copyfileobj(r, fh, length=1 << 24)
            md5 = file_md5(tmp)
            if md5 != f["md5"]:
                tmp.unlink()
                raise ValueError(f"md5 mismatch for downloaded {f['name']}: {md5} != {f['md5']}")
            tmp.rename(p)
            dl[f["name"]] = {"url": f["url"], "md5": md5, "size_bytes": p.stat().st_size,
                             "downloaded_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")}
            C.write_json(dl, log_path)
        rec = file_record(p)
        if rec["md5"] != f["md5"]:
            raise ValueError(f"md5 mismatch for {p}: {rec['md5']} != {f['md5']}")
        rec.update({kk: f[kk] for kk in ("url", "figshare_file_id")})
        if f["name"] in dl:
            rec["downloaded"], rec["downloaded_source"] = dl[f["name"]]["downloaded_utc"], REPLOGLE_DOWNLOAD_LOG
        else:
            rec["downloaded"] = dt.datetime.fromtimestamp(p.stat().st_mtime, dt.timezone.utc).isoformat(
                timespec="seconds")
            rec["downloaded_source"] = "file modification time"
        recs[k] = rec
    return recs


def _parse_replogle_index(idx: pd.Index, var: pd.DataFrame) -> pd.DataFrame:
    parts = idx.to_series().str.split("_")
    df = pd.DataFrame({"row_id": idx, "sym_raw": parts.str[1].to_numpy(),
                       "ens": clean_ensembl(parts.str[-1].to_numpy())}, index=idx)
    ens2sym = dict(zip(var.index.astype(str), var["gene_name"].astype(str)))
    df["name"] = [ens2sym.get(e, s) for e, s in zip(df["ens"], df["sym_raw"])]
    df["is_nt"] = df["sym_raw"].eq("non-targeting")
    return df


def build_replogle(spec: CohortSpec, out_dir: Path, download: bool = True) -> dict:
    t0 = time.time()
    tfs = C.lambert_tfs()
    mapper = gene_mapper()
    recs = ensure_replogle_files(download)
    pz, pr = Path(recs["normalized_bulk"]["path"]), Path(recs["raw_bulk"]["path"])
    az, ar = ad.read_h5ad(pz), ad.read_h5ad(pr)
    assert (az.obs_names == ar.obs_names).all() and (az.var_names == ar.var_names).all()
    Z = np.asarray(az.X, dtype=np.float64)
    R = np.asarray(ar.X, dtype=np.float64)
    obs = az.obs.copy()
    info = _parse_replogle_index(obs.index, az.var)
    ncell = obs["num_cells_filtered"].to_numpy(dtype=float)

    authors_mean = az.var["mean"].to_numpy(dtype=float)
    Mc = mapper.map_columns(az.var["gene_name"].astype(str), az.var.index.astype(str), authors_mean)
    Mc_legacy = mapper.map_columns(az.var["gene_name"].astype(str), az.var.index.astype(str), authors_mean,
                                   legacy=True)
    keep_cols, genes, gene_ens = Mc["keep_cols"], Mc["genes"], Mc["gene_ens"]
    gene_index = {g: i for i, g in enumerate(genes)}
    ens_to_col = {e: i for i, e in enumerate(gene_ens) if e}
    map_rep = mapping_report(mapper, Mc, Mc_legacy, list(az.var["gene_name"].astype(str)), authors_mean,
                             universe_gene_sets())
    map_rep["gene_id_source"] = "var index (Ensembl gene id)"
    Z, R = Z[:, keep_cols], R[:, keep_cols]
    authors_mean = authors_mean[keep_cols]

    ctrl = (obs["core_control"].astype(bool).to_numpy() & np.isfinite(ncell))
    n_ctrl = int(ncell[ctrl].sum())
    ctrl_raw = (R[ctrl] * ncell[ctrl, None]).sum(0) / ncell[ctrl].sum()   # mean raw UMI per control cell
    ctrl_cp10k = ctrl_raw / ctrl_raw.sum() * TARGET_SUM
    # Z calibration on the core non-targeting pseudobulks (they define the z-scale,
    # so this check is optimistic)
    zc = Z[ctrl] * np.sqrt(ncell[ctrl, None])
    calib = {"n_rows": int(ctrl.sum()), "sd_Z": float(zc.std()),
             "frac_absZ_gt_1.96": float((np.abs(zc) > 1.96).mean())}
    # second check, not involving the controls that define the z-scale: targeting
    # perturbations without a detectable transcriptional phenotype according to the
    # authors' energy-distance test (p > 0.5); Z should then be ~N(0, 1) up to
    # residual true effects (so this bounds the miscalibration from above)
    nullish = (np.isfinite(ncell) & ~obs["core_control"].astype(bool).to_numpy()
               & ~obs.index.str.contains("non-targeting")
               & (obs["energy_test_p_value"].to_numpy() > 0.5))
    zn = Z[nullish] * np.sqrt(ncell[nullish, None])
    calib_null = {"selection": "targeting rows with authors' energy_test_p_value > 0.5",
                  "n_rows": int(nullish.sum()), "sd_Z": float(zn.std()),
                  "median_absZ_over_0.6745": float(np.median(np.abs(zn)) / 0.6745),
                  "frac_absZ_gt_1.96": float((np.abs(zn) > 1.96).mean())}

    # perturbation labels: the authors' gene name in the row id, approved name first
    # (PRESPECIFIED['symbol_mapping']), the row's Ensembl id otherwise
    info["pert_gene"] = [mapper.resolve(n, e)[0] for n, e in zip(info["sym_raw"], info["ens"])]
    legacy_lab = [mapper.resolve_ensembl_first(n, e)[0] for n, e in zip(info["name"], info["ens"])]
    nt = info["is_nt"].to_numpy()
    map_rep["labels_changed_vs_ensembl_first"] = sorted(
        {(r, o, n) for r, o, n, x in zip(info["sym_raw"], legacy_lab, info["pert_gene"], nt) if not x and o != n})
    usable = ~info["is_nt"].to_numpy() & np.isfinite(ncell)
    excluded_nan = int((~info["is_nt"].to_numpy() & ~np.isfinite(ncell)).sum())
    de_parts, prow, aliases_of, row_ens_of = [], [], {}, {}
    for pg, grp in info[usable].groupby("pert_gene", sort=True):
        ii = np.flatnonzero(obs.index.isin(grp.index))
        w = ncell[ii]
        n = float(w.sum())
        zbar = (Z[ii] * w[:, None]).sum(0) / n            # pooled mean z (cell-weighted)
        rbar = (R[ii] * w[:, None]).sum(0) / n            # pooled mean raw UMI
        cp = rbar / rbar.sum() * TARGET_SUM
        row_ens_of[pg] = {e for e in grp["ens"] if e}
        aliases_of[pg] = label_aliases(mapper, grp["sym_raw"].unique(), sorted(row_ens_of[pg])) - {pg}
        own = _own_columns(pg, gene_index, ens_to_col, mapper, aliases_of[pg], row_ens_of[pg])
        gi = gene_index.get(pg, own[0] if own else None)
        rec = {"perturbed_gene": pg, "perturbed_gene_raw": ";".join(sorted(grp["sym_raw"].unique())),
               "is_tf": pg in tfs, "n_pert_cells": int(n), "n_ctrl_cells": n_ctrl,
               "n_guides": np.nan, "guides": "",
               "n_pseudobulk_rows": int(len(ii)), "pseudobulk_rows": ";".join(obs.index[ii]),
               "tested": n >= P["de_min_cells"], "primary_de": "pseudobulk_z",
               "n_own_columns_excluded": len(own)}
        zt = zbar * np.sqrt(n)
        if gi is None:
            rec.update(_ontarget_from(np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, "down", False))
        else:
            lfc_t = float(log2fc(cp[gi], ctrl_cp10k[gi]))
            p_t = float(2 * stats.norm.sf(abs(zt[gi])))
            rec.update(_ontarget_from(lfc_t, zt[gi], p_t, np.nan, np.nan, cp[gi], ctrl_cp10k[gi], "down"))
        fe = obs["fold_expr"].to_numpy()[ii]
        rec["authors_fold_expr_cellweighted"] = float(np.nansum(fe * w) / w[np.isfinite(fe)].sum()) \
            if np.isfinite(fe).any() else np.nan
        if rec["tested"]:
            cols = np.setdiff1d(np.arange(len(genes)), np.array(own, dtype=int))
            p = 2 * stats.norm.sf(np.abs(zt[cols]))
            padj = bh(p)
            lfc = log2fc(cp[cols], ctrl_cp10k[cols])
            pos = (padj < P["de_fdr"]) & (np.abs(lfc) > P["de_min_abs_log2fc"])
            df = pd.DataFrame({
                "perturbed_gene": pg, "target": genes[cols], "log2fc": lfc.astype(np.float32),
                "t": zt[cols].astype(np.float32), "pval": p, "padj": padj, "positive_incl_cis": pos,
                "n_pert_cells": np.int32(n), "n_ctrl_cells": np.int32(n_ctrl),
                "mean_pert": cp[cols].astype(np.float32), "mean_ctrl": ctrl_cp10k[cols].astype(np.float32),
                "frac_pert": np.float32(np.nan), "frac_ctrl": np.float32(np.nan)})
            de_parts.append(df)
        prow.append(rec)
    pert_df = pd.DataFrame(prow)
    log(f"replogle: {len(pert_df)} perturbed genes; {int(pert_df.tested.sum())} tested")
    de = finalize_de(de_parts, tfs, mapper)
    del de_parts
    gc.collect()
    pert_df = pert_df.merge(per_perturbation_counts(de), on="perturbed_gene", how="left")
    C.ensure_dir(out_dir)
    de.to_parquet(out_dir / "de.parquet", index=False, compression="zstd")
    pert_df.to_csv(out_dir / "perturbations.csv", index=False)
    cmin = float(ctrl_raw.min())
    meta = {
        "cohort_id": spec.cohort_id, "study_id": spec.study_id, "cell_type_group": spec.cell_type_group,
        "full_name": spec.full_name, "accession": spec.accession,
        "cell_type": spec.cell_type, "modality": spec.modality,
        "input": recs,
        "figshare_article": "https://plus.figshare.com/articles/dataset/20029387 "
                            "(API: https://api.figshare.com/v2/articles/20029387/files)",
        "downloaded": {k: {"when": r["downloaded"], "source": r["downloaded_source"]} for k, r in recs.items()},
        "gene_identity_inputs": hgnc_provenance(),
        "primary_de": "pseudobulk_z", "de_design": DE_DESIGN["replogle"],
        "labels": {"control_definition": "rows with core_control == True (non-targeting guides; "
                                         "cells pooled with num_cells_filtered weights)",
                   "excluded": f"{int((info['is_nt'] & ~obs['core_control'].astype(bool)).sum())} "
                               f"non-core non-targeting rows; {excluded_nan} perturbation rows "
                               "without a cell count",
                   "row_collapse": "pseudobulk rows of the same gene (alternative promoters/"
                                   "transcripts; each row pools several guides) pooled with "
                                   "cell-count weights; n_pseudobulk_rows counts rows, the number "
                                   "of guides is not available in these files"},
        "expression": {"scale": "pseudobulk",
                       "test": "approximate z-test Z = mean_z * sqrt(n_cells) on the z-normalised "
                               "pseudobulk (z relative to control cells of the same GEM group); "
                               "two-sided normal p; BH within perturbation",
                       "log2fc": "raw pseudobulk (mean UMI/cell) scaled to 1e4 per profile; "
                                 "log2((pert + 0.1) / (control + 0.1))",
                       "expression_filter": "authors' expressed-gene set (8,563 genes); per-cell "
                                            "detection unavailable",
                       "core_control_mean_umi_per_cell": {
                           "min": cmin, "median": float(np.median(ctrl_raw)),
                           "expected_detection_at_min_poisson": float(1 - np.exp(-cmin)),
                           "n_genes_below_0.1": int((ctrl_raw < 0.1).sum())},
                       "authors_var_mean_note": "var['mean'] is not the core-control mean: "
                                                "median ratio var['mean'] / core-control mean = %.3f"
                                                % float(np.median(authors_mean / ctrl_raw)),
                       "z_calibration_core_controls": calib,
                       "z_calibration_null_like_perturbations": calib_null},
        "gene_mapping": map_rep,
        "n_genes_after_symbol_dedup": int(len(genes)),
        "n_cells_used": int(ncell[usable].sum() + n_ctrl), "n_control_cells": n_ctrl,
        "n_perturbed_cells": int(ncell[usable].sum()),
    }
    meta.update(_counts_meta(pert_df, de))
    meta.update({
        "ontarget": _ontarget_summary(pert_df, spec.expected_direction),
        "cis_exclusion": cis_summary(de, mapper),
        "self_pairs": self_pair_check(de, mapper, dict(zip(genes, gene_ens)),
                                      {g: str(az.var["gene_name"].iloc[c]) for g, c in zip(genes, keep_cols)},
                                      aliases_of, row_ens_of),
        "parameters": {k: P[k] for k in ("de_min_cells", "de_fdr", "de_min_abs_log2fc",
                                         "de_min_target_frac_expressed", "random_seed", "cis_exclusion",
                                         "symbol_mapping")}
        | {"log2fc_pseudocount_cp10k": LOG2FC_PSEUDOCOUNT_CP10K, "ontarget_p": ONTARGET_P},
        "wall_time_s": round(time.time() - t0, 1), "peak_rss_gb_process": round(peak_rss_gb(), 2),
    })
    C.write_json(meta, out_dir / "meta.json")
    log(f"replogle: done; {len(de)} DE rows; {int(de.positive_incl_cis.sum())} positives incl. cis "
        f"({int(de.positive_primary.sum())} non-cis)")
    return meta


# ----------------------------------------------------------------------------
# Manuscript tables
# ----------------------------------------------------------------------------

def _flag_text(spec: CohortSpec, m: dict) -> str:
    out = []
    if spec.flag == "dixit":
        q = m.get("q99_abs_log2fc_per_perturbation_range") or [np.nan, np.nan]
        out.append("essentially no positives at |log2FC| > 0.25 under Cas9 knockout "
                   f"({m['n_fdr_significant_any_lfc']} pairs at FDR < 0.05 without the fold-change "
                   f"threshold; per-perturbation 99th percentile of |log2FC| {q[0]:.2f}-{q[1]:.2f}); "
                   "kept but flagged")
    nr = (m.get("stratified_test") or {}).get("nonrobust_tf_perturbations") or {}
    if nr:
        out.append("TF perturbations with non-robust DE (non-cis pooled-vs-stratified Jaccard < "
                   f"{NONROBUST_JACCARD}): " + ", ".join(
                       f"{k} ({v['jaccard']:.2f}; {v['n_positive_primary_stratified']} stratified vs. "
                       f"{v['n_positive_primary_pooled']} pooled positives)" for k, v in sorted(nr.items())))
    return "; ".join(out)


def build_table(order, out_root: Path, tables_dir: Path) -> pd.DataFrame:
    rows = []
    for cid in order:
        mp = out_root / cid / "meta.json"
        if not mp.exists():
            continue
        m = json.loads(mp.read_text())
        spec = COHORTS[cid]
        o = m["ontarget"]
        verb = {"CRISPRa": "induction", "CRISPRi": "knockdown",
                "CRISPR KO": "mRNA reduction"}[m["modality"]]
        summ = (f"{o['n_pass']}/{o['n_evaluable']} perturbations with significant on-target "
                f"{verb} (median log2FC {o['median_log2fc']:.2f})"
                if o["n_evaluable"] else "not evaluable")
        st = m.get("stratified_test") or {}
        sens = None
        if st:
            sens = "pooled" if st["role"] == "primary" else "stratified"
        rows.append({
            "cohort_id": cid, "study_id": spec.study_id,
            "independent_unit": {"study_id": spec.study_id, "cohort_id": cid}[P["cohort_cluster_unit"]],
            "cell_type_group": spec.cell_type_group,
            "full_name": m["full_name"], "accession_source": m["accession"],
            "cell_type": m["cell_type"], "modality": m["modality"],
            "primary_de": m.get("primary_de", "pooled"),
            "de_design": m.get("de_design", DE_DESIGN[cid]),
            "n_cells": m["n_cells_used"], "n_controls": m["n_control_cells"],
            "n_genes": m["n_genes_after_symbol_dedup"],
            "n_perturbations_tested": m["n_perturbations_tested"],
            "n_perturbed_tfs": m["n_perturbed_tfs_tested"],
            "median_targets_tested": m["median_targets_tested"],
            "median_positives_per_perturbation": m["median_positives_primary_per_perturbation"],
            "median_positives_per_tf_perturbation": m["median_positives_primary_per_tf_perturbation"],
            "n_positive_edges_primary": m["n_positive_edges_primary"],
            "n_positive_tf_edges_primary": m["n_positive_tf_edges_primary"],
            "n_positive_tf_edges_incl_cis": m["n_positive_tf_edges_incl_cis"],
            "n_cis_positive_removed": m["cis_exclusion"]["n_cis_positive_removed"],
            "n_cis_positive_tf_removed": m["cis_exclusion"]["n_cis_positive_removed_tf"],
            "ontarget_n_pass": o["n_pass"], "ontarget_n_evaluable": o["n_evaluable"],
            "ontarget_median_log2fc": o["median_log2fc"],
            "ontarget_summary": summ,
            "sensitivity_de": sens,
            "n_positive_primary_sensitivity_de": (st.get(f"n_positive_primary_{sens}") if sens else None),
            "jaccard_primary_vs_sensitivity_non_cis": st.get("jaccard_overall_non_cis") if sens else None,
            "n_nonrobust_tf_perturbations": (len(st.get("nonrobust_tf_perturbations") or {})
                                             if sens else None),
            "nonrobust_tf_perturbations": (";".join(sorted(st.get("nonrobust_tf_perturbations") or {}))
                                           if sens else None),
            "flag": _flag_text(spec, m),
            "design_note": DESIGN_NOTES.get(cid, ""),
            "ontarget_note": ONTARGET_NOTES[m["modality"]],
        })
    df = pd.DataFrame(rows)
    if len(df):
        df.insert(3, "n_cohorts_in_unit", df.groupby("independent_unit")["cohort_id"].transform("size"))
    C.ensure_dir(tables_dir)
    df.to_csv(tables_dir / "cohort_table.csv", index=False)
    # per-perturbation summary (supplementary)
    keep = ["perturbed_gene", "perturbed_gene_raw", "is_tf", "primary_de", "n_pert_cells",
            "n_ctrl_cells", "n_guides", "n_pseudobulk_rows", "tested", "n_targets_tested",
            "n_positive_primary", "n_positive_primary_up", "n_positive_primary_down",
            "n_positive_incl_cis", "n_cis_tested", "n_cis_positive_removed",
            "n_fdr_significant_any_lfc", "q99_abs_log2fc",
            "n_positive_primary_stratified", "n_positive_primary_pooled", "n_positive_primary_both_tests",
            "jaccard_pooled_vs_stratified", "de_nonrobust_tf",
            "strata_tvd_vs_ctrl", "ontarget_in_matrix", "ontarget_log2fc", "ontarget_pval",
            "ontarget_pass"]
    parts = []
    for cid in order:
        pp = out_root / cid / "perturbations.csv"
        if pp.exists():
            d = pd.read_csv(pp)
            d.insert(0, "cohort_id", cid)
            d.insert(1, "study_id", COHORTS[cid].study_id)
            parts.append(d.reindex(columns=["cohort_id", "study_id"] + keep))
    if parts:
        pd.concat(parts, ignore_index=True).to_csv(tables_dir / "cohort_perturbations.csv", index=False)
    return df


def _universes() -> dict:
    """Universe TF and gene sets (tissue panels and the global universe), with provenance."""
    out = {}
    for t in C.TISSUES:
        d = UNIVERSE_DIR / t
        if (d / "panel.tsv").exists() and (d / "tfs.txt").exists():
            out[t] = {"tfs": set(Path(d / "tfs.txt").read_text().split()),
                      "genes": set(pd.read_csv(d / "panel.tsv", sep="\t")["gene"].astype(str)),
                      "files": [d / "tfs.txt", d / "panel.tsv"]}
    g = GLOBAL_UNIVERSE_DIR
    if (g / "tfs.txt").exists() and (g / "genes.txt").exists():
        out["global"] = {"tfs": set(Path(g / "tfs.txt").read_text().split()),
                         "genes": set(Path(g / "genes.txt").read_text().split()),
                         "files": [g / "tfs.txt", g / "genes.txt"]}
    return out


def build_universe_overlap(order, out_root: Path, tables_dir: Path) -> pd.DataFrame:
    """Descriptive coverage: tested perturbed TFs and their evaluable pairs per universe.

    Evaluable pairs = tested (TF, target) with target in the universe gene set,
    target != TF and not cis_same_band (cis pairs are excluded from positives
    and negatives). For cohorts with a sensitivity DE (``de_pooled.parquet``;
    same tested pairs) its non-cis positives are reported too, with the TF
    perturbations flagged ``de_nonrobust_tf``. No outcome is used to define
    anything.
    """
    U = _universes()
    rows = []
    cols = ["perturbed_gene", "target", "is_tf", "positive_incl_cis", "cis_same_band", "positive_primary"]
    for cid in order:
        f = out_root / cid / "de.parquet"
        if not f.exists():
            continue
        de = pd.read_parquet(f, columns=cols, filters=[("is_tf", "==", True)])
        de = de[de["is_tf"]].reset_index(drop=True)
        fs = out_root / cid / "de_pooled.parquet"
        has_sens = fs.exists()
        if has_sens:
            ds = pd.read_parquet(fs, columns=["perturbed_gene", "target", "is_tf", "positive_primary"],
                                 filters=[("is_tf", "==", True)])
            ds = ds[ds["is_tf"]]
            key = lambda d: d["perturbed_gene"].astype(str) + "\t" + d["target"].astype(str)  # noqa: E731
            sens_pos = pd.Series(ds["positive_primary"].to_numpy(), index=key(ds).to_numpy())
            de["positive_primary_sensitivity_de"] = sens_pos.reindex(key(de).to_numpy()).to_numpy()
            assert de["positive_primary_sensitivity_de"].notna().all(), f"{cid}: DE tests differ in pairs"
            de["positive_primary_sensitivity_de"] = de["positive_primary_sensitivity_de"].astype(bool)
            pp = pd.read_csv(out_root / cid / "perturbations.csv")
            nonrobust = set(pp.loc[pp["de_nonrobust_tf"].astype(bool), "perturbed_gene"].astype(str))
            del ds, sens_pos
        # membership via categorical codes (np.isin on object arrays is very slow)
        pgc, tgc = de["perturbed_gene"].astype("category"), de["target"].astype("category")
        pc, tc = pgc.cat.codes.to_numpy(), tgc.cat.codes.to_numpy()
        pcat, tcat = [str(x) for x in pgc.cat.categories], [str(x) for x in tgc.cat.categories]
        tidx = {t: i for i, t in enumerate(tcat)}
        not_self = tc != np.array([tidx.get(p, -1) for p in pcat], dtype=np.int64)[pc]
        tested_tfs = {pcat[i] for i in np.unique(pc)}
        for uname, u in U.items():
            tf_in = sorted(tested_tfs & u["tfs"])
            pm = np.array([p in set(tf_in) for p in pcat], dtype=bool)
            tm = np.array([t in u["genes"] for t in tcat], dtype=bool)
            sel = pm[pc] & tm[tc] & not_self
            sub = de[sel]
            cis = sub["cis_same_band"].to_numpy()
            per_tf = sub[~cis].groupby(sub["perturbed_gene"].astype(str)[~cis], observed=True)[
                "positive_primary"].sum() if len(sub) else pd.Series(dtype=int)
            rows.append({
                "cohort_id": cid, "study_id": COHORTS[cid].study_id, "universe": uname,
                "n_universe_tfs": len(u["tfs"]), "n_universe_genes": len(u["genes"]),
                "n_perturbed_tfs_tested": len(tested_tfs),
                "n_perturbed_tfs_in_universe": len(tf_in),
                "n_pairs_tested_in_universe": int(len(sub)),
                "n_cis_pairs_excluded": int(cis.sum()),
                "n_pairs_evaluable": int((~cis).sum()),
                "n_positive_primary": int(sub["positive_primary"].sum()),
                "n_cis_positive_excluded": int((sub["positive_incl_cis"].to_numpy() & cis).sum()),
                "n_tfs_with_positive": int((per_tf > 0).sum()),
                "n_positive_primary_sensitivity_de": (
                    int(sub["positive_primary_sensitivity_de"].sum()) if has_sens else None),
                "nonrobust_tfs_in_universe": (";".join(sorted(nonrobust & set(tf_in))) if has_sens else None),
                "tfs_in_universe": ";".join(tf_in),
                "positives_per_tf": ";".join(f"{k}:{int(v)}" for k, v in per_tf.items()),
            })
        del de
        gc.collect()
    df = pd.DataFrame(rows)
    C.ensure_dir(tables_dir)
    df.to_csv(tables_dir / "cohort_universe_overlap.csv", index=False)
    prov = {"universes": {k: {"files": [{"path": str(p), "md5": file_md5(p),
                                         "mtime": dt.datetime.fromtimestamp(p.stat().st_mtime).isoformat(
                                             timespec="seconds")} for p in v["files"]],
                              "n_tfs": len(v["tfs"]), "n_genes": len(v["genes"])}
                          for k, v in U.items()},
            "table": str(tables_dir / "cohort_universe_overlap.csv"),
            "built": dt.datetime.now().isoformat(timespec="seconds")}
    C.write_json(prov, out_root / "universe_overlap.json")
    return df


# ----------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cohorts", nargs="+", default=list(COHORTS), choices=list(COHORTS))
    ap.add_argument("--table-only", action="store_true",
                    help="only rebuild the tables and the universe overlap from existing outputs")
    ap.add_argument("--out-root", type=Path, default=OUT_ROOT)
    ap.add_argument("--tables-dir", type=Path, default=None,
                    help="default: data/r2/tables for the default --out-root, else <out-root>/tables")
    ap.add_argument("--no-download", action="store_true",
                    help="fail instead of downloading missing Replogle files")
    args = ap.parse_args(argv)
    out_root = args.out_root.resolve()
    tables_dir = args.tables_dir or (C.TABLES if out_root == OUT_ROOT.resolve() else out_root / "tables")
    np.random.seed(P["random_seed"])
    if not args.table_only:
        for cid in args.cohorts:
            spec = COHORTS[cid]
            out = out_root / cid
            if cid == "replogle":
                build_replogle(spec, out, download=not args.no_download)
            else:
                build_singlecell(spec, out)
            gc.collect()
    df = build_table(list(COHORTS), out_root, tables_dir)
    ov = build_universe_overlap(list(COHORTS), out_root, tables_dir)
    with pd.option_context("display.width", 250, "display.max_columns", 40):
        print(df.drop(columns=["full_name", "accession_source", "cell_type", "ontarget_summary",
                               "design_note", "ontarget_note", "flag", "de_design"]).to_string(index=False))
        print(ov.drop(columns=["tfs_in_universe", "positives_per_tf"]).to_string(index=False))


if __name__ == "__main__":
    main()
