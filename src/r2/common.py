"""Shared paths, prespecified constants and helpers for the rebuilt (R2) analysis.

Every stage of the rebuilt pipeline imports this module so that gene-symbol
normalisation, the transcription-factor list, tissue gene panels and output
locations are defined in exactly one place.

Conventions
-----------
* Gene symbols are HGNC-approved symbols in upper case (``normalize_symbol``).
* An *edge* is an ordered pair ``(tf, target)`` with ``tf`` a Lambert et al.
  (2018) human transcription factor and ``tf != target``.
* Tables are written as parquet (pyarrow) unless they are small summary tables
  intended for the manuscript, which are CSV under ``data/r2/``.
* Every analysis choice that could be tuned on evaluation data is fixed here,
  *before* any held-out evaluation is run (see ``PRESPECIFIED``).
"""
from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------
# Locations
# ----------------------------------------------------------------------------
SP = Path(__file__).resolve().parents[2]                     # subproject root
BW = SP.parent                                               # biodyn-work root
EXT = Path(os.environ.get(
    "BIODYN_EXT", "/Volumes/Crucial X6/MacBook/biomechinterp/biodyn-work"))

OUT = SP / "outputs" / "r2"          # large/intermediate outputs (gitignored)
DATA = SP / "data" / "r2"            # small committed tables (inputs + results)
INPUTS = DATA / "inputs"             # committed public input files
TABLES = DATA / "tables"             # script-generated manuscript tables (CSV)

TISSUES = ("immune", "kidney", "lung")
PRIMARY_TISSUE = "immune"
MODELS = ("scgpt", "geneformer")

# Attention matrices (mean over layers and heads) extracted for the
# evaluation-bias study; scGPT matrices follow the var order of the processed
# h5ad, Geneformer matrices follow gene_names.txt.
EVAL_BIAS = EXT / "single_cell_mechinterp" / "outputs" / "eval_bias"
SCGPT_PROCESSED_H5AD = {
    "immune": EXT / "single_cell_mechinterp/outputs/atlas/immune/tabula_sapiens_immune_processed.h5ad",
    "kidney": EXT / "single_cell_mechinterp/outputs/atlas/kidney/tabula_sapiens_kidney_processed.h5ad",
    "lung": EXT / "single_cell_mechinterp/outputs/atlas/lung/tabula_sapiens_lung_processed.h5ad",
}


# Original Tabula Sapiens h5ad files (X = single log1p(CP10k) of decontX counts);
# the processed files above are log-normalised twice, so expression baselines
# read the source files.
TS_SOURCE_H5AD = {
    "immune": EXT / "single_cell_mechinterp/data/raw/tabula_sapiens_immune_subset_20000.h5ad",
    "kidney": EXT / "single_cell_mechinterp/data/raw/tabula_sapiens_kidney.h5ad",
    "lung": EXT / "single_cell_mechinterp/data/raw/tabula_sapiens_lung.h5ad",
}
# Geneformer V2-104M token dictionaries used at extraction time (pinned snapshot)
GF_DICT_DIR = Path(os.environ.get(
    "GENEFORMER_DICT_DIR",
    str(Path.home() / ".cache/huggingface/hub/models--ctheodoris--Geneformer/snapshots"
        / "05fcbeb8a27d49e0a7a4349152202ee2c1cbfd28/geneformer")))


def attention_paths(model: str, tissue: str) -> dict:
    d = EVAL_BIAS / model / tissue
    return {
        "scores": d / "attention_scores.npy",
        "counts": d / "attention_counts.npy",
        "gene_names": d / "gene_names.txt",   # Geneformer only
        "processed_h5ad": SCGPT_PROCESSED_H5AD[tissue],
    }


PERTURB_DIR = BW / "single_cell_mechinterp" / "data" / "perturb"
REPLOGLE_DIR = EXT / "external_data" / "replogle2022"      # downloads go to EXT
OMNIPATH_CACHE = Path.home() / ".cache" / "omnipathdb"
TRRUST_RAW = BW / "single_cell_mechinterp" / "external" / "networks" / "trrust_human.tsv"
ENCODE_TFBS = SP / "data" / "core" / "encode_tf_targets_5celllines_edges.tsv"
ENETS2 = SP / "data" / "core" / "enets2_seqspecific_edges.tsv"
LAMBERT_TF = INPUTS / "humantfs_lambert2018.csv"
HGNC_TABLE = INPUTS / "hgnc_complete_set.txt"
HGNC_FULL = INPUTS / "hgnc_complete_set_full.tsv"   # with ensembl_gene_id, location, locus_group

# ----------------------------------------------------------------------------
# Prespecified analysis constants (fixed before any held-out evaluation)
# ----------------------------------------------------------------------------
PRESPECIFIED = {
    # a method's *predicted* edge set: top-k targets per TF by method score
    "topk_per_tf": 50,
    "topk_per_tf_sensitivity": [20, 100],
    # perturbation differential expression (per perturbation vs. controls)
    "de_min_cells": 20,               # min perturbed cells to test a perturbation
    "de_fdr": 0.05,                   # Benjamini-Hochberg within perturbation
    "de_min_abs_log2fc": 0.25,
    "de_min_target_frac_expressed": 0.05,  # target expressed in >=5% of cells
    # held-out evaluation
    "eval_k": [100, 500, 1000, 5000],
    "n_null_draws": 1000,
    "n_bootstrap": 1000,
    "random_seed": 20261007,
    # power-based support floor for per-cohort replication (Q5/Q8):
    # minimum number of evaluable positives so that a true 2x enrichment is
    # detected with >=80% power at alpha=0.05 (computed analytically, not
    # from observed outcomes)
    "replication_true_enrichment": 2.0,
    "replication_power": 0.80,
    "replication_alpha": 0.05,
    "replication_endpoint_enrichment": 1.5,
    # Bayesian-FDR governance thresholds (label-free)
    "bfdr_strong": 0.05,
    "bfdr_moderate": 0.20,
    # ---- decisions logged 2026-10-07 after the data-layer verification and
    # ---- BEFORE any held-out evaluation (see REVISION_PLAN_R2.md section 7)
    # evidence observation model: 'prespecified_cc' (curated-conditional) is
    # primary; 'prespecified' (literal) and 'both_endpoints' are sensitivities
    "evidence_variant_primary": "prespecified_cc",
    "evidence_variants_sensitivity": ["prespecified", "both_endpoints", "both_endpoints_cc",
                                      "chip_prespecified_cc", "lit_merged_extri_ntnu", "lit_no_ht",
                                      "lit_no_chipderived", "no_motif_coexp"],
    # motif/coexpression annotators under the curated-conditional mask: a TF is
    # covered if the annotator has >=1 record of that TF in the resource as
    # distributed ('whole_resource'). Decision 10 ('within_conditioning_set')
    # was REVERTED on 2026-10-08 before any held-out result was inspected: it
    # makes observation depend on the annotator's own reports (outcome-
    # dependent, not missing at random). All fits use 'whole_resource'.
    "motif_coexp_tf_coverage": "whole_resource",
    # top-k predicted sets: NaN scores are 'not predicted'; ties at the k-th
    # position are broken uniformly at random with the fixed seed below
    "topk_tie_break": "seeded_random",
    "topk_tie_seed": 20261008,
    # ranking metrics: NaN scores rank below every finite score
    "nan_rank": "last",
    # foundation-model attention: raw mean attention is the primary score;
    # 'uniform' (mean 1/L over co-observing cells) is a reported null and
    # 'norm' = attn / uniform a robustness variant
    "fm_attention_variants": ["attn", "attn_T", "attn_norm", "attn_T_norm", "attn_uniform"],
    # perturbation cohorts
    "shifrut_de": "stratified",              # donor/stimulation-adjusted DE is primary
    "cis_exclusion": "same_cytoband",        # drop targets in the perturbed gene's cytoband
    "cohort_cluster_unit": "study_id",       # dependence unit for replication counts
    # held-out ChIP
    "encode_strict_min_cell_lines": 2,        # ENCODE-strict variant (TFs assayed in >=2 lines)
    # expression baselines: GRNBoost2/GENIE3 and correlations share one cell
    # sample per tissue (outputs/r2/grnboost2/<t>/cells.txt); a >=1%-detection
    # gene filter is a sensitivity analysis
    "expr_n_cells": 3000,
    "grnboost2_early_stopping": "same_subset_oob_monitor",   # src/r2/arboreto_compat.py
    "grn_detection_filter_sensitivity": 0.01,                # GRNBoost2 + GENIE3: unfiltered primary
    # ---- second decision round, logged 2026-10-07 before any held-out evaluation
    "fm_min_coobs_sensitivity": 5,           # top-k after requiring >=5 co-observing cells
    "replicate_stratified_de": ["shifrut", "papalexi"],   # multi-replicate designs: stratified DE primary
    "symbol_mapping": "approved_symbol_first_then_ensembl",
    "lit_no_chipderived_drop": ["FANTOM4", "HTRIdb", "ORegAnno", "LIT_other"],
    "target_degree_bins": 10,
    "target_degree_bin_rule": "union_gene_edge_deciles",
    "target_degree_definition": "leave_pair_out_ref_indegree",
}

# ----------------------------------------------------------------------------
# Gene-symbol normalisation
# ----------------------------------------------------------------------------


_MISSING_TOKENS = {"NA", "NAN", "NONE", "NULL", "-"}


@lru_cache(maxsize=1)
def _alias_map() -> dict:
    """Map approved, previous and alias symbols (upper case) to the approved symbol.

    Approved symbols always map to themselves; a previous/alias symbol that is
    ambiguous (points to >1 approved symbol) or that collides with an approved
    symbol is not used.
    """
    df = pd.read_csv(HGNC_TABLE, sep="\t", dtype=str).fillna("")
    approved = set(df["Approved symbol"].str.upper())
    cand: dict[str, set] = {}
    for _, row in df.iterrows():
        a = row["Approved symbol"].upper()
        for col in ("Previous symbols", "Alias symbols"):
            for s in row[col].split(","):
                s = s.strip().upper()
                if s and s not in approved:
                    cand.setdefault(s, set()).add(a)
    m = {a: a for a in approved}
    for tok in _MISSING_TOKENS:
        cand.pop(tok, None)
    for s, targets in cand.items():
        if len(targets) == 1:
            m[s] = next(iter(targets))
    return m


def normalize_symbol(sym) -> str:
    """Return the HGNC-approved upper-case symbol, or the stripped upper-case input."""
    if sym is None or (isinstance(sym, float) and np.isnan(sym)) or sym is pd.NA:
        return ""
    s = str(sym).strip().upper()
    if s in _MISSING_TOKENS:
        return ""
    return _alias_map().get(s, s)


def normalize_series(s: pd.Series) -> pd.Series:
    """Vectorised normalize_symbol; missing values map to ''."""
    m = _alias_map()
    u = s.where(s.notna(), "").astype(str).str.strip().str.upper()
    return u.map(lambda x: "" if (not x or x in _MISSING_TOKENS) else m.get(x, x))


@lru_cache(maxsize=1)
def ensembl_to_symbol() -> dict:
    """Map Ensembl gene ids (without version) to HGNC-approved upper-case symbols."""
    df = pd.read_csv(HGNC_FULL, sep="\t", dtype=str, usecols=["symbol", "ensembl_gene_id"]).dropna()
    return dict(zip(df["ensembl_gene_id"].str.split(".").str[0], df["symbol"].str.upper()))


@lru_cache(maxsize=1)
def symbol_cytoband() -> dict:
    """HGNC cytogenetic location (e.g. '17q21.32') per approved symbol."""
    df = pd.read_csv(HGNC_FULL, sep="\t", dtype=str, usecols=["symbol", "location"]).dropna()
    return dict(zip(df["symbol"].str.upper(), df["location"]))


@lru_cache(maxsize=1)
def lambert_tfs() -> frozenset:
    """Human TFs (Lambert et al. 2018, 'Is TF?' == 'Yes'), HGNC-normalised."""
    df = pd.read_csv(LAMBERT_TF)
    df = df[df["Is TF?"].astype(str).str.strip().str.lower() == "yes"]
    return frozenset(normalize_series(df["HGNC symbol"]).tolist())


# ----------------------------------------------------------------------------
# Small IO helpers
# ----------------------------------------------------------------------------


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def write_json(obj, path: Path) -> None:
    ensure_dir(path.parent)
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=2, default=str)


def edge_key(tf: pd.Series, target: pd.Series) -> pd.Series:
    return tf.astype(str) + "\t" + target.astype(str)
