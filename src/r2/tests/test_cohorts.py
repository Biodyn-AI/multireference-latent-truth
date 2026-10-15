"""Unit tests for the statistics and gene-identity helpers of ``r2.build_cohorts``.

Run: .venv/bin/python -m pytest subproject_24_multireference_latent_truth_model/src/r2/tests/test_cohorts.py -q
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
from scipy import stats
from statsmodels.stats.multitest import multipletests

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from r2 import build_cohorts as B  # noqa: E402
from r2 import common as C  # noqa: E402


# ----------------------------------------------------------------------------
# bh
# ----------------------------------------------------------------------------

@pytest.mark.parametrize("seed", [0, 1, 2])
def test_bh_matches_statsmodels(seed):
    rng = np.random.default_rng(seed)
    p = np.concatenate([rng.uniform(size=500), rng.uniform(0, 1e-4, size=50), np.ones(20),
                        np.repeat(rng.uniform(size=5), 4)])          # include ties
    rng.shuffle(p)
    ref = multipletests(p, method="fdr_bh")[1]
    np.testing.assert_allclose(B.bh(p), ref, rtol=0, atol=1e-15)


def test_bh_edge_cases():
    assert B.bh(np.array([])).size == 0
    np.testing.assert_allclose(B.bh(np.array([0.03])), [0.03])
    out = B.bh(np.array([0.9, 0.95, 1.0]))
    assert np.all(out <= 1.0) and np.all(np.diff(out[np.argsort([0.9, 0.95, 1.0])]) >= 0)


# ----------------------------------------------------------------------------
# welch (via ExprMatrix.group_stats, as used by the pipeline)
# ----------------------------------------------------------------------------

def _expr_from_dense(Y: np.ndarray) -> B.ExprMatrix:
    X = sp.csr_matrix(Y)
    X.sort_indices()
    return B.ExprMatrix(X.indptr.astype(np.int64), X.indices.astype(np.int32),
                        X.data.astype(np.float32), np.expm1(X.data).astype(np.float32),
                        X.shape[1], {})


def test_welch_matches_scipy():
    rng = np.random.default_rng(3)
    n1, n0, G = 37, 120, 40
    Y = np.log1p(rng.poisson(lam=rng.uniform(0.1, 5, size=G), size=(n1 + n0, G)).astype(float))
    Y[:n1, :5] += 0.7                                           # true effects
    E = _expr_from_dense(Y)
    g1 = E.group_stats(np.arange(n1))
    g0 = E.group_stats(np.arange(n1, n1 + n0))
    t, p = B.welch(g1, g0)
    ref = stats.ttest_ind(Y[:n1].astype(np.float32).astype(float),
                          Y[n1:].astype(np.float32).astype(float), axis=0, equal_var=False)
    ok = np.isfinite(ref.statistic)
    np.testing.assert_allclose(t[ok], ref.statistic[ok], rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(p[ok], ref.pvalue[ok], rtol=1e-4, atol=1e-12)
    # column subset gives the same numbers
    cols = np.array([1, 7, 30])
    t2, p2 = B.welch(g1, g0, cols)
    np.testing.assert_allclose(t2, t[cols])
    np.testing.assert_allclose(p2, p[cols])


def test_welch_zero_variance_is_null():
    Y = np.zeros((30, 3))
    Y[:, 1] = 1.0                       # constant in both groups -> se = 0
    Y[::2, 2] = 2.0
    E = _expr_from_dense(Y)
    g1, g0 = E.group_stats(np.arange(10)), E.group_stats(np.arange(10, 30))
    t, p = B.welch(g1, g0)
    assert t[0] == 0 and p[0] == 1 and t[1] == 0 and p[1] == 1
    assert np.isfinite(t[2]) and 0 <= p[2] <= 1


# ----------------------------------------------------------------------------
# classify_scale
# ----------------------------------------------------------------------------

def test_classify_scale_raw_counts():
    rng = np.random.default_rng(4)
    X = sp.random(300, 200, density=0.1, random_state=4, data_rvs=lambda n: rng.integers(1, 60, n)).tocsr()
    X[0, 0] = 100
    assert B.classify_scale(X)["scale"] == "raw_counts"


def test_classify_scale_log1p_cp10k():
    rng = np.random.default_rng(5)
    X = sp.random(300, 200, density=0.1, random_state=5,
                  data_rvs=lambda n: rng.integers(1, 30, n).astype(float)).tocsr()
    X = X.tolil()
    X[np.arange(300), rng.integers(0, 200, 300)] = 1.0     # real cells have count-1 genes
    X = X.tocsr()
    lib = np.asarray(X.sum(1)).ravel()
    L = sp.diags(1e4 / lib) @ X
    L.data = np.log1p(L.data)
    out = B.classify_scale(sp.csr_matrix(L))
    assert out["scale"] == "log1p_cp10k"
    assert out["integer_reconstruction_fraction"] > 0.999
    assert out["stored_genes_fraction_of_cp10k_median"] == pytest.approx(1.0, rel=1e-6)


def test_classify_scale_rejects_other_values():
    rng = np.random.default_rng(6)
    X = sp.random(100, 50, density=0.2, random_state=6, data_rvs=lambda n: rng.normal(3, 1, n) ** 2).tocsr()
    with pytest.raises(ValueError):
        B.classify_scale(X)


# ----------------------------------------------------------------------------
# stratified test (primary for Shifrut) equals the OLS cell-means contrast
# ----------------------------------------------------------------------------

def test_stratified_matches_ols():
    rng = np.random.default_rng(7)
    strata_sizes = {"A": (15, 60), "B": (40, 30), "C": (8, 90), "D": (2, 50)}   # D dropped (<3)
    strata, group = [], []
    for s, (n1, n0) in strata_sizes.items():
        for g, n in ((1, n1), (0, n0)):
            strata += [s] * n
            group += [g] * n
    strata, group = np.array(strata), np.array(group)
    G = 25
    shift = {"A": 0.0, "B": 1.5, "C": -0.5, "D": 0.2}
    Y = np.log1p(rng.poisson(2.0, size=(len(strata), G)).astype(float))
    Y += np.array([shift[s] for s in strata])[:, None] + 1.0
    Y[group == 1, :4] += 0.4
    Y = Y.astype(np.float32).astype(float)                  # ExprMatrix stores float32
    E = _expr_from_dense(Y)
    r1, r0 = np.flatnonzero(group == 1), np.flatnonzero(group == 0)
    res = B.stratified_stats(r1, r0, strata, E, np.arange(G))
    used = [u[0] for u in res["used"]]
    assert used == ["A", "B", "C"]
    keep = np.isin(strata, used)
    t_ref, df_ref = B.ols_cellmeans_contrast(Y[keep], strata[keep], group[keep])
    np.testing.assert_allclose(res["t"], t_ref, rtol=1e-6)
    assert res["df"] == df_ref
    np.testing.assert_allclose(res["p"], 2 * stats.t.sf(np.abs(t_ref), df_ref), rtol=1e-6)


# ----------------------------------------------------------------------------
# gene identity and cis bands (use the committed HGNC inputs)
# ----------------------------------------------------------------------------

@pytest.fixture(scope="module")
def mapper():
    if not (C.HGNC_FULL.exists() and C.HGNC_TABLE.exists()):
        pytest.skip("HGNC inputs missing")
    return B.gene_mapper()


def test_nonapproved_name_uses_ensembl_fixes_map3k21(mapper):
    # Norman: MAP3K21's own column is named RP5-862P8.2 (ENSG00000143674), not an approved symbol
    assert mapper.resolve("RP5-862P8.2", "ENSG00000143674") == ("MAP3K21", "ensembl")
    assert "ENSG00000143674" in mapper.ensembl_of("MAP3K21")
    # no Ensembl id: falls back to the normalised (previous) symbol
    assert mapper.resolve("TCEB2", "") == ("ELOB", "symbol_normalized")


def test_approved_name_wins_over_reassigned_ensembl_id(mapper):
    # PRESPECIFIED['symbol_mapping'] == 'approved_symbol_first_then_ensembl'
    assert C.PRESPECIFIED["symbol_mapping"] == "approved_symbol_first_then_ensembl"
    # Replogle: the authors' HSPA14 id ENSG00000284024 is MSANTD7 in the newer HGNC release
    assert mapper.from_ensembl("ENSG00000284024") == "MSANTD7"
    assert mapper.resolve("HSPA14", "ENSG00000284024") == ("HSPA14", "symbol_ensembl_differs")
    assert mapper.resolve("HSPA14", "ENSG00000187522") == ("HSPA14", "symbol_ensembl_agrees")
    # ZNF670's Dixit/Adamson id now maps to the readthrough ZNF670-ZNF695; the TF is kept
    assert mapper.resolve("ZNF670", "ENSG00000135747")[0] == "ZNF670"
    # a canonical name no longer approved in HGNC_FULL (renamed/withdrawn) is not an approved
    # symbol: its Ensembl id decides (Dixit/Adamson CXorf22 = CFAP47; Norman TMEM155 = SMIM43)
    assert mapper.resolve("CXorf22", "ENSG00000165164") == ("CFAP47", "ensembl_outdated_name")
    assert mapper.resolve("TMEM155", "ENSG00000164112") == ("SMIM43", "ensembl_outdated_name")
    assert mapper.resolve("CXorf22", "") == ("CXORF22", "symbol")       # no id: canonical name kept
    # ... but the newer symbol of a canonical name maps back to it (TMEM30A = CDC50A)
    assert mapper.resolve("TMEM30A", "ENSG00000112697") == ("TMEM30A", "symbol_ensembl_agrees")
    # the legacy Ensembl-first rule (reporting only) relabelled both
    assert mapper.resolve_ensembl_first("HSPA14", "ENSG00000284024")[0] == "MSANTD7"
    assert mapper.resolve_ensembl_first("ZNF670", "ENSG00000135747")[0] == "ZNF670-ZNF695"
    # duplicate approved names: the column whose id agrees with the name is kept
    M = mapper.map_columns(["HSPA14", "HSPA14", "ZNF670"],
                           ["ENSG00000284024", "ENSG00000187522", "ENSG00000135747"],
                           totals=[100.0, 1.0, 1.0])
    assert list(M["genes"]) == ["HSPA14", "ZNF670"] and list(M["keep_cols"]) == [1, 2]
    # ... also over a column whose id HGNC does not know (Replogle TBCE: ENSG00000285053 vs
    # HGNC's ENSG00000284770)
    M = mapper.map_columns(["TBCE", "TBCE"], ["ENSG00000285053", "ENSG00000284770"], totals=[100.0, 1.0])
    assert list(M["keep_cols"]) == [1] and list(M["all_src"]) == ["symbol", "symbol_ensembl_agrees"]


def test_newer_hgnc_symbols_map_back_to_canonical(mapper):
    # ensembl_to_symbol gives CDC50A / SACK1B (newer HGNC); canonical space keeps TMEM30A / FAM83B
    assert mapper.resolve("TMEM30A", "ENSG00000112697")[0] == "TMEM30A"
    assert mapper.resolve("FAM83B", "ENSG00000168143")[0] == "FAM83B"
    assert mapper.canon("SACK1B") == "FAM83B"


def test_dedup_prefers_approved_name_source(mapper):
    names = ["RP5-862P8.2", "MAP3K21", "GATA1"]
    ens = ["ENSG00000143674", "", "ENSG00000102145"]
    M = mapper.map_columns(names, ens, totals=[1000.0, 1.0, 5.0])   # Ensembl-resolved column has more counts
    assert list(M["genes"]) == ["MAP3K21", "GATA1"]
    assert list(M["keep_cols"]) == [1, 2]
    # without a column named by the approved symbol, the Ensembl-resolved column is used
    M = mapper.map_columns(["RP5-862P8.2", "GATA1"], ["ENSG00000143674", ""], totals=[1.0, 1.0])
    assert list(M["genes"]) == ["MAP3K21", "GATA1"] and M["all_src"][0] == "ensembl"


def test_ambiguous_label_resolved_through_column_name(mapper):
    # Adamson: 'QARS' is a previous symbol of QARS1 and EPRS1 (ambiguous), its column
    # carries QARS1's Ensembl id; the label must follow the column, not stay 'QARS'
    names, ens = ["QARS", "GATA1"], ["ENSG00000172053", "ENSG00000102145"]
    M = mapper.map_columns(names, ens, totals=[1.0, 1.0])
    assert list(M["genes"]) == ["QARS1", "GATA1"]
    lab = B.resolve_labels(mapper, ["QARS", "GATA1", "AARS"], names, M)
    assert lab == {"QARS": "QARS1", "GATA1": "GATA1", "AARS": "AARS1"}
    # an approved label is kept even if a column of that name maps elsewhere by Ensembl id
    M = mapper.map_columns(["HSPA14"], ["ENSG00000284024"], totals=[1.0])
    assert B.resolve_labels(mapper, ["HSPA14"], ["HSPA14"], M) == {"HSPA14": "HSPA14"}


def test_label_aliases_and_self_pair_check_catch_raw_label(mapper):
    # Replogle row 3971_HSPA14_P1P2_ENSG00000284024: raw label HSPA14, row id -> MSANTD7
    al = B.label_aliases(mapper, ["HSPA14"], ["ENSG00000284024"])
    assert al == {"HSPA14", "MSANTD7"}
    # a DE table built under the old rule (perturbation relabelled MSANTD7, HSPA14 left as target)
    de = pd.DataFrame({"perturbed_gene": ["MSANTD7", "MSANTD7"], "target": ["HSPA14", "GATA1"]})
    ok_old = B.self_pair_check(de, mapper, {"HSPA14": "ENSG00000187522", "GATA1": "ENSG00000102145"},
                               {"HSPA14": "HSPA14", "GATA1": "GATA1"})
    assert ok_old["passed"]           # the label-only check misses it ...
    chk = B.self_pair_check(de, mapper, {"HSPA14": "ENSG00000187522", "GATA1": "ENSG00000102145"},
                            {"HSPA14": "HSPA14", "GATA1": "GATA1"},
                            {"MSANTD7": B.label_aliases(mapper, ["HSPA14"], ["ENSG00000284024"]) - {"MSANTD7"}})
    assert not chk["passed"] and chk["n_self_pairs_by_raw_label_alias"] == 1   # ... the raw label catches it
    # own-column exclusion uses the aliases too
    gene_index = {"HSPA14": 0, "GATA1": 1}
    assert B._own_columns("MSANTD7", gene_index, {}, mapper, {"HSPA14"}) == [0]


def test_cis_same_band(mapper):
    de = pd.DataFrame({"perturbed_gene": ["HOXB9", "HOXB9", "HOXB9", "GATA1"],
                       "target": ["HOXB2", "GATA1", "MT-ND1", "HOXB2"],
                       "positive_incl_cis": [True, True, True, False]})
    out = B.annotate_cis(de.copy(), mapper)
    assert out["cis_same_band"].tolist() == [True, False, False, False]
    assert out["positive_primary"].tolist() == [False, True, True, False]
    # full band strings are compared: 17q21.32 (HOXB9) vs 17q21.31 (MAPT) differ
    assert mapper.band("HOXB9") == "17q21.32" and mapper.band("MAPT") == "17q21.31"
    assert not B.annotate_cis(pd.DataFrame({"perturbed_gene": ["HOXB9"], "target": ["MAPT"],
                                            "positive_incl_cis": [True]}), mapper)["cis_same_band"].iloc[0]
    # non-band locations never match
    assert mapper.band("MT-ND1") == ""


def test_parse_location_intervals():
    pl = B.parse_location
    assert pl("17q21.32") == (("17q", 213200, 213299),)
    assert B.intervals_overlap(pl("19q13.1"), pl("19q13.11"))          # coarser band contains finer
    assert not B.intervals_overlap(pl("19q13.12"), pl("19q13.11"))
    assert B.intervals_overlap(pl("2q32.2"), pl("2q32.2-q32.3"))       # range
    assert not B.intervals_overlap(pl("2q32.1"), pl("2q32.2-q32.3"))
    assert B.intervals_overlap(pl("1q41-qter"), pl("1q44"))
    assert pl("1pter-q21") == (("1p", 0, 999999), ("1q", 0, 219999))   # across the centromere
    assert {k for k, _, _ in pl("Xp22.33 and Yp11.2")} == {"Xp", "Yp"}
    assert pl("6p21.33 alternate reference locus") == pl("6p21.33")
    assert not B.intervals_overlap(pl("17p13.1"), pl("17q13.1"))       # different arm
    assert not B.intervals_overlap(pl("2q32.2"), pl("12q32.2"))        # different chromosome
    for loc in ("15", "mitochondria", "reserved", "", "1cen GRCh38 patch"):
        assert pl(loc) == ()


def test_cis_interval_rule_and_previous_symbol_fallback(mapper):
    # missed by exact band strings in the phase-1b build (reverify finding 3)
    assert mapper.is_cis("CEBPA", "ZNF146")       # 19q13.11 vs 19q13.1
    assert mapper.is_cis("STAT1", "STAT4")        # 2q32.2 vs 2q32.2-q32.3
    assert not mapper.is_cis("HOXB9", "MAPT")
    # CTAGE5 is not a symbol of HGNC_FULL (merged into MIA2): band via the previous symbol
    assert "CTAGE5" not in mapper.cyto and mapper.band("CTAGE5") == "14q13.2"


def test_robustness_table_flags_nonrobust_tf():
    def de(rows):
        d = pd.DataFrame(rows, columns=["perturbed_gene", "target", "positive_primary", "is_tf"])
        d["perturbed_gene"] = d["perturbed_gene"].astype("category")
        d["target"] = d["target"].astype("category")
        return d
    strat = de([("SPI1", "A", True, True), ("SPI1", "B", False, True), ("SPI1", "C", False, True),
                ("STAT1", "A", True, True), ("STAT1", "B", True, True),
                ("IFNGR1", "A", False, False), ("MYC", "A", False, True)])
    pooled = de([("SPI1", "A", False, True), ("SPI1", "B", True, True), ("SPI1", "C", True, True),
                 ("STAT1", "A", True, True), ("STAT1", "B", False, True),
                 ("IFNGR1", "A", True, False), ("MYC", "A", False, True)])
    r = B.robustness_table(strat, pooled, "stratified", "pooled").set_index("perturbed_gene")
    assert r.loc["SPI1", "jaccard_pooled_vs_stratified"] == 0 and r.loc["SPI1", "de_nonrobust_tf"]
    assert r.loc["STAT1", "jaccard_pooled_vs_stratified"] == 0.5 and not r.loc["STAT1", "de_nonrobust_tf"]
    assert not r.loc["IFNGR1", "de_nonrobust_tf"]                     # not a TF
    assert np.isnan(r.loc["MYC", "jaccard_pooled_vs_stratified"]) and not r.loc["MYC", "de_nonrobust_tf"]
    assert r.loc["SPI1", "n_positive_primary_stratified"] == 1 and r.loc["SPI1", "n_positive_primary_pooled"] == 2


# ----------------------------------------------------------------------------
# table stage honours --out-root / --tables-dir
# ----------------------------------------------------------------------------

def test_build_table_reads_out_root(tmp_path):
    d = tmp_path / "dixit_d7"
    d.mkdir()
    meta = {"full_name": "x", "accession": "y", "cell_type": "K562", "modality": "CRISPR KO",
            "n_cells_used": 10, "n_control_cells": 3, "n_genes_after_symbol_dedup": 5,
            "n_perturbations_tested": 1, "n_perturbed_tfs_tested": 1, "median_targets_tested": 4,
            "median_positives_primary_per_perturbation": 0, "median_positives_primary_per_tf_perturbation": 0,
            "n_positive_edges_primary": 0, "n_positive_tf_edges_primary": 0, "n_positive_tf_edges_incl_cis": 0,
            "n_fdr_significant_any_lfc": 2, "q99_abs_log2fc_per_perturbation_range": [0.1, 0.2],
            "cis_exclusion": {"n_cis_positive_removed": 0, "n_cis_positive_removed_tf": 0},
            "ontarget": {"n_pass": 0, "n_evaluable": 1, "median_log2fc": -0.01}}
    (d / "meta.json").write_text(json.dumps(meta))
    out = tmp_path / "tables"
    df = B.build_table(["dixit_d13", "dixit_d7"], tmp_path, out)
    assert (out / "cohort_table.csv").exists()
    assert df["cohort_id"].tolist() == ["dixit_d7"]
    assert df["study_id"].iloc[0] == "dixit2016" and df["independent_unit"].iloc[0] == "dixit2016"
    assert "kept but flagged" in df["flag"].iloc[0]
    assert "not evidence of a failed knockout" in df["ontarget_note"].iloc[0]
    assert df["de_design"].iloc[0].startswith("single condition")


def test_dixit_cohorts_share_study():
    assert B.COHORTS["dixit_d7"].study_id == B.COHORTS["dixit_d13"].study_id
    assert len({s.study_id for s in B.COHORTS.values()}) == len(B.COHORTS) - 1
    assert {s.cell_type_group for s in B.COHORTS.values()} == {"K562", "primary T cell", "THP-1 monocyte"}
    # PRESPECIFIED['replicate_stratified_de']: multi-replicate designs use the stratified test
    assert B.PRIMARY_DE == {"shifrut": "stratified", "papalexi": "stratified"}
    assert set(B.PRIMARY_DE) == set(C.PRESPECIFIED["replicate_stratified_de"])
    assert set(B.DE_DESIGN) == set(B.COHORTS)
