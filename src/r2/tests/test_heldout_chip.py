"""Checks of the held-out ChIP stage (``r2.build_heldout_chip``).

Toy tests cover the prespecified target-degree null (settings, leave-pair-out degree, margin
matching), the ENETS2 relabel/drop logic, the raw-file readers, coverage counting, the
literature sensitivity groups, the overlap flags and the strict upstream-input checks.  Data
tests (skipped when the raw files / stored outputs are absent) re-derive stored results.

Run: .venv/bin/python -m pytest subproject_24_multireference_latent_truth_model/src/r2/tests/test_heldout_chip.py -q
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from r2 import build_heldout_chip as HC  # noqa: E402
from r2 import common as C  # noqa: E402


# ----------------------------------------------------------------------------
# target-degree null
# ----------------------------------------------------------------------------


def _toy_matrix(seed=0, n_tf=14, n_g=300):
    rng = np.random.default_rng(seed)
    hub = rng.random(n_g) ** 3                       # skewed gene propensity -> hub tail
    act = rng.uniform(0.05, 0.6, n_tf)
    M = rng.random((n_tf, n_g)) < np.clip(act[:, None] * (0.2 + 2 * hub[None, :]), 0, 0.95)
    self_col = np.full(n_tf, -1)
    self_col[: n_tf // 2] = np.arange(n_tf // 2)    # half of the TFs are universe genes
    M[np.arange(n_tf // 2), self_col[: n_tf // 2]] = False
    return M, self_col


def test_null_settings_come_from_prespecified():
    assert HC.TARGET_DEGREE_BINS == C.PRESPECIFIED["target_degree_bins"]
    assert HC.TARGET_DEGREE_BIN_RULE == C.PRESPECIFIED["target_degree_bin_rule"]
    assert HC.TARGET_DEGREE_DEFINITION == C.PRESPECIFIED["target_degree_definition"]
    M, sc = _toy_matrix()
    with pytest.raises(ValueError):
        HC.target_degree_model(M, sc, bin_rule="equal_width")
    with pytest.raises(ValueError):
        HC.target_degree_model(M, sc, degree_definition="full_indegree")
    mod = HC.target_degree_model(M, sc)["model"]
    assert mod["n_bins_prespecified"] == C.PRESPECIFIED["target_degree_bins"]
    assert mod["bin_rule"] == C.PRESPECIFIED["target_degree_bin_rule"]
    assert mod["degree_definition"] == C.PRESPECIFIED["target_degree_definition"]


def test_bin_edges_union_of_gene_and_edge_quantiles():
    deg = np.array([0, 0, 0, 1, 1, 2, 3, 5, 8, 13, 21, 40])
    q = np.linspace(0, 1, 11)
    exp = np.unique(np.concatenate([np.quantile(deg, q, method="lower"),
                                    np.quantile(np.repeat(deg, deg), q, method="lower")]))
    assert np.array_equal(HC.degree_bin_edges(deg, 10), exp)


def test_target_degree_model_matches_margins_brute_force():
    M, sc = _toy_matrix()
    fit = HC.target_degree_model(M, sc)
    assert fit["model"]["converged"]
    n_tf, n_g = M.shape
    deg = M.sum(0)
    nb = fit["P"].shape[1]
    obs_tf, fit_tf = np.zeros(n_tf), np.zeros(n_tf)
    obs_b, fit_b = np.zeros(nb), np.zeros(nb)
    for i in range(n_tf):
        for g in range(n_g):
            if g == sc[i]:
                continue
            b = int(fit["to_bin"](deg[g] - M[i, g]))      # leave-pair-out degree
            obs_tf[i] += M[i, g]
            fit_tf[i] += fit["P"][i, b]
            obs_b[b] += M[i, g]
            fit_b[b] += fit["P"][i, b]
    assert np.max(np.abs(obs_tf - fit_tf)) < 1e-4
    assert np.max(np.abs(obs_b - fit_b)) < 1e-4
    # and the null is not trivially flat: hub bins carry higher fitted odds
    assert fit["model"]["bin_logodds"][-1] > fit["model"]["bin_logodds"][0]


def test_target_degree_model_is_deterministic():
    M, sc = _toy_matrix(seed=3)
    a, b = HC.target_degree_model(M, sc), HC.target_degree_model(M, sc)
    assert np.array_equal(a["P"], b["P"])
    assert a["model"] == b["model"]


def _toy_reference():
    edges = pd.DataFrame({"tf": ["TA", "TA", "TA", "TB", "TB", "TC", "TC", "TC"],
                          "target": ["G1", "G2", "G3", "G1", "G2", "G1", "G4", "TA"]})
    universe = ["G1", "G2", "G3", "G4", "G5", "G6", "TA", "TB"]
    return edges, universe


def test_reference_index_and_pair_queries():
    edges, universe = _toy_reference()
    R = HC.reference_index(edges, universe, assayed=["TA", "TB", "TC", "TD"])   # TD: assayed, no edges
    assert R["tfs"] == ["TA", "TB", "TC", "TD"]
    assert R["targets"]["TD"] == set() and R["base"]["TD"] == 0
    assert R["base"]["TA"] == pytest.approx(3 / 7)      # TA is in the universe: 8 - 1 pairs
    assert R["base"]["TC"] == pytest.approx(3 / 8)
    deg = dict(zip(sorted(universe), R["deg"]))
    assert deg["G1"] == 3 and deg["G2"] == 2
    # leave-pair-out: (TA, G1) is an edge -> degree 2; (TD, G1) is not -> degree 3
    b = HC.degree_bin(R, ["TA", "TD", "TX", "TA", "TA"], ["G1", "G1", "G1", "GZ", "TA"])
    assert b[0] == R["to_bin"](2) and b[1] == R["to_bin"](3)
    assert list(b[2:]) == [-1, -1, -1]                  # unknown TF, gene outside universe, self-pair
    p = HC.null_probability(R, ["TA", "TD", "TX"], ["G1", "G1", "G1"])
    assert p[0] == pytest.approx(R["P"][R["ti"]["TA"], b[0]])
    assert np.isnan(p[2])


# ----------------------------------------------------------------------------
# ENETS2 relabel / readers
# ----------------------------------------------------------------------------


def _raw(rows):
    return pd.DataFrame(rows, columns=["factor_raw", "edge_type", "target_raw"])


def test_unconfirmed_relabel_drops_factor_and_confirmed_relabels():
    raw2 = _raw([("NFKB1", "-", "MYC"), ("NFKB1", "-", "CD19"), ("TCF4", "-", "MYC"),
                 ("TCF4", "-", "TCF7L2"), ("GATA1", "-", "MYC")])
    raw1 = _raw([("NFKB1", "-", "MYC"), ("GATA1", "-", "TP53"), ("JUNB", "-", "MYC")])
    tfs = frozenset({"NFKB1", "RELA", "TCF4", "TCF7L2", "GATA1", "JUNB"})
    ev = {"NFKB1": {"confirmed": False, "relabel_to": "RELA", "encode_antibody_term": "NFKB"},
          "TCF4": {"confirmed": True, "relabel_to": "TCF7L2", "encode_antibody_term": "TCF4"}}
    out = HC.build_enets2(raw2, raw1, tfs, ev)
    e = out["edges"]
    assert not e["tf"].isin(["NFKB1", "RELA"]).any()         # unconfirmed -> dropped, not kept
    assert set(e.loc[e["tf"] == "TCF7L2", "target"]) == {"MYC"}   # relabelled; self-loop dropped
    assert "TCF4" not in set(e["tf"])
    d = pd.DataFrame(out["dropped"]).set_index("factor_raw")
    assert d.at["NFKB1", "action"] == "dropped" and "not confirmed" in d.at["NFKB1", "reason"]
    assert d.at["TCF4", "action"] == "relabelled" and d.at["TCF4", "factor_used"] == "TCF7L2"
    assert d.at["JUNB", "action"] == "not_evaluable"
    assert "TP53" in out["universe"]


def test_read_encodenets_keeps_literal_NA(tmp_path):
    p = tmp_path / "net.txt"
    p.write_text("GATA1\tProximal_filtered\tMYC\nNA\tProximal_filtered\tTP53\nGATA1\tProximal_filtered\tNA\n")
    df = HC.read_encodenets(p)
    assert len(df) == 3
    assert df["factor_raw"].tolist() == ["GATA1", "NA", "GATA1"]
    assert df["target_raw"].iloc[2] == "NA"
    assert not df.isna().any().any()


@pytest.mark.skipif(not HC.RAW_FILES["gerstein2012_table_S4"]["path"].exists(),
                    reason="Gerstein et al. table S4 not downloaded")
def test_read_gerstein_s4a_counts():
    s4 = HC.read_gerstein_s4a(HC.RAW_FILES["gerstein2012_table_S4"]["path"])
    d = dict(zip(s4["factor"], s4["n_targets"]))
    assert d["TCF7L2"] == 377
    assert d["NFKB1"] == 165
    assert "TCF4" not in d


# ----------------------------------------------------------------------------
# coverage, sensitivity groups, flags, strict inputs
# ----------------------------------------------------------------------------


def test_coverage_rows_toy_universe():
    edges, universe = _toy_reference()
    R = HC.reference_index(edges, universe, assayed=["TA", "TB", "TC"])
    built = {"toy": {"edges": edges}}
    utfs, ugenes = {"TA", "TC", "TZ"}, {"G1", "G3", "G4", "TA", "GX"}
    rows = HC.coverage_rows(built, {"toy": R}, {"u": (utfs, ugenes)})
    assert len(rows) == 1
    r = rows[0]
    assert r["evaluable_tfs"] == 2                       # TA, TC (TZ not assayed)
    # genes in both: G1, G3, G4, TA -> TA: 3 (self excluded), TC: 4
    assert r["evaluable_edges"] == 7
    # positives: TA->G1, TA->G3, TC->G1, TC->G4, TC->TA
    assert r["positives"] == 5
    assert r["density"] == pytest.approx(5 / 7)
    assert r["tfs_with_positives"] == 2


def _toy_annotators():
    rows = []
    for src, fam, held_out, pairs in [
        ("FANTOM4", "LIT", False, [("A", "x1"), ("A", "x2")]),
        ("HTRIdb", "LIT", False, [("B", "x1")]),
        ("ORegAnno", "LIT", False, [("CTCF", "x3"), ("B", "x2")]),
        ("LIT_other", "LIT", False, [("A", "x4")]),
        ("TRRUST", "LIT", False, [("A", "x1"), ("B", "x5")]),
        ("PAZAR", "LIT", False, [("HIF1A", "x6")]),
        ("ReMap", "CHIP", True, [("A", "x1")]),
    ]:
        rows += [(src, fam, held_out, t, g) for t, g in pairs]
    return pd.DataFrame(rows, columns=["annotator", "family", "held_out", "tf", "target"])


def test_sensitivity_groups_definitions():
    ann = _toy_annotators()
    sj = {"lit_ht_like_regulons": [["ORegAnno", "CTCF"], ["PAZAR", "HIF1A"]],
          "lit_other_components": ["DoRothEA-reviews", "NFIRegulomeDB"]}
    groups, defs = HC.sensitivity_groups(ann, sj)
    nh, nc = groups["all_held_in_no_ht"], groups["all_held_in_no_chipderived"]
    assert set(map(tuple, nh[["annotator", "tf"]].to_numpy())) & {("ORegAnno", "CTCF"), ("PAZAR", "HIF1A")} == set()
    assert {"FANTOM4", "HTRIdb", "LIT_other"} <= set(nh["annotator"])     # lit_no_ht leaves them in
    assert not nc["annotator"].isin(C.PRESPECIFIED["lit_no_chipderived_drop"]).any()
    assert not nc["held_out"].any() and not nh["held_out"].any()
    # pooled edges that lose all support: no_ht -> CTCF-x3, HIF1A-x6; no_chipderived -> A-x2, CTCF-x3, B-x1, B-x2, A-x4
    assert defs["all_held_in_no_ht"]["pooled_edges_removed"] == 2
    assert defs["all_held_in_no_chipderived"]["pooled_edges_removed"] == 5    # A-x1 kept via TRRUST
    assert defs["all_held_in_no_ht"]["records_kept_of_listed_sources"]["ORegAnno"] == "1 of 2"
    with pytest.raises(ValueError):
        HC.sensitivity_groups(ann[ann["annotator"] != "HTRIdb"], sj)
    with pytest.raises(ValueError):
        HC.sensitivity_groups(ann, {"lit_ht_like_regulons": [["TRED", "MYC"]]})


def test_overlap_flags_consistent():
    chip = ["ENCODE-distal", "ENCODE-proximal", "PAZAR-HT", "ReMap"]
    rows = [(ref, a) for ref in ("encode_tfbs", "encode_strict", "enets2") for a in (
        "ReMap", "PAZAR-HT", "ENCODE-proximal", "family:CHIP", "TRRUST", "all_held_in",
        "all_held_in|with_chip_remap_pazarht", "all_held_in|without_chip_remap_pazarht",
        "all_held_in|with_chip_any", "all_held_in|without_chip_any", "TFe")]
    ov = pd.DataFrame(rows, columns=["reference", "annotator"])
    ov["n_tfs_evaluable"] = np.where(ov["annotator"] == "TFe", 4, 50)
    ov["n_reference_positive"] = np.where(ov["annotator"] == "TFe", 8, 500)
    f = HC.overlap_flags(ov, chip).set_index(["reference", "annotator"])
    for ref in ("encode_tfbs", "encode_strict", "enets2"):
        assert HC.FLAG_SHARES in f.at[(ref, "ReMap"), "flag"]                     # every reference
        assert f.at[(ref, "ReMap"), "shares_encode_experiments"]
        assert "ReMap" in f.at[(ref, "all_held_in|with_chip_remap_pazarht"), "flag"]
        assert f.at[(ref, "all_held_in|with_chip_remap_pazarht"), "shares_encode_experiments"]
        assert f.at[(ref, "all_held_in|without_chip_remap_pazarht"), "flag"] == ""
        assert f.at[(ref, "all_held_in|without_chip_remap_pazarht"), "conservative_headline"]
        assert f.at[(ref, "TRRUST"), "flag"] == "" and not f.at[(ref, "TRRUST"), "shares_encode_experiments"]
        assert f.at[(ref, "PAZAR-HT"), "flag"] == ""
        assert f.at[(ref, "TFe"), "low_support"] and HC.FLAG_LOW in f.at[(ref, "TFe"), "flag"]
        assert f.at[(ref, "family:CHIP"), "shares_encode_experiments"]
    assert f.at[("enets2", "ENCODE-proximal"), "flag"] == HC.FLAG_SAME_NETWORK
    assert f.at[("encode_tfbs", "ENCODE-proximal"), "flag"] == HC.FLAG_SHARES
    assert f.at[("enets2", "all_held_in|with_chip_any"), "flag"].startswith(HC.FLAG_UNINFORMATIVE)
    assert f.at[("enets2", "all_held_in|without_chip_any"), "flag"] == HC.FLAG_UNINFORMATIVE
    assert f.at[("encode_tfbs", "all_held_in|without_chip_any"), "flag"] == ""
    assert "ENCODE-proximal" in f.at[("encode_tfbs", "all_held_in|with_chip_any"), "flag"]


def test_missing_upstream_inputs_raise(tmp_path, monkeypatch):
    monkeypatch.setattr(C, "OUT", tmp_path)
    monkeypatch.setattr(HC, "GLOBAL_DIR", tmp_path / "evidence" / "global")
    with pytest.raises(FileNotFoundError):
        HC.universe_sets()
    with pytest.warns(UserWarning):
        assert HC.universe_sets(allow_partial=True) == {}
    monkeypatch.setattr(HC, "ANNOTATOR_EDGES", tmp_path / "none.parquet")
    with pytest.raises(FileNotFoundError):
        HC.load_annotators(0)
    with pytest.warns(UserWarning):
        assert HC.load_annotators(0, allow_partial=True) is None


# ----------------------------------------------------------------------------
# stored outputs
# ----------------------------------------------------------------------------

OV = HC.OUT_DIR / "annotator_overlap.csv"


@pytest.mark.skipif(not OV.exists(), reason="heldout_chip outputs not built")
def test_stored_overlap_table():
    ov = pd.read_csv(OV, keep_default_na=False, na_values=[""])
    for ref in ("encode_tfbs", "encode_strict", "enets2"):
        r = ov[ov["reference"] == ref].set_index("annotator")
        for g in ("all_held_in_no_chipderived", "all_held_in_no_ht",
                  "all_held_in_no_chipderived|without_chip_remap_pazarht"):
            assert g in r.index
        assert bool(r.at["ReMap", "shares_encode_experiments"])
        assert bool(r.at[HC.CONSERVATIVE_HEADLINE, "conservative_headline"])
        # every row with 'with_' selection on ReMap is flagged
        w = r[r.index.str.contains(r"\|with_chip_remap_pazarht$")]
        assert w["shares_encode_experiments"].astype(bool).all()
        low = (r["n_tfs_evaluable"] < HC.MIN_SUPPORT_TFS) | (r["n_reference_positive"].fillna(0) < HC.MIN_SUPPORT_POSITIVES)
        assert (r["low_support"].astype(bool) == low).all()
