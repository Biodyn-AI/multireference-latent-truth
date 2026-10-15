"""Tests for r2.obs_integration (post hoc: informative-observation credibilities in governance and
the replication check)."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from r2 import common as C  # noqa: E402
from r2 import evidence as E  # noqa: E402
from r2 import governance as GOV  # noqa: E402
from r2 import lcm  # noqa: E402
from r2 import obs_integration as OI  # noqa: E402
from r2 import replication as RP  # noqa: E402

HAVE_EVIDENCE = (C.OUT / "evidence" / "global" / "edge_pattern__elig.npy").exists()
HAVE_OUTPUTS = (OI.MOD / "meta.json").exists() and all((C.TABLES / t).exists() for t in OI.TABLE.values())


# ----------------------------------------------------------------------------- helpers
def test_concentration_from_counts_matches_governance():
    rng = np.random.default_rng(3)
    names = np.array([f"TF{i:02d}" for i in range(12)], dtype=object)
    tfs = rng.choice(names, size=200, p=np.r_[0.3, np.full(11, 0.7 / 11)])
    counts = np.array([(tfs == x).sum() for x in names])
    a = GOV.concentration(tfs, 40)
    b = OI.concentration_from_counts(counts, names, 40)
    assert a.keys() == b.keys()
    for k in a:
        if isinstance(a[k], float):
            assert a[k] == pytest.approx(b[k], rel=1e-12, nan_ok=True)
        else:
            assert a[k] == b[k]
    empty = OI.concentration_from_counts(np.zeros(3, int), names[:3], 40)
    assert empty["n_edges"] == 0 and empty["top1_tf"] == ""


def test_levels_on_refined_patterns_equal_stored_pattern_levels():
    """MAR credibilities mapped onto the eligibility refinement give the same tiers (whole tie blocks)."""
    rng = np.random.default_rng(5)
    c = np.round(rng.random(60) ** 3, 3)
    n = rng.integers(2, 9, size=60).astype(float)
    k = rng.integers(1, 3, size=60)                      # each stored pattern split into 1-2 sub-patterns
    base = np.repeat(np.arange(60), k)
    n_ref = np.concatenate([[n[i]] if k[i] == 1 else [1.0, n[i] - 1.0] for i in range(60)])
    lev_plain = GOV.bfdr_levels(c, n, 0.1, 0.3)
    lev_ref = GOV.bfdr_levels(c[base], n_ref, 0.1, 0.3)
    assert np.array_equal(lev_ref, lev_plain[base])
    # edge level (weight 1 per edge) as used by ObsContext
    e = np.repeat(np.arange(len(base)), n_ref.astype(int))
    assert np.array_equal(GOV.bfdr_levels(c[base][e], np.ones(len(e)), 0.1, 0.3), lev_ref[e])
    # replication tiers agree with governance levels
    assert np.array_equal(RP.pattern_tiers(c, n), GOV.bfdr_levels(c, n))


def test_tier_masks_promoted_definition():
    lev = np.array([2, 1, 0, 2, 1])
    m = OI.tier_masks({"global_fit": lev})
    assert m[("global_fit", "Promoted")][1].tolist() == [True, True, False, True, True]
    dual = np.array([3, 2, 1, 0, 3])
    m = OI.tier_masks({"dual": dual})
    assert m[("dual", "Promoted")][1].tolist() == [True, True, False, False, True]
    assert m[("dual", "Strong")][0] == 3 and m[("dual", "Promoted")][0] == 4


def test_no_power_analysis_stub_and_restore():
    orig = GOV.oe_power_analysis
    with OI.no_power_analysis(True):
        assert GOV.oe_power_analysis is not orig
        out = GOV.power_columns(GOV.oe_power_analysis(np.array([10, 10]), np.array([2, 3]), np.array([1, 0]),
                                                      {"global_oe": 1.5}), 1.5)
        assert np.isnan(out["mde_oe"]) and np.isnan(out["power_at_global_oe"]) and out["n_tfs_in_power"] == 1
    assert GOV.oe_power_analysis is orig
    with OI.no_power_analysis(False):
        assert GOV.oe_power_analysis is orig


def test_cred_unobservable_is_w_T_for_obs_fits(monkeypatch):
    rng = np.random.default_rng(7)
    K, S = 5, 4
    fam = ["LIT", "LIT", "LIT", "MOTIF", "COEXP"]
    ann = [f"a{j}" for j in range(K)]
    w = rng.dirichlet(np.ones(S))
    params = dict(w=w, theta=rng.uniform(0.05, 0.9, size=(K, S)),
                  eta=np.vstack([rng.uniform(0.05, 0.9, size=(3, S)), np.full((2, S), np.nan)]))
    monkeypatch.setattr(OI, "obs_params", lambda u, f: params)
    glob = SimpleNamespace(fam=fam, ann=ann, R=np.zeros((1, K), np.int8))
    row = np.full(K, -2, dtype=np.int8)
    c = OI.cred_unobservable(glob, "family_hb_obs", row, np.zeros(1))
    assert c == pytest.approx(w[0], abs=1e-12)
    # an observed pattern moves away from the prior (sanity of the evaluation path)
    data = lcm.PatternData(np.array([[1, 0, -1, -2, -2]], np.int8), np.ones(1), fam, ann, eligibility=True)
    assert abs(float(lcm.credibility_at(params, data)[0]) - w[0]) > 1e-6


def test_cred_unobservable_mar_uses_all_unobserved_pattern():
    glob = SimpleNamespace(R=np.array([[1, 0], [-1, -2], [-2, -2], [0, -1]], np.int8))
    c = np.array([0.9, 0.2, 0.2, 0.1])
    assert OI.cred_unobservable(glob, "family_hb", np.array([-2, -2], np.int8), c) == 0.2
    with pytest.raises(RuntimeError):
        OI.cred_unobservable(glob, "family_hb", np.array([-2, -2], np.int8), np.array([0.9, 0.2, 0.3, 0.1]))


# ----------------------------------------------------------------------------- data checks
@pytest.mark.skipif(not HAVE_EVIDENCE, reason="evidence files not present")
@pytest.mark.parametrize("u", ["immune", "global"])
def test_attach_edges_matches_replication_attach_patterns(u):
    tfs = RP.universe_tfs(u)
    de = RP.read_cohort_de("papalexi", tfs)
    de = de[~de["cis"]].drop(columns="cis")
    a = RP.attach_patterns(u, de)
    gtfs = OI._read_list(C.OUT / "evidence" / "global" / "tfs.txt")
    ggenes = OI._read_list(C.OUT / "evidence" / "global" / "genes.txt")
    b = OI.attach_edges(u, de, gtfs, ggenes)
    assert len(a) == len(b) > 0
    assert (a["tf"].to_numpy() == b["tf"].to_numpy()).all() and (a["target"].to_numpy() == b["target"].to_numpy()).all()
    ep = np.asarray(E.load_edge_pattern(u, OI.VARIANT, mmap=True))
    assert np.array_equal(ep[b["edge"].to_numpy()], a["pattern"].to_numpy())
    ev = OI.Evidence(u)
    assert np.array_equal(ev.base[ev.ep[b["edge"].to_numpy()]], a["pattern"].to_numpy())


@pytest.mark.skipif(not HAVE_EVIDENCE, reason="evidence files not present")
def test_tissue_map_eligibility_identity_and_outside_edges():
    glob = OI.Evidence("global")
    gtfs = OI._read_list(C.OUT / "evidence" / "global" / "tfs.txt")
    ggenes = OI._read_list(C.OUT / "evidence" / "global" / "genes.txt")
    ev = OI.Evidence("kidney")
    tm = OI.TissueMap("kidney", ev, glob, gtfs, ggenes)      # raises if patterns differ
    assert tm.n_outside > 0
    assert (ev.R[ev.ep[~tm.in_global]] == -2).all()
    cg = tm.global_cred(glob, "family_hb_obs", OI.cred_patterns(glob, "family_hb_obs"))
    w = np.load(OI.fit_dir("global", "family_hb_obs") / "params.npz")["w"]
    assert np.allclose(cg[~tm.in_global], w[0])
    # MAR: governance's all-unobserved pattern of the global fit
    c_map = OI.cred_patterns(glob, "family_hb")
    cg_map = tm.global_cred(glob, "family_hb", c_map)
    Rp, _, _, _ = E.load_patterns("global", OI.VARIANT)
    allu = np.flatnonzero((Rp == -1).all(1))
    assert np.allclose(cg_map[~tm.in_global], np.load(OI.fit_dir("global", "family_hb") / "cred.npy")[allu[0]])


# ----------------------------------------------------------------------------- outputs
@pytest.mark.skipif(not HAVE_OUTPUTS, reason="obs_integration outputs not present")
def test_primary_rows_reproduce_prespecified_tables():
    meta = json.loads((OI.MOD / "meta.json").read_text())
    rep = meta["reproduction"]
    assert set(rep) == {"tiers", "validation", "trend", "degree", "replication"}
    for k, r in rep.items():
        assert r["n_matched"] > 0, k
        assert r["n_reproduced"] == r["n_matched"] and not r["mismatches"], k


@pytest.mark.skipif(not HAVE_OUTPUTS, reason="obs_integration outputs not present")
def test_analysis_status_labels():
    for kind, name in OI.TABLE.items():
        df = pd.read_csv(C.TABLES / name, keep_default_na=False)
        assert set(df["analysis_status"]) <= {"prespecified", "post_hoc"}, name
        other = df[df["fit"] != OI.PRIMARY]
        assert (other["analysis_status"] == "post_hoc").all(), name
        assert (other["reproduces"] == "").all(), name
        pre = df[df["analysis_status"] == "prespecified"]
        assert (pre["fit"] == OI.PRIMARY).all() and (pre["reproduces"] != "").all(), name
        assert set(df["fit"]) == set(OI.FIT_IDS), name


@pytest.mark.skipif(not HAVE_OUTPUTS, reason="obs_integration outputs not present")
def test_obs_tiers_are_dominated_by_evidence_status():
    """The documented behaviour of the degenerate obs credibilities: family_hb_obs promotes mainly
    unobserved edges and em_obs mainly observed-not-reported edges (global universe)."""
    t = pd.read_csv(C.TABLES / OI.TABLE["tiers"])
    g = t[(t.universe == "global") & (t.tier == "Promoted")].set_index("fit")
    assert g.loc["family_hb_nuts", "share_reported"] == 1.0
    assert g.loc["family_hb_obs", "share_unobserved"] > 0.5
    assert g.loc["em_obs", "share_observed_negative"] > 0.9
    assert g.loc["family_hb_obs", "share_of_universe"] > 100 * g.loc["family_hb_nuts", "share_of_universe"]
