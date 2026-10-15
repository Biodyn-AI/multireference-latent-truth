"""Checks of the observation masks, universes, sensitivity variants and integration baselines.

Toy tests compare ``r2.evidence`` with brute-force re-implementations; data tests (skipped
when the committed inputs / stored outputs are absent) re-derive stored results.

Run: .venv/bin/python -m pytest subproject_24_multireference_latent_truth_model/src/r2/tests/test_evidence.py -q
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from r2 import build_annotators as BA  # noqa: E402
from r2 import common as C  # noqa: E402
from r2 import evidence as E  # noqa: E402

FAMS = {"L1": "LIT", "L2": "LIT", "L3": "LIT", "M1": "MOTIF", "X1": "COEXP", "C1": "CHIP"}


def _toy(seed: int = 0):
    rng = np.random.default_rng(seed)
    genes = [f"G{i:03d}" for i in range(60)]
    tfs = genes[:12] + ["TFX"]                  # TFX: only in MOTIF / CHIP (never LIT)
    rows = []
    for s in FAMS:
        pool = tfs if FAMS[s] in ("MOTIF", "CHIP") else genes[:12]
        for tf in rng.choice(pool, size=rng.integers(3, 9), replace=False):
            tg_pool = genes + (["GX_MOTIF_ONLY"] if FAMS[s] != "LIT" else [])
            for g in rng.choice(tg_pool, size=rng.integers(1, 15), replace=False):
                if g != tf:
                    rows.append((s, FAMS[s], "x", tf, g, bool(rng.random() < 0.6)))
    rows += [("M1", "MOTIF", "x", "TFX", "G001", False), ("M1", "MOTIF", "x", "G000", "GX_MOTIF_ONLY", False),
             ("C1", "CHIP", "x", "G001", "GX_MOTIF_ONLY", False)]
    ed = pd.DataFrame(rows, columns=["source_id", "family", "lineage", "tf", "target", "via_dorothea"]).drop_duplicates(
        ["source_id", "tf", "target"])
    ed.loc[ed["family"] != "LIT", "via_dorothea"] = False
    return ed.reset_index(drop=True), tfs, genes


def _brute(ed, cond, sources, tf, g, mask, coverage="whole_resource"):
    """Brute-force evidence value of every source for (tf, g).

    coverage='whole_resource' (prespecified): TF coverage and node sets from all records.
    coverage='within_conditioning_set' (reverted decision 10, non-default option): under the
    _cc masks a MOTIF/COEXP source covers tf only if it has >= 1 record of tf inside cond;
    target conditions always use the whole-resource node set."""
    cset = set(map(tuple, cond[["tf", "target"]].values))
    out = []
    for s in sources:
        e = ed[ed.source_id == s]
        pos = ((e.tf == tf) & (e.target == g)).any()
        nodes = set(e.tf) | set(e.target)
        cov = set(e.tf)
        within = (coverage == "within_conditioning_set" and mask.endswith("_cc")
                  and FAMS[s] in ("MOTIF", "COEXP"))
        if within:
            cov = {t for t, x in zip(e.tf, e.target) if (t, x) in cset}
        if mask.startswith("both"):
            obs = (tf in cov if within else tf in nodes) and g in nodes
        else:
            obs = tf in cov and (FAMS[s] in ("MOTIF", "COEXP", "CHIP") or g in nodes)
        if mask.endswith("_cc") and FAMS[s] in ("MOTIF", "COEXP"):
            obs = obs and (tf, g) in cset
            pos = pos and (tf, g) in cset
        out.append(1 if pos else (0 if obs else -1))
    return out


def test_default_coverage_rule_is_prespecified():
    assert C.PRESPECIFIED["motif_coexp_tf_coverage"] == "whole_resource"
    assert E.CC_TF_COVERAGE == "whole_resource" and E.CC_TF_COVERAGE_RULES[0] == "whole_resource"
    ed, tfs, genes = _toy()
    assert E.AnnotatorBank(ed).cc_tf_coverage == "whole_resource"
    with pytest.raises(ValueError):
        E.AnnotatorBank(ed, cc_tf_coverage="nonsense")


@pytest.mark.parametrize("mask", E.MASKS)
@pytest.mark.parametrize("chip", [False, True])
@pytest.mark.parametrize("coverage", E.CC_TF_COVERAGE_RULES)
def test_masks_and_patterns_match_bruteforce(mask, chip, coverage):
    ed, tfs, genes = _toy()
    cond = E.conditioning_set(ed)
    assert set(map(tuple, cond.values)) == set(map(tuple, ed.loc[(ed.family == "LIT") & ed.via_dorothea, ["tf", "target"]].values))
    sources = [s for s in FAMS if FAMS[s] != "CHIP"] + (["C1"] if chip else [])
    bank = E.AnnotatorBank(ed, sources=sources, cond=cond, cc_tf_coverage=coverage)
    univ_tfs = tfs[:12]
    univ_genes = genes + ["UNSEEN1", "UNSEEN2"]          # genes in no annotator
    ps, dense = E.evidence_patterns(bank, univ_tfs, univ_genes, mask=mask, grid=True, return_dense=True)
    G = len(univ_genes)
    obs = np.zeros((len(univ_tfs), bank.K), bool)
    rep = np.zeros((len(univ_tfs), bank.K), bool)
    for i, tf in enumerate(univ_tfs):
        for k, g in enumerate(univ_genes):
            e = i * G + k
            if tf == g:
                assert ps.edge_pattern[e] == -1
                continue
            want = _brute(ed, cond, bank.sources, tf, g, mask, coverage)
            assert list(dense[e]) == want, (tf, g, mask)
            assert list(ps.patterns[ps.edge_pattern[e]]) == want
            obs[i] |= np.array(want) >= 0
            rep[i] |= np.array(want) == 1
    assert (ps.tf_observed == obs).all() and (ps.tf_reported == rep).all()
    assert ps.counts.sum() == len(univ_tfs) * G - len(univ_tfs)
    # arbitrary edge-list interface agrees with the grid
    sel = np.flatnonzero(ps.edge_pattern >= 0)[::7]
    d2 = E.evidence_for_edges(bank, [univ_tfs[e // G] for e in sel], [univ_genes[e % G] for e in sel], mask)
    assert (d2 == dense[sel]).all()
    # non-grid compression drops self pairs and keeps TF-major order
    ps2 = E.evidence_patterns(bank, univ_tfs, univ_genes, mask=mask, grid=False)
    assert len(ps2.edge_pattern) == ps.counts.sum()
    assert (ps2.patterns[ps2.edge_pattern] == ps.patterns[ps.edge_pattern[ps.edge_pattern >= 0]]).all()


@pytest.mark.parametrize("mask", E.MASKS)
def test_within_cond_option_only_hides_motif_coexp_negatives(mask):
    """The reverted option can only turn observed MOTIF/COEXP negatives (0) into missing (-1)
    under the _cc masks; every report, every LIT/CHIP cell and every non-cc mask is unchanged."""
    ed, tfs, genes = _toy(3)
    cond = E.conditioning_set(ed)
    sources = [s for s in FAMS]
    a = E.AnnotatorBank(ed, sources=sources, cond=cond)
    b = E.AnnotatorBank(ed, sources=sources, cond=cond, cc_tf_coverage="within_conditioning_set")
    univ_genes = genes + ["UNSEEN1"]
    _, da = E.evidence_patterns(a, tfs[:12], univ_genes, mask=mask, grid=False, return_dense=True)
    _, db = E.evidence_patterns(b, tfs[:12], univ_genes, mask=mask, grid=False, return_dense=True)
    diff = da != db
    if not mask.endswith("_cc"):
        assert not diff.any()
        return
    assert not diff[:, ~a.is_cc].any()
    assert ((da[diff] == 0) & (db[diff] == -1)).all()
    assert ((da == 1) == (db == 1)).all()
    # TF coverage of the option is a subset of the whole-resource coverage
    for j in np.flatnonzero(a.is_cc):
        assert b.cov_cc[j] <= a.cov_cc[j] and a.cov_cc[j] == a.cov[j]
    # targets: the option keeps the whole-resource node rule of both_endpoints_cc
    assert a.node_mask.shape == b.node_mask.shape and (a.node_mask == b.node_mask).all()


def _fake_store(root, meta_extra):
    d = root / "toyu"
    d.mkdir(parents=True)
    pd.DataFrame({"pattern_id": [0, 1], "A": [1, -1], "n_edges": [3, 5]}).to_parquet(d / "patterns.parquet")
    np.save(d / "edge_pattern.npy", np.array([0, 1, 1], dtype=np.int32))
    meta = {"universe": "toyu", **meta_extra,
            "variants": {E.PRIMARY_VARIANT: {"sources": ["A"], "family": ["LIT"]}}}
    (d / "meta.json").write_text(json.dumps(meta))
    return root


def test_load_patterns_checks_coverage_rule(tmp_path):
    ok = _fake_store(tmp_path / "ok", {"motif_coexp_tf_coverage": C.PRESPECIFIED["motif_coexp_tf_coverage"]})
    R, n, fam, ann = E.load_patterns("toyu", ev_dir=ok)
    assert R.tolist() == [[1], [-1]] and n.tolist() == [3, 5] and ann == ["A"]
    assert E.load_edge_pattern("toyu", ev_dir=ok).tolist() == [0, 1, 1]
    for name, extra in (("missing", {}), ("other", {"motif_coexp_tf_coverage": "within_conditioning_set"})):
        bad = _fake_store(tmp_path / name, extra)
        with pytest.raises(ValueError, match="motif_coexp_tf_coverage"):
            E.load_patterns("toyu", ev_dir=bad)
        with pytest.raises(ValueError, match="motif_coexp_tf_coverage"):
            E.load_edge_pattern("toyu", ev_dir=bad)


def test_global_universe_uses_literature_only():
    ed, tfs, genes = _toy()
    heldin = ed[ed.family != "CHIP"]
    panel = ["PANEL_ONLY", "G001"]
    gtfs, ggenes, info = E.global_universe(heldin, panel)
    lit = ed[ed.family == "LIT"]
    assert gtfs == sorted(set(lit.tf))
    assert ggenes == sorted(set(lit.target) | set(panel))
    assert "TFX" not in gtfs and "GX_MOTIF_ONLY" not in ggenes
    # invariant to adding MOTIF/COEXP/CHIP edges with new TFs or targets
    extra = pd.DataFrame([("M1", "MOTIF", "x", "NEWTF", "NEWG", False), ("C1", "CHIP", "x", "G000", "NEWG2", False)],
                         columns=ed.columns)
    t2, g2, _ = E.global_universe(pd.concat([heldin, extra]), panel)
    assert (t2, g2) == (gtfs, ggenes)
    assert info["n_lit_tfs"] == len(gtfs)


def test_variant_transforms():
    ed, _, _ = _toy()
    l1, l2 = ed[ed.source_id == "L1"], ed[ed.source_id == "L2"]
    meta = {"merge_sets": {"L1+L2": ["L1", "L2"]},
            "lit_ht_like_regulons": [["L1", l1.tf.iloc[0]], ["L3", "NOT_A_TF"]]}
    m = E.transform_edges(ed, meta, "merge_sets")
    assert "L1" not in set(m.source_id) and "L2" not in set(m.source_id)
    mm = m[m.source_id == "L1+L2"]
    u = pd.concat([l1, l2])
    assert set(zip(mm.tf, mm.target)) == set(zip(u.tf, u.target))
    via = u.groupby(["tf", "target"]).via_dorothea.any()
    assert (mm.set_index(["tf", "target"]).via_dorothea.sort_index() == via.sort_index()).all()
    assert E.conditioning_set(m).equals(E.conditioning_set(ed))
    d = E.transform_edges(ed, meta, "drop_lit_ht_like")
    gone = (d.source_id == "L1") & (d.tf == l1.tf.iloc[0])
    assert not gone.any()
    assert len(d) == len(ed) - int(((ed.source_id == "L1") & (ed.tf == l1.tf.iloc[0])).sum())
    cd = E.conditioning_set(d)
    assert set(map(tuple, cd.values)) <= set(map(tuple, E.conditioning_set(ed).values))
    # whole-source removal (lit_no_chipderived): only the named LIT sources go, cond recomputed
    x = E.transform_edges(ed, {"lit_no_chipderived_drop": ["L1", "L3"]}, "drop_lit_chipderived")
    assert set(x.source_id) == set(ed.source_id) - {"L1", "L3"}
    assert len(x) == int((~ed.source_id.isin(["L1", "L3"])).sum())
    want = ed[(ed.family == "LIT") & ed.via_dorothea & ~ed.source_id.isin(["L1", "L3"])]
    assert set(map(tuple, E.conditioning_set(x).values)) == set(zip(want.tf, want.target))
    with pytest.raises(KeyError):
        E.transform_edges(ed, {"lit_no_chipderived_drop": ["M1"]}, "drop_lit_chipderived")


def test_universe_dependence():
    base = pd.DataFrame({"tf": ["T1", "T1", "T2", "T3"], "target": ["A", "B", "C", "T1"]})
    var = base[base.tf != "T2"]                      # T2 (and its only target C) removed
    tfs, genes = ["T1", "T2", "T3"], ["A", "B", "C", "P", "T1"]
    info, lists = E.universe_dependence(base, var, tfs, genes, panel_genes=["P"], grid=True)
    assert lists["tfs"] == ["T2"] and lists["genes"] == ["C"]
    # cells with TF T2 (5) or gene C (3), minus the shared cell (T2, C), no self-pair involved
    assert info["n_edges_with_tf_or_gene_entering_only_via_removed"] == 5 + 3 - 1
    info2, lists2 = E.universe_dependence(base, var, tfs, genes, panel_genes=["C"], grid=True)
    assert lists2["genes"] == [] and info2["n_genes_lit_target_only_via_removed"] == 1


def test_dorothea_level_rule():
    #            n_cur rev  chip   tfbs   gtex   expected
    cases = [(0, False, False, False, False, None),
             (0, False, False, True, True, None),       # computational only -> E (not distributed)
             (1, False, False, False, False, "D"),
             (1, False, False, False, True, "D"),       # curated & GTEx
             (0, False, True, False, False, "D"),       # ChIP only
             (1, False, False, True, False, "C"),
             (0, False, True, True, False, "C"),
             (0, False, True, False, True, "C"),
             (1, False, True, False, False, "B"),
             (1, False, False, True, True, "B"),
             (0, False, True, True, True, "B"),
             (1, False, True, True, True, "A"),
             (2, False, False, False, False, "A"),
             (1, True, False, False, False, "A")]
    arr = list(zip(*cases))
    got = BA.dorothea_level_rule(*arr[:5])
    assert list(got) == list(arr[5])


# --------------------------------------------------------------------------------------
# data tests on the committed inputs and stored outputs
# --------------------------------------------------------------------------------------
needs_inputs = pytest.mark.skipif(not E.INPUT_EDGES.exists(), reason="committed annotator inputs absent")
needs_global = pytest.mark.skipif(not (E.EV_DIR / "global" / "meta.json").exists(), reason="evidence not built")


@needs_inputs
def test_committed_annotators():
    ed, meta = E.load_annotator_inputs()
    assert "TRRD" not in set(ed.source_id)
    assert meta["duplicate_drops"]["TRRD"]["n_kept"] == 0
    pazar = set(zip(*ed.loc[ed.source_id == "PAZAR", ["tf", "target"]].T.values))
    bc = pd.read_parquet(C.OUT / "annotators" / "annotator_edges_by_channel.parquet")
    nfi_kept = bc[(bc.primary_resource == "NFIRegulomeDB") & (bc.drop_reason == "")]
    assert not any((t, g) in pazar for t, g in zip(nfi_kept.tf, nfi_kept.target))
    lit = ed[ed.family == "LIT"]
    size = lit.groupby(["source_id", "tf"]).size()
    ht = {tuple(x) for x in meta["lit_ht_like_regulons"]}
    assert set(size[size > meta["lit_ht_max_targets"]].index) <= ht
    assert set(map(tuple, BA.LIT_HT_NAMED)) <= ht
    assert set(meta["held_in"]) == set(ed.loc[ed.family != "CHIP", "source_id"])
    for p in (E.INPUT_EDGES, C.INPUTS / "integration_baselines.parquet"):
        assert p.stat().st_size < 20e6
    # lit_no_ht removes exactly the 9 regulons of the committed table
    tab = pd.read_csv(C.TABLES / "lit_ht_like_regulons.csv")
    assert ht == set(zip(tab.source_id, tab.tf)) and len(ht) == 9
    assert "exactly the 9" in meta["lit_ht_like_description"]
    # lit_no_chipderived: the prespecified whole LIT sources
    assert meta["lit_no_chipderived_drop"] == C.PRESPECIFIED["lit_no_chipderived_drop"]
    fam = ed.groupby("source_id").family.first()
    assert all(fam[s] == "LIT" for s in meta["lit_no_chipderived_drop"])


@needs_inputs
def test_lit_removing_variants():
    ed, meta = E.load_annotator_inputs()
    drop = C.PRESPECIFIED["lit_no_chipderived_drop"]
    bank = E.bank_for_variant("lit_no_chipderived")
    assert not set(bank.sources) & set(drop)
    assert set(bank.sources) == set(meta["held_in"]) - set(drop)
    keep = ed[(ed.family == "LIT") & ed.via_dorothea & ~ed.source_id.isin(drop)]
    assert bank.n_cond == len(set(zip(keep.tf, keep.target)))
    # the HT-like variant keeps every source but removes the 9 regulons
    b2 = E.bank_for_variant("lit_no_ht")
    assert set(b2.sources) == set(meta["held_in"])
    for s, tf in meta["lit_ht_like_regulons"]:
        assert tf not in b2.pos[b2.sources.index(s)]
    if (E.EV_DIR / "global" / "meta.json").exists():
        gm = json.loads((E.EV_DIR / "global" / "meta.json").read_text())
        genes = set((E.EV_DIR / "global" / "genes.txt").read_text().split())
        panel = set().union(*(E.tissue_universe(t)[1] for t in C.TISSUES))
        for v in E.LIT_REMOVING_VARIANTS:
            lst = json.loads((E.EV_DIR / "global" / f"lit_only_via_removed{E._suffix(v)}.json").read_text())
            vl = E.variant_edges(v)
            vl = vl[vl.family == "LIT"]
            assert not set(lst["genes"]) & (set(vl.target) | panel)
            assert set(lst["genes"]) <= genes
            assert not set(lst["tfs"]) & set(vl.tf)
            dep = gm["variants"][v]["universe_dependence"]
            assert dep["n_genes_entering_only_via_removed"] == len(lst["genes"])
            assert dep["n_tfs_entering_only_via_removed"] == len(lst["tfs"])


@needs_inputs
def test_portable_raw_inputs():
    man = json.loads((C.INPUTS / "annotator_raw_inputs.json").read_text())
    for k in ("dorothea", "collectri", "trrust"):
        port = man["inputs"][k]["portable"]
        p = C.SP / port["path"]
        assert p.exists() and not Path(port["path"]).is_absolute()
        assert BA.sha256(p) == port["sha256"]
        assert p.stat().st_size < 20e6


@needs_inputs
def test_integration_baselines():
    ib = pd.read_parquet(C.INPUTS / "integration_baselines.parquet")
    ed, _ = E.load_annotator_inputs()
    d = ib[ib.in_dorothea]
    # held-in level missing iff no held-in curated DoRothEA evidence
    assert (d.dorothea_level_heldin.notna() == d.dorothea_curated_heldin.astype(bool)).all()
    assert ib.loc[~ib.in_dorothea, "dorothea_level_heldin"].isna().all()
    # held-in evidence is a subset of the full evidence: the rule can only lower the level
    r_full = d.dorothea_level_rule.map(BA.LEVEL_RANK).fillna(9)
    r_h = d.dorothea_level_heldin.map(BA.LEVEL_RANK).fillna(9)
    assert (r_h >= r_full).all()
    assert (d.dorothea_level_rule == d.dorothea_level).mean() > 0.99
    # the held-in curated set is the conditioning set of the curated-conditional masks
    cond = E.conditioning_set(ed)
    hk = set(zip(d.loc[d.dorothea_curated_heldin.astype(bool), "tf"], d.loc[d.dorothea_curated_heldin.astype(bool), "target"]))
    assert hk == set(zip(cond.tf, cond.target))
    # CollecTRI without DoRothEA-A
    c = ib[ib.in_collectri]
    assert (c.collectri_n_resources_heldin == c.collectri_n_resources - c.collectri_dorothea_a.astype(int)).all()
    assert (c.in_collectri_heldin == (c.collectri_n_resources_heldin >= 1)).all()
    assert not ib.loc[~ib.in_collectri, "in_collectri_heldin"].any()
    # flags agree with the committed annotator edges
    key = pd.MultiIndex.from_frame(ib[["tf", "target"]])
    for col, m in (("heldin_support_lit", ed.family == "LIT"),
                   ("remap", ed.source_id == "ReMap"), ("pazar_ht", ed.source_id == "PAZAR-HT"),
                   ("encode_tag", ed.source_id.isin(BA.TRUNCATED_CHIP))):
        want = key.isin(pd.MultiIndex.from_frame(ed.loc[m, ["tf", "target"]]))
        assert (ib[col].values == want).all(), col
    mc = key.isin(pd.MultiIndex.from_frame(ed.loc[ed.family.isin(["MOTIF", "COEXP"]), ["tf", "target"]]))
    assert (ib.motif_coexp_only_support.values == (mc & ~ib.heldin_support_lit.values)).all()
    assert "heldin_support" not in ib.columns and "heldin_lit_support" not in ib.columns
    # motif/coexpression-only held-in support is ChIP-conditioned (held-out ChIP on every such edge)
    mo = ib[ib.motif_coexp_only_support]
    assert len(mo) > 0 and (mo.remap | mo.pazar_ht).all()
    # the held-in DoRothEA level requires held-in literature support
    assert ib.loc[ib.dorothea_level_heldin.notna(), "heldin_support_lit"].all()
    assert (ib.in_dorothea_heldin == ib.dorothea_level_heldin.notna()).all()
    # held-in CollecTRI effort / references: released values on held-in CollecTRI edges only
    h = ib.in_collectri_heldin
    for a, b in (("collectri_curation_effort_heldin", "collectri_curation_effort"),
                 ("collectri_n_references_heldin", "collectri_n_references")):
        assert ib.loc[~h, a].isna().all()
        assert (ib.loc[h, a] == ib.loc[h, b]).all()


@needs_inputs
@needs_global
def test_stored_global_universe():
    ed, _ = E.load_annotator_inputs()
    panel = set().union(*(E.tissue_universe(t)[1] for t in C.TISSUES))
    tfs, genes, _ = E.global_universe(ed[ed.family != "CHIP"], panel)
    d = E.EV_DIR / "global"
    assert (d / "tfs.txt").read_text().split() == tfs
    assert (d / "genes.txt").read_text().split() == genes
    meta = json.loads((d / "meta.json").read_text())
    assert meta["n_edges"] == len(tfs) * len(genes) - len(set(tfs) & set(genes))
    for u in ("global",) + tuple(C.TISSUES):
        m = json.loads((E.EV_DIR / u / "meta.json").read_text())
        assert m["motif_coexp_tf_coverage"] == C.PRESPECIFIED["motif_coexp_tf_coverage"] == "whole_resource"
    # no TF or gene enters only through MOTIF/COEXP/CHIP evidence
    lit = ed[ed.family == "LIT"]
    assert set(tfs) == set(lit.tf)
    assert set(genes) - set(lit.target) <= panel


@needs_inputs
@needs_global
@pytest.mark.parametrize("variant", list(E.VARIANTS))
def test_stored_evidence_matches_fresh_computation(variant):
    rng = np.random.default_rng(1)
    d = E.EV_DIR / "global"
    tfs = (d / "tfs.txt").read_text().split()
    genes = (d / "genes.txt").read_text().split()
    G = len(genes)
    R, n, fam, ann = E.load_patterns("global", variant)
    ep = E.load_edge_pattern("global", variant)
    assert n.sum() == (ep >= 0).sum()
    bank = E.bank_for_variant(variant)
    assert ann == bank.sources
    # random cells plus cells of reported edges
    lit = E.load_annotator_inputs()[0]
    rep = lit[lit.tf.isin(tfs) & lit.target.isin(genes)].sample(1000, random_state=1)
    ti = pd.Index(tfs).get_indexer(rep.tf) * G + pd.Index(genes).get_indexer(rep.target)
    cells = np.unique(np.r_[rng.choice(len(ep), 2000, replace=False), ti])
    cells = cells[ep[cells] >= 0]
    fresh = E.evidence_for_edges(bank, [tfs[c // G] for c in cells], [genes[c % G] for c in cells],
                                 E.VARIANTS[variant]["mask"])
    assert (fresh == R[ep[cells]]).all()
    # summary table: __ALL__ TF coverage equals the stored per-TF coverage
    summ = pd.read_csv(C.TABLES / "evidence_summary.csv")
    row = summ[(summ.universe == "global") & (summ.variant == variant) & (summ.annotator == "__ALL__")].iloc[0]
    cov = pd.read_parquet(d / f"tf_coverage{E._suffix(variant)}.parquet")
    assert row.n_tfs_covered == int((cov[ann] >= 0).any(axis=1).sum())
    assert row.n_tfs_reported == int((cov[ann] == 1).any(axis=1).sum())
