"""Tests for the label-based tuning optimism module (src/r2/tuning_optimism.py)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from r2 import tuning_optimism as T  # noqa: E402


def _tied_scores(rng, n, n_levels=6):
    return rng.integers(0, n_levels, n).astype(float) / n_levels


def test_ap_lift_matches_sklearn_with_ties():
    rng = np.random.default_rng(0)
    for _ in range(20):
        s = _tied_scores(rng, 400)
        y = rng.random(400) < 0.1 + 0.3 * s
        r = T.ap_lift(T.score_blocks(s), y)
        ap = average_precision_score(y, s)
        assert np.isclose(r["ap"], ap)
        assert np.isclose(r["lift"], ap / y.mean())


def test_ap_lift_no_positive_is_nan():
    r = T.ap_lift(T.score_blocks(np.array([0.1, 0.2])), np.array([False, False]))
    assert np.isnan(r["lift"]) and r["n_pos"] == 0


def test_score_blocks_nan_last_and_rounding():
    s = np.array([0.5, np.nan, 0.5 + 1e-14, 0.9, -np.inf])
    b = T.score_blocks(s)
    assert b[3] == 0 and b[0] == b[2] == 1      # 1e-14 apart: one tie block
    assert b[1] == b[4] == b.max()               # NaN ranks with -inf, last


def test_ap_lift_by_group_matches_per_group():
    rng = np.random.default_rng(1)
    n, G = 900, 7
    s = _tied_scores(rng, n)
    y = rng.random(n) < 0.05 + 0.4 * s
    g = rng.integers(0, G, n)
    g[g == 3] = 4                                 # group 3 empty
    y[g == 5] = False                             # group 5 without positives
    r = T.ap_lift_by_group(T.score_blocks(s), y, g, G)
    for k in range(G):
        m = g == k
        if m.sum() == 0 or y[m].sum() == 0:
            assert np.isnan(r["lift"][k])
            continue
        ap = average_precision_score(y[m], s[m])
        assert np.isclose(r["ap"][k], ap)
        assert np.isclose(r["lift"][k], ap / y[m].mean())


def test_evaluable_pairs_filters_universe_self_and_cis():
    de = pd.DataFrame(dict(
        perturbed_gene=["A", "A", "A", "B", "X", "A"],
        target=["g1", "A", "g2", "g1", "g1", "zz"],
        positive_primary=[True, True, True, False, True, True],
        cis_same_band=[False, False, True, False, False, False]))
    tfs = pd.Index(["A", "B"])
    genes = pd.Index(["A", "B", "g1", "g2"])
    p = T.evaluable_pairs(de, tfs, genes)
    # kept: A-g1 (pos), B-g1 (neg); dropped: self A-A, cis A-g2, TF X not in universe, target zz not in universe
    assert p[["tf_idx", "gene_idx", "positive"]].values.tolist() == [[0, 2, True], [1, 2, False]]


def test_pool_labels_any_positive_per_study_and_pooled():
    pairs = pd.DataFrame(dict(tf_idx=[0, 0, 0, 1, 1], gene_idx=[2, 2, 3, 2, 2],
                              positive=[False, True, False, False, False],
                              study=["s1", "s2", "s1", "s1", "s1"]))
    lab = T.pool_labels(pairs, ["s1", "s2", "s3"], n_genes=4, tfs=["A", "B"])
    assert lab.edge.tolist() == [2, 3, 6]
    assert lab.tf.tolist() == [0, 0, 1]
    m, y = lab.select()
    assert m.tolist() == [True, True, True] and y.tolist() == [True, False, False]
    m1, y1 = lab.select(["s1"])
    assert m1.tolist() == [True, True, True] and y1.tolist() == [False, False, False]
    m2, y2 = lab.select(["s2"])
    assert m2.tolist() == [True, False, False] and y2.tolist() == [True, False, False]
    m3, _ = lab.select(["s3"])
    assert not m3.any()


def test_random_tf_splits_deterministic_disjoint_covering():
    pool = np.array([5, 1, 9, 3, 7, 2, 8])
    a = T.random_tf_splits(pool, n_splits=10, seed=3)
    b = T.random_tf_splits(pool, n_splits=10, seed=3)
    for (t1, h1), (t2, h2) in zip(a, b):
        assert t1.tolist() == t2.tolist() and h1.tolist() == h2.tolist()
        assert len(t1) == 3 and len(h1) == 4
        assert not set(t1) & set(h1)
        assert sorted(set(t1) | set(h1)) == sorted(pool.tolist())
    assert len({tuple(t) for t, _ in a}) > 1


def test_select_best_ties_first_in_order():
    assert T.select_best([0.1, 0.3, 0.3 - 1e-13, 0.2]) == (1, 2)
    assert T.select_best([np.nan, 0.2, np.nan]) == (1, 1)
    assert T.select_best([np.nan, np.nan])[0] == -1


def test_weight_grid_and_minmax():
    W = T.weight_grid(0.1)
    assert W.shape == (66, 3)
    assert np.allclose(W.sum(1), 1) and (W >= 0).all()
    assert len({tuple(np.round(w, 6)) for w in W}) == 66
    assert np.allclose(T.minmax_good([1.0, 3.0, 2.0]), [0, 1, 0.5])
    assert np.allclose(T.minmax_good([1.0, 3.0, 2.0], higher_better=False), [1, 0, 0.5])
    assert np.allclose(T.minmax_good([2.0, 2.0]), [0, 0])
    # 2 adjacent pairs per inner edge: 3 * (10*11/2) = 165 pairs on the step-0.1 simplex
    assert len(T._adjacent_pairs(W)) == 165


def test_joint_select_extremes():
    W = np.array([[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]])
    d_s = np.array([0.3, 0.1, 0.2])     # candidate 1 most robust to source dropout
    d_b = np.array([0.05, 0.2, 0.3])    # candidate 0 most robust to block dropout
    pert = np.array([1.0, 2.0, 5.0])    # candidate 2 best perturbation reward
    sel, J = T.joint_select(W, d_s, d_b, pert)
    assert sel.tolist() == [1, 0, 2]
    assert J.shape == (3, 3)


def test_split_records_gaps():
    ids = ["a", "b", "family_hb", "d"]
    vin = np.array([1.0, 3.0, 2.0, 1.5])
    vout = np.array([1.2, 1.8, 2.1, 1.0])
    rec = T.split_records(vin, vout, ids, {"label_free_cvll": 3, "primary_family_hb": 2})
    assert rec["label_tuned"]["config"] == "b" and np.isclose(rec["label_tuned"]["gap"], 1.2)
    assert rec["oracle_heldout"]["config"] == "family_hb"
    assert np.isclose(rec["primary_family_hb"]["excess_gap"], 0.0)
    assert np.isclose(rec["label_tuned"]["excess_gap"], 1.2 - (-0.1))
    assert np.isclose(rec["label_tuned"]["out_minus_primary"], 1.8 - 2.1)
    assert np.isclose(rec["config_average"]["out_value"], vout.mean())


def test_selection_on_noise_is_optimistic():
    """Configurations that are pure noise: the selected one looks better in-sample than out of
    sample, and out of sample it is no better than the average configuration."""
    rng = np.random.default_rng(0)
    n_tf, per_tf, n_cfg = 40, 60, 15
    tf = np.repeat(np.arange(n_tf), per_tf)
    y = rng.random(len(tf)) < 0.05
    pairs = pd.DataFrame(dict(tf_idx=tf, gene_idx=np.tile(np.arange(per_tf), n_tf), positive=y, study="s"))
    lab = T.pool_labels(pairs, ["s"], n_genes=per_tf, tfs=[f"t{i}" for i in range(n_tf)])
    m, yy = lab.select()
    blocks = [T.score_blocks(rng.random(len(lab.edge))) for _ in range(n_cfg)]
    gaps, rel = [], []
    for tune, hold in T.random_tf_splits(np.arange(n_tf), n_splits=30, seed=1):
        mt, mh = m & np.isin(lab.tf, tune), m & np.isin(lab.tf, hold)
        vin = np.array([T.ap_lift(b[mt], yy[mt])["lift"] for b in blocks])
        vout = np.array([T.ap_lift(b[mh], yy[mh])["lift"] for b in blocks])
        rec = T.split_records(vin, vout, [f"c{i}" for i in range(n_cfg)],
                              {"label_free_cvll": 0, "primary_family_hb": 1})
        # selection bias: gap of the selected configuration beyond the gap of the grid average
        gaps.append(rec["label_tuned"]["gap"] - rec["config_average"]["gap"])
        rel.append(rec["label_tuned"]["out_value"] - rec["config_average"]["out_value"])
    assert np.mean(gaps) > 0.2 and np.mean(np.array(gaps) > 0) >= 0.7
    assert abs(np.mean(rel)) < 0.15


def test_geomean():
    assert np.isclose(T.geomean([1.0, 4.0]), 2.0)
    assert np.isclose(T.geomean([2.0, np.nan, 8.0]), 4.0)
    assert np.isnan(T.geomean([]))


def test_run_tuning_paths_agree_with_direct_criterion():
    """Random-half values computed via the cached per-TF lifts equal the direct criterion, and
    LOSO evaluates the left-out study's labels."""
    rng = np.random.default_rng(3)
    n_tf, per_tf = 12, 80
    tf = np.repeat(np.arange(n_tf), per_tf)
    study = np.where(tf < 6, "s1", "s2")
    y = rng.random(len(tf)) < 0.15
    pairs = pd.DataFrame(dict(tf_idx=tf, gene_idx=np.tile(np.arange(per_tf), n_tf), positive=y, study=study))
    lab = T.pool_labels(pairs, ["s1", "s2"], n_genes=per_tf, tfs=[f"t{i}" for i in range(n_tf)])
    blocks = [T.score_blocks(_tied_scores(rng, len(lab.edge), 8)) for _ in range(4)]
    ids = ["a", "family_hb", "c", "d"]
    detail, sel, splits, null_info = T.run_tuning(blocks, lab, ids, {"label_free_cvll": 2, "primary_family_hb": 1})
    m, yy = lab.select()
    sp0 = splits[(splits.split_type == "random_tf_half") & (splits.split_id == 0) & (splits.half == "tuning")]
    tune = np.array([lab.tfs.index(t) for t in sp0["tf"]])
    mt = m & np.isin(lab.tf, tune)
    d0 = detail[(detail.split_type == "random_tf_half") & (detail.split_id == 0)]
    assert set(detail["criterion"]) == set(T.CRITERIA)
    assert set(detail["split_type"]) == {"random_tf_half", "random_tf_half_stratified", "loso"}
    for c in T.RAW_CRITERIA:
        got = d0[d0.criterion == c].set_index("config_id")["in_value"]
        for i, cid in enumerate(ids):
            assert np.isclose(got[cid], T.criterion(blocks[i], lab, mt, yy, c))
    lo = detail[(detail.split_type == "loso") & (detail.heldout_study == "s2") & (detail.criterion == "pooled_ap_lift")]
    mh, yh = lab.select(["s2"])
    assert np.isclose(lo.set_index("config_id")["out_value"]["c"], T.ap_lift(blocks[2][mh], yh[mh])["lift"])
    assert set(sel["selector"]) == {"label_tuned", "label_free_cvll", "primary_family_hb", "oracle_heldout",
                                    "config_average"}


# ----------------------------------------------------------------------------
# revision 2026-10-08: null-normalised criteria, stratified halves, fold sets
# ----------------------------------------------------------------------------
def _toy_lab(seed=4, n_tf=10, per_tf=120, studies=("s1", "s2", "s3")):
    rng = np.random.default_rng(seed)
    tf = np.repeat(np.arange(n_tf), per_tf)
    study = np.array(studies)[tf % len(studies)]
    y = rng.random(len(tf)) < 0.05 + 0.02 * (tf % 4)
    pairs = pd.DataFrame(dict(tf_idx=tf, gene_idx=np.tile(np.arange(per_tf), n_tf), positive=y, study=study))
    return T.pool_labels(pairs, list(studies), n_genes=per_tf, tfs=[f"t{i}" for i in range(n_tf)]), rng


def test_within_tf_permutations_keep_tf_positive_counts():
    lab, rng = _toy_lab()
    m, y = lab.select()
    Y = T.within_tf_permutations(y[m], lab.tf[m], 30, np.random.default_rng(1))
    k_obs = np.bincount(lab.tf[m], weights=y[m].astype(float))
    for r in range(Y.shape[0]):
        row = np.asarray(Y[r].todense()).ravel()
        assert set(np.unique(row)) <= {0.0, 1.0}
        assert np.array_equal(np.bincount(lab.tf[m], weights=row, minlength=len(k_obs)), k_obs)


def test_perm_ap_functions_match_single_label_versions():
    lab, rng = _toy_lab()
    m, y = lab.select()
    b = T.score_blocks(_tied_scores(rng, int(m.sum()), 7))
    Y = T.within_tf_permutations(y[m], lab.tf[m], 8, np.random.default_rng(2))
    pooled = T.ap_lift_perm(b, Y)
    per = T.ap_lift_by_group_perm(b, Y, lab.tf[m], len(lab.tfs), chunk=3)
    for r in range(Y.shape[0]):
        yr = np.asarray(Y[r].todense()).ravel().astype(bool)
        assert np.isclose(pooled[r], T.ap_lift(b, yr)["lift"])
        ref = T.ap_lift_by_group(b, yr, lab.tf[m], len(lab.tfs))["lift"]
        assert np.allclose(per[r], ref, equal_nan=True)


def test_nullnorm_per_tf_criterion_is_centred_for_random_scores():
    """For scores unrelated to the labels the null-normalised per-TF criterion is ~1 although the
    raw geometric mean is not (it is inflated by tie structure and small positive counts)."""
    lab, rng = _toy_lab(seed=9, n_tf=20, per_tf=150)
    m, y = lab.select()
    b = T.score_blocks(_tied_scores(rng, int(m.sum()), 60))
    Y = T.within_tf_permutations(y[m], lab.tf[m], 200, np.random.default_rng(3))
    null = T.perm_null_per_tf([b], Y, lab.tf[m], len(lab.tfs))[0]
    vals = []
    for s in range(20):
        yr = T.within_tf_permutations(y[m], lab.tf[m], 1, np.random.default_rng(100 + s))
        yr = np.asarray(yr.todense()).ravel().astype(bool)
        lf = np.log2(T.ap_lift_by_group(b, yr, lab.tf[m], len(lab.tfs))["lift"])
        vals.append(np.nanmean(lf - null))
    assert abs(np.mean(vals)) < 0.05
    assert np.nanmean(null) > 0.0           # the raw per-TF log2 lift is biased upwards under the null


def test_stratified_splits_balance_strata():
    pool = np.arange(12)
    strata = pd.Series(["a"] * 6 + ["b"] * 3 + ["c"] * 3, index=pool)
    sp = T.stratified_tf_splits(pool, strata, n_splits=10, seed=5)
    for tune, hold in sp:
        assert len(np.intersect1d(tune, hold)) == 0 and len(tune) + len(hold) == 12
        assert np.sum(np.isin(tune, np.arange(6))) == 3
        assert abs(len(tune) - len(hold)) <= 1
    again = T.stratified_tf_splits(pool, strata, n_splits=10, seed=5)
    assert all(np.array_equal(a[0], b[0]) for a, b in zip(sp, again))


def test_optimism_table_fold_sets_and_retained_ratio():
    rows = []
    for sid, (study, low, npos) in enumerate((("s1", False, 100), ("s2", True, 3), ("s3", False, 50))):
        for sel_, (vin, vout) in {"label_tuned": (1.2, 1.0 + 0.1 * sid), "primary_family_hb": (1.0, 1.0),
                                  "label_free_cvll": (1.0, 1.0), "oracle_heldout": (1.3, 1.3),
                                  "config_average": (1.1, 1.0)}.items():
            rows.append(dict(grid="all", n_configs=3, split_type="loso", split_id=sid, heldout_study=study,
                             heldout_low_support=low, n_pos_heldout=npos, criterion="pooled_ap_lift",
                             selector=sel_, config="x", in_value=vin, out_value=vout, gap=vin - vout,
                             excess_gap=0.0, in_minus_primary=vin - 1.0, out_minus_primary=vout - 1.0,
                             spearman_halves=0.0, n_tied=1))
    t = T.optimism_table(pd.DataFrame(rows), [])
    lt = t[t.selector == "label_tuned"].set_index("fold_set")
    assert set(lt.index) == {"all_folds", "excl_low_support", "weighted_by_heldout_positives",
                             "study:s1", "study:s2", "study:s3"}
    assert np.isclose(lt.loc["all_folds", "retained_gain_ratio"], 0.1 / 0.2)
    assert np.isclose(lt.loc["excl_low_support", "out_minus_primary_mean"], 0.1)        # folds s1 and s3
    w = np.array([100, 3, 50.0])
    assert np.isclose(lt.loc["weighted_by_heldout_positives", "out_minus_primary_mean"],
                      np.average([0.0, 0.1, 0.2], weights=w))
    assert lt.loc["all_folds", "analysis_status"] == "prespecified"
    assert lt.loc["excl_low_support", "analysis_status"] == "post_hoc"


def test_excluded_config_not_in_grid():
    ids = [c.config_id for c in T.config_grid()]
    assert "family_hb@no_motif_coexp" not in ids and "family_hb" in ids and len(ids) == len(set(ids))


def test_run_weights_merged_summary(monkeypatch):
    lab, rng = _toy_lab(seed=12, n_tf=12, per_tf=100)
    m, y = lab.select()
    cands = list(T.CANDIDATE_SETS["lcm5"])
    base = T.score_blocks(_tied_scores(rng, len(lab.edge), 30))
    est = {c: base for c in cands}                     # identical rankings -> identical perturbation reward
    est["em"] = T.score_blocks(_tied_scores(rng, len(lab.edge), 30))
    rob = pd.DataFrame({"D_source": np.linspace(0.1, 0.5, len(cands)), "D_block": np.linspace(0.5, 0.1, len(cands))},
                       index=cands)
    monkeypatch.setattr(T, "robustness_components", lambda metric: rob)
    splits = T.random_tf_splits(np.unique(lab.tf[m]), n_splits=4, seed=2)
    grid, summ, cf, full = T.run_weights({c: est[c] for c in cands}, lab, splits)
    assert len(grid) == len(T.ROB_METRICS) * len(T.CANDIDATE_SETS) * len(T.weight_grid())
    s = summ[summ["candidate_set"] == "lcm5"]
    assert (s["frac_adjacent_weight_pairs_changed_merged"] <= s["frac_adjacent_weight_pairs_changed"] + 1e-12).all()
    assert (s["mean_crossfit_selection_instability_merged"] <= s["mean_crossfit_selection_instability"] + 1e-12).all()


def test_status_decomposition_tf_stratified(monkeypatch, tmp_path):
    lab, rng = _toy_lab(seed=13, n_tf=8, per_tf=200)
    m, y = lab.select()
    P = 3                                               # patterns: 0 unobserved, 1 observed-negative, 2 reported
    R = np.array([[-1, -1], [0, -1], [1, 0]])
    ep_all = rng.integers(0, P, int(lab.edge.max()) + 1)
    monkeypatch.setattr(T.E, "load_patterns", lambda u, v: (R, np.ones(P), ["LIT", "LIT"], ["a", "b"]))
    monkeypatch.setattr(T.E, "load_edge_pattern", lambda u, v: ep_all)
    np.save(tmp_path / "cred.npy", np.array([0.01, 0.001, 0.9]))
    cfg = T.Config("toy", "estimator", "v", "toy", "family_hb")
    monkeypatch.setattr(T.Config, "fit_path", property(lambda self: tmp_path))
    d = T.status_decomposition([cfg], lab).set_index("status")
    assert np.isclose(d["n_edges"].sum(), m.sum())
    assert np.isclose(d["exp_tf_matched"].sum(), y[m].sum())                     # TF-matched E sums to O overall
    st = ep_all[lab.edge[m]]
    assert d.loc["unobserved", "n_pos"] == int(y[m][st == 0].sum())
    assert np.isfinite(d.loc["reported", "mh_or_vs_unobserved"]) and d.loc["reported", "n_tfs_mh"] > 0


# ----------------------------------------------------------------------------
# additions 2026-10-13 (eval_fixes): LOSO held-out contrast CI, label-free CV table, table status
# ----------------------------------------------------------------------------
def test_heldout_contrast_matches_direct_difference_and_brackets_it():
    lab, rng = _toy_lab(seed=21, n_tf=14, per_tf=150)
    m, y = lab.select()
    ih = np.flatnonzero(m)
    ids = ["a", "family_hb", "c"]
    strong = T.score_blocks(np.round(rng.normal(size=len(lab.edge)) + 2.5 * y, 1))      # clearly better than random
    blocks = [strong, T.score_blocks(_tied_scores(rng, len(lab.edge), 12)), T.score_blocks(_tied_scores(rng, len(lab.edge), 12))]
    rows = [dict(config="a"), dict(config="family_hb"), dict(config="mean_of_grid")]
    T.add_heldout_contrast(rows, [b[ih] for b in blocks], y[ih], lab.tf[ih], ids, np.random.default_rng(1), n_boot=300)
    direct = T.ap_lift(blocks[0][ih], y[ih])["lift"] - T.ap_lift(blocks[1][ih], y[ih])["lift"]
    a = rows[0]
    assert np.isclose(a["heldout_pooled_diff_vs_primary"], direct)
    assert a["heldout_pooled_diff_vs_primary_lo"] <= direct <= a["heldout_pooled_diff_vs_primary_hi"]
    assert a["heldout_pooled_diff_vs_primary_lo"] > 0 and a["heldout_pooled_diff_vs_primary_p_z"] < 0.01
    assert a["heldout_n_tfs_boot"] == len(np.unique(lab.tf[ih]))
    # the primary against itself is exactly zero; a non-configuration record gets NaN
    assert rows[1]["heldout_pooled_diff_vs_primary"] == 0 and np.isnan(rows[1]["heldout_pooled_diff_vs_primary_p_z"])
    assert np.isnan(rows[2]["heldout_pooled_diff_vs_primary"])


def test_loso_rows_carry_the_contrast_and_only_single_folds_report_it():
    rng = np.random.default_rng(5)
    n_tf, per_tf = 12, 80
    tf = np.repeat(np.arange(n_tf), per_tf)
    study = np.where(tf < 6, "s1", "s2")
    y = rng.random(len(tf)) < 0.15
    pairs = pd.DataFrame(dict(tf_idx=tf, gene_idx=np.tile(np.arange(per_tf), n_tf), positive=y, study=study))
    lab = T.pool_labels(pairs, ["s1", "s2"], n_genes=per_tf, tfs=[f"t{i}" for i in range(n_tf)])
    blocks = [T.score_blocks(_tied_scores(rng, len(lab.edge), 8)) for _ in range(3)]
    ids = ["a", "family_hb", "c"]
    detail, sel, splits, _ = T.run_tuning(blocks, lab, ids, {"label_free_cvll": 2, "primary_family_hb": 1},
                                          extra_splits=False)
    lo = sel[(sel.split_type == "loso") & (sel.criterion == "pooled_ap_lift") & (sel.selector == "label_tuned")]
    assert len(lo) == 2 and np.allclose(lo["heldout_pooled_diff_vs_primary"], lo["out_minus_primary"])
    assert (lo["heldout_n_tfs_boot"] == 6).all()
    assert sel.loc[sel.split_type == "random_tf_half", "heldout_pooled_diff_vs_primary"].isna().all()
    sel["heldout_low_support"] = sel["heldout_low_support"].fillna(False)
    t = T.optimism_table(sel.assign(grid="all", n_configs=3), [])
    one = t[(t.split_type == "loso") & t.fold_set.str.startswith("study:") & (t.selector == "label_tuned")
            & (t.criterion == "pooled_ap_lift")]
    assert one["heldout_pooled_diff_vs_primary_lo"].notna().all()
    agg = t[(t.split_type == "loso") & (t.fold_set == "all_folds")]
    assert agg["heldout_pooled_diff_vs_primary_lo"].isna().all()


def test_labelfree_cv_table_ranks_and_pairs_by_fold(tmp_path, monkeypatch):
    import json
    fs, tu = tmp_path / "fs", tmp_path / "tu"
    fs.mkdir(); tu.mkdir()
    ll = {"family_hb": [-10.0, -11.0, -12.0, -9.0, -10.5], "em": [-10.2, -11.1, -12.3, -9.1, -10.4]}
    for cfg, vals in ll.items():
        for f, v in enumerate(vals):
            (fs / f"fold{f}__{cfg}.json").write_text(json.dumps(dict(
                fold=f, config=cfg, test_loglik=v * 100, n_test_observed=100.0, test_loglik_per_obs_edge=v,
                test_cond_loglik_per_cell=-0.5, n_within_1e3=3, n_restarts=20)))
    for f in range(2):                                    # an incomplete prior-variant candidate
        (tu / f"fold{f}__family_hb__prior_scale2.json").write_text(json.dumps(dict(
            fold=f, config="family_hb__prior_scale2", test_loglik=-1001.0, n_test_observed=100.0,
            test_loglik_per_obs_edge=-10.01, test_cond_loglik_per_cell=-0.5, n_within_1e3=2, n_restarts=20)))
    monkeypatch.setattr(T, "FS_CV_DIR", fs)
    monkeypatch.setattr(T, "CV_DIR", tu)
    cv = T.labelfree_cv_table().set_index("config")
    assert cv.loc["family_hb", "complete"] and cv.loc["em", "complete"] and not cv.loc["family_hb__prior_scale2", "complete"]
    assert np.isclose(cv.loc["em", "diff_vs_family_hb_per_edge_mean"], np.mean(np.array(ll["em"]) - np.array(ll["family_hb"])))
    assert cv.loc["em", "n_folds_better_than_family_hb"] == 1 and cv.loc["family_hb", "diff_vs_family_hb_total_nats"] == 0
    assert cv.loc["family_hb", "rank"] < cv.loc["em", "rank"]
    best, cands = T.label_free_choice(cv.reset_index(), ["family_hb", "em", "family_hb__prior_scale2"])
    assert best == "family_hb" and cands == ["em", "family_hb"]            # incomplete candidates are not eligible


def test_write_adds_status_and_post_hoc_columns(tmp_path, monkeypatch):
    monkeypatch.setattr(T, "TABLES", tmp_path)
    monkeypatch.setattr(T.C, "SP", tmp_path)
    T._write(pd.DataFrame(dict(a=[1])), "weight_summary")
    T._write(pd.DataFrame(dict(a=[1], analysis_status=["post_hoc"])), "status_decomposition")
    w = pd.read_csv(tmp_path / "tune_weight_summary.csv")
    d = pd.read_csv(tmp_path / "tune_status_decomposition.csv")
    assert w["analysis_status"].iloc[0] == "prespecified" and w["post_hoc_columns"].iloc[0] == "*_merged"
    assert d["analysis_status"].iloc[0] == "post_hoc"
