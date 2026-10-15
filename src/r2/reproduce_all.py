"""Single entry point of the SAGMB R2 pipeline (reviewer Q21 and Q24).

``reproduce_all.py`` lists every stage of the R2 analysis in dependency order, runs the stages
(each step in its own subprocess, one at a time) and records for every executed step the wall
time, the peak resident memory and the CPU time in ``data/r2/tables/pipeline_stage_timings.csv``.
The hardware and software of the run go to ``data/r2/tables/pipeline_environment.csv``.  It
does not change any analysis: every step is the documented command of a module (README), so
the modules keep their own seeds (``r2.common.PRESPECIFIED``), caches and locks.

Modes
-----
* full (default): every step of every stage, from the committed inputs and the external
  raw data (Tabula Sapiens, attention matrices, perturbation and ChIP downloads).  The model
  stages take days on a laptop (NUTS, multi-start MAP, robustness dropout fits, simulations).
* ``--tables-only``: regenerate every manuscript table (``data/r2/tables/*.csv``) and the paper
  assets from the STORED outputs (fits, simulation replicates, compute measurements) without
  refitting anything and without the external raw data.  The annotators (from the committed
  portable inputs) and the evidence patterns are re-built in full; the data stages that read
  external raw data (gene panels, method scores, held-out ChIP references) rebuild their tables
  from their stored outputs; held-out evaluation, governance and replication are evaluations of
  stored fits and run in full; model stages run only their table step.  This needs the
  ``outputs/r2/`` tree of a full run (gitignored, not in the repository), i.e. on a fresh clone
  the supporting archive ``outputs/release/r2_supporting_outputs.tar.gz`` (``--release-archive``
  builds it from a traced run, ``src/r2/release_check.py`` checks a fresh copy with it).

Stage order and what each stage needs are in ``STAGES`` (``--list``, ``--dry-run``).  Options:
``--from STAGE`` (start there), ``--only STAGE ...``, ``--steps STEP ...`` (restrict the steps of
the selected stages), ``--extra-args`` (appended to every step; e.g. ``--universes kidney --root
<scratch>`` to time a fit without overwriting the stored fits, with ``--scope``),
``--repeat-check`` (run the selected stages twice and compare the sha256 of their tables:
determinism), ``--skip STAGE ...``, ``--keep-going``, ``--no-record``, ``--trace-inputs`` (record the
files every step opens, ``outputs/r2/pipeline/logs/<run>/trace/``).  Every run also records which
step wrote which table (``outputs/r2/pipeline/logs/<run>/table_writers.csv``; a table written by a
stage that does not declare it, or overwritten by another stage with different content, fails the
run).

Utility actions (no stage is run): ``--write-environment``, ``--write-requirements`` (pinned
versions of the third-party packages imported by ``src/r2``), ``--collect-recorded`` (stage
runtimes of the original runs from the modules' meta files and run tables),
``--write-manifest`` / ``--check-manifest`` (``data/r2/MANIFEST.sha256`` over ``data/r2/inputs``
and ``data/r2/tables``), ``--refresh-assets`` (re-render the stage-timing table and the
digest after the timing table changed), ``--release-archive RUN_DIR`` (the supporting archive from
the trace of a tables-only run; ``--archive-extra`` adds paths).  ``--write-requirements`` also
writes ``requirements-r2.lock`` (the direct pins plus their installed dependencies).

Timing record (``pipeline_stage_timings.csv``)
---------------------------------------------
One row per stage x step x mode x scope x timing source.  ``timing_source``:
``measured_isolated`` = measured by this script, one step process at a time with no other
process of this pipeline running; "isolated" means sequential within the pipeline, not an idle
machine (other users' load is recorded: ``loadavg_1m_*``; ``n_other_python_procs`` counts only
other Python script processes at the start of the step); ``recorded_meta`` / ``recorded_table`` = the runtime the module stored
when the step originally ran (``runtime_s`` in fit / replicate meta files, run tables), summed
over its units (``n_units``; units ran partly in parallel, on a shared machine, start-up and
I/O excluded; no peak memory unless the module stored it).  ``selected`` marks, per stage x
step x mode x scope, the row used for totals (measured over recorded).  ``mode`` is ``full``,
``tables_only`` or ``full+tables`` (the same command serves both).  Peak RSS: ``peak_rss_mib``
is the largest single process (``os.wait4`` ``ru_maxrss`` and psutil sampling every 0.25 s),
``peak_rss_tree_mib`` the sampled peak of the summed RSS of the step's process tree.

Usage (from ``biodyn-work``; threads as in the README; run under ``caffeinate -i``)::

    export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \\
        XLA_FLAGS='--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=2'
    caffeinate -i nice -n 10 .venv/bin/python subproject_24_multireference_latent_truth_model/src/r2/reproduce_all.py --tables-only
    .venv/bin/python subproject_24_multireference_latent_truth_model/src/r2/reproduce_all.py --dry-run
    .venv/bin/python subproject_24_multireference_latent_truth_model/src/r2/reproduce_all.py --check-manifest
"""
from __future__ import annotations

import argparse
import ast
import csv
import datetime as dt
import glob
import hashlib
import importlib.metadata as md
import json
import os
import platform
import shlex
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

SP = Path(__file__).resolve().parents[2]                 # subproject root
BW = SP.parent                                           # biodyn-work root
SRC = SP / "src" / "r2"
DATA = SP / "data" / "r2"
TABLES = DATA / "tables"
INPUTS = DATA / "inputs"
OUT = SP / "outputs" / "r2"
PIPE_OUT = OUT / "pipeline"
TIMINGS_CSV = TABLES / "pipeline_stage_timings.csv"
ENV_CSV = TABLES / "pipeline_environment.csv"
MANIFEST = DATA / "MANIFEST.sha256"
REFIT_CSV = TABLES / "pipeline_refit_check.csv"
REQUIREMENTS = SP / "requirements-r2.txt"
REQUIREMENTS_LOCK = SP / "requirements-r2.lock"
RELEASE = SP / "outputs" / "release"
ARCHIVE = RELEASE / "r2_supporting_outputs.tar.gz"
ARCHIVE_LIST = RELEASE / "r2_supporting_outputs_files.sha256"   # sha256 of every archive member
ARCHIVE_SUM = RELEASE / "r2_supporting_outputs.tar.gz.sha256"   # sha256 of the archive itself
ARCHIVE_README = RELEASE / "README_r2_supporting_outputs.md"
TRACE_DIR = SRC / "trace_inputs"                                 # sitecustomize.py of --trace-inputs
PY = sys.executable
EXT = Path(os.environ.get("BIODYN_EXT", "/Volumes/Crucial X6/MacBook/biomechinterp/biodyn-work"))

THREAD_ENV = {"OMP_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2", "MKL_NUM_THREADS": "2",
              "VECLIB_MAXIMUM_THREADS": "2",
              "XLA_FLAGS": "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=2"}
SOURCE_PRIORITY = {"measured_isolated": 0, "recorded_meta": 1, "recorded_table": 1, "recorded_log": 2}
TIMING_COLUMNS = ["stage", "layer", "step", "mode", "scope", "timing_source", "selected", "command",
                  "wall_time_s", "peak_rss_mib", "peak_rss_tree_mib", "cpu_user_s", "cpu_sys_s", "n_units",
                  "exit_code", "loadavg_1m_start", "loadavg_1m_max", "loadavg_1m_end", "n_other_python_procs",
                  "started_at", "finished_at", "log", "note", "analysis_status", "post_hoc_columns"]


# =============================================================================================
# Stage registry
# =============================================================================================
@dataclass
class Req:
    """An input a stage needs. kind: 'external' (raw data outside the repository), 'committed'
    (data/r2/inputs) or 'upstream' (outputs/r2 of an earlier stage)."""
    kind: str
    label: str
    pattern: str            # path or glob (absolute)
    producer: str = ""      # stage that writes it (upstream)

    def ok(self) -> bool:
        return bool(glob.glob(self.pattern)) if any(ch in self.pattern for ch in "*?[") else Path(self.pattern).exists()


@dataclass
class Step:
    name: str
    argv: list[str]
    full: bool = True
    tables: bool = False
    note: str = ""

    @property
    def mode(self) -> str:
        return "full+tables" if (self.full and self.tables) else ("full" if self.full else "tables_only")


@dataclass
class Stage:
    name: str
    layer: str
    label: str
    steps: list[Step]
    needs: list[Req] = field(default_factory=list)
    needs_tables: list[Req] = field(default_factory=list)   # needs of the tables-only steps
    outputs: list[str] = field(default_factory=list)          # data/r2/tables/<name>.csv written
    also_writes: list[str] = field(default_factory=list)      # tables another stage owns that this stage rewrites
    full_cost: str = ""
    note: str = ""

    def steps_for(self, tables_only: bool) -> list[Step]:
        return [s for s in self.steps if (s.tables if tables_only else s.full)]

    def reqs_for(self, tables_only: bool) -> list[Req]:
        return self.needs_tables if (tables_only and self.needs_tables) else self.needs


def _py(script: str, *args: str) -> list[str]:
    return [PY, "-W", "ignore", str(SRC / script), *args]


def _ext(label: str, rel: str) -> Req:
    return Req("external", label, str(EXT / rel))


def _up(label: str, rel: str, producer: str) -> Req:
    return Req("upstream", label, str(OUT / rel), producer)


SHARED_OUTPUTS = ["eval_analysis_decisions"]     # dated decision log, rows replaced per module
TISSUE_ATT = "single_cell_mechinterp/outputs/eval_bias/{m}/*/attention_scores.npy"

STAGES: list[Stage] = [
    # ------------------------------------------------------------------ data layer
    Stage("panels", "data", "Tissue gene panels and universes", [
        Step("build", _py("build_panels.py"), True, False),
        Step("table", _py("build_panels.py", "--table-only"), False, True,
             "universe_panels.csv from the stored panel files (no raw data)")],
        needs=[_ext("scGPT attention matrices", TISSUE_ATT.format(m="scgpt")),
               _ext("Geneformer attention matrices + gene names", "single_cell_mechinterp/outputs/eval_bias/geneformer/*/gene_names.txt"),
               _ext("processed Tabula Sapiens h5ad (scGPT var order)", "single_cell_mechinterp/outputs/atlas/*/tabula_sapiens_*_processed.h5ad")],
        needs_tables=[_up("stored tissue panels", "universe/*/panel_meta.json", "panels")],
        outputs=["universe_panels"], full_cost="seconds"),
    Stage("annotators", "data", "Annotators and integration baselines (committed portable inputs)", [
        Step("build", _py("build_annotators.py", "--from-portable"), True, True)],
        needs=[Req("committed", "portable raw inputs", str(INPUTS / "annotator_raw" / "*")),
               Req("committed", "raw-input manifest", str(INPUTS / "annotator_raw_inputs.json"))],
        outputs=["source_table", "source_components", "lit_ht_like_regulons"], full_cost="~20 s"),
    Stage("evidence", "data", "Observation-masked evidence patterns (4 universes x 9 variants)", [
        Step("build", _py("evidence.py"), True, True)],
        needs=[_up("tissue panels", "universe/*/panel.tsv", "panels"),
               _up("annotator edges", "annotators/annotator_edges.parquet", "annotators")],
        outputs=["evidence_summary"], full_cost="~20 s, 1.1 GB"),
    Stage("grnboost2", "data", "GRNBoost2 / GENIE3 expression baselines", [
        Step("run", ["zsh", str(SRC / "run_grnboost2.sh")], True, False,
             "6 dask workers; the only stage that runs more than one compute process"),
        Step("table", _py("build_grnboost2.py", "table"), False, True),
        Step("summary", _py("build_grnboost2.py", "summary"), False, True)],
        needs=[_ext("Tabula Sapiens source h5ad", "single_cell_mechinterp/data/raw/tabula_sapiens_*.h5ad"),
               _up("tissue panels", "universe/*/panel.tsv", "panels")],
        needs_tables=[_up("GRNBoost2 outputs", "grnboost2/*/grnboost2.parquet", "grnboost2")],
        outputs=["grn_baseline_runtime"], full_cost="~60 min"),
    Stage("fm_scores", "data", "Attention, correlation and random method scores", [
        Step("build", _py("build_fm_scores.py"), True, False),
        Step("table", _py("build_fm_scores.py", "--tables-only"), False, True,
             "fm_* tables from the stored meta.json files (no raw data)")],
        needs=[_ext("scGPT attention matrices", TISSUE_ATT.format(m="scgpt")),
               _ext("Geneformer attention matrices", TISSUE_ATT.format(m="geneformer")),
               _ext("Tabula Sapiens source h5ad", "single_cell_mechinterp/data/raw/tabula_sapiens_*.h5ad"),
               _up("GRNBoost2 cell sample", "grnboost2/*/cells.txt", "grnboost2")],
        needs_tables=[_up("stored method scores", "fm_scores/*/meta.json", "fm_scores")],
        outputs=["fm_universe_summary", "fm_extraction_cells", "fm_attention_null", "fm_float16_error",
                 "fm_float16_rowsum_check", "fm_min_coobs_sensitivity"], full_cost="~1 min"),
    Stage("cohorts", "data", "Held-out perturbation cohorts (DE per perturbation)", [
        Step("build", _py("build_cohorts.py"), True, False, "downloads the Replogle pseudobulk files if missing"),
        Step("table", _py("build_cohorts.py", "--table-only", "--no-download"), False, True)],
        needs=[Req("external", "perturbation h5ad files", str(BW / "single_cell_mechinterp/data/perturb/*/perturb_processed_symbols.h5ad")),
               _ext("Replogle 2022 pseudobulk (downloaded if missing)", "external_data/replogle2022")],
        needs_tables=[_up("cohort DE tables", "cohorts/*/de.parquet", "cohorts")],
        outputs=["cohort_table", "cohort_perturbations", "cohort_universe_overlap"],
        full_cost="~2 min recorded (Replogle download not included)"),
    Stage("heldout_chip", "data", "Held-out ChIP references (ENCODE, ENCODE-strict, ENETS2)", [
        Step("build", _py("build_heldout_chip.py"), True, False),
        Step("table", _py("build_heldout_chip.py", "--table-only"), False, True,
             "the three heldout_chip tables from the stored provenance.json / annotator_overlap.csv (no raw data)")],
        needs=[_ext("ChIP raw downloads (md5-checked; --download)", "external_data/heldout_chip/encodenets_gerstein2012/enets2.Proximal_filtered.txt"),
               _up("annotator outputs", "annotators/heldout_chip_edges.parquet", "annotators"),
               _up("evidence of the four universes", "evidence/global/edge_pattern*.npy", "evidence")],
        needs_tables=[_up("stored ChIP references", "heldout_chip/provenance.json", "heldout_chip"),
                      _up("stored annotator overlap", "heldout_chip/annotator_overlap.csv", "heldout_chip")],
        outputs=["heldout_chip_table", "heldout_chip_coverage", "heldout_chip_annotator_overlap"], full_cost="~1 min"),
    # ------------------------------------------------------------------ model layer
    Stage("fit_primary", "model", "Primary fits: MAP of five estimators, Family-HB NUTS", [
        Step("map", _py("fit_primary.py", "map")),
        Step("laplace", _py("fit_primary.py", "laplace")),
        Step("nuts", _py("fit_primary.py", "nuts", "--init", "random", "map_laplace")),
        Step("tables", _py("fit_primary.py", "tables"), True, True),
        Step("edge-tables", _py("fit_primary.py", "edge-tables"), True, True),
        Step("rerun-check", _py("nuts_rerun_check.py"), True, True,
             "post hoc provenance check (outputs/r2/fits/_superseded_firstpass/); needs the kept first-pass fits")],
        needs=[_up("evidence patterns", "evidence/global/patterns.parquet", "evidence")],
        needs_tables=[_up("stored primary fits", "fits/global/prespecified_cc/family_hb_nuts/cred_mean.npy", "fit_primary")],
        outputs=["fit_convergence", "fit_source_parameters", "fit_class_weights", "fit_hyperparameters",
                 "fit_map_optima", "fit_nuts_chains", "fit_map_vs_nuts", "fit_estimator_agreement"],
        full_cost="MAP 5-20 min, NUTS ~8 h recorded (8 fits)"),
    Stage("fit_sensitivity", "model", "Label-free sensitivity fits, CV, PPC, recovery", [
        Step("fits", _py("fit_sensitivity.py", "fits")),
        Step("nuts", _py("fit_sensitivity.py", "nuts")),
        Step("cv", _py("fit_sensitivity.py", "cv")),
        Step("ppc", _py("fit_sensitivity.py", "ppc")),
        Step("recovery", _py("fit_sensitivity.py", "recovery", "--nuts")),
        Step("truthstart", _py("fit_sensitivity.py", "truthstart")),
        Step("modes", _py("fit_sensitivity.py", "modes")),
        Step("tables", _py("fit_sensitivity.py", "tables"), True, True)],
        needs=[_up("primary fits", "fits/global/prespecified_cc/family_hb/meta.json", "fit_primary")],
        needs_tables=[_up("sensitivity fits", "fits/global/*/family_hb/meta.json", "fit_sensitivity"),
                      _up("CV folds", "fit_sensitivity/cv/global/*.json", "fit_sensitivity")],
        outputs=["fit_family_groupings", "fit_coreporting_assoc", "fit_sensitivity", "fit_sensitivity_params",
                 "fit_nuts_prior_sensitivity", "fit_cv_loglik", "fit_cv_loglik_folds", "fit_ppc", "fit_modes",
                 "fit_sensitivity_nuts_chains"],
        also_writes=["fit_recovery", "fit_recovery_calibration", "fit_recovery_replicates", "fit_recovery_nuts_chainsets"],
        full_cost="~10 h recorded"),
    Stage("obs_models", "model", "Informative-observation models: evidence, MAP, PPC, CV (post hoc)", [
        Step("evidence", _py("informative_observation.py", "evidence")),
        Step("map", _py("informative_observation.py", "map")),
        Step("map_tfw", _py("informative_observation.py", "map_tfw")),
        Step("ppc", _py("informative_observation.py", "ppc")),
        Step("cv", _py("informative_observation.py", "cv"))],
        needs=[_up("evidence patterns", "evidence/global/patterns.parquet", "evidence")],
        full_cost="~10 h (MAP 8.3 h)"),
    Stage("posterior_modes", "model", "Family-HB posterior modes: per-mode NUTS, bridge, simulation (post hoc)", [
        Step("nuts", _py("posterior_modes.py", "nuts")),
        Step("bridge", _py("posterior_modes.py", "bridge")),
        Step("mixture", _py("posterior_modes.py", "mixture")),
        Step("simulate", _py("posterior_modes.py", "simulate")),       # design-faithful simulators (primary)
        Step("simulate-fixed", _py("posterior_modes.py", "simulate", "--universes", "immune", "global",
                                   "immune_truthmode1")),           # fixed-mask sensitivity (modes2, 2026-10-14)
        Step("sensmodes", _py("posterior_modes.py", "sensmodes")),
        Step("clusterstab", _py("posterior_modes.py", "clusterstab"))],
        needs=[_up("primary fits", "fits/global/prespecified_cc/family_hb_nuts/meta.json", "fit_primary")],
        full_cost="~11 h recorded"),
    # ------------------------------------------------------------------ evaluation layer
    Stage("robustness", "evaluation", "Annotator / family / lineage dropout robustness", [
        Step("fit-tissues", _py("robustness.py", "fit", "--universes", "immune", "kidney", "lung")),
        Step("fit-global", _py("robustness.py", "fit", "--universes", "global")),
        Step("search", _py("robustness.py", "search")),
        Step("seedcheck", _py("robustness.py", "seedcheck"), True, False,
             "post hoc (robfix): independent-seed check of the fits that needed extra random starts; adopts a better "
             "optimum if one is found"),
        Step("analyze", _py("robustness.py", "analyze"), True, True)],
        needs=[_up("evidence patterns", "evidence/global/patterns.parquet", "evidence")],
        needs_tables=[_up("stored dropout fits", "robustness/fits/global/full.json", "robustness")],
        outputs=["rob_method_ranking", "rob_edge_stability", "rob_config_detail", "rob_method_scores_full",
                 "rob_fit_convergence", "rob_fit_search", "rob_estimator_contrasts", "rob_mode_switches"],
        full_cost="~5 h recorded"),
    Stage("heldout_eval", "evaluation", "Held-out ChIP and perturbation evaluation", [
        Step("refs", _py("heldout_eval.py", "refs"), True, True),
        Step("estimators", _py("heldout_eval.py", "estimators"), True, True),
        Step("estimators-reported_any", _py("heldout_eval.py", "estimators", "--subset", "reported_any"), True, True),
        Step("decomposition", _py("heldout_eval.py", "decomposition"), True, True),
        Step("degree", _py("heldout_eval.py", "degree"), True, True),
        Step("status", _py("heldout_eval.py", "status"), True, True),
        Step("sensitivity", _py("heldout_eval.py", "sensitivity"), True, True)],
        needs=[_up("primary fits", "fits/global/prespecified_cc/family_hb_nuts/cred_mean.npy", "fit_primary"),
               _up("ChIP references", "heldout_chip/encode_tfbs.parquet", "heldout_chip"),
               _up("cohort DE tables", "cohorts/*/de.parquet", "cohorts"),
               _up("method scores", "fm_scores/*/scores.parquet", "fm_scores")],
        outputs=["heldout_estimators", "heldout_paired", "heldout_methods_context", "heldout_universe_decomposition",
                 "heldout_downsampling_null", "heldout_degree_calibration", "heldout_evidence_status",
                 "heldout_sensitivity_fits"], full_cost="~40 min"),
    Stage("governance", "evaluation", "Label-free BFDR claim governance", [
        Step("all", _py("governance.py", "all"), True, True)],
        needs=[_up("primary NUTS fits", "fits/*/prespecified_cc/family_hb_nuts/cred_mean.npy", "fit_primary")],
        outputs=["gov_tiers", "gov_consistency", "gov_promoted_by_method", "gov_tier_validation", "gov_tier_trend",
                 "gov_concentration", "gov_deconcentration", "gov_deconcentration_eval", "gov_sensitivity_grid",
                 "gov_promoted_edges"], full_cost="~5 min"),
    Stage("modes_tables", "evaluation", "Posterior-mode tier sensitivity and tables (post hoc)", [
        Step("governance", _py("posterior_modes.py", "governance"), True, True),
        Step("tables", _py("posterior_modes.py", "tables"), True, True)],
        needs=[_up("bridge estimates", "posterior_modes/bridge/*.json", "posterior_modes"),
               _up("simulation replicates", "posterior_modes/sim/*", "posterior_modes")],
        outputs=["modes_bridge", "modes_governance_sensitivity", "modes_simulation", "modes_simulation_replicates",
                 "modes_simulation_calibration", "modes_faithful_design", "modes_lit_node_check",
                 "fit_sensitivity_modes", "fit_grouping_stability",
                 "fit_recovery", "fit_recovery_calibration", "fit_recovery_replicates", "fit_recovery_nuts_chainsets"],
        full_cost="minutes", note="after governance (reads gov_tiers.csv)"),
    Stage("replication", "evaluation", "Cross-cohort replication check (locked power design)", [
        Step("all", _py("replication.py", "all"), True, True,
             "keeps the existing power lock; replacing it needs 'power --relock-reason'")],
        needs=[_up("primary NUTS fits", "fits/global/prespecified_cc/family_hb_nuts/cred_mean.npy", "fit_primary"),
               _up("cohort DE tables", "cohorts/*/de.parquet", "cohorts")],
        outputs=["rep_power", "rep_units", "rep_pass_rates", "rep_tf_strata", "rep_threshold_loo", "rep_bayes",
                 "rep_prior_sensitivity", "rep_leavek"], full_cost="~12 min"),
    Stage("tuning", "evaluation", "Optimism of label-based tuning", [
        Step("cv", _py("tuning_optimism.py", "cv")),
        Step("analyze", _py("tuning_optimism.py", "analyze"), True, True)],
        needs=[_up("sensitivity fits", "fits/global/*/family_hb/meta.json", "fit_sensitivity")],
        needs_tables=[_up("prior-variant CV folds", "tuning_optimism/cv/global/*.json", "tuning")],
        outputs=["tune_optimism", "tune_selection_frequency", "tune_weight_sensitivity", "tune_weight_summary",
                 "tune_label_sets", "tune_labelfree_cv", "tune_status_decomposition"],
        full_cost="~30 min", note="after robustness (reads rob_method_ranking.csv)"),
    Stage("obs_eval", "evaluation", "Informative-observation models: held-out evaluation and tables (post hoc)", [
        Step("heldout", _py("informative_observation.py", "heldout"), True, True),
        Step("tables", _py("informative_observation.py", "tables"), True, True),
        Step("check", _py("informative_observation.py", "check"), True, True)],
        needs=[_up("informative-observation fits", "fits/global/prespecified_cc__elig/family_hb_obs/meta.json", "obs_models")],
        outputs=["obs_fit_summary", "obs_map_optima", "obs_params", "obs_ppc", "obs_cv", "obs_heldout",
                 "obs_heldout_paired", "obs_heldout_status"], full_cost="~12 min"),
    Stage("sim_hub", "evaluation", "Hub-bias simulation with known truth (post hoc)", [
        Step("fit-shard0", _py("sim_hub_bias.py", "fit", "--shard", "0/2")),
        Step("fit-shard1", _py("sim_hub_bias.py", "fit", "--shard", "1/2")),
        Step("restart-check", _py("sim_hub_bias.py", "restart-check")),
        Step("oracle-check", _py("sim_hub_bias.py", "oracle-check")),
        Step("metrics", _py("sim_hub_bias.py", "metrics")),
        Step("tables", _py("sim_hub_bias.py", "tables"), True, True)],
        needs=[_up("fit_source_parameters.csv inputs", "fits/global/prespecified_cc/family_hb/meta.json", "fit_primary")],
        needs_tables=[_up("per-replicate metrics", "sim_hub/metrics/*.json", "sim_hub")],
        outputs=["sim_hub_summary", "sim_hub_deciles", "sim_hub_calibration", "sim_hub_contrasts", "sim_hub_modes",
                 "sim_hub_modes_replicates", "sim_hub_by_mode", "sim_hub_restart_check", "sim_hub_obs_signature",
                 "sim_hub_oracle_check", "sim_hub_datasets", "sim_hub_design", "sim_hub_empirical"],
        full_cost="~11 h recorded", note="tables after heldout_eval (heldout_degree_calibration.csv) and obs_models"),
    Stage("compute_scaling", "evaluation", "Compute scaling on synthetic evidence", [
        Step("session-a", _py("compute_scaling.py", "all")),
        Step("session-b", _py("compute_scaling.py", "replicate")),
        Step("tables", _py("compute_scaling.py", "tables"), True, True)],
        needs=[_up("global evidence (template)", "evidence/global/patterns.parquet", "evidence")],
        needs_tables=[_up("stored measurements", "scale/runs/map/*.json", "compute_scaling")],
        outputs=["scale_summary", "scale_compression", "scale_map", "scale_map_restarts", "scale_nuts", "scale_fits",
                 "scale_replicate", "scale_generator_check", "scale_complexity", "scale_environment"],
        full_cost="~6 h recorded"),
    Stage("obs_integration", "evaluation", "Informative-observation credibilities in governance / replication (post hoc)", [
        Step("all", _py("obs_integration.py", "all"), True, True),
        Step("check", _py("obs_integration.py", "check"), True, True)],
        needs=[_up("informative-observation fits", "fits/global/prespecified_cc__elig/family_hb_obs/meta.json", "obs_models")],
        outputs=["obs_governance_tiers", "obs_governance", "obs_governance_trend", "obs_governance_degree",
                 "obs_replication"], full_cost="~1 min",
        note="after governance, heldout_eval degree, replication outcomes"),
    # ------------------------------------------------------------------ manuscript assets
    Stage("paper_assets", "assets", "Manuscript tables (LaTeX), figures, digest and prose numbers", [
        Step("tables", [PY, str(SRC / "make_paper_assets.py"), "tables", "--check-latex"], True, True),
        Step("figures", [PY, str(SRC / "make_paper_assets.py"), "figures"], True, True),
        Step("digest", [PY, str(SRC / "make_paper_assets.py"), "digest"], True, True),
        Step("numbers", [PY, str(SRC / "make_paper_assets.py"), "numbers"], True, True,
             note="paper/r2/numbers.tex (one macro per number of the prose) and the literal-number check")],
        full_cost="~3 min (pdflatex for the table check)"),
]
STAGE_BY_NAME = {s.name: s for s in STAGES}


def stage_table_owner() -> dict[str, str]:
    return {t: s.name for s in STAGES for t in s.outputs}


PIPELINE_TABLES = {"pipeline_stage_timings", "pipeline_environment", "pipeline_refit_check",
                   "pipeline_release_check"}   # the last: written by src/r2/release_check.py


def unclaimed_tables() -> list[str]:
    """CSV files in data/r2/tables that no stage declares (should be empty)."""
    own = set(stage_table_owner()) | set(SHARED_OUTPUTS) | PIPELINE_TABLES
    return sorted(p.stem for p in TABLES.glob("*.csv") if p.stem not in own)


def table_state() -> dict[str, tuple[str, int]]:
    """(sha256, mtime_ns) of every data/r2/tables/*.csv: which tables a step wrote (before / after)."""
    return {p.stem: (sha256(p), p.stat().st_mtime_ns) for p in sorted(TABLES.glob("*.csv"))}


# =============================================================================================
# Running and measuring one step
# =============================================================================================
def child_env(trace: Path | None = None) -> dict:
    env = dict(os.environ)
    for k, v in THREAD_ENV.items():
        env.setdefault(k, v)
    env["PY"] = PY                       # interpreter used by run_grnboost2.sh
    pre = [str(TRACE_DIR)] if trace else []
    env["PYTHONPATH"] = os.pathsep.join(pre + [str(SP / "src")] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
    env["PYTHONUNBUFFERED"] = "1"
    if trace:
        env["R2_TRACE_INPUTS"] = str(trace)
    else:
        env.pop("R2_TRACE_INPUTS", None)
    return env


def _other_python_procs(exclude: set[int]) -> int:
    try:
        import psutil
    except ImportError:
        return -1
    n = 0
    for p in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            if p.info["pid"] in exclude:
                continue
            nm = (p.info["name"] or "").lower()
            cl = " ".join(p.info["cmdline"] or []).lower()
            if "python" in nm and any(x in cl for x in (".py", " -m ")):
                n += 1
        except Exception:
            continue
    return n


class Sampler(threading.Thread):
    """Samples the summed and the largest RSS of a process tree and the 1-min load average."""

    def __init__(self, pid: int, every: float = 0.25):
        super().__init__(daemon=True)
        self.pid, self.every = pid, every
        self.peak_tree = 0.0
        self.peak_single = 0.0
        self.load_max = 0.0
        self.stop = threading.Event()

    def run(self):
        try:
            import psutil
            root = psutil.Process(self.pid)
        except Exception:
            return
        while not self.stop.is_set():
            try:
                procs = [root] + root.children(recursive=True)
                rss = []
                for p in procs:
                    try:
                        rss.append(p.memory_info().rss)
                    except Exception:
                        pass
                if rss:
                    self.peak_tree = max(self.peak_tree, sum(rss) / 2 ** 20)
                    self.peak_single = max(self.peak_single, max(rss) / 2 ** 20)
                self.load_max = max(self.load_max, os.getloadavg()[0])
            except Exception:
                pass
            self.stop.wait(self.every)


def _maxrss_mib(ru) -> float:
    # macOS reports ru_maxrss in bytes, Linux in KiB
    return ru.ru_maxrss / 2 ** 20 if sys.platform == "darwin" else ru.ru_maxrss / 1024


def run_step(argv: list[str], log_path: Path, trace: Path | None = None) -> dict:
    """Run one command in a fresh subprocess; wall time, peak RSS, CPU time, load average.  ``trace``: file
    that receives the files the step opens (``--trace-inputs``)."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    la0 = os.getloadavg()[0]
    n_other = _other_python_procs({os.getpid()})
    started = dt.datetime.now()
    t0 = time.perf_counter()
    with open(log_path, "w") as fh:
        fh.write(f"# {started:%Y-%m-%d %H:%M:%S} {shlex.join(argv)}\n")
        fh.flush()
        if trace:
            trace.parent.mkdir(parents=True, exist_ok=True)
        p = subprocess.Popen(argv, cwd=SP, env=child_env(trace), stdout=fh, stderr=subprocess.STDOUT)
        smp = Sampler(p.pid)
        smp.start()
        _, status, ru = os.wait4(p.pid, 0)
        wall = time.perf_counter() - t0
        smp.stop.set()
        smp.join(2)
        p.returncode = os.waitstatus_to_exitcode(status)
    finished = dt.datetime.now()
    return dict(wall_time_s=round(wall, 2),
                peak_rss_mib=round(max(_maxrss_mib(ru), smp.peak_single), 1),
                peak_rss_tree_mib=round(max(smp.peak_tree, _maxrss_mib(ru)), 1),
                cpu_user_s=round(ru.ru_utime, 2), cpu_sys_s=round(ru.ru_stime, 2), exit_code=p.returncode,
                loadavg_1m_start=round(la0, 2), loadavg_1m_max=round(max(smp.load_max, la0), 2),
                loadavg_1m_end=round(os.getloadavg()[0], 2), n_other_python_procs=n_other,
                started_at=started.isoformat(timespec="seconds"), finished_at=finished.isoformat(timespec="seconds"))


# =============================================================================================
# Timing table
# =============================================================================================
def _key(r: dict) -> tuple:
    return (r["stage"], r["step"], r["mode"], r.get("scope") or "all", r["timing_source"])


def read_timings(path: Path | None = None) -> list[dict]:
    path = path or TIMINGS_CSV
    if not path.exists():
        return []
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def _stage_order() -> dict[str, int]:
    return {s.name: i for i, s in enumerate(STAGES)}


def _step_order() -> dict[tuple, int]:
    return {(s.name, st.name): j for s in STAGES for j, st in enumerate(s.steps)}


def write_timings(rows: list[dict], path: Path | None = None) -> Path:
    """Upsert ``rows`` (by stage, step, mode, scope, timing source), mark the selected row per
    stage x step x mode x scope (measured over recorded), sort in pipeline order and write."""
    path = path or TIMINGS_CSV
    steps = {(s.name, st.name): st for s in STAGES for st in s.steps}
    cur = {}
    for r in read_timings(path) + [{c: r.get(c, "") for c in TIMING_COLUMNS} for r in rows]:
        r["scope"] = r["scope"] or "all"
        # a step that no longer serves the tables-only run (e.g. the data builds that read external raw data,
        # replaced there by table steps): its 'full+tables' rows are full-run rows
        st = steps.get((r["stage"], r["step"]))
        if st is not None and r["mode"] == "full+tables" and st.mode == "full":
            r["mode"] = "full"
        cur[_key(r)] = r
    allr = list(cur.values())
    best: dict[tuple, dict] = {}
    for r in allr:
        if str(r.get("exit_code", "0")) not in ("0", "", "0.0"):
            continue
        g = (r["stage"], r["step"], r["mode"], r["scope"])
        if g not in best or SOURCE_PRIORITY.get(r["timing_source"], 9) < SOURCE_PRIORITY.get(best[g]["timing_source"], 9):
            best[g] = r
    # a selected whole-step row (scope 'all') supersedes the per-scope rows of the same step and mode group
    # (full vs tables-only), so that a future full run is not counted twice next to per-universe records
    grp = {"full": "full", "full+tables": "full", "tables_only": "tables"}
    whole = {(r["stage"], r["step"], grp.get(r["mode"], r["mode"])) for g, r in best.items() if g[3] == "all"}
    best = {g: r for g, r in best.items()
            if g[3] == "all" or (r["stage"], r["step"], grp.get(r["mode"], r["mode"])) not in whole}
    sel = {id(r) for r in best.values()}
    so, sto = _stage_order(), _step_order()
    for r in allr:
        r["selected"] = "True" if id(r) in sel else "False"
        r["analysis_status"] = r.get("analysis_status") or "prespecified"
    allr.sort(key=lambda r: (so.get(r["stage"], 99), sto.get((r["stage"], r["step"]), 99),
                             {"full": 0, "full+tables": 1, "tables_only": 2}.get(r["mode"], 3), r["scope"],
                             SOURCE_PRIORITY.get(r["timing_source"], 9)))
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=TIMING_COLUMNS, lineterminator="\n")
        w.writeheader()
        for r in allr:
            w.writerow({c: r.get(c, "") for c in TIMING_COLUMNS})
    return path


def stage_summary(rows: list[dict] | None = None) -> list[dict]:
    """Per stage: full-run and tables-only wall time (sum of selected steps) and peak RSS (max)."""
    rows = read_timings() if rows is None else rows
    out = []
    for s in STAGES:
        rec = dict(stage=s.name, layer=s.layer, label=s.label)
        for mode, modes in (("full", ("full", "full+tables")), ("tables", ("tables_only", "full+tables"))):
            want = {st.name for st in s.steps if (st.full if mode == "full" else st.tables)}
            sel = [r for r in rows if r["stage"] == s.name and r["selected"] == "True" and r["mode"] in modes
                   and r["step"] in want]
            covered = {r["step"] for r in sel}
            wall = sum(float(r["wall_time_s"]) for r in sel if r["wall_time_s"] not in ("", "nan"))
            rss = [float(r["peak_rss_mib"]) for r in sel if r["peak_rss_mib"] not in ("", "nan")]
            rec[f"{mode}_wall_s"] = round(wall, 1) if sel else None
            rec[f"{mode}_peak_rss_mib"] = round(max(rss), 1) if rss else None
            rec[f"{mode}_sources"] = "+".join(sorted({r["timing_source"] for r in sel}))
            rec[f"{mode}_steps_missing"] = ",".join(sorted(want - covered))
        out.append(rec)
    return out


# =============================================================================================
# Recorded runtimes of the original runs (modules' meta files and run tables)
# =============================================================================================
def _jload(p: Path) -> dict:
    try:
        return json.loads(Path(p).read_text())
    except Exception:
        return {}


def _sum_runtime(files, key=("runtime_s",)) -> tuple[float, int]:
    tot, n = 0.0, 0
    for f in files:
        m = _jload(f)
        v = m
        for k in key:
            v = v.get(k) if isinstance(v, dict) else None
        if isinstance(v, (int, float)):
            tot += float(v)
            n += 1
    return tot, n


def _done_sum(files, step: str) -> tuple[float, int]:
    """Sum of the 'done <step> in <N>s' / 'done <step> (<N>s)' lines (one per completed invocation)."""
    import re
    rx = re.compile(rf"done {re.escape(step)} (?:in )?\(?([0-9.]+)s\)?")
    tot, n = 0.0, 0
    for f in files:
        try:
            txt = Path(f).read_text(errors="replace")
        except OSError:
            continue
        for m in rx.finditer(txt):
            tot += float(m.group(1))
            n += 1
    return tot, n


def _rec(stage, step, wall, n, source, note, scope="all", rss=None, command=None) -> dict:
    st = STAGE_BY_NAME[stage]
    stp = next(x for x in st.steps if x.name == step)
    return dict(stage=stage, layer=st.layer, step=step, mode=stp.mode, scope=scope, timing_source=source,
                command=command or shlex.join(_display(stp.argv)), wall_time_s=round(wall, 1),
                peak_rss_mib=("" if rss is None else round(rss, 1)), n_units=n, exit_code=0, note=note)


def collect_recorded() -> list[dict]:
    """Runtimes stored by the modules when the steps originally ran.  Not measured here."""
    rows: list[dict] = []
    F = OUT / "fits"
    NOTE = ("sum of the runtimes the module recorded per unit at the original run (units partly in "
            "parallel, shared machine; start-up and I/O excluded)")
    # grnboost2: run table written by the module (runtime and peak RSS of every run)
    gt = TABLES / "grn_baseline_runtime.csv"
    if gt.exists():
        with open(gt, newline="") as fh:
            g = list(csv.DictReader(fh))
        wall = sum(float(r["runtime_total_s"]) for r in g)
        rss = max(float(r["peak_rss_total_mib"]) for r in g)
        rows.append(_rec("grnboost2", "run", wall, len(g), "recorded_table",
                         f"grn_baseline_runtime.csv: runtime_total_s summed over {len(g)} runs, peak RSS = largest "
                         "process-tree total (6 dask workers); validate-es not included", rss=rss))
    # cohorts: wall_time_s / peak_rss_gb_process in every cohort meta.json
    cm = sorted((OUT / "cohorts").glob("*/meta.json"))
    if cm:
        ms = [_jload(f) for f in cm]
        wall = sum(float(m.get("wall_time_s") or 0) for m in ms)
        rss = max(float(m.get("peak_rss_gb_process") or 0) for m in ms) * 1024
        rows.append(_rec("cohorts", "build", wall, len(ms), "recorded_meta",
                         "outputs/r2/cohorts/*/meta.json wall_time_s summed over cohorts; peak RSS = largest cohort process",
                         rss=rss))
    # fit_primary: MAP fits (5 estimators x 4 universes) and NUTS fits (per universe x init)
    maps = [F / u / "prespecified_cc" / m / "meta.json" for u in ("global", "immune", "kidney", "lung")
            for m in ("em", "dependence", "multifactor", "family_regularized", "family_hb")]
    wall, n = _sum_runtime([p for p in maps if p.exists()])
    if n:
        rows.append(_rec("fit_primary", "map", wall, n, "recorded_meta", "meta.json runtime_s of the 20 MAP fits; " + NOTE))
    for u in ("global", "immune", "kidney", "lung"):
        for fid, init in (("family_hb_nuts", "map_laplace"), ("family_hb_randinit_nuts", "random")):
            w, n = _sum_runtime([F / u / "prespecified_cc" / fid / "meta.json"])
            if n:
                rows.append(_rec("fit_primary", "nuts", w, n, "recorded_meta",
                                 f"meta.json runtime_s of {u}/{fid} (4 x 1000/1000)", scope=f"{u}, {init} init",
                                 command=shlex.join(_display(_py("fit_primary.py", "nuts", "--universes", u, "--init", init)))))
    # fit_sensitivity
    sens_fits = [p for u in ("global", "immune") for p in (F / u).glob("*/*/meta.json")
                 if p.parent.parent.name != "prespecified_cc__elig"
                 and not p.parent.name.endswith("_nuts") and p.parent.name not in
                 ("family_hb_mixture",) and (p.parent.parent.name != "prespecified_cc" or "__" in p.parent.name)]
    w, n = _sum_runtime(sens_fits)
    if n:
        rows.append(_rec("fit_sensitivity", "fits", w, n, "recorded_meta",
                         "meta.json runtime_s of the evidence-variant, grouping and prior MAP fits; " + NOTE))
    nuts_pr = [p for p in (F / "immune" / "prespecified_cc").glob("family_hb__*_nuts/meta.json")]
    w, n = _sum_runtime(nuts_pr)
    if n:
        rows.append(_rec("fit_sensitivity", "nuts", w, n, "recorded_meta", "prior-sensitivity NUTS fits; " + NOTE))
    for step, pat in (("cv", "cv/*/*.json"), ("modes", "modes/*/optima.json")):
        w, n = _sum_runtime(sorted((OUT / "fit_sensitivity").glob(pat)))
        if n:
            rows.append(_rec("fit_sensitivity", step, w, n, "recorded_meta", NOTE))
    # MAP recovery replicates are stored as rep*.npz without a runtime: the 'done recovery in N s' line of each
    # recovery_<universe>.log covers one universe's replicates; NUTS replicates store runtime_s in rep*.json
    n_map_rep = len(sorted((OUT / "fit_sensitivity" / "recovery").glob("*/rep*.npz")))
    w1, n1 = _done_sum(sorted((OUT / "fit_sensitivity" / "logs").glob("recovery_*.log")), "recovery")
    frn = sorted((OUT / "fit_sensitivity" / "recovery_nuts").glob("*/rep*.json"))
    w2, n2 = _sum_runtime(frn)
    if n1 + n2:
        rows.append(_rec("fit_sensitivity", "recovery", w1 + w2, n_map_rep + n2, "recorded_meta",
                         f"{n_map_rep} MAP recovery replicates ({n1} 'done recovery' lines of "
                         f"outputs/r2/fit_sensitivity/logs/recovery_*.log, {w1:.0f} s) and {n2} NUTS recovery "
                         f"replicates (runtime_s of recovery_nuts/*/rep*.json, {w2:.0f} s); " + NOTE))
    w, n = _sum_runtime(sorted((OUT / "fit_sensitivity" / "recovery").glob("*/truthstart_rep*.json")))
    if n:
        rows.append(_rec("fit_sensitivity", "truthstart", w, n, "recorded_meta", NOTE))
    LOGNOTE = "sum of the 'done ... in N s' lines of the module logs (completed invocations; shared machine)"
    w, n = _done_sum(sorted((OUT / "fit_sensitivity" / "logs").glob("ppc_*.log")), "ppc")
    if n:
        rows.append(_rec("fit_sensitivity", "ppc", w, n, "recorded_log", "outputs/r2/fit_sensitivity/logs/ppc_*.log; " + LOGNOTE))
    for step in ("ppc", "cv"):
        w, n = _done_sum(sorted((OUT / "informative_observation" / "logs").glob(f"{step}_*.log")), step)
        if n:
            rows.append(_rec("obs_models", step, w, n, "recorded_log",
                             f"outputs/r2/informative_observation/logs/{step}_*.log; " + LOGNOTE))
    # simulate*.log = the fixed-mask simulations (2026-10-08/13); modes2_sim_*.log = the design-faithful ones (2026-10-14)
    for step, pat in (("mixture", "mixture*.log"), ("simulate-fixed", "simulate*.log"), ("simulate", "modes2_sim_*.log"),
                      ("clusterstab", "clusterstab*.log")):
        w, n = _done_sum(sorted((OUT / "posterior_modes" / "logs").glob(pat)), step.split("-")[0])
        if n:
            rows.append(_rec("posterior_modes", step, w, n, "recorded_log",
                             f"outputs/r2/posterior_modes/logs/{pat}; " + LOGNOTE))
    # simulation metrics: last logged time of every (scenario, replicate) = one complete pass
    rl = OUT / "sim_hub" / "run.log"
    if rl.exists():
        import re
        last = {}
        for m in re.finditer(r"\[metrics\] (\S+) rep(\d+):.*?([0-9.]+)s\s*$", rl.read_text(errors="replace"), re.M):
            last[(m.group(1), m.group(2))] = float(m.group(3))
        if last:
            rows.append(_rec("sim_hub", "metrics", sum(last.values()), len(last), "recorded_log",
                             "outputs/r2/sim_hub/run.log: last logged time of every scenario x replicate (one pass)"))
    # informative-observation MAP fits
    obs = sorted(F.glob("*/prespecified_cc__elig/*/meta.json"))
    w_map, n_map = _sum_runtime([p for p in obs if not p.parent.name.endswith("__tfw")])
    w_tfw, n_tfw = _sum_runtime([p for p in obs if p.parent.name.endswith("__tfw")])
    if n_map:
        rows.append(_rec("obs_models", "map", w_map, n_map, "recorded_meta", "family_hb_obs / em_obs MAP fits; " + NOTE))
    if n_tfw:
        rows.append(_rec("obs_models", "map_tfw", w_tfw, n_tfw, "recorded_meta", "__tfw sensitivity fits; " + NOTE))
    # posterior modes
    pm = sorted(F.glob("*/prespecified_cc/family_hb_nuts_mode*/meta.json"))
    w, n = _sum_runtime(pm)
    if n:
        rows.append(_rec("posterior_modes", "nuts", w, n, "recorded_meta", "per-mode NUTS fits; " + NOTE))
    for step, pat in (("bridge", "bridge/*.json"), ("sensmodes", "sensmodes/*/*.json")):
        w, n = _sum_runtime(sorted((OUT / "posterior_modes").glob(pat)))
        if n:
            rows.append(_rec("posterior_modes", step, w, n, "recorded_meta", NOTE))
    # simulation replicates: fixed-mask labels (sensitivity) vs design-faithful labels (*_cc, *_ccs; primary)
    fixed_labels = ("immune", "global", "immune_truthmode1")
    sim_dirs = sorted(p for p in (OUT / "posterior_modes" / "sim").glob("*") if p.is_dir())
    for step, dirs in (("simulate", [d for d in sim_dirs if d.name not in fixed_labels]),
                       ("simulate-fixed", [d for d in sim_dirs if d.name in fixed_labels])):
        w, n = _sum_runtime([f for d in dirs for f in sorted(d.glob("rep[0-9][0-9][0-9].json"))])
        if n:
            rows.append(_rec("posterior_modes", step, w, n, "recorded_meta", NOTE))
    # robustness dropout fits: per-model runtime_s in every configuration file
    for step, us in (("fit-tissues", ("immune", "kidney", "lung")), ("fit-global", ("global",))):
        tot, n = 0.0, 0
        for u in us:
            for f in sorted((OUT / "robustness" / "fits" / u).glob("*.json")):
                for mm in (_jload(f).get("models") or {}).values():
                    v = ((mm or {}).get("summary") or {}).get("runtime_s")
                    if isinstance(v, (int, float)):
                        tot += v
                        n += 1
        if n:
            rows.append(_rec("robustness", step, tot, n, "recorded_meta",
                             "runtime_s per model x configuration (incl. adaptive search starts); " + NOTE))
    # robustness adaptive search: start ('--- <date> ...') and end ('exit=0 <date>') lines of each run in its log
    sl = OUT / "robustness" / "fit_fixes_search.log"
    if sl.exists():
        import re
        fmt = "%a %b %d %H:%M:%S %Y"

        def _t(m):
            return dt.datetime.strptime(" ".join(m.group(1).split()) + " " + m.group(2), fmt)
        txt = sl.read_text(errors="replace")
        starts = [_t(m) for m in re.finditer(r"^--- (\w{3} \w{3} +\d+ [\d:]+) \w+ (\d{4})", txt, re.M)]
        ends = [_t(m) for m in re.finditer(r"^exit=0 (\w{3} \w{3} +\d+ [\d:]+) \w+ (\d{4})", txt, re.M)]
        pairs = [(a_, b_) for a_, b_ in zip(starts, ends) if b_ >= a_]
        if pairs:
            # 'fit' itself calls refit_incomplete_search, and a refit replaces the stored per-model runtime_s, so
            # the search time is already in fit-tissues / fit-global; the logged runs are not added (0 s)
            logged = sum((b_ - a_).total_seconds() for a_, b_ in pairs)
            rows.append(_rec("robustness", "search", 0.0, len(pairs), "recorded_log",
                             "counted in fit-tissues / fit-global: 'fit' runs the adaptive search "
                             "(refit_incomplete_search) and the refits' runtime_s replaces the stored one; the "
                             f"{len(pairs)} logged runs in outputs/r2/robustness/fit_fixes_search.log ({logged:.0f} s: "
                             "search with cap 25, extension to cap 100, a diagnostic refit) are not added"))
    # robustness independent-seed check (post hoc): runtime_s of every entry of seedcheck/<universe>.json
    tot, n = 0.0, 0
    for f in sorted((OUT / "robustness" / "seedcheck").glob("*.json")):
        for e in (_jload(f) or {}).values():
            v = (e or {}).get("runtime_s") if isinstance(e, dict) else None
            if isinstance(v, (int, float)):
                tot += v
                n += 1
    if n:
        rows.append(_rec("robustness", "seedcheck", tot, n, "recorded_meta",
                         "runtime_s of the fresh-start checks in outputs/r2/robustness/seedcheck/*.json (an adopted "
                         "optimum also adds its check time to the fit's runtime_s); " + NOTE))
    # tuning CV folds
    w, n = _sum_runtime(sorted((OUT / "tuning_optimism" / "cv").glob("*/*.json")))
    if n:
        rows.append(_rec("tuning", "cv", w, n, "recorded_meta", "prior-variant CV fold fits; " + NOTE))
    # simulation fits
    # the shards split the (scenario, replicate) list as sim_hub_bias.shard_items: k-th item -> shard k % 2
    pairs = [(sc, r) for sc in ("null", "bio_hub", "copying") for r in range(20)]
    for i in (0, 1):
        mine = {f"{sc}/rep{r:02d}" for k, (sc, r) in enumerate(pairs) if k % 2 == i}
        sf = [f for f in sorted((OUT / "sim_hub" / "fits").glob("*/rep*/*/meta.json"))
              if f"{f.parents[2].name}/{f.parents[1].name}" in mine]
        w, n = _sum_runtime(sf)
        if n:
            rows.append(_rec("sim_hub", f"fit-shard{i}", w, n, "recorded_meta",
                             f"meta.json runtime_s of the simulation fits of shard {i}/2; " + NOTE))
    rc = sorted((OUT / "sim_hub" / "restart_check").glob("*/*/*.json"))
    w, n = _sum_runtime(rc)
    if n:
        rows.append(_rec("sim_hub", "restart-check", w, n, "recorded_meta", NOTE))
    # compute scaling: wall time of every stored measurement subprocess, per session
    for step, d in (("session-a", "runs"), ("session-b", "runs_b")):
        files = sorted((OUT / "scale" / d).glob("*/*.json"))
        tot, n, rss = 0.0, 0, 0.0
        for f in files:
            m = _jload(f)
            v = m.get("wall_total_s")
            if v is None and m.get("status") == "measured" and isinstance(m.get("t_nuts_s"), (int, float)):
                v = m["t_nuts_s"] + float(m.get("t_probe_s") or 0.0)
            if isinstance(v, (int, float)):
                tot += v
                n += 1
            r_ = m.get("peak_rss_mib", m.get("peak_rss_nuts_mib"))
            if isinstance(r_, (int, float)):
                rss = max(rss, r_)
        if n:
            rows.append(_rec("compute_scaling", step, tot, n, "recorded_meta",
                             f"outputs/r2/scale/{d}/*/*.json measurement wall times summed (records without a stored "
                             "total, e.g. projections, excluded)", rss=rss or None))
    return rows


# =============================================================================================
# Environment, requirements, manifest
# =============================================================================================
def _sh(cmd: list[str]) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=20).stdout.strip()
    except Exception:
        return ""


def imported_distributions() -> dict[str, str]:
    """{distribution: version} of the third-party modules imported anywhere in src/r2."""
    mods: set[str] = set()
    for f in SRC.rglob("*.py"):
        try:
            tree = ast.parse(f.read_text())
        except SyntaxError:
            continue
        for n in ast.walk(tree):
            if isinstance(n, ast.Import):
                mods.update(a.name.split(".")[0] for a in n.names)
            elif isinstance(n, ast.ImportFrom) and n.level == 0 and n.module:
                mods.add(n.module.split(".")[0])
    std = set(sys.stdlib_module_names) | {"r2", "__future__"}
    pkg = md.packages_distributions()
    out = {}
    for m in sorted(mods - std):
        for dist in pkg.get(m, []):
            try:
                out[dist] = md.version(dist)
            except md.PackageNotFoundError:
                pass
    # used without an import statement: parquet engine and the jax backend
    for dist in ("pyarrow", "jaxlib", "dask"):
        try:
            out[dist] = md.version(dist)
        except md.PackageNotFoundError:
            pass
    return dict(sorted(out.items(), key=lambda kv: kv[0].lower()))


OPTIONAL_DISTS = {"omnipath": "only build_annotators --refresh (re-download)",
                  "arboreto": "grnboost2 stage", "distributed": "grnboost2 stage", "dask": "grnboost2 stage",
                  "scanpy": "cohorts stage", "anndata": "cohorts stage", "statsmodels": "cohorts stage + tests",
                  "pytest": "tests", "pillow": "figure tests", "matplotlib": "paper assets"}


def write_requirements(path: Path | None = None) -> Path:
    path = path or REQUIREMENTS
    d = imported_distributions()
    lines = ["# Pinned versions of the third-party packages imported by src/r2 (SAGMB R2 pipeline).",
             f"# Generated by src/r2/reproduce_all.py --write-requirements on {dt.date.today()} from the",
             f"# interpreter {PY} (Python {platform.python_version()}).",
             "# Install: python -m pip install -r requirements-r2.txt   (Python 3.11).",
             "# Not pip-installable: pdflatex (paper_assets --check-latex) and zsh (grnboost2 stage).",
             "# The early-stopping replay of the grnboost2 stage (validate-es) also uses a second interpreter",
             "# with scikit-learn <= 1.2 (R2_SKLEARN_REF_PYTHON); see README."]
    for dist, ver in d.items():
        note = OPTIONAL_DISTS.get(dist.lower())
        lines.append(f"{dist}=={ver}" + (f"    # {note}" if note else ""))
    path.write_text("\n".join(lines) + "\n")
    write_lock(path.with_suffix(".lock"))          # requirements-r2.txt -> requirements-r2.lock
    return path


def _canon(name: str) -> str:
    import re
    return re.sub(r"[-_.]+", "-", name).lower()


def dependency_closure(roots: list[str]) -> dict[str, str]:
    """{distribution: version} of ``roots`` and every installed distribution they require (unconditional
    requirements and those whose environment marker holds here; extras are not followed)."""
    import re
    try:
        from packaging.requirements import Requirement
    except ImportError:                                   # pragma: no cover
        Requirement = None
    seen: dict[str, str] = {}
    todo = list(roots)
    while todo:
        name = todo.pop()
        try:
            dist = md.distribution(name)
        except md.PackageNotFoundError:
            continue
        key = _canon(dist.metadata["Name"])
        if key in seen:
            continue
        seen[key] = f"{dist.metadata['Name']}=={dist.version}"
        for req in dist.requires or []:
            if Requirement is not None:
                try:
                    r = Requirement(req)
                except Exception:
                    continue
                if r.marker is not None:
                    try:
                        if not r.marker.evaluate({"extra": ""}):
                            continue
                    except Exception:
                        continue
                todo.append(r.name)
            else:
                if ";" in req:
                    continue
                todo.append(re.split(r"[ <>=!~\[(]", req, 1)[0])
    return dict(sorted(seen.items()))


def write_lock(path: Path | None = None) -> Path:
    """``requirements-r2.lock``: the pinned direct requirements plus their installed dependencies."""
    path = path or REQUIREMENTS_LOCK
    closure = dependency_closure(list(imported_distributions()))
    lines = ["# Full lock of the SAGMB R2 pipeline environment: the packages of requirements-r2.txt and every",
             "# installed distribution they require (transitively; extras not followed).",
             f"# Generated by src/r2/reproduce_all.py --write-requirements on {dt.date.today()} from",
             f"# {Path(PY).resolve()} (Python {platform.python_version()}; .venv/bin/python resolves to this",
             "# interpreter, whose site-packages provide the packages).",
             "# Install: python -m pip install -r requirements-r2.lock"]
    path.write_text("\n".join(lines + list(closure.values())) + "\n")
    return path


def environment_rows() -> list[tuple[str, str]]:
    rows = [("written", dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S %Z").strip())]
    if sys.platform == "darwin":
        for k, cmd in (("cpu", "machdep.cpu.brand_string"), ("hw.ncpu", "hw.ncpu"),
                       ("hw.physicalcpu", "hw.physicalcpu"),
                       ("hw.perflevel0.physicalcpu (performance cores)", "hw.perflevel0.physicalcpu"),
                       ("hw.perflevel1.physicalcpu (efficiency cores)", "hw.perflevel1.physicalcpu"),
                       ("hw.memsize (bytes)", "hw.memsize")):
            rows.append((k, _sh(["sysctl", "-n", cmd])))
        rows.append(("os", " ".join(_sh(["sw_vers", f"-{k}"]) for k in ("productName", "productVersion", "buildVersion"))))
    else:
        rows.append(("cpu", platform.processor()))
        rows.append(("hw.ncpu", str(os.cpu_count())))
        rows.append(("os", platform.platform()))
    try:
        mem = int(rows[[k for k, _ in rows].index("hw.memsize (bytes)")][1])
        rows.append(("RAM (GiB)", f"{mem / 2 ** 30:.0f}"))
    except Exception:
        pass
    rows += [("platform", platform.platform()), ("python", platform.python_version()),
             ("python executable", PY), ("python executable (resolved)", str(Path(PY).resolve())),
             ("python prefix", sys.prefix)]
    for dist, ver in imported_distributions().items():
        rows.append((f"package {dist}", ver))
    rows.append(("pdflatex", (_sh(["pdflatex", "--version"]).splitlines() or [""])[0]))
    for k in THREAD_ENV:
        rows.append((f"env {k} (children)", child_env().get(k, "")))
    rows.append(("niceness of the orchestrator", str(os.nice(0))))
    rows.append(("execution", "one step subprocess at a time (sequential); machine shared with other users' jobs "
                              "(load average recorded per step in pipeline_stage_timings.csv)"))
    return rows


def write_environment(path: Path | None = None) -> Path:
    path = path or ENV_CSV
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["key", "value", "analysis_status"])
        for k, v in environment_rows():
            w.writerow([k, v, "prespecified"])
    return path


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def manifest_files() -> list[Path]:
    files = [p for d in (INPUTS, TABLES) for p in d.rglob("*") if p.is_file() and p.name != ".DS_Store"]
    return sorted(files, key=lambda p: str(p.relative_to(DATA)))


def write_manifest(path: Path | None = None) -> Path:
    path = path or MANIFEST
    lines = [f"{sha256(p)}  {p.relative_to(DATA)}" for p in manifest_files()]
    path.write_text("\n".join(lines) + "\n")
    return path


def check_manifest(path: Path | None = None) -> list[str]:
    """Problems ([] = every listed file exists with its checksum and no file is unlisted)."""
    path = path or MANIFEST
    if not path.exists():
        return [f"{path} missing"]
    listed = {}
    for ln in path.read_text().splitlines():
        if ln.strip():
            h, rel = ln.split("  ", 1)
            listed[rel] = h
    probs = []
    for rel, h in listed.items():
        p = DATA / rel
        if not p.exists():
            probs.append(f"missing: {rel}")
        elif sha256(p) != h:
            probs.append(f"checksum differs: {rel}")
    for p in manifest_files():
        if str(p.relative_to(DATA)) not in listed:
            probs.append(f"not in manifest: {p.relative_to(DATA)}")
    return probs


# =============================================================================================
# Supporting archive: the stored outputs the tables-only path reads (traced), for a fresh clone
# =============================================================================================
def _step_rank() -> dict[str, int]:
    return {f"{s.name}__{st.name}": i for i, (s, st) in enumerate((s, st) for s in STAGES for st in s.steps)}


def traced_inputs(run_dir: Path) -> dict:
    """Read the per-step trace files of a ``--trace-inputs`` run (``<run_dir>/trace/<stage>__<step>.txt``) in
    pipeline order.  Returns ``needed`` (files under outputs/r2 read before any step of the run wrote them:
    the stored outputs the run depends on), ``written`` (files under outputs/r2 the run wrote), ``listed``
    (directories under outputs/r2 that were listed) and ``outside`` (files read outside the subproject other
    than the interpreter's: a fresh clone does not have them)."""
    rank = _step_rank()
    files = sorted((Path(run_dir) / "trace").glob("*.txt"), key=lambda f: rank.get(f.stem, 10 ** 6))
    out_root = str(OUT) + os.sep
    sp_root = str(SP) + os.sep
    needed, written, listed, outside, outside_env = {}, set(), set(), {}, {}
    for f in files:
        for ln in f.read_text(errors="replace").splitlines():
            parts = ln.split("\t", 2)
            if len(parts) != 3:
                continue
            kind, _, path = parts
            if path.startswith(out_root):
                if kind == "W":
                    written.add(path)
                elif kind == "R" and path not in written and Path(path).is_file():
                    needed.setdefault(path, f.stem)
                elif kind == "L":
                    listed.add(path)
            elif kind == "W":
                written.add(path)
            elif kind == "R" and not path.startswith(sp_root) and path not in written and Path(path).is_file():
                (outside_env if ENV_PATH.search(path) else outside).setdefault(path, f.stem)
    return dict(needed=needed, written=written, listed=listed, outside=outside, outside_env=outside_env)


# files outside the repository that belong to the Python environment or to library caches / settings, not data
import re as _re  # noqa: E402
ENV_PATH = _re.compile(r"/(site-packages|dist-packages)/|/\.(matplotlib|bokeh|cache|config|jupyter|keras|local)/|"
                       r"/arviz_data/|/\.condarc$")


# Stored outputs a tables-only step tests for existence without opening them (the trace records only files
# that are opened): sim_hub_bias.py counts the fitted replicates by their meta.json and reads params.npz only
# where meta.json marks a complete fit (sim_hub_design / sim_hub_obs_signature).  Found by the fresh
# release-copy check (src/r2/release_check.py); globs relative to the subproject.
ARCHIVE_EXTRA = ["outputs/r2/sim_hub/fits/*/rep*/*/meta.json", "outputs/r2/sim_hub/fits/*/rep*/*/params.npz"]


def build_release_archive(run_dir: Path, extra: list[str] | None = None) -> dict:
    """Write ``outputs/release/r2_supporting_outputs.tar.gz``: the stored outputs (``needed`` of
    ``traced_inputs``) plus ``extra`` paths (relative to the subproject), with members relative to the
    subproject root, a member list with sha256 (also inside the archive as
    ``outputs/r2/SUPPORTING_OUTPUTS.sha256``), the archive's own sha256 and a README."""
    import tarfile
    tr = traced_inputs(run_dir)
    members = {Path(p_).relative_to(SP).as_posix() for p_ in tr["needed"]}
    n_traced = len(members)
    for e in ARCHIVE_EXTRA + list(extra or []):
        for q in (sorted(SP.glob(e)) if any(ch in e for ch in "*?[") else [SP / e]):
            members |= {x.relative_to(SP).as_posix() for x in ([q] if q.is_file() else q.rglob("*")) if x.is_file()}
    members = sorted(m for m in members if Path(m).name != ".DS_Store")
    RELEASE.mkdir(parents=True, exist_ok=True)
    lines = [f"{sha256(SP / m)}  {m}" for m in members]
    inner = OUT / "SUPPORTING_OUTPUTS.sha256"
    inner.write_text("\n".join(lines) + "\n")
    ARCHIVE_LIST.write_text(inner.read_text())
    tmp = ARCHIVE.with_suffix(".tmp")

    def _norm(ti):
        ti.uid = ti.gid = 0
        ti.uname = ti.gname = ""
        return ti
    with tarfile.open(tmp, "w:gz", compresslevel=6) as tf:
        for m in members + [inner.relative_to(SP).as_posix()]:
            tf.add(SP / m, arcname=m, recursive=False, filter=_norm)
    tmp.replace(ARCHIVE)
    h = sha256(ARCHIVE)
    ARCHIVE_SUM.write_text(f"{h}  {ARCHIVE.name}\n")
    size = sum((SP / m).stat().st_size for m in members)
    by_dir: dict[str, list] = {}
    for m in members:
        k = "/".join(m.split("/")[:3])
        by_dir.setdefault(k, [0, 0])
        by_dir[k][0] += 1
        by_dir[k][1] += (SP / m).stat().st_size
    info = dict(n_files=len(members), n_traced=n_traced, extra_patterns=ARCHIVE_EXTRA + list(extra or []),
                bytes_uncompressed=size, bytes_archive=ARCHIVE.stat().st_size, sha256=h,
                by_dir=by_dir, outside=tr["outside"], outside_env=tr["outside_env"], run_dir=str(run_dir))
    write_archive_readme(info)
    return info


def write_archive_readme(info: dict) -> Path:
    gib = 2 ** 30
    rows = "\n".join(f"| `{k}/` | {v[0]:,} | {v[1] / 2 ** 20:,.1f} |" for k, v in sorted(info["by_dir"].items()))
    txt = f"""# Supporting outputs of the SAGMB R2 analysis (tables-only reproduction)

`r2_supporting_outputs.tar.gz` holds the stored outputs that `src/r2/reproduce_all.py --tables-only` reads
and that the repository does not contain (`outputs/r2/` is not committed): the fitted models (MAP optima,
NUTS summaries), the robustness dropout fits, the simulation replicates and metrics, the compute-scaling
measurements, the cohort differential-expression tables, the method scores, the held-out ChIP references
and the other intermediate files the table steps read. With it, every table in `data/r2/tables/` is
regenerated without refitting a model and without the external raw data (Tabula Sapiens, attention
matrices, perturbation and ChIP downloads).

- Archive: `{ARCHIVE.name}`, {info['bytes_archive'] / gib:.2f} GiB compressed, {info['n_files']:,} files,
  {info['bytes_uncompressed'] / gib:.2f} GiB extracted; SHA-256 in `{ARCHIVE_SUM.name}`.
- Member list with the SHA-256 of every file: `{ARCHIVE_LIST.name}` (the same list is inside the archive as
  `outputs/r2/SUPPORTING_OUTPUTS.sha256`).
- How the member list was chosen: the tables-only run was traced (`reproduce_all.py --tables-only
  --trace-inputs`; every file a step opened); the archive holds each file under `outputs/r2/` that a step read
  before any step of the run wrote it ({info['n_traced']:,} files), plus the files that a step only tests for
  existence ({info['n_files'] - info['n_traced']:,} files: `{'`, `'.join(info['extra_patterns'])}`). Files the
  run regenerates itself (annotators, evidence patterns of the primary variants) are not included.

## Use

```bash
cd subproject_24_multireference_latent_truth_model          # the repository (subproject) root
(cd outputs/release && shasum -a 256 -c {ARCHIVE_SUM.name})    # archive downloaded into outputs/release/
tar -xzf outputs/release/{ARCHIVE.name}                       # creates outputs/r2/...
shasum -a 256 -c outputs/r2/SUPPORTING_OUTPUTS.sha256          # every extracted file
cd .. && export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \\
    XLA_FLAGS='--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=2'
python subproject_24_multireference_latent_truth_model/src/r2/reproduce_all.py --tables-only
python subproject_24_multireference_latent_truth_model/src/r2/reproduce_all.py --check-manifest
```

Python 3.11 with the packages of `requirements-r2.txt` (`requirements-r2.lock` pins their dependencies too);
pdflatex for the LaTeX check of the paper-asset stage. The external raw data are not needed: the data stages
that read them (gene panels, method scores, held-out ChIP references) rebuild their tables from the stored
files in the archive, and the model stages run only their table steps.

## Contents by directory

| directory | files | MiB (extracted) |
|---|---:|---:|
{rows}
"""
    ARCHIVE_README.write_text(txt)
    return ARCHIVE_README


# =============================================================================================
# Refit check: fits re-run into a scratch root (timing runs) against the stored fits
# =============================================================================================
def compare_fits(root: Path, label: str = "", path: Path | None = None) -> list[dict]:
    """Compare every fit under ``root`` (layout outputs/r2/fits/<universe>/<variant>/<fit_id>/) with the
    stored fit of the same id: max |credibility difference| (cred.npy, or cred_mean / lo / hi for
    NUTS), best log posterior (MAP), NUTS diagnostics and runtimes.  Rows are upserted into
    ``pipeline_refit_check.csv`` (post hoc procedural check; reads no held-out labels)."""
    import numpy as np
    path = path or REFIT_CSV
    root = Path(root)
    rows = []
    for meta_p in sorted(root.glob("*/*/*/meta.json")):
        d = meta_p.parent
        u, variant, fid = d.parent.parent.name, d.parent.name, d.name
        ref = OUT / "fits" / u / variant / fid
        if not (ref / "meta.json").exists():
            continue
        mn, mo = _jload(meta_p), _jload(ref / "meta.json")
        files = ["cred_mean.npy", "cred_lo.npy", "cred_hi.npy"] if (d / "cred_mean.npy").exists() else ["cred.npy"]
        diffs, ident = [], True
        for f in files:
            x, y = np.load(ref / f), np.load(d / f)
            ident &= bool(x.shape == y.shape and np.array_equal(x, y))
            diffs.append(float(np.max(np.abs(x - y))) if x.shape == y.shape else float("nan"))
        cn, co = mn.get("convergence") or {}, mo.get("convergence") or {}
        lp_n, lp_o = mn.get("best_log_post"), mo.get("best_log_post")
        rows.append(dict(
            universe=u, variant=variant, fit_id=fid, inference=mn.get("inference", ""), label=label,
            files_compared=";".join(files), max_abs_cred_diff=max(diffs), bit_identical=ident,
            best_log_post_stored=lp_o, best_log_post_rerun=lp_n,
            abs_log_post_diff=(abs(lp_n - lp_o) if isinstance(lp_n, (int, float)) and isinstance(lp_o, (int, float)) else ""),
            r_hat_max_stored=co.get("r_hat_max", ""), r_hat_max_rerun=cn.get("r_hat_max", ""),
            n_divergent_stored=mo.get("n_divergent", ""), n_divergent_rerun=mn.get("n_divergent", ""),
            runtime_s_stored=mo.get("runtime_s", ""), runtime_s_rerun=mn.get("runtime_s", ""),
            rerun_root=str(root), checked_at=dt.datetime.now().isoformat(timespec="seconds"),
            analysis_status="post_hoc",
            note="re-run of a stored fit with the same code, seed and settings into a scratch root (timing run); "
                 "procedural reproducibility check, no held-out labels read"))
    cols = list(rows[0]) if rows else []
    old = []
    if path.exists():
        with open(path, newline="") as fh:
            old = [r for r in csv.DictReader(fh)
                   if (r["universe"], r["variant"], r["fit_id"]) not in {(x["universe"], x["variant"], x["fit_id"]) for x in rows}]
        cols = cols or (list(old[0]) if old else [])
    if rows or old:
        allr = old + rows
        allr.sort(key=lambda r: (r["fit_id"].endswith("nuts"), str(r["universe"]), str(r["fit_id"])))
        with open(path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=cols, lineterminator="\n")
            w.writeheader()
            for r in allr:
                w.writerow({c: r.get(c, "") for c in cols})
    return rows


# =============================================================================================
# Orchestration
# =============================================================================================
def _rel(p: Path) -> str:
    try:
        return str(Path(p).relative_to(SP))
    except ValueError:
        return str(p)


def _display(argv: list[str]) -> list[str]:
    """Command as documented: interpreter and paths relative to biodyn-work."""
    out = []
    for a in argv:
        if a == PY:
            out.append(".venv/bin/python")
        elif a.startswith(str(BW) + "/"):
            out.append(os.path.relpath(a, BW))
        else:
            out.append(a)
    return out


def select_stages(a) -> list[Stage]:
    names = [s.name for s in STAGES]
    skip = set(getattr(a, "skip", None) or [])
    for n in (a.only or []) + ([a.from_] if a.from_ else []) + sorted(skip):
        if n not in STAGE_BY_NAME:
            raise SystemExit(f"unknown stage {n!r}; stages: {', '.join(names)}")
    if a.only:
        sel = [s for s in STAGES if s.name in a.only]
    elif a.from_:
        sel = STAGES[names.index(a.from_):]
    else:
        sel = list(STAGES)
    return [s for s in sel if s.name not in skip]


def table_hashes(stage: Stage) -> dict[str, str]:
    return {t: (sha256(TABLES / f"{t}.csv") if (TABLES / f"{t}.csv").exists() else "missing") for t in stage.outputs}


def refresh_assets(log_dir: Path) -> int:
    """Re-render the stage-timing table (the first float of tab_compute), the digest and the prose number
    macros (paper/r2/numbers.tex) after pipeline_stage_timings.csv changed."""
    rc = 0
    for nm, argv in (("tab_compute", [PY, "-m", "r2.paper_assets.tables", "--only", "tab_compute"]),
                     ("digest", [PY, str(SRC / "make_paper_assets.py"), "digest"]),
                     ("numbers", [PY, str(SRC / "make_paper_assets.py"), "numbers"])):
        res = run_step(argv, log_dir / f"refresh_{nm}.log")
        rc |= int(res["exit_code"] != 0)
    return rc


def log_progress(msg: str) -> None:
    line = f"{dt.datetime.now():%Y-%m-%d %H:%M:%S} {msg}"
    print(line, flush=True)
    PIPE_OUT.mkdir(parents=True, exist_ok=True)
    with open(PIPE_OUT / "run.log", "a") as fh:
        fh.write(line + "\n")


def record_writers(stage: Stage, step: Step, before: dict, after: dict, writers: list, last: dict) -> int:
    """Append the tables ``step`` wrote (content changed, or rewritten: compared by sha256 and listed by the
    step's own table writes) to ``writers``; returns 1 when a table that another stage wrote earlier in this
    run is overwritten with different content (two writers of one table that disagree)."""
    owner = stage_table_owner()
    rc = 0
    for t in sorted(after):
        if before.get(t) == after[t]:            # same content and same modification time: not written
            continue
        h, h0 = after[t][0], (before.get(t) or ("", 0))[0]
        declared = owner.get(t, "")
        status = "declared" if (declared == stage.name or t in stage.also_writes) else (
            "shared" if t in SHARED_OUTPUTS else ("pipeline" if t in PIPELINE_TABLES else "UNDECLARED"))
        prev = last.get(t)
        conflict = bool(prev and prev[0] != stage.name and prev[2] != h)
        if conflict and t not in SHARED_OUTPUTS:
            rc = 1
            log_progress(f"[{stage.name}] CONFLICT: {t}.csv written by {prev[0]}/{prev[1]} and now by "
                         f"{stage.name}/{step.name} with different content")
        if status == "UNDECLARED":
            log_progress(f"[{stage.name}] {step.name} wrote {t}.csv, which stage {declared or 'none'} declares")
        writers.append(dict(stage=stage.name, step=step.name, table=t, declared_by=declared, status=status,
                            content_changed=h != h0, sha256=h,
                            conflict_with=(f"{prev[0]}/{prev[1]}" if conflict else "")))
        last[t] = (stage.name, step.name, h)
    return rc


def write_writers(writers: list, plan: list, tables_only: bool, log_dir: Path, complete: bool) -> int:
    """Write ``<log_dir>/table_writers.csv`` (one row per table written per step) and report the declared
    tables of the executed stages that no step rewrote.  ``complete``: every stage of the mode ran."""
    p = log_dir / "table_writers.csv"
    p.parent.mkdir(parents=True, exist_ok=True)
    cols = ["stage", "step", "table", "declared_by", "status", "content_changed", "sha256", "conflict_with"]
    with open(p, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, lineterminator="\n")
        w.writeheader()
        for r in writers:
            w.writerow(r)
    written = {r["table"] for r in writers}
    ran = {s.name for s, _ in plan}
    not_written = sorted(t for s in STAGES if s.name in ran for t in s.outputs if t not in written)
    undeclared = sorted({r["table"] for r in writers if r["status"] == "UNDECLARED"})
    changed = sorted({r["table"] for r in writers if r["content_changed"]})
    log_progress(f"[pipeline] table writers -> {_rel(p)}: {len(written)} tables written, {len(changed)} with changed "
                 f"content{': ' + ', '.join(changed) if changed else ''}; undeclared writers: {undeclared or 'none'}; "
                 f"declared but not rewritten: {not_written or 'none'}")
    return 1 if (complete and undeclared) else 0


def run(a) -> int:
    stages = select_stages(a)
    tables_only = a.tables_only
    run_id = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = PIPE_OUT / "logs" / run_id
    mode_word = "tables-only" if tables_only else "full"
    plan = []
    for s in stages:
        steps = [st for st in s.steps_for(tables_only) if not a.steps or st.name in a.steps]
        if steps:
            plan.append((s, steps))
    if a.dry_run:
        print(f"# dry run ({mode_word}); cwd {SP}; threads {THREAD_ENV}")
        for s, steps in plan:
            reqs = s.reqs_for(tables_only)
            miss = [r for r in reqs if not r.ok()]
            status = "ready" if not miss else ("SKIP (external data missing)" if all(r.kind == "external" for r in miss)
                                               else "BLOCKED (inputs missing)")
            print(f"\n[{s.layer}] {s.name}: {s.label} -- {status}; full cost {s.full_cost}")
            for r in reqs:
                print(f"    needs {r.kind:9s} {'ok ' if r.ok() else 'MISSING'} {r.label}: {r.pattern}"
                      + (f" (stage {r.producer})" if r.producer and not r.ok() else ""))
            for st in steps:
                cmd = _display(st.argv) + (shlex.split(a.extra_args) if a.extra_args else [])
                print(f"    {st.name:24s} [{st.mode}] {shlex.join(cmd)}" + (f"   # {st.note}" if st.note else ""))
            if s.outputs:
                print(f"    tables: {', '.join(s.outputs)}")
        unc = unclaimed_tables()
        print(f"\n# tables in data/r2/tables not declared by any stage: {unc or 'none'}")
        return 0
    if not a.no_record:
        write_environment()
        write_requirements()
    log_progress(f"[pipeline] run {run_id} ({mode_word}): {', '.join(s.name for s, _ in plan)}")
    rc_all, n_skipped = 0, 0
    determinism = []
    writers: list[dict] = []           # which step wrote which table (sha256 before / after every step)
    last_writer: dict[str, tuple] = {}  # table -> (stage, step, sha256 after that write)
    trace_dir = log_dir / "trace" if a.trace_inputs else None
    for s, steps in plan:
        miss = [r for r in s.reqs_for(tables_only) if not r.ok()]
        if miss:
            ext = [r for r in miss if r.kind == "external"]
            if ext and len(ext) == len(miss):
                n_skipped += 1
                log_progress(f"[{s.name}] SKIPPED: needs external data not in the repository: "
                             + "; ".join(f"{r.label} ({r.pattern})" for r in ext)
                             + ". Set BIODYN_EXT to the data root or obtain the raw files (README); the committed "
                               "tables of this stage are kept.")
                continue
            msg = (f"[{s.name}] BLOCKED: missing " + "; ".join(
                f"{r.label} ({r.pattern}; written by stage {r.producer or '?'})" for r in miss)
                + ". Run the producing stage in full mode or restore outputs/r2 from the supporting archive.")
            log_progress(msg)
            rc_all = 1
            if a.keep_going:
                continue
            break
        repeats = 2 if a.repeat_check else 1
        hashes = []
        for rep in range(repeats):
            stage_ok = True
            for st in steps:
                argv = st.argv + (shlex.split(a.extra_args) if a.extra_args else [])
                lp = log_dir / f"{s.name}__{st.name}{'' if rep == 0 else f'__repeat{rep}'}.log"
                log_progress(f"[{s.name}] {st.name} ({st.mode}): {shlex.join(_display(argv))}")
                before = table_state()
                res = run_step(argv, lp, trace=(trace_dir / f"{s.name}__{st.name}.txt") if trace_dir and rep == 0
                               else None)
                rc_all |= record_writers(s, st, before, table_state(), writers, last_writer)
                log_progress(f"[{s.name}] {st.name}: exit {res['exit_code']}, {res['wall_time_s']:.1f} s, "
                             f"peak RSS {res['peak_rss_mib']:.0f} MiB (tree {res['peak_rss_tree_mib']:.0f}), "
                             f"load {res['loadavg_1m_start']}-{res['loadavg_1m_max']}")
                if not a.no_record and rep == 0:
                    row = dict(stage=s.name, layer=s.layer, step=st.name,
                               mode=("tables_only" if (tables_only and not st.full) else st.mode)
                               if not a.extra_args else "full",
                               scope=a.scope or "all", timing_source="measured_isolated",
                               command=shlex.join(_display(argv)), n_units=1,
                               log=_rel(lp), note=(a.note or st.note), **res)
                    if trace_dir:
                        row["note"] = (row["note"] + "; " if row["note"] else "") + "input tracing on (audit hook)"
                    if a.extra_args:
                        row["note"] = (row["note"] + "; " if row["note"] else "") + \
                            "run with --extra-args (outputs redirected / restricted; stored outputs untouched)"
                    write_timings([row])
                if res["exit_code"] != 0:
                    stage_ok = False
                    rc_all = 1
                    log_progress(f"[{s.name}] FAILED at step {st.name}; see {lp}")
                    break
            if not stage_ok:
                break
            hashes.append(table_hashes(s))
        if a.repeat_check and len(hashes) == 2:
            for t in s.outputs:
                determinism.append(dict(stage=s.name, table=t, sha256_run1=hashes[0][t], sha256_run2=hashes[1][t],
                                        identical=hashes[0][t] == hashes[1][t]))
            same = all(d["identical"] for d in determinism if d["stage"] == s.name)
            log_progress(f"[{s.name}] repeat check: {'all tables byte-identical' if same else 'TABLES DIFFER'}")
        if rc_all and not a.keep_going:
            break
    if writers:
        rc_all |= write_writers(writers, plan, tables_only, log_dir, complete=not (a.only or a.from_ or a.steps
                                                                                 or a.extra_args or rc_all))
    if determinism:
        p = PIPE_OUT / "determinism_check.csv"
        old = list(csv.DictReader(open(p))) if p.exists() else []
        old = [r for r in old if r["stage"] not in {d["stage"] for d in determinism}]
        stamp = dt.datetime.now().isoformat(timespec="seconds")
        with open(p, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=["stage", "table", "sha256_run1", "sha256_run2", "identical", "checked_at"],
                               lineterminator="\n")
            w.writeheader()
            for r in old:
                w.writerow(r)
            for d in determinism:
                w.writerow({**d, "checked_at": stamp})
    if rc_all == 0 and not a.only and not a.from_ and not a.skip and not a.steps and not a.extra_args and not a.no_record:
        write_manifest()            # data/r2/tables is final here; the refresh below writes paper/r2 only
        log_progress(f"[pipeline] wrote {_rel(MANIFEST)}")
    if not a.no_record and any(s.name == "paper_assets" for s, _ in plan) and rc_all == 0:
        log_progress("[pipeline] refreshing tab_compute, the digest and numbers.tex (timing table changed after the asset stage)")
        rc_all |= refresh_assets(log_dir)
    log_progress(f"[pipeline] run {run_id} finished: exit {rc_all}; {n_skipped} stage(s) skipped for missing external data")
    return rc_all


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tables-only", action="store_true", help="regenerate every table from stored outputs (no refit)")
    ap.add_argument("--from", dest="from_", metavar="STAGE", help="start at this stage")
    ap.add_argument("--only", nargs="+", metavar="STAGE", help="run only these stages (pipeline order)")
    ap.add_argument("--skip", nargs="+", metavar="STAGE", default=[],
                    help="leave out these stages (e.g. --skip paper_assets: every table, no asset); no manifest is "
                         "written")
    ap.add_argument("--steps", nargs="+", metavar="STEP", help="run only these steps of the selected stages")
    ap.add_argument("--dry-run", action="store_true", help="print the plan, inputs and commands; run nothing")
    ap.add_argument("--list", action="store_true", help="list the stages and their tables")
    ap.add_argument("--keep-going", action="store_true", help="continue after a failed or blocked stage")
    ap.add_argument("--repeat-check", action="store_true",
                    help="run each selected stage twice and compare the sha256 of its tables (determinism)")
    ap.add_argument("--extra-args", default="", help="arguments appended to every selected step (timing runs)")
    ap.add_argument("--scope", default="", help="scope label of the timing rows (with --extra-args)")
    ap.add_argument("--note", default="", help="note stored with the timing rows")
    ap.add_argument("--no-record", action="store_true", help="do not write the timing / environment tables")
    ap.add_argument("--trace-inputs", action="store_true",
                    help="record every file each step opens (outputs/r2/pipeline/logs/<run>/trace/); defines the "
                         "supporting archive (--release-archive)")
    ap.add_argument("--release-archive", metavar="RUN_DIR",
                    help="build outputs/release/r2_supporting_outputs.tar.gz from the trace of a --tables-only "
                         "--trace-inputs run (RUN_DIR = outputs/r2/pipeline/logs/<run id>)")
    ap.add_argument("--archive-extra", nargs="*", default=[],
                    help="further paths (relative to the subproject) to put into the archive")
    ap.add_argument("--write-environment", action="store_true")
    ap.add_argument("--write-requirements", action="store_true")
    ap.add_argument("--collect-recorded", action="store_true",
                    help="add the runtimes recorded by the modules at the original runs to the timing table")
    ap.add_argument("--write-manifest", action="store_true")
    ap.add_argument("--check-manifest", action="store_true")
    ap.add_argument("--refresh-assets", action="store_true")
    ap.add_argument("--compare-fits", metavar="ROOT", nargs="+",
                    help="compare fits re-run into scratch fits roots with the stored fits (pipeline_refit_check.csv)")
    a = ap.parse_args(argv)
    util = False
    if a.list:
        util = True
        for s in STAGES:
            print(f"{s.layer:10s} {s.name:16s} {s.label}")
            print(f"{'':27s}full: {', '.join(x.name for x in s.steps_for(False))}; "
                  f"tables-only: {', '.join(x.name for x in s.steps_for(True)) or '-'}")
            if s.outputs:
                print(f"{'':27s}tables: {', '.join(s.outputs)}")
        print(f"undeclared tables: {unclaimed_tables() or 'none'}")
    if a.write_environment:
        util = True
        print(write_environment())
    if a.write_requirements:
        util = True
        print(write_requirements())
    if a.collect_recorded:
        util = True
        rows = collect_recorded()
        write_timings(rows)
        print(f"{len(rows)} recorded rows -> {TIMINGS_CSV}")
    if a.compare_fits:
        util = True
        for root in a.compare_fits:
            rows = compare_fits(Path(root), a.note)
            n_id = sum(bool(r["bit_identical"]) for r in rows)
            print(f"{root}: {len(rows)} fits compared, {n_id} bit-identical, max |dcred| "
                  f"{max([r['max_abs_cred_diff'] for r in rows] or [float('nan')]):.3g} -> {REFIT_CSV}")
    if a.release_archive:
        util = True
        info = build_release_archive(Path(a.release_archive), a.archive_extra)
        print(f"{ARCHIVE}: {info['n_files']} files, {info['bytes_uncompressed'] / 2 ** 30:.2f} GiB extracted, "
              f"{info['bytes_archive'] / 2 ** 30:.2f} GiB archive, sha256 {info['sha256']}")
        for k, v in sorted(info["by_dir"].items()):
            print(f"    {k}: {v[0]} files, {v[1] / 2 ** 20:.1f} MiB")
        if info["outside"]:
            print("data files read outside the subproject (not in a fresh clone):")
            for q, st in sorted(info["outside"].items()):
                print(f"    {st}: {q}")
        print(f"environment / library-cache files read outside the subproject: {len(info['outside_env'])}")
    if a.refresh_assets:
        util = True
        rc = refresh_assets(PIPE_OUT / "logs" / ("refresh_" + dt.datetime.now().strftime("%Y%m%d_%H%M%S")))
        if rc:
            return rc
    if a.write_manifest:
        util = True
        print(write_manifest())
    if a.check_manifest:
        util = True
        probs = check_manifest()
        print("manifest ok" if not probs else "\n".join(probs))
        if probs:
            return 1
    if util:
        return 0
    return run(a)


if __name__ == "__main__":
    sys.exit(main())
