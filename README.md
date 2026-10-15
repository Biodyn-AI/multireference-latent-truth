# Multi-Reference Latent Truth Model for GRN Benchmarking

A framework for estimating per-edge regulatory credibility from multiple noisy reference databases, with dual-universe evaluation and conservative claim governance. All analyses run on laptop-scale table-level pipelines with no GPU requirement.

**Paper**: `paper/main.pdf` (11 pages, 6 figures, 3 tables, 15 references)

**Author**: Ihor Kendiukhov, University of Tübingen

**Repository**: [https://github.com/Biodyn-AI/multireference-latent-truth](https://github.com/Biodyn-AI/multireference-latent-truth)

**Target venue**: Statistical Applications in Genetics and Molecular Biology (SAGMB), De Gruyter


## Key Results

- **Hierarchical Family-Bayes** achieves lowest rank disagreement under both source-dropout and family-block-dropout stress protocols (0.067-0.072 mean disagreement vs. 0.104-0.139 for alternatives).
- **Universe-aware evaluation.** The large raw precision/F1 gap between the context-matched and full universes (23-31x) is shown, by explicit decomposition, to be almost entirely a base-rate (denominator) effect: the context universe has a 25.8x higher reference density, and the genuine denominator-invariant ranking difference is only ~1.4x. All promotion/validation decisions use denominator-invariant enrichment over random.
- **Dual-universe governance** restricts promotion to 842 out of 421,521 candidate edges (0.2%), with 2.2x denominator-invariant perturbation-backed enrichment (p << 0.05) on pre-registered panels.
- **Genuinely external validation.** The promoted set is enriched 1.77x-2.12x (p < 1e-7) against an external ENCODE ChIP-seq reference that is not used anywhere in the credibility model, with a clean monotone separation across governance tiers.
- **Governance is well-specified and robust:** exact rank-based tier cutoffs, a +-10% cutoff sensitivity analysis (promoted pool 834-849), a distinct Strong-vs-Moderate breakdown, an automated diversity-precision operating-point selector, and a functional/pathway characterization of the promoted set.
- **LOCO replication** across 6 perturbation cohorts yields a 5/6 pass rate; given the small cohort count this is reported as an exploratory support-density reliability threshold.


## Repository Structure

```
.
├── paper/                      # Manuscript (primary deliverable)
│   ├── main.tex                # LaTeX source
│   ├── main.pdf                # Compiled paper (11 pages)
│   ├── figures/                # Publication-quality figures (PDF + PNG)
│   └── generate_figures.py     # Figure generation from data/paper_tables/
│
├── src/                        # Source code (59 scripts, organized by function)
│   ├── models/                 # LTM model fitting (7 variants)
│   │   ├── fit_em_ltm.py              # EM baseline (Dawid-Skene analog)
│   │   ├── fit_dependent_ltm.py       # Dependence-aware model
│   │   ├── fit_multifactor_ltm.py     # Multifactor extension
│   │   ├── fit_family_regularized_ltm.py  # Family-regularized model
│   │   └── fit_family_hierarchical_bayes_ltm.py  # Primary: Family-HB model
│   ├── evaluation/             # Calibration, tier certification, variant comparison
│   ├── robustness/             # Source/family-block dropout, synthetic stress
│   ├── governance/             # Dual-universe governance, deconcentration policies
│   ├── replication/            # LOCO, L2O, L3O, blind holdout protocols
│   ├── endpoint_modeling/      # Bayesian endpoint tiering (5 model variants)
│   ├── sensitivity/            # Prior sensitivity, fragility analysis
│   ├── revision/               # Universe-aware decomposition, external ENCODE
│   │                           #   validation, tier-cutoff sensitivity, tier
│   │                           #   breakdown, functional enrichment, auto-deconcentration
│   ├── tuning/                 # Family-HB hyperparameter tuning
│   ├── packaging/              # Table/figure generation, submission bundle builders
│   ├── utils/                  # Shared utilities (ltm_utils.py, source_catalog.json)
│   └── run_full_pipeline.py    # Main pipeline entry point
│
├── data/                       # Key data files
│   ├── paper_tables/           # 17 summary CSVs feeding paper figures and tables
│   ├── paper_figures/          # Original analysis figures
│   └── core/                   # Evidence matrix, edge universe, source metadata
│
├── outputs/                    # Full computed outputs (134 entries, ~5.7 GB)
│                               # Intermediate results from all analysis phases
│
└── archive/                    # Development artifacts (not needed for reproduction)
    ├── reports/                # 74 iterative phase reports
    ├── planning/               # 35 research planning documents
    ├── simulation_scripts/     # 37 late-phase prospective simulation scripts
    ├── old_paper/              # Original 37-page paper draft + camera-ready
    ├── submission_bundles/     # Frozen submission bundle snapshots
    └── handoff/                # Agent handoff logs
```


## Paper-to-Code Mapping

| Paper Section | Key Scripts | Key Data |
|---|---|---|
| §5.1 Robustness comparison | `src/robustness/run_source_dropout_ablation.py`, `run_family_block_dropout_ablation.py` | `data/paper_tables/robustness_progression_summary.csv` |
| §5.2 Context conditioning / universe-aware decomposition | `src/evaluation/build_context_matched_universe.py`, `src/revision/universe_aware_decomposition.py` | `data/paper_tables/calibration_full_vs_context.csv`, `rev_universe_aware_*.csv`, `rev_enrichment_vs_depth.csv` |
| §5.3 Governance tiers + cutoff sensitivity + Strong/Moderate breakdown | `src/governance/enforce_dual_universe_claim_governance.py`, `src/revision/tier_cutoff_sensitivity.py`, `src/revision/tier_breakdown.py` | `data/paper_tables/dual_universe_claim_tiers_summary.csv`, `rev_tier_cutoff_sensitivity.csv`, `rev_strong_vs_moderate_breakdown.csv` |
| §5.4 Source concentration + functional enrichment + auto-deconcentration | `src/governance/run_deconcentration_policy_sweep.py`, `src/revision/functional_enrichment.py`, `src/revision/auto_deconcentration.py` | `data/paper_tables/deconcentration_*.csv`, `rev_functional_enrichment.csv`, `rev_deconcentration_autotune_*.csv` |
| §5.5 External validation (perturbation + ENCODE ChIP-seq) | `src/replication/run_preregistered_external_validation.py`, `src/revision/external_encode_validation.py` | `outputs/top_tier_readiness_v*/`, `data/paper_tables/rev_external_encode_enrichment.csv` |
| §5.6 LOCO replication | `src/replication/run_multilevel_leave_one_cohort_out.py` | `outputs/top_tier_readiness_v7/multilevel_leave_one_cohort_out_v1/` |
| §5.7 Cohort fragility | `src/replication/run_multilevel_leave_two_cohort_out.py`, `run_multilevel_leave_three_cohort_out.py` | `outputs/top_tier_readiness_v7/multilevel_leave_*_cohort_out_v1/` |
| Figures 1-8 | `paper/generate_figures.py`, `src/revision/make_revision_figures.py` | `data/paper_tables/*.csv`, `outputs/revision/*.csv` |


## Reproducing the paper

The R2 analysis has one entry point, `src/r2/reproduce_all.py` (reviewer Q21, Q24). It lists the 22 stages of the pipeline in dependency order, runs each step as the documented module command in its own subprocess, one at a time, and records the wall time, peak resident memory (largest process; `os.wait4` plus psutil sampling of the process tree) and CPU time of every step in `data/r2/tables/pipeline_stage_timings.csv`. The machine and software go to `data/r2/tables/pipeline_environment.csv`. It changes no analysis: every module keeps its own seeds (`r2.common.PRESPECIFIED`), caches and locks (for example the replication power lock).

```bash
cd /Users/ihorkendiukhov/biodyn-work
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 XLA_FLAGS='--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=2'
P=subproject_24_multireference_latent_truth_model/src/r2/reproduce_all.py
.venv/bin/python $P --list                                     # stages, steps and the tables each one writes
.venv/bin/python $P --tables-only --dry-run                     # plan, input checks and commands; runs nothing
caffeinate -i nice -n 10 .venv/bin/python $P --tables-only       # every table + paper asset from the stored outputs (~72 min)
caffeinate -i nice -n 10 .venv/bin/python $P --tables-only --skip paper_assets --trace-inputs   # every table, no asset; trace the files read
.venv/bin/python $P --release-archive subproject_24_multireference_latent_truth_model/outputs/r2/pipeline/logs/<run id>  # supporting archive from that trace
caffeinate -i nice -n 10 .venv/bin/python subproject_24_multireference_latent_truth_model/src/r2/release_check.py --dest <empty scratch dir>  # fresh-copy check
caffeinate -i nice -n 10 .venv/bin/python subproject_24_multireference_latent_truth_model/src/r2/release_check.py --dest <same dir> --paper-assets  # + paper assets in the copy
caffeinate -i nice -n 10 .venv/bin/python $P                     # full rebuild from the committed and external inputs (days)
.venv/bin/python $P --tables-only --only fit_primary governance --repeat-check --no-record   # determinism check
.venv/bin/python $P --check-manifest                            # sha256 of data/r2/inputs + data/r2/tables
.venv/bin/python -m pytest subproject_24_multireference_latent_truth_model/src/r2/tests -q
```

Options: `--from STAGE`, `--only STAGE ...`, `--skip STAGE ...` (no manifest is written then), `--steps STEP ...`, `--keep-going`, `--repeat-check` (runs the selected stages twice and compares the sha256 of their tables; result in `outputs/r2/pipeline/determinism_check.csv`), `--extra-args "..." --scope LABEL` (appended to every selected step, used to time fits into a scratch `--root` without touching the stored fits), `--no-record`, `--trace-inputs` (an audit hook, `src/r2/trace_inputs/sitecustomize.py`, records every file each step opens in `outputs/r2/pipeline/logs/<run id>/trace/`). Utility actions: `--write-environment`, `--write-requirements` (also writes `requirements-r2.lock`), `--collect-recorded` (adds the runtimes the modules stored when the long steps originally ran), `--write-manifest`, `--check-manifest`, `--refresh-assets`, `--release-archive RUN_DIR` (`--archive-extra PATH ...`), `--compare-fits ROOT ...`. Step logs: `outputs/r2/pipeline/logs/<run id>/<stage>__<step>.log`; run log `outputs/r2/pipeline/run.log`. Every run writes `outputs/r2/pipeline/logs/<run id>/table_writers.csv` (which step wrote which table, before/after sha256); a table written by a stage that does not declare it, or overwritten by a second stage with different content, fails the run.

**Stages (dependency order).** Data layer: `panels` (build_panels; tables-only: `--table-only`), `annotators` (build_annotators `--from-portable`), `evidence`, `grnboost2` (run_grnboost2.sh; tables-only: `table`, `summary`), `fm_scores` (build_fm_scores; tables-only: `--tables-only`), `cohorts` (tables-only: `--table-only --no-download`), `heldout_chip` (build_heldout_chip; tables-only: `--table-only`). Model layer: `fit_primary` (map, laplace, nuts, tables, edge-tables, nuts_rerun_check), `fit_sensitivity`, `obs_models` (informative_observation evidence/map/map_tfw/ppc/cv), `posterior_modes` (nuts, bridge, mixture, simulate, simulate-fixed, sensmodes, clusterstab). Evaluation: `robustness` (fit, search, seedcheck, analyze), `heldout_eval`, `governance`, `modes_tables` (posterior_modes governance + tables; after governance), `replication`, `tuning`, `obs_eval` (informative_observation heldout/tables/check), `sim_hub`, `compute_scaling`, `obs_integration`. Assets: `paper_assets` (make_paper_assets tables `--check-latex`, figures, digest, numbers).

**What `--tables-only` runs.** No model is refitted and no external raw data are read. The annotators (from the committed portable inputs) and the evidence patterns are rebuilt in full. The three data stages that read external raw data rebuild their tables from their stored outputs: gene panels from `outputs/r2/universe/<t>/panel_meta.json` (checked against panel.tsv / tfs.txt), method scores from `outputs/r2/fm_scores/<t>/meta.json`, held-out ChIP references from `outputs/r2/heldout_chip/provenance.json` and `annotator_overlap.csv`. Held-out evaluation, governance, replication (against the existing power lock), the informative-observation held-out evaluation and obs_integration are evaluations of stored fits and run in full; every model module runs only its table step (`fit_primary tables/edge-tables`, `fit_sensitivity tables`, `robustness analyze`, `tuning_optimism analyze`, `posterior_modes governance/tables`, `sim_hub_bias tables`, `compute_scaling tables`). Then the LaTeX tables, figures, digest and prose number macros are rebuilt and `data/r2/MANIFEST.sha256` is rewritten (not with `--only`, `--from`, `--skip` or `--steps`).

**Inputs.**
- Committed: `data/r2/inputs/` (annotator raw copies, HGNC, Lambert TF list; checksums in `data/r2/MANIFEST.sha256`).
- Stored outputs: `--tables-only` reads the `outputs/r2/` tree of a full run (fits, robustness fits, simulation replicates, compute measurements, cohort DE tables, method scores, ChIP references). `outputs/` is gitignored; on a fresh clone extract the supporting archive (below). A stage whose upstream outputs are missing stops the run with a message naming the stage that writes them.
- External raw data (full rebuild only; root `$BIODYN_EXT`, default `/Volumes/Crucial X6/MacBook/biomechinterp/biodyn-work`): Tabula Sapiens h5ad files and scGPT / Geneformer attention matrices (panels, grnboost2, fm_scores), perturbation h5ad files in `single_cell_mechinterp/data/perturb/` and the Replogle pseudobulk download (cohorts), ChIP downloads in `external_data/heldout_chip/` (heldout_chip). In a full run a stage whose external data are missing is skipped with a message and its committed tables are kept.

**Supporting archive (fresh clone).** `outputs/release/r2_supporting_outputs.tar.gz` (0.80 GiB; 2,799 files, 0.90 GiB extracted; SHA-256 in `r2_supporting_outputs.tar.gz.sha256`, per-file SHA-256 in `r2_supporting_outputs_files.sha256` and, inside the archive, `outputs/r2/SUPPORTING_OUTPUTS.sha256`; description in `outputs/release/README_r2_supporting_outputs.md`). It holds the 2,366 files under `outputs/r2/` that a traced tables-only run read before any step of that run wrote them (`--trace-inputs`, then `--release-archive`) and 433 files that a step only tests for existence (the `meta.json` / `params.npz` of the hub-bias simulation fits, `ARCHIVE_EXTRA` in `reproduce_all.py`; found by the release-copy check); the outputs the run regenerates (annotators, evidence patterns of the primary variants) are not included, and no data file outside the subproject was read. Use: from the subproject root `tar -xzf outputs/release/r2_supporting_outputs.tar.gz && shasum -a 256 -c outputs/r2/SUPPORTING_OUTPUTS.sha256`, then `--tables-only` as above.

**Fresh release-copy check** (`src/r2/release_check.py --dest <empty dir>`). Copies exactly the files a commit of the working tree would contain (`git ls-files -co --exclude-standard`; `.gitignore` keeps `outputs/`, the internal `planning/` folder, whose decision letter holds a personal submission-system link, `REVISION_PLAN.md` and LaTeX build files out of the release; and the confidential review correspondence listed there are also kept out; the reviewer digest is then not written), extracts and verifies the archive, moves every committed table out of `data/r2/tables` except the four the tables-only path updates rather than writes (`rep_power`, the locked power design; `eval_analysis_decisions`; `pipeline_stage_timings`; `pipeline_refit_check`), and runs the copy's own `reproduce_all.py --tables-only --skip paper_assets --trace-inputs` with `PYTHONPATH` cleared and `BIODYN_EXT` pointing to a directory that does not exist, so neither the external drive nor its default location can be read. The regenerated tables are compared byte for byte with the working tree (`outputs/r2/pipeline/release_check/compare.csv`, `summary.json`); the copy's traces show any file read outside the copy. `--paper-assets` then works on the same copy: it re-syncs the copy's code and paper sources from the working tree (the same file list without `data/r2/tables`, so the copy keeps the tables it regenerated; a kept table the copy lacks is added), moves the copy's generated assets (`paper/r2/{tables,figures,digest}`, `numbers.tex`, `references_supp*.tex`) aside, runs the copy's `reproduce_all.py --tables-only --only paper_assets` and compares every regenerated asset with the working tree (`paper_assets_compare.csv`, `summary_paper_assets.json`): text identical apart from time stamps in provenance lines, figure PDFs rendered pixel-identical, PNGs pixel-identical, or differences confined to run times (`tab_compute` and the `\nPipe*` macros, which come from the copy's own timing table). Both checks are summarised in `data/r2/tables/pipeline_release_check.csv` (key/value, `analysis_status = post_hoc`, procedural; `--write-table` rewrites it from the stored results), which the data-availability statement (archive size, `\nArchiveGiB`) and the response letter (Q24, `\nLtRel*`) read. The table is kept in the copy like the four tables above.

**Environment.** `requirements-r2.txt` pins the third-party packages imported by `src/r2`; `requirements-r2.lock` adds every installed distribution they require (transitively). Both are written by `--write-requirements` from the interpreter that ran the pipeline: `.venv/bin/python` resolves to the anaconda base interpreter (Python 3.11.9), whose site-packages provide the packages. pdflatex (table check) and zsh (grnboost2) are system tools. The validate-es step of grnboost2 also uses a second interpreter with scikit-learn <= 1.2 (`R2_SKLEARN_REF_PYTHON`).

**Timing table** (`pipeline_stage_timings.csv`, one row per stage x step x mode x scope x timing source; first float of the main table `tab_compute`): `timing_source = measured_isolated` (measured by `reproduce_all.py`, one step process at a time with no other process of this pipeline running; "isolated" means sequential within the pipeline, not an idle machine: other users' load is recorded in `loadavg_1m_*`), `recorded_meta` / `recorded_table` / `recorded_log` (runtime stored by the module when the step originally ran, summed over its fits or replicates, which partly ran in parallel on a more heavily loaded shared machine; no peak memory unless the module stored it). `selected` marks the row used for totals (measured over recorded; a whole-step row, scope `all`, supersedes per-universe rows of the same step); `mode` is `full`, `tables_only` or `full+tables` (same command in both modes). The full-run total is the cost of the runs as made, not a bound on the cost on an idle machine: isolated re-runs of the 20 MAP fits and of the kidney NUTS fit took 3.3-3.7 times less wall time than recorded, while three short steps have no recorded runtime (`fit_primary laplace`, the informative-observation `evidence` step, `sim_hub oracle-check`) and are left out.

**Refit check.** `--compare-fits ROOT ...` compares fits re-run into a scratch fits root (for example the timing runs `--only fit_primary --steps map --extra-args "--root <scratch>/fits"` and `--steps nuts --extra-args "--universes kidney --init map_laplace --root <scratch>/fits --map-root outputs/r2/fits"`) with the stored fits and writes `data/r2/tables/pipeline_refit_check.csv` (max |credibility difference|, bit identity, log posterior, NUTS diagnostics, runtimes; `analysis_status = post_hoc`, procedural check).

**Verified runs (2026-10-14, M2 Pro, two threads, one step at a time; external drive not mounted).**
- In place, `--tables-only --skip paper_assets --trace-inputs` (run `20261014_192748`): 19 stages, 32 steps, 71 min, all exit 0; peak memory 8.1 GiB (informative-observation held-out evaluation); 1-min load average 3.7-8.0 from other users. Every table was written by a step of the stage that declares it (`table_writers.csv`; `rep_power`, the locked power design, is kept). Against a copy of `data/r2/tables` taken before the run, 123 of 130 tables are byte-identical. The others: `pipeline_stage_timings` / `pipeline_environment` (this run), `rep_bayes` and `rep_prior_sensitivity` (only their `runtime_s` column), `scale_environment` (only its build time stamp), `eval_analysis_decisions` (the same rows, re-sorted by the modules that rewrite their own rows) and `fit_sensitivity` (column `n_converged` of the 8 primary-estimator rows restored: `fit_primary` stores the count as `n_lbfgs_ftol_stop`). In `outputs/r2` only meta, provenance and log time stamps changed.
- Fresh release copy (`src/r2/release_check.py`, copy run `20261014_215715`, 72 min, all steps exit 0): 588 repository files plus the 2,799 archive files (all SHA-256-verified); the 126 committed tables other than the four kept ones were moved aside, and every one was regenerated. Against the working tree, 125 tables are byte-identical; `rep_bayes` and `rep_prior_sensitivity` differ only in `runtime_s`; `pipeline_stage_timings`, `pipeline_environment` and `scale_environment` describe the run (timings, machine, build time). No step read a data file outside the copy (29 environment or library-cache files: package metadata, the matplotlib font cache, ArviZ settings). Two earlier copies found the problems fixed since: the method-score table step required a score file it does not read (`coobs.parquet`), and the hub-bias table step tests the simulation fits' `meta.json` for existence without opening them (now `ARCHIVE_EXTRA`).
- Paper assets (finalize, 2026-10-14): in place, `--tables-only --only paper_assets` (runs `20261014_231344` and, after the last text change, `20261014_234026`, each under 1 min, exit 0) left every generated table, figure, digest and macro unchanged apart from provenance time stamps, the run-time entries of `tab_compute` and `\nPipeFullS`, and the new macro `\nArchiveGiB`; `numbers --check` 0 problems. In the release copy (`release_check.py --paper-assets`, copy run `20261014_233926`, exit 0) all 115 generated assets were regenerated from the copy's own tables: 113 identical apart from time stamps or PDF metadata, `tab_compute.tex` and `numbers.tex` different only in run times (`\nPipeTablesMin` 72 vs 73 min, `\nPipeTablesPeakGiB`, `\nPipeFullS`), none different otherwise.
- Final release copy (2026-10-15, after `.gitignore` excluded `planning/`, `REVISION_PLAN.md` and LaTeX build files): `release_check.py --trace-inputs` (copy run `20261015_014523`, 3 h 10 min on a machine with a 1-min load average of 4-10 from other users, all steps exit 0) copied the 509 files of the release list plus the 2,799 archive files (all SHA-256-verified); all 126 committed tables moved aside were regenerated, 122 byte-identical, `rep_bayes` / `rep_prior_sensitivity` different only in `runtime_s`, 2 run records; no data file read outside the copy. `--paper-assets` on the same copy (run `20261015_045527`, exit 0): all 115 assets regenerated, 113 identical apart from time stamps or PDF metadata, `tab_compute.tex` and `numbers.tex` different only in run times, none otherwise. In place, `--tables-only --only paper_assets` (runs `20261015_033438`, `20261015_040622`, exit 0) and `numbers --check` 0 problems.
- Fresh clone from GitHub (2026-10-15, commit `2577d1a`, after `.gitignore` also excluded the review correspondence): the clone's own `src/r2/release_check.py --dest <empty dir> --archive <archive>`, then `--paper-assets` (copy run `20261015_154201`, 2 h, all steps exit 0) copied the 481 committed files plus the 2,799 archive files (all SHA-256-verified); of the 126 regenerated tables 122 are byte-identical, `rep_bayes` and `rep_prior_sensitivity` differ only in `runtime_s` and two are run records; 0 data reads outside the copy. Of the 114 regenerated paper assets 112 are identical apart from time stamps or PDF metadata and two (`tab_compute.tex`, `numbers.tex`) differ only in run times (`pipeline_release_check.csv`).
- Determinism (earlier run, 12:47): the table stages of `fit_primary`, `robustness` and `governance` run twice gave 25 of 25 byte-identical tables (`outputs/r2/pipeline/determinism_check.csv`). The 20 MAP fits (5.3 min) and the kidney NUTS fit (4 x 1000/1000, 6.6 min) re-run into a scratch root are bit-identical to the stored fits (`pipeline_refit_check.csv`).
- Full run: about 66 h of compute as recorded (sum over steps), 98% of it runtimes recorded in the original runs: NUTS (primary 8.4 h, per-mode 5.8 h, prior sensitivity 2.0 h), simulations (hub bias 9.3 h of fits, mode selection 5.4 h), informative-observation MAP 8.3 h, robustness dropout fits 4.9 h plus 0.4 h independent-seed check, compute-scaling sessions 6.4 h.

R1 (superseded): `cd paper/ && python generate_figures.py` built the R1 figures from `data/paper_tables/`.


## Full Pipeline

The core analysis pipeline runs through these stages:

1. **Evidence matrix construction** (`src/evaluation/build_evidence_matrix.py`) — builds the 421,521-edge candidate space with binary reference observations from all sources.
2. **Model fitting** (`src/models/fit_*.py`) — fits five LTM variants of increasing complexity on the evidence matrix.
3. **Hyperparameter tuning** (`src/tuning/tune_family_hb_joint_objective.py`) — optimizes Family-HB parameters via joint robustness + perturbation calibration objective.
4. **Robustness stress testing** (`src/robustness/`) — source-dropout and family-block-dropout ablations.
5. **Context-matched evaluation** (`src/evaluation/build_context_matched_universe.py`) — constructs the 7,536-edge context-matched universe.
6. **Perturbation calibration** (`src/evaluation/calibrate_perturbation_claim_tiers.py`) — computes calibration metrics in both evaluation universes.
7. **Governance & promotion** (`src/governance/enforce_dual_universe_claim_governance.py`) — applies four-tier dual-universe governance with optional deconcentration.
8. **Replication** (`src/replication/run_multilevel_leave_one_cohort_out.py`) — leave-one-cohort-out replication across six perturbation cohorts.
9. **Table packaging** (`src/packaging/build_unified_paper_tables.py`) — consolidates results into the CSVs that feed the paper figures.


## R2 rebuild: annotators, integration baselines and evidence matrices

`src/r2/build_annotators.py` splits the public TF-target resources by primary resource (OmniPath DoRothEA A–D, CollecTRI, TRRUST v2) into held-in LIT/MOTIF/COEXP annotators and held-out CHIP sources, and writes the integration baselines (DoRothEA level recomputed from held-in evidence, CollecTRI without DoRothEA-A, `heldin_support_lit` as the only held-in support flag). Inputs are sha256-checked frozen copies in `outputs/r2/annotators/raw/` or, on a fresh clone, the committed portable copies in `data/r2/inputs/annotator_raw/`. `src/r2/evidence.py` builds the observation-masked evidence patterns for the tissue universes and the global universe, for the primary variant `prespecified_cc` and 8 sensitivity variants. These include `lit_no_ht`, which removes exactly the 9 regulons in `data/r2/tables/lit_ht_like_regulons.csv`, and `lit_no_chipderived`, which removes the whole sources FANTOM4, HTRIdb, ORegAnno and LIT_other.

```bash
cd /Users/ihorkendiukhov/biodyn-work/subproject_24_multireference_latent_truth_model
/Users/ihorkendiukhov/biodyn-work/.venv/bin/python src/r2/build_annotators.py                   # from the frozen raw/ inputs (~20 s)
/Users/ihorkendiukhov/biodyn-work/.venv/bin/python src/r2/build_annotators.py --from-portable   # from data/r2/inputs/annotator_raw/ (fresh clone)
/Users/ihorkendiukhov/biodyn-work/.venv/bin/python src/r2/build_annotators.py --refresh collectri --export-portable  # re-download + new portable copies
/Users/ihorkendiukhov/biodyn-work/.venv/bin/python src/r2/build_annotators.py --freeze-legacy   # one-time: freeze the original inputs (done)
/Users/ihorkendiukhov/biodyn-work/.venv/bin/python src/r2/evidence.py                           # all universes x 9 variants (~20 s, ~1.1 GB RAM)
/Users/ihorkendiukhov/biodyn-work/.venv/bin/python src/r2/evidence.py --out-dir /tmp/ev --tables-dir /tmp/evt --compare-cc-coverage  # scratch rebuild (md5 check) + before/after table of the reverted coverage rule
/Users/ihorkendiukhov/biodyn-work/.venv/bin/python -m pytest src/r2/tests/test_evidence.py -q
```

- **Committed inputs:** `data/r2/inputs/{annotator_edges.parquet, integration_baselines.parquet, annotator_sources.json, annotator_raw_inputs.json, annotator_raw/}`.
- **Outputs:** `outputs/r2/annotators/` and `outputs/r2/evidence/<universe>/{patterns,edge_pattern,tf_coverage}[__<variant>]*` with `meta.json`. For the global universe, `lit_only_via_removed__<variant>.json` lists the TFs and genes that enter only through evidence that `lit_no_ht` or `lit_no_chipderived` removes.
- **Tables:** `data/r2/tables/{source_table, source_components, lit_ht_like_regulons, evidence_summary}.csv`.
- `evidence.py` needs the tissue panels in `outputs/r2/universe/` (`src/r2/build_panels.py`).
- MOTIF/COEXP TF coverage under the curated-conditional masks is the whole-resource rule (`PRESPECIFIED['motif_coexp_tf_coverage']`; decision 10 reverted by item 11); every `meta.json` records it and `evidence.load_patterns` refuses evidence built under another rule.


## R2 rebuild: held-out perturbation cohorts

`src/r2/build_cohorts.py` builds the seven held-out perturbation cohorts (Dixit d7/d13, Adamson, Shifrut, Norman, Papalexi, Replogle) with per-perturbation DE, same-cytoband cis flags (`cis_same_band`, `positive_primary`) and the cohort tables. It downloads the Replogle pseudobulk files with md5 checks if they are missing. Multi-replicate designs (Shifrut, Papalexi; `PRESPECIFIED['replicate_stratified_de']`) use the replicate-stratified test as primary, with the pooled test as a sensitivity analysis. Gene symbols follow `PRESPECIFIED['symbol_mapping']`: an approved column name or label is kept, and the Ensembl id is used only for non-approved names. Evaluate with `positive_primary` and drop `cis_same_band` rows; `positive_incl_cis` is not for evaluation.

```bash
cd /Users/ihorkendiukhov/biodyn-work/subproject_24_multireference_latent_truth_model
/Users/ihorkendiukhov/biodyn-work/.venv/bin/python -W ignore src/r2/build_cohorts.py              # all cohorts + tables
/Users/ihorkendiukhov/biodyn-work/.venv/bin/python -W ignore src/r2/build_cohorts.py --table-only # tables + universe overlap only
/Users/ihorkendiukhov/biodyn-work/.venv/bin/python -m pytest src/r2/tests/test_cohorts.py -q     # unit tests
```

- **Outputs:** `outputs/r2/cohorts/<cohort>/{de.parquet, perturbations.csv, meta.json}`; Shifrut and Papalexi also have `de_pooled.parquet` (pooled-test sensitivity analysis, same tested pairs). `perturbations.csv` flags TF perturbations whose pooled vs stratified non-cis positives have Jaccard < 0.3 (`de_nonrobust_tf`).
- **Tables:** `data/r2/tables/{cohort_table, cohort_perturbations, cohort_universe_overlap}.csv`.


## R2 rebuild: expression baselines (GRNBoost2/GENIE3) and method scores

`src/r2/build_grnboost2.py` runs GRNBoost2 (all tissues) and GENIE3 (immune) on the tissue context universes, on one shared 3,000-cell sample per tissue (`outputs/r2/grnboost2/<t>/cells.txt`), with arboreto 0.1.6 on scikit-learn 1.5.1 and the same-subset early-stopping monitor in `src/r2/arboreto_compat.py`. Unfiltered runs are primary; the >=1%-detection gene filter (`*_detect1pct`) is a sensitivity analysis for both methods. `src/r2/build_fm_scores.py` then scores every universe edge with scGPT/Geneformer attention (raw, transposed, 1/L null, normalised), Pearson/Spearman on the same cells, and a random score. Predicted edge sets use `src/r2/predicted_sets.py::topk_per_tf` (top 50 per TF, seeded tie-break; optional `min_coobs` filter for the attention sensitivity sets).

```bash
cd /Users/ihorkendiukhov/biodyn-work/subproject_24_multireference_latent_truth_model
zsh src/r2/run_grnboost2.sh      # all runs, validate-es, table, summary (~60 min, <= 6 dask workers)
/Users/ihorkendiukhov/biodyn-work/.venv/bin/python src/r2/build_grnboost2.py run --tissue immune --method genie3 --min-detect-frac 0.01   # a single run
/Users/ihorkendiukhov/biodyn-work/.venv/bin/python src/r2/build_grnboost2.py table      # data/r2/tables/grn_baseline_runtime.csv
/Users/ihorkendiukhov/biodyn-work/.venv/bin/python src/r2/build_grnboost2.py summary    # outputs/r2/grnboost2/summary{.json,_rare_targets.csv}
/Users/ihorkendiukhov/biodyn-work/.venv/bin/python src/r2/build_fm_scores.py            # needs grnboost2/<t>/cells.txt (~1-2 min)
/Users/ihorkendiukhov/biodyn-work/.venv/bin/python -m pytest src/r2/tests/test_arboreto_compat.py src/r2/tests/test_grnboost2.py src/r2/tests/test_predicted_sets.py src/r2/tests/test_fm_scores.py -q
```

- **Outputs:** `outputs/r2/grnboost2/<t>/{grnboost2,genie3}[_detect1pct].parquet` (+ `_rounds.tsv`, `_meta.json`, `time_l_*.txt`), `outputs/r2/fm_scores/<t>/{scores,coobs}.parquet`, `meta.json`.
- **Tables:** `data/r2/tables/grn_baseline_runtime.csv`, `data/r2/tables/fm_{universe_summary,extraction_cells,attention_null,float16_error,float16_rowsum_check,min_coobs_sensitivity}.csv`.


## R2 rebuild: held-out ChIP references

`src/r2/build_heldout_chip.py` builds the three held-out ChIP references from md5-checked official downloads: ENCODE TFBS (Harmonizome, union over 5 cell lines), ENCODE-strict (bound in >= 2 lines, TFs assayed in >= 2 lines) and ENETS2 (Gerstein et al. 2012 proximal filtered, with the NFKB1->RELA and TCF4->TCF7L2 label corrections confirmed by target overlap and ENCODE antibody metadata). The target-degree null (`PRESPECIFIED['target_degree_bins' / 'target_degree_bin_rule' / 'target_degree_definition']`) is implemented once, in `reference_index` / `target_degree_model` (`degree_bin`, `null_probability` for arbitrary pairs); the held-out evaluation imports these. The stage also writes the held-in annotator overlap with each reference. This includes the sensitivity groups `all_held_in_no_ht` and `all_held_in_no_chipderived`, the `shares_encode_experiments` / `low_support` / `conservative_headline` flags, and an ENETS2 regulon-label concordance check, which documents the regulons and drops none. It needs the annotator outputs and all four universes; `--allow-partial` turns a missing input into a warning.

```bash
cd /Users/ihorkendiukhov/biodyn-work
.venv/bin/python subproject_24_multireference_latent_truth_model/src/r2/build_heldout_chip.py [--download]   # ~1-2 min
.venv/bin/python -m pytest subproject_24_multireference_latent_truth_model/src/r2/tests/test_heldout_chip.py -q
```

- **Outputs:** `outputs/r2/heldout_chip/{encode_tfbs,encode_strict,enets2}{.parquet,_assayed_tfs.txt,_gene_universe.txt}`, plus `assayed_tfs.tsv`, `dropped_factors.tsv`, `enets2_label_concordance.tsv`, `annotator_overlap.csv`, `omnipath_source_overlap.csv` and `provenance.json`. Raw files are in `$BIODYN_EXT/external_data/heldout_chip/`.
- **Tables:** `data/r2/tables/{heldout_chip_table, heldout_chip_coverage, heldout_chip_annotator_overlap}.csv`.

## R2 model fits: primary MAP and NUTS fits (reviewer Q1, Q11, Q13)

`src/r2/fit_primary.py` fits the five estimators by multi-start MAP (the restarts of `lcm.fit_map`, each Newton-polished; the stored optimum is the polished best restart; converged = polished max|gradient| < 1e-6 at a positive-definite Hessian) on the primary evidence (`prespecified_cc`) of the global and the three tissue universes, groups restarts into distinct optima with Laplace-approximated masses, and runs Family-HB NUTS with FROZEN settings (`NUTS_SETTINGS`, committed copy `src/r2/configs/fit_primary_nuts.json`: 4 x 1000/1000, target accept 0.95; `family_hb_nuts` starts in the dominant mode, `family_hb_randinit_nuts` from random starts). Agreement tables use observed edges only. No held-out data is read.

```bash
cd /Users/ihorkendiukhov/biodyn-work
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 XLA_FLAGS='--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=2'
.venv/bin/python -W ignore subproject_24_multireference_latent_truth_model/src/r2/fit_primary.py all        # map -> laplace -> nuts (random, map_laplace) -> tables -> edge-tables
# or by stage: map | laplace | nuts [--init random map_laplace] [--settings frozen] | tables | edge-tables  (--universes ...)
.venv/bin/python -W ignore subproject_24_multireference_latent_truth_model/src/r2/nuts_rerun_check.py   # first-pass vs re-run of family_hb_nuts (post hoc procedural check)
.venv/bin/python -m pytest subproject_24_multireference_latent_truth_model/src/r2/tests/test_fit_primary.py -q
```

- **Outputs:** `outputs/r2/fits/<universe>/prespecified_cc/{em,dependence,multifactor,family_regularized,family_hb,family_hb_nuts,family_hb_randinit_nuts}/` (fit storage contract; MAP adds `restarts.csv`, `cred_restarts.npy`, `optima.csv`, `optima_params.npz`), `outputs/r2/fits/<tissue>/primary_edge_cred.parquet`.
- **Tables:** `data/r2/tables/fit_{convergence,source_parameters,class_weights,hyperparameters,map_optima,nuts_chains,map_vs_nuts,estimator_agreement}.csv`. Each table has `analysis_status` (row level: `prespecified` / `post_hoc`) and `post_hoc_columns` (columns added or redefined after the project's first held-out results, e.g. the marginal LR+ and the observed-edge top-k agreement); the dated decisions are in `data/r2/tables/eval_analysis_decisions.csv` (module `fit_primary`). NUTS sites with split R-hat > 1.01 or bulk ESS < 400 are flagged (`sites_flagged` in `fit_convergence.csv`, `diag_flag` in `fit_hyperparameters.csv`). Post hoc bookkeeping (robfix, 2026-10-14): NUTS rows of `fit_source_parameters.csv` give the 95% posterior interval of the per-draw rank of the marginal LR+ (`rank_lo`, `rank_hi`) next to `rank` (rank of the posterior mean), and every row of the eight tables carries `values_changed_post_hoc` / `value_change_note` (the row comes from a fit recomputed after the first held-out outcomes: polished MAP refits, frozen-settings NUTS re-runs, relabelled random-init chains). First-pass fits and tables are kept in `outputs/r2/fits/_superseded_firstpass/`; `src/r2/nuts_rerun_check.py` writes `outputs/r2/fits/_superseded_firstpass/nuts_rerun_vs_firstpass.csv` (per universe with a kept first-pass fit: settings and diagnostics of both runs, credibility differences over observed edges, Strong/Moderate-set churn; `analysis_status = post_hoc`).

## R2 evaluation: robustness to annotator dropout (E1; reviewer Q11/Q13)

`src/r2/robustness.py` stresses every credibility estimator (the five `lcm.py` models by MAP, plus union, majority, vote fraction, Fisher and Stouffer) with source dropout (each annotator in turn), modality-family dropout (LIT, MOTIF, COEXP) and integrator-lineage dropout (CollecTRI, DoRothEA, Multi-integrator), in the immune/kidney/lung context universes and the global universe (primary evidence variant). Full-data MAP fits use 50 random starts; every dropout configuration uses a warm start at the restricted full-data optimum plus 5 random starts, then an adaptive global search (more random starts until >= 2 random starts reach the best mode, at most 100); each start is L-BFGS followed by Newton polishing to max|gradient| <= 1e-6. Edge-level stability also uses a size-matched top-|S| Jaccard (|S| = full-data Family-HB BFDR-0.05 set); Family-HB with one family left (lineage:DoRothEA) is reported as not identified; family/lineage summaries are given with and without family:LIT and lineage:DoRothEA; contrasts carry iid and source-clustered bootstrap intervals. It reports method-ranking stability (rank disagreement D over the top-50-per-TF method panel under posterior-weighted F1 and precision; tissue universes) and edge-level stability (Spearman, Jaccard of the Bayesian-FDR <= 0.05 set and of the top 2,000 edges, |delta credibility|, |delta w_T|). No held-out data is read.

```bash
cd /Users/ihorkendiukhov/biodyn-work
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 XLA_FLAGS='--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=2'
.venv/bin/python -W ignore subproject_24_multireference_latent_truth_model/src/r2/robustness.py fit --universes immune kidney lung
.venv/bin/python -W ignore subproject_24_multireference_latent_truth_model/src/r2/robustness.py fit --universes global
.venv/bin/python -W ignore subproject_24_multireference_latent_truth_model/src/r2/robustness.py search   # complete the adaptive search of stored fits (fit does this too)
.venv/bin/python -W ignore subproject_24_multireference_latent_truth_model/src/r2/robustness.py seedcheck --universes lung immune     # post hoc, ~10 min each
.venv/bin/python -W ignore subproject_24_multireference_latent_truth_model/src/r2/robustness.py seedcheck --universes kidney global   # (second process)
.venv/bin/python -W ignore subproject_24_multireference_latent_truth_model/src/r2/robustness.py analyze
.venv/bin/python -m pytest subproject_24_multireference_latent_truth_model/src/r2/tests/test_robustness.py -q
```

- **Outputs:** `outputs/r2/robustness/fits/<universe>/{full,drop_<hash>}.{npz,json}` (best MAP parameters, credibilities and per-start diagnostics), `fit_<universe>.log`, `analyze.log`, `analysis_meta.json`. `fit` skips stored configurations (`--force` refits).
- **Tables:** `data/r2/tables/{rob_method_ranking, rob_edge_stability, rob_config_detail, rob_method_scores_full, rob_fit_convergence, rob_fit_search, rob_estimator_contrasts, rob_mode_switches}.csv`. Each table has `analysis_status` (`post_hoc` for the `excl_uninformative` subsets, the size-matched contrasts and `rob_fit_search.csv`) and `post_hoc_columns`; decisions in `data/r2/tables/eval_analysis_decisions.csv` (module `robustness`). `rob_fit_search.csv` gives, per configuration and model, the random starts run, those at the best mode, and what the adaptive search changed relative to the warm start + 5 random starts alone. First-pass fits and tables are kept in `outputs/r2/robustness/_superseded_firstpass_{fits,tables}/`.
- **Post hoc, robfix (2026-10-14; verifier fitfix M1 and minors 1/8/11).** A latent-class dropout fit has a *posterior-mode switch* when its selected optimum is not the warm-started continuation of the full-data mode (stored warm-start gap > 1e-3; `mode_switch`, `fhb_mode_switch` in `rob_config_detail.csv`; every switch of every model in the new `rob_mode_switches.csv`, counts in `rob_fit_convergence.csv`). The ranking, edge-stability and contrast tables repeat every row without the configurations in which Family-HB switches (`subset` = `no_fhb_mode_switch`, and `excl_uninformative_no_fhb_mode_switch` for family/lineage), for every estimator, with 4,000 source-clustered bootstrap resamples drawn from separate random streams (`n_boot`), so no earlier row changes; `fhb_mode_switch_excluded` names the removed configurations. Also: `jaccard_top_obs` (the top-2,000 Jaccard only where both sets consist of observed edges; the first-pass `jaccard_top` is dominated by the seeded unobserved tie block in the tissue universes), `fit_basin_fraction`, an independent-seed check (`seedcheck`: 30 fresh random starts, key 20261014, for the 39 dropout fits whose adaptive search needed extra starts; results in `outputs/r2/robustness/seedcheck/` and the `seedcheck_*` columns of `rob_fit_search.csv`) and `values_changed_post_hoc` (prespecified rows whose point values changed in the fit_fixes round). All intervals are unadjusted 95% intervals. `seedcheck` can replace a stored fit when a fresh start finds a better optimum (the superseded copy is kept in `outputs/r2/robustness/fits/<universe>/_seedcheck_superseded/`); for such an adopted fit, `fit_seedcheck_adopted` is True and the `seedcheck_*` columns of `rob_fit_search.csv` describe the check against the superseded optimum, so the adopted optimum itself is not certified by further starts.
- **Post hoc, EM-referenced contrasts (assets2, 2026-10-14).** `rob_estimator_contrasts.csv` also holds the paired contrasts of every estimator with EM (`reference` = `em`, difference estimator - EM; every metric, scope and subset, 4,000 source-clustered resamples from two separate random streams), appended after the unchanged Family-HB-referenced rows; every such row is `post_hoc`. Readers of the table must filter on `reference`. `analyze --tables-dir X` now also writes its `analysis_meta.json` to X.


## R2 model checks: label-free sensitivity, CV, PPC, recovery (reviewer Q1, Q20)

`src/r2/fit_sensitivity.py` runs the label-free checks of the credibility models on the global and immune universes; no held-out data is read. It fits Family-HB and EM under the 8 evidence variants and Family-HB under three alternative family groupings (singleton, integrator lineage, data-driven co-reporting clusters) and four prior settings (MAP, 20 restarts), plus NUTS for the two scale priors (immune). It also runs TF-grouped 5-fold cross-validation of every estimator and grouping (held-out log-likelihood per observed edge), posterior predictive checks with real observation masks (full and LIT-block statistics, plus a leave-one-annotator-out conditional check), and simulation-based recovery and calibration with a truth-start identifiability check and NUTS coverage. A last step finds the distinct posterior modes of the primary Family-HB fit and gives the Laplace mass of each.

```bash
cd /Users/ihorkendiukhov/biodyn-work
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 XLA_FLAGS='--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=2'
.venv/bin/python subproject_24_multireference_latent_truth_model/src/r2/fit_sensitivity.py all        # everything (several hours on 2 threads)
# or by step: fits | nuts | cv [--folds ..] | ppc | recovery [--reps ..] | recovery_nuts [--reps ..] | truthstart | modes | tables
.venv/bin/python -m pytest subproject_24_multireference_latent_truth_model/src/r2/tests/test_fit_sensitivity.py -q
```

- **Outputs:** fits in `outputs/r2/fits/<universe>/<variant>/<fit_id>/` (fit storage contract), and checks in `outputs/r2/fit_sensitivity/{cv,ppc,recovery,recovery_nuts,modes,logs}/`. Primary fits are read from `outputs/r2/fits` (`fit_primary.py`).
- **Tables:** `data/r2/tables/fit_{sensitivity,sensitivity_params,family_groupings,coreporting_assoc,nuts_prior_sensitivity,cv_loglik,cv_loglik_folds,ppc,modes,sensitivity_nuts_chains,recovery,recovery_calibration,recovery_replicates}.csv`. The per-chain table was renamed `fit_sensitivity_nuts_chains.csv` on 2026-10-14: it used to be written as `fit_nuts_chains.csv`, the name of a different `fit_primary.py` table, which it overwrote whenever `fit_sensitivity.py tables` ran after `fit_primary.py tables`.

## R2 evaluation: held-out ChIP and perturbation evaluation (E2/E3/E6; reviewer Q2, Q4, Q9, Q10, Q13, Q16, Q20)

`src/r2/heldout_eval.py` evaluates every credibility estimator, the pattern and integration baselines and (tissue universes) the GRN-inference methods against held-out ChIP (ENCODE, ENCODE-strict, ENETS2) and the perturbation cohorts (`pert_pooled` = any-cohort positive, cis pairs dropped). Inference unit = TF (TF-cluster bootstrap CIs, within-TF permutation p). The confirmatory family per universe x reference is Family-HB vs {EM, Fisher, Stouffer, vote_count, DoRothEA held-in, CollecTRI held-in} on AP lift (paired bootstrap B = 10,000, Holm within the 6, within-TF permutation B = 5,000); all other contrasts are descriptive and carry a bootstrap-SE z-test and `p_floor` columns. Post hoc analyses (status-mechanism scores, TF-block down-sampling nulls, coverage-aware and matched-null method metrics, posterior-draw AP intervals, `reported_any` subset, evidence-status table) are labelled `analysis_status = post_hoc`; the dated decision log is `data/r2/tables/eval_analysis_decisions.csv`. Post hoc (2026-10-14, verifier `evalfix` M1): the composition-adjusted AP lift `ap_lift_comp_adj` = AP / its exact expectation under the within-TF label permutation (removes the gain from ranking high-prevalence TFs first; 1 = within-TF random) with TF-bootstrap CIs for every score, paired Family-HB contrasts in it for the confirmatory comparators (`heldout_paired.csv`, `metric = ap_lift_comp_adj`, B = 10,000, Holm within the 6) and 10,000-resample CIs of within-TF AUROC and Enr@500 (`confirmatory_comparator` marks the confirmatory rows of every metric); a section-6 Fisher sensitivity score (`fisher_sec6`); R-hat / mode-mixing columns of the sensitivity fits. Limitation: the TF clustering of `pert_pooled` ignores the dependence between TFs of one cohort (Norman 39 and Replogle 56 TFs share cells and controls).

```bash
cd /Users/ihorkendiukhov/biodyn-work
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
.venv/bin/python subproject_24_multireference_latent_truth_model/src/r2/heldout_eval.py all     # ~40 min on 2 threads (2026-10-14: estimators global 18 min, three tissues 16 min, reported_any 1 min, other stages 6 min; two universes in parallel)
# or by stage: refs | estimators [--universes U ..] [--subset reported_any] | estimators --merge-partials
#              | decomposition | degree | status | sensitivity
.venv/bin/python -m pytest subproject_24_multireference_latent_truth_model/src/r2/tests/test_heldout_eval.py -q
```

- **Outputs:** `outputs/r2/heldout_eval/{refs/<universe>/<ref>.{parquet,json} (input-md5-checked cache), partials/, logs/, meta_<stage>.json}`.
- **Tables:** `data/r2/tables/heldout_{estimators,paired,universe_decomposition,downsampling_null,degree_calibration,sensitivity_fits,methods_context,evidence_status}.csv`.
- **Status columns (2026-10-13):** every `heldout_*` table has `analysis_status` (row level) and `post_hoc_columns` (columns of prespecified rows added after held-out results existed). `heldout_sensitivity_fits.csv` also scores the fits of the post hoc modules (`kind = informative_observation_model` for variant `*__elig`, `posterior_mode` for `family_hb_mixture` / `family_hb_nuts_mode<k>`; both `post_hoc`) and dates every fit (`fit_mtime`). `heldout_universe_decomposition.csv` gives how many context TFs and which share of context edges and positives are evaluable in the global universe (`n_ctx_tfs_in_glob`, `frac_ctx_edges_tf_in_glob`, `frac_ctx_pos_tf_in_glob`). All tables were regenerated on 2026-10-14 from the frozen 4 x 1000/1000 `family_hb_nuts` fits (relative to the tables of 2026-10-08, kept in `outputs/r2/reports/evalfix_tables_before_rerun_20261008/`, only `family_hb_nuts` rows, the Holm families containing them and the posterior-draw intervals changed). The `posterior_mode` rows of `heldout_sensitivity_fits.csv` score the final `posterior_modes.py` fits (module complete 2026-10-14 11:50; `sensitivity` ran 12:01-12:03); re-run `sensitivity` (~3 min) if that module rewrites its fits.

## R2 evaluation: label-free claim governance (E4; reviewer Q7, Q9, Q13)

`src/r2/governance.py` builds Bayesian-FDR tiers (whole tie blocks) from the Family-HB NUTS credibility of the global and the tissue-context fits, dual tiers, posterior-draw tier intervals, the label-free deconcentration operating point, method promotion and the Q7 sensitivity grid; held-out ChIP and `pert_pooled` labels only evaluate. Post hoc additions: minimum detectable O/E and power at the global-universe effect from the margins (`mde_oe`, `power_at_global_oe`), size-matched sensitivity rows, an exact TF-matched test of method promotion, and a Pielou-evenness operating point.

```bash
cd /Users/ihorkendiukhov/biodyn-work
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
.venv/bin/python subproject_24_multireference_latent_truth_model/src/r2/governance.py all [--tissues ...] [--no-global]   # ~5 min
.venv/bin/python -m pytest subproject_24_multireference_latent_truth_model/src/r2/tests/test_governance.py -q
```

- **Outputs:** `outputs/r2/governance/{<t>/edge_tiers.parquet, labels/, meta.json, run.log}`.
- **Tables:** `data/r2/tables/gov_{tiers,consistency,promoted_by_method,tier_validation,tier_trend,concentration,deconcentration,deconcentration_eval,sensitivity_grid,promoted_edges}.csv`.
- **Status columns (2026-10-13):** every `gov_*` table has `analysis_status` and `post_hoc_columns`. `gov_tier_validation.csv` and `gov_deconcentration_eval.csv` carry `top_tf_eval` / `top_tf_share_eval` (the TF holding most evaluable edges of the validated set), because a context O/E can rest on one or two TFs. `gov_tiers.csv` reports the NUTS settings and diagnostics of the global and context fits it used (now 4 x 1000/1000 for all four universes).

## R2 evaluation: optimism of label-based tuning (reviewer Q6)

`src/r2/tuning_optimism.py` measures what R1-style tuning on perturbation labels would buy (global universe). Nineteen configurations are ranked: the twenty stored MAP fits (five estimators, Family-HB under the 8 evidence variants, 4 prior settings, 3 family groupings) minus `family_hb@no_motif_coexp`, excluded post hoc (see below). They are ranked by the AP lift on the pooled perturbation labels (`pert_pooled`; any-cohort positive, cis pairs dropped) of a random half of the perturbed TFs (50 seeded splits) or of all studies but one (leave-one-study-out). The winner is evaluated on the held-out half or study. The module reports in-sample vs out-of-sample AP lift (optimism), selection frequencies and between-half rank agreement. It compares the label-tuned choice with the label-free choices (best TF-grouped CV held-out log-likelihood; the four prior variants are cross-validated here with the folds of `fit_sensitivity.py`) and with the prespecified Family-HB. It also measures how the selection of the R1 joint objective (normalised source/block dropout robustness from `rob_method_ranking.csv` plus perturbation reward) changes over a step-0.1 weight simplex. This is the only module that uses held-out labels for a choice, and nothing it selects is used elsewhere. Post hoc (2026-10-08, labelled in the tables): the per-TF criterion is a geometric mean (switched from the arithmetic mean after SRF reached lift 342), null-normalised criteria from within-TF label permutations, study-stratified halves, LOSO without the flagged low-support study (Dixit) and per study, exclusion of `family_hb@no_motif_coexp` (unidentified T/burst labels; 19 configurations), full-label TF-bootstrap contrasts and a TF-stratified status decomposition.

```bash
cd /Users/ihorkendiukhov/biodyn-work
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 XLA_FLAGS='--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=2'
.venv/bin/python subproject_24_multireference_latent_truth_model/src/r2/tuning_optimism.py cv        # prior-variant CV (20 MAP fits, ~27 min)
.venv/bin/python subproject_24_multireference_latent_truth_model/src/r2/tuning_optimism.py analyze   # tuning, comparators, weight grid (~10-20 min: permutation nulls)
.venv/bin/python -m pytest subproject_24_multireference_latent_truth_model/src/r2/tests/test_tuning_optimism.py -q
```

- **Outputs:** `outputs/r2/tuning_optimism/{cv/global/fold*__*.json, labels_global.npz, tf_splits.csv, split_detail.csv, selector_detail.csv, config_scores_full.csv, weight_crossfit_detail.csv, run_meta.json, tuning_optimism.log}`.
- **Tables:** `data/r2/tables/tune_{optimism,selection_frequency,weight_sensitivity,weight_summary,label_sets,labelfree_cv,status_decomposition}.csv`.
- **Added 2026-10-13:** LOSO rows carry `heldout_pooled_diff_vs_primary` with a TF-cluster bootstrap CI and z-test (pooled AP lift of the selected configuration minus the primary on the held-out study; selection held fixed; on the `fold_set = study:<id>` rows of `tune_optimism.csv`); every `tune_*` table has `analysis_status` and `post_hoc_columns`. `analyze` reads no NUTS fit; it was re-run because part 3 reads `rob_method_ranking.csv`, which `robustness.py` regenerated on 2026-10-13. The prior-variant CV folds of 2026-10-08 are reused. Final `analyze` run 2026-10-14 (105 s).

## R2 evaluation: cross-cohort replication check (E5; reviewer Q5, Q8, Q17)

`src/r2/replication.py` checks whether held-out perturbation cohorts have more DE targets among the promoted edges of a TF than among its other edges. Promoted = Strong U Moderate Bayesian-FDR tiers (whole tie blocks) of the global Family-HB NUTS credibility; tissue universes are secondary. Units are cohorts and independent studies (`study_id`; Dixit d7 + d13 = one study). The within-TF test is the one-sided exact conditional (stratified hypergeometric, exact CMH) test. Step 1 (`power`) computes design-based power, minimum detectable enrichment and the power-based support floor from the stratum margins only and writes a time-stamped lock (`outputs/r2/replication/power_lock.json`, append-only `power_lock_history.jsonl`); every later step refuses to run if the recomputed design differs from the locked one. Step 2 (`outcomes`) gives O/E, MH OR, exact and permutation p, Holm, TF-cluster bootstrap CIs and Wilson pass rates; step 3 (`threshold`) re-creates the R1 data-derived support threshold with leave-one-unit-out re-estimation; steps 4-5 (`bayes`, `leavek`) fit a random-effects logistic model (numpyro NUTS) under 8 prior profiles and refit it leaving 1, 2 or 3 units out. Post hoc controls computed after the lock (columns listed in `post_hoc_columns` of `rep_units.csv`): wrong-TF control and specific excess, TF x target-responsiveness-decile strata, attainable maximum O/E (`endpoint_status = endpoint_unattainable`), endpoint power, a linear-scan floor, TF-clustered bootstrap CIs (none below 5 TFs), per-unit R-hat/ESS and a TF-level random-slope Bayesian profile.

```bash
cd /Users/ihorkendiukhov/biodyn-work
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 XLA_FLAGS='--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=2'
.venv/bin/python -W ignore subproject_24_multireference_latent_truth_model/src/r2/replication.py all   # ~25 min
# or by step (power first): power | outcomes | threshold | bayes | leavek
# 'all' keeps an existing power lock and runs steps 2-5 against it; replacing a lock needs: power --relock-reason "<why>"
.venv/bin/python -m pytest subproject_24_multireference_latent_truth_model/src/r2/tests/test_replication.py -q
```

- **Outputs:** `outputs/r2/replication/{strata_design.csv, strata_outcomes.csv, power_lock.json, power_lock_history.jsonl, lock_check.json, bayes/*_draws.npz, *.log}`.
- **Lock after the fit re-runs (2026-10-13):** the final 4 x 1000/1000 `family_hb_nuts` fits moved a few boundary edges between tiers (global Strong 4,064 -> 4,060), but the stratum design recomputed from them has the same digest as the locked design, so the lock of 2026-10-08T16:40:31Z and `rep_power.csv` are unchanged. `outcomes` writes this check to `lock_check.json` and to the `promoted_set_changed_since_lock` / `locked_design_reproduced` columns of `rep_units.csv`. Steps 2-4 were re-run on 2026-10-13 and step 5 (`leavek`) on 2026-10-14; `rep_leavek.csv` is byte-identical to the 2026-10-08 table because the locked design, and therefore every stratum outcome, is unchanged.
- **Tables:** `data/r2/tables/rep_{power,units,pass_rates,tf_strata,threshold_loo,bayes,prior_sensitivity,leavek}.csv`.

## R2 evaluation: hub-bias simulation with known truth (E6; reviewer Q9)

`src/r2/sim_hub_bias.py` simulates TF-target evidence with known truth: 400 TFs x 3,000 targets, heavy-tailed TF popularity, 12 literature annotators whose TF coverage and sensitivity rise with popularity and whose observation follows the real target-node rule, 2 motif annotators with correlated promiscuous-motif bursts, 1 coexpression annotator, and the curated-conditional observation mask. There are three scenarios (truth independent of popularity; truth mildly increasing with popularity; literature copying concentrated at popular TFs) with 20 replicates each. Per replicate it fits EM, dependence, family-regularized, Family-HB and the informative-observation models (`family_hb_obs`, `em_obs`, on eligibility-coded evidence) by multi-start MAP, and scores union, majority, vote fraction, Fisher, Stouffer and an oracle posterior (true reporting rates and priors). It compares the promoted sets (Bayesian FDR <= 0.05 by whole tie blocks, and size-matched top sets) with the truth: precision, recall, AUROC, the over-promotion index by popularity decile and literature-degree quintile, calibration by decile, and mode diagnostics (restarts at the best optimum, label swap against the truth, inflated class T).

**Status:** no held-out data is read, and the design (`SIM_DESIGN`, module docstring) was fixed before any simulation fit, but after the results checkpoint of 2026-10-08. Every table row therefore carries `analysis_status = post_hoc`; `status_basis` separates the core design (`design_fixed_after_checkpoint`), the informative-observation models (`model_designed_after_heldout`) and the comparison with the empirical table (`comparison_with_heldout_table`).

```bash
cd /Users/ihorkendiukhov/biodyn-work/subproject_24_multireference_latent_truth_model
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 XLA_FLAGS='--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=2'
# per replicate: MAR ladder about 7 min, em_obs about 0.5 min, family_hb_obs 10-15 min (20 restarts)
nice -n 10 ../.venv/bin/python -W ignore src/r2/sim_hub_bias.py fit --shard 0/2 &   # 60 replicates; stored fits are skipped; family_hb_obs only on the realised coverage (below)
nice -n 10 ../.venv/bin/python -W ignore src/r2/sim_hub_bias.py fit --shard 1/2; wait
nice -n 10 ../.venv/bin/python -W ignore src/r2/sim_hub_bias.py restart-check --models em_obs family_hb_obs   # 50-restart check (below)
nice -n 10 ../.venv/bin/python -W ignore src/r2/sim_hub_bias.py oracle-check        # ~5 s per replicate
nice -n 10 ../.venv/bin/python -W ignore src/r2/sim_hub_bias.py metrics             # 25-90 s per replicate; --only-stale after adding fits
nice -n 10 ../.venv/bin/python -W ignore src/r2/sim_hub_bias.py tables
../.venv/bin/python -m pytest src/r2/tests/test_sim_hub_bias.py -q
```

- **Outputs:** `outputs/r2/sim_hub/fits/<scenario>/rep<NN>/<model>/{cred.npy, cred_restarts.npy, class_post.npy, params.npz, restarts.csv, meta.json}`, `outputs/r2/sim_hub/metrics/<scenario>_rep<NN>.json`, `restart_check/<scenario>/rep<NN>/<model>.json`, `oracle_check/<scenario>_rep<NN>.json`, `replicates_{summary,strata}.csv`, `provenance.json`, `run.log`. `fit` skips stored fits with a matching data checksum and restart count; `--models`, `--scenarios`, `--reps` select a subset. The scenario with truth independent of popularity is `null` in commands and directories and `null_independent` in the CSV tables (pandas reads the string `null` as a missing value).
- **Tables:** `data/r2/tables/sim_hub_{summary,deciles,calibration,contrasts,modes,modes_replicates,by_mode,restart_check,obs_signature,oracle_check,datasets,design,empirical}.csv`. `sim_hub_summary.csv` is long (scenario x method x set rule x metric, with replicate mean, SD, median and t-based 95% CI); `n_reps` gives the number of replicates behind each row.
- **Realised coverage (final, 2026-10-13):** every model is fitted on all 60 replicates except `family_hb_obs`, which is fitted on 12 (`null` 0-4, `bio_hub` 0-3, `copying` 1-3; 10-35 min per fit). The fits were stopped for compute reasons when the 11 finished by then were known to select the same optimum, so these rows are not a designed-size sample; `n_reps` in every table gives the number of replicates behind a row, and `sim_hub_design.csv` (section `realised`) records the coverage and the reason. `fit` and `all` honour `REALISED_COVERAGE`, so they reproduce the reported tables; `fit --full-design` fits the remaining 48. A fit directory without `meta.json` (for example `copying/rep00/family_hb_obs`, left by a crashed fit) is ignored by every stage.
- **Reading the over-promotion index:** it is the share of a promoted set at top-popularity-decile TFs divided by the share of true edges there. The `oracle` row shows how far above 1 the index is when the reporting rates and priors are known, because true edges of little-studied TFs are rarely reported; the excess of a method over the oracle at the same set size is the part attributable to the estimator. The index falls as a set grows, so sets are compared at equal size (`size_matched`, `size_oracle`).
- **The oracle** uses the true rates and priors but, like the MAR models, skips unobserved entries. `oracle-check` compares it with the oracle that counts covered-but-unobserved literature entries as non-reports (`sim_hub_oracle_check.csv`).
- **Mode diagnostics added on 2026-10-13:** `t_class_inflated` (class-T weight above 10 times the true prevalence) and the label swap against every other class; `sim_hub_by_mode.csv` splits the set and ranking metrics by the kind of optimum that was selected. `restart-check` refits, with 50 fresh restarts, the informative-observation fits whose designed fit looked truth-aligned and records whether a higher optimum exists (`sim_hub_restart_check.csv`); the designed fits are not replaced. `sim_hub_obs_signature.csv` puts the parameter signature of the simulated informative-observation fits next to the real-data fits of `informative_observation.py` (read-only).

## R2 secondary analysis: informative-observation latent-class models (post hoc extension)

`src/r2/informative_observation.py` fits the selection-model extension of `lcm.py` in which a literature annotator's *observation* of an edge is itself an emission of the latent class (`lcm.MODEL_SPECS['family_hb_obs']`, `['em_obs']`: o_ij | z ~ Bernoulli(eta_jT) for true edges, Bernoulli(eta_jNonT) otherwise, for eligible entries = the annotator covers the TF; reports given observation as before; MOTIF/COEXP observation conditioned on). It uses eligibility-coded evidence (`evidence.build_elig_patterns`, files `patterns__elig.parquet`, `edge_pattern__elig.npy`, `meta__elig.json`; the stored evidence is unchanged) and stores fits under `outputs/r2/fits/<universe>/prespecified_cc__elig/{family_hb_obs,em_obs,*__tfw}/`. The extension was designed after the held-out results were seen: every row carries `analysis_status` (post_hoc / prespecified for reproduced MAR quantities).

**Inference and result.** The models are reported from MAP fits with Laplace intervals only; NUTS was not run (decision of 2026-10-13, below). The result is negative: as a generative model the extension is degenerate. The observation term is 96-98% of the joint log-likelihood, the latent class becomes a property of the target gene (75-83% of the variance of the credibility lies between target genes; 0.6-3% for Family-HB), and the label "true" is not anchored (`family_hb_obs` and `em_obs` select opposite labellings; each model's label-swapped optimum is the other model's best). The credibilities of `family_hb_obs` and `em_obs` must not be used as edge scores. Stages:

- `evidence`: eligibility-coded patterns of the four universes (codes 1 reported, 0 observed, -1 eligible but not observed, -2 not eligible / not modelled).
- `map`: MAP of both models (50 restarts, Newton polish, distinct optima and their Laplace mass, as `fit_primary.py`) in global, immune, kidney and lung.
- `map_tfw`: sensitivity fits `*__tfw` (20 restarts, all universes). The observation log-likelihood of literature annotator j is weighted by 1 / (number of universe TFs eligible for j). Eligibility is TF-level and observation gene-level, so the node indicator of a gene is repeated over every eligible TF; the weight counts it once per gene and annotator (composite likelihood).
- `nuts` (**not run, not part of `all`**): NUTS of `family_hb_obs` from basin-checked Laplace starts around the best MAP mode (fit id `family_hb_obs_nuts`; no such fit exists). In the non-centred coordinates of the MAR fits the sampler saturates the tree depth (immune pilot: 1,023 leapfrog steps per iteration; projected 4.7 h for immune and about 14 h for global at 4 x 1000/1000). Partially centred coordinates (`lcm.build_model(prior={'centered': ('b', 'eT', 'eN')})`, the same posterior) reduce this to about 450 steps (3.8 h for immune). Sampling one basin of a posterior whose two label-swapped basins lie thousands of log-posterior units apart would not change a conclusion, so the stage was dropped. The directories `family_hb_obs__cmode` (global, immune) hold only the start centre of the pilots; nothing reads them. If a `family_hb_obs_nuts` fit is ever produced, `heldout` and `tables` add it together with `family_hb_nuts`.
- `ppc`: posterior predictive checks with the statistics of `fit_sensitivity.py`: `conditional` (real masks kept, z drawn from P(z | o)) and `joint` (literature observation simulated too).
- `cv`: TF-grouped 5-fold cross-validation (folds of `fit_sensitivity.py`; global and immune) of log p(r | o) per observed test edge, comparable between MAR and informative-observation models, and of the joint log p(o, r).
- `heldout`: held-out ChIP / perturbation evaluation with `heldout_eval` functions (encode, encode_strict, enets2, pert_pooled in every universe), including the label-swapped optimum of each generative fit, the label-free 3-level status score and Family-HB with unobserved edges ranked last.
- `tables`: the tables below and this module's rows of `eval_analysis_decisions.csv`; then `check`.
- `check`: verifies that every PPC file, CV file, held-out table and summary table is newer than the fits it was computed from, and that the credibility files behind the held-out scores still have the md5 recorded by `heldout`. It writes `outputs/r2/informative_observation/output_check.json` and fails if anything is stale.
- `map_cond` (exploratory, not run by `all`): the same models fitted by the conditional likelihood log p(r | o) (`lcm` option `obs_conditional`). A check on immune found the objective strongly multimodal with unanchored labels (`outputs/r2/informative_observation/logs/cond_exploration_immune.log`); no fit is stored or reported.

```bash
cd /Users/ihorkendiukhov/biodyn-work/subproject_24_multireference_latent_truth_model
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 XLA_FLAGS='--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=2'
# every reported result (evidence, map, map_tfw, ppc, cv, heldout, tables); about 11 h on the shared, loaded
# laptop (MAP 8.3 h, of which global family_hb_obs 4.6 h; CV 1.3 h; PPC 0.3 h; heldout 0.3 h; tables 0.1 h)
caffeinate -i nice -n 10 ../.venv/bin/python -W ignore src/r2/informative_observation.py all
# or by stage: evidence | map [--universes ..] [--models ..] | map_tfw | ppc | cv [--folds ..] | heldout | tables | check
../.venv/bin/python -m pytest src/r2/tests/test_informative_observation.py -q
```

- **Outputs:** fits (contract above), `outputs/r2/informative_observation/{ppc,cv,logs}/`, `heldout_provenance.json` (md5 and settings of every stored credibility file behind a held-out score; the MAR fits belong to `fit_primary.py`), `output_check.json`.
- **Tables:** `data/r2/tables/obs_{fit_summary,map_optima,params,ppc,cv,heldout,heldout_paired,heldout_status}.csv`. `obs_fit_summary.csv` has one row per fit (class weights, share of the observation term in the log-likelihood, mean credibility by evidence status, variance share between target genes, mode structure); `obs_map_optima.csv` one row per distinct optimum; `obs_params.csv` eta_T and eta_NonT per literature annotator with Laplace 95% intervals, and the class-T weight with its Laplace interval. `obs_heldout.csv` holds the estimator metrics (AP lift, within-TF AUROC, Enr@k, TF x target-degree stratified enrichment, permutation p), `obs_heldout_paired.csv` the paired TF-cluster bootstrap contrasts, `obs_heldout_status.csv` the evidence-status O/E with mean credibility per fit. Score names: `<fit>_swap` = label-swapped optimum of a generative fit, `<fit>_opt2` = second optimum of a weighted fit.
- **Intervals:** the Laplace intervals describe one posterior basin under the model's independent-edge likelihood. Node indicators are repeated over 37-270 eligible TFs per annotator, so the intervals understate the uncertainty, and they say nothing about the label-swapped basin.
- **Run the `heldout` and `tables` stages after `fit_primary.py`** has finished any re-run of the MAR MAP fits they read (`family_hb`, `em`). The primary NUTS fits (`family_hb_nuts`) are not read.

## R2 secondary analysis: informative-observation credibilities in governance and replication (post hoc)

`src/r2/obs_integration.py` puts the MAP credibilities of `family_hb_obs` and `em_obs` (no NUTS fit exists) through the two downstream uses of a credibility, with the code of `governance.py` and `replication.py`, next to `family_hb_nuts` (prespecified primary), `family_hb` and `em` (MAR MAP). Every row is `analysis_status = post_hoc`, except `family_hb_nuts` rows that reproduce a row of a prespecified table exactly (they take that row's status and name the table in `reproduces`; the run fails if any does not reproduce). Prespecified tables are not written.

- `governance`: Bayesian-FDR tiers (global-fit levels in the global universe; global-fit, context-fit and dual tiers in each tissue), held-out tier validation (O/E, TF and TF x target-degree expectations, TF-cluster CIs; `governance.global_validation` / `validation_rows`), Mantel trend tests (Holm within fit x scope x strata) and tier composition (evidence status of the tier edges, TF concentration, Jaccard with the primary set). Minimum-detectable-O/E columns are computed for `family_hb_nuts` only.
- `degree`: hub over-promotion by TF literature-degree quintile (`heldout_eval.run_degree` definitions) for the Strong and Promoted sets of every universe; the hub index of a set is its quintile-5 `over_promotion`.
- `replication`: study-level replication check (comparators `universe` / `observed`): O/E, MH OR, exact and permutation p, TF-cluster bootstrap CIs, endpoint, responsiveness-stratified O/E and the wrong-TF control, with the seeds of `replication.run_outcomes`. The step-1 power lock belongs to the primary promoted set; no power columns are computed.

```bash
cd /Users/ihorkendiukhov/biodyn-work/subproject_24_multireference_latent_truth_model
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
caffeinate -i nice -n 10 ../.venv/bin/python -W ignore src/r2/obs_integration.py all     # about 1 min
# or by stage: governance | degree | replication | check (md5 freshness of the inputs)
../.venv/bin/python -m pytest src/r2/tests/test_obs_integration.py -q
```

- **Tables:** `data/r2/tables/obs_governance_tiers.csv` (tier composition), `obs_governance.csv` (tier validation), `obs_governance_trend.csv`, `obs_governance_degree.csv`, `obs_replication.csv`; `outputs/r2/obs_integration/meta.json` (inputs with md5, reproduction check).
- **Result:** the obs credibilities do not give claims. `family_hb_obs` promotes 85% of the global universe and 79-85% of each tissue universe (mostly unobserved edges), `em_obs` 37% and 29-33% (mostly observed-not-reported edges); held-out O/E of these sets is close to 1 after TF x target-degree matching, the hub index falls to about 1 only because the sets cover most of the universe, and no global study passes the replication endpoint (`em_obs` O/E 1.1-1.3 equals its wrong-TF O/E; its one tissue pass, kidney Replogle, rests on 9 positives and 3 TFs). Run after `informative_observation.py`, `governance.py`, `heldout_eval.py degree` and `replication.py outcomes` (it reproduces their primary rows).

## R2 model checks: identifiability of Family-HB (posterior modes; post hoc)

`src/r2/posterior_modes.py` (post hoc, designed after the held-out checkpoint; reads no held-out data) estimates the posterior mass of the two leading Family-HB modes without a Gaussian assumption: NUTS restricted to each mode (basin-checked Laplace starts at the primary MAP optima, 4 x 1000/1000 in the tissues; global mode 1 4 x 500/500, global mode 0 = the canonical primary NUTS fit; every chain checked to stay in its mode by nearest-optimum credibility and L-BFGS basin checks), bridge sampling of each mode's marginal likelihood (Meng-Wong iteration, Gronau et al. 2017; multivariate-normal proposal fitted in the CENTRED hierarchical coordinates with an exact Jacobian, N2 = 5 N1; relative MC error from the Fruehwirth-Schnatter formula and 20 repeated splits; the textbook proposal in the raw non-centred coordinates, a Student-t proposal and, for mode 0, the independent canonical primary NUTS draws are reported as checks), mode probabilities vs Laplace, the mixture credibility `family_hb_mixture` (c_mix = sum_m p_m c_m with NUTS means per mode) and the sensitivity of the dual-universe BFDR tiers to the second-mode mass (`governance.py` tier functions). It calibrates mode selection by simulation from the immune and global MAP fits, plus a reverse check generated at the real second immune optimum. Since 2026-10-14 (module `modes2`, fix of verifier B1) the primary simulator is design-faithful (`faithful_cc`, `simulate_cc`): it keeps the real LIT masks, simulates the LIT reports and REGENERATES the curated-conditional MOTIF/COEXP masks from the simulated DoRothEA-channel LIT reports exactly as `evidence.py` builds them (real per-edge `via_dorothea` flags where the annotator has a real record, else the source's / per-(source, TF) channel rule; whole-resource TF coverage; `cc_design` checks that the rule reproduces every real MOTIF/COEXP mask from the real reports). A second faithful simulator (`faithful_cc_strat`) makes the class weights depend on the LIT observation mask (mean real posterior per mask stratum at the generating parameters) and reproduces the real LIT positive counts and about 90% of the informative MOTIF/COEXP observations (`modes_faithful_design.csv`); LIT node-set masks are not regenerated (`modes_lit_node_check.csv` says why). Replicates: immune 100 x 20 restarts, reverse check 30 x 20, global 20 x 10 per faithful simulator; the first 20 immune / 5 global replicates also form the recovery study under each faithful simulator. The earlier fixed-mask simulation (immune 100, global 10, reverse 30; every real mask kept, which decouples the MOTIF/COEXP masks from the LIT reports) is kept as a labelled sensitivity (`simulator` column of `modes_simulation*.csv` and `fit_recovery*.csv`). The module also augments the recovery tables of `fit_sensitivity.py` (realised FDR, ECE over bins 1-9, w_T bias/RMSE by mode, NUTS coverage from converged single-mode chain sets only), gives the mode gaps of every MAP sensitivity fit and the stability of the data-driven grouping (ARI over CV folds and 50 TF bootstraps). Primary MAP optima are read from the frozen snapshot `outputs/r2/posterior_modes/primary_map_snapshot/` (MANIFEST.sha256; equal to the canonical polished optima to 1e-7, `snapshot_check.csv`); the mode-0 NUTS mean used in the mixture and governance stages is the canonical `family_hb_nuts`. Dated design decisions are appended to `data/r2/tables/eval_analysis_decisions.csv` (module `posterior_modes`).

```bash
cd /Users/ihorkendiukhov/biodyn-work
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 XLA_FLAGS='--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=2'
caffeinate -i nice -n 10 .venv/bin/python -W ignore subproject_24_multireference_latent_truth_model/src/r2/posterior_modes.py all   # many hours; or by stage:
# nuts --universes immune kidney lung global [--modes 0 1] | bridge | mixture | governance
# | simulate [--universes immune_cc immune_truthmode1_cc immune_ccs immune_truthmode1_ccs global_cc global_ccs]
#   (default: the six design-faithful labels; the fixed-mask labels immune / global / immune_truthmode1 are stored)
# | sensmodes --universes immune global | clusterstab | tables
# (stages are resumable; --force recomputes; at most 2 compute processes)
.venv/bin/python -m pytest subproject_24_multireference_latent_truth_model/src/r2/tests/test_posterior_modes.py -q
```

- **Outputs:** `outputs/r2/fits/<u>/prespecified_cc/{family_hb_nuts_mode0,family_hb_nuts_mode1,family_hb_mixture}/`, `outputs/r2/posterior_modes/{bridge,sim,sensmodes,logs}/`, `clusterstab.parquet`.
- **Tables:** `data/r2/tables/modes_{bridge,governance_sensitivity,simulation,simulation_replicates,simulation_calibration,faithful_design,lit_node_check}.csv`, `fit_sensitivity_modes.csv`, `fit_grouping_stability.csv`, `fit_recovery_nuts_chainsets.csv`, and new columns/rows in `fit_recovery*.csv` (`simulator`, `post_hoc_columns`; recovery-NUTS coverage under the module's lenient rule and the project's frozen flag rule, kinds `nuts_coverage_single_mode` / `nuts_coverage_single_mode_frozen_rule`).
- **Simulation outputs:** `outputs/r2/posterior_modes/sim/<label>/rep###.json` (+ `_optima.npz`), labels in `SIM_LABEL_INFO`; the fixed-mask design statistics are cached in `outputs/r2/posterior_modes/cc_design/`.

## R2 compute cost and scalability (E8; reviewer Q21/Q22)

`src/r2/compute_scaling.py` measures how the pattern-compressed latent-class pipeline scales on synthetic evidence; it reads no held-out data and does not time the real pipeline. Synthetic TF x gene grids of N = 1e5, 1e6, 1e7 edges with K = 10, 21, 42 annotators, a real-size grid and a 10x-real grid (93.2M edges, K = 21 and 42) keep the TF-blocked observation structure of the real global evidence: each synthetic TF / gene draws the coverage / node-set profile of a real one (`template` generator), with `independent` (per-annotator rates of `evidence_summary.csv`) and `stress` (dense independent observation, compression does not help) as sensitivity generators. For every configuration it records the number of unique patterns P, the wall time and peak RSS of streaming compression by TF chunk (and of whole-matrix `np.unique` where feasible), the time of one log-posterior + gradient evaluation, the time of a 10-restart Family-HB MAP fit and of NUTS (2 chains x 300/300), each in a fresh subprocess with the load average. Budget rules fixed in advance decide which large runs are measured, extrapolated from completed restarts, or projected; every number carries its status. Two sessions exist: `a` (complete grid, 2026-10-08/09, heavily loaded machine) and `b` (replicate, 2026-10-13), which also checks that regenerated patterns, L-BFGS paths and NUTS step counts are identical. The complexity derivation is in the module docstring and `scale_complexity.csv`.

```bash
cd /Users/ihorkendiukhov/biodyn-work
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 XLA_FLAGS='--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=2'
nice -n 10 .venv/bin/python -W ignore subproject_24_multireference_latent_truth_model/src/r2/compute_scaling.py all        # session a + tables (hours)
nice -n 10 .venv/bin/python -W ignore subproject_24_multireference_latent_truth_model/src/r2/compute_scaling.py replicate  # session b + tables
# or by stage: template | compress | map | nuts [--session a|b] | tables   (stored measurements are kept unless --force)
.venv/bin/python -m pytest subproject_24_multireference_latent_truth_model/src/r2/tests/test_compute_scaling.py -q
```

- **Outputs:** `outputs/r2/scale/{template.npz,template.json}` (frozen snapshot of the real evidence structure and generating parameters; `template` rebuilds it and then every stage needs `--force`), `patterns/<cid>.npz`, `runs/` and `runs_b/` (`{compress,map,nuts}/<cid>.json`, one per measurement and session), `logs/`, `logs_b/`.
- **Tables:** `data/r2/tables/scale_{summary,compression,map,map_restarts,nuts,fits,replicate,generator_check,complexity,environment}.csv`. `scale_summary.csv` gives one row per configuration (most direct record over the sessions, then the shortest time); the other tables carry a `session` column.

## R2 manuscript assets: LaTeX tables

`src/r2/paper_assets/tables.py` (main tables `tab_*`) and `tables_supp.py` (supplementary tables `tabS_*`) write `paper/r2/tables/*.tex` from `data/r2/tables/*.csv`; they read no other data and edit no CSV. Every printed number is read from a CSV row by `textable.Table`, formatted with one fixed format per quantity type (ratios 2 decimals, p-values 2 significant digits in scientific notation, counts with `{,}` thousands separators) and registered with its CSV file, row and column; `outputs/r2/paper_assets/tables_cells.csv` lists all of them. Each `.tex` starts with a provenance comment (source CSVs, their modification times, row filters), uses booktabs without vertical rules in `\small`, carries caption and label, and marks post hoc rows or columns (from `analysis_status`, `post_hoc_columns` or the dated decision log `eval_analysis_decisions.csv`) with a dagger and a note. Long supplementary tables are split into several floats in one file, because the R1 preamble has no longtable. `--check-latex` compiles every table with the R1 preamble (`paper/main.tex` lines 1-30) and reports errors, overfull boxes and floats too large for the page. `--stale PREFIX ...` marks tables built from CSVs that were still being regenerated (a `% WARNING` line and `outputs/r2/paper_assets/tables_stale_sources.json`).

```bash
cd /Users/ihorkendiukhov/biodyn-work
.venv/bin/python subproject_24_multireference_latent_truth_model/src/r2/make_paper_assets.py tables --check-latex
# or: cd subproject_24_multireference_latent_truth_model/src && ../../.venv/bin/python -m r2.paper_assets.tables [--only tab_fit ...] [--out DIR] [--check-latex] [--stale heldout_ ...]
.venv/bin/python -m pytest subproject_24_multireference_latent_truth_model/src/r2/tests/test_paper_assets_tables.py -q
```

The tests rebuild every table in memory, re-read each registered CSV cell independently and require that its formatted value is the printed text and occurs in the `.tex` (derived numbers are checked against their source cells), and compile all tables with `pdflatex` (no errors, no overfull box over 1 pt, no float too large).

## R2 manuscript assets: main figures

`src/r2/paper_assets/figures.py` writes the main figures `fig1_design` ... `fig8_replication` to `paper/r2/figures/<name>.pdf` (vector, 16 cm wide), `<name>.png` (300 dpi) and `<name>_caption.txt` (a `% provenance:` header with source CSVs, modification times and row filters, followed by a LaTeX caption draft). Every plotted or printed value is read from `data/r2/tables/*.csv`. Design constants come from `r2.common.PRESPECIFIED` and from the constants of the analysis modules, and no number is typed by hand. The same provenance is embedded in the PDF and PNG metadata. `src/r2/paper_assets/style.py` is shared by all asset modules: fonts (STIX, 7-9 pt), page geometry, a fixed Okabe-Ito colour/marker mapping per estimator, reference, family, tier and universe, number formatters and the `Provenance` helper. Each figure has `data_<name>()` (tidy data read from the CSVs), `plot_<name>()` and `caption_<name>()`; `FIGURE_SOURCES` lists the CSVs each figure depends on.

```bash
cd /Users/ihorkendiukhov/biodyn-work
.venv/bin/python subproject_24_multireference_latent_truth_model/src/r2/make_paper_assets.py figures
# or: cd subproject_24_multireference_latent_truth_model/src && ../../.venv/bin/python -m r2.paper_assets.figures [fig4_heldout ...] [--out-dir DIR]
.venv/bin/python -m pytest subproject_24_multireference_latent_truth_model/src/r2/tests/test_paper_assets.py -q
```

The tests check:
- that the palette uses Okabe-Ito colours only;
- the formatters;
- CSV round trips of the plotted values;
- that every figure renders at the 16 cm text width and 300 dpi with a provenance header;
- that every number printed in a caption is a value of the figure's data, rounded as printed.

**Figure floats.** Every figure (main and supplementary) is also written as `paper/r2/figures/<name>_float.tex`, a `figure` float that includes `<name>.pdf` at its printed width with the generated caption and a fixed label (`fig:design`, `fig:model`, `fig:robustness`, `fig:heldout`, `fig:universe`, `fig:governance`, `fig:hubs`, `fig:replication`; `fig:S-ppc`, `fig:S-obs`, `fig:S-deconc`, `fig:S-gov-grid`, `fig:S-scaling`, `fig:S-fm-null`, `fig:S-recovery`, `fig:S-leavek`; `style.FIG_LABEL`). Captions are checked before writing (`style.latex_caption_problems`: no non-ASCII characters, no unescaped `%`, balanced `$` and braces). `tab_compute.tex` holds two floats: the per-stage pipeline timings (`tab:compute`, from `pipeline_stage_timings.csv`) and the synthetic scaling experiment (`tab:compute-scaling`); `tab_robustness.tex` holds one float with two tabulars (`Table.render_blocks`).

**Decision log.** The presentation decisions of the asset scripts (modules `paper_assets` and `paper_assets_binding`, post hoc) are kept in `src/r2/paper_assets/decisions.py` and written to `data/r2/tables/eval_analysis_decisions.csv` with `cd subproject_24_multireference_latent_truth_model/src && ../../.venv/bin/python -m r2.paper_assets.decisions` (idempotent).

**Prose numbers.** Every number in the manuscript prose (`paper/r2/sec_*.tex`, the front and back matter of `manuscript.tex`, the prose of `supplement.tex`) is a macro `\nXxx` of `paper/r2/numbers.tex`, which `manuscript.tex` reads with `\InputIfFileExists{numbers.tex}`. `src/r2/paper_assets/prose_numbers_spec.py` defines one specification per macro: a CSV of `data/r2/tables/` with a pandas row filter and a column (without aggregation the selected rows must hold one value; otherwise `min`, `max`, `mean`, `sum`, `count`, `nunique`, `absmax` or `same`, the last requiring that all selected rows print the same), a `PRESPECIFIED` constant, a frozen configuration value or module constant, or a documented arithmetic of earlier macros, with a fixed format (bounds of "at least" / "at most" statements are rounded towards the safe side). `src/r2/paper_assets/prose_numbers.py` writes the macros with a provenance comment each, writes the audit `outputs/r2/reports/binding_number_audit.csv` (macro, value, CSV, filter, column, use sites, the value printed before the binding) and checks the prose: it fails on any literal number outside equation environments, references and citations, years, identifiers, the interval level 95% and the documented allow-list (`ALLOW`, `MATH_ALLOW`), on undefined macros and on defined but unused macros. The module is deliberately not called `numbers.py`, which would shadow the standard-library module when a `paper_assets` script is run directly. `reproduce_all.py` runs it as the last step of the `paper_assets` stage and again after the stage-timing table is rewritten.

```bash
.venv/bin/python subproject_24_multireference_latent_truth_model/src/r2/make_paper_assets.py numbers           # write numbers.tex + audit, then check
.venv/bin/python subproject_24_multireference_latent_truth_model/src/r2/make_paper_assets.py numbers --check   # check only (exit 1 on any problem)
.venv/bin/python -m pytest subproject_24_multireference_latent_truth_model/src/r2/tests/test_paper_assets_numbers.py -q
```

**Submission correspondence.** The response to the reviewers, the cover letter, the declarations form, the submission bundle and the scripts that build them (`letter_numbers.py`, `build_declarations.py`, `build_submission.py`) are confidential peer-review correspondence and are kept out of the public repository (`.gitignore`). The reviewer digest `paper/r2/digest/reviewer_points.md` is written only where the review correspondence is present and is not part of the release.

`make_paper_assets.py all` builds tables, figures (main and supplementary, with their float wrappers), digests and the prose number macros. Compile the documents supplement first (they cross-reference through `xr-hyper`): `cd paper/r2 && latexmk -pdf supplement && latexmk -pdf manuscript`, or by hand `pdflatex supplement` twice and then `pdflatex manuscript` three times (from a clean checkout the second manuscript pass can still report changed labels; the third converges), followed by one more `pdflatex supplement` if the supplement refers to manuscript labels.

**Building the manuscript.** After the assets: compile in the order below (cross-references through `xr-hyper`), then build the marked copy.

```bash
cd subproject_24_multireference_latent_truth_model/paper/r2
latexmk -pdf supplement && latexmk -pdf manuscript && latexmk -g -pdf supplement && latexmk -g -pdf manuscript
cd /Users/ihorkendiukhov/biodyn-work
.venv/bin/python subproject_24_multireference_latent_truth_model/paper/r2/marked/build_marked.py      # marked copy vs paper/main_clean.tex
```

`marked/build_marked.py` flattens `manuscript.tex` (every `\input` inlined, every `numbers.tex` macro replaced by its printed value: `marked/manuscript_flat.tex`) and runs `latexdiff --type=UNDERLINE --math-markup=coarse --graphics-markup=new-only` against the previously submitted `paper/main_clean.tex`, with tabular, figure and table environments and the reference list diffed as a whole (they appear in their current form without marks: word-level markup inside tables does not compile, captions of moved floats would be diffed against unrelated text, and deleted references would be merged into current ones); citations in deleted text of references the current list no longer contains are printed as author-year text, and a legend under the title explains the marks; references in deleted text to labels that no longer exist are printed with their old numbers (from `paper/main_clean.aux`). Output `marked/manuscript_marked.tex` / `.pdf`.

## Dependencies

- Python 3.9+
- numpy, scipy, pandas, matplotlib
- All model fitting is CPU-only (no GPU, no deep learning frameworks)


## Citation

If you use this framework, please cite the paper:

```
Kendiukhov, I. (2026). Latent Truth Without a Gold Standard: Multi-Reference
Credibility Modeling and Claim Governance for Gene-Regulatory-Network Benchmarking.
Statistical Applications in Genetics and Molecular Biology (under review).
```
