# Multi-Reference Latent Truth Model for GRN Benchmarking

A framework for estimating per-edge regulatory credibility from multiple noisy reference databases, with dual-universe evaluation and conservative claim governance. All analyses run on laptop-scale table-level pipelines with no GPU requirement.

**Paper**: `paper/main.pdf` (11 pages, 6 figures, 3 tables, 15 references)

**Author**: Ihor Kendiukhov, University of Tübingen


## Key Results

- **Hierarchical Family-Bayes** achieves lowest rank disagreement under both source-dropout and family-block-dropout stress protocols (0.067-0.072 mean disagreement vs. 0.104-0.139 for alternatives).
- **Context conditioning** produces a 23-31x calibration uplift over naïve full-universe evaluation, demonstrating that context matching is critical for meaningful GRN benchmarking.
- **Dual-universe governance** restricts promotion to 842 out of 421,521 candidate edges (0.2%), with 2.2x perturbation-backed enrichment (p << 0.05) on pre-registered external panels.
- **LOCO replication** across 6 perturbation cohorts yields a 5/6 pass rate, with the single failure cleanly attributable to sparse reference support (<10 edges), providing an operational reliability threshold.


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
| §5.2 Context conditioning | `src/evaluation/build_context_matched_universe.py`, `calibrate_perturbation_claim_tiers.py` | `data/paper_tables/calibration_full_vs_context.csv` |
| §5.3 Governance tiers | `src/governance/enforce_dual_universe_claim_governance.py` | `data/paper_tables/dual_universe_claim_tiers_summary.csv` |
| §5.4 Source concentration | `src/governance/run_deconcentration_policy_sweep.py`, `run_deconcentration_budget_sensitivity.py` | `data/paper_tables/deconcentration_*.csv` |
| §5.5 External enrichment | `src/replication/run_preregistered_external_validation.py` | `outputs/top_tier_readiness_v*/` |
| §5.6 LOCO replication | `src/replication/run_multilevel_leave_one_cohort_out.py` | `outputs/top_tier_readiness_v7/multilevel_leave_one_cohort_out_v1/` |
| §5.7 Cohort fragility | `src/replication/run_multilevel_leave_two_cohort_out.py`, `run_multilevel_leave_three_cohort_out.py` | `outputs/top_tier_readiness_v7/multilevel_leave_*_cohort_out_v1/` |
| Figures 1-6 | `paper/generate_figures.py` | `data/paper_tables/*.csv` |


## Reproducing the Paper Figures

```bash
cd paper/
python generate_figures.py    # reads from ../data/paper_tables/
pdflatex main.tex             # compile the paper
```

Figures are generated from `data/paper_tables/` CSVs using matplotlib. No additional dependencies beyond numpy and matplotlib are required.


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


## Dependencies

- Python 3.9+
- numpy, scipy, pandas, matplotlib
- All model fitting is CPU-only (no GPU, no deep learning frameworks)


## Citation

If you use this framework, please cite the paper:

```
Kendiukhov, I. (2026). Latent Truth Without a Gold Standard: Multi-Reference
Credibility Modeling and Claim Governance for Gene-Regulatory-Network Benchmarking.
```
