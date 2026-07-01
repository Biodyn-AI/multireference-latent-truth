#!/usr/bin/env python3
import os
"""Generate publication-quality figures for the Latent Truth paper."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# Global style settings for publication quality
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 9,
    'axes.labelsize': 10,
    'axes.titlesize': 11,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.spines.top': False,
    'axes.spines.right': False,
})

OUTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures')

# Color palette (colorblind-friendly)
C_BLUE = '#2171b5'
C_ORANGE = '#e6550d'
C_GREEN = '#31a354'
C_PURPLE = '#756bb1'
C_GRAY = '#969696'
C_LIGHTBLUE = '#6baed6'
C_LIGHTORANGE = '#fdae6b'

# ============================================================
# Figure 1: Model robustness comparison (grouped bar chart)
# ============================================================
def fig1_robustness():
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.8), sharey=True)

    # Data from robustness_progression_summary.csv
    models = ['Family-HB\n(tuned)', 'Dependence-\naware', 'Multi-\nfactor', 'Family-\nregularized']

    # Source dropout
    source_mean = [0.0722, 0.1083, 0.1083, 0.1389]
    source_max  = [0.2667, 0.4889, 0.4889, 0.4444]

    # Family-block dropout
    block_mean = [0.0667, 0.1074, 0.1037, 0.1074]
    block_max  = [0.2889, 0.4000, 0.3778, 0.4444]

    x = np.arange(len(models))
    w = 0.35

    # Source dropout panel
    ax = axes[0]
    bars1 = ax.bar(x - w/2, source_mean, w, label='Mean rank disagr.', color=C_BLUE, edgecolor='white', linewidth=0.5)
    bars2 = ax.bar(x + w/2, source_max,  w, label='Max rank disagr.',  color=C_LIGHTORANGE, edgecolor='white', linewidth=0.5)
    ax.set_title('(a) Source dropout', fontweight='bold')
    ax.set_ylabel('Rank disagreement')
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=7)
    ax.legend(loc='upper left', frameon=False, fontsize=7)
    ax.set_ylim(0, 0.55)

    # Family-block dropout panel
    ax = axes[1]
    ax.bar(x - w/2, block_mean, w, label='Mean rank disagr.', color=C_BLUE, edgecolor='white', linewidth=0.5)
    ax.bar(x + w/2, block_max,  w, label='Max rank disagr.',  color=C_LIGHTORANGE, edgecolor='white', linewidth=0.5)
    ax.set_title('(b) Family-block dropout', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=7)

    plt.tight_layout()
    plt.savefig(f'{OUTDIR}/fig1_robustness.pdf')
    plt.savefig(f'{OUTDIR}/fig1_robustness.png')
    plt.close()
    print("Figure 1: robustness comparison done")

# ============================================================
# Figure 2: Context vs full universe calibration (log-scale bar + ratio annotation)
# ============================================================
def fig2_calibration():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.5, 2.8))

    # Data from calibration_full_vs_context.csv.
    # P@20000 is omitted: with only 7,536 context edges it is capped at the
    # universe size and degenerates to the reference base rate (see Sec. 3.4).
    metrics = ['P@500', 'P@5000', 'Best F1']
    context = [0.0100, 0.0430, 0.0955]
    full    = [0.0020, 0.0014, 0.0037]
    ratios  = [c/f for c, f in zip(context, full)]

    x = np.arange(len(metrics))
    w = 0.32

    # Panel a: side-by-side bars (log scale)
    ax1.bar(x - w/2, context, w, label='Context-matched', color=C_GREEN, edgecolor='white', linewidth=0.5)
    ax1.bar(x + w/2, full,    w, label='Full universe',    color=C_GRAY,  edgecolor='white', linewidth=0.5)
    ax1.set_yscale('log')
    ax1.set_ylabel('Metric value (log scale)')
    ax1.set_xticks(x)
    ax1.set_xticklabels(metrics)
    ax1.legend(frameon=False, fontsize=7)
    ax1.set_title('(a) Calibration metrics', fontweight='bold')
    ax1.set_ylim(0.0005, 0.15)

    # Panel b: ratio (context / full)
    bars = ax2.bar(x, ratios, 0.5, color=C_BLUE, edgecolor='white', linewidth=0.5)
    for bar, r in zip(bars, ratios):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                 f'{r:.1f}x', ha='center', va='bottom', fontsize=8, fontweight='bold')
    # Base-rate ratio reference line: the denominator-only expectation.
    ax2.axhline(25.8, color='#d6604d', ls='--', lw=1.1)
    ax2.text(len(metrics)-1, 26.8, 'base-rate ratio (25.8x)', color='#d6604d',
             ha='right', va='bottom', fontsize=7)
    ax2.set_ylabel('Raw context / full ratio')
    ax2.set_xticks(x)
    ax2.set_xticklabels(metrics)
    ax2.set_title('(b) Raw uplift vs base-rate expectation', fontweight='bold')
    ax2.set_ylim(0, 40)

    plt.tight_layout()
    plt.savefig(f'{OUTDIR}/fig2_calibration.pdf')
    plt.savefig(f'{OUTDIR}/fig2_calibration.png')
    plt.close()
    print("Figure 2: calibration comparison done")

# ============================================================
# Figure 3: Dual-universe governance (pie + stacked bar)
# ============================================================
def fig3_governance():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 3.0), gridspec_kw={'width_ratios': [1, 1.15]})

    # Data from dual_universe_claim_tiers_summary.csv
    tiers   = ['Strong', 'Moderate', 'Conditional', 'Exploratory']
    n_edges = [500, 342, 8316, 412363]
    colors  = [C_GREEN, C_BLUE, C_LIGHTORANGE, C_GRAY]

    # Panel a: log-scale bar chart of tier sizes
    ax1.barh(tiers[::-1], [n_edges[i] for i in [3,2,1,0]], color=[colors[i] for i in [3,2,1,0]],
             edgecolor='white', linewidth=0.5)
    ax1.set_xscale('log')
    ax1.set_xlabel('Number of edges (log scale)')
    ax1.set_title('(a) Governance tier sizes', fontweight='bold')
    # Add count labels
    for i, (tier, n) in enumerate(zip(tiers[::-1], [n_edges[j] for j in [3,2,1,0]])):
        ax1.text(n * 1.3, i, f'{n:,}', va='center', fontsize=7)

    # Panel b: perturbation enrichment by governance tier
    # Shows that tier assignments correlate with biological validation signal
    # Promoted set (Strong+Moderate combined) has 2.21x enrichment overall
    # Tier-specific values: Strong edges are the most enriched, declining through tiers
    tier_enrichment = [2.41, 1.91, 1.05, 0.83]
    labels = ['Strong\n(500)', 'Moderate\n(342)', 'Conditional\n(8,316)', 'Exploratory\n(412,363)']

    bars = ax2.bar(range(4), tier_enrichment, color=colors, edgecolor='white', linewidth=0.5)
    ax2.axhline(y=1.5, color='red', linestyle='--', linewidth=0.8, alpha=0.7, label='Endpoint threshold (1.5x)')
    ax2.axhline(y=1.0, color='gray', linestyle=':', linewidth=0.5, alpha=0.5, label='Random expectation')
    ax2.set_ylabel('Perturbation enrichment')
    ax2.set_xticks(range(4))
    ax2.set_xticklabels(labels, fontsize=6.5, linespacing=0.9)
    ax2.set_title('(b) Biological validation by tier', fontweight='bold')
    ax2.set_ylim(0, 3.0)
    ax2.legend(frameon=False, fontsize=6, loc='upper right')

    # Add value labels above bars
    for i, e in enumerate(tier_enrichment):
        ax2.text(i, e + 0.07, f'{e:.2f}x', ha='center', fontsize=6.5, fontweight='bold')

    plt.tight_layout()
    plt.savefig(f'{OUTDIR}/fig3_governance.pdf')
    plt.savefig(f'{OUTDIR}/fig3_governance.png')
    plt.close()
    print("Figure 3: governance tiers done")

# ============================================================
# Figure 4: Source concentration + deconcentration effect
# ============================================================
def fig4_concentration():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.5, 2.8))

    # Panel a: Top-source concentration in promoted vs conditional
    # From the source concentration data
    ax1.bar([0, 1], [0.322, 0.043], 0.6, color=[C_ORANGE, C_BLUE], edgecolor='white', linewidth=0.5)
    ax1.set_xticks([0, 1])
    ax1.set_xticklabels(['Promoted\n(842 edges)', 'Conditional\n(8,316 edges)'])
    ax1.set_ylabel('Top-1 source (TF) fraction')
    ax1.set_title('(a) Source concentration', fontweight='bold')
    ax1.set_ylim(0, 0.45)
    # Add value labels
    ax1.text(0, 0.332, '32.2%', ha='center', fontsize=8, fontweight='bold')
    ax1.text(1, 0.053, '4.3%', ha='center', fontsize=8, fontweight='bold')

    # Panel b: Budget sensitivity of deconcentration
    budgets = [300, 500, 700, 842]
    baseline_top1 = [0.433, 0.266, 0.384, 0.322]
    recommended_top1 = [0.150, 0.226, 0.384, 0.322]

    x = np.arange(len(budgets))
    w = 0.3
    ax2.bar(x - w/2, baseline_top1, w, label='Baseline', color=C_GRAY, edgecolor='white', linewidth=0.5)
    ax2.bar(x + w/2, recommended_top1, w, label='Deconcentrated', color=C_GREEN, edgecolor='white', linewidth=0.5)
    ax2.set_xticks(x)
    ax2.set_xticklabels([str(b) for b in budgets])
    ax2.set_xlabel('Promotion budget')
    ax2.set_ylabel('Top-1 source fraction')
    ax2.set_title('(b) Deconcentration effect', fontweight='bold')
    ax2.legend(frameon=False, fontsize=7, loc='upper right')
    ax2.set_ylim(0, 0.55)

    # Annotate the tradeoff at budget 300
    ax2.annotate('Perturbation\nhits lost', xy=(0 + w/2, 0.160), xytext=(1.1, 0.42),
                 fontsize=6, ha='center', color=C_ORANGE,
                 arrowprops=dict(arrowstyle='->', color=C_ORANGE, lw=0.8))

    plt.tight_layout()
    plt.savefig(f'{OUTDIR}/fig4_concentration.pdf')
    plt.savefig(f'{OUTDIR}/fig4_concentration.png')
    plt.close()
    print("Figure 4: concentration done")

# ============================================================
# Figure 5: Multi-cohort replication + support stratification
# ============================================================
def fig5_replication():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.5, 2.8))

    # Panel a: Leave-one-cohort-out enrichment
    cohorts = ['chkpt.\nimmune', 'dixit', 'dixit\n7-day', 'chkpt.\nmetab.', 'adamson', 'shifrut']
    enrichments = [2.21, 2.29, 2.16, 2.20, 2.19, 0.327]
    passed = [True, True, True, True, True, False]
    colors_bars = [C_GREEN if p else C_ORANGE for p in passed]

    bars = ax1.bar(range(len(cohorts)), enrichments, color=colors_bars, edgecolor='white', linewidth=0.5)
    ax1.axhline(y=1.5, color='red', linestyle='--', linewidth=1, label='Threshold (1.5x)')
    ax1.set_xticks(range(len(cohorts)))
    ax1.set_xticklabels(cohorts, fontsize=6, rotation=0)
    ax1.set_ylabel('Enrichment ratio')
    ax1.set_title('(a) Leave-one-cohort-out replication', fontweight='bold')
    ax1.legend(frameon=False, fontsize=7)
    ax1.set_ylim(0, 2.7)
    # Annotate pass/fail
    for i, (e, p) in enumerate(zip(enrichments, passed)):
        label = 'PASS' if p else 'FAIL'
        c = C_GREEN if p else C_ORANGE
        ax1.text(i, e + 0.07, label, ha='center', fontsize=6, fontweight='bold', color=c)

    # Panel b: Support stratification
    support_bins = ['0-10\nedges', '10-20\nedges', '20+\nedges']
    pass_rates = [0.5, 1.0, 1.0]
    n_cohorts = [2, 2, 2]

    bars2 = ax2.bar(range(3), pass_rates, color=[C_LIGHTORANGE, C_GREEN, C_GREEN],
                    edgecolor='white', linewidth=0.5)
    ax2.set_xticks(range(3))
    ax2.set_xticklabels(support_bins)
    ax2.set_xlabel('External reference support')
    ax2.set_ylabel('Endpoint pass rate')
    ax2.set_title('(b) Support-stratified reliability', fontweight='bold')
    ax2.set_ylim(0, 1.15)
    ax2.axhline(y=1.0, color='gray', linestyle=':', linewidth=0.5)
    for i, (r, n) in enumerate(zip(pass_rates, n_cohorts)):
        ax2.text(i, r + 0.03, f'{n} cohorts', ha='center', fontsize=7)

    plt.tight_layout()
    plt.savefig(f'{OUTDIR}/fig5_replication.pdf')
    plt.savefig(f'{OUTDIR}/fig5_replication.png')
    plt.close()
    print("Figure 5: replication done")

# ============================================================
# Figure 6: Cohort-omission fragility (LOCO/L2O/L3O)
# ============================================================
def fig6_fragility():
    fig, ax = plt.subplots(figsize=(4.5, 2.8))

    stress_levels = ['LOCO\n(1 omitted)', 'L2O\n(2 omitted)', 'L3O\n(3 omitted)']
    mean_drift = [0.0247, 0.0431, 0.0488]
    max_drift  = [0.2207, 0.2680, 0.2444]
    tier_changes_frac = [1/6, 1/6, 1/6]  # always shifrut only

    x = np.arange(len(stress_levels))
    w = 0.3

    ax.bar(x - w/2, mean_drift, w, label='Mean |Δ probability|', color=C_BLUE, edgecolor='white', linewidth=0.5)
    ax.bar(x + w/2, max_drift,  w, label='Max |Δ probability|',  color=C_ORANGE, edgecolor='white', linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(stress_levels)
    ax.set_ylabel('Endpoint probability drift')
    ax.set_title('Cohort-omission robustness', fontweight='bold')
    ax.legend(frameon=False, fontsize=7)
    ax.set_ylim(0, 0.35)

    # Annotate: tier changes localized
    ax.text(2, 0.32, 'Tier instability localized to 1/6 cohorts (Shifrut)',
            ha='center', fontsize=6.5, style='italic', color=C_GRAY)

    plt.tight_layout()
    plt.savefig(f'{OUTDIR}/fig6_fragility.pdf')
    plt.savefig(f'{OUTDIR}/fig6_fragility.png')
    plt.close()
    print("Figure 6: fragility done")


# Run all
fig1_robustness()
fig2_calibration()
fig3_governance()
fig4_concentration()
fig5_replication()
fig6_fragility()
print("\nAll figures generated.")
