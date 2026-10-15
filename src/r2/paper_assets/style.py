"""Shared style, colour mapping, labels, formatters and provenance helpers.

This module is shared by ``tables``, ``figures`` and ``digest``.  It defines
ONE fixed mapping from every estimator / score, reference, evidence family,
claim tier and universe to a colour of the Okabe-Ito palette (and, for scores,
a marker), so that the same entity looks the same in every figure.

Page geometry follows the SAGMB template used for the paper: A4 with 2.5 cm
margins, i.e. a 16 cm text width; full-width figures are 16 cm wide and
half-width figures 7.8 cm.
"""
from __future__ import annotations

import datetime as _dt
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------- paths
SP = Path(__file__).resolve().parents[3]          # subproject root
TABLES = SP / "data" / "r2" / "tables"            # analysis CSVs (read-only here)
PAPER = SP / "paper" / "r2"
TAB_DIR = PAPER / "tables"
FIG_DIR = PAPER / "figures"
DIGEST_DIR = PAPER / "digest"
REPORTS = SP / "outputs" / "r2" / "reports"

# ----------------------------------------------------------------------------- geometry
CM = 1.0 / 2.54
TEXT_WIDTH_CM = 16.0
FULL_W = 16.0 * CM          # inches
HALF_W = 7.8 * CM           # inches
DPI_PNG = 300

# ----------------------------------------------------------------------------- palette
OKABE_ITO = {
    "black": "#000000",
    "orange": "#E69F00",
    "skyblue": "#56B4E9",
    "green": "#009E73",      # bluish green
    "yellow": "#F0E442",
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "purple": "#CC79A7",     # reddish purple
    "grey": "#999999",
}
OI = OKABE_ITO

# Latent-class estimators each have their own colour; combination rules share
# bluish green, integration schemes black, random grey (marker distinguishes
# members of a group).  Expression / attention methods of the tissue universes
# never share a panel with the latent-class estimators; they use squares and
# diamonds so that they cannot be mistaken for estimators (circles).


@dataclass(frozen=True)
class ScoreStyle:
    label: str
    group: str
    color: str
    marker: str
    short: str = ""


SCORE_STYLE: dict[str, ScoreStyle] = {
    # latent-class estimators (ladder order)
    "em": ScoreStyle("EM baseline", "lcm", OI["blue"], "o", "EM"),
    "dependence": ScoreStyle("Dependence-aware", "lcm", OI["skyblue"], "o", "Dep."),
    "multifactor": ScoreStyle("Multifactor", "lcm", OI["purple"], "o", "Multif."),
    "family_regularized": ScoreStyle("Family-regularized", "lcm", OI["orange"], "o", "Fam.-reg."),
    "family_hb": ScoreStyle("Family-HB", "lcm", OI["vermillion"], "o", "Fam.-HB"),
    "family_hb_nuts": ScoreStyle("Family-HB (NUTS)", "lcm", OI["vermillion"], "h", "Fam.-HB NUTS"),
    "family_hb_globalfit": ScoreStyle("Family-HB (global fit)", "lcm_globalfit", OI["vermillion"], "p", "Fam.-HB glob."),
    "family_hb_unobserved_last": ScoreStyle("Family-HB, unobserved last", "post_hoc_mechanism", OI["vermillion"], "8", "Fam.-HB unobs. last"),
    # informative-observation extension (secondary)
    "em_obs": ScoreStyle("EM + observation", "lcm_obs", OI["blue"], "d", "EM+obs"),
    "family_hb_obs": ScoreStyle("Family-HB + observation", "lcm_obs", OI["vermillion"], "d", "Fam.-HB+obs"),
    # simulation truth
    "oracle": ScoreStyle("Oracle posterior", "oracle", OI["black"], "*", "Oracle"),
    # combination rules (pattern baselines)
    "union": ScoreStyle("Union", "pattern_baseline", OI["green"], "v", "Union"),
    "majority": ScoreStyle("Majority (≥2)", "pattern_baseline", OI["green"], "^", "Majority"),
    "vote_count": ScoreStyle("Vote count", "pattern_baseline", OI["green"], "s", "Votes"),
    "vote_fraction": ScoreStyle("Vote fraction", "pattern_baseline", OI["green"], "D", "Vote frac."),
    "n_families_reporting": ScoreStyle("Families reporting", "pattern_baseline", OI["green"], "P", "Families"),
    "fisher": ScoreStyle("Fisher", "pattern_baseline", OI["green"], "<", "Fisher"),
    "stouffer": ScoreStyle("Stouffer", "pattern_baseline", OI["green"], ">", "Stouffer"),
    "status3": ScoreStyle("3-level status", "post_hoc_mechanism", OI["green"], "X", "Status"),
    # established integration schemes (held-in evidence only)
    "dorothea_level_heldin": ScoreStyle("DoRothEA level", "integration", OI["black"], "s", "DoRothEA"),
    "collectri_heldin": ScoreStyle("CollecTRI", "integration", OI["black"], "D", "CollecTRI"),
    # random
    "random": ScoreStyle("Random", "random", OI["grey"], "x", "Random"),
    # tissue-universe methods
    "scgpt_attn": ScoreStyle("scGPT attention", "method", OI["skyblue"], "s", "scGPT"),
    "scgpt_attn_T": ScoreStyle("scGPT attention (transposed)", "method", OI["skyblue"], "s", "scGPT T"),
    "scgpt_attn_norm": ScoreStyle("scGPT attention, normalised", "method", OI["skyblue"], "D", "scGPT norm."),
    "scgpt_attn_uniform": ScoreStyle("scGPT uniform (1/L) null", "method", OI["skyblue"], "x", "scGPT 1/L"),
    "geneformer_attn": ScoreStyle("Geneformer attention", "method", OI["purple"], "s", "Geneformer"),
    "geneformer_attn_norm": ScoreStyle("Geneformer attention, normalised", "method", OI["purple"], "D", "Geneformer norm."),
    "geneformer_attn_uniform": ScoreStyle("Geneformer uniform (1/L) null", "method", OI["purple"], "x", "Geneformer 1/L"),
    "grnboost2": ScoreStyle("GRNBoost2", "method", OI["green"], "s", "GRNBoost2"),
    "genie3": ScoreStyle("GENIE3", "method", OI["green"], "D", "GENIE3"),
    "pearson_abs": ScoreStyle("|Pearson|", "method", OI["orange"], "s", "|Pearson|"),
    "spearman_abs": ScoreStyle("|Spearman|", "method", OI["orange"], "D", "|Spearman|"),
    "all_edges": ScoreStyle("All universe edges", "method", OI["black"], "|", "All edges"),
}

ESTIMATOR_ORDER = ["em", "dependence", "multifactor", "family_regularized", "family_hb"]
RULE_ORDER = ["union", "majority", "vote_count", "vote_fraction", "n_families_reporting",
              "fisher", "stouffer"]
INTEGRATION_ORDER = ["dorothea_level_heldin", "collectri_heldin"]

GROUP_LABEL = {
    "lcm": "Latent-class estimators",
    "lcm_globalfit": "Latent-class (global fit)",
    "lcm_obs": "Informative-observation extension",
    "post_hoc_mechanism": "Post hoc mechanism scores",
    "pattern_baseline": "Combination rules",
    "integration": "Integration schemes",
    "random": "Random",
    "method": "Network-inference methods",
    "oracle": "Oracle",
    "baseline": "Combination rules",
    "latent_class": "Latent-class estimators",
}


def score_label(key: str) -> str:
    s = SCORE_STYLE.get(key)
    return s.label if s else key


def score_color(key: str) -> str:
    s = SCORE_STYLE.get(key)
    return s.color if s else OI["grey"]


def score_marker(key: str) -> str:
    s = SCORE_STYLE.get(key)
    return s.marker if s else "o"


# references (held-out validation data)
REFERENCE_LABEL = {
    "encode": "ENCODE TFBS",
    "encode_strict": "ENCODE-strict",
    "enets2": "ENETS2",
    "pert_pooled": "Pooled perturbation",
    "dixit_d7": "Dixit d7",
    "dixit_d13": "Dixit d13",
    "dixit2016": "Dixit 2016",
    "adamson": "Adamson",
    "adamson2016": "Adamson 2016",
    "norman": "Norman",
    "norman2019": "Norman 2019",
    "papalexi": "Papalexi",
    "papalexi2021": "Papalexi 2021",
    "replogle": "Replogle",
    "replogle2022": "Replogle 2022",
    "shifrut": "Shifrut",
    "shifrut2018": "Shifrut 2018",
}
REFERENCE_COLOR = {
    "encode": OI["blue"],
    "encode_strict": OI["skyblue"],
    "enets2": OI["purple"],
    "pert_pooled": OI["green"],
}
REFERENCE_MARKER = {"encode": "o", "encode_strict": "o", "enets2": "s", "pert_pooled": "^"}


def reference_label(key: str) -> str:
    return REFERENCE_LABEL.get(key, key)


# evidence families
FAMILY_LABEL = {"LIT": "Literature", "MOTIF": "Motif", "COEXP": "Coexpression", "CHIP": "ChIP"}
FAMILY_COLOR = {"LIT": OI["blue"], "MOTIF": OI["orange"], "COEXP": OI["green"], "CHIP": OI["purple"]}
FAMILY_MARKER = {"LIT": "o", "MOTIF": "s", "COEXP": "D", "CHIP": "^"}

# claim tiers
TIER_ORDER = ["Strong", "Moderate", "Conditional", "Exploratory"]
TIER_COLOR = {
    "Strong": OI["vermillion"],
    "Moderate": OI["orange"],
    "Conditional": OI["skyblue"],
    "Exploratory": OI["grey"],
    "Below": OI["grey"],
    "Promoted": OI["vermillion"],
    "Not promoted": OI["grey"],
}

# evidence status of an edge (heldout_evidence_status.csv); monochrome + shape so
# that it never competes with the estimator colours
STATUS_ORDER = ["unobserved", "observed_neg", "reported"]
STATUS_LABEL = {"unobserved": "Unobserved", "observed_neg": "Observed, not reported",
                "reported": "Reported"}
STATUS_STYLE = {
    "unobserved": dict(marker="o", color=OI["black"], filled=False),
    "observed_neg": dict(marker="s", color=OI["grey"], filled=True),
    "reported": dict(marker="^", color=OI["black"], filled=True),
}

# simulation scenarios (sim_hub_*.csv)
SCENARIO_ORDER = ["null_independent", "bio_hub", "copying"]
SCENARIO_LABEL = {"null_independent": "Null (independent)", "bio_hub": "Biological hubs",
                  "copying": "Hub-concentrated copying"}

# universes
UNIVERSE_ORDER = ["global", "immune", "kidney", "lung"]
UNIVERSE_LABEL = {"global": "Global", "immune": "Immune", "kidney": "Kidney", "lung": "Lung",
                  "all_tissues": "All tissues"}
UNIVERSE_COLOR = {"global": OI["black"], "immune": OI["blue"], "kidney": OI["orange"],
                  "lung": OI["green"]}
UNIVERSE_MARKER = {"global": "o", "immune": "s", "kidney": "D", "lung": "^"}

# robustness protocols
PROTOCOL_LABEL = {"source": "Source dropout", "family": "Family dropout",
                  "lineage": "Lineage dropout"}

# ----------------------------------------------------------------------------- matplotlib
FONT_SIZES = {"base": 7.5, "small": 6.5, "label": 7.5, "title": 8.0, "panel": 9.0}


def apply_style() -> None:
    """Set matplotlib rcParams for print figures (serif, Times-compatible STIX)."""
    import matplotlib as mpl

    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["STIXGeneral", "STIX Two Text", "Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": FONT_SIZES["base"],
        "axes.titlesize": FONT_SIZES["title"],
        "axes.labelsize": FONT_SIZES["label"],
        "xtick.labelsize": FONT_SIZES["small"] + 0.5,
        "ytick.labelsize": FONT_SIZES["small"] + 0.5,
        "legend.fontsize": FONT_SIZES["small"] + 0.5,
        "legend.frameon": False,
        "legend.handlelength": 1.2,
        "legend.handletextpad": 0.4,
        "legend.borderaxespad": 0.3,
        "legend.columnspacing": 0.9,
        "axes.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "axes.titlepad": 3.0,
        "axes.labelpad": 2.0,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.minor.width": 0.4,
        "ytick.minor.width": 0.4,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "xtick.minor.size": 1.5,
        "ytick.minor.size": 1.5,
        "xtick.major.pad": 1.5,
        "ytick.major.pad": 1.5,
        "lines.linewidth": 1.0,
        "lines.markersize": 3.5,
        "errorbar.capsize": 0,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.dpi": DPI_PNG,
        "figure.dpi": 100,
        "svg.fonttype": "none",
    })


# ----------------------------------------------------------------------------- formatting
def fmt_ratio(x, digits: int = 2) -> str:
    """Ratio / effect size with a fixed number of decimals (default 2)."""
    if x is None or (isinstance(x, float) and not math.isfinite(x)) or pd.isna(x):
        return "--"
    return f"{float(x):.{digits}f}"


def fmt_p(p, sig: int = 2, latex: bool = True) -> str:
    """p-value in scientific notation with ``sig`` significant digits."""
    if p is None or pd.isna(p):
        return "--"
    p = float(p)
    if p == 0:
        return "0"
    mant, exp = f"{p:.{sig - 1}e}".split("e")
    exp = int(exp)
    if latex:
        return f"${mant}\\times 10^{{{exp}}}$"
    return f"{mant}e{exp:+03d}"


def fmt_count(n, latex: bool = True) -> str:
    """Integer count with thousands separators ({,} in LaTeX)."""
    if n is None or pd.isna(n):
        return "--"
    s = f"{int(round(float(n))):,}"
    return s.replace(",", "{,}") if latex else s


def fmt_count_plain(n) -> str:
    return fmt_count(n, latex=False)


def fmt_num(x, digits: int = 2) -> str:
    if x is None or pd.isna(x):
        return "--"
    return f"{float(x):.{digits}f}"


def fmt_pct(x, digits: int = 0) -> str:
    """Fraction in [0, 1] printed as a percentage (no % sign)."""
    if x is None or pd.isna(x):
        return "--"
    return f"{100.0 * float(x):.{digits}f}"


def fmt_sig(x, sig: int = 2) -> str:
    """Generic number with ``sig`` significant digits (plain text)."""
    if x is None or pd.isna(x):
        return "--"
    x = float(x)
    if x == 0:
        return "0"
    mag = int(math.floor(math.log10(abs(x))))
    digits = max(sig - 1 - mag, 0)
    return f"{x:.{digits}f}"


# ----------------------------------------------------------------------------- provenance
@dataclass
class Provenance:
    """Records the CSV files and row filters a generated asset was built from."""

    asset: str
    sources: dict = field(default_factory=dict)   # csv name -> list of filter strings

    def read(self, name: str, filt: str | None = None, **kw) -> pd.DataFrame:
        path = TABLES / (name if name.endswith(".csv") else name + ".csv")
        df = pd.read_csv(path, **kw)
        self.note(path.name, filt)
        return df

    def note(self, csv_name: str, filt: str | None = None) -> None:
        lst = self.sources.setdefault(csv_name, [])
        if filt and filt not in lst:
            lst.append(filt)

    def lines(self) -> list[str]:
        out = [f"provenance: {self.asset} generated by src/r2/paper_assets "
               f"on {_dt.datetime.now().strftime('%Y-%m-%d %H:%M')} from data/r2/tables/"]
        for name, filts in self.sources.items():
            f = "; ".join(filts) if filts else "all rows"
            if (TABLES / name).exists():
                mtime = _dt.datetime.fromtimestamp((TABLES / name).stat().st_mtime).strftime("%Y-%m-%d %H:%M")
                out.append(f"  {name} (modified {mtime}): {f}")
            else:
                out.append(f"  {name}: {f}")
        return out

    def tex_comment(self) -> str:
        return "\n".join("% " + ln for ln in self.lines()) + "\n"

    def text(self) -> str:
        return "\n".join(self.lines())


def table_mtime(name: str) -> _dt.datetime:
    path = TABLES / (name if name.endswith(".csv") else name + ".csv")
    return _dt.datetime.fromtimestamp(path.stat().st_mtime)


def finite(x) -> bool:
    try:
        return x is not None and not pd.isna(x) and np.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def as_bool(x) -> bool:
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if isinstance(x, str):
        return x.strip().lower() in {"true", "1", "yes"}
    if x is None or pd.isna(x):
        return False
    return bool(x)


def iter_unique(seq: Iterable) -> list:
    seen, out = set(), []
    for s in seq:
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out


# ----------------------------------------------------------------------------- LaTeX figure floats
FIG_LABEL = {
    "fig1_design": "fig:design", "fig2_model": "fig:model", "fig3_robustness": "fig:robustness",
    "fig4_heldout": "fig:heldout", "fig5_universe": "fig:universe", "fig6_governance": "fig:governance",
    "fig7_hubs": "fig:hubs", "fig8_replication": "fig:replication",
    "figS1_ppc": "fig:S-ppc", "figS2_obs_model": "fig:S-obs", "figS3_deconcentration": "fig:S-deconc",
    "figS4_gov_grid": "fig:S-gov-grid", "figS5_compute_scaling": "fig:S-scaling",
    "figS6_fm_attention_null": "fig:S-fm-null", "figS7_recovery_calibration": "fig:S-recovery",
    "figS8_replication_leavek": "fig:S-leavek",
}


def latex_caption_problems(cap: str) -> list[str]:
    """Reasons why ``cap`` is not a LaTeX-safe caption (empty list = fine): non-ASCII characters,
    unescaped %, unbalanced $ or braces, Python artefacts (nan, None)."""
    import re as _re
    out = []
    bad = sorted({ch for ch in cap if ord(ch) > 127})
    if bad:
        out.append("non-ASCII characters: " + " ".join(bad))
    if _re.search(r"(?<!\\)%", cap):
        out.append("unescaped %")
    if len(_re.findall(r"(?<!\\)\$", cap)) % 2:
        out.append("unbalanced $")
    depth = 0
    for i, ch in enumerate(cap):
        if ch == "{" and (i == 0 or cap[i - 1] != "\\"):
            depth += 1
        elif ch == "}" and (i == 0 or cap[i - 1] != "\\"):
            depth -= 1
            if depth < 0:
                break
    if depth != 0:
        out.append("unbalanced braces")
    if _re.search(r"\bnan\b|\bNone\b|\binf\b", cap):
        out.append("nan/None/inf in the text")
    return out


def write_float(name: str, caption: str, prov: "Provenance", width_in: float, out_dir: Path | None = None) -> Path:
    """Write ``<name>_float.tex``: a ``figure`` float that includes ``<name>.pdf`` at its printed width (fraction of
    the text width from the figure size), with the caption and the fixed label of ``FIG_LABEL``."""
    probs = latex_caption_problems(caption)
    if probs:
        raise ValueError(f"{name}: caption is not LaTeX-safe: {'; '.join(probs)}")
    frac = width_in / FULL_W
    width = "\\textwidth" if frac >= 0.985 else f"{frac:.2f}\\textwidth"
    out_dir = Path(out_dir or FIG_DIR)
    p = out_dir / f"{name}_float.tex"
    lines = ["% generated by src/r2/paper_assets (make_paper_assets.py figures); do not edit by hand",
             *("% " + ln for ln in prov.lines()),
             "\\begin{figure}[htbp]" if name.startswith("figS") else "\\begin{figure}[tbp]", "  \\centering", f"  \\includegraphics[width={width}]{{{name}.pdf}}",
             f"  \\caption{{{caption.strip()}}}", f"  \\label{{{FIG_LABEL[name]}}}", "\\end{figure}"]
    p.write_text("\n".join(lines) + "\n")
    return p
