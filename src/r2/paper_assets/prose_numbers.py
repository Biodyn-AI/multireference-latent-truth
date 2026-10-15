"""Number macros for the manuscript prose of the SAGMB R2 paper, and a literal-number checker.

``python src/r2/make_paper_assets.py numbers`` (or ``python -m r2.paper_assets.prose_numbers`` with ``src``
on ``sys.path``)

1. evaluates every macro specification of ``prose_numbers_spec.specs()``,
2. writes ``paper/r2/numbers.tex``: one ``\\newcommand{\\nXxx}{value}`` per number used in the prose,
   each preceded by a provenance comment (CSV file, pandas row filter, column, aggregation, or the
   ``PRESPECIFIED`` key / frozen config entry / documented arithmetic it comes from),
3. writes ``outputs/r2/reports/binding_number_audit.csv`` (macro, value, source, use sites), and
4. evaluates every wording condition (``require``): a sentence whose wording depends on a relation
   between numbers ("all", "never", "not distinguishable", "below random", ...) carries a condition on
   the macro values, and the build fails when the regenerated values no longer satisfy it, and
5. runs the checker: the prose files (``paper/r2/sec_*.tex``, the front and back matter of
   ``manuscript.tex`` and the prose of ``supplement.tex``) must not contain a literal number other than
   equation constants, years in a date or author-year context (or in :data:`YEAR_ALLOW`),
   cross-reference / citation arguments, interval levels and the explicitly allow-listed literals in
   :data:`ALLOW` and identifiers in :data:`IDENT_ALLOW` (digits glued to letters, e.g. ``ENETS2``);
   it also rejects words that refer to a review or revision history (:data:`ARTEFACT_RE`).  Every
   ``\\nXxx`` macro used must be defined and every defined non-helper macro must be used.

Specification rules (see ``prose_numbers_spec.py``)
---------------------------------------------
* ``csv`` + ``query`` (``DataFrame.query``, python engine) + ``col``: without ``agg`` the selected rows
  must hold one distinct value (otherwise the build fails: ambiguous filter); ``agg`` is one of
  ``min max mean median sum count nunique``.
* ``const``: ``PRESPECIFIED:<key>[<index>]`` (``r2.common.PRESPECIFIED``), ``json:<path>:<dotted.key>``
  (frozen configuration file) or ``code:<path>:<NAME>`` (module-level literal, read with ``ast``).
* ``expr``: arithmetic over the raw values of earlier macros, written ``{nName}``; ``scale``
  multiplies the raw value (unit conversion).  Both are documented in the provenance comment.
* ``fmt`` fixes the printed form: ``rN`` (N decimals), ``floorN`` / ``ceilN`` (rounded towards the
  safe side, for "at least" / "at most" statements), ``int`` (``{,}`` thousands separators), ``eN``
  (N significant digits, ``m\\times10^{e}``), ``pctN`` (percent without the % sign), ``sigN``,
  ``word`` / ``Word`` (English number word), ``fold`` (``twofold``), ``str``.  Negative values and
  powers of ten are wrapped in ``\\ensuremath`` so that a macro works in text and in math mode.
* ``require``: a boolean expression over the raw values of any macros (``{nName}``; ``min max abs
  round`` available) that must hold for the wording of the sentence using the macro; ``require_note``
  names that wording.  It is evaluated after all macros, printed in the provenance comment, and a
  false or failing condition is a build error.
"""
from __future__ import annotations

import argparse
import ast
import json
import math
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from r2.paper_assets import style as S  # type: ignore
else:
    from . import style as S
from r2.common import PRESPECIFIED  # noqa: E402

NUMBERS_TEX = S.PAPER / "numbers.tex"
AUDIT_CSV = S.REPORTS / "binding_number_audit.csv"
MACRO_RE = re.compile(r"\\(n[A-Z][A-Za-z]*)")


# ----------------------------------------------------------------------------- specification
@dataclass
class Num:
    name: str                       # macro name without backslash: 'n' + letters only
    csv: str | None = None
    query: str | None = None
    col: str | None = None
    fmt: str = "r2"
    agg: str | None = None
    const: str | None = None
    expr: str | None = None
    scale: float | None = None
    regex: str | None = None        # extract group 1 of this regex from a string cell (documented parse)
    note: str = ""                  # meaning / documented arithmetic
    helper: bool = False            # used only by other macros (may be unused in the prose)
    was: str | None = None          # literal printed in the prose before binding (audit only)
    require: str | None = None      # wording condition over macro raw values (see module docstring)
    require_note: str = ""          # the wording that depends on the condition
    # filled by evaluate()
    raw: object = None
    tex: str = ""
    status: str = ""                # post hoc flag of the selected CSV rows
    error: str = ""

    @property
    def source(self) -> str:
        if self.csv:
            q = self.query or "all rows"
            agg = f" | {self.agg}" if self.agg else ""
            sc = f" | x {self.scale:g}" if self.scale is not None else ""
            rx = f" | regex {self.regex}" if self.regex else ""
            return f"{self.csv} | {q} | {self.col}{agg}{rx}{sc}"
        if self.const:
            return self.const
        if self.expr:
            sc = f" x {self.scale:g}" if self.scale is not None else ""
            return f"= {self.expr}{sc}"
        return "?"


# ----------------------------------------------------------------------------- formatting
WORDS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
         "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen",
         "nineteen", "twenty"]
FOLD = {2: "twofold", 3: "threefold", 4: "fourfold", 5: "fivefold", 10: "tenfold"}
ORD = ["zeroth", "first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth", "tenth",
       "eleventh", "twelfth", "thirteenth", "fourteenth", "fifteenth", "sixteenth", "seventeenth", "eighteenth",
       "nineteenth", "twentieth"]
FRAC = {0.0: "none", 0.25: "a quarter", 0.5: "half", 0.75: "three quarters", 1.0: "all"}


def _minus(s: str) -> str:
    return rf"\ensuremath{{{s}}}" if s.startswith("-") else s


def _round_dir(x: float, nd: int, how: str) -> float:
    f = 10.0 ** nd
    eps = 1e-9
    return (math.floor(x * f + eps) if how == "floor" else math.ceil(x * f - eps)) / f


def fmt_tex(x, kind: str) -> str:
    """LaTeX text of value ``x`` in format ``kind`` (see the module docstring)."""
    if kind == "str":
        s = str(x)
        return s.replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")
    if x is None or (isinstance(x, float) and not math.isfinite(x)) or (not isinstance(x, str) and pd.isna(x)):
        raise ValueError(f"value {x!r} is not finite")
    if isinstance(x, (bool, np.bool_)):
        raise ValueError("boolean value; use a count")
    x = float(x)
    m = re.fullmatch(r"(r|floor|ceil|pct|sig|e)(\d+)", kind)
    if kind == "int":
        return _minus(f"{int(round(x)):,}".replace(",", "{,}"))
    if kind in ("word", "Word"):
        n = int(round(x))
        if abs(n - x) > 1e-9 or not 0 <= n < len(WORDS):
            raise ValueError(f"{x} has no number word")
        w = WORDS[n]
        return w.capitalize() if kind == "Word" else w
    if kind in ("wfloor", "wround", "Wround"):
        n = int(math.floor(x + 1e-9)) if kind == "wfloor" else int(round(x))
        if not 0 <= n < len(WORDS):
            raise ValueError(f"{x} has no number word")
        return WORDS[n].capitalize() if kind == "Wround" else WORDS[n]
    if kind == "ord":
        n = int(round(x))
        if abs(n - x) > 1e-9 or not 0 <= n < len(ORD):
            raise ValueError(f"{x} has no ordinal word")
        return ORD[n]
    if kind == "frac":
        for v, w in FRAC.items():
            if abs(x - v) < 1e-9:
                return w
        raise ValueError(f"{x} has no fraction word")
    if kind == "ordpct":
        n = int(round(100.0 * x))
        suf = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
        return f"{n}{suf}"
    if kind == "pow10":
        k = math.log10(x) if x > 0 else float("nan")
        if not math.isfinite(k) or abs(k - round(k)) > 1e-9:
            raise ValueError(f"{x} is not a power of ten")
        return rf"\ensuremath{{10^{{{int(round(k))}}}}}"
    if kind in ("lgfloor", "lgceil"):
        if x <= 0:
            raise ValueError("order of magnitude of a non-positive value")
        k = math.floor(math.log10(x)) if kind == "lgfloor" else math.ceil(math.log10(x))
        return rf"\ensuremath{{10^{{{int(k)}}}}}"
    m2 = re.fullmatch(r"approx(\d)", kind)
    if m2:
        sig = int(m2.group(1))
        if x == 0:
            return "0"
        mag = int(math.floor(math.log10(abs(x))))
        v = round(x, sig - 1 - mag)
        return _minus(f"{int(round(v)):,}".replace(",", "{,}")) if mag >= sig - 1 else _minus(f"{v:g}")
    if kind == "fold":
        n = int(round(x))
        if abs(n - x) > 1e-9 or n not in FOLD:
            raise ValueError(f"{x} has no -fold word")
        return FOLD[n]
    if m is None:
        raise ValueError(f"unknown format {kind!r}")
    k, nd = m.group(1), int(m.group(2))
    if k == "r":
        return _minus(f"{x:.{nd}f}")
    if k in ("floor", "ceil"):
        return _minus(f"{_round_dir(x, nd, k):.{nd}f}")
    if k == "pct":
        return _minus(f"{100.0 * x:.{nd}f}")
    if k == "sig":
        return _minus(S.fmt_sig(x, nd))
    if k == "e":
        if x == 0:
            return "0"
        mant, ex = f"{x:.{nd - 1}e}".split("e")
        return rf"\ensuremath{{{mant}\times10^{{{int(ex)}}}}}"
    raise ValueError(kind)


def plain(tex: str) -> str:
    """Plain-text reading of a macro value (for the audit and the old/new comparison)."""
    t = tex.replace(r"\ensuremath{", "").replace("{,}", ",").replace(r"\times10^{", "e")
    t = t.replace("}", "").replace("{", "").replace(r"\_", "_")
    return t


# ----------------------------------------------------------------------------- evaluation
class _Tables:
    def __init__(self):
        self.cache: dict[str, pd.DataFrame] = {}

    def get(self, name: str) -> pd.DataFrame:
        if name not in self.cache:
            self.cache[name] = pd.read_csv(S.TABLES / name, low_memory=False)
        return self.cache[name]


def _post_hoc_status(sel: pd.DataFrame, col: str) -> str:
    out = []
    if "analysis_status" in sel.columns:
        st = set(sel["analysis_status"].dropna().astype(str))
        if "post_hoc" in st:
            out.append("post_hoc" if len(st) == 1 else "partly_post_hoc")
    if "post_hoc_columns" in sel.columns:
        ph = set()
        for v in sel["post_hoc_columns"].dropna().astype(str):
            ph |= {c.strip() for c in v.split(";")}
        if col in ph or any(p.endswith("*") and col.startswith(p[:-1]) for p in ph):
            out.append("post_hoc_column")
    return ";".join(out)


def _const(spec: str):
    kind, _, rest = spec.partition(":")
    if kind == "PRESPECIFIED":
        mm = re.fullmatch(r"([A-Za-z0-9_]+)(?:\[(\d+)\])?", rest)
        if not mm:
            raise ValueError(f"bad PRESPECIFIED spec {spec!r}")
        v = PRESPECIFIED[mm.group(1)]
        return v[int(mm.group(2))] if mm.group(2) is not None else v
    if kind == "json":
        path, _, key = rest.partition(":")
        v = json.loads((S.SP / path).read_text())
        for k in key.split("."):
            v = v[k]
        return v
    if kind == "code":
        path, _, name = rest.partition(":")
        tree = ast.parse((S.SP / path).read_text())
        for node in tree.body:
            if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
                return ast.literal_eval(node.value)
        raise KeyError(f"{name} not a module-level literal in {path}")
    raise ValueError(f"unknown constant source {spec!r}")


_SAFE = {"__builtins__": {}, "min": min, "max": max, "abs": abs, "round": round}


def evaluate(specs: list[Num]) -> list[Num]:
    T = _Tables()
    raw: dict[str, object] = {}
    for n in specs:
        try:
            if n.csv:
                df = T.get(n.csv)
                q = n.query
                if q:
                    refs = re.findall(r"\{(n[A-Za-z]+)\}", q)
                    missing = [x for x in refs if x not in raw]
                    if missing:
                        raise KeyError(f"query refers to undefined/later macros {missing}")
                    q = re.sub(r"\{(n[A-Za-z]+)\}", lambda mm: repr(float(raw[mm.group(1)])), q)
                sel = df.query(q, engine="python") if q else df
                if sel.empty:
                    raise ValueError("filter selects no rows")
                if n.col not in sel.columns:
                    raise KeyError(f"column {n.col!r} not in {n.csv}")
                vals = sel[n.col]
                n.status = _post_hoc_status(sel, n.col)
                if n.agg is None:
                    u = vals.drop_duplicates()
                    if len(u) > 1:
                        raise ValueError(f"{len(sel)} rows with {len(u)} distinct values (ambiguous filter)")
                    v = vals.iloc[0]
                elif n.agg == "count":
                    v = len(sel)
                elif n.agg == "nunique":
                    v = vals.nunique()
                elif n.agg in ("min", "max", "mean", "median", "sum"):
                    v = getattr(vals.astype(float), n.agg)()
                elif n.agg in ("absmax", "absmin"):
                    v = getattr(vals.astype(float).abs(), n.agg[3:])()
                elif n.agg == "same":
                    printed = {fmt_tex(x * (n.scale if n.scale is not None else 1.0), n.fmt) for x in vals.astype(float)}
                    if len(printed) > 1:
                        raise ValueError(f"agg 'same': selected rows print differently {sorted(printed)}")
                    v = vals.astype(float).iloc[0]
                else:
                    raise ValueError(f"unknown agg {n.agg!r}")
            elif n.const:
                v = _const(n.const)
            elif n.expr:
                names = re.findall(r"\{(n[A-Za-z]+)\}", n.expr)
                missing = [x for x in names if x not in raw]
                if missing:
                    raise KeyError(f"expr refers to undefined/later macros {missing}")
                env = {x: float(raw[x]) for x in names}
                v = eval(re.sub(r"\{(n[A-Za-z]+)\}", r"\1", n.expr), _SAFE, env)  # noqa: S307 - arithmetic only
            else:
                raise ValueError("no source (csv, const or expr)")
            if n.regex:
                mm = re.search(n.regex, str(v))
                if not mm:
                    raise ValueError(f"regex {n.regex!r} does not match {v!r}")
                v = float(mm.group(1))
            if n.scale is not None:
                v = float(v) * n.scale
            n.raw = v
            raw[n.name] = v
            n.tex = fmt_tex(v, n.fmt)
        except Exception as e:  # noqa: BLE001 - every failure is reported
            n.error = f"{type(e).__name__}: {e}"
            n.tex = r"\textbf{??}"
    return specs


def check_requires(specs: list[Num]) -> list[str]:
    """Evaluate every wording condition (``Num.require``) over the raw values of all macros.

    Returns one error per condition that is false, refers to an unknown or failed macro, or cannot be
    evaluated.  Conditions see every macro (also later ones), so they are checked after
    :func:`evaluate`."""
    raw = {n.name: n.raw for n in specs if not n.error}
    errs = []
    for n in specs:
        if not n.require:
            continue
        names = re.findall(r"\{(n[A-Za-z]+)\}", n.require)
        missing = [x for x in names if x not in raw]
        what = f"{n.name}: wording condition [{n.require_note or 'unnamed'}] {n.require}"
        if missing:
            errs.append(f"{what}: unknown or failed macros {missing}")
            continue
        try:
            env = {x: float(raw[x]) for x in names}
            ok = eval(re.sub(r"\{(n[A-Za-z]+)\}", r"\1", n.require), _SAFE, env)  # noqa: S307 - comparisons only
        except Exception as e:  # noqa: BLE001
            errs.append(f"{what}: {type(e).__name__}: {e}")
            continue
        if not bool(ok):
            vals = ", ".join(f"{x}={env[x]:.6g}" for x in names)
            errs.append(f"{what} is FALSE ({vals}): reword the sentence or fix the data")
    return errs


def validate_names(specs: list[Num]) -> list[str]:
    errs, seen = [], set()
    for n in specs:
        if not re.fullmatch(r"n[A-Z][A-Za-z]*", n.name):
            errs.append(f"{n.name}: macro names are 'n' + capital letter + letters only")
        if n.name in seen:
            errs.append(f"{n.name}: defined twice")
        seen.add(n.name)
        if n.name.startswith(("nLeft", "nRight")):
            errs.append(f"{n.name}: clashes with amssymb arrows")
    return errs


def render(specs: list[Num]) -> str:
    out = ["% paper/r2/numbers.tex -- GENERATED by src/r2/paper_assets/prose_numbers.py (make_paper_assets.py numbers);",
           "% do not edit. One macro per number used in the manuscript prose. Provenance per macro:",
           "%   <csv> | <row filter> | <column> [| aggregation] [| x scale], or a PRESPECIFIED / config / code",
           "%   constant, or '= expression' over earlier macros. Status: post hoc flag of the selected rows.", ""]
    for n in specs:
        st = f" [{n.status}]" if n.status else ""
        note = f" -- {n.note}" if n.note else ""
        req = f" -- requires ({n.require_note}): {n.require}" if n.require else ""
        out.append(f"% {n.source}{st}{note}{req}".replace("\n", " "))
        out.append(rf"\newcommand{{\{n.name}}}{{{n.tex}}}")
    return "\n".join(out) + "\n"


# ----------------------------------------------------------------------------- prose checker
PROSE = [  # (file, start marker or None, end marker or None)
    ("sec_abstract.tex", None, None), ("sec_intro.tex", None, None), ("sec_related.tex", None, None),
    ("sec_methods.tex", None, None), ("sec_data.tex", None, None), ("sec_results.tex", None, None),
    ("sec_discussion.tex", None, None), ("sec_conclusion.tex", None, None),
    ("sec_availability.tex", None, None),
    ("manuscript.tex", r"\begin{document}", r"\end{document}"),
    ("supplement.tex", r"\begin{document}", r"\end{document}"),
]

# Literal numbers that are not results and have no script source. Each entry: file (or '*'), a regex
# that must match the literal in its line context (the literal is the part matched by group 1), reason.
ALLOW = [
    ("*", r"(95)\\%", "interval level (95% intervals throughout)"),
    ("manuscript.tex", r"Sand (14), 72076", "postal address"),
    ("manuscript.tex", r"Sand 14, (72076)", "postal address"),
    ("manuscript.tex", r"orcid\.org/(0000-0001-5342-1499)", "ORCID identifier"),
    ("*", r"V2-(104)M", "model name (Geneformer V2-104M)"),
    ("*", r"(?:19|20)\d\d-(\d\d)-(\d\d)", "month and day of an ISO date"),
    ("sec_methods.tex", r"^\((\d)\) |\s\((\d)\) \\emph", "enumeration label of the estimator ladder"),
    ("sec_methods.tex", r"\((\d)\) \\emph|and\s+\((\d)\)", "enumeration label of the estimator ladder"),
    ("sec_methods.tex", r"it equals (1) in expectation", "definition: AP lift of a random ranking (equation constant)"),
    ("sec_methods.tex", r"Both equal (1) in expectation", "definition: AP lift and Enr@k of a random ranking"),
    ("sec_results.tex", r"only just excludes (0)", "null value of a difference"),
    ("sec_results.tex", r"interval includes (1) in lung", "null value of a ratio"),
    ("sec_results.tex", r"credibility above (0\.5),", "probability threshold 0.5 (definition; CSV column n_edges_nuts_mean_gt_0.5)"),
    ("sec_results.tex", r"interval that includes (0\.5)\.", "probability threshold 0.5 (definition; CSV column "
                                                              "n_edges_cri_straddles_0.5_mean_gt_0.5)"),
    ("*", r"THP-(1)", "cell-line name"),
    ("sec_results.tex", r"composition-adjusted AP lift includes (1),", "definition: AP lift of a random ranking within TFs"),
    ("sec_results.tex", r"every interval below (0\.5)\)", "definition: AUROC of a random ranking"),
    ("sec_data.tex", r"the (5\{,\}000) most variable genes", "upstream extraction setting of the attention matrices "
                                                           "(single_cell_mechinterp/configs/eval_bias_scgpt_*.yaml: hvg 5000)"),
]
# Years outside a date ("2026-01-08", "8 January 2026") or author-year context ("Dixit 2016",
# "et al. 2019", "(2019)") need an explicit entry here: (file or '*', regex with the year as group 1,
# reason).  A bare four-digit number such as "2000 cells" is a literal result number, not a year.
YEAR_ALLOW: list[tuple[str, str, str]] = []
# Identifiers with digits glued to letters in running text (digits preceded by a letter).  Every other
# such token ("Q20", "x0.37", "AP1.05") is a violation.  Full-token regexes.
IDENT_ALLOW = [
    (r"ENETS2", "reference name (ENCODE regulatory network, filtered proximal set 2)"),
    (r"GRNBoost2|GENIE3", "method names"),
    (r"K562|THP-1", "cell-line names"),
    (r"SPI1|EGR1|NFKB1", "gene symbols"),
    (r"M2", "processor name (Apple M2 Pro)"),
    (r"L[123]O", "leave-k-studies-out labels"),
    (r"v2", "resource version (TRRUST v2)"),
    (r"V2-104M", "model name (Geneformer V2-104M)"),
    (r"Smart-seq2", "assay name"),
]
# Words that refer to a review or revision history; the paper reads as a first submission.
ARTEFACT_RE = re.compile(
    r"\breview(?:er|ers|ed|ing)?\b|\brevis(?:ion|ions|ed|e)\b|previous version|earlier version|"
    r"\bR[12]\b|\bwe now\b|\bcorrected\b|\bin response\b|\bgenuine|\bhonest|\bupdated?\b|\bsupersed|"
    r"\(Q\d+\)", re.I)
# Equation constants: inline-math literals of the model specification (priors, likelihood and test
# definitions) that are part of the equations, not results.  Integers 0-3 in math are allowed unless
# they are the whole math span (``$3$ optima`` is a count, not an equation constant).
MATH_ALLOW = [
    ("sec_methods.tex", r"\\operatorname\{logit\}\((0\.01)\)", "prior mean of b_j (model specification, eq:hier)"),
    ("sec_methods.tex", r"\\mathrm\{HalfNormal\}\((0\.5)\)", "prior of the replication model (specification)"),
    ("sec_methods.tex", r"\\mathcal\{N\}\(-(4),3\^2\)", "prior mean of the stratum intercepts of the replication "
                                                     "model (replication.py RE_M_PRIOR; specification)"),
]
DISPLAY_ENVS = ("equation", "equation*", "align", "align*", "gather", "gather*", "multline", "multline*")
ARG_CMDS = ("label", "ref", "cref", "Cref", "eqref", "cite", "citet", "citep", "citealp", "citeauthor", "citeyear",
            "input", "include", "url", "includegraphics", "externaldocument", "graphicspath", "begin", "end",
            "InputIfFileExists", "hypersetup", "usepackage", "documentclass", "captionsetup",
            "linenumbers", "bibliographystyle", "bibliography")
NUM_RE = re.compile(r"(?<![A-Za-z_^\\\d.])(\d+(?:\{,\}\d{3})*(?:\.\d+)?)")
GLUED_RE = re.compile(r"(?<![A-Za-z0-9_\\])([A-Za-z][A-Za-z\-]*\d[A-Za-z0-9.]*(?:-\d[A-Za-z0-9.]*)?)")
LENGTH_RE = re.compile(r"\s*(\\textwidth|\\linewidth|\\columnwidth|\\textheight|pt\b|cm\b|mm\b|em\b|ex\b)")
MONTHS = r"(?:January|February|March|April|May|June|July|August|September|October|November|December)"


def _year_ok(fname: str, line: str, pos: int, lit: str) -> bool:
    """A four-digit 19xx/20xx literal in running text is a year only in a date or author-year context."""
    end = pos + len(lit)
    before, after = line[:pos], line[end:]
    if re.match(r"-\d\d(?:-\d\d)?\b", after) or re.search(r"\d\d-$", before):
        return True                                            # ISO date 2026-01-08
    if re.search(MONTHS + r"(?:~|\s)+(?:\d{1,2},?(?:~|\s)+)?$", before):
        return True                                            # 8 January 2026 / January 2026
    author_year = re.search(r"(?:\bet(?:~|\s)+al\.|\b[A-Z][A-Za-z\-]+)(?:~|\s)+\(?$", before)
    bracketed = re.search(r"\($", before) and re.match(r"\)", after)
    if author_year or bracketed:
        return True                                            # Dixit 2016, et al. 2019, (2019)
    return _allowed(fname, line, pos, lit, YEAR_ALLOW) is not None


def _mask(s: str, a: int, b: int) -> str:
    return s[:a] + re.sub(r"[^\n]", " ", s[a:b]) + s[b:]


def _strip_comments(text: str) -> str:
    return "\n".join(re.sub(r"(?<!\\)%.*$", "", ln) for ln in text.split("\n"))


def _mask_args(text: str) -> str:
    """Blank the arguments of reference/citation/structure commands (balanced braces)."""
    pat = re.compile(r"\\(" + "|".join(re.escape(c) for c in ARG_CMDS) + r")\*?(\[[^\]]*\])?\{")
    out, pos = text, 0
    while True:
        m = pat.search(out, pos)
        if not m:
            break
        depth, i = 1, m.end()
        while i < len(out) and depth:
            depth += {"{": 1, "}": -1}.get(out[i], 0)
            i += 1
        out = _mask(out, m.start(), i)
        pos = i
    return out


def _mask_display(text: str) -> str:
    for env in DISPLAY_ENVS:
        pat = re.compile(r"\\begin\{" + re.escape(env) + r"\}.*?\\end\{" + re.escape(env) + r"\}", re.S)
        text = pat.sub(lambda m: re.sub(r"[^\n]", " ", m.group(0)), text)
    return re.sub(r"\\\[.*?\\\]", lambda m: re.sub(r"[^\n]", " ", m.group(0)), text, flags=re.S)


def _math_spans(line: str) -> list[tuple[int, int]]:
    spans, i, start = [], 0, None
    while i < len(line):
        if line[i] == "\\":
            i += 2
            continue
        if line[i] == "$":
            if start is None:
                start = i
            else:
                spans.append((start, i + 1))
                start = None
        i += 1
    return spans


def _allowed(fname: str, line: str, pos: int, lit: str, rules) -> str | None:
    for f, rx, reason in rules:
        if f not in ("*", fname):
            continue
        for m in re.finditer(rx, line):
            for g in range(1, (m.lastindex or 0) + 1):
                if m.group(g) == lit and m.start(g) == pos:
                    return reason
    return None


def scan_text(fname: str, text: str, line_offset: int = 0) -> tuple[list[dict], list[dict]]:
    """Return (violations, macro uses) of one prose text."""
    clean = _mask_args(_mask_display(_strip_comments(text)))
    viol, uses = [], []
    for ln_no, (raw_ln, ln) in enumerate(zip(text.split("\n"), clean.split("\n")), start=1 + line_offset):
        for m in MACRO_RE.finditer(ln):
            uses.append(dict(file=fname, line=ln_no, macro=m.group(1)))
        work = MACRO_RE.sub(lambda m: " " * len(m.group(0)), ln)
        spans = _math_spans(work)
        work = re.sub(r"\\[A-Za-z]+", lambda m: " " * len(m.group(0)), work)   # command names
        for m in NUM_RE.finditer(work):
            lit, pos = m.group(1), m.start(1)
            span = next(((a, b) for a, b in spans if a < pos < b), None)
            in_math = span is not None
            if re.fullmatch(r"(19|20)\d\d", lit) and not in_math and _year_ok(fname, ln, pos, lit):
                continue                                  # a year in a date / author-year context
            if LENGTH_RE.match(ln, m.end(1)):
                continue                                  # a length (layout)
            if in_math and re.fullmatch(r"[0-3]", lit):
                body = re.sub(r"[\s{}]|\\[,;!]", "", work[span[0] + 1:span[1] - 1])
                if body != lit:
                    continue                              # equation constant (0-3) inside an expression
            if _allowed(fname, ln, pos, lit, ALLOW):
                continue
            if in_math and _allowed(fname, ln, pos, lit, MATH_ALLOW):
                continue
            viol.append(dict(file=fname, line=ln_no, literal=lit, math=in_math, context=raw_ln.strip()[:160]))
        for m in GLUED_RE.finditer(work):                 # digits glued to letters in running text
            tok, pos = m.group(1).rstrip(".-"), m.start(1)
            if any(a < pos < b for a, b in spans):
                continue                                  # subscripts and names inside math
            if any(re.fullmatch(rx, tok) for rx, _ in IDENT_ALLOW):
                continue
            if _allowed(fname, ln, pos, tok, ALLOW):
                continue
            viol.append(dict(file=fname, line=ln_no, literal=tok, math=False, context=raw_ln.strip()[:160]))
    return viol, uses


def scan_artefacts(fname: str, text: str, line_offset: int = 0) -> list[dict]:
    """Words that refer to a review or revision history (:data:`ARTEFACT_RE`), comments included."""
    out = []
    for ln_no, ln in enumerate(text.split("\n"), start=1 + line_offset):
        masked = _mask_args(ln)                           # labels, citation keys and paths of \input etc.
        for m in ARTEFACT_RE.finditer(masked):
            out.append(dict(file=fname, line=ln_no, literal=m.group(0), math=False, context=ln.strip()[:160]))
    return out


def _prose_text(fname: str, start: str | None, end: str | None) -> tuple[str, int]:
    text = (S.PAPER / fname).read_text()
    off = 0
    if start:
        k = text.index(start)
        off = text[:k].count("\n")
        text = text[k:]
    if end and end in text:
        text = text[:text.index(end)]
    return text, off


def check(specs: list[Num] | None = None) -> dict:
    viol, uses, art = [], [], []
    for fname, a, b in PROSE:
        if not (S.PAPER / fname).exists():
            continue
        text, off = _prose_text(fname, a, b)
        v, u = scan_text(fname, text, off)
        viol += v
        uses += u
        art += scan_artefacts(fname, text, off)
    defined = {n.name for n in specs} if specs is not None else set(_defined_in_numbers_tex())
    helpers = {n.name for n in specs if n.helper} if specs is not None else set()
    used = {u["macro"] for u in uses}
    undefined = sorted(used - defined)
    unused = sorted(defined - used - helpers)
    return dict(violations=viol, uses=uses, undefined=undefined, unused=unused, artefacts=art)


def _defined_in_numbers_tex() -> list[str]:
    if not NUMBERS_TEX.exists():
        return []
    return re.findall(r"\\newcommand\{\\(n[A-Za-z]+)\}", NUMBERS_TEX.read_text())


# ----------------------------------------------------------------------------- audit
def audit_frame(specs: list[Num], uses: list[dict]) -> pd.DataFrame:
    where: dict[str, list[str]] = {}
    for u in uses:
        where.setdefault(u["macro"], []).append(f"{u['file']}:{u['line']}")
    rows = []
    for n in specs:
        src_kind = "csv" if n.csv else ("constant" if n.const else "expression")
        rows.append(dict(
            macro="\\" + n.name, value=plain(n.tex), value_tex=n.tex,
            raw=(f"{float(n.raw):.10g}" if isinstance(n.raw, (int, float, np.integer, np.floating))
                 and not isinstance(n.raw, bool) else str(n.raw)),
            fmt=n.fmt, source_kind=src_kind, csv=n.csv or "", filter=n.query or ("" if not n.csv else "all rows"),
            column=n.col or "", agg=n.agg or "", constant=n.const or "", expression=n.expr or "",
            scale="" if n.scale is None else f"{n.scale:g}", analysis_status=n.status, note=n.note,
            helper=n.helper, prose_before_binding=n.was or "",
            changed_by_binding=bool(n.was is not None and n.was != plain(n.tex)),
            used_in=";".join(where.get(n.name, []))))
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- build
def build(write: bool = True, check_only: bool = False, verbose: bool = True) -> int:
    if __package__ in (None, ""):
        from r2.paper_assets import prose_numbers_spec as SPEC  # type: ignore
    else:
        from . import prose_numbers_spec as SPEC
    specs = evaluate(SPEC.specs())
    errs = validate_names(specs) + [f"{n.name}: {n.error}" for n in specs if n.error] + check_requires(specs)
    tex = render(specs)
    if write and not check_only and not errs:
        NUMBERS_TEX.write_text(tex)
    res = check(specs)
    if write and not check_only and not errs:
        AUDIT_CSV.parent.mkdir(parents=True, exist_ok=True)
        audit_frame(specs, res["uses"]).to_csv(AUDIT_CSV, index=False)
    rc = 0
    if errs:
        rc = 1
        print(f"[numbers] {len(errs)} specification errors:")
        for e in errs:
            print("   ", e)
    if check_only and NUMBERS_TEX.exists() and NUMBERS_TEX.read_text() != tex and not errs:
        rc = 1
        print("[numbers] paper/r2/numbers.tex is not current (run make_paper_assets.py numbers)")
    for key, msg in (("violations", "literal numbers in the prose"), ("undefined", "macros used but not defined"),
                     ("unused", "macros defined but not used"),
                     ("artefacts", "words referring to a review or revision history")):
        if res[key]:
            rc = 1
            print(f"[numbers] {len(res[key])} {msg}:")
            for v in res[key][:400]:
                print("   ", v if isinstance(v, str) else f"{v['file']}:{v['line']}: {v['literal']}"
                      f"{' (math)' if v['math'] else ''} | {v['context']}")
    if verbose:
        changed = [n for n in specs if n.was is not None and n.was != plain(n.tex)]
        print(f"[numbers] {len(specs)} macros ({sum(n.helper for n in specs)} helpers), "
              f"{len(res['uses'])} uses; {len(changed)} differ from the pre-binding prose")
        for n in changed:
            print(f"    changed {n.name}: {n.was} -> {plain(n.tex)}")
    return rc


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--check", action="store_true", help="only check (numbers.tex current, prose clean)")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args(argv)
    return build(write=True, check_only=a.check, verbose=not a.quiet)


if __name__ == "__main__":
    sys.exit(main())
