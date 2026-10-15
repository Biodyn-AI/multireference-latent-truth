"""LaTeX table infrastructure for the manuscript tables (``tables`` and ``tables_supp``).

Every printed number goes through :class:`Table`, which

* reads the source CSVs (via :class:`style.Provenance`, so each table carries a
  provenance comment listing its CSV files and row filters);
* formats each value with ONE of a fixed set of formats (:data:`KINDS`) so that
  ratios, p-values, counts, probabilities ... are printed the same way in every
  table, and
* registers every printed cell (CSV file, CSV row number, column, raw value,
  format, printed text) in ``Table.cells``.  The tests re-read the CSVs and check
  that every registered text is the formatted CSV value and occurs in the .tex
  (numbers round-trip to the CSVs).

The generated LaTeX uses only booktabs, multirow, caption and amsmath (the R1
preamble): no vertical rules, ``\\small`` body, caption and label inside the
``table`` float, table notes in a ``\\footnotesize`` paragraph under the tabular,
post hoc rows / columns marked with a dagger and explained in a note.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np
import pandas as pd

from . import style as S

DAG = r"$^{\dagger}$"
DASH = "--"

# CSV name prefixes whose tables were still being regenerated when the assets were built
# (set by the CLI; tables reading them carry a WARNING comment and are listed in the report)
STALE_PREFIXES: list[str] = []

POSTHOC_NOTE = r"$^{\dagger}$\,Post hoc: defined after the held-out data had been examined."


# ----------------------------------------------------------------------------- escaping
_TEX_REPL = [
    ("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"), ("$", r"\$"), ("#", r"\#"),
    ("_", r"\_"), ("{", r"\{"), ("}", r"\}"), ("~", r"\textasciitilde{}"), ("^", r"\^{}"),
]
_UNICODE = {"≥": r"$\geq$", "≤": r"$\leq$", "×": r"$\times$", "−": "--", "–": "--", "—": "---",
            "→": r"$\to$", "≈": r"$\approx$", "±": r"$\pm$", "α": r"$\alpha$", "β": r"$\beta$",
            "γ": r"$\gamma$", "η": r"$\eta$", "μ": r"$\mu$", "σ": r"$\sigma$", "τ": r"$\tau$",
            "θ": r"$\theta$", "ω": r"$\omega$", "κ": r"$\kappa$", "λ": r"$\lambda$",
            "Δ": r"$\Delta$", "'": "'", "’": "'", "‘": "`", "“": "``", "”": "''", "·": r"$\cdot$",
            "é": r"\'e", "ü": r'\"u', "ö": r'\"o', "ä": r'\"a', "ó": r"\'o", "á": r"\'a"}


def breakable(s) -> str:
    """``tex`` plus line-break opportunities after '.', '_' and '/' inside long identifiers."""
    out = tex(s)
    out = out.replace(r"\_", r"\_\allowbreak{}")
    out = re.sub(r"\.(?=[A-Za-z])", r".\\allowbreak{}", out)
    return out.replace("/", "/\\allowbreak{}")


def tex(s) -> str:
    """Escape plain text for LaTeX (special characters and the unicode used in labels)."""
    if s is None or (isinstance(s, float) and math.isnan(s)):
        return DASH
    s = str(s)
    for a, b in _TEX_REPL:
        s = s.replace(a, b)
    for a, b in _UNICODE.items():
        s = s.replace(a, b)
    return s


# ----------------------------------------------------------------------------- formats
def _isnum(x) -> bool:
    if x is None:
        return False
    if isinstance(x, (bool, np.bool_)):
        return False
    try:
        return not pd.isna(x)
    except (TypeError, ValueError):
        return False


def _fixed(x, d):
    if not _isnum(x):
        return DASH
    x = float(x)
    if math.isinf(x):
        return r"$\infty$" if x > 0 else r"$-\infty$"
    s = f"{x:.{d}f}"
    if s.startswith("-") and float(s) == 0.0:
        s = s[1:]
    return s.replace("-", "$-$", 1) if s.startswith("-") else s


def _sci(x, sig=2):
    """$m\\times10^{e}$ with ``sig`` significant digits (used for p-values and tiny probabilities)."""
    if not _isnum(x):
        return DASH
    x = float(x)
    if x == 0:
        return "0"
    if math.isinf(x):
        return r"$\infty$"
    mant, exp = f"{x:.{sig - 1}e}".split("e")
    return f"${mant}\\times 10^{{{int(exp)}}}$"


def _count(x):
    if not _isnum(x):
        return DASH
    x = float(x)
    if math.isinf(x):
        return r"$\infty$"
    return f"{int(round(x)):,}".replace(",", "{,}")


def _p(x):
    """p-value: p >= 0.995 printed as 1, p >= 0.1 with 2 decimals, smaller values in scientific
    notation with 2 significant digits."""
    if not _isnum(x):
        return DASH
    x = float(x)
    if x >= 0.995:
        return "1"
    if x >= 0.1:
        return _fixed(x, 2)
    return _sci(x, 2)


def _prob(x):
    """Probability / weight: 2 decimals from 0.01 up, 2 significant digits below, and
    scientific notation below 1e-3."""
    if not _isnum(x):
        return DASH
    x = float(x)
    if x == 0:
        return "0"
    if abs(x) >= 0.01:
        return _fixed(x, 2)
    if abs(x) < 1e-3:
        return _sci(x, 2)
    return _sig(x, 2)


def _sig(x, sig=2):
    if not _isnum(x):
        return DASH
    x = float(x)
    if x == 0:
        return "0"
    if math.isinf(x):
        return r"$\infty$"
    if abs(x) < 1e-3:
        return _sci(x, sig)
    mag = int(math.floor(math.log10(abs(x))))
    d = max(sig - 1 - mag, 0)
    out = f"{x:.{d}f}"
    # rounding can carry into the next decade (0.00996 -> 0.0100); drop one decimal then
    if d > 0 and abs(float(out)) >= 10.0 ** (mag + 1):
        out = f"{x:.{d - 1}f}"
    return out.replace("-", "$-$", 1) if out.startswith("-") else out


def _pct(x, d=1):
    if not _isnum(x):
        return DASH
    return _fixed(100.0 * float(x), d)


KINDS: dict[str, Callable] = {
    "ratio": lambda x: _fixed(x, 2),      # ratios, enrichments, O/E, AP lift, Jaccard, kappa
    "ratio3": lambda x: _fixed(x, 3),     # small differences, AUROC, rank disagreement, BFDR
    "num1": lambda x: _fixed(x, 1),       # log-posterior gaps, nats, seconds, MiB
    "num0": lambda x: _fixed(x, 0),
    "num4": lambda x: _fixed(x, 4),
    "count": _count,                      # integers with {,} thousands separators
    "p": _p,                              # p-values
    "prob": _prob,                        # probabilities, class weights
    "sig2": lambda x: _sig(x, 2),
    "sig3": lambda x: _sig(x, 3),
    "sci2": lambda x: _sci(x, 2),
    "pct": lambda x: _pct(x, 1),          # fraction printed as a percentage, 1 decimal
    "pct0": lambda x: _pct(x, 0),
    "pct2": lambda x: _pct(x, 2),
    "rhat": lambda x: _fixed(x, 3),
}


def fmt(x, kind: str) -> str:
    return KINDS[kind](x)


def parse_number(text: str) -> float:
    """Parse a printed number back to float (inverse of the formats, for the tests)."""
    t = text.strip()
    if t in (DASH, ""):
        return float("nan")
    t = t.replace(r"$\leq$", "").replace(r"$\le$", "").replace(r"$\geq$", "").replace("{,}", "")
    t = t.replace(r"$\infty$", "inf").replace(r"$-\infty$", "-inf").replace("$-$", "-")
    m = re.fullmatch(r"\$?(-?[0-9.]+)\\times 10\^\{(-?\d+)\}\$?", t)
    if m:
        return float(m.group(1)) * 10.0 ** int(m.group(2))
    t = t.strip("$")
    return float(t)


# ----------------------------------------------------------------------------- cells
@dataclass
class Cell:
    table: str
    csv: str | None
    row: int | None          # 0-based data-row number in the CSV (pandas index after read_csv)
    column: str | None
    value: float | None
    kind: str
    text: str
    derived: bool = False
    sources: list = field(default_factory=list)   # [(csv, row, column, value)] for derived cells


class Table:
    """Builder for one LaTeX table (one ``table`` float, possibly several panels)."""

    def __init__(self, name: str, label: str | None = None, caption: str = "",
                 short_caption: str | None = None):
        self.name = name
        self.label = label or f"tab:{name}"
        self.caption = caption
        self.short_caption = short_caption
        self.prov = S.Provenance(f"paper/r2/tables/{name}.tex")
        self.cells: list[Cell] = []
        self.notes: list[str] = []
        self.uses_posthoc = False
        self.stale_sources: set[str] = set()
        self._dfs: dict[str, pd.DataFrame] = {}

    # ------------------------------------------------------------------ reading
    def read(self, csv: str, filt: str | None = None) -> pd.DataFrame:
        csv = csv if csv.endswith(".csv") else csv + ".csv"
        if csv not in self._dfs:
            df = pd.read_csv(S.TABLES / csv, low_memory=False)
            df["_csv"] = csv
            df["_row"] = np.arange(len(df))
            self._dfs[csv] = df
        self.prov.note(csv, filt)
        if any(csv.startswith(p) for p in STALE_PREFIXES):
            self.stale_sources.add(csv)
        return self._dfs[csv]

    def note_source(self, csv: str, filt: str | None = None) -> None:
        self.prov.note(csv if csv.endswith(".csv") else csv + ".csv", filt)

    # ------------------------------------------------------------------ cells
    def v(self, r: pd.Series | None, col: str, kind: str = "ratio", prefix: str = "") -> str:
        """Format column ``col`` of CSV row ``r`` (a row of a frame returned by :meth:`read`)."""
        if r is None or col not in r.index:
            return DASH
        val = r[col]
        text = fmt(val, kind)
        if text != DASH:
            self.cells.append(Cell(self.name, r["_csv"], int(r["_row"]), col,
                                   _to_float(val), kind, text))
        return prefix + text if text != DASH else text

    def d(self, value, kind: str, sources: Sequence[tuple[pd.Series, str]] = ()) -> str:
        """Format a value derived by this script from CSV cells (``sources``)."""
        text = fmt(value, kind)
        if text != DASH:
            src = [(r["_csv"], int(r["_row"]), c, _to_float(r[c])) for r, c in sources]
            self.cells.append(Cell(self.name, None, None, None, _to_float(value), kind, text,
                                   derived=True, sources=src))
        return text

    def ci(self, r, lo: str, hi: str, kind: str = "ratio") -> str:
        a, b = self.v(r, lo, kind), self.v(r, hi, kind)
        if a == DASH and b == DASH:
            return ""
        # an interval bound below 0 that rounds to zero keeps its sign ([-0.017, -0.000] excludes 0)
        a, b = (_signed_zero(r, c, x) for c, x in ((lo, a), (hi, b)))
        return f"[{a}, {b}]"

    def est_ci(self, r, col: str, lo: str, hi: str, kind: str = "ratio", stack: bool = False) -> str:
        e = self.v(r, col, kind)
        c = self.ci(r, lo, hi, kind)
        if e == DASH or not c:
            return e
        return stacked(e, c) if stack else f"{e} {c}"

    def p(self, r, col: str, floor_col: str | None = None) -> str:
        """p-value; printed as $\\leq$ floor when it equals the resampling floor of ``floor_col``."""
        text = self.v(r, col, "p")
        if text == DASH or floor_col is None or floor_col not in r.index:
            return text
        pv, fl = r[col], r[floor_col]
        if _isnum(pv) and _isnum(fl) and float(pv) <= float(fl) * (1 + 1e-9):
            return r"$\leq$" + text
        return text

    def ph(self) -> str:
        """Dagger for a post hoc row / column (adds the post hoc note)."""
        self.uses_posthoc = True
        return DAG

    def ph_if(self, flag: bool) -> str:
        return self.ph() if flag else ""

    def note(self, text: str) -> None:
        if text not in self.notes:
            self.notes.append(text)

    # ------------------------------------------------------------------ rendering
    def render(self, colspec: str, header: Sequence[str], body: Sequence, *,
               tabcolsep: float | None = None, size: str = "small", placement: str = "tbp",
               arraystretch: float | None = None, continued: str | None = None) -> str:
        """Return the complete LaTeX source of the table.

        ``header``: list of LaTeX header lines (each ending without ``\\\\``; cmidrules
        may be given as separate items starting with ``\\cmidrule``).
        ``body``: rows (list of cell strings) or markers ``("panel", text)``,
        ``("midrule",)``, ``("addlinespace",)``, ``("raw", latex)``.
        """
        out = self._float_head(size=size, placement=placement, tabcolsep=tabcolsep,
                               arraystretch=arraystretch, continued=continued)
        out += self._tabular(colspec, header, body)
        out += self._float_tail()
        return "\n".join(out) + "\n"

    def render_blocks(self, blocks: Sequence[dict], *, size: str = "small", placement: str = "tbp",
                      arraystretch: float | None = None) -> str:
        """One ``table`` float holding several tabulars (``blocks``), each with an optional
        title line.  A block is a dict with keys ``colspec``, ``header``, ``body`` and optionally
        ``title`` (LaTeX), ``tabcolsep`` (pt) and ``size`` (font size command of that block).
        Used when the parts of one table need different columns."""
        out = self._float_head(size=size, placement=placement, tabcolsep=None,
                               arraystretch=arraystretch, continued=None)
        for i, b in enumerate(blocks):
            if i:
                out.append("\\par\\medskip")
            out.append("{")
            if b.get("size"):
                out.append(f"\\{b['size']}")
            if b.get("tabcolsep") is not None:
                out.append(f"\\setlength{{\\tabcolsep}}{{{b['tabcolsep']}pt}}")
            if b.get("title"):
                out.append(f"\\par\\noindent\\emph{{{b['title']}}}\\par\\smallskip")
            out += self._tabular(b["colspec"], b["header"], b["body"])
            out.append("}")
        out += self._float_tail()
        return "\n".join(out) + "\n"

    def _float_head(self, *, size, placement, tabcolsep, arraystretch, continued) -> list[str]:
        out = [self.prov.tex_comment().rstrip("\n")]
        if self.stale_sources:
            out.append("% WARNING: built from source tables that were being regenerated: "
                       + ", ".join(sorted(self.stale_sources)))
        out.append(f"\\begin{{table}}[{placement}]")
        if continued:       # a continuation keeps the number of the first part (caption package)
            out.append("\\ContinuedFloat")
        out.append("\\centering")
        out.append(f"\\{size}")
        if tabcolsep is not None:
            out.append(f"\\setlength{{\\tabcolsep}}{{{tabcolsep}pt}}")
        if arraystretch is not None:
            out.append(f"\\renewcommand{{\\arraystretch}}{{{arraystretch}}}")
        cap = self.caption + (f" {continued}" if continued else "")
        sc = f"[{self.short_caption}]" if self.short_caption else ""
        out.append(f"\\caption{sc}{{{cap}}}")
        out.append(f"\\label{{{self.label}}}")
        return out

    def _float_tail(self) -> list[str]:
        out = []
        notes = list(self.notes)
        if self.uses_posthoc:
            notes.append(POSTHOC_NOTE)
        if notes:
            out.append("\\par\\smallskip")
            out.append("\\begin{minipage}{\\linewidth}\\footnotesize")
            out.append("\n\n".join(notes))
            out.append("\\end{minipage}")
        out.append("\\end{table}")
        return out

    def _tabular(self, colspec: str, header: Sequence[str], body: Sequence) -> list[str]:
        ncol = _ncols(colspec)
        out = [f"\\begin{{tabular}}{{{colspec}}}", "\\toprule"]
        for h in header:
            out.append(h if h.lstrip().startswith("\\cmidrule") else h + " \\\\")
        out.append("\\midrule")
        first = True
        for item in body:
            if isinstance(item, tuple) and item[0] == "row":
                item = item[1]
            if isinstance(item, tuple):
                tag = item[0]
                if tag == "panel":
                    if not first:
                        out.append("\\addlinespace[3pt]")
                    out.append(f"\\multicolumn{{{ncol}}}{{@{{}}l}}{{\\emph{{{item[1]}}}}} \\\\")
                    if len(item) > 2 and item[2]:
                        out.append(item[2])
                elif tag == "midrule":
                    out.append("\\midrule")
                elif tag == "addlinespace":
                    out.append("\\addlinespace[2pt]")
                elif tag == "raw":
                    out.append(item[1])
                elif tag == "cmidrule":
                    out.append(item[1])
                first = False
                continue
            cells = list(item)
            span = sum(int(m.group(1)) if (m := re.match(r"\\multicolumn\{(\d+)\}", c)) else 1
                       for c in cells)
            if span != ncol:
                raise ValueError(f"{self.name}: row has {len(cells)} cells, colspec has {ncol}: {cells}")
            out.append(" & ".join(cells) + " \\\\")
            first = False
        out.append("\\bottomrule")
        out.append("\\end{tabular}")
        return out


def _signed_zero(r, col: str, text: str) -> str:
    """A negative value printed as zero keeps its sign and, where one more decimal shows it, that decimal
    ('$-$0.0001' instead of '0.000'; otherwise '$-$0.000'), so that an interval that excludes 0 does not seem to
    touch it (the registered cell keeps the plain text)."""
    if r is None or col not in r.index or not _isnum(r[col]):
        return text
    x = float(r[col])
    m = re.fullmatch(r"0(?:\.(0+))?", text)
    if x < 0 and m:
        d = len(m.group(1) or "") + 1
        finer = f"{abs(x):.{d}f}"
        return "$-$" + (finer if float(finer) != 0.0 else text)
    return text


def _body_lines(item) -> float:
    if isinstance(item, tuple):
        if item[0] == "row":
            return float(item[2])
        return 1.2 if item[0] == "panel" else 0.3
    return 1.9 if any("\\\\[-1pt]" in c for c in item) else 1.0


def split_body(body: Sequence, max_lines: float, at_panels_only: bool = False) -> list[list]:
    """Split a body into chunks of at most ``max_lines`` printed lines, preferably at panels.  With
    ``at_panels_only`` a chunk ends only before a panel title (a panel is never split, even if it is longer
    than ``max_lines``; whole panels are packed greedily)."""
    if at_panels_only:
        panels, cur = [], []
        for item in body:
            if isinstance(item, tuple) and item[0] == "panel" and cur:
                panels.append(cur)
                cur = []
            cur.append(item)
        if cur:
            panels.append(cur)
        chunks, cur, n = [], [], 0.0
        for pnl in panels:
            w = sum(_body_lines(it) for it in pnl)
            if cur and n + w > max_lines:
                chunks.append(cur)
                cur, n = [], 0.0
            cur.extend(it for it in pnl if cur or not (isinstance(it, tuple) and it[0] in ("addlinespace", "midrule")))
            n += w
        if cur:
            chunks.append(cur)
        return chunks
    chunks, cur, n = [], [], 0.0
    for item in body:
        w = _body_lines(item)
        is_panel = isinstance(item, tuple) and item[0] == "panel"
        is_row = not isinstance(item, tuple) or item[0] == "row"
        if cur and (n + w > max_lines or (is_panel and n > 0.75 * max_lines)):
            chunks.append(cur)
            cur, n = [], 0.0
        if not cur and not is_row and item[0] in ("addlinespace", "midrule"):
            continue
        cur.append(item)
        n += w
    if cur:
        chunks.append(cur)
    return chunks


def render_split(table: "Table", colspec: str, header: Sequence[str], body: Sequence, *,
                 max_lines: float = 46, at_panels_only: bool = False, **kw) -> str:
    """Render ``body`` as one or more ``table`` floats (parts after the first are captioned
    "(continued)" and labelled ``<label>-b``, ``-c``, ...); notes go under the last part only."""
    parts = split_body(body, max_lines, at_panels_only=at_panels_only)
    if len(parts) <= 1:
        return table.render(colspec, header, body, **kw)
    out = []
    notes, ph = table.notes, table.uses_posthoc
    label = table.label
    for i, part in enumerate(parts):
        last = i == len(parts) - 1
        table.notes = notes if last else []
        table.uses_posthoc = ph if last else False
        table.label = label if i == 0 else f"{label}-{chr(ord('a') + i)}"
        src = table.render(colspec, header, part, continued="(continued)" if i else None, **kw)
        if i:   # one provenance block per file is enough
            src = "\n".join(ln for ln in src.splitlines() if not ln.startswith("%")) + "\n"
        out.append(src)
    table.notes, table.uses_posthoc, table.label = notes, ph, label
    return "\n".join(out)


def stacked(top: str, bottom: str, align: str = "c") -> str:
    """Two-line cell: estimate above, interval below in \\scriptsize (base LaTeX only)."""
    return (f"\\begin{{tabular}}[t]{{@{{}}{align}@{{}}}}{top}\\\\[-1pt]"
            f"{{\\scriptsize {bottom}}}\\end{{tabular}}")


def _ncols(colspec: str) -> int:
    s = re.sub(r"@\{[^}]*\}", "", colspec)
    s = re.sub(r"[pmb]\{[^}]*\}", "X", s)
    s = re.sub(r">\{[^}]*\}|<\{[^}]*\}", "", s)
    return sum(1 for ch in s if ch in "lcrX")


def _to_float(x):
    try:
        if isinstance(x, (bool, np.bool_)):
            return float(x)
        return float(x)
    except (TypeError, ValueError):
        return None


def first_row(df: pd.DataFrame, **eq) -> pd.Series | None:
    """The single row of ``df`` matching all ``column == value`` conditions (None if absent)."""
    m = np.ones(len(df), dtype=bool)
    for k, val in eq.items():
        if k not in df.columns:
            return None
        if val is None:
            m &= df[k].isna().to_numpy()
        else:
            m &= (df[k] == val).to_numpy()
    sub = df[m]
    if len(sub) == 0:
        return None
    return sub.iloc[0]


def rows(df: pd.DataFrame, **eq) -> pd.DataFrame:
    m = np.ones(len(df), dtype=bool)
    for k, val in eq.items():
        if isinstance(val, (list, tuple, set)):
            m &= df[k].isin(list(val)).to_numpy()
        else:
            m &= (df[k] == val).to_numpy()
    return df[m]


def is_posthoc_row(r: pd.Series | None) -> bool:
    return r is not None and "analysis_status" in r.index and str(r["analysis_status"]) == "post_hoc"


def posthoc_patterns(r: pd.Series | None) -> list[str]:
    """Column patterns listed in a row's ``post_hoc_columns`` (``x_lo/hi`` expands to x_lo, x_hi)."""
    if r is None or "post_hoc_columns" not in r.index or not isinstance(r["post_hoc_columns"], str):
        return []
    out = []
    for tok in r["post_hoc_columns"].split(";"):
        tok = tok.strip().split(" (")[0].strip()
        if not tok:
            continue
        if tok.endswith("_lo/hi"):
            base = tok[: -len("_lo/hi")]
            out += [f"{base}_lo", f"{base}_hi"]
        else:
            out.append(tok)
    return out


def is_posthoc_col(r: pd.Series | None, col: str) -> bool:
    """True if ``col`` matches a pattern of the row's ``post_hoc_columns`` (shell wildcards)."""
    import fnmatch
    return any(fnmatch.fnmatchcase(col, pat) for pat in posthoc_patterns(r))


def decision_status(table: Table, module: str, item_id: str) -> str | None:
    """Status ('post_hoc', 'module_design', ...) of an entry of the dated decision log."""
    df = table.read("eval_analysis_decisions.csv", f"module == {module}, id == {item_id}")
    r = first_row(df, module=module, id=item_id)
    return None if r is None else str(r["status"])


def write_table(tex_src: str, out_dir: Path, name: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / f"{name}.tex"
    p.write_text(tex_src)
    return p
