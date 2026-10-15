"""Marked copy of the manuscript: latexdiff of the previously submitted text (paper/main_clean.tex) against
the flattened current manuscript (paper/r2/manuscript.tex with every \\input inlined and every numbers.tex
macro expanded to its printed value), compiled to manuscript_marked.pdf.

Usage (from biodyn-work, after compiling paper/r2/supplement.tex and paper/r2/manuscript.tex)::

    .venv/bin/python subproject_24_multireference_latent_truth_model/paper/r2/marked/build_marked.py

Steps: (1) flatten the current manuscript into manuscript_flat.tex (its comment lines of generated files
dropped); (2) latexdiff --type=UNDERLINE --math-markup=coarse --graphics-markup=new-only, with tabular,
figure and table environments and the reference list diffed as a whole (PICTUREENV; shown in their current form),
because word-level markup inside tables does not compile, captions of moved floats would be diffed against
unrelated text and deleted references would be merged into current ones; (3) post-process: a legend under the
title, citations in deleted text of references the current list no longer has (author-year text from the
previous bibliography), figure paths, the cross-references to
the supplement (../supplement.aux), hyperref bookmark strings, a deleted bibliography item whose key the
current bibliography reuses (renamed so the key is defined once) and, in deleted text only, references to
labels that the current manuscript no longer defines (printed with their number in the previous version,
read from paper/main_clean.aux); (4) latexmk.  Outputs: manuscript_marked.tex, manuscript_marked.pdf.
"""
from __future__ import annotations

import re
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
R2 = HERE.parent
PAPER = R2.parent
OLD = PAPER / "main_clean.tex"
OLD_AUX = PAPER / "main_clean.aux"
FLAT = HERE / "manuscript_flat.tex"
OUT = HERE / "manuscript_marked.tex"
# environments diffed as a whole: tables (word-level markup inside tabular does not compile), figure and table
# floats (their captions, moved relative to the previous text, would otherwise be diffed against unrelated body
# text) and the reference list (word-level diffing merges deleted entries into unrelated current ones); all of them
# appear in their current form
PICTUREENV = r"PICTUREENV=(?:picture|DIFnomarkup|tabular|tabularx|longtable|thebibliography|figure|table)[\w\d*@]*"
# legend printed under the title of the marked copy
LEGEND = ("\\begin{center}\\fbox{\\parbox{0.94\\linewidth}{\\footnotesize\\raggedright "
          "Marked copy (latexdiff against the previously submitted version): \\DIFdel{removed text} is struck through in "
          "red and \\DIFadd{added text} is underlined in blue. Figures, tables (with their captions) and the reference list "
          "are shown in their current form, without marks; figures and tables that were removed are not shown.}}"
          "\\end{center}\n")


def read_macros(p: Path) -> dict[str, str]:
    txt = p.read_text()
    out = {}
    for m in re.finditer(r"\\newcommand\{\\(n[A-Za-z]+)\}\{", txt):
        i, depth = m.end(), 1
        while depth:
            depth += (txt[i] == "{") - (txt[i] == "}")
            i += 1
        out[m.group(1)] = txt[m.end():i - 1]
    return out


def strip_comment_lines(txt: str) -> str:
    return "\n".join(ln for ln in txt.splitlines() if not re.match(r"^\s*%", ln))


def inline(txt: str, base: Path) -> str:
    def rep(m):
        name = m.group(1)
        p = base / (name if name.endswith(".tex") else name + ".tex")
        return inline(strip_comment_lines(p.read_text()), base) if p.exists() else m.group(0)
    txt = re.sub(r"\\InputIfFileExists\{(references_supp_keys)\}\{\}\{\}", rep, txt)
    txt = re.sub(r"\\InputIfFileExists\{numbers\.tex\}\{\}\{\}\n?", "", txt)
    return re.sub(r"(?<!\\)\\input\{([^}]+)\}", rep, txt)


def flatten() -> tuple[int, int]:
    macros = read_macros(R2 / "numbers.tex")
    txt = inline((R2 / "manuscript.tex").read_text(), R2)
    n = [0]

    def mrep(m):
        if m.group(1) not in macros:
            return m.group(0)
        n[0] += 1
        return macros[m.group(1)]
    txt = re.sub(r"\\(n[A-Z][A-Za-z]*)(?:\{\})?(?![A-Za-z])", mrep, txt)
    left = set(re.findall(r"\\(n[A-Z][A-Za-z]*)", txt)) & set(macros)
    if left or re.search(r"(?<!\\)\\input\{", txt):
        raise SystemExit(f"flattening incomplete: macros {sorted(left)}")
    FLAT.write_text(txt)
    return len(macros), n[0]


def old_bib_labels() -> dict[str, tuple[str, str]]:
    """{key: (authors, year)} of the previous version's bibliography (paper/main_clean.tex)."""
    out = {}
    for m in re.finditer(r"\\bibitem\[([^\]]*)\((\d{4}[a-z]?)\)\]\{([^}]+)\}", OLD.read_text()):
        out[m.group(3)] = (m.group(1).strip(), m.group(2))
    return out


def cite_removed_refs(txt: str) -> str:
    """Citations (in deleted text) of references that the current bibliography no longer contains: printed as
    author-year text from the previous bibliography, because the reference list appears in its current form."""
    keys = set(re.findall(r"^\\bibitem\[[^\]]*\]\{([^}]+)\}", txt, flags=re.M))
    old = old_bib_labels()

    def rep(m):
        kind, ks = m.group(1), [k.strip() for k in m.group(2).split(",")]
        if all(k in keys or k not in old for k in ks):
            return m.group(0)
        if kind == "t":
            return ", ".join((f"\\citet{{{k}}}" if k in keys or k not in old else f"{old[k][0]} ({old[k][1]})")
                             for k in ks)
        parts = [(f"\\citealp{{{k}}}" if k in keys or k not in old else f"{old[k][0]}, {old[k][1]}") for k in ks]
        return "(" + "; ".join(parts) + ")"
    lines = []
    for ln in txt.splitlines():
        lines.append(ln if ln.lstrip().startswith("%") else re.sub(r"\\cite([tp])\{([^}]+)\}", rep, ln))
    return "\n".join(lines) + "\n"


def old_labels() -> dict[str, str]:
    out = {}
    for m in re.finditer(r"\\newlabel\{([^}]+)\}\{\{([^}]*)\}", OLD_AUX.read_text()):
        out[m.group(1)] = m.group(2)
    return out


NAMES = {"sec": ("Section", "section"), "fig": ("Figure", "figure"), "tab": ("Table", "table"),
         "eq": ("Equation", "equation"), "app": ("Appendix", "appendix")}


def postprocess(txt: str, missing: set[str] | None = None) -> str:
    txt = txt.replace(r"\graphicspath{{figures/}}", r"\graphicspath{{../figures/}{../../}}")
    txt = txt.replace(r"\externaldocument[]{supplement}", r"\externaldocument[]{../supplement}")
    if r"\maketitle" in txt:
        txt = txt.replace("\\maketitle", "\\maketitle\n" + LEGEND, 1)
    txt = txt.replace(r"\begin{document}",
                      "\\pdfstringdefDisableCommands{\\def\\DIFadd#1{#1}\\def\\DIFdel#1{}\\def\\DIFaddbegin{}"
                      "\\def\\DIFaddend{}\\def\\DIFdelbegin{}\\def\\DIFdelend{}}\n\\begin{document}", 1)
    # a deleted \bibitem whose key a current \bibitem reuses: rename it so that the key is defined once
    new_keys = set(re.findall(r"^\\bibitem\[[^\]]*\]\{([^}]+)\}", txt, flags=re.M))
    txt = re.sub(r"(\\DIFdelbegin \\bibitem\[[^\]]*\]\{)([^}]+)\}",
                 lambda m: m.group(1) + (m.group(2) + "-previous" if m.group(2) in new_keys else m.group(2)) + "}",
                 txt)
    txt = cite_removed_refs(txt)
    if missing:
        lab = old_labels()
        lines = []
        for ln in txt.splitlines():
            if not ln.lstrip().startswith("%"):
                for k in missing:
                    if k not in lab:
                        continue
                    pre = NAMES.get(k.split(":")[0], ("", ""))
                    num = f"({lab[k]})" if k.startswith("eq:") else lab[k]
                    ln = ln.replace(rf"\Cref{{{k}}}", f"{pre[0]}~{num}").replace(rf"\cref{{{k}}}", f"{pre[1]}~{num}")
                    ln = ln.replace(rf"\ref{{{k}}}", num).replace(rf"\eqref{{{k}}}", f"({lab[k]})")
            lines.append(ln)
        txt = "\n".join(lines) + "\n"
    return txt


def compile_() -> int:
    for ext in ("aux", "log", "out", "fls", "fdb_latexmk", "pdf"):
        (HERE / f"manuscript_marked.{ext}").unlink(missing_ok=True)
    return subprocess.run(["latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error", OUT.name], cwd=HERE,
                          capture_output=True).returncode


def undefined_refs() -> set[str]:
    log = (HERE / "manuscript_marked.log").read_text(errors="replace")
    return set(re.findall(r"Reference `([^']+)' on page", log))


def main() -> int:
    n_mac, n_use = flatten()
    print(f"flattened: {n_mac} macros, {n_use} uses expanded -> {FLAT.name}")
    res = subprocess.run(["latexdiff", "--type=UNDERLINE", "--math-markup=coarse", "--graphics-markup=new-only",
                          f"--config={PICTUREENV}", str(OLD), str(FLAT)], capture_output=True, text=True)
    if res.returncode:
        print(res.stderr[-2000:])
        return 1
    raw = res.stdout
    OUT.write_text(postprocess(raw))
    rc = compile_()
    miss = undefined_refs() if (HERE / "manuscript_marked.log").exists() else set()
    if miss:                                      # deleted text that refers to labels of the previous version
        OUT.write_text(postprocess(raw, miss))
        rc = compile_()
    log = (HERE / "manuscript_marked.log").read_text(errors="replace")
    pages = re.search(r"Output written on manuscript_marked\.pdf \((\d+) pages", log)
    n_cit = len(re.findall(r"Citation `[^']+' on page", log))
    n_over = log.count("Overfull")
    print(f"latexmk exit {rc}; pages {pages.group(1) if pages else '?'}; old labels printed as text: {sorted(miss)}; "
          f"undefined refs left: {sorted(undefined_refs())}; undefined citations: {n_cit}; overfull boxes: {n_over}")
    for ext in ("aux", "fls", "fdb_latexmk", "out"):
        (HERE / f"manuscript_marked.{ext}").unlink(missing_ok=True)
    return rc


if __name__ == "__main__":
    if not shutil.which("latexdiff"):
        sys.exit("latexdiff not found")
    sys.exit(main())
