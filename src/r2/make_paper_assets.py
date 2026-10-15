"""Build the manuscript assets of the SAGMB R2 paper from ``data/r2/tables/*.csv``.

Usage (from the subproject root or from ``biodyn-work``)::

    .venv/bin/python src/r2/make_paper_assets.py tables   [module args...]
    .venv/bin/python src/r2/make_paper_assets.py figures  [module args...]
    .venv/bin/python src/r2/make_paper_assets.py digest   [module args...]
    .venv/bin/python src/r2/make_paper_assets.py numbers  [--check]
    .venv/bin/python src/r2/make_paper_assets.py all

Each subcommand runs the ``main(argv)`` of the corresponding modules of the
``r2.paper_assets`` package (main and supplementary variants when present):

* ``tables``  -> ``tables``, ``tables_supp``  -> ``paper/r2/tables/*.tex``
* ``figures`` -> ``figures``, ``figures_supp`` -> ``paper/r2/figures/*.{pdf,png}`` + captions
* ``digest``  -> ``digest``                  -> ``paper/r2/digest/*.md``
* ``numbers`` -> ``prose_numbers``           -> ``paper/r2/numbers.tex`` (one macro per number of the
  manuscript prose, from the CSVs), ``outputs/r2/reports/binding_number_audit.csv``, and a check that the
  prose contains no literal result number (``--check``: check only; exit 1 on any problem)

Extra arguments after a single subcommand are passed to the main module of that
subcommand only (for example ``figures fig4_heldout --out-dir /tmp/x``).
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # src/

GROUPS = {
    "tables": ["tables", "tables_supp"],
    "figures": ["figures", "figures_supp"],
    "digest": ["digest"],
    "numbers": ["prose_numbers"],
}


def _run(module: str, argv: list[str]) -> int:
    try:
        mod = importlib.import_module(f"r2.paper_assets.{module}")
    except ModuleNotFoundError as e:
        if e.name == f"r2.paper_assets.{module}":
            print(f"[make_paper_assets] {module}: module not present, skipped")
            return 0
        raise
    main = getattr(mod, "main", None)
    if main is None:
        print(f"[make_paper_assets] {module}: no main(), skipped")
        return 0
    print(f"[make_paper_assets] running r2.paper_assets.{module} {' '.join(argv)}")
    rc = main(argv)
    return int(rc or 0)


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in {"-h", "--help"} or argv[0] not in {*GROUPS, "all"}:
        print(__doc__)
        return 0 if (not argv or argv[0] in {"-h", "--help"}) else 2
    cmd, rest = argv[0], argv[1:]
    rc = 0
    if cmd == "all":
        for group in ("tables", "figures", "digest", "numbers"):
            for module in GROUPS[group]:
                rc |= _run(module, [])
        return rc
    mods = GROUPS[cmd]
    rc |= _run(mods[0], rest)
    for module in mods[1:]:
        rc |= _run(module, [])
    return rc


if __name__ == "__main__":
    sys.exit(main())
