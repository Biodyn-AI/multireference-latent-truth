"""Manuscript assets (LaTeX tables, figures, number digests) for the SAGMB R2 paper.

Every printed number is read from ``data/r2/tables/*.csv`` by these scripts;
nothing is typed by hand.  Modules:

* ``style``   -- shared paths, page geometry, fonts, the fixed Okabe-Ito colour
  mapping per estimator / reference / family / tier / universe, display labels,
  number formatters and provenance helpers (shared by all asset modules);
* ``tables``  -- LaTeX tables (``paper/r2/tables/*.tex``);
* ``figures`` -- main figures fig1..fig8 (``paper/r2/figures/*.{pdf,png}`` plus
  ``<name>_caption.txt``);
* ``digest``  -- Markdown number digests (``paper/r2/digest/*.md``).

CLI: ``python src/r2/make_paper_assets.py {tables,figures,digest,all}``.
"""
