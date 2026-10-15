"""Fresh release-copy check of the tables-only path (reviewer Q24).

Simulates a fresh clone without committing: copies exactly the files a commit of the working tree would
contain (``git ls-files -co --exclude-standard`` in the subproject) into an empty directory, extracts the
supporting archive (``outputs/release/r2_supporting_outputs.tar.gz``) there, verifies the extracted files
against the archive's SHA-256 list, moves the committed tables out of ``data/r2/tables`` (except
``KEEP_TABLES``, which the run updates rather than writes), and runs the copy's own ``src/r2/reproduce_all.py --tables-only``
with no access to the external raw data (``BIODYN_EXT`` points to a directory that does not exist, so the
default external-drive location is not used either; ``PYTHONPATH`` is cleared).  The tables the copy
produces are then compared byte for byte with the tables of the working tree; a difference confined to
timing or environment columns is reported as such, any other difference, missing table or failed step is
a failure.

Usage (from biodyn-work; threads as in the README)::

    caffeinate -i nice -n 10 .venv/bin/python subproject_24_multireference_latent_truth_model/src/r2/release_check.py \\
        --dest <empty scratch dir>/release_copy [--skip paper_assets] [--trace-inputs]
    .venv/bin/python subproject_24_multireference_latent_truth_model/src/r2/release_check.py --dest <dir> --compare-only

Results: ``outputs/r2/pipeline/release_check/compare.csv`` (one row per table, against
``reference_tables/``, the working tree's tables when the copy was made) and ``summary.json`` in the
working tree; the copy's own run log is ``<dest>/outputs/r2/pipeline/run.log``.

``--paper-assets`` (on the existing copy, after the table check) re-syncs the copy's code and paper sources
from the working tree (the same file list, without ``data/r2/tables``, so the copy keeps the tables it
regenerated; only a kept table the copy lacks, such as this check's own summary, is added), moves the copy's generated paper assets (``paper/r2/{tables,figures,digest}``,
``numbers.tex``, ``references_supp*.tex``) aside, runs the copy's ``reproduce_all.py --tables-only --only
paper_assets`` and compares every regenerated asset with the working tree's
(``paper_assets_compare.csv``, ``summary_paper_assets.json``): text identical apart from time stamps in
provenance lines, PDF pages rendered pixel-identical, PNG pixel-identical, or a difference confined to
run times (the compute table and the ``\nPipe*`` timing macros, which come from the copy's own timing
table).  Both checks are summarised in ``data/r2/tables/pipeline_release_check.csv`` (post hoc procedural
record; the response letter's fresh-copy sentence reads it).  Nothing else in the working tree is written.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

SP = Path(__file__).resolve().parents[2]
TABLES = SP / "data" / "r2" / "tables"
ARCHIVE = SP / "outputs" / "release" / "r2_supporting_outputs.tar.gz"
RESULT_DIR = SP / "outputs" / "r2" / "pipeline" / "release_check"
PY = sys.executable

# tables that describe the run itself (machine, software, timings, build time); every other table must be
# byte-identical (scale_environment records the machine and the time its tables were built)
RUN_TABLES = {"pipeline_stage_timings", "pipeline_environment", "pipeline_refit_check", "scale_environment",
              "pipeline_release_check"}
# the shared dated decision log: each module replaces its own rows and re-sorts the file, so rows appended by
# hand change only the order on the next run; compared as a set of rows
ROW_SET_TABLES = {"eval_analysis_decisions"}
# committed tables the tables-only path updates or keeps rather than writes from scratch: the locked replication
# power design (written only by 'replication.py power'), the shared decision log (each module replaces its own
# rows), the stage-timing table (recorded runtimes of the original runs + this run's measurements) and the
# procedural refit record (written by reproduce_all.py --compare-fits) and this check's own summary
# (pipeline_release_check, written here).  Every other committed table is moved
# out of data/r2/tables in the copy before the run, so the comparison shows that the run regenerated it.
KEEP_TABLES = {"rep_power", "eval_analysis_decisions", "pipeline_stage_timings", "pipeline_refit_check",
               "pipeline_release_check"}
# library caches, settings and the Python environment (not data)
ENV_PATH = re.compile(r"/(site-packages|dist-packages)/|/\.(matplotlib|bokeh|cache|config|jupyter|keras|local)/|"
                      r"/arviz_data/|/\.condarc$")
# columns that record when or how fast something ran, not an analysis result
TIMING_COL = re.compile(r"(^|_)(runtime|wall_time|elapsed|timestamp|written|started|finished|checked_at|built_at|"
                        r"build_time|time_s|seconds)(_|$)", re.I)


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def release_files() -> list[str]:
    """Files a commit of the working tree would contain (tracked + untracked, not ignored)."""
    out = subprocess.run(["git", "-C", str(SP), "ls-files", "-co", "--exclude-standard", "-z"],
                         capture_output=True, check=True).stdout
    return sorted(f for f in out.decode().split("\0") if f and (SP / f).is_file())


def make_copy(dest: Path, archive: Path, keep_tables: bool = False) -> dict:
    if dest.exists() and any(dest.iterdir()):
        raise SystemExit(f"{dest} is not empty; give an empty or new directory")
    dest.mkdir(parents=True, exist_ok=True)
    files = release_files()
    for f in files:
        q = dest / f
        q.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(SP / f, q)
    with tarfile.open(archive, "r:gz") as tf:
        members = tf.getmembers()
        tf.extractall(dest, filter="data")
    lst = dest / "outputs" / "r2" / "SUPPORTING_OUTPUTS.sha256"
    bad = []
    n = 0
    for ln in lst.read_text().splitlines():
        if not ln.strip():
            continue
        h, rel = ln.split("  ", 1)
        n += 1
        if not (dest / rel).is_file() or sha256(dest / rel) != h:
            bad.append(rel)
    if bad:
        raise SystemExit(f"{len(bad)} extracted files do not match the archive's SHA-256 list, e.g. {bad[:3]}")
    # the working tree's tables at copy time: the reference of the comparison (other modules may append to the
    # shared decision log while the copy runs)
    ref = RESULT_DIR / "reference_tables"
    if ref.exists():
        shutil.rmtree(ref)
    shutil.copytree(TABLES, ref)
    moved = []
    if not keep_tables:
        aside = dest / "data" / "r2" / "tables_committed"
        aside.mkdir(parents=True, exist_ok=True)
        for t in sorted((dest / "data" / "r2" / "tables").glob("*.csv")):
            if t.stem not in KEEP_TABLES:
                shutil.move(str(t), aside / t.name)
                moved.append(t.stem)
    return dict(n_repo_files=len(files), n_archive_members=len(members), n_verified=n,
                archive_sha256=sha256(archive), n_tables_moved_aside=len(moved), kept_tables=sorted(KEEP_TABLES))


def clean_env(dest: Path) -> dict:
    env = {k: v for k, v in os.environ.items()
           if k not in ("PYTHONPATH", "R2_TRACE_INPUTS") and "Crucial" not in v}
    env["BIODYN_EXT"] = str(dest / "_no_external_data")        # does not exist: no raw data, no default drive
    for k, v in (("OMP_NUM_THREADS", "2"), ("OPENBLAS_NUM_THREADS", "2"), ("MKL_NUM_THREADS", "2"),
                 ("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=2")):
        env.setdefault(k, v)
    return env


def run_copy(dest: Path, skip: list[str], trace: bool) -> int:
    argv = [PY, str(dest / "src" / "r2" / "reproduce_all.py"), "--tables-only"]
    if skip:
        argv += ["--skip", *skip]
    if trace:
        argv += ["--trace-inputs"]
    log = RESULT_DIR / "copy_run.out"
    with open(log, "w") as fh:
        fh.write(f"# {dt.datetime.now():%Y-%m-%d %H:%M:%S} {' '.join(argv)} (cwd {dest})\n")
        fh.flush()
        return subprocess.run(argv, cwd=dest, env=clean_env(dest), stdout=fh, stderr=subprocess.STDOUT).returncode


def compare_tables(dest: Path, ref: Path = TABLES) -> list[dict]:
    import pandas as pd
    a = {p.name: p for p in ref.glob("*.csv")}
    b = {p.name: p for p in (dest / "data" / "r2" / "tables").glob("*.csv")}
    rows = []
    for name in sorted(set(a) | set(b)):
        stem = name[:-4]
        r = dict(table=stem, status="", differing_columns="", n_differing_cells="", note="")
        if name not in b:
            r["status"] = "missing_in_copy"
        elif name not in a:
            r["status"] = "only_in_copy"
        elif sha256(a[name]) == sha256(b[name]):
            r["status"] = "identical"
        elif stem in ROW_SET_TABLES and sorted(a[name].read_text().splitlines()) == sorted(
                b[name].read_text().splitlines()):
            r["status"] = "same_rows_reordered"
        else:
            x = pd.read_csv(a[name], dtype=str, keep_default_na=False)
            y = pd.read_csv(b[name], dtype=str, keep_default_na=False)
            if list(x.columns) != list(y.columns) or x.shape != y.shape:
                r["status"] = "differs"
                r["note"] = f"shape {x.shape} vs {y.shape}" + (", columns differ" if list(x.columns) != list(y.columns) else "")
            else:
                ne = x.ne(y)
                cols = [c for c in x.columns if ne[c].any()]
                r["differing_columns"] = ";".join(cols)
                r["n_differing_cells"] = int(ne.values.sum())
                paths = any(str(dest) in v for c in cols for v in y.loc[ne[c], c])
                if stem in RUN_TABLES:
                    r["status"] = "run_table"
                elif cols and all(TIMING_COL.search(c) for c in cols):
                    r["status"] = "timing_columns_only"
                else:
                    r["status"] = "differs"
                if paths:
                    r["note"] = "contains the copy's absolute path (path assumption)"
        rows.append(r)
    return rows


def reads_outside(dest: Path) -> dict[str, list[str]]:
    """Files the copy's steps read outside the copy (from its --trace-inputs logs), split into data files
    (a path assumption: e.g. an absolute path of the original working tree stored in an output) and
    environment / library-cache files."""
    out: dict[str, list[str]] = {"data": [], "environment": []}
    root = str(dest) + os.sep
    for f in sorted((dest / "outputs" / "r2" / "pipeline" / "logs").glob("*/trace/*.txt")):
        for ln in f.read_text(errors="replace").splitlines():
            parts = ln.split("\t", 2)
            if len(parts) == 3 and parts[0] == "R" and not parts[2].startswith(root) and Path(parts[2]).is_file():
                k = "environment" if ENV_PATH.search(parts[2]) else "data"
                item = f"{f.stem}: {parts[2]}"
                if item not in out[k]:
                    out[k].append(item)
    return out


# =============================================================================================
# Paper assets in the release copy (--paper-assets) and the summary table (pipeline_release_check.csv)
# =============================================================================================
PAPER = SP / "paper" / "r2"
# generated by make_paper_assets.py (tables, figures, digest, numbers); moved aside in the copy before its run
ASSET_DIRS = ("tables", "figures", "digest")
ASSET_FILES = ("numbers.tex", "references_supp.tex", "references_supp_keys.tex")
RELEASE_TABLE = TABLES / "pipeline_release_check.csv"
# a generation time stamp or the modification time of a source CSV in a provenance comment / digest line
_TS = re.compile(r"\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}(:\d{2})?")
_PROV = re.compile(r"provenance:|^Generated |generated by|\(modified \d{4}-|^\| `[^`]+\.csv` \| \d{4}-", re.I)
# the digest line that lists the module reports (outputs/r2/reports is not part of the release)
_REPORTS_LINE = re.compile(r"^Reports of the modules regenerating tables:")
# assets whose differences are run times by construction: the compute table and the timing macros
TIMING_ASSETS = {"tables/tab_compute.tex"}
_TIMING_MACRO = re.compile(r"^\\newcommand\{\\nPipe[A-Za-z]*\}")


def _norm_lines(text: str) -> list[str]:
    out = []
    for ln in text.splitlines():
        if _REPORTS_LINE.search(ln):
            ln = "<module reports>"
        elif _PROV.search(ln):
            ln = _TS.sub("<TS>", ln)
        out.append(ln)
    return out


def _render(pdf: Path, out_prefix: Path) -> list[Path]:
    subprocess.run(["pdftoppm", "-r", "50", "-png", str(pdf), str(out_prefix)], check=True, capture_output=True)
    return sorted(out_prefix.parent.glob(out_prefix.name + "*.png"))


def _pixels_equal(a: Path, b: Path) -> bool:
    import numpy as np
    from PIL import Image
    x, y = np.asarray(Image.open(a).convert("RGBA")), np.asarray(Image.open(b).convert("RGBA"))
    return x.shape == y.shape and bool(np.array_equal(x, y))


def compare_asset(rel: str, a: Path, b: Path, tmp: Path) -> tuple[str, str]:
    """Status of one generated asset of the copy (b) against the working tree (a): identical,
    provenance_only (time stamps in provenance comments / digest, the list of module reports),
    timing_only (run times: the compute table, \\nPipe* macros), rendered_identical (PDF pages
    pixel-identical at 50 dpi; the bytes differ in metadata), pixel_identical (PNG), or differs."""
    if not b.exists():
        return "missing_in_copy", ""
    if a.read_bytes() == b.read_bytes():
        return "identical", ""
    suf = a.suffix.lower()
    if suf in (".tex", ".txt", ".md", ".csv"):
        la, lb = _norm_lines(a.read_text(errors="replace")), _norm_lines(b.read_text(errors="replace"))
        if la == lb:
            return "provenance_only", ""
        diff = [(x, y) for x, y in zip(la, lb) if x != y]
        if len(la) == len(lb):
            if rel in TIMING_ASSETS:
                return "timing_only", f"{len(diff)} lines"
            if all(_TIMING_MACRO.match(x) and _TIMING_MACRO.match(y) for x, y in diff):
                return "timing_only", "; ".join(f"{x} -> {y}" for x, y in diff)
        return "differs", "; ".join(f"{x[:100]!r} -> {y[:100]!r}" for x, y in diff[:3])
    if suf == ".png":
        return ("pixel_identical", "") if _pixels_equal(a, b) else ("differs", "pixels")
    if suf == ".pdf":
        tmp.mkdir(parents=True, exist_ok=True)
        stem = re.sub(r"\W", "_", rel)
        pa, pb = _render(a, tmp / f"A_{stem}"), _render(b, tmp / f"B_{stem}")
        if len(pa) == len(pb) and all(_pixels_equal(x, y) for x, y in zip(pa, pb)):
            return "rendered_identical", f"{len(pa)} page(s)"
        return "differs", "rendered pages"
    return "differs", "bytes"


# generated locally but not part of the release (.gitignore: it quotes the confidential review correspondence)
EXCLUDED_ASSETS = {"digest/reviewer_points.md"}


def asset_files(root: Path) -> list[str]:
    out = [f"{d}/{p.name}" for d in ASSET_DIRS for p in sorted((root / d).glob("*"))
           if p.is_file() and f"{d}/{p.name}" not in EXCLUDED_ASSETS]
    return out + [f for f in ASSET_FILES if (root / f).is_file()]


def sync_sources(dest: Path) -> int:
    """Refresh the code and paper sources of an existing copy from the working tree (the files a commit
    would contain), keeping the copy's own regenerated data/r2/tables."""
    n = 0
    for f in release_files():
        if f.startswith("data/r2/tables/"):
            # a kept table the copy does not have yet (e.g. this check's own summary, written after the copy
            # was made) is taken from the working tree; every other table stays the copy's own
            if not (Path(f).stem in KEEP_TABLES and not (dest / f).exists()):
                continue
        q = dest / f
        q.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(SP / f, q)
        n += 1
    return n


def paper_assets_check(dest: Path) -> dict:
    """Re-sync the copy's sources, move its generated paper assets aside, run the copy's own
    ``reproduce_all.py --tables-only --only paper_assets`` from the copy's tables and compare every
    regenerated asset with the working tree's (``paper_assets_compare.csv``)."""
    if not (dest / "outputs" / "r2" / "SUPPORTING_OUTPUTS.sha256").is_file():
        raise SystemExit(f"{dest} is not a release copy (run the table check first)")
    n_sync = sync_sources(dest)
    cp = dest / "paper" / "r2"
    aside = dest / "paper" / "r2_assets_committed"
    if aside.exists():
        shutil.rmtree(aside)
    aside.mkdir(parents=True)
    for d in ASSET_DIRS:
        if (cp / d).exists():
            shutil.move(str(cp / d), aside / d)
        (cp / d).mkdir()
    for f in ASSET_FILES:
        if (cp / f).exists():
            shutil.move(str(cp / f), aside / f)
    argv = [PY, str(dest / "src" / "r2" / "reproduce_all.py"), "--tables-only", "--only", "paper_assets"]
    with open(RESULT_DIR / "copy_paper_assets_run.out", "w") as fh:
        fh.write(f"# {dt.datetime.now():%Y-%m-%d %H:%M:%S} {' '.join(argv)} (cwd {dest})\n")
        fh.flush()
        rc = subprocess.run(argv, cwd=dest, env=clean_env(dest), stdout=fh, stderr=subprocess.STDOUT).returncode
    tmp = dest / "_render"
    rows = []
    want = asset_files(PAPER)
    for rel in want:
        st, det = compare_asset(rel, PAPER / rel, cp / rel, tmp)
        rows.append(dict(asset=rel, status=st, detail=det))
    for rel in sorted(set(asset_files(cp)) - set(want)):
        rows.append(dict(asset=rel, status="only_in_copy", detail=""))
    shutil.rmtree(tmp, ignore_errors=True)
    with open(RESULT_DIR / "paper_assets_compare.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["asset", "status", "detail"], lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    counts: dict[str, int] = {}
    for r in rows:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    out = dict(dest=str(dest), finished=dt.datetime.now().isoformat(timespec="seconds"), n_sources_synced=n_sync,
               copy_run_exit=rc, n_assets=len(want), counts=counts,
               differs=[r["asset"] for r in rows if r["status"] in ("differs", "missing_in_copy", "only_in_copy")],
               timing_only=[r["asset"] for r in rows if r["status"] == "timing_only"])
    (RESULT_DIR / "summary_paper_assets.json").write_text(json.dumps(out, indent=2) + "\n")
    return out


def release_table_rows(summary: dict, compare_rows: list[dict], assets: dict | None, archive: Path,
                       dest: Path | None = None) -> list[dict]:
    """Rows of ``pipeline_release_check.csv`` (key / value): the archive, the table check and the
    paper-asset check of the fresh release copy."""
    kept = set(summary.get("kept_tables") or KEEP_TABLES)
    st = {r["table"]: r["status"] for r in compare_rows}
    regen = [t for t in st if t not in kept]
    ext = None
    if dest is not None and (dest / "outputs" / "r2" / "SUPPORTING_OUTPUTS.sha256").is_file():
        rels = [ln.split("  ", 1)[1] for ln in
                (dest / "outputs" / "r2" / "SUPPORTING_OUTPUTS.sha256").read_text().splitlines() if ln.strip()]
        ext = sum((dest / r).stat().st_size for r in rels if (dest / r).is_file())

    def n(names, *status):
        return sum(1 for t in names if st[t] in status)

    rows = [
        ("archive", "bytes_compressed", archive.stat().st_size if archive.is_file() else "", "size of the archive"),
        ("archive", "gib_compressed", round(archive.stat().st_size / 2 ** 30, 2) if archive.is_file() else "",
         "size of the archive (GiB)"),
        ("archive", "gib_extracted", round(ext / 2 ** 30, 2) if ext is not None else "",
         "extracted size of the archive members (GiB)"),
        ("archive", "n_files", summary.get("n_verified", ""), "archive members verified against their SHA-256"),
        ("tables", "n_repo_files", summary.get("n_repo_files", ""), "repository files copied (git ls-files -co)"),
        ("tables", "copy_run_exit", summary.get("copy_run_exit", ""), "exit code of the copy's tables-only run"),
        ("tables", "n_tables", len(st), "committed tables compared"),
        ("tables", "n_regenerated", len(regen), "committed tables moved aside in the copy and regenerated"),
        ("tables", "n_regenerated_identical", n(regen, "identical"), "regenerated tables byte-identical"),
        ("tables", "n_regenerated_timing_only", n(regen, "timing_columns_only"),
         "regenerated tables that differ only in run-time columns"),
        ("tables", "n_regenerated_run_records", n(regen, "run_table"),
         "regenerated tables that record the run itself (machine, build time)"),
        ("tables", "n_kept", len([t for t in st if t in kept]), "tables the run updates rather than writes"),
        ("tables", "n_differ", n(st, "differs", "missing_in_copy", "only_in_copy"),
         "tables that differ in an analysis column, are missing or are extra"),
        ("tables", "n_data_reads_outside", len(summary.get("reads_outside_copy_data") or []),
         "data files read outside the copy"),
        ("tables", "finished", summary.get("finished", ""), "end of the table check"),
    ]
    if assets:
        c = assets.get("counts", {})
        same = sum(c.get(k, 0) for k in ("identical", "provenance_only", "rendered_identical", "pixel_identical"))
        rows += [
            ("paper_assets", "copy_run_exit", assets.get("copy_run_exit", ""),
             "exit code of the copy's paper_assets stage"),
            ("paper_assets", "n_assets", assets.get("n_assets", ""), "generated tables, figures, digest and numbers"),
            ("paper_assets", "n_same", same,
             "assets identical apart from time stamps (text) or metadata (PDF pages rendered pixel-identical)"),
            ("paper_assets", "n_timing_only", c.get("timing_only", 0), "assets that differ only in run times"),
            ("paper_assets", "n_differ", len(assets.get("differs", [])), "assets that differ otherwise or are missing"),
            ("paper_assets", "finished", assets.get("finished", ""), "end of the paper-asset check"),
        ]
    return [dict(check=a, key=k, value=v, description=d, analysis_status="post_hoc") for a, k, v, d in rows]


def write_release_table(archive: Path = ARCHIVE, dest: Path | None = None) -> Path | None:
    sj, cj, aj = RESULT_DIR / "summary.json", RESULT_DIR / "compare.csv", RESULT_DIR / "summary_paper_assets.json"
    if not (sj.is_file() and cj.is_file()):
        return None
    summary = json.loads(sj.read_text())
    with open(cj, newline="") as fh:
        comp = list(csv.DictReader(fh))
    assets = json.loads(aj.read_text()) if aj.is_file() else None
    dest = dest or Path(summary.get("dest", ""))
    rows = release_table_rows(summary, comp, assets, archive, dest if dest.is_dir() else None)
    with open(RELEASE_TABLE, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    return RELEASE_TABLE


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dest", type=Path, required=True, help="empty directory for the release copy")
    ap.add_argument("--archive", type=Path, default=ARCHIVE)
    ap.add_argument("--skip", nargs="*", default=["paper_assets"], help="stages to leave out (default paper_assets)")
    ap.add_argument("--trace-inputs", action="store_true", help="trace the files the copy's steps open")
    ap.add_argument("--compare-only", action="store_true", help="only compare the tables of an existing copy")
    ap.add_argument("--keep-tables", action="store_true",
                    help="leave the committed tables in place in the copy (default: move them aside so that every "
                         "table must be regenerated)")
    ap.add_argument("--paper-assets", action="store_true",
                    help="on an existing copy: re-sync its code and paper sources (keeping its regenerated tables), "
                         "regenerate the paper assets there (paper_assets stage) and compare them with the working "
                         "tree's (paper_assets_compare.csv)")
    ap.add_argument("--write-table", action="store_true",
                    help="only write data/r2/tables/pipeline_release_check.csv from the stored check results")
    a = ap.parse_args(argv)
    dest = a.dest.resolve()
    if str(dest).startswith(str(SP) + os.sep) or dest == SP:
        raise SystemExit("the release copy must be outside the working tree")
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    if a.write_table:
        print(write_release_table(a.archive, dest))
        return 0
    if a.paper_assets:
        out = paper_assets_check(dest)
        print(json.dumps({k: out[k] for k in ("copy_run_exit", "n_assets", "counts", "differs", "timing_only")},
                         indent=2))
        print(write_release_table(a.archive, dest))
        return 1 if (out["copy_run_exit"] or out["differs"]) else 0
    summary = dict(dest=str(dest), started=dt.datetime.now().isoformat(timespec="seconds"), skip=a.skip)
    rc = 0
    if not a.compare_only:
        summary.update(make_copy(dest, a.archive, a.keep_tables))
        print(f"copy: {summary['n_repo_files']} repository files + {summary['n_archive_members']} archive members "
              f"({summary['n_verified']} verified) in {dest}", flush=True)
        rc = run_copy(dest, a.skip, a.trace_inputs)
        summary["copy_run_exit"] = rc
    ref = RESULT_DIR / "reference_tables"
    rows = compare_tables(dest, ref if ref.is_dir() else TABLES)
    if ref.is_dir():      # working-tree tables that changed while the copy ran (not the copy's doing)
        summary["changed_in_working_tree_since_copy"] = sorted(
            p.stem for p in TABLES.glob("*.csv") if not (ref / p.name).exists() or sha256(p) != sha256(ref / p.name))
    outside = reads_outside(dest)
    summary["reads_outside_copy_data"] = outside["data"]
    summary["reads_outside_copy_environment"] = len(outside["environment"])
    with open(RESULT_DIR / "compare.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    counts: dict[str, int] = {}
    for r in rows:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    summary.update(finished=dt.datetime.now().isoformat(timespec="seconds"), counts=counts,
                   differs=[r["table"] for r in rows if r["status"] in ("differs", "missing_in_copy", "only_in_copy")],
                   timing_columns_only=[f"{r['table']} ({r['differing_columns']})" for r in rows
                                        if r["status"] == "timing_columns_only"])
    (RESULT_DIR / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: summary[k] for k in ("counts", "differs", "timing_columns_only",
                                              "reads_outside_copy_data")}, indent=2))
    # a new table check makes an earlier paper-asset check of another copy stale
    if not a.compare_only and (RESULT_DIR / "summary_paper_assets.json").is_file():
        (RESULT_DIR / "summary_paper_assets.json").unlink()
    write_release_table(a.archive, dest)
    return 1 if (rc or summary["differs"] or outside["data"]) else 0


if __name__ == "__main__":
    sys.exit(main())
