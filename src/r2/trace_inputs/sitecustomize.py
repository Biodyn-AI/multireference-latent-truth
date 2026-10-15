"""Input tracing for ``reproduce_all.py --trace-inputs`` (not imported otherwise).

``reproduce_all.py`` puts this directory first on ``PYTHONPATH`` of every step process when tracing
is on, so Python imports this file at start-up.  With ``R2_TRACE_INPUTS=<file>`` set, an audit hook
appends one line per distinct (kind, absolute path) to <file>:

* ``R`` / ``W``: a file opened for reading / writing (``open``, ``io.open``, ``os.open``, ``np.load``,
  pandas readers and writers, and the ``pyarrow.parquet`` functions, which open files natively and are
  wrapped here);
* ``L``: a directory listed (``os.listdir``, ``os.scandir``; ``glob`` and ``pathlib.glob`` use these).

Paths under the interpreter's own prefixes are skipped.  The trace defines which stored outputs the
tables-only path reads (``reproduce_all.py --release-archive``); it changes nothing in the analysis.
"""
import os
import sys

_OUT = os.environ.get("R2_TRACE_INPUTS")
if _OUT:
    _SKIP = tuple(sorted({os.path.realpath(p) + os.sep for p in (sys.prefix, sys.base_prefix, sys.exec_prefix)}
                         | {"/dev/", "/System/", "/usr/", "/Library/", "/etc/"}))
    _fh = open(_OUT, "a", buffering=1)
    _seen = set()
    _W_FLAGS = os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_APPEND | os.O_TRUNC

    def _rec(kind, path):
        try:
            p = os.path.abspath(os.fsdecode(path))
        except Exception:
            return
        if p.startswith(_SKIP) or (kind, p) in _seen:
            return
        _seen.add((kind, p))
        try:
            _fh.write(f"{kind}\t{os.getpid()}\t{p}\n")
        except Exception:
            pass

    def _hook(event, args):
        if event == "open":
            path, mode, flags = (tuple(args) + (None, None, None))[:3]
            if path is None or isinstance(path, int):
                return
            if isinstance(mode, str):
                w = any(c in mode for c in "wax+")
            else:
                w = bool((flags or 0) & _W_FLAGS)
            _rec("W" if w else "R", path)
        elif event in ("os.listdir", "os.scandir"):
            path = args[0] if args else None
            if isinstance(path, int):
                return
            _rec("L", "." if path is None else path)

    sys.addaudithook(_hook)

    # pyarrow.parquet opens files natively (no audit event): wrap its entry points once it is imported
    import importlib.abc
    import importlib.machinery

    def _wrap_pq(mod):
        def wrap(fn, kind):
            def inner(where, *a, **k):
                if isinstance(where, (str, bytes, os.PathLike)):
                    _rec(kind, where)
                return fn(where, *a, **k)
            inner.__wrapped__ = fn
            return inner
        for name in ("read_table", "ParquetFile", "read_metadata", "read_schema"):
            if hasattr(mod, name):
                setattr(mod, name, wrap(getattr(mod, name), "R"))
        if hasattr(mod, "write_table"):
            fn = mod.write_table

            def write_table(table, where, *a, **k):
                if isinstance(where, (str, bytes, os.PathLike)):
                    _rec("W", where)
                return fn(table, where, *a, **k)
            mod.write_table = write_table

    class _Finder(importlib.abc.MetaPathFinder):
        def find_spec(self, name, path, target=None):
            if name != "pyarrow.parquet":
                return None
            for f in sys.meta_path:
                if f is self:
                    continue
                spec = getattr(f, "find_spec", lambda *a: None)(name, path, target)
                if spec is not None and spec.loader is not None:
                    loader = spec.loader
                    orig = loader.exec_module

                    def exec_module(module, _orig=orig):
                        _orig(module)
                        _wrap_pq(module)
                    loader.exec_module = exec_module
                    return spec
            return None

    sys.meta_path.insert(0, _Finder())
