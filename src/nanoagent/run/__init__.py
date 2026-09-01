"""Turning a config into a run, and a run into files on disk.

:mod:`~nanoagent.run.build` (config -> Agent), :mod:`~nanoagent.run.batch` (the concurrent
fan-out and its resume ledger), :mod:`~nanoagent.run.progress` (how that renders),
:mod:`~nanoagent.run.trajectory` (the persisted transcript), plus :mod:`~nanoagent.run.cli`,
the entry point. A downstream benchmark runner imports from here rather than keeping its own copy.
"""
