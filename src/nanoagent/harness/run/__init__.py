"""Turning a config into a run, and a run into files on disk.

:mod:`~nanoagent.harness.run.build` (config -> Agent), :mod:`~nanoagent.harness.run.batch` (the concurrent
fan-out and its resume ledger), :mod:`~nanoagent.harness.run.progress` (how that renders),
:mod:`~nanoagent.harness.run.trajectory` (the persisted transcript), plus :mod:`~nanoagent.harness.run.cli`,
the entry point. A downstream benchmark runner imports from here rather than keeping its own copy.
"""
