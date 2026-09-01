"""Module entry point: ``python -m nanoagent <command> [args...]``.

All dispatch lives in :mod:`nanoagent.run.cli` — see its :func:`main` for the
``run`` / ``chat`` / ``browse`` subcommands.
"""

from __future__ import annotations

from nanoagent.run.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
