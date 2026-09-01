"""The agent loop and the contracts it is written against — the RL rollout hot path.

:mod:`~nanoagent.core.agent` (the one loop), :mod:`~nanoagent.core.tool` (what a tool is and how
a YAML names one), :mod:`~nanoagent.core.model` (the transport), and the three optional seams the
loop checks for ``None``: :mod:`~nanoagent.core.hooks`, :mod:`~nanoagent.core.events` and
:mod:`~nanoagent.core.workspace`. Nothing here imports from :mod:`nanoagent.repl` or
:mod:`nanoagent.run`; the dependency runs the other way, which is what keeps a rollout free of
everything the terminal needs.
"""
