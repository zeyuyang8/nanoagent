"""The agent loop and the contracts it is written against — the RL rollout hot path.

:mod:`~nanoagent.harness.core.agent` (the one loop), :mod:`~nanoagent.harness.core.tool` (what a tool is and how
a YAML names one), :mod:`~nanoagent.harness.core.model` (the transport), and the three optional seams the
loop checks for ``None``: :mod:`~nanoagent.harness.core.hooks`, :mod:`~nanoagent.harness.core.events` and
:mod:`~nanoagent.harness.core.workspace`. Nothing here imports from :mod:`nanoagent.harness.repl` or
:mod:`nanoagent.harness.run`; the dependency runs the other way, which is what keeps a rollout free of
everything the terminal needs.
"""
