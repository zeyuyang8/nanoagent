"""The harness: everything that runs an agent, once a model can be reached.

The package has exactly two halves. :mod:`nanoagent.inference` is how a model is *reached*;
this is what is *done* with one. The split is a real boundary, not a filing convention — the
arrow points one way, from here into ``inference``, through the single adapter
:mod:`nanoagent.harness.core.model` and nowhere else. Serving a model needs none of this, and the
loop needs none of the serving side.

Four subpackages, and the dependency arrows only ever point left:

  * :mod:`nanoagent.harness.core` — the loop and what it is made of: ``agent``, ``tool``,
    ``model``, plus the seams ``hooks`` / ``events`` / ``workspace``. Imports nothing from
    ``tools`` / ``run`` / ``repl``. This is the RL rollout hot path.
  * :mod:`nanoagent.harness.tools` — the tools themselves: ``bash``, ``code``, ``file``,
    ``write``, ``skill``. Each is a ``Tool`` subclass a YAML can name; none is loaded unless
    one does.
  * :mod:`nanoagent.harness.run` — a config becomes a run: ``build`` (config -> Agent),
    ``batch`` (the concurrent fan-out and its resume ledger), ``progress``, ``trajectory``,
    ``cli``.
  * :mod:`nanoagent.harness.repl` — chat only: ``app`` (the REPL over the same loop),
    ``commands``, ``tree`` (a branching transcript), ``browser``.

Alongside them, :mod:`nanoagent.harness.config` (the all-``MISSING`` run schemas) and
``configs/`` (the defaults that ship inside the wheel — ``mgen.yaml`` and the three tool
manifests it names).

The names below are re-exported from :mod:`nanoagent` itself, so ``from nanoagent import Agent``
and ``from nanoagent.harness import Agent`` are the same object.
"""

from __future__ import annotations

from nanoagent.harness.core.agent import Agent, AgentResult, Reply, ToolCall
from nanoagent.harness.core.tool import get_tools, Tool

__all__ = ["Agent", "AgentResult", "get_tools", "Reply", "Tool", "ToolCall"]
