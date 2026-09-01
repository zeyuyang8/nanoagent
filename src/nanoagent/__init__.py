"""nanoagent — a minimal, clean-room agent loop with structured tool calling.

Public surface (the stdlib-only core — no ``openai`` import):
  * :class:`~nanoagent.core.agent.Agent` / :class:`~nanoagent.core.agent.AgentResult` — the
    model-agnostic loop, the ONLY one in the package (the ``chat`` REPL drives it too).
  * :class:`~nanoagent.core.tool.Tool` — a structured, JSON-Schema tool;
    :func:`~nanoagent.core.tool.get_tools` loads them from tool-config YAMLs.

Five subpackages, and the dependency arrows only ever point left:

  * :mod:`nanoagent.inference` — how a model is actually reached: the transport backends and
    their plugin resolver, the concurrent batch engine, and the SGLang serve/launch side. The
    only subpackage the others may import *upwards* from ``core.model``.
  * :mod:`nanoagent.core` — the loop and what it is made of: ``agent``, ``tool``, ``model``,
    plus the seams ``hooks`` / ``events`` / ``workspace``. Imports nothing from ``tools`` /
    ``run`` / ``repl``, and only ``core.model`` reaches into ``inference``.
  * :mod:`nanoagent.tools` — the tools themselves: ``bash``, ``code``, ``file``, ``write``,
    ``skill``. Each is a ``Tool`` subclass a YAML can name; none is loaded unless one does.
  * :mod:`nanoagent.run` — a config becomes a run: ``build`` (config -> Agent), ``batch``
    (the concurrent fan-out and its resume ledger), ``progress``, ``trajectory``, ``cli``.
  * :mod:`nanoagent.repl` — chat only: ``app`` (the REPL over the same loop), ``commands``,
    ``tree`` (a branching transcript), ``browser``.

The model backend lives in :mod:`nanoagent.core.model` and is imported explicitly
(``from nanoagent.core.model import Model``) so the core loop stays free of any provider SDK —
any object satisfying the ``ChatModel`` duck-type works as a model.
:class:`~nanoagent.core.model.Model` is a thin adapter over :mod:`nanoagent.inference`, which
resolves ``ModelConfig.backend`` against its own built-in transports and then against the plugin
directories in ``$NANOAGENT_PLUGINS`` — so an SGLang server, an OpenAI-compatible gateway, or a
private transport dropped in as a plugin file all work with no change here.

Everything outside ``core.agent`` is a seam onto that one loop, off unless a config names it, so
the RL rollout path executes the same code it always did:

  * ``core.hooks`` — ``session_start`` / ``before_llm`` / ``before_tool`` / ``after_tool`` from a
    plain ``.py``; the way a prompt rule ("call search at most once") becomes enforced.
  * ``tools.skill`` — ``SKILL.md`` files indexed by name and description, body loaded on demand;
    folded into the prompt by ``run.build``, along with the ``AGENTS.md`` / ``CLAUDE.md`` files.
  * ``tools.write`` — the agent writing a new tool's ``.py`` + ``.yaml`` and using it on the next
    turn, which is only possible because ``core.tool.get_tools`` already needs nothing else.
  * ``core.events`` — the run mirrored to NDJSON, for watching a rollout from outside.
  * ``core.workspace`` / ``tools.file`` — a per-rollout root and ``Read`` / ``Write`` / ``Edit``
    resolved inside it.
  * ``repl.tree`` / ``repl.commands`` — chat-only: a branching transcript, resume, and the
    slash-command table.
"""

from __future__ import annotations

from nanoagent.core.agent import Agent, AgentResult, Reply, ToolCall
from nanoagent.core.tool import get_tools, Tool

__version__ = "0.2.0"

__all__ = ["Agent", "AgentResult", "get_tools", "Reply", "ToolCall", "Tool"]
