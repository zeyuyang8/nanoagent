"""nanoagent — a minimal, clean-room agent loop with structured tool calling.

Public surface (the stdlib-only core — no ``openai`` import):
  * :class:`~nanoagent.harness.core.agent.Agent` /
    :class:`~nanoagent.harness.core.agent.AgentResult` — the model-agnostic loop, the ONLY one in
    the package (the ``chat`` REPL drives it too).
  * :class:`~nanoagent.harness.core.tool.Tool` — a structured, JSON-Schema tool;
    :func:`~nanoagent.harness.core.tool.get_tools` loads them from tool-config YAMLs.

Two subpackages, because there are only two things here:

  * :mod:`nanoagent.inference` — how a model is *reached*: the transport backends and their
    plugin resolver, the concurrent batch engine, and the SGLang serve / router / launch side.
  * :mod:`nanoagent.harness` — what is *done* with one: the loop (``core``), the ``tools`` it
    calls, the ``run`` drivers that turn a config into a run, and the ``repl``.

The arrow points one way, from ``harness`` into ``inference``, and through exactly one module:
:class:`~nanoagent.harness.core.model.Model` is a thin adapter over :mod:`nanoagent.inference`,
which resolves ``ModelConfig.backend`` against its own built-in transports and then against the
plugin directories in ``$NANOAGENT_PLUGINS`` — so an SGLang server, an OpenAI-compatible gateway,
or a private transport dropped in as a plugin file all work with no change here. The loop itself
imports no provider SDK; any object satisfying the ``ChatModel`` duck-type is a usable model.

Everything outside ``harness.core.agent`` is a seam onto that one loop, off unless a config names
it, so the RL rollout path executes the same code it always did:

  * ``harness.core.hooks`` — ``session_start`` / ``before_llm`` / ``before_tool`` / ``after_tool``
    from a plain ``.py``; the way a prompt rule ("call search at most once") becomes enforced.
  * ``harness.tools.skill`` — ``SKILL.md`` files indexed by name and description, body loaded on
    demand; folded into the prompt by ``harness.run.build``, along with the ``AGENTS.md`` /
    ``CLAUDE.md`` files.
  * ``harness.tools.write`` — the agent writing a new tool's ``.py`` + ``.yaml`` and using it on
    the next turn, which is only possible because ``harness.core.tool.get_tools`` already needs
    nothing else.
  * ``harness.core.events`` — the run mirrored to NDJSON, for watching a rollout from outside.
  * ``harness.core.workspace`` / ``harness.tools.file`` — a per-rollout root and ``Read`` /
    ``Write`` / ``Edit`` resolved inside it.
  * ``harness.repl.tree`` / ``harness.repl.commands`` — chat-only: a branching transcript, resume,
    and the slash-command table.
"""

from __future__ import annotations

from nanoagent.harness import Agent, AgentResult, get_tools, Reply, Tool, ToolCall

__version__ = "0.2.0"

__all__ = ["Agent", "AgentResult", "get_tools", "Reply", "Tool", "ToolCall"]
