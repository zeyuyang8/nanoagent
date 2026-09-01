"""Config → :class:`~nanoagent.harness.core.agent.Agent`: the one place a run is assembled.

Every driver goes through here — nanoagent's batch mode, the REPL, and any external runner (whose
``RunConfig`` reuses nanoagent's :class:`~nanoagent.harness.config.ModelConfig` /
:class:`~nanoagent.harness.config.AgentConfig` verbatim, so one function serves both). A second copy of
this assembly is exactly how chat and batch would start answering the same config differently.

  * :func:`build_prompt_and_tools` — the system prompt (``system_prompt`` + the project context
    files + the one-line skill index) and the toolset, optionally narrowed by
    :func:`select_tools`. Its own function because the REPL builds its session directly rather
    than through :func:`build_agent`.
  * :func:`build_agent` — that, plus the model, the loop budgets, the hooks and the event writer.
  * :func:`context_text` — the ``AGENTS.md`` / ``CLAUDE.md`` concatenation the prompt folds in.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nanoagent.harness.core.agent import Agent
    from nanoagent.harness.config import AgentConfig, ModelConfig
    from nanoagent.harness.core.tool import Tool

# Suffix of the file that shadows a context file: AGENTS.md -> AGENTS.override.md.
OVERRIDE_SUFFIX = ".override.md"


def context_text(paths: Iterable[str | Path]) -> str:
    """Concatenate the project's own instruction files that exist, honoring ``.override.md``.

    A sibling ``<name>.override.md`` REPLACES its file instead of adding to it — so a checkout can
    shadow committed instructions without editing them, and can shadow them down to nothing, which
    concatenation cannot express.
    """
    chunks = []
    for path in paths:
        file = Path(path)
        override = file.with_name(file.stem + OVERRIDE_SUFFIX)
        # The override wins outright — including when it is empty, which is how a checkout says
        # "ignore the committed instructions entirely".
        source = override if override.is_file() else file
        if source.is_file():
            chunks.append(f"\n\n# {file.name}\n\n{source.read_text(encoding='utf-8').strip()}")
    return "".join(chunks)


def build_prompt_and_tools(
    agent_cfg: AgentConfig,
    tool_paths: list[str],
    tools_dir: str | None = None,
    allowed: list[str] | None = None,
) -> tuple[str, list[Tool]]:
    """The system prompt and toolset a config describes, once, for every driver.

    The toolset is the configured tools, plus whatever the agent wrote for itself in a previous
    session, plus a ``skill`` / ``write_tool`` tool when those are configured — then narrowed to
    ``allowed`` (by tool name) when the config gives one.
    """
    from nanoagent.harness.tools.write import WriteTool, written_tool_specs
    from nanoagent.harness.tools.skill import discover, Skill, skill_index
    from nanoagent.harness.core.tool import get_tools

    tools = get_tools([*tool_paths, *written_tool_specs(tools_dir)])
    if tools_dir is not None:
        tools.append(WriteTool(tools_dir))
    skills = discover(agent_cfg.skills)
    if skills:
        tools.append(Skill(skills))
    if allowed is not None:
        tools = select_tools(tools, allowed)
    prompt = agent_cfg.system_prompt + context_text(agent_cfg.context_files) + skill_index(skills)
    return prompt, tools


def select_tools(tools: list[Tool], allowed: list[str]) -> list[Tool]:
    """``tools`` narrowed to the names in ``allowed``, in the configured order.

    A name that matches nothing raises rather than being ignored: the failure mode of a typo is
    an agent quietly missing the tool the run was about, which reads as the model being bad at
    the task.
    """
    unknown = sorted(set(allowed) - {t.name for t in tools})
    if unknown:
        have = ", ".join(sorted(t.name for t in tools)) or "(none)"
        raise ValueError(f"no such tool: {', '.join(unknown)}; the toolset has {have}")
    return [t for t in tools if t.name in set(allowed)]


def build_agent(
    model_cfg: ModelConfig,
    agent_cfg: AgentConfig,
    tool_paths: list[str],
    tools_dir: str | None = None,
    allowed_tools: list[str] | None = None,
) -> Agent:
    """The agent a ``run`` config describes: its model, its toolset and its loop budgets."""
    # Deferred (not module-level) so importing this module — or a CLI whose command only reports —
    # stays light: only this path pulls in openai, via Model.
    from nanoagent.harness.core.agent import Agent
    from nanoagent.harness.core.events import EventWriter
    from nanoagent.harness.core.hooks import get_hooks
    from nanoagent.harness.core.model import Model

    prompt, tools = build_prompt_and_tools(agent_cfg, tool_paths, tools_dir, allowed_tools)
    return Agent(
        Model.from_config(model_cfg),
        tools,
        system_prompt=prompt,
        max_steps=agent_cfg.max_steps,
        cost_limit=agent_cfg.cost_limit,
        token_limit=agent_cfg.token_limit,
        context_window=agent_cfg.context_window,
        hooks=get_hooks(agent_cfg.hooks),
        # One writer for the whole batch: Agent.run splits it per rollout, stamping each line
        # with the task id it was given as `label`.
        events=EventWriter(agent_cfg.events) if agent_cfg.events else None,
    )
