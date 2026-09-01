"""The chat model the agent loop talks to — a thin adapter over :mod:`leaninfer`.

:class:`Model` is the single public backend (a :class:`~nanoagent.core.agent.ChatModel`): the
agent loop and interactive session only ever see
``Model.query(messages, tools, *, on_delta=None) -> Reply`` and never import a provider SDK.
It owns no transport of its own — all inference (tool-schema encoding, streaming, retry, the
two transports) lives in :mod:`leaninfer`. :class:`Model` just:

  1. translates the nanoagent :class:`~nanoagent.config.ModelConfig` into a
     :class:`leaninfer.LeanInferConfig` (mapping the backend name and carrying the fields),
  2. builds the matching leaninfer backend via :func:`leaninfer.backends.build_backend`, and
  3. maps each leaninfer :class:`~leaninfer.types.Response` onto the loop's
     :class:`~nanoagent.core.agent.Reply` (text + tool calls + usage + cost + reasoning).

``ModelConfig.backend`` names the transport and is passed through verbatim: leaninfer resolves it
against its built-ins (``"sglang"``, the OpenAI SDK against an SGLang ``/v1`` endpoint) and then
against the plugin directories in ``$LEANINFER_PLUGINS`` — which is how ``backend: mygateway`` finds
``mygateway.py`` in a directory neither package ships. nanoagent deliberately keeps no allowlist of
its own: it would have to be edited for every new plugin, and leaninfer's own rejection already
names the backends it did find and the directories it searched. leaninfer imports the transport
module only when the backend is built, so constructing one never pulls in another's stack.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import fields
from typing import Any

from leaninfer import LeanInferConfig
from leaninfer.backend import Backend
from leaninfer.backends import build_backend
from leaninfer.types import Response
from nanoagent.core.agent import Reply, ToolCall
from nanoagent.config import ModelConfig


class Model:
    """Unified chat model: a thin adapter mapping nanoagent <-> :mod:`leaninfer`.

    Holds a single long-lived backend across all turns (reusing one connection pool). Unlike
    :func:`leaninfer.infer`, which closes the backend it builds, a :class:`Model` lives for the
    whole CLI/interactive session and relies on process exit to reclaim the pool — there is no
    per-call teardown to wire through the agent loop. A long-lived host (a server, or one that
    rebuilds models) should add an explicit ``aclose`` that awaits ``self._backend.aclose()``.
    """

    def __init__(self, backend: Backend) -> None:
        self._backend = backend

    @classmethod
    def from_config(cls, cfg: ModelConfig) -> Model:
        """Build a :class:`Model` by translating ``cfg`` into a leaninfer backend.

        :class:`~nanoagent.config.ModelConfig`'s fields are a strict subset of
        :class:`LeanInferConfig`'s, with the same names and meanings — it is the same set of
        knobs re-declared as all-required, because a nanoagent config may not inherit a hidden
        default. So the translation is a field-name projection: adding a knob to ModelConfig
        carries it across automatically, and a name that has no LeanInferConfig counterpart
        fails loudly here instead of being silently dropped by a hand-written copy. Fields only
        leaninfer has (parse_thinking, concurrency, retry backoff) keep their leaninfer defaults.
        """
        lean_cfg = LeanInferConfig(**{f.name: getattr(cfg, f.name) for f in fields(cfg)})
        return cls(build_backend(lean_cfg))

    async def query(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        *,
        on_delta: Callable[[str, str], None] | None = None,
    ) -> Reply:
        """Run one tool-calling turn through leaninfer and map the result to a :class:`Reply`.

        ``on_delta(kind, text)`` streams fragments as they arrive; the SGLang transport
        streams token by token.
        """
        response = await self._backend.generate(
            messages, tools=tools, on_delta=on_delta
        )
        return _to_reply(response)


def _to_reply(response: Response) -> Reply:
    """Map a leaninfer :class:`~leaninfer.types.Response` onto a nanoagent :class:`Reply`."""
    return Reply(
        content=response.text,
        tool_calls=[
            ToolCall(id=c.id, name=c.name, arguments=c.arguments)
            for c in response.tool_calls
        ],
        usage=response.usage,
        cost=response.cost,
        reasoning=response.reasoning,
    )
