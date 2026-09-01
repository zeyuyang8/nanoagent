"""Request and Response dataclasses for batch inference.

A :class:`Request` is one chat-completion input — a list of OpenAI-shape messages
(``{"role": ..., "content": ...}``). :meth:`Request.coerce` normalizes the looser
inputs :func:`~nanoagent.inference.engine.infer` accepts (a plain prompt string, a single
message dict, or a bare message list) into one.

A :class:`Response` is the normalized result of one request: the completion ``text``,
the model's ``reasoning`` (``<think>`` trace, when the backend separates it), any
``tool_calls`` the model requested, token ``usage``, computed ``cost``, the model's
``finish_reason``, and an ``error`` string that is set (with ``text`` left ``None``) when
the item failed after the backend exhausted its retries.

A :class:`ToolCall` is one model-requested tool invocation, normalized away from any
provider's wire shape — ``(id, name, arguments)``, so an agent harness's own tool-call type
maps onto it field-for-field through a thin adapter.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# One OpenAI-shape chat message, e.g. {"role": "user", "content": "hi"}.
Message = dict[str, Any]


# slots=True drops each instance's __dict__: smaller footprint and faster attribute access.
# A batch can produce thousands of these, and they only ever carry the fixed fields below.
@dataclass(slots=True)
class ToolCall:
    """One model-requested tool invocation, normalized away from any provider's shape.

    ``arguments`` is the raw JSON string the model emitted (decoded by the caller, not here).
    """

    id: str
    name: str
    arguments: str


@dataclass(slots=True)
class Request:
    """One chat-completion request: a list of OpenAI-shape messages, and optionally the tools offered.

    ``tools`` (OpenAI tool specs) rides the REQUEST rather than the config because it is what
    varies per item: one batch can offer a different toolset to each conversation, and the same
    endpoint serves plain completions and tool-calling turns. ``None`` (the default, and what
    every :meth:`coerce` shape produces) offers no tools, so the plain batch path is unchanged.
    Whatever the model asks for comes back in :attr:`Response.tool_calls`; running those calls and
    feeding the results back is a multi-turn loop, which is :func:`~nanoagent.inference.loop.run_tool_loop`.
    """

    messages: list[Message]
    tools: list[dict[str, Any]] | None = None

    @classmethod
    def coerce(cls, item: str | Message | list[Message] | Request) -> Request:
        """Normalize a prompt string / message / message list / :class:`Request` into a :class:`Request`.

        A ``str`` becomes a single ``user`` message; a single message ``dict`` is wrapped in a
        one-element list; a message list is wrapped as-is; an existing :class:`Request` is
        returned unchanged.
        """
        # list first: the dominant agentic input is a multi-turn message list (an agent rollout,
        # the public run([...]) API), so it dispatches in 1 isinstance check, not 3.
        if isinstance(item, list):
            return cls(messages=list(item))
        if isinstance(item, Request):
            return item
        if isinstance(item, str):
            return cls(messages=[{"role": "user", "content": item}])
        if isinstance(item, dict):  # a single {"role": ..., "content": ...} message
            return cls(messages=[item])
        return cls(messages=list(item))  # catch-all for other iterables (e.g. tuples)


@dataclass(slots=True)
class Response:
    """The normalized result of one request (``error`` set on failure, ``text`` then ``None``)."""

    text: str | None = None
    # the model's separated reasoning / ``<think>`` trace, when the backend pulls it out of the
    # answer (e.g. a reasoning-parser-enabled SGLang server, or config.parse_thinking); else None.
    reasoning: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: dict[str, int] = field(default_factory=dict)
    cost: float = 0.0
    # the model's stop reason for this completion ("stop", "length" = truncated at max_tokens,
    # "tool_calls", ...) when the backend reports it; lets a caller detect a cut-off answer.
    finish_reason: str | None = None
    error: str | None = None
