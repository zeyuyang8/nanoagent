"""Framework-independent lifecycle and event stream for web-hosted agent runs."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from typing import Any, Protocol
from uuid import uuid4

from nanoagent.harness.config import WebConfig
from nanoagent.harness.core.agent import Agent, AgentResult, StopReason
from nanoagent.harness.core.hooks import get_hooks
from nanoagent.harness.core.model import Model
from nanoagent.harness.run.build import build_prompt_and_tools

logger = logging.getLogger(__name__)

_ROLES = {"user", "assistant", "tool"}
_TERMINAL_TYPES = {"done", "error"}


class ValidationError(ValueError):
    """A malformed public run request."""


class OutputLimitError(RuntimeError):
    """The streamed visible answer crossed the operator-owned response bound."""


@dataclass(frozen=True)
class RunRequest:
    """The request-controlled part of a run; model and tools stay in :class:`WebConfig`."""

    input: str
    messages: list[dict[str, Any]]
    instructions: str | None = None
    metadata: dict[str, Any] | None = None


class AgentLike(Protocol):
    async def run(
        self,
        task: str | None = None,
        *,
        on_step: Callable[[AgentResult], None] | None = None,
        messages: list[dict[str, Any]] | None = None,
        on_delta: Callable[[str, str], None] | None = None,
    ) -> AgentResult: ...


class AgentFactory(Protocol):
    def __call__(self, instructions: str | None) -> tuple[AgentLike, str]: ...


def validate_run_request(value: Any) -> RunRequest:
    """Validate the deliberately small JSON request surface.

    System messages are rejected in history. Callers use ``instructions`` instead, which is
    appended to the operator-owned base prompt and cannot replace its safety policy.
    """
    if not isinstance(value, dict):
        raise ValidationError("request body must be a JSON object")
    input_text = value.get("input")
    if not isinstance(input_text, str) or not input_text.strip():
        raise ValidationError("input must be a non-empty string")
    if len(input_text) > 100_000:
        raise ValidationError("input must be 100000 characters or fewer")

    instructions = value.get("instructions")
    if instructions is not None and not isinstance(instructions, str):
        raise ValidationError("instructions must be a string")
    if isinstance(instructions, str) and len(instructions) > 20_000:
        raise ValidationError("instructions must be 20000 characters or fewer")

    raw_messages = value.get("messages", [])
    if not isinstance(raw_messages, list):
        raise ValidationError("messages must be an array")
    if len(raw_messages) > 200:
        raise ValidationError("messages must contain 200 entries or fewer")
    messages: list[dict[str, Any]] = []
    for index, raw in enumerate(raw_messages):
        if not isinstance(raw, dict):
            raise ValidationError(f"messages[{index}] must be an object")
        role = raw.get("role")
        content = raw.get("content")
        if role not in _ROLES:
            raise ValidationError(f"messages[{index}].role must be user, assistant, or tool")
        if not isinstance(content, str):
            raise ValidationError(f"messages[{index}].content must be a string")
        if len(content) > 100_000:
            raise ValidationError(f"messages[{index}].content must be 100000 characters or fewer")
        message = {"role": role, "content": content}
        if role == "tool":
            tool_call_id = raw.get("tool_call_id")
            if not isinstance(tool_call_id, str) or not tool_call_id:
                raise ValidationError(f"messages[{index}].tool_call_id is required for tool messages")
            message["tool_call_id"] = tool_call_id
        messages.append(message)

    metadata = value.get("metadata")
    if metadata is not None and not isinstance(metadata, dict):
        raise ValidationError("metadata must be an object")
    return RunRequest(input=input_text, messages=messages, instructions=instructions, metadata=metadata)


class _ConfiguredAgentFactory:
    def __init__(self, cfg: WebConfig) -> None:
        self._cfg = cfg
        self._model = Model.from_config(cfg.model)

    def __call__(self, instructions: str | None) -> tuple[Agent, str]:
        prompt, tools = build_prompt_and_tools(
            self._cfg.agent,
            self._cfg.tools,
            self._cfg.tools_dir,
            self._cfg.allowed_tools,
        )
        if instructions and instructions.strip():
            prompt = f"{prompt.rstrip()}\n\nApplication instructions:\n{instructions.strip()}"
        return Agent(
            self._model,
            tools,
            system_prompt=prompt,
            max_steps=self._cfg.agent.max_steps,
            cost_limit=self._cfg.agent.cost_limit,
            token_limit=self._cfg.agent.token_limit,
            context_window=self._cfg.agent.context_window,
            hooks=get_hooks(self._cfg.agent.hooks),
        ), prompt

    async def aclose(self) -> None:
        await self._model.aclose()


@dataclass
class ActiveRun:
    id: str
    _queue: asyncio.Queue[dict[str, Any] | None]
    _task: asyncio.Task[None]
    _heartbeat_seconds: float

    async def events(self) -> AsyncIterator[dict[str, Any] | None]:
        """Yield events; ``None`` is a heartbeat rather than end-of-stream."""
        try:
            while True:
                try:
                    event = await asyncio.wait_for(
                        self._queue.get(), timeout=self._heartbeat_seconds
                    )
                except TimeoutError:
                    yield None
                    continue
                if event is None:
                    return
                yield event
                if event.get("type") in _TERMINAL_TYPES:
                    return
        finally:
            if not self._task.done():
                self._task.cancel()


class RunHost:
    """Own active runs, concurrency, cancellation and one long-lived model connection pool."""

    def __init__(self, cfg: WebConfig, *, agent_factory: AgentFactory | None = None) -> None:
        self.cfg = cfg
        self._configured_factory = None if agent_factory is not None else _ConfiguredAgentFactory(cfg)
        self._factory = agent_factory or self._configured_factory
        self._semaphore = asyncio.Semaphore(cfg.max_concurrency)
        self._active: dict[str, asyncio.Task[None]] = {}

    @property
    def active_count(self) -> int:
        return len(self._active)

    async def start(self, request: RunRequest) -> ActiveRun:
        run_id = str(uuid4())
        queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()
        task = asyncio.create_task(self._run(run_id, request, queue), name=f"nanoagent-run-{run_id}")
        self._active[run_id] = task
        return ActiveRun(run_id, queue, task, self.cfg.heartbeat_seconds)

    def cancel(self, run_id: str) -> bool:
        task = self._active.get(run_id)
        if task is None or task.done():
            return False
        task.cancel()
        return True

    async def aclose(self) -> None:
        tasks = list(self._active.values())
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        if self._configured_factory is not None:
            await self._configured_factory.aclose()

    async def _run(
        self,
        run_id: str,
        request: RunRequest,
        queue: asyncio.Queue[dict[str, Any] | None],
    ) -> None:
        tool_cursor = 0
        output_chars = 0

        def emit(event_type: str, **fields: Any) -> None:
            queue.put_nowait({"type": event_type, "runId": run_id, **fields})

        def on_delta(kind: str, text: str) -> None:
            nonlocal output_chars
            if kind == "reasoning" and not self.cfg.include_reasoning:
                return
            if kind == "content":
                output_chars += len(text)
                if output_chars > self.cfg.max_output_chars:
                    raise OutputLimitError("agent output exceeded the configured character limit")
            emit("delta", kind=kind, text=text)

        def on_step(result: AgentResult) -> None:
            nonlocal tool_cursor
            for tool in result.tool_calls[tool_cursor:]:
                emit("tool", **tool)
            tool_cursor = len(result.tool_calls)
            if result.stop_reason is StopReason.RUNNING:
                emit("step", step=result.steps, usage=result.usage, cost=result.cost)

        emit("start", metadata=request.metadata)
        try:
            async with self._semaphore:
                agent, system_prompt = self._factory(request.instructions)
                transcript = [
                    {"role": "system", "content": system_prompt},
                    *[dict(message) for message in request.messages],
                ]
                async with asyncio.timeout(self.cfg.request_timeout):
                    result = await agent.run(
                        request.input,
                        messages=transcript,
                        on_delta=on_delta,
                        on_step=on_step,
                    )
            emit(
                "done",
                answer=result.answer,
                stopReason=result.stop_reason.value,
                steps=result.steps,
                usage=result.usage,
                cost=result.cost,
                error=result.error,
            )
        except TimeoutError:
            emit("error", code="timeout", error="Agent run timed out")
        except OutputLimitError:
            emit("error", code="output_limit", error="Agent output exceeded its limit")
        except asyncio.CancelledError:
            emit("error", code="cancelled", error="Agent run was cancelled")
        except Exception:
            logger.exception("web agent run %s failed", run_id)
            emit("error", code="internal_error", error="Agent run failed")
        finally:
            self._active.pop(run_id, None)
            queue.put_nowait(None)
