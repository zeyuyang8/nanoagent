"""Framework-independent lifecycle and event stream for web-hosted agent runs."""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from importlib.resources import files
from typing import Any, cast, Protocol, TYPE_CHECKING
from uuid import uuid4

from jsonschema import Draft202012Validator

from nanoagent.core.agent import Agent
from nanoagent.runtime.model import Model
from nanoagent.runtime.build import build_agent
from nanoagent.runtime.events import RunEvents
from nanoagent.web.config import WebConfig

if TYPE_CHECKING:
    from nanoagent.core.agent import AgentResult

logger = logging.getLogger(__name__)

_TERMINAL_TYPES = {"done", "error"}
_REQUEST_SCHEMA = json.loads(
    files("nanoagent.web.schema").joinpath("run-request.schema.json").read_text(encoding="utf-8")
)
_REQUEST_VALIDATOR = Draft202012Validator(_REQUEST_SCHEMA)


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
    errors = sorted(_REQUEST_VALIDATOR.iter_errors(value), key=lambda error: list(error.path))
    if errors:
        error = errors[0]
        location = ".".join(str(part) for part in error.path)
        raise ValidationError(f"{location + ': ' if location else ''}{error.message}")

    body = cast(dict[str, Any], value)
    input_text = cast(str, body["input"])
    if not input_text.strip():
        raise ValidationError("input must be a non-empty string")
    instructions = cast(str | None, body.get("instructions"))
    messages = [dict(message) for message in cast(list[dict[str, Any]], body.get("messages", []))]
    metadata = cast(dict[str, Any] | None, body.get("metadata"))
    return RunRequest(input=input_text, messages=messages, instructions=instructions, metadata=metadata)


class _ConfiguredAgentFactory:
    def __init__(self, cfg: WebConfig) -> None:
        self._cfg = cfg
        self._model = Model.from_config(cfg.model)

    def __call__(self, instructions: str | None) -> tuple[Agent, str]:
        clean_instructions = instructions.strip() if instructions else ""
        suffix = f"Application instructions:\n{clean_instructions}" if clean_instructions else None
        agent = build_agent(self._cfg, model=self._model, prompt_suffix=suffix)
        return agent, agent.system_prompt

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
        output_chars = 0
        terminal_emitted = False

        def emit(event_type: str, **fields: Any) -> None:
            queue.put_nowait({"type": event_type, "runId": run_id, **fields})

        def project(_label: str | None, **event: Any) -> None:
            nonlocal terminal_emitted
            event_type = event.pop("type")
            if event_type == "done" and event.get("error"):
                terminal_emitted = True
                detail = str(event["error"])
                code = "output_limit" if detail.startswith("OutputLimitError:") else "agent_error"
                emit("error", code=code, error=detail.partition(": ")[2] or detail)
                return
            if event_type in _TERMINAL_TYPES:
                terminal_emitted = True
            emit(event_type, **event)

        projected = RunEvents(project, None)

        def on_delta(kind: str, text: str) -> None:
            nonlocal output_chars
            if kind == "reasoning" and not self.cfg.include_reasoning:
                return
            if kind == "content":
                output_chars += len(text)
                if output_chars > self.cfg.max_output_chars:
                    raise OutputLimitError("agent output exceeded the configured character limit")
            projected.on_delta(kind, text)

        emit("start", metadata=request.metadata)
        try:
            async with self._semaphore:
                agent, system_prompt = self._factory(request.instructions)
                transcript = [
                    {"role": "system", "content": system_prompt},
                    *[dict(message) for message in request.messages],
                ]
                async with asyncio.timeout(self.cfg.request_timeout):
                    await agent.run(
                        request.input,
                        messages=transcript,
                        on_delta=on_delta,
                        on_step=projected.on_step,
                    )
        except TimeoutError:
            emit("error", code="timeout", error="Agent run timed out")
        except asyncio.CancelledError:
            emit("error", code="cancelled", error="Agent run was cancelled")
        except Exception:
            if not terminal_emitted:
                logger.exception("web agent run %s failed", run_id)
                emit("error", code="internal_error", error="Agent run failed")
        finally:
            self._active.pop(run_id, None)
            queue.put_nowait(None)
