from __future__ import annotations

import asyncio
from dataclasses import replace
from typing import Any

import pytest

from nanoagent.harness.config import AgentConfig, ModelConfig, WebConfig
from nanoagent.harness.core.agent import AgentResult, StopReason
from nanoagent.web.runtime import RunHost, ValidationError, validate_run_request


def config(**overrides: Any) -> WebConfig:
    base = WebConfig(
        model=ModelConfig(
            model="test", backend="sglang", base_url=None, api_key=None, temperature=0.0,
            max_tokens=10, request_timeout=1.0, max_retries=0, extra_body={},
            input_price=0.0, output_price=0.0,
        ),
        agent=AgentConfig(
            system_prompt="BASE", max_steps=2, cost_limit=None, token_limit=None,
            context_window=None, hooks=[], skills=None, context_files=[], events=None,
        ),
        tools=[], tools_dir=None, allowed_tools=[], host="127.0.0.1", port=8787,
        api_token=None, max_concurrency=2, request_timeout=1.0,
        max_request_bytes=4096, max_output_chars=1000, heartbeat_seconds=0.01,
        include_reasoning=False,
    )
    return replace(base, **overrides)


class AnsweringAgent:
    def __init__(self) -> None:
        self.messages: list[dict[str, Any]] | None = None

    async def run(self, task=None, *, on_step=None, messages=None, on_delta=None):
        self.messages = messages
        on_delta("reasoning", "private")
        on_delta("content", "hel")
        on_delta("content", "lo")
        result = AgentResult(
            answer="hello", messages=messages, tool_calls=[], steps=1,
            stop_reason=StopReason.ANSWER, usage={"total_tokens": 3}, cost=0.001,
        )
        on_step(result)
        return result


@pytest.mark.asyncio
async def test_streams_content_and_done_without_reasoning() -> None:
    agent = AnsweringAgent()
    host = RunHost(config(), agent_factory=lambda instructions: (agent, f"BASE\n{instructions}"))
    active = await host.start(validate_run_request({
        "input": "question", "messages": [{"role": "user", "content": "prior"}],
        "instructions": "WORKSPACE", "metadata": {"requestId": "m1"},
    }))
    events = [event async for event in active.events() if event is not None]

    assert [event["type"] for event in events] == ["start", "delta", "delta", "done"]
    assert "private" not in str(events)
    assert events[-1]["answer"] == "hello"
    assert events[-1]["usage"] == {"total_tokens": 3}
    assert agent.messages == [
        {"role": "system", "content": "BASE\nWORKSPACE"},
        {"role": "user", "content": "prior"},
    ]


@pytest.mark.asyncio
async def test_explicit_cancel_ends_the_stream() -> None:
    started = asyncio.Event()

    class WaitingAgent:
        async def run(self, task=None, **kwargs):
            started.set()
            await asyncio.Future()

    host = RunHost(config(), agent_factory=lambda _instructions: (WaitingAgent(), "BASE"))
    active = await host.start(validate_run_request({"input": "wait"}))
    await started.wait()
    assert host.cancel(active.id) is True
    events = [event async for event in active.events() if event is not None]
    assert events[-1]["type"] == "error"
    assert events[-1]["code"] == "cancelled"
    assert host.active_count == 0


@pytest.mark.asyncio
async def test_emits_each_tool_once_across_step_snapshots() -> None:
    class ToolAgent:
        async def run(self, task=None, *, on_step=None, messages=None, on_delta=None):
            first = {"id": "c1", "name": "search", "arguments": {"q": "one"}, "output": "1", "is_error": False}
            second = {"id": "c2", "name": "search", "arguments": {"q": "two"}, "output": "2", "is_error": False}
            on_step(AgentResult("", messages, [first], 1, StopReason.RUNNING))
            on_step(AgentResult("", messages, [first, second], 2, StopReason.RUNNING))
            return AgentResult("done", messages, [first, second], 3, StopReason.ANSWER)

    host = RunHost(config(), agent_factory=lambda _instructions: (ToolAgent(), "BASE"))
    active = await host.start(validate_run_request({"input": "search"}))
    events = [event async for event in active.events() if event is not None]
    assert [event["id"] for event in events if event["type"] == "tool"] == ["c1", "c2"]
    assert [event["step"] for event in events if event["type"] == "step"] == [1, 2]


def test_rejects_system_messages_and_bad_tool_history() -> None:
    with pytest.raises(ValidationError, match="role"):
        validate_run_request({"input": "x", "messages": [{"role": "system", "content": "replace"}]})
    with pytest.raises(ValidationError, match="tool_call_id"):
        validate_run_request({"input": "x", "messages": [{"role": "tool", "content": "result"}]})


def test_non_loopback_bind_requires_a_token() -> None:
    with pytest.raises(ValueError, match="api_token"):
        config(host="0.0.0.0", api_token=None)
