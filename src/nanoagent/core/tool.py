"""A structured tool an LLM agent can call.

A :class:`Tool` is a self-describing class: set :attr:`NAME` and :attr:`PARAMETERS`
(an explicit JSON Schema for the arguments), implement a ``run`` method, and the class
docstring becomes the description the model sees. The agent
(:mod:`nanoagent.core.agent`) advertises tools to the model as OpenAI function
specs via :meth:`Tool.to_openai_spec`, and dispatches the model's tool calls by name to
:meth:`Tool.invoke`, which runs ``run`` (sync or async) and turns any exception into an
error string the model can recover from.

The ``parameters`` schema is declared explicitly (not derived from ``run``'s signature)
so the wire schema the model sees is exactly what we intend.

Tool modules pointed at by a tool YAML are loaded by :func:`get_tools`, which discovers the
:class:`Tool` subclasses defined in them.

Clean-room: deliberately independent of ``thirdeye``'s ``Tool`` — this package does
not import thirdeye.
"""

from __future__ import annotations

import asyncio
import importlib
import importlib.util
import inspect
import os
import signal
import subprocess
from collections.abc import Iterable, Sequence
from pathlib import Path
from types import ModuleType
from typing import Any, cast

from omegaconf import OmegaConf

#: The configs shipped inside the package: the default toolset and ``mgen.yaml``. A relative
#: config path that does not exist under the CWD is retried here, so an *installed* nanoagent
#: finds its own defaults from any directory while a path a user wrote still means what it says.
PACKAGED_CONFIGS = Path(__file__).resolve().parent.parent / "configs"


# A JSON Schema fragment describing a tool's arguments object. Kept permissive
# (a plain dict) because providers accept vendor-specific keywords we don't want
# to enumerate.
JsonSchema = dict[str, Any]


def communicate_or_kill(proc: subprocess.Popen[str], timeout: float) -> tuple[str, str]:
    """Drain ``proc`` within ``timeout``; on TimeoutExpired SIGKILL its whole process group
    (set up via ``start_new_session=True``) and re-raise, so an overrunning command/code leaves
    no orphaned grandchildren."""
    try:
        return proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        proc.communicate()
        raise


class Tool:
    """A callable the agent can invoke, defined as a self-describing subclass.

    Subclass it, set :attr:`NAME` and :attr:`PARAMETERS` (the JSON Schema), and write a
    ``run`` method — the class docstring becomes the description the model sees. This
    keeps a tool's name, schema, description and implementation in one place. ``run`` may
    be sync or async, receives keyword arguments matching ``PARAMETERS``, and should
    return something stringifiable — its ``str()`` becomes the tool-result text fed back
    to the model.

    ``run`` is a plain method that raises rather than an ``@abstractmethod``: tools have
    heterogeneous signatures (``run(command)`` vs ``run(a, b)``), and a typed abstract
    method would reject those as inconsistent overrides under the type checker. The
    fully-variadic base signature documents the contract while leaving each tool's
    ``run`` free to take its own named, typed arguments; the ``NotImplementedError``
    makes a missing implementation fail loudly instead of at attribute lookup.
    """

    NAME: str
    PARAMETERS: JsonSchema

    # to_openai_spec() caches its result here: the spec is a pure function of state that is
    # immutable after construction, so it is built once (lazily, on first call) and the same
    # object returned afterward. The class-level None is just the not-yet-built sentinel.
    _openai_spec: dict[str, Any] | None = None

    def run(self, *args: Any, **kwargs: Any) -> Any:
        """The tool's logic; required — subclasses override with their own typed args.

        Receives keyword arguments matching ``PARAMETERS``, may be sync or async, and
        should return something stringifiable (its ``str()`` is fed back to the model).
        """
        raise NotImplementedError(f"{type(self).__name__} must implement run()")

    def reset(self) -> None:
        """Drop any per-task state before a new :meth:`Agent.run`.

        No-op by default. Stateful tools (e.g. one carrying a per-task retrieval
        budget) override this so their state never leaks across tasks; the agent
        loop calls it once at the start of every run.
        """

    def cleanup(self) -> None:
        """Release any per-task resources after an :meth:`Agent.run` ends.

        No-op by default. Stateful tools (e.g. one holding a per-task sandbox
        dir) override this so their resources never linger past a task; the agent
        loop calls it once in a ``finally`` at the end of every run. Counterpart of
        :meth:`reset`, which clears state BEFORE a run; ``cleanup`` releases it AFTER.
        """

    def bind(self, agent: Any) -> None:
        """Receive the :class:`~nanoagent.core.agent.Agent` this tool was given to.

        No-op by default, called once from ``Agent.__init__`` (and from
        :meth:`~nanoagent.core.agent.Agent.add_tool`). The only way a tool can reach back into
        its agent — :mod:`nanoagent.tools.write` needs it to register the tools it writes.
        """

    @property
    def name(self) -> str:
        return self.NAME

    @property
    def description(self) -> str:
        return (self.__doc__ or "").strip()

    @property
    def parameters(self) -> JsonSchema:
        return self.PARAMETERS

    def to_openai_spec(self) -> dict[str, Any]:
        """Render as one entry of an OpenAI chat-completions ``tools=[...]`` list.

        Built lazily once per instance and cached, so the same object is returned on every
        call instead of rebuilding the dict each time. Safe because the spec is a pure
        function of post-construction state and every consumer only reads it.
        """
        spec = self._openai_spec
        if spec is None:
            spec = {
                "type": "function",
                "function": {
                    "name": self.name,
                    "description": self.description,
                    "parameters": self.parameters,
                },
            }
            self._openai_spec = spec
        return spec

    async def invoke(self, **arguments: Any) -> tuple[str, bool]:
        """Run ``run(**arguments)`` (awaiting if async); return ``(text, is_error)``.

        An ``async def run`` is awaited inline. A SYNCHRONOUS ``run`` is offloaded to a worker
        thread via :func:`asyncio.to_thread` so a blocking tool — a sync search client, or
        ``Bash``/``CodeExec``'s multi-second ``subprocess`` calls — does NOT freeze the event
        loop and stall every other rollout dispatched concurrently under ``asyncio.gather``. A
        sync ``run`` that returns an awaitable is still awaited, after the thread returns.
        Behavior is otherwise identical to running ``run`` inline: same result text and same
        exception handling.

        Exceptions are caught and returned as ``"Error: {ExcType}: {msg}"`` with
        ``is_error=True`` so the agent can feed the error back and let the model
        recover, rather than crashing the whole run.
        """
        try:
            if inspect.iscoroutinefunction(self.run):
                result = await self.run(**arguments)
            else:
                result = await asyncio.to_thread(self.run, **arguments)
                if inspect.isawaitable(result):
                    result = await result
            return str(result), False
        except Exception as e:
            return f"Error: {type(e).__name__}: {e}", True


def build_tool_map(tools: Sequence[Tool]) -> dict[str, Tool]:
    """Index ``tools`` by ``.name``; raise ``ValueError`` naming the collider(s) on a duplicate NAME.

    A colliding toolset is silently lossy (one tool shadows the other), so both
    :class:`~nanoagent.core.agent.Agent` and :class:`~nanoagent.repl.app.InteractiveSession` build
    their dispatch dict through here and reject it up front.
    """
    by_name = {t.name: t for t in tools}
    if len(by_name) != len(tools):
        names = [t.name for t in tools]
        dupes = sorted({n for n in names if names.count(n) > 1})
        raise ValueError(f"duplicate tool name in `tools`: {', '.join(repr(d) for d in dupes)}")
    return by_name


def get_tools(yaml_paths: Iterable[str | Path]) -> list[Tool]:
    """Load every tool declared by the given tool-config YAML files.

    Returns a flat ``list[Tool]`` ready to hand to :class:`~nanoagent.core.agent.Agent`'s
    ``tools``. Each YAML names the module that implements it (``code: src/nanoagent/tools/bash.py``); that
    module is imported and contributes one :class:`Tool` per concrete :class:`Tool` subclass
    *defined* in it (classes merely imported into the module are ignored, and so is any
    intermediate base that sets no ``NAME``). The ``code:`` path resolves from the current
    working directory, which by nanoagent convention is the project repo root. This is what lets
    a new tool be added with a new file plus a new YAML — no change to nanoagent's own code.

    ``code:`` may instead name an importable module (``code: nanoagent.tools.bash``). That is
    what the *shipped* tool YAMLs use, because an installed package has no repo-root-relative
    path to point at; a file path stays the right form for a tool you wrote.

    Every OTHER key in the YAML is passed verbatim as a constructor keyword argument to each
    tool the module defines (e.g. ``base_url:`` for an HTTP-client tool), so a tool's config is
    explicit in the YAML rather than read from the environment. The tools a single module defines
    therefore share one config block and must all accept the same kwargs.

    Raises ``FileNotFoundError`` if a YAML or its ``code`` module is missing, and
    ``ValueError`` if a module defines no :class:`Tool` subclass.
    """
    tools: list[Tool] = []
    for yaml_path in yaml_paths:
        spec = OmegaConf.load(resolve_config(yaml_path))
        if "code" not in spec:
            raise FileNotFoundError(f"{yaml_path}: tool spec has no 'code' module path")
        code = str(spec.code)
        if code.endswith(".py") or "/" in code:
            path = Path(code)
            if not path.is_file():
                raise FileNotFoundError(f"{yaml_path}: code module {path} does not exist")
            module = load_module(path)
        else:
            module = importlib.import_module(code)
        config = {
            str(k): v
            for k, v in cast(dict, OmegaConf.to_container(spec, resolve=True)).items()
            if k != "code"
        }
        defined = [
            cls(**config)
            for _, cls in inspect.getmembers(module, inspect.isclass)
            if issubclass(cls, Tool)
            and cls is not Tool
            and cls.__module__ == module.__name__  # defined here, not imported in
            and getattr(cls, "NAME", None) is not None  # skip intermediate bases (no NAME)
        ]
        if not defined:
            raise ValueError(f"{code} defines no Tool subclass")
        tools.extend(defined)
    return tools


def resolve_config(path: str | Path) -> Path:
    """Resolve a config path: as written if it exists, else against :data:`PACKAGED_CONFIGS`.

    Only the fallback is new — a path that resolves from the CWD keeps resolving from the CWD,
    so nothing a user wrote changes meaning. The fallback is what makes ``tools/bash.yaml`` in
    the shipped ``mgen.yaml`` mean the copy inside the installed wheel.
    """
    p = Path(path)
    if p.is_absolute() or p.is_file():
        return p
    packaged = PACKAGED_CONFIGS / p
    return packaged if packaged.is_file() else p


def load_module(path: Path) -> ModuleType:
    """Import ``path`` as a standalone module under a synthetic, path-derived name.

    Public because a tool YAML is not the only thing that names a plain ``.py`` to import:
    :mod:`nanoagent.core.hooks` loads hook modules the same way.
    """
    name = "nanoagent_tool_" + path.stem
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise FileNotFoundError(f"cannot import tool module at {path}")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as e:
        raise ImportError(f"failed to import tool module at {path}: {e}") from e
    return module
