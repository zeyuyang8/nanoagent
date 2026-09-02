"""Old import paths remain aliases while consumers migrate to the flatter package layout."""

import importlib

from nanoagent.core.agent import Agent
from nanoagent.core.tool import Tool
from nanoagent.runtime.config import RunConfig


def test_pre_03_harness_imports_reexport_canonical_objects() -> None:
    from nanoagent.harness.config import RunConfig as LegacyRunConfig
    from nanoagent.harness.core.agent import Agent as LegacyAgent
    from nanoagent.harness.core.tool import Tool as LegacyTool

    assert LegacyAgent is Agent
    assert LegacyTool is Tool
    assert LegacyRunConfig is RunConfig


def test_every_pre_03_module_path_remains_importable() -> None:
    modules = [
        "nanoagent.harness.config",
        *(
            f"nanoagent.harness.core.{name}"
            for name in ("agent", "events", "hooks", "model", "tool", "workspace")
        ),
        *(
            f"nanoagent.harness.run.{name}"
            for name in (
                "batch",
                "build",
                "cli",
                "log_capture",
                "mgen",
                "progress",
                "taskselect",
                "trajectory",
            )
        ),
        *(f"nanoagent.harness.tools.{name}" for name in ("bash", "code", "file", "skill", "write")),
        *(f"nanoagent.harness.repl.{name}" for name in ("app", "browser", "commands", "tree")),
    ]

    for module in modules:
        assert importlib.import_module(module).__name__ == module
