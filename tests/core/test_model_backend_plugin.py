"""``ModelConfig.backend`` reaches leaninfer's resolver verbatim — plugin names included.

nanoagent used to gate :meth:`nanoagent.core.model.Model.from_config` behind its own allowlist
(``("sglang", "openai")``). That was wrong twice over once leaninfer became its own package: it
still named ``openai``, which the published leaninfer no longer resolves, and it rejected every
*plugin* backend — so the whole ``$LEANINFER_PLUGINS`` seam that
``src/nanoagent/configs/mgen.yaml`` documents failed before leaninfer was ever consulted.

The rule this pins is that nanoagent keeps no list of backend names. leaninfer owns resolution
(built-ins, then ``$LEANINFER_PLUGINS``), and it owns the rejection too — its message names the
backends it did find and the directories it searched, which an allowlist here cannot.

The plugin is written into ``tmp_path`` rather than reusing a real one: what is under test is the
pass-through, not any particular gateway, and a test that needs a plugin already on disk is a test
that fails outside one checkout.

What it consumes: no network and no server — building a backend only constructs a client.

Run (from the repo root)::

    python3 -m pytest tests/core/test_model_backend_plugin.py -x -q
"""

from __future__ import annotations

from pathlib import Path

import pytest
from nanoagent.config import ModelConfig
from nanoagent.core.model import Model

_PLUGIN = '''
from leaninfer.backends.sglang import SglangBackend


class HouseBackend(SglangBackend):
    """A plugin backend: nothing but a name leaninfer's built-ins do not have."""


BACKEND = HouseBackend
'''


def _cfg(backend: str) -> ModelConfig:
    return ModelConfig(
        model="m",
        backend=backend,
        base_url="http://127.0.0.1:1/v1",
        api_key="k",
        temperature=None,
        max_tokens=16,
        request_timeout=30.0,
        max_retries=0,
        extra_body={},
        input_price=0.0,
        output_price=0.0,
    )


def test_plugin_backend_name_is_passed_through(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / "house.py").write_text(_PLUGIN)
    monkeypatch.setenv("LEANINFER_PLUGINS", str(tmp_path))
    model = Model.from_config(_cfg("house"))
    assert type(model._backend).__name__ == "HouseBackend"


def test_unknown_backend_is_rejected_by_leaninfer(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / "house.py").write_text(_PLUGIN)
    monkeypatch.setenv("LEANINFER_PLUGINS", str(tmp_path))
    # A ValueError either way, so a caller that handled the old allowlist error still works —
    # but this one can list `house`, which is exactly what an allowlist here could not do.
    with pytest.raises(ValueError, match="house") as excinfo:
        Model.from_config(_cfg("nosuchbackend"))
    assert "nosuchbackend" in str(excinfo.value)
