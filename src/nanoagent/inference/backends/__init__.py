"""Backend factory: select a transport by name with deferred imports.

:func:`build_backend` maps ``config.backend`` to a concrete backend and calls its
``from_config``. The lookup is :func:`nanoagent.inference.plugins.load_backend_class`: a built-in module
in this package if there is one, else a ``<name>.py`` in a plugin directory (see
:mod:`nanoagent.inference.plugins` for why a site's own transport lives outside ``src/``). Either way the
module is imported only when its name is selected — so importing this package never pulls in
the ``openai`` SDK until the sglang path is actually chosen.
"""

from __future__ import annotations

from nanoagent.inference.backend import Backend
from nanoagent.inference.config import LeanInferConfig
from nanoagent.inference.plugins import load_backend_class


def build_backend(config: LeanInferConfig) -> Backend:
    """Build the backend named by ``config.backend``, searching ``config.plugin_dirs`` for plugins."""
    cls = load_backend_class(config.backend, config.plugin_dirs)
    return cls.from_config(config)
