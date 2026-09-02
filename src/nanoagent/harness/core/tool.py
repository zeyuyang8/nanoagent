"""Compatibility module; use :mod:`nanoagent.core.tool`."""

from nanoagent.core.tool import *  # noqa: F403
from nanoagent.extensions import get_tools, load_module, PACKAGED_CONFIGS, resolve_config  # noqa: F401
