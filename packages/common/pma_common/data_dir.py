"""Local data directory for the tournament agent.

Honors ``OPTIONS_TOURNAMENT_HOME`` when it points at an existing directory.
``VIBE_TRADING_HOME`` is accepted as a compatibility alias only. Default is
``~/.options-tournament``. There is no host/container volume path here.
"""

from __future__ import annotations

import os
from pathlib import Path


def get_data_dir() -> Path:
    default = Path.home() / ".options-tournament"
    for key in ("OPTIONS_TOURNAMENT_HOME", "VIBE_TRADING_HOME"):
        override = os.getenv(key, "").strip()
        if override:
            candidate = Path(override).expanduser()
            if candidate.is_dir():
                return candidate
    return default
