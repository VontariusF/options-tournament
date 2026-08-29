"""Minimal tool protocol for the tournament agent loop."""

from __future__ import annotations

from typing import Any, Dict


class BaseTool:
    name: str = ""
    description: str = ""
    parameters: Dict[str, Any] = {}

    def execute(self, **kwargs: Any) -> str:
        raise NotImplementedError
