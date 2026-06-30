"""parsing_data.py — JSON parsing helpers for LLM outputs."""
from __future__ import annotations

import logging
from typing import Any

import json_repair

logger = logging.getLogger(__name__)


def parse_json(text: str) -> dict[str, Any] | None:
    """handle common LLM output quirks (trailing commas,
    missing quotes, markdown fences, etc.).
    """
    if not text or not text.strip():
        return None
    try:
        return json_repair.loads(text)
    except Exception as exc:  # noqa: BLE001
        logger.warning("JSON parse failed: %s", exc)
        return None
