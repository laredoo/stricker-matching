"""Logging helpers for the model service and CLI."""

from __future__ import annotations

import logging
import os
from typing import Mapping

_DEFAULT_LEVEL = "INFO"
_LEVELS: Mapping[str, int] = {
    "CRITICAL": logging.CRITICAL,
    "ERROR": logging.ERROR,
    "WARNING": logging.WARNING,
    "INFO": logging.INFO,
    "DEBUG": logging.DEBUG,
}


def configure_logging(level: str | int | None = None) -> None:
    resolved_level = _resolve_level(level)
    logging.basicConfig(
        level=resolved_level,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )


def get_logger(name: str | None = None) -> logging.Logger:
    return logging.getLogger(name)


def _resolve_level(level: str | int | None) -> int:
    if level is None:
        level = os.getenv("STRICKER_LOG_LEVEL", _DEFAULT_LEVEL)
    if isinstance(level, int):
        return level
    return _LEVELS.get(level.upper(), _LEVELS[_DEFAULT_LEVEL])
