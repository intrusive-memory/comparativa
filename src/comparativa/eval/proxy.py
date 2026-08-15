"""Objective proxy: STT WER round-trip via ``mlx_whisper`` directly (FR-13, optional).

The installed mlx-audio STT path is broken (EXECUTION_PLAN.md Risk §8.4), so
the only candidate is ``mlx_whisper`` imported directly. Per the supervisor's
binding ruling for this sortie, comparativa does **not** add the dependency
itself — Sortie 1's pinned ``pyproject.toml``/``uv.lock`` are the supervising
agent's alone to change. If ``mlx_whisper`` is not importable in the pinned
env, the proxy is recorded as dropped, with the reason, rather than attempted.
"""

from __future__ import annotations

import importlib.util

#: Exact reason string the supervisor specified for the dropped path.
DROPPED_REASON = "mlx-whisper not installed in the pinned env"


def mlx_whisper_available() -> bool:
    """Whether ``mlx_whisper`` can be imported without adding a dependency."""
    return importlib.util.find_spec("mlx_whisper") is not None


def objective_proxy_status() -> tuple[bool, str]:
    """``(available, status-line)`` for the STT WER round-trip proxy.

    The status line is written verbatim into the report skeleton as
    ``objective proxy: <status-line>``.
    """
    if mlx_whisper_available():
        return True, "included — mlx-whisper is importable in the pinned env."
    return False, f"dropped — {DROPPED_REASON}"


__all__ = ["DROPPED_REASON", "mlx_whisper_available", "objective_proxy_status"]
