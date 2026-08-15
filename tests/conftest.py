"""Shared pytest configuration.

The ``smoke`` marker is registered here rather than in ``pyproject.toml`` so the
dependency-owning supervising agent stays the only party that edits the project
file. Smoke tests load real MLX checkpoints and take minutes, so they are
**skipped by default** and the fast suite stays fast::

    uv run pytest                          # unit suite only
    uv run pytest --smoke                  # everything, including engine smoke
    COMPARATIVA_SMOKE=1 uv run pytest      # same, via the environment
    uv run pytest -m smoke --smoke         # only the smoke tests
"""

from __future__ import annotations

import os

import pytest

#: Environment variable that enables smoke tests without a CLI flag.
SMOKE_ENV_VAR = "COMPARATIVA_SMOKE"


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--smoke",
        action="store_true",
        default=False,
        help="Run smoke tests that load real MLX checkpoints (slow, minutes).",
    )


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "smoke: loads a real MLX checkpoint and generates audio; skipped unless "
        "--smoke or COMPARATIVA_SMOKE=1.",
    )


def smoke_enabled(config: pytest.Config) -> bool:
    return bool(config.getoption("--smoke")) or os.environ.get(SMOKE_ENV_VAR) == "1"


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    if smoke_enabled(config):
        return
    skip = pytest.mark.skip(
        reason=f"smoke test: run with --smoke or {SMOKE_ENV_VAR}=1"
    )
    for item in items:
        if "smoke" in item.keywords:
            item.add_marker(skip)
