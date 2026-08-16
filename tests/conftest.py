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
from pathlib import Path

import pytest

#: Environment variable that enables smoke tests without a CLI flag.
SMOKE_ENV_VAR = "COMPARATIVA_SMOKE"

#: Environment variable used to override corpus resolution entirely (CI, or a
#: developer machine with the corpus checked out somewhere unusual).
CORPUS_ROOT_ENV_VAR = "COMPARATIVA_CORPUS_ROOT"

REPO_ROOT = Path(__file__).resolve().parents[1]


def resolve_corpus_root() -> Path:
    """Single resolution point for "where is the mission corpus".

    OPERATION BATTLING BARDS benchmarks against the granville corpus. That live
    tree is mutated by other sessions outside this mission's control, so the
    supervisor extracted a verified, mission-consistent snapshot into
    ``corpus/frozen/`` (see ``docs/CORPUS_PIN.md``). Resolution order:

    1. ``$COMPARATIVA_CORPUS_ROOT``, if set — explicit override.
    2. ``<repo>/corpus/frozen``, if it exists — the pinned snapshot.
    3. ``~/Projects/podcasts/granville`` — last-resort fallback for a machine
       without the frozen snapshot (tests then skip via each file's
       ``requires_corpus``-style marker if that path is also absent).
    """
    override = os.environ.get(CORPUS_ROOT_ENV_VAR)
    if override:
        return Path(override).expanduser().resolve()

    frozen = REPO_ROOT / "corpus" / "frozen"
    if frozen.is_dir():
        return frozen

    return Path("~/Projects/podcasts/granville").expanduser()


#: The resolved corpus root, computed once at collection time.
CORPUS_ROOT = resolve_corpus_root()


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
