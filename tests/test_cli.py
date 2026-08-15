"""Sortie 1 scaffold tests: the CLI loads and exposes all six subcommands."""

from __future__ import annotations

import subprocess
import sys

import pytest

from comparativa.cli import SUBCOMMAND_NAMES, build_parser, main

EXPECTED = ("parse", "voices", "generate", "bench", "listen", "report")


def test_subcommand_names_are_the_expected_six() -> None:
    assert SUBCOMMAND_NAMES == EXPECTED


def test_parser_registers_all_subcommands() -> None:
    parser = build_parser()
    actions = [
        a
        for a in parser._subparsers._group_actions  # type: ignore[union-attr]
        if hasattr(a, "choices")
    ]
    assert actions, "no subparser action registered"
    assert set(actions[0].choices) == set(EXPECTED)


def test_top_level_help_lists_every_subcommand(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    for name in EXPECTED:
        assert name in out


def test_no_args_prints_help_and_exits_zero(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main([]) == 0
    out = capsys.readouterr().out
    for name in EXPECTED:
        assert name in out


@pytest.mark.parametrize("name", EXPECTED)
def test_each_subcommand_stub_prints_help_and_exits_zero(
    name: str, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main([name]) == 0
    out = capsys.readouterr().out
    assert f"comparativa {name}" in out


def test_console_script_entry_point_runs() -> None:
    """The installed `comparativa` entry point loads and lists all six commands."""
    result = subprocess.run(
        [sys.executable, "-m", "comparativa", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    for name in EXPECTED:
        assert name in result.stdout
