"""Tests for delegating root CLI commands to ``oasis_analysis``."""

from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace

from oasis_radiomics import cli


def test_analysis_parser_preserves_nested_arguments() -> None:
    args = cli.build_parser().parse_args(
        [
            "analysis",
            "run",
            "demographic_baseline",
            "--",
            "--n-splits",
            "3",
        ]
    )

    assert args.command == "analysis"
    assert args.analysis_args == [
        "run",
        "demographic_baseline",
        "--",
        "--n-splits",
        "3",
    ]


def test_analysis_python_prefers_folder_virtual_environment(tmp_path: Path) -> None:
    interpreter = tmp_path / ".venv" / "bin" / "python"
    interpreter.parent.mkdir(parents=True)
    interpreter.touch()

    assert cli._analysis_python(tmp_path) == interpreter


def test_analysis_handler_forwards_command_and_exit_code(
    monkeypatch, tmp_path: Path
) -> None:
    analysis_cli = tmp_path / "cli.py"
    analysis_cli.touch()
    interpreter = Path("/analysis/python")
    invocation: dict[str, object] = {}

    def fake_run(command, *, cwd, check):
        invocation.update(command=command, cwd=cwd, check=check)
        return SimpleNamespace(returncode=7)

    monkeypatch.setattr(cli, "ANALYSIS_ROOT", tmp_path)
    monkeypatch.setattr(cli, "_analysis_python", lambda: interpreter)
    monkeypatch.setattr(cli.subprocess, "run", fake_run)
    args = argparse.Namespace(
        analysis_args=["run", "demographic_baseline", "--", "--n-splits", "3"]
    )

    assert cli._handle_analysis(args) == 7
    assert invocation == {
        "command": [
            str(interpreter),
            str(analysis_cli),
            "run",
            "demographic_baseline",
            "--",
            "--n-splits",
            "3",
        ],
        "cwd": tmp_path,
        "check": False,
    }


def test_analysis_handler_accepts_separator_before_forwarded_command(
    monkeypatch, tmp_path: Path
) -> None:
    analysis_cli = tmp_path / "cli.py"
    analysis_cli.touch()
    commands: list[list[str]] = []

    def fake_run(command, *, cwd, check):
        commands.append(command)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(cli, "ANALYSIS_ROOT", tmp_path)
    monkeypatch.setattr(cli, "_analysis_python", lambda: Path("python"))
    monkeypatch.setattr(cli.subprocess, "run", fake_run)

    result = cli._handle_analysis(argparse.Namespace(analysis_args=["--", "list"]))

    assert result == 0
    assert commands == [["python", str(analysis_cli), "list"]]


def test_main_does_not_load_radiomics_config_for_analysis(monkeypatch) -> None:
    monkeypatch.setattr(cli, "_handle_analysis", lambda args: 11)
    monkeypatch.setattr(
        cli.PipelineConfig,
        "load",
        lambda path: (_ for _ in ()).throw(AssertionError("config must not load")),
    )

    assert cli.main(["analysis", "list"]) == 11
