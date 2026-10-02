#!/usr/bin/env python3
"""Command-line interface for discovering, running, and cleaning project scripts."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
NON_SCRIPT_MODULES = {"cli.py", "test_cli.py"}


def discover_scripts(project_root: Path = PROJECT_ROOT) -> dict[str, Path]:
    """Return runnable top-level Python scripts, keyed by filename stem."""
    root = project_root.resolve()
    scripts: dict[str, Path] = {}
    for path in sorted(root.glob("*.py")):
        resolved = path.resolve()
        if path.name in NON_SCRIPT_MODULES or not path.is_file():
            continue
        if resolved.parent != root:
            continue
        scripts[path.stem] = resolved
    return scripts


def normalize_script_name(name: str) -> str:
    """Accept either ``name`` or ``name.py`` and return the script stem."""
    candidate = Path(name)
    if candidate.name != name or candidate.suffix not in {"", ".py"}:
        raise ValueError(f"Invalid script name: {name!r}")
    stem = candidate.stem if candidate.suffix == ".py" else candidate.name
    if not stem:
        raise ValueError("The script name cannot be empty.")
    return stem


def resolve_script(name: str, project_root: Path = PROJECT_ROOT) -> tuple[str, Path]:
    """Resolve a user-supplied name to a known project script."""
    stem = normalize_script_name(name)
    scripts = discover_scripts(project_root)
    try:
        return stem, scripts[stem]
    except KeyError as error:
        available = ", ".join(scripts) or "none"
        raise ValueError(
            f"Unknown script {name!r}. Available scripts: {available}."
        ) from error


def clean_script(
    name: str,
    project_root: Path = PROJECT_ROOT,
    output_root: Path | None = None,
) -> tuple[Path, bool]:
    """Delete only ``output/<script_name>`` for a known script."""
    stem, _ = resolve_script(name, project_root)
    root = (output_root or project_root / "output").resolve()
    target = (root / stem).resolve()
    if target.parent != root:
        raise ValueError(f"Unsafe output path for script {name!r}.")
    if not target.exists():
        return target, False
    if not target.is_dir():
        raise ValueError(f"Expected the script output to be a directory: {target}")
    shutil.rmtree(target)
    return target, True


def project_python(project_root: Path = PROJECT_ROOT) -> Path:
    """Prefer the project's virtual-environment Python when it is available."""
    candidates = (
        project_root / ".venv" / "bin" / "python",
        project_root / ".venv" / "Scripts" / "python.exe",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return Path(sys.executable)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tcc",
        description="Run project scripts and clean their isolated outputs.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("list", help="list available scripts")

    run_parser = subparsers.add_parser("run", help="run a script")
    run_parser.add_argument("script", help="script name, with or without .py")
    run_parser.add_argument(
        "script_args",
        nargs=argparse.REMAINDER,
        help="arguments passed to the script (use -- before option arguments)",
    )

    clean_parser = subparsers.add_parser(
        "clean", help="delete output/<script_name> for one script"
    )
    clean_parser.add_argument("script", help="script name, with or without .py")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.command == "list":
        scripts = discover_scripts()
        if not scripts:
            print("No scripts found.")
            return 0
        for name in scripts:
            print(name)
        return 0

    try:
        if args.command == "clean":
            target, removed = clean_script(args.script)
            if removed:
                print(f"Removed {target.relative_to(PROJECT_ROOT)}")
            else:
                print(f"Nothing to clean: {target.relative_to(PROJECT_ROOT)}")
            return 0

        _, script_path = resolve_script(args.script)
    except (OSError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 2

    script_args = args.script_args
    if script_args[:1] == ["--"]:
        script_args = script_args[1:]
    completed = subprocess.run(
        [str(project_python()), str(script_path), *script_args],
        cwd=PROJECT_ROOT,
        check=False,
    )
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
