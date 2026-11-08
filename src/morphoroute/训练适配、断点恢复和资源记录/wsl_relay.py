from __future__ import annotations

import os
import json
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Callable

from .environments import EnvironmentKind


RELAY_GUARD = "MORPHOROUTE_WSL_RELAY"
PATH_ARGUMENTS = frozenset({"--dataset-root", "--output-root", "--config"})


def windows_to_wsl_path(path: str | Path) -> PurePosixPath:
    text = str(path)
    windows_path = PureWindowsPath(text)
    drive = windows_path.drive
    if len(drive) != 2 or drive[1] != ":":
        if text.startswith("/"):
            return PurePosixPath(text)
        raise ValueError(f"expected an absolute Windows drive path, got: {path}")
    return PurePosixPath(
        "/mnt",
        drive[0].lower(),
        *windows_path.parts[1:],
    )


def _translate_path_arguments(arguments: Sequence[str]) -> list[str]:
    translated: list[str] = []
    translate_next = False
    for token in arguments:
        if translate_next:
            translated.append(str(windows_to_wsl_path(token)))
            translate_next = False
            continue
        translated.append(token)
        translate_next = token in PATH_ARGUMENTS
    if translate_next:
        raise ValueError(f"{translated[-1]} requires a path value")
    return translated


def build_wsl_command(
    project_root: str | Path,
    entry_path: str | Path,
    arguments: Sequence[str],
    *,
    environment: EnvironmentKind,
    distribution: str = "Ubuntu",
) -> list[str]:
    windows_project = Path(project_root)
    wsl_project = windows_to_wsl_path(windows_project)
    wsl_entry = windows_to_wsl_path(entry_path)
    environment_directory = (
        "WSL2 与 Python 依赖文件"
        if (windows_project / "WSL2 与 Python 依赖文件").is_dir()
        else "environment"
    )
    fallback_python = (
        wsl_project
        / environment_directory
        / ".venvs"
        / environment.value
        / "bin"
        / "python"
    )
    lock_path = (
        windows_project
        / environment_directory
        / f"environment-lock-{environment.value}.json"
    )
    venv_python = fallback_python
    if lock_path.is_file():
        try:
            payload = json.loads(lock_path.read_text(encoding="utf-8"))
            recorded_python = payload.get("python_executable")
            if recorded_python:
                venv_python = PurePosixPath(recorded_python)
        except (OSError, ValueError, TypeError):
            pass
    return [
        "wsl.exe",
        "-d",
        distribution,
        "--",
        "/usr/bin/env",
        "CUBLAS_WORKSPACE_CONFIG=:4096:8",
        str(venv_python),
        str(wsl_entry),
        *_translate_path_arguments(arguments),
    ]


def should_relay_to_wsl(
    *,
    dry_run: bool,
    platform_name: str | None = None,
    environment: Mapping[str, str] | None = None,
) -> bool:
    active_platform = platform_name or sys.platform
    active_environment = environment if environment is not None else os.environ
    return (
        active_platform == "win32"
        and not dry_run
        and active_environment.get(RELAY_GUARD) != "1"
    )


def _environment_root(project_root: Path) -> Path:
    chinese = project_root / "WSL2 与 Python 依赖文件"
    return chinese if chinese.is_dir() else project_root / "environment"


def _setup_script(project_root: Path) -> Path:
    chinese = (
        project_root
        / "工程辅助运行脚本"
        / "WSL2 实验环境安装脚本.sh"
    )
    return chinese if chinese.is_file() else project_root / "scripts" / "setup_wsl.sh"


def ensure_environment(
    project_root: str | Path,
    environment: EnvironmentKind,
    *,
    distribution: str = "Ubuntu",
    command_runner: Callable[..., Any] = subprocess.run,
) -> Path:
    project = Path(project_root)
    lock_path = (
        _environment_root(project)
        / f"environment-lock-{environment.value}.json"
    )
    if lock_path.is_file():
        return lock_path

    setup_path = _setup_script(project)
    if not setup_path.is_file():
        raise FileNotFoundError(f"WSL bootstrap script not found: {setup_path}")
    command = [
        "wsl.exe",
        "-d",
        distribution,
        "--",
        "bash",
        str(windows_to_wsl_path(setup_path)),
        "--kind",
        environment.value,
    ]
    completed = command_runner(command, check=False)
    return_code = int(getattr(completed, "returncode", 0))
    if return_code != 0:
        raise RuntimeError(
            f"WSL environment bootstrap failed for {environment.value} "
            f"with exit code {return_code}"
        )
    if not lock_path.is_file():
        raise RuntimeError(
            f"bootstrap finished without environment lock: {lock_path}"
        )
    return lock_path


def run_in_wsl(
    project_root: Path,
    entry_path: Path,
    arguments: Sequence[str],
    *,
    environment: EnvironmentKind,
    distribution: str = "Ubuntu",
) -> int:
    ensure_environment(
        project_root,
        environment,
        distribution=distribution,
    )
    command = build_wsl_command(
        project_root,
        entry_path,
        arguments,
        environment=environment,
        distribution=distribution,
    )
    process_environment = os.environ.copy()
    process_environment[RELAY_GUARD] = "1"
    completed = subprocess.run(command, env=process_environment, check=False)
    return int(completed.returncode)
