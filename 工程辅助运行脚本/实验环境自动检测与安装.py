from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from morphoroute.core.layout import ProjectLayout
from morphoroute.runtime.environments import EnvironmentKind
from morphoroute.runtime.wsl_relay import windows_to_wsl_path


LAYOUT = ProjectLayout.discover(PROJECT_ROOT)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="检查并按需安装铁路隧道实验的三个隔离 WSL2 环境。"
    )
    parser.add_argument(
        "--kind",
        choices=("main", "yolo", "stats", "all"),
        default="all",
    )
    parser.add_argument("--distribution", default="Ubuntu")
    parser.add_argument("--verify-only", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    setup_script = windows_to_wsl_path(
        LAYOUT.script_file("WSL2 实验环境安装脚本.sh", "setup_wsl.sh")
    )
    command = [
        "wsl.exe",
        "-d",
        arguments.distribution,
        "--",
        "bash",
        str(setup_script),
        "--kind",
        arguments.kind,
    ]
    if arguments.verify_only:
        command.append("--verify-only")
    completed = subprocess.run(command, check=False)
    report = {
        "command": command,
        "returncode": completed.returncode,
        "requested_kind": arguments.kind,
        "lock_files": {
            kind.value: str(
                LAYOUT.environment / f"environment-lock-{kind.value}.json"
            )
            for kind in EnvironmentKind
        },
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return int(completed.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
