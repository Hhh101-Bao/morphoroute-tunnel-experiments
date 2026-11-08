from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from morphoroute.aliasing import install_module_alias
from morphoroute.core.config import VALID_PROFILES
from morphoroute.core.layout import ProjectLayout


LAYOUT = ProjectLayout.discover(PROJECT_ROOT)
_CONFIG_GENERATOR = install_module_alias(
    "_morphoroute_generate_configs",
    LAYOUT.script_file("三套实验配置生成器.py", "generate_configs.py"),
)
_RUNNER = install_module_alias(
    "_morphoroute_run_experiment",
    LAYOUT.script_file("统一实验运行器.py", "run_experiment.py"),
)
FILE_NAMES = _CONFIG_GENERATOR.FILE_NAMES


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate every experiment configuration without training."
    )
    parser.add_argument(
        "--profile",
        choices=tuple(sorted(VALID_PROFILES)),
        default="local_6gb",
    )
    parser.add_argument(
        "--config-root",
        type=Path,
        default=None,
        help=(
            "Root that contains <profile>/exp00.json style directories. "
            "Defaults to the repository configuration directory."
        ),
    )
    arguments = parser.parse_args()
    config_directory = (
        arguments.config_root / arguments.profile
        if arguments.config_root is not None
        else LAYOUT.profile_config_dir(arguments.profile)
    )

    results = []
    failures = 0
    for experiment_id, file_name in FILE_NAMES.items():
        try:
            result = _RUNNER.execute_config(
                config_directory / file_name,
                dry_run=True,
            )
        except Exception as error:
            failures += 1
            results.append(
                {
                    "experiment_id": experiment_id,
                    "status": "dry_run_failed",
                    "error": f"{type(error).__name__}: {error}",
                }
            )
            continue
        results.append(
            {
                "experiment_id": experiment_id,
                "planned_outputs": result["planned_outputs"],
                "status": "dry_run_valid",
            }
        )
    print(json.dumps(results, ensure_ascii=False, indent=2))
    return 0 if failures == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
