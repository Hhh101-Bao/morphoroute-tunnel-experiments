from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from morphoroute.aliasing import install_module_alias
from morphoroute.core.layout import ProjectLayout


LAYOUT = ProjectLayout.discover(PROJECT_ROOT)
_RUNNER = install_module_alias(
    "_morphoroute_run_experiment",
    LAYOUT.script_file("统一实验运行器.py", "run_experiment.py"),
)


if __name__ == "__main__":
    raise SystemExit(_RUNNER.main([*sys.argv[1:], "--resume"]))
