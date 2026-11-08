from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from morphoroute.core.layout import ProjectLayout


LAYOUT = ProjectLayout.discover(PROJECT_ROOT)


def _setup_script() -> str:
    return LAYOUT.script_file(
        "WSL2 实验环境安装脚本.sh",
        "setup_wsl.sh",
    ).read_text(encoding="utf-8")


class EnvironmentInstallScriptTests(unittest.TestCase):
    def test_interrupted_virtual_environment_is_resumed(self):
        script = _setup_script()

        self.assertIn('if [ ! -x "${python_path}" ]; then', script)
        self.assertIn("reusing partial environment and resuming installation", script)

    def test_exact_cuda_118_builds_use_mirrored_wheel_repository(self):
        script = _setup_script()

        self.assertIn("torch==2.1.2+cu118", script)
        self.assertIn("torchvision==0.16.2+cu118", script)
        self.assertIn("https://mirrors.aliyun.com/pytorch-wheels/cu118/", script)

    def test_verified_local_torch_wheel_is_reused_across_environments(self):
        script = _setup_script()

        self.assertIn("MORPHOROUTE_TORCH_LOCAL_WHEEL", script)
        self.assertIn(
            "60396358193f238888540f4a38d78485f161e28ec17fa445f0373b5350ef21f0",
            script,
        )
        self.assertIn("sha256sum --check", script)

    def test_all_pip_installs_share_the_configurable_fast_index(self):
        script = _setup_script()

        self.assertIn('pypi_index="${MORPHOROUTE_PYPI_INDEX_URL:', script)
        self.assertIn('-r "${requirements_path}"', script)
        self.assertIn('--index-url "${pypi_index}"', script)

    def test_mmcv_uses_the_matching_official_prebuilt_wheel(self):
        script = _setup_script()
        requirements = (
            LAYOUT.environment / "requirements-main.txt"
        ).read_text(encoding="utf-8")

        self.assertNotIn("openmim", requirements.lower())
        self.assertIn('"mmcv==2.1.0"', script)
        self.assertIn(
            "https://download.openmmlab.com/mmcv/dist/cu118/torch2.1/index.html",
            script,
        )


if __name__ == "__main__":
    unittest.main()
