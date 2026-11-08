from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path, PurePosixPath


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


class EnvironmentRelayTests(unittest.TestCase):
    def test_windows_path_is_converted_without_losing_chinese_text(self):
        from morphoroute.runtime.wsl_relay import windows_to_wsl_path

        converted = windows_to_wsl_path(
            Path(r"C:\Users\Baoju\Desktop\实验代码")
        )

        self.assertEqual(
            converted,
            PurePosixPath("/mnt/c/Users/Baoju/Desktop/实验代码"),
        )

    def test_experiments_resolve_to_isolated_environment_kinds(self):
        from morphoroute.runtime.environments import (
            EnvironmentKind,
            EnvironmentResolver,
        )

        expectations = {
            "EXP-02": EnvironmentKind.YOLO,
            "EXP-05": EnvironmentKind.MAIN,
            "EXP-08C": EnvironmentKind.STATS,
            "EXP-15": EnvironmentKind.STATS,
        }
        for experiment_id, expected in expectations.items():
            with self.subTest(experiment_id=experiment_id):
                self.assertIs(
                    EnvironmentResolver.for_experiment(experiment_id),
                    expected,
                )

    def test_unknown_experiment_is_rejected(self):
        from morphoroute.runtime.environments import EnvironmentResolver

        with self.assertRaisesRegex(ValueError, "unknown experiment"):
            EnvironmentResolver.for_experiment("EXP-99")

    def test_wsl_command_passes_arguments_as_individual_tokens(self):
        from morphoroute.runtime.environments import EnvironmentKind
        from morphoroute.runtime.wsl_relay import build_wsl_command

        project = Path(r"C:\Users\Baoju\Desktop\实验代码\project")
        entry = project / "19 个独立实验运行入口" / "exp05_mask2former.py"
        command = build_wsl_command(
            project,
            entry,
            ["--profile", "local_6gb"],
            environment=EnvironmentKind.MAIN,
        )

        self.assertEqual(command[:4], ["wsl.exe", "-d", "Ubuntu", "--"])
        self.assertEqual(command[-2:], ["--profile", "local_6gb"])
        self.assertTrue(
            any("/mnt/c/Users/Baoju/Desktop/实验代码/project" in token for token in command)
        )
        self.assertTrue(
            any("exp05_mask2former.py" in token for token in command)
        )

    def test_wsl_command_sets_cublas_workspace_before_python(self):
        from morphoroute.runtime.environments import EnvironmentKind
        from morphoroute.runtime.wsl_relay import build_wsl_command

        project = Path(r"C:\Users\Baoju\Desktop\project")
        entry = project / "entry.py"
        command = build_wsl_command(
            project,
            entry,
            [],
            environment=EnvironmentKind.MAIN,
        )

        setting = "CUBLAS_WORKSPACE_CONFIG=:4096:8"
        self.assertEqual(command[4], "/usr/bin/env")
        self.assertIn(setting, command)
        setting_index = command.index(setting)
        python_index = next(
            index
            for index, token in enumerate(command)
            if token.endswith("/bin/python")
        )
        self.assertLess(setting_index, python_index)

    def test_relay_guard_prevents_recursive_windows_handoff(self):
        from morphoroute.runtime.wsl_relay import should_relay_to_wsl

        self.assertFalse(
            should_relay_to_wsl(
                dry_run=False,
                platform_name="linux",
                environment={"MORPHOROUTE_WSL_RELAY": "1"},
            )
        )
        self.assertFalse(
            should_relay_to_wsl(
                dry_run=True,
                platform_name="win32",
                environment={},
            )
        )
        self.assertTrue(
            should_relay_to_wsl(
                dry_run=False,
                platform_name="win32",
                environment={},
            )
        )

    def test_existing_environment_lock_is_reused_without_bootstrap(self):
        from morphoroute.runtime.environments import EnvironmentKind
        from morphoroute.runtime.wsl_relay import ensure_environment

        with tempfile.TemporaryDirectory() as temporary:
            project = Path(temporary)
            environment_root = project / "WSL2 与 Python 依赖文件"
            environment_root.mkdir()
            lock = environment_root / "environment-lock-main.json"
            lock.write_text("{}", encoding="utf-8")
            calls = []

            resolved = ensure_environment(
                project,
                EnvironmentKind.MAIN,
                command_runner=lambda *args, **kwargs: calls.append((args, kwargs)),
            )

            self.assertEqual(resolved, lock)
            self.assertEqual(calls, [])


if __name__ == "__main__":
    unittest.main()
