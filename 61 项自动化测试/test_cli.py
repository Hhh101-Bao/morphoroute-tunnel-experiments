import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from morphoroute.aliasing import install_module_alias
from morphoroute.core.layout import ProjectLayout


LAYOUT = ProjectLayout.discover(PROJECT_ROOT)
RUN_EXPERIMENT = LAYOUT.script_file(
    "统一实验运行器.py",
    "run_experiment.py",
)
GENERATE_CONFIGS = LAYOUT.script_file(
    "三套实验配置生成器.py",
    "generate_configs.py",
)


class CLITests(unittest.TestCase):
    def _config(self, root: Path, experiment_id: str = "EXP-00") -> Path:
        data_root = root / "data"
        output_root = root / "output"
        data_root.mkdir(exist_ok=True)
        payload = {
            "experiment_id": experiment_id,
            "profile": "local_6gb",
            "data_root": str(data_root),
            "output_root": str(output_root),
            "seed": 42,
            "allowed_splits": [],
            "parameters": {"model": "analysis"},
        }
        path = root / "config.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def _run(self, *arguments):
        return subprocess.run(
            [sys.executable, *map(str, arguments)],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
            env={**os.environ, "PYTHONIOENCODING": "utf-8"},
            check=False,
        )

    def test_unified_cli_returns_json_dry_plan(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = self._config(Path(tmp))
            result = self._run(
                RUN_EXPERIMENT,
                "--config",
                config,
                "--dry-run",
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            payload = json.loads(result.stdout)
            self.assertEqual(payload["experiment_id"], "EXP-00")
            self.assertTrue(payload["dry_run"])

    def test_unknown_experiment_returns_nonzero_with_clear_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = self._config(Path(tmp), "EXP-99")
            result = self._run(
                RUN_EXPERIMENT,
                "--config",
                config,
                "--dry-run",
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("unknown experiment", result.stderr)

    def test_config_generator_creates_four_profiles_and_controlled_cards(self):
        module = install_module_alias(
            "_test_generate_configs",
            GENERATE_CONFIGS,
        )
        generate_profile_configs = module.generate_profile_configs

        with tempfile.TemporaryDirectory() as tmp:
            paths = generate_profile_configs(Path(tmp))
            self.assertEqual(len(paths), 76)
            self.assertTrue((Path(tmp) / "sop_full" / "exp08a.json").is_file())
            fallback = json.loads(
                (Path(tmp) / "local_6gb_fallback" / "exp15.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(fallback["profile"], "local_6gb_fallback")
            self.assertEqual(fallback["parameters"]["result_status"], "preexperiment")

            local_exp05_payload = json.loads(
                (Path(tmp) / "local_6gb" / "exp05.json").read_text(
                    encoding="utf-8"
                )
            )
            local_exp05 = local_exp05_payload["parameters"]
            local_exp04 = json.loads(
                (Path(tmp) / "local_6gb" / "exp04.json").read_text(
                    encoding="utf-8"
                )
            )["parameters"]
            local_exp09 = json.loads(
                (Path(tmp) / "local_6gb" / "exp09.json").read_text(
                    encoding="utf-8"
                )
            )["parameters"]
            cloud_exp09_payload = json.loads(
                (
                    Path(tmp)
                    / "autodl_5090_32gb"
                    / "exp09.json"
                ).read_text(encoding="utf-8")
            )
            cloud_exp09 = cloud_exp09_payload["parameters"]
            sop_exp09 = json.loads(
                (Path(tmp) / "sop_full" / "exp09.json").read_text(
                    encoding="utf-8"
                )
            )["parameters"]

            self.assertEqual(local_exp04["query_count"], 100)
            self.assertNotIn("budget_gate", local_exp04)
            self.assertEqual(
                local_exp04["validation_update_interval"],
                5_000,
            )
            self.assertEqual(local_exp05["query_count"], 64)
            self.assertEqual(
                local_exp05_payload["data_root"],
                "/mnt/e/桌面/铁路隧道多病害实例分割_原始影像严格划分_v1",
            )
            self.assertEqual(
                local_exp05["budget_gate"]["start_update"],
                20_000,
            )
            self.assertFalse(local_exp05["release_budget_gate"])
            self.assertEqual(
                local_exp09["budget_gate"]["start_update"],
                30_000,
            )
            self.assertNotIn("release_budget_gate_variants", local_exp09)

            self.assertEqual(cloud_exp09["query_count"], 64)
            self.assertEqual(
                cloud_exp09_payload["data_root"],
                "/root/autodl-tmp/铁路隧道多病害实例分割_原始影像严格划分_v1",
            )
            self.assertEqual(cloud_exp09["physical_batch"], 4)
            self.assertEqual(cloud_exp09["effective_batch"], 8)
            self.assertFalse(cloud_exp09["gradient_checkpointing"])
            self.assertEqual(cloud_exp09["dataloader_workers"], 8)
            self.assertTrue(cloud_exp09["pin_memory"])
            self.assertEqual(cloud_exp09["prefetch_factor"], 2)
            self.assertEqual(
                cloud_exp09["budget_gate"]["start_update"],
                30_000,
            )

            self.assertEqual(sop_exp09["query_count"], 100)
            self.assertNotIn("budget_gate", sop_exp09)

    def test_autodl_profile_is_a_valid_experiment_profile(self):
        from morphoroute.core.config import ExperimentConfig

        config = ExperimentConfig.from_mapping(
            {
                "experiment_id": "EXP-09",
                "profile": "autodl_5090_32gb",
                "data_root": "data",
                "output_root": "outputs",
                "seed": 42,
                "allowed_splits": ["D1", "D2", "D3"],
                "parameters": {},
            }
        )

        self.assertEqual(config.profile, "autodl_5090_32gb")

    def test_config_generator_can_target_visible_profile_directories(self):
        module = install_module_alias(
            "_test_generate_configs_visible_dirs",
            GENERATE_CONFIGS,
        )

        with tempfile.TemporaryDirectory() as tmp:
            paths = module.generate_profile_configs(
                Path(tmp),
                profile_directory_names={
                    "autodl_5090_32gb": "cloud-visible",
                },
            )

            self.assertEqual(len(paths), 76)
            self.assertTrue(
                (Path(tmp) / "cloud-visible" / "exp09.json").is_file()
            )


if __name__ == "__main__":
    unittest.main()
