from __future__ import annotations

import contextlib
import importlib
import io
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
from morphoroute.core.layout import ProjectLayout


EXPERIMENTS_ROOT = ProjectLayout.discover(PROJECT_ROOT).experiments
for search_path in (str(EXPERIMENTS_ROOT),):
    if search_path not in sys.path:
        sys.path.insert(0, search_path)


ENTRY_MODULES = (
    "exp00_preregister",
    "exp01_data_audit",
    "exp02_yolo_original",
    "exp03_yolo_strict",
    "exp04_mask_rcnn",
    "exp05_mask2former",
    "exp06_resource_matched",
    "exp07_routing_controls",
    "exp08a_gradient_conflict",
    "exp08b_performance_diagnosis",
    "exp08c_morphology_association",
    "exp08d_expert_intervention",
    "exp09_ablation",
    "exp10_freeze",
    "exp11_external_test",
    "exp12_reliability",
    "exp13_quantification",
    "exp14_sequence_efficiency",
    "exp15_statistics",
)


class StandaloneEntryTests(unittest.TestCase):
    def test_project_layout_discovers_existing_chinese_directories(self):
        from morphoroute.core.layout import ProjectLayout

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            names = {
                "experiments": "19 个独立实验运行入口",
                "tests": "61 项自动化测试",
                "scripts": "工程辅助运行脚本",
                "configs": "三种运行档位的实验配置",
                "environment": "WSL2 与 Python 依赖文件",
            }
            for directory_name in names.values():
                (root / directory_name).mkdir()
            (root / "src").mkdir()

            layout = ProjectLayout.discover(root)

            for field_name, directory_name in names.items():
                self.assertEqual(
                    getattr(layout, field_name),
                    root / directory_name,
                )

    def test_project_layout_resolves_renamed_chinese_script_files(self):
        from morphoroute.core.layout import ProjectLayout

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            scripts = root / "工程辅助运行脚本"
            scripts.mkdir()
            chinese_generator = scripts / "三套实验配置生成器.py"
            chinese_runner = scripts / "统一实验运行器.py"
            chinese_generator.touch()
            chinese_runner.touch()

            layout = ProjectLayout.discover(root)

            self.assertEqual(
                layout.script_file("三套实验配置生成器.py", "generate_configs.py"),
                chinese_generator,
            )
            self.assertEqual(
                layout.script_file("统一实验运行器.py", "run_experiment.py"),
                chinese_runner,
            )

    def test_project_layout_maps_profile_keys_to_chinese_config_directories(self):
        from morphoroute.core.layout import ProjectLayout

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            configs = root / "三种运行档位的实验配置"
            expected = {
                "autodl_5090_32gb": (
                    configs / "RTX 5090 32GB AutoDL 加速正式实验配置"
                ),
                "local_6gb": configs / "RTX 4050 6GB 本机正式实验配置",
                "local_6gb_fallback": configs / "本机低显存预实验和流程验证配置",
                "sop_full": configs / "完整 SOP 正式实验配置",
            }
            for directory in expected.values():
                directory.mkdir(parents=True, exist_ok=True)

            layout = ProjectLayout.discover(root)

            for profile, directory in expected.items():
                self.assertEqual(layout.profile_config_dir(profile), directory)

    def test_cli_accepts_autodl_5090_profile(self):
        from morphoroute.core.standalone import (
            EntrySettings,
            parse_entry_arguments,
        )

        defaults = EntrySettings.defaults(project_root=PROJECT_ROOT)
        resolved = parse_entry_arguments(
            ["--profile", "autodl_5090_32gb"],
            defaults,
        )

        self.assertEqual(resolved.settings.profile, "autodl_5090_32gb")

    def test_defaults_use_local_gpu_profile_and_authoritative_dataset(self):
        from morphoroute.core.standalone import EntrySettings

        settings = EntrySettings.defaults(project_root=PROJECT_ROOT)

        self.assertEqual(settings.profile, "local_6gb")
        self.assertEqual(
            settings.dataset_root,
            Path(
                r"E:\桌面"
                r"\铁路隧道多病害实例分割_原始影像严格划分_v1"
            ),
        )
        self.assertEqual(settings.output_root, PROJECT_ROOT / "outputs")
        self.assertTrue(settings.auto_run_prerequisites)
        self.assertTrue(settings.resume)
        self.assertFalse(settings.smoke_test)

    def test_empty_cli_keeps_visible_entry_defaults(self):
        from morphoroute.core.standalone import EntrySettings, parse_entry_arguments

        defaults = EntrySettings.defaults(project_root=PROJECT_ROOT)
        resolved = parse_entry_arguments([], defaults)

        self.assertEqual(resolved.settings, defaults)
        self.assertFalse(resolved.dry_run)
        self.assertFalse(resolved.analysis_only)
        self.assertIsNone(resolved.config_path)

    def test_every_entry_runs_its_own_dry_plan_without_manual_stitching(self):
        for module_name in ENTRY_MODULES:
            with self.subTest(module=module_name):
                module = importlib.import_module(module_name)
                with contextlib.redirect_stdout(io.StringIO()):
                    status = module.main(["--dry-run"])
                self.assertEqual(status, 0)


if __name__ == "__main__":
    unittest.main()
