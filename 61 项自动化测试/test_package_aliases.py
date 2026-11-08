from __future__ import annotations

import sys
import tempfile
import types
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


class PackageAliasTests(unittest.TestCase):
    def test_valid_python_alias_loads_package_from_chinese_directory(self):
        from morphoroute.aliasing import install_package_alias

        with tempfile.TemporaryDirectory() as temporary:
            package_root = Path(temporary)
            chinese_directory = package_root / "配置、实验注册、运行记录和 D4 防泄漏组件"
            chinese_directory.mkdir()
            (chinese_directory / "__init__.py").write_text(
                "VALUE = 17\n",
                encoding="utf-8",
            )
            parent = types.ModuleType("fixturepkg")
            parent.__path__ = [str(package_root)]
            sys.modules["fixturepkg"] = parent
            try:
                module = install_package_alias(
                    "fixturepkg.core",
                    chinese_directory,
                )
                self.assertEqual(module.VALUE, 17)
                self.assertEqual(
                    Path(module.__path__[0]),
                    chinese_directory,
                )
            finally:
                sys.modules.pop("fixturepkg.core", None)
                sys.modules.pop("fixturepkg", None)

    def test_module_alias_preserves_original_chinese_filename(self):
        from morphoroute.aliasing import install_module_alias

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            module_path = root / "guard.py"
            module_path.write_text("TOKEN = 'guard-loaded'\n", encoding="utf-8")
            parent = types.ModuleType("fixturepkg")
            parent.__path__ = [str(root)]
            sys.modules["fixturepkg"] = parent
            try:
                module = install_module_alias(
                    "fixturepkg.d4_guard",
                    module_path,
                )
                self.assertEqual(module.TOKEN, "guard-loaded")
                self.assertTrue(module_path.is_file())
            finally:
                sys.modules.pop("fixturepkg.d4_guard", None)
                sys.modules.pop("fixturepkg", None)


if __name__ == "__main__":
    unittest.main()
