from __future__ import annotations

import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (
    PROJECT_ROOT
    / "工程辅助运行脚本"
    / "install_and_launch_exp07_strict_fp32.sh"
)


class StrictFP32ShellContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.text = SCRIPT.read_text(encoding="utf-8")

    def test_script_has_all_workflow_modes_and_strict_shell(self):
        self.assertIn("set -euo pipefail", self.text)
        for mode in (
            "probe)",
            "formal)",
            "status)",
            "verify-probe)",
            "verify-formal)",
        ):
            self.assertIn(mode, self.text)

    def test_probe_requires_verified_40000_checkpoint(self):
        self.assertIn("iter_40000.pth", self.text)
        self.assertIn("CHECKPOINT_NUMERICS=PASSED", self.text)
        self.assertIn("EXP07_FP32_MODE=probe", self.text)
        self.assertIn("EXP07_FP32_STOP_ITER=60000", self.text)

    def test_formal_requires_successful_probe_and_starts_without_resume(self):
        self.assertIn("EXP07_STRICT_FP32_PROBE_REACHED_LIMIT", self.text)
        self.assertIn("iter_60000.pth", self.text)
        self.assertIn("--no-resume", self.text)
        self.assertIn("EXP07_FP32_MODE=formal", self.text)

    def test_formal_backup_is_reversible_and_no_destructive_delete_exists(self):
        self.assertIn("AMP_FAILED_BACKUP", self.text)
        self.assertIn('mv "$EXP07_ROOT" "$AMP_FAILED_BACKUP"', self.text)
        self.assertNotIn("rm -rf", self.text)
        self.assertNotIn("rm -r", self.text)

    def test_only_immutable_morphology_cache_is_restored(self):
        self.assertIn("morphology_features_D1.csv", self.text)
        self.assertIn("gmm_D1.json", self.text)
        self.assertIn("REFUSE_CHECKPOINT_REUSE", self.text)

    def test_duplicate_process_guard_is_present(self):
        self.assertIn("pgrep -f", self.text)
        self.assertIn("EXP-07 process already exists", self.text)


if __name__ == "__main__":
    unittest.main()
