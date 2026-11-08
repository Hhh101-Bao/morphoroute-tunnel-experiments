from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


class DependencyCompletionStateTests(unittest.TestCase):
    def test_sealed_and_locked_success_states_are_complete(self):
        from morphoroute.core.dependencies import is_complete

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for experiment_id, status in (
                ("EXP-11", "completed_sealed"),
                ("EXP-15", "completed_locked"),
                ("EXP-12", "prefreeze_completed"),
            ):
                path = root / "experiments" / experiment_id / "run_manifest.json"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps({"status": status}), encoding="utf-8")
                self.assertTrue(is_complete(root, experiment_id))


if __name__ == "__main__":
    unittest.main()
