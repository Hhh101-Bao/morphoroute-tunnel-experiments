from pathlib import Path

from _entry import main_for


PROFILE = "local_6gb"
DATASET_ROOT = Path(r"E:\桌面\铁路隧道多病害实例分割_原始影像严格划分_v1")
SMOKE_TEST = False
AUTO_RUN_PREREQUISITES = True
RESUME = True


def main(argv=None):
    return main_for(
        "EXP-02",
        profile=PROFILE,
        dataset_root=DATASET_ROOT,
        smoke_test=SMOKE_TEST,
        auto_run_prerequisites=AUTO_RUN_PREREQUISITES,
        resume=RESUME,
        entry_path=Path(__file__),
        argv=argv,
    )

if __name__ == "__main__":
    raise SystemExit(main())
