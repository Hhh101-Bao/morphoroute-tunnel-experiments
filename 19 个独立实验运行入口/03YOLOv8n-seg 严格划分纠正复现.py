from pathlib import Path
import runpy

if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).with_name("exp03_yolo_strict.py")), run_name="__main__")
