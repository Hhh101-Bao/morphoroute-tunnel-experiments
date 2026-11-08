from pathlib import Path
import runpy

if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).with_name("exp02_yolo_original.py")), run_name="__main__")
