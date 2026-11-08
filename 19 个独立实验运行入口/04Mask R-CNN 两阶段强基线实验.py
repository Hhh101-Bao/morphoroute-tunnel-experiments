from pathlib import Path
import runpy

if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).with_name("exp04_mask_rcnn.py")), run_name="__main__")
