from pathlib import Path
import runpy

if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).with_name("exp11_external_test.py")), run_name="__main__")
