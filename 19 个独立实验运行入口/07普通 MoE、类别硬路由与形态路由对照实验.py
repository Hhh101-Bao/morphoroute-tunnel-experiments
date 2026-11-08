from pathlib import Path
import runpy

if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).with_name("exp07_routing_controls.py")), run_name="__main__")
