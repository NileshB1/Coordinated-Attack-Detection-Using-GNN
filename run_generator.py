"""Run the random anomaly generator.

    python run_generator.py
    
    python run_generator.py --seed 42 --output-dir runs/seed42
    python run_generator.py --config configs/my_scenario.yaml

"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from cadgnn.cli import main

if __name__ == "__main__":
    args = sys.argv[1:]
    if "--config" not in args and "-c" not in args:
        args = ["--config", str(ROOT / "configs" / "default.yaml"), *args]
    sys.exit(main([*args, "generate"]))
