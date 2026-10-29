"""Evaluate the trained model on the held-out test episodes and write the report.

python run_evaluation.py
python run_evaluation.py --output-dir runs/seed42
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from cadgnn.cli import main  # noqa: E402

if __name__ == "__main__":
    args = sys.argv[1:]
    if "--config" not in args and "-c" not in args:
        args = ["--config", str(ROOT / "configs" / "default.yaml"), *args]
    sys.exit(main([*args, "evaluate"]))
