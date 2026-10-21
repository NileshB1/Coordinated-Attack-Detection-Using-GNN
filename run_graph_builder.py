"""Build one heterogeneous graph per anomaly episode from the generator output.

    python run_generator.py          # first, if output/ is empty
    python run_graph_builder.py
    python run_graph_builder.py --output-dir runs/seed42

Reads the CSVs that run_generator.py wrote to the output directory and saves
episode_graphs.pt and episode_summary.csv next to them. Uses configs/default.yaml
unless --config is given.
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
    sys.exit(main([*args, "build-graphs"]))
