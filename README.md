# Coordinated Attack Detection Using GNN

Research code for deciding whether a group of flagged VM backup anomalies is a coordinated attack campaign or a set of coincidental events, using a dual-stream
(temporal and spatial) graph neural network over an enterprise infrastructure graph.

## Setup

Python 3.14 or newer

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows; use `source .venv/bin/activate` elsewhere
pip install -r requirements.txt       # to run the pipeline
pip install -r requirements-dev.txt   # also tests and linting
```

The `cadgnn` command used below needs the package itself installed as well
(`pip install -e .`). `run_generator.py` does not.

## Datasets

The calibration datasets are not tracked in Git. Place them under `datasets/`:

| Path | Source |
| --- | --- |
| `datasets/ransap/<Trace>.csv` | `ata_write.csv` of each RanSAP win7-120gb-hdd run (Excel, Firefox, Cerber, Darkside, GandCrab4, Ryuk, Sodinokibi, TeslaCrypt, WannaCry), https://github.com/manabu-hirano/RanSAP |
| `datasets/napierone/NapierOne-details.md` | https://github.com/simonrdavies/NapierOne |
| `datasets/filesystems/fslhomes-*/` | FSL Traces and Snapshots Archive, https://tracer.filesystems.org/traces/fslhomes/ |

Check that everything is in place:

```bash
cadgnn --config configs/default.yaml check
```

`python -m cadgnn ...` works the same way. The exit status is 0 when the datasets are complete,
1 when files are missing or malformed, and 2 for an invalid configuration.

## Configuration

`configs/default.yaml` holds the random seed, log level, data/output paths and the generator
scenario (infrastructure size, episode counts, timing ranges). Relative paths are resolved
against the config file's directory. `--seed`, `--datasets-dir`, `--output-dir` and
`--log-level` override the file for a single run.

## Generating synthetic telemetry

```bash
python run_generator.py
python run_generator.py --seed 42 --output-dir runs/seed42
```

`run_generator.py` uses `configs/default.yaml` unless `--config` is given and accepts the same
overrides as the CLI. With the package installed, `cadgnn --config configs/default.yaml generate`
is equivalent.

This calibrates the feature ranges from the datasets, builds the infrastructure graph
(12,000 VMs, 300 storage pools, 200 network fabrics by default), generates the labelled backup
events and writes to the output directory:

| File | Contents |
| --- | --- |
| `synthetic_backup_telemetry.csv` | one row per backup event with the four features, weighted `anomaly_score`, `state`, `episode_id` and `label` (`none`, `coincidental`, `coordinated`) |
| `infrastructure_graph_nodes.csv` | `node_id`, `kind` (`vm`, `backup_image`, `storage`, `network`) |
| `infrastructure_graph_edges.csv` | `source`, `target` |
| `experiment_manifest.json` | seed, scenario, calibrated feature ranges and label counts |

The generator is a port of `notebooks/random_anomaly_generator_simple.ipynb` and makes its
random draws in the same order. With `--seed 42` it reproduces the notebook's graph and events;
`tests/test_notebook_equivalence.py` checks this by running the notebook code itself.

## Building the episode graphs

```bash
python run_graph_builder.py
```

Reads the generator output from the output directory and builds one heterogeneous graph
(PyTorch Geometric `HeteroData`) per anomaly episode, labelled coordinated (1) or
coincidental (0). Benign backups are background and do not form graphs.

| Node type | Comes from |
| --- | --- |
| `vm` | each flagged VM in the episode; features are the event's `change_rate`, `file_entropy`, `backup_size_gb`, `file_count`, plus its timestamp and order in the episode |
| `backup_image`, `storage`, `network` | the image, storage pool and network fabric of those VMs |

| Edge type (each also has a reverse) |
| --- |
| `vm` -`has_backup`-> `backup_image` |
| `backup_image` -`stored_on`-> `storage` |
| `vm` -`attached_to`-> `network` |

It writes `episode_graphs.pt` (load with `cadgnn.graph.load_episode_graphs`) and
`episode_summary.csv` with per-episode size, time span and number of connected components.
Features are not scaled here; scaling is fitted on the training split later.

## Layout

```
run_generator.py      entry point for the random anomaly generator
run_graph_builder.py  entry point for building the episode graphs
configs/      run configurations
notebooks/    reference notebook for the random anomaly generator (kept unchanged)
src/cadgnn/   package source
tests/        pytest suite
```

## Tests

```bash
pytest
ruff check . && ruff format --check .
```
