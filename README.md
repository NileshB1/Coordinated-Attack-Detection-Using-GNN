# Coordinated Attack Detection Using GNN

Research code for deciding whether a group of flagged VM backup anomalies is a coordinated attack campaign or a set of coincidental events, using a dual-stream
(temporal and spatial) graph neural network over an enterprise infrastructure graph.

## Setup

Python 3.14 or newer

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows; use `source .venv/bin/activate` elsewhere
pip install -e ".[dev]"
```

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

`configs/default.yaml` holds the random seed, log level and data/output paths. Relative paths
are resolved against the config file's directory. `--seed`, `--datasets-dir`, `--output-dir`
and `--log-level` override the file for a single run.

## Layout

```
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
