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

## Dual-stream model

`cadgnn.models` holds the model foundation:

- `collate()` batches episode graphs and `check_batch()` validates a batch before the forward pass.
- `FeatureScaler` z-scores the four VM features with statistics from the training graphs only.
- `NodeEncoder` projects VM telemetry and gives the featureless node types a learned vector each.
- `DualStreamModel` runs a temporal and a spatial stream on the same batch and returns one
  embedding per graph from each (`StreamOutputs.temporal`, `StreamOutputs.spatial`).
- `save_checkpoint` / `load_checkpoint` store the weights, scaler statistics and model config.

Sizes, dropout and device are set in the `model` section of the config.

### Temporal severity stream

`cadgnn.temporal.TemporalSeverityStream` reads each episode's flagged events in time order.
Every step is the event's scaled telemetry plus `log1p` of the hours since the episode's first
event and of the hours since the previous event. A forward-only GRU runs over the steps, so the
state after event t depends only on events up to t; the episode vector combines the last state
with the mean state. There are no severity labels in the data, so the stream has no severity
target of its own and learns through the episode label. Missing, negative or decreasing event
times are rejected.

### Spatial topology stream

`cadgnn.spatial.SpatialTopologyStream` runs graph attention (`GATConv` per edge type inside
`HeteroConv`) over each episode's VMs, backup images, storage pools and network fabrics, with
residual connections and layer norm. Resource nodes have no features, so each node also gets
`log1p` of its degree within the episode graph, which tells a network shared by several flagged
VMs apart from one used by a single VM. With the default two layers a VM receives information
from the other VMs on its network. The episode vector is mean and max pooling per node type.
Only VM telemetry and graph structure are used: event times, labels, episode ids and global node
indices are not inputs, and the graph has no edge attributes. Layers and attention heads are set
by `spatial_layers` and `spatial_heads` in the config.

### Fusion and coordination output

`cadgnn.fusion.build_coordination_model(config)` builds the full model: both streams, a fusion
layer and a classifier that gives one logit per episode for coordinated (1) against
coincidental (0).

- `fusion: gated` (default) is a gated multimodal unit (Arevalo et al., 2017, *Gated Multimodal
  Units for Information Fusion*). The two embeddings come from different views, so each is
  projected to `fusion_dim` and a learned gate decides per episode how much each contributes.
  The mean gate value is returned as the temporal share of the decision.
- `fusion: concat` concatenates the embeddings and projects them; it is the fusion baseline.
- `use_temporal: false` or `use_spatial: false` runs a single-stream baseline for ablation.

`sigmoid(logit)` (`CoordinationModel.score`) is an uncalibrated coordination score, not a
probability; the decision threshold is chosen during evaluation. `coordination_loss` is binary
cross-entropy, and `positive_weight` sets its class weight from the training labels only.

## Training and evaluation

```bash
python run_training.py
python run_evaluation.py
```

`run_training.py` reads `episode_graphs.pt`, makes a stratified train / validation / test split
by episode (`val_fraction`, `test_fraction`), fits the feature scaler and class weight on the
training part, trains with Adam and early stopping on validation loss, and picks the decision
threshold on the validation part:

- `threshold_strategy: f1` takes the best validation F1;
- `threshold_strategy: fpr` takes the best validation recall with false positive rate at most
  `max_fpr`, for settings where alert fatigue matters more than missed campaigns.

Candidate thresholds are midpoints between validation scores, and among equally good ones the
middle one is used. The checkpoint (`model/coordination_model.pt`) stores the weights, scaler,
model and training config, seed, threshold and the episode ids of each split.

`run_evaluation.py` loads that checkpoint, scores only the test episodes once and writes to
`evaluation/`:

| File | Contents |
| --- | --- |
| `metrics.json` | precision, recall, F1, FPR, FNR, balanced accuracy, ROC-AUC, PR-AUC, confusion matrix, time to detect, gate summary, config and split sizes |
| `decisions.json` | per test episode: score, threshold, decision, true label, temporal share, VMs in event order, time window, shared networks and storage, and for flagged episodes the step-by-step propagation evidence |
| `decision_report.md` | the same results as a readable report |

Time to detect replays each coordinated test episode event by event and records when its score
first reaches the threshold. Structural path similarity, severity metrics and probability
calibration are not evaluated; the report lists why. Scores are uncalibrated, and all results
come from synthetic episodes.

## Layout

```
run_generator.py      entry point for the random anomaly generator
run_graph_builder.py  entry point for building the episode graphs
run_training.py       entry point for training and threshold selection
run_evaluation.py     entry point for the held-out evaluation and decision report
configs/              run configurations
notebooks/            reference notebook for the random anomaly generator (kept unchanged)
src/cadgnn/
  data/               dataset checks and generator calibration
  generator/          synthetic infrastructure, events and output files
  graph/              heterogeneous episode graph construction
  models/             batching, feature scaling, node encoder, dual-stream container
  temporal/           temporal severity stream
  spatial/            spatial topology stream
  fusion/             fusion layers, coordination model and loss
  training/           stratified split, training loop, training run
  evaluation/         metrics, thresholds, time to detect, decision report
  config.py, cli.py   configuration and command line
tests/                pytest suite
```

## Tests

```bash
pytest
ruff check . && ruff format --check .
```
