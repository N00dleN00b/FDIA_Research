# FDIA detection with graph neural networks (IEEE 118-bus)

Detect and locate **false data injection attacks (FDIA)** on power-grid measurements, bus by bus,
using graph neural networks that take the grid topology as input, so detection keeps working when
lines trip and the topology changes.

Reference papers:
- Kipf & Welling, *Semi-Supervised Classification with Graph Convolutional Networks*, ICLR 2017 (GCN layer)
- Li, Wang & Lu, *Graph-based detection for false data injection attacks in power grid*, Energy 263 (2023) 125865 (GGNN + GAT detector, the method to reproduce)

## Status

| Date | Milestone |
|---|---|
| 2026-10-13 | Test bed built: data pipeline, 4 detectors (MLP, GCN, GGNN, GGNN+GAT), metrics, unit tests; first results on the 3,000-sample IEEE 118-bus dataset |

Latest results: [`results/`](results/)

## Project layout

```
fdia-gnn/
├── data_generation/
│   └── data_generation.py     # professor's generator (original, unmodified)
├── scripts/
│   ├── generate_data.py       # seeded + parallel wrapper around the generator
│   ├── check_dataset.py       # sanity checks + summary -> results/dataset_summary.json
│   └── run_baselines.py       # train/evaluate models -> results/<tag>.json/.md
├── src/fdia/
│   ├── data.py                # load, clean NaNs, splits (random / unseen-topology), normalization
│   ├── models.py              # MLP, GCN, GGNN, GGNN_GAT
│   ├── metrics.py             # F1, precision/recall, MAR, FAR, ROC-AUC, snapshot-level rates
│   └── train.py               # shared training loop with early stopping on validation F1
├── tests/                     # pytest unit tests (run in ~15 s, no dataset needed)
├── data/                      # fdia_dataset_ieee118.npz (~10 MB, committed), see data/README.md
├── results/                   # experiment outputs (committed so progress is visible)
├── docs/
│   ├── SETUP.md               # step-by-step: VS Code, Python env, GitHub
│   └── TEST_PROCEDURE.md      # what to run, in what order, what "pass" means
├── requirements.txt
└── pyproject.toml
```

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m pytest                   # 13 tests should pass
python scripts/generate_data.py --samples 200 --out data/small.npz      # ~2 min
python scripts/run_baselines.py --data data/small.npz --epochs 5        # smoke run
```

Full procedure: [docs/TEST_PROCEDURE.md](docs/TEST_PROCEDURE.md).

## Task definition

- **Input per snapshot:** node features `x` (118 × [V, θ, P, Q]) and admittance-weighted adjacency `a` (118 × 118).
- **Output:** probability that each bus's measurements were falsified.
- **Metrics:** about 1% of buses are attacked, so accuracy alone is misleading (predicting "all normal" scores ~99%).
  We report F1, precision, recall, missed-alarm rate (MAR), false-alarm rate (FAR), ROC-AUC, and snapshot-level detection.
- **Splits:** `random` (70/15/15 by sample) and `topology` (test topologies never appear in training).
