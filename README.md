# FDIA detection with graph neural networks (IEEE 14- and 118-bus)

Detect and locate **false data injection attacks (FDIA)** on power-grid measurements, bus by bus,
using graph neural networks that take the grid topology as input, so detection keeps working when
lines trip and the topology changes.

Reference papers:
- Kipf & Welling, *Semi-Supervised Classification with Graph Convolutional Networks*, ICLR 2017 (GCN layer)
- Li, Wang & Lu, *Graph-based detection for false data injection attacks in power grid*, Energy 263 (2023) 125865 (GGNN + GAT detector, replicated here on IEEE 14-bus)

## Status

| Date | Milestone |
|---|---|
| 2026-10-20 | Li et al. (2023) replication run: gated GNNs within 0–4 points of the paper; GCN, attention gain and missed-alarm rate not replicated |
| 2026-10-19 | Li et al. (2023) replication code added: stealthy DC attacks, CNN baseline, per-bus scaling, comparison with paper tables |
| 2026-10-13 | Test bed built: data pipeline, 4 detectors (MLP, GCN, GGNN, GGNN+GAT), metrics, unit tests; first results on the 3,000-sample IEEE 118-bus dataset |

## Results

All numbers are from one run (seed 0) on a 12-core laptop. Full tables: [`results/`](results/).

### 1. Li et al. (2023) replication, IEEE 14-bus

Stealthy a = Hc attacks under DC state estimation, 20% attack strength, 5,600 samples per scenario,
per-bus feature scaling. Node accuracy %, ours vs the paper. Details and assumptions:
[docs/REPLICATION_LI2023.md](docs/REPLICATION_LI2023.md); full table: [`results/li2023_replication.md`](results/li2023_replication.md).

| Scenario | CNN (paper / ours) | GCN (paper / ours) | GGNN (paper / ours) | GGNN + GAT (paper / ours) |
|---|---|---|---|---|
| Single-node (Table 2) | — / 90.8 | 98.1 / 78.1 | 98.5 / **96.7** | 99.1 / **96.0** |
| Multi-node (Table 3) | — / 90.1 | 94.5 / 78.9 | 95.0 / **95.3** | 99.9 / **94.9** |
| Single-node + topology changes (Table 4) | 83.0 / 90.2 | 90.3 / 76.9 | 95.8 / **96.0** | 96.2 / **95.0** |
| Multi-node + topology changes (Table 5) | 80.9 / 89.0 | 91.4 / 77.3 | 94.1 / **95.2** | 95.8 / **95.3** |

Detection vs attack strength (GGNN + GAT, single-node, Fig. 13):

| Attack strength | 1% | 5% | 10% | 15% | 20% |
|---|---|---|---|---|---|
| Paper | 56.3 | — | — | — | 98.0 |
| Ours | 57.7 | 73.0 | 85.7 | 91.8 | 96.0 |

- **Replicated:** GGNN and GGNN + GAT land within 0–4 points of the paper in every scenario, and the
  attack-strength curve matches both points the paper states (1%, 20%). Attacks are verified stealthy
  (bad-data residual unchanged; `tests/test_li2023.py`). On a second machine, GGNN and GGNN + GAT reproduced within 0.4 points; CNN/GCN within 1.7, and the 1%/5% attack results by ~5 points, so those need multiple seeds.
- **Not replicated:** our GCN is much weaker (77–79% vs 90–98%); attention gives no gain over plain GGNN;
  missed-alarm rate is 8–10% vs the paper's 1–3%.
- **Key finding:** detection only works with per-bus feature scaling (`--norm per_bus`). With shared
  scaling, all models stayed near chance even at 20% attacks.

### 2. Professor's dataset, IEEE 118-bus (3,000 samples, 0.5–2% random P/Q attacks)

ROC-AUC on the test set (0.5 = random guessing). Full tables: [`results/baselines_random.md`](results/baselines_random.md),
[`results/baselines_topology.md`](results/baselines_topology.md).

| Model | Random split | Unseen-topology split |
|---|---|---|
| MLP (no topology) | 0.569 | 0.557 |
| GCN | 0.591 | 0.579 |
| GGNN | 0.572 | 0.572 |
| GGNN + GAT | 0.594 | 0.582 |

All models are only slightly above chance: the 0.5–2% attacks are hidden by ±40% load variation.
Run with the original (shared) feature scaling; per-bus scaling on this dataset is the next experiment.

### Open questions

1. Should the 118-bus attacks be stealthy (a = Hc) instead of random P/Q perturbations?
2. Should transformers be graph edges in the 118-bus generator? (They are in the 14-bus version.)
3. How should islanded buses (NaN V, θ after line trips) be handled?
4. Which "proposed framework" should be implemented next?

## Project layout

```
fdia-gnn/
├── data_generation/
│   └── data_generation.py     # professor's generator (original, unmodified)
├── scripts/
│   ├── generate_data.py       # seeded + parallel wrapper around the generator
│   ├── check_dataset.py       # sanity checks + summary -> results/dataset_summary.json
│   ├── run_baselines.py       # train/evaluate models -> results/<tag>.json/.md
│   ├── attack_strength_sweep.py   # detection vs attack size on the main dataset
│   ├── generate_li2023_data.py    # Li et al.-style 14-bus dataset (DC model, stealthy a = Hc attacks)
│   └── replicate_li2023.py        # run Li et al.'s experiments, compare with the paper's tables
├── src/fdia/
│   ├── data.py                # load, clean NaNs, splits (random / unseen-topology), normalization
│   ├── models.py              # MLP, CNN, GCN, GGNN, GGNN_GAT
│   ├── li2023.py              # DC state estimation + stealthy attack generator (Li et al. 2023)
│   ├── metrics.py             # F1, precision/recall, MAR, FAR, ROC-AUC, snapshot-level rates
│   └── train.py               # shared training loop with early stopping on validation F1
├── tests/                     # pytest unit tests (run in ~15 s, no dataset needed)
├── data/                      # fdia_dataset_ieee118.npz (~10 MB, committed), see data/README.md
├── results/                   # experiment outputs (committed so progress is visible)
├── docs/
│   ├── SETUP.md               # step-by-step: VS Code, Python env, GitHub
│   ├── TEST_PROCEDURE.md      # what to run, in what order, what "pass" means
│   └── REPLICATION_LI2023.md  # replicating Li et al. on IEEE 14-bus: assumptions + steps
├── requirements.txt
└── pyproject.toml
```

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m pytest                   # 24 tests should pass
python scripts/generate_data.py --samples 200 --out data/small.npz      # ~2 min
python scripts/run_baselines.py --data data/small.npz --epochs 5        # smoke run
```

Full procedure: [docs/TEST_PROCEDURE.md](docs/TEST_PROCEDURE.md).

## Task definition

- **Input per snapshot:** node features `x` (N × [V, θ, P, Q]) and admittance-weighted adjacency `a` (N × N); N = 118 or 14.
- **Output:** probability that each bus's measurements were falsified.
- **Metrics:** on the 118-bus data about 1% of buses are attacked, so accuracy alone is misleading (predicting "all normal" scores ~99%).
  We report F1, precision, recall, missed-alarm rate (MAR), false-alarm rate (FAR), ROC-AUC, and snapshot-level detection.
- **Splits:** `random` (70/15/15 by sample) and `topology` (test topologies never appear in training).
