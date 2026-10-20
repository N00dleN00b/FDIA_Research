# Test procedure

Run these in order. Each step says what to run, how long it takes, and what "pass" looks like.
All commands run from the project folder with the `.venv` activated.

## Step 1 — Unit tests (15 s)

```
python -m pytest -v
```
**Pass:** 13 passed. These check, without any real data:
- data loading, NaN cleaning, splits partition the data, the topology split never leaks a topology into test
- every model's forward and backward pass, and that attention only looks at real neighbours
- metrics, including the reason we don't trust accuracy alone (all-normal prediction = 99% accuracy, F1 0)
- a tiny end-to-end training run that must learn an obvious attack (ROC-AUC > 0.9)

## Step 2 — Get data

Small set for fast iteration (~2 min):
```
python scripts/generate_data.py --samples 200 --out data/small.npz
```
Full set (~20–40 min, or use the professor's file):
```
python scripts/generate_data.py --samples 3000 --workers 2 --seed 42
```

## Step 3 — Dataset sanity check (10 s)

```
python scripts/check_dataset.py --data data/fdia_dataset_ieee118.npz
```
Writes `results/dataset_summary.json`.
**Pass:** shape checks say PASS. Two WARNs are **expected** with the current generator and are open questions for
the professor, not bugs in the test bed:
- `no NaN features` → islanded buses after line trips (handled: set to 0 in `fdia/data.py`)
- `no isolated buses` → transformers are not included as edges

Look at: attacked-sample rate (~0.5), attacked-node rate (~0.01), number of unique topologies.

## Step 4 — Baselines, random split (~10–20 min on CPU)

```
python scripts/run_baselines.py
```
Writes `results/baselines_random.md` and `.json`. Models: MLP (ignores topology), GCN, GGNN, GGNN+GAT.

## Step 5 — Baselines, unseen-topology split

```
python scripts/run_baselines.py --split topology
```
This is the experiment that matches the research claim: does using topology help on grid configurations
the model never saw?

## Step 6 — For reportable numbers: multiple seeds

```
python scripts/run_baselines.py --split topology --seeds 0 1 2 3 4
```
Tables then show mean ± std.

## Reading the results table

| Column | Meaning | Good |
|---|---|---|
| F1 | Balance of precision and recall on attacked buses | high |
| ROC-AUC | Ranking quality, threshold-free. 0.5 = random guessing | high |
| Precision | Of buses flagged, fraction truly attacked | high |
| Recall | Of attacked buses, fraction caught | high |
| MAR | Missed-alarm rate = 1 − recall (Li et al. report this) | low |
| FAR | False-alarm rate on normal buses | low |
| Node acc | Per-bus accuracy — **misleading on its own** | — |
| Snapshot detect / false alarm | "Was this snapshot attacked at all?" | high / low |

## After each experiment

```
git add results/
git commit -m "Results: <what you ran>"
git push
```
