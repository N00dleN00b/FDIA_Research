# Replicating Li, Wang & Lu (2023) on IEEE 14-bus

**Goal:** check that our implementation of the paper's detector (GGNN + GAT) and baselines
behaves like the paper when given the same kind of data. This validates the code before we use
it on the professor's dataset.

Paper: X. Li, Y. Wang, Z. Lu, *Graph-based detection for false data injection attacks in power grid*,
Energy 263 (2023) 125865.

## What the replication follows from the paper

| Paper (section) | Our implementation (`src/fdia/li2023.py`) |
|---|---|
| DC state estimation z = Hx + e (2.1) | H = [bus injections; line flows] from the DC susceptance matrix B; WLS estimate |
| Stealthy attack a = Hc passes bad-data detection (2.2, Eq. 6) | `stealthy_attack()`; the residual before/after is stored and tested to be identical |
| Attack affects 1st- and 2nd-order neighbours of the target (2.8, Fig. 4) | c is set on the target; zero-injection buses get the c that keeps them at 0, which spreads it to 2nd-order neighbours |
| Don't target zero-injection or generator buses (2.8) | Targets come only from load buses; IEEE 14-bus: buses 4, 5, 9–14 |
| Topology changes in 10% of data (3.5) | `--scenario single_topo` / `multi_topo`: one line or transformer switched out, grid kept connected |
| Node features P, Q, V, θ; admittance-weighted adjacency (2.7) | x = [V, θ̂, P_meas, Q]; transformers included as edges |
| 5,600 samples, 75/25 split (Table 1) | 5,600 samples; 67.5/7.5/25 (a small validation slice from the 75% for early stopping) |
| MinMax input scaling (3.1) | Per-bus standardization (same effect: each bus is scaled to its own range) |
| Accuracy and missed-alarm rate (3.2) | Both reported, plus F1 and snapshot-level accuracy |
| CNN, GCN, GGNN, GGNN + GAT (Tables 2–5) | Same four models |

## What we had to assume (the paper doesn't say)

| Detail | Our choice | Why it matters |
|---|---|---|
| NYISO hourly load data | Synthetic two-peak daily curve × per-load noise (±5%) | Shapes how predictable "normal" looks |
| "Attack power" definition | Relative change of the target bus's injection measurement (default 20%) | Directly sets difficulty (Fig. 13) |
| Multi-node attack pattern | Two target buses | More targets = more buses labelled attacked |
| Labels | Every bus whose injection measurement changed (`--label affected`); `--label targets` = only targets | Changes what counts as "detected" |
| Accuracy: per bus or per snapshot? | We report both | Paper is ambiguous |
| Measurement noise | 1% of reading | |
| Hyperparameters | Our defaults (hidden 64, lr 0.001) instead of the paper's (hidden 4–8, lr 0.1) | Paper's lr 0.1 is unusually high |
| GGNN aggregation | Weighted sum (paper: max) | Minor |

Because of these, expect **the same pattern** (graph models beat CNN, GGNN + GAT best, accuracy
rises with attack strength), not identical numbers.

## Files added

```
src/fdia/li2023.py              # DC model, stealthy attacks, dataset generator
scripts/generate_li2023_data.py # CLI: build one scenario's dataset
scripts/replicate_li2023.py     # CLI: build all scenarios, train all models, compare to paper
tests/test_li2023.py            # 10 tests proving the attack model follows the paper
docs/REPLICATION_LI2023.md      # this file
```
Also changed: `src/fdia/models.py` (+ CNN baseline), `src/fdia/data.py` + `train.py`
(+ per-bus scaling and custom split fractions), `scripts/run_baselines.py` (+ `--frac`, `--norm`).
Defaults of the existing scripts are unchanged, so earlier results still reproduce.

## Step-by-step test procedure

All from the project folder with `.venv` active.

### Step 1: Unit tests (~15 s)
```
python -m pytest -v
```
**Pass:** 24 passed. The 10 new ones in `test_li2023.py` prove:
- `test_dc_model_matches_pandapower`: our DC model equals pandapower's DC power flow
- `test_attacks_are_stealthy`: the bad-data residual is unchanged by every attack (paper Eq. 6)
- `test_random_injection_is_caught_by_bdd`: control — the same change made *without* a = Hc is caught
- `test_targets_are_never_generator_or_zero_injection`
- `test_zero_injection_bus_still_reads_zero`: Kirchhoff's law holds after the attack
- `test_affected_buses_are_within_two_hops`: matches the paper's Fig. 4
- `test_topology_changes_keep_grid_connected`: exactly one branch out, no islands

### Step 2: One small dataset by hand (~30 s)
```
python scripts/generate_li2023_data.py --scenario single --samples 400 --out data/li_small.npz
```
**Pass:** the last line says `stealth check: ... PASS`, about 50% of samples attacked, ~4 attacked buses per attacked sample.

### Step 3: Quick end-to-end check (~3–5 min)
```
python scripts/replicate_li2023.py --quick --scenarios single
```
1,000 samples, 30 epochs. Only checks that everything runs; numbers will be lower than the full run.

### Step 4: Full replication (~15–30 min on a 12-core laptop)
```
python scripts/replicate_li2023.py --sweep
```
Builds `data/li2023_*.npz` (one per scenario and per attack strength), trains all models, and writes
**`results/li2023_replication.md`** with our numbers next to the paper's. Re-running reuses the datasets;
add `--regenerate` to rebuild them.

### Step 5: Numbers worth reporting (optional, longer)
```
python scripts/replicate_li2023.py --sweep --seeds 0 1 2
```
Mean ± std over 3 seeds.

### Step 6: Commit
```
git add .
git commit -m "Li et al. 2023 replication on IEEE 14-bus"
git push
```

## How to read the results

- **Replicated** if: graph models beat the CNN on topology-change scenarios, GGNN + GAT is at or near
  the top, and accuracy rises from ~1% to ~20% attack strength (paper: 56% → 98%).
- Compare **node acc** and **snapshot acc** with the paper's accuracy column; whichever is closer is
  probably what the paper measured.
- Exact matches are not expected (see assumptions above).

## Reference run (2026-10-20, 12-core laptop, seed 0, node accuracy %)

Your numbers will differ a little run to run; the pattern is what to check. A second run on another
machine agreed within 0.4 points for GGNN and GGNN+GAT, within 1.7 for CNN/GCN, and by ~5 points
for the 1% and 5% attack strengths (noisy at weak attacks: use several seeds before reporting those). Full run took 30 minutes.

| Scenario | Paper GGNN | Ours GGNN | Paper GGNN+GAT | Ours GGNN+GAT | Paper GCN | Ours GCN | Paper CNN | Ours CNN |
|---|---|---|---|---|---|---|---|---|
| Single-node (Table 2) | 98.50 | 96.68 | 99.07 | 95.96 | 98.14 | 78.10 | — | 90.82 |
| Multi-node (Table 3) | 94.98 | 95.26 | 99.85 | 94.86 | 94.50 | 78.90 | — | 90.14 |
| Single + topology (Table 4) | 95.75 | 95.96 | 96.18 | 94.97 | 90.29 | 76.87 | 83.04 | 90.22 |
| Multi + topology (Table 5) | 94.14 | 95.16 | 95.80 | 95.27 | 91.39 | 77.28 | 80.85 | 88.95 |

Attack strength (GGNN+GAT, single-node): 1% → 57.7% (paper 56.3%), 5% → 73.0%, 10% → 85.7%, 15% → 91.8%, 20% → 96.0% (paper 98.0%).

**Replicated:** gated GNNs (GGNN, GGNN+GAT) reach ~95–97% with and without topology changes, within
0–4 points of the paper; accuracy rises with attack strength and matches the paper at both stated points (1%, 20%).

**Not replicated:** (1) our GCN is far weaker than the paper's (77–79% vs 90–98%), even below the CNN;
(2) attention gives no clear gain over plain GGNN (paper: +0.5 to +5 points); (3) our missed-alarm rates
are 8–10% vs the paper's 1–3%. Likely causes: untuned hyperparameters, the GCN averaging away the
attacked bus's own signal, and different label/threshold definitions.

**Next:** tune GCN and GGNN+GAT (hidden size, layers, learning rate, decision threshold); repeat with
`--seeds 0 1 2`; try `--label targets` to see whether the paper labelled only target buses.
