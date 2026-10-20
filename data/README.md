# Data

The main dataset `fdia_dataset_ieee118.npz` (~10 MB) **is committed**, so `git pull` gets it. Scratch files like `small.npz` are git-ignored.

| File | How to make it | Notes |
|---|---|---|
| `fdia_dataset_ieee118.npz` | `python scripts/generate_data.py --samples 3000 --seed 42` | Main dataset, ~20–40 min on a laptop |
| `small.npz` | `python scripts/generate_data.py --samples 200 --out data/small.npz` | Quick tests, ~2 min |

Contents of each `.npz`:

- `x` (S, 118, 4): per bus `[V (pu), theta (rad), P (pu), Q (pu)]`
- `a` (S, 118, 118): adjacency weighted by line admittance 1/|z| (1/ohm); 0 = no in-service line
- `y` (S, 118): 1 = this bus's measurements were falsified, 0 = normal

Generator: `data_generation/data_generation.py` (professor's original, unmodified).
`scripts/generate_data.py` only adds a fixed seed and parallel workers.
