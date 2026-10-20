"""Sanity-check a dataset file and save a summary to results/.

Usage:
    python scripts/check_dataset.py --data data/fdia_dataset_ieee118.npz
"""
import argparse, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from fdia import data as D  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--data", default=str(ROOT / "data" / "fdia_dataset_ieee118.npz"))
p.add_argument("--out", default=str(ROOT / "results" / "dataset_summary.json"))
a = p.parse_args()

X, A, Y = D.load_npz(a.data)
s = D.summarize(X, A, Y)
s["shapes"] = {"x": list(X.shape), "a": list(A.shape), "y": list(Y.shape)}
import numpy as np
Xc, _ = D.clean(X)
s["feature_mean"] = dict(zip(D.FEATURES, np.round(Xc.reshape(-1, 4).mean(0), 4).tolist()))
s["feature_std"] = dict(zip(D.FEATURES, np.round(Xc.reshape(-1, 4).std(0), 4).tolist()))

checks = {
    "x has 4 features": X.shape[-1] == 4,
    "a is square and matches buses": A.shape[1] == A.shape[2] == X.shape[1],
    "y matches x": Y.shape == X.shape[:2],
    "labels are 0/1": set(np.unique(Y)) <= {0, 1},
    "adjacency symmetric": s["symmetric_adjacency"],
    "no NaN features": s["nan_node_rows"] == 0,
    "no isolated buses": s["isolated_buses_per_sample_max"] == 0,
}
s["checks"] = checks
Path(a.out).parent.mkdir(parents=True, exist_ok=True)
Path(a.out).write_text(json.dumps(s, indent=2))

print(json.dumps({k: v for k, v in s.items() if k != "checks"}, indent=2))
print("\nChecks:")
for k, ok in checks.items():
    print(f"  [{'PASS' if ok else 'WARN'}] {k}")
print(f"\nSaved {a.out}")
