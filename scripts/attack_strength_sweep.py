"""Positive control + attack-strength sweep (mirrors Li et al. 2023, Fig. 13).

Takes the CLEAN snapshots (no attacked bus) from an existing dataset and injects attacks
the same way data_generation.py does (a load bus + random subset of its neighbours,
P += s*m, Q += s*m/2), but at several attack magnitudes m. If detection rises with m,
the test bed works and weak results at the original 0.005-0.02 pu are a property of the data.

Usage:
    python scripts/attack_strength_sweep.py --mags 0.01 0.05 0.1 0.2 --models mlp ggnn_gat
"""
import argparse, json, sys, tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from fdia import data as D  # noqa: E402
from fdia.models import MODELS  # noqa: E402
from fdia.train import fit, prepare  # noqa: E402


def inject(X, A, mag_lo, mag_hi, attack_prob, rng, load_buses):
    X = X.copy(); Y = np.zeros(X.shape[:2], np.int64)
    for s in range(len(X)):
        if rng.random() >= attack_prob:
            continue
        c = rng.choice(load_buses)
        pool = list({c, *np.where(A[s, c] > 0)[0].tolist()})
        tgt = rng.choice(pool, size=rng.integers(1, len(pool) + 1), replace=False)
        for b in tgt:
            m = rng.uniform(mag_lo, mag_hi) * rng.choice([-1, 1])
            X[s, b, 2] += m; X[s, b, 3] += m / 2; Y[s, b] = 1
    return X, Y


p = argparse.ArgumentParser()
p.add_argument("--data", default=str(ROOT / "data" / "fdia_dataset_ieee118.npz"))
p.add_argument("--mags", nargs="+", type=float, default=[0.02, 0.05, 0.1, 0.2])
p.add_argument("--models", nargs="+", default=["mlp", "ggnn_gat"], choices=list(MODELS))
p.add_argument("--epochs", type=int, default=60)
p.add_argument("--seed", type=int, default=0)
a = p.parse_args()

import pandapower.networks as pn
load_buses = sorted(set(pn.case118().load.bus.astype(int)))

X, A, Y = D.load_npz(a.data)
clean = Y.sum(1) == 0
Xc, Ac = X[clean], A[clean]
print(f"clean snapshots available: {clean.sum()}")

rows = []
for mag in a.mags:
    rng = np.random.default_rng(a.seed)
    Xa, Ya = inject(Xc, Ac, mag / 4, mag, 0.5, rng, load_buses)   # magnitudes in [mag/4, mag], like 0.005-0.02
    with tempfile.TemporaryDirectory() as td:
        f = Path(td) / "sweep.npz"; np.savez_compressed(f, x=Xa, a=Ac, y=Ya)
        prep = prepare(f, "random", a.seed)
        for m in a.models:
            _, t, _ = fit(m, prep, epochs=a.epochs, patience=12, seed=a.seed, verbose=False)
            rows.append({"max_attack_pu": mag, "model": m, **{k: t[k] for k in ["f1", "roc_auc", "precision", "recall", "MAR", "FAR"]}})
            print(f"max attack {mag:5.3f} pu  {m:9s}  F1 {t['f1']:.3f}  AUC {t['roc_auc']:.3f}  recall {t['recall']:.3f}  FAR {t['FAR']:.4f}")

out = ROOT / "results"
(out / "attack_strength_sweep.json").write_text(json.dumps(rows, indent=2))
lines = ["# Attack-strength sweep (positive control)", "",
         "Clean snapshots from the 3,000-sample dataset, re-attacked at each magnitude (range = [max/4, max] pu, same pattern as the generator). Random split, seed 0.", "",
         "| Max attack (pu) | Model | F1 | ROC-AUC | Recall | FAR |", "|---|---|---|---|---|---|"]
lines += [f"| {r['max_attack_pu']} | {r['model']} | {r['f1']:.3f} | {r['roc_auc']:.3f} | {r['recall']:.3f} | {r['FAR']:.4f} |" for r in rows]
(out / "attack_strength_sweep.md").write_text("\n".join(lines) + "\n")
print("\nSaved results/attack_strength_sweep.md")
