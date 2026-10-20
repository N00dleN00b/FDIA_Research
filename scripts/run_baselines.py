"""Train and evaluate detectors; write results/<tag>.json and results/<tag>.md.

Usage:
    python scripts/run_baselines.py                                   # all models, random split, 1 seed
    python scripts/run_baselines.py --split topology                  # test on unseen topologies
    python scripts/run_baselines.py --models gcn ggnn_gat --seeds 0 1 2 3 4
    python scripts/run_baselines.py --data data/small.npz --epochs 5  # quick smoke run
"""
import argparse, json, sys, time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from fdia.models import MODELS  # noqa: E402
from fdia.train import fit, prepare  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--data", default=str(ROOT / "data" / "fdia_dataset_ieee118.npz"))
p.add_argument("--models", nargs="+", default=list(MODELS), choices=list(MODELS))
p.add_argument("--split", default="random", choices=["random", "topology"])
p.add_argument("--seeds", nargs="+", type=int, default=[0])
p.add_argument("--epochs", type=int, default=100)
p.add_argument("--patience", type=int, default=15)
p.add_argument("--lr", type=float, default=1e-3)
p.add_argument("--bs", type=int, default=64)
p.add_argument("--hidden", type=int, default=64)
p.add_argument("--adj", default="log", choices=["log", "binary", "raw"])
p.add_argument("--frac", nargs=3, type=float, default=[0.7, 0.15, 0.15],
               help="train/val/test fractions, e.g. 0.675 0.075 0.25 for Li et al.'s 75/25 split")
p.add_argument("--norm", default="global", choices=["global", "per_bus"], help="feature scaling (per_bus = each bus vs its own normal range)")
p.add_argument("--tag", default=None)
a = p.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"
tag = a.tag or f"baselines_{a.split}"
print(f"device={device}  data={a.data}  split={a.split}  seeds={a.seeds}")

runs = {}
for seed in a.seeds:
    prep = prepare(a.data, a.split, seed, a.adj, tuple(a.frac), a.norm)
    tr, va, te = prep["splits"]
    if seed == a.seeds[0]:
        print(f"samples train/val/test = {len(tr)}/{len(va)}/{len(te)}  NaN rows fixed: {prep['n_nan']}")
    for m in a.models:
        _, test, hist = fit(m, prep, epochs=a.epochs, lr=a.lr, bs=a.bs, patience=a.patience,
                            seed=seed, device=device, hidden=a.hidden)
        runs.setdefault(m, []).append(test)
        print(f"{m:9s} seed {seed}: F1 {test['f1']:.3f}  AUC {test['roc_auc']:.3f}  "
              f"precision {test['precision']:.3f}  recall {test['recall']:.3f}  "
              f"MAR {test['MAR']:.3f}  FAR {test['FAR']:.4f}  acc {test['accuracy']:.4f}  ({test['train_seconds']}s)")

keys = ["f1", "roc_auc", "precision", "recall", "MAR", "FAR", "accuracy", "sample_accuracy", "sample_detection_rate", "sample_false_alarm_rate"]
summary = {m: {k: {"mean": float(np.mean([r[k] for r in rs])), "std": float(np.std([r[k] for r in rs]))} for k in keys}
           for m, rs in runs.items()}

out = ROOT / "results"; out.mkdir(exist_ok=True)
(out / f"{tag}.json").write_text(json.dumps({"config": vars(a), "summary": summary, "runs": runs}, indent=2))

fmt = lambda d: f"{d['mean']:.3f}" + (f" ± {d['std']:.3f}" if len(a.seeds) > 1 else "")
lines = [f"# Results: {tag}", "", f"Data `{Path(a.data).name}`, split `{a.split}`, seeds {a.seeds}, run {time.strftime('%Y-%m-%d %H:%M')}.", "",
         "Per-bus metrics on the test set. Positive class = attacked bus.", "",
         "| Model | F1 | ROC-AUC | Precision | Recall | MAR | FAR | Node acc | Snapshot detect rate | Snapshot false alarm |",
         "|---|---|---|---|---|---|---|---|---|---|"]
for m, s in summary.items():
    lines.append(f"| {m} | " + " | ".join(fmt(s[k]) for k in ["f1", "roc_auc", "precision", "recall", "MAR", "FAR", "accuracy", "sample_detection_rate", "sample_false_alarm_rate"]) + " |")
(out / f"{tag}.md").write_text("\n".join(lines) + "\n")
print("\n" + "\n".join(lines))
print(f"\nSaved results/{tag}.json and results/{tag}.md")
