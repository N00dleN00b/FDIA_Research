"""Replicate Li, Wang & Lu (2023) on IEEE 14-bus and compare with the paper's numbers.

For each scenario: generate the dataset (if missing), train CNN, GCN, GGNN, GGNN+GAT with a
75/25 split (a small validation slice is taken from the 75% for early stopping), and write
results/li2023_replication.md with our numbers next to the paper's.

Usage:
    python scripts/replicate_li2023.py                         # all 4 scenarios, 1 seed
    python scripts/replicate_li2023.py --scenarios single      # just Table 2
    python scripts/replicate_li2023.py --sweep                 # also Fig. 13 (accuracy vs attack strength)
    python scripts/replicate_li2023.py --seeds 0 1 2           # mean +/- std over seeds
    python scripts/replicate_li2023.py --quick                 # 1,000 samples, 30 epochs: pipeline check only
"""
import argparse, json, sys, time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
from fdia.train import fit, prepare            # noqa: E402
from generate_li2023_data import build         # noqa: E402

# Li et al. (2023), IEEE 14-bus: (accuracy %, missed-alarm rate %)
PAPER = {
    "single":      {"gcn": (98.14, 1.70), "ggnn": (98.50, 1.08), "ggnn_gat": (99.07, 0.92)},                       # Table 2
    "multi":       {"gcn": (94.50, 3.54), "ggnn": (94.98, 2.69), "ggnn_gat": (99.85, 2.85)},                       # Table 3
    "single_topo": {"cnn": (83.04, 3.81), "gcn": (90.29, 3.08), "ggnn": (95.75, 1.89), "ggnn_gat": (96.18, 1.77)},  # Table 4
    "multi_topo":  {"cnn": (80.85, 2.15), "gcn": (91.39, 1.92), "ggnn": (94.14, 1.73), "ggnn_gat": (95.80, 1.19)},  # Table 5
}
PAPER_SWEEP = {0.01: 56.28, 0.20: 98.03}       # Fig. 13: only these two values are stated in the text
TITLES = {"single": "Single-node attack (Table 2)", "multi": "Multi-node attack (Table 3)",
          "single_topo": "Single-node attack + topology changes (Table 4)",
          "multi_topo": "Multi-node attack + topology changes (Table 5)"}
NAMES = {"cnn": "CNN", "gcn": "GCN", "ggnn": "GGNN", "ggnn_gat": "GGNN + GAT"}


def dataset(path, scenario, samples, strength, regenerate, workers):
    if regenerate or not path.exists():
        d = build(scenario, samples, strength, workers=workers)
        np.savez_compressed(path, **d)
        print(f"  saved {path}")
    else:
        print(f"  using existing {path} (pass --regenerate to rebuild)")
    return path


def evaluate(path, models, seeds, epochs, device):
    out = {m: [] for m in models}
    for seed in seeds:
        prep = prepare(path, "random", seed, "log", (0.675, 0.075, 0.25), "per_bus")
        for m in models:
            _, t, _ = fit(m, prep, epochs=epochs, patience=20, seed=seed, device=device, verbose=False)
            out[m].append(t)
            print(f"    {NAMES[m]:10s} seed {seed}: node acc {t['accuracy']:.2%}  MAR {t['MAR']:.2%}  "
                  f"F1 {t['f1']:.3f}  snapshot acc {t['sample_accuracy']:.2%}  ({t['train_seconds']}s)", flush=True)
    return out


def ms(runs, k, pct=True):
    v = np.array([r[k] for r in runs]) * (100 if pct else 1)
    return f"{v.mean():.2f}" + (f" ± {v.std():.2f}" if len(v) > 1 else "")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scenarios", nargs="+", default=list(PAPER), choices=list(PAPER))
    p.add_argument("--models", nargs="+", default=["cnn", "gcn", "ggnn", "ggnn_gat"])
    p.add_argument("--seeds", nargs="+", type=int, default=[0])
    p.add_argument("--samples", type=int, default=5600)
    p.add_argument("--epochs", type=int, default=150, help="Li et al. trained 150 epochs on 14-bus")
    p.add_argument("--strength", type=float, default=0.2)
    p.add_argument("--sweep", action="store_true", help="also run Fig. 13 attack-strength sweep")
    p.add_argument("--sweep-strengths", nargs="+", type=float, default=[0.01, 0.05, 0.10, 0.15, 0.20])
    p.add_argument("--regenerate", action="store_true")
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--quick", action="store_true", help="1,000 samples, 30 epochs: checks the pipeline, numbers not meaningful")
    a = p.parse_args()
    if a.quick:
        a.samples, a.epochs = 1000, 30
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tag = "_quick" if a.quick else ""
    t0 = time.time()

    results, md = {}, ["# Replication of Li, Wang & Lu (2023) on IEEE 14-bus", "",
                       f"Run {time.strftime('%Y-%m-%d %H:%M')}, {a.samples} samples per scenario, attack strength {a.strength:.0%}, "
                       f"split 67.5/7.5/25 (train/val/test; the paper used 75/25 with no validation set), per-bus feature scaling, {a.epochs} epochs max, seeds {a.seeds}.", "",
                       "Paper numbers are accuracy / missed-alarm rate (MAR) as printed in Tables 2-5. "
                       "The paper does not say whether accuracy is per bus or per snapshot, so both are shown.", ""]
    for sc in a.scenarios:
        print(f"\n=== {TITLES[sc]} ===")
        path = dataset(ROOT / "data" / f"li2023_{sc}{tag}.npz", sc, a.samples, a.strength, a.regenerate, a.workers)
        runs = evaluate(path, a.models, a.seeds, a.epochs, device)
        results[sc] = runs
        md += [f"## {TITLES[sc]}", "",
               "| Model | Paper acc % | Ours: node acc % | Ours: snapshot acc % | Paper MAR % | Ours: MAR % | Ours: F1 |",
               "|---|---|---|---|---|---|---|"]
        for m in a.models:
            pa, pm = PAPER[sc].get(m, (None, None))
            md.append(f"| {NAMES[m]} | {pa if pa is not None else '—'} | {ms(runs[m], 'accuracy')} | {ms(runs[m], 'sample_accuracy')} | "
                      f"{pm if pm is not None else '—'} | {ms(runs[m], 'MAR')} | {ms(runs[m], 'f1', False)} |")
        md.append("")

    if a.sweep:
        print("\n=== Attack strength sweep (Fig. 13) ===")
        md += ["## Detection vs attack strength (Fig. 13), single-node, GGNN + GAT", "",
               "| Attack strength | Paper acc % | Ours: node acc % | Ours: snapshot acc % | Ours: MAR % | Ours: F1 |", "|---|---|---|---|---|---|"]
        results["sweep"] = {}
        for s in a.sweep_strengths:
            print(f"  strength {s:.0%}")
            path = dataset(ROOT / "data" / f"li2023_single_s{int(round(s * 100)):02d}{tag}.npz", "single", a.samples, s, a.regenerate, a.workers)
            runs = evaluate(path, ["ggnn_gat"], a.seeds, a.epochs, device)["ggnn_gat"]
            results["sweep"][s] = runs
            pa = PAPER_SWEEP.get(round(s, 2))
            md.append(f"| {s:.0%} | {pa if pa is not None else '— (only in figure)'} | {ms(runs, 'accuracy')} | {ms(runs, 'sample_accuracy')} | "
                      f"{ms(runs, 'MAR')} | {ms(runs, 'f1', False)} |")
        md.append("")

    out = ROOT / "results"; out.mkdir(exist_ok=True)
    (out / f"li2023_replication{tag}.md").write_text("\n".join(md) + "\n")
    (out / f"li2023_replication{tag}.json").write_text(json.dumps({"config": vars(a), "results": results}, indent=2, default=str))
    print("\n" + "\n".join(md))
    print(f"Saved results/li2023_replication{tag}.md  (total {(time.time() - t0) / 60:.1f} min)")


if __name__ == "__main__":
    main()
