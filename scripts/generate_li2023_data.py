"""Generate a Li et al. (2023)-style FDIA dataset (DC model, stealthy a = Hc attacks).

Scenarios match the paper's IEEE 14-bus experiments:
    single       Table 2  single-node attack, fixed topology
    multi        Table 3  multi-node attack, fixed topology
    single_topo  Table 4  single-node attack, 10% of samples with a topology change
    multi_topo   Table 5  multi-node attack, 10% of samples with a topology change

Usage:
    python scripts/generate_li2023_data.py --scenario single
    python scripts/generate_li2023_data.py --scenario multi_topo --samples 5600
    python scripts/generate_li2023_data.py --scenario single --strength 0.01 --out data/li2023_s001.npz
    python scripts/generate_li2023_data.py --scenario single --samples 200 --out data/li_small.npz   # quick test

Output (.npz) has the same x / a / y format as the main dataset, plus diagnostics:
    bdd_J_clean, bdd_J_attacked : bad-data-detection residual before/after the attack (identical = stealthy)
    attacked, topology_changed  : per-sample flags
"""
import argparse, os, sys, time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

SCENARIOS = {
    "single":      dict(mode="single", topo_frac=0.0),
    "multi":       dict(mode="multi",  topo_frac=0.0),
    "single_topo": dict(mode="single", topo_frac=0.1),
    "multi_topo":  dict(mode="multi",  topo_frac=0.1),
}
CHUNK = 200   # samples per job; fixed so the dataset does not depend on --workers


def _worker(job):
    import warnings; warnings.filterwarnings("ignore")
    from fdia import li2023
    n, kw = job
    return li2023.generate(samples=n, **kw)


def build(scenario, samples=5600, strength=0.2, seed=42, workers=None, case="case14",
          noise=0.01, label="affected", dc_only=False, verbose=True):
    cfg = dict(SCENARIOS[scenario], case=case, strength=(strength, strength), noise=noise,
               label=label, dc_only_features=dc_only)
    sizes = [CHUNK] * (samples // CHUNK) + ([samples % CHUNK] if samples % CHUNK else [])
    jobs = [(n, dict(cfg, seed=seed * 100003 + i)) for i, n in enumerate(sizes)]
    workers = max(1, min(workers or (os.cpu_count() or 2) - 1, len(jobs)))
    if verbose:
        print(f"[{scenario}] {samples} samples, attack strength {strength:.0%}, {len(jobs)} chunks on {workers} worker(s)", flush=True)
    t0, done, parts = time.time(), 0, [None] * len(jobs)
    with Pool(workers) as pool:
        for i, part in enumerate(pool.imap(_worker, jobs)):     # imap keeps order -> reproducible
            parts[i] = part; done += len(part["y"])
            if verbose:
                print(f"  {done:5d}/{samples}  ({time.time() - t0:.0f}s)", flush=True)
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scenario", default="single", choices=list(SCENARIOS))
    p.add_argument("--samples", type=int, default=5600, help="Li et al. used 5,600 for IEEE 14-bus")
    p.add_argument("--strength", type=float, default=0.2, help="attack power: relative change of the target's injection")
    p.add_argument("--noise", type=float, default=0.01, help="relative measurement noise std")
    p.add_argument("--label", default="affected", choices=["affected", "targets"])
    p.add_argument("--dc-only", action="store_true", help="features use only DC quantities (V=1, Q=0)")
    p.add_argument("--case", default="case14")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--out", default=None)
    a = p.parse_args()

    out = Path(a.out or ROOT / "data" / f"li2023_{a.scenario}.npz")
    out.parent.mkdir(parents=True, exist_ok=True)
    d = build(a.scenario, a.samples, a.strength, a.seed, a.workers, a.case, a.noise, a.label, a.dc_only)
    np.savez_compressed(out, **d)

    att = d["attacked"]
    dJ = np.abs(d["bdd_J_attacked"] - d["bdd_J_clean"])[att]
    print(f"Saved {out}")
    print(f"  attacked samples: {att.mean():.1%}   topology changed: {d['topology_changed'].mean():.1%}")
    print(f"  attacked buses per attacked sample: {d['y'][att].sum(1).mean():.2f} of {d['y'].shape[1]}")
    print(f"  stealth check: max change in bad-data residual = {dJ.max():.2e}  "
          f"({'PASS: attacks invisible to bad-data detection' if dJ.max() < 1e-6 else 'FAIL'})")


if __name__ == "__main__":
    main()
