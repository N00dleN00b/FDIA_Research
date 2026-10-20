"""
Reproducible, parallel wrapper around the professor's generator, with progress output.

The original script (data_generation/data_generation.py) is NOT modified.
This wrapper (1) splits the work into small chunks with fixed seeds, so the same --seed
gives the same dataset no matter how many workers you use, (2) runs chunks in parallel,
(3) prints progress + time remaining, then merges everything into one .npz.

Usage:
    python scripts/generate_data.py --samples 3000 --seed 42
    python scripts/generate_data.py --samples 200 --out data/small.npz     # quick test set
    python scripts/generate_data.py --samples 3000 --workers 6             # use fewer cores
    python scripts/generate_data.py --merge-from <temp folder>             # only merge chunks left by an earlier run
    python scripts/generate_data.py --merge-from <temp folder>             # only merge chunks left by an earlier run

Press Ctrl+C once to stop; it may take a few seconds for all workers to exit.
"""
import argparse, contextlib, io, os, random, sys, tempfile, time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "data_generation"))
CHUNK = 25   # samples per job; fixed so results don't depend on --workers


def _worker(job):
    n, seed, tmp_path = job
    import warnings; warnings.filterwarnings("ignore")
    import data_generation as gen          # professor's module
    np.random.seed(seed); random.seed(seed)
    with contextlib.redirect_stdout(io.StringIO()):   # silence per-chunk prints
        gen.generate_large_grid_fdia_dataset(case_name="case118", num_samples=n,
                                             output_filename=str(tmp_path))
    return n, str(tmp_path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--samples", type=int, default=3000)
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1),
                   help="parallel processes (default: all cores but one, so the laptop stays usable)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default=str(ROOT / "data" / "fdia_dataset_ieee118.npz"))
    p.add_argument("--merge-from", default=None,
                   help="skip generation; merge chunk*.npz files already in this folder")
    a = p.parse_args()

    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    if a.merge_from:
        files = sorted(Path(a.merge_from).glob("chunk*.npz"))
        if not files:
            sys.exit(f"No chunk*.npz files found in {a.merge_from}")
        print(f"Merging {len(files)} chunks from {a.merge_from}")
        merge(files, out, Path(a.merge_from))
        return
    tmpdir = Path(tempfile.mkdtemp(prefix="fdia_gen_"))
    sizes = [CHUNK] * (a.samples // CHUNK) + ([a.samples % CHUNK] if a.samples % CHUNK else [])
    jobs = [(n, a.seed * 100003 + i, tmpdir / f"chunk{i:05d}.npz") for i, n in enumerate(sizes)]
    w = max(1, min(a.workers, len(jobs)))

    print(f"Generating {a.samples} samples in {len(jobs)} chunks on {w} worker(s), seed {a.seed}.")
    print("First progress line appears after the first chunk finishes (~30-60 s).", flush=True)
    t0, done = time.time(), 0
    with Pool(w) as pool:
        for n, _ in pool.imap_unordered(_worker, jobs):
            done += n
            el = time.time() - t0
            eta = el / done * (a.samples - done)
            print(f"  {done:5d}/{a.samples} samples  |  {el/60:4.1f} min elapsed  |  ~{eta/60:4.1f} min left", flush=True)

    merge([f for _, _, f in jobs], out, tmpdir)   # fixed order -> reproducible file
    print(f"Total time {(time.time() - t0)/60:.1f} min")


def merge(files, out, tmpdir):
    X, A, Y = [], [], []
    for f in files:
        with np.load(f) as d:               # 'with' closes the file; Windows can't delete open files
            X.append(d["x"]); A.append(d["a"]); Y.append(d["y"])
    np.savez_compressed(out, x=np.concatenate(X), a=np.concatenate(A), y=np.concatenate(Y))
    print(f"Saved {out}  ({sum(len(y) for y in Y)} samples)")
    for f in files:                         # clean up only after the merged file is safely written
        try:
            os.remove(f)
        except OSError:
            pass
    try:
        tmpdir.rmdir()
    except OSError:
        pass


if __name__ == "__main__":
    main()
