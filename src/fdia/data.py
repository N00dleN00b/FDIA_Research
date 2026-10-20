"""Loading, cleaning, splitting and normalizing the FDIA dataset.

Dataset file (.npz) from data_generation.py:
    x : (S, 118, 4)   per-bus [V (pu), theta (rad), P (pu), Q (pu)]
    a : (S, 118, 118) admittance-weighted adjacency (1/ohm), 0 = no line
    y : (S, 118)      1 = bus measurements falsified, 0 = normal
"""
import numpy as np

FEATURES = ["V", "theta", "P", "Q"]


def load_npz(path):
    d = np.load(path)
    return d["x"].astype(np.float32), d["a"].astype(np.float32), d["y"].astype(np.int64)


def clean(X):
    """Islanded buses come back NaN from pandapower (V, theta). Replace with 0 and report."""
    nan_rows = np.isnan(X).any(-1)
    return np.nan_to_num(X, nan=0.0), int(nan_rows.sum())


def topology_signature(a):
    return (a > 0).tobytes()


def make_splits(A, mode="random", seed=0, frac=(0.7, 0.15, 0.15)):
    """Return (train_idx, val_idx, test_idx).

    random   : shuffle samples.
    topology : group samples by which lines are in service. The base (intact) grid is
               kept in train; every other topology lands wholly in one split, so test
               topologies are never seen in training.
    """
    rng = np.random.default_rng(seed)
    S = len(A)
    if mode == "random":
        idx = rng.permutation(S)
        n1, n2 = int(frac[0] * S), int((frac[0] + frac[1]) * S)
        return idx[:n1], idx[n1:n2], idx[n2:]
    if mode != "topology":
        raise ValueError(mode)

    groups = {}
    for i, a in enumerate(A):
        groups.setdefault(topology_signature(a), []).append(i)
    base = max(groups, key=lambda k: len(groups[k]))       # the intact grid
    others = [k for k in groups if k != base]
    rng.shuffle(others)
    train, val, test = list(groups[base]), [], []
    n_tr, n_va = frac[0] * S, frac[1] * S
    for k in others:
        tgt = train if len(train) < n_tr else (val if len(val) < n_va else test)
        tgt.extend(groups[k])
    return np.array(train), np.array(val), np.array(test)


def feature_stats(X, idx, mode="global"):
    """Mean/std from training samples only.

    global  : one mean/std per feature, shared by all buses (original behaviour).
    per_bus : mean/std per (bus, feature). The models cannot tell buses apart, so this is how
              they learn each bus's normal range (e.g. its own P/Q ratio). Constant features
              (std ~ 0, e.g. a bus that always reads 0) are left unscaled.
    """
    if mode == "global":
        flat = X[idx].reshape(-1, X.shape[-1])
        return flat.mean(0), flat.std(0) + 1e-8
    if mode == "per_bus":
        sd = X[idx].std(0)
        return X[idx].mean(0), np.where(sd < 1e-6, 1.0, sd)
    raise ValueError(mode)


def normalize_features(X, mean, std):
    return ((X - mean) / std).astype(np.float32)


def normalize_adjacency(A, weighting="log"):
    """Kipf & Welling renormalization: D^-1/2 (A' + I) D^-1/2.

    weighting: 'log'    -> A' = log(1 + A)  (admittances span orders of magnitude)
               'binary' -> A' = 1[A > 0]
               'raw'    -> A' = A
    """
    if weighting == "log":
        W = np.log1p(A)
    elif weighting == "binary":
        W = (A > 0).astype(np.float32)
    elif weighting == "raw":
        W = A.copy()
    else:
        raise ValueError(weighting)
    W = W + np.eye(A.shape[-1], dtype=np.float32)[None]
    dinv = 1.0 / np.sqrt(W.sum(-1))
    return (W * dinv[:, :, None] * dinv[:, None, :]).astype(np.float32)


def summarize(X, A, Y):
    deg = (A > 0).sum(-1)
    return {
        "samples": int(len(Y)),
        "buses": int(Y.shape[1]),
        "attacked_sample_rate": float((Y.sum(1) > 0).mean()),
        "attacked_node_rate": float(Y.mean()),
        "buses_attacked_per_attacked_sample": float(Y.sum(1)[Y.sum(1) > 0].mean()) if (Y.sum(1) > 0).any() else 0.0,
        "nan_node_rows": int(np.isnan(X).any(-1).sum()),
        "samples_with_nan": int(np.isnan(X).any((1, 2)).sum()),
        "unique_topologies": int(len({topology_signature(a) for a in A})),
        "edges_min": int((A > 0).sum((1, 2)).min() // 2),
        "edges_max": int((A > 0).sum((1, 2)).max() // 2),
        "isolated_buses_per_sample_max": int((deg == 0).sum(1).max()),
        "symmetric_adjacency": bool(np.allclose(A, A.transpose(0, 2, 1))),
    }
