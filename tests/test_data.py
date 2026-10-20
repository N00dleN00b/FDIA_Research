import numpy as np
from fdia import data as D


def test_load_shapes(toy_dataset):
    X, A, Y = D.load_npz(toy_dataset)
    assert X.shape == (40, 10, 4) and A.shape == (40, 10, 10) and Y.shape == (40, 10)


def test_clean_removes_nan():
    X = np.ones((2, 3, 4), np.float32); X[0, 1, 0] = np.nan
    Xc, n = D.clean(X)
    assert n == 1 and not np.isnan(Xc).any()


def test_random_split_is_partition(toy_dataset):
    _, A, _ = D.load_npz(toy_dataset)
    tr, va, te = D.make_splits(A, "random", seed=0)
    allidx = np.concatenate([tr, va, te])
    assert sorted(allidx) == list(range(len(A)))


def test_topology_split_has_no_leakage(toy_dataset):
    _, A, _ = D.load_npz(toy_dataset)
    tr, va, te = D.make_splits(A, "topology", seed=0)
    sig = lambda idx: {D.topology_signature(A[i]) for i in idx}
    assert sig(tr).isdisjoint(sig(te)) or len(te) == 0
    assert sorted(np.concatenate([tr, va, te])) == list(range(len(A)))


def test_normalized_adjacency_symmetric_with_self_loops(toy_dataset):
    _, A, _ = D.load_npz(toy_dataset)
    Ah = D.normalize_adjacency(A)
    assert np.allclose(Ah, Ah.transpose(0, 2, 1))
    assert (np.diagonal(Ah, axis1=1, axis2=2) > 0).all()
