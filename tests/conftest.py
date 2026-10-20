import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


@pytest.fixture
def toy_dataset(tmp_path):
    """Small synthetic dataset with the same format as the real one (no power flow needed)."""
    rng = np.random.default_rng(0)
    S, N = 40, 10
    A = np.zeros((S, N, N), np.float32)
    for i in range(N - 1):
        A[:, i, i + 1] = A[:, i + 1, i] = rng.uniform(0.01, 0.5)
    A[S // 2:, 3, 4] = A[S // 2:, 4, 3] = 0          # second topology: one line tripped
    X = rng.normal(size=(S, N, 4)).astype(np.float32)
    Y = np.zeros((S, N), np.int64)
    Y[::2, 2] = 1
    X[Y == 1, 2] += 3.0                                # make attacks learnable
    path = tmp_path / "toy.npz"
    np.savez_compressed(path, x=X, a=A, y=Y)
    return path
