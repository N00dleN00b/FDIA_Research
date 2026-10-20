import numpy as np
import pytest
import torch
from fdia.models import MODELS, build


@pytest.mark.parametrize("name", list(MODELS))
def test_forward_shape_and_backward(name):
    B, N, F = 3, 10, 4
    x = torch.randn(B, N, F)
    a = torch.eye(N).repeat(B, 1, 1) + torch.diag(torch.ones(N - 1), 1) + torch.diag(torch.ones(N - 1), -1)
    m = build(name, F, hidden=16)
    out = m(x, a)
    assert out.shape == (B, N)
    out.sum().backward()
    assert all(p.grad is not None for p in m.parameters() if p.requires_grad)


def test_gat_ignores_non_neighbours():
    """Changing a bus that is not connected to bus 0 must not change GAT's output at bus 0."""
    from fdia.models import GATLayer
    torch.manual_seed(0)
    N = 5
    a = torch.eye(N)[None]; a[0, 0, 1] = a[0, 1, 0] = 1
    layer = GATLayer(4, 8)
    x = torch.randn(1, N, 4)
    y1 = layer(x, a)[0, 0]
    x2 = x.clone(); x2[0, 4] += 10
    assert torch.allclose(y1, layer(x2, a)[0, 0])
