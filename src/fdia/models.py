"""Detectors. All take x: (B, N, F) and a_hat: (B, N, N) and return per-node logits (B, N)."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    """Topology-blind baseline: each bus classified from its own 4 features only."""
    def __init__(self, in_dim, hidden=64, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(dropout),
                                 nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, x, a):
        return self.net(x).squeeze(-1)


class CNN(nn.Module):
    """Topology-blind 1-D CNN over the bus axis (the 'data-driven CNN' baseline in Li et al. 2023).
    Ignores the adjacency matrix: buses are treated as a fixed-order sequence."""
    def __init__(self, in_dim, hidden=64, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(nn.Conv1d(in_dim, hidden, 3, padding=1), nn.ReLU(), nn.Dropout(dropout),
                                 nn.Conv1d(hidden, hidden, 3, padding=1), nn.ReLU(),
                                 nn.Conv1d(hidden, 1, 1))

    def forward(self, x, a):
        return self.net(x.transpose(1, 2)).squeeze(1)


class GCN(nn.Module):
    """Kipf & Welling (ICLR 2017): H' = sigma(A_hat H W). a must already be renormalized."""
    def __init__(self, in_dim, hidden=64, layers=3, dropout=0.2):
        super().__init__()
        dims = [in_dim] + [hidden] * layers
        self.lins = nn.ModuleList(nn.Linear(dims[i], dims[i + 1]) for i in range(layers))
        self.out = nn.Linear(hidden + in_dim, 1)   # skip from raw features keeps the bus's own reading
        self.dropout = dropout

    def forward(self, x, a):
        h = x
        for lin in self.lins:
            h = F.dropout(F.relu(a @ lin(h)), self.dropout, self.training)
        return self.out(torch.cat([h, x], -1)).squeeze(-1)


class GGNNLayer(nn.Module):
    """Li et al. (2023) Eqs. 7-11: neighbour message a_i = sum_j A_ij W h_j + b, then GRU update."""
    def __init__(self, hidden, steps=3):
        super().__init__()
        self.msg = nn.Linear(hidden, hidden)
        self.gru = nn.GRUCell(hidden, hidden)
        self.steps = steps

    def forward(self, h, a):
        B, N, H = h.shape
        for _ in range(self.steps):
            m = a @ self.msg(h)
            h = self.gru(m.reshape(B * N, H), h.reshape(B * N, H)).reshape(B, N, H)
        return h


class GATLayer(nn.Module):
    """Li et al. (2023) Eqs. 12-13 (Velickovic et al. attention), restricted to graph neighbours."""
    def __init__(self, in_dim, out_dim, heads=1):
        super().__init__()
        self.W = nn.Linear(in_dim, out_dim * heads, bias=False)
        self.a_src = nn.Parameter(torch.randn(heads, out_dim) * 0.1)
        self.a_dst = nn.Parameter(torch.randn(heads, out_dim) * 0.1)
        self.heads, self.out_dim = heads, out_dim

    def forward(self, h, a):
        B, N, _ = h.shape
        Wh = self.W(h).view(B, N, self.heads, self.out_dim)
        e = (Wh * self.a_dst).sum(-1).unsqueeze(2) + (Wh * self.a_src).sum(-1).unsqueeze(1)  # B,N,N,K
        e = F.leaky_relu(e, 0.2).masked_fill(~(a > 0).unsqueeze(-1), float("-inf"))
        alpha = torch.softmax(e, dim=2)
        return torch.einsum("bijk,bjkf->bikf", alpha, Wh).mean(2)   # average heads (Eq. 13)


class GGNN(nn.Module):
    """GGNN without attention (Li et al. ablation)."""
    def __init__(self, in_dim, hidden=64, steps=3):
        super().__init__()
        self.inp = nn.Linear(in_dim, hidden)
        self.ggnn = GGNNLayer(hidden, steps)
        self.out = nn.Linear(hidden, 1)

    def forward(self, x, a):
        return self.out(F.leaky_relu(self.ggnn(self.inp(x), a))).squeeze(-1)


class GGNN_GAT(nn.Module):
    """Li et al. (2023) detector: GGNN -> LeakyReLU -> GAT -> LeakyReLU -> Dense (Eqs. 16-18)."""
    def __init__(self, in_dim, hidden=64, steps=3, heads=1):
        super().__init__()
        self.inp = nn.Linear(in_dim, hidden)
        self.ggnn = GGNNLayer(hidden, steps)
        self.gat = GATLayer(hidden, hidden, heads)
        self.out = nn.Linear(hidden, 1)

    def forward(self, x, a):
        h = F.leaky_relu(self.ggnn(self.inp(x), a))
        h = F.leaky_relu(self.gat(h, a))
        return self.out(h).squeeze(-1)


MODELS = {"mlp": MLP, "cnn": CNN, "gcn": GCN, "ggnn": GGNN, "ggnn_gat": GGNN_GAT}


def build(name, in_dim, **kw):
    return MODELS[name](in_dim, **kw)
