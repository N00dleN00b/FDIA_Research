"""Training loop shared by all models."""
import time
import numpy as np
import torch
import torch.nn.functional as F

from . import data as D
from .metrics import node_metrics
from .models import build


def set_seed(seed):
    np.random.seed(seed); torch.manual_seed(seed)


def prepare(path, split="random", seed=0, adj_weighting="log", frac=(0.7, 0.15, 0.15), norm="global"):
    X, A, Y = D.load_npz(path)
    X, n_nan = D.clean(X)
    tr, va, te = D.make_splits(A, split, seed, frac)
    mean, std = D.feature_stats(X, tr, norm)
    Xn = D.normalize_features(X, mean, std)
    Ahat = D.normalize_adjacency(A, adj_weighting)
    return dict(X=Xn, A=Ahat, Y=Y, splits=(tr, va, te), n_nan=n_nan)


@torch.no_grad()
def predict(model, X, A, idx, device, bs=256):
    model.eval(); out = []
    for i in range(0, len(idx), bs):
        b = idx[i:i + bs]
        out.append(model(torch.from_numpy(X[b]).to(device), torch.from_numpy(A[b]).to(device)).cpu().numpy())
    return np.concatenate(out)


def fit(model_name, prep, epochs=100, lr=1e-3, bs=64, patience=15, seed=0, device="cpu", verbose=True, **model_kw):
    set_seed(seed)
    X, A, Y = prep["X"], prep["A"], prep["Y"]
    tr, va, te = prep["splits"]
    model = build(model_name, X.shape[-1], **model_kw).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    pos = Y[tr].mean()
    pos_weight = torch.tensor((1 - pos) / max(pos, 1e-6), device=device)

    best_f1, best_state, wait, history = -1.0, None, 0, []
    t0 = time.time()
    rng = np.random.default_rng(seed)
    for ep in range(epochs):
        model.train(); perm = rng.permutation(tr); losses = []
        for i in range(0, len(perm), bs):
            b = perm[i:i + bs]
            x = torch.from_numpy(X[b]).to(device); a = torch.from_numpy(A[b]).to(device)
            y = torch.from_numpy(Y[b]).float().to(device)
            loss = F.binary_cross_entropy_with_logits(model(x, a), y, pos_weight=pos_weight)
            opt.zero_grad(); loss.backward(); opt.step(); losses.append(loss.item())
        val = node_metrics(predict(model, X, A, va, device), Y[va])
        history.append({"epoch": ep, "loss": float(np.mean(losses)), "val_f1": val["f1"], "val_auc": val["roc_auc"]})
        if val["f1"] > best_f1:
            best_f1, wait = val["f1"], 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
        if verbose and ep % 10 == 0:
            print(f"  [{model_name}] epoch {ep:3d}  loss {np.mean(losses):.4f}  val F1 {val['f1']:.3f}  val AUC {val['roc_auc']:.3f}")
        if wait >= patience:
            break
    model.load_state_dict(best_state)
    test = node_metrics(predict(model, X, A, te, device), Y[te])
    test["epochs_run"] = ep + 1
    test["train_seconds"] = round(time.time() - t0, 1)
    test["params"] = sum(p.numel() for p in model.parameters())
    return model, test, history
