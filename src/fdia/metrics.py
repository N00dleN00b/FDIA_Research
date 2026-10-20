"""Metrics for imbalanced per-bus detection (~1% positives): accuracy alone is misleading."""
import numpy as np


def node_metrics(scores, y, threshold=0.0):
    """scores: logits (S, N); y: labels (S, N). Positive = attacked bus."""
    pred = (scores > threshold).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum()); tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum()); fn = int(((pred == 0) & (y == 1)).sum())
    prec = tp / max(tp + fp, 1); rec = tp / max(tp + fn, 1)
    out = {
        "accuracy": (tp + tn) / max(tp + tn + fp + fn, 1),
        "precision": prec,
        "recall": rec,
        "f1": 2 * prec * rec / max(prec + rec, 1e-12),
        "MAR": fn / max(tp + fn, 1),          # missed-alarm rate (Li et al.)
        "FAR": fp / max(fp + tn, 1),          # false-alarm rate
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
    }
    # Snapshot level: "was anything in this sample attacked?"
    s_pred = pred.max(1); s_true = y.max(1)
    out["sample_accuracy"] = float((s_pred == s_true).mean())
    out["sample_detection_rate"] = float(s_pred[s_true == 1].mean()) if (s_true == 1).any() else float("nan")
    out["sample_false_alarm_rate"] = float(s_pred[s_true == 0].mean()) if (s_true == 0).any() else float("nan")
    out["roc_auc"] = roc_auc(scores.ravel(), y.ravel())
    return out


def roc_auc(scores, y):
    """Threshold-free ranking quality (Mann-Whitney U)."""
    pos, neg = scores[y == 1], scores[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    ranks = np.empty(len(scores)); order = scores.argsort()
    ranks[order] = np.arange(1, len(scores) + 1)
    return float((ranks[y == 1].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))
