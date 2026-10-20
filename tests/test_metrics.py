import numpy as np
from fdia.metrics import node_metrics


def test_perfect_prediction():
    y = np.array([[0, 1, 0], [0, 0, 0]])
    m = node_metrics(np.where(y == 1, 5.0, -5.0), y)
    assert m["f1"] == 1 and m["MAR"] == 0 and m["FAR"] == 0 and m["roc_auc"] == 1


def test_all_normal_has_high_accuracy_but_zero_f1():
    """The reason we never report accuracy alone."""
    y = np.zeros((10, 100), int); y[:, 0] = 1
    m = node_metrics(np.full(y.shape, -1.0), y)
    assert m["accuracy"] == 0.99 and m["f1"] == 0 and m["MAR"] == 1
