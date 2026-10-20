# Results: baselines_random

Data `fdia_dataset_ieee118.npz`, split `random`, seeds [0], run 2026-10-12 23:03.

Per-bus metrics on the test set. Positive class = attacked bus.

| Model | F1 | ROC-AUC | Precision | Recall | MAR | FAR | Node acc | Snapshot detect rate | Snapshot false alarm |
|---|---|---|---|---|---|---|---|---|---|
| mlp | 0.031 | 0.569 | 0.017 | 0.180 | 0.820 | 0.106 | 0.887 | 1.000 | 1.000 |
| gcn | 0.025 | 0.591 | 0.013 | 0.524 | 0.476 | 0.403 | 0.596 | 1.000 | 1.000 |
| ggnn | 0.033 | 0.572 | 0.017 | 0.245 | 0.755 | 0.138 | 0.856 | 1.000 | 1.000 |
| ggnn_gat | 0.026 | 0.594 | 0.014 | 0.334 | 0.666 | 0.241 | 0.755 | 1.000 | 1.000 |
