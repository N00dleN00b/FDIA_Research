# Results: baselines_topology

Data `fdia_dataset_ieee118.npz`, split `topology`, seeds [0], run 2026-10-12 23:07.

Per-bus metrics on the test set. Positive class = attacked bus.

| Model | F1 | ROC-AUC | Precision | Recall | MAR | FAR | Node acc | Snapshot detect rate | Snapshot false alarm |
|---|---|---|---|---|---|---|---|---|---|
| mlp | 0.028 | 0.557 | 0.015 | 0.376 | 0.624 | 0.289 | 0.707 | 1.000 | 1.000 |
| gcn | 0.027 | 0.579 | 0.014 | 0.652 | 0.348 | 0.540 | 0.462 | 1.000 | 1.000 |
| ggnn | 0.026 | 0.572 | 0.013 | 0.480 | 0.520 | 0.407 | 0.592 | 1.000 | 1.000 |
| ggnn_gat | 0.027 | 0.582 | 0.014 | 0.696 | 0.304 | 0.573 | 0.430 | 1.000 | 1.000 |
