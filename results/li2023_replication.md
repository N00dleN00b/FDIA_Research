# Replication of Li, Wang & Lu (2023) on IEEE 14-bus

Run 2026-10-19 23:28, 5600 samples per scenario, attack strength 20%, split 67.5/7.5/25 (train/val/test; the paper used 75/25 with no validation set), per-bus feature scaling, 150 epochs max, seeds [0].

Paper numbers are accuracy / missed-alarm rate (MAR) as printed in Tables 2-5. The paper does not say whether accuracy is per bus or per snapshot, so both are shown.

## Single-node attack (Table 2)

| Model | Paper acc % | Ours: node acc % | Ours: snapshot acc % | Paper MAR % | Ours: MAR % | Ours: F1 |
|---|---|---|---|---|---|---|
| CNN | — | 90.82 | 68.57 | — | 26.03 | 0.70 |
| GCN | 98.14 | 78.10 | 66.79 | 1.7 | 36.84 | 0.45 |
| GGNN | 98.5 | 96.68 | 91.36 | 1.08 | 9.60 | 0.89 |
| GGNN + GAT | 99.07 | 95.96 | 93.07 | 0.92 | 8.68 | 0.87 |

## Multi-node attack (Table 3)

| Model | Paper acc % | Ours: node acc % | Ours: snapshot acc % | Paper MAR % | Ours: MAR % | Ours: F1 |
|---|---|---|---|---|---|---|
| CNN | — | 90.14 | 84.50 | — | 24.36 | 0.79 |
| GCN | 94.5 | 78.90 | 76.36 | 3.54 | 24.51 | 0.64 |
| GGNN | 94.98 | 95.26 | 94.21 | 2.69 | 7.88 | 0.91 |
| GGNN + GAT | 99.85 | 94.86 | 94.14 | 2.85 | 8.54 | 0.90 |

## Single-node attack + topology changes (Table 4)

| Model | Paper acc % | Ours: node acc % | Ours: snapshot acc % | Paper MAR % | Ours: MAR % | Ours: F1 |
|---|---|---|---|---|---|---|
| CNN | 83.04 | 90.22 | 68.00 | 3.81 | 25.74 | 0.68 |
| GCN | 90.29 | 76.87 | 64.93 | 3.08 | 33.71 | 0.44 |
| GGNN | 95.75 | 95.96 | 88.64 | 1.89 | 10.35 | 0.86 |
| GGNN + GAT | 96.18 | 94.97 | 88.93 | 1.77 | 8.31 | 0.83 |

## Multi-node attack + topology changes (Table 5)

| Model | Paper acc % | Ours: node acc % | Ours: snapshot acc % | Paper MAR % | Ours: MAR % | Ours: F1 |
|---|---|---|---|---|---|---|
| CNN | 80.85 | 88.95 | 76.79 | 2.15 | 22.65 | 0.76 |
| GCN | 91.39 | 77.28 | 72.07 | 1.92 | 22.71 | 0.61 |
| GGNN | 94.14 | 95.16 | 94.36 | 1.73 | 8.69 | 0.90 |
| GGNN + GAT | 95.8 | 95.27 | 96.14 | 1.19 | 9.86 | 0.90 |

## Detection vs attack strength (Fig. 13), single-node, GGNN + GAT

| Attack strength | Paper acc % | Ours: node acc % | Ours: snapshot acc % | Ours: MAR % | Ours: F1 |
|---|---|---|---|---|---|
| 1% | 56.28 | 57.70 | 49.71 | 38.94 | 0.29 |
| 5% | — (only in figure) | 72.95 | 51.50 | 41.15 | 0.38 |
| 10% | — (only in figure) | 85.67 | 71.57 | 30.87 | 0.58 |
| 15% | — (only in figure) | 91.77 | 84.29 | 16.50 | 0.74 |
| 20% | 98.03 | 95.96 | 93.07 | 8.68 | 0.87 |

