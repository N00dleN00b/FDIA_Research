from fdia.train import fit, prepare


def test_training_learns_toy_attack(toy_dataset):
    prep = prepare(toy_dataset, "random", seed=0)
    _, test, _ = fit("gcn", prep, epochs=40, patience=40, bs=8, lr=1e-2, hidden=16, verbose=False)
    assert test["roc_auc"] > 0.9
