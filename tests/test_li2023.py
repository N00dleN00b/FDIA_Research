"""Checks that the Li et al. (2023)-style generator follows the paper's attack model.

Each test is one claim we make when presenting the replication.
"""
import numpy as np
import pytest

pp = pytest.importorskip("pandapower")
from fdia import li2023 as L  # noqa: E402


@pytest.fixture(scope="module")
def grid():
    return L.load_case("case14")


@pytest.fixture(scope="module")
def pair():
    """Same random draws, two labelings: 'targets' shows which buses were chosen, 'affected' which changed."""
    kw = dict(samples=80, mode="multi", topo_frac=0.3, seed=7, dc_only_features=True)   # skip slow AC power flow
    return L.generate(label="targets", **kw), L.generate(label="affected", **kw)


def test_case14_structure(grid):
    _, g = grid
    assert g.n == 14 and len(g.branches) == 20                 # 15 lines + 5 transformers, all in the graph
    assert g.zero_inj_buses == {6}                             # IEEE bus 7 has no load and no generator
    assert {0, 1, 2, 5, 7} <= g.gen_buses                      # slack, generators, synchronous condensers


def test_dc_model_matches_pandapower(grid):
    """Our B matrix reproduces pandapower's DC power flow: P = B theta."""
    net, g = grid
    pp.rundcpp(net)
    th = np.radians(net.res_bus.va_degree.values)
    P = -net.res_bus.p_mw.values / net.sn_mva
    assert np.allclose(L.b_matrix(g) @ th, P, atol=1e-9)


def test_attacks_are_stealthy(pair):
    """Paper Eq. 6: with a = Hc the bad-data-detection residual does not change."""
    _, d = pair
    att = d["attacked"]
    assert att.any()
    assert np.allclose(d["bdd_J_attacked"][att], d["bdd_J_clean"][att], rtol=0, atol=1e-6)


def test_random_injection_is_caught_by_bdd(grid):
    """Control: the same-sized change made WITHOUT a = Hc raises the residual (BDD works)."""
    _, g = grid
    rng = np.random.default_rng(0)
    B = L.b_matrix(g)
    Hk, H_full, keep = L.measurement_matrix(g)
    th = np.zeros(g.n); th[keep] = rng.normal(0, 0.1, len(keep))
    z = H_full @ th
    sigma = 0.01 * np.abs(z) + 1e-3
    _, J0 = L.wls_estimate(Hk, z, sigma)
    c = L.stealthy_attack(g, B, B @ th, [12], 0.2, rng)
    a = H_full @ c
    _, J_stealthy = L.wls_estimate(Hk, z + a, sigma)
    bad = np.zeros_like(a); bad[12] = a[12]                    # only the target's injection meter changed
    _, J_naive = L.wls_estimate(Hk, z + bad, sigma)
    assert abs(J_stealthy - J0) < 1e-6
    assert J_naive > J0 + 100


def test_targets_are_never_generator_or_zero_injection(grid, pair):
    _, g = grid
    d, _ = pair
    assert d["y"][:, sorted(g.protected)].sum() == 0


def test_zero_injection_bus_still_reads_zero(grid, pair):
    """Kirchhoff's law holds at the contact bus after the attack."""
    _, g = grid
    _, d = pair
    assert np.abs(d["x"][:, sorted(g.zero_inj_buses), 2]).max() < 1e-9


def test_affected_buses_are_within_two_hops(grid, pair):
    """Paper Sec. 2.8 / Fig. 4: an attack spreads to 1st- and 2nd-order neighbours only."""
    _, g = grid
    dt, da = pair
    for s in np.where(da["attacked"])[0]:
        A = da["a"][s]
        hop1 = (A > 0).astype(int)
        reach2 = ((np.eye(g.n, dtype=int) + hop1) @ (np.eye(g.n, dtype=int) + hop1)) > 0
        targets = np.where(dt["y"][s] == 1)[0]
        affected = np.where(da["y"][s] == 1)[0]
        assert set(targets) <= set(affected)
        allowed = set(np.where(reach2[targets].any(0))[0])
        assert set(affected) <= allowed


def test_topology_changes_keep_grid_connected(pair):
    _, d = pair
    ch = d["topology_changed"]
    assert 0 < ch.mean() < 1
    edges = (d["a"] > 0).sum((1, 2)) // 2
    assert (edges[~ch] == 20).all()                            # intact grid: 20 branches
    assert (edges[ch] == 19).all()                             # exactly one branch switched out
    assert not np.isnan(d["x"]).any()


def test_output_works_with_training_pipeline(tmp_path, pair):
    from fdia.train import fit, prepare
    _, d = pair
    f = tmp_path / "li.npz"
    np.savez_compressed(f, **d)
    prep = prepare(f, "random", 0, "log", (0.675, 0.075, 0.25))
    _, t, _ = fit("gcn", prep, epochs=2, patience=5, hidden=8, verbose=False)
    assert 0 <= t["accuracy"] <= 1
