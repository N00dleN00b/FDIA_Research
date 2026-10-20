"""Li-style FDIA dataset (Li, Wang & Lu, Energy 263 (2023) 125865, Sec. 2.2 and 2.8).

What this reproduces from the paper:
  * DC state estimation  z = H x + e,  x = bus voltage angles (slack removed)
  * Stealthy attack  a = H c  (Liu et al. 2011): the bad-data-detection residual is unchanged
  * Attack centred on a target bus; injections change at the target, its 1st-order and
    (through zero-injection compensation) 2nd-order neighbours
  * Targets are never generator buses or zero-injection ("contact") buses
  * Zero-injection buses keep reading exactly 0 after the attack (Kirchhoff's law holds)
  * Optional topology changes (a line or transformer switched out), default 10% of samples
  * Node features [V, theta, P, Q], admittance-weighted adjacency, per-bus labels

What is NOT in the paper and had to be assumed (see docs/REPLICATION_LI2023.md):
  * NYISO hourly load data -> a synthetic daily load curve (two peaks) with per-load noise
  * "Attack power" -> relative change of the target bus's injection measurement
  * Noise level, exact multi-node pattern, how labels treat ripple buses
"""
import copy
from dataclasses import dataclass, field

import numpy as np

# pandapower is imported lazily inside functions so tests that do not need it stay fast


# ----------------------------------------------------------------------------- network
@dataclass
class Grid:
    n: int                       # buses
    slack: int
    gen_buses: set
    zero_inj_buses: set
    load_buses: list
    branches: list = field(default_factory=list)   # (from, to, x_pu, |z|_pu, kind, idx)

    @property
    def protected(self):
        return self.gen_buses | self.zero_inj_buses | {self.slack}

    @property
    def eligible_targets(self):
        return [b for b in range(self.n) if b not in self.protected]


def load_case(case="case14"):
    import pandapower as pp
    import pandapower.networks as pn
    net = getattr(pn, case)()
    pp.rundcpp(net)
    lookup = net._pd2ppc_lookups["bus"]
    assert list(lookup[: len(net.bus)]) == list(range(len(net.bus))), "bus indices must be 0..N-1"

    from pandapower.pypower.idx_brch import F_BUS, T_BUS, BR_R, BR_X, TAP
    br = net._ppc["branch"]
    n_lines = len(net.line)
    branches = []
    for k, b in enumerate(br):
        f, t = int(b[F_BUS].real), int(b[T_BUS].real)
        tap = b[TAP].real or 1.0
        x = b[BR_X].real * tap
        z = abs(complex(b[BR_R].real, b[BR_X].real))
        kind = "line" if k < n_lines else "trafo"
        branches.append((f, t, x, z, kind, k))

    gen = set(net.gen.bus.astype(int)) | set(net.ext_grid.bus.astype(int))
    load = set(net.load.bus.astype(int))
    zero = {b for b in range(len(net.bus)) if b not in gen and b not in load
            and not (len(net.sgen) and b in set(net.sgen.bus.astype(int)))}
    return net, Grid(n=len(net.bus), slack=int(net.ext_grid.bus.iloc[0]), gen_buses=gen,
                     zero_inj_buses=zero, load_buses=sorted(load), branches=branches)


def b_matrix(grid, out_of_service=()):
    """DC susceptance matrix B (N x N): P = B theta."""
    B = np.zeros((grid.n, grid.n))
    for f, t, x, _, _, k in grid.branches:
        if k in out_of_service:
            continue
        y = 1.0 / x
        B[f, f] += y; B[t, t] += y; B[f, t] -= y; B[t, f] -= y
    return B


def flow_matrix(grid, out_of_service=()):
    """Line-flow measurement rows: P_ft = (theta_f - theta_t) / x."""
    rows = []
    for f, t, x, _, _, k in grid.branches:
        if k in out_of_service:
            continue
        r = np.zeros(grid.n); r[f] = 1 / x; r[t] = -1 / x
        rows.append(r)
    return np.array(rows)


def adjacency(grid, out_of_service=()):
    A = np.zeros((grid.n, grid.n), np.float32)
    for f, t, _, z, _, k in grid.branches:
        if k in out_of_service:
            continue
        A[f, t] += 1.0 / max(z, 1e-6); A[t, f] += 1.0 / max(z, 1e-6)
    return A


def is_connected(grid, out_of_service=()):
    adj = {i: set() for i in range(grid.n)}
    for f, t, *_rest, k in grid.branches:
        if k not in out_of_service:
            adj[f].add(t); adj[t].add(f)
    seen, stack = {0}, [0]
    while stack:
        for j in adj[stack.pop()]:
            if j not in seen:
                seen.add(j); stack.append(j)
    return len(seen) == grid.n


# ----------------------------------------------------------------------------- state estimation
def measurement_matrix(grid, out_of_service=()):
    """H for [bus injections; line flows] w.r.t. non-slack angles."""
    keep = [i for i in range(grid.n) if i != grid.slack]
    H_full = np.vstack([b_matrix(grid, out_of_service), flow_matrix(grid, out_of_service)])
    return H_full[:, keep], H_full, keep


def wls_estimate(H, z, sigma):
    W = 1.0 / sigma ** 2
    G = H.T @ (H * W[:, None])
    x = np.linalg.solve(G, H.T @ (W * z))
    r = (z - H @ x) / sigma
    return x, float(r @ r)          # normalized residual J(x) = chi-square statistic


# ----------------------------------------------------------------------------- attack
def stealthy_attack(grid, B, P_true, targets, strength, rng):
    """State perturbation c (N,) with c[slack] = 0 so that a = H c is stealthy.

    For each target t: c_t = sign * strength * |P_t| / B_tt  (=> ~strength relative change at t).
    Zero-injection buses get the c that keeps their injection exactly 0 (KCL), which spreads
    the attack to 2nd-order neighbours, as described in Li et al. Sec. 2.8 / Fig. 4.
    """
    c = np.zeros(grid.n)
    for t in targets:
        mag = strength * max(abs(P_true[t]), 0.01)
        c[t] = rng.choice([-1.0, 1.0]) * mag / B[t, t]
    Z = sorted(grid.zero_inj_buses - {grid.slack})
    if Z:
        others = [i for i in range(grid.n) if i not in Z]
        c[Z] = -np.linalg.solve(B[np.ix_(Z, Z)], B[np.ix_(Z, others)] @ c[others])
    c[grid.slack] = 0.0
    return c


def neighbours(grid, b, out_of_service=()):
    s = set()
    for f, t, *_rest, k in grid.branches:
        if k in out_of_service:
            continue
        if f == b: s.add(t)
        if t == b: s.add(f)
    return s


# ----------------------------------------------------------------------------- dataset
def daily_load_factor(hour):
    """Synthetic two-peak daily curve in [~0.62, ~1.0] standing in for NYISO hourly data."""
    h = hour % 24
    return 0.62 + 0.20 * np.exp(-((h - 9) / 2.5) ** 2) + 0.38 * np.exp(-((h - 19) / 3.0) ** 2)


def generate(case="case14", samples=5600, attack_prob=0.5, mode="single", strength=(0.2, 0.2),
             topo_frac=0.0, noise=0.01, label="affected", dc_only_features=False, seed=0,
             progress=None):
    """Return dict with x (S,N,4), a (S,N,N), y (S,N) and diagnostics.

    mode      : 'single' (one target bus) or 'multi' (two target buses)
    strength  : (lo, hi) relative change of each target's injection measurement
    topo_frac : fraction of samples with one branch switched out (graph stays connected)
    label     : 'affected' = every bus whose measurement the attack changed (default)
                'targets'  = only the chosen target buses
    dc_only_features : if True, V=1 and Q=0 (detector sees only the DC quantities P, theta)
    """
    import pandapower as pp
    rng = np.random.default_rng(seed)
    net0, grid = load_case(case)
    base_p = net0.load.p_mw.values.copy(); base_q = net0.load.q_mvar.values.copy()
    base_g = net0.gen.p_mw.values.copy()
    branch_ids = [b[-1] for b in grid.branches]
    eligible = grid.eligible_targets

    X = np.zeros((samples, grid.n, 4), np.float32)
    A = np.zeros((samples, grid.n, grid.n), np.float32)
    Y = np.zeros((samples, grid.n), np.int64)
    J_clean = np.zeros(samples); J_att = np.zeros(samples)
    topo_changed = np.zeros(samples, bool); attacked = np.zeros(samples, bool)

    s = 0
    net = net0                                   # modified in place and reset each sample (much faster than deepcopy)
    line_ok = net.line.in_service.values.copy(); trafo_ok = net.trafo.in_service.values.copy()
    while s < samples:
        net.line["in_service"] = line_ok; net.trafo["in_service"] = trafo_ok
        # 1) operating point: daily curve x per-load noise; generators follow total load
        f = daily_load_factor(rng.uniform(0, 24)) * rng.uniform(0.95, 1.05)
        mult = f * rng.normal(1.0, 0.05, size=len(base_p)).clip(0.7, 1.3)
        net.load.p_mw = base_p * mult; net.load.q_mvar = base_q * mult
        net.gen.p_mw = base_g * f

        # 2) topology change
        oos = ()
        if rng.random() < topo_frac:
            for _ in range(20):
                k = int(rng.choice(branch_ids))
                if is_connected(grid, (k,)):
                    oos = (k,); break
            n_lines = len(net.line)
            k = oos[0]
            if k < n_lines: net.line.at[k, "in_service"] = False
            else: net.trafo.at[k - n_lines, "in_service"] = False

        # 3) DC power flow (state): solve B theta = P with the slack balancing the system
        B = b_matrix(grid, oos)
        P_sched = np.zeros(grid.n)
        np.add.at(P_sched, net.load.bus.values.astype(int), -net.load.p_mw.values / net.sn_mva)
        np.add.at(P_sched, net.gen.bus.values.astype(int), net.gen.p_mw.values / net.sn_mva)
        P_sched[grid.slack] = 0.0
        P_sched[grid.slack] = -P_sched.sum()
        keep_ = [i for i in range(grid.n) if i != grid.slack]
        th = np.zeros(grid.n)
        th[keep_] = np.linalg.solve(B[np.ix_(keep_, keep_)], P_sched[keep_])
        P_true = B @ th
        #    AC power flow only for the unattacked V, Q features
        if not dc_only_features:
            try:
                pp.runpp(net, max_iteration=30)
            except Exception:
                continue

        # 4) measurements z = H x + e, then WLS state estimation + bad-data residual
        Hk, H_full, keep = measurement_matrix(grid, oos)
        z_true = H_full @ th
        sigma = noise * np.abs(z_true) + 1e-3
        e = rng.normal(0, sigma)
        Zb = sorted(grid.zero_inj_buses)
        e[Zb] = 0.0; sigma[Zb] = 1e-4          # zero-injection buses: virtual measurement, exact
        z = z_true + e
        _, J_clean[s] = wls_estimate(Hk, z, sigma)

        # 5) attack a = H c
        y = np.zeros(grid.n, np.int64)
        if rng.random() < attack_prob:
            k = 1 if mode == "single" else 2
            targets = list(rng.choice(eligible, size=min(k, len(eligible)), replace=False))
            c = stealthy_attack(grid, B, P_true, targets, rng.uniform(*strength), rng)
            a = H_full @ c
            z = z + a
            dP = a[: grid.n]
            if label == "affected":
                y[np.abs(dP) > 1e-9] = 1
            else:
                y[targets] = 1
            attacked[s] = True
        x_hat, J_att[s] = wls_estimate(Hk, z, sigma)

        theta_hat = np.zeros(grid.n); theta_hat[keep] = x_hat
        P_meas = z[: grid.n]
        if dc_only_features:
            V = np.ones(grid.n); Q = np.zeros(grid.n)
        else:
            V = net.res_bus.vm_pu.values + rng.normal(0, 0.002, grid.n)
            Qt = -net.res_bus.q_mvar.values / net.sn_mva
            Q = Qt + rng.normal(0, noise * np.abs(Qt) + 1e-3)
        X[s] = np.column_stack([V, theta_hat, P_meas, Q])
        A[s] = adjacency(grid, oos)
        Y[s] = y
        topo_changed[s] = bool(oos)
        s += 1
        if progress and s % 500 == 0:
            progress(s, samples)

    return dict(x=X, a=A, y=Y, bdd_J_clean=J_clean, bdd_J_attacked=J_att,
                topology_changed=topo_changed, attacked=attacked)
