import numpy as np
import pandapower as pp
import pandapower.networks as pn
import random

def generate_large_grid_fdia_dataset(
    case_name="case118",       
    num_samples=3000,
    topology_change_prob=0.4,   # 40% unlearned/dynamic topology shifts
    attack_prob=0.5,
    max_attack_power=0.02,     # 2% stealthy perturbation
    output_filename="fdia_dataset_ieee118.npz"
):
    """
    Generates an FDIA dataset for large power system benchmarks under AC power flow.
    """
    if case_name == "case118":
        base_net = pn.case118()
    elif case_name == "case300":
        base_net = pn.case300()
    else:
        raise ValueError(f"Unsupported network case: {case_name}")

    num_buses = len(base_net.bus)
    print(f"Starting data generation for {case_name} ({num_buses} buses)...")

    node_features_list = []
    adjacency_matrices_list = []
    labels_list = []

    valid_samples = 0

    while valid_samples < num_samples:
        # Load fresh instance of network
        net = pn.case118() if case_name == "case118" else pn.case300()

        # 1. Simulate High Volatility (Load/Renewable Fluctuations)
        load_scale = np.random.uniform(0.6, 1.4)
        load_noise = np.random.normal(1.0, 0.20, size=len(net.load))
        net.load['p_mw'] *= load_scale * np.maximum(load_noise, 0.05)
        net.load['q_mvar'] *= load_scale * np.maximum(load_noise, 0.05)

        if len(net.sgen) > 0:
            sgen_noise = np.random.normal(1.0, 0.25, size=len(net.sgen))
            net.sgen['p_mw'] *= np.maximum(sgen_noise, 0.0)

        # 2. Dynamic Topology Changes (N-1 to N-4 Contingencies)
        if np.random.rand() < topology_change_prob:
            active_lines = list(net.line.index[net.line['in_service']])
            num_trips = np.random.randint(1, 5)
            lines_to_trip = random.sample(active_lines, num_trips)
            for line_id in lines_to_trip:
                net.line.at[line_id, 'in_service'] = False

        # 3. Non-Linear AC Power Flow Execution
        try:
            pp.runpp(net, algorithm='nr', max_iteration=30)
        except Exception:
            # Skip non-converged power flow configurations
            continue

        # 4. Extract Bus Measurements [V, Theta, P, Q]
        v_mag = net.res_bus['vm_pu'].values
        v_ang = np.radians(net.res_bus['va_degree'].values)
        p_inj = net.res_bus['p_mw'].fillna(0).values / net.sn_mva
        q_inj = net.res_bus['q_mvar'].fillna(0).values / net.sn_mva

        node_features = np.column_stack([v_mag, v_ang, p_inj, q_inj])

        # Heavy-tailed sensor measurement noise (Student-t)
        sensor_noise = np.random.standard_t(df=4, size=node_features.shape) * 0.004
        node_features += sensor_noise

        # 5. Dynamic Adjacency Matrix (Admittance-weighted)
        adj_matrix = np.zeros((num_buses, num_buses), dtype=np.float32)
        active_lines = net.line[net.line['in_service']]
        
        for _, line in active_lines.iterrows():
            f = int(line['from_bus'])
            t = int(line['to_bus'])
            r = line['r_ohm_per_km'] * line['length_km']
            x = line['x_ohm_per_km'] * line['length_km']
            z = np.sqrt(r**2 + x**2) if (r**2 + x**2) > 0 else 0.001
            y = 1.0 / z
            adj_matrix[f, t] = y
            adj_matrix[t, f] = y

        labels = np.zeros(num_buses, dtype=int)

        # 6. Cluster-based Coordinated FDIA Attack
        if np.random.rand() < attack_prob:
            possible_centers = [b for b in range(num_buses) if b in net.load['bus'].values]
            if possible_centers:
                center_bus = np.random.choice(possible_centers)
                neighbors = np.where(adj_matrix[center_bus] > 0)[0].tolist()
                attack_pool = list(set([center_bus] + neighbors))
                
                num_targets = np.random.randint(1, len(attack_pool) + 1)
                attacked_buses = np.random.choice(attack_pool, size=num_targets, replace=False)

                for bus_idx in attacked_buses:
                    attack_mag = np.random.uniform(0.005, max_attack_power)
                    sign = np.random.choice([-1, 1])
                    
                    node_features[bus_idx, 2] += sign * attack_mag
                    node_features[bus_idx, 3] += sign * (attack_mag * 0.5)
                    labels[bus_idx] = 1

        node_features_list.append(node_features)
        adjacency_matrices_list.append(adj_matrix)
        labels_list.append(labels)

        valid_samples += 1
        if valid_samples % 500 == 0:
            print(f"Generated {valid_samples}/{num_samples} samples...")

    # 7. Save Dataset File
    X = np.array(node_features_list, dtype=np.float32)
    A = np.array(adjacency_matrices_list, dtype=np.float32)
    Y = np.array(labels_list, dtype=np.int64)

    np.savez_compressed(output_filename, x=X, a=A, y=Y)
    print(f"\nDone! Dataset saved to '{output_filename}'.")
    print(f"Features (X) shape: {X.shape}")
    print(f"Adjacency (A) shape: {A.shape}")
    print(f"Labels (Y) shape:    {Y.shape}")

# Execute script
if __name__ == "__main__":
    generate_large_grid_fdia_dataset(
        case_name="case118",
        num_samples=3000,
        output_filename="fdia_dataset_ieee118.npz"
    )