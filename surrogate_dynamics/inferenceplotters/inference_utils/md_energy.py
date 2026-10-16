import numpy as np

def calculate_bond_energies(predictions):
    bond_distances = predictions['Predicted']['bond_dist'].cpu().numpy()
    n_bonds = bond_distances.shape[1]
    bond_k = np.array([265265.6, 476976.0, 410032.0, 282001.0, 259408.0,
                        265265.6, 476976.0, 410032.0, 282001.0
                        ])  # k in kJ/(mol*nm^2) (AMBER03 FF)
    bond_r0 = np.array([0.15220, 0.12290, 0.13350, 0.14490, 0.15260,
                        0.15220, 0.12290, 0.13350, 0.14490
                        ])  # r0 in nm (AMBER03 FF)
    assert n_bonds == len(bond_k), "Number of bonds in predictions does not match length of bond_k"
    assert n_bonds == len(bond_r0), "Number of bonds in predictions does not match length of bond_r0"
    energies = 0.5 * bond_k[np.newaxis, :].repeat(bond_distances.shape[0], axis=0) \
                    * (bond_distances - bond_r0[np.newaxis, :].repeat(bond_distances.shape[0], axis=0))**2  # shape: (n_steps, n_bonds)
    bond_energies = np.sum(energies, axis=1)  # shape: (batch_size, n_steps)   
    return bond_energies

def calculate_angle_energies(predictions):
    angle_values = predictions['Predicted']['angle'].cpu().numpy()
    n_angles = angle_values.shape[1]
    angle_k = np.array([669.440, 585.760, 669.440, 418.400, 669.440, 
                        585.766, 527.184, 669.440, 585.760, 669.440, 418.400
                        ])  # k in kJ/(mol*rad^2) (AMBER03 FF)
    angle_theta0 = np.array([120.4, 116.6, 122.9, 121.9, 109.7, 
                                116.6, 111.1, 120.4, 116.6, 122.9, 121.9
                            ]) * (np.pi / 180.0)  # theta0 in rad (AMBER03 FF)
    angle_k = np.repeat(angle_k, repeats=2) # each angle is represented twice in the predictions
    angle_theta0 = np.repeat(angle_theta0, repeats=2) # each angle is represented twice in the predictions

    assert n_angles == len(angle_k), "Number of angles in predictions does not match length of angle_k"
    assert n_angles == len(angle_theta0), "Number of angles in predictions does not match length of angle_theta0"
    
    angle_k = angle_k[np.newaxis, :].repeat(angle_values.shape[0], axis=0)
    angle_theta0 = angle_theta0[np.newaxis, :].repeat(angle_values.shape[0], axis=0)
    
    energies = 0.5 * angle_k * (angle_values - angle_theta0)**2  # shape: (batch_size, n_angles, n_steps)
    angle_energies = np.sum(energies, axis=1)  # shape: (batch_size, n_steps)
    return angle_energies

def calculate_dihedral_energies(predictions):
    dihedral_cos = predictions['Predicted']['dihedral_cos'].cpu().numpy()
    dihedral_sin = predictions['Predicted']['dihedral_sin'].cpu().numpy()
    dihedral_angles = np.arctan2(dihedral_sin, dihedral_cos)  # shape: (batch_size, n_dihedrals, n_steps)
    n_dihedrals = dihedral_angles.shape[1]
    
    dihedral_k = np.array([5.0] * n_dihedrals)  # force constant in kcal/mol
    dihedral_n = np.array([3] * n_dihedrals)  # multiplicity
    dihedral_delta = np.array([0.0] * n_dihedrals)  # phase in rad
    
    assert n_dihedrals == len(dihedral_k), "Number of dihedrals in predictions does not match length of dihedral_k"
    assert n_dihedrals == len(dihedral_n), "Number of dihedrals in predictions does not match length of dihedral_n"
    assert n_dihedrals == len(dihedral_delta), "Number of dihedrals in predictions does not match length of dihedral_delta"
    
    dihedrals_k = dihedral_k[np.newaxis, :].repeat(dihedral_angles.shape[0], axis=0)
    dihedrals_n = dihedral_n[np.newaxis, :].repeat(dihedral_angles.shape[0], axis=0)
    dihedrals_delta = dihedral_delta[np.newaxis, :].repeat(dihedral_angles.shape[0], axis=0)
    energies = dihedrals_k * (1 + np.cos(dihedrals_n * dihedral_angles - dihedrals_delta))  # shape: (batch_size, n_dihedrals, n_steps)
    dihedral_energies = np.sum(energies, axis=1)  # shape: (batch_size, n_steps)
    return dihedral_energies

def calculate_total_energies(predictions):
    bond_energies = calculate_bond_energies(predictions)
    angle_energies = calculate_angle_energies(predictions)
    # dihedral_energies = calculate_dihedral_energies(predictions)

    total_energies = bond_energies + angle_energies #+ dihedral_energies  # shape: (batch_size, n_steps)
    return total_energies

def plot_energies(predictions):
    total_energies = calculate_total_energies(predictions)
    import matplotlib.pyplot as plt
    
    # plt.figure(figsize=(8,5))
    fig, ax = plt.subplots(figsize=(8,5))
    ax.plot(np.arange(total_energies.shape[0]), total_energies, label='Average Total Energy')
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Total Energy (kJ/mol)')
    ax.set_title('Total Energy over Time')
    ax.legend()
    plt.tight_layout()

    return fig, ax
