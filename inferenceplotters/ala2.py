import os
import pickle
import numpy as np
import torch
from pytorch_lightning.callbacks.prediction_writer import BasePredictionWriter
from matplotlib import pyplot as plt
from tqdm import tqdm

from collective_encoder.utils import compute_mfpt_matrix

def periodic_norm_distance(x1, x2, period=2*np.pi):
    """ Compute the periodic norm distance between two points x1 and x2 """
    # Minimum image convention
    d = (x1 - x2 + period/2) % period - period/2
    
    return np.linalg.norm(d)

def periodic_average(x, period=2*np.pi):
    """ Compute the periodic average of a list of 2D points x """
    # Minimum image convention
    x = np.array(x)
    x1 = np.arctan2(np.mean(np.sin(x[:,0])), np.mean(np.cos(x[:,0])))
    x2 = np.arctan2(np.mean(np.sin(x[:,1])), np.mean(np.cos(x[:,1])))
    return np.array([x1, x2])

class Ala2Writer(BasePredictionWriter):
    def __init__(self, output_dir, write_interval, config = {}):
        super().__init__(write_interval)
        self.output_dir = output_dir
        self.rama_minima_positions = np.array([
           [ -1.3144, -0.3401],
           [ -1.2843, 2.6713],
           [ -2.6904, 2.7835],
           [ -1.4946, -3.0109],
           [ -2.6308, -2.9918],
           [ 3.0341, 2.7583],
           [ 1.2012, 0.0580],
           [ 3.0745, -3.0108],
           [ 3.0320, -0.6649],
           [ 1.0283, -2.7108],
           [ 1.2054, 2.9642],
           [ 3.0375, -1.9775],

        ])
        self.kde_sigma = config.get("kde_sigma", 0.2) # bandwidth for KDE
        self.kde_bin_x = config.get("kde_bin_x", 200)
        self.kde_bin_y = config.get("kde_bin_y", 200)
        self.minima_tolerance = config.get("minima_tolerance", 1.0)
        self.n_minima_compare: int = config.get("n_minima_compare", 3)
        self.n_BFGS_runs: int = config.get("n_BFGS_runs", 100)
        assert self.n_minima_compare <= len(self.rama_minima_positions), "n_minima_compare exceeds number of reference minima"

    #### Ramachandran analysis start ####
    def build_msm(self, phi, psi):
        # Step 1
        from sklearn.neighbors import KernelDensity

        X = np.stack([phi, psi], axis=-1)
        kde = KernelDensity(kernel='gaussian', bandwidth=self.kde_sigma).fit(X) # bandwidth is sigma
        
        # Step 2
        from scipy.optimize import minimize
        def objective(x):
            return -kde.score_samples(x.reshape(1, -1))
        results = []
        for _ in range(self.n_BFGS_runs):
            x0 = np.random.uniform(-np.pi, np.pi, size=2)
            res = minimize(objective, x0, method='BFGS')
            results.append(res)
        min_points = np.array([r.x for r in results if r.success])
        # Remove points whole norm are closer than 'minima_tolerance'
        min_points_reduced = []
        for p in min_points:
            for q in min_points_reduced:
                if any(periodic_norm_distance(np.array(p), np.array(r)) <= self.minima_tolerance for r in q):
                    q.append(p)
                    break
            else:
                min_points_reduced.append([p])
        min_points = np.array([periodic_average(a) for a in min_points_reduced])
        min_vals = np.array([float(objective(p)[0]) for p in min_points])
        sorted_indices = np.argsort(min_vals)
        min_points = min_points[sorted_indices]
        min_vals = min_vals[sorted_indices]
        print("Number of local minima found:", len(min_points))
        return min_points, min_vals, kde

    def plot_kde(self, min_points, min_vals, kde):
        import matplotlib.pyplot as plt
        from matplotlib import cm

        x = np.linspace(-np.pi, np.pi, self.kde_bin_x) # grid points
        y = np.linspace(-np.pi, np.pi, self.kde_bin_y)
        Xgrid, Ygrid = np.meshgrid(x, y)

        # Free energy surface
        kbt = 0.415 # at 300K in kJ/mol
        fes = kbt * -kde.score_samples(np.vstack([Xgrid.ravel(), Ygrid.ravel()]).T).reshape(Xgrid.shape)
        fes -= fes.min() # set minimum to zero

        fig, ax = plt.subplots(1, 1, figsize=(7,5))
        cf = ax.contourf(Xgrid, Ygrid, fes, levels=30, cmap=cm.viridis)
        fig.colorbar(cf, ax=ax, label='Free Energy (kJ/mol)')
        ax.scatter(min_points[:,0], min_points[:,1], 
                   facecolors='w', edgecolors='r', s=200, 
                   label='Local Minima')
        for i, (x, y) in enumerate(min_points):
            ax.text(x, y, str(i + 1), color='black', fontsize=8, ha='center', va='center')
        ax.set_xlabel(r"$\Phi$")
        ax.set_ylabel(r"$\Psi$")
        ax.set_xlim([-np.pi, np.pi])
        ax.set_ylim([-np.pi, np.pi])
        ax.set_xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
        ax.set_xticklabels([r'$-\pi$', r'$-\pi/2$', '0', r'$\pi/2$', r'$\pi$'])
        ax.set_yticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
        ax.set_yticklabels([r'$-\pi$', r'$-\pi/2$', '0', r'$\pi/2$', r'$\pi$'])
        ax.legend(bbox_to_anchor=(1, 1.1))
        fig.savefig(self.output_dir + "/kde_ramachandran.png", dpi=300)
        plt.close()
        print(f"\n[{type(self).__name__}]: Saved KSE Ramachandran plot to {self.output_dir}/kde_ramachandran.png")

    def output_msm_recall(self, min_points):
        n_minima_model = len(min_points)
        n_minima_ref = len(self.rama_minima_positions)
        n_comp = min(self.n_minima_compare, n_minima_model)
        minima_model = min_points[:n_comp]
        minima_ref = self.rama_minima_positions[:n_comp]
        dists = np.linalg.norm(minima_model[:, None, :] - minima_ref[None, :, :], axis=-1)
        from scipy.optimize import linear_sum_assignment
        row_ind, col_ind = linear_sum_assignment(dists)
        matched_dists = dists[row_ind, col_ind]
        avg_dist = np.mean(matched_dists)
        idx_correct = matched_dists < self.minima_tolerance
        n_correct = np.sum(idx_correct)
        print(f"\n[{type(self).__name__}]: MSM Metrics:")
        print(f"  Number of minima (model): {n_minima_model}")
        print(f"  Number of minima (ref): {n_minima_ref}")
        print(f"  Distance between matched minima (top {n_comp}): {matched_dists}")
        print(f"  Average distance between matched minima (top {n_comp}): {avg_dist:.4f} rad")
        print(f"  Number of correctly identified minima (within {self.minima_tolerance} rad): {n_correct} out of {n_comp}")
        correct_minima = minima_model[idx_correct]
        return correct_minima

    def output_mfpt_matrix(self, phi, psi, minima):
        '''
        Compute and save Mean First Passage Time (MFPT) matrix between the given minima.
        Step 1: Construct an MSM from the trajectory data (phi, psi) on state values defined by the given minima using PyEMMA.
        Step 2: Compute the MFPT matrix between the states defined by the minima.
        '''

        lag_time = 1
        if minima.shape[0] < 2:
            print(f"\n[{type(self).__name__}]: Not enough minima to compute MFPT matrix. Skipping.")
            return None
        transition_counts, transition_matrix, mfpt_matrix = compute_mfpt_matrix(np.stack([phi, psi], axis=-1), minima, lag=lag_time)

        # Plot MFPT matrix
        plt.matshow(mfpt_matrix, cmap='viridis')
        plt.colorbar(label='MFPT')
        plt.title(f'MFPT Matrix (lag_time={lag_time})')
        plt.xlabel('Target State')
        plt.ylabel('Source State')
        plt.xticks(np.arange(len(minima)), [str(i+1) for i in range(len(minima))])
        plt.yticks(np.arange(len(minima)), [str(i+1) for i in range(len(minima))])
        plt.grid(False)
        plt.savefig(os.path.join(self.output_dir, "mfpt_matrix.png"), dpi=300)
        plt.close()
        
        
        # Plot Transition counts
        plt.matshow(transition_matrix, cmap='viridis')
        plt.colorbar(label='Transition Probabilities')
        plt.title(f'Transition Matrix (lag_time={lag_time})')
        plt.xlabel('To State')
        plt.ylabel('From State')
        plt.xticks(np.arange(len(minima)), [str(i+1) for i in range(len(minima))])
        plt.yticks(np.arange(len(minima)), [str(i+1) for i in range(len(minima))])
        plt.grid(False)
        plt.savefig(os.path.join(self.output_dir, "transition_matrix.png"), dpi=300)
        plt.close()

        mfpt_metrics = {
            'mfpt_matrix': mfpt_matrix,
            'transition_counts': transition_counts,
            'transition_matrix': transition_matrix,
            'lag_time': lag_time
        }
        np.savez(os.path.join(self.output_dir, "mfpt_metrics.npz"), **mfpt_metrics)
        
        print(f"\n[{type(self).__name__}]: MFPT Matrix:")
        print(f"  Shape: {mfpt_matrix.shape}")
        print(f"  Lag time: {lag_time}")
        print(f"  Transition counts:\n{transition_counts}")
        print(f"  MFPT Matrix:\n{mfpt_matrix}")
        
        return mfpt_matrix

    def output_msm_metrics(self, rama_data):
        assert 'Predicted' in rama_data, "Rama data must contain 'Predicted' key"
        phi, psi = rama_data['Predicted']
        min_points, min_vals, kde = self.build_msm(phi, psi)
        self.plot_kde(min_points, min_vals, kde)
        correct_minima = self.output_msm_recall(min_points)
        if len(correct_minima) > 0:
            self.output_mfpt_matrix(phi, psi, correct_minima)
        else:
            print(f"\n[{type(self).__name__}]: No correct minima found, skipping MFPT computation.")

    def extract_ramachandran(self, predictions):
        idx_phi = 6 # [1,3,4,5]
        idx_psi = 10 # [3,4,6,8]

        plt_data = {}

        for k in predictions.keys():
            dihedrals_cos = predictions[k]['dihedral_cos'].cpu().numpy()
            dihedrals_sin = predictions[k]['dihedral_sin'].cpu().numpy()
            phi = np.arctan2(dihedrals_sin[:,idx_phi], dihedrals_cos[:,idx_phi])
            psi = np.arctan2(dihedrals_sin[:,idx_psi], dihedrals_cos[:,idx_psi])
            plt_data[k] = (phi, psi)
        return plt_data
    
    def save_ramachandran(self, plt_data):
        np.savez(self.output_dir + "/ramachandran.npz", **plt_data)
        print(f"\n[{type(self).__name__}]: Saved Ramachandran data to {self.output_dir}/ramachandran.npz")
    
    def plot_ramachandran(self, plt_data):
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, len(plt_data), figsize=(5*len(plt_data),5))
        if len(plt_data) == 1:
            axes = [axes]

        for ax, (title, (phi, psi)) in zip(axes, plt_data.items()):
            ax.set_title(title)
            ax.scatter(phi, psi, s=1)
            ax.set_xlabel("Phi")
            ax.set_ylabel("Psi")
            ax.set_xlim([-np.pi, np.pi])
            ax.set_ylim([-np.pi, np.pi])
            ax.set_xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
            ax.set_xticklabels([r'$-\pi$', r'$-\pi/2$', '0', r'$\pi/2$', r'$\pi$'])
        fig.savefig(self.output_dir + "/ramachandran.png", dpi=300)
        plt.close()
        print(f"\n[{type(self).__name__}]: Saved Ramachandran plot to {self.output_dir}/ramachandran.png")
    
    #### Ramachandran analysis start ####
    #### Energy analysis start ####
    def calculate_bond_energies(self, predictions):
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

    def calculate_angle_energies(self, predictions):
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
    
    def calculate_dihedral_energies(self, predictions):
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
    
    def calculate_total_energies(self, predictions):
        bond_energies = self.calculate_bond_energies(predictions)
        angle_energies = self.calculate_angle_energies(predictions)
        # dihedral_energies = self.calculate_dihedral_energies(predictions)
        
        total_energies = bond_energies + angle_energies #+ dihedral_energies  # shape: (batch_size, n_steps)
        return total_energies
    
    def plot_energies(self, total_energies):
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(8,5))
        plt.plot(np.arange(total_energies.shape[0]), total_energies, label='Average Total Energy')
        plt.xlabel('Time Step')
        plt.ylabel('Total Energy (kJ/mol)')
        plt.title('Total Energy over Time')
        plt.legend()
        plt.savefig(self.output_dir + "/total_energy.png", dpi=300)
        plt.close()
        print(f"\n[{type(self).__name__}]: Saved Total Energy plot to {self.output_dir}/total_energy.png")
    
    #### Energy analysis end ####
    
    def write_on_epoch_end(self, trainer, pl_module, predictions, batch_indices):
        assert len(predictions) == 1, "Expecting a single batch of predictions"
        # for k, v in predictions[0].items():
        #     print(f"key: {k}, value shape: {v.shape}")
        predictions = predictions[0]  # Always expecting a single batch
        rama_data = self.extract_ramachandran(predictions)
        with open(self.output_dir + "/rama_data.pkl", "wb") as f:
            pickle.dump(rama_data, f)
        self.plot_ramachandran(rama_data)
        self.save_ramachandran(rama_data)
        self.output_msm_metrics(rama_data)
        
        total_energies = self.calculate_total_energies(predictions)
        self.plot_energies(total_energies)





    
    