import os
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
            [-1.4946284 , -3.01094883],
            [ 1.20116279,  0.05799981],
            [ 3.0375002 , -1.97747476],
            [-1.31441039, -0.34013426],
            [ 3.03196092, -0.66485791],
            [ 1.02826208, -2.71076106],
            [ 1.75571281,  2.66234648],
            [ 1.20543751,  2.96417112],
            [ 3.07450587, -3.01082907],
            [ 3.03413505,  2.75833817],
            [-1.28428108,  2.67126545],
            [-2.69038585,  2.78350498],
            [ 1.0292704 , -1.70034419],
            [-2.63080522, -2.9917996 ],
        ])
        self.kde_sigma = config.get("kde_sigma", 0.2) # bandwidth for KDE
        self.kde_bin_x = config.get("kde_bin_x", 200)
        self.kde_bin_y = config.get("kde_bin_y", 200)
        self.minima_tolerance = config.get("minima_tolerance", 1.0)
        self.n_minima_compare: int = config.get("n_minima_compare", 3)
        self.n_BFGS_runs: int = config.get("n_BFGS_runs", 100)
        assert self.n_minima_compare <= len(self.rama_minima_positions), "n_minima_compare exceeds number of reference minima"

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
    
    def write_on_epoch_end(self, trainer, pl_module, predictions, batch_indices):
        assert len(predictions) == 1, "Expecting a single batch of predictions"
        # for k, v in predictions[0].items():
        #     print(f"key: {k}, value shape: {v.shape}")
        predictions = predictions[0]  # Always expecting a single batch
        rama_data = self.extract_ramachandran(predictions)
        self.plot_ramachandran(rama_data)
        self.save_ramachandran(rama_data)
        self.output_msm_metrics(rama_data)

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



    
    