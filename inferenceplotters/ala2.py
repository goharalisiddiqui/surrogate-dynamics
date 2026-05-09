import os
import pickle
import numpy as np
import torch
from pytorch_lightning.callbacks.prediction_writer import BasePredictionWriter
from matplotlib import pyplot as plt
from tqdm import tqdm

from collective_encoder.collective_encoder.testplotters.ala2 import ALA2plotter as ALA2TestPlotter
from collective_encoder.utils import compute_mfpt_matrix

from utils import calculate_transfer_matrix

from inferenceplotters.inference_utils.md_energy import plot_energies
from inferenceplotters.inference_utils.msm import (
    build_kde, 
    plot_kde,
    build_msm,
    check_correct_minima
)

ALA2_MINIMA_POSITIONS = np.array([
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

class Ala2Writer(ALA2TestPlotter, BasePredictionWriter):
    _IDENTIFIER = "Ala2InferencePlotter"
    _OPTIONAL_ARGS = ALA2TestPlotter._OPTIONAL_ARGS.copy()
    _OPTIONAL_ARGS.update({
        'write_interval': "epoch",
        'kde_sigma': 0.2, # bandwidth for KDE
        'kde_bin_x': 200,
        'kde_bin_y': 200,
        'minima_tolerance': 0.1, # tolerance for matching minima in recall
        'n_minima_compare': 3, # number of minima to compare in recall
        'n_BFGS_runs': 100, # number of BFGS runs to find local minima
    })
    
    def __init__(self, 
                 args, 
                 **kwargs):
        ALA2TestPlotter.__init__(self, args=args, **kwargs)
        BasePredictionWriter.__init__(self, self.write_interval)
        if self.n_minima_compare > len(ALA2_MINIMA_POSITIONS):
            self.log_warn(f"n_minima_compare ({self.n_minima_compare}) is greater "
                          f"than the number of reference minima ({len(ALA2_MINIMA_POSITIONS)}). "
                          f"Setting n_minima_compare to {len(ALA2_MINIMA_POSITIONS)}.")
            self.n_minima_compare = len(ALA2_MINIMA_POSITIONS)

    def output_msm(self, data):
        data = self.label_selector(data)
        data = self.cossin_resolver(data)
        phi, psi = data['phi_2'], data['psi_2']

        # Step 1: Build KDE
        kde = build_kde(phi, psi, self.kde_sigma)
        fig, ax = plot_kde(kde, self.kde_bin_x, self.kde_bin_y)
        
        # Step 2: Build MSM and find minima
        min_points, min_vals = build_msm(kde, 
                                        self.n_BFGS_runs, 
                                        self.minima_tolerance)
        self.log_result(f"Number of local minima found: {len(min_points)}")
        
        # Annotate minima on the KDE plot
        ax.scatter(min_points[:,0], min_points[:,1], 
            facecolors='w', edgecolors='r', s=200, 
            label='Local Minima')
        for i, (x, y) in enumerate(min_points):
            ax.text(x, y, str(i + 1), color='black', 
                    fontsize=8, ha='center', va='center')
        ax.legend(bbox_to_anchor=(1, 1.1))
        self.log_image(fig, "kde_estimation")
        
        # Step 3: Check correctness of identified minima and output MFPT matrix if correct minima are found
        n_comp = min(self.n_minima_compare, len(min_points))
        correct_minima, correct_dists, dists = check_correct_minima(
                            minima_model=min_points,
                            minima_ref=ALA2_MINIMA_POSITIONS[:n_comp],
                            tolerance=self.minima_tolerance)
        n_correct = len(correct_minima)

        self.log_result(f"MSM Metrics:")
        self.log_result(f"  Number of minima (model): {len(min_points)}")
        self.log_result(f"  Number of minima (ref): {n_comp}")
        self.log_result(f"  Distance between minima : {dists}")
        self.log_result(f"  Number of correctly identified minima (within {self.minima_tolerance} rad): {n_correct} out of {n_comp}")
        if len(correct_minima) > 1:
            (transition_counts, transition_matrix, mfpt_matrix
            ) = compute_mfpt_matrix(
                np.stack([phi, psi], axis=-1), 
                correct_minima, 
                lag=1)
            fig, ax = self.plot_matrix(transition_counts, 
                                       title="Transition Counts",
                                       tag = "Source_Target")
            self.log_image(fig, "transition_counts")
            fig, ax = self.plot_matrix(transition_matrix, 
                                       title="Transition Matrix",
                                       tag = "Source_Target")
            self.log_image(fig, "transition_matrix")
            fig, ax = self.plot_matrix(mfpt_matrix, 
                                       title="MFPT Matrix",
                                       tag = "Source_Target")
            self.log_image(fig, "mfpt_matrix")
            
        else:
            self.log_result("No correct minima found, skipping MFPT computation.")

    def output_traj_variables(self, data):
        lag = 1  # lag time in units of trajectory frames
        transfer_matrix = calculate_transfer_matrix(data, lag)
        
        # Eigendecomposition — eigenvalues may be complex; take real parts for sorting
        eigenvalues, eigenvectors = np.linalg.eig(transfer_matrix)

        # Sort descending by magnitude (slowest modes first)
        order = np.argsort(np.abs(eigenvalues))[::-1]
        eigenvalues  = eigenvalues[order]
        eigenvectors = eigenvectors[:, order]

        # Implied timescales in units of trajectory frames (skip stationary mode at index 0)
        its = -lag / np.log(np.abs(eigenvalues[1:]))
    
        self.log_result(eigenvalues, "Eigenvalues")
        self.log_result(eigenvectors, "Eigenvectors")
        self.log_result(its, "Implied Timescales")
        
    def write_on_epoch_end(self, trainer, pl_module, predictions, batch_indices):
        if len(predictions) > 1:
            self.raise_error("Expected predictions for a single batch, "
                             "but got multiple batches. Please ensure that "
                             "the DataLoader used for prediction has batch_size=1.")
        predictions = predictions[0]  # Always expecting a single batch

        for name in ['Predicted', 'True', 'Decoded']:
            if name in predictions:
                pred = self.label_selector(predictions[name])
                pred = self.cossin_resolver(pred)
                
                fig, _ = self.plot_2ddihedral(pred['phi_2'], pred['psi_2'])
                self.log_image(fig, f"dihedral_{name.lower()}")
                plt.close(fig)
        
        if 'Predicted' in predictions:
            self.output_msm(predictions['Predicted'])
        
        fig, ax = plot_energies(predictions)
        self.log_image(fig, "predicted_energies")
        
        if 'Latent' in predictions:
            latent = predictions['Latent'].cpu().numpy()
            self.output_traj_variables(latent)





    
    