import sys
sys.path.append("/mnt/projects/sne/Gohar/12.PROJECT_Surrogate_Dynamics/surrogate-dynamics/")
import os
from tqdm import tqdm
import numpy as np
import torch

torch.set_default_dtype(torch.float64)

from dataloaders.sequence import SequenceDataModule

MD_PATH = "/mnt/projects/sne/Gohar/12.PROJECT_Surrogate_Dynamics/surrogate-dynamics/data_generation/ala2/solvated/300K"
CG = 4
STEPS=50000

dm_config = {
    "input_chunk_length": STEPS,
    "output_chunk_length": 0,
    "datareader_type": "XTC_CHUNKS_CG",
    "datareader_args": {
        "tprfile": f"{MD_PATH}/initialization/tpr_initial.tpr",
        "xtcfile": f"{MD_PATH}/ala2_100ns/md.xtc",
        "selection": "(resname ALA or resname ACE or resname NME) and not element H",
        "cg_window": CG,
        "parallel": True
    },
    # "dataset_type": "GRAPH",
    "dataset_type": "GRAPH_LATENT",
    "dataset_args": {
        "encoder_name": "BGE",
        "encoder_ckpt": "../train_runs/encoder/bge/run_1/checkpoints/best_updated.ckpt",
        "precompute_graphs": False,
    },
    "train_size": 1,
    "batch_size": 1,
    "validation_size": 0,
    "val_batch_size": 0,
    "num_workers": 1,
    "n_seq_per_sample": 1,
    "norm_type": "minmax",
    "verbose": False,
}

dm = SequenceDataModule(args=dm_config)
ds = dm.get_train_dataset()
data_points = np.array([ds[i][0].cpu().numpy() for i in tqdm(range(len(ds)))])

lag = 1

# Paired snapshots at lag time
X_t   = data_points[:-lag]   # shape (T-lag, d)
X_t1  = data_points[lag:]    # shape (T-lag, d)

# Instantaneous (C0) and time-lagged (Ctau) correlation matrices
C0   = X_t.T @ X_t    # (d, d)
Ctau = X_t.T @ X_t1   # (d, d)

# Transfer matrix K = C0^{-1} Ctau via least-squares (handles underdetermined case)
K, _, _, _ = np.linalg.lstsq(C0, Ctau, rcond=None)  # (d, d)

# Eigendecomposition — eigenvalues may be complex; take real parts for sorting
eigenvalues, eigenvectors = np.linalg.eig(K)

# Sort descending by magnitude (slowest modes first)
order = np.argsort(np.abs(eigenvalues))[::-1]
eigenvalues  = eigenvalues[order]
eigenvectors = eigenvectors[:, order]

# Implied timescales in units of trajectory frames (skip stationary mode at index 0)
its = -lag / np.log(np.abs(eigenvalues[1:]))

results_dir = os.path.join(MD_PATH, f"transfer_operator_results_cg{CG}_steps{STEPS}_lag{lag}")
os.makedirs(results_dir, exist_ok=True)
with open(os.path.join(results_dir, "results.txt"), "w") as f:
    f.write(f'Transfer matrix K shape: {K.shape}\n')
    f.write(f'Top 10 eigenvalues (|λ|): {np.abs(eigenvalues[:10]).round(4)}\n')
    f.write(f'Top 9 implied timescales (frames): {its[:9].real.round(2)}\n')
    f.write(f'Eigenvectors shape: {eigenvectors.shape}\n')

np.save(os.path.join(results_dir, "transfer_matrix.npy"), K)
np.save(os.path.join(results_dir, "eigenvalues.npy"), eigenvalues)
np.save(os.path.join(results_dir, "eigenvectors.npy"), eigenvectors)