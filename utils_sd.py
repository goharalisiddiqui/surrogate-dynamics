import numpy as np

def compute_mfpt_matrix(vals : np.ndarray, minima : np.ndarray, lag: int = 1):
    '''
    Compute and save Mean First Passage Time (MFPT) matrix between the given minima.
    Step 1: Construct an MSM from the trajectory data (phi, psi) on state values defined by the given minima using PyEMMA.
    Step 2: Compute the MFPT matrix between the states defined by the minima.

    Parameters:
    vals (np.ndarray): Trajectory data of shape (n_samples, n_features).
    minima (np.ndarray): Array of minima of shape (n_states, n_features).
    lag (int): Lag time for the MSM.
    '''
    assert vals.shape[1] == minima.shape[1], "Dimensionality of vals and minima must match."
    assert minima.shape[0] >= 2, "At least two minima are required to compute MFPT."
    # Step 1: Assign trajectory points to states based on nearest minima
    traj_data = vals

    # Assign each point to the nearest minimum (state assignment)
    distances = np.linalg.norm(traj_data[:, None, :] - minima[None, :, :], axis=-1)
    state_assignments = np.argmin(distances, axis=-1)
    
    n_states = len(minima)
    lag_time = lag

    # Build transition count matrix
    transition_counts = np.zeros((n_states, n_states))
    
    for t in range(len(state_assignments) - lag_time):
        i = state_assignments[t]
        j = state_assignments[t + lag_time]
        transition_counts[i, j] += 1
    
    # Convert to transition probability matrix
    row_sums = transition_counts.sum(axis=1)
    # Avoid division by zero
    row_sums[row_sums == 0] = 1
    transition_matrix = transition_counts / row_sums[:, np.newaxis]
    
    # Step 2: Compute MFPT matrix
    # MFPT from state i to state j is computed by solving linear system
    mfpt_matrix = np.zeros((n_states, n_states))
    
    for j in range(n_states):
        # For each target state j, solve for mean hitting times
        # Set up system: (I - P + e_j e_j^T) * tau = 1
        # where e_j is unit vector for state j
        
        A = np.eye(n_states) - transition_matrix
        A[j, :] = 0  # Replace j-th row
        A[j, j] = 1  # Make it absorbing
        
        b = np.ones(n_states)
        b[j] = 0  # No time needed to reach j from j
        
        try:
            tau = np.linalg.solve(A, b)
            mfpt_matrix[:, j] = tau
        except np.linalg.LinAlgError:
            # If matrix is singular, use pseudoinverse
            tau = np.linalg.pinv(A) @ b
            mfpt_matrix[:, j] = tau
    
    # Set diagonal to 0 (no time to reach same state)
    np.fill_diagonal(mfpt_matrix, 0)
    
    # Handle infinite/very large values
    mfpt_matrix = np.where(np.isfinite(mfpt_matrix), mfpt_matrix, np.inf)
    
    return transition_counts, transition_matrix, mfpt_matrix

import pandas as pd
import numpy as np
from itertools import chain
import torch
from pytorch_lightning.callbacks import Callback
  # Added missing import statement

def normalization(a):
    return (a - a.min())/(a.max()- a.min()), a.max(), a.min()

def normalization_with_inputs(a, amax, amin):
    return (a - amin)/(amax - amin)

def reverse_normalization(a,amax,amin):
    return a*(amax-amin) + amin

def data_processing(df, name):
    temp = df[name]
    temp = torch.tensor(temp)
    temp = torch.reshape(temp, (temp.shape[0],1))
    return temp

def flat(list_2D):
    flatten_list = list(chain.from_iterable(list_2D))
    flatten_list = torch.tensor(flatten_list)
    return flatten_list


def reshape_vae(x):
    if len(x.shape) == 5:
        x = torch.reshape(x, (x.shape[0]* x.shape[1],  1, 50, 100))
    elif len(x.shape) == 6:
        x = torch.reshape(x, (x.shape[0]*x.shape[1]*x.shape[2], 1, 50, 100))

    elif len(x.shape) == 4:
        x = torch.reshape(x, (x.shape[0]*x.shape[1], 1, 50,100))

    elif len(x.shape) == 3:
        x = torch.reshape(x, (x.shape[0]*x.shape[1], -1))
    else:
        print("Invalid shape:", x.shape)
        return None
    return x

def unshape_vae(x, n_configs, n_time, lat):
    if lat:
        x = torch.reshape(x, (n_configs, n_time, -1))
    else:
        x = torch.reshape(x, (n_configs, n_time, 50,100))
    return x

def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def s_mape(A, F):

    A = np.array(A)
    F = np.array(F)
    
    # Avoid division by zero
    denominator = np.abs(A) + np.abs(F)
    denominator[denominator == 0] = 1e-10  
    return 100*np.mean((2 * np.abs(F - A) / denominator))

class LossLogger(Callback):
    def __init__(self):
        self.train_loss = []
        self.val_loss = []

    # will automatically be called at the end of each epoch
    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self.train_loss.append(float(trainer.callback_metrics["train_loss"]))

    def on_validation_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self.val_loss.append(float(trainer.callback_metrics["val_loss"]))

def get_encoded_decoded(vae, data_train, data_test):
    with torch.no_grad():
        data_train_vae = reshape_vae(data_train)
        data_test_vae = reshape_vae(data_test)
        data_train_encoded = vae(data_train_vae)[2]
        data_train_encoded_std = vae(data_train_vae)[3]
        data_test_encoded = vae(data_test_vae)[2]
        data_test_encoded_std = vae(data_test_vae)[3]
        data_train_decoded = vae.decoder(data_train_encoded)
        data_test_decoded = vae.decoder(data_test_encoded)
        data_train_decoded_std = vae.decoder(data_train_encoded_std)
        data_test_decoded_std = vae.decoder(data_test_encoded_std)
        data_train_decoded = unshape_vae(data_train_decoded, data_train.shape[0], data_train.shape[1], False)
        data_test_decoded = unshape_vae(data_test_decoded, data_test.shape[0], data_test.shape[1], False)
        data_train_decoded_std = unshape_vae(data_train_decoded_std, data_train.shape[0], data_train.shape[1], False)
        data_test_decoded_std = unshape_vae(data_test_decoded_std, data_test.shape[0], data_test.shape[1], False)

    return data_train_decoded, data_test_decoded , data_train_encoded, data_test_encoded#, data_train_decoded_std, data_test_decoded_std