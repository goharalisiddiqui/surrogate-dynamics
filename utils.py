import numpy as np

def periodic_distance(x1, x2, period=2*np.pi):
    """ Compute the periodic distance between two points x1 and x2 """
    # Minimum image convention
    d = (x1 - x2 + period/2) % period - period/2
    
    return d

def periodic_norm_distance(x1, x2, period=2*np.pi):
    """ Compute the periodic norm distance between two points x1 and x2 """
    
    return float(np.linalg.norm(periodic_distance(x1, x2, period)))

def periodic_average(x, period=2*np.pi):
    """ Compute the periodic average of a list of 2D points x """
    # Minimum image convention
    x = np.array(x)
    x1 = np.arctan2(np.mean(np.sin(x[:,0])), np.mean(np.cos(x[:,0])))
    x2 = np.arctan2(np.mean(np.sin(x[:,1])), np.mean(np.cos(x[:,1])))
    return np.array([x1, x2])












































































# def normalization(a):
#     """
#     Normalize array to [0, 1] range using min-max normalization.
    
#     Parameters
#     ----------
#     a : np.ndarray or array-like
#         Input array to normalize.
    
#     Returns
#     -------
#     tuple
#         - normalized_array : np.ndarray
#             Normalized array in range [0, 1].
#         - max_val : float
#             Maximum value of input array (used for denormalization).
#         - min_val : float
#             Minimum value of input array (used for denormalization).
    
#     Examples
#     --------
#     >>> a = np.array([1, 2, 3, 4, 5])
#     >>> norm_a, amax, amin = normalization(a)
#     >>> norm_a
#     array([0.  , 0.25, 0.5 , 0.75, 1.  ])
#     """
#     return (a - a.min())/(a.max()- a.min()), a.max(), a.min()

# def normalization_with_inputs(a, amax, amin):
#     """
#     Normalize array using provided min/max values (min-max normalization).
    
#     Used to apply the same normalization transformation (from training data)
#     to new data using pre-computed statistics.
    
#     Parameters
#     ----------
#     a : np.ndarray or array-like
#         Input array to normalize.
#     amax : float
#         Maximum value (typically from training data).
#     amin : float
#         Minimum value (typically from training data).
    
#     Returns
#     -------
#     np.ndarray
#         Normalized array in range [0, 1] using the provided bounds.
    
#     Examples
#     --------
#     >>> a_train = np.array([1, 2, 3, 4, 5])
#     >>> _, amax, amin = normalization(a_train)
#     >>> a_test = np.array([2, 3, 4])
#     >>> norm_a_test = normalization_with_inputs(a_test, amax, amin)
#     >>> norm_a_test
#     array([0.25, 0.5 , 0.75])
#     """
#     return (a - amin)/(amax - amin)

# def reverse_normalization(a, amax, amin):
#     """
#     Denormalize array from [0, 1] range back to original scale.
    
#     Inverse operation of min-max normalization.
    
#     Parameters
#     ----------
#     a : np.ndarray or array-like
#         Normalized array in range [0, 1].
#     amax : float
#         Maximum value of original data.
#     amin : float
#         Minimum value of original data.
    
#     Returns
#     -------
#     np.ndarray
#         Denormalized array in original scale.
    
#     Examples
#     --------
#     >>> a_denorm = reverse_normalization(np.array([0.5]), amax=5, amin=1)
#     >>> a_denorm
#     array([3.])
#     """
#     return a*(amax-amin) + amin

# def data_processing(df, name):
#     """
#     Extract a column from DataFrame and convert to 2D PyTorch tensor.
    
#     Parameters
#     ----------
#     df : pd.DataFrame
#         Input DataFrame.
#     name : str
#         Column name to extract.
    
#     Returns
#     -------
#     torch.Tensor
#         2D tensor of shape (n_samples, 1) containing the column data.
    
#     Examples
#     --------
#     >>> import pandas as pd
#     >>> df = pd.DataFrame({'col': [1, 2, 3]})
#     >>> tensor = data_processing(df, 'col')
#     >>> tensor.shape
#     torch.Size([3, 1])
#     """
#     temp = df[name]
#     temp = torch.tensor(temp)
#     temp = torch.reshape(temp, (temp.shape[0],1))
#     return temp

# def flat(list_2D):
#     """
#     Flatten a 2D list and convert to PyTorch tensor.
    
#     Parameters
#     ----------
#     list_2D : list of lists
#         2D nested list structure.
    
#     Returns
#     -------
#     torch.Tensor
#         1D tensor containing all flattened elements.
    
#     Examples
#     --------
#     >>> list_2d = [[1, 2], [3, 4], [5, 6]]
#     >>> tensor = flat(list_2d)
#     >>> tensor
#     tensor([1, 2, 3, 4, 5, 6])
#     """
#     flatten_list = list(chain.from_iterable(list_2D))
#     flatten_list = torch.tensor(flatten_list)
#     return flatten_list

# def reshape_vae(x):
#     """
#     Reshape tensor for VAE input based on input dimensionality.
    
#     Flattens batch and time dimensions into a single batch dimension,
#     while preserving spatial/feature dimensions.
    
#     Parameters
#     ----------
#     x : torch.Tensor
#         Input tensor of shape:
#         - (batch, time, 1, 50, 100) → (batch*time, 1, 50, 100) [5D]
#         - (batch, time, config, 1, 50, 100) → (batch*time*config, 1, 50, 100) [6D]
#         - (batch, time, 1, 50, 100) → (batch*time, 1, 50, 100) [4D]
#         - (batch, time, features) → (batch*time, features) [3D]
    
#     Returns
#     -------
#     torch.Tensor or None
#         Reshaped tensor, or None if shape is invalid.
#     """
#     if len(x.shape) == 5:
#         x = torch.reshape(x, (x.shape[0]* x.shape[1],  1, 50, 100))
#     elif len(x.shape) == 6:
#         x = torch.reshape(x, (x.shape[0]*x.shape[1]*x.shape[2], 1, 50, 100))
#     elif len(x.shape) == 4:
#         x = torch.reshape(x, (x.shape[0]*x.shape[1], 1, 50,100))
#     elif len(x.shape) == 3:
#         x = torch.reshape(x, (x.shape[0]*x.shape[1], -1))
#     else:
#         print("Invalid shape:", x.shape)
#         return None
#     return x

# def unshape_vae(x, n_configs, n_time, lat):
#     """
#     Reshape flattened VAE output back to original batch/time structure.
    
#     Inverse operation of reshape_vae.
    
#     Parameters
#     ----------
#     x : torch.Tensor
#         Flattened tensor from VAE.
#     n_configs : int
#         Number of configurations (batch size).
#     n_time : int
#         Number of time steps.
#     lat : bool
#         If True, reshape to (n_configs, n_time, latent_dim).
#         If False, reshape to (n_configs, n_time, 50, 100).
    
#     Returns
#     -------
#     torch.Tensor
#         Reshaped tensor with original dimensions.
#     """
#     if lat:
#         x = torch.reshape(x, (n_configs, n_time, -1))
#     else:
#         x = torch.reshape(x, (n_configs, n_time, 50,100))
#     return x

# def mean_absolute_percentage_error(y_true, y_pred): 
#     """
#     Calculate Mean Absolute Percentage Error (MAPE).
    
#     Metric: 100 * mean(|y_true - y_pred| / y_true)
    
#     Warning: Undefined for zero values in y_true.
    
#     Parameters
#     ----------
#     y_true : np.ndarray or array-like
#         True values.
#     y_pred : np.ndarray or array-like
#         Predicted values.
    
#     Returns
#     -------
#     float
#         MAPE as percentage (0-100+).
    
#     Examples
#     --------
#     >>> mape = mean_absolute_percentage_error([100, 200], [110, 190])
#     >>> mape
#     10.0
#     """
#     y_true, y_pred = np.array(y_true), np.array(y_pred)
#     return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

# def s_mape(A, F):
#     """
#     Calculate Symmetric Mean Absolute Percentage Error (sMAPE).
    
#     More symmetric variant of MAPE: 100 * mean(2 * |F - A| / (|A| + |F|))
    
#     Handles division by zero and is symmetric in A and F.
    
#     Parameters
#     ----------
#     A : np.ndarray or array-like
#         Actual/true values.
#     F : np.ndarray or array-like
#         Predicted/forecast values.
    
#     Returns
#     -------
#     float
#         sMAPE as percentage (0-200).
    
#     Examples
#     --------
#     >>> smape = s_mape([100, 200], [110, 190])
#     >>> smape
#     9.523809523809524
#     """
#     A = np.array(A)
#     F = np.array(F)
    
#     # Avoid division by zero
#     denominator = np.abs(A) + np.abs(F)
#     denominator[denominator == 0] = 1e-10  
#     return 100*np.mean((2 * np.abs(F - A) / denominator))

# class LossLogger(Callback):
#     """
#     PyTorch Lightning callback to track training and validation loss per epoch.
    
#     Captures train_loss and val_loss metrics from trainer.callback_metrics at the end 
#     of each training and validation epoch, storing them in lists for later analysis.
    
#     Attributes
#     ----------
#     train_loss : list
#         List of training loss values collected at each training epoch end.
#     val_loss : list
#         List of validation loss values collected at each validation epoch end.
    
#     Methods
#     -------
#     on_train_epoch_end(trainer, pl_module)
#         Called at the end of each training epoch to log training loss.
#     on_validation_epoch_end(trainer, pl_module)
#         Called at the end of each validation epoch to log validation loss.
    
#     Examples
#     --------
#     >>> callback = LossLogger()
#     >>> trainer = pl.Trainer(callbacks=[callback])
#     >>> trainer.fit(model, train_loader, val_loader)
#     >>> print(callback.train_loss)  # [loss_epoch1, loss_epoch2, ...]
#     """
#     def __init__(self):
#         self.train_loss = []
#         self.val_loss = []

#     # will automatically be called at the end of each epoch
#     def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
#         self.train_loss.append(float(trainer.callback_metrics["train_loss"]))

#     def on_validation_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
#         self.val_loss.append(float(trainer.callback_metrics["val_loss"]))

# def get_encoded_decoded(vae, data_train, data_test):
#     """
#     Extract encoded and decoded representations from a VAE model.
    
#     Processes training and test data through a VAE: reshapes input, encodes via encoder,
#     decodes via decoder, and unshapes output back to original dimensions.
    
#     Parameters
#     ----------
#     vae : torch.nn.Module
#         Variational Autoencoder with .encoder() and .decoder() methods.
#         Encoder returns (mean, logvar, encoded, std) where encoded is the latent code.
#     data_train : torch.Tensor
#         Training data with shape (batch_size, seq_len, features) or flexible shape.
#     data_test : torch.Tensor
#         Test data with shape (batch_size, seq_len, features) or flexible shape.
    
#     Returns
#     -------
#     tuple of (torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor)
#         - data_train_decoded: Reconstructed training data, same shape as data_train
#         - data_test_decoded: Reconstructed test data, same shape as data_test
#         - data_train_encoded: Latent encodings of training data, shape (batch_size, latent_dim)
#         - data_test_encoded: Latent encodings of test data, shape (batch_size, latent_dim)
    
#     Notes
#     -----
#     Uses torch.no_grad() context to avoid computing gradients. Internally calls
#     reshape_vae() before VAE forward pass and unshape_vae() to restore original shapes.
    
#     Examples
#     --------
#     >>> train_decoded, test_decoded, train_encoded, test_encoded = get_encoded_decoded(
#     ...     vae, data_train, data_test)
#     >>> print(train_decoded.shape, train_encoded.shape)
#     """
#     with torch.no_grad():
#         data_train_vae = reshape_vae(data_train)
#         data_test_vae = reshape_vae(data_test)
#         data_train_encoded = vae(data_train_vae)[2]
#         data_train_encoded_std = vae(data_train_vae)[3]
#         data_test_encoded = vae(data_test_vae)[2]
#         data_test_encoded_std = vae(data_test_vae)[3]
#         data_train_decoded = vae.decoder(data_train_encoded)
#         data_test_decoded = vae.decoder(data_test_encoded)
#         data_train_decoded_std = vae.decoder(data_train_encoded_std)
#         data_test_decoded_std = vae.decoder(data_test_encoded_std)
#         data_train_decoded = unshape_vae(data_train_decoded, data_train.shape[0], data_train.shape[1], False)
#         data_test_decoded = unshape_vae(data_test_decoded, data_test.shape[0], data_test.shape[1], False)
#         data_train_decoded_std = unshape_vae(data_train_decoded_std, data_train.shape[0], data_train.shape[1], False)
#         data_test_decoded_std = unshape_vae(data_test_decoded_std, data_test.shape[0], data_test.shape[1], False)

#     return data_train_decoded, data_test_decoded , data_train_encoded, data_test_encoded#, data_train_decoded_std, data_test_decoded_std