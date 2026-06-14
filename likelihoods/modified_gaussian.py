from typing import Dict, Union

import numpy as np

import torch

from darts.utils.likelihood_models import GaussianLikelihood

from collective_encoder.common.config_check import validate_required_fields


def fixed_variance(mu, sigma, args) -> torch.Tensor:
    try:
        variance = float(args["variance"])    
    except KeyError:
        raise ValueError("Missing 'variance' key in noise_args for fixed_variance")
    sigma = torch.sqrt(variance)
    return torch.normal(mu, sigma)

def scaled_variance(mu, sigma, args) -> torch.Tensor:
    validate_required_fields(args, ["scale_factor"])
    scale_factor = float(args["scale_factor"])
    return torch.normal(mu, sigma * scale_factor)

def variance_by_dimension_mean(mu, sigma, args) -> torch.Tensor:
    validate_required_fields(args, ["mu_data_file"])
    mu_data_file = args["mu_data_file"]
    scale_factor = float(args.get("scale_factor", 1.0))

    try:
        mu_data = np.load(mu_data_file)
    except FileNotFoundError:
        raise ValueError(f"Mu data file not found at {mu_data_file}")

    if mu_data.shape[0] != mu.shape[-1]:
        raise ValueError(f"Mu data shape {mu_data.shape} does not match model output shape {mu.shape}")
    
    mu_data = mu_data * scale_factor
    mu_data = np.clip(mu_data, a_min=1e-6, a_max=None)  # Ensure mean is positive
    sig = torch.sqrt(torch.from_numpy(mu_data).to(mu.device))
    
    return torch.normal(mu, sig)

def variance_by_dimension_sample(mu, sigma, args) -> torch.Tensor:
    validate_required_fields(args, ["mu_data_file", "var_data_file"])
    mu_data_file = args["mu_data_file"]
    var_data_file = args["var_data_file"]
    scale_factor = float(args.get("scale_factor", 1.0))
    try:
        mu_data = np.load(mu_data_file)
        var_data = np.load(var_data_file)
    except FileNotFoundError as e:
        raise ValueError(f"Data file not found: {e.filename}")

    if var_data.shape != mu_data.shape:
        raise ValueError(f"Variance data shape {var_data.shape} does not match model output shape {mu.shape}")
    
    if mu_data.shape[0] != mu.shape[-1]:
        raise ValueError(f"Mu data shape {mu_data.shape} does not match model output shape {mu.shape}")

    var = np.random.normal(mu_data, var_data) * scale_factor
    var = np.clip(var, a_min=1e-6, a_max=None)  # Ensure variance is positive
    sig = torch.sqrt(torch.from_numpy(var).to(mu.device))

    return torch.normal(mu, sig)

_REGISTRY = {
    "fixed_variance": fixed_variance,
    "scaled_variance": scaled_variance,
    "variance_by_dimension_mean": variance_by_dimension_mean,
    "variance_by_dimension_sample": variance_by_dimension_sample,
}

class ModifiedGaussianLikelihood(GaussianLikelihood):
    def sample(self, 
               model_output: torch.Tensor, 
               noise_type: Union[str, None] = None,
               noise_args: Union[Dict[str, Union[str, float, int]], None] = None,
    ) -> torch.Tensor:
        mu, sigma = self._params_from_output(model_output)
        if noise_type == None:
            return torch.normal(mu, sigma)
        return _REGISTRY[noise_type](mu, sigma, noise_args)
            

