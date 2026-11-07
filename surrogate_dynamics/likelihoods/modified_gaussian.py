from typing import Any, Dict, Optional, Union

import numpy as np
import torch
from darts.utils.likelihood_models import GaussianLikelihood

from collective_encoder.common.config_check import validate_required_fields


def fixed_variance(
    mu: torch.Tensor, sigma: torch.Tensor, args: Optional[Dict[str, Any]]
) -> torch.Tensor:
    """
    Sample from a Gaussian distribution with a constant user-specified variance.

    Parameters
    ----------
    mu : torch.Tensor
        Predicted mean tensor.
    sigma : torch.Tensor
        Predicted standard deviation tensor (unused; replaced by fixed variance).
    args : dict
        Noise configuration dictionary containing ``"variance"``.

    Returns
    -------
    torch.Tensor
        Sampled tensor of the same shape, device, and dtype as ``mu``.

    Raises
    ------
    ValueError
        If ``"variance"`` is missing from ``args``.
    """
    if args is None or "variance" not in args:
        raise ValueError("Missing 'variance' key in noise_args for fixed_variance")
    variance = float(args["variance"])
    std = torch.as_tensor(variance, device=mu.device, dtype=mu.dtype).sqrt()
    return torch.normal(mu, std.expand_as(mu))


def scaled_variance(
    mu: torch.Tensor, sigma: torch.Tensor, args: Optional[Dict[str, Any]]
) -> torch.Tensor:
    """
    Sample from a Gaussian distribution with predicted standard deviation scaled
    by ``scale_factor``.

    Parameters
    ----------
    mu : torch.Tensor
        Predicted mean tensor.
    sigma : torch.Tensor
        Predicted standard deviation tensor.
    args : dict
        Noise configuration dictionary containing ``"scale_factor"``.

    Returns
    -------
    torch.Tensor
        Sampled tensor of the same shape, device, and dtype as ``mu``.
    """
    validate_required_fields(args or {}, ["scale_factor"])
    scale_factor = float(args["scale_factor"])
    return torch.normal(mu, sigma * scale_factor)


def variance_by_dimension_mean(
    mu: torch.Tensor, sigma: torch.Tensor, args: Optional[Dict[str, Any]]
) -> torch.Tensor:
    """
    Sample from a Gaussian distribution using per-dimension mean variances loaded
    from a ``.npy`` file.

    Parameters
    ----------
    mu : torch.Tensor
        Predicted mean tensor.
    sigma : torch.Tensor
        Predicted standard deviation tensor.
    args : dict
        Noise configuration dictionary containing ``"mu_data_file"`` and optional
        ``"scale_factor"``.

    Returns
    -------
    torch.Tensor
        Sampled tensor of the same shape, device, and dtype as ``mu``.

    Raises
    ------
    ValueError
        If the file is missing or its shape does not match ``mu.shape[-1]``.
    """
    validate_required_fields(args or {}, ["mu_data_file"])
    mu_data_file = args["mu_data_file"]
    scale_factor = float(args.get("scale_factor", 1.0))

    try:
        mu_data = np.load(mu_data_file)
    except FileNotFoundError:
        raise ValueError(f"Mu data file not found at {mu_data_file}")

    if mu_data.shape[0] != mu.shape[-1]:
        raise ValueError(
            f"Mu data shape {mu_data.shape} does not match model output shape {mu.shape}"
        )

    mu_data = mu_data * scale_factor
    mu_data = np.clip(mu_data, a_min=1e-6, a_max=None)  # Ensure mean is positive
    sig = torch.sqrt(
        torch.from_numpy(mu_data).to(device=mu.device, dtype=mu.dtype)
    )

    return torch.normal(mu, sig)


def variance_by_dimension_sample(
    mu: torch.Tensor, sigma: torch.Tensor, args: Optional[Dict[str, Any]]
) -> torch.Tensor:
    """
    Sample from a Gaussian distribution using per-dimension variances drawn from a
    normal distribution parameterized by ``mu_data_file`` and ``var_data_file``.

    Parameters
    ----------
    mu : torch.Tensor
        Predicted mean tensor.
    sigma : torch.Tensor
        Predicted standard deviation tensor.
    args : dict
        Noise configuration dictionary containing ``"mu_data_file"``,
        ``"var_data_file"``, and optional ``"scale_factor"``.

    Returns
    -------
    torch.Tensor
        Sampled tensor of the same shape, device, and dtype as ``mu``.

    Raises
    ------
    ValueError
        If either file is missing or its shape does not match ``mu.shape[-1]``.
    """
    validate_required_fields(args or {}, ["mu_data_file", "var_data_file"])
    mu_data_file = args["mu_data_file"]
    var_data_file = args["var_data_file"]
    scale_factor = float(args.get("scale_factor", 1.0))
    try:
        mu_data = np.load(mu_data_file)
        var_data = np.load(var_data_file)
    except FileNotFoundError as e:
        raise ValueError(f"Data file not found: {e.filename}")

    if var_data.shape != mu_data.shape:
        raise ValueError(
            f"Variance data shape {var_data.shape} does not match model output shape {mu.shape}"
        )

    if mu_data.shape[0] != mu.shape[-1]:
        raise ValueError(
            f"Mu data shape {mu_data.shape} does not match model output shape {mu.shape}"
        )

    var = np.random.normal(mu_data, var_data) * scale_factor
    var = np.clip(var, a_min=1e-6, a_max=None)  # Ensure variance is positive
    sig = torch.sqrt(
        torch.from_numpy(var).to(device=mu.device, dtype=mu.dtype)
    )

    return torch.normal(mu, sig)


_REGISTRY = {
    "fixed_variance": fixed_variance,
    "scaled_variance": scaled_variance,
    "variance_by_dimension_mean": variance_by_dimension_mean,
    "variance_by_dimension_sample": variance_by_dimension_sample,
}


class ModifiedGaussianLikelihood(GaussianLikelihood):
    """
    Gaussian likelihood extending :class:`darts.utils.likelihood_models.GaussianLikelihood`
    with configurable inference-time noise sampling strategies (``fixed_variance``,
    ``scaled_variance``, ``variance_by_dimension_mean``, ``variance_by_dimension_sample``).
    """

    _IDENTIFIER = "ModifiedGaussian"

    def sample(
        self,
        model_output: torch.Tensor,
        noise_type: Union[str, None] = None,
        noise_args: Union[Dict[str, Union[str, float, int]], None] = None,
    ) -> torch.Tensor:
        """
        Sample a prediction from the predicted Gaussian distribution parameters.

        Parameters
        ----------
        model_output : torch.Tensor
            Raw model output tensor of shape ``(batch_size, n_timesteps, n_components, 2)``.
        noise_type : str, optional
            Optional inference noise strategy name in ``_REGISTRY``. If ``None``,
            samples from ``Normal(mu, sigma)``.
        noise_args : dict, optional
            Arguments forwarded to the selected noise sampling function.

        Returns
        -------
        torch.Tensor
            Sampled prediction tensor of shape ``(batch_size, n_timesteps, n_components)``.
        """
        mu, sigma = self._params_from_output(model_output)
        if noise_type is None:
            return torch.normal(mu, sigma)
        if noise_type not in _REGISTRY:
            raise ValueError(
                f"Unknown noise_type '{noise_type}'. Available: {list(_REGISTRY.keys())}"
            )
        return _REGISTRY[noise_type](mu, sigma, noise_args)
