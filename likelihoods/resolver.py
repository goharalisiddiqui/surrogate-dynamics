import warnings
import importlib

_REGISTRY = {
        "Gaussian": ('likelihoods.modified_gaussian', 'ModifiedGaussianLikelihood'),
        "Laplace": ('darts.utils.likelihood_models', 'LaplaceLikelihood'),
        "LogNormal": ('darts.utils.likelihood_models', 'LogNormalLikelihood'),
        "Quantile": ('.modified_quantile', 'QuantileRegression'),
}

_DEFAULT_ARGS = {
    "Gaussian": {"prior_mu": 0.0, "prior_sigma": 1.0, 
                 "prior_strength": 0.5, "beta_nll": 0.0},
    "Laplace": {"prior_mu": 0.0, "prior_b": 1.0, "prior_strength": 0.5},
    "LogNormal": {"prior_mu": 0.0, "prior_sigma": 1.0, "prior_strength": 0.5},
    "Quantile": {"quantiles": [0.1, 0.5, 0.9]},
}


def likelihood_resolver(likelihood_name: str, likelihood_args: dict):
    """Resolve the likelihood class based on the given name and arguments.

    Args:
        likelihood_name (str): Name of the likelihood.
        likelihood_args (dict): Arguments for the likelihood.

    Returns:
        An instance of the specified likelihood class.
    """
    if likelihood_name not in _REGISTRY:
        raise ValueError(f"Unknown likelihood: {likelihood_name}")
    
    module_name, class_name = _REGISTRY[likelihood_name]
    module = importlib.import_module(module_name)
    likelihood_class = getattr(module, class_name)

    default_args = _DEFAULT_ARGS.get(likelihood_name, {})
    default_args.update(likelihood_args)

    return likelihood_class(**default_args)