import warnings


def LikelihoodResolver(likelihood_name: str, likelihood_args: dict):
    """Resolve the likelihood class based on the given name and arguments.

    Args:
        likelihood_name (str): Name of the likelihood.
        likelihood_args (dict): Arguments for the likelihood.

    Returns:
        An instance of the specified likelihood class.
    """
    from darts.utils.likelihood_models \
        import LaplaceLikelihood, LogNormalLikelihood, QuantileRegression
        
    from likelihoods.modified_gaussian \
        import ModifiedGaussianLikelihood as GaussianLikelihood

    likelihoods = {
        "Gaussian": GaussianLikelihood,
        "Laplace": LaplaceLikelihood,
        "LogNormal": LogNormalLikelihood,
        "Quantile": QuantileRegression,
    }
    defaults = {
        "Gaussian": {"prior_mu": 0.0, "prior_sigma": 1.0, 
                     "prior_strength": 0.5, "beta_nll": 0.0},
        "Laplace": {"prior_mu": 0.0, "prior_b": 1.0, 
                    "prior_strength": 0.5},
        "LogNormal": {"prior_mu": 0.0, "prior_sigma": 1.0, 
                      "prior_strength": 0.5},
        "Quantile": {"quantiles": [0.1, 0.5, 0.9]},
    }

    if likelihood_name not in likelihoods:
        raise ValueError(f"Unknown likelihood: {likelihood_name}")

    args = defaults[likelihood_name]
    
    for a in likelihood_args:
        if a not in args:
            warnings.warn(f"Unknown argument '{a}' for likelihood '{likelihood_name}' ignored.")
            del likelihood_args[a]
    args.update(likelihood_args)
    
    return likelihoods[likelihood_name](**args)