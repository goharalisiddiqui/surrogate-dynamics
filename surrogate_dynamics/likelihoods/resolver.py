import importlib
import pkgutil
from typing import Any, Dict, Optional, Union

from darts.utils.likelihood_models.torch import TorchLikelihood

import surrogate_dynamics.likelihoods  # Ensure "likelihoods" plugin kind is registered
from collective_encoder.common.registry import available, get_module

_REGISTRY = {
    "Gaussian": (
        "surrogate_dynamics.likelihoods.modified_gaussian",
        "ModifiedGaussianLikelihood",
    ),
    "ModifiedGaussian": (
        "surrogate_dynamics.likelihoods.modified_gaussian",
        "ModifiedGaussianLikelihood",
    ),
    "Laplace": ("darts.utils.likelihood_models", "LaplaceLikelihood"),
    "LogNormal": ("darts.utils.likelihood_models", "LogNormalLikelihood"),
    "Quantile": (
        "surrogate_dynamics.likelihoods.modified_quantile",
        "ModifiedQuantileRegression",
    ),
    "ModifiedQuantile": (
        "surrogate_dynamics.likelihoods.modified_quantile",
        "ModifiedQuantileRegression",
    ),
}

_DEFAULT_ARGS: Dict[str, Dict[str, Any]] = {
    "Gaussian": {
        "prior_mu": 0.0,
        "prior_sigma": 1.0,
        "prior_strength": 0.5,
        "beta_nll": 0.0,
    },
    "GAUSSIAN": {
        "prior_mu": 0.0,
        "prior_sigma": 1.0,
        "prior_strength": 0.5,
        "beta_nll": 0.0,
    },
    "ModifiedGaussian": {
        "prior_mu": 0.0,
        "prior_sigma": 1.0,
        "prior_strength": 0.5,
        "beta_nll": 0.0,
    },
    "MODIFIED_GAUSSIAN": {
        "prior_mu": 0.0,
        "prior_sigma": 1.0,
        "prior_strength": 0.5,
        "beta_nll": 0.0,
    },
    "ModifiedGaussianLikelihood": {
        "prior_mu": 0.0,
        "prior_sigma": 1.0,
        "prior_strength": 0.5,
        "beta_nll": 0.0,
    },
    "Laplace": {"prior_mu": 0.0, "prior_b": 1.0, "prior_strength": 0.5},
    "LAPLACE": {"prior_mu": 0.0, "prior_b": 1.0, "prior_strength": 0.5},
    "LogNormal": {"prior_mu": 0.0, "prior_sigma": 1.0, "prior_strength": 0.5},
    "LOGNORMAL": {"prior_mu": 0.0, "prior_sigma": 1.0, "prior_strength": 0.5},
    "Quantile": {"quantiles": [0.1, 0.5, 0.9]},
    "QUANTILE": {"quantiles": [0.1, 0.5, 0.9]},
    "ModifiedQuantile": {"quantiles": [0.1, 0.5, 0.9]},
    "MODIFIED_QUANTILE": {"quantiles": [0.1, 0.5, 0.9]},
    "ModifiedQuantileRegression": {"quantiles": [0.1, 0.5, 0.9]},
}


def _find_likelihood_class(likelihood_name: str) -> type:
    """
    Locate a likelihood class by identifier in the ``"likelihoods"`` plugin registry,
    ``surrogate_dynamics.likelihoods``, or ``darts.utils.likelihood_models``.

    Parameters
    ----------
    likelihood_name : str
        Identifier or class name of the likelihood model.

    Returns
    -------
    type
        Resolved :class:`~darts.utils.likelihood_models.torch.TorchLikelihood` subclass.

    Raises
    ------
    ValueError
        If ``likelihood_name`` cannot be resolved.
    """
    reg_names = available("likelihoods")
    if likelihood_name in reg_names:
        return get_module("likelihoods", likelihood_name)

    upper_map = {k.upper(): k for k in reg_names}
    if likelihood_name.upper() in upper_map:
        return get_module("likelihoods", upper_map[likelihood_name.upper()])

    if likelihood_name in _REGISTRY:
        module_name, class_name = _REGISTRY[likelihood_name]
        module = importlib.import_module(module_name)
        return getattr(module, class_name)

    # Dynamic search across surrogate_dynamics.likelihoods submodules
    sd_pkg = importlib.import_module("surrogate_dynamics.likelihoods")
    for _, modname, _ in pkgutil.iter_modules(sd_pkg.__path__):
        try:
            mod = importlib.import_module(f"surrogate_dynamics.likelihoods.{modname}")
            for attr_name in dir(mod):
                attr = getattr(mod, attr_name)
                if (
                    isinstance(attr, type)
                    and issubclass(attr, TorchLikelihood)
                    and (
                        attr_name == likelihood_name
                        or getattr(attr, "_IDENTIFIER", None) == likelihood_name
                    )
                ):
                    return attr
        except Exception:
            continue

    # Dynamic search in darts.utils.likelihood_models
    darts_lm = importlib.import_module("darts.utils.likelihood_models")
    if hasattr(darts_lm, likelihood_name):
        cls_obj = getattr(darts_lm, likelihood_name)
        if isinstance(cls_obj, type) and issubclass(cls_obj, TorchLikelihood):
            return cls_obj

    raise ValueError(
        f"Unknown likelihood: '{likelihood_name}'. "
        f"Available registered likelihoods: {reg_names}"
    )


def likelihood_resolver(
    likelihood_name: Union[str, Dict[str, Any], TorchLikelihood],
    likelihood_args: Optional[Dict[str, Any]] = None,
) -> TorchLikelihood:
    """
    Resolve and instantiate a likelihood model from a name, dictionary, or instance.

    Parameters
    ----------
    likelihood_name : str or dict or TorchLikelihood
        Name of the likelihood (e.g., ``"Gaussian"``, ``"Quantile"``,
        ``"ModifiedGaussian"``, ``"ModifiedQuantile"``, ``"Laplace"``,
        ``"QuantileRegression"``), a dictionary containing ``"type"`` / ``"name"``
        and optional ``"args"``, or an already-instantiated :class:`TorchLikelihood`.
    likelihood_args : dict, optional
        Keyword arguments forwarded to the likelihood constructor (overrides any
        defaults in ``_DEFAULT_ARGS``).

    Returns
    -------
    TorchLikelihood
        An instance of the specified likelihood class.

    Raises
    ------
    ValueError
        If ``likelihood_name`` cannot be resolved to a valid likelihood class.
    """
    if isinstance(likelihood_name, TorchLikelihood):
        return likelihood_name

    merged_args: Dict[str, Any] = dict(likelihood_args or {})
    if isinstance(likelihood_name, dict):
        spec = dict(likelihood_name)
        name = spec.pop("type", None) or spec.pop("name", None) or spec.pop("likelihood", None)
        if not name:
            raise ValueError(
                f"Dictionary likelihood specification must contain 'type' or 'name', got: {likelihood_name}"
            )
        nested_args = spec.pop("args", None) or spec.pop("likelihood_args", None) or spec
        merged_args = {**dict(nested_args), **merged_args}
        likelihood_name = str(name)

    likelihood_class = _find_likelihood_class(str(likelihood_name))

    init_kwargs = _DEFAULT_ARGS.get(str(likelihood_name), {}).copy()
    for k, v in merged_args.items():
        if v is not None or k not in init_kwargs:
            init_kwargs[k] = v

    return likelihood_class(**init_kwargs)


LikelihoodResolver = likelihood_resolver
