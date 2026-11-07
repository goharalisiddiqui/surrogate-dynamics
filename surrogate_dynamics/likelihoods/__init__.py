"""
Likelihood modules for probabilistic time-series forecasting in ``surrogate_dynamics``.

Defines the ``"likelihoods"`` plugin kind in :mod:`collective_encoder.common.registry`
with :class:`~darts.utils.likelihood_models.torch.TorchLikelihood` as the base class,
and registers both custom ``surrogate_dynamics`` likelihoods and standard Darts
likelihoods.
"""

from darts.utils.likelihood_models.torch import TorchLikelihood

from collective_encoder.common.registry import define_kind, lazy_exports, register

define_kind("likelihoods", TorchLikelihood)

_BUILTINS = {
    # Custom / modified surrogate_dynamics likelihoods
    "Gaussian": (
        "surrogate_dynamics.likelihoods.modified_gaussian",
        "ModifiedGaussianLikelihood",
    ),
    "GAUSSIAN": (
        "surrogate_dynamics.likelihoods.modified_gaussian",
        "ModifiedGaussianLikelihood",
    ),
    "ModifiedGaussian": (
        "surrogate_dynamics.likelihoods.modified_gaussian",
        "ModifiedGaussianLikelihood",
    ),
    "MODIFIED_GAUSSIAN": (
        "surrogate_dynamics.likelihoods.modified_gaussian",
        "ModifiedGaussianLikelihood",
    ),
    "ModifiedGaussianLikelihood": (
        "surrogate_dynamics.likelihoods.modified_gaussian",
        "ModifiedGaussianLikelihood",
    ),
    "Quantile": (
        "surrogate_dynamics.likelihoods.modified_quantile",
        "ModifiedQuantileRegression",
    ),
    "QUANTILE": (
        "surrogate_dynamics.likelihoods.modified_quantile",
        "ModifiedQuantileRegression",
    ),
    "ModifiedQuantile": (
        "surrogate_dynamics.likelihoods.modified_quantile",
        "ModifiedQuantileRegression",
    ),
    "MODIFIED_QUANTILE": (
        "surrogate_dynamics.likelihoods.modified_quantile",
        "ModifiedQuantileRegression",
    ),
    "ModifiedQuantileRegression": (
        "surrogate_dynamics.likelihoods.modified_quantile",
        "ModifiedQuantileRegression",
    ),
    # Standard Darts likelihoods
    "DARTS_GAUSSIAN": ("darts.utils.likelihood_models", "GaussianLikelihood"),
    "GaussianLikelihood": ("darts.utils.likelihood_models", "GaussianLikelihood"),
    "DARTS_QUANTILE": ("darts.utils.likelihood_models", "QuantileRegression"),
    "QuantileRegression": ("darts.utils.likelihood_models", "QuantileRegression"),
    "Laplace": ("darts.utils.likelihood_models", "LaplaceLikelihood"),
    "LAPLACE": ("darts.utils.likelihood_models", "LaplaceLikelihood"),
    "LaplaceLikelihood": ("darts.utils.likelihood_models", "LaplaceLikelihood"),
    "LogNormal": ("darts.utils.likelihood_models", "LogNormalLikelihood"),
    "LOGNORMAL": ("darts.utils.likelihood_models", "LogNormalLikelihood"),
    "LogNormalLikelihood": ("darts.utils.likelihood_models", "LogNormalLikelihood"),
    "Bernoulli": ("darts.utils.likelihood_models", "BernoulliLikelihood"),
    "BernoulliLikelihood": ("darts.utils.likelihood_models", "BernoulliLikelihood"),
    "Beta": ("darts.utils.likelihood_models", "BetaLikelihood"),
    "BetaLikelihood": ("darts.utils.likelihood_models", "BetaLikelihood"),
    "Cauchy": ("darts.utils.likelihood_models", "CauchyLikelihood"),
    "CauchyLikelihood": ("darts.utils.likelihood_models", "CauchyLikelihood"),
    "ContinuousBernoulli": (
        "darts.utils.likelihood_models",
        "ContinuousBernoulliLikelihood",
    ),
    "ContinuousBernoulliLikelihood": (
        "darts.utils.likelihood_models",
        "ContinuousBernoulliLikelihood",
    ),
    "Dirichlet": ("darts.utils.likelihood_models", "DirichletLikelihood"),
    "DirichletLikelihood": ("darts.utils.likelihood_models", "DirichletLikelihood"),
    "Exponential": ("darts.utils.likelihood_models", "ExponentialLikelihood"),
    "ExponentialLikelihood": (
        "darts.utils.likelihood_models",
        "ExponentialLikelihood",
    ),
    "Gamma": ("darts.utils.likelihood_models", "GammaLikelihood"),
    "GammaLikelihood": ("darts.utils.likelihood_models", "GammaLikelihood"),
    "Geometric": ("darts.utils.likelihood_models", "GeometricLikelihood"),
    "GeometricLikelihood": ("darts.utils.likelihood_models", "GeometricLikelihood"),
    "Gumbel": ("darts.utils.likelihood_models", "GumbelLikelihood"),
    "GumbelLikelihood": ("darts.utils.likelihood_models", "GumbelLikelihood"),
    "HalfNormal": ("darts.utils.likelihood_models", "HalfNormalLikelihood"),
    "HalfNormalLikelihood": ("darts.utils.likelihood_models", "HalfNormalLikelihood"),
    "NegativeBinomial": (
        "darts.utils.likelihood_models",
        "NegativeBinomialLikelihood",
    ),
    "NegativeBinomialLikelihood": (
        "darts.utils.likelihood_models",
        "NegativeBinomialLikelihood",
    ),
    "Poisson": ("darts.utils.likelihood_models", "PoissonLikelihood"),
    "PoissonLikelihood": ("darts.utils.likelihood_models", "PoissonLikelihood"),
    "Weibull": ("darts.utils.likelihood_models", "WeibullLikelihood"),
    "WeibullLikelihood": ("darts.utils.likelihood_models", "WeibullLikelihood"),
}

for _name, (_source, _attr) in _BUILTINS.items():
    register("likelihoods", _name, _source, _attr, overwrite=True)

__getattr__, __dir__ = lazy_exports(
    __name__, {t: f"{m}:{t}" for m, t in _BUILTINS.values()}
)

__all__ = list({t for m, t in _BUILTINS.values()})
