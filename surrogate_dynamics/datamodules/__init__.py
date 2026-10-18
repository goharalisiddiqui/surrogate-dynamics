

import collective_encoder
from collective_encoder.common.registry import register, lazy_exports
from .timeseries.base import TimeseriesDataModule

_BUILTINS = {
    "TIMESERIES":              ("surrogate_dynamics.datamodules.timeseries.base",          "TimeseriesDataModule"),
    "UNIVARIATE_TIMESERIES":   ("surrogate_dynamics.datamodules.timeseries.univariate",    "UnivariateTimeseriesDataModule"),
    "MULTIVARIATE_TIMESERIES": ("surrogate_dynamics.datamodules.timeseries.multivariate",  "MultivariateTimeseriesDataModule"),
    "ATOMISTIC_TRAJECTORY":    ("surrogate_dynamics.datamodules.atomistic_trajectory",     "AtomisticTrajectoryDataModule"),
}

for _name, (_source, _attr) in _BUILTINS.items():
    register("datamodules", _name, _source, _attr)          # uses the "source:attr" form

__getattr__, __dir__ = lazy_exports(
    __name__, {t: f"{m}:{t}" for m, t in _BUILTINS.values()}
)

__all__ = ["TimeseriesDataModule"] + [t for m, t in _BUILTINS.values()]