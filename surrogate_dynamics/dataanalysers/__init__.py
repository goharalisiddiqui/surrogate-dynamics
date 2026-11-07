"""
Data analysis and visualization modules for time-series datasets in surrogate_dynamics.

Inherits from `collective_encoder.dataanalysers.simple.SimpleDataAnalyser` and registers
time-series analysers with the shared `collective_encoder` registry.
"""

import collective_encoder
from collective_encoder.common.registry import register, lazy_exports

_BUILTINS = {
    "TIMESERIES": (
        "surrogate_dynamics.dataanalysers.timeseries",
        "TimeseriesDataAnalyser",
    ),
}

for _name, (_source, _attr) in _BUILTINS.items():
    register("dataanalysers", _name, _source, _attr)

__getattr__, __dir__ = lazy_exports(
    __name__, {t: f"{m}:{t}" for m, t in _BUILTINS.values()}
)

__all__ = [t for m, t in _BUILTINS.values()]
