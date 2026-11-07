"""
Test plotting modules for surrogate dynamics models in ``surrogate_dynamics``.

Inherits from :class:`~collective_encoder.testplotters.simple.SimplePlotter` and
registers time-series decoding test plotters with the shared ``collective_encoder``
registry.
"""

import collective_encoder
from collective_encoder.common.registry import lazy_exports, register

_BUILTINS = {
    "SIMPLE": (
        "surrogate_dynamics.testplotters.simple",
        "TimeseriesSimplePlotter",
    ),
    "TIMESERIES": (
        "surrogate_dynamics.testplotters.simple",
        "TimeseriesSimplePlotter",
    ),
    "MSM": (
        "surrogate_dynamics.testplotters.msm",
        "MSMPlotter",
    ),
    "TRANSFER_OPERATOR": (
        "surrogate_dynamics.testplotters.transfer_operator",
        "TransferOperatorPlotter",
    ),
    "IMPLIED_TIMESCALES": (
        "surrogate_dynamics.testplotters.transfer_operator",
        "TransferOperatorPlotter",
    ),
    "INTERNAL_ENERGY": (
        "surrogate_dynamics.testplotters.md_energy",
        "InternalEnergyPlotter",
    ),
    "MD_ENERGY": (
        "surrogate_dynamics.testplotters.md_energy",
        "InternalEnergyPlotter",
    ),
}

for _name, (_source, _attr) in _BUILTINS.items():
    register("testplotters", _name, _source, _attr, overwrite=True)

__getattr__, __dir__ = lazy_exports(
    __name__, {t: f"{m}:{t}" for m, t in _BUILTINS.values()}
)

__all__ = list({t for m, t in _BUILTINS.values()})
