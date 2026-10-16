import collective_encoder
from collective_encoder.common.registry import register, lazy_exports

_BUILTINS = {
    "TIMESERIES": ("surrogate_dynamics.datasets.timeseries", "TimeseriesDataset"),
}

for _name, (_source, _attr) in _BUILTINS.items():
    register("datasets", _name, _source, _attr)          # uses the "source:attr" form

__getattr__, __dir__ = lazy_exports(
    __name__, {t: f"{m}:{t}" for m, t in _BUILTINS.values()}
)

__all__ = [t for m, t in _BUILTINS.values()]
