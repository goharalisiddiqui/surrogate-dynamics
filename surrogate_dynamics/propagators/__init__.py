

import collective_encoder
from collective_encoder.common.registry import register, lazy_exports

_BUILTINS = {
    "TFT":    ("surrogate_dynamics.propagators.tft",   "PropagatorTFT"),
    "TFT_DEV": ("surrogate_dynamics.propagators.tft_dev", "PropagatorTFT"),
    "TFT_LATENT": ("surrogate_dynamics.propagators.tft_latent", "PropagatorTFTLatent"),
    "BGE_TFT": ("surrogate_dynamics.propagators.bge_tft", "BondGraphEncoderTFT"),
    "BGE_TFT_V2": ("surrogate_dynamics.propagators.bge_tft_v2", "BondGraphEncoderTFT"),
}   
for _name, (_source, _attr) in _BUILTINS.items():
    register("models", _name, _source, _attr)          # uses the "source:attr" form

__getattr__, __dir__ = lazy_exports(
    __name__, {t: f"{m}:{t}" for m, t in _BUILTINS.values()}
)

__all__ = [t for m, t in _BUILTINS.values()]