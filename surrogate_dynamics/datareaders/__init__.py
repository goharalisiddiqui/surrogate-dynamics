

import collective_encoder
from collective_encoder.common.registry import register, lazy_exports
from .huggingface.base import HuggingfaceReaderBase

_BUILTINS = {
    "GIFT-EVAL": ("surrogate_dynamics.datareaders.huggingface.gift_eval", "GiftEvalReader"),
    "GIFT-EVAL-PRETRAIN": ("surrogate_dynamics.datareaders.huggingface.gift_eval", "GiftEvalReader"),
}

for _name, (_source, _attr) in _BUILTINS.items():
    register("datareaders", _name, _source, _attr)          # uses the "source:attr" form

__getattr__, __dir__ = lazy_exports(
    __name__, {t: f"{m}:{t}" for m, t in _BUILTINS.values()}
)

__all__ = ["HuggingfaceReaderBase"] + [t for m, t in _BUILTINS.values()]