import importlib

_REGISTRY: dict = {
    "TFT":    ("propagators.tft",   "PropagatorTFT"),
    "TFT_DEV": ("propagators.tft_dev", "PropagatorTFT"),
    "TFT_LATENT": ("propagators.tft_latent", "PropagatorTFTLatent"),
    "BGE_TFT": ("propagators.bge_tft", "BondGraphEncoderTFT"),
    "BGE_TFT_V2": ("propagators.bge_tft_v2", "BondGraphEncoderTFT"),
}   


def get_propagator(model_name: str):
    """Return the neural network class for *model_name*.

    Raises:
        ValueError: If *model_name* is not registered.
    """
    if model_name not in _REGISTRY:
        raise ValueError(
            f"Unknown model name: '{model_name}'. "
            f"Available: {sorted(_REGISTRY)}"
        )
    module_path, class_name = _REGISTRY[model_name]
    return getattr(importlib.import_module(module_path), class_name)