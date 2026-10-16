import importlib

# Maps the canonical identifier (must match _IDENTIFIER on the class) to its
# (module_path, class_name).  Adding a new reader = one line here.
_REGISTRY: dict = {
    "ALA2":           ("inferenceplotters.ala2",           "Ala2Writer"),
}


def get_inference_plotter(inference_plotter_type: str):
    """Return the inference plotter class for *inference_plotter_type*.

    Raises:
        ValueError: If *inference_plotter_type* is not registered.
    """
    if inference_plotter_type not in _REGISTRY:
        raise ValueError(
            f"Unknown inference plotter type: '{inference_plotter_type}'. "
            f"Available: {sorted(_REGISTRY)}"
        )
    module_path, class_name = _REGISTRY[inference_plotter_type]
    return getattr(importlib.import_module(module_path), class_name)
