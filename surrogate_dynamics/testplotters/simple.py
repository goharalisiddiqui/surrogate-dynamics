from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch

from collective_encoder.testplotters.simple import SimplePlotter
from surrogate_dynamics.utils import get_model_dtype, load_embedding_layer


class TimeseriesSimplePlotter(SimplePlotter):
    """
    Test plotter for surrogate dynamics models that decodes latent trajectory
    predictions through an embedding model before running :class:`SimplePlotter`.

    Inherits all selection, angle resolution (``cos_sin_to_angle``), 2D scatter
    plotting, and correlation matrix functionality from
    :class:`~collective_encoder.testplotters.simple.SimplePlotter`, while decoding
    collected forecast predictions ``pred`` via ``embedding_model.get_decoded()``
    (or ``embedding_model()``) prior to applying ``pred_selection`` /
    ``decode_selection``.

    Parameters
    ----------
    args : dict, optional
        Configuration dictionary for the test plotter.
        Required keys:
        - ``embedding_model`` (torch.nn.Module or str or pathlib.Path): Trained
          encoder-decoder model or run directory path used to decode latent
          predictions back into physical coordinates. Automatically injected from
          the datamodule during testing when not explicitly provided in YAML.
        Optional keys (in addition to :class:`SimplePlotter`):
        - ``decode_selection`` (dict, optional): Selection dictionary applied to
          decoded predictions (alias for ``pred_selection``).
        - ``datamodule`` (BaseDataModule, optional): Active datamodule instance
          injected during testing for compatibility validation and topology access.
    kwargs : dict
        Additional keyword arguments forwarded to
        :class:`~collective_encoder.common.module.CEModule`.
    """

    _IDENTIFIER = "SIMPLE"
    _COMPATIBLE_DATAMODULES: List[str] = []
    _COMPATIBLE_EMBEDDINGS: List[str] = []
    _REQUIRED_ARGS = SimplePlotter._REQUIRED_ARGS + ["embedding_model"]
    _OPTIONAL_ARGS = SimplePlotter._OPTIONAL_ARGS.copy()
    _OPTIONAL_ARGS.update({
        "decode_selection": None,
        "datamodule": None,
    })

    def __init__(self, args: Optional[Dict[str, Any]] = None, **kwargs) -> None:
        """
        Initialize ``TimeseriesSimplePlotter``, resolve ``embedding_model``, and
        validate datamodule and embedding compatibility.

        Parameters
        ----------
        args : dict, optional
            Configuration dictionary containing ``embedding_model`` and plotting
            options.
        **kwargs
            Runtime keyword arguments forwarded to ``SimplePlotter``.
        """
        super().__init__(args=args, **kwargs)
        if self.embedding_model is None and self.datamodule is not None:
            ds = (
                self.datamodule.get_any_dataset()
                if hasattr(self.datamodule, "get_any_dataset")
                else None
            )
            if ds is not None:
                self.embedding_model = (
                    getattr(ds, "embedding_model", None)
                    or getattr(ds, "embedding_layer", None)
                )
        self.embedding_model, _ = load_embedding_layer(self, self.embedding_model)
        self.check_module_compatibility()
        if self.decode_selection is not None and self.pred_selection is None:
            self.pred_selection = self.decode_selection

    @classmethod
    def get_compatible_datamodules(cls) -> List[str]:
        """
        Return the list of compatible datamodule identifiers for this test plotter.

        Returns
        -------
        list of str
            Compatible datamodule identifiers (empty list means all datamodules are
            compatible).
        """
        return cls._COMPATIBLE_DATAMODULES

    @classmethod
    def get_compatible_embeddings(cls) -> List[str]:
        """
        Return the list of compatible embedding model identifiers for this test plotter.

        Returns
        -------
        list of str
            Compatible embedding model identifiers (empty list means all embeddings
            are compatible).
        """
        return cls._COMPATIBLE_EMBEDDINGS

    def check_datamodule_compatibility(self, datamodule: Any) -> None:
        """
        Verify that ``datamodule`` is compatible with ``_COMPATIBLE_DATAMODULES``.

        Parameters
        ----------
        datamodule : Any
            DataModule instance or identifier string to validate.

        Raises
        ------
        ValueError
            If ``_COMPATIBLE_DATAMODULES`` is non-empty and ``datamodule`` is missing
            or not listed in ``_COMPATIBLE_DATAMODULES``.
        """
        compatible_dmods = self.get_compatible_datamodules()
        if not compatible_dmods:
            return
        if datamodule is None:
            self.raise_error(
                f"{type(self).__name__} requires a compatible datamodule "
                f"{compatible_dmods}, but 'datamodule' was None."
            )
        dm_id = (
            datamodule
            if isinstance(datamodule, str)
            else getattr(datamodule, "_IDENTIFIER", None)
        )
        dm_cls_name = (
            datamodule if isinstance(datamodule, str) else type(datamodule).__name__
        )
        if dm_id not in compatible_dmods and dm_cls_name not in compatible_dmods:
            self.raise_error(
                f"Datamodule type '{dm_id or dm_cls_name}' is not compatible with "
                f"{type(self).__name__}, which only supports {compatible_dmods}."
            )

    def check_embedding_compatibility(self, embedding_model: Any) -> None:
        """
        Verify that ``embedding_model`` is compatible with ``_COMPATIBLE_EMBEDDINGS``.

        Parameters
        ----------
        embedding_model : Any
            Embedding model instance or identifier string to validate.

        Raises
        ------
        ValueError
            If ``_COMPATIBLE_EMBEDDINGS`` is non-empty and ``embedding_model`` is
            missing or not listed in ``_COMPATIBLE_EMBEDDINGS``.
        """
        compatible_embs = self.get_compatible_embeddings()
        if not compatible_embs:
            return
        if embedding_model is None:
            self.raise_error(
                f"{type(self).__name__} requires a compatible embedding model "
                f"{compatible_embs}, but 'embedding_model' was None."
            )
        emb_id = (
            embedding_model
            if isinstance(embedding_model, str)
            else getattr(embedding_model, "_IDENTIFIER", None)
        )
        emb_cls_name = (
            embedding_model
            if isinstance(embedding_model, str)
            else type(embedding_model).__name__
        )
        if emb_id not in compatible_embs and emb_cls_name not in compatible_embs:
            self.raise_error(
                f"Embedding model '{emb_id or emb_cls_name}' is not compatible with "
                f"{type(self).__name__}, which only supports {compatible_embs}."
            )

    def check_module_compatibility(self) -> None:
        """
        Check that both ``self.datamodule`` and ``self.embedding_model`` satisfy
        ``_COMPATIBLE_DATAMODULES`` and ``_COMPATIBLE_EMBEDDINGS``.
        """
        self.check_datamodule_compatibility(self.datamodule)
        self.check_embedding_compatibility(self.embedding_model)

    def _decode_predictions(
        self, pred: Any
    ) -> Union[np.ndarray, Dict[str, np.ndarray], None]:
        """
        Decode accumulated latent predictions using ``self.embedding_model``.

        Parameters
        ----------
        pred : numpy.ndarray or torch.Tensor or None
            Accumulated latent predictions of shape ``(n_samples, latent_dim)``.

        Returns
        -------
        numpy.ndarray or dict of str to numpy.ndarray or None
            Decoded predictions converted to NumPy arrays (either a 2-D array or a
            dictionary of arrays such as ``{'bond_dist': ..., 'angle': ...,
            'dihedral_cos': ..., 'dihedral_sin': ...}``).
        """
        if pred is None or self.embedding_model is None:
            return pred

        model_dtype = get_model_dtype(self.embedding_model)
        try:
            device = next(self.embedding_model.parameters()).device
        except StopIteration:
            device = torch.device("cpu")

        pred_tensor = torch.as_tensor(pred, dtype=model_dtype, device=device)
        if pred_tensor.ndim > 2:
            pred_tensor = pred_tensor.reshape(-1, pred_tensor.shape[-1])

        with torch.no_grad():
            if hasattr(self.embedding_model, "get_decoded"):
                decoded = self.embedding_model.get_decoded(pred_tensor)
            else:
                decoded = self.embedding_model(pred_tensor)

        if isinstance(decoded, dict):
            return {k: self._convert_data(v) for k, v in decoded.items()}
        return self._convert_data(decoded)

    def _prepare_labels_for_selection(
        self, labels: Union[Dict[str, np.ndarray], np.ndarray, None]
    ) -> Union[Dict[str, np.ndarray], np.ndarray, None]:
        """
        Format accumulated labels to match the index type in ``self.labels_selection``.

        When ``labels`` is collected as a dictionary mapping column names to 1-D
        arrays (the default in :meth:`CEModelBase._plot_test_batch`) but
        ``self.labels_selection`` specifies integer column indices, stacks the
        label columns into a 2-D NumPy array so integer indexing succeeds.

        Parameters
        ----------
        labels : dict of str to numpy.ndarray or numpy.ndarray or None
            Accumulated ground-truth labels.

        Returns
        -------
        dict of str to numpy.ndarray or numpy.ndarray or None
            Labels formatted for :meth:`BasePlotter._parse_selection`.
        """
        if (
            isinstance(labels, dict)
            and isinstance(self.labels_selection, dict)
            and len(self.labels_selection) > 0
        ):
            first_sel_val = next(iter(self.labels_selection.values()))
            if isinstance(first_sel_val, int):
                if len(labels) == 0:
                    return np.empty((0, 0))
                return np.column_stack(
                    [self._convert_to_numpy(v).reshape(-1) for v in labels.values()]
                )
        return labels

    def plot(
        self,
        data: Any,
        latent: Any,
        pred: Any,
        labels: Any,
        meta: Any,
    ) -> None:
        """
        Decode accumulated predictions and execute :meth:`SimplePlotter.plot`.

        Parameters
        ----------
        data : Any
            Accumulated inputs.
        latent : dict or numpy.ndarray
            Accumulated latent representations.
        pred : numpy.ndarray
            Accumulated latent predictions from the propagator.
        labels : dict or numpy.ndarray
            Accumulated ground-truth labels.
        meta : dict
            Accumulated metadata.
        """
        if self.decode_selection is not None and self.pred_selection is None:
            self.pred_selection = self.decode_selection

        decoded_pred = self._decode_predictions(pred)
        prepared_labels = self._prepare_labels_for_selection(labels)
        super().plot(data, latent, decoded_pred, prepared_labels, meta)
