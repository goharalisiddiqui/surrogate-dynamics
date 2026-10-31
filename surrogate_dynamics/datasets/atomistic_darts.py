import inspect
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import ase
import numpy as np
import torch
import torch.nn as nn
import yaml
from darts import TimeSeries
from torch.utils.data import DataLoader

from collective_encoder.common.registry import get_module, register
from collective_encoder.datamodules.coordinates import CoordinatesDataModule
from collective_encoder.datareaders.base import BaseDataReader
from collective_encoder.datareaders.processors.chunk import ChunkProcessor
from collective_encoder.datasets.base import BaseDataset
from collective_encoder.datasets.resolver import get_dataset_cls_dl
from surrogate_dynamics.datamodules.atomistic_trajectory import (
    AtomisticTrajectoryDataModule,
)
from surrogate_dynamics.datasets.darts_timeseries import DartsTimeseriesDataset


class AtomisticDarts(DartsTimeseriesDataset):
    """
    Atomistic trajectory dataset backed by :class:`darts.TimeSeries` that inherits
    from :class:`~surrogate_dynamics.datasets.darts_timeseries.DartsTimeseriesDataset`.

    Designed to work with
    :class:`~surrogate_dynamics.datamodules.atomistic_trajectory.AtomisticTrajectoryDataModule`
    (and :class:`~collective_encoder.datamodules.coordinates.CoordinatesDataModule`).
    Takes a list of :class:`ase.Atoms` structures representing consecutive trajectory
    frames grouped into sequences of length ``sequence_length``, extracts per-frame
    atomistic features using any coordinate dataset from ``collective_encoder.datasets``
    (e.g., ``POSITIONS``, ``DISTANCES``, ``BONDGRAPH``, ``BONDGRAPH_LATENT``, ``SOAP``,
    ``SOAP_POWER_SPECTRUM``), passes each frame's representation through an embedding
    layer (by default :class:`torch.nn.Flatten`), reshapes the embedded frame vectors
    into a 3-D array of shape ``(n_sequences, sequence_length, embedding_dim)``, and
    initializes the parent :class:`DartsTimeseriesDataset`.

    Parameters
    ----------
    structures : list of ase.Atoms, optional
        Flat list of ``n_sequences * sequence_length`` molecular structures read by
        the trajectory datareader.
    labels : list or numpy.ndarray, optional
        Target labels corresponding to each structure (length ``n_sequences * sequence_length``)
        or to each sequence (length ``n_sequences``).
    args : dict, optional
        Configuration dictionary supporting all :class:`DartsTimeseriesDataset`
        arguments plus:
        - ``feature_dataset`` (str or type, required): Identifier (e.g., ``"POSITIONS"``,
          ``"DISTANCES"``, ``"BONDGRAPH"``, ``"SOAP"``) or class of the coordinate
          feature dataset in ``collective_encoder.datasets``.
        - ``feature_dataset_args`` (dict, optional): Arguments forwarded to the
          underlying coordinate feature dataset.
        - ``embedding_layer`` (torch.nn.Module or str or Path, optional): PyTorch module
          or path to a saved PyTorch / TorchScript module that maps each frame's extracted
          features to a 1-D embedding vector. Defaults to ``None``, which uses
          ``torch.nn.Flatten(start_dim=1)``.
        - ``chunk_length`` (int, optional): Number of frames per sequence chunk when
          set by datareader processors.
    datapoints : numpy.ndarray or Sequence[TimeSeries], optional
        Pre-computed 3-D time-series array or sequence of :class:`darts.TimeSeries`
        objects (supported when ``structures`` is omitted).
    kwargs : dict
        Additional keyword arguments forwarded to :class:`DartsTimeseriesDataset`.
    """

    _IDENTIFIER = "ATOMISTIC_DARTS"
    _REQUIRED_ARGS: List[str] = DartsTimeseriesDataset._REQUIRED_ARGS + [
        "feature_dataset",
    ]
    _OPTIONAL_ARGS: Dict[str, Any] = {
        **DartsTimeseriesDataset._OPTIONAL_ARGS,
        "feature_dataset_args": None,
        "embedding_layer": None,
        "chunk_length": None,
        "model_type": None,
        "network_type": None,
    }

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        # ChunkProcessor.get_dataset_wrapper dynamically creates a WrapperDataset
        # subclass whose __getitem__ attempts to re-chunk the returned item.
        # Because AtomisticDarts already groups frames into sequences during __init__
        # and returns (sample_dict, label) from DartsTimeseriesDataset.__getitem__,
        # preserve AtomisticDarts.__getitem__ on WrapperDataset.
        if cls.__name__ == "WrapperDataset":
            cls.__getitem__ = AtomisticDarts.__getitem__

    def _print_init_banner(self) -> None:
        if getattr(self, "_init_banner_printed", False):
            return
        self._init_banner_printed = True
        super()._print_init_banner()

    @classmethod
    def _prepare_dataset(
        cls,
        dataset_args: Dict[str, Any],
        datareader: Optional[BaseDataReader],
    ) -> Tuple[type, Dict[str, Any], type]:
        """
        Prepare dataset arguments and underlying ``feature_dataset_args`` using
        metadata from ``datareader`` when available.
        """
        prepared_args = dict(dataset_args or {})
        feature_dataset = prepared_args.get("feature_dataset")
        feature_dataset_args = dict(prepared_args.get("feature_dataset_args") or {})

        if feature_dataset is not None and datareader is not None:
            if isinstance(feature_dataset, str):
                _, feature_dataset_args, _ = get_dataset_cls_dl(
                    feature_dataset, feature_dataset_args, datareader
                )
            elif isinstance(feature_dataset, type) and issubclass(
                feature_dataset, BaseDataset
            ):
                _, feature_dataset_args, _ = feature_dataset._prepare_dataset(
                    feature_dataset_args, datareader
                )
        prepared_args["feature_dataset_args"] = feature_dataset_args

        if datareader is not None and hasattr(datareader, "get_processors"):
            chunk_len: Optional[int] = None
            cg_window: int = 1
            for proc in datareader.get_processors():
                proc_id = getattr(proc, "_IDENTIFIER", "")
                if proc_id == "CHUNK":
                    chunk_len = getattr(proc, "chunk_length", None) or getattr(
                        proc, "sequence_length", None
                    )
                elif proc_id == "COARSE_GRAIN":
                    cg_window = int(getattr(proc, "cg_window", 1) or 1)
            if chunk_len is not None and prepared_args.get("chunk_length") is None:
                prepared_args["chunk_length"] = max(1, int(chunk_len) // cg_window)

        return super()._prepare_dataset(prepared_args, datareader)

    def __init__(
        self,
        structures: Optional[List[ase.Atoms]] = None,
        labels: Optional[Union[np.ndarray, Sequence[Any]]] = None,
        args: Optional[Dict[str, Any]] = None,
        datapoints: Optional[
            Union[np.ndarray, Sequence[Union[np.ndarray, TimeSeries]]]
        ] = None,
        **kwargs,
    ):
        BaseDataset.__init__(self, args=args, **kwargs)

        if labels is None:
            labels = []

        if structures is not None:
            self.feature_dataset_instance = self._create_feature_dataset(
                structures=structures,
                labels=labels,
                **kwargs,
            )
            self.embedding_model = self._load_embedding_layer(self.embedding_layer)
            embedded_frames = self._embed_dataset(
                self.feature_dataset_instance, self.embedding_model
            )
            datapoints, grouped_labels = self._reshape_sequences(
                embedded_frames, labels
            )
        elif datapoints is not None:
            self.feature_dataset_instance = None
            self.embedding_model = self._load_embedding_layer(self.embedding_layer)
            grouped_labels = labels
        else:
            self.raise_error(
                "Either 'structures' or 'datapoints' must be provided to AtomisticDarts."
            )

        super().__init__(
            datapoints=datapoints,
            labels=grouped_labels,
            args=self.args,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # Internal Helpers: Feature Dataset & Embedding Layer
    # ------------------------------------------------------------------

    def _create_feature_dataset(
        self,
        structures: List[ase.Atoms],
        labels: Union[np.ndarray, Sequence[Any]],
        **kwargs,
    ) -> BaseDataset:
        """
        Instantiate the configured coordinate dataset from ``collective_encoder.datasets``.
        """
        if isinstance(self.feature_dataset, str):
            feature_dataset_cls = get_module("datasets", self.feature_dataset)
        elif isinstance(self.feature_dataset, type) and issubclass(
            self.feature_dataset, BaseDataset
        ):
            feature_dataset_cls = self.feature_dataset
        else:
            self.raise_error(
                f"Invalid feature_dataset={self.feature_dataset!r}. Expected a dataset "
                "identifier string or BaseDataset subclass."
            )

        f_args = dict(self.feature_dataset_args or {})
        init_params = inspect.signature(feature_dataset_cls.__init__).parameters

        call_kwargs = dict(kwargs)
        call_kwargs["args"] = f_args
        if "dataset_args" in init_params:
            call_kwargs["dataset_args"] = f_args

        # Ensure per-frame dummy labels exist if the inner dataset requires len(labels) == len(structures)
        inner_labels = (
            labels
            if (labels is not None and len(labels) == len(structures))
            else [[0.0] for _ in range(len(structures))]
        )

        self.log_info(
            f"Instantiating inner feature dataset '{feature_dataset_cls.__name__}' "
            f"for {len(structures)} frames..."
        )
        return feature_dataset_cls(
            structures=structures,
            labels=inner_labels,
            **call_kwargs,
        )

    def _load_model_from_dir(self, load_path: str) -> nn.Module:
        """
        Load a :class:`~collective_encoder.models.base.CEModelBase` from a run directory
        following the checkpoint resolution and config loading logic in
        ``collective_encoder.ce_run_base.load_model``.
        """
        ckpt_path = os.path.join(load_path, "checkpoints")
        if not os.path.isdir(ckpt_path):
            self.raise_error(
                f"Checkpoint directory not found: {ckpt_path}", FileNotFoundError
            )
        potential_ckpts = [
            a for a in os.listdir(ckpt_path) if a.endswith(".ckpt")
        ]
        if len(potential_ckpts) == 0:
            self.raise_error(
                f"No checkpoint found in {ckpt_path}", FileNotFoundError
            )
        for name in ["best", "saved", "last"]:
            if name + ".ckpt" in potential_ckpts:
                nn_ckpt = os.path.join(ckpt_path, name + ".ckpt")
                break
        else:
            self.log_warn(
                "No 'best', 'saved', or 'last' checkpoint found. "
                "Using the first available checkpoint."
            )
            nn_ckpt = os.path.join(ckpt_path, potential_ckpts[0])
        self.log_info("Loading network from: " + nn_ckpt)

        # Resolve the network/model type from run_config.yaml or dataset args
        run_config_file = os.path.join(load_path, "run_config.yaml")
        if os.path.isfile(run_config_file):
            with open(run_config_file, "r") as f:
                loaded_config = yaml.safe_load(f) or {}
            nn_type = loaded_config.get(
                "network_type", loaded_config.get("model_type", None)
            )
            if nn_type is None:
                self.raise_error(
                    "Missing 'network_type' or 'model_type' in loaded configuration."
                )
        else:
            nn_type = getattr(self, "network_type", None) or getattr(
                self, "model_type", None
            )
            if nn_type is None:
                self.raise_error(
                    "'run_config.yaml' not found in the loaded model directory, "
                    "'model_type' must be specified in the current configuration "
                    "to load the model."
                )

        nn_args_saved = torch.load(
            nn_ckpt,
            weights_only=False,
            map_location=torch.device("cpu"),
        )["hyper_parameters"]["args"]
        nn_cls = get_module("models", nn_type)
        model = nn_cls.load_from_checkpoint(
            nn_ckpt,
            args=nn_args_saved,
            map_location=torch.device("cpu"),
            **self.get_run_args(),
        )
        return model

    def _load_embedding_layer(
        self, embedding_layer: Optional[Union[nn.Module, str, Path]]
    ) -> nn.Module:
        """
        Resolve ``embedding_layer`` into an evaluated :class:`torch.nn.Module`.
        Defaults to :class:`torch.nn.Flatten` when ``embedding_layer`` is ``None``.
        If a directory path is provided, loads the model checkpoint using the same
        resolution logic as ``collective_encoder.ce_run_base.load_model`` and marks
        it to use ``get_latent()`` for embedding.
        """
        self._use_get_latent: bool = False
        if embedding_layer is None:
            if self.feature_dataset in ("BONDGRAPH", "BONDGRAPH_LATENT"):
                self.raise_error(
                    "Embedding layer must be provided for BONDGRAPH or BONDGRAPH_LATENT "
                    "feature datasets, as they produce graph objects instead of tensors."
                )
            model: nn.Module = nn.Flatten(start_dim=1)
        elif isinstance(embedding_layer, nn.Module):
            model = embedding_layer
        elif isinstance(embedding_layer, (str, Path, os.PathLike)):
            path_str = str(embedding_layer)
            if not os.path.exists(path_str):
                self.raise_error(
                    f"Embedding layer path not found: {path_str}", FileNotFoundError
                )
            if os.path.isdir(path_str):
                model = self._load_model_from_dir(path_str)
                self._use_get_latent = True
            else:
                try:
                    model = torch.jit.load(path_str, map_location="cpu")
                except Exception:
                    loaded = torch.load(
                        path_str, map_location="cpu", weights_only=False
                    )
                    if not isinstance(loaded, nn.Module):
                        self.raise_error(
                            f"Loaded object from {path_str!r} is of type {type(loaded).__name__}, "
                            "expected a torch.nn.Module or TorchScript module."
                        )
                    model = loaded
        else:
            self.raise_error(
                f"Unsupported embedding_layer type {type(embedding_layer).__name__}. "
                "Expected None, torch.nn.Module, or path to a saved module or model directory."
            )

        model.eval()
        for param in model.parameters():
            param.requires_grad = False
        return model

    def _get_model_dtype(self, model: nn.Module) -> torch.dtype:
        """
        Infer the floating-point dtype expected by ``model`` from its parameters/buffers.
        """
        for param in model.parameters():
            if param.is_floating_point():
                return param.dtype
        for buf in model.buffers():
            if buf.is_floating_point():
                return buf.dtype
        return self.dtype

    def _encode_single_item(self, model: nn.Module, item: Any) -> torch.Tensor:
        """
        Encode a single frame's feature representation (tensor or PyG Data) into a 1-D tensor.
        """
        model_dtype = self._get_model_dtype(model)
        features = item[0] if isinstance(item, tuple) else item

        if isinstance(features, np.ndarray):
            features = torch.as_tensor(features, dtype=model_dtype)

        if isinstance(features, torch.Tensor):
            if features.is_floating_point() and features.dtype != model_dtype:
                features = features.to(dtype=model_dtype)
            batched = features.unsqueeze(0)
            if getattr(self, "_use_get_latent", False):
                try:
                    out = model.get_latent(batched)
                except RuntimeError:
                    out = model.get_latent(batched.flatten(start_dim=1))
            elif hasattr(model, "_encode"):
                out = model._encode(batched)
            else:
                try:
                    out = model(batched)
                except RuntimeError:
                    out = model(batched.flatten(start_dim=1))
        else:
            # Graph / PyG Data object
            if getattr(self, "_use_get_latent", False):
                out = model.get_latent(features)
            elif hasattr(model, "_encode"):
                out = model._encode(features)
            else:
                out = model(features)

        if isinstance(out, (tuple, list)):
            out = out[0]
        if not isinstance(out, torch.Tensor):
            out = torch.as_tensor(out, dtype=self.dtype)
        return out.detach().cpu().reshape(-1)

    def _embed_dataset(
        self, feature_ds: BaseDataset, model: nn.Module
    ) -> np.ndarray:
        """
        Pass all frames from ``feature_ds`` through ``model`` and return a 2-D NumPy
        array of shape ``(total_frames, embedding_dim)``.
        """
        n_frames = len(feature_ds)
        if n_frames == 0:
            return np.empty((0, 0), dtype=np.float32)

        model_dtype = self._get_model_dtype(model)

        with torch.no_grad():
            first_item = feature_ds[0]
            first_feat = first_item[0] if isinstance(first_item, tuple) else first_item

            # Fast batch path for tensor-based feature datasets
            if isinstance(first_feat, (torch.Tensor, np.ndarray)):
                all_feats = []
                for i in range(n_frames):
                    item = feature_ds[i]
                    f = item[0] if isinstance(item, tuple) else item
                    t = (
                        f
                        if isinstance(f, torch.Tensor)
                        else torch.as_tensor(f, dtype=model_dtype)
                    )
                    if t.is_floating_point() and t.dtype != model_dtype:
                        t = t.to(dtype=model_dtype)
                    all_feats.append(t)
                batch_tensor = torch.stack(all_feats, dim=0)
                try:
                    if getattr(self, "_use_get_latent", False):
                        try:
                            out = model.get_latent(batch_tensor)
                        except RuntimeError:
                            out = model.get_latent(batch_tensor.flatten(start_dim=1))
                    elif hasattr(model, "_encode"):
                        out = model._encode(batch_tensor)
                    else:
                        try:
                            out = model(batch_tensor)
                        except RuntimeError:
                            out = model(batch_tensor.flatten(start_dim=1))
                    if isinstance(out, (tuple, list)):
                        out = out[0]
                    if isinstance(out, torch.Tensor) and out.shape[0] == n_frames:
                        return (
                            out.detach()
                            .cpu()
                            .reshape(n_frames, -1)
                            .numpy()
                            .astype(np.float32)
                        )
                except Exception:
                    pass

            # Per-frame fallback (e.g. for PyG Data objects or per-sample encoders)
            encoded_list = [
                self._encode_single_item(model, feature_ds[i]).numpy()
                for i in range(n_frames)
            ]

        return np.stack(encoded_list, axis=0).astype(np.float32)

    def _get_effective_sequence_length(self) -> int:
        """
        Determine the number of frames per sequence chunk.
        """
        if self.chunk_length is not None:
            return int(self.chunk_length)
        if self.sequence_length is not None:
            return int(self.sequence_length)
        return int(self.context_length + self.prediction_length)

    def _reshape_sequences(
        self,
        embedded_frames: np.ndarray,
        labels: Union[np.ndarray, Sequence[Any]],
    ) -> Tuple[np.ndarray, Union[np.ndarray, List[np.ndarray]]]:
        """
        Reshape flat embedded frames ``(n_sequences * seq_len, embedding_dim)`` into a
        3-D array ``(n_sequences, seq_len, embedding_dim)`` and group frame labels
        into ``n_sequences`` chunks.
        """
        total_frames = len(embedded_frames)
        seq_len = self._get_effective_sequence_length()
        if seq_len <= 0:
            self.raise_error(
                f"Effective sequence_length must be positive, got {seq_len}."
            )

        if total_frames == 0:
            return np.empty((0, seq_len, 0), dtype=np.float32), []

        if total_frames % seq_len != 0:
            self.raise_error(
                f"Total number of embedded frames ({total_frames}) is not divisible "
                f"by sequence length ({seq_len})."
            )

        n_sequences = total_frames // seq_len
        embedding_dim = embedded_frames.shape[-1]
        embedded_3d = embedded_frames.reshape(n_sequences, seq_len, embedding_dim)

        if labels is None or len(labels) == 0:
            grouped_labels: List[np.ndarray] = [
                np.empty((0,), dtype=np.float32) for _ in range(n_sequences)
            ]
        elif len(labels) == total_frames and seq_len > 1:
            grouped_labels = [
                np.asarray(labels[i * seq_len : (i + 1) * seq_len], dtype=np.float32)
                for i in range(n_sequences)
            ]
        elif len(labels) == n_sequences:
            grouped_labels = [
                np.asarray(labels[i], dtype=np.float32) for i in range(n_sequences)
            ]
        else:
            self.raise_error(
                f"Number of labels ({len(labels)}) must match either total frames "
                f"({total_frames}) or number of sequences ({n_sequences})."
            )

        return embedded_3d, grouped_labels
