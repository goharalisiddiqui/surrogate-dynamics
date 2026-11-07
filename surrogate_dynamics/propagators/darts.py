import importlib
import pkgutil
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm

from collective_encoder import TENSOR_PRECISION
from collective_encoder.common.registry import get_module
from collective_encoder.models.neural_nets.base import CENetBase
from gslibs.utils.common import get_init_args, get_required_init_args
from surrogate_dynamics.likelihoods.resolver import likelihood_resolver


class DartsForecastingModel(CENetBase):
    """
    Propagator wrapper for Darts :class:`~darts.models.forecasting.torch_forecasting_model.TorchForecastingModel`
    architectures (such as ``TFTModel``, ``NBEATSModel``, ``TCNModel``, ``TiDEModel``, etc.).

    Instantiates the requested Darts forecasting model and its underlying PyTorch
    Lightning module (:class:`~darts.models.forecasting.pl_forecasting_module.PLForecastingModule`)
    as ``self.propagator``, operating directly in the latent / feature space without
    coupling an encoder-decoder inside the propagator (decoding is handled downstream
    by test plotters such as :class:`~surrogate_dynamics.testplotters.simple.TimeseriesSimplePlotter`).

    Parameters
    ----------
    args : dict, optional
        Configuration dictionary.
        Required keys:
        - ``darts_model_name`` (str): Class name of the Darts forecasting model in
          ``darts.models.forecasting`` (e.g., ``"TFTModel"``).
        Optional keys:
        - ``darts_model_args`` (dict): Keyword arguments forwarded to the Darts model
          constructor (must include ``input_chunk_length`` and ``output_chunk_length``
          unless inferred from the DataModule).
        - ``target_dim`` (int, optional): Target / latent feature dimension (inferred
          from DataModule when omitted).
        - ``feat_dynamic_real_dim`` (int, default 0): Future dynamic covariate dimension.
        - ``past_feat_dynamic_real_dim`` (int, default 0): Past dynamic covariate dimension.
        - ``static_covariates_dim`` (int, default 0): Static covariate dimension.
        - ``long_sequence_length`` (int, optional): Total sequence length per sample.
        - ``likelihood`` (str or object, optional): Optional likelihood model identifier
          or instance.
        - ``likelihood_args`` (dict, optional): Arguments for ``likelihood_resolver``.
        - ``loss_fn`` (torch.nn.Module, default ``nn.MSELoss()``): Point prediction loss.
        - ``inference_settings`` (dict, optional): Settings for ``predict_step`` rollout.
        - ``prediction_type`` (str, default ``"autoregressive"``): Evaluation forecasting
          mode (``"autoregressive"`` or ``"teacher_forced"``).
    **kwargs
        Additional runtime keyword arguments forwarded to :class:`CENetBase`.
    """

    _IDENTIFIER = "DARTS"
    _IDENTFIER = "DARTS"
    _REQUIRED_ARGS = [
        "darts_model_name",
    ]

    _OPTIONAL_ARGS = CENetBase._OPTIONAL_ARGS.copy()
    _OPTIONAL_ARGS.update({
        "darts_model_args": {},
        "target_dim": None,
        "feat_dynamic_real_dim": 0,
        "past_feat_dynamic_real_dim": 0,
        "static_covariates_dim": 0,
        "long_sequence_length": None,
        "likelihood": None,
        "likelihood_args": None,
        "loss_fn": nn.MSELoss(),
        "inference_settings": None,
        "prediction_type": "autoregressive",
    })

    _AUTOREGRESSIVE_TYPES = ("autoregressive", "autoregressive_rollout", "rollout")
    _TEACHER_FORCED_TYPES = ("teacher_forced", "teacher-forced", "teacher")

    @staticmethod
    def extract_args_from_datamodule(datamodule: Any, args: Dict[str, Any]) -> Dict[str, Any]:
        """
        Infer ``target_dim``, covariate dimensions, ``long_sequence_length``, and
        chunk lengths from ``datamodule`` when not already specified in ``args``.

        Parameters
        ----------
        datamodule : BaseDataModule
            DataModule instance providing dataset shape and sequence length metadata.
        args : dict
            Model configuration dictionary.

        Returns
        -------
        dict
            Updated model configuration dictionary.
        """
        out_args = (args or {}).copy()
        darts_args = dict(out_args.get("darts_model_args") or {})

        any_ds = (
            datamodule.get_any_dataset()
            if (datamodule is not None and hasattr(datamodule, "get_any_dataset"))
            else None
        )

        if any_ds is not None and hasattr(any_ds, "get_feature_dims"):
            dims = any_ds.get_feature_dims()
            if out_args.get("target_dim") is None:
                out_args["target_dim"] = int(dims["target_dim"])
            if not out_args.get("feat_dynamic_real_dim"):
                out_args["feat_dynamic_real_dim"] = int(dims.get("feat_dynamic_real_dim", 0))
            if not out_args.get("past_feat_dynamic_real_dim"):
                out_args["past_feat_dynamic_real_dim"] = int(
                    dims.get("past_feat_dynamic_real_dim", 0)
                )
            if not out_args.get("static_covariates_dim"):
                out_args["static_covariates_dim"] = int(dims.get("static_covariates_dim", 0))

            if darts_args.get("input_chunk_length") is None and getattr(
                any_ds, "context_length", None
            ) is not None:
                darts_args["input_chunk_length"] = int(any_ds.context_length)
            if darts_args.get("output_chunk_length") is None and getattr(
                any_ds, "prediction_length", None
            ) is not None:
                darts_args["output_chunk_length"] = int(any_ds.prediction_length)

        if out_args.get("target_dim") is None and datamodule is not None and hasattr(
            datamodule, "get_datapoint_shape"
        ):
            dp_shape = datamodule.get_datapoint_shape()
            if isinstance(dp_shape, (tuple, list)) and len(dp_shape) >= 1:
                out_args["target_dim"] = int(dp_shape[-1])

        if datamodule is not None:
            if darts_args.get("input_chunk_length") is None and getattr(
                datamodule, "input_chunk_length", None
            ) is not None:
                darts_args["input_chunk_length"] = int(datamodule.input_chunk_length)
            if darts_args.get("output_chunk_length") is None and getattr(
                datamodule, "output_chunk_length", None
            ) is not None:
                darts_args["output_chunk_length"] = int(datamodule.output_chunk_length)
            if out_args.get("long_sequence_length") is None and getattr(
                datamodule, "sequence_length", None
            ) is not None:
                out_args["long_sequence_length"] = int(datamodule.sequence_length)

        out_args["darts_model_args"] = darts_args
        return out_args

    def __init__(
        self,
        args: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> None:
        """
        Initialize ``DartsForecastingModel`` and its inner Darts ``PLForecastingModule``.

        Parameters
        ----------
        args : dict, optional
            Configuration dictionary.
        **kwargs
            Additional runtime keyword arguments forwarded to :class:`CENetBase`.
        """
        self.save_hyperparameters()
        super().__init__(args, **kwargs)

        self._validate_prediction_type()
        self._rollout_context: Optional[torch.Tensor] = None
        self._in_validation: bool = False

        if self.inference_settings is None:
            self.inference_settings = {}

        self.likelihood = self._resolve_likelihood()
        self._init_model()

        self.losses = {
            "loss_prop": self.loss_prop,
        }
        self.metrics = {
            "mae_prop": self.metric_mae_prop,
        }
        self.test_metrics = self.metrics.copy()

    def _get_train_params(self) -> List[nn.Parameter]:
        """
        Collect trainable parameters from the model and any parameterized loss modules.

        Returns
        -------
        list of torch.nn.Parameter
            All parameters to pass to the optimizer.
        """
        params_from_losses: List[nn.Parameter] = []
        for loss in self.losses.values():
            if hasattr(loss, "parameters"):
                params_from_losses.extend(loss.parameters())
        return list(self.parameters()) + params_from_losses

    # ------------------------------------------------------------------
    # Darts Model Discovery & Initialization
    # ------------------------------------------------------------------

    def _find_darts_model(self) -> type:
        """
        Locate and import ``self.darts_model_name`` from ``darts.models.forecasting``.

        Returns
        -------
        type
            The Darts forecasting model class matching ``self.darts_model_name``.
        """
        package_name = "darts.models.forecasting"
        package = importlib.import_module(package_name)

        for _, modname, _ in pkgutil.iter_modules(package.__path__):
            full_modname = f"{package_name}.{modname}"
            try:
                module = importlib.import_module(full_modname)
                cls_obj = getattr(module, self.darts_model_name)
                return cls_obj
            except Exception:
                continue

        self.raise_error(
            f"Could not import DARTS model '{self.darts_model_name}'. "
            "Ensure that the 'darts' package is installed and "
            "the model name is correct."
        )

    def _check_model_args(self, model_cls: type) -> None:
        """
        Verify that all required constructor arguments of ``model_cls`` are present
        in ``self.darts_model_args``.

        Parameters
        ----------
        model_cls : type
            Darts forecasting model class to inspect.
        """
        required_args = get_required_init_args(model_cls)
        missing_args = [
            arg for arg in required_args if arg not in (self.darts_model_args or {})
        ]
        if missing_args:
            self.raise_error(
                f"Missing required arguments for DARTS model '{self.darts_model_name}': "
                f"{missing_args}. Please provide these arguments in 'darts_model_args'."
            )

    def _build_dummy_train_sample(self) -> Tuple[Optional[np.ndarray], ...]:
        """
        Construct a 6-tuple ``train_sample`` of dummy NumPy arrays with the configured
        chunk lengths and feature dimensions for Darts ``_init_model()``.

        Returns
        -------
        tuple of (numpy.ndarray or None)
            ``(past_target, past_covariate, historic_future_covariate, future_covariate,
            static_covariates, future_target)``.
        """
        np_dtype = (
            np.float64 if TENSOR_PRECISION == torch.float64 else np.float32
        )
        t_in = int(self.input_chunk_length)
        t_out = int(self.output_chunk_length)
        d_target = int(self.latent_dim)
        d_past_cov = int(self.past_feat_dynamic_real_dim or 0)
        d_fut_cov = int(self.feat_dynamic_real_dim or 0)
        d_static = int(self.static_covariates_dim or 0)

        past_target = np.zeros((t_in, d_target), dtype=np_dtype)
        past_cov = (
            np.zeros((t_in, d_past_cov), dtype=np_dtype) if d_past_cov > 0 else None
        )
        hist_fut_cov = (
            np.zeros((t_in, d_fut_cov), dtype=np_dtype) if d_fut_cov > 0 else None
        )
        fut_cov = (
            np.zeros((t_out, d_fut_cov), dtype=np_dtype) if d_fut_cov > 0 else None
        )
        static_cov = (
            np.zeros((1, d_static), dtype=np_dtype) if d_static > 0 else None
        )
        future_target = np.zeros((t_out, d_target), dtype=np_dtype)
        return (
            past_target,
            past_cov,
            hist_fut_cov,
            fut_cov,
            static_cov,
            future_target,
        )

    def _resolve_likelihood(self) -> Optional[Any]:
        """
        Resolve and instantiate any configured likelihood model from ``self.likelihood``
        or ``self.darts_model_args``.

        Supports string identifiers (resolved via :func:`likelihood_resolver`),
        dictionary specifications (``{"type": ..., "args": ...}``), or pre-instantiated
        :class:`~darts.utils.likelihood_models.torch.TorchLikelihood` objects.

        Returns
        -------
        TorchLikelihood or None
            Instantiated likelihood model or ``None`` if deterministic loss is used.
        """
        darts_args = self.darts_model_args or {}
        raw_likelihood = (
            self.likelihood
            if self.likelihood is not None
            else darts_args.get("likelihood")
        )
        if raw_likelihood is None:
            return None

        merged_likelihood_args: Dict[str, Any] = {}
        if isinstance(darts_args.get("likelihood_args"), dict):
            merged_likelihood_args.update(darts_args["likelihood_args"])
        if isinstance(self.likelihood_args, dict):
            merged_likelihood_args.update(self.likelihood_args)

        return likelihood_resolver(raw_likelihood, merged_likelihood_args)

    def _init_model(self) -> None:
        """
        Instantiate the Darts forecasting wrapper and initialize its underlying
        PyTorch ``PLForecastingModule`` as ``self.propagator``.
        """
        model_cls = self._find_darts_model()
        self._check_model_args(model_cls)

        model_args = get_init_args(model_cls)
        model_kwargs = {
            arg: self.darts_model_args.get(arg)
            for arg in model_args
            if arg in self.darts_model_args
        }
        if self.likelihood is not None:
            model_kwargs["likelihood"] = self.likelihood
            if "loss_fn" in model_args:
                model_kwargs["loss_fn"] = None
        elif (
            "loss_fn" in model_args
            and model_kwargs.get("loss_fn") is None
            and self.loss_fn is not None
        ):
            model_kwargs["loss_fn"] = self.loss_fn

        self.forecast_model = model_cls(**model_kwargs)
        if (
            self.likelihood is None
            and getattr(self.forecast_model, "likelihood", None) is not None
        ):
            self.likelihood = self.forecast_model.likelihood
        self.input_chunk_length = int(self.forecast_model.input_chunk_length)
        self.output_chunk_length = int(self.forecast_model.output_chunk_length)
        self.sequence_length = self.input_chunk_length + self.output_chunk_length

        if self.target_dim is None:
            self.target_dim = int(self.darts_model_args.get("target_dim", 1) or 1)
        self.latent_dim = int(self.target_dim)

        if self.long_sequence_length is None:
            self.long_sequence_length = self.sequence_length

        self.forecast_model.train_sample = self._build_dummy_train_sample()
        self.propagator: nn.Module = self.forecast_model._init_model().to(
            dtype=TENSOR_PRECISION
        )

    # ------------------------------------------------------------------
    # Normalization
    # ------------------------------------------------------------------

    def _validate_norm_sizes(
        self, Mean: torch.Tensor, Range: torch.Tensor
    ) -> None:
        """
        Validate that ``Mean`` and ``Range`` normalization buffers match ``latent_dim``.

        Parameters
        ----------
        Mean : torch.Tensor
            Feature mean buffer.
        Range : torch.Tensor
            Feature range/scale buffer.
        """
        if Mean.size(0) != self.latent_dim:
            raise ValueError(
                f"Mean size {Mean.size(0)} does not match expected {self.latent_dim}"
            )
        if Range.size(0) != self.latent_dim:
            raise ValueError(
                f"Range size {Range.size(0)} does not match expected {self.latent_dim}"
            )

    def get_norm_len(self) -> int:
        """
        Return the number of features expected by the normalization buffers.

        Returns
        -------
        int
            Target / latent dimension ``self.latent_dim``.
        """
        return int(getattr(self, "latent_dim", None) or getattr(self, "target_dim", 1) or 1)

    def _normalize_tensor(self, x: torch.Tensor) -> torch.Tensor:
        """
        Normalize a tensor along its last dimension using ``self.Mean`` and ``self.Range``.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor with last dimension equal to ``self.latent_dim``.

        Returns
        -------
        torch.Tensor
            Normalized tensor.
        """
        if x.shape[-1] != self.Mean.numel():
            self.raise_error(
                f"Mean and Range buffers ({self.Mean.numel()}) must match the last "
                f"dimension of the input tensor ({x.shape})."
            )
        view_shape = (1,) * (x.ndim - 1) + (self.Mean.numel(),)
        mean_exp = self.Mean.to(device=x.device, dtype=x.dtype).view(view_shape)
        range_exp = self.Range.to(device=x.device, dtype=x.dtype).view(view_shape)
        return (x - mean_exp) / range_exp

    def _denormalize_tensor(self, x: torch.Tensor) -> torch.Tensor:
        """
        Denormalize a tensor along its last dimension using ``self.Mean`` and ``self.Range``.

        Parameters
        ----------
        x : torch.Tensor
            Normalized tensor with last dimension equal to ``self.latent_dim``.

        Returns
        -------
        torch.Tensor
            Denormalized tensor.
        """
        if x.shape[-1] != self.Mean.numel():
            self.raise_error(
                f"Mean and Range buffers ({self.Mean.numel()}) must match the last "
                f"dimension of the input tensor ({x.shape})."
            )
        view_shape = (1,) * (x.ndim - 1) + (self.Mean.numel(),)
        mean_exp = self.Mean.to(device=x.device, dtype=x.dtype).view(view_shape)
        range_exp = self.Range.to(device=x.device, dtype=x.dtype).view(view_shape)
        return x * range_exp + mean_exp

    def _normalize(self, x: Union[torch.Tensor, Dict[str, Any]]) -> Union[torch.Tensor, Dict[str, Any]]:
        """
        Normalize a tensor or a ``DartsTimeseriesDataset`` batch dictionary.

        Parameters
        ----------
        x : torch.Tensor or dict
            Input tensor or batch dictionary.

        Returns
        -------
        torch.Tensor or dict
            Normalized tensor or batch dictionary.
        """
        if isinstance(x, dict):
            out = dict(x)
            if out.get("past_target") is not None:
                out["past_target"] = self._normalize_tensor(out["past_target"])
            if out.get("future_target") is not None:
                out["future_target"] = self._normalize_tensor(out["future_target"])
            return out
        return self._normalize_tensor(x)

    def _denormalize(self, x: Union[torch.Tensor, Dict[str, Any]]) -> Union[torch.Tensor, Dict[str, Any]]:
        """
        Denormalize a tensor or a ``DartsTimeseriesDataset`` batch dictionary.

        Parameters
        ----------
        x : torch.Tensor or dict
            Normalized tensor or batch dictionary.

        Returns
        -------
        torch.Tensor or dict
            Denormalized tensor or batch dictionary.
        """
        if isinstance(x, dict):
            out = dict(x)
            if out.get("past_target") is not None:
                out["past_target"] = self._denormalize_tensor(out["past_target"])
            if out.get("future_target") is not None:
                out["future_target"] = self._denormalize_tensor(out["future_target"])
            return out
        return self._denormalize_tensor(x)

    # ------------------------------------------------------------------
    # Forward Pass & Forecasting Modes
    # ------------------------------------------------------------------

    def _validate_prediction_type(self) -> None:
        """
        Validate that ``self.prediction_type`` is a supported forecasting mode.

        Raises
        ------
        ValueError
            If ``self.prediction_type`` is not a recognized autoregressive or
            teacher-forced mode string.
        """
        pred_type = str(self.prediction_type).strip().lower()
        valid = self._AUTOREGRESSIVE_TYPES + self._TEACHER_FORCED_TYPES
        if pred_type not in valid:
            self.raise_error(
                f"Invalid prediction_type='{self.prediction_type}'. "
                f"Expected one of {valid}."
            )
        self.prediction_type = pred_type

    def _is_autoregressive(self) -> bool:
        """
        Return ``True`` if the model is configured for autoregressive rollout during evaluation.

        Returns
        -------
        bool
            ``True`` when ``self.prediction_type`` is in ``_AUTOREGRESSIVE_TYPES``.
        """
        return str(self.prediction_type).strip().lower() in self._AUTOREGRESSIVE_TYPES

    def get_long_seq_length(self) -> int:
        """
        Return the total sequence length per trajectory chunk.

        Returns
        -------
        int
            ``self.long_sequence_length`` if set, otherwise ``self.sequence_length``.
        """
        return int(self.long_sequence_length or self.sequence_length)

    def prop_sample(self, prop_out: torch.Tensor) -> torch.Tensor:
        """
        Convert raw propagator output ``prop_out`` of shape ``(B, T_out, D, n_params)``
        into point/sampled predictions of shape ``(B, T_out, D)``.

        Parameters
        ----------
        prop_out : torch.Tensor
            Raw output tensor from ``self.propagator``.

        Returns
        -------
        torch.Tensor
            Sampled or point prediction tensor of shape ``(B, T_out, D)``.
        """
        extra_args = {}
        if not self.training:
            noise_type = self.inference_settings.get("noise_type", None)
            if noise_type is not None:
                extra_args["noise_type"] = noise_type
                extra_args["noise_args"] = self.inference_settings.get("noise_args", {})

        if self.likelihood is not None:
            try:
                return self.likelihood.sample(prop_out, **extra_args)
            except TypeError:
                return self.likelihood.sample(prop_out)
        if prop_out.ndim == 4:
            if prop_out.shape[-1] == 1:
                return prop_out.squeeze(-1)
            return prop_out[..., prop_out.shape[-1] // 2]
        return prop_out

    def _call_propagator(
        self,
        past_target: torch.Tensor,
        past_cov: Optional[torch.Tensor] = None,
        hist_fut_cov: Optional[torch.Tensor] = None,
        fut_cov: Optional[torch.Tensor] = None,
        static_cov: Optional[torch.Tensor] = None,
        future_target: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Run a single chunk forward pass through ``self.propagator``.

        Ensures cached Darts buffers (such as ``attention_mask`` and ``relative_index``
        in ``_TFTModule``) are regenerated if the device or dtype has changed.

        Parameters
        ----------
        past_target : torch.Tensor
            Past target sequence of shape ``(B, input_chunk_length, D)``.
        past_cov : torch.Tensor, optional
            Past covariates of shape ``(B, input_chunk_length, C_past)``.
        hist_fut_cov : torch.Tensor, optional
            Historic future covariates of shape ``(B, input_chunk_length, C_fut)``.
        fut_cov : torch.Tensor, optional
            Future covariates of shape ``(B, output_chunk_length, C_fut)``.
        static_cov : torch.Tensor, optional
            Static covariates of shape ``(B, 1, C_static)`` or ``(B, C_static)``.
        future_target : torch.Tensor, optional
            Future target sequence of shape ``(B, output_chunk_length, D)``.

        Returns
        -------
        torch.Tensor
            Raw output tensor of shape ``(B, output_chunk_length, D, n_params)``.
        """
        param = next(self.propagator.parameters(), None)
        target_device = param.device if param is not None else past_target.device
        target_dtype = param.dtype if param is not None else past_target.dtype

        def _to(t: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
            return t.to(device=target_device, dtype=target_dtype) if t is not None else None

        past_target = _to(past_target)
        past_cov = _to(past_cov)
        hist_fut_cov = _to(hist_fut_cov)
        fut_cov = _to(fut_cov)
        static_cov = _to(static_cov)
        future_target = _to(future_target)

        if static_cov is not None and static_cov.ndim == 2:
            static_cov = static_cov.unsqueeze(1)

        # Reset Darts _TFTModule cached mask/relative_index if device or dtype changed
        cached_mask = getattr(self.propagator, "attention_mask", None)
        cached_rel = getattr(self.propagator, "relative_index", None)
        if (
            (cached_mask is not None and cached_mask.device != target_device)
            or (
                cached_rel is not None
                and (cached_rel.device != target_device or cached_rel.dtype != target_dtype)
            )
        ):
            self.propagator.batch_size_last = -1

        return self.propagator._produce_train_output((
            past_target,
            past_cov,
            hist_fut_cov,
            fut_cov,
            static_cov,
            future_target if self.training else None,
        ))

    def _parse_forward_inputs(
        self, data: Union[Dict[str, torch.Tensor], torch.Tensor]
    ) -> Dict[str, Optional[torch.Tensor]]:
        """
        Extract ``past_target``, ``future_target``, and optional covariates from either
        a ``DartsTimeseriesDataset`` dictionary or a 2-D/3-D tensor.

        Parameters
        ----------
        data : dict or torch.Tensor
            Batch input data.

        Returns
        -------
        dict of str to (torch.Tensor or None)
            Dictionary with keys ``past_target``, ``future_target``, ``past_cov``,
            ``hist_fut_cov``, ``fut_cov``, ``static_cov``, and ``full_seq``.
        """
        t_in = self.input_chunk_length
        t_out = self.output_chunk_length

        if isinstance(data, dict):
            norm_data = self.normalize(data) if self.normIn else data
            past_target = norm_data["past_target"]
            future_target = norm_data.get("future_target")
            past_cov = norm_data.get("past_feat_dynamic_real")
            feat_dyn = norm_data.get("feat_dynamic_real")
            hist_fut_cov = feat_dyn[:, :t_in, :] if feat_dyn is not None else None
            fut_cov = (
                feat_dyn[:, t_in : t_in + t_out, :]
                if (feat_dyn is not None and feat_dyn.shape[1] >= t_in + t_out)
                else None
            )
            static_cov = norm_data.get("static_covariates")
            full_seq = (
                torch.cat([past_target, future_target], dim=1)
                if future_target is not None
                else past_target
            )
            return {
                "past_target": past_target,
                "future_target": future_target,
                "past_cov": past_cov,
                "hist_fut_cov": hist_fut_cov,
                "fut_cov": fut_cov,
                "static_cov": static_cov,
                "full_seq": full_seq,
            }

        if isinstance(data, torch.Tensor):
            norm_tensor = self.normalize(data) if self.normIn else data
            long_seq_len = self.get_long_seq_length()
            seq = (
                norm_tensor.view(-1, long_seq_len, self.latent_dim)
                if norm_tensor.ndim == 2
                else norm_tensor
            )
            past_target = seq[:, :t_in, :]
            future_target = seq[:, t_in:, :] if seq.shape[1] > t_in else None
            return {
                "past_target": past_target,
                "future_target": future_target,
                "past_cov": None,
                "hist_fut_cov": None,
                "fut_cov": None,
                "static_cov": None,
                "full_seq": seq,
            }

        self.raise_error(
            f"Unsupported input type for DartsForecastingModel.forward: {type(data)}."
        )

    def _forward_autoregressive(
        self, parsed: Dict[str, Optional[torch.Tensor]]
    ) -> torch.Tensor:
        """
        Execute an autoregressive rollout across the batch of sequence windows.

        Starts from ``self._rollout_context`` (if continuing across test batches) or
        ``parsed['past_target'][0:1]``, and iteratively predicts ``output_chunk_length``
        frames at a time, updating the context window with its own sampled predictions.

        Parameters
        ----------
        parsed : dict of str to (torch.Tensor or None)
            Parsed batch dictionary from :meth:`_parse_forward_inputs`.

        Returns
        -------
        torch.Tensor
            Stacked raw propagator outputs of shape
            ``(batch_size, output_chunk_length, latent_dim, n_params)``.
        """
        past_target = parsed["past_target"]
        batch_size = past_target.shape[0]
        cur_context = (
            self._rollout_context.to(
                device=past_target.device, dtype=past_target.dtype
            )
            if self._rollout_context is not None
            else past_target[0:1]
        )

        past_cov = parsed["past_cov"]
        hist_fut_cov = parsed["hist_fut_cov"]
        fut_cov = parsed["fut_cov"]
        static_cov = parsed["static_cov"]

        step_outputs: List[torch.Tensor] = []
        for b in range(batch_size):
            step_out = self._call_propagator(
                past_target=cur_context,
                past_cov=past_cov[b : b + 1] if past_cov is not None else None,
                hist_fut_cov=(
                    hist_fut_cov[b : b + 1] if hist_fut_cov is not None else None
                ),
                fut_cov=fut_cov[b : b + 1] if fut_cov is not None else None,
                static_cov=static_cov[b : b + 1] if static_cov is not None else None,
                future_target=None,
            )
            step_outputs.append(step_out)

            step_sample = self.prop_sample(step_out.clone()).to(
                device=cur_context.device, dtype=cur_context.dtype
            )
            cur_context = torch.cat([cur_context, step_sample], dim=1)[
                :, -self.input_chunk_length :, :
            ]

        self._rollout_context = cur_context.detach()
        return torch.cat(step_outputs, dim=0)

    def forward(
        self, data: Union[Dict[str, torch.Tensor], torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """
        Execute the propagator forward pass and return ``(prop_sample, past_target, meta)``.

        Parameters
        ----------
        data : dict or torch.Tensor
            Batch of input sequences from :class:`DartsTimeseriesDataset` or tensor.

        Returns
        -------
        tuple of (torch.Tensor, torch.Tensor, dict)
            - ``prop_sample``: Predicted future trajectory of shape ``(B, T_pred, D)``.
            - ``past_target``: Context sequence tensor of shape ``(B, T_in, D)``.
            - ``meta``: Metadata dictionary containing ``prop_out`` and ``future_target``.
        """
        parsed = self._parse_forward_inputs(data)
        past_target = parsed["past_target"]
        future_target = parsed["future_target"]
        full_seq = parsed["full_seq"]

        if (
            not self.training
            and not getattr(self, "_in_validation", False)
            and self._is_autoregressive()
        ):
            prop_out = self._forward_autoregressive(parsed)
        elif full_seq.shape[1] > self.sequence_length:
            # Multi-chunk teacher-forced stepping across a long sequence (as in PropagatorTFT)
            i = 0
            chunks: List[torch.Tensor] = []
            while i + self.sequence_length <= full_seq.shape[1]:
                chunk_in = full_seq[:, i : i + self.input_chunk_length, :]
                chunk_fut = full_seq[
                    :,
                    i + self.input_chunk_length : i + self.sequence_length,
                    :,
                ]
                chunks.append(
                    self._call_propagator(
                        past_target=chunk_in,
                        future_target=chunk_fut,
                    )
                )
                i += self.output_chunk_length
            prop_out = torch.cat(chunks, dim=1)
        else:
            prop_out = self._call_propagator(
                past_target=past_target,
                past_cov=parsed["past_cov"],
                hist_fut_cov=parsed["hist_fut_cov"],
                fut_cov=parsed["fut_cov"],
                static_cov=parsed["static_cov"],
                future_target=future_target,
            )

        if hasattr(self.propagator, "reset_cell_state"):
            self.propagator.reset_cell_state()

        prop_sample = self.prop_sample(prop_out.clone())
        if self.normIn:
            prop_sample = self._denormalize_tensor(prop_sample)

        meta: Dict[str, Any] = {
            "prop_out": prop_out,
            "future_target": future_target,
        }
        return prop_sample, past_target, meta

    # ------------------------------------------------------------------
    # Losses & Metrics
    # ------------------------------------------------------------------

    def _resolve_future_target(
        self,
        output: torch.Tensor,
        labels: Optional[torch.Tensor],
        meta: Dict[str, Any],
    ) -> torch.Tensor:
        """
        Extract the ground-truth future target tensor matching ``output``'s device and dtype.

        Parameters
        ----------
        output : torch.Tensor
            Predicted tensor.
        labels : torch.Tensor, optional
            Fallback label tensor.
        meta : dict
            Forward metadata dictionary containing ``future_target``.

        Returns
        -------
        torch.Tensor
            Ground-truth future target tensor of shape ``(B, T_pred, D)``.
        """
        future_target = meta.get("future_target")
        if future_target is not None:
            return future_target.to(device=output.device, dtype=output.dtype)
        if labels is not None and labels.numel() == output.numel():
            return labels.to(device=output.device, dtype=output.dtype).view_as(output)
        self.raise_error(
            "DartsForecastingModel requires 'future_target' in the input batch or "
            "matching target labels to compute loss and metrics."
        )

    def loss_prop(
        self,
        inp: Any,
        latent: torch.Tensor,
        output: torch.Tensor,
        labels: torch.Tensor,
        meta: Dict[str, Any],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute the forecasting training/validation loss on future latent predictions.

        Parameters
        ----------
        inp : Any
            Input batch data.
        latent : torch.Tensor
            Past target context tensor.
        output : torch.Tensor
            Sampled point prediction tensor of shape ``(B, T_pred, D)``.
        labels : torch.Tensor
            Batch labels.
        meta : dict
            Metadata dictionary containing ``prop_out`` and ``future_target``.

        Returns
        -------
        tuple of (torch.Tensor, dict)
            Scalar loss tensor and empty auxiliary dictionary.
        """
        prop_out = meta["prop_out"]
        future_target = self._resolve_future_target(output, labels, meta)

        if self.likelihood is not None:
            try:
                loss_res = self.likelihood.compute_loss(prop_out, future_target, None)
            except TypeError:
                loss_res = self.likelihood.compute_loss(prop_out, future_target)
            if isinstance(loss_res, tuple):
                loss, loss_meta = loss_res
                return loss, loss_meta
            return loss_res, {}
        else:
            pred = (
                prop_out.squeeze(-1)
                if (prop_out.ndim == 4 and prop_out.shape[-1] == 1)
                else prop_out
            )
            loss = self.loss_fn(pred, future_target)
        return loss, {}

    def metric_mae_prop(
        self,
        inp: Any,
        latent: torch.Tensor,
        output: torch.Tensor,
        labels: torch.Tensor,
        meta: Dict[str, Any],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute the Mean Absolute Error (MAE) of the sampled future forecast.

        Parameters
        ----------
        inp : Any
            Input batch data.
        latent : torch.Tensor
            Past target context tensor.
        output : torch.Tensor
            Sampled point prediction tensor of shape ``(B, T_pred, D)``.
        labels : torch.Tensor
            Batch labels.
        meta : dict
            Metadata dictionary containing ``future_target``.

        Returns
        -------
        tuple of (torch.Tensor, dict)
            Scalar MAE tensor and empty auxiliary dictionary.
        """
        future_target = self._resolve_future_target(output, labels, meta)
        if self.normIn:
            future_target = self._denormalize_tensor(future_target)
        mae = F.l1_loss(output, future_target, reduction="mean")
        return mae, {}

    # ------------------------------------------------------------------
    # Test Plotter Hooks
    # ------------------------------------------------------------------

    def _resolve_datamodule(self, datamodule: Any = None) -> Any:
        """
        Resolve the active :class:`~collective_encoder.datamodules.base.BaseDataModule`.

        Parameters
        ----------
        datamodule : Any, optional
            Explicit datamodule instance if passed directly.

        Returns
        -------
        Any or None
            Resolved datamodule or ``None``.
        """
        if datamodule is not None:
            return datamodule
        if (
            hasattr(self, "trainer")
            and self.trainer is not None
            and getattr(self.trainer, "datamodule", None) is not None
        ):
            return self.trainer.datamodule
        if getattr(self, "datamodule", None) is not None:
            return self.datamodule
        return None

    def on_validation_epoch_start(self) -> None:
        """
        Mark validation stage active and reset autoregressive rollout context.
        """
        self._in_validation = True
        self._rollout_context = None

    def on_validation_epoch_end(self) -> None:
        """
        Mark validation stage complete and reset autoregressive rollout context.
        """
        self._in_validation = False
        self._rollout_context = None

    def _plot_test_start(self) -> None:
        """
        Initialize test plotters prior to evaluation and reset rollout state.

        For any test plotter whose class lists ``"embedding_model"`` in
        ``_REQUIRED_ARGS`` and does not have ``"embedding_model"`` specified in its
        configuration dictionary, attempts to extract ``embedding_model`` from the
        active datamodule's dataset before delegating to
        :meth:`~collective_encoder.models.base.CEModelBase._plot_test_start`.
        """
        self._rollout_context = None
        dm = self._resolve_datamodule()
        ds = (
            dm.get_any_dataset()
            if (dm is not None and hasattr(dm, "get_any_dataset"))
            else None
        )
        dm_embedding_model = (
            getattr(ds, "embedding_model", None) or getattr(ds, "embedding_layer", None)
            if ds is not None
            else None
        )

        updated_plotters = []
        for name, args in self.test_plotters:
            plotter_args = (args or {}).copy()
            try:
                plotter_cls = get_module("testplotters", name)
                required_args = getattr(plotter_cls, "_REQUIRED_ARGS", [])
                optional_args = getattr(plotter_cls, "_OPTIONAL_ARGS", {})
                if (
                    "embedding_model" in required_args
                    and plotter_args.get("embedding_model") is None
                    and dm_embedding_model is not None
                ):
                    plotter_args["embedding_model"] = dm_embedding_model
                if (
                    ("datamodule" in required_args or "datamodule" in optional_args)
                    and plotter_args.get("datamodule") is None
                    and dm is not None
                ):
                    plotter_args["datamodule"] = dm
            except Exception:
                pass
            updated_plotters.append((name, plotter_args))

        self.test_plotters = updated_plotters
        super()._plot_test_start()

    def _plot_test_batch(
        self,
        inp: Any,
        latent: torch.Tensor,
        output: torch.Tensor,
        labels: Any,
        meta: Dict[str, Any],
        datamodule: Any = None,
    ) -> None:
        """
        Format sequence predictions, latents, and future horizon labels into 2-D
        frame-level tensors before forwarding to test plotters.

        Parameters
        ----------
        inp : Any
            Batch of input data.
        latent : torch.Tensor
            Context sequence tensor of shape ``(batch_size, input_chunk_length, latent_dim)``.
        output : torch.Tensor
            Prediction tensor of shape ``(batch_size, T_pred, latent_dim)``.
        labels : torch.Tensor or dict
            Per-sequence ground-truth labels.
        meta : dict of str to Any
            Batch metadata dictionary from :meth:`forward`.
        datamodule : Any, optional
            Explicit datamodule instance.
        """
        dm = self._resolve_datamodule(datamodule)
        label_names = dm.get_label_names() if dm is not None else []
        n_labels = len(label_names)
        pred_steps = output.shape[1] if (isinstance(output, torch.Tensor) and output.ndim == 3) else self.output_chunk_length
        seq_len = int(self.input_chunk_length + pred_steps)

        if isinstance(labels, torch.Tensor) and n_labels > 0:
            batch_size = labels.shape[0]
            if (
                labels.ndim == 3
                and labels.shape[1] >= self.input_chunk_length + pred_steps
            ):
                future_labels = labels[
                    :,
                    self.input_chunk_length : self.input_chunk_length + pred_steps,
                    :,
                ]
                labels = future_labels.reshape(-1, n_labels)
            elif labels.ndim == 2 and labels.shape[1] == seq_len * n_labels:
                labels_3d = labels.reshape(batch_size, seq_len, n_labels)
                future_labels = labels_3d[
                    :,
                    self.input_chunk_length : self.input_chunk_length + pred_steps,
                    :,
                ]
                labels = future_labels.reshape(-1, n_labels)
            elif labels.ndim == 2 and labels.shape[1] == pred_steps * n_labels:
                labels = labels.reshape(-1, n_labels)

        if isinstance(output, torch.Tensor) and output.ndim == 3:
            output = output.reshape(-1, output.shape[-1])
        if isinstance(latent, torch.Tensor) and latent.ndim == 3:
            latent = latent.reshape(-1, latent.shape[-1])

        super()._plot_test_batch(inp, latent, output, labels, meta, datamodule=dm)

    def _plot_test_finish(self) -> None:
        """
        Finalize test plotters and reset autoregressive rollout state.
        """
        self._rollout_context = None
        super()._plot_test_finish()

    # ------------------------------------------------------------------
    # Standalone Prediction Rollout
    # ------------------------------------------------------------------

    def set_inference_settings(self, **kwargs) -> None:
        """
        Update inference settings for standalone ``predict_step`` propagation.

        Parameters
        ----------
        **kwargs
            Inference configuration keys (``predict_steps``, ``noise_type``,
            ``noise_args``, ``plot_decoded``).
        """
        allowed_keys = {"predict_steps", "noise_type", "noise_args", "plot_decoded"}
        unknown_keys = set(kwargs.keys()) - allowed_keys
        if unknown_keys:
            self.raise_error(
                f"Unknown keys in inference settings: {unknown_keys}. "
                f"Allowed keys are: {allowed_keys}"
            )
        if (
            "predict_steps" in kwargs
            and kwargs["predict_steps"] < self.sequence_length
        ):
            self.raise_error(
                f"predict_steps must not be less than sequence_length "
                f"({self.sequence_length}). Got predict_steps={kwargs['predict_steps']}."
            )
        if kwargs.get("noise_type") is not None and self.likelihood is None:
            self.raise_error(
                "Cannot specify noise_type in inference_settings when no likelihood model is used."
            )
        self.inference_settings.update(kwargs)

    @torch.inference_mode()
    def predict_step(
        self, batch: Any, batch_idx: int, dataloader_idx: int = 0
    ) -> Dict[str, torch.Tensor]:
        """
        Perform autoregressive multi-step prediction for a batch.

        Parameters
        ----------
        batch : tuple
            Batch tuple ``(data, labels)`` from the DataLoader.
        batch_idx : int
            Index of the current batch.
        dataloader_idx : int, optional
            Index of the current DataLoader.

        Returns
        -------
        dict of str to torch.Tensor
            Dictionary containing ``point_prediction`` and ``past_target`` (or full
            ``Latent`` rollout trajectory when ``predict_steps`` is configured in
            ``inference_settings``).
        """
        data, _ = self._batch_split(batch)
        if "predict_steps" not in self.inference_settings:
            point_pred, past_target, meta = self(data)
            result = {
                "point_prediction": point_pred,
                "past_target": past_target,
            }
            if meta.get("future_target") is not None:
                result["future_target"] = meta["future_target"]
            return result

        parsed = self._parse_forward_inputs(data)
        predict_steps = int(self.inference_settings["predict_steps"])
        inp = parsed["past_target"][0:1, : self.input_chunk_length, :]
        latent_prediction = inp.squeeze(0)

        n_propagations = (
            (predict_steps - self.input_chunk_length) // self.output_chunk_length + 1
        )
        pbar = tqdm(
            total=predict_steps,
            leave=False,
            desc="Autoregressive Propagation",
            ncols=80,
        )
        for _ in range(n_propagations):
            raw_step = self._call_propagator(past_target=inp)
            step_pred = self.prop_sample(raw_step)
            inp = torch.cat([inp, step_pred], dim=1)[
                :, -self.input_chunk_length :, :
            ]
            step_2d = step_pred.squeeze(0)
            latent_prediction = torch.cat([latent_prediction, step_2d], dim=0)
            pbar.update(step_2d.size(0))
        pbar.close()

        latent_prediction = latent_prediction[:predict_steps, :]
        if self.normIn:
            latent_prediction = self._denormalize_tensor(latent_prediction)
        return {"Latent": latent_prediction}
