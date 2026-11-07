import math
from contextlib import contextmanager
from copy import deepcopy
from typing import Any, Dict, Generator, List, Optional, Tuple, Union

import numpy
import numpy as np
import torch
from einops import rearrange, reduce, repeat
from jaxtyping import Bool, Float, Int

from collective_encoder.common.registry import get_module
from collective_encoder.models.neural_nets.base import CENetBase
from surrogate_dynamics.propagators.moirai2.modules.imputation import CausalMeanImputation
from surrogate_dynamics.propagators.moirai2.modules.moirai2 import Moirai2Module


class Moirai2(CENetBase):
    """
    Moirai2 time-series foundation model propagator integrated with
    :class:`~collective_encoder.models.neural_nets.base.CENetBase`.

    Compatible with :class:`~surrogate_dynamics.datasets.darts_timeseries.DartsTimeseriesDataset`
    and :class:`~surrogate_dynamics.datamodules.timeseries.base.TimeseriesDataModule`.
    """

    _IDENTIFIER = "MOIRAI2"
    _IDENTFIER = "MOIRAI2"
    _REQUIRED_ARGS = [
        "prediction_length",
        "target_dim",
        "context_length",
    ]
    _OPTIONAL_ARGS = CENetBase._OPTIONAL_ARGS.copy()
    _OPTIONAL_ARGS.update({
        "feat_dynamic_real_dim": 0,
        "past_feat_dynamic_real_dim": 0,
        "hidden_dim": None,
        "ff_dim": None,
        "num_layers": None,
        "patch_size": None,
        "max_seq_length": None,
        "pretrain_weights": None,
        "dropout_p": 0.0,
        "dropout_attn_p": 0.0,
        "scaling": True,
        "num_predict_token": 1,
        "quantile_levels": (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
        "prediction_type": "autoregressive",
    })

    _AUTOREGRESSIVE_TYPES = ("autoregressive", "autoregressive_rollout", "rollout")
    _TEACHER_FORCED_TYPES = ("teacher_forced", "teacher-forced", "teacher")

    @staticmethod
    def extract_args_from_datamodule(datamodule, args: dict) -> dict:
        """
        Infer ``target_dim``, ``context_length``, ``prediction_length``,
        ``feat_dynamic_real_dim``, and ``past_feat_dynamic_real_dim`` from
        ``datamodule`` when not already set in ``args``.
        """
        out_args = (args or {}).copy()
        if out_args.get("context_length") is None and getattr(
            datamodule, "input_chunk_length", None
        ) is not None:
            out_args["context_length"] = int(datamodule.input_chunk_length)
        if out_args.get("prediction_length") is None and getattr(
            datamodule, "output_chunk_length", None
        ) is not None:
            out_args["prediction_length"] = int(datamodule.output_chunk_length)

        any_ds = (
            datamodule.get_any_dataset()
            if hasattr(datamodule, "get_any_dataset")
            else None
        )
        if any_ds is not None and hasattr(any_ds, "get_feature_dims"):
            dims = any_ds.get_feature_dims()
            if out_args.get("target_dim") is None:
                out_args["target_dim"] = dims["target_dim"]
            if out_args.get("feat_dynamic_real_dim") is None:
                out_args["feat_dynamic_real_dim"] = dims["feat_dynamic_real_dim"]
            if out_args.get("past_feat_dynamic_real_dim") is None:
                out_args["past_feat_dynamic_real_dim"] = dims[
                    "past_feat_dynamic_real_dim"
                ]
        elif out_args.get("target_dim") is None and hasattr(
            datamodule, "get_datapoint_shape"
        ):
            dp_shape = datamodule.get_datapoint_shape()
            if isinstance(dp_shape, (tuple, list)) and len(dp_shape) >= 2:
                out_args["target_dim"] = int(dp_shape[-1])

        return out_args

    def __init__(
        self,
        args: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        self.save_hyperparameters()
        super().__init__(args, **kwargs)

        if self.normIn:
            self.norm_error()

        self._validate_prediction_type()
        self._rollout_context: Optional[torch.Tensor] = None

        pretrained_ckpt = self.pretrain_weights
        if pretrained_ckpt is None:
            if not all([
                self.hidden_dim,
                self.ff_dim,
                self.num_layers,
                self.patch_size,
                self.max_seq_length,
            ]):
                raise ValueError(
                    "If pretrain_weights is not provided, hidden_dim, ff_dim, "
                    "num_layers, patch_size, and max_seq_length must be specified."
                )

        self._init_propagator()

        self.losses = {
            "loss_prop": self.loss_prop,
        }
        self.metrics = {
            "mae_prop": self.metric_mae_prop,
        }
        self.test_metrics = self.metrics.copy()

    def _get_train_params(self):
        params_from_losses = []
        for loss in self.losses.values():
            if hasattr(loss, "parameters"):
                params_from_losses.extend(loss.parameters())
        return list(self.parameters()) + params_from_losses

    def _init_propagator(self):
        pretrained_ckpt = self.pretrain_weights
        if pretrained_ckpt is not None:
            try:
                self.propagator = Moirai2Module.from_pretrained(pretrained_ckpt)
            except Exception as e:
                self.raise_error(
                    f"Error occurred while loading pre-trained weights from '{pretrained_ckpt}': {e}"
                )
        else:
            prop_kwargs = {
                "d_model": int(self.hidden_dim),
                "d_ff": int(self.ff_dim),
                "num_layers": int(self.num_layers),
                "patch_size": int(self.patch_size),
                "max_seq_len": int(self.max_seq_length),
                "attn_dropout_p": float(self.dropout_attn_p),
                "dropout_p": float(self.dropout_p),
                "scaling": bool(self.scaling),
                "num_predict_token": int(self.num_predict_token),
                "quantile_levels": tuple(self.quantile_levels),
            }
            self.propagator = Moirai2Module(**prop_kwargs)

    # ------------------------------------------------------------------
    # Normalization
    # ------------------------------------------------------------------

    def norm_error(self):
        self.raise_error(
            "Moirai2 does not support external normalization (normIn=True). "
            "Use 'scaling: True' in the model configuration instead."
        )

    def get_norm_len(self) -> int:
        return int(getattr(self, "target_dim", 1) or 1)

    def set_norm(self, datamodule=None) -> None:
        self.norm_error()

    def _normalize(self, x):
        self.norm_error()

    def _denormalize(self, x):
        self.norm_error()

    # ------------------------------------------------------------------
    # Token / Patch Helpers & Input Conversion
    # ------------------------------------------------------------------

    @contextmanager
    def hparams_context(
        self,
        prediction_length: Optional[int] = None,
        target_dim: Optional[int] = None,
        feat_dynamic_real_dim: Optional[int] = None,
        past_feat_dynamic_real_dim: Optional[int] = None,
        context_length: Optional[int] = None,
    ) -> Generator["Moirai2", None, None]:
        overrides = {
            "prediction_length": prediction_length,
            "target_dim": target_dim,
            "feat_dynamic_real_dim": feat_dynamic_real_dim,
            "past_feat_dynamic_real_dim": past_feat_dynamic_real_dim,
            "context_length": context_length,
        }
        old_vals = {k: getattr(self, k) for k in overrides}
        try:
            for k, v in overrides.items():
                if v is not None:
                    setattr(self, k, v)
            yield self
        finally:
            for k, v in old_vals.items():
                setattr(self, k, v)

    def context_token_length(self, patch_size: int) -> int:
        return math.ceil(self.context_length / patch_size)

    def prediction_token_length(self, patch_size: int) -> int:
        return math.ceil(self.prediction_length / patch_size)

    @property
    def past_length(self) -> int:
        return int(self.context_length)

    @property
    def max_patch_size(self) -> int:
        if hasattr(self.propagator, "patch_sizes"):
            return max(self.propagator.patch_sizes)
        return int(self.propagator.patch_size)

    def _generate_time_id(
        self,
        patch_size: int,
        past_observed_target: Bool[torch.Tensor, "batch past_seq tgt"],
    ) -> tuple[
        Int[torch.Tensor, "batch past_token"], Int[torch.Tensor, "batch future_token"]
    ]:
        past_seq_id = reduce(
            self._patched_seq_pad(patch_size, past_observed_target, -2, left=True),
            "... (seq patch) dim -> ... seq",
            "max",
            patch=patch_size,
        )
        past_seq_id = torch.clamp(
            past_seq_id.cummax(dim=-1).values.cumsum(dim=-1) - 1, min=0
        )
        batch_shape = " ".join(map(str, past_observed_target.shape[:-2]))
        future_seq_id = (
            repeat(
                torch.arange(
                    self.prediction_token_length(patch_size),
                    device=past_observed_target.device,
                ),
                f"prediction -> {batch_shape} prediction",
            )
            + past_seq_id.max(dim=-1, keepdim=True).values
            + 1
        )
        return past_seq_id, future_seq_id

    def _format_preds(
        self,
        num_quantiles: int,
        patch_size: int,
        preds: Float[torch.Tensor, "batch combine_seq num_quantiles patch"],
        target_dim: int,
    ) -> Float[torch.Tensor, "batch num_quantiles future_time tgt"]:
        start = target_dim * self.context_token_length(patch_size)
        end = start + target_dim * self.prediction_token_length(patch_size)
        preds = preds[..., start:end, :num_quantiles, :patch_size]
        preds = rearrange(
            preds,
            "... (dim seq) num_quantiles patch -> ... num_quantiles (seq patch) dim",
            dim=target_dim,
        )[..., : self.prediction_length, :]
        return preds

    @staticmethod
    def _patched_seq_pad(
        patch_size: int,
        x: torch.Tensor,
        dim: int,
        left: bool = True,
        value: Optional[float] = None,
    ) -> torch.Tensor:
        if dim >= 0:
            dim = -x.ndim + dim
        pad_length = -x.size(dim) % patch_size
        if left:
            pad = (pad_length, 0)
        else:
            pad = (0, pad_length)
        pad = (0, 0) * (abs(dim) - 1) + pad
        return torch.nn.functional.pad(x, pad, value=value)

    def _convert(
        self,
        patch_size: int,
        past_target: Float[torch.Tensor, "batch past_time tgt"],
        past_observed_target: Bool[torch.Tensor, "batch past_time tgt"],
        past_is_pad: Bool[torch.Tensor, "batch past_time"],
        future_target: Optional[Float[torch.Tensor, "batch future_time tgt"]] = None,
        future_observed_target: Optional[
            Bool[torch.Tensor, "batch future_time tgt"]
        ] = None,
        future_is_pad: Optional[Bool[torch.Tensor, "batch future_time"]] = None,
        feat_dynamic_real: Optional[Float[torch.Tensor, "batch time feat"]] = None,
        observed_feat_dynamic_real: Optional[
            Float[torch.Tensor, "batch time feat"]
        ] = None,
        past_feat_dynamic_real: Optional[
            Float[torch.Tensor, "batch past_time past_feat"]
        ] = None,
        past_observed_feat_dynamic_real: Optional[
            Float[torch.Tensor, "batch past_time past_feat"]
        ] = None,
    ) -> tuple[
        Float[torch.Tensor, "batch combine_seq patch"],  # target
        Bool[torch.Tensor, "batch combine_seq patch"],  # observed_mask
        Int[torch.Tensor, "batch combine_seq"],  # sample_id
        Int[torch.Tensor, "batch combine_seq"],  # time_id
        Int[torch.Tensor, "batch combine_seq"],  # variate_id
        Bool[torch.Tensor, "batch combine_seq"],  # prediction_mask
    ]:
        batch_shape = past_target.shape[:-2]
        device = past_target.device

        target = []
        observed_mask = []
        sample_id = []
        time_id = []
        variate_id = []
        prediction_mask = []
        dim_count = 0

        past_seq_id, future_seq_id = self._generate_time_id(
            patch_size, past_observed_target
        )

        if future_target is None:
            future_target = torch.zeros(
                batch_shape
                + (
                    self.prediction_length,
                    past_target.shape[-1],
                ),
                dtype=past_target.dtype,
                device=device,
            )
        target.extend(
            [
                torch.nn.functional.pad(
                    rearrange(
                        self._patched_seq_pad(patch_size, past_target, -2, left=True),
                        "... (seq patch) dim -> ... (dim seq) patch",
                        patch=patch_size,
                    ),
                    (0, 0),
                ),
                torch.nn.functional.pad(
                    rearrange(
                        self._patched_seq_pad(
                            patch_size, future_target, -2, left=False
                        ),
                        "... (seq patch) dim -> ... (dim seq) patch",
                        patch=patch_size,
                    ),
                    (0, 0),
                ),
            ]
        )
        if future_observed_target is None:
            future_observed_target = torch.ones(
                batch_shape
                + (
                    self.prediction_length,
                    past_observed_target.shape[-1],
                ),
                dtype=torch.bool,
                device=device,
            )
        observed_mask.extend(
            [
                torch.nn.functional.pad(
                    rearrange(
                        self._patched_seq_pad(
                            patch_size, past_observed_target, -2, left=True
                        ),
                        "... (seq patch) dim -> ... (dim seq) patch",
                        patch=patch_size,
                    ),
                    (0, 0),
                ),
                torch.nn.functional.pad(
                    rearrange(
                        self._patched_seq_pad(
                            patch_size, future_observed_target, -2, left=False
                        ),
                        "... (seq patch) dim -> ... (dim seq) patch",
                        patch=patch_size,
                    ),
                    (0, 0),
                ),
            ]
        )
        if future_is_pad is None:
            future_is_pad = torch.zeros(
                batch_shape + (self.prediction_length,),
                dtype=torch.long,
                device=device,
            )
        sample_id.extend(
            [
                repeat(
                    reduce(
                        (
                            self._patched_seq_pad(
                                patch_size, past_is_pad, -1, left=True, value=1
                            )
                            == 0
                        ).int(),
                        "... (seq patch) -> ... seq",
                        "max",
                        patch=patch_size,
                    ),
                    "... seq -> ... (dim seq)",
                    dim=past_target.shape[-1],
                ),
                repeat(
                    reduce(
                        (
                            self._patched_seq_pad(
                                patch_size, future_is_pad, -1, left=False, value=1
                            )
                            == 0
                        ).int(),
                        "... (seq patch) -> ... seq",
                        "max",
                        patch=patch_size,
                    ),
                    "... seq -> ... (dim seq)",
                    dim=past_target.shape[-1],
                ),
            ]
        )
        time_id.extend(
            [past_seq_id] * past_target.shape[-1]
            + [future_seq_id] * past_target.shape[-1]
        )
        variate_id.extend(
            [
                repeat(
                    torch.arange(past_target.shape[-1], device=device) + dim_count,
                    f"dim -> {' '.join(map(str, batch_shape))} (dim past)",
                    past=self.context_token_length(patch_size),
                ),
                repeat(
                    torch.arange(past_target.shape[-1], device=device) + dim_count,
                    f"dim -> {' '.join(map(str, batch_shape))} (dim future)",
                    future=self.prediction_token_length(patch_size),
                ),
            ]
        )
        dim_count += past_target.shape[-1]
        prediction_mask.extend(
            [
                torch.zeros(
                    batch_shape
                    + (self.context_token_length(patch_size) * past_target.shape[-1],),
                    dtype=torch.bool,
                    device=device,
                ),
                torch.ones(
                    batch_shape
                    + (
                        self.prediction_token_length(patch_size)
                        * past_target.shape[-1],
                    ),
                    dtype=torch.bool,
                    device=device,
                ),
            ]
        )

        if feat_dynamic_real is not None:
            if observed_feat_dynamic_real is None:
                raise ValueError(
                    "observed_feat_dynamic_real must be provided if feat_dynamic_real is provided"
                )

            target.extend(
                [
                    torch.nn.functional.pad(
                        rearrange(
                            self._patched_seq_pad(
                                patch_size,
                                feat_dynamic_real[
                                    ..., : self.context_length, :
                                ],
                                -2,
                                left=True,
                            ),
                            "... (seq patch) dim -> ... (dim seq) patch",
                            patch=patch_size,
                        ),
                        (0, 0),
                    ),
                    torch.nn.functional.pad(
                        rearrange(
                            self._patched_seq_pad(
                                patch_size,
                                feat_dynamic_real[
                                    ..., self.context_length :, :
                                ],
                                -2,
                                left=False,
                            ),
                            "... (seq patch) dim -> ... (dim seq) patch",
                            patch=patch_size,
                        ),
                        (0, 0),
                    ),
                ]
            )
            observed_mask.extend(
                [
                    torch.nn.functional.pad(
                        rearrange(
                            self._patched_seq_pad(
                                patch_size,
                                observed_feat_dynamic_real[
                                    ..., : self.context_length, :
                                ],
                                -2,
                                left=True,
                            ),
                            "... (seq patch) dim -> ... (dim seq) patch",
                            patch=patch_size,
                        ),
                        (0, 0),
                    ),
                    torch.nn.functional.pad(
                        rearrange(
                            self._patched_seq_pad(
                                patch_size,
                                observed_feat_dynamic_real[
                                    ..., self.context_length :, :
                                ],
                                -2,
                                left=False,
                            ),
                            "... (seq patch) dim -> ... (dim seq) patch",
                            patch=patch_size,
                        ),
                        (0, 0),
                    ),
                ]
            )
            sample_id.extend(
                [
                    repeat(
                        reduce(
                            (
                                self._patched_seq_pad(
                                    patch_size, past_is_pad, -1, left=True
                                )
                                == 0
                            ).int(),
                            "... (seq patch) -> ... seq",
                            "max",
                            patch=patch_size,
                        ),
                        "... seq -> ... (dim seq)",
                        dim=feat_dynamic_real.shape[-1],
                    ),
                    torch.ones(
                        batch_shape
                        + (
                            self.prediction_token_length(patch_size)
                            * feat_dynamic_real.shape[-1],
                        ),
                        dtype=torch.long,
                        device=device,
                    ),
                ]
            )
            time_id.extend(
                [past_seq_id] * feat_dynamic_real.shape[-1]
                + [future_seq_id] * feat_dynamic_real.shape[-1]
            )
            variate_id.extend(
                [
                    repeat(
                        torch.arange(feat_dynamic_real.shape[-1], device=device)
                        + dim_count,
                        f"dim -> {' '.join(map(str, batch_shape))} (dim past)",
                        past=self.context_token_length(patch_size),
                    ),
                    repeat(
                        torch.arange(feat_dynamic_real.shape[-1], device=device)
                        + dim_count,
                        f"dim -> {' '.join(map(str, batch_shape))} (dim future)",
                        future=self.prediction_token_length(patch_size),
                    ),
                ]
            )
            dim_count += feat_dynamic_real.shape[-1]
            prediction_mask.extend(
                [
                    torch.zeros(
                        batch_shape
                        + (
                            self.context_token_length(patch_size)
                            * feat_dynamic_real.shape[-1],
                        ),
                        dtype=torch.bool,
                        device=device,
                    ),
                    torch.zeros(
                        batch_shape
                        + (
                            self.prediction_token_length(patch_size)
                            * feat_dynamic_real.shape[-1],
                        ),
                        dtype=torch.bool,
                        device=device,
                    ),
                ]
            )

        if past_feat_dynamic_real is not None:
            if past_observed_feat_dynamic_real is None:
                raise ValueError(
                    "past_observed_feat_dynamic_real must be provided if past_feat_dynamic_real is provided"
                )
            target.append(
                torch.nn.functional.pad(
                    rearrange(
                        self._patched_seq_pad(
                            patch_size, past_feat_dynamic_real, -2, left=True
                        ),
                        "... (seq patch) dim -> ... (dim seq) patch",
                        patch=patch_size,
                    ),
                    (0, 0),
                )
            )
            observed_mask.append(
                torch.nn.functional.pad(
                    rearrange(
                        self._patched_seq_pad(
                            patch_size, past_observed_feat_dynamic_real, -2, left=True
                        ),
                        "... (seq patch) dim -> ... (dim seq) patch",
                        patch=patch_size,
                    ),
                    (0, 0),
                )
            )
            sample_id.append(
                repeat(
                    reduce(
                        (
                            self._patched_seq_pad(
                                patch_size, past_is_pad, -1, left=True
                            )
                            == 0
                        ).int(),
                        "... (seq patch) -> ... seq",
                        "max",
                        patch=patch_size,
                    ),
                    "... seq -> ... (dim seq)",
                    dim=past_feat_dynamic_real.shape[-1],
                )
            )
            time_id.extend([past_seq_id] * past_feat_dynamic_real.shape[-1])

            variate_id.append(
                repeat(
                    torch.arange(past_feat_dynamic_real.shape[-1], device=device)
                    + dim_count,
                    f"dim -> {' '.join(map(str, batch_shape))} (dim past)",
                    past=self.context_token_length(patch_size),
                )
            )
            dim_count += past_feat_dynamic_real.shape[-1]
            prediction_mask.append(
                torch.zeros(
                    batch_shape
                    + (
                        self.context_token_length(patch_size)
                        * past_feat_dynamic_real.shape[-1],
                    ),
                    dtype=torch.bool,
                    device=device,
                )
            )

        target = torch.cat(target, dim=-2)
        observed_mask = torch.cat(observed_mask, dim=-2)
        sample_id = torch.cat(sample_id, dim=-1)
        time_id = torch.cat(time_id, dim=-1)
        variate_id = torch.cat(variate_id, dim=-1)
        prediction_mask = torch.cat(prediction_mask, dim=-1)
        return (
            target,
            observed_mask,
            sample_id,
            time_id,
            variate_id,
            prediction_mask,
        )

    # ------------------------------------------------------------------
    # Forward Pass
    # ------------------------------------------------------------------

    def _forward_quantiles(
        self,
        past_target: Float[torch.Tensor, "batch past_time tgt"],
        past_observed_target: Bool[torch.Tensor, "batch past_time tgt"],
        past_is_pad: Bool[torch.Tensor, "batch past_time"],
        future_target: Optional[Float[torch.Tensor, "batch future_time tgt"]] = None,
        future_observed_target: Optional[
            Bool[torch.Tensor, "batch future_time tgt"]
        ] = None,
        future_is_pad: Optional[Bool[torch.Tensor, "batch future_time"]] = None,
        feat_dynamic_real: Optional[Float[torch.Tensor, "batch time feat"]] = None,
        observed_feat_dynamic_real: Optional[
            Float[torch.Tensor, "batch time feat"]
        ] = None,
        past_feat_dynamic_real: Optional[
            Float[torch.Tensor, "batch past_time past_feat"]
        ] = None,
        past_observed_feat_dynamic_real: Optional[
            Float[torch.Tensor, "batch past_time past_feat"]
        ] = None,
    ) -> Float[torch.Tensor, "batch num_quantiles future_time tgt"]:
        param_dtype = next(self.propagator.parameters()).dtype
        device = next(self.propagator.parameters()).device

        past_target = past_target.to(device=device, dtype=param_dtype)
        past_observed_target = past_observed_target.to(device=device, dtype=torch.bool)
        past_is_pad = past_is_pad.to(device=device, dtype=torch.bool)

        if future_target is not None:
            future_target = future_target.to(device=device, dtype=param_dtype)
        if future_observed_target is not None:
            future_observed_target = future_observed_target.to(
                device=device, dtype=torch.bool
            )
        if future_is_pad is not None:
            future_is_pad = future_is_pad.to(device=device, dtype=torch.bool)
        if feat_dynamic_real is not None:
            feat_dynamic_real = feat_dynamic_real.to(device=device, dtype=param_dtype)
        if observed_feat_dynamic_real is not None:
            observed_feat_dynamic_real = observed_feat_dynamic_real.to(
                device=device, dtype=torch.bool
            )
        if past_feat_dynamic_real is not None:
            past_feat_dynamic_real = past_feat_dynamic_real.to(
                device=device, dtype=param_dtype
            )
        if past_observed_feat_dynamic_real is not None:
            past_observed_feat_dynamic_real = past_observed_feat_dynamic_real.to(
                device=device, dtype=torch.bool
            )

        (
            target,
            observed_mask,
            sample_id,
            time_id,
            variate_id,
            prediction_mask,
        ) = self._convert(
            self.propagator.patch_size,
            past_target,
            past_observed_target,
            past_is_pad,
            future_target=future_target,
            future_observed_target=future_observed_target,
            future_is_pad=future_is_pad,
            feat_dynamic_real=feat_dynamic_real,
            observed_feat_dynamic_real=observed_feat_dynamic_real,
            past_feat_dynamic_real=past_feat_dynamic_real,
            past_observed_feat_dynamic_real=past_observed_feat_dynamic_real,
        )

        per_var_context_token = self.context_token_length(self.propagator.patch_size)
        total_context_token = self.target_dim * per_var_context_token
        per_var_predict_token = self.prediction_token_length(self.propagator.patch_size)
        total_predict_token = self.target_dim * per_var_predict_token

        pred_index = torch.arange(
            start=per_var_context_token - 1,
            end=total_context_token,
            step=per_var_context_token,
            device=device,
        )
        assign_index = torch.arange(
            start=total_context_token,
            end=total_context_token + total_predict_token,
            step=per_var_predict_token,
            device=device,
        )
        quantile_prediction = repeat(
            target,
            "... patch_size -> ... num_quantiles patch_size",
            num_quantiles=len(self.propagator.quantile_levels),
            patch_size=self.propagator.patch_size,
        ).clone()

        preds = self.propagator(
            target,
            observed_mask,
            sample_id,
            time_id,
            variate_id,
            prediction_mask,
            training_mode=False,
        )

        def structure_multi_predict(
            step_predict_token,
            cur_pred_index,
            cur_assign_index,
            raw_preds,
        ):
            structured = rearrange(
                raw_preds,
                "... (predict_token num_quantiles patch_size) -> ... predict_token num_quantiles patch_size",
                predict_token=self.propagator.num_predict_token,
                num_quantiles=self.propagator.num_quantiles,
                patch_size=self.propagator.patch_size,
            )
            structured = rearrange(
                structured[..., cur_pred_index, :step_predict_token, :, :],
                "... pred_index predict_token num_quantiles patch_size -> ... (pred_index predict_token) num_quantiles patch_size",
            )
            adjusted_assign_index = torch.cat(
                [
                    torch.arange(start=idx, end=idx + step_predict_token, device=device)
                    for idx in cur_assign_index
                ]
            )
            return structured, adjusted_assign_index

        if per_var_predict_token <= self.propagator.num_predict_token:
            preds, adjusted_assign_index = structure_multi_predict(
                per_var_predict_token,
                pred_index,
                assign_index,
                preds,
            )
            quantile_prediction = quantile_prediction.clone()
            quantile_prediction[..., adjusted_assign_index, :, :] = preds
            return self._format_preds(
                self.propagator.num_quantiles,
                self.propagator.patch_size,
                quantile_prediction,
                self.target_dim,
            )
        else:
            expand_target = repeat(
                target,
                "batch_size ...  -> batch_size num_quantiles ...",
                num_quantiles=len(self.propagator.quantile_levels),
                batch_size=target.shape[0],
            ).clone()
            expand_prediction_mask = repeat(
                prediction_mask,
                "batch_size ...  -> batch_size num_quantiles ...",
                num_quantiles=len(self.propagator.quantile_levels),
                batch_size=target.shape[0],
            ).clone()
            expand_observed_mask = repeat(
                observed_mask,
                "batch_size ...  -> batch_size num_quantiles ...",
                num_quantiles=len(self.propagator.quantile_levels),
                batch_size=target.shape[0],
            ).clone()
            expand_sample_id = repeat(
                sample_id,
                "batch_size ...  -> batch_size num_quantiles ...",
                num_quantiles=len(self.propagator.quantile_levels),
                batch_size=target.shape[0],
            ).clone()
            expand_time_id = repeat(
                time_id,
                "batch_size ...  -> batch_size num_quantiles ...",
                num_quantiles=len(self.propagator.quantile_levels),
                batch_size=target.shape[0],
            ).clone()
            expand_variate_id = repeat(
                variate_id,
                "batch_size ...  -> batch_size num_quantiles ...",
                num_quantiles=len(self.propagator.quantile_levels),
                batch_size=target.shape[0],
            ).clone()

            preds, adjusted_assign_index = structure_multi_predict(
                self.propagator.num_predict_token,
                pred_index,
                assign_index,
                preds,
            )
            quantile_prediction = quantile_prediction.clone()
            quantile_prediction[..., adjusted_assign_index, :, :] = preds

            expand_target = expand_target.clone()
            expand_target[..., adjusted_assign_index, :] = rearrange(
                preds.detach(),
                "... predict_token num_quantiles patch_size -> ... num_quantiles predict_token patch_size",
                num_quantiles=self.propagator.num_quantiles,
                patch_size=self.propagator.patch_size,
                predict_token=len(adjusted_assign_index),
            )
            expand_prediction_mask = expand_prediction_mask.clone()
            expand_prediction_mask[..., adjusted_assign_index] = False

            remain_step = per_var_predict_token - self.propagator.num_predict_token
            while remain_step > 0:
                preds = self.propagator(
                    expand_target,
                    expand_observed_mask,
                    expand_sample_id,
                    expand_time_id,
                    expand_variate_id,
                    expand_prediction_mask,
                    training_mode=False,
                )

                pred_index = assign_index + self.propagator.num_predict_token - 1
                assign_index = pred_index + 1
                step_tokens = (
                    self.propagator.num_predict_token
                    if remain_step - self.propagator.num_predict_token > 0
                    else remain_step
                )
                preds, adjusted_assign_index = structure_multi_predict(
                    step_tokens,
                    pred_index,
                    assign_index,
                    preds,
                )
                quantile_prediction_next_step = rearrange(
                    preds,
                    "... num_quantiles_prev pred_index num_quantiles patch_size -> ... pred_index (num_quantiles_prev num_quantiles) patch_size",
                    num_quantiles=self.propagator.num_quantiles,
                    patch_size=self.propagator.patch_size,
                )
                quantile_prediction_next_step = torch.quantile(
                    quantile_prediction_next_step,
                    torch.tensor(
                        self.propagator.quantile_levels,
                        device=device,
                        dtype=quantile_prediction_next_step.dtype,
                    ),
                    dim=-2,
                )
                quantile_prediction = quantile_prediction.clone()
                quantile_prediction[..., adjusted_assign_index, :, :] = rearrange(
                    quantile_prediction_next_step,
                    "num_quantiles ... patch_size -> ... num_quantiles patch_size",
                )

                expand_target = expand_target.clone()
                expand_target[..., adjusted_assign_index, :] = rearrange(
                    quantile_prediction_next_step.detach(),
                    "num_quantiles batch_size predict_token patch_size -> batch_size num_quantiles predict_token patch_size",
                    num_quantiles=self.propagator.num_quantiles,
                    patch_size=self.propagator.patch_size,
                    predict_token=len(adjusted_assign_index),
                )
                expand_prediction_mask = expand_prediction_mask.clone()
                expand_prediction_mask[..., adjusted_assign_index] = False

                remain_step -= self.propagator.num_predict_token

            return self._format_preds(
                self.propagator.num_quantiles,
                self.propagator.patch_size,
                quantile_prediction,
                self.target_dim,
            )

    def _extract_point_prediction(
        self, quantile_preds: Float[torch.Tensor, "batch num_quantiles future_time tgt"]
    ) -> Float[torch.Tensor, "batch future_time tgt"]:
        """
        Extract the median (q=0.5) point prediction from ``quantile_preds``.
        """
        q_levels = list(self.propagator.quantile_levels)
        if 0.5 in q_levels:
            median_idx = q_levels.index(0.5)
            return quantile_preds[:, median_idx, :, :]
        mid_idx = len(q_levels) // 2
        return quantile_preds[:, mid_idx, :, :]

    def _parse_forward_inputs(
        self,
        data: Union[Dict[str, torch.Tensor], torch.Tensor],
        past_observed_target: Optional[torch.Tensor] = None,
        past_is_pad: Optional[torch.Tensor] = None,
        feat_dynamic_real: Optional[torch.Tensor] = None,
        observed_feat_dynamic_real: Optional[torch.Tensor] = None,
        past_feat_dynamic_real: Optional[torch.Tensor] = None,
        past_observed_feat_dynamic_real: Optional[torch.Tensor] = None,
    ) -> Dict[str, Optional[torch.Tensor]]:
        """
        Normalize input representation from either a ``DartsTimeseriesDataset``
        dictionary, a 3-D tensor ``(batch, sequence_length, target_dim)``, or
        explicit tensor arguments.
        """
        if isinstance(data, dict):
            return {
                "past_target": data["past_target"],
                "past_observed_target": data.get(
                    "past_observed_target",
                    ~torch.isnan(data["past_target"]),
                ),
                "past_is_pad": data.get(
                    "past_is_pad",
                    torch.zeros(
                        data["past_target"].shape[:2],
                        dtype=torch.bool,
                        device=data["past_target"].device,
                    ),
                ),
                "future_target": data.get("future_target", None),
                "future_observed_target": data.get("future_observed_target", None),
                "future_is_pad": data.get("future_is_pad", None),
                "feat_dynamic_real": data.get("feat_dynamic_real", None),
                "observed_feat_dynamic_real": data.get(
                    "observed_feat_dynamic_real", None
                ),
                "past_feat_dynamic_real": data.get("past_feat_dynamic_real", None),
                "past_observed_feat_dynamic_real": data.get(
                    "past_observed_feat_dynamic_real", None
                ),
            }

        if isinstance(data, torch.Tensor):
            if past_observed_target is not None and past_is_pad is not None:
                return {
                    "past_target": data,
                    "past_observed_target": past_observed_target,
                    "past_is_pad": past_is_pad,
                    "future_target": None,
                    "future_observed_target": None,
                    "future_is_pad": None,
                    "feat_dynamic_real": feat_dynamic_real,
                    "observed_feat_dynamic_real": observed_feat_dynamic_real,
                    "past_feat_dynamic_real": past_feat_dynamic_real,
                    "past_observed_feat_dynamic_real": past_observed_feat_dynamic_real,
                }

            seq_tensor = data
            if seq_tensor.ndim == 2:
                seq_tensor = seq_tensor.unsqueeze(-1)
            past_t = seq_tensor[:, : self.context_length, :]
            fut_t = (
                seq_tensor[
                    :,
                    self.context_length : self.context_length + self.prediction_length,
                    :,
                ]
                if seq_tensor.shape[1] > self.context_length
                else None
            )
            past_obs = ~torch.isnan(past_t)
            fut_obs = ~torch.isnan(fut_t) if fut_t is not None else None
            past_pad = torch.zeros(
                past_t.shape[:2], dtype=torch.bool, device=past_t.device
            )
            fut_pad = (
                torch.zeros(fut_t.shape[:2], dtype=torch.bool, device=fut_t.device)
                if fut_t is not None
                else None
            )
            return {
                "past_target": torch.nan_to_num(past_t, nan=0.0),
                "past_observed_target": past_obs,
                "past_is_pad": past_pad,
                "future_target": (
                    torch.nan_to_num(fut_t, nan=0.0) if fut_t is not None else None
                ),
                "future_observed_target": fut_obs,
                "future_is_pad": fut_pad,
                "feat_dynamic_real": feat_dynamic_real,
                "observed_feat_dynamic_real": observed_feat_dynamic_real,
                "past_feat_dynamic_real": past_feat_dynamic_real,
                "past_observed_feat_dynamic_real": past_observed_feat_dynamic_real,
            }

        self.raise_error(
            f"Unsupported input type for Moirai2.forward: {type(data)}. "
            "Expected a dict of tensors from DartsTimeseriesDataset or a torch.Tensor."
        )

    def _validate_prediction_type(self) -> None:
        """
        Validate that ``self.prediction_type`` is a supported forecasting mode.

        Raises
        ------
        ValueError
            If ``self.prediction_type`` is not one of the supported autoregressive
            or teacher-forced identifiers.
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
        Return ``True`` if the model is configured for autoregressive rollout.

        Returns
        -------
        bool
            ``True`` when ``self.prediction_type`` is in ``_AUTOREGRESSIVE_TYPES``.
        """
        return str(self.prediction_type).strip().lower() in self._AUTOREGRESSIVE_TYPES

    def _forward_autoregressive(
        self,
        parsed: Dict[str, Optional[torch.Tensor]],
    ) -> Float[torch.Tensor, "batch num_quantiles future_time tgt"]:
        """
        Execute an autoregressive rollout across the batch of sequence windows.

        Starts from ``self._rollout_context`` (when continuing across evaluation
        batches) or the initial context window ``parsed['past_target'][0:1]``, and
        iteratively predicts ``prediction_length`` steps at a time, sliding the
        context window forward using the model's own median point predictions.

        Parameters
        ----------
        parsed : dict of str to torch.Tensor or None
            Normalized forward inputs returned by :meth:`_parse_forward_inputs`.

        Returns
        -------
        torch.Tensor
            Stacked quantile predictions of shape
            ``(batch_size, num_quantiles, prediction_length, target_dim)``.
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

        feat_dyn = parsed["feat_dynamic_real"]
        obs_feat_dyn = parsed["observed_feat_dynamic_real"]
        past_feat_dyn = parsed["past_feat_dynamic_real"]
        past_obs_feat_dyn = parsed["past_observed_feat_dynamic_real"]

        step_quantile_preds: List[torch.Tensor] = []
        for b in range(batch_size):
            cur_obs = ~torch.isnan(cur_context)
            cur_pad = torch.zeros(
                cur_context.shape[:2],
                dtype=torch.bool,
                device=cur_context.device,
            )
            step_q_preds = self._forward_quantiles(
                past_target=cur_context,
                past_observed_target=cur_obs,
                past_is_pad=cur_pad,
                future_target=None,
                future_observed_target=None,
                future_is_pad=None,
                feat_dynamic_real=(
                    feat_dyn[b : b + 1] if feat_dyn is not None else None
                ),
                observed_feat_dynamic_real=(
                    obs_feat_dyn[b : b + 1] if obs_feat_dyn is not None else None
                ),
                past_feat_dynamic_real=(
                    past_feat_dyn[b : b + 1] if past_feat_dyn is not None else None
                ),
                past_observed_feat_dynamic_real=(
                    past_obs_feat_dyn[b : b + 1]
                    if past_obs_feat_dyn is not None
                    else None
                ),
            )
            step_quantile_preds.append(step_q_preds)

            step_point_pred = self._extract_point_prediction(step_q_preds).to(
                device=cur_context.device, dtype=cur_context.dtype
            )
            cur_context = torch.cat([cur_context, step_point_pred], dim=1)[
                :, -self.context_length :, :
            ]

        self._rollout_context = cur_context.detach()
        return torch.cat(step_quantile_preds, dim=0)

    def forward(
        self,
        data: Union[Dict[str, torch.Tensor], torch.Tensor],
        past_observed_target: Optional[torch.Tensor] = None,
        past_is_pad: Optional[torch.Tensor] = None,
        feat_dynamic_real: Optional[torch.Tensor] = None,
        observed_feat_dynamic_real: Optional[torch.Tensor] = None,
        past_feat_dynamic_real: Optional[torch.Tensor] = None,
        past_observed_feat_dynamic_real: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """
        Execute the Moirai2 forward pass and return ``(point_pred, past_target, meta)``
        in the :class:`~collective_encoder.models.neural_nets.base.CENetBase` format.
        """
        parsed = self._parse_forward_inputs(
            data,
            past_observed_target=past_observed_target,
            past_is_pad=past_is_pad,
            feat_dynamic_real=feat_dynamic_real,
            observed_feat_dynamic_real=observed_feat_dynamic_real,
            past_feat_dynamic_real=past_feat_dynamic_real,
            past_observed_feat_dynamic_real=past_observed_feat_dynamic_real,
        )

        if not self.training and self._is_autoregressive():
            quantile_preds = self._forward_autoregressive(parsed)
        else:
            quantile_preds = self._forward_quantiles(
                past_target=parsed["past_target"],
                past_observed_target=parsed["past_observed_target"],
                past_is_pad=parsed["past_is_pad"],
                future_target=parsed["future_target"],
                future_observed_target=parsed["future_observed_target"],
                future_is_pad=parsed["future_is_pad"],
                feat_dynamic_real=parsed["feat_dynamic_real"],
                observed_feat_dynamic_real=parsed["observed_feat_dynamic_real"],
                past_feat_dynamic_real=parsed["past_feat_dynamic_real"],
                past_observed_feat_dynamic_real=parsed["past_observed_feat_dynamic_real"],
            )

        point_pred = self._extract_point_prediction(quantile_preds)
        meta: Dict[str, Any] = {
            "quantile_preds": quantile_preds,
            "quantile_preds_btcq": rearrange(quantile_preds, "b q t c -> b t c q"),
            "future_target": parsed["future_target"],
            "future_observed_target": parsed["future_observed_target"],
            "future_is_pad": parsed["future_is_pad"],
        }
        return point_pred, parsed["past_target"], meta

    # ------------------------------------------------------------------
    # Losses & Metrics
    # ------------------------------------------------------------------

    def _resolve_future_target(
        self,
        output: torch.Tensor,
        labels: Optional[torch.Tensor],
        meta: Dict[str, Any],
    ) -> torch.Tensor:
        future_target = meta.get("future_target")
        if future_target is not None:
            return future_target.to(device=output.device, dtype=output.dtype)
        if labels is not None and labels.numel() == output.numel():
            return labels.to(device=output.device, dtype=output.dtype).view_as(output)
        self.raise_error(
            "Moirai2 requires 'future_target' in the input batch (via DartsTimeseriesDataset "
            "with prediction_length > 0) or matching target labels to compute loss/metrics."
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
        Compute the pinball (quantile) loss across all ``quantile_levels``.
        """
        quantile_preds = meta["quantile_preds_btcq"]  # (B, T_pred, D, Q)
        future_target = self._resolve_future_target(output, labels, meta).to(
            dtype=quantile_preds.dtype
        )

        q_levels = torch.tensor(
            self.propagator.quantile_levels,
            device=quantile_preds.device,
            dtype=quantile_preds.dtype,
        ).view(1, 1, 1, -1)

        errors = future_target.unsqueeze(-1) - quantile_preds
        pinball = torch.maximum(q_levels * errors, (q_levels - 1.0) * errors)

        mask = meta.get("future_observed_target")
        if mask is not None:
            mask = mask.to(device=quantile_preds.device, dtype=quantile_preds.dtype)
            fut_pad = meta.get("future_is_pad")
            if fut_pad is not None:
                mask = mask * (~fut_pad.to(device=quantile_preds.device)).unsqueeze(
                    -1
                ).to(quantile_preds.dtype)
            pinball = pinball * mask.unsqueeze(-1)
            denom = (mask.sum() * quantile_preds.shape[-1]).clamp_min(1.0)
            loss = pinball.sum() / denom
        else:
            loss = pinball.mean()

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
        Compute the Mean Absolute Error (MAE) of the median point forecast.
        """
        future_target = self._resolve_future_target(output, labels, meta)
        abs_err = torch.abs(output - future_target)

        mask = meta.get("future_observed_target")
        if mask is not None:
            mask = mask.to(device=output.device, dtype=output.dtype)
            fut_pad = meta.get("future_is_pad")
            if fut_pad is not None:
                mask = mask * (~fut_pad.to(device=output.device)).unsqueeze(-1).to(
                    output.dtype
                )
            mae = (abs_err * mask).sum() / mask.sum().clamp_min(1.0)
        else:
            mae = abs_err.mean()

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
            Resolved datamodule from ``datamodule``, ``self.trainer.datamodule``,
            or ``self.datamodule``, or ``None`` if unavailable.
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

    def _plot_test_start(self) -> None:
        """
        Initialize test plotters prior to evaluation.

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
            Context sequence tensor of shape ``(batch_size, context_length, target_dim)``.
        output : torch.Tensor
            Point prediction tensor of shape ``(batch_size, prediction_length, target_dim)``.
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
        seq_len = int(self.context_length + self.prediction_length)

        if isinstance(labels, torch.Tensor) and n_labels > 0:
            batch_size = labels.shape[0]
            if (
                labels.ndim == 3
                and labels.shape[1] >= self.context_length + self.prediction_length
            ):
                future_labels = labels[
                    :,
                    self.context_length : self.context_length + self.prediction_length,
                    :,
                ]
                labels = future_labels.reshape(-1, n_labels)
            elif labels.ndim == 2 and labels.shape[1] == seq_len * n_labels:
                labels_3d = labels.reshape(batch_size, seq_len, n_labels)
                future_labels = labels_3d[
                    :,
                    self.context_length : self.context_length + self.prediction_length,
                    :,
                ]
                labels = future_labels.reshape(-1, n_labels)
            elif (
                labels.ndim == 2
                and labels.shape[1] == self.prediction_length * n_labels
            ):
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
    # Prediction
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def predict_step(
        self, batch: Any, batch_idx: int, dataloader_idx: int = 0
    ) -> Dict[str, torch.Tensor]:
        data, labels = self._batch_split(batch)
        point_pred, past_target, meta = self(data)
        result: Dict[str, torch.Tensor] = {
            "point_prediction": point_pred,
            "quantile_predictions": meta["quantile_preds"],
            "past_target": past_target,
        }
        if meta.get("future_target") is not None:
            result["future_target"] = meta["future_target"]
        return result

    def predict(
        self,
        past_target: List[Float[np.ndarray, "past_time tgt"]],
        feat_dynamic_real: Optional[
            List[Float[np.ndarray, "batch past_time tgt"]]
        ] = None,
        past_feat_dynamic_real: Optional[
            List[Float[np.ndarray, "batch past_time tgt"]]
        ] = None,
    ) -> Float[numpy.ndarray, "batch num_quantiles future_time *tgt"]:
        """
        Direct NumPy inference helper accepting lists of arrays.
        """
        data_entry = {
            "past_target": past_target,
            "feat_dynamic_real": feat_dynamic_real,
            "past_feat_dynamic_real": past_feat_dynamic_real,
        }

        data_entry["past_observed_target"] = [~np.isnan(x) for x in past_target]
        if feat_dynamic_real:
            data_entry["observed_feat_dynamic_real"] = [
                ~np.isnan(x) for x in feat_dynamic_real
            ]
        else:
            data_entry["observed_feat_dynamic_real"] = None

        if past_feat_dynamic_real:
            data_entry["past_observed_feat_dynamic_real"] = [
                ~np.isnan(x) for x in past_feat_dynamic_real
            ]
        else:
            data_entry["past_observed_feat_dynamic_real"] = None

        impute = CausalMeanImputation()

        def process_sample(sample):
            arr = np.asarray(sample)
            if arr.ndim == 1:
                arr = arr[:, np.newaxis]
            if np.issubdtype(arr.dtype, np.number) and np.isnan(arr).any():
                for col_idx in range(arr.shape[1]):
                    if np.isnan(arr[:, col_idx]).any():
                        arr[:, col_idx] = impute(arr[:, col_idx].copy())
            return arr

        for key, value in data_entry.items():
            if value is not None:
                data_entry[key] = [process_sample(sample) for sample in value]

        data_entry["past_is_pad"] = np.zeros(
            (len(data_entry["past_target"]), self.context_length), dtype=bool
        )

        for key in data_entry:
            if data_entry[key] is not None and isinstance(data_entry[key], list):
                target_len = (
                    self.context_length + self.prediction_length
                    if key in ("feat_dynamic_real", "observed_feat_dynamic_real")
                    else self.context_length
                )
                for idx in range(len(data_entry[key])):
                    if data_entry[key][idx].shape[0] > target_len:
                        data_entry[key][idx] = data_entry[key][idx][-target_len:, :]
                    elif data_entry[key][idx].shape[0] < target_len:
                        pad_length = target_len - data_entry[key][idx].shape[0]
                        pad_block = np.full(
                            (pad_length, data_entry[key][idx].shape[1]),
                            data_entry[key][idx][0],
                            dtype=data_entry[key][idx].dtype,
                        )
                        data_entry[key][idx] = np.concatenate(
                            [pad_block, data_entry[key][idx]], axis=0
                        )
                        if key == "past_target":
                            data_entry["past_is_pad"][idx, :pad_length] = True

        device = next(self.propagator.parameters()).device
        for k in ["past_target", "feat_dynamic_real", "past_feat_dynamic_real"]:
            if data_entry[k] is not None:
                data_entry[k] = torch.tensor(
                    np.array(data_entry[k]), device=device, dtype=torch.float32
                )

        for k in [
            "past_observed_target",
            "observed_feat_dynamic_real",
            "past_observed_feat_dynamic_real",
            "past_is_pad",
        ]:
            if data_entry[k] is not None:
                data_entry[k] = torch.tensor(
                    np.array(data_entry[k]), device=device, dtype=torch.bool
                )

        with torch.no_grad():
            quantile_preds = self._forward_quantiles(**data_entry)
            if self.target_dim == 1:
                quantile_preds = quantile_preds.squeeze(-1)
            predictions = quantile_preds.detach().cpu().numpy()
        return predictions
