"""
Temporal Fusion Transformer (TFT)
---------------------------------
Copied from Darts git repository
"""

from collections.abc import Sequence
from typing import Optional, Union
from venv import logger
from functools import wraps

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import LSTM as _LSTM

# from darts.logging import get_logger, raise_if, raise_if_not, raise_log
from propagators.tft_model.components import glu_variants, layer_norm_variants
from propagators.tft_model.components.glu_variants import GLU_FFN

from propagators.tft_model.modules import (
    _GateAddNorm,
    _GatedResidualNetwork,
    _InterpretableMultiHeadAttention,
    _MultiEmbedding,
    _VariableSelectionNetwork,
    get_embedding_size,
)
from darts.utils.likelihood_models.torch import QuantileRegression

def io_processor(forward):
    """Applies some input / output processing to PLForecastingModule.forward.
    Note that this wrapper must be added to each of PLForecastinModule's subclasses forward methods.
    Here is an example how to add the decorator:

    ```python
        @io_processor
        def forward(self, *args, **kwargs)
            pass
    ```

    Applies
    -------
    Reversible Instance Normalization
        normalizes batch input target features, and inverse transform the forward output back to the original scale
    """

    @wraps(forward)
    def forward_wrapper(self, x_in, *args, **kwargs):
        if not self.use_reversible_instance_norm:
            return forward(self, x_in, *args, **kwargs)

        # `x_in` is input batch tuple which by definition has the past features in the first element
        # starting with the first n target features; clone it to prevent target re-normalization
        past_features = x_in[0].clone()
        # apply reversible instance normalization
        past_features[:, :, : self.n_targets] = self.rin(
            past_features[:, :, : self.n_targets]
        )
        # run the forward pass
        out = forward(self, *((past_features, *x_in[1:]), *args), **kwargs)
        # inverse transform target output back to original scale
        if isinstance(out, tuple):
            # RNNModel return tuple with hidden state
            return self.rin.inverse(out[0]), *out[1:]
        else:
            # all other models return only the prediction
            return self.rin.inverse(out)

    return forward_wrapper

class ModifiedTFTModel(nn.Module):
    def __init__(
        self,
        input_chunk_length: int,
        output_chunk_length: int,
        output_dim: tuple[int, int],
        variables_meta: dict[str, dict[str, list[str]]],
        num_static_components: int,
        hidden_size: Union[int, list[int]],
        lstm_layers: int,
        num_attention_heads: int,
        full_attention: bool,
        feed_forward: str,
        hidden_continuous_size: int,
        categorical_embedding_sizes: dict[str, tuple[int, int]],
        dropout: float,
        add_relative_index: bool,
        norm_type: Union[str, nn.Module],
        retain_lstm_cell_state: bool = False,
    ):
        """Modified implementing of TFT architecture from `this paper <https://arxiv.org/pdf/1912.09363.pdf>`_
        The implementation is built upon Darts library's TemporalFusionTransformer

        """

        super().__init__()

        self.n_targets, self.loss_size = output_dim
        self.variables_meta = variables_meta
        self.num_static_components = num_static_components
        self.hidden_size = hidden_size
        self.hidden_continuous_size = hidden_continuous_size
        self.categorical_embedding_sizes = categorical_embedding_sizes
        self.lstm_layers = lstm_layers
        self.num_attention_heads = num_attention_heads
        self.full_attention = full_attention
        self.feed_forward = feed_forward
        self.dropout = dropout
        self.add_relative_index = add_relative_index
        self.retain_lstm_cell_state = retain_lstm_cell_state

        self.input_chunk_length = input_chunk_length
        self.output_chunk_length = output_chunk_length
        self.sequence_length = input_chunk_length + output_chunk_length

        self.use_reversible_instance_norm = False
        if self.use_reversible_instance_norm:
            self.rin = layer_norm_variants.RINorm(input_dim=self.n_targets)
        else:
            self.rin = None

        if isinstance(norm_type, str):
            try:
                self.layer_norm = getattr(layer_norm_variants, norm_type)
            except AttributeError:
                raise AttributeError("please provide a valid layer norm type")
        else:
            self.layer_norm = norm_type

        # initialize last batch size to check if new mask needs to be generated
        self.batch_size_last = -1
        self.attention_mask = None
        self.relative_index = None

        # general information on variable name endings:
        # _vsn: VariableSelectionNetwork
        # _grn: GatedResidualNetwork
        # _glu: GatedLinearUnit
        # _gan: GateAddNorm
        # _attn: Attention

        # # processing inputs
        # embeddings
        self.input_embeddings = _MultiEmbedding(
            embedding_sizes=categorical_embedding_sizes,
            variable_names=self.categorical_static_variables,
        )

        # continuous variable processing
        self.prescalers_linear = {
            name: nn.Linear(
                (
                    1
                    if name not in self.numeric_static_variables
                    else self.num_static_components
                ),
                self.hidden_continuous_size,
            )
            for name in self.reals
        }

        if self.retain_lstm_cell_state:
            self.prescalers_linear["decoder_cell_state"] = nn.Linear(
                self.hidden_size, self.hidden_continuous_size
            )

        # static (categorical and numerical) variables
        static_input_sizes = {
            name: self.input_embeddings.output_size[name]
            for name in self.categorical_static_variables
        }
        static_input_sizes.update({
            name: self.hidden_continuous_size for name in self.numeric_static_variables
        })

        if self.retain_lstm_cell_state:
            static_input_sizes["decoder_cell_state"] = self.hidden_continuous_size

        self.static_covariates_vsn = _VariableSelectionNetwork(
            input_sizes=static_input_sizes,
            hidden_size=self.hidden_size,
            input_embedding_flags={
                name: True for name in self.categorical_static_variables
            },
            dropout=self.dropout,
            prescalers=self.prescalers_linear,
            single_variable_grns={},
            context_size=None,  # no context for static variables
            layer_norm=self.layer_norm,
        )

        # variable selection for encoder and decoder
        encoder_input_sizes = {
            name: self.hidden_continuous_size for name in self.encoder_variables
        }

        decoder_input_sizes = {
            name: self.hidden_continuous_size for name in self.decoder_variables
        }

        self.encoder_vsn = _VariableSelectionNetwork(
            input_sizes=encoder_input_sizes,
            hidden_size=self.hidden_size,
            input_embedding_flags={},  # this would be required for non-static categorical inputs
            dropout=self.dropout,
            context_size=self.hidden_size,
            prescalers=self.prescalers_linear,
            single_variable_grns={},
            layer_norm=self.layer_norm,
        )

        self.decoder_vsn = _VariableSelectionNetwork(
            input_sizes=decoder_input_sizes,
            hidden_size=self.hidden_size,
            input_embedding_flags={},  # this would be required for non-static categorical inputs
            dropout=self.dropout,
            context_size=self.hidden_size,
            prescalers=self.prescalers_linear,
            single_variable_grns={},
            layer_norm=self.layer_norm,
        )

        # static encoders
        # for variable selection
        self.static_context_grn = _GatedResidualNetwork(
            input_size=self.hidden_size,
            hidden_size=self.hidden_size,
            output_size=self.hidden_size,
            dropout=self.dropout,
            layer_norm=self.layer_norm,
        )

        # for hidden state of the lstm
        self.static_context_hidden_encoder_grn = _GatedResidualNetwork(
            input_size=self.hidden_size,
            hidden_size=self.hidden_size,
            output_size=self.hidden_size,
            dropout=self.dropout,
            layer_norm=self.layer_norm,
        )

        # for cell state of the lstm
        self.static_context_cell_encoder_grn = _GatedResidualNetwork(
            input_size=self.hidden_size,
            hidden_size=self.hidden_size,
            output_size=self.hidden_size,
            dropout=self.dropout,
            layer_norm=self.layer_norm,
        )

        # for post lstm static enrichment
        self.static_context_enrichment = _GatedResidualNetwork(
            input_size=self.hidden_size,
            hidden_size=self.hidden_size,
            output_size=self.hidden_size,
            dropout=self.dropout,
            layer_norm=self.layer_norm,
        )

        # lstm encoder (history) and decoder (future) for local processing
        self.lstm_encoder = _LSTM(
            input_size=self.hidden_size,
            hidden_size=self.hidden_size,
            num_layers=self.lstm_layers,
            dropout=self.dropout if self.lstm_layers > 1 else 0,
            batch_first=True,
        )

        self.lstm_decoder = _LSTM(
            input_size=self.hidden_size,
            hidden_size=self.hidden_size,
            num_layers=self.lstm_layers,
            dropout=self.dropout if self.lstm_layers > 1 else 0,
            batch_first=True,
        )

        # post lstm GateAddNorm
        self.post_lstm_gan = _GateAddNorm(
            input_size=self.hidden_size, dropout=dropout, layer_norm=self.layer_norm
        )

        # static enrichment and processing past LSTM
        self.static_enrichment_grn = _GatedResidualNetwork(
            input_size=self.hidden_size,
            hidden_size=self.hidden_size,
            output_size=self.hidden_size,
            dropout=self.dropout,
            context_size=self.hidden_size,
            layer_norm=self.layer_norm,
        )

        # attention for long-range processing
        self.multihead_attn = _InterpretableMultiHeadAttention(
            d_model=self.hidden_size,
            n_head=self.num_attention_heads,
            dropout=self.dropout,
        )
        self.post_attn_gan = _GateAddNorm(
            self.hidden_size, dropout=self.dropout, layer_norm=self.layer_norm
        )

        if self.feed_forward == "GatedResidualNetwork":
            self.feed_forward_block = _GatedResidualNetwork(
                self.hidden_size,
                self.hidden_size,
                self.hidden_size,
                dropout=self.dropout,
                layer_norm=self.layer_norm,
            )
        else:
            if self.feed_forward not in glu_variants.__all__:
                raise AttributeError(
                    f"'{self.feed_forward}' is not among implemented GLU variants: {glu_variants.__all__}"
                )
            # use glu variant feedforward layers
            # 4 is a commonly used feedforward multiplier
            self.feed_forward_block = getattr(glu_variants, self.feed_forward)(
                d_model=self.hidden_size, d_ff=self.hidden_size * 4, dropout=dropout
            )

        # output processing -> no dropout at this late stage
        self.pre_output_gan = _GateAddNorm(
            self.hidden_size, dropout=None, layer_norm=self.layer_norm
        )

        self.output_layer = nn.Linear(self.hidden_size, self.n_targets * self.loss_size)

        if self.retain_lstm_cell_state:
            self._retained_cell_state = None

        self._attn_out_weights = None
        self._static_covariate_var = None
        self._encoder_sparse_weights = None
        self._decoder_sparse_weights = None

    def reset_cell_state(self):
        """Reset the retained decoder LSTM cell state to zeros."""
        self._retained_cell_state = None

    @staticmethod
    def collect_meta(
        input_chunk_length: int,
        output_chunk_length: int,
        n_past_covariates: int,
        n_future_covariates: int,
        n_static_covariates: int,
        n_targets: int,
        add_relative_index: bool,
        likelihood=None,
    ):
        # Create fake inputs to extract variable names and sizes
        (
            past_target, 
            past_covariate, 
            historic_future_covariate, 
            future_covariate, 
            static_covariates, 
            future_target
        ) = (
            np.zeros((1, n_targets)) if n_targets > 0 else None,
            np.zeros((1, n_past_covariates)) if n_past_covariates > 0 else None,
            np.zeros((input_chunk_length, n_future_covariates)) if n_future_covariates > 0 else None,
            np.zeros((output_chunk_length, n_future_covariates)) if n_future_covariates > 0 else None,
            np.zeros((1, n_static_covariates)) if n_static_covariates > 0 else None,
            np.zeros((1, n_targets)) if n_targets > 0 else None,
        )

        if add_relative_index:
            time_steps = input_chunk_length + output_chunk_length

            expand_future_covariate = np.arange(time_steps).reshape((time_steps, 1))

            historic_future_covariate = np.concatenate(
                [
                    ts[: input_chunk_length]
                    for ts in [historic_future_covariate, expand_future_covariate]
                    if ts is not None
                ],
                axis=1,
            )
            future_covariate = np.concatenate(
                [
                    ts[-output_chunk_length :]
                    for ts in [future_covariate, expand_future_covariate]
                    if ts is not None
                ],
                axis=1,
            )

        output_dim = (
            (future_target.shape[1], 1)
            if likelihood is None
            else (future_target.shape[1], likelihood.num_parameters)
        )

        tensors = [
            past_target,
            past_covariate,
            historic_future_covariate,  # for time varying encoders
            future_covariate,
            future_target,  # for time varying decoders
            static_covariates,  # for static encoder
        ]
        type_names = [
            "past_target",
            "past_covariate",
            "historic_future_covariate",
            "future_covariate",
            "future_target",
            "static_covariate",
        ]
        variable_names = [
            "target",
            "past_covariate",
            "future_covariate",
            "future_covariate",
            "target",
            "static_covariate",
        ]

        variables_meta = {
            "input": {
                type_name: [f"{var_name}_{i}" for i in range(tensor.shape[1])]
                for type_name, var_name, tensor in zip(
                    type_names, variable_names, tensors
                )
                if tensor is not None
            },
            "model_config": {},
        }
        
        reals_input = []
        categorical_input = []
        time_varying_encoder_input = []
        time_varying_decoder_input = []
        static_input = []
        static_input_numeric = []
        static_input_categorical = []
        categorical_embedding_sizes = {}
        for input_var in type_names:
            if input_var in variables_meta["input"]:
                vars_meta = variables_meta["input"][input_var]
                if input_var in [
                    "past_target",
                    "past_covariate",
                    "historic_future_covariate",
                ]:
                    time_varying_encoder_input += vars_meta
                    reals_input += vars_meta
                elif input_var in ["future_covariate"]:
                    time_varying_decoder_input += vars_meta
                    reals_input += vars_meta
                elif input_var in ["static_covariate"]:
                    if (
                        static_covariates is None
                    ):  # when training with fit_from_dataset
                        static_cols = pd.Index([
                            i for i in range(static_covariates.shape[1])
                        ])
                    else:
                        static_cols = pd.Index([
                            i for i in range(static_covariates.shape[1])
                        ])
                    numeric_mask = ~static_cols.isin(categorical_embedding_sizes)
                    for idx, (static_var, col_name, is_numeric) in enumerate(
                        zip(vars_meta, static_cols, numeric_mask)
                    ):
                        static_input.append(static_var)
                        if is_numeric:
                            static_input_numeric.append(static_var)
                            reals_input.append(static_var)
                        else:
                            # get embedding sizes for each categorical variable
                            embedding = categorical_embedding_sizes[col_name]
                            assert isinstance(embedding, (int, tuple)), \
                                "Dict values of `categorical_embedding_sizes` must either be integers or tuples. Read " \
                                "the TFTModel documentation for more information."
                            if isinstance(embedding, int):
                                embedding = (embedding, get_embedding_size(n=embedding))
                            categorical_embedding_sizes[vars_meta[idx]] = embedding

                            static_input_categorical.append(static_var)
                            categorical_input.append(static_var)

        variables_meta["model_config"]["reals_input"] = list(dict.fromkeys(reals_input))
        variables_meta["model_config"]["categorical_input"] = list(
            dict.fromkeys(categorical_input)
        )
        variables_meta["model_config"]["time_varying_encoder_input"] = list(
            dict.fromkeys(time_varying_encoder_input)
        )
        variables_meta["model_config"]["time_varying_decoder_input"] = list(
            dict.fromkeys(time_varying_decoder_input)
        )
        variables_meta["model_config"]["static_input"] = list(
            dict.fromkeys(static_input)
        )
        variables_meta["model_config"]["static_input_numeric"] = list(
            dict.fromkeys(static_input_numeric)
        )
        variables_meta["model_config"]["static_input_categorical"] = list(
            dict.fromkeys(static_input_categorical)
        )

        n_static_components = (
            len(static_covariates) if static_covariates is not None else 0
        )

        return variables_meta, n_static_components, categorical_embedding_sizes, output_dim

    @property
    def reals(self) -> list[str]:
        """
        List of all continuous variables in model
        """
        return self.variables_meta["model_config"]["reals_input"]

    @property
    def static_variables(self) -> list[str]:
        """
        List of all static variables in model
        """
        return self.variables_meta["model_config"]["static_input"]

    @property
    def numeric_static_variables(self) -> list[str]:
        """
        List of numeric static variables in model
        """
        return self.variables_meta["model_config"]["static_input_numeric"]

    @property
    def categorical_static_variables(self) -> list[str]:
        """
        List of categorical static variables in model
        """
        return self.variables_meta["model_config"]["static_input_categorical"]

    @property
    def encoder_variables(self) -> list[str]:
        """
        List of all encoder variables in model (excluding static variables)
        """
        return self.variables_meta["model_config"]["time_varying_encoder_input"]

    @property
    def decoder_variables(self) -> list[str]:
        """
        List of all decoder variables in model (excluding static variables)
        """
        return self.variables_meta["model_config"]["time_varying_decoder_input"]

    @staticmethod
    def expand_static_context(context: torch.Tensor, time_steps: int) -> torch.Tensor:
        """
        add time dimension to static context
        """
        return context[:, None].expand(-1, time_steps, -1)

    @staticmethod
    def get_relative_index(
        encoder_length: int,
        decoder_length: int,
        batch_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Returns scaled time index relative to prediction point.
        """
        index = torch.arange(
            encoder_length + decoder_length, dtype=dtype, device=device
        )
        prediction_index = encoder_length - 1
        index[:encoder_length] = index[:encoder_length] / prediction_index
        index[encoder_length:] = index[encoder_length:] / prediction_index
        return index.reshape(1, len(index), 1).repeat(batch_size, 1, 1)

    @staticmethod
    def get_attention_mask_full(
        time_steps: int, batch_size: int, dtype: torch.dtype, device: torch.device
    ) -> torch.Tensor:
        """
        Returns causal mask to apply for self-attention layer.
        """
        eye = torch.eye(time_steps, dtype=dtype, device=device)
        mask = torch.cumsum(eye.unsqueeze(0).repeat(batch_size, 1, 1), dim=1)
        return mask < 1

    @staticmethod
    def get_attention_mask_future(
        encoder_length: int,
        decoder_length: int,
        batch_size: int,
        device: str,
        full_attention: bool,
    ) -> torch.Tensor:
        """
        Returns causal mask to apply for self-attention layer that acts on future input only.
        The model will attend to all `False` values.
        """
        if full_attention:
            # attend to entire past and future input
            decoder_mask = torch.zeros(
                (decoder_length, decoder_length), dtype=torch.bool, device=device
            )
        else:
            # attend only to past steps relative to forecasting step in the future
            # indices to which is attended
            attend_step = torch.arange(decoder_length, device=device)
            # indices for which is predicted
            predict_step = torch.arange(0, decoder_length, device=device)[:, None]
            # do not attend to steps to self or after prediction
            decoder_mask = attend_step >= predict_step
        # attend to all past input
        encoder_mask = torch.zeros(
            batch_size, encoder_length, dtype=torch.bool, device=device
        )
        # combine masks along attended time - first encoder and then decoder

        mask = torch.cat(
            (
                encoder_mask.unsqueeze(1).expand(-1, decoder_length, -1),
                decoder_mask.unsqueeze(0).expand(batch_size, -1, -1),
            ),
            dim=2,
        )
        return mask

    @io_processor
    def forward(self, x_in: tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]) -> torch.Tensor:
        """TFT model forward pass.

        Parameters
        ----------
        x_in
            comes as tuple `(x_past, x_future, x_static)` where `x_past` is the input/past chunk and `x_future`
            is the output/future chunk. Input dimensions are `(n_samples, n_time_steps, n_variables)`

        Returns
        -------
        torch.Tensor
            the output tensor
        """
        x_cont_past, x_cont_future, x_static = x_in
        dim_samples, dim_time, dim_variable = 0, 1, 2
        device = x_in[0].device

        batch_size = x_cont_past.shape[dim_samples]
        encoder_length = self.input_chunk_length
        decoder_length = self.output_chunk_length
        time_steps = encoder_length + decoder_length

        # avoid unnecessary regeneration of attention mask
        if batch_size != self.batch_size_last:
            self.attention_mask = self.get_attention_mask_future(
                encoder_length=encoder_length,
                decoder_length=decoder_length,
                batch_size=batch_size,
                device=device,
                full_attention=self.full_attention,
            )
            if self.add_relative_index:
                self.relative_index = self.get_relative_index(
                    encoder_length=encoder_length,
                    decoder_length=decoder_length,
                    batch_size=batch_size,
                    device=device,
                    dtype=x_cont_past.dtype,
                )

            self.batch_size_last = batch_size

        if self.add_relative_index:
            x_cont_past = torch.cat(
                [
                    ts[:, :encoder_length, :]
                    for ts in [x_cont_past, self.relative_index]
                    if ts is not None
                ],
                dim=dim_variable,
            )
            x_cont_future = torch.cat(
                [
                    ts[:, -decoder_length:, :]
                    for ts in [x_cont_future, self.relative_index]
                    if ts is not None
                ],
                dim=dim_variable,
            )

        input_vectors_past = {
            name: x_cont_past[..., idx].unsqueeze(-1)
            for idx, name in enumerate(self.encoder_variables)
        }
        input_vectors_future = {
            name: x_cont_future[..., idx].unsqueeze(-1)
            for idx, name in enumerate(self.decoder_variables)
        }

        # Embedding and variable selection
        if self.static_variables or self.retain_lstm_cell_state:
            # categorical static covariate embeddings
            if self.categorical_static_variables:
                static_embedding = self.input_embeddings(
                    torch.cat(
                        [
                            x_static[:, :, idx]
                            for idx, name in enumerate(self.static_variables)
                            if name in self.categorical_static_variables
                        ],
                        dim=1,
                    ).int()
                )
            else:
                static_embedding = {}
            # add numerical static covariates
            static_embedding.update({
                name: x_static[:, :, idx]
                for idx, name in enumerate(self.static_variables)
                if name in self.numeric_static_variables
            })
            # inject retained decoder cell state as static covariate
            if self.retain_lstm_cell_state:
                if self._retained_cell_state is None:
                    cell_val = torch.zeros(
                        batch_size, self.hidden_size,
                        device=device, dtype=x_cont_past.dtype,
                    )
                else:
                    cell_val = self._retained_cell_state
                static_embedding["decoder_cell_state"] = cell_val
            static_embedding, static_covariate_var = self.static_covariates_vsn(
                static_embedding
            )
        else:
            static_embedding = torch.zeros(
                (x_cont_past.shape[0], self.hidden_size),
                dtype=x_cont_past.dtype,
                device=device,
            )
            static_covariate_var = None
        
        

        static_context_expanded = self.expand_static_context(
            context=self.static_context_grn(static_embedding), time_steps=time_steps
        )

        embeddings_varying_encoder = {
            name: input_vectors_past[name] for name in self.encoder_variables
        }
        embeddings_varying_encoder, encoder_sparse_weights = self.encoder_vsn(
            x=embeddings_varying_encoder,
            context=static_context_expanded[:, :encoder_length],
        )

        embeddings_varying_decoder = {
            name: input_vectors_future[name] for name in self.decoder_variables
        }
        embeddings_varying_decoder, decoder_sparse_weights = self.decoder_vsn(
            x=embeddings_varying_decoder,
            context=static_context_expanded[:, encoder_length:],
        )

        # LSTM
        # calculate initial state
        input_hidden = (
            self.static_context_hidden_encoder_grn(static_embedding)
            .expand(self.lstm_layers, -1, -1)
            .contiguous()
        )
        input_cell = (
            self.static_context_cell_encoder_grn(static_embedding)
            .expand(self.lstm_layers, -1, -1)
            .contiguous()
        )

        # run local lstm encoder
        encoder_out, (hidden, cell) = self.lstm_encoder(
            input=embeddings_varying_encoder, hx=(input_hidden, input_cell)
        )
        # print("\n\n")
        # print(hidden.shape, cell.shape)
        # exit()
        # run local lstm decoder
        decoder_out, (_, decoder_cell) = self.lstm_decoder(
            input=embeddings_varying_decoder, hx=(hidden, cell)
        )
        if self.retain_lstm_cell_state:
            self._retained_cell_state = decoder_cell[-1].detach()

        lstm_layer = torch.cat([encoder_out, decoder_out], dim=dim_time)
        input_embeddings = torch.cat(
            [embeddings_varying_encoder, embeddings_varying_decoder], dim=dim_time
        )

        # post lstm GateAddNorm
        lstm_out = self.post_lstm_gan(x=lstm_layer, skip=input_embeddings)

        # static enrichment
        static_context_enriched = self.static_context_enrichment(static_embedding)
        attn_input = self.static_enrichment_grn(
            x=lstm_out,
            context=self.expand_static_context(
                context=static_context_enriched, time_steps=time_steps
            ),
        )

        # multi-head attention
        attn_out, attn_out_weights = self.multihead_attn(
            q=attn_input[:, encoder_length:],
            k=attn_input,
            v=attn_input,
            mask=self.attention_mask,
        )

        # skip connection over attention
        attn_out = self.post_attn_gan(
            x=attn_out,
            skip=attn_input[:, encoder_length:],
        )

        # feed-forward
        out = self.feed_forward_block(x=attn_out)

        # skip connection over temporal fusion decoder from LSTM post _GateAddNorm
        out = self.pre_output_gan(
            x=out,
            skip=lstm_out[:, encoder_length:],
        )

        # generate output for n_targets and loss_size elements for loss evaluation
        out = self.output_layer(out)
        out = out.view(
            batch_size, self.output_chunk_length, self.n_targets, self.loss_size
        )
        self._attn_out_weights = attn_out_weights
        self._static_covariate_var = static_covariate_var
        self._encoder_sparse_weights = encoder_sparse_weights
        self._decoder_sparse_weights = decoder_sparse_weights
        return out

