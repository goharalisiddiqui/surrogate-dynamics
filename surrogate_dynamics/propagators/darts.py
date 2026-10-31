from typing import Any, Dict, Tuple, Optional
from tqdm import tqdm
import importlib

import numpy as np
import inspect
import pkgutil

import torch
from torch import nn
import torch.nn.functional as F

from gslibs.utils.common import get_init_args, get_required_init_args

from collective_encoder.models.neural_nets.base import CENetBase
from collective_encoder.common.registry import get_module


class DartsForecastingModel(CENetBase):
    _IDENTFIER = "DARTS"
    _REQUIRED_ARGS = [
        'darts_model_name', 
        
    ]
    
    _OPTIONAL_ARGS = CENetBase._OPTIONAL_ARGS.copy()
    _OPTIONAL_ARGS.update({
        'darts_model_args': {},
    })
    
    def __init__(
        self,
        args: Optional[Dict[str, Any]],
        **kwargs
    ):
        super().__init__(args, **kwargs)
        
        # if self.likelihood is not None:
        #     self.likelihood = likelihood_resolver(self.likelihood, 
        #                                         self.likelihood_args or {})
        
        
        self._init_model()
        
        self.losses = {
            "loss_prop": self.loss_prop,
        }
        self.metrics = {
            'mae_prop': self.metric_mae_prop
        }
        self.test_metrics = self.metrics.copy()
        
    def _find_darts_model(self):
        """Import a module and return the class object matching class_name, if present."""
        package_name = "darts.models.forecasting"
        package = importlib.import_module(package_name)
        
        for _, modname, __ in pkgutil.iter_modules(package.__path__):
            full_modname = f"{package_name}.{modname}"
            try:
                module = importlib.import_module(full_modname)
                cls_obj = getattr(module, self.darts_model_name)
            except Exception:
                continue

            return cls_obj

        self.raise_error(
            f"Could not import DARTS model '{self.darts_model_name}'. "
            f"Ensure that the 'darts' package is installed and "
            f"the model name is correct. "
        )
    def _check_model_args(self, model_cls):
        required_args = get_required_init_args(model_cls)
        missing_args = [arg for arg in required_args if arg not in self.darts_model_args]
        if missing_args:
            self.raise_error(
                f"Missing required arguments for DARTS model '{self.darts_model_name}': {missing_args}. "
                f"Please provide these arguments in the configuration.")

    def _init_model(self):
        model_cls = self._find_darts_model()
            
        
        self._check_model_args(model_cls)
        model_args = get_init_args(model_cls)
        model_kwargs = {arg: self.darts_model_args.get(arg) for arg in model_args if arg in self.darts_model_args}
        self.forecast_model = model_cls(**model_kwargs)

    # ------------------------------------------------------------------
    # Normalization
    # ------------------------------------------------------------------

    def _validate_norm_sizes(self, Mean, Range):
        if self._encoder_mode: # Ruduntant check: We already check this in the constructor.
            self.raise_error("Normalization not supported when encoder-decoder model is used. ")
        if Mean.size(0) != self.latent_dim:
            raise ValueError(f"Mean size {Mean.size(0)} does not match expected {self.latent_dim}")
        if Range.size(0) != self.latent_dim:
            raise ValueError(f"Range size {Range.size(0)} does not match expected {self.latent_dim}")
    
    def get_norm_len(self) -> int:
        return 10

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        if self.Mean.numel() != np.prod(x.shape[1:]):
            self.raise_error(f"Mean and Range buffers must have the same number" 
                             f" of elements as the input features. Got Mean shape:"
                             f" {self.Mean.shape}, Range shape: {self.Range.shape},"
                             f" input shape: {x.shape}")
        mean_expanded = self.Mean.view(1, *(x.shape[1:])).expand(x.shape)
        range_expanded = self.Range.view(1, *(x.shape[1:])).expand(x.shape)
        return (x - mean_expanded) / range_expanded

    def _denormalize(self, x: torch.Tensor) -> torch.Tensor:
        if self.Mean.numel() != np.prod(x.shape[1:]):
            self.raise_error(f"Mean and Range buffers must have the same number" 
                             f" of elements as the input features. Got Mean shape:"
                             f" {self.Mean.shape}, Range shape: {self.Range.shape},"
                             f" input shape: {x.shape}")
        mean_expanded = self.Mean.view(1, *(x.shape[1:])).expand(x.shape)
        range_expanded = self.Range.view(1, *(x.shape[1:])).expand(x.shape)
        return x * range_expanded + mean_expanded
    
    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------
    
    def get_long_seq_length(self):
        return self.long_sequence_length
        
    def _batch_split(self, batch):
        if self._encoder_mode:
            return batch, self.extract_labels(batch)
        else:
            return super()._batch_split(batch)
    
    def prop_sample(self, prop_out):
        extra_args = {}
        if not self.training:
            extra_args['noise_type'] = self.inference_settings.get('noise_type', None)
            extra_args['noise_args'] = self.inference_settings.get('noise_args', {})
            
        if self.likelihood is not None:
            return self.likelihood.sample(prop_out, **extra_args)
        else:
            return prop_out.squeeze(-1)
        
    def forward(self, data):
        if self._encoder_mode:
            with torch.no_grad():
                latent, meta = self.encdec_model._encode(data)
        else:
            latent = self.normalize(latent)
        
        long_seq_length = self.get_long_seq_length()
        prop_latent = latent.view(-1, long_seq_length, self.latent_dim)
        
        i = 0
        while i+self.sequence_length <= long_seq_length:
            prop_in = prop_latent[:, i:i+self.propagator.input_chunk_length, :] # We take the input sequence always from encoded latent (Batch, T_in, Latent Dim)
            # TFT expects (B, T, C) inputs; returns (B, T_out, C)
            prop_out_chunk = self.propagator((prop_in, None, None)) # (Batch, T_out, Latent Dim)
            if i == 0:
                prop_out = prop_out_chunk
            else:
                prop_out = torch.cat([prop_out, prop_out_chunk], dim=1)
            i += self.propagator.output_chunk_length
        self.propagator.reset_cell_state()

        prop_sample = self.prop_sample(prop_out.clone())
        prop_sample = prop_sample.view(-1, self.latent_dim)
        
        if self._encoder_mode:
            meta['prop_dec'] = self.encdec_model._decode(prop_sample)[0]
            prop_sample = self.denormalize(prop_sample)
        meta['prop_out'] = prop_out

        return prop_sample, latent, meta
    
    # ------------------------------------------------------------------
    # Losses
    # ------------------------------------------------------------------
    
    def trim_labels_to_predicted_only(self, label):
        long_seq_length = self.get_long_seq_length()
        
        label = label.view(-1, long_seq_length, label.size(-1)) # reshape to (Batch, long_seq_length, C)
        label = label[:, self.input_chunk_length:, :] # keep only the part that we predict (Batch, T_pred, C)
        label = label.contiguous() # ensure memory contiguity for later operations
        label = label.view(-1, label.size(-1)) # flatten to (Batch * T_pred, C) to match the shape of the model output
        return label

    def loss_prop(self, inp, latent, output, labels, meta):
        long_seq_length = self.get_long_seq_length()
        
        latent = latent.view(-1, long_seq_length, self.latent_dim)
        latent = latent[:, self.propagator.input_chunk_length:, :]
        prop_out = meta['prop_out'].squeeze(-1)

        loss_prop = self.likelihood.compute_loss(prop_out, latent, None) if \
            self.likelihood is not None else \
                self.loss_fn(prop_out, latent)

        return loss_prop, {}
    
    def loss_topol(self, inp, latent, output, labels, meta):
        long_seq_length = self.get_long_seq_length()
        topol = meta['prop_dec']

        losses = {}
        loss_labels = self.out_labels.keys()
        for out_label, weight in zip(loss_labels, self.loss_encdec_weights):
            losses[out_label] = self.loss_fn(topol[out_label], 
                                        self.trim_labels_to_predicted_only(labels[out_label])) * weight

        return sum(losses.values()), losses

    def metric_mae_prop(self, inp, latent, output, labels, meta) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        long_seq_length = self.get_long_seq_length()
        
        latent = latent.view(-1, long_seq_length, self.latent_dim)
        latent = latent[:, self.propagator.input_chunk_length:, :]
        prop_out = meta['prop_out'].squeeze(-1)
        prop_out = self.prop_sample(prop_out.clone())

        mae_prop = F.l1_loss(prop_out, latent, reduction='none')
        mae_prop = mae_prop.mean(dim=-1) # Average over latent dim

        return mae_prop, {}

    def metric_mae_topol(self, inp, latent, output, labels, meta) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        topol = meta['prop_dec']

        losses = {}
        loss_labels = self.out_labels.keys()
        for out_label in loss_labels:
            losses[out_label] = F.l1_loss(topol[out_label], 
                                         self.trim_labels_to_predicted_only(labels[out_label]), reduction='mean')
        aggregated_mae = sum(losses.values()) / len(losses)
        return aggregated_mae, losses

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    
    def set_inference_settings(self, **kwargs):
        allowed_keys = {'predict_steps', 'noise_type', 'noise_args', 'plot_decoded'}
        unknown_keys = set(kwargs.keys()) - allowed_keys
        if unknown_keys:
            self.raise_error(f"Unknown keys in inference settings: {unknown_keys}. "
                                f"Allowed keys are: {allowed_keys}")
        if 'predict_steps' in kwargs and kwargs['predict_steps'] < self.sequence_length:
            self.raise_error(f"predict_steps must be not be less than sequence_length "
                             f"({self.sequence_length}) to ensure we have "
                             f"enough warmup steps. Got predict_steps={kwargs['predict_steps']}.")
        
        if 'noise_type':
            if kwargs['noise_type'] is not None and self.likelihood is None:
                self.raise_error("Cannot specify noise_type in inference_settings when no likelihood model is used.")
    
        self.inference_settings.update(kwargs)
            
    def on_predict_start(self):
        self.ce_log_dict("Inference settings", self.inference_settings)
        if 'predict_steps' not in self.inference_settings:
            self.raise_error("predict_steps must be specified in "
                             "inference_settings before starting prediction.")

    @torch.inference_mode()
    def predict_step(self, batch, batch_idx):
        data, labels = self._batch_split(batch)
        if self._encoder_mode:
            with torch.no_grad():
                latent, meta = self.encdec_model._encode(data)
        else:
            latent = self.normalize(latent)

        predict_steps = self.inference_settings['predict_steps']
        inp = latent[:self.input_chunk_length, :].unsqueeze(0) # (1, T_in, Latent Dim)
        
        # Convert the initial input to the output space for the first step of the propagator
        latent_prediction = inp.squeeze(0) # (T_in, Latent Dim)
        if not self._encoder_mode:
            out = self.denormalize(latent_prediction) # (T_in, C)
        else:
            out = self.encdec_model._decode(latent_prediction)[0] # dict of (T_in, C)
        
        n_propagations = (predict_steps - self.input_chunk_length) // self.propagator.output_chunk_length + 1
        pbar = tqdm(total=predict_steps, leave=False, desc="Autoregressive Propagation", ncols=80)
        for _ in range(n_propagations):
            # Forwards pass through the propagator
            latent_prediction_step = self.prop_sample(
                self.propagator((inp, None, None))) # (1, T_out, C)
            
            # Extract inp for the next steps
            inp = torch.cat(
                [inp, latent_prediction_step], dim=1
                )[:, -self.propagator.input_chunk_length:, :] # Keep only the last T_in steps for the next input

            latent_prediction_step = latent_prediction_step.squeeze(0) # (T_out, C)
            # Process the output of this step
            if not self._encoder_mode:
                out_step = self.denormalize(latent_prediction_step) 
            else:
                out_step = self.encdec_model._decode(latent_prediction_step)[0]

            # Aggregate outputs
            if not self._encoder_mode:
                out = torch.cat([out, out_step], dim=0) # (T, C)
            else:
                out = {key: torch.cat(
                    [out[key], out_step[key]], dim=0
                    ) for key in out.keys()}
            latent_prediction = torch.cat(
                [latent_prediction, latent_prediction_step], dim=0) # (T, C)
            pbar.update(latent_prediction_step.size(0))
        pbar.close()
        
        # Trim the output to the requested predict_steps in case we overshoot due to chunking
        if not self._encoder_mode:
            return out[:predict_steps, :]
        else:
            out = {key: out[key][:predict_steps, :] for key in out.keys()}
            latent_prediction = latent_prediction[:predict_steps, :]

        pred = { "Predicted": out }
        pred['Latent'] = latent_prediction
        # If the number of predicted graphs matches the number of input graphs, 
        # we can also include the true labels in the output for comparison.
        if latent.size(0) == out['bond_dist'].size(0): 
            pred['True'] = labels
            pred['Decoded'], _ = self.encdec_model._decode(latent)

        return pred
    
    # ------------------------------------------------------------------
    # Utility functions
    # ------------------------------------------------------------------
    
    def extract_labels(self, batch):
        """extract target labels from a batch.
        """
        num_graphs = batch.batch.max().item() + 1
        
        labels = {}
        for out_label, batch_attr in self.out_labels.items():
            if not hasattr(batch, batch_attr):
                self.raise_error(f"Batch is missing expected attribute "
                                 f"'{batch_attr}' for output label '{out_label}'")
            labels[out_label] = getattr(batch, batch_attr).view(num_graphs, -1) # also keep the original shape for potential use in inference output
        return labels