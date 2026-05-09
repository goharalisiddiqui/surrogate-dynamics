from typing import Dict, List, Optional, Tuple, Union
from tqdm import tqdm

import numpy as np

import torch
from torch import nn, var
import torch.nn.functional as F

from collective_encoder.collective_encoder import trainer
from collective_encoder.nets.base import CENetBase
from likelihoods.resolver import likelihood_resolver
from propagators.tft_model.tft import ModifiedTFTModel as TFTModel

from embeddings.resolver import get_encdec

_DEFAULT_OUTPUTS = {
    'bond_dist': 'y_bonds', 
    'angle': 'y_angles', 
    'dihedral_cos': 'y_torsions_cos', 
    'dihedral_sin': 'y_torsions_sin',
}


    

class PropagatorTFT(CENetBase):
    _IDENTFIER = "PropagatorTFT"
    _REQUIRED_ARGS = ['input_chunk_length', 'output_chunk_length']
    _OPTIONAL_ARGS = CENetBase._OPTIONAL_ARGS.copy()
    _OPTIONAL_ARGS.update({
        'hidden_dim': 64,
        'lstm_layers': 1,
        'num_attention_heads': 4,
        'dropout': 0.1,
        'retain_lstm_cell_state': False,
        'likelihood': None,
        'likelihood_args': None,
        'loss_fn': nn.MSELoss(),
        'loss_e2e_weight': 1.0,
        'loss_encdec_weights': [1.0, 1.0, 1.0, 1.0],
        'out_labels': _DEFAULT_OUTPUTS.copy(),
        'inference_settings': None,
        'encdec_model': None,
        'encdec_type': None,
        'encdec_ckpt': None,
    })

    def __init__(
        self,
        datamodule,
        args = None,
        **kwargs
    ):
        self.save_hyperparameters(ignore=['datamodule'])
        super().__init__(args, **kwargs)
        
        self.sequence_length = (
            self.input_chunk_length + 
            self.output_chunk_length
        )

        if self.likelihood is not None:
            self.likelihood = likelihood_resolver(self.likelihood, 
                                                self.likelihood_args or {})
        
        if self.inference_settings is None:
             self.inference_settings = {}
        self.inference_settings.setdefault('sampling_temperature', 1.0)
        self.inference_settings.setdefault('fixed_sigma', False)
        
        self._init_encdec(datamodule)
        self._init_propagator()
        
        self.losses = {
            "loss_prop": self.loss_prop,
        }
        self.metrics = {
            'mae_prop': self.metric_mae_prop
        }
        if self._encoder_mode and self.loss_e2e_weight > 0.0:
            self.losses["loss_topol"] = self.loss_topol
            self.metrics['mae_topol'] = self.metric_mae_topol
            if len(self.loss_encdec_weights) != len(self.out_labels):
                self.raise_error(f"loss_encdec_weights length "
                                f"{len(self.loss_encdec_weights)} "
                                f"must match out_labels length "
                                f"{len(self.out_labels)}")
            
        self.test_metrics = self.metrics.copy()
    
    def _init_encdec(self, datamodule):
        if self.encdec_model is not None:
            if self.encdec_type is not None or self.encdec_ckpt is not None:
                self.raise_error("If encdec is provided directly, "
                                 "encdec_type and encdec_ckpt should not be provided.")
            self._encoder_mode = True
            self.check_and_solve_backwards_compatibility(datamodule) # In case we are loading a ckpt for older version.
            self.encdec_model.eval()
            for param in self.encdec_model.parameters():
                param.requires_grad = False
            self.latent_dim = self.encdec_model.latent_dim
            self.log_msg("Using provided encoder-decoder model.")
        elif self.encdec_type is not None:
            self._encoder_mode = True
            if self.normIn:
                self.raise_error("Normalization not supported when encoder-decoder model is used. "
                                 "Please set normIn to False.")
            encdec_type = self.encdec_type
            encdec_ckpt = self.encdec_ckpt
            if encdec_ckpt is None:
                self.raise_error("Both encdec_ckpt must be provided to use an encoder-decoder model.")
            encdec_cls = get_encdec(encdec_type)
            encdec = encdec_cls.load_from_checkpoint(encdec_ckpt, 
                                    datamodule=datamodule)
            self.encdec_model = encdec
            self.encdec_model.eval()
            for param in self.encdec_model.parameters():
                param.requires_grad = False
            self.latent_dim = self.encdec_model.latent_dim
        else:
            self._encoder_mode = False
            datapoint_shape = datamodule.get_datapoint_shape()
            if not isinstance(datapoint_shape, (tuple, list)):
                self.raise_error(f"Expected datamodule.get_datapoint_shape() "
                                 f"to return a tuple or list, got {type(datapoint_shape)} "
                                 f"this suggest a mismatch between the datamodule and the propagator."
                                )
            self.latent_dim = datapoint_shape[-1]
            self.log_msg("No encoder-decoder model provided. "
                         "Propagator will expect latent inputs directly.")

    def _init_propagator(self):
        (
            variables_meta, 
            n_static_components, 
            categorical_embedding_sizes, 
            output_dim
        ) = TFTModel.collect_meta(
            input_chunk_length = self.input_chunk_length,
            output_chunk_length = self.output_chunk_length,
            n_past_covariates = 0,
            n_future_covariates = 0,
            n_static_covariates = 0,
            n_targets = self.latent_dim,
            add_relative_index = True,
            likelihood = self.likelihood,
        )

        prop_keywargs = {
            "input_chunk_length": self.input_chunk_length,
            "output_chunk_length": self.output_chunk_length,
            "output_dim": output_dim,
            "variables_meta": variables_meta,
            "num_static_components": n_static_components,
            "hidden_size": self.hidden_dim,
            "lstm_layers": self.lstm_layers,
            "dropout": self.dropout,
            "num_attention_heads": self.num_attention_heads,
            "full_attention": False,
            "feed_forward": "GatedResidualNetwork",
            "hidden_continuous_size": 8,
            "categorical_embedding_sizes": categorical_embedding_sizes,
            "add_relative_index": True,
            "norm_type": 'LayerNorm',
            "retain_lstm_cell_state": self.retain_lstm_cell_state,
        }
        self.propagator = TFTModel(**prop_keywargs)
                
    
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
        return self.network[0]

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

    def _batch_split(self, batch):
        if self._encoder_mode:
            return batch, self.extract_labels(batch)
        else:
            return super()._batch_split(batch)
    
    def prop_sample(self, prop_out):
        extra_args = {}
        if not self.training:
            extra_args['temperature'] = self.inference_settings['sampling_temperature']
            extra_args['fixed_sigma'] = self.inference_settings['fixed_sigma']
            
        if self.likelihood is not None:
            return self.likelihood.sample(prop_out, **extra_args)
        else:
            prop_out = prop_out.squeeze(-1)
            if self.training:
                return prop_out
            else:
                return torch.normal(prop_out, var)
        
    def forward(self, data):
        if self._encoder_mode:
            with torch.no_grad():
                latent, meta = self.encdec_model._encode(data)
        else:
            latent = self.normalize(latent)
        
        long_seq_length = self.trainer.datamodule.sequence_length
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
            meta['prop_dec'] = self.encdec_model._decode(prop_sample)
            prop_sample = self.denormalize(prop_sample)
        meta['prop_out'] = prop_out

        return prop_sample, latent, meta
    
    # ------------------------------------------------------------------
    # Losses
    # ------------------------------------------------------------------
                    
    def loss_prop(self, inp, latent, output, labels, meta):
        long_seq_length = self.trainer.datamodule.sequence_length
        
        latent = latent.view(-1, long_seq_length, self.latent_dim)
        latent = latent[:, self.propagator.input_chunk_length:, :]
        prop_out = meta['prop_out'].squeeze(-1)

        loss_prop = self.likelihood.compute_loss(prop_out, latent, None) if \
            self.likelihood is not None else \
                self.loss_fn(prop_out, latent)

        return loss_prop, {}
    
    def loss_topol(self, inp, latent, output, labels, meta):
        topol = meta['prop_dec']

        losses = {}
        loss_labels = self.out_labels.keys()
        for out_label, weight in zip(loss_labels, self.loss_encdec_weights):
            losses[out_label] = self.loss_fn(topol[out_label], labels[out_label]) * weight

        return sum(losses.values()), losses

    def metric_mae_prop(self, inp, latent, output, labels, meta) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        long_seq_length = self.trainer.datamodule.sequence_length
        
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
            losses[out_label] = F.l1_loss(topol[out_label], labels[out_label], reduction='none')
        aggregated_mae = sum(losses.values()) / len(losses)
        return aggregated_mae, losses

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    
    def set_inference_settings(self, **kwargs):
        if 'predict_steps' in kwargs and kwargs['predict_steps'] < self.sequence_length:
            self.raise_error(f"predict_steps must be not be less than sequence_length "
                             f"({self.sequence_length}) to ensure we have "
                             f"enough warmup steps. Got predict_steps={kwargs['predict_steps']}.")
        
        if 'fixed_sigma' in kwargs and kwargs['fixed_sigma'] == False and self.likelihood is None:
            self.raise_error("fixed_sigma cannot be False if no likelihood is used.")
    
        self.inference_settings.update(kwargs)
            
    def on_predict_start(self):
        self.ce_log_dict("Inference settings", self.inference_settings)
        if 'predict_steps' not in self.inference_settings:
            self.raise_error("predict_steps must be specified in "
                             "inference_settings before starting prediction.")
    
    def propagate(self, warmup, predict_steps):
        # TFT expects (B, T_in, C) inputs; returns (B, T_out, C)
        prop_out = self.propagator((warmup, None, None))
        prop_out = self.prop_sample(prop_out)
        prop_out = torch.cat([warmup, prop_out], dim=1)  # (B, T = T_in + T_out, C)

        pbar = tqdm(total=predict_steps, leave=False, desc="Autoregressive Propagation", ncols=80)
        while prop_out.size(1) < predict_steps:
            inp = prop_out[:, -self.propagator.input_chunk_length:, :]
            prop_extra = self.propagator((inp, None, None))
            prop_extra = self.prop_sample(prop_extra)
            prop_out = torch.cat([prop_out, prop_extra], dim=1)
            pbar.update(prop_extra.size(1))
        pbar.close()
        prop_out = prop_out[:, :predict_steps, :].contiguous().squeeze(0)  # (T, C)
        
        return prop_out

    def predict_step(self, batch, batch_idx):
        data, labels = self._batch_split(batch)
        if self._encoder_mode:
            with torch.no_grad():
                latent, meta = self.encdec_model._encode(data)
        else:
            latent = self.normalize(latent)

        predict_steps = self.inference_settings['predict_steps']
        warmup = latent[:self.input_chunk_length, :].unsqueeze(0) # (1, T_in, Latent Dim)
        prop_out = self.propagate(warmup, predict_steps=predict_steps)

        if not self._encoder_mode:
            return self.denormalize(prop_out)

        # if predict_steps > 1000:
        #     # To save memory, only decode 1000 graphs at a time
        #     gnn_out = {}
        #     for i in range(0, predict_steps, 1000):
        #         chunk = prop_out[i:i+1000, :].contiguous()
        #         chunk_out = self.encdec_model.decode(chunk)
        #         for k, v in chunk_out.items():
        #             if k not in gnn_out:
        #                 gnn_out[k] = []
        #             gnn_out[k].append(v)
        #     for k in gnn_out:
        #         gnn_out[k] = torch.cat(gnn_out[k], dim=0)
        # else:

        gnn_out, _ = self.encdec_model._decode(prop_out)
        pred = { "Predicted": gnn_out }
        pred['Latent'] = prop_out

        if latent.size(0) == gnn_out['bond_dist'].size(0): # If the number of predicted graphs matches the number of input graphs, we can also include the true labels in the output for comparison.
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
            labels[out_label] = getattr(batch, batch_attr).view(num_graphs, -1)
        return labels
    
    # ------------------------------------------------------------------
    # Backwards compatibility handling
    # ------------------------------------------------------------------
        
    def check_and_solve_backwards_compatibility(self, datamodule):
        hprams = self.encdec_model.hparams
        if 'args' in hprams:
            self.log_info("Checkpoint contains 'args' key in hyperparameters. "
                        "This suggests its from the new version. ")

        # Check for type of the encdec model to determine how to handle backwards compatibility
        # We only know the BGE type from the old version
        encdec_type = self.encdec_model.__class__.__name__
        if encdec_type != "BondGraphEncoderDecoder":
            self.raise_error(f"Unsupported encoder-decoder model type: {encdec_type}. "
                            f"Expected 'BondGraphEncoderDecoder'.")

        # Create instance with the new code
        encdec_type = 'BGE'
        encdec_cls = get_encdec(encdec_type)
        
        args = {}
        for key in [
            'encoder_args', 
            'decoder_args',
            'normIn',
        ]:
            if key in hprams:
                self.log_info(f"Found matching hyperparameter for encoder-decoder: {key}")
                args[key] = hprams[key]
            else:
                self.raise_error(f"Missing expected hyperparameter '{key}' for encoder-decoder model.")
        
        encdec_model = encdec_cls(args=args, 
                                    datamodule=datamodule,
                                    **self.run_args)
        if hprams['normIn'] == True:
            encdec_model.set_norm()
        
        compatibility_map = {
            'gnn_enc': 'encoder_net',
            'gnn_dec': 'decoder_net',
        }
        state_dict = self.encdec_model.state_dict()
        encdec_state_dict = {}
        for key in state_dict:
            new_key = key
            for old_substr, new_substr in compatibility_map.items():
                if old_substr in new_key:
                    new_key = new_key.replace(old_substr, new_substr)
            encdec_state_dict[new_key] = state_dict[key]
        encdec_model.load_state_dict(encdec_state_dict, strict=False)
        self.log_info("Loaded encoder-decoder model state dict with backwards compatibility handling.")
        self.encdec_model = encdec_model