import os
import math
import argparse
from typing import Callable, Dict, List, Optional, Union

import numpy as np

from ase.data import covalent_radii

import torch
from torch import nn, Tensor
import torch.nn.functional as F

from torch_geometric.data import Dataset
from torch_geometric.nn import MessagePassing, Set2Set
from torch_geometric.utils import softmax

import pytorch_lightning as pl
from tqdm import tqdm

from propagators.tft_model.tft import ModifiedTFTModel as TFTModel

class PropagatorTFT(pl.LightningModule):
    def __init__(
        self,
        encdec_model: nn.Module,
        propagator_args: Dict[str, Union[int, float]],
        likelihood: str = 'QuantileRegression',
        likelihood_args: Optional[Dict] = None,
        lr: Optional[float] = 1e-4,                                   ### OPTIMIZER ARGS
        weight_decay: Optional[float] = 0.0,
        normIn: Optional[bool] = False,
        scheduler: Optional[bool] = False,
        scheduler_args: Optional[Dict] = None,
        out_labels: Optional[List[str]] = ['bond_dist', 'angle', 'dihedral_cos', 'dihedral_sin'],
        outname: Optional[str] = './TFT_untitled/TFT_',
    ):
        self.save_hyperparameters()
        super().__init__()

        self.latent_dim = encdec_model.latent_dim
        self.sequence_length = (
            propagator_args['input_chunk_length'] + 
            propagator_args['output_chunk_length']
        )
        if likelihood == 'QuantileRegression':
            from darts.utils.likelihood_models import QuantileRegression
            quantiles = likelihood_args.get('quantiles', [0.5]) if likelihood_args else [0.5]
            assert all(isinstance(q, float) and 0.0 < q < 1.0 for q in quantiles), "quantiles must be between 0 and 1"
            self.likelihood = QuantileRegression(quantiles=quantiles)
        elif likelihood == 'Gaussian':
            from darts.utils.likelihood_models import GaussianLikelihood
            prior_mu = likelihood_args.get('prior_mu', None) if likelihood_args else None
            prior_sigma = likelihood_args.get('prior_sigma', None) if likelihood_args else None
            prior_strength = likelihood_args.get('prior_strength', 1.0) if likelihood_args else 1.0
            beta_nll = likelihood_args.get('beta_nll', 0.0) if likelihood_args else 0.0
            self.likelihood = GaussianLikelihood(prior_mu=prior_mu, prior_sigma=prior_sigma, prior_strength=prior_strength, beta_nll=beta_nll)
        else:
            raise ValueError(f"Unsupported likelihood: {likelihood}")
        # Initialize propagator
        (
            variables_meta, 
            n_static_components, 
            categorical_embedding_sizes, 
            output_dim
        ) = TFTModel.collect_meta(
            input_chunk_length = propagator_args['input_chunk_length'],
            output_chunk_length = propagator_args['output_chunk_length'],
            n_past_covariates = 0,
            n_future_covariates = 0,
            n_static_covariates = 0,
            n_targets = self.latent_dim,
            add_relative_index = True,
            likelihood = self.likelihood,
        )
        prop_keywargs = {
            "input_chunk_length": propagator_args['input_chunk_length'],
            "output_chunk_length": propagator_args['output_chunk_length'],
            "output_dim": output_dim,
            "variables_meta": variables_meta,
            "num_static_components": n_static_components,
            "hidden_size": propagator_args['hidden_dim'],
            "lstm_layers": propagator_args['lstm_layers'],
            "dropout": propagator_args['dropout'],
            "num_attention_heads": propagator_args['num_attention_heads'],
            "full_attention": False,
            "feed_forward": "GatedResidualNetwork",
            "hidden_continuous_size": 8,
            "categorical_embedding_sizes": categorical_embedding_sizes,
            "add_relative_index": True,
            "norm_type": 'LayerNorm',
        }
        self.propagator = TFTModel(**prop_keywargs)
        self.encdec_model = encdec_model
        self.encdec_model.eval()
        for param in self.encdec_model.parameters():
            param.requires_grad = False  # Freeze encoder-decoder
        
        if scheduler:
            self.scheduler_args = {
                'factor': 0.7, 
                'patience': 5, 
                'min_lr': 1e-9
            }
            if scheduler_args is not None:
                self.scheduler_args.update(scheduler_args)

    def forward(self, data):
        pred, latent = self.encdec_model(data)

        prop_latent = latent.view(-1, self.sequence_length, self.latent_dim)
        prop_in = prop_latent[:, :self.propagator.input_chunk_length, :]
        # TFT expects (B, T, C) inputs; returns (B, T_out, C)
        prop_out = self.propagator((prop_in, None, None))

        prop_sample = self.likelihood.sample(prop_out)
        prop_dec = self.encdec_model.decode(prop_sample.view(-1, self.latent_dim))
        # pred = self.denormalize(pred)
        return pred, latent, prop_out, prop_dec

    def loss_prop(self, prop_out, latent, stage: str, batch_size=None):
        prop_latent = latent.view(-1, self.sequence_length, self.latent_dim)
        prop_target = prop_latent[:, -self.propagator.output_chunk_length:, :]

        loss_prop = self.likelihood.compute_loss(prop_out, prop_target, None)
        self.log(f"{stage}_prop_loss", loss_prop, prog_bar=(stage=="train"), 
                 on_step=(stage=="train"), on_epoch=True, batch_size=batch_size)
        
        with torch.no_grad():
            mae_prop = torch.abs(self.likelihood.sample(prop_out) - prop_target).mean()
            self.log(f"{stage}_prop_mae", mae_prop, prog_bar=(stage!="train"), on_epoch=True, batch_size=batch_size)

        return loss_prop
    
    def step(self, batch, stage: str):
        pred, latent, prop_out, prop_dec = self.forward(batch)

        batch_size = self.trainer.datamodule.hparams.batch_size if self.trainer and self.trainer.datamodule else None
        
        # Propagation loss in latent space
        loss_prop = self.loss_prop(prop_out, latent, stage, batch_size=self.sequence_length)

        loss = loss_prop
        
        self.log(f"{stage}_loss", loss, prog_bar=(stage=="train"), 
                 on_step=(stage=="train"), on_epoch=True, batch_size=batch_size)

        return loss

    def propagate(self, warmup, predict_steps):
        # TFT expects (B, T_in, C) inputs; returns (B, T_out, C)
        prop_out = self.propagator((warmup, None, None))
        prop_out = self.likelihood.sample(prop_out)
        prop_out = torch.cat([warmup, prop_out], dim=1)  # (B, T = T_in + T_out, C)

        pbar = tqdm(total=predict_steps, leave=False, desc="Autoregressive Propagation", ncols=80)
        while prop_out.size(1) < predict_steps:
            inp = prop_out[:, -self.propagator.input_chunk_length:, :]
            prop_extra = self.propagator((inp, None, None))
            prop_extra = self.likelihood.sample(prop_extra)
            prop_out = torch.cat([prop_out, prop_extra], dim=1)
            pbar.update(prop_extra.size(1))
        pbar.close()
        prop_out = prop_out[:, :predict_steps, :].contiguous().squeeze(0)  # (T, C)
        
        return prop_out
    
    def set_predict_steps(self, steps: int):
        assert steps >= self.sequence_length, f"predict_steps must be at least {self.sequence_length}"
        self.predict_steps = steps

    def predict_step(self, batch, batch_idx):
        latent = self.encdec_model.encode(batch)

        if hasattr(self, 'predict_steps'):
            predict_steps = self.predict_steps
        else:
            predict_steps = latent.size(0)

        if predict_steps < self.hparams.propagator_args['input_chunk_length']:
            raise ValueError(f"Not enough input steps for prediction: have {predict_steps}, "
                             f"need at least {self.hparams.propagator_args['input_chunk_length']}")
        warmup = latent[:self.hparams.propagator_args['input_chunk_length'], :].view(1, self.hparams.propagator_args['input_chunk_length'], self.latent_dim)

        prop_out = self.propagate(warmup, predict_steps=predict_steps)

        if predict_steps > 1000:
            # To save memory, only decode 1000 graphs at a time
            gnn_out = {}
            for i in range(0, predict_steps, 1000):
                chunk = prop_out[i:i+1000, :].contiguous()
                chunk_out = self.gnn_dec(chunk)
                for k, v in chunk_out.items():
                    if k not in gnn_out:
                        gnn_out[k] = []
                    gnn_out[k].append(v)
            for k in gnn_out:
                gnn_out[k] = torch.cat(gnn_out[k], dim=0)
        else:
            gnn_out = self.encdec_model.decode(prop_out)

        pred = { "Predicted":{
            'bond_dist': gnn_out['bond_dist'],
            'angle': gnn_out['angle'],
            'dihedral_cos': gnn_out['dihedral_cos'],
            'dihedral_sin': gnn_out['dihedral_sin'],
        },}

        if latent.size(0) == pred['Predicted']['bond_dist'].size(0): # Only when we get more than warmup steps
            pred['True'] = {
                'bond_dist': batch.y_bonds.view(predict_steps, -1),
                'angle': batch.y_angles.view(predict_steps, -1),
                'dihedral_cos': batch.y_torsions_cos.view(predict_steps, -1),
                'dihedral_sin': batch.y_torsions_sin.view(predict_steps, -1),
            }
            latent_dec = self.gnn_dec(latent)
            pred['Decoded'] = {
                'bond_dist': latent_dec['bond_dist'],
                'angle': latent_dec['angle'],
                'dihedral_cos': latent_dec['dihedral_cos'],
                'dihedral_sin': latent_dec['dihedral_sin'],
            }

        return pred

    def training_step(self, batch, batch_idx):
        return self.step(batch, "train")

    def validation_step(self, batch, batch_idx):
        self.step(batch, "val")

    def test_step(self, batch, batch_idx):
        self.step(batch, "test")

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        if self.hparams.scheduler:
            sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, 
                                                            mode='min', 
                                                           factor=self.scheduler_args['factor'], 
                                                           patience=self.scheduler_args['patience'], 
                                                           min_lr=self.scheduler_args['min_lr'])
            return {"optimizer": opt, "lr_scheduler": sched, "monitor": "val_loss"}
        return opt

__all__ = ["BondGraphEncoderTFT"]

