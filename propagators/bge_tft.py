from typing import Callable, Dict, List, Optional, Union

import torch
from torch import nn

from tqdm import tqdm

from likelihoods.resolver import LikelihoodResolver
from propagators.tft_model.tft import ModifiedTFTModel as TFTModel
from collective_encoder.nets.bge import BondGraphNetEncoderDecoder

class BondGraphEncoderTFT(BondGraphNetEncoderDecoder):
    """LightningModule wrapper for BondGraphNet.

    Args:
        datamodule: PyTorch Lightning DataModule with training/validation/test dataloaders.
        encoder_args: Dict of args for BondGraphNetEncoder.
        decoder_args: Dict of args for BondGraphNetDecoder.
        propagator_args: Dict of args for TFTModel.
        prop_likelihood: Darts likelihood model for propagator outputs.
        lrate: Learning rate for AdamW optimizer.
        weight_decay: Weight decay for AdamW optimizer.
        normIn: Whether to normalize input features (using datamodule statistics).
        scheduler: Whether to use a learning rate scheduler (ReduceLROnPlateau).
        loss_weights: Optional list of 4 floats to weight the 4 loss components.
        loss_latent_weight: Weight for latent MSE loss (default 1e-3).
        out_labels: List of 4 strings naming the output components (default ['bond_dist', 'angle', 'dihedral_cos', 'dihedral_sin']).
        outname: Prefix for saving model checkpoints and outputs.
    """
    def __init__(
        self,
        datamodule,
        encoder_args: Dict[str, Union[int, float]],
        decoder_args: Dict[str, Union[int, float]],
        propagator_args: Dict[str, Union[int, float]],
        likelihood: str = 'QuantileRegression',
        likelihood_args: Optional[Dict] = None,
        lrate: Optional[float] = 1e-4,                                   ### OPTIMIZER ARGS
        weight_decay: Optional[float] = 0.0,
        normIn: Optional[bool] = False,
        scheduler: Optional[bool] = False,
        scheduler_args: Optional[Dict] = None,
        loss_fn: Optional[nn.Module] = None,
        loss_encdec_weights: Optional[List[float]] = None,
        loss_prop_weight: Optional[float] = 1.0,
        loss_prop_start: Optional[int] = 0,
        loss_rec_weight: Optional[float] = 1.0,
        loss_e2e_weight: Optional[float] = 1.0,
        out_labels: Optional[List[str]] = ['bond_dist', 'angle', 'dihedral_cos', 'dihedral_sin'],
        outname: Optional[str] = './BGETFT_untitled/BGETFT_',
    ):
        self.save_hyperparameters(ignore=["datamodule"])
        super().__init__(
            datamodule=datamodule,
            encoder_args=encoder_args,
            decoder_args=decoder_args,
            lrate=lrate,
            weight_decay=weight_decay,
            normIn=normIn,
            scheduler=scheduler,
            scheduler_args=scheduler_args,
            loss_fn=loss_fn,
            loss_weights=loss_encdec_weights,
            out_labels=out_labels,
            outname=outname,
        )

        self.sequence_length = (
            propagator_args['input_chunk_length'] + 
            propagator_args['output_chunk_length']
        )
        self.likelihood = LikelihoodResolver(likelihood, likelihood_args or {})
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
            "retain_lstm_cell_state": propagator_args.get('retain_lstm_cell_state', False),
        }
        self.propagator = TFTModel(**prop_keywargs)

    def forward(self, data):
        data = self.normalize(data) # Normalize input features
        latent = self.gnn_enc(data) # Encode to latent space
        pred = self.gnn_dec(latent) # Decode to output predictions
        
        long_seq_length = self.trainer.datamodule.sequence_length # This is the length of one input sequence
        prop_latent = latent.view(-1, long_seq_length, self.latent_dim) # (Batch, Sequence, Latent Dim)
        
        # We want to create the whole prediction of whole input sequence
        # self.sequence_length is (input_chunk_length + output_chunk_length)
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
        
        prop_sample = self.likelihood.sample(prop_out)
        prop_dec = self.gnn_dec(prop_sample.view(-1, self.latent_dim))
        # pred = self.denormalize(pred)
        return pred, latent, prop_out, prop_dec

    def loss_prop(self, prop_out, latent, stage: str, batch_size=None):
        long_seq_length = self.trainer.datamodule.sequence_length
        
        latent = latent.view(-1, long_seq_length, self.latent_dim)
        latent = latent[:, self.propagator.input_chunk_length:, :]

        loss_prop = self.likelihood.compute_loss(prop_out, latent, None)
        self.log(f"{stage}_prop_loss", loss_prop, prog_bar=(stage=="train"), 
                 on_step=(stage=="train"), on_epoch=True, batch_size=batch_size)
        
        with torch.no_grad():
            mae_prop = torch.abs(self.likelihood.sample(prop_out) - latent).mean()
            self.log(f"{stage}_prop_mae", mae_prop, prog_bar=(stage!="train"), on_epoch=True, batch_size=batch_size)

        return loss_prop

    def loss_e2e(self, prop_dec, labels, stage: str, batch_size=None):
        long_seq_length = self.trainer.datamodule.sequence_length
        losses = {}
        mae = {}
        for out_label, weight in zip(self.hparams.out_labels, 
                                     self.hparams.loss_encdec_weights):
            label = labels[out_label]
            label = label.view(-1, long_seq_length, label.shape[-1])
            label = label[:, self.propagator.input_chunk_length:, :]
            label = label.contiguous().view(-1, label.shape[-1])
            losses[out_label] = self.loss_fn(prop_dec[out_label], label) * weight
            
            self.log(f"{stage}_e2e_{out_label}_loss", 
                     losses[out_label], prog_bar=False, 
                     on_epoch=True, batch_size=batch_size)
            
            with torch.no_grad():
                mae[out_label] = (torch.abs(prop_dec[out_label] - label).mean() 
                                    if label.numel() > 0 
                                    else torch.tensor(0.0, 
                                            device=prop_dec[out_label].device))
                self.log(f"{stage}_e2e_{out_label}_mae", 
                         mae[out_label], prog_bar=False, 
                         on_epoch=True, batch_size=batch_size)
        
        self.log(f"{stage}_e2e_loss", sum(losses.values()), 
                 prog_bar=(stage=="train"), on_step=(stage=="train"), 
                 on_epoch=True, batch_size=batch_size)

        with torch.no_grad():
            self.log(f"{stage}_e2e_mae", sum(mae.values()) / len(mae), 
                     prog_bar=(stage!="train"), on_epoch=True, batch_size=batch_size)

        return sum(losses.values())
    
    def step(self, batch, stage: str):
        pred, latent, prop_out, prop_dec = self.forward(batch)
        labels = self.extract_labels(batch)

        batch_size = self.trainer.datamodule.hparams.batch_size if self.trainer and self.trainer.datamodule else None

        # Reconstruction loss of encoder-decoder
        if self.hparams.loss_rec_weight > 0.0:
            loss_encdec = self.loss_encdec(pred, labels, stage, batch_size=batch_size)
        else:
            loss_encdec = torch.tensor(0.0, device=latent.device)
        
        # Propagation loss in latent space
        if self.hparams.loss_prop_weight > 0.0 and \
           (self.hparams.loss_prop_start < self.trainer.current_epoch):
            loss_prop = self.loss_prop(prop_out, latent, stage, batch_size=self.sequence_length)
        else:
            loss_prop = torch.tensor(0.0, device=latent.device)

        # End-to-end loss of propagated decoded structures
        if self.hparams.loss_e2e_weight > 0.0 and \
           (self.hparams.loss_prop_start < self.trainer.current_epoch):
            loss_e2e = self.loss_e2e(prop_dec, labels, stage, batch_size=batch_size)
        else:
            loss_e2e = torch.tensor(0.0, device=latent.device)

        loss = (self.hparams.loss_rec_weight * loss_encdec 
                + self.hparams.loss_prop_weight * loss_prop
                + self.hparams.loss_e2e_weight * loss_e2e)
        
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
            prop_extra = self.likelihood.sample(prop_extra, temperature=self.sampling_temperature)
            if False: # Add gaussian noise proportional to input variance
                inp_var = torch.var(inp, dim=(1))
                noise = torch.randn_like(prop_extra) * torch.sqrt(inp_var + 1e-6)
                prop_extra = prop_extra + noise
            prop_out = torch.cat([prop_out, prop_extra], dim=1)
            pbar.update(prop_extra.size(1))
        pbar.close()
        prop_out = prop_out[:, :predict_steps, :].contiguous().squeeze(0)  # (T, C)
        
        return prop_out
    
    def set_predict_settings(self, **kwargs):
        steps = kwargs.get('predict_steps', None)
        temperature = kwargs.get('sampling_temperature', 1.0)
        
        assert steps >= self.sequence_length, f"predict_steps must be at least {self.sequence_length}"
        self.predict_steps = steps
        self.sampling_temperature = temperature
    
    def on_predict_start(self):
        print('\n')
        print(''.join(['=']*16))
        print(f"Prediction settings:")
        print(f"  predict_steps: {getattr(self, 'predict_steps', 'Not set')}")
        print(f"  sampling_temperature: {getattr(self, 'sampling_temperature', 1.0)}")
        print(''.join(['=']*16))

    def predict_step(self, batch, batch_idx):
        data = self.normalize(batch)
        latent = self.gnn_enc(data)

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
            gnn_out = self.gnn_dec(prop_out)

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

__all__ = ["BondGraphEncoderTFT"]

