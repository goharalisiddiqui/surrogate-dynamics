import argparse
from darts.models import TFTModel
from darts import TimeSeries
from darts.logging import raise_if_not
from darts.utils.likelihood_models import QuantileRegression
from torch.nn.functional import pairwise_distance

from nets.edvae_net import EDVAE as enc_model

import torch
from typing import Optional


def tft_args():
    desc = "TFT model Arguments"
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('--seqlength', dest="input_chunk_length", default=5,
                        type=int, help='Number of time steps in the past to take as a model input')
    parser.add_argument('--predlength', dest="output_chunk_length",
                        default=5, type=int, help='Number of time steps to predict at once')
    # parser.add_argument('--nhidden', dest="hidden_size", default=[96], nargs='+', help='Hidden state size of the TFT')
    parser.add_argument('--nhidden', dest="hidden_size",
                        default=96, type=int, help='Hidden state size of the TFT')
    parser.add_argument('--nlstm', dest="lstm_layers", default=2, type=int,
                        help='Number of layers for the LSTM Encoder and Decoder')
    parser.add_argument('--nattention', dest="num_attention_heads",
                        default=1, type=int, help='Number of attention heads')
    parser.add_argument('--dropout', dest="dropout", default=0.02,
                        type=float, help='Fraction of neurons affected by dropout.')
    parser.add_argument('--batch_size', dest="batch_size", default=100, type=int,
                        help='Number of time series (input and output sequences) used in each training pass.')

    args, _ = parser.parse_known_args()

    return args


TFT_args = tft_args


class ModifiedQuantileRegression(QuantileRegression):
    def __init__(self, quantiles: Optional[list[float]] = None, enc_ckpt: Optional[str] = None, elems: Optional[list[str]] = None):
        """
        The "likelihood" corresponding to quantile regression modified to add additional loss
        It uses the Quantile Loss Metric for custom quantiles centered around q=0.5.

        This class can be used as any other Likelihood objects even though it is not
        representing the likelihood of a well defined distribution.

        Parameters
        ----------
        quantiles
            list of quantiles
        """
        if enc_ckpt is not None and elems is None:
            raise ValueError("elems must be provided if enc_ckpt is provided")

        super().__init__(quantiles)
        self.encoder = None
        self.enc_ckpt = enc_ckpt
        self.elems = elems

    def load_encoder(self, enc_ckpt: str):
        self.encoder = enc_model.load_from_checkpoint(enc_ckpt)
        self.encoder.eval()

    def compute_loss(
        self,
        model_output: torch.Tensor,
        target: torch.Tensor,
        sample_weight: torch.Tensor,
    ):
        """
        We are re-defining a custom loss (which is not a likelihood loss) compared to Likelihood

        Parameters
        ----------
        model_output
            must be of shape (batch_size, n_timesteps, n_target_variables, n_quantiles)
        target
            must be of shape (n_samples, n_timesteps, n_target_variables)
        sample_weight
            must be of shape (n_samples, n_timesteps, n_target_variables)
        """
        if self.encoder is None and self.enc_ckpt is not None:
            # Dirty fix because log_hyperparameters of PL fails if we do this in __init__
            self.load_encoder(self.enc_ckpt)

        dim_q = 3
        device = model_output.device

        # test if torch model forward produces correct output and store quantiles tensor
        if self.first:
            raise_if_not(
                len(model_output.shape) == 4
                and len(target.shape) == 3
                and model_output.shape[:2] == target.shape[:2],
                "mismatch between predicted and target shape",
            )
            raise_if_not(
                model_output.shape[dim_q] == len(self.quantiles),
                "mismatch between number of predicted quantiles and target quantiles",
            )
            self.quantiles_tensor = torch.tensor(self.quantiles).to(device)
            self.first = False

        errors = target.unsqueeze(-1) - model_output
        losses = torch.max(
            (self.quantiles_tensor - 1) * errors, self.quantiles_tensor * errors
        ).sum(dim=dim_q)

        if sample_weight is not None:
            losses = losses * sample_weight

        if self.encoder is not None:
            enc_latent = torch.max((self.quantiles_tensor - 1) * model_output,
                                   self.quantiles_tensor * model_output).sum(dim=dim_q)
            x = self.encoder.decode_latent(
                enc_latent.view(-1, enc_latent.shape[-1]), keeptensor=True)
            x = x[0]
            print("X: ", x.shape)
            print(f"elems: {self.elems}")
            # calculate euclidean distance between the atoms in x
            x = x.reshape(-1, 3)
            x = x.unsqueeze(0)
            x = x.repeat(x.shape[0], 1, 1)
            x_t = x.transpose(0, 1)
            diff = x.unsqueeze(0) - x_t.unsqueeze(1)
            diff = diff.view(-1, 3)
            diff = diff ** 2
            diff = diff.sum(-1)
            diff = diff.sqrt()
            # diff = diff.view(x.shape[0], x.shape[0])
            print("Diff: ", diff.shape)
            exit()
            diff = diff + torch.eye(diff.shape[0]).to(device)
            # print("Diff: ", diff)
        # print("Losses: ", losses.shape)
        # exit()

        return losses.mean()
