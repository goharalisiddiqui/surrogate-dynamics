import os
import argparse
from random import shuffle
from typing import List, Dict
from tqdm import tqdm
import warnings

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler

import MDAnalysis as mda
from MDAnalysis.analysis import align
from MDAnalysis.analysis import dihedrals
from MDAnalysis.analysis.rms import rmsd
import MDAnalysis.transformations as trans

import torch

from darts import TimeSeries

from collective_encoder.dataloaders.xtc_dataloader import XtcDataset

warnings.filterwarnings("ignore")


def xtctdatset_args():
    desc = "Xtc Dataset Arguments"
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('--xtcfile', required=True, type=str,
                        help='Input compressed coordinate file')
    parser.add_argument('--tprfile', required=True, type=str,
                        help='Input binary file containing the topology')
    parser.add_argument('--selection', required=True, type=str,
                        help='Selection string of mdanalysis')

    parser.add_argument('--dataset_type', type=str, default='GRAPH',
                        help='Type of dataset to use',
                        choices=['DISTANCES', 'GRAPH'])
    parser.add_argument('--dataset_args', metavar="KEY=VALUE", nargs='+', 
                        help='Key-value pairs of arguments for the dataset', 
                        default=[])
    parser.add_argument('--encoder_model', type=str, default='BondGraphEncoder',
                        help='Type of encoder model to use', 
                        choices=['BondGraphEncoder'])
    parser.add_argument('--encoder_model_path', required=True, type=str,
                        help='Path to the trained encoder model checkpoint')
    parser.add_argument('--norm_latent', action='store_true',
                        help='Normalize the latent space to -1, 1')

    parser.add_argument('--series_length', default=0, type=int,
                        help='Length of a single Time series')
    parser.add_argument('--stride', default=None, type=int,
                        help='Stride to use when creating the time series. If None, use series_length')
    parser.add_argument('--train_size', default=0, type=int,
                        help='Number of time series in the training set')
    parser.add_argument('--val_size', default=0, type=int,
                        help='Number of time series in the validation set')

    args, _ = parser.parse_known_args()

    return args


XTCT_args = xtctdatset_args


class XtcTrainer(XtcDataset):
    def __init__(self,
                 xtcfile: str,
                 tprfile: str,
                 selection : str,
                 series_length: int,
                 stride: int,
                 train_size: int,
                 val_size: int,
                 dataset_type: str = 'GRAPH',
                 dataset_args: Dict[str, str] = None,
                 encoder_model: str = 'BondGraphEncoder',
                 encoder_model_path: str = None,
                 verbose: bool = True,
                 norm_latent: bool = False,
                 ):
        # Sanity checks
        assert os.path.isfile(xtcfile), f"XTC file not found: {xtcfile}"
        assert os.path.isfile(tprfile), f"TPR file not found: {tprfile}"
        assert series_length > 1, f"Series length must be > 1, got {series_length}"
        assert train_size >= 0, f"Train size must be > 0, got {train_size}"
        assert val_size >= 0, f"Validation size must be >= 0, got {val_size}"
        assert os.path.isfile(encoder_model_path), f"Encoder model file not found: {encoder_model_path}"

        # Save parameters
        if stride is None:
            stride = series_length
        self.verbose = verbose
        self.dataset_type = dataset_type
        self.dataset_args = dataset_args
        self.series_length = series_length
        self.stride = stride
        self.train_size = train_size
        self.val_size = val_size
        self.encoder_model_path = encoder_model_path

        # Import Model class
        if encoder_model == 'BondGraphEncoder':
            from collective_encoder.nets.bge import BondGraphNetEncoderDecoder as EncoderClass
        else:
            raise ValueError(f"Unknown encoder model: {encoder_model}")
        
        # Initialize parent class dataset
        dataset_size = series_length + (stride * (train_size + val_size - 1))
        super().__init__(xtcfile=xtcfile,
                         tprfile=tprfile,
                         selection=selection,
                         dataset_size=dataset_size,
                         sequential=True,
                         dataset_type=dataset_type,
                         verbose=verbose,
                         )
        
        # Load Encoder model
        print(f"Loading encoder model from {encoder_model_path}") if verbose else None
        self.encoder = EncoderClass.load_from_checkpoint(encoder_model_path, 
                                                    datamodule=self)
        self.encoder.eval()
        if self.encoder.Mean.detach().numpy().flatten().sum() != 0: # This is Fix for old models where the flags were not registered as buffers
            self.encoder.normIn = True
            self.encoder.normSet = True
        
        # Encode full dataset
        print("Encoding full dataset...") if verbose else None
        full_batch = self.get_full_batch()
        with torch.no_grad():
            latent = self.encoder.get_latent(full_batch)
            decoded = self.encoder.get_decoded(latent)
        self.latent = latent
        self.decoded = decoded
        self.latent_predicted = None
        self.warmup_steps = None

        # Scale latent to -1, 1
        if norm_latent:
            self.scaler = MinMaxScaler(feature_range=(-1, 1))
            self.scaler.fit(self.latent.numpy())
            self.latent = torch.tensor(self.scaler.transform(self.latent.numpy()))

        # Create TimeSeries objects
        series_list = []
        for i in range(train_size + val_size):
            start = i * stride
            end = start + series_length
            series = TimeSeries.from_values(self.latent[start:end].numpy())
            series_list.append(series)
        # Randomize the series list
        shuffle(series_list)
        self.train_series = series_list[:train_size]
        self.val_series = series_list[train_size:]

        print(latent.shape)
        # Print 2d tsne embedding of the latent space
        try:
            from sklearn.manifold import TSNE
            tsne = TSNE(n_components=2, random_state=0)
            latent_2d = tsne.fit_transform(self.latent.numpy())
            plt.figure(figsize=(6,6))
            plt.scatter(latent_2d[:,0], latent_2d[:,1], s=1, c=range(latent_2d.shape[0]), cmap='viridis')
            plt.title("t-SNE embedding of the latent space")
            plt.xlabel("t-SNE 1")
            plt.ylabel("t-SNE 2")
            plt.savefig("latent_tsne.png", dpi=300)
            plt.close()
        except Exception as e:
            print(f"Could not compute t-SNE embedding: {e}")
        exit()


        print(f"Created {len(self.train_series)} training series and {len(self.val_series)} validation series of length {series_length}.") if verbose else None
    
    def get_train_series(self) -> List[TimeSeries]:
        return self.train_series
    def get_val_series(self) -> List[TimeSeries]:
        return self.val_series

    def get_warmup_steps(self, nsteps: int) -> torch.Tensor:
        """Get the warmup steps from the start of the dataset"""
        assert nsteps > 0, f"Number of warmup steps must be > 0, got {nsteps}"

        if self.latent_predicted is None:
            warmup = self.latent[:nsteps]
        else:
            assert self.latent_predicted is not None, "No predicted steps available for warmup"
            assert self.latent_predicted.shape[0] >= nsteps, f"Not enough predicted steps for warmup: {self.latent_predicted.shape[0]} < {nsteps}"
            warmup = self.latent_predicted[-nsteps:]

        self.add_predicted(warmup)
        return warmup
        
    def add_predicted(self, predicted: torch.Tensor):
        """Add predicted steps to the dataset"""
        if self.latent_predicted is None:
            self.latent_predicted = predicted
        else:
            self.latent_predicted = torch.cat((self.latent_predicted, predicted), dim=0)

    def output_results(self, output_stem: str):
        assert self.latent_predicted is not None, "No predicted steps to output"
        pred = torch.cat((self.warmup_steps, self.latent_predicted), dim=0) \
            if self.warmup_steps is not None else self.latent_predicted
        len_result = pred.shape[0]


        decoded_orig_cos = self.decoded['dihedral_cos'][:len_result].cpu().numpy()
        decoded_orig_sin = self.decoded['dihedral_sin'][:len_result].cpu().numpy()
        with torch.no_grad():
            if self.hparams.norm_latent:
                decoded_pred = self.scaler.inverse_transform(pred.numpy())
                decoded_pred = torch.tensor(decoded_pred)
            else:
                decoded_pred = pred
            decoded_pred = self.encoder.get_decoded(decoded_pred)
        decoded_pred_cos = decoded_pred['dihedral_cos'].cpu().numpy()
        decoded_pred_sin = decoded_pred['dihedral_sin'].cpu().numpy()

        # _, __ , torsion_index = self.get_dataset().get_label_indices()
        # print(f"Torsion indices: {torsion_index}")
        # exit()
        idx_phi = 6 # [1,3,4,5]
        idx_psi = 10 # [3,4,6,8]


        phi = np.arctan2(decoded_orig_sin[:,idx_phi], decoded_orig_cos[:,idx_phi])
        psi = np.arctan2(decoded_orig_sin[:,idx_psi], decoded_orig_cos[:,idx_psi])

        phi_pred = np.arctan2(decoded_pred_sin[:,idx_phi], decoded_pred_cos[:,idx_phi])
        psi_pred = np.arctan2(decoded_pred_sin[:,idx_psi], decoded_pred_cos[:,idx_psi])

        import matplotlib.pyplot as plt
        plt.figure(figsize=(10,5))
        plt.subplot(1,2,1)
        plt.title("Original")
        plt.scatter(phi, psi, s=1)
        plt.xlabel("Phi")
        plt.ylabel("Psi")
        plt.xlim([-np.pi, np.pi])
        plt.ylim([-np.pi, np.pi])
        plt.xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi],
                   [r'$-\pi$', r'$-\pi/2$', '0', r'$\pi/2$', r'$\pi$'])
        plt.yticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi],
                   [r'$-\pi$', r'$-\pi/2$', '0', r'$\pi/2$', r'$\pi$'])
        plt.subplot(1,2,2)
        plt.title("Predicted")
        plt.scatter(phi_pred, psi_pred, s=1)
        plt.xlabel("Phi")
        plt.ylabel("Psi")
        plt.xlim([-np.pi, np.pi])
        plt.ylim([-np.pi, np.pi])
        plt.xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi],
                   [r'$-\pi$', r'$-\pi/2$', '0', r'$\pi/2$', r'$\pi$'])
        plt.yticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi],
                   [r'$-\pi$', r'$-\pi/2$', '0', r'$\pi/2$', r'$\pi$'])
        plt.savefig(output_stem + "ramachandran.png", dpi=300)
        plt.close()