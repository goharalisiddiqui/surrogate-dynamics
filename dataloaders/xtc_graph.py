import os
import warnings
from typing import List, Dict
from tqdm import tqdm
import random

import numpy as np
import ase
from ase.data import atomic_numbers

import MDAnalysis as mda
from MDAnalysis.analysis import align
from MDAnalysis.analysis.rms import rmsd
from MDAnalysis.lib.distances import calc_dihedrals
import MDAnalysis.transformations as trans

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import torch
from torch.utils.data import Dataset, DataLoader
from torch_geometric.loader import DataLoader as GeoDataLoader
import pytorch_lightning as pl

import torch

import pytorch_lightning as pl

from collective_encoder.dataloaders.xtc_dataloader import XtcData

warnings.filterwarnings("ignore")

class XtcSequence(pl.LightningDataModule):
    def __init__(self,
                 xtcfile: str,
                 tprfile: str,
                 selection : str,
                 sequence_length: int = 0,
                 train_size: int = 0,
                 batch_size: int = 0,
                 validation_size: int = 0,
                 val_batch_size: int = 0,
                 predict_steps: int = 0,
                 dataset_type: str = 'GRAPH',
                 dataset_args: Dict[str, str] = None,
                 verbose: bool = True,
                 num_workers: int = 4,
                 norm_type : str = 'minmax',
                 ):
        super().__init__()
        print(f"\n\n[Initializing {type(self).__name__} Module]") if verbose else None
        print("==========================================") if verbose else None
        print(
            f"Loading coordinates from file {xtcfile} and topology from file {tprfile}") if verbose else None
        if not os.path.exists(xtcfile):
            raise FileNotFoundError(f"File {xtcfile} not found")
        if not os.path.exists(tprfile):
            raise FileNotFoundError(f"File {tprfile} not found")
        

        assert all (isinstance(x, int) and x >= 0 for x in 
                    [train_size, validation_size, predict_steps]), \
                        "train_size, validation_size, and predict_steps must be non-negative integers"
        assert train_size > 0 or predict_steps > 0, "Either train_size or predict_steps must be provided and greater than 0"
        assert not all (x > 0 for x in 
                    [train_size, predict_steps]), \
                        "Only one of training or prediction can be performed at a time (train_size and predict_steps cannot both be greater than 0)"
        self.predicting = predict_steps > 0
        if not self.predicting:
            assert sequence_length > 1, "sequence_length must be greater than 1 when training"
            assert batch_size > 0, "batch_size must be provided and greater than 0 when training"
            assert validation_size > 0, "validation_size must be provided and greater than 0 when training"
            assert val_batch_size > 0, "val_batch_size must be provided and greater than 0 when training"
            assert batch_size <= train_size, "Batch size must be less than or equal to the training size"
            assert val_batch_size <= validation_size, "Validation batch size must be less than or equal to the validation size"
            data_length = train_size + validation_size
        else:
            sequence_length = predict_steps
            data_length = 1

        dataset_args = ({k: eval(v) for k, v in (arg.split('=')
                        for arg in dataset_args)}
                        if dataset_args is not None else {})

        # Load the trajectory
        u = mda.Universe(tprfile, xtcfile)

        traj_length = len(u.trajectory)
        # Checks
        if traj_length < sequence_length:
            raise ValueError(f"Trajectory length {traj_length} is shorter than sequence length {sequence_length}")
        if traj_length < data_length + sequence_length and not self.predicting:
            warnings.warn(f"Trajectory length {traj_length} is short. There will be repeated sequences.")

        # Select the atoms
        try:
            mol = u.select_atoms(selection)
        except Exception as e:
            raise ValueError(f"Selection {selection} is not valid: {e}")
        if mol.n_atoms == 0:
            raise ValueError(f"Selection {selection} does not match any atoms in the trajectory")
        
        # Center the and unwrap the trajectory
        transforms = [trans.unwrap(mol),
                      trans.center_in_box(mol, center='geometry', point=[0.0,0.0,0.0], wrap=False)]
        u.trajectory.add_transformations(*transforms)
    
        # Extract the atomic numbers
        at_elements = [at.element for at in mol]
        self.atns = []
        for elem in at_elements:
            assert elem in atomic_numbers, f"Atom {elem} not found in atomic numbers dictionary"
            self.atns.append(atomic_numbers[elem])

        # Get the atom numbers in the trajectory
        atm_ids = [at.id + 1 for at in mol.atoms]
        
        # Extract the bonds information
        self.bonds = mol.get_connections('bonds', outside=False).indices
        for i in range(len(self.bonds)): # remap to mol atoms indices (without hydrogens)
            self.bonds[i] = (np.where(mol.atoms.indices == self.bonds[i][0])[0][0], np.where(mol.atoms.indices == self.bonds[i][1])[0][0])

        # Read the trajectory and store the frames
        mol_traj = []
        for i in tqdm(range(data_length), disable=not verbose, desc="Reading Sequences"):
            s = random.randint(0, len(u.trajectory) - sequence_length)
            e = s + sequence_length
            read_frame_seq = [a for a in range(s, e)]

            for idx in tqdm(read_frame_seq, disable=not verbose, desc="Reading Frames", leave=False):
                u.trajectory[idx]
                # Create the ASE structure
                structure = ase.Atoms(numbers=self.atns, positions=mol.atoms.positions)

                # Retain topology information
                residues = [str(r.residue.resname) for r in mol.atoms]
                resids = [r.residue.resid for r in mol.atoms]
                atomnames = [str(a.name) for a in mol.atoms]
                structure.set_array('residuenumbers', np.array(resids))
                structure.set_array('residuenames', np.array(residues))
                structure.set_array('atomtypes', np.array(atomnames))

                mol_traj.append(structure)

        print(f"Finished reading trajectory.") if verbose else None

        
        self.mol_traj = mol_traj

        if dataset_type == 'DEFAULT':
            dataset_class = XtcData
            self.dl_cls = DataLoader
        elif dataset_type == 'DISTANCES':
            from collective_encoder.datasets.distances import DistancesDataset as dataset_class
            dataset_args['atm_ids'] = atm_ids
            self.dl_cls = DataLoader
        elif dataset_type == 'GRAPH':
            from collective_encoder.datasets.bondgraph import BondGraphDataset as dataset_class
            dataset_args['bond_indices'] = self.bonds
            self.dl_cls = GeoDataLoader
        else:
            raise ValueError(f"Unknown dataset type: {dataset_type}")

        self.xtcData_full = dataset_class(
                structures=mol_traj,
                **dataset_args
                )

        if dataset_type == 'GRAPH':
            print(f"Loaded graph dataset with {len(self.xtcData_full)} graphs") if verbose else None
            print(f"Number of bonds (nodes): {len(self.bonds)}") if verbose else None
            print(f"Node feature size: {self.xtcData_full[0].x.shape[1]}") if verbose else None
            print(f"Edge feature size: {self.xtcData_full[0].edge_attr.shape[1]}") if verbose else None
            print(f"Edges per graph: {self.xtcData_full[0].edge_index.shape[1]}") if verbose else None
            self.num_inputs = len(self.bonds)
            self.datapoint_shape = (len(self.bonds), self.xtcData_full[0].x.shape[1])

        self.save_hyperparameters()

        if not self.predicting:
            self.train_size = train_size
            self.validation_size = validation_size
            print(f"Train size: {self.train_size}, Validation size: {self.validation_size}") if verbose else None
            # Select train and validation sets
            self.train_indices, self.val_indices = train_test_split(
                np.arange(self.train_size + self.validation_size),
                train_size=self.train_size,
                test_size=self.validation_size,
                shuffle=True
            )
        else:
            self.predict_steps = predict_steps
            print(f"Prediction size: {self.predict_steps}") if verbose else None
            self.pred_indices = [0]
        # print(f"Train indices: {self.train_indices}") if verbose else None
        # print(f"Validation indices: {self.val_indices}") if verbose else None
        self.target_scaler = None
        self.data_length = data_length
        print("==========================================") if verbose else None

    # def prepare_data(self): # only called on 1 GPU/TPU in distributed

    # def setup(self, stage):  # Called on every GPU/TPU in distributed
    
    def get_atns(self):
        return self.atoms

    def get_bond_indices(self):
        return self.bonds
    
    def fit_target_scaler(self):
        if self.target_scaler is not None:
            return
        if self.hparams.norm_type == 'standard':
            self.target_scaler = StandardScaler()
        elif self.hparams.norm_type == 'minmax':
            self.target_scaler = MinMaxScaler()
        else:
            raise ValueError(f"Normalization type {self.hparams.norm_type} not supported")

        if self.hparams.dataset_type in ['DEFAULT', 'DISTANCES']:
            self.num_inputs = self.xtcData_full.num_inputs
            self.datapoint_shape = tuple(self.xtcData_full[0][0].shape)
            self.target_scaler.fit(self.xtcData_full.get_data()[0])
        elif self.hparams.dataset_type == 'GRAPH':
            data_to_normalize = [] 
            for g in self.xtcData_full:
                node_feat = g.x.numpy()
                edge_feat = g.edge_attr.numpy()
                data_to_normalize.append(np.hstack([node_feat.mean(axis=0), edge_feat.mean(axis=0)]))
            data_to_normalize = np.vstack(data_to_normalize)
            self.target_scaler.fit(data_to_normalize)
        else:
            raise ValueError(f"Unsupported dataset type for normalization {self.hparams.dataset_type}")
    
    def get_full_batch(self):
        dl = self.full_dataloader()
        return next(iter(dl))
    
    def get_dataset(self):
        return self.xtcData_full

    def get_train_val_dataloaders(self, indices, batch_size):
        data_indices = []
        for idx in indices:
            s = idx * self.hparams.sequence_length
            e = s + self.hparams.sequence_length
            data_indices.extend([i for i in range(s, e)])
        data = torch.utils.data.Subset(self.xtcData_full, data_indices)
        return self.dl_cls(
            data,
            batch_size=batch_size * self.hparams.sequence_length,
            shuffle=False,
            drop_last=True,
            num_workers=self.hparams.num_workers,
            pin_memory=True)

    def train_dataloader(self):
        assert not self.predicting, "train_dataloader called in prediction mode"
        return self.get_train_val_dataloaders(self.train_indices, 
                                              self.hparams.batch_size)

    def val_dataloader(self):
        assert not self.predicting, "val_dataloader called in prediction mode"
        return self.get_train_val_dataloaders(self.val_indices, 
                                              self.hparams.val_batch_size)

    def test_dataloader(self):
        assert self.predicting, "test_dataloader called in training mode"
        return self.full_dataloader()

    def predict_dataloader(self):
        assert self.predicting, "predict_dataloader called in training mode"
        return self.get_train_val_dataloaders(self.pred_indices, 1)

    def full_dataloader(self):
        all_indices = np.arange(self.data_length)
        return self.get_train_val_dataloaders(all_indices, len(all_indices))

    def target_scaler(self, X):
        self.fit_target_scaler()
        return self.target_scaler.transform(X)

    def target_inverse_scaler(self, X):
        self.fit_target_scaler()
        return self.target_scaler.inverse_transform(X)

    def get_scaler_mean(self):
        self.fit_target_scaler()
        if self.hparams.norm_type == 'minmax':
            return self.target_scaler.min_
        return self.target_scaler.mean_

    def get_scaler_var(self):
        self.fit_target_scaler()
        return self.target_scaler.var_

    def get_scaler_scale(self):
        self.fit_target_scaler()
        return self.target_scaler.scale_

    def get_datapoint_shape(self):
        return self.datapoint_shape
