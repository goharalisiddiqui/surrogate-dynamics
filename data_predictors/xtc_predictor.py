import os
import argparse
from typing import List, Dict
from tqdm import tqdm
import random
import warnings

import numpy as np
import ase

import MDAnalysis as mda
from MDAnalysis.analysis import align
from MDAnalysis.analysis.rms import rmsd
import MDAnalysis.transformations as trans

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl

DOUBLE_PRECISION = False

warnings.filterwarnings("ignore")

atomic_numbers = { 'C' : 6
                , 'H' : 1
                , 'O' : 8
                , 'N' : 7
                , 'S' : 16
                , 'P' : 15
                , 'F' : 9
                , 'Cl' : 17
                , 'Br' : 35
                , 'I' : 53
                , 'Si' : 14
}

class XtcData(Dataset):
    """XTC dataset"""

    def __init__(
        self,
        structures: List[ase.Atoms],
        dtype=torch.float32,
    ):
        self.positions = [torch.tensor(s.positions, dtype=dtype) for s in structures]
        self.num_inputs = len(self.positions[0])

    def __len__(self):
        return len(self.positions)

    def __getitem__(self, index):
        x = ()
        x += (self.positions[index],)
        return x

def xtcpdatset_args():
    desc = "Xtc Dataset Arguments"
    parser = argparse.ArgumentParser(description=desc)


    parser.add_argument('--xtcfile', required=True, type=str, help='Input compressed coordinate file')
    parser.add_argument('--tprfile', required=True, type=str, help='Input binary file containing the topology')
    parser.add_argument('--resnames', required=True, nargs='+', help='Residue names to get the coordinates')
    parser.add_argument('--datasize', dest="dataset_size", type=int, default = None, help='Size of the dataset to use')
    # parser.add_argument('--labels',dest = 'label_list', nargs='+', help='Label columns in the data file')

    args, _ = parser.parse_known_args()

    return args


XTCP_args = xtcpdatset_args

class XtcPredictor():
    def __init__(self,
                 xtcfile : str,
                 tprfile : str,
                 resnames : List[str] = None,
                 dataset_size : int = None,
                 verbose : bool = False
                 ):
        self.verbose = verbose
        print(f"\n\n[Initializing XtcPredictor Module]") if verbose else None
        print("==========================================") if verbose else None
        print(f"Loading coordinates from file {xtcfile} and topology from file {tprfile}") if verbose else None
        if not os.path.exists(xtcfile):
            raise FileNotFoundError(f"File {xtcfile} not found")
        if not os.path.exists(tprfile):
            raise FileNotFoundError(f"File {tprfile} not found")
        u = mda.Universe(tprfile, xtcfile)
        for res in resnames:
            assert res in list(set(u.residues.resnames)), f"Residue \"{res}\" not found in the topology file. Available residues: {list(set(u.residues.resnames))}"
        if resnames is None:
            raise ValueError("Residue names not provided")
        mol = u.select_atoms(f"resname " + " ".join(resnames))
        transforms = [trans.unwrap(mol),
              trans.center_in_box(mol, wrap=True)]
        u.trajectory.add_transformations(*transforms)

        ch = [at.element for at in mol]
        for at in ch:
            if at not in atomic_numbers:
                raise ValueError(f"Atom {at} not found in atomic numbers dictionary")
        ch = [atomic_numbers[at] for at in ch]
        mol_traj = []
        s, e = 0, len(u.trajectory)
        if dataset_size is not None:
            s = random.randint(0, len(u.trajectory) - dataset_size - 1)
            e = dataset_size + s
        print(f"Reading trajectory of {e-s} frames...") if verbose else None
        for ts in tqdm(u.trajectory[s:e], disable=not verbose):
            structure = ase.Atoms(numbers=ch, positions=mol.positions)
            mol_traj.append(structure)
        print(f"Finished reading trajectory.") if verbose else None

        ## Setting up the new universe to write
        self.u_res = mda.Universe.empty(len(ch),
                                   n_residues = len(resnames),
                                   atom_resindex = sum([[i] * len(mol.residues[i].atoms) for i in range(len(resnames))], []),
                                   residue_segindex = [0] * len(resnames),
                                   trajectory=True)
        self.u_res.add_TopologyAttr('segid', ['0'])
        self.u_res.add_TopologyAttr('resid', list(range(1, len(resnames) + 1)))
        self.u_res.add_TopologyAttr('chainIDs', ['1'] * len(ch))
        self.u_res.add_TopologyAttr('occupancies', [1.0] * len(ch))
        self.u_res.add_TopologyAttr('tempfactors', [0.0] * len(ch))
        self.u_res.add_TopologyAttr('name', mol.atoms.names)
        self.u_res.add_TopologyAttr('type', mol.atoms.types)
        self.u_res.add_TopologyAttr('elements', mol.atoms.elements)
        self.u_res.add_TopologyAttr('resname', resnames)
        self.u_res.dimensions = mol.dimensions


        self.new_coordinates = None
        coordinates = [frame.positions for frame in mol_traj[:10]]
        coordinates = np.array(coordinates)

        self.xtcData_full = XtcData(
                structures=mol_traj,
                dtype=torch.float64 if DOUBLE_PRECISION else torch.float32)
        self.num_inputs = self.xtcData_full.num_inputs
        self.datapoint_shape = tuple(self.xtcData_full[0][0].shape)

        print(f"Total frames: {len(self.xtcData_full)}, Number of size: {self.num_inputs}, Data shape: {self.datapoint_shape}") if verbose else None
        print("==========================================") if verbose else None



    def get_warmup_steps(self, size=5):
        if size > len(self.xtcData_full):
            raise ValueError(f"Size of sample is greater than the total number of frames loaded")
        mddata = torch.utils.data.Subset(self.xtcData_full, list(range(size)))
        dl = DataLoader(
            mddata,
            batch_size=len(mddata),
            shuffle=False,
            pin_memory=True)
        res = next(iter(dl))
        coordinates = res[0].detach().cpu().numpy()
        assert coordinates.shape[0] == size
        assert coordinates.shape[1] == self.u_res.atoms.n_atoms
        assert coordinates.shape[2] == 3
        self.extend_trajectory(coordinates)
        print(f"Loaded {size} frames for warmup") if self.verbose else None
        return res

    def get_full_batch(self):
        mddata = self.xtcData_full
        dl = DataLoader(
            mddata,
            batch_size=len(mddata),
            shuffle=False,
            pin_memory=True)
        return next(iter(dl))

    def extend_trajectory(self, new_coordinates):
        if new_coordinates.shape[1] != self.u_res.atoms.n_atoms and new_coordinates.shape[2] != 3:
            raise ValueError(f"Invalid shape of new coordinates. Expected shape: ({self.u_res.atoms.n_atoms}, 3), got {new_coordinates.shape}")
        if self.new_coordinates is None:
            self.new_coordinates = new_coordinates
        else:
            self.new_coordinates = np.concatenate((self.new_coordinates, new_coordinates), axis=0)
        print(f"Added {new_coordinates.shape[0]} predicted frames to the trajectory") if self.verbose else None

    def output_trajectory(self, output_file):
        self.u_res.load_new(self.new_coordinates)
        print(f"Writing {self.new_coordinates.shape[0]} frames to the trajectory") if self.verbose else None
        self.u_res.atoms.write(output_file, frames='all')

    def get_coordinates(self):
        return self.new_coordinates


    def get_datapoint_shape(self):
        return self.datapoint_shape

