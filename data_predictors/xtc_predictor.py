import os
import argparse
from typing import List, Dict
from tqdm import tqdm
import random
import warnings

import numpy as np
import pandas as pd
import ase
from ase.data import covalent_radii

import matplotlib.pyplot as plt

import MDAnalysis as mda
from MDAnalysis.analysis import align
from MDAnalysis.analysis import dihedrals
from MDAnalysis.analysis.rms import rmsd
import MDAnalysis.transformations as trans

import torch
from torch.utils.data import Dataset, DataLoader

from drivers.utils.io_coordinates import pdb_to_ase

DOUBLE_PRECISION = False

warnings.filterwarnings("ignore")

atomic_numbers = {'C': 6, 'H': 1, 'O': 8, 'N': 7, 'S': 16, 'P': 15, 'F': 9, 'Cl': 17, 'Br': 35, 'I': 53, 'Si': 14
                  }


class XtcData(Dataset):
    """XTC dataset"""

    def __init__(
        self,
        structures: List[ase.Atoms],
        dtype=torch.float32,
    ):
        self.positions = [torch.tensor(
            s.positions, dtype=dtype) for s in structures]
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

    parser.add_argument('--xtcfile', required=True, type=str,
                        help='Input compressed coordinate file')
    parser.add_argument('--tprfile', required=True, type=str,
                        help='Input binary file containing the topology')
    parser.add_argument('--resnames', required=True, nargs='+',
                        help='Residue names to get the coordinates')
    # parser.add_argument('--datasize', dest="dataset_size",
    #                     type=int, default=None, help='Size of the dataset to use')
    # parser.add_argument('--labels',dest = 'label_list', nargs='+', help='Label columns in the data file')

    args, _ = parser.parse_known_args()

    return args


XTCP_args = xtcpdatset_args


class XtcPredictor():
    def __init__(self,
                 xtcfile: str,
                 tprfile: str,
                 resnames: List[str] = None,
                 dataset_size: int = None,
                 verbose: bool = False
                 ):
        self.verbose = verbose
        print(f"\n\n[Initializing XtcPredictor Module]") if verbose else None
        print("==========================================") if verbose else None
        print(
            f"Loading coordinates from file {xtcfile} and topology from file {tprfile}") if verbose else None
        if not os.path.exists(xtcfile):
            raise FileNotFoundError(f"File {xtcfile} not found")
        if not os.path.exists(tprfile):
            raise FileNotFoundError(f"File {tprfile} not found")
        u = mda.Universe(tprfile, xtcfile)
        for res in resnames:
            assert res in list(set(
                u.residues.resnames)), f"Residue \"{res}\" not found in the topology file. Available residues: {list(set(u.residues.resnames))}"
        if resnames is None:
            raise ValueError("Residue names not provided")
        mol = u.select_atoms(f"resname " + " ".join(resnames) + " and not name H*")
        transforms = [trans.unwrap(mol),
                      trans.center_in_box(mol, center='geometry', point=[0.0,0.0,0.0], wrap=False)]
        u.trajectory.add_transformations(*transforms)

        ch = [at.element for at in mol]
        for at in ch:
            if at not in atomic_numbers:
                raise ValueError(
                    f"Atom {at} not found in atomic numbers dictionary")
        ch = [atomic_numbers[at] for at in ch]
        self.loatn = ch
        self.bonds = mol.get_connections('bonds', outside=False).indices
        for i in range(len(self.bonds)):
            self.bonds[i] = [np.where(mol.atoms.indices == self.bonds[i][0])[0][0], np.where(mol.atoms.indices == self.bonds[i][1])[0][0]]
        
        mol_traj = []
        s, e = 0, len(u.trajectory)
        if dataset_size is not None:
            if dataset_size > len(u.trajectory):
                raise ValueError(
                    f"Not enough frames in the trajectory. Requested: {dataset_size}, Available: {len(u.trajectory)}")
            s = random.randint(0, len(u.trajectory) - dataset_size)
            e = dataset_size + s
        print(f"Reading trajectory of {e-s} frames...") if verbose else None
        for ts in tqdm(u.trajectory[s:e], disable=not verbose):
            structure = ase.Atoms(numbers=ch, positions=mol.positions)
            mol_traj.append(structure)
        print(f"Finished reading trajectory.") if verbose else None
        
        ## Quick fix, need to change later
        self.u_original = u
        self.data_window = (s, e)
        self.ch = ch
        
        # Setting up the new universe to write
        self.u_res = mda.Universe.empty(len(ch),
                                        n_residues=len(resnames),
                                        atom_resindex=[x.resid - 1 for x in mol.atoms],
                                        residue_segindex=[0] * len(resnames),
                                        trajectory=True)
        self.u_res.add_TopologyAttr('segid', ['0'])
        self.u_res.add_TopologyAttr('resid', list(range(1, len(resnames) + 1)))
        self.u_res.add_TopologyAttr('chainIDs', ['1'] * len(ch))
        self.u_res.add_TopologyAttr('occupancies', [1.0] * len(ch))
        self.u_res.add_TopologyAttr('tempfactors', [0.0] * len(ch))
        self.u_res.add_TopologyAttr('name', mol.atoms.names)
        self.u_res.add_TopologyAttr('type', mol.atoms.types)
        self.u_res.add_TopologyAttr('elements', mol.atoms.elements)
        self.u_res.add_TopologyAttr('resname', mol.residues.resnames)
        self.u_res.add_TopologyAttr('dihedrals', mol.dihedrals.indices)
        self.u_res.dimensions = mol.dimensions

        # Setting up an extra universe to do the validation

        self.new_coordinates = None
        # coordinates = [frame.positions for frame in mol_traj[:10]]
        # coordinates = np.array(coordinates)
        self.switch_idx = []
        cov_radii = [covalent_radii[el] for el in self.loatn]
        cov_radii = np.array(cov_radii)
        cov_radii = np.expand_dims(cov_radii, 0)
        cd_t = cov_radii.transpose()
        cov_mat = np.expand_dims(cov_radii, 0) + np.expand_dims(cd_t, 1)
        self.cov_mat = np.squeeze(cov_mat, 1)
        
        self.xtcData_full = XtcData(
            structures=mol_traj,
            dtype=torch.float64 if DOUBLE_PRECISION else torch.float32)
        self.num_inputs = self.xtcData_full.num_inputs
        self.datapoint_shape = tuple(self.xtcData_full[0][0].shape)

        print(
            f"Total frames: {len(self.xtcData_full)}, Number of size: {self.num_inputs}, Data shape: {self.datapoint_shape}") if verbose else None
        print("==========================================") if verbose else None

    def run_md(self, inputs_folder, run_folder, n_steps):
        if not os.path.exists(inputs_folder):
            raise FileNotFoundError(f"Folder {inputs_folder} not found")
        assert n_steps > 0, "Number of steps should be greater than 0"
        assert self.new_coordinates is not None, "No coordinates to run MD on"
        
        params_file = os.path.join(inputs_folder, "run.mdp")
        topology_file = os.path.join(inputs_folder, "topol.top")
        assert os.path.exists(params_file), "MDP file not found in the input folder"
        assert os.path.exists(topology_file), "MDP file not found in the input folder"
        
        params_file = os.path.abspath(params_file)
        topology_file = os.path.abspath(topology_file)
        structure = ase.Atoms(
            numbers=self.ch, 
            # positions=self.select_frame(self.new_coordinates)
            positions=self.get_synthetic_frame(self.new_coordinates[-1])
        )
        residues = [str(r.residue.resname) for r in self.u_res.atoms]
        resids = [r.residue.resid for r in self.u_res.atoms]
        atomnames = [str(a.name) for a in self.u_res.atoms]
        structure.set_array('residuenumbers', np.array(resids))
        structure.set_array('residuenames', np.array(residues))
        structure.set_array('atomtypes', np.array(atomnames))
        
        cell = structure.get_cell()
        if (cell == 0).all():
            coordinates = structure.get_positions()
            buffer = 50.0
            # xmin, ymin, zmin = coordinates.min(axis=0) 
            xmax, ymax, zmax = coordinates.max(axis=0)
            x, y, z = xmax, ymax, zmax # FIXME: Find a better way to do this
            x, y, z = x + buffer, y + buffer, z + buffer
            x, y, z = int(x), int(y), int(z)
            cell = np.array([[x, 0, 0], [0, y, 0], [0, 0, z]]).astype(float)
        structure.cell  = cell
        structure.pbc = [True, True, True]
        
        i = 1
        while os.path.exists(os.path.join(run_folder, f"md{i}")):
            i += 1
        run_folder = os.path.join(run_folder, f"md{i}")
        run_name = f"md{i}"
        run_folder = os.path.abspath(run_folder)
        
        from drivers.mdengines.gromacs import gromacs_driver
        gromacs = gromacs_driver(
            input_file=params_file, run_name=run_name, run_dir=run_folder, verbose=False)
        old_comm = gromacs.slurm_commands
        gromacs.slurm_commands = ['echo -e "1\\n1" | gmx_mpi pdb2gmx -f frame.pdb -ignh -o frame.gro -p topol.top -i posre_initial.itp']
        for comm in old_comm:
            gromacs.slurm_commands.append(comm)
        gromacs.template_input_script({'${N_STEPS}': str(n_steps)})
        # gromacs.add_aux_file(source=topology_file, dest="topol.top")
        gromacs.add_coordinates_file(structure, "frame.pdb")
        gromacs.run(dry_run=False)
        print(f"Waiting for {run_name}") if self.verbose else None
        gromacs.wait_for_success()
        
        traj = pdb_to_ase(os.path.join(run_folder, "traj.pdb"))
        traj = [t.get_positions() for t in traj]
        traj = np.array(traj)
        curr_length = len(self.new_coordinates) if self.new_coordinates is not None else 0
        self.extend_trajectory(traj)
        self.switch_idx.append(curr_length)
        self.switch_idx.append(curr_length + n_steps)
        print(f"Ran MD for {n_steps} steps") if self.verbose else None
        
        return torch.tensor(traj, 
            dtype=torch.float64 if DOUBLE_PRECISION else torch.float32)
    
    def select_frame(self, frames):
        for i in range(1, len(frames) + 1):
            frame = frames[-i]
            accept_frame = True
            for bond in self.bonds:
                if 0 in bond: # Do not check bonds with hydrogen
                    continue
                dist = np.linalg.norm(frame[bond[0]] - frame[bond[1]])
                cov_dist = self.cov_mat[bond[0], bond[1]]
                if dist > 1.2 * cov_dist or dist < 0.6 * cov_dist:
                    accept_frame = False
                    break
            if accept_frame:
                print(f"Skipped {i} frames for the md run.") if self.verbose else None
                return frame
        print("No valid frame found in the trajectory")
        return frame
        
    def get_synthetic_frame(self, frame):
        u_res = self.u_res.copy()
        u_res.atoms.positions = frame
        phi_current = dihedrals.Dihedral([u_res.residues[1].phi_selection()]).run().results.angles[-1]
        psi_current = dihedrals.Dihedral([u_res.residues[1].psi_selection()]).run().results.angles[-1]
        print(f"Current phi: {phi_current}") 
        print(f"Current psi: {psi_current}") 
        raise ValueError("Not implemented")
        exit()
        pristine_frame  = self.new_coordinates[-1]
        
        
        
        
    def get_warmup_steps(self, size=5):
        curr_length = len(self.new_coordinates) if self.new_coordinates is not None else 0
        if curr_length + size > len(self.xtcData_full):
            raise ValueError(
                f"Size fo the next warmup steps {size} plus previous steps {curr_length} exceeds the total number of frames {len(self.xtcData_full)}")
        mddata = torch.utils.data.Subset(self.xtcData_full, list(range(curr_length, curr_length + size)))
        dl = DataLoader(
            mddata,
            batch_size=len(mddata),
            shuffle=False,
            pin_memory=True)
        res = next(iter(dl))[0]
        coordinates = res.detach().cpu().numpy()
        assert coordinates.shape[0] == size
        assert coordinates.shape[1] == self.u_res.atoms.n_atoms
        assert coordinates.shape[2] == 3
        self.extend_trajectory(coordinates)
        self.switch_idx.append(curr_length)
        self.switch_idx.append(curr_length + size)
        print(f"Loaded {size} real frames for warmup") if self.verbose else None
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
            raise ValueError(
                f"Invalid shape of new coordinates. Expected shape: ({self.u_res.atoms.n_atoms}, 3), got {new_coordinates.shape}")
        if self.new_coordinates is None:
            self.new_coordinates = new_coordinates
        else:
            self.new_coordinates = np.concatenate(
                (self.new_coordinates, new_coordinates), axis=0)
        print(
            f"Added {new_coordinates.shape[0]} predicted frames to the trajectory") if self.verbose else None

    def output_trajectory(self, output_folder):
        self.u_res.load_new(self.new_coordinates)
        # print(self.new_coordinates.shape)
        # exit()
        print(
            f"Writing {self.new_coordinates.shape[0]} frames to the trajectory") if self.verbose else None
        self.u_res.atoms.write(output_folder + "predicted.pdb", frames='all')
        print(f"Predicted trajectory written to {output_folder + 'predicted.pdb'}") if self.verbose else None
        self.print_validation_plots(output_folder)
    
    def output_real_trajectory(self, output_folder):
        self.u_original.atoms.write(output_folder + "real.pdb", frames=self.u_original.trajectory[self.data_window[0]:self.data_window[1]])
        print(f"Real trajectory written to {output_folder + 'real.pdb'}") if self.verbose else None
        # self.print_validation_plots(output_folder)

    def print_validation_plots(self, output_folder):
        len_predicted = len(self.u_res.trajectory)

        phi_sel = self.u_res.residues[1].phi_selection()
        psi_sel = self.u_res.residues[1].psi_selection()

        dih_predicted = dihedrals.Dihedral(
            [phi_sel, psi_sel]).run().results.angles
        dih_predicted = np.array(dih_predicted) * np.pi / 180

        full_traj = self.get_full_batch()[0].detach().cpu().numpy()
        self.u_res.load_new(full_traj)
        dih_all = dihedrals.Dihedral([phi_sel, psi_sel]).run().results.angles
        dih_all = np.array(dih_all) * np.pi / 180
        
        # Revert back for further predictions
        self.u_res.load_new(self.new_coordinates)


        fig, ax = plt.subplots(2, 2, figsize=(10, 10))

        # Phi
        ax[0][0].plot(dih_all[:, 0], label=r'$\phi_{true}$', color='C0',
                      marker='x', linestyle='-', markersize=2, linewidth=0.0)
        ax[0][1].plot(dih_predicted[:, 0], label=r'$\phi_{predicted}$',
                      color='C0', marker='x', linestyle='-', markersize=2, linewidth=0.0)

        ax[0][0].set_title('True', fontsize=16)
        ax[0][1].set_title('Predicted', fontsize=16)
        for axes in ax[0]:
            axes.set_ylabel(r'$\phi$')
            axes.set_xlabel("Trajectory frame")
            axes.set_ylim(-np.pi, np.pi)
            axes.set_yticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
            axes.set_yticklabels(
                [r'$-\pi$', r'$-\pi/2$', r'$0$', r'$\pi/2$', r'$\pi$'])

        wl = ax[0][1].vlines(self.switch_idx, -np.pi, np.pi,
                             color='r', linestyle='--', label='Warmup length')

        ax[0][1].legend([wl], ['Real trajectories'], loc='upper right')
        
        
        # Psi
        ax[1][0].plot(dih_all[:, 1], label=r'$\psi_{true}$', color='C1',
                      marker='x', linestyle='-', markersize=2, linewidth=0.0)
        ax[1][1].plot(dih_predicted[:, 1], label=r'$\psi_{predicted}$',
                      color='C1', marker='x', linestyle='-', markersize=2, linewidth=0.0)

        for axes in ax[1]:
            axes.set_ylabel(r'$\psi$')
            axes.set_xlabel("Trajectory frame")
            axes.set_ylim(-np.pi, np.pi)
            axes.set_yticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
            axes.set_yticklabels(
                [r'$-\pi$', r'$-\pi/2$', r'$0$', r'$\pi/2$', r'$\pi$'])

        wl2 = ax[1][1].vlines(self.switch_idx, -np.pi, np.pi,
                              color='r', linestyle='--', label='Warmup length')

        ax[1][1].legend([wl2], ['Real trajectories'])

        fig.tight_layout()
        plt.savefig(output_folder + "dihedrals.png", dpi=300)
        fig.clear()
        plt.close(fig)
        
        # Save data as csv
        df = pd.DataFrame(columns=['phi_true', 'phi_predicted', 'psi_true', 'psi_predicted','warmupframe'])
        df['phi_true'] = dih_all[:, 0]
        df['phi_predicted'] = dih_predicted[:, 0]
        df['psi_true'] = dih_all[:, 1]
        df['psi_predicted'] = dih_predicted[:, 1]
        warmupflag = [0] * len(dih_predicted)
        i = 0
        while i + 1 < len(self.switch_idx):
            for j in range(self.switch_idx[i], self.switch_idx[i+1]):
                warmupflag[j] = 1
            i += 2
        df['warmupframe'] = warmupflag
        
        df.to_csv(output_folder + "dihedrals.csv", index=False)
        
        # Ramachandran plot
        # real_interval = np.array(self.switch_idx).reshape(-1, 2)
        # real_frames = []
        # for interval in real_interval:
        #     real_frames.extend(list(range(interval[0], interval[1] + 1)))
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        d1 = 5
        d2 = 100
        i = 0
        k = 0
        while i < dih_all.shape[0]:
            if len(self.switch_idx) > k and self.switch_idx[k] == i:
                ax.plot(dih_all[self.switch_idx[k]:self.switch_idx[k+1]:d1, 0], 
                        dih_all[self.switch_idx[k]:self.switch_idx[k+1]:d1, 1], 'x', color='black')
                i += self.switch_idx[k+1] - self.switch_idx[k]
                k += 2
            else:
                if len(self.switch_idx) == k:
                    ax.plot(dih_all[i::d2, 0], dih_all[i::d2, 1], 'x', color='red')
                    i = dih_all.shape[0]
                else:
                    ax.plot(dih_all[i:self.switch_idx[k]:d2, 0], dih_all[i:self.switch_idx[k]:d2, 1], 'x', color='red')
                    i = self.switch_idx[k]
        
        

        
        
        
        
        # ax.plot(dih_all[:, 0], dih_all[:, 1], 'x', color='gray')
        ax.set_xlabel(r'$\phi$')
        ax.set_ylabel(r'$\psi$')
        ax.set_title(r'$\phi$ vs $\psi$')
        ax.set_xlim(-np.pi, np.pi)
        ax.set_ylim(-np.pi, np.pi)
        ax.set_xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
        ax.set_yticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
        ax.set_xticklabels(
            [r'$-\pi$', r'$-\pi/2$', r'$0$', r'$\pi/2$', r'$\pi$'])
        ax.set_yticklabels(
            [r'$-\pi$', r'$-\pi/2$', r'$0$', r'$\pi/2$',r'$\pi$'])
        plt.savefig(output_folder + "phi_vs_psi.png", dpi=300)
    
        fig.clear() 
        plt.close(fig)
        

    def get_coordinates(self):
        return self.new_coordinates

    def get_datapoint_shape(self):
        return self.datapoint_shape

    def __len__(self):
        return len(self.xtcData_full)
