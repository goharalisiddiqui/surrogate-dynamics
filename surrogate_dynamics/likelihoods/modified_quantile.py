import os
from darts.logging import raise_if_not
from darts.utils.likelihood_models import QuantileRegression
from torch.nn.functional import pairwise_distance

from ase.data import covalent_radii

import torch
from typing import Optional


class ModifiedQuantileRegression(QuantileRegression):
    def __init__(self, quantiles: Optional[list[float]] = None, enc_ckpt: Optional[str] = None, elems: Optional[list[int]] = None, bond_connections: Optional[list[tuple[int, int]]] = None):
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
        if enc_ckpt is not None and bond_connections is None:
            raise ValueError(
                "bond_connections must be provided if enc_ckpt is provided")

        super().__init__(quantiles)
        self.encoder = None
        self.enc_ckpt = enc_ckpt
        elements = elems
        cov_radii = [covalent_radii[el] for el in elements]
        cov_radii = torch.tensor(cov_radii).float()
        cov_radii = cov_radii.unsqueeze(0)
        cd_t = cov_radii.transpose(0, 1)
        cov_mat = cov_radii.unsqueeze(0) + cd_t.unsqueeze(1)
        self.cov_mat = cov_mat.squeeze(1)
        self.bond_indices = bond_connections

    def load_encoder(self, enc_ckpt: str):
        encoder_id = os.path.basename(enc_ckpt).split("_")[0]
        if encoder_id == "EDVAE":
            from nets.edvae_net import EDVAE as enc_model
        elif encoder_id == "EDVAEGAN":
            from nets.edvae_gan_net import EDVAEGAN as enc_model
        else:
            raise ValueError(f"Unknown encoder model: {encoder_id}")
        self.encoder = enc_model.load_from_checkpoint(enc_ckpt)
        self.encoder.eval()

    def compute_bondlength_deviation_loss(self, model_output: torch.Tensor):
        bonded_indices = self.bond_indices
        
        if True:
            weights = torch.min((1 - self.quantiles_tensor), self.quantiles_tensor)
            weights = weights / weights.sum()
            model_output = torch.sum(model_output * weights, dim=-1)
        else:
            model_output = self.sample(model_output)
            
        # print(f"Model Output Weighted Avg: {model_output_weighted_avg.shape}")
        
        # print(f"Model Output Example: {model_output[0, 0, 0]}")
        # print(f"Model Output weighted Example: {model_output_weighted_avg[0, 0, 0]}")
        # exit()
        if self.encoder is not None:
            # Get the coordinates from the latent space of the encoder
            if self.encoder.device != model_output.device:
                model_output = model_output.to(self.encoder.device)
            if self.cov_mat.device != model_output.device:
                self.cov_mat = self.cov_mat.to(model_output.device)
            coordinates = self.encoder.decode_latent(
                model_output.view(-1, model_output.shape[-1]), keeptensor=True)
        else:
            coordinates = model_output.view(model_output.shape[0], model_output.shape[1], -1, 3)
        # print(f"Coordinates shape: {coordinates.shape}")
        # print(f"Coordinates: {coordinates}")
        
        # Reshape the coordinates to get the flattened coordinates to be used in the pairwise distance
        n_atoms = coordinates.shape[-2]
        flattened_instances = coordinates.reshape(-1, n_atoms, 3)
        n_instances = flattened_instances.shape[0]
        flattened_coordinates = flattened_instances.reshape(-1, 3)
        # print(f"natoms: {n_atoms}")
        # print(f"flattened_instances: {flattened_instances.shape}")
        # print(f"n_instances: {n_instances}")
        # print(f"flattened_coordinates: {flattened_coordinates.shape}")
        
        # print(f"Flattened Coordinates: {flattened_coordinates.shape}")
        # print(f"Number of Instances: {n_instances}")
        # print(f"Number of Atoms: {n_atoms}")
        for bond in bonded_indices:
            if bond[0] >= n_atoms or bond[1] >= n_atoms:
                raise ValueError(
                    f"Invalid bond indices: {bond} for {n_atoms} atoms")
        

        # Mask the non-bonded atoms
        mask1 = torch.zeros(len(bonded_indices), device=model_output.device)
        mask2 = torch.zeros(len(bonded_indices), device=model_output.device)
        cov_distances = torch.zeros(
            len(bonded_indices), device=model_output.device)

        for ind, (i, j) in enumerate(bonded_indices):
            mask1[ind] = i
            mask2[ind] = j
            cov_distances[ind] = self.cov_mat[i, j]
        # print(f"cov_distances shape: {cov_distances.shape}")
        # print(f"cov_distances: {cov_distances}")
        mask1 = mask1.repeat(n_instances)
        mask2 = mask2.repeat(n_instances)
        cov_distances = cov_distances.repeat(n_instances)
        set1 = flattened_coordinates[mask1.long()]
        set2 = flattened_coordinates[mask2.long()]

        # print(f"Set1: {set1.shape}")
        # print(f"Set2: {set2.shape}")

        # Calculate the pairwise distance between the bonded atoms
        dist = pairwise_distance(set1, set2)
        # print(f"Distance shape: {dist.shape}")
        # exit()


        deviation = (dist - cov_distances) ** 2
        # print(f"Deviation: {deviation[:len(bonded_indices)]}")
        # print(f"cov_distances: {cov_distances[:len(bonded_indices)]}")
        # exit()
        
        # print(f"\nDeviation: {deviation.mean()}")
        return deviation.mean()
    
    def compute_steric_loss(self, model_output: torch.Tensor):
        if True:
            weights = torch.min((1 - self.quantiles_tensor), self.quantiles_tensor)
            weights = weights / weights.sum()
            model_output = torch.sum(model_output * weights, dim=-1)
        else:
            model_output = self.sample(model_output)
            
        if self.encoder is not None:
            # Get the coordinates from the latent space of the encoder
            if self.encoder.device != model_output.device:
                model_output = model_output.to(self.encoder.device)
            if self.cov_mat.device != model_output.device:
                self.cov_mat = self.cov_mat.to(model_output.device)
            coordinates = self.encoder.decode_latent(
                model_output.view(-1, model_output.shape[-1]), keeptensor=True)
        else:
            coordinates = model_output.view(model_output.shape[0], model_output.shape[1], -1, 3)
        # print(f"Coordinates: {coordinates}")
        
        # Reshape the coordinates to get the flattened coordinates to be used in the pairwise distance
        n_atoms = coordinates.shape[-2]
        flattened_instances = coordinates.reshape(-1, n_atoms, 3)
        n_instances = flattened_instances.shape[0]
        flattened_coordinates = flattened_instances.reshape(-1, 3)
        # print(f"natoms: {n_atoms}")
        # print(f"flattened_instances: {flattened_instances.shape}")
        # print(f"n_instances: {n_instances}")
        # print(f"flattened_coordinates: {flattened_coordinates.shape}")
        
        # print(f"Flattened Coordinates: {flattened_coordinates.shape}")
        # print(f"Number of Instances: {n_instances}")
        # print(f"Number of Atoms: {n_atoms}")

        n_pairs = n_atoms * (n_atoms - 1)
        mask1 = torch.zeros(n_pairs, device=model_output.device)
        mask2 = torch.zeros(n_pairs, device=model_output.device)
        cov_distances = torch.zeros(
            n_pairs, device=model_output.device)

        # for ind, (i, j) in enumerate(bonded_indices):
        ind = 0
        for i in range(n_atoms):
            for j in range(n_atoms):
                if i == j:
                    continue
                mask1[ind] = i
                mask2[ind] = j
                cov_distances[ind] = self.cov_mat[i, j]
                ind += 1
        mask1 = mask1.repeat(n_instances)
        mask2 = mask2.repeat(n_instances)
        cov_distances = cov_distances.repeat(n_instances)
        set1 = flattened_coordinates[mask1.long()]
        set2 = flattened_coordinates[mask2.long()]

        # print(f"Set1: {set1.shape}")
        # print(f"Set2: {set2.shape}")

        # Calculate the pairwise distance between the bonded atoms
        dist = pairwise_distance(set1, set2)

        steric_mask = torch.where(dist > 0.7 * cov_distances, torch.zeros_like(dist), torch.ones_like(dist))
        
        strain = (dist - cov_distances) ** 2 * steric_mask
        # print(f"Deviation: {deviation[:len(bonded_indices)]}")
        # print(f"cov_distances: {cov_distances[:len(bonded_indices)]}")
        # exit()
        
        # print(f"\nDeviation: {deviation.mean()}")
        return strain.mean()

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

        losses = losses.mean()
        
        ## Bond deviation loss 
        # print(f"Model Output: {model_output.shape}")
        # print(f"Model Output eg: {model_output[0, 0, 0]}")
        # print(f"Model Output eg: {model_output[0, 0, 0] * self.quantiles_tensor}")
        # print(f"Model Output eg: {model_output[0, 0, 0] * (self.quantiles_tensor - 1)}")
        # print(f"Model Output eg: {torch.min((1 - self.quantiles_tensor), self.quantiles_tensor)}")
        # exit()
        
        meta = {}
        meta["loss_likelihood"] = losses
        
        # loss_steric = self.compute_steric_loss(model_output) * 1000.0
        # meta["loss_steric"] = loss_steric
        # losses = losses + loss_steric
        
        # loss_deviation = self.compute_bondlength_deviation_loss(model_output) * 1000.0
        # meta["loss_deviation"] = loss_deviation
        # losses = losses + loss_deviation * 100.0

        return losses, meta
