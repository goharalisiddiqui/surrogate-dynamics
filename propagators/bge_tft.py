import os
import math
import argparse
from typing import Callable, List, Optional

import numpy as np

from ase.data import covalent_radii

import torch
from torch import nn, Tensor
import torch.nn.functional as F

from torch_geometric.data import Dataset
from torch_geometric.nn import MessagePassing, Set2Set
from torch_geometric.utils import softmax

import pytorch_lightning as pl

from propagators.tft_wrapper import TFTModel

from darts.utils.likelihood_models import QuantileRegression




class ScalarFeatureEmbedding(nn.Module):
    """Applies one independent MLP per scalar feature dimension and sums outputs.

    Given input x of shape (N, F), we create F small MLPs each processing x[:, f:f+1].
    Each MLP outputs (N, hidden_dim); final embedding h is the (optionally scaled) sum.
    """
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int = 128, activation=nn.ELU()):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(1, hidden_dim),
                activation,
                nn.Linear(hidden_dim, out_dim),
            )
            for _ in range(in_dim)
        ])

    def forward(self, x: Tensor) -> Tensor:
        # x: (N, F)
        outs = []
        for f, mlp in enumerate(self.mlps):
            outs.append(mlp(x[:, f:f+1]))
        h = torch.stack(outs, dim=0).sum(dim=0)  # (N, hidden_dim)
        h = h / math.sqrt(self.in_dim)  # scale
        return h


class AttentionMP(MessagePassing):
    """Multi-head dot-product attention message passing layer with edge features.

    Attention score per head: a_ij^h = (q_i^h · k_j^h)/sqrt(d) + b_e^h
    where b_e^h is a learned scalar bias from transformed edge features.
    Message: m_ij^h = a_ij^h * (v_j^h + e_msg_ij^h)
    Aggregation: sum over j -> i
    Output: residual + linear projection + optional norm & activation handled externally.
    """
    def __init__(self, node_feat_dim: int, edge_feat_dim: int, hidden_dim: int = 128, heads: int = 4, dropout: float = 0.0):
        super().__init__(aggr='add', node_dim=0)
        assert hidden_dim % heads == 0, "hidden_dim must be divisible by heads"
        self.node_feat_dim = node_feat_dim
        self.edge_feat_dim = edge_feat_dim
        self.hidden_dim = hidden_dim
        self.heads = heads
        self.d_head = hidden_dim // heads
        self.dropout = dropout

        self.q_proj = nn.Linear(node_feat_dim, hidden_dim, bias=False)
        self.k_proj = nn.Linear(node_feat_dim, hidden_dim, bias=False)
        self.v_proj = nn.Linear(node_feat_dim, hidden_dim, bias=False)

        self.node_msg = nn.Linear(node_feat_dim, hidden_dim, bias=False)
        self.edge_msg = nn.Linear(edge_feat_dim, hidden_dim, bias=False)

        self.beta = nn.Sequential(
            nn.Linear(hidden_dim*3, 1, bias=False), 
            nn.Sigmoid()
        )

        self.attn_drop = nn.Dropout(dropout)

    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Optional[Tensor]) -> Tensor:
        if edge_index.numel() == 0:
            return x  # no edges

        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_i: Tensor, x_j: Tensor, edge_attr: Tensor, index: Tensor) -> Tensor:
        # x_i, x_j: (E, hidden_dim)
        q = self.q_proj(x_i).view(-1, self.heads, self.d_head)
        k = self.k_proj(x_j).view(-1, self.heads, self.d_head)
        v = self.v_proj(x_j).view(-1, self.heads, self.d_head)

        e_msg = self.edge_msg(edge_attr).view(-1, self.heads, self.d_head)  # (E, heads, d_head)

        k_e = k + e_msg  # (E, heads, d_head)

        logits = (q * k_e).sum(dim=-1) / math.sqrt(self.d_head)  # (E, heads)

        alpha = softmax(logits, index)  # softmax over incoming edges per target node
        alpha = self.attn_drop(alpha)
        alpha = alpha.unsqueeze(-1)  # (E, heads, 1)

        msg = alpha * (v + e_msg)  # (E, heads, d_head)

        msg = msg.view(-1, self.hidden_dim)  # (E, hidden_dim)
              
        return msg

    def update(self, aggr_out: Tensor, x_i: Tensor, index: Tensor) -> Tensor:
        # aggr_out: (N, heads, d_head)
        n_msg_all = self.node_msg(x_i)
        n_msg = torch.zeros_like(aggr_out)
        n_msg.index_add_(0, index, n_msg_all)

        beta = self.beta(torch.cat([n_msg, aggr_out, n_msg - aggr_out], dim=-1))  # (N, heads, 1)
        out = beta * n_msg + (1 - beta) * aggr_out  # (N, heads, d_head)
        out = out.view(-1, self.hidden_dim)  # (N, hidden_dim)
        return out

class BondGraphNetEncoder(nn.Module):
    """Bond-based message passing GNN Encoder with multi-head attention and Set2Set pooling.

    Steps:
      1. Per-scalar feature MLP embeddings summed to initial node embedding h.
      2. L attention message passing layers (each with residual, BatchNorm, ELU).
      3. Set2Set pooling over nodes for T processing steps.
      4. Final linear layer maps pooled embedding to latent_dim.

    Args:
        in_features: Number of scalar node features (default 3 for bond nodes).
        edge_dim: Edge feature dimension (default 3: angle one-hot + value).
        hidden_dim: Hidden embedding size.
        num_layers: Number of message passing layers (L).
        heads: Attention heads.
        set2set_steps: T processing steps for Set2Set.
        latent_dim: Output latent embedding size.
        dropout: Dropout applied to attention coefficients.
    """
    def __init__(
        self,
        node_feat: int = 3,
        edge_feat: int = 3,
        node_embed_dim: int = 10,
        edge_embed_dim: int = 2,
        hidden_dim: int = 128,
        num_layers: int = 4,
        heads: int = 4,
        set2set_steps: int = 3,
        latent_dim: int = 256,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.node_embed_dim = node_embed_dim
        self.edge_embed_dim = edge_embed_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        self.node_embed = ScalarFeatureEmbedding(node_feat, node_embed_dim)
        self.edge_embed = ScalarFeatureEmbedding(edge_feat, edge_embed_dim)

        self.node_dim = node_embed_dim #* node_feat
        self.edge_dim = edge_embed_dim #* edge_feat

        self.layers = nn.ModuleList([
            AttentionMP(node_feat_dim=self.node_dim if i == 0 else hidden_dim, 
                        edge_feat_dim=self.edge_dim, 
                        hidden_dim=hidden_dim, 
                        heads=heads, 
                        dropout=dropout)
            for i in range(num_layers)
        ])
        self.bns = nn.ModuleList([nn.BatchNorm1d(hidden_dim) for _ in range(num_layers)])
        self.elu = nn.ELU()
        self.set2set = Set2Set(hidden_dim, processing_steps=set2set_steps, num_layers=1)
        self.readout = nn.Linear(2 * hidden_dim, latent_dim)

    def forward(self, data) -> Tensor:
        x, edge_index, edge_attr = data.x, data.edge_index, getattr(data, 'edge_attr', None)
        if edge_attr is None:
            # create zero edge features if missing
            edge_attr = torch.zeros(edge_index.size(1), self.edge_dim, device=x.device, dtype=x.dtype)
        batch = getattr(data, 'batch', None)
        if batch is None:
            batch = x.new_zeros(x.size(0), dtype=torch.long)

        h = self.node_embed(x)
        e = self.edge_embed(edge_attr)
        for mp, bn in zip(self.layers, self.bns):
            h = mp(h, edge_index, e)
            h = bn(h)
            h = self.elu(h)

        pooled = self.set2set(h, batch)  # (batch_size, 2*hidden_dim)
        latent = self.readout(pooled)
        return latent

def bgnd_parse_args():
    desc = "GRAPH-DECODER Arguments"
    parser = argparse.ArgumentParser(description=desc)


    args, _ = parser.parse_known_args()

    return args


BGND_args = bgnd_parse_args

class BondGraphNetDecoder(nn.Module):
    """Bond-based message passing GNN decoder with multi-head attention.

    Steps:
      1. During initialization, a precomputed graph template is provided
      2. The second feature of the edge attributes first embedded using radial basis functions
      3. Then all the node and edge features are embedded using a individial MLPs like the encoder
      4. L attention message passing layers (each with residual, BatchNorm, ELU).
      5. The steps uptill now are only done once during initialization
      6. During forward pass, we get the latent vector for the batch only.
      7. We concatenate this latent vector and the node features from selected nodes after 
         the message passing layers to get 4 vectors of interest
      8. The first vector denotes the bond distance and is calculated between any two 
         adjacent nodes. We concatenate the final node features of the two nodes and the 
         latent vector to get the input
      9. The second vector denotes the angle and is calculated between any three adjacent nodes
      10. The third vector denotes the cosine of the dihedral angle and is calculated between any four 
          adjacent nodes
      11. The fourth vector denotes the sine of the dihedral angle and is calculated between any four 
          adjacent nodes
      12. Each of these vectors is then passed through a two-layer MLP with ELU activations 
          and dropout to map to a single scalar output

    Args:
        template_data: PyG Data object (atom-level or bond-level template graph)
        latent_dim: Output latent embedding size.
        hidden_dim: Hidden embedding size.
        num_layers: Number of message passing layers (L).
        heads: Attention heads.
        dropout: Dropout applied to attention coefficients.
        rbf_dim: Number of radial basis functions for distance embedding.
        rbf_min: Minimum distance for RBFs.
        rbf_max: Maximum distance for RBFs.
        rbf_gamma: Width parameter for RBFs.
        precompute: Whether to precompute the structural embedding of the template graph.
    """
    def __init__(
        self,
        template_data,  # PyG Data object (atom-level or bond-level template graph)
        latent_dim: int,
        label_indices: Optional[tuple],  # (bond_index, angle_index, torsion_index)
        node_embed_dim: int = 10,
        edge_embed_dim: int = 2,
        hidden_dim: int = 128,
        num_layers: int = 4,
        heads: int = 4,
        dropout_mp: float = 0.0,
        dropout_mlp: float = 0.0,
        final_mlp_layers: int = 3,
        rbf_dim: int = 16,
        rbf_min: float = 0.0,
        rbf_max: float = 4.0,
        rbf_gamma: float = 10.0,
        precompute: bool = True,
        out_labels: List[str] = ['bond_dist', 'angle', 'dihedral_cos', 'dihedral_sin'],
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.bond_index = np.array(label_indices[0])
        self.angle_index = np.array(label_indices[1])
        self.torsion_index = np.array(label_indices[2])
        self.node_embed_dim = node_embed_dim
        self.edge_embed_dim = edge_embed_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.heads = heads
        self.dropout_mp = dropout_mp
        self.dropout_mlp = dropout_mlp
        self.final_mlp_layers = final_mlp_layers
        self.rbf_dim = rbf_dim
        self.register_buffer('rbf_centers', torch.linspace(rbf_min, rbf_max, rbf_dim))
        self.rbf_gamma = rbf_gamma
        self.precompute = precompute
        self.out_labels = out_labels

        # ---------- Template graph processing ----------
        # Expect edge_attr: [bond_type_one_hot(5), bond_length]
        assert hasattr(template_data, 'edge_attr'), "template_data must have edge_attr"
        edge_attr_raw: Tensor = template_data.edge_attr.clone().detach()
        assert edge_attr_raw.size(1) >= 6, "edge_attr must contain 5 bond-type one-hot + length"
        # Ensure consistent dtype (float32 default) for features
        desired_dtype = torch.get_default_dtype()
        bond_types = edge_attr_raw[:, :5].to(desired_dtype)
        bond_len = edge_attr_raw[:, 5].to(desired_dtype)
        # RBF embedding of bond lengths
        bond_rbf = self._rbf_embed(bond_len)
        edge_features = torch.cat([bond_types, bond_rbf], dim=-1)  # (E, 5+rbf_dim)

        # Per-feature MLP embedding for nodes and edges
        self.node_embed = ScalarFeatureEmbedding(in_dim=template_data.x.size(1), out_dim=node_embed_dim)
        self.edge_embed = ScalarFeatureEmbedding(in_dim=edge_features.size(1), out_dim=edge_embed_dim)

        self.node_dim = node_embed_dim #* template_data.x.size(1)
        self.edge_dim = edge_embed_dim #* edge_features.size(1)
        # Attention message passing layers (edge_dim = hidden_dim after embedding)
        self.mp_layers = nn.ModuleList([
            AttentionMP(node_feat_dim=self.node_dim if i == 0 else hidden_dim, 
                        edge_feat_dim=self.edge_dim, 
                        hidden_dim=hidden_dim, 
                        heads=heads, 
                        dropout=dropout_mp)
            for i in range(num_layers)
        ])
        self.bns = nn.ModuleList([nn.BatchNorm1d(hidden_dim) for _ in range(num_layers)])
        self.elu = nn.ELU()

        # Precompute structural embedding (optional)
        if precompute:
            with torch.no_grad():
                h = self.node_embed(template_data.x)
                e = self.edge_embed(edge_features)
                # print("Precomputing template node representations with shape:", h.shape)
                # print("Precomputing template edge representations with shape:", e.shape)
                # print(self.node_dim, self.edge_dim)
                # print(template_data.x.size(1), edge_features.size(1))
                # print(template_data.x.size(0), edge_features.size(0))
                # exit()
                for mp, bn in zip(self.mp_layers, self.bns):
                    h = mp(h, template_data.edge_index, e)
                    h = bn(h)
                    h = self.elu(h)
                self.register_buffer('template_node_repr', h)
        else:
            self.template_node_repr = None  # type: ignore
        # Always store processed edge features for reuse
        self.register_buffer('template_edge_features', edge_features)
        self.register_buffer('template_edge_index', template_data.edge_index.clone())

        # ---------- Build combinatorial sets (bonds, angles, dihedrals) ----------
        # self._build_topology_sets(template_data)

        # ---------- Prediction heads ----------
        bond_in = 2 * hidden_dim + latent_dim
        angle_in = 3 * hidden_dim + latent_dim
        dihedral_in = 4 * hidden_dim + latent_dim

        def head(in_dim, n_layers=2):
            layers = []
            for _ in range(n_layers):
                layers.append(nn.Linear(in_dim, hidden_dim))
                layers.append(nn.ELU())
                layers.append(nn.Dropout(dropout_mlp))
                in_dim = hidden_dim
            layers.append(nn.Linear(hidden_dim, 1))
            return nn.Sequential(*layers)

        self.bond_head = head(bond_in, final_mlp_layers)
        self.angle_head = head(angle_in, final_mlp_layers)
        self.dih_cos_head = head(dihedral_in, final_mlp_layers)
        self.dih_sin_head = head(dihedral_in, final_mlp_layers)

    # ------------------------------------------------------------------
    def _rbf_embed(self, distances: Tensor) -> Tensor:
        # distances: (E,)
        diff = distances.unsqueeze(-1) - self.rbf_centers  # (E, rbf_dim)
        return torch.exp(-self.rbf_gamma * diff * diff)

    # ------------------------------------------------------------------
    def _compute_template_repr(self, device):
        # Compute (or retrieve) node representations for template graph
        if self.template_node_repr is not None:
            return self.template_node_repr.to(device)
        # If not precomputed, run embedding graph dynamically (keeps gradients)
        x = self.node_embed(self.template_edge_index.new_tensor([]))  # placeholder (won't be used)
        raise RuntimeError("Dynamic (non-precomputed) template representation not implemented.")

    # ------------------------------------------------------------------
    def forward(self, latent: Tensor):
        """Decode latent vector(s) into structural predictions.

        Args:
            latent: (B, latent_dim) latent embeddings.

        Returns dict with keys: bond_dist, angle, dihedral_cos, dihedral_sin
        Shapes: (B, num_bonds / num_angles / num_dihedrals)
        """
        B = latent.size(0)
        H = self.hidden_dim
        device = latent.device
        template_h = self._compute_template_repr(device)  # (N, H)

        # Bonds
        bond_idx = self.bond_index
        angle_idx = self.angle_index
        dih_idx = self.torsion_index

        preds = {}
        if bond_idx.size > 0:
            h_bi = template_h[bond_idx[:,0]]
            h_bj = template_h[bond_idx[:,1]]
            bond_feat = torch.cat([h_bi, h_bj], dim=-1)  # (Nb, 2H)
            bond_feat = bond_feat.unsqueeze(0).expand(B, -1, -1)
            latent_exp = latent.unsqueeze(1).expand(-1, bond_feat.size(1), -1)
            bond_in = torch.cat([bond_feat, latent_exp], dim=-1)
            bond_out = self.bond_head(bond_in.view(-1, bond_in.size(-1))).view(B, -1)
        else:
            bond_out = latent.new_empty((B, 0))
        preds[self.out_labels[0]] = bond_out

        # Angles
        if angle_idx.size > 0:
            h_i = template_h[angle_idx[:,0]]
            h_j = template_h[angle_idx[:,1]]
            h_k = template_h[angle_idx[:,2]]
            ang_feat = torch.cat([h_i, h_j, h_k], dim=-1)  # (Na, 3H)
            ang_feat = ang_feat.unsqueeze(0).expand(B, -1, -1)
            latent_exp = latent.unsqueeze(1).expand(-1, ang_feat.size(1), -1)
            ang_in = torch.cat([ang_feat, latent_exp], dim=-1)
            angle_out = self.angle_head(ang_in.view(-1, ang_in.size(-1))).view(B, -1)
        else:
            angle_out = latent.new_empty((B, 0))
        preds[self.out_labels[1]] = angle_out

        # Dihedrals
        if dih_idx.size > 0:
            h_i = template_h[dih_idx[:,0]]
            h_j = template_h[dih_idx[:,1]]
            h_k = template_h[dih_idx[:,2]]
            h_l = template_h[dih_idx[:,3]]
            dih_feat = torch.cat([h_i, h_j, h_k, h_l], dim=-1)  # (Nd, 4H)
            dih_feat = dih_feat.unsqueeze(0).expand(B, -1, -1)
            latent_exp = latent.unsqueeze(1).expand(-1, dih_feat.size(1), -1)
            dih_in = torch.cat([dih_feat, latent_exp], dim=-1)
            cos_out = self.dih_cos_head(dih_in.view(-1, dih_in.size(-1))).view(B, -1)
            sin_out = self.dih_sin_head(dih_in.view(-1, dih_in.size(-1))).view(B, -1)
        else:
            cos_out = latent.new_empty((B, 0))
            sin_out = latent.new_empty((B, 0))
        preds[self.out_labels[2]] = cos_out
        preds[self.out_labels[3]] = sin_out

        return preds


# def bgetft_parse_args():
#     desc = "Bond-GNN-ENCODER-DECODER Arguments"
#     parser = argparse.ArgumentParser(description=desc)

#     # Encoder args
#     parser.add_argument('--node_feat', type=int, default=3, help='Number of scalar node features (default 3 for bond nodes)')
#     parser.add_argument('--edge_feat', type=int, default=3, help='Edge feature dimension (default 3: angle one-hot + value)')
#     parser.add_argument('--enc_node_embed_dim', type=int, default=10, help='Node embedding size per scalar feature')
#     parser.add_argument('--enc_edge_embed_dim', type=int, default=2, help='Edge embedding size per scalar feature')
#     parser.add_argument('--enc_hidden_dim', type=int, default=128, help='Hidden embedding size')
#     parser.add_argument('--enc_num_layers', type=int, default=4, help='Number of message passing layers (L)')
#     parser.add_argument('--enc_heads', type=int, default=4, help='Attention heads')
#     parser.add_argument('--set2set_steps', type=int, default=3, help='T processing steps for Set2Set')
#     parser.add_argument('--latent_dim', type=int, default=256, help='Output latent embedding size')
#     parser.add_argument('--enc_dropout', type=float, default=0.0, help='Dropout applied to attention coefficients')

#     # Decoder args
#     parser.add_argument('--template_khop', type=int, default=2, help='k-hop neighborhood for template graph')
#     parser.add_argument('--dec_node_embed_dim', type=int, default=10, help='Node embedding size per scalar feature for decoder')
#     parser.add_argument('--dec_edge_embed_dim', type=int, default=2, help='Edge embedding size per scalar feature for decoder')
#     parser.add_argument('--dec_hidden_dim', type=int, default=128, help='Hidden embedding size for decoder')
#     parser.add_argument('--dec_num_layers', type=int, default=4, help='Number of message passing layers (L) for decoder')
#     parser.add_argument('--dec_heads', type=int, default=4, help='Attention heads for decoder')
#     parser.add_argument('--dec_dropout_mp', type=float, default=0.0, help='Dropout applied to attention coefficients in decoder')
#     parser.add_argument('--dec_dropout_mlp', type=float, default=0.0, help='Dropout applied in MLP heads of decoder')
#     parser.add_argument('--final_mlp_layers', type=int, default=3, help='Number of layers in final MLP heads of decoder')

#     parser.add_argument('--rbf_dim', type=int, default=16, help='Number of radial basis functions for distance embedding')
#     parser.add_argument('--rbf_min', type=float, default=0.0, help='Minimum distance for RBFs')
#     parser.add_argument('--rbf_max', type=float, default=4.0, help='Maximum distance for RBFs')
#     parser.add_argument('--rbf_gamma', type=float, default=10.0, help='Width parameter for RBFs')

#     # Propagator args
#     parser.add_argument('--seqlength', dest="input_chunk_length", default=5,
#                         type=int, help='Number of time steps in the past to take as a model input')
#     parser.add_argument('--predlength', dest="output_chunk_length",
#                         default=5, type=int, help='Number of time steps to predict at once')
#     parser.add_argument('--prop_hidden_dim', default=96, type=int, 
#                         help='Hidden state size of the TFT')
#     parser.add_argument('--prop_nlstm', default=2, type=int,
#                         help='Number of layers for the LSTM Encoder and Decoder')
#     parser.add_argument('--prop_heads',
#                         default=1, type=int, help='Number of attention heads')
#     parser.add_argument('--prop_dropout', default=0.02,
#                         type=float, help='Fraction of neurons affected by dropout.')

#     # Trainer args
#     parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate for the training')
#     parser.add_argument('--loss_latent_weight', type=float, default=1e-3, help='Weight for latent loss term if used')
#     parser.add_argument('--weight_decay', type=float, default=1e-3, help='Weights regularization for the training')
#     parser.add_argument('--normalize_inputs', dest='normIn', action='store_true', help='Whether to normalize the input features')
#     parser.add_argument('--scheduler', action='store_true', help='Whether to use learning rate scheduler')

#     args, _ = parser.parse_known_args()

#     return args


# BGETFT_args = bgetft_parse_args

class BondGraphEncoderTFT(pl.LightningModule):
    """LightningModule wrapper for BondGraphNet.

    Args:
        gnn_enc_kwargs: keyword arguments passed to BondGraphNetEncoder.
        gnn_dec_kwargs: keyword arguments passed to BondGraphNetDecoder.
        target_dim: output dimension of prediction head.
        lr: learning rate.
        weight_decay: weight decay for AdamW.
        scheduler_gamma: optional exponential LR scheduler gamma (set None to disable).
        loss: loss function (defaults to MSE for regression).
    """
    def __init__(
        self,
        node_feat: int = 3, ### ENCODER ARGS
        edge_feat: int = 3,
        enc_node_embed_dim: int = 10,
        enc_edge_embed_dim: int = 2,
        enc_hidden_dim: int = 128,
        enc_num_layers: int = 4,
        enc_heads: int = 4,
        set2set_steps: int = 3,
        enc_dropout: float = 0.0,
        latent_dim: int = 256,
        datamodule = None,  ### DECODER ARGS
        template_khop: int = 2,
        dec_node_embed_dim: int = 10,
        dec_edge_embed_dim: int = 2,
        dec_hidden_dim: int = 128,
        dec_num_layers: int = 4,
        dec_heads: int = 4,
        dec_dropout_mp: float = 0.0,
        dec_dropout_mlp: float = 0.0,
        final_mlp_layers: int = 3,
        rbf_dim: int = 16,
        rbf_min: float = 0.0,
        rbf_max: float = 4.0,
        rbf_gamma: float = 10.0,
        precompute: bool = True,
        input_chunk_length: int = 5,   ### PROPAGATOR ARGS
        output_chunk_length: int = 5,
        prop_hidden_dim: int = 96,
        prop_nlstm: int = 2,
        prop_heads: int = 1,
        prop_dropout: float = 0.02,
        prop_likelihood = QuantileRegression({'quantiles': [0.5]}),
        lr: float = 1e-4,       ### OPTIMIZER ARGS
        weight_decay: float = 0.0,
        normIn: bool = False,
        scheduler: bool = False,
        loss: Optional[nn.Module] = None,
        loss_weights: Optional[List[float]] = None,
        loss_latent_weight: float = 1e-3,
        out_labels: List[str] = ['bond_dist', 'angle', 'dihedral_cos', 'dihedral_sin'],
        outname: str = './BGE_untitled/BGE_',
    ):
        super().__init__()
        assert datamodule is not None, "datamodule must be provided"
        datasetobject = datamodule.get_dataset()
        template_data = datasetobject.get_template_graph(k=template_khop)
        bond_index, angle_index, torsion_index = datasetobject.get_label_indices()
        assert len(out_labels) == 4, "out_labels must be a list of 4 strings"
        assert loss_weights is None or len(loss_weights) == 4, "loss_weights must be None or a list of 4 floats"
        if loss_weights is None:
            loss_weights = [1.0, 1.0, 1.0, 1.0]
        self.loss_weights = loss_weights
        gnn_enc_kwargs = {
            "node_feat": node_feat,
            "edge_feat": edge_feat,
            "node_embed_dim": enc_node_embed_dim,
            "edge_embed_dim": enc_edge_embed_dim,
            "hidden_dim": enc_hidden_dim,
            "num_layers": enc_num_layers,
            "heads": enc_heads,
            "set2set_steps": set2set_steps,
            "dropout": enc_dropout,
            "latent_dim": latent_dim,
        }
        gnn_dec_kwargs = {
            "template_data": template_data,
            "label_indices": (bond_index, angle_index, torsion_index),
            "latent_dim": latent_dim,
            "node_embed_dim": dec_node_embed_dim,
            "edge_embed_dim": dec_edge_embed_dim,
            "hidden_dim": dec_hidden_dim,
            "num_layers": dec_num_layers,
            "heads": dec_heads,
            "dropout_mp": dec_dropout_mp,
            "dropout_mlp": dec_dropout_mlp,
            "final_mlp_layers": final_mlp_layers,
            "rbf_dim": rbf_dim,
            "rbf_min": rbf_min,
            "rbf_max": rbf_max,
            "rbf_gamma": rbf_gamma,
            "precompute": precompute,
            "out_labels": out_labels,
        }
        
        self.save_hyperparameters(ignore=["loss", "datasetobject"])
        self.gnn_enc = BondGraphNetEncoder(**gnn_enc_kwargs)
        self.gnn_dec = BondGraphNetDecoder(**gnn_dec_kwargs)

        # Initialize propagator
        (
            variables_meta, 
            n_static_components, 
            categorical_embedding_sizes, 
            output_dim
        ) = TFTModel.collect_meta(
            input_chunk_length = input_chunk_length,
            output_chunk_length = output_chunk_length,
            n_past_covariates = 0,
            n_future_covariates = 0,
            n_static_categoricals = 0,
            n_targets = latent_dim,
            add_relative_index = False,
            likelihood = QuantileRegression,
        )
        self.likelihood = prop_likelihood
        prop_keywargs = {
            "output_dim": output_dim,
            "variables_meta": variables_meta,
            "num_static_components": n_static_components,
            "hidden_size": prop_hidden_dim,
            "lstm_layers": prop_nlstm,
            "dropout": prop_dropout,
            "num_attention_heads": prop_heads,
            "full_attention": False,
            "feed_forward": "GatedResidualNetwork",
            "hidden_continuous_size": 8,
            "categorical_embedding_sizes": categorical_embedding_sizes,
            "add_relative_index": True,
            "norm_type": 'LayerNorm',
        }
        self.propagator = TFTModel(**prop_keywargs)
        self.loss_fn = loss if loss is not None else nn.MSELoss()
        # Normalization flag & statistics (avoid name clash with method normalize())
        self.register_buffer('normIn', torch.tensor(normIn, dtype=torch.bool))
        self.register_buffer('normSet', torch.tensor(False, dtype=torch.bool))
        # self.normIn = normIn
        # self.normSet = False
        # Register buffers for feature-wise mean/range; sized by encoder input features
        self.register_buffer('Mean', torch.zeros(node_feat + edge_feat))
        self.register_buffer('Range', torch.ones(node_feat + edge_feat))

    def set_norm(self):
        if not self.trainer.datamodule:
            raise RuntimeError("Trainer datamodule not found; cannot compute normalization.")
        with torch.no_grad():
            Mean = torch.tensor(self.trainer.datamodule.get_scaler_mean(), device=self.device)
            Range = torch.tensor(self.trainer.datamodule.get_scaler_scale(), device=self.device)
            assert Mean.size(0) == self.hparams.node_feat + self.hparams.edge_feat, \
                f"Mean size {Mean.size(0)} does not match expected {(self.hparams.node_feat + self.hparams.edge_feat)}"
            assert Range.size(0) == self.hparams.node_feat + self.hparams.edge_feat, \
                f"Range size {Range.size(0)} does not match expected {(self.hparams.node_feat + self.hparams.edge_feat)}"
            Range = Range.clone()
            Range[Range == 0.0] = 1.0
            print(f"[{type(self).__name__}] Setting normalization for inputs.")
            self.Mean = Mean
            self.Range = Range
            self.normSet = torch.tensor(True, dtype=torch.bool)
    
    def normalize(self, data):
        """Normalize a PyG Data object's node & edge attributes in-place.

        Expects stored Mean/Range concatenated as [node_feats, edge_feats].
        If called multiple times, skips when already normalized (flag _normalized).
        Falls back to tensor behavior if a plain tensor is passed.
        """
        if not self.normIn:
            return data
        if not self.normSet:
            self.set_norm()

        # Graph Data object normalization
        if getattr(data, '_normalized', False):
            return data

        node_dim = self.hparams.node_feat
        edge_dim = self.hparams.edge_feat
        mean_node = self.Mean[:node_dim]
        range_node = self.Range[:node_dim]
        mean_edge = self.Mean[node_dim:node_dim+edge_dim]
        range_edge = self.Range[node_dim:node_dim+edge_dim]

        # Normalize node features
        if hasattr(data, 'x') and data.x is not None:
            if data.x.size(-1) != node_dim:
                raise ValueError(f"Node feature dim mismatch: data.x={data.x.size(-1)} expected={node_dim}")
            if data.x.dtype != mean_node.dtype:
                mean_node = mean_node.to(data.x.dtype)
                range_node = range_node.to(data.x.dtype)
            data.x = (data.x - mean_node.view(1, -1)) / range_node.view(1, -1)

        # Normalize edge features (only first edge_dim columns if extra exist)
        if hasattr(data, 'edge_attr') and data.edge_attr is not None:
            if data.edge_attr.size(-1) < edge_dim:
                raise ValueError(f"Edge feature dim mismatch: edge_attr={data.edge_attr.size(-1)} expected>={edge_dim}")
            ea = data.edge_attr
            if ea.dtype != mean_edge.dtype:
                mean_edge = mean_edge.to(ea.dtype)
                range_edge = range_edge.to(ea.dtype)
            head = (ea[:, :edge_dim] - mean_edge.view(1, -1)) / range_edge.view(1, -1)
            if ea.size(-1) > edge_dim:
                data.edge_attr = torch.cat([head, ea[:, edge_dim:]], dim=-1)
            else:
                data.edge_attr = head

        setattr(data, '_normalized', True)
        return data
    
    def denormalize(self, data):
        """Inverse of normalize for a Data object or tensor.

        Only reverses if object was previously normalized (or is a tensor).
        """
        if not self.normIn:
            return data
        if not self.normSet:
            self.set_norm()

        if not getattr(data, '_normalized', False):
            return data

        node_dim = self.hparams.node_feat
        edge_dim = self.hparams.edge_feat
        mean_node = self.Mean[:node_dim]
        range_node = self.Range[:node_dim]
        mean_edge = self.Mean[node_dim:node_dim+edge_dim]
        range_edge = self.Range[node_dim:node_dim+edge_dim]

        if hasattr(data, 'x') and data.x is not None:
            if data.x.dtype != mean_node.dtype:
                mean_node = mean_node.to(data.x.dtype)
                range_node = range_node.to(data.x.dtype)
            data.x = data.x * range_node.view(1, -1) + mean_node.view(1, -1)
        if hasattr(data, 'edge_attr') and data.edge_attr is not None:
            ea = data.edge_attr
            if ea.size(-1) >= edge_dim:
                if ea.dtype != mean_edge.dtype:
                    mean_edge = mean_edge.to(ea.dtype)
                    range_edge = range_edge.to(ea.dtype)
                head = ea[:, :edge_dim] * range_edge.view(1, -1) + mean_edge.view(1, -1)
                if ea.size(-1) > edge_dim:
                    data.edge_attr = torch.cat([head, ea[:, edge_dim:]], dim=-1)
                else:
                    data.edge_attr = head
        setattr(data, '_normalized', False)
        return data

    def forward(self, data):  # inference

        data = self.normalize(data)
        latent = self.gnn_enc(data)



        latent = self.propagator(latent)

        pred = self.gnn_dec(latent)
        # pred = self.denormalize(pred)
        return pred, latent

    def extract_labels(self, batch):
        """extract target labels from a batch.
        """

        num_graphs = batch.batch.max().item() + 1
        
        labels = {}

        labels[self.hparams['out_labels'][0]] = batch.y_bonds.view(num_graphs, -1)
        labels[self.hparams['out_labels'][1]] = batch.y_angles.view(num_graphs, -1)
        labels[self.hparams['out_labels'][2]] = batch.y_torsions_cos.view(num_graphs, -1)
        labels[self.hparams['out_labels'][3]] = batch.y_torsions_sin.view(num_graphs, -1)

        return labels

    def step(self, batch, stage: str):
        pred, latent = self.forward(batch)
        labels = self.extract_labels(batch)

        # print("PREDICTIONS:")
        # for k, v in pred.items():
        #     print(f"{k}: Shape {v.shape}")
        # print("LABELS:")
        # for k, v in labels.items():
        #     print(f"{k}: Shape {v.shape}")
        # exit()

        loss_latent = torch.tensor(0.0, device=latent.device)
        # To make dynamics in the latent space more "modellable" (For this to work, points in a batch must be sequentially related)
        if latent.size(0) > 1:
            batch_dist = torch.norm(latent[1:] - latent[:-1], dim=1)
            # 1. We try to make the subsequent latent space points in a batch close to each other
            loss_latent = torch.mean(batch_dist)
        if latent.size(0) > 2:
            # 2. We try to keep the subsequent latent space points in a batch equidistant
            loss_latent += torch.var(batch_dist)

        self.log(f"{stage}_loss_latent", loss_latent, prog_bar=False, on_step=(stage=="train"), on_epoch=True)

        loss = self.loss_fn(pred[self.hparams['out_labels'][0]], labels[self.hparams['out_labels'][0]]) * self.loss_weights[0], \
               self.loss_fn(pred[self.hparams['out_labels'][1]], labels[self.hparams['out_labels'][1]]) * self.loss_weights[1], \
                self.loss_fn(pred[self.hparams['out_labels'][2]], labels[self.hparams['out_labels'][2]]) * self.loss_weights[2], \
                self.loss_fn(pred[self.hparams['out_labels'][3]], labels[self.hparams['out_labels'][3]]) * self.loss_weights[3]

        with torch.no_grad():
            mae = (torch.abs(pred[self.hparams['out_labels'][0]] - labels[self.hparams['out_labels'][0]]).mean() if labels[self.hparams['out_labels'][0]].numel() > 0 else torch.tensor(0.0, device=loss.device)), \
                  (torch.abs(pred[self.hparams['out_labels'][1]] - labels[self.hparams['out_labels'][1]]).mean() if labels[self.hparams['out_labels'][1]].numel() > 0 else torch.tensor(0.0, device=loss.device)), \
                  (torch.abs(pred[self.hparams['out_labels'][2]] - labels[self.hparams['out_labels'][2]]).mean() if labels[self.hparams['out_labels'][2]].numel() > 0 else torch.tensor(0.0, device=loss.device)), \
                  (torch.abs(pred[self.hparams['out_labels'][3]] - labels[self.hparams['out_labels'][3]]).mean() if labels[self.hparams['out_labels'][3]].numel() > 0 else torch.tensor(0.0, device=loss.device))

        batch_size = self.trainer.datamodule.hparams.batch_size if self.trainer and self.trainer.datamodule else None
        for i in range(len(self.hparams['out_labels'])):
            self.log(f"{stage}_{self.hparams['out_labels'][i]}_mae", mae[i], prog_bar=False, on_epoch=True, batch_size=batch_size)  
            self.log(f"{stage}_{self.hparams['out_labels'][i]}_loss", loss[i], prog_bar=False, on_step=(stage=="train"), on_epoch=True, batch_size=batch_size)
        mae = sum(mae) / len(mae)
        self.log(f"{stage}_mae", mae, prog_bar=(stage!="train"), on_epoch=True, batch_size=batch_size)
        loss = sum(loss)
        self.log(f"{stage}_loss", loss, prog_bar=(stage=="train"), on_step=(stage=="train"), on_epoch=True, batch_size=batch_size)

        return loss + self.hparams.loss_latent_weight * loss_latent

    def training_step(self, batch, batch_idx):
        return self.step(batch, "train")

    def validation_step(self, batch, batch_idx):
        self.step(batch, "val")

    def test_step(self, batch, batch_idx):
        self.step(batch, "test")

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        return self.forward(batch)

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        if self.hparams.scheduler:
            sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, 
                                                            mode='min', 
                                                           factor=0.7, 
                                                           patience=5, 
                                                           min_lr=1e-9)
            return {"optimizer": opt, "lr_scheduler": sched, "monitor": "val_loss"}
        return opt

    def get_latent(self, data):
        data = self.normalize(data)
        return self.gnn_enc(data)

    def get_decoded(self, latent):
        return self.gnn_dec(latent)

__all__ = ["BondGraphNetEncoderDecoder"]

