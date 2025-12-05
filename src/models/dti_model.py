# src/models/dti_model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool

# ----------------------------
# Protein Encoder (ESM wrapper)
# ----------------------------
from transformers import AutoTokenizer, AutoModel


class ProteinEncoderESM(nn.Module):
    def __init__(
        self, model_name="facebook/esm2_t6_8M_UR50D", device="cuda", freeze=True
    ):
        super().__init__()
        self.device = torch.device(device)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)

        if freeze:
            for p in self.model.parameters():
                p.requires_grad = False

    def forward(self, seqs):
        """
        seqs: list of strings (protein sequences)
        returns: tensor (B, hidden)
        """
        enc = self.tokenizer(
            seqs, return_tensors="pt", padding=True, truncation=True, max_length=1024
        ).to(self.device)

        out = self.model(**enc)
        cls = out.last_hidden_state[:, 0, :]  # (B, hidden_dim)
        return cls


# ----------------------------
# Drug GNN Encoder
# ----------------------------
class DrugEncoderGNN(nn.Module):
    def __init__(self, node_dim=30, hidden_dim=128, num_layers=3, gnn_type="gcn"):
        super().__init__()
        layers = []

        in_dim = node_dim
        for _ in range(num_layers):
            layers.append(GCNConv(in_dim, hidden_dim))
            in_dim = hidden_dim

        self.layers = nn.ModuleList(layers)
        self.hidden_dim = hidden_dim

    def forward(self, batch):
        x, edge_index, batch_idx = batch.x, batch.edge_index, batch.batch

        for layer in self.layers:
            x = F.relu(layer(x, edge_index))

        graph_emb = global_mean_pool(x, batch_idx)
        return graph_emb


# ----------------------------
# Full Fusion Network
# ----------------------------
class DTINetwork(nn.Module):
    def __init__(
        self,
        drug_in_dim,
        drug_hidden_dim=128,
        drug_layers=3,
        gnn_type="gcn",
        esm_model="facebook/esm2_t6_8M_UR50D",
        esm_device="cuda",
        esm_freeze=True,
        fusion_hidden=256,
        dropout=0.2,
    ):
        super().__init__()

        # ---- Drug encoder ----
        self.drug_encoder = DrugEncoderGNN(
            node_dim=drug_in_dim,
            hidden_dim=drug_hidden_dim,
            num_layers=drug_layers,
            gnn_type=gnn_type,
        )

        drug_out_dim = drug_hidden_dim

        # ---- Protein encoder ----
        # NOTE: we still build ESM, but at runtime we decide whether to call it
        self.protein_encoder = ProteinEncoderESM(
            model_name=esm_model, device=esm_device, freeze=esm_freeze
        )

        # ESM t6_8M hidden dimension = 320
        protein_out_dim = 320

        # ---- Fusion MLP ----
        fusion_in = drug_out_dim + protein_out_dim

        self.predictor = nn.Sequential(
            nn.Linear(fusion_in, fusion_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden, fusion_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden, 1),
        )

    def forward(self, graph_batch, seqs):
        # ---- Drug embedding ----
        drug_vec = self.drug_encoder(graph_batch)

        # ---- Protein embedding ----
        # If seqs is a tensor → cached embeddings
        if torch.is_tensor(seqs):
            protein_vec = seqs.to(drug_vec.device)

        else:
            # raw sequence strings → compute ESM embeddings
            protein_vec = self.protein_encoder(seqs)

        # ---- Fusion ----
        fused = torch.cat([drug_vec, protein_vec], dim=1)

        out = self.predictor(fused)
        return out.squeeze(-1)
