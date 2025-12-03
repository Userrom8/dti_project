# src/models/dti_model.py

import torch
import torch.nn as nn

try:
    from src.models.gnn import DrugGNN
    from src.models.esm_encoder import ProteinEncoderESM
except:
    # fallback when running as script
    from .gnn import DrugGNN
    from .esm_encoder import ProteinEncoderESM


class DTINetwork(nn.Module):
    """
    Main Drug–Target Interaction model.
    Combines:
        - Drug GNN embedding
        - Protein ESM2 embedding
        - Fusion MLP → Affinity score
    """

    def __init__(
        self,
        drug_in_dim: int,
        drug_hidden_dim: int = 128,
        drug_layers: int = 3,
        gnn_type: str = "gcn",
        esm_model: str = "facebook/esm2_t6_8M_UR50D",
        esm_device: str = "cpu",
        esm_freeze: bool = True,
        fusion_hidden: int = 256,
        dropout: float = 0.3,
    ):
        super().__init__()

        # ---------------------
        # Drug encoder (GNN)
        # ---------------------
        self.drug_encoder = DrugGNN(
            in_dim=drug_in_dim,
            hidden_dim=drug_hidden_dim,
            num_layers=drug_layers,
            gnn_type=gnn_type,
            dropout=dropout,
        )

        # ---------------------
        # Protein encoder (ESM2)
        # ---------------------
        self.protein_encoder = ProteinEncoderESM(
            model_name=esm_model, device=esm_device, freeze=esm_freeze
        )

        esm_dim = self.protein_encoder.output_dim

        # ---------------------
        # Fusion MLP
        # ---------------------
        self.mlp = nn.Sequential(
            nn.Linear(drug_hidden_dim + esm_dim, fusion_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden, fusion_hidden // 2),
            nn.ReLU(),
            nn.Linear(fusion_hidden // 2, 1),  # output: affinity score
        )

    def forward(self, graph_batch, protein_seqs):
        """
        graph_batch: PyG Batch object
        protein_seqs: list[str]
        """

        # Drug encoding
        drug_vec = self.drug_encoder(graph_batch)  # [B, drug_hidden]

        # Protein encoding
        protein_vec = self.protein_encoder(protein_seqs)  # [B, esm_dim]

        # Concatenate
        combined = torch.cat([drug_vec, protein_vec], dim=1)

        # Regression
        out = self.mlp(combined)

        return out.squeeze(-1)  # shape → [batch]
