# src/models/dti_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Union

from torch_geometric.nn import GINConv, global_mean_pool

from transformers import AutoTokenizer, AutoModel


# ----------------------------
# Protein Encoder (ESM wrapper)
# ----------------------------
class ProteinEncoderESM(nn.Module):
    def __init__(
        self,
        model_name: str = "facebook/esm2_t6_8M_UR50D",
        device: Union[str, torch.device] = "cuda",
        freeze: bool = True,
    ):
        super().__init__()
        # accept both torch.device and str
        self.device = torch.device(device if not isinstance(device, str) else device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)

        if freeze:
            for p in self.model.parameters():
                p.requires_grad = False

    def forward(self, seqs: List[str]) -> torch.Tensor:
        """
        seqs: list of protein sequences (strings)
        returns: (B, hidden_dim) tensor on self.device
        """
        enc = self.tokenizer(
            seqs, return_tensors="pt", padding=True, truncation=True, max_length=1024
        ).to(self.device)

        out = self.model(**enc)
        cls = out.last_hidden_state[:, 0, :]  # (B, hidden_dim)
        return cls


# ----------------------------
# Drug Encoder (GIN)
# ----------------------------
class DrugEncoderGNN(nn.Module):
    def __init__(self, node_dim: int = 30, hidden_dim: int = 128, num_layers: int = 3):
        super().__init__()

        self.convs = nn.ModuleList()
        in_dim = node_dim
        for _ in range(num_layers):
            mlp = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            conv = GINConv(mlp)
            self.convs.append(conv)
            in_dim = hidden_dim

        self.hidden_dim = hidden_dim

    def forward(self, batch):
        """
        batch: pyg Batch with attributes x, edge_index, batch
        returns: graph-level embedding (B, hidden_dim)
        """
        x, edge_index, batch_idx = batch.x, batch.edge_index, batch.batch

        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)

        graph_emb = global_mean_pool(x, batch_idx)  # (B, hidden_dim)
        return graph_emb


# ----------------------------
# Attention Fusion
# ----------------------------
class AttentionFusion(nn.Module):
    def __init__(self, drug_dim: int, protein_dim: int, hidden_dim: int):
        super().__init__()
        # drug_dim should match the projected drug dim (fusion_hidden)
        self.Wq = nn.Linear(drug_dim, hidden_dim)
        self.Wk = nn.Linear(protein_dim, hidden_dim)
        self.Wv = nn.Linear(protein_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(
        self, drug_vec: torch.Tensor, protein_vec: torch.Tensor
    ) -> torch.Tensor:
        """
        drug_vec: (B, Dd)
        protein_vec: (B, Dp)
        returns: (B, hidden_dim)
        """
        Q = self.Wq(drug_vec)  # (B, H)
        K = self.Wk(protein_vec)  # (B, H)
        V = self.Wv(protein_vec)  # (B, H)

        # scaled elementwise attention score -> scalar per sample
        att_score = (Q * K).sum(dim=-1, keepdim=True) / (Q.size(-1) ** 0.5)  # (B,1)
        att_weight = torch.sigmoid(att_score)  # (B,1) in (0,1)

        # attended protein representation
        attended = att_weight * V  # (B,H)

        fused = (
            self.out_proj(attended) + drug_vec
        )  # project and residually add drug signal
        return fused  # (B, H)


# ----------------------------
# Residual Block for Predictor
# ----------------------------
class ResidualBlock(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.2):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(dim, dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(dim, dim)
        )
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.fc(x))


# ----------------------------
# Full DTI Network
# ----------------------------
class DTINetwork(nn.Module):
    def __init__(
        self,
        drug_in_dim: int,
        drug_hidden_dim: int = 128,
        drug_layers: int = 3,
        esm_model: str = "facebook/esm2_t6_8M_UR50D",
        esm_device: Union[str, torch.device] = "cuda",
        esm_freeze: bool = True,
        fusion_hidden: int = 256,
        dropout: float = 0.2,
    ):
        super().__init__()

        # Drug encoder (GIN)
        self.drug_encoder = DrugEncoderGNN(
            node_dim=drug_in_dim, hidden_dim=drug_hidden_dim, num_layers=drug_layers
        )
        drug_out_dim = self.drug_encoder.hidden_dim  # equals drug_hidden_dim

        # Protein encoder (ESM) - still constructed so we can fallback to runtime ESM if no cache
        self.protein_encoder = ProteinEncoderESM(
            model_name=esm_model, device=esm_device, freeze=esm_freeze
        )
        protein_out_dim = 320  # ESM t6_8M hidden size (adjust if you change model)

        # Project drug embedding to fusion dimension
        self.proj_drug = nn.Linear(drug_out_dim, fusion_hidden)

        # Attention fusion: expect drug_dim == fusion_hidden
        self.att_fusion = AttentionFusion(
            drug_dim=fusion_hidden,
            protein_dim=protein_out_dim,
            hidden_dim=fusion_hidden,
        )

        # Predictor with residual blocks
        self.fc_in = nn.Linear(fusion_hidden, fusion_hidden)
        self.res1 = ResidualBlock(fusion_hidden, dropout=dropout)
        self.res2 = ResidualBlock(fusion_hidden, dropout=dropout)
        self.out = nn.Linear(fusion_hidden, 1)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self, graph_batch, seqs: Union[torch.Tensor, List[str]]
    ) -> torch.Tensor:
        """
        graph_batch: pyg Batch
        seqs: either
            - a tensor of shape (B, protein_dim)  -> cached embeddings
            - a list of strings (length B)         -> raw sequences (ESM will be used)
        returns: (B,) tensor of predictions
        """
        # Drug embedding
        drug_vec = self.drug_encoder(graph_batch)  # (B, drug_out_dim)

        # Project drug to fusion dimension
        drug_proj = self.proj_drug(drug_vec)  # (B, fusion_hidden)

        # Protein embedding handling
        if torch.is_tensor(seqs):
            protein_vec = seqs.to(drug_proj.device)
        else:
            protein_vec = self.protein_encoder(seqs).to(drug_proj.device)

        # Fusion via attention (expects drug_proj dim == fusion_hidden)
        fused = self.att_fusion(drug_proj, protein_vec)  # (B, fusion_hidden)

        x = self.fc_in(fused)
        x = F.relu(x)
        x = self.res1(x)
        x = self.res2(x)
        x = self.dropout(x)

        out = self.out(x).squeeze(-1)  # (B,)
        return out
