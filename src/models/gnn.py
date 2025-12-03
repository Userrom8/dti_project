# src/models/gnn.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, GINConv, global_mean_pool


class DrugGNN(nn.Module):
    """
    General-purpose GNN for molecular graph encoding.
    Supports GCN, GAT, or GIN layers.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 3,
        gnn_type: str = "gcn",  # "gcn", "gat", or "gin"
        dropout: float = 0.2,
    ):
        super().__init__()

        self.gnn_type = gnn_type.lower()
        self.convs = nn.ModuleList()
        self.dropout = dropout

        # First layer
        if self.gnn_type == "gcn":
            self.convs.append(GCNConv(in_dim, hidden_dim))
        elif self.gnn_type == "gat":
            self.convs.append(GATConv(in_dim, hidden_dim, heads=1))
        elif self.gnn_type == "gin":
            nn1 = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.convs.append(GINConv(nn1))
        else:
            raise ValueError("Unknown GNN type")

        # Additional layers
        for _ in range(num_layers - 1):
            if self.gnn_type == "gcn":
                self.convs.append(GCNConv(hidden_dim, hidden_dim))
            elif self.gnn_type == "gat":
                self.convs.append(GATConv(hidden_dim, hidden_dim, heads=1))
            elif self.gnn_type == "gin":
                nn_layer = nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                )
                self.convs.append(GINConv(nn_layer))

        # Final MLP projection
        self.project = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)
        )

    def forward(self, data):
        """
        data: PyG Batch
        data.x: node features
        data.edge_index: edges
        data.batch: graph index
        """

        x, edge_index, batch = data.x, data.edge_index, data.batch

        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        # Pool all node embeddings → graph embedding
        x = global_mean_pool(x, batch)

        out = self.project(x)
        return out  # shape [batch, hidden_dim]
