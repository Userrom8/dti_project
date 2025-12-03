from src.utils.mol_utils import mol_to_pyg
from src.models.gnn import DrugGNN
from torch_geometric.data import Batch
import torch

# sample molecule
g = mol_to_pyg("CCO")
if g is None:
	raise ValueError("mol_to_pyg returned None for SMILES 'CCO'")
else:
	batch = Batch.from_data_list([g, g])  # 2 molecules in batch
	model = DrugGNN(in_dim=g.x.size(1), hidden_dim=64, num_layers=3, gnn_type="gcn")

out = model(batch)
print("Output shape:", out.shape)   # should be [2, 64]
