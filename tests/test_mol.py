from src.utils.mol_utils import mol_to_pyg, draw_mol, visualize_graph

smiles = "CCO"   # ethanol

data = mol_to_pyg(smiles)

print(data)
print("Node features:", data.x.shape)
print("Edges:", data.edge_index.shape)

# Visualize molecule
img = draw_mol(smiles)
img.show()

# Visualize graph
visualize_graph(data)