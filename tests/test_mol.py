from src.utils.mol_utils import mol_to_pyg, draw_mol, visualize_graph

smiles = "CCO"  # ethanol

data = mol_to_pyg(smiles)

if data:
    print(data)
    if data.x is not None:
        print("Node features:", data.x.shape)
    if data.edge_index is not None:
        print("Edges:", data.edge_index.shape)

# Visualize molecule
img = draw_mol(smiles)
if img is not None:
    img.show()

# Visualize graph
visualize_graph(data)
