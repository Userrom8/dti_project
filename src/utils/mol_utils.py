from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")


import torch
from torch_geometric.data import Data
from rdkit import Chem
from rdkit.Chem import rdchem

"""
mol_utils.py
------------
SMILES → RDKit Molecule → PyTorch Geometric Graph
"""

# Allowed atom types – extendable
ATOM_LIST = [
    "H",
    "C",
    "N",
    "O",
    "F",
    "P",
    "S",
    "Cl",
    "Br",
    "I",
    "B",
    "Si",
    "Se",
    "Na",
    "K",
    "Ca",
    "Fe",
    "Zn",
    "Cu",
    "Unknown",
]

HYBRIDIZATION_LIST = [
    rdchem.HybridizationType.S,
    rdchem.HybridizationType.SP,
    rdchem.HybridizationType.SP2,
    rdchem.HybridizationType.SP3,
    rdchem.HybridizationType.SP3D,
    rdchem.HybridizationType.SP3D2,
]


def get_atom_features(atom):
    """Return a feature vector for a single atom."""
    atom_symbol = atom.GetSymbol()
    if atom_symbol not in ATOM_LIST:
        atom_symbol = "Unknown"

    # One-hot atom type
    atom_type = [int(atom_symbol == a) for a in ATOM_LIST]

    # Atom features
    degree = atom.GetTotalDegree()  # # of bonds
    formal_charge = atom.GetFormalCharge()
    implicit_valence = atom.GetImplicitValence()
    aromatic = int(atom.GetIsAromatic())

    # Hybridization one-hot
    hyb = atom.GetHybridization()
    hyb_onehot = [int(hyb == h) for h in HYBRIDIZATION_LIST]

    features = (
        atom_type + [degree, formal_charge, implicit_valence, aromatic] + hyb_onehot
    )

    return torch.tensor(features, dtype=torch.float)


def get_bond_features(bond):
    """Return one-hot encoding for bond type."""
    bond_type = bond.GetBondType()

    return torch.tensor(
        [
            int(bond_type == rdchem.BondType.SINGLE),
            int(bond_type == rdchem.BondType.DOUBLE),
            int(bond_type == rdchem.BondType.TRIPLE),
            int(bond_type == rdchem.BondType.AROMATIC),
        ],
        dtype=torch.float,
    )


def smiles_to_mol(smiles: str):
    """Convert SMILES to an RDKit Mol object (safe)."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    try:
        Chem.SanitizeMol(mol)
        return mol
    except:
        return None


def mol_to_pyg(smiles: str):
    """
    Convert a SMILES string into a PyTorch Geometric Data object.
    Returns:
        Data(x, edge_index, edge_attr)
    """
    mol = smiles_to_mol(smiles)
    if mol is None:
        return None

    # Add hydrogens for completeness
    mol = Chem.AddHs(mol)

    # Node features
    atom_features = []
    for atom in mol.GetAtoms():
        atom_features.append(get_atom_features(atom))
    x = torch.stack(atom_features, dim=0)

    # Edge index and attributes
    edge_index = []
    edge_attr = []

    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()

        bf = get_bond_features(bond)

        # Undirected edge (i → j and j → i)
        edge_index.append([i, j])
        edge_attr.append(bf)

        edge_index.append([j, i])
        edge_attr.append(bf)

    if len(edge_index) == 0:
        # No bonds case
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, 4), dtype=torch.float)
    else:
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.stack(edge_attr, dim=0)

    # Create PyG Data object
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    # Attach RDKit mol for optional visualization
    data.rdkit_mol = mol

    return data


# -----------------------------
# Optional visualization helpers
# -----------------------------


def draw_mol(smiles, size=(300, 300)):
    """Return a PIL image of the molecule."""
    from rdkit.Chem import Draw

    mol = smiles_to_mol(smiles)
    if mol:
        return Draw.MolToImage(mol, size=size)
    return None


def visualize_graph(data):
    """Visualize PyG graph structure using networkx."""
    import networkx as nx
    import matplotlib.pyplot as plt
    from torch_geometric.utils import to_networkx

    G = to_networkx(data, to_undirected=True)
    pos = nx.spring_layout(G)

    nx.draw(G, pos, with_labels=True, node_color="lightblue")
    plt.show()
