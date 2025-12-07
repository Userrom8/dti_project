# src/inference/preprocess.py

from src.utils.mol_utils import mol_to_pyg


def prepare_inputs(smiles: str, protein: str):
    g = mol_to_pyg(smiles)
    if g is None:
        raise ValueError("Invalid SMILES string.")

    return g, protein
