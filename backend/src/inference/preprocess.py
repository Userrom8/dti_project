# src/inference/preprocess.py
import pubchempy as pcp
from src.utils.mol_utils import mol_to_pyg


def get_smiles_from_name(drug_name: str):
    """Fetches isomeric SMILES from PubChem for a given drug name."""
    try:
        compounds = pcp.get_compounds(drug_name, "name")
        if compounds:
            return compounds[0].smiles
    except Exception as e:
        print(f"PubChem lookup failed: {e}")
    return None


def prepare_inputs(smiles: str, protein: str, drug_name: str = None):
    # 1. If no SMILES provided, try to fetch from name
    if not smiles and drug_name:
        print(f"Looking up SMILES for: {drug_name}")
        fetched_smiles = get_smiles_from_name(drug_name)
        if fetched_smiles:
            smiles = fetched_smiles
        else:
            raise ValueError(f"Could not find SMILES for drug name: {drug_name}")

    # 2. Validate we have a SMILES string now
    if not smiles:
        raise ValueError("No SMILES string provided or found.")

    # 3. Convert to Graph
    g = mol_to_pyg(smiles)
    if g is None:
        raise ValueError("Invalid SMILES string (conversion to graph failed).")

    return g, protein
