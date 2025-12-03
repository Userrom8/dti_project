import torch
from torch_geometric.data import Batch

from src.utils.mol_utils import mol_to_pyg
from src.models.dti_model import DTINetwork

# Fake batch of 2 molecules
g1 = mol_to_pyg("CCO")
g2 = mol_to_pyg("CCN")
if g1 and g2:
    batch = Batch.from_data_list([g1, g2])

    # Fake protein sequences
    seqs = ["MKTIIALSYIFCLVFADYKDDDDK", "MVLSPADKTNVKAAWGKVGAHAGEY"]

    model = DTINetwork(
        drug_in_dim=g1.x.size(1),
        esm_device="cpu",
    )

    out = model(batch, seqs)
    print("Model output:", out)
    print("Shape:", out.shape)
else:
    print("mol_to_pyg crashed")
