from src.models.esm_encoder import ProteinEncoderESM
import torch

encoder = ProteinEncoderESM(
    model_name="facebook/esm2_t6_8M_UR50D", device="cpu", freeze=True
)

seqs = ["MKTIIALSYIFCLVFADYKDDDDK", "MVLSPADKTNVKAAWGKVGAHAGEY"]

emb = encoder(seqs)
print("Embedding shape:", emb.shape)
print("Embedding:", emb)
