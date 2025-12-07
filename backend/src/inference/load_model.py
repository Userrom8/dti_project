# src/inference/load_model.py

import torch
from src.models.dti_model import DTINetwork

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_trained_model(ckpt_path: str):
    print(f"Loading checkpoint: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=DEVICE)

    model = DTINetwork(
        drug_in_dim=30,
        drug_hidden_dim=128,
        drug_layers=3,
        fusion_hidden=256,
        esm_model="facebook/esm2_t6_8M_UR50D",
        esm_device=DEVICE,
        esm_freeze=True,
    ).to(DEVICE)

    model.load_state_dict(ckpt["model"])
    model.eval()

    return model
