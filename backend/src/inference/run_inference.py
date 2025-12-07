# src/inference/run_inference.py

import torch
from torch_geometric.data import Batch


def run_inference(model, graph, protein):
    DEVICE = next(model.parameters()).device

    batch = Batch.from_data_list([graph]).to(DEVICE)
    preds = model(batch, [protein])
    return preds.item()
