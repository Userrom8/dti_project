# src/data/dataset.py

import os
import torch
import pandas as pd
from torch.utils.data import Dataset
from torch_geometric.data import Batch
from typing import Optional
from tqdm import tqdm
import logging

# logger
logger = logging.getLogger("dti.dataset")
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    logger.addHandler(ch)
logger.setLevel(logging.INFO)

# import smiles->pyg converter
try:
    from src.utils.mol_utils import mol_to_pyg
except Exception:
    from ..utils.mol_utils import mol_to_pyg


class DTIDataset(Dataset):
    """
    Dataset for Drug–Target Interaction.
    Loads SMILES, protein sequences or cached protein embeddings.
    """

    def __init__(
        self,
        csv_path: str,
        cache_path: Optional[str] = None,
        rebuild_cache: bool = False,
        max_rows: Optional[int] = None,
    ):

        self.csv_path = csv_path
        self.cache_path = cache_path
        self.samples = []

        df = pd.read_csv(csv_path)
        if max_rows is not None:
            df = df.head(max_rows)

        # -------------------------
        # Load graph cache if exists
        # -------------------------
        if cache_path and os.path.exists(cache_path) and not rebuild_cache:
            logger.info(f"Loading dataset cache from {cache_path}")
            try:
                self.samples = torch.load(cache_path)
                logger.info(f"Loaded {len(self.samples)} samples from cache")
            except Exception as e:
                logger.warning(f"Cache load failed ({e}) — rebuilding")
            else:
                # load protein embedding cache NOW
                self._load_protein_cache()
                return

        # ---------------------------------
        # Build graph samples from CSV
        # ---------------------------------
        processed = []
        logger.info(f"Processing CSV: {csv_path} ({len(df)} rows)")

        for idx, row in tqdm(df.iterrows(), total=len(df)):
            smi = row.get("smiles")
            seq = row.get("protein_sequence")
            aff = row.get("affinity")

            if pd.isna(smi) or pd.isna(seq) or pd.isna(aff):
                continue

            g = mol_to_pyg(smi)
            if g is None:
                continue

            try:
                y = float(aff)
            except:
                continue

            processed.append((g, seq, y))

        self.samples = processed
        logger.info(f"Processed {len(self.samples)} valid samples")

        # Save graph cache
        if cache_path:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            torch.save(self.samples, cache_path)
            logger.info(f"Saved processed cache to {cache_path}")

        # Load protein embedding cache
        self._load_protein_cache()

    # -----------------------------
    # Load protein embedding cache
    # -----------------------------
    def _load_protein_cache(self):
        cache_file = "data/processed/protein_embeddings.pt"
        if os.path.exists(cache_file):
            logger.info(f"Loading protein embedding cache: {cache_file}")
            self.protein_cache = torch.load(cache_file)
        else:
            logger.warning("No protein embedding cache found — using ESM at runtime.")
            self.protein_cache = None

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        g, seq, y = self.samples[idx]

        # If cache exists → return embedding tensor
        if self.protein_cache is not None:
            emb = self.protein_cache[seq].squeeze(0)  # (320,)
            return g, emb, torch.tensor(y, dtype=torch.float)

        # fallback: return raw sequence for ESM encoder
        return g, seq, torch.tensor(y, dtype=torch.float)


# ---------------------------
# DataLoader collate function
# ---------------------------
def collate_fn(batch):
    graphs, seqs, ys = zip(*batch)
    batched_graph = Batch.from_data_list(list(graphs))
    labels = torch.stack(ys)

    # If seqs are already embeddings → stack into tensor
    if torch.is_tensor(seqs[0]):
        seqs = torch.stack(seqs)  # (B, 320)
    else:
        seqs = list(seqs)  # list of strings

    return batched_graph, seqs, labels
