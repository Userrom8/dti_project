# src/data/dataset.py
import os
import torch
import pandas as pd
from torch.utils.data import Dataset
from torch_geometric.data import Batch
from typing import Optional
from tqdm import tqdm
import logging

# ensure logger is available
logger = logging.getLogger("dti.dataset")
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    logger.addHandler(ch)
logger.setLevel(logging.INFO)

# import your mol conversion utility
try:
    from src.utils.mol_utils import mol_to_pyg
except Exception:
    # if running as script from repo root, allow relative import fallback
    from ..utils.mol_utils import mol_to_pyg  # type: ignore


class DTIDataset(Dataset):
    """
    Dataset for Drug-Target Interaction.

    Expected CSV columns: 'smiles', 'protein_sequence', 'affinity'
    affinity: numeric (e.g., pKd/pIC50). Convert externally if needed.
    """

    def __init__(self,
                 csv_path: str,
                 cache_path: Optional[str] = None,
                 rebuild_cache: bool = False,
                 max_rows: Optional[int] = None):
        """
        csv_path: path to csv with required columns
        cache_path: optional path to save/load processed graphs (torch.save)
        rebuild_cache: if True, ignore cache and reprocess
        max_rows: useful for debugging (process only first N rows)
        """
        self.csv_path = csv_path
        self.cache_path = cache_path
        self.samples = []  # list of tuples (graph, protein_seq, affinity)

        assert os.path.exists(csv_path), f"CSV not found: {csv_path}"
        df = pd.read_csv(csv_path)
        if max_rows is not None:
            df = df.head(max_rows)

        # try loading cached processed graphs
        if cache_path and os.path.exists(cache_path) and not rebuild_cache:
            try:
                logger.info(f"Loading dataset cache from {cache_path}")
                cached = torch.load(cache_path)
                self.samples = cached
                logger.info(f"Loaded {len(self.samples)} samples from cache")
                return
            except Exception as e:
                logger.warning(f"Failed to load cache ({e}), rebuilding")

        # otherwise process rows
        processed = []
        logger.info(f"Processing CSV: {csv_path} ({len(df)} rows)")
        for idx, row in tqdm(df.iterrows(), total=len(df)):
            smi = row.get('smiles') if 'smiles' in row else row.get('SMILES')
            seq = row.get('protein_sequence') if 'protein_sequence' in row else row.get('sequence')
            aff = row.get('affinity') if 'affinity' in row else row.get('affinity_value')

            # basic validation
            if pd.isna(smi) or pd.isna(seq) or pd.isna(aff):
                logger.warning(f"Skipping row {idx}: missing field")
                continue

            try:
                g = mol_to_pyg(smi)
            except Exception as e:
                logger.warning(f"mol_to_pyg crashed for row {idx} ({e}) — skipping")
                continue

            if g is None:
                logger.warning(f"Invalid SMILES at row {idx}: {smi} — skipping")
                continue

            # ensure label is numeric
            try:
                y = float(aff)
            except Exception:
                logger.warning(f"Invalid affinity at row {idx}: {aff} — skipping")
                continue

            processed.append((g, str(seq), y))

        self.samples = processed
        logger.info(f"Processed {len(self.samples)} valid samples")

        # save cache if requested
        if cache_path:
            try:
                os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                torch.save(self.samples, cache_path)
                logger.info(f"Saved processed cache to {cache_path}")
            except Exception as e:
                logger.warning(f"Failed to save cache: {e}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        g, seq, y = self.samples[idx]
        return g, seq, torch.tensor(y, dtype=torch.float)


# Collate function for DataLoader
def collate_fn(batch):
    """
    batch: list of (pyg.Data, seq (str), label tensor)
    returns:
      - batched_graph (torch_geometric.data.Batch)
      - list_of_sequences (List[str])
      - labels_tensor (torch.Tensor) shape (B,)
    """
    graphs, seqs, ys = zip(*batch)
    batched = Batch.from_data_list(list(graphs))
    labels = torch.stack(ys)
    return batched, list(seqs), labels


# ------------------------
# Quick test / usage demo
# ------------------------
if __name__ == "__main__":
    # quick local test when run directly
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default="data/raw/sample.csv", help="sample csv path")
    parser.add_argument("--cache", type=str, default="data/processed/graphs_cache.pt", help="cache path")
    args = parser.parse_args()

    ds = DTIDataset(args.csv, cache_path=args.cache, rebuild_cache=False)
    print(f"Dataset size: {len(ds)}")
    if len(ds) > 0:
        g, seq, y = ds[0]
        print("Sample graph:", g)
        print("Sample seq (len):", len(seq))
        print("Sample label:", y)
