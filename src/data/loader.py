# src/data/loader.py

import torch
from torch.utils.data import DataLoader, random_split
from typing import Tuple, Optional

try:
    from src.data.dataset import DTIDataset, collate_fn
except:
    from .dataset import DTIDataset, collate_fn  # fallback for script mode


def create_dataloaders(
    csv_path: str,
    cache_path: Optional[str] = None,
    batch_size: int = 8,
    train_split: float = 0.8,
    val_split: float = 0.1,
    test_split: float = 0.1,
    num_workers: int = 0,
    shuffle: bool = True,
    rebuild_cache: bool = False,
    max_rows: Optional[int] = None,
):
    """
    Returns (train_loader, val_loader, test_loader)

    Parameters:
    -----------
    csv_path: path to dataset CSV
    cache_path: where to store processed graphs (pt file)
    batch_size: batch size
    train_split/val_split/test_split: fractions
    num_workers: for multiprocessing dataloader (Windows = 0)
    shuffle: shuffle training data
    rebuild_cache: force rebuild of cached graph dataset
    max_rows: limit rows for debugging
    """

    # ------------------------------------------
    # Load full dataset
    # ------------------------------------------
    dataset = DTIDataset(
        csv_path=csv_path,
        cache_path=cache_path,
        rebuild_cache=rebuild_cache,
        max_rows=max_rows
    )

    total_len = len(dataset)
    assert total_len > 0, "Dataset is empty — check your CSV."

    # ------------------------------------------
    # Compute split sizes
    # ------------------------------------------
    train_len = int(total_len * train_split)
    val_len = int(total_len * val_split)
    test_len = total_len - train_len - val_len

    train_set, val_set, test_set = random_split(
        dataset, [train_len, val_len, test_len],
        generator=torch.Generator().manual_seed(42)
    )

    # ------------------------------------------
    # Build loaders
    # ------------------------------------------
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        num_workers=num_workers
    )

    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers
    )

    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers
    )

    return train_loader, val_loader, test_loader


# ------------------------
# Quick test / demo usage
# ------------------------
if __name__ == "__main__":
    csv_path = "data/raw/sample.csv"
    cache_path = "data/processed/graphs_cache.pt"

    train_loader, val_loader, test_loader = create_dataloaders(
        csv_path,
        cache_path=cache_path,
        batch_size=4,
        max_rows=20
    )

    print("Train batches:", len(train_loader))
    print("Val batches:", len(val_loader))
    print("Test batches:", len(test_loader))

    # Inspect one batch
    for batch_graph, seqs, labels in train_loader:
        print("Graph batch:", batch_graph)
        print("Seqs:", seqs)
        print("Labels:", labels)
        break
