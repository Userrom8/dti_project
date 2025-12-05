# src/data/loader.py

import torch
from torch.utils.data import DataLoader, random_split
from typing import Tuple, Optional

try:
    from src.data.dataset import DTIDataset, collate_fn
except:
    from .dataset import DTIDataset, collate_fn


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
    Automatically handles:
        - cached protein embeddings
        - tiny datasets (ensures non-empty splits)
        - reproducible splits
    """

    # ---------------------------
    # Load dataset
    # ---------------------------
    dataset = DTIDataset(
        csv_path=csv_path,
        cache_path=cache_path,
        rebuild_cache=rebuild_cache,
        max_rows=max_rows,
    )

    total_len = len(dataset)
    if total_len == 0:
        raise ValueError("Dataset is empty. Check CSV or preprocessing.")

    # ---------------------------
    # Compute split sizes
    # ---------------------------
    if total_len < 3:
        # Not enough samples → put all in train
        print(
            f"[WARN] Dataset too small for splitting ({total_len} samples). Using train only."
        )
        train_loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn
        )
        return train_loader, None, None

    train_len = int(total_len * train_split)
    val_len = int(total_len * val_split)
    test_len = total_len - train_len - val_len

    # Guarantee at least 1 sample in each split
    if val_len == 0:
        val_len = 1
        train_len -= 1

    if test_len == 0:
        test_len = 1
        train_len -= 1

    print(f"[INFO] Dataset split: train={train_len}, val={val_len}, test={test_len}")

    # ---------------------------
    # Create splits
    # ---------------------------
    train_set, val_set, test_set = random_split(
        dataset,
        [train_len, val_len, test_len],
        generator=torch.Generator().manual_seed(42),
    )

    # ---------------------------
    # Create DataLoaders
    # ---------------------------
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        num_workers=num_workers,
    )

    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
    )

    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
    )

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    csv_path = "data/raw/sample.csv"
    cache_path = "data/processed/graphs_cache.pt"

    train_loader, val_loader, test_loader = create_dataloaders(
        csv_path, cache_path=cache_path, batch_size=4, max_rows=20
    )

    print("Train batches:", len(train_loader))
    print("Val batches:", len(val_loader))
    print("Test batches:", len(test_loader))
