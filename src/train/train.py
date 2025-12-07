# src/train/train.py

import os
import time
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp.grad_scaler import GradScaler
from torch.amp.autocast_mode import autocast

import pandas as pd
from tqdm import tqdm

from src.data.dataset import DTIDataset, collate_fn
from src.models.dti_model import DTINetwork
from src.train.metrics import rmse, mae, r2_score, concordance_index


# --------------------------------------
# Evaluate model on dataloader
# --------------------------------------
def evaluate(model, loader, device, amp=False):
    model.eval()
    preds_all, ys_all = [], []

    with torch.no_grad():
        for graph_batch, seqs, ys in loader:
            graph_batch = graph_batch.to(device)

            if torch.is_tensor(seqs):
                seqs = seqs.to(device)

            with autocast("cuda", enabled=amp):
                preds = model(graph_batch, seqs)

            preds_all.append(preds.cpu())
            ys_all.append(ys.cpu())

    if len(preds_all) == 0:
        return {
            "rmse": float("nan"),
            "mae": float("nan"),
            "r2": float("nan"),
            "ci": float("nan"),
        }

    preds_all = torch.cat(preds_all)
    ys_all = torch.cat(ys_all)

    return {
        "rmse": rmse(preds_all, ys_all),
        "mae": mae(preds_all, ys_all),
        "r2": r2_score(preds_all, ys_all),
        "ci": concordance_index(preds_all, ys_all),
    }


# --------------------------------------
# Train loop
# --------------------------------------
def train():
    parser = argparse.ArgumentParser()

    parser.add_argument("--csv", type=str, required=True)
    parser.add_argument("--cache", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--drug_hidden", type=int, default=128)
    parser.add_argument("--drug_layers", type=int, default=3)
    parser.add_argument("--fusion_hidden", type=int, default=256)
    parser.add_argument("--esm_model", type=str, default="facebook/esm2_t6_8M_UR50D")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--resume", type=str, default=None)

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --------------------------------------
    # Load dataset
    # --------------------------------------
    ds = DTIDataset(args.csv, cache_path=args.cache, rebuild_cache=False)
    N = len(ds)
    train_n = int(0.8 * N)
    val_n = int(0.1 * N)
    test_n = N - train_n - val_n

    train_ds, val_ds, test_ds = torch.utils.data.random_split(
        ds, [train_n, val_n, test_n]
    )
    print(
        f"[INFO] Dataset split: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}"
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
    )

    # --------------------------------------
    # Infer drug feature dimension
    # --------------------------------------
    sample_graph, _, _ = ds[0]
    drug_in_dim = sample_graph.x.shape[1]
    print(f"Inferred drug node feature dim: {drug_in_dim}")

    # --------------------------------------
    # Initialize model
    # --------------------------------------
    model = DTINetwork(
        drug_in_dim=drug_in_dim,
        drug_hidden_dim=args.drug_hidden,
        drug_layers=args.drug_layers,
        fusion_hidden=args.fusion_hidden,
        esm_model=args.esm_model,
        esm_device=device,
        esm_freeze=True,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scaler = GradScaler("cuda", enabled=args.amp)

    start_epoch = 1
    best_val_rmse = float("inf")

    # --------------------------------------
    # Resume checkpoint
    # --------------------------------------
    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optim"])
        start_epoch = ckpt["epoch"] + 1
        best_val_rmse = ckpt.get("best_val_rmse", best_val_rmse)
        print(
            f"[INFO] Resumed from epoch {ckpt['epoch']} with best_val_rmse {best_val_rmse:.4f}"
        )

    # --------------------------------------
    # CSV log file
    # --------------------------------------
    os.makedirs("saved_models", exist_ok=True)
    log_path = "saved_models/training_log.csv"

    if start_epoch == 1:
        with open(log_path, "w") as f:
            f.write(
                "epoch,train_rmse,train_mae,val_rmse,val_mae,val_r2,val_ci,epoch_time\n"
            )

    # --------------------------------------
    # Training loop
    # --------------------------------------
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        t0 = time.time()

        preds_list, ys_list = [], []

        for graph_batch, seqs, ys in tqdm(train_loader, desc=f"Epoch {epoch}"):
            graph_batch = graph_batch.to(device)
            if torch.is_tensor(seqs):
                seqs = seqs.to(device)
            ys = ys.to(device)

            optimizer.zero_grad()

            with autocast("cuda", enabled=args.amp):
                preds = model(graph_batch, seqs)
                loss = nn.MSELoss()(preds, ys)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            preds_list.append(preds.detach().cpu())
            ys_list.append(ys.cpu())

        preds_full = torch.cat(preds_list)
        ys_full = torch.cat(ys_list)

        train_rmse = rmse(preds_full, ys_full)
        train_mae = mae(preds_full, ys_full)

        # --------------------------------------
        # Validation
        # --------------------------------------
        val_metrics = evaluate(model, val_loader, device, amp=args.amp)

        dt = time.time() - t0
        print(
            f"Epoch {epoch} finished in {dt:.1f}s — "
            f"train_rmse={train_rmse:.4f}, val_rmse={val_metrics['rmse']:.4f}, "
            f"val_ci={val_metrics['ci']:.4f}"
        )

        # Log to CSV
        with open(log_path, "a") as f:
            f.write(
                f"{epoch},{train_rmse:.6f},{train_mae:.6f},"
                f"{val_metrics['rmse']:.6f},{val_metrics['mae']:.6f},"
                f"{val_metrics['r2']:.6f},{val_metrics['ci']:.6f},{dt:.2f}\n"
            )

        # --------------------------------------
        # Save best checkpoint
        # --------------------------------------
        if val_metrics["rmse"] < best_val_rmse:
            best_val_rmse = val_metrics["rmse"]
            ckpt = {
                "epoch": epoch,
                "model": model.state_dict(),
                "optim": optimizer.state_dict(),
                "best_val_rmse": best_val_rmse,
            }
            torch.save(ckpt, "saved_models/best_checkpoint.pt")
            print(f"[INFO] New best model saved (val_rmse={best_val_rmse:.4f})")

    # --------------------------------------
    # Final Test Evaluation
    # --------------------------------------
    print("\nEvaluating best checkpoint on test set...")
    ckpt = torch.load("saved_models/best_checkpoint.pt", map_location=device)
    model.load_state_dict(ckpt["model"])

    test_metrics = evaluate(model, test_loader, device, amp=args.amp)
    print("Final test metrics:", test_metrics)

    # Save test summary
    pd.DataFrame([test_metrics]).to_csv("saved_models/test_summary.csv", index=False)


if __name__ == "__main__":
    train()
