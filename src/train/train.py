# src/train/train.py
import os
import math
import time
import argparse
from pathlib import Path
from typing import Tuple, Dict

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.amp.grad_scaler import GradScaler
from torch.amp.autocast_mode import autocast

from src.data.loader import create_dataloaders
from src.models.dti_model import DTINetwork


# small utilities / metrics
def rmse(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    return float(torch.sqrt(torch.mean((y_true - y_pred) ** 2)).item())


def mae(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    return float(torch.mean(torch.abs(y_true - y_pred)).item())


def r2_score(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    y_true_mean = torch.mean(y_true)
    ss_res = torch.sum((y_true - y_pred) ** 2)
    ss_tot = torch.sum((y_true - y_true_mean) ** 2)
    return float((1 - ss_res / (ss_tot + 1e-12)).item())


def concordance_index(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    # naive O(n^2) CI implementation
    y_true = y_true.detach().cpu().numpy()
    y_pred = y_pred.detach().cpu().numpy()
    n = 0
    n_concordant = 0
    n_tied = 0
    N = len(y_true)
    for i in range(N):
        for j in range(i + 1, N):
            if y_true[i] == y_true[j]:
                continue
            n += 1
            if (y_pred[i] - y_pred[j]) * (y_true[i] - y_true[j]) > 0:
                n_concordant += 1
            elif (y_pred[i] - y_pred[j]) * (y_true[i] - y_true[j]) == 0:
                n_tied += 1
    if n == 0:
        return 0.0
    return float((n_concordant + 0.5 * n_tied) / n)


def evaluate(
    model: nn.Module, dataloader: DataLoader, device: torch.device, amp: bool = False
) -> Dict[str, float]:
    model.eval()
    ys_all = []
    preds_all = []
    with torch.no_grad():
        for graph_batch, seqs, ys in dataloader:
            ys = ys.to(device)
            if amp:
                with autocast(device_type="cuda"):
                    preds = model(graph_batch.to(device), seqs)
            else:
                preds = model(graph_batch.to(device), seqs)
            preds_all.append(preds.detach().cpu())
            ys_all.append(ys.detach().cpu())
    ys_all = torch.cat(ys_all, dim=0)
    preds_all = torch.cat(preds_all, dim=0)
    return {
        "rmse": rmse(ys_all, preds_all),
        "mae": mae(ys_all, preds_all),
        "r2": r2_score(ys_all, preds_all),
        "ci": concordance_index(ys_all, preds_all),
    }


def save_checkpoint(state: dict, ckpt_dir: str, name: str):
    os.makedirs(ckpt_dir, exist_ok=True)
    path = os.path.join(ckpt_dir, name)
    torch.save(state, path)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv",
        type=str,
        required=True,
        help="Path to processed CSV (smiles,protein_sequence,affinity)",
    )
    parser.add_argument(
        "--cache",
        type=str,
        default="data/processed/davis_graphs.pt",
        help="Graph cache path",
    )
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--drug_hidden", type=int, default=128)
    parser.add_argument("--drug_layers", type=int, default=3)
    parser.add_argument("--fusion_hidden", type=int, default=256)
    parser.add_argument("--gnn_type", type=str, default="gcn")
    parser.add_argument("--esm_model", type=str, default="facebook/esm2_t6_8M_UR50D")
    parser.add_argument("--batch_log_step", type=int, default=50)
    parser.add_argument(
        "--num_workers", type=int, default=0, help="dataloader workers (0 on Windows)"
    )
    parser.add_argument("--output_dir", type=str, default="saved_models")
    parser.add_argument(
        "--amp", action="store_true", help="Use mixed precision training"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--resume", type=str, default=None, help="Path to checkpoint to resume from"
    )
    parser.add_argument(
        "--max_rows",
        type=int,
        default=None,
        help="Limit rows for debug; None = full dataset",
    )
    args = parser.parse_args()
    return args


def set_seed(seed: int):
    import random, numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(
        csv_path=args.csv,
        cache_path=args.cache,
        batch_size=args.batch_size,
        train_split=0.8,
        val_split=0.1,
        test_split=0.1,
        num_workers=args.num_workers,
        shuffle=True,
        rebuild_cache=False,
        max_rows=args.max_rows,
    )

    # sample to infer drug_in_dim
    sample_graph, _, _ = next(iter(train_loader))
    drug_in_dim = sample_graph.x.size(1)
    print("Inferred drug node feature dim:", drug_in_dim)

    model = DTINetwork(
        drug_in_dim=drug_in_dim,
        drug_hidden_dim=args.drug_hidden,
        drug_layers=args.drug_layers,
        gnn_type=args.gnn_type,
        esm_model=args.esm_model,
        esm_device=str(device),
        esm_freeze=True,  # freeze ESM by default for speed
        fusion_hidden=args.fusion_hidden,
        dropout=0.2,
    )

    model = model.to(device)

    # check devices
    print("GNN device:", next(model.drug_encoder.parameters()).device)
    print("ESM device:", next(model.protein_encoder.model.parameters()).device)

    # optimizer & scaler
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scaler = GradScaler(device="cuda", enabled=args.amp)

    # loss
    criterion = nn.MSELoss()

    start_epoch = 1
    best_val_rmse = float("inf")

    # resume if requested
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optim_state"])
        start_epoch = ckpt.get("epoch", 1) + 1
        best_val_rmse = ckpt.get("best_val_rmse", best_val_rmse)
        print(f"Resumed from {args.resume} at epoch {start_epoch}")

    # prepare csv log
    os.makedirs(args.output_dir, exist_ok=True)
    log_path = os.path.join(args.output_dir, "training_log.csv")
    if not os.path.exists(log_path):
        with open(log_path, "w") as f:
            f.write(
                "epoch,train_rmse,train_mae,val_rmse,val_mae,val_r2,val_ci,epoch_time\n"
            )

    # training loop
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_samples = 0
        t0 = time.time()

        for step, (graph_batch, seqs, ys) in enumerate(train_loader, start=1):
            ys = ys.to(device)
            graph_batch = graph_batch.to(device)

            optimizer.zero_grad()

            if args.amp:
                with autocast(device_type="cuda"):
                    preds = model(graph_batch, seqs)
                    loss = criterion(preds, ys)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                preds = model(graph_batch, seqs)
                loss = criterion(preds, ys)
                loss.backward()
                optimizer.step()

            batch_size = ys.size(0)
            epoch_loss += float(loss.item()) * batch_size
            n_samples += batch_size

            if step % args.batch_log_step == 0:
                print(f"[Epoch {epoch}] Step {step} Loss: {loss.item():.4f}")

        epoch_time = time.time() - t0
        train_rmse = math.sqrt(epoch_loss / (n_samples + 1e-12))
        train_mae = None

        # run validation
        val_metrics = evaluate(model, val_loader, device, amp=args.amp)
        val_rmse = val_metrics["rmse"]
        val_mae = val_metrics["mae"]
        val_r2 = val_metrics["r2"]
        val_ci = val_metrics["ci"]

        print(
            f"Epoch {epoch} finished in {epoch_time:.1f}s — train_rmse: {train_rmse:.4f}, val_rmse: {val_rmse:.4f}, val_ci: {val_ci:.4f}"
        )

        # save last checkpoint
        ckpt = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optim_state": optimizer.state_dict(),
            "best_val_rmse": best_val_rmse,
        }
        save_checkpoint(ckpt, args.output_dir, f"checkpoint_epoch_{epoch}.pt")
        # save best
        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            ckpt_best = {**ckpt, "best_val_rmse": best_val_rmse}
            save_checkpoint(ckpt_best, args.output_dir, "best_checkpoint.pt")
            print(f"New best model (val_rmse {best_val_rmse:.4f}) saved.")

        # append csv log
        with open(log_path, "a") as f:
            f.write(
                f"{epoch},{train_rmse:.6f},{train_mae if train_mae else ''},{val_rmse:.6f},{val_mae:.6f},{val_r2:.6f},{val_ci:.6f},{epoch_time:.2f}\n"
            )

    # final test evaluation
    test_metrics = evaluate(model, test_loader, device, amp=args.amp)
    print("Final test metrics:", test_metrics)
    # save final model
    save_checkpoint(
        {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optim_state": optimizer.state_dict(),
            "best_val_rmse": best_val_rmse,
            "test_metrics": test_metrics,
        },
        args.output_dir,
        "final_model.pt",
    )


if __name__ == "__main__":
    train()
