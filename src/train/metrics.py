# src/train/metrics.py

import torch
import numpy as np


def rmse(pred, true):
    return torch.sqrt(torch.mean((pred - true) ** 2)).item()


def mae(pred, true):
    return torch.mean(torch.abs(pred - true)).item()


def r2_score(pred, true):
    pred = pred.detach().cpu().numpy()
    true = true.detach().cpu().numpy()
    ss_res = np.sum((true - pred) ** 2)
    ss_tot = np.sum((true - np.mean(true)) ** 2)
    if ss_tot == 0:
        return 0.0
    return 1 - ss_res / ss_tot


def concordance_index(pred, true):
    pred = pred.detach().cpu().numpy()
    true = true.detach().cpu().numpy()

    n = 0
    h_sum = 0

    for i in range(len(true)):
        for j in range(i + 1, len(true)):
            if true[i] != true[j]:
                n += 1
                if (pred[i] < pred[j] and true[i] < true[j]) or (
                    pred[i] > pred[j] and true[i] > true[j]
                ):
                    h_sum += 1
                elif pred[i] == pred[j]:
                    h_sum += 0.5

    if n == 0:
        return 0.0

    return h_sum / n
