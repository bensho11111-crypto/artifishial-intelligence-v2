"""
src/ml/train.py

Core training loop for FishCatchTransformer model.

Functions:
    - train_epoch: one pass over training data with backprop
    - eval_epoch: one pass over validation data with metrics computation
"""
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score


def train_epoch(model, train_loader, loss_fn, optimizer, device, grad_clip=1.0, profiler=None):
    """
    Train for one epoch.

    Args:
        model: FishCatchTransformer instance
        train_loader: DataLoader yielding batches with keys:
            "scans", "nav", "scan_valid", "label"
        loss_fn: Loss function (e.g. AsymmetricFocalLoss)
        optimizer: torch.optim.Optimizer instance
        device: torch.device (cuda or cpu)
        grad_clip: Gradient clipping norm (default 1.0)
        profiler: Optional profiler for profiling (unused in synthetic)

    Returns:
        float: average loss over all batches
    """
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch in train_loader:
        scans = batch["scans"].to(device)        # (B, T, 1, 24, 60, 128)
        nav = batch["nav"].to(device)            # (B, T, 7)
        valid = batch["scan_valid"].to(device)   # (B, T)
        labels = batch["label"].to(device)       # (B, 4)

        optimizer.zero_grad()

        # Forward pass
        logits = model(scans, valid, nav)  # (B, 4)

        # Compute loss
        loss = loss_fn(logits, labels)

        # Backward pass
        loss.backward()

        # Gradient clipping
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        # Optimizer step
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / n_batches


def eval_epoch(model, val_loader, loss_fn, device):
    """
    Evaluate on validation set.

    Args:
        model: FishCatchTransformer instance
        val_loader: DataLoader yielding batches
        loss_fn: Loss function
        device: torch.device

    Returns:
        tuple: (avg_loss, mean_auroc, mean_ap, per_species_aurocs, per_species_aps)
            - avg_loss (float): average loss
            - mean_auroc (float): mean AUROC across 4 species
            - mean_ap (float): mean AP across 4 species
            - per_species_aurocs (list of 4 floats): AUROC per species
            - per_species_aps (list of 4 floats): AP per species
    """
    model.eval()
    total_loss = 0.0
    all_logits = []
    all_labels = []
    n_batches_for_metrics = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            scans = batch["scans"].to(device)
            nav = batch["nav"].to(device)
            valid = batch["scan_valid"].to(device)
            labels = batch["label"].to(device)

            # Forward pass
            logits = model(scans, valid, nav)  # (B, 4)

            # Compute loss
            loss = loss_fn(logits, labels)
            total_loss += loss.item()

            # Store for metrics (sample to avoid slow metric computation)
            if n_batches_for_metrics < 10:  # Only use first 10 batches for metrics
                all_logits.append(logits.cpu().numpy())
                all_labels.append(labels.cpu().numpy())
                n_batches_for_metrics += 1

    # Concatenate all batches
    if all_logits:
        all_logits = np.concatenate(all_logits, axis=0)  # (N, 4)
        all_labels = np.concatenate(all_labels, axis=0)  # (N, 4)

        # Compute per-species AUROC and AP
        aurocs = []
        aps = []
        for i in range(all_labels.shape[1]):
            try:
                auc = roc_auc_score(all_labels[:, i], all_logits[:, i])
                ap = average_precision_score(all_labels[:, i], all_logits[:, i])
                aurocs.append(auc)
                aps.append(ap)
            except Exception:
                # Handle case where only one class present in batch
                aurocs.append(0.5)
                aps.append(0.0)

        mean_auroc = np.mean(aurocs)
        mean_ap = np.mean(aps)
    else:
        aurocs = [0.5] * 4
        aps = [0.0] * 4
        mean_auroc = 0.5
        mean_ap = 0.0

    avg_loss = total_loss / max(1, len(val_loader))

    return avg_loss, mean_auroc, mean_ap, aurocs, aps
