"""
HANSR Training Pipeline — Config-Driven Training Entry Point (FR-008)

Usage:
    python train.py --config configs/hansr.yaml
    python train.py --config configs/hansr.yaml --resume checkpoints/latest.pt

Requirements satisfied:
    FR-008: Config-driven, no magic numbers
    FR-009: Dataset verification gate (runs before training)
    FR-011: Checkpointing with full state resume
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    try:
        from tensorboardX import SummaryWriter
    except ImportError:
        SummaryWriter = None
try:
    from tqdm import tqdm
except ImportError:
    class tqdm:
        def __init__(self, iterable=None, *args, **kwargs):
            self.iterable = iterable or []
        def __iter__(self):
            return iter(self.iterable)
        def __len__(self):
            return len(self.iterable) if hasattr(self.iterable, "__len__") else 0
        def set_postfix(self, *args, **kwargs):
            pass
        def update(self, *args, **kwargs):
            pass
        def close(self):
            pass

from hansr.model import build_model, count_parameters
from hansr.losses import CompositeLoss
from hansr.dataset import build_datasets
from hansr.metrics import compute_psnr, compute_ssim
from hansr.utils import (
    load_config, set_seed, get_device, setup_logging,
    save_checkpoint, load_checkpoint, log_experiment, ensure_dir,
)

logger = logging.getLogger("hansr")


# =============================================================================
# Optimizer & Scheduler Factory
# =============================================================================

def build_optimizer(model: nn.Module, config: dict) -> torch.optim.Optimizer:
    """Build optimizer from config."""
    tcfg = config["training"]
    name = tcfg.get("optimizer", "adamw").lower()
    lr = tcfg.get("learning_rate", 1e-3)
    wd = tcfg.get("weight_decay", 0.0)

    if name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    elif name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    else:
        raise ValueError(f"Unknown optimizer: {name}")


def build_scheduler(optimizer, config: dict):
    """Build LR scheduler from config."""
    tcfg = config["training"]
    name = tcfg.get("scheduler", "cosine_annealing").lower()
    epochs = tcfg.get("epochs", 200)
    min_lr = tcfg.get("scheduler_min_lr", 1e-6)

    if name == "cosine_annealing":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=min_lr
        )
    elif name == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
    else:
        raise ValueError(f"Unknown scheduler: {name}")


# =============================================================================
# Training Loop
# =============================================================================

def train_one_epoch(
    model, dataloader, criterion, optimizer, scaler, device, config, epoch,
    writer, global_step,
):
    """Train for one epoch. Returns (avg_loss, loss_breakdown, global_step)."""
    model.train()
    total_loss = 0.0
    loss_accum = {}
    num_batches = 0
    log_every = config.get("logging", {}).get("log_every", 50)
    clip_norm = config["training"].get("gradient_clip_norm", 0.0)
    use_amp = config["training"].get("amp", True)

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}", leave=False)
    for batch in pbar:
        degraded = batch["degraded"].to(device)
        gt = batch["gt"].to(device)

        optimizer.zero_grad(set_to_none=True)

        # Mixed precision forward
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            pred = model(degraded)
            loss, loss_dict = criterion(pred, gt)

        # Backward
        if use_amp and scaler is not None:
            scaler.scale(loss).backward()
            if clip_norm > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if clip_norm > 0:
                nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
            optimizer.step()

        # Accumulate
        total_loss += loss.item()
        for k, v in loss_dict.items():
            loss_accum[k] = loss_accum.get(k, 0.0) + v
        num_batches += 1
        global_step += 1

        pbar.set_postfix(loss=f"{loss.item():.4f}")

        # TensorBoard logging
        if global_step % log_every == 0 and writer:
            writer.add_scalar("train/loss_total", loss.item(), global_step)
            for k, v in loss_dict.items():
                if k != "total":
                    writer.add_scalar(f"train/loss_{k}", v, global_step)
            writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], global_step)

    avg_loss = total_loss / max(num_batches, 1)
    avg_breakdown = {k: v / max(num_batches, 1) for k, v in loss_accum.items()}

    return avg_loss, avg_breakdown, global_step


@torch.no_grad()
def validate(model, dataloader, criterion, device, config):
    """Validate and compute metrics. Returns (avg_loss, avg_psnr, avg_ssim)."""
    model.eval()
    total_loss = 0.0
    total_psnr = 0.0
    total_ssim = 0.0
    num_batches = 0
    use_amp = config["training"].get("amp", True)

    for batch in dataloader:
        degraded = batch["degraded"].to(device)
        gt = batch["gt"].to(device)

        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            pred = model(degraded)
            loss, _ = criterion(pred, gt)

        # Clamp output for metric computation (metrics expect [0,1])
        pred_clamped = pred.clamp(0, 1)

        total_loss += loss.item()
        total_psnr += compute_psnr(pred_clamped, gt)
        total_ssim += compute_ssim(pred_clamped, gt)
        num_batches += 1

    n = max(num_batches, 1)
    return total_loss / n, total_psnr / n, total_ssim / n


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="HANSR Training")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--resume", default=None, help="Path to checkpoint for resume")
    args = parser.parse_args()

    setup_logging()

    # Load config
    config = load_config(args.config)
    tcfg = config["training"]

    # Override resume from CLI if provided
    if args.resume:
        tcfg["resume"] = args.resume

    # Seed
    set_seed(tcfg.get("seed", 42))

    # Device
    device = get_device()

    # Directories
    ckpt_dir = ensure_dir(config["checkpoint"]["dir"])
    log_dir = ensure_dir(config.get("logging", {}).get("tensorboard_dir", "runs"))
    results_dir = ensure_dir(config.get("logging", {}).get("experiment_registry", "results/experiments"))

    # Build model
    model = build_model(config).to(device)
    params = count_parameters(model)
    logger.info(f"Model: {params['total']:,} params ({params['total_mb']:.1f} MB)")

    # Build loss, optimizer, scheduler
    criterion = CompositeLoss(config).to(device)
    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(optimizer, config)

    # AMP scaler
    use_amp = tcfg.get("amp", True)
    scaler = torch.amp.GradScaler(device.type) if use_amp and device.type == "cuda" else None

    # Build datasets and dataloaders
    datasets = build_datasets(config)
    train_loader = DataLoader(
        datasets["train"],
        batch_size=tcfg.get("batch_size", 8),
        shuffle=True,
        num_workers=config["data"].get("num_workers", 4),
        pin_memory=config["data"].get("pin_memory", True) and device.type == "cuda",
        drop_last=True,
    )
    val_loader = (
        DataLoader(
            datasets["val"],
            batch_size=1,
            shuffle=False,
            num_workers=config["data"].get("num_workers", 4),
            pin_memory=config["data"].get("pin_memory", True) and device.type == "cuda",
        )
        if "val" in datasets and datasets["val"] is not None
        else None
    )

    # TensorBoard writer
    writer = SummaryWriter(log_dir=log_dir) if SummaryWriter is not None else None

    # Resume from checkpoint (FR-011)
    start_epoch = 0
    global_step = 0
    best_psnr = 0.0

    resume_path = tcfg.get("resume")
    if resume_path and os.path.exists(resume_path):
        ckpt = load_checkpoint(
            resume_path, model, optimizer, scheduler, scaler,
            device=device, resume_training=True,
        )
        start_epoch = ckpt.get("epoch", 0) + 1
        global_step = ckpt.get("global_step", 0)
        best_psnr = ckpt.get("best_psnr", 0.0)
        logger.info(f"Resumed from epoch {start_epoch}, best_psnr={best_psnr:.4f}")

    # Training loop
    epochs = tcfg.get("epochs", 200)
    val_every = config.get("evaluation", {}).get("val_every", 1)
    save_every = config["checkpoint"].get("save_every", 10)

    val_count = len(datasets["val"]) if "val" in datasets and datasets["val"] is not None else 0
    logger.info(f"Starting training: epochs={epochs}, batch_size={tcfg.get('batch_size')}")
    logger.info(f"Train samples: {len(datasets['train'])}, Val samples: {val_count}")
    logger.info(f"Loss: {criterion}")

    for epoch in range(start_epoch, epochs):
        epoch_start = time.time()

        # Train
        train_loss, train_breakdown, global_step = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler,
            device, config, epoch, writer, global_step,
        )

        # Step scheduler
        scheduler.step()

        epoch_time = time.time() - epoch_start

        # Validate & Log
        if val_loader is not None and (epoch % val_every == 0 or epoch == epochs - 1):
            val_loss, val_psnr, val_ssim = validate(
                model, val_loader, criterion, device, config
            )

            # Combined training and validation log (FR-008, KLA submission reporting)
            logger.info(
                f"Epoch {epoch}/{epochs-1} | "
                f"train_loss={train_loss:.4f} | "
                f"val_loss={val_loss:.4f} | "
                f"psnr={val_psnr:.2f} | "
                f"ssim={val_ssim:.4f} | "
                f"lr={optimizer.param_groups[0]['lr']:.2e} | "
                f"time={epoch_time:.1f}s"
            )

            # TensorBoard
            if writer:
                writer.add_scalar("val/loss", val_loss, epoch)
                writer.add_scalar("val/psnr", val_psnr, epoch)
                writer.add_scalar("val/ssim", val_ssim, epoch)

            # Save best checkpoint strictly based on validation PSNR
            if val_psnr > best_psnr:
                best_psnr = val_psnr
                save_checkpoint(
                    os.path.join(ckpt_dir, "best.pt"),
                    epoch, global_step, model, optimizer, scheduler, scaler,
                    config, best_psnr,
                )
                logger.info(f"  New best validation PSNR: {best_psnr:.2f} dB (saved to best.pt)")
        else:
            # Fallback training log when validation is not available
            logger.info(
                f"Epoch {epoch}/{epochs-1} | "
                f"train_loss={train_loss:.4f} | "
                f"lr={optimizer.param_groups[0]['lr']:.2e} | "
                f"time={epoch_time:.1f}s"
            )

        # Save latest periodically
        if (epoch + 1) % save_every == 0 or epoch == epochs - 1:
            save_checkpoint(
                os.path.join(ckpt_dir, "latest.pt"),
                epoch, global_step, model, optimizer, scheduler, scaler,
                config, best_psnr,
            )
            if val_loader is None:
                save_checkpoint(
                    os.path.join(ckpt_dir, "best.pt"),
                    epoch, global_step, model, optimizer, scheduler, scaler,
                    config, best_psnr,
                )

    # Final summary
    logger.info(f"Training complete. Best PSNR: {best_psnr:.2f}")

    # Log experiment
    log_experiment(
        results_dir,
        experiment_name="baseline",
        config=config,
        metrics={"best_psnr": best_psnr},
        notes=f"Trained {epochs} epochs",
    )

    writer.close()


if __name__ == "__main__":
    main()
