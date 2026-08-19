"""
HANSR Utility Functions — Config Loading, Checkpointing, Logging, Seeding

All shared infrastructure for the training and inference pipelines.
"""

import os
import json
import random
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import yaml


logger = logging.getLogger("hansr")


# =============================================================================
# Config Loading
# =============================================================================

def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load and validate a YAML configuration file.
    
    Args:
        config_path: Path to the YAML config file.
        
    Returns:
        Parsed configuration dictionary.
        
    Raises:
        FileNotFoundError: If config file doesn't exist.
        ValueError: If config is missing required sections.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    # Validate required top-level sections
    required_sections = ["model", "training", "loss", "degradation", "data", "checkpoint"]
    missing = [s for s in required_sections if s not in config]
    if missing:
        raise ValueError(f"Config missing required sections: {missing}")
    
    logger.info(f"Loaded config from {config_path}")
    return config


# =============================================================================
# Reproducibility — Full Seed Control
# =============================================================================

def set_seed(seed: int) -> None:
    """
    Set all random seeds for full reproducibility.
    Covers: Python random, NumPy, PyTorch CPU, PyTorch CUDA.
    
    Args:
        seed: Integer seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # Deterministic algorithms (may reduce performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.info(f"Set all random seeds to {seed}")


def get_rng_state() -> Dict[str, Any]:
    """Capture full RNG state for checkpoint resume (FR-011)."""
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.random.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def set_rng_state(state: Dict[str, Any]) -> None:
    """Restore full RNG state from checkpoint (FR-011)."""
    if "python" in state and state["python"] is not None:
        random.setstate(state["python"])
    if "numpy" in state and state["numpy"] is not None:
        np.random.set_state(state["numpy"])
    if "torch" in state and state["torch"] is not None:
        torch_state = state["torch"]
        if isinstance(torch_state, torch.Tensor):
            torch.random.set_rng_state(torch_state.cpu().to(torch.uint8).contiguous())
        else:
            torch.random.set_rng_state(torch_state)
    if torch.cuda.is_available() and "cuda" in state and state["cuda"] is not None:
        cuda_states = []
        for s in state["cuda"]:
            if isinstance(s, torch.Tensor):
                cuda_states.append(s.cpu().to(torch.uint8).contiguous())
            else:
                cuda_states.append(s)
        if cuda_states:
            torch.cuda.set_rng_state_all(cuda_states)
    logger.info("Restored RNG state from checkpoint")


# =============================================================================
# Device Selection (FR-012, US-10)
# =============================================================================

def get_device(device_override: Optional[str] = None) -> torch.device:
    """
    Automatic device selection with explicit override support.
    Works identically across Kaggle GPU, local CUDA, H100, and CPU.
    
    Args:
        device_override: Explicit device string ("cuda", "cpu", "cuda:0", etc.)
        
    Returns:
        torch.device instance.
    """
    if device_override:
        device = torch.device(device_override)
        logger.info(f"Using explicitly requested device: {device}")
        return device
    
    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        logger.info(f"Auto-detected CUDA device: {gpu_name} ({gpu_mem:.1f} GB)")
    else:
        device = torch.device("cpu")
        logger.info("No CUDA device found, using CPU")
    
    return device


# =============================================================================
# Checkpointing (FR-011)
# =============================================================================

def save_checkpoint(
    path: str,
    epoch: int,
    global_step: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    amp_scaler: Optional[Any],
    config: Dict[str, Any],
    best_psnr: float,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Save a full training checkpoint with all state needed for bit-exact resume.
    
    Contents: model, optimizer, scheduler, AMP scaler, RNG state, epoch, step,
    config, and best validation PSNR.
    """
    checkpoint = {
        "epoch": epoch,
        "global_step": global_step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer else None,
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "amp_scaler_state_dict": amp_scaler.state_dict() if amp_scaler else None,
        "rng_state": get_rng_state(),
        "config": config,
        "best_psnr": best_psnr,
    }
    if extra:
        checkpoint.update(extra)
    
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(checkpoint, path)
    logger.info(f"Saved checkpoint: {path} (epoch={epoch}, step={global_step}, best_psnr={best_psnr:.4f})")


def load_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    amp_scaler: Optional[Any] = None,
    device: Optional[torch.device] = None,
    resume_training: bool = False,
) -> Dict[str, Any]:
    """
    Load a checkpoint. For inference, only model weights are restored.
    For training resume, full state (optimizer, scheduler, RNG) is restored.
    
    Args:
        path: Path to .pt checkpoint file.
        model: Model to load weights into.
        optimizer: Optimizer to restore state (training resume only).
        scheduler: Scheduler to restore state (training resume only).
        amp_scaler: AMP scaler to restore state (training resume only).
        device: Device to map tensors to.
        resume_training: If True, restore full training state.
        
    Returns:
        Checkpoint dictionary with metadata (epoch, best_psnr, etc.)
        
    Raises:
        FileNotFoundError: If checkpoint file doesn't exist.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    
    map_location = device if device else "cpu"
    checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    
    model.load_state_dict(checkpoint["model_state_dict"])
    logger.info(f"Loaded model weights from {path}")
    
    if resume_training:
        if optimizer and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if scheduler and "scheduler_state_dict" in checkpoint and checkpoint["scheduler_state_dict"]:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        if amp_scaler and "amp_scaler_state_dict" in checkpoint and checkpoint["amp_scaler_state_dict"]:
            amp_scaler.load_state_dict(checkpoint["amp_scaler_state_dict"])
        if "rng_state" in checkpoint:
            set_rng_state(checkpoint["rng_state"])
        logger.info(f"Restored full training state from epoch {checkpoint.get('epoch', '?')}")
    
    return checkpoint


# =============================================================================
# Experiment Registry
# =============================================================================

def log_experiment(
    registry_dir: str,
    experiment_name: str,
    config: Dict[str, Any],
    metrics: Dict[str, Any],
    notes: str = "",
) -> str:
    """
    Log an experiment run to the JSON-based experiment registry.
    
    Args:
        registry_dir: Directory for experiment JSON files.
        experiment_name: Human-readable experiment identifier.
        config: Full configuration used for this run.
        metrics: Final metrics dictionary.
        notes: Optional notes about the run.
        
    Returns:
        Path to the saved experiment JSON file.
    """
    os.makedirs(registry_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{timestamp}_{experiment_name}.json"
    filepath = os.path.join(registry_dir, filename)
    
    entry = {
        "experiment_name": experiment_name,
        "timestamp": timestamp,
        "config": config,
        "metrics": metrics,
        "notes": notes,
    }
    
    with open(filepath, "w") as f:
        json.dump(entry, f, indent=2, default=str)
    
    logger.info(f"Logged experiment to {filepath}")
    return filepath


# =============================================================================
# Logging Setup
# =============================================================================

def setup_logging(level: int = logging.INFO) -> None:
    """Configure consistent logging format across all HANSR modules."""
    logging.basicConfig(
        level=level,
        format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


# =============================================================================
# Path Utilities
# =============================================================================

def ensure_dir(path: str) -> str:
    """Create directory if it doesn't exist, return the path."""
    os.makedirs(path, exist_ok=True)
    return path
