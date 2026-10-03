from pathlib import Path
import torch
import torch.nn as nn
from torch.optim import Optimizer

def save_checkpoint(
    model: nn.Module,
    optimizer: Optimizer,
    epoch: int,
    metrics: dict,
    path: Path,
) -> None:
    """Saves model state, optimizer state, epoch, and metrics."""
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics,
    }
    torch.save(checkpoint, path)

def load_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: Optimizer,
) -> tuple[int, dict]:
    """
    Loads checkpoint into model and optimizer in-place.
    Returns: (epoch, metrics)
    """
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {path}")
        
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return checkpoint['epoch'], checkpoint['metrics']

