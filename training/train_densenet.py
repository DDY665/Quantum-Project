import sys
from pathlib import Path
import yaml
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

# Setup project path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from preprocessing.create_dataloaders import create_dataloaders
from models.densenet121 import build_densenet121
from training.trainer import train_one_epoch, validate_one_epoch
from training.checkpoint import save_checkpoint

def run_training(dataset_name: str, config: dict):
    print("=" * 60)
    print(f"TRAINING DENSENET121 ON {dataset_name.upper()}")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1. Load DataLoaders
    train_loader, val_loader, _ = create_dataloaders(dataset_name)
    
    # 2. Build model
    model = build_densenet121(
        pretrained=config['model']['pretrained'],
        num_classes=config['model']['num_classes'],
        dropout_rate=config['model']['dropout_rate'],
    )
    model.to(device)
    
    # 3. Setup optimizer, scheduler, criterion
    optimizer = AdamW(
        model.parameters(), 
        lr=config['training']['learning_rate'], 
        weight_decay=config['training']['weight_decay']
    )
    
    num_epochs = config['training']['num_epochs']
    if config['training']['scheduler'] == 'cosine':
        scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)
    else:
        scheduler = None
        
    criterion = nn.CrossEntropyLoss()
    
    best_metric = float('inf') if config['training']['checkpoint_metric'] == 'val_loss' else 0.0
    patience_counter = 0
    early_stopping_patience = config['training']['early_stopping_patience']
    
    checkpoint_dir = PROJECT_ROOT / "checkpoints"
    best_ckpt_path = checkpoint_dir / f"densenet121_{dataset_name}_best.pt"
    
    # 4. Train loop
    for epoch in range(1, num_epochs + 1):
        print(f"\nEpoch {epoch}/{num_epochs}")
        
        train_metrics = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics = validate_one_epoch(model, val_loader, criterion, device)
        
        if scheduler:
            scheduler.step()
            
        print(f"Train - Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.4f}")
        print(f"Val   - Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.4f}")
        
        # 5. Save best checkpoint and early stopping logic
        current_metric = val_metrics['loss'] # assuming val_loss is the metric
        is_best = current_metric < best_metric
        
        if is_best:
            best_metric = current_metric
            patience_counter = 0
            save_checkpoint(model, optimizer, epoch, val_metrics, best_ckpt_path)
            print(f"[*] Best checkpoint saved to {best_ckpt_path.name} (val_loss: {best_metric:.4f})")
        else:
            patience_counter += 1
            print(f"[!] No improvement for {patience_counter} epochs.")
            
        if patience_counter >= early_stopping_patience:
            print(f"Early stopping triggered after {epoch} epochs.")
            break

if __name__ == "__main__":
    # Can be used to test training independently
    pass

