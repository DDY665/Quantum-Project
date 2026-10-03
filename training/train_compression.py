import sys
from pathlib import Path
import yaml
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

import preprocessing.create_dataloaders as cd
from models.compressed_model import build_compressed_model
from training.trainer import train_one_epoch, validate_one_epoch
from training.checkpoint import save_checkpoint

def run_training(dataset_name: str, config: dict):
    print("=" * 60)
    print(f"TRAINING COMPRESSED MODEL ON {dataset_name.upper()}")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1. Load DataLoaders
    batch_size = config['dataloader'].get('batch_size', 16)
    cd.BATCH_SIZE = batch_size
    train_loader, val_loader, _ = cd.create_dataloaders(dataset_name)
    
    # 2. Build model with 8-dimensional bottleneck
    m_cfg = config['model']
    model = build_compressed_model(
        output_dim=m_cfg.get('output_dim', 8),
        hidden_dim=m_cfg.get('hidden_dim', 128),
        num_classes=m_cfg.get('num_classes', 2),
        pretrained=m_cfg.get('pretrained', True),
        dropout_rate=m_cfg.get('dropout_rate', 0.2),
        use_tanh_scaling=m_cfg.get('use_tanh_scaling', True)
    )
    
    # 3. Warm-start backbone from Phase 4 (DenseNet + ViT) if available
    phase4_ckpt = PROJECT_ROOT / "checkpoints" / f"densenet_vit_{dataset_name}_best.pt"
    if phase4_ckpt.exists():
        print(f"[*] Warm-starting backbone weights from Phase 4: {phase4_ckpt.name}")
        ckpt_data = torch.load(phase4_ckpt, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt_data['model_state_dict'], strict=False)
        print("[*] Pre-trained DenseNet121 and ViT weights successfully loaded into backbone!")
    
    model.to(device)
    
    # 4. Setup optimizer, scheduler, criterion
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
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt_path = checkpoint_dir / f"compression_{dataset_name}_best.pt"
    
    # 5. Training loop
    for epoch in range(1, num_epochs + 1):
        print(f"\nEpoch {epoch}/{num_epochs}")
        
        train_metrics = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics = validate_one_epoch(model, val_loader, criterion, device)
        
        if scheduler:
            scheduler.step()
            
        print(f"Train - Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.4f}")
        print(f"Val   - Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.4f}")
        
        current_metric = val_metrics['loss']
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
    pass
