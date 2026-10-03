import sys
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

import preprocessing.create_dataloaders as cd
from models.compressed_model import build_compressed_model
from training.checkpoint import load_checkpoint

def cache_features_for_dataset(dataset_name: str):
    print("=" * 60)
    print(f"PRE-EXTRACTING 8-DIM QUANTUM FEATURES FOR {dataset_name.upper()}")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device for feature extraction: {device}")
    
    model = build_compressed_model(
        output_dim=8,
        hidden_dim=128,
        num_classes=2,
        pretrained=False,
        dropout_rate=0.0,
        use_tanh_scaling=True
    )
    
    ckpt_path = PROJECT_ROOT / "checkpoints" / f"compression_{dataset_name}_best.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint {ckpt_path} not found.")
        
    optimizer = torch.optim.Adam(model.parameters())
    epoch, val_m = load_checkpoint(ckpt_path, model, optimizer)
    print(f"Loaded trained Phase 5 model from epoch {epoch} (val accuracy: {val_m.get('accuracy', 0):.4f})")
    
    model.to(device)
    model.eval()
    
    cd.BATCH_SIZE = 32
    train_loader, val_loader, test_loader = cd.create_dataloaders(dataset_name)
    
    cache_dir = PROJECT_ROOT / "data" / "cached_features"
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    splits = [
        ("train", train_loader),
        ("val", val_loader),
        ("test", test_loader)
    ]
    
    with torch.no_grad():
        for split_name, loader in splits:
            features_list = []
            labels_list = []
            print(f"\nExtracting {split_name} split ({len(loader.dataset)} images)...")
            
            for images, labels in tqdm(loader, desc=f"Caching {split_name}"):
                images = images.to(device)
                compact_8dim = model.extract_features(images)
                features_list.append(compact_8dim.cpu())
                labels_list.append(labels)
                
            all_features = torch.cat(features_list, dim=0)
            all_labels = torch.cat(labels_list, dim=0)
            
            out_file = cache_dir / f"{dataset_name}_{split_name}_features.pt"
            torch.save({
                "features": all_features,
                "labels": all_labels
            }, out_file)
            print(f"[*] Saved {out_file.name} | Shape: {tuple(all_features.shape)}")
            
    print(f"\n[+] Feature caching complete for {dataset_name}!")

if __name__ == "__main__":
    ds = sys.argv[1] if len(sys.argv) > 1 else "dataset_B"
    cache_features_for_dataset(ds)
