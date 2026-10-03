import sys
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import RocCurveDisplay

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from models.vqc import build_vqc
from evaluation.metrics import compute_metrics

def train_and_eval_vqc(dataset_name: str, n_layers: int = 3, lr: float = 0.02, epochs: int = 15, batch_size: int = 64):
    print("=" * 60)
    print(f"TRAINING VARIATIONAL QUANTUM CLASSIFIER (VQC) ON {dataset_name.upper()}")
    print("=" * 60)
    
    experiment_name = f"vqc_{dataset_name}"
    cache_dir = PROJECT_ROOT / "data" / "cached_features"
    
    train_file = cache_dir / f"{dataset_name}_train_features.pt"
    val_file = cache_dir / f"{dataset_name}_val_features.pt"
    test_file = cache_dir / f"{dataset_name}_test_features.pt"
    
    if not (train_file.exists() and val_file.exists() and test_file.exists()):
        raise FileNotFoundError(f"Cached features for {dataset_name} not found in {cache_dir}. Run cache_features.py first.")
        
    train_data = torch.load(train_file, weights_only=False)
    val_data = torch.load(val_file, weights_only=False)
    test_data = torch.load(test_file, weights_only=False)
    
    train_loader = DataLoader(TensorDataset(train_data["features"], train_data["labels"]), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(val_data["features"], val_data["labels"]), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(TensorDataset(test_data["features"], test_data["labels"]), batch_size=batch_size, shuffle=False)
    
    print(f"Loaded: Train {len(train_data['labels'])}, Val {len(val_data['labels'])}, Test {len(test_data['labels'])} samples.")
    print("Quantum register: 8 Qubits | Circuit: Angle Embedding + CNOT Entanglement")
    
    model = build_vqc(n_qubits=8, n_layers=n_layers)
    optimizer = Adam(model.parameters(), lr=lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()
    
    best_loss = float("inf")
    patience = 0
    max_patience = 5
    
    checkpoint_dir = PROJECT_ROOT / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt = checkpoint_dir / f"{experiment_name}_best.pt"
    
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        
        for x_b, y_b in tqdm(train_loader, desc=f"VQC Epoch {epoch}/{epochs}", leave=False):
            optimizer.zero_grad()
            logits = model(x_b)
            loss = criterion(logits, y_b)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * len(y_b)
            preds = logits.argmax(dim=1)
            correct += (preds == y_b).sum().item()
            total += len(y_b)
            
        train_loss = total_loss / total
        train_acc = correct / total
        scheduler.step()
        
        model.eval()
        v_loss, v_corr, v_tot = 0.0, 0, 0
        with torch.no_grad():
            for vx, vy in val_loader:
                v_logits = model(vx)
                loss = criterion(v_logits, vy)
                v_loss += loss.item() * len(vy)
                v_preds = v_logits.argmax(dim=1)
                v_corr += (v_preds == vy).sum().item()
                v_tot += len(vy)
                
        val_loss = v_loss / v_tot
        val_acc = v_corr / v_tot
        
        print(f"Epoch {epoch:2d}/{epochs} | Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")
        
        if val_loss < best_loss:
            best_loss = val_loss
            patience = 0
            torch.save({"model_state_dict": model.state_dict(), "val_acc": val_acc, "val_loss": val_loss, "epoch": epoch}, best_ckpt)
            print(f"[*] New best VQC checkpoint saved! (Val Loss: {best_loss:.4f}, Acc: {val_acc:.4f})")
        else:
            patience += 1
            if patience >= max_patience:
                print(f"Early stopping triggered after {epoch} epochs.")
                break
                
    print("\n" + "=" * 60)
    print(f"EVALUATING VARIATIONAL QUANTUM CLASSIFIER ON {dataset_name.upper()} TEST SET")
    print("=" * 60)
    
    ckpt_data = torch.load(best_ckpt, weights_only=False)
    model.load_state_dict(ckpt_data["model_state_dict"])
    model.eval()
    
    all_preds, all_probs, all_labels = [], [], []
    with torch.no_grad():
        for tx, ty in tqdm(test_loader, desc="Quantum Testing"):
            logits = model(tx)
            probs = torch.softmax(logits, dim=1)[:, 1]
            preds = logits.argmax(dim=1)
            all_preds.extend(preds.numpy())
            all_probs.extend(probs.numpy())
            all_labels.extend(ty.numpy())
            
    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    y_prob = np.array(all_probs)
    
    metrics = compute_metrics(y_true, y_pred, y_prob)
    
    fig_dir = PROJECT_ROOT / "results" / "figures"
    tbl_dir = PROJECT_ROOT / "results" / "tables"
    fig_dir.mkdir(parents=True, exist_ok=True)
    tbl_dir.mkdir(parents=True, exist_ok=True)
    
    plt.figure(figsize=(6, 5))
    sns.heatmap(metrics["confusion_matrix"], annot=True, fmt="d", cmap="Blues",
                xticklabels=["NON_PNEUMONIA", "PNEUMONIA"],
                yticklabels=["NON_PNEUMONIA", "PNEUMONIA"])
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.title(f"Confusion Matrix: {experiment_name}")
    plt.tight_layout()
    plt.savefig(fig_dir / f"{experiment_name}_cm.png")
    plt.close()
    
    plt.figure(figsize=(6, 5))
    RocCurveDisplay.from_predictions(y_true, y_prob)
    plt.title(f"ROC Curve: {experiment_name}")
    plt.tight_layout()
    plt.savefig(fig_dir / f"{experiment_name}_roc.png")
    plt.close()
    
    serializable = {k: v for k, v in metrics.items() if k != "confusion_matrix"}
    with open(tbl_dir / f"{experiment_name}_metrics.json", "w") as f:
        json.dump(serializable, f, indent=4)
        
    print(f"\n[+] Results saved:")
    print(f"    - Table:   {tbl_dir / f'{experiment_name}_metrics.json'}")
    print(f"    - Figures: {fig_dir / f'{experiment_name}_cm.png'}, {fig_dir / f'{experiment_name}_roc.png'}")
    print("\n" + "=" * 60)
    print(f"VQC ({dataset_name.upper()}) FINAL TEST RESULTS:")
    print("=" * 60)
    print(f"Accuracy:    {metrics['accuracy']:.4f}")
    print(f"F1 Score:    {metrics['f1_weighted']:.4f}")
    print(f"ROC AUC:     {metrics['roc_auc']:.4f}")
    print(f"Specificity: {metrics['specificity']:.4f}")
    return metrics

if __name__ == "__main__":
    ds = sys.argv[1] if len(sys.argv) > 1 else "dataset_A"
    train_and_eval_vqc(ds)
