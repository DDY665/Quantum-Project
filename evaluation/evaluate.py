import sys
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import RocCurveDisplay

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from evaluation.metrics import compute_metrics

def run_evaluation(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    experiment_name: str,
    output_dir: Path,
) -> dict:
    print("\n" + "=" * 60)
    print(f"EVALUATING {experiment_name}")
    print("=" * 60)
    
    model.eval()
    all_preds = []
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Testing", leave=False):
            images = images.to(device)
            outputs = model(images) # logits
            probs = torch.softmax(outputs, dim=1)[:, 1] # Probability of class 1
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    y_prob = np.array(all_probs)
    
    # 2. Compute metrics
    metrics = compute_metrics(y_true, y_pred, y_prob)
    
    # Setup output directories
    figures_dir = output_dir / "figures"
    tables_dir = output_dir / "tables"
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    
    # 3. Save confusion matrix figure
    plt.figure(figsize=(6, 5))
    cm = metrics['confusion_matrix']
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['NON_PNEUMONIA', 'PNEUMONIA'], 
                yticklabels=['NON_PNEUMONIA', 'PNEUMONIA'])
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title(f'Confusion Matrix: {experiment_name}')
    plt.tight_layout()
    plt.savefig(figures_dir / f"{experiment_name}_cm.png")
    plt.close()
    
    # 4. Save ROC curve figure
    plt.figure(figsize=(6, 5))
    RocCurveDisplay.from_predictions(y_true, y_prob)
    plt.title(f'ROC Curve: {experiment_name}')
    plt.tight_layout()
    plt.savefig(figures_dir / f"{experiment_name}_roc.png")
    plt.close()
    
    # 5. Save metric table (JSON format)
    # Exclude confusion matrix from JSON as it's a numpy array
    serializable_metrics = {k: v for k, v in metrics.items() if k != 'confusion_matrix'}
    
    with open(tables_dir / f"{experiment_name}_metrics.json", "w") as f:
        json.dump(serializable_metrics, f, indent=4)
        
    print(f"Metrics saved to {tables_dir / f'{experiment_name}_metrics.json'}")
    print(f"Figures saved to {figures_dir}")
    
    return metrics

