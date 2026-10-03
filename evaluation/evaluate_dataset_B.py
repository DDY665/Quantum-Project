import sys
import yaml
import torch
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from models.densenet_vit import build_densenet_vit
from training.checkpoint import load_checkpoint
from preprocessing.create_dataloaders import create_dataloaders
from evaluation.evaluate import run_evaluation

def main():
    dataset_name = "dataset_B"
    experiment_name = f"densenet_vit_{dataset_name}"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    with open(PROJECT_ROOT / "configs" / "densenet_vit.yaml") as f:
        config = yaml.safe_load(f)

    model = build_densenet_vit(
        pretrained=config['model']['pretrained'],
        num_classes=config['model']['num_classes'],
        dropout_rate=config['model']['dropout_rate'],
        vit_variant=config['model']['vit_variant'],
        fusion=config['model']['fusion']
    )
    optimizer = torch.optim.Adam(model.parameters())
    ckpt_path = PROJECT_ROOT / "checkpoints" / f"{experiment_name}_best.pt"
    
    if not ckpt_path.exists():
        print(f"Error: {ckpt_path} does not exist.")
        return
        
    epoch, val_m = load_checkpoint(ckpt_path, model, optimizer)
    print(f"Loaded best checkpoint from epoch {epoch} with val metrics: {val_m}")
    model.to(device)

    batch_size = config['dataloader'].get('batch_size', 16)
    _, _, test_loader = create_dataloaders(dataset_name, batch_size=batch_size)
    
    metrics = run_evaluation(model, test_loader, device, experiment_name, PROJECT_ROOT / "results")
    print("\n" + "=" * 60)
    print("DATASET B TEST EVALUATION COMPLETE")
    print("=" * 60)
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"F1 Score:  {metrics['f1_weighted']:.4f}")
    print(f"ROC AUC:   {metrics['roc_auc']:.4f}")

if __name__ == "__main__":
    main()
