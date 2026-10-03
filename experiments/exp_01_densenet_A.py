import sys
import yaml
import torch
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from training.train_densenet import run_training
from training.checkpoint import load_checkpoint
from preprocessing.create_dataloaders import create_dataloaders
from models.densenet121 import build_densenet121
from evaluation.evaluate import run_evaluation

def main():
    dataset_name = "dataset_A"
    experiment_name = f"densenet121_{dataset_name}"
    
    # 1. Load config
    config_path = PROJECT_ROOT / "configs" / "densenet121.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    # 2. Run training
    print(f"Starting experiment {experiment_name}...")
    run_training(dataset_name, config)
    
    # 3. Load best checkpoint for evaluation
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_densenet121(
        pretrained=config['model']['pretrained'],
        num_classes=config['model']['num_classes'],
        dropout_rate=config['model']['dropout_rate']
    )
    
    optimizer = torch.optim.Adam(model.parameters()) 
    checkpoint_path = PROJECT_ROOT / "checkpoints" / f"{experiment_name}_best.pt"
    
    if not checkpoint_path.exists():
        print(f"Error: Checkpoint {checkpoint_path} not found.")
        return
        
    load_checkpoint(checkpoint_path, model, optimizer)
    model.to(device)
    
    # 4. Run evaluation
    _, _, test_loader = create_dataloaders(dataset_name)
    output_dir = PROJECT_ROOT / "results"
    
    metrics = run_evaluation(
        model=model,
        test_loader=test_loader,
        device=device,
        experiment_name=experiment_name,
        output_dir=output_dir
    )
    
    # 5. Print summary
    print("\nEvaluation Summary:")
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"F1 Score:  {metrics['f1_weighted']:.4f}")
    print(f"ROC AUC:   {metrics['roc_auc']:.4f}")

if __name__ == "__main__":
    main()

