import sys
import yaml
import torch
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from training.train_compression import run_training
from training.checkpoint import load_checkpoint
import preprocessing.create_dataloaders as cd
from models.compressed_model import build_compressed_model
from evaluation.evaluate import run_evaluation

def main():
    dataset_name = "dataset_A"
    experiment_name = f"compression_{dataset_name}"
    
    # 1. Load config
    config_path = PROJECT_ROOT / "configs" / "compression.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    # 2. Run training
    print(f"Starting experiment {experiment_name}...")
    run_training(dataset_name, config)
    
    # 3. Load best checkpoint for evaluation
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m_cfg = config['model']
    model = build_compressed_model(
        output_dim=m_cfg.get('output_dim', 8),
        hidden_dim=m_cfg.get('hidden_dim', 128),
        num_classes=m_cfg.get('num_classes', 2),
        pretrained=m_cfg.get('pretrained', True),
        dropout_rate=m_cfg.get('dropout_rate', 0.2),
        use_tanh_scaling=m_cfg.get('use_tanh_scaling', True)
    )
    
    optimizer = torch.optim.Adam(model.parameters())
    checkpoint_path = PROJECT_ROOT / "checkpoints" / f"{experiment_name}_best.pt"
    
    if not checkpoint_path.exists():
        print(f"Error: Checkpoint {checkpoint_path} not found.")
        return
        
    load_checkpoint(checkpoint_path, model, optimizer)
    model.to(device)
    
    # 4. Run evaluation
    batch_size = config['dataloader'].get('batch_size', 16)
    cd.BATCH_SIZE = batch_size
    _, _, test_loader = cd.create_dataloaders(dataset_name)
    output_dir = PROJECT_ROOT / "results"
    
    metrics = run_evaluation(
        model=model,
        test_loader=test_loader,
        device=device,
        experiment_name=experiment_name,
        output_dir=output_dir
    )
    
    # 5. Print summary
    print("\n" + "=" * 60)
    print("PHASE 5 (DATASET A) 8-DIM COMPRESSED EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"F1 Score:  {metrics['f1_weighted']:.4f}")
    print(f"ROC AUC:   {metrics['roc_auc']:.4f}")

if __name__ == "__main__":
    main()
