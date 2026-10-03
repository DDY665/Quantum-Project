import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from training.train_vqc import train_and_eval_vqc

def main():
    print("Launching Experiment 07: 8-Qubit VQC on Dataset A...")
    train_and_eval_vqc("dataset_A", n_layers=3, lr=0.03, epochs=15, batch_size=64)

if __name__ == "__main__":
    main()
