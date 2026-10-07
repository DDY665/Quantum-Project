# Task Tracker
> Hybrid Quantum-Classical Pneumonia Detection
> Updated manually as phases complete.

Legend: `[x]` Done · `[/]` In Progress · `[ ]` Remaining · `[-]` Excluded

---

## Phase 0 — Literature Survey

- [x] Survey CNN-based pneumonia detection papers
- [x] Survey Vision Transformer medical imaging papers
- [x] Survey quantum machine learning / VQC classification papers
- [x] Survey hybrid CNN + Transformer approaches
- [x] Document research gaps and motivation
- [x] Finalize architecture decision (DenseNet121 + ViT + Compression + VQC)

---

## Phase 1 — Dataset Preparation

- [x] Inspect and validate Kermany dataset (`inspect_dataset.py`)
- [x] Check and remove corrupted / unreadable images (`clean_images.py`)
- [x] Check and handle duplicates (`check_duplicates.py`)
- [x] Select 500 NON_PNEUMONIA + 500 PNEUMONIA from RSNA (`select_rsna.py`)
- [x] Convert RSNA DICOM images to JPEG/PNG (`rsna_converter.py`)
- [x] Build Dataset A (Kermany only) manifest CSVs (`build_dataset.py`)
- [x] Build Dataset B (Kermany + RSNA) manifest CSVs (`build_dataset.py`)
- [x] Generate dataset verification report (`dataset_report.py`)
- [x] Verify Dataset A: train/val/test split integrity
- [x] Verify Dataset B: train/val/test split integrity
- [x] Confirm label mapping: NON_PNEUMONIA=0, PNEUMONIA=1

---

## Phase 2 — Data Preprocessing & DataLoaders

- [x] Implement `PneumoniaDataset` class (`create_dataloaders.py`)
- [x] Implement training transforms (resize, flip, rotation, normalize)
- [x] Implement eval transforms (resize, normalize — no augmentation)
- [x] Implement `create_dataloaders()` for Dataset A and B
- [x] Implement `validate_split()` path and label checks
- [x] Implement `test_dataloaders()` batch shape and value checks
- [x] Verify image shape: (B, 3, 224, 224)
- [x] Verify no NaN / infinite values in batches
- [x] Verify label integrity (only 0 and 1)

---

## Phase 3 — Stage 1: DenseNet121 CNN Baseline  ← NEXT

### 3.1 Model Definition

- [ ] Create `models/densenet121.py`
  - [ ] `build_densenet121(pretrained: bool, num_classes: int) -> nn.Module`
  - [ ] Replace classifier head: `Linear(1024, num_classes)`
  - [ ] Optional: Dropout before classifier head
  - [ ] Return model ready for training
- [x] Create `models/densenet121.py`
  - [x] `build_densenet121(pretrained: bool, num_classes: int) -> nn.Module`
  - [x] Replace classifier head: `Linear(1024, num_classes)`
  - [x] Optional: Dropout before classifier head
  - [x] Return model ready for training

### 3.2 Training Loop

- [ ] Create `training/trainer.py`
  - [ ] `train_one_epoch(model, loader, optimizer, criterion, device) -> dict`
  - [ ] `validate_one_epoch(model, loader, criterion, device) -> dict`
  - [ ] Return dict: `{loss, accuracy}` per epoch
- [ ] Create `training/checkpoint.py`
  - [ ] `save_checkpoint(model, optimizer, epoch, metrics, path)`
  - [ ] `load_checkpoint(path, model, optimizer) -> epoch, metrics`
- [ ] Create `training/train_densenet.py`
  - [ ] `run_training(dataset_name, config) -> None`
  - [ ] Load DataLoaders (Dataset A or B)
  - [ ] Instantiate model, optimizer, scheduler, criterion
  - [ ] Training loop with early stopping
  - [ ] Save best checkpoint per dataset
- [x] Create `training/trainer.py`
  - [x] `train_one_epoch(model, loader, optimizer, criterion, device) -> dict`
  - [x] `validate_one_epoch(model, loader, criterion, device) -> dict`
  - [x] Return dict: `{loss, accuracy}` per epoch
- [x] Create `training/checkpoint.py`
  - [x] `save_checkpoint(model, optimizer, epoch, metrics, path)`
  - [x] `load_checkpoint(path, model, optimizer) -> epoch, metrics`
- [x] Create `training/train_densenet.py`
  - [x] `run_training(dataset_name, config) -> None`
  - [x] Load DataLoaders (Dataset A or B)
  - [x] Instantiate model, optimizer, scheduler, criterion
  - [x] Training loop with early stopping
  - [x] Save best checkpoint per dataset

### 3.3 Configuration

- [ ] Create `configs/densenet121.yaml` (or `.json`)
  - [ ] `learning_rate`
  - [ ] `weight_decay`
  - [ ] `num_epochs`
  - [ ] `early_stopping_patience`
  - [ ] `pretrained: true`
  - [ ] `num_classes: 2`
  - [ ] `dropout_rate`
  - [ ] `scheduler` type and params
- [x] Create `configs/densenet121.yaml` (or `.json`)
  - [x] `learning_rate`
  - [x] `weight_decay`
  - [x] `num_epochs`
  - [x] `early_stopping_patience`
  - [x] `pretrained: true`
  - [x] `num_classes: 2`
  - [x] `dropout_rate`
  - [x] `scheduler` type and params

### 3.4 Evaluation

- [ ] Create `evaluation/metrics.py`
  - [ ] `compute_metrics(y_true, y_pred, y_prob) -> dict`
  - [ ] Returns: accuracy, precision, recall, f1, specificity, roc_auc
- [ ] Create `evaluation/evaluate.py`
  - [ ] `run_evaluation(model, test_loader, device) -> dict`
  - [ ] Saves confusion matrix to `results/figures/`
  - [ ] Saves ROC curve to `results/figures/`
  - [ ] Saves metric table to `results/tables/`
- [x] Create `evaluation/metrics.py`
  - [x] `compute_metrics(y_true, y_pred, y_prob) -> dict`
  - [x] Returns: accuracy, precision, recall, f1, specificity, roc_auc
- [x] Create `evaluation/evaluate.py`
  - [x] `run_evaluation(model, test_loader, device) -> dict`
  - [x] Saves confusion matrix to `results/figures/`
  - [x] Saves ROC curve to `results/figures/`
  - [x] Saves metric table to `results/tables/`

### 3.5 Experiment Runner

- [ ] Create `experiments/exp_01_densenet_A.py`
  - [ ] Train DenseNet121 on Dataset A
  - [ ] Evaluate on Dataset A test set
  - [ ] Save all results
- [ ] Create `experiments/exp_02_densenet_B.py`
  - [ ] Train DenseNet121 on Dataset B
  - [ ] Evaluate on Dataset B test set (same Kermany test)
- [x] Create `experiments/exp_01_densenet_A.py`
  - [x] Train DenseNet121 on Dataset A
  - [x] Evaluate on Dataset A test set
  - [x] Save all results
- [x] Create `experiments/exp_02_densenet_B.py`
  - [x] Train DenseNet121 on Dataset B
  - [x] Evaluate on Dataset B test set (same Kermany test)

### 3.6 Acceptance Criteria (Phase 3)

- [x] Model trains without error for at least 1 full epoch on Dataset A and B
- [x] Checkpoint saves and loads correctly (resume training verified)
- [x] Metrics dict contains all 6 required metrics + confusion matrix + ROC
- [x] No NaN loss during training
- [x] GPU training confirmed (if hardware available)

---

## Phase 4 — Stage 2: DenseNet121 + Vision Transformer

### 4.1 Model Definition

- [x] Create `models/vit.py`
  - [x] `build_vit(variant, pretrained, num_classes) -> nn.Module`
  - [x] Variant is TBD; implement once decision is made
- [x] Create `models/densenet_vit.py`
  - [x] `build_densenet_vit(pretrained, num_classes) -> nn.Module`
  - [x] Combine DenseNet121 + ViT feature extraction
  - [x] Implement fusion: concatenation or learned (TBD)
  - [x] Classifier head: `Linear(fused_dim, num_classes)`

### 4.2 Training

- [x] Create `configs/densenet_vit.yaml`
  - [x] ViT-specific hyperparameters (patch size, etc.)
  - [x] Learning rates (may differ for backbone vs head)
- [x] Create `training/train_densenet_vit.py`
  - [x] Reuse `train_one_epoch` / `validate_one_epoch` from `trainer.py`

### 4.3 Experiment Runner

- [x] Create `experiments/exp_03_densenet_vit_A.py`
- [x] Create `experiments/exp_04_densenet_vit_B.py`

### 4.4 Acceptance Criteria (Phase 4)

- [ ] DenseNet121 baseline MUST be complete before starting Phase 4
- [ ] ViT integrates without shape errors on 224×224 input
- [ ] Combined model trains on Dataset A and B without error
- [ ] All 6 metrics + confusion matrix + ROC saved per experiment

---

## Phase 5 — Stage 3: Learnable Feature Compression

### 5.1 Model Definition

- [ ] Create `models/compression.py`
  - [ ] `FeatureCompressor(input_dim, output_dim) -> nn.Module`
  - [ ] MLP: `Linear → Activation → [Norm] → Linear`
  - [ ] Output dim = number of qubits (TBD)

### 5.2 Integration

- [ ] Create `models/compressed_model.py`
  - [ ] DenseNet121 + ViT → Compression → output compact vector
  - [ ] Verify output shape matches qubit count

### 5.3 Configuration

- [ ] Add to config: `compressed_dim` (TBD)
- [ ] Add activation and normalization choices

### 5.4 Acceptance Criteria (Phase 5)

- [ ] CNN + ViT baseline MUST be complete before starting Phase 5
- [ ] Compressed vector dimensionality is fixed and matches qubit count
- [ ] Compact vector range is compatible with angle encoding (e.g., [-π, π] or [0, π])
- [ ] Compression block is differentiable (gradients flow through)

---

## Phase 6 — Stage 4: VQC (Quantum Classification)

> Blocked on: quantum framework decision (PennyLane or Qiskit)

- [ ] Select and document quantum framework
- [ ] Create `models/vqc.py`
  - [ ] `build_vqc(n_qubits, n_layers, framework) -> quantum layer`
  - [ ] Angle encoding circuit
  - [ ] Parameterised rotation + entanglement layers
  - [ ] Measurement: Pauli-Z expectation
- [ ] Create `models/full_model.py`
  - [ ] DenseNet121 + ViT + Compression + VQC end-to-end
- [ ] Integrate VQC gradients with PyTorch autograd
- [ ] Create `configs/vqc.yaml`
- [ ] Create `training/train_full_model.py`
- [ ] Create `experiments/exp_05_full_model_A.py`
- [ ] Create `experiments/exp_06_full_model_B.py`

---

## Phase 7 — Full Evaluation & Comparison

- [ ] Run all 6 experiments (Models 1, 2, 3 × Datasets A, B)
- [ ] Collect metrics: accuracy, precision, recall, F1, specificity, ROC-AUC
- [ ] Generate confusion matrices for all 6 runs
- [ ] Generate ROC curves for all 6 runs
- [ ] Assemble comparison table: model × dataset × metric
- [ ] Write evaluation summary

---

## Phase 8 — Documentation

- [/] PROJECT.md — project overview spec
- [/] ARCHITECTURE.md — architecture spec
- [/] TASKS.md — task tracker (this file)
- [/] IMPLEMENTATION_PLAN.md — detailed phase plan
- [ ] Update README.md with final project summary
- [ ] Write experimental results section (after Phase 7)

---

## Excluded (By Design)

- [-] Teacher-student / knowledge distillation
- [-] Deployment / production web application
- [-] Relational database

