# Implementation Plan
> Hybrid Quantum-Classical Pneumonia Detection
> Covers: Phase 3 (DenseNet121) → Phase 4 (+ ViT) → Phase 5 (+ Compression)
> Phase 4 (VQC) is documented separately once quantum framework is decided.

---

## Current Status

| Phase | Status |
|---|---|
| Preprocessing & DataLoaders | ✅ DONE |
| Phase 3 — DenseNet121 CNN Baseline | 🔄 STARTING NOW |
| Phase 4 — DenseNet121 + ViT | ⏳ Blocked on Phase 3 |
| Phase 5 — Feature Compression | ⏳ Blocked on Phase 4 |

---

## Phase 3 — DenseNet121 CNN Baseline

### Goal
Train a binary DenseNet121 classifier independently on Dataset A and Dataset B. Establish the CNN-only performance baseline. Validate the full training pipeline (model → train loop → checkpoint → evaluate → metrics).

### Prerequisites
- `data/processed/dataset_A/splits/{train,val,test}.csv` — exists ✅
- `data/processed/dataset_B/splits/{train,val,test}.csv` — exists ✅
- `preprocessing/create_dataloaders.py` — exists ✅

### Input / Output

| | Value |
|---|---|
| Input | Chest X-ray images via `PneumoniaDataset` + `DataLoader` |
| Input shape | `(B, 3, 224, 224)` float32 tensor |
| Output | Binary logits `(B, 2)` → class 0 (NON_PNEUMONIA) or 1 (PNEUMONIA) |
| Loss | `CrossEntropyLoss` (recommended) |
| Checkpoint | Best model by validation loss (or validation F1) |

---

### Files to Create

#### `models/densenet121.py`

```python
def build_densenet121(
    pretrained: bool = True,
    num_classes: int = 2,
    dropout_rate: float = 0.5,
) -> nn.Module:
    """
    Returns a DenseNet121 with a custom binary classifier head.
    Classifier: Dropout → Linear(1024, num_classes)
    """
```

Config keys consumed:
- `pretrained` (bool)
- `num_classes` (int, = 2)
- `dropout_rate` (float)

---

#### `configs/densenet121.yaml`

```yaml
model:
  pretrained: true
  num_classes: 2
  dropout_rate: 0.5           # TBD — adjust based on val curve

training:
  num_epochs: 30              # TBD
  learning_rate: 1e-4         # TBD
  weight_decay: 1e-4          # TBD
  early_stopping_patience: 5  # TBD
  scheduler: cosine           # TBD: cosine | step | none
  checkpoint_metric: val_loss # or val_f1

dataloader:
  batch_size: 32
  num_workers: 0
  image_size: 224
  random_seed: 42
```

> All `# TBD` values are initial suggestions to be tuned during training. Update this file with final values used in experiments.

---

#### `training/trainer.py`

```python
def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> dict:
    """
    Returns: {"loss": float, "accuracy": float}
    """

def validate_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> dict:
    """
    Returns: {"loss": float, "accuracy": float}
    """
```

---

#### `training/checkpoint.py`

```python
def save_checkpoint(
    model: nn.Module,
    optimizer: Optimizer,
    epoch: int,
    metrics: dict,
    path: Path,
) -> None:
    """Saves model state, optimizer state, epoch, and metrics."""

def load_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: Optimizer,
) -> tuple[int, dict]:
    """
    Loads checkpoint into model and optimizer in-place.
    Returns: (epoch, metrics)
    """
```

Checkpoint save path pattern:
```
checkpoints/densenet121_{dataset_name}_best.pt
checkpoints/densenet121_{dataset_name}_epoch{N}.pt
```

---

#### `training/train_densenet.py`

```python
def run_training(
    dataset_name: str,   # "dataset_A" or "dataset_B"
    config: dict,
) -> None:
    """
    Full training loop:
    1. Load DataLoaders (train, val)
    2. Build model
    3. Setup optimizer, scheduler, criterion
    4. Train for num_epochs with early stopping
    5. Save best checkpoint
    6. Log train/val loss and accuracy per epoch
    """
```

---

#### `evaluation/metrics.py`

```python
def compute_metrics(
    y_true: np.ndarray,    # shape (N,) int {0, 1}
    y_pred: np.ndarray,    # shape (N,) int {0, 1}
    y_prob: np.ndarray,    # shape (N,) float — probability of class 1
) -> dict:
    """
    Returns dict with keys:
    - accuracy
    - precision_weighted
    - precision_per_class   (list [non_pneumonia, pneumonia])
    - recall_weighted       (= sensitivity weighted)
    - recall_per_class
    - f1_weighted
    - f1_per_class
    - specificity           (TNR for NON_PNEUMONIA class)
    - roc_auc
    - confusion_matrix      (2×2 numpy array)
    """
```

---

#### `evaluation/evaluate.py`

```python
def run_evaluation(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    experiment_name: str,   # e.g. "densenet121_dataset_A"
    output_dir: Path,
) -> dict:
    """
    1. Run inference on test_loader
    2. Compute all metrics via compute_metrics()
    3. Save confusion matrix figure → output_dir/figures/
    4. Save ROC curve figure → output_dir/figures/
    5. Save metric table (CSV) → output_dir/tables/
    6. Return metrics dict
    """
```

---

#### `experiments/exp_01_densenet_A.py`

```
Entry point: python -m experiments.exp_01_densenet_A

Steps:
1. Load config from configs/densenet121.yaml
2. run_training("dataset_A", config)
3. Load best checkpoint
4. run_evaluation(model, test_loader, device, "densenet121_dataset_A", results/)
5. Print metric summary
```

#### `experiments/exp_02_densenet_B.py`

Same as above but with `"dataset_B"`.

---

### Acceptance Criteria

| Criterion | Pass Condition |
|---|---|
| Training runs | 1 full epoch completes without error for Dataset A |
| Training runs | 1 full epoch completes without error for Dataset B |
| No NaN loss | `torch.isfinite(loss)` is True every step |
| Checkpoint | `save_checkpoint` + `load_checkpoint` round-trip preserves model weights |
| Metrics dict | Contains all 8 keys: accuracy, precision_weighted, precision_per_class, recall_weighted, recall_per_class, f1_weighted, f1_per_class, specificity, roc_auc, confusion_matrix |
| Confusion matrix | Shape (2, 2), non-negative integers, sums to test set size |
| ROC curve | Saved as PNG to `results/figures/` |
| Result table | CSV saved to `results/tables/` with all metric columns |
| GPU | If CUDA available: model and tensors move to GPU without error |

---

## Phase 4 — DenseNet121 + Vision Transformer

> **Blocked on**: Phase 3 fully complete (trained, evaluated, checkpointed)

### Goal
Augment the DenseNet121 backbone with a Vision Transformer to capture global contextual features. Combine local + global features for classification. Establish the CNN + ViT performance level.

### Prerequisites
- Phase 3 complete ✅ (required)
- ViT variant decision: **TBD**
- Feature fusion strategy decision: **TBD** (concatenation vs learned)

### Input / Output

| | Value |
|---|---|
| Input | Same `(B, 3, 224, 224)` tensors |
| DenseNet121 output | Local feature vector (1024-dim from avg pool) |
| ViT output | Global feature vector (dim TBD — depends on variant) |
| Fused feature | `[local ; global]` concatenated or learned fusion |
| Output | Binary logits `(B, 2)` |

---

### Files to Create

#### `models/vit.py`

```python
def build_vit(
    variant: str,             # TBD: "vit_b_16", "vit_s_16", or "custom_small"
    pretrained: bool = True,
    image_size: int = 224,
) -> nn.Module:
    """
    Returns ViT backbone (feature extractor only, no classification head).
    Output: feature vector of dim TBD.
    """
```

---

#### `models/densenet_vit.py`

```python
def build_densenet_vit(
    pretrained: bool = True,
    vit_variant: str = "TBD",
    fusion: str = "concat",   # TBD: "concat" | "learned"
    num_classes: int = 2,
    dropout_rate: float = 0.5,
) -> nn.Module:
    """
    DenseNet121 (local) + ViT (global) → Fusion → Classifier
    """
```

---

#### `configs/densenet_vit.yaml`

```yaml
model:
  pretrained: true
  vit_variant: TBD
  fusion: concat             # TBD
  num_classes: 2
  dropout_rate: 0.5          # TBD

training:
  num_epochs: 30             # TBD
  learning_rate: 1e-4        # TBD — may use differential LR for backbone vs head
  weight_decay: 1e-4         # TBD
  early_stopping_patience: 5 # TBD
  scheduler: cosine          # TBD
  checkpoint_metric: val_loss
```

---

#### `training/train_densenet_vit.py`

```python
def run_training(
    dataset_name: str,
    config: dict,
) -> None:
    """Mirrors train_densenet.py but for the combined model."""
```

Checkpoint save path:
```
checkpoints/densenet_vit_{dataset_name}_best.pt
```

---

#### `experiments/exp_03_densenet_vit_A.py`
#### `experiments/exp_04_densenet_vit_B.py`

Same pattern as Phase 3 experiments.

---

### Acceptance Criteria

| Criterion | Pass Condition |
|---|---|
| No shape errors | ViT input/output shapes verified at 224×224 |
| Feature fusion | Fused tensor has correct dimensionality |
| Training | 1 full epoch on Dataset A and B without error |
| All metrics | Same 8 metrics as Phase 3 computed and saved |
| Checkpoint | Save/load round-trip verified |

---

## Phase 5 — Learnable Feature Compression

> **Blocked on**: Phase 4 fully complete

### Goal
Add a learnable MLP compression block between the fused classical features (DenseNet121 + ViT) and the quantum encoding layer. Produce a compact vector whose dimensionality matches the number of qubits (TBD). Verify the compact vector is in an encoding-compatible range.

### Prerequisites
- Phase 4 complete ✅ (required)
- Number of qubits decided: **TBD**
- Compressed dimensionality decided: **TBD** (= qubit count)
- Encoding range requirement: **TBD** (e.g., [-π, π] or [0, π])

### Input / Output

| | Value |
|---|---|
| Input | Fused DenseNet121 + ViT feature vector (high-dim) |
| Compression output | Compact vector of dim TBD |
| Activation | TBD |
| Normalization | TBD (pre-encoding; may use Tanh to bound range) |

---

### Files to Create

#### `models/compression.py`

```python
class FeatureCompressor(nn.Module):
    """
    MLP: input_dim → hidden_dim (TBD) → output_dim
    output_dim = n_qubits (TBD)
    Activation: TBD
    Optional: LayerNorm or Tanh at output to control encoding range
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,    # = n_qubits (TBD)
        hidden_dim: int,    # TBD
        activation: str,    # TBD: "relu" | "tanh"
        normalize_output: bool,  # TBD
    ):
        ...

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input:  (B, input_dim)
        Output: (B, output_dim)  — range controlled for angle encoding
        """
```

---

#### `models/compressed_model.py`

```python
def build_compressed_model(
    pretrained: bool = True,
    vit_variant: str = "TBD",
    fusion: str = "concat",
    n_qubits: int = TBD,
    compression_hidden_dim: int = TBD,
    activation: str = "TBD",
) -> nn.Module:
    """
    DenseNet121 + ViT + FeatureCompressor pipeline.
    Output: compact vector of shape (B, n_qubits).
    No classifier head — output feeds into VQC (Phase 6).
    """
```

---

#### `configs/compression.yaml`

```yaml
model:
  pretrained: true
  vit_variant: TBD
  fusion: concat
  n_qubits: TBD
  compression_hidden_dim: TBD
  activation: TBD
  normalize_output: TBD

training:
  num_epochs: TBD
  learning_rate: TBD
  weight_decay: TBD
  early_stopping_patience: TBD
  checkpoint_metric: val_loss
```

---

### Acceptance Criteria

| Criterion | Pass Condition |
|---|---|
| Dimensionality | `compressed_model(x).shape == (B, n_qubits)` |
| Encoding range | All values within target encoding range (TBD) |
| Differentiable | `loss.backward()` propagates through compressor without error |
| No NaN | Compressed vector contains no NaN or Inf values |
| Integration | Ready to accept VQC as next module (shape verified) |

---

## Notes for Future Phases

### Phase 6 — VQC (Not Yet Planned in Detail)

> Blocked on: quantum framework decision (PennyLane or Qiskit), n_qubits, and Phase 5 complete.

Key decisions needed before planning:
1. PennyLane vs Qiskit
2. Number of qubits
3. Circuit depth (layers)
4. Gradient method (parameter shift vs adjoint)
5. PyTorch integration approach

Once decided, create:
- `models/vqc.py`
- `models/full_model.py` (end-to-end)
- `configs/vqc.yaml`
- `training/train_full_model.py`
- `experiments/exp_05_full_model_A.py`
- `experiments/exp_06_full_model_B.py`

---

## File Creation Order (Dependency Chain)

```
configs/densenet121.yaml
models/densenet121.py
training/trainer.py
training/checkpoint.py
training/train_densenet.py
evaluation/metrics.py
evaluation/evaluate.py
experiments/exp_01_densenet_A.py
experiments/exp_02_densenet_B.py
      ↓
configs/densenet_vit.yaml
models/vit.py
models/densenet_vit.py
training/train_densenet_vit.py
experiments/exp_03_densenet_vit_A.py
experiments/exp_04_densenet_vit_B.py
      ↓
configs/compression.yaml
models/compression.py
models/compressed_model.py
      ↓
[Phase 6 — VQC — after framework decision]
```

