# Architecture Specification
> Hybrid Quantum-Classical Pneumonia Detection

---

## 1. Full System Pipeline (ASCII)

```
┌─────────────────────────────────────────────────────────────┐
│                      INPUT                                  │
│          Chest X-ray Image (224 × 224, 3-channel)           │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                  DATA PREPROCESSING                         │
│  • Validation / corrupted-image check                       │
│  • Resize → 224 × 224                                       │
│  • Grayscale → 3-channel (replicate)                        │
│  • Normalize: μ=[0.485,0.456,0.406] σ=[0.229,0.224,0.225]  │
│  • Train only: RandomHorizontalFlip(p=0.5), Rotation(±5°)   │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│              STAGE 1 — DenseNet121                          │
│  • Pretrained on ImageNet (torchvision)                     │
│  • Dense skip-connections for local feature reuse           │
│  • Output: feature map / flattened feature vector           │
│  • Classifier head → Binary output                          │
│                                                             │
│  Used in: Model 1, Model 2 (backbone), Model 3 (backbone)  │
└──────────────────────────┬──────────────────────────────────┘
                           │  local features
                           ▼
┌─────────────────────────────────────────────────────────────┐
│              STAGE 2 — Vision Transformer (ViT)             │
│  • Variant: TBD (e.g. ViT-B/16, ViT-S/16, custom small)    │
│  • Patch-based self-attention for global relationships      │
│  • Input: feature patches from DenseNet121 output           │
│  • Output: global contextual feature vector                 │
│  • Classifier head → Binary output                          │
│                                                             │
│  Used in: Model 2, Model 3                                  │
└──────────────────────────┬──────────────────────────────────┘
                           │  global features
                           ▼
┌─────────────────────────────────────────────────────────────┐
│          STAGE 3 — Learnable Feature Compression            │
│  • Fully-connected / MLP compression block                  │
│  • Input: concatenated or fused DenseNet121 + ViT features  │
│  • Output: compact vector of dimensionality TBD             │
│    (target dimension determined by number of qubits)        │
│  • Compression is learned (not hand-crafted)                │
│                                                             │
│  Used in: Model 3                                           │
└──────────────────────────┬──────────────────────────────────┘
                           │  compact feature vector
                           ▼
┌─────────────────────────────────────────────────────────────┐
│              STAGE 4 — Angle Encoding                       │
│  • Maps each feature value to a qubit rotation angle        │
│  • Encoding strategy: angle/amplitude (TBD per framework)   │
│  • Number of qubits: TBD                                    │
│                                                             │
│  Used in: Model 3                                           │
└──────────────────────────┬──────────────────────────────────┘
                           │  encoded quantum state
                           ▼
┌─────────────────────────────────────────────────────────────┐
│         STAGE 4 — Variational Quantum Circuit (VQC)         │
│  • Parameterised rotation gates + entanglement layers       │
│  • Number of qubits: TBD                                    │
│  • Number of layers / depth: TBD                            │
│  • Quantum framework: TBD (PennyLane or Qiskit)             │
│  • Output: expectation value(s) → binary classification     │
│                                                             │
│  Used in: Model 3                                           │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                   OUTPUT / EVALUATION                       │
│            NON_PNEUMONIA (0) │ PNEUMONIA (1)                │
│  Metrics: Accuracy, Precision, Recall, F1, Specificity,     │
│           ROC-AUC, Confusion Matrix, ROC Curve              │
└─────────────────────────────────────────────────────────────┘
```

---

## 2. Component Specifications

### 2.1 DenseNet121

| Property | Value |
|---|---|
| Source | `torchvision.models.densenet121` |
| Pretrained weights | ImageNet |
| Fine-tuning strategy | TBD (full fine-tune vs frozen backbone) |
| Output (as backbone) | Feature vector (1024-dim from avg pool) |
| Output (as standalone) | 2-class logits (binary classifier head) |
| Used in | Model 1, 2, 3 |

**Classifier head (Model 1 standalone):**
```
DenseNet121 features (1024)
→ Dropout
→ Linear(1024, 2)
→ Softmax / Sigmoid
```

---

### 2.2 Vision Transformer (ViT)

| Property | Value |
|---|---|
| Variant | TBD (ViT-B/16, ViT-S/16, or custom small ViT) |
| Input | Patch embeddings from DenseNet121 output or raw image patches |
| Patch size | TBD |
| Output | Global feature vector (CLS token embedding) |
| Pretrained | TBD |
| Used in | Model 2, 3 |

**Classifier head (Model 2):**
```
DenseNet121 local features + ViT global features
→ Concatenation / fusion
→ Linear → 2-class output
```

---

### 2.3 Learnable Feature Compression

| Property | Value |
|---|---|
| Type | MLP (fully-connected block) |
| Input | Fused DenseNet121 + ViT feature vector |
| Output dimensionality | TBD (equal to number of qubits) |
| Activation | TBD (ReLU, Tanh, or learned) |
| Normalization | TBD (BatchNorm / LayerNorm before encoding) |
| Used in | Model 3 |

---

### 2.4 Angle Encoding

| Property | Value |
|---|---|
| Encoding type | Angle encoding (rotation-based) |
| Mapping | Each scalar feature → rotation angle on a qubit |
| Number of qubits | TBD (matches compressed feature dimension) |
| Framework | TBD (PennyLane or Qiskit) |

---

### 2.5 Variational Quantum Circuit (VQC)

| Property | Value |
|---|---|
| Number of qubits | TBD |
| Circuit depth (layers) | TBD |
| Gate types | Parameterised RX / RY / RZ + CNOT entanglement |
| Measurement | Pauli-Z expectation value |
| Optimizer compatibility | Compatible with PyTorch autograd (via quantum framework) |
| Framework | TBD — PennyLane or Qiskit |

---

## 3. Feature Flow by Model

```
MODEL 1 (DenseNet121 only)
  Image → DenseNet121 → FC Classifier → {0, 1}

MODEL 2 (DenseNet121 + ViT)
  Image → DenseNet121 → Local Features ─┐
                                        ├→ Fusion → FC Classifier → {0, 1}
  Image → ViT          → Global Features ┘

MODEL 3 (Full Hybrid)
  Image → DenseNet121 → Local Features ─┐
                                        ├→ Fusion → Compression → Encoding → VQC → {0, 1}
  Image → ViT          → Global Features ┘
```

---

## 4. Open Architecture Decisions (TBD)

| Decision | Status | Notes |
|---|---|---|
| ViT variant | TBD | ViT-B/16 vs ViT-S/16 vs custom small ViT |
| DenseNet121 fine-tuning strategy | TBD | Full vs partial (frozen early layers) |
| Feature fusion method | TBD | Concatenation vs learned attention fusion |
| ViT variant | **Decided** | `vit_b_16` chosen |
| DenseNet121 fine-tuning strategy | **Decided** | Full fine-tune |
| Feature fusion method | **Decided** | Simple concatenation |
| Compressed vector dimensionality | TBD | Determined by qubit count |
| Number of VQC qubits | TBD | — |
| VQC circuit depth | TBD | — |
| Quantum framework | TBD | PennyLane or Qiskit |
| Compression layer activation | TBD | — |
| Normalization before encoding | TBD | — |

---

## 5. Training Configuration (Planned)

| Parameter | Value |
|---|---|
| Image size | 224 × 224 |
| Batch size | 32 |
| Optimizer | TBD (Adam / AdamW typical) |
| Learning rate | TBD |
| Scheduler | TBD |
| Loss function | Binary Cross-Entropy (BCEWithLogitsLoss) or CrossEntropyLoss |
| Max epochs | TBD |
| Early stopping | TBD |
| Checkpoint metric | Validation loss or F1 |
| GPU | Intended; exact hardware TBD |

---

## 6. Evaluation Protocol

| Metric | Computed for |
|---|---|
| Accuracy | All models × all datasets |
| Precision (weighted + per-class) | All models × all datasets |
| Recall / Sensitivity (weighted + per-class) | All models × all datasets |
| F1-score (weighted + per-class) | All models × all datasets |
| Specificity | All models × all datasets |
| ROC-AUC | All models × all datasets |
| Confusion Matrix | All models × all datasets |
| ROC Curve (figure) | All models × all datasets |

Results are saved to:
- Figures → `results/figures/`
- Tables → `results/tables/`
- Logs → `results/logs/`

> **CAUTION**: Do not report numerical results until experiments are completed.

