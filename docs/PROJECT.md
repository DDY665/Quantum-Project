# Hybrid Quantum-Classical Pneumonia Detection
> Spec-Driven Reference Document — Personal Use / Cross-Session Context

---

## 1. Research Objective

Develop a hybrid deep-learning + quantum machine-learning framework that classifies chest X-ray images as **PNEUMONIA** or **NON_PNEUMONIA**.

The investigation studies the incremental contribution of:
- CNN feature extraction (local spatial features)
- Transformer-based global context (ViT)
- Learnable feature compression (quantum-compatible representation)
- Variational Quantum Classifier (final classification)

> **IMPORTANT**: Do not claim performance improvements or architectural novelty without experimental evidence.

---

## 2. Research Motivation

| Layer | Role |
|---|---|
| DenseNet121 | Dense skip-connections capture local spatial features at multiple scales |
| Vision Transformer | Self-attention captures long-range global relationships across image patches |
| Learnable Compression | Bridges classical high-dimensional features to quantum-compatible compact vector |
| VQC | Explores quantum interference and entanglement for binary classification |

The incremental design allows measuring each stage's contribution independently via ablation comparison.

---

## 3. Experimental Models

| Model | Architecture | Datasets |
|---|---|---|
| Model 1 | DenseNet121 only | Dataset A, Dataset B |
| Model 2 | DenseNet121 + ViT | Dataset A, Dataset B |
| Model 3 | DenseNet121 + ViT + Compression + VQC | Dataset A, Dataset B |

---

## 4. Datasets

### Dataset A — Primary (Kermany)

| Split | Source |
|---|---|
| TRAIN | Kermany training set |
| VAL | Kermany validation set |
| TEST | Kermany test set |

### Dataset B — Extended (Kermany + RSNA)

| Split | Source |
|---|---|
| TRAIN | Kermany training + 500 RSNA NON_PNEUMONIA + 500 RSNA PNEUMONIA |
| VAL | Same Kermany validation (unchanged) |
| TEST | Same Kermany test (unchanged) |

### Label Mapping

| Original Label | Project Label | Integer |
|---|---|---|
| NORMAL | NON_PNEUMONIA | 0 |
| PNEUMONIA | PNEUMONIA | 1 |

### Key Rules
- Raw datasets are **never modified**.
- Kermany VAL and TEST are **identical** across Dataset A and B (fair comparison).
- RSNA images are **training-only additions**.

---

## 5. Preprocessing Pipeline (COMPLETED)

```
Step 1  validate_kermany()          — corrupted / unreadable image check
Step 2  check_kermany_duplicates()  — perceptual hash duplicate detection
Step 3  select_rsna_subset()        — select 500 NON_PNEUMONIA + 500 PNEUMONIA from RSNA
Step 4  convert_selected_rsna()     — DICOM → JPEG/PNG conversion
Step 5  build_datasets()            — assemble Dataset A and Dataset B manifests
Step 6  verify_datasets()           — final integrity verification
```

**Entry point:** `preprocessing/pipeline.py` → `run_complete_pipeline()`

### DataLoader Configuration (from `preprocessing/create_dataloaders.py`)

| Parameter | Value |
|---|---|
| Image size | 224 × 224 |
| Batch size | 32 |
| Normalization mean | [0.485, 0.456, 0.406] |
| Normalization std | [0.229, 0.224, 0.225] |
| Grayscale → 3-channel | Yes |
| Train augmentation | Random horizontal flip (p=0.5), Random rotation (±5°) |
| Val/Test augmentation | None |
| Random seed | 42 |
| Pin memory | Auto (True if CUDA available) |

---

## 6. Technology Stack

| Category | Library / Tool | Version |
|---|---|---|
| Language | Python | (see `.python-version`) |
| Deep Learning | PyTorch | 2.8.0 |
| Vision | torchvision | 0.23.0 |
| Data | Pandas | 2.3.2 |
| Numerics | NumPy | 2.3.2 |
| Image I/O | Pillow | 11.3.0 |
| DICOM | pydicom | 3.0.1 |
| Duplicate detection | ImageHash | 4.3.2 |
| Plotting | matplotlib | 3.10.5 |
| Plotting | seaborn | 0.13.2 |
| Quantum Framework | **TBD** — PennyLane or Qiskit | TBD |
| Package manager | uv | (see `uv.lock`) |

---

## 7. Folder Structure

```
pneumonia_quantum_project/
├── data/
│   ├── raw/
│   │   ├── kermany/
│   │   └── rsna/
│   └── processed/
│       ├── dataset_A/splits/{train,val,test}.csv
│       ├── dataset_B/splits/{train,val,test}.csv
│       └── rsna_selected/
├── preprocessing/
│   ├── __init__.py
│   ├── inspect_dataset.py
│   ├── clean_images.py
│   ├── check_duplicates.py
│   ├── select_rsna.py
│   ├── rsna_converter.py
│   ├── build_dataset.py
│   ├── prepare_dataset.py
│   ├── dataset_report.py
│   ├── create_dataloaders.py
│   └── pipeline.py
├── models/              ← CNN, ViT, Compression, VQC modules go here
├── training/            ← Training loops, schedulers, checkpoint logic
├── evaluation/          ← Metric computation, confusion matrix, ROC
├── experiments/         ← Per-experiment runner scripts
├── configs/             ← YAML/JSON config files (hyperparameters)
├── checkpoints/         ← Saved model weights (.pt / .pth)
├── results/
│   ├── figures/         ← ROC curves, confusion matrices, loss plots
│   ├── tables/          ← Per-metric CSV result tables
│   └── logs/            ← Training logs, preprocessing reports
├── main.py
├── requirements.txt
├── PROJECT.md           ← this file
├── ARCHITECTURE.md
├── TASKS.md
├── IMPLEMENTATION_PLAN.md
└── README.md
```

---

## 8. Evaluation Protocol

Metrics computed for **every model × dataset combination**:

| Metric | Notes |
|---|---|
| Accuracy | Overall correct classifications |
| Precision | Per-class and weighted |
| Recall (Sensitivity) | Per-class and weighted |
| F1-score | Per-class and weighted |
| Specificity | True negative rate for NON_PNEUMONIA |
| ROC-AUC | Area under ROC curve |
| Confusion Matrix | Absolute counts |
| ROC Curve plot | Saved to `results/figures/` |

> **CAUTION**: Do not report numerical values until experiments are completed.

All models are evaluated on the **same Kermany test set** for a fair comparison.

---

## 9. Design Principles (Non-Negotiable)

1. Do not invent implementation details not present in the codebase or context.
2. Do not change the architecture without discussion.
3. Keep Dataset A and Dataset B separate throughout.
4. Keep the test set identical across all experiments.
5. Raw datasets are never modified.
6. Build and test components incrementally (stage by stage).
7. Establish DenseNet121 before adding ViT.
8. Establish CNN + ViT before adding quantum components.
9. Feature compression always sits between classical extraction and VQC.
10. Teacher-student / knowledge distillation is **excluded** unless explicitly requested.
11. Do not claim performance improvements before experiments.
12. If information is missing: state **"Not available in project context."**

---

## 10. Development Phase Status

| Phase | Status |
|---|---|
| Literature Survey | ✅ COMPLETED |
| Dataset Preparation | ✅ COMPLETED |
| Data Preprocessing | ✅ COMPLETED |
| Stage 1 — DenseNet121 CNN Baseline | 🔄 IN PROGRESS (NEXT) |
| Stage 2 — DenseNet121 + ViT | ⏳ REMAINING |
| Stage 3 — Feature Compression | ⏳ REMAINING |
| Stage 4 — VQC | ⏳ REMAINING |
| Full Model Training | ⏳ REMAINING |
| Evaluation | ⏳ REMAINING |
| Integration & Testing | ⏳ REMAINING |
| Deployment | 🚫 Not in project scope |
| Documentation | 🔄 IN PROGRESS |

