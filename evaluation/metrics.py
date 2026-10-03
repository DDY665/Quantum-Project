import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)

def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
) -> dict:
    """
    Computes all standard evaluation metrics for binary classification.
    """
    acc = accuracy_score(y_true, y_pred)
    
    precision_w = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    precision_pc = precision_score(y_true, y_pred, average=None, zero_division=0).tolist()
    
    recall_w = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    recall_pc = recall_score(y_true, y_pred, average=None, zero_division=0).tolist()
    
    f1_w = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    f1_pc = f1_score(y_true, y_pred, average=None, zero_division=0).tolist()
    
    # Specificity = TN / (TN + FP) for class 0
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    else:
        specificity = 0.0
        
    try:
        auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auc = 0.0  # Safe fallback

    return {
        "accuracy": float(acc),
        "precision_weighted": float(precision_w),
        "precision_per_class": precision_pc,
        "recall_weighted": float(recall_w),
        "recall_per_class": recall_pc,
        "f1_weighted": float(f1_w),
        "f1_per_class": f1_pc,
        "specificity": float(specificity),
        "roc_auc": float(auc),
        "confusion_matrix": cm
    }

