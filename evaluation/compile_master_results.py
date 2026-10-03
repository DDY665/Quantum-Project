import os
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TABLES_DIR = PROJECT_ROOT / "results" / "tables"
FIGURES_DIR = PROJECT_ROOT / "results" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)

EXPERIMENTS = [
    {
        "exp_id": "Exp 01",
        "model": "DenseNet121 Baseline (1024d)",
        "dataset": "Dataset A (Kermany)",
        "file": "densenet121_dataset_A_metrics.json",
    },
    {
        "exp_id": "Exp 02",
        "model": "DenseNet121 Baseline (1024d)",
        "dataset": "Dataset B (Kermany+RSNA)",
        "file": "densenet121_dataset_B_metrics.json",
    },
    {
        "exp_id": "Exp 03",
        "model": "DenseNet121 + ViT-B/16 (1792d)",
        "dataset": "Dataset A (Kermany)",
        "file": "densenet_vit_dataset_A_metrics.json",
    },
    {
        "exp_id": "Exp 04",
        "model": "DenseNet121 + ViT-B/16 (1792d)",
        "dataset": "Dataset B (Kermany+RSNA)",
        "file": "densenet_vit_dataset_B_metrics.json",
    },
    {
        "exp_id": "Exp 05",
        "model": "Classical 8-Dim Bottleneck",
        "dataset": "Dataset A (Kermany)",
        "file": "compression_dataset_A_metrics.json",
    },
    {
        "exp_id": "Exp 06",
        "model": "Classical 8-Dim Bottleneck",
        "dataset": "Dataset B (Kermany+RSNA)",
        "file": "compression_dataset_B_metrics.json",
    },
    {
        "exp_id": "Exp 07",
        "model": "8-Qubit Quantum VQC (PennyLane)",
        "dataset": "Dataset A (Kermany)",
        "file": "vqc_dataset_A_metrics.json",
    },
    {
        "exp_id": "Exp 08",
        "model": "8-Qubit Quantum VQC (PennyLane)",
        "dataset": "Dataset B (Kermany+RSNA)",
        "file": "vqc_dataset_B_metrics.json",
    },
]


def main():
    rows = []
    for exp in EXPERIMENTS:
        path = TABLES_DIR / exp["file"]
        with open(path, "r") as f:
            data = json.load(f)

        acc = data.get("accuracy", 0.0) * 100
        f1 = data.get("f1_weighted", 0.0) * 100
        roc_auc = data.get("roc_auc", 0.0) * 100
        spec = data.get("specificity", 0.0) * 100
        prec = data.get("precision_weighted", 0.0) * 100
        sens = (
            data.get("recall_per_class", [0, 0])[1] * 100
            if len(data.get("recall_per_class", [])) > 1
            else data.get("recall_weighted", 0.0) * 100
        )

        rows.append(
            {
                "Experiment": exp["exp_id"],
                "Model Architecture": exp["model"],
                "Dataset": exp["dataset"],
                "Accuracy (%)": round(acc, 2),
                "F1-Score (%)": round(f1, 2),
                "ROC-AUC (%)": round(roc_auc, 2),
                "Specificity (%)": round(spec, 2),
                "Sensitivity (%)": round(sens, 2),
                "Precision (%)": round(prec, 2),
            }
        )

    df = pd.DataFrame(rows)

    # Save CSV
    csv_path = TABLES_DIR / "master_benchmark_table.csv"
    df.to_csv(csv_path, index=False)
    print(f"[+] Master CSV saved to {csv_path}")

    # Save Markdown
    md_path = TABLES_DIR / "master_benchmark_table.md"
    with open(md_path, "w") as f:
        f.write("# Master Benchmark Results: AI-Quantum Pneumonia Detection\n\n")
        f.write(df.to_markdown(index=False))
        f.write("\n")
    print(f"[+] Master Markdown saved to {md_path}")

    # Plot Master Comparative Chart
    models = ["DenseNet121", "DenseNet + ViT", "8-Dim Bottleneck", "8-Qubit VQC"]

    # Extract values for Dataset A and Dataset B
    acc_A = [
        df.loc[
            (df["Model Architecture"] == m) & (df["Dataset"].str.contains("Dataset A")),
            "Accuracy (%)",
        ].values[0]
        for m in [
            "DenseNet121 Baseline (1024d)",
            "DenseNet121 + ViT-B/16 (1792d)",
            "Classical 8-Dim Bottleneck",
            "8-Qubit Quantum VQC (PennyLane)",
        ]
    ]
    acc_B = [
        df.loc[
            (df["Model Architecture"] == m) & (df["Dataset"].str.contains("Dataset B")),
            "Accuracy (%)",
        ].values[0]
        for m in [
            "DenseNet121 Baseline (1024d)",
            "DenseNet121 + ViT-B/16 (1792d)",
            "Classical 8-Dim Bottleneck",
            "8-Qubit Quantum VQC (PennyLane)",
        ]
    ]

    auc_A = [
        df.loc[
            (df["Model Architecture"] == m) & (df["Dataset"].str.contains("Dataset A")),
            "ROC-AUC (%)",
        ].values[0]
        for m in [
            "DenseNet121 Baseline (1024d)",
            "DenseNet121 + ViT-B/16 (1792d)",
            "Classical 8-Dim Bottleneck",
            "8-Qubit Quantum VQC (PennyLane)",
        ]
    ]
    auc_B = [
        df.loc[
            (df["Model Architecture"] == m) & (df["Dataset"].str.contains("Dataset B")),
            "ROC-AUC (%)",
        ].values[0]
        for m in [
            "DenseNet121 Baseline (1024d)",
            "DenseNet121 + ViT-B/16 (1792d)",
            "Classical 8-Dim Bottleneck",
            "8-Qubit Quantum VQC (PennyLane)",
        ]
    ]

    x = np.arange(len(models))
    width = 0.38

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Accuracy subplot
    rects1 = ax1.bar(
        x - width / 2,
        acc_A,
        width,
        label="Dataset A (Kermany)",
        color="#1f77b4",
        edgecolor="black",
        alpha=0.9,
    )
    rects2 = ax1.bar(
        x + width / 2,
        acc_B,
        width,
        label="Dataset B (Kermany+RSNA)",
        color="#ff7f0e",
        edgecolor="black",
        alpha=0.9,
    )
    ax1.set_ylabel("Test Accuracy (%)", fontsize=12, fontweight="bold")
    ax1.set_title(
        "Test Accuracy Comparison across Architectures",
        fontsize=13,
        fontweight="bold",
        pad=12,
    )
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, fontsize=10, fontweight="bold")
    ax1.set_ylim(60, 105)
    ax1.legend(loc="lower left", frameon=True)
    ax1.grid(axis="y", linestyle="--", alpha=0.5)

    for rect in rects1 + rects2:
        h = rect.get_height()
        ax1.annotate(
            f"{h:.1f}%",
            xy=(rect.get_x() + rect.get_width() / 2, h),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    # ROC-AUC subplot
    rects3 = ax2.bar(
        x - width / 2,
        auc_A,
        width,
        label="Dataset A (Kermany)",
        color="#2ca02c",
        edgecolor="black",
        alpha=0.9,
    )
    rects4 = ax2.bar(
        x + width / 2,
        auc_B,
        width,
        label="Dataset B (Kermany+RSNA)",
        color="#d62728",
        edgecolor="black",
        alpha=0.9,
    )
    ax2.set_ylabel("ROC-AUC Score (%)", fontsize=12, fontweight="bold")
    ax2.set_title(
        "ROC-AUC Comparison across Architectures",
        fontsize=13,
        fontweight="bold",
        pad=12,
    )
    ax2.set_xticks(x)
    ax2.set_xticklabels(models, fontsize=10, fontweight="bold")
    ax2.set_ylim(90, 103)
    ax2.legend(loc="lower left", frameon=True)
    ax2.grid(axis="y", linestyle="--", alpha=0.5)

    for rect in rects3 + rects4:
        h = rect.get_height()
        ax2.annotate(
            f"{h:.1f}%",
            xy=(rect.get_x() + rect.get_width() / 2, h),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    plt.suptitle(
        "AI-Quantum Pneumonia Detection: Classical vs Quantum Performance Across Cohorts",
        fontsize=15,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    chart_path = FIGURES_DIR / "master_benchmark_comparison.png"
    plt.savefig(chart_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[+] Master benchmark chart saved to {chart_path}")

    print("\n" + "=" * 80)
    print("MASTER BENCHMARK EVALUATION SUMMARY")
    print("=" * 80)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
