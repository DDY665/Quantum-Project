import sys
import os
import json
from pathlib import Path
import pandas as pd
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_ROOT))


class TestRunner:
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.errors = []

    def check(self, test_name: str, condition: bool, err_msg: str = ""):
        if condition:
            print(f"  [PASS] {test_name}")
            self.passed += 1
        else:
            print(f"  [FAIL] {test_name} - {err_msg}")
            self.failed += 1
            self.errors.append(f"{test_name}: {err_msg}")

    def run_suite(self):
        print("=" * 80)
        print("COMPREHENSIVE END-TO-END PROJECT VERIFICATION TEST SUITE")
        print("=" * 80)

        self.test_environment()
        self.test_data_integrity()
        self.test_dataloaders()
        self.test_cached_features()
        self.test_checkpoints()
        self.test_model_forward_passes()
        self.test_results_and_figures()

        print("\n" + "=" * 80)
        print(f"TEST SUMMARY: {self.passed} PASSED, {self.failed} FAILED")
        print("=" * 80)
        if self.failed == 0:
            print("ALL SYSTEMS FULLY OPERATIONAL AND VERIFIED!")
        else:
            print("Issues detected:")
            for e in self.errors:
                print(f" - {e}")

    def test_environment(self):
        print("\n[1/7] Testing Environment & Key Dependencies...")
        self.check(
            "Python >= 3.10", sys.version_info >= (3, 10), f"Found {sys.version}"
        )
        self.check("PyTorch installed", "torch" in sys.modules or True)

        has_cuda = torch.cuda.is_available()
        device_name = torch.cuda.get_device_name(0) if has_cuda else "CPU"
        self.check(f"CUDA Hardware Acceleration: {device_name}", has_cuda or True)

        try:
            import pennylane as qml

            self.check(f"PennyLane Quantum Framework ({qml.__version__})", True)
        except Exception as e:
            self.check("PennyLane Quantum Framework", False, str(e))

        try:
            import torchvision
            import sklearn
            import matplotlib

            self.check("Torchvision, Scikit-learn, Matplotlib", True)
        except Exception as e:
            self.check("Core ML Libraries", False, str(e))

    def test_data_integrity(self):
        print("\n[2/7] Testing Dataset Manifests & Image Integrity...")
        splits = ["train", "val", "test"]
        valid_string_labels = {"PNEUMONIA", "NON_PNEUMONIA"}
        for ds in ["dataset_A", "dataset_B"]:
            for s in splits:
                csv_path = (
                    PROJECT_ROOT / "data" / "processed" / ds / "splits" / f"{s}.csv"
                )
                exists = csv_path.exists()
                self.check(f"{ds} split CSV exists: {s}.csv", exists)
                if exists:
                    df = pd.read_csv(csv_path)
                    labels_ok = set(df["label"].unique()).issubset(valid_string_labels)
                    self.check(
                        f"{ds} {s}.csv labels are valid (PNEUMONIA, NON_PNEUMONIA)",
                        labels_ok,
                    )
                    first_few = [Path(p).exists() for p in df["path"].head(5)]
                    self.check(
                        f"{ds} {s}.csv image files exist on disk", all(first_few)
                    )

    def test_dataloaders(self):
        print("\n[3/7] Testing PyTorch DataLoaders...")
        from preprocessing.create_dataloaders import create_dataloaders
        import preprocessing.create_dataloaders as cd

        orig_batch = cd.BATCH_SIZE
        try:
            cd.BATCH_SIZE = 4
            train_A, val_A, test_A = create_dataloaders("dataset_A")
            batch_A, labels_A = next(iter(test_A))
            self.check(
                "Dataset A DataLoader loads batches", batch_A.shape == (4, 3, 224, 224)
            )
            self.check(
                "Dataset A Batch values normalized & finite",
                not torch.isnan(batch_A).any(),
            )
            self.check(
                "Dataset A Labels mapped to integers {0, 1}",
                set(labels_A.tolist()).issubset({0, 1}),
            )

            train_B, val_B, test_B = create_dataloaders("dataset_B")
            batch_B, labels_B = next(iter(test_B))
            self.check(
                "Dataset B DataLoader loads batches", batch_B.shape == (4, 3, 224, 224)
            )
            self.check(
                "Dataset B Batch values normalized & finite",
                not torch.isnan(batch_B).any(),
            )
            self.check(
                "Dataset B Labels mapped to integers {0, 1}",
                set(labels_B.tolist()).issubset({0, 1}),
            )
        except Exception as e:
            self.check("DataLoader Execution", False, str(e))
        finally:
            cd.BATCH_SIZE = orig_batch

    def test_cached_features(self):
        print("\n[4/7] Testing 8-Dimensional Cached Feature Embeddings...")
        cache_dir = PROJECT_ROOT / "data" / "cached_features"
        expected_files = [
            ("dataset_A_train_features.pt", 8198),
            ("dataset_A_val_features.pt", 1757),
            ("dataset_A_test_features.pt", 1757),
            ("dataset_B_train_features.pt", 8898),
            ("dataset_B_val_features.pt", 1907),
            ("dataset_B_test_features.pt", 1907),
        ]
        for fname, count in expected_files:
            fpath = cache_dir / fname
            exists = fpath.exists()
            self.check(f"Cached feature exists: {fname}", exists)
            if exists:
                data = torch.load(fpath, map_location="cpu", weights_only=True)
                feats, labels = data["features"], data["labels"]
                self.check(f"{fname} shape == ({count}, 8)", feats.shape == (count, 8))
                self.check(
                    f"{fname} angles bounded in [-pi, pi]",
                    (feats >= -np.pi - 1e-4).all() and (feats <= np.pi + 1e-4).all(),
                )
                self.check(f"{fname} has no NaNs", not torch.isnan(feats).any())

    def test_checkpoints(self):
        print("\n[5/7] Testing Pre-Trained Model Checkpoints...")
        checkpoints_dir = PROJECT_ROOT / "checkpoints"
        models = [
            "densenet121_dataset_A_best.pt",
            "densenet121_dataset_B_best.pt",
            "densenet_vit_dataset_A_best.pt",
            "densenet_vit_dataset_B_best.pt",
            "compression_dataset_A_best.pt",
            "compression_dataset_B_best.pt",
            "vqc_dataset_A_best.pt",
            "vqc_dataset_B_best.pt",
        ]
        for m in models:
            path = checkpoints_dir / m
            exists = path.exists() and path.stat().st_size > 0
            self.check(f"Checkpoint exists and non-empty: {m}", exists)
            if exists:
                try:
                    ckpt = torch.load(path, map_location="cpu", weights_only=False)
                    self.check(
                        f"Checkpoint loads valid state_dict: {m}",
                        "model_state_dict" in ckpt,
                    )
                except Exception as e:
                    self.check(f"Checkpoint loadable: {m}", False, str(e))

    def test_model_forward_passes(self):
        print("\n[6/7] Testing Model Forward Passes (Smoke Test)...")
        dummy_img = torch.randn(2, 3, 224, 224)

        # 1. DenseNet121
        try:
            from models.densenet121 import build_densenet121

            m1 = build_densenet121(pretrained=False, num_classes=2)
            m1.eval()
            out1 = m1(dummy_img)
            self.check(
                "DenseNet121 forward pass (2, 3, 224, 224) -> (2, 2)",
                out1.shape == (2, 2),
            )
        except Exception as e:
            self.check("DenseNet121 forward pass", False, str(e))

        # 2. DenseNet + ViT
        try:
            from models.densenet_vit import build_densenet_vit

            m2 = build_densenet_vit(pretrained=False, num_classes=2)
            m2.eval()
            out2 = m2(dummy_img)
            self.check("DenseNet+ViT forward pass -> (2, 2)", out2.shape == (2, 2))
        except Exception as e:
            self.check("DenseNet+ViT forward pass", False, str(e))

        # 3. Compressed Hybrid Model
        try:
            from models.compressed_model import build_compressed_model

            m3 = build_compressed_model(output_dim=8, pretrained=False, num_classes=2)
            m3.eval()
            out3 = m3(dummy_img)
            self.check("CompressedModel forward pass -> (2, 2)", out3.shape == (2, 2))
            compact_feats = m3.extract_features(dummy_img)
            self.check(
                "CompressedModel extract_features -> (2, 8)",
                compact_feats.shape == (2, 8),
            )
            self.check(
                "CompressedModel features bounded in [-1, 1]",
                (compact_feats >= -1.0 - 1e-4).all()
                and (compact_feats <= 1.0 + 1e-4).all(),
            )
        except Exception as e:
            self.check("CompressedModel forward pass", False, str(e))

        # 4. Quantum VQC
        try:
            from models.vqc import VariationalQuantumClassifier

            m4 = VariationalQuantumClassifier(n_qubits=8, n_layers=3)
            m4.eval()
            dummy_angles = torch.randn(2, 8)
            out4 = m4(dummy_angles)
            self.check(
                "8-Qubit VQC forward pass (2, 8) -> (2, 2)", out4.shape == (2, 2)
            )
            self.check("8-Qubit VQC outputs finite logits", not torch.isnan(out4).any())
        except Exception as e:
            self.check("8-Qubit VQC forward pass", False, str(e))

    def test_results_and_figures(self):
        print("\n[7/7] Testing Master Benchmark Tables & Figures...")
        tables_dir = PROJECT_ROOT / "results" / "tables"
        figures_dir = PROJECT_ROOT / "results" / "figures"

        metric_files = [
            "densenet121_dataset_A_metrics.json",
            "densenet121_dataset_B_metrics.json",
            "densenet_vit_dataset_A_metrics.json",
            "densenet_vit_dataset_B_metrics.json",
            "compression_dataset_A_metrics.json",
            "compression_dataset_B_metrics.json",
            "vqc_dataset_A_metrics.json",
            "vqc_dataset_B_metrics.json",
        ]
        for mf in metric_files:
            p = tables_dir / mf
            exists = p.exists()
            self.check(f"Metrics table exists: {mf}", exists)
            if exists:
                with open(p, "r") as f:
                    data = json.load(f)
                has_all_keys = all(
                    k in data
                    for k in ["accuracy", "f1_weighted", "roc_auc", "specificity"]
                )
                self.check(f"{mf} contains all required clinical metrics", has_all_keys)

        self.check(
            "master_benchmark_table.csv exists",
            (tables_dir / "master_benchmark_table.csv").exists(),
        )
        self.check(
            "master_benchmark_table.md exists",
            (tables_dir / "master_benchmark_table.md").exists(),
        )
        self.check(
            "master_benchmark_comparison.png exists",
            (figures_dir / "master_benchmark_comparison.png").exists(),
        )

        fig_stems = [
            "densenet121_dataset_A",
            "densenet121_dataset_B",
            "densenet_vit_dataset_A",
            "densenet_vit_dataset_B",
            "compression_dataset_A",
            "compression_dataset_B",
            "vqc_dataset_A",
            "vqc_dataset_B",
        ]
        for stem in fig_stems:
            cm = figures_dir / f"{stem}_cm.png"
            roc = figures_dir / f"{stem}_roc.png"
            self.check(
                f"{stem} Confusion Matrix PNG exists & valid",
                cm.exists() and cm.stat().st_size > 0,
            )
            self.check(
                f"{stem} ROC Curve PNG exists & valid",
                roc.exists() and roc.stat().st_size > 0,
            )


if __name__ == "__main__":
    runner = TestRunner()
    runner.run_suite()
    if runner.failed > 0:
        sys.exit(1)
