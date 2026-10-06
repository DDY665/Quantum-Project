import os
import sys
import argparse
import math
from pathlib import Path
from PIL import Image
import torch
import torchvision.transforms as transforms

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_ROOT))

from models.compressed_model import build_compressed_model
from models.vqc import build_vqc

def get_default_sample():
    # Try finding a pneumonia sample from dataset A or RSNA
    candidates = [
        PROJECT_ROOT / "data" / "raw" / "kermany" / "chest_xray" / "chest_xray" / "test" / "PNEUMONIA",
        PROJECT_ROOT / "data" / "processed" / "rsna_selected" / "PNEUMONIA",
    ]
    for folder in candidates:
        if folder.exists():
            files = list(folder.glob("*.jpeg")) + list(folder.glob("*.png"))
            if files:
                return files[0], "PNEUMONIA"
    return None, "UNKNOWN"

def run_demo(image_path: Path = None, ground_truth: str = None):
    print("\n" + "=" * 70)
    print("HYBRID QUANTUM-CLASSICAL PNEUMONIA DETECTION — LIVE INFERENCE DEMO")
    print("=" * 70)

    if image_path is None or not image_path.exists():
        sample_file, gt = get_default_sample()
        if sample_file is None:
            print("[!] No sample chest X-ray image found on disk.")
            return
        image_path = sample_file
        ground_truth = gt

    print(f"\n[+] Input Chest X-Ray: {image_path.name}")
    if ground_truth:
        print(f"[+] Ground Truth Label: {ground_truth}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[+] Device Hardware:   {device}")

    # 1. Image Preprocessing
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    raw_img = Image.open(image_path).convert("L")
    input_tensor = transform(raw_img).unsqueeze(0).to(device) # Shape: (1, 3, 224, 224)
    print(f"[+] Preprocessed Input Tensor: {tuple(input_tensor.shape)}")

    # 2. Stage 1 & 2: Classical Dual-Branch Backbone + 8-Dim Compression
    print("\n--- STAGE 1: Classical Dual-Branch Feature Extraction ---")
    comp_ckpt_path = PROJECT_ROOT / "checkpoints" / "compression_dataset_A_best.pt"
    if not comp_ckpt_path.exists():
        print(f"[!] Checkpoint not found: {comp_ckpt_path}")
        return

    comp_model = build_compressed_model(
        output_dim=8,
        hidden_dim=128,
        num_classes=2,
        pretrained=False,
        dropout_rate=0.0,
        use_tanh_scaling=True
    ).to(device)

    comp_ckpt = torch.load(comp_ckpt_path, map_location=device)
    comp_model.load_state_dict(comp_ckpt["model_state_dict"], strict=True)
    comp_model.eval()

    with torch.no_grad():
        local_f = comp_model.densenet(input_tensor)
        global_f = comp_model.vit(input_tensor)
        fused_f = torch.cat([local_f, global_f], dim=1)
        z_latent = comp_model.compressor(fused_f)

    print(f"  * DenseNet-121 Local Features:     {tuple(local_f.shape)}  (1,024 dims)")
    print(f"  * ViT-B/16 Global Context Features: {tuple(global_f.shape)}  (768 dims)")
    print(f"  * Concatenated Multi-Scale Vector: {tuple(fused_f.shape)}  (1,792 dims)")

    print("\n--- STAGE 2: 8-Dimensional Bottleneck Compression ---")
    print(f"  * Compressed Representation:       {tuple(z_latent.shape)}  (8 dims, Tanh bounded in [-1, 1])")
    latent_vals = [round(float(v), 4) for v in z_latent[0].cpu().numpy()]
    print(f"  * Latent Vector [z_0 ... z_7]:     {latent_vals}")

    # 3. Stage 3: 8-Qubit PennyLane Quantum Classifier
    print("\n--- STAGE 3: 8-Qubit Variational Quantum Classifier (PennyLane) ---")
    vqc_ckpt_path = PROJECT_ROOT / "checkpoints" / "vqc_dataset_A_best.pt"
    if not vqc_ckpt_path.exists():
        print(f"[!] Checkpoint not found: {vqc_ckpt_path}")
        return

    vqc_model = build_vqc(n_qubits=8, n_layers=3)
    vqc_ckpt = torch.load(vqc_ckpt_path, map_location="cpu")
    vqc_model.load_state_dict(vqc_ckpt["model_state_dict"], strict=True)
    vqc_model.eval()

    angles = [round(float(v) * 180, 1) for v in latent_vals]
    print(f"  * Quantum Angle Embeddings (deg):  {angles}")
    print("  * Simulating 8-Qubit State Vector with Circular CNOT Entanglement...")

    with torch.no_grad():
        z_cpu = z_latent.cpu()
        vqc_logits = vqc_model(z_cpu) # Shape: (1, 2)
        probs = torch.softmax(vqc_logits, dim=1)[0].numpy()

    p_normal = probs[0] * 100
    p_pneumonia = probs[1] * 100
    pred_class = "PNEUMONIA" if p_pneumonia >= 50.0 else "NON_PNEUMONIA / NORMAL"
    confidence = max(p_normal, p_pneumonia)

    print("\n" + "=" * 70)
    print("CLINICAL DIAGNOSIS RESULT")
    print("=" * 70)
    print(f"  * Predicted Class:    [{pred_class}]")
    print(f"  * Model Confidence:   {confidence:.2f}%")
    print(f"  * P(Normal):          {p_normal:.2f}%")
    print(f"  * P(Pneumonia):       {p_pneumonia:.2f}%")
    if ground_truth:
        match_str = "MATCH (CORRECT)" if (ground_truth in pred_class or pred_class in ground_truth) else "DISCREPANCY"
        print(f"  * Evaluation:         {match_str}")
    print("=" * 70 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Live Demo for Hybrid Quantum-Classical Pneumonia Detection")
    parser.add_argument("--image", type=str, default=None, help="Path to chest X-ray image (jpeg/png)")
    parser.add_argument("--label", type=str, default=None, help="Ground truth label (optional)")
    args = parser.parse_args()

    img_p = Path(args.image) if args.image else None
    run_demo(img_p, args.label)

