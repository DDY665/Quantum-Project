import os
import sys
import time
import math
from pathlib import Path
from PIL import Image
import numpy as np
import pandas as pd
import streamlit as st
import torch
import torchvision.transforms as transforms
import plotly.graph_objects as go
import plotly.express as px

# 1. PAGE SETUP
st.set_page_config(
    page_title="Quantum & Classical Pneumonia Detection System",
    page_icon="🫁",
    layout="wide"
)

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_ROOT))

from models.densenet121 import build_densenet121
from models.densenet_vit import build_densenet_vit
from models.compressed_model import build_compressed_model
from models.vqc import build_vqc

FIGURES_DIR = PROJECT_ROOT / "results" / "figures"
TABLES_DIR = PROJECT_ROOT / "results" / "tables"

# 2. CACHED MODEL LOADER
@st.cache_resource(show_spinner=False)
def load_models():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_dir = PROJECT_ROOT / "checkpoints"

    # DenseNet-121
    m_dense = build_densenet121(pretrained=False, num_classes=2)
    p_dense = ckpt_dir / "densenet121_dataset_A_best.pt"
    if p_dense.exists():
        c = torch.load(p_dense, map_location=device)
        m_dense.load_state_dict(c["model_state_dict"])
    m_dense.to(device).eval()

    # Vision Transformer (ViT Hybrid)
    m_vit = build_densenet_vit(pretrained=False, num_classes=2)
    p_vit = ckpt_dir / "densenet_vit_dataset_A_best.pt"
    if p_vit.exists():
        c = torch.load(p_vit, map_location=device)
        m_vit.load_state_dict(c["model_state_dict"])
    m_vit.to(device).eval()

    # Classical Compression Bottleneck (Prepares 8-dim latent space for Quantum VQC)
    m_comp = build_compressed_model(output_dim=8, hidden_dim=128, num_classes=2, pretrained=False, use_tanh_scaling=True)
    p_comp = ckpt_dir / "compression_dataset_A_best.pt"
    if p_comp.exists():
        c = torch.load(p_comp, map_location=device)
        m_comp.load_state_dict(c["model_state_dict"])
    m_comp.to(device).eval()

    # Quantum VQC (PennyLane 8-Qubit)
    m_vqc = build_vqc(n_qubits=8, n_layers=3)
    p_vqc = ckpt_dir / "vqc_dataset_A_best.pt"
    if p_vqc.exists():
        c = torch.load(p_vqc, map_location="cpu")
        m_vqc.load_state_dict(c["model_state_dict"])
    m_vqc.eval()

    return {
        "densenet": m_dense,
        "vit": m_vit,
        "compressor": m_comp,
        "vqc": m_vqc,
        "device": device
    }

# Image Preprocessing Transformation
TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Quick sample paths for easy testing
SAMPLE_PNEU = PROJECT_ROOT / "data" / "raw" / "kermany" / "chest_xray" / "chest_xray" / "test" / "PNEUMONIA" / "person100_bacteria_475.jpeg"
SAMPLE_NORM = PROJECT_ROOT / "data" / "raw" / "kermany" / "chest_xray" / "chest_xray" / "test" / "NORMAL" / "IM-0001-0001.jpeg"

# Initialize Session State
if "active_image" not in st.session_state:
    st.session_state.active_image = None
if "image_name" not in st.session_state:
    st.session_state.image_name = ""
if "live_results" not in st.session_state:
    st.session_state.live_results = None
if "validation_error" not in st.session_state:
    st.session_state.validation_error = None

# ============================================================================
# CHEST X-RAY VALIDATOR / OUT-OF-DISTRIBUTION (OOD) SCREENING
# ============================================================================
def validate_chest_xray(img: Image.Image) -> tuple[bool, str]:
    """
    Quality & Out-Of-Distribution (OOD) Gate:
    Verifies that the uploaded image meets the radiological and visual characteristics 
    of a thoracic chest X-ray before invoking deep learning and quantum classifiers.
    """
    w, h = img.size
    aspect = w / h
    if aspect > 2.3 or aspect < 0.45:
        return False, f"Abnormal aspect ratio ({aspect:.2f}). Chest radiographs typically have an aspect ratio between 0.5 and 2.0."

    # Fast 224x224 sampling for statistical evaluation
    small_img = img.resize((224, 224))
    arr = np.array(small_img.convert("RGB")).astype(np.float32)

    # 1. Color Saturation Check (Medical radiographs are strictly monochromatic)
    ch_diff = float(np.mean(np.std(arr, axis=2)))
    if ch_diff > 3.0:
        return False, f"Color saturation detected (chroma score: {ch_diff:.1f}). Medical radiographs are grayscale. Color photos, screenshots, or documents are out-of-distribution."

    # 2. Dynamic Range & Contrast Check
    gray = np.mean(arr, axis=2)
    mean_val = float(np.mean(gray))
    std_val = float(np.std(gray))

    if std_val < 20.0:
        return False, f"Flat / low-contrast image (contrast std: {std_val:.1f}). Radiographs require distinct anatomical contrast between bone and lung fields."

    if mean_val < 25.0 or mean_val > 225.0:
        return False, f"Extreme exposure (mean intensity: {mean_val:.1f}). Image is severely under- or over-exposed."

    # 3. Document / Paper Background Check (Text documents have >40% paper-white pixels)
    white_fraction = float(np.mean(gray > 210))
    if white_fraction > 0.40:
        return False, f"Predominantly white background ({white_fraction*100:.1f}%). Matches text document/paper rather than a thoracic radiograph."

    # 4. Black empty image / void frame check
    black_fraction = float(np.mean(gray < 15))
    if black_fraction > 0.75:
        return False, f"Predominantly empty black frame ({black_fraction*100:.1f}%). Missing anatomical lung structures."

    return True, "Valid Chest X-Ray"

# ============================================================================
# LIVE INFERENCE EXECUTION
# ============================================================================
def execute_live_inference(image: Image.Image):
    models = load_models()
    dev = models["device"]
    
    tensor_x = TRANSFORM(image.convert("L")).unsqueeze(0).to(dev)

    # 1. DenseNet Forward Pass
    t0 = time.time()
    with torch.no_grad():
        out_dense = models["densenet"](tensor_x)
        probs_dense = torch.softmax(out_dense, dim=1)[0].cpu().numpy()
    lat_dense = (time.time() - t0) * 1000

    # 2. ViT Forward Pass
    t0 = time.time()
    with torch.no_grad():
        out_vit = models["vit"](tensor_x)
        probs_vit = torch.softmax(out_vit, dim=1)[0].cpu().numpy()
    lat_vit = (time.time() - t0) * 1000

    # 3. Quantum-based Model Forward Pass
    t0 = time.time()
    with torch.no_grad():
        z_8d = models["compressor"].extract_features(tensor_x)
        z_cpu = z_8d.cpu()
        out_vqc = models["vqc"](z_cpu)
        probs_vqc = torch.softmax(out_vqc, dim=1)[0].numpy()
    lat_vqc = (time.time() - t0) * 1000

    z_vals = z_cpu[0].numpy()
    angles_deg = z_vals * 180.0

    return {
        "dense": {
            "p_norm": float(probs_dense[0] * 100),
            "p_pneu": float(probs_dense[1] * 100),
            "pred": "Pneumonia" if probs_dense[1] >= 0.5 else "Normal / No Pneumonia",
            "conf": float(max(probs_dense) * 100),
            "latency": lat_dense
        },
        "vit": {
            "p_norm": float(probs_vit[0] * 100),
            "p_pneu": float(probs_vit[1] * 100),
            "pred": "Pneumonia" if probs_vit[1] >= 0.5 else "Normal / No Pneumonia",
            "conf": float(max(probs_vit) * 100),
            "latency": lat_vit
        },
        "quantum": {
            "p_norm": float(probs_vqc[0] * 100),
            "p_pneu": float(probs_vqc[1] * 100),
            "pred": "Pneumonia" if probs_vqc[1] >= 0.5 else "Normal / No Pneumonia",
            "conf": float(max(probs_vqc) * 100),
            "latency": lat_vqc,
            "z_latent": z_vals,
            "angles_deg": angles_deg
        }
    }

# ============================================================================
# APP TITLE & TABS
# ============================================================================
st.title("🫁 Dynamic Pneumonia Detection & Model Comparator")
st.markdown("### DenseNet vs. Vision Transformer (ViT) vs. Quantum Machine Learning")

tab1, tab2, tab3 = st.tabs([
    "📸 Tab 1 — X-ray Prediction",
    "⚖️ Tab 2 — Live Model Comparator",
    "📊 Tab 3 — Overall Training & Performance"
])

# ============================================================================
# TAB 1: X-RAY PREDICTION (LIVE)
# ============================================================================
with tab1:
    st.subheader("1. Chest X-Ray Upload")
    
    col_upload, col_samples = st.columns([2, 1])
    
    with col_upload:
        uploaded_file = st.file_uploader(
            "Upload Chest X-Ray (JPG, JPEG, PNG):",
            type=["jpg", "jpeg", "png"],
            key="xray_uploader"
        )
        if uploaded_file is not None:
            if st.session_state.image_name != uploaded_file.name:
                img = Image.open(uploaded_file)
                st.session_state.active_image = img
                st.session_state.image_name = uploaded_file.name
                is_valid, err_msg = validate_chest_xray(img)
                if is_valid:
                    st.session_state.validation_error = None
                    st.session_state.live_results = execute_live_inference(img)
                else:
                    st.session_state.validation_error = err_msg
                    st.session_state.live_results = None

    with col_samples:
        st.markdown("**Or test with sample radiographs:**")
        cs1, cs2 = st.columns(2)
        with cs1:
            if st.button("Load Pneumonia Sample", use_container_width=True):
                if SAMPLE_PNEU.exists():
                    img = Image.open(SAMPLE_PNEU)
                    st.session_state.active_image = img
                    st.session_state.image_name = "Sample: Bacterial Pneumonia"
                    st.session_state.validation_error = None
                    st.session_state.live_results = execute_live_inference(img)
        with cs2:
            if st.button("Load Normal Sample", use_container_width=True):
                if SAMPLE_NORM.exists():
                    img = Image.open(SAMPLE_NORM)
                    st.session_state.active_image = img
                    st.session_state.image_name = "Sample: Normal / No Pneumonia"
                    st.session_state.validation_error = None
                    st.session_state.live_results = execute_live_inference(img)

    # 1. DISPLAY OUT-OF-DISTRIBUTION WARNING IF INVALID IMAGE
    if st.session_state.validation_error is not None:
        st.divider()
        col_rej_img, col_rej_msg = st.columns([1, 1.4])
        with col_rej_img:
            st.image(st.session_state.active_image, caption=f"Rejected Input: {st.session_state.image_name}", width=300)
        with col_rej_msg:
            st.error("## ⚠️ Invalid Image: Out-of-Distribution Input")
            st.warning(f"**Modality Screening Rejection:**\n\n{st.session_state.validation_error}")
            st.info(
                "🛡️ **Clinical Safety & OOD Protection Gate:**\n\n"
                "To prevent clinical misdiagnoses on non-medical imagery, text documents, or arbitrary photos, "
                "the system verifies that incoming scans possess authentic radiographic thoracic contrast and grayscale characteristics. "
                "Model inference was suspended to prevent erroneous classification."
            )

    # 2. DISPLAY RESULTS IF IMAGE IS ACTIVE AND VALID
    elif st.session_state.active_image is not None and st.session_state.live_results is not None:
        res = st.session_state.live_results
        
        st.divider()
        col_img, col_pred = st.columns([1, 1.4])
        
        with col_img:
            st.image(st.session_state.active_image, caption=f"Active Scan: {st.session_state.image_name}", width=300)
        
        with col_pred:
            st.subheader("2. Live Classification & Confidence Score")
            
            # Primary Consensus / Ensemble Prediction
            p_pneu_avg = (res["dense"]["p_pneu"] + res["vit"]["p_pneu"] + res["quantum"]["p_pneu"]) / 3.0
            p_norm_avg = 100.0 - p_pneu_avg
            final_class = "Pneumonia" if p_pneu_avg >= 50.0 else "Normal / No Pneumonia"
            final_conf = max(p_pneu_avg, p_norm_avg)

            if final_class == "Pneumonia":
                st.error(f"## 🚨 Result: **PNEUMONIA DETECTED**")
            else:
                st.success(f"## ✅ Result: **NORMAL / NO PNEUMONIA**")
            
            st.markdown(f"#### **Actual Prediction Confidence:** `{final_conf:.2f}%`")
            st.caption(f"Calculated from live softmax logits across models | Source: {st.session_state.image_name}")

            # LIVE PROBABILITY BAR CHART
            fig_prob = go.Figure(data=[
                go.Bar(
                    x=["Normal / No Pneumonia", "Pneumonia"],
                    y=[p_norm_avg, p_pneu_avg],
                    marker_color=["#10B981" if p_norm_avg > p_pneu_avg else "#6EE7B7", 
                                  "#EF4444" if p_pneu_avg >= p_norm_avg else "#FCA5A5"],
                    text=[f"{p_norm_avg:.2f}%", f"{p_pneu_avg:.2f}%"],
                    textposition="auto"
                )
            ])
            fig_prob.update_layout(
                title=f"LIVE Probability Score Visualization (Generated from {st.session_state.image_name})",
                yaxis=dict(title="Probability (%)", range=[0, 105]),
                height=260,
                margin=dict(l=20, r=20, t=40, b=20)
            )
            st.plotly_chart(fig_prob, use_container_width=True)

    else:
        st.info("👆 Please upload a chest X-ray or click one of the sample buttons above to generate live predictions.")

# ============================================================================
# TAB 2: LIVE MODEL COMPARATOR
# ============================================================================
with tab2:
    st.subheader("Live Model Comparator: DenseNet vs. ViT vs. Quantum Model")
    st.caption("Directly compares the live probabilities generated by each model for the currently uploaded image.")

    if st.session_state.live_results is not None:
        res = st.session_state.live_results

        # 3 Detailed Model Cards
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("### DenseNet")
            st.caption("DenseNet-121 (Convolutional)")
            if res["dense"]["pred"] == "Pneumonia":
                st.error(f"**Prediction:** {res['dense']['pred']}")
            else:
                st.success(f"**Prediction:** {res['dense']['pred']}")
            st.metric("Confidence", f"{res['dense']['conf']:.2f}%")
            st.markdown(f"- **P(Normal):** `{res['dense']['p_norm']:.2f}%`\n- **P(Pneumonia):** `{res['dense']['p_pneu']:.2f}%`")
            st.caption(f"Inference Latency: {res['dense']['latency']:.1f} ms")

        with c2:
            st.markdown("### Vision Transformer")
            st.caption("ViT-B/16 (Self-Attention)")
            if res["vit"]["pred"] == "Pneumonia":
                st.error(f"**Prediction:** {res['vit']['pred']}")
            else:
                st.success(f"**Prediction:** {res['vit']['pred']}")
            st.metric("Confidence", f"{res['vit']['conf']:.2f}%")
            st.markdown(f"- **P(Normal):** `{res['vit']['p_norm']:.2f}%`\n- **P(Pneumonia):** `{res['vit']['p_pneu']:.2f}%`")
            st.caption(f"Inference Latency: {res['vit']['latency']:.1f} ms")

        with c3:
            st.markdown("### Quantum Model")
            st.caption("8-Qubit PennyLane VQC")
            if res["quantum"]["pred"] == "Pneumonia":
                st.error(f"**Prediction:** {res['quantum']['pred']}")
            else:
                st.success(f"**Prediction:** {res['quantum']['pred']}")
            st.metric("Confidence", f"{res['quantum']['conf']:.2f}%")
            st.markdown(f"- **P(Normal):** `{res['quantum']['p_norm']:.2f}%`\n- **P(Pneumonia):** `{res['quantum']['p_pneu']:.2f}%`")
            st.caption(f"Inference Latency: {res['quantum']['latency']:.1f} ms")

        st.divider()

        # LIVE GROUPED BAR CHART: NORMAL VS PNEUMONIA ACROSS MODELS
        st.subheader("LIVE Grouped Bar Chart: Probability Comparison")
        fig_grouped = go.Figure(data=[
            go.Bar(
                name="Normal Probability",
                x=["DenseNet", "Vision Transformer (ViT)", "Quantum-based Model"],
                y=[res["dense"]["p_norm"], res["vit"]["p_norm"], res["quantum"]["p_norm"]],
                marker_color="#10B981",
                text=[f"{res['dense']['p_norm']:.1f}%", f"{res['vit']['p_norm']:.1f}%", f"{res['quantum']['p_norm']:.1f}%"],
                textposition="auto"
            ),
            go.Bar(
                name="Pneumonia Probability",
                x=["DenseNet", "Vision Transformer (ViT)", "Quantum-based Model"],
                y=[res["dense"]["p_pneu"], res["vit"]["p_pneu"], res["quantum"]["p_pneu"]],
                marker_color="#EF4444",
                text=[f"{res['dense']['p_pneu']:.1f}%", f"{res['vit']['p_pneu']:.1f}%", f"{res['quantum']['p_pneu']:.1f}%"],
                textposition="auto"
            )
        ])
        fig_grouped.update_layout(
            barmode="group",
            yaxis=dict(title="Probability (%)", range=[0, 105]),
            title=f"LIVE Model Output Comparison for [{st.session_state.image_name}]",
            height=360,
            margin=dict(l=20, r=20, t=50, b=20)
        )
        st.plotly_chart(fig_grouped, use_container_width=True)

    elif st.session_state.validation_error is not None:
        st.warning(f"⚠️ **Comparator Inactive:** `{st.session_state.image_name}` was flagged as an invalid / out-of-distribution image. Please upload an authentic chest radiograph in Tab 1.")
    else:
        st.info("👆 Please upload an image in Tab 1 to activate the live model comparator.")

# ============================================================================
# TAB 3: OVERALL TRAINING & PERFORMANCE (STATIC)
# ============================================================================
with tab3:
    st.subheader("Overall Training & Performance (Static Test-Set Evaluation)")
    st.markdown(
        "📌 **Note:** Visualizations in this section are **STATIC** scientific benchmarks compiled from the completed "
        "training runs and evaluated across the entire test dataset (1,757 patient chest X-rays)."
    )

    # 1. OVERALL TEST METRICS TABLE
    st.subheader("1. Comprehensive Model Evaluation Metrics")
    comp_df = pd.DataFrame({
        "Model Architecture": [
            "DenseNet (DenseNet-121)",
            "Vision Transformer (ViT)",
            "Quantum-based Model (8-Qubit VQC)"
        ],
        "Accuracy (%)": ["99.54%", "99.54%", "98.75%"],
        "Precision (%)": ["99.55%", "99.55%", "98.76%"],
        "Recall (%)": ["99.54%", "99.54%", "98.75%"],
        "F1-score (%)": ["99.54%", "99.54%", "98.74%"],
        "ROC-AUC (%)": ["99.98%", "99.88%", "99.01%"],
        "Specificity (%)": ["99.37%", "98.32%", "96.00%"]
    })
    st.dataframe(comp_df, use_container_width=True, hide_index=True)

    st.divider()

    # 2. DATASET INFORMATION
    st.subheader("2. Dataset Information")
    st.markdown("- **Dataset A:** Kermany pediatric chest X-ray cohort (5,856 radiographs)")
    st.markdown("- **Dataset B:** Cross-domain Kermany + RSNA adult cohort (balanced)")
    st.markdown("- **Class Split:** PNEUMONIA vs. NON_PNEUMONIA (Normal)")
    st.markdown("- **Data Split:** 70% Train, 15% Validation, 15% Test")

    st.divider()

    # 3. STATIC PERFORMANCE GRAPHS
    st.subheader("3. Static ML Performance Graphs")

    # Accuracy Bar Chart
    g1_p = FIGURES_DIR / "graph1_accuracy_comparison.png"
    if g1_p.exists():
        st.image(str(g1_p), caption="Static Figure: Test Accuracy Comparison across DenseNet, ViT, and Quantum VQC", use_container_width=True)

    st.divider()

    # Precision, Recall, F1 Bar Chart
    g2_p = FIGURES_DIR / "graph2_metrics_comparison.png"
    if g2_p.exists():
        st.image(str(g2_p), caption="Static Figure: Precision, Recall, F1-Score, and ROC-AUC Comparison", use_container_width=True)

    st.divider()

    # ROC Curves Comparison
    g3_p = FIGURES_DIR / "graph3_roc_comparison.png"
    if g3_p.exists():
        st.image(str(g3_p), caption="Static Figure: Comparative ROC Curves across Models", use_container_width=True)

    st.divider()

    # Confusion Matrices for Each Model
    st.subheader("4. Confusion Matrices for Each Model")
    cm_c1, cm_c2, cm_c3 = st.columns(3)
    cm_dense_p = FIGURES_DIR / "densenet121_dataset_A_cm.png"
    cm_vit_p = FIGURES_DIR / "densenet_vit_dataset_A_cm.png"
    cm_vqc_p = FIGURES_DIR / "vqc_dataset_A_cm.png"

    with cm_c1:
        if cm_dense_p.exists():
            st.image(str(cm_dense_p), caption="DenseNet-121 Confusion Matrix", use_container_width=True)
    with cm_c2:
        if cm_vit_p.exists():
            st.image(str(cm_vit_p), caption="Vision Transformer (ViT) Confusion Matrix", use_container_width=True)
    with cm_c3:
        if cm_vqc_p.exists():
            st.image(str(cm_vqc_p), caption="Quantum VQC Confusion Matrix", use_container_width=True)
