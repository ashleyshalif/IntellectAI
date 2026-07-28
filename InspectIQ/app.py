"""
app.py
InspectIQ — Streamlit inspection console.
Run with: streamlit run app.py
"""

import os
import sys
import numpy as np
import torch
import cv2
from PIL import Image
import streamlit as st

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from padim_core import FeatureExtractor, TRANSFORM, mahalanobis_score_map, load_stats, IMG_SIZE

MODELS_ROOT = os.path.join(os.path.dirname(__file__), "models")
CATEGORIES = ["bottle", "cable", "tile"]

st.set_page_config(page_title="InspectIQ — Inspection Console", layout="wide", page_icon="🔍")

# ---------- Premium industrial-console styling ----------
st.markdown("""
<style>
    :root {
        --bg-deep: #0b0e14;
        --bg-panel: #131722;
        --border: #232838;
        --accent: #3ddc97;
        --accent-warn: #ff5470;
        --text-dim: #7d8598;
    }
    .stApp { background-color: var(--bg-deep); }
    section[data-testid="stSidebar"] { background-color: var(--bg-panel); border-right: 1px solid var(--border); }
    h1, h2, h3 { font-family: 'Inter', sans-serif; letter-spacing: -0.01em; }
    .main-title { font-size: 1.9rem; font-weight: 700; color: #eef1f8; margin-bottom: 0; }
    .subtitle { color: var(--text-dim); font-size: 0.95rem; margin-top: 2px; letter-spacing: 0.02em; text-transform: uppercase; }
    .metric-card {
        background: var(--bg-panel); border: 1px solid var(--border); border-radius: 10px;
        padding: 18px 20px; text-align: center;
    }
    .metric-value { font-size: 1.6rem; font-weight: 700; }
    .metric-label { color: var(--text-dim); font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.08em; }
    .status-pass { color: var(--accent); }
    .status-fail { color: var(--accent-warn); }
    .stButton>button {
        background: var(--accent); color: #0b0e14; border: none; font-weight: 600; border-radius: 6px;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-title">InspectIQ</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Unsupervised Industrial Anomaly Detection · PaDiM · Cross-Category Robustness</p>', unsafe_allow_html=True)
st.write("")


@st.cache_resource
def get_extractor():
    return FeatureExtractor()


@st.cache_resource
def get_stats(category):
    path = os.path.join(MODELS_ROOT, f"{category}_padim.pkl")
    if not os.path.exists(path):
        return None
    return load_stats(path)


def overlay_heatmap(pil_img: Image.Image, score_map: np.ndarray):
    img = np.array(pil_img.resize((IMG_SIZE, IMG_SIZE))).astype(np.uint8)
    sm = cv2.resize(score_map, (IMG_SIZE, IMG_SIZE))
    sm_norm = (sm - sm.min()) / (sm.max() - sm.min() + 1e-8)
    heat = cv2.applyColorMap((sm_norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    blended = cv2.addWeighted(img, 0.55, heat, 0.45, 0)
    return blended


with st.sidebar:
    st.markdown("### Inspection Settings")
    category = st.selectbox("Product category", CATEGORIES)
    threshold = st.slider("Anomaly threshold", 0.0, 30.0, 12.0, 0.5,
                           help="Max Mahalanobis distance above which a part is flagged defective")
    st.markdown("---")
    st.caption("Model stats loaded from `models/<category>_padim.pkl`. "
               "Run `python src/train.py` first if missing.")

stats = get_stats(category)

if stats is None:
    st.warning(f"No trained model found for **{category}**. "
               f"Run `python src/train.py --categories {category}` first, then reload this page.")
else:
    mean, inv_cov, hw = stats
    uploaded = st.file_uploader("Upload an inspection image", type=["png", "jpg", "jpeg"])

    if uploaded is not None:
        pil_img = Image.open(uploaded).convert("RGB")
        extractor = get_extractor()

        x = TRANSFORM(pil_img).unsqueeze(0)
        with torch.no_grad():
            emb = extractor(x).cpu().numpy()[0]
        score_map = mahalanobis_score_map(emb, mean, inv_cov, hw)
        anomaly_score = float(score_map.max())

        is_defect = anomaly_score >= threshold
        blended = overlay_heatmap(pil_img, score_map)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Original**")
            st.image(pil_img, use_container_width=True)
        with col2:
            st.markdown("**Anomaly Heatmap**")
            st.image(blended, use_container_width=True)

        st.write("")
        m1, m2, m3 = st.columns(3)
        with m1:
            status = "DEFECT DETECTED" if is_defect else "PASS"
            css_class = "status-fail" if is_defect else "status-pass"
            st.markdown(f"""<div class="metric-card"><div class="metric-value {css_class}">{status}</div>
                        <div class="metric-label">Inspection Result</div></div>""", unsafe_allow_html=True)
        with m2:
            st.markdown(f"""<div class="metric-card"><div class="metric-value">{anomaly_score:.2f}</div>
                        <div class="metric-label">Anomaly Score</div></div>""", unsafe_allow_html=True)
        with m3:
            st.markdown(f"""<div class="metric-card"><div class="metric-value">{threshold:.1f}</div>
                        <div class="metric-label">Threshold</div></div>""", unsafe_allow_html=True)
    else:
        st.info("Upload an image to run inspection.")

st.markdown("---")
st.caption("InspectIQ · Asfiya Fatima · Bhilai Institute of Technology, Durg (BIT Durg)")
