"""
app.py
IntellectAI — Streamlit inspection console.
Run with: streamlit run app.py
"""

import os
import sys
import numpy as np
import torch
import cv2
from PIL import Image
import streamlit as st
import csv
from datetime import datetime
from pathlib import Path

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from padim_core import FeatureExtractor, TRANSFORM, mahalanobis_score_map, load_stats, apply_pca, IMG_SIZE

MODELS_ROOT = os.path.join(os.path.dirname(__file__), "models")
CATEGORIES = ["bottle", "cable", "tile"]

st.set_page_config(page_title="IntellectAI — Inspection Console", layout="wide", page_icon="🔍")

# ---------- Premium industrial-console styling ----------
st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
<style>
    :root {
        --bg-deep: #03060a;
        --bg-panel: #0a1017;
        --grid-line: rgba(90, 148, 255, 0.05);
        --border: #1c3a4a;
        --accent: #4fc3f7;
        --accent-dim: #2e5b6b;
        --accent-warn: #ff5470;
        --accent-mid: #ffb84d;
        --text-dim: #8fb8cc;
    }
    .stApp {
        background-color: var(--bg-deep);
        background-image:
            linear-gradient(var(--grid-line) 1px, transparent 1px),
            linear-gradient(90deg, var(--grid-line) 1px, transparent 1px);
        background-size: 24px 24px;
        animation: gridDrift 50s linear infinite;
    }
    @keyframes gridDrift {
        from { background-position: 0 0, 0 0; }
        to { background-position: 240px 240px, 240px 240px; }
    }
    section[data-testid="stSidebar"] { background-color: var(--bg-panel); border-right: 1px solid var(--border); }
    h1, h2, h3 { font-family: 'JetBrains Mono', monospace; letter-spacing: -0.01em; }
    .main-title, p.main-title {
        font-family: 'Space Mono', monospace !important; font-weight: 800 !important;
        font-size: clamp(3.5rem, 7vw, 8rem) !important; margin-bottom: 0; letter-spacing: -0.02em;
        position: relative; display: inline-block; line-height: 1.02 !important;
        background: linear-gradient(135deg, #eafff7 0%, var(--accent) 55%, #8fe3ff 100%);
        -webkit-background-clip: text; background-clip: text;
        color: transparent !important; -webkit-text-fill-color: transparent !important;
        filter: drop-shadow(0 0 26px rgba(79,195,247,0.35));
        animation: titleIn 0.7s cubic-bezier(0.22, 1, 0.36, 1) both;
    }
    .main-title::before { content: "> "; -webkit-text-fill-color: var(--accent); color: var(--accent); }
    .main-title::after {
        content: ""; display: inline-block; width: clamp(14px, 1.4vw, 24px); height: clamp(40px, 6vw, 64px);
        background: var(--accent); margin-left: 14px; vertical-align: -0.16em;
        animation: blink 1s steps(1) infinite;
        box-shadow: 0 0 14px rgba(79,195,247,0.6);
    }
    @keyframes blink { 50% { opacity: 0; } }
    @keyframes titleIn { from { opacity: 0; transform: translateY(16px); } to { opacity: 1; transform: translateY(0); } }

    .status-tag {
        display: inline-flex; align-items: center; gap: 8px;
        border: 1px solid var(--border); padding: 5px 12px; border-radius: 4px;
        font-family: 'JetBrains Mono', monospace; font-size: 0.7rem;
        color: var(--accent); letter-spacing: 0.08em; margin-bottom: 16px;
        transition: border-color 0.25s ease, box-shadow 0.25s ease;
        animation: titleIn 0.7s cubic-bezier(0.22, 1, 0.36, 1) both;
    }
    .status-tag:hover { border-color: var(--accent); box-shadow: 0 0 16px -4px rgba(79,195,247,0.5); }
    .status-tag::before {
        content: ""; width: 6px; height: 6px; border-radius: 50%; background: var(--accent);
        animation: blink 1.4s steps(1) infinite;
    }
    .subtitle {
        font-family: 'Space Mono', monospace; color: var(--text-dim);
        font-size: 1.15rem; margin-top: 18px; margin-bottom: 4px; letter-spacing: 0.05em;
        padding-left: 16px; border-left: 2px solid var(--accent-dim);
        transition: border-color 0.3s ease, padding-left 0.3s ease;
        animation: titleIn 0.7s cubic-bezier(0.22, 1, 0.36, 1) both;
        animation-delay: 0.12s;
    }
    .subtitle:hover { border-left-color: var(--accent); padding-left: 20px; }
    .metric-card {
        background: var(--bg-panel); border: 1px dashed var(--border); border-radius: 8px;
        padding: 20px 22px; margin-bottom: 12px; text-align: left; position: relative;
        transition: transform 0.32s cubic-bezier(0.22, 1, 0.36, 1),
                    border-color 0.32s cubic-bezier(0.22, 1, 0.36, 1),
                    box-shadow 0.32s cubic-bezier(0.22, 1, 0.36, 1);
    }
    .metric-card:hover {
        transform: translateY(-4px); border-color: var(--accent); border-style: solid;
        box-shadow: 0 14px 32px -10px rgba(79,195,247,0.4);
    }
    /* ---- Entrance animation: CSS-only, always resolves to visible, no JS dependency ---- */
    @keyframes fadeSlideIn {
        from { opacity: 0; transform: translateY(18px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .metric-card, div[data-testid="stExpander"] {
        animation: fadeSlideIn 0.6s cubic-bezier(0.22, 1, 0.36, 1) both;
    }
    .metric-index {
        position: absolute; top: 8px; right: 12px;
        font-family: 'JetBrains Mono', monospace; font-size: 0.62rem; color: var(--accent-dim);
    }
    .metric-value { font-family: 'Space Mono', monospace; font-size: 2.1rem; font-weight: 700; }
    .metric-label { font-family: 'JetBrains Mono', monospace; color: var(--text-dim); font-size: 0.75rem; text-transform: lowercase; letter-spacing: 0.05em; margin-top: 6px; }
    /* ---- Metrics column: sized so the three cards fill roughly the same
       vertical space as the image columns beside them, spaced evenly ---- */
    .metrics-stack {
        display: flex; flex-direction: column; justify-content: space-between;
        height: 100%; gap: 20px;
    }
    .metrics-stack .metric-card {
        flex: 1 1 0; display: flex; flex-direction: column; justify-content: center;
        margin-bottom: 0; min-height: 170px;
    }
    .status-pass { color: var(--accent); }
    .status-warn { color: var(--accent-mid); }
    .status-fail { color: var(--accent-warn); }
    div[data-testid="stImage"] {
        border: 1px solid var(--border); border-radius: 6px; overflow: hidden;
        transition: border-color 0.32s ease, box-shadow 0.32s ease;
        animation: fadeSlideIn 0.6s cubic-bezier(0.22, 1, 0.36, 1) both;
    }
    div[data-testid="stImage"]:hover {
        border-color: var(--accent);
        box-shadow: 0 10px 30px -10px rgba(79,195,247,0.45);
    }
    div[data-testid="stImage"] img { width: 100% !important; height: auto !important; }
    .stButton>button {
        background: transparent; color: var(--accent); border: 1px solid var(--accent);
        font-family: 'JetBrains Mono', monospace; font-weight: 600; border-radius: 4px;
        transition: transform 0.18s ease, box-shadow 0.18s ease, background 0.18s ease;
    }
    .stButton>button:hover {
        transform: translateY(-1px); background: rgba(79,195,247,0.08);
        box-shadow: 0 6px 20px -6px rgba(79,195,247,0.45);
    }
    .stButton>button::before { content: "[ "; }
    .stButton>button::after { content: " ]"; }
    .stCaption, [data-testid="stCaptionContainer"] { font-family: 'JetBrains Mono', monospace; }

    /* ---- Sidebar theming ---- */
    section[data-testid="stSidebar"] h3 {
        font-family: 'JetBrains Mono', monospace; color: var(--accent);
        font-size: 1.15rem; text-transform: uppercase; letter-spacing: 0.08em;
        border-bottom: 1px dashed var(--border); padding-bottom: 10px; margin-bottom: 18px;
    }
    .sidebar-subhead {
        font-family: 'JetBrains Mono', monospace; color: var(--accent-dim);
        font-size: 0.68rem; text-transform: uppercase; letter-spacing: 0.12em;
        border-top: 1px dashed var(--border); padding-top: 16px; margin-top: 20px; margin-bottom: 8px;
    }
    section[data-testid="stSidebar"] label, section[data-testid="stSidebar"] p {
        font-family: 'JetBrains Mono', monospace; color: var(--text-dim) !important; font-size: 0.82rem;
    }
    section[data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] {
        background: var(--bg-deep); border: 1px solid var(--border); border-radius: 4px;
        transition: border-color 0.18s ease;
    }
    section[data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"]:hover {
        border-color: var(--accent);
    }
    section[data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] * {
        font-family: 'JetBrains Mono', monospace; color: var(--accent) !important;
    }

    /* ---- Sliders ---- */
    div[data-testid="stSlider"] [role="slider"] { background-color: var(--accent) !important; border-color: var(--accent) !important; }
    div[data-testid="stSlider"] > div > div > div { background-color: var(--accent) !important; }
    div[data-testid="stTickBar"], div[data-testid="stSliderTickBarMin"], div[data-testid="stSliderTickBarMax"] {
        font-family: 'JetBrains Mono', monospace; color: var(--text-dim) !important;
    }

    /* ---- Expanders ---- */
    div[data-testid="stExpander"] {
        border: 1px solid var(--border); border-radius: 6px; background: var(--bg-panel);
        transition: border-color 0.22s ease;
    }
    div[data-testid="stExpander"]:hover {
        border-color: var(--accent);
    }
    div[data-testid="stExpander"] summary {
        font-family: 'JetBrains Mono', monospace; color: var(--accent) !important;
    }

    /* ---- Alerts (warning / info / error / success) ---- */
    div[data-testid="stAlert"] {
        border-radius: 4px; background: var(--bg-panel) !important;
        border: 1px solid var(--border);
    }
    div[data-testid="stAlert"] p, div[data-testid="stAlert"] * {
        font-family: 'JetBrains Mono', monospace; color: var(--text-dim) !important;
    }

    /* ---- Dataframe / misc ---- */
    div[data-testid="stDataFrame"] { border: 1px solid var(--border); border-radius: 6px; overflow: hidden; }
    hr { border-color: var(--border) !important; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="status-tag">INTELLECTAI SYS v2.1</div>', unsafe_allow_html=True)
st.markdown('<p class="main-title">IntellectAI</p>', unsafe_allow_html=True)
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


def overlay_heatmap(pil_img: Image.Image, score_map: np.ndarray, display_size: int = 600):
    img = np.array(pil_img.resize((IMG_SIZE, IMG_SIZE))).astype(np.uint8)
    sm = cv2.resize(score_map, (IMG_SIZE, IMG_SIZE))
    sm_norm = (sm - sm.min()) / (sm.max() - sm.min() + 1e-8)
    heat = cv2.applyColorMap((sm_norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    blended = cv2.addWeighted(img, 0.55, heat, 0.45, 0)
    # Streamlit's use_container_width won't upscale an image past its native
    # pixel size, and the blend above is only IMG_SIZE x IMG_SIZE (small) --
    # so upscale it here purely for display, after the anomaly score/shape
    # metrics have already been computed from the original-resolution score_map.
    blended = cv2.resize(blended, (display_size, display_size), interpolation=cv2.INTER_CUBIC)
    return blended


def classify_anomaly_shape(category: str, score_map: np.ndarray, flag_threshold: float):
    """
    Heuristic-only defect-type suggestion based on the SHAPE of the anomalous
    region in the score map, scoped per category to the defect vocabulary that
    actually shows up in each category's test set. Returns (short_label,
    description): short_label is a compact tag for the status badge,
    description is the fuller explanation. PaDiM never sees labeled defect
    types during training, so this is not a learned classification -- it's a
    geometric rule of thumb layered on top of the anomaly map, and should be
    read as a suggestion rather than a diagnosis.
    """
    mask = (score_map >= flag_threshold).astype(np.uint8)
    if mask.sum() == 0:
        return "IRREGULARITY", "Region too faint to characterize"

    mask_resized = cv2.resize(mask, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)
    contours, _ = cv2.findContours(mask_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return "IRREGULARITY", "Region too faint to characterize"

    largest = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest)
    if area < 4:
        return "IRREGULARITY", "Small isolated region"

    x, y, w, h = cv2.boundingRect(largest)
    aspect_ratio = max(w, h) / max(1, min(w, h))
    rect_area = w * h
    extent = area / max(1, rect_area)  # how much of the bounding box the region fills
    coverage = mask_resized.sum() / mask_resized.size  # fraction of whole image flagged
    n_contours = len(contours)

    # Solidity: area / convex-hull area. Thin or branching regions (cracks) have a
    # convex hull much bigger than their actual area -> LOW solidity. Smooth compact
    # blobs (stains, contamination, residue) fill their own convex hull -> HIGH solidity.
    # This catches branching/Y-shaped cracks that aspect_ratio alone misses (a Y-crack's
    # bounding box is roughly square, not elongated), and stops compact stains from
    # being mislabeled just because noise fragments them into several small contours.
    hull = cv2.convexHull(largest)
    hull_area = cv2.contourArea(hull)
    solidity = area / max(1.0, hull_area)

    # Stash raw metrics on the function object so the caller can display them
    # for calibration -- this is what lets us set real thresholds instead of
    # guessing blind.
    classify_anomaly_shape.last_metrics = {
        "aspect_ratio": round(aspect_ratio, 3),
        "extent": round(extent, 3),
        "coverage": round(coverage, 3),
        "solidity": round(solidity, 3),
        "n_contours": n_contours,
        "area": round(area, 1),
    }

    if category == "bottle":
        # Crack/break check runs FIRST, using solidity as the primary signal
        # (irregular fracture edges -> low solidity) rather than requiring a
        # strict aspect_ratio -- a break doesn't have to be a thin elongated
        # line to be a break. This fixes real breaks being swallowed by the
        # contamination check below. Per project decision: crack/break size
        # (small vs large) is no longer split into separate labels -- both
        # report as "CRACK", with size kept only as a descriptive detail.
        is_break_shaped = (solidity < 0.55) or (aspect_ratio >= 3.5 and solidity < 0.65)
        if is_break_shaped:
            size_note = "large-area" if coverage >= 0.08 else "small"
            return "BREAK", f"Possible break in the bottle ({size_note}, irregular region)"
        elif solidity >= 0.65 and extent >= 0.55:
            # Genuinely smooth AND fills its own bounding box -- real contamination.
            # Both conditions required now (not solidity alone) so a break with
            # moderate solidity but jagged, non-filling extent isn't caught here.
            if coverage >= 0.08:
                return "CONTAMINATION", "Possible contamination or spillage (large smooth stain)"
            else:
                return "CONTAMINATION", "Possible contamination in the bottle (smooth compact residue)"
        else:
            # Ambiguous middle ground: neither clearly break-shaped nor clearly
            # smooth/filled. Default to BREAK rather than CONTAMINATION -- per
            # project decision, missing a real break is worse than an extra
            # break flag on borderline contamination.
            return "BREAK", "Possible break or irregularity in the bottle (uncertain shape)"

    if category == "tile":
        # Crack check widened: the old solidity<0.4 AND extent<0.35 was too
        # strict and missed real diagonal cracks whose bounding box is nearly
        # square. Now also catches high-aspect-ratio thin regions directly,
        # and loosens the solidity/extent floor slightly.
        is_crack_shaped = (solidity < 0.5 and extent < 0.45) or (aspect_ratio >= 3.0 and solidity < 0.55)
        if is_crack_shaped:
            return "CRACK", "Possible crack (elongated or branching pattern)"
        elif solidity >= 0.75 and coverage < 0.08:
            # Contamination now requires a much higher solidity floor (0.75,
            # was 0.6) so it only fires on genuinely round/compact stains --
            # this was previously catching real cracks AND oil marks that
            # happened to have moderate solidity.
            return "CONTAMINATION", "Possible contamination or spillage (smooth compact stain)"
        elif extent >= 0.5 and coverage < 0.12 and solidity >= 0.6:
            return "GLUE STRIP", "Possible glue strip (contained smear)"
        elif n_contours >= 4 and solidity < 0.5 and coverage < 0.15:
            return "GRAY STROKE", "Possible gray-stroke mark (scattered faint streaks)"
        elif coverage >= 0.15:
            return "ROUGH SURFACE", "Possible rough or patchy surface texture (widespread diffuse pattern)"
        elif coverage >= 0.06:
            # Oil stain threshold lowered slightly (0.08 -> 0.06) so more
            # diffuse, less-solid marks land here instead of being caught by
            # the now-stricter contamination check above.
            return "OIL STAIN", "Possible oil stain or discoloration (diffuse pattern)"
        else:
            return "SURFACE MARK", "Surface irregularity (uneven pattern)"

    # cable: defects here are structural (cuts/missing/swapped parts) rather
    # than textural, so keep this closer to the original generic rule.
    if aspect_ratio >= 3.5:
        return "CUT OR DEFORMATION", "Possible cut, scratch, or wire deformation (elongated pattern)"
    elif extent >= 0.55:
        return "MISSING/SWAPPED PART", "Possible missing or swapped component (compact region)"
    else:
        return "STRUCTURAL IRREGULARITY", "Structural irregularity (diffuse pattern)"


CALIBRATED_THRESHOLDS = {
    "bottle": {"minor": 16.54, "defect": 18.09},
    "cable":  {"minor": 15.32, "defect": 15.33},
    "tile":   {"minor": 13.39, "defect": 13.41},
}

FEEDBACK_LOG_PATH = Path("feedback_log.csv")

def log_feedback(file_key, image_name, category, result, defect_type, feedback, reason=""):
    """Append one feedback row. Creates the file with a header on first write."""
    is_new = not FEEDBACK_LOG_PATH.exists()
    with open(FEEDBACK_LOG_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if is_new:
            writer.writerow(["timestamp", "file_key", "image_name", "category",
                              "result", "defect_type", "feedback", "reason"])
        writer.writerow([datetime.now().isoformat(timespec="seconds"), file_key, image_name,
                          category, result, defect_type, feedback, reason])

def get_agreement_rate(category=None):
    """Returns (agree_count, total_count) from the feedback log, or None if no data yet."""
    if not FEEDBACK_LOG_PATH.exists():
        return None
    agree, total = 0, 0
    with open(FEEDBACK_LOG_PATH, "r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if category and row.get("category") != category:
                continue
            total += 1
            if row.get("feedback") == "agree":
                agree += 1
    return (agree, total) if total else None

with st.container(border=True):
    st.markdown('<div class="sidebar-subhead">Inspection settings</div>', unsafe_allow_html=True)
    ctrl1, ctrl2, ctrl3 = st.columns(3)
    with ctrl1:
        category = st.selectbox("Product category", CATEGORIES)

    if "last_category" not in st.session_state or st.session_state.last_category != category:
        cal = CALIBRATED_THRESHOLDS.get(category, {"minor": 7.0, "defect": 12.0})
        st.session_state.threshold_slider = cal["defect"]
        st.session_state.minor_threshold_slider = cal["minor"]
        st.session_state.last_category = category

    with ctrl2:
        threshold = st.slider("Anomaly threshold", 0.0, 30.0, 0.5,
                               key="threshold_slider",
                               help="Max Mahalanobis distance above which a part is flagged defective")
    with ctrl3:
        minor_threshold = st.slider("Minor irregularity threshold", 0.0, 30.0, 0.5,
                               key="minor_threshold_slider",
                               help="Scores above this but below the defect threshold are flagged as minor irregularities (e.g. glue strips, oil marks) instead of full defects")

    st.caption(f"Calibrated from real test-set scores for **{category}** "
               f"(minor={CALIBRATED_THRESHOLDS.get(category, {}).get('minor', 'n/a')}, "
               f"defect={CALIBRATED_THRESHOLDS.get(category, {}).get('defect', 'n/a')}). "
               f"Adjust manually if needed.")

    st.markdown('<div class="sidebar-subhead">Diagnostics</div>', unsafe_allow_html=True)
    with st.expander("Model Performance"):
        st.caption("Offline benchmark from evaluate.py against real MVTec test-set "
                    "images (top1pct scorer). Not a live per-image confidence score.")
        BENCHMARK = {
            "bottle": {"broken_large": "20/20", "broken_small": "22/22", "contamination": "19/21"},
            "cable": {"bent_wire": "5/13", "cable_swap": "0/12", "combined": "3/11",
                      "cut_inner_insulation": "3/14", "cut_outer_insulation": "5/10",
                      "missing_cable": "0/12", "missing_wire": "1/10", "poke_insulation": "1/10"},
            "tile": {"crack": "16/17", "glue_strip": "18/18", "gray_stroke": "3/16",
                     "oil": "18/18", "rough": "11/15"},
        }
        for defect, ratio in BENCHMARK.get(category, {}).items():
            st.write(f"**{defect}**: {ratio} detected in test set")

        st.markdown("---")
        rate = get_agreement_rate(category)
        if rate:
            agree, total = rate
            st.metric(f"User agreement ({category})", f"{agree}/{total}", f"{100*agree/total:.0f}%")
        else:
            st.caption("No user feedback logged yet for this category.")

stats = get_stats(category)

if stats is None:
    st.warning(f"No trained model found for **{category}**. "
               f"Run `python src/train.py --categories {category}` first, then reload this page.")
else:
    mean, inv_cov, hw, pca_mean, pca_components, raw_centroid = stats
    uploaded_files = st.file_uploader("Upload inspection images", type=["png", "jpg", "jpeg"],
                                       accept_multiple_files=True)

    if "history" not in st.session_state:
        st.session_state.history = []
    if "processed_keys" not in st.session_state:
        st.session_state.processed_keys = set()
    if "feature_cache" not in st.session_state:
        st.session_state.feature_cache = {}

    if uploaded_files:
        extractor = get_extractor()

        results = []
        for uploaded in uploaded_files:
            file_key = f"{category}_{uploaded.name}_{uploaded.size}"

            cached = st.session_state.feature_cache.get(file_key)
            if cached is not None:
                # Model forward pass, cross-category comparison, and heatmap
                # render already done for this file+category -- reuse it.
                # Only severity/status below depends on the sliders, so that
                # part still runs fresh every time.
                pil_img = cached["pil_img"]
                score_map = cached["score_map"]
                anomaly_score = cached["anomaly_score"]
                blended = cached["blended"]
                category_mismatch = cached["category_mismatch"]
                best_category = cached["best_category"]
                likely_unrecognized = cached["likely_unrecognized"]
                max_similarity = cached["max_similarity"]
            else:
                try:
                    pil_img = Image.open(uploaded).convert("RGB")
                except Exception:
                    st.error(f"**{uploaded.name}** could not be read as an image "
                             f"(corrupt file or wrong format despite the extension). Skipped.")
                    continue

                x = TRANSFORM(pil_img).unsqueeze(0)
                with torch.no_grad():
                    raw_emb = extractor(x).cpu().numpy()[0]  # full 448-dim, before PCA
                emb = apply_pca(raw_emb, pca_mean, pca_components)  # this category's PCA space, for scoring
                score_map = mahalanobis_score_map(emb, mean, inv_cov, hw)
                # Top-1% mean instead of a single max patch: more robust to diffuse
                # anomalies while still far more sensitive than a full-map mean.
                _flat = score_map.flatten()
                _k = max(1, int(_flat.size * 0.01))
                anomaly_score = float(np.partition(_flat, -_k)[-_k:].mean())

                # Cross-category sanity check: compare this image's own averaged
                # feature vector against each category's "typical feature"
                # centroid (the per-location training mean, averaged over all
                # spatial locations). This is a pure visual-similarity check --
                # it answers "what product does this look like" independent of
                # whether it's defective, unlike anomaly/defect scores which
                # conflate "wrong category" with "very defective correct
                # category" (a badly damaged bottle can out-score a normal image
                # of the wrong category, which broke the old ratio-based check).
                # Full 448-dim space: each category now has its own PCA basis,
                # so comparing inside any single category's reduced space is invalid.
                img_centroid = raw_emb.mean(axis=(1, 2))  # (448,)
                category_similarities = {}
                for other_cat in CATEGORIES:
                    other_stats = get_stats(other_cat)
                    if other_stats is None:
                        continue
                    _, _, _, _, _, o_raw_centroid = other_stats
                    cos_sim = float(
                        np.dot(img_centroid, o_raw_centroid)
                        / (np.linalg.norm(img_centroid) * np.linalg.norm(o_raw_centroid) + 1e-8)
                    )
                    category_similarities[other_cat] = cos_sim

                best_category = max(category_similarities, key=category_similarities.get) if category_similarities else category
                category_mismatch = best_category != category
                # Even the best-matching category's similarity can be checked: if it's
                # low across the board, this probably isn't a bottle/cable/tile at all
                # (PaDiM has no built-in "unknown object" concept -- it will still
                # produce a score and a category guess for anything). 0.5 is a starting
                # heuristic cutoff, not a calibrated value.
                OOD_SIMILARITY_THRESHOLD = 0.5
                max_similarity = max(category_similarities.values()) if category_similarities else 1.0
                likely_unrecognized = max_similarity < OOD_SIMILARITY_THRESHOLD

                blended = overlay_heatmap(pil_img, score_map)

                st.session_state.feature_cache[file_key] = {
                    "pil_img": pil_img,
                    "score_map": score_map,
                    "anomaly_score": anomaly_score,
                    "blended": blended,
                    "category_mismatch": category_mismatch,
                    "best_category": best_category,
                    "likely_unrecognized": likely_unrecognized,
                    "max_similarity": max_similarity,
                }

            if anomaly_score >= threshold:
                severity = "defect"
            elif anomaly_score >= minor_threshold:
                severity = "minor"
            else:
                severity = "normal"

            status_map = {"defect": ("DEFECT DETECTED", "status-fail"),
                          "minor": ("MINOR IRREGULARITY", "status-warn"),
                          "normal": ("PASS", "status-pass")}
            status, css_class = status_map[severity]

            shape_metrics = getattr(classify_anomaly_shape, "last_metrics", None)
            defect_label = ""
            defect_desc = ""
            if severity != "normal":
                defect_label, defect_desc = classify_anomaly_shape(category, score_map, minor_threshold)
                verb = "DETECTED" if severity == "defect" else "SUSPECTED"
                status = f"{defect_label} {verb}"

            if file_key not in st.session_state.processed_keys:
                st.session_state.processed_keys.add(file_key)
                st.session_state.history.append({
                    "File": uploaded.name,
                    "Category": category,
                    "Score": round(anomaly_score, 2),
                    "Result": status,
                    "Likely Type": defect_desc,
                })

            results.append({
                "name": uploaded.name,
                "pil_img": pil_img,
                "blended": blended,
                "status": status,
                "css_class": css_class,
                "anomaly_score": anomaly_score,
                "threshold": threshold,
                "defect_type": defect_desc,
                "category_mismatch": category_mismatch,
                "best_category": best_category,
                "likely_unrecognized": likely_unrecognized,
                "max_similarity": max_similarity,
                "shape_metrics": shape_metrics,
                "file_key": file_key,
            })

        for i, r in enumerate(reversed(results)):
            with st.expander(f"{r['name']}  —  {r['status']}", expanded=(i == 0)):
                if r.get("shape_metrics"):
                    sm = r["shape_metrics"]
                    st.caption(f"Debug shape metrics -- aspect_ratio={sm['aspect_ratio']}, "
                               f"extent={sm['extent']}, coverage={sm['coverage']}, "
                               f"solidity={sm['solidity']}, n_contours={sm['n_contours']}, "
                               f"area={sm['area']}")
                if r.get("likely_unrecognized"):
                    st.error(f"This image doesn't closely resemble any trained category "
                             f"(bottle, cable, or tile) -- best match similarity was only "
                             f"{r['max_similarity']:.2f}. Results below may not be meaningful.")
                elif r.get("category_mismatch"):
                    st.warning(f"This image looks more like **{r['best_category']}** than "
                               f"**{category}** -- consider switching category and re-uploading "
                               f"for accurate thresholds.")

                img_col1, img_col2, metrics_col = st.columns([3, 3, 2], gap="small")
                with img_col1:
                    st.markdown("**Original**")
                    st.image(r["pil_img"], use_container_width=True)
                with img_col2:
                    st.markdown("**Anomaly Heatmap**")
                    st.image(r["blended"], use_container_width=True)
                with metrics_col:
                    st.markdown(f"""
                    <div class="metrics-stack">
                        <div class="metric-card"><span class="metric-index">01</span><div class="metric-value {r['css_class']}">{r['status']}</div>
                            <div class="metric-label">Inspection Result</div></div>
                        <div class="metric-card"><span class="metric-index">02</span><div class="metric-value">{r['anomaly_score']:.2f}</div>
                            <div class="metric-label">Anomaly Score</div></div>
                        <div class="metric-card"><span class="metric-index">03</span><div class="metric-value">{r['threshold']:.1f}</div>
                            <div class="metric-label">Threshold</div></div>
                    </div>
                    """, unsafe_allow_html=True)

                if r["defect_type"]:
                    st.caption(f"Likely type: {r['defect_type']}  _(heuristic suggestion, not a diagnosis)_")

                st.write("")
                fb_col1, fb_col2 = st.columns([1, 2])
                with fb_col1:
                    if st.button("Agree with result", key=f"agree_{r['file_key']}"):
                        log_feedback(r["file_key"], r["name"], category, r["status"],
                                     r["defect_type"], "agree")
                        st.success("Thanks -- feedback recorded.")
                with fb_col2:
                    reason = st.selectbox(
                        "Report an issue with this result",
                        ["", "Wrong category", "Wrong defect type",
                         "False positive (flagged but actually normal)",
                         "False negative (missed a real defect)", "Other"],
                        key=f"reason_{r['file_key']}",
                    )
                    if reason and st.button("Submit report", key=f"report_{r['file_key']}"):
                        log_feedback(r["file_key"], r["name"], category, r["status"],
                                     r["defect_type"], "report", reason)
                        st.warning("Thanks -- issue logged.")

    if st.session_state.history:
        st.markdown("### Inspection History (this session)")
        st.dataframe(st.session_state.history, use_container_width=True)
        if st.button("Clear history"):
            st.session_state.history = []
            st.session_state.processed_keys = set()
            st.session_state.feature_cache = {}
            st.rerun()
    else:
        st.info("Upload an image to run inspection.")

st.markdown("---")
st.caption("IntellectAI · Asfiya Fatima · Bhilai Institute of Technology, Durg (BIT Durg)")
