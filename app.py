"""
Streamlit LFA Analyzer (Auto‑Detected ROIs)
==========================================
A complete Streamlit application that analyses LFA images with automatic strip ROI
segmentation.  Manual ROI mode is still available as a fallback.

Requirements
------------
* Streamlit
* OpenCV‑Python
* NumPy, SciPy, Pillow, Plotly, Pandas
* roi_auto_detection.py (placed in the same folder or on Python path)

Run with:
    streamlit run lfa_analyzer_app.py
"""

import cv2
import numpy as np
import pandas as pd
import plotly.graph_objs as go
import streamlit as st
from PIL import Image
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks, peak_widths

from roi_auto_detection import detect_strip_rois, draw_rois

# --------------------------- Streamlit UI ------------------------------------

st.set_page_config(page_title="LFA Analyzer", layout="wide")
st.title("📈 LFA Analyzer with Auto‑Detected ROIs")

with st.sidebar:
    st.header("Configuration")

    mode = st.radio("ROI Mode", ["Auto", "Manual"], horizontal=True)

    # Common parameters
    rows = st.number_input("Rows", 1, value=3, step=1)
    cols = st.number_input("Columns", 1, value=4, step=1)
    orientation = st.selectbox("Orientation", ["Vertical", "Horizontal"], index=0)

    if mode == "Manual":
        st.markdown("---")
        st.subheader("Manual ROI Parameters")
        x_offset = st.number_input("X Offset", 0, value=50)
        y_offset = st.number_input("Y Offset", 0, value=50)
        w = st.number_input("Strip Width", 10, value=100)
        h = st.number_input("Strip Height", 10, value=300)
        x_spacing = st.number_input("Horizontal Spacing", 0, value=20)
        y_spacing = st.number_input("Vertical Spacing", 0, value=20)
    else:  # Auto parameters
        st.markdown("---")
        st.subheader("Auto‑Detection Parameters")
        pad = st.slider("Padding around edges (px)", 0, 20, value=5)
        prominence = st.slider("Edge prominence", 1.0, 15.0, value=5.0)

# File uploader
uploaded_file = st.file_uploader("Upload LFA Image", type=["jpg", "jpeg", "png"])

# ----------------------- Exposure helper -------------------------------------

def find_optimal_exposure(strip: np.ndarray, orientation: str) -> float:
    """Return brightness boost (beta) that drives background to ~0 for a strip."""
    for boost in np.arange(0, 100.1, 0.5):
        gray = cv2.cvtColor(strip, cv2.COLOR_RGB2GRAY)
        enhanced = 255 - cv2.convertScaleAbs(gray, beta=boost)
        axis = 1 if orientation == "Vertical" else 0
        profile = gaussian_filter1d(np.mean(enhanced, axis=axis), 4)
        peaks, _ = find_peaks(profile, distance=20, prominence=5)
        mask = np.zeros_like(profile, dtype=bool)
        if peaks.size:
            results = peak_widths(profile, peaks, rel_height=1.0)
            for i in range(len(peaks)):
                mask[int(results[2][i]): int(results[3][i]) + 1] = True
        bg = np.median(profile[~mask]) if profile[~mask].size else 0
        if bg <= 0:
            return round(boost, 1)
    return 10.0  # fallback

# --------------------------- Main logic --------------------------------------

if uploaded_file:
    pil_img = Image.open(uploaded_file).convert("RGB")
    image = np.array(pil_img)
    img_h, img_w = image.shape[:2]

    # ----------------- ROI calculation ------------------
    if mode == "Manual":
        strip_coords = [
            (
                x_offset + j * (w + x_spacing),
                y_offset + i * (h + y_spacing),
                x_offset + j * (w + x_spacing) + w,
                y_offset + i * (h + y_spacing) + h,
            )
            for i in range(rows)
            for j in range(cols)
        ]
    else:
        try:
            strip_coords = detect_strip_rois(
                image, rows, cols, orientation=orientation, pad=pad, prominence=prominence
            )
        except RuntimeError as e:
            st.error(f"ROI auto‑detection failed: {e}")
            st.stop()

    # ----------------- Visual overlay -------------------
    preview_img = draw_rois(image, strip_coords)

    col1, col2 = st.columns(2)
    with col1:
        st.image(preview_img, caption="Original + ROIs", use_container_width=True)
    with col2:
        gray = 255 - cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        st.image(cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB), caption="Inverted Gray", use_container_width=True)

    # ----------------- Analysis per strip ---------------
    st.info("Auto‑exposure applied independently per detected strip.")

    summary, fig, thumbs = [], go.Figure(), []

    for idx, (x1, y1, x2, y2) in enumerate(strip_coords):
        strip = image[y1:y2, x1:x2]
        best_boost = find_optimal_exposure(strip, orientation)
        gray_s = cv2.cvtColor(strip, cv2.COLOR_RGB2GRAY)
        enhanced = 255 - cv2.convertScaleAbs(gray_s, beta=best_boost)
        thumbs.append(Image.fromarray(cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)))

        axis = 1 if orientation == "Vertical" else 0
        profile = gaussian_filter1d(np.mean(enhanced, axis=axis), 4)

        peaks, _ = find_peaks(profile, distance=20, prominence=5)
        mask = np.zeros_like(profile, dtype=bool)
        if peaks.size:
            results = peak_widths(profile, peaks, rel_height=1.0)
            for i in range(len(peaks)):
                mask[int(results[2][i]): int(results[3][i]) + 1] = True
        bg = np.median(profile[~mask]) if profile[~mask].size else 0
        bgsub = profile - bg

        test_peak = control_peak = test_pos = control_pos = None
        for peak in peaks:
            label = (
                "Control"
                if (
                    (orientation == "Vertical" and peak < len(profile) // 2)
                    or (orientation == "Horizontal" and peak >= len(profile) // 2)
                )
                else "Test"
            )
            strength = profile[peak] - bg
            if label == "Test":
                test_peak, test_pos = strength, peak
            else:
                control_peak, control_pos = strength, peak

        summary.append(
            {
                "Strip": idx + 1,
                "ExposureBoost": best_boost,
                "Background": round(float(bg), 2),
                "TLH": test_peak,
                "CLH": control_peak,
                "T_location": test_pos,
                "C_location": control_pos,
            }
        )

        fig.add_trace(
            go.Scatter3d(
                x=list(range(len(bgsub))),
                y=[idx] * len(bgsub),
                z=bgsub,
                mode="lines",
                name=f"Strip {idx + 1}",
            )
        )

    df = pd.DataFrame(summary)

    st.subheader("Peak Summary Table")
    st.dataframe(df)

    st.subheader("Strip Snapshots (Auto Exposure)")
    st.image(thumbs, width=150)

    csv = df.to_csv(index=False).encode()
    st.download_button("Download CSV", csv, "summary.csv", mime="text/csv")

    fig.update_layout(
        scene=dict(xaxis_title="Pixel", yaxis_title="Strip", zaxis_title="Intensity"),
        height=600,
    )
    st.plotly_chart(fig, use_container_width=True)