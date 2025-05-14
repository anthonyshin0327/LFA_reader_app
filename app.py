# Clean LFA Reader App with Auto/Manual Exposure Toggle

import streamlit as st
import numpy as np
import cv2
from PIL import Image
import plotly.graph_objs as go
from scipy.signal import find_peaks, peak_widths
from scipy.ndimage import gaussian_filter1d, binary_dilation
import pandas as pd

st.set_page_config(page_title="LFA Analyzer", layout="wide")
st.title("LFA Analyzer with Per-Strip Auto Exposure")

st.sidebar.header("Configuration")
rows = st.sidebar.number_input("Rows", 1, value=3)
cols = st.sidebar.number_input("Columns", 1, value=4)
x_offset = st.sidebar.number_input("X Offset", 0, value=50)
y_offset = st.sidebar.number_input("Y Offset", 0, value=50)
w = st.sidebar.number_input("Strip Width", 10, value=100)
h = st.sidebar.number_input("Strip Height", 10, value=300)
x_spacing = st.sidebar.number_input("Horizontal Spacing", 0, value=20)
y_spacing = st.sidebar.number_input("Vertical Spacing", 0, value=20)
orientation = st.sidebar.selectbox("Orientation", ["Vertical", "Horizontal"])

# File uploader
uploaded_file = st.file_uploader("Upload LFA Image", type=["jpg", "jpeg", "png"])

# Find optimal exposure

def find_optimal_exposure(image, orientation):
    for boost in np.arange(0, 100.1, 0.1):
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        enhanced = 255 - cv2.convertScaleAbs(gray, beta=boost)
        axis = 1 if orientation == "Vertical" else 0
        profile = np.mean(enhanced, axis=axis)
        profile = gaussian_filter1d(profile, 4)
        peaks, _ = find_peaks(profile, distance=20, prominence=5)
        mask = np.zeros_like(profile, dtype=bool)
        if peaks.size:
            results = peak_widths(profile, peaks, rel_height=1.0)
            for i in range(len(peaks)):
                mask[int(results[2][i]):int(results[3][i])+1] = True
        bg = np.median(profile[~mask]) if profile[~mask].size else 0
        if bg <= 0:
            return round(boost, 1)
    return 10  # fallback

# Main logic
if uploaded_file:
    pil_img = Image.open(uploaded_file).convert("RGB")
    image = np.array(pil_img)
    img_h, img_w = image.shape[:2]

    # Strip coordinates
    strip_coords = [(x_offset + j*(w + x_spacing), y_offset + i*(h + y_spacing),
                     x_offset + j*(w + x_spacing) + w, y_offset + i*(h + y_spacing) + h)
                    for i in range(rows) for j in range(cols)]

    # Exposure handling
    st.info("Auto-exposure will be set independently for each strip to achieve background = 0.")

    # Visual overlay images
    preview_img = image.copy()
    for (x1, y1, x2, y2) in strip_coords:
        cv2.rectangle(preview_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    adj_img = 255 - cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    adj_img_color = cv2.cvtColor(adj_img, cv2.COLOR_GRAY2RGB)
    for (x1, y1, x2, y2) in strip_coords:
        cv2.rectangle(adj_img_color, (x1, y1), (x2, y2), (0, 255, 0), 2)

    col1, col2 = st.columns(2)
    with col1:
        st.image(preview_img, caption="Original + ROI", use_container_width=True)
    with col2:
        st.image(adj_img_color, caption="Adjusted Image", use_container_width=True)

    # Analysis
    summary, fig, thumbs = [], go.Figure(), []
    for idx, (x1, y1, x2, y2) in enumerate(strip_coords):
        strip = image[y1:y2, x1:x2]
        best_boost = find_optimal_exposure(strip, orientation)
        gray = cv2.cvtColor(strip, cv2.COLOR_RGB2GRAY)
        enhanced = 255 - cv2.convertScaleAbs(gray, beta=best_boost)
        thumbs.append(Image.fromarray(cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)))

        axis = 1 if orientation == "Vertical" else 0
        profile = np.mean(enhanced, axis=axis)
        profile = gaussian_filter1d(profile, 4)

        peaks, _ = find_peaks(profile, distance=20, prominence=5)
        mask = np.zeros_like(profile, dtype=bool)
        if peaks.size:
            results = peak_widths(profile, peaks, rel_height=1.0)
            for i in range(len(peaks)):
                mask[int(results[2][i]):int(results[3][i])+1] = True
        bg = np.median(profile[~mask]) if profile[~mask].size else 0
        bgsub = profile - bg

        test_peak = control_peak = test_pos = control_pos = None
        for peak in peaks:
            label = "Control" if (orientation == "Vertical" and peak < len(profile)//2) or (orientation == "Horizontal" and peak >= len(profile)//2) else "Test"
            strength = profile[peak] - bg
            if label == "Test":
                test_peak, test_pos = strength, peak
            else:
                control_peak, control_pos = strength, peak

        summary.append({
            "Strip": idx+1,
            "ExposureBoost": best_boost,
            "Background": round(bg, 2),
            "TLH": test_peak,
            "CLH": control_peak,
            "T_location": test_pos,
            "C_location": control_pos
        })

        fig.add_trace(go.Scatter3d(x=list(range(len(bgsub))),
                                   y=[idx]*len(bgsub),
                                   z=bgsub,
                                   mode='lines',
                                   name=f"Strip {idx+1}"))

    df = pd.DataFrame(summary)
    st.subheader("Peak Summary Table")
    st.dataframe(df)

    st.subheader("Strip Snapshots with Auto Exposure")
    st.image(thumbs, width=150)

    st.download_button("Download CSV", df.to_csv(index=False).encode(), "summary.csv")
    fig.update_layout(scene=dict(xaxis_title="Pixel", yaxis_title="Strip", zaxis_title="Intensity"))
    st.plotly_chart(fig)
