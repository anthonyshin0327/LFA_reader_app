# Clean LFA Reader App: Final Clean Version with Per-Strip Auto Exposure (Enhanced Zero Background)

import streamlit as st
import numpy as np
import cv2
from PIL import Image, ImageDraw
import plotly.graph_objs as go
from scipy.signal import find_peaks, peak_widths
from scipy.ndimage import gaussian_filter1d
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

uploaded_file = st.file_uploader("Upload LFA Image", type=["jpg", "jpeg", "png"])

def find_optimal_exposure(image, orientation):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    for alpha in np.arange(0.1, 4.0, 0.1):
        for beta in np.arange(-50.0, 50.0, 1):
            enhanced = cv2.convertScaleAbs(gray, alpha=alpha, beta=beta)
            inverted = 255 - enhanced
            axis = 1 if orientation == "Vertical" else 0
            profile = np.mean(inverted, axis=axis)
            profile = gaussian_filter1d(profile, 4)
            peaks, _ = find_peaks(profile, distance=20, prominence=5)
            mask = np.zeros_like(profile, dtype=bool)
            if peaks.size:
                results = peak_widths(profile, peaks, rel_height=1.0)
                for i in range(len(peaks)):
                    mask[int(results[2][i]):int(results[3][i])+1] = True
            bg = np.median(profile[~mask]) if profile[~mask].size else 0
            if bg <= 0:
                return round(alpha, 2), beta
    return 1.0, 0

if uploaded_file:
    pil_img = Image.open(uploaded_file).convert("RGB")
    image = np.array(pil_img)
    img_h, img_w = image.shape[:2]

    strip_coords = [(x_offset + j*(w + x_spacing), y_offset + i*(h + y_spacing),
                     x_offset + j*(w + x_spacing) + w, y_offset + i*(h + y_spacing) + h)
                    for i in range(rows) for j in range(cols)]

    st.info("Auto-exposure is applied per strip to achieve zero background.")

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

    summary, fig, thumbs, all_profiles = [], go.Figure(), [], []

    for idx, (x1, y1, x2, y2) in enumerate(strip_coords):
        strip = image[y1:y2, x1:x2]
        best_alpha, best_beta = find_optimal_exposure(strip, orientation)

        gray = cv2.cvtColor(strip, cv2.COLOR_RGB2GRAY)
        enhanced = 255 - cv2.convertScaleAbs(gray, alpha=best_alpha, beta=best_beta)

        axis = 1 if orientation == "Vertical" else 0
        profile = np.mean(enhanced, axis=axis)
        profile = gaussian_filter1d(profile, 4)
        all_profiles.append(profile)

    # Global 3D-validated peak detection pass
    for idx, profile in enumerate(all_profiles):
        peaks, _ = find_peaks(profile, distance=20, prominence=0.01)
        mask = np.zeros_like(profile, dtype=bool)
        if peaks.size:
            results = peak_widths(profile, peaks, rel_height=1.0)
            for i in range(len(peaks)):
                mask[int(results[2][i]):int(results[3][i])+1] = True
        bg = np.median(profile[~mask]) if profile[~mask].size else 0
        bgsub = profile - bg

        control_peak = test_peak = control_pos = test_pos = None
        if peaks.size >= 2:
            control_pos = peaks[0]
            test_pos = peaks[-1]
            control_peak = profile[control_pos] - bg
            test_peak = profile[test_pos] - bg
        elif peaks.size == 1:
            if peaks[0] < len(profile) / 2:
                control_pos = peaks[0]
                control_peak = profile[control_pos] - bg
            else:
                test_pos = peaks[0]
                test_peak = profile[test_pos] - bg

        total = (test_peak or 0) + (control_peak or 0)
        norm_tlh = test_peak / total if test_peak is not None and total else None
        norm_clh = control_peak / total if control_peak is not None and total else None
        clh_div_tlh = control_peak / test_peak if test_peak not in [None, 0] and control_peak is not None else None
        tlh_div_clh = test_peak / control_peak if control_peak not in [None, 0] and test_peak is not None else None
        tlh_minus_clh = (test_peak - control_peak) if test_peak is not None and control_peak is not None else None
        norm_diff = ((test_peak - control_peak) / total) if None not in (test_peak, control_peak) and total else None

        summary.append({
            "Strip": idx+1,
            "Background": round(bg, 2),
            "TLH": test_peak,
            "CLH": control_peak,
            "T_location": test_pos,
            "C_location": control_pos,
            "Norm_TLH": norm_tlh,
            "Norm_CLH": norm_clh,
            "CLH/TLH": clh_div_tlh,
            "TLH/CLH": tlh_div_clh,
            "TLH-CLH": tlh_minus_clh,
            "Norm_TLH-CLH": norm_diff
        })

        fig.add_trace(go.Scatter3d(x=list(range(len(bgsub))),
                                   y=[idx]*len(bgsub),
                                   z=bgsub,
                                   mode='lines',
                                   name=f"Strip {idx+1}"))

    df = pd.DataFrame(summary)
    st.subheader("Peak Summary Table (from 3D profile)")
    st.dataframe(df)

    st.download_button("Download CSV", df.to_csv(index=False).encode(), "summary.csv")
    fig.update_layout(scene=dict(xaxis_title="Pixel", yaxis_title="Strip", zaxis_title="Intensity"))
    st.plotly_chart(fig)
