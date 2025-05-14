# Modified LFA Background Detection Framework

import streamlit as st
import numpy as np
import cv2
from PIL import Image
import plotly.graph_objs as go
from scipy.signal import find_peaks, peak_widths
from scipy.ndimage import gaussian_filter1d, binary_dilation
import pandas as pd

st.set_page_config(page_title="Improved LFA Background Subtraction", layout="wide")
st.title("Enhanced Multi-Strip LFA Analyzer")

st.sidebar.header("Strip Grid Configuration")
rows = st.sidebar.number_input("Number of Rows", min_value=1, value=3)
cols = st.sidebar.number_input("Number of Columns", min_value=1, value=4)
x_offset = st.sidebar.number_input("X Offset (px)", min_value=0, value=50)
y_offset = st.sidebar.number_input("Y Offset (px)", min_value=0, value=50)
w = st.sidebar.number_input("Strip Width (px)", min_value=10, value=100)
h = st.sidebar.number_input("Strip Height (px)", min_value=10, value=300)
x_spacing = st.sidebar.number_input("Horizontal Spacing (px)", min_value=0, value=20)
y_spacing = st.sidebar.number_input("Vertical Spacing (px)", min_value=0, value=20)

orientation = st.sidebar.selectbox("Read Orientation", ["Vertical", "Horizontal"])
exposure_boost = st.sidebar.slider("Exposure Boost (0-100)", min_value=0, max_value=100, value=10)


def compute_intensity_profile(strip_img, orientation):
    if strip_img.size == 0:
        return np.array([])
    gray = cv2.cvtColor(strip_img, cv2.COLOR_RGB2GRAY)
    gray = cv2.convertScaleAbs(gray, alpha=1, beta=exposure_boost)
    inv_gray = 255 - gray
    axis = 1 if orientation == "Vertical" else 0
    return np.mean(inv_gray, axis=axis)


uploaded_file = st.file_uploader("Upload standardized LFA image", type=["jpg", "jpeg", "png"])
if uploaded_file:
    pil_image = Image.open(uploaded_file).convert("RGB")
    image = np.array(pil_image)
    img_h, img_w = image.shape[:2]

    # Prepare overlay coordinates
    strip_coords = []
    for i in range(rows):
        for j in range(cols):
            x1 = x_offset + j * (w + x_spacing)
            y1 = y_offset + i * (h + y_spacing)
            x2 = x1 + w
            y2 = y1 + h
            strip_coords.append((x1, y1, x2, y2))

    # Image with grid overlay and ruler
    roi_image = image.copy()
    for (x1, y1, x2, y2) in strip_coords:
        cv2.rectangle(roi_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

    ruler_layer = roi_image.copy()
    for y in range(0, img_h, 50):
        cv2.line(ruler_layer, (0, y), (25, y), (255, 0, 0), 1)
        cv2.putText(ruler_layer, str(y), (30, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
    for x in range(0, img_w, 50):
        cv2.line(ruler_layer, (x, 0), (x, 25), (255, 0, 0), 1)
        cv2.putText(ruler_layer, str(x), (x + 2, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)

    # Adjusted image for analysis with mirrored green boxes
    adjusted_image = cv2.convertScaleAbs(image, alpha=1, beta=exposure_boost)
    adjusted_gray = 255 - cv2.cvtColor(adjusted_image, cv2.COLOR_RGB2GRAY)
    adjusted_image = cv2.cvtColor(adjusted_gray, cv2.COLOR_GRAY2RGB)
    for (x1, y1, x2, y2) in strip_coords:
        cv2.rectangle(adjusted_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

    st.subheader("Image Preview Panel")
    col1, col2 = st.columns(2)
    with col1:
        st.image(ruler_layer, caption="Original Image with ROI and Ruler", use_container_width=True)
    with col2:
        st.image(adjusted_image, caption="Adjusted Inverted Grayscale Image with ROI", use_container_width=True)

    summary = []
    fig2 = go.Figure()
    for idx, (x1, y1, x2, y2) in enumerate(strip_coords):
        strip = image[y1:y2, x1:x2].copy()
        intensity = compute_intensity_profile(strip, orientation)
        intensity = gaussian_filter1d(intensity, sigma=4)

        peaks, _ = find_peaks(intensity, distance=20, prominence=5)
        mask = np.zeros_like(intensity, dtype=bool)
        results = peak_widths(intensity, peaks, rel_height=1.0)
        for i, peak in enumerate(peaks):
            left = int(np.floor(results[2][i]))
            right = int(np.ceil(results[3][i]))
            mask[left:right+1] = True
        mask = binary_dilation(mask, iterations=2)
        background_pixels = intensity[~mask]
        bg = np.median(background_pixels) if background_pixels.size > 0 else 0
        bgsub = intensity - bg

        test_peak = control_peak = None
        test_pos = control_pos = None
        for peak in peaks:
            label = "Control" if (orientation == "Vertical" and peak < len(intensity) / 2) or (orientation == "Horizontal" and peak >= len(intensity) / 2) else "Test"
            peak_bgsub = intensity[peak] - bg
            if label == "Test":
                test_peak = peak_bgsub
                test_pos = peak
            else:
                control_peak = peak_bgsub
                control_pos = peak

        summary.append({
            "Background": round(bg, 4),
            "Strip": idx + 1,
            "TLH": test_peak,
            "CLH": control_peak,
            "T_location": test_pos,
            "C_location": control_pos,
            "BGsub": bgsub
        })

    if summary:
        df = pd.DataFrame(summary)
        df_display = df[["Background", "Strip", "TLH", "CLH", "T_location", "C_location"]]

        st.subheader("Detected Peak Intensities (Background Subtracted)")
        st.dataframe(df_display)

        csv = df_display.to_csv(index=False).encode("utf-8")
        st.download_button("Download CSV", csv, "lfa_peak_summary.csv", "text/csv", key="peak_summary")

        for row in summary:
            bgsub = row["BGsub"]
            fig2.add_trace(go.Scatter3d(
                x=list(range(len(bgsub))),
                y=[row["Strip"] - 1] * len(bgsub),
                z=bgsub,
                mode='lines',
                line=dict(width=2, color='orange'),
                name=f'S{row["Strip"]} BG-subtracted'
            ))

        fig2.update_layout(
            scene=dict(
                xaxis_title="Pixel Position",
                yaxis_title="Strip",
                zaxis_title="Intensity",
                xaxis=dict(showbackground=False),
                yaxis=dict(showbackground=False),
                zaxis=dict(showbackground=False)
            ),
            title="3D Ridge Plot: Background Subtracted"
        )
        st.plotly_chart(fig2)
