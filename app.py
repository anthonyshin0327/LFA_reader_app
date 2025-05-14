import streamlit as st
import numpy as np
import cv2
from PIL import Image
import plotly.graph_objs as go
from scipy.signal import find_peaks
import pandas as pd

st.set_page_config(page_title="Template-Based LFA Analyzer", layout="wide")
st.title("Fixed-Position Multi-Strip LFA Analyzer")

# -------------------------------
# User Input for Strip Layout
# -------------------------------
st.sidebar.header("Strip Grid Configuration")
rows = st.sidebar.number_input("Number of Rows", min_value=1, value=3)
cols = st.sidebar.number_input("Number of Columns", min_value=1, value=4)
w = st.sidebar.number_input("Strip Width (px)", min_value=10, value=100)
h = st.sidebar.number_input("Strip Height (px)", min_value=10, value=300)
x_spacing = st.sidebar.number_input("Horizontal Spacing (px)", min_value=0, value=20)
y_spacing = st.sidebar.number_input("Vertical Spacing (px)", min_value=0, value=20)
x_offset = st.sidebar.number_input("X Offset (px)", min_value=0, value=50)
y_offset = st.sidebar.number_input("Y Offset (px)", min_value=0, value=50)

orientation = st.sidebar.selectbox("Read Orientation", ["Vertical", "Horizontal"])

# -------------------------------
# Line Intensity Analyzer
# -------------------------------
def compute_intensity_profile(strip_img, orientation):
    if strip_img.size == 0:
        return np.array([])
    gray = cv2.cvtColor(strip_img, cv2.COLOR_RGB2GRAY)
    axis = 1 if orientation == "Vertical" else 0
    return np.mean(gray, axis=axis)

uploaded_file = st.file_uploader("Upload standardized LFA image", type=["jpg", "jpeg", "png"])
if uploaded_file:
    pil_image = Image.open(uploaded_file).convert("RGB")
    image = np.array(pil_image)
    img_h, img_w = image.shape[:2]

    st.subheader("Image Preview with Grid-Based ROI Overlay and Pixel Ruler")
    strip_coords = []
    for i in range(rows):
        for j in range(cols):
            x1 = x_offset + j * (w + x_spacing)
            y1 = y_offset + i * (h + y_spacing)
            x2 = x1 + w
            y2 = y1 + h
            strip_coords.append((x1, y1, x2, y2))
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    # roi_coords.append((x1, y1, x2, y2))  # Removed undefined variable

    # Draw pixel ruler every 50px
    ruler_layer = image.copy()
    for y in range(0, img_h, 50):
        cv2.line(ruler_layer, (0, y), (25, y), (255, 0, 0), 1)
        cv2.putText(ruler_layer, str(y), (30, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
    for x in range(0, img_w, 50):
        cv2.line(ruler_layer, (x, 0), (x, 25), (255, 0, 0), 1)
        cv2.putText(ruler_layer, str(x), (x + 2, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
    display_image = ruler_layer.copy()

    st.image(display_image, caption="ROI Preview with Pixel Ruler", use_column_width=False, width=800)

    st.subheader("Strip Intensity Profiles")

    results = []
    summary = []
    all_x, all_y, all_z = [], [], []
    all_color, all_label = [], []

    for idx, (x1, y1, x2, y2) in enumerate(strip_coords):
        strip = image[y1:y2, x1:x2].copy()
        intensity = compute_intensity_profile(strip, orientation)
        if intensity.size == 0:
            continue

        reversed_intensity = intensity[::-1]
        true_intensity = intensity
        inverted = -intensity
        peaks, _ = find_peaks(inverted, distance=20, prominence=5)

        # Ridge line
        for i, val in enumerate(reversed_intensity):
            all_x.append(i)
            all_y.append(idx)
            all_z.append(val)
            all_color.append("lightgray")
            all_label.append(None)

        # Peaks
        test_peak = control_peak = None
        test_pos = control_pos = None
        for peak in peaks:
            if orientation == "Vertical":
                label = "Control" if peak < len(intensity) / 2 else "Test"
            else:
                label = "Test" if peak < len(intensity) / 2 else "Control"
            color = "blue" if label == "Control" else "red"
            results.append({
                "Strip": idx + 1,
                "Line": label,
                "Peak Position (px)": int(peak),
                "Intensity": float(abs(intensity[peak]))
            })
            all_x.append(peak)
            all_y.append(idx)
            all_z.append(-intensity[peak])
            all_color.append(color)
            all_label.append(f"{label} (S{idx+1})")

            if label == "Test":
                test_peak = abs(intensity[peak])
                test_pos = int(peak)
            elif label == "Control":
                control_peak = abs(intensity[peak])
                control_pos = int(peak)

        summary.append({
            "Strip": idx + 1,
            "TLH": test_peak,
            "CLH": control_peak,
            "T_location": test_pos,
            "C_location": control_pos
        })

    if results:
        df = pd.DataFrame(summary)
        st.subheader("Peak Intensity Data")
        st.dataframe(df)

        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name="lfa_peak_summary.csv",
            mime="text/csv"
        )

        st.subheader("3D Ridge Plot of LFA Strip Intensities")
        ridge3d_fig = go.Figure()

        # Add smooth lines for each strip
        strip_profiles = df.groupby("Strip")
        for strip_id, strip_data in strip_profiles:
            intensity_profile = compute_intensity_profile(
                image[strip_coords[strip_id - 1][1]:strip_coords[strip_id - 1][3],
                      strip_coords[strip_id - 1][0]:strip_coords[strip_id - 1][2]].copy(), orientation
            )[::-1]
            ridge3d_fig.add_trace(go.Scatter3d(
                x=list(range(len(intensity_profile))),
                y=[strip_id - 1] * len(intensity_profile),
                z=-intensity_profile,
                mode='lines',
                line=dict(width=2, color='black'),
                name=f'S{strip_id} Band'
            ))

        ridge3d_fig.update_layout(
            scene=dict(
                xaxis_title="Pixel Position",
                yaxis_title="Strip",
                zaxis_title="Abs Intensity"
            ),
            title="3D Ridge Plot with Filled Peaks"
        )
        st.plotly_chart(ridge3d_fig)
