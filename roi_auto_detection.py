"""
ROI Auto-Detection Utilities for LFA Analyzer — v2
=================================================

Changes in v2
-------------
* **Adaptive bright/dark detection** – works whether strips are darker *or* brighter
  than background.  Chooses whichever polarity returns the expected number of peaks.
* Wider default smoothing (sigma≈7) and dynamic minimum distance between peaks so
  strips that are close together are still separated.
* Option `debug` returns intermediate 1‑D projection for quick Streamlit plotting.

Functions
---------
* `detect_strip_rois(image, rows, cols, *, orientation='Vertical', pad=5,
                     prominence=5.0, debug=False)`  → list[ROI] or (rois, profile).
* `draw_rois(image, rois, color=(0,255,0), thickness=2)` – unchanged.
"""

from __future__ import annotations

from typing import List, Tuple, Union

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks, peak_widths

ROI = Tuple[int, int, int, int]


def _select_peaks(
    profile: np.ndarray,
    expected_peaks: int,
    prominence: float,
    min_dist: int,
):
    """Return (peak_indices, inverted_used) if successful, else (None, None)."""
    # Try bright‑as‑positive first
    peaks_pos, _ = find_peaks(profile, distance=min_dist, prominence=prominence)
    if len(peaks_pos) >= expected_peaks:
        return np.sort(peaks_pos), False

    # Now try treating bright strips as minima
    inv = -profile
    peaks_neg, _ = find_peaks(inv, distance=min_dist, prominence=prominence)
    if len(peaks_neg) >= expected_peaks:
        return np.sort(peaks_neg), True
    return None, None


def detect_strip_rois(
    image: np.ndarray,
    rows: int,
    cols: int,
    *,
    orientation: str = "Vertical",
    pad: int = 5,
    prominence: float = 5.0,
    debug: bool = False,
) -> Union[List[ROI], Tuple[List[ROI], np.ndarray]]:
    """Detect bounding boxes for LFA strips automatically.

    Works for both bright‑on‑dark and dark‑on‑bright strips by adaptively
    selecting the correct polarity.
    """
    if orientation not in {"Vertical", "Horizontal"}:
        raise ValueError("orientation must be 'Vertical' or 'Horizontal'")

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    axis = 0 if orientation == "Vertical" else 1  # collapse rows for vertical strips
    profile = np.mean(gray, axis=axis)

    # Smooth to reduce pixel‑scale noise; sigma scales with profile length
    sigma = max(3, int(0.01 * profile.size))
    profile_s = gaussian_filter1d(profile, sigma=sigma)

    expected = cols if orientation == "Vertical" else rows
    min_dist = max(10, profile.size // (expected * 2))

    peaks, inverted = _select_peaks(profile_s, expected, prominence, min_dist)
    if peaks is None:
        raise RuntimeError(
            "Could not detect expected number of strip edges. Try lowering the "
            "prominence slider or check orientation/row/col settings."
        )

    # Use appropriate signal for width measurement
    wid_signal = -profile_s if inverted else profile_s
    widths = peak_widths(wid_signal, peaks, rel_height=0.5)
    left = widths[2].astype(int) - pad
    right = widths[3].astype(int) + pad

    # Ensure indices in bounds
    left = np.clip(left, 0, profile.size - 1)
    right = np.clip(right, 1, profile.size)

    rois_1d = list(zip(left, right))[:expected]

    all_rois: List[ROI] = []
    if orientation == "Vertical":
        strip_h = gray.shape[0] // rows
        for r in range(rows):
            y1, y2 = r * strip_h, (r + 1) * strip_h
            for c in range(cols):
                x1, x2 = rois_1d[c]
                all_rois.append((x1, y1, x2, y2))
    else:
        strip_w = gray.shape[1] // cols
        for r in range(rows):
            y1, y2 = rois_1d[r]
            for c in range(cols):
                x1, x2 = c * strip_w, (c + 1) * strip_w
                all_rois.append((x1, y1, x2, y2))

    if debug:
        return all_rois, profile_s  # type: ignore[return-value]
    return all_rois


def draw_rois(
    image: np.ndarray,
    rois: List[ROI],
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
) -> np.ndarray:
    vis = image.copy()
    for (x1, y1, x2, y2) in rois:
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, thickness)
    return vis
