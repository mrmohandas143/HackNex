import cv2
import numpy as np

# ── HSV Color ranges (lower, upper) ─────────────────────────────────────────
# Each entry: (name, lower_hsv, upper_hsv)
COLOR_RANGES = [
    ("red",    [0,   100, 80],  [10,  255, 255]),
    ("red",    [165, 100, 80],  [180, 255, 255]),   # red wraps around in HSV
    ("orange", [10,  100, 80],  [25,  255, 255]),
    ("yellow", [25,  100, 80],  [35,  255, 255]),
    ("green",  [35,  50,  40],  [85,  255, 255]),
    ("cyan",   [85,  50,  40],  [100, 255, 255]),
    ("blue",   [100, 50,  40],  [130, 255, 255]),
    ("purple", [130, 50,  40],  [160, 255, 255]),
    ("pink",   [155, 40,  100], [165, 255, 255]),
]

# Display BGR values per color name
COLOR_BGR_MAP = {
    "red":     (0,   0,   210),
    "orange":  (0,   140, 255),
    "yellow":  (0,   220, 255),
    "green":   (0,   200, 50),
    "cyan":    (230, 230, 0),
    "blue":    (230, 80,  0),
    "purple":  (200, 0,   180),
    "pink":    (180, 100, 255),
    "white":   (255, 255, 255),
    "black":   (30,  30,  30),
    "gray":    (140, 140, 140),
    "unknown": (128, 128, 128),
}


def get_dominant_color(crop_bgr):
    """
    Determine the dominant color in a cropped BGR image.

    Returns:
        (bgr_tuple, color_name_str)
    """
    if crop_bgr is None or crop_bgr.size == 0:
        return COLOR_BGR_MAP["unknown"], "unknown"

    # Resize to small patch for speed
    small = cv2.resize(crop_bgr, (50, 50), interpolation=cv2.INTER_AREA)
    hsv   = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)

    s_channel = hsv[:, :, 1].astype(float)
    v_channel = hsv[:, :, 2].astype(float)
    s_mean = float(np.mean(s_channel))
    v_mean = float(np.mean(v_channel))

    # ── Achromatic checks first ───────────────────────────────────────────
    if v_mean < 50:
        return COLOR_BGR_MAP["black"], "black"
    if s_mean < 40:
        if v_mean > 200:
            return COLOR_BGR_MAP["white"], "white"
        return COLOR_BGR_MAP["gray"], "gray"

    # ── Count pixels per color bucket ─────────────────────────────────────
    total_px = small.shape[0] * small.shape[1]
    counts: dict[str, int] = {}
    for (name, lo, hi) in COLOR_RANGES:
        mask  = cv2.inRange(hsv, np.array(lo, np.uint8), np.array(hi, np.uint8))
        count = int(np.count_nonzero(mask))
        counts[name] = counts.get(name, 0) + count

    if not counts:
        return COLOR_BGR_MAP["unknown"], "unknown"

    best_name  = max(counts, key=counts.get)
    best_count = counts[best_name]

    # Must cover at least 5 % of the patch to be meaningful
    if best_count < total_px * 0.05:
        return COLOR_BGR_MAP["gray"], "colorful"

    return COLOR_BGR_MAP.get(best_name, COLOR_BGR_MAP["unknown"]), best_name


def draw_label(frame, text, origin, font_scale=0.52, thickness=1,
               fg=(255, 255, 255), bg=(30, 30, 30)):
    """Draw a text label with a filled background rectangle."""
    (tw, th), baseline = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    x, y = origin
    pad = 3
    cv2.rectangle(frame,
                  (x - pad, y - th - pad - baseline),
                  (x + tw + pad, y + baseline),
                  bg, -1)
    cv2.putText(frame, text, (x, y),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, fg, thickness,
                cv2.LINE_AA)
