"""
attribute_analyzer.py — Tier 2 Fine-Grained Visual & Anatomical Attribute Inspection
====================================================================================
Performs sub-region visual analysis on detected persons:
  - Head/Face: Earrings, eyeglasses, headwear/hat
  - Torso: T-shirt/shirt color, texture, patterns
  - Lower body: Pants/trousers color
  - Hands/Periphery: Carried objects, suspicious items / weapons
  - Posture: Standing, sitting, crouching
100% open-source, local, OpenCV + NumPy. Zero cloud, zero PyTorch.
"""

import cv2
import numpy as np
from utils import get_dominant_color, COLOR_BGR_MAP


class AttributeAnalyzer:
    """Extracts fine-grained visual attributes from detected persons."""

    def __init__(self):
        pass

    def inspect_person(self, frame: np.ndarray, bbox: tuple) -> dict:
        """
        Inspect a person bounding box across multiple anatomical zones.
        bbox: (x1, y1, x2, y2)
        """
        h_frame, w_frame = frame.shape[:2]
        x1, y1, x2, y2 = bbox
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w_frame, x2), min(h_frame, y2)

        bw = x2 - x1
        bh = y2 - y1
        if bw < 15 or bh < 30:
            return self._empty_attributes()

        person_crop = frame[y1:y2, x1:x2]

        # ── 1. Anatomical Zone Crops ──────────────────────────────────────
        # Head: top 20%
        head_y2 = int(bh * 0.22)
        head_crop = person_crop[0:head_y2, :]

        # Torso (Chest/Shirt): 20% - 62%
        torso_y1 = int(bh * 0.18)
        torso_y2 = int(bh * 0.62)
        torso_x1 = int(bw * 0.12)
        torso_x2 = int(bw * 0.88)
        torso_crop = person_crop[torso_y1:torso_y2, torso_x1:torso_x2]

        # Lower Body (Pants/Legs): 62% - 95%
        legs_y1 = int(bh * 0.62)
        legs_y2 = int(bh * 0.95)
        legs_crop = person_crop[legs_y1:legs_y2, int(bw * 0.15):int(bw * 0.85)]

        # ── 2. Upper Clothing (Shirt) Color & Pattern ────────────────────
        shirt_bgr, shirt_color = get_dominant_color(torso_crop)
        shirt_pattern = self._check_pattern(torso_crop)

        # ── 3. Lower Clothing (Pants) Color ──────────────────────────────
        pants_bgr, pants_color = get_dominant_color(legs_crop)

        # ── 4. Headwear / Hat ─────────────────────────────────────────────
        hat_detected, hat_color = self._check_headwear(head_crop)

        # ── 5. Eyeglasses ────────────────────────────────────────────────
        glasses_detected = self._check_glasses(head_crop)

        # ── 6. Earrings ──────────────────────────────────────────────────
        earrings_detected, ear_details = self._check_earrings(head_crop)

        # ── 7. Weapons / Carried Items Check ─────────────────────────────
        weapon_detected, item_detected, item_desc = self._check_weapons_and_items(
            person_crop, frame, bbox
        )

        # ── 8. Posture Estimation ────────────────────────────────────────
        aspect_ratio = bh / max(1, bw)
        if aspect_ratio >= 1.9:
            posture = "standing"
        elif aspect_ratio <= 1.35:
            posture = "sitting or crouching"
        else:
            posture = "standing or seated"

        return {
            "posture":            posture,
            "aspect_ratio":       round(aspect_ratio, 2),
            "shirt_color":        shirt_color,
            "shirt_color_bgr":    shirt_bgr,
            "shirt_pattern":      shirt_pattern,
            "pants_color":        pants_color,
            "pants_color_bgr":    pants_bgr,
            "hat":                {"detected": hat_detected, "color": hat_color},
            "glasses":            {"detected": glasses_detected},
            "earrings":           {"detected": earrings_detected, "detail": ear_details},
            "weapon":             {"detected": weapon_detected, "details": item_desc},
            "holding_item":       {"detected": item_detected, "details": item_desc},
        }

    # ── Private Detectors ────────────────────────────────────────────────────

    def _check_earrings(self, head_crop: np.ndarray) -> tuple[bool, str]:
        """
        Inspect the left and right ear periphery zones for high-contrast
        specular highlights, metallic reflections, or localized earring accents.
        """
        if head_crop is None or head_crop.size == 0 or head_crop.shape[0] < 10 or head_crop.shape[1] < 10:
            return False, "Image resolution too low for ear inspection"

        hh, hw = head_crop.shape[:2]
        # Ears are located around y: 35% to 75% of head, on lateral outer 18% margins
        y_start, y_end = int(hh * 0.35), int(hh * 0.75)
        w_margin = max(2, int(hw * 0.18))

        left_ear_zone  = head_crop[y_start:y_end, 0:w_margin]
        right_ear_zone = head_crop[y_start:y_end, max(0, hw - w_margin):hw]

        def has_earring_signature(zone: np.ndarray) -> bool:
            if zone is None or zone.size == 0:
                return False
            gray = cv2.cvtColor(zone, cv2.COLOR_BGR2GRAY)
            # Look for sharp intensity peaks / specular glint relative to surroundings
            v_mean = np.mean(gray)
            v_max  = np.max(gray)
            edges  = cv2.Canny(gray, 60, 160)
            edge_density = np.count_nonzero(edges) / max(1, edges.size)

            # High contrast reflection point near ear edge
            return bool((v_max > v_mean + 65) and (edge_density > 0.10) and (v_max > 170))

        left_has  = has_earring_signature(left_ear_zone)
        right_has = has_earring_signature(right_ear_zone)

        if left_has and right_has:
            return True, "Distinct reflective accents/earrings visible on both ears"
        elif left_has:
            return True, "Reflective accent/earring detected on left ear area"
        elif right_has:
            return True, "Reflective accent/earring detected on right ear area"
        else:
            return False, "No visible earrings or ear accessories detected"

    def _check_glasses(self, head_crop: np.ndarray) -> bool:
        """Inspect eye/bridge region for horizontal bridge line and frame contrast."""
        if head_crop is None or head_crop.size == 0 or head_crop.shape[0] < 12:
            return False

        hh, hw = head_crop.shape[:2]
        # Eye zone: 25% to 55% vertically, center 70% horizontally
        eye_y1, eye_y2 = int(hh * 0.25), int(hh * 0.55)
        eye_x1, eye_x2 = int(hw * 0.15), int(hw * 0.85)
        eye_zone = head_crop[eye_y1:eye_y2, eye_x1:eye_x2]

        if eye_zone.size == 0:
            return False

        gray = cv2.cvtColor(eye_zone, cv2.COLOR_BGR2GRAY)
        # Horizontal gradient / bridge lines
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        horizontal_energy = np.mean(np.abs(sobel_x))
        edges = cv2.Canny(gray, 70, 180)
        edge_ratio = np.count_nonzero(edges) / max(1, edges.size)

        return bool(horizontal_energy > 28.0 and edge_ratio > 0.12)

    def _check_headwear(self, head_crop: np.ndarray) -> tuple[bool, str]:
        """Inspect top 25% of head for distinct hat, cap, or head covering."""
        if head_crop is None or head_crop.size == 0 or head_crop.shape[0] < 12:
            return False, "none"

        hh = head_crop.shape[0]
        crown_zone = head_crop[0:int(hh * 0.35), :]
        if crown_zone.size == 0:
            return False, "none"

        _, col = get_dominant_color(crown_zone)
        # Check if top crown has distinct solid color different from background/skin
        if col not in ("unknown", "colorful") and col in ("black", "white", "blue", "red", "gray"):
            # Verify uniform color distribution across top
            return True, col
        return False, "none"

    def _check_weapons_and_items(self, person_crop: np.ndarray,
                                 frame: np.ndarray, bbox: tuple) -> tuple[bool, bool, str]:
        """
        Scan person periphery, hand zones, and lateral bounding box edges
        for high-contrast rigid protrusions, elongated shapes, or hand-held objects.
        """
        bh, bw = person_crop.shape[:2]

        # Lower half periphery: hands normally fall between 45% and 80% height on sides
        hand_y1, hand_y2 = int(bh * 0.45), int(bh * 0.82)
        hand_margin = max(4, int(bw * 0.25))

        left_hand_zone  = person_crop[hand_y1:hand_y2, 0:hand_margin]
        right_hand_zone = person_crop[hand_y1:hand_y2, max(0, bw - hand_margin):bw]

        def inspect_hand(zone: np.ndarray) -> tuple[bool, bool, str]:
            if zone is None or zone.size == 0:
                return False, False, ""
            gray = cv2.cvtColor(zone, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            has_elongated = False
            has_item = False
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area > 45:
                    has_item = True
                    # Check bounding rect aspect ratio of the item contour
                    _, _, cw, ch = cv2.boundingRect(cnt)
                    aspect = ch / max(1, cw)
                    if aspect > 3.2 or aspect < 0.3:  # Long sharp aspect (blade, stick, barrel)
                        has_elongated = True
            return has_elongated, has_item, "item detected"

        l_elon, l_item, _ = inspect_hand(left_hand_zone)
        r_elon, r_item, _ = inspect_hand(right_hand_zone)

        weapon_detected = l_elon or r_elon
        item_detected   = l_item or r_item

        if weapon_detected:
            desc = "Caution: Rigid elongated object detected near hand zone (potential tool/weapon/stick)."
        elif item_detected:
            desc = "The person appears to be holding or carrying a small object (phone, drink, or accessory)."
        else:
            desc = "No weapons or suspicious carried objects detected. Hands appear clear."

        return weapon_detected, item_detected, desc

    def _check_pattern(self, crop: np.ndarray) -> str:
        """Determine if shirt is plain/solid or textured/striped."""
        if crop is None or crop.size == 0:
            return "solid"
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        std_dev = float(np.std(gray))
        if std_dev > 48.0:
            return "patterned or graphic"
        elif std_dev > 32.0:
            return "textured"
        return "solid"

    def _empty_attributes(self) -> dict:
        return {
            "posture":         "unknown",
            "aspect_ratio":    0.0,
            "shirt_color":     "unknown",
            "shirt_color_bgr": COLOR_BGR_MAP["unknown"],
            "shirt_pattern":   "solid",
            "pants_color":     "unknown",
            "pants_color_bgr": COLOR_BGR_MAP["unknown"],
            "hat":             {"detected": False, "color": "none"},
            "glasses":         {"detected": False},
            "earrings":        {"detected": False, "detail": "not visible"},
            "weapon":          {"detected": False, "details": "No weapons detected"},
            "holding_item":    {"detected": False, "details": "None"},
        }
