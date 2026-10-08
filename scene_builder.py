import cv2
import numpy as np
from utils import get_dominant_color, draw_label, COLOR_BGR_MAP


class SceneBuilder:
    """
    Enriches raw YOLO detections with:
      - dominant color per object crop
      - spatial position label (e.g. 'top-left')
      - human-readable scene description string
    """

    # ── Public API ────────────────────────────────────────────────────────────

    def build(self, frame: np.ndarray, detections: list[dict]) -> dict:
        """
        Build a scene context dict from a frame + raw YOLO detections.

        Returns:
            {
              objects       : list of enriched object dicts,
              object_counts : {class_name: count},
              total_count   : int,
              description   : str,
            }
        """
        h, w = frame.shape[:2]
        scene = {
            "objects":       [],
            "object_counts": {},
            "total_count":   len(detections),
        }

        for det in detections:
            x1, y1, x2, y2 = det["bbox"]

            # Guard against degenerate boxes
            x1, x2 = max(0, x1), min(w, x2)
            y1, y2 = max(0, y1), min(h, y2)
            if x2 <= x1 or y2 <= y1:
                continue

            crop = frame[y1:y2, x1:x2]
            color_bgr, color_name = get_dominant_color(crop)

            clothing_color = color_name
            clothing_bgr = color_bgr
            if det["class"] == "person":
                box_h = y2 - y1
                box_w = x2 - x1
                # Torso is roughly 15% to 60% down from top, central 70% width
                t_y1 = y1 + int(box_h * 0.15)
                t_y2 = y1 + int(box_h * 0.60)
                t_x1 = x1 + int(box_w * 0.15)
                t_x2 = x2 - int(box_w * 0.15)
                if t_y2 > t_y1 and t_x2 > t_x1:
                    torso_crop = frame[t_y1:t_y2, t_x1:t_x2]
                    t_bgr, t_name = get_dominant_color(torso_crop)
                    if t_name not in ("unknown", "colorful"):
                        clothing_color = t_name
                        clothing_bgr = t_bgr

            enriched = {
                "id":             det["id"],
                "class":          det["class"],
                "confidence":     det["confidence"],
                "bbox":           (x1, y1, x2, y2),
                "color":          color_name,
                "color_bgr":      color_bgr,
                "clothing_color": clothing_color,
                "clothing_bgr":   clothing_bgr,
                "position":       self._position_label(w, h, x1, y1, x2, y2),
            }
            scene["objects"].append(enriched)

            cls = det["class"]
            scene["object_counts"][cls] = scene["object_counts"].get(cls, 0) + 1

        scene["description"] = self._describe(scene)
        return scene

    # ── HUD overlay ───────────────────────────────────────────────────────────

    def draw_summary(self, frame: np.ndarray, scene: dict) -> np.ndarray:
        """
        Draw a heads-up display bar at the top and a hint bar at the bottom.
        """
        h, w = frame.shape[:2]

        # ── Top bar ──────────────────────────────────────────────────────
        bar_h = 32
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, bar_h), (15, 15, 15), -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

        if scene["object_counts"]:
            counts_txt = "  |  ".join(
                f"{v}x {k}" for k, v in scene["object_counts"].items()
            )
        else:
            counts_txt = "No objects detected"

        cv2.putText(frame, f"  {counts_txt}",
                    (4, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.52,
                    (0, 230, 140), 1, cv2.LINE_AA)

        # Frame count badge (top-right)
        badge = f"Objects: {scene['total_count']}"
        (bw, _), _ = cv2.getTextSize(badge, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1)
        cv2.putText(frame, badge,
                    (w - bw - 8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.52,
                    (80, 200, 255), 1, cv2.LINE_AA)

        # ── Bottom hint bar ───────────────────────────────────────────────
        hint_h = 24
        overlay2 = frame.copy()
        cv2.rectangle(overlay2, (0, h - hint_h), (w, h), (15, 15, 15), -1)
        cv2.addWeighted(overlay2, 0.75, frame, 0.25, 0, frame)

        hint = "  [Q] Query   [C] Switch Cam   [S] Save frame   [ESC] Exit"
        cv2.putText(frame, hint,
                    (4, h - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.44,
                    (180, 180, 180), 1, cv2.LINE_AA)

        return frame

    # ── Private helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _position_label(fw, fh, x1, y1, x2, y2) -> str:
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        horiz = "left"   if cx < fw / 3 else ("right" if cx > 2 * fw / 3 else "center")
        vert  = "top"    if cy < fh / 3 else ("bottom" if cy > 2 * fh / 3 else "middle")
        return f"{vert}-{horiz}"

    @staticmethod
    def _describe(scene: dict) -> str:
        if not scene["objects"]:
            return "No objects detected in the frame."
        parts = [
            f"a {o['color']} {o['class']} (ID:{o['id']}, {o['position']})"
            for o in scene["objects"]
        ]
        return "In the frame: " + ", ".join(parts) + "."
