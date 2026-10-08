import cv2
import numpy as np
from utils import get_dominant_color, draw_label, COLOR_BGR_MAP
from attribute_analyzer import AttributeAnalyzer


class SceneBuilder:
    """
    Tier 2 Scene Builder:
    Enriches raw YOLO detections with:
      - Dominant color & posture analysis
      - Fine-grained attributes (earrings, glasses, weapons, pants, hats)
      - Temporal motion metrics (dwell time, speed, direction)
      - Spatial position label
      - Natural scene description
    """

    def __init__(self, incident_manager=None):
        self.attr_analyzer = AttributeAnalyzer()
        self.incident_manager = incident_manager

    # ── Public API ────────────────────────────────────────────────────────────

    def build(self, frame: np.ndarray, detections: list[dict],
              frame_no: int = 0, video_time_sec: float = 0.0) -> dict:
        """
        Build an enriched scene context dict from a frame + detections.
        """
        h, w = frame.shape[:2]
        scene = {
            "objects":       [],
            "object_counts": {},
            "total_count":   len(detections),
            "frame_no":      frame_no,
            "video_time_sec":video_time_sec,
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

            # Deep anatomical & attribute inspection for persons
            attrs = None
            clothing_color = color_name
            pants_color = "unknown"
            posture = "unknown"

            if det["class"] == "person":
                attrs = self.attr_analyzer.inspect_person(frame, (x1, y1, x2, y2))
                if attrs["shirt_color"] not in ("unknown", "colorful"):
                    clothing_color = attrs["shirt_color"]
                pants_color = attrs["pants_color"]
                posture = attrs["posture"]

                # Log weapon incident if detected
                if self.incident_manager and attrs.get("weapon", {}).get("detected"):
                    self.incident_manager.log_incident(
                        event_type="weapon_alert",
                        track_id=det["id"],
                        class_name="person",
                        frame=frame,
                        frame_no=frame_no,
                        video_time_sec=video_time_sec,
                        bbox=(x1, y1, x2, y2),
                        details=attrs["weapon"].get("details", "Suspicious object near hand zone"),
                        attributes=attrs
                    )

            # Update best view in incident manager
            if self.incident_manager:
                area = (x2 - x1) * (y2 - y1)
                tid = det["id"]
                if tid not in self.incident_manager.best_keyframes or area > self.incident_manager.best_keyframes[tid]["area"]:
                    rec = {
                        "id":             len(self.incident_manager.incidents) + 1,
                        "event_type":     "best_view",
                        "track_id":       tid,
                        "class":          det["class"],
                        "frame_no":       frame_no,
                        "video_time_sec": round(video_time_sec, 2),
                        "time_str":       f"{int(video_time_sec//60):02d}:{video_time_sec%60:04.1f}",
                        "bbox":           (x1, y1, x2, y2),
                        "frame":          frame.copy(),
                        "crop":           crop.copy() if crop.size > 0 else None,
                        "details":        f"{det['class'].capitalize()} (ID #{tid}) best view",
                        "attributes":     attrs or {},
                    }
                    self.incident_manager.best_keyframes[tid] = {
                        "record": rec,
                        "area":   area,
                    }

            enriched = {
                "id":               det["id"],
                "class":            det["class"],
                "confidence":       det["confidence"],
                "bbox":             (x1, y1, x2, y2),
                "color":            color_name,
                "color_bgr":        color_bgr,
                "clothing_color":   clothing_color,
                "pants_color":      pants_color,
                "posture":          posture,
                "attributes":       attrs,
                "dwell_time":       det.get("dwell_time_sec", 0.0),
                "motion_direction": det.get("motion_direction", "stationary"),
                "is_moving":        det.get("is_moving", False),
                "velocity_px_s":    det.get("velocity_px_s", 0.0),
                "position":         self._position_label(w, h, x1, y1, x2, y2),
            }
            scene["objects"].append(enriched)

            cls = det["class"]
            scene["object_counts"][cls] = scene["object_counts"].get(cls, 0) + 1

        scene["description"] = self._describe(scene)
        return scene

    # ── HUD overlay ───────────────────────────────────────────────────────────

    def draw_summary(self, frame: np.ndarray, scene: dict) -> np.ndarray:
        """Draw heads-up display bar at top and key hint bar at bottom."""
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

        # Badge: objects + tier indicator
        badge = f"Tier 2 AI  |  Objects: {scene['total_count']}"
        (bw, _), _ = cv2.getTextSize(badge, cv2.FONT_HERSHEY_SIMPLEX, 0.50, 1)
        cv2.putText(frame, badge,
                    (w - bw - 8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.50,
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
        parts = []
        for o in scene["objects"]:
            desc = f"a {o['color']} {o['class']} (ID:#{o['id']}, {o['position']}"
            if o.get("posture") and o["posture"] != "unknown":
                desc += f", {o['posture']}"
            if o.get("motion_direction") and "still" not in o["motion_direction"] and "stationary" not in o["motion_direction"]:
                desc += f", {o['motion_direction']}"
            desc += ")"
            parts.append(desc)
        return "In the frame: " + ", ".join(parts) + "."
