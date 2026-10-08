"""
incident_manager.py — Tier 2 Incident Logging, Visual Evidence & Video Seeking
================================================================================
Captures, indexes, and visualizes incidents where they occur:
  - Object entries, exits, weapon alerts, motion transitions, posture changes
  - Stores high-resolution keyframe snapshots and close-up crops
  - Renders a Picture-in-Picture (PiP) "Incident Evidence" display canvas
  - Coordinates video frame seeking so playback jumps directly to the incident point
100% open-source, local, OpenCV + NumPy.
"""

import os
import time
import cv2
import numpy as np
from collections import deque


class IncidentManager:
    """Manages incident logging, snapshot evidence generation, and query retrieval."""

    def __init__(self, max_incidents: int = 150, save_dir: str = "incidents"):
        self.save_dir = save_dir
        if self.save_dir:
            os.makedirs(self.save_dir, exist_ok=True)
        self.incidents: deque[dict] = deque(maxlen=max_incidents)
        self.best_keyframes: dict[int, dict] = {}  # track_id -> best view record

    def log_incident(self,
                     event_type: str,
                     track_id: int,
                     class_name: str,
                     frame: np.ndarray,
                     frame_no: int,
                     video_time_sec: float,
                     bbox: tuple,
                     details: str,
                     attributes: dict | None = None) -> dict:
        """
        Record a timestamped incident with full frame and zoomed crop.
        """
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = bbox
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        crop = frame[y1:y2, x1:x2].copy() if (x2 > x1 and y2 > y1) else None

        mins = int(video_time_sec // 60)
        secs = video_time_sec % 60
        time_str = f"{mins:02d}:{secs:04.1f}"

        record = {
            "id":             len(self.incidents) + 1,
            "event_type":     event_type,
            "track_id":       track_id,
            "class":          class_name,
            "frame_no":       frame_no,
            "video_time_sec": round(video_time_sec, 2),
            "time_str":       time_str,
            "timestamp":      time.time(),
            "bbox":           (x1, y1, x2, y2),
            "frame":          frame.copy(),
            "crop":           crop,
            "details":        details,
            "attributes":     attributes or {},
        }

        self.incidents.append(record)

        # Update best keyframe for this object if larger or initial
        area = (x2 - x1) * (y2 - y1)
        if track_id not in self.best_keyframes or area > self.best_keyframes[track_id]["area"]:
            self.best_keyframes[track_id] = {
                "record": record,
                "area":   area,
            }

        return record

    def find_incident_for_query(self, query: str, target_id: int | None = None, scene: dict | None = None) -> dict | None:
        """
        Match a query to the most relevant incident in history.
        """
        if not self.incidents:
            return None

        q = query.lower().strip()

        # 1. Weapon / threat / danger queries
        if any(w in q for w in ["weapon", "gun", "knife", "blade", "armed"]):
            for inc in reversed(self.incidents):
                if inc["event_type"] == "weapon_alert":
                    return inc

        # 2. Entry / arrival queries: "when did the person enter", "show entry"
        if any(w in q for w in ["enter", "entry", "arrive", "first appear", "came in", "arrived"]):
            if target_id is not None:
                for inc in self.incidents:
                    if inc["event_type"] == "entry" and inc["track_id"] == target_id:
                        return inc
            for inc in reversed(self.incidents):
                if inc["event_type"] == "entry":
                    return inc

        # 3. Exit / departure queries: "who left", "when did they leave"
        if any(w in q for w in ["exit", "leave", "left", "depart"]):
            for inc in reversed(self.incidents):
                if inc["event_type"] == "exit":
                    return inc

        # 4. Motion / direction queries
        if any(w in q for w in ["move", "moving", "direction", "walk", "walking"]):
            for inc in reversed(self.incidents):
                if inc["event_type"] == "motion":
                    return inc

        # 5. Queries targeting a specific person/ID: "what is person 1 wearing", "show person 1"
        if target_id is not None and target_id in self.best_keyframes:
            return self.best_keyframes[target_id]["record"]

        # 6. Specific color mentions in query: e.g. "red shirt", "green", "blue"
        for inc in reversed(self.incidents):
            attrs = inc.get("attributes", {})
            shirt = attrs.get("shirt_color", "")
            if shirt and shirt in q:
                return inc

        # 7. Generic "show the incident", "where did it happen", "show evidence", "jump"
        if any(w in q for w in ["incident", "evidence", "where did it happen", "where it occur", "occur", "jump", "seek"]):
            return self.incidents[-1]

        # 8. Default fallback: most recent incident
        return self.incidents[-1]

    def render_evidence_view(self, incident: dict) -> np.ndarray:
        """
        Produce a high-impact composite image highlighting the incident:
          - Full frame with bright incident highlight brackets and crosshairs
          - Top alert HUD banner with timestamp and incident type
          - Picture-in-Picture (PiP) close-up zoom window of the subject
          - Bottom navigation hint bar
        """
        frame = incident["frame"].copy()
        h, w = frame.shape[:2]

        event_type = incident["event_type"]
        track_id   = incident["track_id"]
        cls_name   = incident["class"]
        time_str   = incident["time_str"]
        frame_no   = incident["frame_no"]
        details    = incident["details"]
        x1, y1, x2, y2 = incident["bbox"]

        # Color scheme based on severity
        if event_type == "weapon_alert":
            badge_col = (0, 0, 230)      # Bright Red
            box_col   = (0, 0, 255)
            badge_txt = "INCIDENT ALERT: SUSPICIOUS / WEAPON"
        elif event_type == "entry":
            badge_col = (0, 180, 50)      # Vivid Green
            box_col   = (0, 230, 80)
            badge_txt = "INCIDENT EVENT: SUBJECT ENTRY"
        elif event_type == "exit":
            badge_col = (0, 140, 255)     # Amber Orange
            box_col   = (0, 165, 255)
            badge_txt = "INCIDENT EVENT: SUBJECT DEPARTURE"
        else:
            badge_col = (220, 100, 0)     # Blue/Cyan
            box_col   = (255, 180, 0)
            badge_txt = f"INCIDENT LOG: {event_type.upper()}"

        # ── 1. Top Header Banner ──────────────────────────────────────────
        header_h = 56
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, header_h), (12, 12, 12), -1)
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

        # Badge pill
        cv2.rectangle(frame, (8, 6), (360, 28), badge_col, -1)
        cv2.putText(frame, badge_txt, (14, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)

        # Time & Frame stamp (top right)
        meta_txt = f"Time: {time_str}  |  Frame #{frame_no}"
        (tw, _), _ = cv2.getTextSize(meta_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.50, 1)
        cv2.putText(frame, meta_txt, (w - tw - 12, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 230, 255), 1, cv2.LINE_AA)

        # Details line
        cv2.putText(frame, f"Details: {details}", (12, 47),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46, (210, 210, 210), 1, cv2.LINE_AA)

        # ── 2. Prominent Bounding Box & Highlight Brackets ────────────────
        cv2.rectangle(frame, (x1, y1), (x2, y2), box_col, 2)

        # Corner accent brackets (18px)
        sz = min(18, min((x2 - x1)//2, (y2 - y1)//2))
        t = 3
        # Top-left
        cv2.line(frame, (x1, y1), (x1 + sz, y1), box_col, t)
        cv2.line(frame, (x1, y1), (x1, y1 + sz), box_col, t)
        # Top-right
        cv2.line(frame, (x2, y1), (x2 - sz, y1), box_col, t)
        cv2.line(frame, (x2, y1), (x2, y1 + sz), box_col, t)
        # Bottom-left
        cv2.line(frame, (x1, y2), (x1 + sz, y2), box_col, t)
        cv2.line(frame, (x1, y2), (x1, y2 - sz), box_col, t)
        # Bottom-right
        cv2.line(frame, (x2, y2), (x2 - sz, y2), box_col, t)
        cv2.line(frame, (x2, y2), (x2, y2 - sz), box_col, t)

        # Incident target pin
        pin_lbl = f"INCIDENT TARGET: #{track_id} {cls_name}"
        (pw, ph), _ = cv2.getTextSize(pin_lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.50, 1)
        cv2.rectangle(frame, (x1, y1 - ph - 8), (x1 + pw + 8, y1), box_col, -1)
        cv2.putText(frame, pin_lbl, (x1 + 4, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 0, 0), 1, cv2.LINE_AA)

        # ── 3. Picture-in-Picture (PiP) Close-Up ───────────────────────────
        if incident.get("crop") is not None and incident["crop"].size > 0:
            crop = incident["crop"]
            pip_w = 170
            pip_h = 220
            crop_resized = cv2.resize(crop, (pip_w, pip_h), interpolation=cv2.INTER_LINEAR)

            pip_x1 = w - pip_w - 14
            pip_y1 = header_h + 10
            pip_x2 = pip_x1 + pip_w
            pip_y2 = pip_y1 + pip_h

            if pip_y2 < h - 35 and pip_x1 > 0:
                # White border & shadow
                cv2.rectangle(frame, (pip_x1 - 2, pip_y1 - 22), (pip_x2 + 2, pip_y2 + 2), (255, 255, 255), 2)
                cv2.rectangle(frame, (pip_x1 - 2, pip_y1 - 22), (pip_x2 + 2, pip_y1), (30, 30, 30), -1)
                cv2.putText(frame, "EVIDENCE ZOOM", (pip_x1 + 6, pip_y1 - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 240, 255), 1, cv2.LINE_AA)
                frame[pip_y1:pip_y2, pip_x1:pip_x2] = crop_resized

        # ── 4. Bottom Navigation Hint Bar ─────────────────────────────────
        bar_h = 28
        overlay2 = frame.copy()
        cv2.rectangle(overlay2, (0, h - bar_h), (w, h), (10, 10, 10), -1)
        cv2.addWeighted(overlay2, 0.85, frame, 0.15, 0, frame)

        nav_hint = "  [INCIDENT VIEWER] Press any key or Enter to return | Type 'jump' / 'j' in terminal to seek video"
        cv2.putText(frame, nav_hint, (6, h - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0, 240, 180), 1, cv2.LINE_AA)

        # Auto-save evidence image to disk
        if self.save_dir:
            safe_type = str(event_type).upper().replace(" ", "_")
            fname = f"incident_{incident.get('id', 1):03d}_{safe_type}_frame{frame_no:06d}.png"
            fpath = os.path.join(self.save_dir, fname)
            try:
                cv2.imwrite(fpath, frame)
                incident["file_path"] = fpath
            except Exception:
                pass

        return frame
