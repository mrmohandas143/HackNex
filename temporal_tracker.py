"""
temporal_tracker.py — Tier 2 Temporal Dynamics & Event Tracking
===============================================================
Maintains temporal buffers across video frames:
  - Motion vectors: velocity (vx, vy), direction (left/right/approaching/stationary)
  - Dwell time: how many seconds each object has been present
  - Event log: arrivals, departures, movement transitions
  - Movement history: breadcrumb trajectory
100% open-source, local, pure Python & NumPy.
"""

import time
import math
from collections import deque
import numpy as np


class TemporalTracker:
    """Tracks object motion history, dwell times, and entry/exit events over time."""

    def __init__(self, history_seconds: float = 60.0, incident_manager=None):
        self.history_seconds = history_seconds
        self.incident_manager = incident_manager
        # track_id -> dict
        self._tracks: dict[int, dict] = {}
        # Completed event logs
        self.events: deque[dict] = deque(maxlen=100)

    def update(self, detections: list[dict], timestamp: float | None = None,
               frame: np.ndarray | None = None, frame_no: int = 0,
               video_time_sec: float = 0.0) -> list[dict]:
        """
        Update temporal history with current detections and log visual incidents.
        Enriches each detection with:
          - motion_direction: str ("stationary", "moving left", etc.)
          - velocity_px_s: float
          - dwell_time_sec: float
          - is_moving: bool
        """
        now = timestamp if timestamp is not None else time.time()
        active_ids = set()

        for det in detections:
            tid = det.get("id", 0)
            if tid <= 0:
                continue

            active_ids.add(tid)
            bbox = det["bbox"]
            cx = (bbox[0] + bbox[2]) / 2.0
            cy = (bbox[1] + bbox[3]) / 2.0
            area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])

            if tid not in self._tracks:
                # ── New object entry event ────────────────────────────────
                self._tracks[tid] = {
                    "class":           det["class"],
                    "first_seen":      now,
                    "last_seen":       now,
                    "positions":       deque([(now, cx, cy, area)], maxlen=90),
                    "velocity":        (0.0, 0.0),
                    "speed":           0.0,
                    "direction":       "stationary",
                    "is_moving":       False,
                    "total_distance":  0.0,
                }
                msg = f"{det['class'].capitalize()} (ID #{tid}) entered the frame."
                self.events.append({
                    "timestamp": now,
                    "type":      "entry",
                    "id":        tid,
                    "class":     det["class"],
                    "message":   msg,
                })
                if self.incident_manager and frame is not None:
                    self.incident_manager.log_incident(
                        event_type="entry",
                        track_id=tid,
                        class_name=det["class"],
                        frame=frame,
                        frame_no=frame_no,
                        video_time_sec=video_time_sec,
                        bbox=det["bbox"],
                        details=msg
                    )
            else:
                trk = self._tracks[tid]
                trk["last_seen"] = now
                trk["positions"].append((now, cx, cy, area))

                # Calculate velocity & direction over recent window (last ~1.0 sec)
                self._compute_motion(trk, now)

            # Enrich detection dict
            trk = self._tracks[tid]
            dwell = now - trk["first_seen"]
            det["dwell_time_sec"]   = round(dwell, 1)
            det["motion_direction"] = trk["direction"]
            det["velocity_px_s"]    = round(trk["speed"], 1)
            det["is_moving"]        = trk["is_moving"]

        # ── Check for departures (not seen for > 2.5 seconds) ────────────
        dead_ids = []
        for tid, trk in self._tracks.items():
            if tid not in active_ids:
                if now - trk["last_seen"] > 2.5:
                    dead_ids.append(tid)
                    dwell = trk["last_seen"] - trk["first_seen"]
                    msg = f"{trk['class'].capitalize()} (ID #{tid}) left the frame after {dwell:.1f}s."
                    self.events.append({
                        "timestamp": now,
                        "type":      "exit",
                        "id":        tid,
                        "class":     trk["class"],
                        "message":   msg,
                    })
                    if self.incident_manager and frame is not None:
                        self.incident_manager.log_incident(
                            event_type="exit",
                            track_id=tid,
                            class_name=trk["class"],
                            frame=frame,
                            frame_no=frame_no,
                            video_time_sec=video_time_sec,
                            bbox=(0, 0, 0, 0),
                            details=msg
                        )

        for tid in dead_ids:
            del self._tracks[tid]

        return detections

    def get_track_history(self, tid: int) -> dict | None:
        """Get temporal metrics for a specific track ID."""
        return self._tracks.get(tid)

    def get_recent_events(self, seconds: float = 30.0) -> list[dict]:
        """Return events that occurred within the last N seconds."""
        now = time.time()
        return [e for e in self.events if now - e["timestamp"] <= seconds]

    def describe_activity(self) -> str:
        """Produce a natural language summary of movements and temporal activity."""
        now = time.time()
        if not self._tracks:
            recent_exits = [e for e in self.events if now - e["timestamp"] <= 15.0]
            if recent_exits:
                return f"No objects currently in frame. Recently: {recent_exits[-1]['message']}"
            return "No motion or active objects detected."

        parts = []
        for tid, trk in self._tracks.items():
            dwell = round(now - trk["first_seen"], 1)
            status = "standing still" if not trk["is_moving"] else trk["direction"]
            parts.append(
                f"{trk['class']} (ID #{tid}): present for {dwell}s, {status}"
            )
        return " | ".join(parts)

    # ── Private Calculations ─────────────────────────────────────────────────

    def _compute_motion(self, trk: dict, now: float) -> None:
        """Estimate velocity and direction from position history."""
        positions = trk["positions"]
        if len(positions) < 4:
            trk["direction"] = "stationary"
            trk["is_moving"] = False
            return

        # Compare current pos with pos from ~0.6s to 1.2s ago
        curr_t, curr_x, curr_y, curr_a = positions[-1]
        ref_t, ref_x, ref_y, ref_a = positions[0]

        dt = curr_t - ref_t
        if dt < 0.15:
            return

        dx = curr_x - ref_x
        dy = curr_y - ref_y
        dist = math.hypot(dx, dy)
        speed = dist / dt

        trk["speed"] = speed
        trk["velocity"] = (dx / dt, dy / dt)

        # Threshold for meaningful movement (pixels per second)
        if speed < 22.0:
            trk["is_moving"] = False
            trk["direction"] = "stationary / still"
        else:
            trk["is_moving"] = True
            # Check scale change (approaching / moving away)
            da_ratio = curr_a / max(1.0, ref_a)
            if abs(dx) > abs(dy) * 1.3:
                trk["direction"] = "moving right" if dx > 0 else "moving left"
            elif abs(dy) > abs(dx) * 1.3:
                trk["direction"] = "moving down" if dy > 0 else "moving up"
            elif da_ratio > 1.25:
                trk["direction"] = "moving towards camera"
            elif da_ratio < 0.80:
                trk["direction"] = "moving away from camera"
            else:
                trk["direction"] = "moving right" if dx > 0 else "moving left"
