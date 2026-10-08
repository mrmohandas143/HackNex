"""
main.py  —  Vision AI  (Tier 2 + Incident Evidence Viewer & Video Seeking)
========================================================================
Usage:
    python main.py              # auto-selects external webcam if connected, otherwise webcam 0
    python main.py 0            # built-in webcam (index 0)
    python main.py 1            # external webcam (index 1)
    python main.py video.mp4    # pre-recorded video file (loops automatically)

Controls (OpenCV window):
    [Q]   Pause + enter query mode in terminal (views incident evidence & jump)
    [J]   Jump video playback to latest incident point
    [C]   Switch camera (cycles built-in <-> external webcam)
    [S]   Save annotated frame as PNG
    [ESC] Exit
"""

import os
import sys
import time

# Silence noisy low-level OpenCV backend enumeration warnings on Windows
os.environ["OPENCV_LOG_LEVEL"] = "ERROR"

import cv2
import numpy as np

from detector import YOLODetector
from scene_builder import SceneBuilder
from qa_engine import QAEngine
from temporal_tracker import TemporalTracker
from incident_manager import IncidentManager


# ─────────────────────────────────────────────────────────────────────────────
BANNER = r"""
  ╔══════════════════════════════════════════════════════════════╗
  ║     Vision AI  —  Tier 2  (Deep Attributes & Reasoning)     ║
  ║  YOLO + Temporal Tracking + Anatomical Analysis + Memory Q&A ║
  ║       ★ INCIDENT EVIDENCE VIEWER & VIDEO SEEKING ★          ║
  ╚══════════════════════════════════════════════════════════════╝
"""

HELP = """
  Controls (OpenCV window)
  ─────────────────────────────────────
  [Q]    Pause video  ->  enter query in terminal (auto-shows incident evidence!)
  [J]    Jump video to latest incident point
  [C]    Switch camera (toggle built-in / external webcam)
  [S]    Save annotated frame as PNG
  [ESC]  Exit

  Incident & Visual Reasoning Queries:
  ─────────────────────────────────────
  Incident Review:"Show when the person entered" / "When did they enter?"
                  "Show where the incident occurred" / "Show evidence"
  Weapons & Items:"Does the person have any weapon?" (shows close-up)
                  "Are they holding anything?"
  Accessories:    "Does the person wear any earrings?" (shows ear zoom)
                  "Is the person wearing glasses?" / "Any hat?"
  Outfit & Color: "What is the person wearing?"
                  "Which color is the tshirt?" / "What color are the pants?"
  Posture & Motion:"Is the person sitting or standing?"
                  "Which direction are they moving?"
  Temporal Memory:"How long has the person been here?"
                  "Did anyone enter or leave recently?"
  Conversational: "And what about the pants?" (follow-up)
"""
# ─────────────────────────────────────────────────────────────────────────────


def find_available_cameras(max_check: int = 4) -> list[tuple[int, str]]:
    """Scan and return all functional camera devices as [(index, resolution)]."""
    available = []
    for idx in range(max_check):
        cap = None
        if sys.platform.startswith("win"):
            cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        if cap is None or not cap.isOpened():
            cap = cv2.VideoCapture(idx)

        if cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None:
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                available.append((idx, f"{w}x{h}"))
            cap.release()
    return available


def open_capture(source):
    """
    Open a video source.
    On Windows, uses cv2.CAP_DSHOW for webcams to ensure external USB cameras
    initialize reliably and capture at full hardware resolution.
    """
    if isinstance(source, int):
        if sys.platform.startswith("win"):
            cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)
            if cap.isOpened():
                return cap
        return cv2.VideoCapture(source)
    return cv2.VideoCapture(source)


def resolve_source(args: list[str]) -> tuple[int | str, list[int], int]:
    """
    Determine initial video source and list of available camera indices.
    Returns: (initial_source, available_cam_indices, current_cam_idx)
    """
    # 1. Check if user passed a video file
    if args and not args[0].isdigit() and os.path.isfile(args[0]):
        return args[0], [], -1

    # 2. Probe system cameras
    print("[CAMERA] Scanning available cameras...")
    cams = find_available_cameras(max_check=4)
    cam_indices = [idx for idx, _ in cams]

    if not cam_indices:
        print("[WARN] No cameras detected during scan. Defaulting to index 0.")
        cam_indices = [0]
        cams = [(0, "unknown")]

    print(f"[CAMERA] Found {len(cams)} camera(s):")
    for idx, res in cams:
        desc = "Built-in Webcam" if idx == 0 else f"External Camera #{idx}"
        print(f"  - [{idx}] {desc} ({res})")

    # 3. User passed explicit index: e.g. python main.py 1
    if args and args[0].isdigit():
        chosen = int(args[0])
        print(f"[CAMERA] Using requested Camera #{chosen}")
        return chosen, cam_indices, chosen

    # 4. No argument passed: if external camera (index >= 1) exists, prefer it!
    if len(cam_indices) > 1 and 1 in cam_indices:
        print("[CAMERA] External camera detected at index 1 -> Auto-selected external webcam.")
        print("[CAMERA] (Tip: press [C] anytime in the window to switch to built-in webcam)")
        return 1, cam_indices, 1

    # Otherwise default to camera 0
    return 0, cam_indices, 0


def query_loop(qa: QAEngine,
               incident_manager: IncidentManager,
               scene: dict,
               raw_frame: np.ndarray,
               frame_no: int,
               video_time_sec: float,
               is_video_file: bool) -> int | None:
    """
    Blocking terminal Q&A session.
    Displays visual incident evidence on screen and returns a target frame to seek video to (if requested).
    """
    print("\n" + "=" * 60)
    print("  [PHOTO] VIDEO PAUSED  --  Frame captured for visual reasoning")
    print("-" * 60)

    if scene.get("objects"):
        print(f"  Scene: {scene['description']}")
    else:
        print("  Scene: No objects detected in the paused frame.")

    print("-" * 60)
    print("  Type a question -- or press Enter (blank) to resume playback.")
    if is_video_file:
        print("  (Tip: ask 'when did person enter?' or type 'jump' to seek video)\n")
    else:
        print()

    jump_target_frame = None
    active_evidence_win = False

    while True:
        try:
            raw = input("  (?) You : ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not raw or raw.lower() in {"resume", "r", "exit", "quit", "q"}:
            break

        # Check if user explicitly asked to seek/jump to current incident
        if raw.lower() in {"jump", "j", "seek"} and incident_manager.incidents:
            target_inc = incident_manager.incidents[-1]
            jump_target_frame = target_inc["frame_no"]
            print(f"\n  [VIDEO SEEK] Seeking video to Frame #{jump_target_frame} ({target_inc['time_str']})!\n")
            break

        # Answer query and find matching incident
        answer, incident = qa.answer_with_incident(
            raw, scene, raw_frame, frame_no=frame_no, video_time_sec=video_time_sec
        )
        print(f"\n  (AI) AI  : {answer}\n")

        # Visual incident display
        if incident is not None:
            try:
                evidence_card = incident_manager.render_evidence_view(incident)
                cv2.namedWindow("Incident Evidence", cv2.WINDOW_NORMAL)
                cv2.imshow("Incident Evidence", evidence_card)
                cv2.waitKey(1)
                active_evidence_win = True

                print("-" * 60)
                print(f"  📸  INCIDENT EVIDENCE LOCATED:")
                print(f"      Event : {incident['event_type'].upper()} ({incident['class']} #{incident['track_id']})")
                print(f"      Point : {incident['time_str']}  (Frame #{incident['frame_no']})")
                print(f"      Image : Opened in 'Incident Evidence' window!")
                if is_video_file:
                    print(f"  👉  Type 'jump' or 'j' to seek video to this incident.")
                print("-" * 60)

                # If query explicitly asked to jump or seek, auto-set jump target
                if any(w in raw.lower() for w in ["jump", "seek", "where it occur", "where did it occur"]):
                    jump_target_frame = incident["frame_no"]
                    print(f"  [SEEK] Video will seek to Frame #{jump_target_frame} on resume.\n")
            except Exception as e:
                print(f"[WARN] Could not display incident window: {e}")

    # Clean up evidence popup when resuming main stream
    if active_evidence_win:
        cv2.destroyWindow("Incident Evidence")

    print("=" * 60)
    print("  [PLAY] Resuming video playback.\n")
    return jump_target_frame


def main() -> None:
    print(BANNER)

    source, cam_indices, current_cam = resolve_source(sys.argv[1:])
    is_video_file = isinstance(source, str)

    # ── Component init ────────────────────────────────────────────────────
    detector         = YOLODetector()
    incident_manager = IncidentManager()
    temporal_tracker = TemporalTracker(incident_manager=incident_manager)
    scene_builder    = SceneBuilder(incident_manager=incident_manager)
    qa_engine        = QAEngine(temporal_tracker=temporal_tracker, incident_manager=incident_manager)

    print(HELP)

    # ── Open video ────────────────────────────────────────────────────────
    cap = open_capture(source)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open source: {source}")
        if isinstance(source, int) and len(cam_indices) > 1:
            alt = [c for c in cam_indices if c != source][0]
            print(f"[INFO] Trying fallback to camera #{alt}...")
            source = alt
            current_cam = alt
            cap = open_capture(source)

    if not cap.isOpened():
        print("[ERROR] Failed to open any video capture device. Exiting.")
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = fps if (fps and fps > 0) else 30.0

    def get_title(src):
        if isinstance(src, int):
            kind = "Built-in" if src == 0 else f"External Cam #{src}"
            return f"Vision AI [{kind}]  [Q=Query | J=Jump Incident | C=Switch | S=Save | ESC=Exit]"
        return f"Vision AI [{os.path.basename(src)}]  [Q=Query | J=Jump Incident | S=Save | ESC=Exit]"

    win_title = get_title(source)
    cv2.namedWindow("Vision AI", cv2.WINDOW_NORMAL)
    cv2.setWindowTitle("Vision AI", win_title)

    latest_scene     = {"objects": [], "object_counts": {}, "total_count": 0,
                        "description": "No detections yet."}
    latest_raw_frame = None
    annotated_frame  = None
    paused           = False
    frame_no         = 0
    start_time       = time.time()

    print("[INFO] Stream started -- press [Q] for query & incident view, [J] to jump video.\n")

    while True:
        # ── Capture ───────────────────────────────────────────────────────
        if not paused:
            ret, frame = cap.read()
            if not ret or frame is None:
                if is_video_file:  # video file ended -> loop
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    frame_no = 0
                    continue
                print("[WARN] Camera grab returned empty frame. Retrying...")
                key = cv2.waitKey(30) & 0xFF
                if key == 27:
                    break
                continue

            frame_no += 1
            video_time_sec = (frame_no / fps) if is_video_file else (time.time() - start_time)

            # ── YOLO detect + track ───────────────────────────────────────
            detections = detector.detect(frame)
            detections = temporal_tracker.update(
                detections, frame=frame, frame_no=frame_no, video_time_sec=video_time_sec
            )

            # ── Build scene context ───────────────────────────────────────
            scene = scene_builder.build(
                frame, detections, frame_no=frame_no, video_time_sec=video_time_sec
            )
            latest_scene     = scene
            latest_raw_frame = frame.copy()

            # ── Annotate ──────────────────────────────────────────────────
            annotated_frame = detector.draw(frame, detections, scene)
            annotated_frame = scene_builder.draw_summary(annotated_frame, scene)

        # ── Display ───────────────────────────────────────────────────────
        if annotated_frame is not None:
            cv2.imshow("Vision AI", annotated_frame)

        key = cv2.waitKey(1) & 0xFF

        # ── ESC -> exit ───────────────────────────────────────────────────
        if key == 27:
            print("\n[INFO] Exiting Vision AI.")
            break

        # ── Q -> query mode (with Incident Evidence display & Seek) ───────
        elif key in (ord('q'), ord('Q')):
            paused = True
            cv2.setWindowTitle("Vision AI", "Vision AI  [PAUSED -- check terminal for Q&A]")
            target_seek_frame = query_loop(
                qa_engine, incident_manager, latest_scene, latest_raw_frame,
                frame_no=frame_no, video_time_sec=video_time_sec, is_video_file=is_video_file
            )
            if target_seek_frame is not None and is_video_file:
                cap.set(cv2.CAP_PROP_POS_FRAMES, target_seek_frame)
                frame_no = target_seek_frame
                print(f"[VIDEO] Jumped playback to Frame #{target_seek_frame}!\n")

            paused = False
            cv2.setWindowTitle("Vision AI", get_title(source))

        # ── J -> jump to latest incident ──────────────────────────────────
        elif key in (ord('j'), ord('J')) and incident_manager.incidents:
            latest_inc = incident_manager.incidents[-1]
            print(f"\n[INCIDENT] Displaying latest incident: {latest_inc['details']} ({latest_inc['time_str']})")
            card = incident_manager.render_evidence_view(latest_inc)
            cv2.namedWindow("Incident Evidence", cv2.WINDOW_NORMAL)
            cv2.imshow("Incident Evidence", card)
            if is_video_file:
                cap.set(cv2.CAP_PROP_POS_FRAMES, latest_inc["frame_no"])
                frame_no = latest_inc["frame_no"]
                print(f"[VIDEO] Playback jumped to incident Frame #{latest_inc['frame_no']}!")

        # ── C -> switch camera ────────────────────────────────────────────
        elif key in (ord('c'), ord('C')) and cam_indices:
            if len(cam_indices) > 1:
                # Cycle to next camera
                curr_idx_in_list = cam_indices.index(current_cam) if current_cam in cam_indices else 0
                next_cam = cam_indices[(curr_idx_in_list + 1) % len(cam_indices)]
                print(f"\n[CAMERA] Switching from Camera #{current_cam} to Camera #{next_cam}...")

                cap.release()
                new_cap = open_capture(next_cam)
                if new_cap.isOpened():
                    cap = new_cap
                    source = next_cam
                    current_cam = next_cam
                    win_title = get_title(source)
                    cv2.setWindowTitle("Vision AI", win_title)
                    print(f"[CAMERA] Successfully connected to Camera #{current_cam}!\n")
                else:
                    print(f"[ERROR] Could not switch to Camera #{next_cam}. Reverting to #{current_cam}...")
                    cap = open_capture(current_cam)
            else:
                print("\n[CAMERA] Only 1 camera detected. Re-checking system devices...")
                fresh_cams = find_available_cameras()
                cam_indices = [idx for idx, _ in fresh_cams]
                if len(cam_indices) > 1:
                    print(f"[CAMERA] Found new camera devices: {cam_indices}! Press [C] again to switch.")
                else:
                    print("[CAMERA] No additional camera found. Please ensure USB webcam is plugged in.")

        # ── S -> save frame ───────────────────────────────────────────────
        elif key in (ord('s'), ord('S')):
            if annotated_frame is not None:
                fname = f"capture_{frame_no:06d}.png"
                cv2.imwrite(fname, annotated_frame)
                print(f"[INFO] Saved screenshot -> {fname}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
