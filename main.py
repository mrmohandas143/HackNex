"""
main.py  —  Vision AI  (Tier 1)
========================================
Usage:
    python main.py              # auto-selects external webcam if connected, otherwise webcam 0
    python main.py 0            # built-in webcam (index 0)
    python main.py 1            # external webcam (index 1)
    python main.py video.mp4    # pre-recorded video file (loops automatically)

Controls (OpenCV window):
    [Q]   Pause + enter query mode in terminal
    [C]   Switch camera (cycles built-in <-> external webcam)
    [S]   Save annotated frame as PNG
    [ESC] Exit
"""

import os
import sys

# Silence noisy low-level OpenCV backend enumeration warnings on Windows
os.environ["OPENCV_LOG_LEVEL"] = "ERROR"

import cv2

from detector import YOLODetector
from scene_builder import SceneBuilder
from qa_engine import QAEngine


# ─────────────────────────────────────────────────────────────────────────────
BANNER = r"""
  ╔══════════════════════════════════════════════════════════════╗
  ║        Vision AI  —  Tier 1  (YOLO + Rule-based Q&A)        ║
  ║   Object Detection  |  Tracking  |  Color Analysis  |  QA   ║
  ╚══════════════════════════════════════════════════════════════╝
"""

HELP = """
  Controls (OpenCV window)
  ─────────────────────────────────────
  [Q]    Pause video  ->  enter query in terminal
  [C]    Switch camera (toggle built-in / external webcam)
  [S]    Save annotated frame as PNG
  [ESC]  Exit

  Example queries
  ─────────────────────────────────────
  "What is in the frame?" / "What is in the image?"
  "Which color is the tshirt?" / "What color is the shirt?"
  "How many people are there?"
  "Where is the person?"
  "Is there a car?"
  "Describe the scene"
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


def query_loop(qa: QAEngine, scene: dict, raw_frame) -> None:
    """Blocking terminal Q&A session. Returns when user is done."""
    print("\n" + "=" * 58)
    print("  [PHOTO] VIDEO PAUSED  --  Frame captured for analysis")
    print("-" * 58)

    if scene.get("objects"):
        print(f"  Scene: {scene['description']}")
    else:
        print("  Scene: No objects detected in the paused frame.")

    print("-" * 58)
    print("  Type a question -- or press Enter (blank) to resume.\n")

    while True:
        try:
            raw = input("  (?) You : ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not raw or raw.lower() in {"resume", "r", "exit", "quit", "q"}:
            break

        answer = qa.answer(raw, scene, raw_frame)
        print(f"\n  (AI) AI  : {answer}\n")

    print("=" * 58)
    print("  [PLAY] Video resumed.\n")


def main() -> None:
    print(BANNER)

    source, cam_indices, current_cam = resolve_source(sys.argv[1:])

    # ── Component init ────────────────────────────────────────────────────
    detector      = YOLODetector()
    scene_builder = SceneBuilder()
    qa_engine     = QAEngine()

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

    def get_title(src):
        if isinstance(src, int):
            kind = "Built-in" if src == 0 else f"External Cam #{src}"
            return f"Vision AI [{kind}]  [Q=Query | C=Switch Cam | S=Save | ESC=Exit]"
        return f"Vision AI [{os.path.basename(src)}]  [Q=Query | S=Save | ESC=Exit]"

    win_title = get_title(source)
    cv2.namedWindow("Vision AI", cv2.WINDOW_NORMAL)
    cv2.setWindowTitle("Vision AI", win_title)

    latest_scene     = {"objects": [], "object_counts": {}, "total_count": 0,
                        "description": "No detections yet."}
    latest_raw_frame = None
    annotated_frame  = None
    paused           = False
    frame_no         = 0

    print("[INFO] Stream started -- press [Q] for query, [C] to switch cameras.\n")

    while True:
        # ── Capture ───────────────────────────────────────────────────────
        if not paused:
            ret, frame = cap.read()
            if not ret or frame is None:
                if isinstance(source, str):  # video file ended -> loop
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                print("[WARN] Camera grab returned empty frame. Retrying...")
                key = cv2.waitKey(30) & 0xFF
                if key == 27:
                    break
                continue

            frame_no += 1

            # ── YOLO detect + track ───────────────────────────────────────
            detections = detector.detect(frame)

            # ── Build scene context ───────────────────────────────────────
            scene = scene_builder.build(frame, detections)
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

        # ── Q -> query mode ───────────────────────────────────────────────
        elif key in (ord('q'), ord('Q')):
            paused = True
            cv2.setWindowTitle("Vision AI", "Vision AI  [PAUSED -- check terminal for Q&A]")
            query_loop(qa_engine, latest_scene, latest_raw_frame)
            paused = False
            cv2.setWindowTitle("Vision AI", get_title(source))

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
