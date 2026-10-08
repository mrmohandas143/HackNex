"""
detector.py  --  YOLOv4-tiny inference via cv2.dnn  +  IoU Tracker
=================================================================
100% torch-free.  Requires only:  opencv-python  +  numpy
Model files (auto-downloaded by setup_model.py):
  models/yolov4-tiny.cfg
  models/yolov4-tiny.weights
  models/coco.names
"""

import os
import cv2
import numpy as np
from utils import COLOR_BGR_MAP, draw_label

# ── Paths ────────────────────────────────────────────────────────────────────
_MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
CFG_PATH     = os.path.join(_MODELS_DIR, "yolov4-tiny.cfg")
WEIGHTS_PATH = os.path.join(_MODELS_DIR, "yolov4-tiny.weights")
NAMES_PATH   = os.path.join(_MODELS_DIR, "coco.names")

INPUT_SIZE   = (416, 416)   # YOLOv4-tiny native input
CONF_THRESH  = 0.40
NMS_THRESH   = 0.45


# ── Simple IoU Tracker (pure numpy, no torch) ─────────────────────────────
class _IoUTracker:
    """
    Greedy IoU-based tracker.
    Assigns consistent IDs to detections across frames by matching the
    bounding box with the highest IoU to the previous frame's boxes.
    Tracks are dropped after MAX_AGE frames without a match.
    """
    MAX_AGE  = 10   # frames a track survives without a detection
    MIN_IOU  = 0.25

    def __init__(self):
        self._next_id  = 1
        self._tracks   = {}   # id -> {bbox, age}

    def update(self, detections: list[dict]) -> list[dict]:
        """Match detections to existing tracks; return detections with .id filled."""
        if not detections:
            # age out all tracks
            for tid in list(self._tracks):
                self._tracks[tid]["age"] += 1
                if self._tracks[tid]["age"] > self.MAX_AGE:
                    del self._tracks[tid]
            return detections

        det_boxes = np.array([d["bbox"] for d in detections], dtype=float)
        trk_ids   = list(self._tracks.keys())
        assigned  = [False] * len(detections)
        used_trks = set()

        if trk_ids:
            trk_boxes = np.array([self._tracks[t]["bbox"] for t in trk_ids], dtype=float)
            iou_mat   = self._iou_matrix(trk_boxes, det_boxes)

            # greedy assignment: best IoU pair first
            flat_idx  = np.argsort(-iou_mat.ravel())
            for fi in flat_idx:
                ti, di = divmod(fi, len(detections))
                if iou_mat[ti, di] < self.MIN_IOU:
                    break
                tid = trk_ids[ti]
                if tid in used_trks or assigned[di]:
                    continue
                used_trks.add(tid)
                assigned[di] = True
                detections[di]["id"] = tid
                self._tracks[tid]["bbox"] = detections[di]["bbox"]
                self._tracks[tid]["age"]  = 0

        # age-out unmatched tracks
        for tid in trk_ids:
            if tid not in used_trks:
                self._tracks[tid]["age"] += 1
                if self._tracks[tid]["age"] > self.MAX_AGE:
                    del self._tracks[tid]

        # spawn new tracks for unmatched detections
        for i, det in enumerate(detections):
            if not assigned[i]:
                det["id"] = self._next_id
                self._tracks[self._next_id] = {"bbox": det["bbox"], "age": 0}
                self._next_id += 1

        return detections

    @staticmethod
    def _iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Compute IoU between all pairs of boxes in a (N,4) and b (M,4)."""
        ax1, ay1, ax2, ay2 = a[:,0], a[:,1], a[:,2], a[:,3]
        bx1, by1, bx2, by2 = b[:,0], b[:,1], b[:,2], b[:,3]

        inter_x1 = np.maximum(ax1[:,None], bx1[None,:])
        inter_y1 = np.maximum(ay1[:,None], by1[None,:])
        inter_x2 = np.minimum(ax2[:,None], bx2[None,:])
        inter_y2 = np.minimum(ay2[:,None], by2[None,:])

        inter_w  = np.maximum(0, inter_x2 - inter_x1)
        inter_h  = np.maximum(0, inter_y2 - inter_y1)
        inter    = inter_w * inter_h

        area_a   = (ax2 - ax1) * (ay2 - ay1)
        area_b   = (bx2 - bx1) * (by2 - by1)
        union    = area_a[:,None] + area_b[None,:] - inter
        return np.where(union > 0, inter / union, 0.0)


# ── Main Detector ─────────────────────────────────────────────────────────
class YOLODetector:
    """
    YOLOv4-tiny loaded via cv2.dnn (no torch, no onnxruntime).
    Tracking is handled by _IoUTracker (pure numpy).
    """

    def __init__(self):
        for path in (CFG_PATH, WEIGHTS_PATH, NAMES_PATH):
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"[Detector] Missing model file: {path}\n"
                    "  Run:  python setup_model.py"
                )

        print("[Detector] Loading YOLOv4-tiny via cv2.dnn ...")
        self.net = cv2.dnn.readNetFromDarknet(CFG_PATH, WEIGHTS_PATH)
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

        with open(NAMES_PATH) as f:
            self.class_names = [l.strip() for l in f if l.strip()]

        # names of the output layers
        layer_names   = self.net.getLayerNames()
        unconnected   = self.net.getUnconnectedOutLayers()
        # OpenCV >= 4.x returns flat array; older returns [[idx]]
        if unconnected.ndim == 2:
            unconnected = unconnected.flatten()
        self._out_layers = [layer_names[i - 1] for i in unconnected]

        self._tracker    = _IoUTracker()
        print(f"[Detector] Ready  ({len(self.class_names)} classes, "
              f"output layers: {self._out_layers})")

    # ── Inference ──────────────────────────────────────────────────────────

    def detect(self, frame: np.ndarray) -> list[dict]:
        """Run YOLOv4-tiny + IoU tracking on one BGR frame."""
        h, w = frame.shape[:2]

        blob = cv2.dnn.blobFromImage(
            frame, 1/255.0, INPUT_SIZE,
            swapRB=True, crop=False
        )
        self.net.setInput(blob)
        outputs = self.net.forward(self._out_layers)

        boxes, confs, class_ids = [], [], []
        for out in outputs:
            for det in out:
                scores  = det[5:]
                cls_id  = int(np.argmax(scores))
                conf    = float(scores[cls_id])
                if conf < CONF_THRESH:
                    continue
                cx, cy, bw, bh = det[:4]
                x1 = int((cx - bw/2) * w)
                y1 = int((cy - bh/2) * h)
                x2 = int((cx + bw/2) * w)
                y2 = int((cy + bh/2) * h)
                boxes.append([x1, y1, x2-x1, y2-y1])
                confs.append(conf)
                class_ids.append(cls_id)

        # NMS
        indices = cv2.dnn.NMSBoxes(boxes, confs, CONF_THRESH, NMS_THRESH)
        if len(indices) == 0:
            return self._tracker.update([])

        detections = []
        for i in indices.flatten():
            x, y, bw, bh = boxes[i]
            x1, y1 = max(0, x), max(0, y)
            x2, y2 = min(w, x+bw), min(h, y+bh)
            detections.append({
                "id":         0,   # filled by tracker
                "class":      self.class_names[class_ids[i]],
                "confidence": confs[i],
                "bbox":       (x1, y1, x2, y2),
            })

        return self._tracker.update(detections)

    # ── Drawing ────────────────────────────────────────────────────────────

    def draw(self, frame: np.ndarray,
             detections: list[dict],
             scene: dict | None = None) -> np.ndarray:
        """Annotate frame with bounding boxes and labels."""
        out = frame.copy()

        # build color lookup from scene
        color_map: dict[int, tuple] = {}
        if scene:
            for obj in scene.get("objects", []):
                color_map[obj["id"]] = obj.get("color_bgr", (0, 210, 80))

        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            tid      = det["id"]
            cls_name = det["class"]
            conf     = det["confidence"]
            box_col  = color_map.get(tid, (0, 210, 80))

            # Main box
            cv2.rectangle(out, (x1, y1), (x2, y2), box_col, 2)

            # Corner accents
            sz = 8
            for px, py in [(x1,y1),(x2-sz,y1),(x1,y2-sz),(x2-sz,y2-sz)]:
                cv2.rectangle(out, (px, py), (px+sz, py+sz), box_col, -1)

            # Label
            color_name = ""
            if scene:
                for obj in scene.get("objects", []):
                    if obj["id"] == tid:
                        color_name = obj.get("color", "")
                        break
            label  = f"#{tid} {cls_name}"
            if color_name and color_name not in ("unknown", "colorful"):
                label += f" [{color_name}]"
            label += f"  {conf:.0%}"
            draw_label(out, label, (x1, y1 - 5),
                       fg=(255, 255, 255), bg=box_col)

        return out
